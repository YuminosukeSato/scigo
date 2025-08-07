# Core API Reference

Complete API documentation for the `core` package.

## Package Overview

```go
import "github.com/YuminosukeSato/scigo/core"
```

The `core` package provides fundamental interfaces, types, and utilities used throughout SciGo.

## Model Interface

### Base Interfaces

```go
package model

// Model is the base interface for all models
type Model interface {
    Fit(X, y mat.Matrix) error
    Predict(X mat.Matrix) (mat.Matrix, error)
    IsFitted() bool
    Reset() error
    GetParams() map[string]interface{}
    SetParams(params map[string]interface{}) error
}

// SupervisedModel adds scoring capability
type SupervisedModel interface {
    Model
    Score(X, y mat.Matrix) (float64, error)
}

// UnsupervisedModel for clustering and dimensionality reduction
type UnsupervisedModel interface {
    Model
    FitPredict(X mat.Matrix) (mat.Matrix, error)
    Transform(X mat.Matrix) (mat.Matrix, error)
}
```

### Transformer Interface

```go
// Transformer modifies data
type Transformer interface {
    Fit(X mat.Matrix) error
    Transform(X mat.Matrix) (mat.Matrix, error)
    FitTransform(X mat.Matrix) (mat.Matrix, error)
    IsFitted() bool
}

// TransformerMixin provides default FitTransform
type TransformerMixin struct{}

func (t *TransformerMixin) FitTransform(transformer Transformer, X mat.Matrix) (mat.Matrix, error) {
    if err := transformer.Fit(X); err != nil {
        return nil, err
    }
    return transformer.Transform(X)
}
```

### Predictor Interface

```go
// Predictor makes predictions
type Predictor interface {
    Predict(X mat.Matrix) (mat.Matrix, error)
}

// Classifier adds probability predictions
type Classifier interface {
    Predictor
    PredictProba(X mat.Matrix) (mat.Matrix, error)
    PredictLogProba(X mat.Matrix) (mat.Matrix, error)
    Classes() []interface{}
}

// Regressor for continuous targets
type Regressor interface {
    Predictor
    Score(X, y mat.Matrix) (float64, error)
}
```

## State Management

### StateManager

```go
package model

import "sync"

// StateManager handles model state
type StateManager struct {
    fitted    bool
    nFeatures int
    nSamples  int
    mu        sync.RWMutex
    metadata  map[string]interface{}
}

// NewStateManager creates a new state manager
func NewStateManager() *StateManager {
    return &StateManager{
        metadata: make(map[string]interface{}),
    }
}
```

#### Methods

##### SetFitted

```go
func (s *StateManager) SetFitted()
```

Marks model as fitted (thread-safe).

##### IsFitted

```go
func (s *StateManager) IsFitted() bool
```

Checks if model is fitted (thread-safe).

##### SetDimensions

```go
func (s *StateManager) SetDimensions(samples, features int)
```

Stores training data dimensions.

##### ValidateX

```go
func (s *StateManager) ValidateX(X mat.Matrix) error
```

Validates input matrix dimensions.

**Returns:**
- `error`: `NotFittedError` if not fitted, `DimensionError` if wrong shape

##### Reset

```go
func (s *StateManager) Reset()
```

Resets state to unfitted.

##### SetMetadata

```go
func (s *StateManager) SetMetadata(key string, value interface{})
```

Stores metadata (thread-safe).

##### GetMetadata

```go
func (s *StateManager) GetMetadata(key string) (interface{}, bool)
```

Retrieves metadata (thread-safe).

## Error Types

### Error Hierarchy

```go
package errors

// BaseError is the root error type
type BaseError struct {
    Op      string // Operation that failed
    Kind    string // Error category
    Message string // Human-readable message
    Err     error  // Wrapped error
}

func (e *BaseError) Error() string
func (e *BaseError) Unwrap() error
```

### Specific Error Types

#### NotFittedError

```go
// NotFittedError indicates model not trained
type NotFittedError struct {
    BaseError
    ModelName string
}

func NewNotFittedError(modelName string) *NotFittedError
```

**Example:**
```go
if !model.IsFitted() {
    return errors.NewNotFittedError("LinearRegression")
}
```

#### DimensionError

```go
// DimensionError indicates shape mismatch
type DimensionError struct {
    BaseError
    Expected []int
    Actual   []int
}

func NewDimensionError(op string, expected, actual []int) *DimensionError
```

**Example:**
```go
if cols != expectedCols {
    return errors.NewDimensionError("Predict",
        []int{rows, expectedCols},
        []int{rows, cols})
}
```

#### ValueError

```go
// ValueError indicates invalid parameter
type ValueError struct {
    BaseError
    Parameter string
    Value     interface{}
    Valid     string // Description of valid values
}

func NewValueError(param string, value interface{}, valid string) *ValueError
```

**Example:**
```go
if alpha < 0 {
    return errors.NewValueError("alpha", alpha, "non-negative")
}
```

#### NumericalError

```go
// NumericalError indicates computation issues
type NumericalError struct {
    BaseError
    Operation string
    Values    []float64
}

func NewNumericalError(op string, values ...float64) *NumericalError
```

#### ConvergenceWarning

```go
// ConvergenceWarning indicates optimization didn't converge
type ConvergenceWarning struct {
    BaseError
    Iterations int
    Tolerance  float64
    Loss       float64
}

func NewConvergenceWarning(iter int, tol, loss float64) *ConvergenceWarning
```

## Parallel Processing

### Parallel Package

```go
package parallel

import "runtime"

// Options configures parallel execution
type Options struct {
    MinSize    int  // Minimum size for parallelization
    MaxWorkers int  // Maximum number of workers
    ChunkSize  int  // Size of work chunks
}

// DefaultOptions provides sensible defaults
var DefaultOptions = Options{
    MinSize:    1000,
    MaxWorkers: runtime.NumCPU(),
    ChunkSize:  0,  // Auto-calculate
}
```

### Functions

#### ParallelFor

```go
func ParallelFor(n int, fn func(start, end int))
```

Executes function in parallel over range [0, n).

**Parameters:**
- `n`: Total iterations
- `fn`: Function to execute on each chunk

**Example:**
```go
parallel.ParallelFor(len(data), func(start, end int) {
    for i := start; i < end; i++ {
        processItem(data[i])
    }
})
```

#### ParallelMap

```go
func ParallelMap(data []float64, fn func(float64) float64) []float64
```

Applies function to each element in parallel.

**Parameters:**
- `data`: Input slice
- `fn`: Function to apply

**Returns:**
- `[]float64`: Transformed data

**Example:**
```go
result := parallel.ParallelMap(data, math.Sqrt)
```

#### ParallelReduce

```go
func ParallelReduce(data []float64, fn func(float64, float64) float64) float64
```

Performs parallel reduction.

**Parameters:**
- `data`: Input slice
- `fn`: Reduction function

**Returns:**
- `float64`: Reduced value

**Example:**
```go
sum := parallel.ParallelReduce(data, func(a, b float64) float64 {
    return a + b
})
```

#### SetNumThreads

```go
func SetNumThreads(n int)
```

Sets maximum number of parallel workers.

**Parameters:**
- `n`: Number of threads (0 for auto)

## Tensor Operations

### Tensor Package

```go
package tensor

import "gonum.org/v1/gonum/mat"

// Tensor represents multi-dimensional array
type Tensor struct {
    data  []float64
    shape []int
    strides []int
}

// NewTensor creates a new tensor
func NewTensor(shape ...int) *Tensor
```

### Methods

#### At

```go
func (t *Tensor) At(indices ...int) float64
```

Gets value at specified indices.

**Parameters:**
- `indices`: Position in each dimension

**Returns:**
- `float64`: Value at position

#### Set

```go
func (t *Tensor) Set(value float64, indices ...int)
```

Sets value at specified indices.

**Parameters:**
- `value`: Value to set
- `indices`: Position in each dimension

#### Reshape

```go
func (t *Tensor) Reshape(shape ...int) (*Tensor, error)
```

Reshapes tensor to new dimensions.

**Parameters:**
- `shape`: New dimensions

**Returns:**
- `*Tensor`: Reshaped view
- `error`: If total size doesn't match

#### Slice

```go
func (t *Tensor) Slice(ranges ...Range) *Tensor
```

Creates a view of tensor slice.

**Parameters:**
- `ranges`: Start and end for each dimension

**Returns:**
- `*Tensor`: Sliced view

#### ToMatrix

```go
func (t *Tensor) ToMatrix() (mat.Matrix, error)
```

Converts 2D tensor to matrix.

**Returns:**
- `mat.Matrix`: Matrix representation
- `error`: If not 2D

### Operations

#### Add

```go
func Add(a, b *Tensor) (*Tensor, error)
```

Element-wise addition.

#### Multiply

```go
func Multiply(a, b *Tensor) (*Tensor, error)
```

Element-wise multiplication.

#### Dot

```go
func Dot(a, b *Tensor) (*Tensor, error)
```

Tensor dot product.

#### Transpose

```go
func Transpose(t *Tensor, axes ...int) *Tensor
```

Transposes tensor dimensions.

## Utilities

### Math Utilities

```go
package utils

// Clip limits values to range
func Clip(x, min, max float64) float64 {
    if x < min {
        return min
    }
    if x > max {
        return max
    }
    return x
}

// Sign returns sign of value
func Sign(x float64) float64 {
    if x > 0 {
        return 1
    }
    if x < 0 {
        return -1
    }
    return 0
}

// LogSumExp computes log(sum(exp(x))) stably
func LogSumExp(values []float64) float64 {
    max := Max(values)
    sum := 0.0
    for _, v := range values {
        sum += math.Exp(v - max)
    }
    return max + math.Log(sum)
}
```

### Array Utilities

```go
// ArgMax returns index of maximum value
func ArgMax(values []float64) int {
    maxIdx := 0
    maxVal := values[0]
    for i, v := range values[1:] {
        if v > maxVal {
            maxVal = v
            maxIdx = i + 1
        }
    }
    return maxIdx
}

// ArgMin returns index of minimum value
func ArgMin(values []float64) int

// Unique returns unique values
func Unique(values []int) []int

// Bincount counts occurrences
func Bincount(values []int) map[int]int
```

### Matrix Utilities

```go
// RowSum computes sum of each row
func RowSum(m mat.Matrix) []float64 {
    rows, _ := m.Dims()
    sums := make([]float64, rows)
    for i := 0; i < rows; i++ {
        sums[i] = mat.Sum(m.RowView(i))
    }
    return sums
}

// ColSum computes sum of each column
func ColSum(m mat.Matrix) []float64

// RowMean computes mean of each row
func RowMean(m mat.Matrix) []float64

// ColMean computes mean of each column
func ColMean(m mat.Matrix) []float64
```

### Random Utilities

```go
package random

import "math/rand"

// RandomState manages random number generation
type RandomState struct {
    source rand.Source
    rand   *rand.Rand
}

// NewRandomState creates seeded RNG
func NewRandomState(seed int64) *RandomState {
    source := rand.NewSource(seed)
    return &RandomState{
        source: source,
        rand:   rand.New(source),
    }
}

// Permutation returns random permutation
func (r *RandomState) Permutation(n int) []int

// Choice selects random elements
func (r *RandomState) Choice(n, k int, replace bool) []int

// Shuffle shuffles slice in-place
func (r *RandomState) Shuffle(slice interface{})
```

## Memory Management

### Buffer Pool

```go
package memory

import "sync"

// BufferPool manages reusable buffers
type BufferPool struct {
    pool sync.Pool
}

// NewBufferPool creates a new pool
func NewBufferPool(size int) *BufferPool {
    return &BufferPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]float64, size)
            },
        },
    }
}

// Get retrieves a buffer
func (p *BufferPool) Get() []float64 {
    return p.pool.Get().([]float64)
}

// Put returns a buffer
func (p *BufferPool) Put(buf []float64) {
    // Clear buffer
    for i := range buf {
        buf[i] = 0
    }
    p.pool.Put(buf)
}
```

### Memory Optimizer

```go
// MemoryOptimizer tracks and optimizes memory usage
type MemoryOptimizer struct {
    allocated   uint64
    deallocated uint64
    peakUsage   uint64
    mu          sync.Mutex
}

// Track records allocation
func (m *MemoryOptimizer) Track(size uint64)

// Release records deallocation
func (m *MemoryOptimizer) Release(size uint64)

// Stats returns memory statistics
func (m *MemoryOptimizer) Stats() MemoryStats
```

## Validation

### Input Validation

```go
package validation

// ValidateMatrix checks matrix validity
func ValidateMatrix(X mat.Matrix, name string) error {
    if X == nil {
        return errors.NewValueError(name, nil, "non-nil matrix")
    }
    
    rows, cols := X.Dims()
    if rows == 0 || cols == 0 {
        return errors.NewDimensionError(
            "ValidateMatrix",
            []int{1, 1},
            []int{rows, cols},
        )
    }
    
    return nil
}

// ValidateTarget checks target validity
func ValidateTarget(y mat.Matrix, nSamples int) error

// ValidateConsistentLength checks equal lengths
func ValidateConsistentLength(arrays ...interface{}) error
```

### Parameter Validation

```go
// ValidateParam checks parameter constraints
func ValidateParam(name string, value, min, max float64) error {
    if value < min || value > max {
        return errors.NewValueError(
            name,
            value,
            fmt.Sprintf("between %g and %g", min, max),
        )
    }
    return nil
}

// ValidateOption checks discrete options
func ValidateOption(name string, value string, options []string) error
```

## Logging

### Logger Interface

```go
package log

// Logger provides structured logging
type Logger interface {
    Debug(msg string, fields ...Field)
    Info(msg string, fields ...Field)
    Warn(msg string, fields ...Field)
    Error(msg string, fields ...Field)
}

// Field represents a log field
type Field struct {
    Key   string
    Value interface{}
}

// DefaultLogger returns default logger
func DefaultLogger() Logger

// WithFields creates logger with preset fields
func WithFields(fields ...Field) Logger
```

### Usage

```go
logger := log.DefaultLogger()

logger.Info("Model training started",
    log.Field{"samples", 1000},
    log.Field{"features", 20},
)

logger.Error("Training failed",
    log.Field{"error", err},
    log.Field{"iteration", 150},
)
```

## Performance Monitoring

### Timer

```go
package perf

import "time"

// Timer measures execution time
type Timer struct {
    start time.Time
    name  string
}

// Start begins timing
func Start(name string) *Timer {
    return &Timer{
        start: time.Now(),
        name:  name,
    }
}

// Stop ends timing and logs
func (t *Timer) Stop() time.Duration {
    duration := time.Since(t.start)
    log.Info("Operation completed",
        log.Field{"operation", t.name},
        log.Field{"duration_ms", duration.Milliseconds()},
    )
    return duration
}
```

### Usage

```go
timer := perf.Start("model.fit")
defer timer.Stop()

// Training code...
```

## Configuration

### Config Management

```go
package config

// Config holds global configuration
type Config struct {
    NumThreads      int
    VerboseLevel    int
    RandomSeed      int64
    CacheSize       int
    ParallelMinSize int
}

// Global returns global config
func Global() *Config

// Load loads config from file
func Load(path string) (*Config, error)

// Save saves config to file
func (c *Config) Save(path string) error
```

## Example Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/core/model"
    "github.com/YuminosukeSato/scigo/core/errors"
    "github.com/YuminosukeSato/scigo/core/parallel"
    "github.com/YuminosukeSato/scigo/core/validation"
    "gonum.org/v1/gonum/mat"
)

// CustomModel implements Model interface
type CustomModel struct {
    state   *model.StateManager
    weights *mat.VecDense
}

func NewCustomModel() *CustomModel {
    return &CustomModel{
        state: model.NewStateManager(),
    }
}

func (m *CustomModel) Fit(X, y mat.Matrix) error {
    // Validate input
    if err := validation.ValidateMatrix(X, "X"); err != nil {
        return err
    }
    if err := validation.ValidateMatrix(y, "y"); err != nil {
        return err
    }
    
    rows, cols := X.Dims()
    
    // Parallel processing for large datasets
    if rows > parallel.DefaultOptions.MinSize {
        parallel.ParallelFor(rows, func(start, end int) {
            // Process batch
            for i := start; i < end; i++ {
                // Training logic
            }
        })
    }
    
    // Update state
    m.state.SetDimensions(rows, cols)
    m.state.SetFitted()
    
    return nil
}

func (m *CustomModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Check fitted
    if !m.state.IsFitted() {
        return nil, errors.NewNotFittedError("CustomModel")
    }
    
    // Validate input
    if err := m.state.ValidateX(X); err != nil {
        return nil, err
    }
    
    // Prediction logic
    rows, _ := X.Dims()
    predictions := mat.NewDense(rows, 1, nil)
    
    return predictions, nil
}

func main() {
    // Create and train model
    model := NewCustomModel()
    
    X := mat.NewDense(100, 5, nil)
    y := mat.NewVecDense(100, nil)
    
    if err := model.Fit(X, y); err != nil {
        log.Fatal(err)
    }
    
    // Make predictions
    predictions, err := model.Predict(X)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Predictions shape: %v\n", predictions.Dims())
}
```

## Next Steps

- Explore [Model Interface](../core-concepts/model-interface.md)
- Learn about [Error Handling](../core-concepts/error-handling.md)
- See [Performance Guide](../core-concepts/performance.md)
- Read [API Examples](../../examples/core/)