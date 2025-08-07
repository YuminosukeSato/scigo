# Model Interface

The model interface is the foundation of SciGo's architecture, providing a consistent API across all machine learning algorithms.

## Core Interface Design

### Base Model Interface

```go
package model

import "gonum.org/v1/gonum/mat"

// Model is the base interface for all models
type Model interface {
    // Core methods
    Fit(X, y mat.Matrix) error
    Predict(X mat.Matrix) (mat.Matrix, error)
    
    // State management
    IsFitted() bool
    Reset() error
    
    // Parameter management
    GetParams() map[string]interface{}
    SetParams(params map[string]interface{}) error
    
    // Persistence
    Save(path string) error
    Load(path string) error
}
```

### Supervised Learning

```go
// SupervisedModel extends Model with scoring capability
type SupervisedModel interface {
    Model
    Score(X, y mat.Matrix) (float64, error)
}

// Regressor for continuous targets
type Regressor interface {
    SupervisedModel
    // Returns R² score by default
}

// Classifier for discrete targets
type Classifier interface {
    SupervisedModel
    PredictProba(X mat.Matrix) (mat.Matrix, error)
    PredictLogProba(X mat.Matrix) (mat.Matrix, error)
    Classes() []interface{}
}

// BinaryClassifier for two-class problems
type BinaryClassifier interface {
    Classifier
    DecisionFunction(X mat.Matrix) (*mat.VecDense, error)
}

// MultiClassifier for multi-class problems
type MultiClassifier interface {
    Classifier
    NClasses() int
}
```

### Unsupervised Learning

```go
// UnsupervisedModel for clustering and dimensionality reduction
type UnsupervisedModel interface {
    Model
    FitPredict(X mat.Matrix) (mat.Matrix, error)
    Transform(X mat.Matrix) (mat.Matrix, error)
}

// Clusterer groups similar data points
type Clusterer interface {
    UnsupervisedModel
    NClusters() int
    ClusterCenters() mat.Matrix
    Labels() []int
    Inertia() float64
}

// DimensionalityReducer reduces feature dimensions
type DimensionalityReducer interface {
    UnsupervisedModel
    NComponents() int
    ExplainedVariance() []float64
    Components() mat.Matrix
}
```

## State Management

All models use a consistent state management pattern:

```go
// StateManager handles model state
type StateManager struct {
    fitted    bool
    nFeatures int
    nSamples  int
    mu        sync.RWMutex
    metadata  map[string]interface{}
}

// Thread-safe state operations
func (s *StateManager) SetFitted() {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.fitted = true
}

func (s *StateManager) IsFitted() bool {
    s.mu.RLock()
    defer s.mu.RUnlock()
    return s.fitted
}

func (s *StateManager) ValidateX(X mat.Matrix) error {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    if !s.fitted {
        return &NotFittedError{"model must be fitted first"}
    }
    
    _, cols := X.Dims()
    if cols != s.nFeatures {
        return &DimensionError{
            Expected: s.nFeatures,
            Actual:   cols,
        }
    }
    
    return nil
}

func (s *StateManager) Reset() {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    s.fitted = false
    s.nFeatures = 0
    s.nSamples = 0
    s.metadata = make(map[string]interface{})
}
```

## Implementation Pattern

### Standard Implementation Structure

```go
package mymodel

import (
    "github.com/YuminosukeSato/scigo/core/model"
    "gonum.org/v1/gonum/mat"
)

// MyModel implements the Model interface
type MyModel struct {
    // State management
    state *model.StateManager
    
    // Model parameters
    weights *mat.VecDense
    bias    float64
    
    // Hyperparameters
    learningRate float64
    maxIter      int
    tolerance    float64
    
    // Training history
    lossHistory []float64
}

// NewMyModel creates a new model instance
func NewMyModel(opts ...Option) *MyModel {
    m := &MyModel{
        state:        model.NewStateManager(),
        learningRate: 0.01,
        maxIter:      1000,
        tolerance:    1e-4,
    }
    
    // Apply options
    for _, opt := range opts {
        opt(m)
    }
    
    return m
}

// Fit trains the model
func (m *MyModel) Fit(X, y mat.Matrix) error {
    // Validate input
    rows, cols := X.Dims()
    yRows, yCols := y.Dims()
    
    if rows != yRows {
        return &model.DimensionError{
            Expected: rows,
            Actual:   yRows,
            Message:  "X and y must have same number of samples",
        }
    }
    
    if yCols != 1 {
        return &model.ValueError{
            Message: "y must be a column vector",
        }
    }
    
    // Initialize weights
    m.weights = mat.NewVecDense(cols, nil)
    m.bias = 0.0
    
    // Training loop
    for iter := 0; iter < m.maxIter; iter++ {
        // Compute predictions
        predictions := m.computePredictions(X)
        
        // Compute loss
        loss := m.computeLoss(predictions, y)
        m.lossHistory = append(m.lossHistory, loss)
        
        // Check convergence
        if iter > 0 {
            improvement := m.lossHistory[iter-1] - loss
            if improvement < m.tolerance {
                break
            }
        }
        
        // Update weights
        m.updateWeights(X, y, predictions)
    }
    
    // Update state
    m.state.SetFitted()
    m.state.SetDimensions(rows, cols)
    
    return nil
}

// Predict makes predictions
func (m *MyModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Validate state and input
    if err := m.state.ValidateX(X); err != nil {
        return nil, err
    }
    
    return m.computePredictions(X), nil
}

// IsFitted checks if model is trained
func (m *MyModel) IsFitted() bool {
    return m.state.IsFitted()
}

// Reset clears the model state
func (m *MyModel) Reset() error {
    m.state.Reset()
    m.weights = nil
    m.bias = 0.0
    m.lossHistory = nil
    return nil
}

// GetParams returns model parameters
func (m *MyModel) GetParams() map[string]interface{} {
    params := make(map[string]interface{})
    
    if m.weights != nil {
        params["weights"] = m.weights.RawVector().Data
    }
    params["bias"] = m.bias
    params["learning_rate"] = m.learningRate
    params["max_iter"] = m.maxIter
    params["tolerance"] = m.tolerance
    
    return params
}

// SetParams sets model parameters
func (m *MyModel) SetParams(params map[string]interface{}) error {
    if lr, ok := params["learning_rate"].(float64); ok {
        m.learningRate = lr
    }
    
    if maxIter, ok := params["max_iter"].(int); ok {
        m.maxIter = maxIter
    }
    
    if tol, ok := params["tolerance"].(float64); ok {
        m.tolerance = tol
    }
    
    if weights, ok := params["weights"].([]float64); ok {
        m.weights = mat.NewVecDense(len(weights), weights)
    }
    
    if bias, ok := params["bias"].(float64); ok {
        m.bias = bias
    }
    
    return nil
}
```

## Option Pattern

Configure models using functional options:

```go
type Option func(*MyModel)

func WithLearningRate(lr float64) Option {
    return func(m *MyModel) {
        m.learningRate = lr
    }
}

func WithMaxIter(maxIter int) Option {
    return func(m *MyModel) {
        m.maxIter = maxIter
    }
}

func WithTolerance(tol float64) Option {
    return func(m *MyModel) {
        m.tolerance = tol
    }
}

// Usage
model := NewMyModel(
    WithLearningRate(0.001),
    WithMaxIter(5000),
    WithTolerance(1e-6),
)
```

## Validation Helpers

### Input Validation

```go
// ValidateInput checks input dimensions and values
func ValidateInput(X, y mat.Matrix) error {
    xRows, xCols := X.Dims()
    
    // Check for empty input
    if xRows == 0 || xCols == 0 {
        return &ValueError{"input matrix cannot be empty"}
    }
    
    // Check for NaN or Inf
    for i := 0; i < xRows; i++ {
        for j := 0; j < xCols; j++ {
            val := X.At(i, j)
            if math.IsNaN(val) || math.IsInf(val, 0) {
                return &NumericalError{
                    Message: fmt.Sprintf("invalid value at [%d,%d]: %v", i, j, val),
                }
            }
        }
    }
    
    // Validate y if provided
    if y != nil {
        yRows, yCols := y.Dims()
        
        if yRows != xRows {
            return &DimensionError{
                Expected: xRows,
                Actual:   yRows,
                Message:  "X and y must have same number of samples",
            }
        }
        
        if yCols != 1 && yCols != xCols {
            return &ValueError{
                Message: "y must be a vector or have same columns as X",
            }
        }
    }
    
    return nil
}

// ValidateProbability checks probability constraints
func ValidateProbability(proba mat.Matrix) error {
    rows, cols := proba.Dims()
    
    for i := 0; i < rows; i++ {
        sum := 0.0
        for j := 0; j < cols; j++ {
            p := proba.At(i, j)
            
            if p < 0 || p > 1 {
                return &ValueError{
                    Message: fmt.Sprintf("probability out of range [0,1]: %v", p),
                }
            }
            
            sum += p
        }
        
        if math.Abs(sum-1.0) > 1e-6 {
            return &ValueError{
                Message: fmt.Sprintf("probabilities don't sum to 1: %v", sum),
            }
        }
    }
    
    return nil
}
```

## Persistence Interface

### Binary Serialization

```go
import (
    "encoding/gob"
    "os"
)

// Save saves model to binary file
func (m *MyModel) Save(path string) error {
    if !m.IsFitted() {
        return &NotFittedError{"cannot save unfitted model"}
    }
    
    file, err := os.Create(path)
    if err != nil {
        return fmt.Errorf("failed to create file: %w", err)
    }
    defer file.Close()
    
    encoder := gob.NewEncoder(file)
    
    // Encode model data
    modelData := ModelData{
        Weights:      m.weights.RawVector().Data,
        Bias:         m.bias,
        LearningRate: m.learningRate,
        MaxIter:      m.maxIter,
        Tolerance:    m.tolerance,
        NFeatures:    m.state.NFeatures(),
    }
    
    if err := encoder.Encode(modelData); err != nil {
        return fmt.Errorf("failed to encode model: %w", err)
    }
    
    return nil
}

// Load loads model from binary file
func (m *MyModel) Load(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("failed to open file: %w", err)
    }
    defer file.Close()
    
    decoder := gob.NewDecoder(file)
    
    var modelData ModelData
    if err := decoder.Decode(&modelData); err != nil {
        return fmt.Errorf("failed to decode model: %w", err)
    }
    
    // Restore model state
    m.weights = mat.NewVecDense(len(modelData.Weights), modelData.Weights)
    m.bias = modelData.Bias
    m.learningRate = modelData.LearningRate
    m.maxIter = modelData.MaxIter
    m.tolerance = modelData.Tolerance
    
    m.state.SetFitted()
    m.state.SetDimensions(0, modelData.NFeatures)
    
    return nil
}
```

### JSON Serialization

```go
import "encoding/json"

// ExportJSON exports model to JSON
func (m *MyModel) ExportJSON(path string) error {
    params := m.GetParams()
    
    data, err := json.MarshalIndent(params, "", "  ")
    if err != nil {
        return err
    }
    
    return os.WriteFile(path, data, 0644)
}

// ImportJSON imports model from JSON
func (m *MyModel) ImportJSON(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    
    var params map[string]interface{}
    if err := json.Unmarshal(data, &params); err != nil {
        return err
    }
    
    return m.SetParams(params)
}
```

## Streaming Interface

For online learning:

```go
// StreamingModel supports incremental learning
type StreamingModel interface {
    Model
    PartialFit(X, y mat.Matrix) error
    FitStream(ctx context.Context, dataChan <-chan *Batch) error
}

// Batch represents a data batch
type Batch struct {
    X mat.Matrix
    Y mat.Matrix
    Weight float64
}

// Implementation
func (m *MyModel) PartialFit(X, y mat.Matrix) error {
    // Single batch update
    predictions := m.computePredictions(X)
    m.updateWeights(X, y, predictions)
    
    if !m.IsFitted() {
        rows, cols := X.Dims()
        m.state.SetDimensions(rows, cols)
        m.state.SetFitted()
    }
    
    return nil
}

func (m *MyModel) FitStream(ctx context.Context, dataChan <-chan *Batch) error {
    for {
        select {
        case batch, ok := <-dataChan:
            if !ok {
                return nil // Channel closed
            }
            
            if err := m.PartialFit(batch.X, batch.Y); err != nil {
                return err
            }
            
        case <-ctx.Done():
            return ctx.Err()
        }
    }
}
```

## Ensemble Interface

For combining multiple models:

```go
// Ensemble combines multiple models
type Ensemble interface {
    Model
    AddModel(model Model, weight float64)
    Models() []Model
    Weights() []float64
}

// VotingEnsemble for classification
type VotingEnsemble struct {
    models  []Classifier
    weights []float64
    voting  string // "hard" or "soft"
}

func (e *VotingEnsemble) Predict(X mat.Matrix) (mat.Matrix, error) {
    if e.voting == "hard" {
        return e.hardVoting(X)
    }
    return e.softVoting(X)
}

func (e *VotingEnsemble) hardVoting(X mat.Matrix) (mat.Matrix, error) {
    rows, _ := X.Dims()
    votes := make([][]int, rows)
    
    for i, model := range e.models {
        pred, err := model.Predict(X)
        if err != nil {
            return nil, err
        }
        
        // Count votes
        for j := 0; j < rows; j++ {
            class := int(pred.At(j, 0))
            votes[j][class] += int(e.weights[i])
        }
    }
    
    // Return majority vote
    result := mat.NewDense(rows, 1, nil)
    for i, vote := range votes {
        result.Set(i, 0, float64(argmax(vote)))
    }
    
    return result, nil
}
```

## Model Registry

Manage model versions:

```go
// ModelRegistry manages model versions
type ModelRegistry struct {
    models map[string]map[string]Model // name -> version -> model
    mu     sync.RWMutex
}

func NewModelRegistry() *ModelRegistry {
    return &ModelRegistry{
        models: make(map[string]map[string]Model),
    }
}

func (r *ModelRegistry) Register(name, version string, model Model) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    if r.models[name] == nil {
        r.models[name] = make(map[string]Model)
    }
    
    if _, exists := r.models[name][version]; exists {
        return fmt.Errorf("model %s version %s already exists", name, version)
    }
    
    r.models[name][version] = model
    return nil
}

func (r *ModelRegistry) Get(name, version string) (Model, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    versions, ok := r.models[name]
    if !ok {
        return nil, fmt.Errorf("model %s not found", name)
    }
    
    model, ok := versions[version]
    if !ok {
        return nil, fmt.Errorf("model %s version %s not found", name, version)
    }
    
    return model, nil
}

func (r *ModelRegistry) GetLatest(name string) (Model, string, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    versions, ok := r.models[name]
    if !ok {
        return nil, "", fmt.Errorf("model %s not found", name)
    }
    
    var latestVersion string
    for version := range versions {
        if version > latestVersion {
            latestVersion = version
        }
    }
    
    return versions[latestVersion], latestVersion, nil
}
```

## Testing Utilities

### Mock Model

```go
// MockModel for testing
type MockModel struct {
    FitFunc     func(X, y mat.Matrix) error
    PredictFunc func(X mat.Matrix) (mat.Matrix, error)
    fitted      bool
}

func (m *MockModel) Fit(X, y mat.Matrix) error {
    if m.FitFunc != nil {
        return m.FitFunc(X, y)
    }
    m.fitted = true
    return nil
}

func (m *MockModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    if m.PredictFunc != nil {
        return m.PredictFunc(X)
    }
    if !m.fitted {
        return nil, &NotFittedError{"mock model not fitted"}
    }
    rows, _ := X.Dims()
    return mat.NewDense(rows, 1, nil), nil
}
```

### Model Testing

```go
func TestModelInterface(t *testing.T, model Model) {
    // Generate test data
    X := mat.NewDense(100, 5, nil)
    y := mat.NewVecDense(100, nil)
    
    // Test unfitted state
    assert.False(t, model.IsFitted())
    
    _, err := model.Predict(X)
    assert.Error(t, err)
    assert.IsType(t, &NotFittedError{}, err)
    
    // Test fitting
    err = model.Fit(X, y)
    assert.NoError(t, err)
    assert.True(t, model.IsFitted())
    
    // Test prediction
    pred, err := model.Predict(X)
    assert.NoError(t, err)
    assert.NotNil(t, pred)
    
    rows, cols := pred.Dims()
    assert.Equal(t, 100, rows)
    assert.Equal(t, 1, cols)
    
    // Test parameters
    params := model.GetParams()
    assert.NotNil(t, params)
    
    err = model.SetParams(params)
    assert.NoError(t, err)
    
    // Test reset
    err = model.Reset()
    assert.NoError(t, err)
    assert.False(t, model.IsFitted())
}
```

## Best Practices

### 1. Always Validate Input
```go
func (m *Model) Fit(X, y mat.Matrix) error {
    if err := ValidateInput(X, y); err != nil {
        return fmt.Errorf("invalid input: %w", err)
    }
    // Training logic
}
```

### 2. Use Composition Over Inheritance
```go
type MyModel struct {
    state *StateManager  // Composed, not embedded
    // Other fields
}
```

### 3. Thread-Safe State
```go
func (m *Model) Predict(X mat.Matrix) (mat.Matrix, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    // Prediction logic
}
```

### 4. Clear Error Messages
```go
if cols != m.nFeatures {
    return fmt.Errorf("expected %d features, got %d", 
        m.nFeatures, cols)
}
```

### 5. Efficient Memory Use
```go
// Reuse buffers
if m.buffer == nil || len(m.buffer) < n {
    m.buffer = make([]float64, n)
}
```

## Next Steps

- Learn about [Error Handling](./error-handling.md)
- Explore [Performance Optimization](./performance.md)
- See [API Reference](../api/core.md)
- Read [Implementation Examples](../../examples/)