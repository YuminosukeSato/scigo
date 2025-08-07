# Basic Concepts

Understanding the fundamental concepts of SciGo will help you build better machine learning applications.

## Core Philosophy

SciGo follows these principles:

1. **Go-Idiomatic**: Native Go patterns and conventions
2. **Type-Safe**: Compile-time type checking for reliability
3. **Performance-First**: Optimized for production workloads
4. **Scikit-learn Compatible**: Familiar APIs for Python developers
5. **Error-Explicit**: Clear error handling without hidden failures

## Key Components

### 1. Models

Models are the core learning algorithms:

```go
type Model interface {
    Fit(X, y mat.Matrix) error
    Predict(X mat.Matrix) (mat.Matrix, error)
    IsFitted() bool
}
```

All models implement this interface:
- `Fit`: Train the model on data
- `Predict`: Make predictions on new data
- `IsFitted`: Check if model is trained

### 2. Estimators

Estimators are objects that learn from data:

```go
// Supervised estimator
estimator.Fit(X, y)  // X: features, y: targets

// Unsupervised estimator
estimator.Fit(X, nil)  // X: features only
```

Types of estimators:
- **Regressors**: Predict continuous values
- **Classifiers**: Predict discrete classes
- **Clusterers**: Group similar data points
- **Transformers**: Transform data

### 3. Transformers

Transformers modify data:

```go
type Transformer interface {
    Fit(X mat.Matrix) error
    Transform(X mat.Matrix) (mat.Matrix, error)
    FitTransform(X mat.Matrix) (mat.Matrix, error)
}
```

Common transformers:
- `StandardScaler`: Normalize features
- `OneHotEncoder`: Encode categorical variables
- `PCA`: Reduce dimensionality

### 4. Predictors

Predictors make predictions:

```go
type Predictor interface {
    Predict(X mat.Matrix) (mat.Matrix, error)
    PredictProba(X mat.Matrix) (mat.Matrix, error)  // For classifiers
}
```

## Data Representation

### Matrices and Vectors

SciGo uses `gonum/mat` for numerical data:

```go
// Create a matrix (samples × features)
X := mat.NewDense(3, 2, []float64{
    1, 2,  // Sample 1
    3, 4,  // Sample 2
    5, 6,  // Sample 3
})

// Create a vector (targets)
y := mat.NewVecDense(3, []float64{10, 20, 30})

// Access elements
value := X.At(0, 1)  // Get element at row 0, column 1
X.Set(0, 1, 42)      // Set element

// Matrix operations
var result mat.Dense
result.Mul(X, X.T())  // Matrix multiplication
```

### Data Types

| Type | Usage | Example |
|------|-------|---------|
| `*mat.Dense` | Feature matrix | Training data |
| `*mat.VecDense` | Target vector | Labels/values |
| `[]float64` | Raw data | Input arrays |
| `float64` | Single value | Predictions |

## Model Lifecycle

### 1. Initialization

```go
model := linear.NewLinearRegression()
// or with options
model := linear.NewLinearRegression(
    linear.WithFitIntercept(true),
    linear.WithNormalize(true),
)
```

### 2. Training (Fitting)

```go
err := model.Fit(XTrain, yTrain)
if err != nil {
    // Handle training errors
}
```

### 3. Prediction

```go
predictions, err := model.Predict(XTest)
if err != nil {
    // Handle prediction errors
}
```

### 4. Evaluation

```go
score := model.Score(XTest, yTest)
fmt.Printf("Model accuracy: %.2f\n", score)
```

### 5. Persistence

```go
// Save
model.Save("model.gob")

// Load
model.Load("model.gob")
```

## Error Handling

SciGo provides explicit error handling:

### Error Types

```go
// Dimension mismatch
type DimensionError struct {
    Expected, Actual int
}

// Model not fitted
type NotFittedError struct {
    ModelName string
}

// Convergence issues
type ConvergenceWarning struct {
    Iterations int
    Message    string
}

// Numerical errors
type NumericalError struct {
    Operation string
    Value     float64
}
```

### Error Handling Pattern

```go
result, err := operation()
if err != nil {
    switch e := err.(type) {
    case *errors.DimensionError:
        log.Printf("Dimension mismatch: expected %d, got %d", 
            e.Expected, e.Actual)
    case *errors.NotFittedError:
        log.Printf("Model %s not fitted", e.ModelName)
    default:
        log.Fatal(err)
    }
}
```

## Parallel Processing

SciGo automatically parallelizes operations:

```go
// Automatic parallelization for large datasets
model := linear.NewLinearRegression()
model.Fit(X, y)  // Uses all CPU cores for X.Rows() > 1000

// Manual control
parallel.SetNumThreads(4)
```

## Memory Management

### Efficient Data Handling

```go
// Reuse memory
var result mat.Dense
result.Mul(A, B)  // Reuses result's memory

// In-place operations
X.Scale(2.0, X)  // Scale X in-place

// View without copying
col := X.ColView(0)  // View of first column
```

### Memory Patterns

```go
// Good: Reuse buffers
buffer := make([]float64, n)
for i := 0; i < iterations; i++ {
    processData(buffer)
}

// Bad: Allocate in loop
for i := 0; i < iterations; i++ {
    buffer := make([]float64, n)  // Allocates every iteration
    processData(buffer)
}
```

## State Management

Models maintain internal state:

```go
type StateManager struct {
    fitted    bool
    nFeatures int
    nSamples  int
}

// Check state
if model.IsFitted() {
    predictions := model.Predict(X)
}

// Reset state
model.Reset()
```

## Composition over Inheritance

SciGo uses composition instead of inheritance:

```go
type LinearRegression struct {
    state   *StateManager  // Composed, not inherited
    weights *mat.VecDense
}

// Not this (inheritance-style):
// type LinearRegression struct {
//     BaseEstimator  // Embedded
// }
```

## Interfaces and Contracts

### Core Interfaces

```go
// Fitter trains on data
type Fitter interface {
    Fit(X, y mat.Matrix) error
}

// Predictor makes predictions
type Predictor interface {
    Predict(X mat.Matrix) (mat.Matrix, error)
}

// Transformer modifies data
type Transformer interface {
    Transform(X mat.Matrix) (mat.Matrix, error)
}

// Model combines fitting and prediction
type Model interface {
    Fitter
    Predictor
}
```

### Implementation Example

```go
// Your custom model
type MyModel struct {
    // fields
}

func (m *MyModel) Fit(X, y mat.Matrix) error {
    // Training logic
    return nil
}

func (m *MyModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Prediction logic
    return predictions, nil
}

// Now MyModel implements Model interface
var _ Model = (*MyModel)(nil)  // Compile-time check
```

## Configuration and Options

### Option Pattern

```go
type Option func(*Config)

func WithLearningRate(lr float64) Option {
    return func(c *Config) {
        c.LearningRate = lr
    }
}

// Usage
model := NewModel(
    WithLearningRate(0.01),
    WithMaxIter(1000),
    WithTolerance(1e-4),
)
```

## Best Practices

### 1. Check Model State
```go
if !model.IsFitted() {
    return errors.New("model must be fitted first")
}
```

### 2. Validate Input Dimensions
```go
rows, cols := X.Dims()
if cols != model.NFeatures() {
    return fmt.Errorf("expected %d features, got %d", 
        model.NFeatures(), cols)
}
```

### 3. Handle Numerical Stability
```go
if math.IsNaN(value) || math.IsInf(value, 0) {
    return &NumericalError{
        Operation: "division",
        Value: value,
    }
}
```

### 4. Use Defer for Cleanup
```go
func process() error {
    resource := acquire()
    defer resource.Release()
    
    // Processing...
    return nil
}
```

## Common Patterns

### Builder Pattern
```go
model := NewModelBuilder().
    SetLearningRate(0.01).
    SetMaxIter(1000).
    SetRegularization(0.1).
    Build()
```

### Pipeline Pattern
```go
pipeline := []Transformer{
    preprocessing.NewStandardScaler(),
    preprocessing.NewPCA(n_components=10),
}

for _, transformer := range pipeline {
    X = transformer.FitTransform(X)
}
```

### Strategy Pattern
```go
type Optimizer interface {
    Optimize(weights []float64, grad []float64) []float64
}

model := NewModel(WithOptimizer(SGDOptimizer{}))
```

## Next Steps

- Learn about [Architecture](../core-concepts/architecture.md)
- Explore [Linear Models](../guides/linear-models.md)
- Understand [Error Handling](../core-concepts/error-handling.md)
- See [Complete Examples](../../examples/)