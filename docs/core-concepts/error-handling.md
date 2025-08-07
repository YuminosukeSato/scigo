# Error Handling

SciGo provides comprehensive error handling to help you build robust machine learning applications.

## Error Philosophy

SciGo follows Go's explicit error handling philosophy:

1. **Explicit over Implicit**: All errors are returned explicitly
2. **Early Detection**: Validate input early and fail fast
3. **Clear Messages**: Provide actionable error messages
4. **Type Safety**: Use typed errors for better handling
5. **No Panics**: Never panic in library code

## Error Types

### Core Error Types

```go
package errors

import "fmt"

// BaseError is the base error type
type BaseError struct {
    Op      string // Operation that failed
    Kind    string // Error category
    Message string // Human-readable message
    Err     error  // Wrapped error
}

func (e *BaseError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %s: %v", e.Op, e.Message, e.Err)
    }
    return fmt.Sprintf("%s: %s", e.Op, e.Message)
}

func (e *BaseError) Unwrap() error {
    return e.Err
}
```

### Dimension Errors

```go
// DimensionError indicates shape mismatch
type DimensionError struct {
    BaseError
    Expected []int
    Actual   []int
}

func NewDimensionError(op string, expected, actual []int) *DimensionError {
    return &DimensionError{
        BaseError: BaseError{
            Op:      op,
            Kind:    "DimensionError",
            Message: fmt.Sprintf("dimension mismatch: expected %v, got %v", expected, actual),
        },
        Expected: expected,
        Actual:   actual,
    }
}

// Example usage
func ValidateDimensions(X, y mat.Matrix) error {
    xRows, _ := X.Dims()
    yRows, _ := y.Dims()
    
    if xRows != yRows {
        return NewDimensionError(
            "Fit",
            []int{xRows},
            []int{yRows},
        )
    }
    return nil
}
```

### Not Fitted Errors

```go
// NotFittedError indicates model not trained
type NotFittedError struct {
    BaseError
    ModelName string
}

func NewNotFittedError(modelName string) *NotFittedError {
    return &NotFittedError{
        BaseError: BaseError{
            Op:      "Predict",
            Kind:    "NotFittedError",
            Message: fmt.Sprintf("model %s is not fitted yet", modelName),
        },
        ModelName: modelName,
    }
}

// Helper to check fitted state
func CheckFitted(model Model) error {
    if !model.IsFitted() {
        return NewNotFittedError(model.Name())
    }
    return nil
}
```

### Value Errors

```go
// ValueError indicates invalid parameter values
type ValueError struct {
    BaseError
    Parameter string
    Value     interface{}
    Valid     string // Description of valid values
}

func NewValueError(param string, value interface{}, valid string) *ValueError {
    return &ValueError{
        BaseError: BaseError{
            Op:      "SetParam",
            Kind:    "ValueError",
            Message: fmt.Sprintf("invalid value for %s: %v (expected %s)", param, value, valid),
        },
        Parameter: param,
        Value:     value,
        Valid:     valid,
    }
}

// Validation helper
func ValidateRange(param string, value float64, min, max float64) error {
    if value < min || value > max {
        return NewValueError(
            param,
            value,
            fmt.Sprintf("value between %g and %g", min, max),
        )
    }
    return nil
}
```

### Numerical Errors

```go
// NumericalError indicates numerical computation issues
type NumericalError struct {
    BaseError
    Operation string
    Values    []float64
}

func NewNumericalError(op string, values ...float64) *NumericalError {
    return &NumericalError{
        BaseError: BaseError{
            Op:      op,
            Kind:    "NumericalError",
            Message: "numerical computation failed",
        },
        Operation: op,
        Values:    values,
    }
}

// Check for numerical issues
func CheckNumerical(value float64) error {
    if math.IsNaN(value) {
        return NewNumericalError("computation", value)
    }
    if math.IsInf(value, 0) {
        return NewNumericalError("overflow", value)
    }
    return nil
}
```

### Convergence Warnings

```go
// ConvergenceWarning indicates optimization didn't converge
type ConvergenceWarning struct {
    BaseError
    Iterations int
    Tolerance  float64
    Loss       float64
}

func NewConvergenceWarning(iter int, tol, loss float64) *ConvergenceWarning {
    return &ConvergenceWarning{
        BaseError: BaseError{
            Op:   "Fit",
            Kind: "ConvergenceWarning",
            Message: fmt.Sprintf(
                "failed to converge after %d iterations (tol=%g, loss=%g)",
                iter, tol, loss,
            ),
        },
        Iterations: iter,
        Tolerance:  tol,
        Loss:       loss,
    }
}

// Check convergence
func CheckConvergence(iter, maxIter int, loss, tol float64) error {
    if iter >= maxIter && loss > tol {
        return NewConvergenceWarning(iter, tol, loss)
    }
    return nil
}
```

## Error Handling Patterns

### Basic Error Handling

```go
func TrainModel(X, y mat.Matrix) (*Model, error) {
    // Validate input
    if err := ValidateInput(X, y); err != nil {
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    // Create and train model
    model := NewModel()
    if err := model.Fit(X, y); err != nil {
        return nil, fmt.Errorf("training failed: %w", err)
    }
    
    return model, nil
}

// Usage
model, err := TrainModel(X, y)
if err != nil {
    log.Printf("Failed to train model: %v", err)
    return err
}
```

### Type-Based Error Handling

```go
func HandleModelError(err error) {
    switch e := err.(type) {
    case *NotFittedError:
        log.Printf("Model not fitted: %s", e.ModelName)
        // Train the model
        
    case *DimensionError:
        log.Printf("Dimension mismatch: expected %v, got %v",
            e.Expected, e.Actual)
        // Reshape data
        
    case *ValueError:
        log.Printf("Invalid %s: %v (valid: %s)",
            e.Parameter, e.Value, e.Valid)
        // Use default value
        
    case *ConvergenceWarning:
        log.Printf("Warning: Model didn't converge after %d iterations",
            e.Iterations)
        // Continue with partial result
        
    default:
        log.Printf("Unexpected error: %v", err)
        // Generic handling
    }
}
```

### Error Wrapping

```go
func ProcessData(data [][]float64) error {
    X := ConvertToMatrix(data)
    
    // Wrap errors with context
    model := LoadModel()
    if err := model.Predict(X); err != nil {
        return fmt.Errorf("prediction on %d samples failed: %w",
            len(data), err)
    }
    
    return nil
}

// Unwrap to check error type
func IsNotFittedError(err error) bool {
    var notFitted *NotFittedError
    return errors.As(err, &notFitted)
}
```

## Validation Functions

### Input Validation

```go
package validation

import (
    "math"
    "gonum.org/v1/gonum/mat"
)

// ValidateMatrix checks matrix validity
func ValidateMatrix(X mat.Matrix, name string) error {
    if X == nil {
        return NewValueError(name, nil, "non-nil matrix")
    }
    
    rows, cols := X.Dims()
    if rows == 0 || cols == 0 {
        return NewDimensionError(
            "ValidateMatrix",
            []int{1, 1}, // At least 1x1
            []int{rows, cols},
        )
    }
    
    // Check for NaN/Inf
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            if err := CheckNumerical(X.At(i, j)); err != nil {
                return fmt.Errorf("invalid value at [%d,%d]: %w", i, j, err)
            }
        }
    }
    
    return nil
}

// ValidateVector checks vector validity
func ValidateVector(v *mat.VecDense, name string) error {
    if v == nil {
        return NewValueError(name, nil, "non-nil vector")
    }
    
    n := v.Len()
    if n == 0 {
        return NewDimensionError(
            "ValidateVector",
            []int{1},
            []int{0},
        )
    }
    
    for i := 0; i < n; i++ {
        if err := CheckNumerical(v.AtVec(i)); err != nil {
            return fmt.Errorf("invalid value at [%d]: %w", i, err)
        }
    }
    
    return nil
}

// ValidateLabels checks classification labels
func ValidateLabels(y mat.Matrix, classes []int) error {
    rows, cols := y.Dims()
    
    if cols != 1 {
        return NewDimensionError(
            "ValidateLabels",
            []int{rows, 1},
            []int{rows, cols},
        )
    }
    
    classSet := make(map[int]bool)
    for _, c := range classes {
        classSet[c] = true
    }
    
    for i := 0; i < rows; i++ {
        label := int(y.At(i, 0))
        if !classSet[label] {
            return NewValueError(
                "label",
                label,
                fmt.Sprintf("one of %v", classes),
            )
        }
    }
    
    return nil
}
```

### Parameter Validation

```go
// ValidateHyperparameters checks model hyperparameters
func ValidateHyperparameters(params map[string]interface{}) error {
    // Learning rate
    if lr, ok := params["learning_rate"].(float64); ok {
        if err := ValidateRange("learning_rate", lr, 0, 1); err != nil {
            return err
        }
    }
    
    // Max iterations
    if maxIter, ok := params["max_iter"].(int); ok {
        if maxIter <= 0 {
            return NewValueError("max_iter", maxIter, "positive integer")
        }
    }
    
    // Regularization
    if alpha, ok := params["alpha"].(float64); ok {
        if alpha < 0 {
            return NewValueError("alpha", alpha, "non-negative")
        }
    }
    
    // Tolerance
    if tol, ok := params["tolerance"].(float64); ok {
        if tol <= 0 {
            return NewValueError("tolerance", tol, "positive")
        }
    }
    
    return nil
}

// ValidateProbability checks probability values
func ValidateProbability(p float64) error {
    if p < 0 || p > 1 {
        return NewValueError("probability", p, "between 0 and 1")
    }
    return nil
}

// ValidateSplitRatio checks train/test split ratio
func ValidateSplitRatio(ratio float64) error {
    if ratio <= 0 || ratio >= 1 {
        return NewValueError("split_ratio", ratio, "between 0 and 1 (exclusive)")
    }
    return nil
}
```

## Error Recovery

### Graceful Degradation

```go
type RobustModel struct {
    primary   Model
    fallback  Model
    logger    *log.Logger
}

func (r *RobustModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Try primary model
    result, err := r.primary.Predict(X)
    if err == nil {
        return result, nil
    }
    
    // Log error
    r.logger.Printf("Primary model failed: %v", err)
    
    // Check if recoverable
    var dimErr *DimensionError
    if errors.As(err, &dimErr) {
        // Try to reshape input
        X = r.reshapeInput(X, dimErr.Expected)
        result, err = r.primary.Predict(X)
        if err == nil {
            return result, nil
        }
    }
    
    // Fall back to simpler model
    r.logger.Printf("Falling back to secondary model")
    return r.fallback.Predict(X)
}
```

### Retry Logic

```go
func TrainWithRetry(model Model, X, y mat.Matrix, maxRetries int) error {
    var lastErr error
    
    for i := 0; i < maxRetries; i++ {
        err := model.Fit(X, y)
        if err == nil {
            return nil
        }
        
        lastErr = err
        
        // Check if retryable
        var convErr *ConvergenceWarning
        if errors.As(err, &convErr) {
            // Adjust parameters and retry
            params := model.GetParams()
            params["max_iter"] = convErr.Iterations * 2
            params["tolerance"] = convErr.Tolerance * 10
            model.SetParams(params)
            
            log.Printf("Retry %d with adjusted parameters", i+1)
            continue
        }
        
        // Non-retryable error
        break
    }
    
    return fmt.Errorf("failed after %d attempts: %w", maxRetries, lastErr)
}
```

## Error Aggregation

### Multi-Error Handling

```go
type MultiError struct {
    Errors []error
}

func (m *MultiError) Error() string {
    if len(m.Errors) == 0 {
        return "no errors"
    }
    
    if len(m.Errors) == 1 {
        return m.Errors[0].Error()
    }
    
    return fmt.Sprintf("%d errors occurred: %v...",
        len(m.Errors), m.Errors[0])
}

func (m *MultiError) Add(err error) {
    if err != nil {
        m.Errors = append(m.Errors, err)
    }
}

func (m *MultiError) HasErrors() bool {
    return len(m.Errors) > 0
}

// Usage
func ValidateDataset(X, y mat.Matrix) error {
    var errs MultiError
    
    if err := ValidateMatrix(X, "features"); err != nil {
        errs.Add(fmt.Errorf("invalid features: %w", err))
    }
    
    if err := ValidateMatrix(y, "targets"); err != nil {
        errs.Add(fmt.Errorf("invalid targets: %w", err))
    }
    
    if err := ValidateDimensions(X, y); err != nil {
        errs.Add(err)
    }
    
    if errs.HasErrors() {
        return &errs
    }
    
    return nil
}
```

## Error Logging

### Structured Error Logging

```go
type ErrorLogger struct {
    logger *log.Logger
    level  LogLevel
}

func (l *ErrorLogger) LogError(err error, context map[string]interface{}) {
    entry := map[string]interface{}{
        "error":     err.Error(),
        "timestamp": time.Now().Unix(),
    }
    
    // Add error type
    switch e := err.(type) {
    case *DimensionError:
        entry["type"] = "dimension"
        entry["expected"] = e.Expected
        entry["actual"] = e.Actual
        
    case *NotFittedError:
        entry["type"] = "not_fitted"
        entry["model"] = e.ModelName
        
    case *ConvergenceWarning:
        entry["type"] = "convergence"
        entry["iterations"] = e.Iterations
        entry["loss"] = e.Loss
        
    default:
        entry["type"] = "unknown"
    }
    
    // Add context
    for k, v := range context {
        entry[k] = v
    }
    
    // Log as JSON
    data, _ := json.Marshal(entry)
    l.logger.Println(string(data))
}
```

## Testing Error Conditions

### Error Testing Helpers

```go
func TestErrorConditions(t *testing.T) {
    tests := []struct {
        name      string
        input     func() error
        wantError error
    }{
        {
            name: "dimension mismatch",
            input: func() error {
                X := mat.NewDense(10, 5, nil)
                y := mat.NewVecDense(8, nil) // Wrong size
                model := NewModel()
                return model.Fit(X, y)
            },
            wantError: &DimensionError{},
        },
        {
            name: "not fitted",
            input: func() error {
                X := mat.NewDense(10, 5, nil)
                model := NewModel()
                _, err := model.Predict(X)
                return err
            },
            wantError: &NotFittedError{},
        },
        {
            name: "invalid parameter",
            input: func() error {
                model := NewModel()
                return model.SetParams(map[string]interface{}{
                    "learning_rate": -0.1, // Invalid
                })
            },
            wantError: &ValueError{},
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := tt.input()
            
            if err == nil {
                t.Fatal("expected error, got nil")
            }
            
            if !errors.As(err, &tt.wantError) {
                t.Errorf("expected error type %T, got %T",
                    tt.wantError, err)
            }
        })
    }
}
```

### Fuzzing Error Handling

```go
func FuzzErrorHandling(f *testing.F) {
    // Add seed corpus
    f.Add(10, 5, 0.1)
    f.Add(0, 0, -1.0)
    f.Add(1000000, 1000000, 1e-10)
    
    f.Fuzz(func(t *testing.T, rows, cols int, value float64) {
        // Create potentially invalid input
        if rows < 0 || cols < 0 {
            rows, cols = 0, 0
        }
        
        defer func() {
            if r := recover(); r != nil {
                t.Fatalf("panic with rows=%d, cols=%d, value=%f: %v",
                    rows, cols, value, r)
            }
        }()
        
        // Should handle gracefully without panic
        model := NewModel()
        
        if rows > 0 && cols > 0 {
            X := mat.NewDense(rows, cols, nil)
            y := mat.NewVecDense(rows, nil)
            
            // Fill with test value
            for i := 0; i < rows; i++ {
                y.SetVec(i, value)
            }
            
            // Should return error, not panic
            _ = model.Fit(X, y)
        }
    })
}
```

## Performance Considerations

### Error Allocation

```go
// Pre-allocated error pool
var errorPool = sync.Pool{
    New: func() interface{} {
        return &BaseError{}
    },
}

func GetError() *BaseError {
    return errorPool.Get().(*BaseError)
}

func PutError(err *BaseError) {
    err.Op = ""
    err.Kind = ""
    err.Message = ""
    err.Err = nil
    errorPool.Put(err)
}
```

### Fast Path Optimization

```go
func (m *Model) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Fast path: skip detailed validation in production
    if m.fastMode {
        if !m.fitted {
            return nil, ErrNotFitted // Pre-allocated error
        }
        return m.computePredictions(X), nil
    }
    
    // Detailed validation for development
    if err := m.validateForPrediction(X); err != nil {
        return nil, err
    }
    
    return m.computePredictions(X), nil
}
```

## Best Practices

1. **Always Return Errors**: Never swallow errors silently
2. **Wrap with Context**: Add context when propagating errors
3. **Use Typed Errors**: Create specific error types for different failures
4. **Validate Early**: Check inputs at function entry
5. **Document Errors**: Document which errors a function can return
6. **Test Error Paths**: Write tests for error conditions
7. **Log Strategically**: Log errors at the appropriate level
8. **Fail Fast**: Return immediately on unrecoverable errors
9. **Provide Solutions**: Include hints for fixing errors in messages
10. **Avoid Panic**: Never panic in library code

## Next Steps

- Learn about [Performance](./performance.md)
- Explore [Model Interface](./model-interface.md)
- See [API Reference](../api/core.md)
- Read [Testing Guide](../guides/testing.md)