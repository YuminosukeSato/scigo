# Linear Models API Reference

Complete API documentation for the `linear` package.

## Package Overview

```go
import "github.com/YuminosukeSato/scigo/linear"
```

The `linear` package provides linear models for regression and classification tasks.

## LinearRegression

Linear regression model using ordinary least squares.

### Constructor

```go
func NewLinearRegression() *LinearRegression
```

Creates a new linear regression model.

**Returns:**
- `*LinearRegression`: A new untrained linear regression model

**Example:**
```go
lr := linear.NewLinearRegression()
```

### Methods

#### Fit

```go
func (lr *LinearRegression) Fit(X, y mat.Matrix) error
```

Trains the linear regression model.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)
- `y`: Target vector (n_samples × 1)

**Returns:**
- `error`: Training error if any

**Errors:**
- `DimensionError`: If X and y dimensions don't match
- `SingularMatrixError`: If X'X is singular
- `ValueError`: If input contains NaN or Inf

**Example:**
```go
X := mat.NewDense(100, 3, data)
y := mat.NewVecDense(100, targets)

err := lr.Fit(X, y)
if err != nil {
    log.Fatal(err)
}
```

#### Predict

```go
func (lr *LinearRegression) Predict(X mat.Matrix) (mat.Matrix, error)
```

Makes predictions on new data.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)

**Returns:**
- `mat.Matrix`: Predictions (n_samples × 1)
- `error`: Prediction error if any

**Errors:**
- `NotFittedError`: If model hasn't been trained
- `DimensionError`: If X has wrong number of features

**Example:**
```go
XTest := mat.NewDense(10, 3, testData)
predictions, err := lr.Predict(XTest)
if err != nil {
    log.Fatal(err)
}
```

#### Score

```go
func (lr *LinearRegression) Score(X, y mat.Matrix) (float64, error)
```

Calculates R² score of the model.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)
- `y`: True target values (n_samples × 1)

**Returns:**
- `float64`: R² score (0.0 to 1.0, higher is better)
- `error`: Scoring error if any

**Example:**
```go
r2, err := lr.Score(XTest, yTest)
fmt.Printf("R² score: %.3f\n", r2)
```

#### GetWeights

```go
func (lr *LinearRegression) GetWeights() []float64
```

Returns the learned coefficients.

**Returns:**
- `[]float64`: Model coefficients for each feature

**Example:**
```go
weights := lr.GetWeights()
for i, w := range weights {
    fmt.Printf("Feature %d weight: %.4f\n", i, w)
}
```

#### GetIntercept

```go
func (lr *LinearRegression) GetIntercept() float64
```

Returns the learned intercept.

**Returns:**
- `float64`: Model intercept (bias term)

**Example:**
```go
intercept := lr.GetIntercept()
fmt.Printf("Intercept: %.4f\n", intercept)
```

#### IsFitted

```go
func (lr *LinearRegression) IsFitted() bool
```

Checks if the model has been trained.

**Returns:**
- `bool`: true if model is fitted, false otherwise

**Example:**
```go
if lr.IsFitted() {
    predictions, _ := lr.Predict(X)
}
```

### Persistence Methods

#### SaveModel

```go
func (lr *LinearRegression) SaveModel(filename string) error
```

Saves the model to a file.

**Parameters:**
- `filename`: Path to save the model

**Returns:**
- `error`: Save error if any

**Example:**
```go
err := lr.SaveModel("model.gob")
```

#### LoadModel

```go
func (lr *LinearRegression) LoadModel(filename string) error
```

Loads a model from a file.

**Parameters:**
- `filename`: Path to the model file

**Returns:**
- `error`: Load error if any

**Example:**
```go
lr := linear.NewLinearRegression()
err := lr.LoadModel("model.gob")
```

### Scikit-learn Compatibility

#### LoadFromSKLearn

```go
func (lr *LinearRegression) LoadFromSKLearn(filename string) error
```

Loads a scikit-learn trained model.

**Parameters:**
- `filename`: Path to the JSON file from scikit-learn

**Returns:**
- `error`: Load error if any

**Example:**
```go
lr := linear.NewLinearRegression()
err := lr.LoadFromSKLearn("sklearn_model.json")
```

#### ExportToSKLearn

```go
func (lr *LinearRegression) ExportToSKLearn(filename string) error
```

Exports the model to scikit-learn format.

**Parameters:**
- `filename`: Path to save the JSON file

**Returns:**
- `error`: Export error if any

**Example:**
```go
err := lr.ExportToSKLearn("model_for_python.json")
```

## Complete Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "github.com/YuminosukeSato/scigo/metrics"
    "github.com/YuminosukeSato/scigo/preprocessing"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Generate sample data
    X := mat.NewDense(100, 3, nil)
    y := mat.NewVecDense(100, nil)
    
    // ... fill X and y with your data ...
    
    // Split data (80/20)
    trainSize := 80
    XTrain := X.Slice(0, trainSize, 0, 3).(*mat.Dense)
    yTrain := mat.VecDenseCopyOf(y.SliceVec(0, trainSize))
    XTest := X.Slice(trainSize, 100, 0, 3).(*mat.Dense)
    yTest := mat.VecDenseCopyOf(y.SliceVec(trainSize, 100))
    
    // Preprocess data
    scaler := preprocessing.NewStandardScaler()
    scaler.Fit(XTrain)
    XTrainScaled, _ := scaler.Transform(XTrain)
    XTestScaled, _ := scaler.Transform(XTest)
    
    // Create and train model
    lr := linear.NewLinearRegression()
    
    err := lr.Fit(XTrainScaled, yTrain)
    if err != nil {
        log.Fatal("Training failed:", err)
    }
    
    // Make predictions
    predictions, err := lr.Predict(XTestScaled)
    if err != nil {
        log.Fatal("Prediction failed:", err)
    }
    
    // Evaluate model
    mse, _ := metrics.MSE(yTest, predictions.(*mat.Dense).ColView(0))
    r2, _ := metrics.R2Score(yTest, predictions.(*mat.Dense).ColView(0))
    
    fmt.Printf("MSE: %.4f\n", mse)
    fmt.Printf("R²: %.4f\n", r2)
    
    // Get model parameters
    fmt.Printf("Weights: %v\n", lr.GetWeights())
    fmt.Printf("Intercept: %.4f\n", lr.GetIntercept())
    
    // Save model
    err = lr.SaveModel("my_model.gob")
    if err != nil {
        log.Fatal("Save failed:", err)
    }
    
    // Export for Python
    err = lr.ExportToSKLearn("model_for_sklearn.json")
    if err != nil {
        log.Fatal("Export failed:", err)
    }
}
```

## Mathematical Background

### Ordinary Least Squares

The linear regression model finds weights **w** and intercept **b** that minimize:

```
L = Σ(y_i - (w·x_i + b))²
```

Using the normal equation:
```
w = (X'X)⁻¹X'y
```

Where:
- X' is the transpose of X
- (X'X)⁻¹ is the inverse of X'X

### Assumptions

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Fit | O(n·m² + m³) | O(m²) |
| Predict | O(n·m) | O(n) |
| Score | O(n·m) | O(n) |

Where:
- n = number of samples
- m = number of features

## Thread Safety

LinearRegression is thread-safe for predictions after fitting:

```go
// Safe concurrent predictions
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(data mat.Matrix) {
        defer wg.Done()
        predictions, _ := lr.Predict(data)
        // Process predictions
    }(testData[i])
}
wg.Wait()
```

## Common Pitfalls

### 1. Multicollinearity
```go
// Check for highly correlated features
correlation := ComputeCorrelation(X)
if HasHighCorrelation(correlation) {
    // Consider removing correlated features
}
```

### 2. Outliers
```go
// Consider robust regression for outliers
if HasOutliers(y) {
    // Use robust methods or remove outliers
}
```

### 3. Feature Scaling
```go
// Always scale features with different ranges
scaler := preprocessing.NewStandardScaler()
XScaled, _ := scaler.FitTransform(X)
lr.Fit(XScaled, y)
```

## See Also

- [Preprocessing API](./preprocessing.md) - Data preparation
- [Metrics API](./metrics.md) - Model evaluation
- [SGD Models](./sklearn.md#sgdregressor) - For large datasets
- [Examples](../../examples/linear_regression/) - Working examples