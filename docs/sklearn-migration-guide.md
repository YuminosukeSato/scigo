# Migrating from scikit-learn to SciGo

A comprehensive guide for Python developers transitioning from scikit-learn to SciGo.

## Table of Contents

- [Overview](#overview)
- [Quick Comparison](#quick-comparison)
- [API Mapping](#api-mapping)
- [Code Examples](#code-examples)
- [Performance Benefits](#performance-benefits)
- [Common Patterns](#common-patterns)
- [Advanced Migration](#advanced-migration)
- [Troubleshooting](#troubleshooting)

## Overview

SciGo provides a familiar, scikit-learn compatible API in Go, enabling Python developers to leverage their existing ML knowledge while gaining the performance benefits of Go.

### Key Benefits of Migration

- **🚀 3.6× Performance Improvement**: Native Go concurrency and optimization
- **🔧 Familiar API**: Same `fit()`, `predict()`, `transform()` patterns
- **📦 Single Binary**: No Python runtime or dependency management
- **🌊 Built-in Streaming**: Real-time ML without additional frameworks
- **🛡️ Type Safety**: Compile-time error detection and prevention

## Quick Comparison

| Feature | scikit-learn (Python) | SciGo (Go) |
|---------|----------------------|------------|
| **API Style** | `model.fit(X, y)` | `model.Fit(X, y)` |
| **Error Handling** | Exceptions | Explicit error returns |
| **Data Type** | NumPy arrays | Gonum matrices |
| **Performance** | Single-threaded default | Parallel by default |
| **Memory** | GC pauses | Predictable allocation |
| **Deployment** | Python + dependencies | Single binary |

## API Mapping

### Core Estimator Interface

```python
# scikit-learn (Python)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

```go
// SciGo (Go)
import (
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

model := linear.NewLinearRegression()
err := model.Fit(XTrain, yTrain)
predictions, err := model.Predict(XTest)
score, err := model.Score(XTest, yTest)
```

### Data Preprocessing

```python
# scikit-learn (Python)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_test)

# MinMax scaling
minmax = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax.fit_transform(X_train)
```

```go
// SciGo (Go)
import "github.com/YuminosukeSato/scigo/preprocessing"

// Standard scaling
scaler := preprocessing.NewStandardScaler(true, true)
err := scaler.Fit(XTrain)
XScaled, err := scaler.Transform(XTest)

// MinMax scaling
minmax := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})
XMinmax, err := minmax.FitTransform(XTrain)
```

### Model Evaluation

```python
# scikit-learn (Python)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

```go
// SciGo (Go)
import "github.com/YuminosukeSato/scigo/metrics"

mse, err := metrics.MSE(yTrue, yPred)
r2, err := metrics.R2Score(yTrue, yPred)
```

## Code Examples

### Complete Migration Example

Here's a complete example showing the migration of a typical ML pipeline:

#### Python (scikit-learn)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

#### Go (SciGo)

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "github.com/YuminosukeSato/scigo/preprocessing" 
    "github.com/YuminosukeSato/scigo/metrics"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Load data (assuming you have X and y as *mat.Dense)
    X := mat.NewDense(100, 4, data)  // Your data here
    y := mat.NewDense(100, 1, targets) // Your targets here
    
    // Split data (implement train_test_split or use existing)
    XTrain, XTest, yTrain, yTest := trainTestSplit(X, y, 0.2, 42)
    
    // Preprocessing
    scaler := preprocessing.NewStandardScaler(true, true)
    if err := scaler.Fit(XTrain); err != nil {
        log.Fatal(err)
    }
    
    XTrainScaled, err := scaler.Transform(XTrain)
    if err != nil {
        log.Fatal(err)
    }
    
    XTestScaled, err := scaler.Transform(XTest)
    if err != nil {
        log.Fatal(err)
    }
    
    // Train model
    model := linear.NewLinearRegression()
    if err := model.Fit(XTrainScaled, yTrain); err != nil {
        log.Fatal(err)
    }
    
    // Evaluate
    predictions, err := model.Predict(XTestScaled)
    if err != nil {
        log.Fatal(err)
    }
    
    mse, err := metrics.MSE(yTest, predictions)
    if err != nil {
        log.Fatal(err)
    }
    
    r2, err := metrics.R2Score(yTest, predictions)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("MSE: %.4f\n", mse)
    fmt.Printf("R²: %.4f\n", r2)
}
```

### Streaming Example

SciGo provides built-in streaming capabilities that aren't available in base scikit-learn:

```go
// Real-time learning (unique to SciGo)
model := linear.NewSGDRegressor()

// Process streaming data
for batch := range dataStream {
    if err := model.PartialFit(batch.X, batch.y); err != nil {
        log.Printf("Training error: %v", err)
        continue
    }
    
    // Make real-time predictions
    predictions, err := model.Predict(batch.X)
    if err == nil {
        // Process predictions in real-time
        processRealtimePredictions(predictions)
    }
}
```

## Performance Benefits

### Benchmark Comparison

| Task | Dataset Size | scikit-learn | SciGo | Speedup |
|------|--------------|--------------|-------|---------|
| Linear Regression | 1M × 100 | 890ms | 245ms | **3.6×** |
| StandardScaler | 500K × 50 | 120ms | 41ms | **2.9×** |
| Batch Prediction | 100K × 20 | 85ms | 28ms | **3.0×** |

### Memory Usage

```go
// SciGo: Predictable memory allocation
model := linear.NewLinearRegression()
// Memory usage: ~O(n_features²) for coefficient matrix
// No unexpected GC pauses
```

```python
# scikit-learn: Subject to Python GC
model = LinearRegression()
# Memory usage: Higher overhead + Python object costs
# Potential GC pauses during large operations
```

## Common Patterns

### Error Handling Migration

```python
# Python: Exception handling
try:
    model.fit(X, y)
    predictions = model.predict(X_test)
except ValueError as e:
    print(f"Error: {e}")
```

```go
// Go: Explicit error handling
if err := model.Fit(X, y); err != nil {
    log.Printf("Error: %v", err)
    return
}

predictions, err := model.Predict(XTest)
if err != nil {
    log.Printf("Prediction error: %v", err)
    return
}
```

### Pipeline Migration

```python
# Python: sklearn Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
```

```go
// Go: Manual pipeline (or use SciGo pipeline when available)
scaler := preprocessing.NewStandardScaler(true, true)
model := linear.NewLinearRegression()

// Fit pipeline
if err := scaler.Fit(XTrain); err != nil {
    return err
}
XScaled, err := scaler.Transform(XTrain)
if err != nil {
    return err
}
if err := model.Fit(XScaled, yTrain); err != nil {
    return err
}
```

### Cross-Validation Pattern

```python
# Python: Built-in cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```

```go
// Go: Implement custom cross-validation
func crossValidate(model Estimator, X, y *mat.Dense, folds int) ([]float64, error) {
    scores := make([]float64, folds)
    // Implementation here...
    return scores, nil
}
```

## Advanced Migration

### Custom Estimators

```python
# Python: Custom sklearn estimator
from sklearn.base import BaseEstimator, RegressorMixin

class CustomRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # Custom fitting logic
        return self
    
    def predict(self, X):
        # Custom prediction logic
        return predictions
```

```go
// Go: Implement estimator interface
type CustomRegressor struct {
    model.BaseEstimator
    // Custom fields
}

func (c *CustomRegressor) Fit(X, y mat.Matrix) error {
    // Custom fitting logic
    c.SetFitted(true)
    return nil
}

func (c *CustomRegressor) Predict(X mat.Matrix) (mat.Matrix, error) {
    if !c.IsFitted() {
        return nil, errors.New("not fitted")
    }
    // Custom prediction logic
    return predictions, nil
}
```

### Hyperparameter Optimization

```python
# Python: GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X, y)
best_model = grid_search.best_estimator_
```

```go
// Go: Custom grid search
func gridSearchCV(model Estimator, params map[string][]float64, X, y *mat.Dense) (Estimator, error) {
    var bestModel Estimator
    var bestScore float64 = -math.Inf(1)
    
    for paramName, values := range params {
        for _, value := range values {
            // Set parameter and evaluate
            testModel := cloneModel(model)
            setParameter(testModel, paramName, value)
            
            score, err := crossValidateScore(testModel, X, y, 5)
            if err != nil {
                continue
            }
            
            if score > bestScore {
                bestScore = score
                bestModel = testModel
            }
        }
    }
    
    return bestModel, nil
}
```

## Migration Checklist

### Phase 1: Setup
- [ ] Install Go (1.23+)
- [ ] Add SciGo dependency: `go get github.com/YuminosukeSato/scigo`
- [ ] Set up Go development environment
- [ ] Convert data loading to Gonum matrices

### Phase 2: Core Migration
- [ ] Replace scikit-learn imports with SciGo imports
- [ ] Update estimator instantiation (constructor patterns)
- [ ] Add explicit error handling for all operations
- [ ] Convert NumPy arrays to `*mat.Dense` matrices

### Phase 3: Testing
- [ ] Verify numerical accuracy matches Python results
- [ ] Add unit tests for critical paths
- [ ] Performance benchmarking vs original Python code
- [ ] Memory usage profiling

### Phase 4: Optimization
- [ ] Enable Go's built-in parallelization
- [ ] Optimize hot paths with Go-specific patterns  
- [ ] Add streaming capabilities where beneficial
- [ ] Configure for production deployment

## Troubleshooting

### Common Issues

#### 1. Matrix Dimension Mismatches

```go
// Problem: Incorrect matrix dimensions
X := mat.NewDense(100, 4, data)
y := mat.NewDense(100, 2, targets) // Wrong! Should be 100x1

// Solution: Verify dimensions
rows, cols := X.Dims()
yRows, yCols := y.Dims()
if rows != yRows || yCols != 1 {
    return fmt.Errorf("dimension mismatch: X(%d,%d), y(%d,%d)", rows, cols, yRows, yCols)
}
```

#### 2. Error Handling Patterns

```go
// Problem: Ignoring errors
model.Fit(X, y)
predictions, _ := model.Predict(X)

// Solution: Proper error handling
if err := model.Fit(X, y); err != nil {
    return fmt.Errorf("fit failed: %w", err)
}
predictions, err := model.Predict(X)
if err != nil {
    return fmt.Errorf("prediction failed: %w", err)
}
```

#### 3. Data Type Conversions

```go
// Problem: Assuming float64 compatibility
data := []float32{1.0, 2.0, 3.0}
X := mat.NewDense(1, 3, data) // Won't compile

// Solution: Use float64
data := []float64{1.0, 2.0, 3.0}
X := mat.NewDense(1, 3, data) // Correct
```

### Performance Optimization Tips

1. **Use appropriate matrix sizes**: Pre-allocate matrices when possible
2. **Enable parallelization**: SciGo uses Go routines automatically for large datasets
3. **Memory profiling**: Use `go tool pprof` to identify bottlenecks
4. **Batch operations**: Process data in chunks for memory efficiency

### Getting Help

- **Documentation**: [pkg.go.dev/scigo](https://pkg.go.dev/github.com/YuminosukeSato/scigo)
- **Examples**: [GitHub Examples](https://github.com/YuminosukeSato/scigo/tree/main/examples)
- **Issues**: [GitHub Issues](https://github.com/YuminosukeSato/scigo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YuminosukeSato/scigo/discussions)

## Next Steps

After successful migration:

1. **Explore Go-specific features**: Streaming, concurrency, type safety
2. **Optimize for production**: Single binary deployment, configuration
3. **Contribute back**: Share improvements and new algorithms
4. **Stay updated**: Follow releases for new scikit-learn compatibility

---

**Ready to make the switch?** Start with our [Quick Start Guide](./docs/getting-started/quickstart/) or try the [30-second Docker demo](./docker-compose.yml).

**Questions?** Join the discussion on [GitHub](https://github.com/YuminosukeSato/scigo/discussions) or check out our [FAQ](./docs/faq/).

🚀 **Ready, Set, SciGo!**