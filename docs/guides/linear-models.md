# Linear Models Guide

Comprehensive guide to using linear models in SciGo for regression and classification tasks.

## Overview

Linear models are fundamental machine learning algorithms that assume a linear relationship between input features and target values. SciGo provides efficient implementations optimized for production use.

## Model Types

### Regression Models

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| LinearRegression | General regression | Fast, interpretable | Assumes linearity |
| SGDRegressor | Large-scale learning | Memory efficient, online learning | Requires tuning |
| Ridge | Regression with L2 regularization | Handles multicollinearity | May underfit |
| Lasso | Feature selection | Automatic feature selection | Can be unstable |
| ElasticNet | Combined regularization | Balanced approach | More hyperparameters |

### Classification Models

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| LogisticRegression | Binary/multiclass | Probabilistic output | Linear boundaries |
| SGDClassifier | Large-scale classification | Streaming capable | Sensitive to scaling |
| PassiveAggressive | Online learning | Aggressive updates | Can overfit |

## Linear Regression

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "github.com/YuminosukeSato/scigo/metrics"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Generate sample data
    X := mat.NewDense(100, 3, nil) // 100 samples, 3 features
    y := mat.NewVecDense(100, nil) // 100 targets
    
    // Fill with your data...
    
    // Create and train model
    lr := linear.NewLinearRegression()
    
    err := lr.Fit(X, y)
    if err != nil {
        log.Fatal("Training failed:", err)
    }
    
    // Make predictions
    predictions, err := lr.Predict(X)
    if err != nil {
        log.Fatal("Prediction failed:", err)
    }
    
    // Evaluate
    r2, _ := metrics.R2Score(y, predictions.(*mat.Dense).ColView(0))
    fmt.Printf("R² score: %.4f\n", r2)
    
    // Get coefficients
    weights := lr.GetWeights()
    intercept := lr.GetIntercept()
    
    fmt.Printf("Equation: y = %.2f", intercept)
    for i, w := range weights {
        fmt.Printf(" + %.2f*x%d", w, i+1)
    }
    fmt.Println()
}
```

### Mathematical Foundation

Linear regression finds parameters that minimize the residual sum of squares:

```
min ||Xw - y||²
```

The closed-form solution using normal equations:

```
w = (X'X)⁻¹X'y
```

### When to Use Linear Regression

**Good for:**
- Continuous target variables
- Linear relationships
- Interpretable models
- Quick baseline models

**Not suitable for:**
- Non-linear relationships
- Categorical targets
- High-dimensional data (p > n)

## SGD Models

### SGDRegressor

```go
package main

import (
    "github.com/YuminosukeSato/scigo/sklearn/linear_model"
    "github.com/YuminosukeSato/scigo/preprocessing"
)

func main() {
    // Create SGD regressor with options
    sgd := linear_model.NewSGDRegressor(
        linear_model.WithLearningRate(0.01),
        linear_model.WithAlpha(0.0001),      // L2 regularization
        linear_model.WithMaxIter(1000),
        linear_model.WithTolerance(1e-3),
        linear_model.WithEtaSchedule("invscaling"),
    )
    
    // Scale features (important for SGD)
    scaler := preprocessing.NewStandardScaler()
    XScaled, _ := scaler.FitTransform(X)
    
    // Train model
    err := sgd.Fit(XScaled, y)
    if err != nil {
        log.Fatal(err)
    }
    
    // Online learning with partial_fit
    for batch := range dataStream {
        batchScaled, _ := scaler.Transform(batch.X)
        sgd.PartialFit(batchScaled, batch.Y)
    }
}
```

### Learning Rate Schedules

```go
// Constant learning rate
sgd := NewSGDRegressor(WithEtaSchedule("constant"))
// eta = eta0

// Optimal learning rate (default)
sgd := NewSGDRegressor(WithEtaSchedule("optimal"))
// eta = 1.0 / (alpha * (t + t0))

// Inverse scaling
sgd := NewSGDRegressor(WithEtaSchedule("invscaling"))
// eta = eta0 / pow(t, power_t)

// Adaptive (per-feature)
sgd := NewSGDRegressor(WithEtaSchedule("adaptive"))
// eta adapts based on gradient history
```

### SGDClassifier

```go
func trainClassifier() {
    // Create classifier with different loss functions
    
    // SVM (hinge loss)
    svm := linear_model.NewSGDClassifier(
        linear_model.WithLoss("hinge"),
        linear_model.WithAlpha(0.001),
    )
    
    // Logistic Regression (log loss)
    logistic := linear_model.NewSGDClassifier(
        linear_model.WithLoss("log"),
        linear_model.WithAlpha(0.001),
    )
    
    // Perceptron
    perceptron := linear_model.NewSGDClassifier(
        linear_model.WithLoss("perceptron"),
        linear_model.WithAlpha(0),  // No regularization
    )
    
    // Train
    svm.Fit(X, y)
    
    // Get probabilities (only for log loss)
    proba, _ := logistic.PredictProba(XTest)
    
    // Get decision function
    decision, _ := svm.DecisionFunction(XTest)
}
```

## Regularized Models

### Ridge Regression (L2)

```go
type RidgeRegression struct {
    *LinearRegression
    alpha float64
}

func NewRidgeRegression(alpha float64) *RidgeRegression {
    return &RidgeRegression{
        LinearRegression: NewLinearRegression(),
        alpha: alpha,
    }
}

func (r *RidgeRegression) Fit(X, y mat.Matrix) error {
    // Minimize: ||Xw - y||² + alpha * ||w||²
    
    rows, cols := X.Dims()
    
    // Add L2 penalty to normal equations
    // w = (X'X + alpha*I)⁻¹X'y
    
    var XtX mat.Dense
    XtX.Mul(X.T(), X)
    
    // Add regularization
    for i := 0; i < cols; i++ {
        XtX.Set(i, i, XtX.At(i, i) + r.alpha)
    }
    
    // Solve for weights
    var Xty mat.VecDense
    Xty.MulVec(X.T(), y)
    
    r.weights = mat.NewVecDense(cols, nil)
    r.weights.SolveVec(&XtX, &Xty)
    
    return nil
}
```

### Lasso Regression (L1)

```go
// Lasso uses coordinate descent
func LassoCoordinateDescent(X, y mat.Matrix, alpha float64, maxIter int) *mat.VecDense {
    rows, cols := X.Dims()
    weights := mat.NewVecDense(cols, nil)
    
    for iter := 0; iter < maxIter; iter++ {
        for j := 0; j < cols; j++ {
            // Compute residual without feature j
            residual := computeResidual(X, y, weights, j)
            
            // Soft thresholding
            rho := dotProduct(X.ColView(j), residual)
            weights.SetVec(j, softThreshold(rho, alpha))
        }
    }
    
    return weights
}

func softThreshold(x, lambda float64) float64 {
    if x > lambda {
        return x - lambda
    } else if x < -lambda {
        return x + lambda
    }
    return 0
}
```

### ElasticNet (L1 + L2)

```go
type ElasticNet struct {
    alpha float64  // Total regularization
    l1Ratio float64  // Balance between L1 and L2
}

// Minimize: ||Xw - y||² + alpha * (l1_ratio * ||w||₁ + (1-l1_ratio) * ||w||²/2)
```

### Choosing Regularization

| Method | When to Use | Effect |
|--------|-------------|--------|
| None | Low-dimensional, no multicollinearity | Maximum likelihood |
| Ridge (L2) | Multicollinearity, all features relevant | Shrinks coefficients |
| Lasso (L1) | Feature selection needed | Sets some coefficients to zero |
| ElasticNet | Many correlated features | Selects groups of features |

## Feature Engineering

### Polynomial Features

```go
func createPolynomialFeatures() {
    // Original features: [x1, x2]
    poly := preprocessing.NewPolynomialFeatures(2, true)
    
    // Transforms to: [1, x1, x2, x1², x1*x2, x2²]
    XPoly, _ := poly.Transform(X)
    
    // Fit linear model on polynomial features
    lr := linear.NewLinearRegression()
    lr.Fit(XPoly, y)
}
```

### Interaction Terms

```go
func addInteractionTerms(X mat.Matrix) mat.Matrix {
    rows, cols := X.Dims()
    
    // Calculate number of interaction terms
    nInteractions := cols * (cols - 1) / 2
    
    // Create new matrix with interactions
    result := mat.NewDense(rows, cols+nInteractions, nil)
    
    // Copy original features
    result.Copy(X)
    
    // Add interactions
    idx := cols
    for i := 0; i < cols; i++ {
        for j := i + 1; j < cols; j++ {
            for row := 0; row < rows; row++ {
                interaction := X.At(row, i) * X.At(row, j)
                result.Set(row, idx, interaction)
            }
            idx++
        }
    }
    
    return result
}
```

## Multi-Output Regression

```go
type MultiOutputRegressor struct {
    models []Model
}

func NewMultiOutputRegressor(baseModel func() Model, nOutputs int) *MultiOutputRegressor {
    models := make([]Model, nOutputs)
    for i := range models {
        models[i] = baseModel()
    }
    return &MultiOutputRegressor{models: models}
}

func (m *MultiOutputRegressor) Fit(X, Y mat.Matrix) error {
    _, nOutputs := Y.Dims()
    
    // Train separate model for each output
    for i := 0; i < nOutputs; i++ {
        yCol := Y.(*mat.Dense).ColView(i)
        if err := m.models[i].Fit(X, yCol); err != nil {
            return fmt.Errorf("failed to fit output %d: %w", i, err)
        }
    }
    
    return nil
}

func (m *MultiOutputRegressor) Predict(X mat.Matrix) (mat.Matrix, error) {
    rows, _ := X.Dims()
    predictions := mat.NewDense(rows, len(m.models), nil)
    
    for i, model := range m.models {
        pred, err := model.Predict(X)
        if err != nil {
            return nil, err
        }
        predictions.SetCol(i, mat.Col(nil, 0, pred))
    }
    
    return predictions, nil
}
```

## Handling Categorical Variables

```go
func preprocessCategorical() {
    // One-hot encoding
    encoder := preprocessing.NewOneHotEncoder()
    encoder.Fit(XCategorical)
    XEncoded, _ := encoder.Transform(XCategorical)
    
    // Combine with numerical features
    XCombined := concatenate(XNumerical, XEncoded)
    
    // Train model
    lr := linear.NewLinearRegression()
    lr.Fit(XCombined, y)
}
```

## Cross-Validation

```go
func crossValidateLinearModel() {
    // K-fold cross-validation
    k := 5
    folds := createKFolds(X, y, k)
    
    scores := make([]float64, k)
    
    for i, fold := range folds {
        // Train on k-1 folds
        XTrain, yTrain := fold.Train()
        XVal, yVal := fold.Validation()
        
        // Train model
        lr := linear.NewLinearRegression()
        lr.Fit(XTrain, yTrain)
        
        // Evaluate
        pred, _ := lr.Predict(XVal)
        scores[i], _ = metrics.R2Score(yVal, pred.(*mat.Dense).ColView(0))
    }
    
    meanScore := mean(scores)
    stdScore := std(scores)
    
    fmt.Printf("CV R² Score: %.4f (+/- %.4f)\n", meanScore, stdScore)
}
```

## Model Selection

### Grid Search

```go
func gridSearchRegularization() {
    alphas := []float64{0.001, 0.01, 0.1, 1.0, 10.0}
    bestAlpha := 0.0
    bestScore := -math.Inf(1)
    
    for _, alpha := range alphas {
        model := NewRidgeRegression(alpha)
        
        // Cross-validation score
        scores := crossValidate(model, X, y, 5)
        meanScore := mean(scores)
        
        if meanScore > bestScore {
            bestScore = meanScore
            bestAlpha = alpha
        }
        
        fmt.Printf("Alpha: %.3f, Score: %.4f\n", alpha, meanScore)
    }
    
    fmt.Printf("Best alpha: %.3f\n", bestAlpha)
}
```

### Learning Curves

```go
func plotLearningCurves(model Model, X, y mat.Matrix) {
    trainSizes := []float64{0.1, 0.25, 0.5, 0.75, 1.0}
    trainScores := make([]float64, len(trainSizes))
    valScores := make([]float64, len(trainSizes))
    
    for i, size := range trainSizes {
        // Sample data
        n := int(float64(X.(*mat.Dense).RawMatrix().Rows) * size)
        XSample := X.(*mat.Dense).Slice(0, n, 0, X.(*mat.Dense).RawMatrix().Cols)
        ySample := y.(*mat.VecDense).SliceVec(0, n)
        
        // Train and evaluate
        model.Fit(XSample, ySample)
        
        trainPred, _ := model.Predict(XSample)
        trainScores[i], _ = metrics.R2Score(ySample, trainPred.(*mat.Dense).ColView(0))
        
        valPred, _ := model.Predict(XVal)
        valScores[i], _ = metrics.R2Score(yVal, valPred.(*mat.Dense).ColView(0))
    }
    
    // Plot or analyze curves
    for i, size := range trainSizes {
        fmt.Printf("Size: %.0f%%, Train: %.3f, Val: %.3f\n",
            size*100, trainScores[i], valScores[i])
    }
}
```

## Performance Tips

### 1. Feature Scaling

```go
// Always scale features for SGD
scaler := preprocessing.NewStandardScaler()
XScaled, _ := scaler.FitTransform(X)
```

### 2. Sparse Data

```go
// Use sparse matrix operations
type SparseLinearModel struct {
    weights map[int]float64  // Only non-zero weights
}

func (m *SparseLinearModel) Predict(X SparseMatrix) []float64 {
    predictions := make([]float64, X.Rows())
    
    for i, j, value := range X.NonZero() {
        if weight, exists := m.weights[j]; exists {
            predictions[i] += weight * value
        }
    }
    
    return predictions
}
```

### 3. Batch Processing

```go
// Process large datasets in batches
func batchFit(model Model, X, y mat.Matrix, batchSize int) error {
    rows, _ := X.Dims()
    
    for start := 0; start < rows; start += batchSize {
        end := min(start+batchSize, rows)
        
        XBatch := X.(*mat.Dense).Slice(start, end, 0, X.(*mat.Dense).RawMatrix().Cols)
        yBatch := y.(*mat.VecDense).SliceVec(start, end)
        
        if partialFitter, ok := model.(PartialFitter); ok {
            partialFitter.PartialFit(XBatch, yBatch)
        }
    }
    
    return nil
}
```

## Common Pitfalls

### 1. Multicollinearity

```go
// Check VIF (Variance Inflation Factor)
func calculateVIF(X mat.Matrix) []float64 {
    _, cols := X.Dims()
    vif := make([]float64, cols)
    
    for i := 0; i < cols; i++ {
        // Regress column i on other columns
        y := X.(*mat.Dense).ColView(i)
        XOther := removeColumn(X, i)
        
        lr := linear.NewLinearRegression()
        lr.Fit(XOther, y)
        
        r2 := lr.Score(XOther, y)
        vif[i] = 1.0 / (1.0 - r2)
    }
    
    return vif
}

// VIF > 10 indicates multicollinearity
```

### 2. Outliers

```go
// Use robust regression or remove outliers
func removeOutliers(X, y mat.Matrix, threshold float64) (mat.Matrix, mat.Matrix) {
    // Fit initial model
    lr := linear.NewLinearRegression()
    lr.Fit(X, y)
    
    // Calculate residuals
    pred, _ := lr.Predict(X)
    residuals := computeResiduals(y, pred)
    
    // Remove points with large residuals
    mask := make([]bool, len(residuals))
    for i, r := range residuals {
        mask[i] = math.Abs(r) < threshold
    }
    
    return filterByMask(X, mask), filterByMask(y, mask)
}
```

### 3. Non-linearity

```go
// Check residual plots
func checkLinearity(model Model, X, y mat.Matrix) {
    pred, _ := model.Predict(X)
    residuals := computeResiduals(y, pred)
    
    // Plot residuals vs predicted
    // Should show no pattern for linear relationship
    
    // Calculate correlation of residuals with predicted
    correlation := pearsonCorrelation(pred, residuals)
    
    if math.Abs(correlation) > 0.1 {
        fmt.Println("Warning: Non-linear relationship detected")
    }
}
```

## Best Practices

1. **Start Simple**: Begin with linear regression as baseline
2. **Scale Features**: Essential for SGD and regularized models
3. **Check Assumptions**: Verify linearity, homoscedasticity
4. **Cross-Validate**: Always use CV for model selection
5. **Interpret Coefficients**: Understand feature importance
6. **Monitor Convergence**: Check loss curves for SGD
7. **Handle Missing Data**: Impute before training
8. **Feature Selection**: Use Lasso or statistical tests

## Next Steps

- Explore [Classification Guide](./classification.md)
- Learn about [Model Persistence](./model-persistence.md)
- See [Linear Regression Examples](../../examples/linear_regression/)
- Read [API Reference](../api/linear.md)