# Metrics API Reference

Complete API documentation for the `metrics` package.

## Package Overview

```go
import "github.com/YuminosukeSato/scigo/metrics"
```

The `metrics` package provides evaluation metrics for regression and classification models.

## Regression Metrics

### Mean Squared Error (MSE)

```go
func MSE(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the mean squared error between true and predicted values.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: MSE value (lower is better)
- `error`: Calculation error if any

**Formula:**
```
MSE = (1/n) * Σ(y_true[i] - y_pred[i])^2
```

**Example:**
```go
yTrue := mat.NewVecDense(100, trueValues)
yPred := mat.NewVecDense(100, predictions)

mse, err := metrics.MSE(yTrue, yPred)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("MSE: %.4f\n", mse)
```

### Root Mean Squared Error (RMSE)

```go
func RMSE(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the root mean squared error.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: RMSE value (lower is better)
- `error`: Calculation error if any

**Formula:**
```
RMSE = sqrt(MSE) = sqrt((1/n) * Σ(y_true[i] - y_pred[i])^2)
```

**Example:**
```go
rmse, err := metrics.RMSE(yTrue, yPred)
fmt.Printf("RMSE: %.4f\n", rmse)
```

### Mean Absolute Error (MAE)

```go
func MAE(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the mean absolute error.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: MAE value (lower is better)
- `error`: Calculation error if any

**Formula:**
```
MAE = (1/n) * Σ|y_true[i] - y_pred[i]|
```

**Example:**
```go
mae, err := metrics.MAE(yTrue, yPred)
fmt.Printf("MAE: %.4f\n", mae)
```

### R² Score (Coefficient of Determination)

```go
func R2Score(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the R² score.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: R² score (1.0 is perfect, 0.0 is baseline)
- `error`: Calculation error if any

**Formula:**
```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(y_true[i] - y_pred[i])^2
  SS_tot = Σ(y_true[i] - mean(y_true))^2
```

**Interpretation:**
- R² = 1.0: Perfect prediction
- R² = 0.0: Model performs as well as mean baseline
- R² < 0.0: Model performs worse than mean baseline

**Example:**
```go
r2, err := metrics.R2Score(yTrue, yPred)
fmt.Printf("R² Score: %.4f\n", r2)
```

### Mean Absolute Percentage Error (MAPE)

```go
func MAPE(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the mean absolute percentage error.

**Parameters:**
- `yTrue`: True target values (must not contain zeros)
- `yPred`: Predicted values

**Returns:**
- `float64`: MAPE value as percentage
- `error`: Calculation error if any

**Formula:**
```
MAPE = (100/n) * Σ|((y_true[i] - y_pred[i]) / y_true[i])|
```

**Example:**
```go
mape, err := metrics.MAPE(yTrue, yPred)
fmt.Printf("MAPE: %.2f%%\n", mape)
```

### Explained Variance Score

```go
func ExplainedVariance(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the explained variance score.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: Explained variance (1.0 is perfect)
- `error`: Calculation error if any

**Formula:**
```
explained_variance = 1 - Var(y_true - y_pred) / Var(y_true)
```

**Example:**
```go
ev, err := metrics.ExplainedVariance(yTrue, yPred)
fmt.Printf("Explained Variance: %.4f\n", ev)
```

### Max Error

```go
func MaxError(yTrue, yPred mat.Vector) (float64, error)
```

Calculates the maximum residual error.

**Parameters:**
- `yTrue`: True target values
- `yPred`: Predicted values

**Returns:**
- `float64`: Maximum absolute error
- `error`: Calculation error if any

**Formula:**
```
max_error = max(|y_true[i] - y_pred[i]|)
```

**Example:**
```go
maxErr, err := metrics.MaxError(yTrue, yPred)
fmt.Printf("Max Error: %.4f\n", maxErr)
```

## Classification Metrics

### Accuracy

```go
func Accuracy(yTrue, yPred []int) (float64, error)
```

Calculates classification accuracy.

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels

**Returns:**
- `float64`: Accuracy score (0.0 to 1.0)
- `error`: Calculation error if any

**Formula:**
```
accuracy = (# correct predictions) / (# total predictions)
```

**Example:**
```go
yTrue := []int{0, 1, 1, 0, 1}
yPred := []int{0, 1, 0, 0, 1}

acc, err := metrics.Accuracy(yTrue, yPred)
fmt.Printf("Accuracy: %.2f%%\n", acc*100)
```

### Precision

```go
func Precision(yTrue, yPred []int, average string) (float64, error)
```

Calculates precision score.

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels
- `average`: Averaging method ("binary", "macro", "weighted")

**Returns:**
- `float64`: Precision score (0.0 to 1.0)
- `error`: Calculation error if any

**Formula:**
```
precision = TP / (TP + FP)

where:
  TP = True Positives
  FP = False Positives
```

**Example:**
```go
// Binary classification
precision, err := metrics.Precision(yTrue, yPred, "binary")

// Multi-class with macro averaging
precision, err := metrics.Precision(yTrue, yPred, "macro")
```

### Recall

```go
func Recall(yTrue, yPred []int, average string) (float64, error)
```

Calculates recall (sensitivity) score.

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels
- `average`: Averaging method ("binary", "macro", "weighted")

**Returns:**
- `float64`: Recall score (0.0 to 1.0)
- `error`: Calculation error if any

**Formula:**
```
recall = TP / (TP + FN)

where:
  TP = True Positives
  FN = False Negatives
```

**Example:**
```go
recall, err := metrics.Recall(yTrue, yPred, "binary")
fmt.Printf("Recall: %.4f\n", recall)
```

### F1 Score

```go
func F1Score(yTrue, yPred []int, average string) (float64, error)
```

Calculates F1 score (harmonic mean of precision and recall).

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels
- `average`: Averaging method ("binary", "macro", "weighted")

**Returns:**
- `float64`: F1 score (0.0 to 1.0)
- `error`: Calculation error if any

**Formula:**
```
F1 = 2 * (precision * recall) / (precision + recall)
```

**Example:**
```go
f1, err := metrics.F1Score(yTrue, yPred, "macro")
fmt.Printf("F1 Score: %.4f\n", f1)
```

### Confusion Matrix

```go
func ConfusionMatrix(yTrue, yPred []int) (*mat.Dense, error)
```

Computes confusion matrix for classification.

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels

**Returns:**
- `*mat.Dense`: Confusion matrix where C[i,j] is count of true label i predicted as j
- `error`: Calculation error if any

**Example:**
```go
cm, err := metrics.ConfusionMatrix(yTrue, yPred)
if err != nil {
    log.Fatal(err)
}

// Print confusion matrix
rows, cols := cm.Dims()
for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
        fmt.Printf("%3.0f ", cm.At(i, j))
    }
    fmt.Println()
}
```

### Classification Report

```go
func ClassificationReport(yTrue, yPred []int, targetNames []string) (string, error)
```

Generates a text report showing main classification metrics.

**Parameters:**
- `yTrue`: True class labels
- `yPred`: Predicted class labels
- `targetNames`: Optional class names

**Returns:**
- `string`: Formatted report
- `error`: Calculation error if any

**Example:**
```go
report, err := metrics.ClassificationReport(
    yTrue, 
    yPred,
    []string{"class_0", "class_1", "class_2"},
)
fmt.Println(report)

// Output:
//               precision    recall  f1-score   support
// 
//      class_0       0.88      0.91      0.89       100
//      class_1       0.86      0.84      0.85        95
//      class_2       0.90      0.88      0.89       105
// 
//     accuracy                           0.88       300
//    macro avg       0.88      0.88      0.88       300
// weighted avg       0.88      0.88      0.88       300
```

### ROC AUC Score

```go
func ROCAUC(yTrue []int, yScores []float64) (float64, error)
```

Calculates Area Under the ROC Curve.

**Parameters:**
- `yTrue`: True binary labels (0 or 1)
- `yScores`: Predicted probabilities or decision scores

**Returns:**
- `float64`: AUC score (0.5 is random, 1.0 is perfect)
- `error`: Calculation error if any

**Example:**
```go
yTrue := []int{0, 0, 1, 1}
yScores := []float64{0.1, 0.4, 0.35, 0.8}

auc, err := metrics.ROCAUC(yTrue, yScores)
fmt.Printf("ROC AUC: %.4f\n", auc)
```

### ROC Curve

```go
func ROCCurve(yTrue []int, yScores []float64) (fpr, tpr, thresholds []float64, error)
```

Computes Receiver Operating Characteristic curve.

**Parameters:**
- `yTrue`: True binary labels
- `yScores`: Predicted probabilities

**Returns:**
- `fpr`: False positive rates
- `tpr`: True positive rates
- `thresholds`: Decision thresholds
- `error`: Calculation error if any

**Example:**
```go
fpr, tpr, thresholds, err := metrics.ROCCurve(yTrue, yScores)
if err != nil {
    log.Fatal(err)
}

// Plot ROC curve
for i := range fpr {
    fmt.Printf("Threshold: %.2f, FPR: %.3f, TPR: %.3f\n",
        thresholds[i], fpr[i], tpr[i])
}
```

## Clustering Metrics

### Silhouette Score

```go
func SilhouetteScore(X mat.Matrix, labels []int) (float64, error)
```

Calculates mean Silhouette Coefficient.

**Parameters:**
- `X`: Feature matrix
- `labels`: Cluster labels

**Returns:**
- `float64`: Silhouette score (-1 to 1, higher is better)
- `error`: Calculation error if any

**Formula:**
```
silhouette = (b - a) / max(a, b)

where:
  a = mean intra-cluster distance
  b = mean nearest-cluster distance
```

**Example:**
```go
score, err := metrics.SilhouetteScore(X, clusterLabels)
fmt.Printf("Silhouette Score: %.4f\n", score)
```

### Davies-Bouldin Score

```go
func DaviesBouldinScore(X mat.Matrix, labels []int) (float64, error)
```

Calculates Davies-Bouldin clustering evaluation score.

**Parameters:**
- `X`: Feature matrix
- `labels`: Cluster labels

**Returns:**
- `float64`: Davies-Bouldin score (lower is better)
- `error`: Calculation error if any

**Example:**
```go
dbScore, err := metrics.DaviesBouldinScore(X, labels)
fmt.Printf("Davies-Bouldin Score: %.4f\n", dbScore)
```

### Calinski-Harabasz Score

```go
func CalinskiHarabaszScore(X mat.Matrix, labels []int) (float64, error)
```

Calculates Calinski-Harabasz score (Variance Ratio Criterion).

**Parameters:**
- `X`: Feature matrix
- `labels`: Cluster labels

**Returns:**
- `float64`: CH score (higher is better)
- `error`: Calculation error if any

**Example:**
```go
chScore, err := metrics.CalinskiHarabaszScore(X, labels)
fmt.Printf("Calinski-Harabasz Score: %.4f\n", chScore)
```

## Distance Metrics

### Euclidean Distance

```go
func EuclideanDistance(a, b mat.Vector) float64
```

Calculates Euclidean distance between two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:**
- `float64`: Euclidean distance

**Formula:**
```
dist = sqrt(Σ(a[i] - b[i])^2)
```

**Example:**
```go
a := mat.NewVecDense(3, []float64{1, 2, 3})
b := mat.NewVecDense(3, []float64{4, 5, 6})

dist := metrics.EuclideanDistance(a, b)
fmt.Printf("Distance: %.4f\n", dist)
```

### Manhattan Distance

```go
func ManhattanDistance(a, b mat.Vector) float64
```

Calculates Manhattan (L1) distance.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:**
- `float64`: Manhattan distance

**Formula:**
```
dist = Σ|a[i] - b[i]|
```

### Cosine Similarity

```go
func CosineSimilarity(a, b mat.Vector) float64
```

Calculates cosine similarity between vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:**
- `float64`: Cosine similarity (-1 to 1)

**Formula:**
```
similarity = (a · b) / (||a|| * ||b||)
```

**Example:**
```go
sim := metrics.CosineSimilarity(a, b)
fmt.Printf("Cosine Similarity: %.4f\n", sim)
```

## Pairwise Metrics

### Pairwise Distances

```go
func PairwiseDistances(X mat.Matrix, metric string) (*mat.Dense, error)
```

Computes pairwise distances between samples.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)
- `metric`: Distance metric ("euclidean", "manhattan", "cosine")

**Returns:**
- `*mat.Dense`: Distance matrix (n_samples × n_samples)
- `error`: Calculation error if any

**Example:**
```go
distMatrix, err := metrics.PairwiseDistances(X, "euclidean")
if err != nil {
    log.Fatal(err)
}

// distMatrix[i,j] is distance between sample i and j
```

## Custom Metrics

### Creating Custom Metrics

```go
type MetricFunc func(yTrue, yPred mat.Vector) (float64, error)

// Custom metric example: Weighted MSE
func WeightedMSE(weights []float64) MetricFunc {
    return func(yTrue, yPred mat.Vector) (float64, error) {
        if yTrue.Len() != yPred.Len() {
            return 0, fmt.Errorf("dimension mismatch")
        }
        
        var weightedSum float64
        var totalWeight float64
        
        for i := 0; i < yTrue.Len(); i++ {
            diff := yTrue.AtVec(i) - yPred.AtVec(i)
            weightedSum += weights[i] * diff * diff
            totalWeight += weights[i]
        }
        
        return weightedSum / totalWeight, nil
    }
}

// Usage
weights := []float64{1.0, 2.0, 1.5, ...}
wmse := WeightedMSE(weights)
score, err := wmse(yTrue, yPred)
```

### Metric Aggregation

```go
type MetricAggregator struct {
    metrics map[string]MetricFunc
}

func (ma *MetricAggregator) Evaluate(yTrue, yPred mat.Vector) map[string]float64 {
    results := make(map[string]float64)
    
    for name, metric := range ma.metrics {
        score, err := metric(yTrue, yPred)
        if err == nil {
            results[name] = score
        }
    }
    
    return results
}

// Usage
agg := &MetricAggregator{
    metrics: map[string]MetricFunc{
        "mse":  MSE,
        "rmse": RMSE,
        "mae":  MAE,
        "r2":   R2Score,
    },
}

scores := agg.Evaluate(yTrue, yPred)
for name, score := range scores {
    fmt.Printf("%s: %.4f\n", name, score)
}
```

## Cross-Validation Metrics

### Cross-Validated Score

```go
func CrossValScore(model Model, X mat.Matrix, y mat.Vector, cv int, metric MetricFunc) ([]float64, error)
```

Performs cross-validation and returns scores.

**Parameters:**
- `model`: Model to evaluate
- `X`: Feature matrix
- `y`: Target vector
- `cv`: Number of folds
- `metric`: Scoring metric

**Returns:**
- `[]float64`: Score for each fold
- `error`: Evaluation error if any

**Example:**
```go
model := linear.NewLinearRegression()
scores, err := metrics.CrossValScore(model, X, y, 5, metrics.R2Score)

if err != nil {
    log.Fatal(err)
}

meanScore := mean(scores)
stdScore := std(scores)

fmt.Printf("CV Score: %.4f (+/- %.4f)\n", meanScore, stdScore)
```

## Performance Tips

### Batch Evaluation

```go
// Evaluate metrics in batches for large datasets
func BatchEvaluate(yTrue, yPred mat.Vector, batchSize int) (float64, error) {
    n := yTrue.Len()
    var totalMSE float64
    
    for start := 0; start < n; start += batchSize {
        end := min(start+batchSize, n)
        
        batchTrue := yTrue.SliceVec(start, end)
        batchPred := yPred.SliceVec(start, end)
        
        mse, err := MSE(batchTrue, batchPred)
        if err != nil {
            return 0, err
        }
        
        totalMSE += mse * float64(end-start)
    }
    
    return totalMSE / float64(n), nil
}
```

### Parallel Metric Computation

```go
func ParallelMetrics(yTrue, yPred mat.Vector, metrics []MetricFunc) []float64 {
    results := make([]float64, len(metrics))
    var wg sync.WaitGroup
    
    for i, metric := range metrics {
        wg.Add(1)
        go func(idx int, fn MetricFunc) {
            defer wg.Done()
            score, _ := fn(yTrue, yPred)
            results[idx] = score
        }(i, metric)
    }
    
    wg.Wait()
    return results
}
```

## Best Practices

1. **Choose Appropriate Metrics**: Use metrics that align with your problem
2. **Handle Edge Cases**: Check for division by zero, empty inputs
3. **Consider Scale**: Some metrics are sensitive to scale (MSE vs MAPE)
4. **Use Multiple Metrics**: Single metric may not tell the whole story
5. **Cross-Validate**: Always evaluate on held-out data
6. **Report Confidence Intervals**: Include uncertainty estimates

## Next Steps

- Explore [SKLearn API](./sklearn.md)
- Learn about [Core API](./core.md)
- See [Evaluation Examples](../../examples/evaluation/)
- Read [Cross-Validation Guide](../guides/cross-validation.md)