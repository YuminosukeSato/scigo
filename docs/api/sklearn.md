# Scikit-learn Compatible API Reference

Complete API documentation for scikit-learn compatible models in the `sklearn` package.

## Package Overview

```go
import "github.com/YuminosukeSato/scigo/sklearn"
```

The `sklearn` package provides models with APIs compatible with scikit-learn, enabling seamless model interchange between Python and Go.

## Linear Models

### SGDRegressor

Stochastic Gradient Descent regressor for large-scale learning.

#### Constructor

```go
func NewSGDRegressor(opts ...Option) *SGDRegressor
```

Creates a new SGD regressor.

**Options:**
- `WithLearningRate(rate float64)`: Initial learning rate (default: 0.01)
- `WithAlpha(alpha float64)`: Regularization strength (default: 0.0001)
- `WithMaxIter(iter int)`: Maximum iterations (default: 1000)
- `WithTolerance(tol float64)`: Stopping criterion (default: 1e-3)
- `WithPenalty(penalty string)`: Regularization type ("l1", "l2", "elasticnet")
- `WithEtaSchedule(schedule string)`: Learning rate schedule ("constant", "optimal", "invscaling")

**Example:**
```go
sgd := sklearn.NewSGDRegressor(
    sklearn.WithLearningRate(0.001),
    sklearn.WithAlpha(0.01),
    sklearn.WithPenalty("l2"),
    sklearn.WithMaxIter(5000),
)
```

#### Methods

##### Fit

```go
func (sgd *SGDRegressor) Fit(X, y mat.Matrix) error
```

Trains the model using stochastic gradient descent.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)
- `y`: Target vector (n_samples × 1)

**Returns:**
- `error`: Training error if any

**Example:**
```go
err := sgd.Fit(XTrain, yTrain)
if err != nil {
    log.Fatal(err)
}
```

##### PartialFit

```go
func (sgd *SGDRegressor) PartialFit(X, y mat.Matrix) error
```

Performs incremental learning on a batch of samples.

**Parameters:**
- `X`: Feature batch
- `y`: Target batch

**Returns:**
- `error`: Training error if any

**Example:**
```go
// Online learning
for batch := range dataStream {
    err := sgd.PartialFit(batch.X, batch.Y)
    if err != nil {
        log.Printf("Batch training failed: %v", err)
    }
}
```

##### Predict

```go
func (sgd *SGDRegressor) Predict(X mat.Matrix) (mat.Matrix, error)
```

Makes predictions on new data.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Predictions
- `error`: Prediction error if any

### SGDClassifier

Stochastic Gradient Descent classifier for large-scale classification.

#### Constructor

```go
func NewSGDClassifier(opts ...Option) *SGDClassifier
```

Creates a new SGD classifier.

**Options:**
- `WithLoss(loss string)`: Loss function ("hinge", "log", "modified_huber", "squared_hinge", "perceptron")
- `WithClassWeight(weights map[int]float64)`: Class weights for imbalanced data
- `WithNClasses(n int)`: Number of classes for multi-class
- All options from SGDRegressor

**Example:**
```go
clf := sklearn.NewSGDClassifier(
    sklearn.WithLoss("hinge"),  // SVM
    sklearn.WithAlpha(0.001),
    sklearn.WithMaxIter(1000),
)
```

#### Methods

##### Fit

```go
func (clf *SGDClassifier) Fit(X mat.Matrix, y []int) error
```

Trains the classifier.

**Parameters:**
- `X`: Feature matrix
- `y`: Class labels

**Returns:**
- `error`: Training error if any

##### PredictProba

```go
func (clf *SGDClassifier) PredictProba(X mat.Matrix) (mat.Matrix, error)
```

Predicts class probabilities.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Probability matrix (n_samples × n_classes)
- `error`: Prediction error if any

**Example:**
```go
proba, err := clf.PredictProba(XTest)
if err != nil {
    log.Fatal(err)
}

// Get probability for class 1
for i := 0; i < proba.RawMatrix().Rows; i++ {
    fmt.Printf("P(class=1): %.3f\n", proba.At(i, 1))
}
```

##### DecisionFunction

```go
func (clf *SGDClassifier) DecisionFunction(X mat.Matrix) (mat.Matrix, error)
```

Computes decision function values.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Decision values
- `error`: Computation error if any

### PassiveAggressiveRegressor

Online learning algorithm for regression.

#### Constructor

```go
func NewPassiveAggressiveRegressor(C float64, epsilon float64) *PassiveAggressiveRegressor
```

Creates a new Passive-Aggressive regressor.

**Parameters:**
- `C`: Regularization parameter
- `epsilon`: Insensitivity parameter

**Returns:**
- `*PassiveAggressiveRegressor`: New regressor

**Example:**
```go
pa := sklearn.NewPassiveAggressiveRegressor(1.0, 0.1)
```

#### Methods

##### PartialFit

```go
func (pa *PassiveAggressiveRegressor) PartialFit(X, y mat.Matrix) error
```

Performs online update.

**Algorithm:**
```
loss = max(0, |y - pred| - epsilon)
if loss > 0:
    tau = loss / (||x||^2 + 1/(2C))
    w = w + tau * sign(y - pred) * x
```

### PassiveAggressiveClassifier

Online learning algorithm for classification.

#### Constructor

```go
func NewPassiveAggressiveClassifier(C float64, mode string) *PassiveAggressiveClassifier
```

Creates a new Passive-Aggressive classifier.

**Parameters:**
- `C`: Regularization parameter
- `mode`: Algorithm variant ("PA-I" or "PA-II")

**Returns:**
- `*PassiveAggressiveClassifier`: New classifier

## Clustering

### MiniBatchKMeans

Mini-batch variant of KMeans for large datasets.

#### Constructor

```go
func NewMiniBatchKMeans(nClusters int, opts ...Option) *MiniBatchKMeans
```

Creates a new MiniBatchKMeans clusterer.

**Parameters:**
- `nClusters`: Number of clusters
- `opts`: Configuration options

**Options:**
- `WithBatchSize(size int)`: Mini-batch size (default: 100)
- `WithMaxIter(iter int)`: Maximum iterations (default: 100)
- `WithInit(method string)`: Initialization method ("k-means++", "random")
- `WithRandomState(seed int64)`: Random seed for reproducibility

**Example:**
```go
kmeans := sklearn.NewMiniBatchKMeans(5,
    sklearn.WithBatchSize(256),
    sklearn.WithMaxIter(300),
    sklearn.WithInit("k-means++"),
)
```

#### Methods

##### Fit

```go
func (km *MiniBatchKMeans) Fit(X mat.Matrix) error
```

Fits the model to data.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `error`: Fitting error if any

##### PartialFit

```go
func (km *MiniBatchKMeans) PartialFit(X mat.Matrix) error
```

Performs incremental clustering.

**Parameters:**
- `X`: Feature batch

**Returns:**
- `error`: Update error if any

**Example:**
```go
// Stream processing
for batch := range dataStream {
    err := kmeans.PartialFit(batch)
    if err != nil {
        log.Printf("Batch update failed: %v", err)
    }
}
```

##### Predict

```go
func (km *MiniBatchKMeans) Predict(X mat.Matrix) ([]int, error)
```

Predicts cluster labels.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `[]int`: Cluster assignments
- `error`: Prediction error if any

##### Transform

```go
func (km *MiniBatchKMeans) Transform(X mat.Matrix) (mat.Matrix, error)
```

Transforms X to cluster-distance space.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Distance to each cluster center
- `error`: Transformation error if any

##### ClusterCenters

```go
func (km *MiniBatchKMeans) ClusterCenters() mat.Matrix
```

Returns cluster centers.

**Returns:**
- `mat.Matrix`: Centers matrix (n_clusters × n_features)

##### Inertia

```go
func (km *MiniBatchKMeans) Inertia() float64
```

Returns sum of squared distances to nearest cluster.

**Returns:**
- `float64`: Inertia value

## Naive Bayes

### MultinomialNB

Naive Bayes classifier for multinomial models.

#### Constructor

```go
func NewMultinomialNB(alpha float64) *MultinomialNB
```

Creates a new Multinomial Naive Bayes classifier.

**Parameters:**
- `alpha`: Smoothing parameter (0 for no smoothing)

**Returns:**
- `*MultinomialNB`: New classifier

**Example:**
```go
nb := sklearn.NewMultinomialNB(1.0)  // Laplace smoothing
```

#### Methods

##### Fit

```go
func (nb *MultinomialNB) Fit(X mat.Matrix, y []int) error
```

Trains the classifier.

**Parameters:**
- `X`: Feature matrix (typically count features)
- `y`: Class labels

**Returns:**
- `error`: Training error if any

##### PartialFit

```go
func (nb *MultinomialNB) PartialFit(X mat.Matrix, y []int, classes []int) error
```

Incrementally trains the classifier.

**Parameters:**
- `X`: Feature batch
- `y`: Label batch
- `classes`: All possible classes (required on first call)

**Returns:**
- `error`: Training error if any

##### PredictLogProba

```go
func (nb *MultinomialNB) PredictLogProba(X mat.Matrix) (mat.Matrix, error)
```

Returns log-probability estimates.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Log probabilities
- `error`: Prediction error if any

### GaussianNB

Gaussian Naive Bayes for continuous features.

#### Constructor

```go
func NewGaussianNB() *GaussianNB
```

Creates a new Gaussian Naive Bayes classifier.

**Returns:**
- `*GaussianNB`: New classifier

**Example:**
```go
gnb := sklearn.NewGaussianNB()
```

#### Methods

##### Fit

```go
func (gnb *GaussianNB) Fit(X mat.Matrix, y []int) error
```

Trains by computing mean and variance of features.

**Parameters:**
- `X`: Feature matrix
- `y`: Class labels

**Returns:**
- `error`: Training error if any

### BernoulliNB

Naive Bayes for binary/boolean features.

#### Constructor

```go
func NewBernoulliNB(alpha float64, binarize float64) *BernoulliNB
```

Creates a new Bernoulli Naive Bayes classifier.

**Parameters:**
- `alpha`: Smoothing parameter
- `binarize`: Threshold for binarizing features

**Returns:**
- `*BernoulliNB`: New classifier

**Example:**
```go
bnb := sklearn.NewBernoulliNB(1.0, 0.5)
```

## Pipeline

### Pipeline

Sequentially applies a list of transforms and a final estimator.

#### Constructor

```go
func NewPipeline(steps ...Step) *Pipeline
```

Creates a new pipeline.

**Parameters:**
- `steps`: Pipeline steps

**Returns:**
- `*Pipeline`: New pipeline

**Example:**
```go
pipeline := sklearn.NewPipeline(
    sklearn.Step{"scaler", preprocessing.NewStandardScaler()},
    sklearn.Step{"pca", decomposition.NewPCA(10)},
    sklearn.Step{"classifier", sklearn.NewSGDClassifier()},
)
```

#### Methods

##### Fit

```go
func (p *Pipeline) Fit(X mat.Matrix, y interface{}) error
```

Fits all transforms and final estimator.

**Parameters:**
- `X`: Feature matrix
- `y`: Targets (can be nil for unsupervised)

**Returns:**
- `error`: Fitting error if any

##### Predict

```go
func (p *Pipeline) Predict(X mat.Matrix) (interface{}, error)
```

Applies transforms and final prediction.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `interface{}`: Predictions from final estimator
- `error`: Prediction error if any

##### Transform

```go
func (p *Pipeline) Transform(X mat.Matrix) (mat.Matrix, error)
```

Applies transforms (all steps except final).

**Parameters:**
- `X`: Feature matrix

**Returns:**
- `mat.Matrix`: Transformed features
- `error`: Transformation error if any

### FeatureUnion

Concatenates results of multiple transformer objects.

#### Constructor

```go
func NewFeatureUnion(transformers ...NamedTransformer) *FeatureUnion
```

Creates a new feature union.

**Parameters:**
- `transformers`: Named transformers

**Returns:**
- `*FeatureUnion`: New feature union

**Example:**
```go
union := sklearn.NewFeatureUnion(
    sklearn.NamedTransformer{"pca", decomposition.NewPCA(5)},
    sklearn.NamedTransformer{"select", feature_selection.NewSelectKBest(10)},
)
```

## Model Persistence

### SKLearn Format Import/Export

#### ImportFromSKLearn

```go
func ImportFromSKLearn(filename string) (Model, error)
```

Imports a model from scikit-learn JSON format.

**Parameters:**
- `filename`: Path to JSON file

**Returns:**
- `Model`: Imported model
- `error`: Import error if any

**Example:**
```go
model, err := sklearn.ImportFromSKLearn("python_model.json")
if err != nil {
    log.Fatal(err)
}

// Use the model
predictions, _ := model.Predict(X)
```

#### ExportToSKLearn

```go
func ExportToSKLearn(model Model, filename string) error
```

Exports a model to scikit-learn compatible format.

**Parameters:**
- `model`: Model to export
- `filename`: Output path

**Returns:**
- `error`: Export error if any

**Example:**
```go
err := sklearn.ExportToSKLearn(model, "go_model.json")
if err != nil {
    log.Fatal(err)
}
```

### Model Format Specification

```json
{
  "model_type": "SGDClassifier",
  "sklearn_version": "1.0.0",
  "scigo_version": "0.3.0",
  "parameters": {
    "coef_": [[0.1, 0.2, 0.3]],
    "intercept_": [0.5],
    "classes_": [0, 1],
    "loss": "hinge",
    "penalty": "l2",
    "alpha": 0.0001,
    "max_iter": 1000
  },
  "metadata": {
    "n_features": 3,
    "n_classes": 2,
    "n_iter_": 150
  }
}
```

## Utilities

### Cross-Validation

```go
func CrossValidate(estimator Estimator, X mat.Matrix, y interface{}, cv int) (*CVResults, error)
```

Performs cross-validation.

**Parameters:**
- `estimator`: Model to evaluate
- `X`: Feature matrix
- `y`: Targets
- `cv`: Number of folds

**Returns:**
- `*CVResults`: Cross-validation results
- `error`: Evaluation error if any

**Example:**
```go
results, err := sklearn.CrossValidate(model, X, y, 5)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Mean Score: %.3f (+/- %.3f)\n", 
    results.MeanScore(), results.StdScore())
```

### GridSearchCV

```go
func GridSearchCV(estimator Estimator, paramGrid map[string][]interface{}, cv int) *GridSearchCV
```

Exhaustive search over parameter grid.

**Parameters:**
- `estimator`: Base estimator
- `paramGrid`: Parameter combinations
- `cv`: Cross-validation folds

**Returns:**
- `*GridSearchCV`: Grid search object

**Example:**
```go
paramGrid := map[string][]interface{}{
    "alpha": {0.001, 0.01, 0.1},
    "max_iter": {100, 500, 1000},
    "penalty": {"l1", "l2"},
}

gs := sklearn.GridSearchCV(sklearn.NewSGDClassifier(), paramGrid, 5)
err := gs.Fit(X, y)

fmt.Printf("Best params: %v\n", gs.BestParams())
fmt.Printf("Best score: %.3f\n", gs.BestScore())
```

## Complete Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn"
    "github.com/YuminosukeSato/scigo/preprocessing"
    "github.com/YuminosukeSato/scigo/metrics"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Load data
    X, y := loadData()
    
    // Create pipeline
    pipeline := sklearn.NewPipeline(
        sklearn.Step{"scaler", preprocessing.NewStandardScaler()},
        sklearn.Step{"classifier", sklearn.NewSGDClassifier(
            sklearn.WithLoss("log"),  // Logistic regression
            sklearn.WithAlpha(0.001),
            sklearn.WithMaxIter(1000),
        )},
    )
    
    // Cross-validation
    results, err := sklearn.CrossValidate(pipeline, X, y, 5)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("CV Accuracy: %.3f (+/- %.3f)\n",
        results.MeanScore(), results.StdScore())
    
    // Train final model
    err = pipeline.Fit(X, y)
    if err != nil {
        log.Fatal(err)
    }
    
    // Export for Python
    err = sklearn.ExportToSKLearn(pipeline, "model.json")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Model exported successfully!")
}
```

## Performance Comparison

| Model | SciGo | scikit-learn | Speedup |
|-------|-------|--------------|----------|
| SGDClassifier | 0.15s | 0.22s | 1.5x |
| MiniBatchKMeans | 0.8s | 1.2s | 1.5x |
| MultinomialNB | 0.05s | 0.08s | 1.6x |
| PassiveAggressive | 0.12s | 0.18s | 1.5x |

*Benchmark on 100K samples, 100 features*

## Next Steps

- Explore [Core API](./core.md)
- Learn about [Model Persistence](../guides/model-persistence.md)
- See [SKLearn Compatibility Guide](../guides/sklearn-compatibility.md)
- Read [Examples](../../examples/sklearn/)