# Quick Start Guide

Build your first machine learning model with SciGo in 5 minutes!

## Your First Model

Let's create a simple linear regression model to predict house prices:

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
    // Step 1: Prepare your data
    // Features: [size in sqft, bedrooms, age]
    features := []float64{
        1500, 3, 10,  // House 1
        2000, 4, 5,   // House 2
        1200, 2, 15,  // House 3
        1800, 3, 8,   // House 4
    }
    
    // Target: house prices in $1000s
    prices := []float64{300, 400, 250, 350}
    
    // Create matrices
    X := mat.NewDense(4, 3, features)
    y := mat.NewVecDense(4, prices)
    
    // Step 2: Preprocess the data (optional but recommended)
    scaler := preprocessing.NewStandardScalerDefault()
    if err := scaler.Fit(X); err != nil {
        log.Fatal(err)
    }
    
    XScaled, err := scaler.Transform(X)
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 3: Create and train the model
    model := linear.NewLinearRegression()
    if err := model.Fit(XScaled, y); err != nil {
        log.Fatal(err)
    }
    
    // Step 4: Make predictions
    newHouse := mat.NewDense(1, 3, []float64{1600, 3, 7})
    newHouseScaled, _ := scaler.Transform(newHouse)
    
    prediction, err := model.Predict(newHouseScaled)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Predicted price: $%.2fk\n", prediction.At(0, 0))
    
    // Step 5: Evaluate the model
    predictions, _ := model.Predict(XScaled)
    r2, _ := metrics.R2Score(y, predictions.(*mat.Dense).ColView(0))
    fmt.Printf("Model R² score: %.3f\n", r2)
}
```

## Understanding the Code

### 1. Data Preparation
```go
X := mat.NewDense(4, 3, features)  // Feature matrix (samples × features)
y := mat.NewVecDense(4, prices)    // Target vector
```

### 2. Data Preprocessing
```go
scaler := preprocessing.NewStandardScalerDefault()
scaler.Fit(X)                      // Learn mean and std
XScaled, _ := scaler.Transform(X)  // Normalize features
```

### 3. Model Training
```go
model := linear.NewLinearRegression()
model.Fit(XScaled, y)  // Train on normalized data
```

### 4. Making Predictions
```go
prediction, _ := model.Predict(newHouseScaled)
```

## Common ML Tasks

### Classification Example

```go
package main

import (
    "fmt"
    "github.com/YuminosukeSato/scigo/sklearn/linear_model"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Binary classification: spam detection
    // Features: [word_count, exclamation_marks, caps_ratio]
    features := []float64{
        100, 0, 0.1,  // Normal email
        150, 5, 0.3,  // Spam
        80,  0, 0.05, // Normal
        200, 10, 0.8, // Spam
    }
    
    // Labels: 0 = normal, 1 = spam
    labels := []float64{0, 1, 0, 1}
    
    X := mat.NewDense(4, 3, features)
    y := mat.NewVecDense(4, labels)
    
    // Create and train classifier
    classifier := linear_model.NewSGDClassifier()
    classifier.Fit(X, y)
    
    // Predict new email
    newEmail := mat.NewDense(1, 3, []float64{120, 3, 0.2})
    pred, _ := classifier.Predict(newEmail)
    
    if pred.At(0, 0) > 0.5 {
        fmt.Println("This is likely spam!")
    } else {
        fmt.Println("This email looks normal.")
    }
}
```

### Clustering Example

```go
package main

import (
    "fmt"
    "github.com/YuminosukeSato/scigo/sklearn/cluster"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Customer segmentation data
    // Features: [age, income, spending_score]
    customers := []float64{
        25, 30000, 80,
        30, 40000, 60,
        35, 60000, 40,
        40, 70000, 30,
        22, 25000, 90,
        28, 35000, 75,
    }
    
    X := mat.NewDense(6, 3, customers)
    
    // Perform clustering
    kmeans := cluster.NewMiniBatchKMeans(
        cluster.WithNClusters(2),
        cluster.WithMaxIter(100),
    )
    
    kmeans.Fit(X)
    
    // Get cluster assignments
    labels := kmeans.Predict(X)
    
    fmt.Println("Customer segments:", labels)
}
```

## Model Persistence

### Save a Model

```go
import "github.com/YuminosukeSato/scigo/core/model"

// After training
err := model.SaveModel(lr, "my_model.gob")
if err != nil {
    log.Fatal(err)
}
```

### Load a Model

```go
// Create new model instance
lr := linear.NewLinearRegression()

// Load saved weights
err := model.LoadModel(lr, "my_model.gob")
if err != nil {
    log.Fatal(err)
}

// Use loaded model for predictions
predictions, _ := lr.Predict(X)
```

## Scikit-learn Compatibility

Import models trained in Python:

```go
// Load a scikit-learn model
lr := linear.NewLinearRegression()
err := lr.LoadFromSKLearn("sklearn_model.json")

// Use it just like a native model
predictions, _ := lr.Predict(X)
```

## Streaming/Online Learning

Process data in real-time:

```go
package main

import (
    "context"
    "github.com/YuminosukeSato/scigo/sklearn/linear_model"
    "github.com/YuminosukeSato/scigo/core/model"
)

func main() {
    // Create online learning model
    sgd := linear_model.NewSGDRegressor()
    
    // Create data channel
    dataChan := make(chan *model.Batch)
    
    // Start learning
    ctx := context.Background()
    go sgd.FitStream(ctx, dataChan)
    
    // Stream data
    for batch := range getBatches() {
        dataChan <- batch
    }
}
```

## Best Practices

### 1. Always Preprocess Data
```go
// Normalize features for better convergence
scaler := preprocessing.NewStandardScaler()
XScaled, _ := scaler.FitTransform(X)
```

### 2. Split Data for Validation
```go
// Use 80% for training, 20% for testing
trainSize := int(0.8 * float64(n))
XTrain := X.Slice(0, trainSize, 0, X.Cols())
XTest := X.Slice(trainSize, n, 0, X.Cols())
```

### 3. Handle Errors Properly
```go
if err := model.Fit(X, y); err != nil {
    switch err.(type) {
    case *errors.DimensionError:
        log.Fatal("Data dimensions mismatch")
    case *errors.ConvergenceWarning:
        log.Warn("Model didn't fully converge")
    default:
        log.Fatal(err)
    }
}
```

### 4. Use Appropriate Metrics
```go
// Regression metrics
mse, _ := metrics.MSE(yTrue, yPred)
r2, _ := metrics.R2Score(yTrue, yPred)

// Classification metrics
accuracy := metrics.Accuracy(yTrue, yPred)
precision := metrics.Precision(yTrue, yPred)
```

## Next Steps

Now that you've built your first model:

1. Explore [Basic Concepts](./basic-concepts.md) to understand the fundamentals
2. Learn about different [Model Types](../guides/linear-models.md)
3. Master [Data Preprocessing](../guides/preprocessing.md)
4. Build [Production Systems](../tutorials/production.md)
5. Browse [Complete Examples](../../examples/)

## Common Patterns

### Pipeline Pattern
```go
// Chain preprocessing and model
pipeline := Pipeline{
    Steps: []Step{
        {"scaler", preprocessing.NewStandardScaler()},
        {"model", linear.NewLinearRegression()},
    },
}
pipeline.Fit(X, y)
predictions := pipeline.Predict(XTest)
```

### Cross-Validation Pattern
```go
// Evaluate model performance
scores := CrossValidate(model, X, y, cv=5)
fmt.Printf("Average score: %.3f (+/- %.3f)\n", 
    scores.Mean(), scores.Std())
```

### Grid Search Pattern
```go
// Find best hyperparameters
params := map[string][]interface{}{
    "C": []interface{}{0.1, 1.0, 10.0},
    "max_iter": []interface{}{100, 500, 1000},
}
bestModel := GridSearchCV(model, params, X, y)
```

## Tips for Success

1. **Start Simple**: Begin with linear models before moving to complex algorithms
2. **Visualize Data**: Understand your data distribution before modeling
3. **Monitor Performance**: Track training time and memory usage
4. **Version Models**: Keep track of model versions and parameters
5. **Test Thoroughly**: Always validate on unseen data

## Getting Help

- 📖 Read the [API Reference](../api/)
- 💬 Ask questions in [Discussions](https://github.com/YuminosukeSato/scigo/discussions)
- 🐛 Report issues on [GitHub](https://github.com/YuminosukeSato/scigo/issues)
- 📧 Contact: support@scigo.dev