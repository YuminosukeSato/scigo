# LightGBM Complete Guide 🌲

**Complete Python LightGBM compatibility with full training capabilities**

## 🚀 Quick Start

### Basic Training Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create synthetic data
    X := mat.NewDense(1000, 10, nil)
    y := mat.NewDense(1000, 1, nil)
    
    // Generate sample data
    for i := 0; i < 1000; i++ {
        for j := 0; j < 10; j++ {
            X.Set(i, j, rand.Float64())
        }
        // Simple target: sum of first 3 features
        target := X.At(i, 0) + X.At(i, 1) + X.At(i, 2)
        y.Set(i, 0, target)
    }
    
    // Create and configure regressor
    reg := lightgbm.NewLGBMRegressor()
    
    // Use Python parameter names!
    params := map[string]interface{}{
        "n_estimators":      100,
        "learning_rate":     0.1,
        "num_leaves":        31,
        "min_child_samples": 20,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "random_state":      42,
    }
    
    err := reg.SetParams(params)
    if err != nil {
        log.Fatal(err)
    }
    
    // Train the model
    err = reg.Fit(X, y)
    if err != nil {
        log.Fatal(err)
    }
    
    // Make predictions
    predictions, err := reg.Predict(X)
    if err != nil {
        log.Fatal(err)
    }
    
    // Evaluate model
    score := reg.Score(X, y)
    fmt.Printf("R² Score: %.4f\n", score)
    
    fmt.Println("Training completed successfully!")
}
```

## 📊 Cross-Validation

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Load your data
    X, y := loadData() // Your data loading function
    
    // Set up training parameters
    params := lightgbm.TrainingParams{
        NumIterations: 100,
        LearningRate:  0.1,
        NumLeaves:     31,
        MaxDepth:      -1,
        MinDataInLeaf: 20,
        Lambda:        0.1,
        Alpha:         0.1,
        Objective:     "regression",
        Metric:        "rmse",
        Seed:          42,
    }
    
    // Create K-Fold splitter
    splitter := lightgbm.NewKFold(5, true, 42)
    
    // Run cross-validation
    result, err := lightgbm.CrossValidate(
        params,           // Training parameters
        X, y,            // Data
        splitter,        // Cross-validation splitter
        "rmse",          // Evaluation metric
        10,              // Early stopping rounds
        true,            // Verbose output
    )
    
    if err != nil {
        log.Fatal(err)
    }
    
    // Print results
    fmt.Printf("Mean CV Score: %.4f (±%.4f)\n", 
        result.GetMeanScore(), 
        result.GetStdScore())
    fmt.Printf("Best iteration: %d\n", result.BestIteration)
    
    // Access fold-level results
    for i, score := range result.TestScores {
        fmt.Printf("Fold %d: %.4f\n", i+1, score)
    }
}
```

## 📈 Early Stopping with Validation

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Split data into train/validation
    XTrain, yTrain, XVal, yVal := trainTestSplit(X, y, 0.2, 42)
    
    // Create trainer
    params := lightgbm.TrainingParams{
        NumIterations: 1000,  // Will stop early
        LearningRate:  0.1,
        NumLeaves:     31,
        Objective:     "regression",
        Metric:        "rmse",
        Verbosity:     1,
    }
    
    trainer := lightgbm.NewTrainer(params)
    
    // Set up validation data for early stopping
    valData := &lightgbm.ValidationData{
        X: XVal,
        y: yVal,
    }
    
    // Train with early stopping
    err := trainer.FitWithValidation(XTrain, yTrain, valData)
    if err != nil {
        log.Fatal(err)
    }
    
    // Get the trained model
    model := trainer.GetModel()
    
    // Make predictions
    predictor := lightgbm.NewPredictor(model)
    predictions, _ := predictor.Predict(XVal)
    
    fmt.Println("Early stopping training completed!")
}
```

## 🎛️ Advanced Callbacks

```go
package main

import (
    "fmt"
    "log"
    "time"
    
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create trainer
    params := lightgbm.TrainingParams{
        NumIterations: 1000,
        LearningRate:  0.1,
        NumLeaves:     31,
        Objective:     "regression",
        Verbosity:     -1, // Quiet - callbacks will handle logging
    }
    
    trainer := lightgbm.NewTrainer(params)
    
    // Set up callbacks
    callbacks := []lightgbm.Callback{
        // Print evaluation every 10 iterations
        lightgbm.PrintEvaluation(10),
        
        // Early stopping after 50 rounds without improvement
        lightgbm.EarlyStoppingCallback(50, "training_loss", true),
        
        // Time limit: stop after 5 minutes
        lightgbm.TimeLimit(5 * time.Minute),
        
        // Learning rate schedule
        lightgbm.LearningRateSchedule(map[int]float64{
            100: 0.05,  // Reduce LR to 0.05 at iteration 100
            200: 0.01,  // Further reduce to 0.01 at iteration 200
        }),
        
        // Save model checkpoints
        lightgbm.ModelCheckpoint("model_checkpoint_%d.json", 100),
    }
    
    // Set callbacks
    trainer.WithCallbacks(callbacks...)
    
    // Train with callbacks
    err := trainer.Fit(X, y)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Training with callbacks completed!")
}
```

## 🔄 Loading Python Models

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Option 1: Load from LightGBM text file
    model, err := lightgbm.LoadFromFile("python_model.txt")
    if err != nil {
        log.Fatal(err)
    }
    
    // Option 2: Load from JSON file
    jsonModel, err := lightgbm.LoadFromJSONFile("python_model.json")
    if err != nil {
        log.Fatal(err)
    }
    
    // Option 3: Load from string
    modelString := `tree
version=v3
num_class=1
...` // Your model string
    
    stringModel, err := lightgbm.LoadFromString(modelString)
    if err != nil {
        log.Fatal(err)
    }
    
    // Use any loaded model for prediction
    predictor := lightgbm.NewPredictor(model)
    
    // Your test data
    XTest := mat.NewDense(100, 10, testData)
    
    // Make predictions - identical to Python!
    predictions, err := predictor.Predict(XTest)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Predictions shape: %d×%d\n", predictions.RawMatrix().Rows, predictions.RawMatrix().Cols)
    fmt.Println("Python model loaded and predictions made successfully!")
}
```

## 🎯 Parameter Mapping

All Python LightGBM parameters are supported with full alias compatibility:

```go
// Python parameter names work directly!
params := map[string]interface{}{
    // Core parameters
    "n_estimators":      100,           // or "num_iterations", "num_tree", etc.
    "learning_rate":     0.1,           // or "shrinkage_rate", "eta"
    "num_leaves":        31,            // or "num_leaf", "max_leaves"
    "max_depth":         -1,
    "min_child_samples": 20,            // or "min_data_in_leaf", "min_data"
    
    // Regularization
    "reg_alpha":         0.1,           // or "lambda_l1", "l1_regularization"
    "reg_lambda":        0.1,           // or "lambda_l2", "l2_regularization"
    
    // Sampling
    "subsample":         0.8,           // or "bagging_fraction"
    "colsample_bytree":  0.8,           // or "feature_fraction"
    "subsample_freq":    1,             // or "bagging_freq"
    
    // Other parameters
    "random_state":      42,            // or "seed"
    "n_jobs":           -1,             // or "num_threads"
    "objective":        "regression",   // Full objective support
    "boosting_type":    "gbdt",         // or "dart", "goss", "rf"
    
    // Advanced parameters
    "importance_type":   "gain",
    "deterministic":     true,
    "verbosity":        1,
    "early_stopping_rounds": 10,
}
```

## 📊 Available Objectives

### Regression Objectives
- `regression` / `regression_l2` / `l2` / `mse` - L2 loss (default)
- `regression_l1` / `l1` / `mae` - L1 loss
- `huber` - Huber loss (robust to outliers)
- `fair` - Fair loss
- `poisson` - Poisson regression
- `quantile` - Quantile regression

### Classification Objectives
- `binary` / `binary_logloss` - Binary classification
- `multiclass` / `softmax` - Multiclass classification

Example usage:
```go
// L1 regression (more robust to outliers)
reg := lightgbm.NewLGBMRegressor()
reg.SetParams(map[string]interface{}{
    "objective": "regression_l1",  // or "l1", "mae"
    "n_estimators": 100,
})

// Quantile regression (predict specific percentiles)
reg.SetParams(map[string]interface{}{
    "objective": "quantile",
    "quantile_alpha": 0.9,  // 90th percentile
})

// Binary classification
clf := lightgbm.NewLGBMClassifier()
clf.SetParams(map[string]interface{}{
    "objective": "binary",  // or "binary_logloss"
})
```

## 🔧 Custom Evaluation Metrics

```go
// Available metrics for evaluation
metrics := []string{
    // Regression metrics
    "rmse", "mae", "mape", "r2_score", 
    "explained_variance", "mse",
    
    // Classification metrics  
    "accuracy", "precision", "recall", 
    "f1_score", "auc", "logloss",
}

// Use in cross-validation
result, _ := lightgbm.CrossValidate(
    params, X, y, splitter,
    "rmse",  // Primary metric
    10, true,
)

// Or evaluate manually
predictions, _ := model.Predict(XTest)
rmse, _ := lightgbm.RMSE(yTrue, predictions)
mae, _ := lightgbm.MAE(yTrue, predictions)
r2, _ := lightgbm.R2Score(yTrue, predictions)
```

## 🚀 Performance Tips

1. **Use appropriate number of threads**:
   ```go
   reg.SetParams(map[string]interface{}{
       "n_jobs": -1,  // Use all available cores
       // or specify exact number
       "n_jobs": 4,   // Use 4 threads
   })
   ```

2. **Optimize hyperparameters**:
   ```go
   // For better performance
   params := map[string]interface{}{
       "num_leaves": 31,              // 2^max_depth - 1
       "max_depth": -1,               // No depth limit
       "min_child_samples": 20,       // Prevent overfitting
       "subsample": 0.8,              // Sample 80% of data
       "colsample_bytree": 0.8,       // Use 80% of features
       "reg_alpha": 0.1,              // L1 regularization
       "reg_lambda": 0.1,             // L2 regularization
   }
   ```

3. **Use early stopping**:
   ```go
   params["early_stopping_rounds"] = 10
   // Train with validation data to enable early stopping
   ```

4. **Choose appropriate objective**:
   ```go
   // For robust regression (less sensitive to outliers)
   params["objective"] = "huber"
   params["huber_delta"] = 1.0
   
   // For fair loss (another robust option)
   params["objective"] = "fair"  
   params["fair_c"] = 1.0
   ```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Parameters not recognized
```go
// ❌ Wrong - Go field names won't work
params["NumIterations"] = 100

// ✅ Correct - Use Python parameter names
params["n_estimators"] = 100
```

**Issue**: Model loading fails
```go
// Make sure file exists and is valid LightGBM format
if _, err := os.Stat("model.txt"); os.IsNotExist(err) {
    log.Fatal("Model file does not exist")
}

model, err := lightgbm.LoadFromFile("model.txt")
if err != nil {
    log.Fatal("Failed to load model:", err)
}
```

**Issue**: Poor performance
```go
// Enable verbose logging to monitor training
params["verbosity"] = 1

// Use cross-validation to validate performance
result, _ := lightgbm.CrossValidate(params, X, y, splitter, "rmse", 10, true)
fmt.Printf("CV Score: %.4f ± %.4f\n", result.GetMeanScore(), result.GetStdScore())
```

## 📚 Complete API Reference

For detailed API documentation with examples, visit:
- [pkg.go.dev/scigo/sklearn/lightgbm](https://pkg.go.dev/github.com/YuminosukeSato/scigo/sklearn/lightgbm)

---

**Ready to supercharge your ML with LightGBM in Go? 🚀**