# LightGBM Go Implementation

A pure Go implementation of LightGBM with full Python compatibility.

## Features

### ✅ Completed Features

#### Core Training & Prediction
- **Gradient Boosting Decision Trees (GBDT)** - Full implementation with histogram-based algorithm
- **Binary Classification** - Logistic regression objective with sigmoid transformation
- **Multiclass Classification** - Softmax objective with native multiclass support
- **Regression** - L2, L1, Huber, Quantile, and other regression objectives
- **GOSS (Gradient-based One-Side Sampling)** - Intelligent data sampling based on gradients
- **DART (Dropouts meet Multiple Additive Regression Trees)** - Dropout-based regularization

#### Python Compatibility
- **JSON Model Loading** - Load models trained in Python LightGBM
- **Text Model Format** - Support for LightGBM text format
- **Categorical Features** - Full support for categorical splits
- **Missing Value Handling** - NaN and zero value handling with default directions
- **Raw Score Prediction** - `predict(raw_score=True)` compatibility
- **Leaf Index Prediction** - `predict(pred_leaf=True)` compatibility
- **Numerical Precision** - Complete precision match with Python implementation

#### Training Parameters
- **Feature Sampling** (`feature_fraction`) - Random subset of features per tree
- **Data Bagging** (`bagging_fraction`, `bagging_freq`) - Random subset of data
- **L1/L2 Regularization** (`lambda_l1`, `lambda_l2`) - Prevent overfitting
- **Min Data in Leaf** (`min_data_in_leaf`) - Minimum samples per leaf
- **Learning Rate** (`learning_rate`) - Shrinkage rate for trees
- **Number of Iterations** (`num_iterations`) - Number of boosting rounds
- **Number of Leaves** (`num_leaves`) - Maximum leaves per tree
- **Max Depth** (`max_depth`) - Maximum tree depth

#### Advanced Features
- **Custom Objectives** - Support for custom loss functions
- **Early Stopping** - Stop training when validation metric stops improving
- **Callbacks** - Custom callbacks for monitoring and control
- **Cross-Validation** - K-fold cross-validation support
- **Feature Importance** - Split and gain-based importance

## Installation

```bash
go get github.com/YuminosukeSato/scigo/sklearn/lightgbm
```

## Quick Start

### Training a Model

```go
package main

import (
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create training data
    X := mat.NewDense(100, 4, nil) // 100 samples, 4 features
    y := mat.NewDense(100, 1, nil) // 100 labels
    // ... fill with your data ...

    // Set parameters
    params := lightgbm.TrainingParams{
        NumIterations:   100,
        LearningRate:    0.1,
        NumLeaves:       31,
        FeatureFraction: 0.9,
        BaggingFraction: 0.8,
        BaggingFreq:     5,
        Lambda:          1.0,  // L2 regularization
        Alpha:           0.0,  // L1 regularization
        Objective:       "binary",
    }

    // Train model
    trainer := lightgbm.NewTrainer(params)
    err := trainer.Fit(X, y)
    if err != nil {
        panic(err)
    }

    // Get trained model
    model := trainer.GetModel()

    // Make predictions
    predictor := lightgbm.NewPredictor(model)
    predictions, err := predictor.Predict(X)
    if err != nil {
        panic(err)
    }
}
```

### Loading a Python-Trained Model

```go
// Load model from JSON file
model, err := lightgbm.LoadJSONModelFromFile("model.json")
if err != nil {
    panic(err)
}

// Or auto-detect format
model, err = lightgbm.LoadModelAutoDetect("model.txt")
if err != nil {
    panic(err)
}

// Make predictions
predictor := lightgbm.NewPredictor(model)
predictions, err := predictor.Predict(X)
```

### Advanced Prediction Options

```go
// Raw scores (before sigmoid/softmax)
rawScores, err := predictor.PredictRawScore(X)

// Leaf indices for feature engineering
leafIndices, err := predictor.PredictLeaf(X)

// Probability predictions for classification
probabilities, err := predictor.PredictProba(X)
```

## API Compatibility

The API closely follows Python's LightGBM interface:

```go
// Using the Booster API (similar to Python)
booster := api.NewBooster(params)
booster.InitFromModel(model)

// Predictions with various options
predictions, _ := booster.Predict(data)
rawScores, _ := booster.PredictRawScore(data)
leafIndices, _ := booster.PredictLeaf(data)
```

## Performance

- **Parallel Training** - Multi-threaded histogram construction
- **Memory Efficient** - Histogram-based algorithm reduces memory usage
- **Fast Prediction** - Optimized tree traversal with cache-friendly layout

## Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...
```

## Precision Validation

The implementation includes comprehensive precision testing against Python LightGBM:

```bash
# Generate precision test data
python generate_precision_data.py

# Run precision tests
go test -v -run TestPrecisionAgainstPython
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass
2. Code follows Go best practices
3. Numerical precision matches Python implementation
4. Documentation is updated

## License

This project is part of the SciGo suite and follows the same license as the parent project.

## Roadmap

### 🚧 In Progress
- Sample weights and class weights
- Deterministic training validation
- SHAP value computation (TreeSHAP)
- Version compatibility (v3/v4 differences)

### 📋 Planned
- GPU acceleration
- Distributed training
- Additional objectives (Tweedie, Gamma, etc.)
- Feature interactions and constraints

## Acknowledgments

This implementation is based on the original [LightGBM](https://github.com/microsoft/LightGBM) paper and follows its algorithms closely to ensure compatibility.