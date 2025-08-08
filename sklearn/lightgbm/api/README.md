# LightGBM Python-style API for Go

This package provides a Python-compatible API for LightGBM in Go, allowing users familiar with Python's LightGBM to easily use the Go implementation.

## Features

- **Python-style Dataset creation**: `api.NewDataset()` similar to `lgb.Dataset()`
- **Training API**: `api.Train()` similar to `lgb.train()`
- **Booster model**: Complete booster API with prediction and model persistence
- **Callbacks**: Early stopping, verbose evaluation, and custom callbacks
- **Python-like output format**: Training logs match Python LightGBM's format

## Installation

```bash
go get github.com/YuminosukeSato/scigo/sklearn/lightgbm/api
```

## Quick Start

### Python LightGBM Example

```python
import lightgbm as lgb
import numpy as np

# Create dataset
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 2, 1000)
X_valid = np.random.randn(200, 10)
y_valid = np.random.randint(0, 2, 200)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'verbose': 0
}

# Train model
bst = lgb.train(params, train_data, num_boost_round=100,
                valid_sets=[valid_data], 
                callbacks=[lgb.early_stopping(10)])

# Predict
y_pred = bst.predict(X_valid)

# Save model
bst.save_model('model.txt')
```

### Equivalent Go Code

```go
import (
    "github.com/YuminosukeSato/scigo/sklearn/lightgbm/api"
    "gonum.org/v1/gonum/mat"
)

// Create dataset
XTrain := generateRandomMatrix(1000, 10) // Your data
yTrain := generateBinaryLabels(1000)
XValid := generateRandomMatrix(200, 10)
yValid := generateBinaryLabels(200)

trainData, _ := api.NewDataset(XTrain, yTrain)
validData, _ := api.NewDataset(XValid, yValid, 
    api.WithReference(trainData))

// Set parameters
params := map[string]interface{}{
    "objective":        "binary",
    "metric":          "binary_logloss",
    "num_leaves":      31,
    "learning_rate":   0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "verbose":         0,
}

// Train model
bst, _ := api.Train(params, trainData, 100, 
    []*api.Dataset{validData},
    api.WithEarlyStopping(10))

// Predict
yPred, _ := bst.Predict(XValid)

// Save model
bst.SaveModel("model.txt")
```

## Output Format

The Go implementation produces output identical to Python LightGBM:

```
[LightGBM] [Info] Start training from score 0.500000
[1]	valid-binary_logloss: 0.693147
[11]	valid-binary_logloss: 0.650234
[21]	valid-binary_logloss: 0.615823
[LightGBM] [Info] Early stopping at iteration 31, best iteration is 21
[LightGBM] [Info] Best binary_logloss: 0.615823
```

## Key API Components

### Dataset

```go
// Create dataset with options
dataset, err := api.NewDataset(X, y,
    api.WithFeatureNames([]string{"f1", "f2", "f3"}),
    api.WithCategoricalFeatures([]int{2, 5}),
    api.WithWeight(weights),
    api.WithReference(refDataset),
)
```

### Training

```go
// Train with various options
booster, err := api.Train(
    params,                      // Parameters map
    trainData,                   // Training dataset
    numBoostRound,              // Number of iterations
    validSets,                  // Validation datasets
    api.WithEarlyStopping(10),  // Early stopping
    api.WithVerboseEval(true, 10), // Print every 10 iterations
    api.WithCallbacks(callbacks...), // Custom callbacks
)
```

### Prediction

```go
// Make predictions
predictions, err := booster.Predict(X,
    api.WithNumIteration(50),     // Use first 50 trees
    api.WithPredictType("raw"),   // Raw scores
)
```

### Model Persistence

```go
// Save model
err := booster.SaveModel("model.json", 
    api.WithSaveType("json"))

// Load model
booster, err := api.LoadModel("model.json")
```

## Callbacks

The API supports various callbacks similar to Python:

- **Early Stopping**: Stop training when validation metric stops improving
- **Print Evaluation**: Print metrics during training
- **Record Evaluation**: Record all evaluation results
- **Custom Callbacks**: Implement the `Callback` interface

Example custom callback:

```go
type MyCallback struct{}

func (c *MyCallback) Init(env *api.CallbackEnv) error {
    // Initialize
    return nil
}

func (c *MyCallback) BeforeIteration(env *api.CallbackEnv) error {
    // Called before each iteration
    return nil
}

func (c *MyCallback) AfterIteration(env *api.CallbackEnv) error {
    // Called after each iteration
    fmt.Printf("Iteration %d\n", env.Iteration)
    return nil
}

func (c *MyCallback) Finalize(env *api.CallbackEnv) error {
    // Clean up
    return nil
}
```

## Differences from Python

While the API aims to be as similar as possible to Python's LightGBM, there are some differences:

1. **Type Safety**: Go requires explicit types, while Python is dynamically typed
2. **Matrix Library**: Uses `gonum/mat` instead of NumPy
3. **Error Handling**: Go uses explicit error returns instead of exceptions
4. **Parameter Names**: Some parameters use Go naming conventions (camelCase in struct fields)

## Performance

The Go implementation offers comparable performance to the Python version with the added benefits of:
- Compile-time type checking
- No GIL (Global Interpreter Lock)
- Better memory management
- Easy deployment (single binary)

## Testing

Run the tests to verify Python compatibility:

```bash
go test ./sklearn/lightgbm/api -v
```

Run the example:

```bash
go run sklearn/lightgbm/examples/python_style_example.go
```

## Contributing

Contributions are welcome! Please ensure that any new features maintain compatibility with Python LightGBM's API where possible.