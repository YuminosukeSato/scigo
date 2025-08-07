# Scikit-learn Compatibility Guide

SciGo provides seamless interoperability with scikit-learn, allowing you to import Python-trained models and export Go models for use in Python.

## Overview

The compatibility layer enables:
- **Import** scikit-learn models trained in Python
- **Export** SciGo models for use in scikit-learn
- **API Compatibility** with familiar scikit-learn patterns
- **Data Format** conversion between ecosystems

## Supported Models

| Model Type | Import | Export | Notes |
|------------|--------|--------|-------|
| LinearRegression | ✅ | ✅ | Full compatibility |
| SGDRegressor | ✅ | ✅ | Online learning support |
| SGDClassifier | ✅ | ✅ | Binary and multiclass |
| MiniBatchKMeans | ✅ | ✅ | Clustering |
| PassiveAggressive | ✅ | ✅ | Online learning |
| MultinomialNB | ✅ | ✅ | Text classification |

## Python to Go Workflow

### Step 1: Train Model in Python

```python
# train_model.py
from sklearn.linear_model import LinearRegression
import json
import numpy as np

# Train your model
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([3, 7, 11])

model = LinearRegression()
model.fit(X_train, y_train)

# Export model parameters
model_data = {
    "model_spec": {
        "name": "LinearRegression",
        "format_version": "1.0"
    },
    "parameters": {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "n_features": X_train.shape[1]
    }
}

# Save to JSON
with open("sklearn_model.json", "w") as f:
    json.dump(model_data, f, indent=2)

print("Model exported successfully!")
```

### Step 2: Load Model in Go

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create new model instance
    lr := linear.NewLinearRegression()
    
    // Load scikit-learn model
    err := lr.LoadFromSKLearn("sklearn_model.json")
    if err != nil {
        log.Fatal("Failed to load model:", err)
    }
    
    // Use the model for predictions
    X := mat.NewDense(2, 2, []float64{
        7, 8,
        9, 10,
    })
    
    predictions, err := lr.Predict(X)
    if err != nil {
        log.Fatal("Prediction failed:", err)
    }
    
    fmt.Println("Predictions:", predictions)
}
```

## Go to Python Workflow

### Step 1: Train Model in Go

```go
package main

import (
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Train model in Go
    X := mat.NewDense(3, 2, []float64{
        1, 2,
        3, 4,
        5, 6,
    })
    y := mat.NewVecDense(3, []float64{3, 7, 11})
    
    lr := linear.NewLinearRegression()
    err := lr.Fit(X, y)
    if err != nil {
        log.Fatal("Training failed:", err)
    }
    
    // Export to scikit-learn format
    err = lr.ExportToSKLearn("go_model.json")
    if err != nil {
        log.Fatal("Export failed:", err)
    }
    
    log.Println("Model exported for Python!")
}
```

### Step 2: Load Model in Python

```python
# load_go_model.py
import json
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class GoModelWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for models trained in Go"""
    
    def __init__(self, model_path):
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        params = model_data['parameters']
        self.coef_ = np.array(params['coefficients'])
        self.intercept_ = params['intercept']
        self.n_features_in_ = params['n_features']
    
    def predict(self, X):
        """Make predictions using loaded parameters"""
        return X @ self.coef_ + self.intercept_

# Load and use the model
model = GoModelWrapper("go_model.json")

# Make predictions
X_test = np.array([[7, 8], [9, 10]])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
```

## Model Format Specification

### JSON Structure

```json
{
  "model_spec": {
    "name": "ModelType",
    "format_version": "1.0",
    "created_at": "2024-01-01T00:00:00Z",
    "library": "scigo/scikit-learn"
  },
  "parameters": {
    // Model-specific parameters
  },
  "metadata": {
    "training_samples": 1000,
    "training_features": 10,
    "training_time_seconds": 1.23
  }
}
```

### Linear Regression Format

```json
{
  "model_spec": {
    "name": "LinearRegression",
    "format_version": "1.0"
  },
  "parameters": {
    "coefficients": [0.5, 1.5],
    "intercept": 0.1,
    "n_features": 2
  }
}
```

### SGD Classifier Format

```json
{
  "model_spec": {
    "name": "SGDClassifier",
    "format_version": "1.0"
  },
  "parameters": {
    "coefficients": [[0.1, 0.2], [0.3, 0.4]],
    "intercept": [0.01, 0.02],
    "classes": [0, 1],
    "loss": "hinge",
    "penalty": "l2",
    "alpha": 0.0001
  }
}
```

## Advanced Import/Export

### Custom Serialization

```go
// Custom export with metadata
func ExportWithMetadata(model Model, filename string) error {
    params := model.GetParams()
    
    exportData := map[string]interface{}{
        "model_spec": map[string]interface{}{
            "name": "CustomModel",
            "format_version": "1.0",
            "created_at": time.Now().Format(time.RFC3339),
        },
        "parameters": params,
        "metadata": map[string]interface{}{
            "go_version": runtime.Version(),
            "scigo_version": "0.3.0",
            "hostname": os.Hostname(),
        },
    }
    
    data, err := json.MarshalIndent(exportData, "", "  ")
    if err != nil {
        return err
    }
    
    return os.WriteFile(filename, data, 0644)
}
```

### Streaming Import/Export

```go
// Import from reader
func ImportFromReader(model Model, r io.Reader) error {
    decoder := json.NewDecoder(r)
    var data map[string]interface{}
    if err := decoder.Decode(&data); err != nil {
        return err
    }
    return model.SetParams(data["parameters"].(map[string]interface{}))
}

// Export to writer
func ExportToWriter(model Model, w io.Writer) error {
    encoder := json.NewEncoder(w)
    encoder.SetIndent("", "  ")
    return encoder.Encode(model.GetParams())
}
```

## API Compatibility

### Familiar Patterns

SciGo follows scikit-learn's API conventions:

```go
// Scikit-learn style
model.Fit(X, y)           // Train
model.Predict(X)           // Predict
model.Score(X, y)          // Evaluate
model.GetParams()          // Get parameters
model.SetParams(params)    // Set parameters

// Transformers
transformer.Fit(X)         // Learn parameters
transformer.Transform(X)   // Apply transformation
transformer.FitTransform(X)// Fit and transform
```

### Pipeline Compatibility

```go
// Create scikit-learn compatible pipeline
type Pipeline struct {
    Steps []Step
}

type Step struct {
    Name        string
    Transformer Transformer
}

func (p *Pipeline) Fit(X, y mat.Matrix) error {
    for _, step := range p.Steps[:len(p.Steps)-1] {
        X, _ = step.Transformer.FitTransform(X)
    }
    return p.Steps[len(p.Steps)-1].Transformer.Fit(X, y)
}
```

## Data Conversion

### NumPy to Gonum

```python
# Python: Export data for Go
import numpy as np
import json

X = np.random.randn(100, 5)
y = np.random.randn(100)

data = {
    "X": X.tolist(),
    "y": y.tolist(),
    "shape": list(X.shape)
}

with open("data.json", "w") as f:
    json.dump(data, f)
```

```go
// Go: Import NumPy data
type NumpyData struct {
    X     [][]float64 `json:"X"`
    Y     []float64   `json:"y"`
    Shape []int       `json:"shape"`
}

func LoadNumpyData(filename string) (*mat.Dense, *mat.VecDense, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, nil, err
    }
    
    var npData NumpyData
    if err := json.Unmarshal(data, &npData); err != nil {
        return nil, nil, err
    }
    
    // Convert to mat.Dense
    rows, cols := npData.Shape[0], npData.Shape[1]
    flatX := make([]float64, rows*cols)
    for i, row := range npData.X {
        for j, val := range row {
            flatX[i*cols+j] = val
        }
    }
    
    X := mat.NewDense(rows, cols, flatX)
    y := mat.NewVecDense(len(npData.Y), npData.Y)
    
    return X, y, nil
}
```

## Validation

### Model Validation

```go
// Validate imported model
func ValidateImportedModel(model Model, testData, expected mat.Matrix) error {
    predictions, err := model.Predict(testData)
    if err != nil {
        return fmt.Errorf("prediction failed: %w", err)
    }
    
    // Compare with expected results
    tolerance := 1e-6
    if !mat.EqualApprox(predictions, expected, tolerance) {
        return fmt.Errorf("predictions don't match expected values")
    }
    
    return nil
}
```

### Cross-Platform Testing

```bash
# Test script
#!/bin/bash

# Train in Python
python train_model.py

# Test in Go
go test -run TestImportedModel

# Train in Go
go run train_model.go

# Test in Python
python test_go_model.py
```

## Performance Considerations

### Import Performance

| Model Size | Import Time | Memory Usage |
|------------|-------------|--------------|
| <1MB | <10ms | ~2x file size |
| 1-10MB | 10-100ms | ~2x file size |
| 10-100MB | 100ms-1s | ~2x file size |
| >100MB | >1s | Consider streaming |

### Optimization Tips

1. **Use Binary Formats** for large models:
```go
// Use gob for binary serialization
import "encoding/gob"

func SaveBinary(model Model, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    encoder := gob.NewEncoder(file)
    return encoder.Encode(model)
}
```

2. **Compress Large Models**:
```go
import "compress/gzip"

func SaveCompressed(model Model, filename string) error {
    file, err := os.Create(filename + ".gz")
    if err != nil {
        return err
    }
    defer file.Close()
    
    gz := gzip.NewWriter(file)
    defer gz.Close()
    
    return model.ExportToWriter(gz)
}
```

## Troubleshooting

### Common Issues

**Issue**: Model parameters don't match
```go
// Solution: Validate dimensions
if len(params.Coefficients) != model.NFeatures {
    return fmt.Errorf("coefficient count mismatch: expected %d, got %d",
        model.NFeatures, len(params.Coefficients))
}
```

**Issue**: Prediction results differ
```go
// Solution: Check preprocessing
// Ensure same scaling/normalization in both languages
scaler := preprocessing.NewStandardScaler()
XScaled, _ := scaler.FitTransform(X)
```

**Issue**: JSON parsing errors
```go
// Solution: Use strict typing
type ModelParams struct {
    Coefficients []float64 `json:"coefficients"`
    Intercept    float64   `json:"intercept"`
}

var params ModelParams
if err := json.Unmarshal(data, &params); err != nil {
    return fmt.Errorf("invalid model format: %w", err)
}
```

## Complete Example

### End-to-End Workflow

```python
# Step 1: Train complex model in Python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import json

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SGDRegressor(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Export each component
scaler_params = {
    "mean": pipeline['scaler'].mean_.tolist(),
    "scale": pipeline['scaler'].scale_.tolist()
}

model_params = {
    "coefficients": pipeline['model'].coef_.tolist(),
    "intercept": float(pipeline['model'].intercept_[0])
}

export_data = {
    "pipeline": {
        "scaler": scaler_params,
        "model": model_params
    }
}

with open("pipeline.json", "w") as f:
    json.dump(export_data, f)
```

```go
// Step 2: Load and use in Go
package main

import (
    "encoding/json"
    "os"
    
    "github.com/YuminosukeSato/scigo/preprocessing"
    "github.com/YuminosukeSato/scigo/sklearn/linear_model"
)

type PipelineData struct {
    Pipeline struct {
        Scaler ScalerParams `json:"scaler"`
        Model  ModelParams  `json:"model"`
    } `json:"pipeline"`
}

func LoadPipeline(filename string) (*Pipeline, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    
    var pipelineData PipelineData
    if err := json.Unmarshal(data, &pipelineData); err != nil {
        return nil, err
    }
    
    // Create components
    scaler := preprocessing.NewStandardScaler()
    scaler.Mean = pipelineData.Pipeline.Scaler.Mean
    scaler.Scale = pipelineData.Pipeline.Scaler.Scale
    
    model := linear_model.NewSGDRegressor()
    model.SetParams(map[string]interface{}{
        "coefficients": pipelineData.Pipeline.Model.Coefficients,
        "intercept": pipelineData.Pipeline.Model.Intercept,
    })
    
    return &Pipeline{
        Scaler: scaler,
        Model:  model,
    }, nil
}
```

## Next Steps

- Learn about [Model Persistence](./model-persistence.md)
- Explore [Pipeline Guide](./pipeline.md)
- See [Python Integration Examples](../../examples/sklearn/)
- Read [API Reference](../api/sklearn.md)