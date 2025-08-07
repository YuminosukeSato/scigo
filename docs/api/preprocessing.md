# Preprocessing API Reference

Complete API documentation for the `preprocessing` package.

## Package Overview

```go
import "github.com/YuminosukeSato/scigo/preprocessing"
```

The `preprocessing` package provides data transformation and scaling utilities for machine learning pipelines.

## StandardScaler

Standardizes features by removing mean and scaling to unit variance.

### Constructor

```go
func NewStandardScaler() *StandardScaler
```

Creates a new StandardScaler instance.

**Returns:**
- `*StandardScaler`: A new unfitted scaler

**Example:**
```go
scaler := preprocessing.NewStandardScaler()
```

### Methods

#### Fit

```go
func (s *StandardScaler) Fit(X mat.Matrix) error
```

Computes mean and standard deviation from training data.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)

**Returns:**
- `error`: Fitting error if any

**Formula:**
```
mean[j] = (1/n) * Σ(X[i,j])
std[j] = sqrt((1/n) * Σ((X[i,j] - mean[j])^2))
```

**Example:**
```go
X := mat.NewDense(100, 3, data)
err := scaler.Fit(X)
if err != nil {
    log.Fatal(err)
}
```

#### Transform

```go
func (s *StandardScaler) Transform(X mat.Matrix) (mat.Matrix, error)
```

Transforms features to have zero mean and unit variance.

**Parameters:**
- `X`: Feature matrix to transform

**Returns:**
- `mat.Matrix`: Transformed features
- `error`: Transformation error if any

**Formula:**
```
X_scaled[i,j] = (X[i,j] - mean[j]) / std[j]
```

**Example:**
```go
XScaled, err := scaler.Transform(XTest)
if err != nil {
    log.Fatal(err)
}
```

#### FitTransform

```go
func (s *StandardScaler) FitTransform(X mat.Matrix) (mat.Matrix, error)
```

Fits scaler and transforms data in one step.

**Parameters:**
- `X`: Feature matrix to fit and transform

**Returns:**
- `mat.Matrix`: Transformed features
- `error`: Error if any

**Example:**
```go
XScaled, err := scaler.FitTransform(XTrain)
```

#### InverseTransform

```go
func (s *StandardScaler) InverseTransform(X mat.Matrix) (mat.Matrix, error)
```

Reverts scaling transformation.

**Parameters:**
- `X`: Scaled feature matrix

**Returns:**
- `mat.Matrix`: Original scale features
- `error`: Error if any

**Formula:**
```
X_original[i,j] = X_scaled[i,j] * std[j] + mean[j]
```

**Example:**
```go
XOriginal, err := scaler.InverseTransform(XScaled)
```

#### GetParams

```go
func (s *StandardScaler) GetParams() map[string]interface{}
```

Returns scaler parameters.

**Returns:**
- `map[string]interface{}`: Contains "mean" and "scale" arrays

**Example:**
```go
params := scaler.GetParams()
mean := params["mean"].([]float64)
scale := params["scale"].([]float64)
```

## MinMaxScaler

Scales features to a given range (default [0, 1]).

### Constructor

```go
func NewMinMaxScaler(featureRange ...float64) *MinMaxScaler
```

Creates a new MinMaxScaler.

**Parameters:**
- `featureRange`: Optional min and max values (default [0, 1])

**Returns:**
- `*MinMaxScaler`: A new unfitted scaler

**Example:**
```go
// Default range [0, 1]
scaler := preprocessing.NewMinMaxScaler()

// Custom range [-1, 1]
scaler := preprocessing.NewMinMaxScaler(-1, 1)
```

### Methods

#### Fit

```go
func (s *MinMaxScaler) Fit(X mat.Matrix) error
```

Computes min and max values from training data.

**Parameters:**
- `X`: Feature matrix (n_samples × n_features)

**Returns:**
- `error`: Fitting error if any

**Example:**
```go
err := scaler.Fit(XTrain)
```

#### Transform

```go
func (s *MinMaxScaler) Transform(X mat.Matrix) (mat.Matrix, error)
```

Scales features to specified range.

**Formula:**
```
X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
```

Where `min` and `max` are the feature range bounds.

**Example:**
```go
XScaled, err := scaler.Transform(XTest)
```

## RobustScaler

Scales features using statistics robust to outliers.

### Constructor

```go
func NewRobustScaler() *RobustScaler
```

Creates a new RobustScaler that uses median and IQR.

**Returns:**
- `*RobustScaler`: A new unfitted scaler

**Example:**
```go
scaler := preprocessing.NewRobustScaler()
```

### Methods

#### Fit

```go
func (s *RobustScaler) Fit(X mat.Matrix) error
```

Computes median and interquartile range.

**Formula:**
```
median[j] = median(X[:,j])
IQR[j] = Q75(X[:,j]) - Q25(X[:,j])
```

#### Transform

```go
func (s *RobustScaler) Transform(X mat.Matrix) (mat.Matrix, error)
```

Centers and scales features using robust statistics.

**Formula:**
```
X_scaled[i,j] = (X[i,j] - median[j]) / IQR[j]
```

## Normalizer

Normalizes samples individually to unit norm.

### Constructor

```go
func NewNormalizer(norm string) *Normalizer
```

Creates a new Normalizer.

**Parameters:**
- `norm`: Norm type ("l1", "l2", or "max")

**Returns:**
- `*Normalizer`: A new normalizer

**Example:**
```go
// L2 normalization (default)
normalizer := preprocessing.NewNormalizer("l2")

// L1 normalization
normalizer := preprocessing.NewNormalizer("l1")

// Max normalization
normalizer := preprocessing.NewNormalizer("max")
```

### Methods

#### Transform

```go
func (n *Normalizer) Transform(X mat.Matrix) (mat.Matrix, error)
```

Normalizes each sample to unit norm.

**Formulas:**
- L1: `X_norm[i] = X[i] / Σ|X[i,j]|`
- L2: `X_norm[i] = X[i] / sqrt(Σ(X[i,j]^2))`
- Max: `X_norm[i] = X[i] / max(|X[i,j]|)`

**Example:**
```go
XNorm, err := normalizer.Transform(X)
```

## OneHotEncoder

Encodes categorical features as one-hot numeric arrays.

### Constructor

```go
func NewOneHotEncoder() *OneHotEncoder
```

Creates a new OneHotEncoder.

**Returns:**
- `*OneHotEncoder`: A new unfitted encoder

**Example:**
```go
encoder := preprocessing.NewOneHotEncoder()
```

### Methods

#### Fit

```go
func (e *OneHotEncoder) Fit(X mat.Matrix) error
```

Learns categories from training data.

**Parameters:**
- `X`: Feature matrix with categorical values

**Returns:**
- `error`: Fitting error if any

**Example:**
```go
// X contains categorical features (e.g., 0, 1, 2 for categories)
err := encoder.Fit(XCategorical)
```

#### Transform

```go
func (e *OneHotEncoder) Transform(X mat.Matrix) (mat.Matrix, error)
```

Transforms categorical features to one-hot encoding.

**Example:**
```go
// Input: [0, 1, 2, 0]
// Output: [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
XEncoded, err := encoder.Transform(XCategorical)
```

#### GetFeatureNames

```go
func (e *OneHotEncoder) GetFeatureNames() []string
```

Returns names of encoded features.

**Returns:**
- `[]string`: Feature names like "feature_0_cat_1"

**Example:**
```go
names := encoder.GetFeatureNames()
// ["feature_0_cat_0", "feature_0_cat_1", "feature_0_cat_2"]
```

## LabelEncoder

Encodes target labels with values between 0 and n_classes-1.

### Constructor

```go
func NewLabelEncoder() *LabelEncoder
```

Creates a new LabelEncoder.

**Returns:**
- `*LabelEncoder`: A new unfitted encoder

**Example:**
```go
encoder := preprocessing.NewLabelEncoder()
```

### Methods

#### Fit

```go
func (e *LabelEncoder) Fit(y []string) error
```

Learns labels from training data.

**Parameters:**
- `y`: Array of string labels

**Returns:**
- `error`: Fitting error if any

**Example:**
```go
labels := []string{"cat", "dog", "bird", "cat"}
err := encoder.Fit(labels)
```

#### Transform

```go
func (e *LabelEncoder) Transform(y []string) ([]int, error)
```

Transforms labels to encoded integer values.

**Parameters:**
- `y`: Array of string labels

**Returns:**
- `[]int`: Encoded labels
- `error`: Transformation error if any

**Example:**
```go
labels := []string{"cat", "dog", "bird"}
encoded, err := encoder.Transform(labels)
// Returns: [0, 1, 2]
```

#### InverseTransform

```go
func (e *LabelEncoder) InverseTransform(y []int) ([]string, error)
```

Reverts encoded labels to original strings.

**Parameters:**
- `y`: Array of encoded integers

**Returns:**
- `[]string`: Original labels
- `error`: Error if any

**Example:**
```go
encoded := []int{0, 1, 2, 0}
labels, err := encoder.InverseTransform(encoded)
// Returns: ["cat", "dog", "bird", "cat"]
```

#### Classes

```go
func (e *LabelEncoder) Classes() []string
```

Returns unique classes learned during fit.

**Returns:**
- `[]string`: Unique class labels

**Example:**
```go
classes := encoder.Classes()
// Returns: ["bird", "cat", "dog"] (sorted)
```

## PolynomialFeatures

Generates polynomial and interaction features.

### Constructor

```go
func NewPolynomialFeatures(degree int, includeIntercept bool) *PolynomialFeatures
```

Creates a new PolynomialFeatures transformer.

**Parameters:**
- `degree`: Maximum degree of polynomial features
- `includeIntercept`: Whether to include bias column

**Returns:**
- `*PolynomialFeatures`: A new transformer

**Example:**
```go
// Generate up to degree 2 with intercept
poly := preprocessing.NewPolynomialFeatures(2, true)

// Generate up to degree 3 without intercept
poly := preprocessing.NewPolynomialFeatures(3, false)
```

### Methods

#### Transform

```go
func (p *PolynomialFeatures) Transform(X mat.Matrix) (mat.Matrix, error)
```

Generates polynomial features.

**Example:**
```go
// Input: [a, b]
// Output (degree=2): [1, a, b, a^2, a*b, b^2]
XPoly, err := poly.Transform(X)
```

#### GetFeatureNames

```go
func (p *PolynomialFeatures) GetFeatureNames(inputFeatures []string) []string
```

Returns names of polynomial features.

**Parameters:**
- `inputFeatures`: Names of input features

**Returns:**
- `[]string`: Names of generated features

**Example:**
```go
names := poly.GetFeatureNames([]string{"x1", "x2"})
// Returns: ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
```

## Imputer

Handles missing values in datasets.

### Constructor

```go
func NewImputer(strategy string, missingValue float64) *Imputer
```

Creates a new Imputer.

**Parameters:**
- `strategy`: Imputation strategy ("mean", "median", "most_frequent", "constant")
- `missingValue`: Value to treat as missing (e.g., NaN, -999)

**Returns:**
- `*Imputer`: A new unfitted imputer

**Example:**
```go
// Replace NaN with mean
imputer := preprocessing.NewImputer("mean", math.NaN())

// Replace -999 with median
imputer := preprocessing.NewImputer("median", -999)
```

### Methods

#### Fit

```go
func (i *Imputer) Fit(X mat.Matrix) error
```

Learns imputation values from training data.

**Parameters:**
- `X`: Feature matrix possibly containing missing values

**Returns:**
- `error`: Fitting error if any

#### Transform

```go
func (i *Imputer) Transform(X mat.Matrix) (mat.Matrix, error)
```

Imputes missing values.

**Parameters:**
- `X`: Feature matrix with missing values

**Returns:**
- `mat.Matrix`: Imputed feature matrix
- `error`: Transformation error if any

**Example:**
```go
// Fit on training data
err := imputer.Fit(XTrain)

// Transform both train and test
XTrainImputed, _ := imputer.Transform(XTrain)
XTestImputed, _ := imputer.Transform(XTest)
```

## Binarizer

Binarizes data (sets values to 0 or 1) based on a threshold.

### Constructor

```go
func NewBinarizer(threshold float64) *Binarizer
```

Creates a new Binarizer.

**Parameters:**
- `threshold`: Threshold for binarization

**Returns:**
- `*Binarizer`: A new binarizer

**Example:**
```go
binarizer := preprocessing.NewBinarizer(0.5)
```

### Methods

#### Transform

```go
func (b *Binarizer) Transform(X mat.Matrix) (mat.Matrix, error)
```

Binarizes features based on threshold.

**Formula:**
```
X_binary[i,j] = 1 if X[i,j] > threshold else 0
```

**Example:**
```go
// Values > 0.5 become 1, others become 0
XBinary, err := binarizer.Transform(X)
```

## Pipeline Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/preprocessing"
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Load data
    X := loadData()
    y := loadTargets()
    
    // Create preprocessing pipeline
    imputer := preprocessing.NewImputer("mean", math.NaN())
    scaler := preprocessing.NewStandardScaler()
    poly := preprocessing.NewPolynomialFeatures(2, true)
    
    // Fit and transform
    X, _ = imputer.FitTransform(X)
    X, _ = scaler.FitTransform(X)
    X, _ = poly.Transform(X)
    
    // Train model on preprocessed data
    model := linear.NewLinearRegression()
    err := model.Fit(X, y)
    if err != nil {
        log.Fatal(err)
    }
    
    // Preprocess test data (using fitted transformers)
    XTest := loadTestData()
    XTest, _ = imputer.Transform(XTest)
    XTest, _ = scaler.Transform(XTest)
    XTest, _ = poly.Transform(XTest)
    
    // Make predictions
    predictions, _ := model.Predict(XTest)
    fmt.Println("Predictions:", predictions)
}
```

## Custom Transformers

### Implementing Custom Transformer

```go
type CustomTransformer struct {
    // Your parameters
}

func (t *CustomTransformer) Fit(X mat.Matrix) error {
    // Learn parameters from X
    return nil
}

func (t *CustomTransformer) Transform(X mat.Matrix) (mat.Matrix, error) {
    // Apply transformation
    rows, cols := X.Dims()
    result := mat.NewDense(rows, cols, nil)
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            // Your transformation logic
            result.Set(i, j, customTransform(X.At(i, j)))
        }
    }
    
    return result, nil
}

func (t *CustomTransformer) FitTransform(X mat.Matrix) (mat.Matrix, error) {
    if err := t.Fit(X); err != nil {
        return nil, err
    }
    return t.Transform(X)
}
```

## Performance Considerations

### Sparse Data Handling

```go
// For sparse data, use specialized implementations
type SparseScaler struct {
    mean  map[int]float64
    scale map[int]float64
}

func (s *SparseScaler) TransformSparse(X SparseMatrix) SparseMatrix {
    // Only process non-zero values
    for i, j, v := range X.NonZero() {
        scaled := (v - s.mean[j]) / s.scale[j]
        X.Set(i, j, scaled)
    }
    return X
}
```

### Batch Processing

```go
func BatchTransform(transformer Transformer, X mat.Matrix, batchSize int) (mat.Matrix, error) {
    rows, cols := X.Dims()
    result := mat.NewDense(rows, cols, nil)
    
    for start := 0; start < rows; start += batchSize {
        end := min(start+batchSize, rows)
        batch := X.Slice(start, end, 0, cols)
        
        transformed, err := transformer.Transform(batch)
        if err != nil {
            return nil, err
        }
        
        // Copy to result
        for i := start; i < end; i++ {
            for j := 0; j < cols; j++ {
                result.Set(i, j, transformed.At(i-start, j))
            }
        }
    }
    
    return result, nil
}
```

## Thread Safety

All transformers are thread-safe for transform operations after fitting:

```go
var wg sync.WaitGroup
scaler := preprocessing.NewStandardScaler()
scaler.Fit(XTrain)

// Safe concurrent transforms
for _, batch := range batches {
    wg.Add(1)
    go func(X mat.Matrix) {
        defer wg.Done()
        transformed, _ := scaler.Transform(X)
        processBatch(transformed)
    }(batch)
}
wg.Wait()
```

## Next Steps

- Explore [Metrics API](./metrics.md)
- Learn about [Linear Models API](./linear.md)
- See [Pipeline Guide](../guides/pipeline.md)
- Read [Examples](../../examples/preprocessing/)