# Preprocessing Guide

Comprehensive guide to data preprocessing and feature engineering in SciGo.

## Overview

Data preprocessing is crucial for machine learning success. SciGo provides a complete suite of preprocessing tools compatible with scikit-learn workflows.

## Preprocessing Pipeline

```
Raw Data → Cleaning → Scaling → Encoding → Feature Engineering → Model-Ready Data
```

## Data Cleaning

### Handling Missing Values

```go
package main

import (
    "math"
    "github.com/YuminosukeSato/scigo/preprocessing"
    "gonum.org/v1/gonum/mat"
)

func handleMissingData() {
    // Create imputer for different strategies
    
    // Mean imputation
    meanImputer := preprocessing.NewImputer("mean", math.NaN())
    
    // Median imputation (robust to outliers)
    medianImputer := preprocessing.NewImputer("median", math.NaN())
    
    // Most frequent (mode) imputation
    modeImputer := preprocessing.NewImputer("most_frequent", math.NaN())
    
    // Constant value imputation
    constantImputer := preprocessing.NewImputer("constant", -999)
    constantImputer.SetFillValue(0)
    
    // Fit and transform
    XTrain := loadTrainingData()
    meanImputer.Fit(XTrain)
    
    XTrainClean, _ := meanImputer.Transform(XTrain)
    XTestClean, _ := meanImputer.Transform(XTest)
}
```

### Advanced Imputation

```go
// Iterative imputation (MICE-like)
type IterativeImputer struct {
    estimator Model
    maxIter   int
    tolerance float64
}

func (ii *IterativeImputer) FitTransform(X mat.Matrix) (mat.Matrix, error) {
    rows, cols := X.Dims()
    XFilled := mat.DenseCopyOf(X)
    
    // Initial imputation with mean
    ii.initialImpute(XFilled)
    
    for iter := 0; iter < ii.maxIter; iter++ {
        oldX := mat.DenseCopyOf(XFilled)
        
        // Iterate over features with missing values
        for j := 0; j < cols; j++ {
            if !ii.hasMissing(X, j) {
                continue
            }
            
            // Use other features to predict missing values
            mask := ii.getMissingMask(X, j)
            XTrain, yTrain := ii.prepareTrainingData(XFilled, j, mask)
            
            // Train model
            ii.estimator.Fit(XTrain, yTrain)
            
            // Predict missing values
            XPred := ii.preparePredictionData(XFilled, j, mask)
            predictions, _ := ii.estimator.Predict(XPred)
            
            // Fill missing values
            ii.fillColumn(XFilled, j, predictions, mask)
        }
        
        // Check convergence
        if ii.hasConverged(oldX, XFilled) {
            break
        }
    }
    
    return XFilled, nil
}
```

### Outlier Detection and Removal

```go
// IQR-based outlier detection
func removeOutliersIQR(X mat.Matrix, threshold float64) (mat.Matrix, []int) {
    rows, cols := X.Dims()
    mask := make([]bool, rows)
    
    for j := 0; j < cols; j++ {
        column := mat.Col(nil, j, X)
        
        // Calculate quartiles
        q1 := percentile(column, 25)
        q3 := percentile(column, 75)
        iqr := q3 - q1
        
        // Define bounds
        lowerBound := q1 - threshold*iqr
        upperBound := q3 + threshold*iqr
        
        // Mark outliers
        for i := 0; i < rows; i++ {
            val := X.At(i, j)
            if val < lowerBound || val > upperBound {
                mask[i] = true
            }
        }
    }
    
    // Filter out outliers
    cleanIndices := []int{}
    for i, isOutlier := range mask {
        if !isOutlier {
            cleanIndices = append(cleanIndices, i)
        }
    }
    
    return filterRows(X, cleanIndices), cleanIndices
}

// Z-score based outlier detection
func removeOutliersZScore(X mat.Matrix, threshold float64) (mat.Matrix, []int) {
    scaler := preprocessing.NewStandardScaler()
    XScaled, _ := scaler.FitTransform(X)
    
    rows, cols := XScaled.Dims()
    mask := make([]bool, rows)
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            if math.Abs(XScaled.At(i, j)) > threshold {
                mask[i] = true
                break
            }
        }
    }
    
    return filterByMask(X, mask)
}
```

## Feature Scaling

### StandardScaler (Z-score Normalization)

```go
func standardScaling() {
    // Standardize to mean=0, std=1
    scaler := preprocessing.NewStandardScaler()
    
    // Fit on training data
    scaler.Fit(XTrain)
    
    // Transform both train and test
    XTrainScaled, _ := scaler.Transform(XTrain)
    XTestScaled, _ := scaler.Transform(XTest)
    
    // Get parameters
    params := scaler.GetParams()
    mean := params["mean"].([]float64)
    scale := params["scale"].([]float64)
    
    fmt.Printf("Feature 0 - Mean: %.2f, Std: %.2f\n", mean[0], scale[0])
}
```

### MinMaxScaler (Normalization)

```go
func minMaxScaling() {
    // Scale to [0, 1] range
    scaler := preprocessing.NewMinMaxScaler()
    
    // Scale to custom range [-1, 1]
    scalerCustom := preprocessing.NewMinMaxScaler(-1, 1)
    
    XScaled, _ := scaler.FitTransform(X)
    
    // Verify range
    for j := 0; j < XScaled.(*mat.Dense).RawMatrix().Cols; j++ {
        col := mat.Col(nil, j, XScaled)
        minVal, maxVal := minMax(col)
        fmt.Printf("Feature %d range: [%.2f, %.2f]\n", j, minVal, maxVal)
    }
}
```

### RobustScaler (Outlier-Robust)

```go
func robustScaling() {
    // Uses median and IQR instead of mean and std
    scaler := preprocessing.NewRobustScaler()
    
    // Robust to outliers
    XWithOutliers := addOutliers(X)
    XScaled, _ := scaler.FitTransform(XWithOutliers)
    
    // Compare with StandardScaler
    stdScaler := preprocessing.NewStandardScaler()
    XStdScaled, _ := stdScaler.FitTransform(XWithOutliers)
    
    // RobustScaler less affected by outliers
}
```

### Normalizer (Sample-wise Normalization)

```go
func normalization() {
    // L2 normalization (unit norm)
    l2Normalizer := preprocessing.NewNormalizer("l2")
    XNorm, _ := l2Normalizer.Transform(X)
    
    // Verify unit norm
    for i := 0; i < XNorm.(*mat.Dense).RawMatrix().Rows; i++ {
        row := XNorm.(*mat.Dense).RowView(i)
        norm := mat.Norm(row, 2)
        fmt.Printf("Sample %d norm: %.6f\n", i, norm) // Should be 1.0
    }
    
    // L1 normalization (sum to 1)
    l1Normalizer := preprocessing.NewNormalizer("l1")
    XL1, _ := l1Normalizer.Transform(X)
    
    // Max normalization (max abs value = 1)
    maxNormalizer := preprocessing.NewNormalizer("max")
    XMax, _ := maxNormalizer.Transform(X)
}
```

## Categorical Encoding

### One-Hot Encoding

```go
func oneHotEncoding() {
    // Categorical data
    categories := mat.NewDense(100, 2, nil)
    // Column 0: colors (0=red, 1=green, 2=blue)
    // Column 1: sizes (0=small, 1=medium, 2=large)
    
    encoder := preprocessing.NewOneHotEncoder()
    encoder.Fit(categories)
    
    // Transform to binary matrix
    encoded, _ := encoder.Transform(categories)
    // Result: 100 x 6 matrix (3 colors + 3 sizes)
    
    // Get feature names
    names := encoder.GetFeatureNames()
    // ["color_0", "color_1", "color_2", "size_0", "size_1", "size_2"]
    
    // Handle unknown categories
    encoder.SetHandleUnknown("ignore")
}
```

### Label Encoding

```go
func labelEncoding() {
    // String labels
    labels := []string{"cat", "dog", "bird", "cat", "dog"}
    
    encoder := preprocessing.NewLabelEncoder()
    encoder.Fit(labels)
    
    // Encode to integers
    encoded, _ := encoder.Transform(labels)
    // Result: [0, 1, 2, 0, 1]
    
    // Inverse transform
    decoded, _ := encoder.InverseTransform(encoded)
    // Result: ["cat", "dog", "bird", "cat", "dog"]
    
    // Get classes
    classes := encoder.Classes()
    // ["bird", "cat", "dog"] (sorted)
}
```

### Ordinal Encoding

```go
type OrdinalEncoder struct {
    categories map[int][]string
    mapping    map[int]map[string]int
}

func (oe *OrdinalEncoder) Fit(X [][]string) {
    cols := len(X[0])
    oe.categories = make(map[int][]string)
    oe.mapping = make(map[int]map[string]int)
    
    for j := 0; j < cols; j++ {
        // Collect unique values
        unique := make(map[string]bool)
        for i := range X {
            unique[X[i][j]] = true
        }
        
        // Create ordered mapping
        oe.mapping[j] = make(map[string]int)
        idx := 0
        for val := range unique {
            oe.categories[j] = append(oe.categories[j], val)
            oe.mapping[j][val] = idx
            idx++
        }
    }
}
```

### Target Encoding

```go
// Mean encoding for categorical features
type TargetEncoder struct {
    encodings map[string]float64
    smoothing float64
}

func (te *TargetEncoder) Fit(X []string, y []float64) {
    te.encodings = make(map[string]float64)
    
    // Calculate mean target per category
    categoryStats := make(map[string]struct {
        sum   float64
        count int
    })
    
    for i, cat := range X {
        stats := categoryStats[cat]
        stats.sum += y[i]
        stats.count++
        categoryStats[cat] = stats
    }
    
    // Global mean for smoothing
    globalMean := mean(y)
    
    // Apply smoothing
    for cat, stats := range categoryStats {
        weight := float64(stats.count) / (float64(stats.count) + te.smoothing)
        te.encodings[cat] = weight*(stats.sum/float64(stats.count)) + 
                            (1-weight)*globalMean
    }
}
```

## Feature Engineering

### Polynomial Features

```go
func polynomialFeatures() {
    // Create polynomial and interaction features
    poly := preprocessing.NewPolynomialFeatures(
        2,     // degree
        true,  // include_bias
    )
    
    // [a, b] → [1, a, b, a², ab, b²]
    XPoly, _ := poly.Transform(X)
    
    // Higher degree polynomials
    poly3 := preprocessing.NewPolynomialFeatures(3, false)
    // [a, b] → [a, b, a², ab, b², a³, a²b, ab², b³]
    
    // Get feature names
    inputNames := []string{"x1", "x2"}
    polyNames := poly.GetFeatureNames(inputNames)
    // ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
}
```

### Binning/Discretization

```go
type KBinsDiscretizer struct {
    nBins    int
    strategy string // "uniform", "quantile", "kmeans"
    encode   string // "ordinal", "onehot"
}

func (kbd *KBinsDiscretizer) FitTransform(X mat.Matrix) (mat.Matrix, error) {
    rows, cols := X.Dims()
    result := mat.NewDense(rows, cols, nil)
    
    for j := 0; j < cols; j++ {
        column := mat.Col(nil, j, X)
        
        // Determine bin edges
        var edges []float64
        switch kbd.strategy {
        case "uniform":
            edges = kbd.uniformBins(column)
        case "quantile":
            edges = kbd.quantileBins(column)
        case "kmeans":
            edges = kbd.kmeansBins(column)
        }
        
        // Assign to bins
        for i := 0; i < rows; i++ {
            val := X.At(i, j)
            bin := kbd.findBin(val, edges)
            result.Set(i, j, float64(bin))
        }
    }
    
    if kbd.encode == "onehot" {
        encoder := preprocessing.NewOneHotEncoder()
        return encoder.FitTransform(result)
    }
    
    return result, nil
}
```

### Feature Selection

```go
// Variance Threshold
type VarianceThreshold struct {
    threshold float64
    variances []float64
    mask      []bool
}

func (vt *VarianceThreshold) Fit(X mat.Matrix) {
    _, cols := X.Dims()
    vt.variances = make([]float64, cols)
    vt.mask = make([]bool, cols)
    
    for j := 0; j < cols; j++ {
        column := mat.Col(nil, j, X)
        vt.variances[j] = variance(column)
        vt.mask[j] = vt.variances[j] > vt.threshold
    }
}

func (vt *VarianceThreshold) Transform(X mat.Matrix) mat.Matrix {
    rows, cols := X.Dims()
    
    // Count selected features
    nSelected := 0
    for _, selected := range vt.mask {
        if selected {
            nSelected++
        }
    }
    
    // Create new matrix with selected features
    result := mat.NewDense(rows, nSelected, nil)
    newCol := 0
    
    for j := 0; j < cols; j++ {
        if vt.mask[j] {
            for i := 0; i < rows; i++ {
                result.Set(i, newCol, X.At(i, j))
            }
            newCol++
        }
    }
    
    return result
}
```

### Feature Extraction

```go
// Statistical features from time series
func extractTimeSeriesFeatures(series []float64, windowSize int) []float64 {
    features := []float64{}
    
    // Rolling statistics
    for i := 0; i <= len(series)-windowSize; i++ {
        window := series[i : i+windowSize]
        
        features = append(features,
            mean(window),
            std(window),
            max(window),
            min(window),
            percentile(window, 25),
            percentile(window, 75),
            skewness(window),
            kurtosis(window),
        )
    }
    
    // Lag features
    for lag := 1; lag <= 5; lag++ {
        if lag < len(series) {
            features = append(features, series[len(series)-lag])
        }
    }
    
    // Trend features
    slope, intercept := linearTrend(series)
    features = append(features, slope, intercept)
    
    return features
}
```

## Custom Transformers

### Creating Custom Transformers

```go
type LogTransformer struct {
    offset float64
    base   float64
}

func NewLogTransformer(offset float64) *LogTransformer {
    return &LogTransformer{
        offset: offset,
        base:   math.E,
    }
}

func (lt *LogTransformer) Transform(X mat.Matrix) (mat.Matrix, error) {
    rows, cols := X.Dims()
    result := mat.NewDense(rows, cols, nil)
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            val := X.At(i, j)
            if val+lt.offset <= 0 {
                return nil, fmt.Errorf("cannot take log of non-positive value")
            }
            result.Set(i, j, math.Log(val+lt.offset)/math.Log(lt.base))
        }
    }
    
    return result, nil
}

func (lt *LogTransformer) InverseTransform(X mat.Matrix) (mat.Matrix, error) {
    rows, cols := X.Dims()
    result := mat.NewDense(rows, cols, nil)
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            val := X.At(i, j)
            result.Set(i, j, math.Pow(lt.base, val)-lt.offset)
        }
    }
    
    return result, nil
}
```

### Function Transformer

```go
type FunctionTransformer struct {
    transform func(mat.Matrix) mat.Matrix
    inverse   func(mat.Matrix) mat.Matrix
}

func NewFunctionTransformer(
    transform func(mat.Matrix) mat.Matrix,
    inverse func(mat.Matrix) mat.Matrix,
) *FunctionTransformer {
    return &FunctionTransformer{
        transform: transform,
        inverse:   inverse,
    }
}

// Example: Square root transformer
sqrtTransformer := NewFunctionTransformer(
    func(X mat.Matrix) mat.Matrix {
        return applyElementwise(X, math.Sqrt)
    },
    func(X mat.Matrix) mat.Matrix {
        return applyElementwise(X, func(x float64) float64 {
            return x * x
        })
    },
)
```

## Pipeline Construction

### Sequential Pipeline

```go
type Pipeline struct {
    steps []Step
}

type Step struct {
    name        string
    transformer Transformer
}

func NewPipeline(steps ...Step) *Pipeline {
    return &Pipeline{steps: steps}
}

func (p *Pipeline) FitTransform(X mat.Matrix) (mat.Matrix, error) {
    current := X
    
    for _, step := range p.steps {
        var err error
        current, err = step.transformer.FitTransform(current)
        if err != nil {
            return nil, fmt.Errorf("step %s failed: %w", step.name, err)
        }
    }
    
    return current, nil
}

// Example pipeline
pipeline := NewPipeline(
    Step{"imputer", preprocessing.NewImputer("mean", math.NaN())},
    Step{"scaler", preprocessing.NewStandardScaler()},
    Step{"poly", preprocessing.NewPolynomialFeatures(2, false)},
)

XProcessed, _ := pipeline.FitTransform(XRaw)
```

### Column Transformer

```go
type ColumnTransformer struct {
    transformers []ColumnTransform
}

type ColumnTransform struct {
    name        string
    transformer Transformer
    columns     []int
}

func (ct *ColumnTransformer) FitTransform(X mat.Matrix) (mat.Matrix, error) {
    results := []mat.Matrix{}
    
    for _, t := range ct.transformers {
        // Extract columns
        subset := extractColumns(X, t.columns)
        
        // Transform
        transformed, err := t.transformer.FitTransform(subset)
        if err != nil {
            return nil, err
        }
        
        results = append(results, transformed)
    }
    
    // Concatenate results
    return concatenateHorizontal(results...), nil
}

// Example: Different preprocessing for numerical and categorical
ct := NewColumnTransformer(
    ColumnTransform{
        "numerical",
        preprocessing.NewStandardScaler(),
        []int{0, 1, 2}, // Numerical columns
    },
    ColumnTransform{
        "categorical",
        preprocessing.NewOneHotEncoder(),
        []int{3, 4}, // Categorical columns
    },
)
```

## Data Validation

### Input Validation

```go
func validateDataset(X mat.Matrix) error {
    rows, cols := X.Dims()
    
    // Check dimensions
    if rows == 0 || cols == 0 {
        return fmt.Errorf("empty dataset")
    }
    
    // Check for NaN/Inf
    nanCount := 0
    infCount := 0
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            val := X.At(i, j)
            if math.IsNaN(val) {
                nanCount++
            }
            if math.IsInf(val, 0) {
                infCount++
            }
        }
    }
    
    if nanCount > 0 {
        fmt.Printf("Warning: %d NaN values found\n", nanCount)
    }
    if infCount > 0 {
        fmt.Printf("Warning: %d Inf values found\n", infCount)
    }
    
    // Check variance
    for j := 0; j < cols; j++ {
        column := mat.Col(nil, j, X)
        if variance(column) == 0 {
            fmt.Printf("Warning: Feature %d has zero variance\n", j)
        }
    }
    
    return nil
}
```

## Performance Optimization

### Sparse Data Handling

```go
type SparseScaler struct {
    mean     map[int]float64
    scale    map[int]float64
    withMean bool
}

func (ss *SparseScaler) FitTransform(X SparseMatrix) SparseMatrix {
    // Only process non-zero values
    for _, entry := range X.NonZero() {
        col := entry.Col
        
        if _, exists := ss.mean[col]; !exists {
            ss.mean[col] = 0
            ss.scale[col] = 1
        }
        
        // Update statistics
        // ...
    }
    
    // Transform
    result := NewSparseMatrix(X.Rows(), X.Cols())
    for _, entry := range X.NonZero() {
        scaled := (entry.Value - ss.mean[entry.Col]) / ss.scale[entry.Col]
        if scaled != 0 {
            result.Set(entry.Row, entry.Col, scaled)
        }
    }
    
    return result
}
```

### Batch Processing

```go
func batchPreprocess(dataStream <-chan mat.Matrix, transformer Transformer) <-chan mat.Matrix {
    output := make(chan mat.Matrix)
    
    go func() {
        defer close(output)
        
        first := true
        for batch := range dataStream {
            var processed mat.Matrix
            var err error
            
            if first {
                processed, err = transformer.FitTransform(batch)
                first = false
            } else {
                processed, err = transformer.Transform(batch)
            }
            
            if err != nil {
                log.Printf("Preprocessing error: %v", err)
                continue
            }
            
            output <- processed
        }
    }()
    
    return output
}
```

## Best Practices

1. **Fit on Training Only**: Never fit transformers on test data
2. **Pipeline Everything**: Use pipelines for reproducibility
3. **Handle Missing Data First**: Before other preprocessing
4. **Scale After Splitting**: To prevent data leakage
5. **Save Transformers**: For consistent preprocessing in production
6. **Validate Assumptions**: Check data distributions
7. **Document Preprocessing**: Track all transformations
8. **Test Invertibility**: Ensure transforms can be reversed if needed

## Common Pitfalls

1. **Data Leakage**: Fitting on test data
2. **Wrong Order**: Scaling before outlier removal
3. **Inconsistent Preprocessing**: Different steps for train/test
4. **Losing Information**: Over-aggressive outlier removal
5. **Memory Issues**: Not using sparse formats for sparse data

## Next Steps

- Explore [Feature Engineering](./feature-engineering.md)
- Learn about [Pipeline Guide](./pipeline.md)
- See [Preprocessing Examples](../../examples/preprocessing/)
- Read [API Reference](../api/preprocessing.md)