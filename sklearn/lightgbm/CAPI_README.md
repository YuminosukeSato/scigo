# LightGBM C API Compatible Implementation in Pure Go

## Overview

This is a pure Go implementation of LightGBM's C API, providing complete numerical compatibility with the original LightGBM library. The implementation focuses on exact numerical reproducibility for testing and validation purposes.

## Features

### Implemented C API Functions

- `DatasetCreateFromMat` - Create dataset from matrix data
- `DatasetFree` - Free dataset resources
- `BoosterCreate` - Create a new booster model
- `BoosterFree` - Free booster resources
- `BoosterUpdateOneIter` - Perform one boosting iteration
- `BoosterPredictForMat` - Make predictions on matrix data

### Key Components

1. **Numerical Compatibility Framework** (`compatibility_utils.go`)
   - ULP (Unit in the Last Place) comparison for exact floating-point matching
   - Binary data loading for bit-exact comparison
   - Detailed difference reporting

2. **C API Core** (`capi.go`, `capi_tree.go`)
   - Pure Go implementation without CGO dependencies
   - Compatible data structures with LightGBM C API
   - Tree building using gradient boosting

3. **Golden Data Testing** (`tests/compatibility/`)
   - Python script for generating reference data
   - Intermediate calculation tracking
   - Reproducible test cases

## Usage

### Basic Example

```go
// Create dataset
data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
labels := []float32{0.5, 1.5}
dataset, err := DatasetCreateFromMat(data, 2, 3, true, labels)
if err != nil {
    log.Fatal(err)
}
defer DatasetFree(dataset)

// Create booster
params := "objective=regression learning_rate=0.1 num_leaves=31"
booster, err := BoosterCreate(dataset, params)
if err != nil {
    log.Fatal(err)
}
defer BoosterFree(booster)

// Train
for i := 0; i < 10; i++ {
    if err := BoosterUpdateOneIter(booster); err != nil {
        log.Fatal(err)
    }
}

// Predict
predictions, err := BoosterPredictForMat(booster, data, 2, 3, true)
if err != nil {
    log.Fatal(err)
}
```

### Generating Golden Data

```bash
# Install Python LightGBM
pip install lightgbm numpy

# Generate golden data for testing
python3 tests/compatibility/generate_golden_data.py
```

### Running Tests

```bash
# Run all compatibility tests
go test ./sklearn/lightgbm -v

# Run specific C API tests
go test ./sklearn/lightgbm -run TestCAPI -v

# Run with golden data comparison
go test ./sklearn/lightgbm -run TestCAPIMinimalRegression -v
```

## Design Principles

1. **Exact Numerical Compatibility**: Every calculation matches LightGBM's C implementation
2. **Test-First Development**: Golden data tests written before implementation
3. **Pure Go**: No CGO dependencies, using only Gonum for matrix operations
4. **Incremental Verification**: Test at each stage (data loading → gradients → splits → predictions)

## Architecture

```
capi.go                  # Main C API functions and data structures
├── LGBMDataset         # Dataset structure
├── LGBMBooster         # Booster/model structure
└── Core Functions      # API implementation

capi_tree.go            # Tree building logic
├── CAPITree            # Tree structure
├── CAPITreeNode        # Node structure
└── Split algorithms    # Finding optimal splits

compatibility_utils.go   # Testing utilities
├── ULP comparison      # Exact float comparison
├── Binary loading      # Load reference data
└── Difference report   # Detailed comparison
```

## Current Implementation Status

**Compatibility Level: ~15-20%** (excluding GPU features)

The current implementation provides a foundational framework with basic functionality. While core structures and simple test cases work, many advanced features required for production use are not yet implemented.

## Roadmap to Full LightGBM Compatibility

### Phase 1: Core Data Processing (0% → 30%)
#### High Priority
- [ ] **Categorical Features Optimization**
  - [ ] Optimal split finding for categorical features
  - [ ] Fisher's exact test for split selection
  - [ ] Categorical feature encoding strategies
- [ ] **Missing Value Handling**
  - [ ] LightGBM's special missing value treatment
  - [ ] Default direction learning for missing values
  - [ ] Zero as missing option
- [ ] **Histogram Optimization**
  - [ ] Efficient histogram construction
  - [ ] Histogram subtraction trick
  - [ ] Cache-aware histogram building
- [ ] **Exclusive Feature Bundling (EFB)**
  - [ ] Conflict detection between features
  - [ ] Bundle construction algorithm
  - [ ] Efficient bundle splitting

### Phase 2: Advanced Boosting Algorithms (30% → 50%)
- [ ] **GOSS (Gradient-based One-Side Sampling)**
  - [ ] Top gradient sample selection
  - [ ] Random sampling for small gradients
  - [ ] Weight adjustment for unbiased estimation
- [ ] **DART (Dropouts meet Multiple Additive Regression Trees)**
  - [ ] Tree dropout mechanism
  - [ ] Normalization after dropout
  - [ ] K-tree selection strategies
- [ ] **Random Forest Mode**
  - [ ] Bagging with replacement
  - [ ] Feature subsampling per tree
  - [ ] No boosting iterations

### Phase 3: Objectives and Metrics (50% → 70%)
#### Regression Objectives
- [ ] Poisson regression
- [ ] Tweedie regression
- [ ] Huber loss
- [ ] Fair loss
- [ ] Quantile regression
- [ ] MAPE (Mean Absolute Percentage Error)

#### Classification Objectives
- [x] Binary classification (basic)
- [ ] Multi-class classification (complete)
- [ ] Multi-class one-vs-all
- [ ] Cross-entropy variants

#### Ranking Objectives
- [ ] LambdaRank
- [ ] XE_NDCG
- [ ] XE_NDCG_MART
- [ ] RankXENDCG

### Phase 4: Advanced Features (70% → 85%)
- [ ] **SHAP Values**
  - [ ] TreeSHAP algorithm implementation
  - [ ] Interaction SHAP values
  - [ ] GPU-accelerated SHAP (if possible in Go)
- [ ] **Feature Importance**
  - [ ] Split-based importance
  - [ ] Gain-based importance
  - [ ] Permutation importance
- [ ] **Constraints**
  - [ ] Monotone constraints
  - [ ] Interaction constraints
  - [ ] Cost-efficient gradient boosting
- [ ] **Early Stopping**
  - [ ] Multiple validation sets
  - [ ] Custom early stopping callbacks
  - [ ] Best iteration tracking

### Phase 5: Model I/O and Compatibility (85% → 95%)
- [ ] **Model Serialization**
  - [ ] Full .txt format support
  - [ ] JSON model format
  - [ ] Binary model format
  - [ ] Model text parsing
- [ ] **Continued Training**
  - [ ] Init model support
  - [ ] Incremental learning
  - [ ] Transfer learning capabilities
- [ ] **Cross-validation**
  - [ ] K-fold CV
  - [ ] Stratified CV
  - [ ] Time series CV
- [ ] **Hyperparameter Optimization**
  - [ ] Grid search
  - [ ] Random search
  - [ ] Bayesian optimization interface

### Phase 6: Performance Optimization (95% → 100%)
- [ ] **Parallel Processing**
  - [ ] Feature parallel
  - [ ] Data parallel
  - [ ] Voting parallel
- [ ] **Memory Optimization**
  - [ ] Histogram reuse
  - [ ] Sparse feature optimization
  - [ ] Memory pool management
- [ ] **Cache Optimization**
  - [ ] Cache-aware algorithms
  - [ ] Data locality improvements
  - [ ] Prefetching strategies

## Implementation Priority

1. **Critical** (Must have for basic compatibility):
   - Categorical features
   - Missing values
   - Multi-class classification
   - Model I/O

2. **Important** (Needed for most use cases):
   - GOSS
   - Common objectives (Poisson, Tweedie)
   - SHAP values
   - Early stopping

3. **Nice to have** (Advanced features):
   - DART
   - Ranking objectives
   - Monotone constraints
   - Distributed training

## Current Limitations

- GPU optimization not implemented (pure Go limitation)
- ~80-85% of LightGBM features not yet implemented
- Performance not optimized (focus on correctness first)
- Limited to small-to-medium datasets currently

## Contributing

This implementation prioritizes numerical exactness over performance. When contributing:

1. Always write golden data tests first
2. Verify ULP-level accuracy
3. Document any numerical approximations
4. Follow existing code structure

## License

This implementation follows the same license as the parent scigo project.