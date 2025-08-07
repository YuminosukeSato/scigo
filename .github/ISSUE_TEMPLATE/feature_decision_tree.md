---
name: Feature Request - DecisionTree Implementation
about: Implement DecisionTree for v0.4.0
title: '[FEATURE] Implement DecisionTree'
labels: enhancement, v0.4.0, algorithm
assignees: ''
---

## Feature Description

Implement `DecisionTreeClassifier` and `DecisionTreeRegressor` with full scikit-learn API compatibility for v0.4.0 release.

## Requirements

### Core Functionality
- [ ] DecisionTreeClassifier for classification tasks
- [ ] DecisionTreeRegressor for regression tasks
- [ ] CART (Classification and Regression Trees) algorithm
- [ ] Feature importance calculation
- [ ] Tree visualization/export capabilities

### Splitting Criteria
#### Classification
- [ ] Gini impurity (default)
- [ ] Entropy (information gain)
- [ ] Log loss

#### Regression
- [ ] MSE (Mean Squared Error) - default
- [ ] MAE (Mean Absolute Error)
- [ ] Poisson deviance

### API Methods
```go
type DecisionTreeClassifier struct {
    core.BaseEstimator
    tree *TreeNode
}

type DecisionTreeRegressor struct {
    core.BaseEstimator
    tree *TreeNode
}

// Required methods for both
Fit(X, y mat.Matrix) error
Predict(X mat.Matrix) (mat.Matrix, error)
Score(X, y mat.Matrix) float64
GetParams() map[string]interface{}
SetParams(params map[string]interface{}) error

// Classifier specific
PredictProba(X mat.Matrix) (mat.Matrix, error)
PredictLogProba(X mat.Matrix) (mat.Matrix, error)

// Common methods
GetDepth() int
GetNLeaves() int
GetFeatureImportances() []float64
ExportTree() *TreeStructure
```

### Parameters
- `criterion`: string (gini/entropy for classification, mse/mae for regression)
- `max_depth`: int, default=nil (unlimited)
- `min_samples_split`: int, default=2
- `min_samples_leaf`: int, default=1
- `max_features`: int/string, default='auto'
- `max_leaf_nodes`: int, default=nil
- `min_impurity_decrease`: float64, default=0.0
- `random_state`: int, default=nil
- `splitter`: string, default='best' (best/random)

## Implementation Plan

1. **Tree Structure** (Day 1-3)
   - Create `sklearn/tree/tree.go`
   - Implement TreeNode structure
   - Basic tree building logic

2. **Splitting Logic** (Day 4-7)
   - Implement Gini and Entropy criteria
   - Best split finding algorithm
   - Handle numerical and categorical features

3. **Classifier Implementation** (Week 2)
   - DecisionTreeClassifier
   - Probability predictions
   - Multi-class support

4. **Regressor Implementation** (Week 2)
   - DecisionTreeRegressor
   - MSE and MAE criteria
   - Leaf value computation

5. **Advanced Features** (Week 3)
   - Feature importance calculation
   - Tree pruning (minimal cost-complexity)
   - Tree export/visualization

6. **Testing & Documentation** (Week 3)
   - Comprehensive unit tests
   - Performance benchmarks
   - Documentation and examples

## Tree Structure Design

```go
type TreeNode struct {
    IsLeaf       bool
    Feature      int      // Feature index for split
    Threshold    float64  // Split threshold
    Value        float64  // Leaf value (for regression)
    Classes      []int    // Class counts (for classification)
    Left         *TreeNode
    Right        *TreeNode
    Impurity     float64
    NSamples     int
    WeightedNSamples float64
}

type TreeBuilder struct {
    criterion    Criterion
    maxDepth     int
    minSamplesSplit int
    minSamplesLeaf  int
    maxFeatures  int
    randomState  *rand.Rand
}
```

## Testing Requirements

- [ ] Unit tests for all splitting criteria
- [ ] Test on standard datasets (Iris, Boston Housing)
- [ ] Cross-validation with scikit-learn outputs
- [ ] Performance benchmarks
- [ ] Edge cases (single sample, all same class, etc.)
- [ ] Memory usage tests for large trees

## Visualization Support

```go
// Export to DOT format for Graphviz
func (dt *DecisionTree) ExportGraphviz() string

// Export to JSON for web visualization
func (dt *DecisionTree) ExportJSON() []byte

// Text representation
func (dt *DecisionTree) ExportText() string
```

## Documentation

- [ ] Godoc for all public methods
- [ ] Tutorial on decision tree usage
- [ ] Comparison with other tree implementations
- [ ] Feature importance interpretation guide

## Success Criteria

- Identical tree structure to scikit-learn with same parameters
- Performance 3-5x faster than Python implementation
- Memory efficient for deep trees
- Full API compatibility
- Test coverage >95%

## Benchmarks

Target performance on standard datasets:

| Dataset | Size | Features | Target Time | scikit-learn Time |
|---------|------|----------|-------------|-------------------|
| Iris | 150 | 4 | <1ms | 3ms |
| Wine | 178 | 13 | <2ms | 5ms |
| Breast Cancer | 569 | 30 | <5ms | 15ms |
| Digits | 1797 | 64 | <20ms | 60ms |

## References

- [scikit-learn DecisionTree](https://scikit-learn.org/stable/modules/tree.html)
- [CART Algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [C4.5 Algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm)
- [Tree Visualization Best Practices](https://github.com/parrt/dtreeviz)