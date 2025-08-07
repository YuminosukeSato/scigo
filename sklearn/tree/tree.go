package tree

import (
	"fmt"
	"math"
	"sort"

	"github.com/YuminosukeSato/scigo/core/model"
	"gonum.org/v1/gonum/mat"
)

// TreeNode represents a node in the decision tree
type TreeNode struct {
	IsLeaf       bool        // Whether this is a leaf node
	Feature      int         // Feature index for split (internal nodes)
	Threshold    float64     // Threshold value for split (internal nodes)
	Left         *TreeNode   // Left child (values <= threshold)
	Right        *TreeNode   // Right child (values > threshold)
	Value        float64     // Predicted value (leaf nodes - regression)
	ClassCounts  []int       // Class counts (leaf nodes - classification)
	PredictClass int         // Predicted class (leaf nodes - classification)
	Impurity     float64     // Node impurity
	NSamples     int         // Number of samples at this node
	Depth        int         // Depth of this node in the tree
}

// DecisionTreeClassifier implements a decision tree for classification
type DecisionTreeClassifier struct {
	state *model.StateManager // State management

	// Hyperparameters
	criterion         string  // Splitting criterion: "gini", "entropy"
	maxDepth          int     // Maximum depth of tree (0 = unlimited)
	minSamplesSplit   int     // Minimum samples to split a node
	minSamplesLeaf    int     // Minimum samples in a leaf
	maxFeatures       string  // Number of features to consider: "auto", "sqrt", "log2", or integer
	maxLeafNodes      int     // Maximum number of leaf nodes (0 = unlimited)
	minImpurityDecrease float64 // Minimum impurity decrease for split
	randomState       int64   // Random seed

	// Tree structure
	tree_      *TreeNode // Root of the tree
	nClasses_  int       // Number of classes
	nFeatures_ int       // Number of features
	classes_   []int     // Unique class labels
	
	// Feature importance
	featureImportances_ []float64 // Feature importance scores
}

// DecisionTreeClassifierOption is a functional option
type DecisionTreeClassifierOption func(*DecisionTreeClassifier)

// NewDecisionTreeClassifier creates a new decision tree classifier
func NewDecisionTreeClassifier(opts ...DecisionTreeClassifierOption) *DecisionTreeClassifier {
	dt := &DecisionTreeClassifier{
		state:               model.NewStateManager(),
		criterion:           "gini",
		maxDepth:            0, // Unlimited
		minSamplesSplit:     2,
		minSamplesLeaf:      1,
		maxFeatures:         "auto",
		maxLeafNodes:        0, // Unlimited
		minImpurityDecrease: 0.0,
		randomState:         -1,
	}

	// Apply options
	for _, opt := range opts {
		opt(dt)
	}

	return dt
}

// Option functions

// WithCriterion sets the splitting criterion
func WithCriterion(criterion string) DecisionTreeClassifierOption {
	return func(dt *DecisionTreeClassifier) {
		dt.criterion = criterion
	}
}

// WithMaxDepth sets the maximum tree depth
func WithMaxDepth(depth int) DecisionTreeClassifierOption {
	return func(dt *DecisionTreeClassifier) {
		dt.maxDepth = depth
	}
}

// WithMinSamplesSplit sets minimum samples to split
func WithMinSamplesSplit(n int) DecisionTreeClassifierOption {
	return func(dt *DecisionTreeClassifier) {
		dt.minSamplesSplit = n
	}
}

// WithMinSamplesLeaf sets minimum samples in leaf
func WithMinSamplesLeaf(n int) DecisionTreeClassifierOption {
	return func(dt *DecisionTreeClassifier) {
		dt.minSamplesLeaf = n
	}
}

// WithDTRandomState sets the random seed
func WithDTRandomState(seed int64) DecisionTreeClassifierOption {
	return func(dt *DecisionTreeClassifier) {
		dt.randomState = seed
	}
}

// Fit trains the decision tree
func (dt *DecisionTreeClassifier) Fit(X, y mat.Matrix) error {
	// Validate inputs
	nSamples, nFeatures := X.Dims()
	yRows, yCols := y.Dims()

	if nSamples != yRows {
		return fmt.Errorf("x and y must have same number of samples: got %d and %d", nSamples, yRows)
	}

	if yCols != 1 {
		return fmt.Errorf("y must be a column vector: got shape (%d, %d)", yRows, yCols)
	}

	// Extract classes
	dt.extractClasses(y)
	dt.nFeatures_ = nFeatures
	
	// Initialize feature importances
	dt.featureImportances_ = make([]float64, nFeatures)

	// Convert y to class indices
	yIndices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		label := int(y.At(i, 0))
		for j, class := range dt.classes_ {
			if class == label {
				yIndices[i] = j
				break
			}
		}
	}

	// Build the tree
	dt.tree_ = dt.buildTree(X, yIndices, 0)

	// Normalize feature importances
	dt.normalizeFeatureImportances()

	dt.state.SetFitted()
	return nil
}

// extractClasses identifies unique class labels
func (dt *DecisionTreeClassifier) extractClasses(y mat.Matrix) {
	rows, _ := y.Dims()
	classMap := make(map[int]bool)

	for i := 0; i < rows; i++ {
		label := int(y.At(i, 0))
		classMap[label] = true
	}

	dt.classes_ = make([]int, 0, len(classMap))
	for class := range classMap {
		dt.classes_ = append(dt.classes_, class)
	}

	// Sort for consistency
	sort.Ints(dt.classes_)
	dt.nClasses_ = len(dt.classes_)
}

// buildTree recursively builds the decision tree
func (dt *DecisionTreeClassifier) buildTree(X mat.Matrix, y []int, depth int) *TreeNode {
	nSamples, _ := X.Dims()
	
	// Count samples per class
	classCounts := make([]int, dt.nClasses_)
	for _, classIdx := range y {
		classCounts[classIdx]++
	}

	// Find majority class
	maxCount := 0
	predictClass := 0
	for i, count := range classCounts {
		if count > maxCount {
			maxCount = count
			predictClass = i
		}
	}

	// Calculate impurity
	impurity := dt.calculateImpurity(classCounts)

	// Create node
	node := &TreeNode{
		ClassCounts:  classCounts,
		PredictClass: predictClass,
		Impurity:     impurity,
		NSamples:     nSamples,
		Depth:        depth,
	}

	// Check stopping criteria
	if dt.shouldStop(nSamples, impurity, depth) {
		node.IsLeaf = true
		return node
	}

	// Find best split
	bestFeature, bestThreshold, bestImpurityDecrease := dt.findBestSplit(X, y, impurity)

	// Check if we found a valid split
	if bestFeature == -1 || bestImpurityDecrease < dt.minImpurityDecrease {
		node.IsLeaf = true
		return node
	}

	// Split the data
	leftIndices, rightIndices := dt.splitData(X, y, bestFeature, bestThreshold)

	// Check minimum samples in children
	if len(leftIndices) < dt.minSamplesLeaf || len(rightIndices) < dt.minSamplesLeaf {
		node.IsLeaf = true
		return node
	}

	// Create split node
	node.Feature = bestFeature
	node.Threshold = bestThreshold

	// Update feature importance
	dt.featureImportances_[bestFeature] += bestImpurityDecrease * float64(nSamples)

	// Build child nodes
	leftX, leftY := dt.getSubset(X, y, leftIndices)
	rightX, rightY := dt.getSubset(X, y, rightIndices)

	node.Left = dt.buildTree(leftX, leftY, depth+1)
	node.Right = dt.buildTree(rightX, rightY, depth+1)

	return node
}

// shouldStop checks stopping criteria
func (dt *DecisionTreeClassifier) shouldStop(nSamples int, impurity float64, depth int) bool {
	// Check max depth
	if dt.maxDepth > 0 && depth >= dt.maxDepth {
		return true
	}

	// Check minimum samples for split
	if nSamples < dt.minSamplesSplit {
		return true
	}

	// Check if pure node
	if impurity == 0.0 {
		return true
	}

	return false
}

// calculateImpurity calculates node impurity using Gini or Entropy
func (dt *DecisionTreeClassifier) calculateImpurity(classCounts []int) float64 {
	total := 0
	for _, count := range classCounts {
		total += count
	}

	if total == 0 {
		return 0.0
	}

	impurity := 0.0

	switch dt.criterion {
	case "gini":
		// Gini impurity: 1 - sum(p_i^2)
		sumSquared := 0.0
		for _, count := range classCounts {
			if count > 0 {
				p := float64(count) / float64(total)
				sumSquared += p * p
			}
		}
		impurity = 1.0 - sumSquared

	case "entropy":
		// Entropy: -sum(p_i * log2(p_i))
		for _, count := range classCounts {
			if count > 0 {
				p := float64(count) / float64(total)
				impurity -= p * math.Log2(p)
			}
		}

	default:
		// Default to Gini
		sumSquared := 0.0
		for _, count := range classCounts {
			if count > 0 {
				p := float64(count) / float64(total)
				sumSquared += p * p
			}
		}
		impurity = 1.0 - sumSquared
	}

	return impurity
}

// findBestSplit finds the best feature and threshold to split on
func (dt *DecisionTreeClassifier) findBestSplit(X mat.Matrix, y []int, parentImpurity float64) (int, float64, float64) {
	nSamples, nFeatures := X.Dims()
	bestFeature := -1
	bestThreshold := 0.0
	bestImpurityDecrease := 0.0

	for feature := 0; feature < nFeatures; feature++ {
		// Get unique values for this feature
		values := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			values[i] = X.At(i, feature)
		}

		// Sort values
		sortedIndices := make([]int, nSamples)
		for i := range sortedIndices {
			sortedIndices[i] = i
		}
		sort.Slice(sortedIndices, func(i, j int) bool {
			return values[sortedIndices[i]] < values[sortedIndices[j]]
		})

		// Try each potential threshold
		for i := 0; i < nSamples-1; i++ {
			idx1 := sortedIndices[i]
			idx2 := sortedIndices[i+1]

			// Skip if values are the same
			if values[idx1] == values[idx2] {
				continue
			}

			// Threshold is midpoint
			threshold := (values[idx1] + values[idx2]) / 2.0

			// Split samples
			leftCounts := make([]int, dt.nClasses_)
			rightCounts := make([]int, dt.nClasses_)
			nLeft := 0
			nRight := 0

			for j := 0; j < nSamples; j++ {
				if X.At(j, feature) <= threshold {
					leftCounts[y[j]]++
					nLeft++
				} else {
					rightCounts[y[j]]++
					nRight++
				}
			}

			// Skip if split doesn't satisfy min samples
			if nLeft < dt.minSamplesLeaf || nRight < dt.minSamplesLeaf {
				continue
			}

			// Calculate impurity decrease
			leftImpurity := dt.calculateImpurity(leftCounts)
			rightImpurity := dt.calculateImpurity(rightCounts)

			weightedImpurity := (float64(nLeft)*leftImpurity + float64(nRight)*rightImpurity) / float64(nSamples)
			impurityDecrease := parentImpurity - weightedImpurity

			if impurityDecrease > bestImpurityDecrease {
				bestImpurityDecrease = impurityDecrease
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestImpurityDecrease
}

// splitData splits data based on feature and threshold
func (dt *DecisionTreeClassifier) splitData(X mat.Matrix, y []int, feature int, threshold float64) ([]int, []int) {
	nSamples, _ := X.Dims()
	var leftIndices, rightIndices []int

	for i := 0; i < nSamples; i++ {
		if X.At(i, feature) <= threshold {
			leftIndices = append(leftIndices, i)
		} else {
			rightIndices = append(rightIndices, i)
		}
	}

	return leftIndices, rightIndices
}

// getSubset gets subset of data based on indices
func (dt *DecisionTreeClassifier) getSubset(X mat.Matrix, y []int, indices []int) (mat.Matrix, []int) {
	_, nFeatures := X.Dims()
	nSubSamples := len(indices)

	// Create subset X
	subX := mat.NewDense(nSubSamples, nFeatures, nil)
	for i, idx := range indices {
		for j := 0; j < nFeatures; j++ {
			subX.Set(i, j, X.At(idx, j))
		}
	}

	// Create subset y
	subY := make([]int, nSubSamples)
	for i, idx := range indices {
		subY[i] = y[idx]
	}

	return subX, subY
}

// normalizeFeatureImportances normalizes feature importance scores
func (dt *DecisionTreeClassifier) normalizeFeatureImportances() {
	sum := 0.0
	for _, imp := range dt.featureImportances_ {
		sum += imp
	}

	if sum > 0 {
		for i := range dt.featureImportances_ {
			dt.featureImportances_[i] /= sum
		}
	}
}

// Predict makes predictions for input data
func (dt *DecisionTreeClassifier) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !dt.state.IsFitted() {
		return nil, fmt.Errorf("model must be fitted before prediction")
	}

	nSamples, _ := X.Dims()
	predictions := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		node := dt.tree_
		
		// Traverse tree to leaf
		for !node.IsLeaf {
			if X.At(i, node.Feature) <= node.Threshold {
				node = node.Left
			} else {
				node = node.Right
			}
		}

		// Get predicted class
		predictions.Set(i, 0, float64(dt.classes_[node.PredictClass]))
	}

	return predictions, nil
}

// PredictProba returns probability estimates for each class
func (dt *DecisionTreeClassifier) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	if !dt.state.IsFitted() {
		return nil, fmt.Errorf("model must be fitted before prediction")
	}

	nSamples, _ := X.Dims()
	probas := mat.NewDense(nSamples, dt.nClasses_, nil)

	for i := 0; i < nSamples; i++ {
		node := dt.tree_

		// Traverse tree to leaf
		for !node.IsLeaf {
			if X.At(i, node.Feature) <= node.Threshold {
				node = node.Left
			} else {
				node = node.Right
			}
		}

		// Calculate probabilities from class counts
		total := 0
		for _, count := range node.ClassCounts {
			total += count
		}

		for j := 0; j < dt.nClasses_; j++ {
			if total > 0 {
				probas.Set(i, j, float64(node.ClassCounts[j])/float64(total))
			}
		}
	}

	return probas, nil
}

// Score returns the mean accuracy on the given test data
func (dt *DecisionTreeClassifier) Score(X, y mat.Matrix) float64 {
	predictions, err := dt.Predict(X)
	if err != nil {
		return 0.0
	}

	nSamples, _ := X.Dims()
	correct := 0

	for i := 0; i < nSamples; i++ {
		if predictions.At(i, 0) == y.At(i, 0) {
			correct++
		}
	}

	return float64(correct) / float64(nSamples)
}

// GetParams returns the model hyperparameters
func (dt *DecisionTreeClassifier) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"criterion":             dt.criterion,
		"max_depth":             dt.maxDepth,
		"min_samples_split":     dt.minSamplesSplit,
		"min_samples_leaf":      dt.minSamplesLeaf,
		"max_features":          dt.maxFeatures,
		"max_leaf_nodes":        dt.maxLeafNodes,
		"min_impurity_decrease": dt.minImpurityDecrease,
		"random_state":          dt.randomState,
	}
}

// SetParams sets the model hyperparameters
func (dt *DecisionTreeClassifier) SetParams(params map[string]interface{}) error {
	for key, value := range params {
		switch key {
		case "criterion":
			dt.criterion = value.(string)
		case "max_depth":
			dt.maxDepth = value.(int)
		case "min_samples_split":
			dt.minSamplesSplit = value.(int)
		case "min_samples_leaf":
			dt.minSamplesLeaf = value.(int)
		case "max_features":
			dt.maxFeatures = value.(string)
		case "max_leaf_nodes":
			dt.maxLeafNodes = value.(int)
		case "min_impurity_decrease":
			dt.minImpurityDecrease = value.(float64)
		case "random_state":
			dt.randomState = value.(int64)
		default:
			return fmt.Errorf("unknown parameter: %s", key)
		}
	}
	return nil
}

// GetFeatureImportances returns feature importance scores
func (dt *DecisionTreeClassifier) GetFeatureImportances() []float64 {
	if dt.featureImportances_ == nil {
		return nil
	}
	
	// Return copy
	importances := make([]float64, len(dt.featureImportances_))
	copy(importances, dt.featureImportances_)
	return importances
}

// GetDepth returns the depth of the tree
func (dt *DecisionTreeClassifier) GetDepth() int {
	if dt.tree_ == nil {
		return 0
	}
	return dt.getMaxDepth(dt.tree_)
}

// getMaxDepth recursively finds maximum depth
func (dt *DecisionTreeClassifier) getMaxDepth(node *TreeNode) int {
	if node == nil || node.IsLeaf {
		return node.Depth
	}

	leftDepth := dt.getMaxDepth(node.Left)
	rightDepth := dt.getMaxDepth(node.Right)

	if leftDepth > rightDepth {
		return leftDepth
	}
	return rightDepth
}

// GetNLeaves returns the number of leaf nodes
func (dt *DecisionTreeClassifier) GetNLeaves() int {
	if dt.tree_ == nil {
		return 0
	}
	return dt.countLeaves(dt.tree_)
}

// countLeaves recursively counts leaf nodes
func (dt *DecisionTreeClassifier) countLeaves(node *TreeNode) int {
	if node == nil {
		return 0
	}
	if node.IsLeaf {
		return 1
	}
	return dt.countLeaves(node.Left) + dt.countLeaves(node.Right)
}