package lightgbm

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// ErrNotFitted is returned when a model is not fitted
var ErrNotFitted = errors.New("model is not fitted")

// SHAPValues holds SHAP values for model interpretation
type SHAPValues struct {
	Values       *mat.Dense // SHAP values matrix (samples x features)
	BaseValue    float64    // Expected value (base value)
	FeatureNames []string   // Optional feature names
}

// TreeSHAP implements the TreeSHAP algorithm for LightGBM models
type TreeSHAP struct {
	model *Model
}

// NewTreeSHAP creates a new TreeSHAP calculator
func NewTreeSHAP(model *Model) *TreeSHAP {
	return &TreeSHAP{
		model: model,
	}
}

// CalculateSHAP calculates SHAP values for given samples
func (ts *TreeSHAP) CalculateSHAP(X mat.Matrix) (*SHAPValues, error) {
	rows, cols := X.Dims()

	// Initialize SHAP values matrix
	shapValues := mat.NewDense(rows, cols, nil)

	// Calculate base value (expected value)
	baseValue := ts.calculateBaseValue()

	// Calculate SHAP values for each sample
	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		shapRow := ts.calculateSampleSHAP(sample)
		shapValues.SetRow(i, shapRow)
	}

	return &SHAPValues{
		Values:    shapValues,
		BaseValue: baseValue,
	}, nil
}

// calculateBaseValue calculates the expected value (base value)
func (ts *TreeSHAP) calculateBaseValue() float64 {
	if ts.model == nil || len(ts.model.Trees) == 0 {
		return 0.0
	}

	// Base value should be just the initial score
	// The tree contributions are handled separately in SHAP values calculation
	// Adding tree average values here would double-count the tree contributions
	return ts.model.InitScore
}

// getTreeBaseline calculates the baseline (average) prediction of a tree
func (ts *TreeSHAP) getTreeBaseline(tree Tree) float64 {
	if len(tree.Nodes) == 0 {
		return 0.0
	}

	// Collect all leaf values
	var leafValues []float64
	ts.collectLeafValues(&tree.Nodes[0], tree, &leafValues)

	// Calculate simple average (in practice, this should be weighted by sample count)
	if len(leafValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range leafValues {
		sum += value
	}
	return sum / float64(len(leafValues))
}

// collectLeafValues recursively collects all leaf values from a tree
func (ts *TreeSHAP) collectLeafValues(node *Node, tree Tree, leafValues *[]float64) {
	if node == nil {
		return
	}

	if node.NodeType == LeafNode {
		*leafValues = append(*leafValues, node.LeafValue)
		return
	}

	// Recursively collect from children
	if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
		ts.collectLeafValues(&tree.Nodes[node.LeftChild], tree, leafValues)
	}
	if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
		ts.collectLeafValues(&tree.Nodes[node.RightChild], tree, leafValues)
	}
}

// calculateSampleSHAP calculates SHAP values for a single sample
func (ts *TreeSHAP) calculateSampleSHAP(sample []float64) []float64 {
	nFeatures := len(sample)
	shapValues := make([]float64, nFeatures)

	// Process each tree
	for _, tree := range ts.model.Trees {
		if len(tree.Nodes) == 0 {
			continue
		}

		// Get tree SHAP values (simplified implementation)
		treeShap := ts.simpleTreeShap(tree, sample)

		// Add to total SHAP values with learning rate
		for j := 0; j < nFeatures && j < len(treeShap); j++ {
			shapValues[j] += treeShap[j] * ts.model.LearningRate
		}
	}

	return shapValues
}

// simpleTreeShap calculates simplified SHAP values for a single tree
func (ts *TreeSHAP) simpleTreeShap(tree Tree, sample []float64) []float64 {
	nFeatures := len(sample)
	shapValues := make([]float64, nFeatures)

	if len(tree.Nodes) == 0 {
		return shapValues
	}

	// Get the prediction path through the tree
	path := ts.getTreePath(tree, sample)
	if len(path) == 0 {
		return shapValues
	}

	// Get the leaf value for this sample
	lastNodeIdx := path[len(path)-1]
	if lastNodeIdx < 0 || lastNodeIdx >= len(tree.Nodes) {
		return shapValues
	}
	leafValue := tree.Nodes[lastNodeIdx].LeafValue

	// Calculate average leaf value of the tree (baseline)
	baselineValue := ts.getTreeBaseline(tree)

	// Total contribution of this tree
	treeContribution := leafValue - baselineValue

	// Distribute the contribution among features in the path
	featureCounts := make(map[int]int)
	for i := 0; i < len(path)-1; i++ { // Exclude leaf node
		nodeIdx := path[i]
		if nodeIdx >= 0 && nodeIdx < len(tree.Nodes) {
			node := &tree.Nodes[nodeIdx]
			if node.NodeType != LeafNode && node.SplitFeature >= 0 && node.SplitFeature < nFeatures {
				featureCounts[node.SplitFeature]++
			}
		}
	}

	// Distribute contribution proportionally to feature usage in path
	if len(featureCounts) > 0 {
		for featureIdx, count := range featureCounts {
			shapValues[featureIdx] += treeContribution * float64(count) / float64(len(featureCounts))
		}
	}

	return shapValues
}

// getTreePath returns the path of nodes from root to leaf for given sample
func (ts *TreeSHAP) getTreePath(tree Tree, sample []float64) []int {
	path := make([]int, 0)

	if len(tree.Nodes) == 0 {
		return path
	}

	currentIdx := 0 // Start from root

	for currentIdx >= 0 && currentIdx < len(tree.Nodes) {
		node := &tree.Nodes[currentIdx]
		path = append(path, currentIdx)

		if node.NodeType == LeafNode {
			break
		}

		// Navigate to next node
		if node.SplitFeature >= 0 && node.SplitFeature < len(sample) {
			if len(node.Categories) > 0 {
				// Categorical split
				sampleValue := int(sample[node.SplitFeature])
				if contains(node.Categories, sampleValue) {
					currentIdx = node.LeftChild
				} else {
					currentIdx = node.RightChild
				}
			} else {
				// Numerical split
				if sample[node.SplitFeature] <= node.Threshold {
					currentIdx = node.LeftChild
				} else {
					currentIdx = node.RightChild
				}
			}
		} else {
			break
		}
	}

	return path
}

// getNodeValue gets the value of a node (simplified implementation)
func (ts *TreeSHAP) getNodeValue(node *Node) float64 {
	if node.NodeType == LeafNode {
		return node.LeafValue
	}

	// For non-leaf nodes, return internal value
	return node.InternalValue
}

// InteractionSHAP calculates SHAP interaction values
type InteractionSHAP struct {
	shap *TreeSHAP
}

// NewInteractionSHAP creates a new interaction SHAP calculator
func NewInteractionSHAP(model *Model) *InteractionSHAP {
	return &InteractionSHAP{
		shap: NewTreeSHAP(model),
	}
}

// CalculateInteractions calculates SHAP interaction values
func (is *InteractionSHAP) CalculateInteractions(X mat.Matrix) ([]*mat.Dense, error) {
	rows, cols := X.Dims()

	// Initialize interaction matrices for each sample
	interactions := make([]*mat.Dense, rows)

	for i := 0; i < rows; i++ {
		// Create interaction matrix for this sample
		interaction := mat.NewDense(cols, cols, nil)
		sample := mat.Row(nil, i, X)

		// Calculate interactions
		is.calculateSampleInteractions(sample, interaction)
		interactions[i] = interaction
	}

	return interactions, nil
}

// calculateSampleInteractions calculates interaction values for a single sample
func (is *InteractionSHAP) calculateSampleInteractions(sample []float64, interaction *mat.Dense) {
	nFeatures := len(sample)

	// Calculate main effects first
	mainEffects := is.shap.calculateSampleSHAP(sample)

	// Set diagonal to main effects
	for i := 0; i < nFeatures; i++ {
		interaction.Set(i, i, mainEffects[i])
	}

	// Calculate pairwise interactions
	for i := 0; i < nFeatures; i++ {
		for j := i + 1; j < nFeatures; j++ {
			// Calculate interaction between features i and j
			interactionValue := is.calculatePairwiseInteraction(sample, i, j)

			// Symmetric matrix
			interaction.Set(i, j, interactionValue)
			interaction.Set(j, i, interactionValue)
		}
	}
}

// calculatePairwiseInteraction calculates interaction between two features
func (is *InteractionSHAP) calculatePairwiseInteraction(sample []float64, feat1, feat2 int) float64 {
	// Simplified interaction calculation
	// In practice, this would involve more complex tree traversal

	interaction := 0.0

	for _, tree := range is.shap.model.Trees {
		if len(tree.Nodes) == 0 {
			continue
		}

		// Check if both features are used in this tree
		if is.treeUsesFeatures(&tree.Nodes[0], tree, feat1, feat2) {
			// Calculate interaction contribution
			contrib := is.treeInteractionContribution(&tree.Nodes[0], tree, sample, feat1, feat2)
			interaction += contrib * is.shap.model.LearningRate
		}
	}

	return interaction
}

// treeUsesFeatures checks if a tree uses both specified features
func (is *InteractionSHAP) treeUsesFeatures(node *Node, tree Tree, feat1, feat2 int) bool {
	if node == nil || node.NodeType == LeafNode {
		return false
	}

	usesFeat1 := node.SplitFeature == feat1
	usesFeat2 := node.SplitFeature == feat2

	// Check children
	leftUses1, leftUses2 := false, false
	rightUses1, rightUses2 := false, false

	if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
		leftUses1, leftUses2 = is.checkSubtreeFeatures(&tree.Nodes[node.LeftChild], tree, feat1, feat2)
	}
	if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
		rightUses1, rightUses2 = is.checkSubtreeFeatures(&tree.Nodes[node.RightChild], tree, feat1, feat2)
	}

	usesFeat1 = usesFeat1 || leftUses1 || rightUses1
	usesFeat2 = usesFeat2 || leftUses2 || rightUses2

	return usesFeat1 && usesFeat2
}

// checkSubtreeFeatures checks which features are used in a subtree
func (is *InteractionSHAP) checkSubtreeFeatures(node *Node, tree Tree, feat1, feat2 int) (bool, bool) {
	if node == nil || node.NodeType == LeafNode {
		return false, false
	}

	usesFeat1 := node.SplitFeature == feat1
	usesFeat2 := node.SplitFeature == feat2

	if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
		left1, left2 := is.checkSubtreeFeatures(&tree.Nodes[node.LeftChild], tree, feat1, feat2)
		usesFeat1 = usesFeat1 || left1
		usesFeat2 = usesFeat2 || left2
	}

	if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
		right1, right2 := is.checkSubtreeFeatures(&tree.Nodes[node.RightChild], tree, feat1, feat2)
		usesFeat1 = usesFeat1 || right1
		usesFeat2 = usesFeat2 || right2
	}

	return usesFeat1, usesFeat2
}

// treeInteractionContribution calculates the interaction contribution in a tree
func (is *InteractionSHAP) treeInteractionContribution(node *Node, tree Tree, sample []float64, feat1, feat2 int) float64 {
	if node == nil || node.NodeType == LeafNode {
		return 0.0
	}

	// Simplified interaction calculation
	// This is a placeholder for the actual TreeSHAP interaction algorithm

	contribution := 0.0

	// If this node splits on one of our features
	if node.SplitFeature == feat1 || node.SplitFeature == feat2 {
		// Calculate the contribution based on the path taken
		leftValue := 0.0
		rightValue := 0.0

		if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
			leftValue = is.shap.getNodeValue(&tree.Nodes[node.LeftChild])
		}
		if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
			rightValue = is.shap.getNodeValue(&tree.Nodes[node.RightChild])
		}

		// Simple difference as interaction
		contribution = math.Abs(leftValue-rightValue) * 0.1 // Scaled down
	}

	// Recursively calculate for children
	if len(node.Categories) > 0 {
		catValue := int(sample[node.SplitFeature])
		if contains(node.Categories, catValue) {
			if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
				contribution += is.treeInteractionContribution(&tree.Nodes[node.LeftChild], tree, sample, feat1, feat2)
			}
		} else {
			if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
				contribution += is.treeInteractionContribution(&tree.Nodes[node.RightChild], tree, sample, feat1, feat2)
			}
		}
	} else {
		if sample[node.SplitFeature] <= node.Threshold {
			if node.LeftChild >= 0 && node.LeftChild < len(tree.Nodes) {
				contribution += is.treeInteractionContribution(&tree.Nodes[node.LeftChild], tree, sample, feat1, feat2)
			}
		} else {
			if node.RightChild >= 0 && node.RightChild < len(tree.Nodes) {
				contribution += is.treeInteractionContribution(&tree.Nodes[node.RightChild], tree, sample, feat1, feat2)
			}
		}
	}

	return contribution
}

// PredictSHAP calculates SHAP values for predictions
func (lgb *LGBMRegressor) PredictSHAP(X mat.Matrix) (*SHAPValues, error) {
	if !lgb.IsFitted() {
		return nil, ErrNotFitted
	}

	shap := NewTreeSHAP(lgb.Model)
	return shap.CalculateSHAP(X)
}

// PredictSHAP calculates SHAP values for predictions
func (lgb *LGBMClassifier) PredictSHAP(X mat.Matrix) (*SHAPValues, error) {
	if !lgb.IsFitted() {
		return nil, ErrNotFitted
	}

	shap := NewTreeSHAP(lgb.Model)
	return shap.CalculateSHAP(X)
}

// contains checks if a slice contains a value
func contains(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}
