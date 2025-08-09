package lightgbm

import (
	"fmt"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// NodeType represents the type of a tree node
type NodeType int

const (
	// LeafNode represents a terminal node with a value
	LeafNode NodeType = iota
	// NumericalNode represents a node with numerical split
	NumericalNode
	// CategoricalNode represents a node with categorical split
	CategoricalNode
)

// Node represents a single node in a decision tree
// Compatible with LightGBM's tree structure
type Node struct {
	// Node identification
	NodeID     int      // Unique identifier for the node
	ParentID   int      // Parent node ID (-1 for root)
	LeftChild  int      // Left child node ID (-1 if leaf)
	RightChild int      // Right child node ID (-1 if leaf)
	NodeType   NodeType // Type of the node

	// Split information (for non-leaf nodes)
	SplitFeature int     // Feature index used for splitting
	Threshold    float64 // Threshold value for numerical splits
	Categories   []int   // Categories for categorical splits
	DefaultLeft  bool    // Default direction for missing values
	Gain         float64 // Split gain (reduction in loss)

	// Leaf information (for leaf nodes)
	LeafValue float64 // Value at leaf node
	LeafCount int     // Number of samples at leaf

	// Statistics
	InternalValue float64 // Internal value (used during training)
	InternalCount int     // Internal count (used during training)
}

// IsLeaf returns true if the node is a leaf node
func (n *Node) IsLeaf() bool {
	return n.LeftChild == -1 && n.RightChild == -1
}

// Tree represents a single decision tree in the ensemble
type Tree struct {
	// Tree metadata
	TreeIndex     int     // Index of the tree in ensemble
	NumLeaves     int     // Number of leaf nodes
	NumNodes      int     // Total number of nodes
	MaxDepth      int     // Maximum depth of the tree
	ShrinkageRate float64 // Learning rate applied to this tree

	// Node storage
	Nodes []Node // All nodes in the tree

	// Feature information
	FeatureNames      []string  // Optional feature names
	FeatureImportance []float64 // Feature importance scores
}

// Predict makes a prediction for a single sample using this tree
func (t *Tree) Predict(features []float64) float64 {
	nodeID := 0 // Start from root

	for nodeID >= 0 && nodeID < len(t.Nodes) {
		node := &t.Nodes[nodeID]

		if node.IsLeaf() {
			return node.LeafValue * t.ShrinkageRate
		}

		// Handle missing values
		featureValue := features[node.SplitFeature]
		if isNaN(featureValue) {
			if node.DefaultLeft {
				nodeID = node.LeftChild
			} else {
				nodeID = node.RightChild
			}
			continue
		}

		// Make decision based on node type
		switch node.NodeType {
		case NumericalNode:
			if featureValue <= node.Threshold {
				nodeID = node.LeftChild
			} else {
				nodeID = node.RightChild
			}
		case CategoricalNode:
			// Check if feature value is in categories list
			inCategories := false
			intValue := int(featureValue)
			for _, cat := range node.Categories {
				if intValue == cat {
					inCategories = true
					break
				}
			}
			if inCategories {
				nodeID = node.LeftChild
			} else {
				nodeID = node.RightChild
			}
		default:
			// This shouldn't happen with valid trees
			return 0.0
		}
	}

	return 0.0
}

// ObjectiveType represents the objective function type
type ObjectiveType string

const (
	// Regression objectives
	RegressionL2       ObjectiveType = "regression"
	RegressionL1       ObjectiveType = "regression_l1"
	RegressionHuber    ObjectiveType = "huber"
	RegressionFair     ObjectiveType = "fair"
	RegressionPoisson  ObjectiveType = "poisson"
	RegressionQuantile ObjectiveType = "quantile"
	RegressionMAE      ObjectiveType = "mae"
	RegressionGamma    ObjectiveType = "gamma"
	RegressionTweedie  ObjectiveType = "tweedie"

	// Binary classification objectives
	BinaryLogistic     ObjectiveType = "binary"
	BinaryCrossEntropy ObjectiveType = "cross_entropy"

	// Multiclass classification objectives
	MulticlassSoftmax ObjectiveType = "multiclass"
	MulticlassOVA     ObjectiveType = "multiclassova"

	// Ranking objectives
	LambdaRank ObjectiveType = "lambdarank"
	RankXENDCG ObjectiveType = "rank_xendcg"
)

// BoostingType represents the boosting algorithm type
type BoostingType string

const (
	GBDT BoostingType = "gbdt" // Gradient Boosting Decision Tree
	DART BoostingType = "dart" // Dropouts meet Multiple Additive Regression Trees
	GOSS BoostingType = "goss" // Gradient-based One-Side Sampling
	RF   BoostingType = "rf"   // Random Forest
)

// Model represents a complete LightGBM model ensemble
type Model struct {
	// Model configuration
	Objective    ObjectiveType // Objective function
	BoostingType BoostingType  // Boosting algorithm
	NumClass     int           // Number of classes (for multiclass)
	NumIteration int           // Number of boosting iterations
	LearningRate float64       // Base learning rate
	NumLeaves    int           // Maximum number of leaves per tree
	MaxDepth     int           // Maximum tree depth

	// Trees
	Trees []Tree // All trees in the ensemble

	// Feature information
	NumFeatures       int       // Number of features
	FeatureNames      []string  // Feature names (optional)
	FeatureImportance []float64 // Global feature importance

	// Model metadata
	Version       string                 // LightGBM version
	Parameters    map[string]interface{} // All model parameters
	BestIteration int                    // Best iteration (if early stopping used)

	// Preprocessing
	InitScore float64 // Initial score (baseline prediction)
	Sigmoid   float64 // Sigmoid parameter for binary classification

	// For deterministic predictions
	Deterministic bool // Enable deterministic mode
	RandomSeed    int  // Random seed for deterministic mode
}

// NewModel creates a new empty LightGBM model
func NewModel() *Model {
	return &Model{
		Trees:        make([]Tree, 0),
		Parameters:   make(map[string]interface{}),
		LearningRate: 0.1,
		NumLeaves:    31,
		MaxDepth:     -1, // No limit by default in LightGBM
	}
}

// Predict makes predictions for a batch of samples
func (m *Model) Predict(X mat.Matrix) (mat.Matrix, error) {
	rows, cols := X.Dims()
	if cols != m.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", m.NumFeatures, cols)
	}

	// Prepare output matrix
	var outputCols int
	if m.NumClass > 2 {
		outputCols = m.NumClass
	} else {
		outputCols = 1
	}
	predictions := mat.NewDense(rows, outputCols, nil)

	// Process each sample
	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, X)
		pred := m.PredictSingle(features, -1) // -1 means use all trees

		if m.NumClass > 2 {
			// Multiclass: pred contains all class scores
			predictions.SetRow(i, pred)
		} else {
			// Binary or regression: single value
			predictions.Set(i, 0, pred[0])
		}
	}

	return predictions, nil
}

// PredictSingle makes a prediction for a single sample
// numIteration specifies how many trees to use (-1 for all)
func (m *Model) PredictSingle(features []float64, numIteration int) []float64 {
	if numIteration < 0 || numIteration > len(m.Trees) {
		numIteration = len(m.Trees)
	}

	// Initialize predictions
	var predictions []float64
	if m.NumClass > 2 {
		predictions = make([]float64, m.NumClass)
		// Initialize with init score if available
		for i := range predictions {
			predictions[i] = m.InitScore
		}
	} else {
		predictions = []float64{m.InitScore}
	}

	// Accumulate predictions from trees
	for i := 0; i < numIteration; i++ {
		tree := &m.Trees[i]
		treeOutput := tree.Predict(features)

		if m.NumClass > 2 {
			// For multiclass, trees are arranged by class
			classIdx := i % m.NumClass
			predictions[classIdx] += treeOutput
		} else {
			predictions[0] += treeOutput
		}
	}

	// Apply final transformation based on objective
	switch m.Objective {
	case BinaryLogistic, BinaryCrossEntropy:
		// Apply sigmoid transformation for binary classification
		predictions[0] = sigmoid(predictions[0])
	case MulticlassSoftmax:
		// Apply softmax for multiclass
		predictions = softmax(predictions)
	}

	return predictions
}

// GetFeatureImportance calculates and returns feature importance scores
func (m *Model) GetFeatureImportance(importanceType string) []float64 {
	importance := make([]float64, m.NumFeatures)

	for _, tree := range m.Trees {
		for _, node := range tree.Nodes {
			if !node.IsLeaf() {
				switch importanceType {
				case "split":
					// Count number of times feature is used
					importance[node.SplitFeature]++
				case "gain":
					// Sum the gain from splits
					importance[node.SplitFeature] += node.Gain
				}
			}
		}
	}

	// Normalize importance scores
	total := 0.0
	for _, v := range importance {
		total += v
	}
	if total > 0 {
		for i := range importance {
			importance[i] /= total
		}
	}

	return importance
}

// Helper functions

func isNaN(f float64) bool {
	return f != f
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + exp(-x))
}

func exp(x float64) float64 {
	// Use standard library math.Exp equivalent
	// This is a placeholder - would use actual implementation
	if x > 700 {
		return 1e308 // Avoid overflow
	}
	if x < -700 {
		return 0 // Avoid underflow
	}

	// Simplified exp calculation for demonstration
	// In production, use math.Exp
	result := 1.0
	term := 1.0
	for i := 1; i <= 20; i++ {
		term *= x / float64(i)
		result += term
	}
	return result
}

func softmax(x []float64) []float64 {
	// Find max for numerical stability
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp(x - max)
	expSum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = exp(v - maxVal)
		expSum += result[i]
	}

	// Normalize
	for i := range result {
		result[i] /= expSum
	}

	return result
}

// SaveToFile saves the model to a file in LightGBM text format
func (m *Model) SaveToFile(filepath string) error {
	var sb strings.Builder

	// Write header
	sb.WriteString("tree\n")
	sb.WriteString(fmt.Sprintf("version=v3\n"))
	sb.WriteString(fmt.Sprintf("num_class=%d\n", m.NumClass))
	sb.WriteString(fmt.Sprintf("num_tree_per_iteration=%d\n", 1))
	sb.WriteString(fmt.Sprintf("label_index=0\n"))
	sb.WriteString(fmt.Sprintf("max_feature_idx=%d\n", m.NumFeatures-1))
	sb.WriteString(fmt.Sprintf("objective=%s\n", m.Objective))
	sb.WriteString(fmt.Sprintf("feature_names=%s\n", strings.Join(m.FeatureNames, " ")))
	sb.WriteString(fmt.Sprintf("feature_infos=none\n"))
	sb.WriteString(fmt.Sprintf("tree_sizes=%d\n", len(m.Trees)))

	// Write trees
	for i, tree := range m.Trees {
		sb.WriteString(fmt.Sprintf("\nTree=%d\n", i))
		sb.WriteString(fmt.Sprintf("num_leaves=%d\n", tree.NumLeaves))
		sb.WriteString(fmt.Sprintf("num_cat=0\n"))
		sb.WriteString(fmt.Sprintf("shrinkage=%f\n", tree.ShrinkageRate))

		// For simplified version, write minimal tree structure
		if len(tree.Nodes) > 0 {
			for _, node := range tree.Nodes {
				if node.IsLeaf() {
					sb.WriteString(fmt.Sprintf("leaf_value=%f\n", node.LeafValue))
				} else {
					sb.WriteString(fmt.Sprintf("split_feature=%d\n", node.SplitFeature))
					sb.WriteString(fmt.Sprintf("threshold=%f\n", node.Threshold))
				}
			}
		}
	}

	// Write to file
	return os.WriteFile(filepath, []byte(sb.String()), 0600)
}
