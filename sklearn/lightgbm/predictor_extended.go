package lightgbm

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// PredictRawScore returns raw prediction scores without any transformation
// For binary classification, returns raw scores (before sigmoid)
// For multiclass, returns raw scores (before softmax)
// For regression, returns the same as regular prediction
func (p *Predictor) PredictRawScore(X mat.Matrix) (mat.Matrix, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}

	// Determine output dimensions
	var outputCols int
	if p.model.NumClass > 2 {
		outputCols = p.model.NumClass
	} else {
		outputCols = 1
	}

	rawScores := mat.NewDense(rows, outputCols, nil)

	// Convert interface to concrete type
	var xDense *mat.Dense
	switch v := X.(type) {
	case *mat.Dense:
		xDense = v
	default:
		// If not a Dense matrix, convert it
		xDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				xDense.Set(i, j, X.At(i, j))
			}
		}
	}

	// Process each sample
	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, xDense)
		rawPred := p.predictSingleSampleRaw(features)

		if p.model.NumClass > 2 {
			rawScores.SetRow(i, rawPred)
		} else {
			rawScores.Set(i, 0, rawPred[0])
		}
	}

	return rawScores, nil
}

// PredictLeaf returns the leaf indices for each sample in each tree
// Output shape: (n_samples, n_trees)
func (p *Predictor) PredictLeaf(X mat.Matrix) (mat.Matrix, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}

	numTrees := len(p.model.Trees)
	leafIndices := mat.NewDense(rows, numTrees, nil)

	// Convert interface to concrete type
	var xDense *mat.Dense
	switch v := X.(type) {
	case *mat.Dense:
		xDense = v
	default:
		xDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				xDense.Set(i, j, X.At(i, j))
			}
		}
	}

	// Process each sample
	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, xDense)
		leaves := p.predictSingleSampleLeaves(features)

		for j, leafIdx := range leaves {
			leafIndices.Set(i, j, float64(leafIdx))
		}
	}

	return leafIndices, nil
}

// predictSingleSampleRaw predicts raw scores for a single sample without transformation
func (p *Predictor) predictSingleSampleRaw(features []float64) []float64 {
	// Initialize predictions
	var predictions []float64
	if p.model.NumClass > 2 {
		predictions = make([]float64, p.model.NumClass)
		// Initialize with init score if available
		for i := range predictions {
			predictions[i] = p.model.InitScore
		}
	} else {
		predictions = []float64{p.model.InitScore}
	}

	// Accumulate predictions from trees
	for i, tree := range p.model.Trees {
		treeOutput := tree.Predict(features)

		if p.model.NumClass > 2 {
			// For multiclass, trees are arranged by class
			classIdx := i % p.model.NumClass
			predictions[classIdx] += treeOutput
		} else {
			predictions[0] += treeOutput
		}
	}

	// Return raw scores without any transformation
	return predictions
}

// predictSingleSampleLeaves returns the leaf index for each tree for a single sample
func (p *Predictor) predictSingleSampleLeaves(features []float64) []int {
	leafIndices := make([]int, len(p.model.Trees))

	for i, tree := range p.model.Trees {
		leafIdx := predictTreeLeaf(&tree, features)
		leafIndices[i] = leafIdx
	}

	return leafIndices
}

// predictTreeLeaf returns the leaf index for a single tree and sample
func predictTreeLeaf(tree *Tree, features []float64) int {
	nodeID := 0 // Start from root
	leafCounter := 0
	leafMap := make(map[int]int) // Map from nodeID to leaf index

	// First, build leaf index map by traversing the tree
	var buildLeafMap func(nodeID int) int
	buildLeafMap = func(nodeID int) int {
		if nodeID < 0 || nodeID >= len(tree.Nodes) {
			return leafCounter
		}

		node := &tree.Nodes[nodeID]
		if node.IsLeaf() {
			leafMap[nodeID] = leafCounter
			leafCounter++
			return leafCounter
		}

		// Traverse left subtree
		if node.LeftChild >= 0 {
			buildLeafMap(node.LeftChild)
		} else {
			// Left child is a leaf
			leafMap[node.LeftChild] = leafCounter
			leafCounter++
		}

		// Traverse right subtree
		if node.RightChild >= 0 {
			buildLeafMap(node.RightChild)
		} else {
			// Right child is a leaf
			leafMap[node.RightChild] = leafCounter
			leafCounter++
		}

		return leafCounter
	}

	buildLeafMap(0)

	// Now traverse to find which leaf this sample lands in
	nodeID = 0
	for nodeID >= 0 && nodeID < len(tree.Nodes) {
		node := &tree.Nodes[nodeID]

		if node.IsLeaf() {
			return leafMap[nodeID]
		}

		// Get feature value
		featureValue := features[node.SplitFeature]

		// Handle missing values
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
			nodeID = node.LeftChild
		}
	}

	// If we ended on a negative nodeID (leaf reference), get its index
	if nodeID < 0 {
		return leafMap[nodeID]
	}

	return 0
}

// PredictRawScoreForLeavesModel provides raw score prediction for LeavesModel
func (m *LeavesModel) PredictRawScore(features []float64) []float64 {
	if m.NumClass > 2 {
		// Multiclass prediction
		predictions := make([]float64, m.NumClass)
		for i, tree := range m.Trees {
			classIdx := i % m.NumClass
			predictions[classIdx] += tree.Predict(features)
		}
		return predictions
	} else {
		// Binary or regression
		pred := m.InitScore
		for _, tree := range m.Trees {
			pred += tree.Predict(features)
		}
		return []float64{pred}
	}
}

// PredictLeafForLeavesModel returns leaf indices for LeavesModel
func (m *LeavesModel) PredictLeaf(features []float64) []int {
	leafIndices := make([]int, len(m.Trees))

	for i, tree := range m.Trees {
		leafIdx := predictLeavesTreeLeaf(&tree, features)
		leafIndices[i] = leafIdx
	}

	return leafIndices
}

// predictLeavesTreeLeaf returns the leaf index for a single LeavesTree
func predictLeavesTreeLeaf(tree *LeavesTree, features []float64) int {
	if len(tree.Nodes) == 0 {
		// Constant tree with single leaf
		return 0
	}

	idx := uint32(0)
	leafCounter := 0
	visitedLeaves := make(map[uint32]int)

	// Traverse the tree
	for {
		node := &tree.Nodes[idx]

		// Check if this is a leaf
		if node.Flags&leftLeaf != 0 {
			// Left child is a leaf
			if _, exists := visitedLeaves[node.Left]; !exists {
				visitedLeaves[node.Left] = leafCounter
				leafCounter++
			}
		}
		if node.Flags&rightLeaf != 0 {
			// Right child is a leaf
			if _, exists := visitedLeaves[node.Right]; !exists {
				visitedLeaves[node.Right] = leafCounter
				leafCounter++
			}
		}

		// Make decision
		featureVal := features[node.Feature]

		// Handle missing values
		missingType := node.Flags & (missingZero | missingNan)
		if (missingType == missingNan && isNaN(featureVal)) ||
			(missingType == missingZero && featureVal == 0.0) {
			// Use default direction
			if node.Flags&defaultLeft != 0 {
				if node.Flags&leftLeaf != 0 {
					return visitedLeaves[node.Left]
				}
				idx = node.Left
			} else {
				if node.Flags&rightLeaf != 0 {
					return visitedLeaves[node.Right]
				}
				idx = idx + 1 // Right child is at next index in leaves format
			}
			continue
		}

		// Check for categorical split
		if node.Flags&categorical != 0 {
			// For categorical, check if value matches
			if uint32(featureVal) == uint32(node.Threshold) {
				if node.Flags&leftLeaf != 0 {
					return visitedLeaves[node.Left]
				}
				idx = node.Left
			} else {
				if node.Flags&rightLeaf != 0 {
					return visitedLeaves[node.Right]
				}
				idx = idx + 1
			}
		} else {
			// Numerical split
			if featureVal <= node.Threshold {
				if node.Flags&leftLeaf != 0 {
					return visitedLeaves[node.Left]
				}
				idx = node.Left
			} else {
				if node.Flags&rightLeaf != 0 {
					return visitedLeaves[node.Right]
				}
				idx = idx + 1
			}
		}
	}
}
