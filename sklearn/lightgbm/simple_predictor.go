package lightgbm

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// SimplePredictor provides a simple, accurate prediction implementation
// following the leaves library approach for maximum compatibility
type SimplePredictor struct {
	model *Model
}

// NewSimplePredictor creates a new simple predictor
func NewSimplePredictor(model *Model) *SimplePredictor {
	return &SimplePredictor{
		model: model,
	}
}

// PredictSingle makes a prediction for a single sample
func (p *SimplePredictor) PredictSingle(features []float64) []float64 {
	if p.model.NumClass > 2 {
		// Multiclass prediction
		predictions := make([]float64, p.model.NumClass)
		
		// Initialize with init scores for each class
		for i := range predictions {
			predictions[i] = p.model.InitScore
		}
		
		// Accumulate predictions from trees
		for i, tree := range p.model.Trees {
			classIdx := i % p.model.NumClass
			treeOutput := p.predictTree(&tree, features)
			predictions[classIdx] += treeOutput
		}
		
		// Apply softmax transformation for multiclass
		return p.softmax(predictions)
	} else {
		// Binary or regression - start with init score
		prediction := p.model.InitScore
		
		// Sum predictions from all trees
		for _, tree := range p.model.Trees {
			treeOutput := p.predictTree(&tree, features)
			prediction += treeOutput
		}
		
		// Apply transformation based on objective
		switch p.model.Objective {
		case BinaryLogistic, BinaryCrossEntropy:
			// Apply sigmoid for binary classification
			prediction = p.sigmoid(prediction)
		}
		
		return []float64{prediction}
	}
}

// predictTree makes a prediction using a single tree
// This follows the leaves approach for accurate tree traversal
func (p *SimplePredictor) predictTree(tree *Tree, features []float64) float64 {
	if len(tree.Nodes) == 0 {
		if len(tree.LeafValues) > 0 {
			// Constant tree with single leaf value
			return tree.LeafValues[0]
		}
		return 0.0
	}
	
	// Start from root node (index 0)
	nodeIdx := 0
	
	for {
		if nodeIdx < 0 || nodeIdx >= len(tree.Nodes) {
			return 0.0
		}
		
		node := &tree.Nodes[nodeIdx]
		
		// Check if it's a leaf node
		if node.NodeType == LeafNode {
			// Apply shrinkage rate if set
			if tree.ShrinkageRate > 0 {
				return node.LeafValue * tree.ShrinkageRate
			}
			return node.LeafValue
		}
		
		// Get feature value for split
		if node.SplitFeature >= len(features) {
			// Invalid feature index, treat as missing
			if node.DefaultLeft {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
			continue
		}
		
		featureValue := features[node.SplitFeature]
		
		// Handle missing values (NaN)
		if math.IsNaN(featureValue) {
			// Use default direction for missing values
			if node.DefaultLeft {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
			continue
		}
		
		// Make decision based on threshold
		// LightGBM uses <= for numerical splits (not <)
		if featureValue <= node.Threshold {
			nodeIdx = node.LeftChild
		} else {
			nodeIdx = node.RightChild
		}
		
		// Handle case where child indices are negative (leaf indicators)
		if nodeIdx < 0 {
			// In LightGBM text format, negative child index means leaf
			// The actual leaf value index is -(nodeIdx + 1)
			leafIdx := -(nodeIdx + 1)
			if leafIdx < len(tree.LeafValues) {
				if tree.ShrinkageRate > 0 {
					return tree.LeafValues[leafIdx] * tree.ShrinkageRate
				}
				return tree.LeafValues[leafIdx]
			}
			return 0.0
		}
	}
}

// Predict makes predictions for a batch of samples
func (p *SimplePredictor) Predict(X mat.Matrix) (mat.Matrix, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}
	
	// Prepare output matrix
	var outputCols int
	if p.model.NumClass > 2 {
		outputCols = p.model.NumClass
	} else {
		outputCols = 1
	}
	
	predictions := mat.NewDense(rows, outputCols, nil)
	
	// Process each sample
	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, X)
		pred := p.PredictSingle(features)
		
		if p.model.NumClass > 2 {
			// Multiclass: set all class probabilities
			predictions.SetRow(i, pred)
		} else {
			// Binary or regression: single value
			predictions.Set(i, 0, pred[0])
		}
	}
	
	return predictions, nil
}

// sigmoid applies the sigmoid transformation
func (p *SimplePredictor) sigmoid(x float64) float64 {
	// Numerical stability for sigmoid
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}
	
	// Standard sigmoid: 1 / (1 + exp(-x))
	if x >= 0 {
		expNegX := math.Exp(-x)
		return 1.0 / (1.0 + expNegX)
	} else {
		// For negative x, use exp(x) / (1 + exp(x)) for numerical stability
		expX := math.Exp(x)
		return expX / (1.0 + expX)
	}
}

// softmax applies the softmax transformation
func (p *SimplePredictor) softmax(x []float64) []float64 {
	// Find max for numerical stability
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	
	// Compute exp(x - max) and sum
	expSum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		expSum += result[i]
	}
	
	// Normalize
	if expSum > 0 {
		for i := range result {
			result[i] /= expSum
		}
	}
	
	return result
}

// PredictRaw returns raw predictions without transformations
func (p *SimplePredictor) PredictRaw(features []float64) []float64 {
	if p.model.NumClass > 2 {
		// Multiclass raw scores
		predictions := make([]float64, p.model.NumClass)
		
		// Initialize with init scores
		for i := range predictions {
			predictions[i] = p.model.InitScore
		}
		
		// Accumulate predictions from trees
		for i, tree := range p.model.Trees {
			classIdx := i % p.model.NumClass
			treeOutput := p.predictTree(&tree, features)
			predictions[classIdx] += treeOutput
		}
		
		return predictions
	} else {
		// Binary or regression raw score - start with init score
		prediction := p.model.InitScore
		
		// Sum predictions from all trees
		for _, tree := range p.model.Trees {
			treeOutput := p.predictTree(&tree, features)
			prediction += treeOutput
		}
		
		return []float64{prediction}
	}
}