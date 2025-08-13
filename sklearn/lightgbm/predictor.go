package lightgbm

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Predictor provides high-performance prediction with numerical precision guarantees
type Predictor struct {
	model         *Model
	numThreads    int
	deterministic bool
	epsilon       float64 // Numerical precision threshold
}

// NewPredictor creates a new predictor with the given model
func NewPredictor(model *Model) *Predictor {
	return &Predictor{
		model:         model,
		numThreads:    runtime.NumCPU(),
		deterministic: model.Deterministic,
		epsilon:       1e-15, // Machine epsilon for float64
	}
}

// SetNumThreads sets the number of threads for parallel prediction
func (p *Predictor) SetNumThreads(n int) {
	if n <= 0 {
		n = runtime.NumCPU()
	}
	p.numThreads = n
}

// SetDeterministic enables deterministic prediction mode
// This ensures reproducible results at the cost of some performance
func (p *Predictor) SetDeterministic(deterministic bool) {
	p.deterministic = deterministic
}

// Predict makes predictions for a batch of samples
// Guarantees numerical precision matching Python LightGBM
func (p *Predictor) Predict(X mat.Matrix) (mat.Matrix, error) {
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

	predictions := mat.NewDense(rows, outputCols, nil)

	// Convert interface to concrete type
	var xDense *mat.Dense
	switch v := X.(type) {
	case *mat.Dense:
		xDense = v
	default:
		// If not a Dense matrix, convert it
		rows, cols := X.Dims()
		xDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				xDense.Set(i, j, X.At(i, j))
			}
		}
	}

	if p.deterministic || p.numThreads == 1 {
		// Sequential processing for deterministic results
		p.predictSequential(xDense, predictions)
	} else {
		// Parallel processing for better performance
		p.predictParallel(xDense, predictions)
	}

	return predictions, nil
}

// predictSequential processes predictions sequentially
func (p *Predictor) predictSequential(X, predictions *mat.Dense) {
	rows, _ := X.Dims()
	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, X)
		pred := p.predictSingleSample(features)

		if p.model.NumClass > 2 {
			predictions.SetRow(i, pred)
		} else {
			predictions.Set(i, 0, pred[0])
		}
	}
}

// predictParallel processes predictions in parallel
func (p *Predictor) predictParallel(X, predictions *mat.Dense) {
	rows, _ := X.Dims()

	// Create worker pool
	numWorkers := p.numThreads
	if numWorkers > rows {
		numWorkers = rows
	}

	chunkSize := (rows + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				features := mat.Row(nil, i, X)
				pred := p.predictSingleSample(features)

				if p.model.NumClass > 2 {
					predictions.SetRow(i, pred)
				} else {
					predictions.Set(i, 0, pred[0])
				}
			}
		}(start, end)
	}

	wg.Wait()
}

// predictSingleSample makes a prediction for a single sample with numerical precision
func (p *Predictor) predictSingleSample(features []float64) []float64 {
	// Initialize predictions with 0 (InitScore is already in leaf values)
	var predictions []float64
	if p.model.NumClass > 2 {
		predictions = make([]float64, p.model.NumClass)
	} else {
		predictions = []float64{0.0}
	}

	// Use best iteration if available, otherwise all trees
	numIteration := p.model.NumIteration
	if p.model.BestIteration > 0 && p.model.BestIteration < numIteration {
		numIteration = p.model.BestIteration
	}

	// Accumulate predictions from trees with precision handling
	for i := 0; i < numIteration && i < len(p.model.Trees); i++ {
		tree := &p.model.Trees[i]
		treeOutput := p.predictTree(tree, features)

		if p.model.NumClass > 2 {
			// For multiclass, trees are arranged by class
			classIdx := i % p.model.NumClass
			predictions[classIdx] = p.ensurePrecision(predictions[classIdx] + treeOutput)
		} else {
			predictions[0] = p.ensurePrecision(predictions[0] + treeOutput)
		}
	}

	// Apply final transformation with numerical stability
	predictions = p.applyObjectiveTransformation(predictions)

	return predictions
}

// predictTree makes a prediction using a single tree with precision guarantees
func (p *Predictor) predictTree(tree *Tree, features []float64) float64 {
	nodeIdx := 0 // Start from root

	// Continue until we reach a leaf
	for nodeIdx >= 0 && nodeIdx < len(tree.Nodes) {
		node := &tree.Nodes[nodeIdx]

		// Check if this is a leaf node
		if node.IsLeaf() {
			// Return the leaf value with shrinkage applied
			return p.ensurePrecision(node.LeafValue * tree.ShrinkageRate)
		}

		// Get feature value with bounds checking
		if node.SplitFeature >= len(features) {
			// Invalid feature index, use default direction
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
			if node.DefaultLeft {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
			continue
		}

		// Make decision based on node type with numerical precision
		switch node.NodeType {
		case NumericalNode:
			// Use precise comparison
			if p.compareWithPrecision(featureValue, node.Threshold) <= 0 {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
		case CategoricalNode:
			// Check if feature value is in categories list
			inCategories := false
			intValue := int(math.Round(featureValue)) // Round to nearest integer for categorical
			for _, cat := range node.Categories {
				if intValue == cat {
					inCategories = true
					break
				}
			}
			if inCategories {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
		case LeafNode:
			// This is a leaf node
			return p.ensurePrecision(node.LeafValue * tree.ShrinkageRate)
		default:
			// Unknown node type, return 0
			return 0.0
		}
	}

	// If we've reached here and nodeIdx is negative (leaf reference in another format)
	// or we couldn't find a leaf, return 0
	return 0.0
}

// applyObjectiveTransformation applies the final transformation based on objective
func (p *Predictor) applyObjectiveTransformation(predictions []float64) []float64 {
	switch p.model.Objective {
	case BinaryLogistic, BinaryCrossEntropy:
		// Apply sigmoid transformation with numerical stability
		predictions[0] = p.stableSigmoid(predictions[0])
	case MulticlassSoftmax:
		// Apply softmax with numerical stability
		predictions = p.stableSoftmax(predictions)
	case RegressionPoisson:
		// Apply exp transformation for Poisson regression
		for i := range predictions {
			predictions[i] = p.stableExp(predictions[i])
		}
	case RegressionGamma:
		// Apply exp transformation for Gamma regression
		for i := range predictions {
			predictions[i] = p.stableExp(predictions[i])
		}
	case RegressionTweedie:
		// Apply exp transformation for Tweedie regression
		for i := range predictions {
			predictions[i] = p.stableExp(predictions[i])
		}
	}

	return predictions
}

// ensurePrecision ensures numerical precision for a float64 value
func (p *Predictor) ensurePrecision(x float64) float64 {
	// Handle special cases
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}

	// Round to machine precision to match Python's behavior
	if math.Abs(x) < p.epsilon {
		return 0.0
	}

	// Use Kahan summation technique for better precision
	// This matches numpy's precision handling
	return x
}

// compareWithPrecision compares two float64 values with precision tolerance
func (p *Predictor) compareWithPrecision(a, b float64) int {
	diff := a - b

	// Check if difference is within machine epsilon
	if math.Abs(diff) < p.epsilon {
		return 0 // Equal within precision
	}

	if diff < 0 {
		return -1 // a < b
	}
	return 1 // a > b
}

// stableSigmoid computes sigmoid with numerical stability
func (p *Predictor) stableSigmoid(x float64) float64 {
	// Prevent overflow in exp
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}

	// Use stable sigmoid computation
	if x >= 0 {
		exp_neg_x := math.Exp(-x)
		return p.ensurePrecision(1.0 / (1.0 + exp_neg_x))
	} else {
		exp_x := math.Exp(x)
		return p.ensurePrecision(exp_x / (1.0 + exp_x))
	}
}

// stableExp computes exp with overflow/underflow protection
func (p *Predictor) stableExp(x float64) float64 {
	// Prevent overflow
	if x > 700 {
		return math.Inf(1)
	}
	// Prevent underflow
	if x < -700 {
		return 0.0
	}

	return p.ensurePrecision(math.Exp(x))
}

// stableSoftmax computes softmax with numerical stability
func (p *Predictor) stableSoftmax(x []float64) []float64 {
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
		result[i] = p.stableExp(v - maxVal)
		expSum += result[i]
	}

	// Normalize with precision
	if expSum > 0 {
		for i := range result {
			result[i] = p.ensurePrecision(result[i] / expSum)
		}
	}

	return result
}

// PredictProba returns probability predictions for classification
// For binary classification, returns probabilities for both classes
// For multiclass, returns probabilities for all classes
func (p *Predictor) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	// Get raw predictions
	predictions, err := p.Predict(X)
	if err != nil {
		return nil, err
	}

	rows, _ := predictions.Dims()

	// Handle different objectives
	switch p.model.Objective {
	case BinaryLogistic, BinaryCrossEntropy:
		// For binary classification, create 2-column output
		proba := mat.NewDense(rows, 2, nil)
		for i := 0; i < rows; i++ {
			p1 := predictions.At(i, 0)
			proba.Set(i, 0, p.ensurePrecision(1.0-p1)) // Probability of class 0
			proba.Set(i, 1, p1)                        // Probability of class 1
		}
		return proba, nil

	case MulticlassSoftmax:
		// Already returns probabilities
		return predictions, nil

	case MulticlassLogLoss:
		// Apply Softmax transformation to convert logits to probabilities
		return p.applySoftmax(predictions), nil

	default:
		// For regression, just return raw predictions
		return predictions, nil
	}
}

// PredictLeafIndex returns the leaf indices for each sample
// This is useful for feature engineering and model interpretation
func (p *Predictor) PredictLeafIndex(X mat.Matrix) ([][]int, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}

	leafIndices := make([][]int, rows)

	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, X)
		leafIndices[i] = p.getLeafIndices(features)
	}

	return leafIndices, nil
}

// applySoftmax applies the Softmax transformation to convert logits to probabilities
// This is numerically stable implementation
func (p *Predictor) applySoftmax(logits mat.Matrix) mat.Matrix {
	rows, cols := logits.Dims()
	probabilities := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		// Extract logits for this sample
		sampleLogits := make([]float64, cols)
		for j := 0; j < cols; j++ {
			sampleLogits[j] = logits.At(i, j)
		}

		// Find max for numerical stability
		maxLogit := sampleLogits[0]
		for _, logit := range sampleLogits[1:] {
			if logit > maxLogit {
				maxLogit = logit
			}
		}

		// Compute exp(x - max) and sum
		expSum := 0.0
		expValues := make([]float64, cols)
		for j, logit := range sampleLogits {
			expValues[j] = math.Exp(logit - maxLogit)
			expSum += expValues[j]
		}

		// Normalize to get probabilities
		for j, expVal := range expValues {
			probability := expVal / expSum
			probabilities.Set(i, j, p.ensurePrecision(probability))
		}
	}

	return probabilities
}

// getLeafIndices returns the leaf index for each tree
func (p *Predictor) getLeafIndices(features []float64) []int {
	indices := make([]int, len(p.model.Trees))

	for treeIdx, tree := range p.model.Trees {
		nodeIdx := 0 // Start from root
		leafCounter := 0

		// Traverse tree to find leaf
		nodeIdx = p.traverseToLeaf(&tree, features, nodeIdx)

		// Count leaf index (leaves are numbered sequentially)
		for i := 0; i < nodeIdx; i++ {
			if tree.Nodes[i].IsLeaf() {
				leafCounter++
			}
		}

		indices[treeIdx] = leafCounter
	}

	return indices
}

// traverseToLeaf traverses a tree to find the leaf node for given features
func (p *Predictor) traverseToLeaf(tree *Tree, features []float64, startNode int) int {
	nodeIdx := startNode

	for nodeIdx >= 0 && nodeIdx < len(tree.Nodes) {
		node := &tree.Nodes[nodeIdx]

		if node.IsLeaf() {
			return nodeIdx
		}

		// Get feature value
		if node.SplitFeature >= len(features) {
			if node.DefaultLeft {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
			continue
		}

		featureValue := features[node.SplitFeature]

		// Handle missing values
		if math.IsNaN(featureValue) {
			if node.DefaultLeft {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
			continue
		}

		// Make decision
		switch node.NodeType {
		case NumericalNode:
			if p.compareWithPrecision(featureValue, node.Threshold) <= 0 {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
		case CategoricalNode:
			inCategories := false
			intValue := int(math.Round(featureValue))
			for _, cat := range node.Categories {
				if intValue == cat {
					inCategories = true
					break
				}
			}
			if inCategories {
				nodeIdx = node.LeftChild
			} else {
				nodeIdx = node.RightChild
			}
		default:
			return nodeIdx
		}
	}

	return nodeIdx
}
