package lightgbm

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Trainer implements the LightGBM training algorithm
type Trainer struct {
	// Training parameters
	params TrainingParams
	
	// Data
	X           *mat.Dense
	y           *mat.Dense
	sampleWeight []float64
	
	// Histogram data structures
	histograms  [][]Histogram
	orderedIdx  [][]int // Sorted indices for each feature
	
	// Gradient and Hessian
	gradients []float64
	hessians  []float64
	
	// Trees
	trees []Tree
	
	// Training state
	iteration int
	bestScore float64
	
	// Thread pool for parallel processing
	numThreads int
	pool       *sync.Pool
}

// TrainingParams contains all training hyperparameters
type TrainingParams struct {
	// Basic parameters
	NumIterations   int     `json:"num_iterations"`
	LearningRate    float64 `json:"learning_rate"`
	NumLeaves       int     `json:"num_leaves"`
	MaxDepth        int     `json:"max_depth"`
	MinDataInLeaf   int     `json:"min_data_in_leaf"`
	
	// Regularization
	Lambda          float64 `json:"lambda_l2"`
	Alpha           float64 `json:"lambda_l1"`
	MinGainToSplit  float64 `json:"min_gain_to_split"`
	
	// Sampling
	BaggingFraction float64 `json:"bagging_fraction"`
	BaggingFreq     int     `json:"bagging_freq"`
	FeatureFraction float64 `json:"feature_fraction"`
	
	// Histogram parameters
	MaxBin          int     `json:"max_bin"`
	MinDataInBin    int     `json:"min_data_in_bin"`
	
	// Objective
	Objective       string  `json:"objective"`
	NumClass        int     `json:"num_class"`
	
	// Other
	Seed            int     `json:"seed"`
	Deterministic   bool    `json:"deterministic"`
	Verbosity       int     `json:"verbosity"`
	EarlyStopping   int     `json:"early_stopping_rounds"`
}

// Histogram represents a histogram bin
type Histogram struct {
	Count     int
	SumGrad   float64
	SumHess   float64
	BinBounds []float64
}

// SplitInfo contains information about a potential split
type SplitInfo struct {
	Feature      int
	Threshold    float64
	Gain         float64
	LeftCount    int
	RightCount   int
	LeftValue    float64
	RightValue   float64
	LeftGrad     float64
	RightGrad    float64
	LeftHess     float64
	RightHess    float64
}

// NewTrainer creates a new LightGBM trainer
func NewTrainer(params TrainingParams) *Trainer {
	// Set default values
	if params.NumIterations == 0 {
		params.NumIterations = 100
	}
	if params.LearningRate == 0 {
		params.LearningRate = 0.1
	}
	if params.NumLeaves == 0 {
		params.NumLeaves = 31
	}
	if params.MaxBin == 0 {
		params.MaxBin = 255
	}
	if params.MinDataInLeaf == 0 {
		params.MinDataInLeaf = 20
	}
	if params.BaggingFraction == 0 {
		params.BaggingFraction = 1.0
	}
	if params.FeatureFraction == 0 {
		params.FeatureFraction = 1.0
	}
	
	return &Trainer{
		params:     params,
		numThreads: 4, // Default to 4 threads
		pool: &sync.Pool{
			New: func() interface{} {
				return &Histogram{}
			},
		},
	}
}

// Fit trains the LightGBM model
func (t *Trainer) Fit(X, y mat.Matrix) error {
	// Convert to Dense matrices
	var xDense, yDense *mat.Dense
	
	switch v := X.(type) {
	case *mat.Dense:
		xDense = v
	default:
		rows, cols := X.Dims()
		xDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				xDense.Set(i, j, X.At(i, j))
			}
		}
	}
	
	switch v := y.(type) {
	case *mat.Dense:
		yDense = v
	default:
		rows, cols := y.Dims()
		yDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				yDense.Set(i, j, y.At(i, j))
			}
		}
	}
	
	t.X = xDense
	t.y = yDense
	
	// Initialize
	if err := t.initialize(); err != nil {
		return fmt.Errorf("initialization failed: %w", err)
	}
	
	// Build histograms
	if err := t.buildHistograms(); err != nil {
		return fmt.Errorf("histogram building failed: %w", err)
	}
	
	// Main training loop
	for iter := 0; iter < t.params.NumIterations; iter++ {
		t.iteration = iter
		
		// Calculate gradients and hessians
		t.calculateGradients()
		
		// Build one tree
		tree, err := t.buildTree()
		if err != nil {
			return fmt.Errorf("tree building failed at iteration %d: %w", iter, err)
		}
		
		// Add tree to ensemble
		t.trees = append(t.trees, tree)
		
		// Update predictions
		t.updatePredictions(tree)
		
		// Check early stopping
		if t.params.EarlyStopping > 0 && t.checkEarlyStopping() {
			if t.params.Verbosity > 0 {
				fmt.Printf("Early stopping at iteration %d\n", iter)
			}
			break
		}
		
		// Log progress
		if t.params.Verbosity > 0 && iter%10 == 0 {
			fmt.Printf("Iteration %d, Loss: %.6f\n", iter, t.calculateLoss())
		}
	}
	
	return nil
}

// initialize prepares the training data structures
func (t *Trainer) initialize() error {
	rows, cols := t.X.Dims()
	
	// Initialize gradients and hessians
	t.gradients = make([]float64, rows)
	t.hessians = make([]float64, rows)
	
	// Initialize sample weights if not provided
	if t.sampleWeight == nil {
		t.sampleWeight = make([]float64, rows)
		for i := range t.sampleWeight {
			t.sampleWeight[i] = 1.0
		}
	}
	
	// Create sorted indices for each feature
	t.orderedIdx = make([][]int, cols)
	for j := 0; j < cols; j++ {
		indices := make([]int, rows)
		for i := 0; i < rows; i++ {
			indices[i] = i
		}
		
		// Sort indices by feature value
		feature := j
		sort.Slice(indices, func(a, b int) bool {
			return t.X.At(indices[a], feature) < t.X.At(indices[b], feature)
		})
		
		t.orderedIdx[j] = indices
	}
	
	return nil
}

// buildHistograms constructs histogram data structures for fast splitting
func (t *Trainer) buildHistograms() error {
	rows, cols := t.X.Dims()
	
	t.histograms = make([][]Histogram, cols)
	
	// Build histogram for each feature
	for j := 0; j < cols; j++ {
		// Get unique values for this feature
		values := make([]float64, rows)
		for i := 0; i < rows; i++ {
			values[i] = t.X.At(i, j)
		}
		
		// Find bin boundaries
		binBounds := t.findBinBoundaries(values)
		
		// Create histogram bins
		numBins := len(binBounds) - 1
		if numBins > t.params.MaxBin {
			numBins = t.params.MaxBin
		}
		
		t.histograms[j] = make([]Histogram, numBins)
		for k := 0; k < numBins; k++ {
			t.histograms[j][k] = Histogram{
				BinBounds: []float64{binBounds[k], binBounds[k+1]},
			}
		}
	}
	
	return nil
}

// findBinBoundaries finds optimal bin boundaries for a feature
func (t *Trainer) findBinBoundaries(values []float64) []float64 {
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	// Remove duplicates
	unique := []float64{sorted[0]}
	for i := 1; i < len(sorted); i++ {
		if sorted[i] != sorted[i-1] {
			unique = append(unique, sorted[i])
		}
	}
	
	// If fewer unique values than max bins, use all
	if len(unique) <= t.params.MaxBin {
		bounds := make([]float64, len(unique)+1)
		for i := 0; i < len(unique); i++ {
			bounds[i] = unique[i] - 1e-10
		}
		bounds[len(unique)] = unique[len(unique)-1] + 1e-10
		return bounds
	}
	
	// Otherwise, create equal-frequency bins
	binSize := len(unique) / t.params.MaxBin
	bounds := []float64{unique[0] - 1e-10}
	
	for i := binSize; i < len(unique); i += binSize {
		bounds = append(bounds, (unique[i-1]+unique[i])/2)
	}
	bounds = append(bounds, unique[len(unique)-1]+1e-10)
	
	return bounds
}

// calculateGradients computes gradients and hessians for current predictions
func (t *Trainer) calculateGradients() {
	rows, _ := t.y.Dims()
	
	// This is a simplified version for regression
	// In practice, this would depend on the objective function
	for i := 0; i < rows; i++ {
		// For L2 loss: gradient = prediction - target
		prediction := t.getCurrentPrediction(i)
		target := t.y.At(i, 0)
		
		t.gradients[i] = prediction - target
		t.hessians[i] = 1.0 // For L2 loss, hessian is constant
	}
}

// getCurrentPrediction gets the current ensemble prediction for a sample
func (t *Trainer) getCurrentPrediction(sampleIdx int) float64 {
	// Sum predictions from all trees
	pred := 0.0
	for _, tree := range t.trees {
		pred += t.predictSingleTree(tree, sampleIdx) * t.params.LearningRate
	}
	return pred
}

// predictSingleTree makes a prediction using a single tree
func (t *Trainer) predictSingleTree(tree Tree, sampleIdx int) float64 {
	// Navigate through tree nodes
	nodeIdx := 0
	for nodeIdx < len(tree.Nodes) {
		node := tree.Nodes[nodeIdx]
		
		if node.NodeType == LeafNode {
			return node.LeafValue
		}
		
		featureValue := t.X.At(sampleIdx, node.SplitFeature)
		if featureValue <= node.Threshold {
			nodeIdx = node.LeftChild
		} else {
			nodeIdx = node.RightChild
		}
		
		// Safety check
		if nodeIdx < 0 {
			return node.LeafValue
		}
	}
	
	return 0.0
}

// buildTree constructs a single decision tree
func (t *Trainer) buildTree() (Tree, error) {
	tree := Tree{
		TreeIndex:     t.iteration,
		ShrinkageRate: t.params.LearningRate,
		Nodes:         []Node{},
	}
	
	// Start with root node
	rows, _ := t.X.Dims()
	rootIndices := make([]int, rows)
	for i := 0; i < rows; i++ {
		rootIndices[i] = i
	}
	
	// Build tree recursively
	t.buildNode(&tree, rootIndices, 0, 0)
	
	tree.NumLeaves = t.countLeaves(tree)
	
	return tree, nil
}

// buildNode recursively builds tree nodes
func (t *Trainer) buildNode(tree *Tree, indices []int, parentIdx int, depth int) int {
	nodeIdx := len(tree.Nodes)
	
	// Check stopping conditions
	if depth >= t.params.MaxDepth || len(indices) < t.params.MinDataInLeaf {
		// Create leaf node
		leafValue := t.calculateLeafValue(indices)
		tree.Nodes = append(tree.Nodes, Node{
			NodeID:     nodeIdx,
			ParentID:   parentIdx,
			NodeType:   LeafNode,
			LeafValue:  leafValue,
			LeftChild:  -1,
			RightChild: -1,
		})
		return nodeIdx
	}
	
	// Find best split
	bestSplit := t.findBestSplit(indices)
	
	// Check if split is good enough
	if bestSplit.Gain < t.params.MinGainToSplit {
		// Create leaf node
		leafValue := t.calculateLeafValue(indices)
		tree.Nodes = append(tree.Nodes, Node{
			NodeID:     nodeIdx,
			ParentID:   parentIdx,
			NodeType:   LeafNode,
			LeafValue:  leafValue,
			LeftChild:  -1,
			RightChild: -1,
		})
		return nodeIdx
	}
	
	// Create internal node
	tree.Nodes = append(tree.Nodes, Node{
		NodeID:       nodeIdx,
		ParentID:     parentIdx,
		NodeType:     NumericalNode,
		SplitFeature: bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
	})
	
	// Split data
	leftIndices, rightIndices := t.splitData(indices, bestSplit)
	
	// Build child nodes
	leftChild := t.buildNode(tree, leftIndices, nodeIdx, depth+1)
	rightChild := t.buildNode(tree, rightIndices, nodeIdx, depth+1)
	
	// Update parent node
	tree.Nodes[nodeIdx].LeftChild = leftChild
	tree.Nodes[nodeIdx].RightChild = rightChild
	
	return nodeIdx
}

// findBestSplit finds the best split for a set of samples
func (t *Trainer) findBestSplit(indices []int) SplitInfo {
	_, cols := t.X.Dims()
	bestSplit := SplitInfo{Gain: -math.MaxFloat64}
	
	// Try each feature
	for j := 0; j < cols; j++ {
		split := t.findBestSplitForFeature(indices, j)
		if split.Gain > bestSplit.Gain {
			bestSplit = split
		}
	}
	
	return bestSplit
}

// findBestSplitForFeature finds the best split for a specific feature
func (t *Trainer) findBestSplitForFeature(indices []int, feature int) SplitInfo {
	// Get feature values and sort
	values := make([]struct {
		value float64
		idx   int
	}, len(indices))
	
	for i, idx := range indices {
		values[i] = struct {
			value float64
			idx   int
		}{
			value: t.X.At(idx, feature),
			idx:   idx,
		}
	}
	
	sort.Slice(values, func(i, j int) bool {
		return values[i].value < values[j].value
	})
	
	// Calculate total gradient and hessian
	totalGrad := 0.0
	totalHess := 0.0
	for _, idx := range indices {
		totalGrad += t.gradients[idx]
		totalHess += t.hessians[idx]
	}
	
	// Try each split point
	bestSplit := SplitInfo{
		Feature: feature,
		Gain:    -math.MaxFloat64,
	}
	
	leftGrad := 0.0
	leftHess := 0.0
	leftCount := 0
	
	for i := 0; i < len(values)-1; i++ {
		idx := values[i].idx
		leftGrad += t.gradients[idx]
		leftHess += t.hessians[idx]
		leftCount++
		
		// Skip if same value
		if values[i].value == values[i+1].value {
			continue
		}
		
		rightGrad := totalGrad - leftGrad
		rightHess := totalHess - leftHess
		rightCount := len(indices) - leftCount
		
		// Check minimum data constraints
		if leftCount < t.params.MinDataInLeaf || rightCount < t.params.MinDataInLeaf {
			continue
		}
		
		// Calculate gain
		gain := t.calculateSplitGain(leftGrad, leftHess, rightGrad, rightHess, totalGrad, totalHess)
		
		if gain > bestSplit.Gain {
			bestSplit.Gain = gain
			bestSplit.Threshold = (values[i].value + values[i+1].value) / 2
			bestSplit.LeftCount = leftCount
			bestSplit.RightCount = rightCount
			bestSplit.LeftGrad = leftGrad
			bestSplit.RightGrad = rightGrad
			bestSplit.LeftHess = leftHess
			bestSplit.RightHess = rightHess
		}
	}
	
	return bestSplit
}

// calculateSplitGain calculates the gain from a split
func (t *Trainer) calculateSplitGain(leftGrad, leftHess, rightGrad, rightHess, totalGrad, totalHess float64) float64 {
	// LightGBM split gain formula
	lambda := t.params.Lambda
	
	leftScore := (leftGrad * leftGrad) / (leftHess + lambda)
	rightScore := (rightGrad * rightGrad) / (rightHess + lambda)
	totalScore := (totalGrad * totalGrad) / (totalHess + lambda)
	
	return 0.5 * (leftScore + rightScore - totalScore)
}

// splitData splits indices based on a split decision
func (t *Trainer) splitData(indices []int, split SplitInfo) ([]int, []int) {
	var leftIndices, rightIndices []int
	
	for _, idx := range indices {
		value := t.X.At(idx, split.Feature)
		if value <= split.Threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}
	
	return leftIndices, rightIndices
}

// calculateLeafValue calculates the optimal value for a leaf node
func (t *Trainer) calculateLeafValue(indices []int) float64 {
	// For regression with L2 loss
	sumGrad := 0.0
	sumHess := 0.0
	
	for _, idx := range indices {
		sumGrad += t.gradients[idx]
		sumHess += t.hessians[idx]
	}
	
	// Optimal leaf value with L2 regularization
	return -sumGrad / (sumHess + t.params.Lambda)
}

// updatePredictions updates predictions with the new tree
func (t *Trainer) updatePredictions(tree Tree) {
	// This would update cached predictions for efficiency
	// For now, predictions are calculated on-demand
}

// checkEarlyStopping checks if training should stop early
func (t *Trainer) checkEarlyStopping() bool {
	// Simplified early stopping
	// In practice, this would use validation set
	currentLoss := t.calculateLoss()
	
	if currentLoss < t.bestScore {
		t.bestScore = currentLoss
		return false
	}
	
	// Would need to track rounds without improvement
	return false
}

// calculateLoss calculates the current loss
func (t *Trainer) calculateLoss() float64 {
	rows, _ := t.y.Dims()
	loss := 0.0
	
	for i := 0; i < rows; i++ {
		pred := t.getCurrentPrediction(i)
		target := t.y.At(i, 0)
		diff := pred - target
		loss += diff * diff
	}
	
	return loss / float64(rows)
}

// countLeaves counts the number of leaf nodes in a tree
func (t *Trainer) countLeaves(tree Tree) int {
	count := 0
	for _, node := range tree.Nodes {
		if node.NodeType == LeafNode {
			count++
		}
	}
	return count
}

// GetModel returns the trained model
func (t *Trainer) GetModel() *Model {
	model := NewModel()
	model.Trees = t.trees
	model.NumIteration = len(t.trees)
	model.NumFeatures = t.X.RawMatrix().Cols
	model.Objective = ObjectiveType(t.params.Objective)
	model.LearningRate = t.params.LearningRate
	model.NumLeaves = t.params.NumLeaves
	model.MaxDepth = t.params.MaxDepth
	
	if t.params.NumClass > 0 {
		model.NumClass = t.params.NumClass
	} else {
		model.NumClass = 1
	}
	
	return model
}