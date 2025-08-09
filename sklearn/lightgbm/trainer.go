package lightgbm

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/YuminosukeSato/scigo/pkg/log"
	"gonum.org/v1/gonum/mat"
)

// Trainer implements the LightGBM training algorithm
type Trainer struct {
	// Training parameters
	params TrainingParams

	// Data
	X            *mat.Dense
	y            *mat.Dense
	sampleWeight []float64

	// Histogram data structures
	histograms [][]Histogram
	orderedIdx [][]int // Sorted indices for each feature

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

	// Objective function
	objective ObjectiveFunction
	initScore float64

	// Callbacks
	callbacks *CallbackList
}

// TrainingParams contains all training hyperparameters
type TrainingParams struct {
	// Basic parameters
	NumIterations int     `json:"num_iterations"`
	LearningRate  float64 `json:"learning_rate"`
	NumLeaves     int     `json:"num_leaves"`
	MaxDepth      int     `json:"max_depth"`
	MinDataInLeaf int     `json:"min_data_in_leaf"`

	// Regularization
	Lambda         float64 `json:"lambda_l2"`
	Alpha          float64 `json:"lambda_l1"`
	MinGainToSplit float64 `json:"min_gain_to_split"`

	// Sampling
	BaggingFraction float64 `json:"bagging_fraction"`
	BaggingFreq     int     `json:"bagging_freq"`
	FeatureFraction float64 `json:"feature_fraction"`

	// Histogram parameters
	MaxBin       int `json:"max_bin"`
	MinDataInBin int `json:"min_data_in_bin"`

	// Objective
	Objective string `json:"objective"`
	NumClass  int    `json:"num_class"`

	// Objective-specific parameters
	HuberDelta    float64 `json:"huber_delta"`    // Delta for Huber loss
	QuantileAlpha float64 `json:"quantile_alpha"` // Alpha for Quantile regression
	FairC         float64 `json:"fair_c"`         // C parameter for Fair loss

	// Categorical features
	CategoricalFeatures []int   `json:"categorical_features"` // Indices of categorical features
	MaxCatToOnehot      int     `json:"max_cat_to_onehot"`    // Max categories to use one-hot encoding
	CatSmooth           float64 `json:"cat_smooth"`           // Smoothing for categorical splits

	// Other
	Seed          int    `json:"seed"`
	Deterministic bool   `json:"deterministic"`
	Verbosity     int    `json:"verbosity"`
	EarlyStopping int    `json:"early_stopping_rounds"`
	Metric        string `json:"metric"` // Metric for evaluation
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
	Feature    int
	Threshold  float64
	Gain       float64
	LeftCount  int
	RightCount int
	LeftValue  float64
	RightValue float64
	LeftGrad   float64
	RightGrad  float64
	LeftHess   float64
	RightHess  float64
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
		callbacks: nil, // Initialize callbacks as nil
	}
}

// WithCallbacks sets the callbacks for training
func (t *Trainer) WithCallbacks(callbacks ...Callback) *Trainer {
	t.callbacks = NewCallbackList(callbacks...)
	return t
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

	// Create objective function
	objFunc, err := CreateObjectiveFunction(t.params.Objective, &t.params)
	if err != nil {
		return fmt.Errorf("failed to create objective function: %w", err)
	}
	t.objective = objFunc

	// Calculate initial score
	rows, _ := t.y.Dims()
	targets := make([]float64, rows)
	for i := 0; i < rows; i++ {
		targets[i] = t.y.At(i, 0)
	}
	t.initScore = t.objective.GetInitScore(targets)

	// Build histograms
	if err := t.buildHistograms(); err != nil {
		return fmt.Errorf("histogram building failed: %w", err)
	}

	// Main training loop
	for iter := 0; iter < t.params.NumIterations; iter++ {
		t.iteration = iter

		// Before iteration callbacks
		if t.callbacks != nil {
			model := t.GetModel()
			if err := t.callbacks.BeforeIteration(iter, model); err != nil {
				return fmt.Errorf("callback error at iteration %d: %w", iter, err)
			}
			if t.callbacks.ShouldStop() {
				if t.params.Verbosity > 0 {
					logger := log.GetLoggerWithName("lightgbm.trainer")
					logger.Info("Training stopped by callback", "iteration", iter)
				}
				break
			}
		}

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

		// Calculate evaluation metrics
		evalResults := make(map[string]float64)
		loss := t.calculateLoss()
		evalResults["training_loss"] = loss

		// After iteration callbacks
		if t.callbacks != nil {
			model := t.GetModel()
			if err := t.callbacks.AfterIteration(iter, model, evalResults); err != nil {
				return fmt.Errorf("callback error at iteration %d: %w", iter, err)
			}
			if t.callbacks.ShouldStop() {
				if t.params.Verbosity > 0 {
					logger := log.GetLoggerWithName("lightgbm.trainer")
					logger.Info("Training stopped by callback", "iteration", iter)
				}
				break
			}
		}

		// Check early stopping (legacy)
		if t.params.EarlyStopping > 0 && t.checkEarlyStopping() {
			if t.params.Verbosity > 0 {
				logger := log.GetLoggerWithName("lightgbm.trainer")
				logger.Info("Early stopping", "iteration", iter)
			}
			break
		}

		// Log progress
		if t.params.Verbosity > 0 && iter%10 == 0 {
			logger := log.GetLoggerWithName("lightgbm.trainer")
			logger.Debug("Training progress",
				"iteration", iter,
				"loss", loss)
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

	for i := 0; i < rows; i++ {
		prediction := t.getCurrentPrediction(i)
		target := t.y.At(i, 0)

		// Use objective function to calculate gradients and hessians
		t.gradients[i] = t.objective.CalculateGradient(prediction, target)
		t.hessians[i] = t.objective.CalculateHessian(prediction, target)

		// Apply sample weight if provided
		if t.sampleWeight != nil {
			t.gradients[i] *= t.sampleWeight[i]
			t.hessians[i] *= t.sampleWeight[i]
		}
	}
}

// getCurrentPrediction gets the current ensemble prediction for a sample
func (t *Trainer) getCurrentPrediction(sampleIdx int) float64 {
	// Start with initial score
	pred := t.initScore

	// Sum predictions from all trees
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
	numLeaves := t.countLeavesInTree(tree)
	if (t.params.MaxDepth > 0 && depth >= t.params.MaxDepth) ||
		len(indices) < t.params.MinDataInLeaf ||
		(t.params.NumLeaves > 0 && numLeaves >= t.params.NumLeaves-1) {
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

	// Create internal node with gain information
	nodeType := NumericalNode
	var categories []int

	// Check if this is a categorical split
	if t.isCategoricalFeature(bestSplit.Feature) {
		nodeType = CategoricalNode
		categories = t.getCategoriesForSplit(indices, bestSplit.Feature, bestSplit)
	}

	tree.Nodes = append(tree.Nodes, Node{
		NodeID:       nodeIdx,
		ParentID:     parentIdx,
		NodeType:     nodeType,
		SplitFeature: bestSplit.Feature,
		Threshold:    bestSplit.Threshold,
		Categories:   categories,
		Gain:         bestSplit.Gain,
	})

	// Split data
	var leftIndices, rightIndices []int
	if nodeType == CategoricalNode {
		// Create map for fast lookup
		leftCatMap := make(map[int]bool)
		for _, cat := range categories {
			leftCatMap[cat] = true
		}
		leftIndices, rightIndices = t.splitCategoricalData(indices, bestSplit.Feature, leftCatMap)
	} else {
		leftIndices, rightIndices = t.splitData(indices, bestSplit)
	}

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
		var split SplitInfo

		// Check if feature is categorical
		if t.isCategoricalFeature(j) {
			split = t.findBestCategoricalSplit(indices, j)
		} else {
			split = t.findBestSplitForFeature(indices, j)
		}

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
	sumGrad := 0.0
	sumHess := 0.0

	for _, idx := range indices {
		sumGrad += t.gradients[idx]
		sumHess += t.hessians[idx]
	}

	// Ensure numerical stability
	epsilon := 1e-10
	if math.Abs(sumHess) < epsilon {
		sumHess = epsilon
	}

	// Optimal leaf value with L2 regularization
	return -sumGrad / (sumHess + t.params.Lambda + epsilon)
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
	totalWeight := 0.0

	for i := 0; i < rows; i++ {
		pred := t.getCurrentPrediction(i)
		target := t.y.At(i, 0)

		// Use objective function to calculate loss
		sampleLoss := t.objective.CalculateLoss(pred, target)

		if t.sampleWeight != nil {
			sampleLoss *= t.sampleWeight[i]
			totalWeight += t.sampleWeight[i]
		} else {
			totalWeight += 1.0
		}

		loss += sampleLoss
	}

	return loss / totalWeight
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

// countLeavesInTree counts the number of leaf nodes in a tree being built
func (t *Trainer) countLeavesInTree(tree *Tree) int {
	count := 0
	for _, node := range tree.Nodes {
		if node.NodeType == LeafNode || (node.LeftChild == -1 && node.RightChild == -1) {
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
	model.InitScore = t.initScore

	if t.params.NumClass > 0 {
		model.NumClass = t.params.NumClass
	} else {
		model.NumClass = 1
	}

	return model
}
