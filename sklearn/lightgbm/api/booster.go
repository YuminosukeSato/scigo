package api

import (
	"encoding/json"
	"fmt"
	"os"

	lgb "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

// Booster represents a trained LightGBM model, similar to Python's Booster class
type Booster struct {
	// Core model components
	model           *lgb.Model
	predictor       *lgb.Predictor
	leavesPredictor *lgb.LeavesPredictor

	// Training information
	numIterations    int
	currentIteration int
	bestIteration    int

	// Evaluation results during training
	evalResults map[string][]float64

	// Parameters used for training
	params map[string]interface{}

	// Feature information
	featureNames      []string
	featureImportance map[string][]float64
}

// NewBooster creates a new Booster instance
func NewBooster(params map[string]interface{}) *Booster {
	return &Booster{
		params:            params,
		evalResults:       make(map[string][]float64),
		featureImportance: make(map[string][]float64),
		currentIteration:  0,
	}
}

// InitFromModel initializes the booster from an existing model
func (b *Booster) InitFromModel(model *lgb.Model) {
	b.model = model
	b.predictor = lgb.NewPredictor(model)
	b.numIterations = len(model.Trees)
	b.currentIteration = b.numIterations
	b.bestIteration = model.BestIteration
}

// InitFromLeavesModel initializes the booster from a leaves model
func (b *Booster) InitFromLeavesModel(model *lgb.LeavesModel) {
	// Convert LeavesModel to Model for compatibility
	lgbModel := &lgb.Model{
		NumFeatures: model.NumFeatures,
		NumClass:    model.NumClass,
		Objective:   model.Objective,
	}
	b.model = lgbModel
	b.leavesPredictor = lgb.NewLeavesPredictor(model)
	b.numIterations = len(model.Trees)
	b.currentIteration = b.numIterations
}

// Update performs one boosting iteration
// This is called internally during training
func (b *Booster) Update(trainSet *Dataset, fobj func(mat.Matrix, *Dataset) (mat.Matrix, mat.Matrix)) error {
	// This would integrate with the existing trainer
	// For now, we'll provide a placeholder
	b.currentIteration++
	return nil
}

// Predict makes predictions using the trained model
// Similar to Python's booster.predict(data)
func (b *Booster) Predict(data mat.Matrix, options ...PredictOption) (mat.Matrix, error) {
	// Check for either predictor type
	if b.predictor == nil && b.leavesPredictor == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	// Apply prediction options
	opts := &predictOptions{
		numIteration:   -1, // Use all trees by default
		predictType:    "response",
		startIteration: 0,
	}
	for _, opt := range options {
		opt(opts)
	}

	// Make predictions using appropriate predictor
	var predictions mat.Matrix
	var err error
	if b.leavesPredictor != nil {
		predictions, err = b.leavesPredictor.Predict(data)
	} else {
		predictions, err = b.predictor.Predict(data)
	}
	if err != nil {
		return nil, err
	}

	return predictions, nil
}

// PredictProba makes probability predictions for classification tasks
// Similar to Python's booster.predict(data, num_iteration=num_iteration)
// For binary classification, returns probabilities for both classes
// For multiclass, returns probabilities for all classes
func (b *Booster) PredictProba(data mat.Matrix, options ...PredictOption) (mat.Matrix, error) {
	// Check for either predictor type
	if b.predictor == nil && b.leavesPredictor == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	// Apply prediction options
	opts := &predictOptions{
		numIteration:   -1, // Use all trees by default
		predictType:    "response",
		startIteration: 0,
	}
	for _, opt := range options {
		opt(opts)
	}

	// Make probability predictions using appropriate predictor
	var probabilities mat.Matrix
	var err error
	if b.leavesPredictor != nil {
		// LeavesPredictor.Predict already returns probabilities
		probabilities, err = b.leavesPredictor.Predict(data)
	} else {
		probabilities, err = b.predictor.PredictProba(data)
	}
	if err != nil {
		return nil, err
	}

	return probabilities, nil
}

// PredictRawScore returns raw prediction scores without transformation
// Similar to Python's booster.predict(data, raw_score=True)
func (b *Booster) PredictRawScore(data mat.Matrix, options ...PredictOption) (mat.Matrix, error) {
	// Check for either predictor type
	if b.predictor == nil && b.leavesPredictor == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	// Apply prediction options
	opts := &predictOptions{
		numIteration:   -1, // Use all trees by default
		predictType:    "raw",
		startIteration: 0,
	}
	for _, opt := range options {
		opt(opts)
	}

	// Make raw score predictions using appropriate predictor
	var rawScores mat.Matrix
	var err error
	if b.leavesPredictor != nil {
		// LeavesPredictor doesn't have PredictRawScore, use Predict
		// TODO: Implement raw score support for LeavesPredictor
		rawScores, err = b.leavesPredictor.Predict(data)
	} else {
		rawScores, err = b.predictor.PredictRawScore(data)
	}
	if err != nil {
		return nil, err
	}

	return rawScores, nil
}

// PredictLeaf returns leaf indices for each sample
// Similar to Python's booster.predict(data, pred_leaf=True)
func (b *Booster) PredictLeaf(data mat.Matrix, options ...PredictOption) (mat.Matrix, error) {
	// Check for either predictor type
	if b.predictor == nil && b.leavesPredictor == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	// Apply prediction options
	opts := &predictOptions{
		numIteration:   -1, // Use all trees by default
		predictType:    "leaf",
		startIteration: 0,
	}
	for _, opt := range options {
		opt(opts)
	}

	// Get leaf indices using appropriate predictor
	var leafIndices mat.Matrix
	var err error
	if b.leavesPredictor != nil {
		// LeavesPredictor doesn't have PredictLeaf, return error
		// TODO: Implement leaf prediction support for LeavesPredictor
		return nil, fmt.Errorf("leaf prediction not supported for loaded models")
	} else {
		leafIndices, err = b.predictor.PredictLeaf(data)
	}
	if err != nil {
		return nil, err
	}

	return leafIndices, nil
}

// PredictOption is a functional option for prediction
type PredictOption func(*predictOptions)

type predictOptions struct {
	numIteration   int
	predictType    string // "response", "raw", "leaf"
	startIteration int
}

// WithNumIteration sets the number of iterations to use for prediction
func WithNumIteration(n int) PredictOption {
	return func(o *predictOptions) {
		o.numIteration = n
	}
}

// WithPredictType sets the prediction type
func WithPredictType(t string) PredictOption {
	return func(o *predictOptions) {
		o.predictType = t
	}
}

// SaveModel saves the model to a file
// Similar to Python's booster.save_model(filename)
func (b *Booster) SaveModel(filename string, options ...SaveOption) error {
	if b.model == nil {
		return fmt.Errorf("no model to save")
	}

	opts := &saveOptions{
		numIteration:   -1,
		startIteration: 0,
		saveType:       "text",
	}
	for _, opt := range options {
		opt(opts)
	}

	// Save based on file type
	switch opts.saveType {
	case "json":
		return b.saveJSON(filename)
	case "text":
		return b.saveText(filename)
	default:
		return fmt.Errorf("unsupported save type: %s", opts.saveType)
	}
}

// SaveOption is a functional option for saving
type SaveOption func(*saveOptions)

type saveOptions struct {
	numIteration   int
	startIteration int
	saveType       string // "text" or "json"
}

// WithSaveType sets the save format
func WithSaveType(t string) SaveOption {
	return func(o *saveOptions) {
		o.saveType = t
	}
}

// saveJSON saves the model in JSON format
func (b *Booster) saveJSON(filename string) error {
	modelData := b.DumpModel()

	jsonData, err := json.MarshalIndent(modelData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}

	return os.WriteFile(filename, jsonData, 0600)
}

// saveText saves the model in text format
func (b *Booster) saveText(filename string) error {
	if b.model == nil {
		return fmt.Errorf("no model to save")
	}

	// Use the Model's SaveToText method to save in LightGBM text format
	return b.model.SaveToText(filename)
}

// LoadModel loads a model from file
func LoadModel(filename string) (*Booster, error) {
	// Check if file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", filename)
	}

	// Load the model using existing loader
	model, err := lgb.LoadLeavesModelFromFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	booster := NewBooster(nil)
	booster.InitFromLeavesModel(model)

	return booster, nil
}

// DumpModel dumps the model to a dictionary
// Similar to Python's booster.dump_model()
func (b *Booster) DumpModel() map[string]interface{} {
	if b.model == nil {
		return nil
	}

	dump := make(map[string]interface{})

	// Basic information
	dump["name"] = "tree"
	dump["version"] = "v3"
	dump["num_class"] = b.model.NumClass
	dump["num_tree_per_iteration"] = 1
	if b.model.NumClass > 2 {
		dump["num_tree_per_iteration"] = b.model.NumClass
	}
	dump["label_index"] = 0
	dump["max_feature_idx"] = b.model.NumFeatures - 1

	// Objective and parameters
	dump["objective"] = b.model.Objective
	if b.params != nil {
		dump["params"] = b.params
	}

	// Feature information
	if len(b.featureNames) > 0 {
		dump["feature_names"] = b.featureNames
	}
	dump["feature_importances"] = b.featureImportance

	// Tree information
	trees := make([]map[string]interface{}, len(b.model.Trees))
	for i, tree := range b.model.Trees {
		treeInfo := make(map[string]interface{})
		treeInfo["tree_index"] = i
		treeInfo["num_leaves"] = tree.NumLeaves
		treeInfo["num_nodes"] = tree.NumNodes
		treeInfo["shrinkage"] = tree.ShrinkageRate

		// Add tree structure
		treeStructure := b.dumpTreeStructure(tree)
		treeInfo["tree_structure"] = treeStructure

		trees[i] = treeInfo
	}
	dump["tree_info"] = trees

	return dump
}

// dumpTreeStructure dumps a single tree's structure
func (b *Booster) dumpTreeStructure(tree lgb.Tree) map[string]interface{} {
	if len(tree.Nodes) == 0 {
		// Leaf-only tree
		return map[string]interface{}{
			"leaf_value": tree.LeafValues,
		}
	}

	// Build tree structure recursively
	return b.dumpNode(&tree, 0)
}

// dumpNode recursively dumps a node and its children
func (b *Booster) dumpNode(tree *lgb.Tree, nodeIdx int) map[string]interface{} {
	if nodeIdx >= len(tree.Nodes) || nodeIdx < 0 {
		return nil
	}

	node := &tree.Nodes[nodeIdx]
	nodeInfo := make(map[string]interface{})

	if node.IsLeaf() {
		// Leaf node
		nodeInfo["leaf_index"] = nodeIdx
		nodeInfo["leaf_value"] = node.LeafValue
		nodeInfo["leaf_count"] = node.LeafCount
	} else {
		// Internal node
		nodeInfo["split_index"] = nodeIdx
		nodeInfo["split_feature"] = node.SplitFeature
		nodeInfo["split_gain"] = node.Gain
		nodeInfo["threshold"] = node.Threshold
		nodeInfo["default_left"] = node.DefaultLeft
		nodeInfo["internal_value"] = node.InternalValue
		nodeInfo["internal_count"] = node.InternalCount

		// Add children
		if node.LeftChild >= 0 {
			nodeInfo["left_child"] = b.dumpNode(tree, node.LeftChild)
		}
		if node.RightChild >= 0 {
			nodeInfo["right_child"] = b.dumpNode(tree, node.RightChild)
		}
	}

	return nodeInfo
}

// NumTrees returns the number of trees in the model
func (b *Booster) NumTrees() int {
	if b.model == nil {
		return 0
	}
	return len(b.model.Trees)
}

// NumFeatures returns the number of features
func (b *Booster) NumFeatures() int {
	if b.model == nil {
		return 0
	}
	return b.model.NumFeatures
}

// CurrentIteration returns the current iteration number
func (b *Booster) CurrentIteration() int {
	return b.currentIteration
}

// BestIteration returns the best iteration number (for early stopping)
func (b *Booster) BestIteration() int {
	return b.bestIteration
}

// SetBestIteration sets the best iteration number
func (b *Booster) SetBestIteration(iter int) {
	b.bestIteration = iter
}

// FeatureImportance returns feature importance scores
// importanceType can be "split" or "gain"
func (b *Booster) FeatureImportance(importanceType string) []float64 {
	if b.model == nil {
		return nil
	}

	// Check cache
	if importance, ok := b.featureImportance[importanceType]; ok {
		return importance
	}

	// Calculate importance
	importance := b.model.GetFeatureImportance(importanceType)
	b.featureImportance[importanceType] = importance

	return importance
}

// FeatureName returns feature names
func (b *Booster) FeatureName() []string {
	return b.featureNames
}

// SetFeatureNames sets feature names
func (b *Booster) SetFeatureNames(names []string) {
	b.featureNames = names
	if b.model != nil {
		b.model.FeatureNames = names
	}
}

// GetEvalResults returns evaluation results from training
func (b *Booster) GetEvalResults() map[string][]float64 {
	return b.evalResults
}

// AddEvalResult adds an evaluation result for a specific dataset and metric
func (b *Booster) AddEvalResult(datasetName, metricName string, value float64) {
	key := fmt.Sprintf("%s-%s", datasetName, metricName)
	b.evalResults[key] = append(b.evalResults[key], value)
}

// GetModel returns the underlying model
func (b *Booster) GetModel() *lgb.Model {
	return b.model
}

// SetModel sets the underlying model
func (b *Booster) SetModel(model *lgb.Model) {
	b.model = model
	b.predictor = lgb.NewPredictor(model)
}

// Clone creates a copy of the booster
func (b *Booster) Clone() *Booster {
	newBooster := &Booster{
		model:             b.model,
		predictor:         b.predictor,
		numIterations:     b.numIterations,
		currentIteration:  b.currentIteration,
		bestIteration:     b.bestIteration,
		params:            make(map[string]interface{}),
		evalResults:       make(map[string][]float64),
		featureNames:      make([]string, len(b.featureNames)),
		featureImportance: make(map[string][]float64),
	}

	// Copy params
	for k, v := range b.params {
		newBooster.params[k] = v
	}

	// Copy eval results
	for k, v := range b.evalResults {
		results := make([]float64, len(v))
		copy(results, v)
		newBooster.evalResults[k] = results
	}

	// Copy feature names
	copy(newBooster.featureNames, b.featureNames)

	// Copy feature importance
	for k, v := range b.featureImportance {
		importance := make([]float64, len(v))
		copy(importance, v)
		newBooster.featureImportance[k] = importance
	}

	return newBooster
}
