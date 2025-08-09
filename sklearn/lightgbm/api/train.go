package api

import (
	"fmt"
	"time"

	"github.com/YuminosukeSato/scigo/metrics"
	lgb "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

// Train trains a LightGBM model, similar to Python's lgb.train()
//
// Parameters:
//   - params: Parameters for training
//   - trainSet: Training dataset
//   - numBoostRound: Number of boosting iterations
//   - validSets: List of datasets for evaluation during training
//   - options: Additional training options (callbacks, early stopping, etc.)
//
// Returns:
//   - Trained Booster model
func Train(params map[string]interface{}, trainSet *Dataset, numBoostRound int, validSets []*Dataset, options ...TrainOption) (*Booster, error) {
	// Parse parameters
	trainParams := parseParams(params)

	// Create trainer
	trainer := lgb.NewTrainer(trainParams)

	// Create booster
	booster := NewBooster(params)

	// Apply options
	opts := &trainOptions{
		callbacks:          []Callback{},
		validNames:         []string{},
		earlyStoppingRound: 0,
		verboseEval:        true,
		evalFreq:           1,
	}
	for _, opt := range options {
		opt(opts)
	}

	// Add default callbacks
	if opts.verboseEval {
		opts.callbacks = append(opts.callbacks, NewPrintEvaluationCallback(opts.evalFreq))
	}
	if opts.earlyStoppingRound > 0 {
		opts.callbacks = append(opts.callbacks, NewEarlyStoppingCallback(opts.earlyStoppingRound, true))
	}

	// Prepare validation sets
	if len(validSets) > 0 && len(opts.validNames) == 0 {
		for i := range validSets {
			opts.validNames = append(opts.validNames, fmt.Sprintf("valid_%d", i))
		}
	}

	// Initialize callbacks
	env := &CallbackEnv{
		Iteration:     0,
		NumBoostRound: numBoostRound,
		Booster:       booster,
		Params:        params,
		EvalResults:   make(map[string][]float64),
	}

	for _, cb := range opts.callbacks {
		if err := cb.Init(env); err != nil {
			return nil, fmt.Errorf("callback initialization failed: %w", err)
		}
	}

	// Training loop
	fmt.Printf("[LightGBM] [Info] Start training from score %.6f\n", 0.0)
	startTime := time.Now()

	for iter := 0; iter < numBoostRound; iter++ {
		env.Iteration = iter

		// Before iteration callbacks
		for _, cb := range opts.callbacks {
			if err := cb.BeforeIteration(env); err != nil {
				if err == ErrEarlyStop {
					fmt.Printf("[LightGBM] [Info] Early stopping at iteration %d\n", iter)
					booster.SetBestIteration(env.BestIteration)
					break
				}
				return nil, err
			}
		}

		// Check if should stop
		if env.ShouldStop {
			break
		}

		// Perform one boosting iteration
		// This would integrate with the actual trainer
		err := trainOneIteration(trainer, trainSet, booster)
		if err != nil {
			return nil, fmt.Errorf("training iteration %d failed: %w", iter, err)
		}

		// Evaluate on validation sets
		if len(validSets) > 0 {
			for i, validSet := range validSets {
				score := evaluateDataset(booster, validSet, trainParams.Objective)
				metricName := getMetricName(trainParams.Objective)
				datasetName := opts.validNames[i]

				key := fmt.Sprintf("%s-%s", datasetName, metricName)
				env.EvalResults[key] = append(env.EvalResults[key], score)
				booster.AddEvalResult(datasetName, metricName, score)
			}
		}

		// After iteration callbacks
		for _, cb := range opts.callbacks {
			if err := cb.AfterIteration(env); err != nil {
				return nil, err
			}
		}

		booster.currentIteration = iter + 1
	}

	trainTime := time.Since(startTime)
	fmt.Printf("[LightGBM] [Info] Training finished in %.2f seconds\n", trainTime.Seconds())

	// Finalize callbacks
	for _, cb := range opts.callbacks {
		if err := cb.Finalize(env); err != nil {
			return nil, err
		}
	}

	// Set best iteration if not set
	if booster.bestIteration == 0 {
		booster.bestIteration = booster.currentIteration
	}

	return booster, nil
}

// TrainOption is a functional option for training configuration
type TrainOption func(*trainOptions)

type trainOptions struct {
	callbacks          []Callback
	validNames         []string
	earlyStoppingRound int
	verboseEval        bool
	evalFreq           int
	initModel          *Booster
}

// WithCallbacks adds callbacks to the training process
func WithCallbacks(callbacks ...Callback) TrainOption {
	return func(o *trainOptions) {
		o.callbacks = append(o.callbacks, callbacks...)
	}
}

// WithValidNames sets names for validation datasets
func WithValidNames(names []string) TrainOption {
	return func(o *trainOptions) {
		o.validNames = names
	}
}

// WithEarlyStopping enables early stopping
func WithEarlyStopping(rounds int) TrainOption {
	return func(o *trainOptions) {
		o.earlyStoppingRound = rounds
	}
}

// WithVerboseEval controls verbose evaluation output
func WithVerboseEval(verbose bool, freq int) TrainOption {
	return func(o *trainOptions) {
		o.verboseEval = verbose
		o.evalFreq = freq
	}
}

// WithInitModel sets an initial model for continued training
func WithInitModel(model *Booster) TrainOption {
	return func(o *trainOptions) {
		o.initModel = model
	}
}

// parseParams converts the parameter map to TrainingParams
func parseParams(params map[string]interface{}) lgb.TrainingParams {
	tp := lgb.TrainingParams{
		// Default values
		NumIterations:   100,
		LearningRate:    0.1,
		NumLeaves:       31,
		MaxDepth:        -1,
		MinDataInLeaf:   20,
		Lambda:          0.0,
		Alpha:           0.0,
		MinGainToSplit:  0.0,
		BaggingFraction: 1.0,
		FeatureFraction: 1.0,
		MaxBin:          255,
		MinDataInBin:    3,
		Objective:       "regression",
		Seed:            0,
		Deterministic:   false,
		Verbosity:       1,
	}

	// Parse from map
	if val, ok := params["num_iterations"].(int); ok {
		tp.NumIterations = val
	}
	if val, ok := params["learning_rate"].(float64); ok {
		tp.LearningRate = val
	}
	if val, ok := params["num_leaves"].(int); ok {
		tp.NumLeaves = val
	}
	if val, ok := params["max_depth"].(int); ok {
		tp.MaxDepth = val
	}
	if val, ok := params["min_data_in_leaf"].(int); ok {
		tp.MinDataInLeaf = val
	}
	if val, ok := params["lambda_l2"].(float64); ok {
		tp.Lambda = val
	}
	if val, ok := params["lambda_l1"].(float64); ok {
		tp.Alpha = val
	}
	if val, ok := params["min_gain_to_split"].(float64); ok {
		tp.MinGainToSplit = val
	}
	if val, ok := params["bagging_fraction"].(float64); ok {
		tp.BaggingFraction = val
	}
	if val, ok := params["feature_fraction"].(float64); ok {
		tp.FeatureFraction = val
	}
	if val, ok := params["max_bin"].(int); ok {
		tp.MaxBin = val
	}
	if val, ok := params["min_data_in_bin"].(int); ok {
		tp.MinDataInBin = val
	}
	if val, ok := params["objective"].(string); ok {
		tp.Objective = val
	}
	if val, ok := params["random_state"].(int); ok {
		tp.Seed = val
	}
	if val, ok := params["seed"].(int); ok {
		tp.Seed = val
	}
	if val, ok := params["deterministic"].(bool); ok {
		tp.Deterministic = val
	}
	if val, ok := params["verbosity"].(int); ok {
		tp.Verbosity = val
	}
	if val, ok := params["boosting_type"].(string); ok {
		tp.BoostingType = val
	}
	if val, ok := params["num_class"].(int); ok {
		tp.NumClass = val
	}

	return tp
}

// trainOneIteration performs one boosting iteration
func trainOneIteration(trainer *lgb.Trainer, trainSet *Dataset, booster *Booster) error {
	// This is a simplified version - actual implementation would:
	// 1. Use the trainer to build one tree
	// 2. Add the tree to the model
	// 3. Update the booster

	// For now, we just simulate the training
	if booster.model == nil {
		// Initialize model on first iteration
		rows, cols := trainSet.Data.Dims()
		booster.model = lgb.NewModel()
		booster.model.NumFeatures = cols

		// Detect task type
		if trainSet.isBinary {
			booster.model.Objective = lgb.BinaryLogistic
			booster.model.NumClass = 2
		} else if trainSet.nClasses > 2 {
			booster.model.Objective = lgb.MulticlassSoftmax
			booster.model.NumClass = trainSet.nClasses
		} else {
			booster.model.Objective = lgb.RegressionL2
		}

		booster.predictor = lgb.NewPredictor(booster.model)

		// Set feature names
		if len(trainSet.FeatureNames) > 0 {
			booster.SetFeatureNames(trainSet.FeatureNames)
		}

		_ = rows // Use rows to avoid unused variable
	}

	// Simulate adding a tree (actual implementation would use trainer.Fit)
	tree := lgb.Tree{
		TreeIndex:     booster.currentIteration,
		NumLeaves:     31,
		NumNodes:      61,
		ShrinkageRate: 0.1,
	}
	booster.model.Trees = append(booster.model.Trees, tree)

	return nil
}

// evaluateDataset evaluates the model on a dataset
func evaluateDataset(booster *Booster, dataset *Dataset, objective string) float64 {
	if booster.model == nil || dataset == nil {
		return 0.0
	}

	// Make predictions
	predictions, err := booster.Predict(dataset.Data)
	if err != nil {
		return 0.0
	}

	// Calculate metric based on objective
	var score float64
	switch objective {
	case "regression", "regression_l2", "l2", "mean_squared_error", "mse":
		if mse, err := metrics.MSE(getVecDense(dataset.Label), getVecDense(predictions)); err == nil {
			score = mse
		}
	case "regression_l1", "l1", "mean_absolute_error", "mae":
		if mae, err := metrics.MAE(getVecDense(dataset.Label), getVecDense(predictions)); err == nil {
			score = mae
		}
	case "binary", "binary_logloss":
		// For binary classification, return accuracy
		score = calculateAccuracy(dataset.Label, predictions)
	case "multiclass", "softmax":
		// For multiclass, return accuracy
		score = calculateAccuracy(dataset.Label, predictions)
	default:
		score = 0.0
	}

	return score
}

// getMetricName returns the metric name for an objective
func getMetricName(objective string) string {
	switch objective {
	case "regression", "regression_l2", "l2":
		return "l2"
	case "regression_l1", "l1":
		return "l1"
	case "binary":
		return "binary_logloss"
	case "multiclass", "softmax":
		return "multi_logloss"
	default:
		return objective
	}
}

// getVecDense converts a matrix to VecDense (assuming single column)
func getVecDense(m mat.Matrix) *mat.VecDense {
	if m == nil {
		return nil
	}
	rows, _ := m.Dims()
	vec := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		vec.SetVec(i, m.At(i, 0))
	}
	return vec
}

// calculateAccuracy calculates classification accuracy
func calculateAccuracy(yTrue, yPred mat.Matrix) float64 {
	if yTrue == nil || yPred == nil {
		return 0.0
	}

	rows, _ := yTrue.Dims()
	correct := 0

	for i := 0; i < rows; i++ {
		trueVal := yTrue.At(i, 0)
		predVal := yPred.At(i, 0)

		// For binary/multiclass, round predictions
		if predVal > 0.5 {
			predVal = 1.0
		} else {
			predVal = 0.0
		}

		if trueVal == predVal {
			correct++
		}
	}

	return float64(correct) / float64(rows)
}

// QuickTrain provides a simplified training interface
// Similar to Python's quick start examples
func QuickTrain(X, y mat.Matrix, params map[string]interface{}) (*Booster, error) {
	// Create dataset
	trainData, err := NewDataset(X, y)
	if err != nil {
		return nil, err
	}

	// Set default parameters if not provided
	if params == nil {
		params = make(map[string]interface{})
	}
	if _, ok := params["objective"]; !ok {
		// Auto-detect objective
		if trainData.isBinary {
			params["objective"] = "binary"
		} else if trainData.nClasses > 2 {
			params["objective"] = "multiclass"
			params["num_class"] = trainData.nClasses
		} else {
			params["objective"] = "regression"
		}
	}
	if _, ok := params["num_iterations"]; !ok {
		params["num_iterations"] = 100
	}

	// Train model
	numRound := params["num_iterations"].(int)
	return Train(params, trainData, numRound, nil)
}
