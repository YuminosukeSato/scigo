package lightgbm

import (
	"fmt"
	"math"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/metrics"
	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"github.com/YuminosukeSato/scigo/pkg/log"
	"gonum.org/v1/gonum/mat"
)

// LGBMRegressor implements a LightGBM regressor with scikit-learn compatible API
type LGBMRegressor struct {
	model.BaseEstimator

	// Model
	Model     *Model
	Predictor *Predictor

	// Hyperparameters (matching Python LightGBM)
	NumLeaves           int     // Number of leaves in one tree
	MaxDepth            int     // Maximum tree depth
	LearningRate        float64 // Boosting learning rate
	NumIterations       int     // Number of boosting iterations
	MinChildSamples     int     // Minimum number of data in one leaf
	MinChildWeight      float64 // Minimum sum of hessians in one leaf
	Subsample           float64 // Subsample ratio of training data
	SubsampleFreq       int     // Frequency of subsample
	ColsampleBytree     float64 // Subsample ratio of columns when constructing tree
	RegAlpha            float64 // L1 regularization
	RegLambda           float64 // L2 regularization
	RandomState         int     // Random seed
	Objective           string  // Objective function (regression, regression_l1, etc.)
	Metric              string  // Evaluation metric
	NumThreads          int     // Number of threads for prediction
	Deterministic       bool    // Deterministic mode for reproducibility
	Verbosity           int     // Verbosity level
	EarlyStopping       int     // Early stopping rounds
	Alpha               float64 // For quantile and fair regression
	Lambda              float64 // For Tweedie regression
	CategoricalFeatures []int   // Indices of categorical features

	// Progress tracking
	ShowProgress bool // Show progress bar during training

	// Internal state
	// featureNames_ field reserved for future use
	// featureNames_ []string // Feature names
	nFeatures_ int // Number of features
	nSamples_  int // Number of training samples
}

// NewLGBMRegressor creates a new LightGBM regressor with default parameters
func NewLGBMRegressor() *LGBMRegressor {
	return &LGBMRegressor{
		NumLeaves:       31,
		MaxDepth:        -1, // No limit
		LearningRate:    0.1,
		NumIterations:   100,
		MinChildSamples: 20,
		MinChildWeight:  1e-3,
		Subsample:       1.0,
		SubsampleFreq:   0,
		ColsampleBytree: 1.0,
		RegAlpha:        0.0,
		RegLambda:       0.0,
		RandomState:     42,
		Objective:       "regression", // L2 regression by default
		Metric:          "l2",
		NumThreads:      -1, // Use all cores
		Deterministic:   false,
		Verbosity:       -1,
		EarlyStopping:   0,
		Alpha:           0.5, // For quantile regression
		Lambda:          1.5, // For Tweedie regression
		ShowProgress:    false,
	}
}

// WithNumLeaves sets the number of leaves
func (lgb *LGBMRegressor) WithNumLeaves(n int) *LGBMRegressor {
	lgb.NumLeaves = n
	return lgb
}

// WithMaxDepth sets the maximum depth
func (lgb *LGBMRegressor) WithMaxDepth(d int) *LGBMRegressor {
	lgb.MaxDepth = d
	return lgb
}

// WithLearningRate sets the learning rate
func (lgb *LGBMRegressor) WithLearningRate(lr float64) *LGBMRegressor {
	lgb.LearningRate = lr
	return lgb
}

// WithNumIterations sets the number of iterations
func (lgb *LGBMRegressor) WithNumIterations(n int) *LGBMRegressor {
	lgb.NumIterations = n
	return lgb
}

// WithRandomState sets the random seed
func (lgb *LGBMRegressor) WithRandomState(seed int) *LGBMRegressor {
	lgb.RandomState = seed
	return lgb
}

// WithDeterministic enables deterministic mode
func (lgb *LGBMRegressor) WithDeterministic(det bool) *LGBMRegressor {
	lgb.Deterministic = det
	return lgb
}

// WithProgressBar enables progress bar
func (lgb *LGBMRegressor) WithProgressBar() *LGBMRegressor {
	lgb.ShowProgress = true
	return lgb
}

// WithObjective sets the objective function
func (lgb *LGBMRegressor) WithObjective(obj string) *LGBMRegressor {
	lgb.Objective = obj
	return lgb
}

// WithEarlyStopping sets early stopping rounds
func (lgb *LGBMRegressor) WithEarlyStopping(rounds int) *LGBMRegressor {
	lgb.EarlyStopping = rounds
	return lgb
}

// Fit trains the LightGBM regressor
func (lgb *LGBMRegressor) Fit(X, y mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "LGBMRegressor.Fit")

	rows, cols := X.Dims()
	yRows, yCols := y.Dims()

	// Validate input dimensions
	if rows != yRows {
		return scigoErrors.NewDimensionError("Fit", rows, yRows, 0)
	}
	if yCols != 1 {
		return scigoErrors.NewDimensionError("Fit", 1, yCols, 1)
	}

	// Store dimensions
	lgb.nFeatures_ = cols
	lgb.nSamples_ = rows

	// Log training start
	logger := log.GetLoggerWithName("lightgbm.regressor")
	if lgb.ShowProgress {
		logger.Info("Training LGBMRegressor",
			"samples", rows,
			"features", cols,
			"objective", lgb.Objective,
			"metric", lgb.Metric)
	}

	// Create training parameters
	params := TrainingParams{
		NumIterations:   lgb.NumIterations,
		LearningRate:    lgb.LearningRate,
		NumLeaves:       lgb.NumLeaves,
		MaxDepth:        lgb.MaxDepth,
		MinDataInLeaf:   lgb.MinChildSamples,
		Lambda:          lgb.RegLambda,
		Alpha:           lgb.RegAlpha,
		MinGainToSplit:  1e-7,
		BaggingFraction: lgb.Subsample,
		BaggingFreq:     lgb.SubsampleFreq,
		FeatureFraction: lgb.ColsampleBytree,
		MaxBin:          255,
		MinDataInBin:    3,
		Objective:       lgb.Objective,
		NumClass:        1, // Regression always has 1 output
		Seed:            lgb.RandomState,
		Deterministic:   lgb.Deterministic,
		Verbosity:       lgb.Verbosity,
		EarlyStopping:   lgb.EarlyStopping,
	}

	// Create and run trainer
	trainer := NewTrainer(params)
	if err := trainer.Fit(X, y); err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// Get trained model
	lgb.Model = trainer.GetModel()

	// Create predictor
	lgb.Predictor = NewPredictor(lgb.Model)
	if lgb.NumThreads > 0 {
		lgb.Predictor.SetNumThreads(lgb.NumThreads)
	}
	lgb.Predictor.SetDeterministic(lgb.Deterministic)

	// Mark as fitted
	lgb.SetFitted()

	if lgb.ShowProgress {
		logger.Info("Training completed successfully")
	}

	return nil
}

// Predict makes predictions for input samples
func (lgb *LGBMRegressor) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lgb.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LGBMRegressor", "Predict")
	}

	_, cols := X.Dims()
	if cols != lgb.nFeatures_ {
		return nil, scigoErrors.NewDimensionError("Predict", lgb.nFeatures_, cols, 1)
	}

	// Use predictor for predictions
	return lgb.Predictor.Predict(X)
}

// Score returns the coefficient of determination R^2 of the prediction
func (lgb *LGBMRegressor) Score(X, y mat.Matrix) (float64, error) {
	if !lgb.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LGBMRegressor", "Score")
	}

	predictions, err := lgb.Predict(X)
	if err != nil {
		return 0, err
	}

	// Calculate R^2 score
	rows, _ := y.Dims()
	yVec := mat.NewVecDense(rows, nil)
	predVec := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		yVec.SetVec(i, y.At(i, 0))
		predVec.SetVec(i, predictions.At(i, 0))
	}
	return metrics.R2Score(yVec, predVec)
}

// LoadModel loads a pre-trained LightGBM model from file
func (lgb *LGBMRegressor) LoadModel(filepath string) error {
	model, err := LoadFromFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)

	// Set parameters from loaded model
	lgb.nFeatures_ = model.NumFeatures

	// Extract objective
	switch model.Objective {
	case RegressionL2:
		lgb.Objective = "regression"
	case RegressionL1:
		lgb.Objective = "regression_l1"
	case RegressionHuber:
		lgb.Objective = "huber"
	case RegressionFair:
		lgb.Objective = "fair"
	case RegressionPoisson:
		lgb.Objective = "poisson"
	case RegressionQuantile:
		lgb.Objective = "quantile"
	case RegressionGamma:
		lgb.Objective = "gamma"
	case RegressionTweedie:
		lgb.Objective = "tweedie"
	default:
		lgb.Objective = "regression"
	}

	lgb.SetFitted()
	return nil
}

// LoadModelFromString loads a model from string format
func (lgb *LGBMRegressor) LoadModelFromString(modelStr string) error {
	model, err := LoadFromString(modelStr)
	if err != nil {
		return fmt.Errorf("failed to load model from string: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)
	lgb.nFeatures_ = model.NumFeatures

	lgb.SetFitted()
	return nil
}

// LoadModelFromJSON loads a model from JSON data
func (lgb *LGBMRegressor) LoadModelFromJSON(jsonData []byte) error {
	model, err := LoadFromJSON(jsonData)
	if err != nil {
		return fmt.Errorf("failed to load model from JSON: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)
	lgb.nFeatures_ = model.NumFeatures

	lgb.SetFitted()
	return nil
}

// GetFeatureImportance returns feature importance scores
func (lgb *LGBMRegressor) GetFeatureImportance(importanceType string) []float64 {
	if !lgb.IsFitted() || lgb.Model == nil {
		return nil
	}

	return lgb.Model.GetFeatureImportance(importanceType)
}

// GetParams returns the parameters of the regressor
func (lgb *LGBMRegressor) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"num_leaves":        lgb.NumLeaves,
		"max_depth":         lgb.MaxDepth,
		"learning_rate":     lgb.LearningRate,
		"n_estimators":      lgb.NumIterations,
		"min_child_samples": lgb.MinChildSamples,
		"min_child_weight":  lgb.MinChildWeight,
		"subsample":         lgb.Subsample,
		"subsample_freq":    lgb.SubsampleFreq,
		"colsample_bytree":  lgb.ColsampleBytree,
		"reg_alpha":         lgb.RegAlpha,
		"reg_lambda":        lgb.RegLambda,
		"random_state":      lgb.RandomState,
		"objective":         lgb.Objective,
		"metric":            lgb.Metric,
		"n_jobs":            lgb.NumThreads,
		"deterministic":     lgb.Deterministic,
		"verbosity":         lgb.Verbosity,
		"alpha":             lgb.Alpha,
		"lambda":            lgb.Lambda,
	}
}

// SetParams sets the parameters of the regressor
func (lgb *LGBMRegressor) SetParams(params map[string]interface{}) error {
	for key, value := range params {
		switch key {
		case "num_leaves", "n_leaves":
			if v, ok := value.(int); ok {
				lgb.NumLeaves = v
			}
		case "max_depth":
			if v, ok := value.(int); ok {
				lgb.MaxDepth = v
			}
		case "learning_rate":
			if v, ok := value.(float64); ok {
				lgb.LearningRate = v
			}
		case "n_estimators", "num_iterations":
			if v, ok := value.(int); ok {
				lgb.NumIterations = v
			}
		case "min_child_samples":
			if v, ok := value.(int); ok {
				lgb.MinChildSamples = v
			}
		case "min_child_weight":
			if v, ok := value.(float64); ok {
				lgb.MinChildWeight = v
			}
		case "subsample":
			if v, ok := value.(float64); ok {
				lgb.Subsample = v
			}
		case "colsample_bytree":
			if v, ok := value.(float64); ok {
				lgb.ColsampleBytree = v
			}
		case "reg_alpha":
			if v, ok := value.(float64); ok {
				lgb.RegAlpha = v
			}
		case "reg_lambda":
			if v, ok := value.(float64); ok {
				lgb.RegLambda = v
			}
		case "random_state":
			if v, ok := value.(int); ok {
				lgb.RandomState = v
			}
		case "objective":
			if v, ok := value.(string); ok {
				lgb.Objective = v
			}
		case "metric":
			if v, ok := value.(string); ok {
				lgb.Metric = v
			}
		case "n_jobs":
			if v, ok := value.(int); ok {
				lgb.NumThreads = v
			}
		case "deterministic":
			if v, ok := value.(bool); ok {
				lgb.Deterministic = v
			}
		case "alpha":
			if v, ok := value.(float64); ok {
				lgb.Alpha = v
			}
		case "lambda":
			if v, ok := value.(float64); ok {
				lgb.Lambda = v
			}
		}
	}
	return nil
}

// PredictQuantile predicts quantiles for quantile regression
// Only works if objective is set to "quantile"
func (lgb *LGBMRegressor) PredictQuantile(X mat.Matrix, quantiles []float64) ([]mat.Matrix, error) {
	if !lgb.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LGBMRegressor", "PredictQuantile")
	}

	if lgb.Model.Objective != RegressionQuantile {
		return nil, fmt.Errorf("PredictQuantile requires objective='quantile'")
	}

	results := make([]mat.Matrix, len(quantiles))

	// For each quantile, we would need to retrain or adjust the model
	// This is a simplified implementation
	for i, q := range quantiles {
		if q <= 0 || q >= 1 {
			return nil, fmt.Errorf("quantile must be in (0, 1), got %f", q)
		}

		// Predict with the current model
		// In practice, each quantile would require a separate model
		pred, err := lgb.Predict(X)
		if err != nil {
			return nil, err
		}

		// Adjust predictions based on quantile
		// This is a simplified adjustment - proper implementation would
		// require training with the specific quantile loss
		rows, cols := pred.Dims()
		adjusted := mat.NewDense(rows, cols, nil)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				val := pred.At(r, c)
				// Simple quantile adjustment (placeholder)
				adjustment := val * (q - 0.5) * 0.1
				adjusted.Set(r, c, val+adjustment)
			}
		}

		results[i] = adjusted
	}

	return results, nil
}

// GetResiduals returns the residuals (y - y_pred) for the training data
func (lgb *LGBMRegressor) GetResiduals(X, y mat.Matrix) (mat.Matrix, error) {
	if !lgb.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LGBMRegressor", "GetResiduals")
	}

	predictions, err := lgb.Predict(X)
	if err != nil {
		return nil, err
	}

	rows, _ := y.Dims()
	residuals := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		residual := y.At(i, 0) - predictions.At(i, 0)
		residuals.Set(i, 0, residual)
	}

	return residuals, nil
}

// GetMSE returns the mean squared error
func (lgb *LGBMRegressor) GetMSE(X, y mat.Matrix) (float64, error) {
	if !lgb.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LGBMRegressor", "GetMSE")
	}

	predictions, err := lgb.Predict(X)
	if err != nil {
		return 0, err
	}

	// Convert to vectors for MSE calculation
	rows, _ := y.Dims()
	yVec := mat.NewVecDense(rows, nil)
	predVec := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		yVec.SetVec(i, y.At(i, 0))
		predVec.SetVec(i, predictions.At(i, 0))
	}
	return metrics.MSE(yVec, predVec)
}

// GetMAE returns the mean absolute error
func (lgb *LGBMRegressor) GetMAE(X, y mat.Matrix) (float64, error) {
	if !lgb.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LGBMRegressor", "GetMAE")
	}

	predictions, err := lgb.Predict(X)
	if err != nil {
		return 0, err
	}

	// Convert to vectors for MAE calculation
	rows, _ := y.Dims()
	yVec := mat.NewVecDense(rows, nil)
	predVec := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		yVec.SetVec(i, y.At(i, 0))
		predVec.SetVec(i, predictions.At(i, 0))
	}
	return metrics.MAE(yVec, predVec)
}

// GetRMSE returns the root mean squared error
func (lgb *LGBMRegressor) GetRMSE(X, y mat.Matrix) (float64, error) {
	mse, err := lgb.GetMSE(X, y)
	if err != nil {
		return 0, err
	}

	return math.Sqrt(mse), nil
}
