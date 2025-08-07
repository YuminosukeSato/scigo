package lightgbm

import (
	"fmt"
	"math"

	"github.com/YuminosukeSato/scigo/core/model"
	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"github.com/YuminosukeSato/scigo/pkg/log"
	"gonum.org/v1/gonum/mat"
)

// LGBMClassifier implements a LightGBM classifier with scikit-learn compatible API
type LGBMClassifier struct {
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
	NumClass            int     // Number of classes (set automatically)
	Objective           string  // Objective function
	Metric              string  // Evaluation metric
	NumThreads          int     // Number of threads for prediction
	Deterministic       bool    // Deterministic mode for reproducibility
	Verbosity           int     // Verbosity level
	EarlyStopping       int     // Early stopping rounds
	CategoricalFeatures []int   // Indices of categorical features

	// Progress tracking
	ShowProgress bool // Show progress bar during training

	// Internal state
	classes_      []int    // Unique class labels
	nClasses_     int      // Number of classes
	featureNames_ []string // Feature names
	nFeatures_    int      // Number of features
}

// NewLGBMClassifier creates a new LightGBM classifier with default parameters
func NewLGBMClassifier() *LGBMClassifier {
	return &LGBMClassifier{
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
		Objective:       "binary", // Will be updated based on num_class
		Metric:          "auto",
		NumThreads:      -1, // Use all cores
		Deterministic:   false,
		Verbosity:       -1,
		EarlyStopping:   0,
		ShowProgress:    false,
	}
}

// WithNumLeaves sets the number of leaves
func (lgb *LGBMClassifier) WithNumLeaves(n int) *LGBMClassifier {
	lgb.NumLeaves = n
	return lgb
}

// WithMaxDepth sets the maximum depth
func (lgb *LGBMClassifier) WithMaxDepth(d int) *LGBMClassifier {
	lgb.MaxDepth = d
	return lgb
}

// WithLearningRate sets the learning rate
func (lgb *LGBMClassifier) WithLearningRate(lr float64) *LGBMClassifier {
	lgb.LearningRate = lr
	return lgb
}

// WithNumIterations sets the number of iterations
func (lgb *LGBMClassifier) WithNumIterations(n int) *LGBMClassifier {
	lgb.NumIterations = n
	return lgb
}

// WithRandomState sets the random seed
func (lgb *LGBMClassifier) WithRandomState(seed int) *LGBMClassifier {
	lgb.RandomState = seed
	return lgb
}

// WithDeterministic enables deterministic mode
func (lgb *LGBMClassifier) WithDeterministic(det bool) *LGBMClassifier {
	lgb.Deterministic = det
	return lgb
}

// WithProgressBar enables progress bar
func (lgb *LGBMClassifier) WithProgressBar() *LGBMClassifier {
	lgb.ShowProgress = true
	return lgb
}

// WithEarlyStopping sets early stopping rounds
func (lgb *LGBMClassifier) WithEarlyStopping(rounds int) *LGBMClassifier {
	lgb.EarlyStopping = rounds
	return lgb
}

// Fit trains the LightGBM classifier
func (lgb *LGBMClassifier) Fit(X, y mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "LGBMClassifier.Fit")

	rows, cols := X.Dims()
	yRows, yCols := y.Dims()

	// Validate input dimensions
	if rows != yRows {
		return scigoErrors.NewDimensionError("Fit", rows, yRows, 0)
	}
	if yCols != 1 {
		return scigoErrors.NewDimensionError("Fit", 1, yCols, 1)
	}

	// Extract unique classes
	lgb.extractClasses(y)

	// Set objective based on number of classes
	if lgb.nClasses_ == 2 {
		lgb.Objective = "binary"
		lgb.NumClass = 1
	} else if lgb.nClasses_ > 2 {
		lgb.Objective = "multiclass"
		lgb.NumClass = lgb.nClasses_
	} else {
		return fmt.Errorf("invalid number of classes: %d", lgb.nClasses_)
	}

	// Store feature information
	lgb.nFeatures_ = cols

	// Log training start
	logger := log.GetLoggerWithName("lightgbm.classifier")
	if lgb.ShowProgress {
		logger.Info("Training LGBMClassifier",
			"samples", rows,
			"features", cols,
			"classes", lgb.nClasses_)
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
		NumClass:        lgb.NumClass,
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

// extractClasses extracts unique class labels from y
func (lgb *LGBMClassifier) extractClasses(y mat.Matrix) {
	rows, _ := y.Dims()
	classMap := make(map[int]bool)

	for i := 0; i < rows; i++ {
		label := int(y.At(i, 0))
		classMap[label] = true
	}

	// Convert map to sorted slice
	lgb.classes_ = make([]int, 0, len(classMap))
	for class := range classMap {
		lgb.classes_ = append(lgb.classes_, class)
	}

	// Sort classes
	for i := 0; i < len(lgb.classes_)-1; i++ {
		for j := i + 1; j < len(lgb.classes_); j++ {
			if lgb.classes_[i] > lgb.classes_[j] {
				lgb.classes_[i], lgb.classes_[j] = lgb.classes_[j], lgb.classes_[i]
			}
		}
	}

	lgb.nClasses_ = len(lgb.classes_)
}

// Predict makes predictions for input samples
func (lgb *LGBMClassifier) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lgb.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LGBMClassifier", "Predict")
	}

	_, cols := X.Dims()
	if cols != lgb.nFeatures_ {
		return nil, scigoErrors.NewDimensionError("Predict", lgb.nFeatures_, cols, 1)
	}

	// Get probability predictions
	proba, err := lgb.PredictProba(X)
	if err != nil {
		return nil, err
	}

	// Convert probabilities to class predictions
	rows, _ := proba.Dims()
	predictions := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		maxProb := 0.0
		maxClass := 0

		if lgb.nClasses_ == 2 {
			// Binary classification
			prob := proba.At(i, 1)
			if prob >= 0.5 {
				maxClass = lgb.classes_[1]
			} else {
				maxClass = lgb.classes_[0]
			}
		} else {
			// Multiclass
			for j := 0; j < lgb.nClasses_; j++ {
				prob := proba.At(i, j)
				if prob > maxProb {
					maxProb = prob
					maxClass = lgb.classes_[j]
				}
			}
		}

		predictions.Set(i, 0, float64(maxClass))
	}

	return predictions, nil
}

// PredictProba returns probability estimates for each class
func (lgb *LGBMClassifier) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	if !lgb.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LGBMClassifier", "PredictProba")
	}

	_, cols := X.Dims()
	if cols != lgb.nFeatures_ {
		return nil, scigoErrors.NewDimensionError("PredictProba", lgb.nFeatures_, cols, 1)
	}

	// Use predictor for probability predictions
	return lgb.Predictor.PredictProba(X)
}

// PredictLogProba returns log probability estimates for each class
func (lgb *LGBMClassifier) PredictLogProba(X mat.Matrix) (mat.Matrix, error) {
	proba, err := lgb.PredictProba(X)
	if err != nil {
		return nil, err
	}

	rows, cols := proba.Dims()
	logProba := mat.NewDense(rows, cols, nil)

	// Convert to log probabilities
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			p := proba.At(i, j)
			if p <= 0 {
				logProba.Set(i, j, math.Inf(-1))
			} else {
				logProba.Set(i, j, math.Log(p))
			}
		}
	}

	return logProba, nil
}

// Score returns the mean accuracy on the given test data and labels
func (lgb *LGBMClassifier) Score(X, y mat.Matrix) (float64, error) {
	if !lgb.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LGBMClassifier", "Score")
	}

	predictions, err := lgb.Predict(X)
	if err != nil {
		return 0, err
	}

	rows, _ := predictions.Dims()
	correct := 0

	for i := 0; i < rows; i++ {
		if predictions.At(i, 0) == y.At(i, 0) {
			correct++
		}
	}

	return float64(correct) / float64(rows), nil
}

// LoadModel loads a pre-trained LightGBM model from file
func (lgb *LGBMClassifier) LoadModel(filepath string) error {
	model, err := LoadFromFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)

	// Set parameters from loaded model
	lgb.nFeatures_ = model.NumFeatures
	lgb.NumClass = model.NumClass

	// Extract objective
	switch model.Objective {
	case BinaryLogistic, BinaryCrossEntropy:
		lgb.nClasses_ = 2
		lgb.classes_ = []int{0, 1}
	case MulticlassSoftmax:
		lgb.nClasses_ = model.NumClass
		lgb.classes_ = make([]int, lgb.nClasses_)
		for i := range lgb.classes_ {
			lgb.classes_[i] = i
		}
	}

	lgb.SetFitted()
	return nil
}

// LoadModelFromString loads a model from string format
func (lgb *LGBMClassifier) LoadModelFromString(modelStr string) error {
	model, err := LoadFromString(modelStr)
	if err != nil {
		return fmt.Errorf("failed to load model from string: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)
	lgb.nFeatures_ = model.NumFeatures

	// Set classes based on model
	if model.NumClass > 2 {
		lgb.nClasses_ = model.NumClass
		lgb.classes_ = make([]int, lgb.nClasses_)
		for i := range lgb.classes_ {
			lgb.classes_[i] = i
		}
	} else {
		lgb.nClasses_ = 2
		lgb.classes_ = []int{0, 1}
	}

	lgb.SetFitted()
	return nil
}

// LoadModelFromJSON loads a model from JSON data
func (lgb *LGBMClassifier) LoadModelFromJSON(jsonData []byte) error {
	model, err := LoadFromJSON(jsonData)
	if err != nil {
		return fmt.Errorf("failed to load model from JSON: %w", err)
	}

	lgb.Model = model
	lgb.Predictor = NewPredictor(model)
	lgb.nFeatures_ = model.NumFeatures

	// Set classes based on model
	if model.NumClass > 2 {
		lgb.nClasses_ = model.NumClass
		lgb.classes_ = make([]int, lgb.nClasses_)
		for i := range lgb.classes_ {
			lgb.classes_[i] = i
		}
	} else {
		lgb.nClasses_ = 2
		lgb.classes_ = []int{0, 1}
	}

	lgb.SetFitted()
	return nil
}

// GetFeatureImportance returns feature importance scores
func (lgb *LGBMClassifier) GetFeatureImportance(importanceType string) []float64 {
	if !lgb.IsFitted() || lgb.Model == nil {
		return nil
	}

	return lgb.Model.GetFeatureImportance(importanceType)
}

// GetParams returns the parameters of the classifier
func (lgb *LGBMClassifier) GetParams() map[string]interface{} {
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
	}
}

// SetParams sets the parameters of the classifier
func (lgb *LGBMClassifier) SetParams(params map[string]interface{}) error {
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
		case "n_jobs":
			if v, ok := value.(int); ok {
				lgb.NumThreads = v
			}
		case "deterministic":
			if v, ok := value.(bool); ok {
				lgb.Deterministic = v
			}
		}
	}
	return nil
}
