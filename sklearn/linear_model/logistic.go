package linear_model

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/YuminosukeSato/scigo/core/model"
	"gonum.org/v1/gonum/mat"
)

// LogisticRegression implements logistic regression for classification
// Compatible with scikit-learn's LogisticRegression
type LogisticRegression struct {
	state *model.StateManager // State management (composition)

	// Hyperparameters
	penalty          string  // Regularization: "l2", "l1", "elasticnet", "none"
	C                float64 // Inverse regularization strength (1/alpha)
	fitIntercept     bool    // Whether to fit intercept
	interceptScaling float64 // Intercept scaling
	classWeight      string  // Class weight: "balanced", "none"
	randomState      int64   // Random seed
	solver           string  // Solver: "lbfgs", "liblinear", "newton-cg", "sag", "saga"
	maxIter          int     // Maximum iterations
	multiClass       string  // Multi-class: "auto", "ovr", "multinomial"
	verbose          int     // Verbosity level
	warmStart        bool    // Reuse previous solution
	l1Ratio          float64 // L1 ratio for elastic net
	tol              float64 // Tolerance for stopping

	// Model parameters
	coef_      [][]float64 // Coefficients (n_classes x n_features or 1 x n_features for binary)
	intercept_ []float64   // Intercept terms
	classes_   []int       // Unique class labels
	nClasses_  int         // Number of classes
	nFeatures_ int         // Number of features
	nIter_     []int       // Actual iterations per class

	// Internal state
	rand *rand.Rand
}

// LogisticRegressionOption is a functional option for LogisticRegression
type LogisticRegressionOption func(*LogisticRegression)

// NewLogisticRegression creates a new LogisticRegression classifier
func NewLogisticRegression(opts ...LogisticRegressionOption) *LogisticRegression {
	lr := &LogisticRegression{
		state:            model.NewStateManager(),
		penalty:          "l2",
		C:                1.0,
		fitIntercept:     true,
		interceptScaling: 1.0,
		classWeight:      "none",
		randomState:      -1,
		solver:           "lbfgs",
		maxIter:          100,
		multiClass:       "auto",
		verbose:          0,
		warmStart:        false,
		l1Ratio:          0.5,
		tol:              1e-4,
	}

	// Apply options
	for _, opt := range opts {
		opt(lr)
	}

	// Initialize random generator if seed is set
	if lr.randomState >= 0 {
		lr.rand = rand.New(rand.NewSource(lr.randomState))
	} else {
		lr.rand = rand.New(rand.NewSource(rand.Int63()))
	}

	return lr
}

// Option functions

// WithLRPenalty sets the regularization type
func WithLRPenalty(penalty string) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.penalty = penalty
	}
}

// WithLRC sets the inverse regularization strength
func WithLRC(c float64) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.C = c
	}
}

// WithLogisticFitIntercept sets whether to fit intercept
func WithLogisticFitIntercept(fit bool) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.fitIntercept = fit
	}
}

// WithLRSolver sets the optimization solver
func WithLRSolver(solver string) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.solver = solver
	}
}

// WithMaxIter sets the maximum number of iterations
func WithLRMaxIter(maxIter int) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.maxIter = maxIter
	}
}

// WithTol sets the tolerance for stopping criteria
func WithLRTol(tol float64) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.tol = tol
	}
}

// WithRandomState sets the random seed
func WithLRRandomState(seed int64) LogisticRegressionOption {
	return func(lr *LogisticRegression) {
		lr.randomState = seed
		if seed >= 0 {
			lr.rand = rand.New(rand.NewSource(seed))
		}
	}
}

// Fit trains the logistic regression model
func (lr *LogisticRegression) Fit(X, y mat.Matrix) error {
	// Validate inputs
	nSamples, nFeatures := X.Dims()
	yRows, yCols := y.Dims()

	if nSamples != yRows {
		return fmt.Errorf("x and y must have the same number of samples: got %d and %d", nSamples, yRows)
	}

	if yCols != 1 {
		return fmt.Errorf("y must be a column vector: got shape (%d, %d)", yRows, yCols)
	}

	// Extract unique classes
	lr.extractClasses(y)
	lr.nFeatures_ = nFeatures

	// Initialize coefficients
	if !lr.warmStart || lr.coef_ == nil {
		lr.initializeWeights(nFeatures)
	}

	// Binary or multiclass classification
	if lr.nClasses_ == 2 {
		// Binary classification
		err := lr.fitBinary(X, y)
		if err != nil {
			return err
		}
	} else {
		// Multiclass classification
		if lr.multiClass == "multinomial" && lr.solver == "lbfgs" {
			err := lr.fitMultinomial(X, y)
			if err != nil {
				return err
			}
		} else {
			// One-vs-rest
			err := lr.fitOVR(X, y)
			if err != nil {
				return err
			}
		}
	}

	lr.state.SetFitted()
	return nil
}

// extractClasses identifies unique class labels
func (lr *LogisticRegression) extractClasses(y mat.Matrix) {
	rows, _ := y.Dims()
	classMap := make(map[int]bool)

	for i := 0; i < rows; i++ {
		label := int(y.At(i, 0))
		classMap[label] = true
	}

	lr.classes_ = make([]int, 0, len(classMap))
	for class := range classMap {
		lr.classes_ = append(lr.classes_, class)
	}

	// Sort classes for consistency
	for i := 0; i < len(lr.classes_)-1; i++ {
		for j := i + 1; j < len(lr.classes_); j++ {
			if lr.classes_[i] > lr.classes_[j] {
				lr.classes_[i], lr.classes_[j] = lr.classes_[j], lr.classes_[i]
			}
		}
	}

	lr.nClasses_ = len(lr.classes_)
}

// initializeWeights initializes model weights
func (lr *LogisticRegression) initializeWeights(nFeatures int) {
	if lr.nClasses_ == 2 {
		// Binary classification: single set of weights
		lr.coef_ = make([][]float64, 1)
		lr.coef_[0] = make([]float64, nFeatures)
		lr.intercept_ = make([]float64, 1)
	} else {
		// Multiclass: one set of weights per class
		lr.coef_ = make([][]float64, lr.nClasses_)
		for i := range lr.coef_ {
			lr.coef_[i] = make([]float64, nFeatures)
		}
		lr.intercept_ = make([]float64, lr.nClasses_)
	}

	lr.nIter_ = make([]int, len(lr.coef_))

	// Initialize with small random values
	for i := range lr.coef_ {
		for j := range lr.coef_[i] {
			lr.coef_[i][j] = lr.rand.NormFloat64() * 0.01
		}
	}
}

// fitBinary fits binary logistic regression using gradient descent
func (lr *LogisticRegression) fitBinary(X, y mat.Matrix) error {
	nSamples, nFeatures := X.Dims()

	// Convert labels to 0/1
	yBinary := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		if int(y.At(i, 0)) == lr.classes_[1] {
			yBinary.Set(i, 0, 1.0)
		} else {
			yBinary.Set(i, 0, 0.0)
		}
	}

	// Gradient descent with improved learning rate
	weights := lr.coef_[0]
	intercept := &lr.intercept_[0]

	// Better learning rate schedule
	baseLearningRate := 1.0

	for iter := 0; iter < lr.maxIter; iter++ {
		// Compute predictions
		predictions := mat.NewDense(nSamples, 1, nil)
		for i := 0; i < nSamples; i++ {
			z := *intercept
			for j := 0; j < nFeatures; j++ {
				z += X.At(i, j) * weights[j]
			}
			predictions.Set(i, 0, sigmoid(z))
		}

		// Compute gradients
		gradWeights := make([]float64, nFeatures)
		gradIntercept := 0.0

		for i := 0; i < nSamples; i++ {
			error := predictions.At(i, 0) - yBinary.At(i, 0)
			gradIntercept += error
			for j := 0; j < nFeatures; j++ {
				gradWeights[j] += error * X.At(i, j)
			}
		}

		// Scale gradients by number of samples
		for j := range gradWeights {
			gradWeights[j] /= float64(nSamples)
		}
		gradIntercept /= float64(nSamples)

		// Add L2 regularization gradient
		if lr.penalty == "l2" {
			lambda := 1.0 / lr.C
			for j := range weights {
				gradWeights[j] += lambda * weights[j]
			}
		}

		// Adaptive learning rate
		learningRate := baseLearningRate / (1.0 + 0.1*float64(iter))

		// Update weights
		for j := range weights {
			weights[j] -= learningRate * gradWeights[j]
		}
		if lr.fitIntercept {
			*intercept -= learningRate * gradIntercept
		}

		lr.nIter_[0] = iter + 1

		// Check convergence
		maxGrad := math.Abs(gradIntercept)
		for _, g := range gradWeights {
			if math.Abs(g) > maxGrad {
				maxGrad = math.Abs(g)
			}
		}
		if maxGrad < lr.tol {
			break
		}
	}

	return nil
}

// fitOVR fits one-vs-rest multiclass classification
func (lr *LogisticRegression) fitOVR(X, y mat.Matrix) error {
	nSamples, _ := X.Dims()

	for classIdx, class := range lr.classes_ {
		// Create binary labels for this class
		yBinary := mat.NewDense(nSamples, 1, nil)
		for i := 0; i < nSamples; i++ {
			if int(y.At(i, 0)) == class {
				yBinary.Set(i, 0, 1.0)
			} else {
				yBinary.Set(i, 0, 0.0)
			}
		}

		// Fit binary classifier for this class
		err := lr.fitBinaryForClass(X, yBinary, classIdx)
		if err != nil {
			return fmt.Errorf("failed to fit class %d: %v", class, err)
		}
	}

	return nil
}

// fitBinaryForClass fits a binary classifier for a specific class in OVR
func (lr *LogisticRegression) fitBinaryForClass(X, yBinary mat.Matrix, classIdx int) error {
	nSamples, nFeatures := X.Dims()
	weights := lr.coef_[classIdx]
	intercept := &lr.intercept_[classIdx]

	// Better learning rate for OVR
	baseLearningRate := 1.0

	for iter := 0; iter < lr.maxIter; iter++ {
		// Similar to fitBinary but for specific class
		predictions := mat.NewDense(nSamples, 1, nil)
		for i := 0; i < nSamples; i++ {
			z := *intercept
			for j := 0; j < nFeatures; j++ {
				z += X.At(i, j) * weights[j]
			}
			predictions.Set(i, 0, sigmoid(z))
		}

		// Compute gradients
		gradWeights := make([]float64, nFeatures)
		gradIntercept := 0.0

		for i := 0; i < nSamples; i++ {
			error := predictions.At(i, 0) - yBinary.At(i, 0)
			gradIntercept += error
			for j := 0; j < nFeatures; j++ {
				gradWeights[j] += error * X.At(i, j)
			}
		}

		// Scale gradients
		for j := range gradWeights {
			gradWeights[j] /= float64(nSamples)
		}
		gradIntercept /= float64(nSamples)

		// Add L2 regularization
		if lr.penalty == "l2" {
			lambda := 1.0 / lr.C
			for j := range weights {
				gradWeights[j] += lambda * weights[j]
			}
		}

		// Adaptive learning rate
		learningRate := baseLearningRate / (1.0 + 0.1*float64(iter))

		// Update weights
		for j := range weights {
			weights[j] -= learningRate * gradWeights[j]
		}
		if lr.fitIntercept {
			*intercept -= learningRate * gradIntercept
		}

		lr.nIter_[classIdx] = iter + 1

		// Check convergence
		maxGrad := math.Abs(gradIntercept)
		for _, g := range gradWeights {
			if math.Abs(g) > maxGrad {
				maxGrad = math.Abs(g)
			}
		}
		if maxGrad < lr.tol {
			break
		}
	}

	return nil
}

// fitMultinomial fits multinomial logistic regression
func (lr *LogisticRegression) fitMultinomial(X, y mat.Matrix) error {
	// TODO: Implement multinomial logistic regression
	return fmt.Errorf("multinomial logistic regression not yet implemented")
}

// Predict makes predictions for input data
func (lr *LogisticRegression) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lr.state.IsFitted() {
		return nil, fmt.Errorf("model must be fitted before prediction")
	}

	nSamples, _ := X.Dims()
	predictions := mat.NewDense(nSamples, 1, nil)

	if lr.nClasses_ == 2 {
		// Binary classification
		for i := 0; i < nSamples; i++ {
			z := lr.intercept_[0]
			for j := 0; j < lr.nFeatures_; j++ {
				z += X.At(i, j) * lr.coef_[0][j]
			}
			prob := sigmoid(z)
			if prob >= 0.5 {
				predictions.Set(i, 0, float64(lr.classes_[1]))
			} else {
				predictions.Set(i, 0, float64(lr.classes_[0]))
			}
		}
	} else {
		// Multiclass classification
		for i := 0; i < nSamples; i++ {
			maxScore := -math.MaxFloat64
			bestClass := 0

			for classIdx := 0; classIdx < lr.nClasses_; classIdx++ {
				score := lr.intercept_[classIdx]
				for j := 0; j < lr.nFeatures_; j++ {
					score += X.At(i, j) * lr.coef_[classIdx][j]
				}
				if score > maxScore {
					maxScore = score
					bestClass = classIdx
				}
			}
			predictions.Set(i, 0, float64(lr.classes_[bestClass]))
		}
	}

	return predictions, nil
}

// PredictProba returns probability estimates for each class
func (lr *LogisticRegression) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	if !lr.state.IsFitted() {
		return nil, fmt.Errorf("model must be fitted before prediction")
	}

	nSamples, _ := X.Dims()
	probas := mat.NewDense(nSamples, lr.nClasses_, nil)

	if lr.nClasses_ == 2 {
		// Binary classification
		for i := 0; i < nSamples; i++ {
			z := lr.intercept_[0]
			for j := 0; j < lr.nFeatures_; j++ {
				z += X.At(i, j) * lr.coef_[0][j]
			}
			prob1 := sigmoid(z)
			probas.Set(i, 0, 1.0-prob1)
			probas.Set(i, 1, prob1)
		}
	} else {
		// Multiclass using softmax
		for i := 0; i < nSamples; i++ {
			scores := make([]float64, lr.nClasses_)
			maxScore := -math.MaxFloat64

			// Compute scores
			for classIdx := 0; classIdx < lr.nClasses_; classIdx++ {
				score := lr.intercept_[classIdx]
				for j := 0; j < lr.nFeatures_; j++ {
					score += X.At(i, j) * lr.coef_[classIdx][j]
				}
				scores[classIdx] = score
				if score > maxScore {
					maxScore = score
				}
			}

			// Apply softmax
			sum := 0.0
			for classIdx := 0; classIdx < lr.nClasses_; classIdx++ {
				scores[classIdx] = math.Exp(scores[classIdx] - maxScore)
				sum += scores[classIdx]
			}

			for classIdx := 0; classIdx < lr.nClasses_; classIdx++ {
				probas.Set(i, classIdx, scores[classIdx]/sum)
			}
		}
	}

	return probas, nil
}

// Score returns the mean accuracy on the given test data and labels
func (lr *LogisticRegression) Score(X, y mat.Matrix) float64 {
	predictions, err := lr.Predict(X)
	if err != nil {
		return 0.0
	}

	nSamples, _ := X.Dims()
	correct := 0

	for i := 0; i < nSamples; i++ {
		if predictions.At(i, 0) == y.At(i, 0) {
			correct++
		}
	}

	return float64(correct) / float64(nSamples)
}

// GetParams returns the model hyperparameters
func (lr *LogisticRegression) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"penalty":           lr.penalty,
		"C":                 lr.C,
		"fit_intercept":     lr.fitIntercept,
		"intercept_scaling": lr.interceptScaling,
		"class_weight":      lr.classWeight,
		"random_state":      lr.randomState,
		"solver":            lr.solver,
		"max_iter":          lr.maxIter,
		"multi_class":       lr.multiClass,
		"verbose":           lr.verbose,
		"warm_start":        lr.warmStart,
		"l1_ratio":          lr.l1Ratio,
		"tol":               lr.tol,
	}
}

// SetParams sets the model hyperparameters
func (lr *LogisticRegression) SetParams(params map[string]interface{}) error {
	for key, value := range params {
		switch key {
		case "penalty":
			lr.penalty = value.(string)
		case "C":
			lr.C = value.(float64)
		case "fit_intercept":
			lr.fitIntercept = value.(bool)
		case "intercept_scaling":
			lr.interceptScaling = value.(float64)
		case "class_weight":
			lr.classWeight = value.(string)
		case "random_state":
			lr.randomState = value.(int64)
			if lr.randomState >= 0 {
				lr.rand = rand.New(rand.NewSource(lr.randomState))
			}
		case "solver":
			lr.solver = value.(string)
		case "max_iter":
			lr.maxIter = value.(int)
		case "multi_class":
			lr.multiClass = value.(string)
		case "verbose":
			lr.verbose = value.(int)
		case "warm_start":
			lr.warmStart = value.(bool)
		case "l1_ratio":
			lr.l1Ratio = value.(float64)
		case "tol":
			lr.tol = value.(float64)
		default:
			return fmt.Errorf("unknown parameter: %s", key)
		}
	}
	return nil
}

// sigmoid computes the sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
