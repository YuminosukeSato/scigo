package naive_bayes

import (
	"fmt"
	"math"
	"sync"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// MultinomialNB implements the Multinomial Naive Bayes classifier
// for discrete features (e.g., word counts for text classification).
// Scikit-learn compatible implementation with partial_fit support.
type MultinomialNB struct {
	model.BaseEstimator

	// Hyperparameters
	alpha      float64   // Additive (Laplace/Lidstone) smoothing parameter
	fitPrior   bool      // Whether to learn class prior probabilities
	classPrior []float64 // Prior probabilities for each class (if not learned)

	// Learned parameters
	classes_        []int       // Unique classes
	nClasses_       int         // Number of classes
	classLogPrior_  []float64   // Log prior probabilities for each class
	featureLogProb_ [][]float64 // Log probabilities of features given class
	featureCount_   [][]float64 // Count of each feature for each class
	classCount_     []float64   // Number of samples for each class
	nFeatures_      int         // Number of features
	nSamplesSeen_   int         // Total number of samples seen

	// Internal state
	mu sync.RWMutex
}

// NewMultinomialNB creates a new Multinomial Naive Bayes classifier
func NewMultinomialNB(options ...MultinomialNBOption) *MultinomialNB {
	nb := &MultinomialNB{
		alpha:    1.0, // Default Laplace smoothing
		fitPrior: true,
	}

	for _, opt := range options {
		opt(nb)
	}

	return nb
}

// MultinomialNBOption is a configuration option for MultinomialNB
type MultinomialNBOption func(*MultinomialNB)

// WithAlpha sets the smoothing parameter (0 for no smoothing)
func WithAlpha(alpha float64) MultinomialNBOption {
	return func(nb *MultinomialNB) {
		nb.alpha = alpha
	}
}

// WithFitPrior sets whether to learn class prior probabilities
func WithFitPrior(fitPrior bool) MultinomialNBOption {
	return func(nb *MultinomialNB) {
		nb.fitPrior = fitPrior
	}
}

// WithClassPrior sets the prior probabilities for classes
func WithClassPrior(prior []float64) MultinomialNBOption {
	return func(nb *MultinomialNB) {
		nb.classPrior = prior
		nb.fitPrior = false
	}
}

// Fit trains the MultinomialNB classifier
func (nb *MultinomialNB) Fit(X, y mat.Matrix) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()

	// Reset state for fresh training
	nb.reset()

	// Validate input
	if err := nb.validateInput(X, y); err != nil {
		return err
	}

	rows, cols := X.Dims()
	nb.nFeatures_ = cols

	// Extract classes
	nb.extractClasses(y)

	// Initialize counts
	nb.featureCount_ = make([][]float64, nb.nClasses_)
	nb.classCount_ = make([]float64, nb.nClasses_)
	for i := 0; i < nb.nClasses_; i++ {
		nb.featureCount_[i] = make([]float64, nb.nFeatures_)
	}

	// Count features for each class
	for i := 0; i < rows; i++ {
		classIdx := nb.getClassIndex(int(y.At(i, 0)))
		if classIdx == -1 {
			continue
		}

		nb.classCount_[classIdx]++
		nb.nSamplesSeen_++

		for j := 0; j < cols; j++ {
			nb.featureCount_[classIdx][j] += X.At(i, j)
		}
	}

	// Update model parameters
	nb.updateModel()

	nb.SetFitted()
	return nil
}

// PartialFit performs incremental learning on a batch of samples
func (nb *MultinomialNB) PartialFit(X, y mat.Matrix, classes []int) error {
	nb.mu.Lock()
	defer nb.mu.Unlock()

	// Validate input
	if err := nb.validateInput(X, y); err != nil {
		return err
	}

	rows, cols := X.Dims()

	// First call to partial_fit
	if nb.classes_ == nil {
		nb.nFeatures_ = cols

		if classes != nil {
			// Use provided classes
			nb.classes_ = make([]int, len(classes))
			copy(nb.classes_, classes)
			nb.nClasses_ = len(classes)
		} else {
			// Extract classes from y
			nb.extractClasses(y)
		}

		// Initialize counts
		nb.featureCount_ = make([][]float64, nb.nClasses_)
		nb.classCount_ = make([]float64, nb.nClasses_)
		for i := 0; i < nb.nClasses_; i++ {
			nb.featureCount_[i] = make([]float64, nb.nFeatures_)
		}
	}

	// Check feature dimensions
	if cols != nb.nFeatures_ {
		return errors.NewDimensionError("PartialFit", nb.nFeatures_, cols, 1)
	}

	// Update counts
	for i := 0; i < rows; i++ {
		classIdx := nb.getClassIndex(int(y.At(i, 0)))
		if classIdx == -1 {
			// Unknown class in partial_fit - skip
			continue
		}

		nb.classCount_[classIdx]++
		nb.nSamplesSeen_++

		for j := 0; j < cols; j++ {
			nb.featureCount_[classIdx][j] += X.At(i, j)
		}
	}

	// Update model parameters
	nb.updateModel()

	nb.SetFitted()
	return nil
}

// updateModel updates the model parameters based on current counts
func (nb *MultinomialNB) updateModel() {
	// Calculate class log priors
	nb.classLogPrior_ = make([]float64, nb.nClasses_)

	if nb.fitPrior {
		// Learn from data
		totalCount := 0.0
		for i := 0; i < nb.nClasses_; i++ {
			totalCount += nb.classCount_[i]
		}

		for i := 0; i < nb.nClasses_; i++ {
			nb.classLogPrior_[i] = math.Log(nb.classCount_[i] / totalCount)
		}
	} else if nb.classPrior != nil {
		// Use provided priors
		for i := 0; i < nb.nClasses_ && i < len(nb.classPrior); i++ {
			nb.classLogPrior_[i] = math.Log(nb.classPrior[i])
		}
	} else {
		// Uniform prior
		uniformPrior := math.Log(1.0 / float64(nb.nClasses_))
		for i := 0; i < nb.nClasses_; i++ {
			nb.classLogPrior_[i] = uniformPrior
		}
	}

	// Calculate feature log probabilities
	nb.featureLogProb_ = make([][]float64, nb.nClasses_)

	for i := 0; i < nb.nClasses_; i++ {
		nb.featureLogProb_[i] = make([]float64, nb.nFeatures_)

		// Calculate total count for this class (with smoothing)
		totalCount := 0.0
		for j := 0; j < nb.nFeatures_; j++ {
			totalCount += nb.featureCount_[i][j] + nb.alpha
		}

		// Calculate log probabilities
		for j := 0; j < nb.nFeatures_; j++ {
			nb.featureLogProb_[i][j] = math.Log((nb.featureCount_[i][j] + nb.alpha) / totalCount)
		}
	}
}

// Predict performs classification on samples in X
func (nb *MultinomialNB) Predict(X mat.Matrix) (mat.Matrix, error) {
	nb.mu.RLock()
	defer nb.mu.RUnlock()

	if !nb.IsFitted() {
		return nil, errors.NewNotFittedError("MultinomialNB", "Predict")
	}

	rows, cols := X.Dims()
	if cols != nb.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", nb.nFeatures_, cols, 1)
	}

	predictions := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		maxLogProb := math.Inf(-1)
		predictedClass := nb.classes_[0]

		// Calculate log probability for each class
		for c := 0; c < nb.nClasses_; c++ {
			logProb := nb.classLogPrior_[c]

			for j := 0; j < cols; j++ {
				count := X.At(i, j)
				if count < 0 {
					return nil, errors.NewValueError("Predict", "MultinomialNB requires non-negative features")
				}
				logProb += count * nb.featureLogProb_[c][j]
			}

			if logProb > maxLogProb {
				maxLogProb = logProb
				predictedClass = nb.classes_[c]
			}
		}

		predictions.Set(i, 0, float64(predictedClass))
	}

	return predictions, nil
}

// PredictProba returns probability estimates for samples in X
func (nb *MultinomialNB) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	logProba, err := nb.PredictLogProba(X)
	if err != nil {
		return nil, err
	}

	rows, cols := logProba.Dims()
	proba := mat.NewDense(rows, cols, nil)

	// Convert log probabilities to probabilities
	for i := 0; i < rows; i++ {
		// Find max for numerical stability
		maxLogProb := math.Inf(-1)
		for j := 0; j < cols; j++ {
			if logProba.At(i, j) > maxLogProb {
				maxLogProb = logProba.At(i, j)
			}
		}

		// Calculate exp(log_prob - max) and sum
		sum := 0.0
		for j := 0; j < cols; j++ {
			val := math.Exp(logProba.At(i, j) - maxLogProb)
			proba.Set(i, j, val)
			sum += val
		}

		// Normalize
		for j := 0; j < cols; j++ {
			proba.Set(i, j, proba.At(i, j)/sum)
		}
	}

	return proba, nil
}

// PredictLogProba returns log probability estimates for samples in X
func (nb *MultinomialNB) PredictLogProba(X mat.Matrix) (mat.Matrix, error) {
	nb.mu.RLock()
	defer nb.mu.RUnlock()

	if !nb.IsFitted() {
		return nil, errors.NewNotFittedError("MultinomialNB", "PredictLogProba")
	}

	rows, cols := X.Dims()
	if cols != nb.nFeatures_ {
		return nil, errors.NewDimensionError("PredictLogProba", nb.nFeatures_, cols, 1)
	}

	logProba := mat.NewDense(rows, nb.nClasses_, nil)

	for i := 0; i < rows; i++ {
		// Calculate log probability for each class
		for c := 0; c < nb.nClasses_; c++ {
			logProb := nb.classLogPrior_[c]

			for j := 0; j < cols; j++ {
				count := X.At(i, j)
				if count < 0 {
					return nil, errors.NewValueError("PredictLogProba", "MultinomialNB requires non-negative features")
				}
				logProb += count * nb.featureLogProb_[c][j]
			}

			logProba.Set(i, c, logProb)
		}

		// Normalize log probabilities
		maxLogProb := math.Inf(-1)
		for c := 0; c < nb.nClasses_; c++ {
			if logProba.At(i, c) > maxLogProb {
				maxLogProb = logProba.At(i, c)
			}
		}

		// Compute log(sum(exp(log_prob - max))) + max
		sumExp := 0.0
		for c := 0; c < nb.nClasses_; c++ {
			sumExp += math.Exp(logProba.At(i, c) - maxLogProb)
		}
		logSumExp := math.Log(sumExp) + maxLogProb

		// Normalize
		for c := 0; c < nb.nClasses_; c++ {
			logProba.Set(i, c, logProba.At(i, c)-logSumExp)
		}
	}

	return logProba, nil
}

// Score returns the mean accuracy on the given test data and labels
func (nb *MultinomialNB) Score(X, y mat.Matrix) (float64, error) {
	predictions, err := nb.Predict(X)
	if err != nil {
		return 0, err
	}

	rows, _ := y.Dims()
	correct := 0

	for i := 0; i < rows; i++ {
		if int(predictions.At(i, 0)) == int(y.At(i, 0)) {
			correct++
		}
	}

	return float64(correct) / float64(rows), nil
}

// Classes returns the class labels
func (nb *MultinomialNB) Classes() []int {
	nb.mu.RLock()
	defer nb.mu.RUnlock()

	if nb.classes_ == nil {
		return nil
	}

	classes := make([]int, len(nb.classes_))
	copy(classes, nb.classes_)
	return classes
}

// NSamplesSeen returns the number of samples seen during training
func (nb *MultinomialNB) NSamplesSeen() int {
	nb.mu.RLock()
	defer nb.mu.RUnlock()
	return nb.nSamplesSeen_
}

// FeatureLogProb returns the log probability of features given classes
func (nb *MultinomialNB) FeatureLogProb() [][]float64 {
	nb.mu.RLock()
	defer nb.mu.RUnlock()

	if nb.featureLogProb_ == nil {
		return nil
	}

	// Deep copy
	result := make([][]float64, len(nb.featureLogProb_))
	for i := range nb.featureLogProb_ {
		result[i] = make([]float64, len(nb.featureLogProb_[i]))
		copy(result[i], nb.featureLogProb_[i])
	}
	return result
}

// ClassLogPrior returns the log prior probabilities of classes
func (nb *MultinomialNB) ClassLogPrior() []float64 {
	nb.mu.RLock()
	defer nb.mu.RUnlock()

	if nb.classLogPrior_ == nil {
		return nil
	}

	result := make([]float64, len(nb.classLogPrior_))
	copy(result, nb.classLogPrior_)
	return result
}

// validateInput validates the input data
func (nb *MultinomialNB) validateInput(X, y mat.Matrix) error {
	xRows, _ := X.Dims()
	yRows, yCols := y.Dims()

	if xRows != yRows {
		return fmt.Errorf("X and y must have the same number of samples: got %d and %d", xRows, yRows)
	}

	if yCols != 1 {
		return fmt.Errorf("y must be a column vector: got shape (%d, %d)", yRows, yCols)
	}

	// Check for negative values
	rows, cols := X.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if X.At(i, j) < 0 {
				return errors.NewValueError("Fit", "MultinomialNB requires non-negative features")
			}
		}
	}

	return nil
}

// extractClasses extracts unique classes from y
func (nb *MultinomialNB) extractClasses(y mat.Matrix) {
	rows, _ := y.Dims()
	classSet := make(map[int]bool)

	for i := 0; i < rows; i++ {
		class := int(y.At(i, 0))
		classSet[class] = true
	}

	classes := make([]int, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}

	// Sort classes
	for i := 0; i < len(classes); i++ {
		for j := i + 1; j < len(classes); j++ {
			if classes[i] > classes[j] {
				classes[i], classes[j] = classes[j], classes[i]
			}
		}
	}

	nb.classes_ = classes
	nb.nClasses_ = len(classes)
}

// getClassIndex returns the index of a class
func (nb *MultinomialNB) getClassIndex(class int) int {
	for i, c := range nb.classes_ {
		if c == class {
			return i
		}
	}
	return -1
}

// reset resets the internal state
func (nb *MultinomialNB) reset() {
	nb.classes_ = nil
	nb.nClasses_ = 0
	nb.classLogPrior_ = nil
	nb.featureLogProb_ = nil
	nb.featureCount_ = nil
	nb.classCount_ = nil
	nb.nFeatures_ = 0
	nb.nSamplesSeen_ = 0
	nb.Reset() // Reset BaseEstimator
}
