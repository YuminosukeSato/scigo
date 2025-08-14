package lightgbm

import (
	"fmt"
	"math"
	"sync"
)

// ObjectiveFunction defines the interface for different objective functions
type ObjectiveFunction interface {
	// CalculateGradient calculates the gradient for a single sample
	CalculateGradient(prediction, target float64) float64

	// CalculateHessian calculates the hessian for a single sample
	CalculateHessian(prediction, target float64) float64

	// CalculateLoss calculates the loss for a single sample
	CalculateLoss(prediction, target float64) float64

	// GetInitScore returns the initial score for this objective
	GetInitScore(targets []float64) float64

	// Name returns the name of the objective
	Name() string
}

// L2Objective implements L2 (Mean Squared Error) loss
type L2Objective struct{}

func NewL2Objective() *L2Objective {
	return &L2Objective{}
}

func (o *L2Objective) CalculateGradient(prediction, target float64) float64 {
	return prediction - target
}

func (o *L2Objective) CalculateHessian(prediction, target float64) float64 {
	return 1.0
}

func (o *L2Objective) CalculateLoss(prediction, target float64) float64 {
	diff := prediction - target
	return 0.5 * diff * diff
}

func (o *L2Objective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	return sum / float64(len(targets))
}

func (o *L2Objective) Name() string {
	return "regression"
}

// L1Objective implements L1 (Mean Absolute Error) loss
type L1Objective struct {
	epsilon float64 // Small value to approximate the non-differentiable point
}

func NewL1Objective() *L1Objective {
	return &L1Objective{
		epsilon: 1e-7,
	}
}

func (o *L1Objective) CalculateGradient(prediction, target float64) float64 {
	diff := prediction - target
	if math.Abs(diff) < o.epsilon {
		return 0.0
	}
	if diff > 0 {
		return 1.0
	}
	return -1.0
}

func (o *L1Objective) CalculateHessian(prediction, target float64) float64 {
	// L1 has zero second derivative except at the non-differentiable point
	// Use a small positive value for numerical stability
	// LightGBM uses 1.0 as the default hessian for L1
	return 1.0
}

func (o *L1Objective) CalculateLoss(prediction, target float64) float64 {
	return math.Abs(prediction - target)
}

func (o *L1Objective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// For L1, use median as initial score
	return calculateMedian(targets)
}

func (o *L1Objective) Name() string {
	return "regression_l1"
}

// HuberObjective implements Huber loss (combination of L1 and L2)
type HuberObjective struct {
	delta float64 // Threshold for switching between L1 and L2
}

func NewHuberObjective(delta float64) *HuberObjective {
	if delta <= 0 {
		delta = 1.0 // Default delta value
	}
	return &HuberObjective{
		delta: delta,
	}
}

func (o *HuberObjective) CalculateGradient(prediction, target float64) float64 {
	diff := prediction - target
	absDiff := math.Abs(diff)

	if absDiff <= o.delta {
		// L2 region
		return diff
	} else {
		// L1 region
		if diff > 0 {
			return o.delta
		}
		return -o.delta
	}
}

func (o *HuberObjective) CalculateHessian(prediction, target float64) float64 {
	absDiff := math.Abs(prediction - target)

	if absDiff <= o.delta {
		// L2 region
		return 1.0
	} else {
		// L1 region - use small positive value for stability
		return 1e-7
	}
}

func (o *HuberObjective) CalculateLoss(prediction, target float64) float64 {
	diff := prediction - target
	absDiff := math.Abs(diff)

	if absDiff <= o.delta {
		// L2 region
		return 0.5 * diff * diff
	} else {
		// L1 region
		return o.delta * (absDiff - 0.5*o.delta)
	}
}

func (o *HuberObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use mean for Huber loss
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	return sum / float64(len(targets))
}

func (o *HuberObjective) Name() string {
	return "huber"
}

// QuantileObjective implements Quantile regression loss
type QuantileObjective struct {
	alpha float64 // Quantile level (0 < alpha < 1)
}

func NewQuantileObjective(alpha float64) *QuantileObjective {
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.5 // Default to median
	}
	return &QuantileObjective{
		alpha: alpha,
	}
}

func (o *QuantileObjective) CalculateGradient(prediction, target float64) float64 {
	diff := prediction - target
	if diff > 0 {
		return o.alpha
	} else if diff < 0 {
		return o.alpha - 1.0
	}
	return 0.0
}

func (o *QuantileObjective) CalculateHessian(prediction, target float64) float64 {
	// Quantile loss has zero second derivative except at the non-differentiable point
	// Use a constant positive value for numerical stability
	// LightGBM uses 1.0 as the default
	return 1.0
}

func (o *QuantileObjective) CalculateLoss(prediction, target float64) float64 {
	diff := prediction - target
	if diff > 0 {
		return o.alpha * diff
	}
	return (o.alpha - 1.0) * diff
}

func (o *QuantileObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Calculate the alpha-quantile of targets
	return calculateQuantile(targets, o.alpha)
}

func (o *QuantileObjective) Name() string {
	return "quantile"
}

// FairObjective implements Fair loss (another robust loss function)
type FairObjective struct {
	c float64 // Scale parameter
}

func NewFairObjective(c float64) *FairObjective {
	if c <= 0 {
		c = 1.0 // Default scale
	}
	return &FairObjective{
		c: c,
	}
}

func (o *FairObjective) CalculateGradient(prediction, target float64) float64 {
	diff := prediction - target
	return o.c * diff / (math.Abs(diff) + o.c)
}

func (o *FairObjective) CalculateHessian(prediction, target float64) float64 {
	diff := prediction - target
	absDiff := math.Abs(diff)
	denominator := absDiff + o.c
	return o.c * o.c / (denominator * denominator)
}

func (o *FairObjective) CalculateLoss(prediction, target float64) float64 {
	diff := math.Abs(prediction - target)
	return o.c * o.c * (diff/o.c - math.Log(1+diff/o.c))
}

func (o *FairObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use median for robust loss
	return calculateMedian(targets)
}

func (o *FairObjective) Name() string {
	return "fair"
}

// PoissonObjective implements Poisson regression loss
type PoissonObjective struct {
	maxOutputExp float64 // Maximum value for exp(output) to prevent overflow
}

func NewPoissonObjective() *PoissonObjective {
	return &PoissonObjective{
		maxOutputExp: 700.0, // exp(700) is close to max float64
	}
}

func (o *PoissonObjective) CalculateGradient(prediction, target float64) float64 {
	// Gradient of Poisson loss: exp(pred) - target
	predExp := math.Exp(math.Min(prediction, o.maxOutputExp))
	return predExp - target
}

func (o *PoissonObjective) CalculateHessian(prediction, target float64) float64 {
	// Hessian of Poisson loss: exp(pred)
	return math.Exp(math.Min(prediction, o.maxOutputExp))
}

func (o *PoissonObjective) CalculateLoss(prediction, target float64) float64 {
	predExp := math.Exp(math.Min(prediction, o.maxOutputExp))
	return predExp - target*prediction
}

func (o *PoissonObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use log of mean for Poisson
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	mean := sum / float64(len(targets))
	if mean <= 0 {
		return -10.0 // Avoid log(0)
	}
	return math.Log(mean)
}

func (o *PoissonObjective) Name() string {
	return "poisson"
}

// Helper functions

func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Make a copy to avoid modifying original
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple quickselect for median
	n := len(sorted)
	k := n / 2

	// Partial sort to find median
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
		if i >= k {
			break
		}
	}

	if n%2 == 1 {
		return sorted[k]
	}
	return (sorted[k-1] + sorted[k]) / 2.0
}

func calculateQuantile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Make a copy to avoid modifying original
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Sort values
	n := len(sorted)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Calculate quantile position
	pos := q * float64(n-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))

	if lower == upper {
		return sorted[lower]
	}

	// Linear interpolation
	weight := pos - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// CreateObjectiveFunction creates an objective function based on the objective name
func CreateObjectiveFunction(objective string, params *TrainingParams) (ObjectiveFunction, error) {
	switch objective {
	case "regression", "regression_l2", "l2", "mean_squared_error", "mse":
		return NewL2Objective(), nil
	case "regression_l1", "l1", "mean_absolute_error", "mae":
		return NewL1Objective(), nil
	case "huber":
		delta := 1.0
		if params != nil && params.HuberDelta > 0 {
			delta = params.HuberDelta
		}
		return NewHuberObjective(delta), nil
	case "fair":
		c := 1.0
		if params != nil && params.FairC > 0 {
			c = params.FairC
		}
		return NewFairObjective(c), nil
	case "poisson":
		return NewPoissonObjective(), nil
	case "quantile":
		alpha := 0.5
		if params != nil && params.QuantileAlpha > 0 && params.QuantileAlpha < 1 {
			alpha = params.QuantileAlpha
		}
		return NewQuantileObjective(alpha), nil
	case "binary", "binary_logloss", "logistic":
		// For binary classification, use L2 objective as placeholder
		// The actual binary logloss would require sigmoid transformation
		return NewL2Objective(), nil
	case "multiclass", "softmax", "multiclassova":
		// For multiclass, use L2 objective as placeholder
		return NewL2Objective(), nil
	case "multiclass_logloss":
		// Use MulticlassLogLoss objective for proper multiclass support
		numClass := 3 // Default, should be provided via params
		if params != nil && params.NumClass > 0 {
			numClass = params.NumClass
		}
		return NewMulticlassLogLossAdapter(numClass), nil
	default:
		return nil, fmt.Errorf("unknown objective: %s", objective)
	}
}

// MulticlassObjectiveFunction defines the interface for multiclass objective functions
type MulticlassObjectiveFunction interface {
	// CalculateGradientsAndHessians calculates gradients and hessians for all classes
	// yTrue: true class labels [numSamples]
	// yPred: predicted logits [numSamples * numClasses] (flattened)
	// numClasses: number of classes
	// Returns: gradients [numSamples * numClasses], hessians [numSamples * numClasses]
	CalculateGradientsAndHessians(yTrue []int, yPred []float64, numClasses int) ([]float64, []float64)

	// CalculateLoss calculates the total loss
	CalculateLoss(yTrue []int, yPred []float64, numClasses int) float64

	// Name returns the name of the objective
	Name() string
}

// MulticlassLogLossObjective implements multiclass cross-entropy loss with softmax
type MulticlassLogLossObjective struct {
	numClasses int
}

func NewMulticlassLogLoss(numClasses int) *MulticlassLogLossObjective {
	return &MulticlassLogLossObjective{
		numClasses: numClasses,
	}
}

// stableSoftmax computes softmax with numerical stability
func (m *MulticlassLogLossObjective) stableSoftmax(logits []float64) []float64 {
	// Find max for numerical stability
	maxLogit := logits[0]
	for _, logit := range logits[1:] {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	// Compute exp(x - max) and sum
	expSum := 0.0
	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit - maxLogit)
		expSum += probabilities[i]
	}

	// Normalize
	if expSum > 0 {
		for i := range probabilities {
			probabilities[i] /= expSum
		}
	}

	return probabilities
}

// CalculateGradientsAndHessians implements the multiclass logloss gradients and hessians
func (m *MulticlassLogLossObjective) CalculateGradientsAndHessians(yTrue []int, yPred []float64, numClasses int) ([]float64, []float64) {
	numSamples := len(yTrue)
	gradients := make([]float64, numSamples*numClasses)
	hessians := make([]float64, numSamples*numClasses)

	// Use parallelization for large datasets
	var wg sync.WaitGroup

	// Process samples in parallel
	batchSize := 100 // Process in batches to avoid too many goroutines
	numBatches := (numSamples + batchSize - 1) / batchSize

	for batch := 0; batch < numBatches; batch++ {
		wg.Add(1)
		go func(batchIdx int) {
			defer wg.Done()

			start := batchIdx * batchSize
			end := start + batchSize
			if end > numSamples {
				end = numSamples
			}

			for i := start; i < end; i++ {
				// Extract logits for this sample
				sampleLogits := make([]float64, numClasses)
				for k := 0; k < numClasses; k++ {
					sampleLogits[k] = yPred[i*numClasses+k]
				}

				// Compute softmax probabilities
				probabilities := m.stableSoftmax(sampleLogits)

				// Calculate gradients and hessians for each class
				trueClass := yTrue[i]
				for k := 0; k < numClasses; k++ {
					prob := probabilities[k]

					// Gradient: p_k - y_k (where y_k is 1 if k == true class, 0 otherwise)
					var gradient float64
					if k == trueClass {
						gradient = prob - 1.0
					} else {
						gradient = prob
					}

					// Hessian: p_k * (1 - p_k) (diagonal approximation)
					hessian := prob * (1.0 - prob)

					// Ensure numerical stability
					if hessian < 1e-16 {
						hessian = 1e-16
					}

					gradients[i*numClasses+k] = gradient
					hessians[i*numClasses+k] = hessian
				}
			}
		}(batch)
	}

	wg.Wait()
	return gradients, hessians
}

// CalculateLoss calculates the multiclass cross-entropy loss
func (m *MulticlassLogLossObjective) CalculateLoss(yTrue []int, yPred []float64, numClasses int) float64 {
	numSamples := len(yTrue)
	totalLoss := 0.0

	for i := 0; i < numSamples; i++ {
		// Extract logits for this sample
		sampleLogits := make([]float64, numClasses)
		for k := 0; k < numClasses; k++ {
			sampleLogits[k] = yPred[i*numClasses+k]
		}

		// Compute log softmax with numerical stability
		maxLogit := sampleLogits[0]
		for _, logit := range sampleLogits[1:] {
			if logit > maxLogit {
				maxLogit = logit
			}
		}

		logSumExp := 0.0
		for _, logit := range sampleLogits {
			logSumExp += math.Exp(logit - maxLogit)
		}
		logSumExp = math.Log(logSumExp) + maxLogit

		// Cross-entropy loss: -log(p_true_class)
		trueClass := yTrue[i]
		loss := -(sampleLogits[trueClass] - logSumExp)
		totalLoss += loss
	}

	return totalLoss / float64(numSamples)
}

// Name returns the name of the objective
func (m *MulticlassLogLossObjective) Name() string {
	return "multiclass_logloss"
}

// MulticlassLogLossAdapter adapts MulticlassLogLoss to ObjectiveFunction interface
// This is used for compatibility with existing single-prediction training
type MulticlassLogLossAdapter struct {
	multiclassImpl *MulticlassLogLossObjective
}

func NewMulticlassLogLossAdapter(numClasses int) *MulticlassLogLossAdapter {
	return &MulticlassLogLossAdapter{
		multiclassImpl: NewMulticlassLogLoss(numClasses),
	}
}

func (a *MulticlassLogLossAdapter) CalculateGradient(prediction, target float64) float64 {
	// For adapter, use simplified gradient calculation
	// This is mainly for compatibility; actual multiclass training should use the full interface
	return prediction - target
}

func (a *MulticlassLogLossAdapter) CalculateHessian(prediction, target float64) float64 {
	// For adapter, use simplified hessian calculation
	return 1.0
}

func (a *MulticlassLogLossAdapter) CalculateLoss(prediction, target float64) float64 {
	// For adapter, use squared loss as approximation
	diff := prediction - target
	return 0.5 * diff * diff
}

func (a *MulticlassLogLossAdapter) GetInitScore(targets []float64) float64 {
	// Return mean for initial score
	if len(targets) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	return sum / float64(len(targets))
}

func (a *MulticlassLogLossAdapter) Name() string {
	return "multiclass_logloss"
}

// GetMulticlassImpl returns the underlying multiclass implementation
// This can be used when full multiclass functionality is needed
func (a *MulticlassLogLossAdapter) GetMulticlassImpl() *MulticlassLogLossObjective {
	return a.multiclassImpl
}
