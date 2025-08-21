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

// NewL2Objective creates a new L2 objective function
func NewL2Objective() *L2Objective {
	return &L2Objective{}
}

// CalculateGradient computes the gradient for L2 loss
func (o *L2Objective) CalculateGradient(prediction, target float64) float64 {
	return prediction - target
}

// CalculateHessian computes the hessian for L2 loss
func (o *L2Objective) CalculateHessian(prediction, target float64) float64 {
	return 1.0
}

// CalculateLoss computes the loss value for L2 objective
func (o *L2Objective) CalculateLoss(prediction, target float64) float64 {
	diff := prediction - target
	return 0.5 * diff * diff
}

// GetInitScore computes the initial prediction score for L2 objective
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

// Name returns the name of the L2 objective
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

// BinaryLogLossObjective implements Binary Logistic Loss (Log Loss) for binary classification
type BinaryLogLossObjective struct {
	maxOutputExp float64 // Maximum value for exp calculations to prevent overflow
}

// NewBinaryLogLossObjective creates a new Binary Log Loss objective
func NewBinaryLogLossObjective() *BinaryLogLossObjective {
	return &BinaryLogLossObjective{
		maxOutputExp: 700.0, // exp(700) is close to max float64
	}
}

// CalculateGradient calculates gradient for binary log loss
// Gradient: sigmoid(prediction) - target
func (o *BinaryLogLossObjective) CalculateGradient(prediction, target float64) float64 {
	// Clamp prediction to prevent overflow
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	// sigmoid(x) = 1 / (1 + exp(-x))
	sigmoid := 1.0 / (1.0 + math.Exp(-clampedPred))
	return sigmoid - target
}

// CalculateHessian calculates hessian for binary log loss
// Hessian: sigmoid(prediction) * (1 - sigmoid(prediction))
func (o *BinaryLogLossObjective) CalculateHessian(prediction, target float64) float64 {
	// Clamp prediction to prevent overflow
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	// sigmoid(x) = 1 / (1 + exp(-x))
	sigmoid := 1.0 / (1.0 + math.Exp(-clampedPred))
	hessian := sigmoid * (1.0 - sigmoid)
	// Ensure hessian is positive and not too small for numerical stability
	return math.Max(1e-16, hessian)
}

// CalculateLoss calculates binary log loss
// Loss: -target * log(sigmoid(pred)) - (1-target) * log(1-sigmoid(pred))
func (o *BinaryLogLossObjective) CalculateLoss(prediction, target float64) float64 {
	// Clamp prediction to prevent overflow
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))

	// Use log-sum-exp trick for numerical stability
	if clampedPred >= 0 {
		exp_neg_pred := math.Exp(-clampedPred)
		return target*clampedPred + math.Log(1.0+exp_neg_pred)
	} else {
		exp_pred := math.Exp(clampedPred)
		return -target*clampedPred + math.Log(1.0+exp_pred)
	}
}

// GetInitScore returns initial score for binary classification
// For binary classification, initial score is logit of mean target
func (o *BinaryLogLossObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}

	// Calculate mean of targets (proportion of positive class)
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	mean := sum / float64(len(targets))

	// Clamp mean to avoid log(0) or log(inf)
	epsilon := 1e-15
	mean = math.Max(epsilon, math.Min(1.0-epsilon, mean))

	// Return logit: log(p / (1-p))
	return math.Log(mean / (1.0 - mean))
}

// Name returns the name of the binary log loss objective
func (o *BinaryLogLossObjective) Name() string {
	return "binary"
}

// Helper functions

// GammaObjective implements Gamma regression objective
// The target must be positive. Uses log-link: μ = exp(F(x))
type GammaObjective struct {
	maxOutputExp float64 // Maximum value for exp calculations to prevent overflow
}

// NewGammaObjective creates a new Gamma objective
func NewGammaObjective() *GammaObjective {
	return &GammaObjective{
		maxOutputExp: 700.0, // exp(700) is close to max float64
	}
}

// CalculateGradient calculates gradient for Gamma regression
// Gradient: 2 * (1 - y * exp(-prediction))
func (o *GammaObjective) CalculateGradient(prediction, target float64) float64 {
	// Clamp prediction to prevent overflow/underflow
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	return 2.0 * (1.0 - target*math.Exp(-clampedPred))
}

// CalculateHessian calculates hessian for Gamma regression
// Hessian: 2 * y * exp(-prediction)
func (o *GammaObjective) CalculateHessian(prediction, target float64) float64 {
	// Clamp prediction to prevent overflow/underflow
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	hessian := 2.0 * target * math.Exp(-clampedPred)
	// Ensure hessian is positive and not too small
	return math.Max(1e-16, hessian)
}

// CalculateLoss calculates Gamma deviance loss
func (o *GammaObjective) CalculateLoss(prediction, target float64) float64 {
	// Gamma deviance: 2 * (log(μ/y) + y/μ - 1)
	// where μ = exp(prediction)
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	mu := math.Exp(clampedPred)
	// Prevent division by zero
	if target <= 0 {
		return 0.0
	}
	return 2.0 * (math.Log(mu/target) + target/mu - 1.0)
}

// GetInitScore returns initial score for Gamma regression
func (o *GammaObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use log of mean as initial score for Gamma regression
	sum := 0.0
	count := 0
	for _, t := range targets {
		if t > 0 { // Only consider positive values
			sum += t
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	mean := sum / float64(count)
	return math.Log(mean)
}

// Name returns the name of the objective
func (o *GammaObjective) Name() string {
	return "gamma"
}

// TweedieObjective implements Tweedie regression objective
// Supports compound Poisson-Gamma distribution with variance power p
type TweedieObjective struct {
	variancePower float64 // Power parameter p (typically between 1 and 2)
	maxOutputExp  float64 // Maximum value for exp calculations
}

// NewTweedieObjective creates a new Tweedie objective
func NewTweedieObjective(variancePower float64) *TweedieObjective {
	// Default to 1.5 if not specified (common for insurance claims)
	if variancePower <= 1.0 || variancePower >= 2.0 {
		variancePower = 1.5
	}
	return &TweedieObjective{
		variancePower: variancePower,
		maxOutputExp:  700.0,
	}
}

// CalculateGradient calculates gradient for Tweedie regression
// Gradient: 2 * (exp((2-p)*pred) - y*exp((1-p)*pred))
func (o *TweedieObjective) CalculateGradient(prediction, target float64) float64 {
	p := o.variancePower
	// Clamp predictions to prevent overflow
	pred1 := math.Max(-o.maxOutputExp, math.Min((2-p)*prediction, o.maxOutputExp))
	pred2 := math.Max(-o.maxOutputExp, math.Min((1-p)*prediction, o.maxOutputExp))

	return 2.0 * (math.Exp(pred1) - target*math.Exp(pred2))
}

// CalculateHessian calculates hessian for Tweedie regression
// Hessian: 2 * ((2-p)*exp((2-p)*pred) - (1-p)*y*exp((1-p)*pred))
func (o *TweedieObjective) CalculateHessian(prediction, target float64) float64 {
	p := o.variancePower
	// Clamp predictions to prevent overflow
	pred1 := math.Max(-o.maxOutputExp, math.Min((2-p)*prediction, o.maxOutputExp))
	pred2 := math.Max(-o.maxOutputExp, math.Min((1-p)*prediction, o.maxOutputExp))

	hessian := 2.0 * ((2-p)*math.Exp(pred1) - (1-p)*target*math.Exp(pred2))
	// Ensure hessian is positive
	return math.Max(1e-16, math.Abs(hessian))
}

// CalculateLoss calculates Tweedie deviance loss
func (o *TweedieObjective) CalculateLoss(prediction, target float64) float64 {
	p := o.variancePower
	// μ = exp(prediction)
	clampedPred := math.Max(-o.maxOutputExp, math.Min(prediction, o.maxOutputExp))
	mu := math.Exp(clampedPred)

	// Tweedie deviance
	if target == 0 {
		return 2.0 * math.Pow(mu, 2-p) / (2 - p)
	}

	term1 := math.Pow(target, 2-p) / ((1 - p) * (2 - p))
	term2 := target * math.Pow(mu, 1-p) / (1 - p)
	term3 := math.Pow(mu, 2-p) / (2 - p)

	return 2.0 * (term1 - term2 + term3)
}

// GetInitScore returns initial score for Tweedie regression
func (o *TweedieObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use log of mean of positive values as initial score
	sum := 0.0
	count := 0
	for _, t := range targets {
		if t > 0 { // Include only positive values
			sum += t
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	mean := sum / float64(count)
	return math.Log(mean)
}

// Name returns the name of the objective
func (o *TweedieObjective) Name() string {
	return "tweedie"
}

// LambdaRankObjective implements LambdaRank objective for ranking
type LambdaRankObjective struct {
	groups [][]int // Group information for ranking
}

// NewLambdaRankObjective creates a new LambdaRank objective
func NewLambdaRankObjective() *LambdaRankObjective {
	return &LambdaRankObjective{}
}

// SetGroups sets the group information for ranking
func (o *LambdaRankObjective) SetGroups(groups [][]int) {
	o.groups = groups
}

// CalculateGradient calculates gradient for LambdaRank
func (o *LambdaRankObjective) CalculateGradient(prediction, target float64) float64 {
	// Simplified LambdaRank gradient
	// In practice, this requires pairwise computation within groups
	// This is a placeholder implementation
	return prediction - target
}

// CalculateHessian calculates hessian for LambdaRank
func (o *LambdaRankObjective) CalculateHessian(prediction, target float64) float64 {
	// Simplified hessian for LambdaRank
	// In practice, this requires second-order derivatives of pairwise losses
	return 1.0
}

// CalculateLoss calculates LambdaRank loss (NDCG-based)
func (o *LambdaRankObjective) CalculateLoss(prediction, target float64) float64 {
	// Simplified loss - in practice this should compute NDCG
	diff := prediction - target
	return 0.5 * diff * diff
}

// GetInitScore returns initial score for LambdaRank
func (o *LambdaRankObjective) GetInitScore(targets []float64) float64 {
	if len(targets) == 0 {
		return 0.0
	}
	// Use mean of targets as initial score
	sum := 0.0
	for _, t := range targets {
		sum += t
	}
	return sum / float64(len(targets))
}

func (o *LambdaRankObjective) Name() string {
	return "lambdarank"
}

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
	case "gamma":
		return NewGammaObjective(), nil
	case "tweedie":
		variancePower := 1.5 // Default value
		if params != nil && params.TweedieVariancePower > 1.0 && params.TweedieVariancePower < 2.0 {
			variancePower = params.TweedieVariancePower
		}
		return NewTweedieObjective(variancePower), nil
	case "lambdarank", "rank_xendcg":
		return NewLambdaRankObjective(), nil
	case "binary", "binary_logloss", "logistic":
		// Use proper binary log loss objective for binary classification
		return NewBinaryLogLossObjective(), nil
	case "multiclass", "softmax", "multiclassova":
		// For multiclass, use MulticlassLogLoss objective
		numClass := 3 // Default
		if params != nil && params.NumClass > 0 {
			numClass = params.NumClass
		}
		return NewMulticlassLogLossAdapter(numClass), nil
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
	// For multiclass, use softmax gradient calculation
	// In LightGBM's one-vs-rest approach, gradient = sigmoid(prediction) - target_indicator
	// where target_indicator is 1 if this is the true class, 0 otherwise

	// Clamp prediction to prevent overflow
	maxOutputExp := 700.0
	clampedPred := math.Max(-maxOutputExp, math.Min(prediction, maxOutputExp))

	// Calculate sigmoid(prediction)
	sigmoid := 1.0 / (1.0 + math.Exp(-clampedPred))

	// For multiclass one-vs-rest: gradient = sigmoid - target_binary
	return sigmoid - target
}

func (a *MulticlassLogLossAdapter) CalculateHessian(prediction, target float64) float64 {
	// For multiclass, use softmax hessian calculation
	// In LightGBM's one-vs-rest approach, hessian = sigmoid(prediction) * (1 - sigmoid(prediction))

	// Clamp prediction to prevent overflow
	maxOutputExp := 700.0
	clampedPred := math.Max(-maxOutputExp, math.Min(prediction, maxOutputExp))

	// Calculate sigmoid(prediction)
	sigmoid := 1.0 / (1.0 + math.Exp(-clampedPred))

	// Hessian for logistic: sigmoid * (1 - sigmoid)
	hessian := sigmoid * (1.0 - sigmoid)

	// Ensure numerical stability
	return math.Max(1e-16, hessian)
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
