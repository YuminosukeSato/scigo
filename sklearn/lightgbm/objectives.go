package lightgbm

import (
	"fmt"
	"math"
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
	default:
		return nil, fmt.Errorf("unknown objective: %s", objective)
	}
}
