package model

import "gonum.org/v1/gonum/mat"

// IncrementalEstimator is an interface for models capable of online learning (incremental learning)
// Compatible with scikit-learn's partial_fit API
type IncrementalEstimator interface {
	Estimator

	// PartialFit trains the model incrementally with mini-batches
	// classes specifies all class labels for classification problems (required only on first call)
	// Pass nil for regression problems
	PartialFit(X, y mat.Matrix, classes []int) error

	// NIterations returns the number of training iterations executed
	NIterations() int

	// IsWarmStart returns whether warm start is enabled
	// If true, continues learning from existing parameters when Fit is called
	IsWarmStart() bool

	// SetWarmStart enables/disables warm start
	SetWarmStart(warmStart bool)
}

// OnlineMetrics is an interface for tracking metrics during online learning
type OnlineMetrics interface {
	// GetLoss returns the current loss value
	GetLoss() float64

	// GetLossHistory returns the history of loss values
	GetLossHistory() []float64

	// GetConverged returns whether the model has converged
	GetConverged() bool
}

// AdaptiveLearning is an interface for models that can dynamically adjust learning rates
type AdaptiveLearning interface {
	// GetLearningRate returns the current learning rate
	GetLearningRate() float64

	// SetLearningRate sets the learning rate
	SetLearningRate(lr float64)

	// GetLearningRateSchedule returns the learning rate schedule
	// e.g., "constant", "optimal", "invscaling", "adaptive"
	GetLearningRateSchedule() string

	// SetLearningRateSchedule sets the learning rate schedule
	SetLearningRateSchedule(schedule string)
}
