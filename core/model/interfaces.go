// Package model provides additional interfaces and types for machine learning models.
// This file complements the existing interfaces in estimator.go and transformer.go
package model

import (
	"gonum.org/v1/gonum/mat"
)

// Scorer is the interface for models that can compute a score.
type Scorer interface {
	// Score returns the coefficient of determination R^2 of the prediction.
	Score(X mat.Matrix, y mat.Matrix) (float64, error)
}

// IncrementalLearner is the interface for models that support incremental learning.
type IncrementalLearner interface {
	// PartialFit performs one epoch of stochastic gradient descent on given samples.
	PartialFit(X mat.Matrix, y mat.Matrix, classes []int) error
}

// OnlineEstimator is the interface for models that support online learning.
// This interface complements the existing StreamingEstimator in streaming.go
type OnlineEstimator interface {
	Estimator
	IncrementalLearner
}

// Regressor combines interfaces for regression models.
type Regressor interface {
	Estimator
	Predictor
	Scorer
}

// Classifier combines interfaces for classification models.
type Classifier interface {
	Estimator
	Predictor
	Scorer
	
	// PredictProba returns probability estimates for each class.
	PredictProba(X mat.Matrix) (mat.Matrix, error)
	
	// Classes returns the unique classes seen during fitting.
	Classes() []int
}

// RegressorWithPartialFit combines interfaces for online regression models.
type RegressorWithPartialFit interface {
	Regressor
	IncrementalLearner
}

// ClassifierWithPartialFit combines interfaces for online classification models.
type ClassifierWithPartialFit interface {
	Classifier
	IncrementalLearner
}

// ParameterGetter is the interface for models that expose their parameters.
type ParameterGetter interface {
	// GetParams returns the model's hyperparameters.
	GetParams() map[string]interface{}
}

// ParameterSetter is the interface for models that allow parameter modification.
type ParameterSetter interface {
	// SetParams sets the model's hyperparameters.
	SetParams(params map[string]interface{}) error
}

// Persistable is the interface for models that can be saved and loaded.
type Persistable interface {
	// Save saves the model to a file.
	Save(path string) error
	
	// Load loads the model from a file.
	Load(path string) error
}