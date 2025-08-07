package model

import "gonum.org/v1/gonum/mat"

// Fitter is an interface for trainable models
type Fitter interface {
	// Fit trains the model with training data
	Fit(X, y mat.Matrix) error
}

// Predictor is an interface for predictive models
type Predictor interface {
	// Predict performs predictions on input data
	Predict(X mat.Matrix) (mat.Matrix, error)
}

// Estimator is an interface for models that can both learn and predict
type Estimator interface {
	Fitter
	Predictor
}

// LinearModel is an interface for linear models
type LinearModel interface {
	// Weights returns the learned weights (coefficients)
	Weights() []float64
	// Intercept returns the learned intercept
	Intercept() float64
	// Score calculates the model's coefficient of determination (R²)
	Score(X, y mat.Matrix) (float64, error)
}
