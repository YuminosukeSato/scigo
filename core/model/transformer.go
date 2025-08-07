package model

import "gonum.org/v1/gonum/mat"

// Transformer is an interface for data transformation
type Transformer interface {
	// Fit learns parameters necessary for transformation
	Fit(X mat.Matrix) error

	// Transform transforms data
	Transform(X mat.Matrix) (mat.Matrix, error)

	// FitTransform executes Fit and Transform simultaneously
	FitTransform(X mat.Matrix) (mat.Matrix, error)
}
