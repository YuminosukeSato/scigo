//go:build ignore

package main

import (
	"fmt"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Load model
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/compatibility/regression_model.txt")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Model info:\n")
	fmt.Printf("  NumTrees: %d\n", len(model.Trees))
	fmt.Printf("  NumIteration: %d\n", model.NumIteration)
	fmt.Printf("  BestIteration: %d\n", model.BestIteration)
	fmt.Printf("  InitScore: %f\n", model.InitScore)

	// First test sample
	features := []float64{
		1.59040357, -0.39398668, 0.04092475, -0.99844085, 1.99137042,
		0.43494104, 1.62325669, -0.5691482, -0.79711357, 0.39291351,
	}

	X := mat.NewDense(1, 10, features)
	predictor := lightgbm.NewPredictor(model)

	predictions, err := predictor.Predict(X)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nPrediction: %f\n", predictions.At(0, 0))
	fmt.Printf("Expected: 77.486099\n")
}
