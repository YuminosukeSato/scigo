package main

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
)

func main() {
	// Load model
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/compatibility/regression_model.txt")
	if err != nil {
		panic(err)
	}

	fmt.Fprintf(os.Stdout, "Model info:\n")
	fmt.Fprintf(os.Stdout, "  NumTrees: %d\n", len(model.Trees))
	fmt.Fprintf(os.Stdout, "  NumIteration: %d\n", model.NumIteration)
	fmt.Fprintf(os.Stdout, "  BestIteration: %d\n", model.BestIteration)
	fmt.Fprintf(os.Stdout, "  InitScore: %f\n", model.InitScore)

	// First test sample
	features := []float64{
		1.59040357, -0.39398668, 0.04092475, -0.99844085, 1.99137042,
		0.43494104, 1.62325669, -0.5691482, -0.79711357, 0.39291351,
	}

	const numFeatures = 10
	x := mat.NewDense(1, numFeatures, features)
	predictor := lightgbm.NewPredictor(model)

	predictions, err := predictor.Predict(x)
	if err != nil {
		panic(err)
	}

	fmt.Fprintf(os.Stdout, "\nPrediction: %f\n", predictions.At(0, 0))
	fmt.Fprintf(os.Stdout, "Expected: 77.486099\n")
}
