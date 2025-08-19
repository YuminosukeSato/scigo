package lightgbm

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestRealData tests LightGBM with real data scenarios
func TestRealData(t *testing.T) {
	// Load model
	model, err := LoadFromFile("testdata/compatibility/regression_model.txt")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("Model loaded:\n")
	fmt.Printf("  NumTrees: %d\n", len(model.Trees))
	fmt.Printf("  NumIteration: %d\n", model.NumIteration)
	fmt.Printf("  BestIteration: %d\n", model.BestIteration)
	fmt.Printf("  InitScore: %f\n", model.InitScore)

	// First test sample from regression_X_test.csv
	features := []float64{
		1.59040357, -0.39398668, 0.04092475, -0.99844085, 1.99137042,
		0.43494104, 1.62325669, -0.5691482, -0.79711357, 0.39291351,
	}

	predictor := NewPredictor(model)

	// Test single sample
	X := mat.NewDense(1, 10, features)
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	fmt.Printf("\nFirst sample prediction: %f\n", predictions.At(0, 0))
	fmt.Printf("Expected (Python): 77.486099\n")

	// Test first tree only
	tree1Pred := predictor.predictTree(&model.Trees[0], features)
	fmt.Printf("\nFirst tree only: %f\n", tree1Pred)

	// Test accumulation manually
	sum := 0.0
	for i := 0; i < 5 && i < len(model.Trees); i++ {
		treePred := predictor.predictTree(&model.Trees[i], features)
		sum += treePred
		fmt.Printf("After tree %d: pred=%f, cumulative=%f\n", i, treePred, sum)
	}
}
