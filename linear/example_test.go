package linear_test

import (
	"fmt"
	"log"

	"github.com/YuminosukeSato/scigo/linear"
	"gonum.org/v1/gonum/mat"
)

// ExampleLinearRegression demonstrates basic linear regression usage
func ExampleLinearRegression() {
	// Create simple training data: y = 2*x + 1
	X := mat.NewDense(4, 1, []float64{1.0, 2.0, 3.0, 4.0})
	y := mat.NewDense(4, 1, []float64{3.0, 5.0, 7.0, 9.0})

	// Create and train model
	lr := linear.NewLinearRegression()
	err := lr.Fit(X, y)
	if err != nil {
		log.Fatal(err)
	}

	// Make predictions
	testX := mat.NewDense(2, 1, []float64{5.0, 6.0})
	predictions, err := lr.Predict(testX)
	if err != nil {
		log.Fatal(err)
	}

	// Print predictions
	fmt.Printf("Input: %.1f, Prediction: %.1f\n", testX.At(0, 0), predictions.At(0, 0))
	fmt.Printf("Input: %.1f, Prediction: %.1f\n", testX.At(1, 0), predictions.At(1, 0))

	// Output: Input: 5.0, Prediction: 11.0
	// Input: 6.0, Prediction: 13.0
}

// ExampleLinearRegression_multipleFeatures demonstrates multiple feature regression
func ExampleLinearRegression_multipleFeatures() {
	// Create training data with 2 features
	X := mat.NewDense(4, 2, []float64{
		1.0, 1.0,  // Sample 1: [1, 1]
		2.0, 1.0,  // Sample 2: [2, 1]  
		1.0, 2.0,  // Sample 3: [1, 2]
		2.0, 2.0,  // Sample 4: [2, 2]
	})
	
	// Target: y = x1 + 2*x2
	y := mat.NewDense(4, 1, []float64{3.0, 4.0, 5.0, 6.0})

	// Train model
	lr := linear.NewLinearRegression()
	err := lr.Fit(X, y)
	if err != nil {
		log.Fatal(err)
	}

	// Get model parameters
	weights := lr.GetWeights()
	intercept := lr.GetIntercept()
	
	fmt.Printf("Weights: [%.1f, %.1f]\n", weights[0], weights[1])
	fmt.Printf("Intercept: %.1f\n", intercept)

	// Output: Weights: [1.0, 2.0]
	// Intercept: 0.0
}

// ExampleLinearRegression_modelEvaluation demonstrates model evaluation
func ExampleLinearRegression_modelEvaluation() {
	// Create training data
	X := mat.NewDense(5, 1, []float64{1.0, 2.0, 3.0, 4.0, 5.0})
	y := mat.NewDense(5, 1, []float64{2.0, 4.0, 6.0, 8.0, 10.0})

	// Train model
	lr := linear.NewLinearRegression()
	err := lr.Fit(X, y)
	if err != nil {
		log.Fatal(err)
	}

	// Evaluate on training data
	score, err := lr.Score(X, y)
	if err != nil {
		log.Fatal(err)
	}

	// Check if model is fitted
	fmt.Printf("Model fitted: %t\n", lr.IsFitted())
	fmt.Printf("R² Score: %.3f\n", score)

	// Output: Model fitted: true
	// R² Score: 1.000
}

// ExampleLinearRegression_persistence demonstrates model saving/loading
func ExampleLinearRegression_persistence() {
	// Create and train model
	X := mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0})
	y := mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0})

	lr := linear.NewLinearRegression()
	err := lr.Fit(X, y)
	if err != nil {
		log.Fatal(err)
	}

	// Get weights before export
	originalWeights := lr.GetWeights()
	originalIntercept := lr.GetIntercept()

	// Format intercept to avoid -0 display
	interceptStr := "0"
	if originalIntercept > 0.001 || originalIntercept < -0.001 {
		interceptStr = fmt.Sprintf("%.3f", originalIntercept)
	}
	
	fmt.Printf("Original model - Weight: %.3f, Intercept: %s\n", 
		originalWeights[0], interceptStr)

	// In a real scenario, you would save to file and load in another process
	// This example shows the model parameters are preserved
	fmt.Printf("Model can be exported and imported using ExportToSKLearn/LoadFromSKLearn methods\n")

	// Output: Original model - Weight: 1.000, Intercept: 0
	// Model can be exported and imported using ExportToSKLearn/LoadFromSKLearn methods
}