package main

import (
	"fmt"
	"log"

	"github.com/YuminosukeSato/GoML/linear"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== scikit-learn Compatible LinearRegression Demo ===")
	fmt.Println()

	// Example 1: Basic usage with default parameters
	basicExample()
	fmt.Println()

	// Example 2: Without intercept
	noInterceptExample()
	fmt.Println()

	// Example 3: Multiple targets
	multipleTargetsExample()
	fmt.Println()

	// Example 4: With positive constraint
	positiveConstraintExample()
}

func basicExample() {
	fmt.Println("1. Basic Linear Regression (y = 2x + 3)")
	fmt.Println("-----------------------------------------")

	// Create training data: y = 2x + 3
	X := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	y := mat.NewDense(5, 1, []float64{5, 7, 9, 11, 13})

	// Create model with default parameters (similar to scikit-learn)
	model := linear.NewSKLinearRegression()

	// Fit the model
	if err := model.Fit(X, y); err != nil {
		log.Fatal(err)
	}

	// Display learned parameters
	fmt.Printf("Coefficient (slope): %.4f\n", model.Coef_.At(0, 0))
	fmt.Printf("Intercept: %.4f\n", model.Intercept_.At(0, 0))
	fmt.Printf("Rank of X: %d\n", model.Rank_)

	// Make predictions
	XTest := mat.NewDense(3, 1, []float64{6, 7, 8})
	predictions, err := model.Predict(XTest)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Predictions for x = [6, 7, 8]:")
	for i := 0; i < 3; i++ {
		fmt.Printf("  x = %.0f -> y = %.2f\n", XTest.At(i, 0), predictions.At(i, 0))
	}

	// Calculate R² score
	score, err := model.Score(X, y)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("R² Score: %.6f\n", score)
}

func noInterceptExample() {
	fmt.Println("2. Linear Regression without Intercept (y = 2x)")
	fmt.Println("------------------------------------------------")

	// Create training data: y = 2x (no intercept)
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := mat.NewDense(4, 1, []float64{2, 4, 6, 8})

	// Create model without intercept
	model := linear.NewSKLinearRegression(
		linear.WithFitIntercept(false),
	)

	// Fit the model
	if err := model.Fit(X, y); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Coefficient: %.4f\n", model.Coef_.At(0, 0))
	fmt.Printf("Intercept: %.4f (should be 0)\n", model.Intercept_.At(0, 0))

	// Make predictions
	XTest := mat.NewDense(2, 1, []float64{5, 10})
	predictions, err := model.Predict(XTest)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Predictions:")
	for i := 0; i < 2; i++ {
		fmt.Printf("  x = %.0f -> y = %.2f\n", XTest.At(i, 0), predictions.At(i, 0))
	}
}

func multipleTargetsExample() {
	fmt.Println("3. Multiple Target Regression")
	fmt.Println("-----------------------------")

	// Create training data with 2 targets
	// y1 = 2*x + 1
	// y2 = -x + 5
	X := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	y := mat.NewDense(5, 2, []float64{
		3, 4,   // x=1: y1=3, y2=4
		5, 3,   // x=2: y1=5, y2=3
		7, 2,   // x=3: y1=7, y2=2
		9, 1,   // x=4: y1=9, y2=1
		11, 0,  // x=5: y1=11, y2=0
	})

	// Create model
	model := linear.NewSKLinearRegression()

	// Fit the model
	if err := model.Fit(X, y); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Coefficients for each target:")
	fmt.Printf("  Target 1: %.4f (expected: 2.0)\n", model.Coef_.At(0, 0))
	fmt.Printf("  Target 2: %.4f (expected: -1.0)\n", model.Coef_.At(0, 1))

	fmt.Println("Intercepts for each target:")
	fmt.Printf("  Target 1: %.4f (expected: 1.0)\n", model.Intercept_.At(0, 0))
	fmt.Printf("  Target 2: %.4f (expected: 5.0)\n", model.Intercept_.At(0, 1))

	// Make predictions
	XTest := mat.NewDense(2, 1, []float64{6, 7})
	predictions, err := model.Predict(XTest)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Predictions:")
	for i := 0; i < 2; i++ {
		fmt.Printf("  x = %.0f -> y1 = %.2f, y2 = %.2f\n",
			XTest.At(i, 0), predictions.At(i, 0), predictions.At(i, 1))
	}
}

func positiveConstraintExample() {
	fmt.Println("4. Non-Negative Least Squares (positive=True)")
	fmt.Println("----------------------------------------------")

	// Create data where optimal solution would have negative coefficients
	// but we constrain them to be positive
	X := mat.NewDense(4, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 3,
	})
	y := mat.NewDense(4, 1, []float64{2, 3, 5, 7})

	// Create model with positive constraint
	model := linear.NewSKLinearRegression(
		linear.WithPositive(true),
		linear.WithFitIntercept(false), // Simpler for this example
	)

	// Fit the model
	if err := model.Fit(X, y); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Coefficients (all should be >= 0):")
	rows, cols := model.Coef_.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("  Coef[%d,%d] = %.4f\n", i, j, model.Coef_.At(i, j))
		}
	}

	// Display model parameters
	fmt.Println("\nModel Parameters:")
	params := model.GetParams()
	for key, value := range params {
		fmt.Printf("  %s: %v\n", key, value)
	}

	// Test SetParams
	fmt.Println("\nTesting SetParams (changing n_jobs to 4):")
	newParams := map[string]interface{}{
		"n_jobs": 4,
	}
	if err := model.SetParams(newParams); err != nil {
		log.Fatal(err)
	}
	
	updatedParams := model.GetParams()
	fmt.Printf("  n_jobs after update: %v\n", updatedParams["n_jobs"])
}