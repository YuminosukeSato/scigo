package metrics_test

import (
	"fmt"
	"log/slog"

	"github.com/YuminosukeSato/scigo/metrics"
	"gonum.org/v1/gonum/mat"
)

// ExampleMSE demonstrates Mean Squared Error calculation
func ExampleMSE() {
	// Create true and predicted values
	yTrue := mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0})
	yPred := mat.NewVecDense(4, []float64{1.1, 1.9, 3.2, 3.8})

	// Calculate MSE
	mse, err := metrics.MSE(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("MSE: %.3f\n", mse)

	// Output: MSE: 0.025
}

// ExampleRMSE demonstrates Root Mean Squared Error calculation
func ExampleRMSE() {
	// Create sample data with some prediction errors
	yTrue := mat.NewVecDense(3, []float64{10.0, 20.0, 30.0})
	yPred := mat.NewVecDense(3, []float64{12.0, 18.0, 32.0})

	// Calculate RMSE
	rmse, err := metrics.RMSE(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("RMSE: %.2f\n", rmse)

	// Output: RMSE: 2.00
}

// ExampleMAE demonstrates Mean Absolute Error calculation
func ExampleMAE() {
	// Create true and predicted values
	yTrue := mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0})
	yPred := mat.NewVecDense(4, []float64{0.8, 2.2, 2.9, 4.3})

	// Calculate MAE
	mae, err := metrics.MAE(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("MAE: %.2f\n", mae)

	// Output: MAE: 0.20
}

// ExampleR2Score demonstrates R-squared (coefficient of determination) calculation
func ExampleR2Score() {
	// Create perfect predictions (R² should be 1.0)
	yTrue := mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0})
	yPred := mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0})

	// Calculate R² score
	r2, err := metrics.R2Score(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("R² Score: %.1f\n", r2)

	// Output: R² Score: 1.0
}

// ExampleR2Score_imperfectPredictions demonstrates R² with prediction errors
func ExampleR2Score_imperfectPredictions() {
	// Create data with some prediction errors
	yTrue := mat.NewVecDense(4, []float64{1.0, 3.0, 2.0, 4.0})
	yPred := mat.NewVecDense(4, []float64{1.2, 2.8, 2.1, 3.9})

	// Calculate R² score
	r2, err := metrics.R2Score(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("R² Score: %.3f\n", r2)

	// Output: R² Score: 0.980
}

// ExampleMAPE demonstrates Mean Absolute Percentage Error calculation
func ExampleMAPE() {
	// Create true and predicted values (avoiding zeros)
	yTrue := mat.NewVecDense(4, []float64{10.0, 20.0, 30.0, 40.0})
	yPred := mat.NewVecDense(4, []float64{9.0, 22.0, 28.0, 42.0})

	// Calculate MAPE
	mape, err := metrics.MAPE(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("MAPE: %.1f%%\n", mape)

	// Output: MAPE: 7.9%
}

// ExampleExplainedVarianceScore demonstrates explained variance score calculation
func ExampleExplainedVarianceScore() {
	// Create sample data
	yTrue := mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0})
	yPred := mat.NewVecDense(4, []float64{1.1, 1.9, 3.1, 3.9})

	// Calculate explained variance score
	evs, err := metrics.ExplainedVarianceScore(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("Explained Variance Score: %.3f\n", evs)

	// Output: Explained Variance Score: 0.992
}

// ExampleMSEMatrix demonstrates MSE calculation with matrix inputs
func ExampleMSEMatrix() {
	// Create matrix data (column vectors)
	yTrue := mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0})
	yPred := mat.NewDense(3, 1, []float64{1.1, 2.1, 2.9})

	// Calculate MSE using matrix inputs
	mse, err := metrics.MSEMatrix(yTrue, yPred)
	if err != nil {
		slog.Error("Test failed", "error", err)
		return
	}

	fmt.Printf("MSE (matrix input): %.3f\n", mse)

	// Output: MSE (matrix input): 0.010
}
