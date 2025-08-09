package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== LightGBM Basic Regression Example ===")

	// Generate synthetic regression data
	nSamples, nFeatures := 1000, 10
	X, y := generateRegressionData(nSamples, nFeatures, 42)

	// Split into train/test
	splitRatio := 0.8
	trainSize := int(float64(nSamples) * splitRatio)
	
	XTrain := X.Slice(0, trainSize, 0, nFeatures).(*mat.Dense)
	yTrain := y.Slice(0, trainSize, 0, 1).(*mat.Dense)
	XTest := X.Slice(trainSize, nSamples, 0, nFeatures).(*mat.Dense)
	yTest := y.Slice(trainSize, nSamples, 0, 1).(*mat.Dense)

	fmt.Printf("Training set size: %d samples\n", trainSize)
	fmt.Printf("Test set size: %d samples\n", nSamples-trainSize)

	// Create LightGBM regressor
	reg := lightgbm.NewLGBMRegressor()

	// Set parameters using Python names (full compatibility!)
	params := map[string]interface{}{
		"n_estimators":      100,
		"learning_rate":     0.1,
		"num_leaves":        31,
		"max_depth":         -1,
		"min_child_samples": 20,
		"reg_alpha":         0.1,
		"reg_lambda":        0.1,
		"random_state":      42,
		"n_jobs":           -1, // Use all cores
		"verbosity":        1,  // Show training progress
	}

	err := reg.SetParams(params)
	if err != nil {
		log.Fatal("Error setting parameters:", err)
	}

	// Train the model
	fmt.Println("\n--- Training Model ---")
	start := time.Now()
	
	err = reg.Fit(XTrain, yTrain)
	if err != nil {
		log.Fatal("Error training model:", err)
	}
	
	trainTime := time.Since(start)
	fmt.Printf("Training completed in %.3fs\n", trainTime.Seconds())

	// Make predictions on test set
	predictions, err := reg.Predict(XTest)
	if err != nil {
		log.Fatal("Error making predictions:", err)
	}

	// Evaluate model performance
	fmt.Println("\n--- Model Evaluation ---")
	
	// Calculate metrics
	r2Score := reg.Score(XTest, yTest)
	rmse, _ := lightgbm.RMSE(yTest, predictions)
	mae, _ := lightgbm.MAE(yTest, predictions)
	mape, _ := lightgbm.MAPE(yTest, predictions)

	fmt.Printf("R² Score: %.4f\n", r2Score)
	fmt.Printf("RMSE: %.4f\n", rmse)
	fmt.Printf("MAE: %.4f\n", mae)
	fmt.Printf("MAPE: %.2f%%\n", mape)

	// Show sample predictions
	fmt.Println("\n--- Sample Predictions ---")
	fmt.Printf("%-10s %-10s %-10s\n", "Actual", "Predicted", "Difference")
	fmt.Println("-----------------------------------")
	
	for i := 0; i < 5; i++ {
		actual := yTest.At(i, 0)
		predicted := predictions.At(i, 0)
		diff := actual - predicted
		fmt.Printf("%-10.4f %-10.4f %-10.4f\n", actual, predicted, diff)
	}

	// Get feature importance (if available)
	if importances := reg.GetFeatureImportance(); len(importances) > 0 {
		fmt.Println("\n--- Feature Importance ---")
		for i, importance := range importances {
			fmt.Printf("Feature %d: %.4f\n", i, importance)
		}
	}

	// Save the trained model
	fmt.Println("\n--- Saving Model ---")
	err = reg.SaveToFile("basic_regression_model.txt")
	if err != nil {
		log.Printf("Warning: Could not save model: %v", err)
	} else {
		fmt.Println("Model saved to 'basic_regression_model.txt'")
	}

	// Save as JSON for Python compatibility
	err = reg.SaveToJSONFile("basic_regression_model.json")
	if err != nil {
		log.Printf("Warning: Could not save JSON model: %v", err)
	} else {
		fmt.Println("Model saved to 'basic_regression_model.json'")
	}

	fmt.Println("\n✅ Basic regression example completed successfully!")
}

// generateRegressionData creates synthetic regression data
func generateRegressionData(nSamples, nFeatures, seed int) (*mat.Dense, *mat.Dense) {
	rand.Seed(int64(seed))

	// Generate random features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}
	}

	// Generate true coefficients
	coeff := make([]float64, nFeatures)
	for i := range coeff {
		coeff[i] = rand.Float64()*2 - 1 // Random coefficient in [-1, 1]
	}

	// Generate targets with some noise
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		target := 0.0
		for j := 0; j < nFeatures; j++ {
			target += X.At(i, j) * coeff[j]
		}
		// Add some noise
		target += rand.NormFloat64() * 0.1
		y.Set(i, 0, target)
	}

	return X, y
}