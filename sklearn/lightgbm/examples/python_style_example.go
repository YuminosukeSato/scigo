package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm/api"
	"gonum.org/v1/gonum/mat"
)

// This example demonstrates how to use the Python-style API for LightGBM in Go
// Compare with Python:
//
// import lightgbm as lgb
// import numpy as np
//
// # Create dataset
// X_train = np.random.randn(1000, 10)
// y_train = np.random.randint(0, 2, 1000)
// X_valid = np.random.randn(200, 10)
// y_valid = np.random.randint(0, 2, 200)
//
// train_data = lgb.Dataset(X_train, label=y_train)
// valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
//
// # Set parameters
// params = {
//     'objective': 'binary',
//     'metric': 'binary_logloss',
//     'num_leaves': 31,
//     'learning_rate': 0.05,
//     'feature_fraction': 0.9,
//     'bagging_fraction': 0.8,
//     'bagging_freq': 5,
//     'verbose': 0
// }
//
// # Train model
// bst = lgb.train(params, train_data, num_boost_round=100,
//                 valid_sets=[valid_data], 
//                 callbacks=[lgb.early_stopping(10)])

func main() {
	fmt.Println("=== LightGBM Go Example (Python-style API) ===")
	fmt.Println()

	// Set random seed for reproducibility
	rand.Seed(42)

	// Create training data
	fmt.Println("Creating dataset...")
	XTrain := generateRandomMatrix(1000, 10)
	yTrain := generateBinaryLabels(1000)
	
	// Create validation data
	XValid := generateRandomMatrix(200, 10)
	yValid := generateBinaryLabels(200)

	// Create LightGBM datasets (similar to lgb.Dataset)
	trainData, err := api.NewDataset(XTrain, yTrain,
		api.WithFeatureNames([]string{"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"}),
	)
	if err != nil {
		log.Fatal("Failed to create training dataset:", err)
	}

	validData, err := api.NewDataset(XValid, yValid,
		api.WithReference(trainData), // Use same binning as training data
	)
	if err != nil {
		log.Fatal("Failed to create validation dataset:", err)
	}

	// Set parameters (similar to Python params dict)
	params := map[string]interface{}{
		"objective":        "binary",
		"metric":          "binary_logloss",
		"num_leaves":      31,
		"learning_rate":   0.05,
		"feature_fraction": 0.9,
		"bagging_fraction": 0.8,
		"bagging_freq":    5,
		"verbosity":       1,
		"seed":           42,
		"num_threads":    -1,
	}

	fmt.Println("\nParameters:")
	for key, value := range params {
		fmt.Printf("  %s: %v\n", key, value)
	}
	fmt.Println()

	// Train model (similar to lgb.train)
	fmt.Println("Training model...")
	fmt.Println("=" + repeatStr("=", 50))
	
	bst, err := api.Train(
		params,
		trainData,
		100, // num_boost_round
		[]*api.Dataset{validData},
		api.WithValidNames([]string{"valid"}),
		api.WithEarlyStopping(10),
		api.WithVerboseEval(true, 10),
	)
	
	if err != nil {
		log.Fatal("Training failed:", err)
	}

	fmt.Println("=" + repeatStr("=", 50))
	fmt.Println()

	// Print training summary
	fmt.Println("Training Summary:")
	fmt.Printf("  Total iterations: %d\n", bst.CurrentIteration())
	fmt.Printf("  Best iteration: %d\n", bst.BestIteration())
	fmt.Printf("  Number of trees: %d\n", bst.NumTrees())
	fmt.Printf("  Number of features: %d\n", bst.NumFeatures())
	fmt.Println()

	// Get feature importance
	fmt.Println("Feature Importance (gain):")
	importance := bst.FeatureImportance("gain")
	featureNames := trainData.FeatureNames
	for i, imp := range importance {
		if imp > 0 {
			name := fmt.Sprintf("Feature_%d", i)
			if i < len(featureNames) {
				name = featureNames[i]
			}
			fmt.Printf("  %s: %.4f\n", name, imp)
		}
	}
	fmt.Println()

	// Make predictions on test set
	fmt.Println("Making predictions on validation set...")
	predictions, err := bst.Predict(XValid)
	if err != nil {
		log.Fatal("Prediction failed:", err)
	}

	// Calculate accuracy
	accuracy := calculateAccuracy(yValid, predictions)
	fmt.Printf("Validation Accuracy: %.2f%%\n", accuracy*100)
	fmt.Println()

	// Save model (similar to bst.save_model)
	modelFile := "model.json"
	fmt.Printf("Saving model to %s...\n", modelFile)
	err = bst.SaveModel(modelFile, api.WithSaveType("json"))
	if err != nil {
		log.Fatal("Failed to save model:", err)
	}

	// Load model (similar to lgb.Booster(model_file=...))
	fmt.Printf("Loading model from %s...\n", modelFile)
	loadedBst, err := api.LoadModel(modelFile)
	if err != nil {
		log.Fatal("Failed to load model:", err)
	}

	// Verify loaded model works
	loadedPreds, err := loadedBst.Predict(XValid)
	if err != nil {
		log.Fatal("Prediction with loaded model failed:", err)
	}

	// Check predictions match
	if matricesEqual(predictions, loadedPreds) {
		fmt.Println("✓ Loaded model produces identical predictions")
	} else {
		fmt.Println("✗ Loaded model predictions differ")
	}

	fmt.Println()
	fmt.Println("=== Example Complete ===")
}

// Helper functions

// generateRandomMatrix creates a random matrix for demonstration purposes
// Note: Uses math/rand for reproducible examples. For cryptographic purposes,
// use crypto/rand instead.
func generateRandomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64() // math/rand: acceptable for ML examples
	}
	return mat.NewDense(rows, cols, data)
}

// generateBinaryLabels creates binary classification labels for demonstration
// Note: Uses math/rand for reproducible examples. For cryptographic purposes,
// use crypto/rand instead.
func generateBinaryLabels(n int) *mat.Dense {
	data := make([]float64, n)
	for i := range data {
		if rand.Float64() > 0.5 { // math/rand: acceptable for ML examples
			data[i] = 1.0
		}
	}
	return mat.NewDense(n, 1, data)
}

func calculateAccuracy(yTrue, yPred mat.Matrix) float64 {
	rows, _ := yTrue.Dims()
	correct := 0
	
	for i := 0; i < rows; i++ {
		trueVal := yTrue.At(i, 0)
		predVal := yPred.At(i, 0)
		
		// Convert probability to class
		predClass := 0.0
		if predVal > 0.5 {
			predClass = 1.0
		}
		
		if trueVal == predClass {
			correct++
		}
	}
	
	return float64(correct) / float64(rows)
}

func matricesEqual(a, b mat.Matrix) bool {
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	
	if ra != rb || ca != cb {
		return false
	}
	
	for i := 0; i < ra; i++ {
		for j := 0; j < ca; j++ {
			if a.At(i, j) != b.At(i, j) {
				return false
			}
		}
	}
	
	return true
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}