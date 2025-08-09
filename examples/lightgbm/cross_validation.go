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
	fmt.Println("=== LightGBM Cross-Validation Example ===")

	// Generate synthetic data
	nSamples, nFeatures := 2000, 15
	X, y := generateRegressionData(nSamples, nFeatures, 42)

	fmt.Printf("Dataset: %d samples, %d features\n", nSamples, nFeatures)

	// Set up training parameters
	params := lightgbm.TrainingParams{
		NumIterations: 100,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      -1,
		MinDataInLeaf: 20,
		Lambda:        0.1, // L2 regularization
		Alpha:         0.1, // L1 regularization
		Objective:     "regression",
		Metric:        "rmse",
		Seed:          42,
		Verbosity:     0, // Quiet during CV
	}

	fmt.Printf("Training parameters: %d estimators, LR=%.2f\n", 
		params.NumIterations, params.LearningRate)

	// Example 1: K-Fold Cross-Validation
	fmt.Println("\n--- K-Fold Cross-Validation ---")
	
	kfold := lightgbm.NewKFold(5, true, 42) // 5-fold, shuffle, seed=42
	
	start := time.Now()
	result, err := lightgbm.CrossValidate(
		params,   // Training parameters
		X, y,     // Data
		kfold,    // Cross-validation splitter
		"rmse",   // Evaluation metric
		10,       // Early stopping rounds
		true,     // Verbose output
	)
	cvTime := time.Since(start)

	if err != nil {
		log.Fatal("Cross-validation failed:", err)
	}

	// Print results
	fmt.Printf("Cross-validation completed in %.3fs\n", cvTime.Seconds())
	fmt.Printf("Mean RMSE: %.6f (±%.6f)\n", 
		result.GetMeanScore(), result.GetStdScore())
	fmt.Printf("Best iteration: %d\n", result.BestIteration)

	// Show individual fold scores
	fmt.Println("\nIndividual fold scores:")
	for i, score := range result.TestScores {
		fmt.Printf("  Fold %d: %.6f\n", i+1, score)
	}

	// Example 2: Stratified K-Fold for Classification Data
	fmt.Println("\n--- Stratified K-Fold (Classification Data) ---")
	
	// Generate classification data
	XClass, yClass := generateClassificationData(1000, 10, 42)
	
	// Convert to classification parameters
	classParams := params
	classParams.Objective = "binary"
	classParams.Metric = "auc"
	classParams.NumIterations = 50 // Fewer iterations for demo

	skfold := lightgbm.NewStratifiedKFold(5, true, 42)
	
	start = time.Now()
	classResult, err := lightgbm.CrossValidate(
		classParams,
		XClass, yClass,
		skfold,
		"auc",    // AUC for binary classification
		5,        // Early stopping
		true,
	)
	stratTime := time.Since(start)

	if err != nil {
		log.Fatal("Stratified cross-validation failed:", err)
	}

	fmt.Printf("Stratified CV completed in %.3fs\n", stratTime.Seconds())
	fmt.Printf("Mean AUC: %.6f (±%.6f)\n", 
		classResult.GetMeanScore(), classResult.GetStdScore())

	// Example 3: Multiple Metrics Evaluation
	fmt.Println("\n--- Multi-Metric Evaluation ---")
	
	// Train a single model for multi-metric evaluation
	trainer := lightgbm.NewTrainer(params)
	err = trainer.Fit(X, y)
	if err != nil {
		log.Fatal("Training failed:", err)
	}

	model := trainer.GetModel()
	predictor := lightgbm.NewPredictor(model)

	// Split data for evaluation
	splitIdx := nSamples / 2
	XTest := X.Slice(splitIdx, nSamples, 0, nFeatures).(*mat.Dense)
	yTest := y.Slice(splitIdx, nSamples, 0, 1).(*mat.Dense)

	predictions, err := predictor.Predict(XTest)
	if err != nil {
		log.Fatal("Prediction failed:", err)
	}

	// Calculate multiple metrics
	metrics := map[string]func(*mat.Dense, *mat.Dense) (float64, error){
		"RMSE": lightgbm.RMSE,
		"MAE":  lightgbm.MAE,
		"MAPE": lightgbm.MAPE,
		"R²":   lightgbm.R2Score,
	}

	fmt.Println("Multi-metric evaluation:")
	for name, metricFunc := range metrics {
		score, err := metricFunc(yTest, predictions)
		if err != nil {
			fmt.Printf("  %s: Error - %v\n", name, err)
		} else {
			fmt.Printf("  %s: %.6f\n", name, score)
		}
	}

	// Example 4: Cross-Validation with Different Metrics
	fmt.Println("\n--- CV with Different Metrics ---")
	
	metricsToTest := []string{"rmse", "mae", "mape"}
	
	for _, metric := range metricsToTest {
		fmt.Printf("\nEvaluating with %s:\n", metric)
		
		params.Metric = metric
		quickResult, err := lightgbm.CrossValidate(
			params, X, y, 
			lightgbm.NewKFold(3, true, 42), // 3-fold for speed
			metric, 5, false, // No verbose
		)
		
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}
		
		fmt.Printf("  Mean %s: %.6f (±%.6f)\n", 
			metric, quickResult.GetMeanScore(), quickResult.GetStdScore())
	}

	// Summary
	fmt.Println("\n--- Summary ---")
	fmt.Printf("✅ K-Fold CV: %.6f RMSE in %.3fs\n", 
		result.GetMeanScore(), cvTime.Seconds())
	fmt.Printf("✅ Stratified CV: %.6f AUC in %.3fs\n", 
		classResult.GetMeanScore(), stratTime.Seconds())
	fmt.Println("✅ Multi-metric evaluation completed")
	fmt.Println("✅ Cross-validation example completed successfully!")
}

// generateRegressionData creates synthetic regression data
func generateRegressionData(nSamples, nFeatures, seed int) (*mat.Dense, *mat.Dense) {
	rand.Seed(int64(seed))

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}
	}

	// Generate targets with linear relationship plus noise
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		target := 0.0
		// Use first few features for target
		for j := 0; j < min(5, nFeatures); j++ {
			weight := 1.0 / float64(j+1) // Decreasing importance
			target += X.At(i, j) * weight
		}
		// Add noise
		target += rand.NormFloat64() * 0.1
		y.Set(i, 0, target)
	}

	return X, y
}

// generateClassificationData creates synthetic binary classification data
func generateClassificationData(nSamples, nFeatures, seed int) (*mat.Dense, *mat.Dense) {
	rand.Seed(int64(seed))

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		// Generate features
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}

		// Generate binary target based on linear combination
		score := 0.0
		for j := 0; j < min(3, nFeatures); j++ {
			score += X.At(i, j)
		}
		
		// Add noise and convert to probability
		score += rand.NormFloat64() * 0.5
		
		// Convert to binary (0 or 1)
		if score > 0 {
			y.Set(i, 0, 1.0)
		} else {
			y.Set(i, 0, 0.0)
		}
	}

	return X, y
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}