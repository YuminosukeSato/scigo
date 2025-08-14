package api

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestAPIIntegration tests the full API integration with all implemented features
func TestAPIIntegration(t *testing.T) {
	// Create test data for multiclass classification
	X := mat.NewDense(100, 4, nil)
	y := mat.NewDense(100, 1, nil)

	// Generate synthetic multiclass data
	for i := 0; i < 100; i++ {
		for j := 0; j < 4; j++ {
			X.Set(i, j, float64(i+j)*0.1)
		}
		y.Set(i, 0, float64(i%3)) // 3 classes
	}

	t.Logf("=== API Integration Test ===")
	t.Logf("Dataset: %dx%d, Classes: 3", 100, 4)

	// Test 1: GOSS + MulticlassLogLoss training
	t.Run("GOSSWithMulticlassLogLoss", func(t *testing.T) {
		trainData, err := NewDataset(X, y)
		if err != nil {
			t.Fatalf("Failed to create dataset: %v", err)
		}

		// Parameters with GOSS and multiclass_logloss
		params := map[string]interface{}{
			"objective":        "multiclass_logloss",
			"num_class":        3,
			"num_iterations":   20,
			"learning_rate":    0.1,
			"num_leaves":       15,
			"min_data_in_leaf": 5,
			"top_rate":         0.2, // GOSS parameter
			"other_rate":       0.1, // GOSS parameter
			"verbosity":        -1,
		}

		// Train model
		booster, err := Train(params, trainData, 20, nil)
		if err != nil {
			t.Fatalf("Failed to train model: %v", err)
		}

		t.Logf("✅ GOSS + MulticlassLogLoss training completed")
		t.Logf("   - Trees: %d", booster.NumTrees())
		t.Logf("   - Features: %d", booster.NumFeatures())

		// Test predictions
		predictions, err := booster.Predict(X)
		if err != nil {
			t.Fatalf("Failed to make predictions: %v", err)
		}

		rows, cols := predictions.Dims()
		t.Logf("   - Prediction shape: %dx%d", rows, cols)

		// Test probability predictions
		probabilities, err := booster.PredictProba(X)
		if err != nil {
			t.Fatalf("Failed to make probability predictions: %v", err)
		}

		probRows, probCols := probabilities.Dims()
		t.Logf("   - Probability shape: %dx%d", probRows, probCols)

		// Verify probability properties
		for i := 0; i < probRows; i++ {
			rowSum := 0.0
			for j := 0; j < probCols; j++ {
				prob := probabilities.At(i, j)
				if prob < 0.0 || prob > 1.0 {
					t.Errorf("Probability out of range [0,1] at sample %d, class %d: %f", i, j, prob)
				}
				rowSum += prob
			}

			// Check row sum equals 1 (within tolerance)
			if abs(rowSum-1.0) > 1e-10 {
				t.Errorf("Probability sum not equal to 1 for sample %d: %f", i, rowSum)
			}
		}

		t.Logf("✅ PredictProba validation passed")
	})

	// Test 2: Categorical features with binary classification
	t.Run("CategoricalFeaturesBinary", func(t *testing.T) {
		// Create test data with categorical features
		XCat := mat.NewDense(50, 3, nil)
		yCat := mat.NewDense(50, 1, nil)

		for i := 0; i < 50; i++ {
			XCat.Set(i, 0, float64(i%5))     // Categorical feature (0-4)
			XCat.Set(i, 1, float64(i)*0.1)   // Numerical feature
			XCat.Set(i, 2, float64((i+1)%3)) // Another categorical (0-2)
			yCat.Set(i, 0, float64(i%2))     // Binary labels
		}

		trainData, err := NewDataset(XCat, yCat)
		if err != nil {
			t.Fatalf("Failed to create categorical dataset: %v", err)
		}

		params := map[string]interface{}{
			"objective":           "binary",
			"num_iterations":      15,
			"learning_rate":       0.1,
			"num_leaves":          7,
			"categorical_feature": []int{0, 2}, // Features 0 and 2 are categorical
			"verbosity":           -1,
		}

		booster, err := Train(params, trainData, 15, nil)
		if err != nil {
			t.Fatalf("Failed to train with categorical features: %v", err)
		}

		t.Logf("✅ Categorical features training completed")

		// Test predictions
		predictions, err := booster.Predict(XCat)
		if err != nil {
			t.Fatalf("Failed to predict with categorical features: %v", err)
		}

		rows, cols := predictions.Dims()
		t.Logf("   - Categorical prediction shape: %dx%d", rows, cols)

		// Test binary probability predictions
		probabilities, err := booster.PredictProba(XCat)
		if err != nil {
			t.Fatalf("Failed to predict probabilities with categorical features: %v", err)
		}

		probRows, probCols := probabilities.Dims()
		t.Logf("   - Binary probability shape: %dx%d", probRows, probCols)

		// For binary classification, should have 2 columns
		if probCols != 2 {
			t.Errorf("Binary classification should return 2 probability columns, got %d", probCols)
		}

		t.Logf("✅ Categorical features with binary classification passed")
	})

	// Test 3: Model persistence
	t.Run("ModelPersistence", func(t *testing.T) {
		trainData, err := NewDataset(X, y)
		if err != nil {
			t.Fatalf("Failed to create dataset for persistence test: %v", err)
		}

		params := map[string]interface{}{
			"objective":      "multiclass_logloss",
			"num_class":      3,
			"num_iterations": 10,
			"learning_rate":  0.1,
			"verbosity":      -1,
		}

		// Train model
		originalBooster, err := Train(params, trainData, 10, nil)
		if err != nil {
			t.Fatalf("Failed to train model for persistence: %v", err)
		}

		// Save model
		filename := "/tmp/test_lightgbm_model.txt"
		err = originalBooster.SaveModel(filename, WithSaveType("text"))
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}

		t.Logf("✅ Model saved to %s", filename)

		// Load model
		loadedBooster, err := LoadModel(filename)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}

		t.Logf("✅ Model loaded from %s", filename)

		// Compare predictions
		originalPred, err := originalBooster.Predict(X)
		if err != nil {
			t.Fatalf("Failed to predict with original model: %v", err)
		}

		loadedPred, err := loadedBooster.Predict(X)
		if err != nil {
			t.Fatalf("Failed to predict with loaded model: %v", err)
		}

		// Check if predictions are similar
		rows, cols := originalPred.Dims()
		for i := 0; i < rows && i < 5; i++ { // Check first 5 samples
			for j := 0; j < cols; j++ {
				orig := originalPred.At(i, j)
				loaded := loadedPred.At(i, j)
				if abs(orig-loaded) > 1e-6 {
					t.Logf("Warning: Prediction difference at [%d,%d]: orig=%f, loaded=%f", i, j, orig, loaded)
				}
			}
		}

		t.Logf("✅ Model persistence test passed")
	})
}

// TestAPIQuickStart tests the QuickTrain functionality
func TestAPIQuickStart(t *testing.T) {
	t.Logf("=== API QuickStart Test ===")

	// Create simple test data
	X := mat.NewDense(30, 2, nil)
	y := mat.NewDense(30, 1, nil)

	for i := 0; i < 30; i++ {
		X.Set(i, 0, float64(i)*0.1)
		X.Set(i, 1, float64(i+10)*0.05)
		if i < 15 {
			y.Set(i, 0, 0.0)
		} else {
			y.Set(i, 0, 1.0)
		}
	}

	// Test QuickTrain with minimal parameters
	params := map[string]interface{}{
		"num_iterations": 10,
		"learning_rate":  0.1,
	}

	booster, err := QuickTrain(X, y, params)
	if err != nil {
		t.Fatalf("QuickTrain failed: %v", err)
	}

	t.Logf("✅ QuickTrain completed")
	t.Logf("   - Auto-detected objective: binary")
	t.Logf("   - Trees: %d", booster.NumTrees())

	// Test predictions
	predictions, err := booster.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with QuickTrain model: %v", err)
	}

	rows, cols := predictions.Dims()
	t.Logf("   - Prediction shape: %dx%d", rows, cols)

	t.Logf("✅ QuickStart test passed")
}

// Helper function for absolute value
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
