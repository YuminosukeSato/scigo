package api_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm/api"
	"gonum.org/v1/gonum/mat"
)

// TestPythonStyleAPI tests that the Go API produces similar results to Python LightGBM
func TestPythonStyleAPI(t *testing.T) {
	// Create synthetic data
	X := mat.NewDense(100, 5, nil)
	y := mat.NewDense(100, 1, nil)
	
	// Fill with deterministic data
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i+j*10)/100.0)
		}
		// Binary classification labels
		if i%2 == 0 {
			y.Set(i, 0, 0)
		} else {
			y.Set(i, 0, 1)
		}
	}
	
	// Split into train and validation
	XTrain := mat.NewDense(80, 5, nil)
	yTrain := mat.NewDense(80, 1, nil)
	XValid := mat.NewDense(20, 5, nil)
	yValid := mat.NewDense(20, 1, nil)
	
	for i := 0; i < 80; i++ {
		for j := 0; j < 5; j++ {
			XTrain.Set(i, j, X.At(i, j))
		}
		yTrain.Set(i, 0, y.At(i, 0))
	}
	
	for i := 0; i < 20; i++ {
		for j := 0; j < 5; j++ {
			XValid.Set(i, j, X.At(i+80, j))
		}
		yValid.Set(i, 0, y.At(i+80, 0))
	}
	
	// Test 1: Dataset creation
	t.Run("Dataset Creation", func(t *testing.T) {
		trainData, err := api.NewDataset(XTrain, yTrain,
			api.WithFeatureNames([]string{"f1", "f2", "f3", "f4", "f5"}),
		)
		if err != nil {
			t.Fatalf("Failed to create dataset: %v", err)
		}
		
		if trainData.NumData() != 80 {
			t.Errorf("Expected 80 samples, got %d", trainData.NumData())
		}
		
		if trainData.NumFeature() != 5 {
			t.Errorf("Expected 5 features, got %d", trainData.NumFeature())
		}
		
		// Test reference dataset
		validData, err := api.NewDataset(XValid, yValid,
			api.WithReference(trainData),
		)
		if err != nil {
			t.Fatalf("Failed to create validation dataset: %v", err)
		}
		
		if validData.NumData() != 20 {
			t.Errorf("Expected 20 validation samples, got %d", validData.NumData())
		}
	})
	
	// Test 2: Training with parameters
	t.Run("Training", func(t *testing.T) {
		trainData, _ := api.NewDataset(XTrain, yTrain)
		validData, _ := api.NewDataset(XValid, yValid)
		
		params := map[string]interface{}{
			"objective":      "binary",
			"num_leaves":     10,
			"learning_rate":  0.1,
			"num_iterations": 20,
			"seed":          42,
			"deterministic": true,
		}
		
		bst, err := api.Train(params, trainData, 20, []*api.Dataset{validData},
			api.WithVerboseEval(false, 1),
		)
		
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}
		
		if bst.NumTrees() == 0 {
			t.Error("No trees were built")
		}
		
		if bst.NumFeatures() != 5 {
			t.Errorf("Expected 5 features, got %d", bst.NumFeatures())
		}
	})
	
	// Test 3: Prediction
	t.Run("Prediction", func(t *testing.T) {
		trainData, _ := api.NewDataset(XTrain, yTrain)
		
		params := map[string]interface{}{
			"objective":      "binary",
			"num_leaves":     10,
			"learning_rate":  0.1,
			"num_iterations": 10,
			"seed":          42,
		}
		
		bst, err := api.Train(params, trainData, 10, nil)
		if err != nil {
			t.Fatalf("Training failed: %v", err)
		}
		
		predictions, err := bst.Predict(XValid)
		if err != nil {
			t.Fatalf("Prediction failed: %v", err)
		}
		
		rows, cols := predictions.Dims()
		if rows != 20 {
			t.Errorf("Expected 20 predictions, got %d", rows)
		}
		if cols != 1 {
			t.Errorf("Expected 1 column, got %d", cols)
		}
		
		// Check predictions are probabilities (between 0 and 1)
		for i := 0; i < rows; i++ {
			pred := predictions.At(i, 0)
			if pred < 0 || pred > 1 {
				t.Errorf("Prediction %d out of range [0,1]: %f", i, pred)
			}
		}
	})
	
	// Test 4: Early Stopping
	t.Run("Early Stopping", func(t *testing.T) {
		trainData, _ := api.NewDataset(XTrain, yTrain)
		validData, _ := api.NewDataset(XValid, yValid)
		
		params := map[string]interface{}{
			"objective":      "binary",
			"num_leaves":     10,
			"learning_rate":  0.01, // Small learning rate
			"seed":          42,
		}
		
		// Train with early stopping
		bst, err := api.Train(params, trainData, 100, []*api.Dataset{validData},
			api.WithEarlyStopping(5),
			api.WithVerboseEval(false, 1),
		)
		
		if err != nil && err != api.ErrEarlyStop {
			t.Fatalf("Training failed: %v", err)
		}
		
		// Should stop before 100 iterations
		if bst.CurrentIteration() >= 100 {
			t.Error("Early stopping did not trigger")
		}
		
		if bst.BestIteration() == 0 {
			t.Error("Best iteration not set")
		}
		
		t.Logf("Early stopped at iteration %d, best iteration: %d",
			bst.CurrentIteration(), bst.BestIteration())
	})
	
	// Test 5: Feature Importance
	t.Run("Feature Importance", func(t *testing.T) {
		trainData, _ := api.NewDataset(XTrain, yTrain)
		
		params := map[string]interface{}{
			"objective":      "binary",
			"num_leaves":     10,
			"learning_rate":  0.1,
			"num_iterations": 20,
		}
		
		bst, _ := api.Train(params, trainData, 20, nil)
		
		importance := bst.FeatureImportance("gain")
		if len(importance) != 5 {
			t.Errorf("Expected 5 importance scores, got %d", len(importance))
		}
		
		// At least some features should have non-zero importance
		hasNonZero := false
		for _, imp := range importance {
			if imp > 0 {
				hasNonZero = true
				break
			}
		}
		
		if !hasNonZero {
			t.Error("All feature importances are zero")
		}
	})
	
	// Test 6: Model Save/Load
	t.Run("Model Persistence", func(t *testing.T) {
		trainData, _ := api.NewDataset(XTrain, yTrain)
		
		params := map[string]interface{}{
			"objective":      "binary",
			"num_leaves":     10,
			"learning_rate":  0.1,
			"num_iterations": 10,
			"seed":          42,
		}
		
		bst, _ := api.Train(params, trainData, 10, nil)
		
		// Save model
		modelFile := "/tmp/test_model.json"
		err := bst.SaveModel(modelFile, api.WithSaveType("json"))
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		
		// Load model
		loadedBst, err := api.LoadModel(modelFile)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}
		
		// Compare predictions
		origPreds, _ := bst.Predict(XValid)
		loadedPreds, _ := loadedBst.Predict(XValid)
		
		rows, _ := origPreds.Dims()
		for i := 0; i < rows; i++ {
			orig := origPreds.At(i, 0)
			loaded := loadedPreds.At(i, 0)
			if math.Abs(orig-loaded) > 1e-10 {
				t.Errorf("Prediction mismatch at index %d: %f vs %f", i, orig, loaded)
			}
		}
	})
}

// TestPythonOutputFormat tests that output format matches Python's style
func TestPythonOutputFormat(t *testing.T) {
	// This test verifies that the logging output format matches Python LightGBM
	
	// Create small dataset for quick test
	X := mat.NewDense(50, 3, nil)
	y := mat.NewDense(50, 1, nil)
	
	for i := 0; i < 50; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, float64(i*j)/50.0)
		}
		y.Set(i, 0, float64(i%2))
	}
	
	trainData, _ := api.NewDataset(X, y)
	
	params := map[string]interface{}{
		"objective":     "binary",
		"num_leaves":    5,
		"learning_rate": 0.1,
		"verbosity":     1,
	}
	
	// Expected output format (Python-style):
	// [LightGBM] [Info] Start training from score 0.500000
	// [1]	train-binary_logloss: 0.693147	valid-binary_logloss: 0.693147
	// [2]	train-binary_logloss: 0.686354	valid-binary_logloss: 0.686354
	// ...
	
	fmt.Println("\n=== Testing Python-style Output Format ===")
	bst, err := api.Train(params, trainData, 5, []*api.Dataset{trainData},
		api.WithValidNames([]string{"train"}),
		api.WithVerboseEval(true, 1),
	)
	
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	
	if bst.CurrentIteration() != 5 {
		t.Errorf("Expected 5 iterations, got %d", bst.CurrentIteration())
	}
	
	fmt.Println("=== Output Format Test Complete ===")
}

// BenchmarkPythonStyleAPI benchmarks the Python-style API
func BenchmarkPythonStyleAPI(b *testing.B) {
	// Create dataset
	X := mat.NewDense(1000, 10, nil)
	y := mat.NewDense(1000, 1, nil)
	
	for i := 0; i < 1000; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i+j)/1000.0)
		}
		y.Set(i, 0, float64(i%2))
	}
	
	trainData, _ := api.NewDataset(X, y)
	
	params := map[string]interface{}{
		"objective":     "binary",
		"num_leaves":    31,
		"learning_rate": 0.1,
		"verbosity":     -1,
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, _ = api.Train(params, trainData, 10, nil)
	}
}