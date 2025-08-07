package tree

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestDecisionTreeClassifier_FitPredict_Binary tests binary classification
func TestDecisionTreeClassifier_FitPredict_Binary(t *testing.T) {
	// Create simple linearly separable data
	X := mat.NewDense(8, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
		3, 3,
		3, 4,
		4, 3,
		4, 4,
	})

	y := mat.NewDense(8, 1, []float64{
		0, 0, 0, 0, // Class 0 (lower left)
		1, 1, 1, 1, // Class 1 (upper right)
	})

	// Create and train model
	dt := NewDecisionTreeClassifier(
		WithCriterion("gini"),
		WithMaxDepth(5),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Test predictions on training data
	predictions, err := dt.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check all predictions are correct
	for i := 0; i < 8; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if pred != actual {
			t.Errorf("Sample %d: expected %v, got %v", i, actual, pred)
		}
	}

	// Test on new data
	XTest := mat.NewDense(2, 2, []float64{
		0.5, 0.5, // Should be class 0
		3.5, 3.5, // Should be class 1
	})

	testPreds, err := dt.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict on test data: %v", err)
	}

	if testPreds.At(0, 0) != 0 {
		t.Errorf("Test point (0.5,0.5) should be class 0, got %v", testPreds.At(0, 0))
	}

	if testPreds.At(1, 0) != 1 {
		t.Errorf("Test point (3.5,3.5) should be class 1, got %v", testPreds.At(1, 0))
	}
}

// TestDecisionTreeClassifier_PredictProba tests probability predictions
func TestDecisionTreeClassifier_PredictProba(t *testing.T) {
	// Simple data
	X := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		2, 2,
		2, 3,
		3, 2,
	})

	y := mat.NewDense(6, 1, []float64{
		0, 0, 0, // Class 0
		1, 1, 1, // Class 1
	})

	dt := NewDecisionTreeClassifier(
		WithMaxDepth(3),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	probas, err := dt.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	rows, cols := probas.Dims()
	if rows != 6 || cols != 2 {
		t.Errorf("Expected probas shape (6, 2), got (%d, %d)", rows, cols)
	}

	// Check that probabilities sum to 1
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			prob := probas.At(i, j)
			if prob < 0 || prob > 1 {
				t.Errorf("Invalid probability at (%d, %d): %v", i, j, prob)
			}
			sum += prob
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("Probabilities for sample %d don't sum to 1: %v", i, sum)
		}
	}
}

// TestDecisionTreeClassifier_Score tests accuracy calculation
func TestDecisionTreeClassifier_Score(t *testing.T) {
	// Create XOR-like data with more samples for better learning
	X := mat.NewDense(8, 2, []float64{
		0.0, 0.0,
		0.0, 0.1,
		0.1, 1.0,
		0.0, 0.9,
		1.0, 0.0,
		0.9, 0.0,
		1.0, 1.0,
		0.9, 0.9,
	})

	// XOR-like pattern: class 0 when both features are similar (both low or both high)
	y := mat.NewDense(8, 1, []float64{
		0, 0, // Both low -> class 0
		1, 1, // One high, one low -> class 1
		1, 1, // One high, one low -> class 1
		0, 0, // Both high -> class 0
	})

	dt := NewDecisionTreeClassifier(
		WithMaxDepth(5), // Allow deeper tree for XOR pattern
		WithMinSamplesLeaf(1),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	score := dt.Score(X, y)
	if score != 1.0 {
		t.Errorf("Decision tree should perfectly fit XOR-like data with enough samples, got score: %v", score)
	}

	// Also test on simpler linearly separable data
	XSimple := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		2, 2,
		2, 3,
		3, 2,
	})

	ySimple := mat.NewDense(6, 1, []float64{
		0, 0, 0,
		1, 1, 1,
	})

	dtSimple := NewDecisionTreeClassifier(WithMaxDepth(3))
	dtSimple.Fit(XSimple, ySimple)

	scoreSimple := dtSimple.Score(XSimple, ySimple)
	if scoreSimple != 1.0 {
		t.Errorf("Decision tree should perfectly fit linearly separable data, got score: %v", scoreSimple)
	}
}

// TestDecisionTreeClassifier_Multiclass tests multiclass classification
func TestDecisionTreeClassifier_Multiclass(t *testing.T) {
	// Create 3-class data
	X := mat.NewDense(9, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		3, 3,
		3, 4,
		4, 3,
		6, 6,
		6, 7,
		7, 6,
	})

	y := mat.NewDense(9, 1, []float64{
		0, 0, 0, // Class 0
		1, 1, 1, // Class 1
		2, 2, 2, // Class 2
	})

	dt := NewDecisionTreeClassifier(
		WithCriterion("gini"),
		WithMaxDepth(5),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit multiclass model: %v", err)
	}

	// Check that we have 3 classes
	if dt.nClasses_ != 3 {
		t.Errorf("Expected 3 classes, got %d", dt.nClasses_)
	}

	// Check predictions
	predictions, err := dt.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	correct := 0
	for i := 0; i < 9; i++ {
		if predictions.At(i, 0) == y.At(i, 0) {
			correct++
		}
	}

	accuracy := float64(correct) / 9.0
	if accuracy != 1.0 {
		t.Errorf("Expected perfect accuracy on training data, got: %v", accuracy)
	}

	// Test probability predictions
	probas, err := dt.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	rows, cols := probas.Dims()
	if cols != 3 {
		t.Errorf("Expected 3 probability columns, got %d", cols)
	}

	// Check probability constraints
	for i := 0; i < rows; i++ {
		sum := 0.0
		maxProb := 0.0
		maxClass := -1

		for j := 0; j < cols; j++ {
			prob := probas.At(i, j)
			if prob < 0 || prob > 1 {
				t.Errorf("Invalid probability at (%d, %d): %v", i, j, prob)
			}
			sum += prob

			if prob > maxProb {
				maxProb = prob
				maxClass = j
			}
		}

		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("Probabilities for sample %d don't sum to 1: %v", i, sum)
		}

		// Check that max probability corresponds to predicted class
		expectedClass := int(y.At(i, 0))
		if maxClass != expectedClass {
			t.Errorf("Sample %d: max probability class %d doesn't match expected %d",
				i, maxClass, expectedClass)
		}
	}
}

// TestDecisionTreeClassifier_Entropy tests entropy criterion
func TestDecisionTreeClassifier_Entropy(t *testing.T) {
	X := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		2, 2,
		2, 3,
		3, 2,
	})

	y := mat.NewDense(6, 1, []float64{
		0, 0, 0,
		1, 1, 1,
	})

	// Test with entropy criterion
	dt := NewDecisionTreeClassifier(
		WithCriterion("entropy"),
		WithMaxDepth(3),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit with entropy: %v", err)
	}

	score := dt.Score(X, y)
	if score != 1.0 {
		t.Errorf("Expected perfect score on simple data, got %v", score)
	}
}

// TestDecisionTreeClassifier_FeatureImportance tests feature importance calculation
func TestDecisionTreeClassifier_FeatureImportance(t *testing.T) {
	// Create data where feature 0 is more important
	X := mat.NewDense(8, 3, []float64{
		0, 0, 0, // Feature 0 determines class
		0, 1, 1,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0, // When feature 0 = 1, always class 1
		1, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})

	y := mat.NewDense(8, 1, []float64{
		0, 0, 0, 0, // Class 0 when feature 0 = 0
		1, 1, 1, 1, // Class 1 when feature 0 = 1
	})

	dt := NewDecisionTreeClassifier()
	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	importances := dt.GetFeatureImportances()
	if len(importances) != 3 {
		t.Fatalf("Expected 3 feature importances, got %d", len(importances))
	}

	// Feature 0 should have highest importance
	if importances[0] <= importances[1] || importances[0] <= importances[2] {
		t.Errorf("Feature 0 should have highest importance: %v", importances)
	}

	// Sum should be 1 (normalized)
	sum := 0.0
	for _, imp := range importances {
		sum += imp
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Feature importances should sum to 1, got %v", sum)
	}
}

// TestDecisionTreeClassifier_MaxDepth tests max depth constraint
func TestDecisionTreeClassifier_MaxDepth(t *testing.T) {
	// Create data that would normally require deep tree
	X := mat.NewDense(16, 2, nil)
	y := mat.NewDense(16, 1, nil)

	for i := 0; i < 16; i++ {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i%4))
		y.Set(i, 0, float64(i%2))
	}

	// Test with shallow tree
	dt := NewDecisionTreeClassifier(
		WithMaxDepth(2),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	depth := dt.GetDepth()
	if depth > 2 {
		t.Errorf("Tree depth %d exceeds max_depth=2", depth)
	}
}

// TestDecisionTreeClassifier_MinSamples tests minimum samples constraints
func TestDecisionTreeClassifier_MinSamples(t *testing.T) {
	X := mat.NewDense(10, 2, nil)
	y := mat.NewDense(10, 1, nil)

	for i := 0; i < 10; i++ {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i%3))
		y.Set(i, 0, float64(i%2))
	}

	// Test with min_samples_split
	dt := NewDecisionTreeClassifier(
		WithMinSamplesSplit(5),
		WithMinSamplesLeaf(2),
	)

	err := dt.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Tree should be shallow due to constraints
	nLeaves := dt.GetNLeaves()
	if nLeaves > 5 {
		t.Errorf("Too many leaves %d for min_samples constraints", nLeaves)
	}
}

// TestDecisionTreeClassifier_GetSetParams tests parameter management
func TestDecisionTreeClassifier_GetSetParams(t *testing.T) {
	dt := NewDecisionTreeClassifier()

	// Get default params
	params := dt.GetParams()

	// Check some defaults
	if params["criterion"].(string) != "gini" {
		t.Errorf("Default criterion should be 'gini', got %v", params["criterion"])
	}

	if params["min_samples_split"].(int) != 2 {
		t.Errorf("Default min_samples_split should be 2, got %v", params["min_samples_split"])
	}

	// Set new params
	newParams := map[string]interface{}{
		"criterion":         "entropy",
		"max_depth":         5,
		"min_samples_split": 4,
		"min_samples_leaf":  2,
	}

	err := dt.SetParams(newParams)
	if err != nil {
		t.Fatalf("Failed to set params: %v", err)
	}

	// Verify changes
	if dt.criterion != "entropy" {
		t.Errorf("criterion not updated: expected 'entropy', got %v", dt.criterion)
	}

	if dt.maxDepth != 5 {
		t.Errorf("max_depth not updated: expected 5, got %v", dt.maxDepth)
	}

	if dt.minSamplesSplit != 4 {
		t.Errorf("min_samples_split not updated: expected 4, got %v", dt.minSamplesSplit)
	}

	if dt.minSamplesLeaf != 2 {
		t.Errorf("min_samples_leaf not updated: expected 2, got %v", dt.minSamplesLeaf)
	}
}

// TestDecisionTreeClassifier_NotFitted tests error when predicting without fitting
func TestDecisionTreeClassifier_NotFitted(t *testing.T) {
	dt := NewDecisionTreeClassifier()

	X := mat.NewDense(2, 2, []float64{
		1, 2,
		3, 4,
	})

	_, err := dt.Predict(X)
	if err == nil {
		t.Error("Expected error when predicting without fitting")
	}

	_, err = dt.PredictProba(X)
	if err == nil {
		t.Error("Expected error when predicting probabilities without fitting")
	}
}
