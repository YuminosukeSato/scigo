package linear_model

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestLogisticRegression_FitPredict_Binary tests binary classification
func TestLogisticRegression_FitPredict_Binary(t *testing.T) {
	// Create simple linearly separable data
	// Class 0: points around (1, 1)
	// Class 1: points around (3, 3)
	X := mat.NewDense(6, 2, []float64{
		0.5, 0.5,
		1.0, 1.5,
		1.5, 1.0,
		3.0, 2.5,
		2.5, 3.0,
		3.5, 3.5,
	})
	
	y := mat.NewDense(6, 1, []float64{
		0, 0, 0,  // Class 0
		1, 1, 1,  // Class 1
	})
	
	// Create and train model
	lr := NewLogisticRegression(
		WithLRMaxIter(1000),
		WithLRTol(1e-4),
	)
	
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}
	
	// Test predictions on training data
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}
	
	// Check predictions
	for i := 0; i < 6; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if pred != actual {
			t.Errorf("Sample %d: expected %v, got %v", i, actual, pred)
		}
	}
	
	// Test on new data
	XTest := mat.NewDense(2, 2, []float64{
		1.0, 1.0,  // Should be class 0
		3.0, 3.0,  // Should be class 1
	})
	
	testPreds, err := lr.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict on test data: %v", err)
	}
	
	if testPreds.At(0, 0) != 0 {
		t.Errorf("Test point (1,1) should be class 0, got %v", testPreds.At(0, 0))
	}
	
	if testPreds.At(1, 0) != 1 {
		t.Errorf("Test point (3,3) should be class 1, got %v", testPreds.At(1, 0))
	}
}

// TestLogisticRegression_PredictProba tests probability predictions
func TestLogisticRegression_PredictProba(t *testing.T) {
	// Simple data
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	
	y := mat.NewDense(4, 1, []float64{
		0, 0, 1, 1,
	})
	
	lr := NewLogisticRegression(
		WithLRMaxIter(500),
	)
	
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}
	
	probas, err := lr.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}
	
	rows, cols := probas.Dims()
	if rows != 4 || cols != 2 {
		t.Errorf("Expected probas shape (4, 2), got (%d, %d)", rows, cols)
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
	
	// Check that higher probability corresponds to predicted class
	predictions, _ := lr.Predict(X)
	for i := 0; i < rows; i++ {
		pred := int(predictions.At(i, 0))
		prob0 := probas.At(i, 0)
		prob1 := probas.At(i, 1)
		
		if pred == 0 && prob0 <= prob1 {
			t.Errorf("Sample %d: predicted class 0 but P(0)=%v <= P(1)=%v", i, prob0, prob1)
		}
		if pred == 1 && prob1 <= prob0 {
			t.Errorf("Sample %d: predicted class 1 but P(1)=%v <= P(0)=%v", i, prob1, prob0)
		}
	}
}

// TestLogisticRegression_Score tests accuracy calculation
func TestLogisticRegression_Score(t *testing.T) {
	// Create XOR-like data (not linearly separable, but we'll use more features)
	X := mat.NewDense(8, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		0, 1, 1,
		1, 0, 0,
		1, 0, 1,
		1, 1, 0,
		1, 1, 1,
	})
	
	// Simple pattern: class 1 if sum of features > 1.5
	y := mat.NewDense(8, 1, []float64{
		0, 0, 0, 1, 0, 1, 1, 1,
	})
	
	lr := NewLogisticRegression(
		WithLRMaxIter(1000),
		WithLRC(10.0), // Less regularization for better fit
	)
	
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}
	
	score := lr.Score(X, y)
	if score < 0.75 { // Should achieve at least 75% accuracy
		t.Errorf("Score too low: %v", score)
	}
	
	// Perfect classification test with better separated data
	XSimple := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		3, 3,
		3, 4,
		4, 3,
	})
	ySimple := mat.NewDense(6, 1, []float64{
		0, 0, 0,  // Class 0 (lower values)
		1, 1, 1,  // Class 1 (higher values)
	})
	
	lr2 := NewLogisticRegression(
		WithLRMaxIter(1000),
		WithLRC(10.0),  // Less regularization for better fit
	)
	lr2.Fit(XSimple, ySimple)
	
	scoreSimple := lr2.Score(XSimple, ySimple)
	if scoreSimple != 1.0 {
		t.Errorf("Expected perfect score for linearly separable data, got %v", scoreSimple)
	}
}

// TestLogisticRegression_Regularization tests L2 regularization
func TestLogisticRegression_Regularization(t *testing.T) {
	// Create data with many features (prone to overfitting)
	X := mat.NewDense(10, 5, []float64{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 0, 0, 0,
		0, 1, 1, 0, 0,
		0, 0, 1, 1, 0,
		0, 0, 0, 1, 1,
		1, 0, 0, 0, 1,
	})
	
	y := mat.NewDense(10, 1, []float64{
		0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
	})
	
	// Train with strong regularization
	lrStrong := NewLogisticRegression(
		WithLRC(0.01), // Strong regularization (small C)
		WithLRMaxIter(1000),
	)
	lrStrong.Fit(X, y)
	
	// Train with weak regularization
	lrWeak := NewLogisticRegression(
		WithLRC(100.0), // Weak regularization (large C)
		WithLRMaxIter(1000),
	)
	lrWeak.Fit(X, y)
	
	// Check that strong regularization produces smaller weights
	strongNorm := 0.0
	weakNorm := 0.0
	
	for j := 0; j < 5; j++ {
		strongNorm += lrStrong.coef_[0][j] * lrStrong.coef_[0][j]
		weakNorm += lrWeak.coef_[0][j] * lrWeak.coef_[0][j]
	}
	
	strongNorm = math.Sqrt(strongNorm)
	weakNorm = math.Sqrt(weakNorm)
	
	if strongNorm >= weakNorm {
		t.Errorf("Strong regularization should produce smaller weights: strong=%v, weak=%v",
			strongNorm, weakNorm)
	}
}

// TestLogisticRegression_Multiclass tests multiclass classification
func TestLogisticRegression_Multiclass(t *testing.T) {
	// Create 3-class data
	X := mat.NewDense(9, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		2, 2,
		2, 3,
		3, 2,
		4, 4,
		4, 5,
		5, 4,
	})
	
	y := mat.NewDense(9, 1, []float64{
		0, 0, 0,  // Class 0
		1, 1, 1,  // Class 1
		2, 2, 2,  // Class 2
	})
	
	lr := NewLogisticRegression(
		WithLRMaxIter(1000),
		WithLRC(10.0),
	)
	
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit multiclass model: %v", err)
	}
	
	// Check that we have 3 classes
	if lr.nClasses_ != 3 {
		t.Errorf("Expected 3 classes, got %d", lr.nClasses_)
	}
	
	// Check predictions
	predictions, err := lr.Predict(X)
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
	if accuracy < 0.89 { // Should achieve at least 89% accuracy (8/9)
		t.Errorf("Multiclass accuracy too low: %v", accuracy)
	}
	
	// Test probability predictions
	probas, err := lr.PredictProba(X)
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

// TestLogisticRegression_GetSetParams tests parameter management
func TestLogisticRegression_GetSetParams(t *testing.T) {
	lr := NewLogisticRegression()
	
	// Get default params
	params := lr.GetParams()
	
	// Check some defaults
	if params["C"].(float64) != 1.0 {
		t.Errorf("Default C should be 1.0, got %v", params["C"])
	}
	
	if params["max_iter"].(int) != 100 {
		t.Errorf("Default max_iter should be 100, got %v", params["max_iter"])
	}
	
	// Set new params
	newParams := map[string]interface{}{
		"C":         2.0,
		"max_iter":  200,
		"penalty":   "l1",
		"tol":       1e-5,
	}
	
	err := lr.SetParams(newParams)
	if err != nil {
		t.Fatalf("Failed to set params: %v", err)
	}
	
	// Verify changes
	if lr.C != 2.0 {
		t.Errorf("C not updated: expected 2.0, got %v", lr.C)
	}
	
	if lr.maxIter != 200 {
		t.Errorf("max_iter not updated: expected 200, got %v", lr.maxIter)
	}
	
	if lr.penalty != "l1" {
		t.Errorf("penalty not updated: expected 'l1', got %v", lr.penalty)
	}
	
	if lr.tol != 1e-5 {
		t.Errorf("tol not updated: expected 1e-5, got %v", lr.tol)
	}
}

// TestLogisticRegression_NotFitted tests error when predicting without fitting
func TestLogisticRegression_NotFitted(t *testing.T) {
	lr := NewLogisticRegression()
	
	X := mat.NewDense(2, 2, []float64{
		1, 2,
		3, 4,
	})
	
	_, err := lr.Predict(X)
	if err == nil {
		t.Error("Expected error when predicting without fitting")
	}
	
	_, err = lr.PredictProba(X)
	if err == nil {
		t.Error("Expected error when predicting probabilities without fitting")
	}
}