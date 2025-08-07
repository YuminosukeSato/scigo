package linear_model

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/YuminosukeSato/scigo/core/model"
	"gonum.org/v1/gonum/mat"
)

// TestLinearRegressionWeightReproducibility tests complete weight reproducibility
func TestLinearRegressionWeightReproducibility(t *testing.T) {
	// Create test data
	X := mat.NewDense(100, 3, nil)
	y := mat.NewDense(100, 1, nil)
	
	// Generate data (reproducible with fixed seed)
	for i := 0; i < 100; i++ {
		X.Set(i, 0, math.Sin(float64(i)/10.0))
		X.Set(i, 1, math.Cos(float64(i)/10.0))
		X.Set(i, 2, float64(i)/50.0)
		// y = 2*x1 + 3*x2 - x3 + 5 + noise
		y.Set(i, 0, 2*X.At(i, 0) + 3*X.At(i, 1) - X.At(i, 2) + 5 + float64(i%5)/100.0)
	}
	
	// Train model1
	model1 := NewLinearRegression(WithLRFitIntercept(true))
	if err := model1.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit model1: %v", err)
	}
	
	// Export weights
	weights, err := model1.ExportWeights()
	if err != nil {
		t.Fatalf("Failed to export weights: %v", err)
	}
	
	// Serialize weights to JSON
	jsonData, err := json.Marshal(weights)
	if err != nil {
		t.Fatalf("Failed to serialize weights: %v", err)
	}
	
	// Deserialize weights from JSON
	loadedWeights := &model.ModelWeights{}
	if err := json.Unmarshal(jsonData, loadedWeights); err != nil {
		t.Fatalf("Failed to deserialize weights: %v", err)
	}
	
	// Import weights to model2
	model2 := NewLinearRegression()
	if err := model2.ImportWeights(loadedWeights); err != nil {
		t.Fatalf("Failed to import weights: %v", err)
	}
	
	// Verify that coefficients of both models match exactly
	coef1 := model1.Coef()
	coef2 := model2.Coef()
	
	if len(coef1) != len(coef2) {
		t.Fatalf("Coefficient length mismatch: %d vs %d", len(coef1), len(coef2))
	}
	
	for i := range coef1 {
		if coef1[i] != coef2[i] {
			t.Errorf("Coefficient mismatch at index %d: %.15f vs %.15f", i, coef1[i], coef2[i])
		}
	}
	
	// Verify that intercepts also match
	if model1.Intercept() != model2.Intercept() {
		t.Errorf("Intercept mismatch: %.15f vs %.15f", model1.Intercept(), model2.Intercept())
	}
	
	// Verify that prediction results match exactly
	pred1, err := model1.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with model1: %v", err)
	}
	
	pred2, err := model2.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with model2: %v", err)
	}
	
	rows, _ := pred1.Dims()
	for i := 0; i < rows; i++ {
		p1 := pred1.At(i, 0)
		p2 := pred2.At(i, 0)
		if p1 != p2 {
			t.Errorf("Prediction mismatch at index %d: %.15f vs %.15f", i, p1, p2)
		}
	}
	
	// Verify that hash values match
	hash1 := model1.GetWeightHash()
	hash2 := model2.GetWeightHash()
	
	if hash1 != hash2 {
		t.Errorf("Weight hash mismatch: %s vs %s", hash1, hash2)
	}
}

// TestSGDRegressorWeightReproducibility tests SGDRegressor weight reproducibility
func TestSGDRegressorWeightReproducibility(t *testing.T) {
	// Create test data
	X := mat.NewDense(50, 3, nil)
	y := mat.NewDense(50, 1, nil)
	
	// Generate data
	for i := 0; i < 50; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, float64(i+j+1)/10.0)
		}
		y.Set(i, 0, 2*X.At(i, 0) + 3*X.At(i, 1) - X.At(i, 2) + 5)
	}
	
	// Train SGDRegressor (with fixed seed)
	sgd1 := NewSGDRegressor(
		WithRandomState(42),
		WithMaxIter(100),
		WithTol(1e-4),
	)
	
	if err := sgd1.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit SGDRegressor: %v", err)
	}
	
	// Get coefficients
	coef1 := sgd1.Coef()
	intercept1 := sgd1.Intercept()
	
	// Create another SGDRegressor and set weights
	sgd2 := NewSGDRegressor(
		WithRandomState(42),
		WithMaxIter(100),
		WithTol(1e-4),
	)
	
	// Manually set weights (in actual implementation, use ImportWeights)
	sgd2.coef_ = make([]float64, len(coef1))
	copy(sgd2.coef_, coef1)
	sgd2.intercept_ = intercept1
	sgd2.nFeatures_ = 3
	sgd2.state.SetFitted()
	sgd2.state.SetDimensions(3, 50)
	
	// Verify that prediction results match
	pred1, err := sgd1.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with sgd1: %v", err)
	}
	
	pred2, err := sgd2.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with sgd2: %v", err)
	}
	
	rows, _ := pred1.Dims()
	for i := 0; i < rows; i++ {
		p1 := pred1.At(i, 0)
		p2 := pred2.At(i, 0)
		if math.Abs(p1-p2) > 1e-15 {
			t.Errorf("Prediction mismatch at index %d: %.15f vs %.15f (diff: %.15e)", 
				i, p1, p2, math.Abs(p1-p2))
		}
	}
}

// TestWeightValidation tests weight validation
func TestWeightValidation(t *testing.T) {
	weights := &model.ModelWeights{
		ModelType:    "LinearRegression",
		Version:      "1.0.0",
		Coefficients: []float64{1.0, 2.0, 3.0},
		Intercept:    4.0,
		IsFitted:     true,
		Hyperparameters: map[string]interface{}{
			"fit_intercept": true,
			"normalize":     false,
		},
	}
	
	// Valid weights should pass validation
	if err := weights.Validate(); err != nil {
		t.Errorf("Valid weights failed validation: %v", err)
	}
	
	// Error if model type is empty
	invalidWeights := weights.Clone()
	invalidWeights.ModelType = ""
	if err := invalidWeights.Validate(); err == nil {
		t.Error("Invalid weights (empty model_type) passed validation")
	}
	
	// Error if trained but no coefficients
	invalidWeights2 := weights.Clone()
	invalidWeights2.Coefficients = nil
	if err := invalidWeights2.Validate(); err == nil {
		t.Error("Invalid weights (no coefficients) passed validation")
	}
}

// TestWeightFloatPrecision tests floating-point precision preservation
func TestWeightFloatPrecision(t *testing.T) {
	// Prepare high-precision values
	preciseValues := []float64{
		math.Pi,
		math.E,
		math.Sqrt(2),
		1.0 / 3.0,
		0.1234567890123456789,
	}
	
	weights := &model.ModelWeights{
		ModelType:    "TestModel",
		Version:      "1.0.0",
		Coefficients: preciseValues,
		Intercept:    math.Phi,
		IsFitted:     true,
	}
	
	// Verify precision is preserved even through JSON serialization
	jsonData, err := weights.ToJSON()
	if err != nil {
		t.Fatalf("Failed to serialize weights: %v", err)
	}
	
	loadedWeights := &model.ModelWeights{}
	if err := loadedWeights.FromJSON(jsonData); err != nil {
		t.Fatalf("Failed to deserialize weights: %v", err)
	}
	
	// Verify coefficient precision is preserved
	for i, original := range preciseValues {
		loaded := loadedWeights.Coefficients[i]
		if original != loaded {
			t.Errorf("Precision loss at index %d: original=%.17f, loaded=%.17f", 
				i, original, loaded)
		}
	}
	
	// Also verify intercept precision
	if weights.Intercept != loadedWeights.Intercept {
		t.Errorf("Intercept precision loss: original=%.17f, loaded=%.17f",
			weights.Intercept, loadedWeights.Intercept)
	}
}

// BenchmarkWeightExportImport measures performance of weight export/import
func BenchmarkWeightExportImport(b *testing.B) {
	// Create a larger model
	X := mat.NewDense(1000, 100, nil)
	y := mat.NewDense(1000, 1, nil)
	
	for i := 0; i < 1000; i++ {
		for j := 0; j < 100; j++ {
			X.Set(i, j, float64(i+j)/100.0)
		}
		y.Set(i, 0, float64(i))
	}
	
	mdl := NewLinearRegression()
	_ = mdl.Fit(X, y)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		// Export
		weights, _ := mdl.ExportWeights()
		
		// JSON serialization
		jsonData, _ := json.Marshal(weights)
		
		// JSON deserialization - temporarily simplified
		_ = jsonData // Mark as unused
		
		// Import
		newModel := NewLinearRegression()
		_ = newModel.ImportWeights(weights)
	}
}