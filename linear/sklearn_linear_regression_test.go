package linear

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSKLinearRegression_Basic(t *testing.T) {
	// Test basic linear regression y = 2x + 1
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := mat.NewDense(4, 1, []float64{3, 5, 7, 9})

	lr := NewSKLinearRegression()
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Check coefficients
	if lr.Coef_.At(0, 0) < 1.99 || lr.Coef_.At(0, 0) > 2.01 {
		t.Errorf("Expected coefficient ~2.0, got %f", lr.Coef_.At(0, 0))
	}

	// Check intercept
	if lr.Intercept_.At(0, 0) < 0.99 || lr.Intercept_.At(0, 0) > 1.01 {
		t.Errorf("Expected intercept ~1.0, got %f", lr.Intercept_.At(0, 0))
	}

	// Test prediction
	XTest := mat.NewDense(2, 1, []float64{5, 6})
	pred, err := lr.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	expected := []float64{11, 13}
	for i := 0; i < 2; i++ {
		if math.Abs(pred.At(i, 0)-expected[i]) > 0.01 {
			t.Errorf("Expected prediction %f, got %f", expected[i], pred.At(i, 0))
		}
	}
}

func TestSKLinearRegression_NoIntercept(t *testing.T) {
	// Test without intercept: y = 2x
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := mat.NewDense(4, 1, []float64{2, 4, 6, 8})

	lr := NewSKLinearRegression(WithFitIntercept(false))
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Check coefficient
	if lr.Coef_.At(0, 0) < 1.99 || lr.Coef_.At(0, 0) > 2.01 {
		t.Errorf("Expected coefficient ~2.0, got %f", lr.Coef_.At(0, 0))
	}

	// Check intercept is zero
	if lr.Intercept_.At(0, 0) != 0 {
		t.Errorf("Expected intercept 0, got %f", lr.Intercept_.At(0, 0))
	}
}

func TestSKLinearRegression_MultipleFeatures(t *testing.T) {
	// Test with multiple features: y = 2*x1 + 3*x2 + 1
	X := mat.NewDense(5, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
	})
	y := mat.NewDense(5, 1, []float64{6, 8, 13, 15, 20})

	lr := NewSKLinearRegression()
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Check coefficients
	if lr.Coef_.At(0, 0) < 1.9 || lr.Coef_.At(0, 0) > 2.1 {
		t.Errorf("Expected first coefficient ~2.0, got %f", lr.Coef_.At(0, 0))
	}
	if lr.Coef_.At(1, 0) < 2.9 || lr.Coef_.At(1, 0) > 3.1 {
		t.Errorf("Expected second coefficient ~3.0, got %f", lr.Coef_.At(1, 0))
	}
}

func TestSKLinearRegression_MultipleTargets(t *testing.T) {
	// Test with multiple targets
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := mat.NewDense(4, 2, []float64{
		3, 2,  // y1 = 2x + 1, y2 = x + 1
		5, 3,
		7, 4,
		9, 5,
	})

	lr := NewSKLinearRegression()
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Check coefficients for both targets
	if lr.Coef_.At(0, 0) < 1.99 || lr.Coef_.At(0, 0) > 2.01 {
		t.Errorf("Expected first target coefficient ~2.0, got %f", lr.Coef_.At(0, 0))
	}
	if lr.Coef_.At(0, 1) < 0.99 || lr.Coef_.At(0, 1) > 1.01 {
		t.Errorf("Expected second target coefficient ~1.0, got %f", lr.Coef_.At(0, 1))
	}

	// Test prediction
	XTest := mat.NewDense(2, 1, []float64{5, 6})
	pred, err := lr.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check predictions for both targets
	expected := [][]float64{
		{11, 6},  // x=5: y1=11, y2=6
		{13, 7},  // x=6: y1=13, y2=7
	}
	
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(pred.At(i, j)-expected[i][j]) > 0.01 {
				t.Errorf("Expected prediction[%d,%d] = %f, got %f", 
					i, j, expected[i][j], pred.At(i, j))
			}
		}
	}
}

func TestSKLinearRegression_Score(t *testing.T) {
	// Perfect fit case
	X := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	y := mat.NewDense(5, 1, []float64{2, 4, 6, 8, 10})

	lr := NewSKLinearRegression()
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to compute score: %v", err)
	}

	// Should be close to 1.0 for perfect fit
	if score < 0.999 {
		t.Errorf("Expected score ~1.0, got %f", score)
	}
}

func TestSKLinearRegression_Options(t *testing.T) {
	// Test with various options
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := mat.NewDense(4, 1, []float64{2, 4, 6, 8})

	lr := NewSKLinearRegression(
		WithFitIntercept(false),
		WithCopyX(false),
		WithTol(1e-8),
		WithNJobs(2),
	)
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Check that options were applied
	params := lr.GetParams()
	
	if params["fit_intercept"].(bool) != false {
		t.Error("Expected fit_intercept to be false")
	}
	if params["copy_X"].(bool) != false {
		t.Error("Expected copy_X to be false")
	}
	if params["tol"].(float64) != 1e-8 {
		t.Error("Expected tol to be 1e-8")
	}
	if params["n_jobs"].(int) != 2 {
		t.Error("Expected n_jobs to be 2")
	}
}

func TestSKLinearRegression_NotFitted(t *testing.T) {
	lr := NewSKLinearRegression()
	
	X := mat.NewDense(2, 1, []float64{1, 2})
	
	_, err := lr.Predict(X)
	if err == nil {
		t.Error("Expected error when predicting with unfitted model")
	}
}

func TestSKLinearRegression_SetParams(t *testing.T) {
	lr := NewSKLinearRegression()
	
	params := map[string]interface{}{
		"fit_intercept": false,
		"tol":           1e-10,
		"n_jobs":        4,
	}
	
	if err := lr.SetParams(params); err != nil {
		t.Fatalf("Failed to set params: %v", err)
	}
	
	newParams := lr.GetParams()
	
	if newParams["fit_intercept"].(bool) != false {
		t.Error("fit_intercept not updated")
	}
	if newParams["tol"].(float64) != 1e-10 {
		t.Error("tol not updated")
	}
	if newParams["n_jobs"].(int) != 4 {
		t.Error("n_jobs not updated")
	}
}

func TestSKLinearRegression_Positive(t *testing.T) {
	// Test with positive constraint
	// This test uses a simple case where we know the positive solution
	X := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	y := mat.NewDense(3, 1, []float64{1, 2, 3})

	lr := NewSKLinearRegression(
		WithPositive(true),
		WithFitIntercept(false),
	)
	
	if err := lr.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit with positive constraint: %v", err)
	}

	// Check that coefficients are non-negative
	rows, cols := lr.Coef_.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if lr.Coef_.At(i, j) < 0 {
				t.Errorf("Expected positive coefficient, got %f at [%d,%d]", 
					lr.Coef_.At(i, j), i, j)
			}
		}
	}
}

func BenchmarkSKLinearRegression_Fit(b *testing.B) {
	// Benchmark with different sizes
	sizes := []struct {
		name string
		n    int
		p    int
	}{
		{"Small_100x10", 100, 10},
		{"Medium_1000x20", 1000, 20},
		{"Large_5000x50", 5000, 50},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Generate random data
			X := mat.NewDense(size.n, size.p, nil)
			y := mat.NewDense(size.n, 1, nil)
			
			for i := 0; i < size.n; i++ {
				for j := 0; j < size.p; j++ {
					X.Set(i, j, float64(i*j))
				}
				y.Set(i, 0, float64(i))
			}

			lr := NewSKLinearRegression()
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = lr.Fit(X, y)
			}
		})
	}
}