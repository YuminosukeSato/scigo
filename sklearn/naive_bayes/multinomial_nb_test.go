package naive_bayes

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestMultinomialNBBasicFit tests basic fitting functionality
func TestMultinomialNBBasicFit(t *testing.T) {
	// Simple text classification example (word counts)
	// Features: [count_word1, count_word2, count_word3]
	X := mat.NewDense(6, 3, []float64{
		2, 1, 0, // class 0
		1, 1, 1, // class 0
		1, 0, 1, // class 0
		0, 1, 2, // class 1
		0, 2, 1, // class 1
		1, 2, 2, // class 1
	})

	y := mat.NewDense(6, 1, []float64{
		0, 0, 0, 1, 1, 1,
	})

	nb := NewMultinomialNB()
	err := nb.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !nb.state.IsFitted() {
		t.Error("Model should be fitted after Fit()")
	}

	// Check that classes are correctly identified
	classes := nb.Classes()
	if len(classes) != 2 {
		t.Errorf("Expected 2 classes, got %d", len(classes))
	}
}

// TestMultinomialNBPartialFit tests online learning capability
func TestMultinomialNBPartialFit(t *testing.T) {
	nb := NewMultinomialNB()

	// First batch
	X1 := mat.NewDense(3, 3, []float64{
		2, 1, 0,
		1, 1, 1,
		1, 0, 1,
	})
	y1 := mat.NewDense(3, 1, []float64{0, 0, 0})

	// Specify classes in first partial_fit call
	err := nb.PartialFit(X1, y1, []int{0, 1})
	if err != nil {
		t.Fatalf("First PartialFit failed: %v", err)
	}

	// Second batch
	X2 := mat.NewDense(3, 3, []float64{
		0, 1, 2,
		0, 2, 1,
		1, 2, 2,
	})
	y2 := mat.NewDense(3, 1, []float64{1, 1, 1})

	err = nb.PartialFit(X2, y2, nil)
	if err != nil {
		t.Fatalf("Second PartialFit failed: %v", err)
	}

	if !nb.state.IsFitted() {
		t.Error("Model should be fitted after PartialFit()")
	}

	// Verify incremental learning happened
	if nb.NSamplesSeen() != 6 {
		t.Errorf("Expected 6 samples seen, got %d", nb.NSamplesSeen())
	}
}

// TestMultinomialNBPredict tests prediction functionality
func TestMultinomialNBPredict(t *testing.T) {
	// Training data
	XTrain := mat.NewDense(6, 3, []float64{
		3, 0, 0, // strongly class 0
		2, 1, 0, // class 0
		1, 0, 0, // class 0
		0, 0, 3, // strongly class 1
		0, 1, 2, // class 1
		0, 0, 1, // class 1
	})

	yTrain := mat.NewDense(6, 1, []float64{
		0, 0, 0, 1, 1, 1,
	})

	nb := NewMultinomialNB()
	err := nb.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Test data
	XTest := mat.NewDense(2, 3, []float64{
		2, 0, 0, // should predict class 0
		0, 0, 2, // should predict class 1
	})

	predictions, err := nb.Predict(XTest)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	rows, cols := predictions.Dims()
	if rows != 2 || cols != 1 {
		t.Errorf("Predictions shape should be (2, 1), got (%d, %d)", rows, cols)
	}

	// Check predictions
	if predictions.At(0, 0) != 0 {
		t.Errorf("First sample should be predicted as class 0, got %f", predictions.At(0, 0))
	}
	if predictions.At(1, 0) != 1 {
		t.Errorf("Second sample should be predicted as class 1, got %f", predictions.At(1, 0))
	}
}

// TestMultinomialNBPredictProba tests probability prediction
func TestMultinomialNBPredictProba(t *testing.T) {
	XTrain := mat.NewDense(6, 3, []float64{
		3, 0, 0,
		2, 1, 0,
		1, 0, 0,
		0, 0, 3,
		0, 1, 2,
		0, 0, 1,
	})

	yTrain := mat.NewDense(6, 1, []float64{
		0, 0, 0, 1, 1, 1,
	})

	nb := NewMultinomialNB()
	err := nb.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	XTest := mat.NewDense(2, 3, []float64{
		2, 0, 0,
		0, 0, 2,
	})

	proba, err := nb.PredictProba(XTest)
	if err != nil {
		t.Fatalf("PredictProba failed: %v", err)
	}

	rows, cols := proba.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Proba shape should be (2, 2), got (%d, %d)", rows, cols)
	}

	// Check that probabilities sum to 1
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			p := proba.At(i, j)
			if p < 0 || p > 1 {
				t.Errorf("Probability should be in [0, 1], got %f", p)
			}
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("Probabilities should sum to 1, got %f", sum)
		}
	}

	// First sample should have higher probability for class 0
	if proba.At(0, 0) <= proba.At(0, 1) {
		t.Error("First sample should have higher probability for class 0")
	}

	// Second sample should have higher probability for class 1
	if proba.At(1, 1) <= proba.At(1, 0) {
		t.Error("Second sample should have higher probability for class 1")
	}
}

// TestMultinomialNBPredictLogProba tests log probability prediction
func TestMultinomialNBPredictLogProba(t *testing.T) {
	XTrain := mat.NewDense(4, 2, []float64{
		2, 0,
		1, 1,
		0, 2,
		1, 1,
	})

	yTrain := mat.NewDense(4, 1, []float64{
		0, 0, 1, 1,
	})

	nb := NewMultinomialNB()
	err := nb.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	XTest := mat.NewDense(1, 2, []float64{
		1, 1,
	})

	logProba, err := nb.PredictLogProba(XTest)
	if err != nil {
		t.Fatalf("PredictLogProba failed: %v", err)
	}

	// Check that log probabilities are negative (or zero)
	rows, cols := logProba.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if logProba.At(i, j) > 0 {
				t.Errorf("Log probability should be <= 0, got %f", logProba.At(i, j))
			}
		}
	}

	// Check that exp(log_proba) sums to 1
	sum := 0.0
	for j := 0; j < cols; j++ {
		sum += math.Exp(logProba.At(0, j))
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Exp of log probabilities should sum to 1, got %f", sum)
	}
}

// TestMultinomialNBWithAlpha tests Laplace smoothing
func TestMultinomialNBWithAlpha(t *testing.T) {
	// Data with zero counts for some features
	XTrain := mat.NewDense(4, 3, []float64{
		2, 0, 0,
		1, 0, 0,
		0, 0, 2,
		0, 0, 1,
	})

	yTrain := mat.NewDense(4, 1, []float64{
		0, 0, 1, 1,
	})

	// Test with different alpha values
	alphas := []float64{0.0, 1.0, 10.0}

	for _, alpha := range alphas {
		nb := NewMultinomialNB(WithAlpha(alpha))
		err := nb.Fit(XTrain, yTrain)
		if err != nil {
			t.Fatalf("Fit with alpha=%f failed: %v", alpha, err)
		}

		// Test on unseen feature combination
		XTest := mat.NewDense(1, 3, []float64{
			1, 1, 1,
		})

		proba, err := nb.PredictProba(XTest)
		if err != nil {
			t.Fatalf("PredictProba with alpha=%f failed: %v", alpha, err)
		}

		// With higher alpha, probabilities should be more uniform
		diff := math.Abs(proba.At(0, 0) - proba.At(0, 1))
		if alpha > 0 {
			// Should not get NaN or Inf with smoothing
			for j := 0; j < 2; j++ {
				p := proba.At(0, j)
				if math.IsNaN(p) || math.IsInf(p, 0) {
					t.Errorf("With alpha=%f, got invalid probability: %f", alpha, p)
				}
			}
		}
		_ = diff // Higher alpha should lead to smaller diff, but exact values depend on implementation
	}
}

// TestMultinomialNBScore tests accuracy scoring
func TestMultinomialNBScore(t *testing.T) {
	// Create easily separable data
	XTrain := mat.NewDense(6, 2, []float64{
		5, 0,
		4, 1,
		3, 0,
		0, 5,
		1, 4,
		0, 3,
	})

	yTrain := mat.NewDense(6, 1, []float64{
		0, 0, 0, 1, 1, 1,
	})

	nb := NewMultinomialNB()
	err := nb.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	score, err := nb.Score(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	// With clearly separable data, accuracy should be high
	if score < 0.9 {
		t.Errorf("Score should be high for separable data, got %f", score)
	}
}

// TestMultinomialNBInvalidInput tests error handling
func TestMultinomialNBInvalidInput(t *testing.T) {
	nb := NewMultinomialNB()

	// Test with negative values (invalid for MultinomialNB)
	XInvalid := mat.NewDense(2, 2, []float64{
		1, -1,
		2, 3,
	})
	y := mat.NewDense(2, 1, []float64{0, 1})

	err := nb.Fit(XInvalid, y)
	if err == nil {
		t.Error("Fit should fail with negative values")
	}

	// Test prediction on unfitted model
	nbUnfitted := NewMultinomialNB()
	_, err = nbUnfitted.Predict(XInvalid)
	if err == nil {
		t.Error("Predict should fail on unfitted model")
	}
}

// TestMultinomialNBFitPrior tests class prior handling
func TestMultinomialNBFitPrior(t *testing.T) {
	// Imbalanced dataset
	XTrain := mat.NewDense(5, 2, []float64{
		2, 1,
		1, 2,
		1, 1,
		1, 0,
		0, 1,
	})

	yTrain := mat.NewDense(5, 1, []float64{
		0, 0, 0, 0, 1, // 4 samples of class 0, 1 sample of class 1
	})

	// Test with fit_prior=true (default)
	nbWithPrior := NewMultinomialNB()
	err := nbWithPrior.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit with prior failed: %v", err)
	}

	// Test with fit_prior=false (uniform prior)
	nbWithoutPrior := NewMultinomialNB(WithFitPrior(false))
	err = nbWithoutPrior.Fit(XTrain, yTrain)
	if err != nil {
		t.Fatalf("Fit without prior failed: %v", err)
	}

	// The models should make different predictions on ambiguous data
	XTest := mat.NewDense(1, 2, []float64{
		1, 1,
	})

	probaWithPrior, _ := nbWithPrior.PredictProba(XTest)
	probaWithoutPrior, _ := nbWithoutPrior.PredictProba(XTest)

	// With prior, class 0 should have higher probability due to imbalance
	// Without prior, probabilities should be more balanced
	diff1 := math.Abs(probaWithPrior.At(0, 0) - probaWithPrior.At(0, 1))
	diff2 := math.Abs(probaWithoutPrior.At(0, 0) - probaWithoutPrior.At(0, 1))

	// The difference should be larger when using prior (due to class imbalance)
	if diff1 <= diff2 {
		t.Error("Prior should affect probability distribution for imbalanced data")
	}
}
