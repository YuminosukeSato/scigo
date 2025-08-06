package lightgbm

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestTrainerBasic tests basic training functionality
func TestTrainerBasic(t *testing.T) {
	// Create simple regression data
	// y = 2*x1 + 3*x2 + noise
	X := mat.NewDense(100, 2, nil)
	y := mat.NewDense(100, 1, nil)
	
	for i := 0; i < 100; i++ {
		x1 := float64(i) / 10.0
		x2 := float64(i%10) / 5.0
		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		y.Set(i, 0, 2*x1 + 3*x2 + 0.1*(float64(i%3)-1))
	}
	
	// Create trainer
	params := TrainingParams{
		NumIterations:  10,
		LearningRate:   0.1,
		NumLeaves:      31,
		MaxDepth:       5,
		MinDataInLeaf:  5,
		Lambda:         1.0,
		Objective:      "regression",
		Verbosity:      0,
	}
	
	trainer := NewTrainer(params)
	
	// Train model
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	
	// Check that we have trees
	if len(trainer.trees) == 0 {
		t.Error("No trees were built")
	}
	
	// Get model
	model := trainer.GetModel()
	if model == nil {
		t.Fatal("GetModel returned nil")
	}
	
	if model.NumIteration != len(trainer.trees) {
		t.Errorf("Model iteration count mismatch: got %d, want %d", 
			model.NumIteration, len(trainer.trees))
	}
}

// TestClassifierTraining tests classifier training
func TestClassifierTraining(t *testing.T) {
	// Create simple binary classification data
	X := mat.NewDense(50, 2, nil)
	y := mat.NewDense(50, 1, nil)
	
	// Class 0: x1 < 0.5
	for i := 0; i < 25; i++ {
		X.Set(i, 0, float64(i)/50.0)
		X.Set(i, 1, float64(i%5)/5.0)
		y.Set(i, 0, 0)
	}
	
	// Class 1: x1 >= 0.5
	for i := 25; i < 50; i++ {
		X.Set(i, 0, 0.5+float64(i-25)/50.0)
		X.Set(i, 1, float64(i%5)/5.0)
		y.Set(i, 0, 1)
	}
	
	// Train classifier
	clf := NewLGBMClassifier()
	clf.NumIterations = 5
	clf.LearningRate = 0.1
	clf.Verbosity = -1
	
	err := clf.Fit(X, y)
	if err != nil {
		t.Fatalf("Classifier training failed: %v", err)
	}
	
	// Make predictions
	predictions, err := clf.Predict(X)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}
	
	// Check accuracy
	correct := 0
	for i := 0; i < 50; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if pred == actual {
			correct++
		}
	}
	
	accuracy := float64(correct) / 50.0
	if accuracy < 0.8 {
		t.Errorf("Accuracy too low: %.2f", accuracy)
	}
}

// TestRegressorTraining tests regressor training
func TestRegressorTraining(t *testing.T) {
	// Create regression data
	X := mat.NewDense(100, 3, nil)
	y := mat.NewDense(100, 1, nil)
	
	// y = x1 + 2*x2 - x3 + noise
	for i := 0; i < 100; i++ {
		x1 := float64(i) / 100.0
		x2 := float64(i%10) / 10.0
		x3 := float64(i%5) / 5.0
		
		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		X.Set(i, 2, x3)
		
		yVal := x1 + 2*x2 - x3 + 0.01*float64(i%3-1)
		y.Set(i, 0, yVal)
	}
	
	// Train regressor
	reg := NewLGBMRegressor()
	reg.NumIterations = 10
	reg.LearningRate = 0.1
	reg.NumLeaves = 15
	reg.Verbosity = -1
	
	err := reg.Fit(X, y)
	if err != nil {
		t.Fatalf("Regressor training failed: %v", err)
	}
	
	// Make predictions
	predictions, err := reg.Predict(X)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}
	
	// Calculate MSE
	mse := 0.0
	for i := 0; i < 100; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		diff := pred - actual
		mse += diff * diff
	}
	mse /= 100.0
	
	if mse > 0.1 {
		t.Errorf("MSE too high: %.4f", mse)
	}
	
	// Test score method
	score, err := reg.Score(X, y)
	if err != nil {
		t.Fatalf("Score calculation failed: %v", err)
	}
	
	// R² should be positive for reasonable fit
	if score < 0 {
		t.Errorf("R² score negative: %.4f", score)
	}
}

// TestSplitGainCalculation tests the split gain calculation
func TestSplitGainCalculation(t *testing.T) {
	trainer := &Trainer{
		params: TrainingParams{
			Lambda: 1.0,
		},
	}
	
	// Test case 1: Perfect split
	gain := trainer.calculateSplitGain(
		-10.0, 5.0,  // left gradient and hessian
		10.0, 5.0,   // right gradient and hessian
		0.0, 10.0,   // total gradient and hessian
	)
	
	if gain <= 0 {
		t.Errorf("Expected positive gain for good split, got %.4f", gain)
	}
	
	// Test case 2: Bad split (no gain)
	gain = trainer.calculateSplitGain(
		0.0, 5.0,
		0.0, 5.0,
		0.0, 10.0,
	)
	
	if gain != 0 {
		t.Errorf("Expected zero gain for neutral split, got %.4f", gain)
	}
}

// TestLeafValueCalculation tests leaf value calculation
func TestLeafValueCalculation(t *testing.T) {
	trainer := &Trainer{
		params: TrainingParams{
			Lambda: 1.0,
		},
		gradients: []float64{-1.0, -2.0, -3.0, 1.0, 2.0},
		hessians:  []float64{1.0, 1.0, 1.0, 1.0, 1.0},
	}
	
	indices := []int{0, 1, 2} // First three samples
	leafValue := trainer.calculateLeafValue(indices)
	
	// Sum of gradients = -6, sum of hessians = 3
	// Expected: -(-6)/(3+1) = 1.5
	expected := 1.5
	if math.Abs(leafValue-expected) > 1e-6 {
		t.Errorf("Leaf value calculation wrong: got %.4f, want %.4f", leafValue, expected)
	}
}

// TestHistogramBuilding tests histogram construction
func TestHistogramBuilding(t *testing.T) {
	X := mat.NewDense(10, 2, []float64{
		1.0, 5.0,
		2.0, 4.0,
		3.0, 3.0,
		4.0, 2.0,
		5.0, 1.0,
		6.0, 6.0,
		7.0, 7.0,
		8.0, 8.0,
		9.0, 9.0,
		10.0, 10.0,
	})
	
	trainer := &Trainer{
		params: TrainingParams{
			MaxBin: 5,
		},
		X: X,
	}
	
	err := trainer.buildHistograms()
	if err != nil {
		t.Fatalf("Histogram building failed: %v", err)
	}
	
	// Check that histograms were created
	if len(trainer.histograms) != 2 {
		t.Errorf("Expected 2 feature histograms, got %d", len(trainer.histograms))
	}
	
	// Check bin count
	for i, hist := range trainer.histograms {
		if len(hist) == 0 {
			t.Errorf("Feature %d has empty histogram", i)
		}
		if len(hist) > trainer.params.MaxBin {
			t.Errorf("Feature %d has too many bins: %d > %d", i, len(hist), trainer.params.MaxBin)
		}
	}
}

// BenchmarkTraining benchmarks the training speed
func BenchmarkTraining(b *testing.B) {
	// Create larger dataset
	n := 1000
	X := mat.NewDense(n, 10, nil)
	y := mat.NewDense(n, 1, nil)
	
	for i := 0; i < n; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i*j)/float64(n))
		}
		y.Set(i, 0, float64(i%2))
	}
	
	params := TrainingParams{
		NumIterations: 10,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		Verbosity:     -1,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer := NewTrainer(params)
		_ = trainer.Fit(X, y)
	}
}

// TestEarlyStopping tests early stopping functionality
func TestEarlyStopping(t *testing.T) {
	// Create data where model should converge quickly
	X := mat.NewDense(50, 1, nil)
	y := mat.NewDense(50, 1, nil)
	
	for i := 0; i < 50; i++ {
		X.Set(i, 0, float64(i))
		y.Set(i, 0, float64(i)) // Perfect linear relationship
	}
	
	params := TrainingParams{
		NumIterations:  100, // Set high
		LearningRate:   0.3,
		NumLeaves:      5,
		EarlyStopping:  5,
		Verbosity:     -1,
	}
	
	trainer := NewTrainer(params)
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Training with early stopping failed: %v", err)
	}
	
	// Should stop before max iterations
	if len(trainer.trees) >= params.NumIterations {
		t.Errorf("Early stopping didn't work: built %d trees", len(trainer.trees))
	}
}