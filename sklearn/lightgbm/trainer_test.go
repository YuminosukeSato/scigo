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
		y.Set(i, 0, 2*x1+3*x2+0.1*(float64(i%3)-1))
	}

	// Create trainer
	params := TrainingParams{
		NumIterations: 10,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		MinDataInLeaf: 5,
		Lambda:        1.0,
		Objective:     "regression",
		Verbosity:     0,
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

// TestTrainer_Fit_BinaryClassification tests binary classification training
func TestTrainer_Fit_BinaryClassification(t *testing.T) {
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

	// Create trainer with binary objective
	params := TrainingParams{
		NumIterations: 5,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		MinDataInLeaf: 5,
		Lambda:        1.0,
		Objective:     "binary",
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)

	// Train model - this should fail initially
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Binary classification training failed: %v", err)
	}

	// Check that we have trees
	if len(trainer.trees) == 0 {
		t.Error("No trees were built for binary classification")
	}

	// Get model
	model := trainer.GetModel()
	if model == nil {
		t.Fatal("GetModel returned nil")
	}

	// Check objective is set correctly
	if string(model.Objective) != "binary" {
		t.Errorf("Model objective incorrect: got %s, want binary", model.Objective)
	}
}

// TestTrainer_Fit_MultiClassClassification tests multiclass classification training
func TestTrainer_Fit_MultiClassClassification(t *testing.T) {
	// Create simple 3-class classification data
	X := mat.NewDense(60, 2, nil)
	y := mat.NewDense(60, 1, nil)

	// Class 0: x1 < 0.33
	for i := 0; i < 20; i++ {
		X.Set(i, 0, float64(i)/60.0)
		X.Set(i, 1, float64(i%5)/5.0)
		y.Set(i, 0, 0)
	}

	// Class 1: 0.33 <= x1 < 0.66
	for i := 20; i < 40; i++ {
		X.Set(i, 0, 0.33+float64(i-20)/60.0)
		X.Set(i, 1, float64(i%5)/5.0)
		y.Set(i, 0, 1)
	}

	// Class 2: x1 >= 0.66
	for i := 40; i < 60; i++ {
		X.Set(i, 0, 0.66+float64(i-40)/60.0)
		X.Set(i, 1, float64(i%5)/5.0)
		y.Set(i, 0, 2)
	}

	// Create trainer with multiclass objective
	params := TrainingParams{
		NumIterations: 5,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		MinDataInLeaf: 5,
		Lambda:        1.0,
		Objective:     "multiclass",
		NumClass:      3,
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)

	// Train model - this should fail initially
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Multiclass classification training failed: %v", err)
	}

	// Check that we have trees
	if len(trainer.trees) == 0 {
		t.Error("No trees were built for multiclass classification")
	}

	// Get model
	model := trainer.GetModel()
	if model == nil {
		t.Fatal("GetModel returned nil")
	}

	// Check objective and num_class are set correctly
	if string(model.Objective) != "multiclass" {
		t.Errorf("Model objective incorrect: got %s, want multiclass", model.Objective)
	}
	if model.NumClass != 3 {
		t.Errorf("Model NumClass incorrect: got %d, want 3", model.NumClass)
	}
}

// TestRegressorTraining tests regressor training
func TestRegressorTraining(t *testing.T) {
	t.Skip("LightGBM training not yet implemented - planned for v0.7.0")
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
	params := TrainingParams{
		Lambda: 1.0,
		Alpha:  0.0,
	}
	trainer := &Trainer{
		params:      params,
		regularizer: NewRegularizationStrategy(params),
	}

	// Test case 1: Perfect split
	// The gain formula is: left_score + right_score - parent_score
	// where score = -0.5 * G^2 / (H + lambda)
	gain := trainer.calculateSplitGain(
		-10.0, 5.0, // left gradient and hessian
		10.0, 5.0, // right gradient and hessian
		0.0, 10.0, // total gradient and hessian
	)

	// With lambda=1.0:
	// left_score = 0.5 * 100 / 6 = 8.333
	// right_score = 0.5 * 100 / 6 = 8.333
	// parent_score = 0 (since parent gradient is 0)
	// gain = 8.333 + 8.333 - 0 = 16.667
	expectedGain := 16.667
	if math.Abs(gain-expectedGain) > 0.01 {
		t.Errorf("Expected gain %.4f, got %.4f", expectedGain, gain)
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
	params := TrainingParams{
		Lambda: 1.0,
	}
	trainer := &Trainer{
		params:      params,
		gradients:   []float64{-1.0, -2.0, -3.0, 1.0, 2.0},
		hessians:    []float64{1.0, 1.0, 1.0, 1.0, 1.0},
		regularizer: NewRegularizationStrategy(params),
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

// TestBinaryClassificationGradients tests gradient calculation for binary classification
func TestBinaryClassificationGradients(t *testing.T) {
	t.Skip("Binary classification gradients calculation needs debugging")
	// Create simple binary classification data
	X := mat.NewDense(4, 2, []float64{
		0.1, 0.2,
		0.3, 0.4,
		0.7, 0.8,
		0.9, 1.0,
	})
	y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})

	trainer := &Trainer{
		params: TrainingParams{
			Objective: "binary",
		},
		X:         X,
		y:         y,
		gradients: make([]float64, 4),
		hessians:  make([]float64, 4),
		trees:     []Tree{}, // No trees yet, so initial prediction = 0
	}

	// Calculate gradients for binary classification
	trainer.calculateGradients()

	// For binary classification with logistic loss:
	// When initial prediction = 0 (sigmoid(0) = 0.5):
	// gradient = prediction - target
	// For y=0: gradient = 0.5 - 0 = 0.5
	// For y=1: gradient = 0.5 - 1 = -0.5
	// hessian = prediction * (1 - prediction) = 0.5 * 0.5 = 0.25

	for i := 0; i < 4; i++ {
		target := y.At(i, 0)
		if target == 0 {
			// For binary classification, this should NOT be prediction - target (regression)
			// but sigmoid(pred) - target
			if math.Abs(trainer.gradients[i]-0.5) > 0.1 {
				t.Errorf("Sample %d: Expected gradient ~0.5 for y=0, got %.4f", i, trainer.gradients[i])
			}
		} else {
			if math.Abs(trainer.gradients[i]+0.5) > 0.1 {
				t.Errorf("Sample %d: Expected gradient ~-0.5 for y=1, got %.4f", i, trainer.gradients[i])
			}
		}
		// For binary classification, hessian should be p*(1-p), not 1.0
		if math.Abs(trainer.hessians[i]-0.25) > 0.1 && trainer.params.Objective == "binary" {
			t.Errorf("Sample %d: Expected hessian ~0.25 for binary classification, got %.4f", i, trainer.hessians[i])
		}
	}
}

// TestMulticlassClassificationGradients tests gradient calculation for multiclass classification
func TestMulticlassClassificationGradients(t *testing.T) {
	t.Skip("Multiclass classification gradients calculation needs debugging")
	// Create simple 3-class classification data
	X := mat.NewDense(6, 2, []float64{
		0.1, 0.1,
		0.2, 0.2,
		0.5, 0.5,
		0.6, 0.6,
		0.8, 0.8,
		0.9, 0.9,
	})
	y := mat.NewDense(6, 1, []float64{0, 0, 1, 1, 2, 2})

	trainer := &Trainer{
		params: TrainingParams{
			Objective: "multiclass",
			NumClass:  3,
		},
		X:         X,
		y:         y,
		gradients: make([]float64, 6),
		hessians:  make([]float64, 6),
		trees:     []Tree{}, // No trees yet
	}

	// Calculate gradients for multiclass classification
	trainer.calculateGradients()

	// For multiclass with softmax:
	// Initial predictions are uniform (1/3 for each class)
	// gradient = prediction[class] - 1 (for true class)
	// gradient = prediction[class] (for other classes)
	// This test checks that gradients are NOT constant 1.0 (regression behavior)

	for i := 0; i < 6; i++ {
		// For multiclass, gradients should depend on softmax probabilities
		// They should NOT all be 1.0 (regression behavior)
		if trainer.params.Objective == "multiclass" && math.Abs(trainer.hessians[i]-1.0) < 0.01 {
			t.Errorf("Sample %d: Hessian appears to use regression formula (1.0), not multiclass", i)
		}
	}
}

// TestNativeMulticlassWithGonum tests native multiclass implementation using Gonum matrices
func TestNativeMulticlassWithGonum(t *testing.T) {
	t.Skip("Skipping native multiclass tests until v0.7.0 implementation")

	// Create 3-class classification data
	nSamples := 60
	nFeatures := 2
	nClasses := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	// Generate simple separable data for 3 classes
	for i := 0; i < nSamples; i++ {
		classIdx := i / (nSamples / nClasses)
		y.Set(i, 0, float64(classIdx))

		// Create somewhat separable features
		X.Set(i, 0, float64(classIdx)*0.3+0.1*float64(i%10)/10.0)
		X.Set(i, 1, float64(classIdx)*0.4+0.1*float64(i%5)/5.0)
	}

	// Create trainer with native multiclass objective
	params := TrainingParams{
		NumIterations: 5,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		MinDataInLeaf: 5,
		Lambda:        1.0,
		Objective:     "multiclass_native", // New objective type
		NumClass:      nClasses,
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)

	// For native multiclass, check that trainer initializes properly
	if trainer.params.NumClass != nClasses {
		t.Errorf("NumClass not set correctly: got %d, want %d", trainer.params.NumClass, nClasses)
	}

	// Train model
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Native multiclass training failed: %v", err)
	}

	// Check that we have the right number of trees
	// For native multiclass, we should have NumIterations trees (not NumIterations * NumClass)
	expectedTrees := params.NumIterations
	if len(trainer.trees) != expectedTrees {
		t.Errorf("Native multiclass should create %d trees, got %d", expectedTrees, len(trainer.trees))
	}

	// Get model
	model := trainer.GetModel()
	if model == nil {
		t.Fatal("GetModel returned nil")
	}

	// Check that model has correct objective
	if string(model.Objective) != "multiclass_native" {
		t.Errorf("Model objective incorrect: got %s, want multiclass_native", model.Objective)
	}
}

// TestNativeMulticlassGradients tests gradient calculation for native multiclass with Gonum
func TestNativeMulticlassGradients(t *testing.T) {
	// Create simple 3-class data
	X := mat.NewDense(6, 2, []float64{
		0.1, 0.1,
		0.2, 0.2,
		0.5, 0.5,
		0.6, 0.6,
		0.8, 0.8,
		0.9, 0.9,
	})
	y := mat.NewDense(6, 1, []float64{0, 0, 1, 1, 2, 2})

	trainer := &Trainer{
		params: TrainingParams{
			Objective: "multiclass_native",
			NumClass:  3,
		},
		X: X,
		y: y,
	}

	// Initialize trainer for native multiclass
	rows, _ := X.Dims()
	trainer.gradients = make([]float64, rows) // Will be replaced with mat.Dense
	trainer.hessians = make([]float64, rows)  // Will be replaced with mat.Dense
	trainer.trees = []Tree{}

	// When properly implemented, gradients should be a matrix [samples x classes]
	// For now, test that calculateGradients doesn't panic
	trainer.calculateGradients()

	// After native implementation, gradients should be different for each class
	// This is a placeholder test that will be updated when implementation is complete
	if trainer.params.Objective == "multiclass_native" {
		// Check that gradients are computed (not all zero)
		allZero := true
		for i := 0; i < rows; i++ {
			if trainer.gradients[i] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Error("Native multiclass gradients should not be all zero")
		}
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
	t.Skip("LightGBM training not yet implemented - planned for v0.7.0")
	// Create data where model should converge quickly
	X := mat.NewDense(50, 1, nil)
	y := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		X.Set(i, 0, float64(i))
		y.Set(i, 0, float64(i)) // Perfect linear relationship
	}

	params := TrainingParams{
		NumIterations: 100, // Set high
		LearningRate:  0.3,
		NumLeaves:     5,
		EarlyStopping: 5,
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
