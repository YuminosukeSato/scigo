package lightgbm

import (
	"math"
	"sort"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestHistogramBasedSplitting tests the histogram-based splitting algorithm
func TestHistogramBasedSplitting(t *testing.T) {
	// Create test data
	X := mat.NewDense(100, 5, nil)
	y := mat.NewDense(100, 1, nil)

	// Fill with synthetic data
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i*j)/100.0)
		}
		y.Set(i, 0, math.Sin(float64(i)/10.0))
	}

	// Test with histogram-based trainer
	params := TrainingParams{
		NumIterations:   10,
		LearningRate:    0.1,
		NumLeaves:       31,
		MaxBin:          255,
		MinDataInLeaf:   20,
		Lambda:          0.1,
		MinGainToSplit:  0.01,
		BaggingFraction: 1.0,
		FeatureFraction: 1.0,
		Objective:       "regression",
		Verbosity:       -1,
	}

	trainer := NewTrainer(params)
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Verify histograms were created
	if trainer.histPool == nil {
		t.Error("Histogram pool was not initialized")
	}

	// Verify trees were built
	if len(trainer.trees) == 0 {
		t.Error("No trees were built")
	}

	t.Logf("Successfully built %d trees with histogram-based splitting", len(trainer.trees))
}

// TestKahanSummation tests the numerical precision improvements
func TestKahanSummation(t *testing.T) {
	trainer := &Trainer{}

	// Test Kahan summation with numbers that would cause precision loss
	testCases := []struct {
		values   []float64
		expected float64
	}{
		{
			values:   []float64{1e10, 1.0, -1e10},
			expected: 1.0,
		},
		{
			values:   []float64{1e-10, 1e-10, 1e-10, 1e-10, 1e-10},
			expected: 5e-10,
		},
	}

	for _, tc := range testCases {
		sum := 0.0
		compensation := 0.0
		for _, val := range tc.values {
			trainer.kahanAdd(&sum, val, compensation)
		}

		if math.Abs(sum-tc.expected) > 1e-15 {
			t.Errorf("Kahan summation failed: expected %e, got %e", tc.expected, sum)
		}
	}
}

// TestGOSSSampling tests the GOSS sampling implementation
func TestGOSSSampling(t *testing.T) {
	// Create test data with varied gradients
	X := mat.NewDense(1000, 10, nil)
	y := mat.NewDense(1000, 1, nil)

	for i := 0; i < 1000; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i+j)/1000.0)
		}
		y.Set(i, 0, float64(i%3))
	}

	params := TrainingParams{
		NumIterations: 5,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxBin:        255,
		MinDataInLeaf: 20,
		BoostingType:  "goss",
		Objective:     "regression",
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)

	// Initialize trainer
	trainer.X = X
	trainer.y = y
	
	// Set objective function
	trainer.objective = NewObjectiveFunction(params.Objective, params)
	trainer.initScore = trainer.objective.InitScore(y)
	
	err := trainer.initialize()
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Calculate gradients
	trainer.calculateGradients()

    // Test GOSS sampling
    sampledIndices := trainer.gosssampling()

	// Verify sampling worked correctly
	expectedTopCount := int(float64(1000) * trainer.gossTopRate)
	expectedOtherCount := int(float64(1000-expectedTopCount) * trainer.gossOtherRate)
	expectedTotal := expectedTopCount + expectedOtherCount

	if len(sampledIndices) != expectedTotal {
		t.Errorf("GOSS sampling returned wrong number of samples: expected %d, got %d",
			expectedTotal, len(sampledIndices))
	}

	t.Logf("GOSS sampling: selected %d samples from 1000 (top: %d, other: %d)",
		len(sampledIndices), expectedTopCount, expectedOtherCount)
}

// TestGOSSSamplingAmplificationAndDeterminism checks amplification factor and determinism
func TestGOSSSamplingAmplificationAndDeterminism(t *testing.T) {
    // Build synthetic data
    X := mat.NewDense(200, 4, nil)
    y := mat.NewDense(200, 1, nil)
    for i := 0; i < 200; i++ {
        for j := 0; j < 4; j++ {
            X.Set(i, j, float64(i+j))
        }
        y.Set(i, 0, float64(i%2))
    }

    params := TrainingParams{
        BoostingType:  "goss",
        TopRate:       0.2,
        OtherRate:     0.1,
        Seed:          42,
        NumIterations: 1,
        Objective:     "regression",
    }
    trainer := NewTrainer(params)
    trainer.X = X
    trainer.y = y
    _ = trainer.initialize()
    trainer.calculateGradients()

    // Snapshot hessians before sampling
    gBefore := append([]float64(nil), trainer.gradients...)
    hBefore := append([]float64(nil), trainer.hessians...)

    s1 := trainer.gosssampling()
    // Determinism: same seed yields same selection next call
    trainer.gradients = append([]float64(nil), gBefore...)
    trainer.hessians = append([]float64(nil), hBefore...)
    s2 := trainer.gosssampling()
    if len(s1) != len(s2) {
        t.Fatalf("non-deterministic selection length: %d vs %d", len(s1), len(s2))
    }
    for i := range s1 {
        if s1[i] != s2[i] {
            t.Fatalf("non-deterministic selection index at %d: %d vs %d", i, s1[i], s2[i])
        }
    }

    // Amplification applied to the 'other' (non-top) sampled set only
    topCount := int(float64(200) * trainer.gossTopRate)
    amp := (1.0 - trainer.gossTopRate) / trainer.gossOtherRate

    // Build a set for quick lookup
    selected := make(map[int]bool)
    for _, idx := range s1 {
        selected[idx] = true
    }

    // Verify: any selected index beyond topCount should be amplified
    // Note: we don't know exact partition of s1, so we recompute top set by grads
    type pair struct{ idx int; val float64 }
    abs := make([]pair, 200)
    for i := 0; i < 200; i++ { abs[i] = pair{i, math.Abs(gBefore[i])} }
    sort.Slice(abs, func(i, j int) bool { return abs[i].val > abs[j].val })
    topSet := make(map[int]bool)
    for i := 0; i < topCount && i < len(abs); i++ { topSet[abs[i].idx] = true }

    for i := 0; i < 200; i++ {
        if selected[i] && !topSet[i] {
            // amplified
            if math.Abs(trainer.gradients[i]-gBefore[i]*amp) > 1e-12 {
                t.Fatalf("grad not amplified for %d", i)
            }
            if math.Abs(trainer.hessians[i]-hBefore[i]*amp) > 1e-12 {
                t.Fatalf("hess not amplified for %d", i)
            }
        }
    }
}

// TestParallelHistogramConstruction tests parallel histogram building
func TestParallelHistogramConstruction(t *testing.T) {
	t.Skip("Skipping parallel histogram tests until v0.7.0 implementation")

	// Create larger test data for parallel processing
	X := mat.NewDense(10000, 20, nil)
	y := mat.NewDense(10000, 1, nil)

	for i := 0; i < 10000; i++ {
		for j := 0; j < 20; j++ {
			X.Set(i, j, float64(i*j)/10000.0)
		}
		y.Set(i, 0, float64(i%10))
	}

	params := TrainingParams{
		NumIterations: 1,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxBin:        255,
		MinDataInLeaf: 20,
		Objective:     "regression",
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)

	// Initialize
	trainer.X = X
	trainer.y = y
	err := trainer.initialize()
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Calculate gradients
	trainer.calculateGradients()

	// Build histogram for root node
	indices := make([]int, 10000)
	for i := 0; i < 10000; i++ {
		indices[i] = i
	}

	nodeHist := trainer.buildNodeHistogram(indices)

	// Test parallel split finding
	bestSplit := trainer.findBestSplitWithHistogram(nodeHist, indices)

	if bestSplit.Gain <= 0 {
		t.Error("Failed to find valid split with parallel histogram method")
	}

	t.Logf("Found best split: feature=%d, threshold=%f, gain=%f",
		bestSplit.Feature, bestSplit.Threshold, bestSplit.Gain)
}

// TestHistogramSubtraction tests the histogram subtraction optimization
func TestHistogramSubtraction(t *testing.T) {
	t.Skip("Skipping histogram subtraction tests until v0.7.0 implementation")

	// Create test trainer
	params := TrainingParams{MaxBin: 10}
	trainer := NewTrainer(params)

	// Create parent histogram
	parent := &NodeHistogram{
		histograms: make([][]Histogram, 1),
		totalGrad:  100.0,
		totalHess:  50.0,
		count:      100,
	}

	parent.histograms[0] = []Histogram{
		{Count: 30, SumGrad: 30.0, SumHess: 15.0},
		{Count: 40, SumGrad: 40.0, SumHess: 20.0},
		{Count: 30, SumGrad: 30.0, SumHess: 15.0},
	}

	// Create sibling histogram
	sibling := &NodeHistogram{
		histograms: make([][]Histogram, 1),
		totalGrad:  60.0,
		totalHess:  30.0,
		count:      60,
	}

	sibling.histograms[0] = []Histogram{
		{Count: 20, SumGrad: 20.0, SumHess: 10.0},
		{Count: 25, SumGrad: 25.0, SumHess: 12.5},
		{Count: 15, SumGrad: 15.0, SumHess: 7.5},
	}

	// Test subtraction
	child := trainer.subtractHistogram(parent, sibling)

	// Verify results
	if child.count != 40 {
		t.Errorf("Incorrect child count: expected 40, got %d", child.count)
	}

	if math.Abs(child.totalGrad-40.0) > 1e-10 {
		t.Errorf("Incorrect child total gradient: expected 40.0, got %f", child.totalGrad)
	}

	if math.Abs(child.totalHess-20.0) > 1e-10 {
		t.Errorf("Incorrect child total hessian: expected 20.0, got %f", child.totalHess)
	}

	// Check individual bins
	expectedCounts := []int{10, 15, 15}
	for i, hist := range child.histograms[0] {
		if hist.Count != expectedCounts[i] {
			t.Errorf("Bin %d: incorrect count, expected %d, got %d",
				i, expectedCounts[i], hist.Count)
		}
	}

	t.Log("Histogram subtraction test passed")
}

// BenchmarkHistogramVsNaive benchmarks histogram-based vs naive splitting
func BenchmarkHistogramVsNaive(b *testing.B) {
	// Create test data
	X := mat.NewDense(1000, 10, nil)
	y := mat.NewDense(1000, 1, nil)

	for i := 0; i < 1000; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i*j)/1000.0)
		}
		y.Set(i, 0, float64(i%5))
	}

	params := TrainingParams{
		NumIterations: 10,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxBin:        255,
		MinDataInLeaf: 20,
		Objective:     "regression",
		Verbosity:     -1,
	}

	b.Run("HistogramBased", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			trainer := NewTrainer(params)
			_ = trainer.Fit(X, y)
		}
	})

	b.Run("WithGOSS", func(b *testing.B) {
		params.BoostingType = "goss"
		for i := 0; i < b.N; i++ {
			trainer := NewTrainer(params)
			_ = trainer.Fit(X, y)
		}
	})
}

// TestEndToEndImprovement tests the complete improved implementation
func TestEndToEndImprovement(t *testing.T) {
	t.Skip("Skipping end-to-end improvement tests until v0.7.0 implementation")

	// Load test data (similar to the Python test data if available)
	X := mat.NewDense(500, 10, nil)
	y := mat.NewDense(500, 1, nil)

	// Generate synthetic classification data
	for i := 0; i < 500; i++ {
		sum := 0.0
		for j := 0; j < 10; j++ {
			val := float64(i+j) / 500.0
			X.Set(i, j, val)
			sum += val
		}
		// Binary classification target
		if sum > 5.0 {
			y.Set(i, 0, 1.0)
		} else {
			y.Set(i, 0, 0.0)
		}
	}

	params := TrainingParams{
		NumIterations:   20,
		LearningRate:    0.1,
		NumLeaves:       31,
		MaxDepth:        5,
		MinDataInLeaf:   20,
		Lambda:          0.1,
		Alpha:           0.1,
		MinGainToSplit:  0.01,
		BaggingFraction: 0.8,
		FeatureFraction: 0.8,
		MaxBin:          255,
		BoostingType:    "goss",
		Objective:       "binary",
		Seed:            42,
		Deterministic:   true,
		Verbosity:       -1,
	}

	trainer := NewTrainer(params)
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to train improved model: %v", err)
	}

	// Get model and make predictions
	model := trainer.GetModel()
	predictor := NewPredictor(model)
	predictor.SetDeterministic(true)

	// Test on training data
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Calculate accuracy
	correct := 0
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)

		// Convert probability to class
		predClass := 0.0
		if pred > 0.5 {
			predClass = 1.0
		}

		if predClass == actual {
			correct++
		}
	}

	accuracy := float64(correct) / float64(rows)
	t.Logf("End-to-end test accuracy: %.2f%% (%d/%d correct)",
		accuracy*100, correct, rows)

	if accuracy < 0.7 {
		t.Errorf("Accuracy too low: %.2f%%, expected at least 70%%", accuracy*100)
	}

	// Log model statistics
	t.Logf("Model statistics:")
	t.Logf("  - Number of trees: %d", len(model.Trees))
	t.Logf("  - Number of features: %d", model.NumFeatures)
	t.Logf("  - Learning rate: %f", model.LearningRate)

	totalNodes := 0
	totalLeaves := 0
	for _, tree := range model.Trees {
		totalNodes += len(tree.Nodes)
		totalLeaves += tree.NumLeaves
	}
	t.Logf("  - Total nodes: %d", totalNodes)
	t.Logf("  - Total leaves: %d", totalLeaves)
	t.Logf("  - Average leaves per tree: %.1f", float64(totalLeaves)/float64(len(model.Trees)))
}
