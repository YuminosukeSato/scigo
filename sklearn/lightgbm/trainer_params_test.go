package lightgbm

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestFeatureFraction tests feature sampling functionality
func TestFeatureFraction(t *testing.T) {
	// Create test data
	X := mat.NewDense(100, 10, nil)
	y := mat.NewDense(100, 1, nil)

	// Initialize random data
	for i := 0; i < 100; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i*10+j))
		}
		y.Set(i, 0, float64(i%2))
	}

	// Test different feature fraction values
	testCases := []struct {
		name             string
		featureFraction  float64
		expectedFeatures int
	}{
		{"all_features", 1.0, 10},
		{"half_features", 0.5, 5},
		{"third_features", 0.3, 3},
		{"one_feature", 0.1, 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := TrainingParams{
				NumIterations:   5,
				LearningRate:    0.1,
				NumLeaves:       31,
				FeatureFraction: tc.featureFraction,
				Objective:       "binary",
				Seed:            42,
				Deterministic:   true,
			}

			trainer := NewTrainer(params)
			trainer.X = X
			trainer.y = y

			// Initialize trainer
			if err := trainer.initialize(); err != nil {
				t.Fatalf("Failed to initialize: %v", err)
			}

			// Sample features
			_, numFeatures := X.Dims()
			sampled := trainer.sampler.SampleFeatures(numFeatures, 0)

			if len(sampled) != tc.expectedFeatures {
				t.Errorf("Expected %d features, got %d", tc.expectedFeatures, len(sampled))
			}

			// Check all sampled features are unique
			seen := make(map[int]bool)
			for _, f := range sampled {
				if seen[f] {
					t.Errorf("Duplicate feature %d in sampling", f)
				}
				seen[f] = true
			}
		})
	}
}

// TestBaggingFraction tests data sampling functionality
func TestBaggingFraction(t *testing.T) {
	// Create test data
	X := mat.NewDense(100, 5, nil)
	y := mat.NewDense(100, 1, nil)

	// Initialize random data
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i*5+j))
		}
		y.Set(i, 0, float64(i%2))
	}

	// Test different bagging configurations
	testCases := []struct {
		name            string
		baggingFraction float64
		baggingFreq     int
		iteration       int
		expectedSamples int
	}{
		{"no_bagging", 1.0, 1, 0, 100},
		{"half_bagging", 0.5, 1, 0, 50},
		{"bagging_freq_2_iter_0", 0.5, 2, 0, 50},
		{"bagging_freq_2_iter_1", 0.5, 2, 1, 100}, // No bagging on odd iteration
		{"small_bagging", 0.1, 1, 0, 10},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := TrainingParams{
				NumIterations:   5,
				LearningRate:    0.1,
				NumLeaves:       31,
				BaggingFraction: tc.baggingFraction,
				BaggingFreq:     tc.baggingFreq,
				Objective:       "binary",
				Seed:            42,
				Deterministic:   true,
			}

			trainer := NewTrainer(params)
			trainer.X = X
			trainer.y = y

			// Initialize trainer
			if err := trainer.initialize(); err != nil {
				t.Fatalf("Failed to initialize: %v", err)
			}

			// Sample instances
			rows, _ := X.Dims()
			sampled := trainer.sampler.SampleInstances(rows, tc.iteration)

			if len(sampled) != tc.expectedSamples {
				t.Errorf("Expected %d samples, got %d", tc.expectedSamples, len(sampled))
			}

			// Check all sampled instances are unique and valid
			seen := make(map[int]bool)
			for _, idx := range sampled {
				if idx < 0 || idx >= rows {
					t.Errorf("Invalid sample index %d", idx)
				}
				if seen[idx] {
					t.Errorf("Duplicate sample %d", idx)
				}
				seen[idx] = true
			}
		})
	}
}

// TestL1L2Regularization tests regularization functionality
func TestL1L2Regularization(t *testing.T) {
	testCases := []struct {
		name     string
		lambdaL1 float64
		lambdaL2 float64
		sumGrad  float64
		sumHess  float64
		expected float64
	}{
		{
			name:     "no_regularization",
			lambdaL1: 0.0,
			lambdaL2: 0.0,
			sumGrad:  10.0,
			sumHess:  5.0,
			expected: -2.0, // -10/5
		},
		{
			name:     "l2_only",
			lambdaL1: 0.0,
			lambdaL2: 1.0,
			sumGrad:  10.0,
			sumHess:  5.0,
			expected: -10.0 / 6.0, // -10/(5+1)
		},
		{
			name:     "l1_only_positive",
			lambdaL1: 2.0,
			lambdaL2: 0.0,
			sumGrad:  10.0,
			sumHess:  5.0,
			expected: -(10.0 - 2.0) / 5.0, // -(10-2)/5
		},
		{
			name:     "l1_only_negative",
			lambdaL1: 2.0,
			lambdaL2: 0.0,
			sumGrad:  -10.0,
			sumHess:  5.0,
			expected: -(-10.0 + 2.0) / 5.0, // -(-10+2)/5
		},
		{
			name:     "l1_threshold",
			lambdaL1: 15.0,
			lambdaL2: 0.0,
			sumGrad:  10.0,
			sumHess:  5.0,
			expected: 0.0, // Gradient less than L1 threshold
		},
		{
			name:     "l1_and_l2",
			lambdaL1: 2.0,
			lambdaL2: 1.0,
			sumGrad:  10.0,
			sumHess:  5.0,
			expected: -(10.0 - 2.0) / 6.0, // -(10-2)/(5+1)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := TrainingParams{
				Alpha:  tc.lambdaL1,
				Lambda: tc.lambdaL2,
			}

			regularizer := NewRegularizationStrategy(params)
			result := regularizer.ApplyLeafRegularization(tc.sumGrad, tc.sumHess)

			if math.Abs(result-tc.expected) > 1e-10 {
				t.Errorf("Expected leaf value %.10f, got %.10f", tc.expected, result)
			}
		})
	}
}

// TestRegularizedSplitGain tests split gain calculation with regularization
func TestRegularizedSplitGain(t *testing.T) {
	testCases := []struct {
		name       string
		lambdaL1   float64
		lambdaL2   float64
		leftGrad   float64
		leftHess   float64
		rightGrad  float64
		rightHess  float64
		parentGrad float64
		parentHess float64
	}{
		{
			name:       "simple_split",
			lambdaL1:   0.0,
			lambdaL2:   1.0,
			leftGrad:   -5.0,
			leftHess:   3.0,
			rightGrad:  -5.0,
			rightHess:  2.0,
			parentGrad: -10.0,
			parentHess: 5.0,
		},
		{
			name:       "with_l1_regularization",
			lambdaL1:   1.0,
			lambdaL2:   1.0,
			leftGrad:   -6.0,
			leftHess:   3.0,
			rightGrad:  -4.0,
			rightHess:  2.0,
			parentGrad: -10.0,
			parentHess: 5.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			params := TrainingParams{
				Alpha:  tc.lambdaL1,
				Lambda: tc.lambdaL2,
			}

			regularizer := NewRegularizationStrategy(params)
			gain := regularizer.CalculateSplitGain(
				tc.leftGrad, tc.leftHess,
				tc.rightGrad, tc.rightHess,
				tc.parentGrad, tc.parentHess,
			)

			// Just check that gain is calculated (exact value depends on formula)
			if math.IsNaN(gain) || math.IsInf(gain, 0) {
				t.Errorf("Invalid gain value: %v", gain)
			}

			// Gain can be negative for poor splits (this is normal behavior)
			// We just check that it's not extremely negative (which would indicate an error)
			if gain < -1e6 {
				t.Errorf("Extremely negative gain (likely an error): %v", gain)
			}
		})
	}
}

// TestDeterministicSampling tests that sampling is deterministic with same seed
func TestDeterministicSampling(t *testing.T) {
	params := TrainingParams{
		FeatureFraction: 0.5,
		BaggingFraction: 0.5,
		BaggingFreq:     1,
		Seed:            42,
		Deterministic:   true,
	}

	// Create two samplers with same parameters
	sampler1 := NewSamplingStrategy(params)
	sampler2 := NewSamplingStrategy(params)

	// Test feature sampling
	features1 := sampler1.SampleFeatures(10, 0)
	features2 := sampler2.SampleFeatures(10, 0)

	if len(features1) != len(features2) {
		t.Errorf("Different number of features sampled")
	}

	for i := range features1 {
		if features1[i] != features2[i] {
			t.Errorf("Feature sampling not deterministic at index %d: %d vs %d",
				i, features1[i], features2[i])
		}
	}

	// Test instance sampling
	instances1 := sampler1.SampleInstances(100, 0)
	instances2 := sampler2.SampleInstances(100, 0)

	if len(instances1) != len(instances2) {
		t.Errorf("Different number of instances sampled")
	}

	for i := range instances1 {
		if instances1[i] != instances2[i] {
			t.Errorf("Instance sampling not deterministic at index %d: %d vs %d",
				i, instances1[i], instances2[i])
		}
	}
}

// TestIntegrationWithTraining tests parameters work with actual training
func TestIntegrationWithTraining(t *testing.T) {
	// Create simple dataset
	X := mat.NewDense(50, 5, nil)
	y := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i+j))
		}
		// Simple binary classification based on sum
		sum := 0.0
		for j := 0; j < 5; j++ {
			sum += X.At(i, j)
		}
		if sum > 125 {
			y.Set(i, 0, 1.0)
		} else {
			y.Set(i, 0, 0.0)
		}
	}

	params := TrainingParams{
		NumIterations:   3,
		LearningRate:    0.1,
		NumLeaves:       5,
		MinDataInLeaf:   5,
		FeatureFraction: 0.6,
		BaggingFraction: 0.8,
		BaggingFreq:     1,
		Lambda:          1.0,
		Alpha:           0.5,
		Objective:       "binary",
		Seed:            42,
		Deterministic:   true,
		Verbosity:       0,
	}

	trainer := NewTrainer(params)

	// Train model
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Get model
	model := trainer.GetModel()

	// Check trees were built
	if len(model.Trees) == 0 {
		t.Error("No trees built")
	}

	if len(model.Trees) > params.NumIterations {
		t.Errorf("Too many trees built: %d > %d", len(model.Trees), params.NumIterations)
	}

	// Make predictions
	predictor := NewPredictor(model)
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	// Check predictions are in valid range
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		pred := predictions.At(i, 0)
		if pred < 0 || pred > 1 {
			t.Errorf("Invalid prediction at row %d: %f", i, pred)
		}
	}
}

// BenchmarkFeatureSampling benchmarks feature sampling performance
func BenchmarkFeatureSampling(b *testing.B) {
	params := TrainingParams{
		FeatureFraction: 0.5,
		Seed:            42,
	}

	sampler := NewSamplingStrategy(params)
	numFeatures := 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.SampleFeatures(numFeatures, i)
	}
}

// BenchmarkInstanceSampling benchmarks instance sampling performance
func BenchmarkInstanceSampling(b *testing.B) {
	params := TrainingParams{
		BaggingFraction: 0.5,
		BaggingFreq:     1,
		Seed:            42,
	}

	sampler := NewSamplingStrategy(params)
	numInstances := 10000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampler.SampleInstances(numInstances, i)
	}
}
