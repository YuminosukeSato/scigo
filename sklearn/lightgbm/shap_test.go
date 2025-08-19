package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// TestTreeSHAPBasic tests basic SHAP value calculation
func TestTreeSHAPBasic(t *testing.T) {
	// Create a simple dataset
	nSamples := 100
	nFeatures := 4

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	// Generate data
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		// Simple linear relationship
		y.Set(i, 0, X.At(i, 0)*2+X.At(i, 1)*1.5-X.At(i, 2)*0.5+distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
	}

	// Train a model
	reg := NewLGBMRegressor().
		WithNumIterations(10).
		WithNumLeaves(5).
		WithLearningRate(0.1)

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP values
	shapValues, err := reg.PredictSHAP(X)
	require.NoError(t, err)
	assert.NotNil(t, shapValues)

	// Check dimensions
	rows, cols := shapValues.Values.Dims()
	assert.Equal(t, nSamples, rows)
	assert.Equal(t, nFeatures, cols)

	// Check that SHAP values sum to prediction - base value
	predictions, err := reg.Predict(X)
	require.NoError(t, err)

	for i := 0; i < nSamples; i++ {
		shapSum := 0.0
		for j := 0; j < nFeatures; j++ {
			shapSum += shapValues.Values.At(i, j)
		}

		// SHAP values + base value should approximately equal prediction
		expectedPred := shapSum + shapValues.BaseValue
		actualPred := predictions.At(i, 0)

		// Allow some tolerance due to numerical precision
		assert.InDelta(t, actualPred, expectedPred, 0.1,
			"Sample %d: SHAP sum + base value should equal prediction", i)
	}
}

// TestTreeSHAPImportance tests that SHAP values reflect feature importance
func TestTreeSHAPImportance(t *testing.T) {
	// Create dataset where feature 0 is most important
	nSamples := 200
	nFeatures := 5

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		// Feature 0 has strong effect, feature 1 moderate, others weak
		y.Set(i, 0,
			X.At(i, 0)*5.0+ // Strong effect
				X.At(i, 1)*1.0+ // Moderate effect
				X.At(i, 2)*0.1+ // Weak effect
				distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
	}

	// Train model
	reg := NewLGBMRegressor().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP values
	shapValues, err := reg.PredictSHAP(X)
	require.NoError(t, err)

	// Calculate mean absolute SHAP values for each feature
	meanAbsShap := make([]float64, nFeatures)
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += math.Abs(shapValues.Values.At(i, j))
		}
		meanAbsShap[j] = sum / float64(nSamples)
	}

	// Feature 0 should have highest mean absolute SHAP value
	assert.Greater(t, meanAbsShap[0], meanAbsShap[1],
		"Feature 0 should have higher SHAP importance than feature 1")
	assert.Greater(t, meanAbsShap[1], meanAbsShap[2],
		"Feature 1 should have higher SHAP importance than feature 2")

	t.Logf("Mean absolute SHAP values: %v", meanAbsShap)
}

// TestTreeSHAPBinaryClassification tests SHAP for binary classification
func TestTreeSHAPBinaryClassification(t *testing.T) {
	// Create binary classification dataset
	nSamples := 100
	nFeatures := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		// Binary classification based on linear combination
		score := X.At(i, 0)*2 - X.At(i, 1)*1.5 + X.At(i, 2)*0.5
		if score > 0 {
			y.Set(i, 0, 1)
		} else {
			y.Set(i, 0, 0)
		}
	}

	// Train classifier
	clf := NewLGBMClassifier().
		WithNumIterations(15).
		WithNumLeaves(8).
		WithLearningRate(0.1)

	err := clf.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP values
	shapValues, err := clf.PredictSHAP(X)
	require.NoError(t, err)
	assert.NotNil(t, shapValues)

	// Check dimensions
	rows, cols := shapValues.Values.Dims()
	assert.Equal(t, nSamples, rows)
	assert.Equal(t, nFeatures, cols)
}

// TestInteractionSHAP tests SHAP interaction values
func TestInteractionSHAP(t *testing.T) {
	// Create dataset with feature interactions
	nSamples := 100
	nFeatures := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		x0 := distuv.Normal{Mu: 0, Sigma: 1}.Rand()
		x1 := distuv.Normal{Mu: 0, Sigma: 1}.Rand()
		x2 := distuv.Normal{Mu: 0, Sigma: 1}.Rand()

		X.Set(i, 0, x0)
		X.Set(i, 1, x1)
		X.Set(i, 2, x2)

		// Include interaction term x0*x1
		y.Set(i, 0, x0+x1+x0*x1*2+distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
	}

	// Train model
	reg := NewLGBMRegressor().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate interaction SHAP values
	interactionShap := NewInteractionSHAP(reg.Model)
	interactions, err := interactionShap.CalculateInteractions(X)
	require.NoError(t, err)
	assert.NotNil(t, interactions)

	// Check that we have interaction matrices for each sample
	assert.Equal(t, nSamples, len(interactions))

	// Check dimensions of interaction matrices
	for i, interaction := range interactions {
		rows, cols := interaction.Dims()
		assert.Equal(t, nFeatures, rows, "Sample %d: interaction matrix rows", i)
		assert.Equal(t, nFeatures, cols, "Sample %d: interaction matrix cols", i)

		// Check symmetry
		for r := 0; r < rows; r++ {
			for c := r + 1; c < cols; c++ {
				assert.InDelta(t, interaction.At(r, c), interaction.At(c, r), 1e-10,
					"Sample %d: interaction matrix should be symmetric", i)
			}
		}
	}
}

// TestSHAPConsistency tests consistency of SHAP values
func TestSHAPConsistency(t *testing.T) {
	// Create dataset
	nSamples := 50
	nFeatures := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	// Use fixed seed for reproducibility
	source := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, source.Rand())
		}
		y.Set(i, 0, X.At(i, 0)+X.At(i, 1)+source.Rand()*0.1)
	}

	// Train model with deterministic settings
	reg := NewLGBMRegressor().
		WithNumIterations(10).
		WithNumLeaves(5).
		WithLearningRate(0.1).
		WithRandomState(42).
		WithDeterministic(true)

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP values twice
	shap1, err := reg.PredictSHAP(X)
	require.NoError(t, err)

	shap2, err := reg.PredictSHAP(X)
	require.NoError(t, err)

	// SHAP values should be identical
	rows, cols := shap1.Values.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			assert.Equal(t, shap1.Values.At(i, j), shap2.Values.At(i, j),
				"SHAP values should be consistent across calls")
		}
	}

	assert.Equal(t, shap1.BaseValue, shap2.BaseValue,
		"Base values should be consistent")
}

// TestSHAPWithCategoricalFeatures tests SHAP with categorical features
func TestSHAPWithCategoricalFeatures(t *testing.T) {
	// Create dataset with categorical features
	nSamples := 100
	nFeatures := 4
	categoricalFeatures := []int{2} // Feature 2 is categorical

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		// Continuous features
		X.Set(i, 0, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		X.Set(i, 1, distuv.Normal{Mu: 0, Sigma: 1}.Rand())

		// Categorical feature (values 0, 1, 2)
		X.Set(i, 2, float64(i%3))

		// Another continuous feature
		X.Set(i, 3, distuv.Normal{Mu: 0, Sigma: 1}.Rand())

		// Target with categorical effect
		catEffect := 0.0
		switch int(X.At(i, 2)) {
		case 0:
			catEffect = -1.0
		case 1:
			catEffect = 0.0
		case 2:
			catEffect = 1.0
		}

		y.Set(i, 0, X.At(i, 0)*2+catEffect+X.At(i, 3)*0.5+
			distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
	}

	// Train model with categorical features
	reg := NewLGBMRegressor().
		WithNumIterations(15).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	reg.CategoricalFeatures = categoricalFeatures

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP values
	shapValues, err := reg.PredictSHAP(X)
	require.NoError(t, err)

	// Check that categorical feature has non-zero SHAP values
	catFeatureIdx := 2
	hasNonZero := false
	for i := 0; i < nSamples; i++ {
		if math.Abs(shapValues.Values.At(i, catFeatureIdx)) > 1e-10 {
			hasNonZero = true
			break
		}
	}

	assert.True(t, hasNonZero,
		"Categorical feature should have non-zero SHAP values")
}

// TestSHAPEdgeCases tests edge cases for SHAP calculation
func TestSHAPEdgeCases(t *testing.T) {
	// Test with single sample
	X := mat.NewDense(1, 2, []float64{1.0, 2.0})
	y := mat.NewDense(1, 1, []float64{3.0})

	reg := NewLGBMRegressor().
		WithNumIterations(5).
		WithNumLeaves(3)

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Calculate SHAP for single sample
	shapValues, err := reg.PredictSHAP(X)
	require.NoError(t, err)

	rows, cols := shapValues.Values.Dims()
	assert.Equal(t, 1, rows)
	assert.Equal(t, 2, cols)

	// Test with many samples but few features
	X2 := mat.NewDense(100, 1, nil)
	y2 := mat.NewDense(100, 1, nil)
	for i := 0; i < 100; i++ {
		val := distuv.Normal{Mu: 0, Sigma: 1}.Rand()
		X2.Set(i, 0, val)
		y2.Set(i, 0, val*2)
	}

	reg2 := NewLGBMRegressor().
		WithNumIterations(10).
		WithNumLeaves(5)

	err = reg2.Fit(X2, y2)
	require.NoError(t, err)

	shapValues2, err := reg2.PredictSHAP(X2)
	require.NoError(t, err)

	rows2, cols2 := shapValues2.Values.Dims()
	assert.Equal(t, 100, rows2)
	assert.Equal(t, 1, cols2)
}

// BenchmarkSHAPCalculation benchmarks SHAP calculation performance
func BenchmarkSHAPCalculation(b *testing.B) {
	// Create dataset
	nSamples := 1000
	nFeatures := 10

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		y.Set(i, 0, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
	}

	// Train model
	reg := NewLGBMRegressor().
		WithNumIterations(50).
		WithNumLeaves(31)

	err := reg.Fit(X, y)
	if err != nil {
		b.Fatalf("Failed to fit model: %v", err)
	}

	// Benchmark SHAP calculation
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := reg.PredictSHAP(X)
		if err != nil {
			b.Fatalf("Failed to predict SHAP: %v", err)
		}
	}
}
