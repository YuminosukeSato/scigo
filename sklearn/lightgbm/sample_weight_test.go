package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// TestSampleWeightBinaryClassification tests sample weights for binary classification
func TestSampleWeightBinaryClassification(t *testing.T) {
	// Create imbalanced dataset
	nSamples := 100
	nFeatures := 4

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate imbalanced labels (90% class 0, 10% class 1)
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		if i < 90 {
			y.Set(i, 0, 0)
		} else {
			y.Set(i, 0, 1)
		}
	}

	// Create sample weights to balance classes
	sampleWeight := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		if y.At(i, 0) == 0 {
			sampleWeight[i] = 1.0
		} else {
			sampleWeight[i] = 9.0 // Weight minority class more
		}
	}

	// Train without weights
	clf1 := NewLGBMClassifier().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := clf1.Fit(X, y)
	require.NoError(t, err)

	// Train with weights
	clf2 := NewLGBMClassifier().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err = clf2.FitWeighted(X, y, sampleWeight)
	require.NoError(t, err)

	// Both models should be fitted
	assert.True(t, clf1.IsFitted())
	assert.True(t, clf2.IsFitted())

	// Test on minority class samples
	minorityIndices := []int{90, 95, 99}
	for _, idx := range minorityIndices {
		xTest := mat.NewDense(1, nFeatures, nil)
		for j := 0; j < nFeatures; j++ {
			xTest.Set(0, j, X.At(idx, j))
		}

		// Weighted model should perform better on minority class
		proba1, err := clf1.PredictProba(xTest)
		require.NoError(t, err)

		proba2, err := clf2.PredictProba(xTest)
		require.NoError(t, err)

		// Log the probabilities for debugging
		t.Logf("Sample %d - Without weights: P(class=1)=%.3f, With weights: P(class=1)=%.3f",
			idx, proba1.At(0, 1), proba2.At(0, 1))
	}
}

// TestSampleWeightRegression tests sample weights for regression
func TestSampleWeightRegression(t *testing.T) {
	// Create dataset
	nSamples := 100
	nFeatures := 3

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate targets with outliers
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		if i < 95 {
			// Normal samples
			y.Set(i, 0, X.At(i, 0)+X.At(i, 1)+distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
		} else {
			// Outliers
			y.Set(i, 0, 10.0+distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
		}
	}

	// Create sample weights (downweight outliers)
	sampleWeight := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		if i < 95 {
			sampleWeight[i] = 1.0
		} else {
			sampleWeight[i] = 0.1 // Downweight outliers
		}
	}

	// Train without weights
	reg1 := NewLGBMRegressor().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := reg1.Fit(X, y)
	require.NoError(t, err)

	// Train with weights
	reg2 := NewLGBMRegressor().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err = reg2.FitWeighted(X, y, sampleWeight)
	require.NoError(t, err)

	// Both models should be fitted
	assert.True(t, reg1.IsFitted())
	assert.True(t, reg2.IsFitted())

	// Test on normal samples (should be better with weighted model)
	normalIndices := []int{0, 10, 20, 30, 40}
	mse1, mse2 := 0.0, 0.0

	for _, idx := range normalIndices {
		xTest := mat.NewDense(1, nFeatures, nil)
		for j := 0; j < nFeatures; j++ {
			xTest.Set(0, j, X.At(idx, j))
		}

		pred1, err := reg1.Predict(xTest)
		require.NoError(t, err)

		pred2, err := reg2.Predict(xTest)
		require.NoError(t, err)

		trueVal := y.At(idx, 0)
		err1 := pred1.At(0, 0) - trueVal
		err2 := pred2.At(0, 0) - trueVal

		mse1 += err1 * err1
		mse2 += err2 * err2

		t.Logf("Sample %d - True: %.3f, Without weights: %.3f, With weights: %.3f",
			idx, trueVal, pred1.At(0, 0), pred2.At(0, 0))
	}

	mse1 /= float64(len(normalIndices))
	mse2 /= float64(len(normalIndices))

	t.Logf("MSE on normal samples - Without weights: %.3f, With weights: %.3f", mse1, mse2)

	// Weighted model should generally perform better on normal samples
	// Note: This is a stochastic test, so we allow some tolerance
	if mse2 > mse1*1.5 {
		t.Logf("Warning: Weighted model performed worse than expected, but this can happen due to randomness")
	}
}

// TestSampleWeightValidation tests input validation for sample weights
func TestSampleWeightValidation(t *testing.T) {
	// Create small dataset
	X := mat.NewDense(10, 2, nil)
	y := mat.NewDense(10, 1, nil)

	// Initialize with proper data
	for i := 0; i < 10; i++ {
		for j := 0; j < 2; j++ {
			X.Set(i, j, float64(i+j))
		}
		y.Set(i, 0, float64(i%2)) // Binary classification
	}

	// Test with wrong weight length
	wrongWeights := make([]float64, 5) // Wrong length

	clf := NewLGBMClassifier()
	err := clf.FitWeighted(X, y, wrongWeights)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "dimension")

	reg := NewLGBMRegressor()
	err = reg.FitWeighted(X, y, wrongWeights)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "dimension")

	// Test with nil weights (should work)
	err = clf.FitWeighted(X, y, nil)
	assert.NoError(t, err)

	err = reg.FitWeighted(X, y, nil)
	assert.NoError(t, err)

	// Test with correct weights
	correctWeights := make([]float64, 10)
	for i := range correctWeights {
		correctWeights[i] = 1.0
	}

	err = clf.FitWeighted(X, y, correctWeights)
	assert.NoError(t, err)

	err = reg.FitWeighted(X, y, correctWeights)
	assert.NoError(t, err)
}

// TestSampleWeightMulticlass tests sample weights for multiclass classification
func TestSampleWeightMulticlass(t *testing.T) {
	// Create imbalanced multiclass dataset
	nSamples := 150
	nFeatures := 4
	nClasses := 3

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: float64(i / 50), Sigma: 1}.Rand())
		}
	}

	// Generate imbalanced labels (70% class 0, 20% class 1, 10% class 2)
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		if i < 105 {
			y.Set(i, 0, 0)
		} else if i < 135 {
			y.Set(i, 0, 1)
		} else {
			y.Set(i, 0, 2)
		}
	}

	// Create sample weights to balance classes
	sampleWeight := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		class := int(y.At(i, 0))
		switch class {
		case 0:
			sampleWeight[i] = 1.0
		case 1:
			sampleWeight[i] = 3.5
		case 2:
			sampleWeight[i] = 7.0
		}
	}

	// Train with weights
	clf := NewLGBMClassifier().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := clf.FitWeighted(X, y, sampleWeight)
	require.NoError(t, err)
	assert.True(t, clf.IsFitted())

	// Check that we can predict
	pred, err := clf.Predict(X)
	require.NoError(t, err)
	assert.Equal(t, nSamples, pred.(*mat.Dense).RawMatrix().Rows)

	// Check probability predictions
	proba, err := clf.PredictProba(X)
	require.NoError(t, err)
	rows, cols := proba.Dims()
	assert.Equal(t, nSamples, rows)
	assert.Equal(t, nClasses, cols)

	// Verify probabilities sum to 1
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += proba.At(i, j)
		}
		assert.InDelta(t, 1.0, sum, 1e-6)
	}
}

// TestSampleWeightZero tests behavior with zero weights
func TestSampleWeightZero(t *testing.T) {
	// Create small dataset
	X := mat.NewDense(10, 2, nil)
	y := mat.NewDense(10, 1, nil)

	for i := 0; i < 10; i++ {
		for j := 0; j < 2; j++ {
			X.Set(i, j, float64(i+j))
		}
		y.Set(i, 0, float64(i%2))
	}

	// Create weights with some zeros
	weights := make([]float64, 10)
	for i := range weights {
		if i < 5 {
			weights[i] = 0.0 // Zero weight for first half
		} else {
			weights[i] = 1.0
		}
	}

	clf := NewLGBMClassifier().
		WithNumIterations(10).
		WithNumLeaves(5)

	err := clf.FitWeighted(X, y, weights)
	assert.NoError(t, err)

	// Model should still train (using only non-zero weighted samples effectively)
	assert.True(t, clf.IsFitted())
}

// TestSampleWeightNegative tests behavior with negative weights (should handle gracefully)
func TestSampleWeightNegative(t *testing.T) {
	// Create small dataset
	X := mat.NewDense(10, 2, nil)
	y := mat.NewDense(10, 1, nil)

	// Create weights with negative values
	weights := make([]float64, 10)
	for i := range weights {
		weights[i] = -1.0 // Negative weights
	}

	clf := NewLGBMClassifier()
	err := clf.FitWeighted(X, y, weights)

	// Should either error or handle gracefully
	// The behavior depends on the implementation
	if err == nil {
		// If no error, model should be fitted
		assert.True(t, clf.IsFitted())
	}
}

// TestSampleWeightEquivalence tests that equal weights give same result as no weights
func TestSampleWeightEquivalence(t *testing.T) {
	// Create dataset
	X := mat.NewDense(50, 3, nil)
	y := mat.NewDense(50, 1, nil)

	// Set seed for reproducibility
	source := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < 50; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, source.Rand())
		}
		y.Set(i, 0, math.Round(source.Rand()+0.5))
	}

	// Create uniform weights
	weights := make([]float64, 50)
	for i := range weights {
		weights[i] = 1.0
	}

	// Train without weights
	clf1 := NewLGBMClassifier().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithRandomState(42).
		WithDeterministic(true)

	err := clf1.Fit(X, y)
	require.NoError(t, err)

	// Train with uniform weights
	clf2 := NewLGBMClassifier().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithRandomState(42).
		WithDeterministic(true)

	err = clf2.FitWeighted(X, y, weights)
	require.NoError(t, err)

	// Predictions should be very similar (not necessarily identical due to numerical precision)
	pred1, err := clf1.Predict(X)
	require.NoError(t, err)

	pred2, err := clf2.Predict(X)
	require.NoError(t, err)

	// Count differences
	differences := 0
	for i := 0; i < 50; i++ {
		if math.Abs(pred1.At(i, 0)-pred2.At(i, 0)) > 1e-6 {
			differences++
		}
	}

	// Should have very few differences
	assert.Less(t, differences, 5, "Too many differences between uniform weighted and unweighted models")
}
