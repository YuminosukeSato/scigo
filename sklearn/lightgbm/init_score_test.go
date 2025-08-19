package lightgbm

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// TestInitScoreBinaryClassification tests init_score for binary classification
func TestInitScoreBinaryClassification(t *testing.T) {
	// Create dataset
	nSamples := 100
	nFeatures := 4

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate binary labels
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.Set(i, 0, float64(i%2))
	}

	// Different initial scores
	initScores := []float64{0.0, 0.5, -0.5, 1.0, -1.0}

	for _, initScore := range initScores {
		t.Run(fmt.Sprintf("init_score=%.1f", initScore), func(t *testing.T) {
			clf := NewLGBMClassifier().
				WithNumIterations(20).
				WithNumLeaves(10).
				WithLearningRate(0.1)

			err := clf.FitWithInit(X, y, initScore, nil)
			require.NoError(t, err)
			assert.True(t, clf.IsFitted())

			// Check that model has correct init score
			assert.Equal(t, initScore, clf.Model.InitScore)

			// Make predictions
			pred, err := clf.Predict(X)
			require.NoError(t, err)
			assert.NotNil(t, pred)
		})
	}
}

// TestInitScoreRegression tests init_score for regression
func TestInitScoreRegression(t *testing.T) {
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

	// Generate targets
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.Set(i, 0, X.At(i, 0)+X.At(i, 1)+distuv.Normal{Mu: 0, Sigma: 0.1}.Rand())
	}

	// Different initial scores
	initScores := []float64{0.0, 1.0, -1.0, 2.5, -2.5}

	for _, initScore := range initScores {
		t.Run(fmt.Sprintf("init_score=%.1f", initScore), func(t *testing.T) {
			reg := NewLGBMRegressor().
				WithNumIterations(20).
				WithNumLeaves(10).
				WithLearningRate(0.1)

			err := reg.FitWithInit(X, y, initScore, nil)
			require.NoError(t, err)
			assert.True(t, reg.IsFitted())

			// Check that model has correct init score
			assert.Equal(t, initScore, reg.Model.InitScore)

			// Make predictions
			pred, err := reg.Predict(X)
			require.NoError(t, err)
			assert.NotNil(t, pred)
		})
	}
}

// TestInitScoreWithSampleWeight tests init_score combined with sample weights
func TestInitScoreWithSampleWeight(t *testing.T) {
	// Create dataset
	nSamples := 50
	nFeatures := 2

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate binary labels
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.Set(i, 0, float64(i%2))
	}

	// Create sample weights
	sampleWeight := make([]float64, nSamples)
	for i := range sampleWeight {
		if i < 25 {
			sampleWeight[i] = 1.0
		} else {
			sampleWeight[i] = 2.0
		}
	}

	// Test with both init_score and sample_weight
	clf := NewLGBMClassifier().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	initScore := 0.5
	err := clf.FitWithInit(X, y, initScore, sampleWeight)
	require.NoError(t, err)
	assert.True(t, clf.IsFitted())
	assert.Equal(t, initScore, clf.Model.InitScore)

	// Make predictions
	pred, err := clf.Predict(X)
	require.NoError(t, err)
	assert.NotNil(t, pred)

	// Test regression as well
	reg := NewLGBMRegressor().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err = reg.FitWithInit(X, y, initScore, sampleWeight)
	require.NoError(t, err)
	assert.True(t, reg.IsFitted())
	assert.Equal(t, initScore, reg.Model.InitScore)
}

// TestInitScoreTransferLearning simulates transfer learning scenario
func TestInitScoreTransferLearning(t *testing.T) {
	// Create initial training dataset
	nSamples1 := 50
	nFeatures := 3

	X1 := mat.NewDense(nSamples1, nFeatures, nil)
	y1 := mat.NewDense(nSamples1, 1, nil)

	for i := 0; i < nSamples1; i++ {
		for j := 0; j < nFeatures; j++ {
			X1.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		y1.Set(i, 0, X1.At(i, 0)+X1.At(i, 1))
	}

	// Train initial model
	reg1 := NewLGBMRegressor().
		WithNumIterations(30).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err := reg1.Fit(X1, y1)
	require.NoError(t, err)

	// Get average prediction as init score for next model
	pred1, err := reg1.Predict(X1)
	require.NoError(t, err)

	avgPred := 0.0
	for i := 0; i < nSamples1; i++ {
		avgPred += pred1.At(i, 0)
	}
	avgPred /= float64(nSamples1)

	// Create new dataset (simulating transfer learning)
	nSamples2 := 30
	X2 := mat.NewDense(nSamples2, nFeatures, nil)
	y2 := mat.NewDense(nSamples2, 1, nil)

	for i := 0; i < nSamples2; i++ {
		for j := 0; j < nFeatures; j++ {
			X2.Set(i, j, distuv.Normal{Mu: 0.5, Sigma: 1}.Rand()) // Slightly different distribution
		}
		y2.Set(i, 0, X2.At(i, 0)+X2.At(i, 1)+0.5) // Slightly different target
	}

	// Train new model with init score from previous model
	reg2 := NewLGBMRegressor().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err = reg2.FitWithInit(X2, y2, avgPred, nil)
	require.NoError(t, err)
	assert.True(t, reg2.IsFitted())

	// The model should converge faster with good init score
	pred2, err := reg2.Predict(X2)
	require.NoError(t, err)

	// Calculate MSE
	mse := 0.0
	for i := 0; i < nSamples2; i++ {
		diff := pred2.At(i, 0) - y2.At(i, 0)
		mse += diff * diff
	}
	mse /= float64(nSamples2)

	t.Logf("Transfer learning MSE: %.4f", mse)

	// Train model without init score for comparison
	reg3 := NewLGBMRegressor().
		WithNumIterations(20).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	err = reg3.Fit(X2, y2)
	require.NoError(t, err)

	pred3, err := reg3.Predict(X2)
	require.NoError(t, err)

	mseNoInit := 0.0
	for i := 0; i < nSamples2; i++ {
		diff := pred3.At(i, 0) - y2.At(i, 0)
		mseNoInit += diff * diff
	}
	mseNoInit /= float64(nSamples2)

	t.Logf("No init score MSE: %.4f", mseNoInit)

	// With good init score, performance might be better (though not guaranteed due to randomness)
	t.Logf("Improvement ratio: %.2f%%", (mseNoInit-mse)/mseNoInit*100)
}

// TestInitScoreMulticlass tests init_score for multiclass classification
func TestInitScoreMulticlass(t *testing.T) {
	// Create multiclass dataset
	nSamples := 90
	nFeatures := 4
	nClasses := 3

	// Generate features
	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: float64(i / 30), Sigma: 1}.Rand())
		}
	}

	// Generate multiclass labels
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		y.Set(i, 0, float64(i/30)) // 3 classes: 0, 1, 2
	}

	// Test with init score
	clf := NewLGBMClassifier().
		WithNumIterations(30).
		WithNumLeaves(10).
		WithLearningRate(0.1)

	initScore := 0.3
	err := clf.FitWithInit(X, y, initScore, nil)
	require.NoError(t, err)
	assert.True(t, clf.IsFitted())

	// Make predictions
	pred, err := clf.Predict(X)
	require.NoError(t, err)
	assert.NotNil(t, pred)

	// Check probability predictions
	proba, err := clf.PredictProba(X)
	require.NoError(t, err)
	rows, cols := proba.Dims()
	assert.Equal(t, nSamples, rows)
	assert.Equal(t, nClasses, cols)
}

// TestInitScoreEdgeCases tests edge cases for init_score
func TestInitScoreEdgeCases(t *testing.T) {
	// Small dataset
	X := mat.NewDense(5, 2, []float64{
		1, 2,
		2, 3,
		3, 4,
		4, 5,
		5, 6,
	})
	y := mat.NewDense(5, 1, []float64{1, 0, 1, 0, 1})

	testCases := []struct {
		name      string
		initScore float64
	}{
		{"Zero", 0.0},
		{"Large positive", 10.0},
		{"Large negative", -10.0},
		{"Small positive", 0.001},
		{"Small negative", -0.001},
		{"Infinity", math.Inf(1)},
		{"Negative Infinity", math.Inf(-1)},
		{"NaN", math.NaN()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clf := NewLGBMClassifier().
				WithNumIterations(5).
				WithNumLeaves(3)

			err := clf.FitWithInit(X, y, tc.initScore, nil)

			// Infinity and NaN should ideally be handled gracefully
			if math.IsInf(tc.initScore, 0) || math.IsNaN(tc.initScore) {
				// Model might error or handle it internally
				if err != nil {
					t.Logf("Expected error for %s: %v", tc.name, err)
				} else {
					// If no error, model should still work
					assert.True(t, clf.IsFitted())
				}
			} else {
				require.NoError(t, err)
				assert.True(t, clf.IsFitted())
				assert.Equal(t, tc.initScore, clf.Model.InitScore)
			}
		})
	}
}

// TestInitScoreConsistency tests that init_score produces consistent results
func TestInitScoreConsistency(t *testing.T) {
	// Create dataset
	X := mat.NewDense(20, 3, nil)
	y := mat.NewDense(20, 1, nil)

	source := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < 20; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, source.Rand())
		}
		y.Set(i, 0, math.Round(source.Rand()+0.5))
	}

	initScore := 0.5

	// Train multiple models with same init_score and deterministic mode
	models := make([]*LGBMClassifier, 3)
	for i := range models {
		models[i] = NewLGBMClassifier().
			WithNumIterations(10).
			WithNumLeaves(5).
			WithRandomState(42).
			WithDeterministic(true)

		err := models[i].FitWithInit(X, y, initScore, nil)
		require.NoError(t, err)
	}

	// Predictions should be very similar (allowing for minor numerical differences)
	pred1, _ := models[0].Predict(X)
	pred2, _ := models[1].Predict(X)
	pred3, _ := models[2].Predict(X)

	for i := 0; i < 20; i++ {
		p1 := pred1.At(i, 0)
		p2 := pred2.At(i, 0)
		p3 := pred3.At(i, 0)

		// Check consistency
		assert.InDelta(t, p1, p2, 1e-6, "Predictions should be consistent")
		assert.InDelta(t, p2, p3, 1e-6, "Predictions should be consistent")
	}
}
