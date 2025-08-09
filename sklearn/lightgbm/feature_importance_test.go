package lightgbm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestFeatureImportanceGain(t *testing.T) {
	// Create synthetic data where feature 0 should be most important
	// Feature 0: Strongly correlated with target
	// Feature 1: Weakly correlated
	// Feature 2: Noise
	rows := 100
	X := mat.NewDense(rows, 3, nil)
	y := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		// Feature 0: strong signal
		f0 := float64(i) / float64(rows)
		X.Set(i, 0, f0)

		// Feature 1: weak signal
		f1 := float64(i%10) / 10.0
		X.Set(i, 1, f1)

		// Feature 2: random noise
		f2 := float64((i*7)%13) / 13.0
		X.Set(i, 2, f2)

		// Target is mainly determined by feature 0
		target := 2.0*f0 + 0.5*f1 + 0.1*f2
		y.Set(i, 0, target)
	}

	// Train model
	reg := NewLGBMRegressor()
	reg.SetParams(map[string]interface{}{
		"n_estimators":  10,
		"num_leaves":    15,
		"learning_rate": 0.1,
	})

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Get feature importance
	importance := reg.GetFeatureImportance("gain")
	assert.Equal(t, 3, len(importance))

	// Feature 0 should have highest importance
	assert.Greater(t, importance[0], importance[1],
		"Feature 0 should be more important than Feature 1")
	assert.Greater(t, importance[0], importance[2],
		"Feature 0 should be more important than Feature 2")

	// Feature 1 should have higher importance than noise feature
	assert.Greater(t, importance[1], importance[2],
		"Feature 1 should be more important than Feature 2 (noise)")

	// All importances should be non-negative and sum to 1
	sum := 0.0
	for _, imp := range importance {
		assert.GreaterOrEqual(t, imp, 0.0, "Importance should be non-negative")
		sum += imp
	}
	assert.InDelta(t, 1.0, sum, 1e-9, "Importances should sum to 1")
}

func TestFeatureImportanceSplit(t *testing.T) {
	// Create simple data
	X := mat.NewDense(50, 4, nil)
	y := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		for j := 0; j < 4; j++ {
			X.Set(i, j, float64(i*j)*0.1)
		}
		y.Set(i, 0, float64(i))
	}

	// Train model
	reg := NewLGBMRegressor()
	reg.SetParams(map[string]interface{}{
		"n_estimators": 5,
		"num_leaves":   10,
	})

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Get feature importance by split count
	importance := reg.GetFeatureImportance("split")
	assert.Equal(t, 4, len(importance))

	// All importances should be non-negative and sum to 1
	sum := 0.0
	for _, imp := range importance {
		assert.GreaterOrEqual(t, imp, 0.0, "Importance should be non-negative")
		sum += imp
	}
	assert.InDelta(t, 1.0, sum, 1e-9, "Importances should sum to 1")
}

func TestFeatureImportanceEmptyModel(t *testing.T) {
	// Test with untrained model
	reg := NewLGBMRegressor()

	importance := reg.GetFeatureImportance("gain")
	assert.Nil(t, importance, "Untrained model should return nil importance")
}

func TestFeatureImportanceClassification(t *testing.T) {
	// Create binary classification data
	X := mat.NewDense(60, 3, nil)
	y := mat.NewDense(60, 1, nil)

	for i := 0; i < 60; i++ {
		// Feature 0: most important for classification
		f0 := float64(i) / 60.0
		X.Set(i, 0, f0)

		// Feature 1: somewhat important
		f1 := float64(i%5) / 5.0
		X.Set(i, 1, f1)

		// Feature 2: not important
		f2 := float64((i*3)%7) / 7.0
		X.Set(i, 2, f2)

		// Binary target based mainly on feature 0
		if f0 > 0.5 {
			y.Set(i, 0, 1.0)
		} else {
			y.Set(i, 0, 0.0)
		}
	}

	// Train classifier
	clf := NewLGBMClassifier()
	clf.SetParams(map[string]interface{}{
		"n_estimators": 5,
		"num_leaves":   8,
	})

	err := clf.Fit(X, y)
	require.NoError(t, err)

	// Get feature importance
	importance := clf.GetFeatureImportance("gain")
	assert.Equal(t, 3, len(importance))

	// Feature 0 should have highest importance for classification
	assert.Greater(t, importance[0], importance[1],
		"Feature 0 should be most important for classification")
	assert.Greater(t, importance[0], importance[2],
		"Feature 0 should be more important than noise feature")
}

func TestFeatureImportanceInvalidType(t *testing.T) {
	// Train simple model
	X := mat.NewDense(20, 2, nil)
	y := mat.NewDense(20, 1, nil)
	for i := 0; i < 20; i++ {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i)*0.5)
		y.Set(i, 0, float64(i))
	}

	reg := NewLGBMRegressor()
	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Test invalid importance type
	importance := reg.GetFeatureImportance("invalid_type")

	// Should return zeros or handle gracefully
	assert.Equal(t, 2, len(importance))
	for _, imp := range importance {
		assert.Equal(t, 0.0, imp, "Invalid type should return zero importance")
	}
}

func TestFeatureImportanceConsistency(t *testing.T) {
	// Create data
	X := mat.NewDense(40, 3, nil)
	y := mat.NewDense(40, 1, nil)

	for i := 0; i < 40; i++ {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i)*0.5)
		X.Set(i, 2, float64(i)*0.1)
		y.Set(i, 0, float64(i)*2.0)
	}

	// Train model with same parameters twice
	params := map[string]interface{}{
		"n_estimators":  5,
		"num_leaves":    10,
		"random_state":  42,
		"deterministic": true,
	}

	reg1 := NewLGBMRegressor()
	reg1.SetParams(params)
	err := reg1.Fit(X, y)
	require.NoError(t, err)

	reg2 := NewLGBMRegressor()
	reg2.SetParams(params)
	err = reg2.Fit(X, y)
	require.NoError(t, err)

	// Feature importance should be identical for deterministic training
	importance1 := reg1.GetFeatureImportance("gain")
	importance2 := reg2.GetFeatureImportance("gain")

	require.Equal(t, len(importance1), len(importance2))
	for i := range importance1 {
		assert.InDelta(t, importance1[i], importance2[i], 1e-9,
			"Feature importance should be consistent for deterministic training")
	}
}

// Benchmark feature importance calculation
func BenchmarkFeatureImportance(b *testing.B) {
	// Create larger dataset
	rows := 1000
	cols := 50
	X := mat.NewDense(rows, cols, nil)
	y := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
		y.Set(i, 0, float64(i))
	}

	reg := NewLGBMRegressor()
	reg.SetParams(map[string]interface{}{
		"n_estimators": 20,
		"num_leaves":   31,
	})
	_ = reg.Fit(X, y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = reg.GetFeatureImportance("gain")
	}
}
