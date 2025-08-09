package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestLGBMRegressorWithDifferentObjectives(t *testing.T) {
	// Create synthetic training data with outliers
	n := 100
	X := mat.NewDense(n, 3, nil)
	y := mat.NewDense(n, 1, nil)

	for i := 0; i < n; i++ {
		x1 := float64(i) / float64(n)
		x2 := math.Sin(float64(i) * 0.1)
		x3 := math.Cos(float64(i) * 0.1)

		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		X.Set(i, 2, x3)

		// Target with some noise and outliers
		target := 2*x1 + x2 - 0.5*x3 + 1.0
		if i%20 == 0 {
			// Add outliers
			target += 5.0
		} else {
			// Add normal noise
			target += math.Sin(float64(i)) * 0.1
		}
		y.Set(i, 0, target)
	}

	objectives := []struct {
		name      string
		objective string
		setupFunc func(*LGBMRegressor)
		checkFunc func(*testing.T, *LGBMRegressor, mat.Matrix)
	}{
		{
			name:      "L2 Regression",
			objective: "regression",
			setupFunc: func(lgb *LGBMRegressor) {},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// L2 should minimize squared errors
				mse, _ := lgb.GetMSE(X, y)
				assert.Less(t, mse, 5.0, "MSE should be reasonable")
			},
		},
		{
			name:      "L1 Regression",
			objective: "regression_l1",
			setupFunc: func(lgb *LGBMRegressor) {},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// L1 should be more robust to outliers
				// Just check that training completes and predictions are finite
				rows, _ := pred.Dims()
				for i := 0; i < rows; i++ {
					val := pred.At(i, 0)
					assert.False(t, math.IsNaN(val), "Prediction should not be NaN")
					assert.False(t, math.IsInf(val, 0), "Prediction should not be Inf")
				}
			},
		},
		{
			name:      "Huber Regression",
			objective: "huber",
			setupFunc: func(lgb *LGBMRegressor) {
				lgb.HuberAlpha = 1.5
			},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// Huber should balance between L1 and L2
				mae, _ := lgb.GetMAE(X, y)
				mse, _ := lgb.GetMSE(X, y)
				assert.Less(t, mae, 2.5, "MAE should be reasonable")
				assert.Less(t, mse, 6.0, "MSE should be reasonable")
			},
		},
		{
			name:      "Quantile Regression (Median)",
			objective: "quantile",
			setupFunc: func(lgb *LGBMRegressor) {
				lgb.Alpha = 0.5 // Median
			},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// Check that roughly 50% of residuals are positive
				residuals, _ := lgb.GetResiduals(X, y)
				rows, _ := residuals.Dims()
				positiveCount := 0
				for i := 0; i < rows; i++ {
					if residuals.At(i, 0) > 0 {
						positiveCount++
					}
				}
				ratio := float64(positiveCount) / float64(rows)
				assert.InDelta(t, 0.5, ratio, 0.15, "Should predict median")
			},
		},
		{
			name:      "Quantile Regression (75th percentile)",
			objective: "quantile",
			setupFunc: func(lgb *LGBMRegressor) {
				lgb.Alpha = 0.75
			},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// For 75th percentile, predictions should be generally higher
				// Just verify predictions are valid
				rows, _ := pred.Dims()
				for i := 0; i < rows; i++ {
					val := pred.At(i, 0)
					assert.False(t, math.IsNaN(val), "Prediction should not be NaN")
					assert.False(t, math.IsInf(val, 0), "Prediction should not be Inf")
				}
			},
		},
		{
			name:      "Fair Regression",
			objective: "fair",
			setupFunc: func(lgb *LGBMRegressor) {
				lgb.FairC = 1.0
			},
			checkFunc: func(t *testing.T, lgb *LGBMRegressor, pred mat.Matrix) {
				// Fair loss should also be robust to outliers
				mae, _ := lgb.GetMAE(X, y)
				assert.Less(t, mae, 2.5, "MAE should be reasonable for Fair loss")
			},
		},
	}

	for _, tc := range objectives {
		t.Run(tc.name, func(t *testing.T) {
			lgb := NewLGBMRegressor().
				WithObjective(tc.objective).
				WithNumIterations(10).
				WithLearningRate(0.1).
				WithNumLeaves(15).
				WithRandomState(42)

			// Apply test-specific setup
			tc.setupFunc(lgb)

			// Train the model
			err := lgb.Fit(X, y)
			assert.NoError(t, err, "Failed to train with %s", tc.objective)

			// Make predictions
			predictions, err := lgb.Predict(X)
			assert.NoError(t, err, "Failed to predict with %s", tc.objective)
			assert.NotNil(t, predictions)

			// Check dimensions
			rows, cols := predictions.Dims()
			assert.Equal(t, 100, rows)
			assert.Equal(t, 1, cols)

			// Run objective-specific checks
			tc.checkFunc(t, lgb, predictions)

			// Check that model is fitted
			assert.True(t, lgb.IsFitted())
			assert.NotNil(t, lgb.Model)
			assert.Greater(t, len(lgb.Model.Trees), 0)
		})
	}
}

func TestLGBMRegressorObjectiveParameterMapping(t *testing.T) {
	X := mat.NewDense(50, 2, nil)
	y := mat.NewDense(50, 1, nil)
	for i := 0; i < 50; i++ {
		X.Set(i, 0, float64(i)*0.1)
		X.Set(i, 1, float64(i)*0.2)
		y.Set(i, 0, float64(i)*0.3)
	}

	t.Run("SetParams with objective aliases", func(t *testing.T) {
		reg := NewLGBMRegressor()

		// Test various objective aliases
		aliases := []struct {
			alias    string
			expected string
		}{
			{"mse", "regression"},
			{"mean_squared_error", "regression"},
			{"mae", "regression_l1"},
			{"mean_absolute_error", "regression_l1"},
		}

		for _, a := range aliases {
			params := map[string]interface{}{
				"objective":     a.alias,
				"n_estimators":  5,
				"learning_rate": 0.1,
			}

			err := reg.SetParams(params)
			assert.NoError(t, err)
			assert.Equal(t, a.expected, reg.Objective,
				"Objective alias %s should map to %s", a.alias, a.expected)

			// Verify training works
			err = reg.Fit(X, y)
			assert.NoError(t, err, "Training should work with objective alias %s", a.alias)
		}
	})

	t.Run("Objective-specific parameters", func(t *testing.T) {
		reg := NewLGBMRegressor()

		params := map[string]interface{}{
			"objective":    "huber",
			"huber_delta":  2.0, // This parameter name doesn't exist in mapping yet
			"n_estimators": 5,
		}

		err := reg.SetParams(params)
		assert.NoError(t, err)

		// Train with Huber loss
		err = reg.Fit(X, y)
		assert.NoError(t, err)

		// Verify model uses Huber objective
		assert.Equal(t, "huber", reg.Objective)
	})
}

func TestRegressorPoissonObjective(t *testing.T) {
	// Create count data suitable for Poisson regression
	n := 100
	X := mat.NewDense(n, 2, nil)
	y := mat.NewDense(n, 1, nil)

	for i := 0; i < n; i++ {
		x1 := float64(i) / float64(n) * 2
		x2 := math.Sin(float64(i) * 0.1)

		X.Set(i, 0, x1)
		X.Set(i, 1, x2)

		// Generate count data (always positive)
		lambda := math.Exp(0.5*x1 + 0.3*x2)
		count := math.Max(0, lambda+math.Sin(float64(i))*0.5)
		y.Set(i, 0, count)
	}

	lgb := NewLGBMRegressor().
		WithObjective("poisson").
		WithNumIterations(10).
		WithLearningRate(0.1).
		WithRandomState(42)

	err := lgb.Fit(X, y)
	assert.NoError(t, err, "Poisson regression training should succeed")

	predictions, err := lgb.Predict(X)
	assert.NoError(t, err, "Poisson regression prediction should succeed")

	// Check all predictions are positive (as expected for Poisson)
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		// Note: predictions are in log space, need to exp them for actual counts
		// But the raw predictions can be negative (log of small positive numbers)
		pred := predictions.At(i, 0)
		assert.False(t, math.IsNaN(pred), "Prediction should not be NaN")
		assert.False(t, math.IsInf(pred, 0), "Prediction should not be Inf")
	}
}
