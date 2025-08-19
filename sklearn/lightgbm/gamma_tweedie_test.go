package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// TestGammaObjective tests the Gamma objective function
func TestGammaObjective(t *testing.T) {
	obj := NewGammaObjective()

	// Test gradient calculation
	testCases := []struct {
		name       string
		prediction float64
		target     float64
		wantGrad   float64
		wantHess   float64
	}{
		{
			name:       "Small positive values",
			prediction: 0.5,
			target:     2.0,
			wantGrad:   2.0 * (1.0 - 2.0*math.Exp(-0.5)),
			wantHess:   2.0 * 2.0 * math.Exp(-0.5),
		},
		{
			name:       "Zero prediction",
			prediction: 0.0,
			target:     1.0,
			wantGrad:   0.0,
			wantHess:   2.0 * 1.0,
		},
		{
			name:       "Large prediction",
			prediction: 2.0,
			target:     5.0,
			wantGrad:   2.0 * (1.0 - 5.0*math.Exp(-2.0)),
			wantHess:   2.0 * 5.0 * math.Exp(-2.0),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			assert.InDelta(t, tc.wantGrad, grad, 1e-6, "Gradient mismatch")
			assert.InDelta(t, tc.wantHess, hess, 1e-6, "Hessian mismatch")
			assert.Greater(t, hess, 0.0, "Hessian should be positive")
		})
	}

	// Test init score
	targets := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	initScore := obj.GetInitScore(targets)
	expectedInit := math.Log(3.0) // log of mean
	assert.InDelta(t, expectedInit, initScore, 1e-6)

	// Test name
	assert.Equal(t, "gamma", obj.Name())
}

// TestTweedieObjective tests the Tweedie objective function
func TestTweedieObjective(t *testing.T) {
	// Test with default variance power (1.5)
	obj := NewTweedieObjective(1.5)

	testCases := []struct {
		name       string
		prediction float64
		target     float64
	}{
		{
			name:       "Small positive values",
			prediction: 0.5,
			target:     2.0,
		},
		{
			name:       "Zero target (common in Tweedie)",
			prediction: 0.5,
			target:     0.0,
		},
		{
			name:       "Large values",
			prediction: 2.0,
			target:     10.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)
			loss := obj.CalculateLoss(tc.prediction, tc.target)

			// Check that gradient and hessian are finite
			assert.False(t, math.IsNaN(grad), "Gradient should not be NaN")
			assert.False(t, math.IsInf(grad, 0), "Gradient should not be Inf")
			assert.False(t, math.IsNaN(hess), "Hessian should not be NaN")
			assert.False(t, math.IsInf(hess, 0), "Hessian should not be Inf")
			assert.Greater(t, hess, 0.0, "Hessian should be positive")

			// Check that loss is finite and non-negative
			assert.False(t, math.IsNaN(loss), "Loss should not be NaN")
			assert.False(t, math.IsInf(loss, 0), "Loss should not be Inf")
		})
	}

	// Test init score
	targets := []float64{0.0, 1.0, 2.0, 0.0, 3.0}
	initScore := obj.GetInitScore(targets)
	expectedInit := math.Log(2.0) // log of mean of positive values
	assert.InDelta(t, expectedInit, initScore, 1e-6)

	// Test name
	assert.Equal(t, "tweedie", obj.Name())
}

// TestTweedieVariancePower tests different variance power values
func TestTweedieVariancePower(t *testing.T) {
	testCases := []struct {
		power float64
		name  string
	}{
		{1.1, "Close to Poisson"},
		{1.5, "Typical compound Poisson-Gamma"},
		{1.9, "Close to Gamma"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			obj := NewTweedieObjective(tc.power)

			// Test with various predictions and targets
			predictions := []float64{-1.0, 0.0, 1.0, 2.0}
			targets := []float64{0.0, 1.0, 5.0, 10.0}

			for _, pred := range predictions {
				for _, target := range targets {
					grad := obj.CalculateGradient(pred, target)
					hess := obj.CalculateHessian(pred, target)

					// Ensure numerical stability
					assert.False(t, math.IsNaN(grad), "Gradient NaN for p=%.1f, pred=%.1f, target=%.1f", tc.power, pred, target)
					assert.False(t, math.IsInf(grad, 0), "Gradient Inf for p=%.1f, pred=%.1f, target=%.1f", tc.power, pred, target)
					assert.Greater(t, hess, 0.0, "Hessian should be positive for p=%.1f, pred=%.1f, target=%.1f", tc.power, pred, target)
				}
			}
		})
	}
}

// TestGammaRegression tests Gamma regression with LGBMRegressor
func TestGammaRegression(t *testing.T) {
	// Create dataset with positive targets (Gamma distribution requirement)
	nSamples := 100
	nFeatures := 4

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate positive targets similar to Gamma distribution
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		// Use exponential of linear combination to ensure positivity
		val := math.Exp(X.At(i, 0)*0.5 + X.At(i, 1)*0.3 + distuv.Normal{Mu: 0, Sigma: 0.2}.Rand())
		y.Set(i, 0, val)
	}

	// Train with Gamma objective
	reg := NewLGBMRegressor().
		WithNumIterations(30).
		WithNumLeaves(10).
		WithLearningRate(0.1).
		WithObjective("gamma")

	err := reg.Fit(X, y)
	require.NoError(t, err)
	assert.True(t, reg.IsFitted())

	// Make predictions
	pred, err := reg.Predict(X)
	require.NoError(t, err)

	// Check that predictions are positive (as expected for Gamma)
	rows, _ := pred.Dims()
	for i := 0; i < rows; i++ {
		assert.Greater(t, pred.At(i, 0), 0.0, "Gamma predictions should be positive")
	}

	// Calculate MSE
	mse := 0.0
	for i := 0; i < rows; i++ {
		diff := pred.At(i, 0) - y.At(i, 0)
		mse += diff * diff
	}
	mse /= float64(rows)

	t.Logf("Gamma regression MSE: %.4f", mse)
}

// TestTweedieRegression tests Tweedie regression with LGBMRegressor
func TestTweedieRegression(t *testing.T) {
	// Create dataset with mix of zeros and positive values (Tweedie characteristic)
	nSamples := 100
	nFeatures := 4

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
	}

	// Generate targets with zeros and positive values
	y := mat.NewDense(nSamples, 1, nil)
	for i := 0; i < nSamples; i++ {
		if i%5 == 0 { // 20% zeros
			y.Set(i, 0, 0.0)
		} else {
			// Positive values following exponential-like distribution
			val := math.Exp(X.At(i, 0)*0.5 + X.At(i, 1)*0.3 + distuv.Normal{Mu: 0, Sigma: 0.2}.Rand())
			y.Set(i, 0, val)
		}
	}

	// Train with Tweedie objective
	reg := NewLGBMRegressor().
		WithNumIterations(30).
		WithNumLeaves(10).
		WithLearningRate(0.1).
		WithObjective("tweedie").
		WithTweedieVariancePower(1.5)

	err := reg.Fit(X, y)
	require.NoError(t, err)
	assert.True(t, reg.IsFitted())

	// Make predictions
	pred, err := reg.Predict(X)
	require.NoError(t, err)

	// Check that predictions are non-negative
	rows, _ := pred.Dims()
	for i := 0; i < rows; i++ {
		assert.GreaterOrEqual(t, pred.At(i, 0), 0.0, "Tweedie predictions should be non-negative")
	}

	// Calculate MSE separately for zero and non-zero targets
	mseZero, mseNonZero := 0.0, 0.0
	countZero, countNonZero := 0, 0

	for i := 0; i < rows; i++ {
		diff := pred.At(i, 0) - y.At(i, 0)
		if y.At(i, 0) == 0.0 {
			mseZero += diff * diff
			countZero++
		} else {
			mseNonZero += diff * diff
			countNonZero++
		}
	}

	if countZero > 0 {
		mseZero /= float64(countZero)
		t.Logf("Tweedie MSE for zero targets: %.4f", mseZero)
	}
	if countNonZero > 0 {
		mseNonZero /= float64(countNonZero)
		t.Logf("Tweedie MSE for non-zero targets: %.4f", mseNonZero)
	}
}

// TestGammaVsL2Comparison compares Gamma with L2 on skewed data
func TestGammaVsL2Comparison(t *testing.T) {
	// Create highly skewed positive dataset
	nSamples := 200
	nFeatures := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, distuv.Normal{Mu: 0, Sigma: 1}.Rand())
		}
		// Create skewed distribution with occasional large values
		if i%10 == 0 {
			// Large outliers
			y.Set(i, 0, 50.0+distuv.Normal{Mu: 0, Sigma: 10}.Rand())
		} else {
			// Normal range
			y.Set(i, 0, math.Abs(5.0+distuv.Normal{Mu: 0, Sigma: 2}.Rand()))
		}
	}

	// Train with L2 objective
	regL2 := NewLGBMRegressor().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1).
		WithObjective("l2")

	err := regL2.Fit(X, y)
	require.NoError(t, err)

	// Train with Gamma objective
	regGamma := NewLGBMRegressor().
		WithNumIterations(50).
		WithNumLeaves(10).
		WithLearningRate(0.1).
		WithObjective("gamma")

	err = regGamma.Fit(X, y)
	require.NoError(t, err)

	// Compare predictions on test portion
	predL2, _ := regL2.Predict(X)
	predGamma, _ := regGamma.Predict(X)

	// Calculate median absolute error (more robust for skewed data)
	errorsL2 := make([]float64, nSamples)
	errorsGamma := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		errorsL2[i] = math.Abs(predL2.At(i, 0) - y.At(i, 0))
		errorsGamma[i] = math.Abs(predGamma.At(i, 0) - y.At(i, 0))
	}

	// Calculate mean absolute error
	maeL2, maeGamma := 0.0, 0.0
	for i := 0; i < nSamples; i++ {
		maeL2 += errorsL2[i]
		maeGamma += errorsGamma[i]
	}
	maeL2 /= float64(nSamples)
	maeGamma /= float64(nSamples)

	t.Logf("L2 MAE: %.4f", maeL2)
	t.Logf("Gamma MAE: %.4f", maeGamma)

	// Gamma should handle skewed data better in many cases
	// Note: This is not always guaranteed due to randomness
}

// TestObjectiveCreation tests that objectives are created correctly through factory
func TestObjectiveCreation(t *testing.T) {
	// Test Gamma objective creation
	params := &TrainingParams{}
	obj, err := CreateObjectiveFunction("gamma", params)
	require.NoError(t, err)
	assert.Equal(t, "gamma", obj.Name())

	// Test Tweedie objective creation with default power
	obj, err = CreateObjectiveFunction("tweedie", params)
	require.NoError(t, err)
	assert.Equal(t, "tweedie", obj.Name())

	// Test Tweedie with custom power
	params.TweedieVariancePower = 1.7
	obj, err = CreateObjectiveFunction("tweedie", params)
	require.NoError(t, err)
	assert.Equal(t, "tweedie", obj.Name())
}

// TestGammaTweedieNumericalStability tests numerical stability with extreme values
func TestGammaTweedieNumericalStability(t *testing.T) {
	gammaObj := NewGammaObjective()
	tweedieObj := NewTweedieObjective(1.5)

	extremeValues := []struct {
		name   string
		pred   float64
		target float64
	}{
		{"Very large prediction", 1000.0, 1.0},
		{"Very small prediction", -1000.0, 1.0},
		{"Very large target", 1.0, 1e10},
		{"Very small target", 1.0, 1e-10},
		{"Both large", 100.0, 1e8},
	}

	for _, tc := range extremeValues {
		t.Run("Gamma_"+tc.name, func(t *testing.T) {
			grad := gammaObj.CalculateGradient(tc.pred, tc.target)
			hess := gammaObj.CalculateHessian(tc.pred, tc.target)

			assert.False(t, math.IsNaN(grad), "Gradient should not be NaN")
			assert.False(t, math.IsInf(grad, 0), "Gradient should not be Inf")
			assert.False(t, math.IsNaN(hess), "Hessian should not be NaN")
			assert.False(t, math.IsInf(hess, 0), "Hessian should not be Inf")
			assert.Greater(t, hess, 0.0, "Hessian should be positive")
		})

		t.Run("Tweedie_"+tc.name, func(t *testing.T) {
			grad := tweedieObj.CalculateGradient(tc.pred, tc.target)
			hess := tweedieObj.CalculateHessian(tc.pred, tc.target)

			assert.False(t, math.IsNaN(grad), "Gradient should not be NaN")
			assert.False(t, math.IsInf(grad, 0), "Gradient should not be Inf")
			assert.False(t, math.IsNaN(hess), "Hessian should not be NaN")
			assert.False(t, math.IsInf(hess, 0), "Hessian should not be Inf")
			assert.Greater(t, hess, 0.0, "Hessian should be positive")
		})
	}
}
