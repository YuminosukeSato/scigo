package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestL2Objective(t *testing.T) {
	obj := NewL2Objective()

	t.Run("Gradient and Hessian", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
			expGrad    float64
			expHess    float64
		}{
			{prediction: 2.0, target: 1.0, expGrad: 1.0, expHess: 1.0},
			{prediction: 1.0, target: 2.0, expGrad: -1.0, expHess: 1.0},
			{prediction: 3.5, target: 3.5, expGrad: 0.0, expHess: 1.0},
			{prediction: -1.0, target: 1.0, expGrad: -2.0, expHess: 1.0},
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			assert.InDelta(t, tc.expGrad, grad, 1e-6,
				"Gradient mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
			assert.InDelta(t, tc.expHess, hess, 1e-6,
				"Hessian mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
		}
	})

	t.Run("Loss", func(t *testing.T) {
		loss := obj.CalculateLoss(3.0, 1.0)
		assert.InDelta(t, 2.0, loss, 1e-6) // 0.5 * (3-1)^2 = 2.0

		loss = obj.CalculateLoss(1.0, 1.0)
		assert.InDelta(t, 0.0, loss, 1e-6)
	})

	t.Run("InitScore", func(t *testing.T) {
		targets := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		initScore := obj.GetInitScore(targets)
		assert.InDelta(t, 3.0, initScore, 1e-6) // Mean of targets
	})
}

func TestL1Objective(t *testing.T) {
	obj := NewL1Objective()

	t.Run("Gradient and Hessian", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
			expGrad    float64
		}{
			{prediction: 2.0, target: 1.0, expGrad: 1.0},
			{prediction: 1.0, target: 2.0, expGrad: -1.0},
			{prediction: 3.5, target: 3.5, expGrad: 0.0},
			{prediction: -1.0, target: 1.0, expGrad: -1.0},
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			assert.InDelta(t, tc.expGrad, grad, 1e-6,
				"Gradient mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
			assert.True(t, hess > 0, "Hessian should be positive for numerical stability")
		}
	})

	t.Run("Loss", func(t *testing.T) {
		loss := obj.CalculateLoss(3.0, 1.0)
		assert.InDelta(t, 2.0, loss, 1e-6) // |3-1| = 2.0

		loss = obj.CalculateLoss(1.0, 1.0)
		assert.InDelta(t, 0.0, loss, 1e-6)
	})

	t.Run("InitScore", func(t *testing.T) {
		targets := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		initScore := obj.GetInitScore(targets)
		assert.InDelta(t, 3.0, initScore, 1e-6) // Median of targets
	})
}

func TestHuberObjective(t *testing.T) {
	delta := 1.5
	obj := NewHuberObjective(delta)

	t.Run("Gradient and Hessian", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
			desc       string
		}{
			{prediction: 1.0, target: 0.0, desc: "L2 region (|diff| < delta)"},
			{prediction: 3.0, target: 0.0, desc: "L1 region (|diff| > delta)"},
			{prediction: 0.0, target: 3.0, desc: "L1 region negative"},
			{prediction: 1.5, target: 0.0, desc: "Boundary case"},
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			diff := tc.prediction - tc.target
			absDiff := math.Abs(diff)

			if absDiff <= delta {
				// L2 region
				assert.InDelta(t, diff, grad, 1e-6, "L2 gradient: %s", tc.desc)
				assert.InDelta(t, 1.0, hess, 1e-6, "L2 hessian: %s", tc.desc)
			} else {
				// L1 region
				if diff > 0 {
					assert.InDelta(t, delta, grad, 1e-6, "L1 gradient positive: %s", tc.desc)
				} else {
					assert.InDelta(t, -delta, grad, 1e-6, "L1 gradient negative: %s", tc.desc)
				}
				assert.True(t, hess > 0 && hess < 1e-3, "L1 hessian small positive: %s", tc.desc)
			}
		}
	})

	t.Run("Loss", func(t *testing.T) {
		// L2 region
		loss := obj.CalculateLoss(1.0, 0.0)
		assert.InDelta(t, 0.5, loss, 1e-6) // 0.5 * 1^2 = 0.5

		// L1 region
		loss = obj.CalculateLoss(3.0, 0.0)
		assert.InDelta(t, delta*(3.0-0.5*delta), loss, 1e-6)
	})
}

func TestQuantileObjective(t *testing.T) {
	alpha := 0.75
	obj := NewQuantileObjective(alpha)

	t.Run("Gradient", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
			expGrad    float64
		}{
			{prediction: 2.0, target: 1.0, expGrad: alpha},       // Overestimate
			{prediction: 1.0, target: 2.0, expGrad: alpha - 1.0}, // Underestimate
			{prediction: 1.0, target: 1.0, expGrad: 0.0},         // Perfect
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			assert.InDelta(t, tc.expGrad, grad, 1e-6,
				"Gradient mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
		}
	})

	t.Run("Loss", func(t *testing.T) {
		// Overestimate
		loss := obj.CalculateLoss(3.0, 1.0)
		assert.InDelta(t, alpha*2.0, loss, 1e-6)

		// Underestimate
		loss = obj.CalculateLoss(1.0, 3.0)
		assert.InDelta(t, (alpha-1.0)*(-2.0), loss, 1e-6)
	})

	t.Run("InitScore", func(t *testing.T) {
		targets := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		initScore := obj.GetInitScore(targets)
		// 75th percentile of [1,2,3,4,5] should be 4
		assert.InDelta(t, 4.0, initScore, 0.5)
	})
}

func TestFairObjective(t *testing.T) {
	c := 1.0
	obj := NewFairObjective(c)

	t.Run("Gradient and Hessian", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
		}{
			{prediction: 2.0, target: 1.0},
			{prediction: 1.0, target: 2.0},
			{prediction: 3.0, target: 0.0},
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			diff := tc.prediction - tc.target
			absDiff := math.Abs(diff)

			// Check gradient formula
			expGrad := c * diff / (absDiff + c)
			assert.InDelta(t, expGrad, grad, 1e-6,
				"Gradient mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)

			// Check hessian is positive
			assert.True(t, hess > 0, "Hessian should be positive")

			// Check hessian formula
			denominator := absDiff + c
			expHess := c * c / (denominator * denominator)
			assert.InDelta(t, expHess, hess, 1e-6,
				"Hessian mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
		}
	})
}

func TestPoissonObjective(t *testing.T) {
	obj := NewPoissonObjective()

	t.Run("Gradient and Hessian", func(t *testing.T) {
		testCases := []struct {
			prediction float64
			target     float64
		}{
			{prediction: 0.0, target: 1.0},  // exp(0) = 1
			{prediction: 1.0, target: 2.0},  // exp(1) ≈ 2.718
			{prediction: -1.0, target: 0.5}, // exp(-1) ≈ 0.368
		}

		for _, tc := range testCases {
			grad := obj.CalculateGradient(tc.prediction, tc.target)
			hess := obj.CalculateHessian(tc.prediction, tc.target)

			expPred := math.Exp(tc.prediction)
			expGrad := expPred - tc.target
			expHess := expPred

			assert.InDelta(t, expGrad, grad, 1e-3,
				"Gradient mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
			assert.InDelta(t, expHess, hess, 1e-3,
				"Hessian mismatch for pred=%.2f, target=%.2f", tc.prediction, tc.target)
		}
	})

	t.Run("InitScore", func(t *testing.T) {
		targets := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		initScore := obj.GetInitScore(targets)
		// log(mean) = log(3) ≈ 1.099
		assert.InDelta(t, math.Log(3.0), initScore, 1e-6)
	})
}

func TestCreateObjectiveFunction(t *testing.T) {
	params := &TrainingParams{
		HuberDelta:    2.0,
		QuantileAlpha: 0.9,
		FairC:         1.5,
	}

	testCases := []struct {
		objective string
		expected  string
	}{
		{"regression", "regression"},
		{"regression_l2", "regression"},
		{"mse", "regression"},
		{"regression_l1", "regression_l1"},
		{"mae", "regression_l1"},
		{"huber", "huber"},
		{"fair", "fair"},
		{"poisson", "poisson"},
		{"quantile", "quantile"},
	}

	for _, tc := range testCases {
		obj, err := CreateObjectiveFunction(tc.objective, params)
		assert.NoError(t, err, "Failed to create objective: %s", tc.objective)
		assert.Equal(t, tc.expected, obj.Name(), "Objective name mismatch for %s", tc.objective)
	}

	// Test unknown objective
	_, err := CreateObjectiveFunction("unknown_objective", params)
	assert.Error(t, err)
}

func TestObjectivesWithTrainer(t *testing.T) {
	// Create simple training data
	X := mat.NewDense(100, 3, nil)
	y := mat.NewDense(100, 1, nil)
	for i := 0; i < 100; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
		y.Set(i, 0, float64(i)*0.1+1.0)
	}

	objectives := []string{
		"regression",
		"regression_l1",
		"huber",
		"quantile",
		"fair",
	}

	for _, objective := range objectives {
		t.Run(objective, func(t *testing.T) {
			params := TrainingParams{
				NumIterations: 5,
				LearningRate:  0.1,
				NumLeaves:     10,
				MaxDepth:      3,
				MinDataInLeaf: 5,
				Objective:     objective,
				HuberDelta:    1.0,
				QuantileAlpha: 0.5,
				FairC:         1.0,
			}

			trainer := NewTrainer(params)
			err := trainer.Fit(X, y)
			assert.NoError(t, err, "Training failed for objective: %s", objective)

			model := trainer.GetModel()
			assert.NotNil(t, model)
			assert.Equal(t, 5, len(model.Trees), "Expected 5 trees for objective: %s", objective)

			// Make predictions
			pred, err := model.Predict(X)
			assert.NoError(t, err, "Prediction failed for objective: %s", objective)
			assert.NotNil(t, pred)
		})
	}
}

func TestMedianCalculation(t *testing.T) {
	testCases := []struct {
		values   []float64
		expected float64
	}{
		{[]float64{1, 2, 3, 4, 5}, 3.0},
		{[]float64{1, 2, 3, 4}, 2.5},
		{[]float64{5, 1, 3, 2, 4}, 3.0},
		{[]float64{1}, 1.0},
		{[]float64{}, 0.0},
	}

	for _, tc := range testCases {
		result := calculateMedian(tc.values)
		assert.InDelta(t, tc.expected, result, 1e-6,
			"Median mismatch for values: %v", tc.values)
	}
}

func TestQuantileCalculation(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5}

	testCases := []struct {
		quantile float64
		expected float64
	}{
		{0.0, 1.0},
		{0.25, 2.0},
		{0.5, 3.0},
		{0.75, 4.0},
		{1.0, 5.0},
	}

	for _, tc := range testCases {
		result := calculateQuantile(values, tc.quantile)
		assert.InDelta(t, tc.expected, result, 0.5,
			"Quantile %.2f mismatch", tc.quantile)
	}
}
