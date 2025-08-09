package lightgbm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestParameterMapping(t *testing.T) {
	mapper := NewParameterMapper()

	t.Run("Python to Go mapping", func(t *testing.T) {
		pythonParams := map[string]interface{}{
			"n_estimators":      100,
			"learning_rate":     0.1,
			"num_leaves":        31,
			"max_depth":         -1,
			"min_child_samples": 20,
			"reg_alpha":         0.1,
			"reg_lambda":        0.2,
			"subsample":         0.8,
			"colsample_bytree":  0.9,
			"random_state":      42,
		}

		goParams := mapper.MapPythonToGo(pythonParams)

		assert.Equal(t, 100, goParams["NumIterations"])
		assert.Equal(t, 0.1, goParams["LearningRate"])
		assert.Equal(t, 31, goParams["NumLeaves"])
		assert.Equal(t, -1, goParams["MaxDepth"])
		assert.Equal(t, 20, goParams["MinDataInLeaf"])
		assert.Equal(t, 0.1, goParams["Alpha"])
		assert.Equal(t, 0.2, goParams["Lambda"])
		assert.Equal(t, 0.8, goParams["BaggingFraction"])
		assert.Equal(t, 0.9, goParams["FeatureFraction"])
		assert.Equal(t, 42, goParams["Seed"])
	})

	t.Run("Alias mapping", func(t *testing.T) {
		pythonParams := map[string]interface{}{
			"num_iterations":   50,   // Alias for n_estimators
			"shrinkage_rate":   0.05, // Alias for learning_rate
			"num_leaf":         15,   // Alias for num_leaves
			"lambda_l1":        0.01, // Alias for reg_alpha
			"lambda_l2":        0.02, // Alias for reg_lambda
			"bagging_fraction": 0.7,  // Alias for subsample
			"feature_fraction": 0.8,  // Alias for colsample_bytree
			"seed":             123,  // Alias for random_state
		}

		goParams := mapper.MapPythonToGo(pythonParams)

		assert.Equal(t, 50, goParams["NumIterations"])
		assert.Equal(t, 0.05, goParams["LearningRate"])
		assert.Equal(t, 15, goParams["NumLeaves"])
		assert.Equal(t, 0.01, goParams["Alpha"])
		assert.Equal(t, 0.02, goParams["Lambda"])
		assert.Equal(t, 0.7, goParams["BaggingFraction"])
		assert.Equal(t, 0.8, goParams["FeatureFraction"])
		assert.Equal(t, 123, goParams["Seed"])
	})

	t.Run("Go to Python mapping", func(t *testing.T) {
		goParams := map[string]interface{}{
			"NumIterations":   100,
			"LearningRate":    0.1,
			"NumLeaves":       31,
			"MaxDepth":        -1,
			"MinDataInLeaf":   20,
			"Alpha":           0.1,
			"Lambda":          0.2,
			"BaggingFraction": 0.8,
			"FeatureFraction": 0.9,
			"Seed":            42,
		}

		pythonParams := mapper.MapGoToPython(goParams)

		assert.Equal(t, 100, pythonParams["n_estimators"])
		assert.Equal(t, 0.1, pythonParams["learning_rate"])
		assert.Equal(t, 31, pythonParams["num_leaves"])
		assert.Equal(t, -1, pythonParams["max_depth"])
		assert.Equal(t, 20, pythonParams["min_child_samples"])
		assert.Equal(t, 0.1, pythonParams["reg_alpha"])
		assert.Equal(t, 0.2, pythonParams["reg_lambda"])
		assert.Equal(t, 0.8, pythonParams["subsample"])
		assert.Equal(t, 0.9, pythonParams["colsample_bytree"])
		assert.Equal(t, 42, pythonParams["random_state"])
	})

	t.Run("Default values", func(t *testing.T) {
		// Test getting defaults
		val, ok := mapper.GetDefault("n_estimators")
		assert.True(t, ok)
		assert.Equal(t, 100, val)

		val, ok = mapper.GetDefault("learning_rate")
		assert.True(t, ok)
		assert.Equal(t, 0.1, val)

		val, ok = mapper.GetDefault("num_leaves")
		assert.True(t, ok)
		assert.Equal(t, 31, val)

		// Test alias defaults
		val, ok = mapper.GetDefault("num_iterations")
		assert.True(t, ok)
		assert.Equal(t, 100, val)

		// Test non-existent parameter
		_, ok = mapper.GetDefault("non_existent_param")
		assert.False(t, ok)
	})

	t.Run("Objective validation", func(t *testing.T) {
		// Valid objectives
		obj, err := mapper.ValidateObjective("regression")
		assert.NoError(t, err)
		assert.Equal(t, "regression", obj)

		obj, err = mapper.ValidateObjective("binary")
		assert.NoError(t, err)
		assert.Equal(t, "binary", obj)

		obj, err = mapper.ValidateObjective("multiclass")
		assert.NoError(t, err)
		assert.Equal(t, "multiclass", obj)

		// Aliases
		obj, err = mapper.ValidateObjective("mse")
		assert.NoError(t, err)
		assert.Equal(t, "regression", obj)

		obj, err = mapper.ValidateObjective("binary_logloss")
		assert.NoError(t, err)
		assert.Equal(t, "binary", obj)

		// Invalid objective
		_, err = mapper.ValidateObjective("invalid_objective")
		assert.Error(t, err)
	})

	t.Run("Boosting type validation", func(t *testing.T) {
		// Valid types
		boostType, err := mapper.ValidateBoostingType("gbdt")
		assert.NoError(t, err)
		assert.Equal(t, "gbdt", boostType)

		boostType, err = mapper.ValidateBoostingType("dart")
		assert.NoError(t, err)
		assert.Equal(t, "dart", boostType)

		boostType, err = mapper.ValidateBoostingType("goss")
		assert.NoError(t, err)
		assert.Equal(t, "goss", boostType)

		boostType, err = mapper.ValidateBoostingType("rf")
		assert.NoError(t, err)
		assert.Equal(t, "rf", boostType)

		// Invalid type
		_, err = mapper.ValidateBoostingType("invalid_type")
		assert.Error(t, err)
	})

	t.Run("Apply defaults", func(t *testing.T) {
		params := map[string]interface{}{
			"n_estimators": 50,
			"max_depth":    10,
		}

		paramsWithDefaults := mapper.ApplyDefaults(params)

		// User-provided parameters should be preserved
		assert.Equal(t, 50, paramsWithDefaults["n_estimators"])
		assert.Equal(t, 10, paramsWithDefaults["max_depth"])

		// Default values should be added for missing parameters
		assert.Equal(t, 0.1, paramsWithDefaults["learning_rate"])
		assert.Equal(t, 31, paramsWithDefaults["num_leaves"])
		assert.Equal(t, 20, paramsWithDefaults["min_child_samples"])
		assert.Equal(t, 0.0, paramsWithDefaults["reg_alpha"])
		assert.Equal(t, 0.0, paramsWithDefaults["reg_lambda"])
	})
}

func TestLGBMRegressorWithParameterMapping(t *testing.T) {
	// Create simple training data
	X := mat.NewDense(100, 3, nil)
	y := mat.NewDense(100, 1, nil)
	for i := 0; i < 100; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
		y.Set(i, 0, float64(i)*0.1)
	}

	t.Run("SetParams with Python names", func(t *testing.T) {
		reg := NewLGBMRegressor()

		// Use Python parameter names
		params := map[string]interface{}{
			"n_estimators":      10,
			"learning_rate":     0.05,
			"num_leaves":        15,
			"min_child_samples": 10,
			"reg_alpha":         0.01,
			"reg_lambda":        0.02,
			"subsample":         0.9,
			"colsample_bytree":  0.8,
			"random_state":      42,
		}

		err := reg.SetParams(params)
		assert.NoError(t, err)

		// Verify parameters were set correctly
		assert.Equal(t, 10, reg.NumIterations)
		assert.Equal(t, 0.05, reg.LearningRate)
		assert.Equal(t, 15, reg.NumLeaves)
		assert.Equal(t, 10, reg.MinChildSamples)
		assert.Equal(t, 0.01, reg.RegAlpha)
		assert.Equal(t, 0.02, reg.RegLambda)
		assert.Equal(t, 0.9, reg.Subsample)
		assert.Equal(t, 0.8, reg.ColsampleBytree)
		assert.Equal(t, 42, reg.RandomState)

		// Train the model
		err = reg.Fit(X, y)
		assert.NoError(t, err)
	})

	t.Run("SetParams with aliases", func(t *testing.T) {
		reg := NewLGBMRegressor()

		// Use various aliases
		params := map[string]interface{}{
			"num_iterations":   5,
			"shrinkage_rate":   0.1,
			"num_leaf":         20,
			"min_data_in_leaf": 15,
			"lambda_l1":        0.05,
			"lambda_l2":        0.1,
			"bagging_fraction": 0.85,
			"feature_fraction": 0.75,
			"seed":             123,
		}

		err := reg.SetParams(params)
		assert.NoError(t, err)

		// Verify parameters were set correctly
		assert.Equal(t, 5, reg.NumIterations)
		assert.Equal(t, 0.1, reg.LearningRate)
		assert.Equal(t, 20, reg.NumLeaves)
		assert.Equal(t, 15, reg.MinChildSamples)
		assert.Equal(t, 0.05, reg.RegAlpha)
		assert.Equal(t, 0.1, reg.RegLambda)
		assert.Equal(t, 0.85, reg.Subsample)
		assert.Equal(t, 0.75, reg.ColsampleBytree)
		assert.Equal(t, 123, reg.RandomState)
	})

	t.Run("GetParams returns Python names", func(t *testing.T) {
		reg := NewLGBMRegressor()

		// Set parameters
		reg.NumIterations = 20
		reg.LearningRate = 0.15
		reg.NumLeaves = 25
		reg.MinChildSamples = 12
		reg.RegAlpha = 0.03
		reg.RegLambda = 0.04
		reg.Subsample = 0.95
		reg.ColsampleBytree = 0.85
		reg.RandomState = 99

		// Get parameters
		params := reg.GetParams()

		// Verify Python parameter names are returned
		assert.Equal(t, 20, params["n_estimators"])
		assert.Equal(t, 0.15, params["learning_rate"])
		assert.Equal(t, 25, params["num_leaves"])
		assert.Equal(t, 12, params["min_child_samples"])
		assert.Equal(t, 0.03, params["reg_alpha"])
		assert.Equal(t, 0.04, params["reg_lambda"])
		assert.Equal(t, 0.95, params["subsample"])
		assert.Equal(t, 0.85, params["colsample_bytree"])
		assert.Equal(t, 99, params["random_state"])
	})

	t.Run("Objective validation", func(t *testing.T) {
		reg := NewLGBMRegressor()

		// Set objective using alias
		params := map[string]interface{}{
			"objective": "mse",
		}

		err := reg.SetParams(params)
		assert.NoError(t, err)
		assert.Equal(t, "regression", reg.Objective)

		// Set objective using another alias
		params["objective"] = "mean_squared_error"
		err = reg.SetParams(params)
		assert.NoError(t, err)
		assert.Equal(t, "regression", reg.Objective)
	})
}
