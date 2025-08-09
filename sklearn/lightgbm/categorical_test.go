package lightgbm

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestCategoricalFeatures(t *testing.T) {
	// Create dataset with categorical features
	// Features: [continuous, categorical, continuous, categorical]
	// Categorical features at indices 1 and 3
	n := 200
	X := mat.NewDense(n, 4, nil)
	y := mat.NewDense(n, 1, nil)

	rand.Seed(42)

	for i := 0; i < n; i++ {
		// Continuous feature 0
		x0 := rand.Float64() * 10
		X.Set(i, 0, x0)

		// Categorical feature 1 (values: 0, 1, 2)
		cat1 := float64(rand.Intn(3))
		X.Set(i, 1, cat1)

		// Continuous feature 2
		x2 := rand.Float64() * 5
		X.Set(i, 2, x2)

		// Categorical feature 3 (values: 0, 1, 2, 3, 4)
		cat3 := float64(rand.Intn(5))
		X.Set(i, 3, cat3)

		// Target based on features with categorical effects
		target := x0*0.5 + x2*0.3

		// Add categorical effects
		if cat1 == 0 {
			target += 2.0
		} else if cat1 == 1 {
			target -= 1.0
		}

		if cat3 == 0 || cat3 == 1 {
			target += 1.5
		} else if cat3 == 4 {
			target -= 2.0
		}

		// Add noise
		target += rand.NormFloat64() * 0.1

		y.Set(i, 0, target)
	}

	t.Run("Basic categorical training", func(t *testing.T) {
		params := TrainingParams{
			NumIterations:       10,
			LearningRate:        0.1,
			NumLeaves:           15,
			MaxDepth:            5,
			MinDataInLeaf:       5,
			CategoricalFeatures: []int{1, 3}, // Specify categorical features
			MaxCatToOnehot:      4,
			Objective:           "regression",
		}

		trainer := NewTrainer(params)
		err := trainer.Fit(X, y)
		require.NoError(t, err)

		model := trainer.GetModel()
		assert.NotNil(t, model)
		assert.Equal(t, 10, len(model.Trees))

		// Check that some nodes are categorical
		hasCategoricalNode := false
		for _, tree := range model.Trees {
			for _, node := range tree.Nodes {
				if node.NodeType == CategoricalNode {
					hasCategoricalNode = true
					assert.NotEmpty(t, node.Categories, "Categorical node should have categories")
					assert.Contains(t, []int{1, 3}, node.SplitFeature,
						"Categorical split should be on categorical feature")
				}
			}
		}
		assert.True(t, hasCategoricalNode, "Should have at least one categorical node")
	})

	t.Run("Categorical vs Numerical comparison", func(t *testing.T) {
		// Train with categorical features
		paramsCat := TrainingParams{
			NumIterations:       10,
			LearningRate:        0.1,
			NumLeaves:           15,
			MinDataInLeaf:       5,
			CategoricalFeatures: []int{1, 3},
			Objective:           "regression",
			Seed:                42,
		}

		trainerCat := NewTrainer(paramsCat)
		err := trainerCat.Fit(X, y)
		require.NoError(t, err)
		modelCat := trainerCat.GetModel()

		// Train without categorical features (treat as numerical)
		paramsNum := TrainingParams{
			NumIterations: 10,
			LearningRate:  0.1,
			NumLeaves:     15,
			MinDataInLeaf: 5,
			Objective:     "regression",
			Seed:          42,
		}

		trainerNum := NewTrainer(paramsNum)
		err = trainerNum.Fit(X, y)
		require.NoError(t, err)
		modelNum := trainerNum.GetModel()

		// Make predictions
		predCat, err := modelCat.Predict(X)
		require.NoError(t, err)

		predNum, err := modelNum.Predict(X)
		require.NoError(t, err)

		// Calculate MSE for both
		mseCat := 0.0
		mseNum := 0.0
		for i := 0; i < n; i++ {
			trueval := y.At(i, 0)

			errCat := predCat.At(i, 0) - trueval
			mseCat += errCat * errCat

			errNum := predNum.At(i, 0) - trueval
			mseNum += errNum * errNum
		}
		mseCat /= float64(n)
		mseNum /= float64(n)

		// Categorical should generally perform better on this dataset
		t.Logf("MSE with categorical: %.4f", mseCat)
		t.Logf("MSE without categorical: %.4f", mseNum)

		// Both should be reasonable
		assert.Less(t, mseCat, 5.0, "Categorical MSE should be reasonable")
		assert.Less(t, mseNum, 10.0, "Numerical MSE should be reasonable")
	})

	t.Run("Single category feature", func(t *testing.T) {
		// Create simple dataset with one categorical feature
		n := 100
		X := mat.NewDense(n, 2, nil)
		y := mat.NewDense(n, 1, nil)

		for i := 0; i < n; i++ {
			// Continuous feature
			x0 := rand.Float64() * 10
			X.Set(i, 0, x0)

			// Binary categorical feature (0 or 1)
			cat := float64(i % 2)
			X.Set(i, 1, cat)

			// Target with strong categorical effect
			target := x0 * 0.5
			if cat == 0 {
				target += 5.0
			} else {
				target -= 5.0
			}

			y.Set(i, 0, target)
		}

		params := TrainingParams{
			NumIterations:       5,
			LearningRate:        0.1,
			NumLeaves:           10,
			MinDataInLeaf:       5,
			CategoricalFeatures: []int{1},
			Objective:           "regression",
		}

		trainer := NewTrainer(params)
		err := trainer.Fit(X, y)
		require.NoError(t, err)

		model := trainer.GetModel()
		pred, err := model.Predict(X)
		require.NoError(t, err)

		// Check predictions capture the categorical effect
		for i := 0; i < n; i++ {
			cat := X.At(i, 1)
			predVal := pred.At(i, 0)
			trueVal := y.At(i, 0)

			// Predictions should be close to true values
			assert.InDelta(t, trueVal, predVal, 2.0,
				"Prediction should capture categorical effect for cat=%v", cat)
		}
	})
}

func TestCategoricalSplitLogic(t *testing.T) {
	t.Run("isCategoricalFeature", func(t *testing.T) {
		params := TrainingParams{
			CategoricalFeatures: []int{1, 3, 5},
		}
		trainer := &Trainer{params: params}

		assert.False(t, trainer.isCategoricalFeature(0))
		assert.True(t, trainer.isCategoricalFeature(1))
		assert.False(t, trainer.isCategoricalFeature(2))
		assert.True(t, trainer.isCategoricalFeature(3))
		assert.False(t, trainer.isCategoricalFeature(4))
		assert.True(t, trainer.isCategoricalFeature(5))
		assert.False(t, trainer.isCategoricalFeature(6))
	})

	t.Run("Categorical node prediction", func(t *testing.T) {
		// Create a simple tree with categorical node
		tree := Tree{
			TreeIndex:     0,
			ShrinkageRate: 1.0,
			Nodes: []Node{
				{
					NodeID:       0,
					NodeType:     CategoricalNode,
					SplitFeature: 0,
					Categories:   []int{1, 3}, // Categories 1 and 3 go left
					LeftChild:    1,
					RightChild:   2,
				},
				{
					NodeID:     1,
					NodeType:   LeafNode,
					LeafValue:  -1.0,
					LeftChild:  -1,
					RightChild: -1,
				},
				{
					NodeID:     2,
					NodeType:   LeafNode,
					LeafValue:  1.0,
					LeftChild:  -1,
					RightChild: -1,
				},
			},
		}

		// Test predictions for different category values
		testCases := []struct {
			category float64
			expected float64
		}{
			{0.0, 1.0},  // Category 0 -> right (not in [1,3])
			{1.0, -1.0}, // Category 1 -> left (in [1,3])
			{2.0, 1.0},  // Category 2 -> right (not in [1,3])
			{3.0, -1.0}, // Category 3 -> left (in [1,3])
			{4.0, 1.0},  // Category 4 -> right (not in [1,3])
		}

		for _, tc := range testCases {
			features := []float64{tc.category}
			pred := tree.Predict(features)
			assert.Equal(t, tc.expected, pred,
				"Wrong prediction for category %.0f", tc.category)
		}
	})
}

func TestCategoricalWithLGBMRegressor(t *testing.T) {
	// Create dataset with mixed features
	n := 150
	X := mat.NewDense(n, 3, nil)
	y := mat.NewDense(n, 1, nil)

	rand.Seed(42)

	for i := 0; i < n; i++ {
		// Continuous
		X.Set(i, 0, rand.Float64()*10)

		// Categorical (0, 1, 2)
		X.Set(i, 1, float64(rand.Intn(3)))

		// Continuous
		X.Set(i, 2, rand.Float64()*5)

		// Target
		target := X.At(i, 0)*0.5 + X.At(i, 2)*0.3
		if X.At(i, 1) == 0 {
			target += 3.0
		} else if X.At(i, 1) == 2 {
			target -= 2.0
		}
		target += rand.NormFloat64() * 0.1

		y.Set(i, 0, target)
	}

	// Train with categorical features
	reg := NewLGBMRegressor().
		WithNumIterations(10).
		WithNumLeaves(15).
		WithLearningRate(0.1)

	// Set categorical features
	reg.CategoricalFeatures = []int{1}

	err := reg.Fit(X, y)
	require.NoError(t, err)

	// Make predictions
	pred, err := reg.Predict(X)
	require.NoError(t, err)

	// Calculate R2 score
	score, err := reg.Score(X, y)
	require.NoError(t, err)

	assert.Greater(t, score, 0.7, "R2 score should be good with categorical features")

	// Check feature importance
	importance := reg.GetFeatureImportance("gain")
	assert.NotNil(t, importance)
	assert.Equal(t, 3, len(importance))

	// Categorical feature should have some importance
	assert.Greater(t, importance[1], 0.0, "Categorical feature should have importance")

	// Test predictions
	rows, _ := pred.Dims()
	assert.Equal(t, n, rows)
}
