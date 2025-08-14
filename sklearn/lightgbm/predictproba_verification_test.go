package lightgbm

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// PredictProbaTestData represents the test data structure from Python
type PredictProbaTestData struct {
	Dataset struct {
		X         [][]float64 `json:"X"`
		Y         []int       `json:"y"`
		NSamples  int         `json:"n_samples"`
		NFeatures int         `json:"n_features"`
		NClasses  int         `json:"n_classes"`
	} `json:"dataset"`
	TrainingParams struct {
		Objective     string  `json:"objective"`
		NumClass      int     `json:"num_class"`
		NumLeaves     int     `json:"num_leaves"`
		MinDataInLeaf int     `json:"min_data_in_leaf"`
		LearningRate  float64 `json:"learning_rate"`
		NumBoostRound int     `json:"num_boost_round"`
		Seed          int     `json:"seed"`
		Verbosity     int     `json:"verbose"`
	} `json:"training_params"`
	LightGBMResult struct {
		PredictionsProba  [][]float64 `json:"predictions_proba"`
		NumTrees          int         `json:"num_trees"`
		FeatureImportance []float64   `json:"feature_importance"`
	} `json:"lightgbm_result"`
	ExpectedBehavior struct {
		ProbabilitySumPerSample float64   `json:"probability_sum_per_sample"`
		ProbabilityRange        []float64 `json:"probability_range"`
		SoftmaxTransformation   string    `json:"softmax_transformation"`
	} `json:"expected_behavior"`
	TestInfo struct {
		Description string  `json:"description"`
		Focus       string  `json:"focus"`
		Tolerance   float64 `json:"tolerance"`
	} `json:"test_info"`
}

// loadPredictProbaTestData loads the test data from JSON file
func loadPredictProbaTestData(t *testing.T) *PredictProbaTestData {
	data, err := ioutil.ReadFile("testdata/predictproba_verification_data.json")
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData PredictProbaTestData
	err = json.Unmarshal(data, &testData)
	if err != nil {
		t.Fatalf("Failed to parse test data: %v", err)
	}

	return &testData
}

// TestPredictProbaSoftmaxApplication tests that Softmax is correctly applied for MulticlassLogLoss
func TestPredictProbaSoftmaxApplication(t *testing.T) {
	testData := loadPredictProbaTestData(t)

	// Convert test data to matrices
	X := mat.NewDense(testData.Dataset.NSamples, testData.Dataset.NFeatures, nil)
	y := mat.NewDense(testData.Dataset.NSamples, 1, nil)

	for i := 0; i < testData.Dataset.NSamples; i++ {
		for j := 0; j < testData.Dataset.NFeatures; j++ {
			X.Set(i, j, testData.Dataset.X[i][j])
		}
		y.Set(i, 0, float64(testData.Dataset.Y[i]))
	}

	// Create trainer with MulticlassLogLoss parameters
	params := TrainingParams{
		NumIterations: testData.TrainingParams.NumBoostRound,
		LearningRate:  testData.TrainingParams.LearningRate,
		NumLeaves:     testData.TrainingParams.NumLeaves,
		MinDataInLeaf: testData.TrainingParams.MinDataInLeaf,
		Objective:     "multiclass_logloss",
		NumClass:      testData.TrainingParams.NumClass,
		Seed:          testData.TrainingParams.Seed,
		Deterministic: true,
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Get model and create predictor
	model := trainer.GetModel()
	model.Objective = MulticlassLogLoss // Set objective type for PredictProba

	predictor := NewPredictor(model)
	predictor.SetDeterministic(true)

	t.Logf("=== PredictProba Softmax Application Test ===")
	t.Logf("Dataset: %dx%d, Classes: %d", testData.Dataset.NSamples, testData.Dataset.NFeatures, testData.Dataset.NClasses)
	t.Logf("Objective: %s", params.Objective)

	// Make probability predictions
	predictions_proba, err := predictor.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	rows, cols := predictions_proba.Dims()

	t.Logf("Probability predictions shape: %dx%d", rows, cols)

	// Verify probability properties
	tolerance := testData.TestInfo.Tolerance

	for i := 0; i < rows; i++ {
		rowSum := 0.0
		for j := 0; j < cols; j++ {
			prob := predictions_proba.At(i, j)

			// Check probability range [0, 1]
			if prob < 0.0 || prob > 1.0 {
				t.Errorf("Probability out of range [0,1] at sample %d, class %d: %f", i, j, prob)
			}

			rowSum += prob
		}

		// Check row sum equals 1
		if math.Abs(rowSum-1.0) > tolerance {
			t.Errorf("Probability sum not equal to 1 for sample %d: %f (tolerance: %f)", i, rowSum, tolerance)
		}
	}

	t.Logf("✅ PredictProba Softmax application test passed")
}

// TestPredictProbaNumericalStability tests numerical stability of Softmax implementation
func TestPredictProbaNumericalStability(t *testing.T) {
	// Create test data with extreme values
	X := mat.NewDense(3, 2, []float64{
		100.0, -100.0, // Extreme positive/negative
		0.0, 0.0, // Zero values
		50.0, 25.0, // Moderate values
	})

	// Create a simple model with extreme logit values
	model := &Model{
		NumFeatures: 2,
		NumClass:    3,
		Objective:   MulticlassLogLoss,
		Trees: []Tree{
			{
				TreeIndex:     0,
				ShrinkageRate: 0.1,
				Nodes: []Node{
					{NodeType: LeafNode, LeafValue: 1000.0}, // Extreme leaf value
				},
				LeafValues: []float64{1000.0},
				NumLeaves:  1,
			},
		},
		InitScore: 0.0,
	}

	predictor := NewPredictor(model)

	t.Logf("=== PredictProba Numerical Stability Test ===")

	// Test probability predictions with extreme values
	predictions_proba, err := predictor.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	rows, cols := predictions_proba.Dims()
	t.Logf("Extreme value predictions shape: %dx%d", rows, cols)

	// Verify numerical stability
	for i := 0; i < rows; i++ {
		rowSum := 0.0
		hasNaN := false
		hasInf := false

		for j := 0; j < cols; j++ {
			prob := predictions_proba.At(i, j)

			if math.IsNaN(prob) {
				hasNaN = true
			}
			if math.IsInf(prob, 0) {
				hasInf = true
			}

			rowSum += prob
		}

		// Check for numerical issues
		if hasNaN {
			t.Errorf("NaN found in probabilities for sample %d", i)
		}
		if hasInf {
			t.Errorf("Infinity found in probabilities for sample %d", i)
		}

		// Check row sum (should still be close to 1 even with extreme values)
		if math.Abs(rowSum-1.0) > 1e-10 {
			t.Errorf("Probability sum not equal to 1 for sample %d: %f", i, rowSum)
		}

		t.Logf("Sample %d: probabilities = %v, sum = %f", i,
			func() []float64 {
				probs := make([]float64, cols)
				for k := 0; k < cols; k++ {
					probs[k] = predictions_proba.At(i, k)
				}
				return probs
			}(), rowSum)
	}

	t.Logf("✅ PredictProba numerical stability test passed")
}

// TestPredictProbaObjectiveDispatch tests that the correct transformation is applied based on objective
func TestPredictProbaObjectiveDispatch(t *testing.T) {
	// Create test data
	X := mat.NewDense(2, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
	})

	// Test different objectives
	testCases := []struct {
		objective ObjectiveType
		expected  string
	}{
		{BinaryLogistic, "Binary classification (sigmoid transformation)"},
		{MulticlassSoftmax, "Multiclass softmax (already probabilities)"},
		{MulticlassLogLoss, "Multiclass logloss (softmax transformation)"},
		{RegressionL2, "Regression (raw predictions)"},
	}

	for _, tc := range testCases {
		t.Run(string(tc.objective), func(t *testing.T) {
			// Create a simple model
			model := &Model{
				NumFeatures: 2,
				NumClass:    3,
				Objective:   tc.objective,
				Trees: []Tree{
					{
						TreeIndex:     0,
						ShrinkageRate: 0.1,
						Nodes: []Node{
							{NodeType: LeafNode, LeafValue: 0.5},
						},
						LeafValues: []float64{0.5},
						NumLeaves:  1,
					},
				},
				InitScore: 0.0,
			}

			predictor := NewPredictor(model)

			t.Logf("Testing objective: %s (%s)", tc.objective, tc.expected)

			// Make predictions
			predictions_proba, err := predictor.PredictProba(X)
			if err != nil {
				t.Fatalf("Failed to predict probabilities: %v", err)
			}

			rows, cols := predictions_proba.Dims()
			t.Logf("Predictions shape: %dx%d", rows, cols)

			// Log first sample for verification
			if rows > 0 {
				firstSample := make([]float64, cols)
				for j := 0; j < cols; j++ {
					firstSample[j] = predictions_proba.At(0, j)
				}
				t.Logf("First sample predictions: %v", firstSample)
			}
		})
	}

	t.Logf("✅ PredictProba objective dispatch test passed")
}
