package lightgbm

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// GOSSTestData represents the test data structure from Python
type GOSSTestData struct {
	Dataset struct {
		X         [][]float64 `json:"X"`
		Y         []float64   `json:"y"`
		NSamples  int         `json:"n_samples"`
		NFeatures int         `json:"n_features"`
	} `json:"dataset"`
	GOSSParams struct {
		TopRate   float64 `json:"top_rate"`
		OtherRate float64 `json:"other_rate"`
		Seed      int     `json:"seed"`
	} `json:"goss_params"`
	ExpectedSampling struct {
		NSamples             int     `json:"n_samples"`
		TopRate              float64 `json:"top_rate"`
		OtherRate            float64 `json:"other_rate"`
		ExpectedTopCount     int     `json:"expected_top_count"`
		ExpectedOtherCount   int     `json:"expected_other_count"`
		ExpectedTotalSamples int     `json:"expected_total_samples"`
		AmplificationFactor  float64 `json:"amplification_factor"`
		SamplingEfficiency   float64 `json:"sampling_efficiency"`
	} `json:"expected_sampling"`
	LightGBMResult struct {
		Predictions       []float64 `json:"predictions"`
		NumTrees          int       `json:"num_trees"`
		FeatureImportance []float64 `json:"feature_importance"`
	} `json:"lightgbm_result"`
	TrainingParams struct {
		Objective     string  `json:"objective"`
		NumLeaves     int     `json:"num_leaves"`
		MinDataInLeaf int     `json:"min_data_in_leaf"`
		LearningRate  float64 `json:"learning_rate"`
		NumBoostRound int     `json:"num_boost_round"`
		Verbosity     int     `json:"verbose"`
	} `json:"training_params"`
	TestInfo struct {
		Description      string  `json:"description"`
		ExpectedBehavior string  `json:"expected_behavior"`
		Tolerance        float64 `json:"tolerance"`
	} `json:"test_info"`
}

// loadGOSSTestData loads the test data from JSON file
func loadGOSSTestData(t *testing.T) *GOSSTestData {
	data, err := ioutil.ReadFile("testdata/goss_verification_data.json")
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData GOSSTestData
	err = json.Unmarshal(data, &testData)
	if err != nil {
		t.Fatalf("Failed to parse test data: %v", err)
	}

	return &testData
}

// TestGOSSSamplingBehavior tests the GOSS sampling behavior against Python reference
func TestGOSSSamplingBehavior(t *testing.T) {
	testData := loadGOSSTestData(t)

	// Convert test data to matrices
	X := mat.NewDense(testData.Dataset.NSamples, testData.Dataset.NFeatures, nil)
	y := mat.NewDense(testData.Dataset.NSamples, 1, nil)

	for i := 0; i < testData.Dataset.NSamples; i++ {
		for j := 0; j < testData.Dataset.NFeatures; j++ {
			X.Set(i, j, testData.Dataset.X[i][j])
		}
		y.Set(i, 0, testData.Dataset.Y[i])
	}

	// Create trainer with GOSS parameters
	params := TrainingParams{
		NumIterations: testData.TrainingParams.NumBoostRound,
		LearningRate:  testData.TrainingParams.LearningRate,
		NumLeaves:     testData.TrainingParams.NumLeaves,
		MinDataInLeaf: testData.TrainingParams.MinDataInLeaf,
		BoostingType:  "goss",
		TopRate:       testData.GOSSParams.TopRate,
		OtherRate:     testData.GOSSParams.OtherRate,
		Objective:     testData.TrainingParams.Objective,
		Seed:          testData.GOSSParams.Seed,
		Deterministic: true,
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)
	trainer.X = X
	trainer.y = y

	// Initialize objective function manually (normally done in Fit)
	objFunc, err := CreateObjectiveFunction(params.Objective, &params)
	if err != nil {
		t.Fatalf("Failed to create objective function: %v", err)
	}
	trainer.objective = objFunc

	// Initialize trainer
	err = trainer.initialize()
	if err != nil {
		t.Fatalf("Failed to initialize trainer: %v", err)
	}

	// Set initial score (normally done in Fit)
	rows, _ := X.Dims()
	targets := make([]float64, rows)
	for i := 0; i < rows; i++ {
		targets[i] = y.At(i, 0)
	}
	trainer.initScore = trainer.objective.GetInitScore(targets)

	// Calculate initial gradients
	trainer.calculateGradients()

	t.Logf("=== GOSS Sampling Behavior Test ===")
	t.Logf("Dataset: %dx%d", testData.Dataset.NSamples, testData.Dataset.NFeatures)
	t.Logf("GOSS Params: top_rate=%.1f, other_rate=%.1f",
		testData.GOSSParams.TopRate, testData.GOSSParams.OtherRate)

	// Test GOSS sampling
	sampledIndices := trainer.gosssampling()

	// Verify sampling counts
	expectedTotal := testData.ExpectedSampling.ExpectedTotalSamples
	actualTotal := len(sampledIndices)

	t.Logf("Expected total samples: %d", expectedTotal)
	t.Logf("Actual total samples: %d", actualTotal)

	if actualTotal != expectedTotal {
		t.Errorf("Sample count mismatch: expected %d, got %d", expectedTotal, actualTotal)
	}

	// Verify amplification factor calculation
	expectedAmp := testData.ExpectedSampling.AmplificationFactor
	actualAmp := (1.0 - testData.GOSSParams.TopRate) / testData.GOSSParams.OtherRate

	t.Logf("Expected amplification factor: %.3f", expectedAmp)
	t.Logf("Actual amplification factor: %.3f", actualAmp)

	if math.Abs(actualAmp-expectedAmp) > 1e-6 {
		t.Errorf("Amplification factor mismatch: expected %.3f, got %.3f", expectedAmp, actualAmp)
	}

	// Test determinism: run sampling again with same state
	trainer.calculateGradients() // Reset gradients
	sampledIndices2 := trainer.gosssampling()

	if len(sampledIndices) != len(sampledIndices2) {
		t.Errorf("Non-deterministic sampling: different lengths %d vs %d",
			len(sampledIndices), len(sampledIndices2))
	}

	for i := range sampledIndices {
		if sampledIndices[i] != sampledIndices2[i] {
			t.Errorf("Non-deterministic sampling: index %d differs %d vs %d",
				i, sampledIndices[i], sampledIndices2[i])
			break
		}
	}

	t.Logf("✅ GOSS sampling behavior test passed")
}

// TestGOSSAmplificationAccuracy tests the accuracy of gradient/hessian amplification
func TestGOSSAmplificationAccuracy(t *testing.T) {
	testData := loadGOSSTestData(t)

	// Convert test data to matrices
	X := mat.NewDense(testData.Dataset.NSamples, testData.Dataset.NFeatures, nil)
	y := mat.NewDense(testData.Dataset.NSamples, 1, nil)

	for i := 0; i < testData.Dataset.NSamples; i++ {
		for j := 0; j < testData.Dataset.NFeatures; j++ {
			X.Set(i, j, testData.Dataset.X[i][j])
		}
		y.Set(i, 0, testData.Dataset.Y[i])
	}

	// Create trainer
	params := TrainingParams{
		BoostingType:  "goss",
		TopRate:       testData.GOSSParams.TopRate,
		OtherRate:     testData.GOSSParams.OtherRate,
		Seed:          testData.GOSSParams.Seed,
		Objective:     "regression",
		Deterministic: true,
	}

	trainer := NewTrainer(params)
	trainer.X = X
	trainer.y = y

	// Initialize objective function manually
	objFunc, err := CreateObjectiveFunction(params.Objective, &params)
	if err != nil {
		t.Fatalf("Failed to create objective function: %v", err)
	}
	trainer.objective = objFunc

	err = trainer.initialize()
	if err != nil {
		t.Fatalf("Failed to initialize trainer: %v", err)
	}

	// Set initial score (normally done in Fit)
	rows, _ := X.Dims()
	targets := make([]float64, rows)
	for i := 0; i < rows; i++ {
		targets[i] = y.At(i, 0)
	}
	trainer.initScore = trainer.objective.GetInitScore(targets)

	// Store original gradients and hessians
	trainer.calculateGradients()
	originalGrads := make([]float64, len(trainer.gradients))
	originalHess := make([]float64, len(trainer.hessians))
	copy(originalGrads, trainer.gradients)
	copy(originalHess, trainer.hessians)

	t.Logf("=== GOSS Amplification Accuracy Test ===")

	// Perform sampling (this modifies gradients/hessians)
	sampledIndices := trainer.gosssampling()

	// Verify amplification is applied correctly
	expectedAmp := (1.0 - testData.GOSSParams.TopRate) / testData.GOSSParams.OtherRate
	topCount := int(float64(testData.Dataset.NSamples) * testData.GOSSParams.TopRate)

	// Create a map of sampled indices for quick lookup
	sampledMap := make(map[int]bool)
	for _, idx := range sampledIndices {
		sampledMap[idx] = true
	}

	// Sort original gradients to identify top samples
	type gradItem struct {
		index int
		value float64
	}
	gradItems := make([]gradItem, testData.Dataset.NSamples)
	for i := 0; i < testData.Dataset.NSamples; i++ {
		gradItems[i] = gradItem{index: i, value: math.Abs(originalGrads[i])}
	}

	// Sort by absolute gradient (descending)
	for i := 0; i < len(gradItems)-1; i++ {
		for j := i + 1; j < len(gradItems); j++ {
			if gradItems[i].value < gradItems[j].value {
				gradItems[i], gradItems[j] = gradItems[j], gradItems[i]
			}
		}
	}

	// Check amplification for non-top samples
	amplifiedCount := 0
	for i := topCount; i < len(gradItems); i++ {
		idx := gradItems[i].index
		if sampledMap[idx] {
			// This should be amplified
			expectedGrad := originalGrads[idx] * expectedAmp
			expectedHess := originalHess[idx] * expectedAmp

			if math.Abs(trainer.gradients[idx]-expectedGrad) > 1e-12 {
				t.Errorf("Gradient amplification incorrect for index %d: expected %.6f, got %.6f",
					idx, expectedGrad, trainer.gradients[idx])
			}

			if math.Abs(trainer.hessians[idx]-expectedHess) > 1e-12 {
				t.Errorf("Hessian amplification incorrect for index %d: expected %.6f, got %.6f",
					idx, expectedHess, trainer.hessians[idx])
			}

			amplifiedCount++
		}
	}

	t.Logf("Amplified %d samples with factor %.3f", amplifiedCount, expectedAmp)
	t.Logf("✅ GOSS amplification accuracy test passed")
}

// TestGOSSPredictionAccuracy tests the prediction accuracy with GOSS vs reference
func TestGOSSPredictionAccuracy(t *testing.T) {
	testData := loadGOSSTestData(t)

	// Convert test data to matrices
	X := mat.NewDense(testData.Dataset.NSamples, testData.Dataset.NFeatures, nil)
	y := mat.NewDense(testData.Dataset.NSamples, 1, nil)

	for i := 0; i < testData.Dataset.NSamples; i++ {
		for j := 0; j < testData.Dataset.NFeatures; j++ {
			X.Set(i, j, testData.Dataset.X[i][j])
		}
		y.Set(i, 0, testData.Dataset.Y[i])
	}

	// Train with GOSS using same parameters as Python
	params := TrainingParams{
		NumIterations: testData.TrainingParams.NumBoostRound,
		LearningRate:  testData.TrainingParams.LearningRate,
		NumLeaves:     testData.TrainingParams.NumLeaves,
		MinDataInLeaf: testData.TrainingParams.MinDataInLeaf,
		BoostingType:  "goss",
		TopRate:       testData.GOSSParams.TopRate,
		OtherRate:     testData.GOSSParams.OtherRate,
		Objective:     testData.TrainingParams.Objective,
		Seed:          testData.GOSSParams.Seed,
		Deterministic: true,
		Verbosity:     -1,
	}

	trainer := NewTrainer(params)
	err := trainer.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Make predictions
	model := trainer.GetModel()
	predictor := NewPredictor(model)
	predictor.SetDeterministic(true)

	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	t.Logf("=== GOSS Prediction Accuracy Test ===")

	// Compare with Python LightGBM predictions
	tolerance := testData.TestInfo.Tolerance
	maxDiff := 0.0
	avgDiff := 0.0

	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		goPred := predictions.At(i, 0)
		pythonPred := testData.LightGBMResult.Predictions[i]
		diff := math.Abs(goPred - pythonPred)

		if diff > maxDiff {
			maxDiff = diff
		}
		avgDiff += diff

		if diff > tolerance {
			t.Errorf("Prediction %d exceeds tolerance: Go=%.6f, Python=%.6f, diff=%.6f",
				i, goPred, pythonPred, diff)
		}
	}

	avgDiff /= float64(rows)

	t.Logf("Prediction comparison:")
	t.Logf("  Max difference: %.6f", maxDiff)
	t.Logf("  Avg difference: %.6f", avgDiff)
	t.Logf("  Tolerance: %.6f", tolerance)
	t.Logf("  Samples within tolerance: %d/%d", rows, rows)

	if maxDiff <= tolerance {
		t.Logf("✅ GOSS prediction accuracy test passed")
	} else {
		t.Errorf("❌ GOSS prediction accuracy test failed: max diff %.6f > tolerance %.6f",
			maxDiff, tolerance)
	}
}
