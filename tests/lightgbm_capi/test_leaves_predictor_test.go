package lightgbm_capi

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

func TestLeavesPredictorRegression(t *testing.T) {
	// Load test data
	testDataPath := filepath.Join("testdata", "regression", "test_data.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData struct {
		XTest       [][]float64 `json:"X_test"`
		YTest       []float64   `json:"y_test"`
		Predictions []float64   `json:"predictions"`
	}

	if err := json.Unmarshal(data, &testData); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load model using leaves loader
	modelPath := filepath.Join("testdata", "regression", "model.txt")
	model, err := lightgbm.LoadLeavesModelFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Debug output
	t.Logf("Model loaded:")
	t.Logf("  NumTrees: %d", len(model.Trees))
	t.Logf("  NumFeatures: %d", model.NumFeatures)
	t.Logf("  InitScore: %f", model.InitScore)
	t.Logf("  Objective: %s", model.Objective)

	// Debug first tree
	if len(model.Trees) > 0 {
		tree := model.Trees[0]
		t.Logf("Tree 0:")
		t.Logf("  NumNodes: %d", len(tree.Nodes))
		t.Logf("  NumLeafValues: %d", len(tree.LeafValues))
		t.Logf("  ShrinkageRate: %f", tree.ShrinkageRate)
	}

	// Create predictor
	predictor := lightgbm.NewLeavesPredictor(model)

	// Test first sample in detail
	if len(testData.XTest) > 0 {
		sample := testData.XTest[0]
		pred := predictor.PredictSingle(sample)
		pythonPred := testData.Predictions[0]

		t.Logf("First sample prediction:")
		t.Logf("  Go: %.6f", pred[0])
		t.Logf("  Python: %.6f", pythonPred)
		t.Logf("  Difference: %.6f", math.Abs(pred[0]-pythonPred))

		// Debug tree-by-tree predictions
		t.Logf("Tree-by-tree predictions:")
		cumSum := model.InitScore
		for i, tree := range model.Trees {
			treePred := tree.Predict(sample)
			cumSum += treePred
			if i < 5 { // Only show first 5 trees
				t.Logf("  Tree %d: pred=%.6f, shrinkage=%.2f, cum=%.6f",
					i, treePred, tree.ShrinkageRate, cumSum)
			}
		}
	}

	// Test all samples using Gonum
	rows := len(testData.XTest)
	cols := len(testData.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, row := range testData.XTest {
		X.SetRow(i, row)
	}

	// Make batch predictions
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare predictions
	matches := 0
	tolerance := 1e-4
	maxDiff := 0.0

	for i := 0; i < rows; i++ {
		goPred := predictions.At(i, 0)
		pythonPred := testData.Predictions[i]
		diff := math.Abs(goPred - pythonPred)

		if diff > maxDiff {
			maxDiff = diff
		}

		if diff < tolerance {
			matches++
		} else if i < 5 {
			t.Logf("Sample %d mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
				i, goPred, pythonPred, diff)
		}
	}

	matchPercentage := float64(matches) / float64(rows) * 100
	t.Logf("Regression results:")
	t.Logf("  Agreement: %.2f%% (%d/%d samples)", matchPercentage, matches, rows)
	t.Logf("  Max difference: %.6f", maxDiff)

	if matchPercentage < 95.0 {
		t.Errorf("Regression accuracy too low: %.2f%% (expected >= 95%%)", matchPercentage)
	}
}

func TestLeavesPredictorBinary(t *testing.T) {
	// Load test data
	testDataPath := filepath.Join("testdata", "binary", "test_data.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData struct {
		XTest        [][]float64 `json:"X_test"`
		YTest        []float64   `json:"y_test"`
		Predictions  []float64   `json:"predictions"`
		PredictProba [][]float64 `json:"predict_proba"`
	}

	if err := json.Unmarshal(data, &testData); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load model
	modelPath := filepath.Join("testdata", "binary", "model.txt")
	model, err := lightgbm.LoadLeavesModelFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Logf("Binary model loaded:")
	t.Logf("  NumTrees: %d", len(model.Trees))
	t.Logf("  InitScore: %f", model.InitScore)

	// Create predictor
	predictor := lightgbm.NewLeavesPredictor(model)

	// Convert test data to matrix
	rows := len(testData.XTest)
	cols := len(testData.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, row := range testData.XTest {
		X.SetRow(i, row)
	}

	// Make predictions
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare probabilities
	matches := 0
	tolerance := 1e-4

	for i := 0; i < rows; i++ {
		goProb := predictions.At(i, 0)
		pythonProb := testData.PredictProba[i][1] // Probability of positive class

		if math.Abs(goProb-pythonProb) < tolerance {
			matches++
		} else if i < 5 {
			t.Logf("Sample %d mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
				i, goProb, pythonProb, math.Abs(goProb-pythonProb))
		}
	}

	matchPercentage := float64(matches) / float64(rows) * 100
	t.Logf("Binary Classification: Agreement %.2f%% (%d/%d samples)",
		matchPercentage, matches, rows)

	if matchPercentage < 95.0 {
		t.Errorf("Binary classification accuracy too low: %.2f%% (expected >= 95%%)",
			matchPercentage)
	}
}

func TestLeavesPredictorMulticlass(t *testing.T) {
	// Load test data
	testDataPath := filepath.Join("testdata", "multiclass", "test_data.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData struct {
		XTest        [][]float64 `json:"X_test"`
		YTest        []float64   `json:"y_test"`
		Predictions  []float64   `json:"predictions"`
		PredictProba [][]float64 `json:"predict_proba"`
	}

	if err := json.Unmarshal(data, &testData); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load model
	modelPath := filepath.Join("testdata", "multiclass", "model.txt")
	model, err := lightgbm.LoadLeavesModelFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	t.Logf("Multiclass model loaded:")
	t.Logf("  NumTrees: %d", len(model.Trees))
	t.Logf("  NumClass: %d", model.NumClass)
	t.Logf("  InitScore: %f", model.InitScore)

	// Create predictor
	predictor := lightgbm.NewLeavesPredictor(model)

	// Convert test data to matrix
	rows := len(testData.XTest)
	cols := len(testData.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, row := range testData.XTest {
		X.SetRow(i, row)
	}

	// Make predictions
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare class probabilities
	matches := 0
	tolerance := 1e-4
	numClasses := model.NumClass

	for i := 0; i < rows; i++ {
		allMatch := true
		for c := 0; c < numClasses; c++ {
			goProb := predictions.At(i, c)
			pythonProb := testData.PredictProba[i][c]

			if math.Abs(goProb-pythonProb) >= tolerance {
				allMatch = false
				if i < 3 && c == 0 {
					t.Logf("Sample %d, Class %d: Go=%.6f, Python=%.6f, diff=%.6f",
						i, c, goProb, pythonProb, math.Abs(goProb-pythonProb))
				}
			}
		}
		if allMatch {
			matches++
		}
	}

	matchPercentage := float64(matches) / float64(rows) * 100
	t.Logf("Multiclass Classification: Agreement %.2f%% (%d/%d samples)",
		matchPercentage, matches, rows)

	if matchPercentage < 95.0 {
		t.Errorf("Multiclass classification accuracy too low: %.2f%% (expected >= 95%%)",
			matchPercentage)
	}
}