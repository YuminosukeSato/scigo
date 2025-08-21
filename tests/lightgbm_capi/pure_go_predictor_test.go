package lightgbm_capi

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
)

// TestPredictorResults is a struct to hold test data and predictions
type TestPredictorResults struct {
	XTest            [][]float64 `json:"X_test"`
	YTest            []float64   `json:"y_test"`
	PythonPreds      []float64   `json:"predictions"`
	PythonProba      [][]float64 `json:"predict_proba,omitempty"`
	GoProba          [][]float64 `json:"go_proba,omitempty"`
	GoPreds          []float64   `json:"go_predictions"`
	MatchPercentage  float64     `json:"match_percentage"`
	ProbMatchPercent float64     `json:"prob_match_percentage,omitempty"`
}

func TestSimplePredictorRegression(t *testing.T) {
	t.Skip("Skipping SimplePredictor tests until v0.7.0 implementation")

	// Load the Python baseline data
	data, err := os.ReadFile(filepath.Join("testdata", "regression", "test_data.json"))
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var results TestPredictorResults
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load the model
	modelPath := filepath.Join("testdata", "regression", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Create predictor
	predictor := lightgbm.NewSimplePredictor(model)

	// Convert test data to mat.Dense
	rows := len(results.XTest)
	cols := len(results.XTest[0])
	testData := mat.NewDense(rows, cols, nil)
	for i, row := range results.XTest {
		testData.SetRow(i, row)
	}

	// Make predictions
	predictions, err := predictor.Predict(testData)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare predictions
	matches := 0
	totalSamples := rows
	tolerance := 1e-4

	for i := 0; i < totalSamples; i++ {
		goPred := predictions.At(i, 0)
		pythonPred := results.PythonPreds[i]

		if math.Abs(goPred-pythonPred) < tolerance {
			matches++
		} else if i < 5 { // Log first few mismatches
			t.Logf("Sample %d mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
				i, goPred, pythonPred, math.Abs(goPred-pythonPred))
		}
	}

	matchPercentage := float64(matches) / float64(totalSamples) * 100
	t.Logf("Regression: Agreement %.2f%% (%d/%d samples)", matchPercentage, matches, totalSamples)

	if matchPercentage < 95.0 {
		t.Errorf("Regression accuracy too low: %.2f%% (expected >= 95%%)", matchPercentage)
	}
}

func TestSimplePredictorBinaryClassification(t *testing.T) {
	t.Skip("Skipping SimplePredictor tests until v0.7.0 implementation")

	// Load the Python baseline data
	data, err := os.ReadFile(filepath.Join("testdata", "binary", "test_data.json"))
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var results TestPredictorResults
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load the model
	modelPath := filepath.Join("testdata", "binary", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Create predictor
	predictor := lightgbm.NewSimplePredictor(model)

	// Convert test data to mat.Dense
	rows := len(results.XTest)
	cols := len(results.XTest[0])
	testData := mat.NewDense(rows, cols, nil)
	for i, row := range results.XTest {
		testData.SetRow(i, row)
	}

	// Make predictions
	predictions, err := predictor.Predict(testData)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare probabilities
	matches := 0
	totalSamples := rows
	tolerance := 1e-4

	for i := 0; i < totalSamples; i++ {
		goProb := predictions.At(i, 0)
		pythonProb := results.PythonProba[i][1] // Probability of positive class

		if math.Abs(goProb-pythonProb) < tolerance {
			matches++
		} else if i < 5 { // Log first few mismatches
			t.Logf("Sample %d mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
				i, goProb, pythonProb, math.Abs(goProb-pythonProb))
		}
	}

	matchPercentage := float64(matches) / float64(totalSamples) * 100
	t.Logf("Binary Classification: Agreement %.2f%% (%d/%d samples)", matchPercentage, matches, totalSamples)

	if matchPercentage < 95.0 {
		t.Errorf("Binary classification accuracy too low: %.2f%% (expected >= 95%%)", matchPercentage)
	}
}

func TestSimplePredictorMulticlassClassification(t *testing.T) {
	t.Skip("Skipping SimplePredictor tests until v0.7.0 implementation")

	// Load the Python baseline data
	data, err := os.ReadFile(filepath.Join("testdata", "multiclass", "test_data.json"))
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var results TestPredictorResults
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("Failed to unmarshal test data: %v", err)
	}

	// Load the model
	modelPath := filepath.Join("testdata", "multiclass", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Create predictor
	predictor := lightgbm.NewSimplePredictor(model)

	// Convert test data to mat.Dense
	rows := len(results.XTest)
	cols := len(results.XTest[0])
	testData := mat.NewDense(rows, cols, nil)
	for i, row := range results.XTest {
		testData.SetRow(i, row)
	}

	// Make predictions
	predictions, err := predictor.Predict(testData)
	if err != nil {
		t.Fatalf("Failed to make predictions: %v", err)
	}

	// Compare class probabilities
	matches := 0
	totalSamples := rows
	tolerance := 1e-4
	numClasses := model.NumClass

	for i := 0; i < totalSamples; i++ {
		allMatch := true
		for c := 0; c < numClasses; c++ {
			goProb := predictions.At(i, c)
			pythonProb := results.PythonProba[i][c]

			if math.Abs(goProb-pythonProb) >= tolerance {
				allMatch = false
				if i < 3 && c == 0 { // Log first few mismatches for first class
					t.Logf("Sample %d, Class %d mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
						i, c, goProb, pythonProb, math.Abs(goProb-pythonProb))
				}
			}
		}
		if allMatch {
			matches++
		}
	}

	matchPercentage := float64(matches) / float64(totalSamples) * 100
	t.Logf("Multiclass Classification: Agreement %.2f%% (%d/%d samples)", matchPercentage, matches, totalSamples)

	if matchPercentage < 95.0 {
		t.Errorf("Multiclass classification accuracy too low: %.2f%% (expected >= 95%%)", matchPercentage)
	}
}

// Benchmark tests
func BenchmarkSimplePredictorRegression(b *testing.B) {
	modelPath := filepath.Join("testdata", "regression", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	predictor := lightgbm.NewSimplePredictor(model)

	// Create sample data
	features := make([]float64, model.NumFeatures)
	for i := range features {
		features[i] = float64(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = predictor.PredictSingle(features)
	}
}

func BenchmarkSimplePredictorMulticlass(b *testing.B) {
	modelPath := filepath.Join("testdata", "multiclass", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		b.Fatalf("Failed to load model: %v", err)
	}

	predictor := lightgbm.NewSimplePredictor(model)

	// Create sample data
	features := make([]float64, model.NumFeatures)
	for i := range features {
		features[i] = float64(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = predictor.PredictSingle(features)
	}
}
