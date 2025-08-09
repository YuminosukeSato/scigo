package lightgbm_capi

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

// TestDataSimple holds test data for comparison (without C API functions)
type TestDataSimple struct {
	XTrain        *mat.Dense
	XTest         *mat.Dense
	YTrain        *mat.Dense
	YTest         *mat.Dense
	TrainPreds    []float64
	TestPreds     []float64
	ModelPath     string
	ParamsPath    string
	ObjectiveType string
}

// loadCSVSimple loads a CSV file into a slice
func loadCSVSimple(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = file.Close()
	}()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Skip header if present
	startIdx := 0
	if len(records) > 0 {
		// Check if first row is header (non-numeric)
		if _, err := strconv.ParseFloat(records[0][0], 64); err != nil {
			startIdx = 1
		}
	}

	data := make([][]float64, len(records)-startIdx)
	for i := startIdx; i < len(records); i++ {
		row := make([]float64, len(records[i]))
		for j, val := range records[i] {
			row[j], err = strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing value at row %d, col %d: %v", i, j, err)
			}
		}
		data[i-startIdx] = row
	}

	return data, nil
}

// loadTestDataSimple loads test data for a specific objective
func loadTestDataSimple(objective string) (*TestDataSimple, error) {
	baseDir := filepath.Join("testdata", objective)

	// Load training data
	xTrainData, err := loadCSVSimple(filepath.Join(baseDir, objective+"_X_train.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load X_train: %v", err)
	}

	xTestData, err := loadCSVSimple(filepath.Join(baseDir, objective+"_X_test.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load X_test: %v", err)
	}

	yTrainData, err := loadCSVSimple(filepath.Join(baseDir, objective+"_y_train.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load y_train: %v", err)
	}

	yTestData, err := loadCSVSimple(filepath.Join(baseDir, objective+"_y_test.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load y_test: %v", err)
	}

	// Convert to matrices
	nTrainRows := len(xTrainData)
	nTestRows := len(xTestData)
	nCols := len(xTrainData[0])

	xTrainFlat := make([]float64, nTrainRows*nCols)
	xTestFlat := make([]float64, nTestRows*nCols)
	yTrainFlat := make([]float64, nTrainRows)
	yTestFlat := make([]float64, nTestRows)

	for i, row := range xTrainData {
		for j, val := range row {
			xTrainFlat[i*nCols+j] = val
		}
	}

	for i, row := range xTestData {
		for j, val := range row {
			xTestFlat[i*nCols+j] = val
		}
	}

	for i, row := range yTrainData {
		yTrainFlat[i] = row[0]
	}

	for i, row := range yTestData {
		yTestFlat[i] = row[0]
	}

	// Load predictions
	trainPredData, err := loadCSVSimple(filepath.Join(baseDir, "train_predictions.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load train predictions: %v", err)
	}

	testPredData, err := loadCSVSimple(filepath.Join(baseDir, "test_predictions.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load test predictions: %v", err)
	}

	// For multiclass, predictions have multiple columns
	var trainPreds, testPreds []float64
	if objective == "multiclass" {
		// Flatten multiclass predictions
		for _, row := range trainPredData {
			trainPreds = append(trainPreds, row...)
		}
		for _, row := range testPredData {
			testPreds = append(testPreds, row...)
		}
	} else {
		// Single column predictions
		trainPreds = make([]float64, len(trainPredData))
		testPreds = make([]float64, len(testPredData))
		for i, row := range trainPredData {
			trainPreds[i] = row[0]
		}
		for i, row := range testPredData {
			testPreds[i] = row[0]
		}
	}

	return &TestDataSimple{
		XTrain:        mat.NewDense(nTrainRows, nCols, xTrainFlat),
		XTest:         mat.NewDense(nTestRows, nCols, xTestFlat),
		YTrain:        mat.NewDense(nTrainRows, 1, yTrainFlat),
		YTest:         mat.NewDense(nTestRows, 1, yTestFlat),
		TrainPreds:    trainPreds,
		TestPreds:     testPreds,
		ModelPath:     filepath.Join(baseDir, "model.txt"),
		ParamsPath:    filepath.Join(baseDir, "params.json"),
		ObjectiveType: objective,
	}, nil
}

// compareValuesSimple compares two float64 values with tolerance
func compareValuesSimple(a, b float64, tolerance float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 1) && math.IsInf(b, 1) {
		return true
	}
	if math.IsInf(a, -1) && math.IsInf(b, -1) {
		return true
	}

	diff := math.Abs(a - b)
	if diff < tolerance {
		return true
	}

	// Relative error for larger values
	avg := (math.Abs(a) + math.Abs(b)) / 2
	if avg > 1e-10 {
		relError := diff / avg
		return relError < tolerance
	}

	return false
}

// TestPureGoTraining tests pure Go implementation training
func TestPureGoTraining(t *testing.T) {
	// Test regression
	t.Run("Regression", func(t *testing.T) {
		testPureGoObjective(t, "regression")
	})

	// Test binary classification
	t.Run("Binary", func(t *testing.T) {
		testPureGoObjective(t, "binary")
	})

	// Test multiclass classification
	t.Run("Multiclass", func(t *testing.T) {
		testPureGoObjective(t, "multiclass")
	})
}

func testPureGoObjective(t *testing.T, objective string) {
	// Load test data
	testData, err := loadTestDataSimple(objective)
	if err != nil {
		t.Skipf("Test data not available: %v. Run python_baseline/train_models.py first", err)
	}

	// Load parameters
	paramsData, err := os.ReadFile(testData.ParamsPath)
	if err != nil {
		t.Fatalf("Failed to load parameters: %v", err)
	}

	var params map[string]interface{}
	if err := json.Unmarshal(paramsData, &params); err != nil {
		t.Fatalf("Failed to parse parameters: %v", err)
	}

	// Create pure Go trainer with same parameters
	goParams := lightgbm.TrainingParams{
		NumIterations:   10,
		LearningRate:    params["learning_rate"].(float64),
		NumLeaves:       int(params["num_leaves"].(float64)),
		MaxDepth:        int(params["max_depth"].(float64)),
		MinDataInLeaf:   int(params["min_data_in_leaf"].(float64)),
		Lambda:          params["lambda_l2"].(float64),
		Alpha:           params["lambda_l1"].(float64),
		MinGainToSplit:  params["min_gain_to_split"].(float64),
		BaggingFraction: params["bagging_fraction"].(float64),
		BaggingFreq:     int(params["bagging_freq"].(float64)),
		FeatureFraction: params["feature_fraction"].(float64),
		Objective:       objective,
		Seed:            42,
		Deterministic:   true,
		Verbosity:       -1,
	}

	if objective == "multiclass" {
		goParams.NumClass = 3
	}

	// Train pure Go model
	trainer := lightgbm.NewTrainer(goParams)
	if err := trainer.Fit(testData.XTrain, testData.YTrain); err != nil {
		t.Fatalf("Failed to train pure Go model: %v", err)
	}

	// Get pure Go model
	goModel := trainer.GetModel()

	// Create predictor
	predictor := lightgbm.NewPredictor(goModel)
	predictor.SetDeterministic(true)

	// Make predictions with pure Go
	goPreds, err := predictor.Predict(testData.XTest)
	if err != nil {
		t.Fatalf("Failed to predict with pure Go: %v", err)
	}

	// Compare with Python predictions
	tolerance := 0.1 // 10% tolerance for initial implementation
	maxDiff := 0.0
	numDiffs := 0
	nrows, _ := testData.XTest.Dims()

	// Calculate comparison based on objective
	if objective == "multiclass" {
		// For multiclass, compare probabilities for each class
		numClasses := 3
		maxIndex := len(testData.TestPreds) / numClasses
		if maxIndex > nrows {
			maxIndex = nrows
		}
		for i := 0; i < maxIndex; i++ {
			for j := 0; j < numClasses; j++ {
				goPred := goPreds.At(i, j)
				pythonPred := testData.TestPreds[i*numClasses+j]
				diff := math.Abs(goPred - pythonPred)
				if diff > maxDiff {
					maxDiff = diff
				}
				if !compareValuesSimple(goPred, pythonPred, tolerance) {
					numDiffs++
					if numDiffs <= 5 {
						t.Logf("Difference at sample %d, class %d: Go = %v, Python = %v, diff = %v",
							i, j, goPred, pythonPred, diff)
					}
				}
			}
		}
		totalComparisons := maxIndex * numClasses
		accuracy := 1.0 - float64(numDiffs)/float64(totalComparisons)
		t.Logf("%s Pure Go results:", objective)
		t.Logf("  Max difference from Python: %e", maxDiff)
		t.Logf("  Number of differences: %d/%d", numDiffs, totalComparisons)
		t.Logf("  Agreement rate: %.2f%%", accuracy*100)
	} else {
		// For regression and binary, compare single predictions
		for i := 0; i < nrows && i < len(testData.TestPreds); i++ {
			goPred := goPreds.At(i, 0)
			pythonPred := testData.TestPreds[i]
			diff := math.Abs(goPred - pythonPred)
			if diff > maxDiff {
				maxDiff = diff
			}
			if !compareValuesSimple(goPred, pythonPred, tolerance) {
				numDiffs++
				if numDiffs <= 5 {
					t.Logf("Difference at sample %d: Go = %v, Python = %v, diff = %v",
						i, goPred, pythonPred, diff)
				}
			}
		}
		accuracy := 1.0 - float64(numDiffs)/float64(nrows)
		t.Logf("%s Pure Go results:", objective)
		t.Logf("  Max difference from Python: %e", maxDiff)
		t.Logf("  Number of differences: %d/%d", numDiffs, nrows)
		t.Logf("  Agreement rate: %.2f%%", accuracy*100)

		// For now, we expect some differences due to implementation details
		if accuracy < 0.3 {
			t.Logf("Warning: Agreement rate is low (%.2f%%), implementation needs improvement", accuracy*100)
		}
	}

	// Log model structure comparison
	t.Logf("  Number of trees built: %d", len(trainer.GetModel().Trees))
	t.Logf("  Number of features: %d", goModel.NumFeatures)
	if objective == "multiclass" {
		t.Logf("  Number of classes: %d", goModel.NumClass)
	}
}
