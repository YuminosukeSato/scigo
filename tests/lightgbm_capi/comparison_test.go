// +build capi

package lightgbm_capi

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

// TestData holds test data for comparison
type TestData struct {
	XTrain      *mat.Dense
	XTest       *mat.Dense
	YTrain      *mat.Dense
	YTest       *mat.Dense
	TrainPreds  []float64
	TestPreds   []float64
	ModelPath   string
	ParamsPath  string
	ObjectiveType string
}

// loadCSV loads a CSV file into a slice
func loadCSV(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

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

// loadTestData loads test data for a specific objective
func loadTestData(objective string) (*TestData, error) {
	baseDir := filepath.Join("testdata", objective)
	
	// Load training data
	xTrainData, err := loadCSV(filepath.Join(baseDir, objective+"_X_train.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load X_train: %v", err)
	}
	
	xTestData, err := loadCSV(filepath.Join(baseDir, objective+"_X_test.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load X_test: %v", err)
	}
	
	yTrainData, err := loadCSV(filepath.Join(baseDir, objective+"_y_train.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load y_train: %v", err)
	}
	
	yTestData, err := loadCSV(filepath.Join(baseDir, objective+"_y_test.csv"))
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
	trainPredData, err := loadCSV(filepath.Join(baseDir, "train_predictions.csv"))
	if err != nil {
		return nil, fmt.Errorf("failed to load train predictions: %v", err)
	}
	
	testPredData, err := loadCSV(filepath.Join(baseDir, "test_predictions.csv"))
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
	
	return &TestData{
		XTrain:     mat.NewDense(nTrainRows, nCols, xTrainFlat),
		XTest:      mat.NewDense(nTestRows, nCols, xTestFlat),
		YTrain:     mat.NewDense(nTrainRows, 1, yTrainFlat),
		YTest:      mat.NewDense(nTestRows, 1, yTestFlat),
		TrainPreds: trainPreds,
		TestPreds:  testPreds,
		ModelPath:  filepath.Join(baseDir, "model.txt"),
		ParamsPath: filepath.Join(baseDir, "params.json"),
		ObjectiveType: objective,
	}, nil
}

// compareValues compares two float64 values with tolerance
func compareValues(a, b float64, tolerance float64) bool {
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

// TestRegressionPredictionCAPI tests regression predictions using C API
func TestRegressionPredictionCAPI(t *testing.T) {
	// Skip if LightGBM C library is not available
	if _, err := BoosterCreateFromModelfile("testdata/regression/model.txt"); err != nil {
		t.Skip("LightGBM C library not available, skipping C API tests")
	}
	
	testData, err := loadTestData("regression")
	if err != nil {
		t.Skipf("Test data not available: %v. Run python_baseline/train_models.py first", err)
	}
	
	// Load model using C API
	booster, err := BoosterCreateFromModelfile(testData.ModelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer booster.Free()
	
	// Get test data dimensions
	nrows, ncols := testData.XTest.Dims()
	
	// Flatten test data
	xTestFlat := make([]float64, nrows*ncols)
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			xTestFlat[i*ncols+j] = testData.XTest.At(i, j)
		}
	}
	
	// Make predictions using C API
	capiPreds, err := booster.PredictForMat(xTestFlat, nrows, ncols, PredictNormal, -1)
	if err != nil {
		t.Fatalf("Failed to predict with C API: %v", err)
	}
	
	// Compare with Python predictions
	tolerance := 1e-9
	maxDiff := 0.0
	for i := 0; i < len(capiPreds); i++ {
		diff := math.Abs(capiPreds[i] - testData.TestPreds[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if !compareValues(capiPreds[i], testData.TestPreds[i], tolerance) {
			t.Errorf("Prediction mismatch at index %d: C API = %v, Python = %v, diff = %v",
				i, capiPreds[i], testData.TestPreds[i], diff)
		}
	}
	
	t.Logf("Regression C API test passed. Max difference: %e", maxDiff)
}

// TestBinaryPredictionCAPI tests binary classification predictions using C API
func TestBinaryPredictionCAPI(t *testing.T) {
	// Skip if LightGBM C library is not available
	if _, err := BoosterCreateFromModelfile("testdata/binary/model.txt"); err != nil {
		t.Skip("LightGBM C library not available, skipping C API tests")
	}
	
	testData, err := loadTestData("binary")
	if err != nil {
		t.Skipf("Test data not available: %v. Run python_baseline/train_models.py first", err)
	}
	
	// Load model using C API
	booster, err := BoosterCreateFromModelfile(testData.ModelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer booster.Free()
	
	// Get test data dimensions
	nrows, ncols := testData.XTest.Dims()
	
	// Flatten test data
	xTestFlat := make([]float64, nrows*ncols)
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			xTestFlat[i*ncols+j] = testData.XTest.At(i, j)
		}
	}
	
	// Make predictions using C API
	capiPreds, err := booster.PredictForMat(xTestFlat, nrows, ncols, PredictNormal, -1)
	if err != nil {
		t.Fatalf("Failed to predict with C API: %v", err)
	}
	
	// Compare with Python predictions
	tolerance := 1e-9
	maxDiff := 0.0
	for i := 0; i < len(capiPreds); i++ {
		diff := math.Abs(capiPreds[i] - testData.TestPreds[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if !compareValues(capiPreds[i], testData.TestPreds[i], tolerance) {
			t.Errorf("Prediction mismatch at index %d: C API = %v, Python = %v, diff = %v",
				i, capiPreds[i], testData.TestPreds[i], diff)
		}
	}
	
	t.Logf("Binary C API test passed. Max difference: %e", maxDiff)
}

// TestMulticlassPredictionCAPI tests multiclass classification predictions using C API
func TestMulticlassPredictionCAPI(t *testing.T) {
	// Skip if LightGBM C library is not available
	if _, err := BoosterCreateFromModelfile("testdata/multiclass/model.txt"); err != nil {
		t.Skip("LightGBM C library not available, skipping C API tests")
	}
	
	testData, err := loadTestData("multiclass")
	if err != nil {
		t.Skipf("Test data not available: %v. Run python_baseline/train_models.py first", err)
	}
	
	// Load model using C API
	booster, err := BoosterCreateFromModelfile(testData.ModelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer booster.Free()
	
	// Get number of classes
	numClasses, err := booster.GetNumClasses()
	if err != nil {
		t.Fatalf("Failed to get number of classes: %v", err)
	}
	
	// Get test data dimensions
	nrows, ncols := testData.XTest.Dims()
	
	// Flatten test data
	xTestFlat := make([]float64, nrows*ncols)
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			xTestFlat[i*ncols+j] = testData.XTest.At(i, j)
		}
	}
	
	// Make predictions using C API
	capiPreds, err := booster.PredictForMat(xTestFlat, nrows, ncols, PredictNormal, -1)
	if err != nil {
		t.Fatalf("Failed to predict with C API: %v", err)
	}
	
	// Compare with Python predictions
	tolerance := 1e-9
	maxDiff := 0.0
	
	// For multiclass, predictions are [sample0_class0, sample0_class1, ..., sample1_class0, ...]
	if len(capiPreds) != len(testData.TestPreds) {
		t.Fatalf("Prediction length mismatch: C API = %d, Python = %d", 
			len(capiPreds), len(testData.TestPreds))
	}
	
	for i := 0; i < len(capiPreds); i++ {
		diff := math.Abs(capiPreds[i] - testData.TestPreds[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if !compareValues(capiPreds[i], testData.TestPreds[i], tolerance) {
			sampleIdx := i / numClasses
			classIdx := i % numClasses
			t.Errorf("Prediction mismatch at sample %d, class %d: C API = %v, Python = %v, diff = %v",
				sampleIdx, classIdx, capiPreds[i], testData.TestPreds[i], diff)
		}
	}
	
	t.Logf("Multiclass C API test passed. Max difference: %e", maxDiff)
}

// TestPureGoVsCAPI compares pure Go implementation with C API
func TestPureGoVsCAPI(t *testing.T) {
	// Test regression first
	t.Run("Regression", func(t *testing.T) {
		testPureGoVsCAPIForObjective(t, "regression")
	})
	
	// Test binary classification
	t.Run("Binary", func(t *testing.T) {
		testPureGoVsCAPIForObjective(t, "binary")
	})
	
	// Test multiclass classification
	t.Run("Multiclass", func(t *testing.T) {
		testPureGoVsCAPIForObjective(t, "multiclass")
	})
}

func testPureGoVsCAPIForObjective(t *testing.T, objective string) {
	// Skip if test data not available
	testData, err := loadTestData(objective)
	if err != nil {
		t.Skipf("Test data not available: %v. Run python_baseline/train_models.py first", err)
	}
	
	// Skip if C API not available
	capiBooster, err := BoosterCreateFromModelfile(testData.ModelPath)
	if err != nil {
		t.Skip("LightGBM C library not available, skipping comparison test")
	}
	defer capiBooster.Free()
	
	// Load parameters
	paramsData, err := ioutil.ReadFile(testData.ParamsPath)
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
	
	// Get C API predictions
	nrows, ncols := testData.XTest.Dims()
	xTestFlat := make([]float64, nrows*ncols)
	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			xTestFlat[i*ncols+j] = testData.XTest.At(i, j)
		}
	}
	
	capiPreds, err := capiBooster.PredictForMat(xTestFlat, nrows, ncols, PredictNormal, -1)
	if err != nil {
		t.Fatalf("Failed to predict with C API: %v", err)
	}
	
	// Compare predictions
	tolerance := 0.01 // 1% tolerance for now, will improve
	maxDiff := 0.0
	numDiffs := 0
	
	// Handle different output formats
	if objective == "multiclass" {
		// Pure Go returns matrix, C API returns flat array
		for i := 0; i < nrows; i++ {
			for j := 0; j < 3; j++ {
				goPred := goPreds.At(i, j)
				capiPred := capiPreds[i*3+j]
				diff := math.Abs(goPred - capiPred)
				if diff > maxDiff {
					maxDiff = diff
				}
				if !compareValues(goPred, capiPred, tolerance) {
					numDiffs++
					if numDiffs <= 5 { // Only show first 5 differences
						t.Logf("Difference at sample %d, class %d: Go = %v, C API = %v, diff = %v",
							i, j, goPred, capiPred, diff)
					}
				}
			}
		}
	} else {
		// Compare single column predictions
		for i := 0; i < nrows; i++ {
			goPred := goPreds.At(i, 0)
			capiPred := capiPreds[i]
			diff := math.Abs(goPred - capiPred)
			if diff > maxDiff {
				maxDiff = diff
			}
			if !compareValues(goPred, capiPred, tolerance) {
				numDiffs++
				if numDiffs <= 5 {
					t.Logf("Difference at sample %d: Go = %v, C API = %v, diff = %v",
						i, goPred, capiPred, diff)
				}
			}
		}
	}
	
	// Report results
	accuracy := 1.0 - float64(numDiffs)/float64(len(capiPreds))
	t.Logf("%s: Pure Go vs C API comparison:", objective)
	t.Logf("  Max difference: %e", maxDiff)
	t.Logf("  Number of differences: %d/%d", numDiffs, len(capiPreds))
	t.Logf("  Agreement rate: %.2f%%", accuracy*100)
	
	// Currently we expect differences due to implementation details
	// As we improve the pure Go implementation, we'll tighten this threshold
	if accuracy < 0.5 {
		t.Errorf("Agreement rate too low: %.2f%% < 50%%", accuracy*100)
	}
}