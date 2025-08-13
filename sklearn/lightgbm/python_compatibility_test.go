package lightgbm

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// TestPythonLightGBMCompatibility verifies exact compatibility with Python LightGBM
func TestPythonLightGBMCompatibility(t *testing.T) {
	// Test data directory
	testDataDir := "testdata/compatibility"

	// Check if test data exists
	if _, err := os.Stat(testDataDir); os.IsNotExist(err) {
		t.Skip("Compatibility test data not found. Run generate_compatibility_data.py first")
	}

	testCases := []struct {
		name      string
		modelFile string
		dataFile  string
		predFile  string
		tolerance float64
		taskType  string // "regression", "binary", "multiclass"
	}{
		{
			name:      "Regression - Boston Housing",
			modelFile: "regression_model.txt",
			dataFile:  "regression_X_test.csv",
			predFile:  "regression_predictions.csv",
			tolerance: 1e-6,
			taskType:  "regression",
		},
		{
			name:      "Binary Classification - Breast Cancer",
			modelFile: "binary_model.txt",
			dataFile:  "binary_X_test.csv",
			predFile:  "binary_predictions.csv",
			tolerance: 1e-6,
			taskType:  "binary",
		},
		{
			name:      "Multiclass - Iris",
			modelFile: "multiclass_model.txt",
			dataFile:  "multiclass_X_test.csv",
			predFile:  "multiclass_predictions.csv",
			tolerance: 1e-5,
			taskType:  "multiclass",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Load model
			modelPath := filepath.Join(testDataDir, tc.modelFile)
			model, err := LoadFromFile(modelPath)
			if err != nil {
				t.Skipf("Model file not found: %v", err)
			}

			// Load test data
			dataPath := filepath.Join(testDataDir, tc.dataFile)
			X := loadCSVData(t, dataPath)

			// Load expected predictions
			predPath := filepath.Join(testDataDir, tc.predFile)
			expectedPreds := loadCSVPredictions(t, predPath)

			// Make predictions
			predictor := NewPredictor(model)
			predictions, err := predictor.Predict(X)
			require.NoError(t, err)

			// Compare predictions
			// Convert predictions to Dense if needed
			var predDense *mat.Dense
			switch p := predictions.(type) {
			case *mat.Dense:
				predDense = p
			default:
				rows, cols := predictions.Dims()
				predDense = mat.NewDense(rows, cols, nil)
				for i := 0; i < rows; i++ {
					for j := 0; j < cols; j++ {
						predDense.Set(i, j, predictions.At(i, j))
					}
				}
			}
			comparePredictionsDense(t, expectedPreds, predDense, tc.tolerance, tc.taskType)
		})
	}
}

// TestGenerateCompatibilityData creates test data using Python LightGBM
func TestGenerateCompatibilityData(t *testing.T) {
	// Skip if Python is not available
	if _, err := os.Stat("generate_compatibility_data.py"); os.IsNotExist(err) {
		t.Skip("generate_compatibility_data.py not found")
	}

	// This test generates the compatibility test data
	// Run: python generate_compatibility_data.py
	t.Skip("Run generate_compatibility_data.py manually to generate test data")
}

// loadCSVData loads feature data from CSV
func loadCSVData(t *testing.T, filepath string) *mat.Dense {
	file, err := os.Open(filepath)
	if err != nil {
		t.Skipf("Data file not found: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	require.NoError(t, err)

	// Skip header if present
	startRow := 0
	if len(records) > 0 {
		// Check if first row is header (non-numeric)
		if _, err := strconv.ParseFloat(records[0][0], 64); err != nil {
			startRow = 1
		}
	}

	rows := len(records) - startRow
	cols := len(records[startRow])

	data := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(records[i+startRow][j], 64)
			require.NoError(t, err)
			data.Set(i, j, val)
		}
	}

	return data
}

// loadCSVPredictions loads prediction data from CSV
func loadCSVPredictions(t *testing.T, filepath string) *mat.Dense {
	file, err := os.Open(filepath)
	if err != nil {
		t.Skipf("Prediction file not found: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	require.NoError(t, err)

	// Skip header if present
	startRow := 0
	if len(records) > 0 {
		// Check if first row is header
		if _, err := strconv.ParseFloat(records[0][0], 64); err != nil {
			startRow = 1
		}
	}

	rows := len(records) - startRow
	cols := len(records[startRow])

	preds := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(records[i+startRow][j], 64)
			require.NoError(t, err)
			preds.Set(i, j, val)
		}
	}

	return preds
}

// comparePredictions compares predictions with tolerance
// Local helper to compare predictions with tolerance
func comparePredictionsDense(t *testing.T, expected, actual *mat.Dense, tolerance float64, taskType string) {
	expRows, expCols := expected.Dims()
	actRows, actCols := actual.Dims()

	// For multiclass, columns might differ (probabilities vs classes)
	assert.Equal(t, expRows, actRows, "Number of predictions should match")

	switch taskType {
	case "regression":
		assert.Equal(t, expCols, actCols, "Regression should have same dimensions")
		for i := 0; i < expRows; i++ {
			for j := 0; j < expCols; j++ {
				exp := expected.At(i, j)
				act := actual.At(i, j)
				assert.InDelta(t, exp, act, tolerance,
					"Prediction mismatch at row %d, col %d: expected %f, got %f", i, j, exp, act)
			}
		}

	case "binary":
		// Binary can be single column (probability of positive class) or two columns
		if expCols == 1 && actCols == 1 {
			// Compare single column predictions
			for i := 0; i < expRows; i++ {
				exp := expected.At(i, 0)
				act := actual.At(i, 0)
				assert.InDelta(t, exp, act, tolerance,
					"Binary prediction mismatch at row %d: expected %f, got %f", i, exp, act)
			}
		} else if expCols == 2 && actCols == 2 {
			// Compare probability predictions
			for i := 0; i < expRows; i++ {
				for j := 0; j < 2; j++ {
					exp := expected.At(i, j)
					act := actual.At(i, j)
					assert.InDelta(t, exp, act, tolerance,
						"Binary probability mismatch at row %d, class %d: expected %f, got %f", i, j, exp, act)
				}
			}
		} else {
			t.Errorf("Dimension mismatch for binary classification: expected %dx%d, got %dx%d",
				expRows, expCols, actRows, actCols)
		}

	case "multiclass":
		// Multiclass predictions
		switch expCols {
		case actCols:
			// Same number of classes
			for i := 0; i < expRows; i++ {
				for j := 0; j < expCols; j++ {
					exp := expected.At(i, j)
					act := actual.At(i, j)
					assert.InDelta(t, exp, act, tolerance,
						"Multiclass prediction mismatch at row %d, class %d: expected %f, got %f", i, j, exp, act)
				}
			}
		case 1:
			// Expected is class labels, actual might be probabilities
			// Find the argmax of actual
			for i := 0; i < expRows; i++ {
				expClass := int(expected.At(i, 0))

				// Find predicted class (argmax)
				maxProb := actual.At(i, 0)
				maxClass := 0
				for j := 1; j < actCols; j++ {
					if actual.At(i, j) > maxProb {
						maxProb = actual.At(i, j)
						maxClass = j
					}
				}

				assert.Equal(t, expClass, maxClass,
					"Multiclass prediction mismatch at row %d: expected class %d, got %d", i, expClass, maxClass)
			}
		default:
			t.Errorf("Dimension mismatch for multiclass: expected %dx%d, got %dx%d",
				expRows, expCols, actRows, actCols)
		}

	default:
		t.Fatalf("Unknown task type: %s", taskType)
	}
}

// TestPythonModelLoad tests loading various Python-trained models
func TestPythonModelLoad(t *testing.T) {
	testCases := []struct {
		name     string
		filename string
	}{
		{"Simple Regression", "testdata/simple_regression.txt"},
		{"Complex Tree", "testdata/complex_tree.txt"},
		{"Deep Tree", "testdata/deep_tree.txt"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := os.Stat(tc.filename); os.IsNotExist(err) {
				t.Skipf("Test file %s not found", tc.filename)
			}

			model, err := LoadFromFile(tc.filename)
			require.NoError(t, err)

			// Basic validation
			assert.NotNil(t, model)
			assert.Greater(t, len(model.Trees), 0)
			assert.Greater(t, model.NumFeatures, 0)
		})
	}
}

// BenchmarkPythonCompatPrediction benchmarks prediction performance
func BenchmarkPythonCompatPrediction(b *testing.B) {
	// Create a simple model for benchmarking
	model := &Model{
		NumFeatures: 10,
		NumClass:    1,
		Trees:       make([]Tree, 100), // 100 trees
	}

	// Initialize trees with simple structure
	for i := range model.Trees {
		model.Trees[i] = Tree{
			Nodes: []Node{
				{NodeType: NumericalNode, SplitFeature: i % 10, Threshold: 0.5, LeftChild: 1, RightChild: 2},
				{NodeType: LeafNode, LeafValue: 0.1},
				{NodeType: LeafNode, LeafValue: 0.2},
			},
		}
	}

	predictor := NewPredictor(model)

	// Create test data
	X := mat.NewDense(1000, 10, nil)
	for i := 0; i < 1000; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i*j)/1000.0)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = predictor.Predict(X)
	}
}

// TestNumericalStability tests numerical stability of predictions
func TestNumericalStability(t *testing.T) {
	// Create model with extreme values
	model := &Model{
		NumFeatures: 2,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: NumericalNode, SplitFeature: 0, Threshold: 1e10, LeftChild: 1, RightChild: 2},
					{NodeType: LeafNode, LeafValue: 1e-10},
					{NodeType: LeafNode, LeafValue: 1e10},
				},
			},
		},
	}

	predictor := NewPredictor(model)

	// Test with extreme values
	testCases := []struct {
		name string
		X    *mat.Dense
	}{
		{
			name: "Very small values",
			X:    mat.NewDense(1, 2, []float64{1e-20, 1e-20}),
		},
		{
			name: "Very large values",
			X:    mat.NewDense(1, 2, []float64{1e20, 1e20}),
		},
		{
			name: "Mixed extreme values",
			X:    mat.NewDense(1, 2, []float64{1e-20, 1e20}),
		},
		{
			name: "Zero values",
			X:    mat.NewDense(1, 2, []float64{0, 0}),
		},
		{
			name: "NaN values",
			X:    mat.NewDense(1, 2, []float64{math.NaN(), 0}),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			preds, err := predictor.Predict(tc.X)
			require.NoError(t, err)

			// Check predictions are not infinite
			val := preds.At(0, 0)
			if !math.IsNaN(tc.X.At(0, 0)) {
				assert.False(t, math.IsInf(val, 0), "Prediction should not be infinite")
			}
		})
	}
}

// TestConcurrentPredictions tests thread safety
func TestConcurrentPredictions(t *testing.T) {
	// Create a simple model
	model := &Model{
		NumFeatures: 5,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: NumericalNode, SplitFeature: 0, Threshold: 0.5, LeftChild: 1, RightChild: 2},
					{NodeType: LeafNode, LeafValue: 0.1},
					{NodeType: LeafNode, LeafValue: 0.2},
				},
			},
		},
	}

	predictor := NewPredictor(model)

	// Create test data
	X := mat.NewDense(100, 5, nil)
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i+j)/100.0)
		}
	}

	// Run predictions concurrently
	numGoroutines := 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_, err := predictor.Predict(X)
				assert.NoError(t, err)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
}

// TestMemoryEfficiency tests memory usage patterns
func TestMemoryEfficiency(t *testing.T) {
	// Create a large model
	model := &Model{
		NumFeatures: 100,
		NumClass:    1,
		Trees:       make([]Tree, 1000), // 1000 trees
	}

	// Initialize with minimal nodes
	for i := range model.Trees {
		model.Trees[i] = Tree{
			Nodes: []Node{
				{NodeType: LeafNode, LeafValue: float64(i) / 1000.0},
			},
		}
	}

	predictor := NewPredictor(model)

	// Create large test dataset
	X := mat.NewDense(10000, 100, nil)

	// Make predictions
	preds, err := predictor.Predict(X)
	require.NoError(t, err)

	// Verify predictions shape
	rows, cols := preds.Dims()
	assert.Equal(t, 10000, rows)
	assert.Equal(t, 1, cols)
}
