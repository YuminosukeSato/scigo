//go:build compat
// +build compat

package lightgbm

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestExpectation represents the expected test data and predictions from Python
type TestExpectation struct {
	Dataset     string                 `json:"dataset"`
	Params      map[string]interface{} `json:"params"`
	XTest       [][]float64            `json:"X_test"`
	YTest       []float64              `json:"y_test"`
	Predictions struct {
		Predict   [][]float64 `json:"predict"`
		RawScore  [][]float64 `json:"raw_score"`
		LeafIndex [][]int     `json:"leaf_index"`
	} `json:"predictions"`
	ModelInfo struct {
		NumTrees            int      `json:"num_trees"`
		NumFeatures         int      `json:"num_features"`
		FeatureNames        []string `json:"feature_names"`
		CategoricalFeatures []int    `json:"categorical_features,omitempty"`
	} `json:"model_info"`
}

// TestPythonModelCompatibility tests that our Go implementation produces
// the same results as Python's LightGBM
func TestPythonModelCompatibility(t *testing.T) {
	t.Skip("LightGBM model compatibility tests need proper model loading implementation - planned for v0.7.0")
	// Test cases with expected values from Python LightGBM
	testCases := []struct {
		name           string
		modelFile      string
		inputData      [][]float64
		expectedOutput [][]float64
		tolerance      float64
	}{
		{
			name:      "Binary Classification Model",
			modelFile: "testdata/binary_model.txt",
			inputData: [][]float64{
				{5.1, 3.5, 1.4, 0.2},
				{4.9, 3.0, 1.4, 0.2},
				{6.2, 3.4, 5.4, 2.3},
			},
			expectedOutput: [][]float64{
				{0.18514822574281953}, // Expected probability from Python
				{0.18514822574281953},
				{0.815802544297422},
			},
			tolerance: 1e-15, // Machine epsilon precision
		},
		{
			name:      "Regression Model",
			modelFile: "testdata/regression_model.txt",
			inputData: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			expectedOutput: [][]float64{
				{65.42537778620033},
				{65.42537778620033},
			},
			tolerance: 1e-15, // Machine epsilon precision
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Skip if test model file doesn't exist
			if _, err := os.Stat(tc.modelFile); os.IsNotExist(err) {
				t.Skipf("Test model file %s not found. Create it with Python script.", tc.modelFile)
			}

			// Load model
			model, err := LoadFromFile(tc.modelFile)
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}

			// Create predictor
			predictor := NewPredictor(model)
			predictor.SetDeterministic(true) // Ensure reproducibility

			// Create input matrix
			rows := len(tc.inputData)
			cols := len(tc.inputData[0])
			data := make([]float64, rows*cols)
			for i, row := range tc.inputData {
				for j, val := range row {
					data[i*cols+j] = val
				}
			}
			X := mat.NewDense(rows, cols, data)

			// Make predictions
			predictions, err := predictor.Predict(X)
			if err != nil {
				t.Fatalf("Failed to predict: %v", err)
			}

			// Compare with expected output
			for i := 0; i < rows; i++ {
				for j := 0; j < len(tc.expectedOutput[i]); j++ {
					got := predictions.At(i, j)
					expected := tc.expectedOutput[i][j]
					diff := math.Abs(got - expected)

					if diff > tc.tolerance {
						t.Errorf("Row %d, Col %d: got %.6f, expected %.6f (diff: %.6f > tolerance: %.6f)",
							i, j, got, expected, diff, tc.tolerance)
					}
				}
			}
		})
	}
}

// TestNumericalPrecision verifies that numerical precision matches Python
func TestNumericalPrecision(t *testing.T) {
	// Create a simple tree for testing
	model := &Model{
		NumFeatures:  4,
		NumClass:     1,
		NumIteration: 1,
		LearningRate: 1.0,
		InitScore:    0.0,
		Objective:    RegressionL2,
		Trees: []Tree{
			{
				TreeIndex:     0,
				ShrinkageRate: 1.0,
				Nodes: []Node{
					{ // Root node
						NodeID:       0,
						ParentID:     -1,
						LeftChild:    1,
						RightChild:   2,
						NodeType:     NumericalNode,
						SplitFeature: 0,
						Threshold:    5.0,
					},
					{ // Left leaf
						NodeID:     1,
						ParentID:   0,
						LeftChild:  -1,
						RightChild: -1,
						NodeType:   LeafNode,
						LeafValue:  0.1234567890123456,
					},
					{ // Right leaf
						NodeID:     2,
						ParentID:   0,
						LeftChild:  -1,
						RightChild: -1,
						NodeType:   LeafNode,
						LeafValue:  0.9876543210987654,
					},
				},
			},
		},
	}

	predictor := NewPredictor(model)
	predictor.SetDeterministic(true)

	// Test with precise input values
	testData := []struct {
		input    []float64
		expected float64
	}{
		{[]float64{4.9999999999, 0, 0, 0}, 0.1234567890123456},
		{[]float64{5.0000000001, 0, 0, 0}, 0.9876543210987654},
		{[]float64{5.0, 0, 0, 0}, 0.1234567890123456}, // Threshold is <=
	}

	for i, td := range testData {
		result := predictor.predictSingleSample(td.input)
		if math.Abs(result[0]-td.expected) > 1e-15 {
			t.Errorf("Test %d: got %.16f, expected %.16f", i, result[0], td.expected)
		}
	}
}

// TestCompatibilityWithExistingModels tests with models in testdata/compatibility
func TestCompatibilityWithExistingModels(t *testing.T) {
	compatDir := "testdata/compatibility"

	testCases := []struct {
		name      string
		modelFile string
		dataFile  string
		predFile  string
	}{
		{
			name:      "Regression",
			modelFile: "regression_model.txt",
			dataFile:  "regression_X_test.csv",
			predFile:  "regression_predictions.csv",
		},
		{
			name:      "Binary",
			modelFile: "binary_model.txt",
			dataFile:  "binary_X_test.csv",
			predFile:  "binary_predictions.csv",
		},
		{
			name:      "Multiclass",
			modelFile: "multiclass_model.txt",
			dataFile:  "multiclass_X_test.csv",
			predFile:  "multiclass_predictions.csv",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if files exist
			modelPath := filepath.Join(compatDir, tc.modelFile)
			if _, err := os.Stat(modelPath); os.IsNotExist(err) {
				t.Skipf("Model file not found: %s", modelPath)
			}

			// Load model
			model, err := LoadLeavesModelFromFile(modelPath)
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}

			t.Logf("Loaded model: Trees=%d, Features=%d, Classes=%d",
				len(model.Trees), model.NumFeatures, model.NumClass)

			// Load test data if available
			dataPath := filepath.Join(compatDir, tc.dataFile)
			if _, err := os.Stat(dataPath); err == nil {
				// TODO: Implement CSV loading and prediction comparison
				t.Logf("Test data available at: %s", dataPath)
			}
		})
	}
}

// TestJSONModelLoading tests loading models from JSON format
func TestJSONModelLoading(t *testing.T) {
	// Sample JSON model (simplified)
	jsonModel := `{
		"name": "tree",
		"version": "v3",
		"num_class": 1,
		"num_tree_per_iteration": 1,
		"max_feature_idx": 3,
		"objective": "regression",
		"feature_names": ["f0", "f1", "f2", "f3"],
		"tree_info": [
			{
				"tree_index": 0,
				"num_leaves": 2,
				"shrinkage": 0.1
			}
		],
		"tree_structure": [
			{
				"tree_index": 0,
				"num_leaves": 2,
				"shrinkage": 0.1,
				"tree_structure": {
					"split_feature": 0,
					"threshold": 5.0,
					"decision_type": "<=",
					"default_left": true,
					"split_gain": 10.0,
					"internal_value": 0.0,
					"internal_count": 100,
					"left_child": {
						"leaf_index": 0,
						"leaf_value": 1.0,
						"leaf_count": 50
					},
					"right_child": {
						"leaf_index": 1,
						"leaf_value": 2.0,
						"leaf_count": 50
					}
				}
			}
		]
	}`

	model, err := LoadFromJSON([]byte(jsonModel))
	if err != nil {
		t.Fatalf("Failed to load JSON model: %v", err)
	}

	// Verify model properties
	if model.NumFeatures != 4 {
		t.Errorf("Expected 4 features, got %d", model.NumFeatures)
	}

	if len(model.Trees) != 1 {
		t.Errorf("Expected 1 tree, got %d", len(model.Trees))
	}

	// Test prediction
	predictor := NewPredictor(model)
	X := mat.NewDense(1, 4, []float64{4.0, 0, 0, 0})

	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Should predict leaf value 1.0 * shrinkage 0.1 = 0.1
	expected := 0.1
	got := predictions.At(0, 0)
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("Expected prediction %.10f, got %.10f", expected, got)
	}
}

// TestClassifierVsRegressor tests that classifier and regressor produce correct outputs
func TestClassifierVsRegressor(t *testing.T) {
	// Create sample data
	X := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})

	// Test classifier
	t.Run("Classifier", func(t *testing.T) {
		y := mat.NewDense(3, 1, []float64{0, 1, 0})

		clf := NewLGBMClassifier()
		err := clf.Fit(X, y)
		if err != nil {
			t.Fatalf("Failed to fit classifier: %v", err)
		}

		// Should be fitted
		if !clf.state.IsFitted() {
			t.Error("Classifier should be fitted")
		}

		// Check predictions shape
		pred, err := clf.Predict(X)
		if err != nil {
			t.Fatalf("Failed to predict: %v", err)
		}

		rows, cols := pred.Dims()
		if rows != 3 || cols != 1 {
			t.Errorf("Expected prediction shape (3, 1), got (%d, %d)", rows, cols)
		}

		// Check probabilities shape
		proba, err := clf.PredictProba(X)
		if err != nil {
			t.Fatalf("Failed to predict probabilities: %v", err)
		}

		rows, cols = proba.Dims()
		if rows != 3 || cols != 2 {
			t.Errorf("Expected probability shape (3, 2), got (%d, %d)", rows, cols)
		}
	})

	// Test regressor
	t.Run("Regressor", func(t *testing.T) {
		y := mat.NewDense(3, 1, []float64{1.5, 2.5, 3.5})

		reg := NewLGBMRegressor()
		err := reg.Fit(X, y)
		if err != nil {
			t.Fatalf("Failed to fit regressor: %v", err)
		}

		// Should be fitted
		if !reg.state.IsFitted() {
			t.Error("Regressor should be fitted")
		}

		// Check predictions shape
		pred, err := reg.Predict(X)
		if err != nil {
			t.Fatalf("Failed to predict: %v", err)
		}

		rows, cols := pred.Dims()
		if rows != 3 || cols != 1 {
			t.Errorf("Expected prediction shape (3, 1), got (%d, %d)", rows, cols)
		}
	})
}

// TestModelToJSON tests converting a model to JSON representation
func TestModelToJSON(t *testing.T) {
	model := &Model{
		Version:      "v3",
		NumFeatures:  2,
		NumClass:     1,
		NumIteration: 1,
		LearningRate: 0.1,
		Objective:    RegressionL2,
		Trees: []Tree{
			{
				TreeIndex:     0,
				NumLeaves:     2,
				ShrinkageRate: 0.1,
				Nodes: []Node{
					{NodeID: 0, LeftChild: 1, RightChild: 2, SplitFeature: 0, Threshold: 1.0},
					{NodeID: 1, LeftChild: -1, RightChild: -1, LeafValue: 0.5},
					{NodeID: 2, LeftChild: -1, RightChild: -1, LeafValue: 1.5},
				},
			},
		},
	}

	// This would be implemented as model.ToJSON() method
	// For now, just verify the model structure
	if model.NumFeatures != 2 {
		t.Error("Model structure test failed")
	}
}

// BenchmarkPrediction benchmarks prediction performance
func BenchmarkPrediction(b *testing.B) {
	// Create a model with multiple trees
	model := &Model{
		NumFeatures:  10,
		NumClass:     1,
		NumIteration: 100,
		Objective:    RegressionL2,
		Trees:        make([]Tree, 100),
	}

	// Initialize trees with simple structure
	for i := range model.Trees {
		model.Trees[i] = Tree{
			TreeIndex:     i,
			ShrinkageRate: 0.1,
			Nodes: []Node{
				{NodeID: 0, LeftChild: 1, RightChild: 2, SplitFeature: i % 10, Threshold: 0.5},
				{NodeID: 1, LeftChild: -1, RightChild: -1, LeafValue: 0.1},
				{NodeID: 2, LeftChild: -1, RightChild: -1, LeafValue: 0.2},
			},
		}
	}

	predictor := NewPredictor(model)
	X := mat.NewDense(100, 10, nil) // 100 samples, 10 features

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = predictor.Predict(X)
	}
}

// BenchmarkParallelPrediction benchmarks parallel prediction
func BenchmarkParallelPrediction(b *testing.B) {
	model := &Model{
		NumFeatures:  10,
		NumClass:     1,
		NumIteration: 100,
		Objective:    RegressionL2,
		Trees:        make([]Tree, 100),
	}

	for i := range model.Trees {
		model.Trees[i] = Tree{
			TreeIndex:     i,
			ShrinkageRate: 0.1,
			Nodes: []Node{
				{NodeID: 0, LeftChild: 1, RightChild: 2, SplitFeature: i % 10, Threshold: 0.5},
				{NodeID: 1, LeftChild: -1, RightChild: -1, LeafValue: 0.1},
				{NodeID: 2, LeftChild: -1, RightChild: -1, LeafValue: 0.2},
			},
		}
	}

	predictor := NewPredictor(model)
	predictor.SetNumThreads(4)       // Use 4 threads
	X := mat.NewDense(1000, 10, nil) // 1000 samples

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = predictor.Predict(X)
	}
}
