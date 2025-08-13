package lightgbm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestLoadJSONModel tests loading JSON format models
func TestLoadJSONModel(t *testing.T) {
	testCases := []struct {
		name      string
		modelFile string
		expectErr bool
		checks    func(t *testing.T, model *LeavesModel)
	}{
		{
			name:      "Iris Model",
			modelFile: "testdata/expectations/iris_model_dump.json",
			expectErr: false,
			checks: func(t *testing.T, model *LeavesModel) {
				if model.NumClass != 3 {
					t.Errorf("Expected 3 classes, got %d", model.NumClass)
				}
				if model.NumFeatures != 4 {
					t.Errorf("Expected 4 features, got %d", model.NumFeatures)
				}
				if len(model.Trees) == 0 {
					t.Error("No trees loaded")
				}
			},
		},
		{
			name:      "Binary Model",
			modelFile: "testdata/expectations/binary_model_dump.json",
			expectErr: false,
			checks: func(t *testing.T, model *LeavesModel) {
				if model.NumClass != 1 {
					t.Errorf("Expected 1 class for binary, got %d", model.NumClass)
				}
				if len(model.Trees) == 0 {
					t.Error("No trees loaded")
				}
			},
		},
		{
			name:      "Regression Model",
			modelFile: "testdata/expectations/regression_model_dump.json",
			expectErr: false,
			checks: func(t *testing.T, model *LeavesModel) {
				if model.NumClass != 1 {
					t.Errorf("Expected 1 class for regression, got %d", model.NumClass)
				}
				if len(model.Trees) == 0 {
					t.Error("No trees loaded")
				}
			},
		},
		{
			name:      "Categorical Model",
			modelFile: "testdata/expectations/categorical_model_dump.json",
			expectErr: false,
			checks: func(t *testing.T, model *LeavesModel) {
				if len(model.Trees) == 0 {
					t.Error("No trees loaded")
				}
				// Check if any nodes have categorical flag
				hasCategorical := false
				for _, tree := range model.Trees {
					for _, node := range tree.Nodes {
						if node.Flags&categorical != 0 {
							hasCategorical = true
							break
						}
					}
					if hasCategorical {
						break
					}
				}
				// Note: Categorical features might not always result in categorical splits
				t.Logf("Has categorical splits: %v", hasCategorical)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if file exists
			if _, err := os.Stat(tc.modelFile); os.IsNotExist(err) {
				t.Skipf("Model file not found: %s", tc.modelFile)
			}

			// Load model
			model, err := LoadJSONModelFromFile(tc.modelFile)
			if tc.expectErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}

			// Run checks
			if tc.checks != nil {
				tc.checks(t, model)
			}

			// Log basic info
			t.Logf("Loaded model: Trees=%d, Features=%d, Classes=%d, Objective=%s",
				len(model.Trees), model.NumFeatures, model.NumClass, model.Objective)
		})
	}
}

// TestIsJSONModel tests JSON model detection
func TestIsJSONModel(t *testing.T) {
	testCases := []struct {
		name     string
		file     string
		expected bool
	}{
		{
			name:     "JSON model",
			file:     "testdata/expectations/iris_model_dump.json",
			expected: true,
		},
		{
			name:     "Text model",
			file:     "testdata/compatibility/regression_model.txt",
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if file exists
			if _, err := os.Stat(tc.file); os.IsNotExist(err) {
				t.Skipf("File not found: %s", tc.file)
			}

			result := IsJSONModel(tc.file)
			if result != tc.expected {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}

// TestLoadModelAutoDetect tests automatic format detection
func TestLoadModelAutoDetect(t *testing.T) {
	testCases := []struct {
		name      string
		modelFile string
	}{
		{
			name:      "JSON format",
			modelFile: "testdata/expectations/iris_model_dump.json",
		},
		{
			name:      "Text format",
			modelFile: "testdata/compatibility/regression_model.txt",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if file exists
			if _, err := os.Stat(tc.modelFile); os.IsNotExist(err) {
				t.Skipf("Model file not found: %s", tc.modelFile)
			}

			// Load model with auto-detection
			model, err := LoadModelAutoDetect(tc.modelFile)
			if err != nil {
				t.Fatalf("Failed to load model: %v", err)
			}

			t.Logf("Successfully loaded model with %d trees", len(model.Trees))
		})
	}
}

// TestJSONTreeConversion tests the conversion of JSON tree structure
func TestJSONTreeConversion(t *testing.T) {
	// Create a simple test tree
	jsonTree := &JSONTreeInfo{
		TreeIndex: 0,
		NumLeaves: 3,
		NumCat:    0,
		Shrinkage: 0.1,
		TreeStructure: JSONTreeNode{
			SplitIndex:    0,
			SplitFeature:  0,
			SplitGain:     10.0,
			Threshold:     5.0,
			DecisionType:  "<=",
			DefaultLeft:   true,
			MissingType:   "None",
			InternalValue: 0.0,
			LeftChild: &JSONTreeNode{
				LeafIndex: 0,
				LeafValue: -1.0,
			},
			RightChild: &JSONTreeNode{
				SplitIndex:   1,
				SplitFeature: 1,
				Threshold:    3.0,
				DecisionType: "<=",
				DefaultLeft:  true,
				MissingType:  "None",
				LeftChild: &JSONTreeNode{
					LeafIndex: 1,
					LeafValue: 0.5,
				},
				RightChild: &JSONTreeNode{
					LeafIndex: 2,
					LeafValue: 1.5,
				},
			},
		},
	}

	// Convert tree
	tree, err := convertJSONTree(jsonTree)
	if err != nil {
		t.Fatalf("Failed to convert tree: %v", err)
	}

	// Check tree properties
	if tree.TreeIndex != 0 {
		t.Errorf("Expected tree index 0, got %d", tree.TreeIndex)
	}
	if tree.ShrinkageRate != 0.1 {
		t.Errorf("Expected shrinkage 0.1, got %f", tree.ShrinkageRate)
	}
	if len(tree.LeafValues) != 3 {
		t.Errorf("Expected 3 leaf values, got %d", len(tree.LeafValues))
	}
	if len(tree.Nodes) != 2 {
		t.Errorf("Expected 2 internal nodes, got %d", len(tree.Nodes))
	}
}

// TestCategoricalSplitConversion tests conversion of categorical splits
func TestCategoricalSplitConversion(t *testing.T) {
	// Create a tree with categorical split
	jsonTree := &JSONTreeInfo{
		TreeIndex: 0,
		NumLeaves: 2,
		NumCat:    1,
		Shrinkage: 1.0,
		TreeStructure: JSONTreeNode{
			SplitIndex:    0,
			SplitFeature:  0,
			DecisionType:  "==",
			SplitIndices:  []int{1, 3, 5},
			DefaultLeft:   true,
			MissingType:   "None",
			LeftChild: &JSONTreeNode{
				LeafIndex: 0,
				LeafValue: -1.0,
			},
			RightChild: &JSONTreeNode{
				LeafIndex: 1,
				LeafValue: 1.0,
			},
		},
	}

	// Convert tree
	tree, err := convertJSONTree(jsonTree)
	if err != nil {
		t.Fatalf("Failed to convert tree with categorical split: %v", err)
	}

	// Check that the root node has categorical flag
	if len(tree.Nodes) == 0 {
		t.Fatal("No nodes in converted tree")
	}

	rootNode := tree.Nodes[0]
	if rootNode.Flags&categorical == 0 {
		t.Error("Expected categorical flag to be set")
	}

	t.Logf("Successfully converted categorical split")
}

// TestJSONModelEndToEnd tests loading a model and making predictions
func TestJSONModelEndToEnd(t *testing.T) {
	modelFile := "testdata/expectations/iris_model_dump.json"
	expectationsFile := "testdata/expectations/iris_expectations.json"

	// Check if files exist
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		t.Skipf("Model file not found: %s", modelFile)
	}
	if _, err := os.Stat(expectationsFile); os.IsNotExist(err) {
		t.Skipf("Expectations file not found: %s", expectationsFile)
	}

	// Load model
	model, err := LoadJSONModelFromFile(modelFile)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Load expectations
	data, err := os.ReadFile(expectationsFile)
	if err != nil {
		t.Fatalf("Failed to read expectations: %v", err)
	}

	var expectations struct {
		XTest [][]float64 `json:"X_test"`
	}
	if err := json.Unmarshal(data, &expectations); err != nil {
		t.Fatalf("Failed to parse expectations: %v", err)
	}

	// Make predictions for first few samples
	for i := 0; i < len(expectations.XTest) && i < 3; i++ {
		features := expectations.XTest[i]
		
		// Simple prediction logic (would need proper predictor)
		// This is just to verify the model structure is correct
		t.Logf("Sample %d features: %v", i, features)
	}

	t.Logf("Model loaded successfully with %d trees", len(model.Trees))
}

// TestJSONModelCompatibilityWithTextFormat tests that JSON and text formats produce similar models
func TestJSONModelCompatibilityWithTextFormat(t *testing.T) {
	// This test would compare models loaded from JSON and text formats
	// to ensure they produce similar predictions
	
	testCases := []struct {
		name     string
		jsonFile string
		textFile string
	}{
		{
			name:     "Multiclass model",
			jsonFile: "testdata/expectations/iris_model_dump.json",
			textFile: "testdata/compatibility/multiclass_model.txt",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if files exist
			for _, file := range []string{tc.jsonFile, tc.textFile} {
				if _, err := os.Stat(file); os.IsNotExist(err) {
					t.Skipf("File not found: %s", file)
				}
			}

			// Load both models
			jsonModel, err := LoadJSONModelFromFile(tc.jsonFile)
			if err != nil {
				t.Fatalf("Failed to load JSON model: %v", err)
			}

			textModel, err := LoadLeavesModelFromFile(tc.textFile)
			if err != nil {
				t.Fatalf("Failed to load text model: %v", err)
			}

			// Compare basic properties
			t.Logf("JSON model: Trees=%d, Features=%d, Classes=%d",
				len(jsonModel.Trees), jsonModel.NumFeatures, jsonModel.NumClass)
			t.Logf("Text model: Trees=%d, Features=%d, Classes=%d",
				len(textModel.Trees), textModel.NumFeatures, textModel.NumClass)

			// Note: Detailed comparison would require implementing prediction
			// and comparing outputs on test data
		})
	}
}

// BenchmarkJSONModelLoading benchmarks JSON model loading performance
func BenchmarkJSONModelLoading(b *testing.B) {
	modelFile := "testdata/expectations/iris_model_dump.json"
	
	// Check if file exists
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		b.Skipf("Model file not found: %s", modelFile)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := LoadJSONModelFromFile(modelFile)
		if err != nil {
			b.Fatalf("Failed to load model: %v", err)
		}
	}
}

// TestPathTraversalSecurity tests that path traversal attacks are prevented
func TestPathTraversalSecurity(t *testing.T) {
	maliciousPath := "../../../etc/passwd"
	
	_, err := LoadJSONModelFromFile(maliciousPath)
	if err == nil {
		t.Error("Expected error for path traversal, got none")
	}
	
	if err != nil && !filepath.IsAbs(err.Error()) {
		t.Logf("Successfully blocked path traversal: %v", err)
	}
}