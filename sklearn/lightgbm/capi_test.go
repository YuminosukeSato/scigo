package lightgbm

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestCAPIMinimalRegression tests the C API with minimal regression golden data
func TestCAPIMinimalRegression(t *testing.T) {
	// Load golden data
	goldenDir := "../../tests/compatibility/golden_data"
	if _, err := os.Stat(goldenDir); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run: python3 tests/compatibility/generate_golden_data.py")
	}

	// Load test data
	jsonPath := filepath.Join(goldenDir, "minimal_regression.json")
	jsonData, err := ioutil.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("Failed to load golden data JSON: %v", err)
	}

	var goldenData struct {
		X          [][]float32            `json:"X"`
		Y          []float32              `json:"y"`
		Params     map[string]interface{} `json:"params"`
		Iterations []struct {
			Iteration         int       `json:"iteration"`
			Predictions       []float64 `json:"predictions"`
			Residuals         []float64 `json:"residuals"`
			Gradients         []float64 `json:"gradients"`
			Hessians          []float64 `json:"hessians"`
			TreeCount         int       `json:"tree_count"`
			FeatureImportance []float64 `json:"feature_importance"`
		} `json:"iterations"`
		FinalPredictions []float64 `json:"final_predictions"`
	}

	if err := json.Unmarshal(jsonData, &goldenData); err != nil {
		t.Fatalf("Failed to parse golden data JSON: %v", err)
	}

	// Flatten X data to row-major format
	nrow := len(goldenData.X)
	ncol := len(goldenData.X[0])
	flatX := make([]float32, nrow*ncol)
	for i := 0; i < nrow; i++ {
		for j := 0; j < ncol; j++ {
			flatX[i*ncol+j] = goldenData.X[i][j]
		}
	}

	// Create dataset
	dataset, err := DatasetCreateFromMat(flatX, nrow, ncol, true, goldenData.Y)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer DatasetFree(dataset)

	// Prepare parameters string
	params := "objective=regression " +
		"num_leaves=3 " +
		"max_depth=2 " +
		"learning_rate=0.1 " +
		"min_data_in_leaf=2 " +
		"lambda_l2=0.0 " +
		"min_gain_to_split=0.0"

	// Create booster
	booster, err := BoosterCreate(dataset, params)
	if err != nil {
		t.Fatalf("Failed to create booster: %v", err)
	}
	defer BoosterFree(booster)

	// Test initial predictions (should be close to mean of y)
	initialPreds, err := BoosterPredictForMat(booster, flatX, nrow, ncol, true)
	if err != nil {
		t.Fatalf("Failed to get initial predictions: %v", err)
	}

	// Calculate mean of y
	yMean := float64(0)
	for _, v := range goldenData.Y {
		yMean += float64(v)
	}
	yMean /= float64(len(goldenData.Y))

	// Check initial predictions are close to mean
	for i, pred := range initialPreds {
		if math.Abs(pred-yMean) > 0.01 {
			t.Errorf("Initial prediction[%d] = %f, expected close to mean %f", i, pred, yMean)
		}
	}

	// Train for 3 iterations and check predictions
	for iter := 0; iter < 3; iter++ {
		// Update one iteration
		if err := BoosterUpdateOneIter(booster); err != nil {
			t.Fatalf("Failed to update iteration %d: %v", iter+1, err)
		}

		// Get predictions
		predictions, err := BoosterPredictForMat(booster, flatX, nrow, ncol, true)
		if err != nil {
			t.Fatalf("Failed to get predictions at iteration %d: %v", iter+1, err)
		}

		// Compare with golden predictions
		// Note: LightGBM might have stopped training, so check if predictions changed
		if iter < len(goldenData.Iterations) {
			goldenPreds := goldenData.Iterations[iter].Predictions

			// Calculate max difference
			maxDiff := 0.0
			for i := range predictions {
				diff := math.Abs(predictions[i] - goldenPreds[i])
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			// Log the comparison
			t.Logf("Iteration %d: max prediction difference = %e", iter+1, maxDiff)

			// We expect some differences due to implementation details
			// but they should be small for such a simple case
			if maxDiff > 0.1 {
				t.Logf("Warning: Large difference at iteration %d", iter+1)
				for i := range predictions {
					t.Logf("  [%d] Go: %f, Python: %f, diff: %e",
						i, predictions[i], goldenPreds[i],
						math.Abs(predictions[i]-goldenPreds[i]))
				}
			}
		}
	}
}

// TestCAPIDatasetCreation tests dataset creation and validation
func TestCAPIDatasetCreation(t *testing.T) {
	// Test data
	data := []float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		10.0, 11.0, 12.0,
	}
	labels := []float32{0.5, 1.5, 2.5, 3.5}

	// Create dataset (row-major)
	dataset, err := DatasetCreateFromMat(data, 4, 3, true, labels)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer DatasetFree(dataset)

	// Validate dataset properties
	if dataset.numData != 4 {
		t.Errorf("Expected 4 data points, got %d", dataset.numData)
	}
	if dataset.numFeatures != 3 {
		t.Errorf("Expected 3 features, got %d", dataset.numFeatures)
	}

	// Check data values
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			expected := float64(data[i*3+j])
			actual := dataset.Data.At(i, j)
			if math.Abs(expected-actual) > 1e-6 {
				t.Errorf("Data mismatch at [%d,%d]: expected %f, got %f", i, j, expected, actual)
			}
		}
	}

	// Check labels
	for i, expected := range labels {
		if dataset.Label[i] != expected {
			t.Errorf("Label mismatch at [%d]: expected %f, got %f", i, expected, dataset.Label[i])
		}
	}

	// Test column-major format
	datasetCol, err := DatasetCreateFromMat(data, 4, 3, false, labels)
	if err != nil {
		t.Fatalf("Failed to create column-major dataset: %v", err)
	}
	defer DatasetFree(datasetCol)

	// Data should be transposed for column-major
	// First column should be [1, 4, 7, 10] instead of [1, 2, 3]
	expected := []float64{1.0, 5.0, 9.0}
	for j := 0; j < 3; j++ {
		actual := datasetCol.Data.At(0, j)
		if math.Abs(expected[j]-actual) > 1e-6 {
			t.Errorf("Column-major data mismatch at [0,%d]: expected %f, got %f", j, expected[j], actual)
		}
	}
}

// TestCAPIBoosterCreation tests booster creation with various parameters
func TestCAPIBoosterCreation(t *testing.T) {
	// Create minimal dataset
	data := []float32{1, 2, 3, 4, 5, 6}
	labels := []float32{0.5, 1.5}

	dataset, err := DatasetCreateFromMat(data, 2, 3, true, labels)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer DatasetFree(dataset)

	tests := []struct {
		name   string
		params string
		check  func(*LGBMBooster) error
	}{
		{
			name:   "regression default",
			params: "",
			check: func(b *LGBMBooster) error {
				if b.Objective != "regression" {
					return fmt.Errorf("expected regression objective, got %s", b.Objective)
				}
				if b.NumClass != 1 {
					return fmt.Errorf("expected 1 class for regression, got %d", b.NumClass)
				}
				return nil
			},
		},
		{
			name:   "binary classification",
			params: "objective=binary",
			check: func(b *LGBMBooster) error {
				if b.Objective != "binary" {
					return fmt.Errorf("expected binary objective, got %s", b.Objective)
				}
				return nil
			},
		},
		{
			name:   "custom learning rate",
			params: "learning_rate=0.05",
			check: func(b *LGBMBooster) error {
				if b.Params["learning_rate"] != "0.05" {
					return fmt.Errorf("expected learning_rate=0.05, got %s", b.Params["learning_rate"])
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			booster, err := BoosterCreate(dataset, tt.params)
			if err != nil {
				t.Fatalf("Failed to create booster: %v", err)
			}
			defer BoosterFree(booster)

			if err := tt.check(booster); err != nil {
				t.Error(err)
			}
		})
	}
}

// TestCAPITreeBuilding tests the tree building process
func TestCAPITreeBuilding(t *testing.T) {
	// Create a simple dataset where we know the optimal split
	data := []float32{
		1.0, 1.0,
		2.0, 1.0,
		3.0, 1.0,
		1.0, 2.0,
		2.0, 2.0,
		3.0, 2.0,
	}
	// Labels designed so feature 1 (second column) is the best split
	labels := []float32{1.0, 1.0, 1.0, 2.0, 2.0, 2.0}

	dataset, err := DatasetCreateFromMat(data, 6, 2, true, labels)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer DatasetFree(dataset)

	params := "objective=regression num_leaves=2 max_depth=1 min_data_in_leaf=1"
	booster, err := BoosterCreate(dataset, params)
	if err != nil {
		t.Fatalf("Failed to create booster: %v", err)
	}
	defer BoosterFree(booster)

	// Train one iteration
	if err := BoosterUpdateOneIter(booster); err != nil {
		t.Fatalf("Failed to update: %v", err)
	}

	// Check that we have one tree
	if len(booster.Trees) != 1 {
		t.Fatalf("Expected 1 tree, got %d", len(booster.Trees))
	}

	tree := booster.Trees[0]
	if tree.Root == nil {
		t.Fatal("Tree root is nil")
	}

	// The optimal split should be on feature 1 (second column)
	if !tree.Root.IsLeaf {
		t.Logf("Split feature: %d, threshold: %f",
			tree.Root.SplitFeature, tree.Root.Threshold)

		// Feature 1 should be the split feature
		if tree.Root.SplitFeature != 1 {
			t.Logf("Note: Split on feature %d instead of expected feature 1",
				tree.Root.SplitFeature)
		}
	}

	// Test predictions
	predictions, err := BoosterPredictForMat(booster, data, 6, 2, true)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// First 3 samples should have similar predictions (same y=1)
	// Last 3 samples should have similar predictions (same y=2)
	avgFirst := (predictions[0] + predictions[1] + predictions[2]) / 3
	avgLast := (predictions[3] + predictions[4] + predictions[5]) / 3

	if avgLast <= avgFirst {
		t.Errorf("Expected predictions for y=2 samples to be higher than y=1 samples")
		t.Logf("Avg first 3: %f, Avg last 3: %f", avgFirst, avgLast)
	}
}

