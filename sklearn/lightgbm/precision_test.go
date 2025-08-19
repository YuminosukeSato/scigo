package lightgbm

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// PrecisionTestData represents the structure of precision test data
type PrecisionTestData struct {
	Description string              `json:"description"`
	TestCases   []PrecisionTestCase `json:"test_cases"`
}

// PrecisionTestCase represents a single test case
type PrecisionTestCase struct {
	Name      string          `json:"name"`
	ModelFile string          `json:"model_file"`
	Trace     PredictionTrace `json:"trace"`
}

// PredictionTrace represents the detailed trace of prediction process
type PredictionTrace struct {
	Input            []float64     `json:"input"`
	NumTrees         int           `json:"num_trees"`
	NumClasses       int           `json:"num_classes"`
	Objective        string        `json:"objective"`
	Trees            []TreeTrace   `json:"trees"`
	CumulativeScores []interface{} `json:"cumulative_scores"`
	FinalRawScore    interface{}   `json:"final_raw_score"`
	FinalPrediction  interface{}   `json:"final_prediction"`
	ExpectedPred     interface{}   `json:"expected_prediction"`
}

// TreeTrace represents the trace of a single tree
type TreeTrace struct {
	TreeIndex       int         `json:"tree_index"`
	LeafIndex       int         `json:"leaf_index"`
	LeafValue       float64     `json:"leaf_value"`
	Shrinkage       float64     `json:"shrinkage"`
	TreeOutput      float64     `json:"tree_output"`
	CumulativeScore interface{} `json:"cumulative_score"`
}

// TestPrecisionAgainstPython tests numerical precision against Python LightGBM
func TestPrecisionAgainstPython(t *testing.T) {
	// Skip if test data doesn't exist
	if _, err := os.Stat("precision_test_data.json"); os.IsNotExist(err) {
		t.Skip("precision_test_data.json not found. Run generate_precision_data.py first")
	}

	// Load test data
	data, err := os.ReadFile("precision_test_data.json")
	if err != nil {
		t.Fatalf("Failed to read test data: %v", err)
	}

	var testData PrecisionTestData
	if err := json.Unmarshal(data, &testData); err != nil {
		t.Fatalf("Failed to parse test data: %v", err)
	}

	// Run each test case
	for _, tc := range testData.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			testPrecisionCase(t, &tc)
		})
	}
}

func testPrecisionCase(t *testing.T, tc *PrecisionTestCase) {
	// Load model
	model, err := LoadModelAutoDetect(tc.ModelFile)
	if err != nil {
		t.Fatalf("Failed to load model %s: %v", tc.ModelFile, err)
	}

	// Create predictor with tracing
	predictor := &PrecisionTracker{
		Model:     model,
		TreeTrace: make([]TreeTrace, 0),
	}

	// Run prediction with tracing
	input := mat.NewDense(1, len(tc.Trace.Input), tc.Trace.Input)
	predictions := predictor.PredictWithTrace(input)

	// Compare tree-by-tree outputs
	if len(predictor.TreeTrace) != len(tc.Trace.Trees) {
		t.Errorf("Tree count mismatch: got %d, expected %d",
			len(predictor.TreeTrace), len(tc.Trace.Trees))
	}

	// Detailed comparison for each tree
	for i, treeTrace := range predictor.TreeTrace {
		if i >= len(tc.Trace.Trees) {
			break
		}
		expected := tc.Trace.Trees[i]

		// Compare leaf indices
		if treeTrace.LeafIndex != expected.LeafIndex {
			t.Errorf("Tree %d: Leaf index mismatch: got %d, expected %d",
				i, treeTrace.LeafIndex, expected.LeafIndex)
		}

		// Compare leaf values with tolerance
		if !almostEqual(treeTrace.LeafValue, expected.LeafValue, 1e-10) {
			t.Errorf("Tree %d: Leaf value mismatch: got %.15f, expected %.15f (diff: %.15e)",
				i, treeTrace.LeafValue, expected.LeafValue,
				math.Abs(treeTrace.LeafValue-expected.LeafValue))
		}

		// Compare tree outputs
		if !almostEqual(treeTrace.TreeOutput, expected.TreeOutput, 1e-10) {
			t.Errorf("Tree %d: Tree output mismatch: got %.15f, expected %.15f (diff: %.15e)",
				i, treeTrace.TreeOutput, expected.TreeOutput,
				math.Abs(treeTrace.TreeOutput-expected.TreeOutput))
		}

		// Compare cumulative scores
		if tc.Trace.NumClasses > 2 {
			// For multiclass, compare array of scores
			gotScores, ok1 := treeTrace.CumulativeScore.([]float64)
			expectedScores, ok2 := expected.CumulativeScore.([]interface{})

			if ok1 && ok2 {
				for classIdx := 0; classIdx < tc.Trace.NumClasses && classIdx < len(gotScores) && classIdx < len(expectedScores); classIdx++ {
					gotVal := gotScores[classIdx]
					expVal := extractFloat(expectedScores[classIdx])
					if !almostEqual(gotVal, expVal, 1e-10) {
						t.Errorf("Tree %d, Class %d: Cumulative score mismatch: got %.15f, expected %.15f (diff: %.15e)",
							i, classIdx, gotVal, expVal, math.Abs(gotVal-expVal))
					}
				}
			}
		} else {
			// For binary/regression, compare single value
			gotCumulative := extractFloat(treeTrace.CumulativeScore)
			expectedCumulative := extractFloat(expected.CumulativeScore)
			if !almostEqual(gotCumulative, expectedCumulative, 1e-10) {
				t.Errorf("Tree %d: Cumulative score mismatch: got %.15f, expected %.15f (diff: %.15e)",
					i, gotCumulative, expectedCumulative,
					math.Abs(gotCumulative-expectedCumulative))
			}
		}
	}

	// Compare final predictions
	var finalGot float64
	var finalExpected float64

	if tc.Trace.NumClasses > 2 {
		// For multiclass, compare first class probability
		finalGot = predictions.At(0, 0)
		// Extract first element from array
		if arr, ok := tc.Trace.FinalPrediction.([]interface{}); ok && len(arr) > 0 {
			finalExpected = extractFloat(arr[0])
		} else {
			finalExpected = extractFloat(tc.Trace.FinalPrediction)
		}
	} else {
		finalGot = predictions.At(0, 0)
		finalExpected = extractFloat(tc.Trace.FinalPrediction)
	}

	if !almostEqual(finalGot, finalExpected, 1e-9) {
		t.Errorf("Final prediction mismatch: got %.15f, expected %.15f (diff: %.15e)",
			finalGot, finalExpected, math.Abs(finalGot-finalExpected))
	}

	// Log precision metrics
	t.Logf("Precision test passed for %s:", tc.Name)
	t.Logf("  Trees processed: %d", len(predictor.TreeTrace))
	t.Logf("  Final prediction: %.15f", finalGot)
	t.Logf("  Expected: %.15f", finalExpected)
	t.Logf("  Absolute error: %.15e", math.Abs(finalGot-finalExpected))
}

// PrecisionTracker wraps a model to track prediction process
type PrecisionTracker struct {
	Model     *LeavesModel
	TreeTrace []TreeTrace
}

// PredictWithTrace performs prediction while recording intermediate values
func (pt *PrecisionTracker) PredictWithTrace(X mat.Matrix) mat.Matrix {
	rows, _ := X.Dims()

	// Determine output dimensions
	outputCols := 1
	if pt.Model.NumClass > 2 {
		outputCols = pt.Model.NumClass
	}

	predictions := mat.NewDense(rows, outputCols, nil)

	for i := 0; i < rows; i++ {
		features := mat.Row(nil, i, X)
		preds := pt.predictSingleWithTrace(features)

		if pt.Model.NumClass > 2 {
			for j := 0; j < outputCols; j++ {
				predictions.Set(i, j, preds[j])
			}
		} else {
			predictions.Set(i, 0, preds[0])
		}
	}

	return predictions
}

func (pt *PrecisionTracker) predictSingleWithTrace(features []float64) []float64 {
	pt.TreeTrace = make([]TreeTrace, 0)

	// Initialize cumulative scores
	var cumulativeScores []float64
	if pt.Model.NumClass > 2 {
		cumulativeScores = make([]float64, pt.Model.NumClass)
		for i := range cumulativeScores {
			cumulativeScores[i] = pt.Model.InitScore
		}
	} else {
		cumulativeScores = []float64{pt.Model.InitScore}
	}

	// Process each tree
	for treeIdx, tree := range pt.Model.Trees {
		// Get leaf index and value
		leafIdx, leafValue := pt.getLeafDetails(&tree, features)

		// Apply shrinkage
		treeOutput := leafValue * tree.ShrinkageRate

		// Update cumulative score for appropriate class
		var currentCumulative interface{}
		if pt.Model.NumClass > 2 {
			classIdx := treeIdx % pt.Model.NumClass
			cumulativeScores[classIdx] += treeOutput
			// Store copy of all scores for trace
			scoresCopy := make([]float64, len(cumulativeScores))
			copy(scoresCopy, cumulativeScores)
			currentCumulative = scoresCopy
		} else {
			cumulativeScores[0] += treeOutput
			currentCumulative = cumulativeScores[0]
		}

		// Record trace
		trace := TreeTrace{
			TreeIndex:       treeIdx,
			LeafIndex:       leafIdx,
			LeafValue:       leafValue,
			Shrinkage:       tree.ShrinkageRate,
			TreeOutput:      treeOutput,
			CumulativeScore: currentCumulative,
		}
		pt.TreeTrace = append(pt.TreeTrace, trace)
	}

	// Apply objective transformation
	switch pt.Model.Objective {
	case BinaryLogistic:
		return []float64{stableSigmoid(cumulativeScores[0])}
	case MulticlassSoftmax:
		return stableSoftmaxArray(cumulativeScores)
	default:
		return cumulativeScores
	}
}

func (pt *PrecisionTracker) getLeafDetails(tree *LeavesTree, features []float64) (int, float64) {
	// For simple trees without nodes
	if len(tree.Nodes) == 0 {
		if len(tree.LeafValues) > 0 {
			return 0, tree.LeafValues[0]
		}
		return 0, 0.0
	}

	// Traverse tree to find leaf
	nodeIdx := uint32(0)

	for nodeIdx < uint32(len(tree.Nodes)) { // nolint:gosec // Safe conversion: tree nodes count is always positive
		node := &tree.Nodes[nodeIdx]

		// Check if we've reached a leaf
		if node.Flags&leftLeaf != 0 || node.Flags&rightLeaf != 0 {
			// Determine which way to go
			featureVal := features[node.Feature]
			goLeft := featureVal <= node.Threshold

			// Handle categorical
			if node.Flags&categorical != 0 {
				goLeft = uint32(featureVal) == uint32(node.Threshold)
			}

			// Handle missing values
			if (node.Flags&missingNan != 0 && isNaN(featureVal)) ||
				(node.Flags&missingZero != 0 && featureVal == 0.0) {
				goLeft = node.Flags&defaultLeft != 0
			}

			if goLeft {
				if node.Flags&leftLeaf != 0 {
					// Left is a leaf
					return int(node.Left), tree.LeafValues[node.Left]
				}
				nodeIdx = node.Left
			} else {
				if node.Flags&rightLeaf != 0 {
					// Right is a leaf
					leafIdx := int(node.Right)
					if leafIdx < len(tree.LeafValues) {
						return leafIdx, tree.LeafValues[leafIdx]
					}
					// For implicit right child
					return 1, tree.LeafValues[1]
				}
				nodeIdx = nodeIdx + 1 // Right child is next in array
			}
		} else {
			// Both children are internal nodes (shouldn't happen in leaves format)
			break
		}
	}

	// Default to first leaf if something went wrong
	if len(tree.LeafValues) > 0 {
		return 0, tree.LeafValues[0]
	}
	return 0, 0.0
}

// stableSigmoid computes sigmoid with numerical stability
func stableSigmoid(x float64) float64 {
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}

	if x >= 0 {
		expNegX := math.Exp(-x)
		return 1.0 / (1.0 + expNegX)
	} else {
		expX := math.Exp(x)
		return expX / (1.0 + expX)
	}
}

// stableSoftmaxArray computes softmax with numerical stability
func stableSoftmaxArray(x []float64) []float64 {
	// Find max for numerical stability
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp(x - max) and sum
	expSum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		expSum += result[i]
	}

	// Normalize
	for i := range result {
		result[i] = result[i] / expSum
	}

	return result
}

// Helper functions

func almostEqual(a, b, tolerance float64) bool {
	// Handle special cases
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 1) && math.IsInf(b, 1) {
		return true
	}
	if math.IsInf(a, -1) && math.IsInf(b, -1) {
		return true
	}

	// Check absolute difference
	diff := math.Abs(a - b)
	if diff < tolerance {
		return true
	}

	// Check relative difference for large numbers
	maxAbs := math.Max(math.Abs(a), math.Abs(b))
	if maxAbs > 1.0 {
		relDiff := diff / maxAbs
		return relDiff < tolerance
	}

	return false
}

func extractFloat(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case float32:
		return float64(val)
	case int:
		return float64(val)
	case []interface{}:
		// For multiclass, return first element
		if len(val) > 0 {
			return extractFloat(val[0])
		}
	}
	return 0.0
}

// BenchmarkPrecisionTracking benchmarks the overhead of precision tracking
func BenchmarkPrecisionTracking(b *testing.B) {
	// Create a simple model
	model := &LeavesModel{
		NumFeatures: 5,
		NumClass:    1,
		Objective:   RegressionL2,
		Trees:       make([]LeavesTree, 100),
	}

	// Initialize trees
	for i := range model.Trees {
		model.Trees[i] = LeavesTree{
			TreeIndex:     i,
			ShrinkageRate: 0.1,
			LeafValues:    []float64{0.1, 0.2, 0.3},
			Nodes: []LeavesNode{
				{
					Feature:   uint32(i % 5), // nolint:gosec // Safe conversion: modulo result is always positive
					Threshold: 0.5,
					Flags:     leftLeaf | rightLeaf,
					Left:      0,
					Right:     1,
				},
			},
		}
	}

	tracker := &PrecisionTracker{Model: model}
	X := mat.NewDense(10, 5, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tracker.PredictWithTrace(X)
	}
}
