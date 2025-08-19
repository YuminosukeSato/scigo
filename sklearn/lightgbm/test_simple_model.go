package lightgbm

import (
	"fmt"
	"math"
	"testing"
)

// TestSimpleModelPrediction tests basic model prediction functionality
func TestSimpleModelPrediction(t *testing.T) {
	// Load the simple model
	model, err := LoadFromFile("testdata/simple_model.txt")
	if err != nil {
		t.Fatalf("Failed to load simple model: %v", err)
	}

	// Debug model structure
	fmt.Printf("Model loaded:\n")
	fmt.Printf("  NumFeatures: %d\n", model.NumFeatures)
	fmt.Printf("  NumTrees: %d\n", len(model.Trees))
	fmt.Printf("  InitScore: %f\n", model.InitScore)

	if len(model.Trees) > 0 {
		tree := model.Trees[0]
		fmt.Printf("\nTree 0:\n")
		fmt.Printf("  NumLeaves: %d\n", tree.NumLeaves)
		fmt.Printf("  NumNodes: %d\n", len(tree.Nodes))
		fmt.Printf("  ShrinkageRate: %f\n", tree.ShrinkageRate)
		fmt.Printf("  LeafValues: %v\n", tree.LeafValues)

		for i, node := range tree.Nodes {
			fmt.Printf("  Node %d: Type=%v, Feature=%d, Threshold=%f, LeafValue=%f, Left=%d, Right=%d\n",
				i, node.NodeType, node.SplitFeature, node.Threshold, node.LeafValue,
				node.LeftChild, node.RightChild)
		}
	}

	// Test predictions
	testCases := []struct {
		input    float64
		expected float64
	}{
		{0.25, 0.35},
		{0.75, 0.85},
	}

	predictor := NewPredictor(model)

	for _, tc := range testCases {
		features := []float64{tc.input}
		pred := predictor.predictSingleSample(features)

		fmt.Printf("\nPrediction for X=%f:\n", tc.input)
		fmt.Printf("  Expected: %f\n", tc.expected)
		fmt.Printf("  Got: %f\n", pred[0])
		fmt.Printf("  Difference: %f\n", math.Abs(pred[0]-tc.expected))

		// Debug tree traversal
		tree := &model.Trees[0]
		fmt.Printf("  Tree traversal:\n")

		nodeIdx := 0
		for step := 0; step < 5; step++ {
			if nodeIdx < 0 {
				// Handle negative index (leaf reference)
				leafIdx := -(nodeIdx + 1)
				fmt.Printf("    Step %d: Negative index %d -> leaf index %d\n", step, nodeIdx, leafIdx)
				if leafIdx >= 0 && leafIdx < len(tree.LeafValues) {
					fmt.Printf("    -> Leaf value from LeafValues[%d]: %f\n", leafIdx, tree.LeafValues[leafIdx])
				}
				break
			}

			if nodeIdx >= len(tree.Nodes) {
				fmt.Printf("    Step %d: Index %d out of bounds\n", step, nodeIdx)
				break
			}

			node := &tree.Nodes[nodeIdx]
			fmt.Printf("    Step %d: Node %d (Type=%v, Feature=%d, Threshold=%f, LeafValue=%f)\n",
				step, nodeIdx, node.NodeType, node.SplitFeature, node.Threshold, node.LeafValue)

			if node.IsLeaf() {
				fmt.Printf("    -> Leaf reached with value: %f\n", node.LeafValue)
				break
			}

			// Check split
			if node.SplitFeature < len(features) {
				fVal := features[node.SplitFeature]
				if fVal <= node.Threshold {
					fmt.Printf("    -> Going left (%.2f <= %.2f) to node %d\n",
						fVal, node.Threshold, node.LeftChild)
					nodeIdx = node.LeftChild
				} else {
					fmt.Printf("    -> Going right (%.2f > %.2f) to node %d\n",
						fVal, node.Threshold, node.RightChild)
					nodeIdx = node.RightChild
				}
			}
		}

		// Check tolerance
		if math.Abs(pred[0]-tc.expected) > 0.0001 {
			t.Errorf("Prediction mismatch for X=%f: expected %f, got %f",
				tc.input, tc.expected, pred[0])
		}
	}
}
