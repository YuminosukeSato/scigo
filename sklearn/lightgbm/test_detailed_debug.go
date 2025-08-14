package lightgbm

import (
	"fmt"
	"testing"
)

func TestDetailedDebug(t *testing.T) {
	// Load model
	model, err := LoadFromFile("testdata/compatibility/regression_model.txt")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Test features
	features := []float64{0.5, -0.5, 1.0, -1.0, 0.0, 0.2, -0.2, 0.8, -0.8, 0.3}

	// Check first tree structure
	tree := &model.Trees[0]
	fmt.Printf("Tree 0 details:\n")
	fmt.Printf("  NumLeaves: %d\n", tree.NumLeaves)
	fmt.Printf("  NumNodes: %d\n", len(tree.Nodes))
	fmt.Printf("  LeafValues count: %d\n", len(tree.LeafValues))
	fmt.Printf("  ShrinkageRate: %f\n", tree.ShrinkageRate)

	// Print leaf values
	fmt.Printf("  LeafValues: ")
	for i := 0; i < 5 && i < len(tree.LeafValues); i++ {
		fmt.Printf("[%d]=%f ", i, tree.LeafValues[i])
	}
	fmt.Printf("...\n")

	// Manually trace through tree
	nodeIdx := 0
	for step := 0; step < 10; step++ {
		if nodeIdx < 0 {
			leafIdx := -(nodeIdx + 1)
			fmt.Printf("Step %d: Reached leaf index %d (from nodeIdx %d)\n", step, leafIdx, nodeIdx)
			if leafIdx >= 0 && leafIdx < len(tree.LeafValues) {
				fmt.Printf("  Leaf value: %f\n", tree.LeafValues[leafIdx])
				fmt.Printf("  With shrinkage: %f\n", tree.LeafValues[leafIdx]*tree.ShrinkageRate)
			} else {
				fmt.Printf("  ERROR: Leaf index %d out of bounds (max %d)\n", leafIdx, len(tree.LeafValues)-1)
			}
			break
		}

		if nodeIdx >= len(tree.Nodes) {
			fmt.Printf("Step %d: Node index %d out of bounds\n", step, nodeIdx)
			break
		}

		node := &tree.Nodes[nodeIdx]
		fmt.Printf("Step %d: Node %d (Feature=%d, Threshold=%f, Left=%d, Right=%d)\n",
			step, nodeIdx, node.SplitFeature, node.Threshold, node.LeftChild, node.RightChild)

		// Make decision
		if node.SplitFeature < len(features) {
			fVal := features[node.SplitFeature]
			if fVal <= node.Threshold {
				fmt.Printf("  Decision: %f <= %f, going left to %d\n", fVal, node.Threshold, node.LeftChild)
				nodeIdx = node.LeftChild
			} else {
				fmt.Printf("  Decision: %f > %f, going right to %d\n", fVal, node.Threshold, node.RightChild)
				nodeIdx = node.RightChild
			}
		} else {
			fmt.Printf("  ERROR: Feature index %d out of bounds\n", node.SplitFeature)
			break
		}
	}

	// Now test predictor
	predictor := NewPredictor(model)
	pred := predictor.predictSingleSample(features)
	fmt.Printf("\nPredictor result: %f\n", pred[0])
	fmt.Printf("Expected (Python): 0.731009\n")
}
