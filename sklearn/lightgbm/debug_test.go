package lightgbm

import (
	"fmt"
	"testing"
)

func TestDebugPrediction(t *testing.T) {
	// Load model and make a simple prediction
	model, err := LoadFromFile("testdata/compatibility/regression_model.txt")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Create test features (single sample with 10 features)
	features := []float64{0.5, -0.5, 1.0, -1.0, 0.0, 0.2, -0.2, 0.8, -0.8, 0.3}

	predictor := NewPredictor(model)
	pred := predictor.predictSingleSample(features)

	fmt.Printf("Single sample prediction: %v\n", pred)
	fmt.Printf("InitScore: %f\n", model.InitScore)

	// Check prediction process step by step
	sumPred := model.InitScore
	for i := 0; i < 3 && i < len(model.Trees); i++ {
		tree := &model.Trees[i]
		fmt.Printf("\nTree %d (shrinkage=%f):\n", i, tree.ShrinkageRate)

		// Navigate through tree manually
		nodeIdx := 0
		for j := 0; j < 10 && nodeIdx >= 0 && nodeIdx < len(tree.Nodes); j++ {
			node := &tree.Nodes[nodeIdx]
			fmt.Printf("  Node %d: Type=%v, Feature=%d, Threshold=%f, LeafValue=%f, Left=%d, Right=%d\n",
				nodeIdx, node.NodeType, node.SplitFeature, node.Threshold, node.LeafValue,
				node.LeftChild, node.RightChild)

			if node.IsLeaf() {
				fmt.Printf("    -> Leaf reached with value: %f\n", node.LeafValue)
				break
			}

			// Navigate based on feature value
			if node.SplitFeature < len(features) {
				fVal := features[node.SplitFeature]
				if fVal <= node.Threshold {
					fmt.Printf("    -> Going left (feature[%d]=%f <= %f)\n",
						node.SplitFeature, fVal, node.Threshold)
					nodeIdx = node.LeftChild
				} else {
					fmt.Printf("    -> Going right (feature[%d]=%f > %f)\n",
						node.SplitFeature, fVal, node.Threshold)
					nodeIdx = node.RightChild
				}
			}
		}

		treeOut := predictor.predictTree(tree, features)
		sumPred += treeOut
		fmt.Printf("  Tree output: %f, cumulative: %f\n", treeOut, sumPred)
	}
}

func TestDebugModelLoading(t *testing.T) {
	// Load a simple regression model
	model, err := LoadFromFile("testdata/compatibility/regression_model.txt")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Print model information
	fmt.Printf("Model loaded:\n")
	fmt.Printf("  NumFeatures: %d\n", model.NumFeatures)
	fmt.Printf("  NumClass: %d\n", model.NumClass)
	fmt.Printf("  NumIteration: %d\n", model.NumIteration)
	fmt.Printf("  BestIteration: %d\n", model.BestIteration)
	fmt.Printf("  InitScore: %f\n", model.InitScore)
	fmt.Printf("  Number of trees: %d\n", len(model.Trees))
	fmt.Printf("  Objective: %s\n", model.Objective)

	// Check first few trees
	for tIdx := 0; tIdx < 3 && tIdx < len(model.Trees); tIdx++ {
		tree := model.Trees[tIdx]
		fmt.Printf("\nTree %d:\n", tIdx)
		fmt.Printf("  TreeIndex: %d\n", tree.TreeIndex)
		fmt.Printf("  NumLeaves: %d\n", tree.NumLeaves)
		fmt.Printf("  NumNodes: %d\n", len(tree.Nodes))
		fmt.Printf("  ShrinkageRate: %f\n", tree.ShrinkageRate)

		// Check if tree has any nodes
		if len(tree.Nodes) == 0 && len(tree.LeafValues) > 0 {
			fmt.Printf("  Single leaf tree with value: %f\n", tree.LeafValues[0])
		}

		// Check first few nodes
		for i := 0; i < 3 && i < len(tree.Nodes); i++ {
			node := tree.Nodes[i]
			fmt.Printf("  Node %d: Type=%v, Feature=%d, Threshold=%f, LeafValue=%f\n",
				i, node.NodeType, node.SplitFeature, node.Threshold, node.LeafValue)
		}
	}
}
