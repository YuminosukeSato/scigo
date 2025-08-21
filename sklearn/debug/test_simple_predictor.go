//go:build ignore

package main

import (
	"fmt"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

func testSimplePredictor() {
	// Load the simple model
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/simple_model.txt")
	if err != nil {
		panic(err)
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

	predictor := lightgbm.NewPredictor(model)

	for _, tc := range testCases {
		features := []float64{tc.input}

		// Do manual prediction
		tree := &model.Trees[0]
		nodeIdx := 0
		node := &tree.Nodes[nodeIdx]

		fmt.Printf("\nManual prediction for X=%f:\n", tc.input)
		fmt.Printf("  Split at threshold %f\n", node.Threshold)

		if tc.input <= node.Threshold {
			fmt.Printf("  Going left (%.2f <= %.2f) to index %d\n", tc.input, node.Threshold, node.LeftChild)
			leafIdx := -(node.LeftChild + 1)
			fmt.Printf("  Leaf index: %d, value: %f\n", leafIdx, tree.LeafValues[leafIdx])
		} else {
			fmt.Printf("  Going right (%.2f > %.2f) to index %d\n", tc.input, node.Threshold, node.RightChild)
			leafIdx := -(node.RightChild + 1)
			fmt.Printf("  Leaf index: %d, value: %f\n", leafIdx, tree.LeafValues[leafIdx])
		}

		// Use predictor
		X := make([][]float64, 1)
		X[0] = features
		matX := mat.NewDense(1, 1, nil)
		matX.Set(0, 0, features[0])

		predResult, err := predictor.Predict(matX)
		if err != nil {
			panic(err)
		}
		fmt.Printf("  Predictor result: %f\n", predResult.At(0, 0))
		fmt.Printf("  Expected: %f\n", tc.expected)
	}
}
