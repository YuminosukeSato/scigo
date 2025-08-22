package main

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
)

func main() {
	// Load the simple model
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/simple_model.txt")
	if err != nil {
		panic(err)
	}

	// Debug model structure
	fmt.Fprintf(os.Stdout, "Model loaded:\n")
	fmt.Fprintf(os.Stdout, "  NumFeatures: %d\n", model.NumFeatures)
	fmt.Fprintf(os.Stdout, "  NumTrees: %d\n", len(model.Trees))
	fmt.Fprintf(os.Stdout, "  InitScore: %f\n", model.InitScore)

	if len(model.Trees) > 0 {
		tree := model.Trees[0]
		fmt.Fprintf(os.Stdout, "\nTree 0:\n")
		fmt.Fprintf(os.Stdout, "  NumLeaves: %d\n", tree.NumLeaves)
		fmt.Fprintf(os.Stdout, "  NumNodes: %d\n", len(tree.Nodes))
		fmt.Fprintf(os.Stdout, "  ShrinkageRate: %f\n", tree.ShrinkageRate)
		fmt.Fprintf(os.Stdout, "  LeafValues: %v\n", tree.LeafValues)

		for i, node := range tree.Nodes {
			fmt.Fprintf(os.Stdout, "  Node %d: Type=%v, Feature=%d, Threshold=%f, LeafValue=%f, Left=%d, Right=%d\n",
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

		fmt.Fprintf(os.Stdout, "\nManual prediction for X=%f:\n", tc.input)
		fmt.Fprintf(os.Stdout, "  Split at threshold %f\n", node.Threshold)

		if tc.input <= node.Threshold {
			fmt.Fprintf(os.Stdout, "  Going left (%.2f <= %.2f) to index %d\n",
				tc.input, node.Threshold, node.LeftChild)
			leafIdx := -(node.LeftChild + 1)
			fmt.Fprintf(os.Stdout, "  Leaf index: %d, value: %f\n", leafIdx, tree.LeafValues[leafIdx])
		} else {
			fmt.Fprintf(os.Stdout, "  Going right (%.2f > %.2f) to index %d\n",
				tc.input, node.Threshold, node.RightChild)
			leafIdx := -(node.RightChild + 1)
			fmt.Fprintf(os.Stdout, "  Leaf index: %d, value: %f\n", leafIdx, tree.LeafValues[leafIdx])
		}

		// Use predictor
		x := make([][]float64, 1)
		x[0] = features
		matX := mat.NewDense(1, 1, nil)
		matX.Set(0, 0, features[0])

		predResult, predErr := predictor.Predict(matX)
		if predErr != nil {
			panic(predErr)
		}
		fmt.Fprintf(os.Stdout, "  Predictor result: %f\n", predResult.At(0, 0))
		fmt.Fprintf(os.Stdout, "  Expected: %f\n", tc.expected)
	}
}
