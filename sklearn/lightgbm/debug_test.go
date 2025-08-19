package lightgbm

import (
	"fmt"
	"path/filepath"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDebugSinglePrediction(t *testing.T) {
	// Load model
	modelPath := filepath.Join("testdata/compatibility", "regression_model.txt")
	model, err := LoadFromFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Create test features (first row from regression_X_test.csv)
	features := []float64{
		1.5904035685736386, -0.3939866812431204, 0.04092474872892285,
		-0.9984408528767182, 1.9913704184169045, 0.43494103836789866,
		1.6232566905280539, -0.5691481986725339, -0.7971135662249808,
		0.3929135105482163,
	}

	X := mat.NewDense(1, 10, features)

	// Make prediction
	predictor := NewPredictor(model)
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	pred := predictions.At(0, 0)
	fmt.Printf("Prediction: %f\n", pred)
	fmt.Printf("Expected: approximately 77.486099\n")
	fmt.Printf("InitScore: %f\n", model.InitScore)
	fmt.Printf("Number of trees: %d\n", len(model.Trees))

	// Calculate sum manually
	sum := model.InitScore
	for i := 0; i < len(model.Trees); i++ {
		tree := &model.Trees[i]
		treeOutput := predictor.predictTree(tree, features)
		sum += treeOutput
		if i < 5 {
			fmt.Printf("Tree %d: output=%f, cumulative=%f\n", i, treeOutput, sum)
		}
	}
	fmt.Printf("Manual sum (100 trees): %f\n", sum)

	// Debug first tree
	if len(model.Trees) > 0 {
		tree := &model.Trees[0]
		fmt.Printf("\nFirst tree info:\n")
		fmt.Printf("  Nodes: %d\n", len(tree.Nodes))
		fmt.Printf("  LeafValues: %d\n", len(tree.LeafValues))
		fmt.Printf("  ShrinkageRate: %f\n", tree.ShrinkageRate)

		// Check first few nodes
		for i := 0; i < 5 && i < len(tree.Nodes); i++ {
			node := &tree.Nodes[i]
			fmt.Printf("  Node %d: Left=%d, Right=%d, Feature=%d, Threshold=%f\n",
				i, node.LeftChild, node.RightChild, node.SplitFeature, node.Threshold)
		}

		// Check leaf values
		fmt.Printf("\nLeaf values:\n")
		for i := 0; i < 5 && i < len(tree.LeafValues); i++ {
			fmt.Printf("  Leaf %d: %f\n", i, tree.LeafValues[i])
		}
	}
}
