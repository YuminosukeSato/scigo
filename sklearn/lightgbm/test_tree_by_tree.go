package lightgbm

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// VerificationData represents the Python verification data structure
type VerificationData struct {
	SampleFeatures        []float64 `json:"sample_features"`
	FinalPrediction       float64   `json:"final_prediction"`
	CumulativePredictions []float64 `json:"cumulative_predictions"`
	ModelInfo             struct {
		NumTrees      int    `json:"num_trees"`
		BestIteration int    `json:"best_iteration"`
		Objective     string `json:"objective"`
	} `json:"model_info"`
}

func TestTreeByTree(t *testing.T) {
	// Python検証データを読み込み
	verificationFile, err := os.ReadFile("testdata/python_verification.json")
	if err != nil {
		t.Fatalf("Failed to load verification data: %v", err)
	}

	var verification VerificationData
	if err := json.Unmarshal(verificationFile, &verification); err != nil {
		t.Fatalf("Failed to parse verification data: %v", err)
	}

	// モデルを読み込み
	model, err := LoadFromFile("testdata/compatibility/regression_model.txt")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("=== Go LightGBM ツリー別予測検証 ===\n")
	fmt.Printf("Python final prediction: %f\n", verification.FinalPrediction)
	fmt.Printf("Model trees: %d\n", len(model.Trees))
	fmt.Printf("Sample features: %v\n", verification.SampleFeatures)
	fmt.Printf("\n")

	predictor := NewPredictor(model)

	// 最終予測値を確認
	X := mat.NewDense(1, len(verification.SampleFeatures), verification.SampleFeatures)
	predictions, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	finalPred := predictions.At(0, 0)
	fmt.Printf("Go final prediction: %f\n", finalPred)
	fmt.Printf("Difference: %f\n", finalPred-verification.FinalPrediction)
	fmt.Printf("\n")

	// 各ツリーの個別予測を確認
	fmt.Printf("=== 各ツリーの個別予測値 ===\n")
	cumulativePred := 0.0

	for i := 0; i < 10 && i < len(model.Trees); i++ {
		tree := &model.Trees[i]
		treePred := predictor.predictTree(tree, verification.SampleFeatures)
		cumulativePred += treePred

		fmt.Printf("Tree %2d: output=%12.6f, cumulative=%12.6f, shrinkage=%f\n",
			i, treePred, cumulativePred, tree.ShrinkageRate)

		// Python値と比較
		if i < len(verification.CumulativePredictions) {
			pythonCumulative := verification.CumulativePredictions[i]
			diff := cumulativePred - pythonCumulative
			fmt.Printf("         Python cumulative=%12.6f, diff=%12.6f\n",
				pythonCumulative, diff)
		}
		fmt.Printf("\n")
	}

	// 特定のツリー数での予測値を確認
	fmt.Printf("=== 特定ツリー数での予測値比較 ===\n")
	testCounts := []int{1, 2, 5, 10, 20, 50, 100}

	for _, count := range testCounts {
		if count > len(model.Trees) {
			continue
		}

		// Go側で手動累積
		manualCumulative := 0.0
		for i := 0; i < count; i++ {
			tree := &model.Trees[i]
			treePred := predictor.predictTree(tree, verification.SampleFeatures)
			manualCumulative += treePred
		}

		fmt.Printf("Trees 1-%3d: Go manual=%12.6f\n", count, manualCumulative)
	}

	// 問題のあるツリーを特定
	fmt.Printf("\n=== 問題ツリーの特定 ===\n")
	problemTrees := 0
	for i := 0; i < len(model.Trees); i++ {
		tree := &model.Trees[i]
		treePred := predictor.predictTree(tree, verification.SampleFeatures)

		if treePred == 0.0 {
			problemTrees++
			if problemTrees <= 5 { // 最初の5個だけ詳細表示
				fmt.Printf("Tree %d: ZERO output (shrinkage=%f, leaves=%d, nodes=%d)\n",
					i, tree.ShrinkageRate, tree.NumLeaves, len(tree.Nodes))

				// ツリー構造を簡単に確認
				if len(tree.Nodes) > 0 {
					fmt.Printf("  Root node: Feature=%d, Threshold=%f, Left=%d, Right=%d\n",
						tree.Nodes[0].SplitFeature, tree.Nodes[0].Threshold,
						tree.Nodes[0].LeftChild, tree.Nodes[0].RightChild)
				}
				fmt.Printf("  LeafValues count: %d\n", len(tree.LeafValues))
				if len(tree.LeafValues) > 0 {
					fmt.Printf("  First few leaf values: %v\n", tree.LeafValues[:min(3, len(tree.LeafValues))])
				}
				fmt.Printf("\n")
			}
		}
	}

	fmt.Printf("Total trees returning 0.0: %d out of %d\n", problemTrees, len(model.Trees))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
