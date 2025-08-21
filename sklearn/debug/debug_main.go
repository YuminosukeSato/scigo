//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"

	lightgbm "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
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

func main() {
	// Python検証データを読み込み
	verificationFile, err := os.ReadFile("lightgbm/testdata/python_verification.json")
	if err != nil {
		panic(fmt.Sprintf("Failed to load verification data: %v", err))
	}
	
	var verification VerificationData
	if err := json.Unmarshal(verificationFile, &verification); err != nil {
		panic(fmt.Sprintf("Failed to parse verification data: %v", err))
	}
	
	// モデルを読み込み
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/compatibility/regression_model.txt")
	if err != nil {
		panic(fmt.Sprintf("Failed to load model: %v", err))
	}
	
	fmt.Printf("=== Go LightGBM ツリー別予測検証 ===\n")
	fmt.Printf("Python final prediction: %f\n", verification.FinalPrediction)
	fmt.Printf("Model trees: %d\n", len(model.Trees))
	fmt.Printf("Sample features: %v\n", verification.SampleFeatures)
	fmt.Printf("\n")
	
	predictor := lightgbm.NewPredictor(model)
	
	// 最終予測値を確認
	X := mat.NewDense(1, len(verification.SampleFeatures), verification.SampleFeatures)
	predictions, err := predictor.Predict(X)
	if err != nil {
		panic(fmt.Sprintf("Prediction failed: %v", err))
	}
	
	finalPred := predictions.At(0, 0)
	fmt.Printf("Go final prediction: %f\n", finalPred)
	fmt.Printf("Difference: %f\n", finalPred-verification.FinalPrediction)
	fmt.Printf("\n")
	
	// 各ツリーの個別予測を確認（直接メソッドアクセス）
	fmt.Printf("=== 各ツリーの個別予測値 ===\n")
	cumulativePred := 0.0
	
	for i := 0; i < 10 && i < len(model.Trees); i++ {
		tree := &model.Trees[i]
		
		// predictTree は非公開メソッドなので、単一サンプル予測で代替
		singleSamplePred := predictSingleTreeWrapper(predictor, tree, verification.SampleFeatures)
		cumulativePred += singleSamplePred
		
		fmt.Printf("Tree %2d: output=%12.6f, cumulative=%12.6f, shrinkage=%f\n", 
			i, singleSamplePred, cumulativePred, tree.ShrinkageRate)
		
		// Python値と比較
		if i < len(verification.CumulativePredictions) {
			pythonCumulative := verification.CumulativePredictions[i]
			diff := cumulativePred - pythonCumulative
			fmt.Printf("         Python cumulative=%12.6f, diff=%12.6f\n", 
				pythonCumulative, diff)
		}
		fmt.Printf("\n")
	}
}

// Workaround for accessing non-exported predictTree method
func predictSingleTreeWrapper(predictor *lightgbm.Predictor, tree *lightgbm.Tree, features []float64) float64 {
	// Create a temporary model with just this tree
	tempModel := &lightgbm.Model{
		Trees:        []lightgbm.Tree{*tree},
		NumIteration: 1,
		NumFeatures:  len(features),
		NumClass:     1,
		Objective:    lightgbm.RegressionL2,
	}
	
	tempPredictor := lightgbm.NewPredictor(tempModel)
	X := mat.NewDense(1, len(features), features)
	predictions, err := tempPredictor.Predict(X)
	if err != nil {
		return 0.0
	}
	
	return predictions.At(0, 0)
}