package main

import (
	"encoding/json"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"

	lightgbm "github.com/YuminosukeSato/scigo/sklearn/lightgbm"
)

// VerificationData represents the Python verification data structure.
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
	if unmarshalErr := json.Unmarshal(verificationFile, &verification); unmarshalErr != nil {
		panic(fmt.Sprintf("Failed to parse verification data: %v", unmarshalErr))
	}

	// モデルを読み込み
	model, err := lightgbm.LoadFromFile("lightgbm/testdata/compatibility/regression_model.txt")
	if err != nil {
		panic(fmt.Sprintf("Failed to load model: %v", err))
	}

	fmt.Fprintf(os.Stdout, "=== Go LightGBM ツリー別予測検証 ===\n")
	fmt.Fprintf(os.Stdout, "Python final prediction: %f\n", verification.FinalPrediction)
	fmt.Fprintf(os.Stdout, "Model trees: %d\n", len(model.Trees))
	fmt.Fprintf(os.Stdout, "Sample features: %v\n", verification.SampleFeatures)
	fmt.Fprintf(os.Stdout, "\n")

	predictor := lightgbm.NewPredictor(model)

	// 最終予測値を確認
	x := mat.NewDense(1, len(verification.SampleFeatures), verification.SampleFeatures)
	predictions, err := predictor.Predict(x)
	if err != nil {
		panic(fmt.Sprintf("Prediction failed: %v", err))
	}

	finalPred := predictions.At(0, 0)
	fmt.Fprintf(os.Stdout, "Go final prediction: %f\n", finalPred)
	fmt.Fprintf(os.Stdout, "Difference: %f\n", finalPred-verification.FinalPrediction)
	fmt.Fprintf(os.Stdout, "\n")

	// 各ツリーの個別予測を確認（直接メソッドアクセス）
	fmt.Fprintf(os.Stdout, "=== 各ツリーの個別予測値 ===\n")
	cumulativePred := 0.0

	for i := 0; i < 10 && i < len(model.Trees); i++ {
		tree := &model.Trees[i]

		// predictTree は非公開メソッドなので、単一サンプル予測で代替
		singleSamplePred := predictSingleTreeWrapper(predictor, tree, verification.SampleFeatures)
		cumulativePred += singleSamplePred

		fmt.Fprintf(os.Stdout, "Tree %2d: output=%12.6f, cumulative=%12.6f, shrinkage=%f\n",
			i, singleSamplePred, cumulativePred, tree.ShrinkageRate)

		// Python値と比較
		if i < len(verification.CumulativePredictions) {
			pythonCumulative := verification.CumulativePredictions[i]
			diff := cumulativePred - pythonCumulative
			fmt.Fprintf(os.Stdout, "         Python cumulative=%12.6f, diff=%12.6f\n",
				pythonCumulative, diff)
		}
		fmt.Fprintf(os.Stdout, "\n")
	}
}

// Workaround for accessing non-exported predictTree method.
func predictSingleTreeWrapper(_ *lightgbm.Predictor, tree *lightgbm.Tree, features []float64) float64 {
	// Create a temporary model with just this tree
	tempModel := &lightgbm.Model{
		Trees:        []lightgbm.Tree{*tree},
		NumIteration: 1,
		NumFeatures:  len(features),
		NumClass:     1,
		Objective:    lightgbm.RegressionL2,
	}

	tempPredictor := lightgbm.NewPredictor(tempModel)
	x := mat.NewDense(1, len(features), features)
	predictions, err := tempPredictor.Predict(x)
	if err != nil {
		return 0.0
	}

	return predictions.At(0, 0)
}
