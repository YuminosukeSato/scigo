package main

import (
	"fmt"
	"log"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// 二値分類モデルの読み込みテスト
	fmt.Println("LightGBM Python互換性のテスト中...")
	
	// 二値モデルを読み込み
	model, err := lightgbm.LoadFromFile("sklearn/lightgbm/testdata/binary_model.txt")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	
	// 予測器を作成
	predictor := lightgbm.NewPredictor(model)
	predictor.SetDeterministic(true)
	
	// テストデータ
	X := mat.NewDense(3, 4, []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		6.2, 3.4, 5.4, 2.3,
	})
	
	// 予測を実行
	predictions, err := predictor.Predict(X)
	if err != nil {
		log.Fatalf("Failed to predict: %v", err)
	}
	
	// Pythonからの期待値
	expected := []float64{
		0.18514822574281953,
		0.18514822574281953,
		0.815802544297422,
	}
	
	fmt.Println("結果:")
	fmt.Println("サンプル | 予測値 | 期待値 | 差分")
	fmt.Println("-------|-----------|----------|------------")
	for i := 0; i < 3; i++ {
		pred := predictions.At(i, 0)
		exp := expected[i]
		diff := pred - exp
		fmt.Printf("%d      | %.10f | %.10f | %.2e\n", i, pred, exp, diff)
	}
}