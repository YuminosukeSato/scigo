package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/s21066/goml/linear"
	"github.com/s21066/goml/metrics"
	"github.com/s21066/goml/pkg/log"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Initialize logger
	log.SetupLogger("info")
	
	fmt.Println("=== GoML Linear Regression Example ===")
	fmt.Println()

	// 1. 訓練データの準備
	// 住宅の広さ（㎡）と価格（万円）の仮想データ
	fmt.Println("1. Preparing training data...")

	// 特徴量: 住宅の広さ（㎡）
	X_train := mat.NewDense(10, 1, []float64{
		50.0,  // 50㎡
		60.0,  // 60㎡
		70.0,  // 70㎡
		80.0,  // 80㎡
		90.0,  // 90㎡
		100.0, // 100㎡
		110.0, // 110㎡
		120.0, // 120㎡
		130.0, // 130㎡
		140.0, // 140㎡
	})

	// ターゲット: 価格（万円）
	// 仮に 価格 = 50 * 広さ + 1000 + ノイズ とする
	y_train := mat.NewVecDense(10, []float64{
		3500.0, // 50㎡の物件
		4000.0, // 60㎡の物件
		4600.0, // 70㎡の物件
		5100.0, // 80㎡の物件
		5500.0, // 90㎡の物件
		6000.0, // 100㎡の物件
		6600.0, // 110㎡の物件
		7000.0, // 120㎡の物件
		7500.0, // 130㎡の物件
		8100.0, // 140㎡の物件
	})

	fmt.Println("Training data shape:")
	r, c := X_train.Dims()
	fmt.Printf("  X: (%d, %d)\n", r, c)
	fmt.Printf("  y: (%d,)\n", y_train.Len())
	fmt.Println()

	// 2. モデルの作成と学習
	fmt.Println("2. Creating and training the model...")
	model := linear.NewLinearRegression()

	err := model.Fit(X_train, y_train)
	if err != nil {
		slog.Error("Failed to fit model", log.ErrAttr(err))
		os.Exit(1)
	}

	// 学習されたパラメータを表示
	fmt.Println("Learned parameters:")
	fmt.Printf("  Coefficient (slope): %.2f 万円/㎡\n", model.Weights()[0])
	fmt.Printf("  Intercept (base price): %.2f 万円\n", model.Intercept())
	fmt.Println()

	// 3. 訓練データでの予測と評価
	fmt.Println("3. Evaluating on training data...")

	predictions, err := model.Predict(X_train)
	if err != nil {
		slog.Error("Failed to predict", log.ErrAttr(err))
		os.Exit(1)
	}

	// 予測結果をVecDenseに変換
	predVec := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		predVec.SetVec(i, predictions.At(i, 0))
	}

	// 評価指標の計算
	mse, _ := metrics.MSE(y_train, predVec)
	rmse, _ := metrics.RMSE(y_train, predVec)
	mae, _ := metrics.MAE(y_train, predVec)
	r2, _ := metrics.R2Score(y_train, predVec)

	fmt.Println("Evaluation metrics:")
	fmt.Printf("  MSE:  %.2f\n", mse)
	fmt.Printf("  RMSE: %.2f\n", rmse)
	fmt.Printf("  MAE:  %.2f\n", mae)
	fmt.Printf("  R²:   %.4f\n", r2)
	fmt.Println()

	// 4. 新しいデータでの予測
	fmt.Println("4. Making predictions on new data...")

	// テストデータ: 新しい物件の広さ
	X_test := mat.NewDense(5, 1, []float64{
		55.0,  // 55㎡
		75.0,  // 75㎡
		95.0,  // 95㎡
		115.0, // 115㎡
		135.0, // 135㎡
	})

	predictions_test, err := model.Predict(X_test)
	if err != nil {
		slog.Error("Failed to predict on test data", log.ErrAttr(err))
		os.Exit(1)
	}

	fmt.Println("Predictions for new properties:")
	fmt.Println("Size (㎡) | Predicted Price (万円)")
	fmt.Println("---------|-------------------")
	for i := 0; i < 5; i++ {
		size := X_test.At(i, 0)
		price := predictions_test.At(i, 0)
		fmt.Printf("%8.0f | %18.2f\n", size, price)
	}
	fmt.Println()

	// 5. モデルの解釈
	fmt.Println("5. Model interpretation:")
	slope := model.Weights()[0]
	intercept := model.Intercept()

	fmt.Printf("The model learned: Price = %.2f × Size + %.2f\n", slope, intercept)
	fmt.Printf("This means:\n")
	fmt.Printf("  - Base price (0㎡): %.2f 万円\n", intercept)
	fmt.Printf("  - Price increase per ㎡: %.2f 万円\n", slope)
	fmt.Printf("  - For a 100㎡ property: %.2f 万円\n", slope*100+intercept)
	fmt.Println()

	// 6. 実際の値vs予測値の比較（最初の5件）
	fmt.Println("6. Actual vs Predicted (first 5 samples):")
	fmt.Println("Size | Actual Price | Predicted Price | Error")
	fmt.Println("-----|--------------|-----------------|-------")
	for i := 0; i < 5; i++ {
		size := X_train.At(i, 0)
		actual := y_train.AtVec(i)
		predicted := predVec.AtVec(i)
		error := actual - predicted
		fmt.Printf("%4.0f | %12.2f | %15.2f | %6.2f\n",
			size, actual, predicted, error)
	}
}
