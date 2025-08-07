package main

import (
	"fmt"

	"github.com/YuminosukeSato/scigo/linear"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"github.com/YuminosukeSato/scigo/pkg/log"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// ログのセットアップ
	log.SetupLogger("debug")

	fmt.Println("=== GoML Error Handling Demo ===")
	fmt.Println()

	// 1. DimensionErrorのデモ
	demonstrateDimensionError()
	fmt.Println()

	// 2. NotFittedErrorのデモ
	demonstrateNotFittedError()
	fmt.Println()

	// 3. ValueErrorのデモ
	demonstrateValueError()
	fmt.Println()

	// 4. エラーチェーンのデモ
	demonstrateErrorChaining()
}

func demonstrateDimensionError() {
	fmt.Println("1. Dimension Error Demo:")
	fmt.Println("-----------------------")

	// 線形回帰モデルの作成と学習
	model := linear.NewLinearRegression()

	// 訓練データ: 2次元特徴量
	X_train := mat.NewDense(5, 2, []float64{
		1.0, 2.0,
		2.0, 3.0,
		3.0, 4.0,
		4.0, 5.0,
		5.0, 6.0,
	})
	y_train := mat.NewVecDense(5, []float64{5.0, 8.0, 11.0, 14.0, 17.0})

	err := model.Fit(X_train, y_train)
	if err != nil {
		log.LogError(err, "Failed to fit model")
		return
	}

	// 予測データ: 間違った次元数（3次元）
	X_test := mat.NewDense(2, 3, []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	})

	_, err = model.Predict(X_test)
	if err != nil {
		// 構造化ログでエラーを出力
		logger := log.GetLogger()
		logger.Error().Err(err).
			Int("expected_features", 2).
			Int("got_features", 3).
			Msg("Prediction failed due to dimension mismatch")
	}
}

func demonstrateNotFittedError() {
	fmt.Println("2. Not Fitted Error Demo:")
	fmt.Println("-------------------------")

	// 未学習のモデルで予測を試みる
	model := linear.NewLinearRegression()
	X := mat.NewDense(3, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
	})

	_, err := model.Predict(X)
	if err != nil {
		// 構造化ログでエラーを出力

		// 構造化ログでエラーを出力 (scikit-learnスタイル)
		logger := log.GetLogger()
		logger.Warn().Err(err).
			Str("phase", "prediction").
			Str("issue", "model_not_fitted").
			Msg("Model used before fitting")
	}
}

func demonstrateValueError() {
	fmt.Println("3. Value Error Demo:")
	fmt.Println("--------------------")

	// 空のデータで学習を試みる
	model := linear.NewLinearRegression()
	X := &mat.Dense{}
	y := &mat.VecDense{}

	err := model.Fit(X, y)
	if err != nil {
		// 構造化ログでエラーを出力

		// scikit-learnスタイルの構造化エラーログ
		log.LogError(err, "Invalid input data detected")
	}
}

func demonstrateErrorChaining() {
	fmt.Println("4. Error Chaining Demo:")
	fmt.Println("-----------------------")

	// エラーチェーンのシミュレーション
	err := processData()
	if err != nil {
		// 構造化ログでエラーチェーンを出力

		// scikit-learnスタイルの詳細エラーログ
		logger := log.GetLogger()
		if errors.Is(err, errors.ErrEmptyData) {
			logger.Info().Str("root_cause", "empty_data").Msg("Root cause identified")
		}

		log.LogError(err, "Data processing failed")
	}
}

func processData() error {
	// 階層的なエラー処理のシミュレーション
	err := loadData()
	if err != nil {
		return errors.Wrap(err, "failed to process data")
	}
	return nil
}

func loadData() error {
	// データ読み込みエラーのシミュレーション
	err := readFromFile()
	if err != nil {
		return errors.Wrapf(err, "failed to load data from file %s", "data.csv")
	}
	return nil
}

func readFromFile() error {
	// 基底エラー
	return errors.ErrEmptyData
}
