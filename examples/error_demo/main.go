package main

import (
	"fmt"
	"log/slog"

	"github.com/YuminosukeSato/GoML/linear"
	"github.com/YuminosukeSato/GoML/pkg/errors"
	"github.com/YuminosukeSato/GoML/pkg/log"
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
		slog.Error("Failed to fit model", log.ErrAttr(err))
		return
	}
	
	// 予測データ: 間違った次元数（3次元）
	X_test := mat.NewDense(2, 3, []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	})
	
	_, err = model.Predict(X_test)
	if err != nil {
		// 基本的なエラーメッセージ
		fmt.Printf("Error: %v\n", err)
		
		// 詳細なエラー情報（スタックトレース付き）
		fmt.Printf("\nDetailed error:\n%+v\n", err)
		
		// ログにも出力
		slog.Error("Prediction failed due to dimension mismatch", 
			log.ErrAttr(err),
			slog.Int("expected_features", 2),
			slog.Int("got_features", 3),
		)
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
		fmt.Printf("Error: %v\n", err)
		
		// エラー型の判定
		var notFittedErr *errors.NotFittedError
		if errors.As(err, &notFittedErr) {
			fmt.Printf("This is a NotFittedError for model: %s\n", notFittedErr.ModelName)
		}
		
		slog.Warn("Model used before fitting", log.ErrAttr(err))
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
		fmt.Printf("Error: %v\n", err)
		
		// ModelErrorかチェック
		var modelErr *errors.ModelError
		if errors.As(err, &modelErr) {
			fmt.Printf("Operation: %s, Kind: %s\n", modelErr.Op, modelErr.Kind)
		}
		
		slog.Error("Invalid input data", 
			log.ErrAttr(err),
			slog.String("data_type", "empty"),
		)
	}
}

func demonstrateErrorChaining() {
	fmt.Println("4. Error Chaining Demo:")
	fmt.Println("-----------------------")
	
	// エラーチェーンのシミュレーション
	err := processData()
	if err != nil {
		fmt.Printf("Simple error: %v\n", err)
		fmt.Printf("\nError chain:\n%+v\n", err)
		
		// 元のエラーを確認
		if errors.Is(err, errors.ErrEmptyData) {
			fmt.Println("\nRoot cause: Empty data error detected")
		}
		
		slog.Error("Data processing failed", log.ErrAttr(err))
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