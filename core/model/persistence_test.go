package model_test

import (
	"bytes"
	"os"
	"testing"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/linear"
	"gonum.org/v1/gonum/mat"
)

func TestSaveLoadModel(t *testing.T) {
	// テスト用の線形回帰モデルを作成
	reg := linear.NewLinearRegression()

	// 学習データの準備
	X := mat.NewDense(4, 1, []float64{1.0, 2.0, 3.0, 4.0})
	y := mat.NewVecDense(4, []float64{2.0, 4.0, 6.0, 8.0})

	// モデルの学習
	err := reg.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// 元のモデルで予測
	testX := mat.NewDense(1, 1, []float64{5.0})
	originalPred, err := reg.Predict(testX)
	if err != nil {
		t.Fatalf("Failed to predict with original model: %v", err)
	}

	// モデルを一時ファイルに保存
	tmpFile := "test_model.gob"
	defer func() { _ = os.Remove(tmpFile) }()

	err = model.SaveModel(reg, tmpFile)
	if err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// 新しいモデルインスタンスに読み込み
	loadedReg := linear.NewLinearRegression()
	err = model.LoadModel(loadedReg, tmpFile)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// 読み込んだモデルで予測
	loadedPred, err := loadedReg.Predict(testX)
	if err != nil {
		t.Fatalf("Failed to predict with loaded model: %v", err)
	}

	// 予測結果が同じであることを確認
	if originalPred == nil || loadedPred == nil {
		t.Fatal("Predictions should not be nil")
	}

	origPredVals := originalPred.(*mat.Dense).RawMatrix().Data
	loadedPredVals := loadedPred.(*mat.Dense).RawMatrix().Data

	if len(origPredVals) != len(loadedPredVals) || origPredVals[0] != loadedPredVals[0] {
		t.Errorf("Predictions do not match: original=%v, loaded=%v", origPredVals, loadedPredVals)
	}

	// モデルの状態が保持されていることを確認
	if !loadedReg.IsFitted() {
		t.Error("Loaded model should be fitted")
	}
}

func TestSaveLoadModelToWriter(t *testing.T) {
	// テスト用の線形回帰モデルを作成
	reg := linear.NewLinearRegression()

	// 学習データの準備（線形独立なデータ）
	X := mat.NewDense(4, 2, []float64{
		1.0, 2.0,
		2.0, 1.0,
		3.0, 4.0,
		4.0, 3.0,
	})
	y := mat.NewVecDense(4, []float64{5.0, 4.0, 11.0, 10.0})

	// モデルの学習
	err := reg.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// バッファに保存
	var buf bytes.Buffer
	err = model.SaveModelToWriter(reg, &buf)
	if err != nil {
		t.Fatalf("Failed to save model to writer: %v", err)
	}

	// バッファから読み込み
	loadedReg := linear.NewLinearRegression()
	err = model.LoadModelFromReader(loadedReg, &buf)
	if err != nil {
		t.Fatalf("Failed to load model from reader: %v", err)
	}

	// 同じ入力に対して同じ予測結果が得られることを確認
	testX := mat.NewDense(1, 2, []float64{5.0, 6.0})

	originalPred, err := reg.Predict(testX)
	if err != nil {
		t.Fatalf("Failed to predict with original model: %v", err)
	}

	loadedPred, err := loadedReg.Predict(testX)
	if err != nil {
		t.Fatalf("Failed to predict with loaded model: %v", err)
	}

	if originalPred == nil || loadedPred == nil {
		t.Fatal("Predictions should not be nil")
	}

	origPredVals := originalPred.(*mat.Dense).RawMatrix().Data
	loadedPredVals := loadedPred.(*mat.Dense).RawMatrix().Data

	if len(origPredVals) != len(loadedPredVals) || origPredVals[0] != loadedPredVals[0] {
		t.Errorf("Predictions do not match: original=%v, loaded=%v", origPredVals, loadedPredVals)
	}
}

func TestLoadModelFileNotFound(t *testing.T) {
	reg := linear.NewLinearRegression()
	err := model.LoadModel(reg, "nonexistent_file.gob")
	if err == nil {
		t.Error("Expected error for nonexistent file, got nil")
	}
	if err != nil && !bytes.Contains([]byte(err.Error()), []byte("failed to open file")) {
		t.Errorf("Expected error to contain 'failed to open file', got: %v", err)
	}
}

func TestSaveModelInvalidPath(t *testing.T) {
	reg := linear.NewLinearRegression()
	err := model.SaveModel(reg, "/invalid/path/model.gob")
	if err == nil {
		t.Error("Expected error for invalid path, got nil")
	}
	if err != nil && !bytes.Contains([]byte(err.Error()), []byte("failed to create file")) {
		t.Errorf("Expected error to contain 'failed to create file', got: %v", err)
	}
}
