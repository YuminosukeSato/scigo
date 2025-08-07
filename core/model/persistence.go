package model

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
)

// SaveModel はモデルをファイルに保存する
//
// パラメータ:
//   - model: 保存するモデル（BaseEstimatorを埋め込んだ構造体）
//   - filename: 保存先のファイルパス
//
// 戻り値:
//   - error: 保存に失敗した場合のエラー
//
// 使用例:
//
//	var reg linear.Regression
//	// ... モデルの学習 ...
//	err := model.SaveModel(&reg, "model.gob")
func SaveModel(model interface{}, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}

// LoadModel はファイルからモデルを読み込む
//
// パラメータ:
//   - model: 読み込み先のモデル（BaseEstimatorを埋め込んだ構造体のポインタ）
//   - filename: 読み込み元のファイルパス
//
// 戻り値:
//   - error: 読み込みに失敗した場合のエラー
//
// 使用例:
//
//	var reg linear.Regression
//	err := model.LoadModel(&reg, "model.gob")
func LoadModel(model interface{}, filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	return nil
}

// SaveModelToWriter はモデルをio.Writerに保存する
//
// パラメータ:
//   - model: 保存するモデル
//   - w: 保存先のWriter
//
// 戻り値:
//   - error: 保存に失敗した場合のエラー
func SaveModelToWriter(model interface{}, w io.Writer) error {
	encoder := gob.NewEncoder(w)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}
	return nil
}

// LoadModelFromReader はio.Readerからモデルを読み込む
//
// パラメータ:
//   - model: 読み込み先のモデル（ポインタ）
//   - r: 読み込み元のReader
//
// 戻り値:
//   - error: 読み込みに失敗した場合のエラー
func LoadModelFromReader(model interface{}, r io.Reader) error {
	decoder := gob.NewDecoder(r)
	if err := decoder.Decode(model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}
	return nil
}
