package model

import "gonum.org/v1/gonum/mat"

// Transformer はデータ変換のインターフェース
type Transformer interface {
	// Fit は変換に必要なパラメータを学習する
	Fit(X mat.Matrix) error

	// Transform はデータを変換する
	Transform(X mat.Matrix) (mat.Matrix, error)

	// FitTransform はFitとTransformを同時に実行する
	FitTransform(X mat.Matrix) (mat.Matrix, error)
}