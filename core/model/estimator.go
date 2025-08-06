package model

import "gonum.org/v1/gonum/mat"

// Fitter は学習可能なモデルのインターフェース
type Fitter interface {
	// Fit はモデルを訓練データで学習させる
	Fit(X, y mat.Matrix) error
}

// Predictor は予測可能なモデルのインターフェース
type Predictor interface {
	// Predict は入力データに対する予測を行う
	Predict(X mat.Matrix) (mat.Matrix, error)
}

// LinearModel は線形モデルのインターフェース
type LinearModel interface {
	// Weights は学習された重み（係数）を返す
	Weights() []float64
	// Intercept は学習された切片を返す
	Intercept() float64
	// Score はモデルの決定係数（R²）を計算する
	Score(X, y mat.Matrix) (float64, error)
}
