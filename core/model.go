package core

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

// Transformer はデータ変換のインターフェース
type Transformer interface {
	// Fit は変換に必要なパラメータを学習する
	Fit(X mat.Matrix) error

	// Transform はデータを変換する
	Transform(X mat.Matrix) (mat.Matrix, error)

	// FitTransform はFitとTransformを同時に実行する
	FitTransform(X mat.Matrix) (mat.Matrix, error)
}

// Model は教師あり学習モデルの基本インターフェース
type Model interface {
	Fitter
	Predictor
}

// EstimatorState はモデルの学習状態を表す
type EstimatorState int

const (
	// NotFitted はモデルが未学習の状態
	NotFitted EstimatorState = iota
	// Fitted はモデルが学習済みの状態
	Fitted
)

// BaseEstimator は全てのモデルの基底となる構造体
type BaseEstimator struct {
	state EstimatorState
}

// IsFitted はモデルが学習済みかどうかを返す
func (e *BaseEstimator) IsFitted() bool {
	return e.state == Fitted
}

// SetFitted はモデルを学習済み状態に設定する
func (e *BaseEstimator) SetFitted() {
	e.state = Fitted
}

// Reset はモデルを初期状態にリセットする
func (e *BaseEstimator) Reset() {
	e.state = NotFitted
}
