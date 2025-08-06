package model

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