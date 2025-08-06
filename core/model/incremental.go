package model

import "gonum.org/v1/gonum/mat"

// IncrementalEstimator はオンライン学習（逐次学習）可能なモデルのインターフェース
// scikit-learnのpartial_fit APIと互換性を持つ
type IncrementalEstimator interface {
	Estimator

	// PartialFit はミニバッチでモデルを逐次的に学習させる
	// classes は分類問題の場合に全クラスラベルを指定（最初の呼び出し時のみ必須）
	// 回帰問題の場合は nil を渡す
	PartialFit(X, y mat.Matrix, classes []int) error

	// NIterations は実行された学習イテレーション数を返す
	NIterations() int

	// WarmStart が有効かどうかを返す
	// true の場合、Fit 呼び出し時に既存のパラメータから学習を継続
	IsWarmStart() bool

	// SetWarmStart はウォームスタートの有効/無効を設定
	SetWarmStart(warmStart bool)
}

// OnlineMetrics はオンライン学習中のメトリクスを追跡するインターフェース
type OnlineMetrics interface {
	// GetLoss は現在の損失値を返す
	GetLoss() float64

	// GetLossHistory は損失値の履歴を返す
	GetLossHistory() []float64

	// GetConverged は収束したかどうかを返す
	GetConverged() bool
}

// AdaptiveLearning は学習率を動的に調整できるモデルのインターフェース
type AdaptiveLearning interface {
	// GetLearningRate は現在の学習率を返す
	GetLearningRate() float64

	// SetLearningRate は学習率を設定する
	SetLearningRate(lr float64)

	// GetLearningRateSchedule は学習率スケジュールを返す
	// "constant", "optimal", "invscaling", "adaptive" など
	GetLearningRateSchedule() string

	// SetLearningRateSchedule は学習率スケジュールを設定する
	SetLearningRateSchedule(schedule string)
}