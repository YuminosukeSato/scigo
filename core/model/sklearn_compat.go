package model

import (
	"gonum.org/v1/gonum/mat"
)

// SKLearnCompatible はscikit-learn互換のインターフェース
type SKLearnCompatible interface {
	// GetParams はモデルのハイパーパラメータを取得
	GetParams(deep bool) map[string]interface{}
	
	// SetParams はモデルのハイパーパラメータを設定
	SetParams(params map[string]interface{}) error
	
	// Clone はモデルの新しいインスタンスを同じパラメータで作成
	Clone() SKLearnCompatible
}

// ClassifierMixin は分類器のMixinインターフェース
type ClassifierMixin interface {
	Estimator
	
	// PredictProba は各クラスの確率を予測
	PredictProba(X mat.Matrix) (mat.Matrix, error)
	
	// PredictLogProba は各クラスの対数確率を予測
	PredictLogProba(X mat.Matrix) (mat.Matrix, error)
	
	// DecisionFunction は決定関数の値を計算
	DecisionFunction(X mat.Matrix) (mat.Matrix, error)
	
	// Classes は学習されたクラスラベルを返す
	Classes() []interface{}
	
	// NClasses は学習されたクラス数を返す
	NClasses() int
}

// RegressorMixin は回帰器のMixinインターフェース
type RegressorMixin interface {
	Estimator
	
	// Score は決定係数R²を計算
	Score(X, y mat.Matrix) (float64, error)
}

// TransformerMixin は変換器のMixinインターフェース
type TransformerMixin interface {
	// Transform はデータを変換
	Transform(X mat.Matrix) (mat.Matrix, error)
	
	// FitTransform は学習と変換を同時に実行
	FitTransform(X mat.Matrix) (mat.Matrix, error)
}

// InverseTransformerMixin は逆変換可能な変換器のインターフェース
type InverseTransformerMixin interface {
	TransformerMixin
	
	// InverseTransform は変換を逆方向に適用
	InverseTransform(X mat.Matrix) (mat.Matrix, error)
}

// ClusterMixin はクラスタリングのMixinインターフェース
type ClusterMixin interface {
	Fitter
	
	// FitPredict は学習と予測を同時に実行
	FitPredict(X mat.Matrix) ([]int, error)
	
	// PredictCluster は新しいデータのクラスタを予測
	PredictCluster(X mat.Matrix) ([]int, error)
	
	// NClusters はクラスタ数を返す
	NClusters() int
}

// ModelWeights は weights.go に移動済み

// WeightExporter は重みをエクスポート可能なモデルのインターフェース
type WeightExporter interface {
	// ExportWeights はモデルの重みをエクスポート
	ExportWeights() (*ModelWeights, error)
	
	// ImportWeights はモデルの重みをインポート
	ImportWeights(weights *ModelWeights) error
	
	// GetWeightHash は重みのハッシュ値を計算（検証用）
	GetWeightHash() string
}

// PartialFitMixin は逐次学習可能なモデルのインターフェース
type PartialFitMixin interface {
	// PartialFit はミニバッチで逐次学習
	PartialFit(X, y mat.Matrix, classes []int) error
	
	// NIterations は学習イテレーション数を返す
	NIterations() int
	
	// IsWarmStart はウォームスタートが有効かどうか
	IsWarmStart() bool
	
	// SetWarmStart はウォームスタートの有効/無効を設定
	SetWarmStart(warmStart bool)
}

// PipelineCompatible はパイプラインで使用可能なモデルのインターフェース
type PipelineCompatible interface {
	SKLearnCompatible
	
	// GetInputDim は入力次元数を返す
	GetInputDim() int
	
	// GetOutputDim は出力次元数を返す
	GetOutputDim() int
	
	// RequiresFit は学習が必要かどうか
	RequiresFit() bool
}

// ModelValidation はモデルの検証機能を提供
type ModelValidation struct {
	// ValidateInput は入力データの検証
	ValidateInput func(X mat.Matrix) error
	
	// ValidateOutput は出力データの検証
	ValidateOutput func(y mat.Matrix) error
	
	// ValidateWeights は重みの検証
	ValidateWeights func(weights *ModelWeights) error
}

// CrossValidatable はクロスバリデーション可能なモデルのインターフェース
type CrossValidatable interface {
	// GetCVSplits はクロスバリデーションの分割数を返す
	GetCVSplits() int
	
	// SetCVSplits はクロスバリデーションの分割数を設定
	SetCVSplits(n int)
	
	// GetCVScores はクロスバリデーションのスコアを返す
	GetCVScores() []float64
}

// ModelWeights関連のメソッドは weights.go に移動済み