package model

import (
	"encoding/json"
	"fmt"
)

// ModelWeights はモデルの重みを表す構造体（シリアライゼーション用）
type ModelWeights struct {
	// ModelType はモデルの種類（LinearRegression, SGDRegressor等）
	ModelType string `json:"model_type"`
	
	// Version はモデルのバージョン（互換性チェック用）
	Version string `json:"version"`
	
	// Coefficients は重み係数
	Coefficients []float64 `json:"coefficients"`
	
	// Intercept は切片
	Intercept float64 `json:"intercept"`
	
	// Features は特徴量の名前（オプション）
	Features []string `json:"features,omitempty"`
	
	// Hyperparameters はモデルのハイパーパラメータ
	Hyperparameters map[string]interface{} `json:"hyperparameters"`
	
	// Metadata は追加のメタデータ（学習時の統計等）
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	
	// IsFitted はモデルが学習済みかどうか
	IsFitted bool `json:"is_fitted"`
}

// ToJSON はModelWeightsをJSON形式にシリアライズ
func (mw *ModelWeights) ToJSON() ([]byte, error) {
	return json.MarshalIndent(mw, "", "  ")
}

// FromJSON はJSON形式からModelWeightsをデシリアライズ
func (mw *ModelWeights) FromJSON(data []byte) error {
	return json.Unmarshal(data, mw)
}

// Validate はModelWeightsの妥当性を検証
func (mw *ModelWeights) Validate() error {
	if mw.ModelType == "" {
		return fmt.Errorf("model_type is required")
	}
	
	if mw.Version == "" {
		return fmt.Errorf("version is required")
	}
	
	if !mw.IsFitted && len(mw.Coefficients) > 0 {
		return fmt.Errorf("unfitted model should not have coefficients")
	}
	
	if mw.IsFitted && len(mw.Coefficients) == 0 {
		return fmt.Errorf("fitted model must have coefficients")
	}
	
	return nil
}

// Clone はModelWeightsのディープコピーを作成
func (mw *ModelWeights) Clone() *ModelWeights {
	clone := &ModelWeights{
		ModelType:       mw.ModelType,
		Version:         mw.Version,
		Intercept:       mw.Intercept,
		IsFitted:        mw.IsFitted,
		Coefficients:    make([]float64, len(mw.Coefficients)),
		Features:        make([]string, len(mw.Features)),
		Hyperparameters: make(map[string]interface{}),
		Metadata:        make(map[string]interface{}),
	}
	
	copy(clone.Coefficients, mw.Coefficients)
	copy(clone.Features, mw.Features)
	
	for k, v := range mw.Hyperparameters {
		clone.Hyperparameters[k] = v
	}
	
	for k, v := range mw.Metadata {
		clone.Metadata[k] = v
	}
	
	return clone
}