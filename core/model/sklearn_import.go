package model

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/YuminosukeSato/scigo/pkg/errors"
)

// SKLearnModelSpec はscikit-learnモデルのメタデータ
type SKLearnModelSpec struct {
	Name          string `json:"name"`           // モデル名 (e.g., "LinearRegression")
	FormatVersion string `json:"format_version"` // フォーマットバージョン
	SKLearnVersion string `json:"sklearn_version,omitempty"` // scikit-learnのバージョン
}

// SKLearnLinearRegressionParams は線形回帰モデルのパラメータ
type SKLearnLinearRegressionParams struct {
	Coefficients []float64 `json:"coefficients"` // 係数（重み）
	Intercept    float64   `json:"intercept"`    // 切片
	NFeatures    int       `json:"n_features"`   // 特徴量の数
}

// SKLearnModel はscikit-learnからエクスポートされたモデル
type SKLearnModel struct {
	ModelSpec SKLearnModelSpec        `json:"model_spec"`
	Params    json.RawMessage         `json:"params"`
}

// LoadSKLearnModelFromFile はファイルからscikit-learnモデルを読み込む
//
// パラメータ:
//   - filename: JSONファイルのパス
//
// 戻り値:
//   - *SKLearnModel: 読み込まれたモデル
//   - error: 読み込みエラー
//
// 使用例:
//
//	model, err := model.LoadSKLearnModelFromFile("sklearn_model.json")
//	if err != nil {
//	    log.Fatal(err)
//	}
func LoadSKLearnModelFromFile(filename string) (*SKLearnModel, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return LoadSKLearnModelFromReader(file)
}

// LoadSKLearnModelFromReader はReaderからscikit-learnモデルを読み込む
//
// パラメータ:
//   - r: JSONデータを含むReader
//
// 戻り値:
//   - *SKLearnModel: 読み込まれたモデル
//   - error: 読み込みエラー
func LoadSKLearnModelFromReader(r io.Reader) (*SKLearnModel, error) {
	var model SKLearnModel
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	// バージョン検証
	if model.ModelSpec.FormatVersion == "" {
		return nil, errors.NewValueError("LoadSKLearnModel", "format_version is required")
	}

	// サポートされているバージョンか確認
	if model.ModelSpec.FormatVersion != "1.0" {
		return nil, errors.NewValueError("LoadSKLearnModel", 
			fmt.Sprintf("unsupported format version: %s", model.ModelSpec.FormatVersion))
	}

	// モデル名の検証
	if model.ModelSpec.Name == "" {
		return nil, errors.NewValueError("LoadSKLearnModel", "model name is required")
	}

	return &model, nil
}

// LoadLinearRegressionParams はLinearRegressionのパラメータを読み込む
//
// パラメータ:
//   - model: SKLearnModelインスタンス
//
// 戻り値:
//   - *SKLearnLinearRegressionParams: パラメータ
//   - error: パース失敗時のエラー
func LoadLinearRegressionParams(model *SKLearnModel) (*SKLearnLinearRegressionParams, error) {
	if model.ModelSpec.Name != "LinearRegression" {
		return nil, errors.NewValueError("LoadLinearRegressionParams",
			fmt.Sprintf("expected LinearRegression, got %s", model.ModelSpec.Name))
	}

	var params SKLearnLinearRegressionParams
	if err := json.Unmarshal(model.Params, &params); err != nil {
		return nil, fmt.Errorf("failed to unmarshal params: %w", err)
	}

	// パラメータの検証
	if len(params.Coefficients) == 0 {
		return nil, errors.NewValueError("LoadLinearRegressionParams", 
			"coefficients cannot be empty")
	}

	if params.NFeatures != len(params.Coefficients) {
		return nil, errors.NewValueError("LoadLinearRegressionParams",
			fmt.Sprintf("n_features (%d) does not match coefficients length (%d)",
				params.NFeatures, len(params.Coefficients)))
	}

	return &params, nil
}

// ExportSKLearnModel はモデルをscikit-learn互換のJSON形式でエクスポート
//
// パラメータ:
//   - modelName: モデル名
//   - params: モデルパラメータ
//   - w: 出力先Writer
//
// 戻り値:
//   - error: エクスポート失敗時のエラー
func ExportSKLearnModel(modelName string, params interface{}, w io.Writer) error {
	model := SKLearnModel{
		ModelSpec: SKLearnModelSpec{
			Name:          modelName,
			FormatVersion: "1.0",
		},
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return fmt.Errorf("failed to marshal params: %w", err)
	}
	model.Params = paramsJSON

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(&model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}