package preprocessing

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/core/model"
	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
)

// OneHotEncoder はscikit-learn互換のOne-Hotエンコーダー
// カテゴリカルな文字列データを0/1のバイナリベクトルに変換する
type OneHotEncoder struct {
	model.BaseEstimator

	// Categories は各特徴量のカテゴリ一覧（ソート済み）
	Categories [][]string

	// CategoryToIdx は各特徴量のカテゴリ→インデックスマップ
	CategoryToIdx []map[string]int

	// NFeatures は入力特徴量数
	NFeatures int

	// NOutputs は出力特徴量数（全カテゴリの合計数）
	NOutputs int
}

// NewOneHotEncoder は新しいOneHotEncoderを作成する
//
// 戻り値:
//   - *OneHotEncoder: 新しいOneHotEncoderインスタンス
//
// 使用例:
//
//	encoder := preprocessing.NewOneHotEncoder()
//	err := encoder.Fit(data)
//	encoded, err := encoder.Transform(data)
func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{}
}

// Fit は訓練データからカテゴリ情報を学習する
//
// パラメータ:
//   - data: 訓練データ (n_samples × n_features の文字列スライス)
//
// 戻り値:
//   - error: エラーが発生した場合
func (e *OneHotEncoder) Fit(data [][]string) (err error) {
	defer scigoErrors.Recover(&err, "OneHotEncoder.Fit")
	if len(data) == 0 {
		return scigoErrors.NewModelError("OneHotEncoder.Fit", "empty data", scigoErrors.ErrEmptyData)
	}

	if len(data[0]) == 0 {
		return scigoErrors.NewModelError("OneHotEncoder.Fit", "empty features", scigoErrors.ErrEmptyData)
	}

	nSamples := len(data)
	nFeatures := len(data[0])

	// 特徴量数の一貫性チェック
	for i, row := range data {
		if len(row) != nFeatures {
			return scigoErrors.NewDimensionError("OneHotEncoder.Fit", nFeatures, len(row), i)
		}
	}

	e.NFeatures = nFeatures
	e.Categories = make([][]string, nFeatures)
	e.CategoryToIdx = make([]map[string]int, nFeatures)

	// 各特徴量のユニークなカテゴリを収集
	for j := 0; j < nFeatures; j++ {
		categorySet := make(map[string]bool)

		// サンプル全体からユニークなカテゴリを収集
		for i := 0; i < nSamples; i++ {
			categorySet[data[i][j]] = true
		}

		// カテゴリをスライスに変換してソート
		categories := make([]string, 0, len(categorySet))
		for category := range categorySet {
			categories = append(categories, category)
		}
		sort.Strings(categories)

		e.Categories[j] = categories

		// カテゴリ→インデックスマップを作成
		categoryToIdx := make(map[string]int)
		for idx, category := range categories {
			categoryToIdx[category] = idx
		}
		e.CategoryToIdx[j] = categoryToIdx
	}

	// 出力特徴量数を計算
	e.NOutputs = 0
	for _, categories := range e.Categories {
		e.NOutputs += len(categories)
	}

	e.SetFitted()
	return nil
}

// Transform は学習済みのカテゴリ情報を使ってデータをone-hot encodingする
//
// パラメータ:
//   - data: 変換するデータ
//
// 戻り値:
//   - mat.Matrix: one-hot encodingされたデータ
//   - error: エラーが発生した場合
func (e *OneHotEncoder) Transform(data [][]string) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "OneHotEncoder.Transform")
	if !e.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("OneHotEncoder", "Transform")
	}

	if len(data) == 0 {
		return mat.NewDense(0, e.NOutputs, nil), nil
	}

	nSamples := len(data)
	nFeatures := len(data[0])

	if nFeatures != e.NFeatures {
		return nil, scigoErrors.NewDimensionError("OneHotEncoder.Transform", e.NFeatures, nFeatures, 1)
	}

	// 結果行列を初期化（全て0）
	result := mat.NewDense(nSamples, e.NOutputs, nil)

	// 各サンプルを処理
	for i := 0; i < nSamples; i++ {
		outputIdx := 0

		// 各特徴量を処理
		for j := 0; j < nFeatures; j++ {
			category := data[i][j]

			// カテゴリが既知かチェック
			if idx, exists := e.CategoryToIdx[j][category]; exists {
				// 対応する出力位置に1を設定
				result.Set(i, outputIdx+idx, 1.0)
			}
			// 未知カテゴリの場合は0のまま（何もしない）

			// 次の特徴量の出力開始位置へ移動
			outputIdx += len(e.Categories[j])
		}
	}

	return result, nil
}

// FitTransform は訓練データで学習し、同じデータを変換する
//
// パラメータ:
//   - data: 訓練・変換するデータ
//
// 戻り値:
//   - mat.Matrix: one-hot encodingされたデータ
//   - error: エラーが発生した場合
func (e *OneHotEncoder) FitTransform(data [][]string) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "OneHotEncoder.FitTransform")
	if err := e.Fit(data); err != nil {
		return nil, err
	}
	return e.Transform(data)
}

// GetFeatureNamesOut は変換後の特徴量の名前を返す
//
// パラメータ:
//   - inputFeatures: 入力特徴量の名前（nilの場合は"x0", "x1", ...を使用）
//
// 戻り値:
//   - []string: 出力特徴量の名前のスライス
//
// 例:
//   - 入力特徴量名が["animal", "size"]の場合
//   - 出力: ["animal_cat", "animal_dog", "size_small", "size_large"]
func (e *OneHotEncoder) GetFeatureNamesOut(inputFeatures []string) []string {
	if !e.IsFitted() {
		return nil
	}

	var outputFeatures []string

	for i, categories := range e.Categories {
		// 入力特徴量名を決定
		var inputFeatureName string
		if inputFeatures != nil && i < len(inputFeatures) {
			inputFeatureName = inputFeatures[i]
		} else {
			inputFeatureName = fmt.Sprintf("x%d", i)
		}

		// 各カテゴリに対して出力特徴量名を生成
		for _, category := range categories {
			featureName := fmt.Sprintf("%s_%s", inputFeatureName, category)
			outputFeatures = append(outputFeatures, featureName)
		}
	}

	return outputFeatures
}
