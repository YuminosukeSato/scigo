package linear_test

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/linear"
	"gonum.org/v1/gonum/mat"
)

func TestLinearRegression_LoadFromSKLearn(t *testing.T) {
	// テスト用のscikit-learnモデルJSONを作成
	skModel := model.SKLearnModel{
		ModelSpec: model.SKLearnModelSpec{
			Name:          "LinearRegression",
			FormatVersion: "1.0",
			SKLearnVersion: "1.3.0",
		},
	}

	params := model.SKLearnLinearRegressionParams{
		Coefficients: []float64{2.0, 3.0, -1.0},
		Intercept:    5.0,
		NFeatures:    3,
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("Failed to marshal params: %v", err)
	}
	skModel.Params = paramsJSON

	// JSONをバッファに書き込み
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	if err := encoder.Encode(&skModel); err != nil {
		t.Fatalf("Failed to encode model: %v", err)
	}

	// LinearRegressionモデルにロード
	lr := linear.NewLinearRegression()
	if err := lr.LoadFromSKLearnReader(&buf); err != nil {
		t.Fatalf("Failed to load from sklearn: %v", err)
	}

	// パラメータが正しく設定されているか確認
	if lr.NFeatures != 3 {
		t.Errorf("Expected NFeatures=3, got %d", lr.NFeatures)
	}

	if lr.Intercept != 5.0 {
		t.Errorf("Expected Intercept=5.0, got %f", lr.Intercept)
	}

	weights := lr.GetWeights()
	expectedWeights := []float64{2.0, 3.0, -1.0}
	if len(weights) != len(expectedWeights) {
		t.Fatalf("Expected %d weights, got %d", len(expectedWeights), len(weights))
	}

	for i, w := range weights {
		if w != expectedWeights[i] {
			t.Errorf("Weight[%d]: expected %f, got %f", i, expectedWeights[i], w)
		}
	}

	// モデルが学習済み状態になっているか確認
	if !lr.IsFitted() {
		t.Error("Model should be fitted after loading from sklearn")
	}
}

func TestLinearRegression_LoadFromSKLearnFile(t *testing.T) {
	// ファイルから読み込みテスト
	lr := linear.NewLinearRegression()
	err := lr.LoadFromSKLearn("../testdata/sklearn_linear_regression.json")
	if err != nil {
		t.Fatalf("Failed to load from file: %v", err)
	}

	// 基本的な検証
	if lr.NFeatures != 3 {
		t.Errorf("Expected NFeatures=3, got %d", lr.NFeatures)
	}

	if !lr.IsFitted() {
		t.Error("Model should be fitted")
	}
}

func TestLinearRegression_PredictAfterSKLearnLoad(t *testing.T) {
	// scikit-learnモデルをロード
	lr := linear.NewLinearRegression()
	err := lr.LoadFromSKLearn("../testdata/sklearn_linear_regression.json")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// テストデータで予測
	// y = 2*x1 + 3*x2 - 1*x3 + 5
	X := mat.NewDense(3, 3, []float64{
		1.0, 2.0, 3.0,  // 2*1 + 3*2 - 1*3 + 5 = 10
		0.0, 1.0, 2.0,  // 2*0 + 3*1 - 1*2 + 5 = 6
		2.0, 0.0, 1.0,  // 2*2 + 3*0 - 1*1 + 5 = 8
	})

	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	expectedPred := []float64{10.0, 6.0, 8.0}
	
	for i := 0; i < len(expectedPred); i++ {
		pred := predictions.At(i, 0)
		if math.Abs(pred-expectedPred[i]) > 1e-10 {
			t.Errorf("Prediction[%d]: expected %f, got %f", i, expectedPred[i], pred)
		}
	}
}

func TestLinearRegression_ExportToSKLearn(t *testing.T) {
	// まずGoでモデルを学習
	lr := linear.NewLinearRegression()
	
	// 学習データ
	X := mat.NewDense(4, 2, []float64{
		1.0, 2.0,
		2.0, 1.0,
		3.0, 4.0,
		4.0, 3.0,
	})
	y := mat.NewVecDense(4, []float64{5.0, 4.0, 11.0, 10.0})
	
	// モデルの学習
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// エクスポート
	var buf bytes.Buffer
	err = lr.ExportToSKLearnWriter(&buf)
	if err != nil {
		t.Fatalf("Failed to export to sklearn: %v", err)
	}

	// エクスポートされたJSONを検証
	var exported model.SKLearnModel
	decoder := json.NewDecoder(&buf)
	if err := decoder.Decode(&exported); err != nil {
		t.Fatalf("Failed to decode exported model: %v", err)
	}

	// メタデータの検証
	if exported.ModelSpec.Name != "LinearRegression" {
		t.Errorf("Expected model name 'LinearRegression', got %s", exported.ModelSpec.Name)
	}

	if exported.ModelSpec.FormatVersion != "1.0" {
		t.Errorf("Expected format version '1.0', got %s", exported.ModelSpec.FormatVersion)
	}

	// パラメータの検証
	var params model.SKLearnLinearRegressionParams
	if err := json.Unmarshal(exported.Params, &params); err != nil {
		t.Fatalf("Failed to unmarshal params: %v", err)
	}

	if params.NFeatures != 2 {
		t.Errorf("Expected NFeatures=2, got %d", params.NFeatures)
	}

	if len(params.Coefficients) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(params.Coefficients))
	}
}

func TestLinearRegression_RoundTrip(t *testing.T) {
	// モデルを学習 → エクスポート → 新しいモデルにインポート → 同じ予測結果
	lr1 := linear.NewLinearRegression()
	
	// 学習データ（線形独立、より多くのサンプル）
	X := mat.NewDense(10, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		1.0, 1.0, 0.0,
		1.0, 0.0, 1.0,
		0.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		2.0, 1.0, 0.0,
		1.0, 2.0, 0.0,
		1.0, 1.0, 2.0,
	})
	y := mat.NewVecDense(10, []float64{3.0, 4.0, 2.0, 7.0, 5.0, 6.0, 9.0, 7.0, 8.0, 10.0})
	
	// モデル1を学習
	if err := lr1.Fit(X, y); err != nil {
		t.Fatalf("Failed to fit model 1: %v", err)
	}

	// エクスポート
	tmpFile := "test_roundtrip.json"
	defer os.Remove(tmpFile)
	
	if err := lr1.ExportToSKLearn(tmpFile); err != nil {
		t.Fatalf("Failed to export model: %v", err)
	}

	// 新しいモデルにインポート
	lr2 := linear.NewLinearRegression()
	if err := lr2.LoadFromSKLearn(tmpFile); err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// テストデータで両モデルの予測を比較
	XTest := mat.NewDense(2, 3, []float64{
		2.0, 2.0, 1.0,
		3.0, 1.0, 1.0,
	})

	pred1, err := lr1.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict with model 1: %v", err)
	}

	pred2, err := lr2.Predict(XTest)
	if err != nil {
		t.Fatalf("Failed to predict with model 2: %v", err)
	}

	// 予測結果が一致することを確認（浮動小数点誤差を考慮）
	r, _ := pred1.Dims()
	for i := 0; i < r; i++ {
		p1 := pred1.At(i, 0)
		p2 := pred2.At(i, 0)
		if math.Abs(p1-p2) > 1e-10 {
			t.Errorf("Predictions differ at index %d: %f vs %f", i, p1, p2)
		}
	}
}

func TestLinearRegression_InvalidSKLearnData(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr bool
	}{
		{
			name: "wrong model name",
			json: `{
				"model_spec": {"name": "LogisticRegression", "format_version": "1.0"},
				"params": {"coefficients": [1.0], "intercept": 0.0, "n_features": 1}
			}`,
			wantErr: true,
		},
		{
			name: "missing coefficients",
			json: `{
				"model_spec": {"name": "LinearRegression", "format_version": "1.0"},
				"params": {"intercept": 0.0, "n_features": 1}
			}`,
			wantErr: true,
		},
		{
			name: "mismatched n_features",
			json: `{
				"model_spec": {"name": "LinearRegression", "format_version": "1.0"},
				"params": {"coefficients": [1.0, 2.0], "intercept": 0.0, "n_features": 3}
			}`,
			wantErr: true,
		},
		{
			name: "unsupported format version",
			json: `{
				"model_spec": {"name": "LinearRegression", "format_version": "2.0"},
				"params": {"coefficients": [1.0], "intercept": 0.0, "n_features": 1}
			}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lr := linear.NewLinearRegression()
			err := lr.LoadFromSKLearnReader(bytes.NewBufferString(tt.json))
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadFromSKLearnReader() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}