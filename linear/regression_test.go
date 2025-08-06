package linear

import (
	"math"
	"testing"

	"github.com/YuminosukeSato/scigo/metrics"
	"gonum.org/v1/gonum/mat"
)

func TestLinearRegression_Fit(t *testing.T) {
	tests := []struct {
		name    string
		X       *mat.Dense
		y       *mat.VecDense
		wantErr bool
	}{
		{
			name: "simple linear relationship y = 2x + 1",
			X: mat.NewDense(5, 1, []float64{
				1.0,
				2.0,
				3.0,
				4.0,
				5.0,
			}),
			y: mat.NewVecDense(5, []float64{
				3.0,  // 2*1 + 1
				5.0,  // 2*2 + 1
				7.0,  // 2*3 + 1
				9.0,  // 2*4 + 1
				11.0, // 2*5 + 1
			}),
			wantErr: false,
		},
		{
			name: "multiple features",
			X: mat.NewDense(5, 2, []float64{
				1.0, 2.0,
				2.0, 1.0,
				3.0, 4.0,
				4.0, 3.0,
				5.0, 5.0,
			}),
			y: mat.NewVecDense(5, []float64{
				5.0,  // 1*1 + 2*2
				4.0,  // 1*2 + 2*1
				11.0, // 1*3 + 2*4
				10.0, // 1*4 + 2*3
				15.0, // 1*5 + 2*5
			}),
			wantErr: false,
		},
		{
			name:    "empty data",
			X:       &mat.Dense{},
			y:       &mat.VecDense{},
			wantErr: true,
		},
		{
			name: "mismatched dimensions",
			X: mat.NewDense(3, 2, []float64{
				1.0, 2.0,
				3.0, 4.0,
				5.0, 6.0,
			}),
			y:       mat.NewVecDense(2, []float64{1.0, 2.0}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lr := NewLinearRegression()
			err := lr.Fit(tt.X, tt.y)

			if (err != nil) != tt.wantErr {
				t.Errorf("LinearRegression.Fit() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && !lr.IsFitted() {
				t.Error("LinearRegression should be fitted after successful Fit()")
			}
		})
	}
}

func TestLinearRegression_Predict(t *testing.T) {
	// まず簡単なモデルを学習
	lr := NewLinearRegression()

	// y = 2x + 1 の関係を学習
	X_train := mat.NewDense(5, 1, []float64{
		1.0,
		2.0,
		3.0,
		4.0,
		5.0,
	})
	y_train := mat.NewVecDense(5, []float64{
		3.0,
		5.0,
		7.0,
		9.0,
		11.0,
	})

	err := lr.Fit(X_train, y_train)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	tests := []struct {
		name      string
		X         *mat.Dense
		wantShape []int
		wantY     []float64
		tolerance float64
		wantErr   bool
	}{
		{
			name: "predict on training data",
			X: mat.NewDense(2, 1, []float64{
				1.0,
				5.0,
			}),
			wantShape: []int{2, 1},
			wantY:     []float64{3.0, 11.0},
			tolerance: 1e-6,
			wantErr:   false,
		},
		{
			name: "predict on new data",
			X: mat.NewDense(3, 1, []float64{
				0.0,
				6.0,
				10.0,
			}),
			wantShape: []int{3, 1},
			wantY:     []float64{1.0, 13.0, 21.0},
			tolerance: 1e-6,
			wantErr:   false,
		},
		{
			name: "wrong number of features",
			X: mat.NewDense(2, 2, []float64{
				1.0, 2.0,
				3.0, 4.0,
			}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pred, err := lr.Predict(tt.X)

			if (err != nil) != tt.wantErr {
				t.Errorf("LinearRegression.Predict() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				r, c := pred.Dims()
				if r != tt.wantShape[0] || c != tt.wantShape[1] {
					t.Errorf("Prediction shape = [%d, %d], want %v", r, c, tt.wantShape)
				}

				// 予測値の検証
				for i := 0; i < r; i++ {
					got := pred.At(i, 0)
					want := tt.wantY[i]
					if math.Abs(got-want) > tt.tolerance {
						t.Errorf("Prediction[%d] = %v, want %v (tolerance: %v)",
							i, got, want, tt.tolerance)
					}
				}
			}
		})
	}
}

func TestLinearRegression_PredictNotFitted(t *testing.T) {
	lr := NewLinearRegression()

	X := mat.NewDense(2, 1, []float64{1.0, 2.0})
	_, err := lr.Predict(X)

	if err == nil {
		t.Error("Expected error when predicting with unfitted model")
	}
}

func TestLinearRegression_MultipleFeatures(t *testing.T) {
	// y = 1*x1 + 2*x2 + 3 の関係を学習
	lr := NewLinearRegression()

	X_train := mat.NewDense(6, 2, []float64{
		1.0, 1.0,
		2.0, 1.0,
		1.0, 2.0,
		3.0, 2.0,
		2.0, 3.0,
		4.0, 3.0,
	})

	y_train := mat.NewVecDense(6, []float64{
		6.0,  // 1*1 + 2*1 + 3
		7.0,  // 1*2 + 2*1 + 3
		8.0,  // 1*1 + 2*2 + 3
		10.0, // 1*3 + 2*2 + 3
		11.0, // 1*2 + 2*3 + 3
		13.0, // 1*4 + 2*3 + 3
	})

	err := lr.Fit(X_train, y_train)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// 新しいデータで予測
	X_test := mat.NewDense(2, 2, []float64{
		5.0, 1.0,
		1.0, 4.0,
	})

	pred, err := lr.Predict(X_test)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	expectedY := []float64{
		10.0, // 1*5 + 2*1 + 3
		12.0, // 1*1 + 2*4 + 3
	}

	tolerance := 1e-5
	for i := 0; i < 2; i++ {
		got := pred.At(i, 0)
		want := expectedY[i]
		if math.Abs(got-want) > tolerance {
			t.Errorf("Prediction[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestLinearRegression_MSEScore(t *testing.T) {
	// y = 2x + 1 の関係でモデルを学習
	lr := NewLinearRegression()

	X_train := mat.NewDense(10, 1, []float64{
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
	})

	y_train := mat.NewVecDense(10, []float64{
		3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0,
	})

	err := lr.Fit(X_train, y_train)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// 予測を実行
	pred, err := lr.Predict(X_train)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// 予測結果をVecDenseに変換
	r, _ := pred.Dims()
	predVec := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		predVec.SetVec(i, pred.At(i, 0))
	}

	// MSEを計算
	mse, err := metrics.MSE(y_train, predVec)
	if err != nil {
		t.Fatalf("Failed to calculate MSE: %v", err)
	}

	// 完璧な線形関係なのでMSEは非常に小さいはず
	if mse > 1e-10 {
		t.Errorf("MSE = %v, want < 1e-10", mse)
	}

	// R²スコアも計算
	r2score, err := lr.Score(X_train, y_train)
	if err != nil {
		t.Fatalf("Failed to calculate R² score: %v", err)
	}

	// 完璧な線形関係なのでR²は1.0に近いはず
	if math.Abs(r2score-1.0) > 1e-10 {
		t.Errorf("R² score = %v, want ≈ 1.0", r2score)
	}

	// metrics.R2Scoreとの整合性確認
	r2FromMetrics, err := metrics.R2Score(y_train, predVec)
	if err != nil {
		t.Fatalf("Failed to calculate R² score from metrics: %v", err)
	}

	if math.Abs(r2score-r2FromMetrics) > 1e-10 {
		t.Errorf("Score() = %v, metrics.R2Score() = %v, want equal", r2score, r2FromMetrics)
	}
}

func TestLinearRegression_RealWorldExample(t *testing.T) {
	// より現実的なデータ（ノイズあり）
	lr := NewLinearRegression()

	// y ≈ 3x + 5 + noise
	X := mat.NewDense(20, 1, []float64{
		1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
		6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
	})

	y := mat.NewVecDense(20, []float64{
		8.1, 9.5, 11.2, 12.3, 14.1, 15.8, 17.2, 18.4, 20.1, 21.5,
		23.2, 24.3, 26.1, 27.8, 29.2, 30.4, 32.1, 33.5, 35.2, 36.8,
	})

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// 予測を実行
	pred, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// 予測結果をVecDenseに変換
	r, _ := pred.Dims()
	predVec := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		predVec.SetVec(i, pred.At(i, 0))
	}

	// 各種メトリクスを計算
	mse, _ := metrics.MSE(y, predVec)
	rmse, _ := metrics.RMSE(y, predVec)
	mae, _ := metrics.MAE(y, predVec)
	r2, _ := metrics.R2Score(y, predVec)

	// 結果をログ出力（値の妥当性を確認）
	t.Logf("Real-world example metrics:")
	t.Logf("  MSE:  %.4f", mse)
	t.Logf("  RMSE: %.4f", rmse)
	t.Logf("  MAE:  %.4f", mae)
	t.Logf("  R²:   %.4f", r2)

	// モデルパラメータも確認
	t.Logf("Model parameters:")
	t.Logf("  Weights: %v", lr.Weights())
	t.Logf("  Intercept: %.4f", lr.Intercept())

	// R²は高い値（0.95以上）を期待
	if r2 < 0.95 {
		t.Errorf("R² score = %v, want > 0.95", r2)
	}
}

// ベンチマークテスト
func BenchmarkLinearRegression_Fit(b *testing.B) {
	// 1000サンプル、10特徴量のデータを生成
	nSamples := 1000
	nFeatures := 10

	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewVecDense(nSamples, nil)

	// ランダムなデータを生成（実際の実装では適切な初期化が必要）
	for i := 0; i < nSamples; i++ {
		sum := 0.0
		for j := 0; j < nFeatures; j++ {
			val := float64(i*nFeatures+j) / float64(nSamples*nFeatures)
			X.Set(i, j, val)
			sum += val * float64(j+1)
		}
		y.SetVec(i, sum)
	}

	lr := NewLinearRegression()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = lr.Fit(X, y)
	}
}
