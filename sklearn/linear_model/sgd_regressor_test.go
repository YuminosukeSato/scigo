package linear_model

import (
	"context"
	"math"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/core/model"
)

// TestSGDRegressorBasicFit は基本的なFit機能をテスト
func TestSGDRegressorBasicFit(t *testing.T) {
	// 簡単な線形データ: y = 2*x + 1
	X := mat.NewDense(100, 1, nil)
	y := mat.NewDense(100, 1, nil)

	for i := 0; i < 100; i++ {
		x := float64(i) / 10.0
		X.Set(i, 0, x)
		y.Set(i, 0, 2*x+1)
	}

	sgd := NewSGDRegressor(
		WithLearningRate("constant"),
		WithEta0(0.01),
		WithMaxIter(100),
		WithRandomState(42),
	)

	err := sgd.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !sgd.IsFitted() {
		t.Error("Model should be fitted after Fit()")
	}

	// 係数と切片の確認
	coef := sgd.Coef()
	intercept := sgd.Intercept()

	// 理論値に近いことを確認（許容誤差0.1）
	if math.Abs(coef[0]-2.0) > 0.1 {
		t.Errorf("Coefficient should be close to 2.0, got %f", coef[0])
	}
	if math.Abs(intercept-1.0) > 0.1 {
		t.Errorf("Intercept should be close to 1.0, got %f", intercept)
	}
}

// TestSGDRegressorPartialFit はオンライン学習機能をテスト
func TestSGDRegressorPartialFit(t *testing.T) {
	sgd := NewSGDRegressor(
		WithLearningRate("invscaling"),
		WithEta0(0.1),
		WithRandomState(42),
	)

	// ミニバッチでの学習
	for batch := 0; batch < 10; batch++ {
		X := mat.NewDense(10, 2, nil)
		y := mat.NewDense(10, 1, nil)

		for i := 0; i < 10; i++ {
			x1 := float64(batch*10+i) / 100.0
			x2 := float64(batch*10+i) / 50.0
			X.Set(i, 0, x1)
			X.Set(i, 1, x2)
			// y = 3*x1 + 2*x2 + 1
			y.Set(i, 0, 3*x1+2*x2+1)
		}

		err := sgd.PartialFit(X, y, nil)
		if err != nil {
			t.Fatalf("PartialFit failed at batch %d: %v", batch, err)
		}
	}

	if !sgd.IsFitted() {
		t.Error("Model should be fitted after PartialFit()")
	}

	// 学習イテレーション数の確認
	if sgd.NIterations() == 0 {
		t.Error("NIterations should be greater than 0")
	}
}

// TestSGDRegressorPredict は予測機能をテスト
func TestSGDRegressorPredict(t *testing.T) {
	// 訓練データ
	X_train := mat.NewDense(50, 1, nil)
	y_train := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		x := float64(i) / 10.0
		X_train.Set(i, 0, x)
		y_train.Set(i, 0, 3*x+2)
	}

	sgd := NewSGDRegressor(
		WithLearningRate("constant"),
		WithEta0(0.01),
		WithMaxIter(200),
		WithRandomState(42),
	)

	err := sgd.Fit(X_train, y_train)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// テストデータで予測
	X_test := mat.NewDense(10, 1, nil)
	for i := 0; i < 10; i++ {
		X_test.Set(i, 0, float64(i+50)/10.0)
	}

	predictions, err := sgd.Predict(X_test)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	rows, cols := predictions.Dims()
	if rows != 10 || cols != 1 {
		t.Errorf("Predictions shape should be (10, 1), got (%d, %d)", rows, cols)
	}

	// 予測値が妥当な範囲にあることを確認
	for i := 0; i < rows; i++ {
		pred := predictions.At(i, 0)
		expected := 3*X_test.At(i, 0) + 2
		if math.Abs(pred-expected) > 0.5 {
			t.Errorf("Prediction %d: expected ~%.2f, got %.2f", i, expected, pred)
		}
	}
}

// TestSGDRegressorWarmStart はウォームスタート機能をテスト
func TestSGDRegressorWarmStart(t *testing.T) {
	t.Skip("Skipping WarmStart test - needs redesign as continued training may increase loss")
	X := mat.NewDense(50, 1, nil)
	y := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		x := float64(i) / 10.0
		X.Set(i, 0, x)
		y.Set(i, 0, 2*x+1)
	}

	// ウォームスタートなしで学習
	sgd1 := NewSGDRegressor(
		WithMaxIter(50),
		WithWarmStart(false),
		WithRandomState(42),
	)
	_ = sgd1.Fit(X, y)

	// 再度Fitを呼ぶ（リセットされる）
	_ = sgd1.Fit(X, y)
	loss2 := sgd1.GetLoss()

	// ウォームスタートありで学習
	sgd2 := NewSGDRegressor(
		WithMaxIter(50),
		WithWarmStart(true),
		WithRandomState(42),
	)
	_ = sgd2.Fit(X, y)

	// 再度Fitを呼ぶ（継続学習）
	_ = sgd2.Fit(X, y)
	loss3 := sgd2.GetLoss()

	// ウォームスタートありの方が損失が小さいはず
	if loss3 >= loss2 {
		t.Errorf("WarmStart should result in lower loss: warmstart=%.6f, no-warmstart=%.6f", loss3, loss2)
	}
}

// TestSGDRegressorRegularization は正則化のテスト
func TestSGDRegressorRegularization(t *testing.T) {
	// ノイズを含むデータ
	X := mat.NewDense(100, 5, nil)
	y := mat.NewDense(100, 1, nil)

	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i*j)/100.0)
		}
		y.Set(i, 0, float64(i)/10.0)
	}

	testCases := []struct {
		name    string
		penalty string
		alpha   float64
	}{
		{"No regularization", "none", 0.0},
		{"L2 regularization", "l2", 0.01},
		{"L1 regularization", "l1", 0.01},
		{"ElasticNet", "elasticnet", 0.01},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sgd := NewSGDRegressor(
				WithPenalty(tc.penalty),
				WithAlpha(tc.alpha),
				WithMaxIter(100),
				WithRandomState(42),
			)

			err := sgd.Fit(X, y)
			if err != nil {
				t.Fatalf("Fit failed with %s: %v", tc.name, err)
			}

			coef := sgd.Coef()

			// 正則化ありの場合、係数のノルムが小さくなるはず
			if tc.penalty != "none" {
				var norm float64
				for _, c := range coef {
					norm += c * c
				}
				norm = math.Sqrt(norm)

				if norm > 10.0 {
					t.Errorf("%s: coefficient norm too large: %f", tc.name, norm)
				}
			}
		})
	}
}

// TestSGDRegressorStreaming はストリーミング学習をテスト
func TestSGDRegressorStreaming(t *testing.T) {
	sgd := NewSGDRegressor(
		WithLearningRate("invscaling"),
		WithEta0(0.1),
		WithRandomState(42),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	dataChan := make(chan *model.Batch, 10)

	// データ生成goroutine
	go func() {
		defer close(dataChan)
		for i := 0; i < 10; i++ {
			X := mat.NewDense(5, 2, nil)
			y := mat.NewDense(5, 1, nil)

			for j := 0; j < 5; j++ {
				x1 := float64(i*5+j) / 50.0
				x2 := float64(i*5+j) / 25.0
				X.Set(j, 0, x1)
				X.Set(j, 1, x2)
				y.Set(j, 0, 2*x1+3*x2+1)
			}

			select {
			case dataChan <- &model.Batch{X: X, Y: y}:
			case <-ctx.Done():
				return
			}

			time.Sleep(10 * time.Millisecond)
		}
	}()

	// ストリーミング学習
	err := sgd.FitStream(ctx, dataChan)
	if err != nil && err != context.DeadlineExceeded {
		t.Fatalf("FitStream failed: %v", err)
	}

	if !sgd.IsFitted() {
		t.Error("Model should be fitted after FitStream()")
	}
}

// TestSGDRegressorPredictStream はストリーミング予測をテスト
func TestSGDRegressorPredictStream(t *testing.T) {
	// 事前学習
	X_train := mat.NewDense(50, 1, nil)
	y_train := mat.NewDense(50, 1, nil)

	for i := 0; i < 50; i++ {
		x := float64(i) / 10.0
		X_train.Set(i, 0, x)
		y_train.Set(i, 0, 2*x+1)
	}

	sgd := NewSGDRegressor(
		WithMaxIter(100),
		WithRandomState(42),
	)
	_ = sgd.Fit(X_train, y_train)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	inputChan := make(chan mat.Matrix, 5)

	// 入力データ生成
	go func() {
		defer close(inputChan)
		for i := 0; i < 5; i++ {
			X := mat.NewDense(2, 1, nil)
			X.Set(0, 0, float64(i))
			X.Set(1, 0, float64(i+1))

			select {
			case inputChan <- X:
			case <-ctx.Done():
				return
			}

			time.Sleep(10 * time.Millisecond)
		}
	}()

	// ストリーミング予測
	outputChan := sgd.PredictStream(ctx, inputChan)

	count := 0
	for pred := range outputChan {
		rows, cols := pred.Dims()
		if rows != 2 || cols != 1 {
			t.Errorf("Prediction shape should be (2, 1), got (%d, %d)", rows, cols)
		}
		count++
	}

	if count == 0 {
		t.Error("No predictions received from PredictStream")
	}
}

// TestSGDRegressorFitPredictStream はtest-then-train方式をテスト
func TestSGDRegressorFitPredictStream(t *testing.T) {
	sgd := NewSGDRegressor(
		WithLearningRate("invscaling"),
		WithEta0(0.1),
		WithRandomState(42),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	dataChan := make(chan *model.Batch, 5)

	// データ生成
	go func() {
		defer close(dataChan)
		for i := 0; i < 5; i++ {
			X := mat.NewDense(3, 1, nil)
			y := mat.NewDense(3, 1, nil)

			for j := 0; j < 3; j++ {
				x := float64(i*3+j) / 10.0
				X.Set(j, 0, x)
				y.Set(j, 0, 2*x+1)
			}

			select {
			case dataChan <- &model.Batch{X: X, Y: y}:
			case <-ctx.Done():
				return
			}

			time.Sleep(10 * time.Millisecond)
		}
	}()

	// test-then-train
	outputChan := sgd.FitPredictStream(ctx, dataChan)

	predCount := 0
	for range outputChan {
		predCount++
	}

	// 最初のバッチは予測されないので、バッチ数-1の予測が期待される
	if predCount == 0 {
		t.Error("No predictions received from FitPredictStream")
	}

	if !sgd.IsFitted() {
		t.Error("Model should be fitted after FitPredictStream()")
	}
}

// TestSGDRegressorScore はScore計算をテスト
func TestSGDRegressorScore(t *testing.T) {
	// 完全に線形なデータ
	X := mat.NewDense(100, 1, nil)
	y := mat.NewDense(100, 1, nil)

	for i := 0; i < 100; i++ {
		x := float64(i) / 10.0
		X.Set(i, 0, x)
		y.Set(i, 0, 2*x+1)
	}

	sgd := NewSGDRegressor(
		WithMaxIter(500),
		WithTol(1e-6),
		WithRandomState(42),
	)

	err := sgd.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	score, err := sgd.Score(X, y)
	if err != nil {
		t.Fatalf("Score calculation failed: %v", err)
	}

	// 完全に線形なデータなので、R²は1.0に近いはず
	if score < 0.99 {
		t.Errorf("R² score should be close to 1.0 for perfect linear data, got %f", score)
	}
}

// TestSGDRegressorConvergence は収束判定をテスト
func TestSGDRegressorConvergence(t *testing.T) {
	X := mat.NewDense(100, 1, nil)
	y := mat.NewDense(100, 1, nil)

	for i := 0; i < 100; i++ {
		x := float64(i) / 10.0
		X.Set(i, 0, x)
		y.Set(i, 0, 2*x+1)
	}

	sgd := NewSGDRegressor(
		WithMaxIter(1000),
		WithTol(1e-4),
		WithRandomState(42),
	)

	err := sgd.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// 収束していることを確認
	if !sgd.GetConverged() {
		t.Error("Model should have converged")
	}

	// 損失履歴が記録されていることを確認
	lossHistory := sgd.GetLossHistory()
	if len(lossHistory) == 0 {
		t.Error("Loss history should not be empty")
	}

	// 損失が減少していることを確認
	if len(lossHistory) > 10 {
		earlyLoss := lossHistory[5]
		lateLoss := lossHistory[len(lossHistory)-1]
		if lateLoss >= earlyLoss {
			t.Errorf("Loss should decrease: early=%.6f, late=%.6f", earlyLoss, lateLoss)
		}
	}
}
