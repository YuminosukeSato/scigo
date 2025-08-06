package linear_model

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// SGDRegressor は確率的勾配降下法による線形回帰モデル
// scikit-learnのSGDRegressorと互換性を持つ
type SGDRegressor struct {
	model.BaseEstimator

	// ハイパーパラメータ
	loss           string  // 損失関数: "squared_error", "huber", "epsilon_insensitive"
	penalty        string  // 正則化: "l2", "l1", "elasticnet", "none"
	alpha          float64 // 正則化の強度
	l1Ratio        float64 // Elastic Netのl1比率
	fitIntercept   bool    // 切片を学習するか
	maxIter        int     // 最大イテレーション数
	tol            float64 // 収束判定の許容誤差
	shuffle        bool    // 各エポックでデータをシャッフルするか
	verbose        int     // 詳細出力レベル
	epsilon        float64 // epsilon-insensitive損失のepsilon
	randomState    int64   // 乱数シード
	learningRate   string  // 学習率スケジュール: "constant", "optimal", "invscaling", "adaptive"
	eta0           float64 // 初期学習率
	power_t        float64 // invscalingの指数
	warmStart      bool    // 前回の学習から継続するか
	averageSGD     bool    // 平均化SGDを使用するか
	nIterNoChange  int     // 早期停止の判定イテレーション数

	// 学習パラメータ
	coef_      []float64 // 重み係数
	intercept_ float64   // 切片
	avgCoef_   []float64 // 平均化された重み（averageSGD用）
	avgIntercept_ float64 // 平均化された切片

	// 学習状態
	nIter_       int       // 実行されたイテレーション数
	t_           int64     // 総ステップ数（学習率計算用）
	lossHistory_ []float64 // 損失の履歴
	converged_   bool      // 収束フラグ
	
	// 内部状態
	mu       sync.RWMutex
	rng      *rand.Rand
	nFeatures_ int
}

// NewSGDRegressor は新しいSGDRegressorを作成
func NewSGDRegressor(options ...Option) *SGDRegressor {
	sgd := &SGDRegressor{
		loss:          "squared_error",
		penalty:       "l2",
		alpha:         0.0001,
		l1Ratio:       0.15,
		fitIntercept:  true,
		maxIter:       1000,
		tol:           1e-3,
		shuffle:       true,
		verbose:       0,
		epsilon:       0.1,
		randomState:   -1,
		learningRate:  "invscaling",
		eta0:          0.01,
		power_t:       0.25,
		warmStart:     false,
		averageSGD:    false,
		nIterNoChange: 5,
		lossHistory_:  make([]float64, 0),
	}

	for _, opt := range options {
		opt(sgd)
	}

	if sgd.randomState >= 0 {
		sgd.rng = rand.New(rand.NewSource(sgd.randomState))
	} else {
		sgd.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	return sgd
}

// Option はSGDRegressorの設定オプション
type Option func(*SGDRegressor)

// WithLoss は損失関数を設定
func WithLoss(loss string) Option {
	return func(sgd *SGDRegressor) {
		sgd.loss = loss
	}
}

// WithPenalty は正則化を設定
func WithPenalty(penalty string) Option {
	return func(sgd *SGDRegressor) {
		sgd.penalty = penalty
	}
}

// WithAlpha は正則化の強度を設定
func WithAlpha(alpha float64) Option {
	return func(sgd *SGDRegressor) {
		sgd.alpha = alpha
	}
}

// WithLearningRate は学習率スケジュールを設定
func WithLearningRate(lr string) Option {
	return func(sgd *SGDRegressor) {
		sgd.learningRate = lr
	}
}

// WithEta0 は初期学習率を設定
func WithEta0(eta0 float64) Option {
	return func(sgd *SGDRegressor) {
		sgd.eta0 = eta0
	}
}

// WithMaxIter は最大イテレーション数を設定
func WithMaxIter(maxIter int) Option {
	return func(sgd *SGDRegressor) {
		sgd.maxIter = maxIter
	}
}

// WithTol は収束判定の許容誤差を設定
func WithTol(tol float64) Option {
	return func(sgd *SGDRegressor) {
		sgd.tol = tol
	}
}

// WithFitIntercept は切片の学習有無を設定
func WithFitIntercept(fit bool) Option {
	return func(sgd *SGDRegressor) {
		sgd.fitIntercept = fit
	}
}

// WithWarmStart はウォームスタートの有効/無効を設定
func WithWarmStart(warmStart bool) Option {
	return func(sgd *SGDRegressor) {
		sgd.warmStart = warmStart
	}
}

// WithRandomState は乱数シードを設定
func WithRandomState(seed int64) Option {
	return func(sgd *SGDRegressor) {
		sgd.randomState = seed
		if seed >= 0 {
			sgd.rng = rand.New(rand.NewSource(seed))
		}
	}
}

// Fit はバッチ学習でモデルを訓練
func (sgd *SGDRegressor) Fit(X, y mat.Matrix) error {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()

	if !sgd.warmStart || sgd.coef_ == nil {
		sgd.reset()
	}

	rows, cols := X.Dims()
	sgd.nFeatures_ = cols

	if sgd.coef_ == nil {
		sgd.coef_ = make([]float64, cols)
		sgd.avgCoef_ = make([]float64, cols)
		// Xavier初期化
		scale := math.Sqrt(2.0 / float64(cols))
		for i := range sgd.coef_ {
			sgd.coef_[i] = sgd.rng.NormFloat64() * scale
		}
	}

	// バッチSGD実装
	for iter := 0; iter < sgd.maxIter; iter++ {
		epochLoss := 0.0
		
		// データのシャッフル
		indices := make([]int, rows)
		for i := range indices {
			indices[i] = i
		}
		if sgd.shuffle {
			sgd.rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		// ミニバッチ処理
		for _, idx := range indices {
			xi := mat.Row(nil, idx, X)
			yi := y.At(idx, 0)
			
			loss := sgd.updateWeights(xi, yi)
			epochLoss += loss
		}

		epochLoss /= float64(rows)
		sgd.lossHistory_ = append(sgd.lossHistory_, epochLoss)
		sgd.nIter_++

		// 収束判定
		if sgd.checkConvergence() {
			sgd.converged_ = true
			break
		}

		if sgd.verbose > 0 && iter%10 == 0 {
			fmt.Printf("Iteration %d, loss: %.6f\n", iter, epochLoss)
		}
	}

	if !sgd.converged_ {
		errors.Warn(errors.NewConvergenceWarning("SGDRegressor", sgd.nIter_, "Maximum number of iterations reached"))
	}

	sgd.SetFitted()
	return nil
}

// PartialFit はミニバッチでモデルを逐次的に学習（オンライン学習）
func (sgd *SGDRegressor) PartialFit(X, y mat.Matrix, classes []int) error {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()

	rows, cols := X.Dims()
	
	// 初回呼び出し時の初期化
	if sgd.coef_ == nil {
		sgd.nFeatures_ = cols
		sgd.coef_ = make([]float64, cols)
		sgd.avgCoef_ = make([]float64, cols)
		// Xavier初期化
		scale := math.Sqrt(2.0 / float64(cols))
		for i := range sgd.coef_ {
			sgd.coef_[i] = sgd.rng.NormFloat64() * scale
		}
	}

	if cols != sgd.nFeatures_ {
		return errors.NewDimensionError("PartialFit", sgd.nFeatures_, cols, 1)
	}

	// ミニバッチ処理
	batchLoss := 0.0
	for i := 0; i < rows; i++ {
		xi := mat.Row(nil, i, X)
		yi := y.At(i, 0)
		
		loss := sgd.updateWeights(xi, yi)
		batchLoss += loss
	}

	batchLoss /= float64(rows)
	sgd.lossHistory_ = append(sgd.lossHistory_, batchLoss)
	sgd.nIter_++  // Increment iteration count for PartialFit
	
	sgd.SetFitted()
	return nil
}

// updateWeights は単一サンプルで重みを更新（SGDのコア処理）
func (sgd *SGDRegressor) updateWeights(x []float64, y float64) float64 {
	// 予測値計算
	pred := sgd.intercept_
	for i, xi := range x {
		pred += sgd.coef_[i] * xi
	}

	// 予測値の数値安定性チェック
	if err := errors.CheckScalar("prediction", pred, sgd.nIter_); err != nil {
		errors.Warn(err)
		// 数値不安定性が検出された場合、学習率を下げてリトライ
		sgd.eta0 = sgd.eta0 * 0.1
		return 0
	}

	// 損失と勾配計算
	var loss, dloss float64
	switch sgd.loss {
	case "squared_error":
		diff := pred - y
		loss = 0.5 * diff * diff
		dloss = diff
	case "huber":
		diff := pred - y
		if math.Abs(diff) <= sgd.epsilon {
			loss = 0.5 * diff * diff
			dloss = diff
		} else {
			loss = sgd.epsilon * (math.Abs(diff) - 0.5*sgd.epsilon)
			dloss = sgd.epsilon * sign(diff)
		}
	case "epsilon_insensitive":
		diff := math.Abs(pred - y)
		if diff <= sgd.epsilon {
			loss = 0
			dloss = 0
		} else {
			loss = diff - sgd.epsilon
			dloss = sign(pred - y)
		}
	default:
		diff := pred - y
		loss = 0.5 * diff * diff
		dloss = diff
	}

	// 学習率計算
	lr := sgd.getLearningRate()
	sgd.t_++

	// 重み更新（勾配降下 + 正則化）
	gradients := make([]float64, len(x))
	for i, xi := range x {
		grad := dloss * xi
		
		// 正則化項の勾配
		switch sgd.penalty {
		case "l2":
			grad += sgd.alpha * sgd.coef_[i]
		case "l1":
			grad += sgd.alpha * sign(sgd.coef_[i])
		case "elasticnet":
			grad += sgd.alpha * (sgd.l1Ratio*sign(sgd.coef_[i]) + (1-sgd.l1Ratio)*sgd.coef_[i])
		}
		
		gradients[i] = grad
	}
	
	// 勾配クリッピング（勾配爆発防止）
	gradients = errors.ClipGradient(gradients, 10.0)
	
	// 重み更新
	for i, grad := range gradients {
		sgd.coef_[i] -= lr * grad
		
		// 更新後の重みの数値安定性チェック
		if err := errors.CheckScalar("weight_update", sgd.coef_[i], sgd.nIter_); err != nil {
			errors.Warn(err)
			// ロールバック
			sgd.coef_[i] += lr * grad
		}
		
		// 平均化SGD
		if sgd.averageSGD {
			sgd.avgCoef_[i] = (sgd.avgCoef_[i]*float64(sgd.t_-1) + sgd.coef_[i]) / float64(sgd.t_)
		}
	}

	// 切片更新
	if sgd.fitIntercept {
		sgd.intercept_ -= lr * dloss
		if sgd.averageSGD {
			sgd.avgIntercept_ = (sgd.avgIntercept_*float64(sgd.t_-1) + sgd.intercept_) / float64(sgd.t_)
		}
	}

	return loss
}

// getLearningRate は現在の学習率を計算
func (sgd *SGDRegressor) getLearningRate() float64 {
	switch sgd.learningRate {
	case "constant":
		return sgd.eta0
	case "optimal":
		return 1.0 / (sgd.alpha * (float64(sgd.t_) + 1))
	case "invscaling":
		return sgd.eta0 / math.Pow(float64(sgd.t_)+1, sgd.power_t)
	case "adaptive":
		// 簡易的な適応的学習率（改善の余地あり）
		if len(sgd.lossHistory_) > 5 {
			recent := sgd.lossHistory_[len(sgd.lossHistory_)-5:]
			if isIncreasing(recent) {
				return sgd.eta0 * 0.5
			}
		}
		return sgd.eta0
	default:
		return sgd.eta0
	}
}

// checkConvergence は収束判定
func (sgd *SGDRegressor) checkConvergence() bool {
	if len(sgd.lossHistory_) < sgd.nIterNoChange+1 {
		return false
	}

	recent := sgd.lossHistory_[len(sgd.lossHistory_)-sgd.nIterNoChange:]
	maxLoss := recent[0]
	minLoss := recent[0]
	
	for _, loss := range recent {
		if loss > maxLoss {
			maxLoss = loss
		}
		if loss < minLoss {
			minLoss = loss
		}
	}

	return (maxLoss - minLoss) < sgd.tol
}

// Predict は入力データに対する予測を行う
func (sgd *SGDRegressor) Predict(X mat.Matrix) (mat.Matrix, error) {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()

	if !sgd.IsFitted() {
		return nil, errors.NewNotFittedError("SGDRegressor", "Predict")
	}

	rows, cols := X.Dims()
	if cols != sgd.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", sgd.nFeatures_, cols, 1)
	}

	predictions := mat.NewDense(rows, 1, nil)
	
	coef := sgd.coef_
	intercept := sgd.intercept_
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		coef = sgd.avgCoef_
		intercept = sgd.avgIntercept_
	}

	for i := 0; i < rows; i++ {
		pred := intercept
		for j := 0; j < cols; j++ {
			pred += X.At(i, j) * coef[j]
		}
		predictions.Set(i, 0, pred)
	}

	return predictions, nil
}

// FitStream はデータストリームからモデルを学習
func (sgd *SGDRegressor) FitStream(ctx context.Context, dataChan <-chan *model.Batch) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case batch, ok := <-dataChan:
			if !ok {
				return nil
			}
			if err := sgd.PartialFit(batch.X, batch.Y, nil); err != nil {
				return err
			}
		}
	}
}

// PredictStream は入力ストリームに対してリアルタイム予測
func (sgd *SGDRegressor) PredictStream(ctx context.Context, inputChan <-chan mat.Matrix) <-chan mat.Matrix {
	outputChan := make(chan mat.Matrix)
	
	go func() {
		defer close(outputChan)
		
		for {
			select {
			case <-ctx.Done():
				return
			case X, ok := <-inputChan:
				if !ok {
					return
				}
				
				pred, err := sgd.Predict(X)
				if err != nil {
					// In a real-world scenario, we might want to log this error
					continue
				}
				
				select {
				case outputChan <- pred:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	
	return outputChan
}

// FitPredictStream は学習と予測を同時に行う（test-then-train方式）
func (sgd *SGDRegressor) FitPredictStream(ctx context.Context, dataChan <-chan *model.Batch) <-chan mat.Matrix {
	outputChan := make(chan mat.Matrix)
	
	go func() {
		defer close(outputChan)
		
		for {
			select {
			case <-ctx.Done():
				return
			case batch, ok := <-dataChan:
				if !ok {
					return
				}
				
				// まず予測
				if sgd.IsFitted() {
					pred, err := sgd.Predict(batch.X)
					if err == nil {
						select {
						case outputChan <- pred:
						case <-ctx.Done():
							return
						}
					}
				}
				
				// その後学習
				if err := sgd.PartialFit(batch.X, batch.Y, nil); err != nil {
					// In a real-world scenario, we might want to log this error
					continue
				}
			}
		}
	}()
	
	return outputChan
}

// NIterations は実行された学習イテレーション数を返す
func (sgd *SGDRegressor) NIterations() int {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.nIter_
}

// IsWarmStart はウォームスタートが有効かどうかを返す
func (sgd *SGDRegressor) IsWarmStart() bool {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.warmStart
}

// SetWarmStart はウォームスタートの有効/無効を設定
func (sgd *SGDRegressor) SetWarmStart(warmStart bool) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.warmStart = warmStart
}

// GetLoss は現在の損失値を返す
func (sgd *SGDRegressor) GetLoss() float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if len(sgd.lossHistory_) == 0 {
		return math.Inf(1)
	}
	return sgd.lossHistory_[len(sgd.lossHistory_)-1]
}

// GetLossHistory は損失値の履歴を返す
func (sgd *SGDRegressor) GetLossHistory() []float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	history := make([]float64, len(sgd.lossHistory_))
	copy(history, sgd.lossHistory_)
	return history
}

// GetConverged は収束したかどうかを返す
func (sgd *SGDRegressor) GetConverged() bool {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.converged_
}

// GetLearningRate は現在の学習率を返す
func (sgd *SGDRegressor) GetLearningRate() float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.getLearningRate()
}

// SetLearningRate は学習率を設定
func (sgd *SGDRegressor) SetLearningRate(lr float64) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.eta0 = lr
}

// GetLearningRateSchedule は学習率スケジュールを返す
func (sgd *SGDRegressor) GetLearningRateSchedule() string {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.learningRate
}

// SetLearningRateSchedule は学習率スケジュールを設定
func (sgd *SGDRegressor) SetLearningRateSchedule(schedule string) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.learningRate = schedule
}

// Coef は学習された重み係数を返す
func (sgd *SGDRegressor) Coef() []float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		coef := make([]float64, len(sgd.avgCoef_))
		copy(coef, sgd.avgCoef_)
		return coef
	}
	
	coef := make([]float64, len(sgd.coef_))
	copy(coef, sgd.coef_)
	return coef
}

// Intercept は学習された切片を返す
func (sgd *SGDRegressor) Intercept() float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		return sgd.avgIntercept_
	}
	return sgd.intercept_
}

// Score はモデルの決定係数（R²）を計算
func (sgd *SGDRegressor) Score(X, y mat.Matrix) (float64, error) {
	predictions, err := sgd.Predict(X)
	if err != nil {
		return 0, err
	}

	rows, _ := y.Dims()
	
	// 平均値計算
	var yMean float64
	for i := 0; i < rows; i++ {
		yMean += y.At(i, 0)
	}
	yMean /= float64(rows)

	// SS_tot と SS_res 計算
	var ssTot, ssRes float64
	for i := 0; i < rows; i++ {
		yi := y.At(i, 0)
		predi := predictions.At(i, 0)
		
		ssTot += (yi - yMean) * (yi - yMean)
		ssRes += (yi - predi) * (yi - predi)
	}

	if ssTot == 0 {
		return 0, errors.NewValueError("Score", "Cannot compute score with zero variance in y_true")
	}

	return 1.0 - (ssRes / ssTot), nil
}

// reset は内部状態をリセット
func (sgd *SGDRegressor) reset() {
	sgd.coef_ = nil
	sgd.intercept_ = 0
	sgd.avgCoef_ = nil
	sgd.avgIntercept_ = 0
	sgd.nIter_ = 0
	sgd.t_ = 0
	sgd.lossHistory_ = make([]float64, 0)
	sgd.converged_ = false
	sgd.Reset() // BaseEstimatorのリセット
}

// 補助関数
func sign(x float64) float64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}

func isIncreasing(values []float64) bool {
	for i := 1; i < len(values); i++ {
		if values[i] < values[i-1] {
			return false
		}
	}
	return true
}

