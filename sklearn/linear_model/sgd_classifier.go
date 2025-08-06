package linear_model

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/YuminosukeSato/GoML/core/model"
	"github.com/YuminosukeSato/GoML/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// SGDClassifier は確率的勾配降下法による分類モデル
// scikit-learnのSGDClassifierと互換性を持つ
type SGDClassifier struct {
	model.BaseEstimator

	// ハイパーパラメータ
	loss           string  // 損失関数: "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"
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
	classWeight    string  // クラス重み: "balanced", "none"

	// 学習パラメータ
	coef_      [][]float64 // 重み係数（クラス数 x 特徴数）
	intercept_ []float64   // 切片（クラス数）
	avgCoef_   [][]float64 // 平均化された重み
	avgIntercept_ []float64 // 平均化された切片
	classes_   []int       // クラスラベル
	nClasses_  int         // クラス数

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

// NewSGDClassifier は新しいSGDClassifierを作成
func NewSGDClassifier(options ...ClassifierOption) *SGDClassifier {
	sgd := &SGDClassifier{
		loss:          "hinge",
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
		classWeight:   "none",
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

// ClassifierOption はSGDClassifierの設定オプション
type ClassifierOption func(*SGDClassifier)

// WithClassifierLoss は損失関数を設定
func WithClassifierLoss(loss string) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.loss = loss
	}
}

// WithClassifierPenalty は正則化を設定
func WithClassifierPenalty(penalty string) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.penalty = penalty
	}
}

// WithClassifierAlpha は正則化の強度を設定
func WithClassifierAlpha(alpha float64) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.alpha = alpha
	}
}

// WithClassifierLearningRate は学習率スケジュールを設定
func WithClassifierLearningRate(lr string) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.learningRate = lr
	}
}

// WithClassifierEta0 は初期学習率を設定
func WithClassifierEta0(eta0 float64) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.eta0 = eta0
	}
}

// WithClassifierMaxIter は最大イテレーション数を設定
func WithClassifierMaxIter(maxIter int) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.maxIter = maxIter
	}
}

// WithClassifierRandomState は乱数シードを設定
func WithClassifierRandomState(seed int64) ClassifierOption {
	return func(sgd *SGDClassifier) {
		sgd.randomState = seed
		if seed >= 0 {
			sgd.rng = rand.New(rand.NewSource(seed))
		}
	}
}

// Fit はバッチ学習でモデルを訓練
func (sgd *SGDClassifier) Fit(X, y mat.Matrix) error {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()

	if !sgd.warmStart || sgd.coef_ == nil {
		sgd.reset()
	}

	rows, cols := X.Dims()
	sgd.nFeatures_ = cols

	// クラスを特定
	if sgd.classes_ == nil {
		sgd.extractClasses(y)
	}

	// 重みの初期化
	if sgd.coef_ == nil {
		sgd.initializeWeights()
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
			yi := int(y.At(idx, 0))
			
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

	sgd.SetFitted()
	return nil
}

// PartialFit はミニバッチでモデルを逐次的に学習（オンライン学習）
func (sgd *SGDClassifier) PartialFit(X, y mat.Matrix, classes []int) error {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()

	rows, cols := X.Dims()
	
	// 初回呼び出し時の初期化
	if sgd.coef_ == nil {
		sgd.nFeatures_ = cols
		
		// classesが指定されている場合はそれを使用
		if classes != nil {
			sgd.classes_ = make([]int, len(classes))
			copy(sgd.classes_, classes)
			sgd.nClasses_ = len(classes)
		} else {
			sgd.extractClasses(y)
		}
		
		sgd.initializeWeights()
	}

	if cols != sgd.nFeatures_ {
		return errors.NewDimensionError("PartialFit", sgd.nFeatures_, cols, 1)
	}

	// ミニバッチ処理
	batchLoss := 0.0
	for i := 0; i < rows; i++ {
		xi := mat.Row(nil, i, X)
		yi := int(y.At(i, 0))
		
		loss := sgd.updateWeights(xi, yi)
		batchLoss += loss
	}

	batchLoss /= float64(rows)
	sgd.lossHistory_ = append(sgd.lossHistory_, batchLoss)
	
	sgd.SetFitted()
	return nil
}

// updateWeights は単一サンプルで重みを更新（SGDのコア処理）
func (sgd *SGDClassifier) updateWeights(x []float64, y int) float64 {
	// クラスインデックスを取得
	classIdx := sgd.getClassIndex(y)
	if classIdx == -1 {
		return 0 // 未知のクラス
	}

	var totalLoss float64

	// 各クラスについて処理
	for c := 0; c < sgd.nClasses_; c++ {
		// スコア計算
		score := sgd.intercept_[c]
		for i, xi := range x {
			score += sgd.coef_[c][i] * xi
		}

		var loss, dloss float64
		target := -1.0
		if c == classIdx {
			target = 1.0
		}

		// 損失と勾配計算
		switch sgd.loss {
		case "hinge":
			margin := target * score
			if margin < 1 {
				loss = 1 - margin
				dloss = -target
			} else {
				loss = 0
				dloss = 0
			}
		case "squared_hinge":
			margin := target * score
			if margin < 1 {
				diff := 1 - margin
				loss = 0.5 * diff * diff
				dloss = -target * diff
			} else {
				loss = 0
				dloss = 0
			}
		case "log_loss":
			// ロジスティック損失
			z := target * score
			if z > 0 {
				exp_z := math.Exp(-z)
				loss = math.Log(1 + exp_z)
				dloss = -target * exp_z / (1 + exp_z)
			} else {
				exp_z := math.Exp(z)
				loss = -z + math.Log(1 + exp_z)
				dloss = -target * exp_z / (1 + exp_z)
			}
		case "modified_huber":
			margin := target * score
			if margin < -1 {
				loss = -4 * margin
				dloss = -4 * target
			} else if margin < 1 {
				diff := 1 - margin
				loss = diff * diff
				dloss = -2 * target * diff
			} else {
				loss = 0
				dloss = 0
			}
		case "perceptron":
			if target*score <= 0 {
				loss = -target * score
				dloss = -target
			} else {
				loss = 0
				dloss = 0
			}
		default:
			// デフォルトはhinge
			margin := target * score
			if margin < 1 {
				loss = 1 - margin
				dloss = -target
			} else {
				loss = 0
				dloss = 0
			}
		}

		totalLoss += loss

		// 学習率計算
		lr := sgd.getLearningRate()

		// 重み更新（勾配降下 + 正則化）
		for i, xi := range x {
			grad := dloss * xi
			
			// 正則化項の勾配
			switch sgd.penalty {
			case "l2":
				grad += sgd.alpha * sgd.coef_[c][i]
			case "l1":
				grad += sgd.alpha * sign(sgd.coef_[c][i])
			case "elasticnet":
				grad += sgd.alpha * (sgd.l1Ratio*sign(sgd.coef_[c][i]) + (1-sgd.l1Ratio)*sgd.coef_[c][i])
			}
			
			sgd.coef_[c][i] -= lr * grad
			
			// 平均化SGD
			if sgd.averageSGD {
				sgd.avgCoef_[c][i] = (sgd.avgCoef_[c][i]*float64(sgd.t_-1) + sgd.coef_[c][i]) / float64(sgd.t_)
			}
		}

		// 切片更新
		if sgd.fitIntercept {
			sgd.intercept_[c] -= lr * dloss
			if sgd.averageSGD {
				sgd.avgIntercept_[c] = (sgd.avgIntercept_[c]*float64(sgd.t_-1) + sgd.intercept_[c]) / float64(sgd.t_)
			}
		}
	}

	sgd.t_++
	return totalLoss / float64(sgd.nClasses_)
}

// Predict は入力データに対する予測を行う
func (sgd *SGDClassifier) Predict(X mat.Matrix) (mat.Matrix, error) {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()

	if !sgd.IsFitted() {
		return nil, errors.NewNotFittedError("SGDClassifier", "Predict")
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
		maxScore := math.Inf(-1)
		predictedClass := sgd.classes_[0]
		
		// 各クラスのスコアを計算
		for c := 0; c < sgd.nClasses_; c++ {
			score := intercept[c]
			for j := 0; j < cols; j++ {
				score += X.At(i, j) * coef[c][j]
			}
			
			if score > maxScore {
				maxScore = score
				predictedClass = sgd.classes_[c]
			}
		}
		
		predictions.Set(i, 0, float64(predictedClass))
	}

	return predictions, nil
}

// PredictProba は各クラスの予測確率を返す
func (sgd *SGDClassifier) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()

	if !sgd.IsFitted() {
		return nil, errors.NewNotFittedError("SGDClassifier", "Predict")
	}

	rows, cols := X.Dims()
	if cols != sgd.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", sgd.nFeatures_, cols, 1)
	}

	probabilities := mat.NewDense(rows, sgd.nClasses_, nil)
	
	coef := sgd.coef_
	intercept := sgd.intercept_
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		coef = sgd.avgCoef_
		intercept = sgd.avgIntercept_
	}

	for i := 0; i < rows; i++ {
		scores := make([]float64, sgd.nClasses_)
		
		// 各クラスのスコアを計算
		for c := 0; c < sgd.nClasses_; c++ {
			scores[c] = intercept[c]
			for j := 0; j < cols; j++ {
				scores[c] += X.At(i, j) * coef[c][j]
			}
		}
		
		// ソフトマックス変換
		expSum := 0.0
		for c := 0; c < sgd.nClasses_; c++ {
			scores[c] = math.Exp(scores[c])
			expSum += scores[c]
		}
		
		for c := 0; c < sgd.nClasses_; c++ {
			probabilities.Set(i, c, scores[c]/expSum)
		}
	}

	return probabilities, nil
}

// DecisionFunction は決定関数の値を返す
func (sgd *SGDClassifier) DecisionFunction(X mat.Matrix) (mat.Matrix, error) {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()

	if !sgd.IsFitted() {
		return nil, errors.NewNotFittedError("SGDClassifier", "Predict")
	}

	rows, cols := X.Dims()
	if cols != sgd.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", sgd.nFeatures_, cols, 1)
	}

	decisions := mat.NewDense(rows, sgd.nClasses_, nil)
	
	coef := sgd.coef_
	intercept := sgd.intercept_
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		coef = sgd.avgCoef_
		intercept = sgd.avgIntercept_
	}

	for i := 0; i < rows; i++ {
		for c := 0; c < sgd.nClasses_; c++ {
			score := intercept[c]
			for j := 0; j < cols; j++ {
				score += X.At(i, j) * coef[c][j]
			}
			decisions.Set(i, c, score)
		}
	}

	return decisions, nil
}

// FitStream はデータストリームからモデルを学習
func (sgd *SGDClassifier) FitStream(ctx context.Context, dataChan <-chan *model.Batch) error {
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
func (sgd *SGDClassifier) PredictStream(ctx context.Context, inputChan <-chan mat.Matrix) <-chan mat.Matrix {
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
func (sgd *SGDClassifier) FitPredictStream(ctx context.Context, dataChan <-chan *model.Batch) <-chan mat.Matrix {
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
					continue
				}
			}
		}
	}()
	
	return outputChan
}

// インターフェース実装のための各種メソッド

// NIterations は実行された学習イテレーション数を返す
func (sgd *SGDClassifier) NIterations() int {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.nIter_
}

// IsWarmStart はウォームスタートが有効かどうかを返す
func (sgd *SGDClassifier) IsWarmStart() bool {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.warmStart
}

// SetWarmStart はウォームスタートの有効/無効を設定
func (sgd *SGDClassifier) SetWarmStart(warmStart bool) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.warmStart = warmStart
}

// GetLoss は現在の損失値を返す
func (sgd *SGDClassifier) GetLoss() float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if len(sgd.lossHistory_) == 0 {
		return math.Inf(1)
	}
	return sgd.lossHistory_[len(sgd.lossHistory_)-1]
}

// GetLossHistory は損失値の履歴を返す
func (sgd *SGDClassifier) GetLossHistory() []float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	history := make([]float64, len(sgd.lossHistory_))
	copy(history, sgd.lossHistory_)
	return history
}

// GetConverged は収束したかどうかを返す
func (sgd *SGDClassifier) GetConverged() bool {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.converged_
}

// GetLearningRate は現在の学習率を返す
func (sgd *SGDClassifier) GetLearningRate() float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.getLearningRate()
}

// SetLearningRate は学習率を設定
func (sgd *SGDClassifier) SetLearningRate(lr float64) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.eta0 = lr
}

// GetLearningRateSchedule は学習率スケジュールを返す
func (sgd *SGDClassifier) GetLearningRateSchedule() string {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	return sgd.learningRate
}

// SetLearningRateSchedule は学習率スケジュールを設定
func (sgd *SGDClassifier) SetLearningRateSchedule(schedule string) {
	sgd.mu.Lock()
	defer sgd.mu.Unlock()
	sgd.learningRate = schedule
}

// Coef は学習された重み係数を返す
func (sgd *SGDClassifier) Coef() [][]float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if sgd.averageSGD && sgd.avgCoef_ != nil {
		coef := make([][]float64, len(sgd.avgCoef_))
		for i := range sgd.avgCoef_ {
			coef[i] = make([]float64, len(sgd.avgCoef_[i]))
			copy(coef[i], sgd.avgCoef_[i])
		}
		return coef
	}
	
	coef := make([][]float64, len(sgd.coef_))
	for i := range sgd.coef_ {
		coef[i] = make([]float64, len(sgd.coef_[i]))
		copy(coef[i], sgd.coef_[i])
	}
	return coef
}

// Intercept は学習された切片を返す
func (sgd *SGDClassifier) Intercept() []float64 {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	if sgd.averageSGD && sgd.avgIntercept_ != nil {
		intercept := make([]float64, len(sgd.avgIntercept_))
		copy(intercept, sgd.avgIntercept_)
		return intercept
	}
	
	intercept := make([]float64, len(sgd.intercept_))
	copy(intercept, sgd.intercept_)
	return intercept
}

// Classes は学習されたクラスラベルを返す
func (sgd *SGDClassifier) Classes() []int {
	sgd.mu.RLock()
	defer sgd.mu.RUnlock()
	
	classes := make([]int, len(sgd.classes_))
	copy(classes, sgd.classes_)
	return classes
}

// Score はモデルの精度を計算
func (sgd *SGDClassifier) Score(X, y mat.Matrix) (float64, error) {
	predictions, err := sgd.Predict(X)
	if err != nil {
		return 0, err
	}

	rows, _ := y.Dims()
	correct := 0
	
	for i := 0; i < rows; i++ {
		if int(predictions.At(i, 0)) == int(y.At(i, 0)) {
			correct++
		}
	}

	return float64(correct) / float64(rows), nil
}

// 内部ヘルパーメソッド

// extractClasses はデータからクラスを抽出
func (sgd *SGDClassifier) extractClasses(y mat.Matrix) {
	rows, _ := y.Dims()
	classSet := make(map[int]bool)
	
	for i := 0; i < rows; i++ {
		class := int(y.At(i, 0))
		classSet[class] = true
	}
	
	classes := make([]int, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}
	
	// ソート
	for i := 0; i < len(classes); i++ {
		for j := i + 1; j < len(classes); j++ {
			if classes[i] > classes[j] {
				classes[i], classes[j] = classes[j], classes[i]
			}
		}
	}
	
	sgd.classes_ = classes
	sgd.nClasses_ = len(classes)
}

// initializeWeights は重みを初期化
func (sgd *SGDClassifier) initializeWeights() {
	sgd.coef_ = make([][]float64, sgd.nClasses_)
	sgd.intercept_ = make([]float64, sgd.nClasses_)
	sgd.avgCoef_ = make([][]float64, sgd.nClasses_)
	sgd.avgIntercept_ = make([]float64, sgd.nClasses_)
	
	scale := math.Sqrt(2.0 / float64(sgd.nFeatures_))
	for c := 0; c < sgd.nClasses_; c++ {
		sgd.coef_[c] = make([]float64, sgd.nFeatures_)
		sgd.avgCoef_[c] = make([]float64, sgd.nFeatures_)
		
		for i := 0; i < sgd.nFeatures_; i++ {
			sgd.coef_[c][i] = sgd.rng.NormFloat64() * scale
		}
	}
}

// getClassIndex はクラス値からインデックスを取得
func (sgd *SGDClassifier) getClassIndex(class int) int {
	for i, c := range sgd.classes_ {
		if c == class {
			return i
		}
	}
	return -1
}

// getLearningRate は現在の学習率を計算
func (sgd *SGDClassifier) getLearningRate() float64 {
	switch sgd.learningRate {
	case "constant":
		return sgd.eta0
	case "optimal":
		return 1.0 / (sgd.alpha * (float64(sgd.t_) + 1))
	case "invscaling":
		return sgd.eta0 / math.Pow(float64(sgd.t_)+1, sgd.power_t)
	case "adaptive":
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
func (sgd *SGDClassifier) checkConvergence() bool {
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

// reset は内部状態をリセット
func (sgd *SGDClassifier) reset() {
	sgd.coef_ = nil
	sgd.intercept_ = nil
	sgd.avgCoef_ = nil
	sgd.avgIntercept_ = nil
	sgd.classes_ = nil
	sgd.nClasses_ = 0
	sgd.nIter_ = 0
	sgd.t_ = 0
	sgd.lossHistory_ = make([]float64, 0)
	sgd.converged_ = false
	sgd.Reset() // BaseEstimatorのリセット
}