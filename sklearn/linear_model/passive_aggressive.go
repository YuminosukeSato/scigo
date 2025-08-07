package linear_model

import (
	"context"
	"math"
	"sync"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// PassiveAggressiveRegressor is a passive aggressive regression model
// Compatible with scikit-learn's PassiveAggressiveRegressor
type PassiveAggressiveRegressor struct {
	state *model.StateManager // State management (composition instead of embedding)

	// Hyperparameters
	C            float64 // Regularization parameter
	fitIntercept bool    // Whether to learn the intercept
	maxIter      int     // Maximum number of iterations
	tol          float64 // Tolerance for convergence
	shuffle      bool    // Whether to shuffle data at each epoch
	verbose      int     // Verbosity level
	randomState  int64   // Random seed
	warmStart    bool    // Whether to continue from previous training
	averagePA    bool    // Whether to use averaged PA
	loss         string  // Loss function: "epsilon_insensitive", "squared_epsilon_insensitive"
	epsilon      float64 // Epsilon for epsilon-insensitive loss

	// Learning parameters
	coef_         []float64 // Weight coefficients
	intercept_    float64   // Intercept
	avgCoef_      []float64 // Averaged weights
	avgIntercept_ float64   // Averaged intercept

	// Learning state
	nIter_     int   // Number of iterations executed
	t_         int64 // Total step count
	converged_ bool  // Convergence flag

	// Internal state
	mu         sync.RWMutex
	nFeatures_ int
}

// PassiveAggressiveClassifier is a passive aggressive classification model
type PassiveAggressiveClassifier struct {
	state *model.StateManager // State management (composition instead of embedding)

	// Hyperparameters
	C            float64 // Regularization parameter
	fitIntercept bool    // Whether to learn the intercept
	maxIter      int     // Maximum number of iterations
	tol          float64 // Tolerance for convergence
	shuffle      bool    // Whether to shuffle data at each epoch
	verbose      int     // Verbosity level
	randomState  int64   // Random seed
	warmStart    bool    // Whether to continue from previous training
	averagePA    bool    // Whether to use averaged PA
	loss         string  // Loss function: "hinge", "squared_hinge"
	classWeight  string  // Class weight: "balanced", "none"

	// Learning parameters
	coef_         [][]float64 // Weight coefficients (n_classes x n_features)
	intercept_    []float64   // Intercept (n_classes)
	avgCoef_      [][]float64 // Averaged weights
	avgIntercept_ []float64   // Averaged intercept
	classes_      []int       // Class labels
	nClasses_     int         // Number of classes

	// Learning state
	nIter_     int   // Number of iterations executed
	t_         int64 // Total step count
	converged_ bool  // Convergence flag

	// Internal state
	mu         sync.RWMutex
	nFeatures_ int
}

// PassiveAggressiveOption is a configuration option
type PassiveAggressiveOption func(interface{})

// NewPassiveAggressiveRegressor creates a new PassiveAggressiveRegressor
func NewPassiveAggressiveRegressor(options ...PassiveAggressiveOption) *PassiveAggressiveRegressor {
	pa := &PassiveAggressiveRegressor{
		state:        model.NewStateManager(),
		C:            1.0,
		fitIntercept: true,
		maxIter:      1000,
		tol:          1e-3,
		shuffle:      true,
		verbose:      0,
		randomState:  -1,
		warmStart:    false,
		averagePA:    false,
		loss:         "epsilon_insensitive",
		epsilon:      0.1,
	}

	for _, opt := range options {
		opt(pa)
	}

	return pa
}

// NewPassiveAggressiveClassifier creates a new PassiveAggressiveClassifier
func NewPassiveAggressiveClassifier(options ...PassiveAggressiveOption) *PassiveAggressiveClassifier {
	pa := &PassiveAggressiveClassifier{
		state:        model.NewStateManager(),
		C:            1.0,
		fitIntercept: true,
		maxIter:      1000,
		tol:          1e-3,
		shuffle:      true,
		verbose:      0,
		randomState:  -1,
		warmStart:    false,
		averagePA:    false,
		loss:         "hinge",
		classWeight:  "none",
	}

	for _, opt := range options {
		opt(pa)
	}

	return pa
}

// WithPAC sets regularization parameter
func WithPAC(c float64) PassiveAggressiveOption {
	return func(pa interface{}) {
		switch p := pa.(type) {
		case *PassiveAggressiveRegressor:
			p.C = c
		case *PassiveAggressiveClassifier:
			p.C = c
		}
	}
}

// WithPAMaxIter sets maximum number of iterations
func WithPAMaxIter(maxIter int) PassiveAggressiveOption {
	return func(pa interface{}) {
		switch p := pa.(type) {
		case *PassiveAggressiveRegressor:
			p.maxIter = maxIter
		case *PassiveAggressiveClassifier:
			p.maxIter = maxIter
		}
	}
}

// WithPAFitIntercept sets whether to learn intercept
func WithPAFitIntercept(fit bool) PassiveAggressiveOption {
	return func(pa interface{}) {
		switch p := pa.(type) {
		case *PassiveAggressiveRegressor:
			p.fitIntercept = fit
		case *PassiveAggressiveClassifier:
			p.fitIntercept = fit
		}
	}
}

// WithPALoss sets loss function
func WithPALoss(loss string) PassiveAggressiveOption {
	return func(pa interface{}) {
		switch p := pa.(type) {
		case *PassiveAggressiveRegressor:
			p.loss = loss
		case *PassiveAggressiveClassifier:
			p.loss = loss
		}
	}
}

// PassiveAggressiveRegressor のメソッド実装

// Fit はバッチ学習でモデルを訓練
func (pa *PassiveAggressiveRegressor) Fit(X, y mat.Matrix) error {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	if !pa.warmStart || pa.coef_ == nil {
		pa.reset()
	}

	rows, cols := X.Dims()
	pa.nFeatures_ = cols

	if pa.coef_ == nil {
		pa.coef_ = make([]float64, cols)
		pa.avgCoef_ = make([]float64, cols)
	}

	// PassiveAggressive学習
	for iter := 0; iter < pa.maxIter; iter++ {
		for i := 0; i < rows; i++ {
			xi := mat.Row(nil, i, X)
			yi := y.At(i, 0)

			pa.updateWeights(xi, yi)
		}
		pa.nIter_++
	}

	if pa.nIter_ >= pa.maxIter {
		pa.converged_ = false
		errors.Warn(errors.NewConvergenceWarning("PassiveAggressiveRegressor", pa.nIter_, "Maximum number of iterations reached"))
	} else {
		pa.converged_ = true
	}

	pa.state.SetFitted()
	return nil
}

// PartialFit はミニバッチでモデルを逐次的に学習
func (pa *PassiveAggressiveRegressor) PartialFit(X, y mat.Matrix, classes []int) error {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	rows, cols := X.Dims()

	if pa.coef_ == nil {
		pa.nFeatures_ = cols
		pa.coef_ = make([]float64, cols)
		pa.avgCoef_ = make([]float64, cols)
	}

	if cols != pa.nFeatures_ {
		return errors.NewDimensionError("PartialFit", pa.nFeatures_, cols, 1)
	}

	// ミニバッチ処理
	for i := 0; i < rows; i++ {
		xi := mat.Row(nil, i, X)
		yi := y.At(i, 0)

		pa.updateWeights(xi, yi)
	}

	pa.state.SetFitted()
	return nil
}

// updateWeights は単一サンプルで重みを更新
func (pa *PassiveAggressiveRegressor) updateWeights(x []float64, y float64) {
	// 予測値計算
	pred := pa.intercept_
	for i, xi := range x {
		pred += pa.coef_[i] * xi
	}

	// 損失計算
	var loss, tau float64
	switch pa.loss {
	case "epsilon_insensitive":
		loss = math.Max(0, math.Abs(y-pred)-pa.epsilon)
		if loss > 0 {
			tau = loss / (dotProduct(x, x) + 1.0/(2.0*pa.C))
			if y < pred {
				tau = -tau
			}
		}
	case "squared_epsilon_insensitive":
		diff := math.Abs(y - pred)
		if diff > pa.epsilon {
			_ = loss // loss will be used for tracking, currently not returned
			tau = (diff - pa.epsilon) / (dotProduct(x, x) + 1.0/(2.0*pa.C))
			if y < pred {
				tau = -tau
			}
		}
	default:
		// デフォルトはepsilon_insensitive
		loss = math.Max(0, math.Abs(y-pred)-pa.epsilon)
		if loss > 0 {
			tau = loss / (dotProduct(x, x) + 1.0/(2.0*pa.C))
			if y < pred {
				tau = -tau
			}
		}
	}

	// 重み更新
	if tau != 0 {
		for i, xi := range x {
			pa.coef_[i] += tau * xi

			// 平均化PA
			if pa.averagePA {
				pa.avgCoef_[i] = (pa.avgCoef_[i]*float64(pa.t_) + pa.coef_[i]) / float64(pa.t_+1)
			}
		}

		// 切片更新
		if pa.fitIntercept {
			pa.intercept_ += tau
			if pa.averagePA {
				pa.avgIntercept_ = (pa.avgIntercept_*float64(pa.t_) + pa.intercept_) / float64(pa.t_+1)
			}
		}
	}

	pa.t_++
}

// Predict は入力データに対する予測を行う
func (pa *PassiveAggressiveRegressor) Predict(X mat.Matrix) (mat.Matrix, error) {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	if !pa.state.IsFitted() {
		return nil, errors.NewNotFittedError("PassiveAggressiveRegressor", "Predict")
	}

	rows, cols := X.Dims()
	if cols != pa.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", pa.nFeatures_, cols, 1)
	}

	predictions := mat.NewDense(rows, 1, nil)

	coef := pa.coef_
	intercept := pa.intercept_
	if pa.averagePA && pa.avgCoef_ != nil {
		coef = pa.avgCoef_
		intercept = pa.avgIntercept_
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

// PassiveAggressiveClassifier のメソッド実装

// Fit はバッチ学習でモデルを訓練
func (pa *PassiveAggressiveClassifier) Fit(X, y mat.Matrix) error {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	if !pa.warmStart || pa.coef_ == nil {
		pa.reset()
	}

	rows, cols := X.Dims()
	pa.nFeatures_ = cols

	// クラスを特定
	if pa.classes_ == nil {
		pa.extractClasses(y)
	}

	// 重みの初期化
	if pa.coef_ == nil {
		pa.initializeWeights()
	}

	// PassiveAggressive学習
	for iter := 0; iter < pa.maxIter; iter++ {
		for i := 0; i < rows; i++ {
			xi := mat.Row(nil, i, X)
			yi := int(y.At(i, 0))

			pa.updateWeights(xi, yi)
		}
		pa.nIter_++
	}

	if pa.nIter_ >= pa.maxIter {
		pa.converged_ = false
		errors.Warn(errors.NewConvergenceWarning("PassiveAggressiveClassifier", pa.nIter_, "Maximum number of iterations reached"))
	} else {
		pa.converged_ = true
	}

	pa.state.SetFitted()
	return nil
}

// PartialFit はミニバッチでモデルを逐次的に学習
func (pa *PassiveAggressiveClassifier) PartialFit(X, y mat.Matrix, classes []int) error {
	pa.mu.Lock()
	defer pa.mu.Unlock()

	rows, cols := X.Dims()

	// 初回呼び出し時の初期化
	if pa.coef_ == nil {
		pa.nFeatures_ = cols

		if classes != nil {
			pa.classes_ = make([]int, len(classes))
			copy(pa.classes_, classes)
			pa.nClasses_ = len(classes)
		} else {
			pa.extractClasses(y)
		}

		pa.initializeWeights()
	}

	if cols != pa.nFeatures_ {
		return errors.NewDimensionError("PartialFit", pa.nFeatures_, cols, 1)
	}

	// ミニバッチ処理
	for i := 0; i < rows; i++ {
		xi := mat.Row(nil, i, X)
		yi := int(y.At(i, 0))

		pa.updateWeights(xi, yi)
	}

	pa.state.SetFitted()
	return nil
}

// updateWeights は単一サンプルで重みを更新
func (pa *PassiveAggressiveClassifier) updateWeights(x []float64, y int) {
	// クラスインデックスを取得
	classIdx := pa.getClassIndex(y)
	if classIdx == -1 {
		return // 未知のクラス
	}

	// 各クラスについて処理
	for c := 0; c < pa.nClasses_; c++ {
		// スコア計算
		score := pa.intercept_[c]
		for i, xi := range x {
			score += pa.coef_[c][i] * xi
		}

		target := -1.0
		if c == classIdx {
			target = 1.0
		}

		var loss, tau float64

		// 損失計算とτ計算
		switch pa.loss {
		case "hinge":
			margin := target * score
			if margin < 1 {
				loss = 1 - margin
				tau = loss / (dotProduct(x, x) + 1.0/(2.0*pa.C))
				tau = tau * target
			}
		case "squared_hinge":
			margin := target * score
			if margin < 1 {
				diff := 1 - margin
				_ = loss // loss will be set in the common code below
				tau = diff / (dotProduct(x, x) + 1.0/(2.0*pa.C))
				tau = tau * target
			}
		default:
			// デフォルトはhinge
			margin := target * score
			if margin < 1 {
				loss = 1 - margin
				tau = loss / (dotProduct(x, x) + 1.0/(2.0*pa.C))
				tau = tau * target
			}
		}

		// 重み更新
		if tau != 0 {
			for i, xi := range x {
				pa.coef_[c][i] += tau * xi

				// 平均化PA
				if pa.averagePA {
					pa.avgCoef_[c][i] = (pa.avgCoef_[c][i]*float64(pa.t_) + pa.coef_[c][i]) / float64(pa.t_+1)
				}
			}

			// 切片更新
			if pa.fitIntercept {
				pa.intercept_[c] += tau
				if pa.averagePA {
					pa.avgIntercept_[c] = (pa.avgIntercept_[c]*float64(pa.t_) + pa.intercept_[c]) / float64(pa.t_+1)
				}
			}
		}
	}

	pa.t_++
}

// Predict は入力データに対する予測を行う
func (pa *PassiveAggressiveClassifier) Predict(X mat.Matrix) (mat.Matrix, error) {
	pa.mu.RLock()
	defer pa.mu.RUnlock()

	if !pa.state.IsFitted() {
		return nil, errors.NewNotFittedError("PassiveAggressiveClassifier", "Predict")
	}

	rows, cols := X.Dims()
	if cols != pa.nFeatures_ {
		return nil, errors.NewDimensionError("Predict", pa.nFeatures_, cols, 1)
	}

	predictions := mat.NewDense(rows, 1, nil)

	coef := pa.coef_
	intercept := pa.intercept_
	if pa.averagePA && pa.avgCoef_ != nil {
		coef = pa.avgCoef_
		intercept = pa.avgIntercept_
	}

	for i := 0; i < rows; i++ {
		maxScore := math.Inf(-1)
		predictedClass := pa.classes_[0]

		// 各クラスのスコアを計算
		for c := 0; c < pa.nClasses_; c++ {
			score := intercept[c]
			for j := 0; j < cols; j++ {
				score += X.At(i, j) * coef[c][j]
			}

			if score > maxScore {
				maxScore = score
				predictedClass = pa.classes_[c]
			}
		}

		predictions.Set(i, 0, float64(predictedClass))
	}

	return predictions, nil
}

// ストリーミング学習メソッド（共通）

// FitStream はデータストリームからモデルを学習
func (pa *PassiveAggressiveRegressor) FitStream(ctx context.Context, dataChan <-chan *model.Batch) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case batch, ok := <-dataChan:
			if !ok {
				return nil
			}
			if err := pa.PartialFit(batch.X, batch.Y, nil); err != nil {
				return err
			}
		}
	}
}

func (pa *PassiveAggressiveClassifier) FitStream(ctx context.Context, dataChan <-chan *model.Batch) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case batch, ok := <-dataChan:
			if !ok {
				return nil
			}
			if err := pa.PartialFit(batch.X, batch.Y, nil); err != nil {
				return err
			}
		}
	}
}

// インターフェース実装メソッド

// NIterations は実行された学習イテレーション数を返す
func (pa *PassiveAggressiveRegressor) NIterations() int {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.nIter_
}

func (pa *PassiveAggressiveClassifier) NIterations() int {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.nIter_
}

// IsWarmStart はウォームスタートが有効かどうかを返す
func (pa *PassiveAggressiveRegressor) IsWarmStart() bool {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.warmStart
}

func (pa *PassiveAggressiveClassifier) IsWarmStart() bool {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	return pa.warmStart
}

// SetWarmStart はウォームスタートの有効/無効を設定
func (pa *PassiveAggressiveRegressor) SetWarmStart(warmStart bool) {
	pa.mu.Lock()
	defer pa.mu.Unlock()
	pa.warmStart = warmStart
}

func (pa *PassiveAggressiveClassifier) SetWarmStart(warmStart bool) {
	pa.mu.Lock()
	defer pa.mu.Unlock()
	pa.warmStart = warmStart
}

// 内部ヘルパーメソッド

// extractClasses はデータからクラスを抽出
func (pa *PassiveAggressiveClassifier) extractClasses(y mat.Matrix) {
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

	pa.classes_ = classes
	pa.nClasses_ = len(classes)
}

// initializeWeights は重みを初期化
func (pa *PassiveAggressiveClassifier) initializeWeights() {
	pa.coef_ = make([][]float64, pa.nClasses_)
	pa.intercept_ = make([]float64, pa.nClasses_)
	pa.avgCoef_ = make([][]float64, pa.nClasses_)
	pa.avgIntercept_ = make([]float64, pa.nClasses_)

	for c := 0; c < pa.nClasses_; c++ {
		pa.coef_[c] = make([]float64, pa.nFeatures_)
		pa.avgCoef_[c] = make([]float64, pa.nFeatures_)
	}
}

// getClassIndex はクラス値からインデックスを取得
func (pa *PassiveAggressiveClassifier) getClassIndex(class int) int {
	for i, c := range pa.classes_ {
		if c == class {
			return i
		}
	}
	return -1
}

// reset は内部状態をリセット
func (pa *PassiveAggressiveRegressor) reset() {
	pa.coef_ = nil
	pa.intercept_ = 0
	pa.avgCoef_ = nil
	pa.avgIntercept_ = 0
	pa.nIter_ = 0
	pa.t_ = 0
	pa.state.Reset() // Reset state manager
}

func (pa *PassiveAggressiveClassifier) reset() {
	pa.coef_ = nil
	pa.intercept_ = nil
	pa.avgCoef_ = nil
	pa.avgIntercept_ = nil
	pa.classes_ = nil
	pa.nClasses_ = 0
	pa.nIter_ = 0
	pa.t_ = 0
	pa.state.Reset() // Reset state manager
}

// IsFitted returns whether the regressor has been fitted
func (pa *PassiveAggressiveRegressor) IsFitted() bool {
	return pa.state.IsFitted()
}

// IsFitted returns whether the classifier has been fitted
func (pa *PassiveAggressiveClassifier) IsFitted() bool {
	return pa.state.IsFitted()
}

// GetParams returns the hyperparameters for regressor
func (pa *PassiveAggressiveRegressor) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"C":             pa.C,
		"fit_intercept": pa.fitIntercept,
		"max_iter":      pa.maxIter,
		"tol":           pa.tol,
		"shuffle":       pa.shuffle,
		"verbose":       pa.verbose,
		"random_state":  pa.randomState,
		"warm_start":    pa.warmStart,
		"average":       pa.averagePA,
		"loss":          pa.loss,
		"epsilon":       pa.epsilon,
		"fitted":        pa.state.IsFitted(),
	}
}

// GetParams returns the hyperparameters for classifier
func (pa *PassiveAggressiveClassifier) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"C":             pa.C,
		"fit_intercept": pa.fitIntercept,
		"max_iter":      pa.maxIter,
		"tol":           pa.tol,
		"shuffle":       pa.shuffle,
		"verbose":       pa.verbose,
		"random_state":  pa.randomState,
		"warm_start":    pa.warmStart,
		"average":       pa.averagePA,
		"loss":          pa.loss,
		"class_weight":  pa.classWeight,
		"fitted":        pa.state.IsFitted(),
	}
}

// 補助関数
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
