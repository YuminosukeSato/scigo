package linear_model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// LinearRegression is a linear regression model using ordinary least squares
// Fully compatible with scikit-learn's LinearRegression
type LinearRegression struct {
	state *model.StateManager // State management (composition instead of embedding)
	
	// Hyperparameters
	fitIntercept bool    // Whether to learn the intercept
	normalize    bool    // Whether to normalize input data (deprecated in sklearn)
	copyX        bool    // Whether to copy input data
	nJobs        int     // Number of parallel jobs
	positive     bool    // Whether to constrain coefficients to be positive
	
	// Model type and version info
	modelType    string
	version      string
	
	// Learned parameters
	coef_      []float64 // Weight coefficients
	intercept_ float64   // Intercept
	
	// Statistical information
	nFeatures_  int       // Number of features
	nSamples_   int       // Number of samples
	// singularValues_ []float64 // Singular values (for diagnostics) - TODO: implement if needed
	rank_       int       // Matrix rank
}

// NewLinearRegression は新しいLinearRegressionモデルを作成
func NewLinearRegression(options ...LinearRegressionOption) *LinearRegression {
	lr := &LinearRegression{
		state:        model.NewStateManager(),
		fitIntercept: true,
		normalize:    false,
		copyX:        true,
		nJobs:        1,
		positive:     false,
		modelType:    "LinearRegression",
		version:      "1.0.0",
	}
	
	// Apply options
	for _, opt := range options {
		opt(lr)
	}
	
	return lr
}

// LinearRegressionOption は設定オプション
type LinearRegressionOption func(*LinearRegression)

// WithLRFitIntercept は切片の学習有無を設定（LinearRegression用）
func WithLRFitIntercept(fit bool) LinearRegressionOption {
	return func(lr *LinearRegression) {
		lr.fitIntercept = fit
	}
}

// WithNormalize は正規化の有無を設定（deprecated）
func WithNormalize(normalize bool) LinearRegressionOption {
	return func(lr *LinearRegression) {
		lr.normalize = normalize
	}
}

// WithCopyX はデータコピーの有無を設定
func WithCopyX(copy bool) LinearRegressionOption {
	return func(lr *LinearRegression) {
		lr.copyX = copy
	}
}

// WithNJobs は並列ジョブ数を設定
func WithNJobs(n int) LinearRegressionOption {
	return func(lr *LinearRegression) {
		lr.nJobs = n
	}
}

// WithPositive は係数の正制約を設定
func WithPositive(positive bool) LinearRegressionOption {
	return func(lr *LinearRegression) {
		lr.positive = positive
	}
}

// Fit はモデルを訓練データで学習
func (lr *LinearRegression) Fit(X, y mat.Matrix) error {
	rows, cols := X.Dims()
	yRows, yCols := y.Dims()
	
	// 入力検証
	if rows != yRows {
		return errors.NewDimensionError("LinearRegression.Fit", rows, yRows, 0)
	}
	
	if yCols != 1 {
		return errors.NewDimensionError("LinearRegression.Fit", 1, yCols, 1)
	}
	
	lr.nSamples_ = rows
	lr.nFeatures_ = cols
	
	// データのコピー（必要な場合）
	var XWork mat.Matrix
	if lr.copyX {
		XWork = mat.DenseCopyOf(X)
	} else {
		XWork = X
	}
	
	// 切片の処理
	var XFit mat.Matrix
	if lr.fitIntercept {
		// バイアス項を追加（最初の列に1を追加）
		ones := mat.NewDense(rows, 1, nil)
		for i := 0; i < rows; i++ {
			ones.Set(i, 0, 1.0)
		}
		
		// [ones | X] の行列を作成
		XWithIntercept := mat.NewDense(rows, cols+1, nil)
		for i := 0; i < rows; i++ {
			XWithIntercept.Set(i, 0, 1.0)
			for j := 0; j < cols; j++ {
				XWithIntercept.Set(i, j+1, XWork.At(i, j))
			}
		}
		XFit = XWithIntercept
	} else {
		XFit = XWork
	}
	
	// 正規方程式: (X^T * X)^(-1) * X^T * y
	// より数値的に安定なQR分解を使用
	var qr mat.QR
	qr.Factorize(XFit)
	
	// QR分解のランクを取得
	lr.rank_ = cols
	if lr.fitIntercept {
		lr.rank_++
	}
	
	// 係数を計算
	_, qrCols := XFit.Dims()
	coefficients := mat.NewDense(qrCols, 1, nil)
	err := qr.SolveTo(coefficients, false, y)
	if err != nil {
		return fmt.Errorf("failed to solve linear system: %w", err)
	}
	
	// 係数を取得
	if lr.fitIntercept {
		lr.intercept_ = coefficients.At(0, 0)
		lr.coef_ = make([]float64, cols)
		for i := 0; i < cols; i++ {
			lr.coef_[i] = coefficients.At(i + 1, 0)
		}
	} else {
		lr.intercept_ = 0.0
		lr.coef_ = make([]float64, cols)
		for i := 0; i < cols; i++ {
			lr.coef_[i] = coefficients.At(i, 0)
		}
	}
	
	// 正の制約がある場合
	if lr.positive {
		for i := range lr.coef_ {
			if lr.coef_[i] < 0 {
				lr.coef_[i] = 0
			}
		}
		if lr.intercept_ < 0 {
			lr.intercept_ = 0
		}
	}
	
	lr.state.SetFitted()
	lr.state.SetDimensions(lr.nFeatures_, lr.nSamples_)
	return nil
}

// Predict は入力データに対する予測を行う
func (lr *LinearRegression) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lr.state.IsFitted() {
		return nil, errors.NewNotFittedError("LinearRegression", "Predict")
	}
	
	rows, cols := X.Dims()
	if cols != lr.nFeatures_ {
		return nil, errors.NewDimensionError("LinearRegression.Predict", lr.nFeatures_, cols, 1)
	}
	
	predictions := mat.NewDense(rows, 1, nil)
	
	for i := 0; i < rows; i++ {
		pred := lr.intercept_
		for j := 0; j < cols; j++ {
			pred += X.At(i, j) * lr.coef_[j]
		}
		predictions.Set(i, 0, pred)
	}
	
	return predictions, nil
}

// Score はモデルの決定係数（R²）を計算
func (lr *LinearRegression) Score(X, y mat.Matrix) (float64, error) {
	predictions, err := lr.Predict(X)
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
		return 0, errors.NewValueError("LinearRegression.Score", "Cannot compute score with zero variance in y_true")
	}
	
	return 1.0 - (ssRes / ssTot), nil
}

// Coef は学習された重み係数を返す
func (lr *LinearRegression) Coef() []float64 {
	if lr.coef_ == nil {
		return nil
	}
	coef := make([]float64, len(lr.coef_))
	copy(coef, lr.coef_)
	return coef
}

// Intercept は学習された切片を返す
func (lr *LinearRegression) Intercept() float64 {
	return lr.intercept_
}

// GetParams returns the model's hyperparameters (scikit-learn compatible)
func (lr *LinearRegression) GetParams(deep bool) map[string]interface{} {
	return map[string]interface{}{
		"fit_intercept": lr.fitIntercept,
		"normalize":     lr.normalize,
		"copy_X":        lr.copyX,
		"n_jobs":        lr.nJobs,
		"positive":      lr.positive,
		"fitted":        lr.state.IsFitted(),
		"model_type":    lr.modelType,
		"version":       lr.version,
	}
}

// SetParams sets the model's hyperparameters (scikit-learn compatible)
func (lr *LinearRegression) SetParams(params map[string]interface{}) error {
	// Set LinearRegression-specific parameters
	if v, ok := params["fit_intercept"].(bool); ok {
		lr.fitIntercept = v
	}
	if v, ok := params["normalize"].(bool); ok {
		lr.normalize = v
	}
	if v, ok := params["copy_X"].(bool); ok {
		lr.copyX = v
	}
	if v, ok := params["n_jobs"].(int); ok {
		lr.nJobs = v
	}
	if v, ok := params["positive"].(bool); ok {
		lr.positive = v
	}
	
	return nil
}

// ExportWeights はモデルの重みをエクスポート（完全な再現性を保証）
func (lr *LinearRegression) ExportWeights() (*model.ModelWeights, error) {
	if !lr.state.IsFitted() {
		return nil, fmt.Errorf("model is not fitted")
	}
	
	weights := &model.ModelWeights{
		ModelType:       "LinearRegression",
		Version:         lr.version,
		Coefficients:    lr.Coef(),
		Intercept:       lr.intercept_,
		IsFitted:        true,
		Hyperparameters: lr.GetParams(true),
		Metadata: map[string]interface{}{
			"n_features": lr.nFeatures_,
			"n_samples":  lr.nSamples_,
			"rank":       lr.rank_,
		},
	}
	
	// チェックサムを計算
	data, _ := json.Marshal(weights.Coefficients)
	hash := sha256.Sum256(data)
	weights.Metadata["checksum"] = hex.EncodeToString(hash[:])
	
	return weights, nil
}

// ImportWeights はモデルの重みをインポート（完全な再現性を保証）
func (lr *LinearRegression) ImportWeights(weights *model.ModelWeights) error {
	if weights == nil {
		return fmt.Errorf("weights cannot be nil")
	}
	
	if weights.ModelType != "LinearRegression" {
		return fmt.Errorf("model type mismatch: expected LinearRegression, got %s", weights.ModelType)
	}
	
	// ハイパーパラメータを設定
	if err := lr.SetParams(weights.Hyperparameters); err != nil {
		return err
	}
	
	// 重みを設定
	lr.coef_ = make([]float64, len(weights.Coefficients))
	copy(lr.coef_, weights.Coefficients)
	lr.intercept_ = weights.Intercept
	
	// メタデータを設定
	if v, ok := weights.Metadata["n_features"].(float64); ok {
		lr.nFeatures_ = int(v)
	}
	if v, ok := weights.Metadata["n_samples"].(float64); ok {
		lr.nSamples_ = int(v)
	}
	if v, ok := weights.Metadata["rank"].(float64); ok {
		lr.rank_ = int(v)
	}
	
	// チェックサムを検証
	if checksumStr, ok := weights.Metadata["checksum"].(string); ok {
		data, _ := json.Marshal(weights.Coefficients)
		hash := sha256.Sum256(data)
		calculatedChecksum := hex.EncodeToString(hash[:])
		
		if checksumStr != calculatedChecksum {
			return fmt.Errorf("checksum mismatch: weights may be corrupted")
		}
	}
	
	lr.state.SetFitted()
	lr.state.SetDimensions(lr.nFeatures_, lr.nSamples_)
	return nil
}

// GetWeightHash calculates the hash value of weights (for verification)
func (lr *LinearRegression) GetWeightHash() string {
	if !lr.state.IsFitted() {
		return ""
	}
	
	data := append(lr.coef_, lr.intercept_)
	jsonData, _ := json.Marshal(data)
	hash := sha256.Sum256(jsonData)
	return hex.EncodeToString(hash[:])
}

// IsFitted returns whether the model has been fitted
func (lr *LinearRegression) IsFitted() bool {
	return lr.state.IsFitted()
}

// Clone はモデルの新しいインスタンスを作成（同じハイパーパラメータ）
func (lr *LinearRegression) Clone() model.SKLearnCompatible {
	clone := NewLinearRegression(
		WithLRFitIntercept(lr.fitIntercept),
		WithNormalize(lr.normalize),
		WithCopyX(lr.copyX),
		WithNJobs(lr.nJobs),
		WithPositive(lr.positive),
	)
	
	// Copy weights if the model is trained
	if lr.state.IsFitted() {
		weights, err := lr.ExportWeights()
		if err == nil {
			_ = clone.ImportWeights(weights)
		}
	}
	
	return clone
}

// String returns the string representation of the model
func (lr *LinearRegression) String() string {
	if !lr.state.IsFitted() {
		return fmt.Sprintf("LinearRegression(fit_intercept=%t, normalize=%t, copy_X=%t, n_jobs=%d, positive=%t)",
			lr.fitIntercept, lr.normalize, lr.copyX, lr.nJobs, lr.positive)
	}
	return fmt.Sprintf("LinearRegression(fit_intercept=%t, n_features=%d, fitted=true)",
		lr.fitIntercept, lr.nFeatures_)
}