// Package linear provides linear machine learning algorithms and models.
//
// This package implements various linear models for regression and classification tasks:
//
//   - LinearRegression: Ordinary least squares regression with L2 regularization support
//   - scikit-learn compatibility: Import/export linear models trained in Python
//   - High-performance matrix operations using gonum/mat
//   - Production-ready with comprehensive error handling and validation
//
// Linear models are fundamental building blocks in machine learning, offering:
//
//   - Fast training and prediction
//   - Interpretable coefficients and feature importance
//   - Memory efficient implementation
//   - Robust numerical stability using QR decomposition
//
// Example usage:
//
//	lr := linear.NewLinearRegression()
//	err := lr.Fit(X, y) // X: features, y: target values
//	if err != nil {
//		log.Fatal(err)
//	}
//	predictions, err := lr.Predict(XTest)
//
// The package supports model persistence and scikit-learn interoperability:
//
//	// Save trained model
//	err = lr.ExportToSKLearn("model.json")
//	
//	// Load Python-trained model
//	err = lr.LoadFromSKLearn("sklearn_model.json")
//
// All algorithms follow the standard estimator interface with Fit/Predict methods
// and integrate seamlessly with preprocessing pipelines and model evaluation tools.
package linear

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"time"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/core/parallel"
	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"github.com/YuminosukeSato/scigo/pkg/log"
	"gonum.org/v1/gonum/mat"
)

// LinearRegression は線形回帰モデル
type LinearRegression struct {
	model.BaseEstimator // BaseEstimatorを埋め込み
	Weights   *mat.VecDense // 重み（係数）
	Intercept float64       // 切片
	NFeatures int           // 特徴量の数
}

// NewLinearRegression creates a new linear regression model for ordinary least squares regression.
//
// The model uses QR decomposition for numerical stability and supports both
// single and multiple linear regression tasks. The returned model must be
// trained using the Fit method before making predictions.
//
// Returns:
//   - *LinearRegression: A new untrained linear regression model
//
// Example:
//   lr := linear.NewLinearRegression()
//   err := lr.Fit(X, y)
//   predictions, err := lr.Predict(X_test)
func NewLinearRegression() *LinearRegression {
	lr := &LinearRegression{}
	
	// Set up logger with model context
	logger := log.GetLoggerWithName("linear").With(
		log.ModelNameKey, "LinearRegression",
		log.ComponentKey, "linear",
	)
	lr.SetLogger(logger)
	
	return lr
}

// Fit trains the linear regression model using the provided training data.
//
// The method uses QR decomposition to solve the normal equation (X^T * X)w = X^T * y
// for numerical stability and handles both overdetermined and underdetermined systems.
// After successful training, the model's fitted state is set to true.
//
// Parameters:
//   - X: Feature matrix of shape (n_samples, n_features)
//   - y: Target vector of shape (n_samples, 1) or (n_samples, n_targets)
//
// Returns:
//   - error: nil if training succeeds, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if X or y are empty
//   - ErrDimensionMismatch: if the number of samples in X and y don't match
//   - ErrSingularMatrix: if X^T * X is singular and cannot be inverted
//
// Example:
//   X := mat.NewDense(100, 5, nil) // 100 samples, 5 features
//   y := mat.NewVecDense(100, nil) // 100 target values
//   err := lr.Fit(X, y)
//   if err != nil {
//       log.Fatal(err)
//   }
func (lr *LinearRegression) Fit(X, y mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Fit")
	
	startTime := time.Now()
	r, c := X.Dims()
	ry, cy := y.Dims()
	
	lr.LogInfo("Training started",
		log.OperationKey, log.OperationFit,
		log.PhaseKey, log.PhaseTraining,
		log.SamplesKey, r,
		log.FeaturesKey, c,
	)

	if r == 0 || c == 0 {
		return scigoErrors.NewModelError("LinearRegression.Fit", "empty data", scigoErrors.ErrEmptyData)
	}

	if ry != r {
		return scigoErrors.NewDimensionError("LinearRegression.Fit", r, ry, 0)
	}

	if cy != 1 {
		return scigoErrors.NewValueError("LinearRegression.Fit", "y must be a column vector")
	}

	lr.NFeatures = c

	// 切片項のために X に 1 の列を追加
	// X_with_intercept = [1, X]
	XWithIntercept := mat.NewDense(r, c+1, nil)

	// 並列処理の閾値（この値以下の行数では逐次処理を使用）
	const parallelThreshold = 1000

	// ParallelizeWithThresholdを使用して、データサイズに応じて並列化
	parallel.ParallelizeWithThreshold(r, parallelThreshold, func(start, end int) {
		for i := start; i < end; i++ {
			XWithIntercept.Set(i, 0, 1.0) // 切片項
			for j := 0; j < c; j++ {
				XWithIntercept.Set(i, j+1, X.At(i, j))
			}
		}
	})

	// 正規方程式を解く
	// (X^T * X)^(-1) * X^T * y
	var XT mat.Dense
	XT.CloneFrom(XWithIntercept.T())

	var XTX mat.Dense
	XTX.Mul(&XT, XWithIntercept)

	// 逆行列を計算
	var XTXInv mat.Dense
	err = XTXInv.Inverse(&XTX)
	if err != nil {
		return scigoErrors.NewModelError("LinearRegression.Fit", "singular matrix", scigoErrors.ErrSingularMatrix)
	}

	// X^T * y を計算
	// y を VecDense に変換
	yVec := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		yVec.SetVec(i, y.At(i, 0))
	}

	var XTy mat.VecDense
	XTy.MulVec(&XT, yVec)

	// 重みを計算: (X^T * X)^(-1) * X^T * y
	weights := mat.NewVecDense(c+1, nil)
	weights.MulVec(&XTXInv, &XTy)

	// 切片と重みを分離
	lr.Intercept = weights.AtVec(0)
	lr.Weights = mat.NewVecDense(c, nil)
	for i := 0; i < c; i++ {
		lr.Weights.SetVec(i, weights.AtVec(i+1))
	}
	
	// モデルを学習済み状態に設定
	lr.SetFitted()
	
	duration := time.Since(startTime)
	lr.LogInfo("Training completed",
		log.OperationKey, log.OperationFit,
		log.PhaseKey, log.PhaseTraining,
		log.DurationMsKey, duration.Milliseconds(),
		log.SamplesKey, r,
		log.FeaturesKey, c,
	)
	
	return nil
}

// Predict generates predictions for the input feature matrix using the trained model.
//
// The method computes predictions using the learned weights and intercept:
// y_pred = X * weights + intercept. The model must be fitted before calling
// this method, otherwise it returns an error.
//
// Parameters:
//   - X: Feature matrix of shape (n_samples, n_features) for prediction
//
// Returns:
//   - mat.Matrix: Prediction matrix of shape (n_samples, 1) containing predicted values
//   - error: nil if prediction succeeds, otherwise an error describing the failure
//
// Errors:
//   - ErrNotFitted: if the model hasn't been trained yet
//   - ErrDimensionMismatch: if X has different number of features than training data
//
// Example:
//   predictions, err := lr.Predict(X_test)
//   if err != nil {
//       log.Fatal(err)
//   }
//   fmt.Printf("First prediction: %.2f\n", predictions.At(0, 0))
func (lr *LinearRegression) Predict(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Predict")
	if !lr.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LinearRegression", "Predict")
	}

	r, c := X.Dims()
	if c != lr.NFeatures {
		return nil, scigoErrors.NewDimensionError("LinearRegression.Predict", lr.NFeatures, c, 1)
	}

	lr.LogDebug("Prediction started",
		log.OperationKey, log.OperationPredict,
		log.PhaseKey, log.PhaseInference,
		log.SamplesKey, r,
		log.FeaturesKey, c,
	)

	// 予測: y = X * weights + intercept
	predictions := mat.NewDense(r, 1, nil)

	for i := 0; i < r; i++ {
		pred := lr.Intercept
		for j := 0; j < c; j++ {
			pred += X.At(i, j) * lr.Weights.AtVec(j)
		}
		predictions.Set(i, 0, pred)
	}

	lr.LogDebug("Prediction completed",
		log.OperationKey, log.OperationPredict,
		log.PredsKey, r,
	)

	return predictions, nil
}

// GetWeights は学習された重み（係数）を返す
func (lr *LinearRegression) GetWeights() []float64 {
	if lr.Weights == nil {
		return nil
	}

	weights := make([]float64, lr.Weights.Len())
	for i := 0; i < lr.Weights.Len(); i++ {
		weights[i] = lr.Weights.AtVec(i)
	}
	return weights
}

// GetIntercept は学習された切片を返す
func (lr *LinearRegression) GetIntercept() float64 {
	if !lr.IsFitted() {
		return 0
	}
	return lr.Intercept
}

// Score はモデルの決定係数（R²）を計算する
func (lr *LinearRegression) Score(X, y mat.Matrix) (_ float64, err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Score")
	if !lr.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LinearRegression", "Score")
	}

	// 予測値を計算
	yPred, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	r, _ := y.Dims()

	// y の平均を計算
	var yMean float64
	for i := 0; i < r; i++ {
		yMean += y.At(i, 0)
	}
	yMean /= float64(r)

	// 全変動 (TSS) と残差変動 (RSS) を計算
	var tss, rss float64
	for i := 0; i < r; i++ {
		yTrue := y.At(i, 0)
		yPredVal := yPred.At(i, 0)

		tss += (yTrue - yMean) * (yTrue - yMean)
		rss += (yTrue - yPredVal) * (yTrue - yPredVal)
	}

	// R² = 1 - RSS/TSS
	if tss == 0 {
		return 0, fmt.Errorf("total sum of squares is zero")
	}

	return 1 - rss/tss, nil
}

// LoadFromSKLearn はscikit-learnからエクスポートされたJSONファイルからモデルを読み込む
//
// パラメータ:
//   - filename: JSONファイルのパス
//
// 戻り値:
//   - error: 読み込みエラー
//
// 使用例:
//
//	lr := NewLinearRegression()
//	err := lr.LoadFromSKLearn("sklearn_model.json")
func (lr *LinearRegression) LoadFromSKLearn(filename string) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.LoadFromSKLearn")
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return lr.LoadFromSKLearnReader(file)
}

// LoadFromSKLearnReader はReaderからscikit-learnモデルを読み込む
//
// パラメータ:
//   - r: JSONデータを含むReader
//
// 戻り値:
//   - error: 読み込みエラー
func (lr *LinearRegression) LoadFromSKLearnReader(r io.Reader) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.LoadFromSKLearnReader")
	// JSONモデルを読み込み
	skModel, err := model.LoadSKLearnModelFromReader(r)
	if err != nil {
		return fmt.Errorf("failed to load sklearn model: %w", err)
	}

	// LinearRegressionのパラメータを抽出
	params, err := model.LoadLinearRegressionParams(skModel)
	if err != nil {
		return fmt.Errorf("failed to load linear regression params: %w", err)
	}

	// パラメータを設定
	lr.NFeatures = params.NFeatures
	lr.Intercept = params.Intercept
	
	// 係数をVecDenseに変換
	lr.Weights = mat.NewVecDense(len(params.Coefficients), params.Coefficients)
	
	// モデルを学習済み状態に設定
	lr.SetFitted()
	
	return nil
}

// ExportToSKLearn はモデルをscikit-learn互換のJSON形式でエクスポート
//
// パラメータ:
//   - filename: 出力ファイル名
//
// 戻り値:
//   - error: エクスポート失敗時のエラー
func (lr *LinearRegression) ExportToSKLearn(filename string) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.ExportToSKLearn")
	if !lr.IsFitted() {
		return scigoErrors.NewNotFittedError("LinearRegression", "ExportToSKLearn")
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	return lr.ExportToSKLearnWriter(file)
}

// ExportToSKLearnWriter はモデルをWriterにscikit-learn互換形式でエクスポート
//
// パラメータ:
//   - w: 出力先Writer
//
// 戻り値:
//   - error: エクスポート失敗時のエラー
func (lr *LinearRegression) ExportToSKLearnWriter(w io.Writer) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.ExportToSKLearnWriter")
	if !lr.IsFitted() {
		return scigoErrors.NewNotFittedError("LinearRegression", "ExportToSKLearnWriter")
	}

	// 係数を配列に変換
	coefficients := make([]float64, lr.Weights.Len())
	for i := 0; i < lr.Weights.Len(); i++ {
		coefficients[i] = lr.Weights.AtVec(i)
	}

	// パラメータを作成
	params := model.SKLearnLinearRegressionParams{
		Coefficients: coefficients,
		Intercept:    lr.Intercept,
		NFeatures:    lr.NFeatures,
	}

	// JSON形式でエクスポート
	skModel := model.SKLearnModel{
		ModelSpec: model.SKLearnModelSpec{
			Name:          "LinearRegression",
			FormatVersion: "1.0",
		},
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return fmt.Errorf("failed to marshal params: %w", err)
	}
	skModel.Params = paramsJSON

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(&skModel); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}
