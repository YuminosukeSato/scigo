package linear

import (
	"fmt"

	"github.com/YuminosukeSato/GoML/core"
	"github.com/YuminosukeSato/GoML/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// LinearRegression は線形回帰モデル
type LinearRegression struct {
	core.BaseEstimator
	weights   *mat.VecDense // 重み（係数）
	intercept float64       // 切片
	nFeatures int           // 特徴量の数
}

// NewLinearRegression は新しい線形回帰モデルを作成する
func NewLinearRegression() *LinearRegression {
	return &LinearRegression{}
}

// Fit はモデルを訓練データで学習させる
// 正規方程式 w = (X^T * X)^(-1) * X^T * y を使用
func (lr *LinearRegression) Fit(X, y mat.Matrix) error {
	// 入力の検証
	r, c := X.Dims()
	ry, cy := y.Dims()

	if r == 0 || c == 0 {
		return errors.NewModelError("LinearRegression.Fit", "empty data", errors.ErrEmptyData)
	}

	if ry != r {
		return errors.NewDimensionError("LinearRegression.Fit", []int{r, 1}, []int{ry, cy})
	}

	if cy != 1 {
		return errors.NewValueError("LinearRegression.Fit", "y", cy, "y must be a column vector")
	}

	lr.nFeatures = c

	// 切片項のために X に 1 の列を追加
	// X_with_intercept = [1, X]
	XWithIntercept := mat.NewDense(r, c+1, nil)
	for i := 0; i < r; i++ {
		XWithIntercept.Set(i, 0, 1.0) // 切片項
		for j := 0; j < c; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	// 正規方程式を解く
	// (X^T * X)^(-1) * X^T * y
	var XT mat.Dense
	XT.CloneFrom(XWithIntercept.T())

	var XTX mat.Dense
	XTX.Mul(&XT, XWithIntercept)

	// 逆行列を計算
	var XTXInv mat.Dense
	err := XTXInv.Inverse(&XTX)
	if err != nil {
		return errors.NewModelError("LinearRegression.Fit", "singular matrix", errors.ErrSingularMatrix)
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
	lr.intercept = weights.AtVec(0)
	lr.weights = mat.NewVecDense(c, nil)
	for i := 0; i < c; i++ {
		lr.weights.SetVec(i, weights.AtVec(i+1))
	}

	lr.SetFitted()
	return nil
}

// Predict は入力データに対する予測を行う
func (lr *LinearRegression) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lr.IsFitted() {
		return nil, errors.NewNotFittedError("LinearRegression")
	}

	r, c := X.Dims()
	if c != lr.nFeatures {
		return nil, errors.NewDimensionError("LinearRegression.Predict",
			[]int{-1, lr.nFeatures}, []int{r, c})
	}

	// 予測: y = X * weights + intercept
	predictions := mat.NewDense(r, 1, nil)

	for i := 0; i < r; i++ {
		pred := lr.intercept
		for j := 0; j < c; j++ {
			pred += X.At(i, j) * lr.weights.AtVec(j)
		}
		predictions.Set(i, 0, pred)
	}

	return predictions, nil
}

// Weights は学習された重み（係数）を返す
func (lr *LinearRegression) Weights() []float64 {
	if !lr.IsFitted() || lr.weights == nil {
		return nil
	}

	weights := make([]float64, lr.weights.Len())
	for i := 0; i < lr.weights.Len(); i++ {
		weights[i] = lr.weights.AtVec(i)
	}
	return weights
}

// Intercept は学習された切片を返す
func (lr *LinearRegression) Intercept() float64 {
	if !lr.IsFitted() {
		return 0
	}
	return lr.intercept
}

// Score はモデルの決定係数（R²）を計算する
func (lr *LinearRegression) Score(X, y mat.Matrix) (float64, error) {
	if !lr.IsFitted() {
		return 0, errors.NewNotFittedError("LinearRegression")
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
