package metrics

import (
	"math"

	"github.com/YuminosukeSato/GoML/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// MSE は平均二乗誤差（Mean Squared Error）を計算する
func MSE(yTrue, yPred *mat.VecDense) (float64, error) {
	// 入力検証
	n := yTrue.Len()
	if n == 0 {
		return 0, errors.NewValueError("MSE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, errors.NewDimensionError("MSE", n, yPred.Len(), 0)
	}

	// MSE = (1/n) * Σ(yTrue - yPred)²
	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += diff * diff
	}

	return sum / float64(n), nil
}

// MSEMatrix は行列形式の入力に対してMSEを計算する
func MSEMatrix(yTrue, yPred mat.Matrix) (float64, error) {
	// 入力検証
	rTrue, cTrue := yTrue.Dims()
	rPred, cPred := yPred.Dims()

	if rTrue == 0 || cTrue == 0 {
		return 0, errors.NewValueError("MSEMatrix", "empty matrix")
	}

	if rTrue != rPred || cTrue != cPred {
		return 0, errors.NewDimensionError("MSEMatrix", rTrue, rPred, 0)
	}

	if cTrue != 1 {
		return 0, errors.NewValueError("MSEMatrix", "must be a column vector (n×1 matrix)")
	}

	// VecDenseに変換してMSEを計算
	yTrueVec := mat.NewVecDense(rTrue, nil)
	yPredVec := mat.NewVecDense(rPred, nil)

	for i := 0; i < rTrue; i++ {
		yTrueVec.SetVec(i, yTrue.At(i, 0))
		yPredVec.SetVec(i, yPred.At(i, 0))
	}

	return MSE(yTrueVec, yPredVec)
}

// RMSE は平方根平均二乗誤差（Root Mean Squared Error）を計算する
func RMSE(yTrue, yPred *mat.VecDense) (float64, error) {
	mse, err := MSE(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(mse), nil
}

// MAE は平均絶対誤差（Mean Absolute Error）を計算する
func MAE(yTrue, yPred *mat.VecDense) (float64, error) {
	// 入力検証
	n := yTrue.Len()
	if n == 0 {
		return 0, errors.NewValueError("MAE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, errors.NewDimensionError("MAE", n, yPred.Len(), 0)
	}

	// MAE = (1/n) * Σ|yTrue - yPred|
	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += math.Abs(diff)
	}

	return sum / float64(n), nil
}

// R2Score は決定係数（R²）を計算する
func R2Score(yTrue, yPred *mat.VecDense) (float64, error) {
	// 入力検証
	n := yTrue.Len()
	if n == 0 {
		return 0, errors.NewValueError("R2Score", "empty vector")
	}

	if yPred.Len() != n {
		return 0, errors.NewDimensionError("R2Score", n, yPred.Len(), 0)
	}

	// yTrueの平均を計算
	var yMean float64
	for i := 0; i < n; i++ {
		yMean += yTrue.AtVec(i)
	}
	yMean /= float64(n)

	// 全変動（TSS）と残差変動（RSS）を計算
	var tss, rss float64
	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		yPredVal := yPred.AtVec(i)

		tss += (yTrueVal - yMean) * (yTrueVal - yMean)
		rss += (yTrueVal - yPredVal) * (yTrueVal - yPredVal)
	}

	// 全変動が0の場合（すべてのyTrueが同じ値）
	if tss == 0 {
		return 0, errors.Newf("R2Score: total sum of squares is zero (no variance in yTrue)")
	}

	// R² = 1 - RSS/TSS
	return 1 - rss/tss, nil
}

// MeanAbsolutePercentageError (MAPE) は平均絶対パーセンテージ誤差を計算する
func MAPE(yTrue, yPred *mat.VecDense) (float64, error) {
	// 入力検証
	n := yTrue.Len()
	if n == 0 {
		return 0, errors.NewValueError("MAPE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, errors.NewDimensionError("MAPE", n, yPred.Len(), 0)
	}

	// MAPE = (100/n) * Σ|yTrue - yPred|/|yTrue|
	var sum float64
	validCount := 0

	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		if yTrueVal != 0 { // ゼロ除算を避ける
			diff := math.Abs(yTrueVal - yPred.AtVec(i))
			sum += diff / math.Abs(yTrueVal)
			validCount++
		}
	}

	if validCount == 0 {
		return 0, errors.Newf("MAPE: all yTrue values are zero")
	}

	return (sum / float64(validCount)) * 100, nil
}

// ExplainedVarianceScore は説明分散スコアを計算する
func ExplainedVarianceScore(yTrue, yPred *mat.VecDense) (float64, error) {
	// 入力検証
	n := yTrue.Len()
	if n == 0 {
		return 0, errors.NewValueError("ExplainedVarianceScore", "empty vector")
	}

	if yPred.Len() != n {
		return 0, errors.NewDimensionError("ExplainedVarianceScore", n, yPred.Len(), 0)
	}

	// 平均を計算
	var yTrueMean, yPredMean, diffMean float64
	for i := 0; i < n; i++ {
		yTrueMean += yTrue.AtVec(i)
		yPredMean += yPred.AtVec(i)
		diffMean += (yTrue.AtVec(i) - yPred.AtVec(i))
	}
	yTrueMean /= float64(n)
	yPredMean /= float64(n)
	diffMean /= float64(n)

	// 分散を計算
	var varYTrue, varDiff float64
	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		diff := yTrueVal - yPred.AtVec(i)

		varYTrue += (yTrueVal - yTrueMean) * (yTrueVal - yTrueMean)
		varDiff += (diff - diffMean) * (diff - diffMean)
	}
	varYTrue /= float64(n)
	varDiff /= float64(n)

	if varYTrue == 0 {
		return 0, errors.Newf("ExplainedVarianceScore: no variance in yTrue")
	}

	// 説明分散スコア = 1 - Var(yTrue - yPred) / Var(yTrue)
	return 1 - varDiff/varYTrue, nil
}
