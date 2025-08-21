// Package metrics provides evaluation metrics for machine learning models.
//
// This package implements standard evaluation metrics for regression and classification tasks:
//
// Regression Metrics:
//   - MSE: Mean Squared Error for measuring prediction accuracy
//   - RMSE: Root Mean Squared Error (square root of MSE)
//   - MAE: Mean Absolute Error for robust error measurement
//   - R²: R-squared coefficient of determination
//   - MAPE: Mean Absolute Percentage Error
//   - Explained Variance Score: Proportion of variance explained by the model
//
// All metrics support both vector and matrix inputs using gonum/mat for efficient
// computation. The functions are optimized for performance and numerical stability.
//
// Example usage:
//
//	// Calculate regression metrics
//	mse := metrics.MSE(yTrue, yPred)
//	rmse := metrics.RMSE(yTrue, yPred)
//	r2 := metrics.R2Score(yTrue, yPred)
//
//	// Matrix inputs are also supported
//	mseMatrix := metrics.MSEMatrix(yTrueMatrix, yPredMatrix)
//
// The metrics integrate seamlessly with model evaluation pipelines and support
// cross-validation workflows for comprehensive model assessment.
package metrics

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
)

// MSE calculates the Mean Squared Error between true and predicted values.
//
// MSE measures the average squared differences between predictions and actual
// values. Lower values indicate better model performance. MSE is sensitive to
// outliers due to the squared differences.
//
// Parameters:
//   - yTrue: True target values as a vector
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: MSE value (non-negative)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input vectors are empty
//   - ErrDimensionMismatch: if yTrue and yPred have different lengths
//
// Example:
//
//	mse, err := metrics.MSE(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("MSE: %.4f\n", mse)
func MSE(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MSE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MSE", n, yPred.Len(), 0)
	}

	// MSE = (1/n) * Σ(yTrue - yPred)²
	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += diff * diff
	}

	return sum / float64(n), nil
}

// MSEMatrix calculates MSE for matrix inputs (column vectors).
//
// This function provides MSE computation for matrix inputs, specifically
// designed for column vectors (n×1 matrices). It converts the matrices to
// vectors and delegates to the MSE function.
//
// Parameters:
//   - yTrue: True target values as column matrix (n×1)
//   - yPred: Predicted values as column matrix (n×1)
//
// Returns:
//   - float64: MSE value (non-negative)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input matrices are empty
//   - ErrDimensionMismatch: if matrices have different dimensions
//   - ErrValueError: if matrices are not column vectors (n×1)
//
// Example:
//
//	mse, err := metrics.MSEMatrix(yTrueMatrix, yPredMatrix)
func MSEMatrix(yTrue, yPred mat.Matrix) (float64, error) {
	// Input validation
	rTrue, cTrue := yTrue.Dims()
	rPred, cPred := yPred.Dims()

	if rTrue == 0 || cTrue == 0 {
		return 0, scigoErrors.NewValueError("MSEMatrix", "empty matrix")
	}

	if rTrue != rPred || cTrue != cPred {
		return 0, scigoErrors.NewDimensionError("MSEMatrix", rTrue, rPred, 0)
	}

	if cTrue != 1 {
		return 0, scigoErrors.NewValueError("MSEMatrix", "must be a column vector (n×1 matrix)")
	}

	// Convert to VecDense and calculate MSE
	yTrueVec := mat.NewVecDense(rTrue, nil)
	yPredVec := mat.NewVecDense(rPred, nil)

	for i := 0; i < rTrue; i++ {
		yTrueVec.SetVec(i, yTrue.At(i, 0))
		yPredVec.SetVec(i, yPred.At(i, 0))
	}

	return MSE(yTrueVec, yPredVec)
}

// RMSE calculates the Root Mean Squared Error between true and predicted values.
//
// RMSE is the square root of MSE, providing error measurement in the same units
// as the target variable. RMSE is useful for interpretation as it's in the same
// scale as the original data.
//
// Parameters:
//   - yTrue: True target values as a vector
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: RMSE value (non-negative)
//   - error: nil if successful, otherwise error from MSE computation
//
// Example:
//
//	rmse, err := metrics.RMSE(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("RMSE: %.4f\n", rmse)
func RMSE(yTrue, yPred *mat.VecDense) (float64, error) {
	mse, err := MSE(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(mse), nil
}

// MAE calculates the Mean Absolute Error between true and predicted values.
//
// MAE measures the average absolute differences between predictions and actual
// values. MAE is more robust to outliers compared to MSE as it doesn't square
// the differences. Lower values indicate better model performance.
//
// Parameters:
//   - yTrue: True target values as a vector
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: MAE value (non-negative)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input vectors are empty
//   - ErrDimensionMismatch: if yTrue and yPred have different lengths
//
// Example:
//
//	mae, err := metrics.MAE(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("MAE: %.4f\n", mae)
func MAE(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MAE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MAE", n, yPred.Len(), 0)
	}

	// MAE = (1/n) * Σ|yTrue - yPred|
	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += math.Abs(diff)
	}

	return sum / float64(n), nil
}

// R2Score calculates the coefficient of determination (R²) score.
//
// R² represents the proportion of variance in the target variable that is
// predictable from the input features. Values range from negative infinity to 1,
// where 1 indicates perfect predictions, 0 indicates predictions no better than
// the mean, and negative values indicate worse than mean predictions.
//
// Parameters:
//   - yTrue: True target values as a vector
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: R² score (can be negative, best possible score is 1.0)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input vectors are empty
//   - ErrDimensionMismatch: if yTrue and yPred have different lengths
//   - ErrValueError: if all yTrue values are identical (no variance)
//
// Example:
//
//	r2, err := metrics.R2Score(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("R² Score: %.4f\n", r2)
func R2Score(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("R2Score", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("R2Score", n, yPred.Len(), 0)
	}

	// Calculate mean of yTrue
	var yMean float64
	for i := 0; i < n; i++ {
		yMean += yTrue.AtVec(i)
	}
	yMean /= float64(n)

	// Calculate Total Sum of Squares (TSS) and Residual Sum of Squares (RSS)
	var tss, rss float64
	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		yPredVal := yPred.AtVec(i)

		tss += (yTrueVal - yMean) * (yTrueVal - yMean)
		rss += (yTrueVal - yPredVal) * (yTrueVal - yPredVal)
	}

	// When TSS is zero (all yTrue values are identical)
	if tss == 0 {
		return 0, fmt.Errorf("R2Score: total sum of squares is zero (no variance in yTrue)")
	}

	// R² = 1 - RSS/TSS
	return 1 - rss/tss, nil
}

// MAPE calculates the Mean Absolute Percentage Error.
//
// MAPE measures prediction accuracy as a percentage, making it scale-independent
// and easy to interpret. Values are expressed as percentages where lower values
// indicate better performance. MAPE is undefined when yTrue contains zero values.
//
// Parameters:
//   - yTrue: True target values as a vector (must not contain zeros)
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: MAPE value as percentage (non-negative)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input vectors are empty
//   - ErrDimensionMismatch: if yTrue and yPred have different lengths
//   - ErrValueError: if all yTrue values are zero
//
// Example:
//
//	mape, err := metrics.MAPE(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("MAPE: %.2f%%\n", mape)
func MAPE(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MAPE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MAPE", n, yPred.Len(), 0)
	}

	// MAPE = (100/n) * Σ|yTrue - yPred|/|yTrue|
	var sum float64
	validCount := 0

	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		if yTrueVal != 0 { // Avoid division by zero
			diff := math.Abs(yTrueVal - yPred.AtVec(i))
			sum += diff / math.Abs(yTrueVal)
			validCount++
		}
	}

	if validCount == 0 {
		return 0, fmt.Errorf("MAPE: all yTrue values are zero")
	}

	return (sum / float64(validCount)) * 100, nil
}

// ExplainedVarianceScore calculates the explained variance regression score.
//
// This metric measures the proportion of the variance in the target variable
// that is explained by the model. Unlike R², it does not account for systematic
// offset in predictions. The best possible score is 1.0, lower values indicate
// less explained variance.
//
// Parameters:
//   - yTrue: True target values as a vector
//   - yPred: Predicted values as a vector
//
// Returns:
//   - float64: Explained variance score (best possible score is 1.0)
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if input vectors are empty
//   - ErrDimensionMismatch: if yTrue and yPred have different lengths
//   - ErrValueError: if yTrue has no variance (all values identical)
//
// Example:
//
//	evs, err := metrics.ExplainedVarianceScore(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Explained Variance Score: %.4f\n", evs)
func ExplainedVarianceScore(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("ExplainedVarianceScore", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("ExplainedVarianceScore", n, yPred.Len(), 0)
	}

	// Calculate means
	var yTrueMean, yPredMean, diffMean float64
	for i := 0; i < n; i++ {
		yTrueMean += yTrue.AtVec(i)
		yPredMean += yPred.AtVec(i)
		diffMean += (yTrue.AtVec(i) - yPred.AtVec(i))
	}
	yTrueMean /= float64(n)
	_ = yPredMean // yPredMean is not used after this
	diffMean /= float64(n)

	// Calculate variances
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
		return 0, fmt.Errorf("ExplainedVarianceScore: no variance in yTrue")
	}

	// Explained variance score = 1 - Var(yTrue - yPred) / Var(yTrue)
	return 1 - varDiff/varYTrue, nil
}
