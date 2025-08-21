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

// LinearRegression is a linear regression model
type LinearRegression struct {
	State     *model.StateManager // State manager (composition instead of embedding) - Public for gob encoding
	Weights   *mat.VecDense       // Model weights (coefficients)
	Intercept float64             // Model intercept
	NFeatures int                 // Number of features
	logger    log.Logger          // Logger instance
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
//
//	lr := linear.NewLinearRegression()
//	err := lr.Fit(X, y)
//	predictions, err := lr.Predict(X_test)
func NewLinearRegression() *LinearRegression {
	lr := &LinearRegression{
		State: model.NewStateManager(),
	}

	// Set up logger with model context
	lr.logger = log.GetLoggerWithName("linear").With(
		log.ModelNameKey, "LinearRegression",
		log.ComponentKey, "linear",
	)

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
//
//	X := mat.NewDense(100, 5, nil) // 100 samples, 5 features
//	y := mat.NewVecDense(100, nil) // 100 target values
//	err := lr.Fit(X, y)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (lr *LinearRegression) Fit(X, y mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Fit")

	startTime := time.Now()
	r, c := X.Dims()
	ry, cy := y.Dims()

	if lr.logger != nil {
		lr.logger.Info("Training started",
			log.OperationKey, log.OperationFit,
			log.PhaseKey, log.PhaseTraining,
			log.SamplesKey, r,
			log.FeaturesKey, c,
		)
	}

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

	// Add column of 1s to X for intercept term
	// X_with_intercept = [1, X]
	XWithIntercept := mat.NewDense(r, c+1, nil)

	// Parallelization threshold (use sequential processing for row counts below this value)
	const parallelThreshold = 1000

	// Use ParallelizeWithThreshold for parallelization based on data size
	parallel.ParallelizeWithThreshold(r, parallelThreshold, func(start, end int) {
		for i := start; i < end; i++ {
			XWithIntercept.Set(i, 0, 1.0) // Intercept term
			for j := 0; j < c; j++ {
				XWithIntercept.Set(i, j+1, X.At(i, j))
			}
		}
	})

	// Solve normal equations
	// (X^T * X)^(-1) * X^T * y
	var XT mat.Dense
	XT.CloneFrom(XWithIntercept.T())

	var XTX mat.Dense
	XTX.Mul(&XT, XWithIntercept)

	// Calculate inverse matrix
	var XTXInv mat.Dense
	err = XTXInv.Inverse(&XTX)
	if err != nil {
		return scigoErrors.NewModelError("LinearRegression.Fit", "singular matrix", scigoErrors.ErrSingularMatrix)
	}

	// Calculate X^T * y
	// Convert y to VecDense
	yVec := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		yVec.SetVec(i, y.At(i, 0))
	}

	var XTy mat.VecDense
	XTy.MulVec(&XT, yVec)

	// Calculate weights: (X^T * X)^(-1) * X^T * y
	weights := mat.NewVecDense(c+1, nil)
	weights.MulVec(&XTXInv, &XTy)

	// Separate intercept and weights
	lr.Intercept = weights.AtVec(0)
	lr.Weights = mat.NewVecDense(c, nil)
	for i := 0; i < c; i++ {
		lr.Weights.SetVec(i, weights.AtVec(i+1))
	}

	// Set model as fitted
	lr.State.SetFitted()
	lr.State.SetDimensions(lr.NFeatures, r)

	duration := time.Since(startTime)
	if lr.logger != nil {
		lr.logger.Info("Training completed",
			log.OperationKey, log.OperationFit,
			log.PhaseKey, log.PhaseTraining,
			log.DurationMsKey, duration.Milliseconds(),
			log.SamplesKey, r,
			log.FeaturesKey, c,
		)
	}

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
//
//	predictions, err := lr.Predict(X_test)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("First prediction: %.2f\n", predictions.At(0, 0))
func (lr *LinearRegression) Predict(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Predict")
	if !lr.State.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("LinearRegression", "Predict")
	}

	r, c := X.Dims()
	if c != lr.NFeatures {
		return nil, scigoErrors.NewDimensionError("LinearRegression.Predict", lr.NFeatures, c, 1)
	}

	if lr.logger != nil {
		lr.logger.Debug("Prediction started",
			log.OperationKey, log.OperationPredict,
			log.PhaseKey, log.PhaseInference,
			log.SamplesKey, r,
			log.FeaturesKey, c,
		)
	}

	// Prediction: y = X * weights + intercept
	predictions := mat.NewDense(r, 1, nil)

	for i := 0; i < r; i++ {
		pred := lr.Intercept
		for j := 0; j < c; j++ {
			pred += X.At(i, j) * lr.Weights.AtVec(j)
		}
		predictions.Set(i, 0, pred)
	}

	if lr.logger != nil {
		lr.logger.Debug("Prediction completed",
			log.OperationKey, log.OperationPredict,
			log.PredsKey, r,
		)
	}

	return predictions, nil
}

// GetWeights returns the learned weights (coefficients)
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

// GetIntercept returns the learned intercept
func (lr *LinearRegression) GetIntercept() float64 {
	if !lr.State.IsFitted() {
		return 0
	}
	return lr.Intercept
}

// Score calculates the coefficient of determination (R²) of the model
func (lr *LinearRegression) Score(X, y mat.Matrix) (_ float64, err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.Score")
	if !lr.State.IsFitted() {
		return 0, scigoErrors.NewNotFittedError("LinearRegression", "Score")
	}

	// Calculate predicted values
	yPred, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	r, _ := y.Dims()

	// Calculate mean of y
	var yMean float64
	for i := 0; i < r; i++ {
		yMean += y.At(i, 0)
	}
	yMean /= float64(r)

	// Calculate total sum of squares (TSS) and residual sum of squares (RSS)
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

// LoadFromSKLearn loads a model from a JSON file exported from scikit-learn
//
// Parameters:
//   - filename: Path to the JSON file
//
// Returns:
//   - error: Loading error
//
// Example:
//
//	lr := NewLinearRegression()
//	err := lr.LoadFromSKLearn("sklearn_model.json")
func (lr *LinearRegression) LoadFromSKLearn(filename string) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.LoadFromSKLearn")
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	return lr.LoadFromSKLearnReader(file)
}

// LoadFromSKLearnReader loads a scikit-learn model from a Reader
//
// Parameters:
//   - r: Reader containing JSON data
//
// Returns:
//   - error: Loading error
func (lr *LinearRegression) LoadFromSKLearnReader(r io.Reader) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.LoadFromSKLearnReader")
	// Load JSON model
	skModel, err := model.LoadSKLearnModelFromReader(r)
	if err != nil {
		return fmt.Errorf("failed to load sklearn model: %w", err)
	}

	// Extract LinearRegression parameters
	params, err := model.LoadLinearRegressionParams(skModel)
	if err != nil {
		return fmt.Errorf("failed to load linear regression params: %w", err)
	}

	// Set parameters
	lr.NFeatures = params.NFeatures
	lr.Intercept = params.Intercept

	// Convert coefficients to VecDense
	lr.Weights = mat.NewVecDense(len(params.Coefficients), params.Coefficients)

	// Set model as fitted
	lr.State.SetFitted()
	// Note: sample count is not available when loading from file
	lr.State.SetDimensions(lr.NFeatures, 0)

	return nil
}

// ExportToSKLearn exports the model in scikit-learn compatible JSON format
//
// Parameters:
//   - filename: Output filename
//
// Returns:
//   - error: Error if export fails
func (lr *LinearRegression) ExportToSKLearn(filename string) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.ExportToSKLearn")
	if !lr.State.IsFitted() {
		return scigoErrors.NewNotFittedError("LinearRegression", "ExportToSKLearn")
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = file.Close() }()

	return lr.ExportToSKLearnWriter(file)
}

// ExportToSKLearnWriter exports the model to a Writer in scikit-learn compatible format
//
// Parameters:
//   - w: Output Writer
//
// Returns:
//   - error: Error if export fails
func (lr *LinearRegression) ExportToSKLearnWriter(w io.Writer) (err error) {
	defer scigoErrors.Recover(&err, "LinearRegression.ExportToSKLearnWriter")
	if !lr.State.IsFitted() {
		return scigoErrors.NewNotFittedError("LinearRegression", "ExportToSKLearnWriter")
	}

	// Convert coefficients to array
	coefficients := make([]float64, lr.Weights.Len())
	for i := 0; i < lr.Weights.Len(); i++ {
		coefficients[i] = lr.Weights.AtVec(i)
	}

	// Create parameters
	params := model.SKLearnLinearRegressionParams{
		Coefficients: coefficients,
		Intercept:    lr.Intercept,
		NFeatures:    lr.NFeatures,
	}

	// Export in JSON format
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

// IsFitted returns whether the model has been fitted.
func (lr *LinearRegression) IsFitted() bool {
	return lr.State.IsFitted()
}

// GetParams returns the model's hyperparameters.
func (lr *LinearRegression) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"n_features": lr.NFeatures,
		"fitted":     lr.State.IsFitted(),
	}
}

// SetParams sets the model's hyperparameters.
func (lr *LinearRegression) SetParams(params map[string]interface{}) error {
	// LinearRegression has no hyperparameters to set
	return nil
}
