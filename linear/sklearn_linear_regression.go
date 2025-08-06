package linear

import (
	"runtime"

	"github.com/YuminosukeSato/GoML/core/parallel"
	"github.com/YuminosukeSato/GoML/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// SKLinearRegression implements scikit-learn compatible LinearRegression
type SKLinearRegression struct {
	// Configuration parameters
	fitIntercept bool    // Whether to calculate the intercept
	copyX        bool    // Whether to copy X
	tol          float64 // Tolerance for optimization (for sparse matrices)
	nJobs        int     // Number of parallel jobs
	positive     bool    // Whether to constrain coefficients to be positive

	// Model attributes (scikit-learn compatible naming)
	Coef_         *mat.Dense // Coefficients (n_features,) or (n_targets, n_features)
	Intercept_    *mat.Dense // Intercept term(s)
	Rank_         int        // Rank of matrix X
	Singular_     []float64  // Singular values of X
	NFeaturesIn_  int        // Number of features seen during fit
	NTargets_     int        // Number of targets

	// Internal state
	fitted bool
}

// NewSKLinearRegression creates a new scikit-learn compatible LinearRegression
func NewSKLinearRegression(opts ...Option) *SKLinearRegression {
	lr := &SKLinearRegression{
		fitIntercept: true,
		copyX:        true,
		tol:          1e-6,
		nJobs:        1,
		positive:     false,
	}

	// Apply options
	for _, opt := range opts {
		opt(lr)
	}

	// Set nJobs based on system if -1
	if lr.nJobs == -1 {
		lr.nJobs = runtime.NumCPU()
	} else if lr.nJobs <= 0 {
		lr.nJobs = 1
	}

	return lr
}

// Fit trains the linear regression model
func (lr *SKLinearRegression) Fit(X, y mat.Matrix) error {
	// Input validation
	nSamples, nFeatures := X.Dims()
	yRows, yCols := y.Dims()

	if nSamples == 0 || nFeatures == 0 {
		return errors.NewModelError("SKLinearRegression.Fit", "empty data", errors.ErrEmptyData)
	}

	if yRows != nSamples {
		return errors.NewDimensionError("SKLinearRegression.Fit", 
			[]int{nSamples}, []int{yRows})
	}

	// Store dimensions
	lr.NFeaturesIn_ = nFeatures
	lr.NTargets_ = yCols

	// Copy X if requested
	var XWork mat.Matrix
	if lr.copyX {
		XCopy := mat.NewDense(nSamples, nFeatures, nil)
		XCopy.Copy(X)
		XWork = XCopy
	} else {
		XWork = X
	}

	// Handle intercept
	var XFit mat.Matrix
	if lr.fitIntercept {
		// Add intercept column
		XWithIntercept := mat.NewDense(nSamples, nFeatures+1, nil)
		
		// Use parallel processing for large datasets
		const parallelThreshold = 1000
		parallel.ParallelizeWithThreshold(nSamples, parallelThreshold, func(start, end int) {
			for i := start; i < end; i++ {
				XWithIntercept.Set(i, 0, 1.0) // Intercept column
				for j := 0; j < nFeatures; j++ {
					XWithIntercept.Set(i, j+1, XWork.At(i, j))
				}
			}
		})
		
		XFit = XWithIntercept
	} else {
		XFit = XWork
	}

	// Solve the linear regression problem
	if lr.positive {
		// Non-negative least squares
		return lr.fitNNLS(XFit, y)
	}
	
	// Standard least squares using normal equation
	return lr.fitNormalEquation(XFit, y)
}

// fitNormalEquation solves using the normal equation: (X^T X)^-1 X^T y
func (lr *SKLinearRegression) fitNormalEquation(X, y mat.Matrix) error {
	_, cols := X.Dims()
	_, yCols := y.Dims()

	// Compute X^T
	var XT mat.Dense
	XT.CloneFrom(X.T())

	// Compute X^T X
	var XTX mat.Dense
	XTX.Mul(&XT, X)

	// Compute SVD for rank and singular values
	var svd mat.SVD
	ok := svd.Factorize(&XTX, mat.SVDFull)
	if !ok {
		return errors.NewModelError("SKLinearRegression.Fit", 
			"SVD factorization failed", nil)
	}

	// Store rank and singular values
	lr.Singular_ = svd.Values(nil)
	lr.Rank_ = 0
	for _, s := range lr.Singular_ {
		if s > lr.tol {
			lr.Rank_++
		}
	}

	// Regular inverse (handle singular matrices with error)
	var XTXInv mat.Dense
	err := XTXInv.Inverse(&XTX)
	if err != nil {
		// For singular matrices, use a simple regularization approach
		// Add small value to diagonal (Ridge-like regularization)
		r, c := XTX.Dims()
		for i := 0; i < r && i < c; i++ {
			XTX.Set(i, i, XTX.At(i, i) + 1e-10)
		}
		
		// Try inverse again
		err = XTXInv.Inverse(&XTX)
		if err != nil {
			return errors.NewModelError("SKLinearRegression.Fit", 
				"matrix inversion failed even with regularization", err)
		}
	}

	// Compute X^T * y
	var XTy mat.Dense
	XTy.Mul(&XT, y)

	// Compute coefficients: (X^T X)^-1 * X^T * y
	coef := mat.NewDense(cols, yCols, nil)
	coef.Mul(&XTXInv, &XTy)
	
	lr.extractCoefficients(coef)

	lr.fitted = true
	return nil
}

// fitNNLS fits using Non-Negative Least Squares
// Simple implementation using projected gradient descent
func (lr *SKLinearRegression) fitNNLS(X, y mat.Matrix) error {
	rows, cols := X.Dims()
	_, yCols := y.Dims()

	// Initialize coefficients
	coef := mat.NewDense(cols, yCols, nil)

	// Solve NNLS for each target using simple iterative method
	for target := 0; target < yCols; target++ {
		// Extract target column
		yTarget := mat.NewVecDense(rows, nil)
		for i := 0; i < rows; i++ {
			yTarget.SetVec(i, y.At(i, target))
		}

		// Initialize weights to positive values
		weights := make([]float64, cols)
		for i := range weights {
			weights[i] = 0.1
		}

		// Simple gradient descent with projection
		learningRate := 0.01
		maxIter := 1000
		
		for iter := 0; iter < maxIter; iter++ {
			// Compute predictions
			predictions := mat.NewVecDense(rows, nil)
			for i := 0; i < rows; i++ {
				pred := 0.0
				for j := 0; j < cols; j++ {
					pred += X.At(i, j) * weights[j]
				}
				predictions.SetVec(i, pred)
			}

			// Compute gradients
			gradients := make([]float64, cols)
			for j := 0; j < cols; j++ {
				grad := 0.0
				for i := 0; i < rows; i++ {
					error := predictions.AtVec(i) - yTarget.AtVec(i)
					grad += 2 * error * X.At(i, j)
				}
				gradients[j] = grad / float64(rows)
			}

			// Update weights
			for j := 0; j < cols; j++ {
				weights[j] -= learningRate * gradients[j]
				// Project to non-negative (enforce constraint)
				if weights[j] < 0 {
					weights[j] = 0
				}
			}
		}

		// Store coefficients
		for i := 0; i < cols; i++ {
			coef.Set(i, target, weights[i])
		}
	}

	lr.extractCoefficients(coef)
	lr.fitted = true
	return nil
}

// extractCoefficients extracts intercept and coefficients from the solution
func (lr *SKLinearRegression) extractCoefficients(coef *mat.Dense) {
	rows, cols := coef.Dims()
	
	if lr.fitIntercept {
		// Extract intercept (first row)
		lr.Intercept_ = mat.NewDense(1, cols, nil)
		for j := 0; j < cols; j++ {
			lr.Intercept_.Set(0, j, coef.At(0, j))
		}
		
		// Extract coefficients (remaining rows)
		lr.Coef_ = mat.NewDense(rows-1, cols, nil)
		for i := 1; i < rows; i++ {
			for j := 0; j < cols; j++ {
				lr.Coef_.Set(i-1, j, coef.At(i, j))
			}
		}
	} else {
		// No intercept
		lr.Intercept_ = mat.NewDense(1, cols, nil) // Zero intercept
		lr.Coef_ = mat.DenseCopyOf(coef)
	}
}

// Predict makes predictions for input data
func (lr *SKLinearRegression) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !lr.fitted {
		return nil, errors.NewNotFittedError("SKLinearRegression")
	}

	nSamples, nFeatures := X.Dims()
	if nFeatures != lr.NFeaturesIn_ {
		return nil, errors.NewDimensionError("SKLinearRegression.Predict",
			[]int{-1, lr.NFeaturesIn_}, []int{nSamples, nFeatures})
	}

	// Create predictions matrix
	predictions := mat.NewDense(nSamples, lr.NTargets_, nil)

	// Compute predictions in parallel for large datasets
	const parallelThreshold = 1000
	parallel.ParallelizeWithThreshold(nSamples, parallelThreshold, func(start, end int) {
		for i := start; i < end; i++ {
			for t := 0; t < lr.NTargets_; t++ {
				pred := lr.Intercept_.At(0, t)
				for j := 0; j < nFeatures; j++ {
					pred += X.At(i, j) * lr.Coef_.At(j, t)
				}
				predictions.Set(i, t, pred)
			}
		}
	})

	return predictions, nil
}

// Score returns the coefficient of determination R² of the prediction
func (lr *SKLinearRegression) Score(X, y mat.Matrix) (float64, error) {
	// Get predictions
	yPred, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	nSamples, nTargets := y.Dims()
	
	// Compute mean of y
	yMean := make([]float64, nTargets)
	for j := 0; j < nTargets; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += y.At(i, j)
		}
		yMean[j] = sum / float64(nSamples)
	}

	// Compute R² for each target and average
	totalR2 := 0.0
	for j := 0; j < nTargets; j++ {
		ssRes := 0.0 // Residual sum of squares
		ssTot := 0.0 // Total sum of squares
		
		for i := 0; i < nSamples; i++ {
			yTrue := y.At(i, j)
			yPredVal := yPred.At(i, j)
			
			ssRes += (yTrue - yPredVal) * (yTrue - yPredVal)
			ssTot += (yTrue - yMean[j]) * (yTrue - yMean[j])
		}
		
		if ssTot == 0 {
			return 0, errors.Newf("Score: no variance in target %d", j)
		}
		
		totalR2 += 1 - (ssRes / ssTot)
	}

	return totalR2 / float64(nTargets), nil
}

// GetParams returns the parameters of the model
func (lr *SKLinearRegression) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"fit_intercept": lr.fitIntercept,
		"copy_X":        lr.copyX,
		"tol":           lr.tol,
		"n_jobs":        lr.nJobs,
		"positive":      lr.positive,
	}
}

// SetParams sets the parameters of the model
func (lr *SKLinearRegression) SetParams(params map[string]interface{}) error {
	if val, ok := params["fit_intercept"].(bool); ok {
		lr.fitIntercept = val
	}
	if val, ok := params["copy_X"].(bool); ok {
		lr.copyX = val
	}
	if val, ok := params["tol"].(float64); ok {
		lr.tol = val
	}
	if val, ok := params["n_jobs"].(int); ok {
		lr.nJobs = val
	}
	if val, ok := params["positive"].(bool); ok {
		lr.positive = val
	}
	
	return nil
}

// IsFitted returns whether the model has been fitted
func (lr *SKLinearRegression) IsFitted() bool {
	return lr.fitted
}