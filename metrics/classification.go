package metrics

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
)

// AUC calculates the Area Under the ROC Curve for binary classification.
//
// The AUC represents the probability that a classifier will rank a randomly
// chosen positive instance higher than a randomly chosen negative instance.
// AUC values range from 0 to 1, where:
//   - 0.5 indicates random guessing
//   - 1.0 indicates perfect classification
//   - 0.0 indicates perfectly wrong classification
//
// Parameters:
//   - yTrue: Ground truth binary labels (0 or 1)
//   - yPred: Predicted probabilities or decision scores
//
// Returns:
//   - The AUC score
//   - An error if inputs are invalid
//
// Example:
//
//	yTrue := mat.NewVecDense(4, []float64{0, 0, 1, 1})
//	yPred := mat.NewVecDense(4, []float64{0.1, 0.4, 0.35, 0.8})
//	auc, err := AUC(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("AUC: %f\n", auc) // Output: AUC: 0.75
func AUC(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"AUC",
			"input vectors cannot be nil",
		)
	}

	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError(
			"AUC",
			"input vectors cannot be empty",
		)
	}

	if n != yPred.Len() {
		return 0, scigoErrors.NewDimensionError(
			"AUC",
			n,
			yPred.Len(),
			0,
		)
	}

	// Check if yTrue contains only binary values
	for i := 0; i < n; i++ {
		val := yTrue.AtVec(i)
		if val != 0.0 && val != 1.0 {
			return 0, scigoErrors.NewValidationError(
				"yTrue",
				fmt.Sprintf("must contain only binary values (0 or 1), found %f at index %d", val, i),
				val,
			)
		}
	}

	// Create pairs of (score, label) and sort by score
	type pair struct {
		score float64
		label float64
	}
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{
			score: yPred.AtVec(i),
			label: yTrue.AtVec(i),
		}
	}

	// Sort pairs by score in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// Calculate ROC curve points using the trapezoid rule
	// We'll compute TPR and FPR at each threshold
	var tprs []float64
	var fprs []float64

	// Count total positives and negatives
	totalPos := 0.0
	totalNeg := 0.0
	for i := 0; i < n; i++ {
		if yTrue.AtVec(i) == 1.0 {
			totalPos++
		} else {
			totalNeg++
		}
	}

	// Handle edge cases
	if totalPos == 0 || totalNeg == 0 {
		// If all samples belong to one class, AUC is undefined
		// Return 0.5 as a reasonable default (random classifier)
		return 0.5, nil
	}

	// Add initial point (0, 0)
	tprs = append(tprs, 0)
	fprs = append(fprs, 0)

	// Calculate TPR and FPR for each threshold
	tp := 0.0
	fp := 0.0
	prevScore := pairs[0].score + 1 // Initialize to a value higher than any score

	for _, p := range pairs {
		// If score changed, add a point to ROC curve
		if p.score != prevScore {
			tprs = append(tprs, tp/totalPos)
			fprs = append(fprs, fp/totalNeg)
			prevScore = p.score
		}

		// Update counts
		if p.label == 1.0 {
			tp++
		} else {
			fp++
		}
	}

	// Add final point (1, 1)
	tprs = append(tprs, 1)
	fprs = append(fprs, 1)

	// Calculate AUC using trapezoidal rule
	auc := 0.0
	for i := 1; i < len(fprs); i++ {
		// Trapezoid area = (x2 - x1) * (y1 + y2) / 2
		width := fprs[i] - fprs[i-1]
		height := (tprs[i] + tprs[i-1]) / 2
		auc += width * height
	}

	return auc, nil
}

// AUCMatrix is a convenience wrapper for AUC that accepts matrix inputs.
//
// This function extracts the first column from each input matrix and
// calls AUC with the resulting vectors.
//
// Parameters:
//   - yTrue: Matrix where the first column contains ground truth binary labels
//   - yPred: Matrix where the first column contains predicted probabilities
//
// Returns:
//   - The AUC score
//   - An error if inputs are invalid
func AUCMatrix(yTrue, yPred mat.Matrix) (float64, error) {
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"AUCMatrix",
			"input matrices cannot be nil",
		)
	}

	r1, _ := yTrue.Dims()
	r2, _ := yPred.Dims()

	if r1 == 0 || r2 == 0 {
		return 0, scigoErrors.NewValueError(
			"AUCMatrix",
			"input matrices cannot be empty",
		)
	}

	// Extract first column as vectors
	yTrueVec := mat.NewVecDense(r1, nil)
	yPredVec := mat.NewVecDense(r2, nil)

	for i := 0; i < r1; i++ {
		yTrueVec.SetVec(i, yTrue.At(i, 0))
	}
	for i := 0; i < r2; i++ {
		yPredVec.SetVec(i, yPred.At(i, 0))
	}

	return AUC(yTrueVec, yPredVec)
}

// BinaryLogLoss calculates the binary cross-entropy loss for binary classification.
//
// Binary log loss, also known as logistic loss or cross-entropy loss,
// measures the performance of a classification model whose output is a
// probability value between 0 and 1.
//
// Parameters:
//   - yTrue: Ground truth binary labels (0 or 1)
//   - yPred: Predicted probabilities (between 0 and 1)
//
// Returns:
//   - The average binary log loss
//   - An error if inputs are invalid
//
// Example:
//
//	yTrue := mat.NewVecDense(4, []float64{0, 0, 1, 1})
//	yPred := mat.NewVecDense(4, []float64{0.1, 0.2, 0.8, 0.9})
//	loss, err := BinaryLogLoss(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Log Loss: %f\n", loss)
func BinaryLogLoss(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"BinaryLogLoss",
			"input vectors cannot be nil",
		)
	}

	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError(
			"BinaryLogLoss",
			"input vectors cannot be empty",
		)
	}

	if n != yPred.Len() {
		return 0, scigoErrors.NewDimensionError(
			"BinaryLogLoss",
			n,
			yPred.Len(),
			0,
		)
	}

	// Calculate log loss
	const epsilon = 1e-15 // Small value to avoid log(0)
	loss := 0.0

	for i := 0; i < n; i++ {
		y := yTrue.AtVec(i)
		p := yPred.AtVec(i)

		// Check if yTrue is binary
		if y != 0.0 && y != 1.0 {
			return 0, scigoErrors.NewValidationError(
				"yTrue",
				fmt.Sprintf("must contain only binary values (0 or 1), found %f at index %d", y, i),
				y,
			)
		}

		// Clip prediction to avoid log(0)
		if p < epsilon {
			p = epsilon
		} else if p > 1-epsilon {
			p = 1 - epsilon
		}

		// Binary cross-entropy formula
		if y == 1.0 {
			loss -= logSafe(p)
		} else {
			loss -= logSafe(1 - p)
		}
	}

	return loss / float64(n), nil
}

// logSafe computes natural logarithm with safety checks
func logSafe(x float64) float64 {
	if x <= 0 {
		return -1e10 // Return a large negative number instead of -Inf
	}
	return math.Log(x)
}

// ClassificationError calculates the classification error rate.
//
// The error rate is the fraction of incorrect predictions.
//
// Parameters:
//   - yTrue: Ground truth labels (integers)
//   - yPred: Predicted labels (integers)
//
// Returns:
//   - The error rate (between 0 and 1)
//   - An error if inputs are invalid
//
// Example:
//
//	yTrue := mat.NewVecDense(5, []float64{0, 1, 2, 1, 0})
//	yPred := mat.NewVecDense(5, []float64{0, 1, 1, 1, 0})
//	errorRate, err := ClassificationError(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Error Rate: %f\n", errorRate) // Output: Error Rate: 0.2
func ClassificationError(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"ClassificationError",
			"input vectors cannot be nil",
		)
	}

	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError(
			"ClassificationError",
			"input vectors cannot be empty",
		)
	}

	if n != yPred.Len() {
		return 0, scigoErrors.NewDimensionError(
			"ClassificationError",
			n,
			yPred.Len(),
			0,
		)
	}

	// Count misclassifications
	errors := 0
	for i := 0; i < n; i++ {
		if yTrue.AtVec(i) != yPred.AtVec(i) {
			errors++
		}
	}

	return float64(errors) / float64(n), nil
}

// Accuracy calculates the classification accuracy.
//
// Accuracy is the fraction of correct predictions.
//
// Parameters:
//   - yTrue: Ground truth labels
//   - yPred: Predicted labels
//
// Returns:
//   - The accuracy (between 0 and 1)
//   - An error if inputs are invalid
func Accuracy(yTrue, yPred *mat.VecDense) (float64, error) {
	errorRate, err := ClassificationError(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return 1.0 - errorRate, nil
}
