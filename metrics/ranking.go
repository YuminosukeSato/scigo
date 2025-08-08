package metrics

import (
	"fmt"
	"math"
	"sort"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// NDCG calculates the Normalized Discounted Cumulative Gain for ranking evaluation.
//
// NDCG is a measure of ranking quality that takes into account both the relevance
// of items and their position in the ranking. Higher scores indicate better rankings.
// NDCG values range from 0 to 1, where 1 indicates perfect ranking.
//
// The formula is:
//   NDCG@k = DCG@k / IDCG@k
// where:
//   DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)
//   IDCG@k = DCG@k for the ideal ordering
//
// Parameters:
//   - yTrue: Ground truth relevance scores (non-negative values)
//   - yPred: Predicted scores for ranking
//   - k: Number of top results to consider (use -1 for all)
//
// Returns:
//   - The NDCG score
//   - An error if inputs are invalid
//
// Example:
//
//	yTrue := mat.NewVecDense(5, []float64{3, 2, 3, 0, 1})
//	yPred := mat.NewVecDense(5, []float64{2.5, 0.5, 2, 0, 1})
//	ndcg, err := NDCG(yTrue, yPred, 3)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("NDCG@3: %f\n", ndcg)
func NDCG(yTrue, yPred *mat.VecDense, k int) (float64, error) {
	// Input validation
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"NDCG",
			"input vectors cannot be nil",
		)
	}

	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError(
			"NDCG",
			"input vectors cannot be empty",
		)
	}

	if n != yPred.Len() {
		return 0, scigoErrors.NewDimensionError(
			"NDCG",
			n,
			yPred.Len(),
			0,
		)
	}

	// Validate k parameter
	if k <= 0 && k != -1 {
		return 0, scigoErrors.NewValidationError(
			"k",
			fmt.Sprintf("must be positive or -1 (for all), got %d", k),
			k,
		)
	}

	// If k is -1 or larger than n, use all elements
	if k == -1 || k > n {
		k = n
	}

	// Check for non-negative relevance scores
	for i := 0; i < n; i++ {
		if yTrue.AtVec(i) < 0 {
			return 0, scigoErrors.NewValidationError(
				"yTrue",
				fmt.Sprintf("relevance scores must be non-negative, found %f at index %d", yTrue.AtVec(i), i),
				yTrue.AtVec(i),
			)
		}
	}

	// Create pairs of (score, relevance) and sort by predicted score (descending)
	pairs := make([]struct {
		score     float64
		relevance float64
	}, n)
	for i := 0; i < n; i++ {
		pairs[i] = struct {
			score     float64
			relevance float64
		}{
			score:     yPred.AtVec(i),
			relevance: yTrue.AtVec(i),
		}
	}

	// Sort by predicted score in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// Calculate DCG@k
	dcgValue := dcg(pairs, k)

	// Calculate IDCG@k (ideal DCG)
	// Sort by true relevance for ideal ordering
	idealPairs := make([]struct {
		score     float64
		relevance float64
	}, n)
	for i := 0; i < n; i++ {
		idealPairs[i] = struct {
			score     float64
			relevance float64
		}{
			score:     yTrue.AtVec(i), // Use relevance as score for sorting
			relevance: yTrue.AtVec(i),
		}
	}
	sort.Slice(idealPairs, func(i, j int) bool {
		return idealPairs[i].relevance > idealPairs[j].relevance
	})

	idcgValue := dcg(idealPairs, k)

	// Avoid division by zero
	if idcgValue == 0 {
		// If ideal DCG is 0, all relevances must be 0
		return 0.0, nil
	}

	return dcgValue / idcgValue, nil
}

// dcg calculates the Discounted Cumulative Gain
func dcg(pairs []struct {
	score     float64
	relevance float64
}, k int) float64 {
	if k > len(pairs) {
		k = len(pairs)
	}

	result := 0.0
	for i := 0; i < k; i++ {
		// DCG formula: (2^rel - 1) / log2(i + 2)
		// Note: i+2 because ranking positions start at 1, not 0
		numerator := math.Pow(2, pairs[i].relevance) - 1
		denominator := math.Log2(float64(i + 2))
		result += numerator / denominator
	}

	return result
}

// NDCGMatrix is a convenience wrapper for NDCG that accepts matrix inputs.
//
// This function extracts the first column from each input matrix and
// calls NDCG with the resulting vectors.
//
// Parameters:
//   - yTrue: Matrix where the first column contains ground truth relevance scores
//   - yPred: Matrix where the first column contains predicted scores
//   - k: Number of top results to consider (use -1 for all)
//
// Returns:
//   - The NDCG score
//   - An error if inputs are invalid
func NDCGMatrix(yTrue, yPred mat.Matrix, k int) (float64, error) {
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"NDCGMatrix",
			"input matrices cannot be nil",
		)
	}

	r1, _ := yTrue.Dims()
	r2, _ := yPred.Dims()

	if r1 == 0 || r2 == 0 {
		return 0, scigoErrors.NewValueError(
			"NDCGMatrix",
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

	return NDCG(yTrueVec, yPredVec, k)
}

// AveragePrecision calculates the Average Precision for binary relevance.
//
// Average Precision is the average of precision values at each relevant position,
// commonly used for information retrieval evaluation. It emphasizes returning
// relevant documents earlier in the ranking.
//
// The formula is:
//   AP = Σ(Precision@k × rel_k) / number_of_relevant_items
//
// Parameters:
//   - yTrue: Ground truth binary relevance (0 or 1)
//   - yPred: Predicted scores for ranking
//
// Returns:
//   - The Average Precision score
//   - An error if inputs are invalid
//
// Example:
//
//	yTrue := mat.NewVecDense(5, []float64{1, 0, 1, 0, 1})
//	yPred := mat.NewVecDense(5, []float64{0.9, 0.8, 0.7, 0.6, 0.5})
//	ap, err := AveragePrecision(yTrue, yPred)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Average Precision: %f\n", ap)
func AveragePrecision(yTrue, yPred *mat.VecDense) (float64, error) {
	// Input validation
	if yTrue == nil || yPred == nil {
		return 0, scigoErrors.NewValueError(
			"AveragePrecision",
			"input vectors cannot be nil",
		)
	}

	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError(
			"AveragePrecision",
			"input vectors cannot be empty",
		)
	}

	if n != yPred.Len() {
		return 0, scigoErrors.NewDimensionError(
			"AveragePrecision",
			n,
			yPred.Len(),
			0,
		)
	}

	// Check if yTrue contains only binary values and count relevant items
	numRelevant := 0
	for i := 0; i < n; i++ {
		val := yTrue.AtVec(i)
		if val != 0.0 && val != 1.0 {
			return 0, scigoErrors.NewValidationError(
				"yTrue",
				fmt.Sprintf("must contain only binary values (0 or 1), found %f at index %d", val, i),
				val,
			)
		}
		if val == 1.0 {
			numRelevant++
		}
	}

	// If no relevant items, AP is undefined (return 0)
	if numRelevant == 0 {
		return 0.0, nil
	}

	// Create pairs of (score, relevance) and sort by predicted score (descending)
	pairs := make([]struct {
		score     float64
		relevance float64
	}, n)
	for i := 0; i < n; i++ {
		pairs[i] = struct {
			score     float64
			relevance float64
		}{
			score:     yPred.AtVec(i),
			relevance: yTrue.AtVec(i),
		}
	}

	// Sort by predicted score in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// Calculate Average Precision
	sumPrecisions := 0.0
	numHits := 0

	for i := 0; i < n; i++ {
		if pairs[i].relevance == 1.0 {
			numHits++
			precision := float64(numHits) / float64(i+1)
			sumPrecisions += precision
		}
	}

	return sumPrecisions / float64(numRelevant), nil
}

// MeanAveragePrecision calculates the Mean Average Precision across multiple queries.
//
// MAP is the mean of Average Precision scores for multiple queries,
// commonly used in information retrieval evaluation.
//
// Parameters:
//   - yTrueList: List of ground truth relevance vectors for each query
//   - yPredList: List of predicted score vectors for each query
//
// Returns:
//   - The Mean Average Precision score
//   - An error if inputs are invalid
func MeanAveragePrecision(yTrueList, yPredList []*mat.VecDense) (float64, error) {
	if len(yTrueList) == 0 || len(yPredList) == 0 {
		return 0, scigoErrors.NewValueError(
			"MeanAveragePrecision",
			"input lists cannot be empty",
		)
	}

	if len(yTrueList) != len(yPredList) {
		return 0, scigoErrors.NewDimensionError(
			"MeanAveragePrecision",
			len(yTrueList),
			len(yPredList),
			0,
		)
	}

	sumAP := 0.0
	for i := range yTrueList {
		ap, err := AveragePrecision(yTrueList[i], yPredList[i])
		if err != nil {
			return 0, fmt.Errorf("error computing AP for query %d: %w", i, err)
		}
		sumAP += ap
	}

	return sumAP / float64(len(yTrueList)), nil
}