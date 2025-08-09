package lightgbm

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// LeavesPredictor provides prediction using leaves-compatible format
type LeavesPredictor struct {
	model *LeavesModel
}

// NewLeavesPredictor creates a new leaves-compatible predictor
func NewLeavesPredictor(model *LeavesModel) *LeavesPredictor {
	return &LeavesPredictor{
		model: model,
	}
}

// Predict makes predictions for a batch of samples using Gonum matrices
func (p *LeavesPredictor) Predict(X mat.Matrix) (mat.Matrix, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}

	// Prepare output matrix
	var outputCols int
	if p.model.NumClass > 2 {
		outputCols = p.model.NumClass
	} else {
		outputCols = 1
	}

	predictions := mat.NewDense(rows, outputCols, nil)

	// Process each sample
	for i := 0; i < rows; i++ {
		// Extract features for this sample
		features := mat.Row(nil, i, X)

		// Get prediction
		pred := p.PredictSingle(features)

		if p.model.NumClass > 2 {
			// Multiclass: set all class probabilities
			predictions.SetRow(i, pred)
		} else {
			// Binary or regression: single value
			predictions.Set(i, 0, pred[0])
		}
	}

	return predictions, nil
}

// PredictSingle makes a prediction for a single sample
func (p *LeavesPredictor) PredictSingle(features []float64) []float64 {
	if p.model.NumClass > 2 {
		// Multiclass prediction
		predictions := make([]float64, p.model.NumClass)

		// Initialize with init score for each class
		for i := range predictions {
			predictions[i] = p.model.InitScore
		}

		// Accumulate predictions from trees
		for i, tree := range p.model.Trees {
			classIdx := i % p.model.NumClass
			treeOutput := tree.Predict(features)
			predictions[classIdx] += treeOutput
		}

		// Apply softmax transformation
		return leavesSoftmax(predictions)
	} else {
		// Binary or regression
		// BREAKTHROUGH: LightGBM model files store final leaf values
		// with shrinkage already applied. Simply sum all leaf values.
		prediction := 0.0

		for _, tree := range p.model.Trees {
			treeOutput := tree.Predict(features)
			prediction += treeOutput
		}

		// Apply transformation based on objective
		switch p.model.Objective {
		case BinaryLogistic, BinaryCrossEntropy:
			// Apply sigmoid for binary classification
			prediction = leavesSigmoid(prediction)
		}

		return []float64{prediction}
	}
}

// PredictRaw returns raw predictions without transformations
func (p *LeavesPredictor) PredictRaw(features []float64) []float64 {
	if p.model.NumClass > 2 {
		// Multiclass raw scores
		predictions := make([]float64, p.model.NumClass)

		// Initialize with init score
		for i := range predictions {
			predictions[i] = p.model.InitScore
		}

		// Accumulate predictions from trees
		for i, tree := range p.model.Trees {
			classIdx := i % p.model.NumClass
			treeOutput := tree.Predict(features)
			predictions[classIdx] += treeOutput
		}

		return predictions
	} else {
		// Binary or regression raw score
		prediction := p.model.InitScore

		// Sum predictions from all trees
		for _, tree := range p.model.Trees {
			treeOutput := tree.Predict(features)
			prediction += treeOutput
		}

		return []float64{prediction}
	}
}

// Helper functions for transformations
func leavesSigmoid(x float64) float64 {
	// Numerical stability for sigmoid
	if x > 500 {
		return 1.0
	}
	if x < -500 {
		return 0.0
	}

	// Standard sigmoid: 1 / (1 + exp(-x))
	if x >= 0 {
		expNegX := math.Exp(-x)
		return 1.0 / (1.0 + expNegX)
	} else {
		// For negative x, use exp(x) / (1 + exp(x)) for numerical stability
		expX := math.Exp(x)
		return expX / (1.0 + expX)
	}
}

func leavesSoftmax(x []float64) []float64 {
	// Find max for numerical stability
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp(x - max) and sum
	expSum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		expSum += result[i]
	}

	// Normalize
	if expSum > 0 {
		for i := range result {
			result[i] /= expSum
		}
	}

	return result
}

// PredictBatch makes predictions for multiple samples efficiently using Gonum
func (p *LeavesPredictor) PredictBatch(X mat.Matrix) (*mat.Dense, error) {
	rows, cols := X.Dims()
	if cols != p.model.NumFeatures {
		return nil, fmt.Errorf("feature dimension mismatch: expected %d, got %d", p.model.NumFeatures, cols)
	}

	// Determine output dimensions
	var outputCols int
	if p.model.NumClass > 2 {
		outputCols = p.model.NumClass
	} else {
		outputCols = 1
	}

	// Create output matrix
	result := mat.NewDense(rows, outputCols, nil)

	// Create a dense matrix for raw predictions
	rawPreds := mat.NewDense(rows, outputCols, nil)

	// Initialize with init scores
	if p.model.InitScore != 0 {
		for i := 0; i < rows; i++ {
			for j := 0; j < outputCols; j++ {
				rawPreds.Set(i, j, p.model.InitScore)
			}
		}
	}

	// Process each tree
	for treeIdx, tree := range p.model.Trees {
		// Determine which class this tree belongs to (for multiclass)
		classIdx := 0
		if p.model.NumClass > 2 {
			classIdx = treeIdx % p.model.NumClass
		}

		// Apply tree to each sample
		for i := 0; i < rows; i++ {
			features := mat.Row(nil, i, X)
			treePred := tree.Predict(features)

			// Add to cumulative prediction
			current := rawPreds.At(i, classIdx)
			rawPreds.Set(i, classIdx, current+treePred)
		}
	}

	// Apply final transformations
	for i := 0; i < rows; i++ {
		if p.model.NumClass > 2 {
			// Get raw predictions for this sample
			rawRow := mat.Row(nil, i, rawPreds)
			// Apply softmax
			probas := leavesSoftmax(rawRow)
			result.SetRow(i, probas)
		} else {
			rawVal := rawPreds.At(i, 0)

			// Apply transformation based on objective
			switch p.model.Objective {
			case BinaryLogistic, BinaryCrossEntropy:
				rawVal = leavesSigmoid(rawVal)
			}

			result.Set(i, 0, rawVal)
		}
	}

	return result, nil
}
