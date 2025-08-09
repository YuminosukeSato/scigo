package lightgbm

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// EarlyStopping handles early stopping logic
type EarlyStopping struct {
	Rounds          int     // Number of rounds without improvement to stop
	BestScore       float64 // Best validation score so far
	BestIteration   int     // Iteration with best score
	RoundsNoImprove int     // Current rounds without improvement
	Metric          string  // Metric to use for early stopping
	Minimize        bool    // Whether to minimize the metric
	Enabled         bool    // Whether early stopping is enabled
}

// NewEarlyStopping creates a new early stopping handler
func NewEarlyStopping(rounds int, metric string) *EarlyStopping {
	if rounds <= 0 {
		return &EarlyStopping{Enabled: false}
	}

	minimize := true
	switch metric {
	case "auc", "accuracy", "precision", "recall", "f1", "r2":
		minimize = false
	}

	bestScore := math.Inf(1)
	if !minimize {
		bestScore = math.Inf(-1)
	}

	return &EarlyStopping{
		Rounds:        rounds,
		BestScore:     bestScore,
		BestIteration: 0,
		Metric:        metric,
		Minimize:      minimize,
		Enabled:       true,
	}
}

// Update updates early stopping state with new score
func (es *EarlyStopping) Update(iteration int, score float64) bool {
	if !es.Enabled {
		return false
	}

	improved := false
	if es.Minimize {
		improved = score < es.BestScore
	} else {
		improved = score > es.BestScore
	}

	if improved {
		es.BestScore = score
		es.BestIteration = iteration
		es.RoundsNoImprove = 0
	} else {
		es.RoundsNoImprove++
	}

	// Return true if should stop
	return es.RoundsNoImprove >= es.Rounds
}

// ShouldStop returns whether training should stop
func (es *EarlyStopping) ShouldStop() bool {
	if !es.Enabled {
		return false
	}
	return es.RoundsNoImprove >= es.Rounds
}

// GetBestIteration returns the best iteration
func (es *EarlyStopping) GetBestIteration() int {
	if !es.Enabled {
		return -1
	}
	return es.BestIteration
}

// ValidationData holds validation dataset
type ValidationData struct {
	X      mat.Matrix
	Y      mat.Matrix
	Weight []float64
}

// FitWithValidation trains the model with validation and early stopping
func (t *Trainer) FitWithValidation(X, y mat.Matrix, valData *ValidationData) error {
	// Set training data first
	var xDense, yDense *mat.Dense
	switch v := X.(type) {
	case *mat.Dense:
		xDense = v
	default:
		rows, cols := X.Dims()
		xDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				xDense.Set(i, j, X.At(i, j))
			}
		}
	}

	switch v := y.(type) {
	case *mat.Dense:
		yDense = v
	default:
		rows, cols := y.Dims()
		yDense = mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				yDense.Set(i, j, y.At(i, j))
			}
		}
	}

	t.X = xDense
	t.y = yDense

	// Now initialize with data set
	if err := t.initialize(); err != nil {
		return fmt.Errorf("initialization failed: %w", err)
	}

	// Create objective function
	objFunc, err := CreateObjectiveFunction(t.params.Objective, &t.params)
	if err != nil {
		return fmt.Errorf("failed to create objective function: %w", err)
	}
	t.objective = objFunc

	// Calculate initial score
	rows, _ := t.y.Dims()
	targets := make([]float64, rows)
	for i := 0; i < rows; i++ {
		targets[i] = t.y.At(i, 0)
	}
	t.initScore = t.objective.GetInitScore(targets)

	// Build histograms
	if err := t.buildHistograms(); err != nil {
		return fmt.Errorf("histogram building failed: %w", err)
	}

	// Initialize early stopping
	var earlyStopping *EarlyStopping
	if t.params.EarlyStopping > 0 && valData != nil {
		metric := "l2" // Default metric
		if t.params.Metric != "" {
			metric = t.params.Metric
		}
		earlyStopping = NewEarlyStopping(t.params.EarlyStopping, metric)
	}

	// Main training loop
	for iter := 0; iter < t.params.NumIterations; iter++ {
		t.iteration = iter

		// Calculate gradients and hessians
		t.calculateGradients()

		// Build one tree
		tree, err := t.buildTree()
		if err != nil {
			return fmt.Errorf("tree building failed at iteration %d: %w", iter, err)
		}

		// Add tree to ensemble
		t.trees = append(t.trees, tree)

		// Update predictions
		t.updatePredictions(tree)

		// Early stopping check
		if earlyStopping != nil && earlyStopping.Enabled {
			valScore := t.evaluateValidation(valData)
			shouldStop := earlyStopping.Update(iter, valScore)

			if shouldStop {
				// Keep only trees up to best iteration
				if earlyStopping.BestIteration < len(t.trees) {
					t.trees = t.trees[:earlyStopping.BestIteration+1]
				}
				t.bestScore = earlyStopping.BestScore
				break
			}
		}

		// Check legacy early stopping
		if t.params.EarlyStopping > 0 && t.checkEarlyStopping() {
			break
		}
	}

	return nil
}

// evaluateValidation evaluates the model on validation data
func (t *Trainer) evaluateValidation(valData *ValidationData) float64 {
	if valData == nil {
		return 0.0
	}

	model := t.GetModel()
	pred, err := model.Predict(valData.X)
	if err != nil {
		return math.Inf(1)
	}

	// Calculate metric based on objective
	rows, _ := valData.Y.Dims()
	loss := 0.0

	for i := 0; i < rows; i++ {
		p := pred.At(i, 0)
		y := valData.Y.At(i, 0)

		// Use objective's loss function
		loss += t.objective.CalculateLoss(p, y)

		if valData.Weight != nil && i < len(valData.Weight) {
			loss *= valData.Weight[i]
		}
	}

	return loss / float64(rows)
}
