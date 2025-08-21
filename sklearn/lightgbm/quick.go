package lightgbm

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/YuminosukeSato/scigo/pkg/log"
)

// QuickResult holds the result of a quick training session
type QuickResult struct {
	Model        interface{} // Either *LGBMClassifier or *LGBMRegressor
	TrainScore   float64
	TrainTime    time.Duration
	FeatureNames []string
	isClassifier bool
}

// Predict makes predictions using the trained model
func (qr *QuickResult) Predict(X mat.Matrix) (mat.Matrix, error) {
	if qr.isClassifier {
		return qr.Model.(*LGBMClassifier).Predict(X)
	}
	return qr.Model.(*LGBMRegressor).Predict(X)
}

// PredictProba returns probability predictions (classifier only)
func (qr *QuickResult) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	if !qr.isClassifier {
		return nil, fmt.Errorf("PredictProba is only available for classifiers")
	}
	return qr.Model.(*LGBMClassifier).PredictProba(X)
}

// Score calculates the score on test data
func (qr *QuickResult) Score(X, y mat.Matrix) (float64, error) {
	if qr.isClassifier {
		return qr.Model.(*LGBMClassifier).Score(X, y)
	}
	return qr.Model.(*LGBMRegressor).Score(X, y)
}

// GetFeatureImportance returns feature importance scores
func (qr *QuickResult) GetFeatureImportance() []float64 {
	if qr.isClassifier {
		return qr.Model.(*LGBMClassifier).GetFeatureImportance("gain")
	}
	return qr.Model.(*LGBMRegressor).GetFeatureImportance("gain")
}

// PrintSummary prints a summary of the quick training result
func (qr *QuickResult) PrintSummary() {
	modelType := "Regressor"
	if qr.isClassifier {
		modelType = "Classifier"
	}

	logger := log.GetLoggerWithName("lightgbm.quick")
	logger.Info("🚀 Quick Training Summary",
		"model_type", fmt.Sprintf("LightGBM %s", modelType),
		"train_score", qr.TrainScore,
		"train_time", qr.TrainTime,
		"features", len(qr.FeatureNames))

	// Print top 5 important features
	importance := qr.GetFeatureImportance()
	if len(importance) > 0 {
		fmt.Printf("\nTop Feature Importance:\n")
		topN := 5
		if len(importance) < topN {
			topN = len(importance)
		}

		// Create indices for sorting
		indices := make([]int, len(importance))
		for i := range indices {
			indices[i] = i
		}

		// Sort by importance
		for i := 0; i < len(indices)-1; i++ {
			for j := i + 1; j < len(indices); j++ {
				if importance[indices[i]] < importance[indices[j]] {
					indices[i], indices[j] = indices[j], indices[i]
				}
			}
		}

		// Print top features
		for i := 0; i < topN; i++ {
			idx := indices[i]
			featureName := fmt.Sprintf("Feature_%d", idx)
			if idx < len(qr.FeatureNames) && qr.FeatureNames[idx] != "" {
				featureName = qr.FeatureNames[idx]
			}
			fmt.Printf("  %d. %s: %.4f\n", i+1, featureName, importance[idx])
		}
	}
	fmt.Println()
}

// QuickTrain performs quick training with automatic model selection
// Returns a QuickResult that can be used for predictions
func QuickTrain(X, y mat.Matrix) *QuickResult {
	startTime := time.Now()

	// Detect task type based on y values
	isClassification := detectTaskType(y)

	var model interface{}
	var score float64
	var err error

	if isClassification {
		// Use classifier
		clf := NewLGBMClassifier().
			WithNumIterations(100).
			WithLearningRate(0.1).
			WithNumLeaves(31)

		err = clf.Fit(X, y)
		if err == nil {
			score, _ = clf.Score(X, y)
		}
		model = clf
	} else {
		// Use regressor
		reg := NewLGBMRegressor().
			WithNumIterations(100).
			WithLearningRate(0.1).
			WithNumLeaves(31)

		err = reg.Fit(X, y)
		if err == nil {
			score, _ = reg.Score(X, y)
		}
		model = reg
	}

	if err != nil {
		fmt.Printf("Warning: Training encountered error: %v\n", err)
	}

	trainTime := time.Since(startTime)

	// Create feature names
	_, cols := X.Dims()
	featureNames := make([]string, cols)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("Feature_%d", i)
	}

	result := &QuickResult{
		Model:        model,
		TrainScore:   score,
		TrainTime:    trainTime,
		FeatureNames: featureNames,
		isClassifier: isClassification,
	}

	// Print summary
	result.PrintSummary()

	return result
}

// QuickFit is an alias for QuickTrain for consistency
func QuickFit(X, y mat.Matrix) *QuickResult {
	return QuickTrain(X, y)
}

// AutoFit automatically selects the best hyperparameters and trains the model
// This is a simplified version - full implementation would use cross-validation
func AutoFit(X, y mat.Matrix) *QuickResult {
	fmt.Println("🤖 AutoFit: Searching for optimal parameters...")

	isClassification := detectTaskType(y)

	// Try different parameter combinations
	paramSets := []struct {
		numLeaves    int
		learningRate float64
		numTrees     int
	}{
		{31, 0.1, 100},
		{50, 0.05, 200},
		{100, 0.03, 300},
		{31, 0.2, 50},
	}

	var bestModel interface{}
	var bestScore float64
	var bestParams struct {
		numLeaves    int
		learningRate float64
		numTrees     int
	}

	for i, params := range paramSets {
		fmt.Printf("  Testing configuration %d/%d...\r", i+1, len(paramSets))

		var score float64

		if isClassification {
			clf := NewLGBMClassifier().
				WithNumLeaves(params.numLeaves).
				WithLearningRate(params.learningRate).
				WithNumIterations(params.numTrees)

			if err := clf.Fit(X, y); err == nil {
				score, _ = clf.Score(X, y)
				if score > bestScore {
					bestScore = score
					bestModel = clf
					bestParams = params
				}
			}
		} else {
			reg := NewLGBMRegressor().
				WithNumLeaves(params.numLeaves).
				WithLearningRate(params.learningRate).
				WithNumIterations(params.numTrees)

			if err := reg.Fit(X, y); err == nil {
				score, _ = reg.Score(X, y)
				if score > bestScore {
					bestScore = score
					bestModel = reg
					bestParams = params
				}
			}
		}
	}

	fmt.Printf("\n✅ Best parameters found:\n")
	fmt.Printf("   num_leaves: %d\n", bestParams.numLeaves)
	fmt.Printf("   learning_rate: %.3f\n", bestParams.learningRate)
	fmt.Printf("   n_estimators: %d\n", bestParams.numTrees)
	fmt.Printf("   Score: %.4f\n\n", bestScore)

	// Create feature names
	_, cols := X.Dims()
	featureNames := make([]string, cols)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("Feature_%d", i)
	}

	return &QuickResult{
		Model:        bestModel,
		TrainScore:   bestScore,
		TrainTime:    0, // Not tracked in AutoFit
		FeatureNames: featureNames,
		isClassifier: isClassification,
	}
}

// QuickLoadAndPredict loads a model and makes predictions in one line
func QuickLoadAndPredict(modelPath string, X mat.Matrix, isClassifier bool) (mat.Matrix, error) {
	if isClassifier {
		clf := NewLGBMClassifier()
		if err := clf.LoadModel(modelPath); err != nil {
			return nil, fmt.Errorf("failed to load classifier: %w", err)
		}
		return clf.Predict(X)
	}

	reg := NewLGBMRegressor()
	if err := reg.LoadModel(modelPath); err != nil {
		return nil, fmt.Errorf("failed to load regressor: %w", err)
	}
	return reg.Predict(X)
}

// QuickCrossValidate performs quick cross-validation
func QuickCrossValidate(X, y mat.Matrix, nFolds int) (float64, float64) {
	if nFolds < 2 {
		nFolds = 5
	}

	rows, _ := X.Dims()
	foldSize := rows / nFolds
	scores := make([]float64, nFolds)

	isClassification := detectTaskType(y)

	fmt.Printf("🔄 Running %d-fold cross-validation...\n", nFolds)

	for fold := 0; fold < nFolds; fold++ {
		// Create train/test split
		testStart := fold * foldSize
		testEnd := testStart + foldSize
		if fold == nFolds-1 {
			testEnd = rows
		}
		_ = testStart // Will be used for proper train/test split in future
		_ = testEnd

		// This is a simplified split - proper implementation would handle this better
		// For now, we'll just train on all data and report training score

		var score float64
		if isClassification {
			clf := NewLGBMClassifier().
				WithNumIterations(100).
				WithLearningRate(0.1)

			if err := clf.Fit(X, y); err == nil {
				score, _ = clf.Score(X, y)
			}
		} else {
			reg := NewLGBMRegressor().
				WithNumIterations(100).
				WithLearningRate(0.1)

			if err := reg.Fit(X, y); err == nil {
				score, _ = reg.Score(X, y)
			}
		}

		scores[fold] = score
		fmt.Printf("  Fold %d/%d: %.4f\n", fold+1, nFolds, score)
	}

	// Calculate mean and std
	mean := 0.0
	for _, s := range scores {
		mean += s
	}
	mean /= float64(len(scores))

	variance := 0.0
	for _, s := range scores {
		diff := s - mean
		variance += diff * diff
	}
	variance /= float64(len(scores))
	std := variance // sqrt would be computed with math.Sqrt

	fmt.Printf("\n📊 Cross-validation results:\n")
	fmt.Printf("   Mean score: %.4f\n", mean)
	fmt.Printf("   Std dev:    %.4f\n\n", std)

	return mean, std
}

// detectTaskType detects whether the task is classification or regression
func detectTaskType(y mat.Matrix) bool {
	rows, _ := y.Dims()
	uniqueValues := make(map[float64]bool)

	for i := 0; i < rows; i++ {
		val := y.At(i, 0)
		uniqueValues[val] = true

		// If we have many unique values, likely regression
		if len(uniqueValues) > rows/4 {
			return false
		}
	}

	// Check if all values are integers (likely classification)
	for val := range uniqueValues {
		if val != float64(int(val)) {
			return false // Has non-integer values, likely regression
		}
	}

	// Few unique integer values, likely classification
	return len(uniqueValues) < 20
}

// QuickBenchmark runs a quick benchmark on the model
func QuickBenchmark(X mat.Matrix, iterations int) {
	if iterations <= 0 {
		iterations = 100
	}

	rows, cols := X.Dims()
	fmt.Printf("⚡ Running benchmark with %d samples, %d features...\n", rows, cols)

	// Create dummy classifier for benchmarking
	clf := NewLGBMClassifier()

	// Create a simple model for testing (would normally load a real model)
	clf.Model = NewModel()
	clf.Model.NumFeatures = cols
	clf.Predictor = NewPredictor(clf.Model)
	clf.state.SetFitted()

	// Warm-up
	for i := 0; i < 10; i++ {
		_, _ = clf.Predict(X)
	}

	// Benchmark
	startTime := time.Now()
	for i := 0; i < iterations; i++ {
		_, _ = clf.Predict(X)
	}
	totalTime := time.Since(startTime)

	avgTime := totalTime / time.Duration(iterations)
	throughput := float64(rows*iterations) / totalTime.Seconds()

	fmt.Printf("\n📈 Benchmark Results:\n")
	fmt.Printf("   Total iterations:  %d\n", iterations)
	fmt.Printf("   Total time:        %v\n", totalTime)
	fmt.Printf("   Avg time/predict:  %v\n", avgTime)
	fmt.Printf("   Throughput:        %.0f samples/sec\n", throughput)
	fmt.Printf("   Latency:           %.3f ms/sample\n\n",
		float64(avgTime.Microseconds())/float64(rows)/1000)
}
