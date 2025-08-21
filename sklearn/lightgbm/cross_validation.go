package lightgbm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// KFoldSplitter defines interface for cross-validation splitters
type KFoldSplitter interface {
	Split(X, y mat.Matrix) []CVFold
	GetNSplits() int
}

// CVFold represents a single fold in cross-validation
type CVFold struct {
	TrainIndices []int
	TestIndices  []int
}

// KFold implements k-fold cross-validation splitter
type KFold struct {
	NSplits    int
	Shuffle    bool
	RandomSeed int
}

// NewKFold creates a new k-fold splitter
func NewKFold(nSplits int, shuffle bool, randomSeed int) *KFold {
	if nSplits < 2 {
		nSplits = 5 // Default to 5-fold
	}
	return &KFold{
		NSplits:    nSplits,
		Shuffle:    shuffle,
		RandomSeed: randomSeed,
	}
}

// GetNSplits returns the number of splits
func (kf *KFold) GetNSplits() int {
	return kf.NSplits
}

// Split generates train/test indices for each fold
func (kf *KFold) Split(X, _ mat.Matrix) []CVFold {
	nSamples, _ := X.Dims()

	// Create indices
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	// Shuffle if requested
	if kf.Shuffle {
		r := rand.New(rand.NewPCG(uint64(kf.RandomSeed), uint64(kf.RandomSeed)))
		r.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	// Create folds
	folds := make([]CVFold, kf.NSplits)
	foldSize := nSamples / kf.NSplits
	remainder := nSamples % kf.NSplits

	currentIdx := 0
	for i := 0; i < kf.NSplits; i++ {
		testSize := foldSize
		if i < remainder {
			testSize++
		}

		// Test indices for this fold
		testIndices := make([]int, testSize)
		copy(testIndices, indices[currentIdx:currentIdx+testSize])

		// Train indices (all except test)
		trainIndices := make([]int, 0, nSamples-testSize)
		for j := 0; j < nSamples; j++ {
			isTest := false
			for _, testIdx := range testIndices {
				if indices[j] == testIdx {
					isTest = true
					break
				}
			}
			if !isTest {
				trainIndices = append(trainIndices, indices[j])
			}
		}

		folds[i] = CVFold{
			TrainIndices: trainIndices,
			TestIndices:  testIndices,
		}

		currentIdx += testSize
	}

	return folds
}

// StratifiedKFold implements stratified k-fold cross-validation
type StratifiedKFold struct {
	NSplits    int
	Shuffle    bool
	RandomSeed int
}

// NewStratifiedKFold creates a new stratified k-fold splitter
func NewStratifiedKFold(nSplits int, shuffle bool, randomSeed int) *StratifiedKFold {
	if nSplits < 2 {
		nSplits = 5
	}
	return &StratifiedKFold{
		NSplits:    nSplits,
		Shuffle:    shuffle,
		RandomSeed: randomSeed,
	}
}

// GetNSplits returns the number of splits
func (skf *StratifiedKFold) GetNSplits() int {
	return skf.NSplits
}

// Split generates stratified train/test indices for each fold
func (skf *StratifiedKFold) Split(X, y mat.Matrix) []CVFold {
	nSamples, _ := X.Dims()

	// Group indices by class
	classIndices := make(map[float64][]int)
	for i := 0; i < nSamples; i++ {
		label := y.At(i, 0)
		classIndices[label] = append(classIndices[label], i)
	}

	// Shuffle indices within each class if requested
	if skf.Shuffle {
		r := rand.New(rand.NewPCG(uint64(skf.RandomSeed), uint64(skf.RandomSeed)))
		for label := range classIndices {
			indices := classIndices[label]
			r.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
	}

	// Create stratified folds
	folds := make([]CVFold, skf.NSplits)
	for i := 0; i < skf.NSplits; i++ {
		folds[i] = CVFold{
			TrainIndices: make([]int, 0),
			TestIndices:  make([]int, 0),
		}
	}

	// Distribute each class across folds
	for _, indices := range classIndices {
		nClass := len(indices)
		foldSize := nClass / skf.NSplits
		remainder := nClass % skf.NSplits

		currentIdx := 0
		for i := 0; i < skf.NSplits; i++ {
			testSize := foldSize
			if i < remainder {
				testSize++
			}

			// Add to test set for this fold
			for j := 0; j < testSize && currentIdx < nClass; j++ {
				folds[i].TestIndices = append(folds[i].TestIndices, indices[currentIdx])
				currentIdx++
			}
		}
	}

	// Build train sets (all samples not in test)
	for i := 0; i < skf.NSplits; i++ {
		testSet := make(map[int]bool)
		for _, idx := range folds[i].TestIndices {
			testSet[idx] = true
		}

		for j := 0; j < nSamples; j++ {
			if !testSet[j] {
				folds[i].TrainIndices = append(folds[i].TrainIndices, j)
			}
		}
	}

	return folds
}

// CVResult stores cross-validation results
type CVResult struct {
	TrainScores   []float64
	TestScores    []float64
	FitTimes      []float64
	ScoreTimes    []float64
	Models        []*Model
	BestIteration int
	BestScore     float64
}

// GetMeanScore returns mean test score
func (cv *CVResult) GetMeanScore() float64 {
	if len(cv.TestScores) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, score := range cv.TestScores {
		sum += score
	}
	return sum / float64(len(cv.TestScores))
}

// GetStdScore returns standard deviation of test scores
func (cv *CVResult) GetStdScore() float64 {
	if len(cv.TestScores) <= 1 {
		return 0.0
	}

	mean := cv.GetMeanScore()
	sumSq := 0.0
	for _, score := range cv.TestScores {
		diff := score - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(cv.TestScores)-1))
}

// CrossValidate performs cross-validation for LightGBM
func CrossValidate(params TrainingParams, X, y mat.Matrix, splitter KFoldSplitter,
	metric string, earlyStopping int, verbose bool) (*CVResult, error) {

	folds := splitter.Split(X, y)
	nFolds := len(folds)

	result := &CVResult{
		TrainScores: make([]float64, nFolds),
		TestScores:  make([]float64, nFolds),
		FitTimes:    make([]float64, nFolds),
		ScoreTimes:  make([]float64, nFolds),
		Models:      make([]*Model, nFolds),
	}

	// Set metric if not specified
	if metric == "" {
		switch params.Objective {
		case "regression", "regression_l2":
			metric = "l2"
		case "binary":
			metric = "binary_logloss"
		default:
			metric = "l2"
		}
	}

	// Process each fold
	var wg sync.WaitGroup
	errors := make([]error, nFolds)

	for foldIdx := 0; foldIdx < nFolds; foldIdx++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			fold := folds[idx]

			// Create train and test matrices
			trainX, trainY := extractSubset(X, y, fold.TrainIndices)
			testX, testY := extractSubset(X, y, fold.TestIndices)

			// Train model
			trainer := NewTrainer(params)

			// Add early stopping if requested
			if earlyStopping > 0 {
				// Create validation data
				valData := &ValidationData{
					X: testX,
					Y: testY,
				}

				// Use FitWithValidation
				if err := trainer.FitWithValidation(trainX, trainY, valData); err != nil {
					errors[idx] = fmt.Errorf("fold %d training failed: %w", idx, err)
					return
				}
			} else {
				// Regular fit
				if err := trainer.Fit(trainX, trainY); err != nil {
					errors[idx] = fmt.Errorf("fold %d training failed: %w", idx, err)
					return
				}
			}

			// Get model
			model := trainer.GetModel()
			result.Models[idx] = model

			// Evaluate on train set
			trainPred, err := model.Predict(trainX)
			if err != nil {
				errors[idx] = fmt.Errorf("fold %d train prediction failed: %w", idx, err)
				return
			}
			trainScore := evaluateMetric(trainY, trainPred, metric, params.Objective)
			result.TrainScores[idx] = trainScore

			// Evaluate on test set
			testPred, err := model.Predict(testX)
			if err != nil {
				errors[idx] = fmt.Errorf("fold %d test prediction failed: %w", idx, err)
				return
			}
			testScore := evaluateMetric(testY, testPred, metric, params.Objective)
			result.TestScores[idx] = testScore

			if verbose {
				fmt.Printf("Fold %d/%d - Train: %.4f, Test: %.4f\n",
					idx+1, nFolds, trainScore, testScore)
			}
		}(foldIdx)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	// Find best score
	result.BestScore = result.TestScores[0]
	result.BestIteration = 0
	for i := 1; i < len(result.TestScores); i++ {
		// For loss metrics, lower is better
		if isLossMetric(metric) {
			if result.TestScores[i] < result.BestScore {
				result.BestScore = result.TestScores[i]
				result.BestIteration = i
			}
		} else {
			// For score metrics, higher is better
			if result.TestScores[i] > result.BestScore {
				result.BestScore = result.TestScores[i]
				result.BestIteration = i
			}
		}
	}

	if verbose {
		fmt.Printf("\nCross-validation results:\n")
		fmt.Printf("Mean score: %.4f (+/- %.4f)\n", result.GetMeanScore(), result.GetStdScore())
		fmt.Printf("Best fold: %d with score: %.4f\n", result.BestIteration+1, result.BestScore)
	}

	return result, nil
}

// extractSubset extracts subset of data based on indices
func extractSubset(X, y mat.Matrix, indices []int) (mat.Matrix, mat.Matrix) {
	rows := len(indices)
	_, xCols := X.Dims()
	_, yCols := y.Dims()

	// Sort indices for efficient access
	sortedIndices := make([]int, len(indices))
	copy(sortedIndices, indices)
	sort.Ints(sortedIndices)

	// Create subset matrices
	xSubset := mat.NewDense(rows, xCols, nil)
	ySubset := mat.NewDense(rows, yCols, nil)

	for i, idx := range sortedIndices {
		for j := 0; j < xCols; j++ {
			xSubset.Set(i, j, X.At(idx, j))
		}
		for j := 0; j < yCols; j++ {
			ySubset.Set(i, j, y.At(idx, j))
		}
	}

	return xSubset, ySubset
}

// evaluateMetric calculates the specified metric
func evaluateMetric(yTrue, yPred mat.Matrix, metric, _ string) float64 {
	rows, _ := yTrue.Dims()

	switch metric {
	case "l2", "mse", "rmse":
		mse := 0.0
		for i := 0; i < rows; i++ {
			diff := yTrue.At(i, 0) - yPred.At(i, 0)
			mse += diff * diff
		}
		mse /= float64(rows)
		if metric == "rmse" {
			return math.Sqrt(mse)
		}
		return mse

	case "l1", "mae":
		mae := 0.0
		for i := 0; i < rows; i++ {
			mae += math.Abs(yTrue.At(i, 0) - yPred.At(i, 0))
		}
		return mae / float64(rows)

	case "binary_logloss", "logloss":
		logloss := 0.0
		eps := 1e-15
		for i := 0; i < rows; i++ {
			p := yPred.At(i, 0)
			// Clip prediction to avoid log(0)
			if p < eps {
				p = eps
			} else if p > 1-eps {
				p = 1 - eps
			}

			y := yTrue.At(i, 0)
			if y == 1 {
				logloss -= math.Log(p)
			} else {
				logloss -= math.Log(1 - p)
			}
		}
		return logloss / float64(rows)

	case "accuracy":
		correct := 0
		for i := 0; i < rows; i++ {
			pred := 0.0
			if yPred.At(i, 0) > 0.5 {
				pred = 1.0
			}
			if pred == yTrue.At(i, 0) {
				correct++
			}
		}
		return float64(correct) / float64(rows)

	default:
		// Default to MSE
		mse := 0.0
		for i := 0; i < rows; i++ {
			diff := yTrue.At(i, 0) - yPred.At(i, 0)
			mse += diff * diff
		}
		return mse / float64(rows)
	}
}

// isLossMetric returns true if metric is a loss (lower is better)
func isLossMetric(metric string) bool {
	switch metric {
	case "l1", "l2", "mae", "mse", "rmse", "logloss", "binary_logloss":
		return true
	case "accuracy", "auc", "r2":
		return false
	default:
		return true // Default to loss metric
	}
}

// CrossValidateRegressor performs cross-validation for LGBMRegressor
func CrossValidateRegressor(regressor *LGBMRegressor, X, y mat.Matrix,
	cv KFoldSplitter, scoring string, verbose bool) (*CVResult, error) {

	// Convert regressor parameters to TrainingParams
	params := TrainingParams{
		NumIterations:       regressor.NumIterations,
		LearningRate:        regressor.LearningRate,
		NumLeaves:           regressor.NumLeaves,
		MaxDepth:            regressor.MaxDepth,
		MinDataInLeaf:       regressor.MinChildSamples,
		Lambda:              regressor.RegLambda,
		Alpha:               regressor.RegAlpha,
		BaggingFraction:     regressor.Subsample,
		BaggingFreq:         regressor.SubsampleFreq,
		FeatureFraction:     regressor.ColsampleBytree,
		Objective:           regressor.Objective,
		NumClass:            1,
		Seed:                regressor.RandomState,
		Deterministic:       regressor.Deterministic,
		Verbosity:           regressor.Verbosity,
		EarlyStopping:       regressor.EarlyStopping,
		CategoricalFeatures: regressor.CategoricalFeatures,
	}

	// Default scoring for regression
	if scoring == "" {
		scoring = "l2"
	}

	return CrossValidate(params, X, y, cv, scoring, regressor.EarlyStopping, verbose)
}

// CrossValidateClassifier performs cross-validation for LGBMClassifier
func CrossValidateClassifier(classifier *LGBMClassifier, X, y mat.Matrix,
	cv KFoldSplitter, scoring string, verbose bool) (*CVResult, error) {

	// Convert classifier parameters to TrainingParams
	params := TrainingParams{
		NumIterations:       classifier.NumIterations,
		LearningRate:        classifier.LearningRate,
		NumLeaves:           classifier.NumLeaves,
		MaxDepth:            classifier.MaxDepth,
		MinDataInLeaf:       classifier.MinChildSamples,
		Lambda:              classifier.RegLambda,
		Alpha:               classifier.RegAlpha,
		BaggingFraction:     classifier.Subsample,
		BaggingFreq:         classifier.SubsampleFreq,
		FeatureFraction:     classifier.ColsampleBytree,
		Objective:           classifier.Objective,
		NumClass:            classifier.NumClass,
		Seed:                classifier.RandomState,
		Deterministic:       classifier.Deterministic,
		Verbosity:           classifier.Verbosity,
		EarlyStopping:       classifier.EarlyStopping,
		CategoricalFeatures: classifier.CategoricalFeatures,
	}

	// Default scoring for classification
	if scoring == "" {
		if params.NumClass > 2 {
			scoring = "accuracy"
		} else {
			scoring = "binary_logloss"
		}
	}

	return CrossValidate(params, X, y, cv, scoring, classifier.EarlyStopping, verbose)
}
