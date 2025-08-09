package lightgbm

import (
	"fmt"
	"math"
	"sort"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// ClassificationMetrics provides comprehensive evaluation metrics for classification models
type ClassificationMetrics struct {
	yTrue    []int     // True class labels
	yPred    []int     // Predicted class labels
	yProba   []float64 // Prediction probabilities (for binary classification)
	classes  []int     // Unique class labels
	nClasses int       // Number of classes
	nSamples int       // Number of samples
}

// NewClassificationMetrics creates a new classification metrics calculator
func NewClassificationMetrics(yTrue, yPred []int, yProba []float64) (*ClassificationMetrics, error) {
	if len(yTrue) == 0 {
		return nil, scigoErrors.NewValueError("ClassificationMetrics", "empty yTrue")
	}

	if len(yTrue) != len(yPred) {
		return nil, scigoErrors.NewDimensionError("ClassificationMetrics", len(yTrue), len(yPred), 0)
	}

	// For binary classification with probabilities
	if yProba != nil && len(yProba) != len(yTrue) {
		return nil, scigoErrors.NewDimensionError("ClassificationMetrics", len(yTrue), len(yProba), 0)
	}

	// Extract unique classes
	classSet := make(map[int]bool)
	for _, label := range yTrue {
		classSet[label] = true
	}
	for _, label := range yPred {
		classSet[label] = true
	}

	classes := make([]int, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}
	sort.Ints(classes)

	return &ClassificationMetrics{
		yTrue:    yTrue,
		yPred:    yPred,
		yProba:   yProba,
		classes:  classes,
		nClasses: len(classes),
		nSamples: len(yTrue),
	}, nil
}

// Accuracy calculates the classification accuracy
func (cm *ClassificationMetrics) Accuracy() float64 {
	correct := 0
	for i := 0; i < cm.nSamples; i++ {
		if cm.yTrue[i] == cm.yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(cm.nSamples)
}

// Precision calculates precision for binary or multiclass classification
// For multiclass, returns macro-averaged precision
func (cm *ClassificationMetrics) Precision() float64 {
	if cm.nClasses == 2 {
		return cm.binaryPrecision()
	}
	return cm.multiclassPrecision()
}

// Recall calculates recall for binary or multiclass classification
// For multiclass, returns macro-averaged recall
func (cm *ClassificationMetrics) Recall() float64 {
	if cm.nClasses == 2 {
		return cm.binaryRecall()
	}
	return cm.multiclassRecall()
}

// F1Score calculates F1 score for binary or multiclass classification
// For multiclass, returns macro-averaged F1 score
func (cm *ClassificationMetrics) F1Score() float64 {
	precision := cm.Precision()
	recall := cm.Recall()

	if precision+recall == 0 {
		return 0
	}

	return 2 * (precision * recall) / (precision + recall)
}

// AUC calculates Area Under the ROC Curve for binary classification
func (cm *ClassificationMetrics) AUC() (float64, error) {
	if cm.nClasses != 2 {
		return 0, fmt.Errorf("AUC is only supported for binary classification")
	}

	if cm.yProba == nil {
		return 0, fmt.Errorf("AUC requires prediction probabilities")
	}

	// Create pairs of (probability, true_label)
	type probLabel struct {
		prob  float64
		label int
	}

	pairs := make([]probLabel, cm.nSamples)
	for i := 0; i < cm.nSamples; i++ {
		pairs[i] = probLabel{cm.yProba[i], cm.yTrue[i]}
	}

	// Sort by probability in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})

	// Count positive and negative samples
	positives := 0
	for _, label := range cm.yTrue {
		if label == cm.classes[1] { // Assume positive class is the second class
			positives++
		}
	}
	negatives := cm.nSamples - positives

	if positives == 0 || negatives == 0 {
		return 0.5, nil // AUC is 0.5 when all samples are of one class
	}

	// Calculate AUC using Wilcoxon-Mann-Whitney statistic
	// This is equivalent to the area under the ROC curve
	auc := 0.0
	for i := 0; i < cm.nSamples; i++ {
		if cm.yTrue[i] == cm.classes[1] { // Positive sample
			for j := 0; j < cm.nSamples; j++ {
				if cm.yTrue[j] == cm.classes[0] { // Negative sample
					if cm.yProba[i] > cm.yProba[j] {
						auc += 1.0
					} else if cm.yProba[i] == cm.yProba[j] {
						auc += 0.5
					}
				}
			}
		}
	}

	auc = auc / (float64(positives) * float64(negatives))

	return auc, nil
}

// LogLoss calculates logarithmic loss for binary or multiclass classification
func (cm *ClassificationMetrics) LogLoss() (float64, error) {
	if cm.yProba == nil {
		return 0, fmt.Errorf("LogLoss requires prediction probabilities")
	}

	if cm.nClasses == 2 {
		return cm.binaryLogLoss(), nil
	}

	return 0, fmt.Errorf("multiclass LogLoss not yet implemented")
}

// ConfusionMatrix returns the confusion matrix
func (cm *ClassificationMetrics) ConfusionMatrix() *mat.Dense {
	matrix := mat.NewDense(cm.nClasses, cm.nClasses, nil)

	// Create class index mapping
	classIndex := make(map[int]int)
	for i, class := range cm.classes {
		classIndex[class] = i
	}

	// Fill confusion matrix
	for i := 0; i < cm.nSamples; i++ {
		trueIdx := classIndex[cm.yTrue[i]]
		predIdx := classIndex[cm.yPred[i]]
		current := matrix.At(trueIdx, predIdx)
		matrix.Set(trueIdx, predIdx, current+1)
	}

	return matrix
}

// ClassificationReport returns a comprehensive classification report
func (cm *ClassificationMetrics) ClassificationReport() map[string]interface{} {
	report := make(map[string]interface{})

	report["accuracy"] = cm.Accuracy()
	report["precision"] = cm.Precision()
	report["recall"] = cm.Recall()
	report["f1_score"] = cm.F1Score()

	if cm.nClasses == 2 {
		if auc, err := cm.AUC(); err == nil {
			report["auc"] = auc
		}

		if logloss, err := cm.LogLoss(); err == nil {
			report["log_loss"] = logloss
		}
	}

	// Per-class metrics
	perClass := make(map[string]map[string]float64)
	for i, class := range cm.classes {
		perClass[fmt.Sprintf("class_%d", class)] = map[string]float64{
			"precision": cm.classPrecision(i),
			"recall":    cm.classRecall(i),
			"f1_score":  cm.classF1Score(i),
			"support":   cm.classSupport(i),
		}
	}
	report["per_class"] = perClass

	return report
}

// Helper methods

func (cm *ClassificationMetrics) binaryPrecision() float64 {
	return cm.classPrecision(1) // Positive class
}

func (cm *ClassificationMetrics) binaryRecall() float64 {
	return cm.classRecall(1) // Positive class
}

func (cm *ClassificationMetrics) multiclassPrecision() float64 {
	sum := 0.0
	for i := 0; i < cm.nClasses; i++ {
		sum += cm.classPrecision(i)
	}
	return sum / float64(cm.nClasses)
}

func (cm *ClassificationMetrics) multiclassRecall() float64 {
	sum := 0.0
	for i := 0; i < cm.nClasses; i++ {
		sum += cm.classRecall(i)
	}
	return sum / float64(cm.nClasses)
}

func (cm *ClassificationMetrics) classPrecision(classIdx int) float64 {
	classLabel := cm.classes[classIdx]
	truePositives := 0
	falsePositives := 0

	for i := 0; i < cm.nSamples; i++ {
		if cm.yPred[i] == classLabel {
			if cm.yTrue[i] == classLabel {
				truePositives++
			} else {
				falsePositives++
			}
		}
	}

	if truePositives+falsePositives == 0 {
		return 0
	}

	return float64(truePositives) / float64(truePositives+falsePositives)
}

func (cm *ClassificationMetrics) classRecall(classIdx int) float64 {
	classLabel := cm.classes[classIdx]
	truePositives := 0
	falseNegatives := 0

	for i := 0; i < cm.nSamples; i++ {
		if cm.yTrue[i] == classLabel {
			if cm.yPred[i] == classLabel {
				truePositives++
			} else {
				falseNegatives++
			}
		}
	}

	if truePositives+falseNegatives == 0 {
		return 0
	}

	return float64(truePositives) / float64(truePositives+falseNegatives)
}

func (cm *ClassificationMetrics) classF1Score(classIdx int) float64 {
	precision := cm.classPrecision(classIdx)
	recall := cm.classRecall(classIdx)

	if precision+recall == 0 {
		return 0
	}

	return 2 * (precision * recall) / (precision + recall)
}

func (cm *ClassificationMetrics) classSupport(classIdx int) float64 {
	classLabel := cm.classes[classIdx]
	count := 0

	for i := 0; i < cm.nSamples; i++ {
		if cm.yTrue[i] == classLabel {
			count++
		}
	}

	return float64(count)
}

func (cm *ClassificationMetrics) binaryLogLoss() float64 {
	sum := 0.0
	epsilon := 1e-15 // To avoid log(0)

	for i := 0; i < cm.nSamples; i++ {
		prob := cm.yProba[i]
		prob = math.Max(epsilon, math.Min(1-epsilon, prob)) // Clip probabilities

		if cm.yTrue[i] == cm.classes[1] { // Positive class
			sum += math.Log(prob)
		} else { // Negative class
			sum += math.Log(1 - prob)
		}
	}

	return -sum / float64(cm.nSamples)
}

// Additional utility functions for LightGBM integration

// EvaluateRegression provides comprehensive regression evaluation
func EvaluateRegression(yTrue, yPred *mat.VecDense) (map[string]float64, error) {
	results := make(map[string]float64)

	// Import regression metrics from the metrics package
	if mse, err := MSE(yTrue, yPred); err == nil {
		results["mse"] = mse
		results["rmse"] = math.Sqrt(mse)
	} else {
		return nil, fmt.Errorf("MSE calculation failed: %w", err)
	}

	if mae, err := MAE(yTrue, yPred); err == nil {
		results["mae"] = mae
	} else {
		return nil, fmt.Errorf("MAE calculation failed: %w", err)
	}

	if r2, err := R2Score(yTrue, yPred); err == nil {
		results["r2_score"] = r2
	} else {
		return nil, fmt.Errorf("R2 calculation failed: %w", err)
	}

	if mape, err := MAPE(yTrue, yPred); err == nil {
		results["mape"] = mape
	} else {
		// MAPE can fail if yTrue contains zeros, so we skip it
		results["mape"] = math.NaN()
	}

	if evs, err := ExplainedVarianceScore(yTrue, yPred); err == nil {
		results["explained_variance"] = evs
	} else {
		return nil, fmt.Errorf("Explained variance calculation failed: %w", err)
	}

	return results, nil
}

// EvaluateClassification provides comprehensive classification evaluation
func EvaluateClassification(yTrue, yPred []int, yProba []float64) (map[string]interface{}, error) {
	metrics, err := NewClassificationMetrics(yTrue, yPred, yProba)
	if err != nil {
		return nil, fmt.Errorf("failed to create classification metrics: %w", err)
	}

	return metrics.ClassificationReport(), nil
}

// Import functions from metrics package for regression
// These are wrapper functions to make them available in the lightgbm package

func MSE(yTrue, yPred *mat.VecDense) (float64, error) {
	// This would be imported from the metrics package
	// For now, implement a simple version
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MSE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MSE", n, yPred.Len(), 0)
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += diff * diff
	}

	return sum / float64(n), nil
}

func MAE(yTrue, yPred *mat.VecDense) (float64, error) {
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MAE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MAE", n, yPred.Len(), 0)
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := yTrue.AtVec(i) - yPred.AtVec(i)
		sum += math.Abs(diff)
	}

	return sum / float64(n), nil
}

func R2Score(yTrue, yPred *mat.VecDense) (float64, error) {
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

	if tss == 0 {
		return 0, fmt.Errorf("R2Score: total sum of squares is zero (no variance in yTrue)")
	}

	return 1 - rss/tss, nil
}

func MAPE(yTrue, yPred *mat.VecDense) (float64, error) {
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("MAPE", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("MAPE", n, yPred.Len(), 0)
	}

	var sum float64
	validCount := 0

	for i := 0; i < n; i++ {
		yTrueVal := yTrue.AtVec(i)
		if yTrueVal != 0 {
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

func ExplainedVarianceScore(yTrue, yPred *mat.VecDense) (float64, error) {
	n := yTrue.Len()
	if n == 0 {
		return 0, scigoErrors.NewValueError("ExplainedVarianceScore", "empty vector")
	}

	if yPred.Len() != n {
		return 0, scigoErrors.NewDimensionError("ExplainedVarianceScore", n, yPred.Len(), 0)
	}

	// Calculate means
	var yTrueMean, diffMean float64
	for i := 0; i < n; i++ {
		yTrueMean += yTrue.AtVec(i)
		diffMean += (yTrue.AtVec(i) - yPred.AtVec(i))
	}
	yTrueMean /= float64(n)
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

	return 1 - varDiff/varYTrue, nil
}
