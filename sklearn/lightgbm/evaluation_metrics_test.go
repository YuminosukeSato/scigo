package lightgbm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// Helper function for numerical assertions with custom tolerance
func assertInDelta(t *testing.T, expected, actual, delta float64, msgAndArgs ...interface{}) {
	t.Helper()
	if math.IsNaN(expected) && math.IsNaN(actual) {
		return // Both NaN is considered equal
	}
	if math.Abs(expected-actual) > delta {
		t.Errorf("Expected %v, got %v (delta: %v) %v", expected, actual, delta, msgAndArgs)
	}
}

// Test ClassificationMetrics constructor and basic functionality
func TestNewClassificationMetrics(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []int
		yPred   []int
		yProba  []float64
		wantErr bool
	}{
		{
			name:    "binary classification valid",
			yTrue:   []int{0, 1, 0, 1},
			yPred:   []int{0, 1, 0, 1},
			yProba:  []float64{0.1, 0.9, 0.2, 0.8},
			wantErr: false,
		},
		{
			name:    "multiclass valid",
			yTrue:   []int{0, 1, 2, 1, 0},
			yPred:   []int{0, 1, 2, 0, 0},
			yProba:  nil,
			wantErr: false,
		},
		{
			name:    "empty yTrue",
			yTrue:   []int{},
			yPred:   []int{},
			yProba:  nil,
			wantErr: true,
		},
		{
			name:    "dimension mismatch yTrue/yPred",
			yTrue:   []int{0, 1},
			yPred:   []int{0},
			yProba:  nil,
			wantErr: true,
		},
		{
			name:    "dimension mismatch yTrue/yProba",
			yTrue:   []int{0, 1},
			yPred:   []int{0, 1},
			yProba:  []float64{0.5},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClassificationMetrics(tt.yTrue, tt.yPred, tt.yProba)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, cm)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cm)
				assert.Equal(t, len(tt.yTrue), cm.nSamples)
			}
		})
	}
}

func TestClassificationMetrics_Accuracy(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    []int
		yPred    []int
		expected float64
	}{
		{
			name:     "perfect accuracy",
			yTrue:    []int{0, 1, 0, 1},
			yPred:    []int{0, 1, 0, 1},
			expected: 1.0,
		},
		{
			name:     "50% accuracy",
			yTrue:    []int{0, 1, 0, 1},
			yPred:    []int{1, 1, 0, 0},
			expected: 0.5,
		},
		{
			name:     "zero accuracy",
			yTrue:    []int{0, 0, 0, 0},
			yPred:    []int{1, 1, 1, 1},
			expected: 0.0,
		},
		{
			name:     "multiclass accuracy",
			yTrue:    []int{0, 1, 2, 1, 0, 2},
			yPred:    []int{0, 1, 2, 0, 0, 1},
			expected: 4.0 / 6.0, // 4 correct out of 6
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClassificationMetrics(tt.yTrue, tt.yPred, nil)
			require.NoError(t, err)

			accuracy := cm.Accuracy()
			assertInDelta(t, tt.expected, accuracy, 1e-9)
		})
	}
}

func TestClassificationMetrics_BinaryPrecisionRecall(t *testing.T) {
	// Binary classification test cases
	tests := []struct {
		name              string
		yTrue             []int
		yPred             []int
		expectedPrecision float64
		expectedRecall    float64
		expectedF1        float64
	}{
		{
			name:              "perfect binary classification",
			yTrue:             []int{0, 1, 0, 1},
			yPred:             []int{0, 1, 0, 1},
			expectedPrecision: 1.0,
			expectedRecall:    1.0,
			expectedF1:        1.0,
		},
		{
			name:              "typical binary case",
			yTrue:             []int{1, 1, 0, 0, 1, 0},
			yPred:             []int{1, 1, 1, 0, 0, 0},
			expectedPrecision: 2.0 / 3.0, // 2 TP out of 3 predicted positive
			expectedRecall:    2.0 / 3.0, // 2 TP out of 3 actual positive
			expectedF1:        2.0 / 3.0, // Harmonic mean
		},
		{
			name:              "all predicted negative",
			yTrue:             []int{0, 1, 0, 1},
			yPred:             []int{0, 0, 0, 0},
			expectedPrecision: 0.0, // No positive predictions
			expectedRecall:    0.0, // No true positives
			expectedF1:        0.0,
		},
		{
			name:              "all same class true",
			yTrue:             []int{1, 1, 1, 1},
			yPred:             []int{1, 1, 0, 0},
			expectedPrecision: 1.0,       // All predicted positives are correct
			expectedRecall:    0.5,       // Half of actual positives predicted
			expectedF1:        2.0 / 3.0, // 2*1*0.5/(1+0.5)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClassificationMetrics(tt.yTrue, tt.yPred, nil)
			require.NoError(t, err)

			precision := cm.Precision()
			recall := cm.Recall()
			f1 := cm.F1Score()

			assertInDelta(t, tt.expectedPrecision, precision, 1e-9, "Precision")
			assertInDelta(t, tt.expectedRecall, recall, 1e-9, "Recall")
			assertInDelta(t, tt.expectedF1, f1, 1e-9, "F1 Score")
		})
	}
}

func TestClassificationMetrics_MulticlassPrecisionRecall(t *testing.T) {
	// Multiclass case: macro-averaged precision/recall
	yTrue := []int{0, 1, 2, 0, 1, 2}
	yPred := []int{0, 1, 2, 1, 1, 0} // Some misclassifications

	cm, err := NewClassificationMetrics(yTrue, yPred, nil)
	require.NoError(t, err)

	precision := cm.Precision() // Should be macro-averaged
	recall := cm.Recall()       // Should be macro-averaged
	f1 := cm.F1Score()

	// Verify that these are reasonable values for multiclass
	assert.True(t, precision >= 0.0 && precision <= 1.0)
	assert.True(t, recall >= 0.0 && recall <= 1.0)
	assert.True(t, f1 >= 0.0 && f1 <= 1.0)

	// For this specific case, should be > 0 since some predictions are correct
	assert.Greater(t, precision, 0.0)
	assert.Greater(t, recall, 0.0)
}

func TestClassificationMetrics_AUC(t *testing.T) {
	tests := []struct {
		name        string
		yTrue       []int
		yPred       []int
		yProba      []float64
		expected    float64
		expectError bool
		tolerance   float64
	}{
		{
			name:        "perfect AUC",
			yTrue:       []int{0, 0, 1, 1},
			yPred:       []int{0, 0, 1, 1},
			yProba:      []float64{0.1, 0.2, 0.8, 0.9},
			expected:    1.0,
			expectError: false,
			tolerance:   1e-9,
		},
		{
			name:        "random classifier AUC",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 1, 0},
			yProba:      []float64{0.5, 0.5, 0.5, 0.5},
			expected:    0.5,
			expectError: false,
			tolerance:   1e-9,
		},
		{
			name:        "all same true class should return 0.5",
			yTrue:       []int{1, 1, 1, 1},
			yPred:       []int{0, 1, 0, 1}, // Mixed predictions creates 2 classes
			yProba:      []float64{0.9, 0.8, 0.7, 0.6},
			expected:    0.5,
			expectError: false, // No error, returns 0.5 for single true class
			tolerance:   1e-9,
		},
		{
			name:        "multiclass should error",
			yTrue:       []int{0, 1, 2},
			yPred:       []int{0, 1, 2},
			yProba:      []float64{0.3, 0.6, 0.9},
			expected:    0.0,
			expectError: true,
			tolerance:   1e-9,
		},
		{
			name:        "no probabilities should error",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 0, 1},
			yProba:      nil,
			expected:    0.0,
			expectError: true,
			tolerance:   1e-9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClassificationMetrics(tt.yTrue, tt.yPred, tt.yProba)
			require.NoError(t, err)

			auc, err := cm.AUC()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assertInDelta(t, tt.expected, auc, tt.tolerance, "AUC")
			}
		})
	}
}

func TestClassificationMetrics_LogLoss(t *testing.T) {
	tests := []struct {
		name        string
		yTrue       []int
		yPred       []int
		yProba      []float64
		expected    float64
		expectError bool
		tolerance   float64
	}{
		{
			name:        "perfect predictions",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 0, 1},
			yProba:      []float64{1e-15, 1.0 - 1e-15, 1e-15, 1.0 - 1e-15}, // Near perfect probabilities
			expected:    0.0,
			expectError: false,
			tolerance:   1e-10,
		},
		{
			name:        "random predictions",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 1, 0},
			yProba:      []float64{0.5, 0.5, 0.5, 0.5},
			expected:    math.Log(2), // -log(0.5) = log(2) ≈ 0.693
			expectError: false,
			tolerance:   1e-9,
		},
		{
			name:        "typical case",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 0, 1},
			yProba:      []float64{0.1, 0.8, 0.3, 0.7},
			expected:    0.2605, // Calculated: -(log(0.9) + log(0.8) + log(0.7) + log(0.7))/4 ≈ 0.2605
			expectError: false,
			tolerance:   1e-3,
		},
		{
			name:        "no probabilities should error",
			yTrue:       []int{0, 1, 0, 1},
			yPred:       []int{0, 1, 0, 1},
			yProba:      nil,
			expected:    0.0,
			expectError: true,
			tolerance:   1e-9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewClassificationMetrics(tt.yTrue, tt.yPred, tt.yProba)
			require.NoError(t, err)

			logLoss, err := cm.LogLoss()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assertInDelta(t, tt.expected, logLoss, tt.tolerance, "LogLoss")
			}
		})
	}
}

func TestClassificationMetrics_ConfusionMatrix(t *testing.T) {
	yTrue := []int{0, 1, 2, 0, 1, 2}
	yPred := []int{0, 1, 2, 1, 2, 0} // Some misclassifications

	cm, err := NewClassificationMetrics(yTrue, yPred, nil)
	require.NoError(t, err)

	confMatrix := cm.ConfusionMatrix()
	rows, cols := confMatrix.Dims()

	assert.Equal(t, 3, rows) // 3 classes
	assert.Equal(t, 3, cols)

	// Check that diagonal elements represent correct predictions
	// and off-diagonal represent misclassifications
	total := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := confMatrix.At(i, j)
			assert.GreaterOrEqual(t, val, 0.0) // Non-negative counts
			total += val
		}
	}
	assert.Equal(t, float64(len(yTrue)), total) // Total should equal number of samples
}

func TestClassificationMetrics_ClassificationReport(t *testing.T) {
	yTrue := []int{0, 1, 0, 1, 0, 1}
	yPred := []int{0, 1, 1, 1, 0, 0}
	yProba := []float64{0.2, 0.8, 0.6, 0.9, 0.1, 0.3}

	cm, err := NewClassificationMetrics(yTrue, yPred, yProba)
	require.NoError(t, err)

	report := cm.ClassificationReport()

	// Check that all expected keys are present
	assert.Contains(t, report, "accuracy")
	assert.Contains(t, report, "precision")
	assert.Contains(t, report, "recall")
	assert.Contains(t, report, "f1_score")
	assert.Contains(t, report, "auc")
	assert.Contains(t, report, "log_loss")
	assert.Contains(t, report, "per_class")

	// Check that all values are reasonable
	assert.True(t, report["accuracy"].(float64) >= 0.0 && report["accuracy"].(float64) <= 1.0)
	assert.True(t, report["precision"].(float64) >= 0.0 && report["precision"].(float64) <= 1.0)
	assert.True(t, report["recall"].(float64) >= 0.0 && report["recall"].(float64) <= 1.0)
	assert.True(t, report["f1_score"].(float64) >= 0.0 && report["f1_score"].(float64) <= 1.0)

	// Check per-class metrics structure
	perClass := report["per_class"].(map[string]map[string]float64)
	assert.Contains(t, perClass, "class_0")
	assert.Contains(t, perClass, "class_1")

	class0 := perClass["class_0"]
	assert.Contains(t, class0, "precision")
	assert.Contains(t, class0, "recall")
	assert.Contains(t, class0, "f1_score")
	assert.Contains(t, class0, "support")
}

func TestEvaluateRegression(t *testing.T) {
	yTrue := mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0})
	yPred := mat.NewVecDense(5, []float64{1.1, 2.1, 2.9, 3.9, 5.1})

	results, err := EvaluateRegression(yTrue, yPred)
	require.NoError(t, err)

	// Check that all expected metrics are present
	expectedMetrics := []string{"mse", "rmse", "mae", "r2_score", "mape", "explained_variance"}
	for _, metric := range expectedMetrics {
		assert.Contains(t, results, metric, "Missing metric: %s", metric)

		if metric != "mape" { // MAPE can be NaN if yTrue contains zeros
			val := results[metric]
			assert.False(t, math.IsNaN(val), "Metric %s should not be NaN: %v", metric, val)
			assert.False(t, math.IsInf(val, 0), "Metric %s should not be Inf: %v", metric, val)
		}
	}

	// Verify RMSE = sqrt(MSE)
	mse := results["mse"]
	rmse := results["rmse"]
	assertInDelta(t, math.Sqrt(mse), rmse, 1e-9, "RMSE should equal sqrt(MSE)")

	// Verify that metrics are reasonable for this case
	assert.Less(t, mse, 1.0, "MSE should be small for this close prediction")
	assert.Greater(t, results["r2_score"], 0.9, "R² should be high for this close prediction")
}

func TestEvaluateClassification(t *testing.T) {
	yTrue := []int{0, 1, 0, 1, 0, 1}
	yPred := []int{0, 1, 1, 1, 0, 0}
	yProba := []float64{0.2, 0.8, 0.7, 0.9, 0.1, 0.3}

	results, err := EvaluateClassification(yTrue, yPred, yProba)
	require.NoError(t, err)

	// Should return the same structure as ClassificationReport
	assert.Contains(t, results, "accuracy")
	assert.Contains(t, results, "precision")
	assert.Contains(t, results, "recall")
	assert.Contains(t, results, "f1_score")
	assert.Contains(t, results, "per_class")

	// Verify all values are in reasonable ranges
	accuracy := results["accuracy"].(float64)
	precision := results["precision"].(float64)
	recall := results["recall"].(float64)
	f1Score := results["f1_score"].(float64)

	assert.True(t, accuracy >= 0.0 && accuracy <= 1.0)
	assert.True(t, precision >= 0.0 && precision <= 1.0)
	assert.True(t, recall >= 0.0 && recall <= 1.0)
	assert.True(t, f1Score >= 0.0 && f1Score <= 1.0)
}

// Edge cases and error handling tests
func TestClassificationMetrics_EdgeCases(t *testing.T) {
	t.Run("single class all correct", func(t *testing.T) {
		yTrue := []int{1, 1, 1, 1}
		yPred := []int{1, 1, 1, 1}

		cm, err := NewClassificationMetrics(yTrue, yPred, nil)
		require.NoError(t, err)

		accuracy := cm.Accuracy()
		assert.Equal(t, 1.0, accuracy)

		// Precision and recall should handle single class gracefully
		precision := cm.Precision()
		recall := cm.Recall()
		assert.False(t, math.IsNaN(precision))
		assert.False(t, math.IsNaN(recall))
	})

	t.Run("single class all wrong", func(t *testing.T) {
		yTrue := []int{0, 0, 0, 0}
		yPred := []int{1, 1, 1, 1}

		cm, err := NewClassificationMetrics(yTrue, yPred, nil)
		require.NoError(t, err)

		accuracy := cm.Accuracy()
		assert.Equal(t, 0.0, accuracy)
	})

	t.Run("extreme probabilities for LogLoss", func(t *testing.T) {
		yTrue := []int{0, 1}
		yPred := []int{0, 1}
		yProba := []float64{0.0, 1.0} // Extreme probabilities

		cm, err := NewClassificationMetrics(yTrue, yPred, yProba)
		require.NoError(t, err)

		logLoss, err := cm.LogLoss()
		assert.NoError(t, err)
		assert.False(t, math.IsInf(logLoss, 0), "LogLoss should handle extreme probabilities")
	})
}

// Numerical precision tests
func TestClassificationMetrics_NumericalPrecision(t *testing.T) {
	t.Run("very close probabilities", func(t *testing.T) {
		yTrue := []int{0, 1}
		yPred := []int{0, 1}
		yProba := []float64{0.5000000001, 0.4999999999} // Very close to 0.5

		cm, err := NewClassificationMetrics(yTrue, yPred, yProba)
		require.NoError(t, err)

		auc, err := cm.AUC()
		assert.NoError(t, err)
		assert.False(t, math.IsNaN(auc))

		logLoss, err := cm.LogLoss()
		assert.NoError(t, err)
		assert.False(t, math.IsNaN(logLoss))
	})

	t.Run("large number of samples", func(t *testing.T) {
		size := 10000
		yTrue := make([]int, size)
		yPred := make([]int, size)
		yProba := make([]float64, size)

		// Generate alternating pattern
		for i := 0; i < size; i++ {
			yTrue[i] = i % 2
			yPred[i] = i % 2
			yProba[i] = 0.5 + 0.1*float64(i%2)
		}

		cm, err := NewClassificationMetrics(yTrue, yPred, yProba)
		require.NoError(t, err)

		accuracy := cm.Accuracy()
		assert.Equal(t, 1.0, accuracy)

		auc, err := cm.AUC()
		assert.NoError(t, err)
		assert.False(t, math.IsNaN(auc))
	})
}

// Integration tests with regression metrics
func TestRegressionMetricsIntegration(t *testing.T) {
	// Test that our wrapper functions match the underlying metrics package
	yTrue := mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0})
	yPred := mat.NewVecDense(4, []float64{1.1, 2.1, 2.9, 3.9})

	// Test MSE
	mse, err := MSE(yTrue, yPred)
	require.NoError(t, err)
	assert.Greater(t, mse, 0.0)
	assert.Less(t, mse, 1.0) // Should be small for close predictions

	// Test MAE
	mae, err := MAE(yTrue, yPred)
	require.NoError(t, err)
	assert.Greater(t, mae, 0.0)
	assert.Greater(t, mae, mse) // MAE = 0.1 > MSE = 0.01 for this case

	// Test R2Score
	r2, err := R2Score(yTrue, yPred)
	require.NoError(t, err)
	assert.Greater(t, r2, 0.9) // Should be high for close predictions

	// Test MAPE
	mape, err := MAPE(yTrue, yPred)
	require.NoError(t, err)
	assert.Greater(t, mape, 0.0)
	assert.Less(t, mape, 20.0) // Should be reasonable percentage

	// Test ExplainedVarianceScore
	evs, err := ExplainedVarianceScore(yTrue, yPred)
	require.NoError(t, err)
	assert.Greater(t, evs, 0.9) // Should be high for good predictions
}
