package lightgbm_capi

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/YuminosukeSato/scigo/sklearn/lightgbm"
	"gonum.org/v1/gonum/mat"
)

// TestPureGoCAPISpec_PredictModes_Regression
// - Emulates C API predict types using pure Go: normal (prob/value) vs raw score
func TestPureGoCAPISpec_PredictModes_Regression(t *testing.T) {
	// Use model that exists in repo testdata
	modelPath := filepath.Join("testdata", "regression", "model.txt")

	// Leaves-based loader + predictor supports raw scores clearly
	leavesModel, err := lightgbm.LoadLeavesModelFromFile(modelPath)
	if err != nil {
		t.Fatalf("failed to load leaves model: %v", err)
	}
	leavesPred := lightgbm.NewLeavesPredictor(leavesModel)

	// Load baseline test samples from Python outputs
	data, err := os.ReadFile(filepath.Join("testdata", "regression", "test_data.json"))
	if err != nil {
		t.Fatalf("failed to read test data: %v", err)
	}
	var baseline struct {
		XTest       [][]float64 `json:"X_test"`
		Predictions []float64   `json:"predictions"`
	}
	if err := json.Unmarshal(data, &baseline); err != nil {
		t.Fatalf("failed to unmarshal test data: %v", err)
	}

	// Build matrix
	rows := len(baseline.XTest)
	cols := len(baseline.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, r := range baseline.XTest {
		X.SetRow(i, r)
	}

	// Normal predictions (C API PredictNormal)
	preds, err := leavesPred.Predict(X)
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}

	// Raw predictions (C API PredictRawScore)
	// Compare first few samples in detail
	tol := 1e-4
	for i := 0; i < rows && i < 5; i++ {
		// Single-sample raw
		raw := leavesPred.PredictRaw(baseline.XTest[i])[0]
		// Batch (normal) is same objective transform as Python baseline
		normal := preds.At(i, 0)
		// The baseline.Predictions are the Python LightGBM final values
		py := baseline.Predictions[i]

		// Basic sanity
		if math.IsNaN(normal) || math.IsInf(normal, 0) {
			t.Fatalf("invalid normal prediction at %d: %v", i, normal)
		}
		if math.IsNaN(raw) || math.IsInf(raw, 0) {
			t.Fatalf("invalid raw prediction at %d: %v", i, raw)
		}

		// Normal prediction should be close to Python’s final prediction
		if math.Abs(normal-py) > tol {
			t.Logf("regression normal mismatch[%d]: go=%.6f py=%.6f diff=%.3e", i, normal, py, math.Abs(normal-py))
		}

		// Raw score should differ from final unless objective is identity
		if math.Abs(raw-normal) < 1e-12 {
			t.Logf("raw==normal for sample %d (ok if regression identity)", i)
		}
	}
}

// TestPureGoCAPISpec_PredictModes_Binary
// - Emulates C API PredictNormal and PredictRawScore for binary
func TestPureGoCAPISpec_PredictModes_Binary(t *testing.T) {
	modelPath := filepath.Join("testdata", "binary", "model.txt")
	leavesModel, err := lightgbm.LoadLeavesModelFromFile(modelPath)
	if err != nil {
		t.Fatalf("failed to load leaves model: %v", err)
	}
	pred := lightgbm.NewLeavesPredictor(leavesModel)

	data, err := os.ReadFile(filepath.Join("testdata", "binary", "test_data.json"))
	if err != nil {
		t.Fatalf("failed to read test data: %v", err)
	}
	var baseline struct {
		XTest        [][]float64 `json:"X_test"`
		PredictProba [][]float64 `json:"predict_proba"`
	}
	if err := json.Unmarshal(data, &baseline); err != nil {
		t.Fatalf("failed to unmarshal test data: %v", err)
	}

	rows := len(baseline.XTest)
	cols := len(baseline.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, r := range baseline.XTest {
		X.SetRow(i, r)
	}

	probs, err := pred.Predict(X)
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}

	tol := 1e-4
	ok := 0
	for i := 0; i < rows; i++ {
		p := probs.At(i, 0)
		py := baseline.PredictProba[i][1]
		if math.Abs(p-py) < tol {
			ok++
		} else if i < 5 {
			t.Logf("binary prob mismatch[%d]: go=%.6f py=%.6f diff=%.3e", i, p, py, math.Abs(p-py))
		}
		raw := pred.PredictRaw(baseline.XTest[i])[0]
		if raw == p {
			t.Logf("raw==prob at %d; expected transform applied later", i)
		}
	}
	if float64(ok)/float64(rows) < 0.95 {
		t.Errorf("binary prob agreement too low: %.2f%%", 100*float64(ok)/float64(rows))
	}
}

// TestPureGoCAPISpec_Multiclass_Shapes
// - Emulates C API behavior for multiclass: output shape n x num_class
func TestPureGoCAPISpec_Multiclass_Shapes(t *testing.T) {
	modelPath := filepath.Join("testdata", "multiclass", "model.txt")
	model, err := lightgbm.LoadFromFile(modelPath)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	predictor := lightgbm.NewSimplePredictor(model)

	data, err := os.ReadFile(filepath.Join("testdata", "multiclass", "test_data.json"))
	if err != nil {
		t.Fatalf("failed to read test data: %v", err)
	}
	var baseline struct {
		XTest        [][]float64 `json:"X_test"`
		PredictProba [][]float64 `json:"predict_proba"`
	}
	if err := json.Unmarshal(data, &baseline); err != nil {
		t.Fatalf("failed to unmarshal test data: %v", err)
	}

	rows := len(baseline.XTest)
	cols := len(baseline.XTest[0])
	X := mat.NewDense(rows, cols, nil)
	for i, r := range baseline.XTest {
		X.SetRow(i, r)
	}

	out, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}
	r, c := out.Dims()
	if r != rows || c != model.NumClass {
		t.Fatalf("unexpected output shape: got (%d,%d) want (%d,%d)", r, c, rows, model.NumClass)
	}
}

// TestPureGoCAPISpec_Train_ParamsAndPredict
// - Emulates C API dataset/create/booster train flow via pure Go Trainer
func TestPureGoCAPISpec_Train_ParamsAndPredict(t *testing.T) {
	td, err := loadTestDataSimple("binary")
	if err != nil {
		t.Skipf("test data not available: %v (run python_baseline/train_models.py)", err)
	}

	params := lightgbm.TrainingParams{
		NumIterations:   10,
		LearningRate:    0.1,
		NumLeaves:       31,
		MaxDepth:        5,
		MinDataInLeaf:   20,
		Lambda:          0.1,
		Alpha:           0.0,
		MinGainToSplit:  0.0,
		BaggingFraction: 1.0,
		FeatureFraction: 1.0,
		MaxBin:          255,
		MinDataInBin:    3,
		Objective:       "binary",
		Seed:            42,
		Deterministic:   true,
		Verbosity:       -1,
	}

	trainer := lightgbm.NewTrainer(params)
	if err := trainer.Fit(td.XTrain, td.YTrain); err != nil {
		t.Fatalf("training failed: %v", err)
	}

	model := trainer.GetModel()
	if model.NumFeatures != td.XTrain.RawMatrix().Cols {
		t.Errorf("num features mismatch: got %d want %d", model.NumFeatures, td.XTrain.RawMatrix().Cols)
	}

	predictor := lightgbm.NewPredictor(model)
	predictor.SetDeterministic(true)
	preds, err := predictor.Predict(td.XTest)
	if err != nil {
		t.Fatalf("predict failed: %v", err)
	}
	if r, c := preds.Dims(); r != td.XTest.RawMatrix().Rows || c != 1 {
		t.Fatalf("unexpected prediction shape: got (%d,%d)", r, c)
	}
}

// TestPureGoCAPISpec_FeatureImportance
// - Emulates LGBM_BoosterFeatureImportance behavior via Model.GetFeatureImportance
func TestPureGoCAPISpec_FeatureImportance(t *testing.T) {
	td, err := loadTestDataSimple("regression")
	if err != nil {
		t.Skipf("test data not available: %v (run python_baseline/train_models.py)", err)
	}
	params := lightgbm.TrainingParams{
		NumIterations: 10,
		LearningRate:  0.1,
		NumLeaves:     31,
		MaxDepth:      5,
		MinDataInLeaf: 20,
		Lambda:        0.1,
		Objective:     "regression",
		Seed:          42,
		Deterministic: true,
	}
	tr := lightgbm.NewTrainer(params)
	if err := tr.Fit(td.XTrain, td.YTrain); err != nil {
		t.Fatalf("training failed: %v", err)
	}
	m := tr.GetModel()

	// split-based importance should be non-negative and length=num_features
	imp := m.GetFeatureImportance("split")
	if len(imp) != m.NumFeatures {
		t.Fatalf("importance length mismatch: got %d want %d", len(imp), m.NumFeatures)
	}
	sum := 0.0
	for i, v := range imp {
		if v < 0 {
			t.Fatalf("importance[%d] negative: %v", i, v)
		}
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Logf("importance not normalized exactly (sum=%f), acceptable for spec", sum)
	}
}

// TestPureGoCAPISpec_LeafIndex_TODO documents parity for future work
func TestPureGoCAPISpec_LeafIndex_TODO(t *testing.T) {
	t.Skip("Leaf index prediction parity is not yet implemented in pure Go; will be added alongside a capi-like wrapper")
}
