package preprocessing_test

import (
	"math"
	"testing"

	"github.com/YuminosukeSato/scigo/preprocessing"
	"gonum.org/v1/gonum/mat"
)

const epsilon = 1e-10 // Tolerance for floating-point comparisons

func TestStandardScaler_BasicFunctionality(t *testing.T) {
	// Test data: 3 samples, 2 features
	// Feature 1: [1, 2, 3] -> mean=2, std=0.816
	// Feature 2: [4, 5, 6] -> mean=5, std=0.816
	data := []float64{
		1.0, 4.0,
		2.0, 5.0,
		3.0, 6.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewStandardScalerDefault()

	// Fit
	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Verify statistics
	expectedMean := []float64{2.0, 5.0}
	expectedStd := []float64{0.816496580927726, 0.816496580927726}

	if len(scaler.Mean) != 2 {
		t.Errorf("Expected 2 means, got %d", len(scaler.Mean))
	}

	for i, expected := range expectedMean {
		if math.Abs(scaler.Mean[i]-expected) > epsilon {
			t.Errorf("Mean[%d]: expected %f, got %f", i, expected, scaler.Mean[i])
		}
	}

	for i, expected := range expectedStd {
		if math.Abs(scaler.Scale[i]-expected) > epsilon {
			t.Errorf("Scale[%d]: expected %f, got %f", i, expected, scaler.Scale[i])
		}
	}

	// Transform
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// 標準化後のデータ確認
	// Feature 1: [(1-2)/0.816, (2-2)/0.816, (3-2)/0.816] = [-1.225, 0, 1.225]
	// Feature 2: [(4-5)/0.816, (5-5)/0.816, (6-5)/0.816] = [-1.225, 0, 1.225]
	expectedScaled := []float64{
		-1.224744871391589, -1.224744871391589,
		0.0, 0.0,
		1.224744871391589, 1.224744871391589,
	}

	r, c := XScaled.Dims()
	if r != 3 || c != 2 {
		t.Fatalf("Expected 3x2 matrix, got %dx%d", r, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := XScaled.At(i, j)
			expected := expectedScaled[i*c+j]
			if math.Abs(actual-expected) > epsilon {
				t.Errorf("XScaled[%d][%d]: expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestStandardScaler_FitTransform(t *testing.T) {
	data := []float64{
		10.0, 100.0,
		20.0, 200.0,
		30.0, 300.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewStandardScalerDefault()

	// FitTransform
	XScaled, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// 分離したFit + Transformと結果が同じか確認
	scaler2 := preprocessing.NewStandardScalerDefault()
	_ = scaler2.Fit(X)
	XScaled2, _ := scaler2.Transform(X)

	r, c := XScaled.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val1 := XScaled.At(i, j)
			val2 := XScaled2.At(i, j)
			if math.Abs(val1-val2) > epsilon {
				t.Errorf("FitTransform vs Fit+Transform differ at [%d][%d]: %f vs %f", i, j, val1, val2)
			}
		}
	}
}

func TestStandardScaler_InverseTransform(t *testing.T) {
	data := []float64{
		1.0, 10.0,
		2.0, 20.0,
		3.0, 30.0,
		4.0, 40.0,
	}
	X := mat.NewDense(4, 2, data)

	scaler := preprocessing.NewStandardScalerDefault()

	// Fit and Transform
	XScaled, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Inverse Transform
	XRecovered, err := scaler.InverseTransform(XScaled)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// 元のデータと復元されたデータが一致するか確認
	r, c := X.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			original := X.At(i, j)
			recovered := XRecovered.At(i, j)
			if math.Abs(original-recovered) > epsilon {
				t.Errorf("InverseTransform failed at [%d][%d]: expected %f, got %f", i, j, original, recovered)
			}
		}
	}
}

func TestStandardScaler_WithMeanFalse(t *testing.T) {
	data := []float64{
		1.0, 10.0,
		2.0, 20.0,
		3.0, 30.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewStandardScaler(false, true) // with_mean=False, with_std=True

	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// 平均は0に設定されているべき
	for i, mean := range scaler.Mean {
		if math.Abs(mean-0.0) > epsilon {
			t.Errorf("Mean[%d] should be 0.0 when with_mean=False, got %f", i, mean)
		}
	}

	// Transform
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// with_mean=Falseの場合、平均を引かずに標準偏差で割るだけ
	// Feature 1 std = sqrt(((1-2)² + (2-2)² + (3-2)²)/3) = sqrt(2/3) ≈ 0.816
	// でも with_mean=False なので、mean=0として計算
	// std = sqrt((1² + 2² + 3²)/3) = sqrt(14/3) ≈ 2.160
	expectedStdNoMean := math.Sqrt((1.0*1.0 + 2.0*2.0 + 3.0*3.0) / 3.0) // ≈ 2.160
	expectedScaled0 := 1.0 / expectedStdNoMean

	actual := XScaled.At(0, 0)
	if math.Abs(actual-expectedScaled0) > epsilon {
		t.Errorf("First element: expected %f, got %f", expectedScaled0, actual)
	}
}

func TestStandardScaler_WithStdFalse(t *testing.T) {
	data := []float64{
		1.0, 10.0,
		2.0, 20.0,
		3.0, 30.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewStandardScaler(true, false) // with_mean=True, with_std=False

	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// スケールは1に設定されているべき
	for i, scale := range scaler.Scale {
		if math.Abs(scale-1.0) > epsilon {
			t.Errorf("Scale[%d] should be 1.0 when with_std=False, got %f", i, scale)
		}
	}

	// Transform
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// with_std=Falseの場合、平均を引くだけで標準偏差では割らない
	// 平均: Feature1=2, Feature2=20
	expectedValues := []float64{
		1.0 - 2.0, 10.0 - 20.0, // [-1, -10]
		0.0, 0.0, // [0, 0] (2.0 - 2.0 = 0, 20.0 - 20.0 = 0)
		3.0 - 2.0, 30.0 - 20.0, // [1, 10]
	}

	r, c := XScaled.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := XScaled.At(i, j)
			expected := expectedValues[i*c+j]
			if math.Abs(actual-expected) > epsilon {
				t.Errorf("XScaled[%d][%d]: expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestStandardScaler_ErrorCases(t *testing.T) {
	scaler := preprocessing.NewStandardScalerDefault()

	// 未学習状態でTransform
	data := []float64{1.0, 2.0}
	X := mat.NewDense(1, 2, data)

	_, err := scaler.Transform(X)
	if err == nil {
		t.Error("Expected error for unfitted scaler, got nil")
	}

	// 未学習状態でInverseTransform
	_, err = scaler.InverseTransform(X)
	if err == nil {
		t.Error("Expected error for unfitted scaler, got nil")
	}

	// 特徴量数の不一致
	_ = scaler.Fit(X) // 2特徴量で学習
	wrongData := []float64{1.0, 2.0, 3.0}
	XWrong := mat.NewDense(1, 3, wrongData) // 3特徴量

	_, err = scaler.Transform(XWrong)
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

func TestStandardScaler_EmptyDataError(t *testing.T) {
	scaler := preprocessing.NewStandardScalerDefault()

	// スケーラーの実装をテストするためのカスタムMatrixを作成
	// 0x0の次元を返すモック
	emptyMatrix := &mockMatrix{rows: 0, cols: 0}

	err := scaler.Fit(emptyMatrix)
	if err == nil {
		t.Error("Expected error for empty data, got nil")
	}
}

// テスト用のモックMatrix
type mockMatrix struct {
	rows, cols int
	data       []float64
}

func (m *mockMatrix) Dims() (int, int) {
	return m.rows, m.cols
}

func (m *mockMatrix) At(i, j int) float64 {
	if m.data == nil {
		return 0
	}
	return m.data[i*m.cols+j]
}

func (m *mockMatrix) T() mat.Matrix {
	return m // 転置は実装しない（テスト用）
}

func TestStandardScaler_ConstantFeature(t *testing.T) {
	// 定数特徴量（分散が0）のテスト
	data := []float64{
		5.0, 1.0,
		5.0, 2.0,
		5.0, 3.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewStandardScalerDefault()
	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// 第1特徴量の標準偏差は0なので、1.0に設定されるべき
	if math.Abs(scaler.Scale[0]-1.0) > epsilon {
		t.Errorf("Scale[0] should be 1.0 for constant feature, got %f", scaler.Scale[0])
	}

	// Transform後、定数特徴量は(5-5)/1 = 0になるべき
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	for i := 0; i < 3; i++ {
		val := XScaled.At(i, 0)
		if math.Abs(val-0.0) > epsilon {
			t.Errorf("Constant feature should be 0 after scaling, got %f at row %d", val, i)
		}
	}
}

func TestStandardScaler_GetParams(t *testing.T) {
	scaler := preprocessing.NewStandardScaler(true, false)
	params := scaler.GetParams()

	if params["with_mean"] != true {
		t.Errorf("Expected with_mean=true, got %v", params["with_mean"])
	}

	if params["with_std"] != false {
		t.Errorf("Expected with_std=false, got %v", params["with_std"])
	}
}

func TestStandardScaler_String(t *testing.T) {
	scaler := preprocessing.NewStandardScaler(true, false)

	// 未学習状態
	str := scaler.String()
	expected := "StandardScaler(with_mean=true, with_std=false)"
	if str != expected {
		t.Errorf("Expected %q, got %q", expected, str)
	}

	// 学習後
	data := []float64{1.0, 2.0, 3.0, 4.0}
	X := mat.NewDense(2, 2, data)
	_ = scaler.Fit(X)

	str = scaler.String()
	expected = "StandardScaler(with_mean=true, with_std=false, n_features=2)"
	if str != expected {
		t.Errorf("Expected %q, got %q", expected, str)
	}
}

// MinMaxScaler Tests

func TestMinMaxScaler_BasicFunctionality(t *testing.T) {
	// テストデータ: [1,4], [2,5], [3,6]
	// Feature 1: min=1, max=3, range=2, scaled to [0,1] -> [0, 0.5, 1]
	// Feature 2: min=4, max=6, range=2, scaled to [0,1] -> [0, 0.5, 1]
	data := []float64{
		1.0, 4.0,
		2.0, 5.0,
		3.0, 6.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewMinMaxScalerDefault()

	// Fit
	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Verify statistics
	expectedDataMin := []float64{1.0, 4.0}
	expectedDataMax := []float64{3.0, 6.0}
	expectedScale := []float64{2.0, 2.0}

	for i, expected := range expectedDataMin {
		if math.Abs(scaler.DataMin[i]-expected) > epsilon {
			t.Errorf("DataMin[%d]: expected %f, got %f", i, expected, scaler.DataMin[i])
		}
	}

	for i, expected := range expectedDataMax {
		if math.Abs(scaler.DataMax[i]-expected) > epsilon {
			t.Errorf("DataMax[%d]: expected %f, got %f", i, expected, scaler.DataMax[i])
		}
	}

	for i, expected := range expectedScale {
		if math.Abs(scaler.Scale[i]-expected) > epsilon {
			t.Errorf("Scale[%d]: expected %f, got %f", i, expected, scaler.Scale[i])
		}
	}

	// Transform
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// 期待される結果:
	// [1,4] -> [(1-1)/(3-1)*1+0, (4-4)/(6-4)*1+0] = [0, 0]
	// [2,5] -> [(2-1)/(3-1)*1+0, (5-4)/(6-4)*1+0] = [0.5, 0.5]
	// [3,6] -> [(3-1)/(3-1)*1+0, (6-4)/(6-4)*1+0] = [1, 1]
	expectedScaled := []float64{
		0.0, 0.0,
		0.5, 0.5,
		1.0, 1.0,
	}

	r, c := XScaled.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := XScaled.At(i, j)
			expected := expectedScaled[i*c+j]
			if math.Abs(actual-expected) > epsilon {
				t.Errorf("XScaled[%d][%d]: expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestMinMaxScaler_CustomRange(t *testing.T) {
	// [-1, 1] の範囲にスケーリング
	data := []float64{
		10.0, 100.0,
		20.0, 200.0,
		30.0, 300.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewMinMaxScaler([2]float64{-1.0, 1.0})

	XScaled, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Feature 1: min=10, max=30, range=20
	// Feature 2: min=100, max=300, range=200
	// 期待される結果 (範囲 [-1, 1]):
	// [10,100] -> [(10-10)/20*2-1, (100-100)/200*2-1] = [-1, -1]
	// [20,200] -> [(20-10)/20*2-1, (200-100)/200*2-1] = [0, 0]
	// [30,300] -> [(30-10)/20*2-1, (300-100)/200*2-1] = [1, 1]
	expectedScaled := []float64{
		-1.0, -1.0,
		0.0, 0.0,
		1.0, 1.0,
	}

	r, c := XScaled.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := XScaled.At(i, j)
			expected := expectedScaled[i*c+j]
			if math.Abs(actual-expected) > epsilon {
				t.Errorf("XScaled[%d][%d]: expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestMinMaxScaler_InverseTransform(t *testing.T) {
	data := []float64{
		5.0, 50.0,
		10.0, 100.0,
		15.0, 150.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewMinMaxScalerDefault()

	// Fit and Transform
	XScaled, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Inverse Transform
	XRecovered, err := scaler.InverseTransform(XScaled)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// 元のデータと復元されたデータが一致するか確認
	r, c := X.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			original := X.At(i, j)
			recovered := XRecovered.At(i, j)
			if math.Abs(original-recovered) > epsilon {
				t.Errorf("InverseTransform failed at [%d][%d]: expected %f, got %f", i, j, original, recovered)
			}
		}
	}
}

func TestMinMaxScaler_ConstantFeature(t *testing.T) {
	// 定数特徴量のテスト
	data := []float64{
		5.0, 1.0,
		5.0, 2.0,
		5.0, 3.0,
	}
	X := mat.NewDense(3, 2, data)

	scaler := preprocessing.NewMinMaxScalerDefault()
	err := scaler.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// 第1特徴量の範囲は0なので、スケールが1.0に設定されるべき
	if math.Abs(scaler.Scale[0]-1.0) > epsilon {
		t.Errorf("Scale[0] should be 1.0 for constant feature, got %f", scaler.Scale[0])
	}

	// Transform後、定数特徴量は適切にスケーリングされるべき
	XScaled, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// 定数特徴量は全て同じ値（この場合は0）になるべき
	for i := 0; i < 3; i++ {
		val := XScaled.At(i, 0)
		if math.Abs(val-0.0) > epsilon {
			t.Errorf("Constant feature should be 0 after scaling, got %f at row %d", val, i)
		}
	}
}

func TestMinMaxScaler_ErrorCases(t *testing.T) {
	scaler := preprocessing.NewMinMaxScalerDefault()

	// 未学習状態でTransform
	data := []float64{1.0, 2.0}
	X := mat.NewDense(1, 2, data)

	_, err := scaler.Transform(X)
	if err == nil {
		t.Error("Expected error for unfitted scaler, got nil")
	}

	// 未学習状態でInverseTransform
	_, err = scaler.InverseTransform(X)
	if err == nil {
		t.Error("Expected error for unfitted scaler, got nil")
	}

	// 特徴量数の不一致
	_ = scaler.Fit(X) // 2特徴量で学習
	wrongData := []float64{1.0, 2.0, 3.0}
	XWrong := mat.NewDense(1, 3, wrongData) // 3特徴量

	_, err = scaler.Transform(XWrong)
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

func TestMinMaxScaler_GetParams(t *testing.T) {
	scaler := preprocessing.NewMinMaxScaler([2]float64{-2.0, 2.0})
	params := scaler.GetParams()

	featureRange := params["feature_range"].([2]float64)
	expectedRange := [2]float64{-2.0, 2.0}

	if featureRange[0] != expectedRange[0] || featureRange[1] != expectedRange[1] {
		t.Errorf("Expected feature_range=%v, got %v", expectedRange, featureRange)
	}
}

func TestMinMaxScaler_String(t *testing.T) {
	scaler := preprocessing.NewMinMaxScaler([2]float64{-1.0, 2.0})

	// 未学習状態
	str := scaler.String()
	expected := "MinMaxScaler(feature_range=[-1.0, 2.0])"
	if str != expected {
		t.Errorf("Expected %q, got %q", expected, str)
	}

	// 学習後
	data := []float64{1.0, 2.0, 3.0, 4.0}
	X := mat.NewDense(2, 2, data)
	_ = scaler.Fit(X)

	str = scaler.String()
	expected = "MinMaxScaler(feature_range=[-1.0, 2.0], n_features=2)"
	if str != expected {
		t.Errorf("Expected %q, got %q", expected, str)
	}
}
