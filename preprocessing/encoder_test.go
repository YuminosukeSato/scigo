package preprocessing_test

import (
	"testing"

	"github.com/YuminosukeSato/scigo/preprocessing"
)

func TestOneHotEncoder_Fit(t *testing.T) {
	// 基本的なFit動作のテスト
	data := [][]string{
		{"cat", "red"},
		{"dog", "blue"},
		{"cat", "red"},
		{"fish", "green"},
	}
	
	encoder := preprocessing.NewOneHotEncoder()
	
	err := encoder.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// 学習済み状態になっているかチェック
	if !encoder.IsFitted() {
		t.Error("Encoder should be fitted after Fit()")
	}
	
	// 特徴量数のチェック
	if encoder.NFeatures != 2 {
		t.Errorf("Expected NFeatures=2, got %d", encoder.NFeatures)
	}
	
	// カテゴリが正しく学習されているかチェック
	expectedCategories := [][]string{
		{"cat", "dog", "fish"}, // 第1特徴量のカテゴリ（ソート済み）
		{"blue", "green", "red"}, // 第2特徴量のカテゴリ（ソート済み）
	}
	
	if len(encoder.Categories) != 2 {
		t.Fatalf("Expected 2 feature categories, got %d", len(encoder.Categories))
	}
	
	for i, expectedCats := range expectedCategories {
		if len(encoder.Categories[i]) != len(expectedCats) {
			t.Errorf("Feature %d: expected %d categories, got %d", 
				i, len(expectedCats), len(encoder.Categories[i]))
			continue
		}
		
		for j, expectedCat := range expectedCats {
			if encoder.Categories[i][j] != expectedCat {
				t.Errorf("Feature %d, category %d: expected %s, got %s", 
					i, j, expectedCat, encoder.Categories[i][j])
			}
		}
	}
	
	// 出力特徴量数のチェック（3 + 3 = 6）
	if encoder.NOutputs != 6 {
		t.Errorf("Expected NOutputs=6, got %d", encoder.NOutputs)
	}
}

func TestOneHotEncoder_Transform_Basic(t *testing.T) {
	// 基本的なTransform動作のテスト
	trainData := [][]string{
		{"cat", "red"},
		{"dog", "blue"},
		{"fish", "green"},
	}
	
	encoder := preprocessing.NewOneHotEncoder()
	err := encoder.Fit(trainData)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// 変換対象データ
	testData := [][]string{
		{"cat", "red"},   // [1,0,0,0,0,1]
		{"dog", "blue"},  // [0,1,0,1,0,0]
		{"fish", "green"}, // [0,0,1,0,1,0]
	}
	
	result, err := encoder.Transform(testData)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	
	// 結果の次元チェック
	r, c := result.Dims()
	if r != 3 || c != 6 {
		t.Fatalf("Expected 3x6 matrix, got %dx%d", r, c)
	}
	
	// 期待される結果
	// カテゴリ順: ["cat","dog","fish"] x ["blue","green","red"]
	expected := [][]float64{
		{1, 0, 0, 0, 0, 1}, // cat, red
		{0, 1, 0, 1, 0, 0}, // dog, blue
		{0, 0, 1, 0, 1, 0}, // fish, green
	}
	
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := result.At(i, j)
			expectedVal := expected[i][j]
			if actual != expectedVal {
				t.Errorf("Result[%d][%d]: expected %f, got %f", i, j, expectedVal, actual)
			}
		}
	}
}

func TestOneHotEncoder_UnfittedError(t *testing.T) {
	encoder := preprocessing.NewOneHotEncoder()
	
	testData := [][]string{{"cat", "red"}}
	
	_, err := encoder.Transform(testData)
	if err == nil {
		t.Error("Expected error for unfitted encoder, got nil")
	}
}

func TestOneHotEncoder_EmptyDataError(t *testing.T) {
	encoder := preprocessing.NewOneHotEncoder()
	
	// 空データでFit
	emptyData := [][]string{}
	err := encoder.Fit(emptyData)
	if err == nil {
		t.Error("Expected error for empty data, got nil")
	}
}

func TestOneHotEncoder_FitTransform(t *testing.T) {
	// FitTransformがFit+Transformと同じ結果になることを確認
	data := [][]string{
		{"A", "X"},
		{"B", "Y"},
		{"A", "X"},
	}
	
	encoder1 := preprocessing.NewOneHotEncoder()
	result1, err := encoder1.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}
	
	encoder2 := preprocessing.NewOneHotEncoder()
	err = encoder2.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	result2, err := encoder2.Transform(data)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	
	// 結果が同じかチェック
	r1, c1 := result1.Dims()
	r2, c2 := result2.Dims()
	
	if r1 != r2 || c1 != c2 {
		t.Fatalf("Dimension mismatch: FitTransform %dx%d vs Fit+Transform %dx%d", 
			r1, c1, r2, c2)
	}
	
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			val1 := result1.At(i, j)
			val2 := result2.At(i, j)
			if val1 != val2 {
				t.Errorf("Result[%d][%d]: FitTransform %f vs Fit+Transform %f", 
					i, j, val1, val2)
			}
		}
	}
}

func TestOneHotEncoder_UnknownCategory(t *testing.T) {
	// 学習データ
	trainData := [][]string{
		{"cat", "red"},
		{"dog", "blue"},
	}
	
	encoder := preprocessing.NewOneHotEncoder()
	err := encoder.Fit(trainData)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// テストデータに未知カテゴリを含む
	testData := [][]string{
		{"cat", "red"},     // 既知 [1,0,0,1]
		{"fish", "yellow"}, // 未知 [0,0,0,0]
		{"dog", "blue"},    // 既知 [0,1,1,0]
	}
	
	result, err := encoder.Transform(testData)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	
	// 結果の次元チェック (4出力: cat,dog x blue,red)
	r, c := result.Dims()
	if r != 3 || c != 4 {
		t.Fatalf("Expected 3x4 matrix, got %dx%d", r, c)
	}
	
	// 期待される結果: 未知カテゴリは全て0
	expected := [][]float64{
		{1, 0, 0, 1}, // cat, red
		{0, 0, 0, 0}, // fish(未知), yellow(未知) → 全て0
		{0, 1, 1, 0}, // dog, blue
	}
	
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			actual := result.At(i, j)
			expectedVal := expected[i][j]
			if actual != expectedVal {
				t.Errorf("Result[%d][%d]: expected %f, got %f", i, j, expectedVal, actual)
			}
		}
	}
}

func TestOneHotEncoder_DimensionMismatch(t *testing.T) {
	encoder := preprocessing.NewOneHotEncoder()
	
	// 2特徴量で学習
	trainData := [][]string{
		{"A", "X"},
		{"B", "Y"},
	}
	err := encoder.Fit(trainData)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// 異なる特徴量数でTransform
	testData := [][]string{
		{"A", "X", "Z"}, // 3特徴量
	}
	
	_, err = encoder.Transform(testData)
	if err == nil {
		t.Error("Expected error for dimension mismatch, got nil")
	}
}

func TestOneHotEncoder_GetFeatureNamesOut(t *testing.T) {
	encoder := preprocessing.NewOneHotEncoder()
	
	// 学習データ
	data := [][]string{
		{"cat", "small"},
		{"dog", "large"},
		{"bird", "small"},
	}
	
	err := encoder.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// デフォルトの特徴量名（x0, x1, ...）
	names := encoder.GetFeatureNamesOut(nil)
	expected := []string{
		"x0_bird", "x0_cat", "x0_dog",  // 動物（ソート済み）
		"x1_large", "x1_small",         // サイズ（ソート済み）
	}
	
	if len(names) != len(expected) {
		t.Fatalf("Expected %d feature names, got %d", len(expected), len(names))
	}
	
	for i, expectedName := range expected {
		if names[i] != expectedName {
			t.Errorf("Feature name[%d]: expected %s, got %s", i, expectedName, names[i])
		}
	}
	
	// カスタム特徴量名
	inputFeatures := []string{"animal", "size"}
	customNames := encoder.GetFeatureNamesOut(inputFeatures)
	expectedCustom := []string{
		"animal_bird", "animal_cat", "animal_dog",
		"size_large", "size_small",
	}
	
	if len(customNames) != len(expectedCustom) {
		t.Fatalf("Expected %d custom feature names, got %d", len(expectedCustom), len(customNames))
	}
	
	for i, expectedName := range expectedCustom {
		if customNames[i] != expectedName {
			t.Errorf("Custom feature name[%d]: expected %s, got %s", i, expectedName, customNames[i])
		}
	}
}

func TestOneHotEncoder_GetFeatureNamesOut_Unfitted(t *testing.T) {
	encoder := preprocessing.NewOneHotEncoder()
	
	// 未学習状態でGetFeatureNamesOut呼び出し
	names := encoder.GetFeatureNamesOut(nil)
	if names != nil {
		t.Errorf("Expected nil for unfitted encoder, got %v", names)
	}
}