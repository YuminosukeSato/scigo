package linear

import (
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// createBenchmarkData はベンチマーク用のデータを生成する
func createBenchmarkData(rows, cols int) (*mat.Dense, *mat.Dense) {
	// シードを固定して再現性を確保
	rng := rand.New(rand.NewPCG(42, 42))

	// X: rows x cols の行列（ランダムな値を生成）
	X := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// -1.0 から 1.0 の範囲のランダムな値
			X.Set(i, j, rng.Float64()*2.0-1.0)
		}
	}

	// 真の重みベクトルを生成
	trueWeights := make([]float64, cols)
	for j := 0; j < cols; j++ {
		trueWeights[j] = float64(j+1) * 0.5
	}

	// y: rows x 1 の列ベクトル（y = X * weights + 小さなノイズ）
	y := mat.NewDense(rows, 1, nil)
	for i := 0; i < rows; i++ {
		sum := 1.0 // 切片
		for j := 0; j < cols; j++ {
			sum += X.At(i, j) * trueWeights[j]
		}
		// 小さなノイズを追加
		sum += (rng.Float64() - 0.5) * 0.1
		y.Set(i, 0, sum)
	}

	return X, y
}

// BenchmarkLinearRegressionFit はFitメソッドのベンチマークを実行する
func BenchmarkLinearRegressionFit(b *testing.B) {
	// 様々なサイズでベンチマークを実行
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"Small_100x10", 100, 10},
		{"Small_500x10", 500, 10},
		{"Medium_1000x10", 1000, 10}, // 並列処理の閾値
		{"Medium_2000x10", 2000, 10},
		{"Large_5000x20", 5000, 20},
		{"Large_10000x20", 10000, 20},
		{"XLarge_20000x50", 20000, 50},
		{"XLarge_50000x50", 50000, 50},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			X, y := createBenchmarkData(size.rows, size.cols)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				lr := NewLinearRegression()
				err := lr.Fit(X, y)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkLinearRegressionFitSequential は逐次処理版のベンチマーク（比較用）
func BenchmarkLinearRegressionFitSequential(b *testing.B) {
	// LinearRegressionの逐次処理版を作成して比較するために、
	// 並列処理を無効化した版を別途作成する必要があるが、
	// ここでは閾値以下のサイズでテストすることで逐次処理を測定
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"Sequential_100x10", 100, 10},
		{"Sequential_500x10", 500, 10},
		{"Sequential_900x10", 900, 10}, // 閾値(1000)未満
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			X, y := createBenchmarkData(size.rows, size.cols)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				lr := NewLinearRegression()
				err := lr.Fit(X, y)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMatrixCopy は行列コピー部分のみのベンチマーク
func BenchmarkMatrixCopy(b *testing.B) {
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"Copy_1000x10", 1000, 10},
		{"Copy_5000x20", 5000, 20},
		{"Copy_10000x20", 10000, 20},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			X, _ := createBenchmarkData(size.rows, size.cols)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// 行列コピー部分のみを測定
				XWithIntercept := mat.NewDense(size.rows, size.cols+1, nil)
				for i := 0; i < size.rows; i++ {
					XWithIntercept.Set(i, 0, 1.0)
					for j := 0; j < size.cols; j++ {
						XWithIntercept.Set(i, j+1, X.At(i, j))
					}
				}
			}
		})
	}
}
