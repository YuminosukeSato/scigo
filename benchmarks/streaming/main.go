package main

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"time"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/sklearn/cluster"
	"github.com/YuminosukeSato/scigo/sklearn/drift"
	"github.com/YuminosukeSato/scigo/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// BenchmarkResult はベンチマーク結果
type BenchmarkResult struct {
	Algorithm   string
	DatasetSize int
	Features    int
	Duration    time.Duration
	Throughput  float64 // samples/second
	MemoryUsage float64 // MB
	Accuracy    float64
	FinalLoss   float64
}

func main() {
	fmt.Println("=== GoML Streaming Learning Benchmarks ===")

	// メモリ統計の初期化
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	// ベンチマーク実行
	results := []BenchmarkResult{}

	// 1. SGDRegressor ベンチマーク
	fmt.Println("1. SGDRegressor Benchmarks")
	fmt.Println("-" + repeat("=", 49))
	sgdResults := benchmarkSGDRegressor()
	results = append(results, sgdResults...)

	// 2. SGDClassifier ベンチマーク
	fmt.Println("\n2. SGDClassifier Benchmarks")
	fmt.Println("-" + repeat("=", 49))
	sgdClassifierResults := benchmarkSGDClassifier()
	results = append(results, sgdClassifierResults...)

	// 3. PassiveAggressive ベンチマーク
	fmt.Println("\n3. PassiveAggressive Benchmarks")
	fmt.Println("-" + repeat("=", 49))
	paResults := benchmarkPassiveAggressive()
	results = append(results, paResults...)

	// 4. MiniBatchKMeans ベンチマーク
	fmt.Println("\n4. MiniBatchKMeans Benchmarks")
	fmt.Println("-" + repeat("=", 49))
	kmeansResults := benchmarkMiniBatchKMeans()
	results = append(results, kmeansResults...)

	// 5. ドリフト検出 ベンチマーク
	fmt.Println("\n5. Drift Detection Benchmarks")
	fmt.Println("-" + repeat("=", 49))
	driftResults := benchmarkDriftDetection()
	results = append(results, driftResults...)

	// メモリ使用量の測定
	runtime.GC()
	runtime.ReadMemStats(&m2)

	// 結果のサマリー
	fmt.Println("\n" + repeat("=", 80))
	fmt.Println("BENCHMARK SUMMARY")
	fmt.Println(repeat("=", 80))

	printResults(results)

	fmt.Printf("\nTotal Memory Used: %.2f MB\n", float64(m2.Alloc-m1.Alloc)/(1024*1024))
	fmt.Printf("System Memory Usage: %.2f MB\n", float64(m2.Sys)/(1024*1024))
}

// benchmarkSGDRegressor はSGDRegressorのベンチマーク
func benchmarkSGDRegressor() []BenchmarkResult {
	results := []BenchmarkResult{}

	datasets := []struct {
		samples  int
		features int
		name     string
	}{
		{1000, 10, "Small"},
		{10000, 20, "Medium"},
		{100000, 50, "Large"},
	}

	for _, dataset := range datasets {
		fmt.Printf("Dataset: %s (%d samples, %d features)\n", dataset.name, dataset.samples, dataset.features)

		// データ生成
		X, y := generateRegressionData(dataset.samples, dataset.features, 42)

		// バッチ学習
		batchResult := benchmarkSGDRegressorBatch(X, y, dataset.samples, dataset.features)
		results = append(results, batchResult)

		// オンライン学習
		onlineResult := benchmarkSGDRegressorOnline(X, y, dataset.samples, dataset.features)
		results = append(results, onlineResult)

		// ストリーミング学習
		streamResult := benchmarkSGDRegressorStreaming(X, y, dataset.samples, dataset.features)
		results = append(results, streamResult)

		fmt.Printf("  Batch:     %.0f samples/sec, %.2f accuracy\n", batchResult.Throughput, batchResult.Accuracy)
		fmt.Printf("  Online:    %.0f samples/sec, %.2f accuracy\n", onlineResult.Throughput, onlineResult.Accuracy)
		fmt.Printf("  Streaming: %.0f samples/sec, %.2f accuracy\n", streamResult.Throughput, streamResult.Accuracy)
		fmt.Println()
	}

	return results
}

func benchmarkSGDRegressorBatch(X, y mat.Matrix, samples, features int) BenchmarkResult {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	sgd := linear_model.NewSGDRegressor(
		linear_model.WithMaxIter(100),
		linear_model.WithRandomState(42),
	)

	start := time.Now()
	_ = sgd.Fit(X, y)
	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// 精度計算
	predictions, _ := sgd.Predict(X)
	accuracy := calculateR2Score(y, predictions)

	return BenchmarkResult{
		Algorithm:   "SGDRegressor (Batch)",
		DatasetSize: samples,
		Features:    features,
		Duration:    duration,
		Throughput:  float64(samples) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    accuracy,
		FinalLoss:   sgd.GetLoss(),
	}
}

func benchmarkSGDRegressorOnline(X, y mat.Matrix, samples, features int) BenchmarkResult {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	sgd := linear_model.NewSGDRegressor(
		linear_model.WithRandomState(42),
	)

	start := time.Now()

	// ミニバッチでの学習
	batchSize := 10
	for i := 0; i < samples; i += batchSize {
		end := i + batchSize
		if end > samples {
			end = samples
		}

		// Cast mat.Matrix to *mat.Dense for slicing
		XDense := X.(*mat.Dense)
		yDense := y.(*mat.Dense)
		XBatch := XDense.Slice(i, end, 0, features)
		yBatch := yDense.Slice(i, end, 0, 1)
		_ = sgd.PartialFit(XBatch, yBatch, nil)
	}

	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// 精度計算
	predictions, _ := sgd.Predict(X)
	accuracy := calculateR2Score(y, predictions)

	return BenchmarkResult{
		Algorithm:   "SGDRegressor (Online)",
		DatasetSize: samples,
		Features:    features,
		Duration:    duration,
		Throughput:  float64(samples) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    accuracy,
		FinalLoss:   sgd.GetLoss(),
	}
}

func benchmarkSGDRegressorStreaming(X, y mat.Matrix, samples, features int) BenchmarkResult {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	sgd := linear_model.NewSGDRegressor(
		linear_model.WithRandomState(42),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	dataChan := make(chan *model.Batch, 100)

	start := time.Now()

	// データ送信goroutine
	go func() {
		defer close(dataChan)
		batchSize := 10
		for i := 0; i < samples; i += batchSize {
			end := i + batchSize
			if end > samples {
				end = samples
			}

			// Cast mat.Matrix to *mat.Dense for slicing
			XDense := X.(*mat.Dense)
			yDense := y.(*mat.Dense)
			XBatch := XDense.Slice(i, end, 0, features)
			yBatch := yDense.Slice(i, end, 0, 1)

			select {
			case dataChan <- &model.Batch{X: XBatch, Y: yBatch}:
			case <-ctx.Done():
				return
			}
		}
	}()

	// ストリーミング学習
	if err := sgd.FitStream(ctx, dataChan); err != nil {
		fmt.Printf("Error in stream fit: %v\n", err)
		return BenchmarkResult{}
	}
	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// 精度計算
	predictions, _ := sgd.Predict(X)
	accuracy := calculateR2Score(y, predictions)

	return BenchmarkResult{
		Algorithm:   "SGDRegressor (Streaming)",
		DatasetSize: samples,
		Features:    features,
		Duration:    duration,
		Throughput:  float64(samples) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    accuracy,
		FinalLoss:   sgd.GetLoss(),
	}
}

// benchmarkSGDClassifier はSGDClassifierのベンチマーク
func benchmarkSGDClassifier() []BenchmarkResult {
	results := []BenchmarkResult{}

	datasets := []struct {
		samples  int
		features int
		name     string
	}{
		{1000, 10, "Small"},
		{10000, 20, "Medium"},
		{50000, 50, "Large"}, // メモリ制約により縮小
	}

	for _, dataset := range datasets {
		fmt.Printf("Dataset: %s (%d samples, %d features)\n", dataset.name, dataset.samples, dataset.features)

		// データ生成
		X, y := generateClassificationData(dataset.samples, dataset.features, 3, 42)

		// オンライン学習のみテスト（バッチは時間がかかりすぎる）
		result := benchmarkSGDClassifierOnline(X, y, dataset.samples, dataset.features)
		results = append(results, result)

		fmt.Printf("  Online: %.0f samples/sec, %.2f accuracy\n", result.Throughput, result.Accuracy)
		fmt.Println()
	}

	return results
}

func benchmarkSGDClassifierOnline(X, y mat.Matrix, samples, features int) BenchmarkResult {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	sgd := linear_model.NewSGDClassifier(
		linear_model.WithClassifierRandomState(42),
	)

	start := time.Now()

	// ミニバッチでの学習
	batchSize := 10
	for i := 0; i < samples; i += batchSize {
		end := i + batchSize
		if end > samples {
			end = samples
		}

		// Cast mat.Matrix to *mat.Dense for slicing
		XDense := X.(*mat.Dense)
		yDense := y.(*mat.Dense)
		XBatch := XDense.Slice(i, end, 0, features)
		yBatch := yDense.Slice(i, end, 0, 1)
		_ = sgd.PartialFit(XBatch, yBatch, nil)
	}

	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// 精度計算
	predictions, _ := sgd.Predict(X)
	accuracy := calculateAccuracy(y, predictions)

	return BenchmarkResult{
		Algorithm:   "SGDClassifier (Online)",
		DatasetSize: samples,
		Features:    features,
		Duration:    duration,
		Throughput:  float64(samples) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    accuracy,
		FinalLoss:   sgd.GetLoss(),
	}
}

// benchmarkPassiveAggressive はPassiveAggressiveのベンチマーク
func benchmarkPassiveAggressive() []BenchmarkResult {
	results := []BenchmarkResult{}

	// 回帰
	X, y := generateRegressionData(10000, 20, 42)
	pa := linear_model.NewPassiveAggressiveRegressor()

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start := time.Now()
	_ = pa.Fit(X, y)
	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	predictions, _ := pa.Predict(X)
	accuracy := calculateR2Score(y, predictions)

	results = append(results, BenchmarkResult{
		Algorithm:   "PassiveAggressiveRegressor",
		DatasetSize: 10000,
		Features:    20,
		Duration:    duration,
		Throughput:  float64(10000) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    accuracy,
	})

	fmt.Printf("PassiveAggressiveRegressor: %.0f samples/sec, %.2f accuracy\n",
		float64(10000)/duration.Seconds(), accuracy)

	return results
}

// benchmarkMiniBatchKMeans はMiniBatchKMeansのベンチマーク
func benchmarkMiniBatchKMeans() []BenchmarkResult {
	results := []BenchmarkResult{}

	X, _ := generateClusteringData(10000, 20, 5, 42)
	kmeans := cluster.NewMiniBatchKMeans(
		cluster.WithKMeansNClusters(5),
		cluster.WithKMeansBatchSize(100),
		cluster.WithKMeansRandomState(42),
	)

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start := time.Now()
	_ = kmeans.Fit(X, nil)
	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	results = append(results, BenchmarkResult{
		Algorithm:   "MiniBatchKMeans",
		DatasetSize: 10000,
		Features:    20,
		Duration:    duration,
		Throughput:  float64(10000) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    0, // クラスタリングなので精度なし
	})

	fmt.Printf("MiniBatchKMeans: %.0f samples/sec, inertia: %.2f\n",
		float64(10000)/duration.Seconds(), kmeans.Inertia())

	return results
}

// benchmarkDriftDetection はドリフト検出のベンチマーク
func benchmarkDriftDetection() []BenchmarkResult {
	results := []BenchmarkResult{}

	// DDM
	ddm := drift.NewDDM()
	// Use ChaCha8 instead of PCG for Go 1.25
	seedBytes := [32]byte{}
	seedBytes[0] = 42
	rng := rand.New(rand.NewChaCha8(seedBytes))

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start := time.Now()
	driftCount := 0

	for i := 0; i < 100000; i++ {
		// 最初の50000サンプルは低エラー率、後半は高エラー率
		errorProb := 0.05
		if i > 50000 {
			errorProb = 0.20
		}

		correct := rng.Float64() > errorProb
		result := ddm.Update(correct)

		if result.DriftDetected {
			driftCount++
		}
	}

	duration := time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	results = append(results, BenchmarkResult{
		Algorithm:   "DDM",
		DatasetSize: 100000,
		Features:    1,
		Duration:    duration,
		Throughput:  float64(100000) / duration.Seconds(),
		MemoryUsage: float64(m2.Alloc-m1.Alloc) / (1024 * 1024),
		Accuracy:    float64(driftCount), // ドリフト検出回数
	})

	fmt.Printf("DDM: %.0f samples/sec, %d drifts detected\n",
		float64(100000)/duration.Seconds(), driftCount)

	return results
}

// データ生成関数

func generateRegressionData(samples, features int, seed int64) (mat.Matrix, mat.Matrix) {
	// Use ChaCha8 instead of PCG for Go 1.25
	seedBytes := [32]byte{}
	seedBytes[0] = byte(seed)
	rng := rand.New(rand.NewChaCha8(seedBytes))

	X := mat.NewDense(samples, features, nil)
	y := mat.NewDense(samples, 1, nil)

	// 真の重み
	weights := make([]float64, features)
	for i := range weights {
		weights[i] = rng.NormFloat64()
	}

	for i := 0; i < samples; i++ {
		target := 0.0
		for j := 0; j < features; j++ {
			value := rng.NormFloat64()
			X.Set(i, j, value)
			target += weights[j] * value
		}
		// ノイズ追加
		target += rng.NormFloat64() * 0.1
		y.Set(i, 0, target)
	}

	return X, y
}

func generateClassificationData(samples, features, classes int, seed int64) (mat.Matrix, mat.Matrix) {
	// Use ChaCha8 instead of PCG for Go 1.25
	seedBytes := [32]byte{}
	seedBytes[0] = byte(seed)
	rng := rand.New(rand.NewChaCha8(seedBytes))

	X := mat.NewDense(samples, features, nil)
	y := mat.NewDense(samples, 1, nil)

	for i := 0; i < samples; i++ {
		class := rng.IntN(classes)
		y.Set(i, 0, float64(class))

		for j := 0; j < features; j++ {
			// クラスに依存した特徴量
			value := rng.NormFloat64() + float64(class)*0.5
			X.Set(i, j, value)
		}
	}

	return X, y
}

func generateClusteringData(samples, features, clusters int, seed int64) (mat.Matrix, mat.Matrix) {
	// Use ChaCha8 instead of PCG for Go 1.25
	seedBytes := [32]byte{}
	seedBytes[0] = byte(seed)
	rng := rand.New(rand.NewChaCha8(seedBytes))

	X := mat.NewDense(samples, features, nil)

	// クラスタ中心
	centers := make([][]float64, clusters)
	for c := 0; c < clusters; c++ {
		centers[c] = make([]float64, features)
		for j := 0; j < features; j++ {
			centers[c][j] = rng.NormFloat64() * 5
		}
	}

	for i := 0; i < samples; i++ {
		cluster := rng.IntN(clusters)
		for j := 0; j < features; j++ {
			value := centers[cluster][j] + rng.NormFloat64()
			X.Set(i, j, value)
		}
	}

	// Generate labels based on cluster assignment
	y := mat.NewDense(samples, 1, nil)
	for i := 0; i < samples; i++ {
		// Recalculate cluster for labeling (simplified approach)
		y.Set(i, 0, float64(i%clusters))
	}
	return X, y
}

// 評価関数

func calculateR2Score(yTrue, yPred mat.Matrix) float64 {
	rows, _ := yTrue.Dims()

	// 平均値計算
	var yMean float64
	for i := 0; i < rows; i++ {
		yMean += yTrue.At(i, 0)
	}
	yMean /= float64(rows)

	// SS_tot と SS_res 計算
	var ssTot, ssRes float64
	for i := 0; i < rows; i++ {
		yi := yTrue.At(i, 0)
		predi := yPred.At(i, 0)

		ssTot += (yi - yMean) * (yi - yMean)
		ssRes += (yi - predi) * (yi - predi)
	}

	if ssTot == 0 {
		return 0
	}

	return 1.0 - (ssRes / ssTot)
}

func calculateAccuracy(yTrue, yPred mat.Matrix) float64 {
	rows, _ := yTrue.Dims()
	correct := 0

	for i := 0; i < rows; i++ {
		if math.Abs(yTrue.At(i, 0)-yPred.At(i, 0)) < 1e-10 {
			correct++
		}
	}

	return float64(correct) / float64(rows)
}

// 結果表示

func printResults(results []BenchmarkResult) {
	fmt.Printf("%-30s %10s %8s %12s %15s %10s %10s\n",
		"Algorithm", "Samples", "Features", "Duration", "Throughput", "Memory", "Accuracy")
	fmt.Println(repeat("-", 100))

	for _, result := range results {
		fmt.Printf("%-30s %10d %8d %12s %15.0f %10.2f %10.3f\n",
			result.Algorithm,
			result.DatasetSize,
			result.Features,
			result.Duration.Truncate(time.Millisecond),
			result.Throughput,
			result.MemoryUsage,
			result.Accuracy)
	}
}

// 補助関数
func repeat(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
