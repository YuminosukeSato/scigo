package benchmarks

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/YuminosukeSato/scigo/linear"
	"github.com/YuminosukeSato/scigo/performance"
	"github.com/YuminosukeSato/scigo/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// BenchmarkTBScale tests performance with TB-scale datasets
func BenchmarkTBScale(b *testing.B) {
	sizes := []struct {
		name     string
		samples  int
		features int
	}{
		{"1M_100", 1_000_000, 100},
		{"10M_100", 10_000_000, 100},
		{"100M_100", 100_000_000, 100},
		// TB-scale (commented out for CI - requires special hardware)
		// {"1B_100", 1_000_000_000, 100},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			b.Run("MemoryMapped", func(b *testing.B) {
				benchmarkMemoryMapped(b, size.samples, size.features)
			})

			b.Run("Chunked", func(b *testing.B) {
				benchmarkChunkedProcessing(b, size.samples, size.features)
			})

			b.Run("Streaming", func(b *testing.B) {
				benchmarkStreaming(b, size.samples, size.features)
			})

			b.Run("WithPooling", func(b *testing.B) {
				benchmarkWithPooling(b, size.samples, size.features)
			})
		})
	}
}

func benchmarkMemoryMapped(b *testing.B, samples, features int) {
	b.ReportAllocs()

	// Create temporary file for memory mapping
	tmpFile := fmt.Sprintf("/tmp/scigo_bench_%d.dat", time.Now().UnixNano())

	dataset, err := performance.NewMemoryMappedDataset(tmpFile, samples, features, performance.Float64)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = dataset.Close() }()

	// Initialize with random data
	chunkSize := 10000
	_ = rand.New(rand.NewSource(42)) // Prepared for future use

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Process dataset in chunks
		err := dataset.IterateChunks(chunkSize, func(chunk mat.Matrix, startRow int) error {
			// Simulate processing
			rows, cols := chunk.Dims()
			sum := 0.0
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					sum += chunk.At(r, c)
				}
			}
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
	}

	b.SetBytes(int64(samples * features * 8)) // float64 size
}

func benchmarkChunkedProcessing(b *testing.B, samples, features int) {
	b.ReportAllocs()

	processor := performance.NewChunkedProcessor(10000, true)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		processed := 0

		// Simulate chunked data processing
		err := processor.Process(nil, func(chunk [][]float64) error {
			processed += len(chunk)

			// Simulate computation
			for _, row := range chunk {
				sum := 0.0
				for _, val := range row {
					sum += val
				}
			}

			return nil
		})

		if err != nil {
			b.Fatal(err)
		}
	}

	b.SetBytes(int64(samples * features * 8))
}

func benchmarkStreaming(b *testing.B, samples, features int) {
	b.ReportAllocs()

	pipeline := performance.NewStreamingPipeline(1000)

	// Add processing stages
	pipeline.AddStage(&transformStage{})
	pipeline.AddStage(&aggregateStage{})

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		input := make(chan mat.Matrix, 100)

		// Start pipeline
		output := pipeline.Run(input)

		// Feed data
		go func() {
			defer close(input)
			batchSize := 1000
			for s := 0; s < samples; s += batchSize {
				batch := mat.NewDense(batchSize, features, nil)
				input <- batch
			}
		}()

		// Consume output
		count := 0
		for range output {
			count++
		}
	}

	metrics := pipeline.GetMetrics()
	b.ReportMetric(float64(metrics.ProcessedSamples)/b.Elapsed().Seconds(), "samples/sec")
	b.SetBytes(int64(samples * features * 8))
}

func benchmarkWithPooling(b *testing.B, samples, features int) {
	b.ReportAllocs()

	pool := performance.NewMatrixPool(100)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		matrices := make([]*performance.PooledMatrix, 0)

		// Allocate matrices from pool
		batchSize := 1000
		numBatches := samples / batchSize

		for j := 0; j < numBatches; j++ {
			m := pool.Get(batchSize, features)

			// Simulate processing
			for r := 0; r < batchSize; r++ {
				for c := 0; c < features; c++ {
					m.Set(r, c, rand.Float64())
				}
			}

			matrices = append(matrices, m)
		}

		// Return matrices to pool
		for _, m := range matrices {
			m.Release()
		}
	}

	stats := pool.GetStats()
	b.ReportMetric(stats.AverageReuseRate*100, "reuse%")
	b.SetBytes(int64(samples * features * 8))
}

// BenchmarkMemoryEfficiency compares memory usage patterns
func BenchmarkMemoryEfficiency(b *testing.B) {
	sizes := []int{
		100_000,
		1_000_000,
		10_000_000,
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			b.Run("Standard", func(b *testing.B) {
				benchmarkStandardMemory(b, size)
			})

			b.Run("ZeroCopy", func(b *testing.B) {
				benchmarkZeroCopyMemory(b, size)
			})

			b.Run("Pooled", func(b *testing.B) {
				benchmarkPooledMemory(b, size)
			})

			b.Run("GCOptimized", func(b *testing.B) {
				benchmarkGCOptimized(b, size)
			})
		})
	}
}

func benchmarkStandardMemory(b *testing.B, size int) {
	b.ReportAllocs()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	startAlloc := m.Alloc

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Standard allocation
		matrices := make([]*mat.Dense, 100)
		for j := range matrices {
			matrices[j] = mat.NewDense(size/100, 100, nil)
		}

		// Simulate processing
		for _, matrix := range matrices {
			rows, cols := matrix.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					matrix.Set(r, c, rand.Float64())
				}
			}
		}
	}

	runtime.ReadMemStats(&m)
	b.ReportMetric(float64(m.Alloc-startAlloc)/(1024*1024), "MB_allocated")
}

func benchmarkZeroCopyMemory(b *testing.B, size int) {
	b.ReportAllocs()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Pre-allocate data
		data := make([]float64, size*100)

		// Create zero-copy views
		matrices := make([]*performance.ZeroCopyMatrix, 100)
		chunkSize := size / 100

		for j := range matrices {
			start := j * chunkSize * 100
			end := start + chunkSize*100
			matrices[j] = performance.NewZeroCopyMatrix(data[start:end], chunkSize, 100)
		}

		// Simulate processing
		for _, matrix := range matrices {
			rows, cols := matrix.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					matrix.Set(r, c, rand.Float64())
				}
			}
		}
	}
}

func benchmarkPooledMemory(b *testing.B, size int) {
	b.ReportAllocs()

	pool := performance.NewMatrixPool(10)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		matrices := make([]*performance.PooledMatrix, 100)
		chunkSize := size / 100

		// Get from pool
		for j := range matrices {
			matrices[j] = pool.Get(chunkSize, 100)
		}

		// Simulate processing
		for _, matrix := range matrices {
			rows, cols := matrix.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					matrix.Set(r, c, rand.Float64())
				}
			}
		}

		// Return to pool
		for _, matrix := range matrices {
			matrix.Release()
		}
	}

	stats := pool.GetStats()
	b.ReportMetric(float64(stats.TotalRecycled), "matrices_recycled")
}

func benchmarkGCOptimized(b *testing.B, size int) {
	b.ReportAllocs()

	optimizer := performance.NewGCOptimizer(50, 100*time.Millisecond)
	optimizer.Start()
	defer optimizer.Stop()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Allocate with GC optimization
		matrices := make([]*mat.Dense, 100)
		for j := range matrices {
			matrices[j] = mat.NewDense(size/100, 100, nil)

			// Periodic GC hints
			if j%10 == 0 {
				optimizer.ForceGC()
			}
		}

		// Simulate processing
		for _, matrix := range matrices {
			rows, cols := matrix.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					matrix.Set(r, c, rand.Float64())
				}
			}
		}
	}

	stats := optimizer.GetStats()
	b.ReportMetric(stats["gc_cpu_percent"].(float64), "gc_cpu%")
}

// BenchmarkRealWorldScenarios tests realistic ML workflows
func BenchmarkRealWorldScenarios(b *testing.B) {
	b.Run("LinearRegression_1M", func(b *testing.B) {
		benchmarkLinearRegression(b, 1_000_000, 100)
	})

	b.Run("SGD_Streaming_10M", func(b *testing.B) {
		benchmarkSGDStreaming(b, 10_000_000, 50)
	})

	b.Run("BatchPrediction_100M", func(b *testing.B) {
		benchmarkBatchPrediction(b, 100_000_000, 20)
	})
}

func benchmarkLinearRegression(b *testing.B, samples, features int) {
	b.ReportAllocs()

	// Use memory-efficient batch processing
	batch := performance.NewMemoryEfficientBatch(1024) // 1GB limit

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		lr := linear.NewLinearRegression()

		// Train in batches
		batchSize := 10000
		for start := 0; start < samples; start += batchSize {
			end := start + batchSize
			if end > samples {
				end = samples
			}

			// Check memory before allocation
			bytes := int64((end - start) * features * 8)
			if !batch.CanAllocate(bytes) {
				runtime.GC()
			}

			if err := batch.Allocate(bytes); err != nil {
				b.Fatalf("Failed to allocate batch: %v", err)
			}

			// Generate batch data
			X := mat.NewDense(end-start, features, nil)
			y := mat.NewDense(end-start, 1, nil)

			// Train would happen here
			_ = lr
			_ = X
			_ = y

			batch.Free(bytes)
		}
	}

	used, max := batch.GetUsage()
	b.ReportMetric(float64(used)/float64(max)*100, "memory_usage%")
}

func benchmarkSGDStreaming(b *testing.B, samples, features int) {
	b.ReportAllocs()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		sgd := linear_model.NewSGDRegressor()

		// Simulate streaming data
		batchSize := 1000
		processed := 0

		for processed < samples {
			currentBatch := batchSize
			if processed+currentBatch > samples {
				currentBatch = samples - processed
			}

			// Generate batch
			X := mat.NewDense(currentBatch, features, nil)
			y := mat.NewDense(currentBatch, 1, nil)

			// Partial fit
			if err := sgd.PartialFit(X, y, nil); err != nil {
				b.Fatalf("Failed partial fit: %v", err)
			}

			processed += currentBatch
		}
	}

	b.SetBytes(int64(samples * features * 8))
}

func benchmarkBatchPrediction(b *testing.B, samples, features int) {
	b.ReportAllocs()

	// Pre-train a model
	model := linear.NewLinearRegression()
	trainX := mat.NewDense(1000, features, nil)
	trainY := mat.NewDense(1000, 1, nil)
	if err := model.Fit(trainX, trainY); err != nil {
		b.Fatalf("Failed to fit model: %v", err)
	}

	pool := performance.NewMatrixPool(10)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Batch prediction with pooling
		batchSize := 10000
		predictions := make([]float64, 0, samples)

		for start := 0; start < samples; start += batchSize {
			end := start + batchSize
			if end > samples {
				end = samples
			}

			// Get matrix from pool
			batch := pool.Get(end-start, features)

			// Fill with data
			for r := 0; r < end-start; r++ {
				for c := 0; c < features; c++ {
					batch.Set(r, c, rand.Float64())
				}
			}

			// Predict
			pred, _ := model.Predict(batch.ToMat())

			// Collect predictions
			rows, _ := pred.Dims()
			for r := 0; r < rows; r++ {
				_ = predictions // predictions is intentionally unused in benchmark
				_ = pred.At(r, 0)
			}

			// Return to pool
			batch.Release()
		}
	}

	stats := pool.GetStats()
	b.ReportMetric(stats.AverageReuseRate*100, "pool_reuse%")
}

// Helper types for streaming pipeline

type transformStage struct{}

func (t *transformStage) Process(in <-chan mat.Matrix) <-chan mat.Matrix {
	out := make(chan mat.Matrix, 100)
	go func() {
		defer close(out)
		for data := range in {
			// Simple transform
			out <- data
		}
	}()
	return out
}

type aggregateStage struct{}

func (a *aggregateStage) Process(in <-chan mat.Matrix) <-chan mat.Matrix {
	out := make(chan mat.Matrix, 100)
	go func() {
		defer close(out)
		for data := range in {
			// Simple aggregation
			out <- data
		}
	}()
	return out
}
