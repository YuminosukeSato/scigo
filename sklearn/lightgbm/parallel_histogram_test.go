package lightgbm

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// TestParallelHistogramBuilder tests the parallel histogram builder
func TestParallelHistogramBuilder(t *testing.T) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	// Create test data
	nSamples := 1000
	nFeatures := 10

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rng.NormFloat64())
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1 // Ensure positive
	}

	// Test parallel builder
	phb := NewParallelHistogramBuilder(255)
	histograms := phb.BuildHistogramsParallel(X, indices, gradients, hessians, nil)

	assert.Equal(t, nFeatures, len(histograms))

	// Verify each histogram
	for i, hist := range histograms {
		assert.Equal(t, i, hist.FeatureIndex)
		assert.Greater(t, len(hist.Bins), 0)

		// Check that bins have data
		totalCount := 0
		for _, bin := range hist.Bins {
			totalCount += bin.Count
		}
		assert.Equal(t, nSamples, totalCount)
	}
}

// TestParallelVsSequentialConsistency tests that parallel and sequential give same results
func TestParallelVsSequentialConsistency(t *testing.T) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	// Create test data
	nSamples := 500
	nFeatures := 5

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rng.NormFloat64())
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1
	}

	// Build with sequential
	params := &TrainingParams{
		MaxBin:       255,
		MinDataInBin: 3,
	}
	seqBuilder := NewHistogramBuilder(params)
	seqHistograms := seqBuilder.BuildHistograms(X, indices, gradients, hessians, nil)

	// Build with parallel
	parBuilder := NewParallelHistogramBuilder(255)
	parHistograms := parBuilder.BuildHistogramsParallel(X, indices, gradients, hessians, nil)

	// Compare results
	require.Equal(t, len(seqHistograms), len(parHistograms))

	for i := range seqHistograms {
		seqHist := seqHistograms[i]
		parHist := parHistograms[i]

		assert.Equal(t, seqHist.FeatureIndex, parHist.FeatureIndex)

		// Compare bins (may have slight differences due to floating point)
		for j := range seqHist.Bins {
			if j < len(parHist.Bins) {
				assert.Equal(t, seqHist.Bins[j].Count, parHist.Bins[j].Count,
					"Count mismatch for feature %d, bin %d", i, j)
				assert.InDelta(t, seqHist.Bins[j].SumGrad, parHist.Bins[j].SumGrad, 1e-10,
					"SumGrad mismatch for feature %d, bin %d", i, j)
				assert.InDelta(t, seqHist.Bins[j].SumHess, parHist.Bins[j].SumHess, 1e-10,
					"SumHess mismatch for feature %d, bin %d", i, j)
			}
		}
	}
}

// TestParallelPerformance benchmarks parallel vs sequential performance
func TestParallelPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	// Large dataset for performance testing
	nSamples := 10000
	nFeatures := 50

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rng.NormFloat64())
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1
	}

	// Benchmark sequential
	params := &TrainingParams{
		MaxBin:       255,
		MinDataInBin: 3,
	}
	seqBuilder := NewHistogramBuilder(params)
	start := time.Now()
	seqBuilder.BuildHistograms(X, indices, gradients, hessians, nil)
	seqTime := time.Since(start)

	// Benchmark parallel
	parBuilder := NewParallelHistogramBuilder(255)
	start = time.Now()
	parBuilder.BuildHistogramsParallel(X, indices, gradients, hessians, nil)
	parTime := time.Since(start)

	t.Logf("Sequential time: %v", seqTime)
	t.Logf("Parallel time: %v", parTime)
	t.Logf("Speedup: %.2fx", float64(seqTime)/float64(parTime))

	// Parallel should be faster on multi-core systems
	if runtime.NumCPU() > 1 {
		assert.Less(t, parTime, seqTime, "Parallel should be faster than sequential")
	}
}

// TestSampleParallelization tests the sample-level parallelization
func TestSampleParallelization(t *testing.T) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	// Large samples, few features (triggers sample parallelization)
	nSamples := 15000
	nFeatures := 10

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rng.NormFloat64())
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1
	}

	phb := NewParallelHistogramBuilder(255)
	histograms := phb.BuildHistogramsParallel(X, indices, gradients, hessians, nil)

	assert.Equal(t, nFeatures, len(histograms))

	// Verify correctness
	for i, hist := range histograms {
		assert.Equal(t, i, hist.FeatureIndex)

		totalCount := 0
		totalGrad := 0.0
		totalHess := 0.0

		for _, bin := range hist.Bins {
			totalCount += bin.Count
			totalGrad += bin.SumGrad
			totalHess += bin.SumHess
		}

		assert.Equal(t, nSamples, totalCount)

		// Check that gradients and hessians sum approximately match
		expectedGrad := 0.0
		expectedHess := 0.0
		for k := 0; k < nSamples; k++ {
			expectedGrad += gradients[k]
			expectedHess += hessians[k]
		}

		assert.InDelta(t, expectedGrad, totalGrad, 1e-8)
		assert.InDelta(t, expectedHess, totalHess, 1e-8)
	}
}

// TestFeatureParallelization tests the feature-level parallelization
func TestFeatureParallelization(t *testing.T) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	// Few samples, many features (triggers feature parallelization)
	nSamples := 100
	nFeatures := 200

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rng.NormFloat64())
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1
	}

	phb := NewParallelHistogramBuilder(255)
	histograms := phb.BuildHistogramsParallel(X, indices, gradients, hessians, nil)

	assert.Equal(t, nFeatures, len(histograms))

	// Verify each histogram
	for i, hist := range histograms {
		assert.Equal(t, i, hist.FeatureIndex)
		assert.Greater(t, len(hist.Bins), 0)
	}
}

// TestWithCategoricalFeatures tests parallel histogram with categorical features
func TestWithCategoricalFeatures(t *testing.T) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	nSamples := 500
	nFeatures := 10
	categoricalFeatures := []int{2, 5, 8} // Mark some features as categorical

	X := mat.NewDense(nSamples, nFeatures, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			if isCategoricalFeature(j, categoricalFeatures) {
				// Categorical values (0, 1, 2, 3)
				X.Set(i, j, float64(rand.IntN(4)))
			} else {
				// Continuous values
				X.Set(i, j, rng.NormFloat64())
			}
		}
	}

	indices := make([]int, nSamples)
	gradients := make([]float64, nSamples)
	hessians := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		indices[i] = i
		gradients[i] = rng.NormFloat64()
		hessians[i] = rng.Float64() + 0.1
	}

	phb := NewParallelHistogramBuilder(255)
	histograms := phb.BuildHistogramsParallel(X, indices, gradients, hessians, categoricalFeatures)

	assert.Equal(t, nFeatures, len(histograms))

	// Verify categorical features have appropriate bins
	for _, catIdx := range categoricalFeatures {
		hist := histograms[catIdx]
		assert.Equal(t, catIdx, hist.FeatureIndex)
		// Categorical features should have fewer bins (one per unique value)
		assert.LessOrEqual(t, len(hist.Bins), 4)
	}
}

// BenchmarkParallelHistogramBuilding benchmarks parallel histogram building
func BenchmarkParallelHistogramBuilding(b *testing.B) {
	// Set random seed for reproducibility
	rng := rand.New(rand.NewPCG(42, 42))

	sizes := []struct {
		samples  int
		features int
	}{
		{1000, 10},
		{5000, 50},
		{10000, 100},
	}

	for _, size := range sizes {
		// Prepare data
		X := mat.NewDense(size.samples, size.features, nil)
		for i := 0; i < size.samples; i++ {
			for j := 0; j < size.features; j++ {
				X.Set(i, j, rng.NormFloat64())
			}
		}

		indices := make([]int, size.samples)
		gradients := make([]float64, size.samples)
		hessians := make([]float64, size.samples)

		for i := 0; i < size.samples; i++ {
			indices[i] = i
			gradients[i] = rng.NormFloat64()
			hessians[i] = rng.Float64() + 0.1
		}

		b.Run(fmt.Sprintf("Sequential_%dx%d", size.samples, size.features), func(b *testing.B) {
			params := &TrainingParams{
				MaxBin:       255,
				MinDataInBin: 3,
			}
			builder := NewHistogramBuilder(params)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				builder.BuildHistograms(X, indices, gradients, hessians, nil)
			}
		})

		b.Run(fmt.Sprintf("Parallel_%dx%d", size.samples, size.features), func(b *testing.B) {
			builder := NewParallelHistogramBuilder(255)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				builder.BuildHistogramsParallel(X, indices, gradients, hessians, nil)
			}
		})
	}
}

// TestAdaptiveChunkSize tests that chunk size adapts based on CPU count
func TestAdaptiveChunkSize(t *testing.T) {
	phb := NewParallelHistogramBuilder(255)

	// Check that chunk size is set appropriately
	if runtime.NumCPU() > 8 {
		assert.Equal(t, 500, phb.chunkSize)
	} else {
		assert.Equal(t, 1000, phb.chunkSize)
	}

	// Test setting custom chunk size
	phb.SetChunkSize(2000)
	assert.Equal(t, 2000, phb.chunkSize)

	// Test setting custom worker count
	phb.SetNumWorkers(4)
	assert.Equal(t, 4, phb.numWorkers)
}
