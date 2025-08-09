package lightgbm

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestHistogramBuilder(t *testing.T) {
	t.Run("Basic histogram building", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin:       10,
			MinDataInBin: 3,
			Lambda:       0.0,
		}

		builder := NewHistogramBuilder(params)
		assert.NotNil(t, builder)
		assert.Equal(t, 10, builder.MaxBin)
		assert.Equal(t, 3, builder.MinDataInBin)
	})

	t.Run("Find bin boundaries", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin: 5,
		}
		builder := NewHistogramBuilder(params)

		// Test with simple values
		values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		bounds := builder.findBinBoundaries(values)

		assert.Greater(t, len(bounds), 2)
		assert.LessOrEqual(t, len(bounds), 6) // MaxBin + 1

		// Check bounds are sorted
		for i := 1; i < len(bounds); i++ {
			assert.Greater(t, bounds[i], bounds[i-1])
		}
	})

	t.Run("Find bin index", func(t *testing.T) {
		params := &TrainingParams{MaxBin: 5}
		builder := NewHistogramBuilder(params)

		bounds := []float64{0, 2, 4, 6, 8, 10}

		// Test various values
		assert.Equal(t, 0, builder.findBinIndex(1, bounds))
		assert.Equal(t, 1, builder.findBinIndex(3, bounds))
		assert.Equal(t, 2, builder.findBinIndex(5, bounds))
		assert.Equal(t, 3, builder.findBinIndex(7, bounds))
		assert.Equal(t, 4, builder.findBinIndex(9, bounds))
	})

	t.Run("Build feature histogram", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin:       5,
			MinDataInBin: 1,
		}
		builder := NewHistogramBuilder(params)

		// Create simple data
		n := 20
		X := mat.NewDense(n, 1, nil)
		for i := 0; i < n; i++ {
			X.Set(i, 0, float64(i))
		}

		indices := make([]int, n)
		gradients := make([]float64, n)
		hessians := make([]float64, n)

		for i := 0; i < n; i++ {
			indices[i] = i
			gradients[i] = rand.Float64() - 0.5
			hessians[i] = rand.Float64()
		}

		hist := builder.buildFeatureHistogram(X, 0, indices, gradients, hessians)

		assert.Equal(t, 0, hist.FeatureIndex)
		assert.NotEmpty(t, hist.Bins)
		assert.NotEmpty(t, hist.BinBounds)

		// Check total count
		totalCount := 0
		for _, bin := range hist.Bins {
			totalCount += bin.Count
		}
		assert.Equal(t, n, totalCount)
	})
}

func TestHistogramOptimization(t *testing.T) {
	t.Run("Build histograms for multiple features", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin:       10,
			MinDataInBin: 3,
		}
		builder := NewHistogramBuilder(params)

		// Create data with 3 features
		n := 100
		X := mat.NewDense(n, 3, nil)
		for i := 0; i < n; i++ {
			for j := 0; j < 3; j++ {
				X.Set(i, j, rand.Float64()*10)
			}
		}

		indices := make([]int, n)
		gradients := make([]float64, n)
		hessians := make([]float64, n)

		for i := 0; i < n; i++ {
			indices[i] = i
			gradients[i] = rand.NormFloat64()
			hessians[i] = rand.Float64() + 0.1
		}

		histograms := builder.BuildHistograms(X, indices, gradients, hessians)

		assert.Equal(t, 3, len(histograms))

		for i, hist := range histograms {
			assert.Equal(t, i, hist.FeatureIndex)
			assert.NotEmpty(t, hist.Bins)

			// Verify gradient and hessian sums
			totalGrad := 0.0
			totalHess := 0.0
			for _, bin := range hist.Bins {
				totalGrad += bin.SumGrad
				totalHess += bin.SumHess
			}

			expectedGrad := 0.0
			expectedHess := 0.0
			for _, idx := range indices {
				expectedGrad += gradients[idx]
				expectedHess += hessians[idx]
			}

			assert.InDelta(t, expectedGrad, totalGrad, 1e-10)
			assert.InDelta(t, expectedHess, totalHess, 1e-10)
		}
	})

	t.Run("Find best split from histogram", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin:         5,
			MinDataInBin:   3,
			MinDataInLeaf:  5,
			MinGainToSplit: 0.01,
			Lambda:         0.1,
		}
		builder := NewHistogramBuilder(params)

		// Create histogram manually for testing
		hist := FeatureHistogram{
			FeatureIndex: 0,
			BinBounds:    []float64{0, 2, 4, 6, 8, 10},
			Bins: []HistogramBin{
				{LowerBound: 0, UpperBound: 2, Count: 10, SumGrad: -5, SumHess: 10},
				{LowerBound: 2, UpperBound: 4, Count: 10, SumGrad: -3, SumHess: 10},
				{LowerBound: 4, UpperBound: 6, Count: 10, SumGrad: 0, SumHess: 10},
				{LowerBound: 6, UpperBound: 8, Count: 10, SumGrad: 3, SumHess: 10},
				{LowerBound: 8, UpperBound: 10, Count: 10, SumGrad: 5, SumHess: 10},
			},
		}

		totalGrad := 0.0
		totalHess := 50.0

		split := builder.FindBestSplitFromHistogram(hist, totalGrad, totalHess, 5)

		assert.Equal(t, 0, split.Feature)
		assert.Greater(t, split.Gain, -math.MaxFloat64)
		assert.Greater(t, split.Threshold, 0.0)
	})

	t.Run("Histogram subtraction", func(t *testing.T) {
		params := &TrainingParams{}
		builder := NewHistogramBuilder(params)

		parent := FeatureHistogram{
			FeatureIndex: 0,
			BinBounds:    []float64{0, 5, 10},
			Bins: []HistogramBin{
				{Count: 20, SumGrad: 10, SumHess: 20},
				{Count: 30, SumGrad: 15, SumHess: 30},
			},
		}

		sibling := FeatureHistogram{
			FeatureIndex: 0,
			BinBounds:    []float64{0, 5, 10},
			Bins: []HistogramBin{
				{Count: 8, SumGrad: 4, SumHess: 8},
				{Count: 12, SumGrad: 6, SumHess: 12},
			},
		}

		result := builder.HistogramSubtraction(parent, sibling)

		assert.Equal(t, 0, result.FeatureIndex)
		assert.Equal(t, 12, result.Bins[0].Count)
		assert.InDelta(t, 6.0, result.Bins[0].SumGrad, 1e-10)
		assert.InDelta(t, 12.0, result.Bins[0].SumHess, 1e-10)
		assert.Equal(t, 18, result.Bins[1].Count)
		assert.InDelta(t, 9.0, result.Bins[1].SumGrad, 1e-10)
		assert.InDelta(t, 18.0, result.Bins[1].SumHess, 1e-10)
	})
}

func TestOptimizedSplitFinder(t *testing.T) {
	t.Run("Find best split with optimization", func(t *testing.T) {
		params := &TrainingParams{
			MaxBin:         10,
			MinDataInBin:   3,
			MinDataInLeaf:  5,
			MinGainToSplit: 0.01,
			Lambda:         0.1,
		}

		finder := NewOptimizedSplitFinder(params)

		// Create dataset
		n := 100
		X := mat.NewDense(n, 3, nil)
		for i := 0; i < n; i++ {
			// Feature 0: good split at 5
			if i < 50 {
				X.Set(i, 0, rand.Float64()*5)
			} else {
				X.Set(i, 0, 5+rand.Float64()*5)
			}

			// Feature 1: random
			X.Set(i, 1, rand.Float64()*10)

			// Feature 2: poor split
			X.Set(i, 2, float64(i)/10)
		}

		indices := make([]int, n)
		gradients := make([]float64, n)
		hessians := make([]float64, n)

		for i := 0; i < n; i++ {
			indices[i] = i
			// Gradient correlates with feature 0
			if i < 50 {
				gradients[i] = -1 + rand.Float64()*0.2
			} else {
				gradients[i] = 1 + rand.Float64()*0.2
			}
			hessians[i] = 1.0
		}

		split := finder.FindBestSplit(X, indices, gradients, hessians, params)

		// Should find split on feature 0
		assert.Equal(t, 0, split.Feature)
		assert.Greater(t, split.Gain, 0.0)
		assert.InDelta(t, 5.0, split.Threshold, 2.0) // Around 5
	})

	t.Run("Cache feature values", func(t *testing.T) {
		params := &TrainingParams{}
		finder := NewOptimizedSplitFinder(params)

		n := 50
		X := mat.NewDense(n, 2, nil)
		for i := 0; i < n; i++ {
			X.Set(i, 0, float64(i))
			X.Set(i, 1, float64(n-i))
		}

		indices := make([]int, n)
		for i := 0; i < n; i++ {
			indices[i] = i
		}

		// First call should cache
		values1 := finder.GetCachedFeatureValues(X, 0, indices)
		assert.Equal(t, n, len(values1))

		// Second call should use cache
		values2 := finder.GetCachedFeatureValues(X, 0, indices)
		assert.Equal(t, values1, values2)

		// Clear cache
		finder.ClearCache()

		// After clearing, should recalculate
		values3 := finder.GetCachedFeatureValues(X, 0, indices)
		assert.Equal(t, n, len(values3))
	})
}

func TestHistogramPerformance(t *testing.T) {
	t.Run("Performance with large dataset", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping performance test in short mode")
		}

		params := &TrainingParams{
			MaxBin:         255,
			MinDataInBin:   3,
			MinDataInLeaf:  20,
			MinGainToSplit: 0.01,
			Lambda:         1.0,
		}

		finder := NewOptimizedSplitFinder(params)

		// Large dataset
		n := 10000
		features := 50
		X := mat.NewDense(n, features, nil)

		for i := 0; i < n; i++ {
			for j := 0; j < features; j++ {
				X.Set(i, j, rand.Float64()*100)
			}
		}

		indices := make([]int, n)
		gradients := make([]float64, n)
		hessians := make([]float64, n)

		for i := 0; i < n; i++ {
			indices[i] = i
			gradients[i] = rand.NormFloat64()
			hessians[i] = rand.Float64() + 0.1
		}

		// Should complete in reasonable time
		split := finder.FindBestSplit(X, indices, gradients, hessians, params)

		assert.NotNil(t, split)
		assert.GreaterOrEqual(t, split.Feature, 0)
		assert.Less(t, split.Feature, features)
	})
}

func BenchmarkHistogramBuilding(b *testing.B) {
	params := &TrainingParams{
		MaxBin:       255,
		MinDataInBin: 3,
	}
	builder := NewHistogramBuilder(params)

	// Setup data
	n := 1000
	features := 10
	X := mat.NewDense(n, features, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < features; j++ {
			X.Set(i, j, rand.Float64())
		}
	}

	indices := make([]int, n)
	gradients := make([]float64, n)
	hessians := make([]float64, n)
	for i := 0; i < n; i++ {
		indices[i] = i
		gradients[i] = rand.NormFloat64()
		hessians[i] = rand.Float64()
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = builder.BuildHistograms(X, indices, gradients, hessians)
	}
}

func BenchmarkFindBestSplit(b *testing.B) {
	params := &TrainingParams{
		MaxBin:         255,
		MinDataInBin:   3,
		MinDataInLeaf:  20,
		MinGainToSplit: 0.01,
	}
	finder := NewOptimizedSplitFinder(params)

	// Setup data
	n := 1000
	features := 10
	X := mat.NewDense(n, features, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < features; j++ {
			X.Set(i, j, rand.Float64())
		}
	}

	indices := make([]int, n)
	gradients := make([]float64, n)
	hessians := make([]float64, n)
	for i := 0; i < n; i++ {
		indices[i] = i
		gradients[i] = rand.NormFloat64()
		hessians[i] = rand.Float64()
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = finder.FindBestSplit(X, indices, gradients, hessians, params)
	}
}
