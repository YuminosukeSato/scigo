package lightgbm

import (
	"runtime"
	"sync"
	"sync/atomic"

	"gonum.org/v1/gonum/mat"
)

// ParallelHistogramBuilder provides enhanced parallel histogram construction
type ParallelHistogramBuilder struct {
	maxBins    int
	numWorkers int
	chunkSize  int // Number of samples per chunk for parallel processing
}

// NewParallelHistogramBuilder creates a new parallel histogram builder
func NewParallelHistogramBuilder(maxBins int) *ParallelHistogramBuilder {
	numWorkers := runtime.NumCPU()
	// Adaptive chunk size based on CPU count
	chunkSize := 1000
	if numWorkers > 8 {
		chunkSize = 500 // Smaller chunks for more parallelism
	}

	return &ParallelHistogramBuilder{
		maxBins:    maxBins,
		numWorkers: numWorkers,
		chunkSize:  chunkSize,
	}
}

// BuildHistogramsParallel builds histograms with enhanced parallelization
func (phb *ParallelHistogramBuilder) BuildHistogramsParallel(X *mat.Dense, indices []int,
	gradients, hessians []float64, categoricalFeatures []int) []FeatureHistogram {

	_, cols := X.Dims()
	histograms := make([]FeatureHistogram, cols)

	// Use parallel strategy based on data size
	if len(indices) > 10000 && cols < 100 {
		// For large samples with few features: parallelize within features
		phb.buildWithSampleParallelization(X, indices, gradients, hessians,
			categoricalFeatures, histograms)
	} else {
		// For many features or small samples: parallelize across features
		phb.buildWithFeatureParallelization(X, indices, gradients, hessians,
			categoricalFeatures, histograms)
	}

	return histograms
}

// buildWithFeatureParallelization parallelizes across features
func (phb *ParallelHistogramBuilder) buildWithFeatureParallelization(X *mat.Dense, indices []int,
	gradients, hessians []float64, categoricalFeatures []int, histograms []FeatureHistogram) {

	_, cols := X.Dims()
	var wg sync.WaitGroup
	ch := make(chan int, cols)

	// Create a simple params for histogram builder
	params := &TrainingParams{
		MaxBin:       phb.maxBins,
		MinDataInBin: 3,
	}

	// Start worker goroutines
	for w := 0; w < phb.numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			builder := NewHistogramBuilder(params)

			for featureIdx := range ch {
				if isCategoricalFeature(featureIdx, categoricalFeatures) {
					histograms[featureIdx] = builder.buildCategoricalHistogram(
						X, featureIdx, indices, gradients, hessians)
				} else {
					histograms[featureIdx] = phb.buildContinuousHistogramOptimized(
						X, featureIdx, indices, gradients, hessians)
				}
			}
		}()
	}

	// Send work to workers
	for j := 0; j < cols; j++ {
		ch <- j
	}
	close(ch)

	wg.Wait()
}

// buildWithSampleParallelization parallelizes within each feature
func (phb *ParallelHistogramBuilder) buildWithSampleParallelization(X *mat.Dense, indices []int,
	gradients, hessians []float64, categoricalFeatures []int, histograms []FeatureHistogram) {

	_, cols := X.Dims()

	// Create a simple params for histogram builder
	params := &TrainingParams{
		MaxBin:       phb.maxBins,
		MinDataInBin: 3,
	}

	// Process each feature sequentially but parallelize sample processing
	for featureIdx := 0; featureIdx < cols; featureIdx++ {
		if isCategoricalFeature(featureIdx, categoricalFeatures) {
			builder := NewHistogramBuilder(params)
			histograms[featureIdx] = builder.buildCategoricalHistogram(
				X, featureIdx, indices, gradients, hessians)
		} else {
			histograms[featureIdx] = phb.buildContinuousHistogramParallelSamples(
				X, featureIdx, indices, gradients, hessians)
		}
	}
}

// buildContinuousHistogramOptimized builds histogram with optimizations
func (phb *ParallelHistogramBuilder) buildContinuousHistogramOptimized(X *mat.Dense, featureIdx int,
	indices []int, gradients, hessians []float64) FeatureHistogram {

	// Extract feature values with pre-allocation
	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = X.At(idx, featureIdx)
	}

	// Find bin boundaries
	binBounds := phb.findBinBoundariesOptimized(values)

	// Create histogram
	hist := FeatureHistogram{
		FeatureIndex: featureIdx,
		BinBounds:    binBounds,
		Bins:         make([]HistogramBin, len(binBounds)-1),
	}

	// Initialize bins
	for i := 0; i < len(hist.Bins); i++ {
		hist.Bins[i] = HistogramBin{
			LowerBound: binBounds[i],
			UpperBound: binBounds[i+1],
		}
	}

	// Aggregate data into bins with optimized loop
	for i := 0; i < len(values); i++ {
		binIdx := phb.findBinIndexOptimized(values[i], binBounds)
		if binIdx >= 0 && binIdx < len(hist.Bins) {
			idx := indices[i]
			hist.Bins[binIdx].Count++
			hist.Bins[binIdx].SumGrad += gradients[idx]
			hist.Bins[binIdx].SumHess += hessians[idx]
		}
	}

	return hist
}

// buildContinuousHistogramParallelSamples builds histogram with parallel sample processing
func (phb *ParallelHistogramBuilder) buildContinuousHistogramParallelSamples(X *mat.Dense,
	featureIdx int, indices []int, gradients, hessians []float64) FeatureHistogram {

	// Extract feature values
	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = X.At(idx, featureIdx)
	}

	// Find bin boundaries
	binBounds := phb.findBinBoundariesOptimized(values)
	numBins := len(binBounds) - 1

	// Create atomic counters for parallel aggregation
	atomicCounts := make([]int64, numBins)
	atomicSumGrads := make([]atomic.Value, numBins)
	atomicSumHess := make([]atomic.Value, numBins)

	for i := 0; i < numBins; i++ {
		atomicSumGrads[i].Store(0.0)
		atomicSumHess[i].Store(0.0)
	}

	// Parallel aggregation
	var wg sync.WaitGroup
	numChunks := (len(indices) + phb.chunkSize - 1) / phb.chunkSize

	for chunk := 0; chunk < numChunks; chunk++ {
		wg.Add(1)
		start := chunk * phb.chunkSize
		end := start + phb.chunkSize
		if end > len(indices) {
			end = len(indices)
		}

		go func(start, end int) {
			defer wg.Done()

			// Local accumulation to reduce contention
			localCounts := make([]int64, numBins)
			localSumGrads := make([]float64, numBins)
			localSumHess := make([]float64, numBins)

			for i := start; i < end; i++ {
				binIdx := phb.findBinIndexOptimized(values[i], binBounds)
				if binIdx >= 0 && binIdx < numBins {
					idx := indices[i]
					localCounts[binIdx]++
					localSumGrads[binIdx] += gradients[idx]
					localSumHess[binIdx] += hessians[idx]
				}
			}

			// Merge local results with atomic operations
			for b := 0; b < numBins; b++ {
				if localCounts[b] > 0 {
					atomic.AddInt64(&atomicCounts[b], localCounts[b])

					// Use compare-and-swap for float64 addition
					for {
						oldGrad := atomicSumGrads[b].Load().(float64)
						newGrad := oldGrad + localSumGrads[b]
						if atomicSumGrads[b].CompareAndSwap(oldGrad, newGrad) {
							break
						}
					}

					for {
						oldHess := atomicSumHess[b].Load().(float64)
						newHess := oldHess + localSumHess[b]
						if atomicSumHess[b].CompareAndSwap(oldHess, newHess) {
							break
						}
					}
				}
			}
		}(start, end)
	}

	wg.Wait()

	// Create final histogram
	hist := FeatureHistogram{
		FeatureIndex: featureIdx,
		BinBounds:    binBounds,
		Bins:         make([]HistogramBin, numBins),
	}

	for i := 0; i < numBins; i++ {
		hist.Bins[i] = HistogramBin{
			LowerBound: binBounds[i],
			UpperBound: binBounds[i+1],
			Count:      int(atomicCounts[i]),
			SumGrad:    atomicSumGrads[i].Load().(float64),
			SumHess:    atomicSumHess[i].Load().(float64),
		}
	}

	return hist
}

// findBinBoundariesOptimized finds bin boundaries with optimizations
func (phb *ParallelHistogramBuilder) findBinBoundariesOptimized(values []float64) []float64 {
	if len(values) == 0 {
		return []float64{0, 1}
	}

	// Sort values for quantile-based binning
	sorted := make([]float64, len(values))
	copy(sorted, values)
	quickSort(sorted, 0, len(sorted)-1)

	// Remove duplicates
	unique := make([]float64, 0, len(sorted))
	unique = append(unique, sorted[0])
	for i := 1; i < len(sorted); i++ {
		if sorted[i] != sorted[i-1] {
			unique = append(unique, sorted[i])
		}
	}

	if len(unique) <= phb.maxBins {
		// If unique values <= maxBins, use all unique values as boundaries
		bounds := make([]float64, len(unique)+1)
		copy(bounds, unique)
		bounds[len(unique)] = unique[len(unique)-1] + 1e-10
		return bounds
	}

	// Quantile-based binning
	bounds := make([]float64, phb.maxBins+1)
	step := float64(len(unique)-1) / float64(phb.maxBins)

	for i := 0; i < phb.maxBins; i++ {
		idx := int(float64(i) * step)
		bounds[i] = unique[idx]
	}
	bounds[phb.maxBins] = unique[len(unique)-1] + 1e-10

	return bounds
}

// findBinIndexOptimized finds bin index using binary search
func (phb *ParallelHistogramBuilder) findBinIndexOptimized(value float64, bounds []float64) int {
	// Binary search for efficiency
	left, right := 0, len(bounds)-2

	for left <= right {
		mid := (left + right) / 2
		if value < bounds[mid] {
			right = mid - 1
		} else if value >= bounds[mid+1] {
			left = mid + 1
		} else {
			return mid
		}
	}

	// Edge cases
	if value < bounds[0] {
		return 0
	}
	if value >= bounds[len(bounds)-1] {
		return len(bounds) - 2
	}

	return -1
}

// quickSort implements in-place quicksort for efficiency
func quickSort(arr []float64, low, high int) {
	if low < high {
		pi := partition(arr, low, high)
		quickSort(arr, low, pi-1)
		quickSort(arr, pi+1, high)
	}
}

// partition is a helper for quicksort
func partition(arr []float64, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}

	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// SetNumWorkers sets the number of worker goroutines
func (phb *ParallelHistogramBuilder) SetNumWorkers(n int) {
	if n > 0 {
		phb.numWorkers = n
	}
}

// SetChunkSize sets the chunk size for parallel processing
func (phb *ParallelHistogramBuilder) SetChunkSize(size int) {
	if size > 0 {
		phb.chunkSize = size
	}
}
