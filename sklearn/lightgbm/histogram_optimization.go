package lightgbm

import (
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// HistogramBin represents a single bin in a histogram
type HistogramBin struct {
	LowerBound float64
	UpperBound float64
	Count      int
	SumGrad    float64
	SumHess    float64
}

// FeatureHistogram represents histogram for a single feature
type FeatureHistogram struct {
	FeatureIndex int
	Bins         []HistogramBin
	BinBounds    []float64    // Sorted bin boundaries
	Missing      HistogramBin // Special bin for missing values
}

// HistogramBuilder builds histograms for efficient split finding
type HistogramBuilder struct {
	MaxBin         int     // Maximum number of bins
	MinDataInBin   int     // Minimum data in one bin
	CatSmooth      float64 // Smoothing for categorical splits
	MinGainToSplit float64 // Minimum gain to make a split
	Lambda         float64 // L2 regularization
	Alpha          float64 // L1 regularization

	// Thread pool for parallel processing
	numWorkers int
	pool       *sync.Pool
}

// NewHistogramBuilder creates a new histogram builder
func NewHistogramBuilder(params *TrainingParams) *HistogramBuilder {
	maxBin := params.MaxBin
	if maxBin == 0 {
		maxBin = 255
	}

	minDataInBin := params.MinDataInBin
	if minDataInBin == 0 {
		minDataInBin = 3
	}

	return &HistogramBuilder{
		MaxBin:         maxBin,
		MinDataInBin:   minDataInBin,
		CatSmooth:      params.CatSmooth,
		MinGainToSplit: params.MinGainToSplit,
		Lambda:         params.Lambda,
		Alpha:          params.Alpha,
		numWorkers:     4,
		pool: &sync.Pool{
			New: func() interface{} {
				return &FeatureHistogram{}
			},
		},
	}
}

// BuildHistograms builds histograms for all features
func (hb *HistogramBuilder) BuildHistograms(X *mat.Dense, indices []int,
	gradients, hessians []float64) []FeatureHistogram {

	_, cols := X.Dims()
	histograms := make([]FeatureHistogram, cols)

	// Build histogram for each feature in parallel
	var wg sync.WaitGroup
	ch := make(chan int, cols)

	// Start workers
	for w := 0; w < hb.numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for featureIdx := range ch {
				histograms[featureIdx] = hb.buildFeatureHistogram(
					X, featureIdx, indices, gradients, hessians)
			}
		}()
	}

	// Send work
	for j := 0; j < cols; j++ {
		ch <- j
	}
	close(ch)

	wg.Wait()
	return histograms
}

// buildFeatureHistogram builds histogram for a single feature
func (hb *HistogramBuilder) buildFeatureHistogram(X *mat.Dense, featureIdx int,
	indices []int, gradients, hessians []float64) FeatureHistogram {

	// Extract feature values for given indices
	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = X.At(idx, featureIdx)
	}

	// Find bin boundaries
	binBounds := hb.findBinBoundaries(values)

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

	// Aggregate data into bins
	for i, val := range values {
		binIdx := hb.findBinIndex(val, binBounds)
		if binIdx >= 0 && binIdx < len(hist.Bins) {
			idx := indices[i]
			hist.Bins[binIdx].Count++
			hist.Bins[binIdx].SumGrad += gradients[idx]
			hist.Bins[binIdx].SumHess += hessians[idx]
		}
	}

	return hist
}

// findBinBoundaries finds optimal bin boundaries for a feature
func (hb *HistogramBuilder) findBinBoundaries(values []float64) []float64 {
	if len(values) == 0 {
		return []float64{0, 1}
	}

	// Sort values
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	// Find unique values
	unique := []float64{sorted[0]}
	for i := 1; i < len(sorted); i++ {
		if sorted[i] != sorted[i-1] {
			unique = append(unique, sorted[i])
		}
	}

	// If unique values <= max bins, use all unique values
	if len(unique) <= hb.MaxBin {
		// Add boundaries
		bounds := make([]float64, len(unique)+1)
		bounds[0] = unique[0] - 1e-10
		for i := 0; i < len(unique); i++ {
			bounds[i+1] = unique[i] + 1e-10
		}
		return bounds
	}

	// Otherwise, use quantile-based binning
	bounds := make([]float64, hb.MaxBin+1)
	bounds[0] = sorted[0] - 1e-10

	for i := 1; i < hb.MaxBin; i++ {
		idx := (len(sorted) - 1) * i / hb.MaxBin
		bounds[i] = sorted[idx]
	}
	bounds[hb.MaxBin] = sorted[len(sorted)-1] + 1e-10

	// Remove duplicates
	uniqueBounds := []float64{bounds[0]}
	for i := 1; i < len(bounds); i++ {
		if bounds[i] > uniqueBounds[len(uniqueBounds)-1] {
			uniqueBounds = append(uniqueBounds, bounds[i])
		}
	}

	return uniqueBounds
}

// findBinIndex finds which bin a value belongs to
func (hb *HistogramBuilder) findBinIndex(value float64, binBounds []float64) int {
	// Binary search for the bin
	left, right := 0, len(binBounds)-2

	for left <= right {
		mid := (left + right) / 2
		if value >= binBounds[mid] && value < binBounds[mid+1] {
			return mid
		}
		if value < binBounds[mid] {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	// Edge cases
	if value < binBounds[0] {
		return 0
	}
	return len(binBounds) - 2
}

// FindBestSplitFromHistogram finds the best split using histogram
func (hb *HistogramBuilder) FindBestSplitFromHistogram(hist FeatureHistogram,
	totalGrad, totalHess float64, minDataInLeaf int) SplitInfo {

	bestSplit := SplitInfo{
		Feature: hist.FeatureIndex,
		Gain:    -math.MaxFloat64,
	}

	// Cumulative sums from left
	leftGrad := 0.0
	leftHess := 0.0
	leftCount := 0

	// Try each bin boundary as a potential split point
	for i := 0; i < len(hist.Bins)-1; i++ {
		leftGrad += hist.Bins[i].SumGrad
		leftHess += hist.Bins[i].SumHess
		leftCount += hist.Bins[i].Count

		rightGrad := totalGrad - leftGrad
		rightHess := totalHess - leftHess
		rightCount := 0
		for j := i + 1; j < len(hist.Bins); j++ {
			rightCount += hist.Bins[j].Count
		}

		// Check minimum data constraints
		if leftCount < minDataInLeaf || rightCount < minDataInLeaf {
			continue
		}

		// Calculate gain
		gain := hb.calculateGain(leftGrad, leftHess, rightGrad, rightHess, totalGrad, totalHess)

		if gain > bestSplit.Gain {
			bestSplit.Gain = gain
			bestSplit.Threshold = hist.Bins[i].UpperBound
			bestSplit.LeftCount = leftCount
			bestSplit.RightCount = rightCount
			bestSplit.LeftGrad = leftGrad
			bestSplit.RightGrad = rightGrad
			bestSplit.LeftHess = leftHess
			bestSplit.RightHess = rightHess
		}
	}

	return bestSplit
}

// calculateGain calculates the gain for a split
func (hb *HistogramBuilder) calculateGain(leftGrad, leftHess, rightGrad, rightHess,
	totalGrad, totalHess float64) float64 {

	// Add regularization to hessians
	leftHess += hb.Lambda
	rightHess += hb.Lambda
	totalHess += hb.Lambda

	// Calculate gain using LightGBM formula
	leftScore := leftGrad * leftGrad / leftHess
	rightScore := rightGrad * rightGrad / rightHess
	totalScore := totalGrad * totalGrad / totalHess

	gain := 0.5 * (leftScore + rightScore - totalScore)

	// Apply L1 regularization if needed
	if hb.Alpha > 0 {
		gain -= hb.Alpha * (math.Abs(leftGrad/leftHess) + math.Abs(rightGrad/rightHess))
	}

	return gain
}

// HistogramSubtraction performs histogram subtraction for sibling nodes
func (hb *HistogramBuilder) HistogramSubtraction(parentHist, siblingHist FeatureHistogram) FeatureHistogram {
	// Create result histogram
	result := FeatureHistogram{
		FeatureIndex: parentHist.FeatureIndex,
		BinBounds:    parentHist.BinBounds,
		Bins:         make([]HistogramBin, len(parentHist.Bins)),
	}

	// Subtract sibling from parent
	for i := range result.Bins {
		result.Bins[i] = HistogramBin{
			LowerBound: parentHist.Bins[i].LowerBound,
			UpperBound: parentHist.Bins[i].UpperBound,
			Count:      parentHist.Bins[i].Count - siblingHist.Bins[i].Count,
			SumGrad:    parentHist.Bins[i].SumGrad - siblingHist.Bins[i].SumGrad,
			SumHess:    parentHist.Bins[i].SumHess - siblingHist.Bins[i].SumHess,
		}
	}

	return result
}

// OptimizedSplitFinder finds the best split using histogram optimization
type OptimizedSplitFinder struct {
	builder      *HistogramBuilder
	histograms   []FeatureHistogram
	featureCache map[int][]float64 // Cache sorted feature values
	cacheMutex   sync.RWMutex
}

// NewOptimizedSplitFinder creates a new optimized split finder
func NewOptimizedSplitFinder(params *TrainingParams) *OptimizedSplitFinder {
	return &OptimizedSplitFinder{
		builder:      NewHistogramBuilder(params),
		featureCache: make(map[int][]float64),
	}
}

// FindBestSplit finds the best split across all features using histograms
func (osf *OptimizedSplitFinder) FindBestSplit(X *mat.Dense, indices []int,
	gradients, hessians []float64, params *TrainingParams) SplitInfo {

	// Build histograms for all features
	osf.histograms = osf.builder.BuildHistograms(X, indices, gradients, hessians)

	// Calculate total gradient and hessian
	totalGrad := 0.0
	totalHess := 0.0
	for _, idx := range indices {
		totalGrad += gradients[idx]
		totalHess += hessians[idx]
	}

	// Find best split across all features
	bestSplit := SplitInfo{
		Gain: -math.MaxFloat64,
	}

	for _, hist := range osf.histograms {
		split := osf.builder.FindBestSplitFromHistogram(
			hist, totalGrad, totalHess, params.MinDataInLeaf)

		if split.Gain > bestSplit.Gain {
			bestSplit = split
		}
	}

	// Check if gain meets threshold
	if bestSplit.Gain < params.MinGainToSplit {
		bestSplit.Gain = -math.MaxFloat64
	}

	return bestSplit
}

// GetCachedFeatureValues returns cached sorted feature values
func (osf *OptimizedSplitFinder) GetCachedFeatureValues(X *mat.Dense,
	featureIdx int, indices []int) []float64 {

	osf.cacheMutex.RLock()
	if cached, exists := osf.featureCache[featureIdx]; exists {
		osf.cacheMutex.RUnlock()
		return cached
	}
	osf.cacheMutex.RUnlock()

	// Build and cache
	osf.cacheMutex.Lock()
	defer osf.cacheMutex.Unlock()

	// Double-check after acquiring write lock
	if cached, exists := osf.featureCache[featureIdx]; exists {
		return cached
	}

	// Extract and sort feature values
	values := make([]float64, len(indices))
	for i, idx := range indices {
		values[i] = X.At(idx, featureIdx)
	}
	sort.Float64s(values)

	osf.featureCache[featureIdx] = values
	return values
}

// ClearCache clears the feature value cache
func (osf *OptimizedSplitFinder) ClearCache() {
	osf.cacheMutex.Lock()
	defer osf.cacheMutex.Unlock()
	osf.featureCache = make(map[int][]float64)
}
