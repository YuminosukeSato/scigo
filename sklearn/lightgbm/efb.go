package lightgbm

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// EFB (Exclusive Feature Bundling) implementation for sparse feature optimization
// Bundles mutually exclusive sparse features to reduce memory usage and training time

// FeatureBundle represents a bundle of exclusive features
type FeatureBundle struct {
	Features  []int     // Original feature indices
	Values    []float64 // Bundle values
	Conflicts int       // Number of conflicts in this bundle
}

// EFBOptimizer implements Exclusive Feature Bundling
type EFBOptimizer struct {
	MaxConflictRate float64 // Maximum allowed conflict rate (default: 0.0)
	MaxBundles      int     // Maximum number of bundles (default: features/3)
	SparsityRatio   float64 // Minimum sparsity ratio to consider bundling (default: 0.8)
}

// NewEFBOptimizer creates a new EFB optimizer
func NewEFBOptimizer() *EFBOptimizer {
	return &EFBOptimizer{
		MaxConflictRate: 0.0, // Default: no conflicts allowed
		MaxBundles:      0,   // Will be set based on number of features
		SparsityRatio:   0.8, // Features must be 80% sparse
	}
}

// CreateBundles creates feature bundles from the dataset
func (efb *EFBOptimizer) CreateBundles(X mat.Matrix) ([]FeatureBundle, []int, error) {
	_, cols := X.Dims()

	// Set default max bundles if not specified
	if efb.MaxBundles == 0 {
		efb.MaxBundles = cols / 3
		if efb.MaxBundles < 1 {
			efb.MaxBundles = 1
		}
	}

	// Calculate feature sparsity
	sparsity := efb.calculateSparsity(X)

	// Identify sparse features that can be bundled
	sparseFeatures := make([]int, 0)
	for i := 0; i < cols; i++ {
		if sparsity[i] >= efb.SparsityRatio {
			sparseFeatures = append(sparseFeatures, i)
		}
	}

	if len(sparseFeatures) < 2 {
		// Not enough sparse features to bundle
		return nil, make([]int, cols), nil
	}

	// Build conflict graph
	conflictGraph := efb.buildConflictGraph(X, sparseFeatures)

	// Create bundles using greedy algorithm
	bundles := efb.greedyBundling(conflictGraph, sparseFeatures, X)

	// Create feature mapping
	featureMapping := efb.createFeatureMapping(bundles, cols)

	return bundles, featureMapping, nil
}

// calculateSparsity calculates the sparsity ratio for each feature
func (efb *EFBOptimizer) calculateSparsity(X mat.Matrix) []float64 {
	rows, cols := X.Dims()
	sparsity := make([]float64, cols)

	for j := 0; j < cols; j++ {
		zeros := 0
		for i := 0; i < rows; i++ {
			if math.Abs(X.At(i, j)) < 1e-10 {
				zeros++
			}
		}
		sparsity[j] = float64(zeros) / float64(rows)
	}

	return sparsity
}

// buildConflictGraph builds a conflict graph between features
func (efb *EFBOptimizer) buildConflictGraph(X mat.Matrix, features []int) [][]float64 {
	numFeatures := len(features)
	conflicts := make([][]float64, numFeatures)

	for i := 0; i < numFeatures; i++ {
		conflicts[i] = make([]float64, numFeatures)
	}

	rows, _ := X.Dims()

	// Calculate conflict rates between feature pairs
	for i := 0; i < numFeatures; i++ {
		for j := i + 1; j < numFeatures; j++ {
			feat1 := features[i]
			feat2 := features[j]

			conflictCount := 0
			totalNonZero := 0

			for row := 0; row < rows; row++ {
				val1 := X.At(row, feat1)
				val2 := X.At(row, feat2)

				nonZero1 := math.Abs(val1) > 1e-10
				nonZero2 := math.Abs(val2) > 1e-10

				if nonZero1 || nonZero2 {
					totalNonZero++
					if nonZero1 && nonZero2 {
						conflictCount++
					}
				}
			}

			conflictRate := 0.0
			if totalNonZero > 0 {
				conflictRate = float64(conflictCount) / float64(totalNonZero)
			}

			conflicts[i][j] = conflictRate
			conflicts[j][i] = conflictRate
		}
	}

	return conflicts
}

// greedyBundling uses greedy algorithm to create feature bundles
func (efb *EFBOptimizer) greedyBundling(conflicts [][]float64, features []int, _ mat.Matrix) []FeatureBundle {
	numFeatures := len(features)
	used := make([]bool, numFeatures)
	bundles := make([]FeatureBundle, 0)

	// Sort features by degree (number of conflicts)
	type featureConflict struct {
		index     int
		conflicts int
	}

	featureOrder := make([]featureConflict, numFeatures)
	for i := 0; i < numFeatures; i++ {
		conflictCount := 0
		for j := 0; j < numFeatures; j++ {
			if i != j && conflicts[i][j] > efb.MaxConflictRate {
				conflictCount++
			}
		}
		featureOrder[i] = featureConflict{index: i, conflicts: conflictCount}
	}

	// Sort by conflicts (ascending - fewer conflicts first)
	sort.Slice(featureOrder, func(i, j int) bool {
		return featureOrder[i].conflicts < featureOrder[j].conflicts
	})

	// Greedy bundling
	for len(bundles) < efb.MaxBundles && len(bundles) < numFeatures {
		bundle := FeatureBundle{
			Features: make([]int, 0),
		}

		// Find the first unused feature with minimum conflicts
		seedIdx := -1
		for _, fc := range featureOrder {
			if !used[fc.index] {
				seedIdx = fc.index
				break
			}
		}

		if seedIdx == -1 {
			break // All features are used
		}

		// Add seed feature to bundle
		bundle.Features = append(bundle.Features, features[seedIdx])
		used[seedIdx] = true

		// Add compatible features to the bundle
		for _, fc := range featureOrder {
			i := fc.index
			if used[i] {
				continue
			}

			// Check if this feature can be added to the current bundle
			canAdd := true
			for _, bundledIdx := range bundle.Features {
				// Find the index of bundled feature in the features slice
				bundledOrigIdx := -1
				for k, feat := range features {
					if feat == bundledIdx {
						bundledOrigIdx = k
						break
					}
				}

				if bundledOrigIdx != -1 && conflicts[i][bundledOrigIdx] > efb.MaxConflictRate {
					canAdd = false
					break
				}
			}

			if canAdd {
				bundle.Features = append(bundle.Features, features[i])
				used[i] = true
			}
		}

		// Only add bundle if it contains more than one feature
		if len(bundle.Features) > 1 {
			bundles = append(bundles, bundle)
		}
	}

	return bundles
}

// createFeatureMapping creates mapping from original features to compact bundle indices
func (efb *EFBOptimizer) createFeatureMapping(bundles []FeatureBundle, totalFeatures int) []int {
	mapping := make([]int, totalFeatures)

	// Create a map to track which features are bundled
	bundledFeatures := make(map[int]bool)
	for _, bundle := range bundles {
		for _, featIdx := range bundle.Features {
			bundledFeatures[featIdx] = true
		}
	}

	// Create compact mapping
	newIndex := 0

	// First, map unbundled features to consecutive indices
	for i := 0; i < totalFeatures; i++ {
		if !bundledFeatures[i] {
			mapping[i] = newIndex
			newIndex++
		}
	}

	// Then, map bundled features to their bundle indices
	for bundleIdx, bundle := range bundles {
		bundleIndex := newIndex + bundleIdx
		for _, featIdx := range bundle.Features {
			mapping[featIdx] = bundleIndex
		}
	}

	return mapping
}

// BundleFeatures creates bundled representation of the dataset
func (efb *EFBOptimizer) BundleFeatures(X mat.Matrix, bundles []FeatureBundle, mapping []int) (*mat.Dense, error) {
	rows, cols := X.Dims()

	// Calculate the number of features in bundled representation
	maxMappedIndex := 0
	for _, idx := range mapping {
		if idx > maxMappedIndex {
			maxMappedIndex = idx
		}
	}
	newCols := maxMappedIndex + 1

	// Create bundled matrix
	bundled := mat.NewDense(rows, newCols, nil)

	// Create a map to track which features are bundled
	bundledFeatures := make(map[int]bool)
	for _, bundle := range bundles {
		for _, featIdx := range bundle.Features {
			bundledFeatures[featIdx] = true
		}
	}

	// Copy unbundled features to their mapped positions
	for j := 0; j < cols; j++ {
		if !bundledFeatures[j] {
			// This feature is not bundled, copy directly
			mappedIndex := mapping[j]
			for i := 0; i < rows; i++ {
				bundled.Set(i, mappedIndex, X.At(i, j))
			}
		}
	}

	// Create bundled features
	for _, bundle := range bundles {
		bundleIndex := mapping[bundle.Features[0]] // All features in bundle map to same index

		for i := 0; i < rows; i++ {
			bundleValue := efb.createBundleValue(X, i, bundle.Features)
			bundled.Set(i, bundleIndex, bundleValue)
		}
	}

	return bundled, nil
}

// createBundleValue creates a single bundled value from multiple exclusive features
func (efb *EFBOptimizer) createBundleValue(X mat.Matrix, row int, features []int) float64 {
	// Use a large offset encoding to avoid conflicts between different features
	// Each feature gets its own large range to prevent overlap
	bundleValue := 0.0
	baseOffset := 1000.0 // Use large base offset to avoid conflicts

	for i, featIdx := range features {
		value := X.At(row, featIdx)
		if math.Abs(value) > 1e-10 {
			// Each feature gets offset: baseOffset * (i+1) + value
			offset := baseOffset * float64(i+1)
			bundleValue = offset + value
			break // Since features are exclusive, only one should be non-zero
		}
	}

	return bundleValue
}

// UnbundleFeatures reconstructs original features from bundled representation
func (efb *EFBOptimizer) UnbundleFeatures(bundled mat.Matrix, bundles []FeatureBundle, originalCols int) (*mat.Dense, error) {
	rows, _ := bundled.Dims()

	// Create original feature matrix
	original := mat.NewDense(rows, originalCols, nil)

	// Create mapping to understand the bundled structure
	mapping := efb.createFeatureMapping(bundles, originalCols)

	// Create a map to track which features are bundled
	bundledFeatures := make(map[int]bool)
	for _, bundle := range bundles {
		for _, featIdx := range bundle.Features {
			bundledFeatures[featIdx] = true
		}
	}

	// Copy unbundled features back to their original positions
	for j := 0; j < originalCols; j++ {
		if !bundledFeatures[j] {
			mappedIndex := mapping[j]
			for i := 0; i < rows; i++ {
				original.Set(i, j, bundled.At(i, mappedIndex))
			}
		}
	}

	// Unbundle features from bundle columns
	for _, bundle := range bundles {
		bundleIndex := mapping[bundle.Features[0]] // All features in bundle map to same index

		for i := 0; i < rows; i++ {
			bundleValue := bundled.At(i, bundleIndex)
			efb.unbundleValue(original, i, bundle.Features, bundleValue)
		}
	}

	return original, nil
}

// unbundleValue extracts original feature values from bundled value
func (efb *EFBOptimizer) unbundleValue(original *mat.Dense, row int, features []int, bundleValue float64) {
	if math.Abs(bundleValue) < 1e-10 {
		// Bundle value is zero, all features are zero
		for _, featIdx := range features {
			original.Set(row, featIdx, 0.0)
		}
		return
	}

	// Decode the bundled value
	// bundleValue is encoded as offset + value where offset starts at 1.0 for each feature
	// Find which feature is active by finding the correct offset range
	bestMatchIdx := -1
	bestMatchOffset := 0.0

	// Find the feature that was actually encoded
	// Match the new large offset encoding scheme
	baseOffset := 1000.0

	for i := range features {
		offset := baseOffset * float64(i+1)

		// Check if this offset could have produced the bundleValue
		if bundleValue >= offset && bundleValue < offset+baseOffset {
			// This feature is in the correct range
			bestMatchIdx = i
			bestMatchOffset = offset
			break
		}
	}

	// Initialize all features to zero
	for _, featIdx := range features {
		original.Set(row, featIdx, 0.0)
	}

	if bestMatchIdx >= 0 {
		// This feature is active
		value := bundleValue - bestMatchOffset
		original.Set(row, features[bestMatchIdx], value)
	}
}

// EFBInfo provides information about the bundling result
type EFBInfo struct {
	OriginalFeatures int             `json:"original_features"`
	BundledFeatures  int             `json:"bundled_features"`
	NumBundles       int             `json:"num_bundles"`
	CompressionRatio float64         `json:"compression_ratio"`
	Bundles          []FeatureBundle `json:"bundles"`
}

// GetBundlingInfo returns information about the bundling result
func (efb *EFBOptimizer) GetBundlingInfo(bundles []FeatureBundle, originalCols int) EFBInfo {
	bundledFeatureCount := 0
	for _, bundle := range bundles {
		bundledFeatureCount += len(bundle.Features)
	}

	newFeatureCount := originalCols - bundledFeatureCount + len(bundles)
	compressionRatio := float64(originalCols) / float64(newFeatureCount)

	return EFBInfo{
		OriginalFeatures: originalCols,
		BundledFeatures:  newFeatureCount,
		NumBundles:       len(bundles),
		CompressionRatio: compressionRatio,
		Bundles:          bundles,
	}
}

// IsEFBBeneficial determines if EFB would be beneficial for the dataset
func (efb *EFBOptimizer) IsEFBBeneficial(X mat.Matrix) bool {
	rows, cols := X.Dims()

	// EFB is beneficial for high-dimensional sparse datasets
	if cols < 10 {
		return false // Too few features
	}

	// Calculate overall sparsity
	totalElements := rows * cols
	zeroElements := 0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(X.At(i, j)) < 1e-10 {
				zeroElements++
			}
		}
	}

	overallSparsity := float64(zeroElements) / float64(totalElements)

	// EFB is beneficial if dataset is sparse enough
	return overallSparsity >= efb.SparsityRatio
}

// WithMaxConflictRate sets the maximum conflict rate
func (efb *EFBOptimizer) WithMaxConflictRate(rate float64) *EFBOptimizer {
	efb.MaxConflictRate = rate
	return efb
}

// WithMaxBundles sets the maximum number of bundles
func (efb *EFBOptimizer) WithMaxBundles(bundles int) *EFBOptimizer {
	efb.MaxBundles = bundles
	return efb
}

// WithSparsityRatio sets the minimum sparsity ratio
func (efb *EFBOptimizer) WithSparsityRatio(ratio float64) *EFBOptimizer {
	efb.SparsityRatio = ratio
	return efb
}
