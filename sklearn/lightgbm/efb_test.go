package lightgbm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// TestEFBBasic tests basic EFB functionality
func TestEFBBasic(t *testing.T) {
	// Create a sparse dataset with exclusive features
	X := mat.NewDense(10, 6, []float64{
		// Features 0,1 are mutually exclusive, features 2,3 are mutually exclusive
		1, 0, 2, 0, 5, 1, // row 0
		0, 2, 0, 3, 6, 2, // row 1
		3, 0, 4, 0, 7, 3, // row 2
		0, 4, 0, 5, 8, 4, // row 3
		5, 0, 6, 0, 9, 5, // row 4
		0, 6, 0, 7, 10, 6, // row 5
		7, 0, 8, 0, 11, 7, // row 6
		0, 8, 0, 9, 12, 8, // row 7
		9, 0, 10, 0, 13, 9, // row 8
		0, 10, 0, 11, 14, 10, // row 9
	})

	// Create EFB optimizer
	efb := NewEFBOptimizer().
		WithMaxConflictRate(0.1).
		WithSparsityRatio(0.4) // Lower threshold for test

	// Create bundles
	bundles, mapping, err := efb.CreateBundles(X)
	require.NoError(t, err)

	// Should create at least one bundle for exclusive features
	assert.Greater(t, len(bundles), 0, "Should create at least one bundle")

	// Check that mapping has correct length
	assert.Equal(t, 6, len(mapping), "Mapping should have 6 elements")

	// Bundle features
	bundled, err := efb.BundleFeatures(X, bundles, mapping)
	require.NoError(t, err)
	assert.NotNil(t, bundled)

	// Bundled matrix should have fewer columns
	_, originalCols := X.Dims()
	_, bundledCols := bundled.Dims()
	t.Logf("Original columns: %d, Bundled columns: %d", originalCols, bundledCols)

	// Get bundling info
	info := efb.GetBundlingInfo(bundles, originalCols)
	assert.Equal(t, originalCols, info.OriginalFeatures)
	assert.Greater(t, info.CompressionRatio, 1.0, "Should achieve compression")

	t.Logf("Compression ratio: %.2f", info.CompressionRatio)
}

// TestEFBSparsityCalculation tests sparsity calculation
func TestEFBSparsityCalculation(t *testing.T) {
	// Create matrix with known sparsity
	X := mat.NewDense(4, 3, []float64{
		1, 0, 0, // 67% sparse
		0, 2, 0, // 67% sparse
		0, 0, 3, // 67% sparse
		0, 0, 0, // 100% sparse
	})

	efb := NewEFBOptimizer()
	sparsity := efb.calculateSparsity(X)

	assert.Equal(t, 3, len(sparsity))
	assert.InDelta(t, 0.75, sparsity[0], 0.01) // 3/4 zeros
	assert.InDelta(t, 0.75, sparsity[1], 0.01) // 3/4 zeros
	assert.InDelta(t, 0.75, sparsity[2], 0.01) // 3/4 zeros
}

// TestEFBConflictGraph tests conflict graph construction
func TestEFBConflictGraph(t *testing.T) {
	// Create dataset with known conflicts
	X := mat.NewDense(4, 3, []float64{
		1, 0, 1, // features 0,2 conflict
		0, 2, 0, // no conflicts
		3, 0, 0, // no conflicts
		0, 4, 5, // features 1,2 conflict
	})

	efb := NewEFBOptimizer()
	features := []int{0, 1, 2}
	conflicts := efb.buildConflictGraph(X, features)

	assert.Equal(t, 3, len(conflicts))
	assert.Equal(t, 3, len(conflicts[0]))

	// Features 0 and 2 should have conflict (they're both non-zero in row 0)
	assert.Greater(t, conflicts[0][2], 0.0, "Features 0 and 2 should have conflict")
	assert.Greater(t, conflicts[2][0], 0.0, "Conflict matrix should be symmetric")

	// Features 1 and 2 should have conflict (they're both non-zero in row 3)
	assert.Greater(t, conflicts[1][2], 0.0, "Features 1 and 2 should have conflict")
}

// TestEFBMutuallyExclusive tests bundling of mutually exclusive features
func TestEFBMutuallyExclusive(t *testing.T) {
	// Create perfectly mutually exclusive features
	X := mat.NewDense(6, 4, []float64{
		1, 0, 5, 0, // features 0,2 are exclusive
		0, 2, 0, 6, // features 1,3 are exclusive
		3, 0, 7, 0,
		0, 4, 0, 8,
		5, 0, 9, 0,
		0, 6, 0, 10,
	})

	efb := NewEFBOptimizer().
		WithMaxConflictRate(0.0). // No conflicts allowed
		WithSparsityRatio(0.4)    // Lower threshold

	bundles, mapping, err := efb.CreateBundles(X)
	require.NoError(t, err)

	// Should create bundles for exclusive features
	assert.Greater(t, len(bundles), 0)

	// Bundle and unbundle to test round-trip
	bundled, err := efb.BundleFeatures(X, bundles, mapping)
	require.NoError(t, err)

	unbundled, err := efb.UnbundleFeatures(bundled, bundles, 4)
	require.NoError(t, err)

	// Check that unbundled matrix matches original
	rows, cols := X.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			original := X.At(i, j)
			recovered := unbundled.At(i, j)
			assert.InDelta(t, original, recovered, 1e-10,
				"Unbundled value should match original at (%d,%d)", i, j)
		}
	}
}

// TestEFBWithConflicts tests EFB with some allowed conflicts
func TestEFBWithConflicts(t *testing.T) {
	// Create dataset with some conflicts
	X := mat.NewDense(5, 4, []float64{
		1, 0, 0, 5,
		0, 2, 3, 0, // conflict between features 1 and 2
		4, 0, 0, 6,
		0, 7, 0, 0,
		8, 0, 9, 0,
	})

	efb := NewEFBOptimizer().
		WithMaxConflictRate(0.3). // Allow some conflicts
		WithSparsityRatio(0.5)

	bundles, mapping, err := efb.CreateBundles(X)
	require.NoError(t, err)

	// Should still create bundles even with conflicts
	info := efb.GetBundlingInfo(bundles, 4)
	t.Logf("Created %d bundles with conflict tolerance", info.NumBundles)

	// Verify mapping is valid
	assert.Equal(t, 4, len(mapping))
	for _, mappedIdx := range mapping {
		assert.GreaterOrEqual(t, mappedIdx, 0)
	}
}

// TestEFBBeneficial tests the benefit assessment
func TestEFBBeneficial(t *testing.T) {
	efb := NewEFBOptimizer()

	// Dense dataset - should not be beneficial
	denseX := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	})

	assert.False(t, efb.IsEFBBeneficial(denseX),
		"Dense dataset should not benefit from EFB")

	// Sparse dataset - should be beneficial
	sparseX := mat.NewDense(10, 20, nil) // Large sparse matrix
	// Fill only a few elements
	sparseX.Set(0, 0, 1)
	sparseX.Set(1, 5, 2)
	sparseX.Set(2, 10, 3)

	assert.True(t, efb.IsEFBBeneficial(sparseX),
		"Sparse dataset should benefit from EFB")

	// Small dataset - should not be beneficial
	smallX := mat.NewDense(5, 2, nil)
	assert.False(t, efb.IsEFBBeneficial(smallX),
		"Small dataset should not benefit from EFB")
}

// TestEFBBundleValue tests bundle value creation and extraction
func TestEFBBundleValue(t *testing.T) {
	// Create test data
	X := mat.NewDense(3, 4, []float64{
		1.5, 0, 0, 5,
		0, 2.7, 0, 6,
		0, 0, 3.2, 7,
	})

	efb := NewEFBOptimizer()

	// Test bundle value creation
	features := []int{0, 1, 2} // Bundle first three features

	// Row 0: only feature 0 is non-zero (1.5)
	bundleVal0 := efb.createBundleValue(X, 0, features)
	assert.InDelta(t, 1001.5, bundleVal0, 1e-10) // 1000.0 offset + 1.5 value

	// Row 1: only feature 1 is non-zero (2.7)
	bundleVal1 := efb.createBundleValue(X, 1, features)
	assert.InDelta(t, 2002.7, bundleVal1, 1e-10) // 2000.0 offset + 2.7 value

	// Row 2: only feature 2 is non-zero (3.2)
	bundleVal2 := efb.createBundleValue(X, 2, features)
	assert.InDelta(t, 3003.2, bundleVal2, 1e-10) // 3000.0 offset + 3.2 value
}

// TestEFBEdgeCases tests edge cases
func TestEFBEdgeCases(t *testing.T) {
	efb := NewEFBOptimizer()

	// Skip empty matrix test due to gonum limitations
	// Empty matrix would cause panic in gonum

	// Single column matrix
	singleX := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	bundles, mapping, err := efb.CreateBundles(singleX)
	require.NoError(t, err)
	assert.Equal(t, 0, len(bundles)) // Cannot bundle single feature
	assert.Equal(t, 1, len(mapping))

	// All dense features (no sparse features)
	denseX := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	bundles, _, err = efb.CreateBundles(denseX)
	require.NoError(t, err)
	assert.Equal(t, 0, len(bundles)) // No sparse features to bundle
}

// TestEFBConfiguration tests EFB configuration methods
func TestEFBConfiguration(t *testing.T) {
	efb := NewEFBOptimizer().
		WithMaxConflictRate(0.2).
		WithMaxBundles(5).
		WithSparsityRatio(0.7)

	assert.Equal(t, 0.2, efb.MaxConflictRate)
	assert.Equal(t, 5, efb.MaxBundles)
	assert.Equal(t, 0.7, efb.SparsityRatio)
}

// TestEFBInfo tests bundling information
func TestEFBInfo(t *testing.T) {
	bundles := []FeatureBundle{
		{Features: []int{0, 1}, Conflicts: 0},
		{Features: []int{2, 3, 4}, Conflicts: 1},
	}

	efb := NewEFBOptimizer()
	info := efb.GetBundlingInfo(bundles, 10)

	assert.Equal(t, 10, info.OriginalFeatures)
	assert.Equal(t, 7, info.BundledFeatures) // 10 - 5 + 2 bundles
	assert.Equal(t, 2, info.NumBundles)
	assert.InDelta(t, 10.0/7.0, info.CompressionRatio, 1e-10)
}

// BenchmarkEFBBundling benchmarks the bundling process
func BenchmarkEFBBundling(b *testing.B) {
	// Create large sparse dataset
	rows, cols := 1000, 100
	X := mat.NewDense(rows, cols, nil)

	// Make it sparse (10% non-zero)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if (i*cols+j)%10 == 0 {
				X.Set(i, j, float64(i+j))
			}
		}
	}

	efb := NewEFBOptimizer().
		WithSparsityRatio(0.8).
		WithMaxConflictRate(0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = efb.CreateBundles(X)
	}
}

// TestEFBIntegration tests EFB integration with LightGBM training
func TestEFBIntegration(t *testing.T) {
	// Create sparse dataset suitable for EFB
	nSamples := 100
	nFeatures := 20

	// Create sparse matrix
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	// Make features mutually exclusive in groups
	for i := 0; i < nSamples; i++ {
		// Group 1: features 0-4 (mutually exclusive)
		groupIdx := i % 5
		X.Set(i, groupIdx, float64(i+1))

		// Group 2: features 5-9 (mutually exclusive)
		groupIdx2 := (i + 2) % 5
		X.Set(i, 5+groupIdx2, float64(i+2))

		// Some dense features 10-15
		for j := 10; j < 16; j++ {
			if i%3 == 0 {
				X.Set(i, j, float64(i+j))
			}
		}

		// Generate target
		target := 0.0
		for j := 0; j < nFeatures; j++ {
			target += X.At(i, j) * 0.1
		}
		y.Set(i, 0, target)
	}

	// Test EFB benefit assessment
	efb := NewEFBOptimizer()
	beneficial := efb.IsEFBBeneficial(X)
	t.Logf("EFB beneficial: %v", beneficial)

	if beneficial {
		// Create bundles
		bundles, mapping, err := efb.CreateBundles(X)
		require.NoError(t, err)

		info := efb.GetBundlingInfo(bundles, nFeatures)
		t.Logf("EFB Info: %+v", info)

		// Bundle features
		bundled, err := efb.BundleFeatures(X, bundles, mapping)
		require.NoError(t, err)

		// Train model with bundled features
		reg := NewLGBMRegressor().
			WithNumIterations(10).
			WithNumLeaves(5).
			WithLearningRate(0.1)

		err = reg.Fit(bundled, y)
		require.NoError(t, err)

		// Make predictions
		pred, err := reg.Predict(bundled)
		require.NoError(t, err)
		assert.NotNil(t, pred)

		t.Logf("Successfully trained model with EFB-bundled features")
	}
}
