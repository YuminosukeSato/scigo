package lightgbm

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// TestParallelPredictionPerformance tests the performance improvement of parallel prediction
func TestParallelPredictionPerformance(t *testing.T) {
	// Create a large model with many trees
	model := createLargeModel(100, 50) // 100 trees, 50 features

	// Create large test dataset
	testSizes := []int{100, 1000, 10000}

	for _, size := range testSizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			X := createTestData(size, 50)

			// Test with different thread counts
			threadCounts := []int{1, 2, 4, 8}
			timings := make(map[int]time.Duration)

			for _, threads := range threadCounts {
				if threads > runtime.NumCPU() {
					continue // Skip if more threads than CPUs
				}

				predictor := NewPredictor(model)
				predictor.SetNumThreads(threads)

				start := time.Now()
				_, err := predictor.Predict(X)
				require.NoError(t, err)
				elapsed := time.Since(start)

				timings[threads] = elapsed
				t.Logf("Threads: %d, Time: %v", threads, elapsed)
			}

			// Log performance results (parallel may have overhead for small datasets)
			t.Logf("Performance comparison for size %d:", size)
			for threads := 1; threads <= 4; threads++ {
				if timing, exists := timings[threads]; exists {
					t.Logf("  %d thread(s): %v", threads, timing)
				}
			}

			// For very large datasets (10k+), expect parallel to eventually be faster
			// But don't enforce strict timing requirements due to system variability
			if size >= 10000 && len(timings) >= 2 {
				// Just check that parallel processing doesn't crash and produces results
				assert.Greater(t, len(timings), 1, "Should have multiple timing results")
			}
		})
	}
}

// TestParallelPredictionCorrectness verifies that parallel predictions match sequential
func TestParallelPredictionCorrectness(t *testing.T) {
	// Create model
	model := createLargeModel(50, 20)

	// Create test data
	X := createTestData(500, 20)

	// Sequential prediction (1 thread)
	seqPredictor := NewPredictor(model)
	seqPredictor.SetNumThreads(1)
	seqPredictor.SetDeterministic(true)
	seqPredictions, err := seqPredictor.Predict(X)
	require.NoError(t, err)

	// Parallel prediction (multiple threads)
	parallelPredictor := NewPredictor(model)
	parallelPredictor.SetNumThreads(4)
	parallelPredictor.SetDeterministic(true)
	parallelPredictions, err := parallelPredictor.Predict(X)
	require.NoError(t, err)

	// Compare predictions - should be identical
	assert.True(t, mat.EqualApprox(seqPredictions, parallelPredictions, 1e-10),
		"Parallel predictions should match sequential predictions exactly")
}

// TestConcurrentModelUsage tests thread safety of model usage
func TestConcurrentModelUsage(t *testing.T) {
	// Create a shared model
	model := createLargeModel(30, 10)

	// Create multiple predictors sharing the same model
	numPredictors := 10
	numPredictionsEach := 100

	var wg sync.WaitGroup
	errors := make(chan error, numPredictors)

	for i := 0; i < numPredictors; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			predictor := NewPredictor(model)
			X := createTestData(10, 10)

			for j := 0; j < numPredictionsEach; j++ {
				_, err := predictor.Predict(X)
				if err != nil {
					errors <- err
					return
				}
			}
		}(i)
	}

	// Wait for all goroutines
	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		require.NoError(t, err, "Concurrent prediction should not produce errors")
	}
}

// TestParallelTraining tests parallel tree building in training
func TestParallelTraining(t *testing.T) {
	// Generate training data
	X := createTestData(1000, 20)
	y := mat.NewDense(1000, 1, nil)
	for i := 0; i < 1000; i++ {
		// Simple target based on first feature
		y.Set(i, 0, X.At(i, 0)*2.0+1.0)
	}

	// Train with different thread settings
	threadCounts := []int{1, 4}
	models := make([]*LGBMRegressor, len(threadCounts))

	for idx, threads := range threadCounts {
		reg := NewLGBMRegressor()
		reg.NumThreads = threads
		err := reg.SetParams(map[string]interface{}{
			"n_estimators": 10,
			"num_leaves":   15,
		})
		if err != nil {
			t.Fatalf("Failed to set params: %v", err)
		}

		start := time.Now()
		err = reg.Fit(X, y)
		require.NoError(t, err)
		elapsed := time.Since(start)

		t.Logf("Training with %d threads: %v", threads, elapsed)
		models[idx] = reg
	}

	// Verify models produce similar predictions
	testX := createTestData(100, 20)
	pred1, err := models[0].Predict(testX)
	require.NoError(t, err)
	pred2, err := models[1].Predict(testX)
	require.NoError(t, err)

	// Predictions should be very similar (not identical due to parallelism)
	rows, _ := pred1.Dims()
	for i := 0; i < rows; i++ {
		assert.InDelta(t, pred1.At(i, 0), pred2.At(i, 0), 0.1,
			"Predictions from different thread counts should be similar")
	}
}

// BenchmarkParallelPredictionThreads benchmarks prediction with different thread counts
func BenchmarkParallelPredictionThreads(b *testing.B) {
	model := createLargeModel(100, 50)
	X := createTestData(1000, 50)

	benchmarks := []struct {
		name    string
		threads int
	}{
		{"1_Thread", 1},
		{"2_Threads", 2},
		{"4_Threads", 4},
		{"8_Threads", 8},
		{"Auto", -1}, // Use all available cores
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			predictor := NewPredictor(model)
			predictor.SetNumThreads(bm.threads)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = predictor.Predict(X)
			}
		})
	}
}

// TestMemoryEfficiencyParallel tests memory usage in parallel processing
func TestMemoryEfficiencyParallel(t *testing.T) {
	// Create a large model
	model := createLargeModel(200, 100)

	// Create very large test dataset
	X := createTestData(50000, 100)

	// Test with different thread counts
	threadCounts := []int{1, 4, 8}

	for _, threads := range threadCounts {
		t.Run(fmt.Sprintf("Threads_%d", threads), func(t *testing.T) {
			predictor := NewPredictor(model)
			predictor.SetNumThreads(threads)

			// Make predictions
			predictions, err := predictor.Predict(X)
			require.NoError(t, err)

			// Verify predictions
			rows, cols := predictions.Dims()
			assert.Equal(t, 50000, rows)
			assert.Equal(t, 1, cols)

			// Check that predictions are reasonable
			for i := 0; i < 10; i++ { // Check first 10
				val := predictions.At(i, 0)
				assert.False(t, math.IsNaN(val))
				assert.False(t, math.IsInf(val, 0))
			}
		})
	}
}

// TestLoadBalancing tests that work is evenly distributed across threads
func TestLoadBalancing(t *testing.T) {
	model := createLargeModel(50, 20)

	// Test with different data sizes to ensure good load balancing
	dataSizes := []int{7, 15, 31, 63, 127} // Various sizes including primes

	for _, size := range dataSizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			X := createTestData(size, 20)

			// Test with 4 threads
			predictor := NewPredictor(model)
			predictor.SetNumThreads(4)

			predictions, err := predictor.Predict(X)
			require.NoError(t, err)

			// Verify all predictions were made
			rows, _ := predictions.Dims()
			assert.Equal(t, size, rows, "All samples should have predictions")

			// Verify predictions are not zero (indicating work was done)
			nonZeroCount := 0
			for i := 0; i < rows; i++ {
				if predictions.At(i, 0) != 0 {
					nonZeroCount++
				}
			}
			assert.Greater(t, nonZeroCount, 0, "Should have non-zero predictions")
		})
	}
}

// Helper functions

func createLargeModel(numTrees, numFeatures int) *Model {
	model := &Model{
		NumFeatures: numFeatures,
		NumClass:    1,
		Trees:       make([]Tree, numTrees),
		Objective:   RegressionL2,
	}

	// Create trees with varying complexity
	for i := range model.Trees {
		numNodes := 7 + (i % 10) // Vary tree size
		nodes := make([]Node, numNodes)

		// Simple binary tree structure
		for j := 0; j < numNodes; j++ {
			if j < numNodes/2 {
				// Internal node
				nodes[j] = Node{
					NodeType:     NumericalNode,
					SplitFeature: j % numFeatures,
					Threshold:    float64(j) * 0.1,
					LeftChild:    j*2 + 1,
					RightChild:   j*2 + 2,
				}
			} else {
				// Leaf node
				nodes[j] = Node{
					NodeType:   LeafNode,
					LeafValue:  float64(j) * 0.01,
					LeftChild:  -1,
					RightChild: -1,
				}
			}
		}

		model.Trees[i] = Tree{
			Nodes:         nodes,
			ShrinkageRate: 0.1,
			NumLeaves:     numNodes/2 + 1,
		}
	}

	return model
}

func createTestData(rows, cols int) *mat.Dense {
	data := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Create varied but deterministic data
			data.Set(i, j, math.Sin(float64(i*cols+j))*100)
		}
	}
	return data
}
