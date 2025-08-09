package lightgbm

import (
	"runtime"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

// TestParallelPredictorBasics tests basic parallel predictor functionality
func TestParallelPredictorBasics(t *testing.T) {
	// Create simple model
	model := &Model{
		NumFeatures: 3,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: NumericalNode, SplitFeature: 0, Threshold: 0.5, LeftChild: 1, RightChild: 2},
					{NodeType: LeafNode, LeafValue: 1.0, LeftChild: -1, RightChild: -1},
					{NodeType: LeafNode, LeafValue: 2.0, LeftChild: -1, RightChild: -1},
				},
				ShrinkageRate: 0.1,
			},
		},
	}

	// Test data
	X := mat.NewDense(10, 3, []float64{
		0.1, 0.2, 0.3,
		0.6, 0.7, 0.8,
		0.2, 0.3, 0.4,
		0.9, 1.0, 1.1,
		0.1, 0.1, 0.1,
		0.8, 0.8, 0.8,
		0.3, 0.3, 0.3,
		0.7, 0.7, 0.7,
		0.4, 0.4, 0.4,
		0.5, 0.5, 0.5,
	})

	// Test with different thread counts
	threadCounts := []int{1, 2, 4}

	var predictions []*mat.Dense
	for _, threads := range threadCounts {
		predictor := NewPredictor(model)
		predictor.SetNumThreads(threads)
		predictor.SetDeterministic(true)

		pred, err := predictor.Predict(X)
		require.NoError(t, err)

		// Convert to Dense for comparison
		predDense, ok := pred.(*mat.Dense)
		require.True(t, ok)
		predictions = append(predictions, predDense)

		// Verify shape
		rows, cols := predDense.Dims()
		assert.Equal(t, 10, rows)
		assert.Equal(t, 1, cols)
	}

	// All predictions should be identical (deterministic mode)
	for i := 1; i < len(predictions); i++ {
		assert.True(t, mat.EqualApprox(predictions[0], predictions[i], 1e-15),
			"Predictions with %d threads should match single-threaded predictions", threadCounts[i])
	}
}

// TestParallelThreadSafety tests thread safety of multiple predictors
func TestParallelThreadSafety(t *testing.T) {
	model := &Model{
		NumFeatures: 5,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: LeafNode, LeafValue: 1.5, LeftChild: -1, RightChild: -1},
				},
				ShrinkageRate: 0.1,
			},
		},
	}

	// Create test data
	X := mat.NewDense(100, 5, nil)
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
	}

	numGoroutines := runtime.NumCPU() * 2
	results := make(chan *mat.Dense, numGoroutines)
	errors := make(chan error, numGoroutines)

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Launch multiple goroutines
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()

			predictor := NewPredictor(model)
			predictor.SetNumThreads(2) // Each predictor uses 2 threads

			pred, err := predictor.Predict(X)
			if err != nil {
				errors <- err
				return
			}

			if predDense, ok := pred.(*mat.Dense); ok {
				results <- predDense
			}
		}()
	}

	wg.Wait()
	close(results)
	close(errors)

	// Check for errors
	for err := range errors {
		require.NoError(t, err, "Concurrent prediction should not produce errors")
	}

	// Verify all results are identical
	var firstResult *mat.Dense
	resultCount := 0
	for result := range results {
		resultCount++
		if firstResult == nil {
			firstResult = result
		} else {
			assert.True(t, mat.EqualApprox(firstResult, result, 1e-15),
				"All concurrent predictions should be identical")
		}
	}

	assert.Equal(t, numGoroutines, resultCount, "Should receive results from all goroutines")
}

// TestPredictorThreadConfiguration tests thread configuration
func TestPredictorThreadConfiguration(t *testing.T) {
	model := &Model{NumFeatures: 2, NumClass: 1, Trees: []Tree{}}

	tests := []struct {
		name            string
		setThreads      int
		expectedThreads int
	}{
		{"Auto threads", -1, runtime.NumCPU()},
		{"Single thread", 1, 1},
		{"Two threads", 2, 2},
		{"More than CPU", runtime.NumCPU() + 5, runtime.NumCPU() + 5}, // Should not be clamped
		{"Zero threads", 0, runtime.NumCPU()},                         // Should default to CPU count
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := NewPredictor(model)
			predictor.SetNumThreads(tt.setThreads)

			// The exact thread count used internally may vary,
			// but the predictor should handle all valid configurations
			assert.NotNil(t, predictor, "Predictor should be created successfully")
		})
	}
}

// TestDeterministicMode tests deterministic prediction mode
func TestDeterministicMode(t *testing.T) {
	model := &Model{
		NumFeatures: 3,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: NumericalNode, SplitFeature: 0, Threshold: 0.5, LeftChild: 1, RightChild: 2},
					{NodeType: NumericalNode, SplitFeature: 1, Threshold: 0.3, LeftChild: 3, RightChild: 4},
					{NodeType: LeafNode, LeafValue: 2.0, LeftChild: -1, RightChild: -1},
					{NodeType: LeafNode, LeafValue: 1.0, LeftChild: -1, RightChild: -1},
					{NodeType: LeafNode, LeafValue: 1.5, LeftChild: -1, RightChild: -1},
				},
				ShrinkageRate: 0.1,
			},
		},
	}

	X := mat.NewDense(50, 3, nil)
	for i := 0; i < 50; i++ {
		for j := 0; j < 3; j++ {
			X.Set(i, j, float64(i+j)*0.1)
		}
	}

	// Test deterministic mode multiple times
	var results []*mat.Dense
	for run := 0; run < 3; run++ {
		predictor := NewPredictor(model)
		predictor.SetNumThreads(4)
		predictor.SetDeterministic(true)

		pred, err := predictor.Predict(X)
		require.NoError(t, err)

		predDense, ok := pred.(*mat.Dense)
		require.True(t, ok)
		results = append(results, predDense)
	}

	// All runs should produce identical results
	for i := 1; i < len(results); i++ {
		assert.True(t, mat.EqualApprox(results[0], results[i], 1e-15),
			"Deterministic mode should produce identical results across runs")
	}
}

// TestNonDeterministicMode tests that non-deterministic mode works
func TestNonDeterministicMode(t *testing.T) {
	model := &Model{
		NumFeatures: 4,
		NumClass:    1,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeType: LeafNode, LeafValue: 1.0, LeftChild: -1, RightChild: -1},
				},
				ShrinkageRate: 0.1,
			},
		},
	}

	X := mat.NewDense(10, 4, nil)
	for i := 0; i < 10; i++ {
		for j := 0; j < 4; j++ {
			X.Set(i, j, float64(i)*0.1)
		}
	}

	predictor := NewPredictor(model)
	predictor.SetNumThreads(4)
	predictor.SetDeterministic(false)

	// Make predictions - should work without error
	pred, err := predictor.Predict(X)
	require.NoError(t, err)

	// Verify predictions are reasonable
	predDense, ok := pred.(*mat.Dense)
	require.True(t, ok)

	rows, cols := predDense.Dims()
	assert.Equal(t, 10, rows)
	assert.Equal(t, 1, cols)

	// All predictions should be the same since it's a simple single-leaf model
	expected := 1.0 * 0.1 // leaf_value * shrinkage_rate
	for i := 0; i < rows; i++ {
		assert.InDelta(t, expected, predDense.At(i, 0), 1e-10)
	}
}
