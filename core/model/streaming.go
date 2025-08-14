package model

import (
	"context"

	"gonum.org/v1/gonum/mat"
)

// Batch represents a data batch for streaming learning
type Batch struct {
	X mat.Matrix // Feature matrix
	Y mat.Matrix // Target matrix
}

// StreamingEstimator provides channel-based streaming learning interface
type StreamingEstimator interface {
	IncrementalEstimator

	// FitStream trains the model from a data stream
	// Continues learning until the context is canceled or the channel is closed
	FitStream(ctx context.Context, dataChan <-chan *Batch) error

	// PredictStream performs real-time predictions on input stream
	// Output channel is closed when input channel is closed
	PredictStream(ctx context.Context, inputChan <-chan mat.Matrix) <-chan mat.Matrix

	// FitPredictStream performs learning and prediction simultaneously
	// Returns predictions while training on new data (test-then-train approach)
	FitPredictStream(ctx context.Context, dataChan <-chan *Batch) <-chan mat.Matrix
}

// StreamingMetrics provides metrics during streaming learning
type StreamingMetrics interface {
	OnlineMetrics

	// GetThroughput returns current throughput (samples/second)
	GetThroughput() float64

	// GetProcessedSamples returns total number of processed samples
	GetProcessedSamples() int64

	// GetAverageLatency returns average latency in milliseconds
	GetAverageLatency() float64

	// GetMemoryUsage returns current memory usage in bytes
	GetMemoryUsage() int64
}

// BufferedStreaming is a streaming interface with buffering capabilities
type BufferedStreaming interface {
	// SetBufferSize sets the size of streaming buffer
	SetBufferSize(size int)

	// GetBufferSize returns current buffer size
	GetBufferSize() int

	// FlushBuffer forces buffer flush
	FlushBuffer() error
}

// ParallelStreaming is an interface for parallel streaming processing
type ParallelStreaming interface {
	// SetWorkers sets the number of workers
	SetWorkers(n int)

	// GetWorkers returns current number of workers
	GetWorkers() int

	// SetBatchParallelism enables/disables intra-batch parallelism
	SetBatchParallelism(enabled bool)
}
