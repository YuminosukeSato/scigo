package model

import (
	"context"
	"gonum.org/v1/gonum/mat"
)

// Batch はストリーミング学習用のデータバッチを表す
type Batch struct {
	X mat.Matrix // 特徴量行列
	Y mat.Matrix // ターゲット行列
}

// StreamingEstimator はチャネルベースのストリーミング学習を提供するインターフェース
type StreamingEstimator interface {
	IncrementalEstimator

	// FitStream はデータストリームからモデルを学習する
	// コンテキストがキャンセルされるまで、またはチャネルがクローズされるまで学習を継続
	FitStream(ctx context.Context, dataChan <-chan *Batch) error

	// PredictStream は入力ストリームに対してリアルタイム予測を行う
	// 入力チャネルがクローズされると出力チャネルもクローズされる
	PredictStream(ctx context.Context, inputChan <-chan mat.Matrix) <-chan mat.Matrix

	// FitPredictStream は学習と予測を同時に行う
	// 新しいデータで学習しながら、同時に予測も返す（test-then-train方式）
	FitPredictStream(ctx context.Context, dataChan <-chan *Batch) <-chan mat.Matrix
}

// StreamingMetrics はストリーミング学習中のメトリクスを提供
type StreamingMetrics interface {
	OnlineMetrics

	// GetThroughput は現在のスループット（サンプル/秒）を返す
	GetThroughput() float64

	// GetProcessedSamples は処理されたサンプル総数を返す
	GetProcessedSamples() int64

	// GetAverageLatency は平均レイテンシ（ミリ秒）を返す
	GetAverageLatency() float64

	// GetMemoryUsage は現在のメモリ使用量（バイト）を返す
	GetMemoryUsage() int64
}

// BufferedStreaming はバッファリング機能を持つストリーミングインターフェース
type BufferedStreaming interface {
	// SetBufferSize はストリーミングバッファのサイズを設定
	SetBufferSize(size int)

	// GetBufferSize は現在のバッファサイズを返す
	GetBufferSize() int

	// FlushBuffer はバッファを強制的にフラッシュ
	FlushBuffer() error
}

// ParallelStreaming は並列ストリーミング処理のインターフェース
type ParallelStreaming interface {
	// SetWorkers はワーカー数を設定
	SetWorkers(n int)

	// GetWorkers は現在のワーカー数を返す
	GetWorkers() int

	// SetBatchParallelism はバッチ内並列処理の有効/無効を設定
	SetBatchParallelism(enabled bool)
}
