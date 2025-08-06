package parallel

import (
	"runtime"
	"sync"
)

// Parallelize は、指定された総数(items)をCPUコア数に応じて分割し、
// 各範囲(start, end)に対して指定された関数(fn)を並列実行する
func Parallelize(items int, fn func(start, end int)) {
	if items == 0 {
		return
	}

	// 利用可能なCPUコア数を取得
	numWorkers := runtime.NumCPU()
	if numWorkers > items {
		numWorkers = items // アイテム数より多くのワーカーは不要
	}

	// 各ワーカーが担当するアイテム数を計算（天井割り算）
	chunkSize := (items + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup

	// CPUコア数分のワーカーを起動
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > items {
			end = items
		}

		// 担当する範囲がなければスキップ
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}

	// 全てのワーカーの処理が終わるのを待つ
	wg.Wait()
}

// ParallelizeWithThreshold は、アイテム数が閾値を超えた場合のみ並列化を行う
// threshold以下の場合は、通常の逐次処理を実行する
func ParallelizeWithThreshold(items int, threshold int, fn func(start, end int)) {
	if items <= threshold {
		// 閾値以下の場合は逐次処理
		fn(0, items)
		return
	}

	// 閾値を超えた場合は並列処理
	Parallelize(items, fn)
}