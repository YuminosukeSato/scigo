package parallel

import (
	"runtime"
	"sync"
)

// Parallelize divides the specified total number (items) according to the number of CPU cores,
// and executes the specified function (fn) in parallel for each range (start, end)
func Parallelize(items int, fn func(start, end int)) {
	if items == 0 {
		return
	}

	// Get the number of available CPU cores
	numWorkers := runtime.NumCPU()
	if numWorkers > items {
		numWorkers = items // No need for more workers than items
	}

	// Calculate the number of items each worker handles (ceiling division)
	chunkSize := (items + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup

	// Start workers equal to the number of CPU cores
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > items {
			end = items
		}

		// Skip if there's no range to handle
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}

	// Wait for all workers to finish processing
	wg.Wait()
}

// ParallelizeWithThreshold performs parallelization only when the number of items exceeds the threshold
// If below threshold, normal sequential processing is performed
func ParallelizeWithThreshold(items int, threshold int, fn func(start, end int)) {
	if items <= threshold {
		// Sequential processing when below threshold
		fn(0, items)
		return
	}

	// Parallel processing when above threshold
	Parallelize(items, fn)
}