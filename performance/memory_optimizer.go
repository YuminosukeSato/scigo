package performance

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

// MatrixPool provides object pooling for matrices to reduce GC pressure
type MatrixPool struct {
	pool     sync.Pool
	maxSize  int
	inUse    int64
	created  int64
	recycled int64
	mu       sync.RWMutex
	stats    PoolStats
}

// PoolStats tracks pool performance metrics
type PoolStats struct {
	TotalAllocated   int64
	TotalRecycled    int64
	CurrentInUse     int64
	PeakUsage        int64
	AverageReuseRate float64
}

// NewMatrixPool creates a new matrix pool
func NewMatrixPool(maxSize int) *MatrixPool {
	mp := &MatrixPool{
		maxSize: maxSize,
	}
	
	mp.pool = sync.Pool{
		New: func() interface{} {
			atomic.AddInt64(&mp.created, 1)
			return &PooledMatrix{
				pool: mp,
			}
		},
	}
	
	return mp
}

// Get retrieves a matrix from the pool
func (mp *MatrixPool) Get(rows, cols int) *PooledMatrix {
	atomic.AddInt64(&mp.inUse, 1)
	
	m := mp.pool.Get().(*PooledMatrix)
	
	// Resize if necessary
	if m.data == nil || len(m.data) < rows*cols {
		m.data = make([]float64, rows*cols)
	}
	
	m.rows = rows
	m.cols = cols
	m.released = false
	
	// Update stats
	current := atomic.LoadInt64(&mp.inUse)
	mp.updatePeakUsage(current)
	
	return m
}

// Put returns a matrix to the pool
func (mp *MatrixPool) Put(m *PooledMatrix) {
	if m.released {
		return // Already released
	}
	
	m.released = true
	atomic.AddInt64(&mp.inUse, -1)
	atomic.AddInt64(&mp.recycled, 1)
	
	// Clear the data for reuse
	for i := range m.data {
		m.data[i] = 0
	}
	
	mp.pool.Put(m)
}

// GetStats returns current pool statistics
func (mp *MatrixPool) GetStats() PoolStats {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	
	total := atomic.LoadInt64(&mp.created)
	recycled := atomic.LoadInt64(&mp.recycled)
	inUse := atomic.LoadInt64(&mp.inUse)
	
	reuseRate := float64(0)
	if total > 0 {
		reuseRate = float64(recycled) / float64(total)
	}
	
	return PoolStats{
		TotalAllocated:   total,
		TotalRecycled:    recycled,
		CurrentInUse:     inUse,
		PeakUsage:        mp.stats.PeakUsage,
		AverageReuseRate: reuseRate,
	}
}

func (mp *MatrixPool) updatePeakUsage(current int64) {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	if current > mp.stats.PeakUsage {
		mp.stats.PeakUsage = current
	}
}

// PooledMatrix is a matrix that can be returned to a pool
type PooledMatrix struct {
	data     []float64
	rows     int
	cols     int
	pool     *MatrixPool
	released bool
}

// At returns the value at (i, j)
func (m *PooledMatrix) At(i, j int) float64 {
	return m.data[i*m.cols+j]
}

// Set sets the value at (i, j)
func (m *PooledMatrix) Set(i, j int, v float64) {
	m.data[i*m.cols+j] = v
}

// Dims returns the dimensions
func (m *PooledMatrix) Dims() (int, int) {
	return m.rows, m.cols
}

// Release returns the matrix to the pool
func (m *PooledMatrix) Release() {
	if m.pool != nil {
		m.pool.Put(m)
	}
}

// ToMat converts to a gonum matrix
func (m *PooledMatrix) ToMat() mat.Matrix {
	return mat.NewDense(m.rows, m.cols, m.data)
}

// ZeroCopyMatrix provides zero-copy matrix operations
type ZeroCopyMatrix struct {
	data   unsafe.Pointer
	rows   int
	cols   int
	stride int
}

// NewZeroCopyMatrix creates a zero-copy matrix from existing data
func NewZeroCopyMatrix(data []float64, rows, cols int) *ZeroCopyMatrix {
	if len(data) < rows*cols {
		panic("data too small for matrix dimensions")
	}
	
	return &ZeroCopyMatrix{
		data:   unsafe.Pointer(&data[0]),
		rows:   rows,
		cols:   cols,
		stride: cols,
	}
}

// At returns the value at (i, j) with bounds checking
func (m *ZeroCopyMatrix) At(i, j int) float64 {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(fmt.Sprintf("matrix index out of range: (%d, %d) for matrix of size (%d, %d)", i, j, m.rows, m.cols))
	}
	ptr := (*float64)(unsafe.Pointer(uintptr(m.data) + 
		uintptr(i*m.stride+j)*unsafe.Sizeof(float64(0))))
	return *ptr
}

// Set sets the value at (i, j) with bounds checking
func (m *ZeroCopyMatrix) Set(i, j int, v float64) {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(fmt.Sprintf("matrix index out of range: (%d, %d) for matrix of size (%d, %d)", i, j, m.rows, m.cols))
	}
	ptr := (*float64)(unsafe.Pointer(uintptr(m.data) + 
		uintptr(i*m.stride+j)*unsafe.Sizeof(float64(0))))
	*ptr = v
}

// Slice creates a view of a submatrix without copying
func (m *ZeroCopyMatrix) Slice(i0, i1, j0, j1 int) *ZeroCopyMatrix {
	if i0 < 0 || i1 > m.rows || j0 < 0 || j1 > m.cols || i0 >= i1 || j0 >= j1 {
		panic(fmt.Sprintf("invalid slice bounds: [%d:%d, %d:%d] for matrix of size (%d, %d)", i0, i1, j0, j1, m.rows, m.cols))
	}
	offset := i0*m.stride + j0
	ptr := (*float64)(unsafe.Pointer(uintptr(m.data) + 
		uintptr(offset)*unsafe.Sizeof(float64(0))))
	
	return &ZeroCopyMatrix{
		data:   unsafe.Pointer(ptr),
		rows:   i1 - i0,
		cols:   j1 - j0,
		stride: m.stride,
	}
}

// Dims returns the dimensions
func (m *ZeroCopyMatrix) Dims() (int, int) {
	return m.rows, m.cols
}

// GCOptimizer manages garbage collection for better performance
type GCOptimizer struct {
	gcPercent     int
	maxPause      time.Duration
	memLimit      int64
	enabled       bool
	mu            sync.RWMutex
	lastGC        time.Time
	gcCount       uint32
	totalPause    time.Duration
}

// NewGCOptimizer creates a new GC optimizer
func NewGCOptimizer(gcPercent int, maxPause time.Duration) *GCOptimizer {
	return &GCOptimizer{
		gcPercent: gcPercent,
		maxPause:  maxPause,
		memLimit:  1 << 30, // 1GB default
		enabled:   true,
	}
}

// Start begins GC optimization
func (g *GCOptimizer) Start() {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	if !g.enabled {
		return
	}
	
	// Set GC percentage
	debug.SetGCPercent(g.gcPercent)
	
	// Set memory limit if supported
	debug.SetMemoryLimit(g.memLimit)
	
	// Start monitoring goroutine
	go g.monitor()
}

// Stop stops GC optimization
func (g *GCOptimizer) Stop() {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	g.enabled = false
	debug.SetGCPercent(100) // Reset to default
}

// ForceGC forces a garbage collection if needed
func (g *GCOptimizer) ForceGC() {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	if time.Since(g.lastGC) > g.maxPause {
		runtime.GC()
		g.lastGC = time.Now()
		g.gcCount++
	}
}

// GetStats returns GC optimization statistics
func (g *GCOptimizer) GetStats() map[string]interface{} {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	
	avgPause := time.Duration(0)
	if g.gcCount > 0 {
		avgPause = g.totalPause / time.Duration(g.gcCount)
	}
	
	return map[string]interface{}{
		"gc_count":       g.gcCount,
		"last_gc":        g.lastGC,
		"avg_pause":      avgPause,
		"heap_alloc":     ms.HeapAlloc,
		"heap_sys":       ms.HeapSys,
		"heap_objects":   ms.HeapObjects,
		"gc_cpu_percent": ms.GCCPUFraction * 100,
	}
}

func (g *GCOptimizer) monitor() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		g.mu.RLock()
		enabled := g.enabled
		g.mu.RUnlock()
		
		if !enabled {
			return
		}
		
		// Check memory usage and trigger GC if needed
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		
		if ms.HeapAlloc > uint64(g.memLimit)*9/10 { // 90% of limit
			g.ForceGC()
		}
	}
}

// MemoryEfficientBatch processes data in memory-efficient batches
type MemoryEfficientBatch struct {
	maxMemory   int64
	currentUsed int64
	mu          sync.Mutex
}

// NewMemoryEfficientBatch creates a memory-efficient batch processor
func NewMemoryEfficientBatch(maxMemoryMB int64) *MemoryEfficientBatch {
	return &MemoryEfficientBatch{
		maxMemory: maxMemoryMB * 1024 * 1024,
	}
}

// CanAllocate checks if allocation is possible within memory limits
func (m *MemoryEfficientBatch) CanAllocate(bytes int64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	return m.currentUsed+bytes <= m.maxMemory
}

// Allocate tracks memory allocation
func (m *MemoryEfficientBatch) Allocate(bytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.currentUsed+bytes > m.maxMemory {
		return fmt.Errorf("memory limit exceeded: %d + %d > %d", 
			m.currentUsed, bytes, m.maxMemory)
	}
	
	m.currentUsed += bytes
	return nil
}

// Free tracks memory deallocation
func (m *MemoryEfficientBatch) Free(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.currentUsed -= bytes
	if m.currentUsed < 0 {
		m.currentUsed = 0
	}
}

// GetUsage returns current memory usage
func (m *MemoryEfficientBatch) GetUsage() (used, max int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	return m.currentUsed, m.maxMemory
}

// OptimizeDataLayout optimizes data layout for cache efficiency
func OptimizeDataLayout(data []float64, rows, cols int) []float64 {
	// Convert row-major to column-major for better cache locality
	// in certain operations
	optimized := make([]float64, len(data))
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			optimized[j*rows+i] = data[i*cols+j]
		}
	}
	
	return optimized
}

// AlignedAlloc allocates aligned memory for SIMD operations
func AlignedAlloc(size int, alignment int) []float64 {
	// Allocate extra space for alignment
	raw := make([]float64, size+alignment/8)
	
	// Find aligned address
	addr := uintptr(unsafe.Pointer(&raw[0]))
	offset := (alignment - int(addr%uintptr(alignment))) % alignment
	
	// Return aligned slice
	return raw[offset/8 : offset/8+size]
}