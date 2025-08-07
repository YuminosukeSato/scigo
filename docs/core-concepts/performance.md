# Performance Optimization

SciGo is designed for high-performance machine learning in production environments. This guide covers optimization techniques and best practices.

## Performance Philosophy

1. **Zero-Copy Operations**: Minimize data copying
2. **Cache-Friendly**: Optimize for CPU cache locality
3. **Parallel by Default**: Automatic parallelization for large datasets
4. **Memory Pooling**: Reuse allocations
5. **SIMD Optimization**: Leverage vectorized operations

## Benchmarking

### Performance Metrics

```go
package benchmark

import (
    "testing"
    "time"
    "runtime"
)

type Metrics struct {
    Duration    time.Duration
    Operations  int64
    Throughput  float64 // ops/sec
    MemoryUsed  uint64  // bytes
    Allocations uint64
}

func MeasurePerformance(fn func()) *Metrics {
    var m1, m2 runtime.MemStats
    runtime.ReadMemStats(&m1)
    
    start := time.Now()
    fn()
    duration := time.Since(start)
    
    runtime.ReadMemStats(&m2)
    
    return &Metrics{
        Duration:    duration,
        MemoryUsed:  m2.Alloc - m1.Alloc,
        Allocations: m2.Mallocs - m1.Mallocs,
    }
}
```

### Benchmark Suite

```go
func BenchmarkLinearRegression(b *testing.B) {
    sizes := []struct {
        name string
        rows int
        cols int
    }{
        {"small", 100, 10},
        {"medium", 1000, 100},
        {"large", 10000, 1000},
        {"xlarge", 100000, 1000},
    }
    
    for _, size := range sizes {
        b.Run("fit_"+size.name, func(b *testing.B) {
            X := mat.NewDense(size.rows, size.cols, nil)
            y := mat.NewVecDense(size.rows, nil)
            
            b.ResetTimer()
            b.ReportAllocs()
            
            for i := 0; i < b.N; i++ {
                lr := NewLinearRegression()
                lr.Fit(X, y)
            }
            
            b.ReportMetric(float64(size.rows), "samples/op")
        })
        
        b.Run("predict_"+size.name, func(b *testing.B) {
            lr := trainedModel(size.rows, size.cols)
            X := mat.NewDense(size.rows, size.cols, nil)
            
            b.ResetTimer()
            b.ReportAllocs()
            
            for i := 0; i < b.N; i++ {
                lr.Predict(X)
            }
            
            throughput := float64(size.rows*b.N) / b.Elapsed().Seconds()
            b.ReportMetric(throughput, "predictions/sec")
        })
    }
}
```

## Memory Optimization

### Memory Pooling

```go
package memory

import "sync"

// BufferPool manages reusable byte buffers
type BufferPool struct {
    pools []*sync.Pool
}

func NewBufferPool() *BufferPool {
    bp := &BufferPool{
        pools: make([]*sync.Pool, 20), // Up to 2^20 bytes
    }
    
    for i := range bp.pools {
        size := 1 << i
        bp.pools[i] = &sync.Pool{
            New: func() interface{} {
                return make([]float64, size)
            },
        }
    }
    
    return bp
}

func (bp *BufferPool) Get(size int) []float64 {
    // Find appropriate pool
    poolIdx := 0
    poolSize := 1
    
    for poolSize < size && poolIdx < len(bp.pools)-1 {
        poolIdx++
        poolSize <<= 1
    }
    
    buf := bp.pools[poolIdx].Get().([]float64)
    return buf[:size]
}

func (bp *BufferPool) Put(buf []float64) {
    size := cap(buf)
    poolIdx := 0
    poolSize := 1
    
    for poolSize < size && poolIdx < len(bp.pools)-1 {
        poolIdx++
        poolSize <<= 1
    }
    
    bp.pools[poolIdx].Put(buf[:poolSize])
}
```

### Zero-Copy Operations

```go
// View creates a view without copying data
func CreateView(data []float64, rows, cols int) mat.Matrix {
    return mat.NewDense(rows, cols, data) // Uses existing slice
}

// Slice operations don't copy
func ExtractColumns(X mat.Matrix, cols []int) mat.Matrix {
    rows, _ := X.Dims()
    result := mat.NewDense(rows, len(cols), nil)
    
    for j, col := range cols {
        result.SetCol(j, mat.Col(nil, col, X))
    }
    
    return result
}

// In-place operations
func ScaleInPlace(X mat.Matrix, scale float64) {
    X.(*mat.Dense).Scale(scale, X)
}
```

### Memory Layout Optimization

```go
// Row-major layout for cache efficiency
type OptimizedMatrix struct {
    data   []float64
    rows   int
    cols   int
    stride int // Align to cache line
}

func NewOptimizedMatrix(rows, cols int) *OptimizedMatrix {
    // Align stride to 64-byte cache line
    stride := (cols + 7) &^ 7
    
    return &OptimizedMatrix{
        data:   make([]float64, rows*stride),
        rows:   rows,
        cols:   cols,
        stride: stride,
    }
}

func (m *OptimizedMatrix) At(i, j int) float64 {
    return m.data[i*m.stride+j]
}

func (m *OptimizedMatrix) Set(i, j int, v float64) {
    m.data[i*m.stride+j] = v
}

// Cache-friendly matrix multiplication
func (m *OptimizedMatrix) Multiply(other *OptimizedMatrix) *OptimizedMatrix {
    result := NewOptimizedMatrix(m.rows, other.cols)
    
    // Tiled multiplication for cache efficiency
    tileSize := 64 // Fits in L1 cache
    
    for ii := 0; ii < m.rows; ii += tileSize {
        for jj := 0; jj < other.cols; jj += tileSize {
            for kk := 0; kk < m.cols; kk += tileSize {
                // Process tile
                for i := ii; i < min(ii+tileSize, m.rows); i++ {
                    for j := jj; j < min(jj+tileSize, other.cols); j++ {
                        sum := result.At(i, j)
                        for k := kk; k < min(kk+tileSize, m.cols); k++ {
                            sum += m.At(i, k) * other.At(k, j)
                        }
                        result.Set(i, j, sum)
                    }
                }
            }
        }
    }
    
    return result
}
```

## Parallel Processing

### Automatic Parallelization

```go
package parallel

import (
    "runtime"
    "sync"
)

// ParallelOptions configures parallel execution
type ParallelOptions struct {
    MinSize    int  // Minimum size for parallelization
    MaxWorkers int  // Maximum number of workers
    ChunkSize  int  // Size of work chunks
}

var DefaultOptions = ParallelOptions{
    MinSize:    1000,
    MaxWorkers: runtime.NumCPU(),
    ChunkSize:  0, // Auto-calculate
}

// ParallelFor executes function in parallel
func ParallelFor(n int, fn func(start, end int)) {
    if n < DefaultOptions.MinSize {
        // Execute serially for small datasets
        fn(0, n)
        return
    }
    
    workers := DefaultOptions.MaxWorkers
    if workers > n {
        workers = n
    }
    
    chunkSize := (n + workers - 1) / workers
    
    var wg sync.WaitGroup
    for i := 0; i < workers; i++ {
        start := i * chunkSize
        end := min(start+chunkSize, n)
        
        if start >= end {
            break
        }
        
        wg.Add(1)
        go func(s, e int) {
            defer wg.Done()
            fn(s, e)
        }(start, end)
    }
    
    wg.Wait()
}

// ParallelMap applies function to elements in parallel
func ParallelMap(data []float64, fn func(float64) float64) []float64 {
    result := make([]float64, len(data))
    
    ParallelFor(len(data), func(start, end int) {
        for i := start; i < end; i++ {
            result[i] = fn(data[i])
        }
    })
    
    return result
}

// ParallelReduce performs parallel reduction
func ParallelReduce(data []float64, fn func(float64, float64) float64) float64 {
    if len(data) == 0 {
        return 0
    }
    
    if len(data) < DefaultOptions.MinSize {
        // Serial reduction
        result := data[0]
        for i := 1; i < len(data); i++ {
            result = fn(result, data[i])
        }
        return result
    }
    
    // Parallel reduction
    workers := DefaultOptions.MaxWorkers
    chunkSize := (len(data) + workers - 1) / workers
    
    partials := make([]float64, workers)
    var wg sync.WaitGroup
    
    for i := 0; i < workers; i++ {
        start := i * chunkSize
        end := min(start+chunkSize, len(data))
        
        if start >= end {
            break
        }
        
        wg.Add(1)
        go func(idx, s, e int) {
            defer wg.Done()
            
            partial := data[s]
            for j := s + 1; j < e; j++ {
                partial = fn(partial, data[j])
            }
            partials[idx] = partial
        }(i, start, end)
    }
    
    wg.Wait()
    
    // Combine partials
    result := partials[0]
    for i := 1; i < workers; i++ {
        result = fn(result, partials[i])
    }
    
    return result
}
```

### Work Stealing

```go
type WorkQueue struct {
    tasks chan func()
    wg    sync.WaitGroup
}

func NewWorkQueue(workers int) *WorkQueue {
    wq := &WorkQueue{
        tasks: make(chan func(), workers*2),
    }
    
    // Start workers
    for i := 0; i < workers; i++ {
        go wq.worker()
    }
    
    return wq
}

func (wq *WorkQueue) worker() {
    for task := range wq.tasks {
        task()
        wq.wg.Done()
    }
}

func (wq *WorkQueue) Submit(task func()) {
    wq.wg.Add(1)
    wq.tasks <- task
}

func (wq *WorkQueue) Wait() {
    wq.wg.Wait()
}

func (wq *WorkQueue) Close() {
    close(wq.tasks)
}
```

## SIMD Optimization

### Vectorized Operations

```go
// Vectorized dot product
func DotProduct(a, b []float64) float64 {
    if len(a) != len(b) {
        panic("vectors must have same length")
    }
    
    var sum float64
    
    // Process 4 elements at a time (AVX)
    i := 0
    for ; i <= len(a)-4; i += 4 {
        sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
    }
    
    // Handle remaining elements
    for ; i < len(a); i++ {
        sum += a[i] * b[i]
    }
    
    return sum
}

// Vectorized addition
func VecAdd(dst, a, b []float64) {
    n := len(a)
    
    // Unroll loop for better vectorization
    i := 0
    for ; i <= n-8; i += 8 {
        dst[i] = a[i] + b[i]
        dst[i+1] = a[i+1] + b[i+1]
        dst[i+2] = a[i+2] + b[i+2]
        dst[i+3] = a[i+3] + b[i+3]
        dst[i+4] = a[i+4] + b[i+4]
        dst[i+5] = a[i+5] + b[i+5]
        dst[i+6] = a[i+6] + b[i+6]
        dst[i+7] = a[i+7] + b[i+7]
    }
    
    for ; i < n; i++ {
        dst[i] = a[i] + b[i]
    }
}

// Vectorized scaling
func VecScale(dst []float64, scale float64) {
    n := len(dst)
    
    // Process 4 elements at a time
    i := 0
    for ; i <= n-4; i += 4 {
        dst[i] *= scale
        dst[i+1] *= scale
        dst[i+2] *= scale
        dst[i+3] *= scale
    }
    
    for ; i < n; i++ {
        dst[i] *= scale
    }
}
```

### Assembly Optimization

```go
// +build amd64

package performance

// Implemented in assembly for maximum performance
//go:noescape
func dotProductASM(a, b []float64) float64

// Wrapper with fallback
func DotProductOptimized(a, b []float64) float64 {
    if hasAVX2() {
        return dotProductASM(a, b)
    }
    return DotProduct(a, b) // Fallback to Go implementation
}
```

```assembly
// dot_product_amd64.s
// func dotProductASM(a, b []float64) float64
TEXT ·dotProductASM(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI   // a slice base
    MOVQ b_base+24(FP), DI  // b slice base
    MOVQ a_len+8(FP), CX    // length
    
    VXORPD Y0, Y0, Y0       // Clear accumulator
    
    // Process 4 values per iteration
loop:
    CMPQ CX, $4
    JL done
    
    VMOVUPD (SI), Y1        // Load 4 from a
    VMOVUPD (DI), Y2        // Load 4 from b
    VMULPD Y1, Y2, Y3       // Multiply
    VADDPD Y3, Y0, Y0       // Accumulate
    
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $4, CX
    JMP loop
    
done:
    // Horizontal add
    VHADDPD Y0, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    
    MOVSD X0, ret+48(FP)
    RET
```

## Algorithm Optimization

### Algorithm Selection

```go
type MatrixMultiplier interface {
    Multiply(a, b mat.Matrix) mat.Matrix
}

// Choose optimal algorithm based on matrix size
func SelectMultiplier(rows, cols int) MatrixMultiplier {
    size := rows * cols
    
    switch {
    case size < 100:
        return &NaiveMultiplier{}
    case size < 10000:
        return &StrassenMultiplier{}
    default:
        return &ParallelStrassenMultiplier{}
    }
}

// Strassen's algorithm for large matrices
type StrassenMultiplier struct{}

func (s *StrassenMultiplier) Multiply(a, b mat.Matrix) mat.Matrix {
    n, _ := a.Dims()
    
    // Base case
    if n <= 64 {
        var c mat.Dense
        c.Mul(a, b)
        return &c
    }
    
    // Divide matrices
    half := n / 2
    a11, a12, a21, a22 := partitionMatrix(a, half)
    b11, b12, b21, b22 := partitionMatrix(b, half)
    
    // Compute products
    m1 := s.Multiply(addMatrices(a11, a22), addMatrices(b11, b22))
    m2 := s.Multiply(addMatrices(a21, a22), b11)
    m3 := s.Multiply(a11, subMatrices(b12, b22))
    m4 := s.Multiply(a22, subMatrices(b21, b11))
    m5 := s.Multiply(addMatrices(a11, a12), b22)
    m6 := s.Multiply(subMatrices(a21, a11), addMatrices(b11, b12))
    m7 := s.Multiply(subMatrices(a12, a22), addMatrices(b21, b22))
    
    // Combine results
    c11 := addMatrices(subMatrices(addMatrices(m1, m4), m5), m7)
    c12 := addMatrices(m3, m5)
    c21 := addMatrices(m2, m4)
    c22 := addMatrices(subMatrices(addMatrices(m1, m3), m2), m6)
    
    return combineMatrices(c11, c12, c21, c22)
}
```

### Approximation Algorithms

```go
// Approximate matrix multiplication for very large matrices
type RandomizedSVD struct {
    rank      int
    oversampling int
}

func (r *RandomizedSVD) Decompose(A mat.Matrix) (U, S, V mat.Matrix) {
    m, n := A.Dims()
    k := r.rank + r.oversampling
    
    // Random sampling matrix
    Omega := mat.NewDense(n, k, nil)
    for i := 0; i < n; i++ {
        for j := 0; j < k; j++ {
            Omega.Set(i, j, rand.NormFloat64())
        }
    }
    
    // Form sample matrix Y = A * Omega
    var Y mat.Dense
    Y.Mul(A, Omega)
    
    // Orthonormalize
    var Q mat.Dense
    qr := new(mat.QR)
    qr.Factorize(&Y)
    qr.QTo(&Q)
    
    // Form B = Q^T * A
    var B mat.Dense
    B.Mul(Q.T(), A)
    
    // SVD of small matrix B
    var svd mat.SVD
    svd.Factorize(&B, mat.SVDThin)
    
    // Recover U
    var uTilde mat.Dense
    svd.UTo(&uTilde)
    U.Mul(&Q, &uTilde)
    
    return U, S, V
}
```

## Caching Strategies

### Computation Cache

```go
type ComputationCache struct {
    cache map[uint64]interface{}
    mu    sync.RWMutex
    size  int
    maxSize int
}

func NewComputationCache(maxSize int) *ComputationCache {
    return &ComputationCache{
        cache:   make(map[uint64]interface{}),
        maxSize: maxSize,
    }
}

func (c *ComputationCache) Get(key uint64) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    val, exists := c.cache[key]
    return val, exists
}

func (c *ComputationCache) Set(key uint64, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if c.size >= c.maxSize {
        // Simple eviction: remove random entry
        for k := range c.cache {
            delete(c.cache, k)
            c.size--
            break
        }
    }
    
    c.cache[key] = value
    c.size++
}

// Hash matrix for caching
func HashMatrix(m mat.Matrix) uint64 {
    rows, cols := m.Dims()
    hash := uint64(rows) ^ uint64(cols<<32)
    
    // Sample some elements for hash
    for i := 0; i < min(rows, 10); i++ {
        for j := 0; j < min(cols, 10); j++ {
            bits := math.Float64bits(m.At(i, j))
            hash ^= bits * uint64((i+1)*(j+1))
        }
    }
    
    return hash
}
```

### Model Cache

```go
type ModelCache struct {
    models map[string]*CachedModel
    mu     sync.RWMutex
    ttl    time.Duration
}

type CachedModel struct {
    model     Model
    loadTime  time.Time
    lastUsed  time.Time
    hitCount  int64
}

func (c *ModelCache) Get(name string) (Model, bool) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    cached, exists := c.models[name]
    if !exists {
        return nil, false
    }
    
    // Check TTL
    if time.Since(cached.loadTime) > c.ttl {
        delete(c.models, name)
        return nil, false
    }
    
    cached.lastUsed = time.Now()
    cached.hitCount++
    
    return cached.model, true
}

func (c *ModelCache) Evict() {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    // LRU eviction
    var oldest string
    var oldestTime time.Time
    
    for name, cached := range c.models {
        if oldest == "" || cached.lastUsed.Before(oldestTime) {
            oldest = name
            oldestTime = cached.lastUsed
        }
    }
    
    if oldest != "" {
        delete(c.models, oldest)
    }
}
```

## Profiling and Monitoring

### Performance Profiler

```go
import (
    "runtime/pprof"
    "runtime/trace"
)

type Profiler struct {
    cpuProfile *os.File
    memProfile *os.File
    traceFile  *os.File
}

func NewProfiler(prefix string) *Profiler {
    return &Profiler{
        cpuProfile: createFile(prefix + "_cpu.prof"),
        memProfile: createFile(prefix + "_mem.prof"),
        traceFile:  createFile(prefix + ".trace"),
    }
}

func (p *Profiler) StartCPUProfile() error {
    return pprof.StartCPUProfile(p.cpuProfile)
}

func (p *Profiler) StopCPUProfile() {
    pprof.StopCPUProfile()
}

func (p *Profiler) WriteMemProfile() error {
    runtime.GC()
    return pprof.WriteHeapProfile(p.memProfile)
}

func (p *Profiler) StartTrace() error {
    return trace.Start(p.traceFile)
}

func (p *Profiler) StopTrace() {
    trace.Stop()
}

func (p *Profiler) Close() {
    p.cpuProfile.Close()
    p.memProfile.Close()
    p.traceFile.Close()
}
```

### Runtime Metrics

```go
type RuntimeMetrics struct {
    startTime    time.Time
    samples      []Sample
    mu           sync.Mutex
}

type Sample struct {
    Timestamp   time.Time
    MemAlloc    uint64
    NumGoroutine int
    NumGC       uint32
}

func (m *RuntimeMetrics) Collect() {
    var ms runtime.MemStats
    runtime.ReadMemStats(&ms)
    
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.samples = append(m.samples, Sample{
        Timestamp:    time.Now(),
        MemAlloc:     ms.Alloc,
        NumGoroutine: runtime.NumGoroutine(),
        NumGC:        ms.NumGC,
    })
}

func (m *RuntimeMetrics) Report() string {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if len(m.samples) == 0 {
        return "No samples collected"
    }
    
    last := m.samples[len(m.samples)-1]
    duration := time.Since(m.startTime)
    
    return fmt.Sprintf(
        "Runtime: %v | Mem: %.2f MB | Goroutines: %d | GCs: %d",
        duration,
        float64(last.MemAlloc)/1024/1024,
        last.NumGoroutine,
        last.NumGC,
    )
}
```

## Performance Tips

### 1. Pre-allocation
```go
// Good: Pre-allocate slices
result := make([]float64, 0, expectedSize)
for _, v := range data {
    result = append(result, process(v))
}

// Bad: Growing slice dynamically
var result []float64
for _, v := range data {
    result = append(result, process(v))
}
```

### 2. Avoid Interface Boxing
```go
// Good: Use concrete types
func Sum(values []float64) float64 {
    var sum float64
    for _, v := range values {
        sum += v
    }
    return sum
}

// Bad: Interface boxing
func Sum(values []interface{}) float64 {
    var sum float64
    for _, v := range values {
        sum += v.(float64) // Boxing overhead
    }
    return sum
}
```

### 3. Loop Optimization
```go
// Good: Access patterns for cache locality
for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
        process(matrix[i*cols + j]) // Row-major order
    }
}

// Bad: Poor cache locality
for j := 0; j < cols; j++ {
    for i := 0; i < rows; i++ {
        process(matrix[i*cols + j]) // Column access in row-major
    }
}
```

### 4. Bounds Check Elimination
```go
// Good: Bounds check elimination
func Sum(a []float64) float64 {
    if len(a) == 0 {
        return 0
    }
    
    sum := a[0]
    a = a[1:]
    for i := range a { // BCE: compiler knows i is valid
        sum += a[i]
    }
    return sum
}
```

### 5. String Building
```go
// Good: Use strings.Builder
var b strings.Builder
for _, s := range strings {
    b.WriteString(s)
}
result := b.String()

// Bad: String concatenation
var result string
for _, s := range strings {
    result += s // Creates new string each time
}
```

## Performance Checklist

- [ ] Profile before optimizing
- [ ] Measure after each change
- [ ] Use benchmarks for regression testing
- [ ] Optimize hot paths first
- [ ] Consider memory allocation patterns
- [ ] Use parallel processing for large datasets
- [ ] Cache expensive computations
- [ ] Minimize interface conversions
- [ ] Align data structures to cache lines
- [ ] Use SIMD operations where possible
- [ ] Pre-allocate slices and maps
- [ ] Avoid unnecessary copying
- [ ] Use buffered I/O
- [ ] Pool temporary objects
- [ ] Profile in production environment

## Next Steps

- Explore [Benchmarking Guide](../advanced/benchmarking.md)
- Learn about [Memory Management](../advanced/memory.md)
- See [Parallel Processing](../advanced/parallel.md)
- Read [Production Guide](../tutorials/production.md)