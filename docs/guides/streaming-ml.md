# Streaming Machine Learning Guide

Comprehensive guide to online learning and streaming ML in SciGo.

## Overview

Streaming ML enables learning from continuous data streams, essential for real-time applications, large-scale systems, and evolving data distributions.

## Streaming Concepts

### Online vs Batch Learning

| Aspect | Batch Learning | Online Learning |
|--------|---------------|----------------|
| Data Access | All at once | One sample/batch at a time |
| Memory Usage | O(n) | O(1) or O(batch_size) |
| Model Updates | After full pass | After each sample |
| Adaptability | Static | Adapts to drift |
| Use Case | Static datasets | Streaming data |

## Online Learning Algorithms

### Stochastic Gradient Descent

```go
package streaming

import (
    "github.com/YuminosukeSato/scigo/sklearn/linear_model"
    "gonum.org/v1/gonum/mat"
)

func streamingSGD() {
    // Create SGD model for streaming
    sgd := linear_model.NewSGDRegressor(
        linear_model.WithLearningRate(0.01),
        linear_model.WithLearningRateSchedule("invscaling"),
        linear_model.WithEta0(0.01),
        linear_model.WithPowerT(0.25),
    )
    
    // Process streaming data
    for batch := range dataStream {
        // Partial fit on batch
        err := sgd.PartialFit(batch.X, batch.Y)
        if err != nil {
            log.Printf("Update failed: %v", err)
            continue
        }
        
        // Track performance
        if batch.ID % 100 == 0 {
            score := sgd.Score(batch.XVal, batch.YVal)
            log.Printf("Batch %d - Score: %.4f", batch.ID, score)
        }
    }
}
```

### Passive-Aggressive Algorithm

```go
type PassiveAggressive struct {
    weights   *mat.VecDense
    C         float64  // Regularization
    epsilon   float64  // Insensitivity
    mode      string   // "PA-I" or "PA-II"
    nFeatures int
}

func (pa *PassiveAggressive) PartialFit(x mat.Vector, y float64) {
    // Make prediction
    pred := mat.Dot(pa.weights, x)
    
    // Calculate loss
    loss := math.Max(0, math.Abs(y-pred)-pa.epsilon)
    
    if loss > 0 {
        // Calculate update step size
        var tau float64
        xNorm := mat.Norm(x, 2)
        
        switch pa.mode {
        case "PA-I":
            tau = math.Min(pa.C, loss/(xNorm*xNorm))
        case "PA-II":
            tau = loss / (xNorm*xNorm + 1/(2*pa.C))
        default:
            tau = loss / (xNorm * xNorm)
        }
        
        // Update weights
        sign := 1.0
        if pred > y {
            sign = -1.0
        }
        
        for i := 0; i < pa.nFeatures; i++ {
            pa.weights.SetVec(i, 
                pa.weights.AtVec(i) + tau*sign*x.AtVec(i))
        }
    }
}
```

### Online Gradient Boosting

```go
type OnlineGradientBoosting struct {
    trees       []*OnlineTree
    learningRate float64
    maxTrees    int
    currentTree int
}

type OnlineTree struct {
    root      *Node
    maxDepth  int
    minSamples int
}

func (ogb *OnlineGradientBoosting) PartialFit(X, y mat.Matrix) error {
    rows, _ := X.Dims()
    
    // Calculate residuals
    predictions := ogb.Predict(X)
    residuals := mat.NewVecDense(rows, nil)
    
    for i := 0; i < rows; i++ {
        residuals.SetVec(i, y.At(i, 0) - predictions.At(i, 0))
    }
    
    // Update current tree or create new one
    if ogb.currentTree >= len(ogb.trees) {
        ogb.trees = append(ogb.trees, NewOnlineTree(ogb.maxDepth))
    }
    
    tree := ogb.trees[ogb.currentTree]
    tree.PartialFit(X, residuals)
    
    // Move to next tree if current is mature
    if tree.IsMature() {
        ogb.currentTree++
        if ogb.currentTree >= ogb.maxTrees {
            // Remove oldest tree
            ogb.trees = ogb.trees[1:]
            ogb.currentTree--
        }
    }
    
    return nil
}
```

## Streaming Data Processing

### Data Stream Pipeline

```go
type DataStream struct {
    source   DataSource
    buffer   *RingBuffer
    window   *SlidingWindow
    stats    *StreamStats
}

type DataBatch struct {
    ID        int
    X         mat.Matrix
    Y         mat.Matrix
    Timestamp time.Time
    Metadata  map[string]interface{}
}

func ProcessStream(stream <-chan DataBatch, model OnlineModel) {
    preprocessor := NewStreamPreprocessor()
    evaluator := NewStreamEvaluator()
    
    for batch := range stream {
        // Preprocess
        XProcessed, err := preprocessor.Transform(batch.X)
        if err != nil {
            log.Printf("Preprocessing failed: %v", err)
            continue
        }
        
        // Predict before update (for evaluation)
        predictions, _ := model.Predict(XProcessed)
        evaluator.Update(predictions, batch.Y)
        
        // Update model
        err = model.PartialFit(XProcessed, batch.Y)
        if err != nil {
            log.Printf("Model update failed: %v", err)
            continue
        }
        
        // Log metrics
        if batch.ID % 100 == 0 {
            metrics := evaluator.GetMetrics()
            log.Printf("Batch %d - MAE: %.4f, RMSE: %.4f",
                batch.ID, metrics.MAE, metrics.RMSE)
        }
    }
}
```

### Sliding Window

```go
type SlidingWindow struct {
    data      []DataPoint
    maxSize   int
    timeWindow time.Duration
    mu        sync.RWMutex
}

type DataPoint struct {
    X         mat.Vector
    Y         float64
    Timestamp time.Time
}

func (w *SlidingWindow) Add(point DataPoint) {
    w.mu.Lock()
    defer w.mu.Unlock()
    
    w.data = append(w.data, point)
    
    // Remove old points by count
    if len(w.data) > w.maxSize {
        w.data = w.data[len(w.data)-w.maxSize:]
    }
    
    // Remove old points by time
    cutoff := time.Now().Add(-w.timeWindow)
    i := 0
    for i < len(w.data) && w.data[i].Timestamp.Before(cutoff) {
        i++
    }
    w.data = w.data[i:]
}

func (w *SlidingWindow) GetBatch() (mat.Matrix, mat.Matrix) {
    w.mu.RLock()
    defer w.mu.RUnlock()
    
    n := len(w.data)
    if n == 0 {
        return nil, nil
    }
    
    cols := w.data[0].X.Len()
    X := mat.NewDense(n, cols, nil)
    y := mat.NewVecDense(n, nil)
    
    for i, point := range w.data {
        for j := 0; j < cols; j++ {
            X.Set(i, j, point.X.AtVec(j))
        }
        y.SetVec(i, point.Y)
    }
    
    return X, y
}
```

### Ring Buffer

```go
type RingBuffer struct {
    data  []DataBatch
    size  int
    head  int
    tail  int
    count int
    mu    sync.Mutex
}

func NewRingBuffer(size int) *RingBuffer {
    return &RingBuffer{
        data: make([]DataBatch, size),
        size: size,
    }
}

func (rb *RingBuffer) Push(batch DataBatch) {
    rb.mu.Lock()
    defer rb.mu.Unlock()
    
    rb.data[rb.tail] = batch
    rb.tail = (rb.tail + 1) % rb.size
    
    if rb.count < rb.size {
        rb.count++
    } else {
        rb.head = (rb.head + 1) % rb.size
    }
}

func (rb *RingBuffer) GetRecent(n int) []DataBatch {
    rb.mu.Lock()
    defer rb.mu.Unlock()
    
    if n > rb.count {
        n = rb.count
    }
    
    result := make([]DataBatch, n)
    for i := 0; i < n; i++ {
        idx := (rb.head + rb.count - n + i) % rb.size
        result[i] = rb.data[idx]
    }
    
    return result
}
```

## Concept Drift Detection

### Drift Detectors

```go
type DriftDetector interface {
    Update(prediction, actual float64) bool
    Reset()
    GetDriftLevel() float64
}

// Page-Hinkley Test
type PageHinkley struct {
    threshold float64
    alpha     float64
    sum       float64
    min       float64
    count     int
}

func (ph *PageHinkley) Update(prediction, actual float64) bool {
    error := math.Abs(prediction - actual)
    
    ph.sum += error - ph.alpha
    ph.count++
    
    if ph.sum < ph.min {
        ph.min = ph.sum
    }
    
    // Detect drift
    PHt := ph.sum - ph.min
    if PHt > ph.threshold {
        ph.Reset()
        return true
    }
    
    return false
}

// ADWIN (ADaptive WINdowing)
type ADWIN struct {
    window    []float64
    delta     float64
    maxWindow int
}

func (a *ADWIN) Update(value float64) bool {
    a.window = append(a.window, value)
    
    if len(a.window) > a.maxWindow {
        a.window = a.window[1:]
    }
    
    // Check for change
    return a.detectChange()
}

func (a *ADWIN) detectChange() bool {
    n := len(a.window)
    
    for split := 1; split < n; split++ {
        w1 := a.window[:split]
        w2 := a.window[split:]
        
        mean1 := mean(w1)
        mean2 := mean(w2)
        
        // Hoeffding bound
        epsilon := math.Sqrt(
            (1.0/(2.0*float64(len(w1))) + 
             1.0/(2.0*float64(len(w2)))) * 
            math.Log(4.0/a.delta))
        
        if math.Abs(mean1-mean2) > epsilon {
            // Drift detected, shrink window
            a.window = w2
            return true
        }
    }
    
    return false
}
```

### Adaptive Learning

```go
type AdaptiveModel struct {
    baseModel    OnlineModel
    driftDetector DriftDetector
    reservoir    *ReservoirSampling
    retrainThreshold float64
}

func (am *AdaptiveModel) PartialFit(X, y mat.Matrix) error {
    rows, _ := X.Dims()
    
    for i := 0; i < rows; i++ {
        x := X.(*mat.Dense).RowView(i)
        target := y.At(i, 0)
        
        // Make prediction
        pred, _ := am.baseModel.Predict(x.T())
        prediction := pred.At(0, 0)
        
        // Check for drift
        if am.driftDetector.Update(prediction, target) {
            log.Println("Drift detected! Adapting model...")
            am.adaptToDrift()
        }
        
        // Update model
        am.baseModel.PartialFit(x.T(), mat.NewVecDense(1, []float64{target}))
        
        // Update reservoir
        am.reservoir.Add(x, target)
    }
    
    return nil
}

func (am *AdaptiveModel) adaptToDrift() {
    // Retrain on recent samples
    samples := am.reservoir.GetSamples()
    
    if len(samples) > 0 {
        X, y := samplesToMatrix(samples)
        am.baseModel.Reset()
        am.baseModel.Fit(X, y)
    }
}
```

## Stream Processing Patterns

### Micro-Batching

```go
type MicroBatchProcessor struct {
    batchSize    int
    maxWait      time.Duration
    accumulator  *BatchAccumulator
    processor    func(mat.Matrix, mat.Matrix) error
}

func (mb *MicroBatchProcessor) Process(stream <-chan DataPoint) {
    ticker := time.NewTicker(mb.maxWait)
    defer ticker.Stop()
    
    for {
        select {
        case point, ok := <-stream:
            if !ok {
                // Process remaining
                mb.processBatch()
                return
            }
            
            mb.accumulator.Add(point)
            
            if mb.accumulator.Size() >= mb.batchSize {
                mb.processBatch()
            }
            
        case <-ticker.C:
            // Timeout, process what we have
            if mb.accumulator.Size() > 0 {
                mb.processBatch()
            }
        }
    }
}

func (mb *MicroBatchProcessor) processBatch() {
    X, y := mb.accumulator.GetBatch()
    if X != nil {
        mb.processor(X, y)
        mb.accumulator.Clear()
    }
}
```

### Hoeffding Trees

```go
type HoeffdingTree struct {
    root         *HTNode
    splitMetric  string
    maxDepth     int
    minSamples   int
    confidence   float64
    tieThreshold float64
}

type HTNode struct {
    isLeaf       bool
    splitFeature int
    splitValue   float64
    stats        *StreamingStats
    left         *HTNode
    right        *HTNode
    samples      int
}

func (ht *HoeffdingTree) PartialFit(x mat.Vector, y float64) {
    leaf := ht.findLeaf(x)
    leaf.stats.Update(x, y)
    leaf.samples++
    
    // Check if should split
    if leaf.samples > ht.minSamples && leaf.samples % 100 == 0 {
        ht.attemptSplit(leaf)
    }
}

func (ht *HoeffdingTree) attemptSplit(node *HTNode) {
    // Calculate information gain for each feature
    gains := ht.calculateGains(node)
    
    // Find best and second best
    best, secondBest := findTopTwo(gains)
    
    // Hoeffding bound
    R := math.Log2(float64(node.stats.NumClasses()))
    epsilon := math.Sqrt(R*R*math.Log(1/ht.confidence) / (2*float64(node.samples)))
    
    // Check if we can split
    if gains[best] - gains[secondBest] > epsilon || epsilon < ht.tieThreshold {
        ht.splitNode(node, best)
    }
}
```

## Performance Monitoring

### Streaming Metrics

```go
type StreamingMetrics struct {
    window       *SlidingWindow
    cumulative   *CumulativeStats
    prequential  *PrequentialEvaluator
}

type PrequentialEvaluator struct {
    fadingFactor float64
    sumError     float64
    sumSquared   float64
    count        float64
}

func (pe *PrequentialEvaluator) Update(prediction, actual float64) {
    error := actual - prediction
    
    pe.sumError = pe.fadingFactor*pe.sumError + error
    pe.sumSquared = pe.fadingFactor*pe.sumSquared + error*error
    pe.count = pe.fadingFactor*pe.count + 1
    
    if pe.count > 1000 && pe.fadingFactor < 1.0 {
        // Normalize to prevent numerical issues
        pe.sumError /= pe.fadingFactor
        pe.sumSquared /= pe.fadingFactor
        pe.count /= pe.fadingFactor
    }
}

func (pe *PrequentialEvaluator) GetMetrics() map[string]float64 {
    mae := math.Abs(pe.sumError) / pe.count
    mse := pe.sumSquared / pe.count
    rmse := math.Sqrt(mse)
    
    return map[string]float64{
        "MAE":  mae,
        "MSE":  mse,
        "RMSE": rmse,
    }
}
```

### Real-time Dashboard

```go
type StreamingDashboard struct {
    metrics   *StreamingMetrics
    server    *http.Server
    wsClients map[*websocket.Conn]bool
    mu        sync.RWMutex
}

func (d *StreamingDashboard) Start(port int) {
    http.HandleFunc("/metrics", d.handleMetrics)
    http.HandleFunc("/ws", d.handleWebSocket)
    
    d.server = &http.Server{Addr: fmt.Sprintf(":%d", port)}
    
    go d.broadcastMetrics()
    log.Fatal(d.server.ListenAndServe())
}

func (d *StreamingDashboard) broadcastMetrics() {
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        metrics := d.metrics.GetCurrent()
        data, _ := json.Marshal(metrics)
        
        d.mu.RLock()
        for client := range d.wsClients {
            client.WriteMessage(websocket.TextMessage, data)
        }
        d.mu.RUnlock()
    }
}
```

## Distributed Streaming

### Kafka Integration

```go
import (
    "github.com/Shopify/sarama"
)

type KafkaStreamProcessor struct {
    consumer sarama.ConsumerGroup
    producer sarama.AsyncProducer
    model    OnlineModel
}

func (k *KafkaStreamProcessor) ConsumeClaim(
    session sarama.ConsumerGroupSession,
    claim sarama.ConsumerGroupClaim,
) error {
    for message := range claim.Messages() {
        // Parse message
        batch, err := parseMessage(message.Value)
        if err != nil {
            log.Printf("Parse error: %v", err)
            continue
        }
        
        // Process
        predictions, _ := k.model.Predict(batch.X)
        k.model.PartialFit(batch.X, batch.Y)
        
        // Send predictions
        k.producer.Input() <- &sarama.ProducerMessage{
            Topic: "predictions",
            Key:   sarama.StringEncoder(batch.ID),
            Value: sarama.ByteEncoder(serializePredictions(predictions)),
        }
        
        session.MarkMessage(message, "")
    }
    
    return nil
}
```

### Parallel Stream Processing

```go
type ParallelStreamProcessor struct {
    workers   int
    models    []OnlineModel
    router    func(DataBatch) int
}

func (p *ParallelStreamProcessor) Process(stream <-chan DataBatch) {
    // Create worker channels
    channels := make([]chan DataBatch, p.workers)
    for i := range channels {
        channels[i] = make(chan DataBatch, 100)
    }
    
    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < p.workers; i++ {
        wg.Add(1)
        go func(id int, ch <-chan DataBatch) {
            defer wg.Done()
            p.worker(id, ch)
        }(i, channels[i])
    }
    
    // Route batches
    for batch := range stream {
        workerID := p.router(batch)
        channels[workerID] <- batch
    }
    
    // Close channels
    for _, ch := range channels {
        close(ch)
    }
    
    wg.Wait()
}

func (p *ParallelStreamProcessor) worker(id int, batches <-chan DataBatch) {
    model := p.models[id]
    
    for batch := range batches {
        model.PartialFit(batch.X, batch.Y)
    }
}
```

## Memory Management

### Reservoir Sampling

```go
type ReservoirSampling struct {
    capacity int
    samples  []Sample
    count    int64
    rand     *rand.Rand
}

type Sample struct {
    X mat.Vector
    Y float64
}

func (r *ReservoirSampling) Add(x mat.Vector, y float64) {
    r.count++
    
    if len(r.samples) < r.capacity {
        r.samples = append(r.samples, Sample{X: x, Y: y})
    } else {
        // Random replacement
        j := r.rand.Int63n(r.count)
        if j < int64(r.capacity) {
            r.samples[j] = Sample{X: x, Y: y}
        }
    }
}

func (r *ReservoirSampling) GetSamples() []Sample {
    return r.samples
}
```

### Feature Hashing

```go
type FeatureHasher struct {
    nFeatures int
    hashFunc  func(string) uint32
}

func (fh *FeatureHasher) Transform(features map[string]float64) mat.Vector {
    vec := mat.NewVecDense(fh.nFeatures, nil)
    
    for feature, value := range features {
        hash := fh.hashFunc(feature)
        idx := int(hash % uint32(fh.nFeatures))
        
        // Handle collisions with addition
        vec.SetVec(idx, vec.AtVec(idx)+value)
    }
    
    return vec
}
```

## Best Practices

1. **Monitor Drift**: Always track concept drift
2. **Buffer Wisely**: Use appropriate buffer sizes
3. **Handle Failures**: Graceful degradation for stream interruptions
4. **Update Frequency**: Balance accuracy vs computational cost
5. **Memory Bounds**: Implement forgetting mechanisms
6. **Evaluate Continuously**: Use prequential evaluation
7. **Version Models**: Track model versions in production
8. **Scale Horizontally**: Distribute processing when needed

## Common Pitfalls

1. **Memory Leaks**: Not implementing forgetting
2. **Drift Ignorance**: Not adapting to changing distributions
3. **Evaluation Bias**: Using future data for evaluation
4. **Buffer Overflow**: Unbounded buffers
5. **Synchronization Issues**: Race conditions in parallel processing

## Next Steps

- Explore [Online Learning Examples](../../examples/streaming/)
- Learn about [Production Deployment](../tutorials/production.md)
- See [Performance Guide](../core-concepts/performance.md)
- Read [API Reference](../api/sklearn.md#streaming)