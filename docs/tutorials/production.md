# Production Deployment Guide

Learn how to deploy SciGo models to production environments with high availability, monitoring, and performance optimization.

## Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Model   │    │  Model   │    │  Model   │
    │ Service  │    │ Service  │    │ Service  │
    │ Instance │    │ Instance │    │ Instance │
    └──────────┘    └──────────┘    └──────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    ┌──────────┐
                    │  Model   │
                    │  Storage │
                    └──────────┘
```

## Building a Production Service

### 1. Model Service Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"
    
    "github.com/YuminosukeSato/scigo/linear"
    "github.com/YuminosukeSato/scigo/core/model"
    "github.com/YuminosukeSato/scigo/preprocessing"
    "gonum.org/v1/gonum/mat"
)

// ModelService handles ML predictions
type ModelService struct {
    model      *linear.LinearRegression
    scaler     *preprocessing.StandardScaler
    mu         sync.RWMutex
    loadTime   time.Time
    predictions int64
}

// NewModelService creates a new service
func NewModelService(modelPath string) (*ModelService, error) {
    service := &ModelService{}
    
    // Load model
    if err := service.LoadModel(modelPath); err != nil {
        return nil, fmt.Errorf("failed to load model: %w", err)
    }
    
    return service, nil
}

// LoadModel loads or reloads the model
func (s *ModelService) LoadModel(path string) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    // Load model
    lr := linear.NewLinearRegression()
    if err := model.LoadModel(lr, path); err != nil {
        return err
    }
    
    // Load scaler if exists
    scaler := preprocessing.NewStandardScaler()
    scalerPath := path + ".scaler"
    if _, err := os.Stat(scalerPath); err == nil {
        if err := model.LoadModel(scaler, scalerPath); err != nil {
            return err
        }
    }
    
    s.model = lr
    s.scaler = scaler
    s.loadTime = time.Now()
    
    log.Printf("Model loaded successfully at %v", s.loadTime)
    return nil
}

// PredictRequest represents prediction input
type PredictRequest struct {
    Features [][]float64 `json:"features"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// PredictResponse represents prediction output
type PredictResponse struct {
    Predictions []float64 `json:"predictions"`
    ModelVersion string `json:"model_version"`
    Timestamp time.Time `json:"timestamp"`
}

// Predict handles prediction requests
func (s *ModelService) Predict(req PredictRequest) (*PredictResponse, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    // Validate input
    if len(req.Features) == 0 {
        return nil, fmt.Errorf("no features provided")
    }
    
    // Convert to matrix
    rows := len(req.Features)
    cols := len(req.Features[0])
    data := make([]float64, rows*cols)
    
    for i, row := range req.Features {
        for j, val := range row {
            data[i*cols+j] = val
        }
    }
    
    X := mat.NewDense(rows, cols, data)
    
    // Apply scaling if available
    if s.scaler != nil {
        var err error
        X, err = s.scaler.Transform(X)
        if err != nil {
            return nil, fmt.Errorf("scaling failed: %w", err)
        }
    }
    
    // Make predictions
    predictions, err := s.model.Predict(X)
    if err != nil {
        return nil, fmt.Errorf("prediction failed: %w", err)
    }
    
    // Extract results
    predMatrix := predictions.(*mat.Dense)
    results := make([]float64, rows)
    for i := 0; i < rows; i++ {
        results[i] = predMatrix.At(i, 0)
    }
    
    s.predictions++
    
    return &PredictResponse{
        Predictions: results,
        ModelVersion: s.loadTime.Format(time.RFC3339),
        Timestamp: time.Now(),
    }, nil
}

// HTTP Handlers

func (s *ModelService) handlePredict(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var req PredictRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    resp, err := s.Predict(req)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func (s *ModelService) handleHealth(w http.ResponseWriter, r *http.Request) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    health := map[string]interface{}{
        "status": "healthy",
        "model_loaded": s.model != nil,
        "model_version": s.loadTime.Format(time.RFC3339),
        "predictions_served": s.predictions,
        "uptime": time.Since(s.loadTime).String(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(health)
}

func (s *ModelService) handleMetrics(w http.ResponseWriter, r *http.Request) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    // Prometheus format metrics
    fmt.Fprintf(w, "# HELP model_predictions_total Total predictions served\n")
    fmt.Fprintf(w, "# TYPE model_predictions_total counter\n")
    fmt.Fprintf(w, "model_predictions_total %d\n", s.predictions)
    
    fmt.Fprintf(w, "# HELP model_loaded_timestamp Model load timestamp\n")
    fmt.Fprintf(w, "# TYPE model_loaded_timestamp gauge\n")
    fmt.Fprintf(w, "model_loaded_timestamp %d\n", s.loadTime.Unix())
}

func main() {
    // Configuration
    modelPath := os.Getenv("MODEL_PATH")
    if modelPath == "" {
        modelPath = "model.gob"
    }
    
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
    
    // Create service
    service, err := NewModelService(modelPath)
    if err != nil {
        log.Fatal("Failed to create service:", err)
    }
    
    // Setup routes
    http.HandleFunc("/predict", service.handlePredict)
    http.HandleFunc("/health", service.handleHealth)
    http.HandleFunc("/metrics", service.handleMetrics)
    
    // Create server
    srv := &http.Server{
        Addr:         ":" + port,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    // Graceful shutdown
    go func() {
        sigChan := make(chan os.Signal, 1)
        signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
        <-sigChan
        
        log.Println("Shutting down server...")
        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancel()
        
        srv.Shutdown(ctx)
    }()
    
    // Start server
    log.Printf("Model service starting on port %s", port)
    if err := srv.ListenAndServe(); err != http.ErrServerClosed {
        log.Fatal("Server error:", err)
    }
}
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o model-service ./cmd/service

# Production image
FROM alpine:latest
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy binary and model
COPY --from=builder /app/model-service .
COPY --from=builder /app/model.gob .
COPY --from=builder /app/model.gob.scaler .

EXPOSE 8080

CMD ["./model-service"]
```

### 3. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scigo-model-service
  labels:
    app: scigo-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scigo-model
  template:
    metadata:
      labels:
        app: scigo-model
    spec:
      containers:
      - name: model-service
        image: your-registry/scigo-model:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/models/model.gob"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        configMap:
          name: model-files
---
apiVersion: v1
kind: Service
metadata:
  name: scigo-model-service
spec:
  selector:
    app: scigo-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

## Performance Optimization

### 1. Model Loading Strategies

```go
// Lazy loading with caching
type ModelCache struct {
    models map[string]*CachedModel
    mu     sync.RWMutex
    maxAge time.Duration
}

type CachedModel struct {
    model    Model
    loadTime time.Time
}

func (c *ModelCache) GetModel(name string) (Model, error) {
    c.mu.RLock()
    cached, exists := c.models[name]
    c.mu.RUnlock()
    
    if exists && time.Since(cached.loadTime) < c.maxAge {
        return cached.model, nil
    }
    
    // Load and cache model
    c.mu.Lock()
    defer c.mu.Unlock()
    
    model, err := LoadModel(name)
    if err != nil {
        return nil, err
    }
    
    c.models[name] = &CachedModel{
        model:    model,
        loadTime: time.Now(),
    }
    
    return model, nil
}
```

### 2. Request Batching

```go
// Batch predictor for improved throughput
type BatchPredictor struct {
    model       Model
    batchSize   int
    maxWaitTime time.Duration
    requests    chan *PredictJob
}

type PredictJob struct {
    features []float64
    result   chan PredictResult
}

type PredictResult struct {
    prediction float64
    err       error
}

func (b *BatchPredictor) Start(ctx context.Context) {
    batch := make([]*PredictJob, 0, b.batchSize)
    timer := time.NewTimer(b.maxWaitTime)
    
    for {
        select {
        case job := <-b.requests:
            batch = append(batch, job)
            
            if len(batch) >= b.batchSize {
                b.processBatch(batch)
                batch = batch[:0]
                timer.Reset(b.maxWaitTime)
            }
            
        case <-timer.C:
            if len(batch) > 0 {
                b.processBatch(batch)
                batch = batch[:0]
            }
            timer.Reset(b.maxWaitTime)
            
        case <-ctx.Done():
            return
        }
    }
}

func (b *BatchPredictor) processBatch(batch []*PredictJob) {
    // Combine features
    rows := len(batch)
    cols := len(batch[0].features)
    data := make([]float64, rows*cols)
    
    for i, job := range batch {
        copy(data[i*cols:], job.features)
    }
    
    X := mat.NewDense(rows, cols, data)
    
    // Batch prediction
    predictions, err := b.model.Predict(X)
    
    // Send results
    for i, job := range batch {
        if err != nil {
            job.result <- PredictResult{err: err}
        } else {
            job.result <- PredictResult{
                prediction: predictions.At(i, 0),
            }
        }
    }
}
```

### 3. Connection Pooling

```go
// Database connection pool for feature retrieval
type FeatureStore struct {
    db   *sql.DB
    pool *sync.Pool
}

func NewFeatureStore(dsn string) (*FeatureStore, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(10)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    return &FeatureStore{
        db: db,
        pool: &sync.Pool{
            New: func() interface{} {
                return make([]float64, 0, 100)
            },
        },
    }, nil
}

func (fs *FeatureStore) GetFeatures(userID string) ([]float64, error) {
    // Get buffer from pool
    features := fs.pool.Get().([]float64)
    defer func() {
        features = features[:0]
        fs.pool.Put(features)
    }()
    
    // Query features
    rows, err := fs.db.Query(
        "SELECT feature_value FROM features WHERE user_id = $1",
        userID,
    )
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    for rows.Next() {
        var value float64
        if err := rows.Scan(&value); err != nil {
            return nil, err
        }
        features = append(features, value)
    }
    
    // Return copy
    result := make([]float64, len(features))
    copy(result, features)
    return result, nil
}
```

## Monitoring and Observability

### 1. Metrics Collection

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    predictionDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "model_prediction_duration_seconds",
            Help: "Duration of model predictions",
        },
        []string{"model", "status"},
    )
    
    predictionTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "model_predictions_total",
            Help: "Total number of predictions",
        },
        []string{"model", "status"},
    )
    
    modelLoadTime = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "model_load_timestamp",
            Help: "Timestamp when model was loaded",
        },
        []string{"model"},
    )
)

func init() {
    prometheus.MustRegister(predictionDuration)
    prometheus.MustRegister(predictionTotal)
    prometheus.MustRegister(modelLoadTime)
}

func instrumentedPredict(model Model, X mat.Matrix) (mat.Matrix, error) {
    timer := prometheus.NewTimer(predictionDuration.WithLabelValues(
        "linear_regression", "pending",
    ))
    
    predictions, err := model.Predict(X)
    
    status := "success"
    if err != nil {
        status = "error"
    }
    
    timer.ObserveDuration()
    predictionTotal.WithLabelValues("linear_regression", status).Inc()
    
    return predictions, err
}
```

### 2. Structured Logging

```go
import (
    "github.com/sirupsen/logrus"
)

type ModelLogger struct {
    logger *logrus.Logger
}

func NewModelLogger() *ModelLogger {
    logger := logrus.New()
    logger.SetFormatter(&logrus.JSONFormatter{})
    logger.SetLevel(logrus.InfoLevel)
    
    return &ModelLogger{logger: logger}
}

func (l *ModelLogger) LogPrediction(req PredictRequest, resp *PredictResponse, duration time.Duration) {
    l.logger.WithFields(logrus.Fields{
        "event": "prediction",
        "features_count": len(req.Features),
        "predictions_count": len(resp.Predictions),
        "duration_ms": duration.Milliseconds(),
        "model_version": resp.ModelVersion,
        "timestamp": resp.Timestamp,
    }).Info("Prediction completed")
}

func (l *ModelLogger) LogError(operation string, err error) {
    l.logger.WithFields(logrus.Fields{
        "event": "error",
        "operation": operation,
        "error": err.Error(),
    }).Error("Operation failed")
}
```

### 3. Distributed Tracing

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

var tracer = otel.Tracer("model-service")

func tracedPredict(ctx context.Context, model Model, X mat.Matrix) (mat.Matrix, error) {
    ctx, span := tracer.Start(ctx, "model.predict")
    defer span.End()
    
    span.SetAttributes(
        attribute.Int("input.rows", X.Rows()),
        attribute.Int("input.cols", X.Cols()),
    )
    
    predictions, err := model.Predict(X)
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
    } else {
        span.SetAttributes(
            attribute.Int("output.rows", predictions.Rows()),
        )
    }
    
    return predictions, err
}
```

## High Availability

### 1. Load Balancing

```nginx
# nginx.conf
upstream model_service {
    least_conn;
    server model1.internal:8080 weight=5;
    server model2.internal:8080 weight=5;
    server model3.internal:8080 weight=5;
    
    keepalive 32;
}

server {
    listen 80;
    
    location /predict {
        proxy_pass http://model_service;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
        
        # Retries
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        proxy_pass http://model_service;
        proxy_http_version 1.1;
    }
}
```

### 2. Circuit Breaker

```go
import "github.com/sony/gobreaker"

type CircuitBreakerService struct {
    cb    *gobreaker.CircuitBreaker
    model Model
}

func NewCircuitBreakerService(model Model) *CircuitBreakerService {
    settings := gobreaker.Settings{
        Name:        "ModelPredict",
        MaxRequests: 3,
        Interval:    time.Minute,
        Timeout:     30 * time.Second,
        ReadyToTrip: func(counts gobreaker.Counts) bool {
            failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
            return counts.Requests >= 3 && failureRatio >= 0.6
        },
    }
    
    return &CircuitBreakerService{
        cb:    gobreaker.NewCircuitBreaker(settings),
        model: model,
    }
}

func (s *CircuitBreakerService) Predict(X mat.Matrix) (mat.Matrix, error) {
    result, err := s.cb.Execute(func() (interface{}, error) {
        return s.model.Predict(X)
    })
    
    if err != nil {
        return nil, err
    }
    
    return result.(mat.Matrix), nil
}
```

### 3. Graceful Degradation

```go
// Fallback to simpler model
type FallbackPredictor struct {
    primary   Model
    fallback  Model
    threshold time.Duration
}

func (f *FallbackPredictor) Predict(X mat.Matrix) (mat.Matrix, error) {
    done := make(chan struct{})
    var predictions mat.Matrix
    var err error
    
    go func() {
        predictions, err = f.primary.Predict(X)
        close(done)
    }()
    
    select {
    case <-done:
        return predictions, err
    case <-time.After(f.threshold):
        // Primary model too slow, use fallback
        return f.fallback.Predict(X)
    }
}
```

## Security

### 1. Input Validation

```go
func validateInput(features [][]float64) error {
    if len(features) == 0 {
        return fmt.Errorf("empty feature set")
    }
    
    if len(features) > 1000 {
        return fmt.Errorf("batch size exceeds limit")
    }
    
    expectedCols := -1
    for i, row := range features {
        if len(row) == 0 {
            return fmt.Errorf("empty row at index %d", i)
        }
        
        if expectedCols == -1 {
            expectedCols = len(row)
        } else if len(row) != expectedCols {
            return fmt.Errorf("inconsistent feature dimensions")
        }
        
        for j, val := range row {
            if math.IsNaN(val) || math.IsInf(val, 0) {
                return fmt.Errorf("invalid value at [%d][%d]", i, j)
            }
        }
    }
    
    return nil
}
```

### 2. Rate Limiting

```go
import "golang.org/x/time/rate"

type RateLimitedService struct {
    service Model
    limiter *rate.Limiter
}

func NewRateLimitedService(service Model, rps int) *RateLimitedService {
    return &RateLimitedService{
        service: service,
        limiter: rate.NewLimiter(rate.Limit(rps), rps),
    }
}

func (s *RateLimitedService) Predict(ctx context.Context, X mat.Matrix) (mat.Matrix, error) {
    if err := s.limiter.Wait(ctx); err != nil {
        return nil, fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    return s.service.Predict(X)
}
```

## Testing in Production

### 1. A/B Testing

```go
type ABTestService struct {
    modelA      Model
    modelB      Model
    trafficPct  float64 // Percentage for model B
    metrics     *ABMetrics
}

func (s *ABTestService) Predict(X mat.Matrix) (mat.Matrix, error) {
    useModelB := rand.Float64() < s.trafficPct
    
    var predictions mat.Matrix
    var err error
    var modelName string
    
    start := time.Now()
    
    if useModelB {
        predictions, err = s.modelB.Predict(X)
        modelName = "B"
    } else {
        predictions, err = s.modelA.Predict(X)
        modelName = "A"
    }
    
    duration := time.Since(start)
    s.metrics.Record(modelName, duration, err)
    
    return predictions, err
}
```

### 2. Shadow Mode

```go
// Run new model in shadow mode
type ShadowModeService struct {
    production Model
    shadow     Model
    logger     *log.Logger
}

func (s *ShadowModeService) Predict(X mat.Matrix) (mat.Matrix, error) {
    // Production prediction
    prodPred, prodErr := s.production.Predict(X)
    
    // Shadow prediction (async)
    go func() {
        shadowPred, shadowErr := s.shadow.Predict(X)
        
        // Log comparison
        s.logger.Printf("Shadow comparison: prod_err=%v, shadow_err=%v",
            prodErr, shadowErr)
        
        if prodErr == nil && shadowErr == nil {
            diff := comparePredictions(prodPred, shadowPred)
            s.logger.Printf("Prediction difference: %.4f", diff)
        }
    }()
    
    return prodPred, prodErr
}
```

## Best Practices Checklist

- [ ] **Model versioning** - Track model versions and parameters
- [ ] **Health checks** - Implement liveness and readiness probes
- [ ] **Metrics** - Export Prometheus metrics
- [ ] **Logging** - Structured JSON logging
- [ ] **Tracing** - Distributed tracing with OpenTelemetry
- [ ] **Rate limiting** - Protect against overload
- [ ] **Circuit breaking** - Fail fast on errors
- [ ] **Graceful shutdown** - Handle signals properly
- [ ] **Input validation** - Validate all inputs
- [ ] **Error handling** - Comprehensive error handling
- [ ] **Timeouts** - Set appropriate timeouts
- [ ] **Resource limits** - Configure memory and CPU limits
- [ ] **Horizontal scaling** - Design for multiple instances
- [ ] **Load balancing** - Distribute traffic evenly
- [ ] **Caching** - Cache models and predictions
- [ ] **Monitoring** - Set up alerts and dashboards
- [ ] **Documentation** - API documentation
- [ ] **Testing** - Unit, integration, and load tests
- [ ] **Security** - Authentication and authorization
- [ ] **Compliance** - Data privacy and regulations

## Next Steps

- Explore [Performance Guide](../core-concepts/performance.md)
- Learn about [Monitoring](../advanced/monitoring.md)
- See [Kubernetes Examples](../../examples/kubernetes/)
- Read [Security Best Practices](../advanced/security.md)