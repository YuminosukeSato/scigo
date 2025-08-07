# Model Persistence Guide

Comprehensive guide to saving, loading, and deploying machine learning models in SciGo.

## Overview

Model persistence is crucial for deploying machine learning systems. SciGo provides multiple serialization formats and seamless interoperability with scikit-learn.

## Persistence Formats

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| Binary (GOB) | Go-to-Go | Fast, compact | Go-only |
| JSON | Interoperability | Human-readable, cross-language | Larger size |
| Protocol Buffers | Production | Efficient, versioned | Requires schema |
| ONNX | Cross-platform | Industry standard | Limited model support |

## Binary Serialization (GOB)

### Basic Save/Load

```go
package main

import (
    "encoding/gob"
    "os"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "github.com/YuminosukeSato/scigo/core/model"
)

func saveModelBinary() {
    // Train model
    lr := linear.NewLinearRegression()
    lr.Fit(XTrain, yTrain)
    
    // Save to file
    file, err := os.Create("model.gob")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()
    
    encoder := gob.NewEncoder(file)
    if err := encoder.Encode(lr); err != nil {
        log.Fatal("Encoding failed:", err)
    }
    
    log.Println("Model saved successfully")
}

func loadModelBinary() *linear.LinearRegression {
    file, err := os.Open("model.gob")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()
    
    lr := linear.NewLinearRegression()
    decoder := gob.NewDecoder(file)
    
    if err := decoder.Decode(lr); err != nil {
        log.Fatal("Decoding failed:", err)
    }
    
    return lr
}
```

### Custom Serialization

```go
// Implement GobEncoder/GobDecoder for custom types
type CustomModel struct {
    weights []float64
    bias    float64
    metadata map[string]interface{}
}

func (m *CustomModel) GobEncode() ([]byte, error) {
    // Custom encoding logic
    data := struct {
        Weights  []float64
        Bias     float64
        Metadata map[string]interface{}
    }{
        Weights:  m.weights,
        Bias:     m.bias,
        Metadata: m.metadata,
    }
    
    var buf bytes.Buffer
    encoder := gob.NewEncoder(&buf)
    if err := encoder.Encode(data); err != nil {
        return nil, err
    }
    
    return buf.Bytes(), nil
}

func (m *CustomModel) GobDecode(data []byte) error {
    var decoded struct {
        Weights  []float64
        Bias     float64
        Metadata map[string]interface{}
    }
    
    buf := bytes.NewBuffer(data)
    decoder := gob.NewDecoder(buf)
    
    if err := decoder.Decode(&decoded); err != nil {
        return err
    }
    
    m.weights = decoded.Weights
    m.bias = decoded.Bias
    m.metadata = decoded.Metadata
    
    return nil
}
```

## JSON Serialization

### Scikit-learn Compatible Format

```go
package persistence

import (
    "encoding/json"
    "io/ioutil"
    "time"
)

// SKLearnModel represents scikit-learn compatible format
type SKLearnModel struct {
    ModelSpec  ModelSpec              `json:"model_spec"`
    Parameters map[string]interface{} `json:"parameters"`
    Metadata   Metadata               `json:"metadata"`
}

type ModelSpec struct {
    Name          string `json:"name"`
    FormatVersion string `json:"format_version"`
    Library       string `json:"library"`
    CreatedAt     string `json:"created_at"`
}

type Metadata struct {
    TrainingSamples  int     `json:"training_samples"`
    TrainingFeatures int     `json:"training_features"`
    TrainingTime     float64 `json:"training_time_seconds"`
    Metrics          map[string]float64 `json:"metrics,omitempty"`
}

func ExportToSKLearn(model Model, filename string) error {
    params := model.GetParams()
    
    skModel := SKLearnModel{
        ModelSpec: ModelSpec{
            Name:          "LinearRegression",
            FormatVersion: "1.0",
            Library:       "scigo",
            CreatedAt:     time.Now().Format(time.RFC3339),
        },
        Parameters: params,
        Metadata: Metadata{
            TrainingSamples:  params["n_samples"].(int),
            TrainingFeatures: params["n_features"].(int),
        },
    }
    
    data, err := json.MarshalIndent(skModel, "", "  ")
    if err != nil {
        return err
    }
    
    return ioutil.WriteFile(filename, data, 0644)
}

func ImportFromSKLearn(filename string) (Model, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    
    var skModel SKLearnModel
    if err := json.Unmarshal(data, &skModel); err != nil {
        return nil, err
    }
    
    // Create appropriate model based on spec
    model := createModel(skModel.ModelSpec.Name)
    
    // Set parameters
    if err := model.SetParams(skModel.Parameters); err != nil {
        return nil, err
    }
    
    return model, nil
}
```

### Python Integration

```python
# save_sklearn_model.py
import json
import numpy as np
from sklearn.linear_model import LinearRegression

def export_model(model, filename):
    """Export scikit-learn model to JSON format"""
    
    model_data = {
        "model_spec": {
            "name": model.__class__.__name__,
            "format_version": "1.0",
            "library": "scikit-learn",
            "sklearn_version": sklearn.__version__
        },
        "parameters": {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "n_features": model.n_features_in_
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)

# Train and export
model = LinearRegression()
model.fit(X_train, y_train)
export_model(model, "sklearn_model.json")
```

```go
// Load in Go
func loadSKLearnModel() {
    model, err := ImportFromSKLearn("sklearn_model.json")
    if err != nil {
        log.Fatal(err)
    }
    
    // Use model for predictions
    predictions, _ := model.Predict(XTest)
}
```

## Protocol Buffers

### Define Schema

```protobuf
// model.proto
syntax = "proto3";

package scigo;

message ModelProto {
    string name = 1;
    string version = 2;
    
    oneof model_type {
        LinearRegressionProto linear_regression = 3;
        SGDClassifierProto sgd_classifier = 4;
        KMeansProto kmeans = 5;
    }
    
    Metadata metadata = 10;
}

message LinearRegressionProto {
    repeated double coefficients = 1;
    double intercept = 2;
    int32 n_features = 3;
}

message Metadata {
    int64 created_timestamp = 1;
    int32 training_samples = 2;
    map<string, double> metrics = 3;
}
```

### Implementation

```go
package persistence

import (
    "github.com/golang/protobuf/proto"
    pb "github.com/YuminosukeSato/scigo/proto"
)

func SaveModelProto(model Model, filename string) error {
    // Convert to protobuf
    modelProto := &pb.ModelProto{
        Name:    "LinearRegression",
        Version: "1.0",
    }
    
    switch m := model.(type) {
    case *LinearRegression:
        modelProto.ModelType = &pb.ModelProto_LinearRegression{
            LinearRegression: &pb.LinearRegressionProto{
                Coefficients: m.GetWeights(),
                Intercept:    m.GetIntercept(),
                NFeatures:    int32(m.NFeatures()),
            },
        }
    }
    
    // Serialize
    data, err := proto.Marshal(modelProto)
    if err != nil {
        return err
    }
    
    return ioutil.WriteFile(filename, data, 0644)
}

func LoadModelProto(filename string) (Model, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    
    var modelProto pb.ModelProto
    if err := proto.Unmarshal(data, &modelProto); err != nil {
        return nil, err
    }
    
    // Create model from proto
    return createModelFromProto(&modelProto)
}
```

## Model Versioning

### Version Management

```go
type ModelVersion struct {
    Major      int
    Minor      int
    Patch      int
    Timestamp  time.Time
    Checksum   string
    Metrics    map[string]float64
    Parameters map[string]interface{}
}

type ModelRegistry struct {
    basePath string
    versions map[string][]ModelVersion
    mu       sync.RWMutex
}

func NewModelRegistry(basePath string) *ModelRegistry {
    return &ModelRegistry{
        basePath: basePath,
        versions: make(map[string][]ModelVersion),
    }
}

func (r *ModelRegistry) SaveModel(name string, model Model, metrics map[string]float64) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    // Get next version
    versions := r.versions[name]
    var nextVersion ModelVersion
    
    if len(versions) == 0 {
        nextVersion = ModelVersion{Major: 1, Minor: 0, Patch: 0}
    } else {
        lastVersion := versions[len(versions)-1]
        nextVersion = r.incrementVersion(lastVersion, metrics)
    }
    
    // Save model
    filename := fmt.Sprintf("%s/%s_v%d.%d.%d.gob",
        r.basePath, name,
        nextVersion.Major, nextVersion.Minor, nextVersion.Patch)
    
    if err := SaveModel(model, filename); err != nil {
        return err
    }
    
    // Calculate checksum
    checksum, _ := calculateChecksum(filename)
    
    // Update registry
    nextVersion.Timestamp = time.Now()
    nextVersion.Checksum = checksum
    nextVersion.Metrics = metrics
    nextVersion.Parameters = model.GetParams()
    
    r.versions[name] = append(versions, nextVersion)
    
    return r.saveRegistry()
}

func (r *ModelRegistry) incrementVersion(last ModelVersion, metrics map[string]float64) ModelVersion {
    // Semantic versioning logic
    if hasBreakingChanges(last, metrics) {
        return ModelVersion{Major: last.Major + 1, Minor: 0, Patch: 0}
    }
    
    if hasNewFeatures(last, metrics) {
        return ModelVersion{Major: last.Major, Minor: last.Minor + 1, Patch: 0}
    }
    
    return ModelVersion{Major: last.Major, Minor: last.Minor, Patch: last.Patch + 1}
}
```

### A/B Testing Support

```go
type ModelDeployment struct {
    registry *ModelRegistry
    active   map[string]DeploymentConfig
}

type DeploymentConfig struct {
    Primary   ModelVersion
    Canary    *ModelVersion
    Traffic   float64 // Percentage to canary
}

func (d *ModelDeployment) GetModel(name string, requestID string) (Model, string) {
    config := d.active[name]
    
    // Determine which version to use
    useCanary := false
    version := config.Primary
    
    if config.Canary != nil {
        // Hash-based traffic splitting
        hash := hashString(requestID)
        if float64(hash%100) < config.Traffic {
            useCanary = true
            version = *config.Canary
        }
    }
    
    // Load model
    model := d.loadVersion(name, version)
    
    versionStr := fmt.Sprintf("%d.%d.%d", version.Major, version.Minor, version.Patch)
    if useCanary {
        versionStr += "-canary"
    }
    
    return model, versionStr
}
```

## Model Compression

### Quantization

```go
type QuantizedModel struct {
    weights    []int8
    scale      float64
    zeroPoint  int8
    intercept  float64
}

func QuantizeModel(model *LinearRegression) *QuantizedModel {
    weights := model.GetWeights()
    
    // Find min and max
    minWeight, maxWeight := minMax(weights)
    
    // Calculate scale and zero point
    scale := (maxWeight - minWeight) / 255.0
    zeroPoint := int8(-minWeight / scale)
    
    // Quantize weights
    quantized := make([]int8, len(weights))
    for i, w := range weights {
        quantized[i] = int8(math.Round(w/scale)) + zeroPoint
    }
    
    return &QuantizedModel{
        weights:   quantized,
        scale:     scale,
        zeroPoint: zeroPoint,
        intercept: model.GetIntercept(),
    }
}

func (q *QuantizedModel) Predict(X mat.Matrix) (mat.Matrix, error) {
    rows, cols := X.Dims()
    predictions := mat.NewDense(rows, 1, nil)
    
    for i := 0; i < rows; i++ {
        sum := q.intercept
        for j := 0; j < cols; j++ {
            // Dequantize weight
            weight := float64(q.weights[j]-q.zeroPoint) * q.scale
            sum += X.At(i, j) * weight
        }
        predictions.Set(i, 0, sum)
    }
    
    return predictions, nil
}
```

### Pruning

```go
func PruneModel(model *LinearRegression, threshold float64) *SparseModel {
    weights := model.GetWeights()
    
    // Find non-zero weights
    sparseWeights := make(map[int]float64)
    for i, w := range weights {
        if math.Abs(w) > threshold {
            sparseWeights[i] = w
        }
    }
    
    sparsity := 1.0 - float64(len(sparseWeights))/float64(len(weights))
    log.Printf("Model sparsity: %.2f%%", sparsity*100)
    
    return &SparseModel{
        weights:   sparseWeights,
        intercept: model.GetIntercept(),
        nFeatures: len(weights),
    }
}
```

## Pipeline Persistence

### Saving Complete Pipelines

```go
type PipelineState struct {
    Steps []StepState `json:"steps"`
}

type StepState struct {
    Name   string                 `json:"name"`
    Type   string                 `json:"type"`
    Params map[string]interface{} `json:"params"`
}

func SavePipeline(pipeline *Pipeline, filename string) error {
    state := PipelineState{
        Steps: make([]StepState, len(pipeline.steps)),
    }
    
    for i, step := range pipeline.steps {
        state.Steps[i] = StepState{
            Name:   step.name,
            Type:   getTypeName(step.transformer),
            Params: step.transformer.GetParams(),
        }
    }
    
    data, err := json.MarshalIndent(state, "", "  ")
    if err != nil {
        return err
    }
    
    return ioutil.WriteFile(filename, data, 0644)
}

func LoadPipeline(filename string) (*Pipeline, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    
    var state PipelineState
    if err := json.Unmarshal(data, &state); err != nil {
        return nil, err
    }
    
    steps := make([]Step, len(state.Steps))
    for i, stepState := range state.Steps {
        transformer := createTransformer(stepState.Type)
        transformer.SetParams(stepState.Params)
        
        steps[i] = Step{
            name:        stepState.Name,
            transformer: transformer,
        }
    }
    
    return &Pipeline{steps: steps}, nil
}
```

## Model Serving

### HTTP Server

```go
type ModelServer struct {
    model    Model
    version  string
    metadata map[string]interface{}
}

func (s *ModelServer) Start(port int) {
    http.HandleFunc("/predict", s.handlePredict)
    http.HandleFunc("/model-info", s.handleInfo)
    http.HandleFunc("/reload", s.handleReload)
    
    log.Printf("Model server starting on port %d", port)
    log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))
}

func (s *ModelServer) handlePredict(w http.ResponseWriter, r *http.Request) {
    var request PredictRequest
    if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    // Convert to matrix
    X := sliceToMatrix(request.Features)
    
    // Predict
    predictions, err := s.model.Predict(X)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    response := PredictResponse{
        Predictions: matrixToSlice(predictions),
        Version:     s.version,
        Timestamp:   time.Now(),
    }
    
    json.NewEncoder(w).Encode(response)
}

func (s *ModelServer) handleReload(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var request ReloadRequest
    if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    // Load new model
    newModel, err := LoadModel(request.ModelPath)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    // Atomic swap
    s.model = newModel
    s.version = request.Version
    
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{
        "status": "reloaded",
        "version": s.version,
    })
}
```

### gRPC Service

```protobuf
// service.proto
service ModelService {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc GetModelInfo(Empty) returns (ModelInfo);
    rpc ReloadModel(ReloadRequest) returns (ReloadResponse);
}

message PredictRequest {
    repeated double features = 1;
    map<string, string> metadata = 2;
}

message PredictResponse {
    repeated double predictions = 1;
    string model_version = 2;
    int64 timestamp = 3;
}
```

```go
type grpcServer struct {
    model Model
}

func (s *grpcServer) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
    // Implementation
    predictions, err := s.model.Predict(convertFeatures(req.Features))
    if err != nil {
        return nil, status.Errorf(codes.Internal, "prediction failed: %v", err)
    }
    
    return &pb.PredictResponse{
        Predictions:  convertPredictions(predictions),
        ModelVersion: s.version,
        Timestamp:    time.Now().Unix(),
    }, nil
}
```

## Cloud Storage

### S3 Integration

```go
import (
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
)

type S3ModelStore struct {
    client *s3.S3
    bucket string
}

func NewS3ModelStore(bucket string) *S3ModelStore {
    sess := session.Must(session.NewSession())
    return &S3ModelStore{
        client: s3.New(sess),
        bucket: bucket,
    }
}

func (s *S3ModelStore) SaveModel(model Model, key string) error {
    // Serialize model
    var buf bytes.Buffer
    encoder := gob.NewEncoder(&buf)
    if err := encoder.Encode(model); err != nil {
        return err
    }
    
    // Upload to S3
    _, err := s.client.PutObject(&s3.PutObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
        Body:   bytes.NewReader(buf.Bytes()),
        Metadata: map[string]*string{
            "model-type": aws.String(getModelType(model)),
            "timestamp":  aws.String(time.Now().Format(time.RFC3339)),
        },
    })
    
    return err
}

func (s *S3ModelStore) LoadModel(key string) (Model, error) {
    // Download from S3
    result, err := s.client.GetObject(&s3.GetObjectInput{
        Bucket: aws.String(s.bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    defer result.Body.Close()
    
    // Deserialize model
    decoder := gob.NewDecoder(result.Body)
    model := createModelFromMetadata(result.Metadata)
    
    if err := decoder.Decode(model); err != nil {
        return nil, err
    }
    
    return model, nil
}
```

## Best Practices

1. **Version Everything**: Track model versions with metrics
2. **Include Metadata**: Save training parameters and timestamps
3. **Validate on Load**: Check model integrity after loading
4. **Use Compression**: For large models, compress before saving
5. **Atomic Updates**: Use atomic operations for model swapping
6. **Backup Models**: Keep multiple versions for rollback
7. **Monitor Performance**: Track prediction latency after loading
8. **Document Format**: Clear documentation of serialization format

## Security Considerations

```go
// Validate model before loading
func ValidateModel(filename string) error {
    // Check file size
    info, err := os.Stat(filename)
    if err != nil {
        return err
    }
    
    if info.Size() > MaxModelSize {
        return fmt.Errorf("model file too large: %d bytes", info.Size())
    }
    
    // Verify checksum
    expectedChecksum := getExpectedChecksum(filename)
    actualChecksum, err := calculateChecksum(filename)
    if err != nil {
        return err
    }
    
    if expectedChecksum != actualChecksum {
        return fmt.Errorf("checksum mismatch")
    }
    
    // Scan for malicious patterns
    if err := scanForMaliciousContent(filename); err != nil {
        return err
    }
    
    return nil
}
```

## Next Steps

- Explore [Production Deployment](../tutorials/production.md)
- Learn about [Model Monitoring](../advanced/monitoring.md)
- See [Persistence Examples](../../examples/persistence/)
- Read [API Reference](../api/core.md#persistence)