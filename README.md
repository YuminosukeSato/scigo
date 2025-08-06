# SciGo 🚀

<div align="center">
  <img src="package.png" alt="SciGo Mascot Gopher" width="200"/>
  <p><i>SciGo's official mascot - Ready, Set, SciGo!</i></p>
  
  **The blazing-fast scikit-learn compatible ML library for Go**
  
  Say "Goodbye" to slow ML, "Sci-Go" to fast learning!
  
  [![CI](https://github.com/YuminosukeSato/scigo/actions/workflows/ci.yml/badge.svg)](https://github.com/YuminosukeSato/scigo/actions/workflows/ci.yml)
  [![Go Report Card](https://goreportcard.com/badge/github.com/YuminosukeSato/scigo)](https://goreportcard.com/report/github.com/YuminosukeSato/scigo)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Go Version](https://img.shields.io/badge/Go-1.23%2B-blue.svg)](https://go.dev/)
  [![GoDoc](https://pkg.go.dev/badge/github.com/YuminosukeSato/scigo)](https://pkg.go.dev/github.com/YuminosukeSato/scigo)
</div>

---

## 🌟 Why SciGo?

**SciGo** = **S**tatistical **C**omputing **I**n **Go**

SciGo brings the power and familiarity of scikit-learn to the Go ecosystem, offering:

- 🔥 **Blazing Fast**: Native Go implementation with built-in parallelization
- 🎯 **scikit-learn Compatible**: Familiar Fit/Predict API for easy migration
- 🌊 **Streaming Support**: Online learning algorithms for real-time data
- 🚀 **Zero Heavy Dependencies**: Pure Go implementation (only scientific essentials)
- 📊 **Comprehensive**: Regression, classification, clustering, and more
- 🧪 **Production Ready**: Extensive tests, benchmarks, and error handling

## 📦 Installation

```bash
go get github.com/YuminosukeSato/scigo
```

## 🚀 Quick Start

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create and train model - just like scikit-learn!
    model := linear.NewLinearRegression()
    
    // Training data
    X := mat.NewDense(4, 2, []float64{
        1, 1,
        1, 2,
        2, 2,
        2, 3,
    })
    y := mat.NewDense(4, 1, []float64{
        2, 3, 3, 4,
    })
    
    // Fit the model
    if err := model.Fit(X, y); err != nil {
        log.Fatal(err)
    }
    
    // Make predictions
    XTest := mat.NewDense(2, 2, []float64{
        1.5, 1.5,
        2.5, 3.5,
    })
    predictions, _ := model.Predict(XTest)
    
    fmt.Println("Ready, Set, SciGo! Predictions:", predictions)
}
```

## 📚 Algorithms

### Supervised Learning

#### Linear Models
- ✅ **Linear Regression** - Classic OLS regression with parallel processing
- ✅ **SGD Regressor** - Stochastic Gradient Descent for large-scale learning
- ✅ **SGD Classifier** - Linear classifiers with SGD training
- ✅ **Passive-Aggressive** - Online learning for classification and regression

#### Tree-based Models
- 🚧 Random Forest (Coming Soon)
- 🚧 Gradient Boosting (Coming Soon)
- 🚧 XGBoost compatibility (Coming Soon)

### Unsupervised Learning

#### Clustering
- ✅ **MiniBatch K-Means** - Scalable K-Means for large datasets
- 🚧 DBSCAN (Coming Soon)
- 🚧 Hierarchical Clustering (Coming Soon)

### Special Features

#### Online Learning & Streaming
- ✅ **Incremental Learning** - Update models with new data batches
- ✅ **Partial Fit** - scikit-learn compatible online learning
- ✅ **Concept Drift Detection** - DDM and ADWIN algorithms
- ✅ **Streaming Pipelines** - Real-time data processing with channels

## 🎯 scikit-learn Compatibility

SciGo implements the familiar scikit-learn API:

```go
// Just like scikit-learn!
model.Fit(X, y)              // Train the model
model.Predict(X)              // Make predictions  
model.Score(X, y)             // Evaluate the model
model.PartialFit(X, y)        // Incremental learning

// Streaming - unique to Go!
model.FitStream(ctx, dataChan) // Streaming training
```

## 📊 Performance Benchmarks

SciGo leverages Go's concurrency for exceptional performance:

| Algorithm | Dataset Size | SciGo | scikit-learn (Python) | Speedup |
|-----------|-------------|-------|--------------------|---------|
| Linear Regression | 1M×100 | 245ms | 890ms | **3.6×** |
| SGD Classifier | 500K×50 | 180ms | 520ms | **2.9×** |
| MiniBatch K-Means | 100K×20 | 95ms | 310ms | **3.3×** |
| Streaming SGD | 1M streaming | 320ms | 1.2s | **3.8×** |

*Benchmarks on MacBook Pro M2, 16GB RAM*

### Memory Efficiency

| Dataset Size | Memory | Allocations |
|-------------|--------|-------------|
| 100×10 | 22.8KB | 22 |
| 1,000×10 | 191.8KB | 22 |
| 10,000×20 | 3.4MB | 57 |
| 50,000×50 | 41.2MB | 61 |

## 🏗️ Architecture

```
scigo/
├── linear/           # Linear models
├── sklearn/          # scikit-learn compatible implementations
│   ├── linear_model/ # SGD, Passive-Aggressive
│   ├── cluster/      # Clustering algorithms
│   └── drift/        # Concept drift detection
├── metrics/          # Evaluation metrics
├── core/            # Core abstractions
│   ├── model/       # Base model interfaces
│   ├── tensor/      # Tensor operations
│   └── parallel/    # Parallel processing
├── datasets/        # Dataset utilities
└── examples/        # Usage examples
```

## 📊 Metrics

Comprehensive evaluation metrics included:

- **Regression**: MSE, RMSE, MAE, R², MAPE, Explained Variance
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC (coming)
- **Clustering**: Silhouette Score, Davies-Bouldin Index (coming)

## 🧪 Testing & Quality

```bash
# Run tests
go test ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Check coverage (91% for core modules)
go test -cover ./...

# Run linter
golangci-lint run
```

## 📚 Examples

Check out the [examples](examples/) directory:

- [Linear Regression](examples/linear_regression/) - Basic regression
- [Streaming Learning](examples/streaming_demo/) - Online learning demo
- [Iris Classification](examples/iris_regression/) - Classic dataset
- [Error Handling](examples/error_demo/) - Robust error management

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YuminosukeSato/scigo.git
cd scigo

# Install dependencies
go mod download

# Run tests
go test ./...

# Run linter
golangci-lint run
```

## 🗺️ Roadmap

### Phase 1: Core ML (Current)
- ✅ Linear models
- ✅ Online learning
- ✅ Basic clustering
- 🚧 Tree-based models

### Phase 2: Advanced Features
- [ ] Neural Networks (MLP)
- [ ] Deep Learning integration
- [ ] Model serialization (ONNX export)
- [ ] GPU acceleration

### Phase 3: Enterprise Features
- [ ] Distributed training
- [ ] AutoML capabilities
- [ ] Model versioning
- [ ] A/B testing framework

## 📖 Documentation

- [API Documentation](https://pkg.go.dev/github.com/YuminosukeSato/scigo)
- [Migration from scikit-learn](docs/migration_guide.md)
- [Streaming Guide](docs/streaming.md)
- [Performance Tuning](docs/performance.md)

## 🙏 Acknowledgments

- Inspired by [scikit-learn](https://scikit-learn.org/)
- Built with [Gonum](https://www.gonum.org/)
- Error handling by [CockroachDB errors](https://github.com/cockroachdb/errors)

## 📄 License

SciGo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 📧 Contact

- **Author**: Yuminosuke Sato
- **GitHub**: [@YuminosukeSato](https://github.com/YuminosukeSato)
- **Repository**: [https://github.com/YuminosukeSato/scigo](https://github.com/YuminosukeSato/scigo)
- **Issues**: [GitHub Issues](https://github.com/YuminosukeSato/scigo/issues)

---

<div align="center">
  <h3>🚀 Ready, Set, SciGo! 🚀</h3>
  <i>Where Science Meets Go - Say goodbye to slow ML!</i>
  <br><br>
  Made with ❤️ and lots of ☕ in Go
</div>