# GoML - Go Machine Learning Library

[![CI](https://github.com/YuminosukeSato/GoML/actions/workflows/ci.yml/badge.svg)](https://github.com/YuminosukeSato/GoML/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/YuminosukeSato/GoML)](https://goreportcard.com/report/github.com/YuminosukeSato/GoML)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Version](https://img.shields.io/badge/Go-1.21%2B-blue.svg)](https://go.dev/)

GoML is a high-performance machine learning library written in pure Go, designed for backend services and real-time inference applications.

GoMLは、バックエンドサービスとリアルタイム推論アプリケーション向けに設計された、純粋なGo言語で書かれた高性能機械学習ライブラリです。

## ✨ Features / 特徴

- 🚀 **High Performance** - CPU-parallel processing with automatic optimization
- 📚 **scikit-learn-like API** - Intuitive and familiar interface design
- 🛡️ **Robust Error Handling** - Comprehensive error management with cockroachdb/errors
- 💾 **Memory Efficient** - Optimized memory allocation (22-61 allocations for large datasets)
- 🧪 **Well Tested** - 91% test coverage for core modules
- 📊 **Built-in Metrics** - MSE, RMSE, MAE, R²Score, MAPE, and more

## 📦 Installation / インストール

```bash
go get github.com/YuminosukeSato/GoML
```

## 🚀 Quick Start / クイックスタート

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/GoML/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create a new linear regression model
    model := linear.NewLinearRegression()
    
    // Training data
    X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
    y := mat.NewDense(4, 1, []float64{2, 4, 6, 8})
    
    // Train the model
    if err := model.Fit(X, y); err != nil {
        log.Fatal(err)
    }
    
    // Make predictions
    X_test := mat.NewDense(2, 1, []float64{5, 6})
    predictions, err := model.Predict(X_test)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Predictions:", predictions)
}
```

## 📊 Performance Benchmarks / パフォーマンス

Benchmarked on Apple M2 Max:

| Dataset Size | Time | Memory | Allocations |
|-------------|------|--------|-------------|
| 100×10 | 21.7μs | 22.8KB | 22 |
| 1,000×10 | 178.8μs | 191.8KB | 22 |
| 10,000×20 | 4.5ms | 3.4MB | 57 |
| 50,000×50 | 65.9ms | 41.2MB | 61 |

### Parallel Processing / 並列処理

- Automatic parallelization for datasets with >1000 rows
- CPU core detection and optimal worker allocation
- Thread-safe operations with sync.WaitGroup

## 🛠️ Implemented Features / 実装済み機能

### Models / モデル
- ✅ Linear Regression
- 🚧 Logistic Regression (coming soon)
- 🚧 Random Forest (coming soon)
- 🚧 Gradient Boosting (coming soon)

### Metrics / 評価指標
- ✅ Mean Squared Error (MSE)
- ✅ Root Mean Squared Error (RMSE)
- ✅ Mean Absolute Error (MAE)
- ✅ R² Score
- ✅ Mean Absolute Percentage Error (MAPE)
- ✅ Explained Variance Score

### Core Features / コア機能
- ✅ Tensor operations (wrapper for gonum/mat)
- ✅ Parallel processing utilities
- ✅ Comprehensive error handling
- ✅ Model interfaces (Fitter, Predictor, Transformer)

## 📚 Documentation / ドキュメント

### Package Structure / パッケージ構造

```
goml/
├── core/           # Core interfaces and utilities
│   ├── model/      # Model interfaces (Fitter, Predictor)
│   ├── tensor/     # Tensor operations
│   └── parallel/   # Parallel processing
├── linear/         # Linear models
├── metrics/        # Evaluation metrics
├── datasets/       # Dataset utilities
└── examples/       # Usage examples
```

### Examples / 使用例

Check out the [examples](./examples) directory for more detailed usage:

- [Linear Regression](./examples/linear_regression/main.go)
- [Iris Dataset Regression](./examples/iris_regression/main.go)
- [Error Handling Demo](./examples/error_demo/main.go)

## 🧪 Testing / テスト

Run tests:
```bash
go test ./...
```

Run benchmarks:
```bash
go test -bench=. -benchmem ./linear/...
```

Check coverage:
```bash
go test -cover ./...
```

## 🤝 Contributing / コントリビューション

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Roadmap / ロードマップ

- [ ] More ML algorithms (Decision Trees, SVM, Neural Networks)
- [ ] Model serialization and deserialization
- [ ] Distributed training support
- [ ] GPU acceleration support
- [ ] AutoML capabilities
- [ ] More preprocessing utilities

## 📝 License / ライセンス

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments / 謝辞

- [gonum](https://www.gonum.org/) - Numerical libraries for Go
- [cockroachdb/errors](https://github.com/cockroachdb/errors) - Enhanced error handling
- scikit-learn - API design inspiration

## 📧 Contact / 連絡先

- GitHub: [@YuminosukeSato](https://github.com/YuminosukeSato)
- Repository: [https://github.com/YuminosukeSato/GoML](https://github.com/YuminosukeSato/GoML)

---

Made with ❤️ in Go