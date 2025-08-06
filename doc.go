// Package goml provides a high-performance machine learning library for Go,
// designed for backend services and real-time inference applications.
//
// GoML offers a scikit-learn-like API that makes it easy for data scientists
// and engineers familiar with Python's ecosystem to build machine learning
// applications in Go.
//
// # Features
//
// - High Performance: CPU-parallel processing with automatic optimization
// - scikit-learn-like API: Familiar interface design for easy adoption
// - Robust Error Handling: Comprehensive error management
// - Memory Efficient: Optimized memory allocation
// - Well Tested: 91% test coverage for core modules
//
// # Installation
//
// Install GoML using go get:
//
//	go get github.com/YuminosukeSato/GoML
//
// # Quick Start
//
// Here's a simple example of linear regression:
//
//	package main
//
//	import (
//	    "fmt"
//	    "log"
//	    "github.com/YuminosukeSato/GoML/linear"
//	    "gonum.org/v1/gonum/mat"
//	)
//
//	func main() {
//	    // Create training data
//	    X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
//	    y := mat.NewDense(4, 1, []float64{2, 4, 6, 8})
//
//	    // Create and train model
//	    model := linear.NewLinearRegression()
//	    if err := model.Fit(X, y); err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // Make predictions
//	    X_test := mat.NewDense(2, 1, []float64{5, 6})
//	    predictions, err := model.Predict(X_test)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    fmt.Println("Predictions:", predictions)
//	}
//
// # Packages
//
// The library is organized into several packages:
//
//   - linear: Linear models (LinearRegression, LogisticRegression)
//   - tree: Tree-based models (coming soon)
//   - neural: Neural network models (coming soon)
//   - metrics: Evaluation metrics (MSE, RMSE, MAE, R²)
//   - preprocessing: Data preprocessing utilities
//   - core/model: Core interfaces and base types
//   - core/tensor: Tensor operations
//   - core/parallel: Parallel processing utilities
//
// # scikit-learn Compatibility
//
// GoML provides scikit-learn compatible implementations:
//
//	// Using scikit-learn compatible API
//	model := linear.NewSKLinearRegression(
//	    linear.WithFitIntercept(true),
//	    linear.WithCopyX(true),
//	    linear.WithNJobs(-1),  // Use all CPU cores
//	)
//
// # Performance
//
// GoML is optimized for performance with automatic parallelization:
//
//   - Automatic parallelization for datasets with >1000 rows
//   - CPU core detection and optimal worker allocation
//   - Thread-safe operations
//
// Benchmark results on Apple M2 Max:
//   - 100×10 dataset: 21.7μs
//   - 1,000×10 dataset: 178.8μs
//   - 50,000×50 dataset: 65.9ms
//
// # Contributing
//
// Contributions are welcome! Please see our GitHub repository:
// https://github.com/YuminosukeSato/GoML
//
// # License
//
// GoML is released under the MIT License.
package goml