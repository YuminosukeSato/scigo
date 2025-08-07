---
title: "SciGo"
linkTitle: "SciGo"
description: "The blazing-fast scikit-learn compatible ML library for Go"
---

{{< blocks/cover title="SciGo 🚀" image_anchor="top" height="full" color="primary" >}}
<div class="mx-auto">
  <img src="images/GOpher.png" alt="SciGo Mascot" class="mb-4" style="max-width: 200px;" />
  <h2 class="mb-3">The blazing-fast scikit-learn compatible ML library for Go</h2>
  <p class="lead mb-4">Say "Goodbye" to slow ML, "Sci-Go" to fast learning!</p>
  
  <div class="d-flex flex-wrap justify-content-center gap-2 mb-4">
    <img src="https://github.com/YuminosukeSato/scigo/actions/workflows/ci.yml/badge.svg" alt="CI Status" />
    <img src="https://codecov.io/gh/YuminosukeSato/scigo/branch/main/graph/badge.svg" alt="Coverage" />
    <img src="https://goreportcard.com/badge/github.com/YuminosukeSato/scigo" alt="Go Report Card" />
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" />
    <img src="https://img.shields.io/badge/Go-1.23%2B-blue.svg" alt="Go Version" />
  </div>
  
  <div class="mx-auto mt-5">
    {{< blocks/link-down color="secondary" >}}
  </div>
</div>
{{< /blocks/cover >}}

{{< blocks/lead color="dark" >}}
**SciGo** = **S**tatistical **C**omputing **I**n **Go**

SciGo brings the power and familiarity of scikit-learn to the Go ecosystem, offering blazing-fast performance with zero compromise on ease of use.
{{< /blocks/lead >}}

{{% blocks/section color="white" %}}

## Quick Start

Get started with SciGo in less than 30 seconds:

### Installation

```bash
go get github.com/YuminosukeSato/scigo
```

### Your First Model

```go
package main

import (
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create and train model - just like scikit-learn!
    model := linear.NewLinearRegression()
    
    X := mat.NewDense(4, 2, []float64{1, 1, 1, 2, 2, 2, 2, 3})
    y := mat.NewDense(4, 1, []float64{2, 3, 3, 4})
    
    model.Fit(X, y)
    predictions, _ := model.Predict(X)
    
    // Ready, Set, SciGo! 🚀
}
```

{{% /blocks/section %}}

{{< blocks/section color="primary" >}}

{{% blocks/feature icon="fas fa-rocket" title="Blazing Fast Performance" %}}
Native Go implementation with built-in parallelization delivers **3.6×** speedup over scikit-learn.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-code" title="scikit-learn Compatible API" %}}
Familiar `Fit()`, `Predict()`, `Transform()` methods make migration effortless for Python developers.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-tree" title="LightGBM Integration" %}}
Load and inference Python LightGBM models directly - full `.txt`/`.json` format support.
{{% /blocks/feature %}}

{{< /blocks/section >}}

{{< blocks/section color="white" >}}

{{% blocks/feature icon="fas fa-book" title="Complete Documentation" %}}
Comprehensive API documentation with examples on [pkg.go.dev](https://pkg.go.dev/github.com/YuminosukeSato/scigo).
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-stream" title="Real-time ML" %}}
Online learning algorithms with streaming support for production-scale real-time inference.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-shield-alt" title="Production Ready" %}}
Extensive test coverage (76.7%), comprehensive error handling, and memory safety guarantees.
{{% /blocks/feature %}}

{{< /blocks/section >}}

{{< blocks/section color="dark" >}}

## Algorithms & Features

<div class="row">
  <div class="col-lg-6">
    <h3>🧠 Supervised Learning</h3>
    <ul>
      <li><strong>Linear Models:</strong> Linear Regression, SGD Classifier/Regressor</li>
      <li><strong>Tree Models:</strong> LightGBM (inference + training)</li>
      <li><strong>Online Learning:</strong> Passive-Aggressive algorithms</li>
    </ul>
    
    <h3>🔧 Data Preprocessing</h3>
    <ul>
      <li>StandardScaler - Z-score normalization</li>
      <li>MinMaxScaler - Range scaling</li>
      <li>OneHotEncoder - Categorical encoding</li>
    </ul>
  </div>
  
  <div class="col-lg-6">
    <h3>🎯 Unsupervised Learning</h3>
    <ul>
      <li><strong>Clustering:</strong> MiniBatch K-Means</li>
      <li><strong>Dimensionality Reduction:</strong> Coming Soon</li>
    </ul>
    
    <h3>📊 Model Evaluation</h3>
    <ul>
      <li>Regression: MSE, RMSE, MAE, R², MAPE</li>
      <li>Classification: Accuracy, Precision, Recall (Coming)</li>
      <li>Clustering: Silhouette Score (Coming)</li>
    </ul>
  </div>
</div>

{{< /blocks/section >}}

{{< blocks/section color="secondary" >}}

## Performance Benchmarks

SciGo leverages Go's concurrency for exceptional performance:

| Algorithm | Dataset Size | SciGo | scikit-learn | Speedup |
|-----------|-------------|-------|--------------|---------|
| Linear Regression | 1M×100 | **245ms** | 890ms | **3.6×** |
| SGD Classifier | 500K×50 | **180ms** | 520ms | **2.9×** |
| MiniBatch K-Means | 100K×20 | **95ms** | 310ms | **3.3×** |
| Streaming SGD | 1M streaming | **320ms** | 1.2s | **3.8×** |

*Benchmarks on MacBook Pro M2, 16GB RAM*

{{< /blocks/section >}}