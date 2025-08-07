---
title: "Documentation"
linkTitle: "Documentation"
weight: 20
menu:
  main:
    weight: 20
description: >
  Complete documentation for SciGo - the blazing-fast ML library for Go
---

# SciGo Documentation

Welcome to the complete documentation for SciGo - the blazing-fast, scikit-learn compatible machine learning library for Go.

## Getting Started

{{< cardpane >}}
{{< card header="Quick Start" >}}
Get up and running with SciGo in under 30 seconds.

[Quick Start Guide →](./getting-started/quickstart/)
{{< /card >}}

{{< card header="Installation" >}}
Multiple ways to install SciGo in your Go project.

[Installation Guide →](./getting-started/installation/)
{{< /card >}}

{{< card header="First Model" >}}
Train your first machine learning model with SciGo.

[First Model →](./getting-started/first-model/)
{{< /card >}}
{{< /cardpane >}}

## Core Concepts

{{< cardpane >}}
{{< card header="API Design" >}}
Understanding SciGo's scikit-learn compatible API.

[API Design →](./concepts/api-design/)
{{< /card >}}

{{< card header="Data Structures" >}}
Working with matrices and data in SciGo.

[Data Structures →](./concepts/data-structures/)
{{< /card >}}

{{< card header="Error Handling" >}}
Robust error handling patterns in SciGo.

[Error Handling →](./concepts/error-handling/)
{{< /card >}}
{{< /cardpane >}}

## Algorithms

### Supervised Learning
- [Linear Regression](./algorithms/linear-regression/)
- [SGD Classifier](./algorithms/sgd-classifier/)
- [SGD Regressor](./algorithms/sgd-regressor/)
- [Passive-Aggressive](./algorithms/passive-aggressive/)
- [LightGBM](./algorithms/lightgbm/)

### Unsupervised Learning
- [MiniBatch K-Means](./algorithms/minibatch-kmeans/)

### Preprocessing
- [StandardScaler](./preprocessing/standard-scaler/)
- [MinMaxScaler](./preprocessing/minmax-scaler/)
- [OneHotEncoder](./preprocessing/onehot-encoder/)

## Advanced Topics

{{< cardpane >}}
{{< card header="Streaming ML" >}}
Real-time machine learning with streaming algorithms.

[Streaming Guide →](./advanced/streaming/)
{{< /card >}}

{{< card header="Performance" >}}
Optimization techniques and performance tuning.

[Performance Guide →](./advanced/performance/)
{{< /card >}}

{{< card header="Production Deployment" >}}
Best practices for production deployment.

[Deployment Guide →](./advanced/deployment/)
{{< /card >}}
{{< /cardpane >}}

## Migration & Integration

{{< cardpane >}}
{{< card header="scikit-learn Migration" >}}
Migrate from Python scikit-learn to Go SciGo.

[Migration Guide →](./migration/sklearn-migration/)
{{< /card >}}

{{< card header="LightGBM Integration" >}}
Use existing Python LightGBM models in Go.

[LightGBM Guide →](./integration/lightgbm/)
{{< /card >}}

{{< card header="Docker & Cloud" >}}
Deploy SciGo applications with Docker and cloud platforms.

[Cloud Guide →](./deployment/cloud/)
{{< /card >}}
{{< /cardpane >}}

## Reference

- [**API Reference**](https://pkg.go.dev/github.com/YuminosukeSato/scigo) - Complete Go package documentation
- [**Examples**](./examples/) - Code examples and tutorials  
- [**FAQ**](./faq/) - Frequently asked questions
- [**Troubleshooting**](./troubleshooting/) - Common issues and solutions

## Community

- [GitHub Repository](https://github.com/YuminosukeSato/scigo)
- [Issue Tracker](https://github.com/YuminosukeSato/scigo/issues)
- [Discussions](https://github.com/YuminosukeSato/scigo/discussions)
- [Contributing Guide](./contributing/)

---

## What's New

### v0.2.0 Highlights

- **LightGBM Support**: Full Python model compatibility
- **Enhanced Error Handling**: Comprehensive panic recovery
- **Streaming Algorithms**: Real-time ML capabilities
- **Improved Performance**: 3.6× faster than scikit-learn

[See Full Changelog →](./changelog/)