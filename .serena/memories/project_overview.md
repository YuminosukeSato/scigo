# SciGo Project Overview

## Purpose
SciGo is a Go implementation of machine learning algorithms, providing scikit-learn compatible interfaces. The project includes implementations of:
- Linear models (LinearRegression, SGDRegressor, SGDClassifier, PassiveAggressiveRegressor)
- LightGBM (LGBMRegressor, LGBMClassifier)
- Preprocessing tools (StandardScaler, OneHotEncoder)
- Metrics (MSE, MAE, R2, AUC, NDCG)
- Clustering (MiniBatchKMeans)
- Naive Bayes (MultinomialNB)
- Decision Trees
- Pipeline support

## Tech Stack
- Language: Go 1.24.0
- Main Dependencies:
  - gonum.org/v1/gonum v0.16.0 (numerical computing)
  - gonum.org/v1/plot v0.16.0 (plotting)
  - github.com/stretchr/testify v1.8.4 (testing)
  - github.com/cockroachdb/errors v1.12.0 (error handling)
  - github.com/rs/zerolog v1.34.0 (logging)

## Project Structure
- `/core` - Core functionality (model, tensor, parallel processing)
- `/linear` - Linear regression implementation
- `/sklearn` - Scikit-learn compatible implementations
  - `/lightgbm` - LightGBM implementation
  - `/linear_model` - Linear models
  - `/cluster` - Clustering algorithms
  - `/naive_bayes` - Naive Bayes classifiers
  - `/tree` - Decision trees
  - `/pipeline` - Pipeline support
- `/preprocessing` - Data preprocessing tools
- `/metrics` - Evaluation metrics
- `/examples` - Example usage code
- `/tests` - Integration tests
- `/benchmarks` - Performance benchmarks
- `/pkg` - Shared packages (errors, logging)