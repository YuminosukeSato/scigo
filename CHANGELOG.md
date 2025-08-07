# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-08-07

### Added
- **LogisticRegression** implementation
  - Binary and multiclass classification (one-vs-rest)
  - Gradient descent optimization
  - L2 regularization support
  - Probability predictions (PredictProba)
  - Full scikit-learn API compatibility
  
- **DecisionTreeClassifier** implementation
  - CART algorithm with Gini and Entropy criteria
  - Feature importance calculation
  - Max depth and min samples constraints
  - Multiclass classification support
  - Probability predictions (PredictProba)
  - Tree structure introspection (GetDepth, GetNLeaves)

### Fixed
- Test stability for XOR pattern in DecisionTree
- Convergence issues in LogisticRegression tests

### Planned for v0.5.0
- RandomForestClassifier and RandomForestRegressor
- Support Vector Machines (SVM) implementation
- Kernel methods support

### Planned for v0.6.0
- XGBoost integration with Python model compatibility
- Native Go training for gradient boosting

### Planned for v0.7.0
- LightGBM native training implementation
- Categorical feature support

### Added
- GitHub Releases automation with multi-platform binaries
- Enhanced CI/CD pipeline with security scanning
- Development roadmap (ROADMAP.md)

### Changed
- Improved release workflow with automatic changelog generation

## [0.3.0] - 2025-08-07

### Added
- 🎯 **Full scikit-learn API Compatibility**
  - Complete `GetParams`/`SetParams` implementation for all models
  - Standardized parameter management across all estimators
  - Full compatibility with scikit-learn's estimator interface

- ⚖️ **Model Weight Reproducibility**
  - Guaranteed identical predictions with the same weights
  - Full floating-point precision preservation in serialization
  - Comprehensive weight validation and checksums

- 🔄 **gRPC/Protobuf Support**
  - Complete protobuf definitions for model weights and matrices
  - gRPC service definitions for distributed training/prediction
  - Support for streaming predictions over gRPC
  - Weight transfer with full precision guarantee

- 📦 **Model Serialization Framework**
  - `ExportWeights`/`ImportWeights` API for all models
  - JSON serialization with full precision
  - Model versioning and compatibility checks
  - Weight validation and integrity verification

- 🏗️ **Mixin Pattern Implementation**
  - `ClassifierMixin` for classification models
  - `RegressorMixin` for regression models  
  - `TransformerMixin` for preprocessing transformers
  - `WeightExporter` interface for weight management
  - `PartialFitMixin` for incremental learning

- ✨ **LinearRegression Implementation**
  - Full ordinary least squares implementation with QR decomposition
  - Complete scikit-learn API compatibility
  - Support for intercept fitting and positive constraints
  - Comprehensive weight export/import support

### Changed
- `BaseEstimator` enhanced with parameter management and weight export capabilities
- Improved model interfaces with standardized weight management
- Refactored `ModelWeights` into separate module for better organization

### Fixed
- Function name conflicts in linear_model package (`WithFitIntercept` → `WithLRFitIntercept`)
- Unused imports and type mismatches in linear regression
- Test data singularity issues in weight reproducibility tests
- QR decomposition method calls corrected

### Technical Improvements
- Added comprehensive weight reproducibility tests
- Implemented weight hash verification for integrity checks
- Added benchmarks for weight export/import operations
- Enhanced error handling for model serialization

## [0.2.0] - 2025-08-07

### Added
- LightGBM inference support with Python model compatibility
- Comprehensive error handling with panic recovery
- Structured logging with slog-compatible interface
- Memory-efficient streaming support for online learning

### Changed
- Improved API documentation with pkg.go.dev compatibility
- Enhanced test coverage to 76.7%
- Updated CI/CD pipeline with dependency scanning

### Fixed
- Type conversion issues in metrics package
- Memory leaks in streaming operations

## [0.1.0] - 2025-08-06

### Added
- Initial release with scikit-learn compatible API
- Core ML algorithms:
  - Linear Regression with OLS
  - SGD Classifier/Regressor
  - Passive-Aggressive algorithms
  - MiniBatch K-Means clustering
- Data preprocessing:
  - StandardScaler
  - MinMaxScaler
  - OneHotEncoder
- Evaluation metrics:
  - MSE, RMSE, MAE, R²
  - MAPE, Explained Variance Score
- Online learning capabilities:
  - Incremental learning with partial_fit
  - Concept drift detection (DDM, ADWIN)
  - Streaming pipelines
- LightGBM model inference (read-only)
- Comprehensive test suite with 76.7% coverage
- Full API documentation on pkg.go.dev

### Security
- Input validation for all public APIs
- Safe error handling with panic recovery

### Known Issues
- LightGBM training not yet implemented (inference only)
- No GPU/SIMD acceleration
- Limited to float64 data types
- No support for missing values or categorical variables
- No ONNX/Pickle compatibility

[Unreleased]: https://github.com/YuminosukeSato/scigo/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/YuminosukeSato/scigo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/YuminosukeSato/scigo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/YuminosukeSato/scigo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/YuminosukeSato/scigo/releases/tag/v0.1.0