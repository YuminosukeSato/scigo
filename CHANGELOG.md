# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Releases automation with multi-platform binaries
- Enhanced CI/CD pipeline with security scanning

### Changed
- Improved release workflow with automatic changelog generation

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

[Unreleased]: https://github.com/YuminosukeSato/scigo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/YuminosukeSato/scigo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/YuminosukeSato/scigo/releases/tag/v0.1.0