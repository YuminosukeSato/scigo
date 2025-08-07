# SciGo Development Roadmap

## Overview

This document outlines the development roadmap for SciGo, a blazing-fast scikit-learn compatible ML library for Go. Our goal is to reach v1.0.0 with a stable, production-ready API and comprehensive machine learning capabilities.

## Release Philosophy

- **v0.x releases**: Active development, API may change
- **v1.0.0**: API stability guarantee, backward compatibility commitment
- **Semantic Versioning**: Following [semver.org](https://semver.org/) strictly

## Current Status (v0.3.0)

### ✅ Implemented Features

- **Linear Models**
  - LinearRegression (QR decomposition)
  - SGDRegressor / SGDClassifier
  - PassiveAggressive algorithms

- **Preprocessing**
  - StandardScaler
  - MinMaxScaler
  - OneHotEncoder

- **Clustering**
  - MiniBatch K-Means

- **Tree Models**
  - LightGBM inference (Python model compatibility)

- **Advanced Features**
  - Full scikit-learn API compatibility
  - Online learning / Streaming support
  - gRPC/Protobuf support
  - Model serialization framework

## Roadmap to v1.0.0

### Phase 1: Core ML Algorithms

#### v0.4.0 - Essential Classification (Target: 2-3 weeks)
- [ ] **LogisticRegression**
  - Binary and multiclass classification
  - L1/L2 regularization
  - Solver options (lbfgs, liblinear, newton-cg)
- [ ] **DecisionTree**
  - DecisionTreeClassifier
  - DecisionTreeRegressor
  - Feature importance calculation
  - Tree visualization support

#### v0.5.0 - Advanced Algorithms (Target: 2-3 weeks)
- [ ] **RandomForest**
  - RandomForestClassifier
  - RandomForestRegressor
  - Out-of-bag score
  - Parallel tree training
- [ ] **SVM (Support Vector Machines)**
  - SVC (Support Vector Classifier)
  - SVR (Support Vector Regressor)
  - Kernel support (linear, rbf, poly, sigmoid)

### Phase 2: Ensemble Methods

#### v0.6.0 - XGBoost Integration (Target: 3-4 weeks)
- [ ] **XGBoost**
  - XGBClassifier
  - XGBRegressor
  - Python model compatibility (.json, .ubj formats)
  - Native Go training implementation
  - GPU acceleration support (optional)

#### v0.7.0 - LightGBM Complete (Target: 2-3 weeks)
- [ ] **LightGBM Training**
  - Native Go training implementation
  - Full feature parity with Python API
  - Categorical feature support
  - Early stopping
  - Custom objective functions

### Phase 3: Stabilization

#### v0.8.0 - Performance & Optimization (Target: 1-2 weeks)
- [ ] **Performance Improvements**
  - SIMD optimizations
  - Memory pooling
  - Parallel processing enhancements
- [ ] **Benchmarking Suite**
  - Comprehensive performance tests
  - Comparison with scikit-learn
  - Memory profiling

#### v0.9.0 - API Finalization (Target: 1-2 weeks)
- [ ] **API Review**
  - Final API adjustments
  - Deprecation of experimental features
  - Interface stabilization
- [ ] **Documentation**
  - Complete API documentation
  - Migration guides
  - Best practices guide
- [ ] **Testing**
  - 90%+ test coverage
  - Integration tests
  - Cross-compatibility tests with Python

### Phase 4: Production Ready

#### v1.0.0 - Stable Release (Target: Q2 2025)
- [ ] **API Stability Guarantee**
  - Backward compatibility commitment
  - Long-term support (LTS) declaration
  - Deprecation policy establishment
- [ ] **Production Features**
  - Model versioning
  - A/B testing support
  - Monitoring and metrics
- [ ] **Ecosystem**
  - Plugin system
  - Community contributions guide
  - Partner integrations

## Feature Priority Matrix

| Feature | Priority | Complexity | Status |
|---------|----------|------------|--------|
| LogisticRegression | High | Medium | Planned (v0.4.0) |
| DecisionTree | High | Medium | Planned (v0.4.0) |
| RandomForest | High | High | Planned (v0.5.0) |
| SVM | Medium | High | Planned (v0.5.0) |
| XGBoost | High | Very High | Planned (v0.6.0) |
| LightGBM Training | High | High | Planned (v0.7.0) |
| Neural Networks | Low | Very High | Post-v1.0.0 |
| Deep Learning | Low | Very High | Post-v1.0.0 |

## API Stability Commitment

### Pre-v1.0.0
- APIs may change between minor versions
- Breaking changes will be documented in CHANGELOG
- Deprecation warnings for at least one version

### Post-v1.0.0
- No breaking changes in minor/patch releases
- Deprecated features maintained for at least 3 minor versions
- Clear migration paths for any future major version

## Contributing

We welcome contributions! Priority areas for community help:

1. **Algorithm Implementations**
   - Help implement planned algorithms
   - Optimize existing implementations

2. **Testing**
   - Increase test coverage
   - Add benchmark tests
   - Cross-validation with scikit-learn

3. **Documentation**
   - Improve examples
   - Add tutorials
   - Translate documentation

4. **Performance**
   - SIMD optimizations
   - GPU acceleration
   - Memory optimizations

## Success Metrics

- **Performance**: 2-5x faster than scikit-learn for common operations
- **Compatibility**: 100% API compatibility for implemented features
- **Quality**: 90%+ test coverage, zero critical bugs
- **Adoption**: 1000+ GitHub stars, 50+ production deployments
- **Community**: 20+ active contributors

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| API design flaws discovered late | Extensive testing in v0.x releases |
| Performance regression | Continuous benchmarking CI |
| Breaking scikit-learn compatibility | Automated compatibility tests |
| Slow adoption | Focus on documentation and examples |
| Technical debt | Regular refactoring sprints |

## Timeline Summary

- **v0.4.0**: ~2-3 weeks (LogisticRegression, DecisionTree)
- **v0.5.0**: ~2-3 weeks (RandomForest, SVM)
- **v0.6.0**: ~3-4 weeks (XGBoost)
- **v0.7.0**: ~2-3 weeks (LightGBM Training)
- **v0.8.0**: ~1-2 weeks (Performance)
- **v0.9.0**: ~1-2 weeks (API Finalization)
- **v1.0.0**: ~2-3 months from now (estimated Q2 2025)

## Contact

- **Project Lead**: Yuminosuke Sato
- **GitHub**: [https://github.com/YuminosukeSato/scigo](https://github.com/YuminosukeSato/scigo)
- **Issues**: [GitHub Issues](https://github.com/YuminosukeSato/scigo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YuminosukeSato/scigo/discussions)

---

*This roadmap is subject to change based on community feedback and project priorities. Last updated: 2025-08-07*