# Contributing to SciGo

Thank you for your interest in contributing to SciGo! We welcome contributions from everyone, whether you're fixing a typo, adding a test, implementing a new algorithm, or improving documentation.

## 🚀 Quick Start (5 minutes to your first contribution!)

### Prerequisites
- Go 1.21 or later
- Git
- Make (optional but recommended)

### Your First Contribution

1. **Set up your development environment:**
   ```bash
   git clone https://github.com/YuminosukeSato/scigo.git
   cd scigo
   make setup-dev  # Installs all tools and dependencies
   ```

2. **Find an issue to work on:**
   - Look for issues labeled [`good first issue`](https://github.com/YuminosukeSato/scigo/issues?q=label%3A%22good+first+issue%22)
   - Or fix a typo in documentation
   - Or add a missing test

3. **Create your branch:**
   ```bash
   git checkout -b fix/issue-description
   # or
   git checkout -b feature/new-algorithm
   ```

4. **Make your changes and test:**
   ```bash
   make test       # Run tests
   make lint-full  # Check code style
   ```

5. **Submit a Pull Request:**
   - Push your branch to your fork
   - Open a PR with a clear description
   - Wait for review and address feedback

That's it! 🎉

## 📖 Developer's Guide

### Project Philosophy

SciGo aims to provide a high-performance, production-ready machine learning library for Go with scikit-learn compatible APIs. We prioritize:

- **API Compatibility**: Following scikit-learn's proven interface patterns
- **Performance**: Leveraging Go's concurrency and efficiency
- **Reliability**: Comprehensive testing and error handling
- **Simplicity**: Clear, idiomatic Go code

### Project Structure

```
scigo/
├── core/           # Core abstractions and utilities
│   ├── model/      # Base estimator and interfaces
│   ├── tensor/     # Tensor operations
│   └── parallel/   # Parallel processing utilities
├── linear/         # Linear models (regression, classification)
├── preprocessing/  # Data preprocessing (scalers, encoders)
├── metrics/        # Evaluation metrics
├── sklearn/        # Advanced scikit-learn compatible models
├── pkg/           # Shared packages
│   ├── errors/    # Error handling utilities
│   └── log/       # Structured logging
└── examples/      # Usage examples
```

### Coding Standards

#### Go Style

1. **Format your code:**
   ```bash
   make fmt        # or: go fmt ./...
   goimports -w .  # Organize imports
   ```

2. **Follow Go conventions:**
   - Use `camelCase` for unexported identifiers
   - Use `PascalCase` for exported identifiers
   - Keep line length under 100 characters when possible
   - Write clear, concise comments

3. **Run linters:**
   ```bash
   make lint-full  # Runs comprehensive linting
   ```

#### Machine Learning API Conventions

All ML models in SciGo follow the scikit-learn estimator pattern:

```go
// Basic Estimator Pattern
type Estimator interface {
    Fit(X, y mat.Matrix) error
    Predict(X mat.Matrix) (mat.Matrix, error)
    Score(X, y mat.Matrix) (float64, error)
}

// Transformer Pattern
type Transformer interface {
    Fit(X mat.Matrix) error
    Transform(X mat.Matrix) (mat.Matrix, error)
    FitTransform(X mat.Matrix) (mat.Matrix, error)
}
```

**Key Principles:**

1. **State Management**: Use `BaseEstimator` for consistent fitted state tracking
   ```go
   type MyModel struct {
       model.BaseEstimator
       // model-specific fields
   }
   ```

2. **Error Handling**: Use structured errors from `pkg/errors`
   ```go
   if !m.IsFitted() {
       return nil, errors.NewNotFittedError("MyModel", "Predict")
   }
   ```

3. **Logging**: Use structured logging for ML operations
   ```go
   m.LogInfo("Training started",
       log.OperationKey, log.OperationFit,
       log.SamplesKey, nSamples,
   )
   ```

4. **Numerical Precision**: Always use `float64` for numerical computations

5. **Matrix Operations**: Use `gonum.org/v1/gonum/mat` for matrix operations

### Testing Strategy

#### Test Requirements

- **Coverage**: Aim for >80% test coverage for new code
- **Types**: Write unit tests, integration tests, and benchmarks
- **Naming**: Use descriptive test names that explain what is being tested

#### Writing Tests

1. **Unit Tests**: Test individual functions/methods
   ```go
   func TestLinearRegression_Fit(t *testing.T) {
       tests := []struct {
           name    string
           X, y    mat.Matrix
           wantErr bool
       }{
           // test cases
       }
       // test implementation
   }
   ```

2. **Example Tests**: Provide usage examples
   ```go
   func ExampleLinearRegression() {
       // Create and train model
       lr := linear.NewLinearRegression()
       _ = lr.Fit(X, y)
       
       // Output: expected output
   }
   ```

3. **Benchmarks**: Measure performance
   ```go
   func BenchmarkLinearRegression_Fit(b *testing.B) {
       // benchmark implementation
   }
   ```

#### Running Tests

```bash
make test           # Run all tests
make test-short     # Run short tests only
make coverage       # Generate coverage report
make bench          # Run benchmarks
```

### Documentation

#### Code Documentation

Every exported type, function, and method must have a godoc comment:

```go
// LinearRegression implements ordinary least squares regression.
//
// The model minimizes the residual sum of squares between observed
// targets and predictions made by linear approximation.
//
// Example:
//   lr := linear.NewLinearRegression()
//   err := lr.Fit(X, y)
//   predictions, err := lr.Predict(X_test)
type LinearRegression struct {
    // ...
}
```

#### Package Documentation

Each package should have a `doc.go` or package comment explaining:
- Package purpose
- Main types and functions
- Usage examples
- Related packages

### Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass: `make test`
   - Run linters: `make lint-full`
   - Update documentation if needed
   - Add tests for new functionality
   - Update CHANGELOG.md if applicable

2. **PR Description should include:**
   - What problem does this solve?
   - How does it solve it?
   - Any breaking changes?
   - Related issues (use "Fixes #123" to auto-close)

3. **Review process:**
   - CI must pass (tests, linting, coverage)
   - At least one maintainer approval required
   - Address review feedback promptly
   - Squash commits if requested

### Development Workflow

#### Common Tasks

```bash
# Set up development environment
make setup-dev

# Run tests
make test

# Check code coverage
make coverage

# Run linters
make lint-full

# Format code
make fmt

# Run benchmarks
make bench

# Clean build artifacts
make clean

# See all available commands
make help
```

#### Adding a New Algorithm

1. **Create the implementation:**
   ```go
   // mypackage/algorithm.go
   package mypackage
   
   type MyAlgorithm struct {
       model.BaseEstimator
       // fields
   }
   
   func (m *MyAlgorithm) Fit(X, y mat.Matrix) error {
       // implementation
       m.SetFitted()
       return nil
   }
   ```

2. **Add comprehensive tests:**
   ```go
   // mypackage/algorithm_test.go
   func TestMyAlgorithm_Fit(t *testing.T) {
       // test implementation
   }
   ```

3. **Add an example:**
   ```go
   // mypackage/example_test.go
   func ExampleMyAlgorithm() {
       // example usage
   }
   ```

4. **Update documentation:**
   - Add package documentation if new package
   - Update README.md if significant feature

### Error Handling

SciGo uses structured errors for better debugging:

```go
// Use predefined error types
errors.NewNotFittedError("ModelName", "Method")
errors.NewDimensionError("Method", expected, got, axis)
errors.NewValueError("Method", "description")

// Wrap errors with context
fmt.Errorf("failed to train model: %w", err)

// Use panic recovery for public APIs
func (m *MyModel) Fit(X, y mat.Matrix) (err error) {
    defer errors.Recover(&err, "MyModel.Fit")
    // implementation
}
```

### Performance Considerations

1. **Memory Efficiency:**
   - Reuse allocated memory when possible
   - Use in-place operations for large matrices
   - Clear references to allow garbage collection

2. **Parallelization:**
   - Use `core/parallel` utilities for concurrent operations
   - Set appropriate thresholds for parallel vs sequential processing
   - Benchmark to verify performance improvements

3. **Numerical Stability:**
   - Use stable algorithms (e.g., QR decomposition over matrix inversion)
   - Check for numerical edge cases (division by zero, overflow)
   - Use appropriate epsilon values for floating-point comparisons

## 🐛 Reporting Issues

### Before Creating an Issue

1. Check if the issue already exists
2. Try with the latest version
3. Ensure it's not a usage problem (check examples/documentation)

### Creating a Good Issue Report

Include:
- Go version and OS
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces
- Relevant logs (use `log.SetLevel(log.LevelDebug)`)

## 💡 Proposing Features

1. **Check existing issues/PRs** for similar proposals
2. **Open a discussion** for significant changes
3. **Provide use cases** and example API
4. **Consider backward compatibility**

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 📄 License

By contributing to SciGo, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors are recognized in:
- Git history
- CONTRIBUTORS.md file
- Release notes for significant contributions

## 📚 Resources

- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- [Effective Go](https://golang.org/doc/effective_go.html)
- [scikit-learn API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [gonum Documentation](https://godoc.org/gonum.org/v1/gonum)

## ❓ Getting Help

- **Documentation**: Check the [README](README.md) and package documentation
- **Examples**: Look at the [examples](examples/) directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Join GitHub Discussions for questions and ideas

Thank you for contributing to SciGo! 🚀