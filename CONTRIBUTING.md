# Contributing to GoML

Thank you for your interest in contributing to GoML! We welcome contributions from the community.

## How to Contribute

### 1. Fork the Repository
- Fork the repository on GitHub
- Clone your fork locally

```bash
git clone https://github.com/YOUR_USERNAME/GoML.git
cd GoML
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Write clean, idiomatic Go code
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed

### 4. Run Tests
```bash
# Run all tests
go test ./...

# Run with race detector
go test -race ./...

# Check coverage
go test -cover ./...
```

### 5. Submit a Pull Request
- Push your changes to your fork
- Create a pull request against the main branch
- Describe your changes clearly in the PR description

## Code Style Guidelines

- Follow standard Go formatting (use `gofmt`)
- Use meaningful variable and function names
- Add comments for exported functions and types
- Keep functions focused and small
- Use table-driven tests where appropriate

## Testing Guidelines

- Aim for >80% test coverage for new code
- Include both positive and negative test cases
- Add benchmarks for performance-critical code
- Use subtests for better test organization

## Documentation

- Update README.md if adding new features
- Add godoc comments for all exported types and functions
- Include usage examples in documentation
- Keep documentation clear and concise

## Commit Message Format

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `perf:` Performance improvement
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Areas for Contribution

### High Priority
- [ ] More ML algorithms (Decision Trees, SVM, Neural Networks)
- [ ] Model serialization/deserialization
- [ ] Distributed training support
- [ ] GPU acceleration support

### Good First Issues
- [ ] Improve test coverage
- [ ] Add more examples
- [ ] Enhance documentation
- [ ] Add more evaluation metrics

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in existing issues
- Ask questions in the Discussions tab

## License

By contributing, you agree that your contributions will be licensed under the MIT License.