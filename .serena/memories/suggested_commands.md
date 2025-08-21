# Suggested Commands for SciGo Development

## Essential Commands

### Formatting and Linting
```bash
# Fix import ordering (REQUIRED before commit)
goimports -w -local github.com/YuminosukeSato/scigo .

# Format code
go fmt ./...
gofumpt -w .

# Run linting
golangci-lint run --timeout=5m
go vet ./...
```

### Testing
```bash
# Run all tests
go test -v -race -cover ./...

# Run short tests
go test -v -short ./...

# Run with coverage
go test -v -race -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

# Run benchmarks
go test -bench=. -benchmem ./...
```

### Make Commands
```bash
# Run format, lint, and test
make all

# Set up development environment
make setup-dev

# Run tests
make test
make test-short
make coverage

# Format code
make fmt

# Run linting
make lint
make lint-full

# Fast CI checks locally
make ci-fast
```

### Module Management
```bash
# Tidy dependencies
go mod tidy

# Download dependencies
go mod download
```

### Git Operations
```bash
# Check status
git status

# View diffs
git diff

# Commit (after running goimports!)
git add .
git commit -m "message"
```

## CI/CD Related
The CI workflow checks:
1. gofumpt formatting
2. goimports formatting with local flag
3. golangci-lint
4. go mod tidy
5. Tests with coverage

Always run these checks before pushing:
```bash
# Quick check
goimports -l -local github.com/YuminosukeSato/scigo .
gofumpt -l .
go mod tidy
go test -short ./...
```