# Task Completion Checklist

When completing any coding task in SciGo, always run these commands:

## 1. Format Code
```bash
# Fix import ordering (CRITICAL for CI)
goimports -w -local github.com/YuminosukeSato/scigo .

# Format Go code
gofumpt -w .
```

## 2. Run Tests
```bash
# Run tests to ensure nothing is broken
go test -short ./...
```

## 3. Check Linting
```bash
# Run linter
golangci-lint run --timeout=5m
```

## 4. Verify Dependencies
```bash
# Ensure go.mod is tidy
go mod tidy
```

## 5. Final Verification
```bash
# Check that imports are correctly ordered
goimports -l -local github.com/YuminosukeSato/scigo .
# Should return nothing if all files are properly formatted
```

## Common CI Failures
1. **Import ordering**: Third-party imports (gonum) must come before local imports
2. **go mod tidy**: Dependencies must be properly managed
3. **Formatting**: Code must pass gofumpt and goimports checks
4. **Tests**: All tests must pass