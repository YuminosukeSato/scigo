# Installation

This guide covers the installation of SciGo and setting up your development environment.

## Requirements

- **Go**: Version 1.21 or later
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM recommended for large datasets

## Installation Methods

### Using Go Modules (Recommended)

Add SciGo to your project using Go modules:

```bash
go get github.com/YuminosukeSato/scigo@latest
```

This will add SciGo to your `go.mod` file and download all dependencies.

### Installing from Source

Clone the repository and build from source:

```bash
git clone https://github.com/YuminosukeSato/scigo.git
cd scigo
go build ./...
```

## Verifying Installation

Create a simple test file to verify the installation:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/linear"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create a simple linear regression model
    lr := linear.NewLinearRegression()
    
    // Sample data
    X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
    y := mat.NewVecDense(4, []float64{2, 4, 6, 8})
    
    // Train the model
    if err := lr.Fit(X, y); err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("SciGo is successfully installed!")
}
```

Run the test:

```bash
go run test.go
```

## Dependencies

SciGo relies on several high-quality Go packages:

| Package | Purpose | Version |
|---------|---------|---------|
| `gonum.org/v1/gonum` | Numerical computing | v0.14.0+ |
| `golang.org/x/exp` | Experimental packages | Latest |
| `github.com/goccy/go-json` | Fast JSON processing | v0.10.0+ |

All dependencies are automatically managed through Go modules.

## Platform-Specific Notes

### macOS

On macOS, especially Apple Silicon (M1/M2), SciGo can leverage the ARM64 architecture for improved performance:

```bash
# Build with optimizations for Apple Silicon
GOARCH=arm64 go build -tags accelerate ./...
```

### Linux

For optimal performance on Linux, ensure you have BLAS libraries installed:

```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# Fedora/RHEL
sudo dnf install blas-devel lapack-devel
```

### Windows

On Windows, use PowerShell or Command Prompt:

```powershell
go get github.com/YuminosukeSato/scigo@latest
```

## Docker Installation

Use the official Docker image:

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o scigo-app ./cmd/main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/scigo-app .
CMD ["./scigo-app"]
```

Build and run:

```bash
docker build -t scigo-app .
docker run scigo-app
```

## Environment Variables

SciGo supports several environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `SCIGO_NUM_THREADS` | Number of parallel threads | CPU cores |
| `SCIGO_LOG_LEVEL` | Logging level (DEBUG, INFO, WARN, ERROR) | INFO |
| `SCIGO_CACHE_DIR` | Directory for cached models | `~/.scigo/cache` |

Example:

```bash
export SCIGO_NUM_THREADS=8
export SCIGO_LOG_LEVEL=DEBUG
```

## IDE Setup

### Visual Studio Code

Install the Go extension and add these settings to `.vscode/settings.json`:

```json
{
    "go.useLanguageServer": true,
    "go.lintTool": "golangci-lint",
    "go.lintFlags": [
        "--fast"
    ],
    "go.testFlags": ["-v"],
    "go.testTimeout": "30s"
}
```

### GoLand/IntelliJ IDEA

1. Open the project
2. Go to **File → Settings → Go → Go Modules**
3. Enable Go modules integration
4. Click **Download Go Modules**

## Troubleshooting

### Common Issues

**Issue**: `go get` fails with permission denied
```bash
# Solution: Clear the module cache
go clean -modcache
go get github.com/YuminosukeSato/scigo@latest
```

**Issue**: Import cycle errors
```bash
# Solution: Update to the latest version
go get -u github.com/YuminosukeSato/scigo@latest
```

**Issue**: Memory errors with large datasets
```bash
# Solution: Increase Go heap size
export GOGC=200  # Increase garbage collection threshold
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md)
2. Search [existing issues](https://github.com/YuminosukeSato/scigo/issues)
3. Create a [new issue](https://github.com/YuminosukeSato/scigo/issues/new) with:
   - Go version (`go version`)
   - SciGo version
   - Complete error message
   - Minimal reproducible example

## Next Steps

- Continue to [Quick Start](./quick-start.md) to build your first model
- Explore [Basic Concepts](./basic-concepts.md) to understand the fundamentals
- Browse [Examples](../../examples/) for real-world use cases