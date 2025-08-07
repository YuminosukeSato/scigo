# SciGo Quick Start Docker Image
# Provides a 30-second ML experience with SciGo

# Build stage
FROM golang:1.23-alpine AS builder

LABEL maintainer="Yuminosuke Sato <yuminosuke.sato@example.com>"
LABEL description="SciGo - The blazing-fast ML library for Go"
LABEL version="0.2.0"

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the quick start example
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o quick-start ./examples/quick-start

# Build additional examples
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o linear-regression ./examples/linear_regression
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o iris-regression ./examples/iris_regression

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN adduser -D -s /bin/sh -u 1000 scigouser

# Set working directory
WORKDIR /home/scigouser

# Copy built binaries from builder stage
COPY --from=builder /app/quick-start /usr/local/bin/
COPY --from=builder /app/linear-regression /usr/local/bin/
COPY --from=builder /app/iris-regression /usr/local/bin/

# Copy example data and scripts
COPY --from=builder /app/datasets/ ./datasets/
COPY --from=builder /app/examples/ ./examples/

# Create demo script
RUN cat > /usr/local/bin/scigo-demo << 'EOF'
#!/bin/sh
set -e

echo "🚀 Welcome to SciGo Demo!"
echo "========================="
echo ""
echo "Available demos:"
echo "1. Quick Start    - Basic linear regression (30 seconds)"
echo "2. Linear Model   - Detailed linear regression example"
echo "3. Iris Dataset   - Classic ML dataset demonstration"
echo ""

case "${1:-1}" in
    1|quick|quick-start)
        echo "🎯 Running Quick Start Demo..."
        echo "⏱️  This should complete in ~30 seconds"
        echo ""
        time quick-start
        ;;
    2|linear|linear-regression)
        echo "🎯 Running Linear Regression Demo..."
        echo ""
        time linear-regression
        ;;
    3|iris|iris-regression)
        echo "🎯 Running Iris Dataset Demo..."
        echo ""
        time iris-regression
        ;;
    all)
        echo "🎯 Running All Demos..."
        echo ""
        echo "=== Quick Start ==="
        time quick-start
        echo ""
        echo "=== Linear Regression ==="
        time linear-regression
        echo ""
        echo "=== Iris Dataset ==="
        time iris-regression
        ;;
    *)
        echo "Usage: scigo-demo [1|quick|linear|iris|all]"
        echo ""
        echo "Examples:"
        echo "  scigo-demo 1        # Quick start (default)"
        echo "  scigo-demo quick    # Quick start"
        echo "  scigo-demo linear   # Linear regression"
        echo "  scigo-demo iris     # Iris dataset"
        echo "  scigo-demo all      # All demos"
        exit 1
        ;;
esac

echo ""
echo "🎉 Demo completed!"
echo "📚 Learn more: https://pkg.go.dev/github.com/YuminosukeSato/scigo"
echo "💻 Source: https://github.com/YuminosukeSato/scigo"
echo ""
echo "Ready, Set, SciGo! 🚀"
EOF

# Make demo script executable
RUN chmod +x /usr/local/bin/scigo-demo

# Switch to non-root user
USER scigouser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD quick-start > /dev/null 2>&1 || exit 1

# Set default command
CMD ["scigo-demo"]

# Add labels for better container management
LABEL org.opencontainers.image.title="SciGo"
LABEL org.opencontainers.image.description="The blazing-fast scikit-learn compatible ML library for Go"
LABEL org.opencontainers.image.version="0.2.0"
LABEL org.opencontainers.image.source="https://github.com/YuminosukeSato/scigo"
LABEL org.opencontainers.image.documentation="https://pkg.go.dev/github.com/YuminosukeSato/scigo"
LABEL org.opencontainers.image.vendor="Yuminosuke Sato"
LABEL org.opencontainers.image.licenses="MIT"

# Expose no ports (this is a demo/CLI container)
# EXPOSE 8080

# Add usage instructions as environment variables
ENV SCIGO_VERSION="0.2.0"
ENV SCIGO_USAGE="Run 'scigo-demo' to see available demonstrations"
ENV SCIGO_DOCS="https://pkg.go.dev/github.com/YuminosukeSato/scigo"
ENV SCIGO_SOURCE="https://github.com/YuminosukeSato/scigo"