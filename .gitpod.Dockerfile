# GitPod Dockerfile for SciGo development environment

FROM gitpod/workspace-go:latest

USER root

# Install additional system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML packages for comparison
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    lightgbm \
    matplotlib \
    jupyter \
    seaborn

# Install Hugo for documentation
RUN wget https://github.com/gohugoio/hugo/releases/download/v0.120.4/hugo_extended_0.120.4_Linux-64bit.tar.gz && \
    tar -xzf hugo_extended_0.120.4_Linux-64bit.tar.gz && \
    mv hugo /usr/local/bin/ && \
    rm hugo_extended_0.120.4_Linux-64bit.tar.gz

# Install additional Go tools
RUN go install golang.org/x/tools/cmd/goimports@latest && \
    go install golang.org/x/tools/cmd/godoc@latest && \
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest && \
    go install golang.org/x/vuln/cmd/govulncheck@latest

USER gitpod

# Set up Go workspace
ENV GOPATH=/workspace/go
ENV PATH=$GOPATH/bin:$PATH

# Create demo script for easy access
RUN echo '#!/bin/bash\n\
echo "🚀 SciGo GitPod Demo Commands"\n\
echo "=============================="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  demo-quick        # Quick start demo (30 seconds)"\n\
echo "  demo-linear       # Linear regression example"\n\
echo "  demo-iris         # Iris dataset example"\n\
echo "  demo-all          # Run all demos"\n\
echo "  test-all          # Run all tests"\n\
echo "  benchmark         # Run performance benchmarks"\n\
echo "  docs-serve        # Start documentation server"\n\
echo "  compare-python    # Performance comparison with Python"\n\
echo ""\n\
case "$1" in\n\
  quick|demo-quick)\n\
    echo "🎯 Running Quick Start Demo..."\n\
    go run ./examples/quick-start\n\
    ;;\n\
  linear|demo-linear)\n\
    echo "🎯 Running Linear Regression Demo..."\n\
    go run ./examples/linear_regression\n\
    ;;\n\
  iris|demo-iris)\n\
    echo "🎯 Running Iris Dataset Demo..."\n\
    go run ./examples/iris_regression\n\
    ;;\n\
  all|demo-all)\n\
    echo "🎯 Running All Demos..."\n\
    go run ./examples/quick-start\n\
    echo ""\n\
    go run ./examples/linear_regression\n\
    echo ""\n\
    go run ./examples/iris_regression\n\
    ;;\n\
  test|test-all)\n\
    echo "🧪 Running Tests..."\n\
    go test -v ./...\n\
    ;;\n\
  bench|benchmark)\n\
    echo "⚡ Running Benchmarks..."\n\
    go test -bench=. -benchmem ./...\n\
    ;;\n\
  docs|docs-serve)\n\
    echo "📚 Starting Documentation Server..."\n\
    cd docs/hugo-site && hugo server --bind 0.0.0.0\n\
    ;;\n\
  python|compare-python)\n\
    echo "🐍 Running Python Comparison..."\n\
    cd examples/sklearn && python3 comparison.py\n\
    ;;\n\
  *)\n\
    echo "Usage: scigo-demo [quick|linear|iris|all|test|bench|docs|python]"\n\
    ;;\n\
esac' > /home/gitpod/.local/bin/scigo-demo

RUN chmod +x /home/gitpod/.local/bin/scigo-demo

# Add aliases for common commands
RUN echo 'alias scigo="scigo-demo"' >> /home/gitpod/.bashrc && \
    echo 'alias demo="scigo-demo"' >> /home/gitpod/.bashrc && \
    echo 'alias quick="scigo-demo quick"' >> /home/gitpod/.bashrc

# Set environment variables
ENV SCIGO_WORKSPACE=gitpod
ENV SCIGO_VERSION=0.2.0