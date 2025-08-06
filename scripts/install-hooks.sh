#!/usr/bin/env bash
set -euo pipefail

# Script to install pre-commit hooks for SciGo project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "🚀 Installing pre-commit hooks for SciGo..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "❌ pre-commit is not installed. Installing..."
    
    # Try to install with pip
    if command -v pip3 &> /dev/null; then
        pip3 install pre-commit
    elif command -v pip &> /dev/null; then
        pip install pre-commit
    elif command -v brew &> /dev/null; then
        brew install pre-commit
    else
        echo "❌ Could not install pre-commit. Please install it manually:"
        echo "   pip install pre-commit"
        echo "   or"
        echo "   brew install pre-commit"
        exit 1
    fi
fi

# Navigate to project root
cd "${PROJECT_ROOT}"

# Install the git hook scripts
echo "📦 Installing git hook scripts..."
pre-commit install

# Install commit-msg hooks for commit message linting
echo "📝 Installing commit message hooks..."
pre-commit install --hook-type commit-msg

# Install pre-push hooks
echo "🔄 Installing pre-push hooks..."
pre-commit install --hook-type pre-push

# Run all hooks on all files for the first time
echo "🔍 Running all hooks on all files (this may take a moment)..."
pre-commit run --all-files || true

# Install additional Go tools if needed
echo "🛠️ Checking Go tools..."

# Install golangci-lint if not present
if ! command -v golangci-lint &> /dev/null; then
    echo "Installing golangci-lint..."
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
fi

# Install goimports if not present
if ! command -v goimports &> /dev/null; then
    echo "Installing goimports..."
    go install golang.org/x/tools/cmd/goimports@latest
fi

# Install gosec if not present
if ! command -v gosec &> /dev/null; then
    echo "Installing gosec..."
    go install github.com/securego/gosec/v2/cmd/gosec@latest
fi

# Install gocyclo if not present
if ! command -v gocyclo &> /dev/null; then
    echo "Installing gocyclo..."
    go install github.com/fzipp/gocyclo/cmd/gocyclo@latest
fi

echo "✅ Pre-commit hooks installation complete!"
echo ""
echo "📋 Installed hooks:"
echo "  - Go formatters (fmt, vet, imports)"
echo "  - Go linters (golangci-lint, cyclo)"
echo "  - Security scanners (gosec, detect-secrets)"
echo "  - File checks (trailing whitespace, large files, etc.)"
echo "  - Markdown linting"
echo "  - Commit message linting"
echo ""
echo "💡 Usage:"
echo "  - Hooks will run automatically on git commit"
echo "  - To run manually: pre-commit run --all-files"
echo "  - To update hooks: pre-commit autoupdate"
echo "  - To skip hooks: git commit --no-verify"
echo ""
echo "📚 For more information, see:"
echo "  - https://pre-commit.com/"
echo "  - .pre-commit-config.yaml for hook configuration"