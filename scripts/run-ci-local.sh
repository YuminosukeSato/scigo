#!/usr/bin/env bash

# SciGo Local CI Runner
# This script runs all CI checks locally, mimicking GitHub Actions

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SciGo Local CI Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track failures
FAILED_CHECKS=()
TOTAL_CHECKS=12
CURRENT_CHECK=0

# Function to run a check
run_check() {
    local check_name="$1"
    local check_command="$2"
    
    CURRENT_CHECK=$((CURRENT_CHECK + 1))
    echo -e "${GREEN}[$CURRENT_CHECK/$TOTAL_CHECKS] Running: $check_name${NC}"
    
    if eval "$check_command"; then
        echo -e "${GREEN}✓ $check_name passed${NC}"
    else
        echo -e "${RED}✗ $check_name failed${NC}"
        FAILED_CHECKS+=("$check_name")
    fi
    echo ""
}

# Check for required tools
echo -e "${BLUE}Checking for required tools...${NC}"

check_tool() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $1 not found${NC}"
        return 1
    fi
}

MISSING_TOOLS=()

# Check Go tools
check_tool "go" || MISSING_TOOLS+=("go")
check_tool "staticcheck" || MISSING_TOOLS+=("staticcheck")
check_tool "govulncheck" || MISSING_TOOLS+=("govulncheck")
check_tool "gosec" || MISSING_TOOLS+=("gosec")

# Check optional tools (warn but don't fail)
echo -e "${BLUE}Checking optional tools...${NC}"
check_tool "nancy" || echo "  Install with: go install github.com/sonatype-nexus-community/nancy@latest"
check_tool "gitleaks" || echo "  Install with: brew install gitleaks"
check_tool "trivy" || echo "  Install with: brew install trivy"
check_tool "trufflehog" || echo "  Install with: brew install trufflehog"
check_tool "semgrep" || echo "  Install with: python3 -m pip install semgrep"

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${RED}Missing required tools: ${MISSING_TOOLS[*]}${NC}"
    echo -e "${YELLOW}Run 'make install-ci-tools' to install all CI tools${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting CI checks...${NC}"
echo ""

# 1. Go vet
run_check "go vet" "go vet ./..."

# 2. Staticcheck
run_check "staticcheck" "staticcheck ./..."

# 3. Govulncheck
run_check "govulncheck" "govulncheck ./..."

# 4. Gosec
run_check "gosec" "gosec -quiet -fmt text ./..."

# 5. Nancy (if available)
if command -v nancy &> /dev/null; then
    run_check "nancy dependency scan" "go list -json -deps ./... 2>/dev/null | nancy sleuth"
else
    echo -e "${YELLOW}[$((++CURRENT_CHECK))/$TOTAL_CHECKS] Skipping nancy (not installed)${NC}"
fi

# 6. Gitleaks (if available)
if command -v gitleaks &> /dev/null; then
    run_check "gitleaks secret scan" "gitleaks detect --source . --verbose"
else
    echo -e "${YELLOW}[$((++CURRENT_CHECK))/$TOTAL_CHECKS] Skipping gitleaks (not installed)${NC}"
fi

# 7. Trivy (if available)
if command -v trivy &> /dev/null; then
    run_check "trivy vulnerability scan" "trivy fs --severity HIGH,CRITICAL ."
else
    echo -e "${YELLOW}[$((++CURRENT_CHECK))/$TOTAL_CHECKS] Skipping trivy (not installed)${NC}"
fi

# 8. Trufflehog (if available)
if command -v trufflehog &> /dev/null; then
    run_check "trufflehog credential scan" "trufflehog filesystem . --no-update"
else
    echo -e "${YELLOW}[$((++CURRENT_CHECK))/$TOTAL_CHECKS] Skipping trufflehog (not installed)${NC}"
fi

# 9. Go test
run_check "go test" "go test -v -race -cover ./..."

# 10. Go test with race detector (already included above, so just a quick test)
run_check "race condition check" "go test -race ./..."

# 11. Go mod tidy check
echo -e "${GREEN}[11/$TOTAL_CHECKS] Checking go mod tidy${NC}"
cp go.mod go.mod.bak
cp go.sum go.sum.bak
go mod tidy
if diff go.mod go.mod.bak > /dev/null && diff go.sum go.sum.bak > /dev/null; then
    echo -e "${GREEN}✓ go.mod and go.sum are tidy${NC}"
    rm go.mod.bak go.sum.bak
else
    echo -e "${RED}✗ go.mod or go.sum need tidying${NC}"
    FAILED_CHECKS+=("go mod tidy")
    mv go.mod.bak go.mod
    mv go.sum.bak go.sum
fi
echo ""

# 12. Semgrep (if available)
if command -v semgrep &> /dev/null; then
    run_check "semgrep security analysis" "semgrep --config=auto --error --verbose ."
else
    echo -e "${YELLOW}[12/$TOTAL_CHECKS] Skipping semgrep (not installed)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   CI Check Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All CI checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ The following checks failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  - $check${NC}"
    done
    echo ""
    echo -e "${YELLOW}Please fix the issues before pushing to GitHub${NC}"
    exit 1
fi