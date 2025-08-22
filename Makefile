# SciGo Makefile
SHELL := /bin/bash
.PHONY: all build test lint fmt clean install-hooks run-hooks help

# Variables
GO := go
GOLANGCI_LINT := golangci-lint
GOTEST := $(GO) test
GOBUILD := $(GO) build
GOFMT := gofmt
GOVET := $(GO) vet
GOMOD := $(GO) mod

# Build variables
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

all: fmt lint-full test ## Run all: format, full lint, and test

##@ Development

setup-dev: ## Set up development environment
	@echo -e "$(GREEN)Setting up development environment...$(NC)"
	@echo -e "$(GREEN)Installing Go development tools...$(NC)"
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install golang.org/x/tools/cmd/goimports@latest
	@go install github.com/securego/gosec/v2/cmd/gosec@latest
	@go install github.com/fzipp/gocyclo/cmd/gocyclo@latest
	@echo -e "$(GREEN)Installing pre-commit hooks...$(NC)"
	@./scripts/install-hooks.sh
	@echo -e "$(GREEN)Downloading dependencies...$(NC)"
	@$(GOMOD) download
	@$(GOMOD) tidy
	@echo -e "$(GREEN)✅ Development environment ready!$(NC)"
	@echo -e "$(YELLOW)Run 'make help' to see available commands$(NC)"

test: ## Run tests
	@echo -e "$(GREEN)Running tests...$(NC)"
	$(GOTEST) -v -race -cover ./...

test-short: ## Run short tests
	@echo -e "$(GREEN)Running short tests...$(NC)"
	$(GOTEST) -v -short ./...

test-ci: ## Run tests with CI options (-v -short -race -coverprofile)
	@echo -e "$(GREEN)Running CI tests...$(NC)"
	$(GOTEST) -v -short -race -coverprofile=coverage.out ./...

coverage: ## Run tests with coverage report
	@echo -e "$(GREEN)Running tests with coverage...$(NC)"
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo -e "$(GREEN)Coverage report generated: coverage.html$(NC)"

coverage-text: ## Run tests with text coverage report
	@echo -e "$(GREEN)Running tests with text coverage...$(NC)"
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out

coverage-ci: ## Run coverage for CI (with threshold check)
	@echo -e "$(GREEN)Running coverage for CI...$(NC)"
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	@coverage=$$($(GO) tool cover -func=coverage.out | grep total | awk '{print substr($$3, 1, length($$3)-1)}'); \
	echo "Total coverage: $$coverage%"; \
	threshold=70; \
	coverage_int=$$(echo $$coverage | cut -d'.' -f1); \
	if [ $$coverage_int -lt $$threshold ]; then \
		echo -e "$(RED)Coverage $$coverage% is below $$threshold% threshold$(NC)"; \
		exit 1; \
	else \
		echo -e "$(GREEN)Coverage $$coverage% meets $$threshold% threshold$(NC)"; \
	fi

bench: ## Run benchmarks
	@echo -e "$(GREEN)Running benchmarks...$(NC)"
	$(GOTEST) -bench=. -benchmem ./...

##@ Code Quality

# Install formatters for CI
install-formatters: ## Install gofumpt and goimports
	@echo -e "$(GREEN)Installing formatters...$(NC)"
	@go install mvdan.cc/gofumpt@latest
	@go install golang.org/x/tools/cmd/goimports@latest
	@echo -e "$(GREEN)Formatters installed$(NC)"

fmt: ## Format code
	@echo -e "$(GREEN)Formatting code...$(NC)"
	$(GO) fmt ./...

fmt-ci: install-formatters ## Format code with gofumpt and goimports (CI-compatible)
	@echo -e "$(GREEN)Formatting code with gofumpt and goimports...$(NC)"
	@gofumpt -w .
	@goimports -w -local github.com/YuminosukeSato/scigo .

fmt-check: ## Check if code is formatted
	@echo -e "$(GREEN)Checking code formatting...$(NC)"
	@if [ -n "$$(gofmt -l .)" ]; then \
		echo -e "$(RED)The following files need formatting:$(NC)"; \
		gofmt -l .; \
		echo -e "$(YELLOW)Run 'make fmt' or 'go fmt ./...' to fix$(NC)"; \
		exit 1; \
	else \
		echo -e "$(GREEN)All files are properly formatted$(NC)"; \
	fi

check-fmt: install-formatters ## Check formatting with gofumpt and goimports (CI-compatible)
	@echo -e "$(GREEN)Checking gofumpt formatting...$(NC)"
	@gofumpt -l -d .
	@if [ -n "$$(gofumpt -l .)" ]; then \
		echo -e "$(RED)❌ gofumpt formatting issues found. Run 'make fmt-ci' locally$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)Checking goimports formatting...$(NC)"
	@goimports -l -d -local github.com/YuminosukeSato/scigo .
	@if [ -n "$$(goimports -l -local github.com/YuminosukeSato/scigo .)" ]; then \
		echo -e "$(RED)❌ goimports formatting issues found. Run 'make fmt-ci' locally$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✅ All formatting checks passed$(NC)"

lint: ## Run go vet
	@echo -e "$(GREEN)Running go vet...$(NC)"
	$(GOVET) ./...

lint-full: ## Run golangci-lint with enhanced checks
	@echo -e "$(GREEN)Running golangci-lint...$(NC)"
	@if command -v golangci-lint &> /dev/null; then \
		$(GOLANGCI_LINT) run --timeout=5m --enable=errcheck,govet,ineffassign,staticcheck,unused,misspell ./...; \
	else \
		echo -e "$(YELLOW)golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(NC)"; \
	fi

lint-check: ## Check linting issues  
	@echo -e "$(GREEN)Checking code with golangci-lint...$(NC)"
	@if command -v golangci-lint &> /dev/null; then \
		$(GOLANGCI_LINT) run --timeout=5m --enable=errcheck,govet,ineffassign,staticcheck,unused ./...; \
	else \
		echo -e "$(YELLOW)golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(NC)"; \
	fi

lint-ci: ## Run golangci-lint on changed files only (CI-compatible)
	@echo -e "$(GREEN)Running golangci-lint on changed files...$(NC)"
	@if command -v golangci-lint &> /dev/null; then \
		if git diff --quiet HEAD~1 2>/dev/null; then \
			echo -e "$(YELLOW)No changes detected from HEAD~1, running on all files$(NC)"; \
			$(GOLANGCI_LINT) run --timeout=5m ./...; \
		else \
			$(GOLANGCI_LINT) run --timeout=5m --new-from-rev=HEAD~1 ./...; \
		fi \
	else \
		echo -e "$(YELLOW)golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(NC)"; \
		exit 1; \
	fi

##@ CI Local Execution

# Main CI target that mirrors GitHub Actions workflow exactly
ci: ## Run complete CI workflow (mirrors GitHub Actions)
	@echo -e "$(GREEN)========================================$(NC)"
	@echo -e "$(GREEN)Running CI Workflow (GitHub Actions Compatible)$(NC)"
	@echo -e "$(GREEN)========================================$(NC)"
	@echo ""
	@echo -e "$(GREEN)[Step 1/4] Checking code formatting...$(NC)"
	@$(MAKE) check-fmt
	@echo ""
	@echo -e "$(GREEN)[Step 2/4] Running golangci-lint...$(NC)"
	@$(MAKE) lint-ci
	@echo ""
	@echo -e "$(GREEN)[Step 3/4] Checking go mod tidy...$(NC)"
	@$(MAKE) check-mod-tidy
	@echo ""
	@echo -e "$(GREEN)[Step 4/4] Running tests...$(NC)"
	@$(MAKE) test-ci
	@echo ""
	@echo -e "$(GREEN)========================================$(NC)"
	@echo -e "$(GREEN)✅ All CI checks passed successfully!$(NC)"
	@echo -e "$(GREEN)========================================$(NC)"

ci-fast: ## Fast CI checks (minimal security, parallel execution) - 5-7 seconds
	@echo -e "$(GREEN)Running fast CI checks...$(NC)"
	@echo -e "$(GREEN)[1/5] Code formatting and vetting...$(NC)"
	@$(MAKE) fmt-check &
	@$(MAKE) vet &
	wait
	@echo -e "$(GREEN)[2/5] Running staticcheck...$(NC)"
	@if command -v staticcheck &> /dev/null; then \
		staticcheck ./...; \
	else \
		echo -e "$(YELLOW)staticcheck not installed, skipping$(NC)"; \
	fi
	@echo -e "$(GREEN)[3/5] Quick security scan...$(NC)"
	@if command -v govulncheck &> /dev/null; then \
		govulncheck ./... &\
	fi
	@if command -v gosec &> /dev/null; then \
		gosec -severity high -quiet ./... &\
	fi
	wait
	@echo -e "$(GREEN)[4/5] Running tests (parallel)...$(NC)"
	@$(GOTEST) -short -parallel 4 ./...
	@echo -e "$(GREEN)[5/5] Checking go mod tidy...$(NC)"
	@$(MAKE) check-mod-tidy
	@echo -e "$(GREEN)✅ Fast CI completed!$(NC)"

ci-docker: ## Run CI in Docker container (fastest with pre-built image)
	@echo -e "$(GREEN)Running CI in Docker container...$(NC)"
	@if docker images | grep -q "scigo-ci"; then \
		echo -e "$(GREEN)Using cached Docker image$(NC)"; \
	else \
		echo -e "$(YELLOW)Building Docker image (first run only)...$(NC)"; \
		docker build -f Dockerfile.ci -t scigo-ci:latest .; \
	fi
	@docker run --rm -v $(PWD):/workspace -w /workspace scigo-ci:latest -c "make ci-fast"

ci-parallel: ## Run CI checks in parallel (experimental)
	@echo -e "$(GREEN)Running CI checks in parallel...$(NC)"
	@$(MAKE) -j4 fmt-check vet static-analysis test-short

ci-local: ## Run all CI checks locally (equivalent to GitHub Actions)
	@echo -e "$(GREEN)Running complete CI checks locally...$(NC)"
	@echo -e "$(GREEN)[1/13] Auto-formatting code...$(NC)"
	@$(MAKE) fmt
	@echo -e "$(GREEN)[2/13] Running go vet...$(NC)"
	@$(MAKE) lint
	@echo -e "$(GREEN)[3/13] Running staticcheck...$(NC)"
	@$(MAKE) static-analysis
	@echo -e "$(GREEN)[4/13] Running govulncheck...$(NC)"
	@$(MAKE) vuln-check
	@echo -e "$(GREEN)[5/13] Running gosec...$(NC)"
	@$(MAKE) gosec-scan
	@echo -e "$(GREEN)[6/13] Running nancy dependency scanner...$(NC)"
	@$(MAKE) nancy-scan
	@echo -e "$(GREEN)[7/13] Running gitleaks secret scanner...$(NC)"
	@$(MAKE) gitleaks-scan
	@echo -e "$(GREEN)[8/13] Running trivy vulnerability scanner...$(NC)"
	@$(MAKE) trivy-scan
	@echo -e "$(GREEN)[9/13] Running trufflehog credential scanner...$(NC)"
	@$(MAKE) trufflehog-scan
	@echo -e "$(GREEN)[10/13] Running tests...$(NC)"
	@$(MAKE) test
	@echo -e "$(GREEN)[11/13] Running tests with race detector...$(NC)"
	@$(GOTEST) -race ./...
	@echo -e "$(GREEN)[12/13] Checking go mod tidy...$(NC)"
	@$(MAKE) check-mod-tidy
	@echo -e "$(GREEN)[13/13] Running semgrep analysis (if available)...$(NC)"
	@$(MAKE) semgrep-scan
	@echo -e "$(GREEN)✅ All CI checks completed successfully!$(NC)"

install-ci-tools: ## Install all CI tools (Go tools + system tools)
	@echo -e "$(GREEN)Installing all CI tools...$(NC)"
	@echo -e "$(GREEN)Installing Go-based tools...$(NC)"
	go install golang.org/x/vuln/cmd/govulncheck@latest
	go install github.com/securego/gosec/v2/cmd/gosec@latest
	go install honnef.co/go/tools/cmd/staticcheck@latest
	go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@latest
	go install github.com/sonatype-nexus-community/nancy@latest
	@echo -e "$(GREEN)Checking for Homebrew...$(NC)"
	@if ! command -v brew &> /dev/null; then \
		echo -e "$(YELLOW)Homebrew not installed. Some tools require manual installation.$(NC)"; \
		echo -e "$(YELLOW)Visit https://brew.sh to install Homebrew$(NC)"; \
	else \
		echo -e "$(GREEN)Installing system tools via Homebrew...$(NC)"; \
		brew install gitleaks || echo "gitleaks installation failed"; \
		brew install trivy || echo "trivy installation failed"; \
		brew install trufflehog || echo "trufflehog installation failed"; \
	fi
	@echo -e "$(GREEN)Checking for semgrep...$(NC)"
	@if command -v python3 &> /dev/null; then \
		python3 -m pip install semgrep --user || echo "semgrep installation failed"; \
	else \
		echo -e "$(YELLOW)Python3 not found. Semgrep requires Python3.$(NC)"; \
	fi
	@echo -e "$(GREEN)✅ CI tools installation completed!$(NC)"
	@echo -e "$(YELLOW)Note: Some tools may require adding to PATH or system restart$(NC)"

##@ Security Scanning

install-security-tools: ## Install security scanning tools
	@echo -e "$(GREEN)Installing security tools...$(NC)"
	go install golang.org/x/vuln/cmd/govulncheck@latest
	go install github.com/securego/gosec/v2/cmd/gosec@latest
	go install honnef.co/go/tools/cmd/staticcheck@latest
	go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@latest
	go install github.com/sonatype-nexus-community/nancy@latest
	@echo -e "$(GREEN)Core security tools installed successfully$(NC)"

security-scan: ## Run complete security scan
	@echo -e "$(GREEN)Running comprehensive security scan...$(NC)"
	@$(MAKE) vuln-check
	@$(MAKE) gosec-scan
	@$(MAKE) static-analysis
	@$(MAKE) dependency-check
	@$(MAKE) generate-sbom
	@echo -e "$(GREEN)Security scan completed$(NC)"

vuln-check: ## Check for known vulnerabilities
	@echo -e "$(GREEN)Checking for known vulnerabilities with govulncheck...$(NC)"
	@if command -v govulncheck &> /dev/null; then \
		govulncheck ./... || (echo -e "$(RED)Vulnerabilities found!$(NC)" && exit 1); \
		echo -e "$(GREEN)No known vulnerabilities detected$(NC)"; \
	else \
		echo -e "$(YELLOW)govulncheck not installed. Run 'make install-security-tools'$(NC)"; \
	fi

gosec-scan: ## Run gosec security analyzer
	@echo -e "$(GREEN)Running gosec security analysis...$(NC)"
	@mkdir -p security
	@if command -v gosec &> /dev/null; then \
		gosec -fmt json -out security/gosec-report.json -stdout -verbose=text -severity medium ./...; \
		echo -e "$(GREEN)Gosec scan completed. Report: security/gosec-report.json$(NC)"; \
	else \
		echo -e "$(YELLOW)gosec not installed. Run 'make install-security-tools'$(NC)"; \
	fi

static-analysis: ## Run staticcheck for security issues
	@echo -e "$(GREEN)Running staticcheck...$(NC)"
	@mkdir -p security
	@if command -v staticcheck &> /dev/null; then \
		staticcheck -f json ./... > security/staticcheck-report.json 2>&1 || true; \
		staticcheck ./...; \
		echo -e "$(GREEN)Staticcheck completed$(NC)"; \
	else \
		echo -e "$(YELLOW)staticcheck not installed. Run 'make install-security-tools'$(NC)"; \
	fi

dependency-check: ## Check dependencies for vulnerabilities
	@echo -e "$(GREEN)Checking dependencies for vulnerabilities...$(NC)"
	@mkdir -p security
	@echo "Go module dependencies:" > security/dependency-report.txt
	@go list -m all >> security/dependency-report.txt
	@echo "" >> security/dependency-report.txt
	@echo "Available updates:" >> security/dependency-report.txt
	@go list -u -m all >> security/dependency-report.txt
	@echo "" >> security/dependency-report.txt
	@echo "Dependency graph (first 20 lines):" >> security/dependency-report.txt
	@go mod graph | head -20 >> security/dependency-report.txt
	@echo -e "$(GREEN)Dependency analysis completed: security/dependency-report.txt$(NC)"

generate-sbom: ## Generate Software Bill of Materials (SBOM)
	@echo -e "$(GREEN)Generating SBOM in CycloneDX format...$(NC)"
	@mkdir -p security
	@if command -v cyclonedx-gomod &> /dev/null; then \
		cyclonedx-gomod mod -json -output security/sbom.json; \
		cyclonedx-gomod mod -output security/sbom.xml; \
		echo -e "$(GREEN)SBOM generated: security/sbom.json and security/sbom.xml$(NC)"; \
	else \
		echo -e "$(YELLOW)cyclonedx-gomod not installed. Run 'make install-security-tools'$(NC)"; \
	fi

nancy-scan: ## Run nancy dependency vulnerability scanner
	@echo -e "$(GREEN)Running nancy dependency scanner...$(NC)"
	@if command -v nancy &> /dev/null; then \
		go list -json -deps ./... 2>/dev/null | nancy sleuth || echo -e "$(YELLOW)Nancy scan completed (check for vulnerabilities above)$(NC)"; \
	else \
		echo -e "$(YELLOW)nancy not installed. Run 'make install-ci-tools'$(NC)"; \
	fi

gitleaks-scan: ## Run gitleaks secret scanner
	@echo -e "$(GREEN)Running gitleaks secret scanner...$(NC)"
	@if command -v gitleaks &> /dev/null; then \
		gitleaks detect --source . --verbose || echo -e "$(YELLOW)Gitleaks scan completed (check for secrets above)$(NC)"; \
	else \
		echo -e "$(YELLOW)gitleaks not installed. Run 'make install-ci-tools'$(NC)"; \
	fi

trivy-scan: ## Run trivy comprehensive vulnerability scanner
	@echo -e "$(GREEN)Running trivy vulnerability scanner...$(NC)"
	@mkdir -p security
	@if command -v trivy &> /dev/null; then \
		trivy fs --severity HIGH,CRITICAL . || echo -e "$(YELLOW)Trivy scan completed$(NC)"; \
		trivy fs --format json -o security/trivy-report.json . 2>/dev/null || true; \
		echo -e "$(GREEN)Trivy report saved to security/trivy-report.json$(NC)"; \
	else \
		echo -e "$(YELLOW)trivy not installed. Run 'make install-ci-tools'$(NC)"; \
	fi

trufflehog-scan: ## Run trufflehog for hardcoded credentials
	@echo -e "$(GREEN)Running trufflehog credential scanner...$(NC)"
	@if command -v trufflehog &> /dev/null; then \
		trufflehog filesystem . --no-update || echo -e "$(YELLOW)Trufflehog scan completed$(NC)"; \
	else \
		echo -e "$(YELLOW)trufflehog not installed. Run 'make install-ci-tools'$(NC)"; \
	fi

semgrep-scan: ## Run semgrep security pattern analysis
	@echo -e "$(GREEN)Running semgrep security analysis...$(NC)"
	@if command -v semgrep &> /dev/null; then \
		semgrep --config=auto --error --verbose . || echo -e "$(YELLOW)Semgrep scan completed$(NC)"; \
	else \
		echo -e "$(YELLOW)semgrep not installed. Install with: python3 -m pip install semgrep$(NC)"; \
		echo -e "$(YELLOW)Skipping semgrep scan (optional tool)$(NC)"; \
	fi

check-mod-tidy: ## Check if go.mod and go.sum are tidy
	@echo -e "$(GREEN)Checking go mod tidy...$(NC)"
	@cp go.mod go.mod.bak
	@cp go.sum go.sum.bak
	@go mod tidy
	@if diff go.mod go.mod.bak > /dev/null && diff go.sum go.sum.bak > /dev/null; then \
		echo -e "$(GREEN)go.mod and go.sum are tidy$(NC)"; \
		rm go.mod.bak go.sum.bak; \
	else \
		echo -e "$(RED)go.mod or go.sum need tidying. Run 'go mod tidy'$(NC)"; \
		mv go.mod.bak go.mod; \
		mv go.sum.bak go.sum; \
		exit 1; \
	fi

security-report: ## Generate consolidated security report
	@echo -e "$(GREEN)Generating consolidated security report...$(NC)"
	@mkdir -p security/reports
	@echo "# Security Scan Report - $$(date '+%Y-%m-%d %H:%M:%S')" > security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "## Summary" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "- Date: $$(date '+%Y-%m-%d %H:%M:%S')" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "- Go Version: $$(go version)" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "## Vulnerability Check (govulncheck)" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@govulncheck ./... 2>&1 | head -20 >> security/reports/security-report-$$(date '+%Y%m%d').md || echo "No vulnerabilities found" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "## Static Analysis (staticcheck)" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@staticcheck ./... 2>&1 | head -20 >> security/reports/security-report-$$(date '+%Y%m%d').md || echo "No issues found" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo "## Security Analysis (gosec)" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@gosec -fmt text ./... 2>&1 | grep "Summary" -A 5 >> security/reports/security-report-$$(date '+%Y%m%d').md || echo "No security issues found" >> security/reports/security-report-$$(date '+%Y%m%d').md
	@echo -e "$(GREEN)Security report: security/reports/security-report-$$(date '+%Y%m%d').md$(NC)"

##@ Build

build: ## Build all packages
	@echo -e "$(GREEN)Building packages...$(NC)"
	$(GOBUILD) -v ./...

##@ Dependencies

deps: ## Download and tidy dependencies
	@echo -e "$(GREEN)Managing dependencies...$(NC)"
	$(GOMOD) download
	$(GOMOD) tidy

##@ Pre-commit Hooks

install-hooks: ## Install pre-commit hooks
	@echo -e "$(GREEN)Installing pre-commit hooks...$(NC)"
	@./scripts/install-hooks.sh

run-hooks: ## Run pre-commit hooks on all files
	@echo -e "$(GREEN)Running pre-commit hooks...$(NC)"
	@if command -v pre-commit &> /dev/null; then \
		pre-commit run --all-files; \
	else \
		echo -e "$(YELLOW)pre-commit not installed. Run 'make install-hooks' first$(NC)"; \
	fi

##@ Cleanup

clean: ## Clean cache and tidy modules
	@echo -e "$(GREEN)Cleaning...$(NC)"
	$(GO) clean -cache
	$(GOMOD) tidy
	rm -f coverage.out coverage.html

##@ Documentation

docs: ## Start documentation server
	@echo "Starting Go documentation server at http://localhost:6060"
	godoc -http=:6060

.DEFAULT_GOAL := help
## Parity test: generate sklearn golden and run Go tests with parity tag
.PHONY: parity-linear
parity-linear:
	uv run --with scikit-learn --with numpy --with scipy python scripts/golden/gen_logreg.py
	go test ./sklearn/linear_model -tags=parity -run Parity -v

.PHONY: parity-lightgbm
parity-lightgbm:
	# 1) Generate compatibility data via Python (use uv only)
	uv run --with lightgbm --with scikit-learn --with numpy --with scipy python sklearn/lightgbm/generate_compatibility_data.py
	# 2) Run Go-side parity tests (limited selection)
	go test ./sklearn/lightgbm -count=1 -run 'TestPythonLightGBMCompatibility|TestPythonParity_Leaves|TestCAPIModelStructureEquality' -v
