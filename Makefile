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

all: fmt lint test ## Run all: format, lint, and test

##@ Development

test: ## Run tests
	@echo -e "$(GREEN)Running tests...$(NC)"
	$(GOTEST) -v -race -cover ./...

test-short: ## Run short tests
	@echo -e "$(GREEN)Running short tests...$(NC)"
	$(GOTEST) -v -short ./...

coverage: ## Run tests with coverage report
	@echo -e "$(GREEN)Running tests with coverage...$(NC)"
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out

bench: ## Run benchmarks
	@echo -e "$(GREEN)Running benchmarks...$(NC)"
	$(GOTEST) -bench=. -benchmem ./...

##@ Code Quality

fmt: ## Format code
	@echo -e "$(GREEN)Formatting code...$(NC)"
	$(GO) fmt ./...

lint: ## Run go vet
	@echo -e "$(GREEN)Running go vet...$(NC)"
	$(GOVET) ./...

lint-full: ## Run golangci-lint
	@echo -e "$(GREEN)Running golangci-lint...$(NC)"
	@if command -v golangci-lint &> /dev/null; then \
		$(GOLANGCI_LINT) run ./...; \
	else \
		echo -e "$(YELLOW)golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(NC)"; \
	fi

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