#!/usr/bin/env bash

# SciGo CI Tools Installer
# This script installs all tools needed for local CI execution

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SciGo CI Tools Installer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to install Go tool
install_go_tool() {
    local tool_name="$1"
    local install_path="$2"
    
    echo -e "${GREEN}Installing $tool_name...${NC}"
    if go install "$install_path"; then
        echo -e "${GREEN}✓ $tool_name installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install $tool_name${NC}"
        return 1
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check for Go
if ! command_exists go; then
    echo -e "${RED}Go is not installed. Please install Go first.${NC}"
    echo "Visit: https://golang.org/dl/"
    exit 1
fi

echo -e "${BLUE}Installing Go-based tools...${NC}"
echo ""

# Install Go tools
install_go_tool "govulncheck" "golang.org/x/vuln/cmd/govulncheck@latest"
install_go_tool "gosec" "github.com/securego/gosec/v2/cmd/gosec@latest"
install_go_tool "staticcheck" "honnef.co/go/tools/cmd/staticcheck@latest"
install_go_tool "cyclonedx-gomod" "github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@latest"
install_go_tool "nancy" "github.com/sonatype-nexus-community/nancy@latest"
install_go_tool "golangci-lint" "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"

echo ""
echo -e "${BLUE}Checking for system package managers...${NC}"

# Check for Homebrew (macOS/Linux)
if command_exists brew; then
    echo -e "${GREEN}Homebrew found. Installing system tools...${NC}"
    
    # Function to install brew package
    install_brew_package() {
        local package="$1"
        echo -e "${GREEN}Installing $package...${NC}"
        if brew list "$package" &>/dev/null; then
            echo -e "${YELLOW}$package is already installed${NC}"
        else
            if brew install "$package"; then
                echo -e "${GREEN}✓ $package installed successfully${NC}"
            else
                echo -e "${YELLOW}⚠ Failed to install $package${NC}"
            fi
        fi
    }
    
    install_brew_package "gitleaks"
    install_brew_package "trivy"
    install_brew_package "trufflehog"
    
elif command_exists apt-get; then
    echo -e "${GREEN}APT found. Installing system tools...${NC}"
    echo -e "${YELLOW}Note: Some tools may require manual installation on Linux${NC}"
    
    # Gitleaks installation for Linux
    echo -e "${GREEN}Installing gitleaks...${NC}"
    if ! command_exists gitleaks; then
        GITLEAKS_VERSION="8.18.0"
        wget -q "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
        tar -xzf "gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
        sudo mv gitleaks /usr/local/bin/
        rm "gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
        echo -e "${GREEN}✓ gitleaks installed${NC}"
    else
        echo -e "${YELLOW}gitleaks already installed${NC}"
    fi
    
    # Trivy installation for Linux
    echo -e "${GREEN}Installing trivy...${NC}"
    if ! command_exists trivy; then
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install -y trivy
        echo -e "${GREEN}✓ trivy installed${NC}"
    else
        echo -e "${YELLOW}trivy already installed${NC}"
    fi
    
    # Trufflehog installation note
    echo -e "${YELLOW}Note: trufflehog requires manual installation on Linux${NC}"
    echo "Visit: https://github.com/trufflesecurity/trufflehog#installation"
    
else
    echo -e "${YELLOW}No supported package manager found (brew or apt-get)${NC}"
    echo -e "${YELLOW}Please install the following tools manually:${NC}"
    echo "  - gitleaks: https://github.com/gitleaks/gitleaks"
    echo "  - trivy: https://github.com/aquasecurity/trivy"
    echo "  - trufflehog: https://github.com/trufflesecurity/trufflehog"
fi

echo ""
echo -e "${BLUE}Installing Python-based tools...${NC}"

# Check for Python3
if command_exists python3; then
    echo -e "${GREEN}Python3 found. Installing semgrep...${NC}"
    
    # Install semgrep
    if python3 -m pip install --user semgrep; then
        echo -e "${GREEN}✓ semgrep installed${NC}"
        echo -e "${YELLOW}Note: You may need to add ~/.local/bin to your PATH${NC}"
    else
        echo -e "${YELLOW}⚠ Failed to install semgrep${NC}"
        echo "Try: python3 -m pip install --user semgrep"
    fi
else
    echo -e "${YELLOW}Python3 not found. Semgrep requires Python3.${NC}"
    echo "Visit: https://www.python.org/downloads/"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Installation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check installed tools
echo -e "${GREEN}Checking installed tools:${NC}"

check_and_report() {
    local tool="$1"
    if command_exists "$tool"; then
        echo -e "${GREEN}✓ $tool${NC}"
    else
        echo -e "${RED}✗ $tool (not found)${NC}"
    fi
}

echo -e "${BLUE}Go tools:${NC}"
check_and_report "go"
check_and_report "govulncheck"
check_and_report "gosec"
check_and_report "staticcheck"
check_and_report "cyclonedx-gomod"
check_and_report "nancy"
check_and_report "golangci-lint"

echo ""
echo -e "${BLUE}System tools:${NC}"
check_and_report "gitleaks"
check_and_report "trivy"
check_and_report "trufflehog"

echo ""
echo -e "${BLUE}Python tools:${NC}"
check_and_report "semgrep"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Note: You may need to restart your terminal or update your PATH${NC}"
echo ""
echo -e "${BLUE}To run all CI checks locally, use:${NC}"
echo "  make ci-local"
echo ""
echo -e "${BLUE}Or run the script directly:${NC}"
echo "  ./scripts/run-ci-local.sh"