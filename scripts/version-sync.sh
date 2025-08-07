#!/usr/bin/env bash
set -euo pipefail

# Version synchronization script
# Ensures VERSION file and git tags are in sync

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="$PROJECT_ROOT/VERSION"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Usage: log_info <message>
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Usage: log_warn <message>
log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Usage: log_error <message>
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
}

# Read version from VERSION file
read_version_file() {
    if [[ ! -f "$VERSION_FILE" ]]; then
        log_error "VERSION file not found: $VERSION_FILE"
        exit 1
    fi
    
    local version
    version=$(cat "$VERSION_FILE" | tr -d '[:space:]')
    
    if [[ -z "$version" ]]; then
        log_error "VERSION file is empty"
        exit 1
    fi
    
    echo "$version"
}

# Get latest git tag
get_latest_tag() {
    git tag -l --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n1 || echo ""
}

# Validate semantic version format
validate_version() {
    local version="$1"
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: $version (expected: x.y.z)"
        exit 1
    fi
}

# Create git tag
create_tag() {
    local version="$1"
    local tag="v$version"
    
    log_info "Creating git tag: $tag"
    
    # Check if tag already exists
    if git tag -l | grep -q "^$tag$"; then
        log_warn "Tag $tag already exists"
        return 0
    fi
    
    # Create annotated tag
    git tag -a "$tag" -m "Release $tag

Automated version sync from VERSION file.

$(date '+%Y-%m-%d %H:%M:%S %Z')"
    
    log_info "Created tag: $tag"
}

# Update VERSION file
update_version_file() {
    local version="$1"
    echo "$version" > "$VERSION_FILE"
    log_info "Updated VERSION file: $version"
}

# Main function
main() {
    local mode="${1:-sync}"
    
    check_git_repo
    
    case "$mode" in
        "sync")
            log_info "Synchronizing VERSION file with git tags"
            
            local file_version
            file_version=$(read_version_file)
            validate_version "$file_version"
            
            local latest_tag
            latest_tag=$(get_latest_tag)
            
            if [[ -z "$latest_tag" ]]; then
                log_info "No version tags found, creating initial tag v$file_version"
                create_tag "$file_version"
            else
                local tag_version="${latest_tag#v}"
                
                log_info "VERSION file: $file_version"
                log_info "Latest tag: $latest_tag ($tag_version)"
                
                if [[ "$file_version" != "$tag_version" ]]; then
                    log_info "Version mismatch detected"
                    
                    # Compare versions to determine action
                    if printf '%s\n%s\n' "$tag_version" "$file_version" | sort -V | tail -n1 | grep -q "$file_version"; then
                        if [[ "$file_version" != "$tag_version" ]]; then
                            log_info "VERSION file is newer, creating tag"
                            create_tag "$file_version"
                        fi
                    else
                        log_warn "VERSION file ($file_version) is older than latest tag ($tag_version)"
                        log_info "Use 'scripts/version-sync.sh update-version' to update VERSION file"
                    fi
                else
                    log_info "VERSION file and latest tag are synchronized"
                fi
            fi
            ;;
            
        "update-version")
            log_info "Updating VERSION file to match latest tag"
            
            local latest_tag
            latest_tag=$(get_latest_tag)
            
            if [[ -z "$latest_tag" ]]; then
                log_error "No version tags found"
                exit 1
            fi
            
            local tag_version="${latest_tag#v}"
            update_version_file "$tag_version"
            ;;
            
        "create-tag")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 create-tag <version>"
                exit 1
            fi
            
            local version="$2"
            validate_version "$version"
            create_tag "$version"
            update_version_file "$version"
            ;;
            
        "check")
            log_info "Checking VERSION file and git tag synchronization"
            
            local file_version
            file_version=$(read_version_file)
            validate_version "$file_version"
            
            local latest_tag
            latest_tag=$(get_latest_tag)
            
            echo "VERSION file: $file_version"
            if [[ -n "$latest_tag" ]]; then
                echo "Latest tag: $latest_tag (${latest_tag#v})"
                
                if [[ "$file_version" == "${latest_tag#v}" ]]; then
                    log_info "✅ Synchronized"
                    exit 0
                else
                    log_error "❌ Not synchronized"
                    exit 1
                fi
            else
                log_warn "No version tags found"
                exit 1
            fi
            ;;
            
        *)
            echo "Usage: $0 {sync|update-version|create-tag <version>|check}"
            echo ""
            echo "Commands:"
            echo "  sync           Synchronize VERSION file and git tags (default)"
            echo "  update-version Update VERSION file to match latest tag"
            echo "  create-tag     Create new tag and update VERSION file"
            echo "  check          Check if VERSION file and tags are synchronized"
            exit 1
            ;;
    esac
}

main "$@"