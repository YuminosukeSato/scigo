#!/usr/bin/env bash
set -euo pipefail

# Auto-version increment script for CD pipeline
# Automatically increments patch version for main branch pushes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION_FILE="$PROJECT_ROOT/VERSION"

# Read current version
if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: VERSION file not found at $VERSION_FILE"
    exit 1
fi

CURRENT_VERSION=$(cat "$VERSION_FILE")
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Increment patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"

echo "New version: $NEW_VERSION"

# Update VERSION file
echo "$NEW_VERSION" > "$VERSION_FILE"

# Update CHANGELOG.md - add new Unreleased section
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"
RELEASE_DATE=$(date +%Y-%m-%d)

# Create temporary file for new changelog
TEMP_FILE=$(mktemp)

# Write the new version section
{
    # Keep header until [Unreleased]
    sed -n '1,/^## \[Unreleased\]/p' "$CHANGELOG_FILE"
    
    # Add new version section
    echo ""
    echo "## [$NEW_VERSION] - $RELEASE_DATE"
    echo ""
    echo "### Changed"
    echo "- Automated patch release"
    echo ""
    
    # Skip the old [Unreleased] line and keep the rest
    sed -n '/^## \[Unreleased\]/,$ { /^## \[Unreleased\]/!p }' "$CHANGELOG_FILE"
} > "$TEMP_FILE"

# Update the comparison links at the bottom
{
    head -n -1 "$TEMP_FILE"
    echo "[Unreleased]: https://github.com/YuminosukeSato/scigo/compare/v${NEW_VERSION}...HEAD"
    echo "[$NEW_VERSION]: https://github.com/YuminosukeSato/scigo/compare/v${CURRENT_VERSION}...v${NEW_VERSION}"
    tail -n +2 "$CHANGELOG_FILE" | grep -E '^\[[0-9]+\.[0-9]+\.[0-9]+\]:' || true
} > "${TEMP_FILE}.final"

mv "${TEMP_FILE}.final" "$CHANGELOG_FILE"
rm -f "$TEMP_FILE"

echo "Updated VERSION file to $NEW_VERSION"
echo "Updated CHANGELOG.md with new version entry"

# Export for GitHub Actions
echo "version=$NEW_VERSION" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "tag=v$NEW_VERSION" >> "${GITHUB_OUTPUT:-/dev/null}"