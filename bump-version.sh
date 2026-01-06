#!/usr/bin/env bash

# Bump version across all plugins and marketplace.json
# Usage: ./bump-version.sh 1.3.0
#        ./bump-version.sh          # shows current versions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKETPLACE_JSON="${SCRIPT_DIR}/.claude-plugin/marketplace.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Show current versions
show_versions() {
    echo -e "${BLUE}Current versions:${NC}"
    echo ""

    # Marketplace
    echo -e "${YELLOW}marketplace.json:${NC}"
    grep -o '"version": "[^"]*"' "$MARKETPLACE_JSON" | head -1 | sed 's/"version": "/  /' | sed 's/"//'

    # Individual plugins
    echo ""
    echo -e "${YELLOW}Plugin manifests:${NC}"
    for plugin_json in "${SCRIPT_DIR}"/plugins/*/.claude-plugin/plugin.json; do
        plugin_name=$(basename "$(dirname "$(dirname "$plugin_json")")")
        version=$(grep -o '"version": "[^"]*"' "$plugin_json" | sed 's/"version": "//' | sed 's/"//')
        echo "  $plugin_name: $version"
    done
}

# Bump all versions
bump_version() {
    local new_version="$1"

    echo -e "${BLUE}Bumping all versions to ${GREEN}${new_version}${NC}"
    echo ""

    # Update marketplace.json
    echo -e "${YELLOW}Updating marketplace.json...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$MARKETPLACE_JSON"
    else
        sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$MARKETPLACE_JSON"
    fi
    echo -e "  ${GREEN}✓${NC} Updated"

    # Update each plugin
    for plugin_json in "${SCRIPT_DIR}"/plugins/*/.claude-plugin/plugin.json; do
        plugin_name=$(basename "$(dirname "$(dirname "$plugin_json")")")
        echo -e "${YELLOW}Updating ${plugin_name}...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$plugin_json"
        else
            sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$plugin_json"
        fi
        echo -e "  ${GREEN}✓${NC} Updated"
    done

    echo ""
    echo -e "${GREEN}All versions bumped to ${new_version}${NC}"
    echo ""
    echo "Next steps:"
    echo "  git add -A && git commit -m \"Bump version to ${new_version}\""
    echo "  git tag v${new_version}"
    echo "  git push origin main --tags"
}

# Main
if [[ -z "$1" ]]; then
    show_versions
    echo ""
    echo -e "Usage: $0 <new-version>"
    echo -e "Example: $0 1.3.0"
else
    if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 1.3.0)"
        exit 1
    fi
    bump_version "$1"
fi
