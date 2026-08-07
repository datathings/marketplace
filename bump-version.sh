#!/usr/bin/env bash

# Bump version across all Claude/Codex plugin manifests and marketplace metadata
# Usage: ./bump-version.sh 1.3.0
#        ./bump-version.sh          # shows current versions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_MARKETPLACE_JSON="${SCRIPT_DIR}/.claude-plugin/marketplace.json"
SKILLS_MARKETPLACE_JSON="${SCRIPT_DIR}/marketplace.json"
README="${SCRIPT_DIR}/README.md"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Show current versions
show_versions() {
    echo -e "${BLUE}Current versions:${NC}"
    echo ""

    # Marketplaces
    echo -e "${YELLOW}Claude marketplace:${NC}"
    grep -o '"version": "[^"]*"' "$CLAUDE_MARKETPLACE_JSON" | head -1 | sed 's/"version": "/  /' | sed 's/"//'

    echo -e "${YELLOW}Standalone skills marketplace:${NC}"
    grep -o '"version": "[^"]*"' "$SKILLS_MARKETPLACE_JSON" | head -1 | sed 's/"version": "/  /' | sed 's/"//'

    # Individual Claude plugins
    echo ""
    echo -e "${YELLOW}Claude plugin manifests:${NC}"
    for plugin_json in "${SCRIPT_DIR}"/plugins/*/.claude-plugin/plugin.json; do
        plugin_name=$(basename "$(dirname "$(dirname "$plugin_json")")")
        version=$(grep -o '"version": "[^"]*"' "$plugin_json" | sed 's/"version": "//' | sed 's/"//')
        echo "  $plugin_name: $version"
    done

    # Individual Codex plugins
    echo ""
    echo -e "${YELLOW}Codex plugin manifests:${NC}"
    for plugin_json in "${SCRIPT_DIR}"/plugins/*/.codex-plugin/plugin.json; do
        plugin_name=$(basename "$(dirname "$(dirname "$plugin_json")")")
        version=$(grep -o '"version": "[^"]*"' "$plugin_json" | sed 's/"version": "//' | sed 's/"//')
        echo "  $plugin_name: $version"
    done

    # README
    echo ""
    echo -e "${YELLOW}README.md:${NC}"
    grep -oE '\| [0-9]+\.[0-9]+\.[0-9]+ \|' "$README" | head -1 | sed 's/| /  /' | sed 's/ |//'
}

# Bump all versions
bump_version() {
    local new_version="$1"

    echo -e "${BLUE}Bumping all versions to ${GREEN}${new_version}${NC}"
    echo ""

    # Update versioned marketplace metadata. The Codex marketplace catalog does
    # not carry versions; versions live in each .codex-plugin/plugin.json.
    for marketplace_json in "$CLAUDE_MARKETPLACE_JSON" "$SKILLS_MARKETPLACE_JSON"; do
        marketplace_path="${marketplace_json#${SCRIPT_DIR}/}"
        echo -e "${YELLOW}Updating ${marketplace_path}...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$marketplace_json"
        else
            sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$marketplace_json"
        fi
        echo -e "  ${GREEN}✓${NC} Updated"
    done

    # Update each Claude and Codex plugin manifest.
    for manifest_type in .claude-plugin .codex-plugin; do
        for plugin_json in "${SCRIPT_DIR}"/plugins/*/"${manifest_type}"/plugin.json; do
            plugin_name=$(basename "$(dirname "$(dirname "$plugin_json")")")
            echo -e "${YELLOW}Updating ${plugin_name}/${manifest_type}...${NC}"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$plugin_json"
            else
                sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${new_version}\"/g" "$plugin_json"
            fi
            echo -e "  ${GREEN}✓${NC} Updated"
        done
    done

    # Update README.md plugin table
    echo -e "${YELLOW}Updating README.md...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/| [0-9]\+\.[0-9]\+\.[0-9]\+ |/| ${new_version} |/g" "$README"
    else
        sed -i "s/| [0-9]\+\.[0-9]\+\.[0-9]\+ |/| ${new_version} |/g" "$README"
    fi
    echo -e "  ${GREEN}✓${NC} Updated"

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
