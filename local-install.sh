#!/usr/bin/env bash
#
# Install all greycat-related plugins locally using symlinks.
# Uses version 0.0.0 so it never conflicts with marketplace versions.
#
# Usage:
#   ./local-install.sh          # Install (symlink) plugins
#   ./local-install.sh --clean  # Remove local dev plugins
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_DIR="$HOME/.claude"
CACHE_DIR="$CLAUDE_DIR/plugins/cache/datathings"
INSTALLED_JSON="$CLAUDE_DIR/plugins/installed_plugins.json"
SETTINGS_JSON="$CLAUDE_DIR/settings.json"
MARKETPLACE="datathings"
VERSION="0.0.0"

PLUGINS=(greycat greycat-c greycat-lsp)

# --- helpers ---

ensure_jq() {
  if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install it with: apt install jq / dnf install jq / brew install jq"
    exit 1
  fi
}

ensure_claude_dirs() {
  mkdir -p "$CLAUDE_DIR/plugins/cache/datathings"
}

# --- clean ---

clean() {
  echo "Cleaning local dev plugins..."
  for plugin in "${PLUGINS[@]}"; do
    local link="$CACHE_DIR/$plugin/$VERSION"
    if [ -L "$link" ]; then
      rm "$link"
      echo "  Removed symlink: $link"
      # Remove parent dir if empty
      rmdir "$CACHE_DIR/$plugin" 2>/dev/null || true
    fi
  done

  # Remove entries from installed_plugins.json
  if [ -f "$INSTALLED_JSON" ]; then
    local tmp
    tmp=$(mktemp)
    local result="$(<"$INSTALLED_JSON")"
    for plugin in "${PLUGINS[@]}"; do
      local key="${plugin}@${MARKETPLACE}"
      result=$(echo "$result" | jq "del(.plugins[\"$key\"])")
    done
    echo "$result" > "$tmp"
    mv "$tmp" "$INSTALLED_JSON"
    echo "  Updated $INSTALLED_JSON"
  fi

  # Remove entries from settings.json
  if [ -f "$SETTINGS_JSON" ]; then
    local tmp
    tmp=$(mktemp)
    local result="$(<"$SETTINGS_JSON")"
    for plugin in "${PLUGINS[@]}"; do
      local key="${plugin}@${MARKETPLACE}"
      result=$(echo "$result" | jq "del(.enabledPlugins[\"$key\"])")
    done
    echo "$result" > "$tmp"
    mv "$tmp" "$SETTINGS_JSON"
    echo "  Updated $SETTINGS_JSON"
  fi

  echo "Done."
}

# --- install ---

install() {
  ensure_claude_dirs

  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")

  echo "Installing local dev plugins (v$VERSION)..."

  # Create symlinks
  for plugin in "${PLUGINS[@]}"; do
    local src="$SCRIPT_DIR/plugins/$plugin"
    local dst="$CACHE_DIR/$plugin/$VERSION"

    if [ ! -d "$src" ]; then
      echo "  Warning: $src does not exist, skipping $plugin"
      continue
    fi

    mkdir -p "$CACHE_DIR/$plugin"

    # Remove existing (symlink or dir)
    if [ -L "$dst" ]; then
      rm "$dst"
    elif [ -d "$dst" ]; then
      echo "  Warning: $dst is a real directory, skipping (use --clean first)"
      continue
    fi

    ln -s "$src" "$dst"
    echo "  Linked: $dst -> $src"
  done

  # Update installed_plugins.json
  if [ ! -f "$INSTALLED_JSON" ]; then
    echo '{"version": 2, "plugins": {}}' > "$INSTALLED_JSON"
  fi

  local tmp
  tmp=$(mktemp)
  local result="$(<"$INSTALLED_JSON")"
  for plugin in "${PLUGINS[@]}"; do
    local key="${plugin}@${MARKETPLACE}"
    local install_path="$CACHE_DIR/$plugin/$VERSION"
    result=$(echo "$result" | jq \
      --arg key "$key" \
      --arg path "$install_path" \
      --arg ver "$VERSION" \
      --arg now "$now" \
      '.plugins[$key] = [{
        "scope": "user",
        "installPath": $path,
        "version": $ver,
        "installedAt": $now,
        "lastUpdated": $now,
        "gitCommitSha": "local-dev"
      }]')
  done
  echo "$result" > "$tmp"
  mv "$tmp" "$INSTALLED_JSON"
  echo "  Updated $INSTALLED_JSON"

  # Update settings.json
  if [ ! -f "$SETTINGS_JSON" ]; then
    echo '{"enabledPlugins": {}}' > "$SETTINGS_JSON"
  fi

  tmp=$(mktemp)
  result="$(<"$SETTINGS_JSON")"
  for plugin in "${PLUGINS[@]}"; do
    local key="${plugin}@${MARKETPLACE}"
    result=$(echo "$result" | jq --arg key "$key" '.enabledPlugins[$key] = true')
  done
  echo "$result" > "$tmp"
  mv "$tmp" "$SETTINGS_JSON"
  echo "  Updated $SETTINGS_JSON"

  echo ""
  echo "Done. Restart Claude Code to pick up changes."
  echo "Installed plugins:"
  for plugin in "${PLUGINS[@]}"; do
    echo "  - ${plugin}@${MARKETPLACE} v${VERSION}"
  done
}

# --- main ---

ensure_jq

case "${1:-}" in
  --clean)
    clean
    ;;
  *)
    install
    ;;
esac
