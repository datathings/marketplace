#!/usr/bin/env bash

# Package GreyCat Language Skill
# Creates a .skill file (zip archive) from the greycat-language directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Skill directory
SKILL_DIR="greycat-language"

# Skill name (from SKILL.md frontmatter)
SKILL_NAME="greycat"

# Output file
OUTPUT_FILE="${SKILL_NAME}.skill"

# Output directory (default: current directory)
OUTPUT_DIR="${1:-.}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Make output path absolute
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/${OUTPUT_FILE}"

echo -e "${YELLOW}Packaging GreyCat Language Skill...${NC}"
echo ""

# Validate skill directory exists
if [ ! -d "$SKILL_DIR" ]; then
    echo -e "${RED}Error: ${SKILL_DIR} directory not found${NC}"
    exit 1
fi

# Validate SKILL.md exists
if [ ! -f "${SKILL_DIR}/SKILL.md" ]; then
    echo -e "${RED}Error: SKILL.md not found in ${SKILL_DIR}/${NC}"
    exit 1
fi

# Check for required frontmatter
if ! grep -q "^name:" "${SKILL_DIR}/SKILL.md" || ! grep -q "^description:" "${SKILL_DIR}/SKILL.md"; then
    echo -e "${RED}Error: SKILL.md missing required frontmatter (name, description)${NC}"
    exit 1
fi

# Create temporary directory for packaging
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Copying files to temporary directory..."

# Copy all files from skill directory except excluded ones
rsync -a \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "${SKILL_DIR}/" "$TMP_DIR/"

# Create the .skill file (zip archive)
echo "Creating ${OUTPUT_FILE}..."
cd "$TMP_DIR"
zip -q -r "$OUTPUT_PATH" ./*

cd "$SCRIPT_DIR"

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)

echo ""
echo -e "${GREEN}✓ Successfully packaged skill!${NC}"
echo ""
echo "  File: ${OUTPUT_PATH}"
echo "  Size: ${FILE_SIZE}"
echo ""
echo "The skill can now be distributed and installed."
