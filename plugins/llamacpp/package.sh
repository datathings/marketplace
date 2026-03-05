#!/bin/bash

# package.sh - Package llama.cpp skill into a .skill file
# A .skill file is a zip archive containing SKILL.md and optional resources

set -e  # Exit on error

# Configuration
SKILL_NAME="llamacpp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="${SCRIPT_DIR}/skills/${SKILL_NAME}"
OUTPUT_FILE="${SKILL_NAME}.skill"
OUTPUT_DIR="${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-o OUTPUT_DIR] [-h]"
            echo "  -o, --output DIR    Output directory for .skill file (default: current directory)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Packaging ${SKILL_NAME} skill...${NC}"

# Validate skill directory exists
if [[ ! -d "${SKILL_DIR}" ]]; then
    echo -e "${RED}Error: Skill directory not found: ${SKILL_DIR}${NC}"
    echo -e "${YELLOW}Expected to find a '${SKILL_NAME}' folder in ${SCRIPT_DIR}${NC}"
    exit 1
fi

# Validate required files
if [[ ! -f "${SKILL_DIR}/SKILL.md" ]]; then
    echo -e "${RED}Error: SKILL.md not found in ${SKILL_DIR}${NC}"
    exit 1
fi

# Check SKILL.md has required frontmatter
if ! grep -q "^---$" "${SKILL_DIR}/SKILL.md"; then
    echo -e "${YELLOW}Warning: SKILL.md may be missing YAML frontmatter${NC}"
fi

if ! grep -q "^name:" "${SKILL_DIR}/SKILL.md"; then
    echo -e "${RED}Error: SKILL.md missing 'name:' in frontmatter${NC}"
    exit 1
fi

if ! grep -q "^description:" "${SKILL_DIR}/SKILL.md"; then
    echo -e "${RED}Error: SKILL.md missing 'description:' in frontmatter${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SKILL.md validated${NC}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Create temporary directory for packaging
TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT

# Copy skill directory to temporary directory
echo "Copying skill files..."
cp -r "${SKILL_DIR}"/* "${TMP_DIR}/"

# Count what we copied
if [[ -d "${TMP_DIR}/references" ]]; then
    echo -e "${GREEN}✓ Copied references directory${NC}"
fi

if [[ -d "${TMP_DIR}/scripts" ]]; then
    echo -e "${GREEN}✓ Copied scripts directory${NC}"
fi

if [[ -d "${TMP_DIR}/assets" ]]; then
    echo -e "${GREEN}✓ Copied assets directory${NC}"
fi

# Create the .skill file (zip archive)
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"

# Remove existing .skill file if it exists
if [[ -f "${OUTPUT_PATH}" ]]; then
    rm "${OUTPUT_PATH}"
    echo -e "${YELLOW}Removed existing ${OUTPUT_FILE}${NC}"
fi

# Create zip archive
echo "Creating archive..."
cd "${TMP_DIR}"
zip -r "${OUTPUT_PATH}" . -q

# Verify the archive was created
if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo -e "${RED}Error: Failed to create ${OUTPUT_FILE}${NC}"
    exit 1
fi

# Get file size
FILE_SIZE=$(du -h "${OUTPUT_PATH}" | cut -f1)

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Successfully packaged ${SKILL_NAME} skill${NC}"
echo -e "${GREEN}  Output: ${OUTPUT_PATH}${NC}"
echo -e "${GREEN}  Size: ${FILE_SIZE}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# List contents
echo ""
echo "Archive contents:"
unzip -l "${OUTPUT_PATH}"

exit 0
