#!/usr/bin/env bash

# Package All Skills
# Creates .skill files for all skills in the marketplace and outputs them to ./skills/

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/skills"
PLUGINS_DIR="${SCRIPT_DIR}/plugins"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
CLEAN=false
ALL=false
SKILL_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -a|--all)
            ALL=true
            shift
            ;;
        -s|--skill)
            SKILL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-o OUTPUT_DIR] [-c] [-a] [-s SKILL_NAME] [-h]"
            echo "  -o, --output DIR    Output directory for .skill files (default: ./skills)"
            echo "  -c, --clean         Remove existing .skill files before packaging"
            echo "  -a, --all           Package all skills without prompting"
            echo "  -s, --skill NAME    Package a specific skill by name (e.g., greycat, llamacpp)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            # Treat positional argument as skill name if no flag
            if [[ -z "${SKILL_NAME}" ]] && [[ "$1" != -* ]]; then
                SKILL_NAME="$1"
                shift
            else
                echo -e "${RED}Error: Unknown option $1${NC}"
                exit 1
            fi
            ;;
    esac
done

# Adjust header based on mode
if [[ -n "${SKILL_NAME}" ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Packaging Skill: ${SKILL_NAME}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Packaging All Skills${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Clean if requested
if [[ "${CLEAN}" == true ]]; then
    echo -e "${YELLOW}Cleaning existing .skill files...${NC}"
    rm -f "${OUTPUT_DIR}"/*.skill
fi

# Track results
PACKAGED=()
FAILED=()

# Function to validate and package a skill
package_skill() {
    local skill_dir="$1"
    local skill_name="$(basename "${skill_dir}")"
    local skill_md="${skill_dir}/SKILL.md"

    echo -e "${GREEN}Packaging ${skill_name}...${NC}"

    # Validate SKILL.md exists
    if [[ ! -f "${skill_md}" ]]; then
        echo -e "${RED}  ✗ SKILL.md not found${NC}"
        FAILED+=("${skill_name}: missing SKILL.md")
        return 1
    fi

    # Validate frontmatter
    if ! grep -q "^---$" "${skill_md}"; then
        echo -e "${YELLOW}  ⚠ Warning: may be missing YAML frontmatter${NC}"
    fi

    if ! grep -q "^name:" "${skill_md}"; then
        echo -e "${RED}  ✗ Missing 'name:' in frontmatter${NC}"
        FAILED+=("${skill_name}: missing name field")
        return 1
    fi

    if ! grep -q "^description:" "${skill_md}"; then
        echo -e "${RED}  ✗ Missing 'description:' in frontmatter${NC}"
        FAILED+=("${skill_name}: missing description field")
        return 1
    fi

    # Create temporary directory
    local tmp_dir=$(mktemp -d)
    trap "rm -rf ${tmp_dir}" RETURN

    # Copy skill files (dereference symlinks with -L)
    cp -rL "${skill_dir}"/* "${tmp_dir}/" 2>/dev/null || cp -r "${skill_dir}"/* "${tmp_dir}/"

    # Remove unwanted files
    find "${tmp_dir}" -name '.DS_Store' -delete 2>/dev/null || true
    find "${tmp_dir}" -name '*.pyc' -delete 2>/dev/null || true
    find "${tmp_dir}" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find "${tmp_dir}" -type d -name 'node_modules' -exec rm -rf {} + 2>/dev/null || true

    # Create .skill archive
    local output_path="${OUTPUT_DIR}/${skill_name}.skill"
    rm -f "${output_path}"

    (cd "${tmp_dir}" && zip -r "${output_path}" . -q)

    if [[ -f "${output_path}" ]]; then
        local file_size=$(du -h "${output_path}" | cut -f1)
        echo -e "${GREEN}  ✓ Created ${skill_name}.skill (${file_size})${NC}"
        PACKAGED+=("${skill_name}")
        return 0
    else
        echo -e "${RED}  ✗ Failed to create archive${NC}"
        FAILED+=("${skill_name}: archive creation failed")
        return 1
    fi
}

# Discover all available skills
echo "Scanning for skills..."
AVAILABLE_SKILLS=()
SKILL_DIRS=()

for plugin_dir in "${PLUGINS_DIR}"/*/; do
    if [[ -d "${plugin_dir}/skills" ]]; then
        for skill_dir in "${plugin_dir}/skills"/*/; do
            if [[ -d "${skill_dir}" ]]; then
                skill_name=$(basename "${skill_dir}")
                AVAILABLE_SKILLS+=("${skill_name}")
                SKILL_DIRS+=("${skill_dir}")
            fi
        done
    fi
done

if [[ ${#AVAILABLE_SKILLS[@]} -eq 0 ]]; then
    echo -e "${RED}No skills found in ${PLUGINS_DIR}${NC}"
    exit 1
fi

echo -e "Found ${#AVAILABLE_SKILLS[@]} skill(s): ${AVAILABLE_SKILLS[*]}"
echo ""

# Skill selection
SELECTED_INDICES=()

if [[ -n "${SKILL_NAME}" ]]; then
    # Find specific skill by name
    FOUND=false
    for i in "${!AVAILABLE_SKILLS[@]}"; do
        if [[ "${AVAILABLE_SKILLS[$i]}" == "${SKILL_NAME}" ]]; then
            SELECTED_INDICES+=("$i")
            FOUND=true
            echo -e "${GREEN}Selected: ${SKILL_NAME}${NC}"
            break
        fi
    done
    if [[ "${FOUND}" == false ]]; then
        echo -e "${RED}Error: Skill '${SKILL_NAME}' not found${NC}"
        echo -e "${YELLOW}Available skills: ${AVAILABLE_SKILLS[*]}${NC}"
        exit 1
    fi
elif [[ "${ALL}" == true ]]; then
    # Select all skills
    for i in "${!AVAILABLE_SKILLS[@]}"; do
        SELECTED_INDICES+=("$i")
    done
else
    # Interactive selection
    echo -e "${BLUE}Select skills to package:${NC}"
    echo ""
    echo "  0) ALL (default)"
    for i in "${!AVAILABLE_SKILLS[@]}"; do
        echo "  $((i + 1))) ${AVAILABLE_SKILLS[$i]}"
    done
    echo ""
    echo -n "Enter selection (space-separated numbers, or press Enter for ALL): "
    read -r selection

    if [[ -z "${selection}" ]] || [[ "${selection}" == "0" ]]; then
        # Select all
        for i in "${!AVAILABLE_SKILLS[@]}"; do
            SELECTED_INDICES+=("$i")
        done
        echo -e "${GREEN}Selected: ALL${NC}"
    else
        # Parse selection
        for num in ${selection}; do
            if [[ "${num}" == "0" ]]; then
                # User selected ALL along with other numbers - treat as ALL
                SELECTED_INDICES=()
                for i in "${!AVAILABLE_SKILLS[@]}"; do
                    SELECTED_INDICES+=("$i")
                done
                break
            elif [[ "${num}" =~ ^[0-9]+$ ]] && [[ "${num}" -ge 1 ]] && [[ "${num}" -le ${#AVAILABLE_SKILLS[@]} ]]; then
                SELECTED_INDICES+=("$((num - 1))")
            else
                echo -e "${YELLOW}Warning: Invalid selection '${num}' ignored${NC}"
            fi
        done

        if [[ ${#SELECTED_INDICES[@]} -eq 0 ]]; then
            echo -e "${RED}No valid skills selected. Exiting.${NC}"
            exit 1
        fi

        # Show selected skills
        echo -n -e "${GREEN}Selected:${NC}"
        for idx in "${SELECTED_INDICES[@]}"; do
            echo -n " ${AVAILABLE_SKILLS[$idx]}"
        done
        echo ""
    fi
fi

echo ""

# Package selected skills
for idx in "${SELECTED_INDICES[@]}"; do
    package_skill "${SKILL_DIRS[$idx]}" || true
    echo ""
done

# Note: .claude/skills/ contains dev tools (skill-creator) - not packaged for distribution

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ ${#PACKAGED[@]} -gt 0 ]]; then
    echo -e "${GREEN}Successfully packaged (${#PACKAGED[@]}):${NC}"
    for skill in "${PACKAGED[@]}"; do
        echo -e "  ${GREEN}✓${NC} ${skill}.skill"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed (${#FAILED[@]}):${NC}"
    for failure in "${FAILED[@]}"; do
        echo -e "  ${RED}✗${NC} ${failure}"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# List all .skill files
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"/*.skill 2>/dev/null || echo "  (none)"

exit 0
