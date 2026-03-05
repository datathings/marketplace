---
name: upgrade
description: Update GreyCat libraries to latest available versions
allowed-tools: Bash(*), Read, Edit, Grep
---

# Update GreyCat Libraries

**Purpose**: Upgrade all GreyCat libraries to the latest versions from get.greycat.io

**Run After**: Monthly maintenance, before major releases, when breaking changes are announced

---

## Overview

This command upgrades GreyCat libraries by:

1. Fetching latest versions from get.greycat.io
2. Detecting which libraries are currently used in `project.gcl`
3. Comparing current vs latest versions (exit early if up-to-date)
4. **Asking user confirmation before applying changes**
5. Updating `project.gcl` with latest versions
6. Installing updated libraries
7. Verifying with `greycat-lang lint`

---

## Available GreyCat Libraries

### Core Libraries
- **std** - Standard library (required, core functionality)
- **explorer** - Graph explorer UI (recommended for development)

### Pro Libraries (Shared Version)
All pro libraries share the same version:
- **ai** - LLM/embedding support (llama.cpp integration)
- **algebra** - ML pipelines, neural networks, patterns, transforms
- **kafka** - Apache Kafka integration
- **sql** - SQL database integration (PostgreSQL)
- **s3** - Amazon S3 integration
- **finance** - Financial data structures and operations
- **powerflow** - Power flow analysis
- **opcua** - OPC UA industrial protocol
- **useragent** - User agent parsing and detection

---

## Step 1: Fetch Latest Versions

Fetch the latest library versions from get.greycat.io:

```bash
echo "================================================================================"
echo "FETCHING LATEST LIBRARY VERSIONS"
echo "================================================================================"
echo ""

# Fetch core libraries
echo "Core libraries:"
STD_VERSION=$(curl -s "https://get.greycat.io/files/core/dev/latest" | cut -d'/' -f2)
EXPLORER_VERSION=$(curl -s "https://get.greycat.io/files/explorer/dev/latest" | cut -d'/' -f2)

# If std version is empty, use explorer version as fallback
if [ -z "$STD_VERSION" ]; then
  STD_VERSION="$EXPLORER_VERSION"
  echo "  Note: std version endpoint returned empty, using explorer version as fallback"
fi

echo "  std:      $STD_VERSION"
echo "  explorer: $EXPLORER_VERSION"
echo ""

# Fetch pro libraries version (all share same version)
echo "Pro libraries (all share same version):"
PRO_VERSION=$(curl -s "https://get.greycat.io/files/algebra/dev/latest" | cut -d'/' -f2)
echo "  Version: $PRO_VERSION"
echo "  Libraries: ai, algebra, kafka, sql, s3, finance, powerflow, opcua, useragent"
echo ""
```

**Note**: All pro libraries (ai, algebra, kafka, sql, s3, finance, powerflow, opcua, useragent) share the same version. We only need to fetch one endpoint.

---

## Step 2: Detect Current Libraries

Find which libraries are currently declared in `project.gcl`:

```bash
echo "================================================================================"
echo "CURRENT LIBRARIES IN project.gcl"
echo "================================================================================"
echo ""

# Detect which libraries are currently in project.gcl
CURRENT_LIBS=$(grep -o '@library("[^"]*"' project.gcl | sed 's/@library("//;s/"//' | sort -u)

if [ -z "$CURRENT_LIBS" ]; then
    echo "ERROR: No @library declarations found in project.gcl"
    exit 1
fi

echo "Detected libraries:"
echo "$CURRENT_LIBS" | sed 's/^/  - /'
echo ""
```

---

## Step 3: Compare Versions

Compare current versions with latest versions:

```bash
echo "================================================================================"
echo "VERSION COMPARISON"
echo "================================================================================"
echo ""

UPDATES_NEEDED=false

# Check each library that's currently in project.gcl
for lib in $CURRENT_LIBS; do
    CURRENT_VERSION=$(grep "@library(\"$lib\"" project.gcl | sed -n "s/.*@library(\"$lib\", \"\([^\"]*\)\").*/\1/p")

    # Determine the latest version for this library
    case "$lib" in
        std)
            LATEST_VERSION="$STD_VERSION"
            ;;
        explorer)
            LATEST_VERSION="$EXPLORER_VERSION"
            ;;
        ai|algebra|kafka|sql|s3|finance|powerflow|opcua|useragent)
            LATEST_VERSION="$PRO_VERSION"
            ;;
        *)
            echo "  WARNING: Unknown library '$lib' - skipping"
            continue
            ;;
    esac

    # Compare versions
    if [ "$CURRENT_VERSION" = "$LATEST_VERSION" ]; then
        echo "  ✓ $lib: $CURRENT_VERSION (up-to-date)"
    else
        echo "  ↑ $lib: $CURRENT_VERSION → $LATEST_VERSION (update available)"
        UPDATES_NEEDED=true
    fi
done

echo ""

# Exit early if no updates needed
if [ "$UPDATES_NEEDED" = false ]; then
    echo "================================================================================"
    echo "✓ All libraries are already up-to-date. No changes needed."
    echo "================================================================================"
    exit 0
fi

```

**Early Exit**: If all versions match, command exits here. Otherwise, continues with confirmation.

---

## Step 4: User Confirmation

Before applying any changes, confirm with the user:

**Present Update Summary**:
```
===============================================================================
UPDATE SUMMARY
===============================================================================

The following library updates are available:

[List each library with current → latest version]
  std:      7.5.125-dev → 7.6.10-dev
  explorer: 7.5.3-dev   → 7.6.10-dev
  ai:       7.5.51-dev  → 7.6.10-dev

Actions that will be performed:
  1. Update project.gcl with new versions
  2. Run 'greycat install' to download libraries
  3. Run 'greycat-lang lint' to verify compatibility

Proceed with update?
```

**Use AskUserQuestion tool**:
```typescript
AskUserQuestion({
  questions: [{
    question: "Proceed with library update?",
    header: "Confirm Update",
    multiSelect: false,
    options: [
      {
        label: "Yes, update all libraries (Recommended)",
        description: "Update project.gcl, install libraries, and verify with lint"
      },
      {
        label: "Show me what will change in project.gcl first",
        description: "Display the exact changes before applying them"
      },
      {
        label: "Cancel - don't update",
        description: "Keep current library versions"
      }
    ]
  }]
})
```

**Handle User Response**:

**If "Cancel"**:
```bash
echo "================================================================================"
echo "UPDATE CANCELLED"
echo "================================================================================"
echo ""
echo "Library versions remain unchanged:"
for lib in $CURRENT_LIBS; do
    CURRENT_VERSION=$(grep "@library(\"$lib\"" project.gcl | sed -n "s/.*@library(\"$lib\", \"\([^\"]*\)\").*/\1/p")
    echo "  $lib: $CURRENT_VERSION"
done
exit 0
```

**If "Show me what will change"**:
```bash
echo "================================================================================"
echo "CHANGES TO project.gcl"
echo "================================================================================"
echo ""
echo "Current library declarations:"
grep '@library(' project.gcl
echo ""
echo "After update:"
for lib in $CURRENT_LIBS; do
    case "$lib" in
        std) NEW_VERSION="$STD_VERSION" ;;
        explorer) NEW_VERSION="$EXPLORER_VERSION" ;;
        ai|algebra|kafka|sql|s3|finance|powerflow|opcua|useragent) NEW_VERSION="$PRO_VERSION" ;;
    esac
    echo "@library(\"$lib\", \"$NEW_VERSION\");"
done
echo ""
# Ask again: "Proceed with these changes?"
```

**If "Yes, update all libraries"**:
```bash
echo "================================================================================"
echo "PROCEEDING WITH UPDATE"
echo "================================================================================"
echo ""
# Continue to Step 5
```

---

## Step 5: Update project.gcl

Update `project.gcl` with latest versions:

```bash
echo "Updating project.gcl..."
echo ""

# Update each library that's currently in project.gcl
for lib in $CURRENT_LIBS; do
    # Determine the latest version for this library
    case "$lib" in
        std)
            LATEST_VERSION="$STD_VERSION"
            ;;
        explorer)
            LATEST_VERSION="$EXPLORER_VERSION"
            ;;
        ai|algebra|kafka|sql|s3|finance|powerflow|opcua|useragent)
            LATEST_VERSION="$PRO_VERSION"
            ;;
        *)
            echo "  Skipping unknown library: $lib"
            continue
            ;;
    esac

    # Update the version in project.gcl using Edit tool
    # Find old version and replace with new version
    OLD_LINE=$(grep "@library(\"$lib\"" project.gcl)
    NEW_LINE="@library(\"$lib\", \"$LATEST_VERSION\");"

    # Use sed for in-place replacement
    sed -i "s/@library(\"$lib\", \"[^\"]*\")/@library(\"$lib\", \"$LATEST_VERSION\")/" project.gcl

    echo "  ✓ Updated $lib to $LATEST_VERSION"
done

echo ""
echo "Updated library declarations in project.gcl:"
grep '@library(' project.gcl
echo ""
```

---

## Step 6: Install Libraries

Download the updated libraries to `./lib` directory:

```bash
echo "================================================================================"
echo "INSTALLING LIBRARIES"
echo "================================================================================"
echo ""

greycat install

echo ""
echo "✓ Libraries installed to ./lib/"
echo ""
```

---

## Step 7: Verify with Lint

Ensure updated libraries don't introduce compatibility issues:

```bash
echo "================================================================================"
echo "VERIFYING WITH LINT"
echo "================================================================================"
echo ""

greycat-lang lint

LINT_EXIT_CODE=$?

echo ""
if [ $LINT_EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "✓ UPDATE COMPLETE - All libraries updated and verified successfully!"
    echo "================================================================================"
else
    echo "================================================================================"
    echo "⚠ UPDATE COMPLETE - But lint found errors that need to be fixed"
    echo "================================================================================"
    echo ""
    echo "The libraries were updated, but there are compatibility issues."
    echo "Please review the lint errors above and fix them."
fi

echo ""
echo "Updated libraries:"
for lib in $CURRENT_LIBS; do
    NEW_VERSION=$(grep "@library(\"$lib\"" project.gcl | sed -n "s/.*@library(\"$lib\", \"\([^\"]*\)\").*/\1/p")
    echo "  $lib: $NEW_VERSION"
done
echo ""
```

---

## Adding New Libraries

To add a new library to your project:

### Step 1: Add Library Declaration

Edit `project.gcl` and add the library declaration:

```gcl
// Example: Adding Kafka support
@library("kafka", "7.6.10-dev");
```

### Step 2: Run Update Command

Run this command - it will automatically detect and update the new library:

```bash
/upgrade
```

### Step 3: Verify Installation

```bash
# Check library is installed
ls -la lib/ | grep kafka

# Verify lint passes
greycat-lang lint

# Test the library
greycat serve
```

---

## Library Version Channels

GreyCat libraries are available in two channels:

### Dev Channel (Recommended for Development)
- **URL**: `https://get.greycat.io/files/{library}/dev/latest`
- **Stability**: Latest features, may have breaking changes
- **Use When**: Active development, testing new features
- **Example**: `@library("std", "7.6.10-dev")`

### Release Channel (Production)
- **URL**: `https://get.greycat.io/files/{library}/releases/latest`
- **Stability**: Stable, production-ready
- **Use When**: Production deployments
- **Example**: `@library("std", "7.5.0")`

**This Command Uses**: Dev channel (`/dev/latest`) by default

To switch to release channel, manually edit the curl commands to use `/releases/latest` instead of `/dev/latest`.

---

## Troubleshooting

### Lint Fails After Update

**Symptom**: `greycat-lang lint` reports errors after update

**Causes**:
1. Breaking API changes in new library version
2. Deprecated function usage
3. Type signature changes

**Solutions**:

**A. Check Changelog**:
```bash
# Visit GreyCat documentation for breaking changes
# https://doc.greycat.io/changelog
```

**B. Review Lint Errors**:
```bash
greycat-lang lint 2>&1 | tee lint-errors.txt

# Look for patterns:
# - "function not found" → API renamed/removed
# - "type mismatch" → Signature changed
# - "unknown decorator" → Decorator deprecated
```

**C. Rollback if Needed**:
```bash
# Revert project.gcl to previous version
git checkout project.gcl

# Reinstall old versions
greycat install

# Verify
greycat-lang lint
```

**D. Update Code Incrementally**:
```bash
# Update one library at a time
# Edit project.gcl manually to update only "std"
greycat install
greycat-lang lint

# If passes, update next library
# Edit project.gcl to update "explorer"
greycat install
greycat-lang lint

# Continue until all updated or issue found
```

### Library Fetch Fails

**Symptom**: Curl returns empty or error

**Solutions**:

```bash
# 1. Check internet connection
ping get.greycat.io

# 2. Test URL manually
curl -v "https://get.greycat.io/files/std/dev/latest"

# 3. Check if library exists
# Visit https://get.greycat.io in browser

# 4. Try alternative library for version
# If std fails, use explorer version as fallback (they match)
```

### Unknown Library Warning

**Symptom**: `WARNING: Unknown library 'xyz' - skipping`

**Cause**: Custom or third-party library not in standard list

**Solution**: Safe to ignore. Custom libraries won't auto-update. Update manually:

```gcl
// In project.gcl
@library("custom-lib", "1.2.3");  // Update version manually
```

### Installation Fails

**Symptom**: `greycat install` fails

**Solutions**:

```bash
# 1. Clear lib directory
rm -rf lib/
greycat install

# 2. Check disk space
df -h .

# 3. Check permissions
ls -la lib/

# 4. Try manual download
curl "https://get.greycat.io/files/std/dev/7.6.10-dev/std-7.6.10-dev.jar" -o lib/std.jar
```

---

## Migration Guide

### Handling Breaking Changes

When lint fails after update, follow this process:

**1. Identify Breaking Changes**

Common breaking change patterns:

```gcl
// BEFORE (old version)
fn process(data: String): Result {
    return DataService::parse(data);
}

// AFTER (new version - API renamed)
fn process(data: String): Result {
    return DataService::parseString(data);  // Function renamed
}
```

**2. Search for Deprecated Usage**

```bash
# Find all usages of deprecated function
grep -r "DataService::parse" backend/ --include="*.gcl"

# Replace across all files
find backend/ -name "*.gcl" -exec sed -i 's/DataService::parse/DataService::parseString/g' {} \;

# Verify
greycat-lang lint
```

**3. Test Thoroughly**

```bash
# Run all tests
greycat test

# Manual testing
greycat serve
# Test critical paths in UI
```

**4. Commit Changes**

```bash
git add project.gcl lib/
git commit -m "Update GreyCat libraries to latest versions

- std: 7.5.0-dev → 7.6.10-dev
- explorer: 7.5.0-dev → 7.6.10-dev
- ai: 7.5.0-dev → 7.6.10-dev

Breaking changes:
- Updated DataService::parse → DataService::parseString
- Updated deprecated @tag decorator usage"

git push
```

---

## Version Pinning (Advanced)

For production stability, pin specific versions instead of using `latest`:

### Option 1: Manual Version Pinning

```gcl
// project.gcl
@library("std", "7.6.10-dev");        // Pinned to specific version
@library("explorer", "7.6.10-dev");   // Pinned
@library("ai", "7.6.10-dev");         // Pinned
```

Update manually when ready to upgrade.

### Option 2: Version Lock File

Create `.greycat-lock` to track versions:

```json
{
  "locked": "2024-01-15T10:30:00Z",
  "libraries": {
    "std": "7.6.10-dev",
    "explorer": "7.6.10-dev",
    "ai": "7.6.10-dev"
  }
}
```

Update command can respect lock file (future enhancement).

---

## Success Criteria

✓ **Latest versions fetched** from get.greycat.io
✓ **Current libraries detected** in project.gcl
✓ **Version comparison complete** (or early exit if up-to-date)
✓ **project.gcl updated** with latest versions
✓ **Libraries installed** via `greycat install`
✓ **Lint passes** with 0 errors (or warnings documented)

---

## Verification Commands

```bash
# 1. Check updated versions in project.gcl
grep '@library(' project.gcl

# 2. Verify libraries installed
ls -la lib/

# 3. Check for lint issues
greycat-lang lint

# 4. Run tests
greycat test

# 5. Start server to verify runtime
greycat serve
```

---

## Best Practices

### When to Update

**✅ Good Times**:
- Start of development sprint
- After completing major features
- Monthly maintenance window
- Before planning new features (get latest capabilities)

**❌ Bad Times**:
- During active feature development
- Right before production deployment
- When team is unavailable to fix issues
- During critical bug fixes

### Update Strategy

**Option A: Update All (Aggressive)**
```bash
# Update all libraries at once
/upgrade

# Fix all compatibility issues
greycat-lang lint
# ... fix errors

greycat test
# ... fix failing tests
```

**Option B: Update Incrementally (Safe)**
```bash
# 1. Update only std library
# Edit project.gcl manually
greycat install && greycat-lang lint

# 2. Update explorer
# Edit project.gcl manually
greycat install && greycat-lang lint

# 3. Update pro libraries
# Edit project.gcl manually
greycat install && greycat-lang lint
```

**Option C: Test Before Update (Safest)**
```bash
# 1. Create test branch
git checkout -b update-greycat-libs

# 2. Run update
/upgrade

# 3. Fix issues
greycat-lang lint
greycat test

# 4. Manual testing
greycat serve

# 5. Merge if successful
git checkout main
git merge update-greycat-libs
```

### Monitoring for Updates

Check for new versions monthly:

```bash
# Quick version check (without updating)
STD_LATEST=$(curl -s "https://get.greycat.io/files/core/dev/latest" | cut -d'/' -f2)
STD_CURRENT=$(grep '@library("std"' project.gcl | sed -n 's/.*"\([^"]*\)".*/\1/p')

echo "Current std: $STD_CURRENT"
echo "Latest std:  $STD_LATEST"

if [ "$STD_CURRENT" != "$STD_LATEST" ]; then
    echo "⚠ Update available!"
else
    echo "✓ Up to date"
fi
```

---

## Example Output

```
================================================================================
FETCHING LATEST LIBRARY VERSIONS
================================================================================

Core libraries:
  std:      7.6.10-dev
  explorer: 7.6.10-dev

Pro libraries (all share same version):
  Version: 7.6.10-dev
  Libraries: ai, algebra, kafka, sql, s3, finance, powerflow, opcua, useragent

================================================================================
CURRENT LIBRARIES IN project.gcl
================================================================================

Detected libraries:
  - ai
  - explorer
  - std

================================================================================
VERSION COMPARISON
================================================================================

  ↑ std: 7.5.125-dev → 7.6.10-dev (update available)
  ↑ explorer: 7.5.3-dev → 7.6.10-dev (update available)
  ↑ ai: 7.5.51-dev → 7.6.10-dev (update available)

================================================================================
UPDATE SUMMARY
================================================================================

The following library updates are available:

  std:      7.5.125-dev → 7.6.10-dev
  explorer: 7.5.3-dev   → 7.6.10-dev
  ai:       7.5.51-dev  → 7.6.10-dev

Actions that will be performed:
  1. Update project.gcl with new versions
  2. Run 'greycat install' to download libraries
  3. Run 'greycat-lang lint' to verify compatibility

[User is prompted to confirm via AskUserQuestion]
Options:
  → Yes, update all libraries (Recommended)
  → Show me what will change in project.gcl first
  → Cancel - don't update

[User selects: "Yes, update all libraries (Recommended)"]

================================================================================
PROCEEDING WITH UPDATE
================================================================================

Updating project.gcl...

  ✓ Updated std to 7.6.10-dev
  ✓ Updated explorer to 7.6.10-dev
  ✓ Updated ai to 7.6.10-dev

Updated library declarations in project.gcl:
@library("std", "7.6.10-dev");
@library("explorer", "7.6.10-dev");
@library("ai", "7.6.10-dev");

================================================================================
INSTALLING LIBRARIES
================================================================================

Downloading std-7.6.10-dev.jar...
Downloading explorer-7.6.10-dev.jar...
Downloading ai-7.6.10-dev.jar...

✓ Libraries installed to ./lib/

================================================================================
VERIFYING WITH LINT
================================================================================

Linting backend files...
✓ All files passed

================================================================================
✓ UPDATE COMPLETE - All libraries updated and verified successfully!
================================================================================

Updated libraries:
  std: 7.6.10-dev
  explorer: 7.6.10-dev
  ai: 7.6.10-dev
```

---

## Notes

- **No Code Changes**: This command only updates library versions, not your code
- **Data Preservation**: Data in `gcdata/` remains untouched
- **Rollback Available**: Use git to revert if needed
- **Dev Channel**: Uses latest dev versions (more features, less stable)
- **Shared Versions**: All pro libraries use the same version number
