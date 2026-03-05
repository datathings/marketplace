---
description: Update GreyCat skill to latest library versions
allowed-tools: Bash(*), Read, Edit, Write, Grep, Glob
---

# Update GreyCat Skill

This command upgrades the GreyCat skill by syncing with the latest library versions.

**Working directory:** `plugins/greycat/`

## Process Overview

1. Fetch the latest library versions from get.greycat.io
2. Compare with current versions in `project.gcl` (exit early if all up-to-date)
3. Update `project.gcl` with the latest versions
4. Install the latest GreyCat libraries from updated `project.gcl`
5. Delete all old GCL files from the skill to ensure a clean state
6. Replace all GCL files in the skill with the newly installed versions
7. Update all `@library` declarations in skill markdown files
8. Analyze function signatures and update skill documentation
9. Re-package the skill

## Step 1: Fetch Latest Library Versions

Fetch the latest library versions from get.greycat.io:

```bash
cd plugins/greycat

# Fetch core libraries
STD_VERSION=$(curl -s "https://get.greycat.io/files/core/dev/latest" | cut -d'/' -f2)
EXPLORER_VERSION=$(curl -s "https://get.greycat.io/files/explorer/dev/latest" | cut -d'/' -f2)

# If std version is empty, use explorer version as fallback (they typically match)
if [ -z "$STD_VERSION" ]; then
  STD_VERSION="$EXPLORER_VERSION"
  echo "Note: std version endpoint returned empty, using explorer version as fallback"
fi

# Fetch pro libraries version (all pro libraries share the same version)
# We only need to check one library (algebra) to get the version for all
PRO_VERSION=$(curl -s "https://get.greycat.io/files/algebra/dev/latest" | cut -d'/' -f2)

# Fetch web SDK version
WEB_SDK_VERSION=$(curl -s "https://get.greycat.io/files/sdk/web/dev/latest" | cut -d'/' -f2)

echo "Latest versions:"
echo "  std: $STD_VERSION"
echo "  explorer: $EXPLORER_VERSION"
echo "  pro libraries: $PRO_VERSION"
echo "  web SDK: $WEB_SDK_VERSION"
```

**Note:** All pro libraries (ai, algebra, kafka, sql, s3, finance, powerflow, opcua, useragent) share the same version, so we only need to fetch one version.

## Step 2: Compare with Current Versions

Extract the current versions from `project.gcl` and compare with the latest versions:

```bash
# Extract current versions from project.gcl
CURRENT_STD=$(grep '@library("std"' project.gcl | sed -n 's/.*@library("std", "\([^"]*\)").*/\1/p')
CURRENT_EXPLORER=$(grep '@library("explorer"' project.gcl | sed -n 's/.*@library("explorer", "\([^"]*\)").*/\1/p')
CURRENT_PRO=$(grep '@library("algebra"' project.gcl | sed -n 's/.*@library("algebra", "\([^"]*\)").*/\1/p')

echo ""
echo "Current versions in project.gcl:"
echo "  std: $CURRENT_STD"
echo "  explorer: $CURRENT_EXPLORER"
echo "  pro libraries: $CURRENT_PRO"

# Compare versions - if ALL versions are the same, exit early
if [ "$CURRENT_STD" = "$STD_VERSION" ] && \
   [ "$CURRENT_EXPLORER" = "$EXPLORER_VERSION" ] && \
   [ "$CURRENT_PRO" = "$PRO_VERSION" ]; then
    echo ""
    echo "✓ All libraries are already up-to-date. No changes needed."
    exit 0
fi

# If any version is different, show which ones need updating
echo ""
echo "Updates needed:"
[ "$CURRENT_STD" != "$STD_VERSION" ] && echo "  std: $CURRENT_STD → $STD_VERSION"
[ "$CURRENT_EXPLORER" != "$EXPLORER_VERSION" ] && echo "  explorer: $CURRENT_EXPLORER → $EXPLORER_VERSION"
[ "$CURRENT_PRO" != "$PRO_VERSION" ] && echo "  pro libraries: $CURRENT_PRO → $PRO_VERSION"
echo ""
echo "Proceeding with update..."
```

**Early Exit:** If all versions match, the command will exit here. If any version is different, it will continue with the update process.

## Step 3: Update project.gcl

Update `project.gcl` with the latest versions:

```bash
# Update std library
sed -i "s/@library(\"std\", \"[^\"]*\")/@library(\"std\", \"$STD_VERSION\")/" project.gcl

# Update explorer library
sed -i "s/@library(\"explorer\", \"[^\"]*\")/@library(\"explorer\", \"$EXPLORER_VERSION\")/" project.gcl

# Update all pro libraries with the same version
for lib in ai algebra kafka sql s3 finance powerflow opcua useragent; do
  sed -i "s/@library(\"$lib\", \"[^\"]*\")/@library(\"$lib\", \"$PRO_VERSION\")/" project.gcl
done

echo "Updated project.gcl with latest versions"
cat project.gcl
```

## Step 4: Install Latest Libraries

Run `greycat install` to download the latest versions specified in `project.gcl` to the `./lib` directory.

```bash
greycat install
```

## Step 5: Clean Old GCL Files

**IMPORTANT**: First, delete all existing `.gcl` files from the skill folder to remove any stale files:

```bash
find ./skills/greycat/references -type f -name "*.gcl" -delete
```

This ensures:
- No old/renamed files remain
- Clean state before copying new files
- Removed libraries don't leave behind files

## Step 6: Sync GCL Files from lib/ to skill

Copy all `.gcl` files from `./lib/*/` to `skills/greycat/references/*/`:

For each library directory in `./lib`:
- Copy all `.gcl` files to the corresponding `skills/greycat/references/` subdirectory
- Preserve the file structure

Libraries to sync:
- std (core.gcl, runtime.gcl, util.gcl, io.gcl)
- ai (all llm_*.gcl files)
- algebra (ml.gcl, compute.gcl, nn.gcl, patterns.gcl, transforms.gcl, etc.)
- kafka (kafka.gcl)
- sql (postgres.gcl)
- s3 (s3.gcl)
- finance (finance.gcl)
- powerflow (powerflow.gcl)
- opcua (opcua.gcl)
- useragent (useragent.gcl)

## Step 7: Update All @library Declarations in Skill Files

Update all `@library` declarations in the skill markdown files to match the versions in `project.gcl`:

```bash
# Update all @library declarations in skill files using the versions we fetched

# Update std library
find ./skills/greycat -type f -name "*.md" -exec sed -i "s/@library(\"std\", \"[^\"]*\")/@library(\"std\", \"$STD_VERSION\")/g" {} \;

# Update explorer library
find ./skills/greycat -type f -name "*.md" -exec sed -i "s/@library(\"explorer\", \"[^\"]*\")/@library(\"explorer\", \"$EXPLORER_VERSION\")/g" {} \;

# Update all pro libraries with the same version
for lib in ai algebra kafka sql s3 finance powerflow opcua useragent; do
  find ./skills/greycat -type f -name "*.md" -exec sed -i "s/@library(\"$lib\", \"[^\"]*\")/@library(\"$lib\", \"$PRO_VERSION\")/g" {} \;
done

echo "Updated all @library declarations in skill files"
```

**Files that will be updated:**
- `skills/greycat/SKILL.md`
- `skills/greycat/references/kafka/kafka.md`
- `skills/greycat/references/sql/postgres.md`
- `skills/greycat/references/s3/s3.md`
- `skills/greycat/references/finance/finance.md`
- `skills/greycat/references/powerflow/powerflow.md`
- `skills/greycat/references/opcua/opcua.md`
- `skills/greycat/references/useragent/useragent.md`
- `skills/greycat/references/ai/llm.md`
- `skills/greycat/references/LIBRARIES.md`
- Any other markdown files containing `@library` declarations

### Update and Verify Web SDK Version

Update the web SDK version in `frontend.md` and verify the URL exists:

```bash
# Update the web SDK version in frontend.md
# The SDK version pattern in frontend.md should match: https://get.greycat.io/files/sdk/web/dev/X.Y/X.Y.Z-dev.tgz
MAJOR_MINOR=$(echo $WEB_SDK_VERSION | cut -d'.' -f1,2)
SDK_URL="https://get.greycat.io/files/sdk/web/dev/$MAJOR_MINOR/$WEB_SDK_VERSION.tgz"

echo "Updating frontend.md with web SDK version: $WEB_SDK_VERSION"
echo "Expected SDK URL: $SDK_URL"

# Verify the URL exists
HTTP_CODE=$(curl -so /dev/null -w "%{http_code}" "$SDK_URL")
if [ "$HTTP_CODE" = "200" ]; then
    echo "SDK URL verified: $SDK_URL (HTTP $HTTP_CODE)"
    # Update the version in frontend.md
    # Update version references and URLs in frontend.md as needed
else
    echo "WARNING: SDK URL does not exist! HTTP $HTTP_CODE"
fi
```

**If the URL doesn't exist (not HTTP 200):**

1. First, check the available dev versions:
```bash
curl -s "https://get.greycat.io/files/sdk/web/dev/"
```

2. If the major version directory exists (e.g., 7.6/), check for available .tgz files:
```bash
curl -s "https://get.greycat.io/files/sdk/web/dev/7.6/"
```

3. If no suitable dev version is found, navigate to the main SDK directory to find the latest stable version:
```bash
curl -s "https://get.greycat.io/files/sdk/web/"
```

4. Or visit https://get.greycat.io/files/sdk/web in a browser to browse all available versions interactively.

5. Once you find the correct version, update `frontend.md` with the correct URL pattern:
   - Dev versions: `https://get.greycat.io/files/sdk/web/dev/X.Y/X.Y.Z-dev.tgz`
   - Stable versions: `https://get.greycat.io/files/sdk/web/X.Y/X.Y.Z.tgz`

**Suggest to the user** what version to use based on what's available, matching the major.minor version of the libraries in `project.gcl`.

## Step 8: Analyze Function Signatures (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the GCL analysis and
documentation rewrite. This keeps the large GCL file content out of the main context.

Provide the subagent with:
- Path to updated GCL files: `plugins/greycat/skills/greycat/references/` (just synced)
- Path to existing markdown docs in the same directory
- Old versions and new versions (from Steps 1–2 output)

The subagent should, for each updated `.gcl` file:
1. Extract all function signatures, type definitions, and method signatures
2. Compare with the existing markdown documentation
3. Identify new functions, changed signatures, or removed functions
4. Update the markdown documentation:
   - Add documentation for new functions
   - Update parameter types/names for changed signatures
   - Remove functions that no longer exist
   - Ensure all examples are still valid

Focus on documenting:
- Public API functions and methods
- Type definitions and their fields
- Method signatures with parameters and return types
- Any breaking changes or new features

Keep SKILL.md **concise and dense** — it is always loaded into context, so every line must
earn its place. API details belong in the per-library reference files, not in SKILL.md.

**The subagent should return a summary** of: libraries updated, functions added, functions
removed, signatures changed, and any breaking changes — so the main context can report
the outcome.

## Step 9: Re-package the Skill

Run the main packaging script from the repository root to create the updated `.skill` file:

```bash
# From repo root
./package.sh greycat

# Or with explicit flag
./package.sh --skill greycat
```

This will create `skills/greycat.skill` with all the updated files.

## Important Notes

- Always backup the current skill before running this command
- Review all documentation changes to ensure accuracy
- Test the updated skill before distribution
- Check for breaking changes in library updates
- Verify that all examples in the documentation still work with the new versions

## Success Criteria

The upgrade is complete when:
1. Latest versions are successfully fetched from get.greycat.io
2. Current versions are compared with latest versions (or early exit if already up-to-date)
3. `project.gcl` is updated with the latest versions
4. All old GCL files are deleted from skills/greycat/references/
5. All GCL files are copied from lib/ to skills/greycat/references/
6. All `@library` declarations in skill markdown files are updated to match the latest versions
7. Web SDK version is verified and updated in frontend.md
8. All function signatures are documented and accurate
9. The skill is successfully re-packaged
10. No errors occur during the process

**Verification commands:**
```bash
# Verify all @library declarations are updated (should show no old versions)
grep -r "@library" skills/greycat/ | grep -v "$STD_VERSION" | grep -v "$EXPLORER_VERSION" | grep -v "$PRO_VERSION"

# Verify project.gcl versions
cat project.gcl
```
