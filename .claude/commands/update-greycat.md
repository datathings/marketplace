---
description: Update GreyCat skill to latest library versions
allowed-tools: Bash(*), Read, Edit, Write, Grep, Glob
---

# Update GreyCat Skill

This command upgrades the GreyCat skill by syncing with the latest library versions.

**Working directory:** `plugins/greycat/`

## Process Overview

1. Install the latest GreyCat libraries from `project.gcl`
2. Delete all old GCL files from the skill to ensure a clean state
3. Replace all GCL files in the skill with the newly installed versions
4. Update library versions in skill markdown documentation
5. Analyze function signatures and update skill documentation
6. Re-package the skill

## Step 1: Install Latest Libraries

Run `greycat install` to download the latest versions specified in `project.gcl` to the `./lib` directory.

```bash
cd plugins/greycat
greycat install
```

## Step 2: Clean Old GCL Files

**IMPORTANT**: First, delete all existing `.gcl` files from the skill folder to remove any stale files:

```bash
cd plugins/greycat
find ./skills/greycat/references -type f -name "*.gcl" -delete
```

This ensures:
- No old/renamed files remain
- Clean state before copying new files
- Removed libraries don't leave behind files

## Step 3: Sync GCL Files from lib/ to skill

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

## Step 4: Update Library Versions in Markdown Files

Read `project.gcl` to extract all library versions (e.g., `@library("kafka", "7.5.68-dev")`).

For each library's markdown documentation file in `skills/greycat/references/`:
- Find the installation section with `@library("library-name", "version")`
- Update the version to match what's in `project.gcl`

Files to update:
- `skills/greycat/references/kafka/kafka.md`
- `skills/greycat/references/sql/postgres.md`
- `skills/greycat/references/s3/s3.md`
- `skills/greycat/references/finance/finance.md`
- `skills/greycat/references/powerflow/powerflow.md`
- `skills/greycat/references/opcua/opcua.md`
- `skills/greycat/references/useragent/useragent.md`
- `skills/greycat/references/ai/README.md`
- `skills/greycat/references/algebra/README.md` (if exists)
- `skills/greycat/references/std/README.md`
- `skills/greycat/references/LIBRARIES.md`
- `skills/greycat/references/frontend.md` (web SDK version)
- `skills/greycat/SKILL.md` (main skill file - check for version references)

### Verify Web SDK URL Exists

After updating `frontend.md` with the new @greycat/web SDK version, verify the URL exists:

```bash
# Extract the SDK URL from frontend.md and verify it exists
SDK_URL=$(grep -o 'https://get.greycat.io/files/sdk/web/dev/[^"]*' skills/greycat/references/frontend.md | head -1)
echo "Verifying SDK URL: $SDK_URL"
# Note: Server doesn't support HEAD requests, use GET with output to /dev/null
HTTP_CODE=$(curl -so /dev/null -w "%{http_code}" "$SDK_URL")
if [ "$HTTP_CODE" = "200" ]; then
    echo "SDK URL verified: $SDK_URL (HTTP $HTTP_CODE)"
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

## Step 5: Analyze Function Signatures

For each newly copied `.gcl` file:
1. Read the file and extract all function signatures, type definitions, and method signatures
2. Compare with the existing markdown documentation
3. Identify any new functions, changed signatures, or deprecated functions
4. Update the markdown documentation to reflect the changes:
   - Add documentation for new functions
   - Update parameter types/names for changed signatures
   - Mark deprecated functions if they were removed
   - Ensure all examples are still valid

Focus on documenting:
- Public API functions and methods
- Type definitions and their fields
- Method signatures with parameters and return types
- Any breaking changes or new features

## Step 6: Re-package the Skill

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
1. All old GCL files are deleted from skills/greycat/references/
2. All GCL files are copied from lib/ to skills/greycat/references/
3. All library versions in markdown files match project.gcl
4. No old version references remain in skills/greycat/ (run: `grep -r "7\.[0-9]\.[0-9]*-dev" skills/greycat/` to verify)
5. All function signatures are documented and accurate
6. The skill is successfully re-packaged
7. No errors occur during the process
