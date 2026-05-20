# Update llama.cpp Skill to Latest API

This command updates the llama.cpp library to the latest version and synchronizes all skill files with the updated API.

**Working directory:** `plugins/llamacpp/`

## Repository Structure

```
plugins/llamacpp/
├── skills/llamacpp/           # Skill content (SKILL.md and references/)
├── llama.cpp/                 # Upstream repo clone (NOT a submodule)
└── package.sh                 # Packaging script
```

The `llama.cpp/` directory is a **standalone git clone** from https://github.com/ggml-org/llama.cpp - updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update llama.cpp Library to Latest Tag

```bash
cd plugins/llamacpp/llama.cpp

# Fetch all tags from upstream
git fetch --tags

# Check current version
CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Get latest b-series tag (stable releases use b#### format)
LATEST_TAG=$(git tag --sort=-version:refname | grep "^b" | head -n 1)
echo "Latest tag: $LATEST_TAG"

# Preview changes between versions
if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50

  # Look for API-related changes
  echo -e "\nAPI-related commits:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" --grep="API\|deprecat\|llama_" -i
fi

# Checkout latest tag
git checkout "$LATEST_TAG"

# Verify
echo "Now on: $(git describe --tags)"

cd ..
```

- Latest tags viewable at: https://github.com/ggml-org/llama.cpp/tags
- Stable release tags use format: `b####` (e.g., `b7617`, `b7572`)

### 2–6. Analyze & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large header content out of the main context.

Provide the subagent with:
- Path to headers: `plugins/llamacpp/llama.cpp/include/` (`llama.h`, `llama-cpp.h`)
- Path to skill references: `plugins/llamacpp/skills/llamacpp/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan `llama.h` and `llama-cpp.h` for:
- New functions added since last update
- Deprecated functions (marked with deprecation comments or attributes)
- Modified function signatures (parameter changes, return type changes)
- New constants, enums, or structs

**Update Skill Documentation Files** in `skills/llamacpp/references/`:
- **`api-core.md`** - Initialization, parameters, model loading functions
- **`api-model-info.md`** - Model properties and architecture detection
- **`api-context.md`** - Context, memory (KV cache), state management
- **`api-inference.md`** - Batch operations, inference, tokenization, chat
- **`api-sampling.md`** - Sampling strategies (XTC, DRY, temperature, etc.)
- **`api-advanced.md`** - LoRA adapters, performance, training
- **`workflows.md`** - Working examples and usage patterns

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — it is always loaded into context, so every line
  must earn its place; API details belong in `references/` files, not here
- Update the llama.cpp version/tag
- Update the function count in the overview
- Add notes about newly supported features
- Update the "Quick Function Lookup" section with any critical new functions

**Remove Deprecated Content:**
- Search all documentation files for functions marked as deprecated in the headers
- Remove or replace deprecated function references
- Update migration notes for breaking changes

**Validate Changes:**
- Cross-reference all documented functions against header files
- Ensure function signatures match exactly (parameter types, names, return types)
- Verify all new public API functions are documented

**The subagent should return a summary** of: functions added, functions removed, signatures
changed, and any breaking changes — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
# From repo root
./package.sh llamacpp
```

This creates `skills/llamacpp.skill`.

## Key Considerations

- **Function Signatures**: Always verify exact signatures - parameter order, types, and names must match
- **Deprecation Handling**: Remove deprecated functions completely; don't just mark them
- **New Features**: Major new features should get workflow examples
- **Backward Compatibility**: Note any breaking changes in SKILL.md
- **Testing**: Ensure all documented functions actually exist in the updated headers
- **Completeness**: All public API functions should be documented somewhere

## Success Criteria

- ✅ llama.cpp is updated to the latest stable tag
- ✅ SKILL.md documents which tag version it corresponds to
- ✅ All new functions from headers are documented
- ✅ All deprecated functions are removed from documentation
- ✅ All function signatures match the headers exactly
- ✅ Code examples in workflows.md are updated and valid
- ✅ Package builds successfully with `./package.sh llamacpp`
