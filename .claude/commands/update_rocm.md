# Update ROCm Skill to Latest Version

This command updates the ROCm library to the latest version and synchronizes all skill files with the updated API.

**Working directory:** `plugins/rocm/`

## Repository Structure

```
plugins/rocm/
├── skills/rocm/              # Skill content (SKILL.md and references/)
├── ROCm/                     # ROCm meta-repo clone (NOT a submodule)
├── rocm-examples/            # ROCm examples repo clone (NOT a submodule)
└── .claude-plugin/plugin.json  # Plugin manifest
```

The `ROCm/` and `rocm-examples/` directories are **standalone git clones** from
https://github.com/ROCm/ROCm and https://github.com/ROCm/rocm-examples respectively —
updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Libraries to Latest Tag

```bash
cd plugins/rocm/ROCm

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current ROCm tag: $CURRENT_TAG"

# Stable ROCm releases use rocm-#.#.# format
LATEST_TAG=$(git tag --sort=-version:refname | grep "^rocm-[0-9]" | head -n 1)
echo "Latest ROCm tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50
fi

git checkout "$LATEST_TAG"
echo "ROCm now on: $(git describe --tags)"

cd ../rocm-examples

git fetch --tags

CURRENT_EX_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
LATEST_EX_TAG=$(git tag --sort=-version:refname | grep "^rocm-[0-9]" | head -n 1)
echo "Examples: $CURRENT_EX_TAG -> $LATEST_EX_TAG"

git checkout "$LATEST_EX_TAG"
echo "rocm-examples now on: $(git describe --tags)"

cd ../../..
```

- Latest tags viewable at: https://github.com/ROCm/ROCm/tags
- Stable release tags use format: `rocm-#.#.#` (e.g., `rocm-7.2.0`, `rocm-7.3.0`)

### 2–6. Analyze API Changes & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large source content out of the main context.

Provide the subagent with:
- Path to ROCm meta-repo: `plugins/rocm/ROCm/`
- Path to examples repo: `plugins/rocm/rocm-examples/`
- Path to skill references: `plugins/rocm/skills/rocm/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan the examples and docs for:
- New HIP runtime functions or changed signatures in `rocm-examples/HIP-Basic/`
- New library examples in `rocm-examples/Libraries/` (new rocBLAS routines, rocFFT modes, etc.)
- Changes in `rocm-examples/Programming-Guide/` for new programming patterns
- New application examples in `rocm-examples/Applications/`
- Changelog entries in `ROCm/CHANGELOG.md` or `ROCm/docs/`

**Update Skill Documentation Files** in `skills/rocm/references/`:
- **`api-hip-core.md`** — HIP runtime: device management, memory, kernel launch, streams, events, synchronization, occupancy, warp intrinsics, atomics
- **`api-hip-math.md`** — HIP math intrinsics: float/double/half precision, fast approximations, complex math
- **`api-libraries.md`** — rocBLAS/hipBLAS/hipBLASLt, rocFFT/hipFFT, rocRAND, rocSOLVER/hipSOLVER, rocSPARSE, rocWMMA, rocPRIM/hipCUB/rocThrust
- **`api-profiling.md`** — rocProfiler-SDK programmatic API, rocprof CLI, rocm-smi, rocGDB debugging, performance tuning
- **`workflows.md`** — Complete working examples: vector add, SAXPY, GEMM, rocFFT, multi-GPU, streaming, CMake build, CUDA-to-HIP porting

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — it is always loaded into context, so every line
  must earn its place; API details belong in `references/` files, not here
- Update the ROCm version/tag in the documentation
- Add notes about newly supported features or changed default behaviors
- Note AMD wavefront size, rocBLAS column-major convention, async transfer requirements

**Remove Deprecated Content:**
- Search documentation files for functions/APIs removed or deprecated in the new release
- Remove deprecated references completely; don't just mark them
- Note any breaking changes in SKILL.md

**Validate Changes:**
- Cross-reference all documented functions against the new examples
- Verify signatures and argument patterns are consistent with new examples
- Ensure all newly introduced library features are documented

**The subagent should return a summary** of: new features added, deprecated APIs removed,
signature changes, and any breaking changes — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
./package.sh rocm
```

This creates `skills/rocm.skill`.

## Key Considerations

- **Source of truth**: Derive patterns and APIs from `rocm-examples/` (official AMD examples), not secondary docs
- **AMD wavefront size is 64** — never hardcode 32; always use `warpSize` built-in
- **rocBLAS is column-major** — row-major C arrays require swapping A/B and transposing M/N
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them
- **New features**: Major new library features should get workflow examples
- **Backward compatibility**: Note any breaking changes or behavior differences in SKILL.md
- **HIP portability**: HIP code should compile on both AMD (ROCm) and NVIDIA (CUDA); note any AMD-only APIs

## Success Criteria

- ✅ Both ROCm and rocm-examples updated to latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API patterns from examples are documented
- ✅ All removed/deprecated functions are deleted from documentation
- ✅ All signatures and patterns match new examples exactly
- ✅ Code examples in workflows.md are valid for the new version
- ✅ Package builds successfully with `./package.sh rocm`
