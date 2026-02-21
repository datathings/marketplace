# Update CUDA Skill to Latest Version

This command updates the CUDA skill libraries to the latest versions and synchronizes all skill files with the updated APIs.

**Working directory:** `plugins/cuda/`

## Repository Structure

```
plugins/cuda/
├── skills/cuda/              # Skill content (SKILL.md and references/)
├── cuda-samples/             # Upstream repo clone — https://github.com/NVIDIA/cuda-samples
├── CUDALibrarySamples/       # Upstream repo clone — git@github.com:NVIDIA/CUDALibrarySamples.git
└── .claude-plugin/plugin.json   # Plugin manifest
```

Both `cuda-samples/` and `CUDALibrarySamples/` are **standalone git clones** — updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Libraries to Latest Versions

```bash
# Update cuda-samples (uses vX.Y tag format, e.g. v13.1)
cd plugins/cuda/cuda-samples

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "cuda-samples current tag: $CURRENT_TAG"

LATEST_TAG=$(git tag --sort=-version:refname | grep "^v" | head -n 1)
echo "cuda-samples latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50
fi

git checkout "$LATEST_TAG"
echo "cuda-samples now on: $(git describe --tags)"

cd ../CUDALibrarySamples

# CUDALibrarySamples has no tags — pull latest main
git fetch origin
git log --oneline HEAD..origin/main | head -n 20
git checkout main
git pull origin main
echo "CUDALibrarySamples now on: $(git rev-parse --short HEAD)"

cd ../../..
```

### 2–6. Analyze & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and documentation rewrite. This keeps the large sample code out of the main context.

Provide the subagent with:
- Path to cuda-samples: `plugins/cuda/cuda-samples/Samples/`
- Path to CUDALibrarySamples: `plugins/cuda/CUDALibrarySamples/`
- Path to skill references: `plugins/cuda/skills/cuda/references/`
- Old and new versions (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan both repos for:
- New CUDA Runtime API functions or changed signatures in `cuda-samples/`
- New cuBLAS/cuFFT/cuSPARSE/cuRAND/cuSolver sample code in `CUDALibrarySamples/`
- New subdirectories representing new libraries (e.g., cuTENSOR, nvJPEG2K, NCCL)
- Deprecated or removed sample functions
- New CUDA Toolkit version features (e.g., new cooperative groups APIs, new memory types)

**Source of truth for each domain:**
- **CUDA Runtime**: `cuda-samples/Samples/0_Introduction/` and `cuda-samples/Samples/6_Performance/`
- **cuBLAS**: `CUDALibrarySamples/cuBLAS/` — read the `.cpp`/`.cu` files and `README.md`
- **cuFFT**: `CUDALibrarySamples/cuFFT/` — same
- **cuSPARSE**: `CUDALibrarySamples/cuSPARSE/` — same
- **cuRAND**: `CUDALibrarySamples/cuRAND/` — same
- **cuSolver**: `CUDALibrarySamples/cuSolver/` — same
- **Thrust**: `cuda-samples/Samples/` (search for Thrust usage)
- **Cooperative Groups**: `cuda-samples/Samples/` (search for cooperative_groups)

**Update Skill Documentation Files** in `skills/cuda/references/`:
- **`api-runtime.md`** — CUDA Runtime: device management, memory, streams, events, kernel launch, error handling
- **`api-cublas.md`** — cuBLAS: handle/stream, Level-1/2/3 ops, batched GEMM, TRSM, cuBLASLt
- **`api-cufft.md`** — cuFFT: plan creation, 1D/2D/3D/Many, stream binding, execution, transform types
- **`api-cusparse.md`** — cuSPARSE: CSR/COO/BSR formats, SpMM/SpMV/SpGEMM, three-step generic API
- **`api-curand.md`** — cuRAND: host API, device API (curandState), generator types, distributions
- **`api-cusolver.md`** — cuSolver: LU/QR/eigenvalue/SVD dense; sparse ILU/QR; generic X API
- **`api-thrust.md`** — Thrust: containers, execution policies, algorithms, fancy iterators
- **`api-cooperative-groups.md`** — Cooperative Groups: partitions, grid sync, collective ops
- **`workflows.md`** — Complete working examples (update CUDA version in compile flags)

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — always loaded into context; API details belong in `references/`
- Update `cuda-samples` version tag and `CUDALibrarySamples` commit date
- Add notes about newly supported CUDA features
- Update Quick Start example if syntax changed
- Update the API reference table if files were added/removed

**Remove Deprecated Content:**
- Remove functions/patterns no longer present in sample code
- Update compile flags (e.g., `--gpu-architecture`) for new toolkit versions
- Note any breaking changes in SKILL.md

**Validate Changes:**
- Cross-reference documented functions against actual sample `.cu`/`.cpp` files
- Ensure function signatures match exactly (parameter types, return types)
- Verify all new public API patterns are documented
- Confirm code examples use correct CUDA version syntax

**The subagent should return a summary** of: API additions, removals, changed signatures,
new libraries covered, and the updated CUDA toolkit version — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
# From repo root
./package.sh cuda
```

This creates `skills/cuda.skill`.

## Key Considerations

- **Source of truth**: Derive signatures from sample `.cu`/`.cpp` files, not secondary docs
- **Column-major storage**: All cuBLAS and cuSolver calls use Fortran column-major layout — always note this
- **cuFFT normalization**: Inverse FFTs must be divided by N manually — document in workflows
- **cuSPARSE three-step**: `_bufferSize` → `_preprocess` → `_execute` pattern must stay accurate
- **Cooperative launch**: `this_grid().sync()` requires `cudaLaunchCooperativeKernel`, not `<<<>>>`
- **Deprecation handling**: Remove deprecated APIs completely; don't just mark them
- **New libraries**: If `CUDALibrarySamples` gains new subdirectories, add new reference files

## Success Criteria

- ✅ cuda-samples updated to latest stable tag
- ✅ CUDALibrarySamples updated to latest main
- ✅ SKILL.md documents which versions it corresponds to
- ✅ All new API patterns from sample code are documented
- ✅ All removed/deprecated patterns are deleted from documentation
- ✅ All signatures match sample code exactly
- ✅ Code examples in workflows.md compile with updated CUDA version
- ✅ Package builds successfully with `./package.sh cuda`
