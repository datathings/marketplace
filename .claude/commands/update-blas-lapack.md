# Update BLAS/LAPACK Skill to Latest Library Version

This command updates the LAPACK library to the latest version and synchronizes all skill files with the updated CBLAS and LAPACKE APIs.

**Working directory:** `plugins/blas_lapack/`

## Repository Structure

```
plugins/blas_lapack/
├── skills/blas_lapack/          # Skill content (SKILL.md and references/)
├── lapack/                      # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json   # Plugin manifest
```

The `lapack/` directory is a **standalone git clone** from https://github.com/Reference-LAPACK/lapack - updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update LAPACK Library to Latest Tag

```bash
cd plugins/blas_lapack/lapack

# Fetch all tags from upstream
git fetch --tags

# Check current version
CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Get latest v3.x.y tag (stable releases use v#.#.# format)
LATEST_TAG=$(git tag --sort=-v:refname | grep "^v[0-9]" | head -n 1)
echo "Latest tag: $LATEST_TAG"

# Preview changes between versions
if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50

  # Look for API-related changes
  echo -e "\nAPI-related commits:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" --grep="LAPACKE\|CBLAS\|deprecat\|new routine" -i
fi

# Checkout latest tag
git checkout "$LATEST_TAG"

# Verify
echo "Now on: $(git describe --tags)"

cd ..
```

- Latest tags viewable at: https://github.com/Reference-LAPACK/lapack/tags
- Stable release tags use format: `v3.X.Y` (e.g., `v3.12.0`, `v3.12.1`)

### 2. Analyze API Changes

Scan the two source-of-truth header files:

- **`lapack/CBLAS/include/cblas.h`** - All CBLAS function prototypes (Level 1/2/3)
- **`lapack/LAPACKE/include/lapacke.h`** - All LAPACKE function prototypes

Look for:
- New functions added since last update
- Deprecated or removed functions
- Modified function signatures (parameter changes, return type changes)
- New constants, enums, or data types

### 3. Update Skill Documentation Files

Update the following files in `skills/blas_lapack/references/`:

- **`blas-level1.md`** - Vector operations: dot, nrm2, asum, axpy, swap, copy, rot, scal
- **`blas-level2.md`** - Matrix-vector operations: gemv, gbmv, trmv, trsv, symv, hemv, ger, syr, her
- **`blas-level3.md`** - Matrix-matrix operations: gemm, symm, hemm, syrk, herk, trmm, trsm
- **`lapacke-linear-systems.md`** - Solvers: gesv, gbsv, posv, sysv, hesv + expert/refinement variants
- **`lapacke-least-squares.md`** - Least squares: gels, gelsd, gelss + QR/LQ factorizations
- **`lapacke-eigenvalues.md`** - Eigenvalue problems: syev, heev, geev, gees + generalized + Schur
- **`lapacke-svd.md`** - SVD: gesvd, gesdd, gesvdx, gesvj + bidiagonal + CS decomposition
- **`lapacke-factorizations.md`** - Factorizations: LU, Cholesky, LDL, triangular operations
- **`lapacke-auxiliary.md`** - Norms, generators, orthogonal/unitary transforms, utilities
- **`workflows.md`** - Working C examples and usage patterns

### 4. Update Main SKILL.md

- Update the LAPACK version tag in the documentation
- Update the total function count in the overview section
- Add notes about newly supported routines or features
- Update the "Quick Reference" section with any critical new functions

### 5. Remove Deprecated Content

- Search all documentation files for functions removed from the headers
- Remove or replace deprecated function references
- Update migration notes for breaking changes

### 6. Validate Changes

- Cross-reference all documented CBLAS functions against `cblas.h`
- Cross-reference all documented LAPACKE functions against `lapacke.h`
- Ensure function signatures match exactly (parameter types, names, return types)
- Verify that all new public API functions are documented
- Count functions to confirm totals match

### 7. Package Updated Skill

```bash
# From repo root
./package.sh blas_lapack
```

This creates `skills/blas_lapack.skill`.

## Key Considerations

- **Function Signatures**: Always verify exact signatures from the headers - parameter order, types, and names must match
- **Precision Variants**: BLAS/LAPACK functions come in 4 precision types (s/d/c/z) - ensure all variants are documented
- **LAPACKE Conventions**: Document `matrix_layout` parameter (LAPACK_ROW_MAJOR=101, LAPACK_COL_MAJOR=102)
- **Deprecation Handling**: Remove deprecated functions completely; don't just mark them
- **New Routines**: Major new routines should get workflow examples in workflows.md
- **Backward Compatibility**: Note any breaking changes in SKILL.md
- **Completeness**: All public CBLAS and LAPACKE functions should be documented

## Success Criteria

- ✅ LAPACK is updated to the latest stable tag
- ✅ SKILL.md documents which tag version it corresponds to
- ✅ All new functions from cblas.h and lapacke.h are documented
- ✅ All removed functions are removed from documentation
- ✅ All function signatures match the headers exactly
- ✅ Code examples in workflows.md compile with standard CBLAS/LAPACKE
- ✅ Package builds successfully with `./package.sh blas_lapack`
