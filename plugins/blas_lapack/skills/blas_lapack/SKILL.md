---
name: blas_lapack
description: "Complete CBLAS and LAPACKE C API reference (LAPACK v3.12.1) covering 1284 functions for numerical linear algebra: BLAS Level 1/2/3 vector and matrix operations, linear system solvers (LU, Cholesky, LDL), eigenvalue/eigenvector computation, singular value decomposition, least squares, QR/LQ factorizations, and auxiliary routines. Triggers on: BLAS/LAPACK questions, CBLAS/LAPACKE code, linear algebra in C/C++, matrix operations, numerical computing, scientific computing, HPC, linking BLAS/LAPACK."
---

# CBLAS & LAPACKE API Reference

Complete C API documentation for **BLAS** (Basic Linear Algebra Subprograms) and **LAPACK** (Linear Algebra PACKage), version **3.12.1**.

- **152 CBLAS functions** - Level 1/2/3 vector and matrix operations
- **1,132 LAPACKE functions** - Linear systems, eigenvalues, SVD, factorizations, and more
- **Source of truth**: `cblas.h` and `lapacke.h` from Reference-LAPACK v3.12.1

## Quick Start

### 1. Include Headers

```c
#include <cblas.h>      // BLAS operations
#include <lapacke.h>    // LAPACK operations
```

### 2. Compile and Link

```bash
# GCC/Clang
gcc myprogram.c -o myprogram -llapacke -llapack -lcblas -lblas -lm

# With pkg-config (if available)
gcc myprogram.c -o myprogram $(pkg-config --cflags --libs lapacke)

# With OpenBLAS (optimized)
gcc myprogram.c -o myprogram -lopenblas -lm

# With Intel MKL
gcc myprogram.c -o myprogram -lmkl_rt -lm
```

### 3. Typical Workflow

1. Allocate matrices/vectors as flat arrays (row-major or column-major)
2. Call CBLAS for basic operations (multiply, solve triangular, etc.)
3. Call LAPACKE for advanced operations (factorize, solve systems, eigenvalues, SVD)
4. Check `info` return value (0 = success, <0 = bad argument, >0 = numerical issue)
5. Free allocated memory

## When to Use This Skill

- **API Lookup**: Find the right CBLAS/LAPACKE function for a specific operation
- **Code Generation**: Write correct C code using BLAS/LAPACK with proper parameters
- **Linking Help**: Resolve compilation/linking issues with BLAS/LAPACK libraries
- **Solver Selection**: Choose the right solver for your matrix type and problem
- **Performance**: Select optimized routines for specific matrix structures

## Core Concepts

### Precision Prefixes

Every routine comes in up to 4 precision types:

| Prefix | Type | C Type | Description |
|--------|------|--------|-------------|
| `s` | Single real | `float` | 32-bit floating point |
| `d` | Double real | `double` | 64-bit floating point |
| `c` | Single complex | `void*` (lapack_complex_float) | 2x32-bit complex |
| `z` | Double complex | `void*` (lapack_complex_double) | 2x64-bit complex |

Example: `cblas_sgemm` (float), `cblas_dgemm` (double), `cblas_cgemm` (complex float), `cblas_zgemm` (complex double)

### CBLAS Enums

```c
typedef enum CBLAS_LAYOUT    { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans=111,  CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      { CblasUpper=121,    CblasLower=122 } CBLAS_UPLO;
typedef enum CBLAS_DIAG      { CblasNonUnit=131,  CblasUnit=132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE      { CblasLeft=141,     CblasRight=142 } CBLAS_SIDE;
```

### LAPACKE Layout Constants

```c
#define LAPACK_ROW_MAJOR  101
#define LAPACK_COL_MAJOR  102
```

### Leading Dimension (lda, ldb, ldc)

The leading dimension specifies the stride between consecutive columns (column-major) or rows (row-major) in memory:

- **Row-major**: `lda` = number of columns allocated (>= N for an M×N matrix)
- **Column-major**: `lda` = number of rows allocated (>= M for an M×N matrix)

### LAPACKE Info Return Values

All LAPACKE driver routines return `lapack_int info`:

| Value | Meaning |
|-------|---------|
| `0` | Success |
| `< 0` | The `-info`-th argument had an illegal value |
| `> 0` | Routine-specific failure (e.g., singular matrix, no convergence) |

### Matrix Storage Schemes

| Storage | Description | When to Use |
|---------|-------------|-------------|
| **Full (GE)** | Dense M×N array | General matrices |
| **Symmetric (SY/HE)** | Only upper or lower triangle stored | Symmetric/Hermitian matrices |
| **Triangular (TR)** | Only upper or lower triangle | Triangular matrices |
| **Banded (GB/SB/HB/TB)** | Band storage, KL sub + KU super diagonals | Banded matrices |
| **Packed (SP/HP/TP/PP)** | Upper or lower triangle packed into 1D array | Memory-efficient symmetric/triangular |
| **Tridiagonal (GT/PT/ST)** | Three diagonals stored as vectors | Tridiagonal systems |
| **RFP** | Rectangular Full Packed | Cache-friendly packed format |

### LAPACK Naming Convention

```
LAPACKE_<precision><operation>
         ^          ^
         s/d/c/z    2-6 char code describing the operation
```

Common operation codes:
- `gesv` = **GE**neral **S**ol**V**e
- `posv` = **PO**sitive definite **S**ol**V**e
- `syev` = **SY**mmetric **E**igen**V**alues
- `gesvd` = **GE**neral **S**ingular **V**alue **D**ecomposition
- `getrf` = **GE**neral **TR**iangular **F**actorization (LU)
- `potrf` = **PO**sitive definite **TR**iangular **F**actorization (Cholesky)

## API Reference

### CBLAS (152 functions)

| Reference | Functions | Description |
|-----------|-----------|-------------|
| [blas-level1.md](references/blas-level1.md) | ~50 | Vector operations: dot, nrm2, asum, axpy, swap, copy, rot, scal |
| [blas-level2.md](references/blas-level2.md) | ~66 | Matrix-vector: gemv, gbmv, trmv, trsv, symv, hemv, ger, syr, her |
| [blas-level3.md](references/blas-level3.md) | ~36 | Matrix-matrix: gemm, symm, hemm, syrk, herk, trmm, trsm |

### LAPACKE (1,132 functions)

| Reference | Description |
|-----------|-------------|
| [lapacke-linear-systems.md](references/lapacke-linear-systems.md) | Solve Ax=b: gesv, gbsv, posv, sysv, hesv + expert/refinement |
| [lapacke-least-squares.md](references/lapacke-least-squares.md) | Least squares: gels, gelsd + QR/LQ factorizations |
| [lapacke-eigenvalues.md](references/lapacke-eigenvalues.md) | Eigenvalues: syev, heev, geev, gees + generalized + Schur |
| [lapacke-svd.md](references/lapacke-svd.md) | SVD: gesvd, gesdd, gesvj + bidiagonal + CS decomposition |
| [lapacke-factorizations.md](references/lapacke-factorizations.md) | LU, Cholesky, LDL, triangular + storage conversions |
| [lapacke-auxiliary.md](references/lapacke-auxiliary.md) | Norms, generators, orthogonal transforms, utilities |

### Complete Workflow Examples

| Reference | Description |
|-----------|-------------|
| [workflows.md](references/workflows.md) | 15 complete C examples with compile commands |

## Solver Selection Guide

### "I need to solve Ax = b"

| Matrix Type | Simple | Expert (with error bounds) |
|-------------|--------|---------------------------|
| General | `gesv` | `gesvx` |
| General banded | `gbsv` | `gbsvx` |
| General tridiagonal | `gtsv` | `gtsvx` |
| Symmetric positive definite | `posv` | `posvx` |
| SPD banded | `pbsv` | `pbsvx` |
| SPD tridiagonal | `ptsv` | `ptsvx` |
| Symmetric indefinite | `sysv` | `sysvx` |
| Hermitian indefinite | `hesv` | `hesvx` |

### "I need eigenvalues/eigenvectors"

| Matrix Type | Fastest | Most Accurate | Subset |
|-------------|---------|---------------|--------|
| Symmetric | `syevd` | `syevr` | `syevx` |
| Hermitian | `heevd` | `heevr` | `heevx` |
| General | `geev` | `geevx` | - |
| Generalized symmetric | `sygvd` | - | `sygvx` |

### "I need SVD"

| Need | Routine | Notes |
|------|---------|-------|
| All singular values/vectors | `gesdd` | Fastest (divide & conquer) |
| Standard SVD | `gesvd` | Classic algorithm |
| Selected values/vectors | `gesvdx` | Subset by index or range |
| High accuracy | `gesvj` / `gejsv` | Jacobi methods |

## Best Practices

- **Choose the right solver**: Use specialized routines for structured matrices (symmetric, banded, triangular) - they are faster and more accurate
- **Check info**: Always check the return value of LAPACKE functions
- **Row vs column major**: CBLAS and LAPACKE support both via the `layout` parameter; be consistent throughout your code
- **Leading dimension**: Must be >= the actual dimension; can be larger for submatrix operations
- **Workspace**: LAPACKE high-level functions handle workspace allocation automatically (unlike raw LAPACK Fortran calls)
- **Complex types**: Use `lapack_complex_float` / `lapack_complex_double` or pass `void*` to C arrays of `float[2]`/`double[2]`

## Troubleshooting

### Linking errors: undefined reference to `cblas_dgemm`

```bash
# Make sure you link in the right order (dependent libs first)
gcc prog.c -llapacke -llapack -lcblas -lblas -lm -lgfortran

# Or use OpenBLAS which bundles everything
gcc prog.c -lopenblas -lm
```

### Wrong results with row-major layout

LAPACK is natively column-major (Fortran). When using `LAPACK_ROW_MAJOR`, LAPACKE transposes internally. For maximum performance, use `LAPACK_COL_MAJOR` and store matrices in column-major order.

### LAPACKE_dgesv returns info > 0

The matrix is singular (or nearly singular). `info = i` means U(i,i) is exactly zero in the LU factorization. Use `gesvx` for condition number estimation.

### "Illegal value" error (info < 0)

Parameter `-info` has an illegal value. Common causes:
- Wrong `matrix_layout` constant
- `lda` too small (must be >= N for row-major, >= M for column-major)
- Invalid `uplo`, `trans`, or `diag` character

### Segmentation fault in LAPACKE calls

- Ensure arrays are large enough: `A[lda * N]` not `A[M * N]` when `lda > M`
- For `ipiv` arrays, allocate at least `min(M, N)` elements
- For eigenvalue arrays, allocate at least `N` elements

### Performance is poor

- Use an optimized BLAS implementation (OpenBLAS, Intel MKL, BLIS) instead of reference BLAS
- Ensure proper memory alignment (64-byte for AVX-512)
- Use column-major storage to avoid LAPACKE's internal transposition

## What's New in LAPACK 3.12.1

- `gemmtr` - Triangular matrix-matrix multiply (new BLAS Level 3 extension)
- `gesvdq` - SVD with QR preconditioning for high accuracy
- `getsqrhrt` - Tall-skinny QR with Householder reconstruction
- `trsyl3` - Level 3 Sylvester equation solver
- `*_64` variants - 64-bit integer API for large matrices
- Improved `gelq`/`geqr` - Tall-skinny and short-wide QR/LQ
