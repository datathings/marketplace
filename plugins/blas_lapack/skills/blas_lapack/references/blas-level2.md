# CBLAS Level 2 API Reference - Matrix-Vector Operations

> BLAS Level 2 operations perform matrix-vector operations with O(n^2) complexity.
> Source: LAPACK v3.12.1 - `CBLAS/include/cblas.h`

## Table of Contents
- [Enums Used](#enums-used)
- [General Matrix-Vector (gemv, gbmv)](#general-matrix-vector)
- [Triangular Matrix-Vector (trmv, tbmv, tpmv, trsv, tbsv, tpsv)](#triangular-matrix-vector)
- [Symmetric Matrix-Vector (symv, sbmv, spmv)](#symmetric-matrix-vector)
- [Hermitian Matrix-Vector (hemv, hbmv, hpmv)](#hermitian-matrix-vector)
- [Rank-1 Updates (ger, geru, gerc)](#rank-1-updates)
- [Symmetric Rank Updates (syr, spr, syr2, spr2)](#symmetric-rank-updates)
- [Hermitian Rank Updates (her, hpr, her2, hpr2)](#hermitian-rank-updates)

## Naming Convention

| Prefix | Type |
|--------|------|
| `s` | `float` (single precision real) |
| `d` | `double` (double precision real) |
| `c` | `void *` (single precision complex, interleaved float pairs) |
| `z` | `void *` (double precision complex, interleaved double pairs) |

## Enums Used

### CBLAS_LAYOUT
Controls whether matrices are stored in row-major or column-major order.
```c
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
```
| Value | Meaning |
|-------|---------|
| `CblasRowMajor` (101) | Row-major storage (C-style). Element (i,j) is at `A[i*lda + j]`. |
| `CblasColMajor` (102) | Column-major storage (Fortran-style). Element (i,j) is at `A[j*lda + i]`. |

### CBLAS_TRANSPOSE
Controls the operation applied to a matrix before the computation.
```c
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
```
| Value | Meaning |
|-------|---------|
| `CblasNoTrans` (111) | Use A as-is: op(A) = A |
| `CblasTrans` (112) | Transpose: op(A) = A^T |
| `CblasConjTrans` (113) | Conjugate transpose: op(A) = A^H (same as `CblasTrans` for real types) |

### CBLAS_UPLO
Specifies whether the upper or lower triangle of a symmetric/Hermitian matrix is stored.
```c
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
```
| Value | Meaning |
|-------|---------|
| `CblasUpper` (121) | Upper triangular part is stored/referenced |
| `CblasLower` (122) | Lower triangular part is stored/referenced |

### CBLAS_DIAG
Specifies whether a triangular matrix has a unit diagonal.
```c
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
```
| Value | Meaning |
|-------|---------|
| `CblasNonUnit` (131) | Diagonal elements are explicitly stored and used |
| `CblasUnit` (132) | Diagonal elements are assumed to be 1 (not referenced) |

---

## General Matrix-Vector

### cblas_sgemv / cblas_dgemv / cblas_cgemv / cblas_zgemv

General matrix-vector multiply: **y = alpha * op(A) * x + beta * y**

```c
void cblas_sgemv(const CBLAS_LAYOUT layout,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *X, const CBLAS_INT incX, const float beta,
                 float *Y, const CBLAS_INT incY);

void cblas_dgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *X, const CBLAS_INT incX, const double beta,
                 double *Y, const CBLAS_INT incY);

void cblas_cgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta,
                 void *Y, const CBLAS_INT incY);

void cblas_zgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta,
                 void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `TransA` | Operation on A: `CblasNoTrans` (A), `CblasTrans` (A^T), or `CblasConjTrans` (A^H) |
| `M` | Number of rows of matrix A |
| `N` | Number of columns of matrix A |
| `alpha` | Scalar multiplier for op(A)*x. For complex variants, pointer to complex value. |
| `A` | Pointer to matrix A, dimension (lda, N) in column-major or (M, lda) in row-major |
| `lda` | Leading dimension of A. Column-major: lda >= max(1, M). Row-major: lda >= max(1, N). |
| `X` | Input vector. Length is N when `CblasNoTrans`, M when transposed. |
| `incX` | Stride between elements of X. Must not be zero. |
| `beta` | Scalar multiplier for y. For complex variants, pointer to complex value. |
| `Y` | Input/output vector. Length is M when `CblasNoTrans`, N when transposed. Overwritten on output. |
| `incY` | Stride between elements of Y. Must not be zero. |

---

### cblas_sgbmv / cblas_dgbmv / cblas_cgbmv / cblas_zgbmv

General banded matrix-vector multiply: **y = alpha * op(A) * x + beta * y**

A is an M-by-N band matrix with KL sub-diagonals and KU super-diagonals, stored in band storage format.

```c
void cblas_sgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const float alpha,
                 const float *A, const CBLAS_INT lda, const float *X,
                 const CBLAS_INT incX, const float beta, float *Y, const CBLAS_INT incY);

void cblas_dgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const double alpha,
                 const double *A, const CBLAS_INT lda, const double *X,
                 const CBLAS_INT incX, const double beta, double *Y, const CBLAS_INT incY);

void cblas_cgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y, const CBLAS_INT incY);

void cblas_zgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `M` | Number of rows of matrix A |
| `N` | Number of columns of matrix A |
| `KL` | Number of sub-diagonals of A |
| `KU` | Number of super-diagonals of A |
| `alpha` | Scalar multiplier for op(A)*x |
| `A` | Pointer to band matrix A in band storage, dimension (lda, N). lda >= KL + KU + 1. |
| `lda` | Leading dimension of A. Must be >= KL + KU + 1. |
| `X` | Input vector |
| `incX` | Stride between elements of X |
| `beta` | Scalar multiplier for y |
| `Y` | Input/output vector, overwritten on output |
| `incY` | Stride between elements of Y |

---

## Triangular Matrix-Vector

### cblas_strmv / cblas_dtrmv / cblas_ctrmv / cblas_ztrmv

Triangular matrix-vector multiply: **x = op(A) * x**

A is a triangular matrix. The result overwrites x in-place.

```c
void cblas_strmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);

void cblas_dtrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);

void cblas_ctrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);

void cblas_ztrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit`: diagonal is explicit. `CblasUnit`: diagonal assumed to be 1. |
| `N` | Order of matrix A (N x N) |
| `A` | Pointer to triangular matrix A, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= max(1, N). |
| `X` | Input/output vector of length N, overwritten with result |
| `incX` | Stride between elements of X |

---

### cblas_stbmv / cblas_dtbmv / cblas_ctbmv / cblas_ztbmv

Triangular banded matrix-vector multiply: **x = op(A) * x**

A is a triangular band matrix with K diagonals, stored in band format.

```c
void cblas_stbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);

void cblas_dtbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);

void cblas_ctbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);

void cblas_ztbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit` or `CblasUnit` |
| `N` | Order of matrix A (N x N) |
| `K` | Number of super-diagonals (upper) or sub-diagonals (lower) of A |
| `A` | Pointer to triangular band matrix A in band storage, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= K + 1. |
| `X` | Input/output vector of length N, overwritten with result |
| `incX` | Stride between elements of X |

---

### cblas_stpmv / cblas_dtpmv / cblas_ctpmv / cblas_ztpmv

Triangular packed matrix-vector multiply: **x = op(A) * x**

A is a triangular matrix stored in packed format (array of length N*(N+1)/2).

```c
void cblas_stpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *Ap, float *X, const CBLAS_INT incX);

void cblas_dtpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *Ap, double *X, const CBLAS_INT incX);

void cblas_ctpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);

void cblas_ztpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit` or `CblasUnit` |
| `N` | Order of matrix A (N x N) |
| `Ap` | Pointer to triangular matrix A in packed storage, array of length N*(N+1)/2 |
| `X` | Input/output vector of length N, overwritten with result |
| `incX` | Stride between elements of X |

---

### cblas_strsv / cblas_dtrsv / cblas_ctrsv / cblas_ztrsv

Triangular solve: **op(A) * x = b**, solving for x (result overwrites x)

```c
void cblas_strsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *A, const CBLAS_INT lda, float *X,
                 const CBLAS_INT incX);

void cblas_dtrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *A, const CBLAS_INT lda, double *X,
                 const CBLAS_INT incX);

void cblas_ctrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX);

void cblas_ztrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit` or `CblasUnit` |
| `N` | Order of matrix A (N x N) |
| `A` | Pointer to triangular matrix A, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= max(1, N). |
| `X` | On entry, the right-hand side vector b. On exit, the solution vector x. |
| `incX` | Stride between elements of X |

**Note:** A must be non-singular. No test for singularity is performed.

---

### cblas_stbsv / cblas_dtbsv / cblas_ctbsv / cblas_ztbsv

Triangular banded solve: **op(A) * x = b**, solving for x

A is a triangular band matrix with K diagonals.

```c
void cblas_stbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);

void cblas_dtbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);

void cblas_ctbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);

void cblas_ztbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit` or `CblasUnit` |
| `N` | Order of matrix A (N x N) |
| `K` | Number of super-diagonals (upper) or sub-diagonals (lower) of A |
| `A` | Pointer to triangular band matrix A in band storage, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= K + 1. |
| `X` | On entry, the right-hand side vector b. On exit, the solution vector x. |
| `incX` | Stride between elements of X |

**Note:** A must be non-singular. No test for singularity is performed.

---

### cblas_stpsv / cblas_dtpsv / cblas_ctpsv / cblas_ztpsv

Triangular packed solve: **op(A) * x = b**, solving for x

A is a triangular matrix stored in packed format.

```c
void cblas_stpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *Ap, float *X, const CBLAS_INT incX);

void cblas_dtpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *Ap, double *X, const CBLAS_INT incX);

void cblas_ctpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);

void cblas_ztpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: A is upper triangular. `CblasLower`: A is lower triangular. |
| `TransA` | Operation on A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CblasNonUnit` or `CblasUnit` |
| `N` | Order of matrix A (N x N) |
| `Ap` | Pointer to triangular matrix A in packed storage, array of length N*(N+1)/2 |
| `X` | On entry, the right-hand side vector b. On exit, the solution vector x. |
| `incX` | Stride between elements of X |

**Note:** A must be non-singular. No test for singularity is performed.

---

## Symmetric Matrix-Vector

These routines are available with S and D prefixes only (real types). For complex symmetric operations, use the Hermitian variants instead.

### cblas_ssymv / cblas_dsymv

Symmetric matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a symmetric matrix; only the upper or lower triangle is referenced.

```c
void cblas_ssymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);

void cblas_dsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for A*x |
| `A` | Pointer to symmetric matrix A, dimension (lda, N). Only the triangle specified by Uplo is referenced. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Scalar multiplier for y |
| `Y` | Input/output vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

### cblas_ssbmv / cblas_dsbmv

Symmetric banded matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a symmetric band matrix with K super-diagonals (or sub-diagonals).

```c
void cblas_ssbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);

void cblas_dsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `K` | Number of super-diagonals (if upper) or sub-diagonals (if lower) of A |
| `alpha` | Scalar multiplier for A*x |
| `A` | Pointer to symmetric band matrix A in band storage, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= K + 1. |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Scalar multiplier for y |
| `Y` | Input/output vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

### cblas_sspmv / cblas_dspmv

Symmetric packed matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a symmetric matrix stored in packed format (array of length N*(N+1)/2).

```c
void cblas_sspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const float alpha, const float *Ap,
                 const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);

void cblas_dspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const double alpha, const double *Ap,
                 const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for A*x |
| `Ap` | Pointer to symmetric matrix A in packed storage, array of length N*(N+1)/2 |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Scalar multiplier for y |
| `Y` | Input/output vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

## Hermitian Matrix-Vector

These routines are available with C and Z prefixes only (complex types). They are the complex counterpart of the symmetric matrix-vector routines.

### cblas_chemv / cblas_zhemv

Hermitian matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a Hermitian matrix (A = A^H); only the upper or lower triangle is referenced.

```c
void cblas_chemv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);

void cblas_zhemv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Pointer to complex scalar multiplier for A*x |
| `A` | Pointer to Hermitian matrix A, dimension (lda, N). Only the triangle specified by Uplo is referenced. Diagonal elements are assumed to have zero imaginary part. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Pointer to complex scalar multiplier for y |
| `Y` | Input/output complex vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

### cblas_chbmv / cblas_zhbmv

Hermitian banded matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a Hermitian band matrix with K super-diagonals (or sub-diagonals).

```c
void cblas_chbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);

void cblas_zhbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `K` | Number of super-diagonals (if upper) or sub-diagonals (if lower) of A |
| `alpha` | Pointer to complex scalar multiplier for A*x |
| `A` | Pointer to Hermitian band matrix A in band storage, dimension (lda, N) |
| `lda` | Leading dimension of A. Must be >= K + 1. |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Pointer to complex scalar multiplier for y |
| `Y` | Input/output complex vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

### cblas_chpmv / cblas_zhpmv

Hermitian packed matrix-vector multiply: **y = alpha * A * x + beta * y**

A is a Hermitian matrix stored in packed format (array of length N*(N+1)/2).

```c
void cblas_chpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);

void cblas_zhpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle of A is stored. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Pointer to complex scalar multiplier for A*x |
| `Ap` | Pointer to Hermitian matrix A in packed storage, array of length N*(N+1)/2 |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `beta` | Pointer to complex scalar multiplier for y |
| `Y` | Input/output complex vector of length N, overwritten on output |
| `incY` | Stride between elements of Y |

---

## Rank-1 Updates

### cblas_sger / cblas_dger

General rank-1 update (real): **A = alpha * x * y^T + A**

```c
void cblas_sger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const float alpha, const float *X, const CBLAS_INT incX,
                const float *Y, const CBLAS_INT incY, float *A, const CBLAS_INT lda);

void cblas_dger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const double alpha, const double *X, const CBLAS_INT incX,
                const double *Y, const CBLAS_INT incY, double *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `M` | Number of rows of matrix A |
| `N` | Number of columns of matrix A |
| `alpha` | Scalar multiplier for the outer product x*y^T |
| `X` | Input vector of length M |
| `incX` | Stride between elements of X |
| `Y` | Input vector of length N |
| `incY` | Stride between elements of Y |
| `A` | Input/output matrix A, dimension (lda, N). Updated in place. |
| `lda` | Leading dimension of A. Column-major: lda >= max(1, M). Row-major: lda >= max(1, N). |

---

### cblas_cgeru / cblas_zgeru

General rank-1 update (complex, unconjugated): **A = alpha * x * y^T + A**

```c
void cblas_cgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);

void cblas_zgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `M` | Number of rows of matrix A |
| `N` | Number of columns of matrix A |
| `alpha` | Pointer to complex scalar multiplier |
| `X` | Input complex vector of length M |
| `incX` | Stride between elements of X |
| `Y` | Input complex vector of length N (used without conjugation) |
| `incY` | Stride between elements of Y |
| `A` | Input/output complex matrix A, dimension (lda, N). Updated in place. |
| `lda` | Leading dimension of A |

---

### cblas_cgerc / cblas_zgerc

General rank-1 update (complex, conjugated): **A = alpha * x * y^H + A**

```c
void cblas_cgerc(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);

void cblas_zgerc(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `M` | Number of rows of matrix A |
| `N` | Number of columns of matrix A |
| `alpha` | Pointer to complex scalar multiplier |
| `X` | Input complex vector of length M |
| `incX` | Stride between elements of X |
| `Y` | Input complex vector of length N (conjugated before use: y^H) |
| `incY` | Stride between elements of Y |
| `A` | Input/output complex matrix A, dimension (lda, N). Updated in place. |
| `lda` | Leading dimension of A |

---

## Symmetric Rank Updates

These routines are available with S and D prefixes only (real types).

### cblas_ssyr / cblas_dsyr

Symmetric rank-1 update: **A = alpha * x * x^T + A**

A is symmetric; only the triangle specified by Uplo is updated.

```c
void cblas_ssyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, float *A, const CBLAS_INT lda);

void cblas_dsyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, double *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored and updated. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for the rank-1 update |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `A` | Input/output symmetric matrix A, dimension (lda, N). Only the specified triangle is updated. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |

---

### cblas_sspr / cblas_dspr

Symmetric packed rank-1 update: **A = alpha * x * x^T + A**

A is a symmetric matrix in packed storage.

```c
void cblas_sspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, float *Ap);

void cblas_dspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, double *Ap);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for the rank-1 update |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `Ap` | Input/output symmetric matrix A in packed storage, array of length N*(N+1)/2. Updated in place. |

---

### cblas_ssyr2 / cblas_dsyr2

Symmetric rank-2 update: **A = alpha * x * y^T + alpha * y * x^T + A**

```c
void cblas_ssyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, const float *Y, const CBLAS_INT incY, float *A,
                const CBLAS_INT lda);

void cblas_dsyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, const double *Y, const CBLAS_INT incY, double *A,
                const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored and updated. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for the rank-2 update |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `Y` | Input vector of length N |
| `incY` | Stride between elements of Y |
| `A` | Input/output symmetric matrix A, dimension (lda, N). Only the specified triangle is updated. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |

---

### cblas_sspr2 / cblas_dspr2

Symmetric packed rank-2 update: **A = alpha * x * y^T + alpha * y * x^T + A**

A is a symmetric matrix in packed storage.

```c
void cblas_sspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, const float *Y, const CBLAS_INT incY, float *A);

void cblas_dspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, const double *Y, const CBLAS_INT incY, double *A);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Scalar multiplier for the rank-2 update |
| `X` | Input vector of length N |
| `incX` | Stride between elements of X |
| `Y` | Input vector of length N |
| `incY` | Stride between elements of Y |
| `A` | Input/output symmetric matrix A in packed storage, array of length N*(N+1)/2. Updated in place. |

---

## Hermitian Rank Updates

These routines are available with C and Z prefixes only (complex types). They are the complex counterpart of the symmetric rank update routines.

### cblas_cher / cblas_zher

Hermitian rank-1 update: **A = alpha * x * x^H + A**

A is Hermitian; only the triangle specified by Uplo is updated. Note that alpha is a real scalar (float for `cher`, double for `zher`), not complex.

```c
void cblas_cher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const void *X, const CBLAS_INT incX,
                void *A, const CBLAS_INT lda);

void cblas_zher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const void *X, const CBLAS_INT incX,
                void *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored and updated. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | **Real** scalar multiplier (float for `cher`, double for `zher`) |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `A` | Input/output Hermitian matrix A, dimension (lda, N). Only the specified triangle is updated. Diagonal imaginary parts are set to zero. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |

---

### cblas_chpr / cblas_zhpr

Hermitian packed rank-1 update: **A = alpha * x * x^H + A**

A is a Hermitian matrix in packed storage. Alpha is a real scalar.

```c
void cblas_chpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const void *X,
                const CBLAS_INT incX, void *A);

void cblas_zhpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const void *X,
                const CBLAS_INT incX, void *A);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | **Real** scalar multiplier (float for `chpr`, double for `zhpr`) |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `A` | Input/output Hermitian matrix A in packed storage, array of length N*(N+1)/2. Updated in place. |

---

### cblas_cher2 / cblas_zher2

Hermitian rank-2 update: **A = alpha * x * y^H + conj(alpha) * y * x^H + A**

```c
void cblas_cher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);

void cblas_zher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored and updated. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Pointer to complex scalar multiplier |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `Y` | Input complex vector of length N |
| `incY` | Stride between elements of Y |
| `A` | Input/output Hermitian matrix A, dimension (lda, N). Only the specified triangle is updated. |
| `lda` | Leading dimension of A. Must be >= max(1, N). |

---

### cblas_chpr2 / cblas_zhpr2

Hermitian packed rank-2 update: **A = alpha * x * y^H + conj(alpha) * y * x^H + A**

A is a Hermitian matrix in packed storage.

```c
void cblas_chpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *Ap);

void cblas_zhpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *Ap);
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `layout` | Matrix storage order: `CblasRowMajor` or `CblasColMajor` |
| `Uplo` | `CblasUpper`: upper triangle of A is stored. `CblasLower`: lower triangle. |
| `N` | Order of matrix A (N x N) |
| `alpha` | Pointer to complex scalar multiplier |
| `X` | Input complex vector of length N |
| `incX` | Stride between elements of X |
| `Y` | Input complex vector of length N |
| `incY` | Stride between elements of Y |
| `Ap` | Input/output Hermitian matrix A in packed storage, array of length N*(N+1)/2. Updated in place. |

---

## Storage Format Summary

### Full Storage
Matrix A is stored as a 2D array of dimension (lda, N). The leading dimension `lda` must be at least the number of rows (column-major) or columns (row-major).

### Band Storage
A band matrix with KL sub-diagonals and KU super-diagonals is stored in a compact array of dimension (lda, N), where lda >= KL + KU + 1. In column-major order, column j of the band matrix is stored in column j of the array, with the diagonal element in row KU (0-indexed).

### Packed Storage
A triangular, symmetric, or Hermitian N-by-N matrix is stored as a 1D array of length N*(N+1)/2. For `CblasUpper` in column-major order, column j starts at index j*(j+1)/2. For `CblasLower`, column j starts at index j*(2*N - j - 1)/2 + j.

## Function Count Summary

| Category | Operation | Functions |
|----------|-----------|-----------|
| General MV | gemv | cblas_{s,d,c,z}gemv (4) |
| General Banded MV | gbmv | cblas_{s,d,c,z}gbmv (4) |
| Triangular MV | trmv | cblas_{s,d,c,z}trmv (4) |
| Triangular Banded MV | tbmv | cblas_{s,d,c,z}tbmv (4) |
| Triangular Packed MV | tpmv | cblas_{s,d,c,z}tpmv (4) |
| Triangular Solve | trsv | cblas_{s,d,c,z}trsv (4) |
| Triangular Banded Solve | tbsv | cblas_{s,d,c,z}tbsv (4) |
| Triangular Packed Solve | tpsv | cblas_{s,d,c,z}tpsv (4) |
| Symmetric MV | symv | cblas_{s,d}symv (2) |
| Symmetric Banded MV | sbmv | cblas_{s,d}sbmv (2) |
| Symmetric Packed MV | spmv | cblas_{s,d}spmv (2) |
| Hermitian MV | hemv | cblas_{c,z}hemv (2) |
| Hermitian Banded MV | hbmv | cblas_{c,z}hbmv (2) |
| Hermitian Packed MV | hpmv | cblas_{c,z}hpmv (2) |
| Real Rank-1 | ger | cblas_{s,d}ger (2) |
| Complex Rank-1 (unconj) | geru | cblas_{c,z}geru (2) |
| Complex Rank-1 (conj) | gerc | cblas_{c,z}gerc (2) |
| Symmetric Rank-1 | syr | cblas_{s,d}syr (2) |
| Symmetric Packed Rank-1 | spr | cblas_{s,d}spr (2) |
| Symmetric Rank-2 | syr2 | cblas_{s,d}syr2 (2) |
| Symmetric Packed Rank-2 | spr2 | cblas_{s,d}spr2 (2) |
| Hermitian Rank-1 | her | cblas_{c,z}her (2) |
| Hermitian Packed Rank-1 | hpr | cblas_{c,z}hpr (2) |
| Hermitian Rank-2 | her2 | cblas_{c,z}her2 (2) |
| Hermitian Packed Rank-2 | hpr2 | cblas_{c,z}hpr2 (2) |
| **Total** | | **66 functions** |
