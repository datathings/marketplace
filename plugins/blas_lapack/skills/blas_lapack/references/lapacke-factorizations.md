# LAPACKE Factorization API Reference

> Matrix factorization routines: LU, Cholesky, LDL, and triangular operations.
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [Common Parameters](#common-parameters)
- [LU Factorization](#lu-factorization)
  - [getrf - LU factorization with partial pivoting](#getrf)
  - [getrf2 - Recursive LU factorization](#getrf2)
  - [getf2 - Unblocked LU factorization](#getf2)
  - [getrs - Solve using LU factorization](#getrs)
  - [getri - Inverse using LU factorization](#getri)
- [Cholesky Factorization](#cholesky-factorization)
  - [potrf - Cholesky factorization](#potrf)
  - [potrf2 - Recursive Cholesky factorization](#potrf2)
  - [potrs - Solve using Cholesky](#potrs)
  - [potri - Inverse using Cholesky](#potri)
  - [pptrf - Packed Cholesky factorization](#pptrf)
  - [pptrs - Solve using packed Cholesky](#pptrs)
  - [pptri - Inverse using packed Cholesky](#pptri)
  - [pbtrf - Banded Cholesky factorization](#pbtrf)
  - [pbtrs - Solve using banded Cholesky](#pbtrs)
  - [pftrf - RFP Cholesky factorization](#pftrf)
  - [pftri - Inverse using RFP Cholesky](#pftri)
  - [pftrs - Solve using RFP Cholesky](#pftrs)
  - [pstrf - Cholesky with pivoting](#pstrf)
- [LDL^T / LDL^H Factorization](#ldlt--ldlh-factorization)
  - [sytrf - Symmetric indefinite factorization](#sytrf)
  - [sytrs - Solve using symmetric indefinite](#sytrs)
  - [sytri - Inverse using symmetric indefinite](#sytri)
  - [sytri2 - Inverse (unblocked)](#sytri2)
  - [sytri2x - Inverse (blocked)](#sytri2x)
  - [sytrs2 - Solve (revised)](#sytrs2)
  - [hetrf - Hermitian indefinite factorization](#hetrf)
  - [hetrs - Solve using Hermitian indefinite](#hetrs)
  - [hetri - Inverse using Hermitian indefinite](#hetri)
  - [hetri2 - Inverse (unblocked)](#hetri2)
  - [hetri2x - Inverse (blocked)](#hetri2x)
  - [hetrs2 - Solve (revised)](#hetrs2)
  - [sptrf - Symmetric packed factorization](#sptrf)
  - [sptrs - Solve using symmetric packed](#sptrs)
  - [sptri - Inverse using symmetric packed](#sptri)
  - [hptrf - Hermitian packed factorization](#hptrf)
  - [hptrs - Solve using Hermitian packed](#hptrs)
  - [hptri - Inverse using Hermitian packed](#hptri)
- [Tridiagonal Factorization](#tridiagonal-factorization)
  - [pttrf - Positive definite tridiagonal factorization](#pttrf)
  - [pttrs - Solve using positive definite tridiagonal](#pttrs)
  - [gttrf - General tridiagonal factorization](#gttrf)
  - [gttrs - Solve using general tridiagonal](#gttrs)
- [Triangular Operations](#triangular-operations)
  - [trtri - Triangular inverse](#trtri)
  - [trtrs - Triangular solve](#trtrs)
  - [tptri - Triangular packed inverse](#tptri)
  - [tptrs - Triangular packed solve](#tptrs)
  - [tbcon - Triangular banded condition number](#tbcon)
  - [tbrfs - Triangular banded refinement](#tbrfs)
  - [tbtrs - Triangular banded solve](#tbtrs)
  - [trcon - Triangular condition number](#trcon)
  - [trrfs - Triangular refinement](#trrfs)
  - [tpcon - Triangular packed condition number](#tpcon)
  - [tprfs - Triangular packed refinement](#tprfs)
  - [lauum - Product of triangular matrices](#lauum)
- [Matrix Storage Conversions](#matrix-storage-conversions)
  - [tfttr - RFP to Full](#tfttr)
  - [trttf - Full to RFP](#trttf)
  - [tfttp - RFP to Packed](#tfttp)
  - [tpttf - Packed to RFP](#tpttf)
  - [tpttr - Packed to Full](#tpttr)
  - [trttp - Full to Packed](#trttp)

## Common Parameters

### Precision Prefixes

| Prefix | Type | Description |
|--------|------|-------------|
| `s` | `float` | Single-precision real |
| `d` | `double` | Double-precision real |
| `c` | `lapack_complex_float` | Single-precision complex |
| `z` | `lapack_complex_double` | Double-precision complex |

### Layout and Control Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `matrix_layout` | `LAPACK_ROW_MAJOR` (101), `LAPACK_COL_MAJOR` (102) | Storage order |
| `uplo` | `'U'`, `'L'` | Use upper or lower triangular part |
| `trans` | `'N'`, `'T'`, `'C'` | No transpose, transpose, conjugate transpose |
| `diag` | `'N'`, `'U'` | Non-unit or unit diagonal |
| `norm` | `'1'`, `'O'`, `'I'` | 1-norm, 1-norm, infinity-norm |
| `transr` | `'N'`, `'T'` | RFP format: normal or transposed |

### Return Value (`info`)

| Value | Meaning |
|-------|---------|
| `= 0` | Successful exit |
| `< 0` | The `-info`-th argument had an illegal value |
| `> 0` | Factorization-specific failure (e.g., singular matrix, not positive definite) |

---

## LU Factorization

Computes `A = P * L * U` where `P` is a permutation matrix, `L` is lower triangular with unit diagonal, and `U` is upper triangular.

### getrf

Computes the LU factorization of a general m-by-n matrix using partial pivoting with row interchanges. This is the blocked (Level 3 BLAS) algorithm.

```c
lapack_int LAPACKE_sgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_cgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A (m >= 0) |
| `n` | `lapack_int` | Number of columns of A (n >= 0) |
| `a` | `T*` | **[in/out]** m-by-n matrix; overwritten with L and U |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices, dimension min(m,n) |

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### getrf2

Computes the LU factorization of a general m-by-n matrix using partial pivoting. Recursive version for improved accuracy.

```c
lapack_int LAPACKE_sgetrf2( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dgetrf2( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_cgetrf2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zgetrf2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );
```

**Parameters:** Same as [getrf](#getrf).

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### getf2

Computes the LU factorization of a general m-by-n matrix using partial pivoting. Unblocked (Level 2 BLAS) algorithm.

```c
lapack_int LAPACKE_sgetf2( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dgetf2( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_cgetf2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zgetf2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );
```

**Parameters:** Same as [getrf](#getrf).

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### getrs

Solves a system of linear equations `A * X = B`, `A^T * X = B`, or `A^H * X = B` using the LU factorization computed by getrf.

```c
lapack_int LAPACKE_sgetrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dgetrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_cgetrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zgetrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`: solve `A*X=B`; `'T'`: solve `A^T*X=B`; `'C'`: solve `A^H*X=B` |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides (columns of B) |
| `a` | `const T*` | **[in]** LU factorization from getrf, n-by-n |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from getrf |
| `b` | `T*` | **[in/out]** n-by-nrhs RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### getri

Computes the inverse of a matrix using the LU factorization computed by getrf.

```c
lapack_int LAPACKE_sgetri( int matrix_layout, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dgetri( int matrix_layout, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_cgetri( int matrix_layout, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zgetri( int matrix_layout, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** LU factorization from getrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from getrf |

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular, inverse cannot be computed).

---

## Cholesky Factorization

Computes `A = U^T * U` (uplo='U') or `A = L * L^T` (uplo='L') for symmetric/Hermitian positive definite matrices. For complex types, `A = U^H * U` or `A = L * L^H`.

### potrf

Computes the Cholesky factorization of a symmetric/Hermitian positive definite matrix. Blocked (Level 3 BLAS) algorithm.

```c
lapack_int LAPACKE_spotrf( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dpotrf( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_cpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle stored; `'L'`: lower triangle stored |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Symmetric/Hermitian matrix; overwritten with U or L factor |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### potrf2

Computes the Cholesky factorization of a symmetric/Hermitian positive definite matrix. Recursive algorithm.

```c
lapack_int LAPACKE_spotrf2( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dpotrf2( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_cpotrf2( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotrf2( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:** Same as [potrf](#potrf).

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### potrs

Solves `A * X = B` using the Cholesky factorization `A = U^T*U` or `A = L*L^T` computed by potrf.

```c
lapack_int LAPACKE_spotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dpotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_cpotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zpotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match the factorization from potrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Cholesky factor from potrf |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### potri

Computes the inverse of a symmetric/Hermitian positive definite matrix using the Cholesky factorization from potrf.

```c
lapack_int LAPACKE_spotri( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dpotri( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_cpotri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match the factorization from potrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Cholesky factor from potrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success; `> 0` if the `info`-th diagonal element is zero.

---

### pptrf

Computes the Cholesky factorization of a symmetric/Hermitian positive definite matrix stored in packed format.

```c
lapack_int LAPACKE_spptrf( int matrix_layout, char uplo, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dpptrf( int matrix_layout, char uplo, lapack_int n,
                           double* ap );
lapack_int LAPACKE_cpptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_zpptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle packed; `'L'`: lower triangle packed |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed triangular matrix, dimension n*(n+1)/2; overwritten with factor |

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### pptrs

Solves `A * X = B` using packed Cholesky factorization from pptrf.

```c
lapack_int LAPACKE_spptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dpptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_cpptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match pptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed Cholesky factor from pptrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### pptri

Computes the inverse using packed Cholesky factorization from pptrf.

```c
lapack_int LAPACKE_spptri( int matrix_layout, char uplo, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dpptri( int matrix_layout, char uplo, lapack_int n,
                           double* ap );
lapack_int LAPACKE_cpptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_zpptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match pptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed factor from pptrf; overwritten with inverse |

**Returns:** `info` -- 0 on success; `> 0` if the `info`-th diagonal element is zero.

---

### pbtrf

Computes the Cholesky factorization of a symmetric/Hermitian positive definite band matrix.

```c
lapack_int LAPACKE_spbtrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, float* ab, lapack_int ldab );
lapack_int LAPACKE_dpbtrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, double* ab, lapack_int ldab );
lapack_int LAPACKE_cpbtrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_float* ab,
                           lapack_int ldab );
lapack_int LAPACKE_zpbtrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_double* ab,
                           lapack_int ldab );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle of band matrix; `'L'`: lower triangle |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `kd` | `lapack_int` | Number of superdiagonals (uplo='U') or subdiagonals (uplo='L') |
| `ab` | `T*` | **[in/out]** Band matrix in banded storage; overwritten with factor |
| `ldab` | `lapack_int` | Leading dimension of ab |

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### pbtrs

Solves `A * X = B` using the banded Cholesky factorization from pbtrf.

```c
lapack_int LAPACKE_spbtrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const float* ab,
                           lapack_int ldab, float* b, lapack_int ldb );
lapack_int LAPACKE_dpbtrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const double* ab,
                           lapack_int ldab, double* b, lapack_int ldb );
lapack_int LAPACKE_cpbtrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpbtrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match pbtrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `kd` | `lapack_int` | Number of superdiagonals or subdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `const T*` | **[in]** Banded Cholesky factor from pbtrf |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### pftrf

Computes the Cholesky factorization of a symmetric/Hermitian positive definite matrix stored in Rectangular Full Packed (RFP) format.

```c
lapack_int LAPACKE_spftrf( int matrix_layout, char transr, char uplo,
                           lapack_int n, float* a );
lapack_int LAPACKE_dpftrf( int matrix_layout, char transr, char uplo,
                           lapack_int n, double* a );
lapack_int LAPACKE_cpftrf( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_zpftrf( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_complex_double* a );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: normal RFP; `'T'`/`'C'`: transposed/conjugate-transposed RFP |
| `uplo` | `char` | `'U'`: upper triangle; `'L'`: lower triangle |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Matrix in RFP format, dimension n*(n+1)/2; overwritten with factor |

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### pftri

Computes the inverse using RFP Cholesky factorization from pftrf.

```c
lapack_int LAPACKE_spftri( int matrix_layout, char transr, char uplo,
                           lapack_int n, float* a );
lapack_int LAPACKE_dpftri( int matrix_layout, char transr, char uplo,
                           lapack_int n, double* a );
lapack_int LAPACKE_cpftri( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_zpftri( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_complex_double* a );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: normal RFP; `'T'`/`'C'`: transposed/conjugate-transposed RFP |
| `uplo` | `char` | `'U'` or `'L'`: must match pftrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** RFP factor from pftrf; overwritten with inverse |

**Returns:** `info` -- 0 on success; `> 0` if the `info`-th diagonal element is zero.

---

### pftrs

Solves `A * X = B` using RFP Cholesky factorization from pftrf.

```c
lapack_int LAPACKE_spftrs( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_int nrhs, const float* a,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dpftrs( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_int nrhs, const double* a,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_cpftrs( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpftrs( int matrix_layout, char transr, char uplo,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: normal RFP; `'T'`/`'C'`: transposed/conjugate-transposed RFP |
| `uplo` | `char` | `'U'` or `'L'`: must match pftrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** RFP Cholesky factor from pftrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### pstrf

Computes the Cholesky factorization with complete pivoting of a symmetric/Hermitian positive semi-definite matrix: `P^T * A * P = U^T * U` or `P^T * A * P = L * L^T`.

```c
lapack_int LAPACKE_spstrf( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda, lapack_int* piv, lapack_int* rank,
                           float tol );
lapack_int LAPACKE_dpstrf( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda, lapack_int* piv, lapack_int* rank,
                           double tol );
lapack_int LAPACKE_cpstrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* piv, lapack_int* rank, float tol );
lapack_int LAPACKE_zpstrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* piv, lapack_int* rank, double tol );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle; `'L'`: lower triangle |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Symmetric/Hermitian matrix; overwritten with factor |
| `lda` | `lapack_int` | Leading dimension of a |
| `piv` | `lapack_int*` | **[out]** Pivot indices, dimension n |
| `rank` | `lapack_int*` | **[out]** Computed rank of A given tolerance |
| `tol` | `float`/`double` | Tolerance for rank determination; use -1 for default |

**Returns:** `info` -- 0 on success; `> 0` if matrix is not positive semi-definite (factorization stopped at step `info`).

---

## LDL^T / LDL^H Factorization

Computes the Bunch-Kaufman factorization: `A = U * D * U^T` (uplo='U') or `A = L * D * L^T` (uplo='L') for symmetric matrices, and `A = U * D * U^H` or `A = L * D * L^H` for Hermitian matrices. `D` is block diagonal with 1-by-1 and 2-by-2 diagonal blocks.

### sytrf

Computes the Bunch-Kaufman factorization of a symmetric matrix.

```c
lapack_int LAPACKE_ssytrf( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dsytrf( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_csytrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zsytrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle stored; `'L'`: lower triangle stored |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Symmetric matrix; overwritten with block diagonal D and multipliers |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices encoding the permutation and block structure |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero (singular block diagonal).

---

### sytrs

Solves `A * X = B` using the factorization from sytrf.

```c
lapack_int LAPACKE_ssytrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsytrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csytrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsytrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match sytrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Factorization from sytrf |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from sytrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### sytri

Computes the inverse of a symmetric matrix using the factorization from sytrf.

```c
lapack_int LAPACKE_ssytri( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dsytri( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_csytri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zsytri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match sytrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Factorization from sytrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from sytrf |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero (singular, cannot invert).

---

### sytri2

Computes the inverse of a symmetric indefinite matrix using the factorization from sytrf. Uses unblocked algorithm with Level 2 BLAS.

```c
lapack_int LAPACKE_ssytri2( int matrix_layout, char uplo, lapack_int n, float* a,
                            lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dsytri2( int matrix_layout, char uplo, lapack_int n,
                            double* a, lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_csytri2( int matrix_layout, char uplo, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            const lapack_int* ipiv );
lapack_int LAPACKE_zsytri2( int matrix_layout, char uplo, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            const lapack_int* ipiv );
```

**Parameters:** Same as [sytri](#sytri).

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### sytri2x

Computes the inverse of a symmetric indefinite matrix using the factorization from sytrf. Uses blocked algorithm with Level 3 BLAS.

```c
lapack_int LAPACKE_ssytri2x( int matrix_layout, char uplo, lapack_int n,
                             float* a, lapack_int lda, const lapack_int* ipiv,
                             lapack_int nb );
lapack_int LAPACKE_dsytri2x( int matrix_layout, char uplo, lapack_int n,
                             double* a, lapack_int lda, const lapack_int* ipiv,
                             lapack_int nb );
lapack_int LAPACKE_csytri2x( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
lapack_int LAPACKE_zsytri2x( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match sytrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Factorization from sytrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from sytrf |
| `nb` | `lapack_int` | Block size for blocked algorithm |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### sytrs2

Solves `A * X = B` using the factorization from sytrf. Revised algorithm with improved performance.

```c
lapack_int LAPACKE_ssytrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const float* a, lapack_int lda,
                            const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsytrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const double* a, lapack_int lda,
                            const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csytrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_float* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsytrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_double* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same as [sytrs](#sytrs).

**Returns:** `info` -- 0 on success.

---

### hetrf

Computes the Bunch-Kaufman factorization of a Hermitian matrix. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zhetrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle stored; `'L'`: lower triangle stored |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Hermitian matrix; overwritten with block diagonal D and multipliers |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices encoding the permutation and block structure |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### hetrs

Solves `A * X = B` using the factorization from hetrf. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhetrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match hetrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Factorization from hetrf |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from hetrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### hetri

Computes the inverse of a Hermitian matrix using the factorization from hetrf. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zhetri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match hetrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Factorization from hetrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from hetrf |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### hetri2

Computes the inverse of a Hermitian indefinite matrix using the factorization from hetrf. Unblocked algorithm. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetri2( int matrix_layout, char uplo, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            const lapack_int* ipiv );
lapack_int LAPACKE_zhetri2( int matrix_layout, char uplo, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            const lapack_int* ipiv );
```

**Parameters:** Same as [hetri](#hetri).

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### hetri2x

Computes the inverse of a Hermitian indefinite matrix using the factorization from hetrf. Blocked algorithm. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetri2x( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
lapack_int LAPACKE_zhetri2x( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match hetrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Factorization from hetrf; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from hetrf |
| `nb` | `lapack_int` | Block size for blocked algorithm |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### hetrs2

Solves `A * X = B` using the factorization from hetrf. Revised algorithm with improved performance. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chetrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_float* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhetrs2( int matrix_layout, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_double* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same as [hetrs](#hetrs).

**Returns:** `info` -- 0 on success.

---

### sptrf

Computes the Bunch-Kaufman factorization of a symmetric matrix stored in packed format.

```c
lapack_int LAPACKE_ssptrf( int matrix_layout, char uplo, lapack_int n, float* ap,
                           lapack_int* ipiv );
lapack_int LAPACKE_dsptrf( int matrix_layout, char uplo, lapack_int n,
                           double* ap, lapack_int* ipiv );
lapack_int LAPACKE_csptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_int* ipiv );
lapack_int LAPACKE_zsptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle packed; `'L'`: lower triangle packed |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix, dimension n*(n+1)/2; overwritten with factorization |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### sptrs

Solves `A * X = B` using the packed factorization from sptrf.

```c
lapack_int LAPACKE_ssptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zsptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match sptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed factorization from sptrf |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from sptrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### sptri

Computes the inverse using the packed factorization from sptrf.

```c
lapack_int LAPACKE_ssptri( int matrix_layout, char uplo, lapack_int n, float* ap,
                           const lapack_int* ipiv );
lapack_int LAPACKE_dsptri( int matrix_layout, char uplo, lapack_int n,
                           double* ap, const lapack_int* ipiv );
lapack_int LAPACKE_csptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap, const lapack_int* ipiv );
lapack_int LAPACKE_zsptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap, const lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match sptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed factor from sptrf; overwritten with inverse |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from sptrf |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

### hptrf

Computes the Bunch-Kaufman factorization of a Hermitian matrix stored in packed format. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_int* ipiv );
lapack_int LAPACKE_zhptrf( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangle packed; `'L'`: lower triangle packed |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed Hermitian matrix, dimension n*(n+1)/2; overwritten with factorization |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### hptrs

Solves `A * X = B` using the packed factorization from hptrf. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zhptrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match hptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed factorization from hptrf |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from hptrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### hptri

Computes the inverse using the packed factorization from hptrf. Complex types only (`c`/`z`).

```c
lapack_int LAPACKE_chptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* ap, const lapack_int* ipiv );
lapack_int LAPACKE_zhptri( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* ap, const lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'`: must match hptrf |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed factor from hptrf; overwritten with inverse |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from hptrf |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is zero.

---

## Tridiagonal Factorization

### pttrf

Computes the L*D*L^T factorization of a symmetric/Hermitian positive definite tridiagonal matrix. No pivoting is performed.

```c
lapack_int LAPACKE_spttrf( lapack_int n, float* d, float* e );
lapack_int LAPACKE_dpttrf( lapack_int n, double* d, double* e );
lapack_int LAPACKE_cpttrf( lapack_int n, float* d, lapack_complex_float* e );
lapack_int LAPACKE_zpttrf( lapack_int n, double* d, lapack_complex_double* e );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of A (n >= 0) |
| `d` | `float*`/`double*` | **[in/out]** Diagonal elements, dimension n; overwritten with D |
| `e` | `T*` | **[in/out]** Subdiagonal elements, dimension n-1; overwritten with L multipliers |

Note: No `matrix_layout` parameter -- tridiagonal storage is layout-independent.

**Returns:** `info` -- 0 on success; `> 0` if the matrix is not positive definite.

---

### pttrs

Solves `A * X = B` using the L*D*L^T factorization from pttrf.

```c
lapack_int LAPACKE_spttrs( int matrix_layout, lapack_int n, lapack_int nrhs,
                           const float* d, const float* e, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dpttrs( int matrix_layout, lapack_int n, lapack_int nrhs,
                           const double* d, const double* e, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_cpttrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const float* d,
                           const lapack_complex_float* e,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpttrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* d,
                           const lapack_complex_double* e,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | Complex variants only: `'U'` or `'L'` specifies form of factorization |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `d` | `const float*`/`const double*` | **[in]** Diagonal factor D from pttrf |
| `e` | `const T*` | **[in]** Subdiagonal factor from pttrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

Note: The real variants (`s`/`d`) do not have the `uplo` parameter. The complex variants (`c`/`z`) require `uplo` to specify the factorization form.

**Returns:** `info` -- 0 on success.

---

### gttrf

Computes the LU factorization of a general tridiagonal matrix using partial pivoting and row interchanges.

```c
lapack_int LAPACKE_sgttrf( lapack_int n, float* dl, float* d, float* du,
                           float* du2, lapack_int* ipiv );
lapack_int LAPACKE_dgttrf( lapack_int n, double* dl, double* d, double* du,
                           double* du2, lapack_int* ipiv );
lapack_int LAPACKE_cgttrf( lapack_int n, lapack_complex_float* dl,
                           lapack_complex_float* d, lapack_complex_float* du,
                           lapack_complex_float* du2, lapack_int* ipiv );
lapack_int LAPACKE_zgttrf( lapack_int n, lapack_complex_double* dl,
                           lapack_complex_double* d, lapack_complex_double* du,
                           lapack_complex_double* du2, lapack_int* ipiv );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of A (n >= 0) |
| `dl` | `T*` | **[in/out]** Subdiagonal, dimension n-1; overwritten with L multipliers |
| `d` | `T*` | **[in/out]** Diagonal, dimension n; overwritten with D diagonal of U |
| `du` | `T*` | **[in/out]** Superdiagonal, dimension n-1; overwritten with first superdiagonal of U |
| `du2` | `T*` | **[out]** Second superdiagonal of U, dimension n-2 |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices, dimension n |

Note: No `matrix_layout` parameter -- tridiagonal storage is layout-independent.

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero.

---

### gttrs

Solves `A * X = B`, `A^T * X = B`, or `A^H * X = B` using the LU factorization from gttrf.

```c
lapack_int LAPACKE_sgttrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const float* dl, const float* d,
                           const float* du, const float* du2,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dgttrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const double* dl, const double* d,
                           const double* du, const double* du2,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_cgttrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* dl,
                           const lapack_complex_float* d,
                           const lapack_complex_float* du,
                           const lapack_complex_float* du2,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zgttrs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* dl,
                           const lapack_complex_double* d,
                           const lapack_complex_double* du,
                           const lapack_complex_double* du2,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`: solve `A*X=B`; `'T'`: solve `A^T*X=B`; `'C'`: solve `A^H*X=B` |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `dl` | `const T*` | **[in]** LU factor (subdiagonal) from gttrf |
| `d` | `const T*` | **[in]** LU factor (diagonal) from gttrf |
| `du` | `const T*` | **[in]** LU factor (first superdiagonal) from gttrf |
| `du2` | `const T*` | **[in]** LU factor (second superdiagonal) from gttrf |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from gttrf |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

## Triangular Operations

### trtri

Computes the inverse of a triangular matrix.

```c
lapack_int LAPACKE_strtri( int matrix_layout, char uplo, char diag, lapack_int n,
                           float* a, lapack_int lda );
lapack_int LAPACKE_dtrtri( int matrix_layout, char uplo, char diag, lapack_int n,
                           double* a, lapack_int lda );
lapack_int LAPACKE_ctrtri( int matrix_layout, char uplo, char diag, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztrtri( int matrix_layout, char uplo, char diag, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Triangular matrix; overwritten with inverse |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success; `> 0` if `A(info,info)` is zero (singular, cannot invert).

---

### trtrs

Solves `A * X = B`, `A^T * X = B`, or `A^H * X = B` where A is triangular.

```c
lapack_int LAPACKE_strtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* a,
                           lapack_int lda, float* b, lapack_int ldb );
lapack_int LAPACKE_dtrtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* a,
                           lapack_int lda, double* b, lapack_int ldb );
lapack_int LAPACKE_ctrtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztrtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`: solve `A*X=B`; `'T'`: solve `A^T*X=B`; `'C'`: solve `A^H*X=B` |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Triangular matrix A |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### tptri

Computes the inverse of a triangular matrix stored in packed format.

```c
lapack_int LAPACKE_stptri( int matrix_layout, char uplo, char diag, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dtptri( int matrix_layout, char uplo, char diag, lapack_int n,
                           double* ap );
lapack_int LAPACKE_ctptri( int matrix_layout, char uplo, char diag, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztptri( int matrix_layout, char uplo, char diag, lapack_int n,
                           lapack_complex_double* ap );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `T*` | **[in/out]** Packed triangular matrix, dimension n*(n+1)/2; overwritten with inverse |

**Returns:** `info` -- 0 on success; `> 0` if `A(info,info)` is zero (singular).

---

### tptrs

Solves `A * X = B`, `A^T * X = B`, or `A^H * X = B` where A is a packed triangular matrix.

```c
lapack_int LAPACKE_stptrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* ap,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dtptrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* ap,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_ctptrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* ap,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztptrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* ap,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed triangular matrix |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### tbcon

Estimates the reciprocal condition number of a triangular band matrix in either the 1-norm or infinity-norm.

```c
lapack_int LAPACKE_stbcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd, const float* ab,
                           lapack_int ldab, float* rcond );
lapack_int LAPACKE_dtbcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd, const double* ab,
                           lapack_int ldab, double* rcond );
lapack_int LAPACKE_ctbcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd,
                           const lapack_complex_float* ab, lapack_int ldab,
                           float* rcond );
lapack_int LAPACKE_ztbcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd,
                           const lapack_complex_double* ab, lapack_int ldab,
                           double* rcond );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | `'1'`/`'O'`: 1-norm; `'I'`: infinity-norm |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `kd` | `lapack_int` | Number of superdiagonals (uplo='U') or subdiagonals (uplo='L') |
| `ab` | `const T*` | **[in]** Triangular band matrix in banded storage |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `rcond` | `float*`/`double*` | **[out]** Reciprocal condition number estimate |

**Returns:** `info` -- 0 on success.

---

### tbrfs

Provides error bounds and backward error estimates for the solution of a triangular banded system.

```c
lapack_int LAPACKE_stbrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const float* ab, lapack_int ldab, const float* b,
                           lapack_int ldb, const float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dtbrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const double* ab, lapack_int ldab, const double* b,
                           lapack_int ldb, const double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_ctbrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztbrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`, `'T'`, or `'C'`: form of system solved |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `kd` | `lapack_int` | Number of superdiagonals or subdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `const T*` | **[in]** Triangular band matrix in banded storage |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `b` | `const T*` | **[in]** Original RHS matrix B |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `const T*` | **[in]** Computed solution X |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `float*`/`double*` | **[out]** Forward error bound for each solution vector, dimension nrhs |
| `berr` | `float*`/`double*` | **[out]** Backward error bound for each solution vector, dimension nrhs |

**Returns:** `info` -- 0 on success.

---

### tbtrs

Solves `A * X = B`, `A^T * X = B`, or `A^H * X = B` where A is a triangular band matrix.

```c
lapack_int LAPACKE_stbtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const float* ab, lapack_int ldab, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dtbtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const double* ab, lapack_int ldab, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_ctbtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztbtrs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`: solve `A*X=B`; `'T'`: solve `A^T*X=B`; `'C'`: solve `A^H*X=B` |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `kd` | `lapack_int` | Number of superdiagonals (uplo='U') or subdiagonals (uplo='L') |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `const T*` | **[in]** Triangular band matrix in banded storage |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `b` | `T*` | **[in/out]** RHS matrix B; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### trcon

Estimates the reciprocal condition number of a triangular matrix in either the 1-norm or infinity-norm.

```c
lapack_int LAPACKE_strcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const float* a, lapack_int lda,
                           float* rcond );
lapack_int LAPACKE_dtrcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const double* a, lapack_int lda,
                           double* rcond );
lapack_int LAPACKE_ctrcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_float* a,
                           lapack_int lda, float* rcond );
lapack_int LAPACKE_ztrcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_double* a,
                           lapack_int lda, double* rcond );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | `'1'`/`'O'`: 1-norm; `'I'`: infinity-norm |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `const T*` | **[in]** Triangular matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `rcond` | `float*`/`double*` | **[out]** Reciprocal condition number estimate |

**Returns:** `info` -- 0 on success.

---

### trrfs

Provides error bounds and backward error estimates for the solution of a triangular system.

```c
lapack_int LAPACKE_strrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* a,
                           lapack_int lda, const float* b, lapack_int ldb,
                           const float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_dtrrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* a,
                           lapack_int lda, const double* b, lapack_int ldb,
                           const double* x, lapack_int ldx, double* ferr,
                           double* berr );
lapack_int LAPACKE_ctrrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztrrfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`, `'T'`, or `'C'`: form of system solved |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Triangular matrix A |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `const T*` | **[in]** Original RHS matrix B |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `const T*` | **[in]** Computed solution X |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `float*`/`double*` | **[out]** Forward error bound for each solution vector, dimension nrhs |
| `berr` | `float*`/`double*` | **[out]** Backward error bound for each solution vector, dimension nrhs |

**Returns:** `info` -- 0 on success.

---

### tpcon

Estimates the reciprocal condition number of a packed triangular matrix in either the 1-norm or infinity-norm.

```c
lapack_int LAPACKE_stpcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const float* ap, float* rcond );
lapack_int LAPACKE_dtpcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const double* ap, double* rcond );
lapack_int LAPACKE_ctpcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_float* ap,
                           float* rcond );
lapack_int LAPACKE_ztpcon( int matrix_layout, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_double* ap,
                           double* rcond );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | `'1'`/`'O'`: 1-norm; `'I'`: infinity-norm |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `ap` | `const T*` | **[in]** Packed triangular matrix, dimension n*(n+1)/2 |
| `rcond` | `float*`/`double*` | **[out]** Reciprocal condition number estimate |

**Returns:** `info` -- 0 on success.

---

### tprfs

Provides error bounds and backward error estimates for the solution of a packed triangular system.

```c
lapack_int LAPACKE_stprfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* ap,
                           const float* b, lapack_int ldb, const float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dtprfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* ap,
                           const double* b, lapack_int ldb, const double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_ctprfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztprfs( int matrix_layout, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `trans` | `char` | `'N'`, `'T'`, or `'C'`: form of system solved |
| `diag` | `char` | `'N'`: non-unit diagonal; `'U'`: unit diagonal |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed triangular matrix |
| `b` | `const T*` | **[in]** Original RHS matrix B |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `const T*` | **[in]** Computed solution X |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `float*`/`double*` | **[out]** Forward error bound for each solution vector, dimension nrhs |
| `berr` | `float*`/`double*` | **[out]** Backward error bound for each solution vector, dimension nrhs |

**Returns:** `info` -- 0 on success.

---

### lauum

Computes the product `U * U^T` or `L^T * L`, where U (or L) is the triangular factor from a Cholesky factorization. The result is a symmetric/Hermitian positive definite matrix.

```c
lapack_int LAPACKE_slauum( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dlauum( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_clauum( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zlauum( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: compute `U * U^T`; `'L'`: compute `L^T * L` |
| `n` | `lapack_int` | Order of A (n >= 0) |
| `a` | `T*` | **[in/out]** Triangular factor; overwritten with product |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success.

---

## Matrix Storage Conversions

These routines convert between three triangular matrix storage formats:
- **Full (TR)**: Standard dense storage with leading dimension `lda`
- **Packed (TP)**: Column-major packed storage, dimension `n*(n+1)/2`
- **RFP (TF)**: Rectangular Full Packed format, dimension `n*(n+1)/2`

### tfttr

Converts a triangular matrix from RFP format to standard full format.

```c
lapack_int LAPACKE_stfttr( int matrix_layout, char transr, char uplo,
                           lapack_int n, const float* arf, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dtfttr( int matrix_layout, char transr, char uplo,
                           lapack_int n, const double* arf, double* a,
                           lapack_int lda );
lapack_int LAPACKE_ctfttr( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* arf,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztfttr( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* arf,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: arf in normal RFP; `'T'`/`'C'`: arf in transposed RFP |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `arf` | `const T*` | **[in]** Matrix in RFP format, dimension n*(n+1)/2 |
| `a` | `T*` | **[out]** Matrix in standard full format |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success.

---

### trttf

Converts a triangular matrix from standard full format to RFP format.

```c
lapack_int LAPACKE_strttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const float* a, lapack_int lda,
                           float* arf );
lapack_int LAPACKE_dtrttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const double* a, lapack_int lda,
                           double* arf );
lapack_int LAPACKE_ctrttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* arf );
lapack_int LAPACKE_ztrttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* arf );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: store in normal RFP; `'T'`/`'C'`: store in transposed RFP |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `a` | `const T*` | **[in]** Matrix in standard full format |
| `lda` | `lapack_int` | Leading dimension of a |
| `arf` | `T*` | **[out]** Matrix in RFP format, dimension n*(n+1)/2 |

**Returns:** `info` -- 0 on success.

---

### tfttp

Converts a triangular matrix from RFP format to packed format.

```c
lapack_int LAPACKE_stfttp( int matrix_layout, char transr, char uplo,
                           lapack_int n, const float* arf, float* ap );
lapack_int LAPACKE_dtfttp( int matrix_layout, char transr, char uplo,
                           lapack_int n, const double* arf, double* ap );
lapack_int LAPACKE_ctfttp( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* arf,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztfttp( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* arf,
                           lapack_complex_double* ap );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: arf in normal RFP; `'T'`/`'C'`: arf in transposed RFP |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `arf` | `const T*` | **[in]** Matrix in RFP format, dimension n*(n+1)/2 |
| `ap` | `T*` | **[out]** Matrix in packed format, dimension n*(n+1)/2 |

**Returns:** `info` -- 0 on success.

---

### tpttf

Converts a triangular matrix from packed format to RFP format.

```c
lapack_int LAPACKE_stpttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const float* ap, float* arf );
lapack_int LAPACKE_dtpttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const double* ap, double* arf );
lapack_int LAPACKE_ctpttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* ap,
                           lapack_complex_float* arf );
lapack_int LAPACKE_ztpttf( int matrix_layout, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* ap,
                           lapack_complex_double* arf );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'`: store in normal RFP; `'T'`/`'C'`: store in transposed RFP |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `ap` | `const T*` | **[in]** Matrix in packed format, dimension n*(n+1)/2 |
| `arf` | `T*` | **[out]** Matrix in RFP format, dimension n*(n+1)/2 |

**Returns:** `info` -- 0 on success.

---

### tpttr

Converts a triangular matrix from packed format to standard full format.

```c
lapack_int LAPACKE_stpttr( int matrix_layout, char uplo, lapack_int n,
                           const float* ap, float* a, lapack_int lda );
lapack_int LAPACKE_dtpttr( int matrix_layout, char uplo, lapack_int n,
                           const double* ap, double* a, lapack_int lda );
lapack_int LAPACKE_ctpttr( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztpttr( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `ap` | `const T*` | **[in]** Matrix in packed format, dimension n*(n+1)/2 |
| `a` | `T*` | **[out]** Matrix in standard full format |
| `lda` | `lapack_int` | Leading dimension of a |

**Returns:** `info` -- 0 on success.

---

### trttp

Converts a triangular matrix from standard full format to packed format.

```c
lapack_int LAPACKE_strttp( int matrix_layout, char uplo, lapack_int n,
                           const float* a, lapack_int lda, float* ap );
lapack_int LAPACKE_dtrttp( int matrix_layout, char uplo, lapack_int n,
                           const double* a, lapack_int lda, double* ap );
lapack_int LAPACKE_ctrttp( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztrttp( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* ap );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'`: upper triangular; `'L'`: lower triangular |
| `n` | `lapack_int` | Order of the matrix (n >= 0) |
| `a` | `const T*` | **[in]** Matrix in standard full format |
| `lda` | `lapack_int` | Leading dimension of a |
| `ap` | `T*` | **[out]** Matrix in packed format, dimension n*(n+1)/2 |

**Returns:** `info` -- 0 on success.
