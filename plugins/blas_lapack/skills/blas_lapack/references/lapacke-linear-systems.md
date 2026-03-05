# LAPACKE Linear System Solvers API Reference

> Routines for solving linear systems Ax = b: general, symmetric, Hermitian, banded, tridiagonal, and packed storage.
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [Common Parameters](#common-parameters)
- [General Solvers](#general-solvers)
  - [gesv - General linear solve](#gesv)
  - [gesvx - Expert general linear solve](#gesvx)
  - [gbsv - General banded solve](#gbsv)
  - [gbsvx - Expert general banded solve](#gbsvx)
  - [gtsv - General tridiagonal solve](#gtsv)
  - [gtsvx - Expert general tridiagonal solve](#gtsvx)
- [Symmetric/Hermitian Positive Definite Solvers](#symmetrichermitian-positive-definite-solvers)
  - [posv - Positive definite solve](#posv)
  - [posvx - Expert positive definite solve](#posvx)
  - [ppsv - Packed positive definite solve](#ppsv)
  - [ppsvx - Expert packed positive definite solve](#ppsvx)
  - [pbsv - Banded positive definite solve](#pbsv)
  - [pbsvx - Expert banded positive definite solve](#pbsvx)
  - [ptsv - Positive definite tridiagonal solve](#ptsv)
  - [ptsvx - Expert positive definite tridiagonal solve](#ptsvx)
- [Symmetric/Hermitian Indefinite Solvers](#symmetrichermitian-indefinite-solvers)
  - [sysv - Symmetric indefinite solve](#sysv)
  - [sysvx - Expert symmetric indefinite solve](#sysvx)
  - [hesv - Hermitian indefinite solve](#hesv)
  - [hesvx - Expert Hermitian indefinite solve](#hesvx)
  - [spsv - Symmetric packed indefinite solve](#spsv)
  - [spsvx - Expert symmetric packed indefinite solve](#spsvx)
  - [hpsv - Hermitian packed indefinite solve](#hpsv)
  - [hpsvx - Expert Hermitian packed indefinite solve](#hpsvx)
- [Iterative Refinement](#iterative-refinement)
  - [gerfs - General iterative refinement](#gerfs)
  - [gbrfs - General banded iterative refinement](#gbrfs)
  - [gtrfs - General tridiagonal iterative refinement](#gtrfs)
  - [porfs - Positive definite iterative refinement](#porfs)
  - [pbrfs - Banded positive definite iterative refinement](#pbrfs)
  - [pprfs - Packed positive definite iterative refinement](#pprfs)
  - [ptrfs - Positive definite tridiagonal iterative refinement](#ptrfs)
  - [syrfs - Symmetric iterative refinement](#syrfs)
  - [herfs - Hermitian iterative refinement](#herfs)
  - [sprfs - Symmetric packed iterative refinement](#sprfs)
  - [hprfs - Hermitian packed iterative refinement](#hprfs)
- [Condition Number Estimation](#condition-number-estimation)
  - [gecon - General condition number](#gecon)
  - [gbcon - General banded condition number](#gbcon)
  - [gtcon - General tridiagonal condition number](#gtcon)
  - [pocon - Positive definite condition number](#pocon)
  - [ppcon - Packed positive definite condition number](#ppcon)
  - [pbcon - Banded positive definite condition number](#pbcon)
  - [ptcon - Positive definite tridiagonal condition number](#ptcon)
  - [sycon - Symmetric condition number](#sycon)
  - [hecon - Hermitian condition number](#hecon)
  - [spcon - Symmetric packed condition number](#spcon)
  - [hpcon - Hermitian packed condition number](#hpcon)
- [Equilibration (Scaling)](#equilibration-scaling)
  - [geequ - General matrix equilibration](#geequ)
  - [geequb - General matrix equilibration (bounded)](#geequb)
  - [gbequ - General banded equilibration](#gbequ)
  - [gbequb - General banded equilibration (bounded)](#gbequb)
  - [poequ - Positive definite equilibration](#poequ)
  - [poequb - Positive definite equilibration (bounded)](#poequb)
  - [ppequ - Packed positive definite equilibration](#ppequ)
  - [syequb - Symmetric equilibration](#syequb)
  - [heequb - Hermitian equilibration](#heequb)
- [Mixed Precision and Variant Solvers](#mixed-precision-and-variant-solvers)
  - [dsgesv - Mixed precision general solve (double/single)](#dsgesv)
  - [zcgesv - Mixed precision general solve (complex double/single)](#zcgesv)
  - [dsposv - Mixed precision positive definite solve (double/single)](#dsposv)
  - [zcposv - Mixed precision positive definite solve (complex double/single)](#zcposv)
  - [sysv_rook - Symmetric solve with rook pivoting](#sysv_rook)
  - [sysv_rk - Symmetric solve with bounded Bunch-Kaufman](#sysv_rk)
  - [hesv_rk - Hermitian solve with bounded Bunch-Kaufman](#hesv_rk)
  - [sysv_aa - Symmetric solve with Aasen algorithm](#sysv_aa)
  - [hesv_aa - Hermitian solve with Aasen algorithm](#hesv_aa)
  - [sysv_aa_2stage - Symmetric solve with 2-stage Aasen](#sysv_aa_2stage)
  - [hesv_aa_2stage - Hermitian solve with 2-stage Aasen](#hesv_aa_2stage)

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
| `fact` | `'F'`, `'N'`, `'E'` | Factored, not factored, equilibrate then factor |
| `equed` | `'N'`, `'R'`, `'C'`, `'B'` | No equilibration, row, column, both |
| `norm` | `'1'`, `'O'`, `'I'` | 1-norm, 1-norm, infinity-norm |

### Return Value (`info`)

| Value | Meaning |
|-------|---------|
| `= 0` | Successful exit |
| `< 0` | The `-info`-th argument had an illegal value |
| `> 0` | Factorization-specific failure (e.g., singular matrix, not positive definite) |

---

## General Solvers

Solve `A * X = B` for general (non-symmetric) matrices using LU factorization with partial pivoting.

### gesv

Solves a general n-by-n system of linear equations `A * X = B` using the LU factorization computed by `getrf`. The matrix A is overwritten with its LU factors.

```c
lapack_int LAPACKE_sgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* a, lapack_int lda, lapack_int* ipiv, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* a, lapack_int lda, lapack_int* ipiv,
                          double* b, lapack_int ldb );
lapack_int LAPACKE_cgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides (columns of B) |
| `a` | `T*` | **[in/out]** n-by-n coefficient matrix; overwritten with L and U |
| `lda` | `lapack_int` | Leading dimension of a (lda >= max(1,n)) |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices, dimension n |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side matrix; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b (ldb >= max(1,n)) |

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### gesvx

Expert driver for solving `A * X = B` with condition estimation, error bounds, and optional equilibration. Can use a pre-computed LU factorization.

```c
lapack_int LAPACKE_dgesvx( int matrix_layout, char fact, char trans,
                           lapack_int n, lapack_int nrhs, double* a,
                           lapack_int lda, double* af, lapack_int ldaf,
                           lapack_int* ipiv, char* equed, double* r, double* c,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr,
                           double* rpivot );
```

**Variants:** `LAPACKE_sgesvx`, `LAPACKE_dgesvx`, `LAPACKE_cgesvx`, `LAPACKE_zgesvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored, `'E'` = equilibrate then factor |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` for A, A^T, or A^H |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `double*` | **[in/out]** n-by-n coefficient matrix; may be equilibrated on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `double*` | **[in/out]** n-by-n factored form of A (L and U from getrf) |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices from factorization |
| `equed` | `char*` | **[in/out]** Equilibration type: `'N'`, `'R'`, `'C'`, or `'B'` |
| `r` | `double*` | **[in/out]** Row scale factors, dimension n |
| `c` | `double*` | **[in/out]** Column scale factors, dimension n |
| `b` | `double*` | **[in/out]** n-by-nrhs right-hand side; may be equilibrated |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** n-by-nrhs solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number of A |
| `ferr` | `double*` | **[out]** Forward error bound for each solution vector, dimension nrhs |
| `berr` | `double*` | **[out]** Backward error bound for each solution vector, dimension nrhs |
| `rpivot` | `double*` | **[out]** Reciprocal pivot growth factor |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

### gbsv

Solves a general banded system `A * X = B` using LU factorization with partial pivoting. A is stored in band format.

```c
lapack_int LAPACKE_sgbsv( int matrix_layout, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs, float* ab,
                          lapack_int ldab, lapack_int* ipiv, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgbsv( int matrix_layout, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs, double* ab,
                          lapack_int ldab, lapack_int* ipiv, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cgbsv( int matrix_layout, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgbsv( int matrix_layout, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `kl` | `lapack_int` | Number of subdiagonals (kl >= 0) |
| `ku` | `lapack_int` | Number of superdiagonals (ku >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `T*` | **[in/out]** Band matrix A in band storage, dimension (ldab, n); overwritten with L and U |
| `ldab` | `lapack_int` | Leading dimension of ab (ldab >= 2*kl+ku+1) |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices, dimension n |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### gbsvx

Expert driver for solving a general banded system `A * X = B` with condition estimation, error bounds, and optional equilibration.

```c
lapack_int LAPACKE_dgbsvx( int matrix_layout, char fact, char trans,
                           lapack_int n, lapack_int kl, lapack_int ku,
                           lapack_int nrhs, double* ab, lapack_int ldab,
                           double* afb, lapack_int ldafb, lapack_int* ipiv,
                           char* equed, double* r, double* c, double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr,
                           double* rpivot );
```

**Variants:** `LAPACKE_sgbsvx`, `LAPACKE_dgbsvx`, `LAPACKE_cgbsvx`, `LAPACKE_zgbsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored, `'E'` = equilibrate then factor |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `n` | `lapack_int` | Order of matrix A |
| `kl` | `lapack_int` | Number of subdiagonals |
| `ku` | `lapack_int` | Number of superdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `double*` | **[in/out]** Band matrix A in band storage (ldab, n) |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `afb` | `double*` | **[in/out]** Factored band matrix (ldafb, n) |
| `ldafb` | `lapack_int` | Leading dimension of afb (ldafb >= 2*kl+ku+1) |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices |
| `equed` | `char*` | **[in/out]** Equilibration type |
| `r` | `double*` | **[in/out]** Row scale factors |
| `c` | `double*` | **[in/out]** Column scale factors |
| `b` | `double*` | **[in/out]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |
| `rpivot` | `double*` | **[out]** Reciprocal pivot growth factor |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

### gtsv

Solves a general tridiagonal system `A * X = B` using Gaussian elimination with partial pivoting. A is specified by its three diagonals.

```c
lapack_int LAPACKE_sgtsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* dl, float* d, float* du, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgtsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* dl, double* d, double* du, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cgtsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* dl, lapack_complex_float* d,
                          lapack_complex_float* du, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgtsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* dl, lapack_complex_double* d,
                          lapack_complex_double* du, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `dl` | `T*` | **[in/out]** Subdiagonal elements, dimension (n-1); overwritten |
| `d` | `T*` | **[in/out]** Diagonal elements, dimension n; overwritten |
| `du` | `T*` | **[in/out]** Superdiagonal elements, dimension (n-1); overwritten |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `U(info,info)` is exactly zero (singular).

---

### gtsvx

Expert driver for solving a general tridiagonal system `A * X = B` with condition estimation, error bounds, and optional factorization caching.

```c
lapack_int LAPACKE_dgtsvx( int matrix_layout, char fact, char trans,
                           lapack_int n, lapack_int nrhs, const double* dl,
                           const double* d, const double* du, double* dlf,
                           double* df, double* duf, double* du2,
                           lapack_int* ipiv, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_sgtsvx`, `LAPACKE_dgtsvx`, `LAPACKE_cgtsvx`, `LAPACKE_zgtsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `dl` | `const double*` | **[in]** Subdiagonal, dimension (n-1) |
| `d` | `const double*` | **[in]** Diagonal, dimension n |
| `du` | `const double*` | **[in]** Superdiagonal, dimension (n-1) |
| `dlf` | `double*` | **[in/out]** Factored subdiagonal (from gttrf), dimension (n-1) |
| `df` | `double*` | **[in/out]** Factored diagonal, dimension n |
| `duf` | `double*` | **[in/out]** Factored superdiagonal, dimension (n-1) |
| `du2` | `double*` | **[in/out]** Second superdiagonal of U, dimension (n-2) |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices from factorization |
| `b` | `const double*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds, dimension nrhs |
| `berr` | `double*` | **[out]** Backward error bounds, dimension nrhs |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

## Symmetric/Hermitian Positive Definite Solvers

Solve `A * X = B` where A is symmetric (real) or Hermitian (complex) positive definite, using Cholesky factorization.

### posv

Solves a symmetric/Hermitian positive definite system `A * X = B` using Cholesky factorization `A = U^T * U` or `A = L * L^T`.

```c
lapack_int LAPACKE_sposv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, float* a, lapack_int lda, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dposv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cposv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zposv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle stored, `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n SPD matrix; overwritten with Cholesky factor |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if leading minor of order `info` is not positive definite.

---

### posvx

Expert driver for solving a symmetric/Hermitian positive definite system with condition estimation, error bounds, and optional equilibration.

```c
lapack_int LAPACKE_dposvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* af, lapack_int ldaf, char* equed, double* s,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
```

**Variants:** `LAPACKE_sposvx`, `LAPACKE_dposvx`, `LAPACKE_cposvx`, `LAPACKE_zposvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored, `'E'` = equilibrate then factor |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `double*` | **[in/out]** n-by-n matrix; may be equilibrated |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `double*` | **[in/out]** Cholesky factor of A |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `equed` | `char*` | **[in/out]** `'N'` = no equilibration, `'Y'` = equilibrated |
| `s` | `double*` | **[in/out]** Scale factors, dimension n |
| `b` | `double*` | **[in/out]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds, dimension nrhs |
| `berr` | `double*` | **[out]** Backward error bounds, dimension nrhs |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite; `= n+1` if rcond < machine precision.

---

### ppsv

Solves a symmetric/Hermitian positive definite system using packed Cholesky factorization. The matrix A is stored in packed format (upper or lower triangle packed column-wise).

```c
lapack_int LAPACKE_sppsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, float* ap, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dppsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* ap, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cppsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zppsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `T*` | **[in/out]** Packed SPD matrix, dimension n*(n+1)/2; overwritten with Cholesky factor |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite.

---

### ppsvx

Expert driver for solving a packed positive definite system with condition estimation and error bounds.

```c
lapack_int LAPACKE_dppsvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, double* ap, double* afp,
                           char* equed, double* s, double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_sppsvx`, `LAPACKE_dppsvx`, `LAPACKE_cppsvx`, `LAPACKE_zppsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'`, `'N'`, or `'E'` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `double*` | **[in/out]** Packed matrix, dimension n*(n+1)/2 |
| `afp` | `double*` | **[in/out]** Packed Cholesky factor, dimension n*(n+1)/2 |
| `equed` | `char*` | **[in/out]** Equilibration type |
| `s` | `double*` | **[in/out]** Scale factors, dimension n |
| `b` | `double*` | **[in/out]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite; `= n+1` if rcond < machine precision.

---

### pbsv

Solves a symmetric/Hermitian positive definite banded system `A * X = B` using banded Cholesky factorization.

```c
lapack_int LAPACKE_spbsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs, float* ab,
                          lapack_int ldab, float* b, lapack_int ldb );
lapack_int LAPACKE_dpbsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs, double* ab,
                          lapack_int ldab, double* b, lapack_int ldb );
lapack_int LAPACKE_cpbsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpbsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of superdiagonals (uplo='U') or subdiagonals (uplo='L') |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `T*` | **[in/out]** Banded SPD matrix (ldab, n); overwritten with Cholesky factor |
| `ldab` | `lapack_int` | Leading dimension of ab (ldab >= kd+1) |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite.

---

### pbsvx

Expert driver for solving a banded positive definite system with condition estimation and error bounds.

```c
lapack_int LAPACKE_dpbsvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, double* ab,
                           lapack_int ldab, double* afb, lapack_int ldafb,
                           char* equed, double* s, double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_spbsvx`, `LAPACKE_dpbsvx`, `LAPACKE_cpbsvx`, `LAPACKE_zpbsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'`, `'N'`, or `'E'` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/subdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `double*` | **[in/out]** Banded matrix (ldab, n) |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `afb` | `double*` | **[in/out]** Factored banded matrix |
| `ldafb` | `lapack_int` | Leading dimension of afb |
| `equed` | `char*` | **[in/out]** Equilibration type |
| `s` | `double*` | **[in/out]** Scale factors |
| `b` | `double*` | **[in/out]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite; `= n+1` if rcond < machine precision.

---

### ptsv

Solves a symmetric/Hermitian positive definite tridiagonal system `A * X = B` using `L * D * L^T` factorization.

```c
lapack_int LAPACKE_sptsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* d, float* e, float* b, lapack_int ldb );
lapack_int LAPACKE_dptsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* d, double* e, double* b, lapack_int ldb );
lapack_int LAPACKE_cptsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          float* d, lapack_complex_float* e,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zptsv( int matrix_layout, lapack_int n, lapack_int nrhs,
                          double* d, lapack_complex_double* e,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `d` | `float*/double*` | **[in/out]** Diagonal elements, dimension n; overwritten with D from factorization |
| `e` | `T*` | **[in/out]** Off-diagonal elements, dimension (n-1); overwritten |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite.

---

### ptsvx

Expert driver for solving a positive definite tridiagonal system with condition estimation and error bounds.

```c
lapack_int LAPACKE_dptsvx( int matrix_layout, char fact, lapack_int n,
                           lapack_int nrhs, const double* d, const double* e,
                           double* df, double* ef, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
```

**Variants:** `LAPACKE_sptsvx`, `LAPACKE_dptsvx`, `LAPACKE_cptsvx`, `LAPACKE_zptsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `d` | `const double*` | **[in]** Diagonal elements, dimension n |
| `e` | `const double*` | **[in]** Off-diagonal elements, dimension (n-1) |
| `df` | `double*` | **[in/out]** Factored diagonal, dimension n |
| `ef` | `double*` | **[in/out]** Factored off-diagonal, dimension (n-1) |
| `b` | `const double*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite; `= n+1` if rcond < machine precision.

---

## Symmetric/Hermitian Indefinite Solvers

Solve `A * X = B` where A is symmetric (real/complex) or Hermitian (complex only) but not necessarily positive definite, using Bunch-Kaufman diagonal pivoting.

### sysv

Solves a symmetric indefinite system `A * X = B` using the Bunch-Kaufman factorization `A = U * D * U^T` or `A = L * D * L^T`.

```c
lapack_int LAPACKE_ssysv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, float* a, lapack_int lda,
                          lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsysv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda,
                          lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csysv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsysv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; overwritten with block-diagonal D and multipliers |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices from Bunch-Kaufman factorization |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero (singular block diagonal).

---

### sysvx

Expert driver for solving a symmetric indefinite system with condition estimation, error bounds, and factorization caching.

```c
lapack_int LAPACKE_dsysvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* af, lapack_int ldaf, lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* rcond, double* ferr,
                           double* berr );
```

**Variants:** `LAPACKE_ssysvx`, `LAPACKE_dsysvx`, `LAPACKE_csysvx`, `LAPACKE_zsysvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const double*` | **[in]** n-by-n symmetric matrix (original) |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `double*` | **[in/out]** Factored form of A |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices |
| `b` | `const double*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds, dimension nrhs |
| `berr` | `double*` | **[out]** Backward error bounds, dimension nrhs |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

### hesv

Solves a Hermitian indefinite system `A * X = B` using the Bunch-Kaufman factorization `A = U * D * U^H` or `A = L * D * L^H`. Complex types only.

```c
lapack_int LAPACKE_chesv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhesv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n Hermitian matrix; overwritten with D and multipliers |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### hesvx

Expert driver for solving a Hermitian indefinite system with condition estimation and error bounds. Complex types only.

```c
lapack_int LAPACKE_zhesvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* af,
                           lapack_int ldaf, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
```

**Variants:** `LAPACKE_chesvx`, `LAPACKE_zhesvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** n-by-n Hermitian matrix (original) |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `T*` | **[in/out]** Factored form of A |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices |
| `b` | `const T*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `T*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

### spsv

Solves a symmetric indefinite system with A in packed storage using Bunch-Kaufman diagonal pivoting.

```c
lapack_int LAPACKE_sspsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, float* ap, lapack_int* ipiv,
                          float* b, lapack_int ldb );
lapack_int LAPACKE_dspsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* ap, lapack_int* ipiv,
                          double* b, lapack_int ldb );
lapack_int LAPACKE_cspsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zspsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix, dimension n*(n+1)/2; overwritten with factorization |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### spsvx

Expert driver for solving a packed symmetric indefinite system with condition estimation and error bounds.

```c
lapack_int LAPACKE_dspsvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, double* afp,
                           lapack_int* ipiv, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_sspsvx`, `LAPACKE_dspsvx`, `LAPACKE_cspsvx`, `LAPACKE_zspsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const double*` | **[in]** Packed symmetric matrix, dimension n*(n+1)/2 |
| `afp` | `double*` | **[in/out]** Packed factored form, dimension n*(n+1)/2 |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices |
| `b` | `const double*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

### hpsv

Solves a Hermitian indefinite system with A in packed storage using Bunch-Kaufman diagonal pivoting. Complex types only.

```c
lapack_int LAPACKE_chpsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zhpsv( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `T*` | **[in/out]** Packed Hermitian matrix, dimension n*(n+1)/2; overwritten |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### hpsvx

Expert driver for solving a packed Hermitian indefinite system with condition estimation and error bounds. Complex types only.

```c
lapack_int LAPACKE_zhpsvx( int matrix_layout, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           lapack_complex_double* afp, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
```

**Variants:** `LAPACKE_chpsvx`, `LAPACKE_zhpsvx`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `fact` | `char` | `'F'` = factored, `'N'` = not factored |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Packed Hermitian matrix, dimension n*(n+1)/2 |
| `afp` | `T*` | **[in/out]** Packed factored form |
| `ipiv` | `lapack_int*` | **[in/out]** Pivot indices |
| `b` | `const T*` | **[in]** Right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `T*` | **[out]** Solution matrix |
| `ldx` | `lapack_int` | Leading dimension of x |
| `rcond` | `double*` | **[out]** Reciprocal condition number |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success; `> 0` if singular; `= n+1` if rcond < machine precision.

---

## Iterative Refinement

Improve the computed solution to a system of linear equations and provide error bounds. These routines require a pre-computed factorization and the original matrix.

### gerfs

Improves the computed solution to a general system `A * X = B` (or `A^T * X = B`) and provides forward and backward error bounds.

```c
lapack_int LAPACKE_dgerfs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_sgerfs`, `LAPACKE_dgerfs`, `LAPACKE_cgerfs`, `LAPACKE_zgerfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const double*` | **[in]** Original n-by-n matrix A |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `const double*` | **[in]** LU factored form of A (from getrf) |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from factorization |
| `b` | `const double*` | **[in]** Original right-hand side matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution matrix; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bound for each solution vector |
| `berr` | `double*` | **[out]** Backward error bound for each solution vector |

**Returns:** `info` -- 0 on success.

---

### gbrfs

Improves the computed solution to a general banded system and provides error bounds.

```c
lapack_int LAPACKE_dgbrfs( int matrix_layout, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const double* ab, lapack_int ldab, const double* afb,
                           lapack_int ldafb, const lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
```

**Variants:** `LAPACKE_sgbrfs`, `LAPACKE_dgbrfs`, `LAPACKE_cgbrfs`, `LAPACKE_zgbrfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `n` | `lapack_int` | Order of matrix A |
| `kl` | `lapack_int` | Number of subdiagonals |
| `ku` | `lapack_int` | Number of superdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `const double*` | **[in]** Original banded matrix (ldab, n) |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `afb` | `const double*` | **[in]** Factored banded matrix |
| `ldafb` | `lapack_int` | Leading dimension of afb |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### gtrfs

Improves the computed solution to a general tridiagonal system and provides error bounds.

```c
lapack_int LAPACKE_dgtrfs( int matrix_layout, char trans, lapack_int n,
                           lapack_int nrhs, const double* dl, const double* d,
                           const double* du, const double* dlf,
                           const double* df, const double* duf,
                           const double* du2, const lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
```

**Variants:** `LAPACKE_sgtrfs`, `LAPACKE_dgtrfs`, `LAPACKE_cgtrfs`, `LAPACKE_zgtrfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`, `'T'`, or `'C'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `dl` | `const double*` | **[in]** Original subdiagonal |
| `d` | `const double*` | **[in]** Original diagonal |
| `du` | `const double*` | **[in]** Original superdiagonal |
| `dlf` | `const double*` | **[in]** Factored subdiagonal |
| `df` | `const double*` | **[in]** Factored diagonal |
| `duf` | `const double*` | **[in]** Factored superdiagonal |
| `du2` | `const double*` | **[in]** Second superdiagonal of U |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### porfs

Improves the computed solution to a symmetric/Hermitian positive definite system and provides error bounds.

```c
lapack_int LAPACKE_dporfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_sporfs`, `LAPACKE_dporfs`, `LAPACKE_cporfs`, `LAPACKE_zporfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const double*` | **[in]** Original SPD matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `const double*` | **[in]** Cholesky factor of A |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### pbrfs

Improves the computed solution to a banded positive definite system and provides error bounds.

```c
lapack_int LAPACKE_dpbrfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const double* ab,
                           lapack_int ldab, const double* afb, lapack_int ldafb,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
```

**Variants:** `LAPACKE_spbrfs`, `LAPACKE_dpbrfs`, `LAPACKE_cpbrfs`, `LAPACKE_zpbrfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/subdiagonals |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ab` | `const double*` | **[in]** Original banded matrix |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `afb` | `const double*` | **[in]** Factored banded matrix |
| `ldafb` | `lapack_int` | Leading dimension of afb |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### pprfs

Improves the computed solution to a packed positive definite system and provides error bounds.

```c
lapack_int LAPACKE_dpprfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, const double* afp,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
```

**Variants:** `LAPACKE_spprfs`, `LAPACKE_dpprfs`, `LAPACKE_cpprfs`, `LAPACKE_zpprfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const double*` | **[in]** Original packed matrix |
| `afp` | `const double*` | **[in]** Packed Cholesky factor |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### ptrfs

Improves the computed solution to a positive definite tridiagonal system and provides error bounds.

```c
lapack_int LAPACKE_dptrfs( int matrix_layout, lapack_int n, lapack_int nrhs,
                           const double* d, const double* e, const double* df,
                           const double* ef, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* ferr,
                           double* berr );
```

**Variants:** `LAPACKE_sptrfs`, `LAPACKE_dptrfs`, `LAPACKE_cptrfs`, `LAPACKE_zptrfs`

Note: Complex variants (`cptrfs`, `zptrfs`) take an additional `uplo` parameter.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `d` | `const double*` | **[in]** Original diagonal |
| `e` | `const double*` | **[in]** Original off-diagonal |
| `df` | `const double*` | **[in]** Factored diagonal |
| `ef` | `const double*` | **[in]** Factored off-diagonal |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### syrfs

Improves the computed solution to a symmetric indefinite system and provides error bounds.

```c
lapack_int LAPACKE_dsyrfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_ssyrfs`, `LAPACKE_dsyrfs`, `LAPACKE_csyrfs`, `LAPACKE_zsyrfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const double*` | **[in]** Original symmetric matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `const double*` | **[in]** Factored form of A (from sytrf) |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices from factorization |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### herfs

Improves the computed solution to a Hermitian indefinite system and provides error bounds. Complex types only.

```c
lapack_int LAPACKE_zherfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_cherfs`, `LAPACKE_zherfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `const T*` | **[in]** Original Hermitian matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `af` | `const T*` | **[in]** Factored form of A (from hetrf) |
| `ldaf` | `lapack_int` | Leading dimension of af |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `b` | `const T*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `T*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### sprfs

Improves the computed solution to a packed symmetric indefinite system and provides error bounds.

```c
lapack_int LAPACKE_dsprfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, const double* afp,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_ssprfs`, `LAPACKE_dsprfs`, `LAPACKE_csprfs`, `LAPACKE_zsprfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const double*` | **[in]** Original packed symmetric matrix |
| `afp` | `const double*` | **[in]** Packed factored form |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `b` | `const double*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

### hprfs

Improves the computed solution to a packed Hermitian indefinite system and provides error bounds. Complex types only.

```c
lapack_int LAPACKE_zhprfs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_complex_double* afp,
                           const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );
```

**Variants:** `LAPACKE_chprfs`, `LAPACKE_zhprfs`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `ap` | `const T*` | **[in]** Original packed Hermitian matrix |
| `afp` | `const T*` | **[in]** Packed factored form |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `b` | `const T*` | **[in]** Original right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `T*` | **[in/out]** Solution; improved on exit |
| `ldx` | `lapack_int` | Leading dimension of x |
| `ferr` | `double*` | **[out]** Forward error bounds |
| `berr` | `double*` | **[out]** Backward error bounds |

**Returns:** `info` -- 0 on success.

---

## Condition Number Estimation

Estimate the reciprocal condition number of a matrix from its factored form. A small `rcond` indicates an ill-conditioned matrix.

### gecon

Estimates the reciprocal condition number of a general matrix in 1-norm or infinity-norm, using the LU factorization from `getrf`.

```c
lapack_int LAPACKE_dgecon( int matrix_layout, char norm, lapack_int n,
                           const double* a, lapack_int lda, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_sgecon`, `LAPACKE_dgecon`, `LAPACKE_cgecon`, `LAPACKE_zgecon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | `'1'` or `'O'` for 1-norm, `'I'` for infinity-norm |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const double*` | **[in]** LU-factored matrix (from getrf) |
| `lda` | `lapack_int` | Leading dimension of a |
| `anorm` | `double` | **[in]** Norm of the original matrix A (before factoring) |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### gbcon

Estimates the reciprocal condition number of a general banded matrix using its LU factorization.

```c
lapack_int LAPACKE_dgbcon( int matrix_layout, char norm, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* ab,
                           lapack_int ldab, const lapack_int* ipiv,
                           double anorm, double* rcond );
```

**Variants:** `LAPACKE_sgbcon`, `LAPACKE_dgbcon`, `LAPACKE_cgbcon`, `LAPACKE_zgbcon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | `'1'` or `'O'` for 1-norm, `'I'` for infinity-norm |
| `n` | `lapack_int` | Order of matrix A |
| `kl` | `lapack_int` | Number of subdiagonals |
| `ku` | `lapack_int` | Number of superdiagonals |
| `ab` | `const double*` | **[in]** LU-factored banded matrix |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** Norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### gtcon

Estimates the reciprocal condition number of a general tridiagonal matrix using its LU factorization.

```c
lapack_int LAPACKE_dgtcon( char norm, lapack_int n, const double* dl,
                           const double* d, const double* du, const double* du2,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_sgtcon`, `LAPACKE_dgtcon`, `LAPACKE_cgtcon`, `LAPACKE_zgtcon`

Note: `gtcon` does not take a `matrix_layout` parameter.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `norm` | `char` | `'1'` or `'O'` for 1-norm, `'I'` for infinity-norm |
| `n` | `lapack_int` | Order of matrix A |
| `dl` | `const double*` | **[in]** Factored subdiagonal (from gttrf) |
| `d` | `const double*` | **[in]** Factored diagonal |
| `du` | `const double*` | **[in]** Factored superdiagonal |
| `du2` | `const double*` | **[in]** Second superdiagonal of U |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** Norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### pocon

Estimates the reciprocal condition number of a symmetric/Hermitian positive definite matrix using its Cholesky factorization.

```c
lapack_int LAPACKE_dpocon( int matrix_layout, char uplo, lapack_int n,
                           const double* a, lapack_int lda, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_spocon`, `LAPACKE_dpocon`, `LAPACKE_cpocon`, `LAPACKE_zpocon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const double*` | **[in]** Cholesky factor (from potrf) |
| `lda` | `lapack_int` | Leading dimension of a |
| `anorm` | `double` | **[in]** 1-norm of original matrix A |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### ppcon

Estimates the reciprocal condition number of a packed positive definite matrix using its Cholesky factorization.

```c
lapack_int LAPACKE_dppcon( int matrix_layout, char uplo, lapack_int n,
                           const double* ap, double anorm, double* rcond );
```

**Variants:** `LAPACKE_sppcon`, `LAPACKE_dppcon`, `LAPACKE_cppcon`, `LAPACKE_zppcon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `const double*` | **[in]** Packed Cholesky factor |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### pbcon

Estimates the reciprocal condition number of a banded positive definite matrix using its Cholesky factorization.

```c
lapack_int LAPACKE_dpbcon( int matrix_layout, char uplo, lapack_int n,
                           lapack_int kd, const double* ab, lapack_int ldab,
                           double anorm, double* rcond );
```

**Variants:** `LAPACKE_spbcon`, `LAPACKE_dpbcon`, `LAPACKE_cpbcon`, `LAPACKE_zpbcon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/subdiagonals |
| `ab` | `const double*` | **[in]** Factored banded matrix |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### ptcon

Estimates the reciprocal condition number of a positive definite tridiagonal matrix using its `L*D*L^T` factorization.

```c
lapack_int LAPACKE_dptcon( lapack_int n, const double* d, const double* e,
                           double anorm, double* rcond );
```

**Variants:** `LAPACKE_sptcon`, `LAPACKE_dptcon`, `LAPACKE_cptcon`, `LAPACKE_zptcon`

Note: `ptcon` does not take a `matrix_layout` parameter.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of matrix A |
| `d` | `const double*` | **[in]** Factored diagonal (from pttrf) |
| `e` | `const double*` | **[in]** Factored off-diagonal |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### sycon

Estimates the reciprocal condition number of a symmetric indefinite matrix using its Bunch-Kaufman factorization.

```c
lapack_int LAPACKE_dsycon( int matrix_layout, char uplo, lapack_int n,
                           const double* a, lapack_int lda,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_ssycon`, `LAPACKE_dsycon`, `LAPACKE_csycon`, `LAPACKE_zsycon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const double*` | **[in]** Factored matrix (from sytrf) |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### hecon

Estimates the reciprocal condition number of a Hermitian indefinite matrix using its Bunch-Kaufman factorization. Complex types only.

```c
lapack_int LAPACKE_zhecon( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_checon`, `LAPACKE_zhecon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const T*` | **[in]** Factored matrix (from hetrf) |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### spcon

Estimates the reciprocal condition number of a packed symmetric indefinite matrix.

```c
lapack_int LAPACKE_dspcon( int matrix_layout, char uplo, lapack_int n,
                           const double* ap, const lapack_int* ipiv,
                           double anorm, double* rcond );
```

**Variants:** `LAPACKE_sspcon`, `LAPACKE_dspcon`, `LAPACKE_cspcon`, `LAPACKE_zspcon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `const double*` | **[in]** Packed factored matrix |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

### hpcon

Estimates the reciprocal condition number of a packed Hermitian indefinite matrix. Complex types only.

```c
lapack_int LAPACKE_zhpcon( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
```

**Variants:** `LAPACKE_chpcon`, `LAPACKE_zhpcon`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `const T*` | **[in]** Packed factored matrix |
| `ipiv` | `const lapack_int*` | **[in]** Pivot indices |
| `anorm` | `double` | **[in]** 1-norm of original matrix |
| `rcond` | `double*` | **[out]** Reciprocal condition number |

**Returns:** `info` -- 0 on success.

---

## Equilibration (Scaling)

Compute row and column scaling factors to equilibrate a matrix. Equilibration can improve the condition number before solving. These routines are called automatically by the expert (`*svx`) drivers when `fact = 'E'`.

### geequ

Computes row and column scaling factors for a general m-by-n matrix to equilibrate it.

```c
lapack_int LAPACKE_dgeequ( int matrix_layout, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda, double* r,
                           double* c, double* rowcnd, double* colcnd,
                           double* amax );
```

**Variants:** `LAPACKE_sgeequ`, `LAPACKE_dgeequ`, `LAPACKE_cgeequ`, `LAPACKE_zgeequ`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `a` | `const double*` | **[in]** m-by-n matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `r` | `double*` | **[out]** Row scale factors, dimension m |
| `c` | `double*` | **[out]** Column scale factors, dimension n |
| `rowcnd` | `double*` | **[out]** Ratio of smallest to largest row scale factor |
| `colcnd` | `double*` | **[out]** Ratio of smallest to largest column scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success; `> 0` if row/column `info` has a zero row/column.

---

### geequb

Computes row and column scaling factors for a general matrix. Bounded variant that avoids over/underflow compared to `geequ`.

```c
lapack_int LAPACKE_dgeequb( int matrix_layout, lapack_int m, lapack_int n,
                            const double* a, lapack_int lda, double* r,
                            double* c, double* rowcnd, double* colcnd,
                            double* amax );
```

**Variants:** `LAPACKE_sgeequb`, `LAPACKE_dgeequb`, `LAPACKE_cgeequb`, `LAPACKE_zgeequb`

**Parameters:** Same as `geequ`.

**Returns:** `info` -- 0 on success; `> 0` if row/column `info` has a zero row/column.

---

### gbequ

Computes row and column scaling factors for a general banded matrix.

```c
lapack_int LAPACKE_dgbequ( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* ab,
                           lapack_int ldab, double* r, double* c,
                           double* rowcnd, double* colcnd, double* amax );
```

**Variants:** `LAPACKE_sgbequ`, `LAPACKE_dgbequ`, `LAPACKE_cgbequ`, `LAPACKE_zgbequ`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `kl` | `lapack_int` | Number of subdiagonals |
| `ku` | `lapack_int` | Number of superdiagonals |
| `ab` | `const double*` | **[in]** Banded matrix in band storage |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `r` | `double*` | **[out]** Row scale factors |
| `c` | `double*` | **[out]** Column scale factors |
| `rowcnd` | `double*` | **[out]** Ratio of smallest to largest row scale factor |
| `colcnd` | `double*` | **[out]** Ratio of smallest to largest column scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success.

---

### gbequb

Computes row and column scaling factors for a general banded matrix. Bounded variant.

```c
lapack_int LAPACKE_dgbequb( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_int kl, lapack_int ku, const double* ab,
                            lapack_int ldab, double* r, double* c,
                            double* rowcnd, double* colcnd, double* amax );
```

**Variants:** `LAPACKE_sgbequb`, `LAPACKE_dgbequb`, `LAPACKE_cgbequb`, `LAPACKE_zgbequb`

**Parameters:** Same as `gbequ`.

**Returns:** `info` -- 0 on success.

---

### poequ

Computes scaling factors for a symmetric/Hermitian positive definite matrix.

```c
lapack_int LAPACKE_dpoequ( int matrix_layout, lapack_int n, const double* a,
                           lapack_int lda, double* s, double* scond,
                           double* amax );
```

**Variants:** `LAPACKE_spoequ`, `LAPACKE_dpoequ`, `LAPACKE_cpoequ`, `LAPACKE_zpoequ`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const double*` | **[in]** n-by-n SPD matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `s` | `double*` | **[out]** Scale factors, dimension n |
| `scond` | `double*` | **[out]** Ratio of smallest to largest scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success; `> 0` if diagonal element `info` is non-positive.

---

### poequb

Computes scaling factors for a positive definite matrix. Bounded variant.

```c
lapack_int LAPACKE_dpoequb( int matrix_layout, lapack_int n, const double* a,
                            lapack_int lda, double* s, double* scond,
                            double* amax );
```

**Variants:** `LAPACKE_spoequb`, `LAPACKE_dpoequb`, `LAPACKE_cpoequb`, `LAPACKE_zpoequb`

**Parameters:** Same as `poequ`.

**Returns:** `info` -- 0 on success; `> 0` if diagonal element `info` is non-positive.

---

### ppequ

Computes scaling factors for a packed positive definite matrix.

```c
lapack_int LAPACKE_dppequ( int matrix_layout, char uplo, lapack_int n,
                           const double* ap, double* s, double* scond,
                           double* amax );
```

**Variants:** `LAPACKE_sppequ`, `LAPACKE_dppequ`, `LAPACKE_cppequ`, `LAPACKE_zppequ`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `const double*` | **[in]** Packed SPD matrix, dimension n*(n+1)/2 |
| `s` | `double*` | **[out]** Scale factors, dimension n |
| `scond` | `double*` | **[out]** Ratio of smallest to largest scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success; `> 0` if diagonal element `info` is non-positive.

---

### syequb

Computes scaling factors to equilibrate a symmetric indefinite matrix. Bounded variant.

```c
lapack_int LAPACKE_dsyequb( int matrix_layout, char uplo, lapack_int n,
                            const double* a, lapack_int lda, double* s,
                            double* scond, double* amax );
```

**Variants:** `LAPACKE_ssyequb`, `LAPACKE_dsyequb`, `LAPACKE_csyequb`, `LAPACKE_zsyequb`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const double*` | **[in]** n-by-n symmetric matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `s` | `double*` | **[out]** Scale factors, dimension n |
| `scond` | `double*` | **[out]** Ratio of smallest to largest scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success.

---

### heequb

Computes scaling factors to equilibrate a Hermitian indefinite matrix. Complex types only.

```c
lapack_int LAPACKE_zheequb( int matrix_layout, char uplo, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            double* s, double* scond, double* amax );
```

**Variants:** `LAPACKE_cheequb`, `LAPACKE_zheequb`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const T*` | **[in]** n-by-n Hermitian matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `s` | `double*` | **[out]** Scale factors, dimension n |
| `scond` | `double*` | **[out]** Ratio of smallest to largest scale factor |
| `amax` | `double*` | **[out]** Absolute value of largest element |

**Returns:** `info` -- 0 on success.

---

## Mixed Precision and Variant Solvers

Specialized solver variants: mixed-precision iterative refinement, alternative pivoting strategies, and the Aasen algorithm.

### dsgesv

Mixed-precision iterative refinement solver for general systems. Computes the LU factorization in single precision, then refines the solution to double precision accuracy.

```c
lapack_int LAPACKE_dsgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                           double* a, lapack_int lda, lapack_int* ipiv,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           lapack_int* iter );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `double*` | **[in/out]** n-by-n matrix; overwritten with LU factors |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `double*` | **[in]** n-by-nrhs right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** n-by-nrhs solution |
| `ldx` | `lapack_int` | Leading dimension of x |
| `iter` | `lapack_int*` | **[out]** Number of iterative refinement iterations; < 0 if refinement failed and double precision fallback was used |

**Returns:** `info` -- 0 on success; `> 0` if singular.

---

### zcgesv

Mixed-precision iterative refinement solver for complex general systems. Factorizes in single-precision complex, refines to double-precision complex.

```c
lapack_int LAPACKE_zcgesv( int matrix_layout, lapack_int n, lapack_int nrhs,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* iter );
```

**Parameters:** Same structure as `dsgesv` but with `lapack_complex_double` types.

**Returns:** `info` -- 0 on success; `> 0` if singular.

---

### dsposv

Mixed-precision iterative refinement solver for SPD systems. Factorizes in single precision, refines to double precision.

```c
lapack_int LAPACKE_dsposv( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           lapack_int* iter );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `double*` | **[in/out]** n-by-n SPD matrix; overwritten with Cholesky factor |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `double*` | **[in]** n-by-nrhs right-hand side |
| `ldb` | `lapack_int` | Leading dimension of b |
| `x` | `double*` | **[out]** n-by-nrhs solution |
| `ldx` | `lapack_int` | Leading dimension of x |
| `iter` | `lapack_int*` | **[out]** Number of refinement iterations |

**Returns:** `info` -- 0 on success; `> 0` if not positive definite.

---

### zcposv

Mixed-precision iterative refinement solver for Hermitian positive definite systems. Factorizes in single-precision complex, refines to double-precision complex.

```c
lapack_int LAPACKE_zcposv( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* iter );
```

**Parameters:** Same structure as `dsposv` but with `lapack_complex_double` types.

**Returns:** `info` -- 0 on success; `> 0` if not positive definite.

---

### sysv_rook

Solves a symmetric indefinite system using the bounded Bunch-Kaufman ("rook") pivoting algorithm, which provides better stability than standard Bunch-Kaufman for certain matrices.

```c
lapack_int LAPACKE_dsysv_rook( int matrix_layout, char uplo, lapack_int n,
                               lapack_int nrhs, double* a, lapack_int lda,
                               lapack_int* ipiv, double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_ssysv_rook`, `LAPACKE_dsysv_rook`, `LAPACKE_csysv_rook`, `LAPACKE_zsysv_rook`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; overwritten with factorization |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `T*` | **[in/out]** n-by-nrhs right-hand side; overwritten with solution X |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### sysv_rk

Solves a symmetric indefinite system using the bounded Bunch-Kaufman factorization `A = P * U * D * U^T * P^T` (or lower variant). Also known as "rook" factorization with a different storage scheme that stores the block-diagonal factor D separately.

```c
lapack_int LAPACKE_dsysv_rk( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda,
                          double* e, lapack_int* ipiv, double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_ssysv_rk`, `LAPACKE_dsysv_rk`, `LAPACKE_csysv_rk`, `LAPACKE_zsysv_rk`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; overwritten with U (or L) |
| `lda` | `lapack_int` | Leading dimension of a |
| `e` | `T*` | **[out]** Superdiagonal (or subdiagonal) entries of D, dimension n |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices encoding permutation and block structure |
| `b` | `T*` | **[in/out]** Right-hand side; overwritten with solution |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### hesv_rk

Solves a Hermitian indefinite system using the bounded Bunch-Kaufman factorization with separate D storage. Complex types only.

```c
lapack_int LAPACKE_zhesv_rk( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* e, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_chesv_rk`, `LAPACKE_zhesv_rk`

**Parameters:** Same structure as `sysv_rk` but with complex types and Hermitian matrix.

**Returns:** `info` -- 0 on success; `> 0` if `D(info,info)` is exactly zero.

---

### sysv_aa

Solves a symmetric indefinite system using the Aasen algorithm. Factorizes `A = U * T * U^T` (or `L * T * L^T`) where T is tridiagonal and U/L is unit triangular.

```c
lapack_int LAPACKE_dsysv_aa( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda,
                          lapack_int* ipiv, double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_ssysv_aa`, `LAPACKE_dsysv_aa`, `LAPACKE_csysv_aa`, `LAPACKE_zsysv_aa`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; overwritten with factorization |
| `lda` | `lapack_int` | Leading dimension of a |
| `ipiv` | `lapack_int*` | **[out]** Pivot indices |
| `b` | `T*` | **[in/out]** Right-hand side; overwritten with solution |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if singular.

---

### hesv_aa

Solves a Hermitian indefinite system using the Aasen algorithm. Complex types only.

```c
lapack_int LAPACKE_zhesv_aa( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_chesv_aa`, `LAPACKE_zhesv_aa`

**Parameters:** Same structure as `sysv_aa` but with complex types.

**Returns:** `info` -- 0 on success; `> 0` if singular.

---

### sysv_aa_2stage

Solves a symmetric indefinite system using the 2-stage Aasen algorithm. Improved performance over `sysv_aa` for large matrices by using a band reduction to tridiagonal form in two stages.

```c
lapack_int LAPACKE_dsysv_aa_2stage( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda,
                          double* tb, lapack_int ltb,
                          lapack_int* ipiv, lapack_int* ipiv2,
                          double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_ssysv_aa_2stage`, `LAPACKE_dsysv_aa_2stage`, `LAPACKE_csysv_aa_2stage`, `LAPACKE_zsysv_aa_2stage`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; overwritten |
| `lda` | `lapack_int` | Leading dimension of a |
| `tb` | `T*` | **[out]** Band matrix T from factorization, dimension ltb |
| `ltb` | `lapack_int` | Length of tb (recommend 4*n) |
| `ipiv` | `lapack_int*` | **[out]** First set of pivot indices, dimension n |
| `ipiv2` | `lapack_int*` | **[out]** Second set of pivot indices, dimension n |
| `b` | `T*` | **[in/out]** Right-hand side; overwritten with solution |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success; `> 0` if singular.

---

### hesv_aa_2stage

Solves a Hermitian indefinite system using the 2-stage Aasen algorithm. Complex types only.

```c
lapack_int LAPACKE_zhesv_aa_2stage( int matrix_layout, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* tb,
                          lapack_int ltb, lapack_int* ipiv, lapack_int* ipiv2,
                          lapack_complex_double* b, lapack_int ldb );
```

**Variants:** `LAPACKE_chesv_aa_2stage`, `LAPACKE_zhesv_aa_2stage`

**Parameters:** Same structure as `sysv_aa_2stage` but with complex types.

**Returns:** `info` -- 0 on success; `> 0` if singular.
