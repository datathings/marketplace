# LAPACKE Eigenvalue API Reference

> Eigenvalue and eigenvector routines: symmetric, Hermitian, non-symmetric, generalized, and Schur decomposition.
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [Common Parameters](#common-parameters)
- [Symmetric Eigenvalue Problems](#symmetric-eigenvalue-problems)
  - [syev - Symmetric eigenvalues (QR)](#syev)
  - [syevd - Symmetric eigenvalues (divide-and-conquer)](#syevd)
  - [syevr - Symmetric eigenvalues (MRRR)](#syevr)
  - [syevx - Symmetric eigenvalues (bisection)](#syevx)
  - [sbev - Symmetric banded eigenvalues](#sbev)
  - [sbevd - Symmetric banded eigenvalues (divide-and-conquer)](#sbevd)
  - [sbevx - Symmetric banded eigenvalues (bisection)](#sbevx)
  - [spev - Symmetric packed eigenvalues](#spev)
  - [spevd - Symmetric packed eigenvalues (divide-and-conquer)](#spevd)
  - [spevx - Symmetric packed eigenvalues (bisection)](#spevx)
- [Hermitian Eigenvalue Problems](#hermitian-eigenvalue-problems)
  - [heev - Hermitian eigenvalues (QR)](#heev)
  - [heevd - Hermitian eigenvalues (divide-and-conquer)](#heevd)
  - [heevr - Hermitian eigenvalues (MRRR)](#heevr)
  - [heevx - Hermitian eigenvalues (bisection)](#heevx)
  - [hbev - Hermitian banded eigenvalues](#hbev)
  - [hbevd - Hermitian banded eigenvalues (divide-and-conquer)](#hbevd)
  - [hbevx - Hermitian banded eigenvalues (bisection)](#hbevx)
  - [hpev - Hermitian packed eigenvalues](#hpev)
  - [hpevd - Hermitian packed eigenvalues (divide-and-conquer)](#hpevd)
  - [hpevx - Hermitian packed eigenvalues (bisection)](#hpevx)
- [Non-symmetric Eigenvalue Problems](#non-symmetric-eigenvalue-problems)
  - [geev - General eigenvalues](#geev)
  - [geevx - General eigenvalues with balancing and condition](#geevx)
- [Schur Decomposition](#schur-decomposition)
  - [gees - Schur factorization](#gees)
  - [geesx - Schur factorization with condition estimates](#geesx)
- [Generalized Symmetric/Hermitian Eigenvalue Problems](#generalized-symmetrichermitian-eigenvalue-problems)
  - [sygv - Generalized symmetric eigenvalues](#sygv)
  - [sygvd - Generalized symmetric eigenvalues (divide-and-conquer)](#sygvd)
  - [sygvx - Generalized symmetric eigenvalues (bisection)](#sygvx)
  - [hegv - Generalized Hermitian eigenvalues](#hegv)
  - [hegvd - Generalized Hermitian eigenvalues (divide-and-conquer)](#hegvd)
  - [hegvx - Generalized Hermitian eigenvalues (bisection)](#hegvx)
- [Generalized Banded Eigenvalue Problems](#generalized-banded-eigenvalue-problems)
  - [sbgv - Generalized symmetric banded eigenvalues](#sbgv)
  - [sbgvd - Generalized symmetric banded (divide-and-conquer)](#sbgvd)
  - [sbgvx - Generalized symmetric banded (bisection)](#sbgvx)
  - [hbgv - Generalized Hermitian banded eigenvalues](#hbgv)
  - [hbgvd - Generalized Hermitian banded (divide-and-conquer)](#hbgvd)
  - [hbgvx - Generalized Hermitian banded (bisection)](#hbgvx)
- [Generalized Packed Eigenvalue Problems](#generalized-packed-eigenvalue-problems)
  - [spgv - Generalized symmetric packed eigenvalues](#spgv)
  - [spgvd - Generalized symmetric packed (divide-and-conquer)](#spgvd)
  - [spgvx - Generalized symmetric packed (bisection)](#spgvx)
  - [hpgv - Generalized Hermitian packed eigenvalues](#hpgv)
  - [hpgvd - Generalized Hermitian packed (divide-and-conquer)](#hpgvd)
  - [hpgvx - Generalized Hermitian packed (bisection)](#hpgvx)
- [Generalized Non-symmetric Eigenvalue Problems](#generalized-non-symmetric-eigenvalue-problems)
  - [ggev - Generalized eigenvalues](#ggev)
  - [ggev3 - Generalized eigenvalues (Level 3)](#ggev3)
  - [ggevx - Generalized eigenvalues with condition](#ggevx)
  - [gges - Generalized Schur factorization](#gges)
  - [gges3 - Generalized Schur factorization (Level 3)](#gges3)
  - [ggesx - Generalized Schur with condition estimates](#ggesx)
- [Sylvester Equation](#sylvester-equation)
  - [trsyl - Triangular Sylvester equation](#trsyl)
  - [trsyl3 - Triangular Sylvester equation (Level 3)](#trsyl3)
- [Reduction to Standard Form](#reduction-to-standard-form)
  - [sytrd - Symmetric tridiagonal reduction](#sytrd)
  - [hetrd - Hermitian tridiagonal reduction](#hetrd)
  - [sygst - Symmetric generalized to standard](#sygst)
  - [hegst - Hermitian generalized to standard](#hegst)
  - [gebrd - Bidiagonal reduction](#gebrd)
  - [gebal - Balance a general matrix](#gebal)
  - [gehrd - Upper Hessenberg reduction](#gehrd)
- [Tridiagonal and Bidiagonal Eigensolvers](#tridiagonal-and-bidiagonal-eigensolvers)
  - [sterf - Tridiagonal eigenvalues (root-free QR)](#sterf)
  - [steqr - Tridiagonal eigenvalues (QR)](#steqr)
  - [stev - Real symmetric tridiagonal eigenvalues](#stev)
  - [stevd - Real symmetric tridiagonal (divide-and-conquer)](#stevd)
  - [stevr - Real symmetric tridiagonal (MRRR)](#stevr)
  - [stevx - Real symmetric tridiagonal (bisection)](#stevx)
  - [pteqr - Positive definite tridiagonal eigenvalues](#pteqr)

## Common Parameters

### Precision Prefixes

| Prefix | Type | Description |
|--------|------|-------------|
| `s` | `float` | Single-precision real |
| `d` | `double` | Double-precision real |
| `c` | `lapack_complex_float` | Single-precision complex |
| `z` | `lapack_complex_double` | Double-precision complex |

### Eigenvalue Control Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `matrix_layout` | `LAPACK_ROW_MAJOR` (101), `LAPACK_COL_MAJOR` (102) | Storage order |
| `jobz` | `'N'`, `'V'` | `'N'` = eigenvalues only, `'V'` = eigenvalues and eigenvectors |
| `uplo` | `'U'`, `'L'` | Use upper or lower triangular part |
| `range` | `'A'`, `'V'`, `'I'` | `'A'` = all, `'V'` = half-open interval (vl,vu], `'I'` = indices il..iu |
| `jobvl` | `'N'`, `'V'` | Compute left eigenvectors? |
| `jobvr` | `'N'`, `'V'` | Compute right eigenvectors? |
| `jobvs` | `'N'`, `'V'` | Compute Schur vectors? |
| `itype` | `1`, `2`, `3` | Generalized problem type (see below) |

### Generalized Problem Types (`itype`)

| itype | Problem | Description |
|-------|---------|-------------|
| `1` | `A*x = lambda*B*x` | Standard generalized form |
| `2` | `A*B*x = lambda*x` | Multiplied form |
| `3` | `B*A*x = lambda*x` | Multiplied form (B on left) |

### Return Value (`info`)

| Value | Meaning |
|-------|---------|
| `= 0` | Successful exit |
| `< 0` | The `-info`-th argument had an illegal value |
| `> 0` | Algorithm-specific failure (e.g., failed to converge) |

---

## Symmetric Eigenvalue Problems

Computes eigenvalues and optionally eigenvectors of a real symmetric matrix `A`. All eigenvalues are real.

### syev

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix using the QR algorithm. The matrix is first reduced to tridiagonal form.

```c
lapack_int LAPACKE_ssyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          float* a, lapack_int lda, float* w );
lapack_int LAPACKE_dsyev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          double* a, lapack_int lda, double* w );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` = upper triangle stored; `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; if jobz='V', overwritten with eigenvectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### syevd

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix using a divide-and-conquer algorithm. Generally faster than syev for large matrices.

```c
lapack_int LAPACKE_ssyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           float* a, lapack_int lda, float* w );
lapack_int LAPACKE_dsyevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           double* a, lapack_int lda, double* w );
```

**Parameters:** Same as [syev](#syev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### syevr

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix using the Relatively Robust Representations (MRRR) algorithm. Recommended for computing a subset of eigenvalues.

```c
lapack_int LAPACKE_ssyevr( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, float* a, lapack_int lda, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_dsyevr( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, double* a, lapack_int lda, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* isuppz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all eigenvalues; `'V'` = in half-open interval (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` = upper triangle stored; `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; contents destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `vl` | `T` | Lower bound of interval (used if range='V') |
| `vu` | `T` | Upper bound of interval (used if range='V') |
| `il` | `lapack_int` | Index of smallest eigenvalue to compute (used if range='I', 1-based) |
| `iu` | `lapack_int` | Index of largest eigenvalue to compute (used if range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance; if 0, machine precision is used |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `isuppz` | `lapack_int*` | **[out]** Support of eigenvectors, dimension 2*max(1,m) |

**Returns:** `info` -- 0 on success; `> 0` if internal error occurred.

---

### syevx

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix using bisection followed by inverse iteration.

```c
lapack_int LAPACKE_ssyevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, float* a, lapack_int lda, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsyevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, double* a, lapack_int lda, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix; contents destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

### sbev

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric band matrix.

```c
lapack_int LAPACKE_ssbev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int kd, float* ab, lapack_int ldab, float* w,
                          float* z, lapack_int ldz );
lapack_int LAPACKE_dsbev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int kd, double* ab, lapack_int ldab, double* w,
                          double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` = upper triangle stored; `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `kd` | `lapack_int` | Number of super-diagonals (uplo='U') or sub-diagonals (uplo='L') |
| `ab` | `T*` | **[in/out]** Band matrix in banded storage, dimension ldab-by-n; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### sbevd

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric band matrix using divide-and-conquer.

```c
lapack_int LAPACKE_ssbevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int kd, float* ab, lapack_int ldab, float* w,
                           float* z, lapack_int ldz );
lapack_int LAPACKE_dsbevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int kd, double* ab, lapack_int ldab,
                           double* w, double* z, lapack_int ldz );
```

**Parameters:** Same as [sbev](#sbev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### sbevx

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric band matrix using bisection and inverse iteration.

```c
lapack_int LAPACKE_ssbevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd, float* ab,
                           lapack_int ldab, float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsbevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd, double* ab,
                           lapack_int ldab, double* q, lapack_int ldq,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/sub-diagonals |
| `ab` | `T*` | **[in/out]** Band matrix; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `q` | `T*` | **[out]** n-by-n orthogonal matrix from reduction to tridiagonal (if jobz='V') |
| `ldq` | `lapack_int` | Leading dimension of q |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

### spev

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage.

```c
lapack_int LAPACKE_sspev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          float* ap, float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          double* ap, double* w, double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix, dimension n*(n+1)/2; destroyed on exit |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### spevd

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage using divide-and-conquer.

```c
lapack_int LAPACKE_sspevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           float* ap, float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           double* ap, double* w, double* z, lapack_int ldz );
```

**Parameters:** Same as [spev](#spev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### spevx

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix in packed storage using bisection and inverse iteration.

```c
lapack_int LAPACKE_sspevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, float* ap, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dspevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, double* ap, double vl, double vu,
                           lapack_int il, lapack_int iu, double abstol,
                           lapack_int* m, double* w, double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix; destroyed on exit |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

## Hermitian Eigenvalue Problems

Computes eigenvalues and optionally eigenvectors of a complex Hermitian matrix. All eigenvalues are real.

### heev

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix using the QR algorithm.

```c
lapack_int LAPACKE_cheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* a, lapack_int lda, float* w );
lapack_int LAPACKE_zheev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* a, lapack_int lda, double* w );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` = upper triangle stored; `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A (n >= 0) |
| `a` | `complex T*` | **[in/out]** n-by-n Hermitian matrix; if jobz='V', overwritten with eigenvectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order, dimension n (always real) |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### heevd

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix using divide-and-conquer.

```c
lapack_int LAPACKE_cheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* w );
lapack_int LAPACKE_zheevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           double* w );
```

**Parameters:** Same as [heev](#heev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### heevr

Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix using MRRR.

```c
lapack_int LAPACKE_cheevr( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_zheevr( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, lapack_complex_double* z, lapack_int ldz,
                           lapack_int* isuppz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `complex T*` | **[in/out]** Hermitian matrix; contents destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `vl`, `vu` | `real T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `real T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order |
| `z` | `complex T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `isuppz` | `lapack_int*` | **[out]** Support of eigenvectors, dimension 2*max(1,m) |

**Returns:** `info` -- 0 on success; `> 0` if internal error occurred.

---

### heevx

Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix using bisection and inverse iteration.

```c
lapack_int LAPACKE_cheevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_zheevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:** Same as [heevr](#heevr) except `isuppz` is replaced by `ifail` (`lapack_int*`) -- **[out]** indices of eigenvectors that failed to converge.

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

### hbev

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian band matrix.

```c
lapack_int LAPACKE_chbev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int kd, lapack_complex_float* ab,
                          lapack_int ldab, float* w, lapack_complex_float* z,
                          lapack_int ldz );
lapack_int LAPACKE_zhbev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int kd, lapack_complex_double* ab,
                          lapack_int ldab, double* w, lapack_complex_double* z,
                          lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/sub-diagonals |
| `ab` | `complex T*` | **[in/out]** Band matrix in banded storage; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `complex T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### hbevd

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian band matrix using divide-and-conquer.

```c
lapack_int LAPACKE_chbevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_float* ab,
                           lapack_int ldab, float* w, lapack_complex_float* z,
                           lapack_int ldz );
lapack_int LAPACKE_zhbevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_double* ab,
                           lapack_int ldab, double* w, lapack_complex_double* z,
                           lapack_int ldz );
```

**Parameters:** Same as [hbev](#hbev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### hbevx

Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian band matrix using bisection and inverse iteration.

```c
lapack_int LAPACKE_chbevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhbevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* q, lapack_int ldq, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `kd` | `lapack_int` | Number of super/sub-diagonals |
| `ab` | `complex T*` | **[in/out]** Band matrix; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `q` | `complex T*` | **[out]** n-by-n unitary matrix from reduction to tridiagonal (if jobz='V') |
| `ldq` | `lapack_int` | Leading dimension of q |
| `vl`, `vu` | `real T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `real T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order |
| `z` | `complex T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

### hpev

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix in packed storage.

```c
lapack_int LAPACKE_chpev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* ap, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpev( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* ap, double* w,
                          lapack_complex_double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `complex T*` | **[in/out]** Packed Hermitian matrix, dimension n*(n+1)/2; destroyed on exit |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `complex T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### hpevd

Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix in packed storage using divide-and-conquer.

```c
lapack_int LAPACKE_chpevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* ap, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpevd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* ap, double* w,
                           lapack_complex_double* z, lapack_int ldz );
```

**Parameters:** Same as [hpev](#hpev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### hpevx

Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix in packed storage using bisection and inverse iteration.

```c
lapack_int LAPACKE_chpevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* ap, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhpevx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* ap, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `ap` | `complex T*` | **[in/out]** Packed Hermitian matrix; destroyed on exit |
| `vl`, `vu` | `real T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `real T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order |
| `z` | `complex T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

## Non-symmetric Eigenvalue Problems

Computes eigenvalues and optionally left/right eigenvectors of a general (non-symmetric) matrix. Eigenvalues may be complex even for real input.

### geev

Computes all eigenvalues and, optionally, left and/or right eigenvectors of a general matrix. For real matrices, complex eigenvalues occur in conjugate pairs.

```c
lapack_int LAPACKE_sgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* wr,
                          float* wi, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* wr,
                          double* wi, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_cgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* w, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_zgeev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* w,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobvl` | `char` | `'N'` = do not compute left eigenvectors; `'V'` = compute them |
| `jobvr` | `char` | `'N'` = do not compute right eigenvectors; `'V'` = compute them |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `double*` | **[in/out]** n-by-n matrix; destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `wr` | `double*` | **[out]** Real parts of eigenvalues, dimension n |
| `wi` | `double*` | **[out]** Imaginary parts of eigenvalues, dimension n |
| `vl` | `double*` | **[out]** Left eigenvectors (n-by-n if jobvl='V'); complex eigenvectors stored as consecutive pairs |
| `ldvl` | `lapack_int` | Leading dimension of vl |
| `vr` | `double*` | **[out]** Right eigenvectors (n-by-n if jobvr='V') |
| `ldvr` | `lapack_int` | Leading dimension of vr |

For complex variants, `wr`/`wi` are replaced by a single complex array `w`.

**Returns:** `info` -- 0 on success; `> 0` if QR algorithm failed to converge.

---

### geevx

Computes eigenvalues and optionally eigenvectors of a general matrix, with balancing to improve accuracy, and computes reciprocal condition numbers for eigenvalues and/or eigenvectors.

```c
lapack_int LAPACKE_dgeevx( int matrix_layout, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, double* a,
                           lapack_int lda, double* wr, double* wi, double* vl,
                           lapack_int ldvl, double* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, double* scale,
                           double* abnrm, double* rconde, double* rcondv );
lapack_int LAPACKE_zgeevx( int matrix_layout, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* w, lapack_complex_double* vl,
                           lapack_int ldvl, lapack_complex_double* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           double* scale, double* abnrm, double* rconde,
                           double* rcondv );
```

**Variants:** `LAPACKE_sgeevx`, `LAPACKE_dgeevx`, `LAPACKE_cgeevx`, `LAPACKE_zgeevx`

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `balanc` | `char` | `'N'` = none; `'P'` = permute; `'S'` = scale; `'B'` = both |
| `jobvl` | `char` | `'N'` or `'V'` (compute left eigenvectors?) |
| `jobvr` | `char` | `'N'` or `'V'` (compute right eigenvectors?) |
| `sense` | `char` | `'N'` = none; `'E'` = eigenvalue condition; `'V'` = eigenvector condition; `'B'` = both |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `double*` | **[in/out]** n-by-n matrix; destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `wr`, `wi` | `double*` | **[out]** Real and imaginary parts of eigenvalues |
| `vl`, `vr` | `double*` | **[out]** Left/right eigenvectors |
| `ldvl`, `ldvr` | `lapack_int` | Leading dimensions |
| `ilo`, `ihi` | `lapack_int*` | **[out]** Balancing indices |
| `scale` | `double*` | **[out]** Scaling factors from balancing, dimension n |
| `abnrm` | `double*` | **[out]** 1-norm of balanced matrix |
| `rconde` | `double*` | **[out]** Reciprocal condition numbers for eigenvalues, dimension n |
| `rcondv` | `double*` | **[out]** Reciprocal condition numbers for eigenvectors, dimension n |

**Returns:** `info` -- 0 on success; `> 0` if QR algorithm failed to converge.

---

## Schur Decomposition

Computes the Schur factorization `A = Z * T * Z^H` where `Z` is unitary/orthogonal and `T` is upper triangular (complex) or upper quasi-triangular (real).

### gees

Computes the Schur factorization of a general matrix, with optional ordering of eigenvalues on the diagonal of the Schur form.

```c
lapack_int LAPACKE_sgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_S_SELECT2 select, lapack_int n, float* a,
                          lapack_int lda, lapack_int* sdim, float* wr,
                          float* wi, float* vs, lapack_int ldvs );
lapack_int LAPACKE_dgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_D_SELECT2 select, lapack_int n, double* a,
                          lapack_int lda, lapack_int* sdim, double* wr,
                          double* wi, double* vs, lapack_int ldvs );
lapack_int LAPACKE_cgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_C_SELECT1 select, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_float* w,
                          lapack_complex_float* vs, lapack_int ldvs );
lapack_int LAPACKE_zgees( int matrix_layout, char jobvs, char sort,
                          LAPACK_Z_SELECT1 select, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_double* w,
                          lapack_complex_double* vs, lapack_int ldvs );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobvs` | `char` | `'N'` = do not compute Schur vectors; `'V'` = compute them |
| `sort` | `char` | `'N'` = eigenvalues not ordered; `'S'` = eigenvalues ordered by `select` |
| `select` | function ptr | Selection function: `LAPACK_D_SELECT2(double, double)` returns true if eigenvalue is selected (real: two args for real/imag parts; complex: one arg) |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `double*` | **[in/out]** Overwritten with Schur form T on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `sdim` | `lapack_int*` | **[out]** Number of selected eigenvalues (sort='S') |
| `wr`, `wi` | `double*` | **[out]** Real and imaginary parts of eigenvalues, dimension n |
| `vs` | `double*` | **[out]** Schur vectors (n-by-n if jobvs='V') |
| `ldvs` | `lapack_int` | Leading dimension of vs |

**Returns:** `info` -- 0 on success; `> 0` if QR algorithm failed to converge.

---

### geesx

Computes the Schur factorization with optional ordering and reciprocal condition numbers for the average of selected eigenvalues and for the associated invariant subspace.

```c
lapack_int LAPACKE_dgeesx( int matrix_layout, char jobvs, char sort,
                           LAPACK_D_SELECT2 select, char sense, lapack_int n,
                           double* a, lapack_int lda, lapack_int* sdim,
                           double* wr, double* wi, double* vs, lapack_int ldvs,
                           double* rconde, double* rcondv );
lapack_int LAPACKE_zgeesx( int matrix_layout, char jobvs, char sort,
                           LAPACK_Z_SELECT1 select, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* sdim, lapack_complex_double* w,
                           lapack_complex_double* vs, lapack_int ldvs,
                           double* rconde, double* rcondv );
```

**Variants:** `LAPACKE_sgeesx`, `LAPACKE_dgeesx`, `LAPACKE_cgeesx`, `LAPACKE_zgeesx`

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobvs` | `char` | `'N'` or `'V'` |
| `sort` | `char` | `'N'` or `'S'` |
| `select` | function ptr | Selection function for eigenvalue ordering |
| `sense` | `char` | `'N'` = none; `'E'` = eigenvalue condition; `'V'` = subspace condition; `'B'` = both |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `double*` | **[in/out]** Overwritten with Schur form T |
| `lda` | `lapack_int` | Leading dimension of a |
| `sdim` | `lapack_int*` | **[out]** Number of selected eigenvalues |
| `wr`, `wi` | `double*` | **[out]** Eigenvalues |
| `vs` | `double*` | **[out]** Schur vectors |
| `ldvs` | `lapack_int` | Leading dimension of vs |
| `rconde` | `double*` | **[out]** Reciprocal condition number for average of selected eigenvalues |
| `rcondv` | `double*` | **[out]** Reciprocal condition number for right invariant subspace |

**Returns:** `info` -- 0 on success; `> 0` if QR algorithm failed to converge.

---

## Generalized Symmetric/Hermitian Eigenvalue Problems

Computes eigenvalues and optionally eigenvectors of generalized symmetric/Hermitian problems: `A*x = lambda*B*x` (itype=1), `A*B*x = lambda*x` (itype=2), or `B*A*x = lambda*x` (itype=3), where B is positive definite.

### sygv

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem.

```c
lapack_int LAPACKE_ssygv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, float* a, lapack_int lda,
                          float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_dsygv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, double* a, lapack_int lda,
                          double* b, lapack_int ldb, double* w );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | Problem type: 1, 2, or 3 (see Common Parameters) |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `T*` | **[in/out]** n-by-n symmetric matrix A; if jobz='V', overwritten with eigenvectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `T*` | **[in/out]** n-by-n symmetric positive definite matrix B; overwritten with Cholesky factor |
| `ldb` | `lapack_int` | Leading dimension of b |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### sygvd

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem using divide-and-conquer.

```c
lapack_int LAPACKE_ssygvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, float* a, lapack_int lda,
                           float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_dsygvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* w );
```

**Parameters:** Same as [sygv](#sygv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### sygvx

Computes selected eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem using bisection and inverse iteration.

```c
lapack_int LAPACKE_ssygvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsygvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | Problem type: 1, 2, or 3 |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `T*` | **[in/out]** Symmetric matrix A; destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `T*` | **[in/out]** Symmetric positive definite B; overwritten with Cholesky factor |
| `ldb` | `lapack_int` | Leading dimension of b |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hegv

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem.

```c
lapack_int LAPACKE_chegv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_float* a,
                          lapack_int lda, lapack_complex_float* b,
                          lapack_int ldb, float* w );
lapack_int LAPACKE_zhegv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb, double* w );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | Problem type: 1, 2, or 3 |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `complex T*` | **[in/out]** Hermitian matrix A; if jobz='V', overwritten with eigenvectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `complex T*` | **[in/out]** Hermitian positive definite B; overwritten with Cholesky factor |
| `ldb` | `lapack_int` | Leading dimension of b |
| `w` | `real T*` | **[out]** Eigenvalues in ascending order, dimension n |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hegvd

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem using divide-and-conquer.

```c
lapack_int LAPACKE_chegvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* w );
lapack_int LAPACKE_zhegvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* w );
```

**Parameters:** Same as [hegv](#hegv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hegvx

Computes selected eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem using bisection and inverse iteration.

```c
lapack_int LAPACKE_chegvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhegvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:** Same as [sygvx](#sygvx) with complex types for `a`, `b`, `z` and real types for `w`, `vl`, `vu`, `abstol`.

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

## Generalized Banded Eigenvalue Problems

### sbgv

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem `A*x = lambda*B*x`.

```c
lapack_int LAPACKE_ssbgv( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb, float* ab,
                          lapack_int ldab, float* bb, lapack_int ldbb, float* w,
                          float* z, lapack_int ldz );
lapack_int LAPACKE_dsbgv( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb, double* ab,
                          lapack_int ldab, double* bb, lapack_int ldbb,
                          double* w, double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `ka` | `lapack_int` | Number of super/sub-diagonals of A |
| `kb` | `lapack_int` | Number of super/sub-diagonals of B |
| `ab` | `T*` | **[in/out]** Band matrix A; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `bb` | `T*` | **[in/out]** Band matrix B (positive definite); overwritten with Cholesky factor |
| `ldbb` | `lapack_int` | Leading dimension of bb |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### sbgvd

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem using divide-and-conquer.

```c
lapack_int LAPACKE_ssbgvd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, float* ab,
                           lapack_int ldab, float* bb, lapack_int ldbb,
                           float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dsbgvd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, double* ab,
                           lapack_int ldab, double* bb, lapack_int ldbb,
                           double* w, double* z, lapack_int ldz );
```

**Parameters:** Same as [sbgv](#sbgv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### sbgvx

Computes selected eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite banded eigenproblem.

```c
lapack_int LAPACKE_ssbgvx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           float* ab, lapack_int ldab, float* bb,
                           lapack_int ldbb, float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsbgvx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           double* ab, lapack_int ldab, double* bb,
                           lapack_int ldbb, double* q, lapack_int ldq,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `ka`, `kb` | `lapack_int` | Number of super/sub-diagonals of A, B |
| `ab` | `T*` | **[in/out]** Band matrix A; destroyed on exit |
| `ldab` | `lapack_int` | Leading dimension of ab |
| `bb` | `T*` | **[in/out]** Band matrix B; overwritten with Cholesky factor |
| `ldbb` | `lapack_int` | Leading dimension of bb |
| `q` | `T*` | **[out]** n-by-n orthogonal matrix from reduction (if jobz='V') |
| `ldq` | `lapack_int` | Leading dimension of q |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hbgv

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite banded eigenproblem.

```c
lapack_int LAPACKE_chbgv( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_complex_float* bb, lapack_int ldbb, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhbgv( int matrix_layout, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_complex_double* bb, lapack_int ldbb, double* w,
                          lapack_complex_double* z, lapack_int ldz );
```

**Parameters:** Same as [sbgv](#sbgv) with complex types for `ab`, `bb`, `z` and real type for `w`.

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hbgvd

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite banded eigenproblem using divide-and-conquer.

```c
lapack_int LAPACKE_chbgvd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* bb, lapack_int ldbb, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhbgvd( int matrix_layout, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* bb, lapack_int ldbb,
                           double* w, lapack_complex_double* z,
                           lapack_int ldz );
```

**Parameters:** Same as [hbgv](#hbgv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hbgvx

Computes selected eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite banded eigenproblem.

```c
lapack_int LAPACKE_chbgvx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* bb, lapack_int ldbb,
                           lapack_complex_float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhbgvx( int matrix_layout, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* bb, lapack_int ldbb,
                           lapack_complex_double* q, lapack_int ldq, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:** Same as [sbgvx](#sbgvx) with complex types for `ab`, `bb`, `q`, `z` and real types for `w`, `vl`, `vu`, `abstol`.

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

## Generalized Packed Eigenvalue Problems

### spgv

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem with matrices in packed storage.

```c
lapack_int LAPACKE_sspgv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, float* ap, float* bp,
                          float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspgv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, double* ap, double* bp,
                          double* w, double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | Problem type: 1, 2, or 3 |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix A, dimension n*(n+1)/2; destroyed on exit |
| `bp` | `T*` | **[in/out]** Packed symmetric positive definite B; overwritten with Cholesky factor |
| `w` | `T*` | **[out]** Eigenvalues in ascending order, dimension n |
| `z` | `T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### spgvd

Computes all eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem in packed storage using divide-and-conquer.

```c
lapack_int LAPACKE_sspgvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, float* ap, float* bp,
                           float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspgvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, double* ap, double* bp,
                           double* w, double* z, lapack_int ldz );
```

**Parameters:** Same as [spgv](#spgv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### spgvx

Computes selected eigenvalues and, optionally, eigenvectors of a real generalized symmetric-definite eigenproblem in packed storage.

```c
lapack_int LAPACKE_sspgvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, float* ap,
                           float* bp, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           float* z, lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_dspgvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, double* ap,
                           double* bp, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | Problem type: 1, 2, or 3 |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrices A and B |
| `ap` | `T*` | **[in/out]** Packed symmetric matrix A; destroyed on exit |
| `bp` | `T*` | **[in/out]** Packed symmetric positive definite B; overwritten with Cholesky factor |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `ifail` | `lapack_int*` | **[out]** Indices of eigenvectors that failed to converge |

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hpgv

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem with matrices in packed storage.

```c
lapack_int LAPACKE_chpgv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_float* ap,
                          lapack_complex_float* bp, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpgv( int matrix_layout, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_double* ap,
                          lapack_complex_double* bp, double* w,
                          lapack_complex_double* z, lapack_int ldz );
```

**Parameters:** Same as [spgv](#spgv) with complex types for `ap`, `bp`, `z` and real type for `w`.

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hpgvd

Computes all eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem in packed storage using divide-and-conquer.

```c
lapack_int LAPACKE_chpgvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_float* ap,
                           lapack_complex_float* bp, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpgvd( int matrix_layout, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_double* ap,
                           lapack_complex_double* bp, double* w,
                           lapack_complex_double* z, lapack_int ldz );
```

**Parameters:** Same as [hpgv](#hpgv).

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

### hpgvx

Computes selected eigenvalues and, optionally, eigenvectors of a complex generalized Hermitian-definite eigenproblem in packed storage.

```c
lapack_int LAPACKE_chpgvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_complex_float* bp,
                           float vl, float vu, lapack_int il, lapack_int iu,
                           float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_zhpgvx( int matrix_layout, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_complex_double* bp,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );
```

**Parameters:** Same as [spgvx](#spgvx) with complex types for `ap`, `bp`, `z` and real types for `w`, `vl`, `vu`, `abstol`.

**Returns:** `info` -- 0 on success; `> 0` if eigensolver or Cholesky factorization failed.

---

## Generalized Non-symmetric Eigenvalue Problems

Computes eigenvalues and optionally eigenvectors of generalized non-symmetric eigenvalue problems `A*x = lambda*B*x`. Eigenvalues are expressed as ratios `alpha/beta`.

### ggev

Computes all eigenvalues and, optionally, left and/or right eigenvectors of a generalized non-symmetric eigenproblem.

```c
lapack_int LAPACKE_sggev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* b,
                          lapack_int ldb, float* alphar, float* alphai,
                          float* beta, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_dggev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* b,
                          lapack_int ldb, double* alphar, double* alphai,
                          double* beta, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_cggev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb,
                          lapack_complex_float* alpha,
                          lapack_complex_float* beta, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_zggev( int matrix_layout, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb, lapack_complex_double* alpha,
                          lapack_complex_double* beta,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobvl` | `char` | `'N'` or `'V'` (compute left eigenvectors?) |
| `jobvr` | `char` | `'N'` or `'V'` (compute right eigenvectors?) |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `double*` | **[in/out]** n-by-n matrix A; destroyed on exit |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `double*` | **[in/out]** n-by-n matrix B; destroyed on exit |
| `ldb` | `lapack_int` | Leading dimension of b |
| `alphar` | `double*` | **[out]** Real parts of numerators of eigenvalues, dimension n |
| `alphai` | `double*` | **[out]** Imaginary parts of numerators of eigenvalues, dimension n |
| `beta` | `double*` | **[out]** Denominators of eigenvalues, dimension n; eigenvalue j = (alphar[j]+i*alphai[j])/beta[j] |
| `vl` | `double*` | **[out]** Left eigenvectors (n-by-n if jobvl='V') |
| `ldvl` | `lapack_int` | Leading dimension of vl |
| `vr` | `double*` | **[out]** Right eigenvectors (n-by-n if jobvr='V') |
| `ldvr` | `lapack_int` | Leading dimension of vr |

For complex variants, `alphar`/`alphai` are replaced by a single complex array `alpha`.

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

### ggev3

Computes generalized eigenvalues and eigenvectors using Level 3 BLAS. Same interface as ggev but with improved performance for large matrices.

```c
lapack_int LAPACKE_dggev3( int matrix_layout, char jobvl, char jobvr,
                           lapack_int n, double* a, lapack_int lda,
                           double* b, lapack_int ldb,
                           double* alphar, double* alphai, double* beta,
                           double* vl, lapack_int ldvl,
                           double* vr, lapack_int ldvr );
lapack_int LAPACKE_zggev3( int matrix_layout, char jobvl, char jobvr,
                           lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vl, lapack_int ldvl,
                           lapack_complex_double* vr, lapack_int ldvr );
```

**Variants:** `LAPACKE_sggev3`, `LAPACKE_dggev3`, `LAPACKE_cggev3`, `LAPACKE_zggev3`

**Parameters:** Same as [ggev](#ggev).

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

### ggevx

Computes generalized eigenvalues with balancing, and reciprocal condition numbers for eigenvalues and eigenvectors.

```c
lapack_int LAPACKE_dggevx( int matrix_layout, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double* alphar, double* alphai, double* beta,
                           double* vl, lapack_int ldvl, double* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           double* lscale, double* rscale, double* abnrm,
                           double* bbnrm, double* rconde, double* rcondv );
lapack_int LAPACKE_zggevx( int matrix_layout, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vl, lapack_int ldvl,
                           lapack_complex_double* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, double* lscale,
                           double* rscale, double* abnrm, double* bbnrm,
                           double* rconde, double* rcondv );
```

**Variants:** `LAPACKE_sggevx`, `LAPACKE_dggevx`, `LAPACKE_cggevx`, `LAPACKE_zggevx`

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `balanc` | `char` | `'N'` = none; `'P'` = permute; `'S'` = scale; `'B'` = both |
| `jobvl`, `jobvr` | `char` | `'N'` or `'V'` |
| `sense` | `char` | `'N'` = none; `'E'` = eigenvalue condition; `'V'` = eigenvector condition; `'B'` = both |
| `n` | `lapack_int` | Order of matrices A and B |
| `a`, `b` | `double*` | **[in/out]** n-by-n matrices; destroyed on exit |
| `lda`, `ldb` | `lapack_int` | Leading dimensions |
| `alphar`, `alphai` | `double*` | **[out]** Eigenvalue numerators |
| `beta` | `double*` | **[out]** Eigenvalue denominators |
| `vl`, `vr` | `double*` | **[out]** Left/right eigenvectors |
| `ldvl`, `ldvr` | `lapack_int` | Leading dimensions |
| `ilo`, `ihi` | `lapack_int*` | **[out]** Balancing indices |
| `lscale`, `rscale` | `double*` | **[out]** Left/right scaling factors, dimension n |
| `abnrm`, `bbnrm` | `double*` | **[out]** 1-norms of balanced A and B |
| `rconde` | `double*` | **[out]** Reciprocal condition numbers for eigenvalues, dimension n |
| `rcondv` | `double*` | **[out]** Reciprocal condition numbers for eigenvectors, dimension n |

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

### gges

Computes the generalized Schur factorization `A = Q*S*Z^T`, `B = Q*T*Z^T` with optional ordering of eigenvalues.

```c
lapack_int LAPACKE_sgges( int matrix_layout, char jobvsl, char jobvsr, char sort,
                          LAPACK_S_SELECT3 selctg, lapack_int n, float* a,
                          lapack_int lda, float* b, lapack_int ldb,
                          lapack_int* sdim, float* alphar, float* alphai,
                          float* beta, float* vsl, lapack_int ldvsl, float* vsr,
                          lapack_int ldvsr );
lapack_int LAPACKE_dgges( int matrix_layout, char jobvsl, char jobvsr, char sort,
                          LAPACK_D_SELECT3 selctg, lapack_int n, double* a,
                          lapack_int lda, double* b, lapack_int ldb,
                          lapack_int* sdim, double* alphar, double* alphai,
                          double* beta, double* vsl, lapack_int ldvsl,
                          double* vsr, lapack_int ldvsr );
lapack_int LAPACKE_cgges( int matrix_layout, char jobvsl, char jobvsr, char sort,
                          LAPACK_C_SELECT2 selctg, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb,
                          lapack_int* sdim, lapack_complex_float* alpha,
                          lapack_complex_float* beta, lapack_complex_float* vsl,
                          lapack_int ldvsl, lapack_complex_float* vsr,
                          lapack_int ldvsr );
lapack_int LAPACKE_zgges( int matrix_layout, char jobvsl, char jobvsr, char sort,
                          LAPACK_Z_SELECT2 selctg, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb,
                          lapack_int* sdim, lapack_complex_double* alpha,
                          lapack_complex_double* beta,
                          lapack_complex_double* vsl, lapack_int ldvsl,
                          lapack_complex_double* vsr, lapack_int ldvsr );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobvsl` | `char` | `'N'` or `'V'` (compute left Schur vectors?) |
| `jobvsr` | `char` | `'N'` or `'V'` (compute right Schur vectors?) |
| `sort` | `char` | `'N'` = not ordered; `'S'` = ordered by `selctg` |
| `selctg` | function ptr | `LAPACK_D_SELECT3(double, double, double)` returns true if eigenvalue alpha/beta is selected (real: three args alphar,alphai,beta; complex: two args alpha,beta) |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `double*` | **[in/out]** Overwritten with generalized Schur form S |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `double*` | **[in/out]** Overwritten with generalized Schur form T |
| `ldb` | `lapack_int` | Leading dimension of b |
| `sdim` | `lapack_int*` | **[out]** Number of selected eigenvalues |
| `alphar`, `alphai` | `double*` | **[out]** Eigenvalue numerators |
| `beta` | `double*` | **[out]** Eigenvalue denominators |
| `vsl` | `double*` | **[out]** Left Schur vectors (n-by-n if jobvsl='V') |
| `ldvsl` | `lapack_int` | Leading dimension of vsl |
| `vsr` | `double*` | **[out]** Right Schur vectors (n-by-n if jobvsr='V') |
| `ldvsr` | `lapack_int` | Leading dimension of vsr |

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

### gges3

Computes the generalized Schur factorization using Level 3 BLAS. Same interface as gges but with improved performance.

```c
lapack_int LAPACKE_dgges3( int matrix_layout, char jobvsl, char jobvsr,
                           char sort, LAPACK_D_SELECT3 selctg, lapack_int n,
                           double* a, lapack_int lda, double* b, lapack_int ldb,
                           lapack_int* sdim, double* alphar, double* alphai,
                           double* beta, double* vsl, lapack_int ldvsl,
                           double* vsr, lapack_int ldvsr );
lapack_int LAPACKE_zgges3( int matrix_layout, char jobvsl, char jobvsr,
                           char sort, LAPACK_Z_SELECT2 selctg, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_int* sdim, lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vsl, lapack_int ldvsl,
                           lapack_complex_double* vsr, lapack_int ldvsr );
```

**Variants:** `LAPACKE_sgges3`, `LAPACKE_dgges3`, `LAPACKE_cgges3`, `LAPACKE_zgges3`

**Parameters:** Same as [gges](#gges).

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

### ggesx

Computes the generalized Schur factorization with reciprocal condition numbers for the average of the selected eigenvalues and for the associated deflating subspaces.

```c
lapack_int LAPACKE_dggesx( int matrix_layout, char jobvsl, char jobvsr,
                           char sort, LAPACK_D_SELECT3 selctg, char sense,
                           lapack_int n, double* a, lapack_int lda, double* b,
                           lapack_int ldb, lapack_int* sdim, double* alphar,
                           double* alphai, double* beta, double* vsl,
                           lapack_int ldvsl, double* vsr, lapack_int ldvsr,
                           double* rconde, double* rcondv );
lapack_int LAPACKE_zggesx( int matrix_layout, char jobvsl, char jobvsr,
                           char sort, LAPACK_Z_SELECT2 selctg, char sense,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* sdim,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vsl, lapack_int ldvsl,
                           lapack_complex_double* vsr, lapack_int ldvsr,
                           double* rconde, double* rcondv );
```

**Variants:** `LAPACKE_sggesx`, `LAPACKE_dggesx`, `LAPACKE_cggesx`, `LAPACKE_zggesx`

**Parameters:** Same as [gges](#gges) plus:

| Name | Type | Description |
|------|------|-------------|
| `sense` | `char` | `'N'` = none; `'E'` = eigenvalue condition; `'V'` = deflating subspace condition; `'B'` = both |
| `rconde` | `double*` | **[out]** Reciprocal condition numbers for eigenvalues, dimension 2 |
| `rcondv` | `double*` | **[out]** Reciprocal condition numbers for deflating subspaces, dimension 2 |

**Returns:** `info` -- 0 on success; `> 0` if QZ iteration failed to converge.

---

## Sylvester Equation

Solves the Sylvester matrix equation `op(A)*X + isgn*X*op(B) = scale*C` where `op(A) = A`, `A^T`, or `A^H`, and `A` and `B` are quasi-triangular or triangular.

### trsyl

Solves the real or complex Sylvester matrix equation.

```c
lapack_int LAPACKE_strsyl( int matrix_layout, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const float* a, lapack_int lda, const float* b,
                           lapack_int ldb, float* c, lapack_int ldc,
                           float* scale );
lapack_int LAPACKE_dtrsyl( int matrix_layout, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda, const double* b,
                           lapack_int ldb, double* c, lapack_int ldc,
                           double* scale );
lapack_int LAPACKE_ctrsyl( int matrix_layout, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* c, lapack_int ldc,
                           float* scale );
lapack_int LAPACKE_ztrsyl( int matrix_layout, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* c, lapack_int ldc,
                           double* scale );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trana` | `char` | `'N'` = no transpose; `'T'` = transpose; `'C'` = conjugate transpose for op(A) |
| `tranb` | `char` | `'N'`, `'T'`, or `'C'` for op(B) |
| `isgn` | `lapack_int` | +1 or -1; sign in the equation |
| `m` | `lapack_int` | Order of matrix A and rows of C |
| `n` | `lapack_int` | Order of matrix B and columns of C |
| `a` | `const double*` | **[in]** m-by-m upper quasi-triangular (real Schur form) matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `const double*` | **[in]** n-by-n upper quasi-triangular matrix |
| `ldb` | `lapack_int` | Leading dimension of b |
| `c` | `double*` | **[in/out]** m-by-n right-hand side / solution matrix |
| `ldc` | `lapack_int` | Leading dimension of c |
| `scale` | `double*` | **[out]** Scale factor (0 < scale <= 1) to avoid overflow |

**Returns:** `info` -- 0 on success; 1 if A and B have common or very close eigenvalues (perturbed solution returned).

---

### trsyl3

Solves the Sylvester matrix equation using a Level 3 BLAS algorithm for improved performance.

```c
lapack_int LAPACKE_strsyl3( int matrix_layout, char trana, char tranb,
                            lapack_int isgn, lapack_int m, lapack_int n,
                            const float* a, lapack_int lda, const float* b,
                            lapack_int ldb, float* c, lapack_int ldc,
                            float* scale );
lapack_int LAPACKE_dtrsyl3( int matrix_layout, char trana, char tranb,
                            lapack_int isgn, lapack_int m, lapack_int n,
                            const double* a, lapack_int lda, const double* b,
                            lapack_int ldb, double* c, lapack_int ldc,
                            double* scale );
lapack_int LAPACKE_ztrsyl3( int matrix_layout, char trana, char tranb,
                            lapack_int isgn, lapack_int m, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            const lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* c, lapack_int ldc,
                            double* scale );
```

**Parameters:** Same as [trsyl](#trsyl).

**Returns:** `info` -- 0 on success; 1 if A and B have common or very close eigenvalues.

---

## Reduction to Standard Form

These routines reduce matrices to condensed forms (tridiagonal, bidiagonal, Hessenberg) as preprocessing steps for eigenvalue computation.

### sytrd

Reduces a real symmetric matrix to real symmetric tridiagonal form `A = Q * T * Q^T` using orthogonal similarity transformations.

```c
lapack_int LAPACKE_ssytrd( int matrix_layout, char uplo, lapack_int n, float* a,
                           lapack_int lda, float* d, float* e, float* tau );
lapack_int LAPACKE_dsytrd( int matrix_layout, char uplo, lapack_int n, double* a,
                           lapack_int lda, double* d, double* e, double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle stored; `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `T*` | **[in/out]** Symmetric matrix; overwritten with tridiagonal form and Householder vectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `d` | `T*` | **[out]** Diagonal elements of tridiagonal T, dimension n |
| `e` | `T*` | **[out]** Off-diagonal elements of T, dimension n-1 |
| `tau` | `T*` | **[out]** Scalar factors of Householder reflectors, dimension n-1 |

**Returns:** `info` -- 0 on success.

---

### hetrd

Reduces a complex Hermitian matrix to real symmetric tridiagonal form `A = Q * T * Q^H` using unitary similarity transformations.

```c
lapack_int LAPACKE_chetrd( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* d,
                           float* e, lapack_complex_float* tau );
lapack_int LAPACKE_zhetrd( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda, double* d,
                           double* e, lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `complex T*` | **[in/out]** Hermitian matrix; overwritten with tridiagonal form and Householder vectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `d` | `real T*` | **[out]** Diagonal elements of tridiagonal T, dimension n |
| `e` | `real T*` | **[out]** Off-diagonal elements of T, dimension n-1 |
| `tau` | `complex T*` | **[out]** Scalar factors of Householder reflectors, dimension n-1 |

**Returns:** `info` -- 0 on success.

---

### sygst

Reduces a real symmetric-definite generalized eigenproblem to standard form. If itype=1, computes `inv(L)*A*inv(L^T)` or `inv(U^T)*A*inv(U)`.

```c
lapack_int LAPACKE_ssygst( int matrix_layout, lapack_int itype, char uplo,
                           lapack_int n, float* a, lapack_int lda,
                           const float* b, lapack_int ldb );
lapack_int LAPACKE_dsygst( int matrix_layout, lapack_int itype, char uplo,
                           lapack_int n, double* a, lapack_int lda,
                           const double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `itype` | `lapack_int` | 1: `inv(L)*A*inv(L^T)` or `inv(U^T)*A*inv(U)`; 2 or 3: `L^T*A*L` or `U*A*U^T` |
| `uplo` | `char` | `'U'` or `'L'` (must match the Cholesky factor of B) |
| `n` | `lapack_int` | Order of matrices A and B |
| `a` | `T*` | **[in/out]** Symmetric matrix A; overwritten with transformed matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `b` | `const T*` | **[in]** Cholesky factor of B from potrf |
| `ldb` | `lapack_int` | Leading dimension of b |

**Returns:** `info` -- 0 on success.

---

### hegst

Reduces a complex Hermitian-definite generalized eigenproblem to standard form.

```c
lapack_int LAPACKE_chegst( int matrix_layout, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zhegst( int matrix_layout, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:** Same as [sygst](#sygst) with complex types.

**Returns:** `info` -- 0 on success.

---

### gebrd

Reduces a general m-by-n matrix to upper or lower bidiagonal form `A = Q * B * P^T` using Householder transformations. Preprocessing for SVD.

```c
lapack_int LAPACKE_sgebrd( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* d, float* e,
                           float* tauq, float* taup );
lapack_int LAPACKE_dgebrd( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* d, double* e,
                           double* tauq, double* taup );
lapack_int LAPACKE_cgebrd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* d,
                           float* e, lapack_complex_float* tauq,
                           lapack_complex_float* taup );
lapack_int LAPACKE_zgebrd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda, double* d,
                           double* e, lapack_complex_double* tauq,
                           lapack_complex_double* taup );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `double*` | **[in/out]** m-by-n matrix; overwritten with bidiagonal form and Householder vectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `d` | `double*` | **[out]** Diagonal of bidiagonal B, dimension min(m,n) |
| `e` | `double*` | **[out]** Off-diagonal of bidiagonal B, dimension min(m,n)-1 |
| `tauq` | `double*` | **[out]** Scalar factors of left Householder reflectors, dimension min(m,n) |
| `taup` | `double*` | **[out]** Scalar factors of right Householder reflectors, dimension min(m,n) |

**Returns:** `info` -- 0 on success.

---

### gebal

Balances a general matrix A by similarity transformations to make rows and columns as close in norm as possible. Preprocessing for eigenvalue computation to improve accuracy.

```c
lapack_int LAPACKE_sgebal( int matrix_layout, char job, lapack_int n, float* a,
                           lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                           float* scale );
lapack_int LAPACKE_dgebal( int matrix_layout, char job, lapack_int n, double* a,
                           lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                           double* scale );
lapack_int LAPACKE_cgebal( int matrix_layout, char job, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ilo, lapack_int* ihi, float* scale );
lapack_int LAPACKE_zgebal( int matrix_layout, char job, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ilo, lapack_int* ihi, double* scale );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `job` | `char` | `'N'` = none; `'P'` = permute only; `'S'` = scale only; `'B'` = both |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `double*` | **[in/out]** n-by-n matrix; overwritten with balanced matrix |
| `lda` | `lapack_int` | Leading dimension of a |
| `ilo`, `ihi` | `lapack_int*` | **[out]** Indices such that A(i,j)=0 if i>j and j=1,...,ilo-1 or i=ihi+1,...,n |
| `scale` | `double*` | **[out]** Permutation/scaling details, dimension n |

**Returns:** `info` -- 0 on success.

---

### gehrd

Reduces a general matrix to upper Hessenberg form `A = Q * H * Q^T` using orthogonal/unitary similarity transformations. Preprocessing for Schur decomposition.

```c
lapack_int LAPACKE_sgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           float* tau );
lapack_int LAPACKE_dgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           double* tau );
lapack_int LAPACKE_cgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* tau );
lapack_int LAPACKE_zgehrd( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* tau );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A |
| `ilo`, `ihi` | `lapack_int` | Indices from gebal (if balanced) or 1 and n (if not balanced) |
| `a` | `double*` | **[in/out]** n-by-n matrix; overwritten with Hessenberg form and Householder vectors |
| `lda` | `lapack_int` | Leading dimension of a |
| `tau` | `double*` | **[out]** Scalar factors of Householder reflectors, dimension n-1 |

**Returns:** `info` -- 0 on success.

---

## Tridiagonal and Bidiagonal Eigensolvers

Direct eigensolvers for tridiagonal and related matrices.

### sterf

Computes all eigenvalues of a real symmetric tridiagonal matrix using a root-free variant of the QR algorithm. Eigenvalues only (no eigenvectors). No matrix_layout parameter.

```c
lapack_int LAPACKE_ssterf( lapack_int n, float* d, float* e );
lapack_int LAPACKE_dsterf( lapack_int n, double* d, double* e );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of the tridiagonal matrix |
| `d` | `T*` | **[in/out]** Diagonal elements, dimension n; overwritten with eigenvalues in ascending order |
| `e` | `T*` | **[in/out]** Off-diagonal elements, dimension n-1; destroyed on exit |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### steqr

Computes all eigenvalues and, optionally, eigenvectors of a symmetric tridiagonal matrix using the implicit QL or QR method.

```c
lapack_int LAPACKE_ssteqr( int matrix_layout, char compz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dsteqr( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_csteqr( int matrix_layout, char compz, lapack_int n, float* d,
                           float* e, lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zsteqr( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, lapack_complex_double* z,
                           lapack_int ldz );
```

**Parameters (double real):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `compz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors of T; `'I'` = eigenvalues and eigenvectors (z initialized to identity) |
| `n` | `lapack_int` | Order of the tridiagonal matrix |
| `d` | `double*` | **[in/out]** Diagonal elements; overwritten with eigenvalues |
| `e` | `double*` | **[in/out]** Off-diagonal elements; destroyed on exit |
| `z` | `double*` | **[in/out]** If compz='V', input orthogonal matrix from sytrd; overwritten with eigenvectors. If compz='I', initialized internally to identity |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### stev

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix (given by diagonal and off-diagonal arrays).

```c
lapack_int LAPACKE_sstev( int matrix_layout, char jobz, lapack_int n, float* d,
                          float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dstev( int matrix_layout, char jobz, lapack_int n, double* d,
                          double* e, double* z, lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `n` | `lapack_int` | Order of the tridiagonal matrix |
| `d` | `T*` | **[in/out]** Diagonal elements, dimension n; overwritten with eigenvalues in ascending order |
| `e` | `T*` | **[in/out]** Off-diagonal elements, dimension n-1; destroyed on exit |
| `z` | `T*` | **[out]** Eigenvectors (n-by-n if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### stevd

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix using divide-and-conquer.

```c
lapack_int LAPACKE_sstevd( int matrix_layout, char jobz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dstevd( int matrix_layout, char jobz, lapack_int n, double* d,
                           double* e, double* z, lapack_int ldz );
```

**Parameters:** Same as [stev](#stev).

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.

---

### stevr

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix using MRRR.

```c
lapack_int LAPACKE_sstevr( int matrix_layout, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_dstevr( int matrix_layout, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* isuppz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvalues and eigenvectors |
| `range` | `char` | `'A'` = all; `'V'` = in (vl,vu]; `'I'` = il-th through iu-th |
| `n` | `lapack_int` | Order of the tridiagonal matrix |
| `d` | `T*` | **[in/out]** Diagonal elements; may be destroyed on exit |
| `e` | `T*` | **[in/out]** Off-diagonal elements; destroyed on exit |
| `vl`, `vu` | `T` | Interval bounds (range='V') |
| `il`, `iu` | `lapack_int` | Index range (range='I', 1-based) |
| `abstol` | `T` | Absolute error tolerance |
| `m` | `lapack_int*` | **[out]** Number of eigenvalues found |
| `w` | `T*` | **[out]** Eigenvalues in ascending order |
| `z` | `T*` | **[out]** Eigenvectors (n-by-m if jobz='V') |
| `ldz` | `lapack_int` | Leading dimension of z |
| `isuppz` | `lapack_int*` | **[out]** Support of eigenvectors |

**Returns:** `info` -- 0 on success; `> 0` if internal error occurred.

---

### stevx

Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric tridiagonal matrix using bisection and inverse iteration.

```c
lapack_int LAPACKE_sstevx( int matrix_layout, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dstevx( int matrix_layout, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );
```

**Parameters:** Same as [stevr](#stevr) except `isuppz` is replaced by `ifail` (`lapack_int*`) -- **[out]** indices of eigenvectors that failed to converge.

**Returns:** `info` -- 0 on success; `> 0` means `info` eigenvectors failed to converge.

---

### pteqr

Computes all eigenvalues and, optionally, eigenvectors of a real symmetric positive definite tridiagonal matrix by computing the SVD of the bidiagonal form. This is more accurate than steqr when the matrix is positive definite.

```c
lapack_int LAPACKE_spteqr( int matrix_layout, char compz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dpteqr( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_cpteqr( int matrix_layout, char compz, lapack_int n, float* d,
                           float* e, lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zpteqr( int matrix_layout, char compz, lapack_int n,
                           double* d, double* e, lapack_complex_double* z,
                           lapack_int ldz );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `compz` | `char` | `'N'` = eigenvalues only; `'V'` = eigenvectors of original matrix (z input); `'I'` = eigenvectors of tridiagonal (z initialized to identity) |
| `n` | `lapack_int` | Order of the tridiagonal matrix |
| `d` | `real T*` | **[in/out]** Diagonal elements (must be positive); overwritten with eigenvalues in descending order |
| `e` | `real T*` | **[in/out]** Off-diagonal elements; destroyed on exit |
| `z` | `T*` | **[in/out]** Eigenvector matrix; see compz |
| `ldz` | `lapack_int` | Leading dimension of z |

**Returns:** `info` -- 0 on success; `> 0` if the algorithm failed to converge.