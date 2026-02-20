# LAPACKE SVD API Reference

> Singular Value Decomposition routines: A = U * S * V^T
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [Precision Prefixes and Common Parameters](#precision-prefixes-and-common-parameters)
- [SVD Drivers](#svd-drivers)
  - [gesdd - Divide and Conquer SVD](#gesdd---divide-and-conquer-svd)
  - [gesvd - Standard SVD](#gesvd---standard-svd)
  - [gesvdx - SVD with Selected Singular Values](#gesvdx---svd-with-selected-singular-values)
  - [gesvdq - SVD with Preconditioning](#gesvdq---svd-with-preconditioning)
  - [gesvj - Jacobi SVD](#gesvj---jacobi-svd)
  - [gejsv - Preconditioned Jacobi SVD](#gejsv---preconditioned-jacobi-svd)
- [Generalized SVD](#generalized-svd)
  - [ggsvd3 - Generalized SVD (version 3)](#ggsvd3---generalized-svd-version-3)
  - [ggsvd - Generalized SVD (deprecated)](#ggsvd---generalized-svd-deprecated)
  - [ggsvp3 - GSVD Preprocessing (version 3)](#ggsvp3---gsvd-preprocessing-version-3)
  - [ggsvp - GSVD Preprocessing (deprecated)](#ggsvp---gsvd-preprocessing-deprecated)
- [Bidiagonal SVD (Computational)](#bidiagonal-svd-computational)
  - [bdsqr - Bidiagonal SVD via QR Iteration](#bdsqr---bidiagonal-svd-via-qr-iteration)
  - [bdsdc - Bidiagonal SVD via Divide and Conquer](#bdsdc---bidiagonal-svd-via-divide-and-conquer)
  - [bdsvdx - Bidiagonal SVD with Selected Values](#bdsvdx---bidiagonal-svd-with-selected-values)
- [Bidiagonal Reduction](#bidiagonal-reduction)
  - [gebrd - General to Bidiagonal Reduction](#gebrd---general-to-bidiagonal-reduction)
  - [gbbrd - Band to Bidiagonal Reduction](#gbbrd---band-to-bidiagonal-reduction)
- [Generate/Apply Orthogonal Matrices from Bidiagonal Reduction](#generateapply-orthogonal-matrices-from-bidiagonal-reduction)
  - [orgbr / ungbr - Generate Q or P from gebrd](#orgbr--ungbr---generate-q-or-p-from-gebrd)
  - [ormbr / unmbr - Multiply by Q or P from gebrd](#ormbr--unmbr---multiply-by-q-or-p-from-gebrd)
- [CS Decomposition](#cs-decomposition)
  - [bbcsd - CS Decomposition of Unitary Bidiagonal](#bbcsd---cs-decomposition-of-unitary-bidiagonal)
  - [orbdb / unbdb - Simultaneous Bidiagonalization](#orbdb--unbdb---simultaneous-bidiagonalization)
  - [orcsd / uncsd - CS Decomposition](#orcsd--uncsd---cs-decomposition)
  - [orcsd2by1 / uncsd2by1 - 2-by-1 CS Decomposition](#orcsd2by1--uncsd2by1---2-by-1-cs-decomposition)

## Precision Prefixes and Common Parameters

### Precision Prefixes

| Prefix | Precision |
|--------|-----------|
| `s` | Single-precision real (`float`) |
| `d` | Double-precision real (`double`) |
| `c` | Single-precision complex (`lapack_complex_float`) |
| `z` | Double-precision complex (`lapack_complex_double`) |

### Common Parameters

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` (101) or `LAPACK_COL_MAJOR` (102) |
| `m` | `lapack_int` | Number of rows of the matrix |
| `n` | `lapack_int` | Number of columns of the matrix |
| `a` | `T*` | Input matrix, overwritten on output |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `s` | `float*` or `double*` | Singular values in decreasing order, dimension min(m,n) |
| `u` | `T*` | Left singular vectors (matrix U) |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `vt` | `T*` | Right singular vectors (matrix V^T) |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |

### jobu / jobvt Parameter Values (gesvd, gesvdx)

| Value | Meaning |
|-------|---------|
| `'A'` | Compute **all** columns of U (or rows of V^T). U is m-by-m (or V^T is n-by-n) |
| `'S'` | Compute first min(m,n) columns of U (or rows of V^T) -- **economy** SVD |
| `'O'` | **Overwrite** A with the first min(m,n) columns of U (or rows of V^T) |
| `'N'` | **No** singular vectors computed |

### jobz Parameter Values (gesdd)

| Value | Meaning |
|-------|---------|
| `'A'` | Compute all columns of U and all rows of V^T |
| `'S'` | Compute first min(m,n) columns of U and rows of V^T (economy) |
| `'O'` | Overwrite A: if m >= n, A contains first n columns of U and vt contains V^T; if m < n, A contains V^T and u contains U |
| `'N'` | No singular vectors computed |

### Performance Guidance

| Routine | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| `gesdd` | Fastest | Standard | General use, large matrices |
| `gesvd` | Fast | Standard | When gesdd workspace is too large |
| `gesvdx` | Moderate | Standard | Subset of singular values/vectors |
| `gesvdq` | Moderate | High | High accuracy with preconditioning |
| `gesvj` | Slower | High | High accuracy via Jacobi iterations |
| `gejsv` | Slowest | Highest | Maximum accuracy, preconditioned Jacobi |

**Return value:** All functions return `lapack_int`. 0 = success, negative = illegal argument at position |value|, positive = convergence failure.

---

## SVD Drivers

### gesdd - Divide and Conquer SVD

Compute the SVD of a general m-by-n matrix using divide and conquer. Faster than `gesvd` for large matrices, especially when singular vectors are requested.

```c
lapack_int LAPACKE_sgesdd( int matrix_layout, char jobz, lapack_int m,
                           lapack_int n, float* a, lapack_int lda, float* s,
                           float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_dgesdd( int matrix_layout, char jobz, lapack_int m,
                           lapack_int n, double* a, lapack_int lda, double* s,
                           double* u, lapack_int ldu, double* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_cgesdd( int matrix_layout, char jobz, lapack_int m,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_zgesdd( int matrix_layout, char jobz, lapack_int m,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobz` | `char` | `'A'`, `'S'`, `'O'`, or `'N'` -- controls which singular vectors are computed |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; overwritten on output |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `s` | `real*` | [out] Singular values in decreasing order, dimension min(m,n) |
| `u` | `T*` | [out] Left singular vectors |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `vt` | `T*` | [out] Right singular vectors (transposed/conjugate-transposed) |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |

---

### gesvd - Standard SVD

Compute the SVD of a general m-by-n matrix using QR iteration. Allows independent control of left and right singular vectors via separate `jobu` and `jobvt` parameters.

```c
lapack_int LAPACKE_sgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* s, float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt, float* superb );
lapack_int LAPACKE_dgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt, double* superb );
lapack_int LAPACKE_cgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt, float* superb );
lapack_int LAPACKE_zgesvd( int matrix_layout, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt, double* superb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu` | `char` | `'A'`, `'S'`, `'O'`, or `'N'` -- controls left singular vectors |
| `jobvt` | `char` | `'A'`, `'S'`, `'O'`, or `'N'` -- controls right singular vectors |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; overwritten on output |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `s` | `real*` | [out] Singular values in decreasing order, dimension min(m,n) |
| `u` | `T*` | [out] Left singular vectors |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `vt` | `T*` | [out] Right singular vectors (transposed/conjugate-transposed) |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |
| `superb` | `real*` | [out] Unconverged superdiagonal elements of the upper bidiagonal matrix B if info > 0, dimension min(m,n)-1 |

**Note:** `jobu` and `jobvt` cannot both be `'O'`.

---

### gesvdx - SVD with Selected Singular Values

Compute selected singular values and optionally singular vectors of a general m-by-n matrix. Singular values/vectors can be selected by specifying a range of values or indices.

```c
lapack_int LAPACKE_sgesvdx( int matrix_layout, char jobu, char jobvt, char range,
                           lapack_int m, lapack_int n, float* a,
                           lapack_int lda, float vl, float vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           float* s, float* u, lapack_int ldu,
                           float* vt, lapack_int ldvt,
                           lapack_int* superb );
lapack_int LAPACKE_dgesvdx( int matrix_layout, char jobu, char jobvt, char range,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double vl, double vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt,
                           lapack_int* superb );
lapack_int LAPACKE_cgesvdx( int matrix_layout, char jobu, char jobvt, char range,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float vl, float vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           float* s, lapack_complex_float* u, lapack_int ldu,
                           lapack_complex_float* vt, lapack_int ldvt,
                           lapack_int* superb );
lapack_int LAPACKE_zgesvdx( int matrix_layout, char jobu, char jobvt, char range,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double vl, double vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           double* s, lapack_complex_double* u, lapack_int ldu,
                           lapack_complex_double* vt, lapack_int ldvt,
                           lapack_int* superb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu` | `char` | `'V'` = compute left singular vectors, `'N'` = do not |
| `jobvt` | `char` | `'V'` = compute right singular vectors, `'N'` = do not |
| `range` | `char` | `'A'` = all, `'V'` = in half-open interval (vl,vu], `'I'` = il-th through iu-th |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; contents destroyed |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `vl` | `real` | Lower bound of interval for singular values (if range='V') |
| `vu` | `real` | Upper bound of interval for singular values (if range='V') |
| `il` | `lapack_int` | Index of smallest singular value to return (if range='I', 1-based) |
| `iu` | `lapack_int` | Index of largest singular value to return (if range='I', 1-based) |
| `ns` | `lapack_int*` | [out] Number of singular values found |
| `s` | `real*` | [out] Selected singular values in decreasing order |
| `u` | `T*` | [out] Left singular vectors (m-by-ns) |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `vt` | `T*` | [out] Right singular vectors (ns-by-n) |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |
| `superb` | `lapack_int*` | [out] If info > 0, indices of unconverged singular values |

---

### gesvdq - SVD with Preconditioning

Compute the SVD of a general m-by-n matrix with a QR-preconditioned approach for high accuracy. Provides better accuracy than `gesvd`/`gesdd` for ill-conditioned matrices, with numerical rank estimation.

```c
lapack_int LAPACKE_sgesvdq( int matrix_layout, char joba, char jobp, char jobr, char jobu, char jobv,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* s, float* u, lapack_int ldu, float* v,
                           lapack_int ldv, lapack_int* numrank );
lapack_int LAPACKE_dgesvdq( int matrix_layout, char joba, char jobp, char jobr, char jobu, char jobv,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* v, lapack_int ldv, lapack_int* numrank);
lapack_int LAPACKE_cgesvdq( int matrix_layout, char joba, char jobp, char jobr, char jobu, char jobv,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* v,
                           lapack_int ldv, lapack_int* numrank );
lapack_int LAPACKE_zgesvdq( int matrix_layout, char joba, char jobp, char jobr, char jobu, char jobv,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* v,
                           lapack_int ldv, lapack_int* numrank );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `joba` | `char` | Accuracy level: `'A'` = full accuracy, `'H'` = high accuracy, `'M'` = medium, `'E'` = economy |
| `jobp` | `char` | `'P'` = use rank-revealing QR with column pivoting, `'N'` = no pivoting |
| `jobr` | `char` | `'T'` = truncate to numerical rank (if determined), `'N'` = no truncation |
| `jobu` | `char` | `'A'` = all m columns of U, `'S'`/`'U'` = first min(m,n) columns, `'R'` = numrank columns, `'F'` = full row-rank representation, `'N'` = none |
| `jobv` | `char` | `'A'` = all n columns of V, `'V'` = first min(m,n) columns, `'R'` = numrank columns, `'N'` = none |
| `m` | `lapack_int` | Number of rows of A (m >= 0) |
| `n` | `lapack_int` | Number of columns of A (n >= 0) |
| `a` | `T*` | [in/out] m-by-n matrix; overwritten on output |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `s` | `real*` | [out] Singular values in decreasing order |
| `u` | `T*` | [out] Left singular vectors |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `v` | `T*` | [out] Right singular vectors (not transposed, unlike gesvd) |
| `ldv` | `lapack_int` | Leading dimension of `v` |
| `numrank` | `lapack_int*` | [out] Estimated numerical rank |

**Note:** This routine returns `v` (not `vt`), so V is stored directly, not as its transpose.

---

### gesvj - Jacobi SVD

Compute the SVD of a general m-by-n matrix using one-sided Jacobi rotations. Provides higher accuracy than QR-based methods (`gesvd`/`gesdd`) at the cost of slower execution.

```c
lapack_int LAPACKE_sgesvj( int matrix_layout, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* sva, lapack_int mv, float* v, lapack_int ldv,
                           float* stat );
lapack_int LAPACKE_dgesvj( int matrix_layout, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* sva, lapack_int mv,
                           double* v, lapack_int ldv, double* stat );
lapack_int LAPACKE_cgesvj( int matrix_layout, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* sva, lapack_int mv,
                           lapack_complex_float* v, lapack_int ldv, float* stat );
lapack_int LAPACKE_zgesvj( int matrix_layout, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* sva, lapack_int mv,
                           lapack_complex_double* v, lapack_int ldv, double* stat );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `joba` | `char` | Structure: `'L'` = lower triangular, `'U'` = upper triangular, `'G'` = general, `'F'` = first call with general, `'C'` = continuation |
| `jobu` | `char` | `'U'` = compute U (in `a`), `'C'` = compute U (accumulated into `a`), `'N'` = no U |
| `jobv` | `char` | `'V'` = compute V, `'A'` = accumulate V (multiply input v by computed V), `'N'` = no V |
| `m` | `lapack_int` | Number of rows of A (m >= n) |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; on exit may contain left singular vectors |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `sva` | `real*` | [out] Singular values, dimension n |
| `mv` | `lapack_int` | Number of rows of V (if jobv='A', leading dimension of initial V) |
| `v` | `T*` | [out] Right singular vectors, n-by-n |
| `ldv` | `lapack_int` | Leading dimension of `v` |
| `stat` | `real*` | [out] Workspace for statistics, dimension >= 6 |

**`stat` output:** `stat[0]` = scale factor for singular values, `stat[1]` = number of sweeps, `stat[2]` = number of Jacobi pairs, etc.

---

### gejsv - Preconditioned Jacobi SVD

Compute the SVD of a general m-by-n matrix using preconditioned Jacobi rotations for the highest possible accuracy. The most accurate SVD driver in LAPACK.

```c
lapack_int LAPACKE_sgejsv( int matrix_layout, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, float* a, lapack_int lda, float* sva,
                           float* u, lapack_int ldu, float* v, lapack_int ldv,
                           float* stat, lapack_int* istat );
lapack_int LAPACKE_dgejsv( int matrix_layout, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, double* a, lapack_int lda, double* sva,
                           double* u, lapack_int ldu, double* v, lapack_int ldv,
                           double* stat, lapack_int* istat );
lapack_int LAPACKE_cgejsv( int matrix_layout, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, lapack_complex_float* a, lapack_int lda, float* sva,
                           lapack_complex_float* u, lapack_int ldu, lapack_complex_float* v, lapack_int ldv,
                           float* stat, lapack_int* istat );
lapack_int LAPACKE_zgejsv( int matrix_layout, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, lapack_complex_double* a, lapack_int lda, double* sva,
                           lapack_complex_double* u, lapack_int ldu, lapack_complex_double* v, lapack_int ldv,
                           double* stat, lapack_int* istat );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `joba` | `char` | Accuracy/condition: `'C'` = cautious (condition estimated), `'E'` = economy (condition estimated), `'F'` = full SVD, `'G'`/`'A'`/`'R'` = variants |
| `jobu` | `char` | `'U'` = full U, `'F'` = full U in `a`, `'W'` = U in `a` (partial), `'N'` = no U |
| `jobv` | `char` | `'V'` = compute V, `'J'` = V computed via Jacobi, `'W'` = partial, `'N'` = no V |
| `jobr` | `char` | `'N'` = no rank determination, `'R'` = restrict range (numerical rank) |
| `jobt` | `char` | `'T'` = transpose if beneficial, `'N'` = no transpose |
| `jobp` | `char` | `'P'` = perturbation for rank deficiency, `'N'` = no perturbation |
| `m` | `lapack_int` | Number of rows of A (m >= n) |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; overwritten on output |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `sva` | `real*` | [out] Singular values, dimension n |
| `u` | `T*` | [out] Left singular vectors |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `v` | `T*` | [out] Right singular vectors |
| `ldv` | `lapack_int` | Leading dimension of `v` |
| `stat` | `real*` | [out] Workspace for statistics, dimension >= 7 |
| `istat` | `lapack_int*` | [out] Integer statistics, dimension >= 3 |

---

## Generalized SVD

The generalized SVD (GSVD) of an m-by-n matrix A and p-by-n matrix B produces:
- `A = U * D1 * [0 R] * Q^T`
- `B = V * D2 * [0 R] * Q^T`

where U, V, Q are orthogonal/unitary, R is upper triangular, and D1, D2 are diagonal with `D1^T*D1 + D2^T*D2 = I`.

### ggsvd3 - Generalized SVD (version 3)

Compute the GSVD of an m-by-n matrix A and p-by-n matrix B. Preferred over deprecated `ggsvd`.

```c
lapack_int LAPACKE_sggsvd3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int n, lapack_int p,
                            lapack_int* k, lapack_int* l, float* a,
                            lapack_int lda, float* b, lapack_int ldb,
                            float* alpha, float* beta, float* u, lapack_int ldu,
                            float* v, lapack_int ldv, float* q, lapack_int ldq,
                            lapack_int* iwork );
lapack_int LAPACKE_dggsvd3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int n, lapack_int p,
                            lapack_int* k, lapack_int* l, double* a,
                            lapack_int lda, double* b, lapack_int ldb,
                            double* alpha, double* beta, double* u,
                            lapack_int ldu, double* v, lapack_int ldv, double* q,
                            lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_cggsvd3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int n, lapack_int p,
                            lapack_int* k, lapack_int* l,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb,
                            float* alpha, float* beta, lapack_complex_float* u,
                            lapack_int ldu, lapack_complex_float* v,
                            lapack_int ldv, lapack_complex_float* q,
                            lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_zggsvd3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int n, lapack_int p,
                            lapack_int* k, lapack_int* l,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb,
                            double* alpha, double* beta,
                            lapack_complex_double* u, lapack_int ldu,
                            lapack_complex_double* v, lapack_int ldv,
                            lapack_complex_double* q, lapack_int ldq,
                            lapack_int* iwork );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu` | `char` | `'U'` = compute orthogonal matrix U, `'N'` = do not |
| `jobv` | `char` | `'V'` = compute orthogonal matrix V, `'N'` = do not |
| `jobq` | `char` | `'Q'` = compute orthogonal matrix Q, `'N'` = do not |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A and B |
| `p` | `lapack_int` | Number of rows of B |
| `k` | `lapack_int*` | [out] Dimension k of the GSVD (see LAPACK docs) |
| `l` | `lapack_int*` | [out] Dimension l of the GSVD; effective rank of [A; B] = k + l |
| `a` | `T*` | [in/out] m-by-n matrix A; on exit contains triangular form |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `b` | `T*` | [in/out] p-by-n matrix B; on exit contains triangular form |
| `ldb` | `lapack_int` | Leading dimension of `b` |
| `alpha` | `real*` | [out] Generalized singular value pairs (alpha), dimension n |
| `beta` | `real*` | [out] Generalized singular value pairs (beta), dimension n |
| `u` | `T*` | [out] m-by-m orthogonal matrix U |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `v` | `T*` | [out] p-by-p orthogonal matrix V |
| `ldv` | `lapack_int` | Leading dimension of `v` |
| `q` | `T*` | [out] n-by-n orthogonal matrix Q |
| `ldq` | `lapack_int` | Leading dimension of `q` |
| `iwork` | `lapack_int*` | [out] Sorting information, dimension n |

**Generalized singular values:** `sigma_i = alpha[i] / beta[i]` for `i = 0, ..., k+l-1`.

---

### ggsvd - Generalized SVD (deprecated)

Deprecated. Use `ggsvd3` instead. Same interface as `ggsvd3`.

```c
lapack_int LAPACKE_sggsvd( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l, float* a,
                           lapack_int lda, float* b, lapack_int ldb,
                           float* alpha, float* beta, float* u, lapack_int ldu,
                           float* v, lapack_int ldv, float* q, lapack_int ldq,
                           lapack_int* iwork );
lapack_int LAPACKE_dggsvd( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double* alpha, double* beta, double* u,
                           lapack_int ldu, double* v, lapack_int ldv, double* q,
                           lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_cggsvd( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           float* alpha, float* beta, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* v,
                           lapack_int ldv, lapack_complex_float* q,
                           lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_zggsvd( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           double* alpha, double* beta,
                           lapack_complex_double* u, lapack_int ldu,
                           lapack_complex_double* v, lapack_int ldv,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_int* iwork );
```

---

### ggsvp3 - GSVD Preprocessing (version 3)

Compute the preprocessing for the GSVD: orthogonal transformations that reduce A and B to triangular form for input to `tgsja`. Preferred over deprecated `ggsvp`.

```c
lapack_int LAPACKE_sggsvp3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int p, lapack_int n, float* a,
                            lapack_int lda, float* b, lapack_int ldb, float tola,
                            float tolb, lapack_int* k, lapack_int* l, float* u,
                            lapack_int ldu, float* v, lapack_int ldv, float* q,
                            lapack_int ldq );
lapack_int LAPACKE_dggsvp3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int p, lapack_int n, double* a,
                            lapack_int lda, double* b, lapack_int ldb,
                            double tola, double tolb, lapack_int* k,
                            lapack_int* l, double* u, lapack_int ldu, double* v,
                            lapack_int ldv, double* q, lapack_int ldq );
lapack_int LAPACKE_cggsvp3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int p, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb, float tola,
                            float tolb, lapack_int* k, lapack_int* l,
                            lapack_complex_float* u, lapack_int ldu,
                            lapack_complex_float* v, lapack_int ldv,
                            lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zggsvp3( int matrix_layout, char jobu, char jobv, char jobq,
                            lapack_int m, lapack_int p, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb,
                            double tola, double tolb, lapack_int* k,
                            lapack_int* l, lapack_complex_double* u,
                            lapack_int ldu, lapack_complex_double* v,
                            lapack_int ldv, lapack_complex_double* q,
                            lapack_int ldq );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu` | `char` | `'U'` = compute U, `'N'` = do not |
| `jobv` | `char` | `'V'` = compute V, `'N'` = do not |
| `jobq` | `char` | `'Q'` = compute Q, `'N'` = do not |
| `m` | `lapack_int` | Number of rows of A |
| `p` | `lapack_int` | Number of rows of B |
| `n` | `lapack_int` | Number of columns of A and B |
| `a` | `T*` | [in/out] m-by-n matrix A |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `b` | `T*` | [in/out] p-by-n matrix B |
| `ldb` | `lapack_int` | Leading dimension of `b` |
| `tola` | `real` | Tolerance for A; entries < tola*||A|| treated as zero |
| `tolb` | `real` | Tolerance for B; entries < tolb*||B|| treated as zero |
| `k` | `lapack_int*` | [out] Dimension k of the GSVD |
| `l` | `lapack_int*` | [out] Dimension l of the GSVD |
| `u` | `T*` | [out] Orthogonal matrix U (m-by-m) |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `v` | `T*` | [out] Orthogonal matrix V (p-by-p) |
| `ldv` | `lapack_int` | Leading dimension of `v` |
| `q` | `T*` | [out] Orthogonal matrix Q (n-by-n) |
| `ldq` | `lapack_int` | Leading dimension of `q` |

---

### ggsvp - GSVD Preprocessing (deprecated)

Deprecated. Use `ggsvp3` instead. Same interface as `ggsvp3`.

```c
lapack_int LAPACKE_sggsvp( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float tola,
                           float tolb, lapack_int* k, lapack_int* l, float* u,
                           lapack_int ldu, float* v, lapack_int ldv, float* q,
                           lapack_int ldq );
lapack_int LAPACKE_dggsvp( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double tola, double tolb, lapack_int* k,
                           lapack_int* l, double* u, lapack_int ldu, double* v,
                           lapack_int ldv, double* q, lapack_int ldq );
lapack_int LAPACKE_cggsvp( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb, float tola,
                           float tolb, lapack_int* k, lapack_int* l,
                           lapack_complex_float* u, lapack_int ldu,
                           lapack_complex_float* v, lapack_int ldv,
                           lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zggsvp( int matrix_layout, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           double tola, double tolb, lapack_int* k,
                           lapack_int* l, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* v,
                           lapack_int ldv, lapack_complex_double* q,
                           lapack_int ldq );
```

---

## Bidiagonal SVD (Computational)

These routines compute the SVD of a bidiagonal matrix. They are the computational core used by the driver routines after reducing the input matrix to bidiagonal form.

### bdsqr - Bidiagonal SVD via QR Iteration

Compute the SVD of a real upper or lower bidiagonal matrix using the implicit zero-shift QR algorithm. Optionally applies the transformations to given matrices.

```c
lapack_int LAPACKE_sbdsqr( int matrix_layout, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           float* d, float* e, float* vt, lapack_int ldvt,
                           float* u, lapack_int ldu, float* c, lapack_int ldc );
lapack_int LAPACKE_dbdsqr( int matrix_layout, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           double* d, double* e, double* vt, lapack_int ldvt,
                           double* u, lapack_int ldu, double* c,
                           lapack_int ldc );
lapack_int LAPACKE_cbdsqr( int matrix_layout, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           float* d, float* e, lapack_complex_float* vt,
                           lapack_int ldvt, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* c,
                           lapack_int ldc );
lapack_int LAPACKE_zbdsqr( int matrix_layout, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           double* d, double* e, lapack_complex_double* vt,
                           lapack_int ldvt, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* c,
                           lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper bidiagonal, `'L'` = lower bidiagonal |
| `n` | `lapack_int` | Order of the bidiagonal matrix B |
| `ncvt` | `lapack_int` | Number of columns of matrix VT (0 if no right singular vectors) |
| `nru` | `lapack_int` | Number of rows of matrix U (0 if no left singular vectors) |
| `ncc` | `lapack_int` | Number of columns of matrix C (0 if not used) |
| `d` | `real*` | [in/out] Diagonal elements (n); on exit, singular values in decreasing order |
| `e` | `real*` | [in/out] Off-diagonal elements (n-1); on exit, zeroed |
| `vt` | `T*` | [in/out] n-by-ncvt matrix; on exit, updated with right rotations |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |
| `u` | `T*` | [in/out] nru-by-n matrix; on exit, updated with left rotations |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `c` | `T*` | [in/out] n-by-ncc matrix; on exit, updated with left rotations |
| `ldc` | `lapack_int` | Leading dimension of `c` |

---

### bdsdc - Bidiagonal SVD via Divide and Conquer

Compute the SVD of a real upper bidiagonal matrix using divide and conquer. Real precision only (no complex variants).

```c
lapack_int LAPACKE_sbdsdc( int matrix_layout, char uplo, char compq,
                           lapack_int n, float* d, float* e, float* u,
                           lapack_int ldu, float* vt, lapack_int ldvt, float* q,
                           lapack_int* iq );
lapack_int LAPACKE_dbdsdc( int matrix_layout, char uplo, char compq,
                           lapack_int n, double* d, double* e, double* u,
                           lapack_int ldu, double* vt, lapack_int ldvt,
                           double* q, lapack_int* iq );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper bidiagonal, `'L'` = lower bidiagonal |
| `compq` | `char` | `'N'` = singular values only, `'P'` = compact form, `'I'` = full U and VT |
| `n` | `lapack_int` | Order of the bidiagonal matrix B |
| `d` | `real*` | [in/out] Diagonal elements; on exit, singular values in decreasing order |
| `e` | `real*` | [in/out] Off-diagonal elements (n-1); destroyed on exit |
| `u` | `real*` | [out] n-by-n left singular vectors (if compq='I') |
| `ldu` | `lapack_int` | Leading dimension of `u` |
| `vt` | `real*` | [out] n-by-n right singular vectors (if compq='I') |
| `ldvt` | `lapack_int` | Leading dimension of `vt` |
| `q` | `real*` | [out] Compact singular vectors (if compq='P') |
| `iq` | `lapack_int*` | [out] Integer array for compact form (if compq='P') |

**Note:** Only `s` and `d` variants exist (no complex).

---

### bdsvdx - Bidiagonal SVD with Selected Values

Compute selected singular values and optionally singular vectors of a real upper bidiagonal matrix. Real precision only.

```c
lapack_int LAPACKE_sbdsvdx( int matrix_layout, char uplo, char jobz, char range,
                           lapack_int n, float* d, float* e,
                           float vl, float vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           float* s, float* z, lapack_int ldz,
                           lapack_int* superb );
lapack_int LAPACKE_dbdsvdx( int matrix_layout, char uplo, char jobz, char range,
                           lapack_int n, double* d, double* e,
                           double vl, double vu,
                           lapack_int il, lapack_int iu, lapack_int* ns,
                           double* s, double* z, lapack_int ldz,
                           lapack_int* superb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper bidiagonal, `'L'` = lower bidiagonal |
| `jobz` | `char` | `'N'` = singular values only, `'V'` = singular values and vectors |
| `range` | `char` | `'A'` = all, `'V'` = in (vl, vu], `'I'` = il-th through iu-th |
| `n` | `lapack_int` | Order of the bidiagonal matrix |
| `d` | `real*` | [in] Diagonal elements, dimension n |
| `e` | `real*` | [in] Off-diagonal elements, dimension n-1 |
| `vl` | `real` | Lower bound of interval (if range='V') |
| `vu` | `real` | Upper bound of interval (if range='V') |
| `il` | `lapack_int` | Index of smallest singular value to return (if range='I', 1-based) |
| `iu` | `lapack_int` | Index of largest singular value to return (if range='I', 1-based) |
| `ns` | `lapack_int*` | [out] Number of singular values found |
| `s` | `real*` | [out] Selected singular values in decreasing order |
| `z` | `real*` | [out] Singular vectors as 2n-by-ns matrix (first n rows = left, last n rows = right) |
| `ldz` | `lapack_int` | Leading dimension of `z` |
| `superb` | `lapack_int*` | [out] If info > 0, indices of unconverged values |

**Note:** Only `s` and `d` variants exist (no complex).

---

## Bidiagonal Reduction

### gebrd - General to Bidiagonal Reduction

Reduce a general m-by-n matrix to upper or lower bidiagonal form by orthogonal/unitary transformations: `Q^H * A * P = B`. This is the first step in computing the SVD.

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

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] m-by-n matrix; on exit, bidiagonal form and Householder vectors |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `d` | `real*` | [out] Diagonal elements of B, dimension min(m,n) |
| `e` | `real*` | [out] Off-diagonal elements of B, dimension min(m,n)-1 |
| `tauq` | `T*` | [out] Scalar factors of elementary reflectors for Q, dimension min(m,n) |
| `taup` | `T*` | [out] Scalar factors of elementary reflectors for P, dimension min(m,n) |

**Result:** If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

---

### gbbrd - Band to Bidiagonal Reduction

Reduce a general band matrix to upper bidiagonal form by orthogonal/unitary transformations: `Q^H * A * P = B`. Optionally also applies Q^H to a given matrix C.

```c
lapack_int LAPACKE_sgbbrd( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, float* ab, lapack_int ldab, float* d,
                           float* e, float* q, lapack_int ldq, float* pt,
                           lapack_int ldpt, float* c, lapack_int ldc );
lapack_int LAPACKE_dgbbrd( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, double* ab, lapack_int ldab,
                           double* d, double* e, double* q, lapack_int ldq,
                           double* pt, lapack_int ldpt, double* c,
                           lapack_int ldc );
lapack_int LAPACKE_cgbbrd( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, lapack_complex_float* ab,
                           lapack_int ldab, float* d, float* e,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_complex_float* pt, lapack_int ldpt,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zgbbrd( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, lapack_complex_double* ab,
                           lapack_int ldab, double* d, double* e,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* pt, lapack_int ldpt,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `vect` | `char` | `'N'` = no Q or P^T, `'Q'` = generate Q, `'P'` = generate P^T, `'B'` = both |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `ncc` | `lapack_int` | Number of columns of C (0 if not referenced) |
| `kl` | `lapack_int` | Number of subdiagonals (lower bandwidth) |
| `ku` | `lapack_int` | Number of superdiagonals (upper bandwidth) |
| `ab` | `T*` | [in/out] Band matrix in band storage, (kl+ku+1)-by-n |
| `ldab` | `lapack_int` | Leading dimension of `ab` |
| `d` | `real*` | [out] Diagonal elements of B, dimension min(m,n) |
| `e` | `real*` | [out] Off-diagonal elements of B, dimension min(m,n)-1 |
| `q` | `T*` | [out] m-by-m orthogonal matrix Q (if vect='Q' or 'B') |
| `ldq` | `lapack_int` | Leading dimension of `q` |
| `pt` | `T*` | [out] n-by-n orthogonal matrix P^T (if vect='P' or 'B') |
| `ldpt` | `lapack_int` | Leading dimension of `pt` |
| `c` | `T*` | [in/out] m-by-ncc matrix C; on exit, Q^H * C |
| `ldc` | `lapack_int` | Leading dimension of `c` |

---

## Generate/Apply Orthogonal Matrices from Bidiagonal Reduction

### orgbr / ungbr - Generate Q or P from gebrd

Generate the orthogonal/unitary matrix Q or P^T determined by `gebrd`. Use `orgbr` for real matrices, `ungbr` for complex matrices.

```c
// Real: orgbr
lapack_int LAPACKE_sorgbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, double* a,
                           lapack_int lda, const double* tau );

// Complex: ungbr
lapack_int LAPACKE_cungbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `vect` | `char` | `'Q'` = generate Q (use tauq from gebrd), `'P'` = generate P^T (use taup from gebrd) |
| `m` | `lapack_int` | Number of rows of the matrix to generate |
| `n` | `lapack_int` | Number of columns of the matrix to generate |
| `k` | `lapack_int` | If vect='Q': number of columns in original A passed to gebrd; if vect='P': number of rows |
| `a` | `T*` | [in/out] On entry, Householder vectors from gebrd; on exit, the orthogonal matrix |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `tau` | `const T*` | [in] Scalar factors from gebrd (tauq if vect='Q', taup if vect='P') |

---

### ormbr / unmbr - Multiply by Q or P from gebrd

Multiply a general matrix by the orthogonal/unitary matrix Q or P determined by `gebrd`, without explicitly forming Q or P. Use `ormbr` for real matrices, `unmbr` for complex matrices.

```c
// Real: ormbr
lapack_int LAPACKE_sormbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

// Complex: unmbr
lapack_int LAPACKE_cunmbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `vect` | `char` | `'Q'` = apply Q (use tauq), `'P'` = apply P^T (use taup) |
| `side` | `char` | `'L'` = multiply from left (Q*C or Q^T*C), `'R'` = from right (C*Q or C*Q^T) |
| `trans` | `char` | `'N'` = no transpose, `'T'` = transpose (real), `'C'` = conjugate transpose (complex) |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | If vect='Q': number of columns in original A; if vect='P': number of rows |
| `a` | `const T*` | [in] Householder vectors from gebrd |
| `lda` | `lapack_int` | Leading dimension of `a` |
| `tau` | `const T*` | [in] Scalar factors from gebrd (tauq if vect='Q', taup if vect='P') |
| `c` | `T*` | [in/out] m-by-n matrix C; on exit, overwritten with the product |
| `ldc` | `lapack_int` | Leading dimension of `c` |

---

## CS Decomposition

The CS (Cosine-Sine) decomposition factorizes a partitioned unitary/orthogonal matrix into block-diagonal unitary factors and a middle factor containing cosines and sines. Used in computing the GSVD and in applications involving angles between subspaces.

### bbcsd - CS Decomposition of Unitary Bidiagonal

Compute the CS decomposition of a unitary matrix whose bidiagonal blocks are given by angles theta.

```c
lapack_int LAPACKE_sbbcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, float* theta, float* phi,
                           float* u1, lapack_int ldu1, float* u2,
                           lapack_int ldu2, float* v1t, lapack_int ldv1t,
                           float* v2t, lapack_int ldv2t, float* b11d,
                           float* b11e, float* b12d, float* b12e, float* b21d,
                           float* b21e, float* b22d, float* b22e );
lapack_int LAPACKE_dbbcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, double* theta,
                           double* phi, double* u1, lapack_int ldu1, double* u2,
                           lapack_int ldu2, double* v1t, lapack_int ldv1t,
                           double* v2t, lapack_int ldv2t, double* b11d,
                           double* b11e, double* b12d, double* b12e,
                           double* b21d, double* b21e, double* b22d,
                           double* b22e );
lapack_int LAPACKE_cbbcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, float* theta, float* phi,
                           lapack_complex_float* u1, lapack_int ldu1,
                           lapack_complex_float* u2, lapack_int ldu2,
                           lapack_complex_float* v1t, lapack_int ldv1t,
                           lapack_complex_float* v2t, lapack_int ldv2t,
                           float* b11d, float* b11e, float* b12d, float* b12e,
                           float* b21d, float* b21e, float* b22d, float* b22e );
lapack_int LAPACKE_zbbcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, double* theta,
                           double* phi, lapack_complex_double* u1,
                           lapack_int ldu1, lapack_complex_double* u2,
                           lapack_int ldu2, lapack_complex_double* v1t,
                           lapack_int ldv1t, lapack_complex_double* v2t,
                           lapack_int ldv2t, double* b11d, double* b11e,
                           double* b12d, double* b12e, double* b21d,
                           double* b21e, double* b22d, double* b22e );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu1` | `char` | `'Y'` = update U1, `'N'` = do not |
| `jobu2` | `char` | `'Y'` = update U2, `'N'` = do not |
| `jobv1t` | `char` | `'Y'` = update V1T, `'N'` = do not |
| `jobv2t` | `char` | `'Y'` = update V2T, `'N'` = do not |
| `trans` | `char` | `'T'` = X is transposed, `'N'` = not transposed |
| `m` | `lapack_int` | Rows and columns of the unitary matrix |
| `p` | `lapack_int` | Rows of upper-left block |
| `q` | `lapack_int` | Columns of upper-left block |
| `theta` | `real*` | [in/out] CS values, dimension q; on exit, sorted angles in [0, pi/2] |
| `phi` | `real*` | [in/out] Off-diagonal elements, dimension q-1 |
| `u1` | `T*` | [in/out] p-by-p unitary matrix U1 |
| `ldu1` | `lapack_int` | Leading dimension of `u1` |
| `u2` | `T*` | [in/out] (m-p)-by-(m-p) unitary matrix U2 |
| `ldu2` | `lapack_int` | Leading dimension of `u2` |
| `v1t` | `T*` | [in/out] q-by-q unitary matrix V1T |
| `ldv1t` | `lapack_int` | Leading dimension of `v1t` |
| `v2t` | `T*` | [in/out] (m-q)-by-(m-q) unitary matrix V2T |
| `ldv2t` | `lapack_int` | Leading dimension of `v2t` |
| `b11d` | `real*` | [out] Diagonal of B11, dimension q |
| `b11e` | `real*` | [out] Off-diagonal of B11, dimension q-1 |
| `b12d` | `real*` | [out] Diagonal of B12, dimension q |
| `b12e` | `real*` | [out] Off-diagonal of B12, dimension q-1 |
| `b21d` | `real*` | [out] Diagonal of B21, dimension q |
| `b21e` | `real*` | [out] Off-diagonal of B21, dimension q-1 |
| `b22d` | `real*` | [out] Diagonal of B22, dimension q |
| `b22e` | `real*` | [out] Off-diagonal of B22, dimension q-1 |

---

### orbdb / unbdb - Simultaneous Bidiagonalization

Simultaneously bidiagonalize the blocks of a partitioned orthogonal/unitary matrix. Use `orbdb` for real matrices, `unbdb` for complex matrices.

```c
// Real: orbdb
lapack_int LAPACKE_sorbdb( int matrix_layout, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q, float* x11,
                           lapack_int ldx11, float* x12, lapack_int ldx12,
                           float* x21, lapack_int ldx21, float* x22,
                           lapack_int ldx22, float* theta, float* phi,
                           float* taup1, float* taup2, float* tauq1,
                           float* tauq2 );
lapack_int LAPACKE_dorbdb( int matrix_layout, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           double* x11, lapack_int ldx11, double* x12,
                           lapack_int ldx12, double* x21, lapack_int ldx21,
                           double* x22, lapack_int ldx22, double* theta,
                           double* phi, double* taup1, double* taup2,
                           double* tauq1, double* tauq2 );

// Complex: unbdb
lapack_int LAPACKE_cunbdb( int matrix_layout, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_float* x11, lapack_int ldx11,
                           lapack_complex_float* x12, lapack_int ldx12,
                           lapack_complex_float* x21, lapack_int ldx21,
                           lapack_complex_float* x22, lapack_int ldx22,
                           float* theta, float* phi,
                           lapack_complex_float* taup1,
                           lapack_complex_float* taup2,
                           lapack_complex_float* tauq1,
                           lapack_complex_float* tauq2 );
lapack_int LAPACKE_zunbdb( int matrix_layout, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_double* x11, lapack_int ldx11,
                           lapack_complex_double* x12, lapack_int ldx12,
                           lapack_complex_double* x21, lapack_int ldx21,
                           lapack_complex_double* x22, lapack_int ldx22,
                           double* theta, double* phi,
                           lapack_complex_double* taup1,
                           lapack_complex_double* taup2,
                           lapack_complex_double* tauq1,
                           lapack_complex_double* tauq2 );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'T'` = X is transposed, `'N'` = not transposed |
| `signs` | `char` | `'O'` = lower-left block is negated, `'D'` = not negated |
| `m` | `lapack_int` | Rows of X (unitary matrix) |
| `p` | `lapack_int` | Rows of X11 and X12 |
| `q` | `lapack_int` | Columns of X11 and X21 |
| `x11` | `T*` | [in/out] p-by-q block |
| `ldx11` | `lapack_int` | Leading dimension of `x11` |
| `x12` | `T*` | [in/out] p-by-(m-q) block |
| `ldx12` | `lapack_int` | Leading dimension of `x12` |
| `x21` | `T*` | [in/out] (m-p)-by-q block |
| `ldx21` | `lapack_int` | Leading dimension of `x21` |
| `x22` | `T*` | [in/out] (m-p)-by-(m-q) block |
| `ldx22` | `lapack_int` | Leading dimension of `x22` |
| `theta` | `real*` | [out] CS angles, dimension q |
| `phi` | `real*` | [out] Off-diagonal angles, dimension q-1 |
| `taup1` | `T*` | [out] Scalar factors for P1 reflectors, dimension p |
| `taup2` | `T*` | [out] Scalar factors for P2 reflectors, dimension m-p |
| `tauq1` | `T*` | [out] Scalar factors for Q1 reflectors, dimension q |
| `tauq2` | `T*` | [out] Scalar factors for Q2 reflectors, dimension m-q |

---

### orcsd / uncsd - CS Decomposition

Compute the CS decomposition of a partitioned orthogonal/unitary matrix. Use `orcsd` for real, `uncsd` for complex.

```c
// Real: orcsd
lapack_int LAPACKE_sorcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q, float* x11,
                           lapack_int ldx11, float* x12, lapack_int ldx12,
                           float* x21, lapack_int ldx21, float* x22,
                           lapack_int ldx22, float* theta, float* u1,
                           lapack_int ldu1, float* u2, lapack_int ldu2,
                           float* v1t, lapack_int ldv1t, float* v2t,
                           lapack_int ldv2t );
lapack_int LAPACKE_dorcsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           double* x11, lapack_int ldx11, double* x12,
                           lapack_int ldx12, double* x21, lapack_int ldx21,
                           double* x22, lapack_int ldx22, double* theta,
                           double* u1, lapack_int ldu1, double* u2,
                           lapack_int ldu2, double* v1t, lapack_int ldv1t,
                           double* v2t, lapack_int ldv2t );

// Complex: uncsd
lapack_int LAPACKE_cuncsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_float* x11, lapack_int ldx11,
                           lapack_complex_float* x12, lapack_int ldx12,
                           lapack_complex_float* x21, lapack_int ldx21,
                           lapack_complex_float* x22, lapack_int ldx22,
                           float* theta, lapack_complex_float* u1,
                           lapack_int ldu1, lapack_complex_float* u2,
                           lapack_int ldu2, lapack_complex_float* v1t,
                           lapack_int ldv1t, lapack_complex_float* v2t,
                           lapack_int ldv2t );
lapack_int LAPACKE_zuncsd( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_double* x11, lapack_int ldx11,
                           lapack_complex_double* x12, lapack_int ldx12,
                           lapack_complex_double* x21, lapack_int ldx21,
                           lapack_complex_double* x22, lapack_int ldx22,
                           double* theta, lapack_complex_double* u1,
                           lapack_int ldu1, lapack_complex_double* u2,
                           lapack_int ldu2, lapack_complex_double* v1t,
                           lapack_int ldv1t, lapack_complex_double* v2t,
                           lapack_int ldv2t );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu1` | `char` | `'Y'` = compute U1, `'N'` = do not |
| `jobu2` | `char` | `'Y'` = compute U2, `'N'` = do not |
| `jobv1t` | `char` | `'Y'` = compute V1T, `'N'` = do not |
| `jobv2t` | `char` | `'Y'` = compute V2T, `'N'` = do not |
| `trans` | `char` | `'T'` = X is transposed, `'N'` = not transposed |
| `signs` | `char` | `'O'` = lower-left block is negated, `'D'` = not negated |
| `m` | `lapack_int` | Rows and columns of unitary matrix X |
| `p` | `lapack_int` | Rows of X11 and X12 |
| `q` | `lapack_int` | Columns of X11 and X21 |
| `x11` | `T*` | [in/out] p-by-q block |
| `ldx11` | `lapack_int` | Leading dimension of `x11` |
| `x12` | `T*` | [in/out] p-by-(m-q) block |
| `ldx12` | `lapack_int` | Leading dimension of `x12` |
| `x21` | `T*` | [in/out] (m-p)-by-q block |
| `ldx21` | `lapack_int` | Leading dimension of `x21` |
| `x22` | `T*` | [in/out] (m-p)-by-(m-q) block |
| `ldx22` | `lapack_int` | Leading dimension of `x22` |
| `theta` | `real*` | [out] CS angles, dimension min(min(p,m-p), min(q,m-q)) |
| `u1` | `T*` | [out] p-by-p unitary matrix |
| `ldu1` | `lapack_int` | Leading dimension of `u1` |
| `u2` | `T*` | [out] (m-p)-by-(m-p) unitary matrix |
| `ldu2` | `lapack_int` | Leading dimension of `u2` |
| `v1t` | `T*` | [out] q-by-q unitary matrix |
| `ldv1t` | `lapack_int` | Leading dimension of `v1t` |
| `v2t` | `T*` | [out] (m-q)-by-(m-q) unitary matrix |
| `ldv2t` | `lapack_int` | Leading dimension of `v2t` |

---

### orcsd2by1 / uncsd2by1 - 2-by-1 CS Decomposition

Compute the CS decomposition of a 2-by-1 partitioned orthogonal/unitary matrix (tall-skinny case). Use `orcsd2by1` for real, `uncsd2by1` for complex.

```c
// Real: orcsd2by1
lapack_int LAPACKE_sorcsd2by1( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, lapack_int m, lapack_int p, lapack_int q,
                           float* x11, lapack_int ldx11, float* x21, lapack_int ldx21,
                           float* theta, float* u1, lapack_int ldu1, float* u2,
                           lapack_int ldu2, float* v1t, lapack_int ldv1t);
lapack_int LAPACKE_dorcsd2by1( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, lapack_int m, lapack_int p, lapack_int q,
                           double* x11, lapack_int ldx11, double* x21, lapack_int ldx21,
                           double* theta, double* u1, lapack_int ldu1, double* u2,
                           lapack_int ldu2, double* v1t, lapack_int ldv1t);

// Complex: uncsd2by1
lapack_int LAPACKE_cuncsd2by1( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_float* x11, lapack_int ldx11,
                           lapack_complex_float* x21, lapack_int ldx21,
                           float* theta, lapack_complex_float* u1,
                           lapack_int ldu1, lapack_complex_float* u2,
                           lapack_int ldu2, lapack_complex_float* v1t, lapack_int ldv1t );
lapack_int LAPACKE_zuncsd2by1( int matrix_layout, char jobu1, char jobu2,
                           char jobv1t, lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_double* x11, lapack_int ldx11,
                           lapack_complex_double* x21, lapack_int ldx21,
                           double* theta, lapack_complex_double* u1,
                           lapack_int ldu1, lapack_complex_double* u2,
                           lapack_int ldu2, lapack_complex_double* v1t, lapack_int ldv1t );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `jobu1` | `char` | `'Y'` = compute U1, `'N'` = do not |
| `jobu2` | `char` | `'Y'` = compute U2, `'N'` = do not |
| `jobv1t` | `char` | `'Y'` = compute V1T, `'N'` = do not |
| `m` | `lapack_int` | Number of rows: m = rows(X11) + rows(X21) |
| `p` | `lapack_int` | Rows of X11 |
| `q` | `lapack_int` | Columns of X11 and X21 |
| `x11` | `T*` | [in/out] p-by-q block |
| `ldx11` | `lapack_int` | Leading dimension of `x11` |
| `x21` | `T*` | [in/out] (m-p)-by-q block |
| `ldx21` | `lapack_int` | Leading dimension of `x21` |
| `theta` | `real*` | [out] CS angles, dimension min(p, m-p, q, m-q) |
| `u1` | `T*` | [out] p-by-p unitary matrix |
| `ldu1` | `lapack_int` | Leading dimension of `u1` |
| `u2` | `T*` | [out] (m-p)-by-(m-p) unitary matrix |
| `ldu2` | `lapack_int` | Leading dimension of `u2` |
| `v1t` | `T*` | [out] q-by-q unitary matrix |
| `ldv1t` | `lapack_int` | Leading dimension of `v1t` |
