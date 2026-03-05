# LAPACKE Least Squares & QR/LQ API Reference

> Routines for least squares problems and orthogonal factorizations.
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [Common Parameters](#common-parameters)
- [Least Squares Drivers](#least-squares-drivers)
  - [gels - QR/LQ Factorization](#gels---qrlq-factorization)
  - [gelsd - SVD Divide and Conquer (Recommended)](#gelsd---svd-divide-and-conquer-recommended)
  - [gelss - SVD](#gelss---svd)
  - [gelsy - Complete Orthogonal Factorization](#gelsy---complete-orthogonal-factorization)
  - [getsls - Simplified QR/LQ](#getsls---simplified-qrlq)
- [Constrained Least Squares](#constrained-least-squares)
  - [ggglm - General Gauss-Markov Linear Model](#ggglm---general-gauss-markov-linear-model)
  - [gglse - Equality-Constrained Least Squares](#gglse---equality-constrained-least-squares)
- [QR Factorizations](#qr-factorizations)
  - [geqrf - Standard QR (Blocked)](#geqrf---standard-qr-blocked)
  - [geqrfp - QR with Non-Negative Diagonal](#geqrfp---qr-with-non-negative-diagonal)
  - [geqr2 - Unblocked QR](#geqr2---unblocked-qr)
  - [geqp3 - QR with Column Pivoting](#geqp3---qr-with-column-pivoting)
  - [geqpf - QR with Column Pivoting (Deprecated)](#geqpf---qr-with-column-pivoting-deprecated)
  - [geqr - Tall-Skinny QR](#geqr---tall-skinny-qr)
- [Blocked QR (WY Representation)](#blocked-qr-wy-representation)
  - [geqrt - Blocked QR with WY](#geqrt---blocked-qr-with-wy)
  - [geqrt2 - Unblocked QR with WY](#geqrt2---unblocked-qr-with-wy)
  - [geqrt3 - Recursive QR with WY](#geqrt3---recursive-qr-with-wy)
  - [getsqrhrt - Tall-Skinny QR with Householder Reconstruction](#getsqrhrt---tall-skinny-qr-with-householder-reconstruction)
- [LQ Factorizations](#lq-factorizations)
  - [gelqf - Standard LQ (Blocked)](#gelqf---standard-lq-blocked)
  - [gelq2 - Unblocked LQ](#gelq2---unblocked-lq)
  - [gelq - Tall-Skinny LQ](#gelq---tall-skinny-lq)
- [QL and RQ Factorizations](#ql-and-rq-factorizations)
  - [geqlf - QL Factorization](#geqlf---ql-factorization)
  - [gerqf - RQ Factorization](#gerqf---rq-factorization)
- [Apply Q from Factorizations](#apply-q-from-factorizations)
  - [gemqr - Apply Q from geqr](#gemqr---apply-q-from-geqr)
  - [gemlq - Apply Q from gelq](#gemlq---apply-q-from-gelq)
  - [gemqrt - Apply Q from geqrt](#gemqrt---apply-q-from-geqrt)
  - [tpmqrt - Apply Q from tpqrt](#tpmqrt---apply-q-from-tpqrt)
- [Triangular-Pentagonal QR](#triangular-pentagonal-qr)
  - [tpqrt - Blocked TP QR](#tpqrt---blocked-tp-qr)
  - [tpqrt2 - Unblocked TP QR](#tpqrt2---unblocked-tp-qr)
  - [tprfb - Apply T from tpqrt](#tprfb---apply-t-from-tpqrt)

## Common Parameters

All LAPACKE routines share these conventions:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `matrix_layout` | `LAPACK_ROW_MAJOR=101`, `LAPACK_COL_MAJOR=102` | Memory layout of matrices. LAPACKE handles internal transposition when row-major is specified. |
| Return value | `lapack_int` | `0` = success, `< 0` = illegal argument at position `-info`, `> 0` = algorithm-specific failure |

**Precision prefixes:** `s` = float, `d` = double, `c` = lapack_complex_float, `z` = lapack_complex_double.

**Choosing a least squares solver:**
- **gelsd** is recommended for most cases -- uses SVD with divide-and-conquer, handles rank-deficient matrices robustly
- **gels** is fastest when A has full rank -- uses QR/LQ factorization
- **getsls** is a simplified interface to gels
- **gelsy** uses complete orthogonal factorization with column pivoting -- good for rank-deficient problems when you need the pivoting information
- **gelss** uses plain SVD -- slower than gelsd, use gelsd instead unless you specifically need gelss behavior

---

## Least Squares Drivers

### gels - QR/LQ Factorization

Solves overdetermined or underdetermined linear systems **min ||b - A*x||_2** using QR or LQ factorization. Assumes A has full rank.

- If m >= n: solves the least squares problem (QR factorization)
- If m < n: finds the minimum norm solution (LQ factorization)

#### LAPACKE_sgels
Single-precision real.
```c
lapack_int LAPACKE_sgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, float* a,
                          lapack_int lda, float* b, lapack_int ldb );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `trans` | `char` | `'N'`: solve A*x = b; `'T'`: solve A^T*x = b |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `nrhs` | `lapack_int` | Number of right-hand sides (columns of B) |
| `a` | `float*` | Matrix A (m x n), overwritten with factorization on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | On entry: matrix B (max(m,n) x nrhs). On exit: solution X |
| `ldb` | `lapack_int` | Leading dimension of B |

#### LAPACKE_dgels
Double-precision real.
```c
lapack_int LAPACKE_dgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, double* a,
                          lapack_int lda, double* b, lapack_int ldb );
```

**Parameters:** Same as `LAPACKE_sgels` with `double` replacing `float`.

#### LAPACKE_cgels
Single-precision complex.
```c
lapack_int LAPACKE_cgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb );
```

**Parameters:** Same structure. `trans` = `'N'` or `'C'` (conjugate transpose). Complex arrays replace float arrays.

#### LAPACKE_zgels
Double-precision complex.
```c
lapack_int LAPACKE_zgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same structure as `LAPACKE_cgels` with `lapack_complex_double` replacing `lapack_complex_float`.

---

### gelsd - SVD Divide and Conquer (Recommended)

Solves **min ||b - A*x||_2** using the singular value decomposition (SVD) with divide-and-conquer. Handles rank-deficient matrices. **Preferred over gelss for most applications.**

#### LAPACKE_sgelsd
Single-precision real.
```c
lapack_int LAPACKE_sgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `nrhs` | `lapack_int` | Number of right-hand sides (columns of B) |
| `a` | `float*` | Matrix A (m x n), overwritten on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | On entry: B (max(m,n) x nrhs). On exit: solution X |
| `ldb` | `lapack_int` | Leading dimension of B |
| `s` | `float*` | Output: singular values of A in decreasing order, length min(m,n) |
| `rcond` | `float` | Threshold for determining effective rank. Singular values s(i) <= rcond*s(1) are treated as zero. Use -1 for machine precision. |
| `rank` | `lapack_int*` | Output: effective rank of A |

#### LAPACKE_dgelsd
Double-precision real.
```c
lapack_int LAPACKE_dgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
```

**Parameters:** Same as `LAPACKE_sgelsd` with `double` replacing `float`.

#### LAPACKE_cgelsd
Single-precision complex.
```c
lapack_int LAPACKE_cgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure. Note `s` remains `float*` (singular values are always real). Matrix arrays use `lapack_complex_float`.

#### LAPACKE_zgelsd
Double-precision complex.
```c
lapack_int LAPACKE_zgelsd( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure as `LAPACKE_cgelsd` with double-precision types. `s` is `double*`.

---

### gelss - SVD

Solves **min ||b - A*x||_2** using the singular value decomposition (SVD). Same interface as gelsd but uses a different SVD algorithm. **Prefer gelsd** unless you specifically need this variant.

#### LAPACKE_sgelss
Single-precision real.
```c
lapack_int LAPACKE_sgelss( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
```

**Parameters:** Identical to `LAPACKE_sgelsd`.

#### LAPACKE_dgelss
Double-precision real.
```c
lapack_int LAPACKE_dgelss( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
```

**Parameters:** Same as `LAPACKE_sgelss` with `double` replacing `float`.

#### LAPACKE_cgelss
Single-precision complex.
```c
lapack_int LAPACKE_cgelss( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure. `s` remains `float*`.

#### LAPACKE_zgelss
Double-precision complex.
```c
lapack_int LAPACKE_zgelss( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure as `LAPACKE_cgelss` with double-precision types.

---

### gelsy - Complete Orthogonal Factorization

Solves **min ||b - A*x||_2** using complete orthogonal factorization with column pivoting. Handles rank-deficient matrices. Provides pivot information via `jpvt`.

#### LAPACKE_sgelsy
Single-precision real.
```c
lapack_int LAPACKE_sgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `nrhs` | `lapack_int` | Number of right-hand sides |
| `a` | `float*` | Matrix A (m x n), overwritten on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | On entry: B (max(m,n) x nrhs). On exit: solution X |
| `ldb` | `lapack_int` | Leading dimension of B |
| `jpvt` | `lapack_int*` | On entry: if jpvt(i) != 0, column i is permuted to front; if 0, column is free. On exit: columns reordered as jpvt(i)-th column of A*P. Length n. |
| `rcond` | `float` | Threshold for determining effective rank |
| `rank` | `lapack_int*` | Output: effective rank of A |

#### LAPACKE_dgelsy
Double-precision real.
```c
lapack_int LAPACKE_dgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, lapack_int* jpvt,
                           double rcond, lapack_int* rank );
```

**Parameters:** Same as `LAPACKE_sgelsy` with `double` replacing `float`.

#### LAPACKE_cgelsy
Single-precision complex.
```c
lapack_int LAPACKE_cgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure with complex types. `rcond` remains `float`.

#### LAPACKE_zgelsy
Double-precision complex.
```c
lapack_int LAPACKE_zgelsy( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* jpvt, double rcond,
                           lapack_int* rank );
```

**Parameters:** Same structure as `LAPACKE_cgelsy` with double-precision types.

---

### getsls - Simplified QR/LQ

Solves overdetermined or underdetermined linear systems using QR or LQ factorization. Simplified interface compared to gels, same algorithm. Assumes A has full rank.

#### LAPACKE_sgetsls
Single-precision real.
```c
lapack_int LAPACKE_sgetsls( int matrix_layout, char trans, lapack_int m,
                            lapack_int n, lapack_int nrhs, float* a,
                            lapack_int lda, float* b, lapack_int ldb );
```

**Parameters:** Same as `LAPACKE_sgels`.

#### LAPACKE_dgetsls
Double-precision real.
```c
lapack_int LAPACKE_dgetsls( int matrix_layout, char trans, lapack_int m,
                            lapack_int n, lapack_int nrhs, double* a,
                            lapack_int lda, double* b, lapack_int ldb );
```

**Parameters:** Same as `LAPACKE_sgetsls` with `double` replacing `float`.

#### LAPACKE_cgetsls
Single-precision complex.
```c
lapack_int LAPACKE_cgetsls( int matrix_layout, char trans, lapack_int m,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb );
```

**Parameters:** Same structure. `trans` = `'N'` or `'C'`.

#### LAPACKE_zgetsls
Double-precision complex.
```c
lapack_int LAPACKE_zgetsls( int matrix_layout, char trans, lapack_int m,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same structure as `LAPACKE_cgetsls` with double-precision types.

---

## Constrained Least Squares

### ggglm - General Gauss-Markov Linear Model

Solves the general Gauss-Markov linear model problem: **min ||y||_2** subject to **d = A*x + B*y**, where A is n-by-m, B is n-by-p, with m <= n <= m+p. Uses the generalized QR factorization of (A, B).

#### LAPACKE_sggglm
Single-precision real.
```c
lapack_int LAPACKE_sggglm( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* d, float* x, float* y );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Number of rows of A and B |
| `m` | `lapack_int` | Number of columns of A |
| `p` | `lapack_int` | Number of columns of B |
| `a` | `float*` | Matrix A (n x m), overwritten on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | Matrix B (n x p), overwritten on exit |
| `ldb` | `lapack_int` | Leading dimension of B |
| `d` | `float*` | On entry: vector d (length n). Overwritten on exit. |
| `x` | `float*` | Output: solution vector x (length m) |
| `y` | `float*` | Output: solution vector y (length p) |

#### LAPACKE_dggglm
Double-precision real.
```c
lapack_int LAPACKE_dggglm( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, double* a, lapack_int lda, double* b,
                           lapack_int ldb, double* d, double* x, double* y );
```

**Parameters:** Same as `LAPACKE_sggglm` with `double` replacing `float`.

#### LAPACKE_cggglm
Single-precision complex.
```c
lapack_int LAPACKE_cggglm( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* d,
                           lapack_complex_float* x, lapack_complex_float* y );
```

**Parameters:** Same structure with `lapack_complex_float` types.

#### LAPACKE_zggglm
Double-precision complex.
```c
lapack_int LAPACKE_zggglm( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* d,
                           lapack_complex_double* x, lapack_complex_double* y );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### gglse - Equality-Constrained Least Squares

Solves the equality-constrained least squares problem: **min ||c - A*x||_2** subject to **B*x = d**, where A is m-by-n, B is p-by-n, with p <= n <= m+p. Uses the generalized RQ factorization of (B, A).

#### LAPACKE_sgglse
Single-precision real.
```c
lapack_int LAPACKE_sgglse( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int p, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* c, float* d, float* x );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A and B |
| `p` | `lapack_int` | Number of rows of B (number of equality constraints) |
| `a` | `float*` | Matrix A (m x n), overwritten on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | Matrix B (p x n), overwritten on exit |
| `ldb` | `lapack_int` | Leading dimension of B |
| `c` | `float*` | On entry: vector c (length m). Overwritten on exit. |
| `d` | `float*` | On entry: vector d (length p). Overwritten on exit. |
| `x` | `float*` | Output: solution vector x (length n) |

#### LAPACKE_dgglse
Double-precision real.
```c
lapack_int LAPACKE_dgglse( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int p, double* a, lapack_int lda, double* b,
                           lapack_int ldb, double* c, double* d, double* x );
```

**Parameters:** Same as `LAPACKE_sgglse` with `double` replacing `float`.

#### LAPACKE_cgglse
Single-precision complex.
```c
lapack_int LAPACKE_cgglse( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* c,
                           lapack_complex_float* d, lapack_complex_float* x );
```

**Parameters:** Same structure with `lapack_complex_float` types.

#### LAPACKE_zgglse
Double-precision complex.
```c
lapack_int LAPACKE_zgglse( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* c,
                           lapack_complex_double* d, lapack_complex_double* x );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

## QR Factorizations

### geqrf - Standard QR (Blocked)

Computes QR factorization of an m-by-n matrix: **A = Q * R**. Uses blocked algorithm for performance. Q is represented implicitly as a product of elementary reflectors: Q = H(1) * H(2) * ... * H(k), where k = min(m,n).

#### LAPACKE_sgeqrf
Single-precision real.
```c
lapack_int LAPACKE_sgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `float*` | On entry: matrix A (m x n). On exit: R in upper triangle; reflectors below diagonal. |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `float*` | Output: scalar factors of reflectors, length min(m,n) |

#### LAPACKE_dgeqrf
Double-precision real.
```c
lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

**Parameters:** Same as `LAPACKE_sgeqrf` with `double` replacing `float`.

#### LAPACKE_cgeqrf
Single-precision complex.
```c
lapack_int LAPACKE_cgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgeqrf
Double-precision complex.
```c
lapack_int LAPACKE_zgeqrf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### geqrfp - QR with Non-Negative Diagonal

Same as geqrf but guarantees non-negative diagonal elements in R. Computes **A = Q * R** where the diagonal of R has non-negative entries.

#### LAPACKE_sgeqrfp
Single-precision real.
```c
lapack_int LAPACKE_sgeqrfp( int matrix_layout, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* tau );
```

#### LAPACKE_dgeqrfp
Double-precision real.
```c
lapack_int LAPACKE_dgeqrfp( int matrix_layout, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* tau );
```

#### LAPACKE_cgeqrfp
Single-precision complex.
```c
lapack_int LAPACKE_cgeqrfp( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* tau );
```

#### LAPACKE_zgeqrfp
Double-precision complex.
```c
lapack_int LAPACKE_zgeqrfp( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* tau );
```

**Parameters:** Same as geqrf for all variants.

---

### geqr2 - Unblocked QR

Computes QR factorization using the unblocked (Level 2 BLAS) algorithm. Use geqrf for better performance on large matrices. Primarily used as a building block by other routines.

#### LAPACKE_sgeqr2
Single-precision real.
```c
lapack_int LAPACKE_sgeqr2( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

#### LAPACKE_dgeqr2
Double-precision real.
```c
lapack_int LAPACKE_dgeqr2( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

#### LAPACKE_cgeqr2
Single-precision complex.
```c
lapack_int LAPACKE_cgeqr2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

#### LAPACKE_zgeqr2
Double-precision complex.
```c
lapack_int LAPACKE_zgeqr2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:** Same as geqrf for all variants.

---

### geqp3 - QR with Column Pivoting

Computes QR factorization with column pivoting: **A * P = Q * R**, where P is a permutation matrix. Useful for rank-revealing factorizations.

#### LAPACKE_sgeqp3
Single-precision real.
```c
lapack_int LAPACKE_sgeqp3( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* jpvt,
                           float* tau );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `float*` | On entry: matrix A (m x n). On exit: R in upper triangle; reflectors below. |
| `lda` | `lapack_int` | Leading dimension of A |
| `jpvt` | `lapack_int*` | On entry: if jpvt(i) != 0, column i is permuted first; if 0, free column. On exit: permutation indices. Length n. |
| `tau` | `float*` | Output: scalar factors of reflectors, length min(m,n) |

#### LAPACKE_dgeqp3
Double-precision real.
```c
lapack_int LAPACKE_dgeqp3( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* jpvt,
                           double* tau );
```

**Parameters:** Same as `LAPACKE_sgeqp3` with `double` replacing `float`.

#### LAPACKE_cgeqp3
Single-precision complex.
```c
lapack_int LAPACKE_cgeqp3( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_float* tau );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgeqp3
Double-precision complex.
```c
lapack_int LAPACKE_zgeqp3( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_double* tau );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### geqpf - QR with Column Pivoting (Deprecated)

**Deprecated:** Use geqp3 instead. Computes QR factorization with column pivoting using Level 2 BLAS. Same interface as geqp3 but slower.

#### LAPACKE_sgeqpf
Single-precision real.
```c
lapack_int LAPACKE_sgeqpf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* jpvt,
                           float* tau );
```

#### LAPACKE_dgeqpf
Double-precision real.
```c
lapack_int LAPACKE_dgeqpf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* jpvt,
                           double* tau );
```

#### LAPACKE_cgeqpf
Single-precision complex.
```c
lapack_int LAPACKE_cgeqpf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_float* tau );
```

#### LAPACKE_zgeqpf
Double-precision complex.
```c
lapack_int LAPACKE_zgeqpf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_double* tau );
```

**Parameters:** Same as geqp3 for all variants.

---

### geqr - Tall-Skinny QR

Computes QR factorization using the tall-skinny QR (TSQR) algorithm. Designed for matrices where m >> n. The reflector representation is stored in a compact format in the output array `t`.

#### LAPACKE_sgeqr
Single-precision real.
```c
lapack_int LAPACKE_sgeqr( int matrix_layout, lapack_int m, lapack_int n,
                          float* a, lapack_int lda,
                          float* t, lapack_int tsize );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `float*` | On entry: matrix A (m x n). On exit: R in upper triangle; reflectors below. |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `float*` | Output: compact representation of Q. Use with gemqr to apply Q. |
| `tsize` | `lapack_int` | Size of array t. Use -1 for workspace query. Minimum 5 for workspace query. |

#### LAPACKE_dgeqr
Double-precision real.
```c
lapack_int LAPACKE_dgeqr( int matrix_layout, lapack_int m, lapack_int n,
                          double* a, lapack_int lda,
                          double* t, lapack_int tsize );
```

**Parameters:** Same as `LAPACKE_sgeqr` with `double` replacing `float`.

#### LAPACKE_cgeqr
Single-precision complex.
```c
lapack_int LAPACKE_cgeqr( int matrix_layout, lapack_int m, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* t, lapack_int tsize );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgeqr
Double-precision complex.
```c
lapack_int LAPACKE_zgeqr( int matrix_layout, lapack_int m, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* t, lapack_int tsize );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

## Blocked QR (WY Representation)

These routines store Q in the WY representation (compact block reflector format) using an upper triangular matrix T, enabling efficient Level 3 BLAS operations when applying Q.

### geqrt - Blocked QR with WY

Computes QR factorization with WY representation: **A = Q * R** where Q = I - V * T * V^T. V and T together represent the orthogonal matrix Q in a form suitable for block operations.

#### LAPACKE_sgeqrt
Single-precision real.
```c
lapack_int LAPACKE_sgeqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nb, float* a, lapack_int lda, float* t,
                           lapack_int ldt );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A (m >= n) |
| `n` | `lapack_int` | Number of columns of A |
| `nb` | `lapack_int` | Block size (1 <= nb <= n) |
| `a` | `float*` | On entry: matrix A (m x n). On exit: R in upper triangle; V (reflectors) below. |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `float*` | Output: upper triangular block reflectors T (nb x n). Use with gemqrt to apply Q. |
| `ldt` | `lapack_int` | Leading dimension of T |

#### LAPACKE_dgeqrt
Double-precision real.
```c
lapack_int LAPACKE_dgeqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nb, double* a, lapack_int lda, double* t,
                           lapack_int ldt );
```

**Parameters:** Same as `LAPACKE_sgeqrt` with `double` replacing `float`.

#### LAPACKE_cgeqrt
Single-precision complex.
```c
lapack_int LAPACKE_cgeqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nb, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* t,
                           lapack_int ldt );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgeqrt
Double-precision complex.
```c
lapack_int LAPACKE_zgeqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int nb, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* t,
                           lapack_int ldt );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### geqrt2 - Unblocked QR with WY

Computes QR factorization with WY representation using unblocked algorithm. Used as the base case for recursive/blocked variants.

#### LAPACKE_sgeqrt2
Single-precision real.
```c
lapack_int LAPACKE_sgeqrt2( int matrix_layout, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* t,
                            lapack_int ldt );
```

#### LAPACKE_dgeqrt2
Double-precision real.
```c
lapack_int LAPACKE_dgeqrt2( int matrix_layout, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* t,
                            lapack_int ldt );
```

#### LAPACKE_cgeqrt2
Single-precision complex.
```c
lapack_int LAPACKE_cgeqrt2( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* t, lapack_int ldt );
```

#### LAPACKE_zgeqrt2
Double-precision complex.
```c
lapack_int LAPACKE_zgeqrt2( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* t, lapack_int ldt );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A (m >= n) |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `<type>*` | On entry: matrix A (m x n). On exit: R in upper triangle; V below. |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `<type>*` | Output: upper triangular block reflector T (n x n) |
| `ldt` | `lapack_int` | Leading dimension of T |

---

### geqrt3 - Recursive QR with WY

Computes QR factorization with WY representation using a recursive algorithm. Divides the matrix and computes QR factorizations recursively.

#### LAPACKE_sgeqrt3
Single-precision real.
```c
lapack_int LAPACKE_sgeqrt3( int matrix_layout, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* t,
                            lapack_int ldt );
```

#### LAPACKE_dgeqrt3
Double-precision real.
```c
lapack_int LAPACKE_dgeqrt3( int matrix_layout, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* t,
                            lapack_int ldt );
```

#### LAPACKE_cgeqrt3
Single-precision complex.
```c
lapack_int LAPACKE_cgeqrt3( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* t, lapack_int ldt );
```

#### LAPACKE_zgeqrt3
Double-precision complex.
```c
lapack_int LAPACKE_zgeqrt3( int matrix_layout, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* t, lapack_int ldt );
```

**Parameters:** Same as geqrt2 for all variants.

---

### getsqrhrt - Tall-Skinny QR with Householder Reconstruction

Computes the tall-skinny QR factorization with Householder reconstruction. Takes the output of a TSQR factorization and reconstructs the standard Householder representation. This enables communication-avoiding QR for distributed/parallel computations.

#### LAPACKE_sgetsqrhrt
Single-precision real.
```c
lapack_int LAPACKE_sgetsqrhrt( int matrix_layout, lapack_int m, lapack_int n,
                               lapack_int mb1, lapack_int nb1, lapack_int nb2,
                               float* a, lapack_int lda,
                               float* t, lapack_int ldt );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A (m >= n) |
| `n` | `lapack_int` | Number of columns of A |
| `mb1` | `lapack_int` | Row block size used in TSQR factorization |
| `nb1` | `lapack_int` | Column block size for internal TSQR factorization |
| `nb2` | `lapack_int` | Column block size for Householder reconstruction |
| `a` | `float*` | On entry: output from TSQR. On exit: standard QR representation. |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `float*` | Output: block reflectors (nb2 x n) |
| `ldt` | `lapack_int` | Leading dimension of T |

#### LAPACKE_dgetsqrhrt
Double-precision real.
```c
lapack_int LAPACKE_dgetsqrhrt( int matrix_layout, lapack_int m, lapack_int n,
                               lapack_int mb1, lapack_int nb1, lapack_int nb2,
                               double* a, lapack_int lda,
                               double* t, lapack_int ldt );
```

**Parameters:** Same as `LAPACKE_sgetsqrhrt` with `double` replacing `float`.

#### LAPACKE_cgetsqrhrt
Single-precision complex.
```c
lapack_int LAPACKE_cgetsqrhrt( int matrix_layout, lapack_int m, lapack_int n,
                               lapack_int mb1, lapack_int nb1, lapack_int nb2,
                               lapack_complex_float* a, lapack_int lda,
                               lapack_complex_float* t, lapack_int ldt );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgetsqrhrt
Double-precision complex.
```c
lapack_int LAPACKE_zgetsqrhrt( int matrix_layout, lapack_int m, lapack_int n,
                               lapack_int mb1, lapack_int nb1, lapack_int nb2,
                               lapack_complex_double* a, lapack_int lda,
                               lapack_complex_double* t, lapack_int ldt );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

## LQ Factorizations

### gelqf - Standard LQ (Blocked)

Computes LQ factorization of an m-by-n matrix: **A = L * Q**, where L is lower triangular and Q is orthogonal/unitary. Uses blocked algorithm. Q is represented as a product of elementary reflectors.

#### LAPACKE_sgelqf
Single-precision real.
```c
lapack_int LAPACKE_sgelqf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `float*` | On entry: matrix A (m x n). On exit: L in lower triangle; reflectors above diagonal. |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `float*` | Output: scalar factors of reflectors, length min(m,n) |

#### LAPACKE_dgelqf
Double-precision real.
```c
lapack_int LAPACKE_dgelqf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

**Parameters:** Same as `LAPACKE_sgelqf` with `double` replacing `float`.

#### LAPACKE_cgelqf
Single-precision complex.
```c
lapack_int LAPACKE_cgelqf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgelqf
Double-precision complex.
```c
lapack_int LAPACKE_zgelqf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### gelq2 - Unblocked LQ

Computes LQ factorization using unblocked (Level 2 BLAS) algorithm. Use gelqf for better performance on large matrices.

#### LAPACKE_sgelq2
Single-precision real.
```c
lapack_int LAPACKE_sgelq2( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

#### LAPACKE_dgelq2
Double-precision real.
```c
lapack_int LAPACKE_dgelq2( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

#### LAPACKE_cgelq2
Single-precision complex.
```c
lapack_int LAPACKE_cgelq2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

#### LAPACKE_zgelq2
Double-precision complex.
```c
lapack_int LAPACKE_zgelq2( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:** Same as gelqf for all variants.

---

### gelq - Tall-Skinny LQ

Computes LQ factorization using the tall-skinny LQ (TSLQ) algorithm, designed for matrices where n >> m. Compact representation stored in `t`.

#### LAPACKE_sgelq
Single-precision real.
```c
lapack_int LAPACKE_sgelq( int matrix_layout, lapack_int m, lapack_int n,
                          float* a, lapack_int lda,
                          float* t, lapack_int tsize );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `float*` | On entry: matrix A (m x n). On exit: L in lower triangle; reflectors above. |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `float*` | Output: compact representation of Q. Use with gemlq to apply Q. |
| `tsize` | `lapack_int` | Size of array t. Use -1 for workspace query. |

#### LAPACKE_dgelq
Double-precision real.
```c
lapack_int LAPACKE_dgelq( int matrix_layout, lapack_int m, lapack_int n,
                          double* a, lapack_int lda,
                          double* t, lapack_int tsize );
```

**Parameters:** Same as `LAPACKE_sgelq` with `double` replacing `float`.

#### LAPACKE_cgelq
Single-precision complex.
```c
lapack_int LAPACKE_cgelq( int matrix_layout, lapack_int m, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* t, lapack_int tsize );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_zgelq
Double-precision complex.
```c
lapack_int LAPACKE_zgelq( int matrix_layout, lapack_int m, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* t, lapack_int tsize );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

## QL and RQ Factorizations

### geqlf - QL Factorization

Computes QL factorization of an m-by-n matrix: **A = Q * L**, where Q is orthogonal/unitary and L is lower triangular. The last min(m,n) columns of Q form the orthonormal basis.

#### LAPACKE_sgeqlf
Single-precision real.
```c
lapack_int LAPACKE_sgeqlf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

#### LAPACKE_dgeqlf
Double-precision real.
```c
lapack_int LAPACKE_dgeqlf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

#### LAPACKE_cgeqlf
Single-precision complex.
```c
lapack_int LAPACKE_cgeqlf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

#### LAPACKE_zgeqlf
Double-precision complex.
```c
lapack_int LAPACKE_zgeqlf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `<type>*` | On entry: matrix A (m x n). On exit: L in lower triangle of last min(m,n) columns; reflectors above. |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `<type>*` | Output: scalar factors of reflectors, length min(m,n) |

---

### gerqf - RQ Factorization

Computes RQ factorization of an m-by-n matrix: **A = R * Q**, where R is upper triangular and Q is orthogonal/unitary.

#### LAPACKE_sgerqf
Single-precision real.
```c
lapack_int LAPACKE_sgerqf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
```

#### LAPACKE_dgerqf
Double-precision real.
```c
lapack_int LAPACKE_dgerqf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
```

#### LAPACKE_cgerqf
Single-precision complex.
```c
lapack_int LAPACKE_cgerqf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
```

#### LAPACKE_zgerqf
Double-precision complex.
```c
lapack_int LAPACKE_zgerqf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `<type>*` | On entry: matrix A (m x n). On exit: R in upper triangle of last min(m,n) rows; reflectors below. |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `<type>*` | Output: scalar factors of reflectors, length min(m,n) |

---

## Apply Q from Factorizations

### gemqr - Apply Q from geqr

Multiplies a matrix C by the orthogonal/unitary matrix Q obtained from geqr (tall-skinny QR). Computes **Q*C**, **Q^T*C**, **C*Q**, or **C*Q^T**.

#### LAPACKE_sgemqr
Single-precision real.
```c
lapack_int LAPACKE_sgemqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda,
                           const float* t, lapack_int tsize,
                           float* c, lapack_int ldc );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'`: apply Q from left (Q*C); `'R'`: apply Q from right (C*Q) |
| `trans` | `char` | `'N'`: apply Q; `'T'`/`'C'`: apply Q^T or Q^H |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | Number of reflectors (columns used in QR factorization) |
| `a` | `const float*` | Matrix A from geqr output |
| `lda` | `lapack_int` | Leading dimension of A |
| `t` | `const float*` | Array t from geqr output |
| `tsize` | `lapack_int` | Size of array t (from geqr) |
| `c` | `float*` | On entry: matrix C (m x n). On exit: product Q*C, Q^T*C, C*Q, or C*Q^T. |
| `ldc` | `lapack_int` | Leading dimension of C |

#### LAPACKE_dgemqr
Double-precision real.
```c
lapack_int LAPACKE_dgemqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda,
                           const double* t, lapack_int tsize,
                           double* c, lapack_int ldc );
```

**Parameters:** Same as `LAPACKE_sgemqr` with `double` replacing `float`.

#### LAPACKE_cgemqr
Single-precision complex.
```c
lapack_int LAPACKE_cgemqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* t, lapack_int tsize,
                           lapack_complex_float* c, lapack_int ldc );
```

**Parameters:** Same structure. `trans` = `'N'` or `'C'` (conjugate transpose).

#### LAPACKE_zgemqr
Double-precision complex.
```c
lapack_int LAPACKE_zgemqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* t, lapack_int tsize,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### gemlq - Apply Q from gelq

Multiplies a matrix C by the orthogonal/unitary matrix Q obtained from gelq (tall-skinny LQ). Computes **Q*C**, **Q^T*C**, **C*Q**, or **C*Q^T**.

#### LAPACKE_sgemlq
Single-precision real.
```c
lapack_int LAPACKE_sgemlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda,
                           const float* t, lapack_int tsize,
                           float* c, lapack_int ldc );
```

#### LAPACKE_dgemlq
Double-precision real.
```c
lapack_int LAPACKE_dgemlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda,
                           const double* t, lapack_int tsize,
                           double* c, lapack_int ldc );
```

#### LAPACKE_cgemlq
Single-precision complex.
```c
lapack_int LAPACKE_cgemlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* t, lapack_int tsize,
                           lapack_complex_float* c, lapack_int ldc );
```

#### LAPACKE_zgemlq
Double-precision complex.
```c
lapack_int LAPACKE_zgemlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* t, lapack_int tsize,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as gemqr. The `a` and `t` arrays come from gelq output instead of geqr.

---

### gemqrt - Apply Q from geqrt

Multiplies a matrix C by the orthogonal/unitary matrix Q obtained from geqrt (blocked QR with WY representation). Uses the V and T factors for efficient block application.

#### LAPACKE_sgemqrt
Single-precision real.
```c
lapack_int LAPACKE_sgemqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const float* v, lapack_int ldv,
                            const float* t, lapack_int ldt, float* c,
                            lapack_int ldc );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'`: apply Q from left; `'R'`: apply Q from right |
| `trans` | `char` | `'N'`: apply Q; `'T'`/`'C'`: apply Q^T or Q^H |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | Number of reflectors |
| `nb` | `lapack_int` | Block size used in geqrt factorization |
| `v` | `const float*` | Matrix V from geqrt (reflector vectors) |
| `ldv` | `lapack_int` | Leading dimension of V |
| `t` | `const float*` | Matrix T from geqrt (block reflectors) |
| `ldt` | `lapack_int` | Leading dimension of T |
| `c` | `float*` | On entry: matrix C (m x n). On exit: product with Q. |
| `ldc` | `lapack_int` | Leading dimension of C |

#### LAPACKE_dgemqrt
Double-precision real.
```c
lapack_int LAPACKE_dgemqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const double* v, lapack_int ldv,
                            const double* t, lapack_int ldt, double* c,
                            lapack_int ldc );
```

**Parameters:** Same as `LAPACKE_sgemqrt` with `double` replacing `float`.

#### LAPACKE_cgemqrt
Single-precision complex.
```c
lapack_int LAPACKE_cgemqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const lapack_complex_float* v,
                            lapack_int ldv, const lapack_complex_float* t,
                            lapack_int ldt, lapack_complex_float* c,
                            lapack_int ldc );
```

**Parameters:** Same structure with complex types. `trans` = `'N'` or `'C'`.

#### LAPACKE_zgemqrt
Double-precision complex.
```c
lapack_int LAPACKE_zgemqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const lapack_complex_double* v,
                            lapack_int ldv, const lapack_complex_double* t,
                            lapack_int ldt, lapack_complex_double* c,
                            lapack_int ldc );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### tpmqrt - Apply Q from tpqrt

Applies the orthogonal/unitary matrix Q obtained from tpqrt (triangular-pentagonal QR) to a pair of matrices (A, B).

#### LAPACKE_stpmqrt
Single-precision real.
```c
lapack_int LAPACKE_stpmqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int l, lapack_int nb, const float* v,
                            lapack_int ldv, const float* t, lapack_int ldt,
                            float* a, lapack_int lda, float* b,
                            lapack_int ldb );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'`: apply Q from left; `'R'`: apply Q from right |
| `trans` | `char` | `'N'`: apply Q; `'T'`/`'C'`: apply Q^T or Q^H |
| `m` | `lapack_int` | Number of rows of A and B |
| `n` | `lapack_int` | Number of columns of A and B |
| `k` | `lapack_int` | Number of reflectors |
| `l` | `lapack_int` | Number of rows of pentagonal part of V |
| `nb` | `lapack_int` | Block size |
| `v` | `const float*` | Matrix V from tpqrt (pentagonal reflectors) |
| `ldv` | `lapack_int` | Leading dimension of V |
| `t` | `const float*` | Matrix T from tpqrt (block reflectors) |
| `ldt` | `lapack_int` | Leading dimension of T |
| `a` | `float*` | Matrix A, modified on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | Matrix B, modified on exit |
| `ldb` | `lapack_int` | Leading dimension of B |

#### LAPACKE_dtpmqrt
Double-precision real.
```c
lapack_int LAPACKE_dtpmqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int l, lapack_int nb, const double* v,
                            lapack_int ldv, const double* t, lapack_int ldt,
                            double* a, lapack_int lda, double* b,
                            lapack_int ldb );
```

**Parameters:** Same as `LAPACKE_stpmqrt` with `double` replacing `float`.

#### LAPACKE_ctpmqrt
Single-precision complex.
```c
lapack_int LAPACKE_ctpmqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int l, lapack_int nb,
                            const lapack_complex_float* v, lapack_int ldv,
                            const lapack_complex_float* t, lapack_int ldt,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb );
```

**Parameters:** Same structure with complex types. `trans` = `'N'` or `'C'`.

#### LAPACKE_ztpmqrt
Double-precision complex.
```c
lapack_int LAPACKE_ztpmqrt( int matrix_layout, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int l, lapack_int nb,
                            const lapack_complex_double* v, lapack_int ldv,
                            const lapack_complex_double* t, lapack_int ldt,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

## Triangular-Pentagonal QR

Communication-avoiding QR factorization for distributed/parallel systems. Factorizes a matrix pair where the top block is triangular and the bottom block is general (pentagonal).

### tpqrt - Blocked TP QR

Computes a blocked QR factorization of a triangular-pentagonal matrix pair (A, B): replaces A with R and computes block reflectors V and T such that Q = I - V*T*V^T.

#### LAPACKE_stpqrt
Single-precision real.
```c
lapack_int LAPACKE_stpqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int l, lapack_int nb, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float* t,
                           lapack_int ldt );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of B |
| `n` | `lapack_int` | Number of columns of A and B (order of triangular A) |
| `l` | `lapack_int` | Number of rows of upper trapezoidal part of B (0 <= l <= min(m,n)) |
| `nb` | `lapack_int` | Block size (1 <= nb <= n) |
| `a` | `float*` | Upper triangular matrix A (n x n), modified on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | Pentagonal matrix B (m x n), overwritten with V on exit |
| `ldb` | `lapack_int` | Leading dimension of B |
| `t` | `float*` | Output: upper triangular block reflectors T (nb x n) |
| `ldt` | `lapack_int` | Leading dimension of T |

#### LAPACKE_dtpqrt
Double-precision real.
```c
lapack_int LAPACKE_dtpqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int l, lapack_int nb, double* a,
                           lapack_int lda, double* b, lapack_int ldb, double* t,
                           lapack_int ldt );
```

**Parameters:** Same as `LAPACKE_stpqrt` with `double` replacing `float`.

#### LAPACKE_ctpqrt
Single-precision complex.
```c
lapack_int LAPACKE_ctpqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int l, lapack_int nb,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* t, lapack_int ldt );
```

**Parameters:** Same structure with complex types.

#### LAPACKE_ztpqrt
Double-precision complex.
```c
lapack_int LAPACKE_ztpqrt( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int l, lapack_int nb,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* t, lapack_int ldt );
```

**Parameters:** Same structure with `lapack_complex_double` types.

---

### tpqrt2 - Unblocked TP QR

Computes QR factorization of a triangular-pentagonal matrix pair using the unblocked algorithm. Used as a building block by tpqrt.

#### LAPACKE_stpqrt2
Single-precision real.
```c
lapack_int LAPACKE_stpqrt2( int matrix_layout,
                            lapack_int m, lapack_int n, lapack_int l,
                            float* a, lapack_int lda,
                            float* b, lapack_int ldb,
                            float* t, lapack_int ldt );
```

#### LAPACKE_dtpqrt2
Double-precision real.
```c
lapack_int LAPACKE_dtpqrt2( int matrix_layout,
                            lapack_int m, lapack_int n, lapack_int l,
                            double* a, lapack_int lda,
                            double* b, lapack_int ldb,
                            double* t, lapack_int ldt );
```

#### LAPACKE_ctpqrt2
Single-precision complex.
```c
lapack_int LAPACKE_ctpqrt2( int matrix_layout,
                            lapack_int m, lapack_int n, lapack_int l,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* t, lapack_int ldt );
```

#### LAPACKE_ztpqrt2
Double-precision complex.
```c
lapack_int LAPACKE_ztpqrt2( int matrix_layout,
                            lapack_int m, lapack_int n, lapack_int l,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* t, lapack_int ldt );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of B |
| `n` | `lapack_int` | Number of columns of A and B |
| `l` | `lapack_int` | Number of rows of upper trapezoidal part of B |
| `a` | `<type>*` | Upper triangular matrix A (n x n), modified on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `<type>*` | Pentagonal matrix B (m x n), overwritten with V on exit |
| `ldb` | `lapack_int` | Leading dimension of B |
| `t` | `<type>*` | Output: upper triangular block reflectors T (n x n) |
| `ldt` | `lapack_int` | Leading dimension of T |

---

### tprfb - Apply T from tpqrt

Applies the block reflector obtained from tpqrt to a pair of matrices (A, B). This is the application routine corresponding to the tpqrt factorization, operating with the block reflector T and the pentagonal matrix V.

#### LAPACKE_stprfb
Single-precision real.
```c
lapack_int LAPACKE_stprfb( int matrix_layout, char side, char trans, char direct,
                           char storev, lapack_int m, lapack_int n,
                           lapack_int k, lapack_int l, const float* v,
                           lapack_int ldv, const float* t, lapack_int ldt,
                           float* a, lapack_int lda, float* b, lapack_int ldb );
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'`: apply from left; `'R'`: apply from right |
| `trans` | `char` | `'N'`: no transpose; `'T'`/`'C'`: transpose or conjugate transpose |
| `direct` | `char` | `'F'`: forward direction; `'B'`: backward direction |
| `storev` | `char` | `'C'`: column-wise storage of V; `'R'`: row-wise storage of V |
| `m` | `lapack_int` | Number of rows of A and B |
| `n` | `lapack_int` | Number of columns of A and B |
| `k` | `lapack_int` | Order of the block reflector T |
| `l` | `lapack_int` | Number of rows of pentagonal part of V |
| `v` | `const float*` | Pentagonal matrix V from tpqrt |
| `ldv` | `lapack_int` | Leading dimension of V |
| `t` | `const float*` | Block reflector T from tpqrt |
| `ldt` | `lapack_int` | Leading dimension of T |
| `a` | `float*` | Matrix A, modified on exit |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `float*` | Matrix B, modified on exit |
| `ldb` | `lapack_int` | Leading dimension of B |

#### LAPACKE_dtprfb
Double-precision real.
```c
lapack_int LAPACKE_dtprfb( int matrix_layout, char side, char trans, char direct,
                           char storev, lapack_int m, lapack_int n,
                           lapack_int k, lapack_int l, const double* v,
                           lapack_int ldv, const double* t, lapack_int ldt,
                           double* a, lapack_int lda, double* b, lapack_int ldb );
```

**Parameters:** Same as `LAPACKE_stprfb` with `double` replacing `float`.

#### LAPACKE_ctprfb
Single-precision complex.
```c
lapack_int LAPACKE_ctprfb( int matrix_layout, char side, char trans, char direct,
                           char storev, lapack_int m, lapack_int n,
                           lapack_int k, lapack_int l,
                           const lapack_complex_float* v, lapack_int ldv,
                           const lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb );
```

**Parameters:** Same structure with complex types. `trans` = `'N'` or `'C'`.

#### LAPACKE_ztprfb
Double-precision complex.
```c
lapack_int LAPACKE_ztprfb( int matrix_layout, char side, char trans, char direct,
                           char storev, lapack_int m, lapack_int n,
                           lapack_int k, lapack_int l,
                           const lapack_complex_double* v, lapack_int ldv,
                           const lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:** Same structure with `lapack_complex_double` types.
