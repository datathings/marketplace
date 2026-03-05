# LAPACKE Auxiliary API Reference

> Auxiliary and utility routines for matrix operations, norms, and transformations.
> Source: LAPACK v3.12.1 - `LAPACKE/include/lapacke.h`

## Table of Contents
- [LAPACKE Utility Macros and Functions](#lapacke-utility-macros-and-functions)
- [Machine Parameters (lamch)](#machine-parameters)
- [LAPACK Version (ilaver)](#lapack-version)
- [Matrix Norms](#matrix-norms)
  - [General Matrix Norm (lange)](#general-matrix-norm)
  - [Symmetric Matrix Norm (lansy)](#symmetric-matrix-norm)
  - [Hermitian Matrix Norm (lanhe)](#hermitian-matrix-norm)
  - [Triangular Matrix Norm (lantr)](#triangular-matrix-norm)
  - [General Band Matrix Norm (langb)](#general-band-matrix-norm)
- [Matrix Operations](#matrix-operations)
  - [Copy Matrix (lacpy)](#copy-matrix)
  - [Initialize Matrix (laset)](#initialize-matrix)
  - [Row Interchanges (laswp)](#row-interchanges)
  - [Scale Matrix (lascl)](#scale-matrix)
- [Precision Conversion](#precision-conversion)
  - [lag2d, lag2s, lag2c, lag2z](#precision-conversion-lag2x)
  - [Copy with Precision Change (lacp2)](#copy-with-precision-change)
- [Random Matrix Generation](#random-matrix-generation)
  - [Random Vector (larnv)](#random-vector)
  - [Random Matrix with Properties (latms)](#random-matrix-with-properties)
  - [Random General Matrix (lagge)](#random-general-matrix)
  - [Random Symmetric Matrix (lagsy)](#random-symmetric-matrix)
  - [Random Hermitian Matrix (laghe)](#random-hermitian-matrix)
- [Orthogonal/Unitary Matrix Generators](#orthogonalunitary-matrix-generators)
  - [From QR Factorization (orgqr/ungqr)](#from-qr-factorization)
  - [From LQ Factorization (orglq/unglq)](#from-lq-factorization)
  - [From QL Factorization (orgql/ungql)](#from-ql-factorization)
  - [From RQ Factorization (orgrq/ungrq)](#from-rq-factorization)
  - [From Hessenberg Reduction (orghr/unghr)](#from-hessenberg-reduction)
  - [From Tridiagonal Reduction (orgtr/ungtr)](#from-tridiagonal-reduction)
  - [From Bidiagonal Reduction (orgbr/ungbr)](#from-bidiagonal-reduction)
  - [From Packed Tridiagonal (opgtr/upgtr)](#from-packed-tridiagonal)
- [Orthogonal/Unitary Matrix Multipliers](#orthogonalunitary-matrix-multipliers)
  - [Multiply by Q from QR (ormqr/unmqr)](#multiply-by-q-from-qr)
  - [Multiply by Q from LQ (ormlq/unmlq)](#multiply-by-q-from-lq)
  - [Multiply by Q from QL (ormql/unmql)](#multiply-by-q-from-ql)
  - [Multiply by Q from RQ (ormrq/unmrq)](#multiply-by-q-from-rq)
  - [Multiply by Q from Hessenberg (ormhr/unmhr)](#multiply-by-q-from-hessenberg)
  - [Multiply by Q from Tridiagonal (ormtr/unmtr)](#multiply-by-q-from-tridiagonal)
  - [Multiply by Q from Bidiagonal (ormbr/unmbr)](#multiply-by-q-from-bidiagonal)
  - [Multiply by Q from RZ (ormrz/unmrz)](#multiply-by-q-from-rz)
  - [Multiply by Q from Packed Tridiagonal (opmtr/upmtr)](#multiply-by-q-from-packed-tridiagonal)
- [Generalized QR/RQ Factorizations](#generalized-qrrq-factorizations)
- [Upper Trapezoidal to Triangular (tzrzf)](#upper-trapezoidal-to-triangular)
- [Householder Reflectors](#householder-reflectors)
- [Miscellaneous](#miscellaneous)
  - [Reciprocal Condition Numbers (disna)](#reciprocal-condition-numbers)
  - [Safe Pythagorean Distance (lapy2, lapy3)](#safe-pythagorean-distance)
  - [Multiply Real by Complex (larcm, lacrm)](#multiply-real-by-complex)
  - [Conjugate Vector (lacgv)](#conjugate-vector)
  - [Estimate 1-Norm (lacn2)](#estimate-1-norm)
  - [Generate Plane Rotation (lartgp, lartgs)](#generate-plane-rotation)
  - [Permute Rows/Columns (lapmr, lapmt)](#permute-rowscolumns)
  - [Sum of Squares (lassq)](#sum-of-squares)
  - [Sort (lasrt)](#sort)
  - [RFP Format Operations (sfrk, hfrk, tfsm, tftri)](#rfp-format-operations)
  - [Symmetric/Hermitian Conversions and Swaps](#symmetrichermitian-conversions-and-swaps)
  - [Complex Symmetric Rank-1 Update (syr)](#complex-symmetric-rank-1-update)

## Common Conventions

### Layout Parameter
All LAPACKE functions taking `matrix_layout` accept:
- `LAPACK_ROW_MAJOR` (101) -- row-major storage
- `LAPACK_COL_MAJOR` (102) -- column-major storage

### Precision Prefixes

| Prefix | Type |
|--------|------|
| `s` | `float` (single real) |
| `d` | `double` (double real) |
| `c` | `lapack_complex_float` (single complex) |
| `z` | `lapack_complex_double` (double complex) |

### Norm Parameter Convention

Many norm-computing functions accept a `char norm` parameter:

| Value | Norm Computed |
|-------|--------------|
| `'M'` or `'m'` | max(abs(A(i,j))) -- maximum absolute element |
| `'1'` or `'O'` or `'o'` | one-norm -- maximum column sum |
| `'I'` or `'i'` | infinity-norm -- maximum row sum |
| `'F'` or `'f'` or `'E'` or `'e'` | Frobenius norm -- sqrt(sum of squares) |

---

## LAPACKE Utility Macros and Functions

### LAPACKE_malloc / LAPACKE_free

Memory management macros that default to standard `malloc`/`free`. Can be overridden before including `lapacke.h`.

```c
#define LAPACKE_malloc( size ) malloc( size )
#define LAPACKE_free( p )      free( p )
```

### Layout Constants

```c
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
```

### Error Codes

```c
#define LAPACK_WORK_MEMORY_ERROR       -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR  -1011
```

### Complex Number Constructors

```c
lapack_complex_float  lapack_make_complex_float( float re, float im );
lapack_complex_double lapack_make_complex_double( double re, double im );
```

---

## Machine Parameters

### LAPACKE_slamch / LAPACKE_dlamch

Determine machine parameters for floating-point arithmetic.

```c
float  LAPACKE_slamch( char cmach );
double LAPACKE_dlamch( char cmach );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `cmach` | `char` | Specifies the machine parameter to return |

**Values for `cmach`:**

| Value | Parameter | Description |
|-------|-----------|-------------|
| `'E'` | epsilon | Relative machine precision |
| `'S'` | sfmin | Safe minimum, such that 1/sfmin does not overflow |
| `'B'` | base | Base of the machine |
| `'P'` | prec | epsilon * base |
| `'N'` | t | Number of digits in the mantissa |
| `'R'` | rnd | 1.0 when rounding occurs in addition, 0.0 otherwise |
| `'M'` | emin | Minimum exponent before gradual underflow |
| `'U'` | rmin | Underflow threshold -- base^(emin-1) |
| `'L'` | emax | Largest exponent before overflow |
| `'O'` | rmax | Overflow threshold -- (base^emax)*(1-eps) |

**Returns:** The requested machine parameter as `float` or `double`.

---

## LAPACK Version

### LAPACKE_ilaver

Return the LAPACK version number.

```c
void LAPACKE_ilaver( lapack_int* vers_major,
                     lapack_int* vers_minor,
                     lapack_int* vers_patch );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `vers_major` | `lapack_int*` | [out] Major version number |
| `vers_minor` | `lapack_int*` | [out] Minor version number |
| `vers_patch` | `lapack_int*` | [out] Patch version number |

---

## Matrix Norms

### General Matrix Norm

**lange** -- Compute the norm of a general m-by-n matrix.

```c
float  LAPACKE_slange( int matrix_layout, char norm, lapack_int m,
                       lapack_int n, const float* a, lapack_int lda );
double LAPACKE_dlange( int matrix_layout, char norm, lapack_int m,
                       lapack_int n, const double* a, lapack_int lda );
float  LAPACKE_clange( int matrix_layout, char norm, lapack_int m,
                       lapack_int n, const lapack_complex_float* a,
                       lapack_int lda );
double LAPACKE_zlange( int matrix_layout, char norm, lapack_int m,
                       lapack_int n, const lapack_complex_double* a,
                       lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | Norm type: `'M'`, `'1'`/`'O'`, `'I'`, or `'F'`/`'E'` |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `const T*` | Input matrix A (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |

**Returns:** The computed norm value.

---

### Symmetric Matrix Norm

**lansy** -- Compute the norm of a symmetric matrix.

```c
float  LAPACKE_slansy( int matrix_layout, char norm, char uplo, lapack_int n,
                       const float* a, lapack_int lda );
double LAPACKE_dlansy( int matrix_layout, char norm, char uplo, lapack_int n,
                       const double* a, lapack_int lda );
float  LAPACKE_clansy( int matrix_layout, char norm, char uplo, lapack_int n,
                       const lapack_complex_float* a, lapack_int lda );
double LAPACKE_zlansy( int matrix_layout, char norm, char uplo, lapack_int n,
                       const lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | Norm type: `'M'`, `'1'`/`'O'`, `'I'`, or `'F'`/`'E'` |
| `uplo` | `char` | `'U'` = upper triangle stored, `'L'` = lower triangle stored |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `const T*` | Input symmetric matrix A (n x n) |
| `lda` | `lapack_int` | Leading dimension of A |

**Returns:** The computed norm value.

---

### Hermitian Matrix Norm

**lanhe** -- Compute the norm of a Hermitian matrix (complex only).

```c
float  LAPACKE_clanhe( int matrix_layout, char norm, char uplo, lapack_int n,
                       const lapack_complex_float* a, lapack_int lda );
double LAPACKE_zlanhe( int matrix_layout, char norm, char uplo, lapack_int n,
                       const lapack_complex_double* a, lapack_int lda );
```

**Parameters:** Same as `lansy` but for Hermitian matrices. The diagonal elements are assumed to be real.

**Returns:** The computed norm value.

---

### Triangular Matrix Norm

**lantr** -- Compute the norm of a triangular matrix.

```c
float  LAPACKE_slantr( int matrix_layout, char norm, char uplo, char diag,
                       lapack_int m, lapack_int n, const float* a,
                       lapack_int lda );
double LAPACKE_dlantr( int matrix_layout, char norm, char uplo, char diag,
                       lapack_int m, lapack_int n, const double* a,
                       lapack_int lda );
float  LAPACKE_clantr( int matrix_layout, char norm, char uplo, char diag,
                       lapack_int m, lapack_int n, const lapack_complex_float* a,
                       lapack_int lda );
double LAPACKE_zlantr( int matrix_layout, char norm, char uplo, char diag,
                       lapack_int m, lapack_int n, const lapack_complex_double* a,
                       lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | Norm type: `'M'`, `'1'`/`'O'`, `'I'`, or `'F'`/`'E'` |
| `uplo` | `char` | `'U'` = upper triangular, `'L'` = lower triangular |
| `diag` | `char` | `'N'` = non-unit diagonal, `'U'` = unit diagonal (diagonal elements not referenced) |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `const T*` | Input triangular matrix A (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |

**Returns:** The computed norm value.

---

### General Band Matrix Norm

**langb** -- Compute the norm of a general band matrix stored in band format.

```c
float  LAPACKE_slangb( int matrix_layout, char norm, lapack_int n,
                       lapack_int kl, lapack_int ku, const float* ab,
                       lapack_int ldab );
double LAPACKE_dlangb( int matrix_layout, char norm, lapack_int n,
                       lapack_int kl, lapack_int ku, const double* ab,
                       lapack_int ldab );
float  LAPACKE_clangb( int matrix_layout, char norm, lapack_int n,
                       lapack_int kl, lapack_int ku,
                       const lapack_complex_float* ab, lapack_int ldab );
double LAPACKE_zlangb( int matrix_layout, char norm, lapack_int n,
                       lapack_int kl, lapack_int ku,
                       const lapack_complex_double* ab, lapack_int ldab );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `norm` | `char` | Norm type: `'M'`, `'1'`/`'O'`, `'I'`, or `'F'`/`'E'` |
| `n` | `lapack_int` | Order of matrix A |
| `kl` | `lapack_int` | Number of sub-diagonals |
| `ku` | `lapack_int` | Number of super-diagonals |
| `ab` | `const T*` | Band matrix A in band storage (kl+ku+1 rows, n columns) |
| `ldab` | `lapack_int` | Leading dimension of ab |

**Returns:** The computed norm value.

---

## Matrix Operations

### Copy Matrix

**lacpy** -- Copy all or part of a matrix A to another matrix B.

```c
lapack_int LAPACKE_slacpy( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const float* a, lapack_int lda,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dlacpy( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const double* a, lapack_int lda,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_clacpy( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zlacpy( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle only, `'L'` = lower triangle only, `'A'`/other = full matrix |
| `m` | `lapack_int` | Number of rows of A and B |
| `n` | `lapack_int` | Number of columns of A and B |
| `a` | `const T*` | [in] Source matrix A (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `T*` | [out] Destination matrix B (m x n) |
| `ldb` | `lapack_int` | Leading dimension of B |

---

### Initialize Matrix

**laset** -- Initialize a matrix with values: diagonal elements set to `beta`, off-diagonal elements set to `alpha`.

```c
lapack_int LAPACKE_slaset( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, float alpha, float beta, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dlaset( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, double alpha, double beta, double* a,
                           lapack_int lda );
lapack_int LAPACKE_claset( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, lapack_complex_float alpha,
                           lapack_complex_float beta, lapack_complex_float* a,
                           lapack_int lda );
lapack_int LAPACKE_zlaset( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, lapack_complex_double alpha,
                           lapack_complex_double beta, lapack_complex_double* a,
                           lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle, `'L'` = lower triangle, other = full matrix |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `alpha` | `T` | Value for off-diagonal elements |
| `beta` | `T` | Value for diagonal elements |
| `a` | `T*` | [out] Matrix A (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |

**Note:** To create an identity matrix, use `alpha=0, beta=1`.

---

### Row Interchanges

**laswp** -- Perform a series of row interchanges on a matrix.

```c
lapack_int LAPACKE_slaswp( int matrix_layout, lapack_int n, float* a,
                           lapack_int lda, lapack_int k1, lapack_int k2,
                           const lapack_int* ipiv, lapack_int incx );
lapack_int LAPACKE_dlaswp( int matrix_layout, lapack_int n, double* a,
                           lapack_int lda, lapack_int k1, lapack_int k2,
                           const lapack_int* ipiv, lapack_int incx );
lapack_int LAPACKE_claswp( int matrix_layout, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                           lapack_int incx );
lapack_int LAPACKE_zlaswp( int matrix_layout, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                           lapack_int incx );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Number of columns of matrix A |
| `a` | `T*` | [in/out] Matrix A to permute |
| `lda` | `lapack_int` | Leading dimension of A |
| `k1` | `lapack_int` | First element of ipiv for row interchange (1-indexed) |
| `k2` | `lapack_int` | Last element of ipiv for row interchange (1-indexed) |
| `ipiv` | `const lapack_int*` | Pivot indices; row i is interchanged with row ipiv(i) |
| `incx` | `lapack_int` | Increment between successive values of ipiv. If negative, pivots applied in reverse order |

---

### Scale Matrix

**lascl** -- Multiply a matrix by the scalar cto/cfrom, with careful handling of overflow/underflow.

```c
lapack_int LAPACKE_slascl( int matrix_layout, char type, lapack_int kl,
                           lapack_int ku, float cfrom, float cto,
                           lapack_int m, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dlascl( int matrix_layout, char type, lapack_int kl,
                           lapack_int ku, double cfrom, double cto,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_clascl( int matrix_layout, char type, lapack_int kl,
                           lapack_int ku, float cfrom, float cto,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda );
lapack_int LAPACKE_zlascl( int matrix_layout, char type, lapack_int kl,
                           lapack_int ku, double cfrom, double cto,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `type` | `char` | Matrix type: `'G'`=general, `'L'`=lower triangular, `'U'`=upper triangular, `'H'`=upper Hessenberg, `'B'`=lower half of symmetric band, `'Q'`=upper half of symmetric band, `'Z'`=band matrix |
| `kl` | `lapack_int` | Lower bandwidth (used only for band types) |
| `ku` | `lapack_int` | Upper bandwidth (used only for band types) |
| `cfrom` | `float`/`double` | Scale factor denominator (must not be zero) |
| `cto` | `float`/`double` | Scale factor numerator |
| `m` | `lapack_int` | Number of rows of A |
| `n` | `lapack_int` | Number of columns of A |
| `a` | `T*` | [in/out] Matrix to be scaled by cto/cfrom |
| `lda` | `lapack_int` | Leading dimension of A |

---

## Precision Conversion

### Precision Conversion (lag2x)

Convert matrices between single and double precision (real and complex).

```c
/* Double complex -> Single complex */
lapack_int LAPACKE_zlag2c( int matrix_layout, lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           lapack_complex_float* sa, lapack_int ldsa );

/* Single real -> Double real */
lapack_int LAPACKE_slag2d( int matrix_layout, lapack_int m, lapack_int n,
                           const float* sa, lapack_int ldsa, double* a,
                           lapack_int lda );

/* Double real -> Single real */
lapack_int LAPACKE_dlag2s( int matrix_layout, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda, float* sa,
                           lapack_int ldsa );

/* Single complex -> Double complex */
lapack_int LAPACKE_clag2z( int matrix_layout, lapack_int m, lapack_int n,
                           const lapack_complex_float* sa, lapack_int ldsa,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters (slag2d as representative):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `sa` | `const float*` | [in] Source matrix in lower precision |
| `ldsa` | `lapack_int` | Leading dimension of source |
| `a` | `double*` | [out] Destination matrix in higher precision |
| `lda` | `lapack_int` | Leading dimension of destination |

**Note:** When converting to lower precision (dlag2s, zlag2c), returns info > 0 if any element would overflow in the target precision.

---

### Copy with Precision Change

**lacp2** -- Copy a real matrix into a complex matrix (real-to-complex promotion).

```c
lapack_int LAPACKE_clacp2( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zlacp2( int matrix_layout, char uplo, lapack_int m,
                           lapack_int n, const double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle, `'L'` = lower triangle, other = full matrix |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `a` | `const float*`/`const double*` | [in] Real source matrix |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `lapack_complex_float*`/`lapack_complex_double*` | [out] Complex destination matrix (imaginary parts set to zero) |
| `ldb` | `lapack_int` | Leading dimension of B |

---

## Random Matrix Generation

### Random Vector

**larnv** -- Generate a vector of random numbers from a uniform or normal distribution.

```c
lapack_int LAPACKE_slarnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                           float* x );
lapack_int LAPACKE_dlarnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                           double* x );
lapack_int LAPACKE_clarnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                           lapack_complex_float* x );
lapack_int LAPACKE_zlarnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                           lapack_complex_double* x );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `idist` | `lapack_int` | Distribution: 1=uniform(0,1), 2=uniform(-1,1), 3=normal(0,1) |
| `iseed` | `lapack_int*` | [in/out] Array of 4 seed integers (0 <= iseed(i) <= 4095, iseed(4) must be odd) |
| `n` | `lapack_int` | Number of random numbers to generate |
| `x` | `T*` | [out] Generated random vector of length n |

---

### Random Matrix with Properties

**latms** -- Generate a random matrix with specified singular values, eigenvalues, bandwidth, and packing.

```c
lapack_int LAPACKE_slatms( int matrix_layout, lapack_int m, lapack_int n,
                           char dist, lapack_int* iseed, char sym, float* d,
                           lapack_int mode, float cond, float dmax,
                           lapack_int kl, lapack_int ku, char pack, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dlatms( int matrix_layout, lapack_int m, lapack_int n,
                           char dist, lapack_int* iseed, char sym, double* d,
                           lapack_int mode, double cond, double dmax,
                           lapack_int kl, lapack_int ku, char pack, double* a,
                           lapack_int lda );
lapack_int LAPACKE_clatms( int matrix_layout, lapack_int m, lapack_int n,
                           char dist, lapack_int* iseed, char sym, float* d,
                           lapack_int mode, float cond, float dmax,
                           lapack_int kl, lapack_int ku, char pack,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zlatms( int matrix_layout, lapack_int m, lapack_int n,
                           char dist, lapack_int* iseed, char sym, double* d,
                           lapack_int mode, double cond, double dmax,
                           lapack_int kl, lapack_int ku, char pack,
                           lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `dist` | `char` | Distribution of underlying random numbers: `'U'`=uniform(0,1), `'S'`=symmetric uniform(-1,1), `'N'`=normal(0,1) |
| `iseed` | `lapack_int*` | [in/out] Random seed array (length 4) |
| `sym` | `char` | Matrix type: `'N'`=nonsymmetric, `'S'`=symmetric, `'H'`=Hermitian, `'P'`=positive semi-definite |
| `d` | `T_real*` | [in/out] Array of singular/eigenvalues |
| `mode` | `lapack_int` | Mode for computing d values (0-6) |
| `cond` | `T_real` | Condition number (used with modes 1-6) |
| `dmax` | `T_real` | Maximum absolute value of d entries |
| `kl` | `lapack_int` | Lower bandwidth |
| `ku` | `lapack_int` | Upper bandwidth |
| `pack` | `char` | Packing type: `'N'`=no packing, `'U'`/`'L'`=upper/lower band, `'C'`/`'R'`/`'B'`/`'Q'`/`'Z'`=various band formats |
| `a` | `T*` | [out] Generated matrix |
| `lda` | `lapack_int` | Leading dimension of A |

---

### Random General Matrix

**lagge** -- Generate a random general m-by-n matrix with specified singular values.

```c
lapack_int LAPACKE_slagge( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const float* d,
                           float* a, lapack_int lda, lapack_int* iseed );
lapack_int LAPACKE_dlagge( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* d,
                           double* a, lapack_int lda, lapack_int* iseed );
lapack_int LAPACKE_clagge( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const float* d,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* iseed );
lapack_int LAPACKE_zlagge( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* d,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* iseed );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `kl` | `lapack_int` | Number of nonzero sub-diagonals |
| `ku` | `lapack_int` | Number of nonzero super-diagonals |
| `d` | `const T_real*` | Array of singular values (length min(m,n)) |
| `a` | `T*` | [out] Generated matrix (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |
| `iseed` | `lapack_int*` | [in/out] Random seed array (length 4) |

---

### Random Symmetric Matrix

**lagsy** -- Generate a random symmetric matrix with specified eigenvalues.

```c
lapack_int LAPACKE_slagsy( int matrix_layout, lapack_int n, lapack_int k,
                           const float* d, float* a, lapack_int lda,
                           lapack_int* iseed );
lapack_int LAPACKE_dlagsy( int matrix_layout, lapack_int n, lapack_int k,
                           const double* d, double* a, lapack_int lda,
                           lapack_int* iseed );
lapack_int LAPACKE_clagsy( int matrix_layout, lapack_int n, lapack_int k,
                           const float* d, lapack_complex_float* a,
                           lapack_int lda, lapack_int* iseed );
lapack_int LAPACKE_zlagsy( int matrix_layout, lapack_int n, lapack_int k,
                           const double* d, lapack_complex_double* a,
                           lapack_int lda, lapack_int* iseed );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix A |
| `k` | `lapack_int` | Number of nonzero sub-diagonals in the generated band matrix used to reduce to tridiagonal form |
| `d` | `const T_real*` | Array of eigenvalues (length n) |
| `a` | `T*` | [out] Generated symmetric matrix (n x n) |
| `lda` | `lapack_int` | Leading dimension of A |
| `iseed` | `lapack_int*` | [in/out] Random seed array (length 4) |

---

### Random Hermitian Matrix

**laghe** -- Generate a random Hermitian matrix with specified eigenvalues (complex only).

```c
lapack_int LAPACKE_claghe( int matrix_layout, lapack_int n, lapack_int k,
                           const float* d, lapack_complex_float* a,
                           lapack_int lda, lapack_int* iseed );
lapack_int LAPACKE_zlaghe( int matrix_layout, lapack_int n, lapack_int k,
                           const double* d, lapack_complex_double* a,
                           lapack_int lda, lapack_int* iseed );
```

**Parameters:** Same as `lagsy` but generates a Hermitian matrix (with real diagonal and conjugate-symmetric off-diagonal elements).

---

## Orthogonal/Unitary Matrix Generators

These routines generate explicit orthogonal (real) or unitary (complex) matrices Q from the implicit representation stored by various factorization routines. The real routines use the `org` prefix; the complex routines use the `ung` prefix.

### From QR Factorization

**orgqr/ungqr** -- Generate the orthogonal/unitary matrix Q from a QR factorization as computed by `geqrf`.

```c
/* Real */
lapack_int LAPACKE_sorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

/* Complex */
lapack_int LAPACKE_cungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungqr( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of Q (m >= n >= k >= 0) |
| `n` | `lapack_int` | Number of columns of Q |
| `k` | `lapack_int` | Number of elementary reflectors (from geqrf) |
| `a` | `T*` | [in/out] On entry, reflectors from geqrf. On exit, the m-by-n matrix Q |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors (length k, from geqrf) |

---

### From LQ Factorization

**orglq/unglq** -- Generate Q from an LQ factorization as computed by `gelqf`.

```c
/* Real */
lapack_int LAPACKE_sorglq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorglq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

/* Complex */
lapack_int LAPACKE_cunglq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zunglq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:** Same as `orgqr`/`ungqr` (with n >= m >= k >= 0 constraint for LQ).

---

### From QL Factorization

**orgql/ungql** -- Generate Q from a QL factorization as computed by `geqlf`.

```c
/* Real */
lapack_int LAPACKE_sorgql( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgql( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

/* Complex */
lapack_int LAPACKE_cungql( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungql( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:** Same as `orgqr`/`ungqr`.

---

### From RQ Factorization

**orgrq/ungrq** -- Generate Q from an RQ factorization as computed by `gerqf`.

```c
/* Real */
lapack_int LAPACKE_sorgrq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgrq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

/* Complex */
lapack_int LAPACKE_cungrq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungrq( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:** Same as `orgqr`/`ungqr` (with n >= m >= k >= 0 constraint for RQ).

---

### From Hessenberg Reduction

**orghr/unghr** -- Generate Q from a Hessenberg reduction as computed by `gehrd`.

```c
/* Real */
lapack_int LAPACKE_sorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           const double* tau );

/* Complex */
lapack_int LAPACKE_cunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zunghr( int matrix_layout, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Order of matrix Q |
| `ilo` | `lapack_int` | Determined by balancing (from gebal); set to 1 if not balanced |
| `ihi` | `lapack_int` | Determined by balancing (from gebal); set to n if not balanced |
| `a` | `T*` | [in/out] On entry, reflectors from gehrd. On exit, matrix Q |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors (length n-1, from gehrd) |

---

### From Tridiagonal Reduction

**orgtr/ungtr** -- Generate Q from a tridiagonal reduction as computed by `sytrd`/`hetrd`.

```c
/* Real */
lapack_int LAPACKE_sorgtr( int matrix_layout, char uplo, lapack_int n,
                           float* a, lapack_int lda, const float* tau );
lapack_int LAPACKE_dorgtr( int matrix_layout, char uplo, lapack_int n,
                           double* a, lapack_int lda, const double* tau );

/* Complex */
lapack_int LAPACKE_cungtr( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau );
lapack_int LAPACKE_zungtr( int matrix_layout, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` -- must match what was used in sytrd/hetrd |
| `n` | `lapack_int` | Order of matrix Q |
| `a` | `T*` | [in/out] On entry, reflectors from sytrd/hetrd. On exit, matrix Q |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors (length n-1) |

---

### From Bidiagonal Reduction

**orgbr/ungbr** -- Generate the orthogonal/unitary matrix Q or P^T from a bidiagonal reduction as computed by `gebrd`.

```c
/* Real */
lapack_int LAPACKE_sorgbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgbr( int matrix_layout, char vect, lapack_int m,
                           lapack_int n, lapack_int k, double* a,
                           lapack_int lda, const double* tau );

/* Complex */
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
| `vect` | `char` | `'Q'` = generate Q, `'P'` = generate P^T (or P^H for complex) |
| `m` | `lapack_int` | Number of rows of the matrix to generate |
| `n` | `lapack_int` | Number of columns of the matrix to generate |
| `k` | `lapack_int` | If vect='Q', the number of columns in the original matrix. If vect='P', the number of rows |
| `a` | `T*` | [in/out] On entry, reflectors from gebrd. On exit, the matrix Q or P^T |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors (tauq for Q, taup for P^T) |

---

### From Packed Tridiagonal

**opgtr/upgtr** -- Generate Q from a tridiagonal reduction of a packed symmetric/Hermitian matrix as computed by `sptrd`/`hptrd`.

```c
/* Real */
lapack_int LAPACKE_sopgtr( int matrix_layout, char uplo, lapack_int n,
                           const float* ap, const float* tau, float* q,
                           lapack_int ldq );
lapack_int LAPACKE_dopgtr( int matrix_layout, char uplo, lapack_int n,
                           const double* ap, const double* tau, double* q,
                           lapack_int ldq );

/* Complex */
lapack_int LAPACKE_cupgtr( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* tau,
                           lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zupgtr( int matrix_layout, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* tau,
                           lapack_complex_double* q, lapack_int ldq );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` -- must match what was used in sptrd/hptrd |
| `n` | `lapack_int` | Order of matrix Q |
| `ap` | `const T*` | Packed matrix from sptrd/hptrd (length n*(n+1)/2) |
| `tau` | `const T*` | Scalar factors of reflectors (length n-1) |
| `q` | `T*` | [out] The n-by-n orthogonal/unitary matrix Q |
| `ldq` | `lapack_int` | Leading dimension of Q |

---

## Orthogonal/Unitary Matrix Multipliers

These routines multiply a general matrix C by the orthogonal/unitary matrix Q obtained from various factorizations, without forming Q explicitly. The operation is: **C = op(Q) * C** or **C = C * op(Q)**, where `op(Q) = Q` or `Q^T` (real) or `Q^H` (complex). Real routines use the `orm`/`opm` prefix; complex routines use the `unm`/`upm` prefix.

### Multiply by Q from QR

**ormqr/unmqr** -- Multiply by Q from a QR factorization (from `geqrf`).

```c
/* Real */
lapack_int LAPACKE_sormqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmqr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'` = apply Q from left (Q*C), `'R'` = apply Q from right (C*Q) |
| `trans` | `char` | `'N'` = apply Q, `'T'` = apply Q^T (real), `'C'` = apply Q^H (complex) |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | Number of elementary reflectors |
| `a` | `const T*` | Reflectors from geqrf |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors (length k) |
| `c` | `T*` | [in/out] Matrix C (m x n); overwritten with op(Q)*C or C*op(Q) |
| `ldc` | `lapack_int` | Leading dimension of C |

---

### Multiply by Q from LQ

**ormlq/unmlq** -- Multiply by Q from an LQ factorization (from `gelqf`).

```c
/* Real */
lapack_int LAPACKE_sormlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmlq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as `ormqr`/`unmqr`.

---

### Multiply by Q from QL

**ormql/unmql** -- Multiply by Q from a QL factorization (from `geqlf`).

```c
/* Real */
lapack_int LAPACKE_sormql( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormql( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmql( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmql( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as `ormqr`/`unmqr`.

---

### Multiply by Q from RQ

**ormrq/unmrq** -- Multiply by Q from an RQ factorization (from `gerqf`).

```c
/* Real */
lapack_int LAPACKE_sormrq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormrq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmrq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmrq( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as `ormqr`/`unmqr`.

---

### Multiply by Q from Hessenberg

**ormhr/unmhr** -- Multiply by Q from a Hessenberg reduction (from `gehrd`).

```c
/* Real */
lapack_int LAPACKE_sormhr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const float* a, lapack_int lda,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dormhr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const double* a, lapack_int lda,
                           const double* tau, double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmhr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmhr( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'` = apply Q from left, `'R'` = apply Q from right |
| `trans` | `char` | `'N'` = Q, `'T'` = Q^T (real), `'C'` = Q^H (complex) |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `ilo` | `lapack_int` | Index range from balancing (1 if not balanced) |
| `ihi` | `lapack_int` | Index range from balancing (m or n if not balanced) |
| `a` | `const T*` | Reflectors from gehrd |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors of reflectors |
| `c` | `T*` | [in/out] Matrix C |
| `ldc` | `lapack_int` | Leading dimension of C |

---

### Multiply by Q from Tridiagonal

**ormtr/unmtr** -- Multiply by Q from a tridiagonal reduction (from `sytrd`/`hetrd`).

```c
/* Real */
lapack_int LAPACKE_sormtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const float* a,
                           lapack_int lda, const float* tau, float* c,
                           lapack_int ldc );
lapack_int LAPACKE_dormtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const double* a,
                           lapack_int lda, const double* tau, double* c,
                           lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as `ormhr`/`unmhr` but with `uplo` replacing `ilo`/`ihi`:

| Name | Type | Description |
|------|------|-------------|
| `uplo` | `char` | `'U'` or `'L'` -- must match the value used in sytrd/hetrd |

---

### Multiply by Q from Bidiagonal

**ormbr/unmbr** -- Multiply by Q or P^T from a bidiagonal reduction (from `gebrd`).

```c
/* Real */
lapack_int LAPACKE_sormbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormbr( int matrix_layout, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

/* Complex */
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
| `vect` | `char` | `'Q'` = apply Q (using tauq from gebrd), `'P'` = apply P^T/P^H (using taup from gebrd) |
| `side` | `char` | `'L'` = from left, `'R'` = from right |
| `trans` | `char` | `'N'` = no transpose, `'T'` = transpose, `'C'` = conjugate transpose |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | Number of columns (vect='Q') or rows (vect='P') of the original matrix reduced by gebrd |
| `a` | `const T*` | Reflectors from gebrd |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `const T*` | Scalar factors (tauq for Q, taup for P) |
| `c` | `T*` | [in/out] Matrix C |
| `ldc` | `lapack_int` | Leading dimension of C |

---

### Multiply by Q from RZ

**ormrz/unmrz** -- Multiply by Q from an RZ factorization (from `tzrzf`).

```c
/* Real */
lapack_int LAPACKE_sormrz( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const float* a, lapack_int lda,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dormrz( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const double* a, lapack_int lda,
                           const double* tau, double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cunmrz( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmrz( int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:** Same as `ormqr`/`unmqr` with the addition of:

| Name | Type | Description |
|------|------|-------------|
| `l` | `lapack_int` | Number of columns of the trapezoidal part (from tzrzf) |

---

### Multiply by Q from Packed Tridiagonal

**opmtr/upmtr** -- Multiply by Q from a tridiagonal reduction of a packed symmetric/Hermitian matrix (from `sptrd`/`hptrd`).

```c
/* Real */
lapack_int LAPACKE_sopmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const float* ap,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dopmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const double* ap,
                           const double* tau, double* c, lapack_int ldc );

/* Complex */
lapack_int LAPACKE_cupmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zupmtr( int matrix_layout, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'` = from left, `'R'` = from right |
| `uplo` | `char` | `'U'` or `'L'` -- must match sptrd/hptrd |
| `trans` | `char` | `'N'` = Q, `'T'` = Q^T (real), `'C'` = Q^H (complex) |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `ap` | `const T*` | Packed matrix from sptrd/hptrd |
| `tau` | `const T*` | Scalar factors of reflectors |
| `c` | `T*` | [in/out] Matrix C |
| `ldc` | `lapack_int` | Leading dimension of C |

---

## Generalized QR/RQ Factorizations

### ggqrf

**ggqrf** -- Compute the generalized QR factorization of an n-by-m matrix A and an n-by-p matrix B: A = Q*R, B = Q*T*Z.

```c
lapack_int LAPACKE_sggqrf( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, float* a, lapack_int lda, float* taua,
                           float* b, lapack_int ldb, float* taub );
lapack_int LAPACKE_dggqrf( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, double* a, lapack_int lda,
                           double* taua, double* b, lapack_int ldb,
                           double* taub );
lapack_int LAPACKE_cggqrf( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* taua,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* taub );
lapack_int LAPACKE_zggqrf( int matrix_layout, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* taua,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* taub );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `n` | `lapack_int` | Number of rows of A and B |
| `m` | `lapack_int` | Number of columns of A |
| `p` | `lapack_int` | Number of columns of B |
| `a` | `T*` | [in/out] On exit, upper triangle contains R |
| `lda` | `lapack_int` | Leading dimension of A |
| `taua` | `T*` | [out] Scalar factors of reflectors for Q (length min(n,m)) |
| `b` | `T*` | [in/out] On exit, upper triangle contains T |
| `ldb` | `lapack_int` | Leading dimension of B |
| `taub` | `T*` | [out] Scalar factors of reflectors for Z (length min(n,p)) |

---

### ggrqf

**ggrqf** -- Compute the generalized RQ factorization of an m-by-n matrix A and a p-by-n matrix B: A = R*Q, B = Z*T*Q.

```c
lapack_int LAPACKE_sggrqf( int matrix_layout, lapack_int m, lapack_int p,
                           lapack_int n, float* a, lapack_int lda, float* taua,
                           float* b, lapack_int ldb, float* taub );
lapack_int LAPACKE_dggrqf( int matrix_layout, lapack_int m, lapack_int p,
                           lapack_int n, double* a, lapack_int lda,
                           double* taua, double* b, lapack_int ldb,
                           double* taub );
lapack_int LAPACKE_cggrqf( int matrix_layout, lapack_int m, lapack_int p,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* taua,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* taub );
lapack_int LAPACKE_zggrqf( int matrix_layout, lapack_int m, lapack_int p,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* taua,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* taub );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A |
| `p` | `lapack_int` | Number of rows of B |
| `n` | `lapack_int` | Number of columns of A and B |
| `a` | `T*` | [in/out] On exit, upper triangle contains R |
| `lda` | `lapack_int` | Leading dimension of A |
| `taua` | `T*` | [out] Scalar factors for Q of A (length min(m,n)) |
| `b` | `T*` | [in/out] On exit, upper triangle contains T |
| `ldb` | `lapack_int` | Leading dimension of B |
| `taub` | `T*` | [out] Scalar factors for Z of B (length min(p,n)) |

---

## Upper Trapezoidal to Triangular

**tzrzf** -- Reduce an upper trapezoidal matrix to upper triangular form by orthogonal/unitary transformations.

Factors the m-by-n (m <= n) upper trapezoidal matrix A = [ R 0 ] * Z, where R is upper triangular and Z is orthogonal/unitary.

```c
lapack_int LAPACKE_stzrzf( int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dtzrzf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_ctzrzf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_ztzrzf( int matrix_layout, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows (m <= n) |
| `n` | `lapack_int` | Number of columns |
| `a` | `T*` | [in/out] On entry, upper trapezoidal matrix. On exit, R in upper triangle |
| `lda` | `lapack_int` | Leading dimension of A |
| `tau` | `T*` | [out] Scalar factors of reflectors (length m) |

---

## Householder Reflectors

### larfg -- Generate an Elementary Reflector

Generate a Householder reflector H such that H * [alpha; x] = [beta; 0], where H = I - tau * v * v^H.

```c
lapack_int LAPACKE_slarfg( lapack_int n, float* alpha, float* x,
                           lapack_int incx, float* tau );
lapack_int LAPACKE_dlarfg( lapack_int n, double* alpha, double* x,
                           lapack_int incx, double* tau );
lapack_int LAPACKE_clarfg( lapack_int n, lapack_complex_float* alpha,
                           lapack_complex_float* x, lapack_int incx,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zlarfg( lapack_int n, lapack_complex_double* alpha,
                           lapack_complex_double* x, lapack_int incx,
                           lapack_complex_double* tau );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of the reflector (length of v including the 1) |
| `alpha` | `T*` | [in/out] On entry, the first element. On exit, beta |
| `x` | `T*` | [in/out] On entry, elements 2:n. On exit, elements of v (v(1)=1 is implicit) |
| `incx` | `lapack_int` | Stride between elements of x |
| `tau` | `T*` | [out] Scalar factor tau |

---

### larft -- Form Block Reflector

Form the triangular factor T of a block reflector H = I - V * T * V^H.

```c
lapack_int LAPACKE_slarft( int matrix_layout, char direct, char storev,
                           lapack_int n, lapack_int k, const float* v,
                           lapack_int ldv, const float* tau, float* t,
                           lapack_int ldt );
lapack_int LAPACKE_dlarft( int matrix_layout, char direct, char storev,
                           lapack_int n, lapack_int k, const double* v,
                           lapack_int ldv, const double* tau, double* t,
                           lapack_int ldt );
lapack_int LAPACKE_clarft( int matrix_layout, char direct, char storev,
                           lapack_int n, lapack_int k,
                           const lapack_complex_float* v, lapack_int ldv,
                           const lapack_complex_float* tau,
                           lapack_complex_float* t, lapack_int ldt );
lapack_int LAPACKE_zlarft( int matrix_layout, char direct, char storev,
                           lapack_int n, lapack_int k,
                           const lapack_complex_double* v, lapack_int ldv,
                           const lapack_complex_double* tau,
                           lapack_complex_double* t, lapack_int ldt );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `direct` | `char` | `'F'` = forward direction (H = H(1)*H(2)*...*H(k)), `'B'` = backward |
| `storev` | `char` | `'C'` = column-wise storage of V, `'R'` = row-wise |
| `n` | `lapack_int` | Order of block reflector |
| `k` | `lapack_int` | Number of elementary reflectors (order of T) |
| `v` | `const T*` | Matrix V containing the reflector vectors |
| `ldv` | `lapack_int` | Leading dimension of V |
| `tau` | `const T*` | Scalar factors (length k) |
| `t` | `T*` | [out] Triangular factor T (k x k) |
| `ldt` | `lapack_int` | Leading dimension of T |

---

### larfb -- Apply Block Reflector

Apply a block reflector H or its transpose/conjugate-transpose to a matrix C.

```c
lapack_int LAPACKE_slarfb( int matrix_layout, char side, char trans,
                           char direct, char storev, lapack_int m,
                           lapack_int n, lapack_int k, const float* v,
                           lapack_int ldv, const float* t, lapack_int ldt,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dlarfb( int matrix_layout, char side, char trans,
                           char direct, char storev, lapack_int m,
                           lapack_int n, lapack_int k, const double* v,
                           lapack_int ldv, const double* t, lapack_int ldt,
                           double* c, lapack_int ldc );
lapack_int LAPACKE_clarfb( int matrix_layout, char side, char trans,
                           char direct, char storev, lapack_int m,
                           lapack_int n, lapack_int k,
                           const lapack_complex_float* v, lapack_int ldv,
                           const lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zlarfb( int matrix_layout, char side, char trans,
                           char direct, char storev, lapack_int m,
                           lapack_int n, lapack_int k,
                           const lapack_complex_double* v, lapack_int ldv,
                           const lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'` = apply H from left, `'R'` = from right |
| `trans` | `char` | `'N'` = apply H, `'T'` = apply H^T (real), `'C'` = apply H^H (complex) |
| `direct` | `char` | `'F'` = forward, `'B'` = backward (must match larft) |
| `storev` | `char` | `'C'` = column-wise, `'R'` = row-wise (must match larft) |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `k` | `lapack_int` | Number of elementary reflectors |
| `v` | `const T*` | Matrix V containing reflector vectors |
| `ldv` | `lapack_int` | Leading dimension of V |
| `t` | `const T*` | Triangular factor T (from larft) |
| `ldt` | `lapack_int` | Leading dimension of T |
| `c` | `T*` | [in/out] Matrix C |
| `ldc` | `lapack_int` | Leading dimension of C |

---

### larfx -- Apply Elementary Reflector (Unblocked)

Apply a Householder reflector H to a matrix C from either side, specialized for small order.

```c
lapack_int LAPACKE_slarfx( int matrix_layout, char side, lapack_int m,
                           lapack_int n, const float* v, float tau, float* c,
                           lapack_int ldc, float* work );
lapack_int LAPACKE_dlarfx( int matrix_layout, char side, lapack_int m,
                           lapack_int n, const double* v, double tau, double* c,
                           lapack_int ldc, double* work );
lapack_int LAPACKE_clarfx( int matrix_layout, char side, lapack_int m,
                           lapack_int n, const lapack_complex_float* v,
                           lapack_complex_float tau, lapack_complex_float* c,
                           lapack_int ldc, lapack_complex_float* work );
lapack_int LAPACKE_zlarfx( int matrix_layout, char side, lapack_int m,
                           lapack_int n, const lapack_complex_double* v,
                           lapack_complex_double tau, lapack_complex_double* c,
                           lapack_int ldc, lapack_complex_double* work );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `side` | `char` | `'L'` = H*C, `'R'` = C*H |
| `m` | `lapack_int` | Number of rows of C |
| `n` | `lapack_int` | Number of columns of C |
| `v` | `const T*` | Reflector vector (length m if side='L', n if side='R') |
| `tau` | `T` | Scalar factor of the reflector |
| `c` | `T*` | [in/out] Matrix C |
| `ldc` | `lapack_int` | Leading dimension of C |
| `work` | `T*` | Workspace (length n if side='L', m if side='R'; not needed if order <= 10) |

---

## Miscellaneous

### Reciprocal Condition Numbers

**disna** -- Compute the reciprocal condition numbers for eigenvectors of a real symmetric or complex Hermitian matrix, or for singular vectors of a general matrix.

```c
lapack_int LAPACKE_sdisna( char job, lapack_int m, lapack_int n,
                           const float* d, float* sep );
lapack_int LAPACKE_ddisna( char job, lapack_int m, lapack_int n,
                           const double* d, double* sep );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `job` | `char` | `'E'` = eigenvectors of symmetric/Hermitian matrix, `'L'` = left singular vectors, `'R'` = right singular vectors |
| `m` | `lapack_int` | Number of rows of the matrix (if job='L' or 'R') or order (if job='E') |
| `n` | `lapack_int` | Number of columns (if job='L' or 'R'); not referenced if job='E' |
| `d` | `const T*` | Eigenvalues or singular values in decreasing order |
| `sep` | `T*` | [out] Reciprocal condition numbers (length m if job='L'/'E', n if job='R') |

---

### Safe Pythagorean Distance

**lapy2/lapy3** -- Compute sqrt(x^2 + y^2) or sqrt(x^2 + y^2 + z^2) safely, without unnecessary overflow or underflow.

```c
float  LAPACKE_slapy2( float x, float y );
double LAPACKE_dlapy2( double x, double y );

float  LAPACKE_slapy3( float x, float y, float z );
double LAPACKE_dlapy3( double x, double y, double z );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `x` | `float`/`double` | First value |
| `y` | `float`/`double` | Second value |
| `z` | `float`/`double` | Third value (lapy3 only) |

**Returns:** `sqrt(x^2 + y^2)` or `sqrt(x^2 + y^2 + z^2)`.

---

### Multiply Real by Complex

**lacrm** -- Multiply a complex matrix by a real matrix: C = A * B where A is complex and B is real.

**larcm** -- Multiply a real matrix by a complex matrix: C = A * B where A is real and B is complex.

```c
/* Complex * Real -> Complex */
lapack_int LAPACKE_clacrm( int matrix_layout, lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const float* b, lapack_int ldb,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zlacrm( int matrix_layout, lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const double* b, lapack_int ldb,
                           lapack_complex_double* c, lapack_int ldc );

/* Real * Complex -> Complex */
lapack_int LAPACKE_clarcm( int matrix_layout, lapack_int m, lapack_int n,
                           const float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zlarcm( int matrix_layout, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* c, lapack_int ldc );
```

**Parameters (lacrm):**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `m` | `lapack_int` | Number of rows of A and C |
| `n` | `lapack_int` | Number of columns of B and C |
| `a` | `const lapack_complex_T*` | [in] Complex matrix A (m x n) |
| `lda` | `lapack_int` | Leading dimension of A |
| `b` | `const T_real*` | [in] Real matrix B (n x n) |
| `ldb` | `lapack_int` | Leading dimension of B |
| `c` | `lapack_complex_T*` | [out] Result matrix C (m x n) |
| `ldc` | `lapack_int` | Leading dimension of C |

---

### Conjugate Vector

**lacgv** -- Conjugate a complex vector in place (complex only).

```c
lapack_int LAPACKE_clacgv( lapack_int n, lapack_complex_float* x,
                           lapack_int incx );
lapack_int LAPACKE_zlacgv( lapack_int n, lapack_complex_double* x,
                           lapack_int incx );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Number of elements |
| `x` | `lapack_complex_T*` | [in/out] Complex vector to conjugate |
| `incx` | `lapack_int` | Stride between elements |

---

### Estimate 1-Norm

**lacn2** -- Estimate the 1-norm of a matrix using reverse communication. This routine is called repeatedly to refine the estimate.

```c
/* Real */
lapack_int LAPACKE_slacn2( lapack_int n, float* v, float* x, lapack_int* isgn,
                           float* est, lapack_int* kase, lapack_int* isave );
lapack_int LAPACKE_dlacn2( lapack_int n, double* v, double* x,
                           lapack_int* isgn, double* est, lapack_int* kase,
                           lapack_int* isave );

/* Complex */
lapack_int LAPACKE_clacn2( lapack_int n, lapack_complex_float* v,
                           lapack_complex_float* x,
                           float* est, lapack_int* kase, lapack_int* isave );
lapack_int LAPACKE_zlacn2( lapack_int n, lapack_complex_double* v,
                           lapack_complex_double* x,
                           double* est, lapack_int* kase, lapack_int* isave );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Order of the matrix |
| `v` | `T*` | Workspace vector (length n) |
| `x` | `T*` | [in/out] Vector used in the iteration (length n) |
| `isgn` | `lapack_int*` | Integer workspace (length n, real variants only) |
| `est` | `T_real*` | [in/out] Estimated 1-norm |
| `kase` | `lapack_int*` | [in/out] Reverse communication flag (set to 0 on first call) |
| `isave` | `lapack_int*` | [in/out] Integer workspace for saving state (length 3) |

---

### Generate Plane Rotation

**lartgp** -- Generate a plane rotation so that the second component is zero, with the cosine non-negative.

```c
lapack_int LAPACKE_slartgp( float f, float g, float* cs, float* sn, float* r );
lapack_int LAPACKE_dlartgp( double f, double g, double* cs, double* sn,
                            double* r );
```

**lartgs** -- Generate a plane rotation for use in the bidiagonal SVD.

```c
lapack_int LAPACKE_slartgs( float x, float y, float sigma, float* cs,
                            float* sn );
lapack_int LAPACKE_dlartgs( double x, double y, double sigma, double* cs,
                            double* sn );
```

**Parameters (lartgp):**

| Name | Type | Description |
|------|------|-------------|
| `f` | `T` | First component of vector to be rotated |
| `g` | `T` | Second component of vector to be rotated |
| `cs` | `T*` | [out] Cosine of the rotation |
| `sn` | `T*` | [out] Sine of the rotation |
| `r` | `T*` | [out] The nonzero component of the rotated vector |

**Parameters (lartgs):**

| Name | Type | Description |
|------|------|-------------|
| `x` | `T` | First input value |
| `y` | `T` | Second input value |
| `sigma` | `T` | Shift parameter |
| `cs` | `T*` | [out] Cosine of the rotation |
| `sn` | `T*` | [out] Sine of the rotation |

---

### Permute Rows/Columns

**lapmr** -- Rearrange the rows of a matrix as specified by a permutation vector.

```c
lapack_int LAPACKE_slapmr( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, float* x, lapack_int ldx,
                           lapack_int* k );
lapack_int LAPACKE_dlapmr( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, double* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_clapmr( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_float* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_zlapmr( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* k );
```

**lapmt** -- Rearrange the columns of a matrix as specified by a permutation vector.

```c
lapack_int LAPACKE_slapmt( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, float* x, lapack_int ldx,
                           lapack_int* k );
lapack_int LAPACKE_dlapmt( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, double* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_clapmt( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_float* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_zlapmt( int matrix_layout, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* k );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `forwrd` | `lapack_logical` | If nonzero, apply forward permutation. If zero, apply backward permutation |
| `m` | `lapack_int` | Number of rows |
| `n` | `lapack_int` | Number of columns |
| `x` | `T*` | [in/out] Matrix to permute |
| `ldx` | `lapack_int` | Leading dimension of X |
| `k` | `lapack_int*` | [in/out] Permutation vector (length m for lapmr, n for lapmt; 1-indexed) |

---

### Sum of Squares

**lassq** -- Update a sum of squares representation: scale and sumsq such that `scale^2 * sumsq = x(1)^2 + ... + x(n)^2 + scale_in^2 * sumsq_in`.

```c
lapack_int LAPACKE_slassq( lapack_int n, float* x, lapack_int incx,
                           float* scale, float* sumsq );
lapack_int LAPACKE_dlassq( lapack_int n, double* x, lapack_int incx,
                           double* scale, double* sumsq );
lapack_int LAPACKE_classq( lapack_int n, lapack_complex_float* x,
                           lapack_int incx, float* scale, float* sumsq );
lapack_int LAPACKE_zlassq( lapack_int n, lapack_complex_double* x,
                           lapack_int incx, double* scale, double* sumsq );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n` | `lapack_int` | Number of elements in x |
| `x` | `T*` | Input vector |
| `incx` | `lapack_int` | Stride between elements of x |
| `scale` | `T_real*` | [in/out] Scaling factor |
| `sumsq` | `T_real*` | [in/out] Sum of squares (scaled) |

---

### Sort

**lasrt** -- Sort the elements of a real vector in increasing or decreasing order.

```c
lapack_int LAPACKE_slasrt( char id, lapack_int n, float* d );
lapack_int LAPACKE_dlasrt( char id, lapack_int n, double* d );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `char` | `'I'` = sort in increasing order, `'D'` = sort in decreasing order |
| `n` | `lapack_int` | Number of elements to sort |
| `d` | `T*` | [in/out] Array to sort (length n) |

---

### RFP Format Operations

Rectangular Full Packed (RFP) format stores a triangular or symmetric/Hermitian matrix in a compact rectangular array, enabling use of Level 3 BLAS for improved performance.

#### sfrk -- Symmetric Rank-k Update (RFP)

Perform a symmetric rank-k update on a matrix in RFP format: C = alpha * A * A^T + beta * C (or A^T * A).

```c
lapack_int LAPACKE_ssfrk( int matrix_layout, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, float alpha,
                          const float* a, lapack_int lda, float beta,
                          float* c );
lapack_int LAPACKE_dsfrk( int matrix_layout, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, double alpha,
                          const double* a, lapack_int lda, double beta,
                          double* c );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'` = normal RFP storage, `'T'` = transposed RFP storage |
| `uplo` | `char` | `'U'` = upper triangle of C stored, `'L'` = lower triangle |
| `trans` | `char` | `'N'` = C := alpha*A*A^T + beta*C, `'T'` = C := alpha*A^T*A + beta*C |
| `n` | `lapack_int` | Order of matrix C |
| `k` | `lapack_int` | Number of columns of A (if trans='N') or rows (if trans='T') |
| `alpha` | `T` | Scalar multiplier for A*A^T |
| `a` | `const T*` | Input matrix A |
| `lda` | `lapack_int` | Leading dimension of A |
| `beta` | `T` | Scalar multiplier for C |
| `c` | `T*` | [in/out] Symmetric matrix C in RFP format |

#### hfrk -- Hermitian Rank-k Update (RFP)

Perform a Hermitian rank-k update on a matrix in RFP format: C = alpha * A * A^H + beta * C (complex only).

```c
lapack_int LAPACKE_chfrk( int matrix_layout, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, float alpha,
                          const lapack_complex_float* a, lapack_int lda,
                          float beta, lapack_complex_float* c );
lapack_int LAPACKE_zhfrk( int matrix_layout, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, double alpha,
                          const lapack_complex_double* a, lapack_int lda,
                          double beta, lapack_complex_double* c );
```

**Parameters:** Same as `sfrk` but `alpha` and `beta` are real scalars (float/double) and A, C are complex.

#### tfsm -- Triangular Solve (RFP)

Solve a triangular system with a matrix in RFP format: op(A) * X = alpha * B or X * op(A) = alpha * B.

```c
lapack_int LAPACKE_stfsm( int matrix_layout, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          float alpha, const float* a, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dtfsm( int matrix_layout, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          double alpha, const double* a, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_ctfsm( int matrix_layout, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          lapack_complex_float alpha,
                          const lapack_complex_float* a,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztfsm( int matrix_layout, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          lapack_complex_double alpha,
                          const lapack_complex_double* a,
                          lapack_complex_double* b, lapack_int ldb );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'` = normal RFP, `'T'`/`'C'` = transposed/conjugate-transposed RFP |
| `side` | `char` | `'L'` = op(A)*X = alpha*B, `'R'` = X*op(A) = alpha*B |
| `uplo` | `char` | `'U'` = upper triangular, `'L'` = lower triangular |
| `trans` | `char` | `'N'` = A, `'T'` = A^T, `'C'` = A^H |
| `diag` | `char` | `'N'` = non-unit diagonal, `'U'` = unit diagonal |
| `m` | `lapack_int` | Number of rows of B |
| `n` | `lapack_int` | Number of columns of B |
| `alpha` | `T` | Scalar multiplier |
| `a` | `const T*` | Triangular matrix A in RFP format |
| `b` | `T*` | [in/out] On entry, matrix B. On exit, solution X |
| `ldb` | `lapack_int` | Leading dimension of B |

#### tftri -- Triangular Inverse (RFP)

Compute the inverse of a triangular matrix stored in RFP format.

```c
lapack_int LAPACKE_stftri( int matrix_layout, char transr, char uplo, char diag,
                           lapack_int n, float* a );
lapack_int LAPACKE_dtftri( int matrix_layout, char transr, char uplo, char diag,
                           lapack_int n, double* a );
lapack_int LAPACKE_ctftri( int matrix_layout, char transr, char uplo, char diag,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_ztftri( int matrix_layout, char transr, char uplo, char diag,
                           lapack_int n, lapack_complex_double* a );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `transr` | `char` | `'N'` = normal RFP, `'T'`/`'C'` = transposed/conjugate-transposed RFP |
| `uplo` | `char` | `'U'` = upper triangular, `'L'` = lower triangular |
| `diag` | `char` | `'N'` = non-unit diagonal, `'U'` = unit diagonal |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `T*` | [in/out] On entry, triangular matrix in RFP. On exit, its inverse in RFP |

---

### Symmetric/Hermitian Conversions and Swaps

#### syconv -- Convert Symmetric Factorization Format

Convert between the standard and alternate representations of a symmetric factorization from `sytrf`.

```c
lapack_int LAPACKE_ssyconv( int matrix_layout, char uplo, char way,
                            lapack_int n, float* a, lapack_int lda,
                            const lapack_int* ipiv, float* e );
lapack_int LAPACKE_dsyconv( int matrix_layout, char uplo, char way,
                            lapack_int n, double* a, lapack_int lda,
                            const lapack_int* ipiv, double* e );
lapack_int LAPACKE_csyconv( int matrix_layout, char uplo, char way,
                            lapack_int n, lapack_complex_float* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_float* e );
lapack_int LAPACKE_zsyconv( int matrix_layout, char uplo, char way,
                            lapack_int n, lapack_complex_double* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_double* e );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` or `'L'` -- must match sytrf |
| `way` | `char` | `'C'` = convert, `'R'` = revert |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `T*` | [in/out] Factored matrix from sytrf |
| `lda` | `lapack_int` | Leading dimension of A |
| `ipiv` | `const lapack_int*` | Pivot indices from sytrf |
| `e` | `T*` | [out] Auxiliary array (length n) |

#### syswapr -- Symmetric Matrix Row/Column Swap

Apply a symmetric permutation (swap rows and columns i1 and i2) to a symmetric matrix.

```c
lapack_int LAPACKE_ssyswapr( int matrix_layout, char uplo, lapack_int n,
                             float* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_dsyswapr( int matrix_layout, char uplo, lapack_int n,
                             double* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_csyswapr( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_zsyswapr( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle stored, `'L'` = lower triangle |
| `n` | `lapack_int` | Order of matrix A |
| `a` | `T*` | [in/out] Symmetric matrix |
| `lda` | `lapack_int` | Leading dimension of A |
| `i1` | `lapack_int` | First index to swap (0-indexed) |
| `i2` | `lapack_int` | Second index to swap (0-indexed) |

#### heswapr -- Hermitian Matrix Row/Column Swap

Apply a Hermitian permutation to a Hermitian matrix (complex only).

```c
lapack_int LAPACKE_cheswapr( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_zheswapr( int matrix_layout, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             lapack_int i1, lapack_int i2 );
```

**Parameters:** Same as `syswapr` but for Hermitian matrices.

---

### Complex Symmetric Rank-1 Update

**syr** -- LAPACKE version of symmetric rank-1 update for complex types (not in standard BLAS): A = alpha * x * x^T + A.

```c
lapack_int LAPACKE_csyr( int matrix_layout, char uplo, lapack_int n,
                         lapack_complex_float alpha,
                         const lapack_complex_float* x, lapack_int incx,
                         lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zsyr( int matrix_layout, char uplo, lapack_int n,
                         lapack_complex_double alpha,
                         const lapack_complex_double* x, lapack_int incx,
                         lapack_complex_double* a, lapack_int lda );
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `matrix_layout` | `int` | `LAPACK_ROW_MAJOR` or `LAPACK_COL_MAJOR` |
| `uplo` | `char` | `'U'` = upper triangle, `'L'` = lower triangle |
| `n` | `lapack_int` | Order of matrix A |
| `alpha` | `lapack_complex_T` | Scalar multiplier |
| `x` | `const lapack_complex_T*` | Input vector of length n |
| `incx` | `lapack_int` | Stride between elements of x |
| `a` | `lapack_complex_T*` | [in/out] Complex symmetric matrix (n x n) |
| `lda` | `lapack_int` | Leading dimension of A |

**Note:** This is a symmetric (not Hermitian) rank-1 update: uses x * x^T (transpose, not conjugate transpose). Standard BLAS `cher`/`zher` perform the Hermitian x * x^H update instead.
