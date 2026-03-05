# CBLAS Level 1 API Reference - Vector Operations

> BLAS Level 1 operations perform scalar, vector, and vector-vector operations.
> Source: LAPACK v3.12.1 - `CBLAS/include/cblas.h`

## Table of Contents
- [Types and Enums](#types-and-enums)
- [Complex Absolute Value](#complex-absolute-value)
- [Dot Products](#dot-products)
- [Norms and Absolute Value Sums](#norms-and-absolute-value-sums)
- [Index of Maximum](#index-of-maximum)
- [Vector Swap](#vector-swap)
- [Vector Copy](#vector-copy)
- [AXPY (y = alpha*x + y)](#axpy-y--alphax--y)
- [Rotation](#rotation)
- [Modified Givens Rotation](#modified-givens-rotation)
- [Scaling](#scaling)

## Types and Enums

| Type | Definition | Description |
|------|-----------|-------------|
| `CBLAS_INT` | `int32_t` (default) or `int64_t` (WeirdNEC) | Integer type for dimensions and increments |
| `CBLAS_INDEX` | `size_t` | Return type for index-of-maximum functions |

### Precision Prefixes

| Prefix | Precision |
|--------|-----------|
| `s` | Single-precision real (`float`) |
| `d` | Double-precision real (`double`) |
| `c` | Single-precision complex (`void*` pointing to `float[2]`) |
| `z` | Double-precision complex (`void*` pointing to `double[2]`) |
| `sc` | Single-precision complex input, single-precision real output |
| `dz` | Double-precision complex input, double-precision real output |
| `cs` | Single-precision complex with real scalar |
| `zd` | Double-precision complex with real scalar |
| `sd` / `ds` | Mixed single/double precision |

### Complex Type Convention

Complex numbers are passed as `const void *` (input) or `void *` (output). The underlying data is an array of two floats `{real, imag}` for single-precision complex, or two doubles `{real, imag}` for double-precision complex. Complex scalars like `alpha` are also passed as `const void *` for the `c` and `z` prefixed routines.

---

## Complex Absolute Value

### cblas_dcabs1

Compute the sum of absolute values of real and imaginary parts of a double-precision complex number: `|Re(z)| + |Im(z)|`.

```c
double cblas_dcabs1(const void  *z);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `z` | `const void *` | Pointer to a double-precision complex number (double[2]) |

**Returns:** `double` -- `|Re(z)| + |Im(z)|`

---

### cblas_scabs1

Compute the sum of absolute values of real and imaginary parts of a single-precision complex number: `|Re(c)| + |Im(c)|`.

```c
float  cblas_scabs1(const void  *c);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `c` | `const void *` | Pointer to a single-precision complex number (float[2]) |

**Returns:** `float` -- `|Re(c)| + |Im(c)|`

---

## Dot Products

### cblas_sdsdot

Compute the dot product of two single-precision vectors with a single-precision scalar added, using double-precision accumulation internally: `result = alpha + X^T * Y`.

```c
float  cblas_sdsdot(const CBLAS_INT N, const float alpha, const float *X,
                    const CBLAS_INT incX, const float *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `alpha` | `const float` | Scalar to add to the dot product |
| `X` | `const float *` | Pointer to first input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const float *` | Pointer to second input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

**Returns:** `float` -- `alpha + sum(X[i] * Y[i])` computed in double precision, cast to float

---

### cblas_dsdot

Compute the dot product of two single-precision vectors using double-precision accumulation: `result = X^T * Y`.

```c
double cblas_dsdot(const CBLAS_INT N, const float *X, const CBLAS_INT incX, const float *Y,
                   const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const float *` | Pointer to first input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const float *` | Pointer to second input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

**Returns:** `double` -- `sum(X[i] * Y[i])` accumulated in double precision

---

### cblas_sdot

Compute the dot product of two single-precision real vectors: `result = X^T * Y`.

```c
float  cblas_sdot(const CBLAS_INT N, const float  *X, const CBLAS_INT incX,
                  const float  *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const float *` | Pointer to first input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const float *` | Pointer to second input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

**Returns:** `float` -- `sum(X[i] * Y[i])`

---

### cblas_ddot

Compute the dot product of two double-precision real vectors: `result = X^T * Y`.

```c
double cblas_ddot(const CBLAS_INT N, const double *X, const CBLAS_INT incX,
                  const double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const double *` | Pointer to first input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const double *` | Pointer to second input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

**Returns:** `double` -- `sum(X[i] * Y[i])`

---

### cblas_cdotu_sub

Compute the unconjugated dot product of two single-precision complex vectors: `dotu = X^T * Y`.

```c
void   cblas_cdotu_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotu);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const void *` | Pointer to first single-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const void *` | Pointer to second single-precision complex input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `dotu` | `void *` | Output: pointer to the resulting single-precision complex dot product |

**Returns:** void (result stored in `dotu`)

---

### cblas_cdotc_sub

Compute the conjugated dot product of two single-precision complex vectors: `dotc = X^H * Y`.

```c
void   cblas_cdotc_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotc);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const void *` | Pointer to first single-precision complex input vector (conjugated) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const void *` | Pointer to second single-precision complex input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `dotc` | `void *` | Output: pointer to the resulting single-precision complex dot product |

**Returns:** void (result stored in `dotc`)

---

### cblas_zdotu_sub

Compute the unconjugated dot product of two double-precision complex vectors: `dotu = X^T * Y`.

```c
void   cblas_zdotu_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotu);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const void *` | Pointer to first double-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const void *` | Pointer to second double-precision complex input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `dotu` | `void *` | Output: pointer to the resulting double-precision complex dot product |

**Returns:** void (result stored in `dotu`)

---

### cblas_zdotc_sub

Compute the conjugated dot product of two double-precision complex vectors: `dotc = X^H * Y`.

```c
void   cblas_zdotc_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotc);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `const void *` | Pointer to first double-precision complex input vector (conjugated) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `const void *` | Pointer to second double-precision complex input vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `dotc` | `void *` | Output: pointer to the resulting double-precision complex dot product |

**Returns:** void (result stored in `dotc`)

---

## Norms and Absolute Value Sums

### cblas_snrm2

Compute the Euclidean norm (L2 norm) of a single-precision real vector: `result = ||X||_2`.

```c
float  cblas_snrm2(const CBLAS_INT N, const float *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const float *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `float` -- Euclidean norm `sqrt(sum(X[i]^2))`

---

### cblas_sasum

Compute the sum of absolute values of a single-precision real vector: `result = ||X||_1`.

```c
float  cblas_sasum(const CBLAS_INT N, const float *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const float *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `float` -- `sum(|X[i]|)`

---

### cblas_dnrm2

Compute the Euclidean norm (L2 norm) of a double-precision real vector: `result = ||X||_2`.

```c
double cblas_dnrm2(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const double *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `double` -- Euclidean norm `sqrt(sum(X[i]^2))`

---

### cblas_dasum

Compute the sum of absolute values of a double-precision real vector: `result = ||X||_1`.

```c
double cblas_dasum(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const double *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `double` -- `sum(|X[i]|)`

---

### cblas_scnrm2

Compute the Euclidean norm (L2 norm) of a single-precision complex vector: `result = ||X||_2`.

```c
float  cblas_scnrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to single-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `float` -- Euclidean norm `sqrt(sum(|X[i]|^2))` where `|X[i]|^2 = Re(X[i])^2 + Im(X[i])^2`

---

### cblas_scasum

Compute the sum of absolute values of real and imaginary parts of a single-precision complex vector.

```c
float  cblas_scasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to single-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `float` -- `sum(|Re(X[i])| + |Im(X[i])|)`

---

### cblas_dznrm2

Compute the Euclidean norm (L2 norm) of a double-precision complex vector: `result = ||X||_2`.

```c
double cblas_dznrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to double-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `double` -- Euclidean norm `sqrt(sum(|X[i]|^2))` where `|X[i]|^2 = Re(X[i])^2 + Im(X[i])^2`

---

### cblas_dzasum

Compute the sum of absolute values of real and imaginary parts of a double-precision complex vector.

```c
double cblas_dzasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to double-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `double` -- `sum(|Re(X[i])| + |Im(X[i])|)`

---

## Index of Maximum

### cblas_isamax

Find the index of the element with the maximum absolute value in a single-precision real vector.

```c
CBLAS_INDEX cblas_isamax(const CBLAS_INT N, const float  *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const float *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `CBLAS_INDEX` -- Zero-based index of the element with the largest absolute value

---

### cblas_idamax

Find the index of the element with the maximum absolute value in a double-precision real vector.

```c
CBLAS_INDEX cblas_idamax(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const double *` | Pointer to input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `CBLAS_INDEX` -- Zero-based index of the element with the largest absolute value

---

### cblas_icamax

Find the index of the element with the maximum absolute value in a single-precision complex vector. The absolute value used is `|Re(X[i])| + |Im(X[i])|`.

```c
CBLAS_INDEX cblas_icamax(const CBLAS_INT N, const void   *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to single-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `CBLAS_INDEX` -- Zero-based index of the element with the largest absolute value

---

### cblas_izamax

Find the index of the element with the maximum absolute value in a double-precision complex vector. The absolute value used is `|Re(X[i])| + |Im(X[i])|`.

```c
CBLAS_INDEX cblas_izamax(const CBLAS_INT N, const void   *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `X` | `const void *` | Pointer to double-precision complex input vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

**Returns:** `CBLAS_INDEX` -- Zero-based index of the element with the largest absolute value

---

## Vector Swap

### cblas_sswap

Swap the elements of two single-precision real vectors.

```c
void cblas_sswap(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `float *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `float *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_dswap

Swap the elements of two double-precision real vectors.

```c
void cblas_dswap(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `double *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `double *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_cswap

Swap the elements of two single-precision complex vectors.

```c
void cblas_cswap(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `void *` | Pointer to first single-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to second single-precision complex vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_zswap

Swap the elements of two double-precision complex vectors.

```c
void cblas_zswap(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `void *` | Pointer to first double-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to second double-precision complex vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

## Vector Copy

### cblas_scopy

Copy a single-precision real vector X into vector Y.

```c
void cblas_scopy(const CBLAS_INT N, const float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements to copy |
| `X` | `const float *` | Pointer to source vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `float *` | Pointer to destination vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_dcopy

Copy a double-precision real vector X into vector Y.

```c
void cblas_dcopy(const CBLAS_INT N, const double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements to copy |
| `X` | `const double *` | Pointer to source vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `double *` | Pointer to destination vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_ccopy

Copy a single-precision complex vector X into vector Y.

```c
void cblas_ccopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements to copy |
| `X` | `const void *` | Pointer to single-precision complex source vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to single-precision complex destination vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_zcopy

Copy a double-precision complex vector X into vector Y.

```c
void cblas_zcopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements to copy |
| `X` | `const void *` | Pointer to double-precision complex source vector |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to double-precision complex destination vector |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

## AXPY (y = alpha*x + y)

### cblas_saxpy

Compute `Y = alpha * X + Y` for single-precision real vectors.

```c
void cblas_saxpy(const CBLAS_INT N, const float alpha, const float *X,
                 const CBLAS_INT incX, float *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `alpha` | `const float` | Scalar multiplier for X |
| `X` | `const float *` | Pointer to input vector X |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `float *` | Pointer to input/output vector Y (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_daxpy

Compute `Y = alpha * X + Y` for double-precision real vectors.

```c
void cblas_daxpy(const CBLAS_INT N, const double alpha, const double *X,
                 const CBLAS_INT incX, double *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `alpha` | `const double` | Scalar multiplier for X |
| `X` | `const double *` | Pointer to input vector X |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `double *` | Pointer to input/output vector Y (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_caxpy

Compute `Y = alpha * X + Y` for single-precision complex vectors.

```c
void cblas_caxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `alpha` | `const void *` | Pointer to single-precision complex scalar multiplier for X |
| `X` | `const void *` | Pointer to single-precision complex input vector X |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to single-precision complex input/output vector Y (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

### cblas_zaxpy

Compute `Y = alpha * X + Y` for double-precision complex vectors.

```c
void cblas_zaxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `alpha` | `const void *` | Pointer to double-precision complex scalar multiplier for X |
| `X` | `const void *` | Pointer to double-precision complex input vector X |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to double-precision complex input/output vector Y (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |

---

## Rotation

### cblas_srotg

Construct a Givens plane rotation for single-precision real values. Given `(a, b)`, computes `(r, z, c, s)` such that the rotation matrix `[[c, s], [-s, c]]` applied to `[a, b]^T` yields `[r, 0]^T`.

```c
void cblas_srotg(float *a, float *b, float *c, float *s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `float *` | On entry: x-coordinate. On exit: r = norm(a, b) |
| `b` | `float *` | On entry: y-coordinate. On exit: z (reconstruction parameter) |
| `c` | `float *` | Output: cosine of the rotation angle |
| `s` | `float *` | Output: sine of the rotation angle |

---

### cblas_drotg

Construct a Givens plane rotation for double-precision real values.

```c
void cblas_drotg(double *a, double *b, double *c, double *s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `double *` | On entry: x-coordinate. On exit: r = norm(a, b) |
| `b` | `double *` | On entry: y-coordinate. On exit: z (reconstruction parameter) |
| `c` | `double *` | Output: cosine of the rotation angle |
| `s` | `double *` | Output: sine of the rotation angle |

---

### cblas_crotg

Construct a Givens plane rotation for single-precision complex values.

```c
void cblas_crotg(void *a, void *b, float *c, void *s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `void *` | On entry: single-precision complex x-coordinate. On exit: r |
| `b` | `void *` | On entry: single-precision complex y-coordinate. On exit: z |
| `c` | `float *` | Output: real cosine of the rotation angle |
| `s` | `void *` | Output: single-precision complex sine of the rotation angle |

---

### cblas_zrotg

Construct a Givens plane rotation for double-precision complex values.

```c
void cblas_zrotg(void *a, void *b, double *c, void *s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `void *` | On entry: double-precision complex x-coordinate. On exit: r |
| `b` | `void *` | On entry: double-precision complex y-coordinate. On exit: z |
| `c` | `double *` | Output: real cosine of the rotation angle |
| `s` | `void *` | Output: double-precision complex sine of the rotation angle |

---

### cblas_srot

Apply a Givens plane rotation to single-precision real vectors: `X[i] = c*X[i] + s*Y[i]`, `Y[i] = c*Y[i] - s*X[i]`.

```c
void cblas_srot(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                float *Y, const CBLAS_INT incY, const float c, const float s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `float *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `float *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `c` | `const float` | Cosine of the rotation angle |
| `s` | `const float` | Sine of the rotation angle |

---

### cblas_drot

Apply a Givens plane rotation to double-precision real vectors: `X[i] = c*X[i] + s*Y[i]`, `Y[i] = c*Y[i] - s*X[i]`.

```c
void cblas_drot(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                double *Y, const CBLAS_INT incY, const double c, const double  s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `double *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `double *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `c` | `const double` | Cosine of the rotation angle |
| `s` | `const double` | Sine of the rotation angle |

---

### cblas_csrot

Apply a Givens plane rotation to single-precision complex vectors with real cosine and sine parameters.

```c
void cblas_csrot(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY, const float c, const float s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `void *` | Pointer to first single-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to second single-precision complex vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `c` | `const float` | Real cosine of the rotation angle |
| `s` | `const float` | Real sine of the rotation angle |

---

### cblas_zdrot

Apply a Givens plane rotation to double-precision complex vectors with real cosine and sine parameters.

```c
void cblas_zdrot(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY, const double c, const double s);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `void *` | Pointer to first double-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `void *` | Pointer to second double-precision complex vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `c` | `const double` | Real cosine of the rotation angle |
| `s` | `const double` | Real sine of the rotation angle |

---

## Modified Givens Rotation

### cblas_srotmg

Construct a modified Givens rotation for single-precision real values. Computes the parameters for a modified Givens transformation that zeros the second component of `[sqrt(d1)*b1, sqrt(d2)*b2]^T`.

```c
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `d1` | `float *` | On entry/exit: scaling factor for x-component |
| `d2` | `float *` | On entry/exit: scaling factor for y-component |
| `b1` | `float *` | On entry/exit: x-coordinate of the input vector |
| `b2` | `const float` | y-coordinate of the input vector |
| `P` | `float *` | Output: 5-element parameter array `[flag, h11, h21, h12, h22]` defining the rotation matrix |

---

### cblas_srotm

Apply a modified Givens rotation to single-precision real vectors using the parameter array P computed by `cblas_srotmg`.

```c
void cblas_srotm(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY, const float *P);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `float *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `float *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `P` | `const float *` | 5-element parameter array from `cblas_srotmg`: `[flag, h11, h21, h12, h22]` |

---

### cblas_drotmg

Construct a modified Givens rotation for double-precision real values.

```c
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `d1` | `double *` | On entry/exit: scaling factor for x-component |
| `d2` | `double *` | On entry/exit: scaling factor for y-component |
| `b1` | `double *` | On entry/exit: x-coordinate of the input vector |
| `b2` | `const double` | y-coordinate of the input vector |
| `P` | `double *` | Output: 5-element parameter array `[flag, h11, h21, h12, h22]` defining the rotation matrix |

---

### cblas_drotm

Apply a modified Givens rotation to double-precision real vectors using the parameter array P computed by `cblas_drotmg`.

```c
void cblas_drotm(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY, const double *P);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vectors |
| `X` | `double *` | Pointer to first vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |
| `Y` | `double *` | Pointer to second vector (modified in-place) |
| `incY` | `const CBLAS_INT` | Storage spacing (stride) between elements of Y |
| `P` | `const double *` | 5-element parameter array from `cblas_drotmg`: `[flag, h11, h21, h12, h22]` |

---

## Scaling

### cblas_sscal

Scale a single-precision real vector by a single-precision real scalar: `X = alpha * X`.

```c
void cblas_sscal(const CBLAS_INT N, const float alpha, float *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const float` | Scalar multiplier |
| `X` | `float *` | Pointer to vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

### cblas_dscal

Scale a double-precision real vector by a double-precision real scalar: `X = alpha * X`.

```c
void cblas_dscal(const CBLAS_INT N, const double alpha, double *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const double` | Scalar multiplier |
| `X` | `double *` | Pointer to vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

### cblas_cscal

Scale a single-precision complex vector by a single-precision complex scalar: `X = alpha * X`.

```c
void cblas_cscal(const CBLAS_INT N, const void *alpha, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const void *` | Pointer to single-precision complex scalar multiplier |
| `X` | `void *` | Pointer to single-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

### cblas_zscal

Scale a double-precision complex vector by a double-precision complex scalar: `X = alpha * X`.

```c
void cblas_zscal(const CBLAS_INT N, const void *alpha, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const void *` | Pointer to double-precision complex scalar multiplier |
| `X` | `void *` | Pointer to double-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

### cblas_csscal

Scale a single-precision complex vector by a single-precision real scalar: `X = alpha * X`.

```c
void cblas_csscal(const CBLAS_INT N, const float alpha, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const float` | Real scalar multiplier |
| `X` | `void *` | Pointer to single-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

### cblas_zdscal

Scale a double-precision complex vector by a double-precision real scalar: `X = alpha * X`.

```c
void cblas_zdscal(const CBLAS_INT N, const double alpha, void *X, const CBLAS_INT incX);
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `N` | `const CBLAS_INT` | Number of elements in input vector |
| `alpha` | `const double` | Real scalar multiplier |
| `X` | `void *` | Pointer to double-precision complex vector (modified in-place) |
| `incX` | `const CBLAS_INT` | Storage spacing (stride) between elements of X |

---

## Function Quick Reference

| Function | Operation | Precision |
|----------|-----------|-----------|
| `cblas_dcabs1` | `\|Re(z)\| + \|Im(z)\|` | double-complex to double |
| `cblas_scabs1` | `\|Re(c)\| + \|Im(c)\|` | single-complex to single |
| `cblas_sdsdot` | `alpha + X^T * Y` (double accumulation) | single (with double internal) |
| `cblas_dsdot` | `X^T * Y` (double accumulation) | single to double |
| `cblas_sdot` | `X^T * Y` | single |
| `cblas_ddot` | `X^T * Y` | double |
| `cblas_cdotu_sub` | `X^T * Y` (unconjugated) | single-complex |
| `cblas_cdotc_sub` | `X^H * Y` (conjugated) | single-complex |
| `cblas_zdotu_sub` | `X^T * Y` (unconjugated) | double-complex |
| `cblas_zdotc_sub` | `X^H * Y` (conjugated) | double-complex |
| `cblas_snrm2` | `\|\|X\|\|_2` | single |
| `cblas_sasum` | `\|\|X\|\|_1` | single |
| `cblas_dnrm2` | `\|\|X\|\|_2` | double |
| `cblas_dasum` | `\|\|X\|\|_1` | double |
| `cblas_scnrm2` | `\|\|X\|\|_2` | single-complex to single |
| `cblas_scasum` | `sum(\|Re\| + \|Im\|)` | single-complex to single |
| `cblas_dznrm2` | `\|\|X\|\|_2` | double-complex to double |
| `cblas_dzasum` | `sum(\|Re\| + \|Im\|)` | double-complex to double |
| `cblas_isamax` | index of max `\|X[i]\|` | single |
| `cblas_idamax` | index of max `\|X[i]\|` | double |
| `cblas_icamax` | index of max `\|X[i]\|` | single-complex |
| `cblas_izamax` | index of max `\|X[i]\|` | double-complex |
| `cblas_sswap` | `X <-> Y` | single |
| `cblas_dswap` | `X <-> Y` | double |
| `cblas_cswap` | `X <-> Y` | single-complex |
| `cblas_zswap` | `X <-> Y` | double-complex |
| `cblas_scopy` | `Y = X` | single |
| `cblas_dcopy` | `Y = X` | double |
| `cblas_ccopy` | `Y = X` | single-complex |
| `cblas_zcopy` | `Y = X` | double-complex |
| `cblas_saxpy` | `Y = alpha*X + Y` | single |
| `cblas_daxpy` | `Y = alpha*X + Y` | double |
| `cblas_caxpy` | `Y = alpha*X + Y` | single-complex |
| `cblas_zaxpy` | `Y = alpha*X + Y` | double-complex |
| `cblas_srotg` | construct Givens rotation | single |
| `cblas_drotg` | construct Givens rotation | double |
| `cblas_crotg` | construct Givens rotation | single-complex |
| `cblas_zrotg` | construct Givens rotation | double-complex |
| `cblas_srot` | apply Givens rotation | single |
| `cblas_drot` | apply Givens rotation | double |
| `cblas_csrot` | apply Givens rotation (real c,s) | single-complex |
| `cblas_zdrot` | apply Givens rotation (real c,s) | double-complex |
| `cblas_srotmg` | construct modified Givens | single |
| `cblas_drotmg` | construct modified Givens | double |
| `cblas_srotm` | apply modified Givens | single |
| `cblas_drotm` | apply modified Givens | double |
| `cblas_sscal` | `X = alpha*X` | single |
| `cblas_dscal` | `X = alpha*X` | double |
| `cblas_cscal` | `X = alpha*X` | single-complex |
| `cblas_zscal` | `X = alpha*X` | double-complex |
| `cblas_csscal` | `X = alpha*X` (real alpha) | single-complex (real scalar) |
| `cblas_zdscal` | `X = alpha*X` (real alpha) | double-complex (real scalar) |
