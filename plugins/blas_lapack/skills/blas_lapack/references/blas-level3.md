# CBLAS Level 3 API Reference - Matrix-Matrix Operations

> BLAS Level 3 operations perform matrix-matrix operations.
> Source: LAPACK v3.12.1 - `CBLAS/include/cblas.h`

## Table of Contents
- [General Matrix Multiply (gemm, gemmtr)](#general-matrix-multiply)
- [Symmetric Matrix Multiply (symm)](#symmetric-matrix-multiply)
- [Hermitian Matrix Multiply (hemm)](#hermitian-matrix-multiply)
- [Symmetric Rank-k Update (syrk, syr2k)](#symmetric-rank-k-update)
- [Hermitian Rank-k Update (herk, her2k)](#hermitian-rank-k-update)
- [Triangular Matrix Operations (trmm, trsm)](#triangular-matrix-operations)
- [Error Handler (xerbla)](#error-handler)

## Enums Used

| Enum | Values | Purpose |
|------|--------|---------|
| `CBLAS_LAYOUT` | `CblasRowMajor=101`, `CblasColMajor=102` | Memory layout of matrices |
| `CBLAS_TRANSPOSE` | `CblasNoTrans=111`, `CblasTrans=112`, `CblasConjTrans=113` | Transpose operation on matrix |
| `CBLAS_UPLO` | `CblasUpper=121`, `CblasLower=122` | Upper or lower triangular storage |
| `CBLAS_SIDE` | `CblasLeft=141`, `CblasRight=142` | Multiply from left or right side |
| `CBLAS_DIAG` | `CblasNonUnit=131`, `CblasUnit=132` | Unit or non-unit diagonal |

---

## General Matrix Multiply

Performs general matrix-matrix multiply: **C = alpha \* op(A) \* op(B) + beta \* C**

Where `op(X)` is one of `X`, `X^T`, or `X^H` depending on the `Trans` parameter.

### cblas_sgemm
Single-precision real general matrix multiply.
```c
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
                 const float beta, float *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major (`CblasRowMajor`) or column-major (`CblasColMajor`) storage |
| `TransA` | `CBLAS_TRANSPOSE` | Operation applied to A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `TransB` | `CBLAS_TRANSPOSE` | Operation applied to B: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `M` | `CBLAS_INT` | Number of rows of op(A) and C |
| `N` | `CBLAS_INT` | Number of columns of op(B) and C |
| `K` | `CBLAS_INT` | Number of columns of op(A) and rows of op(B) |
| `alpha` | `float` | Scalar multiplier for op(A)*op(B) |
| `A` | `const float*` | Matrix A; dimensions depend on TransA |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const float*` | Matrix B; dimensions depend on TransB |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `float` | Scalar multiplier for C |
| `C` | `float*` | Matrix C (M x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_dgemm
Double-precision real general matrix multiply.
```c
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_sgemm` with `double` replacing `float`.

### cblas_cgemm
Single-precision complex general matrix multiply.
```c
void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_sgemm`. Scalars `alpha` and `beta` are pointers to single-precision complex values (`float[2]`). Matrix pointers are `void*` to single-precision complex arrays.

### cblas_zgemm
Double-precision complex general matrix multiply.
```c
void cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_cgemm` with double-precision complex values (`double[2]`).

---

### cblas_sgemmtr
Single-precision real general matrix multiply, triangular result (new in LAPACK 3.12). Only the upper or lower triangular part of C is computed: **C = alpha \* op(A) \* op(B) + beta \* C**
```c
void cblas_sgemmtr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT N,
                 const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
                 const float beta, float *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper`: compute upper triangle; `CblasLower`: compute lower triangle |
| `TransA` | `CBLAS_TRANSPOSE` | Operation applied to A |
| `TransB` | `CBLAS_TRANSPOSE` | Operation applied to B |
| `N` | `CBLAS_INT` | Order of matrix C (N x N) |
| `K` | `CBLAS_INT` | Number of columns of op(A) and rows of op(B) |
| `alpha` | `float` | Scalar multiplier for op(A)*op(B) |
| `A` | `const float*` | Matrix A |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const float*` | Matrix B |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `float` | Scalar multiplier for C |
| `C` | `float*` | Symmetric matrix C (N x N), input/output; only the `Uplo` triangle is referenced and updated |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_dgemmtr
Double-precision real general matrix multiply, triangular result.
```c
void cblas_dgemmtr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_sgemmtr` with `double` replacing `float`.

### cblas_cgemmtr
Single-precision complex general matrix multiply, triangular result.
```c
void cblas_cgemmtr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_sgemmtr`. Scalars and matrices use `void*` for single-precision complex values.

### cblas_zgemmtr
Double-precision complex general matrix multiply, triangular result.
```c
void cblas_zgemmtr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_cgemmtr` with double-precision complex values.

---

## Symmetric Matrix Multiply

Performs symmetric matrix-matrix multiply where A is symmetric:
- **Side=Left:** C = alpha \* A \* B + beta \* C
- **Side=Right:** C = alpha \* B \* A + beta \* C

### cblas_ssymm
Single-precision real symmetric matrix multiply.
```c
void cblas_ssymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *B, const CBLAS_INT ldb, const float beta,
                 float *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Side` | `CBLAS_SIDE` | `CblasLeft`: A*B; `CblasRight`: B*A |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of A is referenced |
| `M` | `CBLAS_INT` | Number of rows of C |
| `N` | `CBLAS_INT` | Number of columns of C |
| `alpha` | `float` | Scalar multiplier |
| `A` | `const float*` | Symmetric matrix A; order M if Side=Left, N if Side=Right |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const float*` | Matrix B (M x N) |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `float` | Scalar multiplier for C |
| `C` | `float*` | Matrix C (M x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_dsymm
Double-precision real symmetric matrix multiply.
```c
void cblas_dsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *B, const CBLAS_INT ldb, const double beta,
                 double *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_ssymm` with `double` replacing `float`.

### cblas_csymm
Single-precision complex symmetric matrix multiply.
```c
void cblas_csymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_ssymm`. Scalars and matrices use `void*` for single-precision complex values. Note: this is symmetric (not Hermitian); for Hermitian, use `cblas_chemm`.

### cblas_zsymm
Double-precision complex symmetric matrix multiply.
```c
void cblas_zsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_csymm` with double-precision complex values.

---

## Hermitian Matrix Multiply

Performs Hermitian matrix-matrix multiply where A is Hermitian (A = A^H):
- **Side=Left:** C = alpha \* A \* B + beta \* C
- **Side=Right:** C = alpha \* B \* A + beta \* C

Complex types only (C and Z prefixes).

### cblas_chemm
Single-precision complex Hermitian matrix multiply.
```c
void cblas_chemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Side` | `CBLAS_SIDE` | `CblasLeft`: A*B; `CblasRight`: B*A |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of Hermitian A is referenced |
| `M` | `CBLAS_INT` | Number of rows of C |
| `N` | `CBLAS_INT` | Number of columns of C |
| `alpha` | `const void*` | Pointer to single-precision complex scalar |
| `A` | `const void*` | Hermitian matrix A; order M if Side=Left, N if Side=Right |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const void*` | Matrix B (M x N) |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `const void*` | Pointer to single-precision complex scalar |
| `C` | `void*` | Matrix C (M x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_zhemm
Double-precision complex Hermitian matrix multiply.
```c
void cblas_zhemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_chemm` with double-precision complex values.

---

## Symmetric Rank-k Update

### syrk - Symmetric Rank-k Update

Performs symmetric rank-k update: **C = alpha \* op(A) \* op(A)^T + beta \* C**

Where C is symmetric and only the `Uplo` triangle is referenced and updated.

### cblas_ssyrk
Single-precision real symmetric rank-k update.
```c
void cblas_ssyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float beta, float *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of C is referenced and updated |
| `Trans` | `CBLAS_TRANSPOSE` | `CblasNoTrans`: C = alpha*A*A^T + beta*C; `CblasTrans`: C = alpha*A^T*A + beta*C |
| `N` | `CBLAS_INT` | Order of matrix C |
| `K` | `CBLAS_INT` | If NoTrans: number of columns of A; if Trans: number of rows of A |
| `alpha` | `float` | Scalar multiplier for A*A^T |
| `A` | `const float*` | Matrix A |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `beta` | `float` | Scalar multiplier for C |
| `C` | `float*` | Symmetric matrix C (N x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_dsyrk
Double-precision real symmetric rank-k update.
```c
void cblas_dsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double beta, double *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_ssyrk` with `double` replacing `float`.

### cblas_csyrk
Single-precision complex symmetric rank-k update.
```c
void cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_ssyrk`. Scalars and matrices use `void*` for single-precision complex values. Note: symmetric (not Hermitian); for Hermitian rank-k update, use `cblas_cherk`.

### cblas_zsyrk
Double-precision complex symmetric rank-k update.
```c
void cblas_zsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_csyrk` with double-precision complex values.

---

### syr2k - Symmetric Rank-2k Update

Performs symmetric rank-2k update: **C = alpha \* op(A) \* op(B)^T + alpha \* op(B) \* op(A)^T + beta \* C**

Where C is symmetric and only the `Uplo` triangle is referenced and updated.

### cblas_ssyr2k
Single-precision real symmetric rank-2k update.
```c
void cblas_ssyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const float alpha, const float *A, const CBLAS_INT lda,
                  const float *B, const CBLAS_INT ldb, const float beta,
                  float *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of C is referenced and updated |
| `Trans` | `CBLAS_TRANSPOSE` | `CblasNoTrans`: C = alpha*A*B^T + alpha*B*A^T + beta*C; `CblasTrans`: C = alpha*A^T*B + alpha*B^T*A + beta*C |
| `N` | `CBLAS_INT` | Order of matrix C |
| `K` | `CBLAS_INT` | If NoTrans: number of columns of A and B; if Trans: number of rows of A and B |
| `alpha` | `float` | Scalar multiplier |
| `A` | `const float*` | Matrix A |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const float*` | Matrix B |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `float` | Scalar multiplier for C |
| `C` | `float*` | Symmetric matrix C (N x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_dsyr2k
Double-precision real symmetric rank-2k update.
```c
void cblas_dsyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const double alpha, const double *A, const CBLAS_INT lda,
                  const double *B, const CBLAS_INT ldb, const double beta,
                  double *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_ssyr2k` with `double` replacing `float`.

### cblas_csyr2k
Single-precision complex symmetric rank-2k update.
```c
void cblas_csyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const void *beta,
                  void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_ssyr2k`. Scalars and matrices use `void*` for single-precision complex values.

### cblas_zsyr2k
Double-precision complex symmetric rank-2k update.
```c
void cblas_zsyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const void *beta,
                  void *C, const CBLAS_INT ldc);
```

**Parameters:** Same structure as `cblas_csyr2k` with double-precision complex values.

---

## Hermitian Rank-k Update

### herk - Hermitian Rank-k Update

Performs Hermitian rank-k update: **C = alpha \* op(A) \* op(A)^H + beta \* C**

Where C is Hermitian and only the `Uplo` triangle is referenced and updated. Note: `alpha` and `beta` are real scalars (not complex).

Complex types only (C and Z prefixes).

### cblas_cherk
Single-precision complex Hermitian rank-k update.
```c
void cblas_cherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const float alpha, const void *A, const CBLAS_INT lda,
                 const float beta, void *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of C is referenced and updated |
| `Trans` | `CBLAS_TRANSPOSE` | `CblasNoTrans`: C = alpha*A*A^H + beta*C; `CblasConjTrans`: C = alpha*A^H*A + beta*C |
| `N` | `CBLAS_INT` | Order of matrix C |
| `K` | `CBLAS_INT` | If NoTrans: number of columns of A; if ConjTrans: number of rows of A |
| `alpha` | `float` | **Real** scalar multiplier for A*A^H |
| `A` | `const void*` | Complex matrix A |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `beta` | `float` | **Real** scalar multiplier for C |
| `C` | `void*` | Hermitian matrix C (N x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_zherk
Double-precision complex Hermitian rank-k update.
```c
void cblas_zherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const double alpha, const void *A, const CBLAS_INT lda,
                 const double beta, void *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_cherk` with `double` replacing `float` for the real scalars `alpha` and `beta`, and double-precision complex arrays for A and C.

---

### her2k - Hermitian Rank-2k Update

Performs Hermitian rank-2k update: **C = alpha \* op(A) \* op(B)^H + conj(alpha) \* op(B) \* op(A)^H + beta \* C**

Where C is Hermitian and only the `Uplo` triangle is referenced and updated. Note: `beta` is a real scalar; `alpha` is complex.

Complex types only (C and Z prefixes).

### cblas_cher2k
Single-precision complex Hermitian rank-2k update.
```c
void cblas_cher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const float beta,
                  void *C, const CBLAS_INT ldc);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper` or `CblasLower` triangle of C is referenced and updated |
| `Trans` | `CBLAS_TRANSPOSE` | `CblasNoTrans`: uses A*B^H + conj(alpha)*B*A^H; `CblasConjTrans`: uses A^H*B + conj(alpha)*B^H*A |
| `N` | `CBLAS_INT` | Order of matrix C |
| `K` | `CBLAS_INT` | If NoTrans: number of columns of A and B; if ConjTrans: number of rows of A and B |
| `alpha` | `const void*` | Pointer to **complex** scalar |
| `A` | `const void*` | Complex matrix A |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `const void*` | Complex matrix B |
| `ldb` | `CBLAS_INT` | Leading dimension of B |
| `beta` | `float` | **Real** scalar multiplier for C |
| `C` | `void*` | Hermitian matrix C (N x N), input/output |
| `ldc` | `CBLAS_INT` | Leading dimension of C |

### cblas_zher2k
Double-precision complex Hermitian rank-2k update.
```c
void cblas_zher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const double beta,
                  void *C, const CBLAS_INT ldc);
```

**Parameters:** Same as `cblas_cher2k` with `double` replacing `float` for the real scalar `beta`, and double-precision complex values for `alpha`, A, B, and C.

---

## Triangular Matrix Operations

### trmm - Triangular Matrix Multiply

Performs triangular matrix-matrix multiply:
- **Side=Left:** B = alpha \* op(A) \* B
- **Side=Right:** B = alpha \* B \* op(A)

Where A is a triangular matrix.

### cblas_strmm
Single-precision real triangular matrix multiply.
```c
void cblas_strmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 float *B, const CBLAS_INT ldb);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Side` | `CBLAS_SIDE` | `CblasLeft`: op(A)*B; `CblasRight`: B*op(A) |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper`: A is upper triangular; `CblasLower`: A is lower triangular |
| `TransA` | `CBLAS_TRANSPOSE` | Operation applied to A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CBLAS_DIAG` | `CblasNonUnit`: diagonal of A is used; `CblasUnit`: diagonal of A is assumed to be 1 |
| `M` | `CBLAS_INT` | Number of rows of B |
| `N` | `CBLAS_INT` | Number of columns of B |
| `alpha` | `float` | Scalar multiplier |
| `A` | `const float*` | Triangular matrix A; order M if Side=Left, N if Side=Right |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `float*` | Matrix B (M x N), input/output; overwritten with result |
| `ldb` | `CBLAS_INT` | Leading dimension of B |

### cblas_dtrmm
Double-precision real triangular matrix multiply.
```c
void cblas_dtrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 double *B, const CBLAS_INT ldb);
```

**Parameters:** Same as `cblas_strmm` with `double` replacing `float`.

### cblas_ctrmm
Single-precision complex triangular matrix multiply.
```c
void cblas_ctrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
```

**Parameters:** Same structure as `cblas_strmm`. Scalar `alpha` is a pointer to single-precision complex value. Matrix pointers are `void*` to single-precision complex arrays.

### cblas_ztrmm
Double-precision complex triangular matrix multiply.
```c
void cblas_ztrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
```

**Parameters:** Same structure as `cblas_ctrmm` with double-precision complex values.

---

### trsm - Triangular Solve (Matrix)

Solves a triangular matrix equation:
- **Side=Left:** op(A) \* X = alpha \* B
- **Side=Right:** X \* op(A) = alpha \* B

Where A is a triangular matrix. B is overwritten with the solution X.

### cblas_strsm
Single-precision real triangular solve.
```c
void cblas_strsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 float *B, const CBLAS_INT ldb);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `CBLAS_LAYOUT` | Row-major or column-major storage |
| `Side` | `CBLAS_SIDE` | `CblasLeft`: solve op(A)*X = alpha*B; `CblasRight`: solve X*op(A) = alpha*B |
| `Uplo` | `CBLAS_UPLO` | `CblasUpper`: A is upper triangular; `CblasLower`: A is lower triangular |
| `TransA` | `CBLAS_TRANSPOSE` | Operation applied to A: `CblasNoTrans`, `CblasTrans`, or `CblasConjTrans` |
| `Diag` | `CBLAS_DIAG` | `CblasNonUnit`: diagonal of A is used; `CblasUnit`: diagonal of A is assumed to be 1 |
| `M` | `CBLAS_INT` | Number of rows of B |
| `N` | `CBLAS_INT` | Number of columns of B |
| `alpha` | `float` | Scalar multiplier for B |
| `A` | `const float*` | Triangular matrix A; order M if Side=Left, N if Side=Right |
| `lda` | `CBLAS_INT` | Leading dimension of A |
| `B` | `float*` | Matrix B (M x N), input/output; overwritten with solution X |
| `ldb` | `CBLAS_INT` | Leading dimension of B |

### cblas_dtrsm
Double-precision real triangular solve.
```c
void cblas_dtrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 double *B, const CBLAS_INT ldb);
```

**Parameters:** Same as `cblas_strsm` with `double` replacing `float`.

### cblas_ctrsm
Single-precision complex triangular solve.
```c
void cblas_ctrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
```

**Parameters:** Same structure as `cblas_strsm`. Scalar `alpha` is a pointer to single-precision complex value. Matrix pointers are `void*` to single-precision complex arrays.

### cblas_ztrsm
Double-precision complex triangular solve.
```c
void cblas_ztrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
```

**Parameters:** Same structure as `cblas_ctrsm` with double-precision complex values.

---

## Error Handler

### cblas_xerbla
CBLAS error handler. Called internally when an input parameter has an invalid value. Can be overridden by the user (weak symbol when `HAS_ATTRIBUTE_WEAK_SUPPORT` is defined).
```c
void cblas_xerbla(CBLAS_INT p, const char *rout, const char *form, ...);
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `CBLAS_INT` | Position of the invalid parameter in the argument list |
| `rout` | `const char*` | Name of the routine that called `xerbla` |
| `form` | `const char*` | Format string for the error message (printf-style) |
| `...` | variadic | Additional arguments for the format string |

---

## Precision Prefix Summary

| Prefix | Type | Scalar Type | Matrix Element |
|--------|------|-------------|----------------|
| `s` | Single-precision real | `float` | `float` |
| `d` | Double-precision real | `double` | `double` |
| `c` | Single-precision complex | `void*` (float[2]) | `void*` (float[2]) |
| `z` | Double-precision complex | `void*` (double[2]) | `void*` (double[2]) |

## Complete Function Index

| Function | Operation | Precision |
|----------|-----------|-----------|
| `cblas_sgemm` | General matrix multiply | single real |
| `cblas_dgemm` | General matrix multiply | double real |
| `cblas_cgemm` | General matrix multiply | single complex |
| `cblas_zgemm` | General matrix multiply | double complex |
| `cblas_sgemmtr` | General matrix multiply, triangular result | single real |
| `cblas_dgemmtr` | General matrix multiply, triangular result | double real |
| `cblas_cgemmtr` | General matrix multiply, triangular result | single complex |
| `cblas_zgemmtr` | General matrix multiply, triangular result | double complex |
| `cblas_ssymm` | Symmetric matrix multiply | single real |
| `cblas_dsymm` | Symmetric matrix multiply | double real |
| `cblas_csymm` | Symmetric matrix multiply | single complex |
| `cblas_zsymm` | Symmetric matrix multiply | double complex |
| `cblas_chemm` | Hermitian matrix multiply | single complex |
| `cblas_zhemm` | Hermitian matrix multiply | double complex |
| `cblas_ssyrk` | Symmetric rank-k update | single real |
| `cblas_dsyrk` | Symmetric rank-k update | double real |
| `cblas_csyrk` | Symmetric rank-k update | single complex |
| `cblas_zsyrk` | Symmetric rank-k update | double complex |
| `cblas_ssyr2k` | Symmetric rank-2k update | single real |
| `cblas_dsyr2k` | Symmetric rank-2k update | double real |
| `cblas_csyr2k` | Symmetric rank-2k update | single complex |
| `cblas_zsyr2k` | Symmetric rank-2k update | double complex |
| `cblas_cherk` | Hermitian rank-k update | single complex |
| `cblas_zherk` | Hermitian rank-k update | double complex |
| `cblas_cher2k` | Hermitian rank-2k update | single complex |
| `cblas_zher2k` | Hermitian rank-2k update | double complex |
| `cblas_strmm` | Triangular matrix multiply | single real |
| `cblas_dtrmm` | Triangular matrix multiply | double real |
| `cblas_ctrmm` | Triangular matrix multiply | single complex |
| `cblas_ztrmm` | Triangular matrix multiply | double complex |
| `cblas_strsm` | Triangular solve (matrix) | single real |
| `cblas_dtrsm` | Triangular solve (matrix) | double real |
| `cblas_ctrsm` | Triangular solve (matrix) | single complex |
| `cblas_ztrsm` | Triangular solve (matrix) | double complex |
| `cblas_xerbla` | Error handler | N/A |
