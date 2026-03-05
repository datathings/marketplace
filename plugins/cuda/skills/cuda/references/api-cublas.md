# cuBLAS API

## Table of Contents
1. [Handle and Stream](#handle-and-stream)
2. [Level-1: Vector Operations](#level-1-vector-operations)
3. [Level-2: Matrix-Vector Operations](#level-2-matrix-vector-operations)
4. [Level-3: Matrix-Matrix Operations (GEMM)](#level-3-matrix-matrix-operations-gemm)
5. [Batched and Strided GEMM](#batched-and-strided-gemm)
6. [Triangular Solvers (TRSM)](#triangular-solvers-trsm)
7. [cuBLASLt (Lightweight / Tensor Core)](#cublaslt-lightweight--tensor-core)
8. [Error Handling](#error-handling)

---

## Handle and Stream

### `cublasCreate(cublasHandle_t *handle) -> cublasStatus_t`
**Description:** Creates a cuBLAS context. One handle per thread/GPU recommended.

### `cublasDestroy(cublasHandle_t handle) -> cublasStatus_t`
**Description:** Destroys cuBLAS context and releases resources.

### `cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) -> cublasStatus_t`
**Description:** Binds a CUDA stream to the handle for async execution.

### `cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) -> cublasStatus_t`
**Description:** Controls use of Tensor Cores. `CUBLAS_TENSOR_OP_MATH` enables them (Volta+).

**Full setup pattern:**
```cpp
cublasHandle_t handle;
cudaStream_t stream;
CUBLAS_CHECK(cublasCreate(&handle));
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
CUBLAS_CHECK(cublasSetStream(handle, stream));
// ... operations ...
CUBLAS_CHECK(cublasDestroy(handle));
CUDA_CHECK(cudaStreamDestroy(stream));
```

```c
#define CUBLAS_CHECK(err) do {                                             \
    cublasStatus_t _s = (err);                                             \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", _s,                 \
                __FILE__, __LINE__);                                       \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)
```

---

## Level-1: Vector Operations

### `cublas<t>axpy(handle, n, alpha, x, incx, y, incy) -> cublasStatus_t`
**Description:** `y = alpha*x + y`. `<t>` is `S`(float), `D`(double), `C`(complex float), `Z`(complex double).
**Example:**
```cpp
float alpha = 2.0f;
cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);  // y = 2*x + y
```

### `cublas<t>dot(handle, n, x, incx, y, incy, result) -> cublasStatus_t`
**Description:** Dot product `result = x · y` (host or device pointer).

### `cublas<t>nrm2(handle, n, x, incx, result) -> cublasStatus_t`
**Description:** Euclidean norm `||x||₂`.

### `cublas<t>scal(handle, n, alpha, x, incx) -> cublasStatus_t`
**Description:** `x = alpha * x`.

---

## Level-2: Matrix-Vector Operations

### `cublas<t>gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) -> cublasStatus_t`
**Description:** `y = alpha * op(A) * x + beta * y`. `op(A)` is `CUBLAS_OP_N` (none), `_T` (transpose), `_C` (conjugate transpose).
**Note:** cuBLAS uses **column-major** storage. For row-major C arrays, transpose the operation.
**Example:**
```cpp
double alpha = 1.0, beta = 0.0;
// y = A * x  (m x n matrix, m-vector result)
cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, lda, d_x, 1, &beta, d_y, 1);
```

---

## Level-3: Matrix-Matrix Operations (GEMM)

### `cublas<t>gemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) -> cublasStatus_t`
**Description:** `C = alpha * op(A) * op(B) + beta * C`.
- `m, n, k` — dimensions: C is m×n, A is m×k, B is k×n (after transpose).
- `lda, ldb, ldc` — leading dimensions (column-major, so ≥ number of rows).
- Precision variants: `S`=float, `D`=double, `H`=half, `C`=complex float, `Z`=complex double.

**Example (double precision):**
```cpp
const double alpha = 1.0, beta = 0.0;
// C = A * B  (2x2 matrices, column-major)
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha, d_A, lda,
                    d_B, ldb,
            &beta,  d_C, ldc);
cudaStreamSynchronize(stream);
```

### `cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo) -> cublasStatus_t`
**Description:** Mixed-precision GEMM. Allows FP16 inputs with FP32 accumulation.
**Key types:** `CUDA_R_16F` (FP16), `CUDA_R_32F` (FP32), `CUBLAS_COMPUTE_32F_FAST_16F` (Tensor Core).
**Example:**
```cpp
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             m, n, k,
             &alpha, d_A, CUDA_R_16F, lda,
                     d_B, CUDA_R_16F, ldb,
             &beta,  d_C, CUDA_R_32F, ldc,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

---

## Batched and Strided GEMM

### `cublas<t>gemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount) -> cublasStatus_t`
**Description:** Executes `batchCount` independent GEMMs. `Aarray`, `Barray`, `Carray` are device arrays of pointers.

### `cublas<t>gemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount) -> cublasStatus_t`
**Description:** Contiguous-memory batched GEMM. `strideA = lda * k` for non-transposed A.
**Example:**
```cpp
long long strideA = lda * k, strideB = ldb * n, strideC = ldc * n;
cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          m, n, k,
                          &alpha, d_A, lda, strideA,
                                  d_B, ldb, strideB,
                          &beta,  d_C, ldc, strideC,
                          batchCount);
```

---

## Triangular Solvers (TRSM)

### `cublas<t>trsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb) -> cublasStatus_t`
**Description:** Solves triangular system `op(A) * X = alpha * B` (left) or `X * op(A) = alpha * B` (right). Result overwrites B.
**Parameters:**
- `side` — `CUBLAS_SIDE_LEFT` or `CUBLAS_SIDE_RIGHT`
- `uplo` — `CUBLAS_FILL_MODE_UPPER` or `LOWER`
- `diag` — `CUBLAS_DIAG_NON_UNIT` or `UNIT`

---

## cuBLASLt (Lightweight / Tensor Core)

cuBLASLt provides fine-grained control over Tensor Core operations. Use it when you need explicit FP8/FP4, custom epilogues, or algorithm search.

**Key objects:** `cublasLtHandle_t`, `cublasLtMatmulDesc_t`, `cublasLtMatrixLayout_t`, `cublasLtMatmulPreference_t`.

**Workflow:**
```cpp
cublasLtHandle_t ltHandle;
cublasLtCreate(&ltHandle);

cublasLtMatmulDesc_t opDesc;
cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));

cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, lda);
cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb);
cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);

cublasLtMatmulPreference_t pref;
cublasLtMatmulPreferenceCreate(&pref);
cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     &wsSize, sizeof(wsSize));

cublasLtMatmulHeuristicResult_t hResult;
int returnedAlgos;
cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                pref, 1, &hResult, &returnedAlgos);

cublasLtMatmul(ltHandle, opDesc, &alpha, d_A, Adesc, d_B, Bdesc,
               &beta, d_C, Cdesc, d_C, Cdesc,
               &hResult.algo, workspace, wsSize, stream);

// Cleanup
cublasLtMatmulDescDestroy(opDesc);
cublasLtMatrixLayoutDestroy(Adesc); /* etc. */
cublasLtDestroy(ltHandle);
```

---

## Error Handling

```c
#define CUBLAS_CHECK(err) do {                                        \
    cublasStatus_t _s = (err);                                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n",                \
                _s, __FILE__, __LINE__);                              \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)
```

**Common status codes:** `CUBLAS_STATUS_SUCCESS`, `CUBLAS_STATUS_NOT_INITIALIZED`, `CUBLAS_STATUS_ALLOC_FAILED`, `CUBLAS_STATUS_INVALID_VALUE`, `CUBLAS_STATUS_ARCH_MISMATCH` (Tensor Cores not available), `CUBLAS_STATUS_EXECUTION_FAILED`.
