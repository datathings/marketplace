# cuBLAS API

## Table of Contents
1. [Handle and Stream](#handle-and-stream)
2. [Level-1: Vector Operations](#level-1-vector-operations)
3. [Level-2: Matrix-Vector Operations](#level-2-matrix-vector-operations)
4. [Level-3: Matrix-Matrix Operations (GEMM)](#level-3-matrix-matrix-operations-gemm)
5. [Batched and Strided GEMM](#batched-and-strided-gemm)
6. [Grouped Batched GEMM](#grouped-batched-gemm)
7. [Triangular Solvers (TRSM)](#triangular-solvers-trsm)
8. [cuBLASLt (Lightweight / Tensor Core)](#cublaslt-lightweight--tensor-core)
9. [Error Handling](#error-handling)

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
**Description:** Dot product `result = x . y` (host or device pointer).

### `cublas<t>nrm2(handle, n, x, incx, result) -> cublasStatus_t`
**Description:** Euclidean norm `||x||_2`.

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
- `m, n, k` — dimensions: C is m x n, A is m x k, B is k x n (after transpose).
- `lda, ldb, ldc` — leading dimensions (column-major, so >= number of rows).
- Precision variants: `S`=float, `D`=double, `H`=half, `C`=complex float, `Z`=complex double.

**Example (double precision):**
```cpp
const double alpha = 1.0, beta = 0.0;
// C = A * B  (column-major layout)
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

## Grouped Batched GEMM

### `cublas<t>gemmGroupedBatched(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, d_Aarray, lda_array, d_Barray, ldb_array, beta_array, d_Carray, ldc_array, group_count, group_size) -> cublasStatus_t`
**Description:** Executes multiple groups of batched GEMMs where each group can have different dimensions (m, n, k), leading dimensions, transpose operations, and alpha/beta scalars. Matrices within the same group share the same parameters.
**Parameters:**
- `transa_array/transb_array` — array of `group_count` transpose operations
- `m_array/n_array/k_array` — array of `group_count` dimension sets
- `alpha_array/beta_array` — array of `group_count` scalar values
- `d_Aarray/d_Barray/d_Carray` — device arrays of pointers (total length = sum of group_size[])
- `lda_array/ldb_array/ldc_array` — array of `group_count` leading dimensions
- `group_count` — number of groups
- `group_size` — array of batch sizes per group

**Example:**
```cpp
CUBLAS_CHECK(cublasDgemmGroupedBatched(
    cublasH, transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, d_A_array, lda_array, d_B_array, ldb_array, beta_array,
    d_C_array, ldc_array, group_count, group_size));
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

### Grouped GEMM with device-side dimensions

The grouped cuBLASLt path runs `batchCount` GEMMs whose per-group `M, N, K` and leading
dimensions live **on the device** (as arrays), so the shapes need not be known at launch time
(MoE / variable-length batching). Layouts are built with `cublasLtGroupedMatrixLayoutCreate`
instead of `cublasLtMatrixLayoutCreate`.

### `cublasLtGroupedMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, int batchCount, const void *rowsArrayDev, const void *colsArrayDev, const void *ldArrayDev) -> cublasStatus_t`
**Description:** Grouped matrix layout. `rowsArrayDev`, `colsArrayDev`, `ldArrayDev` are **device**
pointers to `int64_t[batchCount]` arrays of row counts, column counts, and leading dimensions.
Rows/cols are swapped per-operand based on transpose, e.g. for A:
`rows = (transa==CUBLAS_OP_N ? mArrayDev : kArrayDev)`, `cols = (transa==CUBLAS_OP_N ? kArrayDev : mArrayDev)`.

**Because M/N/K are on the device, supply averages to the heuristic via preference attributes:**
- `CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS` — average M (`int64_t`)
- `CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS` — average N (`int64_t`)
- `CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM` — average K (`int64_t`)

**Pointer mode.** Default (host) pointer mode shares one `alpha`/`beta` scalar across all groups.
To give each group its own device-resident `alpha`/`beta` (as in the H/S/H sample), set:
```cpp
cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
int64_t stride = 1;
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &stride, sizeof(stride));
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,  &stride, sizeof(stride));
```
Then `alpha`/`beta` passed to `cublasLtMatmul` are `const float *const *` device pointer arrays.
The A/B/C/D pointers are always device arrays of per-matrix pointers; `cublasLtMatmul` is called
once with them and a stream (or `0`), e.g.:
```cpp
cublasLtMatmul(ltHandle, op, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc,
               (void *)D, Ddesc, &heur.algo, workspace, workspaceSize, /*stream*/ 0);
```
> These grouped GEMM entry points are marked **EXPERIMENTAL** in the samples and may change.
> Samples: `CUDALibrarySamples/cuBLASLt/LtHSHgemmGroupedSimple/` (FP16 in/out, FP32 compute,
> device alpha/beta), `LtFp8gemmGroupedSimple/`.

### FP8 / MXFP8 / NVFP4 block-scaled grouped GEMM

Low-precision grouped GEMMs use the same flow but add per-operand scale factors. Set the scale
**mode** and scale **pointer** on the matmul descriptor for A and B:
```cpp
cublasLtMatmulMatrixScale_t aMode = /* see table */, bMode = /* see table */;
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &aMode, sizeof(aMode));
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &bMode, sizeof(bMode));
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
```
`a_scale`/`b_scale` are `const <scaleT> *const *` device pointer arrays (one scale buffer per group).
All four variants create the operation descriptor with `CUBLAS_COMPUTE_32F` + scale type `CUDA_R_32F`,
use `__nv_bfloat16` (`CUDA_R_16BF`) C/D, host `alpha`/`beta`, and pick the layout element type by
operand precision.

| Sample | A/B element type (layout) | Scale mode (A, B) | Scale buffer element type |
|--------|---------------------------|-------------------|---------------------------|
| `LtFp8gemmGroupedSimple` | `CUDA_R_8F_E4M3` (`__nv_fp8_e4m3`) | `CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F` (both) | `float` |
| `LtBlk128x128Fp8gemmGroupedSimple` | `CUDA_R_8F_E4M3` | A: `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F`, B: `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F` | `float` |
| `LtMxfp8gemmGroupedSimple` | `CUDA_R_8F_E4M3` | `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` (both) | `__nv_fp8_e8m0` |
| `LtNvfp4gemmGroupedSimple` | `CUDA_R_4F_E2M1` (`__nv_fp4_e2m1`) | `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` (both) | `__nv_fp8_e4m3` |

Notes:
- `PER_BATCH_SCALAR_32F` = one `float` scalar per group; `VEC128`/`BLK128x128` = block scaling
  (128-element vector for A, 128×128 tile for B); `VEC32_UE8M0` = MXFP8 (32-element E8M0 blocks);
  `VEC16_UE4M3` = NVFP4 (16-element E4M3 blocks).
- Only A and B are scaled here (C/D are bf16). `cublasLtMatmulAlgoGetHeuristic` returning 0 results
  means the (mode, shapes, arch) combination is unsupported — the samples raise `CUBLAS_STATUS_NOT_SUPPORTED`.

### Green-context GEMM (`cublasLtMatmulAlgoGetHeuristicForStream`)

To confine a matmul to a subset of SMs, run it on a stream created from a CUDA **green context** and
let cuBLASLt pick an algorithm sized for that context via the stream-aware heuristic.

### `cublasLtMatmulAlgoGetHeuristicForStream(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t *heuristicResultsArray, int *returnAlgoCount, cudaStream_t stream) -> cublasStatus_t`
**Description:** Like `cublasLtMatmulAlgoGetHeuristic`, but the returned algorithms account for the
SM resources of the green context bound to `stream`. Pass the **same** `stream` to `cublasLtMatmul`.

**Green context creation (CUDA runtime API, from `LtSgemmGreenContext`):**
```cpp
cudaExecutionContext_t greenCtx = 0;   // runtime green context handle
cudaStream_t stream = 0;
cudaDevResource input, smPartition;
cudaDevResourceDesc_t smPartitionDesc;
unsigned int nbGroups = 1;
cudaDeviceGetDevResource(device, &input, cudaDevResourceTypeSm);
cudaDevSmResourceSplitByCount(&smPartition, &nbGroups, &input, NULL, 0, minGreenContextSmCount);
cudaDevResourceGenerateDesc(&smPartitionDesc, &smPartition, 1);
cudaGreenCtxCreate(&greenCtx, smPartitionDesc, device, 0);
cudaExecutionCtxStreamCreate(&stream, greenCtx, 0, 0);  // non-blocking stream on green ctx
// sync from the primary-context stream via an event before enqueuing work
...
cublasLtMatmulAlgoGetHeuristicForStream(ltHandle, op, Adesc, Bdesc, Cdesc, Cdesc,
                                        preference, 1, &heur, &returnedResults, stream);
cublasLtMatmul(ltHandle, op, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
               &heur.algo, workspace, workspaceSize, stream);
cudaStreamSynchronize(stream);
// cleanup: cudaStreamDestroy(stream); cudaExecutionCtxDestroy(greenCtx);
```
Streams created on a green context are non-blocking and require explicit event synchronization with
the primary context's stream. Sample: `CUDALibrarySamples/cuBLASLt/LtSgemmGreenContext/`.

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
