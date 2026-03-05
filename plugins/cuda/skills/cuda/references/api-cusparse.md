# cuSPARSE API

## Table of Contents
1. [Handle Management](#handle-management)
2. [Sparse Matrix Descriptors](#sparse-matrix-descriptors)
3. [Dense Matrix/Vector Descriptors](#dense-matrixvector-descriptors)
4. [SpMM — Sparse × Dense Matrix Multiply](#spmm--sparse--dense-matrix-multiply)
5. [SpMV — Sparse Matrix × Dense Vector](#spmv--sparse-matrix--dense-vector)
6. [SpGEMM — Sparse × Sparse](#spgemm--sparse--sparse)
7. [Format Conversion](#format-conversion)
8. [Error Handling](#error-handling)

---

## Handle Management

### `cusparseCreate(cusparseHandle_t *handle) -> cusparseStatus_t`
**Description:** Creates a cuSPARSE context.

### `cusparseDestroy(cusparseHandle_t handle) -> cusparseStatus_t`
**Description:** Releases cuSPARSE resources.

### `cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) -> cusparseStatus_t`
**Description:** Binds a stream for async operations.

**Setup pattern:**
```c
cusparseHandle_t handle;
CHECK_CUSPARSE(cusparseCreate(&handle));
// ... operations ...
CHECK_CUSPARSE(cusparseDestroy(handle));
```

```c
#define CHECK_CUSPARSE(func) do {                                           \
    cusparseStatus_t status = (func);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        printf("CUSPARSE error %s at line %d\n",                           \
               cusparseGetErrorString(status), __LINE__);                  \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)
```

---

## Sparse Matrix Descriptors

cuSPARSE uses opaque descriptors created from raw device pointers.

### `cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *csrRowOffsets, void *csrColInd, void *csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) -> cusparseStatus_t`
**Description:** Creates a CSR (Compressed Sparse Row) matrix descriptor.
**Parameters:**
- `rows/cols` — matrix dimensions
- `nnz` — number of non-zeros
- `csrRowOffsets` — row pointer array, length `rows+1`
- `csrColInd` — column index array, length `nnz`
- `csrValues` — values array, length `nnz`
- Common types: `CUSPARSE_INDEX_32I`, `CUSPARSE_INDEX_BASE_ZERO`, `CUDA_R_32F`

**Example:**
```c
cusparseSpMatDescr_t matA;
CHECK_CUSPARSE(cusparseCreateCsr(&matA,
    A_num_rows, A_num_cols, A_nnz,
    dA_csrOffsets, dA_columns, dA_values,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
```

### `cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *cooRowInd, void *cooColInd, void *cooValues, ...) -> cusparseStatus_t`
**Description:** Creates a COO (Coordinate) format descriptor.

### `cusparseCreateBsr(...)` — Block CSR format (blocked non-zeros).

### `cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) -> cusparseStatus_t`
**Description:** Frees the sparse matrix descriptor (not the underlying device memory).

### `cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void *data, size_t dataSize) -> cusparseStatus_t`
**Description:** Sets attributes like `CUSPARSE_SPMAT_FILL_MODE` or `CUSPARSE_SPMAT_DIAG_TYPE`.

---

## Dense Matrix/Vector Descriptors

### `cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void *values, cudaDataType valueType, cusparseOrder_t order) -> cusparseStatus_t`
**Description:** Creates a dense matrix descriptor. `order` is `CUSPARSE_ORDER_COL` (column-major) or `CUSPARSE_ORDER_ROW`.

### `cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size, void *values, cudaDataType valueType) -> cusparseStatus_t`
**Description:** Creates a dense vector descriptor.

### `cusparseDestroyDnMat` / `cusparseDestroyDnVec`
**Description:** Destroy descriptors.

---

## SpMM — Sparse × Dense Matrix Multiply

### Workflow: Buffer Size → Preprocess → Execute

```c
// y = alpha * A * x + beta * y
// A: sparse CSR, B: dense matrix, C: dense matrix

void *dBuffer = NULL;
size_t bufferSize = 0;

// 1. Query buffer size
CHECK_CUSPARSE(cusparseSpMM_bufferSize(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
    CUSPARSE_OPERATION_NON_TRANSPOSE,   // opB
    &alpha, matA, matB, &beta, matC,
    CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT,
    &bufferSize));
CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

// 2. Optional preprocess (builds internal data structures)
CHECK_CUSPARSE(cusparseSpMM_preprocess(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

// 3. Execute
CHECK_CUSPARSE(cusparseSpMM(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, matB, &beta, matC,
    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

// Cleanup
CHECK_CUSPARSE(cusparseDestroySpMat(matA));
CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
CHECK_CUSPARSE(cusparseDestroy(handle));
CHECK_CUDA(cudaFree(dBuffer));
```

**Algorithm options:** `CUSPARSE_SPMM_ALG_DEFAULT`, `CUSPARSE_SPMM_CSR_ALG1`, `CUSPARSE_SPMM_CSR_ALG2`, `CUSPARSE_SPMM_BLOCKED_ELL_ALG1`.

---

## SpMV — Sparse Matrix × Dense Vector

### `cusparseSpMV_bufferSize`, `cusparseSpMV`
**Description:** Computes `y = alpha * A * x + beta * y` for sparse A and dense vectors x, y.
**Example:**
```c
size_t bufSize;
CHECK_CUSPARSE(cusparseSpMV_bufferSize(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize));
void *buf; cudaMalloc(&buf, bufSize);

CHECK_CUSPARSE(cusparseSpMV(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
```

---

## SpGEMM — Sparse × Sparse

### `cusparseSpGEMM_*` API (multi-step)
**Description:** Sparse matrix product `C = A * B` where all matrices are sparse.
**Steps:** `cusparseSpGEMM_createDescr` → `cusparseSpGEMM_workEstimation` → `cusparseSpGEMM_compute` → `cusparseSpGEMM_copy` → `cusparseSpGEMM_destroyDescr`.
See sample: `CUDALibrarySamples/cuSPARSE/spgemm/`.

---

## Format Conversion

### `cusparseCsr2cscEx2` — CSR to CSC (column-sorted) conversion
**Description:** Transposes storage; useful for column access patterns.

### `cusparseCreateCsr` with `cusparseSpConvert`
**Description:** Convert between COO, CSR, BSR formats.
**Common pattern for COO → CSR:**
```c
// Sort COO by row, then use cusparseXcoo2csr
cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, dCooRows, dCooCols, &bufSize);
// ... sort then convert
cusparseXcoo2csr(handle, dCooRows, nnz, m, dCsrOffsets, CUSPARSE_INDEX_BASE_ZERO);
```

---

## Error Handling

```c
#define CHECK_CUSPARSE(func) do {                                           \
    cusparseStatus_t status = (func);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        printf("CUSPARSE error '%s' at %s:%d\n",                           \
               cusparseGetErrorString(status), __FILE__, __LINE__);        \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)
```

**Common errors:** `CUSPARSE_STATUS_NOT_INITIALIZED`, `CUSPARSE_STATUS_ALLOC_FAILED`, `CUSPARSE_STATUS_INVALID_VALUE`, `CUSPARSE_STATUS_ARCH_MISMATCH`, `CUSPARSE_STATUS_INSUFFICIENT_RESOURCES`.

**Header:** `#include <cusparse.h>` — link: `-lcusparse`.
