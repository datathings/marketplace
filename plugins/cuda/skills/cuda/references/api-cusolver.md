# cuSolver API

## Table of Contents
1. [cusolverDn — Dense Solver](#cusolverdn--dense-solver)
2. [LU Factorization (getrf / getrs)](#lu-factorization-getrf--getrs)
3. [QR Factorization (geqrf / ormqr)](#qr-factorization-geqrf--ormqr)
4. [Eigenvalue Decomposition (syevd)](#eigenvalue-decomposition-syevd)
5. [SVD (gesvd)](#svd-gesvd)
6. [cusolverDn Generic API (Xgetrf etc.)](#cusolverdn-generic-api-xgetrf-etc)
7. [Error Handling](#error-handling)

---

## cusolverDn — Dense Solver

### Handle Lifecycle

```cpp
cusolverDnHandle_t cusolverH;
cudaStream_t stream;

CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

// ... operations ...

CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
CUDA_CHECK(cudaStreamDestroy(stream));
```

```cpp
#define CUSOLVER_CHECK(err) do {                                        \
    cusolverStatus_t _s = (err);                                        \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuSolver error %d at %s:%d\n",                \
                _s, __FILE__, __LINE__);                                \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)
```

**Note:** cuSolver uses **column-major** (Fortran) storage; C row-major arrays must be transposed or reinterpreted.

---

## LU Factorization (getrf / getrs)

Computes `P*A = L*U` then solves `A*X = B`.

### `cusolverDn<t>getrf_bufferSize(handle, m, n, A, lda, Lwork) -> cusolverStatus_t`
**Description:** Queries workspace size (in elements of type `t`).

### `cusolverDn<t>getrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo) -> cusolverStatus_t`
**Description:** LU factorization in-place. `devIpiv` receives pivot indices; `devInfo` receives singularity info.

### `cusolverDn<t>getrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) -> cusolverStatus_t`
**Description:** Solves the system using the factored A from `getrf`. Result overwrites B.

**Full example (double precision):**
```cpp
int m = 3, lda = m, ldb = m;
double *d_A, *d_B;
int *d_Ipiv, *d_info;
// ... allocate and copy A, B ...

int lwork;
CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
double *d_work;
CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

// Factorize
CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));

// Solve A*X = B
CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info));
CUDA_CHECK(cudaStreamSynchronize(stream));

int info;
CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
if (info != 0) fprintf(stderr, "getrf/getrs failed: info=%d\n", info);
```

**Pivot-free variant:** Pass `NULL` for `devIpiv` to skip pivoting (faster but numerically unstable for non-diagonally-dominant matrices).

---

## QR Factorization (geqrf / ormqr)

Computes `A = Q*R`.

### `cusolverDn<t>geqrf_bufferSize(handle, m, n, A, lda, lwork) -> cusolverStatus_t`
**Description:** Query workspace.

### `cusolverDn<t>geqrf(handle, m, n, A, lda, TAU, Workspace, lwork, devInfo) -> cusolverStatus_t`
**Description:** QR factorization. Factored A and Householder reflectors (TAU) stored in A.

### `cusolverDn<t>ormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo) -> cusolverStatus_t`
**Description:** Applies Q (or Q^T) from geqrf to matrix C.

---

## Eigenvalue Decomposition (syevd)

Symmetric/Hermitian eigenvalue decomposition. Computes `A = V * diag(W) * V^T`.

### `cusolverDn<t>syevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) -> cusolverStatus_t`
**Description:** Query workspace. `jobz = CUSOLVER_EIG_MODE_VECTOR` (eigenvectors + values) or `NOVECTOR`.

### `cusolverDn<t>syevd(handle, jobz, uplo, n, A, lda, W, Work, lwork, devInfo) -> cusolverStatus_t`
**Description:** Eigendecomposition. `W` receives eigenvalues in ascending order; `A` is overwritten with eigenvectors if `jobz = VECTOR`.

**Example:**
```cpp
int m = 3, lda = m;
cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;

int lwork;
CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork));
double *d_work;
CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

CUSOLVER_CHECK(cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, d_info));
CUDA_CHECK(cudaStreamSynchronize(stream));
// d_W: eigenvalues (ascending), d_A: eigenvectors (columns)
```

**Jacobi variant (`syevj`):** Iterative Jacobi method; better for small matrices and batched problems. Use `cusolverDnCreateSyevjInfo` to configure tolerance and max sweeps.

**Batched:** `cusolverDnDsyevjBatched` solves multiple independent eigenvalue problems.

---

## SVD (gesvd)

Computes `A = U * diag(S) * V^T`. A is m×n.

### `cusolverDn<t>gesvd_bufferSize(handle, m, n, lwork) -> cusolverStatus_t`

### `cusolverDn<t>gesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, lwork, rwork, devInfo) -> cusolverStatus_t`
**Parameters:**
- `jobu` — `'A'` (all columns of U), `'S'` (leading min(m,n) columns), `'N'` (skip U)
- `jobvt` — `'A'`, `'S'`, or `'N'` for V^T
- `S` — singular values (descending order)
- `rwork` — real workspace for complex types (NULL for real)

**Example:**
```cpp
signed char jobu = 'A', jobvt = 'A';
int lwork;
CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));
double *d_work;
CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt,
    m, n, d_A, lda, d_S, d_U, ldu, d_VT, ldvt,
    d_work, lwork, NULL /* rwork */, d_info));
CUDA_CHECK(cudaStreamSynchronize(stream));
// d_S: singular values, d_U: left singular vectors, d_VT: right singular vectors (transposed)
```

**Randomized SVD (`Xgesvdr`):** Faster approximate SVD for low-rank approximations.

---

## cusolverDn Generic API (Xgetrf etc.)

For mixed-precision and 64-bit index support, use the generic `X` prefix API (CUDA 11+):
`cusolverDnXgetrf`, `cusolverDnXgesvd`, `cusolverDnXsyevd`, etc.

**Key difference:** Uses `cusolverDnParams_t` for configuration and supports `CUDA_R_16F`, `CUDA_R_32F`, `CUDA_R_64F` type parameters.

```cpp
cusolverDnParams_t params;
CUSOLVER_CHECK(cusolverDnCreateParams(&params));
// configure params if needed...

size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverH, params, m, n,
    CUDA_R_64F, d_A, lda, CUDA_R_64F, &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost));
// Allocate and call cusolverDnXgetrf(...)
```

---

## Error Handling

```cpp
#define CUSOLVER_CHECK(err) do {                                        \
    cusolverStatus_t _s = (err);                                        \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuSolver error %d at %s:%d\n",                \
                _s, __FILE__, __LINE__);                                \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)
```

**Check `devInfo` after factorization calls:**
```cpp
int info;
cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
if (info < 0)  printf("Parameter %d is invalid\n", -info);
if (info > 0)  printf("Matrix is singular at diagonal %d\n", info);
```

**Header:** `#include <cusolverDn.h>` — link: `-lcusolver`.
