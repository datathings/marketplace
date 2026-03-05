# cuFFT API

## Table of Contents
1. [Plan Creation](#plan-creation)
2. [Advanced Plan (Many)](#advanced-plan-many)
3. [Stream Binding](#stream-binding)
4. [Execution](#execution)
5. [Cleanup](#cleanup)
6. [Error Handling](#error-handling)
7. [Transform Types Reference](#transform-types-reference)

---

## Plan Creation

cuFFT separates plan creation (describing the transform) from execution. Plans can be reused.

### `cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) -> cufftResult`
**Description:** Creates a 1D FFT plan for `batch` transforms of size `nx`.
**Parameters:**
- `nx` — number of complex/real elements per transform
- `type` — transform type (see reference table below)
- `batch` — number of transforms to execute

**Example (complex-to-complex, batched):**
```cpp
cufftHandle plan;
int fft_size = 1024, batch = 32;
CUFFT_CHECK(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch));
```

### `cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) -> cufftResult`
**Description:** Creates a 2D FFT plan. `nx` rows, `ny` columns. Batch=1.
**Example:**
```cpp
cufftHandle plan;
CUFFT_CHECK(cufftPlan2d(&plan, 512, 512, CUFFT_C2C));
```

### `cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) -> cufftResult`
**Description:** Creates a 3D FFT plan.

---

## Advanced Plan (Many)

### `cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) -> cufftResult`
**Description:** Full-featured plan for arbitrary stride/embed batched transforms.
**Parameters:**
- `rank` — dimensionality (1, 2, or 3)
- `n` — array of `rank` transform sizes
- `inembed/onembed` — storage dimensions of input/output (NULL = contiguous)
- `istride/ostride` — element stride in input/output
- `idist/odist` — distance between successive batches (in elements)

**Example (batched 1D, contiguous):**
```cpp
int n[] = {1024};
cufftHandle plan;
// 64 transforms of size 1024, contiguous layout
CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                           NULL, 1, 1024,   // inembed, istride, idist
                           NULL, 1, 1024,   // onembed, ostride, odist
                           CUFFT_C2C, 64));
```

### `cufftEstimate1d / cufftGetSize1d` — query workspace size before allocation.

---

## Stream Binding

### `cufftSetStream(cufftHandle plan, cudaStream_t stream) -> cufftResult`
**Description:** Associates an async stream with the plan. All executions run in that stream.
**Example:**
```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
cufftSetStream(plan, stream);
```

---

## Execution

### `cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) -> cufftResult`
**Description:** Executes complex-to-complex FFT.
**Parameters:** `direction` — `CUFFT_FORWARD` (-1) or `CUFFT_INVERSE` (+1).
**Note:** `idata == odata` is valid for in-place transforms.
**Example:**
```cpp
// Forward then inverse (round-trip)
CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
// Normalize manually: output is unscaled by 1/N
scale_kernel<<<grid, block, 0, stream>>>(d_data, n, 1.0f / n);
CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
```

### `cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) -> cufftResult`
**Description:** Real-to-complex (forward) transform. Output has `n/2+1` complex elements per transform.
**Example:**
```cpp
// Out-of-place R2C
CUFFT_CHECK(cufftExecR2C(planR2C, d_real_input, d_complex_output));
```

### `cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) -> cufftResult`
**Description:** Complex-to-real (inverse) transform. Input must be Hermitian-symmetric.
**Example:**
```cpp
CUFFT_CHECK(cufftExecC2R(planC2R, d_complex_input, d_real_output));
cudaStreamSynchronize(stream);
```

### `cufftExecZ2Z` / `cufftExecD2Z` / `cufftExecZ2D`
**Description:** Double-precision equivalents: `Z2Z` (complex128 to complex128), `D2Z` (real64 to complex128), `Z2D` (complex128 to real64).

---

## Cleanup

### `cufftDestroy(cufftHandle plan) -> cufftResult`
**Description:** Frees all resources associated with the plan.
**Example:**
```cpp
CUFFT_CHECK(cufftDestroy(plan));
CUDA_CHECK(cudaFree(d_data));
CUDA_CHECK(cudaStreamDestroy(stream));
```

---

## Error Handling

```cpp
#define CUFFT_CHECK(err) do {                                           \
    cufftResult _r = (err);                                             \
    if (_r != CUFFT_SUCCESS) {                                          \
        fprintf(stderr, "cuFFT error %d at %s:%d\n",                   \
                _r, __FILE__, __LINE__);                                \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)
```

**Common errors:** `CUFFT_INVALID_PLAN`, `CUFFT_ALLOC_FAILED`, `CUFFT_INVALID_SIZE`, `CUFFT_INCOMPLETE_PARAMETER_LIST`.

---

## Transform Types Reference

| Type | Input | Output | Direction |
|------|-------|--------|-----------|
| `CUFFT_C2C` | `cufftComplex` (FP32) | `cufftComplex` | Forward or Inverse |
| `CUFFT_R2C` | `cufftReal` (FP32) | `cufftComplex` | Forward only |
| `CUFFT_C2R` | `cufftComplex` | `cufftReal` | Inverse only |
| `CUFFT_Z2Z` | `cufftDoubleComplex` | `cufftDoubleComplex` | Forward or Inverse |
| `CUFFT_D2Z` | `cufftDoubleReal` | `cufftDoubleComplex` | Forward only |
| `CUFFT_Z2D` | `cufftDoubleComplex` | `cufftDoubleReal` | Inverse only |

**Key notes:**
- cuFFT does **not** normalize inverse FFTs — divide by N manually.
- R2C output size is `(n/2 + 1)` complex elements (Hermitian symmetry).
- For performance: transforms whose size has only small prime factors (2, 3, 5, 7) are fastest.
- Sizes that are powers of 2 give best performance.
- Include: `#include <cufft.h>`, link: `-lcufft`.
