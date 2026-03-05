# ROCm Compute Libraries Reference

## Table of Contents
1. [rocBLAS — Dense Linear Algebra](#rocblas--dense-linear-algebra)
2. [hipBLAS — Portable BLAS Wrapper](#hipblas--portable-blas-wrapper)
3. [hipBLASLt — Advanced GEMM with Epilogue](#hipblaslt--advanced-gemm-with-epilogue)
4. [rocFFT — Fast Fourier Transform](#rocfft--fast-fourier-transform)
5. [hipFFT — Portable FFT Wrapper](#hipfft--portable-fft-wrapper)
6. [rocRAND — Random Number Generation](#rocrand--random-number-generation)
7. [rocSOLVER — Dense Linear Algebra Solvers](#rocsolver--dense-linear-algebra-solvers)
8. [hipSOLVER — Portable Solver Wrapper](#hipsolver--portable-solver-wrapper)
9. [rocSPARSE — Sparse Linear Algebra](#rocsparse--sparse-linear-algebra)
10. [rocWMMA — Wavefront Matrix Multiply-Accumulate](#rocwmma--wavefront-matrix-multiply-accumulate)
11. [rocPRIM / hipCUB / rocThrust](#rocprim--hipcub--rocthrust)
12. [Library Selection Guide](#library-selection-guide)

---

## rocBLAS — Dense Linear Algebra

**Header:** `#include <rocblas/rocblas.h>`
**Link:** `-lrocblas`
**Install path:** `/opt/rocm/include/rocblas/`, `/opt/rocm/lib/`

### Handle Lifecycle
```cpp
rocblas_handle handle;
rocblas_create_handle(&handle);
// ... use handle ...
rocblas_destroy_handle(handle);

// Error check macro:
#define ROCBLAS_CHECK(expr) \
    { rocblas_status s = (expr); if(s != rocblas_status_success) { \
      fprintf(stderr, "rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); exit(1); } }
```

### Pointer Mode
```cpp
// Scalars (alpha, beta) can live on host or device:
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);   // default
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
```

### Level 1 (vector operations)
```cpp
// SAXPY: y = alpha*x + y
rocblas_saxpy(handle, n, &alpha, d_x, incx, d_y, incy);
rocblas_daxpy(handle, n, &alpha, d_x, incx, d_y, incy);  // double

// Scale: x = alpha*x
rocblas_sscal(handle, n, &alpha, d_x, incx);

// Dot product: result = x · y
float result;
rocblas_sdot(handle, n, d_x, incx, d_y, incy, &result);

// Euclidean norm: result = ||x||_2
rocblas_snrm2(handle, n, d_x, incx, &result);

// SWAP: swap x and y
rocblas_sswap(handle, n, d_x, incx, d_y, incy);
```

### Level 2 (matrix-vector operations)
```cpp
// SGEMV: y = alpha*A*x + beta*y  (A: m×n)
rocblas_sgemv(handle,
              rocblas_operation_none,  // or rocblas_operation_transpose
              m, n, &alpha,
              d_A, lda,
              d_x, incx,
              &beta,
              d_y, incy);
```

### Level 3 (matrix-matrix operations)
```cpp
// SGEMM: C = alpha*op(A)*op(B) + beta*C
// A: m×k, B: k×n, C: m×n
rocblas_sgemm(handle,
              rocblas_operation_none,  // trans_A
              rocblas_operation_none,  // trans_B
              m, n, k,
              &alpha,
              d_A, lda,   // lda >= m
              d_B, ldb,   // ldb >= k
              &beta,
              d_C, ldc);  // ldc >= m

// Batched GEMM (multiple matrices in one call)
rocblas_sgemm_strided_batched(handle,
    trans_A, trans_B, m, n, k,
    &alpha,
    d_A, lda, stride_A,
    d_B, ldb, stride_B,
    &beta,
    d_C, ldc, stride_C,
    batch_count);
```

### Types
```cpp
rocblas_int         // int32
rocblas_float       // float
rocblas_double      // double
rocblas_float_complex   // {float re, im}
rocblas_double_complex  // {double re, im}
rocblas_operation   // rocblas_operation_none / _transpose / _conjugate_transpose
rocblas_fill        // rocblas_fill_upper / _lower
rocblas_diagonal    // rocblas_diagonal_unit / _non_unit
rocblas_side        // rocblas_side_left / _right
```

---

## hipBLAS — Portable BLAS Wrapper

**Header:** `#include <hipblas/hipblas.h>`
**Link:** `-lhipblas`

hipBLAS wraps rocBLAS (AMD) or cuBLAS (NVIDIA) with a unified API. Use when portability across platforms matters.

```cpp
hipblasHandle_t handle;
hipblasCreate(&handle);

// SGEMM identical to rocBLAS naming, prefix hipblas:
hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
             m, n, k, &alpha,
             d_A, lda, d_B, ldb,
             &beta, d_C, ldc);

hipblasDestroy(handle);

// Operation enum: HIPBLAS_OP_N / HIPBLAS_OP_T / HIPBLAS_OP_C
// Fill: HIPBLAS_FILL_MODE_UPPER / _LOWER
```

---

## hipBLASLt — Advanced GEMM with Epilogue

**Header:** `#include <hipblaslt/hipblaslt.h>`
**Link:** `-lhipblaslt`

hipBLASLt extends GEMM with activation functions, bias, scale, and mixed-precision support. Intended for deep learning use cases.

```cpp
hipblasLtHandle_t lt_handle;
hipblasLtCreate(&lt_handle);

// Matrix layout descriptor
hipblasLtMatrixLayout_t mat_A, mat_B, mat_C;
hipblasLtMatrixLayoutCreate(&mat_A, HIPBLASLT_R_32F, m, k, lda);
hipblasLtMatrixLayoutCreate(&mat_B, HIPBLASLT_R_32F, k, n, ldb);
hipblasLtMatrixLayoutCreate(&mat_C, HIPBLASLT_R_32F, m, n, ldc);

// GEMM operation descriptor
hipblasLtMatmulDesc_t op_desc;
hipblasLtMatmulDescCreate(&op_desc, HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F);

// Optional epilogue (e.g., add bias + ReLU)
hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epilogue, sizeof(epilogue));

// Algorithm selection (get_all_algos → pick best)
// ... algorithm query omitted for brevity ...

// Execute
hipblasLtMatmul(lt_handle, op_desc, &alpha,
                d_A, mat_A, d_B, mat_B, &beta,
                d_C, mat_C, d_C, mat_C,
                &algo, workspace, workspace_size, stream);

hipblasLtMatmulDescDestroy(op_desc);
hipblasLtDestroy(lt_handle);
```

**Supported data types:** F32, F16, BF16, F8 (MI300+), I8
**Key epilogues:** `RELU`, `GELU`, `BIAS`, `RELU_BIAS`, `GELU_BIAS`, `BGRADB` (grad bias)

---

## rocFFT — Fast Fourier Transform

**Header:** `#include <rocfft/rocfft.h>`
**Link:** `-lrocfft`

### Plan-Execute-Destroy Pattern
```cpp
rocfft_setup();  // global init (once per process)

// 1. Create plan description (optional — for non-default layouts)
rocfft_plan_description desc = nullptr;
rocfft_plan_description_create(&desc);
rocfft_plan_description_set_data_layout(desc,
    rocfft_array_type_complex_interleaved,  // input format
    rocfft_array_type_complex_interleaved,  // output format
    nullptr, nullptr,
    stride.size(), stride.data(), 0,        // input strides/distance
    stride.size(), stride.data(), 0);       // output strides/distance

// 2. Create plan
size_t lengths[] = {N};   // 1D: {N}, 2D: {rows, cols}, 3D: {x, y, z}
rocfft_plan plan = nullptr;
rocfft_plan_create(&plan,
    rocfft_placement_inplace,          // or rocfft_placement_notinplace
    rocfft_transform_type_complex_forward,  // or _complex_inverse / _real_forward / _real_inverse
    rocfft_precision_double,           // or rocfft_precision_single
    1,                                 // dimensions
    lengths,
    1,                                 // batch count
    desc);

// 3. Allocate work buffer
rocfft_execution_info info = nullptr;
rocfft_execution_info_create(&info);
size_t wbuf_size = 0;
rocfft_plan_get_work_buffer_size(plan, &wbuf_size);
void* wbuf = nullptr;
if (wbuf_size > 0) {
    hipMalloc(&wbuf, wbuf_size);
    rocfft_execution_info_set_work_buffer(info, wbuf, wbuf_size);
}

// 4. Execute
rocfft_execute(plan, (void**)&d_input, (void**)&d_output, info);

// 5. Cleanup
rocfft_execution_info_destroy(info);
if (wbuf) hipFree(wbuf);
rocfft_plan_description_destroy(desc);
rocfft_plan_destroy(plan);
rocfft_cleanup();
```

### Transform Types
| Constant | Description |
|----------|-------------|
| `rocfft_transform_type_complex_forward` | C2C forward (DFT) |
| `rocfft_transform_type_complex_inverse` | C2C inverse (IDFT) |
| `rocfft_transform_type_real_forward`    | R2C (real to complex) |
| `rocfft_transform_type_real_inverse`    | C2R (complex to real) |

**Note:** Inverse transforms do NOT normalize — divide by N after.

---

## hipFFT — Portable FFT Wrapper

**Header:** `#include <hipfft/hipfft.h>`
**Link:** `-lhipfft`

```cpp
hipfftHandle plan;
hipfftCreate(&plan);

// 1D complex-to-complex
hipfftPlan1d(&plan, N, HIPFFT_C2C, batch);
hipfftExecC2C(plan, d_input, d_output, HIPFFT_FORWARD);   // or HIPFFT_BACKWARD

// 2D, 3D, and many-plan variants
hipfftPlan2d(&plan, nx, ny, HIPFFT_C2C);
hipfftPlanMany(&plan, rank, n, inembed, istride, idist,
                onembed, ostride, odist, type, batch);

// Real transforms
hipfftPlan1d(&plan, N, HIPFFT_R2C, batch);
hipfftExecR2C(plan, (hipfftReal*)d_in, (hipfftComplex*)d_out);

hipfftDestroy(plan);

// Types: HIPFFT_R2C, HIPFFT_C2R, HIPFFT_C2C, HIPFFT_D2Z, HIPFFT_Z2D, HIPFFT_Z2Z
```

---

## rocRAND — Random Number Generation

**Header:** `#include <rocrand/rocrand.h>` (C API) or `#include <rocrand/rocrand.hpp>` (C++)
**Link:** `-lrocrand`

### C++ API (recommended)
```cpp
#include <rocrand/rocrand.hpp>

// Engines
rocrand_cpp::default_random_engine engine;     // alias for xorwow
rocrand_cpp::philox4x32_10         engine;
rocrand_cpp::mrg32k3a              engine;
rocrand_cpp::sobol32               engine;     // quasi-random
rocrand_cpp::lfsr113               engine;

// Distributions (output to device pointer)
rocrand_cpp::uniform_real_distribution<float>    uniform_f;   // [0,1)
rocrand_cpp::uniform_int_distribution<unsigned>  uniform_i;
rocrand_cpp::normal_distribution<float>          normal(mean=0, stddev=1);
rocrand_cpp::lognormal_distribution<float>       lognormal(mean=0, stddev=1);
rocrand_cpp::poisson_distribution<unsigned>      poisson(lambda=1.0);

// Generate: writes N values to device pointer d_out
float* d_out;
hipMalloc(&d_out, N * sizeof(float));
uniform_f(engine, d_out, N);
normal(engine, d_out, N);
```

### C API
```cpp
rocrand_generator gen;
rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
rocrand_set_seed(gen, 12345ULL);

float* d_out;
hipMalloc(&d_out, N * sizeof(float));
rocrand_generate_uniform(gen, d_out, N);         // uniform [0,1)
rocrand_generate_normal(gen, d_out, N, 0.f, 1.f); // normal
rocrand_generate_poisson(gen, d_uint, N, lambda); // Poisson

rocrand_destroy_generator(gen);

// Generator types: ROCRAND_RNG_PSEUDO_XORWOW / _MRG32K3A / _PHILOX4_32_10
//                  ROCRAND_RNG_QUASI_SOBOL32 / _SCRAMBLED_SOBOL32
```

---

## rocSOLVER — Dense Linear Algebra Solvers

**Header:** `#include <rocsolver/rocsolver.h>`
**Link:** `-lrocsolver -lrocblas`

rocSOLVER uses rocBLAS handles.

```cpp
rocblas_handle handle;
rocblas_create_handle(&handle);

// LU factorization: A = P*L*U (overwrites A)
// A: n×n, lda: leading dim, ipiv: pivot indices, info: result (0=success)
rocblas_int info;
rocsolver_sgetf2(handle, n, n, d_A, lda, d_ipiv, &info);   // single-precision
rocsolver_dgetf2(handle, n, n, d_A, lda, d_ipiv, &info);   // double-precision

// Solve Ax = b using LU factorization from getf2/getrf
rocsolver_sgetrs(handle,
    rocblas_operation_none,  // no transpose
    n,                       // order of A
    nrhs,                    // number of RHS columns
    d_A, lda,                // LU factors from getf2
    d_ipiv,                  // pivot indices
    d_B, ldb);               // on input: B; on output: X

// Eigenvalue decomposition: A*V = V*D (symmetric/Hermitian)
rocsolver_ssyev(handle,
    rocblas_evect_original,   // compute eigenvectors
    rocblas_fill_lower,       // use lower triangle of A
    n,
    d_A, lda,                 // on output: eigenvectors
    d_D,                      // eigenvalues (ascending)
    &info);

// QR factorization
rocsolver_sgeqrf(handle, m, n, d_A, lda, d_tau);

// Inverse via getri (requires prior getrf)
rocsolver_sgetri(handle, n, d_A, lda, d_ipiv, &info);

rocblas_destroy_handle(handle);
```

---

## hipSOLVER — Portable Solver Wrapper

**Header:** `#include <hipsolver/hipsolver.h>`
**Link:** `-lhipsolver`

```cpp
hipsolverHandle_t handle;
hipsolverCreate(&handle);

// Eigenvalues of symmetric matrix
int lwork;
hipsolverSsyevd_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_LOWER, n, d_A, lda, d_W, &lwork);
float* d_work;
hipMalloc(&d_work, lwork * sizeof(float));
int* d_info;
hipMalloc(&d_info, sizeof(int));
hipsolverSsyevd(handle, HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_LOWER, n, d_A, lda, d_W,
    d_work, lwork, d_info);

hipsolverDestroy(handle);
```

---

## rocSPARSE — Sparse Linear Algebra

**Header:** `#include <rocsparse/rocsparse.h>`
**Link:** `-lrocsparse`

```cpp
rocsparse_handle handle;
rocsparse_create_handle(&handle);

// Build CSR matrix descriptor
rocsparse_mat_descr descr;
rocsparse_create_mat_descr(&descr);

// SpMV: y = alpha * A * x + beta * y  (A in CSR format)
rocsparse_scsrmv(handle,
    rocsparse_operation_none,
    m, n, nnz,              // rows, cols, non-zeros
    &alpha,
    descr,
    d_csr_val,              // non-zero values
    d_csr_row_ptr,          // row pointers (size m+1)
    d_csr_col_ind,          // column indices
    nullptr,                // mat_info
    d_x, &beta, d_y);

// CSR → CSC format conversion
rocsparse_scsr2csc(handle, m, n, nnz,
    d_csr_val, d_csr_row_ptr, d_csr_col_ind,
    d_csc_val, d_csc_col_ptr, d_csc_row_ind,
    rocsparse_action_numeric,
    rocsparse_index_base_zero);

rocsparse_destroy_mat_descr(descr);
rocsparse_destroy_handle(handle);
```

**Supported formats:** CSR, CSC, COO, ELL, HYB, BSR
**Operations:** SpMV, SpMM, SpSV (triangular solve), SpGEMM, ILU, IC, conversion

---

## rocWMMA — Wavefront Matrix Multiply-Accumulate

**Header:** `#include <rocwmma/rocwmma.hpp>`
**Link:** `-lrocwmma` (or header-only)

WMMA provides C++ templates for hardware matrix multiply-accumulate (analogous to CUDA `wmma`). Targets AMD CDNA/RDNA matrix cores.

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// Fragment types
fragment<matrix_a, M, N, K, float16_t, row_major>  frag_a;
fragment<matrix_b, M, N, K, float16_t, col_major>  frag_b;
fragment<accumulator, M, N, K, float32_t>           frag_c;

// Load/store
load_matrix_sync(frag_a, d_A + offset, lda);
load_matrix_sync(frag_b, d_B + offset, ldb);
load_matrix_sync(frag_c, d_C + offset, ldc, mem_row_major);

// Multiply-accumulate: frag_c += frag_a * frag_b
mma_sync(frag_c, frag_a, frag_b, frag_c);

store_matrix_sync(d_C + offset, frag_c, ldc, mem_row_major);

// Supported tile sizes (M, N, K):
// FP16: 16×16×16, 32×32×8, 64×64×4, ...
// BF16, INT8, FP8: architecture-dependent
```

---

## rocPRIM / hipCUB / rocThrust

### rocPRIM (device primitives)
**Header:** `#include <rocprim/rocprim.hpp>`

```cpp
// Reduction
void* d_temp = nullptr; size_t temp_bytes = 0;
rocprim::reduce(d_temp, temp_bytes, d_input, d_output, N, rocprim::plus<float>());
hipMalloc(&d_temp, temp_bytes);
rocprim::reduce(d_temp, temp_bytes, d_input, d_output, N, rocprim::plus<float>());

// Scan (inclusive prefix sum)
rocprim::inclusive_scan(d_temp, temp_bytes, d_input, d_output, N, rocprim::plus<float>());

// Sort
rocprim::radix_sort_keys(d_temp, temp_bytes, d_keys_in, d_keys_out, N);

// Select / partition
rocprim::select(d_temp, temp_bytes, d_input, d_output, d_selected_count, N,
                [](float x) { return x > 0; });
```

### hipCUB (portable CUB wrapper)
**Header:** `#include <hipcub/hipcub.hpp>`

```cpp
// DeviceReduce
void* d_temp = nullptr; size_t temp_bytes = 0;
hipcub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, N);
hipMalloc(&d_temp, temp_bytes);
hipcub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, N);

// DeviceScan
hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_input, d_output, N);

// DeviceRadixSort
hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_in, d_keys_out, N);

// BlockReduce (in device kernel)
using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
__shared__ typename BlockReduce::TempStorage smem;
float block_sum = BlockReduce(smem).Sum(thread_val);
```

### rocThrust
**Header:** `#include <thrust/...>` with `thrust::hip::par`

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

thrust::device_vector<float> d_vec(N, 1.0f);

// Reduction
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

// Transform
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                  [] __device__ (float x) { return x * 2.0f; });

// Sort
thrust::sort(d_vec.begin(), d_vec.end());
```

---

## Library Selection Guide

| Need | Library | Notes |
|------|---------|-------|
| Dense matmul (single GPU) | rocBLAS | Highest performance on AMD |
| Dense matmul (portable) | hipBLAS | Thin wrapper, near-zero overhead |
| GEMM + activation/bias (DL) | hipBLASLt | Use for training kernels |
| Small matrix cores (tile-level) | rocWMMA | Direct WMMA intrinsics |
| FFT | rocFFT or hipFFT | hipFFT for portability |
| Random numbers (GPU) | rocRAND | Supports quasi-random too |
| Eigenvalues/LU/QR (dense) | rocSOLVER / hipSOLVER | hipSOLVER for portability |
| Sparse matmul / solve | rocSPARSE | Full sparse format support |
| Parallel scan/sort/reduce | rocPRIM | Lowest-level, fastest |
| Portable scan/sort/reduce | hipCUB | Wraps rocPRIM or CUB |
| High-level parallel STL | rocThrust | Thrust-compatible API |
