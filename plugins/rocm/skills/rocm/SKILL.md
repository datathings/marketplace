---
name: rocm
description: "AMD ROCm GPU computing stack for HIP kernel development and GPU-accelerated library usage. Use when: writing HIP kernels (.hip files), using rocBLAS/rocFFT/rocRAND/rocSOLVER/rocSPARSE/hipBLAS/hipBLASLt compute libraries, profiling with rocProfiler or rocprof, porting CUDA code to HIP, building CMake/Makefile projects targeting AMD GPUs, or debugging GPU code with rocGDB."
---

# ROCm GPU Development

**Version:** rocm-7.2.0
**Language:** C/C++ with HIP (`.hip` or `.cpp` files)
**License:** MIT (examples and most libraries); some components Apache-2.0
**Docs:** https://rocm.docs.amd.com

## Overview

ROCm is AMD's open-source GPU computing stack. HIP (Heterogeneous-compute Interface for Portability) is the primary programming API — it is syntactically close to CUDA and compiles on both AMD and NVIDIA GPUs. ROCm includes a full suite of optimized compute libraries (BLAS, FFT, RNG, solvers, sparse) plus profiling and debugging tools.

## Quick Start

```cpp
// minimal.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello(int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) printf("thread %d\n", i);
}

int main() {
    hello<<<dim3(4), dim3(8)>>>(32);    // 4 blocks × 8 threads
    hipDeviceSynchronize();
}
```

```bash
# Compile and run
hipcc -O2 -o hello minimal.hip && ./hello

# CMake (preferred for larger projects)
# cmake -S . -B build && cmake --build build
```

**Error-check macro (use throughout your code):**
```cpp
#define HIP_CHECK(expr) do { \
    hipError_t err = (expr); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Usage:
HIP_CHECK(hipMalloc(&d_ptr, bytes));
HIP_CHECK(hipMemcpy(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice));
my_kernel<<<grid, block>>>(args);
HIP_CHECK(hipGetLastError());       // catch launch errors
HIP_CHECK(hipDeviceSynchronize());  // catch runtime errors
```

## Core HIP Concepts

### Device Management
```cpp
int n_devices;
hipGetDeviceCount(&n_devices);
hipSetDevice(0);                          // select GPU

hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);
// props.name, .totalGlobalMem, .warpSize (64 on AMD!),
// .maxThreadsPerBlock, .gcnArchName
```

### Memory Pattern
```cpp
float *d_buf;
HIP_CHECK(hipMalloc(&d_buf, N * sizeof(float)));
HIP_CHECK(hipMemcpy(d_buf, h_buf, N*sizeof(float), hipMemcpyHostToDevice));
// ... kernel launches ...
HIP_CHECK(hipMemcpy(h_buf, d_buf, N*sizeof(float), hipMemcpyDeviceToHost));
HIP_CHECK(hipFree(d_buf));
```

For async transfers (overlap with compute), use `hipHostMalloc` + `hipMemcpyAsync` + streams.

### Kernel Launch Syntax
```cpp
// kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args)
constexpr int BLOCK = 256;
int grid = (N + BLOCK - 1) / BLOCK;   // ceiling division
my_kernel<<<dim3(grid), dim3(BLOCK), 0, hipStreamDefault>>>(d_ptr, N);
```

### Key Device Built-ins
```cpp
// Inside __global__ kernel:
int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 1D global index
// 2D:
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

__syncthreads();                    // block-level barrier
__shared__ float smem[BLOCK_SIZE]; // shared memory declaration
atomicAdd(&shared_counter, 1);     // atomic operations

// AMD: warpSize = 64 (wavefront), NVIDIA: warpSize = 32
// Always use warpSize built-in, not a hardcoded 32!
```

### Function Qualifiers
| Qualifier | Executes on | Called from |
|-----------|-------------|-------------|
| `__global__` | GPU | host only — kernel entry |
| `__device__` | GPU | GPU only |
| `__host__` | CPU | CPU only (default) |
| `__host__ __device__` | both | both |

## ROCm Installation

```bash
# Default install path
/opt/rocm/              # ROCm root
/opt/rocm/bin/hipcc     # HIP compiler
/opt/rocm/bin/rocm-smi  # GPU monitor
/opt/rocm/bin/rocgdb    # GPU debugger

# Verify installation
hipcc --version
rocm-smi                # show GPU status
/opt/rocm/bin/rocm_agent_enumerator  # list GPU targets (e.g. gfx1100, gfx90a)

# Target architecture flags
hipcc --offload-arch=gfx1100 ...   # RX 7900 (Navi31/RDNA3)
hipcc --offload-arch=gfx90a  ...   # MI200 (CDNA2)
hipcc --offload-arch=gfx942  ...   # MI300 (CDNA3)
hipcc --offload-arch=gfx1030 ...   # RX 6800 (Navi21/RDNA2)
```

## API Reference

| Domain | Reference File | Key APIs / Purpose |
|--------|---------------|-------------------|
| HIP Runtime | `references/api-hip-core.md` | `hipMalloc`, `hipMemcpy`, `hipMemcpyAsync`, `hipStream_t`, `hipEvent_t`, occupancy API, warp intrinsics, atomics |
| HIP Math | `references/api-hip-math.md` | `sinf/cosf/expf/rsqrtf`, half-precision `__half`, complex math, bit manipulation |
| Compute Libraries | `references/api-libraries.md` | rocBLAS, hipBLAS, hipBLASLt, rocFFT, hipFFT, rocRAND, rocSOLVER, rocSPARSE, rocWMMA, rocPRIM, hipCUB |
| Profiling & Debug | `references/api-profiling.md` | HIP events, rocProfiler-SDK, `rocprof` CLI, `rocm-smi`, `rocgdb`, ASAN |
| Workflows | `references/workflows.md` | Complete examples: SAXPY, tiled matmul, rocBLAS GEMM, rocFFT, rocRAND, multi-GPU, streaming overlap, CUDA→HIP porting, pitfalls |

## Common Workflows

### SAXPY (hello-GPU equivalent)
```cpp
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
// See workflows.md for full host setup.
```

### rocBLAS GEMM (matrix multiply)
```cpp
rocblas_handle h; rocblas_create_handle(&h);
rocblas_set_pointer_mode(h, rocblas_pointer_mode_host);
rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
rocblas_destroy_handle(h);
```

### rocFFT 1D Transform
```cpp
rocfft_setup();
rocfft_plan plan;
size_t len = N;
rocfft_plan_create(&plan, rocfft_placement_inplace,
                   rocfft_transform_type_complex_forward,
                   rocfft_precision_double, 1, &len, 1, nullptr);
rocfft_execute(plan, (void**)&d_data, nullptr, nullptr);
rocfft_plan_destroy(plan); rocfft_cleanup();
```

### Kernel Timing
```cpp
hipEvent_t start, stop;
hipEventCreate(&start); hipEventCreate(&stop);
hipEventRecord(start, nullptr);
my_kernel<<<grid, block>>>(args);
hipEventRecord(stop, nullptr);
hipEventSynchronize(stop);
float ms; hipEventElapsedTime(&ms, start, stop);
```

### Occupancy-Based Launch
```cpp
int min_grid, block_size;
hipOccupancyMaxPotentialBlockSize(&min_grid, &block_size, my_kernel, 0, 0);
int grid_size = (N + block_size - 1) / block_size;
my_kernel<<<dim3(grid_size), dim3(block_size)>>>(args);
```

## Build System (CMake)

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyApp LANGUAGES CXX)
enable_language(HIP)

find_package(hip REQUIRED)
find_package(rocblas)   # optional
find_package(rocfft)    # optional

add_executable(my_app main.hip)
target_link_libraries(my_app PRIVATE hip::host roc::rocblas roc::rocfft)
set_property(TARGET my_app PROPERTY HIP_ARCHITECTURES gfx90a gfx1100)
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## CUDA to HIP Porting

```bash
# Automated conversion (included with ROCm)
hipify-perl cuda_code.cu > hip_code.hip
hipify-perl -inplace *.cu *.cuh
```

| CUDA | HIP equivalent |
|------|---------------|
| `cudaMalloc / cudaFree` | `hipMalloc / hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cublasCreate` | `rocblas_create_handle` |
| `cufftPlan1d` | `hipfftPlan1d` |
| `curandCreateGenerator` | `rocrand_create_generator` |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` (AMD) |

## Key Considerations

- **AMD wavefront = 64 threads** (not 32). Always use `warpSize` built-in, never hardcode 32. Warp-level reduction loops must start at `warpSize/2`.
- **rocBLAS uses column-major** storage. For row-major C arrays: swap A/B and transpose M/N, or preprocess data.
- **Async copies require pinned memory**: `hipMemcpyAsync` falls back to synchronous with pageable memory. Use `hipHostMalloc`.
- **Kernel errors are asynchronous**: always call `hipGetLastError()` after launch and `hipDeviceSynchronize()` to flush.
- **rocFFT inverse transforms do not normalize**: divide output by N after IFFT.
- **Shared memory limit**: 64 KB per block on CDNA (MI), 32 KB on RDNA (RX). Exceeding it silently reduces occupancy.
- **Managed memory** (`hipMallocManaged`) is convenient but typically slower than explicit transfers on discrete GPUs.
- **Library selection**: prefer rocBLAS for AMD GEMM performance; use hipBLAS/hipFFT for portability; hipBLASLt for DL epilogues.
