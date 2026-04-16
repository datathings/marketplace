---
name: cuda
description: "NVIDIA CUDA parallel computing platform — use when writing .cu kernels, using cuBLAS/cuDNN/cuFFT/cuSPARSE/cuRAND/cuSolver, Thrust, or Cooperative Groups for GPU-accelerated computing"
---

# CUDA

## Overview

CUDA is NVIDIA's parallel computing platform and programming model for GPU-accelerated applications. It provides direct access to the GPU's virtual instruction set and parallel compute elements for executing kernels in C, C++, and Fortran.

**cuda-samples version:** v13.2 (CUDA Toolkit 13.2)
**CUDALibrarySamples:** main (April 2026)
**Language:** C/C++ (.cu files)
**Licenses:** BSD-3-Clause (cuda-samples), Apache-2.0 (CUDALibrarySamples)

## Quick Start

```c
// Minimal kernel + launch
__global__ void addVectors(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    addVectors<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

## Core Concepts

- **Kernel**: `__global__` function executed on GPU by many parallel threads
- **Grid/Block/Thread**: Launch hierarchy — `<<<gridDim, blockDim>>>` configures parallelism
- **Device memory**: Must be explicitly allocated with `cudaMalloc` and freed with `cudaFree`
- **Streams**: Async execution queues; default stream is synchronous with host
- **Unified Memory** (`cudaMallocManaged`): Automatically migrates data between CPU and GPU; check support with `device_prop.managedMemory`

## API Reference

| Domain | File | Description |
|--------|------|-------------|
| CUDA Runtime | [api-runtime.md](references/api-runtime.md) | Device mgmt, memory, streams, events, kernel launch |
| cuBLAS | [api-cublas.md](references/api-cublas.md) | Dense linear algebra: GEMM, GEMV, TRSM, grouped batched ops |
| cuFFT | [api-cufft.md](references/api-cufft.md) | 1D/2D/3D FFT and batched transforms |
| cuSPARSE | [api-cusparse.md](references/api-cusparse.md) | Sparse matrix ops: SpMM, SpMV, format conversions |
| cuRAND | [api-curand.md](references/api-curand.md) | Random number generation on GPU |
| cuSolver | [api-cusolver.md](references/api-cusolver.md) | Dense/sparse solvers: QR, LU, eigenvalue, SVD |
| Thrust | [api-thrust.md](references/api-thrust.md) | STL-like GPU algorithms: sort, reduce, transform, scan |
| Cooperative Groups | [api-cooperative-groups.md](references/api-cooperative-groups.md) | Flexible thread synchronization beyond blocks |
| Workflows | [workflows.md](references/workflows.md) | Complete working examples |

## Common Workflows

See [references/workflows.md](references/workflows.md) for complete examples.

Quick reference:
- **Matrix multiply**: see workflows.md — cuBLAS GEMM section
- **FFT**: see workflows.md — cuFFT 1D section
- **Custom kernel**: see workflows.md — Custom CUDA Kernel section
- **Unified memory**: see workflows.md — Unified Memory section
- **Error-check macros**: see workflows.md — Error-Checked CUDA Boilerplate

## Key Considerations

- **Error checking**: Always check return codes; use `CUDA_CHECK(err)` macro pattern
- **Synchronization**: `cudaDeviceSynchronize()` or stream synchronize before reading results on host
- **Memory alignment**: 128-byte alignment for coalesced global memory access
- **Occupancy**: Use `cudaOccupancyMaxPotentialBlockSize` to tune block dimensions; uses `uint32_t` for array indexing to avoid overflow
- **Tensor Cores**: Available on Volta+ (sm_70+); cuBLAS uses them automatically for GEMM with correct types
- **Column-major**: cuBLAS and cuSolver use Fortran (column-major) layout — transpose row-major C arrays or swap dimensions
- **cuFFT normalization**: cuFFT does NOT normalize inverse transforms; divide by N manually
- **Streams**: Always use `cudaStreamNonBlocking` when creating non-default streams to avoid implicit synchronization with the null stream
- **Unified Memory support**: Check `device_prop.managedMemory` before using `cudaMallocManaged`; use `cudaMemPrefetchAsync` to avoid page faults
- **Driver API cleanup**: Always call `cuModuleUnload` before `cuCtxDestroy` when using the CUDA Driver API
- **Compile flags**: Use `--gpu-architecture=sm_90a` for Hopper (H100), `sm_80` for Ampere (A100), `sm_70` for Volta (V100)
- **Thrust tuples**: Use `cuda::std::tuple`, `cuda::std::make_tuple`, and `cuda::std::get` (the `thrust::tuple` variants are replaced)
