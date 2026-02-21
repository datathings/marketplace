# CUDA Workflows — Complete Working Examples

## Table of Contents
1. [Custom CUDA Kernel](#custom-kernel)
2. [cuBLAS GEMM](#cublas-gemm)
3. [cuFFT 1D Transform](#cufft-1d)
4. [Unified Memory](#unified-memory)
5. [Error-Checked CUDA Boilerplate](#error-checked-boilerplate)
6. [Thrust Sort and Reduce](#thrust-sort-and-reduce)

---

## Custom Kernel

Element-wise vector addition with proper launch configuration and error checking.

```c
// compile: nvcc -o vecadd vecadd.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do {                                             \
    cudaError_t _e = (err);                                              \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(_e)); exit(EXIT_FAILURE);             \
    }                                                                    \
} while(0)

__global__ void addVectors(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // Device arrays
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch: tune block size with occupancy API
    int blockSize, minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addVectors, 0, 0));
    int gridSize = (N + blockSize - 1) / blockSize;
    addVectors<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // D2H
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) { fprintf(stderr, "FAILED at %d\n", i); exit(1); }
    }
    printf("PASSED: all elements = 3.0\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```

---

## cuBLAS GEMM

Double-precision matrix multiply `C = A * B` using cuBLAS.

```cpp
// compile: nvcc -o gemm gemm.cu -lcublas
#include <cstdio>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(e)   do { cudaError_t     _s=(e); if(_s!=cudaSuccess)  { fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_s)); exit(1); } } while(0)
#define CUBLAS_CHECK(e) do { cublasStatus_t  _s=(e); if(_s!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS error %d at %s:%d\n",_s,__FILE__,__LINE__); exit(1); } } while(0)

int main() {
    // C (m x n) = A (m x k) * B (k x n)
    const int m = 4, n = 4, k = 4;
    const int lda = m, ldb = k, ldc = m;

    // Column-major: fill A and B
    std::vector<double> A(m * k), B(k * n), C(m * n, 0.0);
    for (int i = 0; i < m * k; i++) A[i] = i + 1;
    for (int i = 0; i < k * n; i++) B[i] = i + 1;

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));

    cublasHandle_t handle;
    cudaStream_t   stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), m*k*sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), k*n*sizeof(double), cudaMemcpyHostToDevice, stream));

    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha, d_A, lda,
                d_B, ldb,
        &beta,  d_C, ldc));

    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, m*n*sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("C[0,0] = %f (expected %f)\n", C[0], 4*1.0+4*2.0+4*3.0+4*4.0-30.0); // sanity

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```

**Column-major note:** cuBLAS uses Fortran column-major order. For a C row-major `A[M][N]`, pass it as `B` with transposed dimensions to cuBLAS — effectively computing `C^T = B^T * A^T` then treating result as `C`.

---

## cuFFT 1D

Batched 1D complex-to-complex FFT (forward + inverse round-trip with normalization).

```cpp
// compile: nvcc -o fft fft.cu -lcufft
#include <complex>
#include <cstdio>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(e)  do { cudaError_t _s=(e); if(_s!=cudaSuccess) { fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_s)); exit(1); } } while(0)
#define CUFFT_CHECK(e) do { cufftResult _r=(e); if(_r!=CUFFT_SUCCESS) { fprintf(stderr,"cuFFT error %d at %s:%d\n",_r,__FILE__,__LINE__); exit(1); } } while(0)

__global__ void scale(cufftComplex *data, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { data[i].x *= s; data[i].y *= s; }
}

int main() {
    const int FFT_SIZE = 8, BATCH = 2;
    const int N = FFT_SIZE * BATCH;

    std::vector<std::complex<float>> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = {(float)i, -(float)i};

    cufftComplex *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    cudaStream_t stream;
    CUFFT_CHECK(cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, BATCH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    // Forward FFT (in-place)
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // Normalize by 1/FFT_SIZE (cuFFT does NOT normalize automatically)
    int threads = 128;
    scale<<<(N + threads - 1) / threads, threads, 0, stream>>>(d_data, N, 1.0f / FFT_SIZE);

    // Inverse FFT (in-place)
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data, N * sizeof(cufftComplex),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Should recover original data
    printf("h_data[0] = (%.1f, %.1f) (expected (0, 0))\n",
           h_data[0].real(), h_data[0].imag());

    CUDA_CHECK(cudaFree(d_data));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```

---

## Unified Memory

Demonstrates `cudaMallocManaged` with prefetching and advice for optimal performance.

```cpp
// compile: nvcc -o umem umem.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(e) do { cudaError_t _s=(e); if(_s!=cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_s)); exit(1); } } while(0)

__global__ void init_and_double(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f;
}

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    const int N = 1 << 20;
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));

    // Initialize on CPU
    for (int i = 0; i < N; i++) data[i] = (float)i;

    // Hint: we're done with CPU access, prefetch to GPU to avoid page faults
    CUDA_CHECK(cudaMemPrefetchAsync(data, N * sizeof(float), device, NULL));

    int blockSize, minGrid;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, init_and_double, 0, 0));
    init_and_double<<<(N + blockSize - 1) / blockSize, blockSize>>>(data, N);
    CUDA_CHECK(cudaGetLastError());

    // Prefetch back to CPU before host reads
    CUDA_CHECK(cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, NULL));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("data[1] = %.0f (expected 2)\n", data[1]);
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```

---

## Error-Checked CUDA Boilerplate

Copy-paste header for new CUDA files with all error check macros.

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(err) do {                                                   \
    cudaError_t _e = (err);                                                    \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CUBLAS_CHECK(err) do {                                                 \
    cublasStatus_t _s = (err);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", _s, __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CUFFT_CHECK(err) do {                                                  \
    cufftResult _r = (err);                                                    \
    if (_r != CUFFT_SUCCESS) {                                                 \
        fprintf(stderr, "cuFFT error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CUSPARSE_CHECK(err) do {                                               \
    cusparseStatus_t _s = (err);                                               \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuSPARSE error %s at %s:%d\n",                       \
                cusparseGetErrorString(_s), __FILE__, __LINE__);               \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CUSOLVER_CHECK(err) do {                                               \
    cusolverStatus_t _s = (err);                                               \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuSolver error %d at %s:%d\n", _s,__FILE__,__LINE__);\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CURAND_CHECK(err) do {                                                 \
    curandStatus_t _s = (err);                                                 \
    if (_s != CURAND_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cuRAND error %d at %s:%d\n", _s, __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)
```

---

## Thrust Sort and Reduce

Sorting and reduction using Thrust with raw CUDA pointer interop.

```cpp
// compile: nvcc -o thrust_demo thrust_demo.cu
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

int main() {
    const int N = 1024;

    // Fill host vector with descending values
    thrust::host_vector<float> h(N);
    for (int i = 0; i < N; i++) h[i] = (float)(N - i);

    // Copy to device
    thrust::device_vector<float> d = h;

    // Sort in ascending order
    thrust::sort(d.begin(), d.end());

    // Sum reduction
    float sum = thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());
    printf("Sum = %.0f (expected %.0f)\n", sum, (float)N * (N + 1) / 2);

    // Square each element in-place using transform
    thrust::transform(d.begin(), d.end(), d.begin(),
                      [] __device__ (float x) { return x * x; });

    // Sum of squares
    float sum_sq = thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());
    printf("Sum of squares = %.0f\n", sum_sq);

    // Interop: get raw pointer and call custom kernel
    float *raw = thrust::raw_pointer_cast(d.data());
    // myKernel<<<grid, block>>>(raw, N);

    return 0;
}
```
