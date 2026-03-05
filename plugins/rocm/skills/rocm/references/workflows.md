# ROCm Development Workflows and Complete Examples

## Table of Contents
1. [HIP Hello World Kernel](#hip-hello-world-kernel)
2. [Vector Addition (SAXPY)](#vector-addition-saxpy)
3. [Matrix Multiplication with Shared Memory](#matrix-multiplication-with-shared-memory)
4. [rocBLAS GEMM Workflow](#rocblas-gemm-workflow)
5. [rocFFT 1D Complex Transform](#rocfft-1d-complex-transform)
6. [rocRAND Random Number Generation](#rocrand-random-number-generation)
7. [Multi-GPU Data Parallel Workflow](#multi-gpu-data-parallel-workflow)
8. [Streaming Overlap (Compute + Transfer)](#streaming-overlap-compute--transfer)
9. [Build System Setup](#build-system-setup)
10. [CUDA to HIP Porting](#cuda-to-hip-porting)
11. [Common Pitfalls](#common-pitfalls)

---

## HIP Hello World Kernel

The minimal template for any HIP program.

```cpp
// hello_world.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

// Device function — runs on GPU, called from GPU
__device__ unsigned int get_thread_idx() {
    return threadIdx.x;
}

// Global function — kernel entry point, called from host
__global__ void hello_kernel() {
    unsigned int tid = get_thread_idx();
    unsigned int bid = blockIdx.x;
    printf("Hello from block %u, thread %u!\n", bid, tid);
}

int main() {
    // Launch 2 blocks × 4 threads = 8 total threads
    hello_kernel<<<dim3(2), dim3(4), 0, hipStreamDefault>>>();

    // Wait for all GPU work to finish
    hipDeviceSynchronize();
    return 0;
}
```

**Build:**
```bash
hipcc -o hello_world hello_world.hip
./hello_world
```

---

## Vector Addition (SAXPY)

y[i] = a * x[i] + y[i] — canonical HIP workflow.

```cpp
// saxpy.hip
#include <hip/hip_runtime.h>
#include <vector>
#include <numeric>
#include <iostream>

#define HIP_CHECK(expr) do { \
    hipError_t err = (expr); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void saxpy_kernel(const float a, const float* x, float* y, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    constexpr unsigned int N          = 1'000'000;
    constexpr unsigned int BLOCK_SIZE = 256;
    constexpr unsigned int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr float        A          = 2.0f;

    // Host data
    std::vector<float> h_x(N), h_y(N);
    std::iota(h_x.begin(), h_x.end(), 1.f);
    std::fill(h_y.begin(), h_y.end(), 0.f);

    // Device allocation
    float *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));

    // Host → Device
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Launch
    saxpy_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, hipStreamDefault>>>(A, d_x, d_y, N);
    HIP_CHECK(hipGetLastError());

    // Device → Host
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    std::cout << "y[0..4]: ";
    for (int i = 0; i < 5; i++) std::cout << h_y[i] << " ";
    std::cout << "\n";
    return 0;
}
```

---

## Matrix Multiplication with Shared Memory

Tiled GEMM kernel — the classic shared-memory optimization.

```cpp
// matmul.hip
#include <hip/hip_runtime.h>
#include <vector>

template<unsigned int TILE>
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               unsigned int M, unsigned int K, unsigned int N) {
    // A: M×K, B: K×N, C: M×N (row-major)
    const unsigned int row = blockIdx.y * TILE + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    float sum = 0.f;
    const unsigned int num_tiles = (K + TILE - 1) / TILE;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        unsigned int a_col = t * TILE + threadIdx.x;
        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // Load tile from B
        unsigned int b_row = t * TILE + threadIdx.y;
        tile_B[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        for (unsigned int k = 0; k < TILE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul(const float* h_A, const float* h_B, float* h_C,
            unsigned int M, unsigned int K, unsigned int N) {
    constexpr unsigned int TILE = 16;

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);

    matmul_kernel<TILE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    hipDeviceSynchronize();

    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
}
```

**Tip:** For production, use rocBLAS `rocblas_sgemm` instead — it is highly tuned and orders of magnitude faster than a custom kernel.

---

## rocBLAS GEMM Workflow

C = alpha * A * B + beta * C, using the rocBLAS library.

```cpp
// rocblas_gemm.cpp
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <numeric>

#define ROCBLAS_CHECK(expr) do { \
    rocblas_status s = (expr); \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    const rocblas_int M = 512, N = 512, K = 512;
    const float alpha = 1.f, beta = 0.f;

    // Host matrices (column-major for BLAS)
    std::vector<float> h_A(M * K, 1.f);
    std::vector<float> h_B(K * N, 1.f);
    std::vector<float> h_C(M * N, 0.f);

    // Device allocation
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, M * N * sizeof(float));

    // Create rocBLAS handle
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Alpha/beta on host by default
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // SGEMM: C = alpha*A*B + beta*C
    // rocBLAS uses column-major convention. For row-major matrices A(m×k) and B(k×n):
    // Swap A and B, and swap M and N:
    ROCBLAS_CHECK(rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        N, M, K,              // (N, M, K) for row-major A×B
        &alpha,
        d_B, N,               // B (becomes "A" in col-major view)
        d_A, K,               // A (becomes "B")
        &beta,
        d_C, N));

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_A); hipFree(d_B); hipFree(d_C);

    printf("C[0][0] = %f (expected %f)\n", h_C[0], (float)K);
    return 0;
}
```

**Build:**
```bash
hipcc -o rocblas_gemm rocblas_gemm.cpp -lrocblas
```

**Column-major vs Row-major:**
BLAS libraries use column-major ordering. For row-major storage (C-style), either:
- Transpose A and B before calling, or
- Swap the A/B arguments and M/N dimensions (as shown above).

---

## rocFFT 1D Complex Transform

```cpp
// rocfft_1d.cpp
#include <rocfft/rocfft.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <cmath>
#include <stdio.h>

#define ROCFFT_CHECK(expr) do { \
    rocfft_status s = (expr); \
    if (s != rocfft_status_success) { \
        fprintf(stderr, "rocFFT error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    const size_t N = 256;
    rocfft_setup();

    // Host data: complex signal (real tone at k=4)
    std::vector<hipDoubleComplex> h_data(N);
    for (size_t i = 0; i < N; i++) {
        double angle = 2.0 * M_PI * 4 * i / N;
        h_data[i] = {std::cos(angle), std::sin(angle)};
    }

    // Allocate device buffer
    hipDoubleComplex* d_data;
    hipMalloc(&d_data, N * sizeof(hipDoubleComplex));
    hipMemcpy(d_data, h_data.data(), N * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

    // Create plan: 1D forward C2C, double precision, in-place
    size_t lengths[] = {N};
    rocfft_plan plan;
    ROCFFT_CHECK(rocfft_plan_create(&plan,
        rocfft_placement_inplace,
        rocfft_transform_type_complex_forward,
        rocfft_precision_double,
        1, lengths, 1, nullptr));

    // Work buffer
    rocfft_execution_info info;
    ROCFFT_CHECK(rocfft_execution_info_create(&info));
    size_t wbuf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &wbuf_size));
    void* wbuf = nullptr;
    if (wbuf_size > 0) {
        hipMalloc(&wbuf, wbuf_size);
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info, wbuf, wbuf_size));
    }

    // Execute FFT
    ROCFFT_CHECK(rocfft_execute(plan, (void**)&d_data, nullptr, info));

    // Get results
    hipMemcpy(h_data.data(), d_data, N * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

    // Peak should be at bin 4 (input frequency)
    double max_mag = 0;
    size_t peak_bin = 0;
    for (size_t i = 0; i < N; i++) {
        double mag = std::hypot(h_data[i].x, h_data[i].y);
        if (mag > max_mag) { max_mag = mag; peak_bin = i; }
    }
    printf("Peak at bin %zu (expected 4), magnitude %.1f\n", peak_bin, max_mag);

    // Cleanup
    if (wbuf) hipFree(wbuf);
    ROCFFT_CHECK(rocfft_execution_info_destroy(info));
    ROCFFT_CHECK(rocfft_plan_destroy(plan));
    ROCFFT_CHECK(rocfft_cleanup());
    hipFree(d_data);
    return 0;
}
```

**Build:**
```bash
hipcc -o rocfft_1d rocfft_1d.cpp -lrocfft
```

---

## rocRAND Random Number Generation

```cpp
// rocrand_example.cpp
#include <rocrand/rocrand.hpp>
#include <hip/hip_runtime.h>
#include <vector>
#include <numeric>
#include <stdio.h>

int main() {
    const size_t N = 10'000'000;

    // Allocate device buffer
    float* d_vals;
    hipMalloc(&d_vals, N * sizeof(float));

    // C++ API: engine + distribution
    rocrand_cpp::default_random_engine engine(42 /* seed */);
    rocrand_cpp::uniform_real_distribution<float> uniform;

    // Generate N uniform [0,1) floats on GPU
    uniform(engine, d_vals, N);
    hipDeviceSynchronize();

    // Copy back and compute mean
    std::vector<float> h_vals(N);
    hipMemcpy(h_vals.data(), d_vals, N * sizeof(float), hipMemcpyDeviceToHost);

    double mean = std::accumulate(h_vals.begin(), h_vals.end(), 0.0) / N;
    printf("Mean of %zu uniform samples: %.6f (expected 0.5)\n", N, mean);

    // Normal distribution
    rocrand_cpp::normal_distribution<float> normal(0.f, 1.f);
    normal(engine, d_vals, N);
    hipMemcpy(h_vals.data(), d_vals, N * sizeof(float), hipMemcpyDeviceToHost);

    double mean2 = std::accumulate(h_vals.begin(), h_vals.end(), 0.0) / N;
    printf("Mean of %zu normal samples: %.6f (expected ~0.0)\n", N, mean2);

    hipFree(d_vals);
    return 0;
}
```

**Build:**
```bash
hipcc -o rocrand_example rocrand_example.cpp -lrocrand
```

---

## Multi-GPU Data Parallel Workflow

Distribute work across all available GPUs.

```cpp
// multi_gpu.hip
#include <hip/hip_runtime.h>
#include <vector>
#include <thread>

__global__ void scale_kernel(float* data, float factor, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

void process_on_device(int device_id, const std::vector<float>& h_chunk,
                       std::vector<float>& h_result) {
    hipSetDevice(device_id);

    const size_t n = h_chunk.size();
    float* d_data;
    hipMalloc(&d_data, n * sizeof(float));
    hipMemcpy(d_data, h_chunk.data(), n * sizeof(float), hipMemcpyHostToDevice);

    const unsigned int block = 256;
    const unsigned int grid  = (n + block - 1) / block;
    scale_kernel<<<dim3(grid), dim3(block)>>>(d_data, 2.0f, n);
    hipDeviceSynchronize();

    hipMemcpy(h_result.data(), d_data, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_data);
}

int main() {
    int num_devices;
    hipGetDeviceCount(&num_devices);
    printf("Found %d GPU(s)\n", num_devices);

    const size_t TOTAL = 1'000'000;
    const size_t chunk = TOTAL / num_devices;

    std::vector<float> h_data(TOTAL, 1.f);
    std::vector<std::vector<float>> h_results(num_devices, std::vector<float>(chunk));
    std::vector<std::thread> threads;

    for (int dev = 0; dev < num_devices; ++dev) {
        std::vector<float> h_chunk(h_data.begin() + dev * chunk,
                                   h_data.begin() + (dev + 1) * chunk);
        threads.emplace_back(process_on_device, dev, h_chunk, std::ref(h_results[dev]));
    }

    for (auto& t : threads) t.join();

    // Combine results
    float total = 0.f;
    for (auto& r : h_results)
        for (float v : r) total += v;
    printf("Sum: %.0f (expected %.0f)\n", (double)total, (double)TOTAL * 2);
    return 0;
}
```

---

## Streaming Overlap (Compute + Transfer)

Overlap PCIe data transfer with GPU computation using multiple streams.

```cpp
// streaming_overlap.hip
#include <hip/hip_runtime.h>
#include <vector>

__global__ void process_kernel(float* data, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * data[i];  // element-wise square
}

int main() {
    const int NUM_STREAMS = 4;
    const size_t TOTAL    = 4'000'000;
    const size_t CHUNK    = TOTAL / NUM_STREAMS;
    const size_t BYTES    = CHUNK * sizeof(float);

    // Pinned host memory — required for async transfers
    float* h_input;
    float* h_output;
    hipHostMalloc(&h_input, TOTAL * sizeof(float));
    hipHostMalloc(&h_output, TOTAL * sizeof(float));

    // Initialize input
    for (size_t i = 0; i < TOTAL; i++) h_input[i] = (float)i;

    // Create streams and device buffers
    std::vector<hipStream_t> streams(NUM_STREAMS);
    std::vector<float*> d_bufs(NUM_STREAMS);
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStreamCreate(&streams[s]);
        hipMalloc(&d_bufs[s], BYTES);
    }

    // Pipeline: transfer → compute → transfer back (overlapping across streams)
    constexpr unsigned int BLOCK = 256;
    const unsigned int GRID = (CHUNK + BLOCK - 1) / BLOCK;

    for (int s = 0; s < NUM_STREAMS; s++) {
        float* h_in  = h_input  + s * CHUNK;
        float* h_out = h_output + s * CHUNK;

        // Async copy host→device
        hipMemcpyAsync(d_bufs[s], h_in, BYTES, hipMemcpyHostToDevice, streams[s]);

        // Kernel launch in same stream (starts after copy completes)
        process_kernel<<<dim3(GRID), dim3(BLOCK), 0, streams[s]>>>(d_bufs[s], CHUNK);

        // Async copy device→host (starts after kernel completes)
        hipMemcpyAsync(h_out, d_bufs[s], BYTES, hipMemcpyDeviceToHost, streams[s]);
    }

    // Wait for all streams
    hipDeviceSynchronize();

    // Cleanup
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStreamDestroy(streams[s]);
        hipFree(d_bufs[s]);
    }
    hipHostFree(h_input);
    hipHostFree(h_output);
    return 0;
}
```

**Key insight:** Operations in the same stream are serialized; operations in different streams can run concurrently. Pinned (`hipHostMalloc`) host memory is required for truly asynchronous transfers.

---

## Build System Setup

### Makefile
```makefile
HIPCC   = hipcc
CXXFLAGS = -O2 -std=c++17
LIBS    = -lrocblas -lrocfft -lrocrand -lrocsolver

all: my_app

my_app: main.hip
	$(HIPCC) $(CXXFLAGS) -o $@ $< $(LIBS)

# For GPU_RUNTIME=CUDA portability:
# HIPCC = hipcc
# (same command — HIP selects backend automatically)

clean:
	rm -f my_app
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.21)
project(MyROCmProject LANGUAGES CXX)

# Enable HIP language
enable_language(HIP)

find_package(hip REQUIRED)

# Optional library packages
find_package(rocblas REQUIRED)
find_package(rocfft REQUIRED)
find_package(rocrand REQUIRED)

add_executable(my_app main.hip)

# HIP files need LANGUAGE HIP property if extension is .cpp
# set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(my_app PRIVATE
    hip::host
    roc::rocblas
    roc::rocfft
    rocrand)

# Target GPU architectures
set_property(TARGET my_app PROPERTY HIP_ARCHITECTURES gfx90a gfx1100)
```

**Configure and build:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/my_app
```

### ROCm Installation Paths
```
/opt/rocm/                    — ROCm root
/opt/rocm/include/hip/        — HIP headers
/opt/rocm/include/rocblas/    — rocBLAS headers
/opt/rocm/include/rocfft/     — rocFFT headers
/opt/rocm/include/rocrand/    — rocRAND headers
/opt/rocm/lib/                — libraries (.so)
/opt/rocm/bin/hipcc           — HIP compiler
/opt/rocm/bin/rocm-smi        — system monitor
/opt/rocm/bin/rocgdb          — GPU debugger
```

---

## CUDA to HIP Porting

HIP is designed to be CUDA-compatible. `hipify-perl` or `hipify-clang` automatically convert CUDA code.

### Automated Porting
```bash
# Install hipify (included with ROCm)
# Convert a single file
hipify-perl cuda_code.cu > hip_code.hip

# Convert entire directory
hipify-perl -inplace *.cu *.cuh

# More accurate (uses Clang):
hipify-clang cuda_code.cu -- -x cuda --cuda-path=/usr/local/cuda
```

### Manual Mapping (key API equivalents)
| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cudaEventCreate` | `hipEventCreate` |
| `cudaEventElapsedTime` | `hipEventElapsedTime` |
| `cudaGetLastError` | `hipGetLastError` |
| `cudaSuccess` | `hipSuccess` |
| `__syncthreads()` | `__syncthreads()` (same) |
| `atomicAdd` | `atomicAdd` (same) |
| `cublasCreate` | `rocblas_create_handle` / `hipblasCreate` |
| `cufftPlan1d` | `hipfftPlan1d` |
| `curandCreateGenerator` | `rocrand_create_generator` |

### Platform Detection
```cpp
// Detect platform in source code
#ifdef __HIP_PLATFORM_AMD__
    // AMD-specific code
    printf("Running on AMD GPU\n");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    // NVIDIA-specific code
    printf("Running on NVIDIA GPU (via HIP)\n");
#endif
```

---

## Common Pitfalls

### 1. Missing `hipDeviceSynchronize()` before reading results
```cpp
// WRONG: h_result may be garbage
my_kernel<<<grid, block>>>(d_result);
printf("%f\n", h_result[0]);  // undefined!

// CORRECT:
my_kernel<<<grid, block>>>(d_result);
hipMemcpy(h_result, d_result, size, hipMemcpyDeviceToHost);  // implicit sync
// OR explicitly:
hipDeviceSynchronize();
```

### 2. Kernel launch errors are asynchronous
```cpp
my_kernel<<<grid, block>>>(args);
// The above line returns immediately even if kernel will fail.
// Check errors:
hipError_t err = hipGetLastError();   // catches launch config errors
if (err != hipSuccess) { /* handle */ }
hipDeviceSynchronize();               // catches runtime errors in kernel
err = hipGetLastError();
```

### 3. rocBLAS column-major convention
BLAS uses column-major storage. For row-major C arrays: pass `N, M` instead of `M, N` and swap A/B pointers. Alternatively, use the transpose to compensate: `C^T = B^T * A^T`.

### 4. Wavefront size on AMD (64, not 32)
On AMD GPUs, `warpSize = 64` (wavefront). Code that assumes `warpSize == 32` will miscompute reductions. Always use the `warpSize` built-in.

```cpp
// WRONG (assumes 32):
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);  // only works for warpSize=32

// CORRECT (portable):
for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
```

### 5. hipMemcpyAsync requires pinned memory
`hipMemcpyAsync` with pageable (regular malloc) host memory falls back to synchronous behavior. Use `hipHostMalloc` for true async transfers.

### 6. rocFFT normalization
rocFFT inverse transforms do not normalize automatically. After C2C inverse FFT, divide all elements by N.

### 7. Device pointer arithmetic is undefined on host
Never dereference or do arithmetic on device pointers from host code:
```cpp
float* d_ptr;
hipMalloc(&d_ptr, N * sizeof(float));
d_ptr[0] = 1.f;          // WRONG: segfault on AMD, undefined behavior
hipMemset(d_ptr, 0, N);  // CORRECT
```

### 8. Shared memory size limits
Check `props.sharedMemPerBlock` (typically 64 KB on CDNA, 32 KB on RDNA). Exceeding it silently reduces occupancy or causes launch failure.

```cpp
// Request more shared memory per block (if supported)
hipFuncSetAttribute(my_kernel,
    hipFuncAttributeMaxDynamicSharedMemorySize,
    max_shared);
```
