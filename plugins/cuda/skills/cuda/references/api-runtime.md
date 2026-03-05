# CUDA Runtime API

## Table of Contents
1. [Device Management](#device-management)
2. [Memory Management](#memory-management)
3. [Unified Memory](#unified-memory)
4. [Stream Management](#stream-management)
5. [Event Management](#event-management)
6. [Kernel Launch](#kernel-launch)
7. [Error Handling](#error-handling)

---

## Device Management

### `cudaGetDeviceCount(int *count) -> cudaError_t`
**Description:** Returns number of CUDA-capable devices.
**Example:**
```c
int nDevices;
cudaGetDeviceCount(&nDevices);
```

### `cudaSetDevice(int device) -> cudaError_t`
**Description:** Sets device to use for GPU execution.
**Parameters:** `device` — zero-based device index.

### `cudaGetDeviceProperties(cudaDeviceProp *prop, int device) -> cudaError_t`
**Description:** Returns properties for the specified device (SM count, memory, warp size, etc.).
**Example:**
```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Device: %s, SMs: %d, Mem: %zu MB\n",
       prop.name, prop.multiProcessorCount,
       prop.totalGlobalMem / (1024*1024));
```

### `cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) -> cudaError_t`
**Description:** Query a single device attribute (e.g., `cudaDevAttrMaxThreadsPerBlock`).

### `cudaDeviceSynchronize() -> cudaError_t`
**Description:** Blocks host until all GPU work on current device completes.

### `cudaDeviceReset() -> cudaError_t`
**Description:** Destroys all allocations and resets device state. Call at program end.

---

## Memory Management

### `cudaMalloc(void **devPtr, size_t size) -> cudaError_t`
**Description:** Allocates linear memory on the device.
**Example:**
```c
float *d_A;
cudaMalloc((void **)&d_A, N * sizeof(float));
```

### `cudaFree(void *devPtr) -> cudaError_t`
**Description:** Frees device memory previously allocated by `cudaMalloc`.

### `cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) -> cudaError_t`
**Description:** Copies `count` bytes between host and device (synchronous).
**Parameters:** `kind` — `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`.
**Example:**
```c
cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
```

### `cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) -> cudaError_t`
**Description:** Asynchronous copy; overlaps with kernel execution on other streams.
**Example:**
```c
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
```

### `cudaMemset(void *devPtr, int value, size_t count) -> cudaError_t`
**Description:** Sets device memory to a byte value.
**Example:**
```c
cudaMemset(d_C, 0, N * sizeof(float));
```

### `cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) -> cudaError_t`
**Description:** Allocates pitched (row-padded) 2D memory for coalesced access.

### `cudaMalloc3DArray` / `cudaMemcpy3D`
**Description:** For 3D arrays and textures. Use `cudaExtent` to specify dimensions.

---

## Unified Memory

### `cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=cudaMemAttachGlobal) -> cudaError_t`
**Description:** Allocates managed memory accessible by both CPU and GPU. Data migrates automatically.
**Example:**
```c
float *data;
cudaMallocManaged(&data, N * sizeof(float));
// Write on host
for (int i = 0; i < N; i++) data[i] = i;
// Use on device
kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();
// Read on host
printf("%f\n", data[0]);
cudaFree(data);
```

### `cudaMemAdvise(const void *devPtr, size_t count, cudaMemoryAdvise advice, int device) -> cudaError_t`
**Description:** Hints memory placement policy.
**Common advice:** `cudaMemAdviseSetReadMostly`, `cudaMemAdviseSetPreferredLocation`, `cudaMemAdviseSetAccessedBy`.

### `cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) -> cudaError_t`
**Description:** Pre-migrates managed memory to `dstDevice` before kernel launch to avoid page faults.

### `cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags) -> cudaError_t`
**Description:** Attaches managed memory to a stream. `cudaMemAttachHost` lets CPU access while GPU runs on other streams; `cudaMemAttachSingle` restricts to one stream.

---

## Stream Management

### `cudaStreamCreate(cudaStream_t *pStream) -> cudaError_t`
**Description:** Creates an asynchronous stream.

### `cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) -> cudaError_t`
**Description:** `flags=cudaStreamNonBlocking` prevents synchronization with the null (default) stream.

### `cudaStreamSynchronize(cudaStream_t stream) -> cudaError_t`
**Description:** Blocks host until all operations in `stream` complete.

### `cudaStreamDestroy(cudaStream_t stream) -> cudaError_t`
**Description:** Destroys a stream. Waits for pending operations first.

### `cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags=0) -> cudaError_t`
**Description:** Makes `stream` wait until `event` is recorded before executing further work.

---

## Event Management

### `cudaEventCreate(cudaEvent_t *event) -> cudaError_t`
**Description:** Creates an event for timing or stream synchronization.

### `cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0) -> cudaError_t`
**Description:** Records an event in the stream's work queue.

### `cudaEventSynchronize(cudaEvent_t event) -> cudaError_t`
**Description:** Blocks host until the event has been recorded on device.

### `cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) -> cudaError_t`
**Description:** Returns elapsed time in milliseconds between two recorded events.
**Example:**
```c
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start, 0);
kernel<<<grid, block>>>(d_A, d_B, d_C, N);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
cudaEventDestroy(start); cudaEventDestroy(stop);
```

### `cudaEventDestroy(cudaEvent_t event) -> cudaError_t`
**Description:** Destroys an event object.

---

## Kernel Launch

### Triple-angle-bracket syntax: `kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...)`
**Description:** Launches a `__global__` kernel.
- `gridDim` — `dim3` or `int` specifying number of blocks in x/y/z.
- `blockDim` — `dim3` or `int` specifying threads per block in x/y/z (max 1024 total).
- `sharedMem` — bytes of dynamic shared memory per block (optional, default 0).
- `stream` — `cudaStream_t` (optional, default NULL = default stream).

**Example:**
```c
dim3 block(32, 32);
dim3 grid((W + 31) / 32, (H + 31) / 32);
matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
```

### `cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, T func, size_t dynamicSMemSize=0, int blockSizeLimit=0) -> cudaError_t`
**Description:** Suggests (gridSize, blockSize) that maximizes occupancy.
**Example:**
```c
int blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(d_data, N);
```

### `cudaFuncSetAttribute(const void *func, cudaFuncAttribute attr, int value) -> cudaError_t`
**Description:** Sets kernel attributes, e.g., `cudaFuncAttributeMaxDynamicSharedMemorySize` to request more than 48 KB shared memory (Volta+).

---

## Error Handling

### `cudaGetLastError() -> cudaError_t`
**Description:** Returns and clears the last error from a CUDA call or kernel launch.

### `cudaGetErrorString(cudaError_t error) -> const char*`
**Description:** Returns a human-readable error string.

### Recommended error-check macro:
```c
#define CUDA_CHECK(err) do {                                         \
    cudaError_t _e = (err);                                          \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_A, size));
kernel<<<grid, block>>>(d_A, N);
CUDA_CHECK(cudaGetLastError());   // catch async kernel errors
CUDA_CHECK(cudaDeviceSynchronize());
```

### `cudaPeekAtLastError() -> cudaError_t`
**Description:** Returns the last error without clearing it.
