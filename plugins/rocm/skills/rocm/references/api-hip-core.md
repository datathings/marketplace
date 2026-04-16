# HIP Core Runtime API Reference

## Table of Contents
1. [Initialization and Device Management](#initialization-and-device-management)
2. [Memory Management](#memory-management)
3. [Stream-Ordered Memory Allocation](#stream-ordered-memory-allocation)
4. [Kernel Launch](#kernel-launch)
5. [Streams](#streams)
6. [Events and Timing](#events-and-timing)
7. [Error Handling](#error-handling)
8. [HIP Graphs](#hip-graphs)
9. [Peer Access (Multi-GPU)](#peer-access-multi-gpu)
10. [Unified Memory](#unified-memory)
11. [Occupancy API](#occupancy-api)
12. [Library API](#library-api)
13. [Function Qualifiers and Built-ins](#function-qualifiers-and-built-ins)

---

## Initialization and Device Management

```cpp
#include <hip/hip_runtime.h>

// Count available GPUs
hipError_t hipGetDeviceCount(int* count);

// Set active device (default: device 0)
hipError_t hipSetDevice(int device);
hipError_t hipGetDevice(int* device);

// Query device properties
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);

// Key fields of hipDeviceProp_t:
//   prop.name               — device name string
//   prop.totalGlobalMem     — total global memory in bytes
//   prop.sharedMemPerBlock  — max shared mem per block (bytes)
//   prop.maxThreadsPerBlock — max threads per block (usually 1024)
//   prop.maxThreadsDim[3]   — max threads per dim (x/y/z)
//   prop.maxGridSize[3]     — max grid dims
//   prop.warpSize           — warp size (64 on AMD, 32 on NVIDIA)
//   prop.clockRate          — GPU clock in kHz
//   prop.gcnArchName        — AMD arch name (AMD only, e.g. "gfx1100")
//   prop.multiProcessorCount — number of compute units (CUs)

// Synchronize all work on current device
hipError_t hipDeviceSynchronize();

// Reset device (frees all resources on device)
hipError_t hipDeviceReset();
```

### Example: Device Query
```cpp
int device_count;
hipGetDeviceCount(&device_count);
for (int i = 0; i < device_count; i++) {
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, i);
    printf("Device %d: %s, %.1f GiB global mem, %d CUs\n",
           i, props.name,
           props.totalGlobalMem / (1024.0*1024.0*1024.0),
           props.multiProcessorCount);
}
```

---

## Memory Management

### Device Memory
```cpp
// Allocate device memory
hipError_t hipMalloc(void** ptr, size_t size);

// Free device memory
hipError_t hipFree(void* ptr);

// Synchronous copy (blocks host until done)
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
// kind: hipMemcpyHostToDevice | hipMemcpyDeviceToHost |
//       hipMemcpyDeviceToDevice | hipMemcpyHostToHost | hipMemcpyDefault

// Asynchronous copy (requires pinned host memory for true async)
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                           hipMemcpyKind kind, hipStream_t stream);

// Zero-fill device memory
hipError_t hipMemset(void* dst, int value, size_t count);
hipError_t hipMemsetAsync(void* dst, int value, size_t count, hipStream_t stream);

// 2D memory operations
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                        size_t width, size_t height, hipMemcpyKind kind);

// Query free/total memory on current device
hipError_t hipMemGetInfo(size_t* free, size_t* total);
```

### Pinned (Page-Locked) Host Memory
```cpp
// Allocate pinned host memory (enables async DMA, faster transfers)
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags = 0);
// flags: hipHostMallocDefault | hipHostMallocMapped | hipHostMallocWriteCombined

// Free pinned host memory
hipError_t hipHostFree(void* ptr);

// Get device pointer for mapped pinned memory
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
```

### Managed/Unified Memory
```cpp
// Allocate managed memory accessible from both host and device
hipError_t hipMallocManaged(void** ptr, size_t size,
                             unsigned int flags = hipMemAttachGlobal);

// Prefetch managed memory to a device (-1 = CPU)
hipError_t hipMemPrefetchAsync(const void* ptr, size_t count,
                                int dst_device, hipStream_t stream);
```

---

## Stream-Ordered Memory Allocation

Stream-ordered memory (SOMA) ties allocation lifetime to stream execution order, enabling efficient memory reuse without explicit synchronization.

```cpp
// Allocate device memory on a stream (freed when stream reaches hipFreeAsync)
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);

// Free stream-ordered allocation
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
```

### Memory Pools
```cpp
// Get/set the default memory pool for a device
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* pool, int device);
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t pool);

// Create a custom memory pool
hipMemPoolProps props = {};
props.allocType = hipMemAllocationTypePinned;
props.location.type = hipMemLocationTypeDevice;
props.location.id = device_id;
hipMemPool_t pool;
hipMemPoolCreate(&pool, &props);

// Set pool release threshold (0 = return memory to OS immediately)
uint64_t threshold = 0;
hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold);

hipMemPoolDestroy(pool);
```

### Pattern
```cpp
hipStream_t stream;
hipStreamCreate(&stream);

int* d_data;
hipMallocAsync(&d_data, N * sizeof(int), stream);

my_kernel<<<grid, block, 0, stream>>>(d_data, N);

// Copy results while still on stream
hipMemcpyAsync(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost, stream);

hipFreeAsync(d_data, stream);  // freed when stream reaches this point
hipStreamSynchronize(stream);
```

---

## Kernel Launch

### Triple-Angle-Bracket Syntax
```cpp
// kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...)
my_kernel<<<dim3(grid_x, grid_y, grid_z),
            dim3(block_x, block_y, block_z),
            shared_mem_bytes,
            stream>>>(arg1, arg2, ...);

// Check kernel launch errors
hipError_t err = hipGetLastError();
```

### hipLaunchKernelGGL (alternative)
```cpp
hipLaunchKernelGGL(kernel_name, gridDim, blockDim, sharedMem, stream, args...);
```

### Grid/Block Sizing Pattern
```cpp
constexpr int BLOCK_SIZE = 256;  // typical: 128, 256, or 512
int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;  // ceiling division

kernel<<<dim3(grid_size), dim3(BLOCK_SIZE), 0, hipStreamDefault>>>(d_data, n);
```

### Built-in Coordinate Variables (device only)
```cpp
threadIdx.x / .y / .z   // thread index within block
blockIdx.x  / .y / .z   // block index within grid
blockDim.x  / .y / .z   // block dimensions (threads per block)
gridDim.x   / .y / .z   // grid dimensions (blocks per grid)

// Common pattern: 1D global thread index
int gid = blockIdx.x * blockDim.x + threadIdx.x;
// 2D:
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## Streams

```cpp
// Create/destroy a stream
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamDestroy(hipStream_t stream);

// Create stream with priority (lower = higher priority)
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);

// Asynchronous memcpy within a stream
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t count,
                           hipMemcpyKind kind, hipStream_t stream);

// Block host until all ops in stream complete
hipError_t hipStreamSynchronize(hipStream_t stream);

// Non-blocking query of stream completion
hipError_t hipStreamQuery(hipStream_t stream);
// Returns hipSuccess (done) or hipErrorNotReady (still running)

// Make stream2 wait for stream1 event before proceeding
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
```

### Multi-stream Pattern (overlap compute + transfer)
```cpp
hipStream_t stream_compute, stream_transfer;
hipStreamCreate(&stream_compute);
hipStreamCreate(&stream_transfer);

// Overlap: copy next chunk while computing current
hipMemcpyAsync(d_buf_next, h_next, size, hipMemcpyHostToDevice, stream_transfer);
my_kernel<<<grid, block, 0, stream_compute>>>(d_buf_cur, n);

hipDeviceSynchronize();  // wait for all streams
hipStreamDestroy(stream_compute);
hipStreamDestroy(stream_transfer);
```

---

## Events and Timing

```cpp
// Create and destroy events
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventDestroy(hipEvent_t event);

// Record an event in a stream (timestamp when GPU reaches this point)
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr);

// Block host until event is complete
hipError_t hipEventSynchronize(hipEvent_t event);

// Elapsed time between two events (milliseconds, float)
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
```

### Timing Example
```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, nullptr);
my_kernel<<<grid, block>>>(args);
hipEventRecord(stop, nullptr);
hipEventSynchronize(stop);

float ms;
hipEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);

hipEventDestroy(start);
hipEventDestroy(stop);
```

---

## Error Handling

```cpp
// Returns a string description of the error code
const char* hipGetErrorString(hipError_t error);
const char* hipGetErrorName(hipError_t error);

// Common error codes:
// hipSuccess               — 0, no error
// hipErrorInvalidValue     — invalid argument
// hipErrorOutOfMemory      — allocation failure
// hipErrorNotReady         — async operation not yet complete
// hipErrorInvalidDevice    — invalid device ordinal
// hipErrorLaunchFailure    — kernel launch failed

// Idiomatic error-check macro (from examples):
#define HIP_CHECK(expression)                                        \
    {                                                                \
        const hipError_t err = (expression);                         \
        if(err != hipSuccess) {                                      \
            std::cerr << "HIP error: " << hipGetErrorString(err)     \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    }

// Usage:
HIP_CHECK(hipMalloc(&d_ptr, size));
HIP_CHECK(hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice));
// After kernel launch:
my_kernel<<<grid, block>>>(args);
HIP_CHECK(hipGetLastError());
HIP_CHECK(hipDeviceSynchronize());
```

---

## HIP Graphs

HIP Graphs capture a sequence of operations (kernels, memcpy, memset) into a reusable DAG, reducing launch overhead for repeated execution.

### Graph Creation (Explicit API)
```cpp
hipGraph_t graph;
hipGraphCreate(&graph, 0);

// Add memory allocation node
hipMemAllocNodeParams alloc_params = {};
alloc_params.poolProps.allocType = hipMemAllocationTypePinned;
alloc_params.poolProps.location.type = hipMemLocationTypeDevice;
alloc_params.poolProps.location.id = 0;
alloc_params.bytesize = N * sizeof(float);

hipGraphNode_t alloc_node;
hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_params);
float* d_ptr = (float*)alloc_params.dptr;

// Add kernel node (depends on alloc_node)
hipKernelNodeParams kern_params = {};
kern_params.func = (void*)my_kernel;
kern_params.gridDim = dim3(grid);
kern_params.blockDim = dim3(block);
void* args[] = {&d_ptr, &N};
kern_params.kernelParams = args;

hipGraphNode_t kern_node;
hipGraphAddKernelNode(&kern_node, graph, &alloc_node, 1, &kern_params);

// Add memcpy, memset, host function nodes similarly:
// hipGraphAddMemcpyNode1D, hipGraphAddMemsetNode, hipGraphAddHostNode

// Add free node
hipGraphNode_t free_node;
hipGraphAddMemFreeNode(&free_node, graph, &kern_node, 1, d_ptr);

// Instantiate and launch
hipGraphExec_t exec;
hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
hipGraphLaunch(exec, stream);
hipStreamSynchronize(stream);

hipGraphExecDestroy(exec);
hipGraphDestroy(graph);
```

### Stream Capture (Automatic Graph Construction)
```cpp
hipStream_t stream;
hipStreamCreate(&stream);

// Capture operations into a graph
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

hipMallocAsync(&d_ptr, N * sizeof(float), stream);
hipMemcpyAsync(d_ptr, h_ptr, N * sizeof(float), hipMemcpyHostToDevice, stream);
my_kernel<<<grid, block, 0, stream>>>(d_ptr, N);
hipMemcpyAsync(h_ptr, d_ptr, N * sizeof(float), hipMemcpyDeviceToHost, stream);
hipFreeAsync(d_ptr, stream);

hipGraph_t graph;
hipStreamEndCapture(stream, &graph);

// Instantiate and reuse
hipGraphExec_t exec;
hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int iter = 0; iter < 100; ++iter)
    hipGraphLaunch(exec, stream);

hipStreamSynchronize(stream);
hipGraphExecDestroy(exec);
hipGraphDestroy(graph);
```

**When to use graphs:** Repeated execution of the same operation sequence (e.g., inference loops). Graph launch overhead is much lower than individual API calls.

---

## Peer Access (Multi-GPU)

```cpp
// Check if device i can access device j's memory directly (NVLink/xGMI)
int can_access;
hipDeviceCanAccessPeer(&can_access, i, j);

// Enable/disable peer access
hipDeviceEnablePeerAccess(peer_device, 0);
hipDeviceDisablePeerAccess(peer_device);

// Peer-to-peer async copy
hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, size, stream);
```

---

## Unified Memory

```cpp
// hipMallocManaged — accessible from host and any device
float* um_ptr;
HIP_CHECK(hipMallocManaged(&um_ptr, N * sizeof(float)));

// Use directly on host:
for (int i = 0; i < N; i++) um_ptr[i] = (float)i;

// Use directly in kernel (no explicit copy needed):
my_kernel<<<grid, block>>>(um_ptr, N);
HIP_CHECK(hipDeviceSynchronize());

// Results available on host immediately after sync
printf("um_ptr[0] = %f\n", um_ptr[0]);

HIP_CHECK(hipFree(um_ptr));
```

---

## Occupancy API

```cpp
// Get the block size that maximizes occupancy for a kernel
int min_grid_size = 0, block_size = 0;
HIP_CHECK(hipOccupancyMaxPotentialBlockSize(
    &min_grid_size,        // minimum grid size for full occupancy
    &block_size,           // recommended block size
    my_kernel,             // kernel function pointer
    0,                     // dynamic shared memory bytes
    0));                   // max block size (0 = no limit)

int grid_size = (N + block_size - 1) / block_size;
my_kernel<<<dim3(grid_size), dim3(block_size)>>>(args);

// Query active blocks per multiprocessor for a given block size
int num_blocks;
HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, my_kernel, block_size, 0));

hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);
double occupancy = (double)num_blocks * block_size / props.maxThreadsPerMultiProcessor;
printf("Occupancy: %.1f%%\n", occupancy * 100);

// Query available dynamic shared memory per block (ROCm 7.2+)
size_t dyn_smem;
hipOccupancyAvailableDynamicSMemPerBlock(&dyn_smem, my_kernel, num_blocks, block_size);
```

---

## Library API

ROCm 7.2 added APIs for loading and querying GPU code objects at runtime (alternative to `hipModuleLoad`):

```cpp
// Load a code object from file or memory
hipLibrary_t lib;
hipLibraryLoadFromFile(&lib, "my_kernels.co");
// or: hipLibraryLoadData(&lib, code_ptr);

// Query kernel count and enumerate
int count;
hipLibraryGetKernelCount(&count, lib);

hipKernel_t* kernels = new hipKernel_t[count];
hipLibraryEnumerateKernels(kernels, count, lib);

// Get a specific kernel by name
hipKernel_t kernel;
hipLibraryGetKernel(&kernel, lib, "my_kernel");

// Query kernel properties
const char* name;
hipKernelGetName(&name, kernel);

hipLibrary_t parent;
hipKernelGetLibrary(&parent, kernel);

// Unload
hipLibraryUnload(lib);
```

---

## Function Qualifiers and Built-ins

### Qualifiers
| Qualifier | Compiled for | Called from | Notes |
|-----------|-------------|-------------|-------|
| `__global__` | device | host | kernel entry point, returns void |
| `__device__` | device | device | inlined by default |
| `__host__` | host | host | default (implicit) |
| `__host__ __device__` | both | both | no threadIdx/blockIdx access |

### Synchronization
```cpp
__syncthreads();         // barrier for all threads in a block
__syncwarp();            // barrier for threads in a warp (AMD: wavefront)
__threadfence();         // memory fence — all threads in device
__threadfence_block();   // memory fence — threads in same block
__threadfence_system();  // memory fence — all threads + host
```

### Warp/Wavefront Intrinsics
```cpp
// AMD: warpSize = 64 (wavefront64); NVIDIA: warpSize = 32
// Portable: use warpSize built-in

// Shuffle (exchange values within warp)
// AMD:
float val = __shfl(src, lane_id);
float val = __shfl_up(src, delta);
float val = __shfl_down(src, delta);
float val = __shfl_xor(src, mask);

// NVIDIA (CUDA 9+, synced variants):
float val = __shfl_sync(mask, src, lane_id);
```

### Atomic Operations
```cpp
atomicAdd(ptr, val);    // int, unsigned int, float, double, long long
atomicSub(ptr, val);
atomicMin(ptr, val);
atomicMax(ptr, val);
atomicExch(ptr, val);   // exchange
atomicCAS(ptr, compare, val);  // compare-and-swap
atomicAnd(ptr, val);
atomicOr(ptr, val);
atomicXor(ptr, val);
```
