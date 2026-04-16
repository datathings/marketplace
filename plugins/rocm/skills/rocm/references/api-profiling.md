# ROCm Profiling and Debugging Reference

## Table of Contents
1. [HIP Event Timing](#hip-event-timing)
2. [rocProfiler-SDK — Programmatic Profiling](#rocprofiler-sdk--programmatic-profiling)
3. [rocprof CLI Tool](#rocprof-cli-tool)
4. [ROCm System Monitor (rocm-smi)](#rocm-system-monitor-rocm-smi)
5. [rocGDB — GPU Debugger](#rocgdb--gpu-debugger)
6. [ASAN on GPU](#asan-on-gpu)
7. [Compilation Flags for Profiling](#compilation-flags-for-profiling)
8. [Performance Tuning Tips](#performance-tuning-tips)

---

## HIP Event Timing

The simplest way to measure kernel execution time (no external tools needed):

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

// Record start in stream (nullptr = default stream)
hipEventRecord(start, nullptr);

// Launch workload
my_kernel<<<grid, block>>>(args);

// Record stop
hipEventRecord(stop, nullptr);

// Wait for stop event to complete
hipEventSynchronize(stop);

// Get elapsed time in milliseconds
float ms = 0.f;
hipEventElapsedTime(&ms, start, stop);
printf("Kernel: %.3f ms (%.2f GB/s)\n", ms,
       (double)bytes_moved / (ms * 1e-3) / 1e9);

hipEventDestroy(start);
hipEventDestroy(stop);
```

**Notes:**
- `hipEventElapsedTime` measures GPU-side time only (excludes host overhead).
- For library calls (rocBLAS, etc.), wrap the library call between events.
- For multiple kernels, record one stop event after the last kernel.
- `hipStreamSynchronize` vs `hipEventSynchronize`: the latter is more granular.

---

## rocProfiler-SDK — Programmatic Profiling

**Header:** `#include <rocprofiler-sdk/rocprofiler.h>`
**Link:** `-lrocprofiler-sdk`

rocProfiler-SDK (ROCm 6+) provides a stable C API for in-process profiling. Applications load a shared client library at runtime.

### Architecture
```
Application → rocprofiler-sdk → GPU driver
                    ↕
              Client library (your .so)
              - Registers callbacks
              - Processes profiling data
```

### API Tracing (buffered callback)
```cpp
// In your client .cpp compiled to a shared library:
#include <rocprofiler-sdk/rocprofiler.h>

// Called when a HIP API call completes (buffered mode)
void api_tracing_callback(rocprofiler_context_id_t      context,
                           rocprofiler_buffer_id_t       buffer,
                           rocprofiler_record_header_t** headers,
                           size_t                        num_headers,
                           void*                         data,
                           uint64_t                      drop_count)
{
    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];
        if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
            header->kind == ROCPROFILER_BUFFER_TRACING_KIND_HIP_RUNTIME_API) {
            auto* rec = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(
                header->payload);
            printf("HIP API: %s  start=%lu ns  end=%lu ns\n",
                   rocprofiler_get_hip_runtime_api_operation_name(rec->operation),
                   rec->start_timestamp,
                   rec->end_timestamp);
        }
    }
}

// Called when client library is loaded
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version,
                      uint32_t prio, rocprofiler_client_id_t* id)
{
    // Create a context
    rocprofiler_context_id_t ctx;
    rocprofiler_create_context(&ctx);

    // Create a buffer for async delivery
    rocprofiler_buffer_id_t buf;
    rocprofiler_create_buffer(ctx, 4096, 2048,
                              ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                              api_tracing_callback, nullptr, &buf);

    // Subscribe to HIP runtime API tracing
    rocprofiler_configure_buffer_tracing_service(
        ctx, ROCPROFILER_BUFFER_TRACING_KIND_HIP_RUNTIME_API,
        nullptr, 0, buf);

    rocprofiler_start_context(ctx);
    return nullptr;
}
```

### Counter Collection (Hardware Performance Counters)
```cpp
// In rocprofiler_configure():

// Create a profile config with desired counters
rocprofiler_profile_counting_dispatch_profile_config_t profile_cfg;
// ... setup counter names (e.g. "SQ_WAVES", "TA_BUSY", "FETCH_SIZE") ...

rocprofiler_configure_buffered_dispatch_profile_counting_service(
    ctx, buf, profile_cfg);
```

### PC Sampling (Instruction-Level Profiling)
```cpp
// PC sampling collects program counter samples at regular intervals
// to identify hot instructions without instrumenting the code.

// In tool_init — find agents that support PC sampling:
rocprofiler_query_pc_sampling_agent_configurations(
    agent_id, config_callback, max_configs, user_data);

// Configure and enable PC sampling on an agent:
rocprofiler_configure_pc_sampling_service(
    ctx, agent_id,
    ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,  // or HOST_TRAP
    ROCPROFILER_PC_SAMPLING_UNIT_CYCLES,
    interval,  // e.g., 1048576 cycles for stochastic
    buf);

// Samples arrive in buffer as rocprofiler_pc_sampling_record_t:
//   .pc              — program counter address
//   .exec_mask       — active lanes bitmask
//   .dispatch_id     — identifies the kernel dispatch
//   .correlation_id  — links to API tracing records
```

### Thread Trace (Wavefront-Level Instruction Tracing)
```cpp
// Thread trace captures per-instruction execution data including latency.
// Uses experimental API: <rocprofiler-sdk/experimental/thread_trace.h>

rocprofiler_configure_thread_trace_service(
    ctx, agent_id,
    TARGET_CU,     // CU (gfx9) or WGP (gfx10+) to trace
    SHADER_MASK,   // shader engine mask
    BUFFER_SIZE,   // trace buffer size (e.g., 256MB)
    buf);

// Decode results with:
//   rocprofiler_thread_trace_decoder_pc_t  — PC + instruction info
//   Latency tables mapping address → cycle count
//   Wave lifetime statistics
```

### Running with the Client Library
```bash
# Build client as a shared library
hipcc -shared -fPIC client.cpp -o libmy_profiler.so -lrocprofiler-sdk

# Run application with client loaded
ROCPROFILER_PRELOAD=./libmy_profiler.so ./my_app
```

---

## rocprof CLI Tool

`rocprof` is the command-line profiler — no code changes needed.

### Basic Usage
```bash
# Collect kernel timing
rocprof ./my_app

# Specify output file
rocprof --output-file profile.csv ./my_app

# Collect hardware counters defined in input.txt
rocprof --input input.txt ./my_app
```

### Input File Format (`input.txt`)
```
# input.txt - counter list
pmc: SQ_WAVES GRBM_COUNT TA_BUSY_avr
pmc: FETCH_SIZE WRITE_SIZE
range: 0:10    # only profile kernel launches 0 through 10
kernel: my_kernel_name   # only profile kernels matching this name
```

### Common Hardware Counters
| Counter | Description |
|---------|-------------|
| `SQ_WAVES` | Number of wavefronts executed |
| `GRBM_COUNT` | GPU busy cycles |
| `TA_BUSY_avr` | Texture addresser busy (L1 cache) |
| `TCP_TCC_WRITE_REQ_sum` | L1→L2 write requests |
| `TCC_HIT_sum` | L2 cache hits |
| `TCC_MISS_sum` | L2 cache misses |
| `FETCH_SIZE` | Total data fetched (bytes) |
| `WRITE_SIZE` | Total data written (bytes) |
| `SQ_BUSY_CYCLES` | Shader engine busy cycles |

### Output
`rocprof` produces a CSV with columns: `Kernel_Name`, `grd`, `wgr`, `lds`, `scr`, `arch_vgpr`, `accum_vgpr`, `sgpr`, `wave_size`, `sig`, plus any counters requested.

```bash
# View summary
cat results.csv

# Timeline trace (Chrome tracing format)
rocprof --sys-trace ./my_app
# Open results.json in chrome://tracing or Perfetto
```

### ROCm Compute Profiler (ROCm 7.2+)
```bash
# List available IP blocks for an architecture
rocprof --list-blocks gfx942

# Block aliases for shorter counter specs
rocprof -b SQ ./my_app   # -b/--block accepts block aliases

# CU Utilization metric (percentage of CUs used during execution)
# Available in analysis report alongside per-kernel metrics
```
**Note:** The `database` mode (Grafana/MongoDB) has been removed in ROCm 7.2. Use the analysis DB-based visualizer or Textual UI instead.

---

## ROCm System Monitor (rocm-smi)

```bash
# Show GPU summary (temperature, power, utilization, memory)
rocm-smi

# Continuous monitoring (refresh every 1s)
rocm-smi -d 0 --showtemp --showpower --showuse --showmemuse -w 1000

# Show all GPU info
rocm-smi --showall

# Specific fields
rocm-smi --showtemp         # temperatures
rocm-smi --showpower        # power consumption
rocm-smi --showuse          # GPU utilization %
rocm-smi --showmemuse       # VRAM utilization %
rocm-smi --showmeminfo vram # memory total/used/free
rocm-smi --showclocks       # clock frequencies
rocm-smi --showproductname  # GPU model
rocm-smi --showserial       # serial number
rocm-smi --showid           # GPU IDs

# JSON output for scripting
rocm-smi --json

# Set performance level (auto / low / high / manual)
rocm-smi --setperflevel high -d 0

# Set clock frequencies (manual mode required)
rocm-smi --setsclk 7 -d 0   # set to SCLK level 7

# Reset to defaults
rocm-smi --resetfans --resetprofile
```

---

## rocGDB — GPU Debugger

rocGDB is an extension of GDB for debugging GPU kernels on AMD GPUs.

### Build for Debugging
```bash
# Compile with debug info (-g) and no optimization (-O0)
# Also set GPU architecture target
hipcc -g -O0 -o my_app my_app.cpp

# Or with CMake:
# CMAKE_BUILD_TYPE=Debug
```

### Basic Session
```bash
# Launch rocGDB
rocgdb ./my_app

# Common GDB commands (all work normally)
(gdb) break my_kernel       # breakpoint in kernel
(gdb) run
(gdb) info threads          # show all host + GPU threads
(gdb) thread 2              # switch to thread 2 (GPU wave)
(gdb) bt                    # backtrace
(gdb) print threadIdx.x     # print HIP built-in
(gdb) print d_array[0]      # print device memory value
(gdb) continue
(gdb) quit
```

### GPU-Specific Commands
```bash
# List all GPU wavefronts
(gdb) info wavefronts

# Select a specific wavefront
(gdb) wavefront 5

# Print work-item coordinates
(gdb) print threadIdx
(gdb) print blockIdx

# Step through device code
(gdb) step
(gdb) next

# Watchpoints work on device memory
(gdb) watch d_array[42]
```

### Environment Setup
```bash
# Enable GPU debugging (must be set before launching application)
export HSA_TOOLS_LIB=librocm-debug-agent.so  # optional debug agent
```

---

## ASAN on GPU

ROCm supports Address Sanitizer for detecting GPU memory errors.

```bash
# Build with GPU ASAN
hipcc -fsanitize=address -shared-libasan my_app.cpp -o my_app

# Run (ASAN output goes to stderr)
./my_app

# Typical errors detected:
# - Out-of-bounds global memory access
# - Out-of-bounds shared memory access
# - Use after free
```

```bash
# LLVM_ASAN with ROCm example:
# See rocm-examples/LLVM_ASAN/ for sample programs
hipcc -fsanitize=address -g -O1 my_app.cpp -o my_app
./my_app 2>&1 | head -50
```

---

## Compilation Flags for Profiling

**Note:** As of ROCm 7.2, `hipcc` is deprecated and now invokes AMD Clang. It is recommended to invoke `amdclang++` directly. Both compilers accept the same flags.

```bash
# Profile build (optimize but keep debug info for profiler)
amdclang++ -O2 -g -fno-omit-frame-pointer -x hip my_app.cpp -o my_app
# OR (still works but deprecated):
hipcc -O2 -g -fno-omit-frame-pointer my_app.cpp -o my_app

# Specify target GPU architecture (important for performance)
amdclang++ --offload-arch=gfx1100 -x hip my_app.cpp  # RX 7900 / Navi31
amdclang++ --offload-arch=gfx90a  -x hip my_app.cpp  # MI200 series
amdclang++ --offload-arch=gfx942  -x hip my_app.cpp  # MI300 series
amdclang++ --offload-arch=gfx1030 -x hip my_app.cpp  # RX 6800 / Navi21

# Multiple targets (FAT binary)
amdclang++ --offload-arch=gfx90a --offload-arch=gfx942 -x hip my_app.cpp

# Check available targets
/opt/rocm/bin/rocm_agent_enumerator

# Enable verbose compilation
amdclang++ -v -x hip my_app.cpp

# Dump assembly (GCN ISA)
amdclang++ --save-temps -x hip my_app.cpp   # saves .s files with ISA

# Enable fast math (relaxed IEEE, may affect precision)
amdclang++ -ffast-math -x hip my_app.cpp

# Enable ThinLTO for link-time optimization (ROCm 7.2+)
amdclang++ -foffload-lto=thin -x hip my_app.cpp
```

### CMake Integration
```cmake
cmake_minimum_required(VERSION 3.21)
project(MyROCmApp LANGUAGES CXX)

# Find HIP package
find_package(hip REQUIRED)

add_executable(my_app main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(my_app PRIVATE hip::host)

# Set GPU targets (can also be passed at cmake configure time)
set_property(TARGET my_app PROPERTY HIP_ARCHITECTURES gfx90a gfx1100)
```

---

## Performance Tuning Tips

### Memory Bandwidth
- Coalesced access: threads in a warp/wavefront should access consecutive memory addresses.
- AMD wavefront = 64 threads; optimal load/store = 64 consecutive 4-byte elements = 256 bytes.
- Use `hipMemcpyAsync` with pinned host memory to overlap transfers with compute.
- Avoid strided global memory access; use shared memory to transpose data before use.

### Occupancy
```cpp
// Use the occupancy API to find the best block size automatically:
int min_grid_size = 0, best_block_size = 0;
hipOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size, my_kernel, 0, 0);
// Then launch with best_block_size
```

### Shared Memory
```cpp
// Prefer __shared__ arrays for data reused within a block
// Avoid bank conflicts: on AMD, shared memory has 32 or 64 banks of 4 bytes
// Padding can eliminate conflicts:
__shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 pad avoids conflicts
```

### Register Usage
```cpp
// Limit register pressure with launch bounds
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_CU)
__global__ void my_kernel(...) { ... }
// This hints the compiler to target a specific occupancy level
```

### Warp/Wavefront Efficiency
- Minimize divergent branches within a warp (both paths execute serially).
- Use warp shuffle (`__shfl_*`) for intra-warp reductions instead of shared memory.
- AMD: warpSize = 64; NVIDIA: warpSize = 32. Use `warpSize` built-in for portability.

### Cache Hints
```cpp
// Hint to use L1 cache (streaming / no cache for write-once data)
// AMD uses __builtin_nontemporal_store for streaming writes
__builtin_nontemporal_store(val, ptr);   // AMD extension
```

### Benchmarking Checklist
1. Warm-up runs (first launch may include JIT compilation latency)
2. Measure multiple iterations and report median
3. Account for PCIe transfer time separately from kernel time
4. Use `rocm-smi` to verify GPU is not thermally throttled
5. Set GPU to fixed performance level: `rocm-smi --setperflevel high -d 0`
6. Check achieved memory bandwidth against theoretical peak (via `FETCH_SIZE` + `WRITE_SIZE` counters)
