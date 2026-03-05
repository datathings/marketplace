# Cooperative Groups API

## Table of Contents
1. [Overview and Include](#overview-and-include)
2. [Thread Block Groups](#thread-block-groups)
3. [Tiled Partitions](#tiled-partitions)
4. [Coalesced Groups](#coalesced-groups)
5. [Grid Groups](#grid-groups)
6. [Synchronization](#synchronization)
7. [Collective Operations](#collective-operations)
8. [Reduction and Scan](#reduction-and-scan)
9. [Binary Partition](#binary-partition)

---

## Overview and Include

Cooperative Groups (CG) provide a flexible, composable model for thread synchronization beyond the `__syncthreads()` barrier. Groups can be created, passed as parameters, and used to express algorithmic structure.

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>    // cg::reduce
#include <cooperative_groups/memcpy_async.h>  // async shared memory loads

using namespace cooperative_groups;  // common convention
// or: namespace cg = cooperative_groups;
```

**Requirements:**
- Thread block CG: any compute capability
- Grid group (`this_grid()`): requires `cudaLaunchCooperativeKernel` (cc 6.0+)
- Multi-grid: requires NVLink (cc 7.0+)

---

## Thread Block Groups

### `this_thread_block() -> thread_block`
**Description:** Returns a group representing all threads in the current thread block. Equivalent to block-level `__syncthreads()` but composable.

**Key properties:**
- `.size()` — total threads in block (`blockDim.x * blockDim.y * blockDim.z`)
- `.thread_rank()` — linear index of this thread within the group
- `.group_index()` — block index in the grid (`blockIdx`)
- `.thread_index()` — thread index in the block (`threadIdx`)
- `.dim_threads()` — block dimensions (`blockDim`)

**Example:**
```cpp
__global__ void kernel(int *workspace, int N) {
    thread_block tb = this_thread_block();
    extern __shared__ int smem[];

    int rank = tb.thread_rank();

    // ... work ...

    tb.sync();   // equivalent to __syncthreads()

    if (rank == 0) printf("Block has %u threads\n", tb.size());
}
```

### `thread_group` (base type)
**Description:** Base type for all CG groups. `thread_block`, `coalesced_group`, and `grid_group` inherit from it. Generic algorithms can accept `thread_group` parameters:

```cpp
__device__ int reduction(thread_group g, int *smem, int val) {
    int lane = g.thread_rank();
    for (int i = g.size() / 2; i > 0; i /= 2) {
        smem[lane] = val;
        g.sync();
        if (lane < i) val += smem[lane + i];
        g.sync();
    }
    return (lane == 0) ? val : -1;
}
```

---

## Tiled Partitions

Divide a group into equal-size sub-groups. Tile size must be a power of 2 and ≤ 32 (warp size) for warp-level intrinsics.

### `tiled_partition<N>(parent_group) -> thread_block_tile<N>`
**Description:** Compile-time tile size. Enables warp shuffle primitives when N ≤ 32.

```cpp
thread_block tb = this_thread_block();
thread_block_tile<32> warp = tiled_partition<32>(tb);  // one warp
thread_block_tile<16> half = tiled_partition<16>(tb);  // half-warp tiles

// Each tile acts independently
int lane = warp.thread_rank();  // 0..31
int val  = warp.shfl(data, 0);  // broadcast from lane 0 within warp
```

### `tiled_partition(parent_group, N) -> thread_group`
**Description:** Runtime tile size (not eligible for warp-level ops when N > warp size).

**Available warp-level operations on `thread_block_tile<N>` (N ≤ 32):**
- `.shfl(val, src_lane)` — broadcast a value from `src_lane`
- `.shfl_up(val, delta)` — shift up within the tile
- `.shfl_down(val, delta)` — shift down within the tile
- `.shfl_xor(val, mask)` — butterfly exchange
- `.ballot(predicate)` — bitmask of which threads have true predicate
- `.any(predicate)` / `.all(predicate)` — warp vote

---

## Coalesced Groups

Groups that represent active (non-divergent) threads within a warp.

### `coalesced_threads() -> coalesced_group`
**Description:** Returns a group of all currently active threads in the warp. Handles control flow divergence.

```cpp
__global__ void kernel(int *arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && arr[idx] > 0) {
        coalesced_group active = coalesced_threads();
        // All threads passing the if are grouped together
        int leader = active.thread_rank() == 0;
        if (leader) atomicAdd(&counter, active.size());
    }
}
```

### `labeled_partition(group, label) -> coalesced_group`
**Description:** Groups threads by a label value. Threads with the same label form a group.

---

## Grid Groups

Requires cooperative kernel launch; allows synchronization across the entire grid.

### `this_grid() -> grid_group`
**Description:** Returns a group spanning all threads in the grid. `.sync()` is a global barrier.

**Properties:**
- `.size()` — total threads in the grid
- `.thread_rank()` — unique linear index across all blocks

**Launch:**
```cpp
void *args[] = {(void *)&d_data, (void *)&N};
cudaLaunchCooperativeKernel((void *)myKernel, gridDim, blockDim, args, sharedMem, stream);
```

**Kernel:**
```cpp
__global__ void myKernel(float *data, int N) {
    grid_group grid = this_grid();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Phase 1: all threads work
    for (int i = grid.thread_rank(); i < N; i += grid.size()) {
        data[i] *= 2.0f;
    }

    grid.sync();  // all blocks rendezvous here

    // Phase 2: use results from phase 1
    if (grid.thread_rank() == 0) printf("Grid sync complete\n");
}
```

---

## Synchronization

### `.sync()`
**Description:** Synchronizes all threads in the group. Equivalent to `__syncthreads()` for `thread_block`, barrier across all blocks for `grid_group`.

### `synchronize(group)` (free function)
**Description:** Same as `group.sync()`.

### `wait(group)` — for async memcpy operations
**Description:** Waits for `memcpy_async` operations initiated by the group to complete.

---

## Collective Operations

### `cg::reduce(group, val, op) -> T`
**Description:** Reduces `val` across all threads in `group` using `op`. Returns result to all threads.
**Requires:** `#include <cooperative_groups/reduce.h>`. Works on warp-sized tiles without shared memory.

```cpp
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void sumKernel(float *in, float *out, int N) {
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.0f;

    // Warp-level reduction without shared memory
    float warpSum = cg::reduce(warp, val, cg::plus<float>());

    if (warp.thread_rank() == 0) atomicAdd(out, warpSum);
}
```

**Supported ops:** `cg::plus<T>()`, `cg::less<T>()` (min), `cg::greater<T>()` (max), `cg::bit_and<T>()`, `cg::bit_or<T>()`, `cg::bit_xor<T>()`.

---

## Reduction and Scan

```cpp
// Inclusive scan within a warp tile using CG
__device__ int inclusive_scan(cg::thread_block_tile<32> tile, int val) {
    for (int offset = 1; offset < tile.size(); offset <<= 1) {
        int n = tile.shfl_up(val, offset);
        if (tile.thread_rank() >= offset) val += n;
    }
    return val;
}
```

---

## Binary Partition

### `cg::binary_partition(group, predicate) -> coalesced_group`
**Description:** Splits `group` into two sub-groups based on a boolean predicate. Threads with `predicate == true` form one group; others form another.

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void partition_kernel(int *arr, int *odd_sum, int *even_sum, int N) {
    cg::thread_block         cta   = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int val = arr[idx];
    bool is_odd = val & 1;

    // Split into odd/even groups
    auto subgroup = cg::binary_partition(tile, is_odd);
    int group_sum = cg::reduce(subgroup, val, cg::plus<int>());

    if (subgroup.thread_rank() == 0) {
        if (is_odd) atomicAdd(odd_sum,  group_sum);
        else        atomicAdd(even_sum, group_sum);
    }
}
```
