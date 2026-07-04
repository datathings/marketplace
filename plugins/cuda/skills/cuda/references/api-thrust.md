# Thrust API

## Table of Contents
1. [Containers](#containers)
2. [Execution Policies](#execution-policies)
3. [Sorting](#sorting)
4. [Reductions](#reductions)
5. [Transformations](#transformations)
6. [Prefix Scans](#prefix-scans)
7. [Searching and Set Operations](#searching-and-set-operations)
8. [Fancy Iterators](#fancy-iterators)
9. [Tuple Types (cuda::std)](#tuple-types-cudastd)
10. [Interoperability with Raw CUDA Pointers](#interoperability-with-raw-cuda-pointers)
11. [CUB Device Algorithms (CCCL 3.3)](#cub-device-algorithms-cccl-33)
12. [libcu++ Random and mdspan (CCCL 3.3)](#libcu-random-and-mdspan-cccl-33)

---

## Containers

Thrust provides STL-like containers that manage GPU memory automatically.

### `thrust::device_vector<T>`
**Description:** Contiguous array in GPU global memory. Automatically allocates/frees.
**Example:**
```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

thrust::host_vector<float>   h_vec(N, 1.0f);      // host array
thrust::device_vector<float> d_vec = h_vec;        // H2D copy
// ... GPU operations on d_vec ...
thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin()); // D2H
```

### `thrust::host_vector<T>`
**Description:** Contiguous host-side vector. Seamlessly copies to/from `device_vector`.

### Indexing and iteration:
```cpp
d_vec[0] = 42.0f;                             // single element (slow, triggers D2H)
float *raw = thrust::raw_pointer_cast(d_vec.data()); // get raw device pointer
```

---

## Execution Policies

Control where algorithms execute.

```cpp
#include <thrust/execution_policy.h>

thrust::sort(thrust::device, d_vec.begin(), d_vec.end()); // GPU (default for device_vector)
thrust::sort(thrust::host,   h_vec.begin(), h_vec.end()); // CPU
```

**CUDA stream policy:**
```cpp
#include <thrust/system/cuda/execution_policy.h>
thrust::sort(thrust::cuda::par.on(my_stream), d_vec.begin(), d_vec.end());
```

---

## Sorting

### `thrust::sort(first, last [, comp])`
**Description:** In-place sort of a range. Default ascending; provide comparator for custom order.
**Example:**
```cpp
#include <thrust/sort.h>
thrust::sort(d_vec.begin(), d_vec.end());                          // ascending
thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<float>()); // descending
```

### `thrust::sort_by_key(keys_first, keys_last, values_first [, comp])`
**Description:** Sorts `keys` and permutes `values` correspondingly.
**Example:**
```cpp
thrust::device_vector<int>   keys(N);
thrust::device_vector<float> vals(N);
// ... fill ...
thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
```

### `thrust::stable_sort` / `thrust::stable_sort_by_key`
**Description:** Preserves relative order of equal elements.

### `thrust::is_sorted(first, last [, comp]) -> bool`
**Description:** Returns true if the range is sorted.

---

## Reductions

### `thrust::reduce(first, last [, init] [, binary_op]) -> T`
**Description:** Reduces a range to a single value.
**Example:**
```cpp
#include <thrust/reduce.h>
float sum  = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
float maxv = thrust::reduce(d_vec.begin(), d_vec.end(),
                            -FLT_MAX, thrust::maximum<float>());
```

### `thrust::count(first, last, value) -> ptrdiff_t`
**Description:** Counts occurrences of `value`.

### `thrust::count_if(first, last, pred) -> ptrdiff_t`
**Description:** Counts elements satisfying predicate.

### `thrust::min_element` / `thrust::max_element`
**Description:** Returns iterator to min/max element.

### `thrust::inner_product(first1, last1, first2, init [, binary_op1] [, binary_op2]) -> T`
**Description:** Inner product of two sequences.

---

## Transformations

### `thrust::transform(first, last, result, op)` — Unary transform
**Description:** Applies `op` to each element and writes to `result`.
**Example:**
```cpp
#include <thrust/transform.h>
thrust::transform(d_in.begin(), d_in.end(), d_out.begin(),
                  [] __device__ (float x) { return x * x; });  // square each element
```

### `thrust::transform(first1, last1, first2, result, binary_op)` — Binary transform
**Example:**
```cpp
// element-wise add
thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                  thrust::plus<float>());
```

### `thrust::fill(first, last, value)`
**Description:** Sets all elements to `value`.

### `thrust::sequence(first, last [, init [, step]])`
**Description:** Fills with 0, 1, 2, ... (or with `init` and `step`).

### `thrust::replace_if(first, last, pred, new_val)`
**Description:** Replaces elements satisfying predicate.

### `thrust::generate(first, last, gen)`
**Description:** Fills using a generator function (e.g., `thrust::default_random_engine`).
**Example:**
```cpp
#include <thrust/generate.h>
#include <thrust/random.h>
thrust::default_random_engine rng(42);
thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
thrust::generate(h_vec.begin(), h_vec.end(), [&]() { return dist(rng); });
```

---

## Prefix Scans

### `thrust::inclusive_scan(first, last, result [, binary_op])`
**Description:** Inclusive prefix scan (e.g., cumulative sum where result[i] includes input[i]).
**Example:**
```cpp
#include <thrust/scan.h>
// [1, 2, 3, 4] -> [1, 3, 6, 10]
thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin(), thrust::plus<int>());
```

### `thrust::exclusive_scan(first, last, result [, init [, binary_op]])`
**Description:** Exclusive scan (result[i] does NOT include input[i]).
**Example:**
```cpp
// [1, 2, 3, 4] with init=0 -> [0, 1, 3, 6]
thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin(), 0);
```

### `thrust::transform_inclusive_scan` / `thrust::transform_exclusive_scan`
**Description:** Apply unary transform then scan in a single pass.

---

## Searching and Set Operations

### `thrust::find(first, last, value) -> iterator`
**Description:** Returns iterator to first occurrence of `value`.

### `thrust::partition(first, last, pred)` / `thrust::stable_partition`
**Description:** Reorders elements so those satisfying `pred` come first.

### `thrust::unique(first, last [, binary_pred]) -> iterator`
**Description:** Removes consecutive duplicates; returns iterator to new end.

### `thrust::set_intersection`, `thrust::set_union`, `thrust::set_difference`
**Description:** Set operations on sorted ranges.

---

## Fancy Iterators

Thrust iterators compose to express complex access patterns without temporary allocations.

```cpp
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

// counting_iterator: generates 0, 1, 2, ...
thrust::counting_iterator<int> counter(0);
thrust::reduce(counter, counter + N, 0, thrust::plus<int>());  // = N*(N-1)/2

// constant_iterator: infinite stream of a single value
thrust::constant_iterator<float> one(1.0f);

// transform_iterator: apply functor on-the-fly during iteration
auto squared = thrust::make_transform_iterator(d_vec.begin(),
    [] __device__ (float x) { return x * x; });
float sum_of_squares = thrust::reduce(squared, squared + N, 0.0f);

// zip_iterator: iterate multiple ranges in lock-step (like Python zip)
// CUDA 13.2: use variadic arguments directly (no make_tuple needed)
auto begin = thrust::make_zip_iterator(d_keys.begin(), d_vals.begin());
auto end   = thrust::make_zip_iterator(d_keys.end(),   d_vals.end());
```

---

## Tuple Types (cuda::std)

As of CUDA 13.2, `thrust::tuple`, `thrust::make_tuple`, and `thrust::get` are replaced by their `cuda::std` equivalents. The `thrust::make_zip_iterator` now accepts variadic arguments directly instead of requiring `make_tuple`.

**Updated usage:**
```cpp
#include <cuda/std/tuple>

// Creating tuples
cuda::std::tuple<float, float> t = cuda::std::make_tuple(1.0f, 2.0f);

// Accessing elements
float x = cuda::std::get<0>(t);
float y = cuda::std::get<1>(t);

// Return tuples from device functions
__host__ __device__ __forceinline__ cuda::std::tuple<float, float> compute() {
    return cuda::std::make_tuple(3.14f, 2.71f);
}

// zip_iterator with variadic arguments (no make_tuple wrapper needed)
thrust::for_each(
    thrust::make_zip_iterator(d_pos, d_vel),
    thrust::make_zip_iterator(d_pos + N, d_vel + N),
    my_functor());

// Accessing zip_iterator elements in device code
struct my_functor {
    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        float4 pos = cuda::std::get<0>(t);
        float4 vel = cuda::std::get<1>(t);
        // ...
        cuda::std::get<0>(t) = make_float4(new_pos, 0.0f);
    }
};

// Use with thrust algorithms (e.g., minimum over tuples)
thrust::reduce(thrust::make_zip_iterator(d_weights, d_edges),
               thrust::make_zip_iterator(d_weights + N, d_edges + N),
               thrust::minimum<cuda::std::tuple<float, uint>>());
```

---

## Interoperability with Raw CUDA Pointers

```cpp
// Wrap raw device pointer as Thrust iterator
float *raw_ptr;
cudaMalloc(&raw_ptr, N * sizeof(float));
thrust::device_ptr<float> thrust_ptr(raw_ptr);
thrust::fill(thrust_ptr, thrust_ptr + N, 0.0f);

// Get raw pointer back from device_vector
float *raw = thrust::raw_pointer_cast(d_vec.data());
myCustomKernel<<<grid, block>>>(raw, N);
```

**Header:** `#include <thrust/*.h>` — included with CUDA Toolkit, no separate linking.

---

## CUB Device Algorithms (CCCL 3.3)

CCCL 3.3 (CUDA Toolkit 13.3) adds three device-wide CUB algorithm families. `DeviceFind` and
`DeviceSegmentedScan` follow CUB's **two-call temp-storage** convention (first call with
`d_temp_storage = nullptr` reports `temp_storage_bytes`; second call does the work). `DeviceTransform`
allocates no temp storage and is called once. Iterators are typically `thrust::device_vector<T>::begin()`
or raw device pointers.

### `cub::DeviceFind::FindIf(void *d_temp, size_t &temp_bytes, InputIt d_in, OutputIt d_out, Predicate op, NumItemsT num_items) -> cudaError_t`
**Header:** `#include <cub/device/device_find.cuh>`. Writes the **index** of the first element
satisfying `op` into `d_out[0]` (== `num_items` if none). `op` is a `__host__ __device__` functor.
```cpp
cub::DeviceFind::FindIf(nullptr, temp_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
thrust::device_vector<char> temp(temp_bytes);
cub::DeviceFind::FindIf(thrust::raw_pointer_cast(temp.data()), temp_bytes,
                        d_in.begin(), d_out.begin(), predicate, num_items);
```

### `cub::DeviceFind::LowerBound(void *d_temp, size_t &temp_bytes, HaystackIt d_range, OffsetT num_range, NeedlesIt d_values, OffsetT num_values, OutputIt d_out, CompareOp compare_op) -> cudaError_t`
### `cub::DeviceFind::UpperBound(...)` — identical signature
**Description:** Parallel binary search of every value in `d_values` against the sorted range `d_range`;
writes insertion indices to `d_out`. `compare_op` is usually `cuda::std::less{}` (`<cuda/std/functional>`).
`LowerBound`/`UpperBound` differ only for values present in the range (lower = first, upper = past-last).
```cpp
cub::DeviceFind::LowerBound(nullptr, temp_bytes, d_range.begin(), (int)d_range.size(),
                            d_values.begin(), (int)d_values.size(), d_out.begin(), cuda::std::less{});
```

### `cub::DeviceSegmentedScan::ExclusiveSegmentedSum(void *d_temp, size_t &temp_bytes, InputIt d_in, OutputIt d_out, BeginOffsetIt d_begin_offsets, EndOffsetIt d_end_offsets, NumSegmentsT num_segments) -> cudaError_t`
### `cub::DeviceSegmentedScan::InclusiveSegmentedScan(void *d_temp, size_t &temp_bytes, InputIt d_in, OutputIt d_out, BeginOffsetIt d_begin_offsets, EndOffsetIt d_end_offsets, NumSegmentsT num_segments, ScanOp scan_op) -> cudaError_t`
**Header:** `#include <cub/device/device_segmented_scan.cuh>`. Runs an independent scan per segment.
Segment `s` spans `[d_begin_offsets[s], d_end_offsets[s])`; the common idiom uses one offsets array
with `begin = offsets.begin()` and `end = offsets.begin() + 1` (num_segments = offsets.size() - 1).
`ExclusiveSegmentedSum` is a fixed `+` exclusive scan; `InclusiveSegmentedScan` takes a custom
associative `scan_op` (e.g. `cuda::maximum<>{}` from `<cuda/functional>` for a running max).
```cpp
auto begin_offsets = d_offsets.begin();
auto end_offsets   = d_offsets.begin() + 1;
cub::DeviceSegmentedScan::ExclusiveSegmentedSum(nullptr, temp_bytes, d_in.begin(), d_out.begin(),
                                                begin_offsets, end_offsets, num_segments);
```

### `cub::DeviceTransform::Transform(InputTuple d_in, OutputIt d_out, NumItemsT num_items, TransformOp op) -> cudaError_t`  (N inputs → 1 output)
### `cub::DeviceTransform::Transform(InputTuple d_in, OutputTuple d_out, NumItemsT num_items, TransformOp op) -> cudaError_t`  (N inputs → M outputs)
**Header:** `#include <cub/device/device_transform.cuh>`. No temp storage — one call. Inputs (and, in
the N→M form, outputs) are passed as a `cuda::std::tuple` of iterators; any iterator type works,
including `cuda::counting_iterator<int>{100}` (`<cuda/iterator>`). For N→M, `op` returns a
`cuda::std::tuple` of M values.
```cpp
// N=3 -> 1 : result[i] = (a[i] + b[i]) * c[i]
auto op1 = [] __host__ __device__(int x, float y, int z) -> int { return int((x + y) * z); };
cub::DeviceTransform::Transform(cuda::std::tuple{a.begin(), b.begin(), counting},
                                result.begin(), a.size(), op1);
// N=2 -> M=2 : (sum, diff) in one pass
auto op2 = [] __host__ __device__(int x, int y) -> cuda::std::tuple<int,int> { return {x+y, x-y}; };
cub::DeviceTransform::Transform(cuda::std::tuple{a.begin(), b.begin()},
                                cuda::std::tuple{sum.begin(), diff.begin()}, a.size(), op2);
```

---

## libcu++ Random and mdspan (CCCL 3.3)

**`<cuda/std/random>`** provides host- and device-compatible standard C++ distributions and, new in
CCCL 3.3, the C++26 counter-based Philox engines. **`<cuda/random>`** adds `cuda::pcg64` (the
NumPy-default generator) as an NVIDIA extension. Engines and distributions are used inside kernels
exactly like `<random>`: `dist(engine)`.
```cpp
#include <cuda/random>        // cuda::pcg64
#include <cuda/std/random>    // engines + distributions
__global__ void k(unsigned long long seed, float *u, float *g, int *p, int *b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pcg64 rng(seed + tid);
    cuda::std::philox4x32 philox((cuda::std::uint32_t)(seed + tid));
    cuda::std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    cuda::std::normal_distribution<float>       nrm(0.0f, 1.0f);
    cuda::std::poisson_distribution<int>        poi(4.0);
    cuda::std::bernoulli_distribution           ber(0.25);
    u[tid] = uni(rng); g[tid] = nrm(rng); p[tid] = poi(rng); b[tid] = ber(philox);
}
```

**`<cuda/std/mdspan>`** provides `cuda::std::mdspan` / `cuda::std::dextents<IndexT, Rank>`
multi-dimensional views. **`<cuda/mdspan>`** adds device-side extensions: `cuda::device_mdspan`,
`cuda::to_device_mdspan<T, Rank>(DLTensor)` and `cuda::to_dlpack_tensor(...)` for DLPack interchange
(PyTorch/JAX/CuPy), and `cuda::shared_memory_mdspan` for shared-memory tiles with address-space-checked
load/store accessors. Sample: `cuda-samples/cpp/4_CUDA_Libraries/libcuxxMdspan/`.
