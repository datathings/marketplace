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
9. [Interoperability with Raw CUDA Pointers](#interoperability-with-raw-cuda-pointers)

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
// [1, 2, 3, 4] → [1, 3, 6, 10]
thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin(), thrust::plus<int>());
```

### `thrust::exclusive_scan(first, last, result [, init [, binary_op]])`
**Description:** Exclusive scan (result[i] does NOT include input[i]).
**Example:**
```cpp
// [1, 2, 3, 4] with init=0 → [0, 1, 3, 6]
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
auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_vals.begin()));
auto end   = thrust::make_zip_iterator(thrust::make_tuple(d_keys.end(),   d_vals.end()));
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
