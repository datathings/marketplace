# GreyCat C SDK — Collections: Array, Map, Table & Tensor

In-memory data structures: dynamic arrays, hash maps, 2D tables, and multi-dimensional tensors (incl. complex c64/c128).

_Part of the GreyCat C SDK reference (each file is linked from the skill's SKILL.md). Sibling references: api_core.md · api_memory_text.md · api_collections.md · api_runtime_storage.md · api_services.md._

## Contents

- [gc/array.h — Dynamic Arrays](#gcarray-h)
- [gc/map.h — Hash Maps](#gcmap-h)
- [gc/table.h — 2D Tables](#gctable-h)
- [gc/tensor.h — Multi-dimensional Tensors](#gctensor-h)

---

<a id="gcarray-h"></a>
## gc/array.h — Dynamic Arrays

A resizable, type-tagged array. Each element has a value (`gc_slot_t`) and a type tag (`gc_type_t`), allowing heterogeneous collections.

### Structure

```c
typedef struct {
    gc_object_t header;    // Object header
    u8_t *types;           // Per-element type tags (gc_type_t[])
    gc_slot_t *slots;      // Per-element values
    u32_t size;            // Number of elements currently in the array
    u32_t capacity;        // Allocated capacity (number of slots)
    u32_t start;           // Start offset (supports efficient dequeue from front)
} gc_array_t;
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_ARRAY_INITIAL_CAPACITY` | 8 | Default initial capacity for new arrays |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_array__init` | `void gc_array__init(gc_array_t *self, u32_t capacity, const gc_machine_t *ctx)` | Initialize the array with a given capacity (allocates storage via the call's allocator). |
| `gc_array__add_slot` | `bool gc_array__add_slot(gc_array_t *self, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Append an element to the end. Returns `false` on error. |
| `gc_array__set_slot` | `bool gc_array__set_slot(gc_array_t *self, u32_t offset, gc_slot_t value, gc_type_t type, gc_machine_t *ctx)` | Set the element at a given index. |
| `gc_array__get_slot` | `bool gc_array__get_slot(const gc_array_t *self, u32_t offset, gc_slot_t *value, gc_type_t *type)` | Get the element at a given index. Returns `false` if out of bounds. |
| `gc_array__remove_at` | `bool gc_array__remove_at(gc_array_t *self, u32_t offset, gc_slot_t *result, gc_type_t *result_type, gc_machine_t *ctx)` | Remove the element at `offset` and return it. **The result is marked; caller must unmark.** |
| `gc_array__remove_all` | `void gc_array__remove_all(gc_array_t *self, gc_machine_t *ctx)` | Remove all elements from the array. |
| `gc_array__swap` | `bool gc_array__swap(gc_array_t *self, u32_t i, u32_t j)` | Swap two elements by index. |
| `gc_array__sort` | `void gc_array__sort(gc_array_t *self, bool asc, gc_slot_tuple_u32_t field, gc_machine_t *ctx)` | Sort the array. `field` specifies a sub-field to sort by (for object arrays). |

### Usage Examples

**Initialize and append elements** — `gc_array__init` allocates storage using the call's allocator; `gc_array__add_slot` appends with a value + type tag. Pattern from building a numeric array:

```c
// 'arr' is a freshly created Array object (e.g. via gc_machine__create_object)
gc_array__init(arr, nb, ctx);                 // nb = desired capacity
for (i64_t i = 0; i < nb; i++) {
    gc_array__add_slot(arr,
        (gc_slot_t) {.i64 = some_int_value},  // value as a slot
        gc_type_int,                          // type tag must match the union member set
        ctx);
}
// when handing the array back as the call result, unmark it
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) arr}, gc_type_object);
gc_object__un_mark((gc_object_t *) arr, ctx);
```

Object elements are stored the same way, with `gc_type_object` and the pointer in `.object`:

```c
gc_array__add_slot(result, (gc_slot_t) {.object = child}, gc_type_object, ctx);
```

**Read elements with bounds checking** — `gc_array__get_slot` returns `false` when `offset >= size`. Front/back access uses index `0` and `size - 1`:

```c
gc_array_t *values = (gc_array_t *) values_slot.object;
if (values->size != 0) {
    gc_slot_t slot;
    gc_type_t slot_type;
    if (gc_array__get_slot(values, values->size - 1, &slot, &slot_type)) {
        gc_machine__set_result(ctx, slot, slot_type);  // back()
        return;
    }
}
gc_machine__set_result(ctx, (gc_slot_t) {.object = 0}, gc_type_null);
```

**Remove from the front (dequeue) — observe the mark discipline.** `gc_array__remove_at` returns the removed element *already marked*; the caller is responsible for unmarking it once it is no longer retained:

```c
gc_slot_t value = {0};
gc_type_t value_type = gc_type_null;
if (!gc_array__remove_at(values, 0, &value, &value_type, ctx)) {
    gc_machine__set_runtime_error(ctx, "index 0 is out of range");
    return;
}
gc_machine__set_result(ctx, value, value_type);
if (value_type == gc_type_object) {
    gc_object__un_mark(value.object, ctx);   // required: result was marked by remove_at
}
```

**Clear all elements** with `gc_array__remove_all` (releases element references; the array itself stays usable):

```c
gc_array__remove_all(values, ctx);
```

**Swap two elements by index** — `gc_array__swap` returns `false` if either index is out of range; guard the indices before calling:

```c
if (offset_i < 0 || offset_j < 0 ||
    offset_i > (i64_t) UINT32_MAX || offset_j > (i64_t) UINT32_MAX ||
    !gc_array__swap(self, (u32_t) offset_i, (u32_t) offset_j)) {
    gc_machine__set_runtime_error(ctx, "swap index out of range");
}
```

**Sort, optionally by an object sub-field** — the `field` argument is a `gc_slot_tuple_u32_t` of `{.left = <type id>, .right = <field offset>}`. For arrays of objects this sorts by that field; `asc = true` for ascending order:

```c
// sort an array of mymod::Hit objects ascending by their 'distance' field
gc_array__sort(array_result, true,
    (gc_slot_tuple_u32_t) {.left = gc_mymod_Hit,
                           .right = gc_mymod_Hit_distance},
    ctx);
```

**Overwrite an existing slot** with `gc_array__set_slot` (used when filling a pre-sized array by index rather than appending):

```c
gc_array__init(arr, fields_size, ctx);
for (u32_t i = 0; i < fields_size; i++) {
    gc_unused bool cannot_fail =
        gc_array__set_slot(arr, i, (gc_slot_t) {.tu32 = tu32}, gc_type_field, ctx);
}
```


---

<a id="gcmap-h"></a>
## gc/map.h — Hash Maps

An open-addressing hash map with typed keys and values.

### Structures

```c
typedef struct {
    u64_t hash;            // Cached hash value
    gc_slot_t key;         // Key value
    gc_slot_t value;       // Value
    gc_type_t key_type;    // Key type tag
    gc_type_t value_type;  // Value type tag
} gc_map_bucket_t;

typedef struct {
    gc_object_t header;          // Object header
    gc_map_bucket_t *buckets;    // Bucket array
    u32_t size;                  // Number of entries
    u32_t capacity;              // Bucket array length
    u64_t resize_threshold;      // Load factor threshold for resize
    u64_t mask;                  // Bitmask for index computation (capacity - 1)
} gc_map_t;
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_core_map_INITIAL_CAPACITY` | 16 | Default initial capacity |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_map__init` | `void gc_map__init(gc_map_t *self, u64_t capacity, const gc_machine_t *ctx)` | Initialize the map with a given capacity (allocates storage via the call's allocator). |
| `gc_map__set` | `void gc_map__set(gc_map_t *self_map, gc_slot_t key, gc_type_t key_type, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Insert or update a key-value pair. |
| `gc_map__get` | `gc_slot_t gc_map__get(const gc_map_t *self, gc_slot_t key, gc_type_t key_type, gc_type_t *value_type, const gc_program_t *prog)` | Lookup a value by key. Writes value type to `*value_type`. |
| `gc_map__contains` | `bool gc_map__contains(const gc_map_t *self, gc_slot_t key, gc_type_t key_type, const gc_program_t *prog)` | Check if a key exists in the map. |
| `gc_map__remove` | `bool gc_map__remove(gc_map_t *self, gc_slot_t key, gc_type_t key_type, gc_machine_t *ctx)` | Remove an entry by key. Returns `true` if the key was found and removed. |

### Usage Examples

All map functions take typed `gc_slot_t` key/value pairs. Keys must be primitives or `String` objects; `gc_type_null` is never a valid key. Maps are GreyCat objects, so create them with `gc_machine__create_object` and initialize their storage with `gc_map__init` (which allocates buckets via the call's allocator).

#### Create, initialize, and populate a map

```c
// Create a Map object and allocate its bucket storage with a capacity hint.
gc_map_t *map = (gc_map_t *) gc_machine__create_object(ctx, gc_core_Map);
gc_map__init(map, 100, ctx); // grows to the next power of two >= 100

// Insert string -> int entries (e.g. counting enum occurrences).
gc_string_t *key = gc_string__create_from("apple", 5, ctx);
gc_map__set(map,
            (gc_slot_t) {.object = (gc_object_t *) key}, gc_type_object,
            (gc_slot_t) {.i64 = 1}, gc_type_int,
            ctx);
```

When no capacity is known up front, pass `gc_core_map_INITIAL_CAPACITY`; `gc_map__init` also auto-runs on the first `gc_map__set` if `capacity` is still 0.

#### Look up a value and branch on its type

`gc_map__get` writes the stored value's type tag into `*value_type` and returns `gc_type_null` there when the key is absent. Always inspect the out-parameter before reading the slot union:

```c
// Increment a counter keyed by a string, inserting 1 on first sight.
gc_type_t value_type = gc_type_null;
gc_slot_t current = gc_map__get(map,
                                (gc_slot_t) {.object = (gc_object_t *) key}, gc_type_object,
                                &value_type, gc_machine__program(ctx));
if (value_type == gc_type_int) {
    gc_map__set(map,
                (gc_slot_t) {.object = (gc_object_t *) key}, gc_type_object,
                (gc_slot_t) {.i64 = current.i64 + 1}, gc_type_int,
                ctx);
} else {
    gc_map__set(map,
                (gc_slot_t) {.object = (gc_object_t *) key}, gc_type_object,
                (gc_slot_t) {.i64 = 1}, gc_type_int,
                ctx);
}
```

#### Membership test and removal

`gc_map__contains` only needs the read-only `gc_program_t *`, while `gc_map__remove` needs the full `gc_machine_t *ctx` because it unmarks/detaches stored objects:

```c
gc_slot_t k = {.i64 = 42};
if (!gc_map__contains(map, k, gc_type_int, gc_machine__program(ctx))) {
    gc_map__set(map, k, gc_type_int, (gc_slot_t) {.object = NULL}, gc_type_null, ctx);
}

bool removed = gc_map__remove(map, k, gc_type_int, ctx);
// removed == true if the key was present and the entry was cleared
```

#### Iterate over every entry

There is no iterator API; walk the `buckets` array up to `capacity` and skip empty slots (those with `key_type == gc_type_null`). Each occupied `gc_map_bucket_t` carries the key/value slots and their type tags:

```c
gc_map_bucket_t *cursor = NULL;
for (u64_t i = 0; i < map->capacity; i++) {
    cursor = map->buckets + i;
    if (cursor->key_type == gc_type_null) {
        continue; // empty bucket
    }
    // Copy each entry into another map, preserving its key/value types.
    gc_map__set(dst, cursor->key, cursor->key_type, cursor->value, cursor->value_type, ctx);
}
```

Use `map->size` for the live entry count (e.g. to pre-size an output array) and `map->capacity` as the loop bound.


---

<a id="gctable-h"></a>
## gc/table.h — 2D Tables

A row-columnar data structure (like a DataFrame). Stores cells as `gc_slot_t` with per-cell type tags.

### Structure

```c
typedef struct {
    gc_object_t header;    // Object header
    u8_t *types;           // Per-cell type tags (cols * rows elements)
    gc_slot_t *slots;      // Per-cell values (cols * rows elements)
    u32_t cols;            // Number of columns
    u32_t rows;            // Number of rows
    u32_t capacity;        // Allocated row capacity
    u32_t start;           // Start offset (for sliding window behavior)
} gc_table_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_table__create` | `gc_table_t *gc_table__create(const gc_machine_t *ctx)` | Create a new table. |
| `gc_table__init` | `void gc_table__init(gc_table_t *table, u32_t capacity, const gc_machine_t *ctx)` | Initialize with row capacity (allocates storage via the call's allocator). |
| `gc_table__init_cols` | `bool gc_table__init_cols(gc_table_t *self, u32_t cols, const gc_machine_t *ctx)` | Set the number of columns (must be done before adding rows). |
| `gc_table__get_cell` | `bool gc_table__get_cell(const gc_table_t *self, i64_t row, i64_t col, gc_slot_t *value, gc_type_t *type)` | Read a cell value. |
| `gc_table__set_cell` | `bool gc_table__set_cell(gc_table_t *self, i64_t row, i64_t col, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Write a cell value. |
| `gc_table__set_row` | `bool gc_table__set_row(gc_table_t *self, i64_t row, gc_slot_t value, gc_type_t type, gc_machine_t *ctx)` | Set all cells in a row to the same value. |
| `gc_table__remove_row` | `bool gc_table__remove_row(gc_table_t *self, u32_t row, gc_machine_t *ctx)` | Remove a row by index. |

### Usage Examples

**Build a table by setting cells, then attach it to an object.** Mirrors how `GaussianProfile` materializes its `bins` table: create via the object's field type, size it with `gc_table__init`, then set `rows`/`cols` explicitly before writing cells. Note that the table is created with `gc_machine__create_object` (it carries an object header) and must be unmarked once stored.

```c
// Allocate a 3-column table sized for q_size rows
gc_table_t *bins = (gc_table_t *) gc_machine__create_object(ctx, table_type_id);
gc_table__init(bins, q_size * 3, ctx);   // capacity = rows * cols worth of cells
bins->rows = q_size;
bins->cols = 3;

// Write per-cell values with explicit type tags
gc_table__set_cell(bins, bin_id, 0 /*sum*/,   (gc_slot_t){.i64 = sum},   gc_type_int, ctx);
gc_table__set_cell(bins, bin_id, 1 /*sumsq*/, (gc_slot_t){.i64 = sumsq}, gc_type_int, ctx);
gc_table__set_cell(bins, bin_id, 2 /*count*/, (gc_slot_t){.i64 = count}, gc_type_int, ctx);

// Store the table on the owning object, then release the builder's mark
gc_object__set_at(self, field_idx, (gc_slot_t){.object = (gc_object_t *) bins}, gc_type_object, ctx);
gc_object__un_mark((gc_object_t *) bins, ctx);
```

**Read cells back, always checking the returned type tag.** `gc_table__get_cell` fills both the slot and its `gc_type_t`; never trust the slot union without inspecting the type (cells may be `gc_type_null` or a different type than expected).

```c
gc_slot_t slot;
gc_type_t slot_type;
if (!gc_table__get_cell(bins, bin_id, 0, &slot, &slot_type)) {
    // row/col out of bounds
    return;
}
if (slot_type != gc_type_int) {
    slot.i64 = 0;          // treat missing/null cell as zero
    slot_type = gc_type_int;
}
i64_t sum = slot.i64;
```

**Sliding-window pattern: append a new row, evict the oldest.** From `TimeWindow`, which appends `(time, value)` pairs and trims expired rows from the front via `gc_table__remove_row(table, 0, ctx)`.

```c
// Evict expired rows from the front of the window
gc_slot_t slot;
gc_type_t slot_type;
while (values->rows > 0) {
    if (!gc_table__get_cell(values, 0, 0 /*time col*/, &slot, &slot_type)) {
        break;
    }
    if (slot.i64 < cutoff) {
        gc_table__remove_row(values, 0, ctx);   // drop oldest row, shifts start
    } else {
        break;
    }
}

// Append the new sample at the next row index
u32_t row = values->rows;
gc_table__set_cell(values, row, 0, (gc_slot_t){.i64 = now.i64},      gc_type_time, ctx);
gc_table__set_cell(values, row, 1, value_slot,                       value_type,   ctx);
```

**Fill a whole row at once and pre-declare columns.** `gc_table__set_row` writes the same value/type across every column of `row`; it lazily calls `gc_table__init_cols` based on the value's shape (1 column for a scalar, 2 for a tuple, `arr->size` for an array, field-count for an object). Call `gc_table__init_cols` directly when you need to fix the column count up front.

```c
gc_table_t *table = gc_table__create(ctx);
gc_table__init(table, expected_rows, ctx);
if (!gc_table__init_cols(table, 2, ctx)) {   // two columns, set before adding rows
    return;
}

// Broadcast one value across the whole row (used by the CSV reader per parsed line)
if (!gc_table__set_row(table, row, value_slot, value_type, ctx)) {
    // allocation/columns failure
    return;
}
```


---

<a id="gctensor-h"></a>
## gc/tensor.h — Multi-dimensional Tensors

High-performance multi-dimensional arrays supporting up to 8 dimensions and multiple numeric types (i32, i64, f32, f64, c64, c128). Designed for ML and numerical computing workloads.

### Structure

```c
typedef struct {
    gc_object_t header;                        // Object header
    gc_core_tensor_descriptor_t descriptor;    // Shape, type, and size metadata
    gc_object_t *proxy_owner;                  // Proxy object that owns the data (NULL if self-owned)
    char *data;                                // Raw data pointer
} gc_core_tensor_t;
```

### Descriptor

```c
typedef struct {
    i64_t dim[GC_CORE_TENSOR_DIM_MAX];  // Shape: size of each dimension (max 8)
    i8_t nb_dim;                         // Number of dimensions (1-8)
    i8_t batch_dim;                      // Batch dimension index (-1 if none)
    u8_t type;                           // Element type (TensorType enum)
    u8_t nature;                         // Tensor nature/category
    i64_t size;                          // Total number of elements
    i64_t capacity;                      // Allocated capacity in bytes
} gc_core_tensor_descriptor_t;
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_CORE_TENSOR_DIM_MAX` | 8 | Maximum number of dimensions |
| `GC_CORE_TENSOR_SIZE_MAX` | `i64_t MAX` | Maximum total elements per tensor |
| `GC_CORE_TENSOR_CAPACITY_MAX` | `i64_t MAX` | Maximum capacity in bytes |

### Error Messages

| Constant | Message |
|----------|---------|
| `GC_ERROR_TENSOR_UNSUPPORTED_TYPE` | "Unsupported type, only float and double are supported for now" |
| `GC_ERROR_TENSOR_INCORRECT_DIM` | "Incorrect dimension" |
| `GC_ERROR_TENSOR_INCORRECT_TYPE` | "Incorrect type" |
| `GC_ERROR_TENSOR_DIMENSION_MISMATCH` | "Tensors dimension mismatch" |
| `GC_ERROR_TENSOR_MATMUL` | "MatMul can apply on at least 2 dimensions" |
| `GC_ERROR_TENSOR_DIFFERENT_TYPE` | "Tensors have different type" |
| `GC_ERROR_TENSOR_DIFFERENT_SIZE` | "Tensors have different size" |
| `GC_ERROR_TENSOR_INITIALIZE_FIRST` | "Initialize tensor first" |
| `GC_ERROR_TENSOR_TOO_LARGE` | "Tensor size is too large, over the limit" |
| `GC_ERROR_TENSOR_OUT_OF_BOUND` | "Tensor position out of bound" |
| `GC_ERROR_TENSOR_INCORRECT_SHAPE` | "Incorrect shape" |

### Creation & Initialization

| Function | Description |
|----------|-------------|
| `gc_core_tensor__create(ctx)` | Create an empty tensor (must be initialized before use) |
| `gc_machine__init_tensor(desc, proxy, data, ctx)` | Create a tensor from a pre-existing descriptor and data pointer |
| `gc_core_tensor__init_1d(self, size, type, ctx)` | Initialize as 1D (vector) |
| `gc_core_tensor__init_2d(self, rows, cols, type, ctx)` | Initialize as 2D (matrix) |
| `gc_core_tensor__init_3d(self, c, h, w, type, ctx)` | Initialize as 3D (e.g., image: channels x height x width) |
| `gc_core_tensor__init_4d(self, n, c, h, w, type, ctx)` | Initialize as 4D (e.g., batch: batch x channels x height x width) |
| `gc_core_tensor__init_5d(self, t, n, c, h, w, type, ctx)` | Initialize as 5D (e.g., video: time x batch x channels x height x width) |
| `gc_core_tensor__init_xd(self, dim, shape[], type, ctx)` | Initialize with an arbitrary number of dimensions |

### Element Access (Get / Set / Add)

Each numeric type has 1D, 2D, 3D, and N-D variants:

**Get** — Returns the element value at the given position:

```c
// Pattern: gc_core_tensor__get_{dim}d_{type}(tensor, ...indices, ctx)
f32_t gc_core_tensor__get_1d_f32(tensor, pos, ctx);
f32_t gc_core_tensor__get_2d_f32(tensor, row, col, ctx);
f32_t gc_core_tensor__get_3d_f32(tensor, c, h, w, ctx);
f32_t gc_core_tensor__get_nd_f32(tensor, n, pos[], ctx);
// Same pattern for: i32, i64, f64, c64, c128
```

**Set** — Write a value at the given position:

```c
// Pattern: gc_core_tensor__set_{dim}d_{type}(tensor, ...indices, value, ctx)
void gc_core_tensor__set_1d_f32(tensor, pos, value, ctx);
void gc_core_tensor__set_2d_f32(tensor, row, col, value, ctx);
// ... same pattern for all types and dimensions
```

**Add** — Atomically add a value at the given position and return the new value:

```c
// Pattern: gc_core_tensor__add_{dim}d_{type}(tensor, ...indices, value, ctx) -> new_value
f32_t gc_core_tensor__add_1d_f32(tensor, pos, value, ctx);
f32_t gc_core_tensor__add_2d_f32(tensor, row, col, value, ctx);
// ... same pattern for all types and dimensions
```

**Supported types for get/set/add:** `i32`, `i64`, `f32`, `f64`, `c64`, `c128`
**Supported dimensions:** 1D, 2D, 3D, N-D

### Tensor Utilities

| Function | Description |
|----------|-------------|
| `gc_core_tensor__get_data(t)` | Get the raw data pointer |
| `gc_core_tensor__get_descriptor(t)` | Get a pointer to the descriptor |
| `gc_core_tensor__set_descriptor(t, desc)` | Replace the descriptor |
| `gc_core_tensor__set_proxy(tensor, proxy)` | Set the proxy owner object |
| `gc_core_tensor__diff(t1, t2, ctx)` | Compute the sum of absolute element-wise differences between two tensors |
| `gc_core_tensor__check_shape(shape, tot_size, skip_zero)` | Validate a shape array |
| `gc_core_tensor__pos_to_offset(self, p[], ctx)` | Convert multi-dimensional position to flat offset |
| `gc_core_tensor__update_capacity(tensor, ctx)` | Reallocate data to match descriptor size |
| `gc_core_tensor__reset_internal(self)` | Reset internal state (descriptor + data pointer) |
| `gc_core_tensor__clone_internal(dst, src, ctx)` | Deep-copy tensor internals from src to dst (allocates new data buffer via the call's allocator) |
| `gc_core_tensor__print(self, name)` | Print tensor contents to stdout (debug) |

### Descriptor Utilities

| Function | Description |
|----------|-------------|
| `gc_core_tensor_descriptor__type_size(t)` | Get the byte size per element for a tensor type |
| `gc_core_tensor_descriptor__type(t)` | Convert a tensor type to `gc_type_t` |
| `gc_core_tensor_descriptor__nb_arrays(desc)` | Get the number of 1D arrays (product of dims except last) |
| `gc_core_tensor_descriptor__check_type(a, b, ctx)` | Verify two descriptors have the same element type |
| `gc_core_tensor_descriptor__check_dim(a, b, ctx)` | Verify two descriptors have the same dimensions |
| `gc_core_tensor_descriptor__update_size(desc, ctx)` | Recompute `size` from `dim[]` and `nb_dim` |
| `gc_core_tensor_descriptor__to_json(desc, buffer, prog)` | Serialize descriptor to JSON |
| `gc_core_tensor_descriptor__increment_index(nb_dim, dim, index)` | Increment a multi-dim index |
| `gc_core_tensor_descriptor__index_to_offset(nb_dim, dim, index)` | Convert multi-dim index to flat offset |
| `gc_core_tensor_descriptor__leading_dim(desc)` | Get the leading dimension size |
| `gc_core_tensor_descriptor__matrix_count(a)` | Number of matrices in a batched tensor |
| `gc_core_tensor_descriptor__supported_types(a, ctx)` | Check if the tensor type is supported for operations |
| `gc_core_tensor_descriptor__default_check(a, b, ctx)` | Default compatibility check (type + dim) |

### Matrix Multiplication Utilities

| Function | Description |
|----------|-------------|
| `gc_core_tensor_descriptor__tensor_mult_check(a, trans_a, b, trans_b, ctx)` | Validate dimensions for matrix multiplication |
| `gc_core_tensor_descriptor__tensor_mult_check_result(a, trans_a, b, trans_b, result, ctx)` | Validate result tensor dimensions for matmul |
| `gc_core_tensor_descriptor__tensor_mult_size(a, trans_a, b, trans_b, result, ctx)` | Compute result tensor size for matmul |

### Sum / Bias Utilities

| Function | Description |
|----------|-------------|
| `gc_core_tensor_descriptor__sum_check_result(a, dim, result, ctx)` | Validate result tensor for sum-reduce |
| `gc_core_tensor_descriptor__sum_size(a, dim, result, ctx)` | Compute result tensor size for sum-reduce |
| `gc_core_tensor_descriptor__add_bias_check_result(a, b, result, ctx)` | Validate result tensor for bias addition |
| `gc_core_tensor_descriptor__add_bias_size(a, b, result, ctx)` | Compute result tensor size for bias addition |

### Usage Examples

The TensorType element-type tags used everywhere below are integer constants:
`gc_core_TensorType_i32` (0), `_i64` (1), `_f32` (2), `_f64` (3), `_c64` (4), `_c128` (5).

> Note: these are the integer ordinals of the `TensorType` enum. The public tensor API takes a raw `u8_t tensor_type`, and the public headers expose only the `gc_core_TensorType` type id — not per-member constants. Pass the ordinal directly; plugins typically `#define gc_core_TensorType_f64 3` (etc.) in their own header for readability, as the bundled libraries do.

**Create, initialize, fill, and return a tensor** — the canonical native-function pattern. `gc_core_tensor__create` produces an empty (uninitialized) tensor object; you must call one of the `init_*` functions before any element access. `init_*` may set a runtime error (e.g. `GC_ERROR_TENSOR_TOO_LARGE`, `GC_ERROR_TENSOR_INCORRECT_DIM`) and leave the tensor untouched, so always check before writing. Because `create` returns a freshly marked object, un-mark it after publishing the result:

```c
void my_make_matrix(gc_machine_t *ctx) {
    i64_t rows = gc_machine__this(ctx).i64; // example dimension source
    i64_t cols = 4;

    gc_core_tensor_t *t = gc_core_tensor__create(ctx);
    gc_core_tensor__init_2d(t, rows, cols, gc_core_TensorType_f64, ctx);
    if (gc_machine__error(ctx)) {
        gc_object__un_mark((gc_object_t *) t, ctx);
        return;
    }

    // Fill: set_2d writes value at (row, column); add_2d adds and returns the new value
    for (i64_t r = 0; r < rows; r++) {
        for (i64_t c = 0; c < cols; c++) {
            gc_core_tensor__set_2d_f64(t, r, c, (f64_t) (r * cols + c), ctx);
        }
    }
    f64_t updated = gc_core_tensor__add_2d_f64(t, 0, 0, 10.0, ctx); // accumulate in place

    gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) t}, gc_type_object);
    gc_object__un_mark((gc_object_t *) t, ctx);
}
```

**1-D vectors via `init_xd` and element write** — `init_1d`/`init_2d`/`init_3d`/`init_4d`/`init_5d` are thin wrappers over `init_xd`, which takes an explicit dimension count and a `shape[]` array (length up to `GC_CORE_TENSOR_DIM_MAX`). Use `init_xd` when the rank is dynamic:

```c
gc_core_tensor_t *vec = gc_core_tensor__create(ctx);
i64_t shape[1] = { values->size };
gc_core_tensor__init_xd(vec, 1, shape, gc_core_TensorType_f64, ctx);
for (i64_t i = 0; i < values->size; i++) {
    gc_core_tensor__set_1d_f64(vec, i, source_data[i], ctx);
}
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) vec}, gc_type_object);
gc_object__un_mark((gc_object_t *) vec, ctx);
```

**N-dimensional access** — for ranks above 3 (or generic code), use the `_nd_` variants, which take the dimension count `n` and an index array `pos[]`:

```c
i64_t pos[4] = { batch, channel, y, x };
f32_t pixel = gc_core_tensor__get_nd_f32(t, 4, pos, ctx);
gc_core_tensor__set_nd_f32(t, 4, pos, pixel * 0.5f, ctx);
```

**Reading the descriptor and raw data** — `gc_core_tensor__get_descriptor` returns a pointer into the tensor (no copy); `gc_core_tensor__get_data` returns the raw byte buffer. Combine with `gc_core_tensor_descriptor__type_size` (bytes per element) to walk memory directly, the same way the runtime hashes/compares tensors:

```c
gc_core_tensor_descriptor_t *d = gc_core_tensor__get_descriptor(t);
u8_t elem_size = gc_core_tensor_descriptor__type_size(d->type);
char *raw = gc_core_tensor__get_data(t);
u64_t byte_len = (u64_t) d->size * elem_size;
// e.g. raw + (offset * elem_size) addresses element `offset`
```

**Validating two tensors before an element-wise op** — `check_type` and `check_dim` set a runtime error and return false on mismatch, so they double as guards. This mirrors the distance/diff implementations:

```c
gc_core_tensor_t *x = /* ... */;
gc_core_tensor_t *y = /* ... */;
if (!gc_core_tensor_descriptor__check_type(&x->descriptor, &y->descriptor, ctx)) {
    return; // GC_ERROR_TENSOR_DIFFERENT_TYPE already set
}
if (x->descriptor.size != y->descriptor.size) {
    gc_machine__set_runtime_error(ctx, GC_ERROR_TENSOR_DIFFERENT_SIZE);
    return;
}
if (!gc_core_tensor_descriptor__check_dim(&x->descriptor, &y->descriptor, ctx)) {
    return; // GC_ERROR_TENSOR_DIMENSION_MISMATCH already set
}
f64_t max_abs_diff = gc_core_tensor__diff(x, y, ctx); // sum/max of |x[i] - y[i]|
```

**`supported_types` / `default_check`** — `default_check` bundles the usual precondition for binary ops: same dims, same type, and both types supported. Use it instead of chaining the individual checks:

```c
if (!gc_core_tensor_descriptor__default_check(&a->descriptor, &b->descriptor, ctx)) {
    return; // appropriate GC_ERROR_TENSOR_* set by the failing sub-check
}
// safe to proceed with the element-wise kernel
```

**Wrapping an externally-owned buffer** — `gc_machine__init_tensor` builds a tensor over a pre-existing `data` pointer owned by another object (`proxy`). It marks the proxy so the buffer stays alive; do not free `data` through this tensor:

```c
gc_core_tensor_descriptor_t desc = {0};
desc.nb_dim = 2;
desc.dim[0] = rows;
desc.dim[1] = cols;
desc.batch_dim = -1;
desc.type = gc_core_TensorType_f32;
gc_core_tensor_descriptor__update_size(&desc, ctx); // recompute desc.size from dim[]/nb_dim
gc_core_tensor_t *view = gc_machine__init_tensor(desc, owner_obj, owner_buffer, ctx);
```

**Validating a GCL-supplied shape** — `gc_core_tensor__check_shape` validates an `array<int>` shape (rank ≤ 8, non-negative entries, no overflow) and writes the total element count through `tot_size`. Pass `skip_zero = false` to reject empty dims:

```c
gc_array_t *new_shape = (gc_array_t *) gc_machine__get_param(ctx, 0).object; // the array<int> shape parameter
u64_t new_size;
if (!gc_core_tensor__check_shape(new_shape, &new_size, false)) {
    gc_machine__set_runtime_error(ctx, GC_ERROR_TENSOR_INCORRECT_SHAPE);
    return;
}
```

**Matrix-multiply descriptor planning** — before allocating a result tensor for a (possibly batched, possibly transposed) matmul, validate the operand shapes with `tensor_mult_check`, then derive the result size with `tensor_mult_size` (which fills `result->dim[]`/`size`). `tensor_mult_check_result` verifies a caller-provided result descriptor instead. `matrix_count` and `leading_dim` are the building blocks these use:

```c
gc_core_tensor_descriptor_t *a = &lhs->descriptor;
gc_core_tensor_descriptor_t *b = &rhs->descriptor;
if (!gc_core_tensor_descriptor__tensor_mult_check(a, /*trans_a*/ false, b, /*trans_b*/ false, ctx)) {
    return; // GC_ERROR_TENSOR_MATMUL or dimension mismatch already set
}
gc_core_tensor_descriptor_t out = {0};
out.type = a->type;
if (!gc_core_tensor_descriptor__tensor_mult_size(a, false, b, false, &out, ctx)) {
    return;
}
i64_t matrices = gc_core_tensor_descriptor__matrix_count(a); // product of all dims except last two
```

**Sum-reduce and bias-add planning** — `sum_size` computes the result descriptor for reducing over `dim`; `add_bias_size` computes the broadcast result for adding a bias tensor `b` to `a`. The `*_check_result` siblings validate a result descriptor you already built:

```c
gc_core_tensor_descriptor_t reduced = {0};
reduced.type = src->descriptor.type;
if (!gc_core_tensor_descriptor__sum_size(&src->descriptor, /*dim*/ 0, &reduced, ctx)) {
    return;
}

gc_core_tensor_descriptor_t biased = {0};
biased.type = src->descriptor.type;
if (!gc_core_tensor_descriptor__add_bias_size(&src->descriptor, &bias->descriptor, &biased, ctx)) {
    return;
}
```

**Iterating an arbitrary-rank tensor by multi-index** — `increment_index` advances a `nb_dim`-length index (row-major, returns false when it wraps past the end), and `index_to_offset` converts that index to a flat element offset. `nb_arrays` gives the number of trailing 1-D rows:

```c
gc_core_tensor_descriptor_t *d = &t->descriptor;
i64_t index[GC_CORE_TENSOR_DIM_MAX] = {0};
do {
    i64_t off = gc_core_tensor_descriptor__index_to_offset(d->nb_dim, d->dim, index);
    // ... visit element at flat position `off`
} while (gc_core_tensor_descriptor__increment_index(d->nb_dim, d->dim, index));
```

**Serializing a descriptor to JSON** — used by the tensor's `to_string`/save path:

```c
gc_core_tensor_descriptor__to_json(&t->descriptor, buffer, prog);
```


---
