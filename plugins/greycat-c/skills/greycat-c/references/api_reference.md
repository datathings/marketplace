# GreyCat C API Complete Reference

This document provides comprehensive coverage of the GreyCat C API, organized by functional area with extensive examples from real usage.

## Table of Contents

1. [Core Types and Structures](#core-types-and-structures)
2. [Machine API](#machine-api)
3. [Object API](#object-api)
4. [Tensor API](#tensor-api)
5. [Tensor Operation Examples](#tensor-operation-examples)
6. [Array API](#array-api)
7. [Table API](#table-api)
8. [Buffer API](#buffer-api)
9. [String API](#string-api)
10. [Memory Allocation](#memory-allocation)
11. [Program and Type System](#program-and-type-system)
12. [Host and Task Management](#host-and-task-management)
13. [HTTP Client](#http-client)
14. [Cryptography](#cryptography)
15. [Utility Functions](#utility-functions)
16. [I/O Operations](#io-operations)
17. [Common Patterns](#common-patterns)

---

## Core Types and Structures

### Basic Type Definitions

```c
// Integer types
typedef int8_t i8_t;
typedef uint8_t u8_t;
typedef int16_t i16_t;
typedef uint16_t u16_t;
typedef int32_t i32_t;
typedef uint32_t u32_t;
typedef int64_t i64_t;
typedef uint64_t u64_t;

// Floating point types
typedef float f32_t;
typedef double f64_t;

// Complex types (not available on WASM)
typedef double _Complex c128_t;
typedef float _Complex c64_t;

// Geospatial type
typedef u64_t geo_t;
```

### gc_type_t Enum

The `gc_type_t` enum defines all possible GreyCat types. Must fit in 8 bits.

```c
typedef enum {
    gc_type_null = 0,
    gc_type_bool = 1,
    gc_type_char = 2,
    gc_type_int = 3,
    gc_type_float = 4,
    gc_type_node = 5,
    gc_type_node_time = 6,
    gc_type_node_index = 7,
    gc_type_node_list = 8,
    gc_type_node_geo = 9,
    gc_type_geo = 10,
    gc_type_time = 11,
    gc_type_duration = 12,
    gc_type_cubic = 13,
    gc_type_static_field = 14,
    gc_type_object = 15,
    gc_type_t2 = 16,
    gc_type_t3 = 17,
    gc_type_t4 = 18,
    gc_type_str = 19,
    gc_type_t2f = 20,
    gc_type_t3f = 21,
    gc_type_t4f = 22,
    gc_type_block_ref = 23,
    gc_type_block_inline = 24,
    gc_type_function = 25,
    gc_type_undefined = 26,
    gc_type_type = 27,
    gc_type_field = 28,
    gc_type_stringlit = 29,  // serialization only
    gc_type_error = 30       // C internal only
} gc_type_t;
```

### gc_object_t Structure

Generic handle for GreyCat Objects. Must be packed to 128 bits (64+32+32).

```c
typedef struct {
    gc_block_t *block;
    u32_t marks;
    u32_t type_id;
} gc_object_t;
```

### gc_slot_t Union

Generic GreyCat variable handle that can hold any type of value.

```c
typedef struct {
    union {
        bool b;
        u8_t byte[8];
        u32_t u32;
        i64_t i64;
        u64_t u64;
        f64_t f64;
        gc_slot_tuple_u32_t tu32;
        gc_object_t *object;
    };
} gc_slot_t;
```

**Example: Creating slots with different types**

```c
// Boolean slot
gc_slot_t bool_slot = {.b = true};

// Integer slot
gc_slot_t int_slot = {.i64 = 42};

// Float slot
gc_slot_t float_slot = {.f64 = 3.14159};

// Object slot
gc_slot_t obj_slot = {.object = some_object};

// Static field (enum) slot
gc_slot_t enum_slot = {.tu32 = {.left = module_id, .right = value}};
```

---

## Machine API

The machine context (`gc_machine_t`) is the primary interface for native function implementations.

### Getting Parameters

```c
// Get parameter by offset (0-indexed)
gc_slot_t gc_machine__get_param(const gc_machine_t *ctx, u32_t offset);

// Get parameter type
gc_type_t gc_machine__get_param_type(const gc_machine_t *ctx, u32_t offset);

// Get number of parameters
u32_t gc_machine__get_param_nb(const gc_machine_t *ctx);
```

**Example: Reading function parameters**

```c
void my_native_function(gc_machine_t *ctx) {
    // Check parameter count
    u32_t param_count = gc_machine__get_param_nb(ctx);
    if (param_count < 2) {
        gc_machine__set_runtime_error(ctx, "Expected at least 2 parameters");
        return;
    }

    // Get first parameter
    gc_slot_t param0 = gc_machine__get_param(ctx, 0);
    gc_type_t type0 = gc_machine__get_param_type(ctx, 0);

    // Check type and use value
    if (type0 == gc_type_int) {
        i64_t value = param0.i64;
        // Use the value...
    } else if (type0 == gc_type_object) {
        gc_object_t *obj = param0.object;
        // Use the object...
    }
}
```

### Setting Results

```c
// Set the function result
void gc_machine__set_result(gc_machine_t *self, gc_slot_t slot, gc_type_t slot_type);

// Get the expected return type
u32_t gc_machine__return_type(gc_machine_t *self);

// Create an object of the return type
gc_object_t *gc_machine__create_return_type_object(gc_machine_t *ctx);
```

**Example: Returning values**

```c
void compute_sum(gc_machine_t *ctx) {
    gc_slot_t a = gc_machine__get_param(ctx, 0);
    gc_slot_t b = gc_machine__get_param(ctx, 1);

    i64_t sum = a.i64 + b.i64;

    // Return the result
    gc_slot_t result = {.i64 = sum};
    gc_machine__set_result(ctx, result, gc_type_int);
}
```

### Error Handling

```c
// Set a runtime error with custom message
void gc_machine__set_runtime_error(gc_machine_t *ctx, const char *msg);

// Set a runtime error from system errno
void gc_machine__set_runtime_error_syserr(gc_machine_t *ctx);

// Check if an error occurred
bool gc_machine__error(gc_machine_t *ctx);
```

**Example: Error handling (from matmul.c)**

```c
if (!gc_core_tensor_descriptor__check_type(&inputs[0]->desc, &inputs[1]->desc, ctx)) {
    gc_machine__set_runtime_error(ctx, "type incompatible");
    return false;
}
```

### Object Creation

```c
// Create an object of a specific type
gc_object_t *gc_machine__create_object(const gc_machine_t *ctx, u32_t object_type_code);
```

**Example: Reading layer configuration (from dense_layer.c)**

```c
gc_type_t type = gc_type_null;
gc_slot_t value = gc_object__get_at(layer, gc_compute_ComputeLayerDense_use_bias, &type, ctx);
if (type == gc_type_bool) {
    use_bias = value.b;
}

value = gc_object__get_at(layer, gc_compute_ComputeLayerDense_type, &type, ctx);
if (type == gc_type_static_field) {
    tensor_type = value.tu32.right;
}
```

---

## Object API

### Getting Field Values

```c
// Get field by key (symbol ID)
gc_slot_t gc_object__get(gc_object_t *self, u32_t key, gc_type_t *type, gc_machine_t *ctx);

// Get field by offset
gc_slot_t gc_object__get_at(const gc_object_t *self, u32_t offset, gc_type_t *type_res, const gc_machine_t *ctx);

// Get default value for a field
gc_slot_t gc_object__get_default_at(u32_t type_id, u32_t offset, gc_type_t *type_res, gc_machine_t *ctx);
```

**Example: Reading object fields (from dense_layer.c)**

```c
bool use_bias = true;
gc_type_t type = gc_type_null;
gc_slot_t value = gc_object__get_at(layer, gc_compute_ComputeLayerDense_use_bias, &type, ctx);
if (type == gc_type_bool) {
    use_bias = value.b;
}

value = gc_object__get_at(layer, gc_compute_ComputeLayerDense_inputs, &type, ctx);
if (type == gc_type_int) {
    i64_t inputs = value.i64;
}

value = gc_object__get_at(layer, gc_compute_ComputeLayerDense_activation, &type, ctx);
if (type == gc_type_object) {
    gc_object_t *activation = value.object;
    // Use activation object...
}
```

### Setting Field Values

```c
// Set field by key
bool gc_object__set(gc_object_t *self, u32_t key, gc_slot_t value, gc_type_t type, gc_machine_t *ctx);

// Set field by offset
bool gc_object__set_at(gc_object_t *self, u32_t offset, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx);
```

### Object Operations

```c
// Mark/unmark object (for GC)
void gc_object__mark(gc_object_t *self);
void gc_object__un_mark(gc_object_t *self, gc_machine_t *ctx);

// Deep clone an object
gc_object_t *gc_object__deep_clone_object(gc_object_t *self, gc_machine_t *ctx);

// Declare object dirty (for persistence)
void gc_object__declare_dirty(gc_object_t *self);

// Type checking
bool gc_object__is_instance_of(const gc_object_t *self, u32_t of_type, gc_machine_t *ctx);
```

### Slot Type Conversion

```c
// Convert slot to int64
i64_t gc_slot__to_i64(gc_slot_t slot, gc_type_t slot_type);

// Convert slot to float64
f64_t gc_slot__to_f64(gc_slot_t slot, gc_type_t slot_type);
```

---

## Tensor API

Tensors are multi-dimensional arrays with support for various numeric types, optimized for ML and numerical computing workloads.

### Tensor Types

```c
#define gc_core_TensorType_i32  0  // 32-bit integer
#define gc_core_TensorType_i64  1  // 64-bit integer
#define gc_core_TensorType_f32  2  // 32-bit float (single precision)
#define gc_core_TensorType_f64  3  // 64-bit float (double precision)
#define gc_core_TensorType_c64  4  // Complex float (32-bit real + 32-bit imaginary)
#define gc_core_TensorType_c128 5  // Complex double (64-bit real + 64-bit imaginary)
```

### Tensor Descriptor

The descriptor contains tensor metadata: shape, type, size, and memory layout.

```c
#define GC_CORE_TENSOR_DIM_MAX 8  // Maximum 8 dimensions
#define GC_CORE_TENSOR_SIZE_MAX 9223372036854775807     // Max elements per tensor
#define GC_CORE_TENSOR_CAPACITY_MAX 9223372036854775807 // Max capacity in bytes

typedef struct {
    i64_t dim[GC_CORE_TENSOR_DIM_MAX];  // Shape: dimensions along each axis
    i8_t nb_dim;                         // Number of dimensions (1-8)
    i8_t batch_dim;                      // Batch dimension index (-1 if none)
    u8_t type;                           // Tensor type (TensorType enum)
    u8_t nature;                         // Tensor nature/category
    i64_t size;                          // Total number of elements
    i64_t capacity;                      // Allocated capacity in bytes
} gc_core_tensor_descriptor_t;
```

**Key descriptor fields:**
- `dim[]`: Shape array (e.g., `[3, 224, 224]` for 3-channel 224x224 image)
- `nb_dim`: Number of dimensions (3 for above example)
- `batch_dim`: Which dimension is the batch (-1 if not batched)
- `type`: Data type (f32, f64, i32, etc.)
- `size`: Total elements (`3 * 224 * 224 = 150528` for above)
- `capacity`: Memory allocated in bytes (`size * sizeof(type)`)

### Creating and Initializing Tensors

```c
// Create empty tensor (must initialize before use)
gc_core_tensor_t *gc_core_tensor__create(const gc_machine_t *ctx);

// Initialize with custom descriptor and data
gc_core_tensor_t *gc_machine__init_tensor(gc_core_tensor_descriptor_t desc,
                                          gc_object_t *proxy,
                                          char *data,
                                          const gc_machine_t *ctx);

// Initialize with specific dimensionality (convenience functions)
void gc_core_tensor__init_1d(gc_core_tensor_t *self, i64_t size, u8_t tensor_type, gc_machine_t *ctx);
void gc_core_tensor__init_2d(gc_core_tensor_t *self, i64_t rows, i64_t columns, u8_t tensor_type, gc_machine_t *ctx);
void gc_core_tensor__init_3d(gc_core_tensor_t *self, i64_t c, i64_t h, i64_t w, u8_t tensor_type, gc_machine_t *ctx);
void gc_core_tensor__init_4d(gc_core_tensor_t *self, i64_t n, i64_t c, i64_t h, i64_t w, u8_t tensor_type, gc_machine_t *ctx);
void gc_core_tensor__init_5d(gc_core_tensor_t *self, i64_t t, i64_t n, i64_t c, i64_t h, i64_t w, u8_t tensor_type, gc_machine_t *ctx);

// Initialize with arbitrary dimensions
void gc_core_tensor__init_xd(gc_core_tensor_t *self, i64_t tensor_dim, const i64_t shape[], u8_t tensor_type, gc_machine_t *ctx);
```

**Example: Creating various tensor shapes**

```c
// 1D vector (1000 elements)
gc_core_tensor_t *vec = gc_core_tensor__create(ctx);
gc_core_tensor__init_1d(vec, 1000, gc_core_TensorType_f32, ctx);

// 2D matrix (100 rows x 50 columns)
gc_core_tensor_t *matrix = gc_core_tensor__create(ctx);
gc_core_tensor__init_2d(matrix, 100, 50, gc_core_TensorType_f64, ctx);

// 3D tensor - RGB image (3 channels x 224 height x 224 width)
gc_core_tensor_t *image = gc_core_tensor__create(ctx);
gc_core_tensor__init_3d(image, 3, 224, 224, gc_core_TensorType_f32, ctx);

// 4D tensor - batch of images (32 batch x 3 channels x 224 height x 224 width)
gc_core_tensor_t *batch = gc_core_tensor__create(ctx);
gc_core_tensor__init_4d(batch, 32, 3, 224, 224, gc_core_TensorType_f32, ctx);

// Custom 6D tensor
i64_t shape[] = {2, 3, 4, 5, 6, 7};
gc_core_tensor_t *custom = gc_core_tensor__create(ctx);
gc_core_tensor__init_xd(custom, 6, shape, gc_core_TensorType_f64, ctx);
```

### Getting Tensor Values

Each type has specialized getter functions for 1D, 2D, 3D, and N-D access:

```c
// i32 getters
i32_t gc_core_tensor__get_1d_i32(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
i32_t gc_core_tensor__get_2d_i32(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
i32_t gc_core_tensor__get_3d_i32(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
i32_t gc_core_tensor__get_nd_i32(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);

// i64 getters
i64_t gc_core_tensor__get_1d_i64(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
i64_t gc_core_tensor__get_2d_i64(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
i64_t gc_core_tensor__get_3d_i64(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
i64_t gc_core_tensor__get_nd_i64(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);

// f32 getters
f32_t gc_core_tensor__get_1d_f32(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
f32_t gc_core_tensor__get_2d_f32(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
f32_t gc_core_tensor__get_3d_f32(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
f32_t gc_core_tensor__get_nd_f32(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);

// f64 getters
f64_t gc_core_tensor__get_1d_f64(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
f64_t gc_core_tensor__get_2d_f64(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
f64_t gc_core_tensor__get_3d_f64(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
f64_t gc_core_tensor__get_nd_f64(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);

// c64 getters (complex float)
c64_t gc_core_tensor__get_1d_c64(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
c64_t gc_core_tensor__get_2d_c64(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
c64_t gc_core_tensor__get_3d_c64(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
c64_t gc_core_tensor__get_nd_c64(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);

// c128 getters (complex double)
c128_t gc_core_tensor__get_1d_c128(const gc_core_tensor_t *tensor, i64_t pos, gc_machine_t *ctx);
c128_t gc_core_tensor__get_2d_c128(const gc_core_tensor_t *tensor, i64_t row, i64_t column, gc_machine_t *ctx);
c128_t gc_core_tensor__get_3d_c128(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, gc_machine_t *ctx);
c128_t gc_core_tensor__get_nd_c128(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], gc_machine_t *ctx);
```

**Example: Reading tensor values**

```c
// Read from 1D vector
f32_t value = gc_core_tensor__get_1d_f32(vec, 42, ctx);

// Read from 2D matrix
f64_t element = gc_core_tensor__get_2d_f64(matrix, 5, 10, ctx);

// Read pixel from RGB image (channel 0, row 112, col 112)
f32_t pixel = gc_core_tensor__get_3d_f32(image, 0, 112, 112, ctx);

// Read from 6D tensor using N-D accessor
i64_t indices[] = {1, 2, 3, 4, 5, 6};
f64_t nd_value = gc_core_tensor__get_nd_f64(custom, 6, indices, ctx);
```

### Setting Tensor Values

Each type has specialized setter functions matching the getters:

```c
// f32 setters
void gc_core_tensor__set_1d_f32(const gc_core_tensor_t *tensor, i64_t pos, f32_t value, gc_machine_t *ctx);
void gc_core_tensor__set_2d_f32(const gc_core_tensor_t *tensor, i64_t row, i64_t column, f32_t value, gc_machine_t *ctx);
void gc_core_tensor__set_3d_f32(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, f32_t value, gc_machine_t *ctx);
void gc_core_tensor__set_nd_f32(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], f32_t value, gc_machine_t *ctx);

// f64 setters
void gc_core_tensor__set_1d_f64(const gc_core_tensor_t *tensor, i64_t pos, f64_t value, gc_machine_t *ctx);
void gc_core_tensor__set_2d_f64(const gc_core_tensor_t *tensor, i64_t row, i64_t column, f64_t value, gc_machine_t *ctx);
void gc_core_tensor__set_3d_f64(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, f64_t value, gc_machine_t *ctx);
void gc_core_tensor__set_nd_f64(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], f64_t value, gc_machine_t *ctx);

// Similar patterns for i32, i64, c64, c128...
```

**Example: Writing tensor values**

```c
// Write to 1D vector
gc_core_tensor__set_1d_f32(vec, 42, 3.14f, ctx);

// Write to 2D matrix
gc_core_tensor__set_2d_f64(matrix, 5, 10, 2.71828, ctx);

// Initialize 3D tensor to zeros
gc_core_tensor_descriptor_t *desc = gc_core_tensor__get_descriptor(image);
for (i64_t c = 0; c < desc->dim[0]; c++) {
    for (i64_t h = 0; h < desc->dim[1]; h++) {
        for (i64_t w = 0; w < desc->dim[2]; w++) {
            gc_core_tensor__set_3d_f32(image, c, h, w, 0.0f, ctx);
        }
    }
}
```

### Atomic Add Operations

Adds a value to the current value and returns the new value (atomic for gradient accumulation):

```c
// f32 add operations
f32_t gc_core_tensor__add_1d_f32(const gc_core_tensor_t *tensor, i64_t pos, f32_t value, gc_machine_t *ctx);
f32_t gc_core_tensor__add_2d_f32(const gc_core_tensor_t *tensor, i64_t row, i64_t column, f32_t value, gc_machine_t *ctx);
f32_t gc_core_tensor__add_3d_f32(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, f32_t value, gc_machine_t *ctx);
f32_t gc_core_tensor__add_nd_f32(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], f32_t value, gc_machine_t *ctx);

// f64 add operations
f64_t gc_core_tensor__add_1d_f64(const gc_core_tensor_t *tensor, i64_t pos, f64_t value, gc_machine_t *ctx);
f64_t gc_core_tensor__add_2d_f64(const gc_core_tensor_t *tensor, i64_t row, i64_t column, f64_t value, gc_machine_t *ctx);
f64_t gc_core_tensor__add_3d_f64(const gc_core_tensor_t *tensor, i64_t c, i64_t h, i64_t w, f64_t value, gc_machine_t *ctx);
f64_t gc_core_tensor__add_nd_f64(const gc_core_tensor_t *tensor, i64_t n, const i64_t pos[], f64_t value, gc_machine_t *ctx);

// Similar patterns for i32, i64, c64, c128...
```

**Example: Gradient accumulation (from mul.c backward pass)**

```c
// Element-wise multiply backward: dA[i] += B[i] * dC[i]
const i64_t N = inputs_desc[0]->size;
f64_t *dA = inputs_grad[0];
const f64_t *B = inputs[1];
const f64_t *dC = outputs_grad[0];

if (dA != NULL) {
    for (i64_t i = 0; i < N; i++) {
        dA[i] += B[i] * dC[i];  // Accumulate gradient
    }
}
```

### Tensor Descriptor Operations

```c
// Get/set descriptor
gc_core_tensor_descriptor_t *gc_core_tensor__get_descriptor(gc_core_tensor_t *t);
void gc_core_tensor__set_descriptor(gc_core_tensor_t *t, gc_core_tensor_descriptor_t desc);

// Get raw data pointer (for direct memory access)
char *gc_core_tensor__get_data(gc_core_tensor_t *t);

// Set proxy object
void gc_core_tensor__set_proxy(gc_core_tensor_t *tensor, gc_object_t *proxy);

// Descriptor type utilities
u8_t gc_core_tensor_descriptor__type_size(u8_t t);        // Size in bytes for type
gc_type_t gc_core_tensor_descriptor__type(u8_t t);        // Convert to gc_type_t
i64_t gc_core_tensor_descriptor__nb_arrays(gc_core_tensor_descriptor_t *descriptor);
i64_t gc_core_tensor_descriptor__matrix_count(gc_core_tensor_descriptor_t *a);    // For batched matrices
i64_t gc_core_tensor_descriptor__leading_dim(gc_core_tensor_descriptor_t *descriptor);

// Index utilities
bool gc_core_tensor_descriptor__increment_index(i8_t nb_dim, const i64_t *dim, i64_t *index);
i64_t gc_core_tensor_descriptor__index_to_offset(i8_t nb_dim, const i64_t *dim, const i64_t *index);

// JSON serialization
void gc_core_tensor_descriptor__to_json(const gc_core_tensor_descriptor_t *self, gc_buffer_t *buffer, const gc_program_t *prog);
```

**Example: Direct memory access pattern (from add.c)**

```c
// Get raw pointers for performance-critical loops
const i64_t N = inputs_desc[0]->size;
const f64_t *A = (f64_t *)gc_core_tensor__get_data(input_a);
const f64_t *B = (f64_t *)gc_core_tensor__get_data(input_b);
f64_t *C = (f64_t *)gc_core_tensor__get_data(output);

// Element-wise addition
for (i64_t i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

### Tensor Validation and Shape Operations

```c
// Validate shape array
bool gc_core_tensor__check_shape(gc_core_array_t *shape, u64_t *tot_size, bool skip_zero);

// Update capacity based on size
bool gc_core_tensor__update_capacity(gc_core_tensor_t *tensor, gc_machine_t *ctx);

// Type and dimension checking
bool gc_core_tensor_descriptor__check_type(gc_core_tensor_descriptor_t *a, gc_core_tensor_descriptor_t *b, gc_machine_t *ctx);
bool gc_core_tensor_descriptor__check_dim(gc_core_tensor_descriptor_t *a, gc_core_tensor_descriptor_t *b, gc_machine_t *ctx);
bool gc_core_tensor_descriptor__update_size(gc_core_tensor_descriptor_t *descriptor, gc_machine_t *ctx);

// Supported type check
bool gc_core_tensor_descriptor__supported_types(gc_core_tensor_descriptor_t *a, gc_machine_t *ctx);
bool gc_core_tensor_descriptor__default_check(gc_core_tensor_descriptor_t *a, gc_core_tensor_descriptor_t *b, gc_machine_t *ctx);
```

**Example: Type validation (from matmul.c)**

```c
// Ensure tensors have compatible types
if (!gc_core_tensor_descriptor__check_type(&inputs[0]->desc, &inputs[1]->desc, ctx)) {
    gc_machine__set_runtime_error(ctx, "type incompatible");
    return false;
}
```

### Matrix Multiplication Validation

```c
// Check if matrix multiplication is valid
bool gc_core_tensor_descriptor__tensor_mult_check(gc_core_tensor_descriptor_t *a, bool trans_a,
                                                  gc_core_tensor_descriptor_t *b, bool trans_b,
                                                  gc_machine_t *ctx);

// Check and compute result shape
bool gc_core_tensor_descriptor__tensor_mult_check_result(gc_core_tensor_descriptor_t *a, bool trans_a,
                                                        gc_core_tensor_descriptor_t *b, bool trans_b,
                                                        gc_core_tensor_descriptor_t *result,
                                                        gc_machine_t *ctx);

// Compute output shape for matmul
bool gc_core_tensor_descriptor__tensor_mult_size(gc_core_tensor_descriptor_t *a, bool trans_a,
                                                gc_core_tensor_descriptor_t *b, bool trans_b,
                                                gc_core_tensor_descriptor_t *result,
                                                gc_machine_t *ctx);
```

**Example: Matrix multiplication validation (from matmul.c)**

```c
bool transpose_a = constants[0].b;
bool transpose_b = constants[1].b;

// Validate multiplication is possible
if (!gc_core_tensor_descriptor__tensor_mult_check(&inputs[0]->desc, transpose_a,
                                                  &inputs[1]->desc, transpose_b, ctx)) {
    return false;
}

// Compute output shape
if (!gc_core_tensor_descriptor__tensor_mult_size(&inputs[0]->desc, transpose_a,
                                                &inputs[1]->desc, transpose_b,
                                                &outputs[0]->desc, ctx)) {
    return false;
}
```

### Add Bias Validation

```c
// Check if add bias operation is valid
bool gc_core_tensor_descriptor__add_bias_check_result(gc_core_tensor_descriptor_t *a,
                                                     gc_core_tensor_descriptor_t *b,
                                                     gc_core_tensor_descriptor_t *result,
                                                     gc_machine_t *ctx);

// Compute output shape for add bias
bool gc_core_tensor_descriptor__add_bias_size(gc_core_tensor_descriptor_t *a,
                                             gc_core_tensor_descriptor_t *b,
                                             gc_core_tensor_descriptor_t *result,
                                             gc_machine_t *ctx);
```

### Sum Operations

```c
// Check sum reduction validity
bool gc_core_tensor_descriptor__sum_check_result(gc_core_tensor_descriptor_t *a, i64_t dim,
                                                gc_core_tensor_descriptor_t *result,
                                                gc_machine_t *ctx);

// Compute output shape for sum
bool gc_core_tensor_descriptor__sum_size(gc_core_tensor_descriptor_t *a, i64_t dim,
                                        gc_core_tensor_descriptor_t *result,
                                        gc_machine_t *ctx);
```

### Tensor Utility Functions

```c
// Clone and reset operations
void gc_core_tensor__reset_internal(gc_core_tensor_t *self);
void gc_core_tensor__clone_internal(gc_core_tensor_t *dst, gc_core_tensor_t *src);

// Compute difference between tensors
f64_t gc_core_tensor__diff(gc_core_tensor_t *t1, gc_core_tensor_t *t2, gc_machine_t *ctx);

// Print tensor for debugging
void gc_core_tensor__print(gc_core_tensor_t *self, const char *name);
```

---

## Tensor Operation Examples

### Example 1: Element-wise Addition (from add.c)

```c
static void tensor_add_f64(f64_t **inputs,
                          gc_core_tensor_descriptor_t **inputs_desc,
                          f64_t **outputs) {
    const i64_t N = inputs_desc[0]->size;
    const f64_t *A = inputs[0];
    const f64_t *B = inputs[1];
    f64_t *C = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
```

### Example 2: Element-wise Multiplication (from mul.c)

```c
static void tensor_mul_f64(f64_t **inputs,
                          gc_core_tensor_descriptor_t **inputs_desc,
                          f64_t **outputs) {
    const i64_t N = inputs_desc[0]->size;
    const f64_t *A = inputs[0];
    const f64_t *B = inputs[1];
    f64_t *C = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        C[i] = A[i] * B[i];
    }
}

// Backward pass with gradient accumulation
static void tensor_mul_backward_f64(f64_t **inputs,
                                   f64_t **inputs_grad,
                                   gc_core_tensor_descriptor_t **inputs_desc,
                                   f64_t **outputs_grad) {
    const i64_t N = inputs_desc[0]->size;
    f64_t *dA = inputs_grad[0];
    f64_t *dB = inputs_grad[1];
    const f64_t *A = inputs[0];
    const f64_t *B = inputs[1];
    const f64_t *dC = outputs_grad[0];

    if (dA != NULL && dB != NULL) {
        // Both gradients needed
        for (i64_t i = 0; i < N; i++) {
            dA[i] += B[i] * dC[i];  // ∂L/∂A = B ⊙ ∂L/∂C
            dB[i] += A[i] * dC[i];  // ∂L/∂B = A ⊙ ∂L/∂C
        }
    }
}
```

### Example 3: ReLU Activation (from relu.c)

```c
static void tensor_relu_forward_f64(f64_t **inputs,
                                   gc_core_tensor_descriptor_t **inputs_desc,
                                   f64_t **outputs) {
    const i64_t N = inputs_desc[0]->size;
    const f64_t *X = inputs[0];
    f64_t *Y = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        Y[i] = (X[i] >= 0) ? X[i] : 0.0;  // max(0, x)
    }
}

static void tensor_relu_backward_f64(f64_t **inputs,
                                    f64_t **inputs_grad,
                                    gc_core_tensor_descriptor_t **inputs_desc,
                                    f64_t **outputs_grad) {
    const i64_t N = inputs_desc[0]->size;
    f64_t *dX = inputs_grad[0];
    const f64_t *X = inputs[0];
    const f64_t *dY = outputs_grad[0];

    if (dX != NULL) {
        for (i64_t i = 0; i < N; i++) {
            if (X[i] >= 0) {
                dX[i] += dY[i];  // Gradient flows through if x > 0
            }
            // else: gradient is zero (derivative of 0)
        }
    }
}
```

### Example 4: Sigmoid Activation (from sigmoid.c)

```c
static void tensor_sigmoid_forward_f64(f64_t **inputs,
                                      gc_core_tensor_descriptor_t **inputs_desc,
                                      f64_t **outputs) {
    const i64_t N = inputs_desc[0]->size;
    const f64_t *X = inputs[0];
    f64_t *Y = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        Y[i] = 1.0 / (1.0 + exp(-X[i]));  // σ(x) = 1/(1+e^(-x))
    }
}

static void tensor_sigmoid_backward_f64(f64_t **inputs,
                                       f64_t **inputs_grad,
                                       gc_core_tensor_descriptor_t **inputs_desc,
                                       f64_t **outputs,
                                       f64_t **outputs_grad) {
    const i64_t N = inputs_desc[0]->size;
    f64_t *dX = inputs_grad[0];
    const f64_t *Y = outputs[0];  // Use cached output
    const f64_t *dY = outputs_grad[0];

    if (dX != NULL) {
        for (i64_t i = 0; i < N; i++) {
            // σ'(x) = σ(x) * (1 - σ(x))
            dX[i] += Y[i] * (1.0 - Y[i]) * dY[i];
        }
    }
}
```

### Example 5: Softmax (from softmax.c)

```c
static void tensor_softmax_forward_f64(f64_t **inputs,
                                      gc_core_tensor_descriptor_t **inputs_desc,
                                      f64_t **outputs) {
    // Compute softmax over last dimension
    const i64_t N = inputs_desc[0]->size / inputs_desc[0]->dim[inputs_desc[0]->nb_dim - 1];
    const i64_t M = inputs_desc[0]->dim[inputs_desc[0]->nb_dim - 1];
    const f64_t *X = inputs[0];
    f64_t *Y = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        i64_t offset = i * M;

        // Find max for numerical stability
        f64_t max_val = X[offset];
        for (i64_t j = 1; j < M; j++) {
            if (X[offset + j] > max_val) {
                max_val = X[offset + j];
            }
        }

        // Compute exp(x - max) and sum
        f64_t sum = 0.0;
        for (i64_t j = 0; j < M; j++) {
            f64_t exp_val = exp(X[offset + j] - max_val);
            Y[offset + j] = exp_val;
            sum += exp_val;
        }

        // Normalize by sum
        for (i64_t j = 0; j < M; j++) {
            Y[offset + j] /= sum;
        }
    }
}
```

### Example 6: Matrix Multiplication (from matmul.c)

```c
static void tensor_matmul_f64(f64_t **inputs,
                             gc_core_tensor_descriptor_t **inputs_desc,
                             f64_t **outputs,
                             gc_core_tensor_descriptor_t **outputs_desc,
                             const gc_slot_t *constants) {
    i8_t dim_a = inputs_desc[0]->nb_dim;
    i8_t dim_b = inputs_desc[1]->nb_dim;
    i8_t dim_res = outputs_desc[0]->nb_dim;

    bool transpose_a = constants[0].b;
    bool transpose_b = constants[1].b;
    f64_t alpha = constants[2].f64;
    f64_t beta = constants[3].f64;

    if (dim_res == 2) {
        // Simple 2D matrix multiplication: C = alpha * A @ B + beta * C
        i64_t M = outputs_desc[0]->dim[0];
        i64_t N = outputs_desc[0]->dim[1];
        i64_t K = transpose_a ? inputs_desc[0]->dim[0] : inputs_desc[0]->dim[1];

        gc_compute_ops__matmul_f64(
            transpose_a, transpose_b, M, N, K, alpha,
            (f64_t *)inputs[0], gc_core_tensor_descriptor__leading_dim(inputs_desc[0]),
            (f64_t *)inputs[1], gc_core_tensor_descriptor__leading_dim(inputs_desc[1]),
            beta,
            (f64_t *)outputs[0], gc_core_tensor_descriptor__leading_dim(outputs_desc[0])
        );
    } else {
        // Batched matrix multiplication
        i64_t M = outputs_desc[0]->dim[dim_res - 2];
        i64_t N = outputs_desc[0]->dim[dim_res - 1];
        i64_t K = transpose_a ? inputs_desc[0]->dim[dim_a - 2] : inputs_desc[0]->dim[dim_a - 1];

        i64_t stepA = (dim_a == 2) ? 0 : M * K;
        i64_t stepB = (dim_b == 2) ? 0 : N * K;
        i64_t stepC = M * N;

        i64_t batch_count = (dim_a >= dim_b)
            ? gc_core_tensor_descriptor__matrix_count(inputs_desc[0])
            : gc_core_tensor_descriptor__matrix_count(inputs_desc[1]);

        gc_compute_ops__matmul_tensor_f64(
            transpose_a, transpose_b, M, N, K, alpha,
            (f64_t *)inputs[0], gc_core_tensor_descriptor__leading_dim(inputs_desc[0]),
            (f64_t *)inputs[1], gc_core_tensor_descriptor__leading_dim(inputs_desc[1]),
            beta,
            (f64_t *)outputs[0], gc_core_tensor_descriptor__leading_dim(outputs_desc[0]),
            stepA, stepB, stepC, batch_count
        );
    }
}
```

### Example 7: Dense Layer Compilation (from dense_layer.c)

```c
// Create weight matrix and matmul operation for dense layer
i64_t weight_dims[2] = {inputs, outputs};
u32_t weight_var_offset = gc_compute_engine__compile_add_varo(
    self, layer_name_offset, weight_name_offset,
    weight_dims, 2, inputs * outputs,
    weight_l1, weight_l2, init_w_p1, init_w_p2, init_w_p3, init_w_p4, init_w_p5,
    tensor_type, weight_initializer
);

// Create matmul operation: output = input @ weights
u32_t op_offset = gc_compute_ComputeEngine__create_op(self);
gc_compute_engine_op_t *op = (self->ops.data + op_offset);
op->params_offset = self->constants.size;
op->code = gc_compute_operation_code_matmul;

// Add operation parameters
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = input_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = weight_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = output_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.b = false});  // transpose_a
gc_compute_engine__compile_add_const(self, (gc_slot_t){.b = false});  // transpose_b
gc_compute_engine__compile_add_const(self, (gc_slot_t){.f64 = 1.0});  // alpha
gc_compute_engine__compile_add_const(self, (gc_slot_t){.f64 = 0.0});  // beta

op->nb_inputs = 2;
op->nb_outputs = 1;
op->nb_params = 4;
```

---

## Array API

Arrays are dynamic collections of elements.

```c
// Set element at offset
bool gc_core_array__set_slot(gc_core_array_t *self, u32_t offset, gc_slot_t value, gc_type_t type, gc_machine_t *ctx);

// Get element at offset
bool gc_core_array__get_slot(const gc_core_array_t *self, u32_t offset, gc_slot_t *value, gc_type_t *type);

// Add element to end
bool gc_core_array__add_slot(gc_core_array_t *self, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx);

// Remove all elements
void gc_core_array__remove_all(gc_core_array_t *self, gc_machine_t *ctx);

// Get array size
u32_t gc_core_array__size(const gc_core_array_t *self);
```

---

## Table API

Tables are 2D data structures with rows and columns.

```c
// Initialize with capacity
void gc_core_table__init(gc_core_table_t *table, u32_t capacity);

// Get dimensions
u32_t gc_core_table__nb_cols(gc_core_table_t *self);
u32_t gc_core_table__nb_rows(gc_core_table_t *self);

// Get/set cell value
bool gc_core_table__get_cell_value(const gc_core_table_t *self, i64_t row, i64_t col, gc_slot_t *value, gc_type_t *type);
void gc_core_table__set_cell_value(gc_core_table_t *self, i64_t row, i64_t col, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx);
```

---

## Buffer API

Buffers are dynamic byte arrays with helper functions for building strings and serializing data.

### Basic Operations

```c
// Create and destroy
gc_buffer_t *gc_buffer__create();
void gc_buffer__finalize(gc_buffer_t *self);

// Clear buffer
void gc_buffer__clear(gc_buffer_t *self);
void gc_buffer__clear_secure(gc_buffer_t *self);  // Overwrites with zeros

// Get buffer properties
char *gc_buffer__data(gc_buffer_t *self);
u64_t gc_buffer__size(gc_buffer_t *self);
u64_t gc_buffer__capacity(gc_buffer_t *self);
```

### Adding Data

```c
// Add strings
void gc_buffer__add_str(gc_buffer_t *self, const char *c, u32_t len);
void gc_buffer__add_cstr(gc_buffer_t *self, const char *c);  // Null-terminated
void gc_buffer__prepend_str(gc_buffer_t *self, const char *c, u32_t len);

// Add characters
void gc_buffer__add_char(gc_buffer_t *self, char c);
void gc_buffer__add_escaped_char(gc_buffer_t *self, char c);
void gc_buffer__add_escaped_str(gc_buffer_t *self, const char *str, u32_t str_len);

// Add numbers
void gc_buffer__add_u64(gc_buffer_t *self, u64_t i);
void gc_buffer__add_f64(gc_buffer_t *self, f64_t f);

// Add slots and objects
void gc_buffer__add_slot(gc_buffer_t *self, gc_slot_t slot, gc_type_t type, const gc_program_t *prog);
void gc_buffer__add_slot_as_json(gc_buffer_t *self, gc_slot_t slot, gc_type_t type, const gc_program_t *prog);
```

### Path and URL Operations

```c
// Path utilities
void gc_buffer__add_path_sep(gc_buffer_t *self);
void gc_buffer__add_cwd(gc_buffer_t *self);

// URL encoding
void gc_buffer__add_uri_encoded(gc_buffer_t *self, const char *c, u32_t len);
void gc_buffer__add_queryparams(gc_buffer_t *buf, const char *query_data, u32_t query_len);
```

---

## String API

```c
// Create string from buffer
gc_core_string_t *gc_core_string__create_from(const char *str, u64_t len);

// Create from buffer or use existing symbol
gc_core_string_t *gc_core_string__create_from_or_symbol(const gc_program_t *prog, const char *str, u64_t len);

// Create from gc_buffer_t
gc_core_string_t *gc_core_string__create_from_buffer(const gc_buffer_t *buf);

// Get string data
const char *gc_core_string__buffer(const gc_core_string_t *str);
u32_t gc_core_string__size(const gc_core_string_t *str);

// Check if string is a literal symbol
bool gc_core_string__is_lit(const gc_core_string_t *str);
```

---

## Memory Allocation

### GNU-style Allocator

```c
// GNU-style malloc/free (per-worker)
void *gc_gnu_malloc(size_t size);
void gc_gnu_free(void *ptr);
void *gc_gnu_calloc(size_t count, size_t size);
void *gc_gnu_realloc(void *ptr, size_t new_size);
size_t gc_gnu_alloc_size(void *ptr);

// Global allocator (shared across workers)
void *gc_global_gnu_malloc(size_t size);
void *gc_global_gnu_calloc(size_t count, size_t size);
void *gc_global_gnu_realloc(void *ptr, size_t new_size);
void gc_global_gnu_free(void *ptr);
```

---

## Program and Type System

### Program Access

```c
// Resolve symbols
u32_t gc_program__resolve_symbol(const gc_program_t *program, const char *str, u32_t len);
u32_t gc_program__resolve_module(const gc_program_t *program, u32_t mod_name_offset);
u32_t gc_program__resolve_type(const gc_program_t *prog, u32_t mod_offset, u32_t type_name_off);

// Get symbol by offset
gc_program_symbol_t *gc_program__get_symbol(const gc_program_t *program, u32_t symb_off);
u32_t gc_program__get_symbol_off(const gc_program_symbol_t *symb);
```

### Type Information

```c
// Get type information
gc_program_type_t *gc_program__get_program_type(const gc_program_t *prog, u32_t type_id);

// Type fields
gc_program_type_field_t *gc_program_type__get_field(const gc_program_type_t *program_type, u32_t field_offset);
gc_program_type_field_t *gc_program_type__get_field_by_key(const gc_program_type_t *program_type, u32_t key);
u32_t gc_program_type__nb_fields(const gc_program_type_t *program_type);

// Generic type information
u32_t gc_program_type__get_g1_type_id(const gc_program_type_t *program_type);
u32_t gc_program_type__get_g2_type_id(const gc_program_type_t *program_type);
```

### Type Configuration (for library authors)

```c
// Configure native type with finalizer
void gc_program_type__configure(gc_program_t *prog, u32_t type_id, u32_t header_bytes,
                                gc_object_type_native_finalize_t *native_finalize);
```

**Example: Type configuration (from algebra.c)**

```c
bool gc_lib_algebra__link_native(gc_program_t *prog, gc_program_library_t *lib) {
    gc_program_library__set_lib_hooks(lib, gc_lib_algebra__start, gc_lib_algebra__stop);

    // Configure custom types with finalizers
    gc_program_type__configure(prog, gc_compute_ComputeState,
                              sizeof(gc_compute_state_t),
                              gc_compute_ComputeState__finalize);
    gc_program_type__configure(prog, gc_compute_ComputeEngine,
                              sizeof(gc_compute_engine_t),
                              gc_compute_ComputeEngine__finalize);
    return true;
}
```

### Function Information

```c
// Get function count
u32_t gc_program__nb_functions(const gc_program_t *prog);

// Get function details
gc_program_function_t *gc_program__get_function(const gc_program_t *prog, u32_t fn_off);
bool gc_program_function__is_native_without_body(const gc_program_t *prog, u32_t fn_off);
void gc_program_function__set_body(const gc_program_t *prog, u32_t fn_off, gc_program_function_body_t *fn_body);

// Function parameters
u8_t gc_program_function__nb_params(const gc_program_function_t *fn);
bool gc_program_function__get_param_by_off(const gc_program_function_t *fn, u32_t offset, gc_function_param_t *out);
```

### Linking Native Functions

```c
// Link module function
bool gc_program__link_mod_fn(const gc_program_t *prg, u32_t module_id,
                            gc_program_function_body_t *function, u32_t fn_name_symbol);

// Link type method
bool gc_program__link_type_fn(const gc_program_t *prg, u32_t type_id,
                             gc_program_function_body_t *function, u32_t fn_name_symbol);
```

### Library Hooks

```c
// Set library lifecycle hooks
void gc_program_library__set_lib_hooks(gc_program_library_t *lib,
                                       gc_hook_function_t *lib_start_hook,
                                       gc_hook_function_t *lib_stop_hook);

// Set worker lifecycle hooks
void gc_program_library__set_worker_hooks(gc_program_library_t *lib,
                                         gc_hook_function_t *worker_start_hook,
                                         gc_hook_function_t *worker_stop_hook);
```

---

## Host and Task Management

### Task Management

```c
typedef enum {
    gc_task_status_empty = 0,
    gc_task_status_waiting = 1,
    gc_task_status_running = 2,
    gc_task_status_await = 3,
    gc_task_status_cancelled = 4,
    gc_task_status_error = 5,
    gc_task_status_ended = 6,
    gc_task_status_ended_with_errors = 7,
} gc_task_status_t;

// Get global host
gc_host_t *gc_host__get_global();

// Spawn task without arguments
bool gc_host__spawn_task(gc_host_t *self, u32_t fn_off, u32_t user_id,
                        u64_t roles_flags, i64_t *created_task_id);

// Get task status
bool gc_host__get_task_status(gc_host_t *self, u32_t task_id, gc_task_status_t *status);

// Cancel task
bool gc_host__cancel_task(gc_host_t *self, u32_t task_id);
```

---

## HTTP Client

### HTTP Request Structure

```c
typedef enum gc_http_method {
    gc_http_method_get = 0,
    gc_http_method_post = 2,
    gc_http_method_put = 3,
    gc_http_method_delete = 4,
} gc_http_method_t;

typedef struct gc_http_request {
    gc_http_method_t method;
    bool use_ssl;
    i32_t port;
    struct { const char *data; u32_t len; } host;
    struct { const char *data; u32_t len; } path;
    struct { const char *data; u32_t len; } query;
    gc_buffer_t *headers;
    u16_t status_code;
} gc_http_request_t;
```

### HTTP Operations

```c
// Parse URL into request
void gc_http_request__parse(gc_http_request_t *req, const char *url, u32_t url_len);

// Execute HTTP request
bool gc_http_request__call(gc_http_request_t *req, bool *json_detected,
                          gc_buffer_t *buf, const gc_program_t *prog);

// Add headers
u32_t gc_http_request__add_header(gc_http_request_t *req,
                                  const char *name, u32_t name_len,
                                  const char *value, u32_t value_len);

// Clear and finalize
void gc_http_request__clear(gc_http_request_t *req);
void gc_http_request__finalize(gc_http_request_t *req);
```

---

## Cryptography

### SHA-256

```c
#define gc_crypto_sha256_len 32

typedef struct gc_crypto_sha256 {
    union {
        u32_t u32[8];
        unsigned char u8[gc_crypto_sha256_len];
    } u;
} gc_crypto_sha256_t;

// Compute SHA-256 hash
void gc_crypto_sha256(gc_crypto_sha256_t *sha, const void *p, size_t size);
```

### HMAC-SHA256

```c
// Compute HMAC-SHA256
void gc_crypto_hmac_sha256(gc_crypto_hmac_sha256_ctx_t *ctx,
                          gc_crypto_hmac_sha256_t *hmac,
                          const void *k, size_t ksize,
                          const void *d, size_t dsize);
```

### Hex Encoding/Decoding

```c
// Convert hex string to binary
bool gc_common__hex2bin(char *dest, const char *src, u32_t len);
u32_t gc_common__hex2bin_len(u32_t len);

// Convert binary to hex string
void gc_common__bin2hex(char *dest, const char *src, u32_t len);
u32_t gc_common__bin2hex_len(u32_t len);
```

---

## Utility Functions

### Parsing

```c
// Parse unsigned integer
u64_t gc_common__parse_number(const char *str, u32_t *str_len);

// Parse signed integer
i64_t gc_common__parse_sign_number(const char *str, u32_t *str_len);

// Parse ISO 8601 date
bool gc_common__parse_date_iso8601(char *data, u32_t data_len, i64_t *epoch_utc);
```

### Time

```c
// Get current time in microseconds
i64_t gc_common__current_us();
```

### JSON Parsing

```c
// Parse JSON string into a value
bool gc_json__parse(gc_machine_t *ctx, char *str, u32_t str_len,
                   gc_slot_t *result, gc_type_t *result_type, u32_t type_d);
```

---

## I/O Operations

```c
// Open file read/write
i32_t gc_io_file__open_rdwr(gc_core_string_t *path, gc_machine_t *ctx);

// Open file read-only
i32_t gc_io_file__open_read(gc_core_string_t *path, gc_machine_t *ctx);

// Sync file to disk
void gc_io_file__sync(i32_t fp);
```

---

## Common Patterns

### Error Handling Pattern

```c
void safe_operation(gc_machine_t *ctx) {
    // Validate parameters
    if (gc_machine__get_param_nb(ctx) < 2) {
        gc_machine__set_runtime_error(ctx, "Expected 2 parameters");
        return;
    }

    // Get parameters with type checking
    gc_slot_t param0 = gc_machine__get_param(ctx, 0);
    gc_type_t type0 = gc_machine__get_param_type(ctx, 0);

    if (type0 != gc_type_int) {
        gc_machine__set_runtime_error(ctx, "Parameter 0 must be int");
        return;
    }

    // Perform operation...
    // Check for errors
    if (gc_machine__error(ctx)) {
        return;  // Error already set
    }

    // Set result
    gc_machine__set_result(ctx, result, result_type);
}
```

### Tensor Operation Pattern (from sigmoid.c)

```c
static void gc_compute_op_sigmoid__fwd_f64(gc_unused gc_compute_engine_t *engine,
                                           f64_t **inputs,
                                           gc_core_tensor_descriptor_t **inputs_desc,
                                           f64_t **outputs,
                                           gc_unused gc_core_tensor_descriptor_t **outputs_desc,
                                           gc_unused const gc_slot_t *constants,
                                           gc_unused char *workspace) {
    const i64_t N = inputs_desc[0]->size;
    const f64_t *X = inputs[0];
    f64_t *Y = outputs[0];

    for (i64_t i = 0; i < N; i++) {
        Y[i] = 1.0 / (1.0 + exp(-X[i]));
    }
}
```

### Computing Engine Pattern (from dense_layer.c)

```c
// Create operation
u32_t new_op_offset = gc_compute_ComputeEngine__create_op(self);
gc_compute_engine_op_t *op = (self->ops.data + new_op_offset);
op->params_offset = self->constants.size;
op->code = gc_compute_operation_code_matmul;

// Add parameters as constants
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = input_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = weight_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.u32 = output_var_offset});
gc_compute_engine__compile_add_const(self, (gc_slot_t){.b = false});  // transpose_a
gc_compute_engine__compile_add_const(self, (gc_slot_t){.b = false});  // transpose_b
gc_compute_engine__compile_add_const(self, (gc_slot_t){.f64 = 1.0});  // alpha
gc_compute_engine__compile_add_const(self, (gc_slot_t){.f64 = 0.0});  // beta

op->nb_inputs = 2;
op->nb_outputs = 1;
op->nb_params = 4;
```
