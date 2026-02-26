# GreyCat Native Plugin Development Guide

Practical guide for building GreyCat plugins in C

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [Project Structure](#project-structure)
3. [CMake Build Configuration](#cmake-build-configuration)
4. [GCL Type Definitions](#gcl-type-definitions)
5. [Native Function Implementation](#native-function-implementation)
6. [Code Generation (nativegen)](#code-generation-nativegen)
7. [Library Lifecycle Hooks](#library-lifecycle-hooks)
8. [Custom Type Configuration](#custom-type-configuration)
9. [Global State Management](#global-state-management)
10. [Memory Management Patterns](#memory-management-patterns)
11. [Working with Parameters](#working-with-parameters)
12. [Returning Results](#returning-results)
13. [Error Handling](#error-handling)
14. [Working with Strings](#working-with-strings)
15. [Working with Arrays](#working-with-arrays)
16. [Working with Tensors](#working-with-tensors)
17. [Working with Buffers](#working-with-buffers)
18. [Enum Mapping Pattern](#enum-mapping-pattern)
19. [Object Boxing Pattern](#object-boxing-pattern)
20. [Thread Safety](#thread-safety)
21. [Complete Example: Building a Plugin](#complete-example-building-a-plugin)

---

## Plugin Architecture

A GreyCat native plugin consists of:

1. **GCL definitions** (`.gcl`) - Type and function declarations in GreyCat language
2. **C implementation** (`.c/.h`) - Native function bodies
3. **Code generation** (`nativegen.c/h`) - Bridges GCL types to C symbols
4. **Build system** (`CMakeLists.txt`) - Produces a `.gclib` shared library

The flow:
```
.gcl types/functions → nativegen resolves symbols → .c implements functions → .gclib output
```

## Project Structure

```
my_plugin/
├── CMakeLists.txt              # Build configuration
├── include/
│   ├── nativegen.h             # Generated: extern declarations for all type/field offsets
│   └── my_plugin/
│       └── my_module.h         # Module-specific headers
├── src/
│   ├── nativegen.c             # Generated: symbol resolution and linking
│   ├── my_module.c             # Native function implementations
│   └── my_module_helpers.c     # Additional implementation files
├── example/
│   └── example.gcl             # Usage examples
└── test/
    └── test.gcl                # Tests
```

## CMake Build Configuration

```cmake
cmake_minimum_required(VERSION 3.11)
project(my_plugin VERSION 0.0.0 LANGUAGES C)

# Source files
set(SOURCES
    src/nativegen.c
    src/my_module.c
)

# Create shared library module
add_library(${PROJECT_NAME} MODULE ${SOURCES})

# Include paths
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PRIVATE ${GREYCAT_INCLUDE})

# Link greycat library
if (LINUX)
    target_link_libraries(${PROJECT_NAME} ${GREYCAT_LIB})
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${GREYCAT_LIB})
elseif (WIN32)
    target_link_libraries(${PROJECT_NAME} ${GREYCAT_LIB})
endif ()

# Output as .gclib (not .so/.dylib)
set_target_properties(${PROJECT_NAME} PROPERTIES
    NO_SONAME ON
    PREFIX ""
    SUFFIX ""
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ${PROJECT_NAME}.gclib
)
```

## GCL Type Definitions

Define your types and native function signatures in GCL:

```gcl
// my_module.gcl
type ModelParams {
    n_gpu_layers: int;
    use_mmap: bool?;
    n_threads: int?;
}

type Model {
    // Native functions (implemented in C, no body in GCL)
    static native fn load(id: String, path: String, params: ModelParams?): Model;
    native fn embed(text: String, tensor_type: TensorType): Tensor;
    native fn tokenize(text: String): Array<int>;
    static native fn free(id: String);
}
```

## Native Function Implementation

### Function Signature Convention

All native functions follow the same signature:

```c
void gc_<module>_<Type>__<method_name>(gc_machine_t *ctx)
```

- Double underscore `__` separates type from method
- Always takes `gc_machine_t *ctx` as sole parameter
- Returns void (result set via `gc_machine__set_result`)

### Static vs Instance Methods

```c
// Static method: gc_module_Type__method(ctx)
// Parameters start at offset 0
void gc_mymod_Model__load(gc_machine_t *ctx) {
    gc_slot_t param0 = gc_machine__get_param(ctx, 0);  // first param
}

// Instance method: gc_module_Type__method(ctx)
// 'this' is available via gc_machine__this()
void gc_mymod_Model__embed(gc_machine_t *ctx) {
    gc_slot_t self = gc_machine__this(ctx);
    gc_object_t *this_obj = self.object;

    gc_slot_t param0 = gc_machine__get_param(ctx, 0);  // first param after self
}
```

### Module-Level Functions

For functions that are not methods on a type (module-level functions in GCL), use `gc_program__link_mod_fn` instead of `gc_program__link_type_fn`:

```c
// GCL declaration:   native fn my_module_fn(x: int): int;
// C implementation:  void gc_mymod__my_module_fn(gc_machine_t *ctx) { ... }
// Linking:           gc_program__link_mod_fn(prg, mymod, fn_ptr, symbol)
```

## Code Generation (nativegen)

### nativegen.h - Type and Field Offset Declarations

```c
#pragma once
#include <greycat.h>

// Module-level type IDs (resolved at link time)
extern u32_t gc_mymod_Model;
extern u32_t gc_mymod_ModelParams;

// Field offsets for ModelParams
#define gc_mymod_ModelParams_n_gpu_layers 0
#define gc_mymod_ModelParams_use_mmap 1
#define gc_mymod_ModelParams_n_threads 2

// Parameter access macros for Model::load(id, path, params)
#define gc_mymod_Model_load_id(ctx)         (gc_machine__get_param(ctx, 0))
#define gc_mymod_Model_load_id_type(ctx)    (gc_machine__get_param_type(ctx, 0))
#define gc_mymod_Model_load_path(ctx)       (gc_machine__get_param(ctx, 1))
#define gc_mymod_Model_load_path_type(ctx)  (gc_machine__get_param_type(ctx, 1))
#define gc_mymod_Model_load_params(ctx)     (gc_machine__get_param(ctx, 2))
#define gc_mymod_Model_load_params_type(ctx)(gc_machine__get_param_type(ctx, 2))

// Declare native function implementations
extern void gc_mymod_Model__load(gc_machine_t *ctx);
extern void gc_mymod_Model__embed(gc_machine_t *ctx);
extern void gc_mymod_Model__tokenize(gc_machine_t *ctx);
extern void gc_mymod_Model__free(gc_machine_t *ctx);
```

### nativegen.c - Symbol Resolution and Linking

```c
#include "nativegen.h"

// Type IDs (initialized during linking)
u32_t gc_mymod_Model = 0;
u32_t gc_mymod_ModelParams = 0;

// Forward declare the native link function
extern bool gc_lib_my_plugin__link_native(gc_program_t *prog, gc_program_library_t *lib);

// Main link entry point (called by GreyCat runtime)
bool gc_lib_my_plugin__link(gc_program_t *prg, gc_program_library_t *lib) {
    bool res = true;

    // Resolve dependency modules
    u32_t core = gc_program__resolve_module(prg,
        gc_program__resolve_symbol(prg, "core", 4));

    // Resolve well-known types from core
    gc_core_any = gc_program__resolve_type(prg, core,
        gc_program__resolve_symbol(prg, "any", 3));
    res &= (gc_core_any != 0);

    gc_core_String = gc_program__resolve_type(prg, core,
        gc_program__resolve_symbol(prg, "String", 6));
    res &= (gc_core_String != 0);

    gc_core_Array = gc_program__resolve_type(prg, core,
        gc_program__resolve_symbol(prg, "Array", 5));
    res &= (gc_core_Array != 0);

    gc_core_Tensor = gc_program__resolve_type(prg, core,
        gc_program__resolve_symbol(prg, "Tensor", 6));
    res &= (gc_core_Tensor != 0);

    // Resolve this plugin's module
    u32_t mymod = gc_program__resolve_module(prg,
        gc_program__resolve_symbol(prg, "mymod", 5));
    res &= (mymod != 0);

    // Resolve this plugin's types
    gc_mymod_Model = gc_program__resolve_type(prg, mymod,
        gc_program__resolve_symbol(prg, "Model", 5));
    res &= (gc_mymod_Model != 0);

    gc_mymod_ModelParams = gc_program__resolve_type(prg, mymod,
        gc_program__resolve_symbol(prg, "ModelParams", 11));
    res &= (gc_mymod_ModelParams != 0);

    // Link native function implementations
    res &= gc_program__link_type_fn(prg, gc_mymod_Model,
        (gc_program_function_body_t *)gc_mymod_Model__load,
        gc_program__resolve_symbol(prg, "load", 4));

    res &= gc_program__link_type_fn(prg, gc_mymod_Model,
        (gc_program_function_body_t *)gc_mymod_Model__embed,
        gc_program__resolve_symbol(prg, "embed", 5));

    res &= gc_program__link_type_fn(prg, gc_mymod_Model,
        (gc_program_function_body_t *)gc_mymod_Model__tokenize,
        gc_program__resolve_symbol(prg, "tokenize", 8));

    res &= gc_program__link_type_fn(prg, gc_mymod_Model,
        (gc_program_function_body_t *)gc_mymod_Model__free,
        gc_program__resolve_symbol(prg, "free", 4));

    // For module-level functions (not type methods), use gc_program__link_mod_fn:
    // res &= gc_program__link_mod_fn(prg, mymod,
    //     (gc_program_function_body_t *)gc_mymod__my_module_fn,
    //     gc_program__resolve_symbol(prg, "my_module_fn", 12));

    if (!res) {
        return false;
    }

    // Call native setup (type configuration, hooks)
    return gc_lib_my_plugin__link_native(prg, lib);
}
```

## Library Lifecycle Hooks

### Hook Types

```c
// Function type (not a pointer typedef) — defined in program.h
typedef bool(gc_hook_function_t)(gc_program_library_t *lib, gc_program_t *prog, void **user_data);
```

### Setting Hooks

```c
bool gc_lib_my_plugin__link_native(gc_program_t *prog, gc_program_library_t *lib) {
    // Set library-level hooks (called once for the process)
    gc_program_library__set_lib_hooks(lib, lib_start, lib_stop);

    // Set worker-level hooks (called for each worker thread)
    gc_program_library__set_worker_hooks(lib, worker_start, worker_stop);

    // Configure custom types (see next section)
    gc_program_type__configure(prog, gc_mymod_Model,
                               sizeof(gc_mymod_model_t),
                               gc_mymod_Model__finalize);

    return true;
}
```

### Library Start/Stop (Global)

```c
static bool lib_start(gc_unused gc_program_library_t *lib,
                       gc_unused gc_program_t *prog,
                       gc_unused void **user_data) {
    // Initialize global state (called once on startup)
    global_state = gc_global_gnu_malloc(sizeof(my_global_state_t));
    memset(global_state, 0, sizeof(my_global_state_t));
    pthread_mutex_init(&global_state->lock, NULL);

    // Initialize third-party libraries
    some_library_init();

    return true;  // false = abort startup
}

static bool lib_stop(gc_unused gc_program_library_t *lib,
                      gc_unused gc_program_t *prog,
                      gc_unused void **user_data) {
    // Cleanup global state (called once on shutdown)
    some_library_cleanup();
    pthread_mutex_destroy(&global_state->lock);
    gc_global_gnu_free(global_state);

    return true;
}
```

### Worker Start/Stop (Per-Thread)

```c
static bool worker_start(gc_unused gc_program_library_t *lib,
                          gc_unused gc_program_t *prog,
                          gc_unused void **user_data) {
    // Initialize per-worker state (thread-local)
    return true;
}

static bool worker_stop(gc_unused gc_program_library_t *lib,
                         gc_unused gc_program_t *prog,
                         gc_unused void **user_data) {
    // Cleanup per-worker state
    return true;
}
```

## Custom Type Configuration

When your GCL type wraps a C struct with extra data (e.g., a pointer to a third-party library object), use `gc_program_type__configure`:

### Define the C Struct

```c
typedef struct {
    gc_object_t header;         // REQUIRED: must be first field
    struct some_lib_handle *handle;  // Your custom data
    const char *cached_name;
    u32_t ref_count;
} gc_mymod_model_t;
```

### Configure the Type

```c
gc_program_type__configure(prog, gc_mymod_Model,
                           sizeof(gc_mymod_model_t),      // Total struct size
                           gc_mymod_Model__finalize);      // Cleanup callback
```

### Implement Finalizer

```c
static void gc_mymod_Model__finalize(gc_object_t *self, gc_unused gc_machine_t *ctx) {
    gc_mymod_model_t *model = (gc_mymod_model_t *)self;
    if (model->handle != NULL) {
        some_lib_free(model->handle);
        model->handle = NULL;
    }
    if (model->cached_name != NULL) {
        gc_global_gnu_free((void *)model->cached_name);
        model->cached_name = NULL;
    }
}
```

## Global State Management

### Thread-Safe Global Store Pattern

```c
typedef struct {
    char *id;
    struct some_handle *handle;
} my_store_entry_t;

typedef struct {
    pthread_mutex_t lock;
    my_store_entry_t *entries;
    u32_t nb_entries;
} my_global_store_t;

my_global_store_t *global_store;  // Initialized in lib_start

// Find by ID (called with lock held)
static i32_t find_by_id(const gc_string_t *id) {
    for (u32_t i = 0; i < global_store->nb_entries; i++) {
        if (strcmp(global_store->entries[i].id, id->buffer) == 0) {
            return (i32_t)i;
        }
    }
    return -1;
}

// Usage in native function
void gc_mymod_Model__load(gc_machine_t *ctx) {
    pthread_mutex_lock(&global_store->lock);

    gc_string_t *id = (gc_string_t *)gc_mymod_Model_load_id(ctx).object;
    i32_t idx = find_by_id(id);

    if (idx < 0) {
        // Not found, create new entry...
        // Grow array using gc_global_gnu_malloc + memcpy + gc_global_gnu_free
    }

    // Return result...
    pthread_mutex_unlock(&global_store->lock);
}
```

## Memory Management Patterns

### Temporary Allocations (within a function)

```c
void my_function(gc_machine_t *ctx) {
    // Per-worker allocation (no mutex needed)
    char *temp = gc_gnu_malloc(1024);

    // ... use temp ...

    gc_gnu_free(temp);
}
```

### Persistent Allocations (global state)

```c
// In lib_start or native functions (use global allocator + mutex)
pthread_mutex_lock(&global_store->lock);
char *persistent = gc_global_gnu_malloc(size);
// Store in global state...
pthread_mutex_unlock(&global_store->lock);

// In lib_stop (cleanup)
gc_global_gnu_free(persistent);
```

### Buffer Reuse Pattern

```c
void my_function(gc_machine_t *ctx) {
    // Reuse the machine's scratch buffer (avoids allocation)
    gc_buffer_t *buf = gc_machine__get_buffer(ctx);
    gc_buffer__clear(buf);

    // Ensure capacity for raw data
    gc_buffer__prepare(buf, needed_bytes);

    // Use buf->data directly
    memcpy(buf->data, source, needed_bytes);
}
```

### Dynamic Array Growth Pattern

```c
// Grow a global array by 1 element
my_store_entry_t *new_entries = gc_global_gnu_malloc(
    (global_store->nb_entries + 1) * sizeof(my_store_entry_t));

// Copy existing entries
memcpy(new_entries, global_store->entries,
       global_store->nb_entries * sizeof(my_store_entry_t));

// Initialize new entry
my_store_entry_t *new_entry = new_entries + global_store->nb_entries;
new_entry->id = gc_global_gnu_malloc(id->size + 1);
memcpy(new_entry->id, id->buffer, id->size);
new_entry->id[id->size] = '\0';

// Swap and free old array
if (global_store->entries != NULL) {
    gc_global_gnu_free(global_store->entries);
}
global_store->entries = new_entries;
global_store->nb_entries += 1;
```

## Working with Parameters

### Type Checking Parameters

```c
void my_function(gc_machine_t *ctx) {
    gc_type_t type0 = gc_machine__get_param_type(ctx, 0);

    if (type0 == gc_type_null) {
        // Parameter is null (nullable parameter not provided)
        return;
    }

    if (type0 == gc_type_int) {
        i64_t value = gc_machine__get_param(ctx, 0).i64;
    } else if (type0 == gc_type_float) {
        f64_t value = gc_machine__get_param(ctx, 0).f64;
    } else if (type0 == gc_type_bool) {
        bool value = gc_machine__get_param(ctx, 0).b;
    } else if (type0 == gc_type_object) {
        gc_object_t *obj = gc_machine__get_param(ctx, 0).object;
    } else if (type0 == gc_type_static_field) {
        // Enum value: .tu32.right is the enum variant index
        u32_t enum_val = gc_machine__get_param(ctx, 0).tu32.right;
    }

    // Check object type with inheritance support
    if (type0 == gc_type_object) {
        gc_object_t *obj = gc_machine__get_param(ctx, 0).object;
        if (gc_object__is_instance_of(obj, gc_mymod_Model, ctx)) {
            // obj is an instance of Model or a subtype
        }
    }
}
```

### Reading Object Fields (Optional/Nullable Parameters)

For nullable fields (declared with `?` in GCL), use the nullable bitset macros from `object.h`:

```c
// Nullable bitset macros (from object.h):
//   gc_object__is_not_null(bitset, offset)  — returns 1 if field is set, 0 if null
//   gc_object__set_not_null(bitset, offset) — mark field as non-null
//   gc_object__set_null(bitset, offset)     — mark field as null

void apply_params(gc_object_t *params, gc_machine_t *ctx) {
    if (params == NULL) return;

    const gc_program_t *prog = gc_machine__program(ctx);
    gc_program_type_t *type = gc_program__get_program_type(prog, params->type_id);
    gc_object_data_t data = gc_object__segments(params, type);

    gc_type_t field_type;
    gc_slot_t field_val;

    // Non-nullable field: always read directly
    field_val = gc_object__get_at(params, gc_mymod_ModelParams_n_gpu_layers,
                                   &field_type, ctx);
    if (field_type == gc_type_int) {
        i64_t n_gpu_layers = field_val.i64;
        // Use value...
    }

    // Nullable field: check bitset before reading
    if (gc_object__is_not_null(data.nullable_bitset, gc_mymod_ModelParams_use_mmap)) {
        field_val = gc_object__get_at(params, gc_mymod_ModelParams_use_mmap,
                                       &field_type, ctx);
        if (field_type == gc_type_bool) {
            bool use_mmap = field_val.b;
            // Use value...
        }
    }

    // Alternative: for non-nullable fields, just check the type after reading
    field_val = gc_object__get_at(params, gc_mymod_ModelParams_n_threads,
                                   &field_type, ctx);
    if (field_type != gc_type_null) {
        i64_t n_threads = field_val.i64;
        // Use value...
    }
}
```

## Returning Results

### Primitive Results

```c
// Return int
gc_machine__set_result(ctx, (gc_slot_t){.i64 = 42}, gc_type_int);

// Return float
gc_machine__set_result(ctx, (gc_slot_t){.f64 = 3.14}, gc_type_float);

// Return bool
gc_machine__set_result(ctx, (gc_slot_t){.b = true}, gc_type_bool);

// Return null
gc_machine__set_result(ctx, (gc_slot_t){0}, gc_type_null);
```

### Object Results (IMPORTANT: un_mark pattern)

```c
// Create and return an object
gc_object_t *result = gc_machine__create_object(ctx, gc_mymod_Model);

// Set fields...
gc_object__set_at(result, field_offset, value, value_type, ctx);

// CRITICAL: un_mark before returning (prevents premature GC)
gc_machine__set_result(ctx, (gc_slot_t){.object = result}, gc_type_object);
gc_object__un_mark(result, ctx);
```

### Enum Results

```c
// Return an enum value (e.g., MyEnum::variant2 where variant2 is index 1)
gc_slot_t result = {.tu32 = {.left = 0, .right = 1}};
gc_machine__set_result(ctx, result, gc_type_static_field);
```

## Error Handling

```c
void my_function(gc_machine_t *ctx) {
    // Validate parameter count
    if (gc_machine__get_param_nb(ctx) < 2) {
        gc_machine__set_runtime_error(ctx, "Expected at least 2 parameters");
        return;
    }

    // Check for null
    if (gc_machine__get_param_type(ctx, 0) == gc_type_null) {
        gc_machine__set_runtime_error(ctx, "Parameter 'name' cannot be null");
        return;
    }

    // Check system errors
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        gc_machine__set_runtime_error_syserr(ctx);  // Uses errno
        return;
    }

    // Check for propagated errors
    if (gc_machine__error(ctx)) {
        return;  // Error already set by a called function
    }
}
```

## Conditional Logging

Use `gc_machine__log_level` to check the machine's verbosity level before performing expensive logging operations:

```c
void my_function(gc_machine_t *ctx) {
    // gc_log_level_t: none=0, error=1, warn=2, info=3, perf=4, trace=5
    if (gc_machine__log_level(ctx) >= gc_log_level_trace) {
        // Only build expensive debug strings when trace logging is active
        gc_buffer_t *buf = gc_machine__get_buffer(ctx);
        gc_buffer__clear(buf);
        gc_buffer__add_cstr(buf, "Processing input of size ");
        gc_buffer__add_u64(buf, input_size);
        // ... output to log ...
    }
}
```

## Working with Strings

### Reading String Parameters

```c
void my_function(gc_machine_t *ctx) {
    gc_string_t *str = (gc_string_t *)gc_machine__get_param(ctx, 0).object;

    // Access string data
    const char *data = str->buffer;
    u32_t length = str->size;

    // Use with C functions (string is NOT null-terminated by default)
    // Option 1: Use length-aware functions
    memcpy(dest, str->buffer, str->size);

    // Option 2: Create null-terminated copy
    char *cstr = gc_gnu_malloc(str->size + 1);
    memcpy(cstr, str->buffer, str->size);
    cstr[str->size] = '\0';
    // ... use cstr ...
    gc_gnu_free(cstr);
}
```

### Creating String Results

```c
// From buffer with known length
gc_string_t *result = gc_string__create_from(data, length);
gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *)result}, gc_type_object);
gc_object__un_mark((gc_object_t *)result, ctx);

// From gc_buffer_t
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__clear(buf);
gc_buffer__add_cstr(buf, "Hello ");
gc_buffer__add_u64(buf, 42);
gc_string_t *result = gc_string__create_from(buf->data, buf->size);
```

## Working with Arrays

### Creating and Populating Arrays

```c
void create_token_array(gc_machine_t *ctx, int *tokens, int count) {
    gc_array_t *arr = (gc_array_t *)gc_machine__create_object(ctx, gc_core_Array);

    for (int i = 0; i < count; i++) {
        gc_array__add_slot(arr, (gc_slot_t){.i64 = tokens[i]}, gc_type_int, ctx);
    }

    gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *)arr}, gc_type_object);
    gc_object__un_mark((gc_object_t *)arr, ctx);
}
```

### Reading Array Parameters

```c
void process_array(gc_machine_t *ctx) {
    gc_array_t *arr = (gc_array_t *)gc_machine__get_param(ctx, 0).object;

    for (u32_t i = 0; i < arr->size; i++) {
        gc_slot_t slot;
        gc_type_t slot_type;
        gc_array__get_slot(arr, i, &slot, &slot_type);

        if (slot_type == gc_type_int) {
            i64_t value = slot.i64;
            // Process value...
        }
    }
}
```

## Working with Tensors

### Creating and Returning Tensors

```c
void create_embedding(gc_machine_t *ctx, float *data, int n_embd) {
    gc_core_tensor_t *tensor = gc_core_tensor__create(ctx);
    gc_core_tensor__init_1d(tensor, n_embd, gc_core_TensorType_f32, ctx);

    // Copy data into tensor
    for (i32_t i = 0; i < n_embd; i++) {
        gc_core_tensor__set_1d_f32(tensor, i, data[i], ctx);
    }

    gc_machine__set_result(ctx,
        (gc_slot_t){.object = (gc_object_t *)tensor}, gc_type_object);
    gc_object__un_mark((gc_object_t *)tensor, ctx);
}
```

### Direct Memory Access (High Performance)

```c
void fast_tensor_fill(gc_machine_t *ctx) {
    gc_core_tensor_t *tensor = gc_core_tensor__create(ctx);
    gc_core_tensor__init_1d(tensor, 1000, gc_core_TensorType_f64, ctx);

    // Get raw data pointer for fast access
    f64_t *data = (f64_t *)gc_core_tensor__get_data(tensor);
    gc_core_tensor_descriptor_t *desc = gc_core_tensor__get_descriptor(tensor);

    for (i64_t i = 0; i < desc->size; i++) {
        data[i] = (f64_t)i * 0.001;
    }
}
```

## Working with Buffers

### Tokenization Retry Pattern

When a C library uses negative return values to indicate "buffer too small":

```c
void tokenize_text(gc_machine_t *ctx) {
    gc_string_t *text = (gc_string_t *)gc_machine__get_param(ctx, 0).object;
    gc_buffer_t *buf = gc_machine__get_buffer(ctx);

    // First attempt with reasonable buffer
    gc_buffer__prepare(buf, 2048 * sizeof(int));
    int n_tokens = external_tokenize(text->buffer, text->size,
                                      (int *)buf->data, 2048);

    if (n_tokens < 0) {
        // Negative means buffer too small, absolute value is needed size
        gc_buffer__prepare(buf, (-n_tokens) * sizeof(int));
        n_tokens = external_tokenize(text->buffer, text->size,
                                      (int *)buf->data, -n_tokens);

        if (n_tokens < 0) {
            gc_machine__set_runtime_error(ctx, "Tokenization failed");
            return;
        }
    }

    // Build result array from buffer
    gc_array_t *arr = (gc_array_t *)gc_machine__create_object(ctx, gc_core_Array);
    int *tokens = (int *)buf->data;
    for (int i = 0; i < n_tokens; i++) {
        gc_array__add_slot(arr, (gc_slot_t){.i64 = tokens[i]}, gc_type_int, ctx);
    }

    gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *)arr}, gc_type_object);
    gc_object__un_mark((gc_object_t *)arr, ctx);
}
```

## Enum Mapping Pattern

When your GCL enum maps to a C library's enum with potentially different value ordering:

```c
// GCL enum values (auto-assigned 0, 1, 2, ...)
#define gc_mymod_PoolingType_unspecified 0
#define gc_mymod_PoolingType_none       1
#define gc_mymod_PoolingType_mean       2
#define gc_mymod_PoolingType_cls        3

// C library enum may have different values
enum lib_pooling_type {
    LIB_POOLING_UNSPECIFIED = -1,
    LIB_POOLING_NONE = 0,
    LIB_POOLING_MEAN = 1,
    LIB_POOLING_CLS = 2,
};

// Always use explicit mapping (never assume values match!)
static enum lib_pooling_type gc_to_lib_pooling_type(u32_t gc_value) {
    switch (gc_value) {
    case gc_mymod_PoolingType_unspecified: return LIB_POOLING_UNSPECIFIED;
    case gc_mymod_PoolingType_none:        return LIB_POOLING_NONE;
    case gc_mymod_PoolingType_mean:        return LIB_POOLING_MEAN;
    case gc_mymod_PoolingType_cls:         return LIB_POOLING_CLS;
    default:                               return LIB_POOLING_UNSPECIFIED;
    }
}

// Apply enum parameter
void apply_pooling(gc_object_t *params, struct lib_config *cfg, gc_machine_t *ctx) {
    gc_type_t type;
    gc_slot_t val = gc_object__get_at(params, FIELD_OFFSET, &type, ctx);
    if (type == gc_type_static_field) {
        cfg->pooling_type = gc_to_lib_pooling_type(val.tu32.right);
    }
}
```

## Object Boxing Pattern

Wrapping a C library handle inside a GreyCat object:

```c
typedef struct {
    gc_object_t header;       // Must be first
    struct lib_handle *handle; // C library pointer
    u32_t store_index;        // Index into global cache
} gc_mymod_model_t;

// Box: create GreyCat object wrapping C handle
gc_object_t *box_model(u32_t store_idx, gc_machine_t *ctx) {
    gc_object_t *obj = gc_machine__create_object(ctx, gc_mymod_Model);
    gc_mymod_model_t *model = (gc_mymod_model_t *)obj;
    model->handle = global_store->entries[store_idx].handle;
    model->store_index = store_idx;
    return obj;
}

// Unbox: extract C handle from GreyCat object
void use_model(gc_machine_t *ctx) {
    gc_mymod_model_t *self = (gc_mymod_model_t *)gc_machine__this(ctx).object;
    struct lib_handle *handle = self->handle;
    // Use handle...
}
```

## Thread Safety

### Rules

1. **Per-worker allocators** (`gc_gnu_malloc`) are thread-local - no mutex needed
2. **Global allocators** (`gc_global_gnu_malloc`) require your own mutex
3. **GreyCat objects** created in a function are local to that call - no mutex needed
4. **Global state** accessed from native functions needs protection

### Mutex Pattern

```c
// In global state struct
typedef struct {
    pthread_mutex_t lock;
    // ... shared data ...
} my_state_t;

// In lib_start
pthread_mutex_init(&state->lock, NULL);

// In native functions
void my_function(gc_machine_t *ctx) {
    pthread_mutex_lock(&global_state->lock);

    // Critical section - access/modify shared state

    // IMPORTANT: always unlock, even on error paths
    pthread_mutex_unlock(&global_state->lock);
}

// In lib_stop
pthread_mutex_destroy(&state->lock);
```

## Complete Example: Building a Plugin

### Step 1: Define GCL types

```gcl
// my_plugin.gcl
type Config {
    threshold: float;
    max_items: int?;
}

type Result {
    score: float;
    label: String;
}

type Processor {
    static native fn create(name: String, config: Config?): Processor;
    native fn process(input: Tensor): Result;
    static native fn destroy(name: String);
}
```

### Step 2: Implement C functions

```c
// processor.c
#include "nativegen.h"

void gc_mymod_Processor__create(gc_machine_t *ctx) {
    gc_string_t *name = (gc_string_t *)gc_mymod_Processor_create_name(ctx).object;

    // Optional config parameter
    gc_object_t *config = NULL;
    if (gc_mymod_Processor_create_config_type(ctx) == gc_type_object) {
        config = gc_mymod_Processor_create_config(ctx).object;
    }

    // Read config fields
    f64_t threshold = 0.5;
    if (config != NULL) {
        gc_type_t t;
        gc_slot_t v = gc_object__get_at(config, gc_mymod_Config_threshold, &t, ctx);
        if (t == gc_type_float) threshold = v.f64;
    }

    // Create processor (with global state caching)
    pthread_mutex_lock(&global_store->lock);
    // ... create and cache processor ...
    pthread_mutex_unlock(&global_store->lock);

    // Return boxed result
    gc_object_t *result = gc_machine__create_object(ctx, gc_mymod_Processor);
    gc_machine__set_result(ctx, (gc_slot_t){.object = result}, gc_type_object);
    gc_object__un_mark(result, ctx);
}

void gc_mymod_Processor__process(gc_machine_t *ctx) {
    gc_mymod_processor_t *self = (gc_mymod_processor_t *)gc_machine__this(ctx).object;
    gc_core_tensor_t *input = (gc_core_tensor_t *)gc_machine__get_param(ctx, 0).object;

    // Get tensor data
    f32_t *data = (f32_t *)gc_core_tensor__get_data(input);
    gc_core_tensor_descriptor_t *desc = gc_core_tensor__get_descriptor(input);

    // Process...
    f64_t score = 0.95;
    const char *label = "positive";

    // Build result object
    gc_object_t *result = gc_machine__create_object(ctx, gc_mymod_Result);
    gc_object__set_at(result, gc_mymod_Result_score,
                       (gc_slot_t){.f64 = score}, gc_type_float, ctx);

    gc_string_t *label_str = gc_string__create_from(label, strlen(label));
    gc_object__set_at(result, gc_mymod_Result_label,
                       (gc_slot_t){.object = (gc_object_t *)label_str}, gc_type_object, ctx);

    gc_machine__set_result(ctx, (gc_slot_t){.object = result}, gc_type_object);
    gc_object__un_mark(result, ctx);
}
```

### Step 3: Build

```bash
mkdir build && cd build
cmake .. -DGREYCAT_INCLUDE=/path/to/greycat/include -DGREYCAT_LIB=/path/to/libgreycat.so
make
# Output: my_plugin.gclib
```

### Step 4: Use in GCL

```gcl
fn main() {
    var p = Processor::create("test", Config { threshold: 0.8 });
    var input = Tensor::random([100], TensorType::f32);
    var result = p.process(input);
    println("Score: ${result.score}, Label: ${result.label}");
    Processor::destroy("test");
}
```
