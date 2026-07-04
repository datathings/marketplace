# GreyCat C SDK — Core: Values, Execution, Type System & Logging

The foundation of native GreyCat development: the value model (`gc_slot_t`/`gc_type_t`), the execution context (`gc_machine_t`), the program/type system, and logging. **Start here.** Include `greycat.h` to pull in the entire SDK (it aggregates every `gc/` header).

_Part of the GreyCat C SDK reference (each file is linked from the skill's SKILL.md). Sibling references: api_core.md · api_memory_text.md · api_collections.md · api_runtime_storage.md · api_services.md._

## Contents

- [gc/type.h — Fundamental Types](#gctype-h)
- [gc/machine.h — Machine (Execution Context)](#gcmachine-h)
- [gc/program.h — Program & Type System](#gcprogram-h)
- [gc/log.h — Logging](#gclog-h)
- [Conventions & Patterns](#conventions--patterns)

---

<a id="gctype-h"></a>
## gc/type.h — Fundamental Types

Defines all primitive type aliases, the GreyCat type system enum, the universal value container (`gc_slot_t`), object headers, complex number types, and SDK visibility macros. This is the foundational header included by all others.

### Primitive Type Aliases

| Alias    | C Type      | Description                |
|----------|-------------|----------------------------|
| `i8_t`   | `int8_t`    | Signed 8-bit integer       |
| `u8_t`   | `uint8_t`   | Unsigned 8-bit integer     |
| `i16_t`  | `int16_t`   | Signed 16-bit integer      |
| `u16_t`  | `uint16_t`  | Unsigned 16-bit integer    |
| `i32_t`  | `int32_t`   | Signed 32-bit integer      |
| `u32_t`  | `uint32_t`  | Unsigned 32-bit integer    |
| `i64_t`  | `int64_t`   | Signed 64-bit integer      |
| `u64_t`  | `uint64_t`  | Unsigned 64-bit integer    |
| `f32_t`  | `float`     | 32-bit IEEE 754 float      |
| `f64_t`  | `double`    | 64-bit IEEE 754 double     |
| `geo_t`  | `u64_t`     | Geospatial hash (encoded lat/lng) |

### Complex Number Types

```c
typedef struct {
    f32_t r;  // Real part
    f32_t i;  // Imaginary part
} c64_t;   // 64-bit complex (two 32-bit floats)

typedef struct {
    f64_t r;  // Real part
    f64_t i;  // Imaginary part
} c128_t;  // 128-bit complex (two 64-bit doubles)
```

### Complex Arithmetic Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_c128__add` | `c128_t gc_c128__add(c128_t a, c128_t b)` | Add two complex128 numbers |
| `gc_c128__sub` | `c128_t gc_c128__sub(c128_t a, c128_t b)` | Subtract two complex128 numbers |
| `gc_c128__mul` | `c128_t gc_c128__mul(c128_t a, c128_t b)` | Multiply two complex128 numbers |
| `gc_c128__div` | `c128_t gc_c128__div(c128_t a, c128_t b)` | Divide: `(ac+bd + (bc-ad)i) / (c²+d²)` |
| `gc_c128__addto` | `void gc_c128__addto(c128_t *a, c128_t b)` | In-place add: `a += b` |
| `gc_c128__conj` | `c128_t gc_c128__conj(c128_t a)` | Complex conjugate: `(a+bi)* = a - bi` |
| `gc_c128__arg` | `f64_t gc_c128__arg(c128_t z)` | Argument (angle) of complex number |
| `gc_c128__abs` | `f64_t gc_c128__abs(c128_t z)` | Absolute value (modulus) |
| `gc_c64__add` | `c64_t gc_c64__add(c64_t a, c64_t b)` | Add two complex64 numbers |
| `gc_c64__sub` | `c64_t gc_c64__sub(c64_t a, c64_t b)` | Subtract two complex64 numbers |
| `gc_c64__mul` | `c64_t gc_c64__mul(c64_t a, c64_t b)` | Multiply two complex64 numbers |
| `gc_c64__div` | `c64_t gc_c64__div(c64_t a, c64_t b)` | Divide two complex64 numbers |
| `gc_c64__addto` | `void gc_c64__addto(c64_t *a, c64_t b)` | In-place add: `a += b` |
| `gc_c64__arg` | `f64_t gc_c64__arg(c64_t z)` | Argument (angle) of complex number |
| `gc_c64__abs` | `f64_t gc_c64__abs(c64_t z)` | Absolute value (modulus) |

### Node Reference Parsing

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_node__parse` | `bool gc_node__parse(const char *str, u32_t str_len, gc_slot_t *value, gc_type_t *value_type)` | Parse a node reference from a string representation. Returns `true` on success. |

### gc_type_t — GreyCat Type Enum

Must fit in 8 bits. Used everywhere a type tag is needed (slot types, serialization, etc.).

| Value | Name | Description |
|-------|------|-------------|
| 0  | `gc_type_null` | Null value |
| 1  | `gc_type_bool` | Boolean |
| 2  | `gc_type_char` | Character |
| 3  | `gc_type_int` | 64-bit signed integer |
| 4  | `gc_type_float` | 64-bit double-precision float |
| 5  | `gc_type_node` | Graph node reference |
| 6  | `gc_type_node_time` | Time-indexed node reference |
| 7  | `gc_type_node_index` | Indexed node reference |
| 8  | `gc_type_node_list` | Node list reference |
| 9  | `gc_type_node_geo` | Geospatially-indexed node reference |
| 10 | `gc_type_geo` | Geospatial hash |
| 11 | `gc_type_time` | Timestamp (microseconds since epoch) |
| 12 | `gc_type_duration` | Duration (microseconds) |
| 13 | `gc_type_cubic` | Cubic interpolation value |
| 14 | `gc_type_static_field` | Static field / enum value (stored as `gc_slot_tuple_u32_t`) |
| 15 | `gc_type_object` | Heap-allocated object pointer |
| 16 | `gc_type_block_ref` | Storage block reference |
| 17 | `gc_type_block_inline` | Inline storage block |
| 18 | `gc_type_function` | Function reference |
| 19 | `gc_type_undefined` | Undefined / unknown type |
| 20 | `gc_type_type` | Type reference |
| 21 | `gc_type_field` | Field reference |
| 22 | `gc_type_stringlit` | String literal (serialization only) |
| 23 | `gc_type_error` | Error (C internal only) |

### gc_object_t — Object Header

Generic handle for all GreyCat heap objects. Packed to 128 bits (64 + 32 + 32).

```c
typedef struct {
    gc_block_t *block;   // Pointer to the storage block owning this object (NULL if volatile)
    u32_t marks;         // GC mark bits and internal flags
    u32_t type_id;       // Type identifier (offset into the program's type table)
} gc_object_t;
```

Every GreyCat collection and object type (Array, Map, Table, Tensor, String, Buffer, etc.) begins with a `gc_object_t header` as its first field, enabling polymorphic handling through the common header.

### gc_slot_t — Universal Value Container

A tagged union that can hold any GreyCat value. Always paired with a `gc_type_t` discriminator.

```c
typedef struct {
    union {
        bool b;                    // gc_type_bool
        u8_t byte[8];              // Raw byte access
        u32_t u32;                 // 32-bit unsigned
        i64_t i64;                 // gc_type_int, gc_type_time, gc_type_duration
        u64_t u64;                 // gc_type_geo, gc_type_node, etc.
        f64_t f64;                 // gc_type_float
        gc_slot_tuple_u32_t tu32;  // gc_type_static_field (enum: .left=type, .right=value)
        gc_object_t *object;       // gc_type_object (String, Array, Map, Tensor, etc.)
    };
} gc_slot_t;
```

### gc_slot_tuple_u32_t — Enum / Static Field Pair

```c
typedef struct {
    u32_t left;   // Type offset (identifies the enum type)
    u32_t right;  // Value offset (identifies the enum value within the type)
} gc_slot_tuple_u32_t;
```

### gc_f32_conv_t — Float/Uint32 Conversion Union

Conversion union for reinterpreting `f32` as `u32` (and vice versa) without aliasing violations:

```c
typedef union gc_f32_conv_t_ {
    f32_t f;  // Float interpretation
    u32_t u;  // Unsigned integer interpretation
} gc_f32_conv_t;
```

### Compiler / Visibility Macros

| Macro | Description |
|-------|-------------|
| `gc_sdk` | DLL export/import or `__attribute__((visibility("default")))`. Marks functions as part of the public SDK. |
| `gc_unused` | `__attribute__((unused))` — suppresses unused variable warnings |
| `gc_fallthrough` | `__attribute__((fallthrough))` — marks intentional switch fallthrough |
| `likely(x)` | `__builtin_expect((x), 1)` — branch prediction hint (likely true) |
| `unlikely(x)` | `__builtin_expect((x), 0)` — branch prediction hint (likely false) |

### Well-Known Type IDs (extern globals)

These globals are resolved at program startup and identify built-in GreyCat types by their program type offset.

| Variable | GreyCat Type |
|----------|-------------|
| `gc_core_any` | `any` (top type) |
| `gc_core_geo` | `geo` |
| `gc_core_String` | `String` (heap string object) |
| `gc_core_Buffer` | `Buffer` |
| `gc_core_Array` | `Array` |
| `gc_core_Tuple` | `Tuple` |
| `gc_core_Map` | `Map` |
| `gc_core_Table` | `Table` |
| `gc_core_Tensor` | `Tensor` |
| `gc_core_TensorType` | `TensorType` (enum type for tensor element types) |

### Usage Examples

The entire section above is reference tables and struct layouts; none of the declarations have a runnable snippet. The examples below are grounded in real runtime call sites.

#### Parsing a node reference from a string (`gc_node__parse`)

`gc_node__parse` inspects the trailing suffix of a string (e.g. a `node`/`nodeTime`/`nodeIndex` literal) and, on success, writes the decoded handle into `*value` and the concrete tag into `*value_type`. Mirrors how the JSON reader resolves a stringified node into a slot (`src/std/io/json_reader.c:2162`).

```c
// Resolve a stringified node reference into a typed slot.
gc_slot_t value;
gc_type_t value_type;
if (gc_node__parse(buf.data, buf.size, &value, &value_type)) {
    if (value_type == gc_type_node) {
        // value.u64 now holds the node handle
        gc_machine__set_result(ctx, value, value_type);
        return true;
    }
}
// Parse failed: fall back to undefined.
value.object = NULL;
value_type = gc_type_undefined;
```

#### Dispatching on a built-in type id (`gc_core_*` externs)

The `gc_core_*` globals are resolved at program startup and let native code branch on the concrete built-in type of an object. This is exactly how the JSON reader routes an object's `type_id` to the right parser (`src/std/io/json_reader.c:2392`).

```c
const gc_program_t *prog = gc_machine__program(ctx);
const u32_t type_id = target_type_id;
const gc_program_type_t *type = gc_program__get_program_type(prog, type_id);

if (type_id == gc_core_String) {
    return parse_string(ctx);
}
if (type_id == gc_core_Map || gc_program_type__get_generic_id(type) == gc_core_Map) {
    return parse_map(ctx);            // raw Map or typed Map<K,V>
}
if (type_id == gc_core_Array || gc_program_type__get_generic_id(type) == gc_core_Array) {
    return parse_array(ctx);          // raw Array or typed Array<T>
}
if (type_id == gc_core_Tensor) {
    return parse_tensor(ctx);
}
```

When you allocate a built-in collection directly, pass the corresponding `gc_core_*` id as the type id:

```c
// Create a fresh Array object (json_reader.c:584).
gc_array_t *arr = (gc_array_t *) gc_machine__create_object(ctx, gc_core_Array);
```

#### Reading and tagging values through `gc_slot_t` / `gc_type_t`

A value is always a `(gc_slot_t, gc_type_t)` pair: the slot is an untyped union and the type tag tells you which member is live. Always set the result slot together with its tag.

```c
// Return a 64-bit integer result.
gc_machine__set_result(ctx, (gc_slot_t) {.i64 = 42}, gc_type_int);

// Return a heap object (String/Array/Tensor/...): the .object member is live.
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) destination}, gc_type_object);

// Read an argument back out, switching on its tag.
gc_slot_t arg = gc_machine__this(ctx);
switch (value_type) {
    case gc_type_bool:   use_bool(arg.b);            break;
    case gc_type_int:    // also gc_type_time / gc_type_duration
                         use_int(arg.i64);           break;
    case gc_type_float:  use_double(arg.f64);        break;
    case gc_type_geo:    use_geo((geo_t) arg.u64);   break;
    case gc_type_object: use_object(arg.object);     break;
    default:             break;
}
```

For an enum / static-field value the live union member is `tu32` (`gc_slot_tuple_u32_t`), where `.left` identifies the enum type and `.right` the value within it:

```c
gc_slot_t enum_val = gc_machine__this(ctx);   // value_type == gc_type_static_field
u32_t enum_type_offset = enum_val.tu32.left;
u32_t enum_value_offset = enum_val.tu32.right;
```

#### Complex arithmetic (`gc_c128__*`, `gc_c64__*`)

`gc_c128__abs` and `gc_c128__arg` are used to derive magnitude and phase when converting complex tensors to real ones (`src/std/core/tensor.c:1329`, `tensor.c:1360`). Combine the conjugate/divide/add helpers for higher-level math.

```c
// Magnitude tensor: |z| for every complex128 element (tensor.c:1329).
c128_t *src = (c128_t *) source->data;
f64_t  *dst = (f64_t *)  destination->data;
for (i64_t i = 0; i < source->descriptor.size; i++) {
    dst[i] = gc_c128__abs(src[i]);            // hypot(z.r, z.i)
}

// Phase tensor in degrees: arg(z) * 180/pi (tensor.c:1360).
for (i64_t i = 0; i < source->descriptor.size; i++) {
    dst[i] = gc_c128__arg(src[i]) * (180.0 / M_PI);   // atan2(z.i, z.r)
}
```

```c
// Building up a complex accumulator with the arithmetic helpers.
c128_t a = {.r = 1.0, .i = 2.0};
c128_t b = {.r = 3.0, .i = -1.0};

c128_t prod = gc_c128__mul(a, b);             // (a*b)
c128_t quot = gc_c128__div(prod, b);          // back to ~a
c128_t cj   = gc_c128__conj(a);               // (1 - 2i)

c128_t acc = {0};
gc_c128__addto(&acc, prod);                   // acc += prod (in place)
gc_c128__addto(&acc, gc_c128__sub(a, b));     // acc += (a - b)

// The c64 (two f32) variants mirror the c128 API.
c64_t s = gc_c64__add((c64_t){.r = 0.5f, .i = 0.5f}, (c64_t){.r = 0.5f, .i = -0.5f});
f64_t mag = gc_c64__abs(s);                   // hypotf-based modulus
f64_t ang = gc_c64__arg(s);                   // phase in radians
```

#### Branch hints (`likely` / `unlikely`)

```c
// Hint the optimizer toward the common (success) path.
if (unlikely(value_type == gc_type_error)) {
    gc_machine__set_runtime_error(ctx, "unexpected error value");
    return;
}
if (likely(value_type == gc_type_object)) {
    process(value.object);
}
```


---

<a id="gcmachine-h"></a>
## gc/machine.h — Machine (Execution Context)

The `gc_machine_t` is the execution context passed to all native functions. It provides access to function parameters, result setting, error reporting, program introspection, and object creation.

### Context Type

```c
typedef struct {
    const gc_program_t *prog;
    gc_allocator_t *allocator;  // Per-call allocator (see gc/alloc.h)
    bool flattening;            // Internal: set while the machine is flattening (serializing) a value graph
} gc_ctx_t;
```

> `gc_machine_t` begins with this layout. Inside any native function, `((gc_ctx_t *)ctx)->allocator` (or `gc_machine__allocator(ctx)`) is the allocator to use for per-call scratch memory.

### Functions

#### Parameter Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__get_param` | `gc_slot_t gc_machine__get_param(const gc_machine_t *ctx, u32_t offset)` | Get a function parameter value by its 0-based index |
| `gc_machine__get_param_type` | `gc_type_t gc_machine__get_param_type(const gc_machine_t *ctx, u32_t offset)` | Get the type of a function parameter |
| `gc_machine__get_param_nb` | `u32_t gc_machine__get_param_nb(const gc_machine_t *ctx)` | Get the number of parameters passed |
| `gc_machine__this` | `gc_slot_t gc_machine__this(gc_machine_t *self)` | Get the `this` (self) object for instance methods |

#### Result & Return

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__set_result` | `void gc_machine__set_result(gc_machine_t *self, gc_slot_t slot, gc_type_t slot_type)` | Set the return value of the native function |
| `gc_machine__return_type` | `u32_t gc_machine__return_type(gc_machine_t *self)` | Get the declared return type descriptor |
| `gc_machine__create_return_type_object` | `gc_object_t *gc_machine__create_return_type_object(gc_machine_t *ctx)` | Create a new object matching the declared return type |

#### Error Handling

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__set_runtime_error` | `void gc_machine__set_runtime_error(gc_machine_t *ctx, const char *msg)` | Report a runtime error with a custom message |
| `gc_machine__set_runtime_error_syserr` | `void gc_machine__set_runtime_error_syserr(gc_machine_t *ctx)` | Report a runtime error using the current system `errno` |
| `gc_machine__error` | `bool gc_machine__error(gc_machine_t *ctx)` | Check if an error has occurred |
| `gc_machine__error_buffer` | `gc_buffer_t *gc_machine__error_buffer(gc_machine_t *ctx)` | Get the error message buffer for custom error building |

#### Program, Allocator & Object

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__program` | `const gc_program_t *gc_machine__program(const gc_machine_t *ctx)` | Get the program associated with this machine |
| `gc_machine__allocator` | `gc_allocator_t *gc_machine__allocator(gc_machine_t *ctx)` | Get the per-call allocator for scratch memory (equivalent to `((gc_ctx_t *)ctx)->allocator`). |
| `gc_machine__create_object` | `gc_object_t *gc_machine__create_object(const gc_machine_t *ctx, u32_t object_type_code)` | Create a new object of a specific type by type ID |
| `gc_machine__get_buffer` | `gc_buffer_t *gc_machine__get_buffer(gc_machine_t *ctx)` | Get a reusable scratch buffer (worker-local) |
| `gc_machine__current_fn_off` | `u32_t gc_machine__current_fn_off(gc_machine_t *ctx)` | Get the current function's offset in the program |
| `gc_machine__get_host` | `gc_host_t *gc_machine__get_host(gc_machine_t *ctx)` | Get the host managing this machine |
| `gc_machine__log_level` | `gc_log_level_t gc_machine__log_level(gc_machine_t *ctx)` | Get the current log level (proxy to `gc_log__level`). |
| `gc_machine__impersonate` | `void gc_machine__impersonate(gc_machine_t *ctx, u32_t user_id)` | Switch the machine's effective user id for permission-aware calls. |

#### Advanced

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__call_function` | `bool gc_machine__call_function(gc_machine_t *ctx, const gc_program_function_t *fn, gc_slot_t self, gc_type_t self_type, const gc_slot_t *params, const gc_type_t *params_type, u32_t nb_params, gc_slot_t *marked_res, gc_type_t *marked_res_type)` | Synchronously invoke `fn` (native or bytecode) from a C context with the given `self` and `params`. Always writes the result through `*marked_res` / `*marked_res_type`; the bool return is the Result<T, E> Ok/Err discriminator. On failure `*marked_res` is an `Error` object (type `gc_core_Error`) and `*marked_res_type` is `gc_type_object`; in both branches `ctx`'s error state is left clean. The caller owns one mark on `*marked_res` when it is an object and must call `gc_object__un_mark` when done. |
| `gc_machine__push_function` | `void gc_machine__push_function(gc_machine_t *ctx, const gc_program_function_t *fn, gc_slot_t self, gc_type_t self_type, const gc_task_t *task)` | Push a function frame onto the execution stack |
| `gc_machine__load` | `gc_type_t gc_machine__load(const gc_machine_t *ctx, char *data, u32_t len, gc_slot_t *value)` | Deserialize a slot from raw binary data |
| `gc_machine__lru_add` | `void gc_machine__lru_add(gc_machine_t *ctx, gc_block_t *page)` | Add a block to the LRU cache |

### Usage Examples

#### Canonical native function: read `this`, set a result

Every native function receives `gc_machine_t *ctx`. The common shape is: read the receiver via `gc_machine__this`, do the work, then publish the return value with `gc_machine__set_result`. Object results are created marked, set into the slot, then un-marked by the native (the machine took its own reference inside `set_result`).

```c
// Native impl of MyText::to_string (GCL module "mymod")
void gc_mymod_MyText__to_string(gc_machine_t *ctx) {
    const gc_object_t *self = gc_machine__this(ctx).object;

    // get_buffer returns a reusable worker-local scratch buffer; clear before use
    gc_buffer_t *buf = gc_machine__get_buffer(ctx);
    gc_buffer__clear(buf);

    gc_type_t stub;
    const gc_array_t *codepoints =
        (gc_array_t *) gc_object__get_at(self, gc_mymod_MyText_codepoints, &stub, ctx).object;
    for (u32_t i = 0; i < codepoints->size; i++) {
        gc_slot_t value = {0};
        gc_type_t value_type = gc_type_null;
        gc_array__get_slot(codepoints, i, &value, &value_type);
        gc_buffer__add(buf, value, value_type, ((gc_ctx_t *) ctx)->prog);
    }

    gc_string_t *str = gc_string__create_from(buf->data, buf->size, ctx);
    gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) str}, gc_type_object);
    gc_object__un_mark((gc_object_t *) str, ctx); // machine holds its own ref now
}
```

For a primitive result, set the slot inline with no marking needed:

```c
// Native returning a primitive bool result
void mylib_node_remove(gc_machine_t *ctx) {
    const u64_t node_ref = gc_machine__this(ctx).u64;
    bool removed = do_remove(node_ref, ctx);
    gc_machine__set_result(ctx, (gc_slot_t) {.b = removed}, gc_type_bool);
}
```

#### Reading parameters by offset (with type and arity)

Parameters are addressed by 0-based offset. `get_param` returns the raw slot, `get_param_type` its dynamic type, and `get_param_nb` the count actually passed (useful for variadic / optional-argument natives).

```c
void mylib_concat(gc_machine_t *ctx) {
    u32_t argc = gc_machine__get_param_nb(ctx);
    if (argc < 1) {
        gc_machine__set_runtime_error(ctx, "expected at least one argument");
        return;
    }

    for (u32_t i = 0; i < argc; i++) {
        gc_slot_t arg      = gc_machine__get_param(ctx, i);
        gc_type_t arg_type = gc_machine__get_param_type(ctx, i);
        if (arg_type != gc_type_object) {
            gc_machine__set_runtime_error(ctx, "argument must be a String");
            return;
        }
        process((gc_string_t *) arg.object);
    }
}
```

#### Error reporting: custom message vs. system errno

Use `gc_machine__set_runtime_error` for a domain message; use `gc_machine__set_runtime_error_syserr` after a failing syscall so the raised error carries the current `errno` text. After signalling, `return` immediately — do not also set a result.

```c
// Grounded in src/std/io/io.c / src/std/io/reader.c
int fp = open(path, flags, 0666);
if (fp == -1) {
    gc_machine__set_runtime_error_syserr(ctx); // formats from errno
    return;
}
// ... later, on a logical (non-syscall) failure:
if (!parse_ok) {
    gc_machine__set_runtime_error(ctx, "malformed input");
    return;
}
```

`gc_machine__error` lets long-running loops bail out as soon as an error has been latched (e.g. a callback raised), and `gc_machine__error_buffer` exposes the raw message buffer when you need to build an error string incrementally:

```c
// Grounded in src/std/io/csv_reader.c
while (read_next_row(reader, ctx)) {
    if (gc_machine__error(ctx)) {
        break; // a nested native/callback already raised; stop and propagate
    }
}
```

#### Creating a typed return object

When a native declares an object return type, `gc_machine__create_return_type_object` allocates an instance of exactly that declared type; `gc_machine__create_object` allocates a specific type by its type id (e.g. resolved from the function's `return_type_desc`). Both return a marked object you must un-mark after handing it to `set_result`.

```c
// Native that allocates and returns a user Tuple{x, y} object (module "mymod")
const gc_program_function_t *fn = current_function(ctx);
u32_t tuple_type_id = gc_type_desc__to_type_id(fn->return_type_desc);
gc_object_t *tuple = gc_machine__create_object(ctx, tuple_type_id);

gc_object__set_at(tuple, gc_mymod_Tuple_x, (gc_slot_t) {.u64 = key},  gc_type_geo, ctx);
gc_object__set_at(tuple, gc_mymod_Tuple_y, val_slot, val_type, ctx);

gc_machine__set_result(ctx, (gc_slot_t) {.object = tuple}, gc_type_object);
gc_object__un_mark(tuple, ctx);
```

#### Per-call allocator for scratch memory

`gc_machine__allocator(ctx)` is the per-call allocator (identical to `((gc_ctx_t *) ctx)->allocator`). Use it for temporary buffers scoped to a single native invocation.

```c
gc_allocator_t *alloc = gc_machine__allocator(ctx);
double *scratch = gc_alloc__malloc(alloc, n * sizeof(double));
// ... use scratch for the duration of this call ...
gc_alloc__free(alloc, scratch, n * sizeof(double)); // size is mandatory on free
```

#### Impersonation for permission-aware work

`gc_machine__impersonate` switches the machine's effective user id. After authenticating a login, switch to that user; `impersonate(ctx, 0)` resets to the anonymous/system identity (logout).

```c
// GCL: type Auth in module "myauth" with `native fn login(); native fn logout();`
void gc_myauth_Auth__login(gc_machine_t *ctx) {
    u32_t user_id = 0;
    if (!check_password(ctx, &user_id)) {     // your auth check fills user_id
        gc_machine__set_runtime_error(ctx, "login rejected");
        return;
    }
    gc_machine__impersonate(ctx, user_id); // subsequent calls run as this user
    // ... emit a session token as the result ...
}

void gc_myauth_Auth__logout(gc_machine_t *ctx) {
    gc_machine__impersonate(ctx, 0); // back to anonymous
}
```

#### Synchronously invoking a GreyCat function from C

`gc_machine__call_function` runs a bytecode or native `fn` and always writes the outcome through `*marked_res` / `*marked_res_type`. The bool return is the `Result<T,E>` discriminator: `true` = normal return, `false` = the callee raised and `*marked_res` is a synthesized `gc_core_Error` object. In both cases `ctx`'s error state is left clean, so you decide whether to swallow or propagate. When the result is an object you own one mark and must un-mark it.

```c
gc_slot_t  res;
gc_type_t  res_type;
gc_slot_t  params[1]      = { {.i64 = 42} };
gc_type_t  params_type[1] = { gc_type_int };

bool ok = gc_machine__call_function(ctx, fn,
                                    self, self_type,
                                    params, params_type, 1,
                                    &res, &res_type);
if (!ok) {
    // res is a gc_core_Error object (res_type == gc_type_object); propagate it
    gc_machine__set_runtime_error(ctx, "callee failed");
    if (res_type == gc_type_object) {
        gc_object__un_mark(res.object, ctx);
    }
    return;
}
// success: consume res, then release our mark if it is an object
if (res_type == gc_type_object) {
    gc_object__un_mark(res.object, ctx);
}
```

#### Deserializing a slot from raw bytes

`gc_machine__load` decodes one value from a binary buffer into `*value` and returns its type.

```c
// Grounded in core/src/machine.c / src/store/zone.c
gc_slot_t value;
gc_type_t value_type = gc_machine__load(ctx, data, len, &value);
if (value_type == gc_type_object) {
    use_object(value.object);
    gc_object__un_mark(value.object, ctx);
}
```


---

<a id="gcprogram-h"></a>
## gc/program.h — Program & Type System

The program is the compiled representation of a GreyCat project. It contains all types, functions, modules, symbols, opcodes, and the ABI. This header defines the full program structure, the type system, the bytecode opcodes, and lookup/resolution functions.

### Key Structures

#### gc_program_t

The top-level compiled program:

```c
struct gc_program {
    gc_abi_t *abi;                                // Application Binary Interface
    gc_allocator_t *allocator;                    // Allocator owning program tables
    struct { ... } symbols;                       // Symbol table (interned strings)
    gc_buffer_t fragments;                        // String fragments for doc comments, paths, etc.
    struct { ... } libraries;                     // Loaded native libraries
    struct { ... } modules;                       // Module table with map
    struct { ... } permissions;                   // Permission definitions
    struct { ... } roles;                         // Role definitions
    struct { ... } expose_names;                  // Exposed API name mappings
    gc_program_type_table_t types;                // All types
    gc_program_function_table_t functions;        // All functions
    struct { ... } ops;                           // Bytecode operations and source maps
    gc_program_specialized_type_table_t s_types;  // Specialized (generic) types
    gc_block_t stub;                              // Stub block for volatile objects
};
```

#### gc_object_type_t

Runtime type descriptor for object types:

```c
typedef struct gc_object_type {
    u32_t id;            // Type ID
    u32_t generic_id;    // Generic type ID (for Array<T>, Map<K,V>, etc.)
    u32_t symbol;        // Symbol offset (name)
    u32_t mod;           // Module offset
    struct {
        bool is_enum;             // Enum type
        bool is_abstract;         // Cannot be instantiated directly
        bool is_native;           // Has C native implementation
        bool is_private;          // Not accessible outside its module
        bool is_hidden;           // Type was removed but instances may still exist in storage
        bool is_volatile;         // Not persisted to storage
        bool is_object_container; // Can contain child objects
        bool is_global;           // Global singleton
    } flags;
    gc_object_type_foreach_slots_t *foreach_slots;  // GC slot traversal callback
    gc_object_type_load_t *load;                    // Deserialization callback
    gc_object_type_save_t *save;                    // Serialization callback
    gc_object_type_to_string_t *to_string;          // String representation callback
    gc_type_t type;                                 // Primitive type tag
} gc_object_type_t;
```

#### gc_program_type_t

Extended type information (wraps `gc_object_type_t`):

```c
struct gc_program_type {
    gc_object_type_t header;       // Runtime type info
    struct { ... } fields;         // Instance fields (map + table; table has nb_any / nb_nullable counters)
    struct { ... } static_fields;  // Static fields (enum values, constants)
    struct { ... } functions;      // Methods (map + offset table)
    u32_t g1, g2;                  // Generic type parameters
    u32_t super_type;              // Parent type offset
    u32_t super_type_s_type_off;   // Super type specialized type offset
    u32_t companion_type;          // Companion type offset
    u32_t super_types[GC_PROGRAM_INHERITANCE_MAX_DEPTH];  // Inheritance chain (max depth 4)
    u32_t abi_type;                // ABI type ID
    u32_t binary_size;             // Binary serialization size
    u32_t binary_data_off;         // Offset for binary data fields
    u32_t binary_any_types_off;    // Offset for binary any-type fields
    u32_t binary_nullable_off;     // Offset for binary nullable bitset
    u32_t binary_nullable_len;     // Length of nullable bitset in bytes
    gc_object_type_native_finalize_t *native_finalizer;  // C destructor
};
```

#### gc_program_function_t

Function descriptor:

```c
struct gc_program_function {
    u32_t name_off;                 // Symbol offset for the function name
    u32_t type_off;                 // Owning type offset (0 for module-level functions)
    u32_t mod_off;                  // Module offset
    u32_t op_off;                   // Bytecode start offset
    u32_t type_local_off;           // Type-local function offset
    u32_t generic_type_symb;        // Generic type symbol
    u32_t doc_comment_fragment_off; // Documentation comment fragment offset
    u32_t return_type_desc;         // Return type descriptor
    u32_t return_s_type_off;        // Return specialized type offset
    u32_t expose_rename_off;        // Exposed name override symbol offset
    u8_t nb_params;                 // Number of parameters
    bool is_static;                 // Static method
    bool is_native;                 // Implemented in C
    bool is_private;                // Private visibility
    bool is_reserved;               // Reserved/internal function
    bool is_exposed;                // Exposed via HTTP (@expose annotation)
    bool is_test;                   // Test function
    bool is_abstract;               // Abstract method
    u64_t permissions_mask;         // Required permissions bitmask
    gc_program_src_id_t source;     // Source location
    gc_program_function_body_t *body;  // Native function pointer (NULL for GCL functions)
    u32_t params_type_desc[GC_ABI_FUNCTION_MAX_PARAMS];   // Parameter type descriptors
    u32_t params_s_type_off[GC_ABI_FUNCTION_MAX_PARAMS];  // Parameter specialized type offsets
    u32_t signature_symbols[GC_ABI_FUNCTION_MAX_PARAMS];  // Parameter name symbols
    u32_t tags[gc_compiler_tags_max];                     // Compiler tags (max 2)
};
```

#### gc_program_type_field_t

Field descriptor for a type's instance fields:

```c
typedef struct gc_program_type_field {
    u32_t name_off;                 // Symbol offset for the field name
    u32_t type_desc;                // Type descriptor of the field
    gc_type_t sbi_type;             // Serialization Binary Interface type tag
    u32_t any_offset;               // Offset in the any_fields array (for dynamically-typed fields)
    u32_t nullable_offset;          // Offset in the nullable bitset
    u32_t s_type_off;               // Specialized type offset
    u32_t doc_comment_fragment_off; // Documentation comment fragment offset
    bool is_private;                // Private visibility
    u8_t precision;                 // Numeric precision (gc_abi_precision_t)
    gc_program_src_id_t src_id;     // Source location
} gc_program_type_field_t;
```

#### gc_program_type_static_field_t

Static field (enum value or constant) descriptor:

```c
typedef struct gc_program_type_static_field {
    u32_t name_off;        // Symbol offset for the field name
    gc_type_t def_type;    // Default value type
    gc_slot_t def_value;   // Default value
} gc_program_type_static_field_t;
```

#### gc_program_src_id_t

Source location tracking:

```c
typedef struct gc_program_src_id {
    u32_t mod;     // Module offset
    u32_t line;    // Line number
    u32_t offset;  // Character offset
} gc_program_src_id_t;
```

#### gc_function_param_t

Function parameter descriptor:

```c
typedef struct {
    u32_t name;        // Parameter name symbol
    u32_t type_desc;   // Type descriptor
    u32_t s_type_off;  // Specialized type offset
} gc_function_param_t;
```

#### gc_program_library_t

Native library descriptor:

```c
struct gc_program_library {
    u32_t name_off;          // Library name symbol
    u32_t version_off;       // Library version symbol
    bool is_native;          // True for C libraries
    gc_program_src_id_t src; // Source location
    void *dl_handle;         // dlopen handle
    void *user_data;         // Custom user data
    gc_hook_function_t *start;         // Called on library start
    gc_hook_function_t *stop;          // Called on library stop
    gc_hook_function_t *worker_start;  // Called on each worker start
    gc_hook_function_t *worker_stop;   // Called on each worker stop
};
```

#### gc_program_module_t

Module descriptor:

```c
struct gc_program_module {
    u32_t symbol;           // Module name symbol
    u32_t lib_name_offset;  // Library name offset
    u32_t path_frag_offset; // Path fragment offset
    u32_t path_frag_len;    // Path fragment length
    gc_program_map_t types;      // Type map
    struct {
        gc_program_map_t map;
        gc_program_offset_table_t table;
    } functions;
    struct {
        struct {
            gc_program_module_var_t *data;
            u32_t size;
            u32_t capacity;
        } table;
    } vars;
};
```

#### gc_program_module_var_t

Module-level variable descriptor:

```c
typedef struct gc_program_module_var {
    u32_t symb;            // Variable name symbol
    u32_t type_desc;       // Type descriptor
    u32_t offset;          // Variable offset in module storage
    bool is_private;       // Private visibility
    gc_program_src_id_t declr_src;  // Declaration source location
} gc_program_module_var_t;
```

#### Internal Table Types

```c
// Generic offset table (used for function/type offset arrays)
typedef struct gc_program_offset_table {
    u32_t *data;
    u32_t size;
    u32_t capacity;
} gc_program_offset_table_t;

// Internal hash map bucket
typedef struct {
    u64_t hash;
    u32_t offset;
    u32_t key;
} gc_program_bucket_t;

// Internal hash map (used for symbol/type/function resolution)
typedef struct {
    gc_program_bucket_t *buckets;
    u32_t capacity;
    u32_t size;
    u32_t resize_threshold;
    u64_t mask;
} gc_program_map_t;

// Specialized (generic) type entry
typedef struct gc_program_specialized_type {
    u32_t generic_symbol_off;  // Generic type symbol
    u32_t type_desc;           // Type descriptor
    u32_t g1_s_type_off;       // First generic parameter offset
    u32_t g2_s_type_off;       // Second generic parameter offset
} gc_program_specialized_type_t;
```

### Callback Type Signatures

```c
// Slot traversal callback (used by GC)
typedef bool(gc_object_type_foreach_slots_action_t)(
    gc_object_t *parent, gc_slot_t *slot, gc_type_t slot_type, const gc_machine_t *ctx);

// GC traversal function (iterates all slots in an object)
typedef bool(gc_object_type_foreach_slots_t)(
    gc_object_t *self,
    gc_object_type_foreach_slots_action_t *callback,
    gc_object_type_foreach_slots_action_t *rollback,
    const gc_machine_t *ctx);

// Custom deserialization callback
typedef gc_type_t(gc_object_type_load_t)(
    gc_slot_t *s, gc_block_t *owner, const gc_abi_type_t *abi_type,
    gc_buffer_t *buffer, const gc_machine_t *ctx);

// Custom serialization callback
typedef void(gc_object_type_save_t)(
    gc_object_t *self, gc_buffer_t *buffer, const gc_machine_t *ctx, const bool finalize);

// Custom string representation callback
typedef void(gc_object_type_to_string_t)(
    const gc_object_t *self, gc_buffer_t *target, const gc_program_t *prog);
```

### Function Pointer Types

| Type | Signature | Description |
|------|-----------|-------------|
| `gc_program_function_body_t` | `void(gc_machine_t *ctx)` | Native function body |
| `gc_object_type_native_finalize_t` | `void(gc_object_t *self, gc_machine_t *ctx)` | Native object destructor |
| `gc_hook_function_t` | `bool(gc_program_library_t *lib, gc_program_t *prog, void **user_data)` | Library lifecycle hook |

### Bytecode Instruction Structure

```c
typedef struct gc_program_code {
    gc_program_opcode_t type;   // Opcode
    union {
        u32_t operand;          // Full 32-bit operand
        struct {
            u16_t l;            // Left half-operand
            u16_t r;            // Right half-operand
        } top;
    };
} gc_program_code_t;

typedef struct gc_program_op {
    union {
        gc_program_code_t code; // Instruction
        gc_slot_t data;         // Inline data (constants)
    };
} gc_program_op_t;
```

### Bytecode Opcodes (`gc_program_opcode_t`)

The GreyCat VM bytecode instruction set. Used internally by the interpreter.

<details>
<summary>Full opcode list (85 opcodes, 0-84)</summary>

| Value | Name | Description |
|-------|------|-------------|
| 0 | `interrupt` | Halt execution |
| 1 | `push` | Push N values onto the stack |
| 2 | `pop` | Pop N values off the stack |
| 3 | `jmp` | Unconditional jump |
| 4 | `pop_jof` | Pop and jump on false |
| 5 | `pop_jot` | Pop and jump on true |
| 6 | `jof` | Jump on false (stack unchanged) |
| 7 | `jot` | Jump on true (stack unchanged) |
| 8 | `jon` | Jump on null (stack unchanged) |
| 9 | `joa` | Jump on any not null (stack unchanged) |
| 10 | `load_fn` | Push function reference onto stack |
| 11 | `load_const` | Push constant onto stack |
| 12 | `load_const_str` | Push constant string onto stack |
| 13 | `load_lvar` | Push local variable onto stack |
| 14 | `load_mvar` | Push module variable onto stack |
| 15 | `read_nfield` | Replace top of stack with named field |
| 16 | `push_nfield` | Push named field value |
| 17 | `read_offset` | Pop and replace with offset-ed field |
| 18 | `read_offset_push` | Read offset-ed field (push) |
| 19 | `store_lvar` | Pop and store to local variable |
| 20 | `store_mvar` | Pop and store to module variable |
| 21 | `store_nfield` | Store to named field |
| 22 | `store_ofield` | Store to offset-ed field |
| 23 | `store_offset` | Pop and store to offset-ed field |
| 24 | `is` | Type check |
| 25 | `cast` | Type cast |
| 26 | `load_this` | Load `this` onto stack |
| 27-30 | `inc_load_lvar` / `load_inc_lvar` / `dec_load_lvar` / `load_dec_lvar` | Pre/post increment/decrement local vars |
| 31-32 | `def_time` / `undef_time` | Define/undefine time scope |
| 33-36 | `load_str` / `store_str` / `store_frag` / `push_str` | String building operations |
| 37-42 | `push_arr` / `push_map` / `push_table` / `push_obj` / `push_tuple` / `push_geo` | Literal construction |
| 43-47 | `push_node` / `push_node_time` / `push_node_list` / `push_node_index` / `push_node_geo` | Node literal construction |
| 48 | `push_const_def` | Push constant definition |
| 49-51 | `call_fn` / `call_met` / `call_end` | Function/method calls |
| 52 | `ret` | Return from function |
| 53 | `not` | Logical NOT |
| 54 | `noop` | No operation |
| 55-59 | `try` / `untry` / `throw` / `catch_a` / `catch_p` | Exception handling |
| 60-65 | `eq` / `ne` / `lt` / `le` / `gt` / `ge` | Comparison operators |
| 66-67 | `uminus` / `unref` | Unary minus / dereference |
| 68-75 | `add` / `sub` / `mul` / `div` / `mod` / `pow` / `and` / `or` | Arithmetic & logical |
| 76-77 | `for_st` / `for_do` | For-loop start/body |
| 78 | `breakpoint` | Debugger breakpoint |
| 79-82 | `break` / `break_for_in` / `continue` / `continue_for_in` | Loop control flow |
| 83 | `volatile_mod` | Volatile module access |
| 84 | `jol` | Jump on log level greater than machine log level |

</details>

### Operator Enum (`gc_program_operator_t`)

Maps to the binary operator opcodes. Values 0-18 covering: `not`, `uminus`, `unref`, `+`, `-`, `/`, `*`, `%`, `**`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `||`, `&&`, `is`, `as`.

### Iterator Parameters (`gc_program_iterator_param_t`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `gc_program_iterator_param_from` | Start key (inclusive) |
| 1 | `gc_program_iterator_param_to` | End key (inclusive) |
| 2 | `gc_program_iterator_param_nullable` | Include nullable entries |
| 3 | `gc_program_iterator_param_from_excl` | Start key (exclusive) |
| 4 | `gc_program_iterator_param_to_excl` | End key (exclusive) |

### Program Functions

#### Program Linking

| Function | Description |
|----------|-------------|
| `gc_program__link_mod_fn(prg, module_id, function, fn_name_symbol)` | Link a native function to a module-level function slot. |
| `gc_program__link_type_fn(prg, type_id, function, fn_name_symbol)` | Link a native function to a type method slot. |
| `gc_program_function__set_body(prog, fn_off, fn_body)` | Set the native body of a function. |
| `gc_program_function__is_native_without_body(prog, fn_off)` | Check if a native function has no body yet. |

#### Symbol Resolution

| Function | Description |
|----------|-------------|
| `gc_program__get_symbol(program, symb_off)` | Get a symbol string by its offset. |
| `gc_program__get_symbol_off(symb)` | Get the symbol offset from a string pointer. |
| `gc_program__resolve_symbol(program, str, len)` | Resolve a raw string to a symbol offset. |
| `gc_program__resolve_module(program, mod_name_offset)` | Resolve a module by name offset. |
| `gc_program__resolve_type(prog, mod_offset, type_name_off)` | Resolve a type within a module. |
| `gc_program__resolve_field(type, name_off, res)` | Resolve a field within a type. |

#### Type Introspection

| Function | Description |
|----------|-------------|
| `gc_program__get_program_type(prog, type_id)` | Get the full program type by ID. |
| `gc_program_type__get_field(type, field_offset)` | Get a field descriptor by offset. |
| `gc_program_type__get_field_by_key(type, key)` | Get a field descriptor by symbol key. |
| `gc_program_type__get_g1_type_id(type)` / `gc_program_type__get_g2_type_id(type)` | Get generic type parameter IDs. |
| `gc_program_type__get_g1_type_desc(type)` / `gc_program_type__get_g2_type_desc(type)` | Get generic type parameter descriptors. |
| `gc_program_type__get_generic_id(type)` | Get the generic type ID. |
| `gc_program_type__configure(prog, type_id, header_bytes, native_finalize)` | Configure native type (header size and destructor). |
| `gc_program_type__abi_type_id(prog, type_id)` | Get the ABI type ID for a program type. |
| `gc_program__is_type(prog, source_type_id, target_type_id)` | Return `true` if `source_type_id` is (or is a monomorphized form of) `target_type_id`. |

#### Function Introspection

| Function | Description |
|----------|-------------|
| `gc_program__get_function(prog, fn_off)` | Get a function descriptor by offset. |
| `gc_program_function__module_name_off(prog, fn_off)` | Get the module name offset. |
| `gc_program_function__type_name_off(prog, fn_off)` | Get the type name offset. |
| `gc_program_function__function_name_off(prog, fn_off)` | Get the function name offset. |
| `gc_program_function__return_type_desc(prog, fn_off)` | Get the return type descriptor. |
| `gc_program_function__get_param_by_off(fn, offset, out)` | Get parameter info by index. |

#### Field Introspection

| Function | Description |
|----------|-------------|
| `gc_program_type_field__get_type_id(field)` | Get the type ID from a field descriptor. |
| `gc_program_type_field__get_type_desc(field)` | Get the type descriptor from a field descriptor. |
| `gc_program_type_field__get_name(field, prog)` | Get the field name as a `gc_string_t`. |

#### Type Descriptor Utilities

| Function | Description |
|----------|-------------|
| `gc_type_desc__is_nullable(type_d)` | Check if a type descriptor marks the type as nullable. |
| `gc_type_desc__to_type_id(type_d)` | Extract the type ID from a type descriptor. |
| `gc_type_desc__to_type_desc(type_id, is_nullable)` | Build a type descriptor from a type ID and nullable flag. |

#### Format Pragma Extraction

| Function | Description |
|----------|-------------|
| `gc_program_type__extract_format_pragma_arg(type, field_name, pragma_name, arg_index, buf, prog, slot, slot_type)` | Extract a specific argument from a type field's pragma. Returns `true` if found. |
| `gc_program_type__extract_field_format(type, field, default_tz, buf, prog, out)` | Extract the full field format (format string, DurationUnit, TimeZone) from a field's `@format` pragma. Writes result to `out`. |

#### Module, Program & Library

| Function | Description |
|----------|-------------|
| `gc_program__create_module(program, mod_name_offset, result_offset)` | Create a new module in the program. Returns `true` on success, writes offset to `*result_offset`. |
| `gc_program__create(abi, allocator)` | Create an empty program bound to the given ABI and allocator. |
| `gc_program__create_from_abi(abi, allocator)` | Create a populated program from an ABI definition. |
| `gc_program__finalize(program)` | Finalize and free a program created with `gc_program__create` / `gc_program__create_from_abi`. |
| `gc_program_library__set_lib_hooks(lib, start, stop)` | Set library start/stop hooks. |
| `gc_program_library__set_worker_hooks(lib, start, stop)` | Set worker start/stop hooks. |
| `gc_lib_std__link(prg, lib)` | Link the standard library to a program. |

#### Internal Map Operations

These are used internally for program symbol/type/function resolution:

| Function | Description |
|----------|-------------|
| `gc_program_map__put(prog, self, hash, offset, key)` | Insert an entry into a program map. |
| `gc_program_map__hash_off(offset)` | Compute a hash for an offset. |
| `gc_program_map__get_key(self, key, hash, offset)` | Look up an entry by key and hash. |

### DurationUnit Constants

Predefined ordinals for the `DurationUnit` enum, used with the field format pragma system:

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_core_DurationUnit_microseconds` | 0 | Microseconds |
| `gc_core_DurationUnit_milliseconds` | 1 | Milliseconds |
| `gc_core_DurationUnit_seconds` | 2 | Seconds |
| `gc_core_DurationUnit_minutes` | 3 | Minutes |
| `gc_core_DurationUnit_hours` | 4 | Hours |
| `gc_core_DurationUnit_days` | 5 | Days |

### Field Format (from `@format` pragma)

```c
typedef struct gc_program_type_field_format {
    gc_string_t *format;  // Custom format string (NULL if none)
    u32_t unit;           // DurationUnit ordinal
    u32_t tz;             // TimeZone ordinal
} gc_program_type_field_format_t;
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_compiler_tags_max` | 2 | Max compiler tags per function |
| `GC_PROGRAM_INHERITANCE_MAX_DEPTH` | 4 | Maximum type inheritance depth |
| `GC_PROGRAM_TEMPLATE_MAX_NESTED` | 4 | Maximum nested generic type depth |

### Usage Examples

These examples mirror the real GreyCat runtime: the program (`gc_program_t`) is created once from the ABI, native code is linked into it during the library `__link` callback, and type/field metadata is queried from the program tables during introspection (code generation, serialization, CSV/JSON parsing).

#### Linking native functions and types into a program

Inside a library link function (`<lib>__link(gc_program_t *prg, gc_program_library_t *lib)`), resolve each symbol then bind the native body. Module-level functions use `gc_program__link_mod_fn`; type methods use `gc_program__link_type_fn`. This is exactly the pattern emitted by `nativegen`.

```c
bool mylib__link(gc_program_t *prg, gc_program_library_t *lib) {
    bool res = true;

    // Resolve your module's offset once (the GCL module that declares the natives).
    u32_t mymod = gc_program__resolve_module(prg, gc_program__resolve_symbol(prg, "mymod", 5));
    res &= (mymod != 0);

    // Bind a module-level function: mymod::compute -> native gc_mymod__compute.
    res &= gc_program__link_mod_fn(prg, mymod, gc_mymod__compute,
                                   gc_program__resolve_symbol(prg, "compute", 7));

    // Resolve a type, then bind a method on it: Point.norm -> gc_mymod_Point__norm.
    u32_t point_t = gc_program__resolve_type(prg, mymod,
                                             gc_program__resolve_symbol(prg, "Point", 5));
    res &= (point_t != 0);
    res &= gc_program__link_type_fn(prg, point_t, gc_mymod_Point__norm,
                                    gc_program__resolve_symbol(prg, "norm", 4));

    // Register lifecycle hooks for the library and per-worker callbacks.
    gc_program_library__set_lib_hooks(lib, mylib_start, mylib_stop);
    gc_program_library__set_worker_hooks(lib, mylib_worker_start, mylib_worker_stop);

    return res;
}
```

A library hook has the `gc_hook_function_t` signature and returns success:

```c
static bool mylib_start(gc_program_library_t *lib, gc_program_t *prog, void **user_data) {
    (void) prog;
    *user_data = malloc(sizeof(my_state_t));   // becomes lib->user_data
    return *user_data != NULL;
}
```

#### Configuring a native type (header bytes + finalizer)

Native types reserve extra header bytes for their C struct and supply a finalizer that runs when the object is freed. `gc_program_type__configure` takes the type id, the header size (often `sizeof(...)`), and a `gc_object_type_native_finalize_t *`.

```c
extern void gc_mymod_Reader__native_finalize(gc_object_t *self, gc_machine_t *ctx);

// Reserve sizeof(gc_mymod_reader_t) bytes in the object header and install the destructor.
gc_program_type__configure(prog, gc_mymod_Reader,
                           sizeof(gc_mymod_reader_t),
                           gc_mymod_Reader__native_finalize);
```

#### Resolving a field and reading its descriptor

To read or write an object field by symbol, resolve the field offset on the type, then index `type->fields.table.data`. `gc_program__resolve_field` returns `false` for unknown fields. This is the path used by `gc_object__get_at` and the JSON reader.

```c
const gc_program_t *prog = gc_machine__program(ctx);
const gc_program_type_t *type = gc_program__get_program_type(prog, self->type_id);

u32_t offset;
if (!gc_program__resolve_field(type, key, &offset)) {
    gc_machine__set_runtime_error(ctx, "field not found");
    return;
}

const gc_program_type_field_t *field = gc_program_type__get_field(type, offset);

// Decode the field's type descriptor.
u32_t field_type_id = gc_type_desc__to_type_id(field->type_desc);
bool  is_nullable   = gc_type_desc__is_nullable(field->type_desc);

// Same data via the accessor helpers:
u32_t same_id   = gc_program_type_field__get_type_id(field);   // == field_type_id
u32_t same_desc = gc_program_type_field__get_type_desc(field); // == field->type_desc

// Follow the descriptor to the field's full program type (e.g. for codegen).
const gc_program_type_t *field_type = gc_program__get_program_type(prog, field_type_id);

// Field name as an interned string.
gc_string_t *name = gc_program_type_field__get_name(field, prog);
```

#### Extracting a field's `@format` pragma

When serializing/parsing temporal fields, resolve the per-field format (custom format string, DurationUnit, TimeZone) with `gc_program_type__extract_field_format`. It writes into a caller-owned `gc_program_type_field_format_t` and uses a scratch buffer plus a default timezone fallback. Grounded in `csv.c` and `json_reader.c`.

```c
gc_program_type_field_format_t fmt;
gc_program_type__extract_field_format(type, field, csv->tz, gc_machine__get_buffer(ctx),
                                      gc_machine__program(ctx), &fmt);

if (fmt.format != NULL) {
    col->format = fmt.format;      // custom format string from @format
}
if (fmt.tz != csv->tz) {
    col->tz = fmt.tz;              // overridden TimeZone ordinal
}
col->unit = fmt.unit;             // DurationUnit ordinal (e.g. gc_core_DurationUnit_seconds)
```

#### Iterating a module's functions and reading symbols

The program tables can be walked directly. Each module keeps a function offset table indexing into `program->functions.data`; `gc_program__get_symbol` turns a name offset into a string. Grounded in the test runner (`cli/test.c`).

```c
for (u32_t i = 0; i < mod->functions.table.size; i++) {
    u32_t fn_off = mod->functions.table.data[i];
    const gc_program_function_t *fn = program->functions.data + fn_off;

    const gc_string_t *fn_name = gc_program__get_symbol(program, fn->name_off);
    if (fn_name->size == 5 && strcmp(fn_name->buffer, "setup") == 0) {
        // ... found the module's setup() function
    }

    // Inspect parameters by index.
    for (u32_t p = 0; p < fn->nb_params; p++) {
        gc_function_param_t param;
        if (gc_program_function__get_param_by_off(fn, p, ¶m)) {
            const gc_string_t *pname = gc_program__get_symbol(program, param.name);
            u32_t ptype_id = gc_type_desc__to_type_id(param.type_desc);
            (void) pname; (void) ptype_id;
        }
    }
}
```

#### Resolving a CLI argument: symbol -> module

Resolution functions return `0` to signal "not found"; chain them and check each step. From `cli/test.c`.

```c
u32_t symb_off = gc_program__resolve_symbol(program, cli_arg, strlen(cli_arg));
if (symb_off != 0) {
    u32_t mod_off = gc_program__resolve_module(program, symb_off);
    if (mod_off != 0) {
        gc_program_module_t *mod = program->modules.table.data + mod_off;
        // run all tests in that module ...
    }
}
```


---

<a id="gclog-h"></a>
## gc/log.h — Logging

Structured logging from both VM (machine) and host contexts. Emitted records are decorated with the level, ISO timestamp, user/task ids, and (when emitted from a VM frame) the current `module::Type::fn`. CSV records are appended to the host log file. Calls return immediately when `level` exceeds the configured threshold, so it is safe to use the convenience helpers in hot paths.

### Log Levels

```c
typedef enum gc_log_level {
    gc_log_level_none  = 0,
    gc_log_level_error = 1,
    gc_log_level_warn  = 2,
    gc_log_level_info  = 3,
    gc_log_level_perf  = 4,
    gc_log_level_trace = 5
} gc_log_level_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_log__level` | `gc_log_level_t gc_log__level(const gc_host_t *host)` | Return the host-configured log level threshold. |
| `gc_log__is_tty` | `bool gc_log__is_tty(const gc_host_t *host)` | Return `true` if the host log destination is a TTY (controls coloured output). |
| `gc_log__enabled` | `bool gc_log__enabled(const gc_host_t *host, gc_log_level_t level)` | Convenience check: `true` iff `level` would be emitted. |
| `gc_log__machine` | `void gc_log__machine(gc_machine_t *ctx, gc_log_level_t level, gc_slot_t value, gc_type_t value_type)` | Emit a record from a VM context, formatting `value` according to its type. |
| `gc_log__machinef` | `void gc_log__machinef(gc_machine_t *ctx, gc_log_level_t level, const char *cstr)` | Emit a NUL-terminated C-string record from a VM context. |
| `gc_log__host` | `void gc_log__host(gc_host_t *host, gc_log_level_t level, gc_buffer_t *body)` | Emit a record from a host context (no VM frame). The buffer is left cleared on return. |
| `gc_log__hostf` | `void gc_log__hostf(gc_host_t *host, gc_log_level_t level, const char *cstr)` | Emit a NUL-terminated C-string record from a host context. |

### Usage Examples

**Host-context logging (CLI / no VM frame).** Build the message into a `gc_buffer_t`, then emit it. `gc_log__host` clears the buffer on return, so reuse it directly for the next record:

```c
// from a host context (e.g. a CLI command): report progress and warnings.
// gc_host__get_global() returns the process-wide host when you have no ctx.
gc_host_t *host = gc_host__get_global();
gc_buffer_t *buf = gc_buffer__create(gc_host__allocator(host));
gc_buffer__add_cstr(buf, "GreyCat is serving on port: ");
gc_buffer__add(buf, (gc_slot_t) {.i64 = port}, gc_type_int, gc_host__program(host));
gc_log__host(host, gc_log_level_info, buf);  // buffer left cleared on return

gc_buffer__add_cstr(buf, "server ready");
gc_log__host(host, gc_log_level_warn, buf);
```

**Convenience C-string helpers.** When you already hold a NUL-terminated string, skip the buffer dance with `gc_log__hostf` (host) or `gc_log__machinef` (VM):

```c
// host context
gc_log__hostf(host, gc_log_level_error, "failed to open store");

// VM context (inside a native function with a gc_machine_t *ctx)
gc_log__machinef(ctx, gc_log_level_trace, "entering native handler");
```

Both return immediately when `level` exceeds the configured threshold, so they are cheap to leave in hot paths.

**VM-context structured logging.** `gc_log__machine` formats a `gc_slot_t` according to its `gc_type_t` and decorates the TTY line with the current frame's `module::Type::fn`. Guard expensive object construction with `gc_log__enabled` so you allocate nothing when the level is suppressed; mark/unmark the temporary object across the call:

```c
if (gc_log__enabled(gc_machine__get_host(ctx), gc_log_level_perf)) {
    gc_object_t *usage = gc_machine__create_object(ctx, gc_mymod_UsageStat);
    gc_object__set_at(usage, gc_mymod_UsageStat_read_bytes,
                      (gc_slot_t) {.u64 = stat->read_bytes}, gc_type_int, ctx);
    gc_object__set_at(usage, gc_mymod_UsageStat_write_bytes,
                      (gc_slot_t) {.u64 = stat->write_bytes}, gc_type_int, ctx);
    gc_log__machine(ctx, gc_log_level_perf, (gc_slot_t) {.object = usage}, gc_type_object);
    gc_object__un_mark(usage, ctx);
}
```

You can also log a primitive slot directly, without allocating an object:

```c
gc_slot_t arg      = gc_machine__get_param(ctx, 0);
gc_type_t arg_type = gc_machine__get_param_type(ctx, 0);
gc_log__machine(ctx, gc_log_level_error, arg, arg_type);
```

**Inspecting the configured level / TTY.** `gc_log__level` returns the threshold (useful for opting into verbose paths only under `trace`); `gc_log__is_tty` reports whether the destination is a terminal, which callers use to decide between coloured/interactive output and plain log lines:

```c
const bool trace = (gc_log__level(gc_machine__get_host(ctx)) == gc_log_level_trace);
if (trace) {
    // gather and emit detailed diagnostics only under trace
}

gc_host_t *host = gc_host__get_global();   // or gc_machine__get_host(ctx) in a VM frame
if (gc_log__is_tty(host)) {
    // emit hyperlinked / coloured output directly to the terminal
} else {
    gc_log__hostf(host, gc_log_level_info, "ready");
}
```


---

## Conventions & Patterns

### Naming Conventions

- **Types:** `gc_{module}_{name}_t` (e.g., `gc_buffer_t`, `gc_tensor_t`)
- **Functions:** `gc_{module}__{verb}` (e.g., `gc_buffer__add_str`, `gc_object__get_at`)
- **Constants/Macros:** `GC_{MODULE}_{NAME}` (e.g., `GC_CORE_TENSOR_DIM_MAX`)
- **Well-known type globals:** `gc_core_{TypeName}` (e.g., `gc_core_Array`)
- Double underscore `__` separates the type/module from the method name

### Native Function Pattern

```c
void my_native_function(gc_machine_t *ctx) {
    // 1. Get parameters
    gc_slot_t param0 = gc_machine__get_param(ctx, 0);
    gc_type_t type0 = gc_machine__get_param_type(ctx, 0);

    // 2. Access `this` for instance methods
    gc_slot_t self_slot = gc_machine__this(ctx);
    gc_object_t *self = self_slot.object;

    // 3. Read object fields
    gc_type_t field_type;
    gc_slot_t field_val = gc_object__get_at(self, FIELD_OFFSET, &field_type, ctx);

    // 4. Do computation...

    // 5. Set result
    gc_slot_t result = {.i64 = 42};
    gc_machine__set_result(ctx, result, gc_type_int);
}
```

### Memory Rules

- **Per-call allocations** (`gc_alloc__malloc(gc_machine__allocator(ctx), sz)`): Default for anything scoped to a single native call. Freed by caller when the call ends.
- **Global allocations** (`gc_alloc__malloc(gc_host__global_allocator(), sz)`): Use for module-level state initialized in `lib_start` / freed in `lib_stop`. Requires your own mutex when shared across workers.
- **Object creation** (`gc_machine__create_object`): Use for GreyCat objects — they are managed by the GC.
- **Mark/Unmark**: When you receive a marked object (e.g., from `gc_array__remove_at`), you must `gc_object__un_mark` it when you're done if you don't want to keep it alive.
