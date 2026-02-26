# GreyCat C SDK API Reference

Complete reference documentation for the GreyCat C SDK header files. This SDK enables native C function development for the GreyCat runtime, allowing you to extend the GCL language with high-performance C implementations.

**Include hierarchy:** Include `greycat.h` to get the entire SDK. It aggregates all individual headers under `gc/`.

---

## Table of Contents

- [gc/type.h — Fundamental Types](#gctype-h)
- [gc/alloc.h — Memory Allocation](#gcalloc-h)
- [gc/buffer.h — Buffer Operations](#gcbuffer-h)
- [gc/string.h — String Objects](#gcstring-h)
- [gc/str.h — Inline Short Strings](#gcstr-h)
- [gc/object.h — Object Manipulation](#gcobject-h)
- [gc/machine.h — Machine (Execution Context)](#gcmachine-h)
- [gc/program.h — Program & Type System](#gcprogram-h)
- [gc/host.h — Host & Task Management](#gchost-h)
- [gc/array.h — Dynamic Arrays](#gcarray-h)
- [gc/map.h — Hash Maps](#gcmap-h)
- [gc/table.h — 2D Tables](#gctable-h)
- [gc/tensor.h — Multi-dimensional Tensors](#gctensor-h)
- [gc/block.h — Storage Blocks](#gcblock-h)
- [gc/abi.h — ABI (Application Binary Interface)](#gcabi-h)
- [gc/io.h — File I/O](#gcio-h)
- [gc/crypto.h — Cryptography](#gccrypto-h)
- [gc/geo.h — Geospatial Operations](#gcgeo-h)
- [gc/time.h — Date & Time](#gctime-h)
- [gc/math.h — Math Functions (WASM)](#gcmath-h)
- [gc/util.h — Utility Functions](#gcutil-h)

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
    f32_t real;
    f32_t imag;
} c64_t;   // 64-bit complex (two 32-bit floats)

typedef struct {
    f64_t real;
    f64_t imag;
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
| 16 | `gc_type_t2` | Tuple of 2 ints |
| 17 | `gc_type_t3` | Tuple of 3 ints |
| 18 | `gc_type_t4` | Tuple of 4 ints |
| 19 | `gc_type_str` | Inline short string (encoded in 8 bytes) |
| 20 | `gc_type_t2f` | Tuple of 2 floats |
| 21 | `gc_type_t3f` | Tuple of 3 floats |
| 22 | `gc_type_t4f` | Tuple of 4 floats |
| 23 | `gc_type_block_ref` | Storage block reference |
| 24 | `gc_type_block_inline` | Inline storage block |
| 25 | `gc_type_function` | Function reference |
| 26 | `gc_type_undefined` | Undefined / unknown type |
| 27 | `gc_type_type` | Type reference |
| 28 | `gc_type_field` | Field reference |
| 29 | `gc_type_stringlit` | String literal (serialization only) |
| 30 | `gc_type_error` | Error (C internal only) |

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
| `gc_core_t2` / `gc_core_t3` / `gc_core_t4` | `t2` / `t3` / `t4` (int tuples) |
| `gc_core_t2f` / `gc_core_t3f` / `gc_core_t4f` | `t2f` / `t3f` / `t4f` (float tuples) |
| `gc_core_str` | `str` (inline short string) |
| `gc_core_String` | `String` (heap string object) |
| `gc_core_Buffer` | `Buffer` |
| `gc_core_Array` | `Array` |
| `gc_core_Tuple` | `Tuple` |
| `gc_core_Map` | `Map` |
| `gc_core_Table` | `Table` |
| `gc_core_Tensor` | `Tensor` |
| `gc_core_TensorType` | `TensorType` (enum type for tensor element types) |

---

<a id="gcalloc-h"></a>
## gc/alloc.h — Memory Allocation

Provides multiple allocation strategies. On WASM targets, also declares standard C memory functions (`memset`, `memcpy`, `memcmp`, `strcmp`, `memmove`, `strlen`).

### Per-Worker Allocators (Thread-Local)

Use these in native functions executing on a specific GreyCat worker. Allocations are tracked per-worker and reclaimed when the worker's task completes.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_malloc` | `void *gc_malloc(size_t size)` | Allocate `size` bytes (worker-local, sized free) |
| `gc_free` | `void gc_free(void *ptr, size_t size)` | Free memory allocated with `gc_malloc` (requires `size`) |

### Per-Worker GNU-Style Allocators

Standard malloc/free-style interface (no size required for free). Thread-local.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_gnu_malloc` | `void *gc_gnu_malloc(size_t size)` | Allocate `size` bytes |
| `gc_gnu_free` | `void gc_gnu_free(void *ptr)` | Free memory (no size required) |
| `gc_gnu_calloc` | `void *gc_gnu_calloc(size_t count, size_t size)` | Allocate and zero-initialize |
| `gc_gnu_realloc` | `void *gc_gnu_realloc(void *ptr, size_t new_size)` | Resize allocation |

### Global Allocators (Thread-Safe)

For memory shared across workers. These use global (non-thread-local) allocators with appropriate synchronization.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_global_gnu_malloc` | `void *gc_global_gnu_malloc(size_t size)` | Global allocate |
| `gc_global_gnu_calloc` | `void *gc_global_gnu_calloc(size_t count, size_t size)` | Global allocate + zero |
| `gc_global_gnu_realloc` | `void *gc_global_gnu_realloc(void *ptr, size_t new_size)` | Global resize |
| `gc_global_gnu_free` | `void gc_global_gnu_free(void *ptr)` | Global free |

### Aligned Allocators

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_align_malloc` | `void *gc_align_malloc(size_t size, size_t block_size)` | Allocate aligned to `block_size` |
| `gc_aligned_free` | `void gc_aligned_free(void *ptr, size_t size)` | Free aligned memory |

### Helper Macros

```c
// Free ptr only if non-NULL; requires knowing the allocation size
gc_free_not_null(ptr, size)
```

---

<a id="gcbuffer-h"></a>
## gc/buffer.h — Buffer Operations

A growable byte buffer used throughout GreyCat for serialization (binary, JSON, text), string building, I/O, and internal protocol encoding. The buffer maintains a write cursor (`current`) and supports both high-level append operations and low-level inline read/write of fixed-width primitives.

### Structure

```c
typedef struct gc_buffer {
    gc_object_t header;       // Object header (makes Buffer a first-class GreyCat object)
    char *data;               // Pointer to the raw byte array
    u64_t capacity;           // Allocated capacity in bytes
    u64_t size;               // Logical size (total written bytes)
    char *current;            // Read/write cursor position
    gc_buffer_options_t options;  // Formatting options
} gc_buffer_t;
```

### Buffer Options

```c
typedef struct {
    bool json;          // JSON output mode
    bool tty;           // Terminal (color) output mode
    bool pretty;        // Pretty-print with indentation
    bool global;        // Use global allocator
    char dec_sep;       // Decimal separator character (e.g., '.' or ',')
    char th_sep;        // Thousands separator character
    i32_t f_digit;      // Float digit precision
    i32_t level;        // Indentation level (for pretty printing)
    u64_t block_size;   // Block size hint
    u32_t tz;           // Timezone offset for time formatting
} gc_buffer_options_t;
```

### Lifecycle

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_buffer__create` | `gc_buffer_t *gc_buffer__create()` | Allocate and initialize a new buffer |
| `gc_buffer__finalize` | `void gc_buffer__finalize(gc_buffer_t *self)` | Release all buffer memory |
| `gc_buffer__clear` | `void gc_buffer__clear(gc_buffer_t *self)` | Reset size to 0 (keeps allocation) |
| `gc_buffer__clear_secure` | `void gc_buffer__clear_secure(gc_buffer_t *self)` | Reset and zero-fill the whole allocation (for secrets) |
| `gc_buffer__prepare` | `void gc_buffer__prepare(gc_buffer_t *self, u64_t needed)` | Ensure at least `needed` extra bytes of capacity |

### High-Level Append (String/Text Building)

| Function | Description |
|----------|-------------|
| `gc_buffer__add_str(self, c, len)` | Append raw bytes |
| `gc_buffer__add_cstr(self, c)` | Append a null-terminated C string |
| `gc_buffer__add_char(self, c)` | Append a single character |
| `gc_buffer__add_pstr(self, c)` | Append a string literal (compile-time `sizeof` for length) |
| `gc_buffer__prepend_str(self, c, len)` | Prepend raw bytes (shifts existing content) |
| `gc_buffer__prepend_pstr(self, c)` | Prepend a string literal |
| `gc_buffer__add_escaped_char(self, c)` | Append a character with JSON escaping |
| `gc_buffer__add_escaped_str(self, str, len)` | Append a string with JSON escaping |
| `gc_buffer__add_str_escaped(self, c, len, sep)` | Append with escape using a custom separator |
| `gc_buffer__add_str_doubled(self, c, len, sep)` | Append with doubled separator escaping |
| `gc_buffer__add_ident_protected(self, c, len)` | Append with identifier-safe characters |
| `gc_buffer__trim(self)` | Remove trailing whitespace |

### Value Formatting

| Function | Description |
|----------|-------------|
| `gc_buffer__add_i64(self, i)` | Append a signed 64-bit integer as decimal text |
| `gc_buffer__add_u64(self, i)` | Append an unsigned 64-bit integer as decimal text |
| `gc_buffer__add_u64_inplace(self, i, offset)` | Overwrite an unsigned 64-bit integer at a specific byte offset in the buffer |
| `gc_buffer__add_vu32(self, i)` | Append a variable-length encoded `u32_t` (LEB128 style) |
| `gc_buffer__add_vu64(self, i)` | Append a variable-length encoded `u64_t` (LEB128 style) |
| `gc_buffer__add_f64(self, f)` | Append a double as decimal text |
| `gc_buffer__add_f64_hex(self, slot, prog)` | Append a double in hexadecimal notation |
| `gc_buffer__add_hex(self, data, len)` | Append raw bytes as hex string |
| `gc_buffer__add_duration(self, value)` | Append a duration in human-readable form |
| `gc_buffer__add_byte_size(self, value)` | Append byte size in human-readable IEC units (KiB, MiB...) |
| `gc_buffer__add_byte_size_si(self, value)` | Append byte size in SI units (KB, MB...) |
| `gc_buffer__add_symbol(self, symb_id, prog)` | Append a symbol name by its ID |
| `gc_buffer__add_protected_symbol(buf, symb_off, prog)` | Append symbol with non-alphanumeric replaced by `_` |
| `gc_buffer__add_function(self, fn_off, prog)` | Append a function's qualified name |
| `gc_buffer__add_type_name(self, value, type, prog)` | Append the type name of a slot |
| `gc_buffer__add_type_name_by_id(self, type_id, prog)` | Append a type name by its type ID |
| `gc_buffer__add_type_qname(self, value, type, prog)` | Append the qualified type name (module::type) |

### Slot Serialization

| Function | Description |
|----------|-------------|
| `gc_buffer__add_slot(self, slot, type, prog)` | Append a slot value as text |
| `gc_buffer__add_slot_as_json(self, slot, type, prog)` | Append a slot value as JSON |
| `gc_buffer__add_slot_as_binary(self, slot, type, ctx)` | Append a slot value in binary (GCB) format |
| `gc_buffer__add_as_json(self, slot, type, prog)` | Append a slot as JSON (alias variant) |
| `gc_buffer__add(self, slot, type, prog)` | Append a slot using current buffer options |

### Pretty Printing

| Function | Description |
|----------|-------------|
| `gc_buffer__pretty_print_new_line(self, offset)` | Add newline + indentation at the given level |

### File I/O

| Function | Description |
|----------|-------------|
| `gc_buffer__read_all_from_fd(buf, fd)` | Read everything from a file descriptor into the buffer. Returns `true` on success. |

### Inline Read/Write (Fixed-Width Binary)

These are `static inline` functions for zero-overhead binary serialization. Call `gc_buffer_write_check(buf, len)` before writing to ensure capacity. They advance `buf->current`.

**Writers:**

| Function | Type Written |
|----------|-------------|
| `gc_buffer_write_bool` | `bool` (1 byte) |
| `gc_buffer_write_u8` / `gc_buffer_write_i8` | 1 byte |
| `gc_buffer_write_u16` | 2 bytes |
| `gc_buffer_write_u32` / `gc_buffer_write_i32` | 4 bytes |
| `gc_buffer_write_u64` / `gc_buffer_write_i64` / `gc_buffer_write_f64` | 8 bytes |
| `gc_buffer_write_slot` | 8 bytes (raw `gc_slot_t`) |
| `gc_buffer_write_vu32` | Variable-length encoded `u32_t` (1-5 bytes, LEB128 style) |
| `gc_buffer_write_vu64` | Variable-length encoded `u64_t` (1-9 bytes, LEB128 style) |
| `gc_buffer_write_vi64` | Variable-length zig-zag encoded `i64_t` (1-9 bytes) |

**Readers:**

| Function | Type Read |
|----------|-----------|
| `gc_buffer_read_bool` | `bool` |
| `gc_buffer_read_u8` / `gc_buffer_read_i8` | 1 byte |
| `gc_buffer_read_u16` | 2 bytes |
| `gc_buffer_read_u32` / `gc_buffer_read_i32` | 4 bytes |
| `gc_buffer_read_u64` / `gc_buffer_read_i64` / `gc_buffer_read_f64` / `gc_buffer_read_f32` | 4 or 8 bytes |
| `gc_buffer_read_vu32` | Variable-length `u32_t` |
| `gc_buffer_read_vu32_size_checked` | Like `gc_buffer_read_vu32` but returns `false` on buffer overflow |
| `gc_buffer_read_vu64` | Variable-length `u64_t` |
| `gc_buffer_read_vi64` | Variable-length zig-zag `i64_t` |

**Raw pointer read/write macros:**

```c
gc_buffer_read_ptr(buf, target, len)    // memcpy from buffer cursor to target
gc_buffer_write_ptr(buf, value, len)    // memcpy from value to buffer cursor (NULL-safe)
```

### Buffer Boundary Check

```c
// Returns true if reading `len` bytes at `current` would exceed `size`
static inline bool gc_buffer_unavailable(gc_buffer_t *buf_ptr, u64_t len);
```

### Non-Inline Write Functions

These are non-inline versions that handle capacity management internally:

| Function | Description |
|----------|-------------|
| `gc_buffer__write_u8` | Write `u8_t` with auto-prepare |
| `gc_buffer__write_bool` | Write `bool` with auto-prepare |
| `gc_buffer__write_u16` | Write `u16_t` with auto-prepare |
| `gc_buffer__write_u32` | Write `u32_t` with auto-prepare |
| `gc_buffer__write_u64` | Write `u64_t` with auto-prepare |
| `gc_buffer__write_u64_at(buf, value, off)` | Write `u64_t` at a specific offset |
| `gc_buffer__write_f64` | Write `f64_t` with auto-prepare |
| `gc_buffer__write_vu32` | Write variable-length `u32_t` with auto-prepare |
| `gc_buffer__write_vu64` | Write variable-length `u64_t` with auto-prepare |
| `gc_buffer__write_vi64` | Write variable-length zig-zag `i64_t` with auto-prepare |
| `gc_buffer__write_ptr(buf, data, len)` | Write raw bytes with auto-prepare |

### Terminal Color Constants

The buffer header defines ANSI color escape codes for terminal-colored output:

| Constant | Color / Purpose |
|----------|----------------|
| `GC_COLOR_LIGHT_GREEN` | Light green |
| `GC_COLOR_RESET` | Reset colors |
| `GC_COLOR_LIGHT_GREY` | Light grey |
| `GC_COLOR_BCYN` | Bold cyan |
| `GC_COLOR_BRED` | Bold red |
| `GC_COLOR_FN` | Blue (function names) |
| `GC_COLOR_TYPE` | Orange (type names) |
| `GC_COLOR_BOOL` | Purple (boolean values) |
| `GC_COLOR_NUMBER` | Purple (numeric values) |
| `GC_COLOR_TIME` | Purple (time values) |
| `GC_COLOR_NODE` | Purple (node references) |
| `GC_COLOR_DURATION` | Purple (duration values) |
| `GC_COLOR_ENUM` | Purple (enum values) |

---

<a id="gcstring-h"></a>
## gc/string.h — String Objects

Heap-allocated, hash-cached strings. GreyCat strings are immutable objects with a flexible array member for the character data.

### Structure

```c
typedef struct {
    gc_object_t header;  // Object header
    u32_t size;          // String length in bytes (not including null terminator)
    u64_t hash;          // Precomputed hash value (0 = not yet computed)
    char buffer[];       // Flexible array member holding the string bytes
} gc_string_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_string__create_from` | `gc_string_t *gc_string__create_from(const char *str, u64_t len)` | Create a new string from a raw buffer and length |
| `gc_string__create_concat` | `gc_string_t *gc_string__create_concat(const char *str, u64_t len, const char *str2, u64_t len2)` | Create a new string by concatenating two raw buffers |
| `gc_string__is_lit` | `bool gc_string__is_lit(const gc_string_t *str)` | Returns `true` if the string is a literal symbol (interned in the program's symbol table, not heap-allocated) |
| `gc_string__hash` | `u64_t gc_string__hash(const char *str, u32_t len)` | Compute hash of a raw character buffer |
| `gc_string__create_from_or_symbol` | `gc_string_t *gc_string__create_from_or_symbol(const gc_program_t *prog, const char *str, u64_t len)` | Lookup the string in the program's symbol table; return the interned symbol if found, otherwise allocate a new string |

---

<a id="gcstr-h"></a>
## gc/str.h — Inline Short Strings

Compact encoding that packs short strings (up to ~7 bytes) directly into a `u64_t` value. This avoids heap allocation for very short strings and is used with `gc_type_str`.

### Type

```c
typedef u64_t gc_str_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_core_str__encode` | `u64_t gc_core_str__encode(const char *buf, u32_t len)` | Encode a short string into a `u64_t` |
| `gc_core_str__add_to_buffer` | `void gc_core_str__add_to_buffer(u64_t value, gc_buffer_t *buf)` | Decode and append the inline string to a buffer |

---

<a id="gcobject-h"></a>
## gc/object.h — Object Manipulation

Functions for getting/setting fields on GreyCat objects, managing GC marks, serialization, and type checking.

### Object Data Layout

Objects store their fields in a segmented layout: data slots, nullable bitset, and any-type fields.

```c
typedef struct gc_object_data {
    u8_t *nullable_bitset;   // Bitset tracking which nullable fields are non-null
    u8_t *any_fields;        // Type tags for fields declared as `any`
    gc_slot_t *data;         // Array of field values
} gc_object_data_t;

gc_object_data_t gc_object__segments(const gc_object_t *self, const gc_program_type_t *type);
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_object_bitset_block_size` | 8 | Number of bits per block in the nullable bitset |

### Nullable Bitset Macros

```c
gc_object__set_not_null(bitset, offset)  // Mark a nullable field as non-null
gc_object__set_null(bitset, offset)      // Mark a nullable field as null
gc_object__is_not_null(bitset, offset)   // Check if a nullable field is non-null
```

### Field Type Resolution Macro

```c
// Resolve the runtime type of a field. For dynamically-typed (any) fields,
// reads from the any_fields array. For statically-typed fields, uses the
// field's declared sbi_type directly.
gc_object__field_to_type(field)
```

### Field Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_object__get_at` | `gc_slot_t gc_object__get_at(const gc_object_t *self, u32_t offset, gc_type_t *type_res, const gc_machine_t *ctx)` | Get a field value by its field offset. Writes the field type to `*type_res`. |
| `gc_object__set_at` | `bool gc_object__set_at(gc_object_t *self, u32_t offset, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Set a field value by offset. Returns `false` on error. |
| `gc_object__get_offset` | `gc_slot_t gc_object__get_offset(const gc_object_t *self, u32_t offset, gc_type_t *type_res, const gc_program_t *prog)` | Get a field value using the program directly (no machine context). |

### GC & Lifecycle

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_object__mark` | `void gc_object__mark(gc_object_t *self)` | Mark the object as reachable (prevent GC collection) |
| `gc_object__un_mark` | `void gc_object__un_mark(gc_object_t *self, gc_machine_t *ctx)` | Unmark the object (allow GC to collect it if unreachable) |
| `gc_object__declare_dirty` | `void gc_object__declare_dirty(gc_object_t *self)` | Mark the object as modified (for persistence layer) |
| `gc_object__is_instance_of` | `bool gc_object__is_instance_of(const gc_object_t *self, u32_t of_type, gc_machine_t *ctx)` | Check if the object is an instance of a given type (supports inheritance) |
| `gc_object__finalize` | `void gc_object__finalize(gc_object_t *self, gc_machine_t *ctx)` | Finalize (destroy) the object |
| `gc_object__create` | `gc_object_t *gc_object__create(const gc_object_type_t *type)` | Create a new object of the given type |

### Slot Serialization

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_slot__save` | `void gc_slot__save(gc_slot_t slot, gc_type_t slot_type, gc_buffer_t *buffer, const gc_machine_t *ctx, bool finalize)` | Serialize a slot to binary format (type byte + value) |
| `gc_slot__save_value` | `void gc_slot__save_value(gc_slot_t slot, gc_type_t slot_type, gc_buffer_t *buffer, const gc_machine_t *ctx, bool finalize)` | Serialize a slot value only (no type byte) |
| `gc_slot__load` | `gc_type_t gc_slot__load(gc_slot_t *slot, gc_block_t *owner, gc_buffer_t *buffer, const gc_machine_t *ctx)` | Deserialize a slot from binary format (reads type byte + value). Returns the type. |
| `gc_slot__load_value` | `gc_type_t gc_slot__load_value(gc_type_t t, gc_slot_t *slot, gc_block_t *owner, gc_buffer_t *buffer, const gc_machine_t *ctx)` | Deserialize a slot value when the type is already known (skips type byte). |

---

<a id="gcmachine-h"></a>
## gc/machine.h — Machine (Execution Context)

The `gc_machine_t` is the execution context passed to all native functions. It provides access to function parameters, result setting, error reporting, program introspection, and object creation.

### Context Type

```c
typedef struct {
    const gc_program_t *prog;
} gc_ctx_t;
```

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

#### Program & Object

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__program` | `const gc_program_t *gc_machine__program(const gc_machine_t *ctx)` | Get the program associated with this machine |
| `gc_machine__create_object` | `gc_object_t *gc_machine__create_object(const gc_machine_t *ctx, u32_t object_type_code)` | Create a new object of a specific type by type ID |
| `gc_machine__get_buffer` | `gc_buffer_t *gc_machine__get_buffer(gc_machine_t *ctx)` | Get a reusable scratch buffer (worker-local) |
| `gc_machine__current_fn_off` | `u32_t gc_machine__current_fn_off(gc_machine_t *ctx)` | Get the current function's offset in the program |
| `gc_machine__get_host` | `gc_host_t *gc_machine__get_host(gc_machine_t *ctx)` | Get the host managing this machine |
| `gc_machine__log_level` | `gc_log_level_t gc_machine__log_level(gc_machine_t *ctx)` | Get the current log level |

#### Advanced

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_machine__call_function` | `bool gc_machine__call_function(gc_machine_t *ctx, gc_program_function_body_t *body, gc_slot_t self, gc_type_t self_type, const gc_slot_t *params, const gc_type_t *params_type, u32_t nb_params, gc_slot_t *marked_res, gc_type_t *marked_res_type)` | Call a GCL function from C. The result is marked (caller must unmark). |
| `gc_machine__push_function` | `void gc_machine__push_function(gc_machine_t *ctx, const gc_program_function_t *fn, gc_slot_t self, gc_type_t self_type, const gc_task_t *task)` | Push a function frame onto the execution stack |
| `gc_machine__load` | `gc_type_t gc_machine__load(gc_machine_t *ctx, char *data, u32_t len, gc_slot_t *value)` | Deserialize a slot from raw binary data |
| `gc_machine__init_tensor` | `gc_core_tensor_t *gc_machine__init_tensor(gc_core_tensor_descriptor_t desc, gc_object_t *proxy, char *data, const gc_machine_t *ctx)` | Create a tensor with a pre-built descriptor and data pointer |
| `gc_machine__lru_add` | `void gc_machine__lru_add(gc_machine_t *ctx, gc_block_t *page)` | Add a block to the LRU cache |

---

<a id="gcprogram-h"></a>
## gc/program.h — Program & Type System

The program is the compiled representation of a GreyCat project. It contains all types, functions, modules, symbols, opcodes, and the ABI. This header defines the full program structure, the type system, the bytecode opcodes, and lookup/resolution functions.

### Key Structures

#### gc_program_t

The top-level compiled program:

```c
struct gc_program {
    gc_abi_t *abi;                            // Application Binary Interface
    struct { ... } symbols;                   // Symbol table (interned strings)
    gc_buffer_t fragments;                    // String fragments for doc comments, paths, etc.
    struct { ... } libraries;                 // Loaded native libraries
    struct { ... } modules;                   // Module table with map
    struct { ... } permissions;               // Permission definitions
    struct { ... } roles;                     // Role definitions
    struct { ... } expose_names;              // Exposed API name mappings
    gc_program_type_table_t types;            // All types
    gc_program_function_table_t functions;    // All functions
    struct { ... } ops;                       // Bytecode operations and source maps
    gc_program_specialized_type_table_t s_types;  // Specialized (generic) types
    gc_block_t stub;                          // Stub block for volatile objects
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
    struct { ... } fields;         // Instance fields (map + table)
    struct { ... } static_fields;  // Static fields (enum values, constants)
    struct { ... } functions;      // Methods (map + offset table)
    u32_t g1, g2;                  // Generic type parameters
    u32_t super_type;              // Parent type offset
    u32_t super_type_s_type_off;   // Super type specialized type offset
    u32_t companion_type;          // Companion type offset
    u32_t super_types[4];          // Inheritance chain (max depth 4)
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
    u32_t name_off;             // Symbol offset for the function name
    u32_t type_off;             // Owning type offset (0 for module-level functions)
    u32_t mod_off;              // Module offset
    u32_t op_off;               // Bytecode start offset
    u32_t type_local_off;       // Type-local function offset
    u32_t generic_type_symb;    // Generic type symbol
    u32_t doc_comment_fragment_off; // Documentation comment fragment offset
    u32_t return_type_desc;     // Return type descriptor
    u32_t return_s_type_off;    // Return specialized type offset
    u32_t expose_rename_off;    // Exposed name override symbol offset
    u8_t nb_params;             // Number of parameters
    bool is_static;             // Static method
    bool is_native;             // Implemented in C
    bool is_private;            // Private visibility
    bool is_reserved;           // Reserved/internal function
    bool is_exposed;            // Exposed via HTTP (@expose annotation)
    bool is_test;               // Test function
    bool is_abstract;           // Abstract method
    u64_t permissions_mask;     // Required permissions bitmask
    gc_program_src_id_t source; // Source location
    gc_program_function_body_t *body;  // Native function pointer (NULL for GCL functions)
    u32_t params_type_desc[16]; // Parameter type descriptors
    u32_t params_s_type_off[16]; // Parameter specialized type offsets
    u32_t signature_symbols[16]; // Parameter name symbols
    u32_t tags[2];              // Compiler tags (max 2)
};
```

#### gc_program_type_field_t

Field descriptor for a type's instance fields:

```c
typedef struct gc_program_type_field {
    u32_t name_off;                // Symbol offset for the field name
    u32_t type_desc;               // Type descriptor of the field
    gc_type_t sbi_type;            // Serialization Binary Interface type tag
    u32_t any_offset;              // Offset in the any_fields array (for dynamically-typed fields)
    u32_t nullable_offset;         // Offset in the nullable bitset
    u32_t s_type_off;              // Specialized type offset
    u32_t doc_comment_fragment_off; // Documentation comment fragment offset
    bool is_private;               // Private visibility
    u8_t precision;                // Numeric precision (gc_abi_precision_t)
    gc_program_src_id_t src_id;    // Source location
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
    struct { ... } functions;    // Function map + offset table
    struct { ... } vars;         // Module-level variables
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
<summary>Full opcode list (84 opcodes)</summary>

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
| 37-46 | `push_arr` / `push_map` / `push_table` / `push_obj` / `push_tuple` / `push_node` / `push_node_time` / `push_node_list` / `push_node_index` / `push_node_geo` | Literal construction |
| 47 | `push_const_def` | Push constant definition |
| 48-50 | `call_fn` / `call_met` / `call_end` | Function/method calls |
| 51 | `ret` | Return from function |
| 52 | `not` | Logical NOT |
| 53 | `noop` | No operation |
| 54-58 | `try` / `untry` / `throw` / `catch_a` / `catch_p` | Exception handling |
| 59-64 | `eq` / `ne` / `lt` / `le` / `gt` / `ge` | Comparison operators |
| 65-66 | `uminus` / `unref` | Unary minus / dereference |
| 67-74 | `add` / `sub` / `mul` / `div` / `mod` / `pow` / `and` / `or` | Arithmetic & logical |
| 75-76 | `for_st` / `for_do` | For-loop start/body |
| 77 | `breakpoint` | Debugger breakpoint |
| 78-81 | `break` / `break_for_in` / `continue` / `continue_for_in` | Loop control flow |
| 82 | `volatile_mod` | Volatile module access |
| 83 | `jol` | Jump on log level greater than machine log level |

</details>

### Operator Enum (`gc_program_operator_t`)

Maps to the binary operator opcodes. Values 0-18 covering: `not`, `uminus`, `unref`, `+`, `-`, `/`, `*`, `%`, `**`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `||`, `&&`, `is`, `as`.

### Iterator Parameters (`gc_program_iterator_param_t`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `from` | Start key (inclusive) |
| 1 | `to` | End key (inclusive) |
| 2 | `skip` | Number of entries to skip |
| 3 | `limit` | Maximum entries to return |
| 4 | `sampling` | Sampling rate |
| 5 | `nullable` | Include nullable entries |
| 6 | `from_excl` | Start key (exclusive) |
| 7 | `to_excl` | End key (exclusive) |

### Program Functions

#### Program Linking

| Function | Description |
|----------|-------------|
| `gc_program__link_mod_fn(prg, module_id, function, fn_name_symbol)` | Link a native function to a module-level function slot |
| `gc_program__link_type_fn(prg, type_id, function, fn_name_symbol)` | Link a native function to a type method slot |
| `gc_program_function__set_body(prog, fn_off, fn_body)` | Set the native body of a function |
| `gc_program_function__is_native_without_body(prog, fn_off)` | Check if a native function has no body yet |

#### Symbol Resolution

| Function | Description |
|----------|-------------|
| `gc_program__get_symbol(program, symb_off)` | Get a symbol string by its offset |
| `gc_program__get_symbol_off(symb)` | Get the symbol offset from a string pointer |
| `gc_program__resolve_symbol(program, str, len)` | Resolve a raw string to a symbol offset |
| `gc_program__resolve_module(program, mod_name_offset)` | Resolve a module by name offset |
| `gc_program__resolve_type(prog, mod_offset, type_name_off)` | Resolve a type within a module |
| `gc_program__resolve_field(type, name_off, res)` | Resolve a field within a type |

#### Type Introspection

| Function | Description |
|----------|-------------|
| `gc_program__get_program_type(prog, type_id)` | Get the full program type by ID |
| `gc_program_type__get_field(type, field_offset)` | Get a field descriptor by offset |
| `gc_program_type__get_field_by_key(type, key)` | Get a field descriptor by symbol key |
| `gc_program_type__get_g1_type_id(type)` / `gc_program_type__get_g2_type_id(type)` | Get generic type parameter IDs |
| `gc_program_type__get_g1_type_desc(type)` / `gc_program_type__get_g2_type_desc(type)` | Get generic type parameter descriptors |
| `gc_program_type__get_generic_id(type)` | Get the generic type ID |
| `gc_program_type__configure(prog, type_id, header_bytes, native_finalize)` | Configure native type (header size and destructor) |
| `gc_program_type__abi_type_id(prog, type_id)` | Get the ABI type ID for a program type |

#### Function Introspection

| Function | Description |
|----------|-------------|
| `gc_program__get_function(prog, fn_off)` | Get a function descriptor by offset |
| `gc_program_function__module_name_off(prog, fn_off)` | Get the module name offset |
| `gc_program_function__type_name_off(prog, fn_off)` | Get the type name offset |
| `gc_program_function__function_name_off(prog, fn_off)` | Get the function name offset |
| `gc_program_function__return_type_desc(prog, fn_off)` | Get the return type descriptor |
| `gc_program_function__get_param_by_off(fn, offset, out)` | Get parameter info by index |

#### Field Introspection

| Function | Description |
|----------|-------------|
| `gc_program_type_field__get_type_id(field)` | Get the type ID from a field descriptor |
| `gc_program_type_field__get_type_desc(field)` | Get the type descriptor from a field descriptor |
| `gc_program_type_field__get_name(field, prog)` | Get the field name as a `gc_string_t` |

#### Type Descriptor Utilities

| Function | Description |
|----------|-------------|
| `gc_type_desc__is_nullable(type_d)` | Check if a type descriptor marks the type as nullable |
| `gc_type_desc__to_type_id(type_d)` | Extract the type ID from a type descriptor |
| `gc_type_desc__to_type_desc(type_id, is_nullable)` | Build a type descriptor from a type ID and nullable flag |

#### Object & Library

| Function | Description |
|----------|-------------|
| `gc_program__create_object(program, type_code)` | Create a new object by type code |
| `gc_program_library__set_lib_hooks(lib, start, stop)` | Set library start/stop hooks |
| `gc_program_library__set_worker_hooks(lib, start, stop)` | Set worker start/stop hooks |
| `gc_lib_std__link(prg, lib)` | Link the standard library to a program |

#### Internal Map Operations

These are used internally for program symbol/type/function resolution:

| Function | Description |
|----------|-------------|
| `gc_program_map__put(self, hash, offset, key)` | Insert an entry into a program map |
| `gc_program_map__hash_off(offset)` | Compute a hash for an offset |
| `gc_program_map__get_key(self, key, hash, offset)` | Look up an entry by key and hash |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_compiler_tags_max` | 2 | Max compiler tags per function |
| `GC_PROGRAM_INHERITANCE_MAX_DEPTH` | 4 | Maximum type inheritance depth |
| `GC_PROGRAM_TEMPLATE_MAX_NESTED` | 4 | Maximum nested generic type depth |

---

<a id="gchost-h"></a>
## gc/host.h — Host & Task Management

The host manages the GreyCat runtime: program, workers, task queue, and external request dispatch. Use it to spawn background tasks, cancel them, and query task status from C native code.

### Data Format Enum

```c
typedef enum {
    gc_args_format_gcb  = 0,   // GreyCat binary format
    gc_args_format_json = 1,   // JSON format
    gc_args_format_text = 2,   // Plain text format
    gc_args_format_none = 3,   // No arguments
} gc_format_t;
```

### Task Status Enum

```c
typedef enum {
    gc_task_status_empty            = 0,  // Slot is empty
    gc_task_status_waiting          = 1,  // Queued, waiting to run
    gc_task_status_running          = 2,  // Currently executing
    gc_task_status_await            = 3,  // Suspended (awaiting I/O or sub-task)
    gc_task_status_cancelled        = 4,  // Cancelled by user
    gc_task_status_error            = 5,  // Completed with error
    gc_task_status_ended            = 6,  // Completed successfully
    gc_task_status_ended_with_errors = 7, // Completed with partial errors
    gc_task_status_breakpoint       = 8,  // Stopped at a breakpoint (debugging)
} gc_task_status_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_host__get_global` | `gc_host_t *gc_host__get_global()` | Get the global singleton host instance |
| `gc_host__spawn_task` | `bool gc_host__spawn_task(gc_host_t *self, u32_t fn_off, u32_t user_id, u64_t roles_flags, i64_t *created_task_id)` | Spawn a new task (no arguments). Returns `false` when the queue is full. |
| `gc_host__spawn_task_with_args` | `bool gc_host__spawn_task_with_args(gc_host_t *self, u32_t fn_off, const char *args_payload, u64_t args_payload_len, gc_format_t args_format, u32_t user_id, u64_t roles_flags, i64_t *created_task_id, gc_buffer_t *extra_buffer)` | Spawn a new task with serialized arguments |
| `gc_host__cancel_task` | `bool gc_host__cancel_task(gc_host_t *self, u32_t task_id)` | Cancel a running or queued task |
| `gc_host__get_task_status` | `bool gc_host__get_task_status(gc_host_t *self, u32_t task_id, gc_task_status_t *status)` | Query the current status of a task |
| `gc_host__program` | `const gc_program_t *gc_host__program(const gc_host_t *host)` | Get the program from the host |
| `gc_host__add_request` | `bool gc_host__add_request(u32_t fn, char *data, u32_t data_len)` | Add a request to the host's request queue |

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
| `gc_array__init` | `void gc_array__init(gc_array_t *self, u32_t capacity)` | Initialize the array with a given capacity |
| `gc_array__add_slot` | `bool gc_array__add_slot(gc_array_t *self, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Append an element to the end. Returns `false` on error. |
| `gc_array__set_slot` | `bool gc_array__set_slot(gc_array_t *self, u32_t offset, gc_slot_t value, gc_type_t type, gc_machine_t *ctx)` | Set the element at a given index |
| `gc_array__get_slot` | `bool gc_array__get_slot(const gc_array_t *self, u32_t offset, gc_slot_t *value, gc_type_t *type)` | Get the element at a given index. Returns `false` if out of bounds. |
| `gc_array__remove_at` | `bool gc_array__remove_at(gc_array_t *self, u32_t offset, gc_slot_t *result, gc_type_t *result_type, gc_machine_t *ctx)` | Remove the element at `offset` and return it. **The result is marked; caller must unmark.** |
| `gc_array__remove_all` | `void gc_array__remove_all(gc_array_t *self, gc_machine_t *ctx)` | Remove all elements from the array |
| `gc_array__swap` | `bool gc_array__swap(const gc_array_t *self, u32_t i, u32_t j)` | Swap two elements by index |
| `gc_array__sort` | `void gc_array__sort(gc_array_t *self, bool asc, gc_slot_tuple_u32_t field, gc_machine_t *ctx)` | Sort the array. `field` specifies a sub-field to sort by (for object arrays). |

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
| `gc_map__init` | `void gc_map__init(gc_map_t *self, u64_t capacity)` | Initialize the map with a given capacity |
| `gc_map__set` | `void gc_map__set(gc_map_t *self, gc_slot_t key, gc_type_t key_type, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Insert or update a key-value pair |
| `gc_map__get` | `gc_slot_t gc_map__get(const gc_map_t *self, gc_slot_t key, gc_type_t key_type, gc_type_t *value_type, const gc_program_t *prog)` | Lookup a value by key. Writes value type to `*value_type`. |
| `gc_map__contains` | `bool gc_map__contains(const gc_map_t *self, gc_slot_t key, gc_type_t key_type, const gc_program_t *prog)` | Check if a key exists in the map |
| `gc_map__remove` | `bool gc_map__remove(gc_map_t *self, gc_slot_t key, gc_type_t key_type, gc_machine_t *ctx)` | Remove an entry by key. Returns `true` if the key was found and removed. |

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
| `gc_table__create` | `gc_table_t *gc_table__create(const gc_machine_t *ctx)` | Create a new table |
| `gc_table__init` | `void gc_table__init(gc_table_t *table, u32_t capacity)` | Initialize with row capacity |
| `gc_table__init_cols` | `bool gc_table__init_cols(gc_table_t *self, u32_t cols)` | Set the number of columns (must be done before adding rows) |
| `gc_table__get_cell` | `bool gc_table__get_cell(const gc_table_t *self, i64_t row, i64_t col, gc_slot_t *value, gc_type_t *type)` | Read a cell value |
| `gc_table__set_cell` | `bool gc_table__set_cell(gc_table_t *self, i64_t row, i64_t col, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Write a cell value |
| `gc_table__set_row` | `bool gc_table__set_row(gc_table_t *self, i64_t row, gc_slot_t value, gc_type_t type, gc_machine_t *ctx)` | Set all cells in a row to the same value |
| `gc_table__remove_row` | `bool gc_table__remove_row(gc_table_t *self, u32_t row, gc_machine_t *ctx)` | Remove a row by index |

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
| `gc_core_tensor__diff(t1, t2, ctx)` | Compute the L2 difference between two tensors |
| `gc_core_tensor__check_shape(shape, tot_size, skip_zero)` | Validate a shape array |
| `gc_core_tensor__pos_to_offset(self, p[], ctx)` | Convert multi-dimensional position to flat offset |
| `gc_core_tensor__update_capacity(tensor, ctx)` | Reallocate data to match descriptor size |
| `gc_core_tensor__reset_internal(self)` | Reset internal state (descriptor + data pointer) |
| `gc_core_tensor__clone_internal(dst, src)` | Copy descriptor from src to dst (shallow) |
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

---

<a id="gcblock-h"></a>
## gc/block.h — Storage Blocks

Storage blocks are the persistence unit for GreyCat's graph database. They store key-value entries (`gc_block_slot_t`) and manage caching, dirty tracking, and compression state. Most SDK users interact with blocks indirectly through the object API.

### Types

```c
typedef u64_t gc_block_key_t;           // Block identification key (48 bits effective)
typedef u64_t gc_block_slot_key_t;      // Block entry indexing key
typedef u8_t gc_block_scale_id_t;       // Block scale identifier
```

### Block Slot (Entry)

```c
typedef struct gc_block_slot {
    gc_block_slot_key_t key;       // Entry key
    gc_slot_t value;               // Entry value
    u64_t deep_size : 48;          // Deep size in bytes (48-bit field)
    u8_t level : 8;                // B-tree level
    gc_type_t value_type : 8;      // Value type tag
} gc_block_slot_t;
```

### Block Structure

```c
struct gc_block {
    u32_t marks;               // GC and internal marks
    u32_t type;                // Block type
    gc_block_key_t key;        // Block key
    u64_t origin;              // Origin identifier
    struct {
        gc_block_t *cache_prev;  // LRU cache previous
        gc_block_t *cache_next;  // LRU cache next
    } meta;
    gc_block_scale_id_t scale; // Block scale
    u8_t level;                // B-tree level
    u8_t backend_refs;         // Backend reference count
    u8_t backend_unordered;    // Whether backend is unordered
    u32_t iterators;           // Active iterator count
    u16_t backend_size;        // Number of entries in backend
    u16_t backend_min_value;   // Minimum value index in backend
    u16_t backend_max_value;   // Maximum value index in backend
    u64_t deep_size;           // Total deep size
    u64_t cmp_deep_size;       // Compressed deep size
    u16_t cmp_start;           // Compression start index
    u16_t cmp_len;             // Compression length
    u16_t cmp_level;           // Compression level
    bool value_only : 1;      // Block stores values only (no sub-blocks)
    bool cmp_is_full : 1;     // Compression covers entire block
    bool cmp_with_values : 1; // Compressed with values
    bool cmp_up_to_date : 1;  // Compression is current
    bool cmp_found : 1;       // Compression data found
    bool dropped : 1;         // Block has been dropped
    bool contains_objects : 1; // Block contains object entries
    bool dirty : 1;           // Block has been modified since last save
    gc_block_slot_t *backend;  // Array of entries
};
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_block__is_stub` | `bool gc_block__is_stub(const gc_block_t *b)` | Check if the block is the program's stub (volatile placeholder) |
| `gc_block__object_attach` | `bool gc_block__object_attach(gc_block_t *b, gc_object_t **obj, gc_machine_t *ctx)` | Attach an object to this block (for persistence) |
| `gc_block__object_detach` | `void gc_block__object_detach(gc_block_t *b, gc_object_t *obj, gc_machine_t *ctx)` | Detach an object from this block |

---

<a id="gcabi-h"></a>
## gc/abi.h — ABI (Application Binary Interface)

The ABI defines the binary wire format for GreyCat's type system — how types, attributes, and functions are described in a version-controlled binary format. It enables safe schema evolution and cross-version compatibility.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_ABI_PROTO` | 2 | Current ABI protocol version |
| `GC_ABI_HEADER_SZ` | 8 | ABI header size in bytes |
| `GC_ABI_FUNCTION_MAX_PARAMS` | 16 | Maximum parameters per function |

### Types

```c
typedef u32_t gc_symbol_t;  // Symbol reference
```

### Precision Enum

```c
typedef enum {
    gc_abi_precision_1          = 0,
    gc_abi_precision_10         = 1,
    gc_abi_precision_100        = 2,
    gc_abi_precision_1000       = 3,
    gc_abi_precision_10000      = 4,
    gc_abi_precision_100000     = 5,
    gc_abi_precision_1000000    = 6,
    gc_abi_precision_10000000   = 7,
    gc_abi_precision_100000000  = 8,
    gc_abi_precision_1000000000 = 9,
    gc_abi_precision_10000000000 = 10
} gc_abi_precision_t;
```

Used for numeric precision/quantization (e.g., storing a float as an integer with known precision).

### ABI Attribute

```c
typedef struct {
    u32_t name;              // Attribute name (symbol offset)
    u32_t abi_type;          // ABI type ID
    u32_t prog_type_offset;  // Program type offset (computed on upgrade)
    u32_t mapped_any_offset; // Mapped any offset (computed on upgrade)
    u32_t mapped_att_offset; // Mapped attribute offset (computed on upgrade)
    u8_t sbi_type;           // Serialization Binary Interface type
    u8_t nullable;           // Whether the field is nullable
    u8_t mapped;             // Whether the field is mapped (computed on upgrade)
    u8_t precision;          // Numeric precision
} gc_abi_attribute_t;
```

### ABI Type

```c
typedef struct {
    u32_t module;                // Module name symbol
    u32_t name;                  // Type name symbol
    u32_t lib_name;              // Library name symbol
    u32_t generic_abi_type;      // Generic type ABI ID
    u32_t g1_abi_type_desc;      // First generic parameter type descriptor
    u32_t g2_abi_type_desc;      // Second generic parameter type descriptor
    u32_t parent_type_id;        // Parent type ABI ID (inheritance)
    u32_t companion_type_id;     // Companion type ABI ID
    u32_t attributes_len;        // Number of attributes
    u32_t attributes_offset;     // Offset into the attributes array
    u32_t mapped_prog_type_offset;  // (computed on upgrade)
    u32_t mapped_abi_type_offset;   // (computed on upgrade)
    u32_t masked_abi_type_offset;   // (computed on upgrade)
    u32_t nullable_nb_bytes;     // Bytes needed for nullable bitset
    u8_t is_native;              // Native C implementation
    u8_t is_abstract;            // Abstract type
    u8_t is_ambiguous;           // Ambiguous mapping
    u8_t is_enum;                // Enum type
    u8_t is_masked;              // Masked type
    u8_t is_volatile;            // Volatile type (not persisted)
} gc_abi_type_t;
```

### ABI Function

```c
typedef struct {
    u32_t module;
    u32_t type;
    u32_t name;
    u32_t lib_name;
    u32_t param_nb;
    u8_t param_nullable[16];     // Per-parameter nullable flags
    u32_t param_types[16];       // Per-parameter type descriptors
    u32_t param_symbols[16];     // Per-parameter name symbols
    u32_t return_type;           // Return type descriptor
    u8_t return_nullable;        // Whether return is nullable
} gc_abi_function_t;
```

### ABI Container

```c
typedef struct {
    gc_abi_symbols symbols;        // Symbol table
    gc_abi_attribute_t *attributes; // Attribute array
    gc_abi_type_t *types;          // Type array
    gc_abi_function_t *functions;   // Function array
    u32_t attributes_len;
    u32_t types_len;
    u32_t functions_len;
    u32_t version;                 // ABI version counter (incremented on schema changes)
    u16_t magic;                   // Magic number for validation
    u64_t crc;                     // CRC checksum
} gc_abi_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_abi__create` | `gc_abi_t *gc_abi__create()` | Allocate a new ABI instance |
| `gc_abi__finalize` | `void gc_abi__finalize(gc_abi_t *abi)` | Free all ABI memory |
| `gc_abi__save` | `void gc_abi__save(const gc_abi_t *abi, gc_buffer_t *output)` | Serialize the entire ABI to a buffer |
| `gc_abi__load` | `bool gc_abi__load(gc_abi_t *abi, gc_buffer_t *input)` | Deserialize an ABI from a buffer |
| `gc_abi__get_symbol` | `gc_string_t *gc_abi__get_symbol(const gc_abi_t *abi, gc_symbol_t symb)` | Retrieve a symbol string by its symbol ID |
| `gc_abi__save_header` | `void gc_abi__save_header(const gc_abi_t *abi, gc_buffer_t *b)` | Write only the ABI header to a buffer |
| `gc_abi__check_header` | `gc_abi_header_check_error_t gc_abi__check_header(const gc_abi_t *abi, gc_buffer_t *b)` | Validate an ABI header against the current ABI |

### Header Check Errors

```c
typedef enum {
    gc_abi_header_check_error_none          = 0,  // No error
    gc_abi_header_check_error_wrong_proto   = 1,  // Protocol version mismatch
    gc_abi_header_check_error_wrong_magic   = 2,  // Magic number mismatch
    gc_abi_header_check_error_wrong_version = 3   // ABI version mismatch
} gc_abi_header_check_error_t;
```

---

<a id="gcio-h"></a>
## gc/io.h — File I/O

Basic file I/O operations with platform-aware path separators.

### Constants

| Constant | Value (Unix) | Value (Windows) | Description |
|----------|-------------|----------------|-------------|
| `GC_PATH_SEP` | `'/'` | `'\\'` | Path separator character |
| `GC_PATH_SEP_S` | `"/"` | `"\\"` | Path separator string |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_io_file__open_rdwr` | `i32_t gc_io_file__open_rdwr(gc_string_t *path, gc_machine_t *ctx)` | Open a file for read/write. Returns file descriptor, or -1 on error. |
| `gc_io_file__open_read` | `i32_t gc_io_file__open_read(gc_string_t *path, gc_machine_t *ctx)` | Open a file for read-only. Returns file descriptor, or -1 on error. |
| `gc_io_open` | `i32_t gc_io_open(const gc_string_t *path, i32_t flags, gc_machine_t *ctx)` | Open a file with custom flags. Returns file descriptor. |
| `gc_io_file__sync` | `void gc_io_file__sync(i32_t fp)` | Flush file data to disk (fsync). |

---

<a id="gccrypto-h"></a>
## gc/crypto.h — Cryptography

SHA-256 hashing, HMAC-SHA-256, and Base64/Base64URL encoding/decoding.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_crypto_sha256_len` | 32 | SHA-256 digest length in bytes |

### SHA-256

```c
typedef struct {
    union {
        u32_t u32[8];
        unsigned char u8[gc_crypto_sha256_len];  // 32-byte hash
    } u;
} gc_crypto_sha256_t;

typedef struct {
    u32_t s[8];
    union {
        u32_t u32[16];
        unsigned char u8[64];
    } buf;
    size_t bytes;
} gc_crypto_sha256_ctx_t;
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__sha256` | `void gc_crypto__sha256(gc_crypto_sha256_t *sha, const void *p, size_t size)` | Compute SHA-256 hash of `p` (one-shot) |

### HMAC-SHA-256

```c
#define GC_CRYPTO_HMAC_SHA256_BLOCKSIZE 64

typedef struct {
    gc_crypto_sha256_t sha;
} gc_crypto_hmac_sha256_t;

typedef struct {
    gc_crypto_sha256_ctx_t sha;
    u64_t k_opad[8];  // HMAC outer padding
} gc_crypto_hmac_sha256_ctx_t;
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__hmac_sha256` | `void gc_crypto__hmac_sha256(gc_crypto_hmac_sha256_ctx_t *ctx, gc_crypto_hmac_sha256_t *hmac, const void *k, size_t ksize, const void *d, size_t dsize)` | Compute HMAC-SHA-256 with key `k` and data `d` |

### Base64 / Base64URL

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__base64_encode_len` | `u32_t gc_crypto__base64_encode_len(u32_t len)` | Compute output length for Base64 encoding |
| `gc_crypto__base64_encode` | `u32_t gc_crypto__base64_encode(const char *in, u32_t inlen, char *out)` | Encode to Base64. Returns output length. |
| `gc_crypto__base64_decode_len` | `u32_t gc_crypto__base64_decode_len(u32_t input_len)` | Compute output length for Base64 decoding |
| `gc_crypto__base64_decode` | `u32_t gc_crypto__base64_decode(const char *input, u32_t input_len, char *output)` | Decode from Base64. Returns output length. |
| `gc_crypto__base64url_encode_len` | `u32_t gc_crypto__base64url_encode_len(u32_t len)` | Compute output length for Base64URL encoding |
| `gc_crypto__base64url_encode` | `u32_t gc_crypto__base64url_encode(const char *str, u32_t str_len, char *output)` | Encode to Base64URL (URL-safe variant). Returns output length. |
| `gc_crypto__base64url_decode_len` | `u32_t gc_crypto__base64url_decode_len(u32_t input_len)` | Compute output length for Base64URL decoding |
| `gc_crypto__base64url_decode` | `u32_t gc_crypto__base64url_decode(const char *input, u32_t input_len, char *output)` | Decode from Base64URL. Returns output length. |

---

<a id="gcgeo-h"></a>
## gc/geo.h — Geospatial Operations

Geospatial encoding/decoding using a 64-bit geohash and Haversine distance calculation.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_CORE_GEO_LAT_MIN` | -85.05112878 | Minimum latitude (Web Mercator) |
| `GC_CORE_GEO_LAT_MAX` | 85.05112878 | Maximum latitude (Web Mercator) |
| `GC_CORE_GEO_LNG_MIN` | -180 | Minimum longitude |
| `GC_CORE_GEO_LNG_MAX` | 180 | Maximum longitude |
| `GC_CORE_GEO_LAT_EPS` | 0.00000001 | Latitude epsilon for comparison |
| `GC_CORE_GEO_STEP_MAX` | 32 | Maximum hash step resolution |
| `EARTH_RADIUS_M` | 6371000.0 | Earth radius in meters |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_geo__encode` | `geo_t gc_geo__encode(f64_t latitude, f64_t longitude)` | Encode lat/lng to a 64-bit geohash |
| `gc_geo__decode` | `void gc_geo__decode(geo_t hash, f64_t *lat, f64_t *lon)` | Decode a geohash back to lat/lng |
| `gc_geo__distance` | `f64_t gc_geo__distance(geo_t h1, geo_t h2)` | Compute the great-circle distance (in meters) between two geohashes |

---

<a id="gctime-h"></a>
## gc/time.h — Date & Time

Calendar conversion, timezone handling, ISO 8601 parsing, and formatting. GreyCat timestamps are microseconds since Unix epoch.

### Constants

**Time unit conversions (microseconds):**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_CORE_TIME_1_MILLISECOND` | 1,000 | 1ms in microseconds |
| `GC_CORE_TIME_1_SECOND` | 1,000,000 | 1s in microseconds |
| `GC_CORE_TIME_1_MINUTE` | 60,000,000 | 1min in microseconds |
| `GC_CORE_TIME_1_HOUR` | 3,600,000,000 | 1h in microseconds |
| `GC_CORE_TIME_1_DAY` | 86,400,000,000 | 1d in microseconds |

**Sub-second conversions:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_MICROSECONDS_IN_MILLISECOND` | 1,000 | Microseconds per millisecond |
| `GC_MILLISECONDS_IN_SECOND` | 1,000 | Milliseconds per second |
| `GC_MICROSECONDS_IN_SECOND` | 1,000,000 | Microseconds per second |

**Calendar constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_SECONDS_IN_MINUTE` | 60 | |
| `GC_SECONDS_IN_HOUR` | 3600 | |
| `GC_SECONDS_IN_DAY` | 86400 | |
| `GC_MINUTES_IN_HOUR` | 60 | |
| `GC_HOURS_IN_DAY` | 24 | |
| `GC_DAYS_IN_WEEK` | 7 | |
| `GC_MONTHS_IN_YEAR` | 12 | |
| `GC_DAYS_PER_ERA` | 146097 | Days in a 400-year era |
| `GC_DAYS_PER_CENTURY` | 36524 | Days in a 100-year period |
| `GC_DAYS_PER_4_YEARS` | 1461 | Days in a 4-year leap cycle (3×365+366) |
| `GC_DAYS_PER_YEAR` | 365 | Days in a non-leap year |
| `GC_DAYS_IN_JANUARY` | 31 | |
| `GC_DAYS_IN_FEBRUARY` | 28 | Days in non-leap February |
| `GC_YEARS_PER_ERA` | 400 | Years per era |
| `GC_YEAR_BASE` | 1900 | Base year for `gc_tm_t.tm_year` offset |

**Epoch adjustment constants (for internal calendar math):**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_EPOCH_ORIG_YEAR` | 70 | Unix epoch year offset (1970 - 1900) |
| `GC_EPOCH_ORIG_WEEK_DAY` | 4 | 1970-01-01 was a Thursday (0=Sunday) |
| `GC_EPOCH_ADJUSTMENT_DAYS` | 719468 | Days from 0000-03-01 to 1970-01-01 |
| `GC_ADJUSTED_EPOCH_YEAR` | 0 | Adjusted epoch year (Year 0, March 1) |
| `GC_ADJUSTED_EPOCH_WDAY` | 3 | 0000-03-01 was a Wednesday |
| `GC_MIN_YEAR` | `INT32_MIN + 1900` | Minimum representable year |
| `GC_MAX_YEAR` | `INT32_MAX + 1900` | Maximum representable year |

**Timezone conversion status codes:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_DTZ_OK` | 0 | Conversion succeeded |
| `GC_DTZ_ILLEGAL_TIMESTAMP` | 1 | Invalid or illegal timestamp |

**Days-in-month lookup:**

```c
static const int DAYS_IN_MONTH[12] = {31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
// Note: index 1 (February) is 0 — use GC_DAYS_IN_MONTH(m, y) macro instead
```

### Calendar Structure

```c
typedef struct {
    i32_t tm_sec;        // Seconds [0-60] (60 for leap second)
    i32_t tm_min;        // Minutes [0-59]
    i32_t tm_hour;       // Hours [0-23]
    i32_t tm_mday;       // Day of month [1-31]
    i32_t tm_mon;        // Month [0-11]
    i32_t tm_year;       // Year - 1900
    i32_t tm_wday;       // Day of week [0-6] (0 = Sunday)
    i32_t tm_yday;       // Day of year [0-365]
    i32_t tm_us_offset;  // Microsecond offset within the second
} gc_tm_t;
```

### Timezone Parse Result

```c
typedef struct {
    i32_t tz_hh_offset;   // Timezone hour offset
    i32_t tz_mm_offset;   // Timezone minute offset
    bool parsed;           // Whether a timezone was found in the string
} gc_core_time_parse_tz_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_time__now` | `i64_t gc_time__now()` | Get current time in microseconds since epoch |
| `gc_time__tm_from_time` | `gc_tm_t gc_time__tm_from_time(i64_t time, u32_t timezone)` | Convert a GreyCat timestamp to calendar fields |
| `gc_gmtime_r_safe` | `void gc_gmtime_r_safe(i64_t tim_p, gc_tm_t *res)` | Convert a timestamp to UTC calendar (safe, no timezone) |
| `gc_mktime_safe` | `i64_t gc_mktime_safe(gc_tm_t *tim_p)` | Convert calendar fields to a timestamp |
| `gc_dtz_utc_to_time_zone` | `i32_t gc_dtz_utc_to_time_zone(u32_t tz, i64_t utc_epoch, i64_t *localized_epoch)` | Convert UTC epoch to a localized epoch. Returns `GC_DTZ_OK` on success. |
| `gc_dtz_time_zone_to_utc` | `i32_t gc_dtz_time_zone_to_utc(u32_t tz, i64_t localized_epoch, i64_t *utc_epoch, u32_t *next_utc_offset)` | Convert localized epoch to UTC |
| `gc_dtz_first_day_of_week` | `u8_t gc_dtz_first_day_of_week(u32_t timezone)` | Get the first day of the week for a timezone (0=Sunday..6=Saturday) |
| `gc_core_time__parse_iso` | `bool gc_core_time__parse_iso(const char *c, size_t len, gc_tm_t *tm, gc_core_time_parse_tz_t *tz)` | Parse an ISO 8601 date/time string |
| `gc__print_iso` | `void gc__print_iso(gc_buffer_t *buffer, const gc_tm_t *cal, i64_t epoch_us, i64_t localized_epoch_s)` | Format a time as ISO 8601 string |
| `gc_strftime_safe` | `size_t gc_strftime_safe(char *s, size_t maxsize, const char *format, const gc_tm_t *t, i32_t utc_offset)` | Format a time with a `strftime`-style format string |

### Helper Macros

```c
GC_ISLEAP(y)              // Is year y a leap year?
GC_DAYS_IN_MONTH(m, y)    // Days in month m of year y
GC_DAYS_IN_YEAR(x)        // 365 or 366
GC_MODULO(x, y)           // True modulo (not C remainder)
```

---

<a id="gcmath-h"></a>
## gc/math.h — Math Functions (WASM)

On WASM targets, provides standalone implementations of standard math functions and constants (since `<math.h>` is unavailable). On native targets, simply includes `<math.h>` and `<stdlib.h>`.

### Constants (WASM only)

| Constant | Value | Description |
|----------|-------|-------------|
| `M_PI` | 3.14159265358979323846 | Pi (f64) |
| `PI` | 3.14159265f | Pi (f32) |
| `TAU` | 6.28318530f | 2*Pi (f32) |
| `HALF_PI` | 1.57079632f | Pi/2 (f32) |
| `E` | 2.71828182f | Euler's number (f32) |
| `LN2` | 0.69314718f | Natural log of 2 (f32) |
| `EPSILON` | 1.19209290e-7f | Machine epsilon for f32 |
| `SQRT2` | 1.41421356f | Square root of 2 (f32) |

### Functions (WASM only)

Standard math functions: `abs`, `fabs`, `fabsf`, `hypotf`, `hypot`, `sin`, `cos`, `tan`, `atan`, `atan2`, `sqrtf`, `sqrt`, `exp`, `pow`, `powl`, `log`, `log2`, `log2l`, `round`, `floor`, `ceil`, `ceill`, `isnan`, `div`, `llabs`, `labs`.

---

<a id="gcutil-h"></a>
## gc/util.h — Utility Functions

Miscellaneous utilities: Morton code (Z-order curve) encoding/decoding for spatial indexing, hex conversion, number parsing, JSON parsing, deep equality, slot hashing, duration parsing, sorting, and license level checking.

### Morton Code (Z-Order Curve)

These functions interleave/deinterleave integer coordinates into a single 64-bit key for spatial indexing.

**2D:**

| Function | Description |
|----------|-------------|
| `interleave64(xlo, ylo)` | Interleave two `u32_t` values into a `u64_t` |
| `deinterleave64(interleaved)` | Deinterleave a `u64_t` back into two `u32_t` |
| `gc_morton__deinterleave64_2di(interleaved, xlo)` | Deinterleave to two `i32_t` |
| `gc_morton__deinterleave64_2df(interleaved, xlo)` | Deinterleave to two `f32_t` |

**3D:**

| Function | Description |
|----------|-------------|
| `interleave64_3d(xlo)` | Interleave three `u32_t` values |
| `deinterleave64_3d(c, xlo)` | Deinterleave to three `u32_t` |
| `gc_morton__deinterleave64_3di(interleaved, xlo)` | Deinterleave to three `i32_t` |
| `gc_morton__deinterleave64_3df(interleaved, xlo)` | Deinterleave to three `f32_t` |
| `deinterleave64_3d_x0/x1/x2(c)` | Extract individual components |

**4D:**

| Function | Description |
|----------|-------------|
| `interleave64_4d(xlo)` | Interleave four `u16_t` values |
| `deinterleave64_4d(x, xlo)` | Deinterleave to four `u16_t` |
| `gc_morton__deinterleave64_4di(interleaved, xlo)` | Deinterleave to four `i16_t` |
| `gc_morton__deinterleave64_4df(interleaved, xlo)` | Deinterleave to four `f32_t` |
| `deinterleave64_4d_x0/x1/x2/x3(x)` | Extract individual components |

### Hex Conversion

| Function | Description |
|----------|-------------|
| `gc_common__hex2bin(dest, src, len)` | Convert hex string to binary bytes |
| `gc_common__hex2bin_len(len)` | Compute output length for hex-to-binary |
| `gc_common__bin2hex(dest, src, len)` | Convert binary bytes to hex string |
| `gc_common__bin2hex_len(len)` | Compute output length for binary-to-hex |

### Parsing

| Function | Description |
|----------|-------------|
| `gc_common__parse_number(str, str_len)` | Parse an unsigned integer from a string. Updates `*str_len` to bytes consumed. |
| `gc_common__parse_sign_number(str, str_len)` | Parse a signed integer from a string. Updates `*str_len` to bytes consumed. |
| `gc_common__parse_date_iso8601(data, len, epoch_utc)` | Parse an ISO 8601 date string to a UTC epoch (microseconds) |
| `gc_json__parse(ctx, str, len, result, type, type_d)` | Parse a JSON string into a GreyCat slot. Returns `true` on success. |
| `gc_duration__parse(str, len, duration)` | Parse a duration string (e.g., "1h30m") into microseconds |

### Deep Equality & Hashing

| Function | Description |
|----------|-------------|
| `gc__deep_equals(left, left_type, right, right_type, prog)` | Deep structural equality comparison of two slots |
| `gc_slot__hash(slot, type, prog)` | Compute a hash value for a slot |

### Sorting

```c
typedef struct gc_sort_slot {
    u64_t value;   // Sort key
    u64_t index;   // Original index
} gc_sort_slot_t;

void gc_sort__piposort(gc_sort_slot_t *array, u64_t nmemb, gc_type_t t, bool asc);
```

A stable sort implementation for arrays of sort slots.

### License

```c
typedef enum gc_license_level {
    gc_license_level_community = 0,
    gc_license_level_pro       = 1,
    gc_license_level_server    = 2,
    gc_license_level_platform  = 3
} gc_license_level_t;

gc_license_level_t gc_license__level();  // Get the current license level
```

---

## Conventions & Patterns

### Naming Conventions

- **Types:** `gc_{module}_{name}_t` (e.g., `gc_buffer_t`, `gc_core_tensor_t`)
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

- **Worker-local allocations** (`gc_gnu_malloc`, `gc_malloc`): Use for temporary data within a single task.
- **Global allocations** (`gc_global_gnu_malloc`): Use for data shared across workers.
- **Object creation** (`gc_machine__create_object`): Use for GreyCat objects — they are managed by the GC.
- **Mark/Unmark**: When you receive a marked object (e.g., from `gc_array__remove_at`), you must `gc_object__un_mark` it when you're done if you don't want to keep it alive.
