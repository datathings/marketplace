# GreyCat C SDK — Memory, Buffers, Strings & Objects

The memory substrate (allocator-first API), byte/text buffer construction, immutable heap strings, and object field access with GC mark/un-mark discipline.

_Part of the GreyCat C SDK reference (each file is linked from the skill's SKILL.md). Sibling references: api_core.md · api_memory_text.md · api_collections.md · api_runtime_storage.md · api_services.md._

## Contents

- [gc/alloc.h — Memory Allocation](#gcalloc-h)
- [gc/buffer.h — Buffer Operations](#gcbuffer-h)
- [gc/string.h — String Objects](#gcstring-h)
- [gc/object.h — Object Manipulation](#gcobject-h)

---

<a id="gcalloc-h"></a>
## gc/alloc.h — Memory Allocation

All non-trivial allocation goes through explicit `gc_allocator_t *` handles. The SDK provides the opaque allocator type and a `gc_alloc__*` family of functions. The shorter `gc_malloc` / `gc_free` / `gc_realloc` helpers route through the current thread-bound allocator. On WASM targets, `alloc.h` also declares standard C memory functions (`memset`, `memcpy`, `memcmp`, `strcmp`, `strncmp`, `memmove`, `strlen`).

### Opaque Type

```c
typedef struct gc_allocator gc_allocator_t;
```

### Allocator Lifecycle & Binding

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_alloc__create` | `gc_allocator_t *gc_alloc__create(bool shared)` | Create a new allocator. Pass `true` if it will be touched by multiple threads (jemalloc shared arena), `false` for a thread-private arena. |
| `gc_alloc__destroy` | `void gc_alloc__destroy(gc_allocator_t *allocator)` | Destroy an allocator created with `gc_alloc__create`. |
| `gc_alloc__bind` | `void gc_alloc__bind(gc_allocator_t *allocator)` | Bind the allocator to the current thread (thread-local binding for internal use). |
| `gc_alloc__allocated` | `u64_t gc_alloc__allocated(const gc_allocator_t *allocator)` | Return the number of bytes currently live in `allocator`'s accounting counter. |
| `gc_alloc__stats` | `void gc_alloc__stats(gc_allocator_t *allocator, const char *tag)` | Dump live allocation stats for `allocator` to stderr (jemalloc per-arena breakdown on jemalloc builds, otherwise just the gc_allocator_t counter). |

### Allocation Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_alloc__malloc` | `void *gc_alloc__malloc(gc_allocator_t *allocator, size_t size)` | Allocate `size` bytes from `allocator`. |
| `gc_alloc__calloc` | `void *gc_alloc__calloc(gc_allocator_t *allocator, size_t size)` | Allocate `size` bytes, zero-initialized. |
| `gc_alloc__free` | `void gc_alloc__free(gc_allocator_t *allocator, void *ptr, size_t size)` | Free memory back to `allocator`. The original allocation `size` MUST be passed. |
| `gc_alloc__size` | `size_t gc_alloc__size(gc_allocator_t *allocator, void *ptr)` | Return the usable size of the allocation `ptr` belongs to. |
| `gc_alloc__realloc` | `void *gc_alloc__realloc(gc_allocator_t *allocator, void *ptr, size_t old_size, size_t new_size)` | Resize allocation from `old_size` to `new_size`. |
| `gc_alloc__align_malloc` | `void *gc_alloc__align_malloc(gc_allocator_t *allocator, size_t size, size_t block_size)` | Allocate `size` bytes aligned to `block_size`. |
| `gc_alloc__align_free` | `void gc_alloc__align_free(gc_allocator_t *allocator, void *ptr, size_t size, size_t block_size)` | Free aligned memory. |

### Thread-Bound Convenience Helpers

These wrap the allocator currently bound to the calling thread via `gc_alloc__bind`. They are part of the public SDK; reach for them when you do not want to thread an allocator handle through every call site.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_malloc` | `void *gc_malloc(size_t size)` | Allocate via the thread-bound allocator. |
| `gc_free` | `void gc_free(void *ptr, size_t size)` | Free via the thread-bound allocator. The original `size` must be passed. |
| `gc_realloc` | `void *gc_realloc(void *ptr, size_t old_size, size_t new_size)` | Reallocate via the thread-bound allocator. |

### Allocator Selection — Required Rule

Inside a native function, pick the allocator whose lifetime matches the data:

| Allocator | Lifecycle | When to use |
|-----------|-----------|-------------|
| `((gc_ctx_t *)ctx)->allocator` (or `gc_machine__allocator(ctx)`) | Bound to the current native call / task scope. Reclaimed automatically. | Default for anything scoped to a single native function: scratch buffers, intermediate arrays, per-call result strings, any temporary memory that must not outlive the call. |
| `gc_host__allocator(gc_host__get_global())` (or `gc_host__global_allocator()`) | Plugin-global, persists across threads and native calls. You own the free. | Module-level state allocated in `lib_start` and freed in `lib_stop`: global indices, caches, precomputed lookup tables that are too large for `static const`, per-plugin singletons. Requires your own mutex when multiple workers can touch it concurrently. |

`gc_ctx_t` (see `gc/machine.h`) is `{ const gc_program_t *prog; gc_allocator_t *allocator; }`. The layout starts every `gc_machine_t`, so casting `(gc_ctx_t *)ctx` inside a native function — or calling `gc_machine__allocator(ctx)` — is the documented way to reach the per-call allocator.

### Example — Per-Call Scratch Inside a Native

```c
void gc_mymod_Foo__scratch(gc_machine_t *ctx) {
    gc_allocator_t *a = gc_machine__allocator(ctx);
    size_t sz = 4096;
    char *buf = (char *)gc_alloc__malloc(a, sz);
    // ... use buf (lives only for this call) ...
    gc_alloc__free(a, buf, sz);
}
```

### Example — Global State Across lib_start / lib_stop

```c
static my_state_t *g_state;         // plugin-global singleton
static gc_allocator_t *g_alloc;     // cached pointer for paired free

static bool lib_start(gc_unused gc_program_library_t *lib,
                      gc_unused gc_program_t *prog,
                      gc_unused void **user_data) {
    g_alloc = gc_host__global_allocator();
    g_state = (my_state_t *)gc_alloc__calloc(g_alloc, sizeof(my_state_t));
    pthread_mutex_init(&g_state->lock, NULL);
    return true;
}

static bool lib_stop(gc_unused gc_program_library_t *lib,
                     gc_unused gc_program_t *prog,
                     gc_unused void **user_data) {
    pthread_mutex_destroy(&g_state->lock);
    gc_alloc__free(g_alloc, g_state, sizeof(my_state_t));
    return true;
}
```

### Helper Macros

```c
// Frees only if non-NULL. Wraps gc_free(ptr, size) — uses the thread-bound allocator.
#define gc_free_not_null(x, x_size)                                            \
    do {                                                                       \
        if ((x) != NULL) {                                                     \
            gc_free((x), (x_size));                                            \
        }                                                                      \
    } while (0)
```

### Usage Examples

#### Growable buffer with `gc_alloc__realloc`

The realloc growth pattern always passes both the old and new byte sizes, then zeroes the freshly grown tail.

```c
typedef struct {
    block_t *data;                                 // your element type
    u64_t len;
    u64_t cap;
} block_array_t;

static void block_array__prepare(block_array_t *arr, u64_t nb_elems, gc_allocator_t *allocator) {
    u64_t new_cap = arr->cap == 0 ? 8 : arr->cap;
    while (new_cap < (arr->len + nb_elems)) {
        new_cap <<= 1;
    }
    if (arr->cap < new_cap) {
        u64_t old_cap = arr->cap;
        arr->data = gc_alloc__realloc(
            allocator, arr->data,
            old_cap * sizeof(block_t),
            new_cap * sizeof(block_t));
        // zero only the newly grown region
        memset(arr->data + old_cap, 0, (new_cap - old_cap) * sizeof(block_t));
        arr->cap = new_cap;
    }
}
```

#### Block-aligned allocation with `gc_alloc__align_malloc` / `gc_alloc__align_free`

Use the aligned pair for disk/IO buffers that must be aligned to a storage block size. The `block_size` passed to `gc_alloc__align_free` MUST match the value passed to `gc_alloc__align_malloc`.

```c
size_t block_size = 4096;                          // storage block size (typical disk page)
size_t cap = 64 * 1024;
u8_t *disk_buf = (u8_t *)gc_alloc__align_malloc(allocator, cap, block_size);

// ... perform aligned reads/writes into disk_buf ...

if (disk_buf != NULL) {
    gc_alloc__align_free(allocator, disk_buf, cap, block_size);
}
```

#### Standalone allocator lifecycle: `gc_alloc__create` / `gc_alloc__bind` / `gc_alloc__destroy`

Outside a native call (e.g. an embedding/CLI entry point) you create your own allocator, optionally bind it to the current thread so the `gc_malloc`/`gc_free` convenience helpers resolve to it, and destroy it at shutdown. Pass `true` to `gc_alloc__create` for an allocator shared across threads. Mirrors the global allocator setup in `main.c` and the per-worker arenas in `host.c`.

```c
gc_allocator_t *allocator = gc_alloc__create(true);  // shared arena
gc_alloc__bind(allocator);                            // thread-bound helpers now use it

// thread-bound convenience helpers route through the bound allocator:
size_t sz = 256;
char *scratch = (char *)gc_malloc(sz);
scratch = (char *)gc_realloc(scratch, sz, sz * 2);
gc_free(scratch, sz * 2);

gc_alloc__destroy(allocator);
```

#### Leak probing with `gc_alloc__allocated`

`gc_alloc__allocated` returns the live byte counter for an arena. Snapshot it before and after a unit of work to detect allocations that outlived their scope. Modeled on the per-module leak probe in `test.c`.

```c
gc_allocator_t *a = gc_machine__allocator(ctx);
const u64_t before = gc_alloc__allocated(a);

run_module_under_test(ctx);

const u64_t after = gc_alloc__allocated(a);
if (after != before) {
    const i64_t delta = (i64_t)after - (i64_t)before;
    fprintf(stderr, "leak: %lld bytes\n", (long long)delta);
    gc_alloc__stats(a, "post-test");   // dump per-arena breakdown to stderr
}
```


---

<a id="gcbuffer-h"></a>
## gc/buffer.h — Buffer Operations

A growable byte buffer used throughout GreyCat for serialization (binary, JSON, text), string building, I/O, and internal protocol encoding. The buffer maintains a write cursor (`current`) and supports both high-level append operations and low-level inline read/write of fixed-width primitives.

### Structure

```c
typedef struct gc_buffer {
    gc_object_t header;            // Object header (makes Buffer a first-class GreyCat object)
    char *data;                    // Pointer to the raw byte array
    u64_t capacity;                // Allocated capacity in bytes
    u64_t size;                    // Logical size (total written bytes)
    char *current;                 // Read/write cursor position
    gc_allocator_t *allocator;     // Allocator this buffer was created with (used for grow/finalize)
    gc_buffer_options_t options;   // Formatting options
} gc_buffer_t;
```

The buffer owns its allocator explicitly — which allocator gets used is determined by what was passed to `gc_buffer__create`.

### Buffer Options

```c
typedef struct {
    bool json;          // JSON output mode
    bool tty;           // Terminal (color) output mode
    bool pretty;        // Pretty-print with indentation
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
| `gc_buffer__create` | `gc_buffer_t *gc_buffer__create(gc_allocator_t *allocator)` | Allocate and initialize a new buffer bound to `allocator`. |
| `gc_buffer__finalize` | `void gc_buffer__finalize(gc_buffer_t *self)` | Release the buffer's data (keeps the buffer header). |
| `gc_buffer__finalize_ex` | `void gc_buffer__finalize_ex(gc_buffer_t *self)` | Extended finalize variant (releases any extra/embedded resources). |
| `gc_buffer__clear` | `void gc_buffer__clear(gc_buffer_t *self)` | Reset size to 0 (keeps allocation). |
| `gc_buffer__clear_secure` | `void gc_buffer__clear_secure(gc_buffer_t *self)` | Reset and zero-fill the whole allocation (for secrets). |
| `gc_buffer__prepare` | `void gc_buffer__prepare(gc_buffer_t *self, u64_t needed)` | Ensure at least `needed` extra bytes of capacity. |

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
| `gc_buffer__add_time(self, value)` | Append a time (microseconds since epoch) as a human-readable timestamp |
| `gc_buffer__add_duration(self, value)` | Append a duration in human-readable form |
| `gc_buffer__add_byte_size(self, value)` | Append byte size in human-readable IEC units (KiB, MiB...) |
| `gc_buffer__add_byte_size_si(self, value)` | Append byte size in SI units (KB, MB...) |
| `gc_buffer__add_symbol(self, symb_id, prog)` | Append a symbol name by its ID |
| `gc_buffer__add_protected_symbol(buf, symb_off, prog)` | Append symbol with non-alphanumeric replaced by `_` |
| `gc_buffer__add_escaped_symbol(buf, symb_off, prog)` | Append symbol with `"` escaped to `\"` |
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

### Usage Examples

All snippets below assume a `gc_host_t *host` (or `gc_machine_t *ctx`) is in scope. Buffers are bound to an allocator at creation and must be finalized with `gc_buffer__finalize`.

#### Building text and serializing to a file

A buffer created from the host allocator can be filled with mixed text/value appends, then its raw `data`/`size` written out. Note the `gc_buffer__add_pstr` macro for compile-time string literals and the manual `'\0'` append when the buffer is used as a C path:

```c
gc_buffer_t *buf = gc_buffer__create(gc_host__allocator(host));

gc_buffer__add_str(buf, data_dir, data_dir_len);   // const char* dir + its length, in scope
gc_buffer__add_char(buf, GC_PATH_SEP);             // public path separator from gc/io.h
gc_buffer__add_pstr(buf, "program");   // sizeof(literal)-1 length, no terminator
gc_buffer__add_char(buf, '\0');        // make buf->data a valid C string for open()

const i32_t fd = open(buf->data, O_RDONLY | O_BINARY);
if (fd == -1) {
    gc_buffer__finalize(buf);
    return false;
}
// ... use fd ...
gc_buffer__finalize(buf);
```

When emitting decimal numbers as text, use the typed appenders rather than formatting yourself:

```c
gc_buffer_t *buf = gc_buffer__create(gc_host__allocator(host));
gc_buffer__add_cstr(buf, "elapsed=");
gc_buffer__add_i64(buf, delta);        // signed decimal
gc_buffer__add_cstr(buf, "us f64=");
gc_buffer__add_f64(buf, ratio);        // double as decimal text
gc_buffer__finalize(buf);
```

#### Reading a whole file descriptor, then clearing for reuse

`gc_buffer__read_all_from_fd` slurps the entire fd into the buffer (advancing the OS file pointer). Reset the read cursor to `buf->data` before parsing, and call `gc_buffer__clear` to reuse the same allocation for the next read:

```c
gc_buffer_t *buf = gc_buffer__create(gc_host__allocator(host));
// (buf already used to build a path and open `fd` above)
gc_buffer__clear(buf);                 // keep capacity, reset size to 0
if (!gc_buffer__read_all_from_fd(buf, fd)) {
    fprintf(stderr, "read failed\n");
    gc_buffer__finalize(buf);
    return 1;
}
buf->current = buf->data;              // rewind cursor for reading
// Walk the slurped bytes with the inline readers (see below). For example,
// decode a leading varint length prefix:
u64_t payload_len = 0;
if (!gc_buffer_unavailable(buf, 1)) {
    gc_buffer_read_vu64(buf, &payload_len);
}
gc_buffer__finalize(buf);
```

#### Inline binary writing (LEB128 varints)

The `static inline gc_buffer_write_*` helpers do NOT grow the buffer themselves — you must reserve worst-case capacity with `gc_buffer_write_check(buf, len)` first (a `vu64` is at most 9 bytes), then write. After a batch of inline writes, sync `buf->size` from the advanced cursor. `gc_buffer_write_ptr` is NULL-safe and copies raw bytes:

```c
const u64_t reserved_bin_len = sizeof(u64_t) * nb_zones;
gc_buffer_write_check(buf, reserved_bin_len);
gc_buffer_write_ptr(buf, (char *) reserved_blocks_per_zones, reserved_bin_len);
buf->size = (buf->current - buf->data);

gc_buffer_write_check(buf, 9);         // worst case for one vu64
gc_buffer_write_vu64(buf, nb_tx_log_diff);
buf->size = (buf->current - buf->data);

// reserve for three varints at once, then write the record
gc_buffer_write_check(buf, 9 + 9 + 9);
gc_buffer_write_vu64(buf, bucket->key);
gc_buffer_write_vu64(buf, bucket->offset);
gc_buffer_write_vu64(buf, bucket->origin);
buf->size = (buf->current - buf->data);
```

#### Inline binary reading with bounds checks

On the read side, guard each read with `gc_buffer_unavailable(buf, len)` (returns `true` when fewer than `len` bytes remain at the cursor) before calling the inline reader. The reader advances `buf->current` and writes through the output pointer:

```c
u64_t root = 0, block_key_generator = 0, task_generator = 0;

if (gc_buffer_unavailable(in_buf, 1)) {
    return false;                      // truncated input
}
gc_buffer_read_vu64(in_buf, &root);

if (gc_buffer_unavailable(in_buf, 1)) {
    return false;
}
gc_buffer_read_vu64(in_buf, &block_key_generator);

if (gc_buffer_unavailable(in_buf, 1)) {
    return false;
}
gc_buffer_read_vu64(in_buf, &task_generator);
```

For untrusted varint streams where you cannot pre-check the exact byte count, `gc_buffer_read_vu32_size_checked` validates against `buf->size` internally and returns `false` on overrun:

```c
u32_t count = 0;
if (!gc_buffer_read_vu32_size_checked(in_buf, &count)) {
    return false;                      // overran the buffer
}
```

#### Serializing a slot to JSON

Given a slot and its runtime type, `gc_buffer__add_as_json` renders the value as JSON. Here it uses the machine's scratch buffer (`gc_machine__get_buffer(ctx)`), cleared before each use, then the result is materialized into a GreyCat string:

```c
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__clear(buf);
gc_buffer__add_as_json(buf, slot, type, gc_machine__program(ctx));
gc_object_t *text = (gc_object_t *) gc_string__create_from(buf->data, buf->size, ctx);
// ... store `text` into a field ...
gc_object__un_mark(text, ctx);
```


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
| `gc_string__create_from` | `gc_string_t *gc_string__create_from(const char *str, u64_t len, const gc_machine_t *ctx)` | Create a new string from a raw buffer and length (allocated via the call's allocator). |
| `gc_string__create_concat` | `gc_string_t *gc_string__create_concat(const char *str, u64_t len, const char *str2, u64_t len2, const gc_machine_t *ctx)` | Create a new string by concatenating two raw buffers. |
| `gc_string__is_lit` | `bool gc_string__is_lit(const gc_string_t *str)` | Returns `true` if the string is a literal symbol (interned in the program's symbol table, not heap-allocated). |
| `gc_string__hash` | `u64_t gc_string__hash(const char *str, u32_t len)` | Compute hash of a raw character buffer. |
| `gc_string__create_from_or_symbol` | `gc_string_t *gc_string__create_from_or_symbol(const gc_program_t *prog, const char *str, u64_t len, const gc_machine_t *ctx)` | Lookup the string in the program's symbol table; return the interned symbol if found, otherwise allocate a new string via the call's allocator. |

> Note: `gc_string_t.buffer` is NOT null-terminated. Use the `size` field for length.

### Usage Examples

**Build a result string from a raw buffer, then unmark it**

The most common pattern: create a `gc_string_t` from a byte buffer, hand it back as the call result, then immediately drop the temporary reference. The newly created object is born marked, so it must be unmarked once it has been published into a slot.

```c
// e.g. returning the hex encoding of a 16-byte buffer (uses gc/util.h)
u8_t bytes[16] = { /* ... */ };
char out[32];                                  // gc_common__bin2hex_len(16) == 32
gc_common__bin2hex(out, (const char *) bytes, 16);

gc_string_t *res = gc_string__create_from(out, sizeof(out), ctx);
gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *) res}, gc_type_object);
gc_object__un_mark((gc_object_t *) res, ctx);
```

When the source is a C string of unknown length, pair it with `strlen`:

```c
const char *cstr = "...";
gc_object_t *msg = (gc_object_t *) gc_string__create_from(cstr, strlen(cstr), ctx);
// ... store msg somewhere, then drop the temporary ref ...
gc_object__un_mark(msg, ctx);
```

**Intern short / repeating keys with `gc_string__create_from_or_symbol`**

For map keys, column names, content types, and other strings that frequently repeat, prefer `gc_string__create_from_or_symbol`. It returns the existing interned symbol (no allocation) when the program already knows the literal, and only allocates a fresh `gc_string_t` otherwise. It needs the program (`gc_machine__program(ctx)`) for the symbol-table lookup.

```c
// store a string-keyed entry in a map, interning the key (std/runtime/openapi.c)
static void map_set_string(gc_map_t *map, const char *key, u32_t key_len,
                           gc_object_t *value, gc_machine_t *ctx) {
    gc_object_t *key_obj =
        (gc_object_t *) gc_string__create_from_or_symbol(gc_machine__program(ctx), key, key_len, ctx);
    gc_map__set(map,
                (gc_slot_t){.object = key_obj}, gc_type_object,
                (gc_slot_t){.object = value}, gc_type_object,
                ctx);
    gc_object__un_mark(key_obj, ctx);
}

// parsing a CSV field straight into a slot (std/io/csv.c)
self->value_slot.object =
    (gc_object_t *) gc_string__create_from_or_symbol(gc_machine__program(ctx),
                                                     self->field_view.data,
                                                     self->field_view.len, ctx);
self->value_type = gc_type_object;
```

**Concatenate two buffers**

`gc_string__create_concat` builds a single new string from two `(buffer, len)` pairs — used by the VM's string `+` opcode, reading the `buffer`/`size` fields of each operand:

```c
// data[slot1_off] = lhs + rhs  (vm/machine.c, sadd_object opcode)
gc_string_t *p1 = (gc_string_t *) slot1.object;
gc_string_t *p2 = (gc_string_t *) slot2.object;
data[slot1_off].object = (gc_object_t *)
    gc_string__create_concat(p1->buffer, p1->size, p2->buffer, p2->size, ctx);
gc_object__un_mark((gc_object_t *) p1, ctx);
gc_object__un_mark((gc_object_t *) p2, ctx);
```

**Reading fields, hashing, and detecting interned literals**

Access content through the `buffer`/`size` fields directly (the buffer is NOT null-terminated). `gc_string__hash` computes the same hash GreyCat caches in `gc_string_t.hash`, and `gc_string__is_lit` tells you whether a string is an interned symbol (backed by the program) rather than a heap object — useful before deciding whether you may free or must copy it.

```c
// lazily fill the cached hash from the raw bytes (core/src/type.c)
if (str->hash == 0) {
    str->hash = gc_string__hash(str->buffer, str->size);
}

// treat interned literals differently from heap-allocated strings (std/core/type.c)
if (gc_string__is_lit(str)) {
    // symbol: lives in the program's symbol table, do not free
} else {
    // ordinary heap string: owned via the allocator / refcount
}
```


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
| `gc_object__clone` | `gc_object_t *gc_object__clone(gc_object_t *self, gc_machine_t *ctx)` | Deep-clone `self`, returning a new (marked) object; the caller owns the mark and must `gc_object__un_mark` it. |

> To create an object, use `gc_machine__create_object(ctx, type_id)` (or `gc_machine__create_return_type_object(ctx)`).

### Slot Serialization

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_slot__save` | `void gc_slot__save(gc_slot_t slot, gc_type_t slot_type, gc_buffer_t *buffer, const gc_machine_t *ctx, bool finalize)` | Serialize a slot to binary format (type byte + value) |
| `gc_slot__save_value` | `void gc_slot__save_value(gc_slot_t slot, gc_type_t slot_type, gc_buffer_t *buffer, const gc_machine_t *ctx, bool finalize)` | Serialize a slot value only (no type byte) |
| `gc_slot__load` | `gc_type_t gc_slot__load(gc_slot_t *slot, gc_block_t *owner, gc_buffer_t *buffer, const gc_machine_t *ctx)` | Deserialize a slot from binary format (reads type byte + value). Returns the type. |
| `gc_slot__load_value` | `gc_type_t gc_slot__load_value(gc_type_t t, gc_slot_t *slot, gc_block_t *owner, gc_buffer_t *buffer, const gc_machine_t *ctx)` | Deserialize a slot value when the type is already known (skips type byte). |

### Usage Examples

#### Populating an object's fields and releasing the temporary mark

Newly created objects (and any objects/strings you allocate to put in their
fields) come back GC-marked. Set the field with `gc_object__set_at`, then
un-mark the temporary you allocated — the parent object now retains it. The
`gc_<module>_<Type>` type ids and `gc_<module>_<Type>_<field>` field offsets
used below are emitted by `greycat codegen c` into your plugin's `nativegen.h`
(field offsets are generated for your own GCL types).

```c
// Build a mymod::Account object from raw values.
gc_object_t *result = gc_machine__create_object(ctx, gc_mymod_Account);

// Scalar fields: write the slot inline with its type tag.
gc_object__set_at(result, gc_mymod_Account_id, (gc_slot_t) {.i64 = (i64_t) user_id}, gc_type_int, ctx);

// Object fields: allocate, assign, then drop your local mark.
gc_string_t *name = gc_string__create_from(name_bytes, name_size, ctx);
gc_object__set_at(result, gc_mymod_Account_name, (gc_slot_t) {.object = (gc_object_t *) name}, gc_type_object, ctx);
gc_object__un_mark((gc_object_t *) name, ctx); // result now retains `name`

// Enum/static-field values use the tuple-u32 slot (type id, member id).
gc_object__set_at(
    result, gc_mymod_Account_role,
    (gc_slot_t) {.tu32 = (gc_slot_tuple_u32_t) {.left = gc_mymod_Role, .right = gc_mymod_Role_admin}},
    gc_type_static_field, ctx);

// `result` is returned to the caller still marked; the receiver owns the mark.
return result;
```

#### Reading a field with `gc_object__get_at`

`gc_object__get_at` writes the field's runtime type into `*type_res`; always
branch on that before touching the slot union.

```c
// Extract Record.label (an object field that may be null).
gc_type_t label_type = gc_type_null;
gc_slot_t label = gc_object__get_at(self, gc_mymod_Record_label, &label_type, ctx);
if (label_type == gc_type_object) {
    gc_string_t *label_str = (gc_string_t *) label.object;
    gc_buffer__add_str(buf, label_str->buffer, label_str->size);
}
```

#### Type-checking before extracting a field (`gc_object__is_instance_of`)

```c
// A Selector stores a `field` reference (tuple of {type id, field offset}).
gc_type_t field_type;
const gc_slot_t field_slot = gc_object__get_at(self, gc_mymod_Selector_field, &field_type, ctx);
if (field_type != gc_type_field) {
    return 0.0;
}
// Guard against a value whose type does not declare that field.
if (!gc_object__is_instance_of(value.object, field_slot.tu32.left, ctx)) {
    return 0.0;
}
gc_type_t extract_type;
gc_slot_t extract = gc_object__get_at(value.object, field_slot.tu32.right, &extract_type, ctx);
// Convert the slot to a double by branching on its runtime type.
switch (extract_type) {
case gc_type_float:
    return extract.f64;
case gc_type_int:
    return (f64_t) extract.i64;
default:
    return 0.0;
}
```

#### Reading a field without a machine context (`gc_object__get_offset`)

When you hold a `gc_program_t *` but no full machine, read fields directly off
the program. The type-out parameter still tells you whether the field is set.

```c
gc_type_t t = gc_type_null;
u64_t seed = (u64_t) gc_object__get_offset(gen, gc_mymod_Generator_seed, &t, gc_machine__program(ctx)).i64;
if (t == gc_type_null) {
    // field unset: fall back to a time-derived seed
    seed = (u64_t) gc_time__now() ^ (u64_t) (uintptr_t) gen;
}
```

#### Iterating fields via the segmented layout (`gc_object__segments`, `gc_object__field_to_type`)

`gc_object__segments` splits an object into its data/bitset/any-type segments so
you can walk every field generically. `gc_object__field_to_type` resolves each
field's concrete type (reading the `any_fields` segment for dynamically-typed
fields).

```c
const gc_object_data_t self_data = gc_object__segments(src.object, type);
for (u32_t i = 0; i < type->fields.table.size; i++) {
    const gc_program_type_field_t *field = &type->fields.table.data[i];
    const gc_type_t field_type = gc_object__field_to_type(field);
    // self_data.data[i] holds the i-th field's slot value
    if (field_type == gc_type_object && self_data.data[i].object != NULL) {
        // recurse / clone / inspect the referenced object
    }
}
```

#### Marking modifications for the persistence layer (`gc_object__declare_dirty`)

Any in-place mutation of a persisted object must flag it dirty so the storage
layer flushes it. This is the standard tail of mutating collection ops.

```c
void gc_mymod_MyList__add(gc_array_t *self, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx) {
    gc_array__add_slot(self, value, value_type, ctx);
    gc_object__declare_dirty((gc_object_t *) self);
}
```

#### Serializing and deserializing slots (`gc_slot__save`, `gc_slot__load`)

`gc_slot__save` writes a type byte followed by the value; `gc_slot__load`
reads it back, inferring and returning the type. Use `gc_slot__load_value`
(and `gc_slot__save_value`) when the type is fixed and known on both ends, to
skip the type byte.

```c
// Write: serialize an object result into a response buffer (type byte + value).
gc_slot__save((gc_slot_t) {.object = res}, gc_type_object, buf, ctx, false);

// Read back: type is inferred from the leading byte.
gc_slot_t slot;
gc_type_t slot_type = gc_slot__load(&slot, NULL, buf, ctx);
if (slot_type == gc_type_error) {
    // truncated / corrupt payload
}

// Known-type fast path inside a block load (no type byte on the wire):
gc_slot_t value;
if (gc_slot__load_value(all_slot_type, &value, owner, buffer, ctx) == gc_type_error) {
    return; // deserialization failed
}
```

#### Destroying an object (`gc_object__finalize`)

`gc_object__finalize` tears down an object once it is no longer reachable
(typically driven by the GC / block teardown path), as opposed to
`gc_object__un_mark` which merely releases a reference.

```c
gc_object__finalize(slot.object, ctx);
```


---
