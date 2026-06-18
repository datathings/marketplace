---
name: greycat-c
description: "GreyCat C API and GCL Standard Library reference. Use for: (1) Native C development with gc_machine_t context, tensors, objects, memory management, crypto, I/O; (2) GCL Standard Library modules - std::core (Date/Time/Tuple/geospatial types), std::runtime (Scheduler/Task/Logger/Identity/Security/System/License/OpenAPI/MCP), std::io (CSV/JSON/XML/HTTP/Email/FileWalker/S3), std::util (Queue/Stack/SlidingWindow/Gaussian/Histogram/Quantizers/Random/Uuid/Crypto/Plot); (3) Plugin development patterns - lifecycle hooks, type configuration, nativegen, module-level and type-level function linking, global state, thread safety, conditional logging. Keywords: GreyCat, GCL, native functions, tensors, task automation, scheduler, plugin development."
---

# GreyCat SDK - C API, Standard Library & Plugin Development

Comprehensive reference for GreyCat native development (C API), the GCL Standard Library, and plugin development patterns. Tracks SDK **8.0**.

## Key Considerations

- **Allocator API is mandatory.** Every non-trivial allocation routes through an explicit `gc_allocator_t *`. Per-call scratch comes from `gc_machine__allocator(ctx)` (or `((gc_ctx_t *)ctx)->allocator`); plugin-global state comes from `gc_host__global_allocator()` (a convenience wrapper around `gc_host__allocator(gc_host__get_global())`). `gc_alloc__create(bool shared)` takes a `shared` flag (`true` for multi-thread arenas, `false` for thread-private). `gc_alloc__size(allocator, ptr)` returns an allocation's usable size; `gc_alloc__allocated` / `gc_alloc__stats` give live-bytes accounting and debug dumps. The thread-bound helpers `gc_malloc` / `gc_free` / `gc_realloc` are public too — they target whichever allocator is bound to the calling thread via `gc_alloc__bind`.
- **Structured logging (`gc/log.h`).** `gc_log_level_t` is `none / error / warn / info / perf / trace`. Use `gc_log__machine` / `gc_log__machinef` (VM context) and `gc_log__host` / `gc_log__hostf` (host context); gate hot paths with `gc_log__enabled(host, level)`.
- **No inline short strings.** There is no `gc/str.h`: `gc_str_t` and the `gc_core_str` / `gc_core_t2…t4f` globals are not public. Use `gc_string_t` (heap, immutable, hash-cached; `buffer` is NOT NUL-terminated — use `size`).
- **`gc_machine__call_function` takes a `const gc_program_function_t *fn`** (not a raw body pointer). On `false` the result is a synthesized `Error` object (type `gc_core_Error`) and `*marked_res_type` is `gc_type_object`; the caller owns one mark on the result. `gc_machine__impersonate(ctx, user_id)` switches the effective user for permission-aware sub-calls; `gc_machine__allocator(ctx)` is the per-call allocator sugar.
- **Host/Scheduler (`gc/host.h`).** `gc_host__cancel_task` / `gc_host__get_task_status` take `i64_t task_id`. Periodic scheduling via `gc_scheduler_t`, `gc_periodic_task_t`, `gc_periodicity_t` (fixed / daily / weekly / monthly / yearly), `gc_scheduler__add/activate/deactivate/create_object`.
- **ABI.** `GC_ABI_PROTO` is `3`. `gc_abi_header_check_error_t` includes `..._truncated = 4`; `gc_abi_t` carries its own allocator.
- **Iterator params** `gc_program_iterator_param_t`: `from=0`, `to=1`, `nullable=2`, `from_excl=3`, `to_excl=4` (no `limit`). Geo epsilon constant is `GC_CORE_GEO_EPS`.
- **Stdlib (GreyCat 8.0).** Security model is `Identity` / `IdentityGrant` / `IdentityGrantType` (the old `User` / `UserGroup` / `SecurityPolicy` / `OpenIDConnect` types are gone). GCL logging is via module-level `info` / `warn` / `error` / `perf` / `trace` functions — `Log` is a parse record, not a callable namespace. `std::io` has S3 object storage (`S3`, `S3Bucket`, `S3Object`, `S3BasicCredentials`) and the `HttpMethod` / `HttpRequest` / `HttpResponse` request model (headers are `Map<String, String>?`, no `HttpHeader` type); `Csv::analyze` takes `Array<String>` paths. `std::util` adds `Uuid` (v4 / v7). Periodicity field shapes (weekly/monthly use a nested `daily`, yearly uses `dates: Array<DateTuple>`) and the `LogLevel` / `TaskStatus` / `LicenseType` enums — see the stdlib reference.

## Contents

1. **C API** - Native function implementation, tensor operations, object manipulation, maps, arrays, tables, geospatial, time/date, crypto, buffers, I/O
2. **Standard Library (std)** - GCL runtime features, I/O, collections, and utilities
3. **Plugin Development** - Complete guide to building native plugins with lifecycle hooks, type configuration, and real-world patterns

---

# GreyCat C API

## Core Concepts

**gc_machine_t** - Execution context passed to all native functions. Use to get parameters, set results, report errors, create objects, and access scratch buffers.

**gc_slot_t** - Universal value container (tagged union) holding any GreyCat value: integers, floats, bools, objects, enums, tuples, etc.

**gc_type_t** - Type system enum (8-bit, 24 values) defining all GreyCat types: null, bool, char, int, float, node variants, geo, time, duration, cubic, static_field, object, block_ref, block_inline, function, undefined, type, field, stringlit, error.

**gc_object_t** - Generic handle for heap-allocated objects. Packed to 128 bits. Every collection type (Array, Map, Table, Tensor, String, Buffer) starts with this as its first member.

## Common Operations

**Parameter handling:**
```c
gc_slot_t param = gc_machine__get_param(ctx, 0);        // 0-indexed
gc_type_t type = gc_machine__get_param_type(ctx, 0);
u32_t count = gc_machine__get_param_nb(ctx);
gc_slot_t self = gc_machine__this(ctx);                  // 'this' for instance methods
```

**Enum parameter handling (CRITICAL — common source of bugs):**

GCL enum values are **NOT** `gc_type_int`. They are `gc_type_static_field` and the ordinal is in `.tu32.right`, not `.i64`.

```c
// WRONG — enum will always hit the default fallback:
i64_t variant = (gc_machine__get_param_type(ctx, 0) == gc_type_int) ? slot.i64 : 0;

// CORRECT — reads the enum ordinal properly:
i64_t variant = (gc_machine__get_param_type(ctx, 0) == gc_type_static_field) ? (i64_t)slot.tu32.right : 0;
```

The `.tu32` field is a struct with `.left` (type offset, identifies the enum type) and `.right` (value offset / ordinal within the enum). For dispatch purposes you almost always want `.tu32.right`.

**Setting results:**
```c
gc_machine__set_result(ctx, (gc_slot_t){.i64 = 42}, gc_type_int);
gc_machine__set_result(ctx, (gc_slot_t){.f64 = 3.14}, gc_type_float);
gc_machine__set_result(ctx, (gc_slot_t){.b = true}, gc_type_bool);
gc_machine__set_result(ctx, (gc_slot_t){.object = obj}, gc_type_object);
gc_object__un_mark(obj, ctx);  // CRITICAL for objects -- prevents premature GC
// Returning an enum value (e.g., MyEnum::variant2 where variant2 is ordinal 1):
gc_machine__set_result(ctx, (gc_slot_t){.tu32 = {.left = 0, .right = 1}}, gc_type_static_field);
```

**Error handling:**
```c
gc_machine__set_runtime_error(ctx, "Something failed");
gc_machine__set_runtime_error_syserr(ctx);  // Uses errno
if (gc_machine__error(ctx)) return;         // Check propagated errors
```

**Object field access:**
```c
gc_slot_t value = gc_object__get_at(obj, field_offset, &type, ctx);
gc_object__set_at(obj, field_offset, value, value_type, ctx);
gc_object__declare_dirty(obj);  // Mark modified for persistence write-back
```

**Object creation:**
```c
gc_object_t *obj = gc_machine__create_object(ctx, gc_core_Map);
gc_object_t *ret = gc_machine__create_return_type_object(ctx);
```

**Tensor operations:**
```c
gc_core_tensor_t *t = gc_core_tensor__create(ctx);
gc_core_tensor__init_2d(t, rows, cols, gc_core_TensorType_f32, ctx);
f32_t val = gc_core_tensor__get_2d_f32(t, row, col, ctx);
gc_core_tensor__set_2d_f32(t, row, col, 3.14f, ctx);
f64_t *raw = (f64_t *)gc_core_tensor__get_data(t);  // Direct memory access
```

**Array operations:**
```c
gc_array_t *arr = (gc_array_t *)gc_machine__create_object(ctx, gc_core_Array);
gc_array__add_slot(arr, (gc_slot_t){.i64 = 42}, gc_type_int, ctx);
gc_array__get_slot(arr, 0, &value, &type);
gc_array__set_slot(arr, 0, value, type, ctx);
```

**Map operations:**
```c
gc_map_t *map = (gc_map_t *)gc_machine__create_object(ctx, gc_core_Map);
gc_map__set(map, key, key_type, value, value_type, ctx);
gc_slot_t val = gc_map__get(map, key, key_type, &val_type, prog);
bool found = gc_map__contains(map, key, key_type, prog);
```

**String operations:**
```c
gc_string_t *s = gc_string__create_from(data, len, ctx);
gc_string_t *s2 = gc_string__create_concat(buf1, len1, buf2, len2, ctx);
// Note: gc_string_t.buffer is NOT null-terminated. Use .size for length.
```

**Logging (gc/log.h):**
```c
// VM context (decorates with module::Type::fn + user/task ids):
gc_log__machinef(ctx, gc_log_level_info, "warmed cache");

// Host context (no VM frame):
gc_host_t *host = gc_host__get_global();
if (gc_log__enabled(host, gc_log_level_perf)) {
    gc_log__hostf(host, gc_log_level_perf, "ingest loop done");
}
```

**Memory management — allocator-first API:**

| Allocator | Lifecycle | When to use |
|-----------|-----------|-------------|
| `gc_machine__allocator(ctx)` (= `((gc_ctx_t *)ctx)->allocator`) | Per-native-call; reclaimed when the call ends | Default for everything inside a native: scratch buffers, intermediate arrays, per-call result strings. |
| `gc_host__global_allocator()` (= `gc_host__allocator(gc_host__get_global())`) | Plugin-global, persists across threads and calls | `lib_start` / `lib_stop` state, global caches, precomputed lookup tables. Protect with your own mutex if shared across workers. |

```c
// Per-call scratch (default):
gc_allocator_t *a = gc_machine__allocator(ctx);
char *temp = (char *)gc_alloc__malloc(a, size);
// ... use temp ...
gc_alloc__free(a, temp, size);

// Plugin-global (cache once in lib_start):
static gc_allocator_t *g_alloc;    // = gc_host__global_allocator();
double *shared = (double *)gc_alloc__malloc(g_alloc, size);
gc_alloc__free(g_alloc, shared, size);

// Reusable scratch buffer owned by the machine (no manual free):
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
```

> Use `gc_alloc__*` with an explicit allocator everywhere it matters. The shorter `gc_malloc` / `gc_free` / `gc_realloc` helpers are also public, but they only target the allocator currently bound to the thread (see `gc_alloc__bind`). When you need cross-thread or plugin-global lifetime, always pass the explicit allocator.

**Buffer building:**
```c
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__clear(buf);
gc_buffer__add_cstr(buf, "prefix_");
gc_buffer__add_u64(buf, 42);
gc_buffer__prepare(buf, needed_bytes);  // Ensure capacity
```

**Program introspection:**
```c
const gc_program_t *prog = gc_machine__program(ctx);
u32_t sym = gc_program__resolve_symbol(prog, "name", 4);
u32_t mod = gc_program__resolve_module(prog, sym);
u32_t type_id = gc_program__resolve_type(prog, mod, type_sym);
```

## Detailed Reference

**File:** [references/api_reference.md](references/api_reference.md)

**Load when implementing:**
- Native C functions with gc_machine_t (params, result, errors, `gc_machine__allocator`, `gc_machine__impersonate`, `gc_machine__call_function` via `gc_program_function_t *`)
- Tensor operations (multi-dimensional arrays, complex numbers c64/c128)
- Object/field manipulation, type introspection, GC mark/unmark
- Map, Array, Table operations
- Buffer building, binary read/write (varint, zig-zag encoding)
- Heap-allocated immutable strings (gc_string_t, allocator-aware constructors)
- Structured logging from VM or host contexts (gc/log.h)
- Node resolution (gc_node__resolve, gc_node__parse)
- Geospatial (geohashing, Haversine distance), Time/Date/Timezone (formatting with gc_dtz_time__print/parse)
- Cryptography (SHA-256, HMAC-SHA-256, Base64, Base64URL)
- I/O operations (file open/sync)
- Memory allocation (per-call, plugin-global, aligned, thread-bound helpers, `gc_alloc__create(bool shared)`, stats)
- Program/Type System, ABI (allocator-aware), symbol resolution
- Host/Task management (spawn, cancel, status), periodic scheduler (gc_scheduler_t, gc_periodic_task_t), plugin-global allocator (gc_host__allocator / gc_host__global_allocator)
- Block storage (attach/detach objects)
- Utility (Morton codes, parsing, sorting with allocator, licensing)

**Contains:** Complete function signatures organized by header file: type.h (primitives, gc_type_t, gc_slot_t, gc_object_t, complex arithmetic, node parsing), alloc.h (gc_alloc__ family, allocator selection rules, thread-bound `gc_malloc`/`gc_free`/`gc_realloc` helpers), buffer.h (allocator-aware create, text append, escaped symbol, binary read/write, varint), string.h (heap-allocated immutable strings), object.h (field access, GC marks, serialization), log.h (gc_log_level_t, machine/host emit helpers), machine.h (parameters, results, errors, object creation, function calls via `gc_program_function_t`, gc_ctx_t with per-call allocator, `gc_machine__allocator`/`gc_machine__impersonate`), program.h (linking, type configuration, introspection, DurationUnit constants, field format pragma, module/program creation with allocator), host.h (task spawning, plugin-global allocator, periodic scheduler), array.h, map.h, table.h, tensor.h (creation, init_tensor, get/set/add for i32/i64/f32/f64/c64/c128, descriptor utilities, matmul/bias/sum validation), block.h, abi.h (schema, serialization, allocator-aware, truncated-header error), io.h, crypto.h, geo.h, time.h (timezone-aware print/parse), math.h (WASM math shims), node.h, util.h (Morton codes, hex, parsing, deep equality, sorting with allocator, licensing).

---

# GreyCat Standard Library (std)

## Module Organization

- **std::core** - Fundamental types (Date, Time, Duration, Tuple, Error, geospatial types, enumerations)
- **std::runtime** - Scheduler, Task, Job, Logger, Identity/Security, System, ChildProcess, License, OpenAPI, MCP
- **std::io** - Text/Binary I/O, CSV, JSON, XML, HTTP client, Email/SMTP, FileWalker, S3 object storage
- **std::util** - Collections (Queue, Stack, SlidingWindow, TimeWindow), Statistics (Gaussian, Histogram, GaussianProfile), Quantizers, Assert, ProgressTracker, Crypto, Uuid, Random, Plot

## Detailed Reference

**File:** [references/standard_library.md](references/standard_library.md)

**Load when working with:**
- Task scheduling and automation (Scheduler with periodicities)
- File I/O operations (CSV, JSON, XML, binary files)
- HTTP integration and REST APIs, S3 object storage
- Statistical analysis and data processing
- Identity, security, and authentication
- System operations and logging

**Contains:** Complete documentation for all four standard library modules with code examples, usage patterns, and best practices.

---

# Plugin Development Guide

## Overview

Build native GreyCat plugins in C with proper lifecycle management, type configuration, and thread safety.

## Key Patterns

**Function naming (CRITICAL — must match nativegen):** `gc_<module>_<Type>__<methodName>(gc_machine_t *ctx)`

When GreyCat compiles GCL code with `native` function declarations, it auto-generates `nativegen.c` and `nativegen.h` files. The nativegen `extern` declarations define the **exact C symbol names** the runtime will look for at dlopen time. Your C function definitions **MUST** use these exact names or you'll get `undefined symbol` errors.

**Naming convention:**
- Type method: `gc_<gcl_module>_<GclType>__<methodName>` (double underscore before method)
- Module function: `gc_<gcl_module>__<functionName>` (double underscore before function)
- The `<gcl_module>` comes from the GCL file's module path (e.g., `text_normalizer` for a file in the `text_normalizer/` module)
- The `<GclType>` matches the GCL type name exactly (PascalCase)
- The `<methodName>` matches the GCL method name exactly (camelCase)

**Example mapping (GCL → C):**
```
// GCL (in module "text_normalizer", type TextNormalizer):
//   native static fn rejoinHyphenatedWords(text: String): String;
//
// nativegen.h generates:
//   extern void gc_text_normalizer_TextNormalizer__rejoinHyphenatedWords(gc_machine_t *ctx);
//
// Your C implementation MUST be named:
void gc_text_normalizer_TextNormalizer__rejoinHyphenatedWords(gc_machine_t *ctx) { ... }

// GCL (in module "bm25_engine", type BM25Engine):
//   native static fn computeIDF(docFreq: int, totalDocs: int): float;
//
// C implementation:
void gc_bm25_engine_BM25Engine__computeIDF(gc_machine_t *ctx) { ... }
```

**Plugin lifecycle:** `link -> lib_start -> [worker_start -> native calls -> worker_stop] -> lib_stop`

**Type configuration:** `gc_program_type__configure(prog, type_id, sizeof(my_struct_t), finalizer)`

**Library hooks:**
```c
gc_program_library__set_lib_hooks(lib, lib_start, lib_stop);
gc_program_library__set_worker_hooks(lib, worker_start, worker_stop);
```

## Detailed Reference

**File:** [references/plugin_development.md](references/plugin_development.md)

**Load when:**
- Building a new native plugin from scratch
- Setting up CMake build configuration for .gclib output
- Implementing nativegen.c/h (symbol resolution, type/function linking)
- Linking module-level native functions (gc_program__link_mod_fn) or type methods (gc_program__link_type_fn)
- Managing library/worker lifecycle hooks
- Wrapping C library handles in GreyCat objects (boxing pattern)
- Implementing thread-safe global state with mutexes
- Mapping GCL enums to C library enums
- Using the buffer reuse and tokenization retry patterns
- Implementing conditional logging with gc_machine__log_level

**Contains:** Complete project structure, CMake configuration, GCL type definitions, nativegen implementation, lifecycle hooks, custom type configuration with finalizers, global state management, memory management patterns, parameter handling (including type checking with gc_object__is_instance_of), result returning, error handling, conditional logging, and a full end-to-end plugin example.
