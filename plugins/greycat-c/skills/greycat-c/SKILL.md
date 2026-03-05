---
name: greycat-c
description: "GreyCat C API and GCL Standard Library reference. Use for: (1) Native C development with gc_machine_t context, tensors, objects, memory management, crypto, I/O; (2) GCL Standard Library modules - std::core (Date/Time/Tuple/geospatial types), std::runtime (Scheduler/Task/Logger/User/Security/System/OpenAPI/MCP), std::io (CSV/JSON/XML/HTTP/Email/FileWalker), std::util (Queue/Stack/SlidingWindow/Gaussian/Histogram/Quantizers/Random/Plot); (3) Plugin development patterns - lifecycle hooks, type configuration, nativegen, module-level and type-level function linking, global state, thread safety, conditional logging. Keywords: GreyCat, GCL, native functions, tensors, task automation, scheduler, plugin development."
---

# GreyCat SDK - C API, Standard Library & Plugin Development

Comprehensive reference for GreyCat native development (C API), the GCL Standard Library, and plugin development patterns.

## Contents

1. **C API** - Native function implementation, tensor operations, object manipulation, maps, arrays, tables, geospatial, time/date, crypto, buffers, I/O
2. **Standard Library (std)** - GCL runtime features, I/O, collections, and utilities
3. **Plugin Development** - Complete guide to building native plugins with lifecycle hooks, type configuration, and real-world patterns

---

# GreyCat C API

## Core Concepts

**gc_machine_t** - Execution context passed to all native functions. Use to get parameters, set results, report errors, create objects, and access scratch buffers.

**gc_slot_t** - Universal value container (tagged union) holding any GreyCat value: integers, floats, bools, objects, enums, tuples, etc.

**gc_type_t** - Type system enum (8-bit) defining all GreyCat types: null, bool, char, int, float, str, object, static_field, time, duration, geo, node, function, etc.

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
gc_string_t *s = gc_string__create_from(data, len);
gc_string_t *s2 = gc_string__create_concat(buf1, len1, buf2, len2);
// Note: gc_string_t.buffer is NOT null-terminated. Use .size for length.
```

**Memory management:**
```c
char *temp = (char *)gc_gnu_malloc(size);         // Per-worker (thread-local)
gc_gnu_free(temp);
double *shared = (double *)gc_global_gnu_malloc(size);  // Global (thread-safe)
gc_global_gnu_free(shared);
gc_buffer_t *buf = gc_machine__get_buffer(ctx);   // Reusable scratch buffer
```

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
- Native C functions with gc_machine_t
- Tensor operations (multi-dimensional arrays, complex numbers c64/c128)
- Object/field manipulation, type introspection, GC mark/unmark
- Map, Array, Table operations
- Buffer building, binary read/write (varint, zig-zag encoding)
- String operations (heap strings, inline short strings)
- Geospatial (geohashing, Haversine distance), Time/Date/Timezone
- Cryptography (SHA-256, HMAC-SHA-256, Base64, Base64URL)
- I/O operations (file open/sync)
- Memory allocation (per-worker, global, aligned)
- Program/Type System, ABI, symbol resolution
- Host/Task management (spawn, cancel, status)
- Block storage (attach/detach objects)
- Utility (Morton codes, parsing, sorting, licensing)

**Contains:** Complete function signatures organized by header file: type.h (primitives, gc_type_t, gc_slot_t, gc_object_t, complex arithmetic), machine.h (parameters, results, errors, object creation, function calls), object.h (field access, GC marks, serialization), tensor.h (creation, get/set/add for i32/i64/f32/f64/c64/c128, descriptor utilities, matmul/bias/sum validation), array.h, map.h, string.h, str.h, buffer.h (text append, binary read/write, varint), table.h, alloc.h, program.h (linking, type configuration, introspection), abi.h (schema, serialization), host.h (task spawning), io.h, crypto.h, geo.h, time.h, math.h, block.h, util.h (Morton codes, hex, parsing, deep equality, sorting, licensing).

---

# GreyCat Standard Library (std)

## Module Organization

- **std::core** - Fundamental types (Date, Time, Duration, Tuple, Error, geospatial types, enumerations)
- **std::runtime** - Scheduler, Task, Job, Logger, User/Security, System, ChildProcess, License, OpenAPI, MCP
- **std::io** - Text/Binary I/O, CSV, JSON, XML, HTTP client, Email/SMTP, FileWalker
- **std::util** - Collections (Queue, Stack, SlidingWindow, TimeWindow), Statistics (Gaussian, Histogram), Quantizers, Assert, ProgressTracker, Crypto, Random, Plot

## Detailed Reference

**File:** [references/standard_library.md](references/standard_library.md)

**Load when working with:**
- Task scheduling and automation (Scheduler with periodicities)
- File I/O operations (CSV, JSON, XML, binary files)
- HTTP integration and REST APIs
- Statistical analysis and data processing
- Security, authentication, and user management
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
