---
name: greycat-c
description: "GreyCat C API and GCL Standard Library reference. Use for: (1) Native C development with gc_machine_t context, tensors, objects, memory management, crypto, I/O; (2) GCL Standard Library modules - std::core (Date/Time/Tuple/geospatial types), std::runtime (Scheduler/Task/Logger/Identity/Security/System/License/OpenAPI/MCP), std::io (CSV/JSON/XML/HTTP/Email/FileWalker/S3), std::util (Queue/Stack/SlidingWindow/TimeWindow/Gaussian/Histogram/Quantizers/Random/Uuid/Crypto); (3) Plugin development patterns - lifecycle hooks, type configuration, nativegen, module-level and type-level function linking, global state, thread safety, conditional logging. Keywords: GreyCat, GCL, native functions, tensors, task automation, scheduler, plugin development."
---

# GreyCat SDK - C API, Standard Library & Plugin Development

Comprehensive reference for GreyCat native development (C API), the GCL Standard Library, and plugin development patterns. Tracks SDK **8.1**.

## Key Considerations

- **Allocator API is mandatory.** Every non-trivial allocation routes through an explicit `gc_allocator_t *`: per-call scratch from `gc_machine__allocator(ctx)` (= `((gc_ctx_t *)ctx)->allocator`), plugin-global state from `gc_host__global_allocator()`. `gc_alloc__create(bool shared)` (`true` = multi-thread arena). `gc_alloc__free(a, ptr, size)` requires the original size. The thread-bound `gc_malloc` / `gc_free` / `gc_realloc` helpers target whatever `gc_alloc__bind` set. **New in 8.1: `gc_alloc__reset(allocator)`** — destructively resets a whole arena (reclaims even leaked/lost pointers on the jemalloc path), invalidating every prior pointer from that allocator; use between iterations of a long-lived worker loop after tearing everything down, no-op on native-malloc/WASM/standalone. Sizing/stats and full patterns: [api_memory_text.md](references/api_memory_text.md).
- **Structured logging (`gc/log.h`).** `gc_log_level_t` is `none / error / warn / info / perf / trace`. Use `gc_log__machine` / `gc_log__machinef` (VM context) and `gc_log__host` / `gc_log__hostf` (host context); gate hot paths with `gc_log__enabled(host, level)`.
- **No inline short strings.** There is no `gc/str.h`: `gc_str_t` and the `gc_core_str` / `gc_core_t2…t4f` globals are not public. Use `gc_string_t` (heap, immutable, hash-cached; `buffer` is NOT NUL-terminated — use `size`).
- **`gc_machine__call_function` takes a `const gc_program_function_t *fn`** (not a raw body pointer). On `false` the result is a synthesized `Error` object (type `gc_core_Error`) and `*marked_res_type` is `gc_type_object`; the caller owns one mark on the result. `gc_machine__impersonate(ctx, user_id)` switches the effective user for permission-aware sub-calls.
- **Host/Scheduler (`gc/host.h`).** `gc_host__cancel_task(self, task_id, requester_id, requester_permissions, out_task)` is now thread-safe and permission-checked (`out_task` optional, receives a copy of the cancelled task); `gc_host__get_task_status` still takes just `i64_t task_id`. `gc_host__spawn_task` takes a `u64_t user_permissions` mask. Periodic scheduling via `gc_scheduler_t`, `gc_periodic_task_t`, and `gc_periodicity_t` (a struct holding a `gc_periodicity_type_t type`: fixed / daily / weekly / monthly / yearly). See [api_runtime_storage.md](references/api_runtime_storage.md).
- **ABI.** `GC_ABI_PROTO` is `3`. `gc_abi_header_check_error_t` includes `..._truncated = 4`; `gc_abi_t` carries its own allocator.
- **`gc_block_t` gained `u64_t node_ref`** (new in 8.1) — the node reference the block backs, used by suspend/resume serialization to relocate the block's entries. See [api_runtime_storage.md](references/api_runtime_storage.md).
- **Iterator params** `gc_program_iterator_param_t`: `from=0`, `to=1`, `nullable=2`, `from_excl=3`, `to_excl=4` (no `limit`). Geo epsilon constant is `GC_CORE_GEO_EPS`.
- **Tensor struct rename (breaking).** The tensor structs are now `gc_tensor_t` / `gc_tensor_descriptor_t` (formerly `gc_core_tensor_t` / `gc_core_tensor_descriptor_t`); the `gc_core_tensor__*` and `gc_core_tensor_descriptor__*` **function** names are unchanged, and `gc_machine__init_tensor` now takes/returns the renamed types. Plugin code that referenced the old struct typedefs must be updated. Full tensor API: [api_collections.md](references/api_collections.md).
- **Stdlib (GreyCat 8.0) breaking changes.** Security model is `Identity` / `IdentityGrant` / `IdentityGrantType` (the old `User` / `UserGroup` / `SecurityPolicy` / `OpenIDConnect` types are gone). GCL logging is via module-level `info` / `warn` / `error` / `perf` / `trace` functions — `Log` is a parse record, not a callable namespace. Remaining 8.0 surface (S3 object storage, the `HttpMethod`/`HttpRequest`/`HttpResponse` model (`HttpRequest.headers` is `Map<String, String>?`, `HttpResponse.headers` is `Map<String, String>`), `Csv::analyze(Array<String>)`, `Uuid` v4/v7, periodicity field shapes, `LogLevel`/`TaskStatus`/`LicenseType` enums): [standard_library.md](references/standard_library.md).
- **`ProgressTracker.update(nb)` is now absolute, not incremental (breaking).** It sets the step counter to `nb` rather than adding `nb` to it. New fields `speed_smoothed` (EMA of the per-update pace) and `smoothing` (EMA weight, default `ProgressTracker.DEFAULT_SMOOTHING = 0.1`) drive a more reactive `remaining` estimate. Also new since the last sync: `HttpRequest.max_response_size` (caps chunked/unbounded response reads) and `Task::live(ids)` / `Task::tasks(ids)` (bulk liveness check / bulk fetch by id).
- **`TensorDistance` gained `lorentz` and `poincare`** (hyperbolic distances — Lorentz/hyperboloid model and Poincaré ball model, both curvature fixed at -1). **`Identity.set_role(name, role)`** is a new admin-only static native — it returns nothing (`void`), not `bool`.
- **Other 8.0→8.1 signature/field changes.** `gc_common__parse_number`'s `str_len` param widened `u32_t *` → `u64_t *` (its sibling `gc_common__parse_sign_number` is still `u32_t *` — the two now disagree, match the local variable's type to the callee). `gc_buffer_read_vu64_size_checked` added (the `u64_t` counterpart of the existing `_vu32_` variant). `gc_object__clone`, `gc_array__fill`, and `gc_array__ensure_capacity` (replaced `gc_array__init`; now grow-only/idempotent, rounds to a power of two, preserves contents) round out the collections/memory surface. Stdlib: `Task.duration: duration?` was replaced by `Task.completion: time?`, `Task` gained `user_name: String`, and `nodeGeo<T>` gained `search(center: geo, max: int): Array<SearchResult<geo,T>>`.

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

## Core Operations

The few patterns below are the trigger-level essentials. **Full runnable examples for every operation** (objects, tensors, arrays, maps, strings, buffers, allocators, logging, introspection) live in the five C-API reference files listed under *Detailed Reference*.

**Parameters & results:**
```c
gc_slot_t p = gc_machine__get_param(ctx, 0);   gc_type_t t = gc_machine__get_param_type(ctx, 0);
u32_t n = gc_machine__get_param_nb(ctx);        gc_slot_t self = gc_machine__this(ctx); // instance methods
gc_machine__set_result(ctx, (gc_slot_t){.i64 = 42}, gc_type_int);
gc_machine__set_result(ctx, (gc_slot_t){.object = obj}, gc_type_object);
gc_object__un_mark(obj, ctx);  // CRITICAL: every object result must be un-marked or it leaks / GC-faults
```

**Enum parameters & results (CRITICAL — #1 native bug).** GCL enum values are **NOT** `gc_type_int`; they are `gc_type_static_field` with the ordinal in `.tu32.right` (`.tu32.left` is the enum type offset), never `.i64`:
```c
// WRONG — always hits the default fallback:  (type == gc_type_int) ? slot.i64 : 0
// CORRECT:
i64_t variant = (gc_machine__get_param_type(ctx, 0) == gc_type_static_field) ? (i64_t) slot.tu32.right : 0;
// Returning MyEnum::variant2 (ordinal 1):
gc_machine__set_result(ctx, (gc_slot_t){.tu32 = {.left = 0, .right = 1}}, gc_type_static_field);
```

**Errors:** `gc_machine__set_runtime_error(ctx, "msg")` / `gc_machine__set_runtime_error_syserr(ctx)` (uses errno); check propagated errors with `if (gc_machine__error(ctx)) return;`.

## Detailed Reference

The C API reference is split by domain — each file below is linked directly (one level deep) and loads on demand.

**[references/api_core.md](references/api_core.md)** — value model, execution context, type system, logging. **Start here.**
- Native C functions with `gc_machine_t` (params, result, errors, `gc_machine__allocator`, `gc_machine__impersonate`, `gc_machine__call_function` via `gc_program_function_t *`, object creation)
- `gc_type_t` / `gc_slot_t` value model, complex c64/c128 arithmetic, `gc_node__parse`
- Program/Type system: linking (`gc_program__link_mod_fn` / `gc_program__link_type_fn`), type configuration, introspection, iterator params, DurationUnit
- Structured logging (`gc/log.h`) and the cross-cutting Conventions & Patterns index

**[references/api_memory_text.md](references/api_memory_text.md)** — memory, buffers, strings, objects
- Memory allocation: per-call, plugin-global, aligned, thread-bound helpers, `gc_alloc__create(bool shared)`, sizing/stats
- Buffer building and binary read/write (varint, zig-zag encoding)
- Heap-allocated immutable strings (`gc_string_t`, allocator-aware constructors)
- Object/field manipulation, type introspection, GC mark/un-mark, serialization

**[references/api_collections.md](references/api_collections.md)** — array, map, table, tensor
- Array, Map, Table get/set/add operations
- Tensors: `init_Nd`, get/set/add for i32/i64/f32/f64/c64/c128, descriptor utilities, raw data access, matmul/bias/sum

**[references/api_runtime_storage.md](references/api_runtime_storage.md)** — runtime, persistence, graph nodes
- Host/Task management (spawn, cancel, status), periodic scheduler (`gc_scheduler_t`, `gc_periodic_task_t`), plugin-global allocator (`gc_host__allocator` / `gc_host__global_allocator`)
- Storage blocks (attach/detach objects), ABI (allocator-aware, proto=3, truncated-header error), file I/O (open/sync)
- Node resolution (`gc_node__resolve`, `gc_node__parse`) and direct node-entry read/write (`gc_machine_native__node_get` / `node_set_at` via `gc_node_single_value_t`, released with `gc_node_single_value__clear`) — this `u64_t node_ref` API is uniform across all node variants (node, nodeTime, nodeList, nodeGeo, nodeIndex); only the key encoding differs per variant

**[references/api_services.md](references/api_services.md)** — crypto, geo, time, math, util
- Cryptography (SHA-256, HMAC-SHA-256, Base64, Base64URL)
- Geospatial (geohashing, Haversine distance); Time/Date/Timezone (formatting with `gc_dtz_time__print` / `parse`)
- Math (WASM shims, f64 PI/TAU/E/LN2 constants); Utility (Morton codes, hex, parsing, deep equality, sorting with allocator, licensing)

---

# GreyCat Standard Library (std)

## Module Organization

- **std::core** - Fundamental types (Date, Time, Duration, Tuple, Error, geospatial types, enumerations)
- **std::runtime** - Scheduler, Task, Job, Logger, Identity/Security, System, ChildProcess, License, OpenAPI, MCP
- **std::io** - Text/Binary I/O, CSV, JSON, XML, HTTP client, Email/SMTP, FileWalker, S3 object storage
- **std::util** - Collections (Queue, Stack, SlidingWindow, TimeWindow), Statistics (Gaussian, Histogram, GaussianProfile), Quantizers (Linear/Log/Custom/Multi), Assert, ProgressTracker, Crypto, Uuid, Random

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

When GreyCat compiles GCL code with `native` declarations, it auto-generates `nativegen.c` / `nativegen.h`. Those `extern` declarations define the **exact C symbol names** the runtime resolves at dlopen — your C definitions **MUST** match or you get `undefined symbol` errors. Convention:
- Type method: `gc_<gcl_module>_<GclType>__<methodName>` (double underscore before the method)
- Module function: `gc_<gcl_module>__<functionName>` (double underscore before the function)
- `<gcl_module>` is the GCL file's module path; `<GclType>` matches the GCL type (PascalCase); `<methodName>` matches the GCL method (camelCase)

```c
// GCL (module "text_normalizer", type TextNormalizer):
//   native static fn rejoinHyphenatedWords(text: String): String;
// nativegen.h generates:
//   extern void gc_text_normalizer_TextNormalizer__rejoinHyphenatedWords(gc_machine_t *ctx);
// Your C implementation MUST be named exactly:
void gc_text_normalizer_TextNormalizer__rejoinHyphenatedWords(gc_machine_t *ctx) { ... }
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
- Implementing conditional logging (gc_machine__log_level / gc_log__enabled)

**Contains:** Complete project structure, CMake configuration, GCL type definitions, nativegen implementation, lifecycle hooks, custom type configuration with finalizers, global state management, memory management patterns, parameter handling (including type checking with gc_object__is_instance_of), result returning, error handling, conditional logging, and a full end-to-end plugin example.
