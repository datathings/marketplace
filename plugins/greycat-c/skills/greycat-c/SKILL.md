---
name: greycat-c
description: "GreyCat C API and GCL Standard Library reference. Use for: (1) Native C development with gc_machine_t context, tensors, objects, memory management, crypto, I/O; (2) GCL Standard Library modules - std::core (Date/Time/Tuple/nodes/Tensor/geospatial types), std::runtime (Scheduler/Task/Logger/World/Identity/Security/System/HttpRequest/OpenAPI/MCP), std::io (CSV/JSON/HTTP/Email/FileWalker/Stream), std::util (Queue/Stack/SlidingWindow/TimeWindow/Gaussian/Histogram/Quantizers/Random/Uuid/Crypto); (3) Plugin development patterns - lifecycle hooks, type configuration, nativegen, module-level and type-level function linking, global state, thread safety, conditional logging. Keywords: GreyCat, GCL, native functions, tensors, task automation, scheduler, plugin development."
---

# GreyCat SDK - C API, Standard Library & Plugin Development

Comprehensive reference for GreyCat native development (C API), the GCL Standard Library, and plugin development patterns. Tracks SDK **9.0** (headers re-verified 2026-11-16 against upstream `24f7110f4`).

## Changes since 8.4

**Breaking — rebuild every plugin, stores of 8.4 are refused:**
- **`gc_type_t` renumbered** (`GC_ABI_PROTO` 3 → **4**): new `gc_type_node_set` (7) and `gc_type_node_index_set` (9); every tag from `gc_type_node_index` on moved up (now 26 tags, `GC_TYPE_LAST` = `gc_type_error` = 25). New `gc_type__is_node(t)` / `gc_type__is_keyed_node(t)`. Opcodes renumbered: one `push_node_empty` (44) replaced the four `push_node_{time,list,index,geo}`.
- **Enums with fixed C values, converted by entry name** (never assume GCL ordinals): new `gc_tensor_type_t` (`gc/tensor.h`), `gc_duration_unit_t` (new `gc/duration.h`, replaces the removed `gc_core_DurationUnit_*` macros), dtz zone indexes (`GC_DTZ_UTC`, `gc_dtz_zone_names`; a `tz` is no longer the `TimeZone` ordinal — also in `gc_buffer_options_t.tz` and `gc_program_type_field_format_t`). Convert via `prog->enums.{tensor_type,duration_unit,time_zone,float_precision}` (`gc_program_enum_map_t`, `gc_program_enum_map__to_fixed` / `__to_entry`) and `gc_program__resolve_enum_entry`. nativegen.h now declares `extern u32_t gc_<mod>_<Enum>_<entry>` resolved by name at link (variables: no `case` labels).
- **`gc_error_code_t`** (C-only, `gc/machine.h`): none 0, interrupted 1, await 2, timeout 3, forbidden 4, runtime_error 5 — GCL `ErrorCode` removed.
- **Removed:** `gc_program_library__set_install_hook` + `gc_program_library_t.install` (`greycat install` only downloads); `gc_program_map_t` / `gc_program_bucket_t` / `gc_program_map__*` (name indexes are `gc_u32_map_t`, symbols a `gc_str_map_t`); `gc_program_function_t.is_raw` (`@raw` gone); `gc_license_level_t` / `gc_license__level`; `gc_buffer__read_all_from_fd`.
- **Moved headers** (all included by `greycat.h`): `c64_t` / `c128_t` + `gc_c64__*` / `gc_c128__*` → `gc/complex.h`; `gc_common__parse_number` / `_parse_sign_number` → `gc/int.h`; `gc_duration__parse` → `gc/duration.h`.
- **Signature changes:** `gc_core_tensor__clone_internal` returns `bool` (allocation failure); `const` on read-only tensor params (`diff`, `check_shape`, `get_data`, all descriptor checks/sizes); `gc_program_function_t` flags are 1-bit fields (+ `takes_http_request`, `returns_http_response`); `gc_abi_function_t.is_static`. Tensor errors now name shapes/types (`GC_ERROR_TENSOR_*` are only for your own messages); getters no longer read out-of-bounds.

**New C API:** `gc_u32_map_t` / `gc_u64_map_t` (`gc/u32_map.h`, `gc/u64_map.h`) and `gc_str_map_t` (`gc/str_map.h`) hash maps; `gc_char__parse` (`gc/char.h`); `gc_slot__to_f64`; `gc_host__run_function(host)`; `gc_core_tensor__add_type_name` / `_add_shape`; `prog->core` gained ids for every primitive/node type (`int_`, `nodeSet`, `nodeIndexSet`, `DurationUnit`, `TimeZone`, …); `gc_program__encode_symbol`, `__runtime_type`, `__resolve_ptype`, `__compute_layout`, `.gcp` `gc_program__save` / `__load` / `__check` / `__rebuild_indexes`; env options `__label`, `__repair`, `__task_retention`, `__tls`, `__acme`, `__acme_email`, `__acme_directory`, `__paste` (appended; HTTPS via OpenSSL 3.5).

**Stdlib (breaking):** `Task::header/headers/uri/body` + `@raw` removed → an `@expose`d fn takes `runtime::HttpRequest` (only param) / returns `runtime::HttpResponse`; `nodeList` is positional (`set` removed, `remove` shifts); grants are per permission (`IdentityGrantType` + `Identity.grants` removed; `grant/revoke(name, target, permission): bool`, `grant_permission`, `has_grant`; new permissions `files:read`/`files:write`/`tasks:read`/`tasks:cancel`); `License` / `LicenseType` / `RuntimeInfo.license` / `ErrorCode` removed; `Tensor.scale(value, inPlace: bool)`; `System::exec` returns stdout+stderr. **New:** `nodeSet<V>`, `nodeIndexSet<K, V>`, `Task::result(id)`, `Task.worker_id` / `.jobs`, `Identity::tokens` / `revoke_token(s)` / `IdentityToken`, `Runtime::program()`, `Tensor.mean/min/max`, `GeoBox.crosses_antimeridian/lobes`; task folders under `gcdata/tasks/<id>/` (`File::workingDir()`, kept `--task_retention`). Details: [standard_library.md](references/standard_library.md).

## Key Considerations

- **Allocator API is mandatory.** Every non-trivial allocation routes through an explicit `gc_allocator_t *`: per-call scratch from `gc_machine__allocator(ctx)` (= `((gc_ctx_t *)ctx)->allocator`), plugin-global state from `gc_host__global_allocator()`. `gc_alloc__create(bool shared)` (`true` = multi-thread arena); `gc_alloc__free(a, ptr, size)` requires the original size; `gc_alloc__reset(allocator)` wipes a whole arena. See [api_memory_text.md](references/api_memory_text.md).
- **Enums: entries, not ordinals.** GCL enum values arrive as `gc_type_static_field` with the entry position in `.tu32.right`; compare against nativegen's link-resolved `gc_<mod>_<Enum>_<entry>`, and convert `TensorType` / `DurationUnit` / `TimeZone` through `prog->enums` (api_core.md, *Enum Entries*).
- **Core type ids are per program:** `gc_machine__program(ctx)->core.Array` / `.Tensor` / `.nodeTime` … (no `gc_core_*` globals for out-of-tree plugins).
- **Structured logging (`gc/log.h`).** `gc_log_level_t` is `none / error / warn / info / trace` (`trace` = 4). Use `gc_log__machine(f)` / `gc_log__host(f)`; gate hot paths with `gc_log__enabled(host, level)`.
- **Strings.** Use `gc_string_t` (heap, immutable, hash-cached; NUL-terminated at `buffer[size]`, but `size` is authoritative since content may embed NUL). No inline short strings (`gc_str_t` is not public).
- **`gc_machine__call_function` takes a `const gc_program_function_t *fn`.** On `false` the result is a synthesized `Error` object (`gc_type_object`); the caller owns one mark. `gc_machine__impersonate(ctx, user_id)` switches the effective user.
- **Host/Scheduler (`gc/host.h`).** `gc_host__cancel_task(self, task_id, requester_id, requester_permissions, out_task)` is thread-safe and permission-checked; `gc_host__spawn_task` takes a `u64_t user_permissions` mask; `gc_host__options(self)` exposes resolved CLI/env config (`gc/env.h`); `gc_host__run_function(self)` tells a start hook which function `greycat run` will call. Periodic scheduling via `gc_scheduler_t` / `gc_periodic_task_t` / `gc_periodicity_t`. See [api_runtime_storage.md](references/api_runtime_storage.md).
- **Nodes.** `gc_node__resolve` / `gc_machine_native__node_get` / `node_set_at` work across node kinds via a `u64_t node_ref`; `nodeList` positions are not raw keys since 9.0 (append only via `node_set_at(..., add_semantic = true, ...)`).
- **Tensors.** `gc_tensor_t` / `gc_tensor_descriptor_t` (`descriptor.type` is a `gc_tensor_type_t`); `data` survives a reshape to zero elements — test `descriptor.size`. A tensor over caller-owned data (`gc_machine__init_tensor` + proxy) must not be re-shaped. See [api_collections.md](references/api_collections.md).
- **Streams (`gc/stream.h`, not in `greycat.h`).** Appends only fill a buffer (flushed by size / step / reader / clean shutdown); call `gc_stream__flush` before persisting a drain offset.
- **Map keys.** `gc_map__set` stores object keys uncopied; keys hash with `gc_slot__hash` and compare with `gc__deep_equals` (`int` and `float` of equal value are one key).

## Contents

1. **C API** - Native function implementation, tensor operations, object manipulation, maps, arrays, tables, geospatial, time/date, crypto, buffers, I/O
2. **Standard Library (std)** - GCL runtime features, I/O, collections, and utilities
3. **Plugin Development** - Complete guide to building native plugins with lifecycle hooks, type configuration, and real-world patterns

---

# GreyCat C API

## Core Concepts

**gc_machine_t** - Execution context passed to all native functions. Use to get parameters, set results, report errors, create objects, and access scratch buffers.

**gc_slot_t** - Universal value container (tagged union) holding any GreyCat value: integers, floats, bools, objects, enums, tuples, etc.

**gc_type_t** - Type system enum (8-bit, 26 values, renumbered in 9.0) defining all GreyCat types: null, bool, char, int, float, node variants (node, nodeTime, nodeSet, nodeIndex, nodeIndexSet, nodeList, nodeGeo), geo, time, duration, cubic, static_field, object, block_ref, block_inline, function, undefined, type, field, stringlit, error.

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

**Enum parameters & results (CRITICAL — #1 native bug).** GCL enum values are **NOT** `gc_type_int`; they are `gc_type_static_field` with the entry position in `.tu32.right` (`.tu32.left` is the enum type offset), never `.i64`. Since 9.0 compare/return positions through nativegen's link-resolved `gc_<mod>_<Enum>_<entry>` variables rather than literals:
```c
// WRONG — always hits the default fallback:  (type == gc_type_int) ? slot.i64 : 0
// CORRECT:
i64_t variant = (gc_machine__get_param_type(ctx, 0) == gc_type_static_field) ? (i64_t) slot.tu32.right : 0;
// Returning MyEnum::variant2:
gc_machine__set_result(ctx, (gc_slot_t){.tu32 = {.left = 0, .right = gc_mymod_MyEnum_variant2}}, gc_type_static_field);
```

**Errors:** `gc_machine__set_runtime_error(ctx, "msg")` / `gc_machine__set_runtime_error_syserr(ctx)` (uses errno); check propagated errors with `if (gc_machine__error(ctx)) return;`.

## Detailed Reference

The C API reference is split by domain — each file below is linked directly (one level deep) and loads on demand.

**[references/api_core.md](references/api_core.md)** — value model, execution context, type system, logging. **Start here.**
- Native C functions with `gc_machine_t` (params, result, errors, `gc_machine__allocator`, `gc_machine__impersonate`, `gc_machine__call_function` via `gc_program_function_t *`, object creation)
- `gc_type_t` / `gc_slot_t` value model, complex c64/c128 arithmetic (`gc/complex.h`), `gc_node__parse`, `gc_error_code_t`
- Program/Type system: linking (`gc_program__link_mod_fn` / `gc_program__link_type_fn`), type configuration, introspection, enum entry resolution + fixed-value enum maps, `.gcp` load/save/check, iterator params, `gc_duration_unit_t`
- Structured logging (`gc/log.h`) and the cross-cutting Conventions & Patterns index

**[references/api_memory_text.md](references/api_memory_text.md)** — memory, buffers, strings, objects
- Memory allocation: per-call, plugin-global, aligned, thread-bound helpers, `gc_alloc__create(bool shared)`, sizing/stats
- Buffer building and binary read/write (varint, zig-zag encoding)
- Heap-allocated immutable strings (`gc_string_t`, allocator-aware constructors)
- Object/field manipulation, type introspection, GC mark/un-mark, serialization

**[references/api_collections.md](references/api_collections.md)** — array, map, table, tensor, C hash maps
- Array, Map, Table get/set/add operations
- Tensors: `gc_tensor_type_t`, `init_Nd`, get/set/add for i32/i64/f32/f64/c64/c128, descriptor utilities, raw data access, matmul/bias/sum
- `gc_u32_map_t` / `gc_u64_map_t` / `gc_str_map_t` (new in 9.0)

**[references/api_runtime_storage.md](references/api_runtime_storage.md)** — runtime, persistence, graph nodes
- Host/Task management (spawn, cancel, status), periodic scheduler (`gc_scheduler_t`, `gc_periodic_task_t`), plugin-global allocator (`gc_host__allocator` / `gc_host__global_allocator`)
- Storage blocks (attach/detach objects), ABI (allocator-aware, proto=4, `is_static`, truncated-header error, signed `@precision`), file I/O (open/sync), streams (`gc/stream.h`: buffered, `gc_stream__flush`, ndjson/csv), resolved env options (incl. metrics / flush / TLS / task retention)
- Node resolution (`gc_node__resolve`, `gc_node__parse`) and direct node-entry read/write (`gc_machine_native__node_get` / `node_set_at` via `gc_node_single_value_t`, released with `gc_node_single_value__clear`); `node_get` takes an `expected_type_id` (`prog->core.nodeTime` / `.nodeGeo` / `.nodeIndex`) and refuses other block types

**[references/api_services.md](references/api_services.md)** — crypto, geo, time, math, util
- Cryptography (SHA-256, HMAC-SHA-256, Base64, Base64URL)
- Geospatial (geohashing, Haversine distance, geo boxes with BIGMIN/LITMAX); Time/Date/Timezone (dtz zone indexes, `GC_DTZ_UTC`, `gc_dtz_time__print` / `parse`)
- Math (WASM shims, f64 PI/TAU/E/LN2 constants); Utility (Morton codes, hex, number/char parsing, deep equality, sorting with allocator)

---

# GreyCat Standard Library (std)

## Module Organization

- **std::core** - Fundamental types (Date, Time, Duration, Tuple, Error, nodes incl. nodeSet / nodeIndexSet, Tensor, geospatial types, enumerations)
- **std::runtime** - Scheduler, Task, Job, Logger, Identity/Security (per-permission grants, tokens), System, ChildProcess, raw HTTP (`runtime::HttpRequest` / `HttpResponse`), Worlds, OpenAPI, MCP
- **std::io** - Text/Binary I/O, CSV, JSON, HTTP client, Email/SMTP, FileWalker, Stream (XML and S3 live in the `xml` / `s3` libraries)
- **std::util** - Collections (Queue, Stack, SlidingWindow, TimeWindow), Statistics (Gaussian, Histogram, GaussianProfile), Quantizers (Linear/Log/Custom/Multi), Assert, ProgressTracker, Crypto, Uuid, Random

## Detailed Reference

**File:** [references/standard_library.md](references/standard_library.md)

**Load when working with:**
- Task scheduling and automation (Scheduler with periodicities)
- File I/O operations (CSV, JSON, binary files)
- HTTP integration and REST APIs, webhook handlers (`runtime::HttpRequest` / `HttpResponse`)
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
gc_program_library__set_codegen_hook(lib, lib_codegen);   // optional: `greycat codegen <lang>`
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
- Mapping GCL enums to C library enums (link-resolved entry variables)
- Using the buffer reuse and tokenization retry patterns
- Implementing conditional logging (gc_machine__log_level / gc_log__enabled)

**Contains:** Complete project structure, CMake configuration, GCL type definitions, nativegen implementation, lifecycle hooks, custom type configuration with finalizers, global state management, memory management patterns, parameter handling (including type checking with gc_object__is_instance_of), result returning, error handling, conditional logging, and a full end-to-end plugin example.
