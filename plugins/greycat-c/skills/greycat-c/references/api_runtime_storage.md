# GreyCat C SDK — Runtime & Storage: Host/Scheduler, Block, ABI, I/O & Nodes

Runtime orchestration (tasks, periodic scheduler, plugin-global allocator), persistent storage blocks, the ABI/serialization layer, file I/O, and graph node resolution. The `u64_t node_ref` node API is uniform across `node`/`nodeTime`/`nodeList`/`nodeGeo`/`nodeIndex` — only key encoding differs per variant.

_Part of the GreyCat C SDK reference (each file is linked from the skill's SKILL.md). Sibling references: api_core.md · api_memory_text.md · api_collections.md · api_runtime_storage.md · api_services.md._

## Contents

- [gc/host.h — Host, Tasks & Scheduler](#gchost-h)
- [gc/block.h — Storage Blocks](#gcblock-h)
- [gc/abi.h — ABI (Application Binary Interface)](#gcabi-h)
- [gc/io.h — File I/O](#gcio-h)
- [gc/node.h — Node Resolution](#gcnode-h)

---

<a id="gchost-h"></a>
## gc/host.h — Host, Tasks & Scheduler

The host manages the GreyCat runtime: program, workers, task queue, the plugin-global allocator, external request dispatch, and the periodic-task scheduler.

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
    gc_task_status_empty             = 0,  // Slot is empty
    gc_task_status_waiting           = 1,  // Queued, waiting to run
    gc_task_status_running           = 2,  // Currently executing
    gc_task_status_await             = 3,  // Suspended (awaiting I/O or sub-task)
    gc_task_status_cancelled         = 4,  // Cancelled by user
    gc_task_status_error             = 5,  // Completed with error
    gc_task_status_ended             = 6,  // Completed successfully
    gc_task_status_ended_with_errors = 7,  // Completed with partial errors
    gc_task_status_breakpoint        = 8,  // Stopped at a breakpoint (debugging)
} gc_task_status_t;
```

### Scheduler Enums

```c
typedef enum {
    gc_periodicity_fixed,
    gc_periodicity_daily,
    gc_periodicity_weekly,
    gc_periodicity_monthly,
    gc_periodicity_yearly
} gc_periodicity_type_t;

typedef enum {
    gc_day_mon = 0, gc_day_tue, gc_day_wed, gc_day_thu,
    gc_day_fri, gc_day_sat, gc_day_sun
} gc_day_of_week_t;

typedef enum {
    gc_month_jan = 0, gc_month_feb, gc_month_mar, gc_month_apr,
    gc_month_may, gc_month_jun, gc_month_jul, gc_month_aug,
    gc_month_sep, gc_month_oct, gc_month_nov, gc_month_dec
} gc_month_t;
```

### Scheduler Structures

```c
typedef struct {
    u8_t day;          // 1-31
    gc_month_t month;
} gc_date_tuple_t;

typedef struct {
    u8_t hour;         // 0-23
    u8_t minute;       // 0-59
    u8_t second;       // 0-59
    u32_t timezone;    // TimeZone enum field offset
} gc_daily_timing_t;

typedef struct {
    gc_periodicity_type_t type;
    union {
        struct { i64_t every_us; } fixed;
        struct { gc_daily_timing_t timing; } daily;
        struct {
            gc_day_of_week_t *days;         // allocator-owned
            u8_t nb_days;
            gc_daily_timing_t timing;
        } weekly;
        struct {
            i32_t *days;                    // allocator-owned, positive or negative day-of-month
            u8_t nb_days;
            gc_daily_timing_t timing;
        } monthly;
        struct {
            gc_date_tuple_t *dates;         // allocator-owned
            u8_t nb_dates;
            u32_t timezone;                 // TimeZone enum field offset
        } yearly;
    } config;
} gc_periodicity_t;

typedef struct {
    bool immediate;            // default true
    bool activated;            // default true
    i64_t start_time_us;       // default: current time (µs since epoch)
    i64_t max_duration_us;     // 0 for unlimited
} gc_periodic_options_t;

typedef struct {
    i64_t task_id;
    u32_t fn_off;              // function offset in the program
    gc_periodicity_t periodicity;
    gc_periodic_options_t options;
    i64_t next_execution_us;
    u64_t execution_count;
} gc_periodic_task_t;
```

### Host Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_host__get_global` | `gc_host_t *gc_host__get_global()` | Get the global singleton host instance. |
| `gc_host__allocator` | `gc_allocator_t *gc_host__allocator(const gc_host_t *self)` | Get the host's allocator (used for `lib_start`/`lib_stop` state when called on the global host). |
| `gc_host__global_allocator` | `gc_allocator_t *gc_host__global_allocator()` | Convenience: same as `gc_host__allocator(gc_host__get_global())`. |
| `gc_host__program` | `const gc_program_t *gc_host__program(const gc_host_t *host)` | Get the program from the host. |
| `gc_host__scheduler` | `gc_scheduler_t *gc_host__scheduler(gc_host_t *self)` | Get the host's periodic scheduler. |
| `gc_host__spawn_task` | `bool gc_host__spawn_task(gc_host_t *self, u32_t fn_off, u32_t user_id, u64_t roles_flags, i64_t *created_task_id)` | Spawn a new task (no arguments). Returns `false` when the queue is full. |
| `gc_host__spawn_task_with_args` | `bool gc_host__spawn_task_with_args(gc_host_t *self, u32_t fn_off, const char *args_payload, u64_t args_payload_len, gc_format_t args_format, u32_t user_id, u64_t roles_flags, i64_t *created_task_id, gc_buffer_t *extra_buffer)` | Spawn a new task with serialized arguments. |
| `gc_host__cancel_task` | `bool gc_host__cancel_task(gc_host_t *self, i64_t task_id)` | Cancel a running or queued task. |
| `gc_host__get_task_status` | `bool gc_host__get_task_status(gc_host_t *self, i64_t task_id, gc_task_status_t *status)` | Query the current status of a task. |
| `gc_host__add_request` | `bool gc_host__add_request(u32_t fn, char *data, u32_t data_len)` | Add a request to the host's request queue. |

### Scheduler Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_scheduler__add` | `bool gc_scheduler__add(gc_scheduler_t *self, gc_periodic_task_t task, gc_allocator_t *allocator)` | Register a periodic task. Any dynamic arrays inside `task.periodicity` (weekly/monthly/yearly day lists) must live on the allocator you pass here. |
| `gc_scheduler__activate` | `bool gc_scheduler__activate(gc_scheduler_t *self, u32_t fn_off)` | Activate the periodic task bound to `fn_off`. |
| `gc_scheduler__deactivate` | `bool gc_scheduler__deactivate(gc_scheduler_t *self, u32_t fn_off)` | Deactivate the periodic task bound to `fn_off`. |
| `gc_scheduler__create_object` | `gc_object_t *gc_scheduler__create_object(gc_periodic_task_t *task, gc_machine_t *ctx)` | Build a GreyCat `Task` object representation of a `gc_periodic_task_t`. |

### Usage Examples

#### Resolving the global host and its allocator

`gc_host__get_global()` returns the process-wide singleton; `gc_host__global_allocator()` is shorthand for `gc_host__allocator(gc_host__get_global())`. Use the host allocator for state that must outlive a single task (e.g. `lib_start`/`lib_stop` plugin state), and `gc_host__program()` to introspect the loaded program.

```c
gc_host_t *host = gc_host__get_global();

// Allocator that survives across tasks — the same as gc_host__allocator(host)
gc_allocator_t *galloc = gc_host__global_allocator();

// Read-only access to the loaded program
const gc_program_t *prog = gc_host__program(host);

// Persistent plugin state allocated on the host allocator (not the task ctx)
my_state_t *state = gc_alloc__calloc(galloc, sizeof(my_state_t));
```

#### Spawning a task with no arguments

`gc_host__spawn_task()` queues a function (identified by its program offset `fn_off`) for execution under a given `user_id` and role flags. It writes the new task id into `created_task_id` and returns `false` when the task queue is full.

```c
gc_host_t *host = gc_host__get_global();

i64_t task_id;
// UINT64_MAX = all roles; 1u is the boot/system user id
if (!gc_host__spawn_task(host, fn_off, 1u, UINT64_MAX, &task_id)) {
    // Queue is full — back off and retry later
    return;
}
```

#### Spawning a task with serialized arguments

`gc_host__spawn_task_with_args()` is the same as `gc_host__spawn_task()` but accepts a serialized argument payload. The `extra_buffer` is a scratch `gc_buffer_t` the runtime uses internally to build the params file path; pass any reusable buffer. The payload bytes must match `args_format`.

```c
gc_host_t *host = gc_host__get_global();
gc_buffer_t *scratch = gc_machine__get_buffer(ctx); // any reusable buffer works

i64_t task_id;
const bool ok = gc_host__spawn_task_with_args(host,
                                               fn_off,
                                               args_payload,      // const char *
                                               args_payload_len,  // u64_t
                                               gc_args_format_json,
                                               user_id,
                                               UINT64_MAX,
                                               &task_id,
                                               scratch);
if (!ok) {
    // queue full or argument file could not be written
}
```

#### Querying and cancelling a task

`gc_host__get_task_status()` fills a `gc_task_status_t` and returns `false` if the task id is unknown. `gc_host__cancel_task()` cancels a queued or running task.

```c
gc_host_t *host = gc_host__get_global();

gc_task_status_t status;
if (gc_host__get_task_status(host, task_id, &status)) {
    switch (status) {
    case gc_task_status_running:
    case gc_task_status_waiting:
        gc_host__cancel_task(host, task_id);
        break;
    case gc_task_status_ended:
    case gc_task_status_error:
    case gc_task_status_ended_with_errors:
        // terminal — nothing to cancel
        break;
    default:
        break;
    }
}
```

#### Registering a periodic task with the scheduler

`gc_host__scheduler()` returns the host's scheduler. `gc_scheduler__add()` takes the task **by value** and an allocator: any dynamic arrays inside the periodicity config (weekly/monthly/yearly day lists) must be allocated on the allocator you pass — use `gc_host__allocator(host)` so they live as long as the scheduler. Zero-initialize the task and set only the fields you need. For trace-level diagnostics, build a GreyCat `Task` object with `gc_scheduler__create_object()` and unmark it after use.

```c
gc_host_t *host = gc_machine__get_host(ctx);
gc_allocator_t *galloc = gc_host__allocator(host);

gc_periodic_task_t task = {0};
task.fn_off = fn_off;

// Run every 60 seconds (durations are in microseconds)
task.periodicity.type = gc_periodicity_fixed;
task.periodicity.config.fixed.every_us = 60LL * 1000 * 1000;

task.options.immediate = true;   // also fire once right away
task.options.activated = true;

if (gc_scheduler__add(gc_host__scheduler(host), task, galloc)) {
    // Optional: materialize a runtime::PeriodicTask object for logging
    gc_object_t *task_obj = gc_scheduler__create_object(&task, ctx);
    gc_log__machine(ctx, gc_log_level_trace, (gc_slot_t){.object = task_obj}, gc_type_object);
    gc_object__un_mark(task_obj, ctx);
}
```

A weekly schedule allocates its `days` array on the same allocator passed to `gc_scheduler__add`:

```c
gc_periodic_task_t task = {0};
task.fn_off = fn_off;
task.periodicity.type = gc_periodicity_weekly;
task.periodicity.config.weekly.nb_days = 2;
task.periodicity.config.weekly.days =
    gc_alloc__calloc(galloc, 2 * sizeof(gc_day_of_week_t));
task.periodicity.config.weekly.days[0] = gc_day_mon;
task.periodicity.config.weekly.days[1] = gc_day_thu;
task.periodicity.config.weekly.timing = (gc_daily_timing_t){
    .hour = 8, .minute = 30, .second = 0, .timezone = 0 /* TimeZone enum offset, e.g. UTC */,
};
gc_scheduler__add(gc_host__scheduler(host), task, galloc);
```

#### Activating and deactivating periodic tasks

Both lookups key off the bound `fn_off` and return `false` when no task matches.

```c
gc_scheduler_t *sched = gc_host__scheduler(gc_machine__get_host(ctx));

bool found = gc_scheduler__deactivate(sched, fn_off);
// ... later ...
found = gc_scheduler__activate(sched, fn_off);
```

#### Dispatching an external request

`gc_host__add_request()` is a global entry point (no host argument — it targets the global host) used to enqueue an inbound request for function `fn` with raw payload bytes. It returns `false` when the request cannot be queued (and is a no-op stub in WASM/standalone builds).

```c
if (!gc_host__add_request(fn, data, data_len)) {
    // request queue rejected the payload
}
```


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
    bool value_only : 1;       // Block stores values only (no sub-blocks)
    bool cmp_is_full : 1;      // Compression covers entire block
    bool cmp_with_values : 1;  // Compressed with values
    bool cmp_up_to_date : 1;   // Compression is current
    bool cmp_found : 1;        // Compression data found
    bool dropped : 1;          // Block has been dropped
    bool contains_objects : 1; // Block contains object entries
    bool dirty : 1;            // Block has been modified since last save
    gc_block_slot_t *backend;  // Array of entries
};
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_block__is_stub` | `bool gc_block__is_stub(const gc_block_t *b)` | Check if the block is the program's stub (volatile placeholder) |
| `gc_block__object_attach` | `bool gc_block__object_attach(gc_block_t *b, gc_object_t **obj, gc_machine_t *ctx)` | Attach an object to this block (for persistence) |
| `gc_block__object_detach` | `void gc_block__object_detach(gc_block_t *b, gc_object_t *obj, gc_machine_t *ctx)` | Detach an object from this block |

### Usage Examples

These functions are low-level persistence primitives. SDK code normally touches them only when implementing custom node-storage operations; the snippets below mirror the runtime's own usage.

**Detect whether an object lives in a volatile stub block**

`gc_block__is_stub` returns `true` when a block is the program's volatile placeholder (`b->origin == UINT64_MAX`). Use it to decide whether an object is already persisted or still detached/program-literal before saving or re-parenting it.

```c
// Before persisting a stack value, only stub-backed objects are safe to save inline.
gc_object_t *obj = slot.object;
if (obj->block != NULL) {
    if (gc_block__is_stub(obj->block)) {
        // object is not yet attached to a real node block -> safe to serialize
        gc_slot__save(slot, slot_type, buf, ctx, false);
    } else {
        // already stored inside a node block: this is an illegal state here
        gc_machine__set_runtime_error(ctx, "object already stored in a node");
    }
}
```

**Attach an object to a block (with error handling)**

`gc_block__object_attach(gc_block_t *b, gc_object_t **obj, gc_machine_t *ctx)` takes a *pointer to* the object reference because it may rewrite it (e.g. a `String` literal is copied into a fresh attachable instance). It returns `false` if the object is already attached to another node or is volatile; on failure you must unwind your object mark and raise a runtime error.

The destination block is whatever block already backs the persistent slot you are writing into — for example the block reachable through an already-attached parent object (`parent->block`). The SDK does not expose a block-creation primitive; new blocks are produced internally by the node API (`gc_machine_native__node_set_at`), so reach for that when you need to add an entry, and use `gc_block__object_attach` only to re-home an object into a block you already hold.

```c
// Attach an object value into a block you already hold (e.g. the parent's block).
gc_block_t *block = parent->block; // a real, already-persisted node block

if (value_type == gc_type_object) {
    if (!gc_block__object_attach(block, &value.object, ctx)) {
        // attach failed: release the object mark and signal the error
        gc_object__un_mark(value.object, ctx);
        gc_machine__set_runtime_error(ctx, "failed to attach object to node block");
        return;
    }
    // gc_block__object_attach may have rewritten value.object (e.g. String literal
    // copied into a fresh attachable instance); release the mark we held on it.
    gc_object__un_mark(value.object, ctx);
}
```

**Re-parent an object: detach then attach**

`gc_block__object_detach(gc_block_t *b, gc_object_t *obj, gc_machine_t *ctx)` is a no-op when the object has no block or is stub-backed, so the canonical move-between-blocks idiom guards with `gc_block__is_stub` and always detaches before attaching.

```c
// Moving an object slot from current_block into new_block during a block split.
if (slot->value_type == gc_type_object) {
    gc_block_t *value_block = slot->value.object->block;
    if (value_block != NULL && !gc_block__is_stub(value_block)) {
        gc_block__object_detach(current_block, slot->value.object, ctx);
        gc_block__object_attach(new_block, &slot->value.object, ctx);
        new_block->contains_objects = 1;
    }
}
```


---

<a id="gcabi-h"></a>
## gc/abi.h — ABI (Application Binary Interface)

The ABI defines the binary wire format for GreyCat's type system — how types, attributes, and functions are described in a version-controlled binary format. It enables safe schema evolution and cross-version compatibility.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_ABI_PROTO` | 3 | Current ABI protocol version |
| `GC_ABI_HEADER_SZ` | 8 | ABI header size in bytes |
| `GC_ABI_FUNCTION_MAX_PARAMS` | 16 | Maximum parameters per function |

### Types

```c
typedef u32_t gc_symbol_t;  // Symbol reference
```

### Precision Enum

```c
typedef enum {
    gc_abi_precision_1            = 0,
    gc_abi_precision_10           = 1,
    gc_abi_precision_100          = 2,
    gc_abi_precision_1000         = 3,
    gc_abi_precision_10000        = 4,
    gc_abi_precision_100000       = 5,
    gc_abi_precision_1000000      = 6,
    gc_abi_precision_10000000     = 7,
    gc_abi_precision_100000000    = 8,
    gc_abi_precision_1000000000   = 9,
    gc_abi_precision_10000000000  = 10
} gc_abi_precision_t;
```

Used for numeric precision/quantization (e.g., storing a float as an integer with known precision).

### ABI Symbols Table

```c
typedef struct {
    gc_buffer_t all;         // Concatenated symbol bytes
    i64_t *inverted_index;   // Inverted lookup table
    u32_t nb;                // Number of symbols
} gc_abi_symbols;
```

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
    u8_t param_nullable[GC_ABI_FUNCTION_MAX_PARAMS];  // Per-parameter nullable flags
    u32_t param_types[GC_ABI_FUNCTION_MAX_PARAMS];    // Per-parameter type descriptors
    u32_t param_symbols[GC_ABI_FUNCTION_MAX_PARAMS];  // Per-parameter name symbols
    u32_t return_type;           // Return type descriptor
    u8_t return_nullable;        // Whether return is nullable
} gc_abi_function_t;
```

### ABI Container

```c
typedef struct {
    gc_allocator_t *allocator;      // Allocator owning the internal arrays
    gc_abi_symbols symbols;         // Symbol table
    gc_abi_attribute_t *attributes; // Attribute array
    gc_abi_type_t *types;           // Type array
    gc_abi_function_t *functions;   // Function array
    u32_t attributes_len;
    u32_t types_len;
    u32_t functions_len;
    u32_t version;                  // ABI version counter (incremented on schema changes)
    u16_t magic;                    // Magic number for validation
    u64_t crc;                      // CRC checksum
} gc_abi_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_abi__create` | `gc_abi_t *gc_abi__create(gc_allocator_t *allocator)` | Allocate a new ABI instance bound to `allocator`. |
| `gc_abi__finalize` | `void gc_abi__finalize(gc_abi_t *abi)` | Free all ABI memory (uses the allocator stored on the ABI). |
| `gc_abi__save` | `void gc_abi__save(const gc_abi_t *abi, gc_buffer_t *output)` | Serialize the entire ABI to a buffer. |
| `gc_abi__load` | `bool gc_abi__load(gc_abi_t *abi, gc_buffer_t *input)` | Deserialize an ABI from a buffer. |
| `gc_abi__get_symbol` | `gc_string_t *gc_abi__get_symbol(const gc_abi_t *abi, gc_symbol_t symb)` | Retrieve a symbol string by its symbol ID. |
| `gc_abi__save_header` | `void gc_abi__save_header(const gc_abi_t *abi, gc_buffer_t *b)` | Write only the ABI header to a buffer. |
| `gc_abi__check_header` | `gc_abi_header_check_error_t gc_abi__check_header(const gc_abi_t *abi, gc_buffer_t *b)` | Validate an ABI header against the current ABI. |

### Header Check Errors

```c
typedef enum {
    gc_abi_header_check_error_none          = 0,  // No error
    gc_abi_header_check_error_wrong_proto   = 1,  // Protocol version mismatch
    gc_abi_header_check_error_wrong_magic   = 2,  // Magic number mismatch
    gc_abi_header_check_error_wrong_version = 3,  // ABI version mismatch
    gc_abi_header_check_error_truncated     = 4   // Truncated header bytes
} gc_abi_header_check_error_t;
```

### Usage Examples

The ABI is normally created once at host startup, bound to the global allocator, and then attached to a `gc_program_t`. The program owns the ABI for its lifetime and frees it at shutdown.

```c
// Host startup: create the ABI, hand it to the program (program.c keeps it).
gc_allocator_t *global_allocator = gc_alloc__create(true);

gc_abi_t *abi = gc_abi__create(global_allocator);
gc_program_t *program = gc_program__create(abi, global_allocator);

// ... run the host using `program` ...

// Shutdown: release every ABI-owned array (symbols, types, attributes,
// functions). Uses the allocator stored on the ABI, so no allocator arg.
gc_abi__finalize(abi);
```

**Persisting the ABI to and from a buffer.** `gc_abi__save` writes the full
ABI (symbol table, types, attributes, functions, header) into a buffer;
`gc_abi__load` reconstructs it. `gc_abi__load` returns `false` on a malformed
or truncated buffer, so always check it.

```c
gc_buffer_t *buf = gc_buffer__create(global_allocator);

// Serialize the whole ABI to bytes (e.g. before writing the "abi" file).
gc_buffer__clear(buf);
gc_abi__save(abi, buf);
// buf->data / buf->size now hold the serialized ABI; persist as needed.

// Later: deserialize into a fresh ABI bound to an allocator.
gc_abi_t *loaded = gc_abi__create(global_allocator);
buf->current = buf->data;            // rewind read cursor
if (!gc_abi__load(loaded, buf)) {
    // corrupt / truncated / wrong version — bail out
    gc_abi__finalize(loaded);
    return false;
}
```

**Header validation on the wire.** Binary requests/responses are prefixed with
the ABI header. The receiver writes its own header with `gc_abi__save_header`
and the peer validates the incoming header with `gc_abi__check_header`, which
returns a `gc_abi_header_check_error_t`. Anything other than
`gc_abi_header_check_error_none` means the client's schema is incompatible.

The ABI is reachable from the public, fully-defined `gc_program_t` (`prog->abi`),
which you obtain from the host with `gc_host__program(host)` or from the machine
context with `gc_machine__program(ctx)`.

```c
// Sender side: prefix the response buffer with the current ABI header,
// then serialize the payload after it.
const gc_program_t *prog = gc_machine__program(ctx);
gc_buffer_t *out = gc_machine__get_buffer(ctx);
gc_abi__save_header(prog->abi, out);
gc_slot__save((gc_slot_t){.b = found}, gc_type_bool, out, ctx, false);

// Receiver side: validate an incoming header against the current ABI.
// `in` is a gc_buffer_t* whose read cursor sits at the start of the peer's header.
gc_abi_header_check_error_t check_err =
    gc_abi__check_header(prog->abi, in);
if (check_err != gc_abi_header_check_error_none) {
    // wrong_proto / wrong_magic / wrong_version / truncated
    gc_machine__set_runtime_error(ctx, "incompatible ABI header from peer");
    return;
}
```

**Resolving symbols.** Symbol IDs (`gc_symbol_t`) index into the ABI symbol
table. `gc_abi__get_symbol` returns the interned `gc_string_t *` for an ID.
Note symbol 0 is reserved, so iteration starts at 1.

```c
// Look up a symbol string by its ID, and resolve a name back to an ID.
gc_string_t *name = gc_abi__get_symbol(abi, symb);          // ID -> string
u32_t symb = gc_program__resolve_symbol(prog, "MyType", 6); // string -> ID (0 if absent)
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
| `gc_io__write_all` | `bool gc_io__write_all(i32_t fd, const char *data, u64_t data_len, u64_t *written)` | Write all `data_len` bytes of `data` to `fd`, retrying partial writes and `EINTR`. Returns `true` on success; `false` on error with `errno` set (`EIO` when `write` returns 0). `*written`, if non-NULL, is advanced by the number of bytes written. |

### Usage Examples

**Open a file for reading, then close it**

`gc_io_file__open_read` is a thin wrapper over `gc_io_open` with `O_RDONLY | O_BINARY`. It returns a raw POSIX file descriptor (or `-1` on error, with the runtime error already set on `ctx`). The caller owns the descriptor and must `close()` it.

```c
// path is a gc_string_t* already resolved (e.g. from an object field)
i32_t fp = gc_io_file__open_read(path, ctx);
if (fp == -1) {
    // gc_io_open already called gc_machine__set_runtime_error_syserr(ctx).
    // To attach context, overwrite it with a descriptive static message instead:
    gc_machine__set_runtime_error(ctx, "could not open file for reading");
    return;
}

// stat / mmap / read from fp ... (standard POSIX fstat on the returned fd)
struct stat s;
if (fstat(fp, &s) != 0) {
    close(fp);
    gc_machine__set_runtime_error(ctx, "error while getting file stats");
    return;
}

close(fp);
```

**Open with custom flags (write / append / truncate)**

Use `gc_io_open` directly when you need explicit `open(2)` flags. For write/create flags (`O_RDWR`/`O_WRONLY`), `gc_io_open` auto-creates missing parent directories along the path. Always OR in `O_BINARY` for portability.

```c
// Truncate-or-create for a fresh write, append to extend an existing file
i32_t fp = append
    ? gc_io_open(path, O_WRONLY | O_CREAT | O_APPEND | O_BINARY, ctx)
    : gc_io_open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, ctx);
if (fp == -1) {
    // ctx error already set by gc_io_open
    return;
}

// ... write bytes to fp ...

gc_io_file__sync(fp);   // flush buffered data to disk (fsync)
close(fp);
```

**Read/write descriptor with directory creation**

`gc_io_file__open_rdwr` opens with `O_RDWR | O_CREAT | O_TRUNC | O_BINARY`, so it creates the file (and any missing parent directories) fresh. Pair it with `gc_io_file__open_read` for round-trips such as staging then re-reading a downloaded artifact.

```c
i32_t out_fd = gc_io_file__open_rdwr(filepath, ctx);
if (out_fd == -1) {
    return; // error set on ctx
}
// ... write the payload, then make it durable ...
gc_io_file__sync(out_fd);
close(out_fd);

// later, read it back
i32_t in_fd = gc_io_file__open_read(filepath, ctx);
if (in_fd == -1) {
    return;
}
// ... consume ...
close(in_fd);
```

**Write a whole buffer with partial-write/`EINTR` handling**

`gc_io__write_all` is the robust counterpart to a raw `write(2)`: it loops until every byte of `data` is written, restarting on `EINTR` and resuming after partial writes. It returns `false` with `errno` set on error (`EIO` when `write` returns 0), and advances `*written` (if non-NULL) by the number of bytes actually written so the caller can track progress.

```c
i32_t out_fd = gc_io_file__open_rdwr(filepath, ctx);
if (out_fd == -1) {
    return; // error set on ctx
}

u64_t written = 0;
if (!gc_io__write_all(out_fd, buf->data, buf->size, &written)) {
    close(out_fd);
    gc_machine__set_runtime_error_syserr(ctx); // errno set by gc_io__write_all
    return;
}
gc_io_file__sync(out_fd); // flush buffered data to disk (fsync)
close(out_fd);
```


---

<a id="gcnode-h"></a>
## gc/node.h — Node Resolution

Functions for resolving node references from their compact `u64_t` representation into full slot values, and for reading/writing entries on node-backed collections.

> **These functions are NOT specific to plain `node` — the same API drives every node-backed collection variant.** All of `node`, `nodeTime`, `nodeList`, `nodeGeo`, and `nodeIndex` are stored on the same Sorted-Block-Index (SBI) backend and are addressed by an opaque `u64_t node_ref` (a packed block-id + in-block offset). `gc_node__resolve`, `gc_machine_native__node_get`, `gc_machine_native__node_set_at`, and `gc_node_single_value__clear` all take that `node_ref` and work identically regardless of which variant produced it. The runtime's own implementations confirm this: `node.c`, `nodetime.c`, `nodelist.c`, `nodegeo.c`, and `nodeindex.c` are thin wrappers that all delegate to this one shared family. So a native function holding a `node_ref` for any variant can read, write, and resolve entries with these calls — **the only thing that varies per variant is how the GCL-level key is encoded into the raw `u64_t key`** (see *Key encoding per variant* below).

### Single-Value Result

```c
typedef struct {
    gc_slot_t key;         // Entry key
    gc_slot_t value;       // Entry value
    gc_type_t value_type;  // Value type tag
    gc_type_t key_type;    // Key type tag
    u32_t node_type;       // Node type id
} gc_node_single_value_t;
```

### Functions

All four work on any node variant — `node`, `nodeTime`, `nodeList`, `nodeGeo`, `nodeIndex`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_node__resolve` | `gc_slot_t gc_node__resolve(u64_t node_ref, gc_type_t *result_type, gc_machine_t *ctx)` | Resolve a node reference to its stored slot value. Writes the resolved type to `*result_type`. Marks the value if it is an object (caller must `gc_object__un_mark` it). This is the plain-`node<T>` dereference, and also the per-entry resolve used by every variant. |
| `gc_machine_native__node_set_at` | `void gc_machine_native__node_set_at(u64_t node_ref, gc_slot_t key_value, gc_type_t key_type, bool add_semantic, gc_slot_t value, gc_type_t value_type, gc_machine_t *ctx)` | Insert/update the entry identified by the typed `key_value` on `node_ref`. When `add_semantic` is `true` the key is treated as an append/auto position (e.g. `nodeList.add`). Used by `setAt`/`set`/`add` on every variant. |
| `gc_machine_native__node_get` | `gc_node_single_value_t gc_machine_native__node_get(u64_t node_ref, u64_t key, gc_machine_t *ctx)` | **Exact-key** read of a single entry (`key`/`value`/types/node type) from `node_ref`. `key` is the already-encoded raw `u64_t` (see encoding table). Returns a null single if no entry matches exactly. Backs `nodeTime.getAt`, `nodeList.get`, `nodeGeo.get`. |
| `gc_node_single_value__clear` | `void gc_node_single_value__clear(gc_node_single_value_t *value, gc_machine_t *ctx)` | Release/unmark the contents of a `gc_node_single_value_t` returned by `gc_machine_native__node_get` (and the internal `node_first`/`node_last`/`node_resolve`). **Always call this** on the returned single once you have consumed its slots, or object values leak a GC mark. |

### Key encoding per variant

`node_set_at` takes a typed key slot (`key_value` + `key_type`), but `node_get` takes the **raw `u64_t` key** that the entry is physically sorted by. Each variant maps its GCL-facing key to that raw `u64_t` differently — this is the one place the variants are not interchangeable:

| Variant | GCL key type | Raw `u64_t` key passed to `node_get` |
|---------|--------------|--------------------------------------|
| `nodeList` | `int` (index) | `(u64_t)(index + INT64_MIN)` — offset-binary so signed ints sort correctly as unsigned |
| `nodeTime` | `time` (µs since epoch) | `(u64_t)(time + INT64_MIN)` — same offset-binary on the timestamp; `nodeTime.resolve()` uses the ambient `ctx->current_time_offset` |
| `nodeGeo` | `geo` (geohash/Morton code) | the `geo` value used directly as `.u64` (already an unsigned, order-preserving code) |
| `node` | none (single slot) | n/a — a plain `node<T>` holds one value; use `gc_node__resolve` to dereference it (there is no per-key `node_get`) |
| `nodeIndex` | typed / multi-field key | not a flat `u64_t` — `nodeIndex` has no public per-key `node_get`; writes still use `node_set_at` |

> **Exact vs. floor lookup.** `gc_machine_native__node_get` is an *exact* match. The temporal/range "resolve" semantics (`nodeTime.resolve`/`resolveAt`, `nodeList.resolve`) — return the entry with the largest key ≤ the requested key — plus broader range helpers (first / last / size / range-size / remove) are powered by internal *floor* lookups declared in the runtime's internal `machine.h`, **not** in the public `gc/node.h` SDK header, so they are not guaranteed available to out-of-tree plugins. The four functions documented above are the publicly exported, uniform-across-variants surface.

### Usage Examples

#### Dereference a plain `node<T>` with `gc_node__resolve`

`gc_node__resolve` reads the single slot a `node<T>` points at and writes the resolved type through `result_type`. If the value is an object it is returned **marked**, so the caller owns one reference and must `gc_object__un_mark` it once consumed (e.g. after handing it to the result, which takes its own mark).

```c
// node<T>.resolve(): this == the node_ref
const u64_t node_ref = gc_machine__this(ctx).u64;
gc_type_t result_type = gc_type_null;
gc_slot_t result = gc_node__resolve(node_ref, &result_type, ctx);

gc_machine__set_result(ctx, result, result_type);
if (result_type == gc_type_object) {
    gc_object__un_mark(result.object, ctx); // release the mark gc_node__resolve added
}
```

Resolving many node references in a loop follows the same mark/unmark discipline per element:

```c
for (u32_t i = 0; i < param->size; i++) {
    gc_type_t arr_slot_type;
    gc_slot_t arr_slot;
    gc_array__get_slot(param, i, &arr_slot, &arr_slot_type);
    if (arr_slot_type == gc_type_node) {
        gc_type_t resolved_value_type;
        gc_slot_t resolved_value = gc_node__resolve(arr_slot.u64, &resolved_value_type, ctx);
        gc_array__add_slot(result, resolved_value, resolved_value_type, ctx);
        if (resolved_value_type == gc_type_object) {
            gc_object__un_mark(resolved_value.object, ctx); // un-mark each resolved object
        }
    } else {
        gc_array__add_slot(result, (gc_slot_t) {.object = NULL}, gc_type_null, ctx);
    }
}
```

#### Exact-key read with `gc_machine_native__node_get` + `gc_node_single_value__clear`

`node_get` takes the **raw** `u64_t` key (already encoded per variant — see the encoding table above) and returns a `gc_node_single_value_t`. Always pair it with `gc_node_single_value__clear` to release any object slots it holds. For `nodeList` / `nodeTime` the index/timestamp is converted to offset-binary with `+ INT64_MIN`:

```c
// nodeList.get(index) — offset-binary encode the signed int index
const u64_t node_ref = gc_machine__this(ctx).u64;
const u64_t to_get_key = ((u64_t) gc_machine__get_param(ctx, 0).i64 + INT64_MIN); // index param

gc_node_single_value_t single = gc_machine_native__node_get(node_ref, to_get_key, ctx);
gc_machine__set_result(ctx, single.value, single.value_type);
gc_node_single_value__clear(&single, ctx); // mandatory: frees marked object values/keys
```

```c
// nodeTime.getAt(exactTime) — same offset-binary encode on the microsecond timestamp
const u64_t node_ref = gc_machine__this(ctx).u64;
const u64_t to_get_key = ((u64_t) gc_machine__get_param(ctx, 0).i64 + INT64_MIN); // exactTime param

gc_node_single_value_t single = gc_machine_native__node_get(node_ref, to_get_key, ctx);
gc_machine__set_result(ctx, single.value, single.value_type);
gc_node_single_value__clear(&single, ctx);
```

A `nodeGeo` lookup uses the `geo` code directly as the `.u64` key (no offset shift), but the read/clear shape is identical:

```c
// nodeGeo.get(geo) — geohash/Morton code is already an order-preserving unsigned key
const u64_t to_get_key = key.u64;
gc_node_single_value_t single = gc_machine_native__node_get(node_ref, to_get_key, ctx);
gc_machine__set_result(ctx, single.value, single.value_type);
gc_node_single_value__clear(&single, ctx);
```

#### Insert / update with `gc_machine_native__node_set_at`

`node_set_at` takes the key as a **typed slot** (`key_value` + `key_type`), not a pre-encoded `u64_t` — the runtime encodes it for the variant. Pass `add_semantic = false` for an explicit set/setAt, and `true` for append-style adds where the key slot is ignored:

```c
// nodeList.set(index, value): explicit position, add_semantic = false
const u64_t node_ref = gc_machine__this(ctx).u64;
const gc_slot_t key = gc_machine__get_param(ctx, 0);            // index
const gc_type_t key_type = gc_machine__get_param_type(ctx, 0);
const gc_slot_t value = gc_machine__get_param(ctx, 1);          // value
const gc_type_t value_type = gc_machine__get_param_type(ctx, 1);
gc_machine_native__node_set_at(node_ref, key, key_type, /*add_semantic*/ false, value, value_type, ctx);
gc_machine__set_result(ctx, value, value_type);
```

```c
// nodeList.add(value): append — add_semantic = true, key slot is a placeholder
const u64_t node = gc_machine__this(ctx).u64;
const gc_slot_t value = gc_machine__get_param(ctx, 0);          // value
const gc_type_t value_type = gc_machine__get_param_type(ctx, 0);
gc_machine_native__node_set_at(node, (gc_slot_t) {.i64 = 0}, gc_type_int, /*add_semantic*/ true, value, value_type, ctx);
gc_machine__set_result(ctx, value, value_type);
```

```c
// nodeTime.setAt(exactTime, value): typed time key, add_semantic = false
const u64_t node = gc_machine__this(ctx).u64;
const gc_slot_t key = gc_machine__get_param(ctx, 0);            // exactTime
const gc_type_t key_type = gc_machine__get_param_type(ctx, 0);
const gc_slot_t value = gc_machine__get_param(ctx, 1);          // value
const gc_type_t value_type = gc_machine__get_param_type(ctx, 1);
gc_machine_native__node_set_at(node, key, key_type, /*add_semantic*/ false, value, value_type, ctx);
gc_machine__set_result(ctx, value, value_type);
```


---
