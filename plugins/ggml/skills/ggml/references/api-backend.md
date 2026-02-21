# Backend & Memory API

## Table of Contents
1. [Buffer Types](#buffer-types)
2. [Buffers](#buffers)
3. [Backend Operations](#backend-operations)
4. [Events](#events)
5. [Devices](#devices)
6. [Backend Registry](#backend-registry)
7. [Scheduler](#scheduler)
8. [Memory Allocation](#memory-allocation)
9. [CPU Backend](#cpu-backend)
10. [Utilities & Float Conversions](#utilities--float-conversions)

---

## Buffer Types

A buffer type describes how memory for tensors is allocated (CPU, GPU VRAM, etc.).

```c
const char *             ggml_backend_buft_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_t    ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
size_t                   ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft);
size_t                   ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft);
size_t                   ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
bool                     ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft);
ggml_backend_dev_t       ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft);
```

---

## Buffers

```c
const char *             ggml_backend_buffer_name(ggml_backend_buffer_t buffer);
void                     ggml_backend_buffer_free(ggml_backend_buffer_t buffer);
void *                   ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);
size_t                   ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer);
enum ggml_status         ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
size_t                   ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer);
size_t                   ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer);
size_t                   ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
void                     ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);
bool                     ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer);
void                     ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer);
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer);
void                     ggml_backend_buffer_reset(ggml_backend_buffer_t buffer);

// Copy tensor data between backends
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                    struct ggml_tensor * src, struct ggml_tensor * dst);
```

**ggml_backend_buffer_usage:**
- `GGML_BACKEND_BUFFER_USAGE_ANY`
- `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` — mark as model weights (hint for backends)
- `GGML_BACKEND_BUFFER_USAGE_COMPUTE` — intermediate activations

---

## Backend Operations

```c
ggml_guid_t              ggml_backend_guid(ggml_backend_t backend);
const char *             ggml_backend_name(ggml_backend_t backend);
void                     ggml_backend_free(ggml_backend_t backend);
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
ggml_backend_buffer_t    ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
size_t                   ggml_backend_get_alignment(ggml_backend_t backend);
size_t                   ggml_backend_get_max_size(ggml_backend_t backend);

// Transfer data to/from backend tensors
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);

// Compute
void             ggml_backend_synchronize(ggml_backend_t backend);
enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

// Graph plans (pre-compiled execution plan)
ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
void                      ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
enum ggml_status          ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

// Capability queries
bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
```

---

## Events

Synchronization primitives for async compute:

```c
ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
void                 ggml_backend_event_free(ggml_backend_event_t event);
void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
void                 ggml_backend_event_synchronize(ggml_backend_event_t event);  // CPU waits for GPU event
void                 ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event); // GPU waits for event
```

---

## Devices

```c
const char *           ggml_backend_dev_name(ggml_backend_dev_t device);
const char *           ggml_backend_dev_description(ggml_backend_dev_t device);
void                   ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device);
void                   ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
ggml_backend_reg_t     ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
ggml_backend_t         ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
ggml_backend_buffer_t  ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);
bool                   ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
bool                   ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
bool                   ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
```

**ggml_backend_dev_type:**
- `GGML_BACKEND_DEVICE_TYPE_CPU`
- `GGML_BACKEND_DEVICE_TYPE_GPU`
- `GGML_BACKEND_DEVICE_TYPE_GPU_UMA` — integrated GPU sharing host memory
- `GGML_BACKEND_DEVICE_TYPE_ACCEL` — specialized accelerator (DSP, NPU)

---

## Backend Registry

Global registry of all available backend plugins:

```c
// Per-registry
const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

// Registration
void ggml_backend_register(ggml_backend_reg_t reg);
void ggml_backend_device_register(ggml_backend_dev_t device);

// Global registry queries
size_t             ggml_backend_reg_count(void);
ggml_backend_reg_t ggml_backend_reg_get(size_t index);
ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);

size_t             ggml_backend_dev_count(void);
ggml_backend_dev_t ggml_backend_dev_get(size_t index);
ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

// Initialize backends by name, type, or auto-select best
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
ggml_backend_t ggml_backend_init_best(void);  // picks best available device

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path);
void               ggml_backend_unload(ggml_backend_reg_t reg);
void               ggml_backend_load_all(void);
void               ggml_backend_load_all_from_path(const char * dir_path);
```

**Example — enumerate all devices:**
```c
ggml_backend_load_all();
size_t n_devs = ggml_backend_dev_count();
for (size_t i = 0; i < n_devs; i++) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    printf("Device %zu: %s (%s)\n", i,
           ggml_backend_dev_name(dev),
           ggml_backend_dev_description(dev));
}
ggml_backend_t backend = ggml_backend_init_best();
```

---

## Scheduler

Splits a computation graph across multiple backends automatically:

```c
// Create scheduler
// backends: array of backends (highest priority first)
// bufts: matching buffer types for each backend (or NULL to use defaults)
// n_backends: number of backends
// graph_size: expected max graph nodes
// parallel: allow parallel execution
// op_offload: auto-offload ops to best device
ggml_backend_sched_t ggml_backend_sched_new(
    ggml_backend_t * backends,
    ggml_backend_buffer_type_t * bufts,
    int n_backends,
    size_t graph_size,
    bool parallel,
    bool op_offload);

void ggml_backend_sched_free(ggml_backend_sched_t sched);

// Reserve memory by running a measurement graph
bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);

// Query
int                    ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
ggml_backend_t         ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
int                    ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
int                    ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
size_t                 ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

// Force specific tensor to specific backend
void           ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

// Execution
void             ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
bool             ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
void             ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
void             ggml_backend_sched_reset(ggml_backend_sched_t sched);

// Per-node evaluation callback (for profiling or inspection)
void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched,
                                          ggml_backend_sched_eval_callback callback,
                                          void * user_data);
```

---

## Memory Allocation

### Tensor Allocator (single buffer)

```c
struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);
```

### Graph Allocator (multi-buffer, lifetime-aware)

```c
// Single buffer type
ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);

// Multiple buffer types (for split graphs)
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);

void ggml_gallocr_free(ggml_gallocr_t galloc);

// Reserve memory for a graph (call once with representative graph)
bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                            const int * node_buffer_ids, const int * leaf_buffer_ids);
void ggml_gallocr_reserve_n_size(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                                 const int * node_buffer_ids, const int * leaf_buffer_ids,
                                 size_t * sizes);

// Allocate tensors for a specific graph instance
bool   ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
```

### Allocate All Context Tensors

```c
// Allocate all tensors in a ggml_context into a backend buffer
size_t                  ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);
```

### View and Address Allocation

```c
enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);
```

---

## CPU Backend

```c
// Initialize CPU backend
ggml_backend_t ggml_backend_cpu_init(void);
bool           ggml_backend_is_cpu(ggml_backend_t backend);

// Configuration
void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu,
                                         ggml_abort_callback abort_callback,
                                         void * abort_callback_data);
void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

// CPU buffer types
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

ggml_backend_reg_t ggml_backend_cpu_reg(void);

// Direct CPU graph compute (no backend abstraction)
struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads, struct ggml_threadpool * threadpool);
enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
```

### Threadpool

```c
struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads);
void ggml_threadpool_params_init(struct ggml_threadpool_params * p, int n_threads);
bool ggml_threadpool_params_match(const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1);

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * params);
void                     ggml_threadpool_free(struct ggml_threadpool * threadpool);
int                      ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool);
void                     ggml_threadpool_pause(struct ggml_threadpool * threadpool);
void                     ggml_threadpool_resume(struct ggml_threadpool * threadpool);
```

### CPU Feature Detection

```c
void ggml_numa_init(enum ggml_numa_strategy numa);
bool ggml_is_numa(void);
void ggml_cpu_init(void);

// Returns 1 if supported, 0 otherwise
int ggml_cpu_has_sse3(void);     int ggml_cpu_has_ssse3(void);
int ggml_cpu_has_avx(void);      int ggml_cpu_has_avx_vnni(void);
int ggml_cpu_has_avx2(void);     int ggml_cpu_has_bmi2(void);
int ggml_cpu_has_f16c(void);     int ggml_cpu_has_fma(void);
int ggml_cpu_has_avx512(void);   int ggml_cpu_has_avx512_vbmi(void);
int ggml_cpu_has_avx512_vnni(void); int ggml_cpu_has_avx512_bf16(void);
int ggml_cpu_has_amx_int8(void);
int ggml_cpu_has_neon(void);     int ggml_cpu_has_arm_fma(void);
int ggml_cpu_has_fp16_va(void);  int ggml_cpu_has_dotprod(void);
int ggml_cpu_has_matmul_int8(void);
int ggml_cpu_has_sve(void);      int ggml_cpu_get_sve_cnt(void);
int ggml_cpu_has_sme(void);
int ggml_cpu_has_riscv_v(void);  int ggml_cpu_get_rvv_vlen(void);
int ggml_cpu_has_vsx(void);      int ggml_cpu_has_vxe(void);
int ggml_cpu_has_wasm_simd(void);
int ggml_cpu_has_llamafile(void);
const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
```

---

## Utilities & Float Conversions

```c
// Version info
const char * ggml_version(void);
const char * ggml_commit(void);
const char * ggml_status_to_string(enum ggml_status status);

// Timing
void    ggml_time_init(void);
int64_t ggml_time_ms(void);
int64_t ggml_time_us(void);
int64_t ggml_cycles(void);
int64_t ggml_cycles_per_ms(void);

// Logging
void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
void ggml_log_set(ggml_log_callback log_callback, void * user_data);

// Abort callback
ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback);

// GUID
bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);

// Debug
void ggml_print_object(const struct ggml_object * obj);
void ggml_print_objects(const struct ggml_context * ctx);
FILE * ggml_fopen(const char * fname, const char * mode);

// F16 / BF16 conversions
float        ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t  ggml_fp32_to_fp16(float x);
void         ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n);
void         ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n);
ggml_bf16_t  ggml_fp32_to_bf16(float x);
float        ggml_bf16_to_fp32(ggml_bf16_t x);
void         ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n);
void         ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n);
void         ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n);

// CPU type conversion utilities
void ggml_cpu_fp32_to_fp32(const float *, float *, int64_t);
void ggml_cpu_fp32_to_i32(const float *, int32_t *, int64_t);
void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);
```
