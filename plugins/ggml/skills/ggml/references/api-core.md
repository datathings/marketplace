# Core API — Context, Tensors & Graphs

## Table of Contents
1. [Initialization Params](#initialization-params)
2. [Context Management](#context-management)
3. [Tensor Creation](#tensor-creation)
4. [Tensor Properties & Metadata](#tensor-properties--metadata)
5. [Tensor Data Access & Naming](#tensor-data-access--naming)
6. [Tensor Set Operations](#tensor-set-operations)
7. [Computation Graph Management](#computation-graph-management)
8. [Scalar Helpers](#scalar-helpers)
9. [Key Constants & Macros](#key-constants--macros)

---

## Initialization Params

```c
struct ggml_init_params {
    size_t mem_size;    // bytes of pre-allocated memory
    void * mem_buffer;  // pointer to externally allocated buffer (or NULL)
    bool   no_alloc;    // do not allocate tensor data
};
```

---

## Context Management

```c
struct ggml_context * ggml_init(struct ggml_init_params params);
void                  ggml_reset(struct ggml_context * ctx);
void                  ggml_free(struct ggml_context * ctx);
size_t                ggml_used_mem(const struct ggml_context * ctx);
bool                  ggml_get_no_alloc(struct ggml_context * ctx);
void                  ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
void *                ggml_get_mem_buffer(const struct ggml_context * ctx);
size_t                ggml_get_mem_size(const struct ggml_context * ctx);
size_t                ggml_get_max_tensor_size(const struct ggml_context * ctx);
```

**Example:**
```c
size_t buf_size = 128 * 1024 * 1024; // 128 MB
struct ggml_init_params params = {
    .mem_size   = buf_size,
    .mem_buffer = NULL,
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);
// ... use ctx ...
ggml_free(ctx);
```

---

## Tensor Creation

```c
struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx, enum ggml_type type, int n_dims, const int64_t * ne);
struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0);
struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
void *               ggml_new_buffer(struct ggml_context * ctx, size_t nbytes);
struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
struct ggml_tensor * ggml_get_next_tensor(const struct ggml_context * ctx, struct ggml_tensor * tensor);
struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
```

**Note:** Dimensions are stored in `ne[0..3]` with `ne[0]` being the innermost (fastest). For a matrix `[rows x cols]`, use `ne0=cols`, `ne1=rows`.

---

## Tensor Properties & Metadata

```c
// Element and byte counts
int64_t ggml_nelements(const struct ggml_tensor * tensor);
int64_t ggml_nrows(const struct ggml_tensor * tensor);
size_t  ggml_nbytes(const struct ggml_tensor * tensor);
size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor);

// Type info
int64_t     ggml_blck_size(enum ggml_type type);
size_t      ggml_type_size(enum ggml_type type);
size_t      ggml_row_size(enum ggml_type type, int64_t ne);
const char *ggml_type_name(enum ggml_type type);
bool        ggml_is_quantized(enum ggml_type type);
enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
size_t      ggml_element_size(const struct ggml_tensor * tensor);
size_t      ggml_tensor_overhead(void);
const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type);

// Shape tests
bool ggml_is_transposed(const struct ggml_tensor * tensor);
bool ggml_is_permuted(const struct ggml_tensor * tensor);
bool ggml_is_empty(const struct ggml_tensor * tensor);
bool ggml_is_scalar(const struct ggml_tensor * tensor);
bool ggml_is_vector(const struct ggml_tensor * tensor);
bool ggml_is_matrix(const struct ggml_tensor * tensor);
bool ggml_is_3d(const struct ggml_tensor * tensor);
int  ggml_n_dims(const struct ggml_tensor * tensor);

// Contiguity tests
bool ggml_is_contiguous(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_0(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_1(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_2(const struct ggml_tensor * tensor);
bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);

// Comparison / compatibility
bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

// Validation
bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);

// Op names
const char * ggml_op_name(enum ggml_op op);
const char * ggml_op_symbol(enum ggml_op op);
const char * ggml_op_desc(const struct ggml_tensor * t);
const char * ggml_unary_op_name(enum ggml_unary_op op);
const char * ggml_glu_op_name(enum ggml_glu_op op);
enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
enum ggml_glu_op   ggml_get_glu_op(const struct ggml_tensor * tensor);
```

---

## Tensor Data Access & Naming

```c
void *       ggml_get_data(const struct ggml_tensor * tensor);
float *      ggml_get_data_f32(const struct ggml_tensor * tensor);
const char * ggml_get_name(const struct ggml_tensor * tensor);
struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...);

// Flags — mark roles in the graph
void ggml_set_input(struct ggml_tensor * tensor);
void ggml_set_output(struct ggml_tensor * tensor);
void ggml_set_param(struct ggml_tensor * tensor);    // trainable parameter
void ggml_set_loss(struct ggml_tensor * tensor);

// Index helpers
void ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i,
                        int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
```

---

## Tensor Set Operations

Write a slice of tensor `a` with the contents of tensor `b`:

```c
struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset);
struct ggml_tensor * ggml_set_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      size_t nb1, size_t nb2, size_t nb3, size_t offset);
struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
struct ggml_tensor * ggml_set_1d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
struct ggml_tensor * ggml_set_2d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
```

---

## Computation Graph Management

```c
// Build graph nodes
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * cgraph, struct ggml_tensor ** grad_accs);
struct ggml_tensor * ggml_build_forward_select(struct ggml_cgraph * cgraph, struct ggml_tensor ** tensors, int n_tensors, int idx);

// Create / duplicate graphs
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx);
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads);
struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads);
void                 ggml_graph_cpy(struct ggml_cgraph * src, struct ggml_cgraph * dst);
void                 ggml_graph_reset(struct ggml_cgraph * cgraph);
void                 ggml_graph_clear(struct ggml_cgraph * cgraph);

// Graph inspection
int                  ggml_graph_size(struct ggml_cgraph * cgraph);
struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i);
struct ggml_tensor **ggml_graph_nodes(struct ggml_cgraph * cgraph);
int                  ggml_graph_n_nodes(struct ggml_cgraph * cgraph);
void                 ggml_graph_add_node(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
size_t               ggml_graph_overhead(void);
size_t               ggml_graph_overhead_custom(size_t size, bool grads);
struct ggml_tensor * ggml_graph_get_tensor(const struct ggml_cgraph * cgraph, const char * name);
struct ggml_tensor * ggml_graph_get_grad(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);
struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);

// Debug output
void ggml_graph_print(const struct ggml_cgraph * cgraph);
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * cgraph, const char * filename);
```

**Example — build and run a forward pass:**
```c
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, result);
ggml_backend_graph_compute(backend, gf);
```

---

## Scalar Helpers

```c
struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
struct ggml_tensor * ggml_set_i32(struct ggml_tensor * tensor, int32_t value);   // fill all elements
struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value);     // fill all elements

// 1-D element access (linear index)
int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

// N-D element access
int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
```

---

## Key Constants & Macros

| Constant | Value | Description |
|----------|-------|-------------|
| `GGML_FILE_MAGIC` | 0x67676d6c | "ggml" |
| `GGML_MAX_DIMS` | 4 | max tensor dimensions |
| `GGML_MAX_PARAMS` | 2048 | max number of params |
| `GGML_MAX_SRC` | 10 | max source tensors per op |
| `GGML_MAX_N_THREADS` | 512 | max CPU threads |
| `GGML_MAX_OP_PARAMS` | 64 | bytes of op-specific params |
| `GGML_DEFAULT_N_THREADS` | 4 | default thread count |
| `GGML_DEFAULT_GRAPH_SIZE` | 2048 | default graph node capacity |
| `GGML_MAX_NAME` | 64 | max tensor name length |

**ggml_type enum (selected):**
- `GGML_TYPE_F32` — 32-bit float (default for computation)
- `GGML_TYPE_F16` — 16-bit half float
- `GGML_TYPE_BF16` — bfloat16
- `GGML_TYPE_Q4_0` / `GGML_TYPE_Q4_1` — 4-bit quantized
- `GGML_TYPE_Q8_0` — 8-bit quantized
- `GGML_TYPE_I8`, `GGML_TYPE_I16`, `GGML_TYPE_I32`, `GGML_TYPE_I64` — integers

**ggml_status:**
- `GGML_STATUS_SUCCESS` = 0
- `GGML_STATUS_FAILED` = -1
- `GGML_STATUS_ALLOC_FAILED` = -2
- `GGML_STATUS_ABORTED` = 1
