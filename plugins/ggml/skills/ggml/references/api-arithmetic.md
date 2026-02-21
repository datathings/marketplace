# Arithmetic & Matrix Operations

## Table of Contents
1. [Binary Arithmetic](#binary-arithmetic)
2. [Reductions & Aggregations](#reductions--aggregations)
3. [Element-wise Math](#element-wise-math)
4. [Matrix Operations](#matrix-operations)
5. [Accumulate](#accumulate)

---

## Binary Arithmetic

Most binary ops have an **inplace** variant that overwrites `a`. Output tensor has the same shape as the broadcast result.

```c
struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_add_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

struct ggml_tensor * ggml_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_sub_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_mul_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

struct ggml_tensor * ggml_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_div_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```

**Scalar-like variants:**
```c
// add a scalar tensor b (broadcast) to a
struct ggml_tensor * ggml_add1(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_add1_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// add with type cast
struct ggml_tensor * ggml_add_cast(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_type type);

// scatter-add: add rows of b into a at positions given by ids
struct ggml_tensor * ggml_add_id(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * ids);
```

---

## Reductions & Aggregations

```c
// Sum all elements → scalar
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a);

// Sum along rows (last dim) → [1, ne1, ne2, ne3]
struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx, struct ggml_tensor * a);

// Cumulative sum along ne0
struct ggml_tensor * ggml_cumsum(struct ggml_context * ctx, struct ggml_tensor * a);

// Mean of all elements → scalar
struct ggml_tensor * ggml_mean(struct ggml_context * ctx, struct ggml_tensor * a);

// Index of maximum element per row
struct ggml_tensor * ggml_argmax(struct ggml_context * ctx, struct ggml_tensor * a);

// Count equal elements between a and b → scalar I64
struct ggml_tensor * ggml_count_equal(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Top-k values and indices
struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);

// Sort indices
struct ggml_tensor * ggml_argsort(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order);
struct ggml_tensor * ggml_argsort_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);

// ggml_sort_order values: GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC
```

---

## Element-wise Math

```c
struct ggml_tensor * ggml_sqr(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sqr_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sqrt_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

struct ggml_tensor * ggml_log(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_log_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_dup(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_dup_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

struct ggml_tensor * ggml_clamp(struct ggml_context * ctx, struct ggml_tensor * a, float min, float max);

// Range tensor: arange([start, stop), step) → 1D F32
struct ggml_tensor * ggml_arange(struct ggml_context * ctx, float start, float stop, float step);

// Fill tensor with constant value
struct ggml_tensor * ggml_fill(struct ggml_context * ctx, struct ggml_tensor * a, float c);
struct ggml_tensor * ggml_fill_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float c);
```

---

## Matrix Operations

```c
// General matrix multiply: C = A^T * B
// a: [k, m, ...], b: [k, n, ...] → result: [m, n, ...]
struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Set computation precision (GGML_PREC_DEFAULT or GGML_PREC_F32)
void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec);

// Expert-routing: batch matmul with expert IDs
// as: [k, m, n_experts], b: [k, n, batch], ids: [batch] → [m, n, batch]
struct ggml_tensor * ggml_mul_mat_id(struct ggml_context * ctx,
                                     struct ggml_tensor * as,
                                     struct ggml_tensor * b,
                                     struct ggml_tensor * ids);

// Outer product: a: [m] ⊗ b: [n] → [m, n]
struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Element-wise scale by scalar s
struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s);
struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s);

// Element-wise: x * s + b
struct ggml_tensor * ggml_scale_bias(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b);
struct ggml_tensor * ggml_scale_bias_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b);

// Diagonal matrix from a vector, or extract diagonal
struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a);

// Causal (upper-triangular) masking with -inf
struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
struct ggml_tensor * ggml_diag_mask_inf_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
struct ggml_tensor * ggml_diag_mask_zero_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);

// Triangular matrix: type = GGML_TRI_UPPER, GGML_TRI_LOWER, etc.
struct ggml_tensor * ggml_tri(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_tri_type type);

// Triangular solve: A x = B (left) or x A = B (right)
struct ggml_tensor * ggml_solve_tri(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                    bool left, bool lower, bool uni);
```

**Example — matrix multiplication:**
```c
// weights [K, M], input [K, N] → output [M, N]
struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
struct ggml_tensor * y = ggml_mul_mat(ctx, W, x);
```

---

## Accumulate

Write a sub-tensor into a specific offset of a larger tensor:

```c
// a += b at byte offset within a (nb1/nb2/nb3 = strides, offset = byte start)
struct ggml_tensor * ggml_acc(struct ggml_context * ctx,
                              struct ggml_tensor * a, struct ggml_tensor * b,
                              size_t nb1, size_t nb2, size_t nb3, size_t offset);
struct ggml_tensor * ggml_acc_inplace(struct ggml_context * ctx,
                                      struct ggml_tensor * a, struct ggml_tensor * b,
                                      size_t nb1, size_t nb2, size_t nb3, size_t offset);
```

---

## Loss Functions

```c
// Cross-entropy loss: logits a vs labels b
struct ggml_tensor * ggml_cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_cross_entropy_loss_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);

// AdamW step
struct ggml_tensor * ggml_opt_step_adamw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * grad, struct ggml_tensor * m, struct ggml_tensor * v, struct ggml_tensor * adamw_params);

// SGD step
struct ggml_tensor * ggml_opt_step_sgd(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * grad, struct ggml_tensor * sgd_params);
```

---

## Quantization

```c
// Initialize / teardown global quantization tables for `type`
void ggml_quantize_init(enum ggml_type type);
void ggml_quantize_free(void);

// Whether this type needs an importance matrix for quantization
bool ggml_quantize_requires_imatrix(enum ggml_type type);

// Quantize a chunk of floats
// src: float input, dst: quantized output
// start: row index to start from, nrows: rows to quantize
// n_per_row: elements per row, imatrix: importance matrix (or NULL)
size_t ggml_quantize_chunk(enum ggml_type type,
                           const float * src, void * dst,
                           int64_t start, int64_t nrows, int64_t n_per_row,
                           const float * imatrix);
```
