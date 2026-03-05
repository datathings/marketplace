# Activations, Normalization & Shape

## Table of Contents
1. [Activation Functions](#activation-functions)
2. [GLU Variants](#glu-variants)
3. [Normalization](#normalization)
4. [Shape & Layout Operations](#shape--layout-operations)
5. [Custom Operators](#custom-operators)
6. [Specialized Ops](#specialized-ops)

---

## Activation Functions

All have an `_inplace` variant unless noted. Operate element-wise.

```c
// Sign and magnitude
struct ggml_tensor * ggml_abs(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_abs_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sgn(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sgn_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_neg(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_neg_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// Step and threshold
struct ggml_tensor * ggml_step(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_step_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_relu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_relu_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx, struct ggml_tensor * a, float negative_slope, bool inplace);

// Sigmoid-family
struct ggml_tensor * ggml_sigmoid(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sigmoid_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_tanh(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_tanh_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_elu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_elu_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// GELU variants
struct ggml_tensor * ggml_gelu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_gelu_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_gelu_quick(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_gelu_quick_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_gelu_erf(struct ggml_context * ctx, struct ggml_tensor * a);

// SiLU (Swish)
struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_silu_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Hard activations (mobile-friendly)
struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_softplus(struct ggml_context * ctx, struct ggml_tensor * a);

// Trig
struct ggml_tensor * ggml_sin(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sin_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_cos(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_cos_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// Rounding
struct ggml_tensor * ggml_floor(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_ceil(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_round(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_trunc(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_expm1(struct ggml_context * ctx, struct ggml_tensor * a);

// XIELU (xi+ELU) — parameterized
struct ggml_tensor * ggml_xielu(struct ggml_context * ctx, struct ggml_tensor * a,
                                float alpha_n, float alpha_p, float beta, float eps);
struct ggml_tensor * ggml_xielu_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                        float alpha_n, float alpha_p, float beta, float eps);

// Generic unary dispatch (enum ggml_unary_op)
struct ggml_tensor * ggml_unary(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op);
struct ggml_tensor * ggml_unary_inplace(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op);
```

**ggml_unary_op values (21):**
`GGML_UNARY_OP_ABS, NEG, SGN, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID, EXP, SIN, COS, GELU_ERF, RELU_THRESHOLD, SOFTPLUS, FLOOR, CEIL, ROUND, TRUNC`

---

## GLU Variants

Gated Linear Units — split input in half and apply a gate function:

```c
// Generic dispatch (op selects gate: REGLU, GEGLU, SWIGLU, etc.)
struct ggml_tensor * ggml_glu(struct ggml_context * ctx, struct ggml_tensor * a,
                              enum ggml_glu_op op, bool swapped);

// Concrete variants — split along ne0
struct ggml_tensor * ggml_reglu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_geglu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_swiglu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_reglu_swapped(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_geglu_swapped(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_swiglu_swapped(struct ggml_context * ctx, struct ggml_tensor * a);

// Split variants — a and b are separate gate/value tensors
struct ggml_tensor * ggml_glu_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_glu_op op);
struct ggml_tensor * ggml_reglu_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_geglu_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_swiglu_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// OAI-style SwiGLU with scale and clamp
struct ggml_tensor * ggml_swiglu_oai(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                     float alpha, float limit);
```

---

## Normalization

```c
// Layer Norm: (x - mean) / std * gamma + beta
struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps);

// RMS Norm: x / rms(x), used in LLaMA
struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_rms_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float eps);

// Group Norm: normalize within n_groups
struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps);
struct ggml_tensor * ggml_group_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps);

// L2 Norm: x / |x|_2
struct ggml_tensor * ggml_l2_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_l2_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
```

---

## Shape & Layout Operations

```c
// Reshape (must be same number of elements)
struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_reshape_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_reshape_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
struct ggml_tensor * ggml_reshape_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// Views (zero-copy, may not be contiguous)
struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, size_t offset);
struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
struct ggml_tensor * ggml_view_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
struct ggml_tensor * ggml_view_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

// Permute / transpose
struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3);
struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a);

// Make contiguous (copy if needed)
struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_cont_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_cont_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// Copy / cast
struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_cast(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);

// Repeat / tile
struct ggml_tensor * ggml_repeat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_repeat_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
struct ggml_tensor * ggml_repeat_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Concatenate along dim (0-3)
struct ggml_tensor * ggml_concat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int dim);

// Row gather / scatter
struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_get_rows_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
struct ggml_tensor * ggml_set_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);

// Circular shift along all dimensions
struct ggml_tensor * ggml_roll(struct ggml_context * ctx, struct ggml_tensor * a, int shift0, int shift1, int shift2, int shift3);
```

---

## Custom Operators

```c
// Custom op with 1, 2, or 3 input tensors — fun is called during compute
struct ggml_tensor * ggml_map_custom1(struct ggml_context * ctx, struct ggml_tensor * a,
                                      ggml_custom1_op_t fun, int n_tasks, void * userdata);
struct ggml_tensor * ggml_map_custom1_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                              ggml_custom1_op_t fun, int n_tasks, void * userdata);

struct ggml_tensor * ggml_map_custom2(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                      ggml_custom2_op_t fun, int n_tasks, void * userdata);
struct ggml_tensor * ggml_map_custom2_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                              ggml_custom2_op_t fun, int n_tasks, void * userdata);

struct ggml_tensor * ggml_map_custom3(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                      ggml_custom3_op_t fun, int n_tasks, void * userdata);
struct ggml_tensor * ggml_map_custom3_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                              ggml_custom3_op_t fun, int n_tasks, void * userdata);

// New-style flexible custom op with arbitrary inputs
struct ggml_tensor * ggml_custom_4d(struct ggml_context * ctx,
                                    enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                    struct ggml_tensor ** args, int n_args,
                                    ggml_custom_op_t fun, int n_tasks, void * userdata);
struct ggml_tensor * ggml_custom_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                         struct ggml_tensor ** args, int n_args,
                                         ggml_custom_op_t fun, int n_tasks, void * userdata);
```

**Custom op callback signature:**
```c
// ggml_custom1_op_t
typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
```

---

## Specialized Ops

```c
// Window partitioning (SAM/Swin transformer)
struct ggml_tensor * ggml_win_part(struct ggml_context * ctx, struct ggml_tensor * a, int w);
struct ggml_tensor * ggml_win_unpart(struct ggml_context * ctx, struct ggml_tensor * a, int w0, int h0, int w);

// Relative position embeddings (SAM)
struct ggml_tensor * ggml_get_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a, int qh, int kh);
struct ggml_tensor * ggml_add_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph);
struct ggml_tensor * ggml_add_rel_pos_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph);

// RWKV WKV (state-space) operations
struct ggml_tensor * ggml_rwkv_wkv6(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v,
                                     struct ggml_tensor * r, struct ggml_tensor * tf, struct ggml_tensor * td, struct ggml_tensor * state);
struct ggml_tensor * ggml_rwkv_wkv7(struct ggml_context * ctx, struct ggml_tensor * r, struct ggml_tensor * w,
                                     struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * state);

// SSM (Mamba / Mamba2)
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor * sx, struct ggml_tensor * c);
struct ggml_tensor * ggml_ssm_scan(struct ggml_context * ctx, struct ggml_tensor * s, struct ggml_tensor * x,
                                   struct ggml_tensor * dt, struct ggml_tensor * A, struct ggml_tensor * B,
                                   struct ggml_tensor * C, struct ggml_tensor * ids);

// Gated linear attention
struct ggml_tensor * ggml_gated_linear_attn(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v,
                                            struct ggml_tensor * q, struct ggml_tensor * g, struct ggml_tensor * state, float scale);

// Diffusion timestep embedding
struct ggml_tensor * ggml_timestep_embedding(struct ggml_context * ctx, struct ggml_tensor * timesteps, int dim, int max_period);
```
