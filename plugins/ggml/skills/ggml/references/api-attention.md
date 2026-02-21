# Attention Mechanisms

## Table of Contents
1. [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
2. [Softmax](#softmax)
3. [Flash Attention](#flash-attention)
4. [Convolution & Pooling](#convolution--pooling)
5. [Padding & Interpolation](#padding--interpolation)

---

## Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating query/key embeddings.

```c
// Basic RoPE
// a: input tensor [head_dim, n_heads, n_tokens]
// b: positions tensor [n_tokens] (I32)
// n_dims: number of rotated dimensions
// mode: GGML_ROPE_TYPE_* constant
struct ggml_tensor * ggml_rope(struct ggml_context * ctx,
                               struct ggml_tensor * a, struct ggml_tensor * b,
                               int n_dims, int mode);
struct ggml_tensor * ggml_rope_inplace(struct ggml_context * ctx,
                                       struct ggml_tensor * a, struct ggml_tensor * b,
                                       int n_dims, int mode);

// Extended RoPE with frequency scaling (used for long-context models)
// c: freq_factors tensor (optional, NULL for default)
struct ggml_tensor * ggml_rope_ext(struct ggml_context * ctx,
                                   struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                   int n_dims, int mode, int n_ctx_orig,
                                   float freq_base, float freq_scale, float ext_factor,
                                   float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_ext_inplace(struct ggml_context * ctx,
                                           struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                           int n_dims, int mode, int n_ctx_orig,
                                           float freq_base, float freq_scale, float ext_factor,
                                           float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_ext_back(struct ggml_context * ctx,
                                        struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                        int n_dims, int mode, int n_ctx_orig,
                                        float freq_base, float freq_scale, float ext_factor,
                                        float attn_factor, float beta_fast, float beta_slow);

// Multi-section RoPE (e.g., MRoPE for Qwen2-VL)
// sections: int[4] array of dimension counts per section
struct ggml_tensor * ggml_rope_multi(struct ggml_context * ctx,
                                     struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                     int n_dims, int sections[4], int mode, int n_ctx_orig,
                                     float freq_base, float freq_scale, float ext_factor,
                                     float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_multi_inplace(struct ggml_context * ctx,
                                             struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                             int n_dims, int sections[4], int mode, int n_ctx_orig,
                                             float freq_base, float freq_scale, float ext_factor,
                                             float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_multi_back(struct ggml_context * ctx,
                                          struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                          int n_dims, int sections[4], int mode, int n_ctx_orig,
                                          float freq_base, float freq_scale, float ext_factor,
                                          float attn_factor, float beta_fast, float beta_slow);

// YaRN correction dimension computation
void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                              float beta_fast, float beta_slow, float dims[2]);
```

**RoPE mode constants:**
| Constant | Value | Description |
|----------|-------|-------------|
| `GGML_ROPE_TYPE_NORMAL` | 0 | Standard GPT-NeoX-style RoPE |
| `GGML_ROPE_TYPE_NEOX` | 2 | GPT-NeoX interleaved |
| `GGML_ROPE_TYPE_MROPE` | 8 | Multi-section (Qwen2-VL) |
| `GGML_ROPE_TYPE_VISION` | 24 | Vision encoder RoPE |

**Example — LLaMA-style RoPE:**
```c
// positions = [0, 1, 2, ..., n_tokens-1]
struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
// fill pos with 0..n_tokens-1 ...

struct ggml_tensor * q_roped = ggml_rope_ext(ctx, q, pos, NULL,
    n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
    freq_base, freq_scale,
    0.0f /*ext_factor*/, 1.0f /*attn_factor*/,
    0.0f /*beta_fast*/, 0.0f /*beta_slow*/);
```

---

## Softmax

```c
// Standard softmax (over ne0)
struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_soft_max_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// Scaled softmax with optional bias mask (for causal attention)
// scale: multiply logits before softmax
// max_bias: ALiBi-style position bias; set 0.0f to disable
struct ggml_tensor * ggml_soft_max_ext(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       struct ggml_tensor * mask,  // optional: [n_kv, n_q] or NULL
                                       float scale, float max_bias);
struct ggml_tensor * ggml_soft_max_ext_inplace(struct ggml_context * ctx,
                                               struct ggml_tensor * a,
                                               struct ggml_tensor * mask,
                                               float scale, float max_bias);

// Add "sink" tokens (sink attention pattern)
void ggml_soft_max_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);

// Backward pass
struct ggml_tensor * ggml_soft_max_ext_back(struct ggml_context * ctx,
                                            struct ggml_tensor * a, struct ggml_tensor * b,
                                            float scale, float max_bias);
struct ggml_tensor * ggml_soft_max_ext_back_inplace(struct ggml_context * ctx,
                                                    struct ggml_tensor * a, struct ggml_tensor * b,
                                                    float scale, float max_bias);
```

---

## Flash Attention

Fused attention kernel that avoids materialising the full attention matrix:

```c
// q: [head_dim, n_heads, n_q, batch]
// k: [head_dim, n_kv_heads, n_kv, batch]
// v: [head_dim_v, n_kv_heads, n_kv, batch]
// mask: [n_kv, n_q] (optional, NULL for unmasked)
// scale: 1/sqrt(head_dim)
// max_bias: ALiBi slope; 0.0f to disable
// logit_softcap: Gemini-style logit cap; 0.0f to disable
struct ggml_tensor * ggml_flash_attn_ext(struct ggml_context * ctx,
                                         struct ggml_tensor * q,
                                         struct ggml_tensor * k,
                                         struct ggml_tensor * v,
                                         struct ggml_tensor * mask,
                                         float scale,
                                         float max_bias,
                                         float logit_softcap);

// Set/get precision for Flash Attention output accumulation
void           ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec);
enum ggml_prec ggml_flash_attn_ext_get_prec(const struct ggml_tensor * a);

// Add sink tokens to flash attention
void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);

// Backward pass (non-fused)
struct ggml_tensor * ggml_flash_attn_back(struct ggml_context * ctx,
                                          struct ggml_tensor * q,
                                          struct ggml_tensor * k,
                                          struct ggml_tensor * v,
                                          struct ggml_tensor * d,
                                          bool masked);
```

**Example — standard causal self-attention:**
```c
float scale = 1.0f / sqrtf((float)head_dim);

// Build causal mask (upper triangular -inf)
struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_q);
ggml_set_f32(mask, -INFINITY);
// fill lower triangle with 0 ...

struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx,
    q, k, v, mask,
    scale,    // scale
    0.0f,     // max_bias (ALiBi off)
    0.0f);    // logit_softcap off
```

---

## Convolution & Pooling

### 1D Convolution

```c
// Standard 1D conv
// a: filter [K, IC, OC], b: input [L, IC, N]
struct ggml_tensor * ggml_conv_1d(struct ggml_context * ctx,
                                  struct ggml_tensor * a, struct ggml_tensor * b,
                                  int s0, int p0, int d0);  // stride, padding, dilation

// Same-padding shorthand (half-padding)
struct ggml_tensor * ggml_conv_1d_ph(struct ggml_context * ctx,
                                     struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s, int d);

// Depthwise 1D conv
struct ggml_tensor * ggml_conv_1d_dw(struct ggml_context * ctx,
                                     struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int p0, int d0);
struct ggml_tensor * ggml_conv_1d_dw_ph(struct ggml_context * ctx,
                                        struct ggml_tensor * a, struct ggml_tensor * b,
                                        int s0, int d0);

// Transposed 1D conv
struct ggml_tensor * ggml_conv_transpose_1d(struct ggml_context * ctx,
                                            struct ggml_tensor * a, struct ggml_tensor * b,
                                            int s0, int p0, int d0);
```

### 2D Convolution

```c
// General 2D conv
struct ggml_tensor * ggml_conv_2d(struct ggml_context * ctx,
                                  struct ggml_tensor * a, struct ggml_tensor * b,
                                  int s0, int s1, int p0, int p1, int d0, int d1);

// Stride=kernel, padding=0 shorthand
struct ggml_tensor * ggml_conv_2d_sk_p0(struct ggml_context * ctx,
                                        struct ggml_tensor * a, struct ggml_tensor * b);

// Stride=1, half-padding shorthand
struct ggml_tensor * ggml_conv_2d_s1_ph(struct ggml_context * ctx,
                                        struct ggml_tensor * a, struct ggml_tensor * b);

// Depthwise 2D conv
struct ggml_tensor * ggml_conv_2d_dw(struct ggml_context * ctx,
                                     struct ggml_tensor * a, struct ggml_tensor * b,
                                     int s0, int s1, int p0, int p1, int d0, int d1);
struct ggml_tensor * ggml_conv_2d_dw_direct(struct ggml_context * ctx,
                                            struct ggml_tensor * a, struct ggml_tensor * b,
                                            int stride0, int stride1, int pad0, int pad1,
                                            int dilation0, int dilation1);

// Transposed 2D conv (with padding=0)
struct ggml_tensor * ggml_conv_transpose_2d_p0(struct ggml_context * ctx,
                                               struct ggml_tensor * a, struct ggml_tensor * b,
                                               int stride);

// Direct (non-im2col) 2D conv
struct ggml_tensor * ggml_conv_2d_direct(struct ggml_context * ctx,
                                         struct ggml_tensor * a, struct ggml_tensor * b,
                                         int s0, int s1, int p0, int p1, int d0, int d1);
```

### 3D Convolution

```c
struct ggml_tensor * ggml_conv_3d(struct ggml_context * ctx,
                                  struct ggml_tensor * a, struct ggml_tensor * b,
                                  int64_t IC,
                                  int s0, int s1, int s2,
                                  int p0, int p1, int p2,
                                  int d0, int d1, int d2);
struct ggml_tensor * ggml_conv_3d_direct(struct ggml_context * ctx,
                                         struct ggml_tensor * a, struct ggml_tensor * b,
                                         int s0, int s1, int s2,
                                         int p0, int p1, int p2,
                                         int d0, int d1, int d2,
                                         int n_channels, int n_batch, int n_channels_out);
```

### im2col

```c
// Rearrange input patches into columns for GEMM-based conv
struct ggml_tensor * ggml_im2col(struct ggml_context * ctx,
                                 struct ggml_tensor * a, struct ggml_tensor * b,
                                 int s0, int s1, int p0, int p1, int d0, int d1,
                                 bool is_2D, enum ggml_type dst_type);
struct ggml_tensor * ggml_im2col_back(struct ggml_context * ctx,
                                      struct ggml_tensor * a, struct ggml_tensor * b,
                                      int64_t * ne,
                                      int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D);
struct ggml_tensor * ggml_im2col_3d(struct ggml_context * ctx,
                                    struct ggml_tensor * a, struct ggml_tensor * b,
                                    int64_t IC, int s0, int s1, int s2,
                                    int p0, int p1, int p2, int d0, int d1, int d2,
                                    enum ggml_type dst_type);
```

### Pooling

```c
// op: GGML_OP_POOL_MAX or GGML_OP_POOL_AVG
struct ggml_tensor * ggml_pool_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  enum ggml_op_pool op, int k0, int s0, int p0);
struct ggml_tensor * ggml_pool_2d(struct ggml_context * ctx, struct ggml_tensor * a,
                                  enum ggml_op_pool op, int k0, int k1, int s0, int s1,
                                  float p0, float p1);
struct ggml_tensor * ggml_pool_2d_back(struct ggml_context * ctx,
                                       struct ggml_tensor * a, struct ggml_tensor * af,
                                       enum ggml_op_pool op, int k0, int k1, int s0, int s1,
                                       float p0, float p1);
```

---

## Padding & Interpolation

```c
// Zero-pad along all 4 dimensions (p0..p3 elements on each side)
struct ggml_tensor * ggml_pad(struct ggml_context * ctx, struct ggml_tensor * a,
                              int p0, int p1, int p2, int p3);

// Circular (wrap-around) padding
struct ggml_tensor * ggml_pad_circular(struct ggml_context * ctx, struct ggml_tensor * a,
                                       int p0, int p1, int p2, int p3);

// Asymmetric padding (left+right per dim)
struct ggml_tensor * ggml_pad_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int lp0, int rp0, int lp1, int rp1,
                                  int lp2, int rp2, int lp3, int rp3);
struct ggml_tensor * ggml_pad_ext_circular(struct ggml_context * ctx, struct ggml_tensor * a,
                                           int lp0, int rp0, int lp1, int rp1,
                                           int lp2, int rp2, int lp3, int rp3);

// Reflect padding in dimension 0
struct ggml_tensor * ggml_pad_reflect_1d(struct ggml_context * ctx, struct ggml_tensor * a,
                                         int p0, int p1);

// Nearest-neighbor upscale by integer factor
struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int scale_factor, enum ggml_scale_mode mode);

// General interpolation to target shape
// mode: GGML_SCALE_MODE_NEAREST, GGML_SCALE_MODE_BILINEAR, GGML_SCALE_MODE_BICUBIC
struct ggml_tensor * ggml_interpolate(struct ggml_context * ctx, struct ggml_tensor * a,
                                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                      uint32_t mode);
```
