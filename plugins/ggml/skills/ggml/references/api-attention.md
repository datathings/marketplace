# Attention Mechanisms

## Table of Contents
1. [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
2. [Softmax](#softmax)
3. [Diagonal Masking & Sorting](#diagonal-masking--sorting)
4. [Flash Attention](#flash-attention)
5. [Convolution & Pooling](#convolution--pooling)
6. [Padding & Interpolation](#padding--interpolation)
7. [State-Space, RWKV & Linear Attention](#state-space-rwkv--linear-attention)
8. [Windowing & Relative Position](#windowing--relative-position)

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

// Multi-section RoPE (VISION, MROPE or IMROPE; cannot combine with NORMAL/NEOX)
// sections: int[GGML_MROPE_SECTIONS] (== int[4]) array of cos/sin-pair counts per section
//           (sum of sections expected to be n_dims/2; trailing sections may be 0)
struct ggml_tensor * ggml_rope_multi(struct ggml_context * ctx,
                                     struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                     int n_dims, int sections[GGML_MROPE_SECTIONS], int mode, int n_ctx_orig,
                                     float freq_base, float freq_scale, float ext_factor,
                                     float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_multi_inplace(struct ggml_context * ctx,
                                             struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                             int n_dims, int sections[GGML_MROPE_SECTIONS], int mode, int n_ctx_orig,
                                             float freq_base, float freq_scale, float ext_factor,
                                             float attn_factor, float beta_fast, float beta_slow);
struct ggml_tensor * ggml_rope_multi_back(struct ggml_context * ctx,
                                          struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c,
                                          int n_dims, int sections[4], int mode, int n_ctx_orig,
                                          float freq_base, float freq_scale, float ext_factor,
                                          float attn_factor, float beta_fast, float beta_slow);

// DEPRECATED — use ggml_rope_ext / ggml_rope_ext_inplace instead
struct ggml_tensor * ggml_rope_custom(...);          // GGML_DEPRECATED
struct ggml_tensor * ggml_rope_custom_inplace(...);  // GGML_DEPRECATED

// YaRN correction dimension computation
void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                              float beta_fast, float beta_slow, float dims[2]);
```

**RoPE mode constants** (`mode` argument):
| Constant | Value | Description |
|----------|-------|-------------|
| `GGML_ROPE_TYPE_NORMAL` | 0 | Standard (paired) RoPE |
| `GGML_ROPE_TYPE_NEOX` | 2 | GPT-NeoX block ordering (`mode & GGML_ROPE_TYPE_NEOX`) |
| `GGML_ROPE_TYPE_MROPE` | 8 | Multi-section M-RoPE (Qwen2-VL); use `ggml_rope_multi` |
| `GGML_ROPE_TYPE_IMROPE` | 40 | Interleaved M-RoPE; use `ggml_rope_multi` |
| `GGML_ROPE_TYPE_VISION` | 24 | Vision encoder RoPE; use `ggml_rope_multi` |

`GGML_MROPE_SECTIONS` is `4`. For VISION, `n_dims` must be `head_size/2`. NEOX
ordering is forced (and cannot be disabled) for MROPE/IMROPE/VISION.

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

The `mask` for `ggml_soft_max_ext` is `[ne0, ne11, ne12, ne13]` (with `ne11 >= ne01`),
F16 or F32, optional. It broadcasts over the head/batch dims (`ne02 % ne12 == 0`,
`ne03 % ne13 == 0`). The fused op computes `soft_max(a*scale + mask*(ALiBi slope))`.

---

## Diagonal Masking & Sorting

```c
// set elements above the diagonal to -INF (causal masking)
struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx,
                                        struct ggml_tensor * a, int n_past);
struct ggml_tensor * ggml_diag_mask_inf_inplace(struct ggml_context * ctx,
                                                struct ggml_tensor * a, int n_past);

// set elements above the diagonal to 0
struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx,
                                         struct ggml_tensor * a, int n_past);
struct ggml_tensor * ggml_diag_mask_zero_inplace(struct ggml_context * ctx,
                                                 struct ggml_tensor * a, int n_past);

// argsort rows (returns I32 indices)
// order: GGML_SORT_ORDER_ASC or GGML_SORT_ORDER_DESC
struct ggml_tensor * ggml_argsort(struct ggml_context * ctx,
                                  struct ggml_tensor * a, enum ggml_sort_order order);

// top-k indices per row (no particular order)
struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);

// top-k via argsort + view (descending order preserved)
struct ggml_tensor * ggml_argsort_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);
```

**Sort order enum:** `GGML_SORT_ORDER_ASC`, `GGML_SORT_ORDER_DESC`.

---

## Flash Attention

Fused attention kernel that avoids materialising the full attention matrix:

```c
// q:    [n_embd_k, n_batch,   n_head,    ne3]
// k:    [n_embd_k, n_kv,      n_head_kv, ne3]
// v:    [n_embd_v, n_kv,      n_head_kv, ne3]   !! not transposed !!
// mask: [n_kv,     n_batch,   ne32,      ne33]  (optional, NULL for unmasked)
// res:  [n_embd_v, n_head,    n_batch,   ne3]   !! permuted !!
// broadcast: n_head % n_head_kv == 0, n_head % ne32 == 0, ne3 % ne33 == 0
// scale: 1/sqrt(head_dim)
// max_bias: ALiBi slope; 0.0f to disable
// logit_softcap: Gemma-style logit cap; 0.0f to disable
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

// Add sink tokens to flash attention (applied post-construction, like soft_max_add_sinks)
void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);

// Backward pass (non-fused; not yet adapted to flash_attn_ext)
struct ggml_tensor * ggml_flash_attn_back(struct ggml_context * ctx,
                                          struct ggml_tensor * q,
                                          struct ggml_tensor * k,
                                          struct ggml_tensor * v,
                                          struct ggml_tensor * d,
                                          bool masked);
```

**Precision enum (`ggml_prec`):** `GGML_PREC_DEFAULT = 0`, `GGML_PREC_F32 = 10`.
Note: sinks are not a constructor argument; attach them with
`ggml_flash_attn_ext_add_sinks` after building the node.

**Example — standard causal self-attention:**
```c
float scale = 1.0f / sqrtf((float)head_dim);

// Build causal mask (upper triangular -inf); shape [n_kv, n_batch]
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

// col2im_1d: scatter-add GEMM columns back to a 1D signal
// a: [K*OC, T_in] columns from matmul (K = a->ne[0]/oc)
// result: [T_out, OC] with T_out = (T_in - 1)*s0 + K - 2*p0
struct ggml_tensor * ggml_col2im_1d(struct ggml_context * ctx,
                                    struct ggml_tensor * a,
                                    int s0,    // stride
                                    int oc,    // output channels
                                    int p0);   // padding cropped from both sides
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

// Upscale ne0/ne1 by an integer factor
struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a,
                                  int scale_factor, enum ggml_scale_mode mode);

// DEPRECATED — use ggml_interpolate instead
struct ggml_tensor * ggml_upscale_ext(struct ggml_context * ctx, struct ggml_tensor * a,
                                      int ne0, int ne1, int ne2, int ne3,
                                      enum ggml_scale_mode mode);  // GGML_DEPRECATED

// Up/down-sample to a target size; 2D modes apply to the first two dims.
// mode is a ggml_scale_mode optionally OR'd with ggml_scale_flag bits.
struct ggml_tensor * ggml_interpolate(struct ggml_context * ctx, struct ggml_tensor * a,
                                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                      uint32_t mode);

// Sinusoidal timestep embedding (diffusion models): timesteps [N] -> [N, dim]
struct ggml_tensor * ggml_timestep_embedding(struct ggml_context * ctx,
                                             struct ggml_tensor * timesteps,
                                             int dim, int max_period);
```

**Scale mode enum (`ggml_scale_mode`):**
| Constant | Value |
|----------|-------|
| `GGML_SCALE_MODE_NEAREST` | 0 |
| `GGML_SCALE_MODE_BILINEAR` | 1 |
| `GGML_SCALE_MODE_BICUBIC` | 2 |

**Scale flags (`ggml_scale_flag`, OR into `ggml_interpolate` mode):**
`GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8)`, `GGML_SCALE_FLAG_ANTIALIAS = (1 << 9)`.

---

## State-Space, RWKV & Linear Attention

```c
// State-space model 1D convolution (Mamba)
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx,
                                   struct ggml_tensor * sx, struct ggml_tensor * c);

// Selective state-space scan (Mamba/Mamba-2)
// NOTE: gained an `ids` argument vs older releases
struct ggml_tensor * ggml_ssm_scan(struct ggml_context * ctx,
                                   struct ggml_tensor * s,
                                   struct ggml_tensor * x,
                                   struct ggml_tensor * dt,
                                   struct ggml_tensor * A,
                                   struct ggml_tensor * B,
                                   struct ggml_tensor * C,
                                   struct ggml_tensor * ids);

// RWKV-6 WKV operator
struct ggml_tensor * ggml_rwkv_wkv6(struct ggml_context * ctx,
                                    struct ggml_tensor * k, struct ggml_tensor * v,
                                    struct ggml_tensor * r, struct ggml_tensor * tf,
                                    struct ggml_tensor * td, struct ggml_tensor * state);

// RWKV-7 WKV operator
struct ggml_tensor * ggml_rwkv_wkv7(struct ggml_context * ctx,
                                    struct ggml_tensor * r, struct ggml_tensor * w,
                                    struct ggml_tensor * k, struct ggml_tensor * v,
                                    struct ggml_tensor * a, struct ggml_tensor * b,
                                    struct ggml_tensor * state);

// Gated linear attention (GLA)
struct ggml_tensor * ggml_gated_linear_attn(struct ggml_context * ctx,
                                            struct ggml_tensor * k, struct ggml_tensor * v,
                                            struct ggml_tensor * q, struct ggml_tensor * g,
                                            struct ggml_tensor * state, float scale);

// Gated delta-net (Qwen3-Next style); K = number of state snapshots to emit
// q,k: [S_k, H_k, n_tokens, n_seqs]   v: [S_v, H_v, n_tokens, n_seqs]
// g,beta: gate tensors   state: [S_v, S_v, H_v, n_seqs]
// output packs attention scores followed by K state snapshots (most recent first)
struct ggml_tensor * ggml_gated_delta_net(struct ggml_context * ctx,
                                          struct ggml_tensor * q, struct ggml_tensor * k,
                                          struct ggml_tensor * v, struct ggml_tensor * g,
                                          struct ggml_tensor * beta, struct ggml_tensor * state,
                                          int64_t K);
```

---

## Windowing & Relative Position

```c
// Partition into non-overlapping windows (with padding if needed) — used in SAM
struct ggml_tensor * ggml_win_part(struct ggml_context * ctx, struct ggml_tensor * a, int w);

// Reverse of ggml_win_part
struct ggml_tensor * ggml_win_unpart(struct ggml_context * ctx, struct ggml_tensor * a,
                                     int w0, int h0, int w);

// Relative position helpers (used in SAM)
struct ggml_tensor * ggml_get_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a,
                                      int qh, int kh);
struct ggml_tensor * ggml_add_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a,
                                      struct ggml_tensor * pw, struct ggml_tensor * ph);
struct ggml_tensor * ggml_add_rel_pos_inplace(struct ggml_context * ctx, struct ggml_tensor * a,
                                              struct ggml_tensor * pw, struct ggml_tensor * ph);
```
