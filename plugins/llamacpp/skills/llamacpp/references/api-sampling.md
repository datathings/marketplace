# llama.cpp C API Reference - Sampling

Part 5 of 6 | [Core](api-core.md) | [Model Info](api-model-info.md) | [Context](api-context.md) | [Inference](api-inference.md) | **Sampling** | [Advanced](api-advanced.md)

This file covers:
- Sampling - All 25+ sampling strategies including XTC, DRY, penalties, top-k/p, temperature, etc.
- Backend Sampling API [EXPERIMENTAL] - GPU-accelerated sampling

For complete API navigation, see [api-core.md](api-core.md).

---

## Sampling

Sampling in llama.cpp uses a chain architecture where multiple samplers can be combined.

### Core Sampler Functions

```c
struct llama_sampler * llama_sampler_init(
    struct llama_sampler_i * iface,
    llama_sampler_context_t ctx);
```
Initialize a custom sampler (for advanced users implementing custom sampling).

```c
const char * llama_sampler_name(const struct llama_sampler * smpl);
```
Get the name of a sampler.

```c
void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
```
Accept a token (updates sampler state).

```c
void llama_sampler_apply(
    struct llama_sampler * smpl,
    llama_token_data_array * cur_p);
```
Apply the sampler to modify the token data array.

```c
void llama_sampler_reset(struct llama_sampler * smpl);
```
Reset the sampler state.

```c
struct llama_sampler * llama_sampler_clone(const struct llama_sampler * smpl);
```
Clone a sampler.

```c
void llama_sampler_free(struct llama_sampler * smpl);
```
Free a sampler. **Important:** Do not free if added to a chain via `llama_sampler_chain_add()`.

### Sampler Chain

```c
struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params);
```
Initialize a sampler chain.

**Usage:**
```c
struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
struct llama_sampler * chain = llama_sampler_chain_init(params);
```

```c
void llama_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * smpl);
```
Add a sampler to the chain. **Important:** The chain takes ownership and will free the sampler.

```c
struct llama_sampler * llama_sampler_chain_get(
    struct llama_sampler * chain,
    int32_t i);
```
Get the i-th sampler in the chain. Returns NULL if:
- the sampler is NULL
- the sampler is not a `llama_sampler_chain`
- the index is out of bounds, unless i == -1
- if i == -1, returns the chain itself (can be used to check if the sampler is a chain)

```c
int llama_sampler_chain_n(const struct llama_sampler * chain);
```
Get the number of samplers in the chain.

```c
struct llama_sampler * llama_sampler_chain_remove(
    struct llama_sampler * chain,
    int32_t i);
```
Remove a sampler from the chain. The chain no longer owns it and will not free it.

### Built-in Samplers

#### Basic Samplers

```c
struct llama_sampler * llama_sampler_init_greedy(void);
```
Greedy sampling (always pick the most likely token).

```c
struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
```
Sample from the probability distribution.

**Parameters:**
- `seed`: Random seed. Use `LLAMA_DEFAULT_SEED` (0xFFFFFFFF) to use a random seed.

#### Top-K Sampling

```c
struct llama_sampler * llama_sampler_init_top_k(int32_t k);
```
Top-K sampling. Setting k <= 0 makes this a noop.

**Reference:** "The Curious Case of Neural Text Degeneration" (https://arxiv.org/abs/1904.09751)

#### Top-P (Nucleus) Sampling

```c
struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
```
Nucleus sampling.

**Parameters:**
- `p`: Cumulative probability threshold
- `min_keep`: Minimum number of tokens to keep

**Reference:** "The Curious Case of Neural Text Degeneration" (https://arxiv.org/abs/1904.09751)

#### Min-P Sampling

```c
struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
```
Minimum P sampling.

**Reference:** https://github.com/ggml-org/llama.cpp/pull/3841

#### Typical Sampling

```c
struct llama_sampler * llama_sampler_init_typical(float p, size_t min_keep);
```
Locally Typical Sampling.

**Reference:** https://arxiv.org/abs/2202.00666

#### Temperature

```c
struct llama_sampler * llama_sampler_init_temp(float t);
```
Temperature sampling. Updates logits: `l_i' = l_i/t`. When t <= 0, max logit is kept, rest set to -inf.

```c
struct llama_sampler * llama_sampler_init_temp_ext(
    float t,
    float delta,
    float exponent);
```
Dynamic temperature (entropy-based).

**Reference:** https://arxiv.org/abs/2309.02772

#### XTC Sampler

```c
struct llama_sampler * llama_sampler_init_xtc(
    float p,
    float t,
    size_t min_keep,
    uint32_t seed);
```
XTC sampler.

**Reference:** https://github.com/oobabooga/text-generation-webui/pull/6335

#### Top-nσ Sampling

```c
struct llama_sampler * llama_sampler_init_top_n_sigma(float n);
```
Top-nσ sampling.

**Reference:** "Top-nσ: Not All Logits Are You Need" (https://arxiv.org/pdf/2411.07641)

#### Mirostat

```c
struct llama_sampler * llama_sampler_init_mirostat(
    int32_t n_vocab,
    uint32_t seed,
    float tau,
    float eta,
    int32_t m);
```
Mirostat 1.0 algorithm.

**Parameters:**
- `n_vocab`: Vocabulary size
- `seed`: Random seed
- `tau`: Target cross-entropy
- `eta`: Learning rate
- `m`: Number of tokens considered (paper uses m=100)

**Reference:** https://arxiv.org/abs/2007.14966

```c
struct llama_sampler * llama_sampler_init_mirostat_v2(
    uint32_t seed,
    float tau,
    float eta);
```
Mirostat 2.0 algorithm.

#### Grammar

```c
struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);
```
Initialize a GBNF grammar sampler. Returns NULL if parsing fails.

**Parameters:**
- `vocab`: Vocabulary
- `grammar_str`: Production rules as a string
- `grammar_root`: Name of the start symbol

```c
struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);
```
Lazy grammar sampler (triggers based on patterns or tokens).

**Reference:** https://github.com/ggml-org/llama.cpp/pull/9639

#### Penalties

```c
struct llama_sampler * llama_sampler_init_penalties(
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present);
```
Apply repetition penalties. **Note:** Avoid using on full vocabulary (slow). Apply top-k or top-p first.

**Parameters:**
- `penalty_last_n`: Last n tokens to penalize (0 = disabled, -1 = context size)
- `penalty_repeat`: Repeat penalty (1.0 = disabled)
- `penalty_freq`: Frequency penalty (0.0 = disabled)
- `penalty_present`: Presence penalty (0.0 = disabled)

#### DRY Sampler

```c
struct llama_sampler * llama_sampler_init_dry(
    const struct llama_vocab * vocab,
    int32_t n_ctx_train,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_penalty_last_n,
    const char ** seq_breakers,
    size_t num_breakers);
```
DRY (Don't Repeat Yourself) sampler.

**Reference:** https://github.com/oobabooga/text-generation-webui/pull/5677

#### Logit Bias

```c
struct llama_sampler * llama_sampler_init_logit_bias(
    int32_t n_vocab,
    int32_t n_logit_bias,
    const llama_logit_bias * logit_bias);
```
Apply logit biases to specific tokens.

#### Infill Sampler

```c
struct llama_sampler * llama_sampler_init_infill(
    const struct llama_vocab * vocab);
```
Fill-in-the-middle infilling sampler. Use after top-k + top-p sampling.

### Sampling Functions

```c
uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);
```
Get the seed used by the sampler (if applicable), otherwise `LLAMA_DEFAULT_SEED`.

```c
llama_token llama_sampler_sample(
    struct llama_sampler * smpl,
    struct llama_context * ctx,
    int32_t idx);
```
Sample and accept a token from the idx-th output of the last evaluation.

**Usage:**
```c
// Setup sampler chain
struct llama_sampler * sampler = llama_sampler_chain_init(
    llama_sampler_chain_default_params());
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

// Decoding loop
while (...) {
    llama_decode(ctx, batch);
    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    // Use token...
}

llama_sampler_free(sampler);
```

---

## Backend Sampling API [EXPERIMENTAL]

Backend sampling allows sampling operations to be performed directly on the GPU as part of the computation graph. This can improve performance by reducing data transfer between device and host memory.

**Note:** Use only if the `llama_context` was created with at least one `llama_sampler_seq_config`.

### Configuration Struct

```c
struct llama_sampler_seq_config {
    llama_seq_id           seq_id;
    struct llama_sampler * sampler;
};
```
Configuration for per-sequence backend sampling. Add to `llama_context_params.samplers` when creating context.

### Attaching Samplers

```c
bool llama_set_sampler(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    struct llama_sampler * smpl);
```
Attach a sampler to the context for a specific sequence.

**Notes:**
- Prefer initializing the context with `llama_context_params.samplers` when possible
- Changing the samplers of a context can cause graph reallocations and degraded performance
- The sampler must be a sampler chain (use `llama_sampler_chain_init`)

### Retrieving Sampled Results

```c
llama_token llama_get_sampled_token_ith(
    struct llama_context * ctx,
    int32_t i);
```
Get the backend sampled token for the i-th token. Returns `LLAMA_TOKEN_NULL` if no token was sampled.

```c
float * llama_get_sampled_probs_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled probabilities for the i-th token. Returns NULL if no probabilities were generated.

```c
float * llama_get_sampled_logits_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled logits for the i-th token. Returns NULL if no logits were sampled.

```c
llama_token * llama_get_sampled_candidates_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled candidates (token ids) for the i-th token. These are needed to map probability/logit indices to vocab token ids. Returns NULL if no candidates were sampled.

### Custom Sampler Interface for Backend

When implementing custom samplers with backend support, the `llama_sampler_i` interface includes:

```c
struct llama_sampler_data {
    struct ggml_tensor * logits;
    struct ggml_tensor * probs;
    struct ggml_tensor * sampled;
    struct ggml_tensor * candidates;
};
```
Data structure for backend sampling operations.

**Interface methods for backend sampling:**

```c
// Return true if the backend supports all ops needed by the sampler
bool (*backend_init)(struct llama_sampler * smpl, ggml_backend_buffer_type_t buft);

// Called after backend_apply()
void (*backend_accept)(
    struct llama_sampler * smpl,
    struct ggml_context  * ctx,
    struct ggml_cgraph   * gf,
    struct ggml_tensor   * selected_token);

// Called after backend_init()
void (*backend_apply)(
    struct llama_sampler      * smpl,
    struct ggml_context       * ctx,
    struct ggml_cgraph        * gf,
    struct llama_sampler_data * data);

// Called before graph execution to set inputs for the current ubatch
void (*backend_set_input)(struct llama_sampler * smpl);
```

**Usage Example:**
```c
// Create sampler chain for backend sampling
struct llama_sampler * chain = llama_sampler_chain_init(
    llama_sampler_chain_default_params());
llama_sampler_chain_add(chain, llama_sampler_init_top_k(50));
llama_sampler_chain_add(chain, llama_sampler_init_dist(42));

// Configure backend sampling in context params
struct llama_sampler_seq_config sampler_configs[] = {
    { .seq_id = 0, .sampler = chain }
};

struct llama_context_params ctx_params = llama_context_default_params();
ctx_params.samplers = sampler_configs;
ctx_params.n_samplers = 1;

// Create context with backend sampling
struct llama_context * ctx = llama_init_from_model(model, ctx_params);

// After decode, retrieve backend-sampled token
llama_decode(ctx, batch);
llama_token token = llama_get_sampled_token_ith(ctx, -1);
if (token != LLAMA_TOKEN_NULL) {
    // Token was sampled on backend
}
```

---

