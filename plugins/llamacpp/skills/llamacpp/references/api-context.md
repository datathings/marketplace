# llama.cpp C API Reference - Context & Memory

Part 3 of 6 | [Core](api-core.md) | [Model Info](api-model-info.md) | **Context** | [Inference](api-inference.md) | [Sampling](api-sampling.md) | [Advanced](api-advanced.md)

This file covers:
- Context Management - Create and manage inference contexts
- Memory (KV Cache) Management - Manipulate KV cache sequences
- State & Session Management - Save/load full and per-sequence states

For complete API navigation, see [api-core.md](api-core.md).

---

## Context Management

### llama_init_from_model
```c
struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);
```
Create a new context from a loaded model. Returns NULL on failure.

**Usage:**
```c
struct llama_context_params params = llama_context_default_params();
params.n_ctx = 4096;
params.n_batch = 512;
struct llama_context * ctx = llama_init_from_model(model, params);
if (!ctx) {
    // Handle error
}
```

**New parameters in b7631:**

- `samplers` (`struct llama_sampler_seq_config *`) - [EXPERIMENTAL] Backend sampler chain configuration. Enables GPU-accelerated sampling as part of the computation graph. The caller must keep the sampler chains alive. Samplers must be sampler chains (use `llama_sampler_chain_init`). Default: NULL.

- `n_samplers` (`size_t`) - Number of sampler configurations in the `samplers` array. Default: 0.

**Notable parameters:**

- `op_offload` (bool) - Offload host tensor operations to device for improved performance. Default: true.

- `swa_full` (bool) - For models with Sliding Window Attention (SWA), allocate full context size instead of just the attention window. Set to true when you need to access tokens outside the SWA window. Check `llama_model_n_swa()` to detect if a model uses SWA. Default: false. Note: setting to false when `n_seq_max > 1` can cause bad performance.

- `kv_unified` (bool) - Use a unified buffer across input sequences when computing attention. Try disabling when `n_seq_max > 1` for improved performance when sequences do not share a large prefix. Default: true.

**Example with Backend Sampling:**
```c
// Create sampler chain for backend sampling
struct llama_sampler * chain = llama_sampler_chain_init(
    llama_sampler_chain_default_params());
llama_sampler_chain_add(chain, llama_sampler_init_top_k(50));
llama_sampler_chain_add(chain, llama_sampler_init_dist(42));

// Configure backend sampling
struct llama_sampler_seq_config sampler_configs[] = {
    { .seq_id = 0, .sampler = chain }
};

struct llama_context_params params = llama_context_default_params();
params.samplers = sampler_configs;
params.n_samplers = 1;

struct llama_context * ctx = llama_init_from_model(model, params);
```

**Example with SWA:**
```c
struct llama_context_params params = llama_context_default_params();
params.n_ctx = 32768;

// Check if model uses Sliding Window Attention
int32_t swa_size = llama_model_n_swa(model);
if (swa_size > 0) {
    printf("Model uses SWA with window: %d\n", swa_size);
    params.swa_full = true;  // Enable full context access
}

struct llama_context * ctx = llama_init_from_model(model, params);
```

### llama_free
```c
void llama_free(struct llama_context * ctx);
```
Free all allocated memory for a context. Always call this when done with a context.

**Usage:**
```c
llama_free(ctx);
```

### llama_params_fit
```c
enum llama_params_fit_status llama_params_fit(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);
```
Fits model and context parameters to available device memory. Returns a status enum (SUCCESS, FAILURE, or ERROR). This function is NOT thread-safe. Only parameters matching defaults are modified, except context size which is always modified when set to 0.

**Return Values:**
- `LLAMA_PARAMS_FIT_STATUS_SUCCESS (0)`: Found allocations that are projected to fit
- `LLAMA_PARAMS_FIT_STATUS_FAILURE (1)`: Could not find allocations that fit
- `LLAMA_PARAMS_FIT_STATUS_ERROR (2)`: Hard error occurred (e.g., model not found)

**Parameters:**
- `path_model`: Path to model file
- `mparams`: Writable model params (will be modified)
- `cparams`: Writable context params (will be modified)
- `tensor_split`: Writable buffer for tensor split (needs at least `llama_max_devices()` elements)
- `tensor_buft_overrides`: Writable buffer for overrides (needs at least `llama_max_tensor_buft_overrides()` elements)
- `margins`: Margins of memory to leave per device in bytes (array with `llama_max_devices()` elements)
- `n_ctx_min`: Minimum context size to set when trying to reduce memory use
- `log_level`: Minimum log level to print during fitting

### llama_get_model
```c
const struct llama_model * llama_get_model(const struct llama_context * ctx);
```
Get the model associated with a context.

### llama_get_memory
```c
llama_memory_t llama_get_memory(const struct llama_context * ctx);
```
Get the memory handle for a context.

### llama_pooling_type
```c
enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);
```
Get the pooling type used by the context.

### llama_n_ctx
```c
uint32_t llama_n_ctx(const struct llama_context * ctx);
```
Get the actual context size. After creating a context, query this to get the actual value (may differ from requested).

### llama_n_ctx_seq
```c
uint32_t llama_n_ctx_seq(const struct llama_context * ctx);
```
Get the context size for sequences.

### llama_n_batch
```c
uint32_t llama_n_batch(const struct llama_context * ctx);
```
Get the logical maximum batch size.

### llama_n_ubatch
```c
uint32_t llama_n_ubatch(const struct llama_context * ctx);
```
Get the physical maximum batch size.

### llama_n_seq_max
```c
uint32_t llama_n_seq_max(const struct llama_context * ctx);
```
Get the maximum number of sequences.

---

## Memory (KV Cache) Management

The memory functions operate on the KV cache and allow for advanced sequence management.

### llama_memory_clear
```c
void llama_memory_clear(llama_memory_t mem, bool data);
```
Clear the memory contents.

**Parameters:**
- `mem`: Memory handle
- `data`: If true, data buffers will also be cleared together with metadata

### llama_memory_seq_rm
```c
bool llama_memory_seq_rm(
    llama_memory_t mem,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1);
```
Remove all tokens that belong to the specified sequence and have positions in [p0, p1). Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.

**Parameters:**
- `mem`: Memory handle
- `seq_id`: Sequence ID (< 0: match any sequence)
- `p0`: Start position (< 0: [0, p1])
- `p1`: End position (< 0: [p0, inf))

### llama_memory_seq_cp
```c
void llama_memory_seq_cp(
    llama_memory_t mem,
    llama_seq_id seq_id_src,
    llama_seq_id seq_id_dst,
    llama_pos p0,
    llama_pos p1);
```
Copy all tokens that belong to the specified sequence to another sequence.

**Parameters:**
- `mem`: Memory handle
- `seq_id_src`: Source sequence ID
- `seq_id_dst`: Destination sequence ID
- `p0`: Start position (< 0: [0, p1])
- `p1`: End position (< 0: [p0, inf))

### llama_memory_seq_keep
```c
void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
```
Remove all tokens that do not belong to the specified sequence.

### llama_memory_seq_add
```c
void llama_memory_seq_add(
    llama_memory_t mem,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    llama_pos delta);
```
Add relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1).

**Parameters:**
- `mem`: Memory handle
- `seq_id`: Sequence ID
- `p0`: Start position (< 0: [0, p1])
- `p1`: End position (< 0: [p0, inf))
- `delta`: Position delta to add

### llama_memory_seq_div
```c
void llama_memory_seq_div(
    llama_memory_t mem,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    int d);
```
Integer division of the positions by factor of `d > 1`.

**Parameters:**
- `mem`: Memory handle
- `seq_id`: Sequence ID
- `p0`: Start position (< 0: [0, p1])
- `p1`: End position (< 0: [p0, inf))
- `d`: Divisor (must be > 1)

### llama_memory_seq_pos_min
```c
llama_pos llama_memory_seq_pos_min(llama_memory_t mem, llama_seq_id seq_id);
```
Get the smallest position present in the memory for the specified sequence. Returns -1 if the sequence is empty. Typically non-zero only for SWA caches.

### llama_memory_seq_pos_max
```c
llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);
```
Get the largest position present in the memory for the specified sequence. Returns -1 if the sequence is empty.

### llama_memory_can_shift
```c
bool llama_memory_can_shift(llama_memory_t mem);
```
Check if the memory supports shifting.

---

## State & Session Management

### llama_state_get_size
```c
size_t llama_state_get_size(struct llama_context * ctx);
```
Get the actual size in bytes of the state (logits, embedding, and memory). Only use when saving the state.

### llama_state_get_data
```c
size_t llama_state_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size);
```
Copy the state to the specified destination address. Returns the number of bytes copied.

**Parameters:**
- `ctx`: Context
- `dst`: Destination buffer (must have enough memory allocated)
- `size`: Size of destination buffer

### llama_state_set_data
```c
size_t llama_state_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size);
```
Set the state from the specified address. Returns the number of bytes read.

### llama_state_load_file
```c
bool llama_state_load_file(
    struct llama_context * ctx,
    const char * path_session,
    llama_token * tokens_out,
    size_t n_token_capacity,
    size_t * n_token_count_out);
```
Load session from file.

### llama_state_save_file
```c
bool llama_state_save_file(
    struct llama_context * ctx,
    const char * path_session,
    const llama_token * tokens,
    size_t n_token_count);
```
Save session to file.

### llama_state_seq_get_size
```c
size_t llama_state_seq_get_size(
    struct llama_context * ctx,
    llama_seq_id seq_id);
```
Get the exact size needed to copy the state of a single sequence.

### llama_state_seq_get_data
```c
size_t llama_state_seq_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    llama_seq_id seq_id);
```
Copy the state of a single sequence into the specified buffer.

### llama_state_seq_set_data
```c
size_t llama_state_seq_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    llama_seq_id dest_seq_id);
```
Copy sequence data into the specified sequence. Returns positive on success, zero on failure.

### llama_state_seq_save_file
```c
size_t llama_state_seq_save_file(
    struct llama_context * ctx,
    const char * filepath,
    llama_seq_id seq_id,
    const llama_token * tokens,
    size_t n_token_count);
```
Save sequence state to file.

### llama_state_seq_load_file
```c
size_t llama_state_seq_load_file(
    struct llama_context * ctx,
    const char * filepath,
    llama_seq_id dest_seq_id,
    llama_token * tokens_out,
    size_t n_token_capacity,
    size_t * n_token_count_out);
```
Load sequence state from file.

### llama_state_seq_get_size_ext
```c
size_t llama_state_seq_get_size_ext(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_state_seq_flags flags);
```
Get size of sequence state with flags (e.g., `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` for SWA/recurrent cache only).

### llama_state_seq_get_data_ext
```c
size_t llama_state_seq_get_data_ext(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    llama_seq_id seq_id,
    llama_state_seq_flags flags);
```
Get sequence state data with flags.

### llama_state_seq_set_data_ext
```c
size_t llama_state_seq_set_data_ext(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    llama_seq_id dest_seq_id,
    llama_state_seq_flags flags);
```
Set sequence state data with flags.

---

