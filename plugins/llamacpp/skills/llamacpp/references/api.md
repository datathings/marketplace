# llama.cpp C API Reference

This document provides a comprehensive reference for all non-deprecated functions in the llama.cpp C API. Functions are organized by category for easy navigation.

## Table of Contents

1. [Initialization & Backend](#initialization--backend)
2. [Parameter Helpers](#parameter-helpers)
3. [Model Loading & Management](#model-loading--management)
4. [Model Properties & Metadata](#model-properties--metadata)
5. [Context Management](#context-management)
6. [Memory (KV Cache) Management](#memory-kv-cache-management)
7. [State & Session Management](#state--session-management)
8. [Batch Operations](#batch-operations)
9. [Inference & Decoding](#inference--decoding)
10. [Vocabulary & Tokenization](#vocabulary--tokenization)
11. [Chat Templates](#chat-templates)
12. [Sampling](#sampling)
13. [LoRA Adapters](#lora-adapters)
14. [Performance & Utilities](#performance--utilities)
15. [Training](#training)

## Quick Reference

### Most Common Functions

**Model & Context:**
- `llama_backend_init()` - Initialize backend
- `llama_model_load_from_file()` - Load GGUF model
- `llama_init_from_model()` - Create inference context
- `llama_model_free()`, `llama_free()` - Cleanup

**Tokenization:**
- `llama_tokenize()` - Text → tokens
- `llama_detokenize()` - Tokens → text
- `llama_token_to_piece()` - Single token → text

**Inference:**
- `llama_decode()` - Process token batch
- `llama_get_logits_ith()` - Get token probabilities
- `llama_get_embeddings_ith()` - Extract embeddings

**Sampling:**
- `llama_sampler_chain_init()` - Create sampler
- `llama_sampler_sample()` - Sample next token
- `llama_vocab_is_eog()` - Check for end-of-generation

**Memory Management:**
- `llama_memory_clear()` - Clear KV cache
- `llama_memory_seq_rm()` - Remove sequence
- `llama_memory_seq_cp()` - Copy sequence

See categories below for complete function listings.

---

## Initialization & Backend

### llama_backend_init
```c
void llama_backend_init(void);
```
Initialize the llama + ggml backend. Call once at the start of the program.

**Usage:**
```c
llama_backend_init();
```

### llama_backend_free
```c
void llama_backend_free(void);
```
Free backend resources. Call once at the end of the program. Currently only used for MPI.

### llama_numa_init
```c
void llama_numa_init(enum ggml_numa_strategy numa);
```
Optional: Initialize NUMA optimizations.

**Parameters:**
- `numa`: NUMA strategy to use (from ggml)

### llama_attach_threadpool
```c
void llama_attach_threadpool(
    struct llama_context * ctx,
    ggml_threadpool_t threadpool,
    ggml_threadpool_t threadpool_batch);
```
Optional: Attach a custom threadpool. An auto threadpool is created in ggml if not passed explicitly.

**Parameters:**
- `ctx`: Context to attach threadpool to
- `threadpool`: Threadpool for single-token generation
- `threadpool_batch`: Threadpool for batch processing

### llama_detach_threadpool
```c
void llama_detach_threadpool(struct llama_context * ctx);
```
Detach threadpool from context.

---

## Parameter Helpers

### llama_model_default_params
```c
struct llama_model_params llama_model_default_params(void);
```
Get default model parameters. Always use this to initialize `llama_model_params` before modifying specific fields.

**Usage:**
```c
struct llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 32;  // Override specific fields
```

### llama_context_default_params
```c
struct llama_context_params llama_context_default_params(void);
```
Get default context parameters. Always use this to initialize `llama_context_params`.

**Usage:**
```c
struct llama_context_params params = llama_context_default_params();
params.n_ctx = 4096;
params.n_batch = 512;
```

### llama_sampler_chain_default_params
```c
struct llama_sampler_chain_params llama_sampler_chain_default_params(void);
```
Get default sampler chain parameters.

### llama_model_quantize_default_params
```c
struct llama_model_quantize_params llama_model_quantize_default_params(void);
```
Get default quantization parameters.

---

## Model Loading & Management

### llama_model_load_from_file
```c
struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params);
```
Load a model from a file. If the file is split into multiple parts, the file name must follow this pattern: `<name>-%05d-of-%05d.gguf`. Returns NULL on failure.

**Parameters:**
- `path_model`: Path to the GGUF model file
- `params`: Model loading parameters (get from `llama_model_default_params()`)

**Usage:**
```c
struct llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 32;
struct llama_model * model = llama_model_load_from_file("model.gguf", params);
if (!model) {
    // Handle error
}
```

### llama_model_load_from_splits
```c
struct llama_model * llama_model_load_from_splits(
    const char ** paths,
    size_t n_paths,
    struct llama_model_params params);
```
Load a model from multiple split files (supports custom naming schemes). The paths must be in the correct order.

**Parameters:**
- `paths`: Array of paths to split files
- `n_paths`: Number of split files
- `params`: Model loading parameters

### llama_model_save_to_file
```c
void llama_model_save_to_file(
    const struct llama_model * model,
    const char * path_model);
```
Save a model to a file.

### llama_model_free
```c
void llama_model_free(struct llama_model * model);
```
Free a loaded model. Always call this when done with a model.

**Usage:**
```c
llama_model_free(model);
```

### llama_model_quantize
```c
uint32_t llama_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    const llama_model_quantize_params * params);
```
Quantize a model. Returns 0 on success.

**Parameters:**
- `fname_inp`: Input model file path
- `fname_out`: Output model file path
- `params`: Quantization parameters

---

## Model Properties & Metadata

### llama_model_get_vocab
```c
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
```
Get the model's vocabulary.

### llama_model_rope_type
```c
enum llama_rope_type llama_model_rope_type(const struct llama_model * model);
```
Get the model's RoPE type.

### llama_model_n_ctx_train
```c
int32_t llama_model_n_ctx_train(const struct llama_model * model);
```
Get the context size the model was trained with.

### llama_model_n_embd
```c
int32_t llama_model_n_embd(const struct llama_model * model);
```
Get the embedding dimension.

### llama_model_n_embd_inp
```c
int32_t llama_model_n_embd_inp(const struct llama_model * model);
```
Get the input embedding dimension.

### llama_model_n_layer
```c
int32_t llama_model_n_layer(const struct llama_model * model);
```
Get the number of layers in the model.

### llama_model_n_head
```c
int32_t llama_model_n_head(const struct llama_model * model);
```
Get the number of attention heads.

### llama_model_n_head_kv
```c
int32_t llama_model_n_head_kv(const struct llama_model * model);
```
Get the number of KV heads (for grouped-query attention).

### llama_model_n_swa
```c
int32_t llama_model_n_swa(const struct llama_model * model);
```
Get the sliding window attention size.

### llama_model_rope_freq_scale_train
```c
float llama_model_rope_freq_scale_train(const struct llama_model * model);
```
Get the model's RoPE frequency scaling factor.

### llama_model_n_cls_out
```c
uint32_t llama_model_n_cls_out(const struct llama_model * model);
```
Get the number of classifier outputs (only valid for classifier models). Undefined behavior for non-classifier models.

### llama_model_cls_label
```c
const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);
```
Get the label of a classifier output by index. Returns NULL if no label provided.

### llama_model_meta_val_str
```c
int32_t llama_model_meta_val_str(
    const struct llama_model * model,
    const char * key,
    char * buf,
    size_t buf_size);
```
Get metadata value as a string by key name. Returns the length of the string on success, or -1 on failure. The output string is always null-terminated.

### llama_model_meta_count
```c
int32_t llama_model_meta_count(const struct llama_model * model);
```
Get the number of metadata key/value pairs.

### llama_model_meta_key_str
```c
const char * llama_model_meta_key_str(enum llama_model_meta_key key);
```
Get sampling metadata key name. Returns NULL if the key is invalid.

### llama_model_meta_key_by_index
```c
int32_t llama_model_meta_key_by_index(
    const struct llama_model * model,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata key name by index.

### llama_model_meta_val_str_by_index
```c
int32_t llama_model_meta_val_str_by_index(
    const struct llama_model * model,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata value as a string by index.

### llama_model_desc
```c
int32_t llama_model_desc(
    const struct llama_model * model,
    char * buf,
    size_t buf_size);
```
Get a string describing the model type.

### llama_model_size
```c
uint64_t llama_model_size(const struct llama_model * model);
```
Get the total size of all tensors in the model in bytes.

### llama_model_n_params
```c
uint64_t llama_model_n_params(const struct llama_model * model);
```
Get the total number of parameters in the model.

### llama_model_has_encoder
```c
bool llama_model_has_encoder(const struct llama_model * model);
```
Returns true if the model contains an encoder that requires `llama_encode()` call.

### llama_model_has_decoder
```c
bool llama_model_has_decoder(const struct llama_model * model);
```
Returns true if the model contains a decoder that requires `llama_decode()` call.

### llama_model_decoder_start_token
```c
llama_token llama_model_decoder_start_token(const struct llama_model * model);
```
For encoder-decoder models, returns the token ID that must be provided to the decoder to start generating. Returns -1 for other models.

### llama_model_is_recurrent
```c
bool llama_model_is_recurrent(const struct llama_model * model);
```
Returns true if the model is recurrent (like Mamba, RWKV, etc.).

### llama_model_is_hybrid
```c
bool llama_model_is_hybrid(const struct llama_model * model);
```
Returns true if the model is hybrid (like Jamba, Granite, etc.).

### llama_model_is_diffusion
```c
bool llama_model_is_diffusion(const struct llama_model * model);
```
Returns true if the model is diffusion-based (like LLaDA, Dream, etc.).

### llama_model_chat_template
```c
const char * llama_model_chat_template(
    const struct llama_model * model,
    const char * name);
```
Get the default chat template. Returns NULL if not available. If `name` is NULL, returns the default chat template.

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

**New parameters in b7572:**

- `kv_unified` (bool) - Use unified KV cache buffer (experimental). Enables a more memory-efficient cache layout. Default: false.

- `swa_full` (bool) - For models with Sliding Window Attention (SWA), allocate full context size instead of just the attention window. Set to true when you need to access tokens outside the SWA window. Check `llama_model_n_swa()` to detect if a model uses SWA. Default: false.

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

## Batch Operations

### llama_batch_get_one
```c
struct llama_batch llama_batch_get_one(
    llama_token * tokens,
    int32_t n_tokens);
```
Return batch for single sequence of tokens. The sequence ID will be fixed to 0. The position of tokens will be tracked automatically. This is a helper function to facilitate transition to the new batch API - avoid using it for new code.

**Usage:**
```c
llama_token tokens[] = {1, 2, 3, 4, 5};
struct llama_batch batch = llama_batch_get_one(tokens, 5);
llama_decode(ctx, batch);
```

### llama_batch_init
```c
struct llama_batch llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max);
```
Allocate a batch of tokens on the heap. Must be freed with `llama_batch_free()`.

**Parameters:**
- `n_tokens`: Maximum number of tokens
- `embd`: If != 0, allocate `llama_batch.embd` with size `n_tokens * embd * sizeof(float)`. Otherwise, allocate `llama_batch.token` to store `n_tokens` token IDs
- `n_seq_max`: Maximum number of sequences each token can be assigned to

**Usage:**
```c
struct llama_batch batch = llama_batch_init(512, 0, 1);
// Use batch...
llama_batch_free(batch);
```

### llama_batch_free
```c
void llama_batch_free(struct llama_batch batch);
```
Free a batch allocated with `llama_batch_init()`.

---

## Inference & Decoding

### llama_encode
```c
int32_t llama_encode(
    struct llama_context * ctx,
    struct llama_batch batch);
```
Process a batch of tokens without using KV cache. For encoder-decoder models, processes the batch using the encoder and stores the output internally for later use by the decoder's cross-attention layers.

**Returns:**
- `0`: Success
- `< 0`: Error (memory state is restored)

### llama_decode
```c
int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);
```
Process a batch of tokens. Requires the context to have memory. For encoder-decoder models, processes using the decoder.

**Returns:**
- `0`: Success
- `1`: Could not find a KV slot (try reducing batch size or increase context)
- `2`: Aborted (processed ubatches remain in memory)
- `-1`: Invalid input batch
- `< -1`: Fatal error (processed ubatches remain in memory)

**Usage:**
```c
struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    // Handle error
}
```

### llama_set_n_threads
```c
void llama_set_n_threads(
    struct llama_context * ctx,
    int32_t n_threads,
    int32_t n_threads_batch);
```
Set the number of threads used for decoding.

**Parameters:**
- `ctx`: Context
- `n_threads`: Number of threads for generation (single token)
- `n_threads_batch`: Number of threads for batch processing (multiple tokens)

### llama_n_threads
```c
int32_t llama_n_threads(struct llama_context * ctx);
```
Get the number of threads used for generation of a single token.

### llama_n_threads_batch
```c
int32_t llama_n_threads_batch(struct llama_context * ctx);
```
Get the number of threads used for batch processing.

### llama_set_embeddings
```c
void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
```
Set whether the context outputs embeddings or not.

### llama_set_causal_attn
```c
void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);
```
Set whether to use causal attention or not. If set to true, the model will only attend to past tokens.

### llama_set_warmup
```c
void llama_set_warmup(struct llama_context * ctx, bool warmup);
```
Set whether the model is in warmup mode. If true, all model tensors are activated during `llama_decode()` to load and cache their weights.

### llama_set_abort_callback
```c
void llama_set_abort_callback(
    struct llama_context * ctx,
    ggml_abort_callback abort_callback,
    void * abort_callback_data);
```
Set abort callback. If it returns true, execution will be aborted (currently only works with CPU execution).

### llama_synchronize
```c
void llama_synchronize(struct llama_context * ctx);
```
Wait until all computations are finished. Automatically done when obtaining results, not usually necessary to call explicitly.

### llama_get_logits
```c
float * llama_get_logits(struct llama_context * ctx);
```
Get token logits from the last `llama_decode()` call. Logits for which `llama_batch.logits[i] != 0` are stored contiguously.

**Returns:** Pointer to logits array. Shape: `[n_outputs, n_vocab]`

### llama_get_logits_ith
```c
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
```
Get logits for the i-th token. Negative indices access logits in reverse order (-1 is the last token). Returns NULL for invalid IDs.

**Usage:**
```c
float * logits = llama_get_logits_ith(ctx, -1);  // Get logits for last token
```

### llama_get_embeddings
```c
float * llama_get_embeddings(struct llama_context * ctx);
```
Get all output token embeddings. Returns NULL when `pooling_type == LLAMA_POOLING_TYPE_NONE` with generative models.

**Returns:** Pointer to embeddings array. Shape: `[n_outputs * n_embd]`

### llama_get_embeddings_ith
```c
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
```
Get embeddings for the i-th token. Negative indices can be used (-1 is last). Returns NULL for invalid IDs.

**Returns:** Shape: `[n_embd]`

### llama_get_embeddings_seq
```c
float * llama_get_embeddings_seq(
    struct llama_context * ctx,
    llama_seq_id seq_id);
```
Get embeddings for a sequence ID. Returns NULL if `pooling_type` is `LLAMA_POOLING_TYPE_NONE`. For `LLAMA_POOLING_TYPE_RANK`, returns `float[n_cls_out]` with rank(s).

**Returns:** Shape: `[n_embd]` or `[n_cls_out]` for ranking models

---

## Vocabulary & Tokenization

### llama_vocab_type
```c
enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);
```
Get the vocabulary type (SPM, BPE, WPM, UGM, RWKV, PLAMO2).

### llama_vocab_n_tokens
```c
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
```
Get the number of tokens in the vocabulary.

### llama_vocab_get_text
```c
const char * llama_vocab_get_text(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the text representation of a token.

### llama_vocab_get_score
```c
float llama_vocab_get_score(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the score of a token.

### llama_vocab_get_attr
```c
enum llama_token_attr llama_vocab_get_attr(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the attributes of a token (bitfield of `llama_token_attr`).

### llama_vocab_is_eog
```c
bool llama_vocab_is_eog(
    const struct llama_vocab * vocab,
    llama_token token);
```
Check if the token is an end-of-generation token (EOS, EOT, etc.).

### llama_vocab_is_control
```c
bool llama_vocab_is_control(
    const struct llama_vocab * vocab,
    llama_token token);
```
Check if the token is a control token or a renderable token.

### Special Token Functions

Get special token IDs:

```c
llama_token llama_vocab_bos(const struct llama_vocab * vocab);   // beginning-of-sentence
llama_token llama_vocab_eos(const struct llama_vocab * vocab);   // end-of-sentence
llama_token llama_vocab_eot(const struct llama_vocab * vocab);   // end-of-turn
llama_token llama_vocab_sep(const struct llama_vocab * vocab);   // sentence separator
llama_token llama_vocab_nl(const struct llama_vocab * vocab);    // next-line
llama_token llama_vocab_pad(const struct llama_vocab * vocab);   // padding
llama_token llama_vocab_mask(const struct llama_vocab * vocab);  // mask
```

Check if special tokens should be added:

```c
bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);
```

Fill-in-the-middle tokens:

```c
llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
```

### llama_tokenize
```c
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);
```
Convert text into tokens.

**Parameters:**
- `vocab`: Vocabulary
- `text`: Text to tokenize
- `text_len`: Length of text
- `tokens`: Output buffer (must be large enough)
- `n_tokens_max`: Maximum number of tokens
- `add_special`: Allow adding BOS and EOS tokens if model is configured to do so
- `parse_special`: Allow tokenizing special/control tokens (otherwise treated as plaintext)

**Returns:**
- Positive: Number of tokens (no more than `n_tokens_max`)
- Negative: Number of tokens that would have been returned (buffer too small)
- `INT32_MIN`: Overflow

**Usage:**
```c
const char * text = "Hello, world!";
llama_token tokens[128];
int n = llama_tokenize(vocab, text, strlen(text), tokens, 128, true, false);
if (n < 0) {
    // Buffer too small, need -n tokens
}
```

### llama_token_to_piece
```c
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);
```
Convert a token ID to text. Does not write null terminator.

**Parameters:**
- `vocab`: Vocabulary
- `token`: Token ID
- `buf`: Output buffer
- `length`: Buffer length
- `lstrip`: Number of leading spaces to skip (useful when encoding/decoding multiple tokens)
- `special`: If true, special tokens are rendered

### llama_detokenize
```c
int32_t llama_detokenize(
    const struct llama_vocab * vocab,
    const llama_token * tokens,
    int32_t n_tokens,
    char * text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special);
```
Convert tokens back into text.

**Parameters:**
- `vocab`: Vocabulary
- `tokens`: Array of tokens
- `n_tokens`: Number of tokens
- `text`: Output buffer
- `text_len_max`: Maximum text length
- `remove_special`: Remove BOS and EOS tokens if model is configured to do so
- `unparse_special`: If true, special tokens are rendered

**Returns:**
- Positive: Number of chars/bytes (no more than `text_len_max`)
- Negative: Number of chars/bytes that would have been returned

---

## Chat Templates

### llama_chat_apply_template
```c
int32_t llama_chat_apply_template(
    const char * tmpl,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);
```
Apply chat template to format a conversation. Does not use a Jinja parser - only supports a pre-defined list of templates.

**Parameters:**
- `tmpl`: Jinja template (NULL to use model's default)
- `chat`: Array of chat messages
- `n_msg`: Number of messages
- `add_ass`: Whether to end prompt with assistant message start token(s)
- `buf`: Output buffer (recommended size: 2 * total characters of all messages)
- `length`: Buffer size

**Returns:** Total number of bytes of the formatted prompt

**Usage:**
```c
llama_chat_message messages[] = {
    {"system", "You are a helpful assistant."},
    {"user", "Hello!"}
};
char buf[1024];
int len = llama_chat_apply_template(NULL, messages, 2, true, buf, 1024);
```

### llama_chat_builtin_templates
```c
int32_t llama_chat_builtin_templates(const char ** output, size_t len);
```
Get list of built-in chat templates.

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
- if i == -1, returns the chain itself (can check if sampler is a chain)

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

#### Adaptive-P Sampler

```c
struct llama_sampler * llama_sampler_init_adaptive_p(
    float target,
    float decay,
    uint32_t seed);
```
Adaptive-P sampler - selects tokens near a configurable target probability over time.

The sampler transforms the token probability distribution to favor tokens near a user-configurable probability target. Internally maintains an exponential moving average (EMA) of original probabilities of selected tokens, using this to compute an adapted target at each step.

**Parameters:**
- `target`: Select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
- `decay`: EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
- `seed`: Random seed. Use `LLAMA_DEFAULT_SEED` for a random seed.

**Important:** This sampler selects a token ID (like mirostat, dist, greedy), so it must be **last in the sampler chain**. Only mild truncation before this sampler is recommended - use min-p as the only other active sampler.

**Reference:** https://github.com/ggml-org/llama.cpp/pull/17927

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

### Backend Sampling API [EXPERIMENTAL]

Backend sampling allows sampling operations to be performed directly on the GPU as part of the computation graph.

**Note:** Use only if the `llama_context` was created with at least one `llama_sampler_seq_config`.

#### llama_set_sampler
```c
bool llama_set_sampler(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    struct llama_sampler * smpl);
```
Attach a sampler to the context for a specific sequence. Prefer initializing the context with `llama_context_params.samplers` when possible.

#### llama_get_sampled_token_ith
```c
llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled token for the i-th token. Returns `LLAMA_TOKEN_NULL` if no token was sampled.

#### llama_get_sampled_probs_ith / llama_get_sampled_probs_count_ith
```c
float * llama_get_sampled_probs_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled probabilities for the i-th token. Returns NULL if no probabilities were generated.

#### llama_get_sampled_logits_ith / llama_get_sampled_logits_count_ith
```c
float * llama_get_sampled_logits_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled logits for the i-th token. Returns NULL if no logits were sampled.

#### llama_get_sampled_candidates_ith / llama_get_sampled_candidates_count_ith
```c
llama_token * llama_get_sampled_candidates_ith(struct llama_context * ctx, int32_t i);
uint32_t llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);
```
Get the backend sampled candidates (token ids) for the i-th token. Returns NULL if no candidates were sampled.

---

## LoRA Adapters

### llama_adapter_lora_init
```c
struct llama_adapter_lora * llama_adapter_lora_init(
    struct llama_model * model,
    const char * path_lora);
```
Load a LoRA adapter from file. Adapters are automatically freed when the model is freed.

### llama_adapter_meta_val_str
```c
int32_t llama_adapter_meta_val_str(
    const struct llama_adapter_lora * adapter,
    const char * key,
    char * buf,
    size_t buf_size);
```
Get adapter metadata value as a string by key.

### llama_adapter_meta_count
```c
int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);
```
Get the number of metadata key/value pairs.

### llama_adapter_meta_key_by_index
```c
int32_t llama_adapter_meta_key_by_index(
    const struct llama_adapter_lora * adapter,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata key name by index.

### llama_adapter_meta_val_str_by_index
```c
int32_t llama_adapter_meta_val_str_by_index(
    const struct llama_adapter_lora * adapter,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata value as a string by index.

### llama_adapter_get_alora_n_invocation_tokens
```c
uint64_t llama_adapter_get_alora_n_invocation_tokens(
    const struct llama_adapter_lora * adapter);
```
Get the number of invocation tokens if this is an A-LoRA adapter.

### llama_adapter_get_alora_invocation_tokens
```c
const llama_token * llama_adapter_get_alora_invocation_tokens(
    const struct llama_adapter_lora * adapter);
```
Get the invocation tokens if this is an A-LoRA adapter.

### llama_set_adapters_lora
```c
int32_t llama_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    float * scales);
```
Set multiple LoRA adapters to the context with individual scaling factors. Replaces any currently active adapters.

**Parameters:**
- `ctx`: Context
- `adapters`: Array of LoRA adapter pointers
- `n_adapters`: Number of adapters in the array
- `scales`: Array of scaling factors (one per adapter)

**Returns:** 0 on success, non-zero on failure

**Note:** Pass `n_adapters = 0` to clear all adapters from the context.

### llama_set_adapter_cvec
```c
int32_t llama_set_adapter_cvec(
    struct llama_context * ctx,
    const float * data,
    size_t len,
    int32_t n_embd,
    int32_t il_start,
    int32_t il_end);
```
Apply a loaded control vector to the context. If `data` is NULL, clear the currently loaded vector.

**Parameters:**
- `ctx`: Context
- `data`: Control vector data (n_embd x n_layers buffer starting from layer 1), or NULL to clear
- `len`: Length of data
- `n_embd`: Size of a single layer's control
- `il_start`: Start layer (inclusive)
- `il_end`: End layer (inclusive)

**Returns:** 0 on success, non-zero on failure

---

## Performance & Utilities

### System Information

```c
const char * llama_print_system_info(void);
```
Get system information as a string.

```c
int64_t llama_time_us(void);
```
Get current time in microseconds.

```c
size_t llama_max_devices(void);
```
Get the maximum number of devices.

```c
size_t llama_max_parallel_sequences(void);
```
Get the maximum number of parallel sequences.

```c
size_t llama_max_tensor_buft_overrides(void);
```
Get the maximum number of tensor buffer type overrides.

```c
bool llama_supports_mmap(void);
```
Check if mmap is supported.

```c
bool llama_supports_mlock(void);
```
Check if mlock is supported.

```c
bool llama_supports_gpu_offload(void);
```
Check if GPU offload is supported.

```c
bool llama_supports_rpc(void);
```
Check if RPC is supported.

### Performance Measurement

```c
struct llama_perf_context_data llama_perf_context(
    const struct llama_context * ctx);
```
Get performance data for the context.

```c
void llama_perf_context_print(const struct llama_context * ctx);
```
Print performance statistics for the context.

```c
void llama_perf_context_reset(struct llama_context * ctx);
```
Reset performance counters for the context.

```c
struct llama_perf_sampler_data llama_perf_sampler(
    const struct llama_sampler * chain);
```
Get performance data for the sampler chain. **Note:** Only works with samplers constructed via `llama_sampler_chain_init()`.

```c
void llama_perf_sampler_print(const struct llama_sampler * chain);
```
Print performance statistics for the sampler.

```c
void llama_perf_sampler_reset(struct llama_sampler * chain);
```
Reset performance counters for the sampler.

```c
void llama_memory_breakdown_print(const struct llama_context * ctx);
```
Print a breakdown of per-device memory use via `LLAMA_LOG`.

### Logging

```c
void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
```
Get the current log callback and user data.

```c
void llama_log_set(ggml_log_callback log_callback, void * user_data);
```
Set callback for all future logging events. If NULL, everything is output on stderr. **Note:** Logger state is global, so these functions are NOT thread-safe.

### Model Split Utilities

```c
int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count);
```
Build a split GGUF file path for a chunk.

**Example:**
```c
char split_path[256];
llama_split_path(split_path, 256, "/models/ggml-model-q4_0", 2, 4);
// Result: "/models/ggml-model-q4_0-00002-of-00004.gguf"
```

```c
int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count);
```
Extract the path prefix from a split path if and only if split_no and split_count match.

### Flash Attention

```c
const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);
```
Get the name of a flash attention type.

---

## Training

### llama_opt_param_filter_all
```c
bool llama_opt_param_filter_all(
    const struct ggml_tensor * tensor,
    void * userdata);
```
Parameter filter that always returns true (all tensors contain trainable parameters).

### llama_opt_init
```c
void llama_opt_init(
    struct llama_context * lctx,
    struct llama_model * model,
    struct llama_opt_params lopt_params);
```
Initialize optimization/training for a model.

**Parameters:**
- `lctx`: Context
- `model`: Model to train
- `lopt_params`: Optimization parameters

### llama_opt_epoch
```c
void llama_opt_epoch(
    struct llama_context * lctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result_train,
    ggml_opt_result_t result_eval,
    int64_t idata_split,
    ggml_opt_epoch_callback callback_train,
    ggml_opt_epoch_callback callback_eval);
```
Run a training epoch.

**Parameters:**
- `lctx`: Context
- `dataset`: Training dataset
- `result_train`: Training results
- `result_eval`: Evaluation results
- `idata_split`: Data split index
- `callback_train`: Training callback
- `callback_eval`: Evaluation callback

---

## Important Constants

```c
#define LLAMA_DEFAULT_SEED 0xFFFFFFFF
#define LLAMA_TOKEN_NULL -1
```

## Key Data Structures

### llama_batch
Input data for `llama_encode`/`llama_decode`:
- `n_tokens`: Number of tokens in the batch
- `token`: Token IDs (used when `embd` is NULL)
- `embd`: Token embeddings (used when `token` is NULL)
- `pos`: Token positions (NULL for automatic tracking)
- `seq_id`: Sequence IDs for each token (NULL defaults to sequence 0)
- `logits`: Whether to output logits for each token (NULL outputs last token only for generation)

### llama_model_params
Model loading parameters (get defaults via `llama_model_default_params()`):
- `devices`: NULL-terminated list of devices for offloading
- `n_gpu_layers`: Number of layers to store in VRAM
- `split_mode`: How to split the model across GPUs
- `vocab_only`: Only load vocabulary, no weights
- `use_mmap`: Use mmap if possible
- `use_direct_io`: Use direct I/O when supported (takes precedence over use_mmap)
- `use_mlock`: Force system to keep model in RAM

### llama_context_params
Context parameters (get defaults via `llama_context_default_params()`):
- `n_ctx`: Text context size (0 = from model)
- `n_batch`: Logical maximum batch size
- `n_ubatch`: Physical maximum batch size
- `n_seq_max`: Max number of sequences
- `n_threads`: Threads for generation
- `n_threads_batch`: Threads for batch processing
- `embeddings`: Extract embeddings
- `rope_scaling_type`: RoPE scaling type
- `pooling_type`: Pooling type
- `attention_type`: Attention type
- `flash_attn_type`: Flash attention configuration

### llama_token_data / llama_token_data_array
Used for sampling:
- `llama_token_data`: Contains token ID, logit, and probability
- `llama_token_data_array`: Array of token data with selection index and sorted flag
