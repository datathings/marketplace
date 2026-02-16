# llama.cpp C API Reference - Advanced Features

Part 6 of 6 | [Core](api-core.md) | [Model Info](api-model-info.md) | [Context](api-context.md) | [Inference](api-inference.md) | [Sampling](api-sampling.md) | **Advanced**

This file covers:
- LoRA Adapters - Load and apply LoRA adapters, control vectors
- Performance & Utilities - Performance measurement, logging, system info
- Training - Fine-tuning and training functions
- Important Constants - Key constants and enums
- Key Data Structures - Core struct definitions

For complete API navigation, see [api-core.md](api-core.md).

---

## LoRA Adapters

### llama_adapter_lora_init
```c
struct llama_adapter_lora * llama_adapter_lora_init(
    struct llama_model * model,
    const char * path_lora);
```
Load a LoRA adapter from file.

**Important:**
- Adapters are automatically freed when the model is freed
- All adapters must be loaded before context creation

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
