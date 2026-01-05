# llama.cpp C API Reference - Core Functions

Part 1 of 6 | **Core** | [Model Info](api-model-info.md) | [Context](api-context.md) | [Inference](api-inference.md) | [Sampling](api-sampling.md) | [Advanced](api-advanced.md)

This is the core API reference covering initialization, parameters, and model loading. The complete API is split across 6 files for efficient targeted loading.

**Other API sections:**
- [api-model-info.md](api-model-info.md) - Model properties, architecture detection
- [api-context.md](api-context.md) - Context, memory (KV cache), state management
- [api-inference.md](api-inference.md) - Batch operations, inference, tokenization, chat templates
- [api-sampling.md](api-sampling.md) - All 25+ sampling strategies
- [api-advanced.md](api-advanced.md) - LoRA, performance, training, constants, structures

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

