# Model - LLM Model API

The Model type provides comprehensive access to the llama.cpp library for loading and using large language models (LLMs). This is the main entry point for working with language models in GreyCat.

## Overview

The Model API supports:
- Loading GGUF model files (single or split)
- Text embeddings generation
- Text completion and generation
- Chat conversations with template formatting
- Tokenization and detokenization
- Model metadata inspection
- Performance monitoring
- Model quantization

## Quick Start

```gcl
// Load a model
var model = Model::load("my-model", "./Llama-3.2-1B.gguf", ModelParams { n_gpu_layers: -1 });

// Get model information
var info = model.info();
println("Loaded ${info.description} with ${info.n_params} parameters");

// Generate text
var result = model.generate("Once upon a time", GenerationParams { max_tokens: 100 }, null);
println(result.text);

// Generate embeddings
var emb = model.embed("Hello world", TensorType::f32, null);
println("Embedding dimension: ${emb.size()}");

// Clean up (optional - GC handles this automatically)
model.free();
```

## Types

### Model

Main language model type with methods for inference, embeddings, tokenization, and metadata access.

### ModelParams

Model loading parameters.

**Fields:**
- `n_gpu_layers: int?` - Number of layers to offload to GPU (0 = CPU only, -1 = all layers)
- `split_mode: SplitMode?` - How to split the model across multiple GPUs
- `main_gpu: int?` - The GPU index to use when split_mode is none (default: 0)
- `tensor_split: Array<float>?` - Proportion of tensor rows to offload to each GPU
- `vocab_only: bool?` - Only load the vocabulary, no weights (useful for tokenization only)
- `use_mmap: bool?` - Use memory-mapped files if possible (faster loading, shared memory)
- `use_mlock: bool?` - Force system to keep model in RAM (prevent swapping)
- `check_tensors: bool?` - Validate model tensor data during loading

### ContextParams

Context creation parameters. Controls inference behavior, memory usage, and performance.

**Core Parameters:**
- `n_ctx: int?` - Text context size, 0 = from model
- `n_batch: int?` - Logical maximum batch size that can be submitted to decode
- `n_ubatch: int?` - Physical maximum batch size
- `n_seq_max: int?` - Max number of sequences (for recurrent models)

**Threading:**
- `n_threads: int?` - Number of threads to use for generation
- `n_threads_batch: int?` - Number of threads to use for batch processing

**Attention & Pooling:**
- `rope_scaling_type: RopeScalingType?` - RoPE scaling type
- `pooling_type: PoolingType?` - Whether to pool (sum) embedding results by sequence id
- `attention_type: AttentionType?` - Attention type to use for embeddings
- `flash_attn_type: FlashAttnType?` - When to enable Flash Attention

**RoPE Parameters:**
- `rope_freq_base: float?` - RoPE base frequency, 0 = from model
- `rope_freq_scale: float?` - RoPE frequency scaling factor, 0 = from model

**YaRN Parameters:**
- `yarn_ext_factor: float?` - YaRN extrapolation mix factor
- `yarn_attn_factor: float?` - YaRN magnitude scaling factor
- `yarn_beta_fast: float?` - YaRN low correction dim
- `yarn_beta_slow: float?` - YaRN high correction dim
- `yarn_orig_ctx: int?` - YaRN original context size

**KV Cache:**
- `defrag_thold: float?` - Defragment KV cache if holes/size > thold, <= 0 disabled
- `type_k: GgmlType?` - Data type for K cache
- `type_v: GgmlType?` - Data type for V cache

**Feature Flags:**
- `embeddings: bool?` - If true, extract embeddings (together with logits)
- `normalize: bool?` - If true, normalize embeddings to unit length (L2 norm)
- `offload_kqv: bool?` - Offload the KQV ops (including the KV cache) to GPU
- `no_perf: bool?` - Disable performance timings (for production use)
- `op_offload: bool?` - Offload host tensor operations to device
- `swa_full: bool?` - Use full-size SWA cache
- `kv_unified: bool?` - Use a unified buffer across input sequences for attention

## Enums

### SplitMode

How to split the model across multiple GPUs.

- `none` - Single GPU
- `layer` - Split layers and KV across GPUs
- `row` - Split layers and KV across GPUs, use tensor parallelism if supported

### RopeScalingType

RoPE (Rotary Position Embedding) scaling type for context extension.

- `unspecified` - No scaling, unspecified
- `none` - No scaling
- `linear` - Linear scaling
- `yarn` - YaRN scaling
- `longrope` - Long RoPE scaling

### PoolingType

Pooling strategy for embeddings.

- `unspecified` - Use model default
- `none` - No pooling
- `mean` - Average all token embeddings
- `cls` - Use CLS token embedding
- `last` - Use last token embedding
- `rank` - Ranking mode (outputs class scores)

### AttentionType

Attention mechanism type.

- `unspecified` - Use model default
- `causal` - Causal attention (GPT-style)
- `non_causal` - Non-causal attention

### FlashAttnType

Flash Attention configuration.

- `disabled` - Disabled
- `enabled_for_fa` - Only enable for models that support FlashAttention natively
- `enabled_for_all` - Enable for all models (may be slower for non-FA models)

### GgmlType

GGML tensor data types (quantization formats).

- `f32`, `f16`, `bf16`, `f64` - Floating point formats
- `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1` - Standard quantization
- `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `q8_k` - K-quantization
- `iq2_xxs`, `iq2_xs`, `iq3_xxs`, `iq1_s`, `iq4_nl`, `iq3_s`, `iq2_s`, `iq4_xs`, `iq1_m` - IQ quantization
- `tq1_0`, `tq2_0` - TQ quantization
- `i8`, `i16`, `i32`, `i64` - Integer formats

## Static Methods

### `static fn load(name: String, path: String, params: ModelParams?): Model`

Load model from file with a name.

Creates a new Model instance by loading a GGUF model file from disk. The model is stored under the given name and can be retrieved later with get(). The model is kept in memory until freed (explicitly or by GC).

For split models, the file path must follow the pattern: `<name>-%05d-of-%05d.gguf` (e.g., `model-00001-of-00005.gguf`)

**Parameters:**
- `name` - Unique name to store the model under
- `path` - Path to GGUF model file
- `params` - Optional loading parameters

**Returns:** Model instance or null if loading fails

**Example:**
```gcl
var model = Model::load("llama3", "./models/Llama-3.2-1B.gguf", ModelParams {
    n_gpu_layers: -1,  // Offload all layers to GPU
    use_mmap: true,
    use_mlock: false
});
```

### `static fn get(name: String): Model?`

Get a previously loaded model by name.

Retrieves a model that was loaded with load().

**Parameters:**
- `name` - Name of the model to retrieve

**Returns:** Model instance or null if no model with that name exists

**Example:**
```gcl
var model = Model::get("llama3");
if (model == null) {
    println("Model not loaded");
}
```

### `static fn load_from_splits(paths: Array<String>, params: ModelParams?): Model`

Load model from custom-named split files.

Like load(), but allows custom file naming instead of the standard pattern. The paths array must contain all split files in the correct order.

**Parameters:**
- `paths` - Array of paths to split model files
- `params` - Optional loading parameters

**Returns:** Model instance or null if loading fails

**Example:**
```gcl
var model = Model::load_from_splits([
    "./models/part1.gguf",
    "./models/part2.gguf",
    "./models/part3.gguf"
], null);
```

### `static fn quantize(input_path: String, output_path: String, params: QuantizeParams?)`

Quantize a model file (convert to smaller precision).

Converts an existing GGUF model to a different quantization format. This is a static utility function - no Model instance required.

**Parameters:**
- `input_path` - Path to input model
- `output_path` - Path to output quantized model
- `params` - Optional quantization parameters

**Example:**
```gcl
Model::quantize(
    "./models/model-f16.gguf",
    "./models/model-q4.gguf",
    QuantizeParams { ftype: GgmlType::q4_k }
);
```

## Instance Methods - Model Information

### `fn info(): ModelInfo`

Get comprehensive model information.

Returns detailed metadata about the model architecture, size, training parameters, and capabilities.

**Returns:** ModelInfo structure with detailed model metadata

**Example:**
```gcl
var info = model.info();
println("Model: ${info.description}");
println("Parameters: ${info.n_params}");
println("Context size: ${info.n_ctx_train}");
println("Vocab size: ${info.n_vocab}");
println("Has encoder: ${info.has_encoder}");
```

### `fn meta(key: String): String?`

Get model metadata value by key.

Retrieves a specific metadata field from the GGUF file.

**Parameters:**
- `key` - Metadata key (e.g., "general.name", "general.author")

**Returns:** Metadata value or null if key doesn't exist

**Example:**
```gcl
var name = model.meta("general.name");
var author = model.meta("general.author");
var license = model.meta("general.license");
```

### `fn meta_count(): int`

Get number of metadata key/value pairs.

**Returns:** Total count of metadata entries in the model

### `fn meta_key_by_index(index: int): String?`

Get metadata key name by index.

**Parameters:**
- `index` - Index (0-based)

**Returns:** Key name or null if out of bounds

### `fn meta_val_by_index(index: int): String?`

Get metadata value by index.

**Parameters:**
- `index` - Index (0-based)

**Returns:** Value or null if out of bounds

### `fn desc(): String`

Get model description.

**Returns:** Human-readable string describing the model architecture

### `fn chat_template(name: String?): String?`

Get chat template by name.

**Parameters:**
- `name` - Template name (null for default)

**Returns:** Chat template string or null if not available

### `fn cls_label(index: int): String?`

Get classifier label by index.

For classifier models, returns the label for the output at the given index.

**Parameters:**
- `index` - Output index

**Returns:** Label or null if not a classifier or out of bounds

### `fn decoder_start_token(): int`

Get decoder start token.

For encoder-decoder models, returns the token ID that must be provided to the decoder to start generating output.

**Returns:** Decoder start token ID or -1 if not applicable

## Instance Methods - Embeddings

### `fn embed(text: String, tensor_type: TensorType, ctx_params: ContextParams?): Tensor`

Compute embedding vector for text.

Generates a dense vector representation of the input text. The pooling strategy is determined by ctx_params.pooling_type (defaults to model's trained pooling type).

**Parameters:**
- `text` - Input text to embed
- `tensor_type` - Output tensor data type
- `ctx_params` - Optional context parameters

**Returns:** 1D Tensor of size n_embd

**Example:**
```gcl
var emb = model.embed("Hello world", TensorType::f32, ContextParams {
    pooling_type: PoolingType::mean,
    normalize: true
});
println("Embedding: ${emb}");
```

### `fn embed_batch(texts: Array<String>, tensor_type: TensorType, ctx_params: ContextParams?): Array<Tensor>`

Compute embeddings for multiple texts (batched).

More efficient than calling embed() multiple times. All texts are processed in a single context creation.

**Parameters:**
- `texts` - Array of input texts
- `tensor_type` - Output tensor data type
- `ctx_params` - Optional context parameters

**Returns:** Array of Tensors, one per input text

**Example:**
```gcl
var texts = ["Hello world", "Goodbye world", "How are you?"];
var embs = model.embed_batch(texts, TensorType::f32, null);
for (var i = 0; i < embs.length; i++) {
    println("Embedding ${i}: dimension ${embs[i].size()}");
}
```

## Instance Methods - Text Generation

### `fn generate(prompt: String, params: GenerationParams?, ctx_params: ContextParams?): GenerationResult`

Generate text completion for a prompt.

Performs auto-regressive text generation starting from the prompt. Returns when max_tokens is reached or an end-of-generation token is produced.

**Parameters:**
- `prompt` - Input prompt text
- `params` - Optional generation parameters
- `ctx_params` - Optional context parameters

**Returns:** GenerationResult with generated text and metadata

**Example:**
```gcl
var result = model.generate(
    "Once upon a time",
    GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40
    },
    ContextParams { n_ctx: 2048 }
);
println("Generated: ${result.text}");
println("Tokens: ${result.n_tokens}");
println("Speed: ${result.perf.context.tokens_per_second} tok/s");
```

### `fn generate_stream(prompt: String, callback: function, params: GenerationParams?, ctx_params: ContextParams?): GenerationResult`

Generate text with streaming callback.

Like generate(), but calls the callback function for each generated token.

**Parameters:**
- `prompt` - Input prompt text
- `callback` - Callback function `fn(token: String, is_final: bool): bool`
- `params` - Optional generation parameters
- `ctx_params` - Optional context parameters

**Returns:** GenerationResult with complete generated text

**Example:**
```gcl
var result = model.generate_stream(
    "Write a story:",
    fn(token: String, is_final: bool): bool {
        print(token);  // Print each token as it's generated
        return true;   // Return false to stop generation
    },
    GenerationParams { max_tokens: 200 },
    null
);
```

## Instance Methods - Chat

### `fn format_chat(messages: Array<ChatMessage>, add_assistant: bool): String`

Format chat messages using model's chat template.

Applies the model's built-in chat template to format a conversation. If add_assistant is true, adds the assistant message start token(s).

**Parameters:**
- `messages` - Array of chat messages
- `add_assistant` - Whether to add assistant message start

**Returns:** Formatted prompt string ready for generation

**Example:**
```gcl
var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "What is the capital of France?" }
];
var prompt = model.format_chat(messages, true);
println(prompt);
```

### `fn chat(messages: Array<ChatMessage>, params: GenerationParams?, ctx_params: ContextParams?): GenerationResult`

Chat completion.

Convenience method that formats messages and generates a response. Equivalent to: `generate(format_chat(messages, true), params, ctx_params)`

**Parameters:**
- `messages` - Array of chat messages
- `params` - Optional generation parameters
- `ctx_params` - Optional context parameters

**Returns:** GenerationResult with assistant's response

**Example:**
```gcl
var result = model.chat([
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "Explain quantum computing" }
], GenerationParams { max_tokens: 300 }, null);
println(result.text);
```

### `fn chat_stream(messages: Array<ChatMessage>, callback: function, params: GenerationParams?, ctx_params: ContextParams?): GenerationResult`

Chat completion with streaming.

Like chat(), but streams tokens via callback.

**Parameters:**
- `messages` - Array of chat messages
- `callback` - Callback function `fn(token: String, is_final: bool): bool`
- `params` - Optional generation parameters
- `ctx_params` - Optional context parameters

**Returns:** GenerationResult with complete response

**Example:**
```gcl
model.chat_stream(
    [ChatMessage { role: "user", content: "Tell me a joke" }],
    fn(token: String, is_final: bool): bool {
        print(token);
        return true;
    },
    null,
    null
);
```

## Instance Methods - Tokenization

### `fn tokenize(text: String, add_special: bool, parse_special: bool): Array<int>`

Tokenize text into token IDs.

Converts text to an array of integer token IDs using the model's vocabulary.

**Parameters:**
- `text` - Input text
- `add_special` - If true, add BOS/EOS tokens if model is configured to
- `parse_special` - If true, parse special tokens like `<|endoftext|>`

**Returns:** Array of token IDs

**Example:**
```gcl
var tokens = model.tokenize("Hello world", true, false);
println("Tokens: ${tokens}");
println("Count: ${tokens.length}");
```

### `fn detokenize(tokens: Array<int>, remove_special: bool, unparse_special: bool): String`

Detokenize token IDs back to text.

Converts an array of token IDs back to a string.

**Parameters:**
- `tokens` - Array of token IDs
- `remove_special` - If true, remove BOS/EOS tokens
- `unparse_special` - If true, render special tokens as text

**Returns:** Decoded text string

**Example:**
```gcl
var text = model.detokenize(tokens, true, false);
println(text);
```

### `fn token_to_text(token: int): String`

Convert single token ID to text.

**Parameters:**
- `token` - Token ID

**Returns:** Text representation of the token

### `fn token_to_piece(token: int, lstrip: int, special: bool): String`

Convert token ID to piece with lstrip support.

Like token_to_text(), but allows skipping leading spaces.

**Parameters:**
- `token` - Token ID
- `lstrip` - Number of leading spaces to skip
- `special` - If true, render special tokens as text

**Returns:** Token piece string

### `fn token_score(token: int): float`

Get token score (log probability from training).

**Parameters:**
- `token` - Token ID

**Returns:** Token's score in the vocabulary

### `fn token_attr(token: int): TokenAttr`

Get token attributes (bitfield).

Returns metadata about the token (control, byte, normalized, etc.)

**Parameters:**
- `token` - Token ID

**Returns:** TokenAttr bitfield

### `fn is_eog_token(token: int): bool`

Check if token is end-of-generation.

**Parameters:**
- `token` - Token ID

**Returns:** True if the token marks the end of generation

### `fn is_control_token(token: int): bool`

Check if token is a control token.

**Parameters:**
- `token` - Token ID

**Returns:** True if the token is a special control token

## Instance Methods - Special Tokens

### `fn token_sep(): int`

Get separator token ID.

### `fn token_pad(): int`

Get padding token ID.

### `fn token_mask(): int`

Get mask token ID.

### `fn token_nl(): int`

Get newline token ID.

### `fn token_cls(): int`

Get CLS (classifier) token ID.

Returns the CLS token ID used by BERT-style models. Returns -1 if the model doesn't have a CLS token.

### `fn add_sep_token(): bool`

Check if model adds SEP token automatically.

## Instance Methods - Fill-in-the-Middle Tokens

### `fn token_fim_pre(): int`

Get fill-in-the-middle prefix token ID.

### `fn token_fim_suf(): int`

Get fill-in-the-middle suffix token ID.

### `fn token_fim_mid(): int`

Get fill-in-the-middle middle token ID.

### `fn token_fim_pad(): int`

Get fill-in-the-middle padding token ID.

### `fn token_fim_rep(): int`

Get fill-in-the-middle repeat token ID.

### `fn token_fim_sep(): int`

Get fill-in-the-middle separator token ID.

## Instance Methods - Performance

### `fn perf(): PerfData`

Get performance metrics for last operation.

Returns timing and throughput statistics from the most recent inference operation (embed, generate, etc.)

**Returns:** PerfData with timing metrics

**Example:**
```gcl
var perf = model.perf();
println("Generation speed: ${perf.context.tokens_per_second} tok/s");
println("Prompt speed: ${perf.context.prompt_tokens_per_second} tok/s");
```

### `fn print_memory()`

Print detailed memory breakdown to log.

Outputs per-device memory usage information via llama.cpp logging.

## Instance Methods - Resource Management

### `fn save(path: String)`

Save model to file.

Saves the current model to a GGUF file on disk.

**Parameters:**
- `path` - Output file path

### `fn free()`

Explicitly free model resources.

Immediately releases model memory. Optional - the GC will automatically free the model when it's no longer referenced.

## LLM Utility Type

### Static Methods

### `static fn logging(enabled: bool)`

Enable or disable llama.cpp logging.

By default, logging is disabled. Call `LLM::logging(true)` to enable llama.cpp internal logging to stderr.

**Parameters:**
- `enabled` - true to enable logging, false to disable

### `static fn system_info(): String`

Get system information string.

Returns detailed information about the runtime environment: CPU, GPU, SIMD support, backend capabilities, etc.

**Returns:** System information string

**Example:**
```gcl
println(LLM::system_info());
```

### `static fn chat_templates(): Array<String>`

Get list of built-in chat templates.

**Returns:** Names of all chat templates supported by llama.cpp

### `static fn supports_mmap(): bool`

Check if memory mapping (mmap) is supported.

**Returns:** True if the current platform supports mmap

### `static fn supports_mlock(): bool`

Check if memory locking (mlock) is supported.

**Returns:** True if the current platform supports mlock

### `static fn supports_gpu(): bool`

Check if GPU offload is supported.

**Returns:** True if GPU acceleration is available (CUDA, Metal, Vulkan, etc.)

### `static fn max_devices(): int`

Get maximum number of devices.

**Returns:** Maximum number of GPUs that can be used for model distribution

### `static fn max_parallel_sequences(): int`

Get maximum parallel sequences.

**Returns:** Maximum number of sequences that can be processed in parallel

### `static fn supports_rpc(): bool`

Check if RPC is supported.

**Returns:** True if remote procedure call support is available

### `static fn time_us(): int`

Get time in microseconds.

**Returns:** Current time in microseconds since epoch

### `static fn split_path(prefix: String, split_no: int, split_count: int): String`

Build split model file path.

Constructs a split GGUF file path following the standard naming pattern.

**Parameters:**
- `prefix` - Base path
- `split_no` - Split number (1-indexed)
- `split_count` - Total number of splits

**Returns:** Formatted split path

**Example:**
```gcl
var path = LLM::split_path("model", 2, 4);
// Returns: "model-00002-of-00004.gguf"
```

### `static fn split_prefix(split_path: String, split_no: int, split_count: int): String?`

Extract prefix from split model file path.

Extracts the base path from a split file name if split_no and split_count match.

**Parameters:**
- `split_path` - Split file path
- `split_no` - Expected split number
- `split_count` - Expected split count

**Returns:** Base path or null if pattern doesn't match

### `static fn params_fit(model_path: String, mparams: ModelParams, cparams: ContextParams, margin: int, n_ctx_min: int): bool`

Fit model and context parameters to available device memory.

Adjusts ModelParams and ContextParams to fit within device memory constraints. Assumes system memory is unlimited.

**WARNING:** This function is NOT thread-safe as it modifies global logger state.

**Parameters:**
- `model_path` - Path to model file
- `mparams` - Model parameters to adjust (modified in-place)
- `cparams` - Context parameters to adjust (modified in-place)
- `margin` - Memory margin to leave per device (bytes)
- `n_ctx_min` - Minimum context size when reducing memory

**Returns:** True if successful

## Common Use Cases

### Basic Text Generation

```gcl
var model = Model::load("gpt", "./model.gguf", null);
var result = model.generate(
    "Write a haiku about programming:",
    GenerationParams { max_tokens: 50, temperature: 0.7 },
    null
);
println(result.text);
```

### Embedding Similarity

```gcl
var model = Model::load("embedder", "./all-minilm-l6-v2.gguf", null);

var texts = ["cat", "dog", "house"];
var embs = model.embed_batch(texts, TensorType::f32, ContextParams {
    pooling_type: PoolingType::mean,
    normalize: true
});

// Compute cosine similarity between first two embeddings
var dot_product = 0.0;
for (var i = 0; i < embs[0].size(); i++) {
    dot_product += embs[0][i] * embs[1][i];
}
println("Similarity: ${dot_product}");
```

### Chat Assistant

```gcl
var model = Model::load("chat", "./Llama-3.2-1B-Instruct.gguf", null);

var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "What is the capital of France?" }
];

var result = model.chat(messages, GenerationParams { max_tokens: 100 }, null);
println("Assistant: ${result.text}");
```

### GPU-Accelerated Generation

```gcl
var model = Model::load("fast", "./model.gguf", ModelParams {
    n_gpu_layers: -1,         // Offload all layers
    split_mode: SplitMode::layer,
    use_mmap: true,
    use_mlock: false
});

var result = model.generate("Tell me a story", null, ContextParams {
    n_ctx: 4096,
    n_threads: 8,
    offload_kqv: true
});
```

## Best Practices

- **Memory Management**: Models are automatically freed by GC, but you can call `free()` explicitly for immediate cleanup
- **GPU Offload**: Use `n_gpu_layers: -1` to offload all layers to GPU for maximum performance
- **Context Size**: Set `n_ctx` based on your needs - larger contexts use more memory
- **Batch Processing**: Use `embed_batch()` instead of multiple `embed()` calls for better performance
- **Streaming**: Use streaming generation for interactive applications to show results as they're generated
- **Thread Count**: Set `n_threads` to match your CPU core count for optimal performance
- **Temperature**: Lower temperature (0.1-0.5) for factual/deterministic output, higher (0.7-1.0) for creative output
- **Model Selection**: Choose quantized models (q4_k, q5_k) for good balance of quality and speed
- **Reuse Models**: Load models once and reuse them - loading is expensive
- **Check Support**: Use `LLM::supports_gpu()` to check if GPU acceleration is available before enabling it
