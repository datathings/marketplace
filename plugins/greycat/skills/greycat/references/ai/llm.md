# LLM - Local Language Model API

GreyCat's LLM API provides comprehensive access to llama.cpp for local language model inference. Load GGUF models, generate text, compute embeddings, and build chat applications.

## Quick Start

```gcl
@library("ai", "7.7.120-dev");

// Load a model
var model = Model::load("llama3", "./Llama-3.2-1B.gguf", ModelParams { n_gpu_layers: -1 });

// Get model info
var info = model.info();
println("Loaded ${info.description} with ${info.n_params} parameters");

// Generate text
var result = model.generate("Once upon a time", GenerationParams { max_tokens: 100 }, null);
println(result.text);

// Chat completion
var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "What is the capital of France?" }
];
var response = model.chat(messages, GenerationParams { max_tokens: 100 }, null);
println(response.text);

// Generate embeddings
var emb = model.embed("Hello world", TensorType::f32, null);
println("Embedding dimension: ${emb.size()}");

// Clean up (optional - GC handles this)
model.free();
```

## Model API

### Loading Models

```gcl
// Basic loading
var model = Model::load("name", "./model.gguf", null);

// With GPU offload
var model = Model::load("name", "./model.gguf", ModelParams {
    n_gpu_layers: -1,  // All layers to GPU
    use_mmap: true,
    use_mlock: false
});

// Retrieve loaded model by name
var model = Model::get("name");

// Load split model files
var model = Model::load_from_splits("name", [
    "./model-00001-of-00003.gguf",
    "./model-00002-of-00003.gguf",
    "./model-00003-of-00003.gguf"
], null);
```

**ModelParams Fields:**
- `n_gpu_layers: int?` - Layers to GPU (0 = CPU, -1 = all)
- `split_mode: SplitMode?` - Multi-GPU split mode (none, layer, row)
- `main_gpu: int?` - GPU index when split_mode is none
- `tensor_split: Array<float>?` - Proportion of tensor rows per GPU (for split_mode layer/row)
- `vocab_only: bool?` - Load vocabulary only (for tokenization)
- `use_mmap: bool?` - Use memory-mapped files
- `use_mlock: bool?` - Prevent swapping
- `check_tensors: bool?` - Validate tensor data during loading

### Text Generation

```gcl
// Basic generation
var result = model.generate("Once upon a time", null, null);
println(result.text);

// With parameters
var result = model.generate(
    "Write a story:",
    GenerationParams {
        max_tokens: 200,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40
    },
    ContextParams { n_ctx: 2048 }
);

// Streaming generation
model.generate_stream(
    "Tell me a joke:",
    fn(token: String, is_final: bool): bool {
        print(token);  // Print each token
        return true;   // Return false to stop
    },
    GenerationParams { max_tokens: 100 },
    null
);
```

**GenerationParams Fields:**
- `max_tokens: int?` - Maximum tokens to generate
- `temperature: float?` - Randomness (0.0 = deterministic, 0.7-1.0 = creative)
- `top_p: float?` - Nucleus sampling threshold (0.9-0.95 typical)
- `top_k: int?` - Top-K sampling (40-50 typical)
- `stop_sequences: Array<String>?` - Stop generation on these strings
- `grammar: String?` - GBNF grammar constraint
- `sampler: SamplerParams?` - Advanced sampler configuration

### Chat Completion

```gcl
var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "Hello!" }
];

// Basic chat
var result = model.chat(messages, GenerationParams { max_tokens: 100 }, null);
println(result.text);

// Streaming chat
model.chat_stream(
    messages,
    fn(token: String, is_final: bool): bool {
        print(token);
        return true;
    },
    null, null
);

// Manual formatting
var prompt = model.format_chat(messages, true);  // add_assistant = true
```

### Embeddings

```gcl
// Single text
var emb = model.embed("Hello world", TensorType::f32, ContextParams {
    pooling_type: PoolingType::mean,
    normalize: true
});

// Batch processing (more efficient)
var texts = ["Hello", "World", "Test"];
var embeddings = model.embed_batch(texts, TensorType::f32, null);

// Compute similarity
fn cosine_similarity(a: Tensor, b: Tensor): float {
    var dot = 0.0;
    for (var i = 0; i < a.size(); i++) {
        dot = dot + (a[i] * b[i]);
    }
    return dot;  // Assumes normalized embeddings
}
```

### Tokenization

```gcl
// Tokenize
var tokens = model.tokenize("Hello world", true, false);
println("Token count: ${tokens.size()}");

// Detokenize
var text = model.detokenize(tokens, true, false);

// Single token operations
var piece = model.token_to_text(token_id);
var score = model.token_score(token_id);
var attr = model.token_attr(token_id); // TokenAttr bitfield
var is_eog = model.is_eog_token(token_id);
var is_ctrl = model.is_control_token(token_id);

// Special tokens
var sep = model.token_sep();
var pad = model.token_pad();
var mask = model.token_mask();
var nl = model.token_nl();
var cls = model.token_cls(); // -1 if not available
var adds_sep = model.add_sep_token();

// Fill-in-the-middle (FIM) tokens
var fim_pre = model.token_fim_pre();
var fim_suf = model.token_fim_suf();
var fim_mid = model.token_fim_mid();
```

### Model Information

```gcl
var info = model.info();
println("Description: ${info.description}");
println("Parameters: ${info.n_params}");
println("Context size: ${info.n_ctx_train}");
println("Vocab size: ${info.n_vocab}");
println("Has encoder: ${info.has_encoder}");

// Model description shorthand
println(model.desc());

// Get metadata by key
var name = model.meta("general.name");
var author = model.meta("general.author");

// Iterate all metadata
var count = model.meta_count();
for (var i = 0; i < count; i++) {
    var key = model.meta_key_by_index(i);
    var val = model.meta_val_by_index(i);
    println("${key}: ${val}");
}

// Get chat template
var template = model.chat_template(null);

// For classifier models: get label by output index
var label = model.cls_label(0);

// For encoder-decoder models: get decoder start token
var dec_start = model.decoder_start_token(); // -1 if not applicable
```

### Model Quantization

```gcl
// Quantize a model to a smaller format (no Model instance needed)
Model::quantize("./model-f16.gguf", "./model-q4_k.gguf", QuantizeParams {
    ftype: GgmlType::q4_k,
    nthread: 4
});
```

### Model State & Performance

```gcl
// Save model state to file
model.save("./model-checkpoint.gguf");

// Get performance data from last operation
var perf = model.perf();
println("Speed: ${perf.context.tokens_per_second} tok/s");

// Print memory usage breakdown to log
model.print_memory();

// Convert a token ID to text with full control
var piece = model.token_to_piece(42, 0, true);
```

### Performance Metrics

```gcl
var result = model.generate(prompt, params, null);

var perf = result.perf.context;
println("Generated ${perf.n_eval} tokens in ${perf.t_eval_ms}ms");
println("Speed: ${perf.tokens_per_second} tok/s");
println("Prompt: ${perf.n_p_eval} tokens at ${perf.prompt_tokens_per_second} tok/s");
```

## Advanced: Context API

For low-level control over KV cache, sequences, and batching. Most users should use the high-level Model API.

```gcl
var model = Model::load("model", "./model.gguf", null);
var ctx = Context::create(model, ContextParams { n_ctx: 4096 });

// Tokenize and decode
var tokens = model.tokenize("Hello world", true, false);
var batch = Batch::from_array(tokens, 0, 0);
ctx.decode(batch);

// Get logits
var logits = ctx.get_logits(-1);

// KV cache management
ctx.kv_cache_clear();
ctx.kv_cache_seq_rm(SeqId { id: 0 }, 0, 100);  // Remove tokens 0-100
ctx.kv_cache_seq_cp(SeqId { id: 0 }, SeqId { id: 1 }, 0, -1);  // Copy sequence

// State save/restore
ctx.save_state("./checkpoint.state");
ctx.load_state("./checkpoint.state");

ctx.free();
```

**Context Methods:**
- `decode(batch)` / `encode(batch)` - Process tokens
- `get_logits(i)` / `get_embeddings(i)` - Get outputs
- `kv_cache_clear()` / `kv_cache_seq_rm()` - Cache management
- `get_state()` / `set_state()` - State persistence
- `apply_lora_adapter()` / `remove_lora_adapter()` - LoRA management

## Advanced: Sampler API

For custom sampling chains. Most users should use GenerationParams instead.

```gcl
// Create sampler chain
var chain = SamplerChain::create(null);
chain.add(Sampler::penalties(64, 1.1, 0.0, 0.0));
chain.add(Sampler::top_k(40));
chain.add(Sampler::top_p(0.95, 1));
chain.add(Sampler::temp(0.8));
chain.add(Sampler::dist(12345));

// Sample from context
var token = chain.sample(ctx, -1);
chain.accept(token);

chain.free();
```

**Available Samplers:**
- `Sampler::greedy()` - Always select highest probability
- `Sampler::dist(seed)` - Sample from distribution
- `Sampler::top_k(k)` - Keep top K tokens
- `Sampler::top_p(p, min_keep)` - Nucleus sampling
- `Sampler::min_p(p, min_keep)` - Min-P sampling
- `Sampler::temp(t)` - Temperature scaling
- `Sampler::penalties(last_n, repeat, freq, present)` - Repetition penalties
- `Sampler::grammar(model, grammar, root)` - Grammar constraint
- `Sampler::mirostat_v2(seed, tau, eta)` - Mirostat v2

## LoRA Adapters

Apply fine-tuning adapters without modifying base weights.

```gcl
var model = Model::load("base", "./base-model.gguf", null);
var ctx = Context::create(model, null);

// Load and apply adapter
var lora = LoraAdapter::load(model, "./medical-lora.gguf", 1.0, null);
ctx.apply_lora_adapter(lora, 1.0);

// Generate with adapter
var result = model.generate("Symptoms of diabetes:", null, null);

// Remove or switch adapters
ctx.remove_lora_adapter(lora);
ctx.clear_lora_adapters();

// Multiple adapters
var code_lora = LoraAdapter::load(model, "./code-lora.gguf", 1.0, null);
var math_lora = LoraAdapter::load(model, "./math-lora.gguf", 1.0, null);
ctx.apply_lora_adapter(code_lora, 1.0);
ctx.apply_lora_adapter(math_lora, 0.5);  // 50% strength
```

## LLM Utility Functions

```gcl
// System info
println(LLM::system_info());

// Check capabilities
if (LLM::supports_gpu()) {
    println("GPU acceleration available");
}
println("Max devices: ${LLM::max_devices()}");
println("Supports mmap: ${LLM::supports_mmap()}");
println("Supports mlock: ${LLM::supports_mlock()}");
println("Max parallel sequences: ${LLM::max_parallel_sequences()}");

// Enable logging
LLM::logging(true);

// Built-in chat templates
var templates = LLM::chat_builtin_templates();

// Check if model fits in device memory
var mparams = ModelParams { n_gpu_layers: -1 };
var cparams = ContextParams { n_ctx: 4096 };
if (LLM::params_fit("./model.gguf", mparams, cparams, 512_000_000, 512)) {
    println("Model fits in device memory");
}

// Additional utility functions
println("Supports RPC: ${LLM::supports_rpc()}");
println("Time (us): ${LLM::time_us()}");
println("Max tensor buft overrides: ${LLM::max_tensor_buft_overrides()}");
var fa_name = LLM::flash_attn_type_name(FlashAttnType::enabled_for_fa);

// Build split model file paths
var split = LLM::split_path("model", 2, 4); // "model-00002-of-00004.gguf"
var prefix = LLM::split_prefix("model-00002-of-00004.gguf", 2, 4); // extract prefix
```

## Types Reference

### Enums

**SplitMode** - GPU distribution: `none`, `layer`, `row`

**PoolingType** - Embedding pooling: `unspecified`, `none`, `mean`, `cls`, `last`, `rank`

**GgmlType** - Quantization formats: `f32`, `f16`, `bf16`, `q4_0`, `q4_k`, `q5_k`, `q6_k`, `q8_0`, etc.

**StopReason** - Generation stop: `max_tokens`, `eog_token`, `aborted`, `error`

**SamplerType** - Sampler identifiers: `greedy`, `dist`, `top_k`, `top_p`, `min_p`, `typical`, `temp`, `temp_ext`, `xtc`, `top_n_sigma`, `mirostat`, `mirostat_v2`, `grammar`, `penalties`, `dry`, `logit_bias`, `infill`

**FlashAttnType** - Flash attention mode: `disabled`, `enabled_for_fa` (only for FA-native models), `enabled_for_all`

**AttentionType** - Attention mechanism: `unspecified`, `causal`, `non_causal`

**RopeScalingType** - RoPE scaling: `unspecified`, `none`, `linear`, `yarn`, `longrope`

**VocabType** - Vocabulary format: `none`, `spm` (SentencePiece), `bpe` (Byte-Pair Encoding), `wpm` (WordPiece), `ugm` (Unigram), `rwkv`, `plamo2`

**RopeType** - RoPE type: `none`, `norm` (normal), `neox` (NeoX-style)

**TokenAttr** - Token attributes: `undefined`, `unknown`, `unused`, `normal`, `control`, `user_defined`, `byte`, `normalized`, `lstrip`, `rstrip`, `single_word`

### Result Types

**GenerationResult:**
```gcl
type GenerationResult {
    text: String;              // Generated text
    tokens: Array<int>;        // Token IDs
    n_tokens: int;             // Token count
    stop_reason: StopReason;   // Why generation stopped
    perf: PerfData;            // Performance metrics
}
```

**ModelInfo:**
```gcl
type ModelInfo {
    description: String;       // Model description
    size: int;                 // Total model size in bytes
    n_params: int;             // Parameter count
    // Architecture dimensions
    n_embd: int;               // Embedding dimension
    n_embd_inp: int;           // Input embedding dimension (encoder-decoder)
    n_layer: int;              // Layer count
    n_head: int;               // Attention heads
    n_head_kv: int;            // Key-value heads (GQA)
    n_swa: int;                // Sliding window attention size (0 = none)
    n_cls_out: int;            // Classifier output dimensions
    // Context and vocabulary
    n_ctx_train: int;          // Training context size
    n_vocab: int;              // Vocabulary size
    vocab_type: VocabType;     // Vocabulary format
    // RoPE parameters
    rope_type: RopeType;       // RoPE type (none, norm, neox)
    rope_freq_scale_train: float; // RoPE frequency scaling from training
    // Model capabilities
    has_encoder: bool;         // Encoder-decoder model
    has_decoder: bool;
    is_recurrent: bool;        // Recurrent model (Mamba, RWKV)
    is_hybrid: bool;           // Hybrid architecture
    is_diffusion: bool;        // Diffusion-based model
    // Special tokens
    add_bos: bool;             // Add BOS token automatically
    add_eos: bool;             // Add EOS token automatically
    token_bos: int;            // BOS token ID
    token_eos: int;            // EOS token ID
    token_eot: int;            // EOT (End-of-Turn) token ID
    // Chat and metadata
    chat_template: String?;    // Chat template (if available)
    metadata: Map<String, String>; // All model metadata
}
```

**ChatMessage:**
```gcl
type ChatMessage {
    role: String;     // "system", "user", "assistant"
    content: String;  // Message content
}
```

### Context Parameters

**ContextParams** - Full control over inference context:
```gcl
type ContextParams {
    // Core
    n_ctx: int?;                       // Context size (0 = from model)
    n_batch: int?;                     // Logical max batch size
    n_ubatch: int?;                    // Physical max batch size
    n_seq_max: int?;                   // Max sequences for recurrent models
    // Threading
    n_threads: int?;                   // Threads for generation
    n_threads_batch: int?;             // Threads for batch processing
    // Attention & Pooling
    rope_scaling_type: RopeScalingType?;
    pooling_type: PoolingType?;        // Embedding pooling strategy
    attention_type: AttentionType?;    // Attention mechanism
    flash_attn_type: FlashAttnType?;   // Flash attention mode
    // RoPE
    rope_freq_base: float?;            // RoPE base frequency
    rope_freq_scale: float?;           // RoPE frequency scaling factor
    // YaRN (RoPE scaling)
    yarn_ext_factor: float?;
    yarn_attn_factor: float?;
    yarn_beta_fast: float?;
    yarn_beta_slow: float?;
    yarn_orig_ctx: int?;
    // KV Cache
    defrag_thold: float?;              // KV cache defrag threshold
    type_k: GgmlType?;                // K cache data type
    type_v: GgmlType?;                // V cache data type
    // Feature Flags
    embeddings: bool?;                 // Extract embeddings with logits
    normalize: bool?;                  // Normalize embeddings (L2 norm)
    offload_kqv: bool?;               // Offload KQV ops to GPU
    no_perf: bool?;                    // Disable performance timings
    op_offload: bool?;                 // Offload host tensor ops to device
    swa_full: bool?;                   // Full-size SWA cache
    kv_unified: bool?;                 // Unified buffer for attention
}
```

### Advanced Sampler Parameters

**SamplerParams** - Full control over sampling:
```gcl
type SamplerParams {
    temperature: float?;       // Randomness (0.0-2.0)
    dynatemp_range: float?;    // Dynamic temperature range
    dynatemp_exponent: float?; // Dynamic temperature exponent
    top_k: int?;               // Top-K filter
    top_p: float?;             // Nucleus sampling
    min_p: float?;             // Min-P filter
    xtc_threshold: float?;     // XTC sampling threshold
    xtc_probability: float?;   // XTC sampling probability
    typical_p: float?;         // Typical sampling
    penalty: PenaltyParams?;   // Repetition penalties
    dry: DryParams?;           // DRY sampling
    mirostat: MirostatParams?; // Mirostat v1 parameters
    mirostat_v2: MirostatV2Params?;
    grammar: String?;          // GBNF grammar
    logit_bias: Array<LogitBias>?;
    seed: int?;                // Random seed
}
```

**QuantizeParams** - Model quantization configuration:
```gcl
type QuantizeParams {
    nthread: int?;             // Threads (0 = auto)
    ftype: GgmlType?;         // Target quantization format
    allow_requantize: bool?;   // Allow requantizing from quantized source
    quantize_output_tensor: bool?; // Quantize output.weight
    only_copy: bool?;          // Copy tensors without quantization
    pure: bool?;               // Disable k-quant mixtures
    imatrix_file: String?;     // Importance matrix file path
}
```

**MirostatParams** - Mirostat v1 sampling:
```gcl
type MirostatParams {
    tau: float?;   // Target entropy
    eta: float?;   // Learning rate
    m: int?;       // Number of candidates
}
```

**TokenData** - Single token with probability:
```gcl
type TokenData {
    id: int;       // Token ID
    logit: float;  // Unnormalized log probability
    p: float;      // Probability (after softmax)
}
```

## Common Patterns

### Factual/Technical Output

```gcl
var params = GenerationParams {
    max_tokens: 200,
    temperature: 0.2,
    top_p: 0.9,
    top_k: 10
};
```

### Creative Writing

```gcl
var params = GenerationParams {
    max_tokens: 500,
    temperature: 0.9,
    top_p: 0.95,
    sampler: SamplerParams {
        penalty: PenaltyParams {
            last_n: 128,
            repeat: 1.2,
            freq: 0.1
        }
    }
};
```

### JSON Output with Grammar

```gcl
var json_grammar = `
root ::= object
object ::= "{" ws members ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
string ::= "\"" [^"]* "\""
value ::= string | number | "true" | "false" | "null"
number ::= [0-9]+
ws ::= [ \t\n]*
`;

var params = GenerationParams {
    max_tokens: 300,
    grammar: json_grammar,
    temperature: 0.7
};
```

### Embedding Similarity Search

```gcl
var model = Model::load("embedder", "./all-minilm-l6-v2.gguf", null);

// Index documents
var docs = ["Document 1...", "Document 2...", "Document 3..."];
var doc_embs = model.embed_batch(docs, TensorType::f32, ContextParams {
    pooling_type: PoolingType::mean,
    normalize: true
});

// Query
var query_emb = model.embed("search query", TensorType::f32, ContextParams {
    pooling_type: PoolingType::mean,
    normalize: true
});

// Find most similar
var best_idx = 0;
var best_score = -1.0;
for (var i = 0; i < doc_embs.size(); i++) {
    var score = cosine_similarity(query_emb, doc_embs[i]);
    if (score > best_score) {
        best_score = score;
        best_idx = i;
    }
}
println("Best match: ${docs[best_idx]}");
```

## Best Practices

- **GPU Offload**: Use `n_gpu_layers: -1` for maximum performance
- **Reuse Models**: Load once, reuse - loading is expensive
- **Batch Embeddings**: Use `embed_batch()` instead of multiple `embed()` calls
- **Temperature**: 0.1-0.3 for factual, 0.7-0.9 for creative
- **Context Size**: Set `n_ctx` based on needs - larger uses more memory
- **Thread Count**: Match `n_threads` to CPU cores
- **Stop Sequences**: Use to control output boundaries
- **Check GPU**: Use `LLM::supports_gpu()` before enabling GPU offload
- **Memory**: Call `free()` explicitly for immediate cleanup
- **Quantization**: Use q4_k or q5_k for balance of quality and speed
