# ai

```gcl
@library("ai", "7.5.70-dev");
```

Large Language Model (LLM) integration library powered by llama.cpp. Provides comprehensive support for loading, running, and interacting with GGUF models including text generation, embeddings, chat interfaces, and fine-grained control over sampling and generation parameters.

## Quick Start

```gcl
use ai;

fn main() {
    // Load a model from disk
    var model = Model::load(
        "llama-3.2-1b",  // unique ID
        "./models/Llama-3.2-1B.gguf",
        ModelParams { n_gpu_layers: -1 } // offload all layers to GPU
    );

    // Get model information
    var info = model.info();
    println("Loaded: ${info.description}");
    println("Parameters: ${info.n_params}");
    println("Context: ${info.n_ctx_train} tokens");

    // Generate text
    var result = model.generate(
        "Once upon a time,",
        GenerationParams {
            max_tokens: 100,
            temperature: 0.7
        },
        null
    );

    println(result.text);
    println("Generated ${result.n_tokens} tokens in ${result.perf.context.t_eval_ms}ms");

    // Free model resources
    model.free();
}
```

## Model Loading

### Basic Loading

```gcl
// Load model with default parameters
var model = Model::load("my-model", "./path/to/model.gguf", null);

// Load with GPU acceleration
var model = Model::load(
    "gpu-model",
    "./model.gguf",
    ModelParams {
        n_gpu_layers: -1,  // -1 = all layers to GPU
        use_mmap: true,    // memory-mapped loading
        use_mlock: false   // don't lock in RAM
    }
);

// Load for CPU only
var model = Model::load(
    "cpu-model",
    "./model.gguf",
    ModelParams {
        n_gpu_layers: 0,  // CPU only
        use_mmap: true
    }
);
```

### Split Models

Large models can be split across multiple files:

```gcl
// Standard split naming: model-00001-of-00005.gguf, model-00002-of-00005.gguf, etc.
var model = Model::load("big-model", "./models/model-00001-of-00005.gguf", null);

// Custom split file names
var model = Model::load_from_splits(
    "custom-model",
    [
        "./splits/part1.gguf",
        "./splits/part2.gguf",
        "./splits/part3.gguf"
    ],
    ModelParams { n_gpu_layers: -1 }
);
```

### Multi-GPU Configuration

```gcl
// Distribute across multiple GPUs
var model = Model::load(
    "distributed",
    "./model.gguf",
    ModelParams {
        n_gpu_layers: -1,
        split_mode: SplitMode::layer,  // split layers across GPUs
        main_gpu: 0,                   // primary GPU index
        tensor_split: [0.5, 0.5]       // 50% on GPU 0, 50% on GPU 1
    }
);

// Row-based tensor parallelism (if supported)
var model = Model::load(
    "tensor-parallel",
    "./model.gguf",
    ModelParams {
        n_gpu_layers: -1,
        split_mode: SplitMode::row  // tensor parallelism
    }
);
```

### Model Retrieval

```gcl
// Load once, retrieve later
var model = Model::load("shared-model", "./model.gguf", null);

// Retrieve the same model elsewhere
var same_model = Model::get("shared-model");
if (same_model != null) {
    // Use the already-loaded model
    var result = same_model.generate("Hello", null, null);
}
```

## Text Generation

### Basic Generation

```gcl
// Simple generation
var result = model.generate("Tell me a joke about programming", null, null);
println(result.text);

// With parameters
var result = model.generate(
    "Explain quantum computing",
    GenerationParams {
        max_tokens: 200,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40
    },
    null
);
```

### Streaming Generation

```gcl
// Stream tokens as they're generated
fn on_token(token: String, is_final: bool): bool {
    pprint(token);  // print without newline
    if (is_final) {
        println("");
    }
    return true;  // continue generation
}

var result = model.generate_stream(
    "Write a haiku about databases",
    on_token,
    GenerationParams { max_tokens: 50 },
    null
);

// To stop generation early, return false from callback
fn stop_early(token: String, is_final: bool): bool {
    if (token.contains("ERROR")) {
        return false;  // stop generation
    }
    pprint(token);
    return true;
}
```

### Advanced Sampling

```gcl
// Fine-grained sampler control
var result = model.generate(
    "Continue this story:",
    GenerationParams {
        max_tokens: 150,
        sampler: SamplerParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            penalty: PenaltyParams {
                last_n: 64,
                repeat: 1.1,    // penalize repetition
                freq: 0.0,
                present: 0.0
            },
            dry: DryParams {
                multiplier: 0.8,
                base: 1.75,
                allowed_length: 2,
                penalty_last_n: 256
            },
            seed: 12345  // reproducible results
        }
    },
    null
);
```

### Stop Sequences

```gcl
// Stop generation at specific strings
var result = model.generate(
    "List three fruits:\n1.",
    GenerationParams {
        max_tokens: 100,
        stop_sequences: ["\n4.", "END", "---"]
    },
    null
);
```

### Grammar-Constrained Generation

```gcl
// Force JSON output using GBNF grammar
var json_grammar = "root ::= object\nobject ::= \"{\" ws members ws \"}\"\nmembers ::= member (\",\" ws member)*\nmember ::= string \":\" ws value\nstring ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\"\nvalue ::= string | number | object\nnumber ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\nws ::= [ \\t\\n\\r]*";

var result = model.generate(
    "Generate a person's info as JSON:",
    GenerationParams {
        max_tokens: 200,
        grammar: json_grammar
    },
    null
);
```

## Chat Interface

### Basic Chat

```gcl
// Define conversation
var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "What is machine learning?" }
];

// Generate response
var result = model.chat(messages, null, null);
println("Assistant: ${result.text}");

// Continue conversation
messages.add(ChatMessage { role: "assistant", content: result.text });
messages.add(ChatMessage { role: "user", content: "Can you give an example?" });

var result2 = model.chat(messages, null, null);
```

### Streaming Chat

```gcl
fn on_chat_token(token: String, is_final: bool): bool {
    pprint(token);
    if (is_final) println("");
    return true;
}

var messages = [
    ChatMessage { role: "user", content: "Tell me a short story" }
];

var result = model.chat_stream(messages, on_chat_token, null, null);
```

### Custom Chat Templates

```gcl
// Format messages manually
var formatted = model.format_chat(messages, true);
println("Formatted prompt:");
println(formatted);

// Generate from formatted prompt
var result = model.generate(formatted, null, null);
```

## Embeddings

### Single Text Embedding

```gcl
// Get embedding vector
var embedding = model.embed(
    "Hello, world!",
    TensorType::f32,
    null
);

println("Embedding dimension: ${embedding.size()}");

// Access values
var pos = embedding.initPos();
while (embedding.incPos(pos)) {
    var value = embedding.get(pos);
    // process embedding component
}
```

### Batch Embeddings

```gcl
// Embed multiple texts efficiently
var texts = [
    "The quick brown fox",
    "jumps over the lazy dog",
    "Machine learning is fascinating"
];

var embeddings = model.embed_batch(
    texts,
    TensorType::f32,
    ContextParams {
        pooling_type: PoolingType::mean  // average token embeddings
    }
);

Assert::equals(embeddings.size(), 3);

// Compute cosine similarity
fn cosine_similarity(e1: Tensor, e2: Tensor): float {
    return 1.0 - e1.distance(e2, TensorDistance::cosine);
}

var sim = cosine_similarity(embeddings.get(0), embeddings.get(1));
println("Similarity: ${sim}");
```

### Embedding Configuration

```gcl
// Different pooling strategies
var mean_emb = model.embed(
    text,
    TensorType::f32,
    ContextParams { pooling_type: PoolingType::mean }
);

var cls_emb = model.embed(
    text,
    TensorType::f32,
    ContextParams { pooling_type: PoolingType::cls }
);

var last_emb = model.embed(
    text,
    TensorType::f32,
    ContextParams { pooling_type: PoolingType::last }
);

// Normalized embeddings (unit length)
var norm_emb = model.embed(
    text,
    TensorType::f32,
    ContextParams {
        embeddings: true,
        normalize: true
    }
);
```

## Tokenization

### Basic Tokenization

```gcl
// Text to tokens
var tokens = model.tokenize("Hello, world!", true, false);
println("Tokens: ${tokens}");

// Tokens to text
var text = model.detokenize(tokens, true, false);
println("Text: ${text}");
```

### Token-by-Token

```gcl
// Convert individual tokens
var token_id = 42;
var text = model.token_to_text(token_id);
var piece = model.token_to_piece(token_id, 0, false);

// Get token metadata
var score = model.token_score(token_id);
var attrs = model.token_attr(token_id);
var is_control = model.is_control_token(token_id);
var is_eog = model.is_eog_token(token_id);
```

### Special Tokens

```gcl
// Access special tokens
var bos = model.info().token_bos;  // beginning of sequence
var eos = model.info().token_eos;  // end of sequence
var eot = model.info().token_eot;  // end of turn
var cls = model.token_cls();       // classifier token
var sep = model.token_sep();       // separator
var pad = model.token_pad();       // padding
var nl = model.token_nl();         // newline

// Fill-in-the-middle tokens (for code completion)
var fim_pre = model.token_fim_pre();
var fim_suf = model.token_fim_suf();
var fim_mid = model.token_fim_mid();
```

## Model Information

### Comprehensive Info

```gcl
var info = model.info();

// Architecture
println("Model: ${info.description}");
println("Parameters: ${info.n_params}");
println("Layers: ${info.n_layer}");
println("Embedding dim: ${info.n_embd}");
println("Attention heads: ${info.n_head}");
println("KV heads: ${info.n_head_kv}");

// Context
println("Training context: ${info.n_ctx_train}");
println("Vocabulary: ${info.n_vocab}");
println("Vocab type: ${info.vocab_type}");

// Capabilities
if (info.has_encoder) println("Has encoder");
if (info.has_decoder) println("Has decoder");
if (info.is_recurrent) println("Recurrent model (Mamba/RWKV)");

// Special tokens
println("Add BOS: ${info.add_bos}");
println("Add EOS: ${info.add_eos}");

// Chat template
if (info.chat_template != null) {
    println("Chat template available");
}
```

### Metadata Access

```gcl
// Get specific metadata
var author = model.meta("general.author");
var license = model.meta("general.license");
var name = model.meta("general.name");

// Iterate all metadata
var count = model.meta_count();
for (var i = 0; i < count; i++) {
    var key = model.meta_key_by_index(i);
    var value = model.meta_val_by_index(i);
    println("${key}: ${value}");
}

// Alternative: use info.metadata map
for (key, value in info.metadata) {
    println("${key}: ${value}");
}
```

## Performance

### Monitoring Performance

```gcl
var result = model.generate("Tell me a joke", null, null);

var perf = result.perf;

// Context/generation performance
println("Tokens generated: ${perf.context.n_eval}");
println("Prompt tokens: ${perf.context.n_p_eval}");
println("Generation time: ${perf.context.t_eval_ms}ms");
println("Prompt time: ${perf.context.t_p_eval_ms}ms");
println("Speed: ${perf.context.tokens_per_second} tok/s");

// Sampler performance (if available)
if (perf.sampler != null) {
    println("Samples: ${perf.sampler.n_sample}");
    println("Sampling time: ${perf.sampler.t_sample_ms}ms");
    println("Sampling rate: ${perf.sampler.samples_per_second} samp/s");
}
```

### Memory Usage

```gcl
// Print detailed memory breakdown to logs
model.print_memory();

// Check model size
var info = model.info();
println("Model size: ${info.size} bytes");
```

## Context Configuration

### Basic Context Parameters

```gcl
var result = model.generate(
    "Long prompt here...",
    null,
    ContextParams {
        n_ctx: 4096,         // context window size
        n_batch: 512,        // batch size
        n_threads: 8,        // generation threads
        n_threads_batch: 8   // batch processing threads
    }
);
```

### Advanced Context Configuration

```gcl
// Configure KV cache
var result = model.generate(
    prompt,
    null,
    ContextParams {
        n_ctx: 8192,
        type_k: GgmlType::f16,  // K cache precision
        type_v: GgmlType::f16,  // V cache precision
        defrag_thold: 0.1,      // defrag when 10% fragmented
        offload_kqv: true       // offload KV ops to GPU
    }
);

// RoPE configuration for long context
var result = model.generate(
    prompt,
    null,
    ContextParams {
        n_ctx: 16384,
        rope_scaling_type: RopeScalingType::yarn,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        yarn_ext_factor: 1.0,
        yarn_attn_factor: 1.0,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        yarn_orig_ctx: 8192
    }
);

// Flash Attention
var result = model.generate(
    prompt,
    null,
    ContextParams {
        flash_attn_type: FlashAttnType::enabled_for_all,
        attention_type: AttentionType::causal
    }
);
```

## Model Quantization

### Quantize Models

```gcl
// Convert model to lower precision
Model::quantize(
    "./models/original-f16.gguf",
    "./models/quantized-q4_k_m.gguf",
    QuantizeParams {
        nthread: 8,
        ftype: GgmlType::q4_k,
        allow_requantize: false,
        quantize_output_tensor: true,
        pure: false
    }
);

// Use importance matrix for better quality
Model::quantize(
    "./models/original.gguf",
    "./models/quantized-imatrix.gguf",
    QuantizeParams {
        ftype: GgmlType::q4_k,
        imatrix_file: "./imatrix.dat"
    }
);
```

## Advanced Sampling Algorithms

### Mirostat

```gcl
// Mirostat v2 for controlled perplexity
var result = model.generate(
    "Write a technical document about databases",
    GenerationParams {
        max_tokens: 500,
        sampler: SamplerParams {
            mirostat_v2: MirostatV2Params {
                tau: 5.0,  // target entropy
                eta: 0.1   // learning rate
            }
        }
    },
    null
);
```

### Logit Bias

```gcl
// Bias specific tokens
var token_yes = model.tokenize("Yes", false, false).get(0);
var token_no = model.tokenize("No", false, false).get(0);

var result = model.generate(
    "Is the sky blue?",
    GenerationParams {
        sampler: SamplerParams {
            logit_bias: [
                LogitBias { token: token_yes, bias: 2.0 },   // favor "Yes"
                LogitBias { token: token_no, bias: -2.0 }    // disfavor "No"
            ]
        }
    },
    null
);
```

## Utility Functions

### System Information

```gcl
// Get llama.cpp system info
var sys_info = LLM::system_info();
println(sys_info);

// Check capabilities
var has_gpu = LLM::supports_gpu();
var has_mmap = LLM::supports_mmap();
var has_mlock = LLM::supports_mlock();
var max_gpus = LLM::max_devices();

println("GPU support: ${has_gpu}");
println("Max GPUs: ${max_gpus}");
```

### Logging

```gcl
// Enable llama.cpp logging
LLM::logging(true);

// Load and use model (logs will appear)
var model = Model::load("debug-model", "./model.gguf", null);

// Disable logging
LLM::logging(false);
```

### Chat Templates

```gcl
// List available templates
var templates = LLM::chat_templates();
for (_, template_name in templates) {
    println("Available template: ${template_name}");
}

// Get model's chat template
var template = model.chat_template(null);  // default template
if (template != null) {
    println("Model chat template: ${template}");
}
```

### Memory Fitting

```gcl
// Automatically fit parameters to available memory
var mparams = ModelParams { n_gpu_layers: -1 };
var cparams = ContextParams { n_ctx: 8192 };

var fits = LLM::params_fit(
    "./model.gguf",
    mparams,
    cparams,
    1024 * 1024 * 100,  // 100MB margin
    512                  // minimum context
);

if (fits) {
    println("Adjusted n_ctx: ${cparams.n_ctx}");
    var model = Model::load("fitted", "./model.gguf", mparams);
} else {
    error("Model won't fit in available memory");
}
```

## Complete Example: RAG System

```gcl
use ai;

type Document {
    id: int;
    text: String;
    embedding: node<Tensor>;
}

fn main() {
    // Load embedding model
    var emb_model = Model::load(
        "embedder",
        "./models/bge-base-en-v1.5.gguf",
        ModelParams { n_gpu_layers: -1 }
    );

    // Load generation model
    var gen_model = Model::load(
        "generator",
        "./models/Llama-3.2-3B-Instruct.gguf",
        ModelParams { n_gpu_layers: -1 }
    );

    // Create document embeddings
    var docs = [
        "Paris is the capital of France",
        "Machine learning is a subset of AI",
        "The Earth orbits the Sun"
    ];

    var index = VectorIndex<int> { distance: TensorDistance::cosine };

    for (var i = 0; i < docs.size(); i++) {
        var emb = emb_model.embed(docs.get(i), TensorType::f32, null);
        var emb_node: node<Tensor>;
        emb_node.set(emb);
        index.add(emb_node, i);
    }

    // Query
    var query = "What is the capital city of France?";
    var query_emb = emb_model.embed(query, TensorType::f32, null);
    var results = index.search(query_emb, 2);

    // Build context from top results
    var context = Buffer {};
    context.add("Relevant information:\n");
    for (_, result in results) {
        context.add("- ${docs.get(result.value)}\n");
    }

    // Generate answer
    var messages = [
        ChatMessage {
            role: "system",
            content: "Answer questions based on the provided context."
        },
        ChatMessage {
            role: "user",
            content: "${context.toString()}\n\nQuestion: ${query}"
        }
    ];

    var answer = gen_model.chat(messages, null, null);
    println("Answer: ${answer.text}");

    // Cleanup
    emb_model.free();
    gen_model.free();
}
```

## Type Reference

See the inline documentation in `llm.gcl` for complete type definitions including:

- `ModelParams`, `ContextParams`, `GenerationParams`, `SamplerParams`
- `PenaltyParams`, `DryParams`, `MirostatParams`, `MirostatV2Params`
- `ModelInfo`, `PerfData`, `GenerationResult`
- `ChatMessage`, `TokenData`, `LogitBias`
- Enums: `GgmlType`, `SplitMode`, `PoolingType`, `AttentionType`, `VocabType`, `RopeType`, `RopeScalingType`, `TokenAttr`, `SamplerType`, `StopReason`
