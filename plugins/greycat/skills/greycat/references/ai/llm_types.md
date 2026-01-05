# LLM Types - Supporting Types and Enums

This reference contains all return types, parameter types, and enums that support the Model API. These types represent the comprehensive configuration options and return values from llama.cpp.

## Overview

These types are used throughout the LLM API for:
- Configuring model behavior and performance
- Receiving generation results and metadata
- Controlling sampling and token selection
- Managing model state and quantization

## Enums

### VocabType

Vocabulary type used by the model.

**Values:**
- `none` - No vocabulary
- `spm` - SentencePiece (LLaMA, Mistral, etc.)
- `bpe` - Byte-Pair Encoding (GPT-2, GPT-J, etc.)
- `wpm` - WordPiece (BERT)
- `ugm` - Unigram (T5)
- `rwkv` - RWKV tokenizer
- `plamo2` - Plamo2 tokenizer

### RopeType

RoPE (Rotary Position Embedding) type.

**Values:**
- `none` - No RoPE
- `norm` - Normal RoPE
- `neox` - NeoX-style RoPE

### TokenAttr

Token attributes (bitfield).

**Values:**
- `undefined` - Undefined token
- `unknown` - Unknown token
- `unused` - Unused token
- `normal` - Normal token
- `control` - Control token (BOS, EOS, etc.)
- `user_defined` - User-defined token
- `byte` - Byte token
- `normalized` - Normalized token
- `lstrip` - Left-strip whitespace
- `rstrip` - Right-strip whitespace
- `single_word` - Single-word token

### SamplerType

Sampler type identifier.

**Values:**
- `greedy` - Greedy sampling
- `dist` - Distribution sampling
- `top_k` - Top-K sampling
- `top_p` - Top-P (nucleus) sampling
- `min_p` - Min-P sampling
- `typical` - Typical sampling
- `temp` - Temperature sampling
- `temp_ext` - Temperature with exponent
- `xtc` - XTC sampling
- `top_n_sigma` - Top-N-Sigma sampling
- `mirostat` - Mirostat v1
- `mirostat_v2` - Mirostat v2
- `grammar` - Grammar-based sampling
- `penalties` - Repetition penalties
- `dry` - DRY (Don't Repeat Yourself) sampling
- `logit_bias` - Logit bias
- `infill` - Infill sampling

### StopReason

Generation stop reason.

**Values:**
- `max_tokens` - Maximum tokens reached
- `eog_token` - End-of-generation token produced
- `aborted` - Generation aborted by user/callback
- `error` - Error occurred during generation

## Model Information

### ModelInfo

Comprehensive model information containing detailed metadata about the loaded model.

**Fields:**

**Basic Information:**
- `description: String` - Human-readable model description
- `size: int` - Total model size in bytes
- `n_params: int` - Total number of parameters

**Architecture Dimensions:**
- `n_embd: int` - Embedding dimension
- `n_embd_inp: int` - Input embedding dimension (for encoder-decoder models)
- `n_layer: int` - Number of layers
- `n_head: int` - Number of attention heads
- `n_head_kv: int` - Number of key-value heads (for Grouped Query Attention)
- `n_swa: int` - Sliding window attention size (0 = no sliding window)
- `n_cls_out: int` - Number of classifier output dimensions

**Context and Vocabulary:**
- `n_ctx_train: int` - Context size the model was trained on
- `n_vocab: int` - Vocabulary size
- `vocab_type: VocabType` - Vocabulary type

**RoPE Parameters:**
- `rope_type: RopeType` - RoPE type
- `rope_freq_scale_train: float` - RoPE frequency scaling factor used during training

**Model Capabilities:**
- `has_encoder: bool` - Whether model has encoder
- `has_decoder: bool` - Whether model has decoder
- `is_recurrent: bool` - Whether model is recurrent (Mamba, RWKV)
- `is_hybrid: bool` - Whether model is hybrid (mixed architecture)
- `is_diffusion: bool` - Whether model is diffusion-based

**Special Tokens:**
- `add_bos: bool` - Whether to add BOS token automatically
- `add_eos: bool` - Whether to add EOS token automatically
- `token_bos: int` - BOS (Beginning-of-Sequence) token ID
- `token_eos: int` - EOS (End-of-Sequence) token ID
- `token_eot: int` - EOT (End-of-Turn) token ID

**Chat and Metadata:**
- `chat_template: String?` - Chat template string (if available)
- `metadata: Map<String, String>` - All model metadata key-value pairs

**Example:**
```gcl
var info = model.info();
println("Model: ${info.description}");
println("Parameters: ${info.n_params}");
println("Context: ${info.n_ctx_train}");
println("Vocab: ${info.n_vocab} (${info.vocab_type})");
println("Architecture: ${info.n_layer} layers, ${info.n_head} heads");
println("Capabilities: encoder=${info.has_encoder}, decoder=${info.has_decoder}");
```

## Performance Metrics

### PerfContextData

Context performance data with timing and throughput metrics for inference operations.

**Fields:**
- `n_eval: int` - Number of tokens evaluated
- `n_p_eval: int` - Number of tokens in prompt evaluation
- `t_eval_ms: float` - Total evaluation time (milliseconds)
- `t_p_eval_ms: float` - Prompt evaluation time (milliseconds)
- `tokens_per_second: float` - Tokens per second (generation)
- `prompt_tokens_per_second: float` - Prompt tokens per second

**Example:**
```gcl
var perf = result.perf.context;
println("Generated ${perf.n_eval} tokens in ${perf.t_eval_ms}ms");
println("Speed: ${perf.tokens_per_second} tok/s");
println("Prompt: ${perf.n_p_eval} tokens in ${perf.t_p_eval_ms}ms");
```

### PerfSamplerData

Sampler performance data with metrics for sampling operations.

**Fields:**
- `n_sample: int` - Number of samples taken
- `t_sample_ms: float` - Total sampling time (milliseconds)
- `samples_per_second: float` - Samples per second

### PerfData

Complete performance data combining context and sampler metrics.

**Fields:**
- `context: PerfContextData` - Context/inference performance
- `sampler: PerfSamplerData?` - Sampler performance (if applicable)

## Generation Types

### ChatMessage

Single message in a conversation.

**Fields:**
- `role: String` - Message role (e.g., "system", "user", "assistant")
- `content: String` - Message content

**Example:**
```gcl
var messages = [
    ChatMessage { role: "system", content: "You are a helpful assistant." },
    ChatMessage { role: "user", content: "Hello!" },
    ChatMessage { role: "assistant", content: "Hi! How can I help you?" },
    ChatMessage { role: "user", content: "Tell me a joke" }
];
```

### GenerationResult

Contains the generated text, tokens, and metadata from a generation operation.

**Fields:**
- `text: String` - Generated text
- `tokens: Array<int>` - Generated token IDs
- `n_tokens: int` - Number of tokens generated
- `stop_reason: StopReason` - Why generation stopped
- `perf: PerfData` - Performance metrics

**Example:**
```gcl
var result = model.generate("Once upon a time", params, null);
println("Text: ${result.text}");
println("Tokens: ${result.n_tokens}");
println("Stop reason: ${result.stop_reason}");
println("Speed: ${result.perf.context.tokens_per_second} tok/s");
```

## State Management

### StateData

Saved state from a context (KV cache, etc.) that can be restored later.

**Fields:**
- `data: Buffer` - State data bytes (binary blob)
- `size: int` - State size in bytes

**Example:**
```gcl
var state = ctx.get_state();
println("State size: ${state.size} bytes");

// Later restore
ctx.set_state(state);
```

## Token Data

### TokenData

Single token with probability, used for custom sampling and logit inspection.

**Fields:**
- `id: int` - Token ID
- `logit: float` - Token logit (unnormalized log probability)
- `p: float` - Token probability (after softmax)

### TokenDataBatch

Batch of token data with size information.

**Fields:**
- `data: Array<TokenData>` - Token data array
- `size: int` - Number of tokens
- `sorted: bool` - Whether probabilities are sorted (descending)

## Quantization

### QuantizeParams

Model quantization parameters controlling how models are quantized (converted to lower precision).

**Fields:**
- `nthread: int?` - Number of threads to use (0 = auto)
- `ftype: GgmlType?` - Target quantization format
- `allow_requantize: bool?` - Allow requantizing from a quantized source
- `quantize_output_tensor: bool?` - Quantize output.weight
- `only_copy: bool?` - Only copy tensors, no quantization
- `pure: bool?` - Disable k-quant mixtures and quantize all tensors to the same type
- `imatrix_file: String?` - Path to importance matrix file for improved quantization

**Example:**
```gcl
Model::quantize(
    "./model-f16.gguf",
    "./model-q4.gguf",
    QuantizeParams {
        nthread: 8,
        ftype: GgmlType::q4_k,
        allow_requantize: false,
        pure: false
    }
);
```

## Sampling Parameters

### PenaltyParams

Controls repetition penalties during generation.

**Fields:**
- `last_n: int?` - Number of last tokens to consider for penalties (default: 64)
- `repeat: float?` - Repetition penalty strength (1.0 = disabled, >1.0 = penalize)
- `freq: float?` - Frequency penalty (0.0 = disabled)
- `present: float?` - Presence penalty (0.0 = disabled)

**Example:**
```gcl
var penalty = PenaltyParams {
    last_n: 64,
    repeat: 1.1,
    freq: 0.0,
    present: 0.0
};
```

### DryParams

Advanced repetition penalty that penalizes repeating sequences.

**Fields:**
- `multiplier: float?` - DRY multiplier (0.0 = disabled)
- `base: float?` - DRY base value
- `allowed_length: int?` - Allowed length of repeated sequences
- `penalty_last_n: int?` - Penalty range
- `seq_breakers: Array<String>?` - Sequence breakers (tokens that break repetition detection)

**Example:**
```gcl
var dry = DryParams {
    multiplier: 0.8,
    base: 1.75,
    allowed_length: 2,
    penalty_last_n: 256,
    seq_breakers: ["\n", ".", "!", "?"]
};
```

### MirostatParams

Mirostat v1 parameters.

**Fields:**
- `tau: float?` - Target entropy (τ parameter)
- `eta: float?` - Learning rate (η parameter)
- `m: int?` - Number of candidates to consider

**Example:**
```gcl
var mirostat = MirostatParams {
    tau: 5.0,
    eta: 0.1,
    m: 100
};
```

### MirostatV2Params

Mirostat v2 parameters.

**Fields:**
- `tau: float?` - Target entropy (τ parameter)
- `eta: float?` - Learning rate (η parameter)

**Example:**
```gcl
var mirostat_v2 = MirostatV2Params {
    tau: 5.0,
    eta: 0.1
};
```

### LogitBias

Bias logits for specific tokens (positive = more likely, negative = less likely).

**Fields:**
- `token: int` - Token ID to bias
- `bias: float` - Bias value (-100.0 to 100.0)

**Example:**
```gcl
var biases = [
    LogitBias { token: 100, bias: 5.0 },    // Make token 100 much more likely
    LogitBias { token: 200, bias: -10.0 }   // Suppress token 200
];
```

### SamplerParams

Comprehensive sampler parameters controlling token sampling behavior during generation. All fields are optional and override llama.cpp defaults when specified.

**Basic Sampling:**
- `temperature: float?` - Temperature (0.0 = deterministic, higher = more random)
- `dynatemp_range: float?` - Dynamic temperature range (min, max)
- `dynatemp_exponent: float?` - Dynamic temperature exponent

**Top Sampling Methods:**
- `top_k: int?` - Top-K sampling (0 = disabled, N = keep top N tokens)
- `top_p: float?` - Top-P (nucleus) sampling (0.0-1.0, 1.0 = disabled)
- `min_p: float?` - Min-P sampling (0.0-1.0, 0.0 = disabled)
- `xtc_threshold: float?` - XTC (cross-token correlation) threshold
- `xtc_probability: float?` - XTC probability
- `typical_p: float?` - Typical sampling (0.0-1.0, 1.0 = disabled)

**Penalties:**
- `penalty: PenaltyParams?` - Repetition penalty parameters
- `dry: DryParams?` - DRY sampling parameters

**Advanced Sampling Algorithms:**
- `mirostat: MirostatParams?` - Mirostat v1 parameters
- `mirostat_v2: MirostatV2Params?` - Mirostat v2 parameters

**Constraints:**
- `grammar: String?` - Grammar string (GBNF format) to constrain generation
- `logit_bias: Array<LogitBias>?` - Logit biases for specific tokens

**Random Seed:**
- `seed: int?` - Random seed for sampling (for reproducibility)

**Example:**
```gcl
var sampler = SamplerParams {
    temperature: 0.8,
    top_k: 40,
    top_p: 0.95,
    penalty: PenaltyParams {
        last_n: 64,
        repeat: 1.1
    },
    seed: 12345
};
```

### GenerationParams

High-level parameters controlling text generation.

**Fields:**
- `max_tokens: int?` - Maximum number of tokens to generate
- `sampler: SamplerParams?` - Sampler configuration
- `grammar: String?` - Grammar to constrain generation (GBNF format)
- `stop_sequences: Array<String>?` - Stop sequences (generation stops when any is encountered)
- `temperature: float?` - Temperature override (shortcut for sampler.temperature)
- `top_p: float?` - Top-P override (shortcut for sampler.top_p)
- `top_k: int?` - Top-K override (shortcut for sampler.top_k)

**Example:**
```gcl
// Simple generation
var params = GenerationParams {
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9
};

// Advanced generation with full sampler config
var params = GenerationParams {
    max_tokens: 200,
    sampler: SamplerParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        penalty: PenaltyParams {
            last_n: 64,
            repeat: 1.1,
            freq: 0.0,
            present: 0.0
        },
        seed: 12345
    },
    stop_sequences: ["\n\n", "User:", "Assistant:"]
};
```

## Common Use Cases

### Basic Generation Configuration

```gcl
var params = GenerationParams {
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40
};

var result = model.generate("Once upon a time", params, null);
```

### Creative Writing

```gcl
var creative_params = GenerationParams {
    max_tokens: 500,
    temperature: 0.9,
    top_p: 0.95,
    sampler: SamplerParams {
        penalty: PenaltyParams {
            last_n: 128,
            repeat: 1.2,
            freq: 0.1,
            present: 0.1
        }
    }
};
```

### Factual/Technical Output

```gcl
var factual_params = GenerationParams {
    max_tokens: 200,
    temperature: 0.2,
    top_p: 0.9,
    top_k: 10
};
```

### Structured Output with Grammar

```gcl
var json_grammar = `
root ::= object
object ::= "{" ws members ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
string ::= "\\"" [^"]* "\\""
value ::= string | number | "true" | "false" | "null"
number ::= [0-9]+
ws ::= [ \\t\\n]*
`;

var params = GenerationParams {
    max_tokens: 300,
    grammar: json_grammar,
    temperature: 0.7
};
```

### Controlled Generation with Stop Sequences

```gcl
var params = GenerationParams {
    max_tokens: 500,
    temperature: 0.8,
    stop_sequences: [
        "\nUser:",
        "\nAssistant:",
        "\n\n\n"
    ]
};
```

### Inspecting Generation Results

```gcl
var result = model.generate(prompt, params, null);

println("=== Generation Result ===");
println("Text: ${result.text}");
println("Tokens generated: ${result.n_tokens}");
println("Stop reason: ${result.stop_reason}");
println("");
println("=== Performance ===");
println("Generation speed: ${result.perf.context.tokens_per_second} tok/s");
println("Generation time: ${result.perf.context.t_eval_ms}ms");
println("Prompt tokens: ${result.perf.context.n_p_eval}");
println("Prompt time: ${result.perf.context.t_p_eval_ms}ms");
```

### Chat Conversation Building

```gcl
var messages = [
    ChatMessage {
        role: "system",
        content: "You are a helpful AI assistant. Be concise and accurate."
    },
    ChatMessage {
        role: "user",
        content: "What is the capital of France?"
    }
];

var result = model.chat(messages, GenerationParams { max_tokens: 50 }, null);

// Add assistant response to conversation
messages.push(ChatMessage {
    role: "assistant",
    content: result.text
});

// Continue conversation
messages.push(ChatMessage {
    role: "user",
    content: "What is its population?"
});

result = model.chat(messages, GenerationParams { max_tokens: 50 }, null);
```

## Best Practices

- **Temperature**: Use 0.1-0.3 for factual, 0.7-0.9 for creative, 1.0+ for very random
- **Top-P**: 0.9-0.95 works well for most cases
- **Top-K**: 40-50 is a good default, lower for more focused output
- **Penalties**: Start with repeat penalty of 1.1-1.2, adjust based on repetitiveness
- **Max Tokens**: Set based on expected output length plus buffer
- **Stop Sequences**: Use to control conversation boundaries and formatting
- **Grammar**: Test grammars with simple inputs before production use
- **Seed**: Set seed for reproducible results, omit for varied outputs
- **Performance**: Monitor tokens_per_second to identify bottlenecks
- **Context Size**: Larger n_ctx in ContextParams allows longer conversations but uses more memory
- **Thread Count**: Match n_threads to CPU cores for optimal performance
- **GPU Layers**: Set n_gpu_layers=-1 to offload all layers for maximum speed
