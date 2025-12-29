# AI Library (LLM Integration)

The AI library provides comprehensive integration with llama.cpp for running Large Language Models (LLMs) in GreyCat applications.

## Overview

Built on llama.cpp, this library enables:
- **Model Loading**: Load and manage GGUF format models with GPU acceleration
- **Text Generation**: Generate text completions with advanced sampling strategies
- **Chat**: Multi-turn conversations with chat templates
- **Embeddings**: Generate dense vector representations of text
- **Tokenization**: Convert between text and tokens
- **Advanced Control**: Low-level access to contexts, KV cache, and sampling chains
- **LoRA Adapters**: Fine-tune models with Low-Rank Adaptation

## Quick Start

```gcl
@library("ai", "7.5.51-dev");

// Load model with GPU acceleration
var model = Model::load("my_model", "./Llama-3.2-1B.gguf", ModelParams {
    n_gpu_layers: -1  // Use all GPU layers
});

// Generate text
var result = model.generate(
    "Once upon a time",
    GenerationParams {
        max_tokens: 100,
        temperature: 0.7
    },
    null
);

println(result.text);
model.free();
```

## Library Files

### Core Types

- **[llm_model.md](llm_model.md)** — Model loading, generation, chat, embeddings, tokenization
  - `Model` — Main LLM model type with 40+ methods
  - `LLM` — System utilities and information
  - `ModelParams` — Model loading configuration
  - `ContextParams` — Inference context configuration

- **[llm_types.md](llm_types.md)** — Supporting types and parameters
  - `GenerationParams` — High-level generation control
  - `GenerationResult` — Generation output and metadata
  - `ChatMessage` — Chat conversation messages
  - `SamplerParams` — Detailed sampling configuration
  - `ModelInfo` — Model metadata and capabilities
  - `PerfData` — Performance metrics

### Advanced Control

- **[llm_context.md](llm_context.md)** — Low-level context and KV cache management
  - `Context` — Advanced inference context (53 methods)
  - `Batch` — Token batching for parallel processing
  - `SeqId` — Sequence identification for multi-turn conversations
  - `StateData` — State save/restore

- **[llm_sampler.md](llm_sampler.md)** — Custom sampling strategies
  - `SamplerChain` — Compose multiple samplers
  - `Sampler` — Individual sampling strategies (26+ types)
  - `TokenCandidates` — Inspect and filter token candidates

- **[llm_lora.md](llm_lora.md)** — LoRA adapter support
  - `LoraAdapter` — Load and apply fine-tuning adapters
  - `LoraParams` — LoRA configuration

## Common Use Cases

### Text Generation

```gcl
var result = model.generate("Explain quantum computing", GenerationParams {
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.95
}, null);

println(result.text);
println("Generated ${result.n_tokens} tokens");
println("Stop reason: ${result.stop_reason}");
```

### Chat Completion

```gcl
var messages = Array<ChatMessage>{};
messages.add(ChatMessage { role: "system", content: "You are a helpful assistant." });
messages.add(ChatMessage { role: "user", content: "What is the capital of France?" });

var response = model.chat(messages, GenerationParams {
    max_tokens: 100,
    temperature: 0.7
}, null);

println(response.text);
```

### Streaming Generation

```gcl
fn on_token(token: String, is_final: bool): bool {
    print(token);
    return true;  // Continue generation
}

model.generate_stream("Tell me a story", on_token, GenerationParams {
    max_tokens: 500
}, null);
```

### Embeddings

```gcl
var embedding = model.embed("Hello world", TensorType::f32, null);
println("Embedding dimension: ${embedding.size()}");

// Batch embeddings
var texts = Array<String>{ "Text 1", "Text 2", "Text 3" };
var embeddings = model.embed_batch(texts, TensorType::f32, null);
```

### Multi-turn Conversations with Context

```gcl
var ctx = Context::create(model, ContextParams { n_ctx: 2048 });

// First turn
var tokens1 = model.tokenize("User: Hello!", true, false);
var batch1 = Batch::from_array(tokens1, 0, 0);
ctx.decode(batch1);

// Sample response tokens...
// (See llm_context.md for detailed multi-turn examples)

ctx.free();
```

### LoRA Fine-tuning

```gcl
var lora = LoraAdapter::load(model, "./medical-lora.gguf", 1.0, null);
var ctx = Context::create(model, null);

// Apply adapter
ctx.apply_lora_adapter(lora, 1.0);

// Use context with fine-tuned behavior...

ctx.remove_lora_adapter(lora);
```

## Performance Tips

1. **GPU Acceleration**: Use `n_gpu_layers: -1` to offload all layers to GPU
2. **Memory Mapping**: Enable `use_mmap: true` for faster loading
3. **Batch Size**: Adjust `n_batch` based on available VRAM
4. **Context Size**: Use smallest `n_ctx` that fits your use case
5. **Quantization**: Use quantized models (Q4_K_M, Q5_K_M) for better performance

## System Information

```gcl
println(LLM::system_info());  // CPU, GPU, SIMD support
println("GPU support: ${LLM::supports_gpu()}");
println("Max devices: ${LLM::max_devices()}");
```

## Documentation Structure

Each file provides:
- Type and enum definitions
- Method signatures with parameters
- Usage examples
- Best practices
- Performance considerations

Refer to individual .md files for detailed API documentation.
