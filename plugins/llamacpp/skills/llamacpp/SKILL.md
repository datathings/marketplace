---
name: llamacpp
description: "Complete llama.cpp C/C++ API reference covering model loading, inference, text generation, embeddings, chat, tokenization, sampling, batching, KV cache, LoRA adapters, and state management. Triggers on: llama.cpp questions, LLM inference code, GGUF models, local AI/ML inference, C/C++ LLM integration, \"how do I use llama.cpp\", API function lookups, implementation questions, troubleshooting llama.cpp issues, and any llama-cpp or ggerganov/llama.cpp mentions."
---

# llama.cpp C API Guide

Comprehensive reference for the llama.cpp C API, documenting all non-deprecated functions and common usage patterns.

## Overview

llama.cpp is a C/C++ implementation for LLM inference with minimal dependencies and state-of-the-art performance. This skill provides:

- **Complete API Reference**: All non-deprecated functions organized by category
- **Common Workflows**: Working examples for typical use cases
- **Best Practices**: Patterns for efficient and correct API usage

## Quick Start

See **[references/workflows.md](references/workflows.md)** for complete working examples. Basic workflow:

1. `llama_backend_init()` - Initialize backend
2. `llama_model_load_from_file()` - Load model
3. `llama_init_from_model()` - Create context
4. `llama_tokenize()` - Convert text to tokens
5. `llama_decode()` - Process tokens
6. `llama_sampler_sample()` - Sample next token
7. Cleanup in reverse order

## When to Use This Skill

Use this skill when:

1. **API Lookup**: You need to find a specific function (e.g., "How do I load a model?", "What function creates a context?")
2. **Code Generation**: You're writing C code that uses llama.cpp
3. **Workflow Guidance**: You need to understand the steps for a task (e.g., text generation, embeddings, chat)
4. **Advanced Features**: You're working with batches, sequences, LoRA adapters, state management, or custom sampling
5. **Migration**: You're updating code from deprecated functions to current API

## Core Concepts

### Key Objects

- **`llama_model`**: Loaded model weights and architecture
- **`llama_context`**: Inference state (KV cache, compute buffers)
- **`llama_batch`**: Input tokens and positions for processing
- **`llama_sampler`**: Token sampling configuration
- **`llama_vocab`**: Vocabulary and tokenizer
- **`llama_memory_t`**: KV cache memory handle

### Typical Flow

1. **Initialize**: `llama_backend_init()`
2. **Load Model**: `llama_model_load_from_file()`
3. **Create Context**: `llama_init_from_model()`
4. **Tokenize**: `llama_tokenize()`
5. **Process**: `llama_encode()` or `llama_decode()`
6. **Sample**: `llama_sampler_sample()`
7. **Generate**: Repeat steps 5-6
8. **Cleanup**: Free in reverse order

## API Reference

For detailed API documentation, the complete API is split across 6 files for efficient targeted loading. Start with **[references/api-core.md](references/api-core.md)** which links to all other sections.

**API Files:**

- **[api-core.md](references/api-core.md)** (304 lines) - Initialization, parameters, model loading, quantization structs
- **[api-model-info.md](references/api-model-info.md)** (223 lines) - Model properties, architecture detection, metadata enums
- **[api-context.md](references/api-context.md)** (421 lines) - Context, memory (KV cache), state management
- **[api-inference.md](references/api-inference.md)** (418 lines) - Batch operations, inference, tokenization, chat
- **[api-sampling.md](references/api-sampling.md)** (490 lines) - All 20+ sampling strategies (incl. adaptive-p) + backend sampling API
- **[api-advanced.md](references/api-advanced.md)** (401 lines) - LoRA adapters, performance, training, constants

**Total:** 198 active functions (b8827) across 6 organized files

### Quick Function Lookup

Most common: `llama_backend_init()`, `llama_model_load_from_file()`, `llama_init_from_model()`, `llama_tokenize()`, `llama_decode()`, `llama_sampler_sample()`, `llama_vocab_is_eog()`, `llama_memory_clear()`

See **[references/api-core.md](references/api-core.md)** for the full API index linking to all function signatures.

## Common Workflows

See **[references/workflows.md](references/workflows.md)** for 13 complete working examples: basic text generation, chat, embeddings, batch processing, multi-sequence, LoRA, state save/load, custom sampling (XTC/DRY), encoder-decoder models, model detection, and memory management patterns.


## Best Practices

See **[references/workflows.md](references/workflows.md)** for detailed best practices. Key points:

- Always use default parameter functions (`llama_model_default_params()`, etc.)
- Check return values for errors
- Free resources in reverse order of creation
- Handle dynamic buffer sizes for tokenization
- Query actual context size after creation (`llama_n_ctx()`)
- Check for end-of-generation with `llama_vocab_is_eog()`

## Common Patterns

End-of-generation check (`llama_vocab_is_eog()`), logits retrieval (`llama_get_logits_ith()`), batch creation (`llama_batch_get_one()`), tokenization buffer handling. See **[references/workflows.md](references/workflows.md)** for complete code examples.

## Troubleshooting

### Common Issues

**Model loading fails:**
- Verify file path and GGUF format validity
- Check available RAM/VRAM for model size
- Reduce `n_gpu_layers` if GPU memory insufficient

**Tokenization returns negative value:**
- Buffer too small; reallocate with `-n` size and retry
- See tokenization pattern in [Common Patterns](#common-patterns)

**Decode/encode returns non-zero:**
- Verify batch initialization (`llama_batch_get_one()` or `llama_batch_init()`)
- Check context capacity (`llama_n_ctx()`)
- Ensure positions within context window

**Silent failures / no output:**
- Check if `llama_vocab_is_eog()` immediately returns true
- Verify sampler initialization
- Enable logging: `llama_log_set()`

**Performance issues:**
- Increase `n_threads` for CPU
- Set `n_gpu_layers` for GPU offloading
- Use larger `n_batch` for prompts
- See [Performance & Utilities](references/api-advanced.md#performance--utilities)

**Sliding Window Attention (SWA) issues:**
- If using Mistral-style models with SWA, set `ctx_params.swa_full = true` to access beyond attention window
- Check: `llama_model_n_swa(model)` to detect SWA size and configuration needs
- Symptoms: Token positions beyond window size causing decode errors

**Per-sequence state errors:**
- Ensure sequence ID matches when loading: `llama_state_seq_load_file(ctx, "file", dest_seq_id, ...)`
- Verify token buffer is large enough for loaded tokens
- Check sequence wasn't cleared or removed before loading state

**Model type detection:**
- Use `llama_model_has_encoder()` before assuming decoder-only architecture
- For recurrent models (Mamba/RWKV), KV cache behavior differs from standard transformers
- Encoder-decoder models require `llama_encode()` then `llama_decode()` workflow

For advanced issues: https://github.com/ggerganov/llama.cpp/discussions

## Resources

- **API Reference** (6 files, 2,257 lines total) - Complete API reference split by category for targeted loading:
  - [api-core.md](references/api-core.md) - Initialization, parameters, model loading, quantization structs
  - [api-model-info.md](references/api-model-info.md) - Model properties, architecture detection, metadata enums
  - [api-context.md](references/api-context.md) - Context, memory, state management
  - [api-inference.md](references/api-inference.md) - Batch, inference, tokenization, chat
  - [api-sampling.md](references/api-sampling.md) - All 20+ sampling strategies (incl. adaptive-p) + backend sampling API
  - [api-advanced.md](references/api-advanced.md) - LoRA, performance, training, constants
- **[references/workflows.md](references/workflows.md)** (1,613 lines) - 15 complete working examples: basic workflows (text generation, chat, embeddings, batching, sequences), intermediate (LoRA, state, sampling, encoder-decoder, memory), advanced features (XTC/DRY, per-sequence state, model detection), and production applications (interactive chat, streaming).

## What's New in b8827

**b8827** is a maintenance release (from b8809) focused on OpenCL/Hexagon optimizations, model refactors, and internal improvements. No public C API changes -- all 198 active functions retain the same signatures.

**Internal Improvements (no API impact):**
- OpenCL: refactored q8_0 set_tensor and mul_mat host side dispatch for Adreno
- Hexagon: optimized HMX matmul operations
- Model layer: unified single `llm_build` per architecture
- CMake: switched to glob for src/models sources
- CLI: use `get_media_marker` for media handling

**Stable Since b8809:**
- Model loading: `llama_model_load_from_file_ptr()`, `llama_model_init_from_user()`
- Quantization types: MXFP4_MOE (38), NVFP4 (39), Q1_0 (40)
- Split mode: `LLAMA_SPLIT_MODE_TENSOR` (3) for backend-agnostic tensor parallelism
- Backend sampling API (EXPERIMENTAL): GPU-accelerated sampling via context params
- Adaptive-P sampler: `llama_sampler_init_adaptive_p()`

## Key Differences from Deprecated API

If you're updating old code:

- Use `llama_model_load_from_file()` instead of `llama_load_model_from_file()`
- Use `llama_model_free()` instead of `llama_free_model()`
- Use `llama_init_from_model()` instead of `llama_new_context_with_model()`
- Use `llama_vocab_*()` functions instead of `llama_token_*()`
- Use `llama_state_*()` functions instead of deprecated state functions
- Use `llama_set_adapters_lora()` instead of `llama_set_adapter_lora()` for LoRA adapters
- Use `llama_vocab_bos()` instead of `llama_vocab_cls()` (CLS is equivalent to BOS)
- Use `llama_sampler_init_grammar_lazy_patterns()` instead of `llama_sampler_init_grammar_lazy()`

See the API reference for complete mappings.
