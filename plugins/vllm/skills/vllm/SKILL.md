---
name: vllm
description: "Complete vLLM v0.19.0 Python API reference for high-throughput LLM inference: offline batch generation, chat, embeddings, classification, structured outputs, LoRA adapters, multimodal inputs, and OpenAI-compatible server. Triggers on: vLLM questions, Python LLM serving, GPU inference, \"how do I use vllm\", batch inference, vllm serve, OpenAI-compatible API, structured JSON output, LoRA serving."
---

# vLLM v0.19.0 Python API Guide

Comprehensive reference for vLLM -- a high-throughput, memory-efficient inference engine for large language models.

## Overview

vLLM provides two main interfaces:

1. **Offline inference** via the `LLM` class -- batch processing with automatic memory management
2. **Online serving** via `vllm serve` -- OpenAI-compatible REST API with streaming

Key capabilities:
- High-throughput batched generation with PagedAttention
- OpenAI-compatible chat completions, completions, and embeddings API
- Structured outputs (JSON schema, regex, grammar, choice)
- LoRA adapter hot-swapping
- Multimodal inputs (images, audio, video)
- Tensor, pipeline, and data parallelism
- Quantization (AWQ, GPTQ, FP8, and more)
- Prefix caching for shared prompt prefixes
- Tool calling / function calling

## Quick Start

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.8, max_tokens=256)

# Text completion (no chat template)
outputs = llm.generate(["The future of AI is"], params)
print(outputs[0].outputs[0].text)

# Chat completion (applies chat template)
outputs = llm.chat(
    [{"role": "user", "content": "What is vLLM?"}],
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
```

## When to Use This Skill

- Writing Python code that uses vLLM for inference
- Configuring `vllm serve` for production deployment
- Using structured outputs (JSON, regex, grammar)
- Setting up LoRA adapter serving
- Passing multimodal inputs (images, audio, video)
- Tuning sampling parameters
- Understanding output types and result handling
- Migrating from vLLM v0.16.x to v0.19.0

## Core Concepts

### LLM Class

The main offline inference API. Created with a model name/path, automatically manages GPU memory and batching.

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

### SamplingParams

Controls generation behavior: temperature, top-p, max tokens, stop sequences, structured outputs, etc.

```python
from vllm import SamplingParams
params = SamplingParams(temperature=0.7, max_tokens=256, top_p=0.95)
```

### generate() vs chat()

**Critical distinction:**
- `generate()` takes raw text prompts. Does NOT apply chat templates.
- `chat()` takes message lists (role/content dicts). Applies the model's chat template automatically.

Use `chat()` for conversational interactions. Use `generate()` for raw text completion.

### generation_config.json

vLLM reads `generation_config.json` from HuggingFace models by default, which may override `SamplingParams` defaults. To disable: pass `generation_config="vllm"` to `LLM()` or `--generation-config vllm` to `vllm serve`.

### StructuredOutputsParams

Imported separately. Constrains output to JSON schema, regex, choice list, or grammar.

```python
from vllm.sampling_params import StructuredOutputsParams

params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json={"type": "object", "properties": {"name": {"type": "string"}}}
    )
)
```

### Output Types

| Method | Returns | Key Field |
|---|---|---|
| `generate()` / `chat()` | `list[RequestOutput]` | `.outputs[0].text` |
| `embed()` | `list[EmbeddingRequestOutput]` | `.outputs.embedding` |
| `classify()` | `list[ClassificationRequestOutput]` | `.outputs.probs` |
| `score()` | `list[ScoringRequestOutput]` | `.outputs.score` |

## API Reference Files

| File | Content | Lines |
|---|---|---|
| [references/api-llm.md](references/api-llm.md) | LLM class constructor, all public methods, PromptType variants | ~380 |
| [references/api-sampling.md](references/api-sampling.md) | SamplingParams, StructuredOutputsParams, BeamSearchParams, all output types, PoolingParams | ~330 |
| [references/api-server.md](references/api-server.md) | `vllm serve` CLI flags, REST endpoints, OpenAI client usage, auth, tools, structured outputs | ~370 |
| [references/api-lora.md](references/api-lora.md) | LoRARequest class, offline/server LoRA usage, multi-LoRA, limitations | ~190 |
| [references/api-multimodal.md](references/api-multimodal.md) | Image/audio/video inputs, offline and server API, model-specific formats | ~250 |
| [references/workflows.md](references/workflows.md) | Complete working examples for all major use cases | ~420 |

## Key Gotchas

1. **generate() does not apply chat templates.** Use `chat()` for conversations. Using `generate()` with a chat model without the template will produce poor results.

2. **generation_config.json changes defaults.** HuggingFace models ship with `generation_config.json` that vLLM reads by default. This can change temperature, max_tokens, and other defaults. Pass `generation_config="vllm"` to disable.

3. **StructuredOutputsParams is not in vllm top-level.** Import from `vllm.sampling_params`:
   ```python
   from vllm.sampling_params import StructuredOutputsParams
   ```

4. **max_tokens defaults to 16.** This is intentionally low. Always set it explicitly for production use.

5. **Greedy sampling forces n=1.** When `temperature=0`, you cannot set `n > 1`.

6. **Pooling models need runner="pooling".** For `embed()`, `classify()`, and `score()`, the model must be loaded with `runner="pooling"` (or auto-detected).

7. **score() task is deprecated.** The `"score"` pooling task is deprecated in v0.19.0. Use `"classify"` instead. Will be removed in v0.20.

8. **LoRA needs enable_lora=True.** Must be set at `LLM()` construction time. Cannot be enabled after.

9. **swap_space is deprecated.** The parameter is accepted but ignored with a warning.

10. **Multimodal placeholders are model-specific.** When using `generate()` with images, the placeholder token (e.g., `<image>`) varies by model. Use `chat()` for automatic handling.

## What's New in v0.19.0 (vs v0.16.0)

### New Features
- **`enqueue()` / `wait_for_completion()`** -- Decouple request submission from result collection
- **`reward()` method** -- Generate reward scores via token-level classification
- **`RepetitionDetectionParams`** -- Detect and terminate repetitive N-gram patterns early
- **`thinking_token_budget`** in SamplingParams -- Control thinking token limits for reasoning models
- **`flat_logprobs`** option -- High-performance flat logprob format (`FlatLogprobs`)
- **`sleep()` / `wake_up()`** -- Engine lifecycle management with multi-level sleep (levels 0, 1, 2)
- **`collective_rpc()` / `apply_model()`** -- Direct worker and model access
- **`get_metrics()`** -- Retrieve Prometheus metrics snapshot
- **Tool server integration** -- `--tool-server` flag for external MCP tool servers
- **Anthropic Messages API** -- `/v1/messages` endpoint
- **Responses API** -- `/v1/responses` endpoint (OpenAI Responses format)
- **Cohere embedding API** -- `/v2/embed` endpoint
- **`structural_tag`** constraint in StructuredOutputsParams
- **Late interaction scoring** -- ColBERT-style MaxSim scoring in `score()`
- **`default_chat_template_kwargs`** -- Server-level defaults for chat template (useful for reasoning models)
- **`runner` and `convert` params** -- Explicit control over runner type and model conversion
- **Weight transfer** -- `init_weight_transfer_engine()` / `update_weights()` for RL training
- **Prefetch offloading** -- `offload_group_size`, `offload_num_in_group`, `offload_prefetch_step`, `offload_params`
- **`kv_cache_memory_bytes`** -- Fine-grained KV cache memory control
- **`logits_processors`** -- Custom logits processor support
- **`tokens_only` mode** -- Disaggregated serving token endpoint
- **Render endpoints** -- `/v1/chat/completions/render` and `/v1/completions/render`
- **Batch chat completions** -- `/v1/chat/completions/batch`

### Breaking Changes
- **V1 engine is now default.** The legacy `LLMEngine` is replaced by `vllm.v1.engine.llm_engine.LLMEngine`.
- **`swap_space` parameter deprecated** -- accepted but ignored with a warning.
- **`score` pooling task deprecated** -- use `classify` instead. Will be removed in v0.20.
- **Pooling multitask deprecated** -- specifying a different task than the model's default triggers a deprecation warning. Will be removed in v0.20. Use `PoolerConfig(task=...)` instead.

### Removed
- The legacy V0 engine code paths.
- Direct `LLMEngine` instantiation in `LLM` class (now always uses `LLMEngine.from_engine_args()`).

## Troubleshooting

### Out of Memory (OOM)
- Lower `gpu_memory_utilization` (default 0.9)
- Reduce `max_model_len`
- Use quantization (`quantization="awq"` or `"gptq"`)
- Use `cpu_offload_gb` for partial CPU offloading
- Use tensor parallelism (`tensor_parallel_size=2`)

### Poor Output Quality
- Check if `generation_config.json` is overriding your params. Use `generation_config="vllm"` to disable.
- For chat models, use `chat()` not `generate()`
- Increase `max_tokens` from the default of 16
- Ensure the chat template matches the model

### Structured Output Failures
- Verify your JSON schema is valid
- Try a different backend: `structured_outputs_config={"backend": "outlines"}`
- `grammar` cannot be an empty string
- `choice` cannot be an empty list
- Mistral tokenizers do not support `"guidance"` or `"lm-format-enforcer"` backends

### LoRA Not Working
- Ensure `enable_lora=True` at LLM construction
- `lora_int_id` must be > 0
- `lora_path` must not be empty
- Adapter rank must not exceed `max_lora_rank`

### Server Connection Issues
- Default port is 8000, check with `--port`
- Health check: `GET /health`
- API key required? Check `--api-key` or `VLLM_API_KEY`

## Resources

- [references/api-llm.md](references/api-llm.md) -- LLM class full reference
- [references/api-sampling.md](references/api-sampling.md) -- SamplingParams and output types
- [references/api-server.md](references/api-server.md) -- OpenAI-compatible server
- [references/api-lora.md](references/api-lora.md) -- LoRA adapters
- [references/api-multimodal.md](references/api-multimodal.md) -- Multimodal inputs
- [references/workflows.md](references/workflows.md) -- Complete working examples
