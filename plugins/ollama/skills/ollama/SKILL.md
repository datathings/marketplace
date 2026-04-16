---
name: ollama
description: "Run and manage local LLMs via Ollama REST API — text generation, chat completions, embeddings, tool calling, structured output, and model management. Use when code imports ollama, references localhost:11434, or user asks about local LLM inference."
---

# Ollama API Reference (v0.20.7)

Ollama runs large language models locally. It exposes a REST API on `http://localhost:11434` for text generation, chat, embeddings, model management, and more.

## Key Concepts

- **Model names** follow `model:tag` format (e.g., `llama3.2:latest`, `orca-mini:3b-q8_0`). Tag defaults to `latest`.
- **Streaming** is enabled by default on generation endpoints. Disable with `"stream": false`.
- **Durations** are returned in nanoseconds.
- **Tokens/sec** = `eval_count / eval_duration * 10^9`.
- **keep_alive** controls how long a model stays loaded in memory (default `5m`). Set to `0` to unload immediately, `-1` to keep loaded indefinitely.
- **Thinking models** support `"think": true` (or `"high"`, `"medium"`, `"low"`) to enable chain-of-thought reasoning.
- **Structured output** via `"format"` parameter: set to `"json"` for JSON mode, or pass a JSON Schema object.
- **Tool calling** is supported in `/api/chat` by providing a `tools` array.
- **Modelfile** is a blueprint for creating custom models (FROM, PARAMETER, TEMPLATE, SYSTEM, ADAPTER, LICENSE, MESSAGE instructions).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate text completion (streaming) |
| `POST` | `/api/chat` | Chat completion with message history (streaming) |
| `POST` | `/api/embed` | Generate embeddings (single or batch) |
| `POST` | `/api/embeddings` | Generate embeddings (legacy, deprecated) |
| `GET` | `/api/tags` | List locally available models |
| `POST` | `/api/show` | Show model details and metadata |
| `POST` | `/api/create` | Create a model from Modelfile, GGUF, or safetensors |
| `POST` | `/api/pull` | Pull/download a model from registry |
| `POST` | `/api/push` | Push a model to registry |
| `POST` | `/api/copy` | Copy/clone a model locally |
| `DELETE` | `/api/delete` | Delete a model |
| `GET` | `/api/ps` | List currently loaded/running models |
| `HEAD` | `/api/blobs/:digest` | Check if a blob exists |
| `POST` | `/api/blobs/:digest` | Upload a blob (for GGUF/safetensors creation) |
| `GET` | `/api/version` | Get Ollama server version |

## Quick Start

```bash
# Pull a model
curl http://localhost:11434/api/pull -d '{"model": "llama3.2"}'

# Generate text
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# Embeddings
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": "Why is the sky blue?"
}'
```

## Model Options (passed via `options` field)

These runtime parameters can be passed in the `options` object of generate/chat/embed requests:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ctx` | int | 2048 | Context window size |
| `num_batch` | int | 512 | Batch size for prompt processing |
| `num_gpu` | int | -1 (auto) | Number of layers to offload to GPU |
| `main_gpu` | int | 0 | Main GPU index |
| `use_mmap` | bool | (auto) | Use memory-mapped files |
| `num_thread` | int | 0 (auto) | Number of threads |
| `num_keep` | int | 4 | Number of tokens to keep from initial prompt |
| `seed` | int | -1 | Random seed (-1 = random) |
| `num_predict` | int | -1 | Max tokens to generate (-1 = infinite) |
| `top_k` | int | 40 | Top-K sampling |
| `top_p` | float | 0.9 | Top-P (nucleus) sampling |
| `min_p` | float | 0.0 | Min-P sampling |
| `typical_p` | float | 1.0 | Typical-P sampling |
| `repeat_last_n` | int | 64 | Lookback for repeat penalty (0=disabled, -1=num_ctx) |
| `temperature` | float | 0.8 | Sampling temperature |
| `repeat_penalty` | float | 1.1 | Repetition penalty |
| `presence_penalty` | float | 0.0 | Presence penalty |
| `frequency_penalty` | float | 0.0 | Frequency penalty |
| `stop` | string[] | [] | Stop sequences |

## Detailed References

- [Generation and Chat API](references/api-generation.md) -- `/api/generate` and `/api/chat` with all fields, streaming, tool calling, structured output, thinking, images
- [Model Management API](references/api-models.md) -- `/api/tags`, `/api/show`, `/api/create`, `/api/pull`, `/api/push`, `/api/copy`, `/api/delete`, `/api/ps`, blobs, version
- [Embeddings API](references/api-embeddings.md) -- `/api/embed` and legacy `/api/embeddings`
- [Modelfile Reference](references/modelfile.md) -- FROM, PARAMETER, TEMPLATE, SYSTEM, ADAPTER, LICENSE, MESSAGE instructions
- [Workflow Examples](references/workflows.md) -- Complete curl/code examples for common tasks
