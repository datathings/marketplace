# OpenAI-Compatible Server Reference

## Starting the Server

```bash
vllm serve <model> [options]
```

or equivalently:

```bash
python -m vllm.entrypoints.openai.api_server <model> [options]
```

### Basic Examples

```bash
# Serve a model
vllm serve meta-llama/Llama-3.1-8B-Instruct

# With tensor parallelism across 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# With quantization
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq

# From a YAML config file
vllm serve --config config.yaml
```

---

## CLI Flags

### Positional Arguments

| Argument | Description |
|---|---|
| `model_tag` | The model to serve (optional if specified in config) |

### Server Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | None | YAML config file path |
| `--headless` | flag | `False` | Run in headless mode (multi-node data parallel) |
| `--api-server-count`, `-asc` | `int` | None | Number of API server processes. Defaults to `data_parallel_size` |

### Network and Host

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | None | Bind hostname |
| `--port` | `int` | `8000` | Bind port |
| `--uds` | `str` | None | Unix domain socket path (ignores host/port) |
| `--uvicorn-log-level` | `str` | `"info"` | Uvicorn log level: critical/error/warning/info/debug/trace |
| `--disable-uvicorn-access-log` | flag | `False` | Disable uvicorn access log |
| `--disable-access-log-for-endpoints` | `str` | None | Comma-separated endpoint paths to exclude from access logs |

### CORS

| Flag | Type | Default | Description |
|---|---|---|---|
| `--allow-credentials` | flag | `False` | Allow CORS credentials |
| `--allowed-origins` | JSON | `["*"]` | Allowed CORS origins |
| `--allowed-methods` | JSON | `["*"]` | Allowed CORS methods |
| `--allowed-headers` | JSON | `["*"]` | Allowed CORS headers |

### Authentication and SSL

| Flag | Type | Default | Description |
|---|---|---|---|
| `--api-key` | `str` (multi) | None | API key(s) required in Authorization header. Falls back to `VLLM_API_KEY` env var |
| `--ssl-keyfile` | `str` | None | SSL key file path |
| `--ssl-certfile` | `str` | None | SSL certificate file path |
| `--ssl-ca-certs` | `str` | None | CA certificates file |
| `--enable-ssl-refresh` | flag | `False` | Auto-refresh SSL context on cert change |
| `--ssl-cert-reqs` | `int` | `0` | Client certificate requirement (stdlib ssl module) |
| `--ssl-ciphers` | `str` | None | SSL cipher suites (TLS 1.2 and below) |

### Chat and Templates

| Flag | Type | Default | Description |
|---|---|---|---|
| `--chat-template` | `str` | None | Chat template file path or inline template |
| `--chat-template-content-format` | `str` | `"auto"` | Content format: `"string"` or `"openai"` |
| `--trust-request-chat-template` | flag | `False` | Allow client-specified chat templates |
| `--default-chat-template-kwargs` | JSON | None | Default kwargs for chat template (e.g., `'{"enable_thinking": false}'`) |
| `--response-role` | `str` | `"assistant"` | Role name for assistant responses |

### Tool Calling

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-auto-tool-choice` | flag | `False` | Enable auto tool choice for supported models |
| `--tool-call-parser` | `str` | None | Tool call parser: hermes, mistral, llama3_json, etc. Required with `--enable-auto-tool-choice` |
| `--tool-parser-plugin` | `str` | `""` | Custom tool parser plugin |
| `--tool-server` | `str` | None | Tool server host:port pairs or `"demo"` for built-in demo tools |
| `--exclude-tools-when-tool-choice-none` | flag | `False` | Exclude tools from prompt when `tool_choice='none'` |

### LoRA

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lora-modules` | `str` (multi) | None | LoRA configs: `name=path` or JSON `{"name":"...", "path":"...", "base_model_name":"..."}` |

### Logging and Metrics

| Flag | Type | Default | Description |
|---|---|---|---|
| `--log-config-file` | `str` | None | Logging config JSON file |
| `--max-log-len` | `int` | None | Max prompt chars/IDs to log. None = unlimited |
| `--return-tokens-as-token-ids` | flag | `False` | Return non-JSON-encodable tokens as `token_id:{id}` |
| `--enable-prompt-tokens-details` | flag | `False` | Enable `prompt_tokens_details` in usage |
| `--enable-server-load-tracking` | flag | `False` | Track server load metrics |
| `--enable-force-include-usage` | flag | `False` | Include usage on every request |
| `--enable-log-outputs` | flag | `False` | Log model outputs (requires `--enable-log-requests`) |
| `--enable-log-deltas` | flag | `True` | Log output deltas (with `--enable-log-outputs`) |
| `--log-error-stack` | flag | False | Log error stack traces (True in dev mode) |

### Server Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--root-path` | `str` | None | FastAPI root path (behind reverse proxy) |
| `--middleware` | `str` (multi) | `[]` | ASGI middleware import paths |
| `--enable-request-id-headers` | flag | `False` | Add X-Request-Id response headers |
| `--disable-fastapi-docs` | flag | `False` | Disable Swagger/ReDoc documentation |
| `--enable-offline-docs` | flag | `False` | Offline FastAPI docs (air-gapped environments) |
| `--enable-tokenizer-info-endpoint` | flag | `False` | Enable `/tokenizer_info` endpoint |
| `--tokens-only` | flag | `False` | Only enable Tokens In/Out endpoint (disaggregated) |
| `--h11-max-incomplete-event-size` | `int` | 4194304 | Max incomplete HTTP event size (bytes) |
| `--h11-max-header-count` | `int` | 256 | Max HTTP headers per request |

### Engine Arguments

All `AsyncEngineArgs` flags are also available. Key ones:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | required | Model name or path |
| `--tokenizer` | `str` | None | Tokenizer name/path |
| `--tensor-parallel-size`, `-tp` | `int` | 1 | Number of GPUs for tensor parallelism |
| `--pipeline-parallel-size`, `-pp` | `int` | 1 | Pipeline parallelism stages |
| `--data-parallel-size`, `-dp` | `int` | 1 | Data parallelism |
| `--dtype` | `str` | `"auto"` | Model dtype |
| `--quantization`, `-q` | `str` | None | Quantization method |
| `--max-model-len` | `int` | None | Max sequence length |
| `--gpu-memory-utilization` | `float` | 0.9 | GPU memory fraction |
| `--trust-remote-code` | flag | False | Trust remote code |
| `--seed` | `int` | 0 | Random seed |
| `--enable-prefix-caching` | flag | False | Enable prefix caching |
| `--enable-lora` | flag | False | Enable LoRA |
| `--max-lora-rank` | `int` | 16 | Max LoRA rank |
| `--max-loras` | `int` | 1 | Max concurrent LoRA adapters |
| `--served-model-name` | `str` (multi) | None | Custom model name(s) |
| `--generation-config` | `str` | None | `"vllm"` to disable HF generation_config |
| `--enable-log-requests` | flag | False | Log requests |
| `--structured-output-backend` | `str` | `"auto"` | Backend: auto/xgrammar/guidance/outlines/lm-format-enforcer |

---

## REST API Endpoints

### Generation Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| POST | `/v1/chat/completions/batch` | Batch chat completions |
| POST | `/v1/completions` | Text completions (OpenAI-compatible) |
| POST | `/v1/chat/completions/render` | Render chat template without generating |
| POST | `/v1/completions/render` | Render completion prompt without generating |
| POST | `/v1/responses` | OpenAI Responses API |

### Pooling Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| POST | `/v2/embed` | Cohere-compatible embeddings |
| POST | `/pooling` | Generic pooling |
| POST | `/v1/score` | Scoring / reranking |
| POST | `/v1/classify` | Classification |

### Model Management

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | List available models |
| POST | `/v1/load_lora_adapter` | Load a LoRA adapter |
| POST | `/v1/unload_lora_adapter` | Unload a LoRA adapter |

### Operations

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/ping` | Sagemaker-compatible health check |
| GET | `/version` | Server version |
| GET | `/metrics` | Prometheus metrics |
| POST | `/v1/tokenize` | Tokenize text |
| POST | `/v1/detokenize` | Detokenize token IDs |
| POST | `/start_profile` | Start profiling |
| POST | `/stop_profile` | Stop profiling |
| POST | `/v1/sleep` | Put engine to sleep |
| POST | `/v1/wake_up` | Wake engine from sleep |
| POST | `/reset_prefix_cache` | Reset prefix cache |

### Anthropic-Compatible

| Method | Path | Description |
|---|---|---|
| POST | `/v1/messages` | Anthropic Messages API |

### Speech-to-Text (when supported)

| Method | Path | Description |
|---|---|---|
| POST | `/v1/audio/transcriptions` | Audio transcription |

---

## Using the OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",           # or your API key
    base_url="http://localhost:8000/v1",
)

# Chat completions
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is vLLM?"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings

```python
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=["Hello world", "Goodbye world"],
)
for item in response.data:
    print(f"Embedding dim: {len(item.embedding)}")
```

### Completions (Legacy)

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The capital of France is",
    max_tokens=32,
    temperature=0.0,
)
print(response.choices[0].text)
```

---

## Authentication

### Server-side

```bash
# Via CLI
vllm serve model --api-key sk-my-secret-key

# Multiple keys
vllm serve model --api-key sk-key1 --api-key sk-key2

# Via environment variable
export VLLM_API_KEY=sk-my-secret-key
vllm serve model
```

### Client-side

```python
client = OpenAI(
    api_key="sk-my-secret-key",
    base_url="http://localhost:8000/v1",
)
```

Or via header:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [...]}'
```

---

## Tool Calling / Function Calling

### Server Setup

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Available parsers: `hermes`, `mistral`, `llama3_json`, `granite`, `granite_20b_fc`, `jamba`, `pythonic`, `internlm`, `llama4`, and custom plugins.

### Client Usage

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

### Built-in Tool Server (Demo)

```bash
vllm serve model \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tool-server demo
```

The `demo` tool server provides a browser tool and Python code interpreter (runs in Docker).

---

## Structured Outputs via Server

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Generate a person"}],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    },
)
```

### Structured Output Request Fields

These can be passed in `extra_body`:

| Field | Description |
|---|---|
| `guided_json` | JSON schema (string or dict) |
| `guided_regex` | Regular expression pattern |
| `guided_choice` | List of allowed strings |
| `guided_grammar` | Context-free grammar |

Or use the OpenAI-compatible `response_format`:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Generate a person"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    },
)
```

---

## YAML Configuration

```yaml
# config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 2
gpu_memory_utilization: 0.9
max_model_len: 8192
host: 0.0.0.0
port: 8000
api_key: sk-my-key
enable_auto_tool_choice: true
tool_call_parser: hermes
```

```bash
vllm serve --config config.yaml
```

---

## Multi-Worker / Data Parallel

```bash
# Multi-worker with multiple API servers
vllm serve model \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --api-server-count 2
```

---

## Thinking / Reasoning Models

For models that support thinking tokens (e.g., Qwen3, DeepSeek):

```bash
vllm serve Qwen/Qwen3-8B \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

This disables thinking mode by default. Clients can override per-request.
