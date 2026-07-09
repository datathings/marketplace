# LoRA Adapter Reference

## LoRARequest Class

```python
from vllm.lora.request import LoRARequest

lora_request = LoRARequest(
    lora_name: str,                        # Unique adapter name
    lora_int_id: int,                      # Unique integer ID (must be > 0)
    lora_path: str = "",                   # Path to adapter weights (required, cannot be empty)
    base_model_name: str | None = None,    # Optional base model name
    tensorizer_config_dict: dict | None = None,  # Tensorizer configuration
    load_inplace: bool = False,            # Force reload even if cached
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `lora_name` | `str` | required | Unique name for the adapter. Used for equality/hashing |
| `lora_int_id` | `int` | required | Globally unique integer ID. Must be > 0 |
| `lora_path` | `str` | `""` | Local path or HuggingFace repo to adapter weights. Must not be empty |
| `base_model_name` | `str \| None` | `None` | Optional base model name for tracking |
| `tensorizer_config_dict` | `dict \| None` | `None` | Configuration for tensorizer loading |
| `load_inplace` | `bool` | `False` | If True, force reload the adapter even if one with the same ID exists in cache |

### Properties

| Property | Type | Description |
|---|---|---|
| `adapter_id` | `int` | Alias for `lora_int_id` |
| `name` | `str` | Alias for `lora_name` |
| `path` | `str` | Alias for `lora_path` |

### Equality and Hashing

LoRA requests are compared and hashed by `lora_name` only. Two requests with the same `lora_name` are considered equal regardless of other fields.

---

## Offline LoRA Usage with LLM

### Enable LoRA Support

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=16,        # Must match or exceed adapter rank
    max_loras=4,             # Max adapters in memory simultaneously
)
```

### Generate with LoRA

```python
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

lora = LoRARequest(
    lora_name="my-adapter",
    lora_int_id=1,
    lora_path="/path/to/lora/adapter",
)

outputs = llm.generate(
    ["Write a poem about AI"],
    sampling_params,
    lora_request=lora,
)
```

### Chat with LoRA

```python
outputs = llm.chat(
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    sampling_params=sampling_params,
    lora_request=lora,
)
```

### Multiple Adapters in a Batch

Different prompts can use different LoRA adapters in the same batch:

```python
lora_a = LoRARequest("adapter-a", 1, "/path/to/adapter-a")
lora_b = LoRARequest("adapter-b", 2, "/path/to/adapter-b")

outputs = llm.generate(
    ["Prompt for adapter A", "Prompt for adapter B"],
    sampling_params,
    lora_request=[lora_a, lora_b],
)
```

When `lora_request` is a single `LoRARequest`, it is applied to all prompts. When it is a list, it must have the same length as the prompts.

---

## Server-Side LoRA Configuration

### Static LoRA Modules

Register LoRA adapters at server startup:

```bash
# Old format: name=path
vllm serve base-model --enable-lora \
  --lora-modules adapter1=/path/to/adapter1 adapter2=/path/to/adapter2

# JSON format
vllm serve base-model --enable-lora \
  --lora-modules '{"name": "adapter1", "path": "/path/to/adapter1"}' \
  --lora-modules '{"name": "adapter2", "path": "/path/to/adapter2", "base_model_name": "base"}'
```

### Dynamic LoRA Loading/Unloading

```python
import requests

# Load a LoRA adapter at runtime
requests.post("http://localhost:8000/v1/load_lora_adapter", json={
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter",
})

# Unload a LoRA adapter
requests.post("http://localhost:8000/v1/unload_lora_adapter", json={
    "lora_name": "my-adapter",
})
```

### Using LoRA via OpenAI Client

Once registered (statically or dynamically), reference the LoRA adapter by name as the `model`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Use the LoRA adapter by its registered name
response = client.chat.completions.create(
    model="adapter1",  # LoRA adapter name
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### List Available Models (including LoRA)

```python
models = client.models.list()
for model in models.data:
    print(model.id)
# Output includes base model and all registered LoRA adapters
```

---

## LoRA with Embeddings

LoRA adapters can also be used with embedding/pooling models:

```python
llm = LLM(
    model="BAAI/bge-large-en-v1.5",
    enable_lora=True,
    runner="pooling",
)

lora = LoRARequest("embed-adapter", 1, "/path/to/embed-adapter")

outputs = llm.embed(
    ["Text to embed"],
    lora_request=lora,
)
```

---

## Engine Configuration Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-lora` | flag | False | Enable LoRA adapter support |
| `--max-lora-rank` | `int` | 16 | Maximum LoRA rank supported |
| `--max-loras` | `int` | 1 | Maximum number of LoRA adapters in GPU memory |
| `--lora-extra-vocab-size` | `int` | 256 | Extra vocabulary size for LoRA adapters |
| `--long-lora-scaling-factors` | `float` (multi) | None | Scaling factors for Long LoRA |
| `--lora-dtype` | `str` | `"auto"` | LoRA weight data type |

---

## Limitations and Requirements

1. **Adapter format:** Must be HuggingFace PEFT-compatible LoRA weights.
2. **Rank constraint:** The adapter rank must not exceed `max_lora_rank`.
3. **Memory:** Each loaded adapter consumes GPU memory. Control with `max_loras`.
4. **ID uniqueness:** `lora_int_id` should be globally unique per adapter. This is currently not enforced by vLLM.
5. **One adapter per request:** Each request can use at most one LoRA adapter. Concurrent requests can use different adapters.
6. **Supported architectures:** Not all model architectures support LoRA. Check the vLLM documentation for the list of supported models.
7. **Multimodal LoRA:** vLLM supports default per-modality LoRA adapters via `default_mm_loras` in the LoRA config. These are automatically applied when multimodal inputs are detected.
