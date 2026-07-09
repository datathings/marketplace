# Multimodal Inputs Reference

vLLM supports images, audio, and video inputs for multimodal models. This reference covers how to pass multimodal data through both the offline `LLM` API and the OpenAI-compatible server.

## Supported Modalities

| Modality | Input Types | Description |
|---|---|---|
| `image` | PIL Image, numpy array, torch.Tensor, bytes | Single image input |
| `video` | list of PIL Images, numpy array, torch.Tensor | Video as frame sequence |
| `audio` | list of floats, numpy array, torch.Tensor, tuple(array, sample_rate) | Audio waveform |

---

## Offline API (LLM class)

### Images with generate()

For `generate()`, pass multimodal data via the `TextPrompt` or `TokensPrompt` dict format using the `multi_modal_data` key:

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
params = SamplingParams(temperature=0.8, max_tokens=256)

image = Image.open("photo.jpg")

# TextPrompt with multimodal data
outputs = llm.generate(
    {
        "prompt": "<image>\nDescribe this image in detail.",
        "multi_modal_data": {"image": image},
    },
    params,
)
print(outputs[0].outputs[0].text)
```

**Important:** The prompt must contain the model-specific image placeholder token (e.g., `<image>` for LLaVA). Different models use different placeholder formats.

### Multiple Images

Pass a list of images:

```python
images = [Image.open("photo1.jpg"), Image.open("photo2.jpg")]

outputs = llm.generate(
    {
        "prompt": "<image><image>\nCompare these two images.",
        "multi_modal_data": {"image": images},
    },
    params,
)
```

The number of placeholder tokens must match the number of images.

### Images with chat()

The `chat()` method uses OpenAI-style message format, which is typically more convenient:

```python
outputs = llm.chat(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///path/to/photo.jpg"}},
            {"type": "text", "text": "What is in this image?"},
        ],
    }],
    sampling_params=params,
)
```

Supported `image_url` formats:
- `file:///path/to/image.jpg` -- local file
- `https://example.com/image.jpg` -- remote URL
- `data:image/jpeg;base64,...` -- base64-encoded

### Audio

```python
import numpy as np

llm = LLM(model="fixie-ai/ultravox-v0_5")

# Audio as numpy array
audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

outputs = llm.generate(
    {
        "prompt": "<audio>\nTranscribe this audio.",
        "multi_modal_data": {"audio": audio_data},
    },
    params,
)
```

Audio can also be passed as a tuple `(array, sample_rate)` when the audio sample rate differs from what the model expects -- vLLM will resample automatically:

```python
audio_at_44k = np.random.randn(44100).astype(np.float32)  # 44.1kHz
outputs = llm.generate(
    {
        "prompt": "<audio>\nTranscribe.",
        "multi_modal_data": {"audio": (audio_at_44k, 44100)},
    },
    params,
)
```

### Audio via chat()

```python
outputs = llm.chat(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_encoded_audio,
                    "format": "wav",
                },
            },
            {"type": "text", "text": "What does this audio say?"},
        ],
    }],
    sampling_params=params,
)
```

### Video

```python
from PIL import Image

llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf")

# Video as list of PIL frames
frames = [Image.open(f"frame_{i:04d}.jpg") for i in range(16)]

outputs = llm.generate(
    {
        "prompt": "<video>\nDescribe what happens in this video.",
        "multi_modal_data": {"video": frames},
    },
    params,
)
```

### Pre-computed Embeddings

Instead of raw media, you can pass pre-computed embeddings as a torch.Tensor:

```python
import torch

# 3D tensor: treated as media embeddings, passed directly to model
image_embeds = torch.randn(1, 336, 4096)  # [batch, seq_len, hidden_dim]

outputs = llm.generate(
    {
        "prompt": "<image>\nDescribe this.",
        "multi_modal_data": {"image": image_embeds},
    },
    params,
)
```

### Processor Overrides (mm_processor_kwargs)

Override multimodal processor behavior per prompt or globally:

```python
# Per-prompt override
outputs = llm.generate(
    {
        "prompt": "<image>\nDescribe.",
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": {"num_crops": 4},
    },
    params,
)

# Global override at LLM init
llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    mm_processor_kwargs={"num_crops": 4},
)
```

For `chat()`, use the `mm_processor_kwargs` parameter:

```python
outputs = llm.chat(
    messages=[...],
    sampling_params=params,
    mm_processor_kwargs={"num_crops": 4},
)
```

---

## Server API (OpenAI-Compatible)

### Images via Chat Completions

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/photo.jpg"},
            },
            {"type": "text", "text": "What is in this image?"},
        ],
    }],
    max_tokens=256,
)
```

### Base64 Images

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }],
)
```

### Multiple Images via Server

```python
response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/img2.jpg"}},
            {"type": "text", "text": "Compare these images."},
        ],
    }],
)
```

### Server Security for Local Files

To allow the server to access local images:

```bash
vllm serve model --allowed-local-media-path /path/to/media/directory
```

Then use `file:///path/to/media/directory/image.jpg` in URLs.

To restrict remote URLs to specific domains:

```bash
vllm serve model --allowed-media-domains example.com cdn.example.com
```

---

## Model-Specific Prompt Formats

Different multimodal models expect different placeholder tokens and prompt formats. Common patterns:

| Model Family | Image Placeholder | Notes |
|---|---|---|
| LLaVA | `<image>` | Place before the question |
| Phi-3 Vision | `<|image_1|>` | Numbered placeholders |
| Qwen-VL | `<img>...</img>` | URL or base64 in tags |
| InternVL | `<image>` | Similar to LLaVA |
| Fuyu | (no placeholder) | Image processed implicitly |

When using `chat()` or the OpenAI server API, the chat template handles placeholder insertion automatically. When using `generate()` directly, you must include the correct placeholders.

---

## Multimodal Caching

vLLM caches multimodal processor outputs to avoid redundant computation. You can provide custom UUIDs for cache keys:

```python
outputs = llm.generate(
    {
        "prompt": "<image>\nDescribe.",
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": "my-custom-uuid-123"},
    },
    params,
)
```

To reset the cache:

```python
llm.reset_mm_cache()
```

---

## Supported Input Type Aliases

From `vllm.multimodal.inputs`:

| Type | Accepted Formats |
|---|---|
| `ImageItem` | PIL Image, numpy array, torch.Tensor (3D = embeddings), MediaWithBytes |
| `VideoItem` | list[PIL Image], numpy array, torch.Tensor (3D = embeddings), tuple(frames, metadata) |
| `AudioItem` | list[float], numpy array, torch.Tensor (3D = embeddings), tuple(array, sample_rate) |
