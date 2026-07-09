# Complete Working Examples

## Basic Text Generation with generate()

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

prompts = [
    "The future of artificial intelligence is",
    "In a world where robots",
    "The best programming language is",
]

outputs = llm.generate(prompts, params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Output: {generated!r}\n")
```

**Important:** `generate()` does NOT apply chat templates. For raw text completion only.

---

## Chat Completion with chat()

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

params = SamplingParams(temperature=0.7, max_tokens=512)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to compute Fibonacci numbers."},
]

outputs = llm.chat(messages, sampling_params=params)
print(outputs[0].outputs[0].text)
```

### Batch Chat

```python
conversations = [
    [
        {"role": "user", "content": "What is 2+2?"},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": "Write a haiku about coding."},
    ],
]

outputs = llm.chat(conversations, sampling_params=params)
for conv, output in zip(conversations, outputs):
    print(f"Q: {conv[-1]['content']}")
    print(f"A: {output.outputs[0].text}\n")
```

---

## Embeddings with embed()

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-large-en-v1.5", runner="pooling")

texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "The weather is nice today.",
]

outputs = llm.embed(texts)

for text, output in zip(texts, outputs):
    embedding = output.outputs.embedding
    print(f"Text: {text!r}")
    print(f"Embedding dim: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}\n")
```

### Cosine Similarity

```python
import numpy as np

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

emb_ml = outputs[0].outputs.embedding
emb_dl = outputs[1].outputs.embedding
emb_weather = outputs[2].outputs.embedding

print(f"ML vs DL: {cosine_sim(emb_ml, emb_dl):.4f}")       # High similarity
print(f"ML vs Weather: {cosine_sim(emb_ml, emb_weather):.4f}")  # Low similarity
```

---

## Structured JSON Output

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "hobbies": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "age", "hobbies"],
}

params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    structured_outputs=StructuredOutputsParams(json=schema),
)

outputs = llm.chat(
    messages=[
        {"role": "user", "content": "Generate a profile for a fictional person."},
    ],
    sampling_params=params,
)

import json
result = json.loads(outputs[0].outputs[0].text)
print(result)
# {"name": "Alice Chen", "age": 28, "hobbies": ["hiking", "photography"]}
```

### Regex Constraint

```python
params = SamplingParams(
    max_tokens=32,
    structured_outputs=StructuredOutputsParams(
        regex=r"\d{3}-\d{3}-\d{4}"
    ),
)
```

### Choice Constraint

```python
params = SamplingParams(
    max_tokens=16,
    structured_outputs=StructuredOutputsParams(
        choice=["positive", "negative", "neutral"]
    ),
)
```

---

## Streaming with OpenAI Client

```python
from openai import OpenAI

# Start server: vllm serve meta-llama/Llama-3.1-8B-Instruct
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a storyteller."},
        {"role": "user", "content": "Tell me a short story about a robot."},
    ],
    stream=True,
    temperature=0.9,
    max_tokens=512,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

### Async Streaming

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    stream = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Explain quantum computing."}],
        stream=True,
        max_tokens=256,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

asyncio.run(main())
```

---

## LoRA Adapter Usage

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=16,
)

# Base model generation
base_output = llm.generate(
    ["Translate to French: Hello, how are you?"],
    SamplingParams(max_tokens=64),
)
print(f"Base: {base_output[0].outputs[0].text}")

# LoRA-adapted generation
lora = LoRARequest(
    lora_name="french-translator",
    lora_int_id=1,
    lora_path="/path/to/french-lora-adapter",
)

lora_output = llm.generate(
    ["Translate to French: Hello, how are you?"],
    SamplingParams(max_tokens=64),
    lora_request=lora,
)
print(f"LoRA: {lora_output[0].outputs[0].text}")
```

---

## Multi-Image Input

### Offline (chat)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.chat(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///path/to/cat.jpg"}},
            {"type": "image_url", "image_url": {"url": "file:///path/to/dog.jpg"}},
            {"type": "text", "text": "What animals are in these images? How are they different?"},
        ],
    }],
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
```

### Offline (generate with multi_modal_data)

```python
from PIL import Image

img1 = Image.open("cat.jpg")
img2 = Image.open("dog.jpg")

outputs = llm.generate(
    {
        "prompt": "<image><image>\nCompare these two images.",
        "multi_modal_data": {"image": [img1, img2]},
    },
    params,
)
```

---

## Server Deployment with vllm serve

### Basic Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Production Server

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --api-key sk-production-key \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
```

### With LoRA and Tools

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules adapter1=/path/to/adapter1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### YAML Config

```yaml
# config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 2
gpu_memory_utilization: 0.9
max_model_len: 8192
host: 0.0.0.0
port: 8000
enable_prefix_caching: true
api_key: sk-my-key
```

```bash
vllm serve --config config.yaml
```

---

## Batch Processing

### Large Batch with generate()

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.0, max_tokens=128)

# Load prompts from file
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"Processing {len(prompts)} prompts...")

# vLLM automatically batches for optimal throughput
outputs = llm.generate(prompts, params)

results = []
for output in outputs:
    results.append({
        "prompt": output.prompt,
        "response": output.outputs[0].text,
        "tokens": len(output.outputs[0].token_ids),
    })

import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Using enqueue/wait_for_completion

```python
# Enqueue without waiting
request_ids = llm.enqueue(prompts[:100], params)
print(f"Enqueued {len(request_ids)} requests")

# Add more
request_ids += llm.enqueue(prompts[100:200], params)

# Process everything and get results
outputs = llm.wait_for_completion()
print(f"Completed {len(outputs)} requests")
```

---

## Custom Sampling Parameters

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Creative writing
creative_params = SamplingParams(
    temperature=1.2,
    top_p=0.95,
    top_k=50,
    min_p=0.05,
    max_tokens=1024,
    presence_penalty=0.6,
    frequency_penalty=0.3,
)

# Precise/factual
precise_params = SamplingParams(
    temperature=0.0,  # Greedy
    max_tokens=256,
)

# With log probabilities
logprob_params = SamplingParams(
    temperature=0.7,
    max_tokens=128,
    logprobs=5,        # Top 5 logprobs per token
    prompt_logprobs=1, # 1 logprob per prompt token
)

# With stop sequences
stop_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    stop=["\n\n", "END", "---"],
    stop_token_ids=[128001],  # Model-specific EOS variants
    include_stop_str_in_output=False,
)

# With repetition detection (new in v0.19.0)
no_repeat_params = SamplingParams(
    temperature=0.8,
    max_tokens=2048,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=10,
        min_count=3,
    ),
)

# Multiple completions per prompt
multi_params = SamplingParams(
    n=3,               # Generate 3 completions
    temperature=0.9,
    max_tokens=256,
)

# With bad words
filtered_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    bad_words=["horrible", "terrible", "awful"],
)

# Generate with different params
outputs_creative = llm.generate(["Write a poem:"], creative_params)
outputs_precise = llm.generate(["What is 2+2?"], precise_params)
```

### Accessing Log Probabilities

```python
outputs = llm.generate(["The capital of France is"], logprob_params)

output = outputs[0].outputs[0]
if output.logprobs:
    for position, token_logprobs in enumerate(output.logprobs):
        for token_id, logprob_info in token_logprobs.items():
            print(f"  Position {position}: "
                  f"token={logprob_info.decoded_token!r} "
                  f"logprob={logprob_info.logprob:.4f} "
                  f"rank={logprob_info.rank}")
```

---

## Disabling generation_config.json

By default, vLLM reads `generation_config.json` from HuggingFace models to set default sampling parameters. To disable this:

```python
# Offline
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", generation_config="vllm")

# Server
# vllm serve model --generation-config vllm
```

When disabled, `SamplingParams()` defaults apply (temperature=1.0, max_tokens=16, etc.).

---

## Classification

```python
from vllm import LLM

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", runner="pooling")

texts = [
    "Machine learning is fascinating",
    "I love sunny weather",
]

outputs = llm.classify(texts)

for text, output in zip(texts, outputs):
    probs = output.outputs.probs
    print(f"Text: {text!r}")
    print(f"Probs: {probs}")
    print(f"Num classes: {output.outputs.num_classes}\n")
```

---

## Scoring (Cross-Encoder)

```python
from vllm import LLM

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", runner="pooling")

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather is nice today.",
    "Deep learning uses neural networks for learning.",
]

# 1-to-N scoring
outputs = llm.score(query, documents)

for doc, output in zip(documents, outputs):
    print(f"Score: {output.outputs.score:.4f} | {doc}")
```

---

## Sleep / Wake for Weight Management

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Generate normally
outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=32))

# Sleep: offload weights to CPU
llm.sleep(level=1)

# ... do something else, free GPU memory ...

# Wake up: reload weights to GPU
llm.wake_up()

# Generate again
outputs = llm.generate(["Hello again!"], SamplingParams(max_tokens=32))
```
