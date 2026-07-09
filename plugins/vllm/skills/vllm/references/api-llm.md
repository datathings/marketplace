# LLM Class Reference

Complete reference for the `vllm.LLM` class -- the primary offline inference API.

## Constructor

```python
from vllm import LLM

llm = LLM(
    model: str,
    *,
    runner: RunnerOption = "auto",
    convert: ConvertOption = "auto",
    tokenizer: str | None = None,
    tokenizer_mode: TokenizerMode | str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    allowed_media_domains: list[str] | None = None,
    tensor_parallel_size: int = 1,
    dtype: ModelDType = "auto",
    quantization: QuantizationMethods | None = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    chat_template: Path | str | None = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    cpu_offload_gb: float = 0,
    offload_group_size: int = 0,
    offload_num_in_group: int = 1,
    offload_prefetch_step: int = 1,
    offload_params: set[str] | None = None,
    enforce_eager: bool = False,
    enable_return_routed_experts: bool = False,
    disable_custom_all_reduce: bool = False,
    hf_token: bool | str | None = None,
    hf_overrides: HfOverrides | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
    pooler_config: PoolerConfig | None = None,
    structured_outputs_config: dict[str, Any] | StructuredOutputsConfig | None = None,
    profiler_config: dict[str, Any] | ProfilerConfig | None = None,
    attention_config: dict[str, Any] | AttentionConfig | None = None,
    kv_cache_memory_bytes: int | None = None,
    compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
    logits_processors: list[str | type[LogitsProcessor]] | None = None,
    **kwargs: Any,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | HuggingFace model name or local path |
| `runner` | `RunnerOption` | `"auto"` | Runner type: `"auto"`, `"generate"`, or `"pooling"` |
| `convert` | `ConvertOption` | `"auto"` | Model conversion: `"auto"`, `"embed"`, or `"classify"` |
| `tokenizer` | `str \| None` | `None` | Custom tokenizer name/path. If None, uses model's tokenizer |
| `tokenizer_mode` | `TokenizerMode \| str` | `"auto"` | `"auto"` (fast if available), `"slow"`, or `"mistral"` |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer init; expects `prompt_token_ids` input |
| `trust_remote_code` | `bool` | `False` | Trust remote code when downloading model/tokenizer |
| `allowed_local_media_path` | `str` | `""` | Local filesystem paths allowed for multimodal inputs |
| `allowed_media_domains` | `list[str] \| None` | `None` | Restrict multimodal URLs to these domains |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism |
| `dtype` | `ModelDType` | `"auto"` | Data type: `"auto"`, `"float32"`, `"float16"`, `"bfloat16"` |
| `quantization` | `str \| None` | `None` | Quantization method: `"awq"`, `"gptq"`, `"fp8"`, etc. |
| `revision` | `str \| None` | `None` | Model version (branch, tag, or commit id) |
| `tokenizer_revision` | `str \| None` | `None` | Tokenizer version (branch, tag, or commit id) |
| `chat_template` | `Path \| str \| None` | `None` | Custom chat template path or inline template |
| `seed` | `int` | `0` | Random seed for sampling |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory for model + KV cache (0-1) |
| `kv_cache_memory_bytes` | `int \| None` | `None` | Explicit KV cache size in bytes (overrides `gpu_memory_utilization`) |
| `cpu_offload_gb` | `float` | `0` | GiB of CPU memory for weight offloading |
| `offload_group_size` | `int` | `0` | Prefetch offloading: group every N layers. 0 = disabled |
| `offload_num_in_group` | `int` | `1` | Number of layers to offload per group |
| `offload_prefetch_step` | `int` | `1` | Layers to prefetch ahead |
| `offload_params` | `set[str] \| None` | `None` | Parameter name segments to selectively offload |
| `enforce_eager` | `bool` | `False` | Disable CUDA graph, always use eager mode |
| `enable_return_routed_experts` | `bool` | `False` | Return routed expert info (MoE models) |
| `disable_custom_all_reduce` | `bool` | `False` | Disable custom all-reduce kernel |
| `hf_token` | `bool \| str \| None` | `None` | HuggingFace auth token. `True` uses cached token |
| `hf_overrides` | `HfOverrides \| None` | `None` | Dict or callable to override HF config |
| `mm_processor_kwargs` | `dict[str, Any] \| None` | `None` | Extra kwargs for multimodal processor |
| `pooler_config` | `PoolerConfig \| None` | `None` | Custom pooling configuration for embedding/classification models |
| `structured_outputs_config` | `dict \| StructuredOutputsConfig \| None` | `None` | Structured output backend config |
| `profiler_config` | `dict \| ProfilerConfig \| None` | `None` | Profiling configuration |
| `attention_config` | `dict \| AttentionConfig \| None` | `None` | Attention backend configuration |
| `compilation_config` | `int \| dict \| CompilationConfig \| None` | `None` | Compilation optimization mode or config |
| `logits_processors` | `list[str \| type] \| None` | `None` | Custom logits processors |

**Deprecated parameters:**
- `swap_space` -- ignored with a deprecation warning. Will be removed in a future version.

**Additional engine kwargs** are passed through to `EngineArgs` via `**kwargs`. Common extras include:
- `data_parallel_size` -- data parallelism (requires external launcher)
- `pipeline_parallel_size` -- pipeline parallelism
- `max_model_len` -- override maximum sequence length
- `enable_prefix_caching` -- enable prefix caching (automatic prefix sharing)
- `generation_config` -- `"vllm"` to disable reading HuggingFace `generation_config.json`
- `enable_lora` -- enable LoRA adapter support
- `max_lora_rank` -- maximum LoRA adapter rank
- `max_loras` -- maximum number of LoRA adapters in memory
- `served_model_name` -- custom model name(s)
- `enable_log_requests` -- log incoming requests
- `disable_log_stats` -- disable periodic stats logging (default `True` for `LLM` class)

## Public Methods

---

### generate()

Generate text completions from prompts. Does NOT apply chat templates.

```python
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | `PromptType \| Sequence[PromptType]` | required | Input prompt(s). Can be strings, token ID lists, `TextPrompt`, `TokensPrompt`, or `EmbedsPrompt` dicts |
| `sampling_params` | `SamplingParams \| Sequence[SamplingParams] \| None` | `None` | Sampling config. `None` uses model's `generation_config.json` defaults |
| `use_tqdm` | `bool \| Callable` | `True` | Show progress bar. Pass a callable for custom tqdm |
| `lora_request` | `LoRARequest \| Sequence \| None` | `None` | LoRA adapter(s) to use |
| `priority` | `list[int] \| None` | `None` | Request priorities (for priority scheduling) |
| `tokenization_kwargs` | `dict[str, Any] \| None` | `None` | Override tokenizer.encode() kwargs |

**Returns:** `list[RequestOutput]` -- one per input prompt, in the same order.

**Key notes:**
- Only works when `runner_type == "generate"`. Raises `ValueError` otherwise.
- For chat conversations, use `chat()` instead -- it applies the chat template.
- Batches prompts automatically for optimal throughput.
- When `sampling_params` is a single instance, it is applied to all prompts.

---

### chat()

Generate responses from chat conversations. Applies chat templates automatically.

```python
def chat(
    self,
    messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
    chat_template: str | None = None,
    chat_template_content_format: ChatTemplateContentFormatOption = "auto",
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[dict] \| Sequence[list[dict]]` | required | Single conversation or batch of conversations. Each message has `role` and `content` keys |
| `sampling_params` | `SamplingParams \| Sequence \| None` | `None` | Sampling config |
| `use_tqdm` | `bool \| Callable` | `True` | Show progress bar |
| `lora_request` | `LoRARequest \| Sequence \| None` | `None` | LoRA adapter(s) |
| `chat_template` | `str \| None` | `None` | Override chat template. `None` uses model's default |
| `chat_template_content_format` | `str` | `"auto"` | `"string"` or `"openai"` format for message content |
| `add_generation_prompt` | `bool` | `True` | Add generation prompt marker |
| `continue_final_message` | `bool` | `False` | Continue the last message instead of starting new. Cannot be `True` when `add_generation_prompt` is `True` |
| `tools` | `list[dict] \| None` | `None` | Tool definitions for function calling |
| `chat_template_kwargs` | `dict \| None` | `None` | Extra kwargs for chat template |
| `tokenization_kwargs` | `dict \| None` | `None` | Override tokenizer kwargs |
| `mm_processor_kwargs` | `dict \| None` | `None` | Override multimodal processor kwargs |

**Returns:** `list[RequestOutput]`

**Key notes:**
- Converts conversations to text using the model's chat template, then generates.
- Multimodal inputs (images, audio, video) can be passed in OpenAI format within message `content`.
- Only works with generative models (`runner_type == "generate"`).

---

### embed()

Generate embedding vectors.

```python
def embed(
    self,
    prompts: PromptType | Sequence[PromptType],
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[EmbeddingRequestOutput]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | `PromptType \| Sequence[PromptType]` | required | Text or token prompts to embed |
| `pooling_params` | `PoolingParams \| Sequence \| None` | `None` | Pooling parameters |
| `use_tqdm` | `bool \| Callable` | `True` | Show progress bar |
| `lora_request` | `LoRARequest \| list \| None` | `None` | LoRA adapter(s) |
| `tokenization_kwargs` | `dict \| None` | `None` | Tokenizer overrides |

**Returns:** `list[EmbeddingRequestOutput]` -- each contains `.outputs.embedding` (a `list[float]`).

**Key notes:**
- Requires a pooling model (`runner_type == "pooling"`).
- Model must support the `"embed"` task. Use `--convert embed` if needed.
- Uses `pooling_task="embed"` internally.

---

### classify()

Generate classification probabilities.

```python
def classify(
    self,
    prompts: PromptType | Sequence[PromptType],
    *,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[ClassificationRequestOutput]
```

**Returns:** `list[ClassificationRequestOutput]` -- each contains `.outputs.probs` (a `list[float]`).

**Key notes:**
- Requires a pooling model supporting the `"classify"` task.
- Use `--convert classify` if model does not natively support it.
- Uses `pooling_task="classify"` internally.

---

### score()

Compute similarity scores between text pairs.

```python
def score(
    self,
    data_1: SingletonPrompt | Sequence[SingletonPrompt]
        | ScoreMultiModalParam | list[ScoreMultiModalParam],
    data_2: SingletonPrompt | Sequence[SingletonPrompt]
        | ScoreMultiModalParam | list[ScoreMultiModalParam],
    /,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    pooling_params: PoolingParams | None = None,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
) -> list[ScoringRequestOutput]
```

**Pairing modes:** `1->1`, `1->N`, or `N->N`. In `1->N` mode, `data_1` is replicated.

**Scoring strategies (determined automatically by model):**
- **Cross-encoder:** Concatenates pairs and classifies (uses `chat_template` if provided)
- **Embedding:** Encodes separately and computes cosine similarity
- **Late interaction (ColBERT):** Computes MaxSim between per-token embeddings

**Returns:** `list[ScoringRequestOutput]` -- each contains `.outputs.score` (a `float`).

**Key notes:**
- Requires a pooling model with `"embed"` or `"classify"` task support.
- The `score` pooling task is deprecated in v0.19.0 and will be removed in v0.20. Use `"classify"` instead.

---

### reward()

Generate reward scores (token-level classification).

```python
def reward(
    self,
    prompts: PromptType | Sequence[PromptType],
    /,
    *,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[PoolingRequestOutput]
```

**Returns:** `list[PoolingRequestOutput]`

**Key notes:**
- Uses `pooling_task="token_classify"` internally.
- New in v0.19.0.

---

### encode()

Low-level pooling method that supports all pooling tasks.

```python
def encode(
    self,
    prompts: PromptType | Sequence[PromptType] | DataPrompt,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    pooling_task: PoolingTask | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[PoolingRequestOutput]
```

**Supported `pooling_task` values:**
- `"embed"` -- embeddings
- `"classify"` -- classification
- `"token_embed"` -- per-token embeddings (multi-vector retrieval)
- `"token_classify"` -- per-token classification (rewards)
- `"plugin"` -- IO processor plugin task

**Key notes:**
- `pooling_task` is required. Use `embed()`, `classify()`, `score()`, or `reward()` for convenience.
- Also supports `DataPrompt` inputs for IO processor plugins.

---

### beam_search()

Generate sequences using beam search.

```python
def beam_search(
    self,
    prompts: list[TokensPrompt | TextPrompt],
    params: BeamSearchParams,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    use_tqdm: bool = False,
    concurrency_limit: int | None = None,
) -> list[BeamSearchOutput]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | `list[TokensPrompt \| TextPrompt]` | required | Input prompts |
| `params` | `BeamSearchParams` | required | Beam search configuration |
| `lora_request` | `LoRARequest \| list \| None` | `None` | LoRA adapter(s) |
| `use_tqdm` | `bool` | `False` | Show progress bar |
| `concurrency_limit` | `int \| None` | `None` | Max concurrent requests. `None` = unlimited |

**Returns:** `list[BeamSearchOutput]` -- each contains `.sequences`, a list of `BeamSearchSequence` with `.text`, `.tokens`, `.cum_logprob`.

---

### enqueue() and wait_for_completion()

Separate request submission from result collection.

```python
# Enqueue requests without blocking
def enqueue(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
    use_tqdm: bool | Callable[..., tqdm] = True,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[str]  # returns request IDs

# Process all queued requests and get results
def wait_for_completion(
    self,
    output_type: type | tuple[type, ...] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[RequestOutput | PoolingRequestOutput]
```

**Key notes:**
- New in v0.19.0.
- `enqueue()` only works with generative models.
- Call `wait_for_completion()` after `enqueue()` to process and retrieve results.

---

### Utility Methods

```python
def get_tokenizer(self) -> TokenizerLike
```
Returns the underlying tokenizer.

```python
def get_world_size(self, include_dp: bool = True) -> int
```
Returns TP * PP (optionally * DP) world size.

```python
def get_default_sampling_params(self) -> SamplingParams
```
Returns default params (from model's `generation_config.json` if available).

```python
def get_metrics(self) -> list[Metric]
```
Returns a snapshot of aggregated Prometheus metrics (V1 engine only).

```python
def reset_mm_cache(self) -> None
```
Clear the multimodal data cache.

```python
def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool
```
Reset the prefix cache.

---

### Model Lifecycle Methods

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort")
```
Put the engine to sleep.
- **Level 0:** Pause scheduling, queue requests.
- **Level 1:** Offload weights to CPU, discard KV cache.
- **Level 2:** Discard all GPU memory (weights + KV cache).
- **mode:** `"abort"`, `"wait"`, or `"keep"` for existing requests.

```python
def wake_up(self, tags: list[str] | None = None)
```
Wake from sleep. Optional `tags`: `["weights", "kv_cache", "scheduling"]`.

```python
def start_profile(self, profile_prefix: str | None = None) -> None
```
Start GPU profiling.

```python
def stop_profile(self) -> None
```
Stop GPU profiling.

---

### Advanced Methods

```python
def collective_rpc(
    self,
    method: str | Callable[..., _R],
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[_R]
```
Execute an RPC call on all workers. Use for control messages.

```python
def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]
```
Run a function directly on the model in each worker. Avoid returning large tensors.

```python
def init_weight_transfer_engine(self, request: WeightTransferInitRequest | dict) -> None
def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None
```
Weight transfer methods for RL training integration.

---

## PromptType Variants

The `PromptType` accepted by `generate()` and other methods supports these formats:

| Format | Example |
|---|---|
| Plain string | `"Hello, world"` |
| Token IDs list | `[15496, 11, 995]` |
| TextPrompt dict | `{"prompt": "Hello", "multi_modal_data": {...}}` |
| TokensPrompt dict | `{"prompt_token_ids": [15496, 11], "multi_modal_data": {...}}` |
| EmbedsPrompt dict | `{"prompt_embeds": tensor}` |

For multimodal inputs, use the `TextPrompt` or `TokensPrompt` dict form with `multi_modal_data`:

```python
{"prompt": "<image>Describe this image",
 "multi_modal_data": {"image": pil_image}}
```
