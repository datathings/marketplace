# SamplingParams, Output Types, and Related Classes

## SamplingParams

Controls text generation behavior. Follows the OpenAI text completion API conventions.

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `1` | Number of output sequences per prompt. Max controlled by `VLLM_MAX_N_SEQUENCES` env var (default 16384) |
| `temperature` | `float` | `1.0` | Randomness control. 0 = greedy (deterministic). Higher = more random |
| `top_p` | `float` | `1.0` | Nucleus sampling: cumulative probability cutoff. Must be in (0, 1] |
| `top_k` | `int` | `0` | Top-k sampling. 0 or -1 = disabled (consider all tokens) |
| `min_p` | `float` | `0.0` | Minimum probability relative to most likely token. [0, 1] |
| `presence_penalty` | `float` | `0.0` | Penalty for tokens already in output. Range [-2, 2]. Positive = encourage new tokens |
| `frequency_penalty` | `float` | `0.0` | Penalty based on token frequency in output. Range [-2, 2]. Positive = encourage new tokens |
| `repetition_penalty` | `float` | `1.0` | Penalty for tokens in prompt + output. Values > 1 encourage new tokens |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. -1 treated as None |
| `stop` | `str \| list[str] \| None` | `None` | Stop string(s). Generation stops when any is produced |
| `stop_token_ids` | `list[int] \| None` | `None` | Stop token IDs. Output includes stop tokens unless they are special tokens |
| `ignore_eos` | `bool` | `False` | Continue generating past EOS token |
| `max_tokens` | `int \| None` | `16` | Maximum tokens to generate. `None` for unlimited |
| `min_tokens` | `int` | `0` | Minimum tokens before EOS/stop can trigger |
| `logprobs` | `int \| None` | `None` | Number of log probabilities per output token. `None` = disabled, `-1` = all vocab |
| `prompt_logprobs` | `int \| None` | `None` | Log probabilities per prompt token. `-1` = all vocab |
| `flat_logprobs` | `bool` | `False` | Return logprobs in flat format (FlatLogprobs) for better performance |
| `detokenize` | `bool` | `True` | Whether to detokenize output |
| `skip_special_tokens` | `bool` | `True` | Skip special tokens in output |
| `spaces_between_special_tokens` | `bool` | `True` | Add spaces between special tokens |
| `include_stop_str_in_output` | `bool` | `False` | Include stop strings in output text |
| `output_kind` | `RequestOutputKind` | `CUMULATIVE` | Output mode: `CUMULATIVE`, `DELTA`, or `FINAL_ONLY` |
| `structured_outputs` | `StructuredOutputsParams \| None` | `None` | Structured output constraints (JSON, regex, etc.) |
| `logit_bias` | `dict[int, float] \| None` | `None` | Token ID to bias mapping. Clamped to [-100, 100] |
| `allowed_token_ids` | `list[int] \| None` | `None` | Restrict generation to only these token IDs |
| `bad_words` | `list[str] \| None` | `None` | Words not allowed in generation |
| `extra_args` | `dict[str, Any] \| None` | `None` | Arbitrary args for custom sampling implementations |
| `thinking_token_budget` | `int \| None` | `None` | Maximum tokens for thinking operations (new in v0.19.0) |
| `repetition_detection` | `RepetitionDetectionParams \| None` | `None` | Detect and stop repetitive N-gram patterns (new in v0.19.0) |
| `skip_reading_prefix_cache` | `bool \| None` | `None` | Skip prefix cache reading (auto-set when `prompt_logprobs` is used) |

### Greedy Sampling Behavior

When `temperature < 1e-5`:
- Automatically sets `top_p=1.0`, `top_k=0`, `min_p=0.0`
- `n` must be `1`

### Static Methods

```python
SamplingParams.from_optional(
    n=1, presence_penalty=0.0, frequency_penalty=0.0,
    repetition_penalty=1.0, temperature=1.0, top_p=1.0,
    top_k=0, min_p=0.0, seed=None, stop=None,
    stop_token_ids=None, bad_words=None,
    thinking_token_budget=None,
    include_stop_str_in_output=False, ignore_eos=False,
    max_tokens=16, min_tokens=0, logprobs=None,
    prompt_logprobs=None, detokenize=True,
    skip_special_tokens=True, spaces_between_special_tokens=True,
    output_kind=RequestOutputKind.CUMULATIVE,
    structured_outputs=None, logit_bias=None,
    allowed_token_ids=None, extra_args=None,
    skip_clone=False, repetition_detection=None,
) -> SamplingParams
```
Create from optional parameters (converts None values to defaults). Also converts `logit_bias` string keys to ints.

---

## RequestOutputKind

```python
from vllm.sampling_params import RequestOutputKind

RequestOutputKind.CUMULATIVE   # Return entire output so far in every update
RequestOutputKind.DELTA        # Return only new tokens in each update
RequestOutputKind.FINAL_ONLY   # Only return the final complete output
```

---

## StructuredOutputsParams

Constrain generation to follow a specific format. Import from `vllm.sampling_params`.

```python
from vllm.sampling_params import StructuredOutputsParams

# Exactly one constraint must be specified
params = StructuredOutputsParams(
    json: str | dict | None = None,         # JSON schema
    regex: str | None = None,               # Regular expression
    choice: list[str] | None = None,        # One of these strings
    grammar: str | None = None,             # Context-free grammar (GBNF/Lark)
    json_object: bool | None = None,        # Any valid JSON object
    structural_tag: str | None = None,      # Structural tag constraint
    disable_any_whitespace: bool = False,    # Disable flexible whitespace
    disable_additional_properties: bool = False,  # Strict JSON schema
    whitespace_pattern: str | None = None,  # Custom whitespace regex
)
```

### Constraint Types (mutually exclusive)

| Constraint | Type | Description |
|---|---|---|
| `json` | `str \| dict` | JSON schema (as string or dict). Enforces valid JSON matching the schema |
| `regex` | `str` | Regular expression pattern the output must match |
| `choice` | `list[str]` | Output must be exactly one of these strings. Cannot be empty |
| `grammar` | `str` | Context-free grammar in GBNF or Lark format. Cannot be empty |
| `json_object` | `bool` | If `True`, output must be any valid JSON object |
| `structural_tag` | `str` | Structural tag constraint |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `disable_any_whitespace` | `bool` | `False` | Disable flexible whitespace handling |
| `disable_additional_properties` | `bool` | `False` | Reject extra properties in JSON schemas |
| `whitespace_pattern` | `str \| None` | `None` | Custom regex for whitespace matching |

### Backend Selection

The backend is determined by `--structured-output-backend` (server) or `structured_outputs_config` (offline):
- `"auto"` (default): Tries xgrammar, falls back to guidance, then outlines
- `"xgrammar"`: Fast grammar-based engine
- `"guidance"`: Feature-rich grammar engine (does not support Mistral tokenizers)
- `"outlines"`: Broad compatibility
- `"lm-format-enforcer"`: Alternative enforcer (does not support Mistral tokenizers)

### Usage with SamplingParams

```python
params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    structured_outputs=StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    ),
)
```

---

## RepetitionDetectionParams

New in v0.19.0. Detect and terminate repetitive N-gram patterns early.

```python
from vllm.sampling_params import RepetitionDetectionParams

params = SamplingParams(
    max_tokens=4096,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=10,    # Max N-gram size to check
        min_pattern_size=1,     # Min N-gram size to check (default: 1)
        min_count=3,            # Min repetitions to trigger (must be >= 2)
    ),
)
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `max_pattern_size` | `int` | `0` | Max N-gram pattern size. 0 = disabled |
| `min_pattern_size` | `int` | `0` | Min pattern size. Defaults to 1 if 0 |
| `min_count` | `int` | `0` | Min repeat count to trigger. Must be >= 2 when enabled |

---

## BeamSearchParams

Configuration for beam search generation.

```python
from vllm.sampling_params import BeamSearchParams

params = BeamSearchParams(
    beam_width=4,
    max_tokens=256,
    temperature=0.0,
    length_penalty=1.0,
    ignore_eos=False,
    include_stop_str_in_output=False,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `beam_width` | `int` | required | Number of beams |
| `max_tokens` | `int` | required | Maximum tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `length_penalty` | `float` | `1.0` | Length normalization penalty |
| `ignore_eos` | `bool` | `False` | Continue past EOS |
| `include_stop_str_in_output` | `bool` | `False` | Include stop strings |

---

## Output Types

### RequestOutput

Returned by `generate()` and `chat()`.

```python
class RequestOutput:
    request_id: str                           # Unique request ID
    prompt: str | None                        # Original prompt text
    prompt_token_ids: list[int] | None        # Prompt token IDs
    prompt_logprobs: PromptLogprobs | None     # Prompt log probabilities
    outputs: list[CompletionOutput]           # Generated completions (one per `n`)
    finished: bool                            # Whether request is complete
    metrics: RequestStateStats | None         # Performance metrics
    lora_request: LoRARequest | None          # LoRA adapter used
    encoder_prompt: str | None                # Encoder prompt (enc-dec models)
    encoder_prompt_token_ids: list[int] | None  # Encoder prompt tokens
    num_cached_tokens: int | None             # Prefix cache hits
    kv_transfer_params: dict | None           # Remote KV transfer params
```

### CompletionOutput

One generated sequence within a `RequestOutput`.

```python
@dataclass
class CompletionOutput:
    index: int                                # Output index (0 to n-1)
    text: str                                 # Generated text
    token_ids: Sequence[int]                  # Generated token IDs
    cumulative_logprob: float | None          # Cumulative log probability
    logprobs: SampleLogprobs | None           # Per-token log probabilities
    routed_experts: np.ndarray | None         # Routed experts (MoE)
    finish_reason: str | None                 # "stop", "length", or None
    stop_reason: int | str | None             # Stop token/string that triggered finish
    lora_request: LoRARequest | None          # LoRA adapter used

    def finished(self) -> bool                # Check if generation is complete
```

### Logprob

Per-token log probability information.

```python
@dataclass
class Logprob:
    logprob: float                # Log probability of the token
    rank: int | None = None       # Vocabulary rank (>= 1)
    decoded_token: str | None = None  # Decoded token string
```

**Type aliases:**
- `LogprobsOnePosition = dict[int, Logprob]` -- token_id to Logprob mapping for one position
- `SampleLogprobs = list[LogprobsOnePosition]` -- log probs for all generated positions
- `PromptLogprobs = list[LogprobsOnePosition | None]` -- log probs for prompt positions

### PoolingRequestOutput

Base output for all pooling operations.

```python
class PoolingRequestOutput(Generic[_O]):
    request_id: str                    # Unique request ID
    outputs: _O                        # Pooling output (varies by task)
    prompt_token_ids: list[int]        # Prompt token IDs
    num_cached_tokens: int             # Prefix cache hits
    finished: bool                     # Whether request is complete
```

### EmbeddingOutput / EmbeddingRequestOutput

Returned by `embed()`.

```python
@dataclass
class EmbeddingOutput:
    embedding: list[float]            # Embedding vector

    @property
    def hidden_size(self) -> int      # Embedding dimension

class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    # .outputs is an EmbeddingOutput
    pass
```

### ClassificationOutput / ClassificationRequestOutput

Returned by `classify()`.

```python
@dataclass
class ClassificationOutput:
    probs: list[float]                # Probability vector

    @property
    def num_classes(self) -> int       # Number of classes

class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    # .outputs is a ClassificationOutput
    pass
```

### ScoringOutput / ScoringRequestOutput

Returned by `score()`.

```python
@dataclass
class ScoringOutput:
    score: float                       # Similarity score

class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    # .outputs is a ScoringOutput
    pass
```

### PoolingOutput

Base pooling output with raw tensor data.

```python
@dataclass
class PoolingOutput:
    data: torch.Tensor                 # Raw pooled hidden states
```

---

## PoolingParams

Parameters for pooling models (embeddings, classification, scoring).

```python
from vllm import PoolingParams

params = PoolingParams(
    use_activation: bool | None = None,      # Apply activation function
    dimensions: int | None = None,           # Output embedding dimensions (matryoshka)
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `use_activation` | `bool \| None` | `None` | Apply activation. None uses model default (usually True) |
| `dimensions` | `int \| None` | `None` | Reduce embedding dimensions (matryoshka models only) |
| `step_tag_id` | `int \| None` | `None` | Step tag token ID (step pooling models) |
| `returned_token_ids` | `list[int] \| None` | `None` | Token IDs to return (step pooling) |
| `task` | `PoolingTask \| None` | `None` | Override pooling task |
