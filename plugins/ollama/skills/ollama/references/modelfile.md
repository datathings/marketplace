# Modelfile Reference

A Modelfile is the blueprint for creating and sharing customized models using Ollama. It defines the base model, parameters, templates, system prompts, and more.

## Format

```
# comment
INSTRUCTION arguments
```

Instructions are not case-sensitive, but uppercase is conventional. Instructions can appear in any order, though `FROM` is typically first.

## Instructions Summary

| Instruction | Required | Description |
|-------------|----------|-------------|
| `FROM` | Yes | Base model to use |
| `PARAMETER` | No | Set model runtime parameters |
| `TEMPLATE` | No | Prompt template (Go template syntax) |
| `SYSTEM` | No | System message for the template |
| `ADAPTER` | No | LoRA adapter to apply |
| `LICENSE` | No | License text |
| `MESSAGE` | No | Predefined message history |
| `REQUIRES` | No | Minimum Ollama version required |

## FROM (Required)

Defines the base model.

### From an existing model

```
FROM llama3.2
FROM llama3.2:70b
```

### From a safetensors directory

```
FROM /path/to/safetensor/directory
```

Supported architectures: Llama (1/2/3/3.1/3.2), Mistral (1/2, Mixtral), Gemma (1/2), Phi3.

### From a GGUF file

```
FROM ./ollama-model.gguf
```

Path can be absolute or relative to the Modelfile location.

## PARAMETER

Set runtime parameters for the model.

```
PARAMETER <parameter> <value>
```

### Valid Parameters and Values

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ctx` | int | 2048 | Context window size |
| `repeat_last_n` | int | 64 | Lookback for repeat penalty (0=disabled, -1=num_ctx) |
| `repeat_penalty` | float | 1.1 | Repetition penalty strength |
| `temperature` | float | 0.8 | Sampling temperature (higher=more creative) |
| `seed` | int | 0 | Random seed for reproducibility |
| `stop` | string | (none) | Stop sequence (use multiple PARAMETER lines for multiple stops) |
| `num_predict` | int | -1 | Max tokens to generate (-1=infinite) |
| `top_k` | int | 40 | Top-K sampling (higher=more diverse) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `min_p` | float | 0.0 | Minimum probability relative to most likely token |

### Examples

```
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER seed 42
PARAMETER num_predict 256
```

## TEMPLATE

The full prompt template passed to the model. Uses Go [template syntax](https://pkg.go.dev/text/template).

### Template Variables

| Variable | Description |
|----------|-------------|
| `{{ .System }}` | System message |
| `{{ .Prompt }}` | User prompt |
| `{{ .Response }}` | Model response (text after this is omitted during generation) |

### Example

```
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

## SYSTEM

Set the system message used in the template.

```
SYSTEM """You are a helpful assistant that responds concisely."""
```

## ADAPTER

Apply a fine-tuned LoRA adapter to the base model.

### Safetensor adapter

```
ADAPTER /path/to/safetensor/adapter
```

Supported: Llama (2/3/3.1), Mistral (1/2, Mixtral), Gemma (1/2).

### GGUF adapter

```
ADAPTER ./ollama-lora.gguf
```

The base model specified with `FROM` must match the model the adapter was trained on.

## LICENSE

Specify the legal license for the model.

```
LICENSE """
MIT License

Copyright (c) 2024 ...
"""
```

## MESSAGE

Define message history to guide the model's response style. Use multiple MESSAGE instructions to build a conversation.

```
MESSAGE <role> <message>
```

### Valid Roles

| Role | Description |
|------|-------------|
| `system` | Alternate way to set system message |
| `user` | Example user input |
| `assistant` | Example model response |

### Example

```
MESSAGE user Is Toronto in Canada?
MESSAGE assistant yes
MESSAGE user Is Sacramento in Canada?
MESSAGE assistant no
MESSAGE user Is Ontario in Canada?
MESSAGE assistant yes
```

## REQUIRES

Specify the minimum Ollama version required to run this model.

```
REQUIRES 0.14.0
```

## Complete Modelfile Example

```
FROM llama3.2

# Set runtime parameters
PARAMETER temperature 1
PARAMETER num_ctx 4096
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"

# Custom system prompt
SYSTEM You are Mario from super mario bros, acting as an assistant.

# Few-shot examples
MESSAGE user What is your favorite food?
MESSAGE assistant I love mushrooms! They make me grow big and strong!
```

### Usage

1. Save the file (e.g., `Modelfile`)
2. Create the model: `ollama create mario -f ./Modelfile`
3. Run it: `ollama run mario`
4. View a model's Modelfile: `ollama show --modelfile llama3.2`

## Creating Models via API

Instead of using a Modelfile on disk, you can create models via the `/api/create` endpoint:

```bash
curl http://localhost:11434/api/create -d '{
  "model": "mario",
  "from": "llama3.2",
  "system": "You are Mario from Super Mario Bros.",
  "parameters": {
    "temperature": 1,
    "num_ctx": 4096
  }
}'
```

See the [Model Management API](api-models.md) for full details on `/api/create`.
