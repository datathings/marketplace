# Embeddings API

## POST /api/embed -- Generate Embeddings

Generate embeddings from a model. Supports single or batch input.

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name (e.g., `"all-minilm"`, `"nomic-embed-text"`) |
| `input` | string or string[] | Yes | Text or list of texts to embed |
| `truncate` | bool | No | Truncate input to fit context length (default: `true`). Returns error if `false` and context exceeded. |
| `options` | object | No | Runtime model parameters (e.g., temperature) |
| `keep_alive` | string or number | No | How long to keep model loaded (default: `"5m"`) |
| `dimensions` | int | No | Truncate embedding output to this many dimensions |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name used |
| `embeddings` | float32[][] | Array of embedding vectors |
| `total_duration` | int | Total time (nanoseconds) |
| `load_duration` | int | Model loading time (nanoseconds) |
| `prompt_eval_count` | int | Number of tokens processed |

### Example: Single Input

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": "Why is the sky blue?"
}'
```

Response:
```json
{
  "model": "all-minilm",
  "embeddings": [
    [
      0.010071029, -0.0017594862, 0.05007221, 0.04692972, 0.054916814,
      0.008599704, 0.105441414, -0.025878139, 0.12958129, 0.031952348
    ]
  ],
  "total_duration": 14143917,
  "load_duration": 1019500,
  "prompt_eval_count": 8
}
```

### Example: Multiple Inputs (Batch)

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": ["Why is the sky blue?", "Why is the grass green?"]
}'
```

Response:
```json
{
  "model": "all-minilm",
  "embeddings": [
    [
      0.010071029, -0.0017594862, 0.05007221, 0.04692972, 0.054916814,
      0.008599704, 0.105441414, -0.025878139, 0.12958129, 0.031952348
    ],
    [
      -0.0098027075, 0.06042469, 0.025257962, -0.006364387, 0.07272725,
      0.017194884, 0.09032035, -0.051705178, 0.09951512, 0.09072481
    ]
  ]
}
```

### Example: With Dimensions

Reduce embedding dimensionality (useful for storage/performance):

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": "Why is the sky blue?",
  "dimensions": 128
}'
```

---

## POST /api/embeddings -- Generate Embedding (Legacy/Deprecated)

This endpoint has been superseded by `/api/embed`. It only accepts a single text string (not batch).

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name |
| `prompt` | string | Yes | Text to embed (single string only) |
| `options` | object | No | Runtime model parameters |
| `keep_alive` | string or number | No | How long to keep model loaded (default: `"5m"`) |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | float64[] | Single embedding vector |

Note: The legacy endpoint returns `embedding` (singular, float64 array) while the current endpoint returns `embeddings` (plural, array of float32 arrays).

### Example

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "all-minilm",
  "prompt": "Here is an article about llamas..."
}'
```

Response:
```json
{
  "embedding": [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074,
    -0.2916173040866852, -0.8924556970596313, 0.8785552978515625,
    -0.34576427936553955, 0.5742510557174683, -0.04222835972905159,
    -0.137906014919281
  ]
}
```

## Migration from /api/embeddings to /api/embed

| Feature | `/api/embeddings` (deprecated) | `/api/embed` (current) |
|---------|-------------------------------|----------------------|
| Input field | `prompt` (string) | `input` (string or string[]) |
| Batch support | No | Yes |
| Response field | `embedding` (float64[]) | `embeddings` (float32[][]) |
| Truncation control | No | Yes (`truncate` param) |
| Dimension control | No | Yes (`dimensions` param) |
| Timing metrics | No | Yes (`total_duration`, `load_duration`, `prompt_eval_count`) |
