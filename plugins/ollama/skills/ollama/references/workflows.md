# Ollama Workflow Examples

Working examples using curl. All assume Ollama is running on `http://localhost:11434`.

## Quick Start: Pull and Generate

```bash
# 1. Pull a model
curl http://localhost:11434/api/pull -d '{"model": "llama3.2", "stream": false}'

# 2. Generate text
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Explain quantum computing in one paragraph.",
  "stream": false
}'
```

## Streaming vs Non-Streaming

### Streaming (default)

Streaming returns one JSON object per token. Useful for showing output progressively.

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Write a haiku about coding."
}'
```

Each response is a separate JSON line:
```json
{"model":"llama3.2","created_at":"...","response":"Lines","done":false}
{"model":"llama3.2","created_at":"...","response":" of","done":false}
{"model":"llama3.2","created_at":"...","response":" code","done":false}
```

The final object has `"done": true` with timing metrics.

### Non-Streaming

Get the complete response in a single JSON object:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Write a haiku about coding.",
  "stream": false
}'
```

## Multi-Turn Chat

Maintain conversation history by including all previous messages:

```bash
# Turn 1
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is the derivative of x^3?"}
  ],
  "stream": false
}'

# Turn 2 - include the assistant's previous response
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is the derivative of x^3?"},
    {"role": "assistant", "content": "The derivative of x^3 is 3x^2."},
    {"role": "user", "content": "What about the second derivative?"}
  ],
  "stream": false
}'
```

## Tool Calling

### Step 1: Send request with tools

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is the weather in San Francisco?"}
  ],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }
  ]
}'
```

Response (model requests tool call):
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_weather",
          "arguments": {"city": "San Francisco", "units": "fahrenheit"}
        }
      }
    ]
  },
  "done": true
}
```

### Step 2: Execute tool and send result back

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is the weather in San Francisco?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "function": {
            "name": "get_weather",
            "arguments": {"city": "San Francisco", "units": "fahrenheit"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "72°F, sunny with light clouds",
      "tool_name": "get_weather"
    }
  ],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }
  ]
}'
```

Response (natural language summary):
```json
{
  "message": {
    "role": "assistant",
    "content": "The current weather in San Francisco is 72°F (22°C), sunny with light clouds."
  },
  "done": true
}
```

## Structured Output (JSON Schema)

Force the model to return data matching a specific schema:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1",
  "messages": [
    {"role": "user", "content": "List 3 programming languages with their year of creation and primary use case."}
  ],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "languages": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "year": {"type": "integer"},
            "use_case": {"type": "string"}
          },
          "required": ["name", "year", "use_case"]
        }
      }
    },
    "required": ["languages"]
  }
}'
```

The `message.content` will be a valid JSON string matching the schema:
```json
{
  "languages": [
    {"name": "Python", "year": 1991, "use_case": "general purpose scripting and data science"},
    {"name": "Rust", "year": 2010, "use_case": "systems programming with memory safety"},
    {"name": "Go", "year": 2009, "use_case": "cloud infrastructure and backend services"}
  ]
}
```

## JSON Mode (Simple)

For simple JSON output without a specific schema:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What color is the sky at different times of day? Respond using JSON.",
  "format": "json",
  "stream": false
}'
```

Note: Always instruct the model to use JSON in the prompt when using `"format": "json"`, otherwise it may generate excessive whitespace.

## RAG / Embeddings Workflow

### Step 1: Generate embeddings for your documents

```bash
# Embed multiple documents at once
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": [
    "Ollama runs large language models locally on your machine.",
    "Docker is a platform for containerizing applications.",
    "Kubernetes orchestrates container deployments at scale."
  ]
}'
```

### Step 2: Embed the user query

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "all-minilm",
  "input": "How do I run LLMs on my laptop?"
}'
```

### Step 3: Find relevant documents

Compute cosine similarity between the query embedding and document embeddings in your application code, then retrieve the top-k most similar documents.

### Step 4: Generate response with context

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {
      "role": "system",
      "content": "Answer questions using only the provided context. If the context does not contain the answer, say so."
    },
    {
      "role": "user",
      "content": "Context: Ollama runs large language models locally on your machine.\n\nQuestion: How do I run LLMs on my laptop?"
    }
  ],
  "stream": false
}'
```

## Custom Model Creation with Modelfile

### Via CLI

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
SYSTEM You are a senior software engineer. Give concise, accurate technical answers.

MESSAGE user How should I handle errors in Go?
MESSAGE assistant Use explicit error returns: if err != nil { return fmt.Errorf("context: %w", err) }. Wrap errors for context, use errors.Is/As for checking.
EOF

# Create the model
ollama create code-helper -f Modelfile

# Run it
ollama run code-helper
```

### Via API

```bash
curl http://localhost:11434/api/create -d '{
  "model": "code-helper",
  "from": "llama3.2",
  "system": "You are a senior software engineer. Give concise, accurate technical answers.",
  "parameters": {
    "temperature": 0.7,
    "num_ctx": 8192
  },
  "stream": false
}'
```

## Image/Vision Input

Use multimodal models like `llava` to analyze images:

```bash
# Base64-encode an image
IMAGE_B64=$(base64 -w0 photo.jpg)

# Ask about the image via generate
curl http://localhost:11434/api/generate -d "{
  \"model\": \"llava\",
  \"prompt\": \"Describe what you see in this image in detail.\",
  \"images\": [\"$IMAGE_B64\"],
  \"stream\": false
}"

# Or via chat (supports multi-turn with images)
curl http://localhost:11434/api/chat -d "{
  \"model\": \"llava\",
  \"messages\": [
    {
      \"role\": \"user\",
      \"content\": \"What objects are in this image?\",
      \"images\": [\"$IMAGE_B64\"]
    }
  ],
  \"stream\": false
}"
```

## Thinking Models

Enable chain-of-thought reasoning with `think`:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1",
  "messages": [
    {"role": "user", "content": "How many r letters are in the word strawberry?"}
  ],
  "think": true,
  "stream": false
}'
```

The response will include a `thinking` field with the reasoning steps:
```json
{
  "message": {
    "role": "assistant",
    "content": "There are 3 r letters in the word strawberry.",
    "thinking": "Let me spell it out: s-t-r-a-w-b-e-r-r-y. I see r at positions 3, 8, and 9. So there are 3 r letters."
  },
  "done": true
}
```

You can also use think levels: `"think": "high"`, `"think": "medium"`, `"think": "low"`.

## Model Loading and Memory Management

### Preload a model

```bash
curl http://localhost:11434/api/generate -d '{"model": "llama3.2"}'
```

### Keep a model loaded indefinitely

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "keep_alive": -1
}'
```

### Unload a model

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "keep_alive": 0
}'
```

### Check what is loaded

```bash
curl http://localhost:11434/api/ps
```

## Reproducible Outputs

Set `seed` and low `temperature` for deterministic results:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is 2+2?",
  "options": {
    "seed": 42,
    "temperature": 0
  },
  "stream": false
}'
```

## Configuring Runtime Options

Pass any model parameters via the `options` object:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Write a creative story.",
  "stream": false,
  "options": {
    "temperature": 1.2,
    "top_k": 100,
    "top_p": 0.95,
    "num_ctx": 4096,
    "num_predict": 500,
    "repeat_penalty": 1.3,
    "stop": ["\n\nThe End"]
  }
}'
```
