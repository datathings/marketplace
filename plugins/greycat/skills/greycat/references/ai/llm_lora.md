# LoraAdapter - LoRA Adapter API

The LoraAdapter type provides support for LoRA (Low-Rank Adaptation) adapters, which allow fine-tuning of models without modifying the base weights.

## Overview

LoRA adapters are small files that contain task-specific adjustments to a model. Multiple adapters can be loaded and applied with different scaling factors to customize model behavior.

Use cases:
- Apply domain-specific fine-tuning (medical, legal, code, etc.)
- Switch between different task adaptations
- Combine multiple adapters for multi-task models
- A/B test different fine-tuning approaches

## Quick Start

```gcl
var model = Model::load("base", "./base-model.gguf", null);
var ctx = Context::create(model, null);

// Load LoRA adapter
var lora = LoraAdapter::load(model, "./medical-lora.gguf", 1.0, null);

// Apply adapter to context
ctx.apply_lora_adapter(lora, 1.0);

// Use context with adapter applied
// ... inference code ...

// Remove adapter from context
ctx.remove_lora_adapter(lora);

// Or clear all adapters
ctx.clear_lora_adapters();
```

## Types

### LoraAdapter

Represents a loaded LoRA adapter that can be applied to contexts. The adapter remains active until removed or the model is freed.

## Static Methods

### `static fn load(model: Model, path: String, scale: float, n_threads: int?): LoraAdapter`

Load and apply LoRA adapter to model.

Loads a LoRA adapter from file and applies it to the model with the specified scaling factor.

**Parameters:**
- `model` - Base model to apply adapter to
- `path` - Path to LoRA adapter file (.gguf format)
- `scale` - Scaling factor (0.0-1.0, typically 1.0)
- `n_threads` - Number of threads for loading (null = auto)

**Returns:** LoraAdapter instance or null if loading fails

**Example:**
```gcl
var lora = LoraAdapter::load(model, "./lora-adapter.gguf", 1.0, null);
if (lora == null) {
    println("Failed to load adapter");
}
```

## Instance Methods - LoRA Management

### `fn path(): String`

Get adapter path.

Returns the file path this adapter was loaded from.

**Returns:** File path string

**Example:**
```gcl
println("Adapter loaded from: ${lora.path()}");
```

### `fn scale(): float`

Get scaling factor.

Returns the current scaling factor.

**Returns:** Scaling factor (0.0-1.0)

**Example:**
```gcl
println("Current scale: ${lora.scale()}");
```

### `fn set_scale(new_scale: float)`

Update scaling factor.

Changes the scaling factor. You'll need to re-apply the adapter to contexts using `Context.apply_lora_adapter()` for changes to take effect.

**Parameters:**
- `new_scale` - New scaling factor (0.0-1.0)

**Example:**
```gcl
lora.set_scale(0.5);  // Reduce adapter strength to 50%

// Re-apply to context for changes to take effect
ctx.remove_lora_adapter(lora);
ctx.apply_lora_adapter(lora, 0.5);
```

## Instance Methods - Adapter Metadata

### `fn meta(key: String): String?`

Get adapter metadata value by key.

Retrieves a specific metadata field from the LoRA adapter file.

**Parameters:**
- `key` - Metadata key

**Returns:** Metadata value or null if key doesn't exist

**Example:**
```gcl
var name = lora.meta("adapter.name");
var author = lora.meta("adapter.author");
```

### `fn meta_count(): int`

Get number of adapter metadata key/value pairs.

Returns the total count of metadata entries in the adapter.

**Returns:** Metadata entry count

### `fn meta_key_by_index(index: int): String?`

Get adapter metadata key name by index.

Retrieves the key name at the specified index (0-based).

**Parameters:**
- `index` - Index (0-based)

**Returns:** Key name or null if out of bounds

**Example:**
```gcl
for (var i = 0; i < lora.meta_count(); i++) {
    var key = lora.meta_key_by_index(i);
    var val = lora.meta_val_by_index(i);
    println("${key}: ${val}");
}
```

### `fn meta_val_by_index(index: int): String?`

Get adapter metadata value by index.

Retrieves the value at the specified index (0-based).

**Parameters:**
- `index` - Index (0-based)

**Returns:** Value or null if out of bounds

## Instance Methods - ALORA Support

### `fn alora_invocation_token_count(): int`

Get number of ALORA invocation tokens.

For adapters using ALORA (activation-aware LoRA), returns the count of invocation tokens. Returns 0 if not an ALORA adapter.

**Returns:** Number of invocation tokens

**Example:**
```gcl
var count = lora.alora_invocation_token_count();
if (count > 0) {
    println("This is an ALORA adapter with ${count} invocation tokens");
}
```

### `fn alora_invocation_tokens(): Array<int>`

Get ALORA invocation tokens.

Returns the array of token IDs used to invoke this ALORA adapter. Returns empty array if not an ALORA adapter.

**Returns:** Array of token IDs

**Example:**
```gcl
var tokens = lora.alora_invocation_tokens();
for (var token in tokens) {
    println("Invocation token: ${model.token_to_text(token)}");
}
```

## Instance Methods - Resource Management

### `fn free()`

Free adapter resources.

Removes adapter and releases memory. Optional - GC handles cleanup.

**Example:**
```gcl
lora.free();
```

## Common Use Cases

### Basic LoRA Application

```gcl
var model = Model::load("base", "./base-model.gguf", null);
var ctx = Context::create(model, ContextParams { n_ctx: 2048 });

// Load medical domain adapter
var medical_lora = LoraAdapter::load(model, "./medical-lora.gguf", 1.0, null);

// Apply to context
ctx.apply_lora_adapter(medical_lora, 1.0);

// Generate medical text
var result = model.generate("Symptoms of diabetes include", null, null);
println(result.text);

// Clean up
ctx.remove_lora_adapter(medical_lora);
```

### Multiple Adapters

```gcl
var model = Model::load("base", "./model.gguf", null);
var ctx = Context::create(model, null);

// Load multiple adapters
var code_lora = LoraAdapter::load(model, "./code-lora.gguf", 1.0, null);
var math_lora = LoraAdapter::load(model, "./math-lora.gguf", 1.0, null);

// Apply both adapters
ctx.apply_lora_adapter(code_lora, 1.0);
ctx.apply_lora_adapter(math_lora, 0.5);  // 50% strength for math

// Model now has both adaptations applied
var result = model.generate("Write a function to calculate", null, null);
println(result.text);

// Remove specific adapter
ctx.remove_lora_adapter(math_lora);

// Or clear all adapters
ctx.clear_lora_adapters();
```

### Adapter Scaling

```gcl
var model = Model::load("base", "./model.gguf", null);
var ctx = Context::create(model, null);

var lora = LoraAdapter::load(model, "./lora.gguf", 1.0, null);

// Try different strengths
for (var scale in [0.25, 0.5, 0.75, 1.0]) {
    ctx.clear_lora_adapters();
    ctx.apply_lora_adapter(lora, scale);

    var result = model.generate("Test prompt", GenerationParams { max_tokens: 50 }, null);
    println("Scale ${scale}: ${result.text}");
}
```

### Adapter Metadata Inspection

```gcl
var lora = LoraAdapter::load(model, "./lora.gguf", 1.0, null);

println("Adapter path: ${lora.path()}");
println("Adapter scale: ${lora.scale()}");
println("Metadata:");

for (var i = 0; i < lora.meta_count(); i++) {
    var key = lora.meta_key_by_index(i);
    var val = lora.meta_val_by_index(i);
    println("  ${key}: ${val}");
}

// Check specific metadata
var adapter_name = lora.meta("adapter.name");
if (adapter_name != null) {
    println("Adapter name: ${adapter_name}");
}
```

### ALORA Adapter Usage

```gcl
var model = Model::load("base", "./model.gguf", null);
var ctx = Context::create(model, null);

var alora = LoraAdapter::load(model, "./alora-adapter.gguf", 1.0, null);

// Check if it's an ALORA adapter
var token_count = alora.alora_invocation_token_count();
if (token_count > 0) {
    println("ALORA adapter detected");

    var tokens = alora.alora_invocation_tokens();
    println("Invocation tokens:");
    for (var token in tokens) {
        println("  ${model.token_to_text(token)}");
    }

    // Apply adapter
    ctx.apply_lora_adapter(alora, 1.0);

    // The adapter will be activated when invocation tokens are encountered
}
```

### Dynamic Adapter Switching

```gcl
var model = Model::load("base", "./model.gguf", null);
var ctx = Context::create(model, null);

// Load multiple task-specific adapters
var adapters = Map<String, LoraAdapter>();
adapters["medical"] = LoraAdapter::load(model, "./medical-lora.gguf", 1.0, null);
adapters["legal"] = LoraAdapter::load(model, "./legal-lora.gguf", 1.0, null);
adapters["code"] = LoraAdapter::load(model, "./code-lora.gguf", 1.0, null);

fn generate_with_adapter(task: String, prompt: String): String {
    // Clear previous adapters
    ctx.clear_lora_adapters();

    // Apply task-specific adapter
    if (adapters.contains(task)) {
        ctx.apply_lora_adapter(adapters[task], 1.0);
    }

    // Generate
    var result = model.generate(prompt, GenerationParams { max_tokens: 100 }, null);
    return result.text;
}

// Use different adapters for different tasks
println("Medical: ${generate_with_adapter('medical', 'Symptoms of flu:')}");
println("Legal: ${generate_with_adapter('legal', 'Contract terms:')}");
println("Code: ${generate_with_adapter('code', 'Write a function:')}");
```

### A/B Testing Adapters

```gcl
var model = Model::load("base", "./model.gguf", null);
var ctx = Context::create(model, null);

var adapter_a = LoraAdapter::load(model, "./adapter-a.gguf", 1.0, null);
var adapter_b = LoraAdapter::load(model, "./adapter-b.gguf", 1.0, null);

var prompts = [
    "Explain quantum computing",
    "Write a story about",
    "What is machine learning?"
];

println("=== Testing Adapter A ===");
ctx.clear_lora_adapters();
ctx.apply_lora_adapter(adapter_a, 1.0);
for (var prompt in prompts) {
    var result = model.generate(prompt, GenerationParams { max_tokens: 50 }, null);
    println("${prompt}: ${result.text}");
}

println("\n=== Testing Adapter B ===");
ctx.clear_lora_adapters();
ctx.apply_lora_adapter(adapter_b, 1.0);
for (var prompt in prompts) {
    var result = model.generate(prompt, GenerationParams { max_tokens: 50 }, null);
    println("${prompt}: ${result.text}");
}
```

## Best Practices

- **Context-Based**: LoRA adapters are applied per-context, not globally to the model
- **Scale Tuning**: Start with scale 1.0 and adjust down if adapter effect is too strong
- **Adapter Combinations**: Multiple adapters can be applied to the same context
- **Metadata**: Store adapter purpose and settings in metadata for documentation
- **Reapply After Scale Change**: Call `remove_lora_adapter()` then `apply_lora_adapter()` after changing scale
- **Clear Before Switching**: Call `clear_lora_adapters()` before applying a new set of adapters
- **Memory Management**: Free adapters when done, or rely on GC
- **ALORA Detection**: Check `alora_invocation_token_count()` to determine if adapter is ALORA
- **Thread Count**: Use more threads (`n_threads`) for faster adapter loading
- **Compatibility**: Ensure adapter was trained on the same base model architecture
- **Testing**: Test adapter effect on varied prompts to understand its impact
- **Production**: Keep adapter files organized and version-controlled
