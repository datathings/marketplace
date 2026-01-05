# Context - Advanced LLM Context API

The Context type provides low-level access to llama.cpp contexts for expert users who need fine-grained control over KV cache, sequences, batching, and state management.

## Overview

Most users should use the high-level Model API instead. Use the Context API when you need:
- Multi-turn conversations with explicit KV cache management
- Parallel sequence processing (multiple conversations simultaneously)
- Custom batching strategies
- State save/restore for long-running sessions
- Manual memory management
- Low-level control over inference

## Quick Start

```gcl
var model = Model::load("model", "./model.gguf", null);
var ctx = Context::create(model, ContextParams { n_ctx: 4096 });

// Tokenize and create batch
var tokens = model.tokenize("Hello world", true, false);
var batch = Batch::from_array(tokens, 0, 0);

// Decode
ctx.decode(batch);

// Get logits and sample
var logits = ctx.get_logits(-1);

// Cleanup
ctx.free();
```

## Types

### Context

Inference context with KV cache. Provides low-level control over inference, KV cache, and sequence management. Each context has its own KV cache and can handle multiple sequences.

### Batch

Token batch for inference. Represents a batch of tokens to be processed by the context. Supports single-sequence (simple) and multi-sequence (advanced) batching.

**Methods:**
- `static fn create(n_tokens: int, embd: int, n_seq_max: int): Batch` - Create empty batch
- `static fn from_array(tokens: Array<int>, pos_start: int, seq_id: int): Batch` - Create from token array
- `fn add_token(token: int, pos: int, seq_ids: Array<SeqId>, logits: bool)` - Add token to batch
- `fn add_embedding(embd: Array<float>, pos: int, seq_ids: Array<SeqId>, logits: bool)` - Add embedding
- `fn clear()` - Clear batch contents
- `fn size(): int` - Get number of tokens
- `fn free()` - Free resources

### SeqId

Sequence identifier. Sequences are independent conversation threads in the same context. Multiple sequences can be processed in parallel using batching.

**Fields:**
- `id: int` - Sequence ID value (0 to n_seq_max-1)

### StateData

Saved state from a context (KV cache, etc.) that can be restored later.

**Fields:**
- `data: Buffer` - State data bytes
- `size: int` - State size in bytes

## Static Methods

### `static fn create(model: Model, params: ContextParams?): Context`

Create context from model.

Creates a new inference context with the specified parameters.

**Parameters:**
- `model` - Loaded model instance
- `params` - Optional context parameters

**Returns:** Context instance or null if creation fails

**Example:**
```gcl
var ctx = Context::create(model, ContextParams {
    n_ctx: 4096,
    n_batch: 512,
    n_threads: 8,
    offload_kqv: true
});
```

## Instance Methods - Context Properties

### `fn n_ctx(): int`

Get context size. Returns the actual context size (may differ from requested).

### `fn n_batch(): int`

Get batch size.

### `fn n_ubatch(): int`

Get ubatch size.

### `fn n_seq_max(): int`

Get max sequences.

### `fn n_ctx_seq(): int`

Get sequence-specific context size.

### `fn n_embd(): int`

Get embedding dimension.

### `fn n_vocab(): int`

Get vocabulary size.

### `fn pooling_type(): PoolingType`

Get pooling type.

### `fn n_threads(): int`

Get number of threads used for generation (single token).

### `fn n_threads_batch(): int`

Get number of threads used for batch processing (multiple tokens).

## Instance Methods - Dynamic Configuration

### `fn set_n_threads(n_threads: int, n_threads_batch: int)`

Set number of threads for generation and batch processing.

Allows dynamic adjustment of thread count during execution.

**Parameters:**
- `n_threads` - Number of threads for generation (single token)
- `n_threads_batch` - Number of threads for batch processing

**Example:**
```gcl
ctx.set_n_threads(8, 16);
```

### `fn set_embeddings(enabled: bool)`

Set whether context outputs embeddings.

Toggle embeddings computation on or off.

**Parameters:**
- `enabled` - True to enable embeddings

### `fn set_causal_attn(enabled: bool)`

Set whether to use causal attention.

If true, the model will only attend to past tokens.

**Parameters:**
- `enabled` - True to enable causal attention

### `fn set_warmup(enabled: bool)`

Set warmup mode.

If true, all model tensors are activated during decode to load and cache weights.

**Parameters:**
- `enabled` - True to enable warmup

### `fn synchronize()`

Wait until all computations are finished.

Automatically called when getting results. Usually not needed explicitly.

## Instance Methods - Inference

### `fn encode(batch: Batch): int`

Encode tokens (for encoder models).

Processes a batch through the encoder.

**Parameters:**
- `batch` - Token batch

**Returns:** 0 on success, non-zero on error

**Example:**
```gcl
var tokens = model.tokenize("Encode this text", true, false);
var batch = Batch::from_array(tokens, 0, 0);
var result = ctx.encode(batch);
if (result == 0) {
    println("Encoding successful");
}
```

### `fn decode(batch: Batch): int`

Decode tokens (for decoder models).

Processes a batch through the decoder.

**Parameters:**
- `batch` - Token batch

**Returns:** 0 on success, 1 if no KV slot available

**Example:**
```gcl
var tokens = model.tokenize("Hello world", true, false);
var batch = Batch::from_array(tokens, 0, 0);
var result = ctx.decode(batch);
if (result == 0) {
    println("Decoding successful");
}
```

## Instance Methods - Logits and Embeddings Access

### `fn get_logits(i: int): Array<float>`

Get logits for sequence position.

Returns logits for the token at the specified batch index.

**Parameters:**
- `i` - Batch index (-1 = last token)

**Returns:** Array of logits (size = vocab_size)

**Example:**
```gcl
var logits = ctx.get_logits(-1);
println("Vocab size: ${logits.length}");
```

### `fn get_embeddings(i: int): Array<float>`

Get embeddings for sequence position.

Returns embedding vector for the token at batch index i.

**Parameters:**
- `i` - Batch index

**Returns:** Array of embeddings (size = n_embd)

### `fn get_embeddings_seq(seq_id: SeqId): Array<float>`

Get pooled embeddings for sequence.

Returns pooled embedding for the entire sequence.

**Parameters:**
- `seq_id` - Sequence ID

**Returns:** Array of pooled embeddings

**Example:**
```gcl
var emb = ctx.get_embeddings_seq(SeqId { id: 0 });
```

## Instance Methods - KV Cache Management

### `fn kv_cache_clear()`

Clear entire KV cache.

Removes all tokens from all sequences.

**Example:**
```gcl
ctx.kv_cache_clear();
```

### `fn kv_cache_seq_rm(seq_id: SeqId, p0: int, p1: int)`

Remove tokens from sequence.

Removes tokens in range [p0, p1) from the specified sequence.

**Parameters:**
- `seq_id` - Sequence ID
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive, -1 = end of sequence)

**Example:**
```gcl
// Remove tokens 10-20 from sequence 0
ctx.kv_cache_seq_rm(SeqId { id: 0 }, 10, 20);

// Remove all tokens from position 100 onwards
ctx.kv_cache_seq_rm(SeqId { id: 0 }, 100, -1);
```

### `fn kv_cache_seq_cp(src: SeqId, dst: SeqId, p0: int, p1: int)`

Copy sequence.

Copies tokens from src sequence to dst sequence.

**Parameters:**
- `src` - Source sequence ID
- `dst` - Destination sequence ID
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive, -1 = end)

**Example:**
```gcl
// Copy all tokens from sequence 0 to sequence 1
ctx.kv_cache_seq_cp(SeqId { id: 0 }, SeqId { id: 1 }, 0, -1);
```

### `fn kv_cache_seq_keep(keep: Array<SeqId>)`

Keep only specified sequences.

Removes all sequences except those in the keep list.

**Parameters:**
- `keep` - Array of sequence IDs to keep

**Example:**
```gcl
// Keep only sequences 0 and 2
ctx.kv_cache_seq_keep([SeqId { id: 0 }, SeqId { id: 2 }]);
```

### `fn kv_cache_seq_add(seq_id: SeqId, p0: int, p1: int, delta: int)`

Shift sequence positions.

Adds delta to all positions in range [p0, p1) for the sequence. Used for context shifting.

**Parameters:**
- `seq_id` - Sequence ID
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive, -1 = end)
- `delta` - Position delta to add

**Example:**
```gcl
// Shift all tokens forward by 10 positions
ctx.kv_cache_seq_add(SeqId { id: 0 }, 0, -1, 10);
```

### `fn kv_cache_seq_div(seq_id: SeqId, p0: int, p1: int, d: int)`

Divide sequence positions.

Divides all positions in range [p0, p1) by d for the sequence. Integer division: pos[i] = pos[i] / d

**Parameters:**
- `seq_id` - Sequence ID
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive, -1 = end)
- `d` - Divisor

### `fn kv_cache_seq_pos_max(seq_id: SeqId): int`

Get max contiguous position in sequence.

Returns the highest position with continuous tokens in the sequence.

**Parameters:**
- `seq_id` - Sequence ID

**Returns:** Max position

### `fn kv_cache_seq_pos_min(seq_id: SeqId): int`

Get min contiguous position in sequence.

Returns the smallest position present in the memory for the sequence. Typically non-zero only for SWA caches.

**Parameters:**
- `seq_id` - Sequence ID

**Returns:** Min position or -1 if sequence is empty

### `fn kv_cache_can_shift(): bool`

Check if KV cache memory supports shifting.

**Returns:** True if the memory implementation supports position shifting

### `fn kv_cache_update()`

Update KV cache.

Updates internal KV cache state. Call after manual cache manipulation.

### `fn kv_cache_defrag()`

Defragment KV cache.

Defragments the KV cache to reduce fragmentation and improve memory efficiency. Call this periodically in long-running sessions with many sequence operations.

### `fn kv_cache_can_use_seqs(): bool`

Check if KV cache supports sequences.

**Returns:** True if the memory implementation supports multiple sequences. Some memory backends may not support sequence operations.

## Instance Methods - State Management

### `fn get_state_size(): int`

Get state size in bytes.

Returns the size needed to save the current state.

**Returns:** State size in bytes

### `fn get_state(): StateData`

Save state to bytes.

Saves the current inference state (KV cache, etc.) to a byte array.

**Returns:** StateData containing the state

**Example:**
```gcl
var state = ctx.get_state();
println("State size: ${state.size} bytes");
```

### `fn set_state(state: StateData)`

Restore state from bytes.

Restores inference state from previously saved data.

**Parameters:**
- `state` - Previously saved state

**Example:**
```gcl
// Save state
var state = ctx.get_state();

// ... do some work ...

// Restore state
ctx.set_state(state);
```

### `fn load_state(path: String)`

Load state from file.

**Parameters:**
- `path` - File path to load from

### `fn save_state(path: String)`

Save state to file.

**Parameters:**
- `path` - File path to save to

**Example:**
```gcl
ctx.save_state("./session.state");
// Later...
ctx.load_state("./session.state");
```

## Instance Methods - Sequence State Management

### `fn get_state_seq_size(seq_id: SeqId): int`

Get size needed to save sequence state.

**Parameters:**
- `seq_id` - Sequence ID

**Returns:** State size in bytes

### `fn get_state_seq(seq_id: SeqId): StateData`

Save sequence state.

Saves state for a single sequence.

**Parameters:**
- `seq_id` - Sequence ID

**Returns:** StateData for the sequence

### `fn set_state_seq(seq_id: SeqId, state: StateData)`

Restore sequence state.

Restores state for a single sequence.

**Parameters:**
- `seq_id` - Sequence ID
- `state` - Previously saved sequence state

### `fn save_state_seq_file(path: String, seq_id: SeqId, tokens: Array<int>?): int`

Save sequence state to file.

Saves state for a single sequence directly to a file on disk. More efficient than get_state_seq() + manual file write for large states.

**Parameters:**
- `path` - File path to save state to
- `seq_id` - Sequence ID to save
- `tokens` - Optional token context for the sequence

**Returns:** Number of bytes written

### `fn load_state_seq_file(path: String, seq_id: SeqId): int`

Load sequence state from file.

Loads state for a single sequence directly from a file on disk. More efficient than manual file read + set_state_seq() for large states.

**Parameters:**
- `path` - File path to load state from
- `seq_id` - Destination sequence ID

**Returns:** Number of bytes read

### `fn get_state_seq_size_ext(seq_id: SeqId, flags: int): int`

Get size needed to save sequence state with flags.

Like get_state_seq_size(), but supports partial state (SWA-only, etc.)

**Parameters:**
- `seq_id` - Sequence ID
- `flags` - State flags (1 = SWA/partial only)

**Returns:** State size in bytes

### `fn get_state_seq_ext(seq_id: SeqId, flags: int): StateData`

Save sequence state with flags.

Like get_state_seq(), but supports partial state.

**Parameters:**
- `seq_id` - Sequence ID
- `flags` - State flags (1 = SWA/partial only)

**Returns:** StateData

### `fn set_state_seq_ext(seq_id: SeqId, state: StateData, flags: int)`

Restore sequence state with flags.

Like set_state_seq(), but supports partial state.

**Parameters:**
- `seq_id` - Sequence ID
- `state` - State data
- `flags` - State flags (1 = SWA/partial only)

## Instance Methods - Performance

### `fn perf(): PerfData`

Get performance metrics.

Returns timing data for this context's operations.

**Returns:** PerfData with timing information

### `fn perf_reset()`

Reset performance counters.

### `fn print_memory()`

Print memory usage.

Logs detailed memory breakdown to llama.cpp logger.

## Instance Methods - LoRA Adapter Management

### `fn apply_lora_adapter(adapter: LoraAdapter, scale: float): int`

Apply LoRA adapter to context.

Applies a LoRA adapter to this context with the specified scale. Multiple adapters can be applied to the same context.

**Parameters:**
- `adapter` - LoRA adapter to apply
- `scale` - Scaling factor (0.0-1.0, typically 1.0)

**Returns:** 0 on success, non-zero on error

**Example:**
```gcl
var lora = LoraAdapter::load(model, "./adapter.gguf", 1.0, null);
ctx.apply_lora_adapter(lora, 1.0);
```

### `fn remove_lora_adapter(adapter: LoraAdapter): int`

Remove LoRA adapter from context.

Removes a previously applied LoRA adapter from this context.

**Parameters:**
- `adapter` - LoRA adapter to remove

**Returns:** 0 on success, non-zero on error

### `fn clear_lora_adapters()`

Clear all LoRA adapters from context.

Removes all LoRA adapters, returning context to base model state.

## Instance Methods - Control Vectors

### `fn apply_control_vector(data: Array<float>?, n_embd: int, il_start: int, il_end: int)`

Apply control vector to context.

Applies a control vector to steer model behavior. Control vectors are per-layer adjustments to embeddings.

**Parameters:**
- `data` - Control vector data (n_embd x n_layers, starting from layer 1), null to clear
- `n_embd` - Embedding dimension
- `il_start` - Start layer (inclusive)
- `il_end` - End layer (inclusive)

**Example:**
```gcl
// Apply control vector to layers 10-20
ctx.apply_control_vector(vector_data, 4096, 10, 20);

// Clear control vector
ctx.apply_control_vector(null, 0, 0, 0);
```

## Instance Methods - Resource Management

### `fn free()`

Free context resources.

Explicitly releases context memory. Optional - GC will handle cleanup.

## Common Use Cases

### Multi-Turn Conversation

```gcl
var model = Model::load("chat", "./model.gguf", null);
var ctx = Context::create(model, ContextParams { n_ctx: 2048 });

// First turn
var tokens1 = model.tokenize("User: Hello!", true, false);
var batch1 = Batch::from_array(tokens1, 0, 0);
ctx.decode(batch1);

// Generate response...
var logits = ctx.get_logits(-1);
// ... sampling logic ...

// Second turn (continue sequence)
var tokens2 = model.tokenize("User: How are you?", false, false);
var batch2 = Batch::from_array(tokens2, tokens1.length, 0);
ctx.decode(batch2);
```

### Parallel Sequences

```gcl
var ctx = Context::create(model, ContextParams { n_seq_max: 4 });

// Create batch for multiple sequences
var batch = Batch::create(512, 0, 4);

// Add tokens for sequence 0
batch.add_token(token1, 0, [SeqId { id: 0 }], true);

// Add tokens for sequence 1
batch.add_token(token2, 0, [SeqId { id: 1 }], true);

// Decode both sequences at once
ctx.decode(batch);

// Get logits for each sequence
var logits0 = ctx.get_logits(0);
var logits1 = ctx.get_logits(1);
```

### State Save/Restore

```gcl
var ctx = Context::create(model, ContextParams { n_ctx: 2048 });

// Process some tokens
var tokens = model.tokenize("Some context", true, false);
var batch = Batch::from_array(tokens, 0, 0);
ctx.decode(batch);

// Save state for later
ctx.save_state("./checkpoint.state");

// Continue processing...

// Later, restore to this point
ctx.load_state("./checkpoint.state");
```

### Context Shifting

```gcl
var ctx = Context::create(model, ContextParams { n_ctx: 2048 });

// When context is full, shift it
if (current_pos >= ctx.n_ctx()) {
    // Remove first half of tokens
    var shift = ctx.n_ctx() / 2;
    ctx.kv_cache_seq_rm(SeqId { id: 0 }, 0, shift);

    // Shift remaining tokens back
    ctx.kv_cache_seq_add(SeqId { id: 0 }, shift, -1, -shift);
}
```

## Best Practices

- **Use High-Level API**: Prefer the Model API unless you specifically need low-level control
- **Batch Efficiently**: Group multiple tokens/sequences into batches for better performance
- **Manage Sequences**: Use separate sequences for independent conversations to avoid interference
- **Clear Cache**: Call `kv_cache_clear()` when starting a new conversation to avoid context pollution
- **Save State**: Use state save/restore for long-running sessions or checkpointing
- **Defragment**: Call `kv_cache_defrag()` periodically in long sessions with many operations
- **Thread Tuning**: Adjust thread count dynamically based on workload
- **Memory Monitoring**: Use `print_memory()` to understand memory usage patterns
- **Sequence Limits**: Check `n_seq_max()` and don't exceed the maximum number of sequences
- **Position Tracking**: Keep track of token positions when building batches manually
