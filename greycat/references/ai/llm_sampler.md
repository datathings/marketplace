# Sampler - Advanced LLM Sampling API

The Sampler API provides low-level access to llama.cpp's sampling system for expert users who need fine-grained control over token selection, custom sampler chains, and token candidate inspection.

## Overview

Most users should use the high-level GenerationParams (llm_types.gcl) instead. Use the Sampler API when you need:
- Custom sampler chains with specific ordering
- Token candidate inspection and filtering
- Custom sampling algorithms
- Manual token acceptance and probability updates

## Quick Start

```gcl
// Create sampler chain
var chain = SamplerChain::create(SamplerChainParams { no_perf: false });
chain.add(Sampler::top_k(40));
chain.add(Sampler::top_p(0.95, 1));
chain.add(Sampler::temp(0.8));
chain.add(Sampler::dist(12345));

// Sample token
var token = chain.sample(ctx, -1);

// Accept and continue
chain.accept(token);

// Cleanup
chain.free();
```

## Types

### SamplerChain

A chain of samplers applied in sequence to select the next token. Samplers are applied in the order they are added.

**Common chains:**
- Greedy: `[greedy]`
- Temperature: `[temp, dist]`
- Top-K + Top-P: `[top_k, top_p, temp, dist]`
- With penalties: `[penalties, top_k, top_p, temp, dist]`
- Mirostat: `[temp, mirostat_v2]`

### Sampler

Individual sampling strategy that can be added to a SamplerChain. Create using static factory methods.

### SamplerChainParams

Sampler chain configuration.

**Fields:**
- `no_perf: bool?` - Disable performance measurements

### TokenDataBatch

Array of token data with size information.

**Fields:**
- `data: Array<TokenData>` - Token data array
- `size: int` - Number of tokens
- `sorted: bool` - Whether probabilities are sorted (descending)

### TokenData

Single token with probability.

**Fields:**
- `id: int` - Token ID
- `logit: float` - Token logit (unnormalized log probability)
- `p: float` - Token probability (after softmax)

## SamplerChain Methods

### Static Methods

### `static fn create(params: SamplerChainParams?): SamplerChain`

Create empty sampler chain.

**Parameters:**
- `params` - Optional chain configuration

**Returns:** New sampler chain

**Example:**
```gcl
var chain = SamplerChain::create(null);
```

### Instance Methods

### `fn add(sampler: Sampler)`

Add sampler to chain.

Samplers are applied in the order added.

**Parameters:**
- `sampler` - Sampler to add

**Example:**
```gcl
chain.add(Sampler::top_k(40));
chain.add(Sampler::top_p(0.95, 1));
chain.add(Sampler::temp(0.8));
chain.add(Sampler::dist(12345));
```

### `fn size(): int`

Get number of samplers in chain.

**Returns:** Number of samplers

### `fn get(index: int): Sampler?`

Get sampler at index.

**Parameters:**
- `index` - Sampler index (0-based)

**Returns:** Sampler or null if out of bounds

### `fn remove(index: int): Sampler?`

Remove sampler at index.

Removes and returns the sampler at the specified index. After removal, the chain no longer owns the sampler.

**Parameters:**
- `index` - Sampler index (0-based)

**Returns:** Removed sampler or null if out of bounds

### `fn sample(ctx: Context, idx: int): int`

Sample next token from context.

Applies all samplers in sequence and returns selected token.

**Parameters:**
- `ctx` - Inference context
- `idx` - Batch index to sample from (-1 = last)

**Returns:** Selected token ID

**Example:**
```gcl
var token = chain.sample(ctx, -1);
```

### `fn accept(token: int)`

Accept token.

Notifies samplers that a token was accepted (updates state for stateful samplers like Mirostat).

**Parameters:**
- `token` - Accepted token ID

**Example:**
```gcl
var token = chain.sample(ctx, -1);
chain.accept(token);
```

### `fn apply(candidates: TokenDataBatch)`

Apply samplers to token candidates.

Manually apply the sampler chain to a batch of token candidates. Used for custom sampling logic.

**Parameters:**
- `candidates` - Token candidates to filter/modify

### `fn reset()`

Reset sampler state.

Resets internal state of all samplers in chain.

### `fn clone(): SamplerChain`

Clone sampler chain.

Creates a deep copy of the sampler chain.

**Returns:** Cloned sampler chain

### `fn perf(): PerfSamplerData`

Get performance data.

**Returns:** Performance metrics for sampling operations

### `fn perf_reset()`

Reset performance counters.

### `fn free()`

Free sampler chain.

Releases all samplers in chain. Optional - GC handles cleanup.

## Sampler Factory Methods

### Basic Samplers

### `static fn greedy(): Sampler`

Greedy sampler (always select highest probability token).

**Example:**
```gcl
var chain = SamplerChain::create(null);
chain.add(Sampler::greedy());
```

### `static fn dist(seed: int): Sampler`

Distribution sampler (sample from probability distribution).

**Parameters:**
- `seed` - Random seed (use different seeds for different results)

**Example:**
```gcl
chain.add(Sampler::dist(12345));
```

### Top Samplers

### `static fn top_k(k: int): Sampler`

Top-K sampler.

Keep only top K most probable tokens.

**Parameters:**
- `k` - Number of tokens to keep (e.g., 40)

**Example:**
```gcl
chain.add(Sampler::top_k(40));
```

### `static fn top_p(p: float, min_keep: int): Sampler`

Top-P (nucleus) sampler.

Keep tokens until cumulative probability reaches P.

**Parameters:**
- `p` - Cumulative probability threshold (0.0-1.0, e.g., 0.95)
- `min_keep` - Minimum tokens to keep (e.g., 1)

**Example:**
```gcl
chain.add(Sampler::top_p(0.95, 1));
```

### `static fn min_p(p: float, min_keep: int): Sampler`

Min-P sampler.

Remove tokens with probability less than P * max_prob.

**Parameters:**
- `p` - Threshold (0.0-1.0, e.g., 0.05)
- `min_keep` - Minimum tokens to keep

**Example:**
```gcl
chain.add(Sampler::min_p(0.05, 1));
```

### `static fn typical(p: float, min_keep: int): Sampler`

Typical sampler.

Sample from tokens close to expected entropy.

**Parameters:**
- `p` - Typical probability (0.0-1.0, e.g., 0.95)
- `min_keep` - Minimum tokens to keep

**Example:**
```gcl
chain.add(Sampler::typical(0.95, 1));
```

### `static fn top_n_sigma(n: float): Sampler`

Top-N-Sigma sampler.

Samples from tokens within N standard deviations of the mean.

**Parameters:**
- `n` - Number of standard deviations

**Example:**
```gcl
chain.add(Sampler::top_n_sigma(2.0));
```

### Temperature Samplers

### `static fn temp(temp: float): Sampler`

Temperature sampler.

Scale logits by temperature (higher = more random).

**Parameters:**
- `temp` - Temperature (0.0-2.0, e.g., 0.8)

**Example:**
```gcl
chain.add(Sampler::temp(0.8));
```

### `static fn temp_ext(temp: float, delta: float, exponent: float): Sampler`

Temperature with exponent.

Apply temperature with custom exponent.

**Parameters:**
- `temp` - Temperature
- `delta` - Delta for smoothing
- `exponent` - Exponent (1.0 = normal temperature)

**Example:**
```gcl
chain.add(Sampler::temp_ext(0.8, 0.0, 1.0));
```

### `static fn xtc(probability: float, threshold: float, min_keep: int, seed: int): Sampler`

XTC (cross-token correlation) sampler.

**Parameters:**
- `probability` - XTC probability
- `threshold` - XTC threshold
- `min_keep` - Minimum tokens to keep
- `seed` - Random seed

**Example:**
```gcl
chain.add(Sampler::xtc(0.5, 0.1, 1, 12345));
```

### Penalty Samplers

### `static fn penalties(penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float): Sampler`

Repetition penalties sampler.

Penalize tokens based on their frequency in recent context.

**Parameters:**
- `penalty_last_n` - Number of last tokens to consider (0 = disabled, -1 = context size)
- `penalty_repeat` - Repeat penalty (1.0 = disabled, >1.0 = penalize)
- `penalty_freq` - Frequency penalty (0.0 = disabled)
- `penalty_present` - Presence penalty (0.0 = disabled)

**Example:**
```gcl
chain.add(Sampler::penalties(64, 1.1, 0.0, 0.0));
```

### `static fn dry(model: Model, multiplier: float, base: float, allowed_length: int, penalty_last_n: int, seq_breakers: Array<int>): Sampler`

DRY (Don't Repeat Yourself) sampler.

Advanced repetition penalty that detects and penalizes repeating sequences.

**Parameters:**
- `model` - Model instance (for vocabulary)
- `multiplier` - DRY multiplier
- `base` - DRY base
- `allowed_length` - Allowed length of repeated sequences
- `penalty_last_n` - Penalty range
- `seq_breakers` - Sequence breakers (tokens that break repetition detection)

**Example:**
```gcl
chain.add(Sampler::dry(model, 0.8, 1.75, 2, 256, []));
```

### Advanced Samplers

### `static fn grammar(model: Model, grammar_str: String, grammar_root: String): Sampler`

Grammar-based sampler.

Constrain generation to follow a grammar (GBNF format).

**Parameters:**
- `model` - Model instance (for vocabulary)
- `grammar_str` - Grammar string (GBNF format)
- `grammar_root` - Root rule name

**Example:**
```gcl
var grammar = "root ::= number\nnumber ::= [0-9]+";
chain.add(Sampler::grammar(model, grammar, "root"));
```

### `static fn grammar_lazy_patterns(model: Model, grammar_str: String, grammar_root: String, trigger_patterns: Array<String>, trigger_tokens: Array<int>): Sampler`

Lazy grammar sampler with pattern triggers.

Like grammar(), but only activates when trigger patterns or tokens are encountered. Patterns are matched from the start of generation output.

**Parameters:**
- `model` - Model instance (for vocabulary)
- `grammar_str` - Grammar string (GBNF format)
- `grammar_root` - Root rule name
- `trigger_patterns` - Array of regex patterns to trigger grammar
- `trigger_tokens` - Array of token IDs to trigger grammar

**Example:**
```gcl
chain.add(Sampler::grammar_lazy_patterns(
    model,
    grammar,
    "root",
    ["^\\{"],  // Trigger on opening brace
    []
));
```

### `static fn mirostat(vocab_size: int, seed: int, tau: float, eta: float, m: int): Sampler`

Mirostat v1 sampler.

**Parameters:**
- `vocab_size` - Vocabulary size
- `seed` - Random seed
- `tau` - Target entropy
- `eta` - Learning rate
- `m` - Number of candidates

**Example:**
```gcl
chain.add(Sampler::mirostat(32000, 12345, 5.0, 0.1, 100));
```

### `static fn mirostat_v2(seed: int, tau: float, eta: float): Sampler`

Mirostat v2 sampler.

**Parameters:**
- `seed` - Random seed
- `tau` - Target entropy
- `eta` - Learning rate

**Example:**
```gcl
chain.add(Sampler::temp(0.8));
chain.add(Sampler::mirostat_v2(12345, 5.0, 0.1));
```

### `static fn logit_bias(vocab_size: int, biases: Array<LogitBias>): Sampler`

Logit bias sampler.

Apply fixed biases to specific tokens.

**Parameters:**
- `vocab_size` - Vocabulary size
- `biases` - Array of (token, bias) pairs

**Example:**
```gcl
var biases = [
    LogitBias { token: 100, bias: 2.0 },   // Boost token 100
    LogitBias { token: 200, bias: -10.0 }  // Suppress token 200
];
chain.add(Sampler::logit_bias(32000, biases));
```

### `static fn infill(model: Model): Sampler`

Infill sampler.

For code infilling (fill-in-the-middle).

**Parameters:**
- `model` - Model instance

**Example:**
```gcl
chain.add(Sampler::infill(model));
```

## Sampler Instance Methods

### `fn apply(candidates: TokenDataBatch)`

Apply sampler to token candidates.

Manually apply this sampler to a batch of token candidates.

**Parameters:**
- `candidates` - Token candidates to filter/modify

### `fn accept(token: int)`

Accept token.

For stateful samplers (Mirostat), update state after token selection.

**Parameters:**
- `token` - Accepted token ID

### `fn clone(): Sampler`

Clone sampler.

**Returns:** Cloned sampler

### `fn name(): String`

Get sampler name.

**Returns:** Sampler type name

### `fn get_seed(): int`

Get sampler seed.

Returns the random seed used by this sampler if applicable. Returns 0xFFFFFFFF (LLAMA_DEFAULT_SEED) if not applicable.

**Returns:** Random seed

### `fn free()`

Free sampler resources.

## TokenCandidates Utility Methods

### `static fn from_context(ctx: Context, idx: int): TokenDataBatch`

Create token candidates from context logits.

**Parameters:**
- `ctx` - Inference context
- `idx` - Batch index (-1 = last)

**Returns:** Token candidates

**Example:**
```gcl
var candidates = TokenCandidates::from_context(ctx, -1);
```

### `static fn create(vocab_size: int): TokenDataBatch`

Create empty token candidates.

**Parameters:**
- `vocab_size` - Vocabulary size

**Returns:** Empty token candidates

### `static fn sort(candidates: TokenDataBatch)`

Sort candidates by probability (descending).

**Parameters:**
- `candidates` - Token candidates to sort

### `static fn sample_temp(candidates: TokenDataBatch, temp: float, seed: int): int`

Sample from candidates using temperature.

**Parameters:**
- `candidates` - Token candidates
- `temp` - Temperature
- `seed` - Random seed

**Returns:** Selected token ID

### `static fn sample_top(candidates: TokenDataBatch): int`

Sample from candidates (top token).

Returns the token with highest probability.

**Parameters:**
- `candidates` - Token candidates

**Returns:** Top token ID

## Common Use Cases

### Standard Sampling Chain

```gcl
var chain = SamplerChain::create(null);
chain.add(Sampler::penalties(64, 1.1, 0.0, 0.0));
chain.add(Sampler::top_k(40));
chain.add(Sampler::top_p(0.95, 1));
chain.add(Sampler::temp(0.8));
chain.add(Sampler::dist(12345));

// Use in generation loop
while (n_tokens < max_tokens) {
    ctx.decode(batch);
    var token = chain.sample(ctx, -1);
    chain.accept(token);

    if (model.is_eog_token(token)) {
        break;
    }

    // Add token to next batch...
}
```

### Greedy Decoding

```gcl
var chain = SamplerChain::create(null);
chain.add(Sampler::greedy());

var token = chain.sample(ctx, -1);
```

### Temperature-Only Sampling

```gcl
var chain = SamplerChain::create(null);
chain.add(Sampler::temp(0.7));
chain.add(Sampler::dist(12345));

var token = chain.sample(ctx, -1);
chain.accept(token);
```

### Mirostat v2 Sampling

```gcl
var chain = SamplerChain::create(null);
chain.add(Sampler::temp(0.8));
chain.add(Sampler::mirostat_v2(12345, 5.0, 0.1));

var token = chain.sample(ctx, -1);
chain.accept(token);
```

### Grammar-Constrained Generation

```gcl
var grammar = `
root ::= object
object ::= "{" ws members ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
string ::= "\\"" [^"]* "\\""
value ::= string | number
number ::= [0-9]+
ws ::= [ \\t\\n]*
`;

var chain = SamplerChain::create(null);
chain.add(Sampler::grammar(model, grammar, "root"));
chain.add(Sampler::temp(0.7));
chain.add(Sampler::dist(12345));
```

### Custom Token Inspection

```gcl
var candidates = TokenCandidates::from_context(ctx, -1);

// Inspect top candidates before sampling
TokenCandidates::sort(candidates);
for (var i = 0; i < 5 && i < candidates.size; i++) {
    var token_data = candidates.data[i];
    var token_text = model.token_to_text(token_data.id);
    println("Token ${i}: ${token_text} (p=${token_data.p})");
}

// Apply custom filtering
// ... modify candidates ...

// Sample from filtered candidates
var token = TokenCandidates::sample_top(candidates);
```

### Dynamic Sampler Chain

```gcl
var chain = SamplerChain::create(null);

// Add base samplers
chain.add(Sampler::top_k(40));
chain.add(Sampler::top_p(0.95, 1));

// Conditionally add temperature
if (use_temperature) {
    chain.add(Sampler::temp(0.8));
}

chain.add(Sampler::dist(12345));

// Later, remove a sampler
var removed = chain.remove(0);  // Remove top_k
removed.free();
```

## Best Practices

- **Use High-Level API**: Prefer GenerationParams unless you need custom sampling logic
- **Sampler Ordering**: Order matters - typically use: penalties → top_k → top_p → temp → dist
- **Always Accept**: Call `accept()` after sampling for stateful samplers to work correctly
- **Clone for Reuse**: Clone sampler chains if you need multiple independent instances
- **Free Resources**: Call `free()` on chains and samplers when done (or let GC handle it)
- **Temperature First**: For Mirostat, apply temperature before the Mirostat sampler
- **Grammar Validation**: Test grammars with simple inputs before using in production
- **Seed Management**: Use different seeds for different generation runs to get varied outputs
- **Penalty Tuning**: Start with small penalties (1.1-1.2) and adjust based on output
- **Min Keep**: Always set min_keep to at least 1 to avoid empty candidate sets
- **Performance**: Disable performance tracking (`no_perf: true`) in production for minimal overhead
