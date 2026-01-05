# llama.cpp C API Reference - Inference & Tokenization

Part 4 of 6 | [Core](api-core.md) | [Model Info](api-model-info.md) | [Context](api-context.md) | **Inference** | [Sampling](api-sampling.md) | [Advanced](api-advanced.md)

This file covers:
- Batch Operations - Create and manage token batches
- Inference & Decoding - Run inference, get logits and embeddings
- Vocabulary & Tokenization - Tokenize text, convert tokens
- Chat Templates - Format chat conversations

For complete API navigation, see [api-core.md](api-core.md).

---

## Batch Operations

### llama_batch_get_one
```c
struct llama_batch llama_batch_get_one(
    llama_token * tokens,
    int32_t n_tokens);
```
Return batch for single sequence of tokens. The sequence ID will be fixed to 0. The position of tokens will be tracked automatically. This is a helper function to facilitate transition to the new batch API - avoid using it for new code.

**Usage:**
```c
llama_token tokens[] = {1, 2, 3, 4, 5};
struct llama_batch batch = llama_batch_get_one(tokens, 5);
llama_decode(ctx, batch);
```

### llama_batch_init
```c
struct llama_batch llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max);
```
Allocate a batch of tokens on the heap. Must be freed with `llama_batch_free()`.

**Parameters:**
- `n_tokens`: Maximum number of tokens
- `embd`: If != 0, allocate `llama_batch.embd` with size `n_tokens * embd * sizeof(float)`. Otherwise, allocate `llama_batch.token` to store `n_tokens` token IDs
- `n_seq_max`: Maximum number of sequences each token can be assigned to

**Usage:**
```c
struct llama_batch batch = llama_batch_init(512, 0, 1);
// Use batch...
llama_batch_free(batch);
```

### llama_batch_free
```c
void llama_batch_free(struct llama_batch batch);
```
Free a batch allocated with `llama_batch_init()`.

---

## Inference & Decoding

### llama_encode
```c
int32_t llama_encode(
    struct llama_context * ctx,
    struct llama_batch batch);
```
Process a batch of tokens without using KV cache. For encoder-decoder models, processes the batch using the encoder and stores the output internally for later use by the decoder's cross-attention layers.

**Returns:**
- `0`: Success
- `< 0`: Error (memory state is restored)

### llama_decode
```c
int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);
```
Process a batch of tokens. Requires the context to have memory. For encoder-decoder models, processes using the decoder.

**Returns:**
- `0`: Success
- `1`: Could not find a KV slot (try reducing batch size or increase context)
- `2`: Aborted (processed ubatches remain in memory)
- `-1`: Invalid input batch
- `< -1`: Fatal error (processed ubatches remain in memory)

**Usage:**
```c
struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    // Handle error
}
```

### llama_set_n_threads
```c
void llama_set_n_threads(
    struct llama_context * ctx,
    int32_t n_threads,
    int32_t n_threads_batch);
```
Set the number of threads used for decoding.

**Parameters:**
- `ctx`: Context
- `n_threads`: Number of threads for generation (single token)
- `n_threads_batch`: Number of threads for batch processing (multiple tokens)

### llama_n_threads
```c
int32_t llama_n_threads(struct llama_context * ctx);
```
Get the number of threads used for generation of a single token.

### llama_n_threads_batch
```c
int32_t llama_n_threads_batch(struct llama_context * ctx);
```
Get the number of threads used for batch processing.

### llama_set_embeddings
```c
void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
```
Set whether the context outputs embeddings or not.

### llama_set_causal_attn
```c
void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);
```
Set whether to use causal attention or not. If set to true, the model will only attend to past tokens.

### llama_set_warmup
```c
void llama_set_warmup(struct llama_context * ctx, bool warmup);
```
Set whether the model is in warmup mode. If true, all model tensors are activated during `llama_decode()` to load and cache their weights.

### llama_set_abort_callback
```c
void llama_set_abort_callback(
    struct llama_context * ctx,
    ggml_abort_callback abort_callback,
    void * abort_callback_data);
```
Set abort callback. If it returns true, execution will be aborted (currently only works with CPU execution).

### llama_synchronize
```c
void llama_synchronize(struct llama_context * ctx);
```
Wait until all computations are finished. Automatically done when obtaining results, not usually necessary to call explicitly.

### llama_get_logits
```c
float * llama_get_logits(struct llama_context * ctx);
```
Get token logits from the last `llama_decode()` call. Logits for which `llama_batch.logits[i] != 0` are stored contiguously.

**Returns:** Pointer to logits array. Shape: `[n_outputs, n_vocab]`

### llama_get_logits_ith
```c
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
```
Get logits for the i-th token. Negative indices access logits in reverse order (-1 is the last token). Returns NULL for invalid IDs.

**Usage:**
```c
float * logits = llama_get_logits_ith(ctx, -1);  // Get logits for last token
```

### llama_get_embeddings
```c
float * llama_get_embeddings(struct llama_context * ctx);
```
Get all output token embeddings. Returns NULL when `pooling_type == LLAMA_POOLING_TYPE_NONE` with generative models.

**Returns:** Pointer to embeddings array. Shape: `[n_outputs * n_embd]`

### llama_get_embeddings_ith
```c
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
```
Get embeddings for the i-th token. Negative indices can be used (-1 is last). Returns NULL for invalid IDs.

**Returns:** Shape: `[n_embd]`

### llama_get_embeddings_seq
```c
float * llama_get_embeddings_seq(
    struct llama_context * ctx,
    llama_seq_id seq_id);
```
Get embeddings for a sequence ID. Returns NULL if `pooling_type` is `LLAMA_POOLING_TYPE_NONE`. For `LLAMA_POOLING_TYPE_RANK`, returns `float[n_cls_out]` with rank(s).

**Returns:** Shape: `[n_embd]` or `[n_cls_out]` for ranking models

---

## Vocabulary & Tokenization

### llama_vocab_type
```c
enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);
```
Get the vocabulary type (SPM, BPE, WPM, UGM, RWKV, PLAMO2).

### llama_vocab_n_tokens
```c
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
```
Get the number of tokens in the vocabulary.

### llama_vocab_get_text
```c
const char * llama_vocab_get_text(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the text representation of a token.

### llama_vocab_get_score
```c
float llama_vocab_get_score(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the score of a token.

### llama_vocab_get_attr
```c
enum llama_token_attr llama_vocab_get_attr(
    const struct llama_vocab * vocab,
    llama_token token);
```
Get the attributes of a token (bitfield of `llama_token_attr`).

### llama_vocab_is_eog
```c
bool llama_vocab_is_eog(
    const struct llama_vocab * vocab,
    llama_token token);
```
Check if the token is an end-of-generation token (EOS, EOT, etc.).

### llama_vocab_is_control
```c
bool llama_vocab_is_control(
    const struct llama_vocab * vocab,
    llama_token token);
```
Check if the token is a control token or a renderable token.

### Special Token Functions

Get special token IDs:

```c
llama_token llama_vocab_bos(const struct llama_vocab * vocab);   // beginning-of-sentence
llama_token llama_vocab_eos(const struct llama_vocab * vocab);   // end-of-sentence
llama_token llama_vocab_eot(const struct llama_vocab * vocab);   // end-of-turn
llama_token llama_vocab_sep(const struct llama_vocab * vocab);   // sentence separator
llama_token llama_vocab_nl(const struct llama_vocab * vocab);    // next-line
llama_token llama_vocab_pad(const struct llama_vocab * vocab);   // padding
llama_token llama_vocab_mask(const struct llama_vocab * vocab);  // mask
```

Check if special tokens should be added:

```c
bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);
```

Fill-in-the-middle tokens:

```c
llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
```

### llama_tokenize
```c
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);
```
Convert text into tokens.

**Parameters:**
- `vocab`: Vocabulary
- `text`: Text to tokenize
- `text_len`: Length of text
- `tokens`: Output buffer (must be large enough)
- `n_tokens_max`: Maximum number of tokens
- `add_special`: Allow adding BOS and EOS tokens if model is configured to do so
- `parse_special`: Allow tokenizing special/control tokens (otherwise treated as plaintext)

**Returns:**
- Positive: Number of tokens (no more than `n_tokens_max`)
- Negative: Number of tokens that would have been returned (buffer too small)
- `INT32_MIN`: Overflow

**Usage:**
```c
const char * text = "Hello, world!";
llama_token tokens[128];
int n = llama_tokenize(vocab, text, strlen(text), tokens, 128, true, false);
if (n < 0) {
    // Buffer too small, need -n tokens
}
```

### llama_token_to_piece
```c
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);
```
Convert a token ID to text. Does not write null terminator.

**Parameters:**
- `vocab`: Vocabulary
- `token`: Token ID
- `buf`: Output buffer
- `length`: Buffer length
- `lstrip`: Number of leading spaces to skip (useful when encoding/decoding multiple tokens)
- `special`: If true, special tokens are rendered

### llama_detokenize
```c
int32_t llama_detokenize(
    const struct llama_vocab * vocab,
    const llama_token * tokens,
    int32_t n_tokens,
    char * text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special);
```
Convert tokens back into text.

**Parameters:**
- `vocab`: Vocabulary
- `tokens`: Array of tokens
- `n_tokens`: Number of tokens
- `text`: Output buffer
- `text_len_max`: Maximum text length
- `remove_special`: Remove BOS and EOS tokens if model is configured to do so
- `unparse_special`: If true, special tokens are rendered

**Returns:**
- Positive: Number of chars/bytes (no more than `text_len_max`)
- Negative: Number of chars/bytes that would have been returned

---

## Chat Templates

### llama_chat_apply_template
```c
int32_t llama_chat_apply_template(
    const char * tmpl,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);
```
Apply chat template to format a conversation. Does not use a Jinja parser - only supports a pre-defined list of templates.

**Parameters:**
- `tmpl`: Jinja template (NULL to use model's default)
- `chat`: Array of chat messages
- `n_msg`: Number of messages
- `add_ass`: Whether to end prompt with assistant message start token(s)
- `buf`: Output buffer (recommended size: 2 * total characters of all messages)
- `length`: Buffer size

**Returns:** Total number of bytes of the formatted prompt

**Usage:**
```c
llama_chat_message messages[] = {
    {"system", "You are a helpful assistant."},
    {"user", "Hello!"}
};
char buf[1024];
int len = llama_chat_apply_template(NULL, messages, 2, true, buf, 1024);
```

### llama_chat_builtin_templates
```c
int32_t llama_chat_builtin_templates(const char ** output, size_t len);
```
Get list of built-in chat templates.

---

