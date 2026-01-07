# llama.cpp C API Reference - Model Properties

Part 2 of 6 | [Core](api-core.md) | **Model Info** | [Context](api-context.md) | [Inference](api-inference.md) | [Sampling](api-sampling.md) | [Advanced](api-advanced.md)

This file covers:
- Model Properties & Metadata - Query model architecture and metadata

For complete API navigation, see [api-core.md](api-core.md).

---

## Model Properties & Metadata

### llama_model_get_vocab
```c
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
```
Get the model's vocabulary.

### llama_model_rope_type
```c
enum llama_rope_type llama_model_rope_type(const struct llama_model * model);
```
Get the model's RoPE type.

### llama_model_n_ctx_train
```c
int32_t llama_model_n_ctx_train(const struct llama_model * model);
```
Get the context size the model was trained with.

### llama_model_n_embd
```c
int32_t llama_model_n_embd(const struct llama_model * model);
```
Get the embedding dimension.

### llama_model_n_embd_inp
```c
int32_t llama_model_n_embd_inp(const struct llama_model * model);
```
Get the input embedding dimension.

### llama_model_n_embd_out
```c
int32_t llama_model_n_embd_out(const struct llama_model * model);
```
Get the output embedding dimension.

### llama_model_n_layer
```c
int32_t llama_model_n_layer(const struct llama_model * model);
```
Get the number of layers in the model.

### llama_model_n_head
```c
int32_t llama_model_n_head(const struct llama_model * model);
```
Get the number of attention heads.

### llama_model_n_head_kv
```c
int32_t llama_model_n_head_kv(const struct llama_model * model);
```
Get the number of KV heads (for grouped-query attention).

### llama_model_n_swa
```c
int32_t llama_model_n_swa(const struct llama_model * model);
```
Get the sliding window attention size.

### llama_model_rope_freq_scale_train
```c
float llama_model_rope_freq_scale_train(const struct llama_model * model);
```
Get the model's RoPE frequency scaling factor.

### llama_model_n_cls_out
```c
uint32_t llama_model_n_cls_out(const struct llama_model * model);
```
Get the number of classifier outputs (only valid for classifier models). Undefined behavior for non-classifier models.

### llama_model_cls_label
```c
const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);
```
Get the label of a classifier output by index. Returns NULL if no label provided.

### llama_model_meta_val_str
```c
int32_t llama_model_meta_val_str(
    const struct llama_model * model,
    const char * key,
    char * buf,
    size_t buf_size);
```
Get metadata value as a string by key name. Returns the length of the string on success, or -1 on failure. The output string is always null-terminated.

### llama_model_meta_count
```c
int32_t llama_model_meta_count(const struct llama_model * model);
```
Get the number of metadata key/value pairs.

### llama_model_meta_key_str
```c
const char * llama_model_meta_key_str(enum llama_model_meta_key key);
```
Get sampling metadata key name. Returns NULL if the key is invalid.

### llama_model_meta_key_by_index
```c
int32_t llama_model_meta_key_by_index(
    const struct llama_model * model,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata key name by index.

### llama_model_meta_val_str_by_index
```c
int32_t llama_model_meta_val_str_by_index(
    const struct llama_model * model,
    int32_t i,
    char * buf,
    size_t buf_size);
```
Get metadata value as a string by index.

### llama_model_desc
```c
int32_t llama_model_desc(
    const struct llama_model * model,
    char * buf,
    size_t buf_size);
```
Get a string describing the model type.

### llama_model_size
```c
uint64_t llama_model_size(const struct llama_model * model);
```
Get the total size of all tensors in the model in bytes.

### llama_model_n_params
```c
uint64_t llama_model_n_params(const struct llama_model * model);
```
Get the total number of parameters in the model.

### llama_model_has_encoder
```c
bool llama_model_has_encoder(const struct llama_model * model);
```
Returns true if the model contains an encoder that requires `llama_encode()` call.

### llama_model_has_decoder
```c
bool llama_model_has_decoder(const struct llama_model * model);
```
Returns true if the model contains a decoder that requires `llama_decode()` call.

### llama_model_decoder_start_token
```c
llama_token llama_model_decoder_start_token(const struct llama_model * model);
```
For encoder-decoder models, returns the token ID that must be provided to the decoder to start generating. Returns -1 for other models.

### llama_model_is_recurrent
```c
bool llama_model_is_recurrent(const struct llama_model * model);
```
Returns true if the model is recurrent (like Mamba, RWKV, etc.).

### llama_model_is_hybrid
```c
bool llama_model_is_hybrid(const struct llama_model * model);
```
Returns true if the model is hybrid (like Jamba, Granite, etc.).

### llama_model_is_diffusion
```c
bool llama_model_is_diffusion(const struct llama_model * model);
```
Returns true if the model is diffusion-based (like LLaDA, Dream, etc.).

### llama_model_chat_template
```c
const char * llama_model_chat_template(
    const struct llama_model * model,
    const char * name);
```
Get the default chat template. Returns NULL if not available. If `name` is NULL, returns the default chat template.

---

