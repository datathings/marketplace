# GGUF File Format API

GGUF (v3) is the binary format used by llama.cpp and ggml to store model weights, metadata key-value pairs, and tensor layout information.

## Table of Contents
1. [Types & Init Params](#types--init-params)
2. [Context Lifecycle](#context-lifecycle)
3. [Reading Key-Value Metadata](#reading-key-value-metadata)
4. [Reading Array Metadata](#reading-array-metadata)
5. [Reading Tensor Metadata](#reading-tensor-metadata)
6. [Writing Key-Value Metadata](#writing-key-value-metadata)
7. [Writing Tensor Metadata](#writing-tensor-metadata)
8. [Serialization](#serialization)

---

## Types & Init Params

```c
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
};

struct gguf_init_params {
    bool no_alloc;           // do not allocate tensor data (metadata only)
    struct ggml_context ** ctx;  // if not NULL, create a ggml context and fill tensor metadata
};
```

**Constants:**
- `GGUF_MAGIC` = `"GGUF"` (0x46554747)
- `GGUF_VERSION` = 3
- `GGUF_DEFAULT_ALIGNMENT` = 32

---

## Context Lifecycle

```c
// Create an empty GGUF context (for writing)
struct gguf_context * gguf_init_empty(void);

// Load a GGUF file (for reading or editing)
// Set params.no_alloc = true for metadata-only; params.ctx for tensor data loading
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);

// Free the GGUF context
void gguf_free(struct gguf_context * ctx);

// File-level metadata
uint32_t    gguf_get_version(const struct gguf_context * ctx);
size_t      gguf_get_alignment(const struct gguf_context * ctx);
size_t      gguf_get_data_offset(const struct gguf_context * ctx);  // byte offset to tensor data

// Type name
const char * gguf_type_name(enum gguf_type type);
```

**Example — load metadata only:**
```c
struct gguf_init_params params = {
    .no_alloc = true,
    .ctx      = NULL,
};
struct gguf_context * gguf = gguf_init_from_file("model.gguf", params);
printf("GGUF version: %u\n", gguf_get_version(gguf));
gguf_free(gguf);
```

**Example — load with tensor data:**
```c
struct ggml_context * model_ctx = NULL;
struct gguf_init_params params = {
    .no_alloc = false,
    .ctx      = &model_ctx,  // will be populated with tensor metadata
};
struct gguf_context * gguf = gguf_init_from_file("model.gguf", params);
// model_ctx now has all tensors; use ggml_get_tensor() to retrieve them
```

---

## Reading Key-Value Metadata

```c
// Number of key-value pairs
int64_t gguf_get_n_kv(const struct gguf_context * ctx);

// Find a key by name; returns -1 if not found
int64_t gguf_find_key(const struct gguf_context * ctx, const char * key);

// Get key name and type
const char *   gguf_get_key(const struct gguf_context * ctx, int64_t key_id);
enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id);
enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id);  // for ARRAY type

// Typed value accessors
uint8_t      gguf_get_val_u8(const struct gguf_context * ctx, int64_t key_id);
int8_t       gguf_get_val_i8(const struct gguf_context * ctx, int64_t key_id);
uint16_t     gguf_get_val_u16(const struct gguf_context * ctx, int64_t key_id);
int16_t      gguf_get_val_i16(const struct gguf_context * ctx, int64_t key_id);
uint32_t     gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id);
int32_t      gguf_get_val_i32(const struct gguf_context * ctx, int64_t key_id);
float        gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id);
uint64_t     gguf_get_val_u64(const struct gguf_context * ctx, int64_t key_id);
int64_t      gguf_get_val_i64(const struct gguf_context * ctx, int64_t key_id);
double       gguf_get_val_f64(const struct gguf_context * ctx, int64_t key_id);
bool         gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id);
const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id);
const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id);
```

**Example — iterate all keys:**
```c
int64_t n_kv = gguf_get_n_kv(gguf);
for (int64_t i = 0; i < n_kv; i++) {
    const char * key  = gguf_get_key(gguf, i);
    enum gguf_type t  = gguf_get_kv_type(gguf, i);
    printf("  %s : %s\n", key, gguf_type_name(t));
}
```

**Example — read a specific key:**
```c
int64_t key_id = gguf_find_key(gguf, "llama.context_length");
if (key_id >= 0) {
    uint32_t ctx_len = gguf_get_val_u32(gguf, key_id);
    printf("context_length = %u\n", ctx_len);
}
```

---

## Reading Array Metadata

```c
size_t       gguf_get_arr_n(const struct gguf_context * ctx, int64_t key_id);       // array length
const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id);    // raw data pointer
const char * gguf_get_arr_str(const struct gguf_context * ctx, int64_t key_id, size_t i); // string array element
```

---

## Reading Tensor Metadata

```c
int64_t      gguf_get_n_tensors(const struct gguf_context * ctx);
int64_t      gguf_find_tensor(const struct gguf_context * ctx, const char * name);  // -1 if not found
size_t       gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id); // byte offset in data section
const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id);
enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id);
size_t       gguf_get_tensor_size(const struct gguf_context * ctx, int64_t tensor_id);  // size in bytes
```

**Example — list all tensors:**
```c
int64_t n_tensors = gguf_get_n_tensors(gguf);
for (int64_t i = 0; i < n_tensors; i++) {
    printf("tensor[%ld]: %s  type=%s  offset=%zu\n",
           i,
           gguf_get_tensor_name(gguf, i),
           ggml_type_name(gguf_get_tensor_type(gguf, i)),
           gguf_get_tensor_offset(gguf, i));
}
```

---

## Writing Key-Value Metadata

```c
// Remove a key (returns index, -1 if not found)
int64_t gguf_remove_key(struct gguf_context * ctx, const char * key);

// Set typed scalar values
void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val);
void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val);
void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val);
void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val);
void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val);
void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val);
void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val);
void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val);
void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val);
void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val);
void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val);
void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val);

// Set array values
void gguf_set_arr_data(struct gguf_context * ctx, const char * key,
                       enum gguf_type type, const void * data, size_t n);
void gguf_set_arr_str(struct gguf_context * ctx, const char * key,
                      const char ** data, size_t n);

// Copy all KV pairs from src to ctx
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src);
```

---

## Writing Tensor Metadata

```c
// Add a tensor to the GGUF context (its data must be set separately)
void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);

// Change the quantization type of an already-registered tensor
void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);

// Point tensor data to external buffer
void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data);
```

---

## Serialization

```c
// Write GGUF file
// only_meta = true: write header + KV + tensor metadata only (no tensor data)
bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);

// Get the size of the metadata section (for in-memory serialization)
size_t gguf_get_meta_size(const struct gguf_context * ctx);
void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);
```

**Example — create and write a minimal GGUF:**
```c
struct gguf_context * gguf = gguf_init_empty();

// Set metadata
gguf_set_val_str(gguf, "general.architecture", "llama");
gguf_set_val_u32(gguf, "llama.context_length", 4096);
gguf_set_val_u32(gguf, "llama.embedding_length", 4096);

// Add tensors (must set data separately)
// gguf_add_tensor(gguf, weight_tensor);
// gguf_set_tensor_data(gguf, "token_embd.weight", weight_data_ptr);

bool ok = gguf_write_to_file(gguf, "output.gguf", false);
gguf_free(gguf);
```
