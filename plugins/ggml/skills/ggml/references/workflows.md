# Workflows & Examples

## Table of Contents
1. [Quick Start — Tensor Computation on CPU](#quick-start--tensor-computation-on-cpu)
2. [Load a GGUF Model File](#load-a-gguf-model-file)
3. [Multi-Backend Inference (CPU + GPU)](#multi-backend-inference-cpu--gpu)
4. [Transformer Attention Block](#transformer-attention-block)
5. [Simple Linear Layer Training (AdamW)](#simple-linear-layer-training-adamw)
6. [Quantize Model Weights](#quantize-model-weights)
7. [Write a GGUF File](#write-a-gguf-file)
8. [Custom Operator](#custom-operator)

---

## Quick Start — Tensor Computation on CPU

Minimal example: allocate a context, create tensors, build a graph, compute.

```c
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

int main(void) {
    // 1. Allocate a memory pool (64 MB)
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors: a [4] + b [4] = c [4]
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    // 3. Fill with data
    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {10, 20, 30, 40};
    memcpy(ggml_get_data(a), a_data, sizeof(a_data));
    memcpy(ggml_get_data(b), b_data, sizeof(b_data));

    // 4. Build computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * c  = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, c);

    // 5. Execute via CPU backend
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_graph_compute(cpu, gf);

    // 6. Read results
    float * result = ggml_get_data_f32(c);
    for (int i = 0; i < 4; i++) {
        printf("c[%d] = %.0f\n", i, result[i]);  // 11, 22, 33, 44
    }

    ggml_backend_free(cpu);
    ggml_free(ctx);
    return 0;
}
```

---

## Load a GGUF Model File

```c
#include "ggml.h"
#include "gguf.h"

void load_model(const char * path) {
    // Load metadata only first
    struct gguf_init_params meta_params = { .no_alloc = true, .ctx = NULL };
    struct gguf_context * gguf = gguf_init_from_file(path, meta_params);
    if (!gguf) { fprintf(stderr, "Failed to load %s\n", path); return; }

    printf("GGUF version: %u\n", gguf_get_version(gguf));
    printf("Tensors: %ld\n", gguf_get_n_tensors(gguf));

    // Read architecture metadata
    int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id >= 0) {
        printf("Architecture: %s\n", gguf_get_val_str(gguf, arch_id));
    }

    // List all tensors
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        printf("  tensor[%ld]: %s  (%s)\n", i,
               gguf_get_tensor_name(gguf, i),
               ggml_type_name(gguf_get_tensor_type(gguf, i)));
    }
    gguf_free(gguf);

    // Load with tensor data into a ggml context
    struct ggml_context * model_ctx = NULL;
    struct gguf_init_params data_params = { .no_alloc = false, .ctx = &model_ctx };
    gguf = gguf_init_from_file(path, data_params);

    // Access a specific tensor by name
    struct ggml_tensor * embed = ggml_get_tensor(model_ctx, "token_embd.weight");
    if (embed) {
        printf("Embedding shape: [%ld, %ld]\n", embed->ne[0], embed->ne[1]);
    }

    gguf_free(gguf);
    ggml_free(model_ctx);
}
```

---

## Multi-Backend Inference (CPU + GPU)

Use the scheduler to split a graph across CPU and GPU automatically:

```c
#include "ggml-backend.h"
#include "ggml-alloc.h"

void run_multi_backend(struct ggml_context * ctx, struct ggml_cgraph * gf) {
    // Load all available backends
    ggml_backend_load_all();

    // Create backends (GPU first = higher priority)
    ggml_backend_t gpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();

    if (!gpu_backend) {
        // Fall back to CPU only
        ggml_backend_graph_compute(cpu_backend, gf);
        ggml_backend_free(cpu_backend);
        return;
    }

    // Build scheduler with GPU first
    ggml_backend_t backends[] = { gpu_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, NULL, 2,
        GGML_DEFAULT_GRAPH_SIZE,
        false,  // parallel
        true    // op_offload
    );

    // Reserve memory
    ggml_backend_sched_reserve(sched, gf);

    // Run
    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Compute failed: %s\n", ggml_status_to_string(status));
    }
    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_free(sched);
    ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);
}
```

---

## Transformer Attention Block

A minimal single-head self-attention layer using Flash Attention:

```c
struct ggml_tensor * self_attention(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,        // [d_model, n_tokens]
        struct ggml_tensor  * W_q,      // [d_model, d_head]
        struct ggml_tensor  * W_k,
        struct ggml_tensor  * W_v,
        struct ggml_tensor  * pos,      // [n_tokens] I32 positions
        int                   n_rot,
        int                   n_ctx_orig) {

    int64_t d_head   = W_q->ne[1];
    int64_t n_tokens = x->ne[1];
    float   scale    = 1.0f / sqrtf((float)d_head);

    // Project
    struct ggml_tensor * q = ggml_mul_mat(ctx, W_q, x);  // [d_head, n_tokens]
    struct ggml_tensor * k = ggml_mul_mat(ctx, W_k, x);
    struct ggml_tensor * v = ggml_mul_mat(ctx, W_v, x);

    // RoPE
    q = ggml_rope_ext(ctx, q, pos, NULL,
                      n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                      10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, NULL,
                      n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                      10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Reshape for flash attn: [d_head, 1, n_tokens]
    q = ggml_reshape_3d(ctx, q, d_head, 1, n_tokens);
    k = ggml_reshape_3d(ctx, k, d_head, 1, n_tokens);
    v = ggml_reshape_3d(ctx, v, d_head, 1, n_tokens);

    // Flash Attention (causal = pass diagonal mask)
    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v,
                                                   NULL,   // no mask = unmasked
                                                   scale, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, out, d_head, n_tokens);
}
```

---

## Simple Linear Layer Training (AdamW)

```c
#include "ggml-opt.h"
#include "ggml-backend.h"

void train_linear_classifier(int n_features, int n_classes, int n_samples) {
    // Build model context
    struct ggml_init_params mparams = { .mem_size = 16 * 1024 * 1024 };
    struct ggml_context * ctx_model = ggml_init(mparams);

    struct ggml_tensor * W = ggml_new_tensor_2d(ctx_model, GGML_TYPE_F32, n_features, n_classes);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx_model, GGML_TYPE_F32, n_classes);
    ggml_set_param(W);  // mark as trainable
    ggml_set_param(b);
    ggml_set_name(W, "W");
    ggml_set_name(b, "b");

    // Compute graph: y = x @ W^T + b
    struct ggml_init_params cparams = { .mem_size = 32 * 1024 * 1024 };
    struct ggml_context * ctx_compute = ggml_init(cparams);
    struct ggml_tensor * inputs  = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, n_features, 32/*batch*/);
    struct ggml_tensor * outputs = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, W, inputs), b);
    ggml_set_input(inputs);
    ggml_set_output(outputs);

    // Dataset
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_I32, n_features, 1, n_samples, 1);
    // ... fill dataset->data, dataset->labels ...

    // Backend scheduler
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, NULL, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // Train
    ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
                 GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
                 GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                 ggml_opt_get_default_optimizer_params,
                 20,    // epochs
                 32,    // batch size
                 0.1f,  // validation split
                 false);

    ggml_opt_dataset_free(dataset);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx_compute);
    ggml_free(ctx_model);
}
```

---

## Quantize Model Weights

```c
#include "ggml.h"

size_t quantize_tensor(const float * f32_data, void * q_data,
                       int64_t n_rows, int64_t n_per_row,
                       enum ggml_type target_type) {
    // Initialize quantization tables
    ggml_quantize_init(target_type);

    size_t total_bytes = ggml_quantize_chunk(
        target_type,
        f32_data,      // source floats
        q_data,        // destination buffer (pre-allocated)
        0,             // start row
        n_rows,        // number of rows
        n_per_row,     // elements per row
        NULL           // importance matrix (NULL = uniform)
    );

    ggml_quantize_free();
    return total_bytes;
}

// Usage:
// size_t q_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row) * n_rows;
// void * q_buf = malloc(q_size);
// quantize_tensor(my_floats, q_buf, n_rows, n_per_row, GGML_TYPE_Q4_0);
```

---

## Write a GGUF File

```c
#include "gguf.h"
#include "ggml.h"

void write_model(const char * out_path,
                 struct ggml_tensor * weights,
                 const char * arch) {
    struct gguf_context * gguf = gguf_init_empty();

    // Set standard metadata
    gguf_set_val_str(gguf, "general.architecture",    arch);
    gguf_set_val_str(gguf, "general.name",            "my-model");
    gguf_set_val_u32(gguf, "general.file_type",       0);  // GGML_FTYPE_ALL_F32

    // Add tensors
    gguf_add_tensor(gguf, weights);
    // Note: tensor data must be contiguous in host memory
    gguf_set_tensor_data(gguf, ggml_get_name(weights), ggml_get_data(weights));

    bool ok = gguf_write_to_file(gguf, out_path, false);
    printf("Write %s: %s\n", out_path, ok ? "OK" : "FAILED");
    gguf_free(gguf);
}
```

---

## Custom Operator

Implement a custom element-wise operation that runs in parallel:

```c
#include "ggml.h"
#include "ggml-cpu.h"

// Custom op: dst[i] = a[i] * a[i] + 1
static void my_op(struct ggml_tensor * dst,
                  const struct ggml_tensor * a,
                  int ith, int nth,   // thread index and total threads
                  void * userdata) {
    int64_t n = ggml_nelements(a);
    int64_t per_thread = (n + nth - 1) / nth;
    int64_t start = ith * per_thread;
    int64_t end   = start + per_thread < n ? start + per_thread : n;

    const float * src = (const float *) a->data;
    float       * out = (float *)       dst->data;

    for (int64_t i = start; i < end; i++) {
        out[i] = src[i] * src[i] + 1.0f;
    }
    (void)userdata;
}

// Usage in graph construction:
// struct ggml_tensor * result = ggml_map_custom1(ctx, input, my_op, GGML_N_TASKS_MAX, NULL);
```
