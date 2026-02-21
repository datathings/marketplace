---
name: ggml
description: "C tensor computation library for ML inference and training. Use when working with ggml graphs, GGUF model files, backend scheduling, quantization, or implementing low-level ML ops in C/C++."
---

# ggml

## Overview

ggml is a minimalistic C tensor computation library powering llama.cpp and many other ML inference engines. It provides:
- A define-and-run computation graph model (similar to TensorFlow 1.x)
- CPU, CUDA, Metal, Vulkan, and other hardware backends
- 40+ quantization formats (Q4_0, Q8_0, Q5_K, etc.)
- GGUF binary file format for model weights and metadata
- Automatic differentiation and AdamW/SGD optimizers
- Zero runtime allocations — all memory is pre-reserved

**Version:** v0.9.7
**Language:** C (C++ optional)
**License:** MIT
**Repo:** https://github.com/ggml-org/ggml

## Quick Start

```c
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

int main(void) {
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,  // 64 MB scratch buffer
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * c = ggml_add(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_graph_compute(backend, gf);

    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
```

## Core Concepts

- **ggml_context** — memory pool that owns all tensors; freed all at once
- **ggml_tensor** — N-D array (max 4 dims); stores type, shape, strides, and a data pointer
- **ggml_cgraph** — lazy computation graph; ops are recorded then executed via a backend
- **ggml_backend_t** — execution engine (CPU, CUDA, Metal, …); use `ggml_backend_load_all()` to discover available hardware
- **ggml_backend_sched_t** — multi-device scheduler that splits a graph across backends automatically
- **GGUF** — binary model format: metadata key-value store + packed tensor data

## API Reference

| Domain | File | Description |
|--------|------|-------------|
| Context, tensors & graphs | [api-core.md](references/api-core.md) | Init, create tensors, graph ops, scalar access, constants |
| Arithmetic & matrix ops | [api-arithmetic.md](references/api-arithmetic.md) | add/mul/matmul, reductions, loss functions, quantize |
| Activations, norms & shapes | [api-activations.md](references/api-activations.md) | relu/gelu/silu, RMS norm, reshape/permute/concat, custom ops |
| Attention, convolution & RoPE | [api-attention.md](references/api-attention.md) | Flash Attention, RoPE variants, 1D/2D/3D conv, pooling, padding |
| Backend, memory & scheduler | [api-backend.md](references/api-backend.md) | Backends, buffer types, scheduler, gallocr, CPU threadpool, F16 conversions |
| GGUF file format | [api-gguf.md](references/api-gguf.md) | Read/write GGUF v3: KV metadata, tensor layout, serialization |
| Optimization & training | [api-optimization.md](references/api-optimization.md) | Datasets, AdamW/SGD optimizer, epoch loop, ggml_opt_fit |
| Working examples | [workflows.md](references/workflows.md) | Quick start, GGUF loading, multi-backend, attention, training, quantize |

## Common Workflows

See [references/workflows.md](references/workflows.md) for complete examples.

Quick reference:
- **Tensor computation on CPU** → workflows.md#quick-start
- **Load GGUF model** → workflows.md#load-a-gguf-model-file
- **Multi-backend (CPU+GPU)** → workflows.md#multi-backend-inference-cpu--gpu
- **Transformer attention** → workflows.md#transformer-attention-block
- **Train a model** → workflows.md#simple-linear-layer-training-adamw
- **Quantize weights** → workflows.md#quantize-model-weights
- **Write GGUF file** → workflows.md#write-a-gguf-file
- **Custom operator** → workflows.md#custom-operator

## Key Considerations

- **Memory is pre-allocated** — choose `mem_size` generously; `ggml_init` fails silently if too small
- **Dimensions are reversed** — `ne[0]` is the innermost (fastest) dimension; for a `[rows × cols]` matrix use `ne0=cols`, `ne1=rows`
- **Graph building does not execute** — operations are recorded lazily; call `ggml_backend_graph_compute()` to run
- **Backend discovery** — call `ggml_backend_load_all()` at startup; use `ggml_backend_init_best()` to pick the best available device
- **Quantized matmul** — `ggml_mul_mat` supports mixed precision (e.g. Q4_0 weights × F32 activations) natively
- **Inplace variants** — `ggml_add_inplace` overwrites tensor `a` and avoids an allocation; only safe when `a` is not used elsewhere in the graph
- **Thread count** — default is 4 threads; use `ggml_backend_cpu_set_n_threads()` or a custom threadpool
