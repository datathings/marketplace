# llama.cpp Workflows and Examples

Complete working examples for common llama.cpp tasks. This document combines workflow patterns with production-ready applications.

## Table of Contents

### Basic Workflows (1-5)
1. [Basic Text Generation](#1-basic-text-generation)
2. [Chat with System Prompt](#2-chat-with-system-prompt)
3. [Embeddings Extraction](#3-embeddings-extraction)
4. [Batch Processing](#4-batch-processing)
5. [Multiple Sequences](#5-multiple-sequences)

### Intermediate Workflows (6-10)
6. [Using LoRA Adapters](#6-using-lora-adapters)
7. [State Management](#7-state-management)
8. [Custom Sampling](#8-custom-sampling)
9. [Encoder-Decoder Models](#9-encoder-decoder-models)
10. [Memory Management Patterns](#10-memory-management-patterns)

### Advanced Workflows - b7572 Features (11-13)
11. [Advanced Sampling: XTC + DRY](#11-advanced-sampling-xtc--dry)
12. [Per-Sequence State Management](#12-per-sequence-state-management)
13. [Model Architecture Detection](#13-model-architecture-detection)

### Production Examples (14-15)
14. [Interactive Chat Application](#14-interactive-chat-application)
15. [Streaming Generation](#15-streaming-generation)

### Reference
- [Best Practices](#best-practices)
- [Compilation](#compilation)

---

## 1. Basic Text Generation

Complete example of loading a model and generating text.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    // 1. Initialize backend
    llama_backend_init();

    // 2. Load model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 32;  // Offload layers to GPU

    struct llama_model * model = llama_model_load_from_file(
        "model.gguf",
        model_params
    );

    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // 3. Create context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;       // Context size
    ctx_params.n_batch = 512;      // Batch size for prompt processing
    ctx_params.n_threads = 8;      // Number of threads

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // 4. Tokenize input
    const char * prompt = "Once upon a time";
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    llama_token tokens[512];
    int n_tokens = llama_tokenize(
        vocab,
        prompt,
        strlen(prompt),
        tokens,
        512,
        true,   // add_special (add BOS if needed)
        false   // parse_special
    );

    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // 5. Setup sampler
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    // 6. Process prompt
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode prompt\n");
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // 7. Generate tokens
    int n_gen = 0;
    int max_tokens = 100;

    while (n_gen < max_tokens) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        // Check for EOS
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Convert token to text and print
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, 256, 0, false);
        if (len > 0) {
            fwrite(buf, 1, len, stdout);
            fflush(stdout);
        }

        // Prepare batch for next token
        batch = llama_batch_get_one(&new_token, 1);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Failed to decode\n");
            break;
        }

        n_gen++;
    }

    printf("\n");

    // 8. Cleanup
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 2. Chat with System Prompt

Using chat templates for conversational AI.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>

int main() {
    llama_backend_init();

    // Load model (same as basic example)
    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    // Setup chat messages
    llama_chat_message messages[] = {
        {"system", "You are a helpful assistant."},
        {"user", "What is the capital of France?"}
    };

    // Apply chat template
    char prompt[2048];
    int prompt_len = llama_chat_apply_template(
        NULL,        // Use model's default template
        messages,
        2,           // Number of messages
        true,        // Add assistant start token
        prompt,
        sizeof(prompt)
    );

    if (prompt_len < 0) {
        fprintf(stderr, "Chat template failed\n");
        return 1;
    }

    printf("Formatted prompt:\n%s\n", prompt);

    // Tokenize and generate (same as basic example)
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tokens[512];
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, tokens, 512, true, false);

    // ... continue with generation ...

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 3. Embeddings Extraction

Extract embeddings for semantic search, similarity, and clustering tasks.

```cpp
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

// Normalize embeddings to unit length
static void normalize_embeddings(float * emb, int n_embd) {
    float norm = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        norm += emb[i] * emb[i];
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        for (int i = 0; i < n_embd; i++) {
            emb[i] /= norm;
        }
    }
}

// Compute cosine similarity
static float cosine_similarity(const float * a, const float * b, int n_embd) {
    float dot = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

int main(int argc, char ** argv) {
    const char * model_path = "model.gguf";

    // Multiple texts to embed
    std::vector<std::string> texts = {
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above an idle canine",
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons"
    };

    ggml_backend_load_all();

    // 1. Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    // 2. Create context with embeddings enabled
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.embeddings = true;  // CRITICAL: Enable embeddings
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // Mean pooling
    ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_pooling_type pooling = llama_pooling_type(ctx);
    llama_memory_t mem = llama_get_memory(ctx);

    // 3. Process texts in batches
    const int n_texts = texts.size();
    std::vector<float> all_embeddings(n_texts * n_embd);

    for (int i = 0; i < n_texts; i++) {
        // Clear KV cache (not needed for embeddings)
        llama_memory_clear(mem, true);

        // Tokenize
        const int n_tokens = -llama_tokenize(
            vocab,
            texts[i].c_str(),
            texts[i].size(),
            NULL,
            0,
            true,
            true
        );

        std::vector<llama_token> tokens(n_tokens);
        if (llama_tokenize(vocab, texts[i].c_str(), texts[i].size(),
                          tokens.data(), tokens.size(), true, true) < 0) {
            fprintf(stderr, "Failed to tokenize text %d\n", i);
            continue;
        }

        // Create batch for this sequence
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

        // Decode
        if (llama_decode(ctx, batch) < 0) {
            fprintf(stderr, "Failed to decode text %d\n", i);
            continue;
        }

        // Get embeddings
        float * emb = nullptr;

        if (pooling == LLAMA_POOLING_TYPE_NONE) {
            // Get token embeddings (last token)
            emb = llama_get_embeddings_ith(ctx, -1);
        } else {
            // Get sequence embeddings (pooled)
            emb = llama_get_embeddings_seq(ctx, 0);
        }

        if (!emb) {
            fprintf(stderr, "Failed to get embeddings for text %d\n", i);
            continue;
        }

        // Copy and normalize
        float * out = &all_embeddings[i * n_embd];
        for (int j = 0; j < n_embd; j++) {
            out[j] = emb[j];
        }
        normalize_embeddings(out, n_embd);

        printf("Embedded text %d: \"%s\"\n", i, texts[i].c_str());
        printf("  First 5 dimensions: ");
        for (int j = 0; j < 5 && j < n_embd; j++) {
            printf("%.4f ", out[j]);
        }
        printf("...\n");
    }

    // 4. Compute similarity matrix
    printf("\nCosine Similarity Matrix:\n");
    printf("     ");
    for (int i = 0; i < n_texts; i++) {
        printf("  T%d  ", i);
    }
    printf("\n");

    for (int i = 0; i < n_texts; i++) {
        printf("T%d: ", i);
        for (int j = 0; j < n_texts; j++) {
            float sim = cosine_similarity(
                &all_embeddings[i * n_embd],
                &all_embeddings[j * n_embd],
                n_embd
            );
            printf(" %.3f", sim);
        }
        printf("\n");
    }

    printf("\nText pairs with high similarity (> 0.8):\n");
    for (int i = 0; i < n_texts; i++) {
        for (int j = i + 1; j < n_texts; j++) {
            float sim = cosine_similarity(
                &all_embeddings[i * n_embd],
                &all_embeddings[j * n_embd],
                n_embd
            );
            if (sim > 0.8f) {
                printf("  Text %d <-> Text %d: %.3f\n", i, j, sim);
                printf("    \"%s\"\n", texts[i].c_str());
                printf("    \"%s\"\n", texts[j].c_str());
            }
        }
    }

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
```

**Key Features:**
- Embeddings context configuration
- Support for different pooling types
- Embedding normalization
- Similarity computation

---

## 4. Batch Processing

Process multiple prompts in a single batch for efficiency.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Multiple prompts
    const char * prompts[] = {
        "The weather today is",
        "In the year 2050,",
        "Once upon a time"
    };
    int n_prompts = 3;

    // Allocate batch manually for multiple sequences
    struct llama_batch batch = llama_batch_init(512, 0, n_prompts);

    for (int i = 0; i < n_prompts; i++) {
        llama_token tokens[128];
        int n = llama_tokenize(vocab, prompts[i], strlen(prompts[i]), tokens, 128, true, false);

        // Add tokens to batch
        for (int j = 0; j < n; j++) {
            batch.token[batch.n_tokens] = tokens[j];
            batch.pos[batch.n_tokens] = j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens] = &((llama_seq_id[]){i})[0];  // Sequence i
            batch.logits[batch.n_tokens] = (j == n - 1);  // Only last token outputs logits
            batch.n_tokens++;
        }
    }

    // Decode batch
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode batch\n");
        return 1;
    }

    // Sample from each sequence
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    for (int i = 0; i < n_prompts; i++) {
        float * logits = llama_get_logits_ith(ctx, i);
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token_data_array candidates = {
            .data = malloc(n_vocab * sizeof(llama_token_data)),
            .size = n_vocab,
            .selected = -1,
            .sorted = false
        };

        for (int j = 0; j < n_vocab; j++) {
            candidates.data[j].id = j;
            candidates.data[j].logit = logits[j];
            candidates.data[j].p = 0.0f;
        }

        llama_sampler_apply(sampler, &candidates);
        llama_token token = candidates.data[candidates.selected].id;

        // Print result
        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, 256, 0, false);
        printf("Prompt %d next token: %.*s\n", i, len, buf);

        free(candidates.data);
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 5. Multiple Sequences

Process multiple independent sequences in parallel using the same context.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_seq_max = 4;  // Support up to 4 sequences

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    llama_memory_t mem = llama_get_memory(ctx);

    // Clear all sequences
    llama_memory_clear(mem, false);

    // Process sequence 0
    const char * prompt1 = "Hello";
    // ... tokenize and decode for seq_id = 0 ...

    // Process sequence 1 independently
    const char * prompt2 = "Goodbye";
    // ... tokenize and decode for seq_id = 1 ...

    // Copy sequence 0 to sequence 2 (for speculative decoding or similar)
    llama_memory_seq_cp(mem, 0, 2, -1, -1);  // Copy all positions

    // Remove sequence 1
    llama_memory_seq_rm(mem, 1, -1, -1);  // Remove all positions

    // Keep only sequence 0, remove all others
    llama_memory_seq_keep(mem, 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 6. Using LoRA Adapters

Load and apply LoRA adapters to customize model behavior.

```c
#include "llama.h"
#include <stdio.h>

int main() {
    llama_backend_init();

    // Load base model
    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("base-model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    // Load LoRA adapter
    struct llama_adapter_lora * lora = llama_adapter_lora_init(
        model,
        "lora-adapter.gguf"
    );

    if (!lora) {
        fprintf(stderr, "Failed to load LoRA adapter\n");
        return 1;
    }

    // Apply LoRA to context
    float scale = 1.0;  // LoRA scaling factor
    if (llama_set_adapter_lora(ctx, lora, scale) < 0) {
        fprintf(stderr, "Failed to apply LoRA adapter\n");
        return 1;
    }

    // Use context with LoRA applied...
    // ... generate text ...

    // Remove LoRA adapter
    llama_rm_adapter_lora(ctx, lora);

    // Or clear all adapters
    llama_clear_adapter_lora(ctx);

    // Note: LoRA adapters are automatically freed with the model
    // llama_adapter_lora_free() is deprecated

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 7. State Management

Save and restore inference state for resuming generation.

```c
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Process some tokens
    const char * prompt = "The quick brown fox";
    llama_token tokens[128];
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, 128, true, false);

    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    llama_decode(ctx, batch);

    // Save state to memory
    size_t state_size = llama_state_get_size(ctx);
    uint8_t * state_data = malloc(state_size);

    size_t written = llama_state_get_data(ctx, state_data, state_size);
    printf("Saved %zu bytes of state\n", written);

    // Save state to file
    bool success = llama_state_save_file(ctx, "state.bin", tokens, n_tokens);
    if (!success) {
        fprintf(stderr, "Failed to save state to file\n");
    }

    // Clear context
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    // Restore state from memory
    size_t read = llama_state_set_data(ctx, state_data, state_size);
    printf("Restored %zu bytes of state\n", read);

    // Or restore from file
    llama_token loaded_tokens[128];
    size_t n_loaded_tokens;
    success = llama_state_load_file(
        ctx,
        "state.bin",
        loaded_tokens,
        128,
        &n_loaded_tokens
    );

    if (success) {
        printf("Loaded %zu tokens from state file\n", n_loaded_tokens);
    }

    free(state_data);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 8. Custom Sampling

Implement custom sampling strategies.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    // Advanced sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );

    // Add penalties for repetition
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        64,     // penalty_last_n (last 64 tokens)
        1.1,    // penalty_repeat (1.1x penalty)
        0.0,    // penalty_freq (disabled)
        0.0     // penalty_present (disabled)
    ));

    // Add top-k filtering
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));

    // Add top-p (nucleus) filtering
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));

    // Add min-p filtering
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1));

    // Add temperature
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));

    // Final sampler (required)
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    // Use sampler in generation loop...
    // llama_token token = llama_sampler_sample(sampler, ctx, -1);

    // Get sampler performance stats
    llama_perf_sampler_print(sampler);

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 9. Encoder-Decoder Models

Use encoder-decoder models (like T5, BART).

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("encoder-decoder.gguf", model_params);

    // Check if model has encoder and decoder
    bool has_encoder = llama_model_has_encoder(model);
    bool has_decoder = llama_model_has_decoder(model);

    printf("Model has encoder: %s\n", has_encoder ? "yes" : "no");
    printf("Model has decoder: %s\n", has_decoder ? "yes" : "no");

    struct llama_context_params ctx_params = llama_context_default_params();
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // 1. Encode input with llama_encode (for encoder)
    const char * input = "Translate to French: Hello, how are you?";
    llama_token input_tokens[128];
    int n_input = llama_tokenize(vocab, input, strlen(input), input_tokens, 128, true, false);

    struct llama_batch encoder_batch = llama_batch_get_one(input_tokens, n_input);

    if (llama_encode(ctx, encoder_batch) != 0) {
        fprintf(stderr, "Encoding failed\n");
        return 1;
    }

    // 2. Start decoder with special start token
    llama_token decoder_start = llama_model_decoder_start_token(model);
    llama_token decoder_tokens[256];
    decoder_tokens[0] = decoder_start;
    int n_decoded = 1;

    // 3. Decode with llama_decode (for decoder)
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    while (n_decoded < 256) {
        struct llama_batch decoder_batch = llama_batch_get_one(
            &decoder_tokens[n_decoded - 1],
            1
        );

        if (llama_decode(ctx, decoder_batch) != 0) {
            fprintf(stderr, "Decoding failed\n");
            break;
        }

        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        decoder_tokens[n_decoded++] = token;

        // Print token
        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, 256, 0, false);
        fwrite(buf, 1, len, stdout);
        fflush(stdout);
    }

    printf("\n");

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 10. Memory Management Patterns

Advanced KV cache manipulation for efficient inference.

```c
#include "llama.h"
#include <stdio.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_seq_max = 2;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    llama_memory_t mem = llama_get_memory(ctx);

    // Pattern 1: Sliding window - remove old tokens when cache is full
    llama_pos max_pos = llama_memory_seq_pos_max(mem, 0);
    if (max_pos >= 2000) {
        // Remove first 500 tokens
        llama_memory_seq_rm(mem, 0, 0, 500);
        // Shift remaining tokens down by 500 positions
        llama_memory_seq_add(mem, 0, 500, -1, -500);
    }

    // Pattern 2: Context shifting - divide positions to fit more context
    if (llama_memory_can_shift(mem)) {
        // Compress positions by 2x (keep every other position)
        llama_memory_seq_div(mem, 0, 0, -1, 2);
    }

    // Pattern 3: Fork sequence for parallel paths (e.g., speculative decoding)
    llama_memory_seq_cp(mem, 0, 1, -1, -1);

    // Pattern 4: Prefix caching - keep common prefix, fork for variants
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    llama_memory_seq_cp(mem, 0, 2, -1, -1);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

---

## 11. Advanced Sampling: XTC + DRY

**NEW in b7572** - Demonstrates new XTC and DRY samplers for reducing repetition and increasing diversity.

```c
#include "llama.h"
#include <stdio.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Create advanced sampler chain with XTC + DRY
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );

    // Add DRY sampler (reduces repetition)
    const char * seq_breakers[] = {"\n", ".", "?", "!", ",", ":", ";", ")"};
    llama_sampler_chain_add(sampler,
        llama_sampler_init_dry(
            vocab,
            llama_model_n_ctx_train(model),
            0.8,    // dry_multiplier
            1.75,   // dry_base
            2,      // dry_allowed_length
            256,    // dry_penalty_last_n
            seq_breakers, 8
        )
    );

    // Add top-k
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));

    // Add XTC (excludes top choices for diversity)
    llama_sampler_chain_add(sampler,
        llama_sampler_init_xtc(
            0.1,    // probability
            0.5,    // threshold
            1,      // min_keep
            1234    // seed
        )
    );

    // Add temperature
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));

    // Add final dist sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    // Tokenize and generate...
    const char * prompt = "Write a creative story about";
    llama_token tokens[2048];
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, 2048, true, false);

    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    llama_decode(ctx, batch);

    printf("%s", prompt);
    for (int i = 0; i < 200; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, 256, 0, false);
        if (n > 0) {
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
        }

        batch = llama_batch_get_one(&new_token, 1);
        llama_decode(ctx, batch);
    }
    printf("\n");

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

**Key Points:**
- DRY sampler prevents repetition using sequence breakers
- XTC adds diversity by occasionally excluding top tokens
- Combine with traditional samplers for best results

---

## 12. Per-Sequence State Management

**NEW in b7572** - Save and load state for individual sequences.

```c
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Simulate conversation with Alice (sequence 0)
    const char * alice_msg = "Hi, I'm Alice. Tell me about quantum physics.";
    llama_token alice_tokens[256];
    int n_alice = llama_tokenize(vocab, alice_msg, strlen(alice_msg), alice_tokens, 256, true, false);

    // Process Alice's sequence
    struct llama_batch batch = llama_batch_init(256, 0, 1);
    for (int i = 0; i < n_alice; i++) {
        batch.token[batch.n_tokens] = alice_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;  // sequence 0
        batch.logits[batch.n_tokens] = (i == n_alice - 1);
        batch.n_tokens++;
    }
    llama_decode(ctx, batch);

    // Save Alice's sequence state to file
    printf("Saving Alice's conversation state...\n");
    size_t saved_bytes = llama_state_seq_save_file(
        ctx, "alice.state", 0, alice_tokens, n_alice
    );
    printf("Saved %zu bytes\n", saved_bytes);

    // Clear memory and work with Bob
    llama_memory_clear(llama_get_memory(ctx));

    // Later: restore Alice's conversation
    printf("\nRestoring Alice's conversation...\n");
    llama_token restored_tokens[1024];
    size_t restored_count;
    size_t loaded_bytes = llama_state_seq_load_file(
        ctx, "alice.state", 0, restored_tokens, 1024, &restored_count
    );
    printf("Loaded %zu bytes, %zu tokens\n", loaded_bytes, restored_count);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
```

**Use Cases:**
- Multi-user chat applications with context switching
- Branching conversation trees
- Efficient memory management for multiple conversations

---

## 13. Model Architecture Detection

**NEW in b7572** - Detect and handle different model architectures dynamically.

```c
#include "llama.h"
#include <stdio.h>

void handle_model(struct llama_model * model) {
    printf("=== Model Architecture Analysis ===\n\n");

    if (llama_model_has_encoder(model)) {
        printf("✓ Encoder-decoder model (T5/BART/Flan-T5)\n");
        llama_token decoder_start = llama_model_decoder_start_token(model);
        printf("  Decoder start token: %d\n", decoder_start);
        printf("  Usage: Use llama_encode() for input, llama_decode() for generation\n\n");

    } else if (llama_model_is_recurrent(model)) {
        printf("✓ Recurrent model (Mamba/RWKV)\n");
        printf("  Features: Different KV cache behavior, efficient long contexts\n\n");

    } else if (llama_model_is_hybrid(model)) {
        printf("✓ Hybrid model (Jamba/Granite MoE)\n");
        printf("  Features: Mix of attention and RNN/SSM layers\n\n");

    } else {
        printf("✓ Standard transformer model\n\n");

        int32_t swa_size = llama_model_n_swa(model);
        if (swa_size > 0) {
            printf("  Sliding Window Attention (SWA) detected\n");
            printf("  Window size: %d tokens\n", swa_size);
            printf("  Tip: Set ctx_params.swa_full = true for full context access\n\n");
        }

        enum llama_rope_type rope = llama_model_rope_type(model);
        const char * rope_names[] = {
            "None", "Normal", "NeoX", "Multi-Res", "Improved Multi-Res", "Vision"
        };
        printf("  RoPE type: %s\n", rope_names[rope]);
        if (rope == LLAMA_ROPE_TYPE_VISION) {
            printf("  Note: Vision model - designed for multimodal input\n");
        }
        printf("\n");
    }

    int32_t n_classes = llama_model_n_cls_out(model);
    if (n_classes > 0) {
        printf("✓ Classifier model detected\n");
        printf("  Output classes: %d\n", n_classes);
        for (int i = 0; i < n_classes && i < 10; i++) {
            const char * label = llama_model_cls_label(model, i);
            if (label) {
                printf("    Class %d: %s\n", i, label);
            }
        }
        printf("\n");
    }

    printf("=== Model Properties ===\n");
    printf("Embedding dimension: %d\n", llama_model_n_embd(model));
    printf("Layers: %d\n", llama_model_n_layer(model));
    printf("Training context: %d tokens\n", llama_model_n_ctx_train(model));
}

int main() {
    llama_backend_init();

    struct llama_model_params params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("model.gguf", params);

    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    handle_model(model);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

**Use Cases:**
- Dynamic model loading systems
- Automatic optimization based on architecture
- Debugging and model analysis tools

---

## 14. Interactive Chat Application

Complete chat application with conversation history management. Demonstrates proper use of chat templates and incremental prompting.

```cpp
#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

int main(int argc, char ** argv) {
    const char * model_path = "model.gguf";
    const int n_ctx = 4096;
    const int n_gpu_layers = 99;

    // Suppress non-error logs
    llama_log_set([](enum ggml_log_level level, const char * text, void * /*user_data*/) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted_prompt(n_ctx);
    int prev_formatted_len = 0;

    const char * chat_template = llama_model_chat_template(model, nullptr);

    printf("Chat started. Type your message and press Enter. Empty line to exit.\n\n");

    while (true) {
        printf("\033[32mYou> \033[0m");
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) {
            break;
        }

        messages.push_back({"user", strdup(user_input.c_str())});

        int new_formatted_len = llama_chat_apply_template(
            chat_template,
            messages.data(),
            messages.size(),
            true,
            formatted_prompt.data(),
            formatted_prompt.size()
        );

        if (new_formatted_len > (int)formatted_prompt.size()) {
            formatted_prompt.resize(new_formatted_len);
            new_formatted_len = llama_chat_apply_template(
                chat_template,
                messages.data(),
                messages.size(),
                true,
                formatted_prompt.data(),
                formatted_prompt.size()
            );
        }

        if (new_formatted_len < 0) {
            fprintf(stderr, "Failed to apply chat template\n");
            break;
        }

        std::string prompt(
            formatted_prompt.begin() + prev_formatted_len,
            formatted_prompt.begin() + new_formatted_len
        );

        llama_memory_t mem = llama_get_memory(ctx);
        bool is_first_turn = (llama_memory_seq_pos_max(mem, 0) == -1);

        const int n_tokens = -llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.size(),
            NULL,
            0,
            is_first_turn,
            true
        );

        std::vector<llama_token> tokens(n_tokens);
        llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                      tokens.data(), tokens.size(),
                      is_first_turn, true);

        printf("\033[33mAssistant> \033[0m");
        std::string response;

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

        while (true) {
            int n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\n\033[31mContext size exceeded!\033[0m\n");
                goto cleanup;
            }

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Failed to decode\n");
                goto cleanup;
            }

            llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            batch = llama_batch_get_one(&new_token, 1);
        }

        printf("\n\033[0m\n");

        messages.push_back({"assistant", strdup(response.c_str())});

        prev_formatted_len = llama_chat_apply_template(
            chat_template,
            messages.data(),
            messages.size(),
            false,
            nullptr,
            0
        );
    }

cleanup:
    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
```

**Key Features:**
- Incremental prompting (only processes new messages)
- Proper conversation history management
- Context size checking
- Colored terminal output

---

## 15. Streaming Generation

Stream tokens as they're generated, useful for real-time applications and servers.

```cpp
#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>

typedef void (*token_callback_t)(const char * text, int length, void * user_data);

struct streaming_context {
    llama_context * ctx;
    llama_sampler * sampler;
    const llama_vocab * vocab;
    token_callback_t callback;
    void * user_data;
};

void generate_streaming(
    streaming_context * stream_ctx,
    const char * prompt,
    int max_tokens
) {
    const llama_vocab * vocab = stream_ctx->vocab;
    llama_context * ctx = stream_ctx->ctx;
    llama_sampler * sampler = stream_ctx->sampler;

    int n = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode prompt\n");
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int n_generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            if (stream_ctx->callback) {
                stream_ctx->callback(buf, len, stream_ctx->user_data);
            }
        }

        n_generated++;

        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    fprintf(stderr, "\n[Generated %d tokens in %lld ms, %.2f tokens/s]\n",
            n_generated, duration.count(),
            n_generated * 1000.0 / duration.count());
}

void print_token(const char * text, int length, void * user_data) {
    fwrite(text, 1, length, stdout);
    fflush(stdout);
}

void append_to_string(const char * text, int length, void * user_data) {
    std::string * output = (std::string*)user_data;
    output->append(text, length);
}

int main(int argc, char ** argv) {
    const char * model_path = "model.gguf";
    const char * prompt = "Write a short story about a robot:";

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) return 1;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }

    llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    // Example 1: Stream to stdout
    printf("Prompt: %s\n\n", prompt);
    printf("Streaming output:\n");

    streaming_context stream_ctx = {
        ctx,
        sampler,
        llama_model_get_vocab(model),
        print_token,
        nullptr
    };

    generate_streaming(&stream_ctx, prompt, 200);

    printf("\n\n");

    // Example 2: Stream to string
    printf("Capturing to string:\n");
    std::string captured;

    stream_ctx.callback = append_to_string;
    stream_ctx.user_data = &captured;

    llama_memory_clear(llama_get_memory(ctx), true);

    generate_streaming(&stream_ctx, "The meaning of life is", 50);

    printf("\nCaptured text (%zu bytes): %s\n", captured.size(), captured.c_str());

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
```

**Key Features:**
- Callback-based streaming architecture
- Real-time token delivery
- Performance measurement
- Multiple streaming targets

---

## Best Practices

### 1. Always Initialize Backend
```c
llama_backend_init();
// ... your code ...
llama_backend_free();
```

### 2. Check Return Values
```c
if (!model || !ctx) {
    fprintf(stderr, "Initialization failed\n");
    // Cleanup and exit
}
```

### 3. Free Resources in Reverse Order
```c
llama_sampler_free(sampler);  // 1. Free samplers first
llama_free(ctx);              // 2. Free context
llama_model_free(model);      // 3. Free model
llama_backend_free();         // 4. Free backend last
```

### 4. Use Default Params as Base
```c
struct llama_model_params params = llama_model_default_params();
// Override only what you need
params.n_gpu_layers = 32;
```

### 5. Query Actual Context Size
```c
ctx_params.n_ctx = 4096;
struct llama_context * ctx = llama_init_from_model(model, ctx_params);
uint32_t actual_ctx = llama_n_ctx(ctx);  // May differ from requested
```

### 6. Handle Tokenization Buffer Sizes
```c
int n = llama_tokenize(vocab, text, len, tokens, max_tokens, true, false);
if (n < 0) {
    // Buffer too small, need -n tokens
    tokens = realloc(tokens, -n * sizeof(llama_token));
    n = llama_tokenize(vocab, text, len, tokens, -n, true, false);
}
```

### 7. Check for End-of-Generation
```c
if (llama_vocab_is_eog(vocab, token)) {
    break;  // Stop generation
}
```

### 8. Performance Monitoring
```c
llama_perf_context_print(ctx);
llama_perf_sampler_print(sampler);
```

---

## Compilation

All examples can be compiled with:

```bash
# C examples
gcc example.c -o example -I../include -L../build -lllama -lm

# C++ examples
g++ -std=c++11 example.cpp -o example -I../include -L../build -lllama

# With CMake
cmake -B build
cmake --build build
./build/example -m model.gguf
```

**Link flags:**
- `-I../include` - llama.cpp header directory
- `-L../build` - llama.cpp library directory
- `-lllama` - Link against llama library
- `-lm` - Math library (for C examples)
