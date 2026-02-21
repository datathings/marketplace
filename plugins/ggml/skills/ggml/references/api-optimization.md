# Optimization & Training API

High-level training framework built on top of the ggml backend system.

## Table of Contents
1. [Types & Enums](#types--enums)
2. [Dataset](#dataset)
3. [Optimizer Context](#optimizer-context)
4. [Result Collection](#result-collection)
5. [Graph Preparation & Execution](#graph-preparation--execution)
6. [High-Level Training Loop](#high-level-training-loop)

---

## Types & Enums

```c
// Loss function selection
enum ggml_opt_loss_type {
    GGML_OPT_LOSS_TYPE_MEAN,
    GGML_OPT_LOSS_TYPE_SUM,
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
};

// Build type controls whether the graph includes backward pass
enum ggml_opt_build_type {
    GGML_OPT_BUILD_TYPE_FORWARD,
    GGML_OPT_BUILD_TYPE_GRAD,
    GGML_OPT_BUILD_TYPE_OPT,
};

// Optimizer algorithm
enum ggml_opt_optimizer_type {
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    GGML_OPT_OPTIMIZER_TYPE_SGD,
};

// Optimizer hyperparameters (union-style)
struct ggml_opt_optimizer_params {
    struct { float alpha; float beta1; float beta2; float eps; float wd; } adamw;
    struct { float alpha; float wd; } sgd;
};

// Top-level training parameters
struct ggml_opt_params {
    ggml_backend_sched_t backend_sched;       // multi-backend scheduler
    struct ggml_context * ctx_compute;         // compute context
    enum ggml_opt_loss_type loss_type;
    enum ggml_opt_build_type build_type;
    int32_t opt_period;                        // optimizer step every N gradient accumulations
    ggml_opt_get_optimizer_params get_opt_pars; // callback for per-step hyperparams
    void * get_opt_pars_ud;
};

// Typedefs
typedef struct ggml_opt_dataset  * ggml_opt_dataset_t;
typedef struct ggml_opt_context  * ggml_opt_context_t;
typedef struct ggml_opt_result   * ggml_opt_result_t;
typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
typedef void (*ggml_opt_epoch_callback)(
    bool train,
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result,
    int64_t ibatch, int64_t ibatch_max,
    int64_t t_start_us);
```

---

## Dataset

```c
// Create dataset
// type_data: tensor type of datapoints (e.g. GGML_TYPE_F32)
// type_label: tensor type of labels (e.g. GGML_TYPE_I32 for class indices)
// ne_datapoint: number of elements per datapoint
// ne_label: number of elements per label
// ndata: total number of samples
// ndata_shard: shard size for shuffling (use 1 for fully random)
ggml_opt_dataset_t ggml_opt_dataset_init(
    enum ggml_type type_data,
    enum ggml_type type_label,
    int64_t ne_datapoint,
    int64_t ne_label,
    int64_t ndata,
    int64_t ndata_shard);

void    ggml_opt_dataset_free(ggml_opt_dataset_t dataset);
int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset);

// Access raw tensors to fill with data
struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset);
struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset);

// Shuffle dataset (call before each epoch)
// idata: index from which to start shuffling (-1 for all)
void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);

// Get a batch of samples (fills pre-allocated tensors)
// data_batch and labels_batch must match the batch size
void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset,
                                struct ggml_tensor * data_batch,
                                struct ggml_tensor * labels_batch,
                                int64_t ibatch);

// Host-side batch (no tensor required)
void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset,
                                     void * data_batch, size_t nb_data_batch,
                                     void * labels_batch,
                                     int64_t ibatch);
```

---

## Optimizer Context

```c
// Build default params (requires a scheduler and loss type)
struct ggml_opt_params ggml_opt_default_params(ggml_backend_sched_t backend_sched,
                                               enum ggml_opt_loss_type loss_type);

// Built-in optimizer param callbacks
struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata);  // AdamW defaults
struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata); // fixed LR

// Create / destroy optimizer context
ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
void               ggml_opt_free(ggml_opt_context_t opt_ctx);

// Reset: optimizer=true resets momentum/velocity; false only resets accumulators
void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);

// Query
bool               ggml_opt_static_graphs(ggml_opt_context_t opt_ctx);
enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t opt_ctx);
const char *       ggml_opt_optimizer_name(enum ggml_opt_optimizer_type type);

// Access built-in tensors
struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx);
struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx);
struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx);
struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx);
struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx);      // predictions
struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx);  // number correct
struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node);
```

---

## Result Collection

```c
ggml_opt_result_t ggml_opt_result_init(void);
void              ggml_opt_result_free(ggml_opt_result_t result);
void              ggml_opt_result_reset(ggml_opt_result_t result);

// Query results
void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata);
void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc);         // loss ± uncertainty
void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred);                      // predictions array
void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc); // accuracy ± uncertainty
```

---

## Graph Preparation & Execution

For fine-grained control over the training loop:

```c
// Prepare allocations for a user-defined graph
// inputs: tensor that will receive data batches
// outputs: tensor that produces logits / predictions
void ggml_opt_prepare_alloc(ggml_opt_context_t opt_ctx,
                            struct ggml_context * ctx_compute,
                            struct ggml_cgraph * gf,
                            struct ggml_tensor * inputs,
                            struct ggml_tensor * outputs);

// Allocate compute graph (backward=true includes gradient computation)
void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward);

// Run one forward/backward/optimizer step and accumulate results
void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result);
```

---

## High-Level Training Loop

```c
// Run one epoch over a dataset
// result_train: training set metrics (updated during training batches)
// result_eval: evaluation set metrics (updated during eval batches)
// idata_split: index in dataset where train/eval split occurs
// callback_train / callback_eval: progress callbacks (or NULL)
void ggml_opt_epoch(ggml_opt_context_t opt_ctx,
                    ggml_opt_dataset_t dataset,
                    ggml_opt_result_t result_train,
                    ggml_opt_result_t result_eval,
                    int64_t idata_split,
                    ggml_opt_epoch_callback callback_train,
                    ggml_opt_epoch_callback callback_eval);

// Built-in progress bar callback
void ggml_opt_epoch_callback_progress_bar(
    bool train,
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result,
    int64_t ibatch, int64_t ibatch_max,
    int64_t t_start_us);

// All-in-one training function (simplest API)
void ggml_opt_fit(
    ggml_backend_sched_t backend_sched,
    struct ggml_context * ctx_compute,
    struct ggml_tensor * inputs,
    struct ggml_tensor * outputs,
    ggml_opt_dataset_t dataset,
    enum ggml_opt_loss_type loss_type,
    enum ggml_opt_optimizer_type optimizer,
    ggml_opt_get_optimizer_params get_opt_pars,
    int64_t nepoch,
    int64_t nbatch_logical,
    float val_split,           // fraction of dataset for validation (0 = no validation)
    bool silent);
```

**Example — simple classifier training:**
```c
// 1. Build model graph (ctx_compute, inputs, outputs)
// 2. Create dataset
ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
    GGML_TYPE_F32, GGML_TYPE_I32,
    input_features,  // ne per sample
    1,               // 1 label per sample
    n_samples,       // total samples
    1                // shard size
);
// Fill dataset->data and dataset->labels tensors...

// 3. Train with ggml_opt_fit
ggml_opt_fit(
    backend_sched,
    ctx_compute,
    inputs,
    outputs,
    dataset,
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    ggml_opt_get_default_optimizer_params,
    10,    // epochs
    32,    // batch size
    0.1f,  // 10% validation split
    false  // print progress
);

ggml_opt_dataset_free(dataset);
```
