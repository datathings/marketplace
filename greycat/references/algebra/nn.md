# Neural Networks

High-level neural network APIs for regression, classification, and autoencoders with comprehensive preprocessing, training, and inference capabilities.

## Overview

The Neural Networks module provides complete ML workflows:
- **RegressionNetwork**: Continuous value prediction
- **ClassificationNetwork**: Categorical prediction with probabilities
- **AutoEncoderNetwork**: Unsupervised encoding/decoding

All networks support:
- Automatic preprocessing (min-max, standard, PCA scaling)
- Automatic postprocessing (inverse scaling)
- Dense, LSTM, and custom layers
- Multiple optimizers and loss functions
- State management for model persistence

## Network Types

| Type | Use Case | Output | Loss Functions |
|------|----------|--------|----------------|
| `RegressionNetwork` | Predicting continuous values | Float tensor | Square, Absolute |
| `ClassificationNetwork` | Predicting categories | Class probabilities | Categorical/Sparse cross-entropy |
| `AutoEncoderNetwork` | Dimensionality reduction, denoising | Reconstruction | Square, Absolute |

---

## RegressionNetwork

Predicts continuous values (e.g., price, temperature, stock value).

### Creation

```typescript
static fn new(
  inputs: int,                  // Number of input features
  outputs: int,                 // Number of output values
  tensor_type: TensorType,      // f32 or f64
  inputs_gradients: bool?,      // Track input gradients (default: false)
  fixed_batch_size: int?,       // 0 for dynamic (default)
  seed: int?                    // Random seed (default: random)
): RegressionNetwork
```

**Example:**
```typescript
var nn = RegressionNetwork::new(
  inputs: 10,                   // 10 features
  outputs: 1,                   // 1 predicted value
  tensor_type: TensorType::f32,
  inputs_gradients: false,
  fixed_batch_size: 32,
  seed: 42
);
```

### Building the Network

#### addLinearLayer(output: int, use_bias: bool, config: InitializerConfig?)
Adds linear transformation layer.

```typescript
nn.addLinearLayer(64, true, null);
```

#### addDenseLayer(output: int, use_bias: bool, activation: ComputeActivation?, config: InitializerConfig?)
Adds dense layer with activation.

```typescript
nn.addDenseLayer(
  128,                           // 128 neurons
  true,                          // Use bias
  ComputeActivationRelu {},      // ReLU activation
  null                           // Default initializers
);
```

#### addLSTMLayer(output: int, layers: int, sequences: int, use_bias: bool, return_sequences: bool, bidirectional: bool, config: InitializerConfig?)
Adds LSTM recurrent layer.

```typescript
nn.addLSTMLayer(
  output: 64,                    // 64 hidden units
  layers: 2,                     // 2-layer stacked LSTM
  sequences: 10,                 // 10 timesteps
  use_bias: true,
  return_sequences: false,       // Return only last output
  bidirectional: true,           // Bidirectional LSTM
  config: null
);
```

### Configuration

#### setPreProcess(preProcess: PreProcessType, object: any?)
Sets input preprocessing (normalization).

```typescript
var gaussian = GaussianND {};
gaussian.learn(training_data);

// Standard scaling (recommended)
nn.setPreProcess(PreProcessType::standard_scaling, gaussian);

// Or min-max scaling
nn.setPreProcess(PreProcessType::min_max_scaling, gaussian);

// Or PCA
var pca = PCA {};
// ... configure pca ...
nn.setPreProcess(PreProcessType::pca_scaling, pca);
```

#### setPostProcess(postProcess: PostProcessType, object: any?)
Sets output postprocessing.

```typescript
var output_gaussian = GaussianND {};
output_gaussian.learn(training_targets);
nn.setPostProcess(PostProcessType::standard_scaling, output_gaussian);
```

#### setOptimizer(optimizer: ComputeOptimizer?)
Sets the optimization algorithm.

```typescript
nn.setOptimizer(ComputeOptimizerAdam {
  learning_rate: 0.001,
  beta1: 0.9,
  beta2: 0.999,
  smooth_epsilon: 1e-07
});
```

#### setLoss(loss_type: ComputeRegressionLoss?, reduction: ComputeReduction?)
Sets loss function.

```typescript
nn.setLoss(ComputeRegressionLoss::square, ComputeReduction::mean);
// Or
nn.setLoss(ComputeRegressionLoss::abs, ComputeReduction::mean);
```

### Training

#### build(learningMode: bool): ComputeModel
Builds the computational graph.

```typescript
var model = nn.build(true); // true = training mode
```

#### initWithBatch(model: ComputeModel?, engine: ComputeEngine, state: ComputeState?, batch: int): int
Initializes with fixed batch size.

```typescript
var engine = ComputeEngine {};
var max_batch = nn.initWithBatch(null, engine, null, 32);
```

#### initWithMemory(model: ComputeModel?, engine: ComputeEngine, state: ComputeState?, maxMemory: int): int
Initializes with memory constraint.

```typescript
var engine = ComputeEngine {};
var batch_size = nn.initWithMemory(null, engine, null, 100 * 1024 * 1024); // 100MB
```

#### getInput(engine: ComputeEngine): Tensor
Gets input tensor to fill with data.

```typescript
nn.getInput(engine).fill(X_batch);
```

#### getTarget(engine: ComputeEngine): Tensor
Gets target tensor to fill with labels.

```typescript
nn.getTarget(engine).fill(y_batch);
```

#### train(engine: ComputeEngine): Tensor
Executes one training step (forward + backward + optimize).

```typescript
var loss = nn.train(engine);
```

#### miniBatch(engine: ComputeEngine): Tensor
Executes forward + backward (no optimization).

```typescript
var loss = nn.miniBatch(engine); // Accumulate gradients
```

#### optimize(engine: ComputeEngine)
Applies accumulated gradients.

```typescript
nn.optimize(engine);
```

#### validation(engine: ComputeEngine): Tensor
Computes loss without training.

```typescript
var val_loss = nn.validation(engine);
```

#### endEpoch(engine: ComputeEngine)
Signals end of epoch.

```typescript
nn.endEpoch(engine);
```

### Inference

#### initForPrediction(model: ComputeModel?, engine: ComputeEngine, state: ComputeState, batchSize: int)
Initializes for inference (no gradients).

```typescript
var pred_engine = ComputeEngine {};
nn.initForPrediction(null, pred_engine, trained_state, 1);
```

#### predict(engine: ComputeEngine, input: Tensor?): Tensor
Makes predictions.

```typescript
var predictions = nn.predict(engine, test_data);
```

#### getPrediction(engine: ComputeEngine): Tensor
Gets last prediction (after calling predict with null input).

```typescript
nn.getInput(engine).fill(X_test);
var _ = nn.predict(engine, null);
var predictions = nn.getPrediction(engine);
```

### Complete Regression Example

```typescript
// 1. Prepare data
var gaussian = GaussianND {};
gaussian.learn(X_train);

// 2. Create network
var nn = RegressionNetwork::new(10, 1, TensorType::f32, false, 32, 42);

// 3. Configure
nn.setPreProcess(PreProcessType::standard_scaling, gaussian);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(32, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(1, true, null, null);
nn.setOptimizer(ComputeOptimizerAdam {});
nn.setLoss(ComputeRegressionLoss::square, ComputeReduction::mean);

// 4. Train
var engine = ComputeEngine {};
nn.initWithBatch(nn.build(true), engine, null, 32);

for (epoch in 0..100) {
  nn.getInput(engine).fill(X_train);
  nn.getTarget(engine).fill(y_train);
  var loss = nn.train(engine);

  if (epoch % 10 == 0) {
    println("Epoch ${epoch}: Loss = ${loss}");
  }
}

// 5. Save state
var state = ComputeState {};
engine.saveState(state);

// 6. Predict
var predictions = nn.predict(engine, X_test);
```

---

## ClassificationNetwork

Predicts categorical classes with probabilities.

### Creation

```typescript
static fn new(
  inputs: int,                      // Input features
  classes: int,                     // Number of classes
  tensor_type: TensorType,
  inputs_gradients: bool?,
  fixed_batch_size: int?,
  seed: int?,
  calculate_probabilities: bool,   // Return probabilities
  from_logits: bool,                // Input is logits (pre-softmax)
  has_class_weights: bool           // Use class weights
): ClassificationNetwork
```

**Example:**
```typescript
var nn = ClassificationNetwork::new(
  inputs: 784,                      // 28x28 flattened image
  classes: 10,                      // 10 digit classes
  tensor_type: TensorType::f32,
  inputs_gradients: false,
  fixed_batch_size: 32,
  seed: 42,
  calculate_probabilities: true,   // Return probabilities
  from_logits: true,                // Raw scores from network
  has_class_weights: false          // No class weights
);
```

### Additional Methods

#### setLoss(loss_type: ComputeClassificationLoss?, reduction: ComputeReduction?)
Sets classification loss.

```typescript
// Sparse (integer labels)
nn.setLoss(
  ComputeClassificationLoss::sparse_categorical_cross_entropy,
  ComputeReduction::mean
);

// Categorical (one-hot encoded labels)
nn.setLoss(
  ComputeClassificationLoss::categorical_cross_entropy,
  ComputeReduction::mean
);
```

#### getPrediction(engine: ComputeEngine): Tensor
Gets predicted class indices.

```typescript
var classes = nn.getPrediction(engine); // [batch, 1] - class indices
```

#### getProbability(engine: ComputeEngine): Tensor
Gets class probabilities (if calculate_probabilities=true).

```typescript
var probs = nn.getProbability(engine); // [batch, classes]
```

#### getClassWeights(engine: ComputeEngine): Tensor
Gets/sets class weights for imbalanced datasets.

```typescript
var weights = nn.getClassWeights(engine);
weights.fill(class_weight_tensor);
```

#### getConfusion(engine: ComputeEngine): Tensor
Gets confusion matrix.

```typescript
var confusion = nn.getConfusion(engine); // [classes, classes]
```

#### resetConfusion(engine: ComputeEngine)
Resets confusion matrix.

```typescript
nn.resetConfusion(engine);
```

### Static Methods

#### getClassificationMetrics(confusionMatrix: Tensor): ClassificationMetrics
Computes precision, recall, F1-score from confusion matrix.

```typescript
var metrics = ClassificationNetwork::getClassificationMetrics(confusion);
println("Precision: ${metrics.precision}");
println("Recall: ${metrics.recall}");
println("F1-Score: ${metrics.f1Score}");
```

**Returns:**
```typescript
type ClassificationMetrics {
  precision: Array<float?>;  // Per-class + average
  recall: Array<float?>;     // Per-class + average
  f1Score: Array<float?>;    // Per-class + average
}
```

#### getDefaultClassWeights(classDistribution: Array<int>, normalize: bool): Array<float>
Computes class weights for imbalanced datasets.

```typescript
var class_dist = Array<int> {100, 500, 50}; // Class counts
var weights = ClassificationNetwork::getDefaultClassWeights(class_dist, true);
// Returns: higher weights for rare classes
```

### Complete Classification Example

```typescript
// 1. Create network
var nn = ClassificationNetwork::new(
  784, 10, TensorType::f32, false, 32, 42, true, true, false
);

// 2. Build architecture
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(10, true, null, null); // Output logits

// 3. Configure
nn.setOptimizer(ComputeOptimizerAdam {});
nn.setLoss(
  ComputeClassificationLoss::sparse_categorical_cross_entropy,
  ComputeReduction::mean
);

// 4. Train
var engine = ComputeEngine {};
nn.initWithBatch(nn.build(true), engine, null, 32);

for (epoch in 0..epochs) {
  nn.resetConfusion(engine);

  for (batch_X, batch_y in training_data) {
    nn.getInput(engine).fill(batch_X);
    nn.getTarget(engine).fill(batch_y);
    var loss = nn.train(engine);
  }

  // Validation
  for (val_X, val_y in validation_data) {
    nn.getInput(engine).fill(val_X);
    nn.getTarget(engine).fill(val_y);
    var val_loss = nn.validation(engine);
  }

  // Metrics
  var confusion = nn.getConfusion(engine);
  var metrics = ClassificationNetwork::getClassificationMetrics(confusion);
  println("Epoch ${epoch}: F1=${metrics.f1Score[metrics.f1Score.size()-1]}");
}

// 5. Inference
var probs = nn.predict(engine, test_X);    // Probabilities
var classes = nn.getPrediction(engine);    // Class indices
```

---

## AutoEncoderNetwork

Unsupervised learning for encoding/decoding data.

### Creation

```typescript
static fn new(
  inputs: int,              // Input/output size (same)
  tensor_type: TensorType,
  inputs_gradients: bool?,
  fixed_batch_size: int?,
  seed: int?
): AutoEncoderNetwork
```

**Example:**
```typescript
var nn = AutoEncoderNetwork::new(
  inputs: 784,              // 28x28 image
  tensor_type: TensorType::f32,
  inputs_gradients: false,
  fixed_batch_size: 32,
  seed: 42
);
```

### Configuration

#### setEncoderLayer(layerIndex: int)
Sets which layer is the encoding (latent) representation.

```typescript
// Add 3 encoder layers + 3 decoder layers
nn.addDenseLayer(256, true, ComputeActivationRelu {}, null); // 0
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null); // 1
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);  // 2 - encoding
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null); // 3
nn.addDenseLayer(256, true, ComputeActivationRelu {}, null); // 4
nn.addDenseLayer(784, true, null, null);                     // 5 - output

nn.setEncoderLayer(2); // Layer 2 is the latent representation
```

### Training

Same as RegressionNetwork, but target = input (reconstruction).

```typescript
nn.getInput(engine).fill(X_batch);
nn.getTarget(engine).fill(X_batch); // Reconstruct input
var loss = nn.train(engine);
```

### Inference

#### encode(engine: ComputeEngine, input: Tensor?): Tensor
Encodes input to latent representation.

```typescript
var encoded = nn.encode(engine, X_test); // [batch, 64]
```

#### decode(engine: ComputeEngine, input: Tensor?): Tensor
Decodes latent representation to output.

```typescript
var reconstructed = nn.decode(engine, encoded); // [batch, 784]
```

#### getEncoderInput(engine: ComputeEngine): Tensor
Gets encoder input tensor.

#### getDecoderInput(engine: ComputeEngine): Tensor
Gets decoder input tensor.

#### getEncoding(engine: ComputeEngine): Tensor
Gets latent encoding.

#### getDecoding(engine: ComputeEngine): Tensor
Gets decoded output.

### Complete AutoEncoder Example

```typescript
// 1. Create autoencoder
var nn = AutoEncoderNetwork::new(784, TensorType::f32, false, 32, 42);

// 2. Build symmetric architecture
nn.addDenseLayer(256, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);  // Encoding
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(256, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(784, true, ComputeActivationSigmoid {}, null); // Output [0,1]

nn.setEncoderLayer(2); // 64-dim encoding

// 3. Train
var engine = ComputeEngine {};
nn.initWithBatch(nn.build(true), engine, null, 32);

for (epoch in 0..epochs) {
  for (X_batch in training_data) {
    nn.getInput(engine).fill(X_batch);
    nn.getTarget(engine).fill(X_batch); // Reconstruct input
    var loss = nn.train(engine);
  }
}

// 4. Use encoder
var encoded = nn.encode(engine, X_test);        // Compress
var reconstructed = nn.decode(engine, encoded); // Decompress

// 5. Anomaly detection
var reconstruction_error = mse(X_test, reconstructed);
// High error = anomaly
```

---

## Advanced Features

### LSTM Sequence Models

```typescript
// Time series prediction
var nn = RegressionNetwork::new(50, 1, TensorType::f32, false, 32, 42);

nn.addLSTMLayer(
  output: 128,
  layers: 2,
  sequences: 10,        // 10 timesteps
  use_bias: true,
  return_sequences: false,
  bidirectional: true,
  config: null
);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(1, true, null, null);

// Input shape: [10, 32, 50] - (sequences, batch, features)
// Output shape: [32, 1] - (batch, predictions)
```

### Multi-class Text Classification

```typescript
var nn = ClassificationNetwork::new(
  inputs: 512,          // Embedding dimension
  classes: 20,          // 20 categories
  tensor_type: TensorType::f32,
  inputs_gradients: false,
  fixed_batch_size: 64,
  seed: 42,
  calculate_probabilities: true,
  from_logits: true,
  has_class_weights: true // Handle class imbalance
);

nn.addLSTMLayer(256, 1, 100, true, false, true, null); // Bidirectional
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(20, true, null, null);

// Set class weights
var class_counts = Array<int> {100, 500, 200, ...}; // Per class
var weights = ClassificationNetwork::getDefaultClassWeights(class_counts, true);
nn.getClassWeights(engine).fill(weights_tensor);
```

### Transfer Learning

```typescript
// 1. Train base model
var base_nn = RegressionNetwork::new(...);
// ... train base_nn ...
var base_state = ComputeState {};
engine.saveState(base_state);

// 2. Create fine-tuning model with similar architecture
var fine_tune_nn = RegressionNetwork::new(...);
// ... same architecture ...

// 3. Load pre-trained weights
var fine_tune_engine = ComputeEngine {};
fine_tune_nn.initWithBatch(null, fine_tune_engine, base_state, 32);

// 4. Fine-tune with lower learning rate
fine_tune_nn.setOptimizer(ComputeOptimizerAdam { learning_rate: 0.0001 });
// ... train on new task ...
```

---

## Best Practices

### Data Preparation
```typescript
// Always normalize inputs
var gaussian = GaussianND {};
gaussian.learn(X_train);
nn.setPreProcess(PreProcessType::standard_scaling, gaussian);
```

### Architecture Guidelines
- Start simple, add complexity as needed
- Use ReLU for hidden layers
- Use 64-256 neurons per layer
- Add dropout for regularization (via custom layers)

### Training Tips
```typescript
// Early stopping
var best_loss = float::max;
var patience = 0;
var best_state = ComputeState {};

for (epoch in 0..max_epochs) {
  var val_loss = nn.validation(engine);

  if (val_loss < best_loss) {
    best_loss = val_loss;
    patience = 0;
    engine.saveState(best_state);
  } else {
    patience++;
    if (patience > 10) {
      break; // Early stop
    }
  }
}

// Restore best
engine.loadState(best_state);
```

### Learning Rate Scheduling
```typescript
// Reduce LR on plateau
if (epoch > 0 && epoch % 30 == 0) {
  optimizer.learning_rate = optimizer.learning_rate * 0.5;
  nn.setOptimizer(optimizer);
}
```

## See Also

- [Compute Engine (compute.md)](compute.md) - Low-level operations
- [Machine Learning (ml.md)](ml.md) - Preprocessing utilities
- [README.md](README.md) - Library overview
