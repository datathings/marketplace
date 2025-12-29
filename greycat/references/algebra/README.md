# ALGEBRA Library

The ALGEBRA library is GreyCat's comprehensive suite for machine learning, neural networks, numerical computing, and pattern recognition. It provides state-of-the-art implementations of ML algorithms, deep learning architectures, and computational operations with GPU acceleration support.

## Overview

The ALGEBRA library consists of 8 specialized modules:

| Module | Description | Key Types |
|--------|-------------|-----------|
| **[ml.md](ml.md)** | Machine Learning Utilities | GaussianND, PCA, Polynomial, Solver |
| **[compute.md](compute.md)** | Computational Engine & Operations | ComputeEngine, Optimizers, Layers |
| **[nn.md](nn.md)** | Neural Networks | RegressionNetwork, ClassificationNetwork, AutoEncoderNetwork |
| **[patterns.md](patterns.md)** | Pattern Recognition | Euclidean, DTW, FFT, SAX detectors |
| **[transforms.md](transforms.md)** | FFT & Signal Processing | FFT, FFTModel |
| **[kmeans.md](kmeans.md)** | K-means Clustering | Kmeans, KmeanResult |
| **[climate.md](climate.md)** | Climate Modeling | UTCI calculation |
| **nn_layers_names.gcl** | Layer Naming Conventions | NNLayersNames enum (200 layers) |

## Quick Start

### Machine Learning

```typescript
// Statistical analysis and normalization
var gaussian = GaussianND {};
gaussian.learn(input_data);
var normalized = gaussian.standard_scaling(raw_data);

// Principal Component Analysis
var pca = PCA {};
pca.learn(gaussian.correlation(), gaussian.avg(), gaussian.std(), 0.95);
pca.set_dimension(10);
var reduced = pca.transform(input_data);
```

### Neural Networks

```typescript
// Create a classification network
var nn = ClassificationNetwork::new(
  inputs: 784,          // Input features (e.g., 28x28 image)
  classes: 10,          // Output classes
  tensor_type: TensorType::f32,
  inputs_gradients: false,
  fixed_batch_size: 32,
  seed: 42,
  calculate_probabilities: true,
  from_logits: true,
  has_class_weights: false
);

// Add layers
nn.addDenseLayer(128, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(10, true, null, null);

// Set optimizer and loss
nn.setOptimizer(ComputeOptimizerAdam {
  learning_rate: 0.001,
  beta1: 0.9,
  beta2: 0.999,
  smooth_epsilon: 1e-07
});

// Build and train
var model = nn.build(true);
var engine = ComputeEngine {};
var batch_size = nn.initWithBatch(model, engine, null, 32);

// Training loop
for (epoch in 0..100) {
  nn.getInput(engine).fill(train_data);
  nn.getTarget(engine).fill(train_labels);
  var loss = nn.train(engine);

  if (epoch % 10 == 0) {
    println("Epoch ${epoch}: Loss = ${loss}");
  }
}

// Inference
var predictions = nn.predict(engine, test_data);
```

### Pattern Detection

```typescript
// Create pattern detector
var detector = EuclideanPatternDetector {};
var engine = detector.getEngine(timeseries);
engine.setState(PatternDetectionEngineState::new());

// Add patterns
engine.addPattern(from_time, to_time);

// Detect patterns
engine.computeScores(null);
engine.detect(PatternDetectionSensitivity {
  threshold: 0.8,
  overlap: 0.3
}, null);
```

### K-means Clustering

```typescript
// Configure clustering
var model = Kmeans::configure(
  nb_clusters: 5,
  nb_features: 10,
  tensor_type: TensorType::f64,
  features_min: 0.0,
  features_max: 1.0,
  calculateInterClusterStats: true
);

// Run meta-learning to find best clustering
var result = Kmeans::meta_meta_learning(
  tensor: data,
  maxClusters: 10,
  stopRatio: 0.1,
  seed: 42,
  metaRounds: 100,
  rounds: 20,
  calculateInterClusterStats: true
);

// Get centroids and assignments
var centroids = result.bestResult!!.centroids;
var assignments = result.bestResult!!.assignment;
```

## Architecture

### Tensor Operations

The library is built on Tensor operations with support for:
- **Data Types**: `f32`, `f64`, `i32`, `i64`, `c64`, `c128`
- **Operations**: Matrix multiplication, element-wise ops, reductions
- **Memory Management**: Efficient allocation and GPU acceleration

### Neural Network Pipeline

```
Input Data → Preprocessing → Neural Layers → Postprocessing → Output
              (Scaling)      (Dense, LSTM)    (Inverse scale)
                             (Activation)
                             (Loss)
```

### ComputeEngine Execution

```
Configure → Compile → Initialize → Training Loop → Save State
                                   ├─ Forward
                                   ├─ Derive
                                   ├─ Backward
                                   └─ Optimize
```

## Key Concepts

### 1. Machine Learning Fundamentals

- **Normalization**: Min-max scaling, standard scaling
- **Dimensionality Reduction**: PCA with variance thresholds
- **Statistical Profiling**: Gaussian distribution analysis
- **Polynomial Regression**: Time series compression

### 2. Neural Network Types

- **Regression**: Continuous value prediction (MSE, MAE loss)
- **Classification**: Categorical prediction (Cross-entropy loss)
- **AutoEncoder**: Unsupervised learning for encoding/decoding

### 3. Optimizers

| Optimizer | Best For | Learning Rate |
|-----------|----------|---------------|
| **SGD** | Simple tasks | 0.01 |
| **Momentum** | Accelerated convergence | 0.001 |
| **Adam** | General purpose (default) | 0.001 |
| **RMSprop** | RNNs | 0.001 |
| **AdaGrad** | Sparse data | 0.001 |

### 4. Pattern Detection Algorithms

- **Euclidean**: Fast distance-based matching
- **DTW**: Dynamic time warping for flexible alignment
- **FFT**: Frequency domain analysis
- **SAX**: Symbolic representation for large-scale search

## Common Workflows

### Train a Regression Model

```typescript
// 1. Prepare data
var gaussian = GaussianND {};
gaussian.learn(training_data);

// 2. Create network
var nn = RegressionNetwork::new(10, 1, TensorType::f32, false, 32, 42);
nn.setPreProcess(PreProcessType::standard_scaling, gaussian);
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(32, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(1, true, null, null);

// 3. Train
var engine = ComputeEngine {};
nn.initWithBatch(nn.build(true), engine, null, 32);

for (epoch in 0..epochs) {
  nn.getInput(engine).fill(X_train);
  nn.getTarget(engine).fill(y_train);
  var loss = nn.train(engine);
}

// 4. Predict
var predictions = nn.predict(engine, X_test);
```

### Build an LSTM Classifier

```typescript
var nn = ClassificationNetwork::new(50, 5, TensorType::f32, false, 32, 42, true, true, false);

// LSTM layer
nn.addLSTMLayer(
  output: 128,
  layers: 2,
  sequences: 10,
  use_bias: true,
  return_sequences: false,
  bidirectional: true,
  config: null
);

// Dense classification head
nn.addDenseLayer(64, true, ComputeActivationRelu {}, null);
nn.addDenseLayer(5, true, null, null);

// Train with sequences
nn.getInput(engine).fill(sequences); // [10, 32, 50]
nn.getTarget(engine).fill(labels);   // [32, 1]
var loss = nn.train(engine);
```

### Detect Patterns in Time Series

```typescript
// Create detector with SAX for efficiency
var detector = SaxPatternDetector {
  alphabet_size: 10,
  fingerprint_length: 20
};
var engine = detector.getEngine(timeseries);
engine.setState(PatternDetectionEngineState::new());

// Define patterns to search for
engine.addPattern(time::new(2024, 1, 1, 0, 0, 0), time::new(2024, 1, 2, 0, 0, 0));
engine.addPattern(time::new(2024, 2, 1, 0, 0, 0), time::new(2024, 2, 2, 0, 0, 0));

// Find similar patterns
engine.initScoring();
engine.computeScores(null);

// Extract high-confidence detections
var detections = engine.detect(PatternDetectionSensitivity {
  threshold: 0.85,
  overlap: 0.2
}, null);
```

## Performance Tips

### Memory Management

```typescript
// Compile with memory limit
var max_batch = nn.initWithMemory(model, engine, null, 1024 * 1024 * 100); // 100MB

// Resize batch size dynamically
engine.resize(new_batch_size);

// Clean up tensors
tensor.reset();
```

### GPU Acceleration

```typescript
// Use appropriate tensor types for GPU
var tensor = Tensor {};
tensor.init(TensorType::f32, Array<int> {1000, 1000}); // f32 is GPU-optimized

// Batch operations for efficiency
engine.resize(larger_batch_size); // Process more samples per forward pass
```

### Training Optimization

```typescript
// Use learning rate scheduling
if (epoch > 50) {
  optimizer.learning_rate = optimizer.learning_rate * 0.5;
}

// Early stopping
var best_loss = float::max;
var patience = 0;
for (epoch in 0..max_epochs) {
  var val_loss = nn.validation(engine);
  if (val_loss < best_loss) {
    best_loss = val_loss;
    patience = 0;
    engine.saveState(best_state);
  } else {
    patience++;
    if (patience > 10) {
      break;
    }
  }
}
```

## Mathematical Operations

### Tensor Shapes

```typescript
// 2D: [batch, features]
var input = Tensor {}; // [32, 784]

// 3D (sequences): [sequence, batch, features]
var lstm_input = Tensor {}; // [10, 32, 50]

// 4D (images): [batch, channels, height, width]
var cnn_input = Tensor {}; // [32, 3, 224, 224]
```

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | `max(0, x)` | Hidden layers (default) |
| Sigmoid | `1 / (1 + e^-x)` | Binary classification |
| Tanh | `(e^x - e^-x) / (e^x + e^-x)` | LSTM gates |
| Softmax | `e^xi / Σe^xj` | Multi-class classification |
| LeakyReLU | `max(0.3x, x)` | Avoiding dead neurons |

### Loss Functions

- **Regression**: Square loss, Absolute loss
- **Classification**: Categorical/Sparse categorical cross-entropy
- **Reduction**: Auto, none, sum, mean

## State Management

```typescript
// Save model state
var state = ComputeState {};
engine.saveState(state);

// Save to string for persistence
var state_string = engine.saveStateString();

// Load from state
engine.loadState(state);
engine.loadStateString(state_string);
```

## Module Links

- **[Machine Learning (ml.md)](ml.md)**: Statistical analysis, PCA, polynomial regression
- **[Compute Engine (compute.md)](compute.md)**: Low-level operations, optimizers, layer definitions
- **[Neural Networks (nn.md)](nn.md)**: High-level network APIs for ML tasks
- **[Pattern Recognition (patterns.md)](patterns.md)**: Time series pattern detection algorithms
- **[Transforms (transforms.md)](transforms.md)**: FFT and signal processing
- **[K-means (kmeans.md)](kmeans.md)**: Clustering algorithms
- **[Climate (climate.md)](climate.md)**: Climate modeling utilities

## Examples Repository

For complete working examples, see the test files in the GreyCat repository.

## Best Practices

1. **Always normalize your data** before training neural networks
2. **Start with Adam optimizer** - it works well for most tasks
3. **Use standard scaling for PCA** input
4. **Monitor validation loss** to prevent overfitting
5. **Save model states** regularly during training
6. **Choose appropriate batch sizes** based on available memory
7. **Use f32 for most tasks** - f64 only when high precision needed
8. **Initialize random seeds** for reproducible results

## Contributing

The ALGEBRA library is actively maintained. For bug reports or feature requests, please contact the GreyCat team.
