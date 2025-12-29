# Compute Engine & Operations

The Compute module provides the low-level computational infrastructure for neural networks, including the execution engine, optimizers, layers, operations, and initializers.

## Overview

This module is the foundation of the ALGEBRA library's neural network capabilities. It provides:

- **ComputeEngine**: Execution engine for compiling and running computational graphs
- **Optimizers**: 10 optimization algorithms (Adam, SGD, RMSprop, etc.)
- **Layers**: Building blocks for neural networks (Dense, LSTM, Activation, etc.)
- **Operations**: Element-wise and tensor operations
- **Initializers**: Weight initialization strategies

## Architecture

```
ComputeModel (layers, sequences)
       ↓
ComputeEngine (compile, initialize)
       ↓
Execution Loop (forward, backward, optimize)
       ↓
ComputeState (save/load)
```

## Type Index

### Core Engine
| Type | Purpose |
|------|---------|
| `ComputeEngine` | Main execution engine |
| `ComputeModel` | Model definition with layers |
| `ComputeState` | Saveable model state |
| `ComputeCounter` | Training progress tracking |

### Optimizers
| Type | Algorithm | Best For |
|------|-----------|----------|
| `ComputeOptimizerSgd` | Stochastic Gradient Descent | Simple problems |
| `ComputeOptimizerMomentum` | SGD + Momentum | Accelerated convergence |
| `ComputeOptimizerNesterov` | Nesterov Momentum | Look-ahead momentum |
| `ComputeOptimizerAdam` | Adaptive Moment Estimation | General purpose (default) |
| `ComputeOptimizerAdaDelta` | Adaptive Learning Rate | No manual LR tuning |
| `ComputeOptimizerAdaGrad` | Adaptive Gradient | Sparse data |
| `ComputeOptimizerAdaMax` | Adam variant (infinity norm) | Time-variant processes |
| `ComputeOptimizerNadam` | Adam + Nesterov | Faster convergence |
| `ComputeOptimizerRmsProp` | Root Mean Square Propagation | RNNs |
| `ComputeOptimizerFtrl` | Follow The Regularized Leader | Large sparse features |

### Layers
| Type | Purpose |
|------|---------|
| `ComputeLayerLinear` | Linear transformation (y = Wx + b) |
| `ComputeLayerDense` | Linear + Activation |
| `ComputeLayerLSTM` | Long Short-Term Memory RNN |
| `ComputeLayerActivation` | Standalone activation |
| `ComputeLayerLoss` | Loss computation |
| `ComputeLayerMinMaxScaler` | Min-max normalization |
| `ComputeLayerStandardScaler` | Standard (z-score) normalization |
| `ComputeLayerPCAScaler` | PCA transformation |
| `ComputeLayerFilter` | Feature filtering |
| `ComputeLayerClassification` | Classification output |
| `ComputeLayerConfusion` | Confusion matrix |
| `ComputeLayerCustom` | Custom operations |
| `ComputeLayerSeq` | Sequence of layer calls |

### Activation Functions
| Type | Formula |
|------|---------|
| `ComputeActivationRelu` | max(0, x) |
| `ComputeActivationLeakyRelu` | max(αx, x) |
| `ComputeActivationSigmoid` | 1 / (1 + e^-x) |
| `ComputeActivationTanh` | (e^x - e^-x) / (e^x + e^-x) |
| `ComputeActivationSoftmax` | e^xi / Σe^xj |
| `ComputeActivationSoftplus` | log(1 + e^x) |
| `ComputeActivationSoftSign` | x / (1 + \|x\|) |
| `ComputeActivationSelu` | λ(α(e^x - 1)) if x<0 else λx |
| `ComputeActivationElu` | α(e^x - 1) if x<0 else x |
| `ComputeActivationCelu` | max(0,x) + min(0,α(e^(x/α)-1)) |
| `ComputeActivationHardSigmoid` | clip(slope*x + shift, 0, 1) |
| `ComputeActivationExp` | e^x |

---

## ComputeEngine

The main execution engine for neural network computation.

### Description

`ComputeEngine` compiles computational graphs from `ComputeModel` definitions and executes forward/backward passes with optimization. It manages memory allocation, tensor operations, and training state.

### Methods

#### configure(forwardOnly: bool)
Configures the engine for training or inference.

**Parameters:**
- `forwardOnly`: `true` for inference (no gradients), `false` for training

**Example:**
```typescript
var engine = ComputeEngine {};
engine.configure(false); // Training mode
```

#### compile(model: ComputeModel, maxBatchSize: int): int
Compiles model and allocates memory for given batch size.

**Parameters:**
- `model`: The model to compile
- `maxBatchSize`: Maximum batch size

**Returns:** Memory size in bytes allocated

**Example:**
```typescript
var memory = engine.compile(model, 32);
println("Allocated ${memory} bytes for batch size 32");
```

#### compileUsing(model: ComputeModel, capacity: int): int
Compiles model using a specified memory capacity.

**Parameters:**
- `model`: The model to compile
- `capacity`: Memory capacity in bytes

**Returns:** Actual batch size that fits in memory

**Example:**
```typescript
var batch = engine.compileUsing(model, 100 * 1024 * 1024); // 100MB
println("Batch size: ${batch}");
```

#### memorySize(): int
Returns current memory allocation in bytes.

#### initialize()
Initializes all weights and biases according to their initializers.

**Must call after:** `compile()`

#### setSeed(seed: int)
Sets random seed for reproducible weight initialization.

**Parameters:**
- `seed`: Random seed

**Must call before:** `initialize()`

#### getSeed(): int
Returns current random seed.

#### resize(batchSize: int)
Dynamically changes batch size without recompiling.

**Parameters:**
- `batchSize`: New batch size (must be ≤ maxBatchSize)

#### getVar(layer_name: String, var_name: String): Tensor?
Retrieves a tensor variable from a layer.

**Parameters:**
- `layer_name`: Name of the layer
- `var_name`: Name of the variable

**Returns:** Tensor or null if not found

**Example:**
```typescript
var weights = engine.getVar("layer_1", "weight");
var output = engine.getVar("layer_1", "output");
```

#### getGrad(layer_name: String, var_name: String): Tensor?
Retrieves gradient tensor for a variable.

**Parameters:**
- `layer_name`: Name of the layer
- `var_name`: Name of the variable

**Returns:** Gradient tensor or null

**Example:**
```typescript
var weight_grad = engine.getGrad("layer_1", "weight");
```

#### forward(layer_name: String)
Executes forward pass for a layer or sequence.

**Parameters:**
- `layer_name`: Name of layer or sequence to execute

**Example:**
```typescript
engine.forward("seq_predict"); // Execute prediction sequence
```

#### derive(layer_name: String)
Computes derivatives of loss with respect to outputs.

**Parameters:**
- `layer_name`: Name of sequence (usually training sequence)

#### backward(layer_name: String)
Executes backward pass (backpropagation).

**Parameters:**
- `layer_name`: Name of sequence

#### optimize(layer_name: String)
Updates weights using gradients and optimizer.

**Parameters:**
- `layer_name`: Name of sequence

#### endEpoch(layer_name: String)
Signals end of epoch (updates optimizer state for epoch-based schedules).

**Parameters:**
- `layer_name`: Name of sequence

#### getCounters(layer_name: String): ComputeCounter
Returns training progress counters.

**Parameters:**
- `layer_name`: Name of sequence

**Returns:** ComputeCounter with epoch, optimization steps, etc.

#### saveState(target: ComputeState)
Saves current model weights and optimizer state.

**Parameters:**
- `target`: ComputeState object to store state

**Example:**
```typescript
var state = ComputeState {};
engine.saveState(state);
```

#### saveStateString(): String
Saves state as serialized string.

**Returns:** String representation of state

#### loadState(target: ComputeState)
Loads model weights and optimizer state.

**Parameters:**
- `target`: ComputeState object to load from

#### loadStateString(input: String)
Loads state from serialized string.

**Parameters:**
- `input`: String representation of state

### Complete Training Example

```typescript
// 1. Configure and compile
var engine = ComputeEngine {};
engine.configure(false); // Training mode

var model = create_model(); // Your model definition
var memory = engine.compile(model, 32);

// 2. Initialize
engine.setSeed(42);
engine.initialize();

// 3. Training loop
for (epoch in 0..100) {
  // Fill input/output tensors
  engine.getVar("placeholders", "input")!!.fill(X_train);
  engine.getVar("placeholders", "target")!!.fill(y_train);

  // Forward pass
  engine.forward("seq_train");

  // Compute derivatives
  engine.derive("seq_train");

  // Backward pass
  engine.backward("seq_train");

  // Update weights
  engine.optimize("seq_train");

  // Get loss
  var loss = engine.getVar("loss_layer", "loss");
  println("Epoch ${epoch}: Loss = ${loss}");

  // End epoch
  engine.endEpoch("seq_train");
}

// 4. Save trained model
var state = ComputeState {};
engine.saveState(state);
```

---

## Optimizers

All optimizers extend `ComputeOptimizer` with a `learning_rate` field.

### ComputeOptimizerSgd

Stochastic Gradient Descent - simplest optimizer.

**Fields:**
```typescript
static learning_rate_def: float = 0.01;
learning_rate: float?;
```

**Update rule:**
```
w = w - learning_rate * gradient
```

**Use when:** Simple problems, baseline comparison

**Example:**
```typescript
var optimizer = ComputeOptimizerSgd {
  learning_rate: 0.01
};
```

### ComputeOptimizerMomentum

SGD with momentum for faster convergence.

**Fields:**
```typescript
static learning_rate_def: float = 0.001;
static decay_rate_def: float = 0.9;
learning_rate: float?;
decay_rate: float; // Momentum coefficient (typically 0.9)
```

**Update rule:**
```
v = decay_rate * v + gradient
w = w - learning_rate * v
```

**Use when:** Accelerating convergence, escaping local minima

### ComputeOptimizerAdam

Adaptive Moment Estimation - most popular optimizer.

**Fields:**
```typescript
static learning_rate_def: float = 0.001;
static beta1_def: float = 0.9;      // 1st moment decay
static beta2_def: float = 0.999;    // 2nd moment decay
static smooth_epsilon_def: float = 1e-07;

learning_rate: float?;
beta1: float?;
beta2: float?;
smooth_epsilon: float?;
```

**Update rule:**
```
m = beta1 * m + (1 - beta1) * gradient         // 1st moment
v = beta2 * v + (1 - beta2) * gradient²        // 2nd moment
m_hat = m / (1 - beta1^t)                       // Bias correction
v_hat = v / (1 - beta2^t)
w = w - learning_rate * m_hat / (√v_hat + ε)
```

**Use when:** Default choice for most problems

**Example:**
```typescript
var optimizer = ComputeOptimizerAdam {
  learning_rate: 0.001,
  beta1: 0.9,
  beta2: 0.999,
  smooth_epsilon: 1e-07
};
```

### ComputeOptimizerRmsProp

Root Mean Square Propagation - good for RNNs.

**Fields:**
```typescript
static learning_rate_def: float = 0.001;
static decay_rate_def: float = 0.9;
static smooth_epsilon_def: float = 1e-07;

learning_rate: float?;
decay_rate: float?;
smooth_epsilon: float?;
```

**Update rule:**
```
v = decay_rate * v + (1 - decay_rate) * gradient²
w = w - learning_rate * gradient / (√v + ε)
```

**Use when:** RNNs, non-stationary objectives

### ComputeOptimizerAdaGrad

Adaptive Gradient - adapts learning rate per parameter.

**Fields:**
```typescript
static learning_rate_def: float = 0.001;
static initial_accumulator_def: float = 0.1;
static smooth_epsilon_def: float = 1e-07;

learning_rate: float?;
initial_accumulator: float?;
smooth_epsilon: float?;
```

**Update rule:**
```
v = v + gradient²
w = w - learning_rate * gradient / (√v + ε)
```

**Use when:** Sparse data, different feature scales

**Warning:** Learning rate continually decreases - may stop learning

### Optimizer Comparison

| Optimizer | LR Tuning | Memory | Convergence | Best For |
|-----------|-----------|--------|-------------|----------|
| SGD | Hard | Low | Slow | Baselines |
| Momentum | Medium | Low | Fast | Acceleration |
| Adam | Easy | Medium | Fast | General (default) |
| RMSprop | Easy | Low | Fast | RNNs |
| AdaGrad | Medium | Low | Moderate | Sparse data |
| AdaDelta | None | Medium | Moderate | No LR tuning |

**Recommendation:** Start with Adam, switch to RMSprop for RNNs.

---

## Layers

### ComputeLayerLinear

Linear transformation: `output = input × weight + bias`

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";
static var_weight_name: String = "weight";
static var_bias_name: String = "bias";

name: String;
type: TensorType;
inputs: int;              // Input features
outputs: int;             // Output features
use_bias: bool;           // Whether to add bias
weight_initializer: ComputeInitializer?;
weight_regularizer: ComputeRegularizer?;
bias_initializer: ComputeInitializer?;
bias_regularizer: ComputeRegularizer?;
```

**Tensor shapes:**
- Input: `[batch, inputs]`
- Weight: `[inputs, outputs]`
- Bias: `[outputs]`
- Output: `[batch, outputs]`

**Example:**
```typescript
var layer = ComputeLayerLinear {
  name: "linear_1",
  type: TensorType::f32,
  inputs: 784,
  outputs: 128,
  use_bias: true,
  weight_initializer: ComputeInitializerXavier {},
  weight_regularizer: null,
  bias_initializer: ComputeInitializerConstant { value: 0.0 },
  bias_regularizer: null
};
```

### ComputeLayerDense

Linear transformation with activation: `output = activation(input × weight + bias)`

**Fields:**
Same as `ComputeLayerLinear` plus:
```typescript
activation: ComputeActivation?;  // Activation function
static var_pre_activation_name: String = "pre_activation";
```

**Most common layer type** in neural networks.

**Example:**
```typescript
var layer = ComputeLayerDense {
  name: "dense_1",
  type: TensorType::f32,
  inputs: 128,
  outputs: 64,
  use_bias: true,
  activation: ComputeActivationRelu {},
  weight_initializer: ComputeInitializerRelu {},
  bias_initializer: ComputeInitializerConstant { value: 0.0 }
};
```

### ComputeLayerLSTM

Long Short-Term Memory recurrent layer.

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";
static var_hx_name: String = "hx";  // Hidden state input
static var_cx_name: String = "cx";  // Cell state input
static var_hy_name: String = "hy";  // Hidden state output
static var_cy_name: String = "cy";  // Cell state output

name: String;
type: TensorType;
inputs: int;              // Input dimension per timestep
outputs: int;             // Output dimension per timestep
layers: int;              // Number of stacked LSTM layers
sequences: int;           // Sequence length
use_bias: bool?;          // Default: true
return_sequences: bool?;  // Default: true (return full sequence)
bidirectional: bool?;     // Default: false
auto_init_states: bool?;  // Default: true (zero-init h/c states)
```

**Tensor shapes:**
- Input: `[sequences, batch, inputs]`
- Output (return_sequences=true): `[sequences, batch, outputs]`
- Output (return_sequences=false): `[batch, outputs]`
- hx/hy: `[layers * directions, batch, outputs]` where directions=2 if bidirectional

**LSTM Gates:**
```
i = sigmoid(W_ii * x + b_ii + W_hi * h + b_hi)  // Input gate
f = sigmoid(W_if * x + b_if + W_hf * h + b_hf)  // Forget gate
g = tanh(W_ig * x + b_ig + W_hg * h + b_hg)     // Cell gate
o = sigmoid(W_io * x + b_io + W_ho * h + b_ho)  // Output gate
c' = f ⊙ c + i ⊙ g                               // New cell state
h' = o ⊙ tanh(c')                                // New hidden state
```

**Example:**
```typescript
var layer = ComputeLayerLSTM {
  name: "lstm_1",
  type: TensorType::f32,
  inputs: 50,          // 50 features per timestep
  outputs: 128,        // 128 hidden units
  layers: 2,           // 2-layer stacked LSTM
  sequences: 10,       // 10 timesteps
  use_bias: true,
  return_sequences: false,  // Return only last timestep
  bidirectional: true,      // Process forward and backward
  auto_init_states: true
};
```

### ComputeLayerActivation

Standalone activation layer.

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";

name: String;
activation: ComputeActivation;
```

**Example:**
```typescript
var layer = ComputeLayerActivation {
  name: "relu_1",
  activation: ComputeActivationRelu {}
};
```

### Scaling Layers

#### ComputeLayerMinMaxScaler
Applies min-max normalization: `(x - min) / (max - min)`

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";
static var_min_name: String = "min";
static var_max_name: String = "max";

name: String;
type: TensorType;
inverse_transform: bool;  // false: scale, true: inverse scale
```

#### ComputeLayerStandardScaler
Applies z-score normalization: `(x - μ) / σ`

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";
static var_avg_name: String = "avg";
static var_std_name: String = "std";

name: String;
type: TensorType;
inverse_transform: bool;
```

#### ComputeLayerPCAScaler
Applies PCA transformation.

**Fields:**
```typescript
static var_input_name: String = "input";
static var_output_name: String = "output";
static var_avg_name: String = "avg";
static var_std_name: String = "std";
static var_space_name: String = "space";

name: String;
type: TensorType;
inverse_transform: bool;
```

### Loss Layers

#### ComputeLayerLossRegression
Regression loss (Square or Absolute).

**Fields:**
```typescript
static var_computed_name: String = "computed";
static var_expected_name: String = "expected";
static var_loss_name: String = "loss";

name: String;
loss_type: ComputeRegressionLoss;  // square or abs
reduction: ComputeReduction?;       // auto, none, sum, mean
```

**Loss formulas:**
- Square: `(computed - expected)²`
- Abs: `|computed - expected|`

#### ComputeLayerLossClassification
Classification loss (Cross-entropy).

**Fields:**
```typescript
static var_computed_name: String = "computed";
static var_expected_name: String = "expected";
static var_loss_name: String = "loss";
static var_class_weights_name: String = "class_weights";
static var_predicted_classes_name: String = "predicted_classes";
static var_probabilities_name: String = "probabilities";

name: String;
loss_type: ComputeClassificationLoss;  // categorical or sparse
has_class_weights: bool;
calculate_probabilities: bool;
from_logits: bool;
reduction: ComputeReduction?;
```

**Loss types:**
- `categorical_cross_entropy`: Expects one-hot encoded targets
- `sparse_categorical_cross_entropy`: Expects integer class indices

---

## Initializers

Weight initialization strategies.

### ComputeInitializerConstant
Initializes all weights to a constant value.

```typescript
var init = ComputeInitializerConstant { value: 0.0 };
```

### ComputeInitializerNormal
Initializes from normal distribution N(μ, σ).

```typescript
var init = ComputeInitializerNormal {
  avg: 0.0,
  std: 0.1
};
```

### ComputeInitializerUniform
Initializes from uniform distribution U(min, max).

```typescript
var init = ComputeInitializerUniform {
  min: -0.1,
  max: 0.1
};
```

### ComputeInitializerXavier (Glorot)
Good for sigmoid/tanh activations.

```typescript
var init = ComputeInitializerXavier {};
```

**Formula:** `U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))`

### ComputeInitializerRelu (He)
Optimized for ReLU activations.

```typescript
var init = ComputeInitializerRelu {};
```

**Formula:** `N(0, √(2/fan_in))`

### Initializer Guidelines

| Activation | Recommended Initializer |
|------------|------------------------|
| ReLU, LeakyReLU | ComputeInitializerRelu |
| Sigmoid, Tanh | ComputeInitializerXavier |
| Linear | ComputeInitializerXavier |
| LSTM | ComputeInitializerLSTM |

---

## Operations

Low-level tensor operations used in custom layers.

### Element-wise Operations

```typescript
// Arithmetic
ComputeOperationAdd { input, input2, output }
ComputeOperationSub { input, input2, output }
ComputeOperationMul { input, input2, output }
ComputeOperationDiv { input, input2, output }
ComputeOperationPow { input, input2, output }

// Mathematical
ComputeOperationAbs { input, output }
ComputeOperationExp { input, output }
ComputeOperationLog { input, output }
ComputeOperationSqrt { input, output }
ComputeOperationNeg { input, output }
ComputeOperationSign { input, output }

// Trigonometric
ComputeOperationSin { input, output }
ComputeOperationCos { input, output }
ComputeOperationTan { input, output }
// ... and their inverses/hyperbolic variants
```

### Matrix Operations

```typescript
ComputeOperationMatMul {
  input: String,
  input2: String,
  output: String,
  transposeA: bool,
  transposeB: bool,
  alpha: float,  // Scaling factor
  beta: float    // Output accumulation factor
}
// Computes: output = alpha * matmul(A, B) + beta * output
```

### Reduction Operations

```typescript
ComputeOperationSum {
  input: String,
  output: String,
  axis: int?  // Axis to reduce (null = all)
}

ComputeOperationArgMax {
  input: String,
  output: String,   // Indices
  output2: String   // Values
}

ComputeOperationArgMin {
  input: String,
  output: String,   // Indices
  output2: String   // Values
}
```

### Utility Operations

```typescript
ComputeOperationFill {
  input: String,
  value: any  // Fill tensor with constant
}

ComputeOperationClip {
  input: String,
  output: String,
  min: float?,
  max: float?
}

ComputeOperationScale {
  input: String,
  output: String,
  alpha: float  // Multiply by constant
}
```

---

## Custom Layers

Create custom layers with arbitrary operations.

```typescript
var custom_layer = ComputeLayerCustom {
  name: "custom",
  vars: Array<ComputeVariable> {
    ComputeVarInOut { name: "input", with_grad: true, shape: Array<int> {0, 10}, type: TensorType::f32 },
    ComputeVar { name: "temp" },
    ComputeVarInOut { name: "output", with_grad: true, shape: Array<int> {0, 10}, type: TensorType::f32 }
  },
  ops: Array<ComputeOperation> {
    ComputeOperationScale { input: "input", output: "temp", alpha: 2.0 },
    ComputeOperationRelu { input: "temp", output: "output" }
  }
};
```

---

## Complete Model Building Example

```typescript
// Build a complete regression model manually
fn create_regression_model(inputs: int, outputs: int): ComputeModel {
  var placeholders = ComputeLayerCustom {
    name: "placeholders",
    vars: Array<ComputeVariable> {
      ComputeVarInOut { name: "input", with_grad: false, shape: Array<int> {0, inputs}, type: TensorType::f32 },
      ComputeVarInOut { name: "target", with_grad: false, shape: Array<int> {0, outputs}, type: TensorType::f32 }
    },
    ops: Array<ComputeOperation> {}
  };

  var hidden1 = ComputeLayerDense {
    name: "hidden_1",
    type: TensorType::f32,
    inputs: inputs,
    outputs: 128,
    use_bias: true,
    activation: ComputeActivationRelu {},
    weight_initializer: ComputeInitializerRelu {},
    bias_initializer: ComputeInitializerConstant { value: 0.0 }
  };

  var hidden2 = ComputeLayerDense {
    name: "hidden_2",
    type: TensorType::f32,
    inputs: 128,
    outputs: 64,
    use_bias: true,
    activation: ComputeActivationRelu {},
    weight_initializer: ComputeInitializerRelu {},
    bias_initializer: ComputeInitializerConstant { value: 0.0 }
  };

  var output_layer = ComputeLayerLinear {
    name: "output",
    type: TensorType::f32,
    inputs: 64,
    outputs: outputs,
    use_bias: true,
    weight_initializer: ComputeInitializerXavier {},
    bias_initializer: ComputeInitializerConstant { value: 0.0 }
  };

  var loss = ComputeLayerLossRegression {
    name: "loss",
    loss_type: ComputeRegressionLoss::square,
    reduction: ComputeReduction::mean
  };

  // Create training sequence
  var train_seq = ComputeLayerSeq {
    name: "train",
    calls: Array<ComputeLayerCall> {
      ComputeLayerCall {
        layer_name: "hidden_1",
        bindings: Array<ComputeBinding> {
          ComputeBinding { src_layer_name: "placeholders", src_var_name: "input", target_var_name: "input" }
        }
      },
      ComputeLayerCall {
        layer_name: "hidden_2",
        bindings: Array<ComputeBinding> {
          ComputeBinding { src_layer_name: "hidden_1", src_var_name: "output", target_var_name: "input" }
        }
      },
      ComputeLayerCall {
        layer_name: "output",
        bindings: Array<ComputeBinding> {
          ComputeBinding { src_layer_name: "hidden_2", src_var_name: "output", target_var_name: "input" }
        }
      },
      ComputeLayerCall {
        layer_name: "loss",
        bindings: Array<ComputeBinding> {
          ComputeBinding { src_layer_name: "output", src_var_name: "output", target_var_name: "computed" },
          ComputeBinding { src_layer_name: "placeholders", src_var_name: "target", target_var_name: "expected" }
        }
      }
    },
    optimizer: ComputeOptimizerAdam {
      learning_rate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      smooth_epsilon: 1e-07
    }
  };

  return ComputeModel {
    layers: Array<ComputeLayer> {
      placeholders,
      hidden1,
      hidden2,
      output_layer,
      loss,
      train_seq
    }
  };
}
```

---

## Best Practices

### Memory Management
```typescript
// Option 1: Fixed batch size
var memory = engine.compile(model, 32);

// Option 2: Fixed memory budget
var batch = engine.compileUsing(model, 100 * 1024 * 1024); // 100MB

// Resize during execution (must be ≤ compiled size)
engine.resize(16);
```

### Optimizer Selection
```typescript
// Start with Adam
var optimizer = ComputeOptimizerAdam {};

// Switch to RMSprop for RNNs
var optimizer = ComputeOptimizerRmsProp {};

// Use AdaGrad for sparse data
var optimizer = ComputeOptimizerAdaGrad {};
```

### Weight Initialization
```typescript
// ReLU networks
weight_initializer: ComputeInitializerRelu {}

// Sigmoid/Tanh networks
weight_initializer: ComputeInitializerXavier {}

// Always use zero bias
bias_initializer: ComputeInitializerConstant { value: 0.0 }
```

## See Also

- [Neural Networks (nn.md)](nn.md) - High-level network APIs
- [Machine Learning (ml.md)](ml.md) - Preprocessing and PCA
- [README.md](README.md) - Library overview
