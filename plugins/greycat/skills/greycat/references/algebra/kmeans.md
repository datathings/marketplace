# K-means Clustering

Efficient K-means clustering implementation with meta-learning for automatic cluster count selection and comprehensive cluster analysis.

## Overview

The K-means module provides:
- **Standard K-means**: Lloyd's algorithm for clustering
- **Meta-learning**: Multiple runs to find best clustering
- **Meta-meta-learning**: Automatic cluster count selection
- **Cluster Analysis**: Inter-cluster distances, silhouette-like metrics
- **Inference Engine**: Fast cluster assignment for new data

## Quick Start

```typescript
// Automatic clustering with optimal cluster count
var result = Kmeans::meta_meta_learning(
  tensors: data_list,          // nodeList<Tensor>, each [batch, features]
  max_clusters: 10,            // Try 2-10 clusters
  stop_ratio: 0.1,             // Stop if improvement < 10%
  nb_meta_rounds: 100,
  nb_rounds: 20,
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

var centroids = result.bestResult!!.centroids;
var assignments = result.bestResult!!.assignment;
println("Optimal clusters: ${centroids.shape()[0]}");
```

## Type Index

| Type | Purpose |
|------|---------|
| `Kmeans` | Static methods for K-means clustering |
| `KmeansResult` | Single clustering result |
| `KmeansMetaResult` | Meta-learning result (multiple runs) |

---

## Kmeans

Static methods for K-means clustering operations.

### Configuration

#### configure(nb_clusters: int, nb_features: int, tensor_type: TensorType, calculate_inter_cluster_stats: bool): ComputeModel (static)
Creates K-means computational model.

**Parameters:**
- `nb_clusters`: Number of clusters (k)
- `nb_features`: Number of features per sample
- `tensor_type`: f32 or f64
- `calculate_inter_cluster_stats`: Compute inter-cluster distances

**Returns:** ComputeModel for K-means

**Example:**
```typescript
var model = Kmeans::configure(
  nb_clusters: 5,
  nb_features: 10,
  tensor_type: TensorType::f64,
  calculateInterClusterStats: true
);
```

### Initialization

#### initialize(engine: ComputeEngine, seed: int) (static)
Initializes K-means engine.

**Parameters:**
- `engine`: ComputeEngine
- `seed`: Random seed for centroid initialization

**Example:**
```typescript
var engine = ComputeEngine {};
engine.compile(model, batch_size);
Kmeans::initialize(engine, 42);
```

### Training Operations

#### init_round(engine: ComputeEngine) (static)
Initializes a new clustering round (resets accumulators).

**Call before:** Processing each round of data

#### learn(engine: ComputeEngine, mini_batch: Tensor) (static)
Processes a mini-batch of data.

**Parameters:**
- `engine`: ComputeEngine
- `mini_batch`: Data tensor [batch, features]

**Example:**
```typescript
Kmeans::init_round(engine);
Kmeans::learn(engine, data_batch);
Kmeans::end_round(engine);
```

#### end_round(engine: ComputeEngine) (static)
Finalizes a round (updates centroids from accumulated data).

#### calculate_stats(engine: ComputeEngine) (static)
Computes final clustering statistics.

**Call after:** All rounds complete

### Inference

#### cluster(engine: ComputeEngine, mini_batch: Tensor): Tensor (static)
Assigns clusters to new data.

**Parameters:**
- `mini_batch`: Data tensor [batch, features]

**Returns:** Cluster assignments tensor [batch]

**Example:**
```typescript
var assignments = Kmeans::cluster(engine, test_data);
```

### Retrieving Results

#### getDistances(engine: ComputeEngine): Tensor (static)
Gets distance matrix [batch, clusters] of last mini-batch to all centroids.

#### getAssignment(engine: ComputeEngine): Tensor (static)
Gets cluster assignments [batch] of last mini-batch.

#### getBestDistances(engine: ComputeEngine): Tensor (static)
Gets distance [batch] to nearest centroid for last mini-batch.

#### getSumOfDistances(engine: ComputeEngine): Tensor (static)
Gets total within-cluster distance (loss metric).

**Example:**
```typescript
var loss = Kmeans::getSumOfDistances(engine).get(Array<int> {0}) as float;
```

#### getClustersCentroids(engine: ComputeEngine): Tensor (static)
Gets centroid positions [clusters, features].

#### getClustersCounts(engine: ComputeEngine): Tensor (static)
Gets number of samples per cluster [clusters].

#### getClustersSumOfDistances(engine: ComputeEngine): Tensor (static)
Gets within-cluster distance sum per cluster [clusters].

#### getClustersAvgOfDistances(engine: ComputeEngine): Tensor (static)
Gets average within-cluster distance per cluster [clusters].

**Requires:** `calculateInterClusterStats=true` and `calculate_stats()` called

#### getClustersDistancesToEachOther(engine: ComputeEngine): Tensor (static)
Gets inter-cluster distance matrix [clusters, clusters].

**Requires:** `calculateInterClusterStats=true` and `calculate_stats()` called

### Utility Methods

#### replaceEmptyClusters(engine: ComputeEngine, tensors: nodeList\<Tensor\>, features_gaussian: GaussianND, replace_empty_clusters_mode: KmeansMode): bool (static)
Replaces empty clusters with new centroids to avoid degeneracy.

**Parameters:**
- `engine`: The ComputeEngine instance
- `tensors`: List of input data tensors to sample from
- `features_gaussian`: Statistics of the feature space
- `replace_empty_clusters_mode`: Mode for generating replacement centroids

**Returns:** true if empty clusters were found and replaced

#### sortClusters(engine: ComputeEngine) (static)
Sorts clusters by centroid sum (for consistent ordering).

---

## High-Level Learning Methods

### learning(tensors: nodeList\<Tensor\>, features_gaussian: GaussianND, nb_clusters: int, nb_rounds: int, seed: int, initialization_mode: KmeansMode, replace_empty_clusters_mode: KmeansMode, calculate_inter_cluster_stats: bool): KmeansResult (static)
Runs standard K-means for fixed cluster count and seed.

**Parameters:**
- `tensors`: Data as a nodeList of Tensors, each [batch, features]
- `features_gaussian`: Pre-computed GaussianND statistics of the feature space
- `nb_clusters`: Number of clusters
- `nb_rounds`: Number of iterations
- `seed`: Random seed
- `initialization_mode`: Mode for centroid initialization (fromInput, randomUniform, randomNormal)
- `replace_empty_clusters_mode`: Mode for replacing empty clusters
- `calculate_inter_cluster_stats`: Compute inter-cluster stats

**Returns:** KmeansResult with centroids, assignments, loss, etc.

**Example:**
```typescript
// Compute feature statistics first
var features_gaussian = GaussianND {};
for (i, t in tensors) {
  features_gaussian.learn(t);
}

var result = Kmeans::learning(
  tensors,
  features_gaussian,
  nb_clusters: 5,
  nb_rounds: 30,
  seed: 42,
  initialization_mode: KmeansMode::fromInput,
  replace_empty_clusters_mode: KmeansMode::randomUniform,
  calculate_inter_cluster_stats: true
);

println("Loss: ${result.loss}");
println("Centroids: ${result.centroids}");
```

### meta_learning(tensors: nodeList\<Tensor\>, nb_clusters: int, nb_meta_rounds: int, nb_rounds: int, seed: int, parallel: bool, initialization_mode: KmeansMode?, replace_empty_clusters_mode: KmeansMode?, calculate_inter_cluster_stats: bool): KmeansMetaResult (static)
Runs multiple K-means with different initializations, returns best.

**Parameters:**
- `tensors`: Data as a nodeList of Tensors
- `nb_clusters`: Fixed cluster count
- `nb_meta_rounds`: Number of random restarts (default: 100)
- `nb_rounds`: Iterations per run (default: 20)
- `seed`: Base random seed
- `parallel`: Whether to run meta-rounds in parallel using Job
- `initialization_mode`: Optional centroid initialization mode (random if null)
- `replace_empty_clusters_mode`: Optional empty cluster replacement mode (random if null)
- `calculate_inter_cluster_stats`: Compute stats

**Returns:** KmeansMetaResult with best result and all meta-round losses

**Example:**
```typescript
var result = Kmeans::meta_learning(
  tensors,
  nb_clusters: 5,
  nb_meta_rounds: 100,
  nb_rounds: 20,
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

println("Best loss: ${result.bestResult!!.loss}");
println("Meta-round losses: ${result.metaRoundsLoss}");
```

### meta_meta_learning(tensors: nodeList\<Tensor\>, max_clusters: int, stop_ratio: float, nb_meta_rounds: int, nb_rounds: int, seed: int, parallel: bool, initialization_mode: KmeansMode?, replace_empty_clusters_mode: KmeansMode?, calculate_inter_cluster_stats: bool): KmeansMetaResult (static)
Automatically finds optimal cluster count.

**Parameters:**
- `tensors`: Data as a nodeList of Tensors
- `max_clusters`: Maximum clusters to try
- `stop_ratio`: Stop if improvement < this ratio (0.0-1.0)
- `nb_meta_rounds`: Number of random restarts per cluster count
- `nb_rounds`: Iterations per run
- `seed`: Base seed
- `parallel`: Whether to run meta-rounds in parallel
- `initialization_mode`: Optional centroid initialization mode
- `replace_empty_clusters_mode`: Optional empty cluster replacement mode
- `calculate_inter_cluster_stats`: Compute inter-cluster stats

**Returns:** KmeansMetaResult for optimal cluster count

**Algorithm:**
1. Try k=2, 3, 4, ...
2. For each k, run meta-learning
3. Calculate improvement ratio: `(loss[k-1] - loss[k]) / loss[k-1]`
4. Stop when improvement < stopRatio

**Example:**
```typescript
var result = Kmeans::meta_meta_learning(
  tensors,
  max_clusters: 15,
  stop_ratio: 0.1,     // Stop if improvement < 10%
  nb_meta_rounds: 100,
  nb_rounds: 20,
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

var k = result.bestResult!!.centroids.shape()[0];
println("Optimal clusters: ${k}");
```

---

## Results

### KmeansResult

Single clustering result.

```typescript
type KmeansResult {
  roundsLoss: Array<float>;            // Loss per round
  initMode: KmeansMode;                // Initialization mode used
  replaceClustersMode: KmeansMode;     // Empty cluster replacement mode
  loss: float;                         // Total within-cluster distance
  centroids: Tensor?;                  // [clusters, features]
  clusters_count: Tensor?;             // [clusters] - samples per cluster
  clusters_sum_distance: Tensor?;      // [clusters] - total distance
  clusters_avg_distance: Tensor?;      // [clusters] - avg distance
  assignment: Tensor?;                 // [samples] - cluster IDs
  distances: Tensor?;                  // [samples] - distance to centroid
  clusterInterDistances: Tensor?;      // [clusters, clusters]
}
```

**Example:**
```typescript
println("Cluster sizes: ${result.clusters_count}");
println("Cluster compactness: ${result.clusters_avg_distance}");

// Find most compact cluster
var min_avg_dist = float::max;
var best_cluster = -1;
for (i in 0..result.clusters_count!!.size()) {
  var avg_dist = result.clusters_avg_distance!!.get(Array<int> {i}) as float;
  if (avg_dist < min_avg_dist) {
    min_avg_dist = avg_dist;
    best_cluster = i;
  }
}
println("Most compact cluster: ${best_cluster}");
```

### KmeansMetaResult

Meta-learning result.

```typescript
type KmeansMetaResult {
  metaRoundsLoss: Array<float>;    // Loss for each meta-round
  bestResult: KmeansResult?;       // Best clustering
}
```

**Example:**
```typescript
println("Tried ${result.metaRoundsLoss.size()} initializations");
println("Best loss: ${result.bestResult!!.loss}");
println("Worst loss: ${max(result.metaRoundsLoss)}");
```

---

## Inference Engine

### getInferenceEngine(bestResult: KmeansResult, maxBatchSize: int, calculate_inter_cluster_stats: bool): ComputeEngine (static)
Creates an engine for fast cluster assignment.

**Parameters:**
- `bestResult`: Trained clustering result
- `maxBatchSize`: Max batch size for inference
- `calculateInterClusterStats`: Compute stats

**Returns:** ComputeEngine with loaded centroids

**Example:**
```typescript
// Train
var train_result = Kmeans::meta_learning(train_tensors, 5, 100, 20, 42, true, null, null, false);

// Create inference engine
var infer_engine = Kmeans::getInferenceEngine(
  train_result.bestResult!!,
  maxBatchSize: 1000,
  calculateInterClusterStats: false
);

// Assign new data
var new_assignments = Kmeans::cluster(infer_engine, new_data);
```

---

## Complete Examples

### Basic K-means

```typescript
// Prepare data as nodeList of Tensors
var tensors = nodeList<Tensor> {};
var data = Tensor {};
data.init(TensorType::f64, Array<int> {1000, 10}); // 1000 samples, 10 features
// ... fill data ...
tensors.add(data);

// Compute feature statistics
var features_gaussian = GaussianND {};
for (i, t in tensors) {
  features_gaussian.learn(t);
}

// Run K-means with k=5
var result = Kmeans::learning(
  tensors,
  features_gaussian,
  nb_clusters: 5,
  nb_rounds: 30,
  seed: 42,
  initialization_mode: KmeansMode::fromInput,
  replace_empty_clusters_mode: KmeansMode::randomUniform,
  calculate_inter_cluster_stats: true
);

println("Final loss: ${result.loss}");
println("Convergence: ${result.roundsLoss}");

// Analyze clusters
for (i in 0..5) {
  var count = result.clusters_count!!.get(Array<int> {i});
  var avg_dist = result.clusters_avg_distance!!.get(Array<int> {i});
  println("Cluster ${i}: ${count} samples, avg distance ${avg_dist}");
}
```

### Automatic Cluster Selection

```typescript
// Let algorithm find optimal k
var result = Kmeans::meta_meta_learning(
  tensors,
  max_clusters: 20,
  stop_ratio: 0.05,    // Stop if improvement < 5%
  nb_meta_rounds: 50,  // 50 random restarts per k
  nb_rounds: 20,       // 20 iterations per restart
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

var k = result.bestResult!!.centroids.shape()[0];
println("Optimal number of clusters: ${k}");

// Visualize elbow
println("\nLoss by cluster count:");
for (i, loss in result.metaRoundsLoss) {
  println("k=${i+2}: ${loss}");
}
```

### Customer Segmentation

```typescript
// Segment customers by behavior
var customer_features = Tensor {}; // [customers, features]
// Features: [purchase_frequency, avg_order_value, recency, ...]

var result = Kmeans::meta_meta_learning(
  customer_tensors,
  max_clusters: 10,
  stop_ratio: 0.1,
  nb_meta_rounds: 100,
  nb_rounds: 30,
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

var k = result.bestResult!!.centroids.shape()[0];
println("Found ${k} customer segments");

// Analyze segments
var centroids = result.bestResult!!.centroids;
for (i in 0..k) {
  println("\nSegment ${i}:");
  for (f in 0..centroids.shape()[1]) {
    var feature_val = centroids.get(Array<int> {i, f});
    println("  Feature ${f}: ${feature_val}");
  }

  var size = result.bestResult!!.clusters_count!!.get(Array<int> {i});
  println("  Size: ${size} customers");
}

// Assign new customers
var new_customers = Tensor {}; // New customer data
var infer_engine = Kmeans::getInferenceEngine(result.bestResult!!, 100, false);
var segments = Kmeans::cluster(infer_engine, new_customers);
```

### Image Color Quantization

```typescript
// Reduce image to k dominant colors
var pixel_tensors = nodeList<Tensor> {};
var image_pixels = Tensor {}; // [width*height, 3] - RGB values
pixel_tensors.add(image_pixels);

var pixel_stats = GaussianND {};
pixel_stats.learn(image_pixels);

var result = Kmeans::learning(
  pixel_tensors,
  pixel_stats,
  nb_clusters: 16,     // 16 colors
  nb_rounds: 50,
  seed: 42,
  initialization_mode: KmeansMode::randomUniform,
  replace_empty_clusters_mode: KmeansMode::randomUniform,
  calculate_inter_cluster_stats: false
);

// Centroids are the dominant colors
var palette = result.centroids; // [16, 3]

// Reconstruct image using palette
var assignments = result.assignment; // [width*height]
// Each pixel → nearest palette color
```

### Anomaly Detection

```typescript
// Use K-means for outlier detection
var normal_data = Tensor {}; // Normal samples

var result = Kmeans::meta_learning(
  normal_tensors,
  nb_clusters: 5,
  nb_meta_rounds: 100,
  nb_rounds: 20,
  seed: 42,
  parallel: true,
  initialization_mode: null,
  replace_empty_clusters_mode: null,
  calculate_inter_cluster_stats: true
);

// Create inference engine
var engine = Kmeans::getInferenceEngine(result.bestResult!!, 1000, false);

// Check new samples
var test_samples = Tensor {};
Kmeans::learn(engine, test_samples);
var distances = Kmeans::getBestDistances(engine);

// Samples far from all centroids = anomalies
var threshold = 3.0 * mean(result.bestResult!!.clusters_avg_distance);
for (i in 0..distances.size()) {
  var dist = distances.get(Array<int> {i}) as float;
  if (dist > threshold) {
    println("Anomaly detected at sample ${i}, distance: ${dist}");
  }
}
```

---

## Performance Tips

### Choosing Parameters

```typescript
// metaRounds: More = better but slower
// - Small datasets (<1000): 50-100
// - Large datasets (>10000): 10-30

// rounds: Iterations per run
// - Simple data: 10-20
// - Complex data: 30-50

// stopRatio for meta_meta_learning:
// - Aggressive (fewer clusters): 0.15-0.20
// - Balanced: 0.05-0.10
// - Conservative (more clusters): 0.01-0.05
```

### Memory Management

```typescript
// For large datasets, use mini-batch approach
var batch_size = 1000;
for (batch_start in 0..data.shape()[0] step batch_size) {
  var batch_end = min(batch_start + batch_size, data.shape()[0]);
  var batch = data.slice(batch_start, batch_end);
  Kmeans::learn(engine, batch);
}
```

### Initialization Strategy

```typescript
// Use data range for better initialization
var stats = GaussianND {};
stats.learn(data);
var min_vals = stats.min;
var max_vals = stats.max;

// Or use K-means++ initialization (manual implementation)
```

---

## Best Practices

### Data Preparation
```typescript
// Normalize features before K-means
var gaussian = GaussianND {};
gaussian.learn(data);
var normalized = gaussian.standard_scaling(data);

// Run K-means on normalized data
var result = Kmeans::meta_meta_learning(normalized, ...);
```

### Validation
```typescript
// Use silhouette score or elbow method
// Check cluster balance
var counts = result.bestResult!!.clusters_count;
for (i in 0..counts.size()) {
  var ratio = counts.get(Array<int> {i}) / total_samples;
  if (ratio < 0.01) {
    warn("Cluster ${i} very small: ${ratio * 100}%");
  }
}
```

### Reproducibility
```typescript
// Always set seed for reproducible results
var result = Kmeans::meta_learning(tensors, 5, 100, 20, 42, true, null, null, true);
// Same result every time with seed=42
```

## See Also

- [Machine Learning (ml.md)](ml.md) - GaussianND for normalization
- [README.md](README.md) - Library overview
