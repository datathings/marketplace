# Machine Learning Utilities

The ML module provides fundamental machine learning algorithms including statistical analysis, dimensionality reduction, polynomial regression, and time series decomposition.

## Overview

This module focuses on classical machine learning techniques that serve as building blocks for more complex models:

- **Statistical Analysis**: Multi-dimensional Gaussian profiling
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Regression**: Polynomial fitting and time series compression
- **Time Series**: Decomposition and aggregation utilities

## Type Index

| Type | Purpose |
|------|---------|
| `GaussianND` | N-dimensional Gaussian distribution analysis and normalization |
| `PCA` | Principal Component Analysis for dimensionality reduction |
| `Solver` | Linear equation solver |
| `Polynomial` | Polynomial regression and time series compression |
| `TimeSeriesDecomposition` | Time series aggregation and decomposition |

---

## GaussianND

Multi-dimensional Gaussian distribution for statistical profiling and data normalization.

### Description

`GaussianND` learns statistical properties (mean, standard deviation, covariance) from multi-dimensional data and provides normalization methods. Each row represents one observation, each column represents one feature/dimension.

### Fields

```typescript
private total: int?;         // Number of observations
private min: Tensor?;        // Minimum values per dimension
private max: Tensor?;        // Maximum values per dimension
private sum: Tensor?;        // Sum of values per dimension
private sum_square: Tensor?; // Sum of squared values per dimension
```

### Methods

#### learn(input: Tensor)
Learns statistical properties from input data.

**Parameters:**
- `input`: Tensor of shape `[BatchSize, N]` where each row is an observation

**Example:**
```typescript
var gaussian = GaussianND {};
var data = Tensor {};
data.init(TensorType::f64, Array<int> {100, 5}); // 100 samples, 5 features
// ... fill data ...
gaussian.learn(data);
```

#### avg(): Tensor?
Returns the average value for each dimension.

**Returns:** 1D Tensor of shape `[N]` with average values

**Example:**
```typescript
var averages = gaussian.avg();
println("Feature averages: ${averages}");
```

#### std(): Tensor?
Returns the standard deviation for each dimension.

**Returns:** 1D Tensor of shape `[N]` with standard deviations

#### covariance(): Tensor?
Calculates the covariance matrix showing how dimensions vary together.

**Returns:** 2D Tensor of shape `[N, N]` where element `[i,j]` is covariance of dimension i with dimension j

**Mathematical Definition:**
```
Cov(X,Y) = E[(X - E[X])(Y - E[Y])]
```

#### correlation(): Tensor?
Calculates the correlation matrix (normalized covariance).

**Returns:** 2D Tensor of shape `[N, N]` with correlation coefficients in range `[-1, 1]`

**Mathematical Definition:**
```
Corr(X,Y) = Cov(X,Y) / (σ_X * σ_Y)
```

#### dimensions(): int
Returns the number of dimensions in the N-dimensional space.

#### clear()
Resets all internal state for memory reuse.

#### min_max_scaling(input: Tensor): Tensor
Normalizes data to range `[0, 1]` using min-max scaling.

**Parameters:**
- `input`: Tensor of shape `[batch, N]`

**Returns:** Normalized tensor of same shape

**Formula:**
```
x_normalized = (x - min) / (max - min)
```

**Example:**
```typescript
var normalized = gaussian.min_max_scaling(test_data);
```

#### inverse_min_max_scaling(input: Tensor): Tensor
Reverses min-max scaling to get original values.

**Parameters:**
- `input`: Normalized tensor of shape `[batch, N]`

**Returns:** Denormalized tensor

**Formula:**
```
x_original = x_normalized * (max - min) + min
```

#### standard_scaling(input: Tensor): Tensor
Standardizes data to zero mean and unit variance.

**Parameters:**
- `input`: Tensor of shape `[batch, N]`

**Returns:** Standardized tensor

**Formula:**
```
x_standardized = (x - μ) / σ
```

**Example:**
```typescript
var standardized = gaussian.standard_scaling(train_data);
```

#### inverse_standard_scaling(input: Tensor): Tensor
Reverses standard scaling.

**Parameters:**
- `input`: Standardized tensor

**Returns:** Original scale tensor

**Formula:**
```
x_original = x_standardized * σ + μ
```

#### crop(from: int, to: int): GaussianND
Creates a new GaussianND with a subset of features.

**Parameters:**
- `from`: Starting feature index (inclusive)
- `to`: Ending feature index (exclusive)

**Returns:** New GaussianND with features `[from, to)`

**Example:**
```typescript
// Extract features 2-5 from 10-feature dataset
var subset = gaussian.crop(2, 5);
```

### Complete Example

```typescript
// 1. Create and learn from data
var gaussian = GaussianND {};
var training_data = Tensor {};
training_data.init(TensorType::f64, Array<int> {1000, 20}); // 1000 samples, 20 features
// ... populate training_data ...

gaussian.learn(training_data);

// 2. Analyze statistics
var means = gaussian.avg();
var stds = gaussian.std();
var corr = gaussian.correlation();

println("Feature means: ${means}");
println("Feature stds: ${stds}");

// 3. Normalize new data
var test_data = Tensor {};
test_data.init(TensorType::f64, Array<int> {100, 20});
// ... populate test_data ...

var normalized = gaussian.standard_scaling(test_data);

// 4. Use normalized data for ML, then denormalize predictions
var predictions_normalized = model.predict(normalized);
var predictions = gaussian.inverse_standard_scaling(predictions_normalized);
```

---

## PCA

Principal Component Analysis for dimensionality reduction.

### Description

PCA identifies the principal components (directions of maximum variance) in high-dimensional data and projects it to a lower-dimensional space while retaining most of the variance.

### Fields

```typescript
static threshold_def: float = 0.95; // Default variance retention threshold

private eigen_vectors: Tensor?;     // Eigenvectors of correlation matrix
private eigen_values: Tensor?;      // Eigenvalues (variance explained)
private avg: Tensor?;               // Mean of original features
private std: Tensor?;               // Std deviation of original features
private correlation: Tensor?;       // Correlation matrix
private explained_variance: Tensor?; // Cumulative variance explained
private space_origin: Tensor?;      // Origin in reduced space
private dimension_info: Tensor?;    // Information about dimensions

best_dimension: int?;    // Best dimension for threshold
selected_dimension: int?; // User-selected dimension
threshold: float?;        // Variance retention threshold
space: Tensor?;          // Projection matrix
```

### Methods

#### learn(correlation: Tensor, avg: Tensor, std: Tensor, threshold: float?)
Learns PCA transformation from correlation matrix.

**Prerequisites:** Create a `GaussianND`, learn from data, then extract correlation, avg, std

**Parameters:**
- `correlation`: Correlation matrix from GaussianND
- `avg`: Average values per feature
- `std`: Standard deviations per feature
- `threshold`: Variance retention ratio (default: 0.95 = 95%)

**Example:**
```typescript
var gaussian = GaussianND {};
gaussian.learn(data);

var pca = PCA {};
pca.learn(
  gaussian.correlation()!!,
  gaussian.avg()!!,
  gaussian.std()!!,
  0.95  // Retain 95% of variance
);

println("Best dimension: ${pca.best_dimension}");
```

#### set_dimension(dimension: int)
Sets the target dimensionality for reduction.

**Parameters:**
- `dimension`: Number of dimensions to reduce to (must be ≤ original dimensions)

**Example:**
```typescript
pca.set_dimension(10); // Reduce to 10 dimensions
```

#### transform(input: Tensor): Tensor
Projects high-dimensional data to reduced space.

**Parameters:**
- `input`: Tensor of shape `[batch, N]` (original dimensions)

**Returns:** Tensor of shape `[batch, dim]` (reduced dimensions)

**Must call:** `learn()` and `set_dimension()` first

**Example:**
```typescript
var reduced = pca.transform(original_data);
// [1000, 50] → [1000, 10]
```

#### inverse_transform(input: Tensor): Tensor
Reconstructs original dimensions from reduced space.

**Parameters:**
- `input`: Tensor of shape `[batch, dim]` (reduced dimensions)

**Returns:** Tensor of shape `[batch, N]` (original dimensions)

**Note:** Reconstruction is approximate due to information loss

**Example:**
```typescript
var reconstructed = pca.inverse_transform(reduced);
// [1000, 10] → [1000, 50]
```

#### get_dimension(threshold: float): int
Calculates minimum dimensions needed to retain a given variance.

**Parameters:**
- `threshold`: Variance retention ratio (e.g., 0.90 = 90%)

**Returns:** Number of dimensions needed

**Example:**
```typescript
var dims_90 = pca.get_dimension(0.90); // Dims for 90% variance
var dims_95 = pca.get_dimension(0.95); // Dims for 95% variance
var dims_99 = pca.get_dimension(0.99); // Dims for 99% variance
```

### Complete PCA Workflow

```typescript
// 1. Prepare data with Gaussian profiling
var gaussian = GaussianND {};
gaussian.learn(training_data); // [10000, 100] - 100 features

// 2. Learn PCA
var pca = PCA {};
pca.learn(
  gaussian.correlation()!!,
  gaussian.avg()!!,
  gaussian.std()!!,
  0.95  // Retain 95% of variance
);

// 3. Find optimal dimensions
println("Dimensions for 90% variance: ${pca.get_dimension(0.90)}");
println("Dimensions for 95% variance: ${pca.best_dimension}");
println("Dimensions for 99% variance: ${pca.get_dimension(0.99)}");

// 4. Set dimension and transform
pca.set_dimension(20); // Reduce from 100 to 20 features

var reduced_train = pca.transform(training_data);   // [10000, 20]
var reduced_test = pca.transform(test_data);        // [1000, 20]

// 5. Train model on reduced data
// model.train(reduced_train, labels);

// 6. Optional: Reconstruct for visualization
var reconstructed = pca.inverse_transform(reduced_test);
var reconstruction_error = compute_error(test_data, reconstructed);
```

### PCA for Preprocessing

```typescript
// Use PCA as preprocessing for neural networks
var nn = RegressionNetwork::new(20, 1, TensorType::f32, false, 32, 42);
nn.setPreProcess(PreProcessType::pca_scaling, pca);

// Network will automatically apply PCA transform
var model = nn.build(true);
```

---

## Solver

Static linear equation solver.

### Description

Solves linear systems of equations: `X * w = Y` for `w`.

### Methods

#### solve(X: Tensor, Y: Tensor): Tensor (static)
Solves the linear system X*w = Y.

**Parameters:**
- `X`: Coefficient matrix of shape `[m, n]`
- `Y`: Target values of shape `[m, k]`

**Returns:** Solution tensor `w` of shape `[n, k]`

**Example:**
```typescript
var X = Tensor {}; // [100, 10] - 100 equations, 10 unknowns
var Y = Tensor {}; // [100, 1]  - 100 targets

var weights = Solver::solve(X, Y); // [10, 1] - solution
```

---

## Polynomial

Polynomial regression for curve fitting and time series compression.

### Description

Fits polynomial curves to data and provides compression for time series using polynomial approximation.

### Fields

```typescript
degree: int?;              // Polynomial degree
coefficients: Array<float>?; // Polynomial coefficients
x_start: float?;           // Starting X value
x_step: float?;            // X step size
```

### Methods

#### learn(degrees: int, X: Tensor, Y: Tensor): float
Fits a polynomial of specified degree to data.

**Parameters:**
- `degrees`: Polynomial degree (0 = constant, 1 = linear, 2 = quadratic, etc.)
- `X`: Input values (1D tensor)
- `Y`: Target values (1D tensor)

**Returns:** Fitting error (RMSE)

**Example:**
```typescript
var poly = Polynomial {};
var X = Tensor {}; // [100] - input values
var Y = Tensor {}; // [100] - target values

var error = poly.learn(3, X, Y); // Fit cubic polynomial
println("Fitting error: ${error}");
```

#### predict(X: Tensor): Tensor
Predicts values using learned polynomial.

**Parameters:**
- `X`: Input tensor (1D)

**Returns:** Predicted values tensor (1D)

#### predictValue(x: float): float
Predicts a single value.

**Parameters:**
- `x`: Input value

**Returns:** Predicted value

**Example:**
```typescript
var y_pred = poly.predictValue(5.0);
```

#### getDegrees(): int
Returns the degree of the fitted polynomial.

### Static Methods for Time Series Compression

#### compress(originalTS: nodeTime<float>, polynomialTS: nodeTime<Polynomial>, maxDegree: int, maxError: float, maxBufferSize: int) (static)
Compresses a time series using adaptive polynomial fitting.

**Parameters:**
- `originalTS`: Input time series
- `polynomialTS`: Output compressed representation
- `maxDegree`: Maximum polynomial degree allowed
- `maxError`: Maximum acceptable error
- `maxBufferSize`: Maximum buffer size for fitting

**How it works:**
1. Fits polynomial to sliding window of data
2. Increases degree if error exceeds threshold
3. Splits into new polynomial if max degree can't achieve error target
4. Stores only polynomial coefficients instead of all points

**Example:**
```typescript
var original = nodeTime<float> {};
var compressed = nodeTime<Polynomial> {};

Polynomial::compress(
  originalTS: original,
  polynomialTS: compressed,
  maxDegree: 5,
  maxError: 0.1,
  maxBufferSize: 1000
);

println("Compression ratio: ${original.size() / compressed.size()}");
```

#### decompress(originalTS: nodeTime<float>, polynomialTS: nodeTime<Polynomial>, maxError: float, decompressedTS: nodeTime<float>, errorTS: nodeTime<float>) (static)
Decompresses polynomial representation back to time series.

**Parameters:**
- `originalTS`: Original time series (for validation)
- `polynomialTS`: Compressed polynomial representation
- `maxError`: Maximum acceptable error (warns if exceeded)
- `decompressedTS`: Output decompressed time series
- `errorTS`: Output error time series

**Example:**
```typescript
var decompressed = nodeTime<float> {};
var errors = nodeTime<float> {};

Polynomial::decompress(
  original,
  compressed,
  0.1,
  decompressed,
  errors
);

// Validate compression quality
for (t, error in errors) {
  if (error > 0.1) {
    warn("High error at ${t}: ${error}");
  }
}
```

### Complete Polynomial Compression Example

```typescript
// Simulate sensor data with noise
var sensor_data = nodeTime<float> {};
for (hour in 0..24*30) { // 30 days of hourly data
  var t = time::new(2024, 1, 1, 0, 0, 0) + duration::new(hour, DurationUnit::hours);
  var value = sin(hour * 0.1) * 10 + random_noise();
  sensor_data.setAt(t, value);
}

// Compress
var compressed = nodeTime<Polynomial> {};
Polynomial::compress(sensor_data, compressed, 5, 0.5, 100);

println("Original size: ${sensor_data.size()} points");
println("Compressed size: ${compressed.size()} polynomials");
println("Compression ratio: ${sensor_data.size() / compressed.size()}x");

// Decompress and validate
var decompressed = nodeTime<float> {};
var errors = nodeTime<float> {};
Polynomial::decompress(sensor_data, compressed, 0.5, decompressed, errors);

// Calculate average error
var total_error = 0.0;
for (_, error in errors) {
  total_error = total_error + error;
}
println("Average error: ${total_error / errors.size()}");
```

---

## TimeSeriesDecomposition

Time series aggregation and decomposition utilities.

### Description

Provides methods to decompose high-frequency time series into lower-frequency aggregates (hourly → daily → weekly → monthly → yearly).

### Static Methods

All methods are static and work with `nodeTime` structures.

#### calculate(originalTS: nodeTime<any>, destinationTS: nodeTime<any>, tz: TimeZone, calendarUnit: CalendarUnit, lastUpdatedTime: time?)
Aggregates time series to coarser granularity.

**Parameters:**
- `originalTS`: High-frequency source time series
- `destinationTS`: Low-frequency destination time series
- `tz`: TimeZone for calendar calculations
- `calendarUnit`: Aggregation unit (hour, day, month, year)
- `lastUpdatedTime`: Optional - only update from this time forward

**Example:**
```typescript
var hourly = nodeTime<float> {};
var daily = nodeTime<float> {};

// Aggregate hourly to daily
TimeSeriesDecomposition::calculate(
  hourly,
  daily,
  TimeZone::Europe_Paris,
  CalendarUnit::day,
  null  // Process all data
);
```

#### calculateWeekly(originalTS: nodeTime<any>, destinationTS: nodeTime<any>, tz: TimeZone, lastUpdatedTime: time?)
Aggregates to weekly granularity (special handling for week boundaries).

#### calculateWithFactor(originalTS: nodeTime<any>, destinationTS: nodeTime<any>, tz: TimeZone, calendarUnit: CalendarUnit, factor: float, lastUpdatedTime: time?)
Aggregates with a multiplication factor.

**Use case:** Unit conversion during aggregation (e.g., Wh → kWh with factor 0.001)

#### calculateAll(instant: nodeTime<any>, hourly: nodeTime<any>?, daily: nodeTime<any>?, weekly: nodeTime<any>?, monthly: nodeTime<any>?, yearly: nodeTime<any>?, tz: TimeZone, lastUpdatedTime: time?)
Cascading aggregation from instant to all time scales.

**Parameters:**
- `instant`: High-frequency source (e.g., per-minute)
- `hourly`, `daily`, `weekly`, `monthly`, `yearly`: Optional destination time series
- `tz`: TimeZone
- `lastUpdatedTime`: Optional incremental update

**Example:**
```typescript
var instant = nodeTime<float> {};  // Per-minute data
var hourly = nodeTime<float> {};
var daily = nodeTime<float> {};
var monthly = nodeTime<float> {};

TimeSeriesDecomposition::calculateAll(
  instant,
  hourly,
  daily,
  null,    // No weekly
  monthly,
  null,    // No yearly
  TimeZone::UTC,
  null     // Process all
);
```

#### reset(ts: nodeTime<any>, tz: TimeZone, calendarUnit: CalendarUnit, lastUpdatedTime: time?)
Resets time series values to zero.

#### resetWeekly(ts: nodeTime<any>, tz: TimeZone, lastUpdatedTime: time?)
Resets weekly time series.

### Complete Energy Monitoring Example

```typescript
// Energy consumption monitoring with multiple granularities
var instant_consumption = nodeTime<float> {};  // kWh per minute
var hourly_consumption = nodeTime<float> {};
var daily_consumption = nodeTime<float> {};
var monthly_consumption = nodeTime<float> {};

// Simulate 30 days of per-minute data
var start_time = time::new(2024, 1, 1, 0, 0, 0);
for (minute in 0..30*24*60) {
  var t = start_time + duration::new(minute, DurationUnit::minutes);
  var consumption = random() * 2.0; // 0-2 kWh per minute
  instant_consumption.setAt(t, consumption);
}

// Decompose to all granularities
TimeSeriesDecomposition::calculateAll(
  instant_consumption,
  hourly_consumption,
  daily_consumption,
  null,  // Skip weekly
  monthly_consumption,
  null,  // Skip yearly
  TimeZone::Europe_Paris,
  null
);

// Report
println("Total consumption (30 days): ${sum(monthly_consumption)} kWh");
println("Average daily consumption: ${avg(daily_consumption)} kWh");
println("Peak hour: ${max_time(hourly_consumption)}");

// Incremental update (add new day)
var last_update = time::new(2024, 1, 30, 23, 59, 59);
// ... add new minute data ...
TimeSeriesDecomposition::calculateAll(
  instant_consumption,
  hourly_consumption,
  daily_consumption,
  null,
  monthly_consumption,
  null,
  TimeZone::Europe_Paris,
  last_update  // Only update from this time
);
```

---

## Best Practices

### Data Normalization

```typescript
// Always normalize before ML
var gaussian = GaussianND {};
gaussian.learn(train_data);

// Choose normalization method:
// - Min-Max: When you need bounded [0,1] range
var normalized = gaussian.min_max_scaling(data);

// - Standard: When you need zero mean, unit variance (better for most ML)
var normalized = gaussian.standard_scaling(data);
```

### PCA Guidelines

```typescript
// 1. Always standardize before PCA
var gaussian = GaussianND {};
gaussian.learn(data);
var standardized = gaussian.standard_scaling(data);

// 2. Then learn PCA
var pca = PCA {};
pca.learn(gaussian.correlation(), gaussian.avg(), gaussian.std(), 0.95);

// 3. Choose dimensions based on variance retained
var dim_90 = pca.get_dimension(0.90);  // Faster, less accurate
var dim_95 = pca.get_dimension(0.95);  // Balanced
var dim_99 = pca.get_dimension(0.99);  // Slower, more accurate
```

### Time Series Compression

```typescript
// Choose parameters based on data characteristics:
// - maxDegree: Higher for complex patterns (3-7 typical)
// - maxError: Based on acceptable data loss (1-5% of signal range)
// - maxBufferSize: Larger for smoother signals (100-1000)

Polynomial::compress(
  sensor_data,
  compressed,
  maxDegree: 5,           // Quintic polynomials
  maxError: 0.1,          // 0.1 unit error tolerance
  maxBufferSize: 500      // 500 points per polynomial
);
```

## See Also

- [Neural Networks (nn.md)](nn.md) - Use GaussianND for preprocessing
- [Compute Engine (compute.md)](compute.md) - Low-level tensor operations
- [README.md](README.md) - Complete library overview
