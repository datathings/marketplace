# FFT & Signal Processing

Fast Fourier Transform (FFT) operations for frequency domain analysis, signal processing, and time series forecasting.

## Overview

The Transforms module provides:
- **FFT**: Bidirectional Fast Fourier Transform
- **FFTModel**: High-level interface for time series analysis
- **Frequency Analysis**: Extract dominant frequencies and harmonics
- **Extrapolation**: Forecast time series using frequency components

## Type Index

| Type | Purpose |
|------|---------|
| `FFT` | Core FFT engine (forward/inverse transforms) |
| `FFTModel` | High-level time series FFT analysis and forecasting |

---

## FFT

Core FFT engine for frequency domain transformations.

### Static Fields

```typescript
static log_min: float = -6666.0; // Minimum log value for dB conversion
```

### Creation

#### new(nb_samples: int, inverse: bool): FFT (static)
Creates an FFT engine.

**Parameters:**
- `nb_samples`: Number of samples (should use `get_next_fast_size()`)
- `inverse`: `false` for time→frequency, `true` for frequency→time

**Example:**
```typescript
var size = FFT::get_next_fast_size(1000); // Get optimal size ≥ 1000
var fft = FFT::new(size, false);          // Time to frequency
var ifft = FFT::new(size, true);          // Frequency to time
```

### Core Transform

#### transform(timeseries_complex: Tensor, frequency_complex: Tensor)
Performs FFT transformation.

**Parameters:**
- `timeseries_complex`: Input tensor (complex128)
- `frequency_complex`: Output tensor (complex128)

**Direction:**
- If `inverse=false`: time → frequency (forward FFT)
- If `inverse=true`: frequency → time (inverse FFT)

**Example:**
```typescript
var time_data = Tensor {};
time_data.init(TensorType::c128, Array<int> {1024});
// ... fill time_data ...

var freq_data = Tensor {};
fft.transform(time_data, freq_data);
```

### High-Level Analysis

#### transform_table(timeseries: Table, timeseries_complex: Tensor, frequency_complex: Tensor): Table
Transforms a time series table and returns frequency analysis.

**Parameters:**
- `timeseries`: Input table with time series data
- `timeseries_complex`: Working tensor for complex time data
- `frequency_complex`: Working tensor for complex frequency data

**Returns:** Table with columns:
- `frequency` (Hz)
- `period` (duration)
- `frequency_abs_power`: Magnitude
- `power_db`: Power in decibels
- `phase_angle`: Phase (radians)
- `cumulated_ratio`: Cumulative energy ratio

**Example:**
```typescript
var fft = FFT::new(1024, false);
var time_complex = Tensor {};
var freq_complex = Tensor {};

var freq_table = fft.transform_table(timeseries_table, time_complex, freq_complex);

// Analyze dominant frequencies
for (row in 0..freq_table.rows()) {
  var freq = freq_table.get_cell(row, 0);
  var power = freq_table.get_cell(row, 2);
  println("Frequency: ${freq} Hz, Power: ${power}");
}
```

### Frequency Analysis

#### get_frequency_labels(sampling_step: duration, nb_samples: int): Tensor (static)
Returns frequency labels for FFT output.

**Parameters:**
- `sampling_step`: Time between samples
- `nb_samples`: Number of samples

**Returns:** Tensor with frequencies in Hz

**Example:**
```typescript
var labels = FFT::get_frequency_labels(
  duration::new(1, DurationUnit::seconds),
  1024
);
```

#### get_frequency_spectrum(frequency_complex: Tensor, frequency_spectrum: Tensor, to_decibel: bool, low_pass_filter: int?)
Extracts frequency spectrum (magnitudes).

**Parameters:**
- `frequency_complex`: Complex frequency tensor
- `frequency_spectrum`: Output spectrum tensor
- `to_decibel`: Convert to dB (20*log10)
- `low_pass_filter`: Optional - only return first N frequencies

**Example:**
```typescript
var spectrum = Tensor {};
FFT::get_frequency_spectrum(freq_complex, spectrum, true, 100);
// Returns first 100 frequencies in dB
```

#### get_frequency_table(frequency_complex: Tensor, sampling_step: duration): Table (static)
Extracts detailed frequency analysis table.

**Returns:** Table with frequency, power, phase, period, dB, cumulative ratio

### Filtering

#### apply_low_pass_filter(frequency_complex: Tensor, destination_frequency_complex: Tensor, low_pass_filter: int)
Applies low-pass filter (retains low frequencies, zeros high frequencies).

**Parameters:**
- `frequency_complex`: Input frequency tensor
- `destination_frequency_complex`: Filtered output
- `low_pass_filter`: Number of frequencies to keep

**Example:**
```typescript
var filtered = Tensor {};
FFT::apply_low_pass_filter(freq_data, filtered, 50); // Keep 50 lowest frequencies
```

#### get_low_pass_filter_size(frequency_complex: Tensor, ratio: float): int (static)
Determines filter size to retain a ratio of signal energy.

**Parameters:**
- `frequency_complex`: Frequency tensor
- `ratio`: Energy ratio to retain (0.0-1.0)

**Returns:** Number of frequencies to keep

**Example:**
```typescript
var filter_size = FFT::get_low_pass_filter_size(freq_data, 0.95);
// Returns N frequencies that contain 95% of signal energy
```

### Extrapolation

#### extrapolate(frequency_complex: Tensor, sampling_step: duration, timeseries_start: time, t: time, low_pass_filter: int?): float (static)
Predicts a single value at time t.

**Parameters:**
- `frequency_complex`: Learned frequency representation
- `sampling_step`: Original sampling step
- `timeseries_start`: Start time of training data
- `t`: Time to predict
- `low_pass_filter`: Optional frequency limit

**Returns:** Predicted value

#### extrapolate_table(timeseries_complex: Tensor, sampling_step: duration, timeseries_start: time, from: time, to: time, skip_elements: int?): Table (static)
Predicts multiple values over a time range.

**Parameters:**
- `timeseries_complex`: Time domain representation
- `sampling_step`: Sampling step
- `timeseries_start`: Training start time
- `from`: Prediction start
- `to`: Prediction end
- `skip_elements`: Optional downsampling

**Returns:** Table with predicted time series

### Utility Methods

#### get_next_fast_size(nb_samples: int): int (static)
Returns next power-of-2 or highly composite number ≥ nb_samples for efficient FFT.

**Example:**
```typescript
var optimal = FFT::get_next_fast_size(1000); // Returns 1024
```

#### frequency_to_period(frequency: float): duration (static)
Converts frequency (Hz) to period.

**Returns:** `0_s` for DC component (frequency=0)

#### period_to_frequency(period: duration): float (static)
Converts period to frequency (Hz).

#### power_to_db(power: float): float (static)
Converts power to decibels (20*log10).

**Returns:** `log_min` (-6666.0) for power ≤ 0

---

## FFTModel

High-level interface for FFT-based time series analysis and forecasting.

### Fields

```typescript
nt: nodeTime;              // Time series data
sampling_step: duration;   // Time between samples
time_complex: Tensor;      // Time domain representation
frequency_complex: Tensor; // Frequency domain representation
frequency_table: Table;    // Frequency analysis
start_time: time;          // Training start time
best_size: int;            // Optimal FFT size
```

### Training

#### train(nt: nodeTime, from: time, to: time): FFTModel (static)
Trains FFT model on time series data.

**Parameters:**
- `nt`: Time series
- `from`: Training start time
- `to`: Training end time

**Returns:** Trained FFTModel

**Example:**
```typescript
var sensor_data = nodeTime<float> {};
// ... populate sensor_data ...

var model = FFTModel::train(
  sensor_data,
  time::new(2024, 1, 1, 0, 0, 0),
  time::new(2024, 2, 1, 0, 0, 0)
);

// Analyze dominant frequencies
var freq_table = model.frequency_table;
```

### Forecasting

#### extrapolate_value(t: time, low_pass_filter_ratio: float?, nb_harmonics: int?): float
Predicts a single value.

**Parameters:**
- `t`: Time to predict
- `low_pass_filter_ratio`: Energy ratio to use (0.0-1.0)
- `nb_harmonics`: Number of harmonics to use

**Example:**
```typescript
// Predict next hour
var next_hour = time::new(2024, 2, 1, 1, 0, 0);
var prediction = model.extrapolate_value(next_hour, 0.95, null);
```

#### extrapolate(from: time, to: time, low_pass_filter_ratio: float?, nb_harmonics: int?, skip_elements: int?): Table
Predicts multiple values.

**Parameters:**
- `from`: Prediction start
- `to`: Prediction end
- `low_pass_filter_ratio`: Energy ratio
- `nb_harmonics`: Number of harmonics
- `skip_elements`: Downsampling factor

**Returns:** Table with predictions

**Example:**
```typescript
var predictions = model.extrapolate(
  time::new(2024, 2, 1, 0, 0, 0),
  time::new(2024, 2, 8, 0, 0, 0),
  0.95,    // Use 95% of energy
  null,    // All harmonics
  null     // No downsampling
);
```

---

## Complete Examples

### Frequency Analysis

```typescript
// Analyze sensor data for dominant frequencies
var sensor_data = nodeTime<float> {};
// ... populate with hourly readings for 30 days ...

var model = FFTModel::train(
  sensor_data,
  time::new(2024, 1, 1, 0, 0, 0),
  time::new(2024, 1, 31, 0, 0, 0)
);

// Find dominant frequencies
var freq_table = model.frequency_table;

println("Top 5 frequencies:");
for (row in 0..min(5, freq_table.rows())) {
  var freq = freq_table.get_cell(row, 0) as float;
  var period = freq_table.get_cell(row, 4) as duration;
  var power_db = freq_table.get_cell(row, 5) as float;

  println("  ${freq} Hz (period: ${period}): ${power_db} dB");
}

// Detect daily/weekly cycles
for (row in 0..freq_table.rows()) {
  var period = freq_table.get_cell(row, 4) as duration;
  var period_hours = period.to(DurationUnit::hours);

  if (period_hours > 23 && period_hours < 25) {
    println("Daily cycle detected!");
  } else if (period_hours > 167 && period_hours < 169) {
    println("Weekly cycle detected!");
  }
}
```

### Time Series Forecasting

```typescript
// Train on historical data
var electricity_usage = nodeTime<float> {};
// ... 6 months of hourly data ...

var model = FFTModel::train(
  electricity_usage,
  time::new(2024, 1, 1, 0, 0, 0),
  time::new(2024, 6, 30, 23, 0, 0)
);

// Forecast next week
var forecast = model.extrapolate(
  time::new(2024, 7, 1, 0, 0, 0),
  time::new(2024, 7, 7, 23, 0, 0),
  0.95,    // 95% energy retention
  10,      // Top 10 harmonics
  null
);

// Store forecasts
var forecast_ts = nodeTime<float> {};
for (row in 0..forecast.rows()) {
  var t = forecast.get_cell(row, 0) as time;
  var value = forecast.get_cell(row, 1) as float;
  forecast_ts.setAt(t, value);
}
```

### Signal Filtering

```typescript
// Remove high-frequency noise
var noisy_signal = nodeTime<float> {};
// ... populate ...

var model = FFTModel::train(noisy_signal, start, end);

// Apply low-pass filter
var filter_size = FFT::get_low_pass_filter_size(
  model.frequency_complex,
  0.9  // Retain 90% of energy
);

var filtered_freq = Tensor {};
FFT::apply_low_pass_filter(
  model.frequency_complex,
  filtered_freq,
  filter_size
);

// Inverse FFT to get filtered signal
var ifft = FFT::new(model.best_size, true);
var filtered_time = Tensor {};
ifft.transform(filtered_freq, filtered_time);

// Extract filtered values
var filtered_signal = nodeTime<float> {};
// ... convert tensor to nodeTime ...
```

### Harmonic Analysis

```typescript
// Analyze specific harmonics (e.g., daily, weekly patterns)
var model = FFTModel::train(temperature_data, start, end);

// Get frequency table sorted by power
var freq_table = model.frequency_table;

// Extract 1st harmonic (fundamental frequency)
var fundamental_freq = freq_table.get_cell(0, 0) as float;
var fundamental_period = freq_table.get_cell(0, 4) as duration;

// Reconstruct using only fundamental
var fundamental_only = model.extrapolate(
  start,
  end,
  null,
  1,     // Only 1st harmonic
  null
);

// Reconstruct using first 3 harmonics
var three_harmonics = model.extrapolate(
  start,
  end,
  null,
  3,     // 3 harmonics
  null
);
```

### Anomaly Detection

```typescript
// Use FFT reconstruction error for anomaly detection
var normal_data = nodeTime<float> {};
var test_data = nodeTime<float> {};

// Train on normal data
var model = FFTModel::train(normal_data, start, end);

// Reconstruct test data
var reconstructed = model.extrapolate(test_start, test_end, 0.95, null, null);

// Compare with actual
for (row in 0..reconstructed.rows()) {
  var t = reconstructed.get_cell(row, 0) as time;
  var predicted = reconstructed.get_cell(row, 1) as float;
  var actual = test_data.getAt(t);

  var error = abs(predicted - actual);
  var threshold = 3.0; // 3 sigma rule

  if (error > threshold) {
    println("Anomaly detected at ${t}: error=${error}");
  }
}
```

---

## Mathematical Background

### FFT Formula

**Forward FFT (time → frequency):**
```
X[k] = Σ(n=0 to N-1) x[n] * e^(-2πikn/N)
```

**Inverse FFT (frequency → time):**
```
x[n] = (1/N) * Σ(k=0 to N-1) X[k] * e^(2πikn/N)
```

### Frequency Resolution

```
Δf = 1 / (N * Δt)
```

Where:
- `Δf`: Frequency resolution (Hz)
- `N`: Number of samples
- `Δt`: Sampling period (seconds)

### Nyquist Frequency

Maximum detectable frequency:
```
f_max = 1 / (2 * Δt)
```

---

## Performance Tips

### Optimal Sample Size
```typescript
// Always use optimal sizes for faster FFT
var size = FFT::get_next_fast_size(actual_samples);
// Pads data to next power-of-2 or highly composite number
```

### Memory Usage
```typescript
// Reuse tensors for multiple transforms
var time_data = Tensor {};
var freq_data = Tensor {};
time_data.init(TensorType::c128, Array<int> {1024});
freq_data.init(TensorType::c128, Array<int> {1024});

for (segment in segments) {
  // Fill time_data with segment
  fft.transform(time_data, freq_data);
  // Process freq_data
}
```

### Low-Pass Filtering
```typescript
// Reduce computation by filtering
var filter_size = FFT::get_low_pass_filter_size(freq_data, 0.9);
// Typically 10-20% of original size for 90% energy
```

---

## Use Cases

### 1. Periodic Pattern Detection
```typescript
// Find daily/weekly/monthly cycles
var freq_table = model.frequency_table;
// Examine dominant frequencies and their periods
```

### 2. Signal Denoising
```typescript
// Remove high-frequency noise
FFT::apply_low_pass_filter(freq_data, filtered, filter_size);
```

### 3. Time Series Forecasting
```typescript
// Extrapolate based on learned frequencies
var forecast = model.extrapolate(future_start, future_end, 0.95, null, null);
```

### 4. Compression
```typescript
// Store only dominant harmonics
var compressed = model.extrapolate(start, end, null, 5, null);
// 5 harmonics instead of all data points
```

### 5. Similarity Analysis
```typescript
// Compare frequency signatures of two time series
var model1 = FFTModel::train(ts1, start, end);
var model2 = FFTModel::train(ts2, start, end);

var distance = frequency_distance(
  model1.frequency_complex,
  model2.frequency_complex
);
```

---

## Best Practices

### Sample Size Selection
```typescript
// Need enough samples for desired frequency resolution
var min_period = 1_h;  // Want to detect hourly patterns
var samples_needed = 24 * 7; // At least 1 week for stability
var optimal_size = FFT::get_next_fast_size(samples_needed);
```

### Windowing
```typescript
// For non-periodic signals, apply windowing to reduce spectral leakage
// (Currently not built-in, apply manually to input data)
```

### Handling Gaps
```typescript
// FFT requires evenly sampled data
// Use interpolation for gaps before FFT
```

## See Also

- [Pattern Recognition (patterns.md)](patterns.md) - FFT-based pattern detection
- [README.md](README.md) - Library overview
