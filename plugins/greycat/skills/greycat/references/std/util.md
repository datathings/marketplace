# [std](index.html) > util

[Source](util.source.html)

Utility module providing fundamental data structures and helper types for GreyCat applications.

## Collections

### Queue
FIFO collection with optional capacity bounds. When capacity is reached, front elements are automatically dropped.

```gcl
// Create a bounded queue that keeps only the last 3 items
var queue = Queue<String> { capacity: 3 };
queue.push("first");
queue.push("second");
queue.push("third");
queue.push("fourth"); // "first" gets dropped automatically

Assert::equals(queue.pop(), "second");
Assert::equals(queue.front(), "third");
Assert::equals(queue.front(), "third");
Assert::equals(queue.back(), "fourth");
Assert::equals(queue.back(), "fourth");
```

### Stack
Standard LIFO collection for last-in-first-out operations.

```gcl
// Basic stack operations
var stack = Stack<int> {};
stack.push(10);
stack.push(20);
stack.push(30);

Assert::equals(stack.pop(), 30);
Assert::equals(stack.last(), 20); // peek without removing
Assert::equals(stack.first(), 10); // bottom element
```

### SlidingWindow
Fixed-size FIFO collection that maintains statistical aggregates (avg, std, median, min, max) over the most recent N values. Perfect for streaming analytics.

```gcl
// Moving average over last 3 values
var window = SlidingWindow<float> { span: 3 };

// Add streaming data
window.add(10.0);
window.add(20.0);
window.add(30.0);

Assert::equalsd(window.avg()!!, 20.0, 0.001);
Assert::equals(window.size(), 3);

// Add another value, pushing out the first
window.add(40.0);
Assert::equalsd(window.avg()!!, 30.0, 0.001); // (20+30+40)/3
Assert::equals(window.min(), 20.0);
Assert::equals(window.max(), 40.0);
```

### TimeWindow
Time-based sliding window that maintains values within a duration span, automatically expiring old entries. Includes statistical functions for time-series analysis.

```gcl
// Keep only values from the last 5 minutes
var time_window = TimeWindow<float> { span: 5min };

// Add timestamped data
time_window.add(time::now(), temperature);
time_window.add(time::now() + 30s, next_temp);

// Get statistics for recent data only
var recent_avg = time_window.avg();
var min_reading = time_window.min(); // returns Tuple<time, float>
```

## Statistics & Analysis

### Gaussian
Live statistical profile that tracks running mean, standard deviation, and distribution properties. Supports normalization, standardization, and probability calculations (PDF/CDF).

```gcl
// Build statistical profile incrementally
var profile = Gaussian<float> {};

// Add data points
profile.add(10.0);
profile.add(20.0);
profile.add(30.0);

Assert::equalsd(profile.avg()!!, 20.0, 0.001);
Assert::equals(profile.min, 10.0);
Assert::equals(profile.max, 30.0);

// Normalization: (value-min)/(max-min)
Assert::equalsd(profile.normalize(15.0)!!, 0.25, 0.001);

// Standardization: (value-avg)/std
var standardized = profile.standardize(25.0);
Assert::isTrue(standardized > 0.0); // above average
```

### Histogram
Binned data distribution analyzer with configurable quantizers. Provides percentile calculations, ratio analysis, and comprehensive statistical summaries.

```gcl
// Create histogram with 20 uniform bins between 0-100
var quantizer = LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 20 };
var histogram = Histogram<float> { quantizer: quantizer };

// Add data points
for (_, score in test_scores) {
    histogram.add(score);
}

// Analyze distribution
var median = histogram.percentile(0.5); // 50th percentile
var top_10_percent = histogram.percentile(0.9);
var below_passing = histogram.ratio_under(60.0); // fraction below 60
var stats = histogram.stats(); // comprehensive statistics
```

### GaussianProfile
Multi-dimensional Gaussian statistics indexed by quantized keys for categorical analysis.

```gcl
// Profile statistics by category
var quantizer = LinearQuantizer<int> { min: 0, max: 100, bins: 10 };
var profile = GaussianProfile<int> { quantizer: quantizer, precision: FloatPrecision::p1000 };

// Add data points with categories
profile.add(age_group, salary);
profile.add(age_group, another_salary);

// Get statistics per category
var avg_salary_for_group = profile.avg(age_group);
var salary_std_for_group = profile.std(age_group);
```

## Quantizers

### LinearQuantizer
Uniform binning with equal-width intervals

```gcl
// 10 equal bins from 0 to 100
var linear = LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 10 };

Assert::equals(linear.size(), 10);
Assert::equals(linear.quantize(25.0), 2); // 25.0 falls in bin 2
Assert::equals(linear.quantize(95.0), 9); // 95.0 falls in bin 9

// Get bounds for bin 2
var bounds = linear.bounds(2);
Assert::equals(bounds.min, 20.0);
Assert::equals(bounds.max, 30.0);
Assert::equals(bounds.center, 25.0);
```

### LogQuantizer
Logarithmic binning for exponential data distributions

```gcl
// Logarithmic bins for data with exponential distribution
var log_quantizer = LogQuantizer<float> { min: 1.0, max: 1000.0, bins: 10 };
var bin = log_quantizer.quantize(50.0); // maps to appropriate log bin
```

### CustomQuantizer
User-defined bin boundaries for irregular distributions

```gcl
// Custom age groups: 0-18, 18-25, 25-40, 40-65, 65+
var age_quantizer = CustomQuantizer<int> {
    min: 0,
    max: 100,
    step_starts: [0, 18, 25, 40, 65]
};
var age_group = age_quantizer.quantize(32); // returns appropriate bin
```

### MultiQuantizer
Multi-dimensional quantization for complex data structures

```gcl
// Quantize multi-dimensional data like [age, income, score]
var quantizers = Array<Quantizer<float>> {
    LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 5 },    // age groups
    LogQuantizer<float> { min: 1000.0, max: 200000.0, bins: 8 }, // income brackets
    LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 10 }    // score ranges
};
var multi = MultiQuantizer<float> { quantizers: quantizers };
var slot = multi.quantize([35.0, 45000.0, 87.5]); // single slot index
var vector = multi.slot_vector(slot); // [age_bin, income_bin, score_bin]
```

## Utilities

### Random
Seeded random number generator with uniform, normal, and Gaussian distributions. Supports various data types including geo coordinates.

```gcl
// Reproducible random numbers with fixed seed
var rng = Random { seed: 12345 };

// Test uniform distribution bounds
var roll = rng.uniform(1, 7); // 1-6 inclusive
Assert::isTrue(roll >= 1 && roll < 7);

var probability = rng.uniformf(0.0, 1.0);
Assert::isTrue(probability >= 0.0 && probability < 1.0);

// Normal distribution should center around mean
var samples = Array<float> {};
rng.fill(samples, 1000, 50.0, 60.0);
Assert::equals(samples.size(), 1000);
```

### Assert
Testing utility with type-aware equality checks and boolean assertions.

```gcl
// Unit testing helpers
Assert::equals(calculated_result, expected_value);
Assert::equalsd(pi_approximation, 3.14159, 0.001); // float comparison with epsilon
Assert::equalst(tensor_a, tensor_b, 0.01); // tensor comparison with epsilon
Assert::isTrue(validation_passed);
Assert::isNotNull(database_connection);
```

### ProgressTracker
Performance monitoring for long-running operations with speed and ETA calculations.

```gcl
// Track progress of batch processing
var tracker = ProgressTracker { start: time::now(), total: 1000 };

// Simulate processing 300 items
tracker.update(300);

Assert::equalsd(tracker.progress!!, 0.3, 0.001); // 30% complete
Assert::equals(tracker.counter, 300);
Assert::isNotNull(tracker.speed);
Assert::isNotNull(tracker.remaining);

// Complete the task
tracker.update(1000);
Assert::equalsd(tracker.progress, 1.0, 0.001); // 100% complete
```

### Crypto
Cryptographic functions including SHA hashing, Base64 encoding, URL encoding, and PKCS1 signing.

```gcl
// Hash functions
var input = "hello world";
var sha1_result = Crypto::sha1hex(input);
var sha256_result = Crypto::sha256hex(input);

Assert::isNotNull(sha1_result);
Assert::isNotNull(sha256_result);
Assert::isTrue(sha1_result.size() > 0);
Assert::isTrue(sha256_result.size() > sha1_result.size());

// Encoding/decoding round trip
var original = "test string with spaces";
var encoded = Crypto::base64_encode(original);
var decoded = Crypto::base64_decode(encoded);
Assert::equals(decoded, original);

// URL encoding round trip
var url_encoded = Crypto::url_encode("param with spaces & symbols");
var url_decoded = Crypto::url_decode(url_encoded);
Assert::equals(url_decoded, "param with spaces & symbols");
```

### Plot
Basic plotting functionality for scatter plots from tabular data.

```gcl
// Simple temperature data over months
var data_table = Table {};
data_table.set_row(0, ["Jan", 1, -2, 8]);
data_table.set_row(1, ["Feb", 2, 1, 12]);
data_table.set_row(2, ["Mar", 3, 8, 18]);
data_table.set_row(3, ["Apr", 4, 15, 22]);
data_table.set_row(4, ["May", 5, 22, 28]);
data_table.set_row(5, ["Jun", 6, 28, 32]);
data_table.set_row(6, ["Jul", 7, 31, 35]);
data_table.set_row(7, ["Aug", 8, 29, 33]);
data_table.set_row(8, ["Sep", 9, 23, 28]);
data_table.set_row(9, ["Oct", 10, 16, 21]);
data_table.set_row(10, ["Nov", 11, 7, 14]);
data_table.set_row(11, ["Dec", 12, 2, 9]);

// Plot month (x) vs min_temp and max_temp (y series)
// Columns: [month_name, month_number, min_temp, max_temp]
Plot::scatter_plot(data_table, 1, [2, 3], "temperature_trends.png");
```
