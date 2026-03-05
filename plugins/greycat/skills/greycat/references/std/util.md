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

// Median and standard deviation of the window
var median = window.median(); // returns float?
var std_dev = window.std();   // returns float?
```

### TimeWindow
Time-based sliding window that maintains values within a duration span, automatically expiring old entries. Includes statistical functions for time-series analysis.

```gcl
// Keep only values from the last 5 minutes
var time_window = TimeWindow<float> { span: 5min };

// Add timestamped data
time_window.add(time::now(), temperature);
time_window.add(time::now() + 30s, next_temp);

// Advance the window in time without adding a value (expires old entries)
time_window.update(time::now() + 10min);

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

// Add multiple counts at once
profile.addx(15.0, 5); // adds 15.0 five times

// Merge another Gaussian into this one
var other = Gaussian<float> {};
other.add(25.0);
other.add(35.0);
profile.add_gaussian(other);

// Normalization: (value-min)/(max-min)
Assert::equalsd(profile.normalize(15.0)!!, 0.25, 0.001);

// Inverse normalization: value*(max-min)+min
var original = profile.inverse_normalize(0.25);

// Standardization: (value-avg)/std
var standardized = profile.standardize(25.0);
Assert::isTrue(standardized > 0.0); // above average

// Inverse standardization: (value*std)+avg
var unstandardized = profile.inverse_standardize(standardized);

// Confidence score for a value
var conf = profile.confidence(25.0);

// Probability density function (PDF) at a value
var density = profile.pdf(20.0);

// Cumulative distribution function (CDF) at a value
var cumulative = profile.cdf(20.0);
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

// Add with count: add a value multiple times
histogram.addx(75.0, 10); // adds 75.0 ten times

// Analyze distribution
var median = histogram.percentile(0.5); // 50th percentile
var top_10_percent = histogram.percentile(0.9);
var below_passing = histogram.ratio_under(60.0); // fraction below 60
var stats = histogram.stats(); // comprehensive statistics (HistogramStats<T>)

// Get all bins as Array<HistogramBin<T>>
var bins = histogram.get_bins();
for (_, bin_entry in bins) {
    // Each HistogramBin has: bin (QuantizerSlotBound<T>), count, ratio, cumulative_count, cumulative_ratio
    var bounds = bin_entry.bin; // QuantizerSlotBound with min, max, center
    var count = bin_entry.count;
    var ratio = bin_entry.ratio;
}
```

#### HistogramBin
Represents a single bin in a histogram with count and ratio statistics. Fields: `bin: QuantizerSlotBound<T>`, `count: int`, `ratio: float`, `cumulative_count: int`, `cumulative_ratio: float`.

#### HistogramStats
Comprehensive statistics returned by `histogram.stats()`. Fields include all standard percentiles (`min`, `max`, `whisker_low`, `whisker_high`, `p1`, `p5`, `p10`, `p20`, `p25`, `p50`, `p75`, `p80`, `p90`, `p95`, `p99`), plus `sum: float`, `avg: T`, `std: T`, `size: int`.

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
var salary_sum_for_group = profile.sum(age_group);
var salary_count_for_group = profile.count(age_group);
```

#### GaussianProfileSlot
Represents a single slot in a GaussianProfile, tracking per-bin statistics. Fields: `sum: int`, `sumsq: int`, `count: int`.

## Quantizers

### QuantizerSlotBound
Represents the bounds of a quantizer slot. Fields: `min: T`, `max: T`, `center: T`. Returned by `quantizer.bounds(slot)` and used in `HistogramBin`.

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

// Random character between 'a' and 'z'
var c = rng.char();

// Test uniform distribution bounds
var roll = rng.uniform(1, 7); // 1-6 inclusive
Assert::isTrue(roll >= 1 && roll < 7);

var probability = rng.uniformf(0.0, 1.0);
Assert::isTrue(probability >= 0.0 && probability < 1.0);

// Random geo point within bounding box
var random_point = rng.uniformGeo(geo::new(40.0, -74.0), geo::new(41.0, -73.0));

// Normal distribution with specified mean and standard deviation
var value = rng.normal(50.0, 10.0); // avg=50, std=10

// Sample from an existing Gaussian profile
var profile = Gaussian<float> {};
profile.add(10.0);
profile.add(20.0);
profile.add(30.0);
var sampled = rng.gaussian(profile);

// Fill collection with random uniform values
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
Assert::isFalse(should_not_happen);
Assert::isNull(optional_value);
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
Cryptographic functions including SHA hashing, HMAC, PKCS1 signing, Base64/Base64URL encoding, hex encoding, and URL encoding.

```gcl
// SHA hash functions (binary and hex output)
var input = "hello world";
var sha1_bin = Crypto::sha1(input); // binary SHA-1
var sha1_hex = Crypto::sha1hex(input); // hex-encoded SHA-1
var sha256_bin = Crypto::sha256(input); // binary SHA-256
var sha256_hex = Crypto::sha256hex(input); // hex-encoded SHA-256

// PKCS1 signing with SHA-256
var signature = Crypto::sha256_sign_pkcs1(input, "/path/to/private_key.pem");
var signature_hex = Crypto::sha256_sign_pkcs1_hex(input, "/path/to/private_key.pem");

// HMAC-SHA256 (hex-encoded)
var hmac = Crypto::sha256_hmac_hex(input, "secret_key");

// Base64 encoding/decoding round trip
var original = "test string with spaces";
var encoded = Crypto::base64_encode(original);
var decoded = Crypto::base64_decode(encoded);
Assert::equals(decoded, original);

// Base64URL encoding/decoding (URL-safe variant)
var b64url_encoded = Crypto::base64url_encode(original);
var b64url_decoded = Crypto::base64url_decode(b64url_encoded);
Assert::equals(b64url_decoded, original);

// Hex encoding/decoding
var hex_encoded = Crypto::hex_encode(original);
var hex_decoded = Crypto::hex_decode(hex_encoded);
Assert::equals(hex_decoded, original);

// URL encoding/decoding round trip
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
