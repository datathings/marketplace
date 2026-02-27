# Pattern Recognition

Time series pattern detection algorithms including Euclidean distance, Dynamic Time Warping (DTW), FFT, and Symbolic Aggregate approXimation (SAX).

## Overview

The Pattern Recognition module enables finding similar patterns in time series data using multiple detection algorithms:

- **Euclidean**: Fast distance-based matching
- **DTW**: Flexible time-aligned matching
- **FFT**: Frequency domain analysis
- **SAX**: Symbolic representation for large-scale search
- **Random**: Baseline/testing detector

## Quick Start

```typescript
// 1. Create detector
var detector = EuclideanPatternDetector {};
var engine = detector.getEngine(timeseries);
engine.setState(PatternDetectionEngineState::new());

// 2. Define patterns to search for
engine.addPattern(
  time::new(2024, 1, 1, 0, 0, 0),
  time::new(2024, 1, 2, 0, 0, 0)
);

// 3. Find similar patterns
engine.initScoring();
engine.computeScores(null);

// 4. Extract detections
var detections = engine.detect(PatternDetectionSensitivity {
  threshold: 0.85,     // Similarity threshold
  overlap: 0.2         // Maximum overlap between detections
}, null);
```

## Type Index

| Type | Purpose |
|------|---------|
| `PatternDetectionEngine` | Base engine for pattern detection |
| `EuclideanPatternDetectionEngine` | Euclidean distance detector |
| `DTWPatternDetectionEngine` | Dynamic Time Warping detector |
| `FFTPatternDetectionEngine` | Frequency domain detector |
| `SaxPatternDetectionEngine` | Symbolic detector |
| `PatternDetectionEngineState` | Engine state management |
| `Detection` / `OverlappingDetection` | Detection results |
| `PatternDetectionSensitivity` | Detection parameters |
| `DistanceMetrics` | Euclidean and DTW distance computation |

## Enums

### PatternDetectors
```typescript
enum PatternDetectors {
  none("None");
  euclidean("Euclidean");
  fft("FFT");
  dtw("DTW");
  random("Random");
  sax("SAX");
}
```

### SamplingPolicy
```typescript
enum SamplingPolicy {
  as_is("As-is");              // Use original sampling
  average_frequency("Average frequency");
  highest_frequency("Highest frequency");
}
```

### PatternNullStrategy
```typescript
enum PatternNullStrategy {
  replace("Replace");          // Replace with constant
  interpolate("Interpolate");  // Linear interpolation
  previous("Previous");        // Use previous non-null
  next("Next");                // Use next non-null
  none("None");                // Error on null
}
```

### MatchingNormalisation
```typescript
enum MatchingNormalisation {
  as_is("As-is");                      // Raw values
  shift("Vertical shift");             // Remove mean
  scaling("Vertical scaling");         // Normalize by std
  shift_and_scaling("Vertical shift and scaling"); // Z-score
}
```

---

## PatternDetectionEngine (Abstract)

Base class for all pattern detectors.

### Common Fields

```typescript
timeseries: nodeTime;                     // Time series to search
state: PatternDetectionEngineState?;      // Engine state
nullStrategy: PatternNullStrategy?;       // How to handle nulls
nullReplaceConstant: float?;              // Replacement value
samplingPolicy: SamplingPolicy?;          // Resampling policy
```

### Methods

#### setState(newState: PatternDetectionEngineState)
Sets engine state.

#### addPattern(from: time, to: time)
Adds a pattern to search for.

**Parameters:**
- `from`: Pattern start time
- `to`: Pattern end time

**Example:**
```typescript
engine.addPattern(
  time::new(2024, 1, 1, 12, 0, 0),
  time::new(2024, 1, 1, 18, 0, 0)
); // 6-hour pattern
```

#### setPatternsFromMarks(marks: nodeTime<bool?>)
Adds patterns from boolean marks.

**Format:**
- `true`: Pattern start
- `null` (or next `true`): Pattern end

**Example:**
```typescript
var marks = nodeTime<bool?> {};
marks.setAt(time::new(2024, 1, 1, 0, 0, 0), true);   // Start 1
marks.setAt(time::new(2024, 1, 2, 0, 0, 0), null);   // End 1
marks.setAt(time::new(2024, 2, 1, 0, 0, 0), true);   // Start 2
marks.setAt(time::new(2024, 2, 2, 0, 0, 0), null);   // End 2

engine.setPatternsFromMarks(marks);
```

#### initScoring()
Initializes scoring (detector-specific preparation).

#### computeScores(path: String?): String?
Computes similarity scores across time series.

**Parameters:**
- `path`: Optional path to save scores to GCB file (for large datasets)

**Returns:** Path if file was written, null otherwise

**Example:**
```typescript
// In-memory (small datasets)
engine.computeScores(null);

// File-based (large datasets)
var scores_path = engine.computeScores("/tmp/scores.gcb");
```

#### detect(sensitivity: PatternDetectionSensitivity, path: String?): String?
Extracts detections from scores.

**Parameters:**
- `sensitivity`: Detection thresholds
- `path`: Optional path to save detections

**Returns:** Path if file was written

**Example:**
```typescript
var detections_path = engine.detect(
  PatternDetectionSensitivity {
    threshold: 0.85,   // Minimum similarity (0-1)
    overlap: 0.3       // Max overlap ratio (0-1)
  },
  null
);
```

### score(index: int, pattern: TimeWindow, timeWindow: TimeWindow): float (abstract)
Computes similarity score (implemented by subclasses).

**Returns:** Score in range [0, 1] where 1 = perfect match

---

## EuclideanPatternDetectionEngine

Fast distance-based pattern detection.

### Description

Uses Euclidean distance in time-aligned windows. Fast and intuitive but requires aligned patterns.

### Additional Fields

```typescript
pattern_tensors: Array<Tensor>?;          // Precomputed pattern tensors
window_tensors: Array<Tensor>?;           // Reusable window tensors
std: float?;                              // Time series standard deviation
matchingNormalisation: MatchingNormalisation?; // Normalization mode
```

### Score Formula

```typescript
// Without normalization:
score = 1 - euclidean_distance / (std * pattern_size)

// With normalization:
score = 1 - euclidean_distance / pattern_size
```

### Example

```typescript
var detector = EuclideanPatternDetector {};
var engine = detector.getEngine(timeseries) as EuclideanPatternDetectionEngine;

// Configure
engine.setState(PatternDetectionEngineState::new());
engine.matchingNormalisation = MatchingNormalisation::shift_and_scaling;
engine.nullStrategy = PatternNullStrategy::interpolate;

// Add patterns
engine.addPattern(pattern_start, pattern_end);

// Detect
engine.initScoring();
engine.computeScores(null);
var detections = engine.state?.detections;
```

---

## DTWPatternDetectionEngine

Dynamic Time Warping for flexible time-aligned matching.

### Description

Allows patterns to match even with time distortions (compression/stretching). More robust but slower than Euclidean.

### Additional Fields

```typescript
std: float?;                              // Time series standard deviation
matchingNormalisation: MatchingNormalisation?;
```

### DTW Algorithm

Computes optimal alignment using dynamic programming:

```
DTW[i,j] = distance(pattern[i], window[j]) +
           min(DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1])
```

### Example

```typescript
var detector = DTWPatternDetector {};
var engine = detector.getEngine(timeseries) as DTWPatternDetectionEngine;
engine.setState(PatternDetectionEngineState::new());
engine.matchingNormalisation = MatchingNormalisation::shift_and_scaling;

engine.addPattern(pattern_start, pattern_end);
engine.initScoring();
engine.computeScores(null);
```

### Performance Note

DTW is O(n*m) where n = pattern length, m = window length. Use for:
- Patterns with variable speed
- When alignment is uncertain
- Smaller datasets (< 10k points)

---

## FFTPatternDetectionEngine

Frequency domain pattern detection.

### Description

Converts patterns and windows to frequency domain using FFT, then compares frequency components. Robust to amplitude shifts.

### Additional Fields

```typescript
low_pass_filter_ratio: float;            // Frequency filter (0-1)
pattern_frequencies: Array<Tensor>?;     // Pattern FFT
window_frequency: Tensor?;                // Window FFT
pattern_distance_profiles: Array<Gaussian<float>>?;
results: nodeTime<FFTResult>;             // Intermediate results
```

### Example

```typescript
var detector = FFTPatternDetector {};
var engine = detector.getEngine(timeseries) as FFTPatternDetectionEngine;
engine.setState(PatternDetectionEngineState::new());
engine.low_pass_filter_ratio = 0.9; // Retain 90% of frequency energy

engine.addPattern(pattern_start, pattern_end);
engine.initScoring();
engine.computeScores(null);
```

### Use Cases

- Periodic patterns (daily/weekly cycles)
- Patterns with phase shifts
- Noise-robust matching

---

## SaxPatternDetectionEngine

Symbolic Aggregate approXimation for efficient large-scale search.

### Description

Converts time series to symbolic strings, enabling fast approximate search. Ideal for very large datasets.

### Additional Fields

```typescript
alphabet_size: int;                      // Number of symbols (2-62)
alphabet_boundaries: Array<float>;       // Symbol boundaries
fingerprint_length: int;                 // Reduced resolution
pattern_fingerprints: Array<String>;     // Pattern symbols
lookup_table: nodeIndex<char, nodeIndex<char, float>>; // Distance table
```

### Creation

```typescript
var detector = SaxPatternDetector {
  alphabet_size: 10,        // 10 symbols
  fingerprint_length: 20    // 20 points per pattern
};
```

### SAX Process

1. **PAA (Piecewise Aggregate Approximation)**: Reduce resolution
   - Pattern → `fingerprint_length` segments
   - Each segment = average value

2. **Discretization**: Map to symbols
   - Values → symbols based on histogram percentiles

3. **Distance**: Compare symbolic strings using lookup table

### Example

```typescript
var detector = SaxPatternDetector {
  alphabet_size: 10,
  fingerprint_length: 20
};
var engine = detector.getEngine(timeseries) as SaxPatternDetectionEngine;
engine.setState(PatternDetectionEngineState::new());

engine.addPattern(pattern_start, pattern_end);
engine.initScoring();
engine.computeScores(null);
```

### Use Cases

- Very large time series (> 1M points)
- Approximate search
- Real-time detection

---

## PatternDetectionEngineState

Manages detector state and results.

### Fields

```typescript
hasScores: bool;                          // Scores computed?
hasDetections: bool;                      // Detections extracted?
patterns: Array<TimeWindow>;              // Defined patterns
scores: nodeList<any?>;                   // Computed scores
detections: nodeTime<OverlappingDetection>; // Extracted detections
```

### Methods

#### resetPatterns()
Clears all patterns and scores.

#### resetScores()
Clears scores and detections.

#### resetDetections()
Clears detections only.

**Example:**
```typescript
var state = PatternDetectionEngineState::new();
engine.setState(state);

// After first run
engine.resetScores(); // Re-compute with different parameters
```

---

## Detection Types

### Detection
```typescript
abstract type Detection {
  score: float;            // Similarity score [0, 1]
  best_pattern: int;       // Which pattern matched
  timespan: duration;      // Detection duration
}
```

### OverlappingDetection
```typescript
type OverlappingDetection extends Detection {
  overlap: duration;       // Overlap with other detections
}
```

### Accessing Detections

```typescript
for (timestamp: time, detection: OverlappingDetection in engine.state?.detections?) {
  println("Detection at ${timestamp}:");
  println("  Pattern: ${detection.best_pattern}");
  println("  Score: ${detection.score}");
  println("  Duration: ${detection.timespan}");
  println("  Overlap: ${detection.overlap}");
}
```

---

## Complete Workflow Example

```typescript
// 1. Prepare time series
var sensor_data = nodeTime<float> {};
// ... populate with sensor readings ...

// 2. Choose detector
var detector = EuclideanPatternDetector {}; // Fast
// var detector = DTWPatternDetector {};    // Flexible
// var detector = FFTPatternDetector {};    // Periodic
// var detector = SaxPatternDetector { alphabet_size: 10, fingerprint_length: 20 }; // Large-scale

var engine = detector.getEngine(sensor_data);
engine.setState(PatternDetectionEngineState::new());

// 3. Configure
engine.nullStrategy = PatternNullStrategy::interpolate;
engine.samplingPolicy = SamplingPolicy::average_frequency;

if (engine is EuclideanPatternDetectionEngine) {
  (engine as EuclideanPatternDetectionEngine).matchingNormalisation =
    MatchingNormalisation::shift_and_scaling;
}

// 4. Define patterns (known anomalies, events, etc.)
engine.addPattern(
  time::new(2024, 1, 15, 10, 0, 0),
  time::new(2024, 1, 15, 12, 0, 0)
); // Morning spike pattern

engine.addPattern(
  time::new(2024, 2, 20, 14, 0, 0),
  time::new(2024, 2, 20, 16, 0, 0)
); // Afternoon dip pattern

// 5. Compute scores
engine.initScoring();
engine.computeScores(null);

// 6. Extract detections
var detections = engine.detect(
  PatternDetectionSensitivity {
    threshold: 0.85,    // High confidence only
    overlap: 0.2        // Allow 20% overlap
  },
  null
);

// 7. Process results
println("Found ${engine.state?.detections?.size()} detections");

for (t: time, det: OverlappingDetection in engine.state?.detections?) {
  println("${t}: Pattern ${det.best_pattern}, Score ${det.score}");

  // Take action
  if (det.score > 0.95) {
    // High confidence match
    alert("Pattern detected at ${t}");
  }
}
```

---

## Detector Comparison

| Detector | Speed | Flexibility | Memory | Best For |
|----------|-------|-------------|--------|----------|
| **Euclidean** | Fast | Low | Low | Aligned patterns, quick search |
| **DTW** | Slow | High | Medium | Variable-speed patterns |
| **FFT** | Medium | Medium | Medium | Periodic patterns, noise |
| **SAX** | Fast | Low | Low | Large-scale approximate search |

### Recommendations

- **Start with Euclidean** for most cases
- **Use DTW** if patterns have time distortions
- **Use FFT** for periodic/frequency-based patterns
- **Use SAX** for very large datasets (> 100k points)

---

## Advanced Usage

### Multi-pattern Search

```typescript
// Add multiple patterns
for (known_pattern in known_patterns) {
  engine.addPattern(known_pattern.start, known_pattern.end);
}

// Detections will identify which pattern matched
for (t, det in engine.state?.detections?) {
  var pattern_name = pattern_names[det.best_pattern];
  println("Matched pattern: ${pattern_name}");
}
```

### Sliding Window Detection

```typescript
// For real-time detection, incrementally add data
var window_size = 1_h; // 1 hour window

for (new_data_point in stream) {
  sensor_data.setAt(new_data_point.time, new_data_point.value);

  if (should_detect()) {
    engine.resetScores();
    engine.computeScores(null);
    engine.detect(sensitivity, null);

    // Check latest detection
    var latest = engine.state?.detections?.last();
    if (latest != null) {
      handle_detection(latest);
    }
  }
}
```

### Custom Sensitivity Tuning

```typescript
// Experiment with thresholds
for (threshold in [0.7, 0.8, 0.9]) {
  engine.resetDetections();
  engine.detect(
    PatternDetectionSensitivity { threshold: threshold, overlap: 0.2 },
    null
  );

  println("Threshold ${threshold}: ${engine.state?.detections?.size()} detections");
}
```

---

## Utility Types

### DistanceMetrics

Static utility for computing distance between tensor representations of patterns.

```typescript
// Euclidean distance between two tensors
var dist = DistanceMetrics::euclidean(tensor_a, tensor_b);

// Dynamic Time Warping distance (requires pre-allocated workspace tensor)
var workspace = Tensor {};
workspace.init(TensorType::f64, [tensor_a.size(), tensor_b.size()]);
var dtw_dist = DistanceMetrics::dtw(tensor_a, tensor_b, workspace);
```

**Methods:**
- `euclidean(t1: Tensor, t2: Tensor): float` - Euclidean distance between two tensors
- `dtw(t1: Tensor, t2: Tensor, workspace: Tensor): float` - Dynamic Time Warping distance (workspace tensor must be pre-allocated)

### PatternDetectionEngine::windowToTensor (static)

Converts pattern and time window tables into a tensor for distance computation.

```typescript
PatternDetectionEngine::windowToTensor(pattern_table, window_table, output_tensor, MatchingNormalisation::as_is);
```

**Signature:** `static fn windowToTensor(pattern: Table<any>, timeWindow: Table<any>, tensor: Tensor, matchingNormalisation: MatchingNormalisation)`

---

## Performance Tips

### For Large Datasets
```typescript
// Use file-based scoring
var scores_path = engine.computeScores("/tmp/scores.gcb");
var detections_path = engine.detect(sensitivity, "/tmp/detections.gcb");
```

### Memory Management
```typescript
// Clear old detections
engine.resetDetections();

// Limit pattern count
// More patterns = longer computation
```

### Sampling
```typescript
// Downsample for faster detection
engine.samplingPolicy = SamplingPolicy::average_frequency;
```

## See Also

- [Transforms (transforms.md)](transforms.md) - FFT operations
- [README.md](README.md) - Library overview
