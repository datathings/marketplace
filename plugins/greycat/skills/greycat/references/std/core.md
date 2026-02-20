# [std](index.html) > core

[Source](core.source.html)

Core primitive types, data structures, and fundamental operations for GreyCat. This module provides the foundation for all GreyCat applications including primitives (int, float, bool), collection types (Array, Map, Table), graph nodes (node, nodeTime, nodeIndex, nodeList, nodeGeo), and mathematical/temporal utilities.

## Primitive Types

### any
Wildcard meta type for storing any value, including primitives. Used when type flexibility is needed.

### null
Meta type representing null placeholders.

### type
Meta type for reflection and introspection. Provides access to type metadata, fields, and enum values.

```gcl
// Get type information
var my_type = type::of(some_object);
var field_count = my_type.nb_fields();
var fields = my_type.fields();

// Access enum by name or offset
var error_code = type::enum_by_name<ErrorCode>(typeof ErrorCode, "runtime_error");
var code_name = type::enum_name(ErrorCode::timeout); // "timeout"
var code_offset = type::enum_offset(ErrorCode::timeout); // 6
```

### field
Descriptor of a type field for meta programming.

```gcl
// Inspect field metadata
var field_info = my_type.field_by_name("username");
var field_name = field_info?.name();
var field_type = field_info?.type();
var is_nullable = field_info?.is_nullable();
```

### bool
Boolean type with `true` or `false` literals.

### char
Character type supporting ASCII and UTF-8 codepoints.

### int
Signed 64-bit integer (-9223372036854775808 to 9223372036854775807).

```gcl
// Static bounds
Assert::equals(int::min, -9223372036854775808);
Assert::equals(int::max, 9223372036854775807);

// Format with thousand separators
var formatted = int::to_string(1000000, ','); // "1,000,000"
```

### float
64-bit floating point (-1.7976931348623157e+308 to 1.7976931348623157e+308).

```gcl
// Format float with precision
var formatted = float::to_string(
    1234.5678,
    '.', // decimal separator
    ',', // thousand separator (optional)
    2,   // max digits after decimal
    false // e-notation disabled
); // "1,234.57"
```

### FloatPrecision
Enum defining compression precision levels for float values (p1 to p10000000000).

```gcl
var precision = FloatPrecision::p1000; // 0.001 precision
```

### String
Immutable UTF-8 string type with comprehensive manipulation methods.

```gcl
// String comparison and searching
var text = "Hello World";
Assert::isTrue(text.startsWith("Hello"));
Assert::isTrue(text.endsWith("World"));
Assert::isTrue(text.contains("lo Wo"));
Assert::equals(text.indexOf('W'), 6);
Assert::equals(text.indexOfString("World"), 6);

// String transformation
var lower = text.lowercase(); // "hello world"
var upper = text.uppercase(); // "HELLO WORLD"
var trimmed = "  text  ".trim(); // "text"
var replaced = text.replace("World", "GreyCat"); // "Hello GreyCat"

// Slicing and splitting
var part = text.slice(0, 5); // "Hello"
var words = text.split(' '); // ["Hello", "World"]

// String similarity metrics
var similarity = "hello".jaro("hallo"); // Jaro distance
var jw_similarity = "hello".jarowinkler("hallo"); // Jaro-Winkler
var distance = "hello".levenshtein("hallo"); // Levenshtein distance

// Normalization
var normalized = "Café".nfkd_casefold(); // NFD normalized, casefolded

// Character decomposition
var chars = "Hello".chars(); // Chars { codepoints: ['H', 'e', 'l', 'l', 'o'] }
var back = chars.to_string(); // "Hello"
```

### Chars
Character decomposition of a String, providing access to individual codepoints.

```gcl
type Chars {
    codepoints: Array<char>;
}

var chars = "Hello".chars();
var codepoints = chars.codepoints; // Array<char>
var original = chars.to_string(); // "Hello"
```

### Buffer
Binary blob manipulation for building strings efficiently.

```gcl
var buffer = Buffer {};
buffer.add("Hello");
buffer.add(" ");
buffer.add("World");
Assert::equals(buffer.toString(), "Hello World");
Assert::equals(buffer.size(), 11);
buffer.clear();
```

## Collections

### Array<T>
Generic ordered container with sorting, searching, and manipulation methods.

```gcl
// Initialize and populate
var numbers = Array<int> {};
numbers.fill(5, 0); // [0, 0, 0, 0, 0]
numbers.set(0, 10);
numbers.add(20);
numbers.add_all([30, 40, 50]);

// Accessing and searching
Assert::equals(numbers.get(0), 10);
Assert::equals(numbers.size(), 8);
Assert::equals(numbers.index_of(20), 1);

// Manipulation
numbers.swap(0, 1);
numbers.sort(SortOrder::asc);
numbers.remove(0);
numbers.remove_first();
numbers.remove_last();

// Capacity management
numbers.set_capacity(100);

// Range comparison
Assert::isTrue(numbers.range_equals(0, 2, 30));
```

### Map<K, V>
Key-value store with iteration support.

```gcl
var map = Map<String, int> {};
map.set("one", 1);
map.set("two", 2);

Assert::equals(map.get("one"), 1);
Assert::isTrue(map.contains("two"));
Assert::equals(map.size(), 2);

var values = map.values(); // [1, 2]
map.remove("one");
```

### Table<T>
Two-dimensional data structure with column-based operations.

```gcl
// Create and populate table
var table = Table<Person> {};
table.init(10, 5); // pre-allocate capacity

// Set individual cells
table.set_cell(0, 0, "John");
table.set_cell(0, 1, 30);

// Set entire rows from objects
table.set_row(0, Person { name: "Alice", age: 25 });
table.add_row(Person { name: "Bob", age: 35 });

// Access data
var cell_value = table.get_cell(0, 0);
var row_obj = table.get_row(0);

// Sorting
table.sort(1, SortOrder::desc); // sort by column 1 (age)
table.sort_by(0, person_name_field, SortOrder::asc);

// Metadata
Assert::equals(table.cols(), 5);
Assert::equals(table.rows(), 2);

// Remove rows
table.remove_row(0);

// Apply column mappings for data transformation
var mapped = Table::applyMappings(table, mappings);
```

### Tuple<T, U>
Simple two-element association structure.

```gcl
var pair = Tuple<String, int> { x: "answer", y: 42 };
Assert::equals(pair.x, "answer");
Assert::equals(pair.y, 42);
```

## Graph Node Types

### node<T>
Persistent singleton value stored in the graph.

```gcl
var user_node: node<User>;
user_node.set(User { name: "Alice", email: "alice@example.com" });
var user = user_node.resolve();
Assert::equals(user.name, "Alice");

// Batch resolution
var users = node::resolve_all([node1, node2, node3]);
```

### nodeTime<T>
Time-series data structure with temporal indexing.

```gcl
var temperature: nodeTime<float>;

// Add timestamped values
temperature.setAt('2025-01-01T00:00:00Z', 20.5);
temperature.setAt('2025-01-01T01:00:00Z', 21.0);
temperature.setAt('2025-01-01T02:00:00Z', 21.5);

// Batch insert with delta
temperature.setAll('2025-01-02T00:00:00Z', [18.0, 19.0, 20.0], 1h);

// Time-based queries
var current = temperature.resolve(); // latest value
var at_time = temperature.resolveAt('2025-01-01T00:30:00Z'); // closest previous
var within = temperature.resolveAtWithin('2025-01-01T00:30:00Z', 1h); // closest previous within max dephase
var resolved_time = temperature.resolveTimeAt('2025-01-01T00:30:00Z'); // closest previous timestamp
var exact = temperature.getAt('2025-01-01T01:00:00Z');
var time_value = temperature.resolveTimeValueAt('2025-01-01T00:30:00Z');

// Navigation
var first_time = temperature.firstTime();
var last_time = temperature.lastTime();
var first_val = temperature.first();
var last_val = temperature.last();
var prev_time = temperature.prev('2025-01-01T01:00:00Z');
var next_time = temperature.next('2025-01-01T01:00:00Z');

// Range operations
var count = temperature.rangeSize('2025-01-01T00:00:00Z', '2025-01-01T23:59:59Z');
var total_size = temperature.size();

// Removal
temperature.removeAt('2025-01-01T00:00:00Z');
temperature.removeAll();

// Sampling (for visualization/analysis)
var sample = nodeTime::sample(
    [temperature],
    '2025-01-01T00:00:00Z',
    '2025-01-01T23:59:59Z',
    1000, // max rows
    SamplingMode::adaptative,
    1min, // max dephasing
    TimeZone::UTC
);

// Get metadata
var info = nodeTime::info([temperature]);
```

### nodeTimeCursor<T>
Iterator for walking through nodeTime series.

```gcl
var cursor = nodeTimeCursor<float> { n: temperature };
cursor.first();
while (cursor.currentTime() != null) {
    var t = cursor.currentTime();
    var v = cursor.current();
    println("${t}: ${v}");
    cursor.next();
}

// Advanced navigation
cursor.lessOrEq('2025-01-01T12:00:00Z');
cursor.skip_values(10);
cursor.skip_duration(1h);
```

### nodeIndex<K, V>
Persistent key-value store with O(log n) retrieval.

```gcl
var user_index: nodeIndex<String, User>;

// Set and get
user_index.set("alice", User { name: "Alice", age: 30 });
user_index.set("bob", User { name: "Bob", age: 25 });

var user = user_index.get("alice");
Assert::equals(user?.name, "Alice");

// Searching
var results = user_index.search("ali", 10); // fuzzy search, max 10 results
var closest = nodeIndex::search_closest(user_index, "alex", 5);

// Management
Assert::equals(user_index.size(), 2);
user_index.remove("alice");
user_index.removeAll();

// Sampling
var sample = nodeIndex::sample([user_index], "a", 100, SamplingMode::dense);

// Metadata
var info = nodeIndex::info([user_index]);
```

### nodeList<T>
Sparse list with efficient large-scale storage.

```gcl
var measurements: nodeList<float>;

// Add and set values
measurements.add(10.5); // append
measurements.add(20.5);
measurements.set(100, 99.9); // sparse indexing

// Access
var val = measurements.get(0);
var resolved = measurements.resolve(50); // closest previous if not found
var entry = measurements.resolveEntry(50); // returns Tuple<int, T>

// Navigation
var first_idx = measurements.firstIndex();
var last_idx = measurements.lastIndex();
var first_val = measurements.first();
var last_val = measurements.last();

// Range operations
var count = measurements.rangeSize(0, 100);
Assert::equals(measurements.size(), 3); // sparse: only 3 values set

// Removal
measurements.remove(100);
measurements.removeAll();

// Sampling
var sample = nodeList::sample(
    [measurements],
    0,
    1000,
    500, // max rows
    SamplingMode::fixed,
    10 // max dephasing
);

// Metadata
var info = nodeList::info([measurements]);
```

### nodeGeo<T>
Geographic indexed values with spatial queries.

```gcl
var poi: nodeGeo<String>; // points of interest

// Add locations
poi.set(geo{48.8566, 2.3522}, "Eiffel Tower");
poi.set(geo{51.5074, -0.1278}, "Big Ben");

// Query by location
var location = poi.get(geo{48.8566, 2.3522});
var resolved = poi.resolve(geo{48.86, 2.35}); // nearest

// Range queries
var box_count = poi.rangeSize(
    geo{48.0, 2.0}, // SW corner
    geo{49.0, 3.0}  // NE corner
);

// Navigation
var sw_point = poi.firstIndex(); // most south-west
var ne_point = poi.lastIndex(); // most north-east

// Management
poi.remove(geo{48.8566, 2.3522});
poi.removeAll();

// Sampling
var sample = nodeGeo::sample(
    [poi],
    geo{48.0, 2.0},
    geo{49.0, 3.0},
    1000,
    SamplingMode::dense
);
```

## Tensor & Numerical Types

### Tensor
Multi-dimensional array for numerical computations.

```gcl
// Create tensor
var tensor = Tensor {};
tensor.init(TensorType::f64, [3, 4]); // 3x4 matrix of f64

// Set and get values
tensor.set([0, 0], 1.5);
tensor.set([0, 1], 2.5);
var value = tensor.get([0, 0]);

// Complex tensors
tensor.setImag([0, 0], 0.5); // imaginary part
var imag_val = tensor.getImag([0, 0]);

// Appending values
tensor.append(42.0); // append value (1D), array (2D), or tensor (general)

// Operations
tensor.add([0, 0], 0.5); // add to existing value
tensor.fill(0.0); // fill all with value
var total = tensor.sum();

// Iteration
var pos = tensor.initPos();
while (tensor.incPos(pos)) {
    var val = tensor.get(pos);
    // process value
}

// Transformations
var complex_t = tensor.to_complex_tensor();
var real_part = complex_t.get_real_tensor();
var imag_part = var_imaginary_tensor();
var absolute = complex_t.get_absolute_tensor();
var phase = complex_t.get_phase_tensor(false); // radians

// Capacity management
tensor.setCapacity(1000); // pre-allocate capacity to avoid repeated resizes from append

// Reshaping and slicing
tensor.reshape([12]); // reshape to 1D
var slice = tensor.slice([0, 0], [2, 2]); // 2x2 sub-tensor
tensor.slide(5); // slide tensor 5 steps across the first dimension (useful for sliding windows, sequences, LSTM, Fourier)

// Copying elements between tensors
tensor.copyElementsFrom(src_tensor, 0, 5, 10); // copy 5 elements from src starting at pos 0, into tensor at pos 10

// Reset
tensor.reset(); // delete all tensor attributes and data

// Scaling
var scaled = tensor.scale(2.0, 1.0); // scale by 2.0

// Distance calculation
var dist = tensor.distance(other_tensor, TensorDistance::euclidean);
var l2_dist = tensor.distance(other_tensor, TensorDistance::l2sq);
var cos_dist = tensor.distance(other_tensor, TensorDistance::cosine);

// Utilities
Assert::equals(tensor.size(), 12);
Assert::equals(tensor.dim(), 2);
var shape = tensor.shape(); // [3, 4]
var tensor_type = tensor.type();

// Conversion
var as_table = tensor.toTable(); // 1D or 2D only
var as_string = tensor.toString();

// Static constructors
var vec = Tensor::wrap_1d(TensorType::f64, [1.0, 2.0, 3.0]);
```

### TensorType
Enum defining tensor data types: i32, i64, f32, f64, c64, c128.

### TensorDistance
Distance metrics: euclidean (l2), l2sq, cosine.

### VectorIndex<T>
High-performance vector similarity search using HNSW algorithm.

```gcl
var index = VectorIndex<String> {
    distance: TensorDistance::cosine,
    rng: Random { seed: 42 }
};

// Index vectors
var vec1 = Tensor::wrap_1d(TensorType::f32, [0.1, 0.2, 0.3]);
var vec2 = Tensor::wrap_1d(TensorType::f32, [0.4, 0.5, 0.6]);

index.add(vec1_node, "item1");
index.add(vec2_node, "item2");

// Search for similar vectors
var query = Tensor::wrap_1d(TensorType::f32, [0.15, 0.25, 0.35]);
var results = index.search(query, 10); // top 10 results

for (_, result in results) {
    println("${result.value}: distance=${result.distance}");
}

Assert::equals(index.size(), 2);
```

## Geographic Types

### geo
Latitude/longitude coordinates with distance calculations.

```gcl
// Create geo point
var paris = geo{48.8566, 2.3522};
var london = geo{51.5074, -0.1278};

// Access components
var latitude = paris.lat();
var longitude = paris.lng();

// Distance in meters
var distance = paris.distance(london); // ~344,000 meters

// String representations
var coords = paris.toString(); // "48.8566,2.3522"
var geohash = paris.toGeohash(); // GeoHash encoding

// Bounds
Assert::equals(geo::min, geo{-85.0511287602, -179.9999999581});
Assert::equals(geo::max, geo{85.0511287602, 179.9999999581});
```

### GeoCircle
Circular geographic region.

```gcl
var circle = GeoCircle {
    center: geo{48.8566, 2.3522},
    radius: 10000.0 // 10km radius
};

var point = geo{48.86, 2.35};
Assert::isTrue(circle.contains(point));

// Bounding box
var sw = circle.sw();
var ne = circle.ne();
```

### GeoBox
Rectangular geographic region.

```gcl
var box = GeoBox {
    sw: geo{48.8, 2.3},
    ne: geo{48.9, 2.4}
};

Assert::isTrue(box.contains(geo{48.85, 2.35}));
Assert::isFalse(box.contains(geo{49.0, 2.5}));
```

### GeoPoly
Polygonal geographic region.

```gcl
var polygon = GeoPoly {
    points: [
        geo{48.8, 2.3},
        geo{48.9, 2.3},
        geo{48.9, 2.4},
        geo{48.8, 2.4}
    ]
};

Assert::isTrue(polygon.contains(geo{48.85, 2.35}));
var sw = polygon.sw();
var ne = polygon.ne();
```

## Time & Duration

### time
Absolute moment in time with calendar operations.

```gcl
// Create time
var now = time::now();
var specific = time::new(1704067200, DurationUnit::seconds);
var parsed = time::parse("2025-01-01T00:00:00Z", null); // ISO 8601

// Arithmetic with durations
var later = now + 1h;
var earlier = now - 30min;

// Calendar operations
var next_day = now.calendar_add(1, CalendarUnit::day, TimeZone::UTC);
var next_month = now.calendar_add(1, CalendarUnit::month, null);
var start_of_day = now.calendar_floor(CalendarUnit::day, null);
var end_of_day = now.calendar_ceiling(CalendarUnit::day, null);
var week_start = now.startOfWeek(null);
var week_end = now.endOfWeek(null);

// Calendar queries
var day_of_year = now.dayOfYear(null); // 0-365
var day_of_week = now.dayOfWeek(null); // 0 (Sunday) - 6 (Saturday)
var week_of_year = now.weekOfYear(null); // 1-53
var days_in_month = now.daysInMonth(null);

Assert::isTrue(time::isLeap(2024));
Assert::equals(time::totalDaysInYear(2024), 366);
Assert::equals(time::totalDaysInMonth(2, 2024), 29);

// Conversion
var epoch_millis = now.to(DurationUnit::milliseconds);
var floored = now.floor(DurationUnit::hours);

// Formatting
var formatted = now.format("%Y-%m-%d %H:%M:%S", TimeZone::UTC);
var iso = now.format("%+", null); // ISO 8601

// Date conversion
var date = now.toDate(TimeZone::UTC);

// Bounds
Assert::equals(time::min, -9223372036854775808_time);
Assert::equals(time::max, 9223372036854775807_time);

// Context
var context_time = time::current(); // contextual time (defaults to time::min)
```

### duration
Time interval with flexible units.

```gcl
// Create durations
var one_hour = duration::new(1, DurationUnit::hours);
var thirty_mins = duration::new(30, DurationUnit::minutes);
var precise = duration::newf(1.5, DurationUnit::hours);

// Conversion
var in_minutes = one_hour.to(DurationUnit::minutes); // 60
var in_seconds = one_hour.to(DurationUnit::seconds); // 3600
var in_hours_float = precise.tof(DurationUnit::hours); // 1.5

// Arithmetic
var extended = one_hour.add(30, DurationUnit::minutes); // 90 minutes
var reduced = one_hour.subtract(15, DurationUnit::minutes); // 45 minutes

// Literal syntax
var d1 = 1h;
var d2 = 30min;
var d3 = 5s;
```

### Date
Calendar date with timezone awareness.

```gcl
// Create dates
var date = Date {
    year: 2025,
    month: 1,
    day: 1,
    hour: 12,
    minute: 30,
    second: 0,
    microsecond: 0
};

var from_time = Date::from_time(time::now(), TimeZone::UTC);
var parsed_date = Date::parse("2025-01-01", "%Y-%m-%d");

// Convert to time
var as_time = date.to_time(TimeZone::UTC);
var nearest = date.to_nearest_time(TimeZone::"Europe/Paris"); // handles DST
```

### DurationUnit
Enum: microseconds, milliseconds, seconds, minutes, hours, days.

### CalendarUnit
Enum: year, month, day, hour, minute, second, microsecond.

### TimeZone
Comprehensive timezone enumeration covering all IANA timezones.

```gcl
var paris_time = now.toDate(TimeZone::"Europe/Paris");
var ny_time = now.toDate(TimeZone::"America/New_York");
var utc_time = now.toDate(TimeZone::UTC);
```

## Enumerations

### ErrorCode
Runtime error codes: none(0), interrupted(1), await(2), timeout(6), forbidden(7), runtime_error(8).

### SamplingMode
Data sampling strategies:
- fixed(0): Fixed delta between indices
- fixed_reg(1): Fixed with linear regression interpolation
- adaptative(2): Adaptive sampling ensuring minimum skip
- dense(3): All elements in range

```gcl
var sample = nodeTime::sample(
    [timeseries],
    from_time,
    to_time,
    1000,
    SamplingMode::adaptative,
    max_dephase,
    tz
);
```

### SortOrder
Sorting direction: asc, desc.

```gcl
array.sort(SortOrder::desc);
table.sort(column_idx, SortOrder::asc);
```

## Primitive Tuples

Compact primitive types for efficient storage and indexing.

### t2, t3, t4
Integer tuples (2-4 dimensions).

```gcl
var point2d: t2; // stores 2 ints as single primitive
var x = point2d.x0();
var y = point2d.x1();

var point3d: t3;
var z = point3d.x2();

var point4d: t4;
var w = point4d.x3();
```

### t2f, t3f, t4f
Float tuples (2-4 dimensions).

```gcl
var vector: t3f;
var x = vector.x0();
var y = vector.x1();
var z = vector.x2();
```

### str
Compact string (max 10 lowercase ASCII chars) stored as primitive. Efficient for indexing.

```gcl
var code: str = "product123"; // error: too long
var valid: str = "item42"; // OK
```

## Error Handling

### Error
Error type with stack traces.

```gcl
try {
    // risky operation
    throw Error { message: "Something went wrong" };
} catch (e: Error) {
    println("Error: ${e.message}");
    for (_, frame in e.stack) {
        println("  at ${frame.function} (${frame.module}:${frame.line}:${frame.column})");
    }
}
```

### ErrorFrame
Stack frame information: module, function, line, column.

## Utility Functions

### Mathematical Functions

```gcl
// Exponential and logarithmic
var e_x = exp(2.0);
var natural_log = log(10.0);
var log_base2 = log2(8.0);
var log_base10 = log10(100.0);
var power = pow(2.0, 8.0); // 256.0

// Trigonometric
var sine = sin(MathConstants::pi / 2); // 1.0
var cosine = cos(0.0); // 1.0
var tangent = tan(MathConstants::pi / 4); // 1.0

// Inverse trigonometric
var arc_sine = asin(1.0); // π/2
var arc_cosine = acos(0.0); // π/2
var arc_tangent = atan(1.0); // π/4

// Hyperbolic
var sinh_val = sinh(1.0);
var cosh_val = cosh(1.0);
var tanh_val = tanh(1.0);

// Rounding
var floored = floor(3.7); // 3.0
var ceiled = ceil(3.2); // 4.0
var rounded = round(3.5); // 4.0
var truncated = trunc(3.9); // 3.0
var precise = roundp(3.14159, 2); // 3.14

// Other
var square_root = sqrt(16.0); // 4.0
var absolute = abs(-5); // 5
var minimum = min(10, 20); // 10
var maximum = max(10, 20); // 20

Assert::isTrue(isNaN(0.0 / 0.0));
```

### MathConstants
Mathematical constants: e, log_2e, log_10e, ln2, ln10, pi, pi_2, pi_4, m1_pi, m2_pi, m2_sqrt_pi, sqrt2, sqrt1_2.

```gcl
var circle_area = MathConstants::pi * r * r;
var half_pi = MathConstants::pi_2;
```

### Other Utility Functions

```gcl
// Parsing
var number = parseNumber("42"); // int or float
var hex_val = parseHex("FF"); // 255

// Cloning
var copy = clone(original_object);

// Enum value access
var code_value = valueOf(ErrorCode::timeout); // 6

// Printing
println("Hello World"); // with newline
print(my_object); // pretty print with formatting
pprint("text"); // print without newline

// Logging
error("Critical error occurred");
warn("This is a warning");
info("Informational message");
perf("Performance metric");
trace("Detailed trace information");
```

## Supporting Types

### NodeInfo<T>
Statistics about graph nodes.

```gcl
type NodeInfo<T> {
    size: int;    // number of elements
    from: T?;     // first value (inclusive)
    to: T?;       // last value (inclusive)
}

var info = nodeTime::info([timeseries]);
println("Size: ${info[0].size}, From: ${info[0].from}, To: ${info[0].to}");
```

### SearchResult<K, V>
Result from vector/index search operations.

```gcl
var results = vector_index.search(query_vector, 10);
for (_, result in results) {
    println("Key: ${result.key}, Value: ${result.value}, Distance: ${result.distance}");
}
```

### TableColumnMapping
Describes column extractors for table transformations.

```gcl
var mapping = TableColumnMapping {
    column: 0,
    extractors: ["*", "name"] // resolve node, then get 'name' field
};
var transformed = Table::applyMappings(original_table, [mapping]);
```
