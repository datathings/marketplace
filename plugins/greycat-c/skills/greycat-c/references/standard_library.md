# GreyCat Standard Library (std)

## Overview

The GreyCat Standard Library provides essential data structures, I/O operations, runtime features, and utilities for GCL applications. The library is organized into four modules:

- **core** - Fundamental types and data structures
- **runtime** - Scheduler, tasks, logging, security, system operations
- **io** - File I/O, CSV/JSON/XML, HTTP client, email
- **util** - Collections, statistics, quantizers, cryptography

## Table of Contents

### Modules
- [std::core - Core Types](#core-module-stdcore)
  - [Fundamental Types](#fundamental-types) - Tuple, Date, Error
  - [Geospatial Types](#geospatial-types) - GeoCircle, GeoPoly, GeoBox
  - [Enumerations](#enumerations) - FloatPrecision, CalendarUnit, TensorType, etc.

- [std::runtime - Runtime Features](#runtime-module-stdruntime)
  - [Scheduler & Task Automation](#scheduler---task-automation)
  - [Task & Job Management](#task--job-management)
  - [Logging](#logging)
  - [System Information](#system-information)
  - [Security](#security) - User, UserGroup, SecurityPolicy, OpenIDConnect
  - [License Management](#license-management)
  - [OpenAPI Integration](#openapi-integration)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)

- [std::io - Input/Output](#io-module-stdio)
  - [Binary I/O](#binary-io) - GcbWriter, GcbReader
  - [Text I/O](#text-io) - TextWriter, TextReader
  - [JSON I/O](#json-io) - JsonWriter, JsonReader
  - [CSV I/O](#csv-io) - CsvWriter, CsvReader, CsvAnalyzer
  - [File System](#file-system) - File, FileWalker
  - [HTTP Client](#http-client) - Http, Url
  - [Email](#email) - Email, Smtp
  - [XML I/O](#xml-io) - XmlReader

- [std::util - Utilities](#util-module-stdutil)
  - [Collections](#collections) - Queue, Stack, SlidingWindow, TimeWindow
  - [Statistics](#statistics) - Gaussian, Histogram, GaussianProfile
  - [Quantizers](#quantizers) - Linear, Log, Custom, Multi
  - [Utilities](#utilities) - Random, Assert, ProgressTracker, Crypto, Plot

### Additional Sections
- [Usage Guidelines](#usage-guidelines)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

---

## Core Module (std::core)

### Fundamental Types

#### Tuple<T, U>
Two-element tuple for pairing values of potentially different types.
```gcl
var pair = Tuple<String, int> { first: "count", second: 42 };
println("${pair.first}: ${pair.second}");
```

#### Date
Date/time representation with calendar operations.
```gcl
var now = Date::now();
var yesterday = now - 1day;
var next_week = now + 7day;
```

#### Error & ErrorFrame
Error handling with stack traces.
```gcl
type Error {
  code: ErrorCode;
  message: String;
  stack: Array<ErrorFrame>;
}
```

### Geospatial Types

#### GeoCircle
Circular geographic region defined by center point and radius.
```gcl
var zone = GeoCircle {
  lat: 45.5,
  lng: -73.6,
  radius: 5000.0  // meters
};
```

#### GeoPoly
Polygon geographic region.
```gcl
var boundary = GeoPoly {
  points: [
    GeoPoint { lat: 45.5, lng: -73.6 },
    GeoPoint { lat: 45.6, lng: -73.5 },
    GeoPoint { lat: 45.4, lng: -73.4 }
  ]
};
```

#### GeoBox
Rectangular bounding box for geographic queries.
```gcl
var bbox = GeoBox {
  south_west: GeoPoint { lat: 45.4, lng: -73.7 },
  north_east: GeoPoint { lat: 45.6, lng: -73.5 }
};
```

### Enumerations

#### FloatPrecision
Precision levels for floating-point calculations: `p1`, `p10`, `p100`, `p1000`, etc.

#### CalendarUnit
Time units for date operations: `year`, `month`, `week`, `day`, `hour`, `minute`, `second`.

#### DurationUnit
Duration units: `us` (microseconds), `ms`, `s`, `min`, `hour`, `day`.

#### TensorType
Tensor data types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `c64`, `c128`.

#### TensorDistance
Distance metrics for tensors: `euclidean`, `cosine`, `manhattan`, `hamming`.

## Runtime Module (std::runtime)

### Scheduler - Task Automation

The scheduler manages recurring tasks with various periodicity options.

#### Basic Usage
```gcl
fn backup_database() {
  // Backup logic...
}

fn schedule_backups() {
  Scheduler::add(
    backup_database,
    DailyPeriodicity { hour: 2 },  // Run at 2 AM daily
    null
  );
}
```

#### Periodicity Types

**FixedPeriodicity** - Run every N time units
```gcl
FixedPeriodicity { every: 5min }
FixedPeriodicity { every: 1hour }
```

**DailyPeriodicity** - Run at specific time each day
```gcl
DailyPeriodicity { hour: 14, minute: 30 }  // 2:30 PM daily
```

**WeeklyPeriodicity** - Run on specific day/time each week
```gcl
WeeklyPeriodicity {
  day: DayOfWeek::monday,
  hour: 9,
  minute: 0
}
```

**MonthlyPeriodicity** - Run on specific day/time each month
```gcl
MonthlyPeriodicity { day: 1, hour: 0, minute: 0 }  // First of month
```

**YearlyPeriodicity** - Run on specific date/time each year
```gcl
YearlyPeriodicity {
  month: Month::january,
  day: 1,
  hour: 0
}
```

#### Advanced Scheduling
```gcl
fn health_check() {
  // Check system health...
}

Scheduler::add(
  health_check,
  FixedPeriodicity { every: 5min },
  PeriodicOptions {
    start: time::now() + 1hour,    // Start in 1 hour
    max_duration: 30s,              // Timeout after 30 seconds
  }
);
```

#### Managing Tasks
```gcl
// Find task
var ptask = Scheduler::find(health_check);
if (ptask != null) {
  println("Runs every ${ptask.periodicity.every}");
}

// Control execution
Scheduler::deactivate(backup_database);  // Pause
Scheduler::activate(backup_database);    // Resume
Scheduler::remove(backup_database);      // Delete

// List all tasks
var all_tasks = Scheduler::list();
for (_, task in all_tasks) {
  println("${task.function}: active=${task.is_active}");
}
```

### Task & Job Management

#### Task
Asynchronous task execution with status tracking.
```gcl
enum TaskStatus { running, completed, failed, cancelled }

type Task {
  id: String;
  status: TaskStatus;
  progress: float?;
  error: Error?;
}
```

#### Job<T>
Generic job container for asynchronous operations.
```gcl
var job = Job<Result> {};
// Submit for processing...
```

### Logging

#### Log
Structured logging with severity levels.
```gcl
enum LogLevel { trace, debug, info, warn, error, fatal }

Log::info("Application started");
Log::error("Failed to connect: ${error_message}");
Log::debug("Processing item ${id}");
```

#### LogDataUsage
Track data usage metrics.
```gcl
var usage = LogDataUsage {
  bytes_read: 1024000,
  bytes_written: 512000,
  timestamp: time::now()
};
```

### System Information

#### Runtime
Query runtime environment information.
```gcl
var info = RuntimeInfo {};
println("Platform: ${info.platform}");
println("Version: ${info.version}");
```

#### System
System-level operations and utilities.
```gcl
var sys_info = System::info();
println("CPU cores: ${sys_info.cpu_count}");
```

#### ChildProcess
Execute external processes.
```gcl
type ChildProcess {
  command: String;
  args: Array<String>?;
  env: Map<String, String>?;
}

type ChildProcessResult {
  exit_code: int;
  stdout: String;
  stderr: String;
}
```

### Security

#### User & UserGroup
User and group management.
```gcl
type User extends SecurityEntity {
  username: String;
  email: String;
  groups: Array<UserGroup>;
}

type UserGroup extends SecurityEntity {
  name: String;
  policies: Array<UserGroupPolicy>;
}
```

#### SecurityPolicy
Access control and permissions.
```gcl
type SecurityPolicy {
  resource: String;
  actions: Array<String>;
  allow: bool;
}
```

#### OpenIDConnect
OpenID Connect authentication.
```gcl
type OpenIDConnect {
  issuer: String;
  client_id: String;
  client_secret: String;
  redirect_uri: String;
}
```

### License Management

```gcl
enum LicenseType { trial, commercial, enterprise, opensource }

type License {
  type: LicenseType;
  valid_until: time;
  features: Array<String>;
}
```

### OpenAPI Integration

```gcl
type OpenApi {
  title: String;
  version: String;
  paths: Map<String, any>;
}
```

### Model Context Protocol (MCP)

Types for MCP server integration:
- `McpInitializeParams`, `McpInitializeResult`
- `McpServerCapabilities`, `McpClientCapabilities`
- `McpTool`, `McpToolsListParams`, `McpToolsCallParams`
- `McpTextContent`, `McpImageContent`, `McpAudioContent`
- `McpPriority`, `McpRole`, `McpContentType`

## I/O Module (std::io)

### Binary I/O

#### GcbWriter<T> / GcbReader<T>
Binary serialization using GreyCat's ABI encoding.
```gcl
// Write
var writer = GcbWriter<MyType> { path: "/data/output.gcb" };
writer.write(my_object);
writer.flush();

// Read
var reader = GcbReader<MyType> { path: "/data/output.gcb" };
while (reader.can_read()) {
  var obj = reader.read();
  process(obj);
}
```

### Text I/O

#### TextWriter<T> / TextReader
UTF-8 text file operations.
```gcl
// Write
var writer = TextWriter<String> { path: "/logs/output.txt" };
writer.writeln("Line 1");
writer.writeln("Line 2");
writer.flush();

// Read
var reader = TextReader { path: "/logs/output.txt" };
while (reader.can_read()) {
  var line = reader.read();
  println(line);
}
```

### JSON I/O

#### JsonWriter<T> / JsonReader<T>
JSON and NDJSON (newline-delimited) support.
```gcl
// Write NDJSON
var writer = JsonWriter<Person> { path: "/data/people.json" };
writer.writeln(person1);  // One JSON object per line
writer.writeln(person2);
writer.flush();

// Read NDJSON
var reader = JsonReader<Person> { path: "/data/people.json" };
while (reader.can_read()) {
  var person = reader.read();
  process(person);
}

// Parse JSON string
var json = Json<Person> {};
var obj = json.parse("{\"name\":\"Alice\",\"age\":30}");
```

### CSV I/O

#### CsvWriter<T> / CsvReader<T>
CSV operations with automatic header generation.
```gcl
// Configure format
var format = CsvFormat {
  header_lines: 1,
  separator: ',',
  string_delimiter: '"'
};

// Write
var writer = CsvWriter<Employee> {
  path: "/data/employees.csv",
  format: format
};
writer.write(emp1);  // Headers auto-generated from type
writer.write(emp2);
writer.flush();

// Read
var reader = CsvReader<Employee> {
  path: "/data/employees.csv",
  format: format
};
while (reader.can_read()) {
  var emp = reader.read();
  process(emp);
}
```

#### CSV Analysis
Analyze CSV structure and generate GCL types.
```gcl
var config = CsvAnalysisConfig {
  row_limit: 1000,
  enumerable_limit: 50
};

var files = Array<File> { File::open("/data/sales.csv")!! };
var stats = Csv::analyze(files, config);

// Generate type definitions
var type_code = Csv::generate(stats);
println(type_code);  // Generated GCL type definitions

// Sample data
var reader = CsvReader<any> { path: "/data/sales.csv" };
var sample = Csv::sample(reader, 100);  // First 100 rows
```

### File System

#### File
Comprehensive file operations.
```gcl
// Discovery
var csv_files = File::ls("/data", ".csv", true);  // Recursive

// Open and inspect
var file = File::open("/data/input.txt")!!;
println("Size: ${file.size}");
println("Extension: ${file.extension()!!}");
println("Is directory: ${file.isDir()}");
println("SHA-256: ${file.sha256()!!}");

// Operations
File::copy("/data/src.txt", "/data/dst.txt");
File::rename("/data/old.txt", "/data/new.txt");
File::delete("/data/temp.txt");
File::mkdir("/data/archive");

// Paths
var base = File::baseDir();
var user = File::userDir();
var work = File::workingDir();
```

#### FileWalker
Iterate through directory hierarchies.
```gcl
var walker = FileWalker { path: "/data" };
var file_count = 0;

while (!walker.isEmpty()) {
  var file = walker.next();
  if (file != null && !file.isDir()) {
    file_count++;
    println(file.path);
  }
}
```

### HTTP Client

#### Http<T>
REST API client with type-safe requests.
```gcl
var headers = [
  HttpHeader { name: "Authorization", value: "Bearer token" },
  HttpHeader { name: "Accept", value: "application/json" }
];

// GET request
var response = Http<User> {}.get(
  "https://api.example.com/users/123",
  headers
);

// Download file
Http<any> {}.getFile(
  "https://example.com/data.csv",
  "/local/data.csv",
  null
);

// POST request
var new_user = User { name: "Alice", email: "alice@example.com" };
var result = Http<User> {}.post(
  "https://api.example.com/users",
  new_user,
  headers
);
```

#### Url
URL parsing and manipulation.
```gcl
var url = Url::parse("https://api.example.com:8080/users?active=true#top");

println("Protocol: ${url.protocol}");  // "https"
println("Host: ${url.host}");          // "api.example.com"
println("Port: ${url.port}");          // 8080
println("Path: ${url.path}");          // "/users"
println("Param: ${url.params?.get("active")}");  // "true"
println("Hash: ${url.hash}");          // "top"
```

### Email

#### Email & Smtp
Email composition and SMTP delivery.
```gcl
var smtp = Smtp {
  host: "smtp.gmail.com",
  port: 587,
  mode: SmtpMode::starttls,
  authenticate: SmtpAuth::plain,
  user: "sender@example.com",
  pass: "app_password"
};

var email = Email {
  from: "sender@example.com",
  to: ["recipient@example.com"],
  cc: ["cc@example.com"],
  subject: "Monthly Report",
  body: "<h1>Report</h1><p>Content here...</p>",
  body_is_html: true,
  attachments: ["/reports/monthly.pdf"]
};

smtp.send(email);
```

### XML I/O

#### XmlReader<T>
XML parsing and deserialization.
```gcl
var reader = XmlReader<Config> { path: "/config/settings.xml" };
while (reader.can_read()) {
  var config = reader.read();
  apply_config(config);
}
```

## Util Module (std::util)

### Collections

#### Queue<T>
FIFO queue with optional capacity bounds.
```gcl
var queue = Queue<String> { capacity: 100 };
queue.push("first");
queue.push("second");

var item = queue.pop();       // "first"
var next = queue.front();     // Peek without removing
var last = queue.back();      // View last element
```

#### Stack<T>
LIFO stack.
```gcl
var stack = Stack<int> {};
stack.push(10);
stack.push(20);

var top = stack.pop();        // 20
var peek = stack.last();      // Peek at top
var bottom = stack.first();   // View bottom
```

#### SlidingWindow<T>
Fixed-size window with streaming statistics.
```gcl
var window = SlidingWindow<float> { span: 100 };

window.add(10.5);
window.add(20.3);
window.add(15.7);

var avg = window.avg()!!;
var std = window.std()!!;
var median = window.median()!!;
var min = window.min();
var max = window.max();
```

#### TimeWindow<T>
Time-based sliding window with automatic expiration.
```gcl
var window = TimeWindow<float> { span: 5min };

window.add(time::now(), temperature);
window.add(time::now() + 30s, next_temp);

// Statistics on recent data only
var avg = window.avg();
var min_tuple = window.min();  // Tuple<time, float>
var max_tuple = window.max();  // Tuple<time, float>
```

### Statistics

#### Gaussian<T>
Running statistical profile with normalization.
```gcl
var profile = Gaussian<float> {};

profile.add(10.0);
profile.add(20.0);
profile.add(30.0);

var avg = profile.avg()!!;        // 20.0
var std = profile.std()!!;
var min = profile.min;            // 10.0
var max = profile.max;            // 30.0

// Normalize: (value - min) / (max - min)
var norm = profile.normalize(15.0)!!;  // 0.25

// Standardize: (value - avg) / std
var z_score = profile.standardize(25.0);
```

#### Histogram<T>
Binned distribution analysis.
```gcl
var quantizer = LinearQuantizer<float> {
  min: 0.0,
  max: 100.0,
  bins: 20
};
var histogram = Histogram<float> { quantizer: quantizer };

for (_, score in test_scores) {
  histogram.add(score);
}

var median = histogram.percentile(0.5);      // 50th percentile
var p90 = histogram.percentile(0.9);         // 90th percentile
var below_60 = histogram.ratio_under(60.0);  // Fraction below 60

var stats = histogram.stats();  // Comprehensive statistics
```

#### GaussianProfile<T>
Multi-dimensional Gaussian statistics by category.
```gcl
var quantizer = LinearQuantizer<int> { min: 0, max: 100, bins: 10 };
var profile = GaussianProfile<int> {
  quantizer: quantizer,
  precision: FloatPrecision::p1000
};

profile.add(age_group, salary);
profile.add(age_group, another_salary);

var avg_salary = profile.avg(age_group);
var std_salary = profile.std(age_group);
```

### Quantizers

#### LinearQuantizer<T>
Uniform bin spacing.
```gcl
var linear = LinearQuantizer<float> {
  min: 0.0,
  max: 100.0,
  bins: 10
};

var bin = linear.quantize(25.0);  // Returns bin index (2)
var bounds = linear.bounds(2);    // QuantizerSlotBound
// bounds.min = 20.0, bounds.max = 30.0, bounds.center = 25.0
```

#### LogQuantizer<T>
Logarithmic bin spacing for exponential distributions.
```gcl
var log_quant = LogQuantizer<float> {
  min: 1.0,
  max: 1000.0,
  bins: 10
};
var bin = log_quant.quantize(50.0);
```

#### CustomQuantizer<T>
User-defined bin boundaries.
```gcl
var age_bins = CustomQuantizer<int> {
  min: 0,
  max: 100,
  step_starts: [0, 18, 25, 40, 65]  // Custom age groups
};
var group = age_bins.quantize(32);  // Returns appropriate bin
```

#### MultiQuantizer<T>
Multi-dimensional quantization.
```gcl
var quantizers = Array<Quantizer<float>> {
  LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 5 },
  LogQuantizer<float> { min: 1000.0, max: 200000.0, bins: 8 },
  LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 10 }
};

var multi = MultiQuantizer<float> { quantizers: quantizers };
var slot = multi.quantize([35.0, 45000.0, 87.5]);
var vector = multi.slot_vector(slot);  // [age_bin, income_bin, score_bin]
```

### Utilities

#### Random
Seeded random number generator.
```gcl
var rng = Random { seed: 12345 };

var dice = rng.uniform(1, 7);           // 1-6 inclusive
var prob = rng.uniformf(0.0, 1.0);      // Float [0.0, 1.0)

// Fill array with normal distribution
var samples = Array<float> {};
rng.fill(samples, 1000, 50.0, 60.0);    // 1000 samples, mean=50, std=10
```

#### Assert
Testing utilities.
```gcl
Assert::equals(actual, expected);
Assert::equalsd(pi, 3.14159, 0.001);    // Float with epsilon
Assert::equalst(tensor_a, tensor_b, 0.01);  // Tensor with epsilon
Assert::isTrue(condition);
Assert::isFalse(condition);
Assert::isNotNull(value);
Assert::isNull(value);
```

#### ProgressTracker
Monitor long-running operations.
```gcl
var tracker = ProgressTracker {
  start: time::now(),
  total: 10000
};

// Update progress
tracker.update(2500);
println("Progress: ${tracker.progress * 100}%");     // 25%
println("Speed: ${tracker.speed} items/sec");
println("Remaining: ${tracker.remaining}");

// Complete
tracker.update(10000);
println("Done! Progress: ${tracker.progress}");      // 1.0
```

#### Crypto
Cryptographic operations.
```gcl
var input = "sensitive data";

// Hashing
var sha1 = Crypto::sha1hex(input);
var sha256 = Crypto::sha256hex(input);

// Encoding
var b64 = Crypto::base64_encode(input);
var decoded = Crypto::base64_decode(b64);

var url_enc = Crypto::url_encode("param with spaces");
var url_dec = Crypto::url_decode(url_enc);

// PKCS1 signing (requires private key)
var signature = Crypto::pkcs1_sign(data, private_key);
```

#### Plot
Basic plotting from tabular data.
```gcl
var data = Table {};
data.set_row(0, ["Jan", 1, 10.5]);
data.set_row(1, ["Feb", 2, 15.3]);
data.set_row(2, ["Mar", 3, 20.1]);

// Plot column 1 (x-axis) vs columns 2+ (y-axis series)
Plot::scatter_plot(data, 1, [2], "output.png");
```

## Usage Guidelines

### When to Use Each Module

**core** - Use for fundamental data types, error handling, and geospatial operations.

**runtime** - Use for:
- Scheduled/recurring tasks (Scheduler)
- Background job processing (Task, Job)
- Application logging (Log)
- Security and authentication (User, SecurityPolicy, OpenIDConnect)
- System information queries (Runtime, System)

**io** - Use for:
- File I/O operations (File, FileWalker)
- Data serialization (GcbWriter, JsonWriter, CsvWriter)
- HTTP API calls (Http, Url)
- Email notifications (Email, Smtp)

**util** - Use for:
- Data structures (Queue, Stack, SlidingWindow)
- Statistical analysis (Gaussian, Histogram)
- Data binning (LinearQuantizer, LogQuantizer)
- Testing (Assert)
- Monitoring (ProgressTracker)

### Best Practices

1. **Resource Management**: Always call `flush()` on writers and close readers when done.
2. **Error Handling**: Check return values (especially `?` nullable types) before use.
3. **Scheduling**: Use appropriate periodicity type for your use case; `FixedPeriodicity` for simple intervals, `DailyPeriodicity` for scheduled times.
4. **CSV Analysis**: Use `Csv::analyze()` to understand data structure before processing large files.
5. **Statistics**: Use `Gaussian` for running statistics, `Histogram` for distribution analysis.
6. **File Operations**: Use `File::ls()` with file extensions for efficient discovery.
7. **HTTP**: Always include appropriate headers (Authorization, Content-Type) in requests.

### Common Patterns

#### Scheduled Data Processing
```gcl
fn process_daily_data() {
  var files = File::ls("/data/incoming", ".csv", false);
  for (_, file in files) {
    var reader = CsvReader<Record> { path: file.path };
    while (reader.can_read()) {
      process_record(reader.read());
    }
    File::rename(file.path, "/data/processed/${file.name}");
  }
}

Scheduler::add(
  process_daily_data,
  DailyPeriodicity { hour: 1, minute: 0 },
  null
);
```

#### Streaming Statistics
```gcl
var window = SlidingWindow<float> { span: 1000 };

fn process_sensor_data(readings: Array<float>) {
  for (_, value in readings) {
    window.add(value);

    if (window.size() >= 100) {
      var avg = window.avg()!!;
      var std = window.std()!!;

      if (value > avg + 3.0 * std) {
        Log::warn("Anomaly detected: ${value}");
      }
    }
  }
}
```

#### HTTP API Integration
```gcl
fn fetch_and_store_data() {
  var headers = [
    HttpHeader { name: "Authorization", value: "Bearer ${api_key}" }
  ];

  var response = Http<Array<Record>> {}.get(
    "https://api.example.com/data",
    headers
  );

  if (response != null) {
    var writer = JsonWriter<Record> { path: "/data/cache.json" };
    for (_, record in response) {
      writer.writeln(record);
    }
    writer.flush();
  }
}
```
