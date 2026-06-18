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
  - [Security & Identity](#security--identity) - Identity, IdentityGrant, IdentityGrantType
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
  - [S3 Object Storage](#s3-object-storage) - S3, S3Bucket, S3Object, S3BasicCredentials

- [std::util - Utilities](#util-module-stdutil)
  - [Collections](#collections) - Queue, Stack, SlidingWindow, TimeWindow
  - [Statistics](#statistics) - Gaussian, Histogram, GaussianProfile
  - [Quantizers](#quantizers) - Linear, Log, Custom, Multi
  - [Utilities](#utilities) - Random, Assert, ProgressTracker, Crypto, Uuid, Plot

### Additional Sections
- [Usage Guidelines](#usage-guidelines)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

---

## Core Module (std::core)

### Fundamental Types

#### Tuple<T, U>
Two-element tuple for pairing values of potentially different types. Fields are `x` and `y`.
```gcl
var pair = Tuple<String, int> { x: "count", y: 42 };
println("${pair.x}: ${pair.y}");
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
  message: String?;
  stack: Array<ErrorFrame>;
}

type ErrorFrame {
  module: String?;
  function: String;
  line: int;
  column: int;
}

// ErrorCode enum: none, interrupted, await, timeout, forbidden, runtime_error
```

### Geospatial Types

#### GeoCircle
Circular geographic region defined by a center `geo` point and a radius in meters. Has `native fn contains(point: geo): bool`.
```gcl
var zone = GeoCircle {
  center: geo::new(45.5, -73.6),
  radius: 5000.0  // meters
};
```

#### GeoPoly
Polygon geographic region: an array of `geo` vertices. Has `native fn contains(point: geo): bool`.
```gcl
var boundary = GeoPoly {
  points: [
    geo::new(45.5, -73.6),
    geo::new(45.6, -73.5),
    geo::new(45.4, -73.4)
  ]
};
```

#### GeoBox
Rectangular bounding box for geographic queries, defined by South-West (`sw`) and North-East (`ne`) `geo` corners. Has `native fn contains(point: geo): bool`.
```gcl
var bbox = GeoBox {
  sw: geo::new(45.4, -73.7),
  ne: geo::new(45.6, -73.5)
};
```

### Enumerations

#### FloatPrecision
Precision levels for floating-point calculations: `p1`, `p10`, `p100`, `p1000`, etc.

#### CalendarUnit
Time units for date operations: `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`.

#### DurationUnit
Duration units: `microseconds`, `milliseconds`, `seconds`, `minutes`, `hours`, `days`.

#### TensorType
Tensor element types: `i32`, `i64`, `f32`, `f64`, `c64`, `c128`.

#### TensorDistance
Distance metrics for tensors: `euclidean` (aka l2), `l2sq`, `cosine`.

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

**WeeklyPeriodicity** - Run on given weekdays, at the time given by `daily`
```gcl
WeeklyPeriodicity {
  days: [DayOfWeek::Mon, DayOfWeek::Fri],  // enum values: Mon..Sun
  daily: DailyPeriodicity { hour: 9 }
}
```

**MonthlyPeriodicity** - Run on given days-of-month (negatives count from end), at `daily` time
```gcl
MonthlyPeriodicity {
  days: [1, 15, -1],                        // 1st, 15th, last day of month
  daily: DailyPeriodicity { hour: 9, minute: 30 }
}
```

**YearlyPeriodicity** - Run on specific calendar dates each year
```gcl
YearlyPeriodicity {
  dates: [
    DateTuple { day: 1, month: Month::Jan },
    DateTuple { day: 25, month: Month::Dec }
  ],
  timezone: TimeZone::"Europe/Luxembourg"
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
// Find task -> PeriodicTask?
var ptask = Scheduler::find(health_check);
if (ptask != null) {
  println("active=${ptask.is_active}, runs=${ptask.execution_count}");
}

// Control execution (return bool)
Scheduler::deactivate(backup_database);  // Pause
Scheduler::activate(backup_database);    // Resume

// List all tasks -> Array<PeriodicTask>
var all_tasks = Scheduler::list();
for (_, task in all_tasks) {
  println("${task.function}: active=${task.is_active}, next=${task.next_execution}");
}
```

### Task & Job Management

#### Task
Asynchronous task execution with status tracking.
```gcl
enum TaskStatus {
  empty, waiting, running, await, cancelled,
  error, ended, ended_with_errors, breakpoint
}

type Task {
  user_id: int;
  task_id: int;
  mod: String?;
  type: String?;
  fun: String?;
  creation: time;
  start: time?;
  duration: duration?;
  status: TaskStatus;
  progress: float?;
  // static native fns: expected_steps, add_steps, parentId, id,
  // no_history, running, history, cancel, is_running
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
enum LogLevel { error, warn, info, perf, trace }

Log::info("Application started");
Log::error("Failed to connect: ${error_message}");
Log::trace("Processing item ${id}");
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

#### Runtime / RuntimeInfo
Query runtime environment information.
```gcl
type RuntimeInfo {
  version: String;
  program_version: String?;
  arch: String;
  timezone: TimeZone;
  license: License;
  io_threads: int;
  bg_threads: int;
  fg_threads: int;
  mem_total: int;
  mem_worker: int;
  disk_data_bytes: int;
}

var info = Runtime::info();
println("Version: ${info.version}, arch: ${info.arch}");
```

#### System
System-level operations: run subprocesses, read env vars, query timezone.
```gcl
// Run a command, returns stdout (throws with stderr on failure)
var out = System::exec("/usr/bin/ls", ["-la"]);

// Spawn without waiting -> ChildProcess
var child = System::spawn("/usr/bin/sleep", ["10"]);
var result = child.wait();   // ChildProcessResult { code, stdout, stderr }
// child.kill();             // force exit

var tz  = System::tz();              // local TimeZone
var home = System::getEnv("HOME");   // String?
```

#### ChildProcess
Handle to a spawned external process.
```gcl
type ChildProcess {
  // private pid: int;
  // native fn wait(): ChildProcessResult;
  // native fn kill();
}

type ChildProcessResult {
  code: int;        // process return code
  stdout: String;
  stderr: String;
}
```

### Security & Identity

#### Identity
A user account on the server: a numeric `id` (`0` is the anonymous/public user), a unique login `name`, a `role` (maps to a set of permission flags granted at login), and the read/write grants this identity extends to others.
```gcl
type Identity {
  id: int;            // 0 = anonymous/public user
  name: String;       // unique login name
  role: String;       // role name -> permission flags
  grants: Array<IdentityGrant>;
  // static native fns include:
  //   current_id(): int           (public, lock-free)
  //   current(): Identity         (requires authentication)
  //   get_by_id(id: int): Identity?   (admin)
  //   get_by_name(name: String): Identity?  (admin)
}

var me = Identity::current();
var caller_id = Identity::current_id();  // 0 if unauthenticated
```

#### IdentityGrant & IdentityGrantType
Per-identity read/write grants extended to other identities.
```gcl
enum IdentityGrantType { read, write, read_write, none }

type IdentityGrant {
  name: String;
  grant: IdentityGrantType;
}
```

### License Management

```gcl
enum LicenseType { community, enterprise, testing }

type License {
  name: String?;        // associated username
  start: time;          // start of validity
  end: time;            // end of validity
  company: String?;     // associated company
  max_memory: int;      // maximum allowed memory in MB
  extra_1: int?;
  extra_2: int?;
  type: LicenseType?;
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
REST API client with type-safe requests. Headers are a `Map<String, String>?`.
```gcl
var headers = Map<String, String> {};
headers.set("Authorization", "Bearer token");
headers.set("Accept", "application/json");

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

// POST / PUT request
var new_user = User { name: "Alice", email: "alice@example.com" };
var result = Http<User> {}.post(
  "https://api.example.com/users",
  new_user,
  headers
);

// Low-level request/response
var resp = Http<User> {}.send(HttpRequest {
  method: HttpMethod::GET,
  url: "https://api.example.com/users/123",
  headers: headers
});
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

### S3 Object Storage

#### S3, S3Bucket, S3Object, S3BasicCredentials
S3-compatible object storage client (access/secret-key authentication).
```gcl
var s3 = S3 {
  host: "localhost:9000",
  region: "us-east-1",
  credentials: S3BasicCredentials {
    access_key: "AKIA...",
    secret_key: "secret"
  },
  force_path_style: true
};

s3.create_bucket("my-bucket");
s3.put_object("my-bucket", "/local/file.txt", "virtual/path/file.txt");

var objects = s3.list_objects("my-bucket", "prefix/", null, 1000);
for (_, obj in objects) {
  println("${obj.key} (${obj.size} bytes, etag=${obj.etag})");
}

s3.get_object("my-bucket", "virtual/path/file.txt", "/local/download.txt");
s3.delete_object("my-bucket", "virtual/path/file.txt");

var buckets = s3.list_buckets(null);  // Array<S3Bucket>
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

var hex = Crypto::hex_encode(input);
var raw = Crypto::hex_decode(hex);

// HMAC-SHA256 (hex output)
var mac = Crypto::sha256_hmac_hex(input, secret_key);

// PKCS1 signing (key is read from a path on disk)
var signature = Crypto::sha256_sign_pkcs1(data, "/keys/private.pem");
var signature_hex = Crypto::sha256_sign_pkcs1_hex(data, "/keys/private.pem");
```

#### Uuid
Cryptographically strong UUID generator (CTR_DRBG seeded from system entropy; not reproducible).
```gcl
var token = Uuid::v4();   // UUID v4: 122 bits CSPRNG entropy, canonical 8-4-4-4-12
var key   = Uuid::v7();   // UUID v7: time-sortable (48-bit ms timestamp + counter + CSPRNG)
// For deterministic/seedable UUIDs use Random.uuid() / Random.uuid_v7()
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
- Security and authentication (Identity, IdentityGrant)
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
  var headers = Map<String, String> {};
  headers.set("Authorization", "Bearer ${api_key}");

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
