# [std](index.html) > io

[Source](io.source.html)

Input/output module providing file operations, data serialization, network communication, and email functionality for GreyCat applications.

## Writers & Readers

All reader types implement position tracking and support streaming operations for memory-efficient processing of large files. Writers support both append and overwrite modes. CSV operations include advanced analysis capabilities for automatic type inference and code generation.

### GcbWriter / GcbReader
Binary format using GreyCat's ABI encoding for efficient serialization of any GreyCat type.

```gcl
// Write binary data
var writer = GcbWriter<MyType> { path: "/path/to/data.gcb" };
writer.write(my_object);
writer.flush();

// Read binary data back
var reader = GcbReader<MyType> { path: "/path/to/data.gcb" };
while (reader.can_read()) {
    var obj = reader.read();
    Assert::isNotNull(obj);
}
```

### BinReader
Low-level binary file reader for reading primitive types and tensors directly from binary files.

```gcl
// Read raw binary data (integers, floats, tensors)
var reader = BinReader { path: "/path/to/data.bin" };

// Read primitive types
var int32_val = reader.read_i32();   // Read 32-bit integer
var int64_val = reader.read_i64();   // Read 64-bit integer
var float32_val = reader.read_f32(); // Read IEEE 754 binary32
var float64_val = reader.read_f64(); // Read IEEE 754 binary64

// Read tensor with specified type and shape
var tensor = reader.read_tensor(TensorType::f32, [3, 3]);

// Check available data
while (reader.can_read()) {
    var bytes_left = reader.available();
    // Process remaining data...
}
```

### XmlReader
XML file reader for parsing XML documents into typed GreyCat objects.

```gcl
// Read XML data
var reader = XmlReader<MyXmlType> { path: "/path/to/data.xml" };
while (reader.can_read()) {
    var obj = reader.read();
    // Process XML element...
}
```

### TextWriter / TextReader
Text file operations with line-based I/O. TextWriter writes UTF-8. TextReader reads byte data line by line (does not perform UTF-8 validation).

```gcl
// Write text lines
var writer = TextWriter<String> { path: "/path/to/output.txt" };
writer.writeln("First line");
writer.writeln("Second line");
writer.flush();

// Read text lines
var reader = TextReader { path: "/path/to/output.txt" };
Assert::equals(reader.read(), "First line");
Assert::equals(reader.read(), "Second line");
Assert::isFalse(reader.can_read());
```

### JsonWriter / JsonReader
JSON serialization supporting both single objects and NDJSON (newline-delimited JSON) streams.

```gcl
// Write JSON objects
var writer = JsonWriter<Person> { path: "/path/to/people.json" };
writer.writeln(person1); // Each call writes one JSON object per line
writer.writeln(person2);
writer.flush();

// Read JSON stream
var reader = JsonReader<Person> { path: "/path/to/people.json" };
var first_person = reader.read();
Assert::equals(first_person.name, "John");

// Parse JSON strings directly
var parsed = Json<Person> {}.parse("{\"name\":\"Alice\",\"age\":30}");
Assert::equals(parsed.name, "Alice");

// Serialize any value to JSON string
var json_str = Json::to_string(parsed);
Assert::isNotNull(json_str);
```

### CsvWriter / CsvReader
CSV file operations with automatic header generation and configurable formatting.

```gcl
// Configure CSV format
var format = CsvFormat {
    header_lines: 1,
    separator: ',',
    string_delimiter: '"'
};

// Write structured data as CSV
var writer = CsvWriter<Employee> { path: "/path/to/employees.csv", format: format };
writer.write(employee1); // Headers written automatically on first write
writer.write(employee2);
writer.flush();

// Read CSV with type validation
var reader = CsvReader<Employee> { path: "/path/to/employees.csv", format: format };
while (reader.can_read()) {
    var emp = reader.read();
    Assert::isNotNull(emp.name);
}

// Write a raw line directly (no automatic header generation)
writer.write_line("custom,raw,line");
writer.flush();

// Re-use reader for multiple files
reader.set_path("/path/to/other_employees.csv");
Assert::isTrue(reader.can_read());

// Get the last read line as raw string
var last = reader.last_line();
```

## CSV Analysis & Code Generation

### CsvFormat
Configuration for CSV parsing and writing.

```gcl
var format = CsvFormat {
    header_lines: 1,
    separator: ',',
    string_delimiter: '"',
    decimal_separator: '.',
    thousands_separator: '_',
    trim: true,
    format: "yyyy-MM-dd",
    tz: TimeZone::Europe_Luxembourg,
    strict: false,
    nearest_time: false
};
```

### CsvSharding
Sharding configuration for partitioned CSV reading.

```gcl
var sharding = CsvSharding {
    id: 0,
    column: 2,
    modulo: 4
};
var reader = CsvReader<MyType> { path: "/path/to/data.csv", sharding: sharding };
```

### CsvColumnStatistics
Per-column statistics collected during CSV analysis.

```gcl
// Fields:
//   name: String?             - column name as in CSV header
//   example: any?             - one value
//   null_count: int           - number of null values
//   bool_count: int           - number of boolean values
//   int_count: int            - number of integer values
//   float_count: int          - number of float values
//   string_count: int         - number of string values
//   date_count: int           - number of date values
//   date_format_count: Map<String, int>  - occurrence of each date format matched
//   enumerable_count: Map<any, int>      - occurrences of enumerable values found
//   profile: Gaussian         - statistics on numeric values
```

### CsvStatistics
Aggregated statistics from CSV analysis across one or more files.

```gcl
// Fields:
//   header_lines: int?
//   separator: char?
//   string_delimiter: char?
//   decimal_separator: char?
//   thousands_separator: char?
//   columns: Array<CsvColumnStatistics>  - statistics per column
//   line_count: int           - accumulated analyzed rows for all CSV files
//   fail_count: int           - accumulated number of failed lines
//   file_count: int           - number of CSV files explored
```

### CsvAnalysisConfig
Configuration for CSV analysis behavior.

```gcl
var config = CsvAnalysisConfig {
    header_lines: 1,
    separator: ',',
    string_delimiter: '"',
    decimal_separator: '.',
    thousands_separator: '_',
    row_limit: 1000,
    enumerable_limit: 50,
    date_check_limit: 100,
    date_formats: ["yyyy-MM-dd", "dd/MM/yyyy"]
};
```

### Csv
Static utility for analyzing CSV files and generating GreyCat types.

```gcl
// Analyze CSV structure
var config = CsvAnalysisConfig {
    row_limit: 1000,
    enumerable_limit: 50
};

var files = Array<File> {File::open("/path/to/sales.csv")!!};
var stats = Csv::analyze(files, config);

// Generate GreyCat types based on analysis
var type_definitions = Csv::generate(stats);
Assert::isTrue(type_definitions.contains("type"));

// Sample data for preview
var reader = CsvReader<any> { path: "/path/to/sales.csv" };
var sample_table = Csv::sample(reader, 100);
Assert::equals(sample_table.rows(), 100);
```

## File System Operations

### File
Comprehensive file system utilities for file and directory management.

```gcl
// File discovery and metadata
var csv_files = File::ls("/path/to/data", ".csv", true); // Recursive search
Assert::isTrue(csv_files.size() > 0);

var file = File::open("/path/to/important.txt")!!;
Assert::isFalse(file.isDir());
Assert::equals(file.extension()!!, "txt");
Assert::isNotNull(file.sha256()!!);

// File operations
Assert::isTrue(File::copy("/path/to/source.txt", "/path/to/dest.txt"));
Assert::isTrue(File::rename("/path/to/old.txt", "/path/to/new.txt"));
Assert::isTrue(File::delete("/path/to/unwanted.txt"));
Assert::isTrue(File::mkdir("/path/to/directory"));

// Working directories
var base_dir = File::baseDir();
var user_dir = File::userDir();
var work_dir = File::workingDir();
Assert::isNotNull(base_dir);
```

### FileWalker
Iterator for traversing file system hierarchies.

```gcl
// Walk through directory structure
var walker = FileWalker { path: "." };
var file_count = 0;

while (!walker.isEmpty()) {
    var file = walker.next();
    if (file != null && !file.isDir()) {
        file_count++;
    }
}

println("file count = ${file_count}");
```

## Network & Web

### Url
URL parsing and manipulation utility.

```gcl
// Parse URL components
var url = Url::parse("https://api.example.com:8080/users?active=true#section1");

Assert::equals(url.protocol, "https");
Assert::equals(url.host, "api.example.com");
Assert::equals(url.port, 8080);
Assert::equals(url.path, "/users");
Assert::equals(url.params?.get("active"), "true");
Assert::equals(url.hash, "section1");

// URL-encode a value
var encoded = Url::encode("hello world&foo=bar");
Assert::isNotNull(encoded);
```

### Http
HTTP client for REST API communication and file downloads.

```gcl
// HTTP GET with custom headers (Map<String, String>)
var headers = Map<String, String> {
    ["Authorization"] = "Bearer token123",
    ["Accept"] = "application/json"
};

var response = Http<String> {}.get("https://api.example.com/users", headers);
Assert::isNotNull(response);

// Download file directly
Http<any> {}.getFile("https://example.com/data.csv", "/path/to/local.csv", null);
var downloaded = File::open("/path/to/local.csv")!!;
Assert::isTrue(downloaded.size!! > 0);

// POST data
var payload = User { name: "John", email: "john@example.com" };
var result = Http<User> {}.post("https://api.example.com/users", payload, headers);
Assert::isNotNull(result);

// PUT data
var updated = Http<User> {}.put("https://api.example.com/users/1", payload, headers);
Assert::isNotNull(updated);

// Send a full HTTP request with HttpRequest/HttpResponse
var request = HttpRequest {
    method: HttpMethod::POST,
    url: "https://api.example.com/users",
    headers: headers,
    body: "{\"name\":\"John\"}",
    timeout: 30_s
};

var resp = Http<String> {}.send(request);
Assert::equals(resp.status_code, 200);
Assert::isNotNull(resp.content);
```

### HttpMethod
Enum of supported HTTP methods: `GET`, `HEAD`, `POST`, `PUT`, `DELETE`, `CONNECT`, `OPTIONS`, `TRACE`, `PATCH`.

### HttpRequest
Request wrapper for `Http::send`.

```gcl
// Fields:
//   method: HttpMethod
//   url: String
//   headers: Map<String, String>?
//   body: String?
//   timeout: duration?
```

### HttpResponse\<T\>
Response wrapper returned by `Http::send`.

```gcl
// Fields:
//   status_code: int
//   headers: Map<String, String>
//   content: T?
//   error_msg: String?
```

## Email & Communication

### Email & Smtp
Email composition and SMTP delivery.

```gcl
// Configure SMTP server
var smtp = Smtp {
    host: "smtp.example.com",
    port: 587,
    mode: SmtpMode::starttls,
    authenticate: SmtpAuth::plain,
    user: "sender@example.com",
    pass: "password123"
};

// Compose and send email
var email = Email {
    from: "sender@example.com",
    to: ["recipient@example.com"],
    cc: ["cc@example.com"],
    bcc: ["hidden@example.com"],
    subject: "Test Email",
    body: "<h1>Hello World</h1>",
    body_is_html: true
};

// Send email (would throw exception on failure)
smtp.send(email);
```

## S3 Object Storage (Built-in)

S3-compatible object storage types are available directly in the standard library without requiring `@library("s3")`.

### S3

Connection to a remote S3-compatible server (AWS S3, MinIO, etc.).

```gcl
var s3 = S3 {
    host: "s3.amazonaws.com",
    region: "us-east-1",
    credentials: S3BasicCredentials {
        access_key: env("AWS_ACCESS_KEY"),
        secret_key: env("AWS_SECRET_KEY")
    },
    force_path_style: true // required for MinIO
};

// Upload and download files
s3.put_object("my-bucket", "/local/file.txt", "remote/path/file.txt");
s3.get_object("my-bucket", "remote/path/file.txt", "/local/downloaded.txt");

// List objects (up to 1000 per call, paginate with start_after)
var objects = s3.list_objects("my-bucket", "prefix/", null, 100);
for (obj in objects) {
    println("${obj.key} - ${obj.size} bytes");
}

// Delete objects
s3.delete_object("my-bucket", "old-file.txt");

// Bucket management
s3.create_bucket("new-bucket");
var buckets = s3.list_buckets(null);
```

### S3Object
Represents an object in S3: `key: String`, `last_modified: time`, `size: int`, `etag: String`.

### S3Bucket
Represents a bucket: `name: String`, `creation_date: time`.

### S3BasicCredentials
Authentication: `access_key: String`, `secret_key: String`.
