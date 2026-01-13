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
UTF-8 text file operations with line-based I/O.

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

// Re-use reader for multiple files
reader.set_path("/path/to/other_employees.csv");
Assert::isTrue(reader.can_read());
```

## CSV Analysis & Code Generation

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
    subject: "Test Email",
    body: "<h1>Hello World</h1>",
    body_is_html: true
};

// Send email (would throw exception on failure)
smtp.send(email);
```
