# File I/O

## CSV Reading

### Basic CsvReader

```gcl
var format = CsvFormat {
    separator: ';',
    string_delimiter: '\'',
    decimal_separator: ','
};

var reader = CsvReader { path: "./data.csv", format: format };
while (reader.can_read()) {
    var row = reader.read();  // Array<any?>
}
```

### Typed CsvReader

```gcl
@volatile
type Entry {
    id: int;
    name: String;
    values: Array<int>;  // Greedy - consumes remaining columns
}

var reader = CsvReader<Entry> {
    path: "files/entries.csv",
    format: CsvFormat { header_lines: 1 }
};

while (reader.can_read()) {
    var entry = reader.read();  // Entry object
}
```

### Special CSV Types

```gcl
type Record {
    position: geo;           // Consumes 2 cols: lat, lng
    skip: null;              // Skip column
    @format("%d/%m/%y %H:%M")
    date: time;              // Parse with format
    @format(DurationUnit::hours)
    elapsed: duration;       // Parse as hours
}
```

### CsvFormat Options

| Attribute | Type | Description |
|-----------|------|-------------|
| header_lines | int? | Lines to skip |
| separator | char? | Column separator (default `,`) |
| decimal_separator | char? | Decimal point (default `.`) |
| thousands_separator | char? | Thousands separator |
| string_delimiter | char? | String quote char |
| format | String? | Date format |
| tz | TimeZone? | Timezone |

### CsvWriter

```gcl
var writer = CsvWriter {
    path: "./data.csv",
    format: CsvFormat { separator: ';', decimal_separator: ',' }
};
writer.write(["John", "Doe", time::now(), 56]);
```

## JSON

### Read JSON File

```gcl
var reader = JsonReader { path: "data.json" };
while (reader.available() > 0) {
    var obj = reader.read();  // Map with key-value pairs
}
```

### Parse JSON String

```gcl
var reader = Json{};
var parsed = reader.parse(jsonString as String);
```

### Typed JSON Reading

```gcl
abstract type Geometry {}
type Point extends Geometry { coordinates: Tuple<float, float>; }
type LineString extends Geometry { coordinates: Array<Tuple<float, float>>; }

type Feature { geometry: Geometry; }

var reader = JsonReader<Feature> { path: "data.json" };
while (reader.can_read()) {
    pprint(reader.read());  // Polymorphic dispatch
}
```

### Write JSON

```gcl
var writer = JsonWriter { path: "./out.json" };
writer.write(Foo { name: "John", age: 42 });
writer.writeln([true, false, null]);  // With newline

// Append mode
var appendWriter = JsonWriter { path: "./out.json", append: true };
```

## Files & Folders

### List Files

```gcl
var files = File::ls("data", "csv", true);  // dir, extension, recursive
```

### FileWalker

```gcl
var walker = FileWalker { path: "./dataFolder" };
while (!walker.isEmpty()) {
    var file = walker.next();
    if (file.isDir()) {
        // Handle directory
    } else {
        // Handle file
    }
}
```

### File Operations

```gcl
File::mkdir("./path/to/dir");
File::copy("./origin.txt", "./copy.txt");
File::delete("./file.txt");
File::rename("./old", "./new");

File::baseDir();     // Path to files folder
File::userDir();     // Path to user directory
File::workingDir();  // Path for current task/request
```

## Network/HTTP

### GET Request

```gcl
var page = Http::get("http://example.com", null);
var data = Http::get("http://api.com", [
    HttpHeader { name: "Accept", value: "application/json" },
    HttpHeader { name: "Bearer", value: "${token}" }
]);
```

### Download File

```gcl
Http::getFile("https://example.com/file", "./local.json", null);
```

### POST/PUT Request

```gcl
var request = { sampling: ["live"], ids: [uuid] };
var result = Http::post(endpoint, request, headers);
var parsed = JsonReader::parse(result as String);
```

### Generic HTTP Client

```gcl
var client = Http{};
var req = HttpRequest {
    method: HttpMethod::GET,
    url: "https://api.example.com?p=1",
    headers: Map<String, String>{ "Content-Type": "application/json" },
    body: "data"
};
var response = client.send(req);
// response.status_code, response.headers, response.content
```

### Typed HTTP Client

```gcl
type ApiResponse { path: String?; }

var client = Http<ApiResponse>{};
var response = client.send(req);
println(response.content?.path);  // Typed access
```

### URL Utilities

```gcl
var url = Url::parse("https://example.com/path?p1=true#section");
// url.protocol, url.host, url.path, url.params, url.hash

var encoded = Url::encode(FormFields { name: "John", age: 42 });
// "name=John&age=42"
```

## SMTP Email

```gcl
var smtp = Smtp {
    host: "smtp.company.com",
    port: 587,
    mode: SmtpMode::starttls,
    authenticate: SmtpAuth::login,
    user: "user",
    pass: "pass"
};

var email = Email {
    from: "\"John\" <john@company.com>",
    to: ["boss@company.com"],
    cc: ["team@company.com"],
    subject: "Report",
    body: "Content here",
    body_is_html: false
};

smtp.send(email);
```
