# File I/O

## CSV Reading

### Basic CsvReader

```gcl
var reader = CsvReader { path: "./data.csv" };
while (reader.can_read()) {
    var row = reader.read(); // Array<any?>
}
```

### Typed CsvReader

```gcl
@volatile
type Entry {
    id: int;
    name: String;
    values: Array<int>; // Greedy - consumes remaining columns
}

var reader = CsvReader<Entry> { path: "files/entries.csv" };
while (reader.can_read()) {
    var entry = reader.read(); // Entry object
}
```

### CsvFormat

`CsvFormat` fields are `char?`, not `String?` — use single quotes:

```gcl
// Wrong — String literals
var fmt = CsvFormat { separator: ",", string_delimiter: "\"" };
// Right — char literals
var fmt = CsvFormat { separator: ',', string_delimiter: '"' };

var reader = CsvReader { path: "./data.csv", format: fmt };
```

### CSV features

```gcl
type Record {
    position: geo;           // Consumes 2 cols: lat, lng
    skip: null;              // Skip column
    @format("%d/%m/%y %H:%M")
    date: time;              // Parse with format — required for non-default time formats
    @format(DurationUnit::hours)
    elapsed: duration;       // Parse as hours
}
```

**`@format` on time fields** — without it, time parsing silently fails or throws. Always annotate `time` fields with the expected format.

### CsvWriter

```gcl
var writer = CsvWriter { path: "./data.csv" };
writer.write(["John", "Doe", time::now(), 56]);
```

## JSON

### Read File

```gcl
var reader = JsonReader { path: "data.json" };
while (reader.can_read()) {
    var obj = reader.read(); // Map with key-value pairs
}
```

### Parse

```gcl
var j = Json<String> {};
var s = j.parse("\"hello\"");
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
var parsed = Json {}.parse(result as String);
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
