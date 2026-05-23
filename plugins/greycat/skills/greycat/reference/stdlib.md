# GreyCat standard library

The stdlib ships under `lib/std/` (split into `core.gcl`, `io.gcl`, `runtime.gcl`, `util.gcl`). Brought into scope by `@library("std", "<version>")` in `project.gcl`. This reference covers the types and functions you reach for most. For exact signatures, read the `.gcl` files directly — they carry doc comments.

## Contents

- Collections (Array, Map, Set, Tuple, Table, Buffer)
- Util collections (Queue, Stack, SlidingWindow, TimeWindow)
- Node tags (graph-persisted: node, nodeTime, nodeList, nodeIndex, nodeGeo)
- Time and duration
- Geographic types (geo, GeoBox, GeoCircle, GeoPoly)
- IO (Reader / Writer / File)
- HTTP and networking
- Crypto and UUID
- Tensor and vector index
- Runtime introspection (Runtime, System, Task, Scheduler, Identity)
- Logging
- Math globals
- Built-in conversions (println, parseNumber, clone)

## Collections

### Array<T>

Ordered container of elements. Built with `[...]` literal or `Array<T> { ... }`.

```gcl
var xs: Array<int> = [1, 2, 3];
var ys = Array<int> { };           // empty
ys.add(4);
ys.add_all([5, 6, 7]);
ys.get(0);                          // 4
ys.set(0, 99);
ys.size();
ys.index_of(5);
ys.remove(0);                       // returns removed value
ys.sort(SortOrder::asc);
ys.sort_by(field, SortOrder::desc); // sort by attribute of contained objects
ys.swap(0, 1);
ys.fill(10, 0);                     // size 10, default value 0
ys.range_equals(0, 4, null);        // all five elements equal null?
```

Iteration:

```gcl
for (k, v in xs) { ... }
for (k, v in xs.size()) { ... }       // NOT supported — must iterate the collection
```

### Map<K, V>

Hash map. Key must be a primitive or `String`.

```gcl
var m = Map<String, int> { };
m.set("a", 1);
m.get("a");                         // int?
m.contains("a");
m.remove("a");
m.size();
for (k, v in m) { ... }            // iterate key/value pairs
```

### Set<T>

GreyCat does not ship a built-in `Set` type as of this stdlib version — model sets as `Map<T, bool>` or a custom type.

### Tuple<T, U>

A pair. Construct with the sugar `(a, b)` or the named-field form `Tuple<T, U> { x: a, y: b }`. Fields are `x` and `y`. The positional brace form `Tuple<T, U> { a, b }` does **not** parse — only `Array`, `Map`, `node`, and `geo` accept positional braces.

```gcl
var p: Tuple<int, String> = (42, "hi");
p.x;       // 42
p.y;       // "hi"
```

### Table<T>

Two-dimensional structure. Useful as a result for samplers and CSV / SQL-style operations.

```gcl
var t = Table<MyRow> { };
t.init(100, 5);                    // capacity
t.set_cell(0, 0, "value");
t.get_cell(0, 0);
t.add_row(MyRow { ... });
t.set_row(0, MyRow { ... });
t.get_row(0);
t.sort(0, SortOrder::asc);
t.cols();
t.rows();
```

### Buffer

Binary blob. Bytes append-and-read.

```gcl
var b = Buffer { };
b.add("hello");
b.add_and_pad("hi", 8, ' ');       // "hi      "
b.size();
b.toString();
b.clear();
```

## Util collections

From `lib/std/util.gcl`. Memory-only — these are not graph-persisted; for persisted equivalents use the node tags below.

### Queue<T>

Optionally bounded FIFO. When `capacity` is set and the queue is full, `push` drops the front element.

```gcl
var q = Queue<String> { capacity: 3 };
q.push("a");   q.push("b");   q.push("c");   q.push("d");   // "a" is dropped
q.pop();          // "b"
q.front();        // peek front, no removal
q.back();         // peek back, no removal
q.clear();
```

### Stack<T>

LIFO.

```gcl
var s = Stack<int> { };
s.push(1);   s.push(2);   s.push(3);
s.pop();          // 3
s.first();        // peek top, no removal — currently 2
s.last();         // peek bottom — 1
```

### SlidingWindow<T>

Fixed-count FIFO with running aggregates. `add()` evicts the oldest value when `span` is reached. All aggregates are O(1) thanks to running sums.

```gcl
var w = SlidingWindow<float> { span: 100 };
w.add(1.0);   w.add(2.0);   w.add(3.0);
w.avg();          // running mean
w.std();          // running std
w.median();       // O(n log n) — recomputed each call
w.min();   w.max();
w.size();
```

### TimeWindow<T>

Time-bounded FIFO with running aggregates. `add(t, v)` evicts entries older than `span` from the most-recent timestamp. `update(t)` slides the window without inserting.

```gcl
var w = TimeWindow<float> { span: 5min };
w.add(time::now(), 42.0);
w.update(time::now() + 1min);     // slide forward without adding
w.avg();   w.std();   w.median();
w.min();   w.max();               // each returns Tuple<time, T>?
```

## Node tags (graph-persisted)

These types persist across worker restarts. They are the bridge between in-memory logic and the GreyCat graph database.

### node<T>

Singleton, lazily resolved.

```gcl
var n: node<User> = ...;
n.resolve();                       // T — the inner value
n.set(updated_user);               // mutate the pointed-to value
n->name;                           // == n.resolve().name (deref + access)
*n;                                // == n.resolve() — unary `*` is deref-shorthand
```

Tagged `@deref("resolve")` — the `->` operator calls `resolve()` then accesses the result.

### nodeTime<T>

Time-series of `T` values keyed by `time`.

```gcl
var s: nodeTime<float> = /* ... */;
s.setAt(time::now(), 1.0);
s.setAll(time::now(), [1.0, 2.0, 3.0], 1s);
s.getAt(t);                        // exact-match read; null if no entry
s.resolveAt(t);                    // closest-prior read
s.resolveTimeAt(t);                // the actual timestamp resolved to
s.firstTime();   s.first();
s.lastTime();    s.last();
s.prev(t);   s.next(t);
s.size();
s.rangeSize(from, to);
s.removeAt(t);   s.removeAll();

// Iteration with time-window query
for (t, v in s[from..to]) {} // from/to included
for (t, v in s[from..to[) {} // to excluded
for (t, v in s]from..to]) {} // from excluded
for (t, v in s]from..to[) {} // from/to execluded
```

Tagged `@iterable @deref("resolve")`.

### nodeList<T>

Sparse int-indexed list. Used for very large series whose indices may be scattered across the i64 range.

```gcl
var l: nodeList<float> = /* ... */;
l.add(1.0);                        // append at index (lastIndex + 1)
l.set(100, 2.0);                   // set at explicit index
l.get(100);
l.resolve(150);                    // closest-prior read
l.firstIndex();  l.first();
l.lastIndex();   l.last();
l.rangeSize(0, 1000);
for (i, v in l[0..1000]) { /* ... */ }
```

### nodeIndex<K, V>

Sorted index with O(log n) keyed access.

```gcl
var idx: nodeIndex<String, User> = /* ... */;
idx.set("alice", User { /* ... */ });
idx.get("alice");                  // User?
idx.search("alice", 10);           // closest-neighbor search
idx.remove("alice");
idx.size();
for (k, v in idx limit 100) { /* ... */ }
```

### nodeGeo<T>

Spatial index keyed by `geo`.

```gcl
var g: nodeGeo<Station> = /* ... */;
g.set(geo { 49.6, 6.1 }, station);
g.get(geo { 49.6, 6.1 });
g.resolve(geo { 49.7, 6.0 });      // closest-prior read
g.rangeSize(geo { 49, 6 }, geo { 50, 7 });
```

### Sampling node tags

Every node tag exposes a static `sample(...)` that returns a `Table` of values from a range, plus `info(...)` returning a per-node `NodeInfo<K>` for metadata. Signatures differ by tag:

```gcl
nodeTime::sample(refs: Array<nodeTime>, from: time?, to: time?, maxRows: int,
                 mode: SamplingMode, maxDephasing: duration?, tz: TimeZone?): Table
nodeIndex::sample(refs: Array<nodeIndex>, from: any?, maxRows: int, mode: SamplingMode): Table
nodeList::sample(refs: Array<nodeList>, from: int?, to: int?, maxRows: int,
                 mode: SamplingMode, maxDephasing: int?): Table
nodeGeo::sample(refs: Array<nodeGeo>, from: geo?, to: geo?, maxRows: int, mode: SamplingMode): Table
```

`SamplingMode` controls how points within the range are picked:

| Mode          | Behavior                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `fixed`       | Pick samples at a fixed delta between index values. Can return a large set — bound `from`/`to` carefully.  |
| `fixed_reg`   | Like `fixed`, but linearly regresses numerical values between the nearest stored points to smooth output. |
| `adaptative`  | Variable delta — bound the number of skipped elements between samples. Useful for non-monotonic series.   |
| `dense`       | All elements in range; no down-sampling.                                                                  |

`maxRows` is the upper bound on rows returned regardless of mode. `maxDephasing` (nodeTime / nodeList only) caps the allowed time / index drift between sampled points across multiple input refs — useful when aligning unsynchronized series.

## Time and duration

### time

```gcl
time::now()                        // wall-clock time
time::current()                    // contextual time (defaults to time::max outside any `at (t) { }` block)
time::min   time::max
time::new(epoch, DurationUnit::milliseconds)
time::parse("2025-05-22", "%Y-%m-%d")
time::isLeap(2024)
time::totalDaysInYear(2024)
time::totalDaysInMonth(2, 2024)

var t = time::now();
t.to(DurationUnit::seconds)                   // int epoch seconds
t.format("%+", null)                          // ISO 8601 default
t.toDate(TimeZone::"Europe/Paris")            // Date object
t.calendar_add(1, CalendarUnit::month, null)
t.calendar_floor(CalendarUnit::day, null)
t.calendar_ceiling(CalendarUnit::day, null)
t.dayOfWeek(null)
t.weekOfYear(null)
t.startOfWeek(null)
t.endOfWeek(null)
```

#### strftime format specifiers

`time::format` and `time::parse` use POSIX strftime conventions. Common ones:

| Specifier | Meaning                              |
| --------- | ------------------------------------ |
| `%Y`      | 4-digit year (`2025`)                |
| `%y`      | 2-digit year (`25`)                  |
| `%m`      | Month, zero-padded (`01`–`12`)       |
| `%d`      | Day-of-month, zero-padded (`01`–`31`) |
| `%e`      | Day-of-month, space-padded (` 1`–`31`) |
| `%H`      | Hour 24h, zero-padded (`00`–`23`)    |
| `%I`      | Hour 12h, zero-padded (`01`–`12`)    |
| `%M`      | Minute (`00`–`59`)                   |
| `%S`      | Second (`00`–`60`)                   |
| `%s`      | Unix epoch seconds                   |
| `%z`      | Timezone offset (`+0200`)            |

Full list: <https://www.gnu.org/software/libc/manual/html_node/Formatting-Calendar-Time.html>

### duration

```gcl
duration::new(60, DurationUnit::seconds)
duration::newf(1.5, DurationUnit::hours)

var d = 2hour_30min;               // literal form
d.to(DurationUnit::seconds);
d.tof(DurationUnit::hours);
d.add(10, DurationUnit::minutes);
d.subtract(5, DurationUnit::seconds);
```

### Enums

```gcl
enum DurationUnit { microseconds, milliseconds, seconds, minutes, hours, days }
enum CalendarUnit { year, month, day, hour, minute, second, microsecond }
enum TimeZone { "UTC", "Europe/Paris", "Asia/Tokyo", ... }  // IANA tz names
enum SortOrder { asc, desc }
```

`TimeZone` uses string-named entries because some names contain `/` or `-`.

**`DurationUnit` vs `CalendarUnit`:** `DurationUnit` values are constant amounts of microseconds — `days` is exactly `86400 × 1e6 µs`. There is no `months` or `years` because their length depends on the calendar. For anything ≥ a month, use `CalendarUnit` with `calendar_add` / `calendar_floor` / `calendar_ceiling` (these honor the supplied `TimeZone` and handle month-length / leap-year edges correctly).

`DurationUnit` microsecond values (useful when converting raw `int` epochs):

| Unit           | µs        |
| -------------- | --------- |
| `microseconds` | `1`       |
| `milliseconds` | `1e3`     |
| `seconds`      | `1e6`     |
| `minutes`      | `60e6`    |
| `hours`        | `3600e6`  |
| `days`         | `86400e6` |

### Date

```gcl
var date = Date::from_time(time::now(), TimeZone::"UTC");
date.year;   date.month;   date.day;
date.hour;   date.minute;  date.second;   date.microsecond;
date.to_time(null);                            // back to time — throws if invalid (e.g. DST gap)
date.to_nearest_time(TimeZone::"Europe/Paris");// returns the next valid instant if the date falls in a DST gap
Date::parse("2025-05-22", "%Y-%m-%d");
```

`to_time` is strict and raises on a date that doesn't exist in the supplied timezone (the hour skipped during a spring-forward DST transition is the canonical case). `to_nearest_time` returns the next-valid `time` instead — prefer it whenever you accept user-supplied dates.

## Geographic types

```gcl
var p: geo = geo { 49.6, 6.1 };               // (lat, lng)
p.lat();   p.lng();   p.distance(other);
p.toString();   p.toGeohash();

var box = GeoBox { sw: ..., ne: ... };
box.contains(p);

var circle = GeoCircle { center: p, radius: 1000.0 };
circle.contains(other);
circle.sw();    circle.ne();

var poly = GeoPoly { points: [p1, p2, p3] };
poly.contains(other);
```

## IO

```gcl
abstract type Writer<T> { fn write(v: T); fn flush(); }
abstract type Reader<T> { fn read(): T; fn can_read(): bool; fn available(): int; }
```

Concrete subtypes:

| Type                              | Format                                             |
| --------------------------------- | -------------------------------------------------- |
| `GcbWriter<T>` / `GcbReader<T>`   | GreyCat binary (compact, fast)                     |
| `TextWriter<T>` / `TextReader`    | UTF-8 line-based                                   |
| `JsonWriter<T>` / `JsonReader<T>` | NDJSON (line-separated JSON)                       |
| `CsvWriter<T>` / `CsvReader<T>`   | CSV (config via `CsvFormat`)                       |
| `XmlReader<T>`                    | XML                                                |
| `BinReader`                       | Raw `i32` / `i64` / `f32` / `f64` / `Tensor` reads |

### File

```gcl
File::baseDir();           // <project>/files/
File::userDir();           // <project>/files/<user_name>/
File::workingDir();        // current task / request scoped dir
File::open(path);          // File?
File::ls(path, ".csv", true);   // recursive list with suffix filter
File::mkdir(path);
File::delete(path);
File::rename(old, new);
File::copy(src, dst);

var f = File::open("data.csv");
f.isDir();   f.name();   f.extension();   f.sha256();
f.size;   f.last_modification;
```

`FileWalker` lazily iterates a directory tree:

```gcl
var walker = FileWalker { path: "./data" };
while (!walker.isEmpty()) {
    var entry = walker.next();             // File?
    if (entry != null && !entry.isDir()) {
        // process file
    }
}
```

### Json

```gcl
Json::to_string(any_value);            // serialize
Json<MyType>{}.parse(json_str);        // typed deserialize
```

Typed reader for NDJSON streams. Works with polymorphism — the parser dispatches to the concrete subtype based on the JSON shape:

```gcl
abstract type Geometry { kind: String; }
type Point      extends Geometry { coords: Tuple<float, float>; }
type LineString extends Geometry { coords: Array<Tuple<float, float>>; }

var reader = JsonReader<Geometry> { path: "shapes.ndjson" };
while (reader.can_read()) {
    var g = reader.read();
    if (g is Point) { /* g.coords is Tuple<float, float> */ }
    else if (g is LineString) { /* ... */ }
}
```

### Csv

```gcl
Csv::analyze(["data.csv"], null);      // infer CsvStatistics
Csv::generate(stats);                  // generate types matching stats
Csv::sample(reader, 1000);             // read into a Table
```

Typed `CsvReader<T>` reads each line into a `T`. The target type's attribute order matches CSV column order; an `Array<U>` attribute marked greedy consumes all remaining columns. Use `@format` on `time` / `duration` fields and tune parsing via `CsvFormat`:

```gcl
@volatile
type Reading {
    @format("%d/%m/%y %H:%M") t: time;
    sensor_id: String;
    location: geo;                    // consumes two columns (lat, lng)
    @format(DurationUnit::seconds) elapsed: duration;
    values: Array<float>;             // greedy: takes every remaining column
}

var fmt = CsvFormat {
    header_lines: 1,
    separator: ',',                   // CHAR literal (single quotes) — NOT a String
    string_delimiter: '"',
    decimal_separator: '.',
    trim: true,
    tz: TimeZone::"Europe/Paris",
    nearest_time: true,               // gracefully handle DST gaps
};
var reader = CsvReader<Reading> { path: "data.csv", format: fmt };
while (reader.can_read()) {
    var r = reader.read();
    // ...
}
```

**Common CSV pitfalls:**

- `CsvFormat.separator` is `char?` (one byte) — write `','`, not `","`. Same for `string_delimiter`, `decimal_separator`, `thousands_separator`.
- Without `@format` on a `time` field, parsing falls back to ISO 8601 / epoch-ms. Mismatched formats throw at parse time.
- `geo` is two columns wide (lat then lng). Account for that when matching column order.
- A greedy `Array<U>` field MUST be the last attribute — there's no way to delimit where it stops otherwise.

## HTTP and networking

### Http<T>

```gcl
var client = Http<JsonResponse> { };
client.get("https://api.example.com/users", null);
client.post("https://api.example.com/users", body, headers_map);
client.put("https://api.example.com/users/1", body, headers_map);
client.getFile("https://example.com/file.bin", "/tmp/out.bin", null);

// Raw form
var req = HttpRequest { method: HttpMethod::GET, url: "...", headers: m, body: null, timeout: 30s };
var resp = client.send(req);
resp.status_code;   resp.content;   resp.error_msg;
```

### Url

```gcl
var url = Url::parse("https://user:pass@host:443/path?key=value#frag");
url.protocol;   url.host;   url.port;   url.path;   url.params;

Url::encode("a b&c");                       // "a%20b%26c"
var m = Map<String, any> {};
m.set("name", "John"); m.set("age", 42);
Url::encode(m);                             // "name=John&age=42"
```

`Url::encode` accepts any value: strings are percent-encoded, objects and `Map`s are flattened to `key=value&key=value` x-www-form-urlencoded form.

### Smtp / S3

`Smtp` for sending email (with `Email`, `SmtpMode`, `SmtpAuth`).
`S3` for object storage (with `S3Object`, `S3Bucket`, `S3BasicCredentials`).

See [io.gcl](../../../../lib/std/io.gcl) for full signatures.

## Crypto and UUID

```gcl
Crypto::sha1(content);             Crypto::sha1hex(content);
Crypto::sha256(content);           Crypto::sha256hex(content);
Crypto::sha256_hmac_hex(input, key);
Crypto::sha256_sign_pkcs1(input, key_path);
Crypto::sha256_sign_pkcs1_hex(input, key_path);
Crypto::base64_encode(content);    Crypto::base64_decode(content);
Crypto::base64url_encode(content); Crypto::base64url_decode(content);
Crypto::hex_encode(content);       Crypto::hex_decode(content);
Crypto::url_encode(content);       Crypto::url_decode(content);

Uuid::v4();                        // cryptographically secure
Uuid::v7();                        // time-sortable, suitable for DB primary keys

var rng = Random { seed: 42 };
rng.uuid();    rng.uuid_v7();      // seedable / reproducible — NOT secure
rng.uniform(0, 100);   rng.uniformf(0.0, 1.0);
rng.normal(0.0, 1.0);
```

## Tensor and vector index

```gcl
var t = Tensor { };
t.init(TensorType::f64, [3, 3]);
t.set([0, 0], 1.0);
t.get([0, 0]);
t.shape();   t.dim();   t.size();   t.sum();
t.reshape([9]);
t.distance(other, TensorDistance::l2sq);

VectorIndex::wrap([1.0, 2.0, 3.0], TensorType::f64);

var vi = VectorIndex<MyType> { };
vi.add(vector_node, payload);
vi.search(query_vector, 10);       // Array<SearchResult<Tensor, MyType>>
```

## Runtime introspection

```gcl
Runtime::info();                   // RuntimeInfo (version, license, threads, ...)
Runtime::usage();                  // current memory/cache/io stats
Runtime::sleep(1s);
Runtime::backup_full();   Runtime::backup_delta();
Runtime::defrag();

System::exec("/bin/ls", ["-la"]);             // blocking subprocess
System::spawn("/bin/sleep", ["60"]);          // non-blocking
System::getEnv("PATH");
System::tz();                                 // host TimeZone

Task::id();   Task::parentId();
Task::expected_steps(100);   Task::add_steps(1);
Task::no_history(true);

Scheduler::add(my_fn, FixedPeriodicity { every: 5min }, null);
Scheduler::list();
Scheduler::activate(my_fn);    Scheduler::deactivate(my_fn);
```

### Periodicity types

```gcl
FixedPeriodicity { every: 30min }
DailyPeriodicity { hour: 9, minute: 0, second: 0, timezone: null }
WeeklyPeriodicity { days: [DayOfWeek::Mon, DayOfWeek::Fri], daily: DailyPeriodicity { hour: 9 } }
MonthlyPeriodicity { days: [1, 15, -1], daily: null }   // -1 = last day of month
YearlyPeriodicity { dates: [DateTuple { day: 1, month: Month::Jan }], timezone: null }
```

## Identity and permissions

```gcl
Identity::current();                          // requires authentication
Identity::current_id();                       // 0 if anonymous
Identity::login("alice", "password");         // returns session token
Identity::token(user_id, 24hour);
Identity::permissions();
Identity::has_permission("admin");
Identity::set_password("alice", "newpass");
Identity::create("bob", "user");
Identity::get_by_name("alice");
```

## Logging

```gcl
println("printed with newline");
print("pretty-printed with breaks");
pprint("printed without newline");

error("an error");
warn("a warning");
info("an info");
perf("perf log");
trace("trace log");
```

Logs are gated by the configured log level.

## Math globals

```gcl
exp(x)   log(x)   log2(x)   log10(x)
sin(x)   cos(x)   tan(x)
asin(x)  acos(x)  atan(x)
sinh(x)  cosh(x)  tanh(x)
sqrt(x)  pow(x, y)
floor(x) ceil(x)  round(x) trunc(x)
abs(x)      min(x, y)      max(x, y)        // generic; do NOT add explicit <T>
isNaN(v)
roundp(x, p)                      // round to p decimal places
```

`MathConstants::e`, `::pi`, `::pi_2`, `::pi_4`, `::ln2`, `::ln10`, `::sqrt2`, etc.

## Conversions and reflection

```gcl
println(value);                    // any?
clone(value);                      // deep clone
parseNumber("3.14");               // any (int or float)
parseHex("ff");                    // int
valueOf(Color::red);               // 0xff0000 (value attached to enum entry)

type::of(value);                   // type
type::all();                       // every type in the program
type::enum_by_offset(Color, 0);
type::enum_by_name(Color, "red");
type::enum_name(some_value);       // "red"
type::enum_offset(some_value);     // 0
```

## Assertions (for tests)

```gcl
Assert::equals(a, b);
Assert::equalsd(a, b, 1e-9);                   // float with epsilon
Assert::equalst(t1, t2, 1e-9);                 // tensor with epsilon
Assert::isTrue(v);
Assert::isFalse(v);
Assert::isNull(v);
Assert::isNotNull(v);
```

## Reading the live stdlib

Whenever a signature in this file looks stale or you need a method that isn't listed:

1. `lib/std/core.gcl` — primitives, value types, node tags, error types, time / duration.
2. `lib/std/io.gcl` — readers, writers, files, http, smtp, s3, csv, json.
3. `lib/std/runtime.gcl` — runtime, system, tasks, scheduler, identity, debug.
4. `lib/std/util.gcl` — queues, stacks, sliding windows, gaussian, quantizers, histograms, crypto, uuid.

Each method carries a `///` doc comment with its contract.
