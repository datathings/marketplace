# GreyCat standard library

The stdlib ships under `lib/std/` (split into `core.gcl`, `io.gcl`, `runtime.gcl`, `util.gcl`). Brought into scope by `@library("std", "<version>")` in `project.gcl`. This reference covers the types and functions you reach for most. For exact signatures, read the `.gcl` files directly — they carry doc comments.

## Contents

- Collections (Array, Map, Set, Tuple, Table, Buffer)
- Util collections (Queue, Stack, SlidingWindow, TimeWindow)
- Node tags (graph-persisted: node, nodeTime, nodeList, nodeIndex, nodeSet, nodeIndexSet, nodeGeo)
- Time and duration
- Geographic types (geo, GeoBox, GeoCircle, GeoPoly)
- IO (Reader / Writer, Json, Csv)
- Networking (Url, HttpRequest and HttpResponse, Smtp)
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
ys.index_of(5);                     // -1 when absent, never null
ys.remove(0);                       // returns removed value
ys.sort(SortOrder::asc);
ys.sort_by(field, SortOrder::desc); // sort by attribute of contained objects
ys.swap(0, 1);
ys.fill(10, 0);                     // size 10, default value 0
ys.range_equals(0, 4, null);        // all five elements equal null?
```

`index_of` returns `-1` when no element matches. Test membership with `xs.index_of(v) >= 0`. `xs.index_of(v) == null`
is always false and `!= null` always true. Lint reports such a comparison as `decidable-condition` only when it is
the condition of an `if` or a loop. Elsewhere, as in `return xs.index_of(v) != null;`, it passes lint.

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

GreyCat has no built-in `Set` type — model sets as `Map<T, bool>` or a custom type.

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

A `Table<T>` holds one column per field of `T`, in field order. `set_row`, `add_row` and `get_row` throw on a table with another column count, and `get_row` throws on a cell its field cannot hold, such as a `null` in a non-null field.

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

A value fetched from the graph (`idx.get(k)`, `n.resolve()`, a `for ... in nodeIndex` loop) is a **live node**: mutate its fields in place and the change persists on commit; do NOT `set` it back, which re-attaches an already-attached node and throws `object is already attached to another node`. `set` is for inserting a brand-new node. See [idioms.md](idioms.md), "Working with the graph".

While a `for` loop walks a `nodeTime`, `nodeList`, `nodeIndex`, `nodeSet`, `nodeIndexSet` or `nodeGeo`, it can set a new value at a key the node already holds. Adding or removing a key of that node (`setAt`, `set` or `add` with a new key or value, `setAll`, `remove`, `removeAt`, `removeAll`) throws `concurrent modification`, and so it does while a `nodeTimeCursor` over the node stands on one of its values. A node that holds a single value is the exception: the loop reads that value before its body runs, so the body can change the node. Collect the keys to add or remove, and change the node after the loop; other nodes can change inside it.

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

A list of values read and removed by position: every index is a position `0..size()-1`, and the values after a removed one shift down. There is no `set`: `greycat lint` reports a call to it as `unknown-member`, and `greycat build` still accepts it, so it fails only when it runs, with `type nodeList has no function named set`. Change an object value in place, or `remove` and `add`.

```gcl
var l = nodeList<float> {};
l.add(1.0);                        // append at position size()
l.add(2.0);
l.get(1);                          // 2.0; null outside 0..size()-1
l.resolve(5);                      // the last value past the end, null below 0
l.firstIndex();  l.first();        // 0, null when empty
l.lastIndex();   l.last();         // size() - 1, null when empty
l.rangeSize(0, 1);                 // positions in the range; reversed bounds are swapped, the range is clamped to the list
l.remove(0);                       // true; later values shift down, false outside the list
for (i, v in l[0..1]) { /* i counts positions; ]a..b], [a..b[ and descending l]4..2] too */ }
```

A list written by an earlier version with sparse keys (from `set`) reads by position: `get(0)` is the first stored value.

A position is not an id: it changes when an earlier value is removed. Code that keeps an index elsewhere uses a `nodeIndex<int, T>` with a counter it owns.

Two tasks appending to one list both keep their values, in merge order, and removals and appends do not conflict: the conflicts left are in [tasks.md](tasks.md#concurrent-appends-to-a-nodelist).

### nodeIndex<K, V>

Sorted index with O(log n) keyed access.

```gcl
var idx: nodeIndex<String, User> = /* ... */;
idx.set("alice", User { /* ... */ });
idx.get("alice");                  // User?
idx.search("alice", 10);           // closest-neighbor search
idx.remove("alice");
idx.size();
for (k, v in idx) { /* ... */ }
```

### nodeSet<V>

Set of unique values, generic over any `V`.

```gcl
var s = nodeSet<String> {};
s.add("red"); // true: "red" was not in the set yet
s.add("red"); // false: the set already holds it
s.contains("red"); // true
s.remove("red");
s.size();
s.removeAll();
// both loop variables bind the value
for (_, v in s) {
    println(v);
}
```

Two values are the same when they are deeply equal, the rule of a `nodeIndex` key: an `int`, `String`, enum entry or
other scalar by value, an object by its content, and a `node<T>` by reference. `null` is never an element: `add(null)`
throws, `contains(null)` is false and `remove(null)` does nothing. A loop walks the values in the order of their
encoding for a scalar `V` and in hash order otherwise. `->` on a `nodeSet` throws.

Tasks that commit to the same set at once never conflict. The merge applies what each side added and removed: a value
added on either side is present, a value removed on either side is absent. `removeAll()` removes the values the task
saw, so a value another task added at the same time stays. `fusion_by_key_last_win` changes nothing for a set, and
`no_fusion_fail_on_conflict` refuses it like any other structured node ([tasks.md](tasks.md#workers-tasks-and-jobs)).

### nodeIndexSet<K, V>

A multimap: each key holds a set of distinct values, as a `nodeIndex<K, nodeSet<V>>` would, without a node per key.

```gcl
var idx = nodeIndexSet<String, String> {};
idx.add("fruit", "apple"); // true: a new pair
idx.add("fruit", "pear"); // true: a second value under the key
idx.add("fruit", "apple"); // false: the key holds it already
idx.get("fruit"); // Array<String> {"apple", "pear"}
idx.get("nut"); // an empty Array
idx.contains("fruit", "pear"); // true
idx.size(); // 2: pairs, not keys
// one turn per pair, the values of a key one after the other
for (k, v in idx) {
    println("${k} ${v}");
}
idx.remove("fruit", "pear"); // one pair
idx.removeKey("fruit"); // the key and every value it holds
idx.removeAll();
```

A pair is a key and one of its values. Keys compare as `nodeIndex` keys do, values by deep equality as the elements of a
`nodeSet` do, and a pair is new when no pair with an equal key and an equal value is there. `null` is never a value:
`add(key, null)` throws `null can not be added to a nodeIndexSet`. A loop walks the keys in the order of a `nodeIndex`,
by encoding for a scalar key and by hash otherwise, the values of one key together and in no set order: a merge or a
removal can change it, and `get` returns them in that order. `->` on a `nodeIndexSet` throws. An object value edited in
place after `add` does not change the pair; change it with `remove(key, old)` then `add(key, new)`. `greycat lint` does
not know `nodeIndexSet` yet: it reports a loop over one as `not-iterable` and a module variable of the type as
`modvar-must-be-node-tag`, and `greycat build` and `greycat run` accept both.

A key with one value costs what it costs in a `nodeIndex`. Up to 32 values, a key keeps them in a chain that every `add`
to the key walks. Its 33rd value moves them into a `nodeSet` of their own, where `add`, `contains` and `remove` take
O(log n). The key keeps that set for the life of the index: `remove`, `removeKey` and `removeAll` empty it and leave it
in the store, and a later `add` fills it again. `NodeReader` lists the set as a `nodeSet`
([storage.md](storage.md#listing-every-node)).

Tasks that commit to the same index at once never conflict under the default strategies: the pairs merge one by one, as
the elements of a `nodeSet`. A pair added on either side is present and a pair removed on either side is absent, so two
values added under one key are both kept. `removeKey` and `removeAll` remove only the pairs the task saw: a pair another
task added at the same time stays. `fusion_by_key_last_win` changes nothing for it, and `no_fusion_fail_on_conflict`
refuses it like any other structured node ([tasks.md](tasks.md#workers-tasks-and-jobs)).

### nodeGeo<T>

Spatial index keyed by `geo`.

```gcl
var g: nodeGeo<Station> = /* ... */;
g.set(geo { 49.6, 6.1 }, station);
g.get(geo { 49.6, 6.1 });
g.resolve(geo { 49.7, 6.0 });      // closest-prior read
g.rangeSize(geo { 49, 6 }, geo { 50, 7 });
for (pos, station in g[geo { 49, 6 }..geo { 50, 7 }]) { /* every key of the box, south-west first */ }
```

A range `g[a..b]` is the box spanned by `a` and `b` (any two opposite corners): it yields exactly the keys inside, in curve order, and `rangeSize` counts the same keys. Walking from the higher code, `g[b..a]`, runs the box backwards. No `GeoBox.contains` filter is needed on the keys.

### Sampling node tags

Every node tag exposes a static `sample(...)` that returns a `Table` of values from a range, plus `info(...)` returning a per-node `NodeInfo<K>` for metadata. Signatures differ by tag:

```gcl
nodeTime::sample(refs: Array<nodeTime>, from: time?, to: time?, maxRows: int,
                 mode: SamplingMode, maxDephasing: duration?, tz: TimeZone?): Table
nodeIndex::sample(refs: Array<nodeIndex>, from: any?, maxRows: int, mode: SamplingMode): Table
nodeSet::sample(refs: Array<nodeSet>, from: any?, maxRows: int, mode: SamplingMode): Table   // one column per set
nodeIndexSet::sample(refs: Array<nodeIndexSet>, from: any?, maxRows: int, mode: SamplingMode): Table   // a row per pair
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

`maxRows` is the upper bound on rows returned regardless of mode. On a `nodeGeo` the range `from..to` is the box of the two corners (a `null` corner is the world's) and no row falls outside it: `dense` and `adaptative` pick among the keys of the box, `fixed` and `fixed_reg` lay `maxRows` grid points over the box, evenly spaced along the curve from the south-west corner, each carrying the value of the stored key before it on the curve when one follows it too. On a `nodeTime` or `nodeList`, `fixed` and `fixed_reg` lay rows at even steps from `from` to `to`, every ref of `refs` sampled at the same keys. Going forward (`from < to`) a row holds the value at or before its key, going backward the value at or after it. A row past the last key going forward, or before the first going backward, is `null`: on a `nodeTime`, it holds that edge value only while it is less than `maxDephasing` away from it (`null` is 0). A `nodeList` sample ignores `maxDephasing`. Its `from` and `to` are positions: without `to` it stops at the last position of the shortest list in `refs`, an empty list there gives an empty table, `fixed_reg` gives the same rows as `fixed` (a list has no gaps), and with no `to` and a `from` past the shortest list's end it walks backward. `nodeList::info` reports the range `0..size()-1`, with a `null` `from` for an empty list.

## Time and duration

### time

```gcl
time::now()                        // wall-clock time
time::current()                    // contextual time (defaults to time::max outside any `at (t) { }` block)
time::min   time::max
time::new(epoch, DurationUnit::milliseconds)
time::parse("2025-05-22", "%Y-%m-%d")
time::parse("2025-05-22T16:47:42+02:00", null)   // null format: ISO 8601; nothing may follow the text but whitespace
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

A `geo` latitude is clamped to ±85.05 (the Web Mercator range), a longitude to [-180, 180). `circle.sw()` / `ne()` bound the whole circle; a circle crossing the antimeridian or holding a pole gets a box spanning every longitude, so a box prefilter followed by `circle.contains` misses nothing.

A `GeoBox` spans from `sw.lng()` going east to `ne.lng()`, so a box whose `sw.lng()` is above its `ne.lng()` crosses the antimeridian: `contains`, `intersects`, `intersection`, `union`, `expand`, `center`, `width` and `split` all take it as the narrow box around longitude 180, not the wide one between. A `nodeGeo` range takes its two keys as opposite corners whatever their order, so scan such a box one lobe at a time:

```gcl
var fiji = GeoBox { sw: geo { -20.0, 175.0 }, ne: geo { -15.0, -178.0 } };
fiji.crosses_antimeridian(); // true
for (_, lobe in fiji.lobes()) { // [175..180], then [-180..-178]
    for (g, v in idx[lobe.sw..lobe.ne]) {
        // ...
    }
}
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
| `BinReader`                       | Raw `i32` / `i64` / `f32` / `f64` / `Tensor` reads |

`GcxWriter<T>` / `GcxReader<T>` write and read `.gcx`, a binary format that does not depend on the ABI: objects carry their field names and the type, field and enum entry names sit once in a dictionary at the end of the file. Choose it over GCB for a file that must survive changes to the program's types or be read by another tool; GCB is smaller and faster but only reads back into the same types. `GcxWriter` is not a `Writer`: it has `write(v)` and `close()`, no `flush()`, and the file is readable only once `close()`d. Read back into changed types, a removed field is skipped and a missing nullable field reads as null. Layout and rules: `lib/std/io.md` (GcxWriter / GcxReader).

`Stream<T>`, an append-only file of records that concurrent tasks write and a callback drains in batches, is covered in [streams.md](streams.md).

### File

`File`, `FileWalker` and the per-user `files/` folders are covered in [files.md](files.md).

### Json

```gcl
Json::to_string(any_value);            // serialize
Json<MyType>::parse(json_str);         // typed deserialize
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

`reader.last_line()` returns the row the last `read()` parsed as the file holds it, quotes and separators included, without its line break: copy a row as it is, or quote it in an error.

**Common CSV pitfalls:**

- `CsvFormat.separator` is `char?` (one byte) — write `','`, not `","`. Same for `string_delimiter`, `decimal_separator`, `thousands_separator`.
- Without `@format` on a `time` field, parsing falls back to ISO 8601 / epoch-ms. Mismatched formats throw at parse time.
- `geo` is two columns wide (lat then lng). Account for that when matching column order.
- A greedy `Array<U>` field MUST be the last attribute — there's no way to delimit where it stops otherwise.
- Into a nullable column, an unquoted cell that is empty, blank or `null` (any case) reads as `null`, even with `trim: false`; quoted, it is the text. `CsvWriter` quotes such Strings, so its own files round-trip `""`, `" "` and `"null"`.

## Networking

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

### HttpRequest and HttpResponse

`runtime::HttpRequest` is the whole request of a call: `method`, `uri`, `query`, `headers`, and the body as `body`, or
as `body_file` when it is larger than `--max_args_memory`. An `@expose`d function taking it as its only parameter
receives it, which is how a signed webhook checks the bytes it was sent. `runtime::HttpResponse` (`status`,
`content_type`, `headers`, `body` or `body_file`) is what a function returning it writes as its response: a text
answer, a download, a redirect. Both are in [http.md](http.md#reading-the-request-writing-the-response).

### Task class

`Task::task_class()` answers the class the current task or job runs in (it raises outside a task, never `null`);
`spawn` and `await` jobs take a `task_class`:

```gcl
Task::task_class();                                         // TaskClass
spawn(Job { function: f, task_class: TaskClass::large });   // also on each await job
```

### Smtp

`Smtp` for sending email (with `Email`, `SmtpMode`, `SmtpAuth`).

See `lib/std/io.gcl` for full signatures.

## Crypto and UUID

```gcl
Crypto::sha1(content);             Crypto::sha1hex(content);
Crypto::sha256(content);           Crypto::sha256hex(content);
Crypto::sha256_hmac_hex(input, key);
Crypto::equals_constant_time(a, b);  // compare digests with this, never ==
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

`Tensor` (numeric arrays of up to 8 dimensions), `TensorDistance` and the `VectorIndex` nearest-vector search are in
[tensors.md](tensors.md).

## Runtime introspection

```gcl
Runtime::info();                   // RuntimeInfo (version, workers and cache per class, ...)
Runtime::sleep(1s);
Runtime::backup_full();   Runtime::backup_delta();
Runtime::defrag();
Runtime::store_stats();             // StoreStats: live blocks and bytes, by type and by world

World::create("what-if", WorldMode::overlay, null);   // branch of the store, see storage.md § Worlds
World::list();   World::id("what-if");   World::reset("what-if");   World::remove("what-if");
World::current();                             // the calling task's world, "main" for main
var r = WorldChangesReader {};                // the nodes the current world modified: can_read(), read(): any, available()
var d = WorldDiffReader { node: n };          // n's keys that differ from the base: read(): WorldDiff {kind, key, base, world}

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
Identity::token(user_id, 24hour, "ci");     // a token labeled "ci", null for the --validity of the server
Identity::tokens("alice", 0, 100);            // the tokens of alice that have not expired, a page at a time
Identity::revoke_token("alice", token_id);    // false when alice has no such token
Identity::revoke_tokens("alice", true);       // revokes all but the token of the call, returns the count
Identity::logout();                           // revokes the token of the call and clears the cookie
Identity::permissions();
Identity::has_permission("admin");
Identity::set_password("alice", "newpass");
Identity::create("bob", "user");              // an existing bob gets the role and keeps its password
Identity::create_with_password("carol", "user", "pw"); // new identity only, throws if carol exists
Identity::set_password("bob", "");            // removes bob's password, so bob cannot sign in
Identity::get_by_name("alice");
Identity::grant("alice", "bob", "files:read");  // bob may read files/alice/
Identity::revoke("alice", "bob", "files:read"); // bob no longer reads files/alice/
Identity::grant_permission("alice", "api", "files:read"); // callers holding api may read files/alice/
Identity::grants("alice", 0, 100);            // the grants of alice, a page at a time
```

`admin` functions such as `create`, `get_by_name`, `grant` and `remove` are only permission-checked over HTTP. From GCL any caller runs them. See [identity.md, Grants and per-user files](identity.md#grants-and-per-user-files).

## Logging

```gcl
println("printed with newline");
print("pretty-printed with breaks");
pprint("printed without newline");

error("an error");
warn("a warning");
info("an info");
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
2. `lib/std/io.gcl` — readers, writers, files, url, smtp, s3, csv, json.
3. `lib/std/runtime.gcl` — runtime, system, tasks, scheduler, identity, debug.
4. `lib/std/util.gcl` — queues, stacks, sliding windows, gaussian, quantizers, histograms, crypto, uuid.

Each method carries a `///` doc comment with its contract.
