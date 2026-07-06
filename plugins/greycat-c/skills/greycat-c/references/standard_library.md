# GreyCat Standard Library (std)

## Overview

The GreyCat Standard Library provides essential data structures, I/O operations, runtime features, and utilities for GCL applications. Documentation tracks GreyCat SDK **8.0**. The library is organized into four modules:

- **core** - Fundamental types and data structures (primitives, time/date, nodes, tensors, geo, error handling, math)
- **runtime** - Scheduled/recurring tasks (Scheduler + periodicities), background job processing (Task/Job/await), application logging, identity/authentication, system-information queries, OpenAPI export, MCP server endpoints
- **io** - File I/O & discovery, data serialization (Gcb/Json/Csv/Text), HTTP API calls, email/SMTP notifications, S3 object storage
- **util** - Data structures (Queue/Stack/SlidingWindow/TimeWindow), statistical analysis, data binning (quantizers), testing (Assert), monitoring (ProgressTracker), cryptography, UUID

> Signatures below are copied verbatim from the `.gcl` source. `native` means the implementation is provided by the runtime. `@expose` marks a function reachable over the API; `@permission("...")` is the required permission. `private` members are omitted.

## Table of Contents

### Modules
- [std::core - Core Types](#core-module-stdcore)
  - [Numbers & Strings](#numbers--strings) - int, float, String, Buffer, Chars
  - [Time & Duration](#time--duration) - time, duration, Date
  - [Containers](#containers) - Array, Map, Tuple, Table
  - [Nodes](#nodes) - node, nodeTime, nodeList, nodeIndex, nodeGeo, VectorIndex
  - [Tensors](#tensors) - Tensor, TensorType, TensorDistance
  - [Geospatial Types](#geospatial-types) - geo, GeoCircle, GeoPoly, GeoBox
  - [Reflection](#reflection) - type, field
  - [Errors](#errors) - Error, ErrorFrame, ErrorCode
  - [Math & Free Functions](#math--free-functions)
  - [Enumerations](#core-enumerations)

- [std::runtime - Runtime Features](#runtime-module-stdruntime)
  - [Scheduler & Periodicities](#scheduler--periodicities)
  - [Task & Job Management](#task--job-management)
  - [Logging](#logging)
  - [System Information](#system-information)
  - [Identity & Security](#identity--security) - Identity, IdentityGrant, IdentityGrantType
  - [License Management](#license-management)
  - [OpenAPI Integration](#openapi-integration)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)

- [std::io - Input/Output](#io-module-stdio)
  - [Readers & Writers](#readers--writers) - Writer, Reader, GcbWriter/Reader, TextWriter/Reader, BinReader
  - [JSON I/O](#json-io) - JsonWriter, JsonReader, Json
  - [XML I/O](#xml-io) - XmlReader
  - [CSV I/O](#csv-io) - CsvWriter, CsvReader, CsvFormat, Csv analysis
  - [File System](#file-system) - File, FileWalker
  - [HTTP Client](#http-client) - Http, HttpRequest, HttpResponse, HttpMethod, Url
  - [Email](#email) - Email, Smtp
  - [S3 Object Storage](#s3-object-storage) - S3, S3Bucket, S3Object, S3BasicCredentials

- [std::util - Utilities](#util-module-stdutil)
  - [Collections](#collections) - Queue, Stack, SlidingWindow, TimeWindow
  - [Statistics](#statistics) - Gaussian, Histogram, HistogramStats, GaussianProfile
  - [Quantizers](#quantizers) - Quantizer, Linear, Log, Custom, Multi
  - [Utilities](#utilities) - Random, Assert, ProgressTracker, Crypto, Uuid

### Additional Sections
- [Usage Guidelines](#usage-guidelines)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

---

## Core Module (std::core)

### Numbers & Strings

#### int / float
`int` is signed 64-bit; `float` is IEEE 754 64-bit.
```gcl
// int::min, int::max, float::min, float::max
var s = int::to_string(1234567, '_');                 // "1_234_567"
var f = float::to_string(3.14159, '.', null, 2, false);
```

#### String
Immutable UTF-8 string with rich query/transform/similarity methods.
```gcl
// native fns: compare, startsWith, endsWith, contains, get(int): char,
//   indexOf(char), indexOfFrom, lastIndexOf, slice(from, to), trim,
//   lowercase, uppercase, size, replace(s, s2), indexOfString,
//   indexOfStringFrom, split(c: char): Array<String>, chars(): Chars,
//   jaro, jarowinkler (similarity), levenshtein, nfkd_casefold
var parts = "a,b,c".split(',');
var d = "kitten".levenshtein("sitting");   // 3
```

#### Chars
```gcl
type Chars {
  codepoints: Array<char>;
  // native fn to_string(): String;
}
```

#### Buffer
Mutable binary blob builder.
```gcl
// native fns: add(any?), add_and_pad(v, max, pad: char), get(i): char,
//   size(): int, clear(), toString(): String
var b = Buffer {};
b.add("hello");
println(b.toString());
```

### Time & Duration

#### duration
```gcl
// static native fn new(v: int, unit: DurationUnit): duration
// static native fn newf(v: float, unit: DurationUnit): duration
// native fns: to(unit): int, tof(unit): float, add(value, unit), subtract(value, unit)
var d = duration::new(5, DurationUnit::minutes);
```

#### time
Universal precise instant. Created with `time::new(...)` or a literal (`'2025-05-06T16:47:42Z'`).
```gcl
// static fields: time::min, time::max
// static: current(), now(), new(epoch, unit), parse(value, format?),
//   isLeap(year), totalDaysInYear(year), totalDaysInMonth(month, year)
// instance: to(unit), floor(unit), calendar_add(value, unit: CalendarUnit, tz?),
//   calendar_floor(unit: CalendarUnit, tz?), calendar_ceiling(unit, tz?),
//   dayOfYear(tz?), dayOfWeek(tz?), weekOfYear(tz?), daysInMonth(tz?),
//   startOfWeek(tz?), endOfWeek(tz?), toDate(tz?): Date, format(format, tz?)
var t = time::now();
var d: Date = t.toDate(null);
var s = t.format("%+", null);                 // ISO8601
var tomorrow = t.calendar_add(1, CalendarUnit::day, null);
```

#### Date
A moment in the human calendar. Built from a `time` (not via `Date::now()`).
```gcl
type Date {
  year: int; month: int; day: int;     // month 1-12, day 1-31
  hour: int; minute: int; second: int;
  microsecond: int;                    // 0..999_999
  // static native fn from_time(time: time, tz: TimeZone?): Date
  // static native fn parse(value: String, format: String?): Date
  // native fn to_time(tz: TimeZone?): time
  // native fn to_nearest_time(tz: TimeZone): time
}
var date = Date::from_time(time::now(), TimeZone::"Europe/Luxembourg");
var back = date.to_time(null);
```

### Containers

#### Array<T>
```gcl
// native fns: fill(size, default), add(value), add_all(values), get(i), set(i, value),
//   swap(i, j), sort(order: SortOrder), sort_by(field, order), size(),
//   index_of(value), remove(i): T, remove_first(): T, remove_last(): T,
//   set_capacity(value), range_equals(start, end, v)
```

#### Map<K, V>
```gcl
// native fns: set(key, value): V, get(key): V?, contains(key): bool,
//   remove(key), size(): int, values(): Array<V>
```

#### Tuple<T, U>
```gcl
type Tuple<T, U> { x: T; y: U; }
var pair = Tuple<String, int> { x: "count", y: 42 };
```

#### Table<T>
Two-dimensional structure of values of any type.
```gcl
// native fns: cols(), rows(), set_cell(row, col: any, value), get_cell(row, col: any),
//   set_row(row, v: T), get_row(row): T, add_row(v: T), remove_row(row),
//   sort(col: any, order: SortOrder), sort_by(col: any, cell_field: field, order),
//   init(rows, cols)
// @expose @permission("debug") static native fn applyMappings(table, mappings: Array<TableColumnMapping>): Table
type TableColumnMapping { column: int; extractors: Array<any>; }
```

### Nodes

Persistent graph node primitives. All are `native` and stored to disk.

- **`node<T>`** - singleton boxed value. `resolve(): T`, `set(value: T)`, static `resolve_all(n: Array<node?>): Array<any?>`.
- **`nodeTime<T>`** - temporal series. `resolve()`, `resolveAt(time)`, `resolveAtWithin(time, duration)`, `resolveTimeAt`, `resolveTimeValueAt`, `getAt`, `removeAt`, `setAt(time, value)`, `setAll(time, values, delta)`, `rangeSize(from, to)`, `size`, `firstTime`/`first`/`lastTime`/`last`, `prev(time)`/`next(time)`, `removeAll`, static `sample(...)`, static `info(...)`.
- **`nodeList<T>`** - sparse indexed list. `get(index)`, `set(index, value)`, `add(value)`, `resolve(index)`, `resolveEntry(index): Tuple<int,T>?`, `remove(index): bool`, `firstIndex`/`first`/`lastIndex`/`last`, etc.
- **`nodeIndex<K, V>`** - keyed map, O(log n). `set(key, value)`, `get(key)`, `remove(key)`, `search(key, max): Array<SearchResult<K,V>>`, static `search_closest`, static `sample`/`info`.
- **`nodeGeo<T>`** - geo-indexed series. `get(geo)`, `set(geo, value)`, `resolve(geo)`, `rangeSize(from: geo, to: geo)`, `firstIndex`/`lastIndex` (SW/NE), etc.
- **`nodeTimeCursor<T>`** (`@volatile`) - iterator over a nodeTime (`first`, `last`, `next`, `previous`, `lessOrEq(t)`, `skip_values`, `skip_duration`, `currentTime`, `current`).

```gcl
type NodeInfo<T> { size: int; from: T?; to: T?; }
type SearchResult<K, V> { key: K; value: V; distance: float; }   // @volatile
```

#### VectorIndex<T>
Persistent vector store with distance-based nearest-neighbour search.
```gcl
// native fn add(vector: node<Tensor>, value: T)
// native fn search(query: Tensor, wanted: int?): Array<SearchResult<Tensor, T>>
// native fn size(): int
// native static fn wrap(values: Array<float>, type: TensorType): Tensor
```

### Tensors

#### Tensor
N-dimensional numeric structure.
```gcl
// native fns include: init(etype: TensorType, shape: Array<int>), get(pos), set(pos, value),
//   getImag/setImag, add(pos, value), append(value), fill(value), setCapacity(value),
//   sum(): float, size(): int, type(): TensorType, shape(): Array<int>, dim(): int,
//   initPos(): Array<int>, incPos(pos): bool, reshape(shape), scale(value, inPlace: float): Tensor,
//   slice(pos, size): Tensor, slide(steps), copyElementsFrom(...), distance(v, d: TensorDistance): float,
//   reset(), toTable(): Table, toString(): String,
//   to_complex_tensor(), get_real_tensor(), get_imaginary_tensor(),
//   get_absolute_tensor(), get_phase_tensor(inDegrees: bool)
// static native fn wrap_1d(etype: TensorType, values: Array): Tensor
```

```gcl
enum TensorType { i32(4); i64(8); f32(4); f64(8); c64(8); c128(16); }
enum TensorDistance { euclidean; l2sq; cosine; }   // euclidean aka l2
```

### Geospatial Types

#### geo
A precise location given by latitude/longitude. Created with a literal: `geo{49.5964, 6.1287}` (NOT `geo::new`).
```gcl
// static: geo::min, geo::max, distance_to_segment(point, a, b): float, meters_per_deg_lon(lat): float
// instance: lat(): float, lng(): float, distance(value: geo): float,
//   bearing(value: geo): float, toString(): String, toGeohash(): String
var hq = geo{ 49.59640167862028, 6.128662935665594 };
var meters = hq.distance(geo{ 49.6, 6.13 });
```

#### GeoCircle
```gcl
type GeoCircle {
  center: geo;
  radius: float;        // meters
  // native fn contains(point: geo): bool
  // native fn sw(): geo; native fn ne(): geo;
}
```

#### GeoPoly
```gcl
type GeoPoly {
  points: Array<geo>;
  // native fns: contains(point): bool, sw(), ne(), bbox(): GeoBox, is_closed(): bool,
  //   perimeter(): float, area(): float, centroid(): geo?, mean_centroid(): geo?,
  //   interpolate(step: float): Array<geo>
}
```

#### GeoBox
```gcl
type GeoBox {
  sw: geo;
  ne: geo;
  // static native fn from_point(center: geo, radius: float): GeoBox
  // native fns: contains(point): bool, intersects(other): bool, union(other): GeoBox,
  //   intersection(other): GeoBox?, expand(margin: float): GeoBox, center(): geo,
  //   width(): float, height(): float, split(rows, cols): Array<GeoBox>
}
```

### Reflection

```gcl
native type type {
  // static: all(): Array<type>, of(v: any?): type, enum_by_offset/name<T>, enum_name(x), enum_offset(x),
  //   fields_set_from(dst, src, clone), field_set(target, offset, value), field_get(target, offset)
  // instance: nb_fields(): int, fields(): Array<field>, field_by_name(name): field?,
  //   field_offset_by_name(name): int?, nb_enum_values(): int, enum_values(): Array,
  //   has_parent(parent): bool, create_instance(): any?
}
native type field {
  // native fns: name(): String, type(): type, is_nullable(): bool, offset(): int
}
```

### Errors

```gcl
type Error      { message: String?; stack: Array<ErrorFrame>; }
type ErrorFrame { module: String?; function: String; line: int; column: int; }

enum ErrorCode {
  none(0); interrupted(1); await(2); timeout(6); forbidden(7); runtime_error(8);
}
```

### Math & Free Functions

Module-level functions (called without a type prefix):
```gcl
// logging (write to the task/console log if level allows):
println(v); pprint(v); print(v); error(v); warn(v); info(v); perf(v); trace(v);
// values:
valueOf(en: any): any?;        // value of an enum field
parseNumber(value: String): any;   // int or float, throws on failure
parseHex(value: String): int;
clone<T>(v: T): T;
// math: exp, cos, sin, tan, sqrt, floor, ceil, cosh, sinh, tanh, acos, asin, atan,
//   atan2(y, x), log, log2, log10, pow(x, y), trunc, round, abs<T>, min<T>, max<T>,
//   isNaN(v: float): bool, roundp(x: float, p: int): float
type MathConstants { /* static: e, log_2e, log_10e, ln2, ln10, pi, pi_2, pi_4, m1_pi,
                        m2_pi, m2_sqrt_pi, sqrt2, sqrt1_2 */ }
```

### Core Enumerations

#### FloatPrecision
`p1`(1.0), `p10`(0.1), `p100`(0.01), ... down to `p10000000000`(1e-10).

#### CalendarUnit
`year`(0), `month`(1), `day`(2), `hour`(3), `minute`(4), `second`(5), `microsecond`(6).

#### DurationUnit
`microseconds`(1), `milliseconds`(1e3), `seconds`(1e6), `minutes`(60e6), `hours`(3600e6), `days`(86400e6).

#### SamplingMode
`fixed`(0), `fixed_reg`(1), `adaptative`(2), `dense`(3).

#### SortOrder
`asc`, `desc`.

#### TensorType / TensorDistance
See [Tensors](#tensors).

#### TimeZone
Large enum of IANA timezone names (e.g. `TimeZone::"UTC"`, `TimeZone::"Europe/Luxembourg"`, `TimeZone::"America/New_York"`).

## Runtime Module (std::runtime)

### Scheduler & Periodicities

`Scheduler` manages recurring tasks; each function may have at most one scheduled task (identified by function-pointer equality, so re-adding the same function replaces it).
```gcl
type Scheduler {
  // @expose @permission("admin"):
  // static native fn add(function: function, periodicity: Periodicity, options: PeriodicOptions?)
  // static native fn list(): Array<PeriodicTask>
  // static native fn find(function: function): PeriodicTask?
  // static native fn activate(function: function): bool
  // static native fn deactivate(function: function): bool
}
```

#### Basic Usage
```gcl
fn backup_database() { /* ... */ }

fn schedule_backups() {
  Scheduler::add(backup_database, DailyPeriodicity { hour: 2 }, null);  // 2 AM daily
}
```

#### Periodicity Types
`Periodicity` is an `abstract type`; use a concrete subtype.

**FixedPeriodicity** - fixed interval
```gcl
type FixedPeriodicity extends Periodicity { every: duration; }
FixedPeriodicity { every: 5min }
```

**DailyPeriodicity** - a time-of-day (all fields optional, default midnight)
```gcl
type DailyPeriodicity extends Periodicity {
  hour: int?; minute: int?; second: int?; timezone: TimeZone?;
}
DailyPeriodicity { hour: 14, minute: 30 }   // 2:30 PM
```

**WeeklyPeriodicity** - selected weekdays at a `daily` time
```gcl
type WeeklyPeriodicity extends Periodicity {
  days: Array<DayOfWeek>;
  daily: DailyPeriodicity?;
}
WeeklyPeriodicity {
  days: [DayOfWeek::Mon, DayOfWeek::Fri],
  daily: DailyPeriodicity { hour: 9 }
}
```

**MonthlyPeriodicity** - days of month (negatives count from end) at a `daily` time
```gcl
type MonthlyPeriodicity extends Periodicity {
  days: Array<int>;            // 1..31, or -1..-31 from end of month
  daily: DailyPeriodicity?;
}
MonthlyPeriodicity {
  days: [1, 15, -1],
  daily: DailyPeriodicity { hour: 9, minute: 30 }
}
```

**YearlyPeriodicity** - specific calendar dates each year
```gcl
type YearlyPeriodicity extends Periodicity {
  dates: Array<DateTuple>;
  timezone: TimeZone?;
}
type DateTuple { day: int; month: Month; }
YearlyPeriodicity {
  dates: [ DateTuple { day: 1, month: Month::Jan }, DateTuple { day: 25, month: Month::Dec } ],
  timezone: TimeZone::"Europe/Luxembourg"
}
```

#### PeriodicOptions
```gcl
type PeriodicOptions {
  immediate: bool?;       // run immediately, default true
  activated: bool?;       // default true
  start: time?;           // default time::now()
  max_duration: duration?;// force-cancel after this; null = unlimited
}
Scheduler::add(health_check, FixedPeriodicity { every: 5min },
  PeriodicOptions { start: time::now() + 1hour, max_duration: 30s });
```

#### PeriodicTask & Management
```gcl
type PeriodicTask {
  function: function;
  periodicity: Periodicity;
  options: PeriodicOptions;
  is_active: bool;
  next_execution: time;
  execution_count: int;
}

var ptask = Scheduler::find(health_check);          // PeriodicTask?
Scheduler::deactivate(backup_database);             // bool
Scheduler::activate(backup_database);               // bool
var all = Scheduler::list();                        // Array<PeriodicTask>
```

#### DayOfWeek / Month enums
```gcl
enum DayOfWeek { Mon(0); Tue(1); Wed(2); Thu(3); Fri(4); Sat(5); Sun(6); }
enum Month { Jan(0); Feb(1); Mar(2); Apr(3); May(4); Jun(5);
             Jul(6); Aug(7); Sep(8); Oct(9); Nov(10); Dec(11); }
```

### Task & Job Management

#### Task
```gcl
enum TaskStatus {
  empty; waiting; running; await; cancelled;
  error; ended; ended_with_errors; breakpoint;
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
  // static native fns:
  //   expected_steps(total_expected_steps: int)
  //   add_steps(steps: int)
  //   parentId(): int
  //   id(): int
  //   no_history(v: bool)
  //   running(): Array<Task>             (@expose @reserved)
  //   history(offset, max): Array<Task>  (@expose @reserved)
  //   cancel(task_id): bool              (@expose @reserved)
  //   is_running(task_id): bool          (@expose)
  //   live(ids): Array<bool>             (@expose; one bool per id, false for unknown/inaccessible)
  //   tasks(ids): Array<Task?>           (@expose; null entries for unknown/inaccessible tasks)
}
```

#### Job<T> & await
```gcl
type Job<T> {
  function: core::function;
  arguments: Array<any?>?;
  // native fn result(): T
}
enum MergeStrategy { strict; first_wins; last_wins; }
// native fn await(jobs: Array<Job>, strategy: MergeStrategy)
```

### Logging

Logging is done with the **module-level functions** `error(v)`, `warn(v)`, `info(v)`, `perf(v)`, `trace(v)` (see [Math & Free Functions](#math--free-functions)). `Log` itself is a `@volatile` data record produced for log parsing — not a callable namespace.
```gcl
enum LogLevel { error; warn; info; perf; trace; }

@volatile
type Log {
  level: LogLevel;
  time: time;
  user_id: int?;
  id: int?;
  id2: int?;
  src: function?;
  data: any?;
}

info("Application started");
error("Failed to connect: ${error_message}");
trace("Processing item ${id}");
```

#### LogDataUsage / RuntimeUsage
```gcl
@volatile
type LogDataUsage {
  read_bytes: int; read_hits: int; read_wasted: int;
  write_bytes: int; write_hits: int;
  cache_bytes: int; cache_hits: int;
}

type RuntimeUsage {
  os_total_bytes: int; os_used_bytes: int;
  proc_virt_bytes: int; proc_res_bytes: int; proc_shr_bytes: int;
  global_memory: int; memory_drift: int;
  workers: Array<WorkerUsage>; zones: Array<ZoneUsage>;
  // static native fn collect()
}
```

### System Information

#### Runtime / RuntimeInfo
```gcl
@volatile
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

type Runtime {
  // @expose @permission("debug"): info(): RuntimeInfo, usage(): RuntimeUsage, root(): any
  // @expose @reserved @permission("api"): abi()
  // static native fns: sbi_tree(node: any?): Array<Tuple<int, int>>?, sleep(d: duration), backup_delta(), defrag(),
  //   on_files_put(handler: function?)
  // @expose @permission("admin"): backup_full()
}

var info = Runtime::info();
println("Version: ${info.version}, arch: ${info.arch}");
Runtime::sleep(1s);
```

#### System
```gcl
type System {
  // static native fn exec(path: String, params: Array<String>): String   // returns stdout
  // static native fn spawn(path: String, params: Array<String>): ChildProcess
  // static native fn tz(): TimeZone
  // static native fn getEnv(key: String): String?
  // @expose @permission("admin") static native fn get_all_envs(): Array<Tuple<String, String?>>
}

var out  = System::exec("/usr/bin/ls", ["-la"]);
var tz   = System::tz();
var home = System::getEnv("HOME");        // String?
```

#### ChildProcess
```gcl
type ChildProcess {
  // private pid: int;
  // native fn wait(): ChildProcessResult
  // native fn kill()
}
type ChildProcessResult { code: int; stdout: String; stderr: String; }

var child  = System::spawn("/usr/bin/sleep", ["10"]);
var result = child.wait();    // ChildProcessResult
```

### Identity & Security

The 8.0 security model is `Identity` / `IdentityGrant` / `IdentityGrantType` (the old `User` / `UserGroup` / `SecurityPolicy` / `OpenIDConnect` types are gone). Anonymous callers are id `0`.

```gcl
@volatile
type Identity {
  id: int;            // 0 = anonymous/public user
  name: String;       // unique login name
  role: String;       // role name -> permission flags at login
  grants: Array<IdentityGrant>;

  // @expose @permission("public") static native fn current_id(): int   (cheap, lock-free)
  // @expose                       static native fn current(): Identity  (requires auth)
  // @expose @permission("admin")  static native fn get_by_id(id: int): Identity?
  // @expose @permission("admin")  static native fn get_by_name(name: String): Identity?
  // @expose @permission("admin")  static native fn all(): Array<Identity>
  // @expose @permission("admin")  static native fn create(name: String, role: String): Identity
  // @expose                       static native fn token(id: int, ttl: duration?): String
  // @expose @permission("public") static native fn login(login: String, password: String): String
  // @expose                       static native fn logout()
  // @expose @permission("public") static native fn permissions(): Array<String>
  //                               static native fn has_permission(permission: String): bool
  // @expose                       static native fn set_password(name: String, pass: String): bool
  // @expose                       static native fn set_grants(name: String, grants: Array<IdentityGrant>)
}

var me        = Identity::current();
var caller_id = Identity::current_id();   // 0 if unauthenticated
var token     = Identity::login("alice", "secret");
```

#### IdentityGrant & IdentityGrantType
```gcl
enum IdentityGrantType { read, write, read_write, none }

@volatile
type IdentityGrant {
  name: String;
  grant: IdentityGrantType;
}
```

### License Management

```gcl
enum LicenseType { community; enterprise; testing; }

type License {
  name: String?;        // associated username
  start: time;          // start of validity
  end: time;            // end of validity
  company: String?;     // associated company
  max_memory: int;      // max allowed memory in MB
  extra_1: int?;
  extra_2: int?;
  type: LicenseType?;
}
```

### OpenAPI Integration

`OpenApi` exposes a single static method that exports all exposed functions as an OpenAPI v3 spec.
```gcl
type OpenApi {
  // @expose @permission("api") static native fn v3(): OpenApiV3
}
var spec = OpenApi::v3();
```

### Model Context Protocol (MCP)

`@expose`d module-level handlers implement an MCP server endpoint. All are `@permission("public")` (no auth required):
```gcl
@expose("initialize")  @permission("public") native fn mcp_initialize(params: McpInitializeParams): McpInitializeResult;
@expose("tools/list")  @permission("public") native fn mcp_tools_list(params: McpToolsListParams?): McpToolsListResult;
@expose("tools/call")  @permission("public") native fn mcp_tools_call(params: McpToolsCallParams): any;
@expose("tasks/get")    @permission("public") native fn mcp_tasks_get(params: McpTasksGetParams): any;
@expose("tasks/result") @permission("public") native fn mcp_tasks_result(params: McpTasksResultParams): McpToolsCallResult;
@expose("tasks/list")   @permission("public") native fn mcp_tasks_list(params: McpTasksListParams?): McpTasksListResult;
@expose("tasks/cancel") @permission("public") native fn mcp_tasks_cancel(params: McpTasksCancelParams): McpTask;
```
Supporting `@volatile` types include: `McpInitializeParams` / `McpInitializeResult`, `McpClientCapabilities` / `McpServerCapabilities` (with prompts/resources/tools/tasks sub-capabilities), `McpImplementation`, `McpTool` (+ `McpToolExecution`), `McpToolsCallResult`, content blocks `McpTextContent` / `McpImageContent` / `McpAudioContent` / `McpResourceContent`, and task types `McpTask` / `McpTaskCreateParams` / the `McpTasks*Params`/`*Result` family. Enums: `McpContentType` (text, image, audio, resource_link, resource), `McpRole` (user, assistant), `McpPriority` (MostImportant(1), LeastImportant(0)), `McpTaskSupport` (forbidden, required, optional), `McpTaskStatus` (working, input_required, completed, failed, cancelled).

## I/O Module (std::io)

### Readers & Writers

`Writer<T>` and `Reader<T>` are `abstract type`s; use a concrete implementation. `Writer` has `write(v: T)` and `flush()`; `Reader` has `read(): T`, `can_read(): bool`, `available(): int` and a public `pos: int?`.

#### GcbWriter<T> / GcbReader<T>
GreyCat Binary (ABI-encoded) serialization.
```gcl
var writer = GcbWriter<MyType> { path: "/data/output.gcb" };
writer.write(my_object);
writer.flush();

var reader = GcbReader<MyType> { path: "/data/output.gcb" };
while (reader.can_read()) { process(reader.read()); }
```

#### TextWriter<T> / TextReader
UTF-8 text, line oriented. `TextWriter` adds `writeln(v: T)`. `TextReader` reads one line per `read()` (trims trailing `\r` and null bytes; throws when exhausted).
```gcl
var writer = TextWriter<String> { path: "/logs/output.txt" };
writer.writeln("Line 1");
writer.flush();

var reader = TextReader { path: "/logs/output.txt" };
while (reader.can_read()) { println(reader.read()); }
```

#### BinReader
Typed binary reader (not generic).
```gcl
type BinReader {
  pos: int?;
  // native fns: read_i32(): int, read_i64(): int, read_f32(): float, read_f64(): float,
  //   read_tensor(etype: TensorType, shape: Array<int>): Tensor,
  //   can_read(): bool, available(): int
}
```

### JSON I/O

#### JsonWriter<T> / JsonReader<T>
NDJSON (one JSON value per line). `JsonWriter` adds `writeln(v: T)`.
```gcl
var writer = JsonWriter<Person> { path: "/data/people.json" };
writer.writeln(person1);
writer.flush();

var reader = JsonReader<Person> { path: "/data/people.json" };
while (reader.can_read()) { process(reader.read()); }
```

#### Json<T>
Parse / serialize JSON strings.
```gcl
type Json<T> {
  // native fn parse(data: String): T
  // static native fn to_string(value: any?): String
}
var obj  = Json<Person> {}.parse("{\"name\":\"Alice\",\"age\":30}");
var text = Json::to_string(obj);
```

### XML I/O

#### XmlReader<T>
```gcl
var reader = XmlReader<Config> { path: "/config/settings.xml" };
while (reader.can_read()) { apply_config(reader.read()); }
```

### CSV I/O

#### CsvWriter<T> / CsvReader<T>
Auto-generates headers from `T`'s fields when `format.header_lines > 0`. `CsvWriter` also has `write_line(line: String)`. `CsvReader` has `last_line(): String?` and `set_path(path: String)` (re-uses buffers across files).
```gcl
var format = CsvFormat { header_lines: 1, separator: ',', string_delimiter: '"' };

var writer = CsvWriter<Employee> { path: "/data/employees.csv", format: format };
writer.write(emp1);     // headers lazily written on first write
writer.flush();

var reader = CsvReader<Employee> { path: "/data/employees.csv", format: format };
while (reader.can_read()) { process(reader.read()); }
```

#### CsvFormat
```gcl
type CsvFormat {
  header_lines: int?;          // null/0 = none
  separator: char?;
  string_delimiter: char?;     // e.g. '"'
  decimal_separator: char?;
  thousands_separator: char?;
  trim: bool?;                 // default false
  format: String?;             // date format, default ISO8601 / epoch ms
  tz: TimeZone?;
  strict: bool?;               // strict null checking
  nearest_time: bool?;         // fall back to nearest valid time, default false
}
```

#### CSV Analysis
```gcl
type Csv {
  // @expose static native fn generate(stats: CsvStatistics): String
  // @expose static native fn analyze(paths: Array<String>, config: CsvAnalysisConfig?): CsvStatistics
  // @expose static native fn sample(reader: CsvReader, max_lines: int?): Table
}

var config = CsvAnalysisConfig { row_limit: 1000, enumerable_limit: 50 };
var stats  = Csv::analyze(["/data/sales.csv"], config);   // paths are Array<String>
var type_code = Csv::generate(stats);                     // generated GCL types

var reader = CsvReader<any> { path: "/data/sales.csv" };
var sample = Csv::sample(reader, 100);                     // Table
```
`CsvStatistics` (per-`CsvColumnStatistics` profiling, `line_count`, `fail_count`, `file_count`), `CsvColumnStatistics` (type counts, `date_format_count`, `enumerable_count`, `profile: Gaussian`), `CsvAnalysisConfig` (`header_lines?`, `separator?`, `row_limit?`, `enumerable_limit?`, `date_check_limit?`, `date_formats?`), and `CsvSharding` (`id`, `column`, `modulo`) round out CSV support.

### File System

#### File
```gcl
type File {
  path: String;
  size: int?;                // null for directories
  last_modification: time?;
  // static native fns: baseDir(): String, userDir(): String, workingDir(): String,
  //   open(path): File?, delete(path): bool, rename(old, new): bool, copy(src, target): bool,
  //   mkdir(path): bool, ls(path, ends_with: String?, recursive: bool): Array<File>
  // native fns: isDir(): bool, name(): String, extension(): String?, sha256(): String?
}

var csv_files = File::ls("/data", ".csv", true);   // recursive
var file = File::open("/data/input.txt")!!;
println("Size: ${file.size}, ext: ${file.extension()!!}, sha: ${file.sha256()!!}");
File::copy("/data/src.txt", "/data/dst.txt");
File::mkdir("/data/archive");
```

#### FileWalker
```gcl
type FileWalker {
  path: String;
  // native fn isEmpty(): bool, next(): File?
}
var walker = FileWalker { path: "/data" };
while (!walker.isEmpty()) {
  var f = walker.next();
  if (f != null && !f.isDir()) { println(f.path); }
}
```

### HTTP Client

#### Http<T>
HTTP client; headers are `Map<String, String>?`.
```gcl
type Http<T> {
  // native fn get(url: String, headers: Map<String, String>?): T
  // native fn getFile(url: String, path: String, headers: Map<String, String>?)
  // native fn post(url: String, body: any?, headers: Map<String, String>?): T
  // native fn put(url: String, body: any?, headers: Map<String, String>?): T
  // native fn send(request: HttpRequest): HttpResponse<T>
}

var headers = Map<String, String> {};
headers.set("Authorization", "Bearer token");

var user = Http<User> {}.get("https://api.example.com/users/123", headers);
Http<any> {}.getFile("https://example.com/data.csv", "/local/data.csv", null);

var resp = Http<User> {}.send(HttpRequest {
  method: HttpMethod::GET,
  url: "https://api.example.com/users/123",
  headers: headers
});                                       // HttpResponse<User>
println("status: ${resp.status_code}");
```

#### HttpMethod / HttpRequest / HttpResponse
```gcl
enum HttpMethod { GET, HEAD, POST, PUT, DELETE, CONNECT, OPTIONS, TRACE, PATCH; }

type HttpRequest {
  method: HttpMethod;
  url: String;
  headers: Map<String, String>?;
  body: String?;
  timeout: duration?;
  max_response_size: int?;   // max response body bytes; null/0 = unlimited. Guards
                              // unbounded chunked / no-Content-Length responses.
}

type HttpResponse<T> {
  status_code: int;
  headers: Map<String, String>;
  content: T?;
  error_msg: String?;
}
```

#### Url
```gcl
type Url {
  protocol: String?; host: String?; port: int?; path: String?;
  user: String?; password: String?; params: Map<String, String>?; hash: String?;
  // static native fn parse(url: String): Url
  // static native fn encode(value: any): String   // x-www-form-urlencoded
}
var url = Url::parse("https://api.example.com:8080/users?active=true#top");
println("${url.protocol} ${url.host} ${url.port} ${url.path}");
```

### Email

#### Email & Smtp
There is **no** `attachments` field on `Email` in 8.0.
```gcl
type Email {
  from: String;
  subject: String;
  body: String;
  body_is_html: bool;
  to: Array<String>;
  cc: Array<String>?;
  bcc: Array<String>?;
}

type Smtp {
  host: String;
  port: int;
  mode: SmtpMode?;
  authenticate: SmtpAuth?;
  user: String?;
  pass: String?;
  // native fn send(email: Email)
}

enum SmtpMode { plain(0); ssl_tls(1); starttls(2); }
enum SmtpAuth { none(0); plain(1); login(2); }

var smtp = Smtp {
  host: "smtp.gmail.com", port: 587,
  mode: SmtpMode::starttls, authenticate: SmtpAuth::login,
  user: "sender@example.com", pass: "app_password"
};
smtp.send(Email {
  from: "sender@example.com",
  to: ["recipient@example.com"],
  subject: "Monthly Report",
  body: "<h1>Report</h1>",
  body_is_html: true
});
```

### S3 Object Storage

#### S3, S3Bucket, S3Object, S3BasicCredentials
```gcl
type S3Object   { key: String; last_modified: time; size: int; etag: String; }
type S3Bucket   { name: String; creation_date: time; }
type S3BasicCredentials { access_key: String; secret_key: String; }

type S3 {
  host: String;
  region: String;
  credentials: S3BasicCredentials;
  force_path_style: bool?;
  // native fns:
  //   list_objects(bucket, prefix: String?, start_after: String?, max_keys: int?): Array<S3Object>
  //   get_object(bucket, key, filepath)
  //   put_object(bucket, filepath, key)
  //   delete_object(bucket, key)
  //   create_bucket(bucket)
  //   list_buckets(prefix: String?): Array<S3Bucket>
}

var s3 = S3 {
  host: "localhost:9000", region: "us-east-1",
  credentials: S3BasicCredentials { access_key: "AKIA...", secret_key: "secret" },
  force_path_style: true
};
s3.create_bucket("my-bucket");
s3.put_object("my-bucket", "/local/file.txt", "virtual/path/file.txt");
var objects = s3.list_objects("my-bucket", "prefix/", null, 1000);
for (_, obj in objects) { println("${obj.key} (${obj.size} bytes, etag=${obj.etag})"); }
s3.get_object("my-bucket", "virtual/path/file.txt", "/local/download.txt");
var buckets = s3.list_buckets(null);     // Array<S3Bucket>
```

## Util Module (std::util)

### Collections

#### Queue<T>
Optionally bounded FIFO (`capacity` is private; drops the front element when full).
```gcl
// native fns: push(value), pop(): T?, front(): T?, back(): T?, clear()
var q = Queue<String> {};
q.push("first");
var item = q.pop();
```

#### Stack<T>
LIFO.
```gcl
// native fns: push(value), pop(): T?, first(): T? (top), last(): T? (bottom), clear()
var s = Stack<int> {};
s.push(10); s.push(20);
var top = s.pop();        // 20
```

#### SlidingWindow<T>
Fixed-count window with streaming stats.
```gcl
type SlidingWindow<T> {
  span: int;              // max number of elements
  sum: float?; sumsq: float?;
  // native fns: add(value), clear(), median(): float?, min(): T?, max(): T?,
  //   std(): float?, avg(): float?, size(): int
}
var w = SlidingWindow<float> { span: 100 };
w.add(10.5);
var avg = w.avg()!!;
```

#### TimeWindow<T>
Window bounded by a `duration` span; values older than `span` from the latest timepoint expire.
```gcl
type TimeWindow<T> {
  span: duration;
  sum: float?; sumsq: float?;
  // native fns: add(t: time, value), update(t: time), clear(),
  //   min(): Tuple<time, T>?, max(): Tuple<time, T>?, median(): float?,
  //   std(): float?, avg(): float?, size(): int
}
var tw = TimeWindow<float> { span: 5min };
tw.add(time::now(), temperature);
var avg = tw.avg();
```

### Statistics

#### Gaussian<T>
Live (streaming) distribution.
```gcl
type Gaussian<T> {
  sum: float?; sumsq: float?; count: int?;
  min: T?; max: T?;
  // native fns: add(value: T?): bool, addx(value: T?, count: int): bool,
  //   add_gaussian(value: Gaussian<T>): bool,
  //   std(): T?, avg(): T?, normalize(value): float?, inverse_normalize(value: float): T,
  //   standardize(value): float, inverse_standardize(value: float): T,
  //   confidence(value): float, pdf(value): float, cdf(value): float
}
var g = Gaussian<float> {};
g.add(10.0); g.add(20.0); g.add(30.0);
var avg = g.avg()!!;             // 20.0
var z   = g.standardize(25.0);
```

#### Histogram<T> & HistogramStats<T>
```gcl
type Histogram<T> {
  quantizer: Quantizer<T>;
  bins: Array<int?>?;
  nb_rejected: int?; nb_accepted: int?;
  min: T?; max: T?; sum: float?; sumsq: float?;
  // native fns: add(value), addx(value, count), stats(): HistogramStats<T>?,
  //   percentile(ratio: float): T?, ratio_under(value): float,
  //   get_bins(): Array<HistogramBin<T>>
}
type HistogramBin<T> {
  bin: QuantizerSlotBound<T>;
  count: int; ratio: float;
  cumulative_count: int; cumulative_ratio: float;
}
// HistogramStats<T> exposes min/max, whisker_low/high, percentile1..99, sum, avg, std, size

var h = Histogram<float> { quantizer: LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 20 } };
for (_, score in test_scores) { h.add(score); }
var p90      = h.percentile(0.9);
var below_60 = h.ratio_under(60.0);
var stats    = h.stats();
```

#### GaussianProfile<T>
Per-quantizer-slot gaussian statistics.
```gcl
type GaussianProfile<T> {
  quantizer: Quantizer<T>;
  value_min: float?; nb_rejected: int?;
  // native fns: add(key: T, value: float), avg(key): float, std(key): float,
  //   sum(key): float, count(key): int
}
type GaussianProfileSlot { sum: int; sumsq: int; count: int; }   // per-slot accumulator
var profile = GaussianProfile<int> { quantizer: LinearQuantizer<int> { min: 0, max: 100, bins: 10 } };
profile.add(age, salary);
var avg_salary = profile.avg(age);
```

### Quantizers

`Quantizer<T>` is an `abstract type` with `size(): int`, `quantize(value: T): int`, `bounds(slot: int): QuantizerSlotBound<T>`.
```gcl
type QuantizerSlotBound<T> { min: T; max: T; center: T; }
```

#### LinearQuantizer<T> / LogQuantizer<T>
Uniform / logarithmic bins. Fields: `min`, `max`, `bins`, `open: bool?`.
```gcl
var linear = LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 10 };
var bin    = linear.quantize(25.0);        // 2
var bounds = linear.bounds(2);             // QuantizerSlotBound

var logq = LogQuantizer<float> { min: 1.0, max: 1000.0, bins: 10 };
```

#### CustomQuantizer<T>
Bins defined by step boundaries. Fields: `min`, `max`, `step_starts: Array<T>`, `open: bool?`.
```gcl
var age_bins = CustomQuantizer<int> { min: 0, max: 100, step_starts: [0, 18, 25, 40, 65] };
var group    = age_bins.quantize(32);
```

#### MultiQuantizer<T>
Combines several quantizers into one multidimensional index. `quantizers: Array<Quantizer<T>>`; adds `slot_vector(slot: int): Array<int>`.
```gcl
var multi = MultiQuantizer<float> {
  quantizers: [
    LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 5 },
    LogQuantizer<float>    { min: 1000.0, max: 200000.0, bins: 8 },
    LinearQuantizer<float> { min: 0.0, max: 100.0, bins: 10 }
  ]
};
var slot   = multi.quantize([35.0, 45000.0, 87.5]);
var vector = multi.slot_vector(slot);
```

### Utilities

#### Random
Seedable PRNG (xorshift64*).
```gcl
type Random {
  seed: int?;
  // native fns: char(): char, uniform(min: int, max: int): int,
  //   uniformf(min: float, max: float): float, uniformGeo(min: geo, max: geo): geo,
  //   normal(avg, std): float, gaussian(profile: Gaussian): float,
  //   fill<T>(target: any, nb: int, min: T, max: T),
  //   uuid(): String  (UUID v4, reproducible), uuid_v7(): String (UUID v7, reproducible tail)
}
var rng  = Random { seed: 12345 };
var dice = rng.uniform(1, 7);            // 1..6
var prob = rng.uniformf(0.0, 1.0);
var id   = rng.uuid();                   // seedable UUID v4 (NOT crypto-secure)
```

#### Assert
```gcl
// static native fns: equals(a: any?, b: any?), equalsd(a: float, b: float, epsilon: float),
//   equalst(a: Tensor, b: Tensor, epsilon: float), isTrue(v: bool), isFalse(v: bool),
//   isNull(v: any?), isNotNull(v: any?)
Assert::equals(actual, expected);
Assert::equalsd(pi, 3.14159, 0.001);
Assert::isNotNull(value);
```

#### ProgressTracker
Reports overall performance (speed, ETA) measured since `start`, plus a `speed_smoothed` that reacts to the recent pace between updates.
```gcl
type ProgressTracker {
  static DEFAULT_SMOOTHING: float = 0.1;  // used when `smoothing` is null; lower = steadier ETA

  start: time;
  total: int?;             // max expected count
  counter: int?;           // current step count, as last set by update() (absolute, not a running sum)
  duration: duration?;     // overall duration since `start`
  progress: float?;        // 0.0 .. 1.0
  speed: float?;           // overall speed since `start`, in counter per second
  remaining: duration?;    // estimated from speed_smoothed
  speed_smoothed: float?;  // EMA of the per-update (lap) speed, counter per second
  smoothing: float?;       // EMA weight in [0.0, 1.0] for the latest lap (0=overall avg, 1=last lap only);
                           // DEFAULT_SMOOTHING is used when null
  // native fn update(nb: int)   -- sets the counter to `nb` (absolute, NOT incremental);
                                  // recomputes duration/speed/speed_smoothed/progress/remaining
}
var tracker = ProgressTracker { start: time::now(), total: 10000 };
tracker.update(2500);       // sets counter to 2500 (not +2500)
println("Progress: ${tracker.progress}");      // 0.25
```

#### Crypto
```gcl
type Crypto {
  // static native fns:
  //   sha1(content): String, sha1hex(content): String,
  //   sha256(content): String, sha256hex(content): String,
  //   sha256_sign_pkcs1(input, key_path): String, sha256_sign_pkcs1_hex(input, key_path): String,
  //   sha256_hmac_hex(input, key): String,
  //   base64_encode/decode(v): String, base64url_encode/decode(v): String,
  //   hex_encode/decode(v): String, url_encode/decode(v): String
}
var digest = Crypto::sha256hex("sensitive data");
var b64    = Crypto::base64_encode("hello");
var mac    = Crypto::sha256_hmac_hex(payload, secret_key);
var sig    = Crypto::sha256_sign_pkcs1("data", "/keys/private.pem");
```

#### Uuid
Cryptographically strong UUID generator (CTR_DRBG seeded from system entropy; not reproducible). Use this for security tokens and database keys; use `Random.uuid()` / `Random.uuid_v7()` for deterministic/seedable values.
```gcl
type Uuid {
  // static native fn v4(): String   // 122 bits CSPRNG, canonical 8-4-4-4-12
  // static native fn v7(): String   // time-sortable: 48-bit ms ts + counter + CSPRNG
}
var token = Uuid::v4();
var key   = Uuid::v7();
```

## Usage Guidelines

(Module purposes and the types each provides are summarized in the [Overview](#overview).)

### Best Practices

1. **Resource Management**: Always call `flush()` on writers before opening a reader on the same path.
2. **Error Handling**: Check nullable (`?`) return values before use.
3. **Scheduling**: Pick the right periodicity — `FixedPeriodicity` for intervals, `DailyPeriodicity`/`WeeklyPeriodicity`/`MonthlyPeriodicity`/`YearlyPeriodicity` for calendar timing.
4. **CSV Analysis**: Use `Csv::analyze(paths, config)` (paths is `Array<String>`) to understand structure before processing large files.
5. **Statistics**: `Gaussian` for running stats, `Histogram` for distributions, `GaussianProfile` for per-bin stats.
6. **File Discovery**: Use `File::ls(path, ends_with, recursive)` to filter by extension.
7. **HTTP**: Build headers as a `Map<String, String>` and pass them to `get`/`post`/`put` or use `send(HttpRequest)` for full control over method/timeout.
8. **Security**: Prefer `Uuid::v4()`/`v7()` for tokens and DB keys; reserve `Random.uuid()` for reproducible test data.

### Common Patterns

#### Scheduled Data Processing
```gcl
fn process_daily_data() {
  var files = File::ls("/data/incoming", ".csv", false);
  for (_, file in files) {
    var reader = CsvReader<Record> { path: file.path };
    while (reader.can_read()) { process_record(reader.read()); }
    File::rename(file.path, "/data/processed/${file.name()}");
  }
}

Scheduler::add(process_daily_data, DailyPeriodicity { hour: 1, minute: 0 }, null);
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
      if (value > avg + 3.0 * std) { warn("Anomaly detected: ${value}"); }
    }
  }
}
```

#### HTTP API Integration
```gcl
fn fetch_and_store_data(api_key: String) {
  var headers = Map<String, String> {};
  headers.set("Authorization", "Bearer ${api_key}");
  // Http<T>.get returns T directly (non-nullable); a failed request throws rather than returning null
  var response = Http<Array<Record>> {}.get("https://api.example.com/data", headers);
  var writer = JsonWriter<Record> { path: "/data/cache.json" };
  for (_, record in response) { writer.writeln(record); }
  writer.flush();
}
```
