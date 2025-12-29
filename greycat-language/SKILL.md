---
name: greycat
description: GreyCat backend development guide for graph-based language with built-in persistence. Use when building GreyCat applications (.gcl files), working with persisted nodes and indexed collections (nodeList, nodeIndex, nodeTime, nodeGeo), creating data models and services, writing API endpoints with @permission decorators, or running GreyCat commands (greycat serve, greycat test, greycat run import). Essential for any GreyCat project development, testing, or debugging.
---

# GreyCat Backend Development

Graph-based language with built-in persistence. Not a database—an evolving coded structure.

## Commands

```bash
curl -fsSL https://get.greycat.io/install.sh | bash -s stable  # install (Linux/Mac)
greycat install       # download libraries
greycat-lang lint     # check errors (ignore /lib/ warnings)
greycat test          # run tests
greycat run import    # run import() from project.gcl
greycat serve         # start server :8080
```

## Architecture

**Backend (GreyCat .gcl)**
- `project.gcl` - Entry point, libs, permissions, roles, main(), init()
- `backend/src/model/` - Data models + global indices
- `backend/src/service/` - XxxService abstract types (::create, ::find)
- `backend/src/api/` - @expose + @permission functions, @volatile response types
- `backend/src/edi/` - Import/export

**project.gcl example:**
```gcl
@library("std", "7.5.125-dev");
@library("explorer", "7.5.3-dev");  // enables graph navigation in explorer UI
@include("backend");  // includes all .gcl files in backend/ recursively

@permission("app.admin", "app admin permission");
@permission("app.user", "app user permission");

@role("admin", "app.admin", "app.user", "public", "admin", "api", "debug", "files");
@role("public", "public", "api", "files");
@role("user", "app.user", "public", "api", "files");

@format_indent(4);
@format_line_width(280);

fn main() { }
```

**@include Rules:**
- **ONLY use `@include` in `project.gcl`** — does NOT work in other `.gcl` files
- `@include("folder")` recursively includes ALL `.gcl` files in that folder

**Essential Libraries:**
- `@library("std", "7.5.125-dev")` — Standard library (required)
- `@library("explorer", "7.5.3-dev")` — Graph navigation UI at `/explorer` (recommended for development)

**Conventions:** GCL: snake_case files, PascalCase types | Unused vars: `_prefix` | Tests: `*_test.gcl`

## Types

| Type | Details | Type | Details |
|------|---------|------|---------|
| `int` | 64-bit signed, `1_000_000` | `float` | 64-bit, `3.14`, `1e7` |
| `bool` | `true`/`false`, AND/OR/NOT ops | `char` | `'a'` |
| `String` | Immutable, `"Hello ${name}"` | `time` | 64-bit μs since epoch UTC |
| `duration` | `5_s`, `7_hour`, `1_day` | `Date` | UI display, needs timezone |
| `geo` | `geo{lat, lng}`, `.lat()`, `.lng()` | | |

```gcl
var i = 42; var j = 1_000_000;              // int with separators
var t = time::now(); var d = 5_s;           // time + duration
t.toDate(TimeZone::"Europe/Luxembourg");    // time → Date
geo{49.6, 6.1};                             // coordinates

// ✅ All std types init with {} — NOT ::new()
var list = Array<String>{}; var map = Map<String, int>{};

@volatile  // non-persisted (for API responses)
type ApiResponse { data: String; status: int; }
```

**Geo shapes**: `GeoCircle`, `GeoBox`, `GeoPoly` — all have `.contains(geo): bool`

## Nullability

All types non-null by default. Use `?` for nullable:
```gcl
var city: City?;                    // nullable
city?.name?.size();                 // optional chaining
city?.name ?? "Unknown";            // nullish coalescing
data.get("key")!!;                  // non-null assertion

if (country == null) { return null; }
return country->name;               // ✅ no !! needed after null check
```

**⚠️ Cast + coalescing needs parens:**
```gcl
// WRONG: answer as String? ?? "default"
// RIGHT:
(answer as String?) ?? "default"
```

**⚠️ NO TERNARY OPERATOR** — use if/else:
```gcl
var result: String;
if (valid) { result = "yes"; } else { result = "no"; }
```

## Nodes (Persistence)

Nodes = 64-bit refs to persistent containers. Core persistence mechanism.

```gcl
type Country { name: String; code: int; }
var obj = Country { name: "LU", code: 352 };  // RAM only
var n = node<Country>{obj};                    // persisted

*n;              // dereference
n->name;         // ✅ arrow: deref + field (NOT (*n)->name)
n.resolve();     // method
n->name = "X";   // modify object field
node<int>{0}.set(5);  // primitives use .set()
```

**Use node refs for sharing**: `type City { country: node<Country>; }` (light, 64-bit) vs embedded object (heavy)

**Ownership**: Objects belong to ONE node only. For multi-index, store node refs:
```gcl
var by_id = nodeList<node<Item>>{};
var by_name = nodeIndex<String, node<Item>>{};
var item = node<Item>{ Item{} };
by_id.set(1, item); by_name.set("x", item);  // both point to same node
```

**Transactions**: Atomic per function, rollback on error.

## Indexed Collections

| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `node<T>` | — | `Array<T>`, `Map<K,V>` |
| `nodeList<node<T>>` | int | `Stack<T>`, `Queue<T>` |
| `nodeIndex<K, node<V>>` | hash | `Set<T>`, `Tuple<A,B>` |
| `nodeTime<node<T>>` | time | `Buffer`, `Table`, `Tensor` |
| `nodeGeo<node<T>>` | geo | `TimeWindow`, `SlidingWindow` |

```gcl
// nodeTime - interpolates between points
var temps = nodeTime<float>{};
temps.setAt(t1, 20.5);
for (t: time, v: float in temps[from..to]) { }

// nodeIndex - uses set/get (NOT add)
var idx = nodeIndex<String, node<X>>{};
idx.set("key", val); idx.get("key");

// nodeList
var list = nodeList<node<X>>{};
for (i: int, v in list[0..100]) { }

// nodeGeo
var geo_idx = nodeGeo<node<B>>{};
for (pos: geo, b in geo_idx.filter(GeoBox{...})) { }
```

**Sampling** large time-series: `nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null)`
Modes: `fixed`, `fixed_reg`, `adaptative`, `dense`

**Array sorting**:
```gcl
cities.sort_by(City::population, SortOrder::desc);  // ✅ native typed sort
// or
buildings.sort_by(Building::value, SortOrder::desc);
```

**⚠️ CRITICAL: Initialize Collection Attributes**
Non-nullable `nodeList`, `nodeIndex`, `nodeTime`, `Array` attributes **MUST be initialized**:
```gcl
// ✅ Correct — initialize collections on creation
var city = node<City>{ City{
    name: "Paris",
    country: country_node,
    streets: nodeList<node<Street>>{}   // ⚠️ MUST initialize!
}};
```

## Module Variables

Root-level vars must be nodes/indexes → auto-persisted:
```gcl
var count: node<int?>;
fn main() { count.set((count.resolve() ?? 0) + 1); }
```

**Global indices are auto-initialized**: Module-level `nodeIndex`, `nodeList`, `nodeTime`, `nodeGeo` are automatically initialized by GreyCat — no `{}` needed:
```gcl
// ✅ Global indices — no initialization needed
var cities_by_name: nodeIndex<String, node<City>>;
var all_users: nodeList<node<User>>;

// ⚠️ Collection ATTRIBUTES in types still need initialization
```

## Model vs API Types

**In model files** — store node refs, declare global indices first:
```gcl
// ✅ Global indices first, then types
var cities_by_name: nodeIndex<String, node<City>>;

type City {
    name: String;
    country: node<Country>;           // ✅ node ref (light, 64-bit)
    streets: nodeList<node<Street>>;  // ✅ store refs, not objects
}
```

**In API files** — return `Array<XxxView>` with `@volatile`, never nodeList:
```gcl
@volatile  // non-persisted, postfix "View" for API responses
type CityView {
    name: String;
    country_name: String;
    street_count: int;
}

@expose  // ⚠️ REQUIRED for API endpoints to be callable via HTTP
@permission("public")  // ⚠️ takes String, not identifier
fn getCities(): Array<CityView> { ... }  // ✅ Array<View>, not nodeList
```

**⚠️ CRITICAL: API functions must have `@expose`** — without it, the function cannot be called via HTTP even if it has `@permission`.

## Functions & Control Flow

```gcl
fn add(x: int, y: int): int { return x + y; }
fn doSomething() { /* no void type — omit return type */ }
var f = fn(x: int): int { return x * 2; };  // lambda

if (c) { } else { }
for (var i = 0; i < 10; i++) { }
for (k: K, v: V in map) { }               // ✅ prefer for-in
for (i, v in nullable_list?) { }          // ✅ use ? for nullable iteration
while (c) { }; do { } while (c);
if (val is String) { }
```

## Services Pattern

```gcl
// service/country_service.gcl — avoids naming conflicts
abstract type CountryService {
    static fn create(name: String, code: String): node<Country> { ... }
    static fn find(name: String): node<Country>? { return countries_by_name.get(name); }
}
// Usage: CountryService::create(...) vs fn createCountry() in API
```

## Abstract Types & Inheritance

```gcl
abstract type Building {
    address: String;
    value: float;

    fn calculateTax(): float;  // Abstract - must be implemented
    fn getDescription(): String {
        return "${address}: $${value}";  // Concrete - shared by all
    }
}

type House extends Building {
    bedrooms: int;
    fn calculateTax(): float { return value * 0.01; }  // 1% tax
}

type Office extends Building {
    floors: int;
    fn calculateTax(): float { return value * 0.025; }  // 2.5% tax
}

// Polymorphic collection
var buildings: nodeIndex<String, node<Building>>{};
buildings.set("123 Main", node<House>{ House {...} });
buildings.set("45 Blvd", node<Office>{ Office {...} });

// Polymorphic method calls
for (addr, building in buildings) {
    var tax = building->calculateTax();  // Calls correct implementation
    if (building is House) { /* type-specific logic */ }
}
```

**Key points:**
- `abstract type` can have fields, concrete methods, and abstract methods
- `abstract fn` must be implemented by extending types
- Concrete methods **cannot be overridden** by subtypes
- Polymorphic dispatch via node refs (`node<BaseType>`)
- Use `is` operator to check concrete type

## Logging & Error Handling

```gcl
info("Number of countries ${countries.size()}");
warn("Cache miss for key ${key}");
error("Failed to connect: ${message}");

print("value: ");       // no newline
println("done");        // with newline

try { riskyOperation(); } catch (ex) { error("Error: ${ex}"); }
```

## Parallelization

GreyCat provides a Job API for parallel execution with fork-join pattern.

### Basic Pattern

```gcl
type CityStats { cityName: String; totalStreets: int; }

fn analyzeCity(city: node<City>): CityStats {
    return CityStats {
        cityName: city->name,
        totalStreets: city->streets.size()
    };
}

fn analyzeCountry(country: node<Country>): Array<CityStats> {
    var jobs = Array<Job<CityStats>> {};

    // 1. Create jobs
    for (city in country->cities) {
        jobs.add(Job<CityStats> {
            function: analyzeCity,
            arguments: [city]
        });
    }

    // 2. Execute in parallel
    await(jobs, MergeStrategy::last_wins);

    // 3. Collect results
    var results = Array<CityStats> {};
    for (job in jobs) {
        results.add(job.result());  // Type-safe: CityStats
    }
    return results;
}
```

### Key Points

- `Job<T>` is generic - specify return type for type safety
- Use `MergeStrategy::last_wins` for parallel writes (recommended)
- **No nested await** - flatten job structure to single `await()` call
- Jobs execute in parallel within task context

### Async Tasks

Spawn async task via HTTP:
```bash
curl -H "task:''" -X POST -d '[]' http://localhost:8080/project::fn
```

Periodic tasks:
```gcl
PeriodicTask::set(Array<PeriodicTask>{
    PeriodicTask{user_id: 0, every: 1_day, function: project::my_task, start: time::now()}
});
```

See [Concurrency Reference](references/concurrency.md) for detailed patterns, error handling, and await limitations.

## Testing

Run with `greycat test`. Test files: `*_test.gcl` in `./backend/test/`.

```gcl
@test
fn test_city_creation() {
    var city = City::create("Paris", country_node);
    Assert::equals(city->name, "Paris");
}

@test
fn test_building_creation() {
    var building = Building::create("123 Main St", BuildingType::Residential);
    Assert::equals(building->buildingType, BuildingType::Residential);
}

fn setup() { /* runs before tests */ }
fn teardown() { /* cleanup after tests */ }
```

**Assert methods**: `Assert::equals(a, b)`, `Assert::equalsd(a, b, epsilon)`, `Assert::isTrue(v)`, `Assert::isFalse(v)`, `Assert::isNull(v)`, `Assert::isNotNull(v)`

## Syntax Reference

| `::` static | `->` node deref+field | `*` deref | `${}` interpolation |
|-------------|----------------------|-----------|---------------------|
| `?` nullable | `?.` optional chain | `??` coalesce | double-bang non-null assert |

## Common Pitfalls

| ❌ Wrong | ✅ Correct | Why |
|----------|-----------|-----|
| `Array<T>::new()` | `Array<T>{}` | std types use `{}` |
| `(*node)->field` | `node->field` | `->` already dereferences |
| `@permission(public)` | `@permission("public")` | takes String |
| `@permission("api") fn getX()` | `@expose @permission("api") fn getX()` | API functions need @expose |
| `for(i=0;i<n;i++) list.get(i)` | `for (i, v in list)` | type inference |
| `nodeList<City>` | `nodeList<node<City>>` | store refs, not objects |
| `fn getX(): nodeList<...>` | `fn getX(): Array<XxxView>` | API returns Array+View |
| `nodeIndex.add(k, v)` | `nodeIndex.set(k, v)` | nodeIndex uses set/get |
| `for(i, v in nullable_list)` | `for(i, v in nullable_list?)` | use `?` for nullable |
| `fn doX(): void` | `fn doX()` | no void type |
| `City{name: "X"}` (missing collections) | `City{name: "X", streets: nodeList<...>{}}` | init non-nullable collections |

### Acceptable Double-Bang Patterns

Using the non-null assertion operator (double-bang) is acceptable for global registry lookups where data is guaranteed to exist:
```gcl
var config = ConfigRegistry::getConfig(key)!!;  // ✅ OK — populated at init
```

## ABI & Database

**⚠️ DEV MODE: No Migrations** — delete deprecated fields immediately. Reset `gcdata/` when schema changes.

Adding non-nullable fields to persisted types fails → make nullable: `newField: int?`

### ⚠️ Reset Database
```bash
rm -rf gcdata    # DELETES ALL DATA — ask user confirmation first!
greycat run import
```

Docs: https://doc.greycat.io/

## Advanced References

For detailed documentation on specific topics, consult these reference files:

- **[frontend.md](references/frontend.md)** — Complete React integration guide: @greycat/web SDK, TypeScript setup, authentication, React Query, hooks, error handling, time handling, best practices
- **[nodes.md](references/nodes.md)** — Deep dive into nodes, persistence, transactions, indexed collections (nodeTime, nodeList, nodeGeo, nodeIndex), sampling
- **[data_structures.md](references/data_structures.md)** — Tensor, Table, Buffer, Windows, Stack, Queue, Tuple
- **[concurrency.md](references/concurrency.md)** — Jobs, await, parallel writes, Tasks, PeriodicTask
- **[io.md](references/io.md)** — CsvReader/Writer, JsonReader/Writer, File operations, HTTP client, SMTP
- **[time.md](references/time.md)** — time, Date, duration, DurationUnit, CalendarUnit, format specifiers
- **[permissions.md](references/permissions.md)** — RBAC, @permission, @role, SSO integration
- **[testing.md](references/testing.md)** — @test, Assert, setup/teardown, test conventions

## Library References (GCL Type Definitions)

Complete type definitions for all available GreyCat libraries are provided as GCL reference files. These files contain the full API surface, method signatures, and documentation for each library.

### Standard Library (std)

Core types, collections, I/O, and runtime utilities:

- **[std/core.gcl](references/std/core.gcl)** — Core types: any, null, type, field, function, bool, char, int, float, nodeTime, nodeList, nodeIndex, nodeGeo, String, geo, time, duration, Date, TimeZone, Array, Map, Set, Stack, Queue, Tuple, Tensor, Table, Buffer, Gaussian, SamplingMode, SortOrder
- **[std/io.gcl](references/std/io.gcl)** — I/O types: Reader, Writer, BinReader, GcbReader/Writer, TextReader/Writer, XmlReader, JsonReader/Writer, Json, CsvReader/Writer, CsvFormat, CsvStatistics, Csv, File, FileWalker, Url, Http, HttpRequest, HttpResponse, HttpMethod, Email, Smtp, SmtpMode, SmtpAuth
- **[std/runtime.gcl](references/std/runtime.gcl)** — Runtime types: Runtime, Task, PeriodicTask, Job, StoreStat, License, UserCredential, UserRole, UserGroupPolicyType, SystemInfo, Spi
- **[std/util.gcl](references/std/util.gcl)** — Utility types: Assert, Histogram, GeoBox, GeoCircle, GeoPoly, BoxWhisker, Quantile, Random, TimeWindow, SlidingWindow, Crypto, HashMode, BitSet, StringBuilder

### AI Library (ai)

LLM integration via llama.cpp:

- **[ai/llm_model.gcl](references/ai/llm_model.gcl)** — Model type: Model::load, Model::info, embed, generate, chat, tokenize, detokenize, free, SplitMode, RopeScalingType, PoolingType, AttentionType
- **[ai/llm_context.gcl](references/ai/llm_context.gcl)** — Context management: Context, ContextParams, ContextBatch, KvCacheView, KvCellInfo
- **[ai/llm_sampler.gcl](references/ai/llm_sampler.gcl)** — Sampling strategies: Sampler, SamplerParams, SamplerChain, LogitBias, GrammarType
- **[ai/llm_lora.gcl](references/ai/llm_lora.gcl)** — LoRA adapters: LoraAdapter, LoraParams
- **[ai/llm_types.gcl](references/ai/llm_types.gcl)** — Common types: ModelParams, ChatMessage, ChatCompletionChunk, VocabType, LogLevel

### Algebra Library (algebra)

Machine learning, neural networks, and numerical computing:

- **[algebra/ml.gcl](references/algebra/ml.gcl)** — Machine learning utilities
- **[algebra/compute.gcl](references/algebra/compute.gcl)** — Computational operations
- **[algebra/nn.gcl](references/algebra/nn.gcl)** — Neural network types and operations
- **[algebra/nn_layers_names.gcl](references/algebra/nn_layers_names.gcl)** — Neural network layer naming conventions
- **[algebra/patterns.gcl](references/algebra/patterns.gcl)** — Pattern recognition algorithms
- **[algebra/transforms.gcl](references/algebra/transforms.gcl)** — Data transformation utilities
- **[algebra/kmeans.gcl](references/algebra/kmeans.gcl)** — K-means clustering
- **[algebra/climate.gcl](references/algebra/climate.gcl)** — Climate data modeling

### Integration Libraries

External system integrations:

- **[kafka/kafka.gcl](references/kafka/kafka.gcl)** — Apache Kafka producer/consumer integration
- **[sql/postgres.gcl](references/sql/postgres.gcl)** — PostgreSQL database integration
- **[s3/s3.gcl](references/s3/s3.gcl)** — Amazon S3 object storage integration
- **[opcua/opcua.gcl](references/opcua/opcua.gcl)** — OPC UA industrial protocol integration
- **[useragent/useragent.gcl](references/useragent/useragent.gcl)** — User agent parsing utilities

### Domain-Specific Libraries

- **[finance/finance.gcl](references/finance/finance.gcl)** — Financial modeling and calculations
- **[powerflow/powerflow.gcl](references/powerflow/powerflow.gcl)** — Electrical power flow analysis

### Using Libraries in Your Project

Add libraries to your `project.gcl`:

```gcl
@library("std", "7.5.125-dev");      // Standard library (required)
@library("ai", "7.5.51-dev");        // AI/LLM support
@library("algebra", "7.5.51-dev");   // ML and numerical computing
@library("kafka", "7.5.51-dev");     // Kafka integration
@library("sql", "7.5.51-dev");       // PostgreSQL support (postgres library)
@library("s3", "7.5.51-dev");        // S3 storage
@library("finance", "7.5.51-dev");   // Financial utilities
@library("powerflow", "7.5.51-dev"); // Power flow analysis
@library("opcua", "7.5.51-dev");     // OPC UA integration
@library("useragent", "7.5.51-dev"); // User agent parsing
@library("explorer", "7.5.3-dev");   // Graph UI (dev only)
```

**Library Installation:**
```bash
greycat install    # downloads all declared @library dependencies
```
