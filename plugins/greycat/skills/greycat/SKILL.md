---
name: greycat
description: "Generates and maintains .gcl files for GreyCat projects. Provides language syntax, node persistence patterns, typed CSV/JSON I/O, time series, geo coordinates, API exposure, and MCP tagging. Use when creating, editing, or debugging .gcl code, or when the user references GreyCat, nodeIndex, nodeTime, nodeList, or project.gcl."
---

# GreyCat

Unified language + database (temporal/graph/vector) + web server + MCP. Built for billion-scale digital twins.

## Installation

Verify with `which greycat` or `greycat --version`. If not found, confirm with user before installing:

**Linux/Mac/FreeBSD**: `curl -fsSL https://get.greycat.io/install.sh | bash -s dev`
**Windows**: `iwr https://get.greycat.io/install_dev.ps1 -useb | iex`

## Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `greycat build` | Compile project | `--log`, `--cache` |
| `greycat serve` | Start server (HTTP + MCP) | `--port=8080`, `--workers=N`, `--user=1` (dev only) |
| `greycat run` | Execute main() or function | `greycat run myFunction` |
| `greycat test` | Run @test functions | Exit 0 on success |
| `greycat install` | Download dependencies | From project.gcl @library |
| `greycat codegen` | Generate typed headers | TS, Python, C, Rust |
| `greycat defrag` | Compact storage | Safe anytime |
| `greycat-lang lint --fix` | **Auto-fix errors** | **Run after code changes** |
| `greycat-lang fmt` | Format files | In-place |
| `greycat-lang server` | Start LSP | `--stdio` for transport |

**Environment**: All `--options` have `GREYCAT_*` env equivalents. Use `.env` next to `project.gcl`.

**Dev mode**: `--user=1` bypasses auth (NEVER in production).

## Development Workflow Commands

Use `/greycat:command-name` in Claude Code:

| Command | Purpose | When |
|---------|---------|------|
| `/greycat:init` | Initialize CLAUDE.md | New projects |
| `/greycat:tutorial` | Interactive learning | Onboarding, learning |
| `/greycat:scaffold` | Generate models, services, APIs, tests | Starting features |
| `/greycat:migrate` | Schema evolution, imports, storage | Schema changes, bulk ops |
| `/greycat:upgrade` | Update libraries | Monthly maintenance |
| `/greycat:backend` | Backend review (dead code, anti-patterns) | Before releases |
| `/greycat:optimize` | Auto-fix performance issues | Quick checks |
| `/greycat:apicheck` | Review @expose endpoints | After endpoints added |
| `/greycat:coverage` | Test coverage + suggestions | After sprints |
| `/greycat:frontend` | Frontend review | Frontend features |
| `/greycat:docs` | Generate README, API docs | Before releases |
| `/greycat:typecheck` | Advanced type safety | After type changes |

## Validation Workflow

After every code change, check for errors using the LSP or lint:

1. **LSP diagnostics (preferred)** — appear automatically when editing `.gcl` files. Check diagnostics after each write/edit. The LSP provides inline errors, hover info, and go-to-definition.
2. **`greycat-lang lint` (fallback)** — use when LSP diagnostics are unavailable or for batch checking:

```bash
greycat-lang lint -p project.gcl          # check all project files
greycat-lang lint --fix -p project.gcl    # auto-fix what it can
```

Repeat until zero errors.

**Always open `project.gcl` first** before any other `.gcl` file. The LSP needs it to initialize the project context — opening other files first will not trigger analysis.

## Architecture

- `project.gcl` - Entry point, libs, permissions, roles, main(), init()

### Backend template
- `src/<feature>/<feature>.gcl` - Data models + global indices
- `src/<feature>/<feature>_api.gcl` - `@expose` functions and associated `@volatile` types
- `src/<feature>/<feature>_reader.gcl` - Readers (CSV, JSON, Parquet, etc.)
- `src/<feature>/<feature>_writer.gcl` - Writers
- `test/<feature>_test.gcl` - Tests

Small features: `src/<feature>.gcl` — one file is fine.

### Frontend template (optional)
See [references/frontend.md](references/frontend.md) for full setup.

**project.gcl**:
```gcl
@library("std", "<version>");       // required — fetch latest: curl https://get.greycat.io/files/core/stable/latest
@library("explorer", "<version>");  // administration app served at /explorer

@include("src");
@include("test");

fn main() {}
```

**Fetching latest library versions**: `curl https://get.greycat.io/files/<lib>/stable/latest` — returns `<MAJOR>.<MINOR>/<MAJOR>.<MINOR>.<PATCH>-<BRANCH>` (use the part after `/`). Exception: `std` uses `core` as the URL path (all others match: `ai`, `explorer`, `algebra`, etc.).

**Conventions**: snake_case files, PascalCase types, `_prefix` unused, `*_test.gcl` tests

## Types

**Primitives**: `int` (64-bit, `1_000_000`), `float` (`3.14`), `bool`, `char`, `String` (`"${name}"`)

**Casting — `as int` vs `floor()`**:
- `x as int` — **ROUNDS** (nearest integer): `0.5 as int` → 1, `2.4 as int` → 2
- `floor(x) as int` — **FLOORS** (truncates toward −∞): `0.5` → 0, `2.9` → 2

**CRITICAL**: For indices/buckets from float division, use `floor(x) as int`, NOT `x as int`.

**String→number**: `parseNumber(s)` returns `any`. Cast: `parseNumber(s) as float` or `parseNumber(s) as int`.

**Time**: `time` (μs epoch), `duration` (`1_us`, `500_ms`, `5_s`, `30_min`, `7_hour`, `2_day`), `Date` (UI, needs timezone). Convert: `Date::parse("2025-01-15 08:00", "%Y-%m-%d %H:%M").to_time(TimeZone::UTC)`. Use `TimeZone::` enum values (e.g. `TimeZone::"Europe/Luxembourg"`), never raw strings.

**Geo**: `geo{lat, lng}` (positional, no field names). Access via **methods**: `location.lat()`, `location.lng()`, `location.distance(other)`. Shapes: `GeoBox`, `GeoCircle`, `GeoPoly` (`.contains(geo)`)

```gcl
var list = Array<String>{}; var map = Map<String, int>{};  // use {}, NOT ::new()
@volatile type ApiResponse { data: String; }  // non-persisted
```

**NO TERNARY** — use if/else: `if (valid) { result = "yes"; } else { result = "no"; }`

## Nullability

Non-null by default. Use `?` for nullable:
```gcl
var city: City?;                   // nullable
city?.name?.size();                // optional chaining
city?.name ?? "Unknown";           // nullish coalescing
data.get("key")!!;                 // non-null assertion (use sparingly)
```

**Control flow narrowing** — after a null guard, the type is narrowed automatically:
```gcl
var n = index.get(id);  // returns node<T>?
if (n == null) {
    return null;
}
// n is now node<T> — NO !! needed
n.resolve();   // valid
n->name;       // valid
```

**CRITICAL**: Do NOT use `!!` after `if (x == null) { return; }` — the LSP will warn about unnecessary non-null assertions. Same applies inside `if (x != null) { ... }` blocks.

**Paren expr for cast + coalescing**: `(answer as String?) ?? "default"` NOT `answer as String? ?? "default"`

## Nodes (Persistence)

Nodes are 64b references to persistent data.

```gcl
type Country { name: String; code: int; }
var n = node<Country>{ Country { name: "LU", code: 352 } };
```

**Node access — three distinct operations:**
```gcl
*n;            // dereference: returns the inner T value
n->name;       // field access: shorthand for (*n).name
n.resolve();   // node method: returns T? (nullable)
```

**CRITICAL — `->` calls methods on the inner type, `.` calls methods on `node` itself:**
```gcl
n->name;       // accesses Country.name (field on inner type)
n.resolve();   // calls node.resolve() (method on node, returns T?)
n->resolve();  // WRONG: calls Country.resolve() which doesn't exist
```

**Use node refs to share data**: `type City { country: node<Country>; }` — light 64b ref vs full copy.

**Single ownership**: objects belong to ONE node. For multi-index, store `node<T>` refs:
```gcl
var by_id = nodeList<node<Item>> {};
var by_name = nodeIndex<String, node<Item>> {};
var item = node<Item> { Item {} };
by_id.set(1, item); by_name.set("x", item); // both share same node
```

**Transactions**: atomic per function, rollback on error.

More patterns → [references/nodes.md](references/nodes.md)

## Node Collections

| Key    | In-Memory      | Persisted               |
| ------ | -------------- | ----------------------- |
| `int`  | `Array<T>`     | `nodeList<node<T>>`     |
| `K`    | `Map<K, V>`    | `nodeIndex<K, node<V>>` |
| `time` | `Map<time, V>` | `nodeTime<node<T>>`     |
| `geo`  | `Map<geo, V>`  | `nodeGeo<node<T>>`      |

Other: `Stack<T>`, `Queue<T>`, `Set<T>`

```gcl
var ni = nodeIndex<String, node<X>> {};
ni.set("key", val); ni.get("key");  // set/get, NOT add

var nt = nodeTime<float> {};
nt.setAt(t1, 20.5);
for (t, v in nt[from..to]) {}

var nl = nodeList<node<X>> {};
for (i, v in nl[0..100]) {}
```

**Sampling**: `nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null)`

**Sort**: `cities.sort_by(City::population, SortOrder::desc);`

**Initialize all non-nullable fields, including node collections inside type constructors:**
```gcl
var b = Box {};        // WRONG: non-nullable fields unset
var b = Box { x: 42 }; // RIGHT

var n = node<String> {};         // WRONG
var n = node<String> { "text" }; // RIGHT
var n = node<String?> {};        // RIGHT (nullable generic)

// Node collections inside types need explicit {} in constructors:
type City { name: String; streets: nodeIndex<String, node<Street>>; }
var c = node<City> { City { name: "Paris" } };                                          // WRONG
var c = node<City> { City { name: "Paris", streets: nodeIndex<String, node<Street>> {} } }; // RIGHT
```

## Module Variables

Root-level variables must be nodes — graph entrypoints. Auto-initialized (no `{}` needed):
```gcl
var count: node<int?>;   // nullable generic for node
var by_id: nodeList<float>;
var cities: nodeIndex<String, node<City>>;
```

## Modules

**Models** — global indices + types:
```gcl
var cities_by_name: nodeIndex<String, node<City>>;
type City { name: String; country: node<Country>; }
```

**API** — return `Array<XxxView>` with `@volatile`, never nodeList:
```gcl
@volatile
type CityView { name: String; country_name: String; }
@expose
fn getCities(): Array<CityView> { ... }
```

**MCP exposure** (sparingly — only high-value APIs):
```gcl
/// Search cities by name
/// @param query City name or partial match
@expose
@tag("mcp")
fn searchCities(query: String): Array<CityView> { ... }
```

## Functions & Control Flow

```gcl
fn add(x: int): int { return x + 2; }
fn noReturn() { }  // no void keyword
var lambda = fn(x: int): int { x * 2 };
for (k, v in map) { }
for (i, v in nullable?) { }  // use ? for nullable iterables
```

**Function parameters**: use `function` keyword, calls return `any?` — always cast:
```gcl
fn applyAll(arr: Array<any>, f: function) {
    for (var i = 0; i < arr.size(); i++) { f(arr[i]); }
}
```

## Modelling

```gcl
abstract type Building { address: String; fn calculateTax(): float; }
type House extends Building { fn calculateTax(): float { return value * 0.01; } }
```

More → [references/modelling.md](references/modelling.md)

## Logging & Error Handling

```gcl
info("msg ${var}"); warn("msg"); error("msg");
try { op(); } catch (ex) { error("${ex}"); }
```

## Parallelization

```gcl
var jobs = Array<Job<ResultType>> {};
for (item in items) { jobs.add(Job<ResultType> { function: processFn, arguments: [item] }); }
await(jobs, MergeStrategy::strict);
for (job in jobs) { results.add(job.result()); }
```

More → [references/concurrency.md](references/concurrency.md)

## Testing

Run `greycat test --quiet`. Single test: `greycat test module_name::test_fn_name`. Single module: `greycat test module_name`.

More → [references/testing.md](references/testing.md)

## Common Pitfalls

**Reserved keywords**: `limit`, `node`, `type`, `var`, `fn` — do NOT use as variable/param names.

| Wrong | Correct |
|-------|---------|
| `Array<T>::new()` | `Array<T>{}` |
| `(*node)->field` | `node->field` |
| `n->resolve()` | `*n` or `n.resolve()` |
| `x!!` after null guard return | `x` (control flow narrows) |
| `@permission("api") fn getX()` | `@expose @permission("api") fn getX()` |
| `nodeList<City>` | `nodeList<node<City>>` for complex types |
| `nodeIndex.add(k, v)` | `nodeIndex.set(k, v)` |
| `for(i, v in nullable_list)` | `for(i, v in nullable_list?)` |
| `fn doX(): void` | `fn doX()` |
| `fn somefn(f: fn(T): R)` | `fn somefn(f: function)` |
| `(x / y) as int` for floor | `floor(x / y) as int` |
| `geo { lat: x, lng: y }` | `geo{x, y}` (positional, no spaces/names) |
| `date.to_time("UTC")` | `date.to_time(TimeZone::UTC)` (enum) |
| `date.toTime(tz)` | `date.to_time(tz)` (snake_case) |
| `City { name: "X" }` (missing nodeIndex) | `City { name: "X", streets: nodeIndex<..> {} }` |

**Native (primitive) types** (`geo`, `time`, `duration`, etc.) have no fields — use methods for access, never field syntax. Construction varies by type: `geo` uses positional `geo{lat, lng}`, `time` uses literals (`5_time`) or static methods (`time::now()`, `time::new(epoch, unit)`, `time::parse(str, fmt)`), `duration` uses literals (`1_us`, `500_ms`) or static methods (`duration::new(v, unit)`).

## ABI & Database

Adding a non-nullable field to an existing type breaks the ABI. Fix: make new fields nullable (`int?`).

```gcl
type Foo { name: String; new_field: int?; }  // nullable = safe ABI update
```

## Full-Stack Development

**[references/frontend.md](references/frontend.md)** — @greycat/web SDK, JSX, web components, codegen, auth.

## Local LLM Integration

```gcl
@library("ai", "<version>");  // fetch latest: curl https://get.greycat.io/files/ai/stable/latest

fn main() {
    var model = Model::load("llama", "./model.gguf", ModelParams { n_gpu_layers: -1 });
    var result = model.chat([ChatMessage { role: "user", content: "Hello!" }], null, null);
}
```

## Libraries & References

**[references/libraries.md](references/libraries.md)** — available libraries with `@library()` declarations.

Use the LSP (hover, completion, go-to-def) for type signatures and API details.

**Versions & downloads**: https://get.greycat.io | **Docs**: https://doc.greycat.io/ | **CLI**: [references/cli.md](references/cli.md)
