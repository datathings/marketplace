---
name: greycat
description: "Use when working with .gcl files or GreyCat projects - efficient language with unified temporal/graph/vector database, built-in web server, native MCP for billion-scale digital twins"
---

# GreyCat

Unified language + database (temporal/graph/vector) + web server + MCP. Built for billion-scale digital twins.

**Nav**: [Types](#types) • [Nullability](#nullability) • [Nodes](#nodes-persistence) • [Collections](#indexed-collections) • [Commands](#commands) • [Testing](#testing) • [Pitfalls](#common-pitfalls)

**Quick Start**:
```gcl
// Model + index
var users_by_id: nodeIndex<int, node<AppUser>>;

type AppUser {
    name: String;
    email: String;

    // CRUD service
    static fn create(name: String): node<AppUser> { var u = node<AppUser>{AppUser{name}}; users_by_id.set(u->id, u); return u; }
    static fn find(id: int): node<AppUser>? { return users_by_id.get(id); }
}

// API endpoint
@expose
fn getUsers(): Array<UserView> { /* ... */ }

// Time-series query
for (t: time, temp: float in temperatures[start..end]) { info("${t}: ${temp}"); }

// Parallel processing
var jobs = Array<Job<Result>> {};
for (item in items) { jobs.add(Job<Result>{function: process, arguments: [item]}); }
await(jobs, MergeStrategy::last_wins);
```

## Installation

Verify with `which greycat` or `greycat --version`. If not found, confirm with user before installing:

**Linux/Mac/FreeBSD**: `curl -fsSL https://get.greycat.io/install.sh | bash -s dev`
**Windows**: `iwr https://get.greycat.io/install_dev.ps1 -useb | iex`

Verify with `greycat --version`, restart shell if needed.

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
| `greycat-lang lint` | Check only | CI/CD pipelines |
| `greycat-lang fmt` | Format files | In-place |
| `greycat-lang server` | Start LSP | `--stdio` for IDE |

**Environment**: All `--options` have `GREYCAT_*` env equivalents. Use `.env` next to `project.gcl`.

**⚠️ CRITICAL**: After generating/modifying .gcl files, IMMEDIATELY run `greycat-lang lint --fix` and verify 0 errors.

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

## Language Server (LSP)

**[references/lsp.md](references/lsp.md)** - IDE integration (VS Code, Neovim, Emacs), diagnostics, programmatic clients.

**Quick start**: `greycat-lang server --stdio` for IDE features (completion, go-to-def, hover, diagnostics, formatting).

**CLI reference**: [references/cli.md](references/cli.md)

## Architecture

### Backend
**Feature layout**
- `project.gcl` - Entry point, libs, permissions, roles, main(), init()
- `backend/src/<feature>/<feature>.gcl` - Data models + global indices
- `backend/src/<feature>/<feature>_api.gcl` - @expose + @permission functions, @volatile response types
- `backend/src/<feature>/<feature>_import.gcl` - Import from another format
- `backend/src/<feature>/<feature>_export.gcl` - Export into another format
- `backend/test/<feature>_test.gcl` - Tests a feature

If the feature is small or has no API needed:
- `backend/src/<feature>.gcl` - Data models + global indices

### Frontend (optional, if required)
**Pages layout**
- `frontend/<page>/index.html` - *Optional*, if a frontend is required
- `frontend/<page>/index.tsx` - *Optional*, if a frontend is required
- `frontend/<page>/<other_page>.html` - *Optional*, if a frontend is required and multiple pages needed
- `frontend/<page>/<other_page>.tsx` - *Optional*, if a frontend is required and multiple pages needed

**project.gcl**:
```gcl
@library("std", "7.7.158-dev");    // required
@library("explorer", "7.7.0-dev"); // administration app served at /explorer

@include("backend");               // ⚠️ project.gcl only - includes ALL .gcl recursively

fn main() { }
```

**Conventions**: snake_case files, PascalCase types, `_prefix` unused, `*_test.gcl` tests

## Types

**Primitives**: `int` (64-bit, `1_000_000`), `float` (`3.14`), `bool`, `char`, `String` (`"${name}"`)

**Casting — `as int` vs `floor()`**:
- `x as int` — **ROUNDS** (nearest integer): `0.5 as int` → 1, `1.5 as int` → 2, `2.4 as int` → 2
- `floor(x) as int` — **FLOORS** (truncates toward −∞): `0.5` → 0, `1.5` → 1, `2.9` → 2

**⚠️ CRITICAL**: When computing indices, buckets, or anything from float division, use `floor(x) as int`, NOT `x as int`:
```gcl
var raw = 5.0 / 10.0;            // 0.5
var wrong = raw as int;          // ❌ 1 (rounds!)
var correct = floor(raw) as int; // ✅ 0 (floors!)
```

**String→number**: `parseNumber(s)` returns `any` (int or float). Cast: `var v = parseNumber(s) as float;` or `var i = parseNumber(s) as int;` (safe for integer strings like `"3"`, use `floor(parseNumber(s)) as int` if you expect truncation for `"3.7"`)

**Time**: `time` (μs epoch), `duration` (`1_us`, `500_ms`, `5_s`, `30_min`, `7_hour`, `2_day`), `Date` (UI, needs timezone)

**Geo**: `geo{lat, lng}` | Shapes: `GeoBox`, `GeoCircle`, `GeoPoly` (`.contains(geo)`)

```gcl
var list = Array<String>{}; var map = Map<String, int>{};  // ✅ use {}, NOT ::new()
@volatile type ApiResponse { data: String; }  // non-persisted
```

## Nullability

Non-null by default. Use `?` for nullable:
```gcl
var city: City?;                   // nullable
city?.name?.size();                // optional chaining
city?.name ?? "Unknown";           // nullish coalescing
data.get("key")!!;                 // non-null assertion
if (city == null) { return null; }
city->name;                        // ✅ no !! needed (control flow analysis)
```


**⚠️ Paren expr for cast + coalescing**: `(answer as String?) ?? "default"` NOT `answer as String? ?? "default"`

**⚠️ NO TERNARY** — use if/else: `if (valid) { result = "yes"; } else { result = "no"; }`

## Nodes (Persistence)

Nodes are 64b references to persistent data. Core graph storage mechanism.

```gcl
type Country { name: String; code: int; }
var o = Country { name: "LU", code: 352 }; // RAM only
var n = node<Country>{ o };                // persisted

*n;            // dereference
n->name;       // equivalent to `(*n).name`
n.resolve();   // calls method on the `node` type, not the inner `T`
n->name = "X"; // modify object field (mutates the graph)
n.set(5);      // replace the inner data with the given `T`
```

**Use node refs to share data**: `type City { country: node<Country>; }` light 64b value vs full Country object

**Single ownership model**: objects belong to **ONE** node only. For multi-index, store nodes:
```gcl
var by_id = nodeList<node<Item>> {};
var by_name = nodeIndex<String, node<Item>> {};
var item = node<Item> { Item {} }; // only one item
by_id.set(1, item); by_name.set("x", item); // both share the same node to item
```

**Transactions**: atomic per function, rollback on error.

**Patterns, multi-index, transaction safety** → [references/nodes.md](references/nodes.md)

## Node collections

| Key    | In-Memory      | Persisted               |
| ------ | -------------- | ----------------------- |
| `int`  | `Array<T>`     | `nodeList<node<T>>`     |
| `K`    | `Map<K, V>`    | `nodeIndex<K, node<V>>` |
| `time` | `Map<time, V>` | `nodeTime<node<T>>`     |
| `geo`  | `Map<geo, V>`  | `nodeGeo<node<T>>`      |

**Other collections**: `Stack<T>`, `Queue<T>`, `Set<T>`

```gcl
// nodeTime - interpolates between points
var nt = nodeTime<float> {};
nt.setAt(t1, 20.5);
for (t, v in nt[from..to]) {}

// nodeIndex - uses set/get (NOT add)
var ni = nodeIndex<String, node<X>> {};
ni.set("key", val); ni.get("key");

// nodeList
var nl = nodeList<node<X>> {};
for (i, v in nl[0..100]) {}

// nodeGeo
var ng = nodeGeo<node<B>> {};
for (pos, b in ng.filter(GeoBox { /*...*/ })) {}
```

**Sampling**: `nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null)` — Modes: `fixed`, `fixed_reg`, `adaptative`, `dense`

**Sort**: `cities.sort_by(City::population, SortOrder::desc);`

**⚠️ CRITICAL: Initialize non-nullable fields and nodes generics can never be nullable**
```gcl
type Box { x: int; }
var b = Box {};        // WRONG: `x` is non-nullable
var b = Box { x: 42 }; // RIGHT

var n = node<String> {};         // WRONG: `String` is not nullable
var n = node<String> { "text" }; // RIGHT
var n = node<String?> {};        // RIGHT
```

## Module Variables

Root-level variables must be nodes → graph entrypoints:
```gcl
var count: node<int?>; // node generic param needs to be nullable
var by_id: nodeList<float>; // node collections do not need nullable generic param
fn main() { count.set((count.resolve() ?? 0) + 1); }
```

**Module variables are auto-initialized**: `node`, `nodeIndex`, `nodeList`, `nodeTime`, `nodeGeo` are automatically initialized by GreyCat — no `{}` needed:
```gcl
// ✅ Global variables — no initialization needed
var cities_by_name: nodeIndex<String, node<City>>;
var all_users: nodeList<node<User>>;

// ⚠️ Non-nullable object fields still need initialization
```

## Modules

**Models** — store node refs, global indices first:
```gcl
var cities_by_name: nodeIndex<String, node<City>>;
type City { name: String; country: node<Country>; streets: nodeList<node<Street>>; }
```

**API** — return `Array<XxxView>` with `@volatile`, never nodeList:
```gcl
@volatile
type CityView { name: String; country_name: String; street_count: int; }
@expose
fn getCities(): Array<CityView> { ... }  // ⚠️ REQUIRES @expose for HTTP
```

**MCP exposure** (sparingly — only high-value APIs):
```gcl
/// Explain the behavior for LLM to know about the tool
///
/// @param query Explain the query param (eg. format constraints, etc.)
@expose
@tag("mcp")
fn searchCities(query: String): Array<CityView> { ... }
```

MCP-exposed endpoints should always have documentation that explains the tool

## Functions & Control Flow

```gcl
fn add(x: int): int { return x + 2; }; fn noReturn() { }  // no void type
var lambda = fn(x: int): int { x * 2 };
for (k: K, v: V in map) { }; for (i, v in nullable?) { }  // ✅ use ? for nullable
```

// First-class function parameters — use `function` keyword (not fn(T): R)
// Calling a `function` parameter returns `any?` — cast the result
```gcl
abstract type ArrayUtils {
    static fn filter(arr: Array<any>, pred: function): Array<any> {
        var result = Array<any> {};
        for (var i = 0; i < arr.size(); i++) {
            if (pred(arr[i]) as bool) { result.add(arr[i]); }
        }
        return result;
    }
}

fn isEven(x: int): bool { return x % 2 == 0; }
var evens = ArrayUtils::filter(nums, isEven);  // pass named function by reference
```
**Key rules**: declare param as `function` (not `fn(T): R`); calls return `any?` — always cast (`f(x) as bool`, `f(x) as int`); pass by name at call site (no lambda syntax needed).

## Patterns

```gcl
// Inheritance: abstract methods, polymorphism
abstract type Building { address: String; fn calculateTax(): float; }
type House extends Building { fn calculateTax(): float { return value * 0.01; } }
```

**Patterns, CRUD, inheritance, polymorphism** → [references/patterns.md](references/patterns.md)

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

**Key**: `Job<T>` generic, `MergeStrategy::strict`, no nested await. **Worker pools, PeriodicTask, async HTTP, patterns** → [references/concurrency.md](references/concurrency.md)

## Testing

Run `greycat test`. Test files: `*_test.gcl` in `./backend/test/`. Run a single test: `greycat test module_name::test_fn_name` (e.g., `greycat test dfr_engine_test::test_dfr_variant`). Run all tests from a single module `greycat test module_name`

```gcl
fn setup() { /* runs before tests */ }
fn teardown() { /* cleanup after tests */ }
@test
fn some_test() {
    Assert::equals(1, 1);
}
```

**Assert**: `equals(a, b)`, `equalsd(a, b, epsilon)`, `isTrue(v)`, `isFalse(v)`, `isNull(v)`, `isNotNull(v)`. **Organization, mocking, fixtures, CI/CD** → [references/testing.md](references/testing.md)

## Common Pitfalls

**⚠️ Reserved Keywords**: `limit`, `node`, `type`, `var`, `fn` — do NOT use as variable/attribute names:
```gcl
// ❌ WRONG
fn process(fn: String) { } // reserved!
fn foo() { var var; }      // reserved!

// ✅ CORRECT
fn process(f: String) { }
fn foo() { var v; }
```

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `Array<T>::new()` | `Array<T>{}` |
| `(*node)->field` | `node->field` |
| `@permission(public)` | `@permission("public")` |
| `@permission("api") fn getX()` | `@expose @permission("api") fn getX()` |
| `for(i=0;i<n;i++)` | `for (i, v in list)` |
| `nodeList<City>` | `nodeList<node<City>>` for complex types |
| `nodeIndex.add(k, v)` | `nodeIndex.set(k, v)` |
| `for(i, v in nullable_list)` | `for(i, v in nullable_list?)` |
| `fn doX(): void` | `fn doX()` |
| `fn somefn(f: fn(T): R)` | `fn somefn(f: function)` |
| `(x / y) as int` for floor | `floor(x / y) as int` — `as int` rounds, `floor()` truncates |

**Non-null assertions** are fine in the case of control-flow analysis not being able to understand nullability in complex flows

## ABI & Database

```gcl
// v0
type Foo {
    type: String;
}
```

```gcl
// v1
type Foo {
    type: String;
    foo: int;
}
```
Running greycat will output: `ERROR: abi update failed: can't add a non-nullable field "foo" in type "project.Foo"`

Fix:
```gcl
// v1
type Foo {
    type: String;
    foo: int?; // nullable field is always valid for updates
}
```

## Full-Stack Development

**[references/frontend.md](references/frontend.md)** - @greycat/web SDK guide: codegen, TypeScript/JavaScript, auth, API patterns, error handling.

## Local LLM Integration

**[references/ai/llm.md](references/ai/llm.md)** - llama.cpp integration: model loading, text gen, chat, embeddings, LoRA.
```gcl
@library("ai", "7.7.164-dev");

fn main() {
    var model = Model::load("llama", "./model.gguf", ModelParams { n_gpu_layers: -1 });
    var result = model.chat([ChatMessage { role: "user", content: "Hello!" }], null, null);
}
```

## Library References

**[references/LIBRARIES.md](references/LIBRARIES.md)** - Complete type definitions and API signatures.

**Core**: std (types, collections, I/O, runtime, util) · explorer (graph UI)
**AI/ML**: [ai](references/ai/llm.md) (LLM) · [algebra](references/algebra/) (ML, NN, patterns, transforms, clustering, climate)
**Integrations**: [kafka](references/kafka/kafka.md) · [postgres](references/sql/postgres.md) · [s3](references/s3/s3.md) · [opcua](references/opcua/opcua.md) · [ftp](references/ftp/ftp.md) (FTP/FTPS) · [ssh](references/ssh/ssh.md) (SFTP)
**Domain**: [finance](references/finance/finance.md) (IBAN) · [powerflow](references/powerflow/powerflow.md) · [powergrid](references/powergrid/powergrid.md) · [fcs](references/fcs/fcs.md) (flow cytometry) · [useragent](references/useragent/useragent.md)

**CLI**: [references/cli.md](references/cli.md) | **Docs**: https://doc.greycat.io/
