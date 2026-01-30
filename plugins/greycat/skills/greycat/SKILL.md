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
var users_by_id: nodeIndex<int, node<User>>;
type User { name: String; email: String; }

// CRUD service
abstract type UserService {
    static fn create(name: String): node<User> { var u = node<User>{User{name}}; users_by_id.set(u->id, u); return u; }
    static fn find(id: int): node<User>? { return users_by_id.get(id); }
}

// API endpoint
@expose @permission("public") fn getUsers(): Array<UserView> { /* ... */ }

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
| `/greycat:frontend` | React/TS frontend review | Frontend features |
| `/greycat:docs` | Generate README, API docs | Before releases |
| `/greycat:typecheck` | Advanced type safety | After type changes |

## Language Server (LSP)

**[references/lsp.md](references/lsp.md)** - IDE integration (VS Code, Neovim, Emacs), diagnostics, programmatic clients.

**Quick start**: `greycat-lang server --stdio` for IDE features (completion, go-to-def, hover, diagnostics, formatting).

**CLI reference**: [references/cli.md](references/cli.md)

## Architecture

**Dirs**: `project.gcl` (entry), `backend/src/model/` (models+indices), `backend/src/service/` (XxxService), `backend/src/api/` (@expose), `backend/src/edi/` (import/export)

**project.gcl**:
```gcl
@library("std", "7.6.158-dev");           // required
@library("explorer", "7.6.13-dev");      // graph UI /explorer (dev)
@include("backend");                     // ⚠️ project.gcl only - includes ALL .gcl

@permission("app.admin", "description");
@role("admin", "app.admin", "public", "admin", "api", "files");

@format_indent(4); @format_line_width(280);
fn main() { }
```

**Conventions**: snake_case files, PascalCase types, `_prefix` unused, `*_test.gcl` tests

## Types

**Primitives**: `int` (64-bit, `1_000_000`), `float` (`3.14`), `bool`, `char`, `String` (`"${name}"`)

**Casting**: Float→int rounds (≥0.5 up, <0.5 down): `var i = 42.9 as int; // 43`

**Time**: `time` (μs epoch), `duration` (`1_us`, `500_ms`, `5_s`, `30_min`, `7_hour`, `2_day`), `Date` (UI, needs timezone)

**Geo**: `geo{lat, lng}` | Shapes: `GeoBox`, `GeoCircle`, `GeoPoly` (`.contains(geo)`)

```gcl
var list = Array<String>{}; var map = Map<String, int>{};  // ✅ use {}, NOT ::new()
@volatile type ApiResponse { data: String; }  // non-persisted
```

## Nullability

Non-null by default. Use `?` for nullable:
```gcl
var city: City?;  // nullable
city?.name?.size(); city?.name ?? "Unknown"; data.get("key")!!;
if (country == null) { return null; }
return country->name;  // ✅ no !! after null check
```

**⚠️ Parens for cast + coalescing**: `(answer as String?) ?? "default"` NOT `answer as String? ?? "default"`

**⚠️ NO TERNARY** — use if/else: `if (valid) { result = "yes"; } else { result = "no"; }`

## Nodes (Persistence)

64-bit refs to persistent containers:
```gcl
type Country { name: String; code: int; }
var obj = Country { name: "LU", code: 352 };  // RAM
var n = node<Country>{obj};                    // persisted
*n; n->name; n.resolve(); n->name = "X"; node<int>{0}.set(5);
```

**Sharing**: `type City { country: node<Country>; }` (64-bit ref) vs embedded (heavy)

**Multi-index ownership** (objects belong to ONE node, store refs):
```gcl
var by_id = nodeList<node<Item>>{}; var by_name = nodeIndex<String, node<Item>>{};
var item = node<Item>{ Item{} };
by_id.set(1, item); by_name.set("x", item);  // both point to same
```

**Transactions**: Atomic per function, rollback on error. **Patterns, multi-index, transaction safety** → [references/nodes.md](references/nodes.md)

## Indexed Collections

| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `node<T>` | — | `Array<T>`, `Map<K,V>` |
| `nodeList<node<T>>` | int | `Stack<T>`, `Queue<T>` |
| `nodeIndex<K, node<V>>` | hash | `Set<T>`, `Tuple<A,B>` |
| `nodeTime<node<T>>` | time | `Buffer`, `Table`, `Tensor` |
| `nodeGeo<node<T>>` | geo | `TimeWindow`, `SlidingWindow` |

```gcl
var temps = nodeTime<float>{}; temps.setAt(t1, 20.5);
for (t: time, v: float in temps[from..to]) { }

var idx = nodeIndex<String, node<X>>{}; idx.set("key", val); idx.get("key");  // ⚠️ set/get, NOT add
var list = nodeList<node<X>>{}; for (i: int, v in list[0..100]) { }
var geo_idx = nodeGeo<node<B>>{}; for (pos: geo, b in geo_idx.filter(GeoBox{...})) { }
```

**Sampling**: `nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null)` — Modes: `fixed`, `fixed_reg`, `adaptative`, `dense`

**Sort**: `cities.sort_by(City::population, SortOrder::desc);`

**⚠️ CRITICAL**: Non-nullable `nodeList`, `nodeIndex`, `nodeTime`, `Array` attributes MUST initialize:
```gcl
var city = node<City>{ City{ name: "Paris", country: country_node,
    streets: nodeList<node<Street>>{}  }};  // ⚠️ MUST init!
```

## Module Variables

Root-level vars must be nodes/indexes (auto-persisted):
```gcl
var count: node<int?>; fn main() { count.set((count.resolve() ?? 0) + 1); }
```

**Global indices auto-initialize**: Module `nodeIndex`/`nodeList`/`nodeTime`/`nodeGeo` — no `{}` needed. Collection ATTRIBUTES in types still need `{}`.

## Model vs API Types

**Models** — store node refs, global indices first:
```gcl
var cities_by_name: nodeIndex<String, node<City>>;
type City { name: String; country: node<Country>; streets: nodeList<node<Street>>; }
```

**API** — return `Array<XxxView>` with `@volatile`, never nodeList:
```gcl
@volatile type CityView { name: String; country_name: String; street_count: int; }
@expose @permission("public") fn getCities(): Array<CityView> { ... }  // ⚠️ REQUIRES @expose for HTTP
```

**MCP exposure** (sparingly — only high-value APIs):
```gcl
@expose @tag("mcp") @permission("public") fn searchCities(query: String): Array<CityView> { ... }
```

## Functions & Control Flow

```gcl
fn add(x: int): int { return x + 2; }; fn noReturn() { }  // no void type
var lambda = fn(x: int): int { x * 2 };
for (k: K, v: V in map) { }; for (i, v in nullable?) { }  // ✅ use ? for nullable
```

## Services & Patterns

```gcl
// Service pattern: static functions for business logic
abstract type CountryService {
    static fn create(name: String): node<Country> { ... }
    static fn find(name: String): node<Country>? { return countries_by_name.get(name); }
}

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
await(jobs, MergeStrategy::last_wins);
for (job in jobs) { results.add(job.result()); }
```

**Key**: `Job<T>` generic, `MergeStrategy::last_wins`, no nested await. **Worker pools, PeriodicTask, async HTTP, patterns** → [references/concurrency.md](references/concurrency.md)

## Testing

Run `greycat test`. Test files: `*_test.gcl` in `./backend/test/`.

```gcl
@test fn test_city_creation() {
    var city = City::create("Paris", country_node);
    Assert::equals(city->name, "Paris");
}
fn setup() { /* runs before tests */ }
fn teardown() { /* cleanup after tests */ }
```

**Assert**: `equals(a, b)`, `equalsd(a, b, epsilon)`, `isTrue(v)`, `isFalse(v)`, `isNull(v)`, `isNotNull(v)`. **Organization, mocking, fixtures, CI/CD** → [references/testing.md](references/testing.md)

## Common Pitfalls

**⚠️ Reserved Keywords**: `limit`, `node`, `type`, `var`, `fn` — do NOT use as variable/attribute names:
```gcl
// ❌ WRONG
type User { limit: int; type: String; }  // reserved!
fn process(node: String) { }             // reserved!

// ✅ CORRECT
type User { max_limit: int; user_type: String; }
fn process(node_name: String) { }
```

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `Array<T>::new()` | `Array<T>{}` |
| `(*node)->field` | `node->field` |
| `@permission(public)` | `@permission("public")` |
| `@permission("api") fn getX()` | `@expose @permission("api") fn getX()` |
| `for(i=0;i<n;i++)` | `for (i, v in list)` |
| `nodeList<City>` | `nodeList<node<City>>` |
| `fn getX(): nodeList<...>` | `fn getX(): Array<XxxView>` |
| `nodeIndex.add(k, v)` | `nodeIndex.set(k, v)` |
| `for(i, v in nullable_list)` | `for(i, v in nullable_list?)` |
| `fn doX(): void` | `fn doX()` |
| `City{name: "X"}` | `City{name: "X", streets: nodeList<...>{}}` |

**Double-bang OK** for global registry lookups: `var config = ConfigRegistry::getConfig(key)!!;`

## ABI & Database

**DEV**: Delete deprecated fields. Reset `gcdata/` on schema changes. Add non-nullable → make nullable first: `newField: int?`
```bash
rm -rf gcdata && greycat run import  # ⚠️ DELETES DATA - confirm first
```

## Full-Stack Development

**[references/frontend.md](references/frontend.md)** - React+GreyCat guide (1,013 lines): @greycat/web SDK, TypeScript codegen, auth, React Query, error handling.

## Local LLM Integration

**[references/ai/llm.md](references/ai/llm.md)** - llama.cpp integration: model loading, text gen, chat, embeddings, LoRA.
```gcl
@library("ai", "7.6.60-dev");
var model = Model::load("llama", "./model.gguf", ModelParams { n_gpu_layers: -1 });
var result = model.chat([ChatMessage { role: "user", content: "Hello!" }], null, null);
```

## Library References

**[references/LIBRARIES.md](references/LIBRARIES.md)** - Complete catalog: std, ai, algebra, kafka, sql, s3, opcua, finance, powerflow, useragent.

**CLI**: [references/cli.md](references/cli.md) | **Docs**: https://doc.greycat.io/
