---
name: greycat
description: "GreyCat is an efficient, scalable programming language with unified temporal/graph/vector database, built-in web server, and native MCP capability—the all-in-one backend for modern large-scale digital twins. Use when: (1) working with .gcl files or GreyCat projects, (2) using persisted nodes and indexed collections (nodeList, nodeIndex, nodeTime, nodeGeo), (3) creating data models, services, or abstract types, (4) writing API endpoints with @expose, @permission, @tag, or @volatile decorators, (5) implementing parallel processing with Jobs, await(), or PeriodicTask, (6) integrating React frontends with @greycat/web SDK or TypeScript type generation, (7) running GreyCat CLI commands (greycat serve/test/run/install/lint), (8) debugging GreyCat applications or working with transactions, (9) exposing functions as MCP tools with @tag(\"mcp\")."
---

# GreyCat: The All-in-One Backend

Efficient, scalable programming language with unified temporal/graph/vector database, built-in web server, and native MCP integration. Eliminates the separation between code and data—write imperative code that directly evolves your data structure. Perfect for modern large-scale digital twins handling billions of data points.

## Quick Navigation

**Core Concepts**: [Types](#types) • [Nullability](#nullability) • [Nodes](#nodes-persistence) • [Collections](#indexed-collections) • [Functions](#functions--control-flow)
**Development**: [Commands](#commands) • [Workflows](#development-workflow-commands) • [Testing](#testing) • [Pitfalls](#common-pitfalls)
**Advanced**: [Parallelization](#parallelization) • [Frontend](references/frontend.md) • [Libraries](references/LIBRARIES.md)

**Quick Recipes**:
```gcl
// Create model with index
var users_by_id: nodeIndex<int, node<User>>;
type User { name: String; email: String; }

// Add CRUD service
abstract type UserService {
    static fn create(name: String): node<User> { var u = node<User>{User{name: name}}; users_by_id.set(u->id, u); return u; }
    static fn find(id: int): node<User>? { return users_by_id.get(id); }
}

// Expose API endpoint
@expose @permission("public") fn getUsers(): Array<UserView> { /* ... */ }

// Query time-series
for (t: time, temp: float in temperatures[start..end]) { info("${t}: ${temp}"); }

// Parallel processing
var jobs = Array<Job<Result>> {}; for (item in items) { jobs.add(Job<Result>{function: process, arguments: [item]}); }
await(jobs, MergeStrategy::last_wins);
```

## Installation

**Before using GreyCat**, verify installation with `which greycat` or `greycat --version`.

**If greycat is not found**, confirm with user before installing:

**Linux, Mac, FreeBSD (x64, arm64)**:
```bash
curl -fsSL https://get.greycat.io/install.sh | bash -s dev
```

**Windows (x64, arm64)**:
```powershell
iwr https://get.greycat.io/install_dev.ps1 -useb | iex
```

After installation, verify with `greycat --version` and restart shell if needed.

## Commands

| Command | Description | Options |
|---------|-------------|---------|
| `greycat build` | Compile project | `--log`, `--cache` |
| `greycat serve` | Start server (HTTP + MCP) | `--port=8080`, `--workers=N`, `--user=1` (dev) |
| `greycat run` | Execute main() or function | `greycat run myFunction` |
| `greycat test` | Run @test functions | Exit code 0 on success |
| `greycat install` | Download dependencies | From project.gcl @library |
| `greycat codegen` | Generate typed headers | TS, Python, C, Rust |
| `greycat defrag` | Compact storage | Safe anytime |
| `greycat-lang lint --fix` | Check and auto-fix errors | **Run after code changes** |
| `greycat-lang lint` | Check only (no fixes) | For CI/CD pipelines |
| `greycat-lang fmt` | Format files | In-place |
| `greycat-lang server` | Start LSP | `--stdio` for IDE |

**Environment**: All `--options` have `GREYCAT_*` equivalents. Use `.env` next to `project.gcl` for config.

**⚠️ CRITICAL**: After generating/modifying .gcl files, IMMEDIATELY run `greycat-lang lint --fix` and verify 0 errors before proceeding.

**Dev mode**: `--user=1` bypasses auth (NEVER in production).

## Development Workflow Commands

The greycat plugin provides Claude Code commands for common GreyCat development workflows. Use these with `/greycat:command-name`:

| Command | Description | When to Use |
|---------|-------------|-------------|
| `/greycat:init` | Initialize CLAUDE.md with GreyCat development guidelines | Starting new project, setting up Claude Code |
| `/greycat:tutorial` | Interactive learning modules for GreyCat concepts | Onboarding, learning features, refreshing knowledge |
| `/greycat:scaffold` | Generate models, services, APIs, tests with templates | Starting features, adding CRUD, creating entities |
| `/greycat:migrate` | Schema evolution, data migrations, import/export, storage health | Schema changes, bulk operations, database maintenance |
| `/greycat:upgrade` | Update all GreyCat libraries to latest versions | Monthly maintenance, before releases |
| `/greycat:backend` | Comprehensive backend review (dead code, duplications, anti-patterns, performance) | After sprints, before releases, during refactoring |
| `/greycat:optimize` | Detect and auto-fix performance anti-patterns | Quick performance checks, optimization needs |
| `/greycat:apicheck` | Review @expose endpoints for security, performance, best practices | After adding endpoints, before releases |
| `/greycat:coverage` | Generate test coverage report and suggest new tests | After sprints, before releases, when adding features |
| `/greycat:frontend` | Review React/TypeScript frontend for quality and performance | After sprints, when adding frontend features |
| `/greycat:docs` | Generate/update README, API docs, and MCP documentation | After sprints, before releases, when APIs change |
| `/greycat:typecheck` | Advanced type safety checks beyond greycat-lang lint | After type changes, before releases |

**Example usage**:
```bash
/greycat:tutorial          # Learn GreyCat interactively
/greycat:scaffold          # Generate model + service + API + tests
/greycat:migrate           # Handle schema changes and migrations
/greycat:optimize          # Quick performance analysis and fixes
/greycat:backend           # Comprehensive code review and cleanup
```

**Note**: These commands guide Claude through comprehensive workflows for GreyCat development. They complement the core `greycat` CLI commands above.

## Language Server (LSP)

**[references/lsp.md](references/lsp.md)** - Comprehensive LSP guide: IDE integration (VS Code, Neovim, Emacs), real-time diagnostics, programmatic GCL clients, Claude Code integration.

**Quick start**: Run in background `greycat-lang server --stdio` for IDE features (completion, go-to-definition, hover, diagnostics, formatting).

**CLI reference**: [references/cli.md](references/cli.md) for all commands, options, environment variables.

## Architecture

**Directories**: `project.gcl` (entry, libs, roles), `backend/src/model/` (models+indices), `backend/src/service/` (XxxService::create/find), `backend/src/api/` (@expose functions), `backend/src/edi/` (import/export)

**project.gcl:**
```gcl
@library("std", "7.6.16-dev");           // required
@library("explorer", "7.6.0-dev");      // graph UI at /explorer (dev)
@include("backend");                     // ⚠️ ONLY in project.gcl - recursively includes ALL .gcl

@permission("app.admin", "description");
@role("admin", "app.admin", "public", "admin", "api", "files");

@format_indent(4); @format_line_width(280);
fn main() { }
```

**Conventions**: snake_case files, PascalCase types, `_prefix` unused vars, `*_test.gcl` tests

## Types

**Primitives:** `int` (64-bit, `1_000_000`), `float` (`3.14`), `bool`, `char`, `String` (`"${name}"`)
**Time:** `time` (μs epoch), `duration` (`5_s`, `7_hour`), `Date` (UI, needs timezone)
**Geo:** `geo{lat, lng}` | Shapes: `GeoBox`, `GeoCircle`, `GeoPoly` (`.contains(geo)`)

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

**Transactions**: Atomic per function, rollback on error. **Production patterns, multi-index ownership, transaction safety** → [references/nodes.md](references/nodes.md)

## Indexed Collections

| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `node<T>` | — | `Array<T>`, `Map<K,V>` |
| `nodeList<node<T>>` | int | `Stack<T>`, `Queue<T>` |
| `nodeIndex<K, node<V>>` | hash | `Set<T>`, `Tuple<A,B>` |
| `nodeTime<node<T>>` | time | `Buffer`, `Table`, `Tensor` |
| `nodeGeo<node<T>>` | geo | `TimeWindow`, `SlidingWindow` |

```gcl
var temps = nodeTime<float>{}; temps.setAt(t1, 20.5); for (t: time, v: float in temps[from..to]) { }
var idx = nodeIndex<String, node<X>>{}; idx.set("key", val); idx.get("key");  // ⚠️ uses set/get, NOT add
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

**Detailed patterns, CRUD examples, inheritance, polymorphism** → [references/patterns.md](references/patterns.md)

## Logging & Error Handling

```gcl
info("msg ${var}"); warn("msg"); error("msg"); try { op(); } catch (ex) { error("${ex}"); }
```

## Parallelization

```gcl
var jobs = Array<Job<ResultType>> {};
for (item in items) { jobs.add(Job<ResultType> { function: processFn, arguments: [item] }); }
await(jobs, MergeStrategy::last_wins); for (job in jobs) { results.add(job.result()); }
```

**Key**: `Job<T>` generic, `MergeStrategy::last_wins`, no nested await. **Worker pools, PeriodicTask, async HTTP, production patterns** → [references/concurrency.md](references/concurrency.md)

## Testing

Run `greycat test`. Test files: `*_test.gcl` in `./backend/test/`.

```gcl
@test fn test_city_creation() { var city = City::create("Paris", country_node); Assert::equals(city->name, "Paris"); }
fn setup() { /* runs before tests */ } fn teardown() { /* cleanup after tests */ }
```

**Assert**: `equals(a, b)`, `equalsd(a, b, epsilon)`, `isTrue(v)`, `isFalse(v)`, `isNull(v)`, `isNotNull(v)`. **Test organization, mocking, fixtures, CI/CD** → [references/testing.md](references/testing.md)

## Common Pitfalls

**⚠️ Reserved Keywords**: `limit`, `node`, `type`, `var`, `fn` are reserved. Do NOT use as variable names or attribute names:
```gcl
// ❌ WRONG - using reserved keywords
type User { limit: int; type: String; }  // 'limit' and 'type' are reserved!
fn process(node: String) { }             // 'node' is reserved!

// ✅ CORRECT - use different names
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

**DEV**: Delete deprecated fields. Reset `gcdata/` on schema changes. Add non-nullable → make nullable: `newField: int?`
```bash
rm -rf gcdata && greycat run import  # ⚠️ DELETES DATA - ask confirmation
```

**CLI**: [references/cli.md](references/cli.md) | **Docs**: https://doc.greycat.io/

## Full-Stack Development

**[references/frontend.md](references/frontend.md)** - Comprehensive React+GreyCat guide (1,013 lines): @greycat/web SDK, TypeScript codegen, auth, React Query, error handling.

## Local LLM Integration

**[references/ai/llm.md](references/ai/llm.md)** - llama.cpp integration: model loading, text gen, chat, embeddings, LoRA.
```gcl
@library("ai", "7.6.10-dev");
var model = Model::load("llama", "./model.gguf", ModelParams { n_gpu_layers: -1 });
var result = model.chat([ChatMessage { role: "user", content: "Hello!" }], null, null);
```

## Library References

**[references/LIBRARIES.md](references/LIBRARIES.md)** - Complete catalog: std, ai, algebra, kafka, sql, s3, opcua, finance, powerflow, useragent.
