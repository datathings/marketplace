---
name: init
description: Initialize CLAUDE.md with generic GreyCat development guidelines
allowed-tools: Write, Read, Bash
---

# Initialize GreyCat Project Documentation

**Purpose**: Generate CLAUDE.md with generic GreyCat best practices.
**Run When**: New GreyCat project, or setting up Claude Code integration.

---

## Steps

1. **Check CLAUDE.md exists** — if yes, ask to backup or cancel.
2. **Detect features**:
   - Frontend: `frontend/` exists, or `package.json` has `vite-plus` / `@greycat/web` / `lit` / `@shoelace-style/shoelace`.
   - GreyCat version: parse `@library("std", ...)` from `project.gcl`.
3. **Check `.gitignore`** — create or append GreyCat entries if missing.
4. **Write CLAUDE.md** from the template below: replace `[One-line project description]` / `[version]`; drop the frontend sections (commands, stack, structure, coding style, testing, env) if there is no frontend; keep all backend sections.
5. **Report**: created files + next steps (replace placeholders, commit).

---

## CLAUDE.md Template

```markdown
# CLAUDE.md

[One-line project description]

## Behavioral Guidelines

Bias toward caution over speed. Use judgment for trivial edits; for `.gcl`, persistence, or `@expose` endpoints follow these strictly.

**1. Think Before Coding** — State assumptions explicitly. If multiple interpretations exist (`nodeIndex` vs `Map`, persisted vs `@volatile`), present them. Read `lib/std/*.gcl` for real examples — don't guess by analogy to Java/TypeScript.

**2. Simplicity First** — Minimum code that solves the problem. No speculative features, single-use abstractions, `@volatile` "for later", `try/catch` on internal helpers, or unused generics.

**3. Surgical Changes** — Touch only what's needed; match existing style (snake_case fields, camelCase functions). `Grep` before deleting any type/function/field; remove orphans your changes create. Regenerate (never hand-edit) `project.d.ts`.

**4. Goal-Driven Execution** — Define verifiable success: lint clean, `greycat test` green, endpoint returns expected shape. For multi-step tasks, state a brief plan with per-step verification.

---

## ⚠️ CRITICAL RULES

### 1. ALWAYS LINT AFTER EACH CHANGE
\`\`\`bash
greycat-lang lint  # Verify 0 errors after ANY change
\`\`\`

### 2. VERIFY BEFORE DELETING
`Grep` for usages before removing any type/function/variable.

### 3. GOTCHAS
Don't write GCL by analogy to another language — check the **Common Pitfalls** table below first: String `slice`/`split(' ')`/`.chars()`, global `log(x)`/`sqrt(x)`, no `static fn <T>`, `Array<T> {}` (not `::new()`), `nodeIndex.set` (not `.add`), named `@volatile` returns, `abstract` from day 1, `T?` on persisted types, reserved words (`limit`/`skip`), generic-return erasure.

### 4. USE GREYCAT SKILL & REVIEW COMMANDS
**Mandatory** for GCL backend work: use the `/greycat` skill. Before releases / after refactors, run the two review hubs:
- **`/greycat:backend`** — dead code, anti-patterns, type safety, performance & concurrency, `@expose` API security, test coverage.
- **`/greycat:frontend`** — the VitePlus + Lit + Shoelace + `@greycat/web` stack, performance, Lighthouse/SEO, type safety, testing (only if a `frontend/` exists).

---

## Commands

\`\`\`bash
# Backend
greycat-lang lint              # Lint (after EVERY change!)
greycat-lang fmt               # Format (default write mode; --mode=check for a CI gate)
greycat build/test/serve       # Build/test/serve (port 8080)
greycat run [function]         # Run function (default: main)
greycat codegen ts             # Generate project.d.ts
greycat install                # Install libraries

# Frontend (if exists) — VitePlus + Lit + Shoelace + lucide-static
vp install                     # install frontend deps (rolldown-based)
vp build                       # build frontend/ -> webroot/ (add --watch for dev)
greycat dev                    # build watcher + serve API and assets on one origin (:8080)
greycat codegen ts             # regenerate project.d.ts after backend type/@expose changes
\`\`\`

> `vp` and pnpm are different layers, not alternatives: **pnpm** is the package manager (fetches deps); **`vp`** (VitePlus, rolldown/oxc) is the build toolchain that bundles `frontend/` → `webroot/`, and `vp install` delegates to pnpm. Install `vp` once: `curl -fsSL https://vite.plus | bash`. Lighthouse: run the CLI against the served app (`greycat dev` first); add a script only if the project wants a CI gate.

---

## Stack

- **Backend**: GreyCat [version] (GCL)
- **Frontend** (if exists) — the one prescribed stack:
  - **VitePlus** (`vp` CLI + `vite-plus`) — toolchain; explicit `vite.config.ts`, no plugin. **MPA**: each route is a real page under `frontend/routes/`, URL == file path, no SPA router
  - **Lit** in **light DOM** — one root `LitElement` per route, a component only for views reused across routes; `@customElement('app-…')`
  - **TypeScript** (`experimentalDecorators: true`, `useDefineForClassFields: false`, `moduleResolution: "bundler"`)
  - **Shoelace** (`@shoelace-style/shoelace`, `sl-*`) — atomic controls (button, input, dialog, tabs, date-picker)
  - **`@greycat/web`** (`gui-*`) — rich/GreyCat-aware widgets (tables, charts, maps, `gui-object` forms, sign-in) + the typed SDK for every backend call
  - **lucide-static** icons — prebuilt SVG strings inlined via Lit `unsafeSVG`; self-hosted, `currentColor` + `aria-hidden`, no CDN fetch
  - **Theme**: `@greycat/web/greycat.css` (a Shoelace theme, dark by default) + `frontend/theme.css` (the `--app-*` tokens + `--sl-*` re-skin) imported **after** it
  - **pnpm** as the package manager
- **Libraries**: `@library("std", "[version]")`, `@library("explorer", "[version]")`
- **Testing**: backend `@test`

**Frontend Setup**: Configs in root (package.json, vite.config.ts, tsconfig.json); source entirely under `frontend/` (`routes/` pages, `components/` reused views, `public/` copied as-is, `theme.css`); builds to `webroot/` (`emptyOutDir: true`). `~` aliases `frontend/` in both `vite.config.ts` and `tsconfig.json`. `@greycat/web` is **not on npm** — pin it as a registry tarball URL (`https://get.greycat.io/files/sdk/web/<branch>/<version>.tgz`) tracking the project's `std` branch; Shoelace/Lit are ordinary semver deps (Shoelace must satisfy `@greycat/web`'s peer range). Optimize with Lighthouse (perf/SEO/a11y ≥ 90); ship LLM-friendly SEO (see Frontend coding style).

---

## Project Structure

\`\`\`
.
├── project.gcl                # Entry point, libraries, permissions
├── src/<feature>/
│   ├── <feature>.gcl          # Data models + global indices + service
│   ├── <feature>_api.gcl      # @expose + @volatile views
│   ├── <feature>_reader.gcl   # CSV/JSON readers (optional)
│   └── <feature>_writer.gcl   # Writers (optional)
├── test/<feature>_test.gcl
├── frontend/                  # entire frontend (VitePlus + Lit + Shoelace, if exists)
│   ├── routes/                # MPA pages — path == URL (index.html + index.ts per route)
│   │   └── index.html         #   SEO head (title, description, OG, JSON-LD, lang); one app-* root
│   ├── components/            # reusable light-DOM Lit views (imported as ~/components/x)
│   ├── public/                # copied as-is into webroot/ (robots.txt, icons, llms.txt)
│   ├── icons.ts               # lucide-static SVG strings, inlined via unsafeSVG (no runtime fetch)
│   └── theme.css              # --app-* design tokens + Shoelace/GreyCat --sl-* re-skin
├── package.json / vite.config.ts / tsconfig.json   # Root level
├── project.d.ts               # generated by `greycat codegen ts` (ambient gc.* types, gitignored)
├── webroot/                   # Built frontend + greycat explorer (gitignored)
│   ├── robots.txt sitemap.xml site.webmanifest      # SEO discoverability
│   └── llms.txt               # LLM-friendly Markdown site index
├── lib/                       # Installed libs (gitignored)
├── gcdata/                    # DB storage (gitignored)
└── CLAUDE.md
\`\`\`

**⚠️ Filename = module name.** Two `.gcl` files with the same basename collide at lint, regardless of folder. Convention: `src/orders/orders.gcl` (model + service), everything else gets a suffix (`orders_api.gcl`, `orders_reader.gcl`, …). On `Unknown type T` lint errors, disambiguate with FQN: `mod::T`.

---

## .gitignore Essentials

\`\`\`gitignore
/gcdata /lib /webroot
project.d.ts project.gcp /project_types
/node_modules /.pnp .pnp.js
/dist /build /coverage /outputs /generated
.env .env.* /.parcel-cache
.DS_Store Thumbs.db
.vscode/ .idea/ *.swp *.swo
__pycache__/ *.pyc *.pyo
npm-debug.log* yarn-*.log* pnpm-debug.log*
.playwright-mcp/
\`\`\`

---

## Coding Style

### Backend (GCL)

- **Documentation (REQUIRED)**: `///` on ALL functions/types with `@param`/`@return`/`@throws`/`@example`; section headers via `// ===`.
- **Services**: abstract types + static methods; `@volatile` for transient types; null-safe with `?`.
- **Error Handling (MANDATORY)**: try/catch on ALL `@expose`; typed errors only (no `throw "string"`); `error()` log + re-throw.
- **Invariants via `private`**: read-public, write-private — mutations only through methods on the type.
- **Collections**: `Array<T> {}`, `Map<K,V> {}`, `nodeIndex<K, node<V>>`. Initialize non-nullable collection fields in the object literal, or declare them nullable.
- **Naming**: snake_case fields, camelCase functions, `…View` suffix for `@volatile` API response types.

\`\`\`gcl
// Typed error hierarchy — src/errors.gcl
@volatile abstract type AppError { code: String; message: String; }
@volatile type NotFoundError   extends AppError { id: String; }
@volatile type ValidationError extends AppError { field: String; }
@volatile type AuthError       extends AppError {}

@expose @permission("public")
fn document(id: String): Document {
  try {
    var doc = documents_by_id.get(id);
    if (doc == null) {
      throw NotFoundError { code: "NOT_FOUND", message: "...", id: id };
    }
    return doc.resolve();
  } catch (ex) {
    error("document(${id}) failed: ${ex}");
    throw ex;
  }
}
\`\`\`

### Frontend (VitePlus + Lit + Shoelace + lucide-static) — if exists

Stack as listed above (Lit light-DOM + Shoelace `sl-*` + `@greycat/web` `gui-*` + typed SDK on VitePlus, MPA under `frontend/routes/`, pnpm). `@greycat/web`'s own widgets are Lit, so the whole UI is web-components.

**⚠️ Init/login gate**: `gc.sdk.init()` loads the ABI over an authenticated endpoint — call it (no args) in the route root's `connectedCallback`; on throw, render a login form and call `gc.sdk.init({ auth: { username, password } })`. Only after it resolves are `gc.<module>.*` calls and `gui-*` tags usable.
\`\`\`ts
// frontend/routes/index.ts
import '@greycat/web/sdk';            // gc global, init, typed bindings — import gui-* components individually, not umbrella '@greycat/web'
import '@greycat/web/greycat.css';    // the theme (dark by default)
import '~/theme.css';                 // app --app-* tokens + --sl-* re-skin — AFTER greycat.css
await gc.sdk.init();                  // then gc.project.* and gui-* are live
\`\`\`

- **Codegen discipline**: re-run `greycat codegen ts` after every backend type/`@expose` change. Never hand-edit `project.d.ts`; a stale ABI gets HTTP 422. Derive backend strings from the SDK (`gc.project.Status.active`, `.key`) — never hard-code.
- **Components (Lit, light DOM)**: one `LitElement` per file, `@customElement('app-…')` kebab prefix. **`createRenderRoot() { return this; }`** so `theme.css` cascades in — no Shadow DOM in app views. `@property()` for public inputs, `@state()` for internal state, `html\`…\`` templates. Charts (`gui-*` or chart.js): create after init, **destroy in `disconnectedCallback`**.
- **Shoelace**: import components **individually** (tree-shaking). Do **not** import Shoelace's own `themes/*.css` — `greycat.css` is the theme; light mode is the `sl-theme-light` class on `<html>`.
- **Icons**: **`lucide-static`** (inline SVG via Lit `unsafeSVG`), `stroke="currentColor"` + `aria-hidden` — self-hosted, no runtime/CDN fetch.
- **Services**: thin layer over the generated SDK client; types from `project.d.ts`.
- **State**: URL query params (shareable view state), localStorage (theme). **Styling**: `--app-*` tokens in `frontend/theme.css` (dark + light); no hardcoded colors/sizes, no inline styles except dynamic values.
- **Naming**: camelCase (vars/fns), PascalCase (TS types/classes), `app-` kebab (custom elements).
- **Lighthouse (perf/SEO/a11y/best-practices ≥ 90 on BOTH mobile and desktop)**: `greycat dev` to serve, then run Lighthouse per form factor — default is mobile (throttled, the harder gate), add `--preset=desktop`. Tree-shake Shoelace, self-host icons (`lucide-static`), lazy-load heavy widgets, defer non-critical JS, inline critical CSS, long-cache hashed assets, reserve sizes to avoid layout shift.
- **Responsive (always)**: `<meta name="viewport" content="width=device-width, initial-scale=1">`; fluid layout (grids / `clamp()` / media queries, no fixed-px page widths); tap targets ≥ 24–48px; no horizontal scroll at 360px.
- **LLM-friendly SEO (always)**: in each route's `index.html` — `<html lang>`, unique `<title>`, `<meta name="description">`, canonical link, Open Graph + Twitter Card, `theme-color`, JSON-LD (`schema.org`). Semantic landmarks + heading order + `alt`/ARIA (light DOM keeps content crawlable). Ship `robots.txt`, `sitemap.xml`, a web app manifest, and **`llms.txt`** (+ optional `llms-full.txt`) via `frontend/public/` — a concise Markdown index of purpose, key routes, and public endpoints for LLM agents.

### Testing

**Backend**: `@test` annotation, `test_function_scenario` naming, `Assert::`.
\`\`\`gcl
@test fn test_search_validQuery() {
  var results = SearchService::search("test");
  Assert::notNull(results);
}
// Optional lifecycle: fn setup() / fn teardown() (detected by name, no annotation)
\`\`\`

**Runner semantics**:
- Tests in the same module share state across `@test` fns (mutations visible to the next; NOT persisted to disk).
- `greycat test` runs the whole project in one process — a SEGFAULT/compile error poisons all tests.
- **Wipe `gcdata/` for clean runs**: `rm -rf gcdata && greycat test`.
- Single test: `greycat test <test_fn_name>` (a bare `@test` function name, e.g. `greycat test test_echo`; omit to run all). It is NOT a file path or directory.
- Cross-module helpers in `test/test_helpers.gcl` must be plain `fn` (not `private`).

**Exit codes**: `0`=success · `1`=generic CLI error (missing file / bad option) · `2`=compile/load error (all tests affected). A crash/timeout (segfault, killed) obviously invalidates the run.

**Test isolation patterns**: `EmailService::disable()` kill switches, redirect writes via `Uuid::v4()` scratch paths, `skipHeavyImportersForTests()` flag, `cleanTearDir()` teardown.

**Frontend**: no test framework is prescribed — rely on `greycat-lang`/`tsc` type-checking and the Lighthouse audit. Add a runner (e.g. Vitest) only if the project needs one.

---

## GreyCat Language Patterns

### Nullability
\`\`\`gcl
var city: City?;                       // nullable
city?.name?.size();                    // optional chaining
city?.name ?? "Unknown";               // nullish coalescing
(answer as String?) ?? "default"       // ⚠️ parens for cast + coalescing
if (country == null) { return null; }
return country->name;                  // ✅ no !! after null check
\`\`\`
**⚠️ NO TERNARY** — use if/else.

### Nodes (Persistence)
\`\`\`gcl
type Country { name: String; code: int; }
var n = node<Country>{ Country { name: "LU", code: 352 }};
n->name;              // arrow: deref + field (read/write)
n.resolve();          // payload (or null)
*n;                   // unary deref
n.set(v);             // rebind primitive node (node<int>{0}.set(5))
\`\`\`

**Node ownership** — an object belongs to exactly ONE node. To index by multiple keys, store the SAME `node<T>` ref:
\`\`\`gcl
var item = node<Item>{ Item { id: 1, name: "x" } };
by_id.set(1, item);     // both indices point at
by_name.set("x", item); //   the same persistent node
\`\`\`

**String dedup**: use `node<String>` instead of `String` when the same value repeats across many objects.

**Lazy init** for nullable collection attrs: `this.entries ?= nodeIndex<String, node<V>>{};`

**Polymorphism** works through node-wrapped values + abstract methods:
\`\`\`gcl
abstract type Animal { abstract fn speak(): String; }
type Dog extends Animal { fn speak(): String { return "woof"; } }
animals.set("d", node<Animal>{ Dog {} });
animals.get("d")?->speak();             // dispatches to Dog
\`\`\`
Concrete methods on `abstract type` CANNOT be overridden — declare `abstract` from day 1 if subtypes need different behavior.

### Indexed Collections
| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `nodeList<node<T>>` | int | `Array<T>` |
| `nodeIndex<K, node<V>>` | K | `Map<K,V>` |
| `nodeTime<node<T>>` | time | — |
| `nodeGeo<node<T>>` | geo | — |

**⚠️ CRITICAL**: Initialize non-nullable collection FIELDS on creation (`City { streets: nodeList<node<Street>>{} }`) or declare them nullable. Module-level `node*` collections auto-initialize; type fields do not.

---

## Concurrency & Tasks

**`await` fans out inside a task context.** An `@expose` HTTP call is already enqueued as a task, so `await(jobs)` fans out over an HTTP POST. The serial case is a one-shot `greycat run` script. To dispatch a long HTTP call as a *background* task (return `task_id` immediately, poll later), set the `task: true` header:
\`\`\`bash
curl -H "task: true" -X POST -d '[]' http://localhost:8080/module::compute  # background task, returns task_id
./bin/greycat run compute                                                    # one-shot CLI run (jobs run serially)
\`\`\`
**Don't** "fix" non-parallel HTTP via `System::exec` + `&; wait` — the second `System::exec` in a non-task HTTP request throws uncatchable `"terminated PID X"`.

**Request TTL kills + try/catch does NOT fire** — the request TTL (`--request_ttl` flag, `serve` only) defaults to 20s. Past that the runtime tears down the handler with no exception. For long endpoints, raise `--request_ttl` or move to a background task.

**Fork-join**:
\`\`\`gcl
// Prefer `Array<Job>` over `Array<Job<T>>`; cast each `.result()` at the call site.
var jobs = Array<Job> {
  Job { function: jobs::compute, arguments: [100] },     // use real <module>::<fn>
};
await(jobs, MergeStrategy::strict);                       // 2nd arg required
var first = jobs[0].result() as int;                      // cast at collection time
\`\`\`

**"Stale await state" error** — if a call-frame variable holds a node-backed value (resolved node, time, Array of payloads) when `await` runs, you get `"wrong state before await..."`. Fix: isolate `await` in its own helper and build the result Array AFTER it returns.

**Batch discipline**:
- Batch `await` jobs in **~120**, never full worker count. Leave ≥8 threads for the OS. No nested `await`.
- Pre-allocate shards in a sequential phase. Inside parallel jobs: only WRITE to existing nodes — no `node<T>{...}` constructors, no `nodeIndex` instantiation, no edges to shared parents. Global indices are read-only during parallel phases.

**Recurring tasks**:
\`\`\`gcl
Scheduler::add(
  jobs::nightly_job,                                                // <module>::<fn>
  DailyPeriodicity { hour: 2, minute: 0, second: 0, timezone: TimeZone::UTC },
  PeriodicOptions { start: time::now(), max_duration: 1_hour }
);
\`\`\`

**Task lifecycle**: `empty → waiting → running → await → ended | error | cancelled | ended_with_errors`.
- Status: `Task::is_running(task_id)` / `Task::running()` / `Task::history(offset, max)` (there is no `Task::info` RPC).
- Result: `GET /files/<user_id>/tasks/<task_id>/result.gcb?json`.

**Atomicity**: each `fn` invocation is one atomic transaction — an uncaught throw rolls back all graph mutations.

---

## API Routing (@expose)

| Route shape | Reachable at | Notes |
|-------------|--------------|-------|
| `POST /<module>::<fn>` | free-standing `@expose` fn | module = source file basename; body = JSON array of positional args |
| `POST /<module>::<Type>::<fn>` | `@expose` static fn on a type | three segments (full FQN), e.g. `/runtime::Identity::current_id` |
| `POST /` (JSON-RPC) | any `@expose` fn | `{ "method": "<module>.<fn>", "params": [...] }` — dots, not `::` |

**Rules**:
- Call the explicit `/<module>::<fn>` (or the JSON-RPC method) from clients — there is no bare `/<fn>` route.
- `/runtime::Identity::logout` works because `runtime` is a module name — there is NO literal `/runtime::` routing namespace.
- `@expose("path")` exposes the function at that exact arbitrary path (e.g. `@expose("api/users")`); it need NOT match the fn name. Whatever path you declare is what clients and the generated SDK must call.
- `@permission(public)` is invalid (identifier); use `@permission("public")` (quoted). A bare `@expose` already requires `api` (authenticated) — the recommended default.
- Declare roles in `project.gcl`: `@role("user", "public", "api");`.

---

## Stdlib Recipes

**Logging**: `info()/warn()/error()` for structured logs (CSV under non-TTY). `println()` for stdout. `pprint(obj)` to dump typed objects.

**Sort**: field-reference (no comparator lambdas):
\`\`\`gcl
entries.sort_by(LeaderboardEntry::balance, SortOrder::desc);
\`\`\`

**Downsampling** (modes: fixed, fixed_reg, adaptative, dense):
\`\`\`gcl
var pts: Table<any?> = nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null);
// Returns a Table; pivot to typed (time, float) pairs yourself if needed.
\`\`\`

**Typed CSV**:
\`\`\`gcl
// Use the Csv:: module (CsvFormat is non-generic). See reference/stdlib.md for full surface.
var fmt = CsvFormat {};                                  // configure separator, quote, header, etc.
var stats = Csv::analyze(["./data.csv"], fmt);           // schema inference
var table = Csv::sample(["./data.csv"], fmt, 1000);      // first 1000 rows into a Table
\`\`\`

**Geo**:
\`\`\`gcl
var box = GeoBox { sw: geo{49.0, 5.9}, ne: geo{50.2, 6.5} };
if (box.contains(geo{49.6, 6.1})) { /* ... */ }
// Iterate then test (nodeGeo has no `.filter()` method):
for (pos: geo, s in sensors) { if (box.contains(pos)) { /* ... */ } }
\`\`\`

**Iterate nullable collection**: `for (i, v in maybe_list?) { /* ... */ }` (`?` propagates null through the head).

---

## Common Pitfalls

| ❌ Don't | ✅ Do |
|---------|-------|
| `String.substring()` | `String.slice(from, to)` |
| `String.split(" ")` | `String.split(' ')` (char) |
| `String.get(i)` | `.chars()` (NFKD+casefold) or Buffer |
| `x.log()` / `x.sqrt()` | `log(x)` / `sqrt(x)` (globals) |
| Delete without Grep | Grep first |
| Skip linting | Lint after EACH change |
| `static fn process<T>` | Remove static OR generics |
| `Array<T>::new()` | `Array<T> {}` |
| `nodeList<T>` for local | `Array<T>` for non-persistent |
| Missing @volatile on DTO | Always add @volatile |
| Uninitialized non-nullable collection field | Initialize `{}` or make nullable |
| `nodeIndex.add(k, v)` | `nodeIndex.set(k, v)` |
| `throw "string"` | `throw TypedError { code, message, id }` |
| @expose without try/catch | Always wrap + `error(...)` |
| Bare-name route | `/<module>::<fn>` |
| `@permission(public)` ident | `@permission("public")` quoted |
| Assuming `@expose("path")` = default route | Custom path overrides `<module>::<fn>` — call the declared path |
| Bare `id`/`ids` params | `documentId`/`partyIds` |
| Anonymous return structs | Named `@volatile` type |
| Override concrete on abstract | Make abstract from day 1 |
| `!!` to silence analyzer | Only on init-guaranteed registry lookups |
| Fresh `node<T>` on re-import | Look up prior, upsert |
| Same object in two indices | Same `node<T>` ref in both |
| Non-nullable field on persisted type | `T?` until backfilled |
| `time::new(n, seconds)` on raw epoch | Magnitude-route first |
| `x as T? ?? "default"` | `(x as T?) ?? "default"` |
| `Array<Job<T>>` | `Array<Job>` + cast `.result() as T` |
| `Array<Wrapper<T>>` generic return (erased at runtime) | Non-generic type + `value: any` + cast at call site |
| `var limit: int` (reserved word) | rename (e.g. `var k`) — `limit` / `skip` are reserved (sampling clause) |
| Background dispatch header `task:''` | `task: true` (truthy) header |
| Catching the 20s TTL kill | You can't — raise `--request_ttl` or move to task |
| Functions without `///` docs | Document ALL functions |

---

## Database

**No migrations.** Adding a non-nullable field **without a default** to a persisted type with existing data **fails to load** — add it as nullable (`T?`) until backfilled, or give it a default (a non-null field WITH a default auto-migrates). Removing a field auto-migrates (dropped on next save); **renaming** requires a reset (looks like remove + add). Dev reset: `rm -rf gcdata && greycat run import`.

**Production safe-rollback deploy**:
\`\`\`bash
systemctl stop greycat
mv gcdata gcdata_bk           # rotate, don't delete
greycat run import            # replay importers
./bin/greycat defrag          # reclaim storage
systemctl start greycat
# rollback: rm -rf gcdata && mv gcdata_bk gcdata && systemctl restart greycat
\`\`\`

**Upsert, never duplicate** on re-import. An importer that wipes a `nodeIndex`/`nodeTime`/`nodeGeo`/`nodeList` MUST reuse the prior `node<T>` per key:
\`\`\`gcl
var prev = items;
items = nodeIndex<String, node<Item>>{};
for (i, row in rows) {                                  // Array iteration requires 2 params
  var existing = prev.get(row.id);
  if (existing != null) { existing->name = row.name; items.set(row.id, existing); }
  else { items.set(row.id, node<Item>{ Item { id: row.id, name: row.name } }); }
}
\`\`\`

**Time ingest**: never `time::new(n, seconds)` blindly — feeds mix units. Magnitude-route:
\`\`\`gcl
fn epochToTime(n: int): time {
  if (n < 100_000_000_000) return time::new(n, DurationUnit::seconds);        // <1e11
  else if (n < 100_000_000_000_000) return time::new(n, DurationUnit::milliseconds);  // <1e14
  else return time::new(n, DurationUnit::microseconds);
}
// Note: DurationUnit has no `nanoseconds` — caller must downscale.
\`\`\`

---

## Production & Operations

- **Logs** — `info/warn/error` are **silently swallowed under non-TTY** (systemd, docker without `-t`). The backend writes every log line to **`files/root/log.csv`**: `tail -f files/root/log.csv`.
- **Security & rate limiting** — `std::Scheduler` has no throttle primitive; `@permission("public")` auth endpoints have NO in-process rate limiting. Mitigate externally: bind to loopback, front with nginx `limit_req_zone` keyed on `$binary_remote_addr`.
- **systemd** — use **project-local `./bin/greycat`** in `ExecStart` (host-wide `~/.greycat/bin/greycat` has been observed stale at `0.0.0`). Dev builds report `0.0.0` from `--version` — treat as "latest".

---

## Environment

\`\`\`bash
# Backend (.env)
GREYCAT_PORT=8080
GREYCAT_WEBROOT=webroot
GREYCAT_CACHE=30000
\`\`\`

**vite.config.ts** (in root) — explicit VitePlus config, **no plugin**. `greycat dev` serves API + assets on one origin, so nothing to proxy. MPA: `root: 'frontend/routes'` makes each page's path its URL; one `rollupOptions.input` entry per route:
\`\`\`ts
import { defineConfig } from 'vite-plus';
import { resolve } from 'node:path';

export default defineConfig({
  root: 'frontend/routes',              // frontend/routes/**/index.html -> a page at that URL
  base: './',                           // relative asset URLs so nested pages resolve under webroot/
  appType: 'mpa',                       // no SPA history fallback
  publicDir: resolve('frontend/public'),
  resolve: { alias: { '~': resolve('frontend') } },
  build: {
    outDir: resolve('webroot'),
    emptyOutDir: true,
    target: 'esnext',
    rollupOptions: { input: [resolve('frontend/routes/index.html')] },  // add one per route
  },
});
\`\`\`

**tsconfig.json** (in root): Lit needs `"experimentalDecorators": true`, `"useDefineForClassFields": false`, `"moduleResolution": "bundler"`, `"paths": { "~/*": ["frontend/*"] }`, and `"include": ["frontend", "project.d.ts"]`.

---

## Auth & Permissions

**project.gcl**:
\`\`\`gcl
@permission("app.admin", "admin permission");
@permission("app.user",  "user permission");
@role("admin", "app.admin", "app.user", "public", "admin", "api");
@role("user",  "app.user", "public", "api");
\`\`\`

**Usage**: `@permission("app.user")`, `SecurityService::getLoggedUser()`.

---

## Development Workflow

1. Use the `/greycat` skill for backend work.
2. `greycat-lang lint` after EVERY change (0 errors required); `greycat-lang fmt` to format.
3. `Grep` before deleting; `greycat codegen ts` after backend type changes.
4. Test: `greycat test` (backend); `vp build` + Lighthouse for frontend.
5. Before releases: `/greycat:backend` (full backend review) and, if a `frontend/` exists, `/greycat:frontend`.

---

## Consistency Checklist

- [ ] `greycat-lang lint` shows 0 errors; `greycat-lang fmt` applied
- [ ] All `@expose` have try/catch + `error()` log; all thrown errors are typed
- [ ] All functions/types have `///` docs
- [ ] Transient/API types are `@volatile` (`…View` suffix)
- [ ] Non-nullable collection fields initialized in literal
- [ ] No `static fn <T>` (generic statics)
- [ ] `@expose` routes use explicit `/<module>::<fn>` from clients
- [ ] `greycat codegen ts` re-run after backend type changes
- [ ] Re-import path is upsert
- [ ] New fields on persisted types are nullable OR `gcdata/` reset planned
- [ ] Frontend on VitePlus (`vite-plus`) — Lit (light DOM) + Shoelace + `@greycat/web`; `@greycat/web` pinned as a registry tarball tracking `std`'s branch (managed with pnpm)
- [ ] Route root gates on `gc.sdk.init()` before any `gc.<module>.*` / `gui-*`; `import '@greycat/web/sdk'` (not umbrella `@greycat/web`)
- [ ] `greycat.css` imported, then `frontend/theme.css` after it — Shoelace's own `themes/*.css` NOT imported
- [ ] App components render to light DOM (`createRenderRoot(){return this}`); no hardcoded colors/sizes (use `--app-*` tokens)
- [ ] Shoelace imported per-component; icons self-hosted via lucide-static (no CDN fetch)
- [ ] Lighthouse ≥ 90 perf/SEO/a11y/best-practices on BOTH mobile (default) and desktop (`--preset=desktop`)
- [ ] Responsive: `viewport` meta, fluid layout, no horizontal scroll at 360px, adequate tap targets
- [ ] SEO head present (title, meta description, canonical, OG/Twitter, JSON-LD, `lang`)
- [ ] `robots.txt`, `sitemap.xml`, web manifest, and `llms.txt` shipped to `webroot/`
- [ ] Tests pass (`rm -rf gcdata && greycat test`)

---

## LSP

`greycat-lang server --stdio` — autocomplete, hover, go-to-def, diagnostics, format, rename. Always run `greycat-lang lint` before commit.

More: https://doc.greycat.io/
```
