---
name: init
description: Initialize CLAUDE.md with generic GreyCat development guidelines
allowed-tools: Write, Read, Bash
---

# Initialize GreyCat Project Documentation

**Purpose**: Generate CLAUDE.md with generic GreyCat best practices.

**Run When**: New GreyCat project, setting up Claude Code integration.

---

## Steps

1. **Check CLAUDE.md exists** — if yes, ask to backup or cancel
2. **Detect features**:
   - Frontend: `app/` exists or `package.json` has `@greycat/web`
   - GreyCat version: parse `@library("std", ...)` from `project.gcl`
3. **Check `.gitignore`** — create or append GreyCat entries if missing
4. **Write CLAUDE.md** from the template below, customizing:
   - Replace `[One-line project description]`, `[version]`
   - Remove frontend sections if no frontend (commands, stack, structure, coding style, testing, env, `gcRuntime` checklist line)
   - Keep all backend sections (always present)
5. **Report**: list created files + next steps (replace placeholders, commit)

---

## CLAUDE.md Template

```markdown
# CLAUDE.md

[One-line project description]

## Behavioral Guidelines

Bias toward caution over speed. For trivial edits use judgment; for `.gcl`, persistence, or `@expose` endpoints follow these strictly.

**1. Think Before Coding** — State assumptions explicitly. If multiple interpretations exist (`nodeIndex` vs `Map`, persisted vs `@volatile`), present them. Read `lib/std/*.gcl` for real examples — don't guess by analogy to Java/TypeScript.

**2. Simplicity First** — Minimum code that solves the problem. No speculative features, no abstractions for single-use code, no `@volatile` "for later", no `try/catch` on internal helpers, no unused generics.

**3. Surgical Changes** — Touch only what's needed. Match existing style (snake_case fields, camelCase functions). `Grep` before deleting any type/function/field. Remove orphans your changes created. Regenerate (don't hand-edit) `project.d.ts`.

**4. Goal-Driven Execution** — Define verifiable success: lint clean, `greycat test` green, endpoint returns expected shape. For multi-step tasks, state a brief plan with verification per step.

---

## ⚠️ CRITICAL RULES

### 1. ALWAYS LINT AFTER EACH CHANGE
\`\`\`bash
greycat-lang lint  # Verify 0 errors after ANY change
\`\`\`

### 2. VERIFY BEFORE DELETING
Use `Grep` to search for usages before removing types/functions/variables.

### 3. GREYCAT GOTCHAS
\`\`\`gcl
// String: ❌ substring()      ✅ slice(from, to)
// String: ❌ split(" ")       ✅ split(' ')          // char, not String
// String: ❌ get(i)           ✅ .chars() or Buffer
// Math:   ❌ x.log()/x.sqrt() ✅ log(x)/sqrt(x)      // globals
// Static: ❌ static fn f<T>() ✅ remove static OR generics
// Coll:   ❌ Array<T>::new()  ✅ Array<T> {}
// Coll:   ❌ nodeList<T> local var  ✅ Array<T> for non-persistent
// Coll:   ❌ nodeIndex.add()  ✅ nodeIndex.set(k,v) / .get(k)
// Type:   ❌ fn(): {x:int}    ✅ named @volatile type
// OOP:    ❌ override concrete on abstract  ✅ make abstract from day 1
// Data:   ❌ add non-nullable to persisted  ✅ add as T? — or reset gcdata/
// Resv:   ❌ var limit: int   ✅ var k: int           // limit is reserved
// Gen:    ❌ Array<Wrapper<T>> return       ✅ non-generic + value:any + cast
\`\`\`

### 4. USE GREYCAT SKILL
**Mandatory** for GCL backend work: use `/greycat` skill.

---

## Commands

\`\`\`bash
# Backend
greycat-lang lint              # Lint (after EVERY change!)
greycat-lang fmt -p project.gcl  # Format all
greycat build/test/serve       # Build/test/serve (port 8080)
greycat run [function]         # Run function (default: main)
greycat codegen ts             # Generate project.d.ts
greycat install                # Install libraries

# Frontend (if exists)
pnpm install/dev/build/lint/test
\`\`\`

---

## Stack

- **Backend**: GreyCat [version] (GCL)
- **Frontend** (if exists): TypeScript + Vite + @greycat/web
- **Libraries**: `@library("std", "[version]")`, `@library("explorer", "[version]")`
- **Testing**: backend `@test`, frontend Vitest

**Frontend Setup**: Configs in root (package.json, vite.config.ts, tsconfig.json), source in `app/`, builds to `webroot/`. Use exact versions (no `^`/`~`).

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
├── app/                       # Frontend source (if exists)
├── package.json / vite.config.ts / tsconfig.json   # Root level
├── webroot/                   # Built frontend (gitignored)
├── lib/                       # Installed libs (gitignored)
├── gcdata/                    # DB storage (gitignored)
└── CLAUDE.md
\`\`\`

**⚠️ Filename = module name.** Two `.gcl` files with same basename collide at lint, regardless of folder. Convention: `src/orders/orders.gcl` (model + service), everything else gets a suffix (`orders_api.gcl`, `orders_reader.gcl`, etc.). On `Unknown type T` lint errors, disambiguate with FQN: `mod::T`.

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

**Documentation (REQUIRED)**: `///` for ALL functions/types with `@param`/`@return`/`@throws`/`@example`. Section headers via `// ===`.

**Services**: Abstract types + static methods. `@volatile` for transient types. Null-safe with `?`.

**Error Handling (MANDATORY)**: try/catch on ALL `@expose`, typed errors only (no `throw "string"`), `error()` log + re-throw.

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

**Invariants via `private`**: read-public, write-private — mutations only through methods on the type.

**Collections**: `Array<T> {}`, `Map<K,V> {}`, `nodeIndex<K, node<V>>`. Initialize non-nullable collection fields in object literal, or declare them nullable.

**Naming**: snake_case fields, camelCase functions, `…View` suffix for `@volatile` API response types.

### Frontend (TypeScript) — if exists

**⚠️ `gc` namespace shadow**: `@greycat/web`'s `gc` is shadowed by `@types/node`'s `var gc`. Use re-export:
\`\`\`ts
// app/gc-runtime.ts
import * as gcRuntime from '@greycat/web';
export { gcRuntime };
// elsewhere:
import { gcRuntime } from '~/gc-runtime';
gcRuntime.project.MyType.create(...);
\`\`\`

**Codegen discipline**: re-run `greycat codegen ts` after every backend type/`@expose` change. Never hand-edit `project.d.ts`. Derive backend strings from codegen via `$fields` — never hard-code.

**Components**: Named export + memo, props interface ABOVE component, JSDoc. (Default export for pages only.)

**Hooks**: `use*` prefix, `useCallback`/`useMemo` with deps, return objects.
**Services**: Named export objects, withRetry wrapper, types from `project.d.ts`.
**State**: URL (`useSearchParams`), localStorage, Context (theme/user).
**Styling**: Theme constants + Tailwind. No inline styles except dynamic values.
**Naming**: camelCase (vars/fns), PascalCase (components/types).

### Testing

**Backend**: `@test` annotation, `test_function_scenario` naming. Use `Assert::`.
\`\`\`gcl
@test fn test_search_validQuery() {
  var results = SearchService::search("test");
  Assert::notNull(results);
}
// Optional lifecycle: fn setup() / fn teardown() (detected by name, no annotation)
\`\`\`

**Runner semantics**:
- Tests in same module share state across `@test` fns (mutations visible to next; NOT persisted to disk).
- `greycat test` runs whole project in one process — SEGFAULT/compile error poisons all.
- **Wipe `gcdata/` for clean runs**: `rm -rf gcdata && greycat test`.
- Single test: `greycat test <module>::<fn>`. Module: `greycat test <dir>`.
- Cross-module helpers in `test/test_helpers.gcl` must be plain `fn` (not `private`).

**Exit codes**: 0=pass · 2=compile error (all fail) · 5=assertion fail · 124/137=timeout · 139=segfault · 134=abort.

**Test isolation patterns**: `EmailService::disable()` kill switches, redirect writes via `Uuid::v4()` scratch paths, `skipHeavyImportersForTests()` flag, `cleanTearDir()` teardown.

**Frontend**: Nested `describe`, fixtures for mocks, `render()` + `expect(screen.getByText(...))`.

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

**String dedup**: use `node<String>` instead of `String` when same value is repeated across many objects.

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

**Parallel `await` only fires inside a task context.** Plain `curl POST` runs serially. To fan out:
\`\`\`bash
curl -H "task:''" -X POST -d '[]' http://localhost:8080/module::compute  # task header
./bin/greycat run compute                                                 # CLI task
\`\`\`
**Don't** "fix" non-parallel HTTP via `System::exec` + `&; wait` — the second `System::exec` in a non-task HTTP request throws uncatchable `"terminated PID X"`.

**Request TTL kills + try/catch does NOT fire** — `GREYCAT_REQUEST_TTL` defaults to 20s. Past that the runtime tears down the handler; no exception. For long endpoints, raise the env var or move to task.

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
- Batch `await` jobs in **~120**, never full worker count. Leave ≥8 threads for OS. No nested `await`.
- Pre-allocate shards in a sequential phase. Inside parallel jobs: only WRITE to existing nodes — no `node<T>{...}` constructors, no `nodeIndex` instantiation, no edges to shared parents. Global indices read-only during parallel phases.

**Recurring tasks**:
\`\`\`gcl
Scheduler::add(
  jobs::nightly_job,                                                // <module>::<fn>
  DailyPeriodicity { hour: 2, minute: 0, second: 0, timezone: TimeZone::UTC },
  PeriodicOptions { start: time::now(), max_duration: 1_hour }
);
\`\`\`

**Task lifecycle**: `empty → waiting → running → await → ended | error | cancelled | ended_with_errors`.
- Status: `POST /runtime::Task::info` with `[user_id, task_id]`.
- Result: `GET /files/<user_id>/tasks/<task_id>/result.gcb?json`.

**Atomicity**: Each `fn` invocation is one atomic transaction — uncaught throw rolls back all graph mutations.

---

## API Routing (@expose)

| Route shape | Always works | Drops when... |
|-------------|--------------|---------------|
| `POST /<module>::<fn>` | ✅ always | — |
| `POST /<fn>` (bare alias) | ✅ when unique project-wide | any other fn shares the bare name |
| `POST /<module>::<Type>::<fn>` | ✅ static fn on abstract type | (no bare alias, no `/runtime::` synonym) |

**Rules**:
- Prefer explicit `/<module>::<fn>` from clients — bare alias is fragile.
- `/runtime::Identity::logout` works because `runtime` is a module name — there is NO literal `/runtime::` routing namespace.
- `@expose("alias")` must match the function name (mismatched aliases break the TS SDK lookup).
- `@permission(public)` is invalid (identifier); use `@permission("public")` (quoted).
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

**Iterate nullable collection**: `for (i, v in maybe_list?) { /* ... */ }` (`?` propagates null through head).

**Lazy init**: `this.entries ?= nodeIndex<String, node<Entry>>{};`

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
| `@expose("renamed")` ≠ fn | Keep alias = fn name (TS SDK) |
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
| `await` from plain POST | `task:''` header or `./bin/greycat run` |
| Catching the 20s TTL kill | You can't — raise TTL or move to task |
| Functions without `///` docs | Document ALL functions |

---

## Database

**No migrations.** Adding a non-nullable field to a persisted type with existing data **fails outright** — add as nullable (`T?`) until backfilled. Removing/renaming also requires reset.

**Dev reset**: `rm -rf gcdata && greycat run import`

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

**Logs** — `info/warn/error` are **silently swallowed under non-TTY** (systemd, docker without `-t`). Backend writes every log line to **`files/root/log.csv`**: `tail -f files/root/log.csv`.

**Security & rate limiting** — `std::Scheduler` has no throttle primitive. `@permission("public")` auth endpoints have NO in-process rate limiting. Mitigate externally: bind to loopback, front with nginx `limit_req_zone` keyed on `$binary_remote_addr`.

**systemd** — use **project-local `./bin/greycat`** in `ExecStart` (host-wide `~/.greycat/bin/greycat` has been observed stale at `0.0.0`). Dev builds report `0.0.0` from `--version` — treat as "latest".

---

## Environment

\`\`\`bash
# Backend (.env)
GREYCAT_PORT=8080
GREYCAT_WEBROOT=webroot
GREYCAT_CACHE=30000
\`\`\`

**vite.config.ts** (in root):
\`\`\`ts
export default defineConfig({
  root: 'app',
  build: { outDir: '../webroot', emptyOutDir: true }
})
\`\`\`

**tsconfig.json** (in root): `"paths": { "~/*": ["app/*"] }`, `"include": ["app", "project.d.ts"]`.

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

1. Use `/greycat` skill for backend
2. `greycat-lang lint` after EVERY change (0 errors required)
3. `greycat-lang fmt -p project.gcl` to format
4. `Grep` before deleting
5. `greycat codegen ts` after backend type changes
6. Test: `greycat test` / `pnpm test`

---

## Consistency Checklist

- [ ] `greycat-lang lint` shows 0 errors
- [ ] `greycat-lang fmt -p project.gcl` applied
- [ ] All `@expose` have try/catch + `error()` log
- [ ] All thrown errors are typed
- [ ] All functions/types have `///` docs
- [ ] Transient/API types are `@volatile` (`…View` suffix)
- [ ] Non-nullable collection fields initialized in literal
- [ ] No `static fn <T>` (generic statics)
- [ ] `@expose` routes use explicit `/<module>::<fn>` from clients
- [ ] `greycat codegen ts` re-run after backend type changes
- [ ] Re-import path is upsert
- [ ] New fields on persisted types are nullable OR `gcdata/` reset planned
- [ ] Frontend: exact versions in package.json
- [ ] Frontend uses `gcRuntime` re-export, not bare `gc.*`
- [ ] Tests pass (`rm -rf gcdata && greycat test`)

---

## LSP

`greycat-lang server --stdio` — autocomplete, hover, go-to-def, diagnostics, format, rename. Always run `greycat-lang lint` before commit.

More: https://doc.greycat.io/
```
