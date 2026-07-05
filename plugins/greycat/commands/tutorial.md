---
name: tutorial
description: Interactive learning modules for GreyCat concepts - from basics to advanced patterns
allowed-tools: AskUserQuestion, Read, Write, Bash, Grep, Glob
---

# GreyCat Interactive Tutorial

**Purpose**: progressive hands-on learning of GreyCat — theory + runnable code + validation. 11 sequential modules (each builds on the prior), ~4.5 hours total.

| # | Module | Time | Topic |
|---|--------|------|-------|
| 1 | Basics | 20m | Types, nullability, functions |
| 2 | Persistence | 25m | `node<T>` vs plain objects |
| 3 | Collections | 30m | nodeList/Index/Time/Geo |
| 4 | Modeling | 25m | Relationships + indices |
| 5 | Services | 20m | Abstract type + static fns |
| 6 | APIs | 25m | @expose, @permission, @volatile |
| 7 | Testing | 20m | @test, Assert |
| 8 | Parallelization | 25m | Jobs, await |
| 9 | Time & Geo | 30m | nodeTime, nodeGeo |
| 10 | Advanced | 30m | Inheritance, polymorphism |
| 11 | Frontend | 35m | Lit + Shoelace + lucide-static, Lighthouse, LLM-friendly SEO |

---

## Progress Tracking

`.greycat-tutorial-progress`:
```json
{
  "started": "<ISO>",
  "current_module": 5,
  "completed_modules": [1, 2, 3, 4],
  "last_session": "<ISO>"
}
```

---

## Step 1: Check progress + choose path

If file exists: show resume prompt. Else: create with `current_module: 1`.

Ask (AskUserQuestion):
- Continue from Module N
- Start from beginning
- Jump to specific module
- Exit

---

## Module 1: Basics (20m)

**Concept**: statically-typed, non-null by default, no ternary (use if/else). Primitives: `int`, `float`, `bool`, `String`, `char`, `time`.

```gcl
fn greet(name: String?): String {
    if (name == null) return "Hello, stranger!";
    return "Hello, ${name}!";
}
```

**Exercise** (`tutorial/module1_basics.gcl`):
```gcl
fn calculate_age(birth_year: int, current_year: int): int {
    return 0;  // TODO
}

@test fn test_calculate_age() {
    Assert::equals(calculate_age(1990, 2024), 34);
    Assert::equals(calculate_age(2000, 2024), 24);
}
```

Validate: `greycat test test_calculate_age` (a `@test` function name, NOT a file path). Hint: `current_year - birth_year`.

---

## Module 2: Persistence (25m)

**Concept**: separate transient (RAM) from persistent (storage).

```
Plain object:   var u = User { name: "Alice" };           // RAM only
Persistent:     var n = node<User>{ User { ... } };        // gcdata/

Use node<T>:
  ✓ Module-level vars (global data)
  ✓ Type fields (relationships)
  ✗ Local variables that only need to be transient (use plain objects / `Array` / `Map` for scratch data)
  ✗ Function parameters/returns (except passing persisted refs)
```

```gcl
type Country { name: String; code: String; }
var countries_by_code: nodeIndex<String, node<Country>>;

fn create_country(name: String, code: String): node<Country> {
    var c = node<Country>{ Country { name: name, code: code }};
    countries_by_code.set(code, c);
    return c;
}
```

**Exercise**: implement `Product` persistence with `products_by_id: nodeIndex<int, node<Product>>`, `create_product()`, `find_product()`.

---

## Module 3: Indexed Collections (30m)

| Persisted | Key | Local |
|-----------|-----|-------|
| `nodeList<node<T>>` | int | `Array<T>` |
| `nodeIndex<K, V>` | K | `Map<K, V>` |
| `nodeTime<T>` | time | — |
| `nodeGeo<node<T>>` | geo | — |

```gcl
type City { name: String; streets: nodeList<node<Street>>; }
type Street { name: String; }
var cities_by_name: nodeIndex<String, node<City>>;

fn create_city(name: String, pop: int): node<City> {
    var c = node<City>{ City { name: name, streets: nodeList<node<Street>>{} }};  // ⚠ MUST init
    cities_by_name.set(name, c);
    return c;
}
```

**Exercise**: school system — Student/Course with many-to-many via two indices.

---

## Module 4: Data Modeling (25m)

**Rules**:
- Store `node<T>` refs for relationships, not embedded objects
- Global indices for primary lookups
- Initialize collection attrs in constructors
- File structure: `src/<feature>/<feature>.gcl` (model+service), `src/<feature>/<feature>_api.gcl` (@expose+@volatile)

**Exercise**: blog — User writes Posts, Posts have Comments.

---

## Module 5: Services & Business Logic (20m)

**Pattern** (in `src/<feature>/<feature>.gcl`):
```gcl
type Xxx { ... }
var xxx_by_id: nodeIndex<int, node<Xxx>>;

abstract type XxxService {
    static fn create(...): node<Xxx> { }
    static fn find(...): node<Xxx>? { }
    static fn update(...) { }
    static fn delete(...) { }
}
```

**Exercise**: `UserService` with email-uniqueness validation.

---

## Module 6: API Development (25m)

**Rules**:
- `@volatile` for request/response types
- Never return `nodeList`/`nodeIndex` from APIs — use `Array<XxxView>`
- `@expose` makes function available via HTTP
- A bare `@expose` already requires the `api` permission (authenticated) — that's the default you want; add `@permission("public")` only to allow anonymous callers (never on mutations, quoted), `@permission("admin")` for privileged ops
- API files: `src/<feature>/<feature>_api.gcl` (@expose can also live in `src/api.gcl`)

```gcl
@volatile type UserView { ... }
@expose
fn get_users(): Array<UserView> { ... }   // bare @expose ⇒ requires `api` (authenticated)
```

**Exercise**: REST API for blog system (Module 4).

---

## Module 7: Testing (20m)

```gcl
@test fn test_name() {
    // Arrange
    var u = UserService::create("a@b.com");
    // Act
    var found = UserService::find("a@b.com");
    // Assert
    Assert::isNotNull(found);
    var f = found!!;                     // narrow node<User>? → node<User>
    Assert::equals(f->email, "a@b.com");
}
```

Assertions: `equals`, `isTrue`, `isFalse`, `isNull`, `isNotNull`.
Lifecycle: `fn setup()` / `fn teardown()`.

**Exercise**: tests for `UserService`.

---

## Module 8: Parallelization (25m)

```gcl
var jobs = Array<Job> {};
for (i, item in items) jobs.add(Job { function: process_fn, arguments: [item] });
await(jobs, MergeStrategy::strict);                  // 2nd arg required
for (i, job in jobs) { var result = job.result() as ResultType; }
```

⚠ Use `Array<Job>` not `Array<Job<T>>` (crashes at runtime). Cast `.result()` at collection. Batch ~120 jobs. An `@expose` HTTP call already runs as a task (so `await` fans out); a one-shot `greycat run` runs jobs serially. To dispatch a long HTTP call as a background task, add the `task: true` header.

**Exercise**: parallelize processing of 1000 items.

---

## Module 9: Time-Series & Geo (30m)

```gcl
var temps: nodeTime<float>;
temps.setAt(timestamp, value);
for (t: time, v: float in temps[start..end]) { /* ... */ }

var devices: nodeGeo<node<Device>>;
devices.set(geo{lat, lng}, device);
// nodeGeo has no `.filter()` method; iterate then test bbox.contains(pos):
var bbox = GeoBox { sw: geo{south, west}, ne: geo{north, east} };
for (pos: geo, d in devices) { if (bbox.contains(pos)) { /* ... */ } }
```

**Exercise**: temperature monitoring with sensor readings over time.

---

## Module 10: Advanced (30m)

```gcl
abstract type Animal {
    name: String;
    abstract fn makeSound(): String;  // abstract from day 1!
}
type Dog extends Animal { fn makeSound(): String { return "Woof!"; } }
type Cat extends Animal { fn makeSound(): String { return "Meow!"; } }

var animals: nodeIndex<String, node<Animal>>;
animals.set("d", node<Animal>{ Dog { name: "Rex" } });
animals.get("d")?->makeSound();      // dispatches to Dog::makeSound
```

Reminder: concrete methods on `abstract type` CANNOT be overridden. Declare `abstract` from day 1.

**Exercise**: payment system with Card/Cash/Crypto types.

---

## Module 11: Frontend (Lit + Shoelace + lucide-static) (35m)

**Concept**: preferred GreyCat frontend = web components — **VitePlus** (`vp`) + **Lit** (light DOM) + **TypeScript** + **Shoelace** (UI kit) + typed **`@greycat/web`** client + **lucide-static** icons (self-hosted inline SVG, no CDN), **pnpm**. Pin exact latest versions; use these native packages.

**Setup** (configs in root, source in `frontend/`, builds to `webroot/`):
```bash
pnpm install
greycat codegen ts   # → project.d.ts (typed client)
greycat dev          # VitePlus build watcher + serve API/assets on :8080
```

**A Lit component** consuming an `@expose` endpoint (from Module 6):
```ts
import { LitElement, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import '@shoelace-style/shoelace/dist/components/card/card.js';   // per-component (tree-shaking)
import '@greycat/web/sdk';                     // `gc` global + typed bindings
import { icon } from '~/icons';               // lucide-static, self-hosted (no CDN)

@customElement('app-products')
export class AppProducts extends LitElement {
  createRenderRoot() { return this; }          // light DOM: theme.css cascades, content is crawlable
  @state() private rows: gc.project.ProductView[] = [];

  async connectedCallback() {
    super.connectedCallback();
    // `gc.<module>.*` only works after `gc.sdk.init()` has resolved (the route root gates on init)
    this.rows = await gc.products_api.get_products();   // module = api file basename
  }
  render() {
    return html`<sl-card><h2 slot="header">${icon('boxes', 18)} Products</h2>
      <ul>${this.rows.map(r => html`<li>${r.name}</li>`)}</ul></sl-card>`;
  }
}
```

**Optimize with Lighthouse** (optional tooling; target ≥ 90 in performance / accessibility / best-practices / SEO):
```bash
greycat serve            # serve the built app first
pnpm lighthouse          # if a lighthouse script is defined (also :desktop / :ci); else the `lighthouse` CLI
```
Levers: per-component Shoelace imports, self-hosted lucide-static icons (no CDN), code-split routes, defer non-critical JS, reserve element sizes (avoid layout shift).

**LLM-friendly SEO** (always): in `index.html` set `<html lang>`, a unique `<title>`, `<meta name="description">`, canonical link, Open Graph + Twitter Card, and JSON-LD (`schema.org`). Use semantic landmarks + `alt`/ARIA (pre-render/SSR if content is in Shadow DOM). Ship `robots.txt`, `sitemap.xml`, a web manifest, and **`llms.txt`** (a Markdown index of pages + endpoints) to `webroot/`.

**Exercise**: build an `<app-products>` page for the Module 4 blog/catalog, wire it to your `@expose` endpoint, then run Lighthouse (the `pnpm lighthouse` script if present, else the `lighthouse` CLI) and get every category ≥ 90.

---

## Between Modules

Ask (AskUserQuestion):
- Continue to Module N+1
- Review this module
- Take a break (save progress)

Update progress file after each completion.

---

## Completion

After Module 11: show certificate listing all completed modules, time invested, tutorial directory, next steps (build a real project with a Lit + Shoelace + lucide-static frontend; use `/greycat:scaffold` / `/greycat:migrate` / `/greycat:frontend`; audit with Lighthouse — `pnpm lighthouse` script if present, else the `lighthouse` CLI; read references; https://greycat.io/community).

---

## Files

```
tutorial/
├── module1_basics.gcl
├── module2_persistence.gcl
├── module3_collections.gcl
├── module4_modeling.gcl
├── module5_services.gcl
├── module6_apis.gcl
├── module7_testing.gcl
├── module8_parallelization.gcl
├── module9_timeseries_geo.gcl
├── module10_advanced.gcl
└── module11_frontend/             # Lit + Shoelace + lucide-static (app-products.ts, index.html, llms.txt)
```

Each module = theory + hands-on runnable code; tests validate understanding; progressive (each builds on prior); self-paced.
