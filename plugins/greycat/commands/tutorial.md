---
name: tutorial
description: Interactive learning modules for GreyCat concepts - from basics to advanced patterns
allowed-tools: AskUserQuestion, Read, Write, Bash, Grep, Glob
---

# GreyCat Interactive Tutorial

**Purpose**: Progressive hands-on learning of GreyCat with code examples + validation.

11 sequential modules (~4.5 hours total):

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
| 11 | Frontend | 35m | Lit + Shoelace + Lucide, Lighthouse, LLM-friendly SEO |

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

**Concept**: statically-typed, non-null by default, no C-style for, no ternary.

Primitives: `int`, `float`, `bool`, `String`, `char`, `time`.

\`\`\`gcl
fn greet(name: String?): String {
    if (name == null) return "Hello, stranger!";
    return "Hello, ${name}!";
}
\`\`\`

**Exercise** (`tutorial/module1_basics.gcl`):
\`\`\`gcl
fn calculate_age(birth_year: int, current_year: int): int {
    return 0;  // TODO
}

@test fn test_calculate_age() {
    Assert::equals(calculate_age(1990, 2024), 34);
    Assert::equals(calculate_age(2000, 2024), 24);
}
\`\`\`

Validate: `greycat test tutorial/module1_basics.gcl`. Hint: `current_year - birth_year`.

---

## Module 2: Persistence (25m)

**Concept**: separate transient (RAM) from persistent (storage).

```
Plain object:   var u = User { name: "Alice" };           // RAM only
Persistent:     var n = node<User>{ User { ... } };        // gcdata/

Use node<T>:
  ✓ Module-level vars (global data)
  ✓ Type fields (relationships)
  ✗ Local variables
  ✗ Function parameters/returns (except passing persisted refs)
```

\`\`\`gcl
type Country { name: String; code: String; }
var countries_by_code: nodeIndex<String, node<Country>>;

fn create_country(name: String, code: String): node<Country> {
    var c = node<Country>{ Country { name: name, code: code }};
    countries_by_code.set(code, c);
    return c;
}
\`\`\`

**Exercise**: implement `Product` persistence with `products_by_id: nodeIndex<int, node<Product>>`, `create_product()`, `find_product()`.

---

## Module 3: Indexed Collections (30m)

| Persisted | Key | Local |
|-----------|-----|-------|
| `nodeList<node<T>>` | int | `Array<T>` |
| `nodeIndex<K, V>` | K | `Map<K, V>` |
| `nodeTime<T>` | time | — |
| `nodeGeo<node<T>>` | geo | — |

\`\`\`gcl
type City { name: String; streets: nodeList<node<Street>>; }
type Street { name: String; }
var cities_by_name: nodeIndex<String, node<City>>;

fn create_city(name: String, pop: int): node<City> {
    var c = node<City>{ City { name: name, streets: nodeList<node<Street>>{} }};  // ⚠ MUST init
    cities_by_name.set(name, c);
    return c;
}
\`\`\`

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
\`\`\`gcl
type Xxx { ... }
var xxx_by_id: nodeIndex<int, node<Xxx>>;

abstract type XxxService {
    static fn create(...): node<Xxx> { }
    static fn find(...): node<Xxx>? { }
    static fn update(...) { }
    static fn delete(...) { }
}
\`\`\`

**Exercise**: `UserService` with email-uniqueness validation.

---

## Module 6: API Development (25m)

**Rules**:
- `@volatile` for request/response types
- Never return `nodeList`/`nodeIndex` from APIs — use `Array<XxxView>`
- `@expose` makes function available via HTTP
- `@permission` for access control
- API files: `src/<feature>/<feature>_api.gcl`

\`\`\`gcl
@volatile type UserView { ... }
@expose @permission("public")
fn get_users(): Array<UserView> { ... }
\`\`\`

**Exercise**: REST API for blog system (Module 4).

---

## Module 7: Testing (20m)

\`\`\`gcl
@test fn test_name() {
    // Arrange
    var u = UserService::create("a@b.com");
    // Act
    var found = UserService::find("a@b.com");
    // Assert
    Assert::isNotNull(found);
    Assert::equals(found->email, "a@b.com");
}
\`\`\`

Assertions: `equals`, `isTrue`, `isFalse`, `isNull`, `isNotNull`.
Lifecycle: `fn setup()` / `fn teardown()`.

**Exercise**: tests for `UserService`.

---

## Module 8: Parallelization (25m)

\`\`\`gcl
var jobs = Array<Job> {};
for (i, item in items) jobs.add(Job { function: process_fn, arguments: [item] });
await(jobs, MergeStrategy::strict);                  // 2nd arg required
for (i, job in jobs) { var result = job.result() as ResultType; }
\`\`\`

⚠ Use `Array<Job>` not `Array<Job<T>>` (crashes at runtime). Cast `.result()` at collection. Batch ~120 jobs. Call via `task:''` header from HTTP, or via CLI.

**Exercise**: parallelize processing of 1000 items.

---

## Module 9: Time-Series & Geo (30m)

\`\`\`gcl
var temps: nodeTime<float>;
temps.setAt(timestamp, value);
for (t: time, v: float in temps[start..end]) { /* ... */ }

var devices: nodeGeo<node<Device>>;
devices.set(geo{lat, lng}, device);
// nodeGeo has no `.filter()` method; iterate then test bbox.contains(pos):
var bbox = GeoBox { sw: geo{south, west}, ne: geo{north, east} };
for (pos: geo, d in devices) { if (bbox.contains(pos)) { /* ... */ } }
\`\`\`

**Exercise**: temperature monitoring with sensor readings over time.

---

## Module 10: Advanced (30m)

\`\`\`gcl
abstract type Animal {
    name: String;
    abstract fn makeSound(): String;  // abstract from day 1!
}
type Dog extends Animal { fn makeSound(): String { return "Woof!"; } }
type Cat extends Animal { fn makeSound(): String { return "Meow!"; } }

var animals: nodeIndex<String, node<Animal>>;
animals.set("d", node<Animal>{ Dog { name: "Rex" } });
animals.get("d")?->makeSound();      // dispatches to Dog::makeSound
\`\`\`

Reminder: concrete methods on `abstract type` CANNOT be overridden. Declare `abstract` from day 1.

**Exercise**: payment system with Card/Cash/Crypto types.

---

## Module 11: Frontend (Lit + Shoelace + Lucide) (35m)

**Concept**: the preferred GreyCat frontend is **web components** — **Lit** + **TypeScript** + **Shoelace** (UI kit) + **Lucide** icons (`lucide`/`lucide-static`), built with **Vite** and the typed **`@greycat/web`** client. Pin exact latest versions; use the native packages above.

**Setup** (configs in root, source in `frontend/`, builds to `webroot/`):
\`\`\`bash
pnpm install
pnpm gen            # greycat codegen ts → project.d.ts (typed client)
pnpm dev            # Vite dev server
\`\`\`

**A Lit component** consuming an `@expose` endpoint (from Module 6):
\`\`\`ts
import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import '@shoelace-style/shoelace/dist/components/card/card.js';   // per-component (tree-shaking)
import { gcRuntime } from '~/gc-runtime';     // re-export — never bare `gc.*` (@types/node shadow)
import { icon } from '~/icons';               // Lucide, self-hosted (no CDN)

@customElement('app-products')
export class AppProducts extends LitElement {
  static styles = css`:host{display:block}`;
  @state() private rows: gcRuntime.project.ProductView[] = [];

  async connectedCallback() {
    super.connectedCallback();
    this.rows = await gcRuntime.default.call('products::get_products', []);
  }
  render() {
    return html`<sl-card><h2 slot="header">${icon('boxes', 18)} Products</h2>
      <ul>${this.rows.map(r => html`<li>${r.name}</li>`)}</ul></sl-card>`;
  }
}
\`\`\`

**Optimize with Lighthouse** (target ≥ 90 in performance / accessibility / best-practices / SEO):
\`\`\`bash
greycat serve            # serve the built app first
pnpm lighthouse          # full report; also :desktop / :ci
\`\`\`
Levers: per-component Shoelace imports, self-hosted Lucide icons (no CDN), code-split routes, defer non-critical JS, reserve element sizes (avoid layout shift).

**LLM-friendly SEO** (always): in `index.html` set `<html lang>`, a unique `<title>`, `<meta name="description">`, canonical link, Open Graph + Twitter Card, and JSON-LD (`schema.org`). Use semantic landmarks + `alt`/ARIA (pre-render/SSR if content is in Shadow DOM). Ship `robots.txt`, `sitemap.xml`, a web manifest, and **`llms.txt`** (a Markdown index of pages + endpoints) to `webroot/`.

**Exercise**: build an `<app-products>` page for the Module 4 blog/catalog, wire it to your `@expose` endpoint, then run `pnpm lighthouse` and get every category ≥ 90.

---

## Between Modules

Ask (AskUserQuestion):
- Continue to Module N+1
- Review this module
- Take a break (save progress)

Update progress file after each completion.

---

## Completion

After Module 11: show certificate listing all completed modules, time invested, tutorial directory, next steps (build a real project with a Lit + Shoelace + Lucide frontend, use `/scaffold` / `/migrate` / `/frontend`, audit with `pnpm lighthouse`, read references, https://greycat.io/community).

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
└── module11_frontend/             # Lit + Shoelace + Lucide (app-products.ts, index.html, llms.txt)
```

---

## Notes

- Each module: theory + hands-on
- Progressive (each builds on prior)
- Real runnable code
- Tests validate understanding
- Self-paced
