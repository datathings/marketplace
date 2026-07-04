---
name: scaffold
description: Generate models, services, APIs, and tests with proper GreyCat structure
allowed-tools: AskUserQuestion, Read, Write, Bash, Grep, Glob
---

# GreyCat Scaffold Generator

**Purpose**: Generate model + service + API + tests with proper GreyCat structure — and, when a frontend exists, a matching **Lit + Shoelace + Lucide** UI component.

Generated files:
- `src/<feature>/<feature>.gcl` — type + indices + service
- `src/<feature>/<feature>_api.gcl` — `@expose` + `@volatile` views
- `test/<feature>_test.gcl` — tests
- `frontend/components/<feature>-table.ts` — Lit component (only if `frontend/` exists)

Templates: **CRUD** | **Time-series collector** | **Graph traversal** | **Custom**

**Frontend stack** (when scaffolding UI): **Lit** + **TypeScript** + **Shoelace** (`@shoelace-style/shoelace`) + **Lucide** (`lucide`/`lucide-static`), VitePlus (`vp`) + `@greycat/web`. Pin exact latest versions; optimize with Lighthouse and ship LLM-friendly SEO.

---

## Step 1: Ask template (AskUserQuestion)

- CRUD Service (Recommended) — create/read/update/delete + indices
- Time-series Collector — `nodeTime` index for temporal data
- Graph Traversal — relationships + traversal queries
- Custom (guided)

---

## Step 2: Detect project structure

\`\`\`bash
[ ! -f "project.gcl" ] && { echo "ERROR: not a GreyCat project root"; exit 1; }
mkdir -p src test
\`\`\`

---

## Step 3: Gather entity details (AskUserQuestion)

- Entity name (PascalCase: `Device`, `User`, `Order`)
- Fields (one per line: `name: String`, `email: String`, `age: int?`, `created_at: time`)
- Indices (multiSelect):
  - By ID: `nodeIndex<int, node<T>>`
  - By unique field: `nodeIndex<String, node<T>>`
  - List: `nodeList<node<T>>`
  - Time-series: `nodeTime<T>` (primitive or node)
  - Geo: `nodeGeo<node<T>>`

---

## Step 4: Detect project style

\`\`\`bash
find src -name "*.gcl" | head -3   # check existing patterns
\`\`\`
Match existing style by inspecting sibling `.gcl` files (indentation, error handling style); run `greycat-lang fmt` — the formatter owns width/layout.

---

## Step 5: Generate Files

⚠ **Filename = module name.** Two `.gcl` files with same basename collide at lint regardless of folder. Convention:
- `src/<feature>/<feature>.gcl` — simple name (model + service)
- `src/<feature>/<feature>_api.gcl` — suffixed
- `src/<feature>/<feature>_reader.gcl` / `_writer.gcl` — IO
- `test/<feature>_test.gcl` — tests

**Typed error hierarchy** (one-time setup, scaffold creates `src/errors.gcl` if absent):
\`\`\`gcl
@volatile abstract type AppError { code: String; message: String; }
@volatile type NotFoundError   extends AppError { id: String; }
@volatile type ValidationError extends AppError { field: String; }
@volatile type ConflictError   extends AppError {}
@volatile type AuthError       extends AppError {}
\`\`\`

### A. Feature file (`src/{snake}/{snake}.gcl`)

\`\`\`gcl
// {Entity} model + indices + service

type {Entity} {
    id: int;
    {fields}
    created_at: time;
}

var {entities}_by_id: nodeIndex<int, node<{Entity}>>;
{additional_indices}
var {entity}_id_counter: node<int?>;

abstract type {Entity}Service {
    static fn create({params}): node<{Entity}> {
        {validation}              // e.g. unique-name check → throw ConflictError {...}

        var id = ({entity}_id_counter.resolve() ?? 0) + 1;
        {entity}_id_counter.set(id);

        var {entity} = node<{Entity}>{ {Entity} {
            id: id, {field_assignments}, created_at: time::now()
        }};

        {entities}_by_id.set(id, {entity});
        {additional_index_inserts}
        return {entity};
    }

    static fn find_by_id(id: int): node<{Entity}>? {
        return {entities}_by_id.get(id);
    }

    {additional_find_methods}

    static fn list_all(): Array<node<{Entity}>> {
        var results = Array<node<{Entity}>> {};
        for (id, e in {entities}_by_id) results.add(e);
        return results;
    }

    static fn update_{field}({entity}: node<{Entity}>, new_{field}: {Type}) {
        {index_update_logic}      // remove from old key, set new
        {entity}->{field} = new_{field};
    }

    static fn delete({entity}: node<{Entity}>) {
        {entities}_by_id.remove({entity}->id);
        {additional_index_removals}
    }
}
\`\`\`

### B. API file (`src/{snake}/{snake}_api.gcl`)

\`\`\`gcl
@volatile type {Entity}View   { id: int; {fields}; created_at: time; }
@volatile type {Entity}Create { {fields_without_id_created_at} }
@volatile type {Entity}Update { {updatable_fields_as_nullable} }

// Use explicit module routes from clients: POST /{snake}_api::get_{entity}_by_id
// The route module is the file basename (`{snake}_api`), not the entity name.
// Every @expose body wraps in try/catch + error() + re-throw.

@expose
fn get_{entities}(): Array<{Entity}View> {
    try {
        var views = Array<{Entity}View> {};
        for ({entity} in {Entity}Service::list_all())
            views.add({Entity}View { {field_mappings} });
        return views;
    } catch (ex) { error("get_{entities}() failed: ${ex}"); throw ex; }
}

@expose
fn get_{entity}_by_id({entity}Id: int): {Entity}View {
    try {
        var {entity} = {Entity}Service::find_by_id({entity}Id);
        if ({entity} == null) {
            throw NotFoundError { code: "NOT_FOUND", message: "{Entity} not found", id: "${{entity}Id}" };
        }
        return {Entity}View { {field_mappings} };
    } catch (ex) { error("get_{entity}_by_id(${{entity}Id}) failed: ${ex}"); throw ex; }
}

@expose @permission("admin")
fn create_{entity}(data: {Entity}Create): {Entity}View {
    try {
        var {entity} = {Entity}Service::create({param_list});
        return {Entity}View { {field_mappings} };
    } catch (ex) { error("create_{entity}() failed: ${ex}"); throw ex; }
}

@expose @permission("admin")
fn update_{entity}({entity}Id: int, data: {Entity}Update): {Entity}View {
    try {
        var {entity} = {Entity}Service::find_by_id({entity}Id);
        if ({entity} == null) {
            throw NotFoundError { code: "NOT_FOUND", message: "{Entity} not found", id: "${{entity}Id}" };
        }
        {update_calls}              // if (data.field != null) Service::update_field(entity, data.field!!);
        return {Entity}View { {field_mappings} };
    } catch (ex) { error("update_{entity}(${{entity}Id}) failed: ${ex}"); throw ex; }
}

@expose @permission("admin")
fn delete_{entity}({entity}Id: int) {
    try {
        var {entity} = {Entity}Service::find_by_id({entity}Id);
        if ({entity} == null) {
            throw NotFoundError { code: "NOT_FOUND", message: "{Entity} not found", id: "${{entity}Id}" };
        }
        {Entity}Service::delete({entity});
    } catch (ex) { error("delete_{entity}(${{entity}Id}) failed: ${ex}"); throw ex; }
}
\`\`\`

### C. Test file (`test/{snake}_test.gcl`)

\`\`\`gcl
@test fn test_{entity}_create() {
    var e = {Entity}Service::create({test_params});
    Assert::isNotNull(e);
    Assert::equals(e->{field}, {expected});
}

@test fn test_{entity}_find_by_id() {
    var e = {Entity}Service::create({test_params});
    var found = {Entity}Service::find_by_id(e->id);
    Assert::isNotNull(found);
    Assert::equals(found->id, e->id);
}

@test fn test_{entity}_find_{unique_field}() {
    var e = {Entity}Service::create({test_params});
    var found = {Entity}Service::find_by_{field}(e->{field});
    Assert::isNotNull(found);
}

@test fn test_{entity}_list_all() {
    {Entity}Service::create({test_params_1});
    {Entity}Service::create({test_params_2});
    Assert::isTrue({Entity}Service::list_all().size() >= 2);
}

@test fn test_{entity}_update() {
    var e = {Entity}Service::create({test_params});
    {Entity}Service::update_field(e, {new_value});
    Assert::equals(e->field, {new_value});
}

@test fn test_{entity}_delete() {
    var e = {Entity}Service::create({test_params});
    var id = e->id;
    {Entity}Service::delete(e);
    Assert::isNull({Entity}Service::find_by_id(id));
}

@test fn test_{entity}_duplicate_validation() {
    {Entity}Service::create({test_params});
    var failed = false;
    try { {Entity}Service::create({test_params}); } catch (ex) { failed = true; }
    Assert::isTrue(failed);
}
\`\`\`

### D. Frontend component (`frontend/components/{snake}-table.ts`) — only if `frontend/` exists

Generate a small **Lit** element that lists the entity via the generated client, using **Shoelace** for chrome and a **Lucide** icon. Then regenerate the typed client (`greycat codegen ts`) so `gc.project.{Entity}View` exists.

\`\`\`ts
import { LitElement, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import '@greycat/web/sdk';                        // runtime: `gc` global, init, typed bindings
import '@shoelace-style/shoelace/dist/components/card/card.js';
import '@shoelace-style/shoelace/dist/components/spinner/spinner.js';
import { icon } from '~/icons';                  // Lucide (lucide / lucide-static), self-hosted

@customElement('{snake}-table')
export class {Entity}Table extends LitElement {
  createRenderRoot() { return this; }            // light DOM — let theme.css cascade
  @state() private rows: gc.project.{Entity}View[] = [];
  @state() private loading = true;

  async connectedCallback() {
    super.connectedCallback();
    // typed call through the SDK (module = file basename `{snake}_api`)
    this.rows = await gc.{snake}_api.get_{entities}();
    this.loading = false;
  }

  render() {
    if (this.loading) return html`<sl-spinner></sl-spinner>`;
    return html`
      <sl-card>
        <h2 slot="header">${icon('table-2', 18)} {Entity} list</h2>
        <table>
          <thead><tr><th scope="col">ID</th>{th_cells}</tr></thead>
          <tbody>
            ${this.rows.map(r => html`<tr><td>${r.id}</td>{td_cells}</tr>`)}
          </tbody>
        </table>
      </sl-card>`;
  }
}
declare global { interface HTMLElementTagNameMap { '{snake}-table': {Entity}Table } }
\`\`\`

Reminders: import Shoelace components **per-component** (tree-shaking); derive endpoint/field strings from `project.d.ts` `$fields` where possible; keep content semantic + accessible (`scope`, `alt`/`aria`) for SEO; run `pnpm lighthouse` after wiring the page.

---

## Step 6: Lint
\`\`\`bash
greycat-lang lint --fix     # backend
# frontend (if generated):
greycat codegen ts && pnpm lint   # regenerate client, then typecheck
\`\`\`

---

## Step 7: Report

```
SCAFFOLD COMPLETE — entity: {Entity}

✓ src/{snake}/{snake}.gcl          — type + N indices + service ({methods})
✓ src/{snake}/{snake}_api.gcl      — N volatile types + N endpoints
✓ test/{snake}_test.gcl            — N test cases
✓ frontend/components/{snake}-table.ts  — Lit + Shoelace + Lucide (if frontend/)

Lint: ✓ passes

Next:
  1. Customize generated code
  2. greycat test test/{snake}_test.gcl
  3. greycat serve → test endpoints
  4. (frontend) greycat codegen ts → mount <{snake}-table> → pnpm lighthouse
```

---

## Template Variations

### Time-series Collector
\`\`\`gcl
type Sensor { id: String; location: geo; readings: nodeTime<float>; }
var sensors_by_id: nodeIndex<String, node<Sensor>>;

// Service additions:
static fn record_reading(s: node<Sensor>, v: float, t: time) { s->readings.setAt(t, v); }

static fn get_readings(s: node<Sensor>, start: time, end: time): Array<Tuple<time, float>> {
    var r = Array<Tuple<time, float>> {};
    for (t: time, v: float in s->readings[start..end]) r.add(Tuple { x: t, y: v });
    return r;
}

static fn get_average(s: node<Sensor>, start: time, end: time): float {
    var sum = 0.0; var n = 0;
    for (t: time, v: float in s->readings[start..end]) { sum = sum + v; n = n + 1; }
    if (n > 0) { return sum / n; } else { return 0.0; }
}
\`\`\`

### Graph Traversal
\`\`\`gcl
type City   { id: int; name: String; country: node<Country>; streets: nodeList<node<Street>>; }
type Street { id: int; name: String; city: node<City>;       buildings: nodeList<node<Building>>; }

// Traversal:
static fn get_city_with_streets(city: node<City>): CityWithStreetsView {
    var sv = Array<StreetView> {};
    for (i, s in city->streets) sv.add(StreetView { id: s->id, name: s->name });
    return CityWithStreetsView { id: city->id, name: city->name, streets: sv };
}
\`\`\`

---

## Notes

- Generated code is a starting point — customize as needed
- Maintains index consistency across all CRUD ops
- Includes duplicate-check validation for unique fields
- reads default to `api` (bare `@expose`, no decorator); writes use `@permission("admin")`; add `@permission("public")` ONLY for intentionally anonymous endpoints
- Throws typed errors (`NotFoundError`, `ConflictError`) on failures
