---
name: greycat
description: GreyCat full-stack development for graph-based language with built-in persistence. CRITICAL WORKFLOW - After generating or modifying ANY GreyCat backend code (.gcl files), IMMEDIATELY run 'greycat-lang lint' to get linting feedback and fix all errors before proceeding. Use when: (1) working with .gcl files or GreyCat projects, (2) using persisted nodes and indexed collections (nodeList, nodeIndex, nodeTime, nodeGeo), (3) creating data models, services, or abstract types, (4) writing API endpoints with @expose, @permission, or @volatile decorators, (5) implementing parallel processing with Jobs, await(), or PeriodicTask, (6) integrating frontends with @greycat/web SDK or TypeScript type generation, (7) running GreyCat CLI commands (greycat serve/test/run/install/lint), (8) debugging GreyCat applications or working with transactions. NOT for: general graph databases (Neo4j, ArangoDB), generic React apps, or SQL databases.
---

# GreyCat Backend Development

Graph-based language with built-in persistence. Not a database — an evolving coded structure.

## Commands

```bash
greycat install      # download libs
greycat test         # run tests
greycat serve        # start :8080
greycat-lang lint    # check for errors
```

## Development Workflow

**⚠️ CRITICAL: Always lint after code generation**

When generating or modifying GreyCat backend code (.gcl files):

1. **Generate/modify** the code
2. **Immediately run** `greycat-lang lint` to get linting feedback
3. **Fix any errors** reported by the linter before proceeding

This ensures code quality and catches errors early. **NEVER skip this step** when working with GreyCat backend code.

```bash
# Example workflow
# 1. Make changes to backend/src/model/city.gcl
# 2. Run linter immediately
greycat-lang lint
# 3. Fix any reported errors
# 4. Continue development
```

## Architecture

**Feature layout**
- `project.gcl` - Entry point, libs, permissions, roles, main(), init()
- `src/<feature>/<feature>.gcl` - Data models + global indices
- `src/<feature>/<feature>_api.gcl` - @expose + @permission functions, @volatile response types
- `src/<feature>/<feature>_import.gcl` - Import from another format
- `src/<feature>/<feature>_export.gcl` - Export into another format
- `src/<feature>/index.html` - *Optional*, if a frontend is required
- `src/<feature>/index.tsx` - *Optional*, if a frontend is required
- `src/<feature>/<other_page>.html` - *Optional*, if a frontend is required and multiple pages needed
- `src/<feature>/<other_page>.tsx` - *Optional*, if a frontend is required and multiple pages needed

**project.gcl example:**
```gcl
@library("std", "7.5.125-dev");
@library("explorer", "7.5.3-dev"); // enables graph navigation in explorer UI

@include("src"); // includes every `.gcl` files in the folder recursively

fn main() {
    println("Hello, world!");
}
```

### Rules
- **ONLY use `@include` in `project.gcl`** — does NOT work in other `.gcl` files
- `@include("folder")` recursively includes ALL `.gcl` files in that folder

**Essential Libraries:**
- `@library("std", "7.5.125-dev")` — Standard library (required)
- `@library("explorer", "7.5.3-dev")` — Graph navigation UI at `/explorer` (recommended for development)

### Conventions
#### GCL
 - types/enums PascalCase
 - snake_case for everything else (functions, variables, fields)
 - unused vars with underscore prefix `_prefix`
 - test modules postfixed with `_test.gcl`

## Types
### Primitives
All primitives are 64bits in width
- `int`: signed integer eg. `1_000_000`, `-42`
- `float`: double floating-point precision, eg. `3.14`
- `bool`: eg. `true` or `false`
- `char`: eg. `'c'`
- `geo`: morton encoded lat/lng eg. `geo { lat, lng }` where field are `float`
- `time`: signed microseconds timestamp eg. `'2025-01-21T16:28:00Z'`, `time::from(unix_epoch, DurationUnit::seconds)`
- `duration`: signed microseconds duration eg. `5s`, `3_min`, `7hour`
- `node`: reference to data eg. `node<int> { 10 }`
- `nodeTime`: reference to data by `time` eg. `nodeTime<T> {}`
- `nodeList`: reference to data by `int` eg. `nodeList<T> {}`
- `nodeGeo`: reference to data by `geo` eg. `nodeList<T> {}`
- `nodeIndex`: reference to data by `K` eg. `nodeIndex<K, V> {}`
- `str`: eg. `str { "hello" }` ASCII lowercase on
- `t2`: signed tuple of `t2 { x, y }` where fields are `i32`
- `t3`: signed tuple of `t3 { x, y, z }` where fields are `i21`
- `t4`: signed tuple of `t4 { x, y, z, w }` where fields are `i16`
- `tf2`: floaint-point tuple of `tf2 { x, y }` where fields are `f32`
- `tf3`: floaint-point tuple of `tf3 { x, y, z }` where fields are `f21`
- `tf4`: floaint-point tuple of `tf4 { x, y, z, w }` where fields are `f16`
- `type`: reference to a type
- `function`: reference to a function
- `field`: reference to a field

### Type structure
```gcl
@volatile // non-persisted
type ApiResponse { data: String; }
```

### Object creation
```gcl
var arr = Array<String> {};
var map = Map<String, int> {}; // ✅ use {}, NOT ::new()
```

## Nullability

All types non-null by default. Use `?` for nullable:
```gcl
var city: City?;                   // nullable
city?.name?.size();                // optional chaining
city?.name ?? "Unknown";           // nullish coalescing
data.get("key")!!;                 // non-null assertion
if (city == null) { return null; }
city->name;                        // ✅ no !! needed (control flow analysis)
```

**⚠️ Cast + coalescing needs parens:**
```gcl
answer as String? ?? "default"   // WRONG
(answer as String?) ?? "default" // RIGHT
```

**⚠️ NO TERNARY OPERATOR** — use if/else:
```gcl
var res = valid ? 0 : 1;                                   // WRONG
var res: int; if (valid) { res = 0; } else { result = 1; } // RIGHT
```

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

**For advanced topics:** See [references/nodes.md](references/nodes.md) for deep dive on transactions, indexed collection sampling, and complex persistence patterns.

## Collections

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

**Sampling** large time-series: `nodeTime::sample([series], start, end, 1000, SamplingMode::adaptative, null, null)`
Modes: `fixed`, `fixed_reg`, `adaptative`, `dense`

**Array sorting**:
```gcl
cities.sort_by(City::population, SortOrder::desc); // ✅ native typed sort
```

**⚠️ CRITICAL: Initialize non-nullable fields and nodes generics can never be nullable**
```gcl
type Box { x: int; }
var b = Box {};        // WRONG: `x` is non-nullable
var b = Box { x: 42 }; // RIGHT

var n = node<String> {};         // WRONG: `String` is not nullable
var n = node<String> { "text" }; // RIGHT
var n = node<String?> {};        // RIGHT
```

## Global Variables

Global variables must be nodes → graph entrypoints:
```gcl
var count: node<int?>;
var by_id: nodeList<float>;
fn main() { count.set((count.resolve() ?? 0) + 1); }
```

**Global variables are auto-initialized**: `nodeIndex`, `nodeList`, `nodeTime`, `nodeGeo` are automatically initialized by GreyCat — no `{}` needed:
```gcl
// ✅ Global variables — no initialization needed
var cities_by_name: nodeIndex<String, node<City>>;
var all_users: nodeList<node<User>>;

// ⚠️ Non-nullable object fields still need initialization
```

## Modules

**In `<feature>.gcl` files** — declare global nodes first:
```gcl
// ✅ Global variables first, then types
var cities_by_name: nodeIndex<String, node<City>>;

type City {
    name: String;
    country: node<Country>;           // ✅ node of Country
    streets: nodeList<node<Street>>;  // ✅ node list to node of Street
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
fn add(x: int): int { return x + 2; }
fn noReturn() { }  // no void type
var lambda = fn(x: int): int { x * 2 };
for (k: K, v: V in map) { }  // ✅ prefer for-in
for (i, v in nullable?) { }  // ✅ use ? for nullable
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
    fn calculateTax(): float;  // abstract - must implement
    fn getInfo(): String { return address; }  // concrete - shared
}
type House extends Building {
    fn calculateTax(): float { return value * 0.01; }
}

var buildings: nodeIndex<String, node<Building>>{};
for (addr, b in buildings) { b->calculateTax(); }  // polymorphic
```

**Key:** `abstract type` has fields + abstract/concrete methods. Use `node<BaseType>` for polymorphism, `is` for type checks. Concrete methods cannot be overridden.

## Logging & Error Handling

```gcl
info("msg ${var}"); warn("msg"); error("msg");
try { op(); } catch (ex) { error("${ex}"); }
```

## Parallelization

```gcl
var jobs = Array<Job<ResultType>> {};
for (item in items) {
    jobs.add(Job<ResultType> { function: processFn, arguments: [item] });
}
await(jobs, MergeStrategy::last_wins);  // execute in parallel
for (job in jobs) { results.add(job.result()); }
```

**Key:** `Job<T>` generic, use `MergeStrategy::last_wins`, no nested await.

**Async:** `curl -H "task:''" -X POST http://localhost:8080/fn` or `PeriodicTask::set(...)`

**For production:** [references/concurrency.md](references/concurrency.md)

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

**For comprehensive testing guide:** See [references/testing.md](references/testing.md) for advanced Assert methods, setup/teardown patterns, and test organization.

## Common Pitfalls

| ❌ Wrong                                 | ✅ Correct                                   | Why                           |
| --------------------------------------- | ------------------------------------------- | ----------------------------- |
| `Array<T>::new()`                       | `Array<T>{}`                                | std types use `{}`            |
| `(*node)->field`                        | `node->field`                               | `->` already dereferences     |
| `@permission(public)`                   | `@permission("public")`                     | takes String                  |
| `@permission("api") fn getX()`          | `@expose @permission("api") fn getX()`      | API functions need @expose    |
| `for(i=0;i<n;i++) list.get(i)`          | `for (i, v in list)`                        | type inference                |
| `nodeList<City>`                        | `nodeList<node<City>>`                      | store refs, not objects       |
| `fn getX(): nodeList<...>`              | `fn getX(): Array<XxxView>`                 | API returns Array+View        |
| `nodeIndex.add(k, v)`                   | `nodeIndex.set(k, v)`                       | nodeIndex uses set/get        |
| `for(i, v in nullable_list)`            | `for(i, v in nullable_list?)`               | use `?` for nullable          |
| `fn doX(): void`                        | `fn doX()`                                  | no void type                  |
| `City{name: "X"}` (missing collections) | `City{name: "X", streets: nodeList<...>{}}` | init non-nullable collections |

### Acceptable Double-Bang Patterns

Using the non-null assertion operator (double-bang) is acceptable for global registry lookups where data is guaranteed to exist:
```gcl
var config = ConfigRegistry::getConfig(key)!!;  // ✅ OK — populated at init
```

## ABI & Database

**DEV MODE:** Delete deprecated fields. Reset `gcdata/` on schema changes. Add non-nullable fields → make nullable: `newField: int?`

```bash
rm -rf gcdata && greycat run import  # ⚠️ DELETES DATA - ask confirmation
```

Docs: https://doc.greycat.io/

## Full-Stack Development

Building React frontends with GreyCat backends?

**[references/frontend.md](references/frontend.md)** provides a comprehensive 1,013-line guide covering:
- @greycat/web SDK setup and configuration
- TypeScript type generation and integration
- Authentication and authorization patterns
- React Query integration and custom hooks
- Error handling and best practices

This is the most detailed reference in the skill package - start here for frontend development.

## Library References

Complete GCL type definitions and API documentation for all GreyCat libraries are available in the references directory.

**See [references/LIBRARIES.md](references/LIBRARIES.md)** for the complete catalog of std, ai, algebra, kafka, sql, s3, opcua, finance, powerflow, and useragent libraries.
