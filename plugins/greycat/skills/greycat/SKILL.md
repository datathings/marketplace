---
name: greycat
description: Build, run, and edit GreyCat projects. GreyCat is a statically-typed language plus runtime for graph-persistent, time-series-aware applications. Use when reading or writing `.gcl` source, when the user mentions GreyCat / project.gcl / nodeTime / nodeList / nodeIndex / nodeGeo / @expose / @library, or when the task involves running `greycat <command>`, deploying a project, or reasoning about gcdata/, lib/, files/, webroot/.
---

# GreyCat

GreyCat is **one language and one runtime in one binary**. A project lives in a directory rooted at `project.gcl`. The `greycat` binary compiles it, runs it, serves it as an HTTP server, manages users, and stores its state in `gcdata/`. There is no separate database, queue, or web server.

`.gcl` source files are organized into projects with a single entrypoint named `project.gcl`, whose `@library` / `@include` pragmas (which **must** appear in that file only) form the closure of analyzed modules. Compiled and run by the `greycat` runtime; statically analyzed by `greycat-analyzer`.

## Anti-hallucination rule

GreyCat is **not** Java, Rust, Kotlin, Python, or TypeScript. It has its own conventions and a small grammar. Before writing GCL by analogy to another language, check [`reference/idioms.md`](reference/idioms.md) — most "obvious" guesses are wrong (no `new`, no ternary, no `switch`, no `import`, `private` ≠ "hidden", `->` ≠ `.`, etc.).

When uncertain about a construct: read `lib/std/*.gcl` for real examples, then run `greycat run` against a minimal `project.gcl`. The runtime is the oracle.

## When to read which file

This file covers the 80% you need across language *and* tooling. Drill into a reference file when the task touches its area:

**Language:**

- **[reference/syntax.md](reference/syntax.md)** — Complete grammar reference: every statement, every expression form, operator precedence, literals (string substitution, time `'…'`, typed-suffix numbers).
- **[reference/types.md](reference/types.md)** — Type system in depth: nullability, narrowing, generic invariance, casting, inheritance, `is`/`as`.
- **[reference/stdlib.md](reference/stdlib.md)** — Built-in types by category (collections, node types, time/duration/geo, IO, HTTP, S3, Crypto). Method signatures.
- **[reference/annotations.md](reference/annotations.md)** — Every annotation (`@expose`, `@permission`, `@reserved`, `@volatile`, `@format`, `@test`, `@tag`) and every modifier (`private`, `static`, `abstract`, `native`). Doc-comment tags like `@param`.
- **[reference/idioms.md](reference/idioms.md)** — Idiomatic patterns and common pitfalls (no ternary, no `void`, no `::new()`, `function` opacity, `private` semantics, generic invariance).

**Tooling / project / runtime:**

- **[reference/project.md](reference/project.md)** — Project model: entrypoint, `@library` / `@include` resolution, `lib/<name>/` layout, FQN, multi-project workspaces.
- **[reference/cli.md](reference/cli.md)** — `greycat` CLI: every command (`run`, `serve`, `dev`, `build`, `test`, `install`, `codegen`, `user`, `backup`, `restore`, …), every option, the `.env` file.
- **[reference/analyzer.md](reference/analyzer.md)** — `greycat-analyzer` CLI: `lint`, `fmt`, LSP `server`, debug dumps. The pre-commit / definition-of-done tooling.
- **[reference/runtime.md](reference/runtime.md)** — What's alive in a running server: the graph store (`gcdata/`), workers and tasks, the HTTP server (JSON-RPC / path-RPC / `/files` / `webroot`), identity and permissions, the scheduler, backups, logging.
- **[reference/workflow.md](reference/workflow.md)** — Operational recipes: bootstrap a project, add an endpoint, add a persisted type, write tests, evolve schemas, generate SDKs, deploy.
- **[reference/webapp.md](reference/webapp.md)** — Bundling a webapp: `app/` sources + Vite/VitePlus config at the project root + bundle into `webroot/` + `greycat dev`. Calling `@expose`d endpoints from the browser.

## File anatomy

A `.gcl` module is a flat sequence of declarations and pragmas. No top-level expressions. No imports — visibility is governed by the project graph.

```gcl
@library("std", "1.2.3");          // pragma: depend on std at this version (project.gcl only)

/// Doc comment for the type.
type Point<T> extends Shape {      // generic, inheriting
    x: T;                          // attribute (terminated by ; or newline)
    y: T = 0;                      // attribute with init
    private label: String?;        // private attr = read-public, write-private

    fn distance(other: Point<T>): float {
        return sqrt((this.x - other.x) ^ 2 + (this.y - other.y) ^ 2);
    }

    static fn origin(): Point<int> {
        return Point<int> { x: 0, y: 0 };
    }
}

enum Color { red, green, blue }

var threshold: node<float?>;       // module-level var: must be node<T?>, nodeList<T>, nodeIndex<K, V>, nodeTime<T>, nodeGeo<T>

@expose
fn ping(): String {
    return "pong";
}
```

## Project anatomy

```
my-project/
├── project.gcl              # @library + @include pragmas — the ONLY file allowed to carry them
├── .env                     # optional GREYCAT_* config picked up at startup
├── bin/                     # `greycat install` populates with the pinned core binary
├── lib/                     # `greycat install` populates from @library pragmas
│   └── std/                 # the stdlib
├── src/                     # @include("src"); — your code
├── test/                    # @include("test"); — *_test.gcl stripped by `greycat build`
├── files/                   # served at /files/<user>/... — user uploads
├── gcdata/                  # graph storage. DO NOT COMMIT. Back this up.
└── webroot/                 # public static assets, served at /
```

Everything in `bin/`, `lib/`, `gcdata/`, and usually `files/` is gitignored. The source of truth is `project.gcl` + `src/` + (optional) `test/` + `webroot/`. See [reference/project.md](reference/project.md) and [reference/runtime.md](reference/runtime.md) for the role of each directory.

## CLI in one minute

```sh
greycat install     # download libraries and the pinned core binary from project.gcl
greycat serve       # build + run as long-lived HTTP server (port 8080 by default)
greycat dev         # serve + spawn a frontend watcher (vp/vite/--with=<cmd>)
greycat run [fn]    # build + run `fn` (default: `main`). One-shot.
greycat test        # build + run every @test function
greycat build       # produce project.gcp (strips *_test.gcl)
greycat codegen     # generate typed client SDKs (c/ts/python/rust/java)
greycat user list   # admin LMDB-backed user database
greycat backup      # snapshot gcdata/ into ./backup/
greycat restore <archive>
```

Options can be flags (`--name=value`) or env vars (`GREYCAT_NAME=value`). `greycat <command> -h` lists the options that apply, with their currently-resolved values. See [reference/cli.md](reference/cli.md) for the full table.

## Declarations

```gcl
type T {}                         // open user type
private type T {}                 // visible cross-module only via mod::T
abstract type T {}                // cannot be instantiated; methods may lack body
native type T {}                  // runtime-implemented

type Sub extends Base {}          // single inheritance
type G<T, U> {}                   // generics

enum E { a, b(1), "c-with-dash" } // entries optionally carry a value
fn name(p: T): R {}               // function
fn name<T>(p: T): T {}            // generic function
var globalName: T;                // module-level variable (must have type; node-tag only)
```

Each can be prefixed with `///` doc comments and annotations.

### Type bodies

A `type` body contains attributes and methods. There is **no constructor syntax** — instances are built with object-init expressions (see "Construction" below).

```gcl
type User {
    /// Doc on the attribute.
    id: int;
    name: String;
    private password_hash: String;   // outside ctor: read-public, write-forbidden
    static MAX_NAME_LEN: int = 64;   // static (class-level) attribute — readonly

    fn rename(new_name: String) {
        this.name = new_name;        // `this` is implicit in methods
    }

    static fn validate_name(name: String): bool {
        return name.size() <= User::MAX_NAME_LEN;
    }

    native fn hash_password();       // body provided by runtime
    abstract fn validate(): bool;    // requires `abstract type`; no body
}
```

Trailing `;` between members is optional but always safe. Methods can omit return type (means "returns nothing").

## Types

Everything is typed. Type references appear after `:` and inside generic brackets.

| Category                            | Names                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| Primitives                          | `bool`, `int` (i64), `float` (f64), `char`, `String`                             |
| Native containers                   | `Array<T>`, `Map<K, V>`, `Tuple<T, U>`, `Buffer`, `Table<T>`, `Tensor`           |
| Native node types (graph-persisted) | `node<T>`, `nodeTime<T>`, `nodeList<T>`, `nodeIndex<K, V>`, `nodeGeo<T>`         |
| Native value types                  | `time`, `duration`, `geo`, `function`, `field`, `type`, `any`, `null`            |
| User-defined                        | Anything declared with `type` / `enum`                                           |

### Nullability

A type without `?` is **non-null**; appending `?` makes it nullable.

```gcl
var a: int = 0;        // ok
var b: int = null;     // ERROR — int is non-null
var c: int? = null;    // ok
var d: String? = "hi"; // ok — non-null value assigns into nullable slot
```

Operations on a nullable value require the analyzer to see the null possibility eliminated, either by `== null` / `!= null` narrowing or by `!!` (force non-null, throws at runtime if null).

```gcl
fn use(x: String?) {
    if (x != null) {
        println(x.size());        // narrowed to String here
    }
    println(x!!.size());          // force; runtime error if x == null
    println(x?.size());           // optional chaining — propagates null
    println((x ?? "default").size());  // nullish coalescing
}
```

### Construction

GreyCat has **no `new` keyword** and no `::new()` convention. Build with object-init expressions:

```gcl
// Field form (named): works for any user type.
var u = User { id: 1, name: "alice", password_hash: "..." };

// Positional form: only for Array, Map, node, geo.
var arr = Array<int> { 1, 2, 3 };
var g = geo { 49.6, 6.1 };        // (lat, lng)

// Array literal sugar: [..] is sugar for Array<inferred>{..}.
var xs = [1, 2, 3];               // Array<int>

// Tuple sugar: (a, b) is sugar for Tuple<T, U>{a, b}.
var pair = (1, "hi");             // Tuple<int, String>

// Node literals wrap a payload:
var n = node<User> { User { id: 1, name: "alice", password_hash: "" } };
```

`Array<T>::new()` and `Map<K,V>::new()` do **not** exist. Always use the brace form.

## Member access

Three distinct operators — pick by what's on the left:

| Form                                    | Meaning                                                                                       | When                                                                                                                          |
| --------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `obj.field` / `obj.method()`            | Access the **value's own** field or method                                                    | Always for user types and value types                                                                                         |
| `n->field`                              | **Deref then access**: resolves the node payload, then `.field` on the result                 | Only on the stdlib node tags: `node<T>`, `nodeTime<T>`, `nodeIndex<K,V>`, `nodeList<T>`, `nodeGeo<T>`. User types cannot opt in.    |
| `Type::member` / `Module::Type::member` | **Static / namespaced** access                                                                | Static fields, static methods, enum entries, fully-qualified names                                                            |

```gcl
var u: User = expr;
u.name;                  // own field
u.rename("bob");         // own method

var nu: node<User> = expr;
nu.resolve();            // node<T>'s OWN method (the deref method)
nu->name;                // == nu.resolve().name — deref then field
nu->rename("bob");       // == nu.resolve().rename("bob")

User::MAX_NAME_LEN;        // static attr
User::validate_name("a");  // static fn
Color::red;                // enum entry
MathConstants::pi;         // module-qualified static (defined in std/core)
```

`->` is **not optional sugar for `.`** — it errors on any receiver other than a stdlib node tag.

### Optional chaining

`?` propagates null along a chain:

```gcl
obj?.field               // null if obj is null, else obj.field
n?->field                // null if n is null, else n->field
arr?[i]                  // null if arr is null, else arr[i]
```

## Statements

```gcl
var x = 1;                        // type inferred from rhs
var x: int = 1;                   // type annotated
var x: int;                       // no init (non-null types must be assigned before use)

if (cond) {} else if (cond) {} else {}
while (cond) {}
do {} while (cond);
for (var i = 0; i < n; i = i + 1) {}    // C-style; var declares the iterator

for (k, v in arr) {}                    // iterate Array (k=index), Map (k=key)
for (k, v in map) {}                    // unpack key/value pairs
for (t, v in node[from..to]) {}         // time-window query on nodeTime
for (k, v in idx) {}                    // iterate nodeIndex/nodeList
for (t, v in series[from..to] limit 100 skip 10) {}  // sampling clauses on series slice

return;          return expr;
throw error;
break;           continue;        breakpoint;    // breakpoint pauses the worker
try {} catch (e) {}                    // catch ident optional
at (targetTime) {}                          // time-aware-scope binding
```

There is **no ternary (`?:`)**, **no `switch`/`match`**, and **no `void` keyword**. A function with no `: T` return type "returns nothing"; calling it in an expression position is an error.

## Operators

```
Precedence (high to low):
  postfix:  . -> :: () []  ++  --  !!         // member, call, offset, increment
  prefix:   -  !  +  *  ++  --                // negation, deref-mul, increment
  ??                                          // nullish coalescing (highest binary)
  ^                                           // power
  * / %
  + -
  < <= > >=
  == !=
  is  as                                      // type test / cast
  &&
  ||
  =  ?=                                       // assignment (?= means "assign if null")
```

Notes:

- `??` binds tighter than `^` — `count ?? 0 > 0` parses as `(count ?? 0) > 0`.
- `is T` and `as T` take a **type**, not an expression. `is` narrows on the then-branch.
- `?=` is "assign only if LHS is null".
- `++` / `--` exist in both prefix and postfix forms.
- The unary `*x` is the deref-op (different in context from binary `*`).

## Literals

```gcl
true   false   null   this

42                                  // int
3.14                                // float
1.5e-10                             // float (scientific)
1_000_000                           // int (underscores ignored)
42_time     42time                  // time literal (suffix form)
1.79e+308_f                         // float (typed suffix)
60s     2hour_42ms                  // duration (compound suffix)

'a'                                 // char
'\n'   'é'   '\xff'                 // char escapes
'2025-05-22T16:47:42Z'              // time literal (ISO 8601 inside '...')

"hello"
"hello ${name}, total = ${count + 1}"   // template substitution
"line1\nline2"                      // escapes
```

## Annotations

Decl-level annotations modify the declaration that follows. Module-level pragmas (`@library`, `@include`, `@permission`, `@role`) appear standalone with a trailing `;`.

```gcl
@expose                             // expose this fn as an HTTP/RPC endpoint
@expose("renamed")                  // expose under a different path
@permission("admin")                // require this permission
@tag("openapi", "mcp")              // include in OpenAPI spec and MCP tool list
@reserved                           // server-only impl; not callable from GCL
fn admin_op() {}

@test                               // marks a test fn; discovered by `greycat test`
fn test_my_thing() {}

@volatile                           // type cannot be persisted to graph storage
type Cache {}

@format(DurationUnit::milliseconds) // serialization hint (ms instead of µs)
ttl: duration?;
```

`@deref` and `@iterable` also exist as type-shape annotations, but they are tooling hints on the stdlib node tags only — adding them to a user type does NOT opt the type into `->` or `for-in`.

Module pragmas (must terminate with `;`, **must appear in `project.gcl`**):

```gcl
@library("std", "1.2.3");                     // bring std at this version into scope
@include("models");                           // include all .gcl under ./models/
@permission("audit", "read audit logs");      // declare a permission
@role("auditor", "public", "audit");          // declare a role
```

See [reference/annotations.md](reference/annotations.md) for the full list and semantics.

## Modifiers

| Modifier   | On decl                                                                | On attribute                                                  | On method                                               |
| ---------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------- |
| `private`  | Cross-module access requires FQN (`mod::T`). Same-module unrestricted. | Read-public, write-private (only the constructor can assign). | Cross-module call requires FQN.                         |
| `static`   | —                                                                      | Class-level attribute (one shared value).                     | Class-level fn; access via `Type::name()`.              |
| `abstract` | Type cannot be instantiated.                                           | —                                                             | Method has no body; concrete subtypes must provide one. |
| `native`   | Type/method body is implemented by the runtime.                        | —                                                             | Body must be absent.                                    |

`private` is **not** "hidden." See [reference/annotations.md](reference/annotations.md) for the full semantics and a worked example.

## Project structure

Every project has a single entrypoint named `project.gcl`. All `@library` and `@include` pragmas **must** appear in this file.

```gcl
// project.gcl
@library("std", "1.2.3");           // resolved against <project>/lib/std/ or GreyCat home
@include("src");                    // recursively loads .gcl files under <project>/src/
```

- `@library("name", "version")` resolves to `<project>/lib/<name>/`; the `std` library may also resolve under the GreyCat install's home.
- `@include("relative/dir")` resolves to `<project>/relative/dir/` and recursively loads `.gcl` there.
- The closure is computed transitively (each loaded module's pragmas are followed).
- **Never flat-walk a directory for `.gcl` files** — always start from the entrypoint.

See [reference/project.md](reference/project.md) for cross-module visibility rules and FQN resolution, and [reference/workflow.md](reference/workflow.md) for the bootstrap-to-deploy flow.

## Quick patterns

### HTTP endpoint

```gcl
@expose
@tag("openapi")
fn add(a: int, b: int): int {
    return a + b;
}
```

Reachable at `POST /<module>::add` (path-RPC) and `"<module>.add"` via JSON-RPC. Without `@permission`, requires the `api` permission. See [reference/runtime.md](reference/runtime.md) for the request lifecycle.

### Time-series query

```gcl
fn readings(sensor: nodeTime<float>, from: time, to: time) {
    for (t, v in sensor[from..to]) {
        println("${t}: ${v}");
    }
}
```

### Graph-persistent record

```gcl
type Sensor {
    name: String;
    measurements: nodeTime<float>;
}

fn record(s: Sensor, v: float) {
    s.measurements.setAt(time::now(), v);
}
```

### Null-flow + narrowing

```gcl
fn label(u: User?): String {
    if (u == null) {
        return "anonymous";
    }
    return u.name;                  // narrowed: u: User here
}
```

### Scheduled background task

```gcl
fn nightly_backup() {
    Runtime::backup_delta();
}

@expose
@permission("admin")
fn install_schedule() {
    Scheduler::add(
        nightly_backup,
        DailyPeriodicity { hour: 2 },
        null,
    );
}
```

## Common pitfalls

The most-bitten gotchas (full list in [reference/idioms.md](reference/idioms.md)):

1. **No ternary.** Write `if (c) { return a; } return b;`, not `c ? a : b`.
2. **No `void` keyword.** Omit the `: T` return-type clause.
3. **No `::new()`.** Always `Type {}` or `Type<G> {}`. Unless an explicit `Type { static fn new(): Type { /*...*/ } }` exists.
4. **`function` parameters are opaque.** A value typed `function` is runtime-checked, not compile-checked.
5. **Generics are invariantly typed.** `Array<int?>` is not assignable to `Array<int>`. Same for `Map`, `nodeIndex`, etc.
6. **`->` is reserved for stdlib node tags.** Only `node<T>`, `nodeTime<T>`, `nodeIndex<K,V>`, `nodeList<T>`, `nodeGeo<T>` support `->`. User types cannot opt in — use `.` instead.
7. **`private` ≠ "hidden."** A `private type` is still visible across modules via its fully-qualified name; only bare-name lookup is blocked. A `private attr` is read-public, write-private. Never gate same-module access on `private`.
8. **No imports.** Visibility comes from the project graph (`@library` / `@include`), not from `import`/`use` statements.
9. **`as` is unchecked at runtime.** The runtime drops `as T` entirely; the analyzer's static check is the only safety net.
10. **Trailing `;` after `}` is lint-rejected.** A method/function body's closing brace stands alone — `greycat-analyzer` fires `warning[redundant-semicolon]` (auto-fixable) although `greycat build` accepts it.
11. **Don't commit `gcdata/`, `bin/`, or `lib/`.** They are runtime / install state, not source. `gcdata/` is the durable application state — back it up but never check it in.

## Verifying syntax assumptions

When uncertain whether a construct is valid GreyCat:

1. **Run `greycat-analyzer lint`** — fastest oracle, catches shape drift the runtime accepts silently (unused locals, non-exhaustive enum chains, redundant null-checks, `->` on non-deref receivers, …). See [reference/analyzer.md](reference/analyzer.md).
2. Search the stdlib (`lib/std/*.gcl`) for real examples of the construct.
3. Run `greycat run` against a minimal `project.gcl` — the runtime is the oracle for valid programs.
After any non-trivial `.gcl` edit, run `greycat-analyzer fmt --mode=check` + `greycat-analyzer lint` as the definition of done — `greycat build` happily produces a `.gcp` from code that still has warnings or formatting drift.
4. If still unsure, ask. Do not assume by analogy to TypeScript / Rust / Kotlin / Java — GreyCat has its own conventions (no `new`, no ternary, `->` vs `.`, `private` semantics).
