# Annotations and modifiers

Every annotation, every modifier, with semantics and where each one is valid.

## Contents

- Module pragmas (`@library`, `@include`, `@permission`, `@role`)
- Declaration annotations (`@expose`, `@permission`, `@reserved`, `@test`, `@tag`)
- Type-shape annotations (`@volatile`)
- Attribute annotations (`@format`)
- Modifiers (`private`, `static`, `abstract`, `native`)
- `private` semantics in depth — the most-misused modifier

`@deref` and `@iterable` are not documented here on purpose: they are tooling hints attached to the stdlib node tags (`node<T>`, `Array<T>`, etc.) so the analyzer can document the runtime-hardcoded `->` and `for-in` behavior. **Adding them to a user type does not grant those semantics** — the runtime ignores the tags and only treats its hard-coded set of types as iterable / dereferenceable.

## Syntax

```gcl
@name                    // bare
@name("string")          // single string arg
@name("a", "b", "c")     // multiple string args
@name(123)               // numeric arg
@name(MyEnum::variant)   // enum arg (e.g. @format)
```

Annotations precede the declaration they modify. Multiple annotations can stack (one per line is conventional):

```gcl
@expose
@permission("admin")
fn do_admin_things() {}
```

Module pragmas appear at module scope and **must** terminate with `;`:

```gcl
@library("std", "1.2.3");
```

## Module pragmas

### `@library("name", "version")`

Declares a dependency on a library named `name` at version `version`. Resolution order:

1. `<project_dir>/lib/<name>/` — vendored copy.
2. For `name == "std"` only: the GreyCat home's `lib/std/` — the runtime install.

**MUST** appear in `project.gcl` (the entrypoint).

```gcl
@library("std", "1.2.3");
@library("mylib", "0.1.0");
```

### `@include("path")`

Recursively loads every `.gcl` file under `<project_dir>/<path>/`.

```gcl
@include("src");
@include("models/user");
```

Cycles are detected and stopped (loading order does not affect resolution — visibility is whole-graph).

### `@permission("name", "description")`

Declares a named permission that can be used by `@permission` annotations on functions.

```gcl
@permission("public", "Default permission, attached to anonymous callers");
@permission("admin", "Full administrative access");
@permission("api", "Access to exposed functions");
@permission("debug", "Low-level graph manipulation");
```

The std library declares the four canonical permissions above (`public`, `admin`, `api`, `debug`). User code may add custom ones.

### `@role("name", "perm1", "perm2", ...)`

Declares a named role as a set of permissions.

```gcl
@role("public", "public");
@role("admin", "public", "admin", "api", "debug");
@role("user", "public", "api");
```

## Declaration annotations

### `@expose` / `@expose("path")`

Makes a function reachable via HTTP / RPC. Without args, the path is the function's qualified name. With a string arg, the function is exposed at that path.

```gcl
@expose
fn list_users(): Array<User> {}
// → POST /<module>::list_users

@expose("api/users")
fn list_users_renamed(): Array<User> {}
// → POST /api/users

@expose("initialize")
fn mcp_initialize(params: McpInitializeParams): McpInitializeResult {}
// stdlib uses this for the MCP protocol
```

A function with `@expose` is reachable from outside the runtime. It is alive even if no other code calls it — the `unused-fn` lint accounts for this.

### `@permission("name")`

Gates the function on the named permission. The caller must hold that permission or the call is rejected with a 403.

```gcl
@expose
@permission("admin")
fn restart_workers() {}                  // admins only

@expose
@permission("api")
fn current_user(): Identity {}            // any authenticated api caller

@expose
@permission("public")
fn ping(): String {}                      // anyone, including anonymous
```

Without `@permission`, an `@expose` function defaults to requiring `api` (authenticated callers).

### `@reserved`

Hint that the function has a **server-side implementation only** — no GCL-level body the VM can dispatch to. Calls go through the server's HTTP/RPC path; GCL code cannot invoke a `@reserved` function directly. Typically paired with `@expose` + `native` on stdlib introspection / admin endpoints.

```gcl
@expose
@reserved
@permission("admin")
native fn cancel(task_id: int): bool;   // body lives in the server binary
```

Slated for removal in a future release — prefer non-reserved alternatives where they exist.

### `@tag("name", ...)`

Attaches one or more named tags to an `@expose`d function. Tags drive runtime discovery for protocols layered on top of `@expose`:

```gcl
@expose
@tag("openapi")              // include in the OpenAPI v3 spec
fn list_users(): Array<User> {}

@expose
@tag("mcp", "openapi")       // also expose as an MCP tool
fn echo(msg: String): String {
    return msg;
}
```

Recognized tag names today:

| Tag       | Behavior                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------- |
| `openapi` | Function appears in the OpenAPI v3 document returned by `OpenApi::v3()`.                       |
| `mcp`     | Function appears in MCP `tools/list` and is callable via MCP `tools/call`.                     |

Tags are additive — stacking `@tag("mcp", "openapi")` is equivalent to `@tag("mcp")` + `@tag("openapi")`. Unknown tag names parse fine and are exposed reflectively, but have no built-in semantics.

`@tag` has no effect on functions without `@expose` — non-exposed functions are not reachable over HTTP/RPC regardless of tags.

### `@test`

Marks the function as a test. `greycat test` discovers `@test` functions, runs them, and reports pass / fail counts. Tests typically use the `Assert::*` helpers from stdlib.

```gcl
@test
fn test_double() {
    Assert::equals(double(2), 4);
}

fn double(x: int): int { return x * 2; }
```

**Convention: keep tests in `*_test.gcl` files.** `greycat build` (which produces the production `.gcp` package) strips every module whose filename ends in `_test.gcl`, so tests do not ship to production. A common project layout is:

```gcl
// project.gcl
@library("std", "1.2.3");
@include("src");
@include("test");                 // contains *_test.gcl files
```

Files outside `_test.gcl` are packaged normally.

#### `setup()` / `teardown()` lifecycle

A module may declare module-level `fn setup()` and `fn teardown()`. `greycat test` runs `setup()` once before the module's tests, then each `@test` in source-definition order, then `teardown()` once at the end. **Tests share module state** — anything `setup()` writes to module-level node variables is visible (and mutable) across every test in the module.

```gcl
private var counter: node<int?>;

fn setup()    { counter.set(100); }
fn teardown() { println("final: ${counter.resolve()}"); }

@test fn test_a() { Assert::equals(counter.resolve(), 100); counter.set(101); }
@test fn test_b() { Assert::equals(counter.resolve(), 101); }       // sees test_a's write
```

`greycat test` reports the lifecycle hooks alongside the tests (each appears in the output with `ok (Nus)`), so the count of "tests success" includes setup/teardown invocations.

Run a single test by name: `greycat test test_a`. Module qualification (e.g. `mymod::test_a`) also works.

## Type-shape annotations

These annotations modify how the type behaves in the language semantics.

### `@volatile`

Declares that instances of this type are **transient** — they cannot be stored in the graph. Mostly used for runtime / introspection types whose values represent live worker state.

```gcl
@volatile
type Log {
    level: LogLevel;
    time: time;
    /* ... */
}
```

Attempting to persist a `@volatile` value (writing it into a `node<T>` or a graph-persisted attribute) is a runtime error.

## Attribute annotations

### `@format(unit)`

Serialization hint for `duration` / `time` attributes. Two forms:

**Enum form** — encode in a given unit instead of the default (microseconds):

```gcl
type Config {
    @format(DurationUnit::milliseconds)
    ttl: duration;
    // JSON output: {"ttl": 1000} instead of {"ttl": 1000000}
}
```

**String form** — strftime pattern used by `CsvReader<T>` when parsing date columns:

```gcl
type Row {
    @format("%d/%m/%y %H:%M") t: time;
}
```

Without `@format` on a `time` field, CSV parsing defaults to ISO 8601 / epoch milliseconds. Used on stdlib types like `McpTask.pollInterval`. Custom formats may be analyzer-version-dependent.

## Modifiers

Modifiers precede the declaration keyword (or, for attributes, the attribute name). Multiple modifiers can stack in any order.

| Modifier   | On `type`                                                        | On `fn` / method                                   | On attribute                   |
| ---------- | ---------------------------------------------------------------- | -------------------------------------------------- | ------------------------------ |
| `private`  | Visible cross-module only via FQN                                | Callable cross-module only via FQN                 | Read-public, write-private     |
| `static`   | —                                                                | Class-level (no `this`); accessed via `Type::name` | Class-level (one shared value) |
| `abstract` | Cannot be instantiated; subtypes must implement abstract methods | No body; subtypes must provide one                 | —                              |
| `native`   | Runtime-implemented; no body                                     | No body                                            | —                              |

### `private` — full semantics

This is the single most-misunderstood modifier. **It is NOT "hidden."**

#### `private` on a declaration

`private type Foo {}`, `private fn bar() {}`, `private enum Baz {}`, `private var globalThing: T;`

- **Cross-module:** bare-name lookup is blocked. You must use the fully-qualified name: `module::Foo`, `module::bar`.
- **Same-module:** identical to a public decl. No restriction whatsoever.

```gcl
// in module: mymod.gcl
private type Internal {}
fn build(): Internal { return Internal {}; }   // OK — same module

// in another module:
var x: Internal = ...;         // ERROR — bare name doesn't resolve
var x: mymod::Internal = ...;  // OK — FQN works
```

#### `private` on an attribute

`type Foo { private attr: int;}`

- **Reading:** unrestricted. Anyone, anywhere, can do `foo.attr`.
- **Writing:** only the type's **constructor** (object-init expression) can write the value.
  - `Foo { attr: 1 }` — OK.
  - `foo.attr = 2;` — ERROR (regardless of which module).
  - Methods inside `Foo` cannot reassign `attr` either. The init expression is the only write site.

```gcl
type User {
    name: String;
    private password_hash: String;
}

var u = User { name: "alice", password_hash: "$2a$..." };   // OK
println(u.password_hash);                                   // OK — read is public
u.password_hash = "new_hash";                               // ERROR — write is private
```

#### What `private` is NOT

- It is **not** "Java private" (same-class only).
- It is **not** "Rust pub(crate)" (since same-module access is unrestricted).
- It is **not** a way to hide member-shape from same-module callers.
- It is **not** a way to gate inherited members.

When gating logic on `is_private`, only two checks are legitimate:

1. **Resolver bare-name cross-module lookup:** "did this name resolve via a bare ident?" → if yes and the decl is private, error.
2. **Assignment LHS member-access on a private attr from outside the constructor:** error.

Anything else (filtering members from `type_members`, hiding from supertype walks, blocking from same-module use sites) is a bug.

### `static`

Class-level. A `static` attribute is a single shared value across all instances and is **readonly** — assignment is rejected. A `static fn` has no `this` and is called via `Type::name(args)`:

```gcl
type MathLimits {
    static max_iterations: int = 10000;        // readonly: cannot be reassigned
    static fn check(n: int): bool {
        return n < MathLimits::max_iterations; // OK: read
    }
}

println(MathLimits::max_iterations);           // OK: read
MathLimits::max_iterations = 5000;             // ERROR: static attributes are readonly

if (MathLimits::check(42)) { /* ... */ }
```

Static methods cannot access instance attributes (no `this`).

### `abstract`

A type marked `abstract` cannot be instantiated directly. Within an `abstract type`, methods may be declared without a body (`abstract fn foo();`) — concrete subtypes are required to provide one.

```gcl
abstract type Shape {
    name: String;
    abstract fn area(): float;
    fn render() { println(this.name); }   // can have concrete methods too
}

type Circle extends Shape {
    radius: float;
    fn area(): float { return 3.14159 * this.radius ^ 2; }
}

// Shape { ... }   // ERROR — cannot instantiate abstract type
// Circle { ... }   // OK
```

Abstract types are how GreyCat models "interface-like" contracts. There is no separate `interface` keyword.

### `native`

The body is implemented by the runtime (in C). `native` type/functions are reserved to library authors.

```gcl
native type String {
    native fn size(): int;
    native fn lowercase(): String;
}
```

A `native fn` has no body in `.gcl`.

## Annotation ordering convention

```gcl
/// Doc comment first.
@expose
@permission("admin")
modifiers fn name(...)
```

- `///` doc comments come first.
- Annotations follow, one per line (idiomatic; same-line stacking parses fine but is less readable).
- Modifiers come last before the declaration keyword.

The formatter normalizes this order.

## Lint visibility

Several lint rules are tied to annotations:

- `@expose` keeps a decl alive — the `unused-fn` lint doesn't fire on exposed functions.
- `@permission("name")` without a corresponding `@permission(...)` module pragma declaring that name fires an unknown-permission lint.

Run `greycat-analyzer lint --list-rules` to see the current set.
