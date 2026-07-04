# Idioms and pitfalls

The patterns that keep GreyCat code correct, and the mistakes that an agent familiar with TS / Rust / Kotlin / Java / Python will instinctively make. Skim this before writing significant `.gcl`.

## Contents

- The big "do not write this" list
- Member access — when to use `.` vs `->` vs `::`
- Construction patterns
- Working with nullability
- Iterating collections and nodes
- Calling a `function`-typed value
- Generic invariance — when assignments don't compose
- Casting safely with `is` then `as`
- HTTP / `@expose` patterns
- Time-series patterns
- Tests with `Assert`

## The big "do not write this" list

| Wrong                                               | Right                                        | Why                                     |
| --------------------------------------------------- | -------------------------------------------- | --------------------------------------- |
| `cond ? a : b`                                      | `if (cond) { a } else { b }` (statement)     | No ternary.                             |
| `void` return type                                  | Omit `: T` clause                            | No `void` keyword.                      |
| `new Type(...)`                                     | `Type { ... }`                               | No `new` keyword.                       |
| `Type::new(...)`                                    | `Type { ... }`                               | No `::new()` convention.                |
| `Array::of(1, 2)`                                   | `[1, 2]` or `Array<int> { 1, 2 }`            | No `of` helper.                         |
| `import foo from "bar"`                             | Add `@library` / `@include` to `project.gcl` | No import statements.                   |
| `switch (x) { case … }`                             | Chained `if`, or `type::enum_offset(x)`      | No `switch` / `match`.                  |
| `interface Foo { ... }`                             | `abstract type Foo { ... }`                  | No `interface` keyword.                 |
| `let x = ...`                                       | `var x = ...`                                | Only `var`.                             |
| `const PI = 3.14`                                   | `static pi: float = 3.14;` inside a type     | No `const`; use `static` attributes.    |
| `T extends Bound` generic                           | `<T>` (no bounds)                            | Generics are unbounded.                 |
| `(a, b, c)` triple                                  | `Tuple<Tuple<A, B>, C>` or a custom type     | Only 2-tuples.                          |
| `obj?.field?.method?.()` JS-style                   | `obj?.field?.method()`                       | Method call doesn't have an extra `?.`. |
| `} ;` after a fn body                               | `}` (no trailing semicolon)                  | Lint flags this.                        |
| `private inside_only: int;` thinking "Java private" | See `reference/annotations.md` § `private`   | `private` ≠ hidden.                     |
| `idx.set(k, n)` where `n = idx.get(k)`              | mutate `n`'s fields in place; `set` only NEW nodes | Re-attaching a live node throws "object is already attached". |

## Member access — when to use `.` vs `->` vs `::`

Rule of thumb:

- **`.`** — accessing a value's **own** field or method.
- **`->`** — the value is a `@deref`-tagged tag (node family), and you want the **inner** payload.
- **`::`** — static / namespaced access (static methods/attrs, enum entries, FQNs).

```gcl
type User { name: String; private password: String; }

fn examples(u: User, n: node<User>) {
    u.name;                    // own field
    u.name.size();             // own method on String

    n.resolve();               // n's OWN method (the deref method)
    n->name;                   // == n.resolve().name — payload field
    n->password;               // ERROR (different reason — read-public still works
                               // here; just illustrating chain)
    User::counter;             // static (if declared)
    time::now();               // stdlib types are at top level — no `std::` prefix
}
```

If you find yourself writing `n.name` on a `node<User>`, it likely doesn't type-check — the node's own surface has only `resolve()` / `set()` / etc. You need `->` or to `.resolve()` first.

## Construction patterns

### Named-field form (most common)

```gcl
var u = User { id: 1, name: "alice", password_hash: "" };
```

- Order doesn't matter.
- Missing nullable / default-having fields are filled in.
- Missing non-null fields without defaults are an error.

### Positional form

Only supported for: `Array`, `Map`, `node`, `geo`.

```gcl
var t = Tuple<int, String> { 42, "hi" }; // ERROR: use `x: 42` and `y: "hi"`
var p = Point<int> { 0, 0 };             // ERROR: use attribute names `x: 0, y: 0`
var a = Array<int> { 0, 1, 2 };          // OK
var m = Map<String, int> { "a": 0, "b": 1 }; // OK
var n = node<String> { "hello" };        // OK
var g = geo { 49.6, 6.1 };               // OK — (lat, lng)
```

### Array literal

```gcl
var xs = [1, 2, 3];           // Array<int>  (inferred from elements)
var grid = [[1, 2], [3, 4]];  // Array<Array<int>>  (inferred when no `null`)
var empty = Array<float> {};  // empty: must construct explicitly — no element to infer from
```

**Caveat — annotation enforcement on `var`.** The runtime currently drops the type decorator on `var` declarations, so a mismatched annotation like `var x: Array<int> = ["a"];` will not be rejected at runtime. The lang is stricter but is not yet wired into the runtime. To get reliable type checking on collections, prefer **function parameters** (where the lang always enforces):

```gcl
fn take_ints(_: Array<int>) {}

take_ints([1, 2, 3]);         // OK
take_ints([1, "x"]);          // lang ERROR — literal infers as Array<any?>, not Array<int>
```

### Tuple literal

```gcl
var pair = (1, "hi");               // Tuple<any?, any?>
var pair2: Tuple<any?, any?> = (1, "hi");
pair.x;   pair.y;
```

Tuple literal are always created as Tuple<any?, any?> and do not infer generic type based on x and y values

### Node-tag literals

```gcl
var n = node<User> { User { id: 1, name: "alice", password_hash: "" } };
```

Node tags wrap a payload of their generic type. Use `node<T> { payload }` to create one in user code (the runtime usually provides them via graph attributes, so writing these by hand is rare).

## Working with nullability

### Three ways to use a nullable value

```gcl
fn options(x: String?) {
    // 1. Narrow with a check
    if (x != null) {
        println(x.size());        // x: String here
    }

    // 2. Force non-null (runtime check, throws if null)
    println(x!!.size());

    // 3. Coalesce
    println((x ?? "default").size());
    println(x?.size());           // optional chain — result is int? (null if x was null)
}
```

### `?=` for "assign only if null"

```gcl
var cache: Map<String, int>?;
cache ?= Map<String, int> { };   // create the map iff cache is currently null
```

Common pattern for lazy initialization of nullable attributes.

### Null in collections

`Array<int>` cannot contain null; `Array<int?>` can. They are unrelated types — `Array<int>` does **not** flow into `Array<int?>` (generic invariance).

```gcl
var ints: Array<int> = [1, 2, 3];
var maybe_ints: Array<int?>;
maybe_ints = ints;                // ERROR — invariance
```

If you need a "possibly-null" collection, declare it `Array<T?>` upfront.

## Iterating collections and nodes

```gcl
// Array
for (i, v in arr) {}
for (i: int, v: any? in arr) {}

// Map (k, v)
for (k, v in m) {}

// nodeTime — full series
for (t, v in series) {}

// nodeTime — time window
for (t, v in series[from..to]) {}

// nodeList / nodeIndex — same shape
for (i, v in nodelist[0..1000]) {}
for (k, v in idx) {}

// Explicit math interval (rare; for inclusive/exclusive control)
for (t, v in series]from..to]) {}
for (t, v in series[from..to[) {}
```

## Calling a `function`-typed value

A `function`-typed parameter is opaque — the signature it was assigned from is gone, so calls through it are runtime-checked:

```gcl
fn run(callback: function): int {
    var r = callback();         // accepted; runtime throws if signature mismatches
    return r as int;
}
```

Values held in a local var keep their structural signature, so calls through them are statically checked:

```gcl
var f = fn(a: int): int { return a + 1; };
f(3.14);                                    // ERROR: float not assignable to int
var g: function = fn(a: int): int { ... };  // explicit `function` annotation forgets the signature
g(3.14);                                    // accepted; runtime-checked
```

Annotating a local as `: function` is legal (Lambda → function is assignable) but discards the static checks you'd otherwise get — drop the annotation when you want the signature preserved.

## Generic invariance — when assignments don't compose

`Foo<X>` is assignable to `Foo<Y>` only when `X` is the same type as `Y`, **with one exception**: `any` is the universal top, so `Foo<T>` flows into `Foo<any>`. Other width-changing substitutions are forbidden:

```gcl
var ints: Array<int> = [1, 2];

var anys: Array<any> = ints;       // OK — `any` is the invariance exception
var nulls: Array<int?> = ints;     // ERROR — `int?` is not `any`; invariance applies

fn take(xs: Array<int?>) {}
take(ints);                        // ERROR
take([1, 2, 3]);                   // ERROR — literal infers as Array<int>, not Array<int?>
```

Raw-form node tags (`nodeTime` without `<T>`) are sugar for `nodeTime<any?>`, so `Array<nodeTime<float>>` flows into `Array<nodeTime>`.

If you need to call a function that takes `Array<int?>` and you have `Array<int>`, you must either:

- Annotate the literal at construction site (`var xs = Array<int?> { 1, 2, 3 };`), or
- Copy element-by-element through a fresh `Array<int?>`, or
- Refactor the function signature.

Casting (`ints as Array<int?>`) does **not** convert the underlying container — it's a static reinterpret and may break at runtime.

## Casting safely with `is` then `as`

`is` narrows; `as` does not check at runtime. The safe pattern is to test first:

```gcl
fn process(thing: any?) {
    if (thing is User) {
        // thing: User here (no need to `as` it)
        println(thing.name);
    } else if (thing is Order) {
        println(thing.total);
    } else if (thing is String) {
        println(thing.size());
    } else {
        println("unknown: ${type::of(thing)}");
    }
}
```

A bare `thing as User` without an `is` check is allowed but will behave unpredictably if the value isn't actually a `User` (the runtime does no enforcement; subsequent member access reads through to garbage). Reserve unchecked `as` for cases where the type is provably known by external invariant.

## HTTP / `@expose` patterns

### Permission defaults — authenticated by default, public is rare

A bare `@expose` already requires the `api` permission, which is exactly what you want for almost every endpoint. **Do not** add `@permission("public")` reflexively to make a frontend work without login — that hands anonymous callers on the network the same write access as an authenticated user. `@permission("public")` is reserved for the small set of endpoints that genuinely *must* serve anonymous traffic (`Identity::login`, a no-auth health probe). When you're unsure, leave the default in place and have the frontend authenticate via `Identity::login` first. Adding `@permission("public")` to a new endpoint is a judgment call that should be made by the project owner, not the agent.

### Authenticated endpoint (the default)

```gcl
@expose
fn ping(): String {
    return "pong";
}
```

No `@permission` clause — calls require a valid session. The browser obtains one via `Identity::login` (cookie) or `greycat token --user=<id>` (header / query string). See [runtime.md § Identity and permissions](runtime.md).

### Per-user data (scope by caller)

```gcl
var todos: nodeIndex<int, nodeIndex<String, Todo>>;     // keyed by user id

@expose
fn my_todos(): Array<Todo> {
    var mine = todos.get(Identity::current_id());
    var out = Array<Todo> {};
    if (mine == null) {
        return out;
    }
    for (_, v in mine) {
        out.add(v);
    }
    return out;
}
```

`Identity::current_id()` returns the authenticated caller. Don't mix users' data in a single flat collection unless every endpoint that touches it filters explicitly.

### Permission-gated endpoint

```gcl
@expose
@permission("admin")
fn restart() {
    Runtime::backup_delta();
}
```

Custom permissions declared in `project.gcl` (`@permission("audit", "read audit logs");`) compose the same way.

### Anonymous endpoint — only when the user asks for it

```gcl
@expose
@permission("public")
fn current_id(): int {
    return Identity::current_id();         // 0 for anonymous
}
```

`@permission("public")` opens the function to anonymous callers. Reach for it **only** when the user has explicitly described an anonymous-access requirement; otherwise default to the authenticated form above.

### Typed input / output

`@expose` functions can take typed parameters and return typed results. The runtime serializes both sides via JSON (or GCB for typed callers).

`login` is one of the rare cases where `@permission("public")` is correct — anonymous callers need to reach it to obtain a session.

```gcl
type LoginRequest { username: String; password: String; }
type LoginResponse { token: String; expires_at: time; }

@expose
@permission("public")
fn login(req: LoginRequest): LoginResponse {
    var tok = Identity::login(req.username, req.password);
    return LoginResponse { token: tok, expires_at: time::now() + 24hour };
}
```

## Time-series patterns

### Recording a measurement

```gcl
fn record(sensor: nodeTime<float>, value: float) {
    sensor.setAt(time::now(), value);
}
```

### Reading the latest value

```gcl
fn latest(sensor: nodeTime<float>): float? {
    return sensor.last();        // null if empty
}
```

### Querying a window

```gcl
fn window_avg(sensor: nodeTime<float>, from: time, to: time): float {
    var sum: float = 0.0;
    var n: int = 0;
    for (t, v in sensor[from..to]) {
        sum = sum + v;
        n = n + 1;
    }
    if (n == 0) {
        return 0.0;
    }
    return sum / n;              // careful: float division semantics; this is fine since both operands are float
}
```

## Working with the graph

### Node-backed records

```gcl
type Account {
    id: int;
    balance: float;
    history: nodeTime<float>;     // graph-persisted timeseries
}

fn deposit(acc: Account, amount: float) {
    acc.balance = acc.balance + amount;
    acc.history.setAt(time::now(), acc.balance);
}
```

### Sorted index

```gcl
fn lookup(idx: nodeIndex<String, Account>, name: String): Account? {
    return idx.get(name);
}

fn put(idx: nodeIndex<String, Account>, acc: Account) {
    idx.set("${acc.id}", acc);
}
```

### Mutating a persisted node

A value from `idx.get(k)`, `n.resolve()`, or a `for ... in` over a `nodeIndex` is a
**live graph node**: assign to its fields and the change persists on commit. `set`
*inserts* a node under a key; calling it on a node you just fetched re-attaches an
already-attached node and throws `object is already attached to another node`.

```gcl
type Counter { hits: int; }

// idx: nodeIndex<String, Counter>
fn bump(idx: nodeIndex<String, Counter>, k: String) {
    var existing = idx.get(k);
    var c = existing ?? Counter { hits: 0 };
    c.hits = c.hits + 1;             // live node: persists on commit, no set
    if (existing == null) {
        idx.set(k, c);               // brand-new node: attach exactly once
    }
}
```

`set` is for a brand-new node, or for replacing a key with a freshly-built object.
Re-setting a fetched node never fails on the first attach nor on a fresh-process
`get`->`set` (the node loads detached), so the error stays hidden until the node is
resident in a long-running server: it surfaces in production, not one-shot local runs.

## Tests with `Assert`

`Assert::*` static methods are how stdlib + projects write tests. They throw on failure.

```gcl
@test
fn test_double() {
    Assert::equals(double(2), 4);
    Assert::equals(double(0), 0);
    Assert::equals(double(-3), -6);
}

@test
fn test_user_label() {
    var u = User { id: 1, name: "alice", password_hash: "" };
    Assert::equals(label(u), "alice");

    var anon: User? = null;
    Assert::equals(label(anon), "anonymous");
}

fn double(x: int): int { return x * 2; }
fn label(u: User?): String {
    if (u == null) { return "anonymous"; }
    return u.name;
}
```

Test runners discover test functions by `@test` annotation. Run with `greycat test` (exact CLI flag depends on the runtime version).

## Strings and templates

```gcl
var name = "alice";
var count = 5;

var s1 = "hello ${name}";                       // simple substitution
var s2 = "you have ${count} items";
var s3 = "math: ${count + 1}";                  // expression in ${...}
var s4 = "nested: ${"${count}!"}";              // works (interior string parses)
var s5 = "literal dollar: \$5";                 // escape with backslash

// String methods
"hello".size();                                  // 5
"hello".contains("ell");                         // true
"hello".startsWith("he");                        // true
"hello".uppercase();                             // "HELLO"
"hello".replace("l", "L");                       // "heLLo"
"a,b,c".split(',');                              // ["a", "b", "c"]
"  hi  ".trim();                                 // "hi"
"hello".slice(1, 4);                             // "ell"
```

## Debugging

### `breakpoint;`

Pauses the worker for the attached debugger:

```gcl
fn complex_calculation(x: int): int {
    var step1 = x * 2;
    breakpoint;                  // execution stops here when running under debugger
    var step2 = step1 + 1;
    return step2;
}
```

Not a control-flow terminator — execution resumes after the debugger detaches.

### `println` / `info` / `error`

Use during development for one-off inspection:

```gcl
fn investigate(data: Array<int>) {
    println("size = ${data.size()}");
    info("first few = ${data}");
    if (data.size() == 0) {
        error("empty input!");
    }
}
```

Production code should use the logging functions (`info`, `warn`, `error`, etc.) which are gated by configured log level. `println` always writes to stdout.

## When stuck

The most common reasons GreyCat code "looks wrong":

1. The lang complains about `.` on a node tag — use `->` instead, or `.resolve()`.
2. The lang complains about `null` flowing into a non-null type — narrow with `if (x != null) { ... }`, or coalesce with `??`, or force with `!!`.
3. A function returns `any?` and the next call doesn't compile — cast (`(result as int) + 1`).
4. `Array<int>` won't flow into a `Array<int?>` parameter — declare the local as `Array<int?>` upfront.
5. A method exists in stdlib but the bare name doesn't resolve — the type is private to its module; use FQN (`std::core::SomeType`).
6. Construction looks "wrong" because you reflexively wrote `Type::new(...)` — use `Type { ... }`.

When the lang / runtime disagree with this skill, trust the runtime. Run `greycat run` against a minimal `project.gcl` to settle disputes.
