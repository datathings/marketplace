# GreyCat type system

Reach for this when reasoning about assignability, nullability, generics, narrowing, casting, or inheritance.

## Contents

- Primitive and value types
- Nullability rules
- Null-flow narrowing
- Generic types and invariance
- Inheritance and `extends`
- The `is` operator (type test + narrowing)
- The `as` operator (cast)
- The `any` and `null` meta-types
- `typeof` and `type` reflection
- Subtyping summary

## Primitive and value types

| Type       | Description                                                                             |
| ---------- | --------------------------------------------------------------------------------------- |
| `bool`     | `true` / `false`.                                                                       |
| `int`      | 64-bit signed integer (`-2^63` to `2^63 - 1`).                                          |
| `float`    | 64-bit IEEE 754 float.                                                                  |
| `char`     | Single character (any UTF-8 codepoint).                                                 |
| `String`   | Immutable Unicode string. Auto-cloned when reassigned across owners.                    |
| `time`     | Universal instant. Microsecond precision. Literal: `'2025-05-22T16:47:42Z'`, `42_time`. |
| `duration` | Span of time. Built via `60s`, `2hour_30min`, or `duration::new(v, unit)`.              |
| `geo`      | A `(lat, lng)` decimal coordinate. Literal: `geo { 49.6, 6.1 }`.                        |
| `function` | Nominal slot for any function value. Lambdas and fn references flow in; the slot itself loses the signature. See "The `function` type" below. |
| `field`    | Reflective field descriptor (used by `Table`, sort-by-field, etc.).                     |
| `type`     | Reflective type descriptor.                                                             |

Primitives are passed by value. `String` behaves as a value semantically (immutable, auto-cloned on assignment across containers).

`time`, `duration`, `geo` are **native types with no fields** — they have methods only. `t.format(fmt, tz)` works; `t.year` is not (use `t.toDate(tz).year` instead).

## Nullability rules

Every type reference is non-null by default. Appending `?` makes it nullable.

```gcl
var a: int;              // declared non-null int — must be assigned before any read
var b: int?;             // declared nullable; reads as null until assigned
var c: int = null;       // ERROR — null is not assignable to int
var d: int? = null;      // ok
var e: int = (b ?? 0);   // ok — ?? produces non-null int
```

A non-null type `T` is a subtype of `T?` — assigning a non-null value into a nullable slot is implicit. The reverse requires elimination of null:

```gcl
fn need_int(x: int) {}

fn use(maybe: int?) {
    need_int(maybe);       // ERROR — int? not assignable to int
    need_int(maybe!!);     // ok — `!!` forces non-null (runtime throw if null)
    need_int(maybe ?? 0);  // ok — ?? produces int
    if (maybe != null) {
        need_int(maybe);   // ok — narrowed to int in this branch
    }
}
```

### Operations that require non-null receivers

These operators throw at runtime if the receiver is null AND error statically when the analyzer can't prove non-null:

- `.field`, `.method()` (use `?.` to propagate null)
- `->field`, `->method()` (use `?->` to propagate null)
- `[i]` indexing (use `?[i]` to propagate null)
- arithmetic / comparison / logical operators on the value

`?.`, `?->`, `?[]` propagate null up the chain — the result of the whole expression becomes nullable.

## Null-flow narrowing

The analyzer tracks null possibility across control flow. After these checks, the value's type is locally narrowed to its non-null counterpart:

```gcl
if (x == null) { /*...*/ } else {   // x: T  here
    use(x);                      // safe
}

if (x != null) {                 // x: T  here
    use(x);
}

if (x == null) {
    return;                      // early return — after this block, x: T
}
use(x);                          // safe

x = something_not_null();        // x: T after this assignment
```

Reassignment to a nullable value re-introduces null possibility. The analyzer joins narrows across `if`/`else` paths.

### `??` and `!!`

```gcl
var s: String? = /*...*/;
var len: int = (s ?? "").size();   // ?? produces non-null
var len2: int = s!!.size();        // !! is the unsafe escape — assert non-null
```

`!!` is **postfix**. It compiles to a runtime null-check that throws on null, then types the expression as non-null.

## Generic types and invariance

Generic parameters use angle brackets and bare identifiers. There are no bounds (`T: SomeBound` is not GreyCat syntax).

```gcl
type Box<T> {
    value: T;
}

fn first<T>(arr: Array<T>): T? {
    if (arr.size() == 0) {
        return null;
    }
    return arr.get(0);
}
```

### Strict invariance

GreyCat generics are **invariant** in their arguments. `Array<int>` is **not** assignable to `Array<float>`, and `Array<int?>` is **not** assignable to `Array<int>`.

Though, greycat allows `any` and `any?` pass-through:

```gcl
fn take_any(_: Array<any>) {}
fn take_any_nullable(_: Array<any?>) {}
fn take_int(_: Array<int>) {}
fn take_int_nullable(_: Array<int?>) {}

fn main() {
    var ints = [1, 2, 3]; // inferred as Array<int>
    take_any(ints);          // OK
    take_any_nullable(ints); // OK
    take_int(ints);          // OK
    take_int_nullable(ints); // ERROR: Array<int> not assignable to Array<int?>

    var ints_nullable = Array<int?> { 0, null, 2 };
    take_any(ints_nullable);          // OK
    take_any_nullable(ints_nullable); // OK
    take_int(ints_nullable);          // ERROR: Array<int?> not assignable to Array<int>
    take_int_nullable(ints_nullable); // OK
}
```

This holds for **every** generic, including built-ins: `Map<K, V>`, `Set<T>`, `Tuple<T, U>`, `nodeIndex<K, V>`, `nodeTime<T>`, etc.

### "Raw" forms

Every generic type without explicit generic params is materialized as: `Generic<any?>`

```gcl
static native fn sample(refs: Array<nodeTime>, ...): Table;
//                                  ^^^^^^^^ no <T>
```

This is the **raw form**, accepted only in source positions where the runtime erases the parameter. **You cannot assign `nodeTime<float>` to `nodeTime` in user code** — it is only valid as a declared parameter type in stdlib `native` signatures.

## Inheritance and `extends`

```gcl
abstract type Shape {
    name: String;
    abstract fn area(): float;
}

type Circle extends Shape {
    radius: float;
    fn area(): float {
        return 3.14159 * this.radius ^ 2;
    }
}
```

Rules:

- Single inheritance — exactly one `extends` clause, naming exactly one type.
- The parent's attributes are inherited (no need to redeclare).
- A subtype's instance literal (`Circle { name: "c1", radius: 2.0 }`) must initialize every required attribute, including those inherited.
- A `Circle` value is assignable to a `Shape` slot (subtype → supertype is implicit).
- `Shape` → `Circle` requires an explicit `as Circle` and is checked at parse/analysis time only; the runtime does not enforce it.
- `abstract type` cannot be instantiated. Its `abstract fn` declarations have no body and must be implemented by every non-abstract subtype.

### `Object` / root type

There is no explicit `Object` / `Any` root for user types — `any` is the meta-type for "anything." User types form a forest of inheritance trees rooted at types that have no `extends`.

## The `is` operator

`is T` is a runtime type test that **also narrows the value's type** in the truthy branch.

```gcl
fn describe(x: any?) {
    if (x is String) {
        println("string of length ${x.size()}");   // x: String here
    } else if (x is int) {
        println("int = ${x + 1}");                  // x: int here
    } else if (x == null) {
        println("nothing");
    }
}
```

- `is T` accepts the same `T` syntax as a declaration (including generics, nullability, FQN).
- It returns `bool`.
- In the then-branch, the LHS value's type is narrowed to `T`.
- Type tests on generic-parametric subtypes (`x is Box<int>`) work at the analyzer level; the runtime performs an erasure-style check.

## The `as` operator

`as T` is a cast. It re-types the expression to `T` without producing a runtime check — the runtime **drops `as` entirely**.

```gcl
fn handle(raw: any?) {
    var s = raw as String;     // analyzer accepts; runtime does not verify
    println(s.size());          // if raw was not a String, behavior is undefined
}
```

Use `as` to:

- Re-type a value the analyzer cannot infer.
- Downcast (`Shape` → `Circle`) after a parallel `is` check.

The analyzer rejects `as` casts between unrelated types. Use `is` first when you want a checked downcast.

## The `function` type

Function-valued expressions carry a structural signature; the nominal `function` type is the slot they flow into.

```gcl
var f = fn(a: int, b: int): int { return a + b; };  // f: fn(int, int): int
var g = Runtime::on_files_put;                       // g: fn(function?)
f(3.14, 42);  // ERROR: float not assignable to int
```

The signature is displayed using fn-decl syntax — `fn(P)` for no declared return, `fn(P): R` when present. Body inference fills in `R` when every reachable `return` agrees on a single `T` or `T?`.

Carrying the signature applies to:

- Lambda expressions (`fn (...) { ... }`)
- Top-level fn references (`fetch_stuff`, `module::fetch_stuff`)
- `static` method references (`Foo::create`, `module::Foo::create`)

Generic fns (`fn<T>(...)`) in value position fall back to the opaque `function` — no GCL-expressible shape exists for `T` outside the call site.

Once a value flows into a `function`-typed slot (e.g. a `function` parameter), the signature is gone. Calls through such values are runtime-checked:

```gcl
fn run(callback: function) {
    callback(); // analyzer accepts; runtime throws if `callback` expects args
}
```

The wrapper rule allows `Lambda → function` freely; the reverse (`function → fn(specific)`) requires a cast.

**Instance method references are not first-class.** `obj.method` (or `Foo::instance_method`) outside a call position raises `instance-method-value-ref` — instance methods carry an implicit `this` with no representation as a free value. And since lambdas don't capture (see [syntax.md](syntax.md#lambdas)), you can't smuggle the receiver in either. Pass it explicitly as a lambda parameter:

```gcl
var f = fn (r: Foo) { r.method(); };  // call later with the receiver: `f(obj);`
```

## The `any` and `null` meta-types

| Type   | Purpose                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `any`  | Top type for non-null values of unknown type. A `String` is assignable to `any`; `any` requires `as` to flow into something specific. |
| `any?` | Top type including null. Common return type for "anything."                                                                           |
| `null` | The meta-type of the `null` literal. Rarely written in source; useful for `is null` and reflective APIs.                              |

`any` is **not** a generic parameter. `Array<any>` is a concrete type (the array can hold heterogeneous non-null values); it is not the supertype of every `Array<T>`.

## `typeof` and `type` reflection

`typeof T` is a type form that names a type reflectively. It appears as a parameter type or generic argument when the runtime needs to receive the _type itself_ as a value:

```gcl
static fn enum_by_offset<T>(enum_type: typeof T, offset: int): T?;

// Call site:
var c = type::enum_by_offset(Color, 0); // inferred as `c: Color`
```

The reflective `type` family lives in `lib/std/core.gcl`:

```gcl
type::of(value)              // returns the type of value
type::all()                  // every type in the program
some_type.fields()           // attributes
some_type.nb_enum_values()   // enum entry count
```

## Subtyping summary

| Relationship                   | Direction                                                      |
| ------------------------------ | -------------------------------------------------------------- |
| `T` → `T?`                     | Implicit. (Non-null narrows to nullable.)                      |
| `T?` → `T`                     | Requires `!!`, `??`, or null-flow narrowing.                   |
| `Sub` → `Super`                | Implicit (one `extends` step).                                 |
| `Super` → `Sub`                | Requires `as Sub`. Best paired with `is Sub` first.            |
| `T` → `any`                    | Implicit.                                                      |
| `any` → `T`                    | Requires `as T`.                                               |
| `Array<int>` → `Array<any>`    | **Allowed.** Generics are invariant BUT `any` is an exception. |
| `nodeTime<float>` → `nodeTime` | Allowed because `nodeTime` is sugar for `nodeTime<any?>`.      |

When in doubt, run the program through `greycat run` — the runtime is the oracle for assignability decisions, especially where the TS-style analyzer and the C-runtime disagree (the runtime wins).
