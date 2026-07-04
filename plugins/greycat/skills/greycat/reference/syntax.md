# GreyCat syntax reference

Complete syntax reference. Reach for this when a construct in `SKILL.md` needs to be expanded — particularly less-common statement forms, full operator precedence, and literal shapes.

## Contents

- Comments and doc comments
- Identifiers and keywords
- Module-level declarations
- Type bodies (attributes and methods)
- Statements (every form)
- Expressions (every form)
- Operator precedence (complete table)
- Literals (numbers, strings, chars, time, typed suffixes)
- Ranges and intervals

## Comments and doc comments

```gcl
// line comment
/* block comment */
/// doc comment — attaches to the declaration that follows
/// (multiple /// lines are joined as one doc block)
```

`///` is **not** a generic line comment — it is captured as documentation by the lang and shown in hover / completion. Place it directly above a `type`, `enum`, `fn`, `var`, attribute, or method. Internal `//` and `/* */` are stripped by the lexer.

## Identifiers and keywords

Identifiers match `[a-zA-Z_][a-zA-Z0-9_]*`. The following are reserved keywords and cannot be used as identifiers:

```
type  enum  fn  var
extends  private  static  abstract  native
if  else  while  do  for  in  at
return  throw  break  continue  breakpoint
try  catch
true  false  null  this  typeof
is  as
```

The following are **contextual** keywords (recognized only at specific grammar positions; usable as identifiers elsewhere): `sampling`, `limit`, `skip` (inside `for-in`).

Annotation names follow `@` and use the identifier regex.

## Module-level declarations

A module is a flat sequence:

```gcl
module ::= ( modvar | fn_decl | type_decl | enum_decl | mod_pragma )*
```

### Module pragma

```gcl
@library("name", "version");
@include("path");
@permission("name", "description");
@role("name", "perm1", "perm2", ...);
```

Pragmas must terminate with `;`. They may carry a leading `///` doc comment.

### Module-level var

```gcl
var name: node<T?>;                // node<T?> MUST be nullable on the inner type
var name: nodeList<T>;
var name: nodeIndex<K, V>;
var name: nodeTime<T>;
var name: nodeGeo<T>;
private var name: node<T?>;
```

Module-level vars are static singletons restricted to the stdlib node tags — no other type is valid here. They cannot have an initializer at the declaration site; populate them through a function body. The `node<T?>` form **must** carry the `?` on the inner type (the runtime requires nullable storage).

### Function

```gcl
@annotations
modifiers fn name<G1, G2>(p1: T1, p2: T2): R { ... }
modifiers fn name(p: T);              // native or abstract: body omitted

// Modifiers: private, static, abstract, native (any subset, in any order)
// Return type clause is optional (no clause = no return value).
```

### Type

```gcl
@annotations
modifiers type Name<G1, G2> extends Parent<G> {
    @annotations
    modifiers attr_name: T;
    @annotations
    modifiers attr_name: T = init_expr;
    "string-named-attr": T;                    // attr name can be a string literal

    @annotations
    modifiers fn method_name<G>(p: T): R { ... }
}
```

Inheritance is single (one `extends`). Generic parameters use angle brackets and bare identifiers (no bounds: `T: Bound` is not GreyCat syntax).

### Enum

```gcl
enum Name {
    field1,                  // bare field
    field2(0),               // field with value
    field3("string"),
    "field-with-dash";       // field name may be a string literal
}
```

Fields are separated by `,` or `;` interchangeably. The trailing separator is optional. The optional `(value)` argument can be any literal expression (primitive-only, it cannot be non-compile-time evaluable). Bare-string entries (`"…"`) are valid; TimeZone enum is the canonical example.

## Statements

Inside a block (`{ ... }`):

```gcl
var name = expr;                       // type inferred
var name: T = expr;                    // type annotated
var name: T;                           // declared but uninitialized

return;
return expr;
throw expr;
break;
continue;
breakpoint;                            // pauses the worker for the debugger

expr;                                  // expression statement

if (cond) { ... }
if (cond) { ... } else { ... }
if (cond) { ... } else if (cond) { ... } else { ... }

while (cond) { ... }
do { ... } while (cond);

for (var i = 0; i < n; i = i + 1) { ... }       // C-style
for (var i: int = 0; i < n; i = i + 1) { ... }  // with explicit iterator type

for (i, v in arr) { ... }                       // Array iteration (i = index, v = value) — 2-param form required
for (i: int, v: int in arr) { ... }             // with annotations
for (k, v in map) { ... }                       // unpack key, value
for (_, v in map) { ... }                       // `_` discards an unused key/index (silences unused-local)
for (t, v in series[from..to]) { ... }          // range slice
for (t, v in series[from..to] sampling expr limit n skip m) { ... }

try { ... } catch (e) { ... }
try { ... } catch { ... }                       // catch ident is optional

at (time_value) { ... }                         // changes the scope `time::current()`, nodeTime resolve against the current time by default
```

Every statement that isn't a block ends with `;` or an automatic semicolon at newline / `}` / EOF. The grammar accepts both forms, but always-writing `;` is conventional.

## Expressions

```gcl
// literals
true   false   null   this
42   3.14   1e6   1_000_000   42_time   60s   2hour
'a'   '\n'   '2025-05-22T16:47:42Z'
"string"   "with ${substitution}"
[1, 2, 3]                                       // array literal
(a, b)                                          // tuple literal (exactly 2 elements)
( expr )                                        // parenthesized expression

// identifier
foo

// member access
obj.field
obj.method(args)
obj?.field                                      // optional chaining
obj?.method(args)
obj->field                                      // deref-then-field (only on stdlib node tags: node, nodeTime, nodeList, nodeIndex, nodeGeo)
obj?->field
Type::staticMember
Module::Type::staticMember
Module::Type::staticMember(args)

// object initialization
Type { field: expr, field2: expr2 }             // named-field form
Type<G> { positional1, positional2 }            // positional form
Type<G> { }                                     // empty

// indexing / slicing
arr[i]
arr[i..j]                                       // range slice (only in for_in)
arr[i..]                                        // open upper (only in for_in)
arr[..j]                                        // open lower (only in for_in)
arr?[i]                                         // optional index

// math intervals (used after `in` in for-in)
]a..b]    [a..b]    ]a..b[    [a..b[            // explicit inclusivity (only in for_in)

// call
foo(args)

// lambda
fn (p: T): R { body }                           // anonymous function

// unary
-x   +x   !x   *x
--x   ++x                                       // prefix
x--   x++   x!!                                 // postfix; !! is "force non-null"

// binary (see Precedence table)
a + b   a - b   a * b   a / b   a % b   a ^ b
a < b   a <= b   a > b   a >= b   a == b   a != b
a && b   a || b
a ?? b                                          // null coalesce
a is T                                          // type test (returns bool, narrows)
a as T                                          // cast
a = b                                           // assignment
a ?= b                                          // assign only if a is null
```

## Operator precedence

Highest binds tightest. All binary operators are left-associative unless noted.

| Prec | Operator                           | Notes                                               |
| ---- | ---------------------------------- | --------------------------------------------------- |
| 13   | `. -> :: () [] ++ -- !!` (postfix) | Member, call, index, increment, force-non-null      |
| 12   | `- ! + * ++ --` (prefix)           | Unary                                               |
| 11   | `??`                               | Nullish coalesce                                    |
| 10   | `^`                                | Power                                               |
| 9    | `* / %`                            |                                                     |
| 8    | `+ -`                              |                                                     |
| 7    | `< <= > >=`                        |                                                     |
| 6    | `== !=`                            |                                                     |
| 5    | `as` `is`                          | Type cast / test (RHS is a type, not an expression) |
| 4    | `&&`                               | Logical AND                                         |
| 3    | `\|\|`                             | Logical OR                                          |
| 2    | `=` `?=`                           | Assignment (right-associative semantics)            |
| 1    | `..`                               | Range (in `arr[a..b]` slice contexts)               |

Gotchas:

- `??` binds tighter than `^`. `count ?? 0 > 0` parses as `(count ?? 0) > 0`.
- `-o.b` parses as `-(o.b)` (postfix `.` binds tighter than prefix `-`).
- `o.x * o.y` parses correctly as `(o.x) * (o.y)`.

## Literals

### Numbers

```gcl
42                  // int (i64)
1_000_000           // int with visual underscores
0                   // int (no leading-zero octal; just zero)
3.14                // float (f64)
.5                  // INVALID — leading digit required
3.14e+10            // float scientific
1e-10               // float scientific (no decimal required)
```

### Typed-suffix numbers

```gcl
42_time             // value of type `time` (epoch microseconds)
42time              // same — underscore is optional (formatter prefers `_`)
1.79e+308_f         // float (forces typed `float` interpretation)
60s   1ms           // duration with named unit
2hour_42ms          // compound duration: 2 hours + 42 ms
3day_4hour_5min_6s  // compound duration (any chain)
```

The lexer accepts any letter sequence as a suffix; the lang validates whether it names a real unit or typed-suffix kind. Bogus `42xyz` parses but errors semantically.

### Strings

```gcl
"hello"
"with newline\n"                    // standard escapes: \n \r \t \\ \" \' \xFF
"interpolation: ${expr}"            // ${...} contains an arbitrary expression
"nested: ${a + b * c}"
""                                  // empty string
```

Strings are double-quoted only. Single quotes are reserved for `char` and `time` literals.

### Chars

```gcl
'a'
'\n'   '\t'   '\\'   '\''
'é'            // unicode codepoint
```

A `char` literal contains exactly one character (or escape).

### Time literals

```gcl
'2025-05-22T16:47:42Z'              // full ISO 8601 (UTC)
'2025-05-22T16:47:42+02:00'         // with timezone offset
'2025-05-22'                        // date only
```

Time literals use **single quotes** with an ISO 8601 payload. They produce a value of type `time`.

## Ranges and intervals

GreyCat distinguishes two range forms based on whether bracket-inclusivity matters:

**Implicit (within `[ ]` slices):**

```gcl
arr[a..b]           // [a..b] — inclusive lower, exclusive upper
arr[a..]            // open upper
arr[..b]            // open lower
arr[..]             // both open (rare)
```

**Explicit (after `in` in `for-in`, bracket markers carry meaning):**

```gcl
for (t, v in series]from..to]) { ... }   // exclusive lower, inclusive upper
for (t, v in series[from..to[) { ... }   // inclusive lower, exclusive upper
for (t, v in series]from..to[) { ... }   // both exclusive
```

Endpoints in explicit intervals can be omitted (`]from..]` is "exclusive lower, open upper").

**IMPORTANT**: ranges can only be used in `for (...) {}`

## Lambdas

```gcl
var add = fn (a: int, b: int): int { return a + b; };
var x = add(40, 2); // x: int
```

A lambda has the same parameter / return-type shape as a `fn` declaration but no name. The resulting value carries its structural signature, displayed as `fn(P0, P1): R` (or `fn(P0, P1)` when no return is declared or inferable). See [types.md](types.md) for how this interacts with the nominal `function` slot.

**Lambdas have a closed scope — there are no closures.** A lambda body can reference its own parameters, locals declared inside the body, and module-scope decls. References to *enclosing* locals, params, or `this` are rejected by the lang (`lambda-capture`) and would fail at runtime (`unresolved identifier`, or segfault for `this`). Pass anything the lambda needs as an explicit parameter:

```gcl
fn main() {
    var threshold = 10;
    var f = fn (x: int): bool { return x > threshold; };  // ERROR: `threshold` captured
    var g = fn (x: int, t: int): bool { return x > t; };  // OK: pass it in
    g(42, threshold);
}
```

## Type identifiers

Type references appear after `:` (annotation) and inside `<...>` generic arguments.

```gcl
T                                   // bare
mod::T                              // module-qualified
mod::sub::T                         // multi-segment
Array<T>                            // generic
Map<String, Array<int>>             // nested generic
T?                                  // nullable
typeof T                            // type-as-value (for type-reflection APIs)
```

`typeof T` is used in static-reflection APIs (e.g. `type::enum_by_offset<T>(enum_type: typeof T, ...)`). It is **not** a way to introduce a generic parameter — generics use bare identifiers in `<...>`.

## Object init forms in detail

```gcl
// Named-field form. Order doesn't matter. Missing fields default to null (if nullable)
// or are reported as errors (if non-null in the type decl).
Point { x: 1, y: 2 }
Point<int> { x: 1, y: 2 }                  // generic args are optional

// Positional form is supported only for: Array, Map, node, geo
Array<int> { 1, 2, 3 }
Map<String, int> { "a": 1, "b": 2 }
node<String> { "text" }
geo { 1, 2 } // lat, lng

// Empty form: every attr must be nullable.
Empty { }
Empty<T> { }
```

## What is NOT in the grammar

These constructs do **not** parse in GreyCat — do not write them:

- `import` / `use` / `from` statements — visibility is project-graph based.
- `new Type(...)` — no `new` keyword. Use `Type { ... }`.
- `Type::new(...)` — not a convention. Use `Type { ... }`.
- Ternary `cond ? a : b` — write an `if` statement.
- `switch` / `match` — write chained `if` (or use `type::enum_offset`).
- `void` return type — omit the `: T` clause.
- `T extends Bound` generic constraints — not supported; generics are unbounded.
- `interface` — use `abstract type`.
- `=>` arrow lambdas — lambdas use `fn (params) { body }`.
- `let` — variables are always `var`.
- `const` — there is no const keyword; use `static` attributes for constants.
- `null` as a member name — `null` is a keyword.
- C-preprocessor directives.
- Statement-level `<expr>` without `;` (every statement requires a terminator, automatic or explicit).
