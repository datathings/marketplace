---
name: typecheck
description: Advanced type safety checks beyond greycat-lang lint
allowed-tools: Bash, Read, Grep, Glob
---

# Type Safety Checker

**Purpose**: Catch type-safety issues `greycat-lang lint` misses.

**Run After**: Each sprint, before releases, when modifying type definitions.

---

## Phase 1: @volatile on API response types

\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 5 | grep "fn.*:" | sed 's/.*: //; s/ {.*//' | sort -u
\`\`\`

For each return type:
- Basic (`int`/`String`/`bool`/`float`) → OK
- `Array<X>`/`Map<K,V>` → check element type
- Custom → verify type has `@volatile` decorator

Missing `@volatile` → DB bloat, persistence overhead.
**Severity**: MEDIUM.

---

## Phase 2: Collection Type Safety

\`\`\`bash
grep -rn "nodeList\|nodeIndex\|nodeTime\|nodeGeo" src/ --include="*.gcl" -B 3 -A 3
\`\`\`

Classify each occurrence:
- ✅ Module-level `var` → OK
- ✅ Type field → OK
- ⚠ Local variable → use `Array`/`Map`
- ⚠ Function parameter → use `Array`/`Map`

**Severity**: HIGH (type mismatch + persistence overhead).

---

## Phase 3: Null Safety

### 3.1 Chained dereferences
\`\`\`bash
grep -rn "\.resolve()\..*\." src/ --include="*.gcl"
grep -rn "->.*->" src/ --include="*.gcl"
\`\`\`
Check whether null checking exists (`if (x == null)`, `?.`, `!!`) before each dereference.

### 3.2 `!!` overuse
\`\`\`bash
grep -rn '!!' src/ --include="*.gcl"
\`\`\`
`!!` acceptable ONLY when:
- Global-registry lookup where init guarantees presence (`GameRegistry::getConfig(game)!!`)
- Value from registry just validated upstream in same function

NOT acceptable:
- Receiver from user input / untyped JSON
- Used to silence the analyzer without a proven invariant
- When `if (x == null) throw NotFoundError {...};` is more honest

Also flag **redundant `!!`** on receivers already narrowed by `if (x != null) { ... }`.

---

## Phase 4: Type Consistency

### 4.1 Collection initialization
\`\`\`bash
grep -rn "^\s*[a-z_]*:\s*\(Array\|Map\|nodeList\|nodeIndex\)" src/ --include="*.gcl"
\`\`\`
Non-nullable collection FIELDS must be initialized in object literal (`Document { chunks: nodeList<node<Chunk>>{} }`) or declared nullable.
**Severity**: HIGH (runtime error).

### 4.2 Node storage
\`\`\`bash
grep -rn "nodeList<[^n]" src/ --include="*.gcl"
grep -rn "nodeIndex<.*,\s*[^n]" src/ --include="*.gcl"
\`\`\`
Persistent collections must store node refs: `nodeList<node<Item>>`, not `nodeList<Item>`. Note: `greycat-analyzer` does NOT currently flag this — review by hand.
**Severity**: MEDIUM (style; not lint-enforced).

---

## Phase 5: Generic Type Safety

### 5.1 Static fn with generics (runtime erasure risk)
\`\`\`bash
grep -rn "static fn.*<.*>.*(" src/ --include="*.gcl"
\`\`\`
Compiles cleanly, but generics are erased at runtime — explicit `<T>` calls can crash inside `greycat test`. Either remove `static` (instance method on `type Utils<T>`) or specialize (`processInt`, `processString`).
**Severity**: MEDIUM (compile passes; runtime risk).

### 5.2 Generic return types (runtime erasure)
\`\`\`bash
grep -rn "fn.*:\s*Array<.*<" src/ --include="*.gcl"
grep -rn "fn.*:\s*[A-Z][a-zA-Z]*<.*<" src/ --include="*.gcl"
\`\`\`
GreyCat erases generics at runtime in `greycat test`. `Array<Wrapper<T>>` return crashes the test runner.
Fix: non-generic result type with `value: any`; cast at call site.
\`\`\`gcl
// ❌
abstract type Service<T> { static fn run(): Array<Result<T>> {...} }
// ✅
@volatile type Result { value: any; meta: String; }
abstract type Service<T> { static fn run(): Array<Result> {...} }
// caller: var typed = res.value as MyType;
\`\`\`

### 5.3 Anonymous return structs
\`\`\`bash
grep -rn "fn.*:\s*{" src/ --include="*.gcl"
\`\`\`
`fn foo(): { x: int, y: int }` doesn't parse. Declare a named (typically `@volatile`) type.

---

## Phase 6: Precedence Pitfalls

### 6.1 Cast + nullish coalescing
\`\`\`bash
grep -rnE 'as +[A-Za-z_][A-Za-z0-9_]*\? +\?\?' src/ --include="*.gcl"
\`\`\`
`x as T? ?? "default"` parses with surprising grouping. Prefer `(x as T?) ?? "default"` for clarity.

### 6.2 Module-level non-node vars
\`\`\`bash
grep -rnE '^var\s+[a-z_]+:\s*(int|float|String|bool|char|Array|Map|Set)[^;]*;' src/ --include="*.gcl"
\`\`\`
Module-level `var` must be a node-tag type (`node<T?>`, `nodeIndex<K,V>`, `nodeList<T>`, `nodeTime<T>`, `nodeGeo<T>`). Wrap primitives: `var counter: node<int?>;`.

---

## Phase 7: Frontend type safety (Lit + TypeScript) — if `frontend/` exists

Stack: **Lit + TypeScript + Shoelace + Lucide** on Vite + `@greycat/web`. Beyond `tsc --noEmit` (`pnpm lint`):

### 7.1 tsconfig for Lit decorators
\`\`\`bash
grep -E 'experimentalDecorators|useDefineForClassFields|moduleResolution' tsconfig*.json
\`\`\`
Lit needs `"experimentalDecorators": true`, `"useDefineForClassFields": false`, `"moduleResolution": "bundler"`, `"strict": true`. Missing these → `@property`/`@customElement` mistype or runtime field-init bugs.

### 7.2 `gc` namespace shadow
\`\`\`bash
grep -rnE '\bgc\.' frontend/ --include="*.ts" | grep -v "gc-runtime.ts"
\`\`\`
`@greycat/web`'s `gc` is shadowed by `@types/node`'s `var gc`. Use the `gcRuntime` re-export, never bare `gc.*`.

### 7.3 Stale codegen / hard-coded strings
`project.d.ts` must be regenerated (`pnpm gen`) after backend type/`@expose` changes; never hand-edit it. Derive endpoint/field strings from `$fields`, not string literals. **No `any`** — use `gc.core.*` / `gc.project.*` types.

### 7.4 Custom element typing
Each `@customElement` must extend `HTMLElementTagNameMap` so templates and `document.createElement` are typed.

**Severity**: HIGH (7.1 decorator config), MEDIUM (shadow, stale codegen), LOW (element-map gaps).

---

## Output

```
ISSUES:
  CRITICAL (compile errors):     N (generic static fns)
  HIGH (runtime errors):          N (uninit collections, bad node storage, persistent locals)
  MEDIUM (safety):                N (missing @volatile, null deref, unsafe casts)
  LOW (best practices):           N (loose types, missing annotations)
```

Per finding: `📍 file:line` + severity + problem + fix snippet.

---

## Verify

\`\`\`bash
greycat-lang lint   # must pass
greycat test        # run tests
greycat build       # build
# frontend (if exists):
pnpm gen && pnpm lint   # regenerate client, then tsc --noEmit
\`\`\`

---

## Notes

- Complements `greycat-lang lint`
- Fix CRITICAL first (compile errors)
- Use BEFORE: `/apicheck`, `/backend`, `/docs`
- Use AFTER: major refactors, type changes, lib upgrades
