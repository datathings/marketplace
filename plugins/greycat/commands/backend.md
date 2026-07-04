---
name: backend
description: Comprehensive backend review and cleanup for GreyCat projects
allowed-tools: Bash, Read, Grep, Glob, Edit, Write
---

# Backend Review & Cleanup

**Purpose**: Multi-phase analysis of GCL backend — dead code, duplication, anti-patterns, optimizations, best-practice gaps.

**Run After**: Each sprint, before releases, during refactors.

---

## Phase 1: Dead Code

### 1.1 Unused types
\`\`\`bash
grep -rn "^type [A-Z]" src/ --include="*.gcl" | grep -v "@volatile"
# For each: grep -r "TypeName" src/ --include="*.gcl" | grep -v "^type TypeName"
\`\`\`
- 0 refs outside definition → SAFE TO DELETE
- Only same-file refs → REVIEW

### 1.2 Unused functions
\`\`\`bash
grep -rn "static fn [a-z_]" src/ --include="*.gcl"     # services
grep -rn "^\s*fn [a-z_]" src/ --include="*.gcl"        # methods
grep -rn "@expose" src/ --include="*.gcl" -A 1        # @expose may live in src/api.gcl, not only *_api.gcl
\`\`\`
For each: `grep -r "FuncName(" src/ --include="*.gcl"`. Keep: `@expose`, `main`, `@test`.

### 1.3 Unused globals
\`\`\`bash
grep -rn "^var [a-z_]" src/ --include="*.gcl"
\`\`\`
Global indices (`nodeIndex`/`nodeList`) may look unused but are populated during import — REVIEW carefully.

### 1.4 Unused files
\`\`\`bash
for file in src/**/*.gcl; do
  fname=$(basename "$file" .gcl)
  refs=$(grep -r "$fname" src/ --include="*.gcl" | grep -v "$file" | wc -l)
  [ $refs -eq 0 ] && echo "Unused: $file"
done
\`\`\`

### 1.5 Commented code
\`\`\`bash
grep -rn "^//.*fn \|^//.*type \|^//.*var " src/ --include="*.gcl"
\`\`\`

---

## Phase 2: Duplication

### 2.1 Repeated validation
\`\`\`bash
grep -rn "if.*== null.*return null" src/ --include="*.gcl"
grep -rn "if.*\.size\(\) == 0.*return" src/ --include="*.gcl"
\`\`\`
Extract repeated patterns into validation utilities or error-handler helpers.

### 2.2 Similar types
\`\`\`bash
grep -rn "^type [A-Z]" src/ --include="*.gcl" -A 10
\`\`\`
Types with >70% field overlap → consider inheritance or shared base.

### 2.3 Copy-pasted blocks
Identical 5+ line sequences in 3+ files (error handling especially). Extract to shared helper.

---

## Phase 3: Anti-Patterns

### 3.1 Persistent collections used as locals
\`\`\`bash
grep -rn "var.*= nodeList\|var.*= nodeIndex" src/ --include="*.gcl"
\`\`\`
Inside `fn` body → wrong. Use `Array<T>`/`Map<K,V>`. Note: `greycat-lang lint` does NOT currently flag this — review by hand.
**Priority**: MEDIUM (style; not lint-enforced).

### 3.2 `@volatile` on API response types (recommended)
\`\`\`bash
grep -rn "@expose" src/ --include="*.gcl" -A 5
\`\`\`
Mark slim `…View` response DTOs `@volatile` — it guards the type against ever being written into the graph. It is a recommendation for API-shaped types, not a persistence fix: returning a plain type from `@expose` does NOT persist it (persistence happens only on write into a `node<T>` / graph attribute).
**Priority**: LOW.

### 3.3 Node-collection element type
\`\`\`bash
grep -rn "nodeList<\|nodeIndex<" src/ --include="*.gcl"
\`\`\`
Node tags store their payload value **directly** — `nodeList<Item>`, `nodeIndex<String, User>`, `nodeGeo<Station>` are all correct and idiomatic. Wrapping the element in an extra `node<...>` (`nodeIndex<K, node<V>>`) is valid but only needed when the **same node must be shared across multiple indices** (store one `node<V>` ref in each). Flag only that sharing case — don't "fix" plain `nodeList<Item>`.
**Priority**: LOW (design; not lint-enforced).

### 3.4 Uninitialized non-nullable collection fields
\`\`\`bash
grep -rn "^\s*[a-z_]*:\s*\(Array\|Map\|nodeList\|nodeIndex\)" src/ --include="*.gcl"
\`\`\`
Either initialize in literal (`streets: nodeList<node<Street>>{}`) or make nullable.
**Priority**: HIGH (runtime error).

### 3.5 Null dereference chains
\`\`\`bash
grep -rn "\.resolve()\..*\..*\|->.*->.*" src/ --include="*.gcl"
\`\`\`
Add null checks or optional chaining + `??`.

### 3.6 Generic static functions (runtime erasure risk)
\`\`\`bash
grep -rnE 'static\s+fn\s+[a-z_][a-zA-Z0-9_]*<' src/ --include="*.gcl"
\`\`\`
Compiles, but generics are erased at runtime; explicit `<T>` instantiation can crash inside `greycat test`. Prefer instance method on generic type or monomorphise (`processInt`, `processString`).
**Priority**: MEDIUM (compile passes; runtime risk).

### 3.7 Raw-string `throw`
\`\`\`bash
grep -rnE 'throw\s+"' src/ --include="*.gcl"
\`\`\`
Define typed errors in `src/errors.gcl` (`AppError`/`NotFoundError`/`ValidationError`) and `throw NotFoundError { code, message, id }`. Always re-throw after `error(...)` log.
**Priority**: HIGH.

### 3.8 `await(...)` where the entry path isn't task-backed
\`\`\`bash
grep -rn "await(" src/ --include="*.gcl"
\`\`\`
An `@expose` HTTP call is already enqueued as a task, so `await(jobs)` **does** fan out over an HTTP POST. The genuinely serial case is a one-shot `./bin/greycat run <fn>` script — there jobs run serially. Flag `await(jobs)` in a function only ever invoked via one-shot `greycat run` (or move that work to a task / `PeriodicTask`).

Related:
- Prefer `Array<Job>` + cast `.result() as T` at the call site. (Parameterised `Array<Job<T>>` compiles and lints clean; runtime correctness depends on per-job result-type discipline.)
- Batch sizes should be ~120, not full worker count (leave ≥8 for OS). No nested `await`.
- No `node<T>{...}` constructors inside parallel jobs — pre-allocate sequentially.
- `System::exec` + `&; wait` for "parallelism" → second exec in non-task HTTP throws uncatchable.

### 3.9 TTL-exceeding endpoint w/o task fallback
\`\`\`bash
grep -rn "@expose" src/ --include="*.gcl" -A 30 | grep -E "for\s*\(.*in\s+|import|System::exec"
\`\`\`
The request-TTL knob is the `--request_ttl` flag (default `20s`, `serve` only) — there is no `GREYCAT_REQUEST_TTL` env var. Past the TTL the runtime kills the handler — `try/catch` does NOT catch. Raise `--request_ttl`, dispatch as a background task (`task: true` request header), or factor into a scheduled task.
**Priority**: HIGH.

### 3.10 `!!` overuse
\`\`\`bash
grep -rn '!!' src/ --include="*.gcl"
\`\`\`
`!!` is OK ONLY for global-registry lookups guaranteed by init. For user data, prefer `if (x == null) throw NotFoundError {...};`. Also flag redundant `!!` on receivers already narrowed by `if (x != null)`.
**Priority**: MEDIUM.

---

## Phase 4: Optimizations

### 4.1 Expensive ops in loops
\`\`\`bash
grep -rn "for.*in.*{" src/ --include="*.gcl" -A 10 | grep -E "\.resolve\(\)|\.get\(|->.*->"
\`\`\`
Repeated `.resolve()` inside nested loops → batch or restructure.

### 4.2 Missing indices on frequent queries
\`\`\`bash
grep -rn "for.*in.*if.*==.*return" src/ --include="*.gcl"
\`\`\`
Linear search over global indices for `==` matches → add a `nodeIndex<K, node<V>>`.
**Priority**: HIGH when called frequently.

### 4.3 C-style loops
\`\`\`bash
grep -rn "for (.*=.*;.*<.*\.size().*;.*++)" src/ --include="*.gcl"
\`\`\`
Use `for (i, v in items)`.

### 4.4 Over-fetching in API responses
Detect `@expose` returning full domain types (20+ fields) when consumer needs 3 — define lightweight `*View` type.

### 4.5 Unnecessary persistence
\`\`\`bash
grep -rn "var [a-z_]* = node\(List\|Index\|Time\|Geo\)?<" src --include="*.gcl"
grep -rn "fn [a-z_]*(.*).*: node\(List\|Index\)" src --include="*.gcl"
\`\`\`
Local vars / function returns using persistent types → use `Array`/`Map`.
**Priority**: CRITICAL.

### 4.6 Reimplemented native functions
\`\`\`bash
grep -rn "fn [a-z_]*sort\|fn find_\(max\|min\)\|fn [a-z_]*join" src --include="*.gcl"
\`\`\`
Custom sort → `.sort_by(...)`. Custom join → `.join(sep)`. Custom min/max → the global `min(x, y)` / `max(x, y)` (no `Math::` namespace) or tensor ops.

### 4.7 Useless one-line wrappers
\`\`\`bash
for file in $(find src -name "*.gcl"); do
  awk '/^fn [a-z_].*{$/{l=NR;f=$0;getline;if($0~/^    return .*::/){print FILENAME":"l": "f}}' "$file"
done
\`\`\`
`fn get_user(id) { return UserService::find_by_id(id); }` adds zero value — call directly.

### 4.8 O(n²)
\`\`\`bash
grep -rn "for.*in.*{" src --include="*.gcl" -A 5 | grep -B 1 "for.*in.*{" | grep -A 3 "if.*=="
\`\`\`
Nested loops with conditional id-match → add `nodeIndex<id, ...>` for O(1) lookup.
**Priority**: CRITICAL on large datasets.

---

## Phase 5: Best-Practice Checks

- **Services**: business logic lives in a service layer (commonly an `abstract type XxxService` with static methods, or plain top-level `fn`s), not in the `@expose` API layer.
- **Cross-module refs resolve via the project graph** — declaration order within a file does not matter; every loaded module sees every other (modulo the `private` FQN rule). Don't flag "index declared after the type that uses it".
- **@permission**: a bare `@expose` already requires the `api` permission (authenticated) — that's the recommended default, so a *missing* `@permission` is NOT an error. Flag instead: `@permission("public")` on mutations (anonymous write), or a privileged op that should be `@permission("admin")`.
- **Try/catch on @expose**: every body wrapped.
- **Lean `…View` types for the frontend**: `@expose` returns feed the **Lit + Shoelace + lucide-static** frontend via the generated `@greycat/web` client — return slim `@volatile …View` types (not full domain types) so payloads stay small (helps Lighthouse) and codegen→TS stays clean. After any backend type/`@expose` change, regenerate the client (`greycat codegen ts`). For full frontend review (Lit patterns, Shoelace tree-shaking, Lighthouse/SEO), use `/greycat:frontend`.

---

## Phase 6: Cleanup (Interactive)

After analysis, present summary + offer cleanup options:
- A) Delete unused code (functions + types)
- B) Fix anti-patterns (persistence, missing @volatile)
- C) Fix performance issues (wrappers, unnecessary persistence)
- D) Report only
- E) Custom per-category

For each cleanup:
1. Verify git status is clean (or backup branch)
2. Apply Edit operations
3. Run `greycat-lang lint` — must pass
4. Report counts (functions/types deleted, lines saved)

---

## Output

Group by priority (HIGH/MEDIUM/LOW). Per issue: `📍 file:line` + problem + fix snippet + estimated effort.

```
SUMMARY:
  Dead code:           N items
  Duplications:        N blocks
  Anti-patterns:       N issues
  Optimizations:       N opportunities
  Best-practice gaps:  N

Priority: HIGH=X, MEDIUM=Y, LOW=Z
```

---

## Notes

- Run when `greycat-lang lint` already passes
- Backup recommended (git checkpoint)
- Fix HIGH first; test (`greycat test`) after each cleanup batch
