---
name: backend
description: Comprehensive GreyCat backend review — dead code, anti-patterns, type safety, performance/concurrency, @expose API security, and test coverage — then interactive cleanup
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion, Task
---

# Backend Review & Cleanup

**Purpose**: One deep pass over the GCL backend across **seven dimensions** — dead code & duplication, anti-patterns, type safety, performance & concurrency, `@expose` API security/design, test coverage, best-practice gaps — followed by interactive cleanup. This is the single hub for backend quality (it replaces the old separate `apicheck` / `typecheck` / `optimize` / `coverage` commands).

**Run After**: each sprint, before releases, during refactors. Run **when `greycat-lang lint` already passes** — every check here targets what the linter cannot see.

**Companion**: for the web UI use `/greycat:frontend`.

---

## How to run this review — ultrathink + ultracode

**Ultrathink (always).** Reason deeply. State assumptions explicitly; when two readings exist (`nodeIndex` vs `Map`, persisted vs `@volatile`, plain `fn` service vs `abstract type`), name both and pick with a reason. Read `lib/std/*.gcl` for real examples instead of guessing by analogy. A finding you cannot tie to a concrete failure or a canonical rule is not a finding.

**Ultracode (when available).** If multi-agent orchestration is on, run this as a **Workflow**, because the seven dimensions are independent:

1. **Fan out** — one agent per Dimension below, each returning structured findings (`file`, `line`, `severity`, `problem`, `fix`).
2. **Adversarially verify** — for every CRITICAL/HIGH finding, a second agent tries to *refute* it (re-reads the code, checks it isn't a false positive from a grep); default to dropping the finding if the refutation is plausible. Persistence/concurrency claims especially need a second look.
3. **Synthesize** — dedup across dimensions (the same `file:line` surfaces in several), then emit one severity-grouped report.

If ultracode is **not** available, walk the seven dimensions sequentially yourself as one deep pass. Either way the dimensions and their checks below are the spec.

---

## Scanning ground rules (apply to every dimension)

- **Scan all `src/**/*.gcl`.** `@expose` and services can live in `src/api.gcl` — a `*_api.gcl` glob misses it. Always `--include="*.gcl"` and filter.
- **Services are a convention, not a keyword.** Business logic may be plain top-level `fn`s **or** an `abstract type XxxService` with statics. Don't assume the Java-style `*Service` naming.
- **Order doesn't matter.** Cross-module refs resolve through the project graph; an index declared after the type that uses it is fine. Don't flag it.
- **Lint already passes.** Many checks here are explicitly *not* lint-enforced (noted per item) — they need human/agent judgment.
- **Grep is a locator, not a verdict.** Every grep below narrows where to look; open the file and confirm before reporting.

---

## Dimension 1 — Dead code & duplication

### 1.1 Unused declarations
```bash
grep -rn "^type [A-Z]" src/ --include="*.gcl"          # types
grep -rn "static fn [a-z_]\|^\s*fn [a-z_]" src/ --include="*.gcl"   # services + methods
grep -rn "^var [a-z_]" src/ --include="*.gcl"          # globals
```
For each: `grep -r "Name" src/ --include="*.gcl"` and count refs outside the definition. **0 external refs → candidate to delete**; only same-file refs → REVIEW. **Keep** `@expose`, `main`, `@test`. Global indices (`nodeIndex`/`nodeList`) can look unused but be populated during import — REVIEW carefully, never auto-delete.

### 1.2 Unused files & commented code
```bash
for f in $(find src -name "*.gcl"); do n=$(basename "$f" .gcl); \
  [ "$(grep -rl "$n" src/ --include='*.gcl' | grep -v "$f" | wc -l)" -eq 0 ] && echo "Unused: $f"; done
grep -rn "^//.*\(fn \|type \|var \)" src/ --include="*.gcl"   # commented-out code
```

### 1.3 Duplication
```bash
grep -rn "if.*== null.*return null" src/ --include="*.gcl"
grep -rn "^fn [a-z_][a-zA-Z0-9_]*(" src/ --include="*.gcl" | awk -F: '{print $3}' | sort | uniq -d
```
Repeated validation → extract a helper. Types with >70% field overlap → shared base. Identical 5+ line blocks in 3+ files (error handling especially) → extract.

---

## Dimension 2 — Anti-patterns (correctness & safety)

### 2.1 Uninitialized non-nullable collection fields — **HIGH (runtime error)**
```bash
grep -rnE "^\s*[a-z_]+:\s*(Array|Map|nodeList|nodeIndex)" src/ --include="*.gcl"
```
Initialize in the literal (`streets: nodeList<node<Street>>{}`) or declare nullable. Module-level `node*` collections auto-init; **type fields do not**.

### 2.2 Raw-string `throw` — **HIGH**
```bash
grep -rnE 'throw\s+"' src/ --include="*.gcl"
```
Define typed errors in `src/errors.gcl` (`AppError`/`NotFoundError`/`ValidationError`) and `throw NotFoundError { code, message, id }`. Always re-throw after an `error(...)` log.

### 2.3 Null-dereference chains
```bash
grep -rn "\.resolve()\..*\.\|->.*->" src/ --include="*.gcl"
```
Long `x.resolve().a.resolve().b` with no guards → cryptic runtime errors. Add null checks + typed `NotFoundError`, or optional chaining + `??`.

### 2.4 `!!` overuse — **MEDIUM**
```bash
grep -rn '!!' src/ --include="*.gcl"
```
`!!` is fine ONLY for global-registry lookups init guarantees (`Registry::get(k)!!`). For user data / untyped JSON, prefer `if (x == null) throw NotFoundError {...};`. Also flag **redundant** `!!` on a receiver already narrowed by `if (x != null)`.

### 2.5 Persistent collections used as locals — **MEDIUM (style; not lint-enforced)**
```bash
grep -rn "var.*= nodeList\|var.*= nodeIndex" src/ --include="*.gcl"
```
Inside a `fn` body, use `Array<T>` / `Map<K,V>`. (Deeper performance angle in Dimension 4.1.)

### 2.6 Node-collection element type — **LOW (design; not lint-enforced)**
Node tags store their payload **directly** — `nodeList<Item>`, `nodeIndex<String, User>`, `nodeGeo<Station>` are all correct and idiomatic. Wrapping in an extra `node<…>` (`nodeIndex<K, node<V>>`) is valid but only needed when the **same node is shared across multiple indices**. Flag only that sharing case — never "fix" a plain `nodeList<Item>`.

### 2.7 `@volatile` on API response types — **LOW (recommendation)**
Mark slim `…View` response DTOs `@volatile`. It guards the type against ever being written into the graph; it is *not* a persistence fix — returning a plain type from `@expose` does not persist it (persistence happens only on write into a `node<T>` / graph attribute).

---

## Dimension 3 — Type safety (beyond `greycat-lang lint`)

### 3.1 Collection type safety — **HIGH**
```bash
grep -rn "nodeList\|nodeIndex\|nodeTime\|nodeGeo" src/ --include="*.gcl" -B2 -A2
```
Module-level `var` ✅ · type field ✅ · local variable / function param / return ⚠ → use `Array`/`Map`.

### 3.2 Generic static functions — **MEDIUM (compiles; runtime erasure risk)**
```bash
grep -rnE 'static\s+fn\s+[a-z_][a-zA-Z0-9_]*<' src/ --include="*.gcl"
```
Generics are erased at runtime; an explicit `<T>` instantiation can crash inside `greycat test`. Remove `static` (instance method on a generic type) or monomorphise (`processInt`, `processString`).

### 3.3 Generic return types — **CRITICAL (crashes the test runner)**
```bash
grep -rnE 'fn.*:\s*(Array<)?[A-Z][a-zA-Z]*<.*<' src/ --include="*.gcl"
```
`Array<Wrapper<T>>` (or any nested generic return) crashes `greycat test`. Fix: non-generic result type carrying `value: any`, cast at the call site.

### 3.4 Anonymous return structs — **won't parse**
```bash
grep -rnE 'fn.*:\s*\{' src/ --include="*.gcl"
```
`fn foo(): { x: int }` does not parse. Declare a named (typically `@volatile`) type.

### 3.5 Precedence pitfalls
```bash
grep -rnE 'as +[A-Za-z_][A-Za-z0-9_]*\? +\?\?' src/ --include="*.gcl"                # cast + ??
grep -rnE '^var\s+[a-z_]+:\s*(int|float|String|bool|char|Array|Map)[^;]*;' src/ --include="*.gcl"  # module var
```
`x as T? ?? "d"` groups surprisingly → write `(x as T?) ?? "d"`. Module-level `var` must be a node-tag (`node<int?>`, `nodeIndex<K,V>`, `nodeList<T>`, `nodeTime<T>`, `nodeGeo<T>`) — wrap primitives: `var counter: node<int?>;`.

---

## Dimension 4 — Performance & concurrency

### 4.1 Unnecessary persistence — **MEDIUM (style; not lint-enforced)**
```bash
grep -rnE "var [a-z_][a-zA-Z0-9_]* = node(List|Index|Time|Geo)?<" src --include="*.gcl"
grep -rnE "fn [a-z_][a-zA-Z0-9_]*\(.*\).*: node(List|Index)" src --include="*.gcl"
```
Local vars / params / returns using node types → `T` / `Array<T>` / `Map<K,V>`. Auto-fix: `nodeList<node<T>>`→`Array<T>`, `nodeIndex<K,node<V>>`→`Map<K,V>`, `node<T>{obj}`→`obj` in local scope. (This is a style/efficiency issue, **not** CRITICAL — the lint does not flag it.)

### 4.2 Missing indices / O(n²) — **HIGH → CRITICAL on large datasets**
```bash
grep -rn "for.*in.*{" src --include="*.gcl" -A5 | grep -E "if.*->id ==|if.*== .*return"
```
Linear scan or nested loop matching on `==` id → add a `nodeIndex<K, node<V>>` for O(1) lookup. Maintain the index in the service's `create`/`update`/`delete`.

### 4.3 Expensive ops in loops / N+1 resolve — **HIGH**
```bash
grep -rn "for.*in.*{" src --include="*.gcl" -A10 | grep -E "\.resolve\(\)|->.*->"
```
Repeated `.resolve()` inside nested loops, or resolving a related node per iteration (1+N) → batch-resolve, embed in the view, or build a deduped map first.

### 4.4 Reimplemented natives — **MEDIUM**
```bash
grep -rnE "fn [a-z_]*sort|fn find_(max|min)|fn [a-z_]*join" src --include="*.gcl"
```
Custom sort → `items.sort_by(Item::field, SortOrder::desc)`. Custom min/max → the **global** `min(x,y)` / `max(x,y)` (there is no `Math::` namespace). Custom join → `Buffer` accumulation (no native `Array<String>.join` in current std).

### 4.5 Useless one-line wrappers — **LOW**
```bash
for f in $(find src -name "*.gcl"); do \
  awk '/^fn [a-z_].*\{$/{l=NR;h=$0;getline;if($0~/^\s*return .*::/)print FILENAME":"l": "h}' "$f"; done
```
`fn get_user(id) { return UserService::find_by_id(id); }` adds nothing — call directly.

### 4.6 Over-fetching in `@expose` responses
`@expose` returning full domain types (20+ fields) where the consumer needs 3 → define a lean `…View`. Smaller payloads also help frontend Lighthouse. (Cross-refs Dimension 5.)

### 4.7 Memory & storage patterns
- **String dedup — MEDIUM**: a `String` field repeated across many objects (`tag`, `category`, `source`, `status`) → `node<String>`; graph storage deduplicates.
  ```bash
  grep -rnE '^\s*(tag|tags|category|source|kind|status|label):\s*String;' src --include="*.gcl"
  ```
- **Multi-index sharing — HIGH**: indexing the same entity by two keys must store the **same** `node<T>` ref in both indices — two separate `node<Item>{item}` constructions double storage.
- **Re-import upsert / "orphan factory" — CRITICAL**: an importer that wipes a `nodeIndex`/`nodeTime`/`nodeGeo`/`nodeList` MUST reuse the prior `node<T>` per key. Constructing fresh nodes every run orphans the old ones and grows `gcdata/` unboundedly. See `/greycat:migrate` for the upsert pattern.
  ```bash
  grep -rnE 'fn (import|reload|ingest|reimport)' src --include="*.gcl" -A30 | grep -E 'node<[A-Z]'
  ```
- Run `./bin/greycat defrag` after large reshuffles (even with upsert).

### 4.8 Concurrency
- **`await(jobs)` without a task-backed entry path — HIGH**: an `@expose` HTTP call is already enqueued as a task, so `await(jobs)` **does** fan out over an HTTP POST (no `task: true` needed for this). The genuinely serial case is a one-shot `./bin/greycat run <fn>`. Flag `await(jobs)` in a function only ever invoked via one-shot `greycat run` — move that work to a task / `PeriodicTask`.
  ```bash
  grep -rnE 'await\s*\(' src --include="*.gcl"
  ```
- **`Array<Job<T>>` — CRITICAL**: crashes at runtime. Use `Array<Job>` and cast `jobs[i].result() as T` at the call site.
- **Batch discipline — MEDIUM**: batch `await` jobs in **~120**, never full worker count (leave ≥8 threads for OS). No nested `await`.
- **`node<T>{...}` inside parallel jobs — HIGH**: parallel jobs may only WRITE to pre-existing nodes — no constructors, no index instantiation, no edges to shared parents. Pre-allocate sequentially; global indices are read-only during the parallel phase.
- **`System::exec` as a parallelism workaround — HIGH**: a second `System::exec` in a non-task HTTP request throws uncatchable `"terminated PID X"`. Use the `task: true` header pattern.
- **"Stale await state"**: if a call-frame variable holds a node-backed value when `await` runs, you get `"wrong state before await..."` — isolate `await` in its own helper and build the result Array after it returns.

---

## Dimension 5 — `@expose` API review (security · performance · design)

### 5.1 Security — **CRITICAL / HIGH**
```bash
grep -rn "@expose" src/ --include="*.gcl" -B1 -A10
grep -rnE '@permission\("public"\)' src/ --include="*.gcl" -A5 | grep -E 'fn (create|update|delete|set|remove)'
```
- **Permission fit** — a bare `@expose` already requires `api` (authenticated) — that's the recommended default, so a *missing* `@permission` is **not** an error. Flag `@permission("public")` on **mutations** (anonymous write — CRITICAL), or a privileged op that should be `@permission("admin")`.
- **Sensitive-data exposure — CRITICAL** — returning a full `User` (with `password_hash`/tokens) leaks secrets. Return a `@volatile …View` without the sensitive fields.
- **Input validation — HIGH** — `String` params with no length/format checks (DoS / injection). Throw `ValidationError { code, message, field }`.

### 5.2 Performance
- **Missing pagination — HIGH** — list endpoints returning `Array<…>` with no `offset`/`limit` can return 10K+ items; return a `Paginated<View>` (`items/total/offset/limit/hasMore`).
- **Nested loops / N+1** — same as Dimension 4.2–4.3, but weighted higher inside an `@expose` because of the request TTL.

### 5.3 Error handling — **HIGH**
Every `@expose` body wraps in `try { … } catch (ex) { error("..."); throw ex; }`. Typed throws only (Dimension 2.2). Specific messages (`"Invalid email format. Must contain @"`, not `"Error"`).

### 5.4 Type & naming
- Return type is a `@volatile …View` (suffix `View`, not `Response`); no `Object`/`any` params — define a `@volatile` input type.
- Verb names (`getX`, `createX`, `searchX`); avoid bare `id`/`ids` params — prefer `documentId`, `partyIds` so the generated TS SDK is self-documenting (`query`/`email`/`caseNumber` are fine).
  ```bash
  grep -rnE 'fn [a-zA-Z_]+\([^)]*\b(id|ids):\s*' src/ --include="*.gcl"
  ```
- `///` docs (`@param`/`@return`/`@throws`) on every `@expose`. Bodies > 50 lines delegate to a service — the API layer is validation + delegation.
- **Misleading names** — `getData` that mutates → rename (`markProcessed`). Pick ONE not-found convention per project (nullable return OR throw), don't mix.

### 5.5 Routing & annotation hygiene
- `@permission(public)` (bare identifier) is a **CRITICAL compile error** — must be quoted `@permission("public")`.
  ```bash
  grep -rnE '@permission\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)' src/ --include="*.gcl"
  ```
- `@expose("path")` exposes at that **exact arbitrary path** (need not equal the fn name) — clients and the generated SDK must call the declared path. Flag code/docs assuming the default `<module>::<fn>` when a custom alias overrides it.
- **No bare `/<fn>` route.** Routes are `POST /<module>::<fn>` and `POST /<module>::<Type>::<fn>` (static fn on a type, e.g. `/runtime::Identity::current_id`); JSON-RPC at `POST /` uses `"<module>.<fn>"` (dots). `/runtime::` is a module name, not a routing prefix.
- `@role("name", "perm", …)` belongs in `project.gcl`.
- **20s TTL — HIGH if reachable** — the request TTL (`--request_ttl` flag, `serve` only) defaults to `20s`; past it the runtime kills the handler and **`try/catch` does NOT fire**. Raise `--request_ttl`, or dispatch as a background task (`task: true` header → returns `task_id`; poll `Task::is_running(id)` / `Task::running()` / `Task::history(offset, max)`; fetch `GET /files/<user_id>/tasks/<task_id>/result.gcb?json`).
  ```bash
  grep -rnE '@expose' src/ --include="*.gcl" -A30 | grep -E "for\s*\(.*in\s+|await\(|System::exec"
  ```

### 5.6 Frontend / LLM consumption
These endpoints feed the **Lit + Shoelace + `@greycat/web`** frontend via the generated client and back the `llms.txt` index. Return lean `@volatile …View` types; self-documenting params clarify both the TS SDK and `llms.txt`. After any type/`@expose` change, regenerate the client: `greycat codegen ts`. Full frontend review: `/greycat:frontend`.

---

## Dimension 6 — Test coverage

GreyCat ships no `--coverage` flag — coverage is by inventory + cross-reference, not instrumented runs.

### 6.1 Run the suite
```bash
rm -rf gcdata   # ⚠ destroys local graph data — dev only, never on a store you need
greycat test
```
Stale persistence causes startup failures and false results; wipe first. Tests in one module **share state** across `@test` fns (mutations visible to the next, not persisted). **Exit codes**: `0` success · `1` generic CLI error · `2` compile/load error (every test affected — coverage numbers unreliable). A segfault/kill invalidates the run even though there's no dedicated exit code.

**Single test** = a bare `@test` **function name** (`greycat test test_echo`), **not** a file path or `module::fn`. Omit to run all. Cross-module helpers go in `test/test_helpers.gcl` as plain `fn` (not `private`).

### 6.2 Find gaps
```bash
grep -rnE "^abstract type.*Service|^(static )?fn [a-z_]" src/ --include="*.gcl"   # service logic (abstract-type OR free fns)
grep -rn "@expose" src/ --include="*.gcl" -A2                                     # endpoints
grep -rn "^type [A-Z]" src/ --include="*.gcl"                                     # models
find test -name "*_test.gcl"
```
For each service fn / `@expose` / type method, check whether `test_<snake_case>` exists. **Risk = Complexity·20 + Usage·30 + Criticality·50** (Criticality: auth/payments/mutations/`@expose` = HIGH; search/filter/validation = MEDIUM; utils/formatters = LOW).

### 6.3 Suggest / generate tests
Per gap: file, function names, ready template, priority, rationale. Templates use `Assert::` (`equals`, `isTrue`, `isFalse`, `isNull`, `isNotNull`); optional `fn setup()` / `fn teardown()` (detected by name). Duplicate/uniqueness paths use `try { … } catch (ex) { failed = true; }`.
```gcl
@test fn test_create_user_valid() {
  var u = UserService::create("a@b.com", "pw", "user");
  Assert::isNotNull(u);
  Assert::equals(u->email, "a@b.com");
}
@test fn test_create_user_duplicate() {
  UserService::create("a@b.com", "p1", "user");
  var failed = false;
  try { UserService::create("a@b.com", "p2", "user"); } catch (ex) { failed = true; }
  Assert::isTrue(failed);
}
```
Write templates to `test/<feature>_test.gcl` with `TODO:` markers, then `greycat-lang lint` + `greycat test` to confirm they compile. (Frontend/Vitest + Lighthouse coverage lives in `/greycat:frontend`.)

---

## Dimension 7 — Best-practice gaps

- **Service layer** — business logic in a service (plain `fn`s or `abstract type XxxService`), not inline in the `@expose` API layer.
- **`@permission` default** — bare `@expose` = authenticated `api`; that's the intended default (re-stated from 5.1 for reviewers scanning this section).
- **Try/catch on every `@expose`** with an `error(...)` log + re-throw.
- **Lean `…View` types + codegen** — return slim `@volatile …View` DTOs; re-run `greycat codegen ts` after any backend type/`@expose` change (a stale ABI gives the frontend HTTP 422).
- **Docs** — `///` with `@param`/`@return`/`@throws` on all functions/types; `// ===` section headers.

---

## Cleanup (interactive)

After analysis, present the summary and offer options (AskUserQuestion):
- **A) Fix CRITICAL only** (recommended)
- **B) Fix all auto-fixable** (persistence, missing `@volatile`, wrappers, `Array<Job<T>>`→`Array<Job>`)
- **C) Report only**
- **D) Custom per-category**
- **E) Cancel**

For each applied fix:
1. Verify git status is clean (or make a checkpoint branch).
2. Apply Edits — **only changes that don't alter logic**.
3. `greycat-lang lint --fix` must pass.
4. `greycat test` after each cleanup batch.
5. Report counts (functions/types deleted, lines saved).

---

## Output

Group by severity. Per issue: `📍 file:line` + problem + fix snippet + estimated effort.

```
SUMMARY
  Dead code / duplication:   N
  Anti-patterns:             N
  Type-safety:               N
  Performance & concurrency: N
  @expose API:               N
  Test-coverage gaps:        N
  Best-practice gaps:        N

Endpoints analyzed: N   |   Tests: X/N covered
CRITICAL=A  HIGH=B  MEDIUM=C  LOW=D
```

---

## Notes

- Run when `greycat-lang lint` already passes; back up first (git checkpoint or `greycat backup`).
- Fix CRITICAL first; `greycat test` after each batch.
- Severity is calibrated to real impact: unnecessary persistence is MEDIUM (not lint-enforced), orphan-factory imports and `Array<Job<T>>` are CRITICAL (unbounded storage / runtime crash).
