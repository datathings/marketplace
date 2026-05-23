---
name: apicheck
description: Review all @expose endpoints for security, performance, and best practices
allowed-tools: Bash, Read, Grep, Glob
---

# API Endpoint Review

**Purpose**: Thorough review of all `@expose` endpoints across security, performance, error handling, type safety, naming, design, and routing.

**Run After**: Each sprint, before releases, when adding endpoints.

---

## Phase 1: Security

### 1.1 Permission decorator
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -B 1 -A 5
\`\`\`
For each `@expose`: verify `@permission` exists, level matches operation, no `"public"` on mutations.
**Severity**: CRITICAL when missing on sensitive ops.

### 1.2 Input validation
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep "fn.*String"
\`\`\`
Verify: length limits, format checks (email/URL), no SQL/cmd injection, no XSS. Throw `ValidationError { code, message, field }` on bad input.
**Severity**: HIGH for missing checks on String params (DoS / injection risk).

### 1.3 Sensitive data exposure
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep -E "User|Password|Token|Secret|Key"
\`\`\`
Returning full `User` (incl. `passwordHash`) leaks secrets. Fix: create `@volatile UserView` without sensitive fields.
**Severity**: CRITICAL.

### 1.4 Auth bypass on mutations
\`\`\`bash
grep -rn '@permission("public")' src/ --include="*_api.gcl" -A 10 | grep -E "fn (create|update|delete|set|modify|remove)"
\`\`\`
Mutations with `@permission("public")` are critical bugs. Require `app.user` or higher.

---

## Phase 2: Performance

### 2.1 Missing pagination
List endpoints returning `Array<...>` without `offset`/`limit` params can return 10K+ items.
Fix: return `Paginated<View>` with `items / total / offset / limit / hasMore`.
**Severity**: HIGH.

### 2.2 Expensive nested loops
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 50 | grep -E "for.*for"
\`\`\`
O(N*M) loops resolving thousands of nodes will TTL. Fix: cache (compute during import → store node), filter, or move to `PeriodicTask`.
**Severity**: HIGH.

### 2.3 N+1 query pattern
\`\`\`bash
grep -rn "for.*in.*" src/ --include="*_api.gcl" -A 5 | grep "\.resolve()"
\`\`\`
Resolving related nodes inside a loop = 1 + N queries. Fix: batch resolve, embed in view, or build a deduplicated map first.
**Severity**: HIGH.

---

## Phase 3: Error Handling

### 3.1 Missing try/catch on @expose
Every `@expose` must wrap body in `try { } catch (ex) { error(...); throw ex; }`. No exceptions.
**Severity**: MEDIUM (poor errors), HIGH for endpoints near 20s TTL.

### 3.2 Null pointer risks
\`\`\`bash
grep -rn "\.resolve()\." src/ --include="*_api.gcl"
\`\`\`
`x.resolve().author.resolve().name` with no checks → cryptic runtime errors. Fix: explicit null checks + typed `NotFoundError`, or optional chaining.

### 3.3 Untyped (string) throws
\`\`\`bash
grep -rnE 'throw\s+"' src/ --include="*_api.gcl"
\`\`\`
`throw "string"` has no structured shape; clients can't branch on code. Replace with `throw NotFoundError { code, message, id }` from `src/errors.gcl`.
**Severity**: HIGH.

### 3.4 Unhelpful messages
`throw "Error"` is useless. Be specific: `"Email is required"`, `"Invalid email format. Must contain @"`.

---

## Phase 4: Type Safety

### 4.1 Missing @volatile on response types
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 5 | grep "fn.*:" | sed 's/.*: //; s/ {.*//' | sort -u
\`\`\`
For each return type used by API: verify type has `@volatile` decorator. Without it, the type gets persisted unnecessarily.

### 4.2 `View` suffix convention
\`\`\`bash
grep -rnE '@volatile' src/ --include="*_api.gcl" -A 1 | grep -E 'type [A-Z][a-zA-Z]*[^V][^i][^e][^w]\b' || true
\`\`\`
API response types should end in `View` (`DocumentView`, not `DocumentResponse`).

### 4.3 Loose parameter types
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep -E ": Object|: any"
\`\`\`
Define specific `@volatile` input types instead of `Object`/`any`.

---

## Phase 5: Best Practices

### 5.1 Naming
Verbs only: `getX`, `createX`, `updateX`, `deleteX`, `searchX`, `listX`. No bare `document`/`data`/`process`.

### 5.2 Parameter naming — avoid bare `id`/`ids`
\`\`\`bash
grep -rnE 'fn [a-zA-Z_]+\([^)]*\b(id|ids):\s*' src/ --include="*_api.gcl"
\`\`\`
Use `documentId`, `partyIds` over bare `id`/`ids` so the generated TS SDK is self-documenting. (Exception: `caseNumber`, `query`, `email` etc. are fine.)

### 5.3 Documentation
Every `@expose` needs `///` with description, `@param`, `@return`, `@throws`.

### 5.4 Endpoint complexity
`@expose` functions > 50 lines should delegate to a `Service`. API layer = validation + delegation only.

---

## Phase 6: API Design

**Misleading names**: `getData` that mutates is a trap. Rename to `updateDataStatus` / `markAsProcessed`.

**Return consistency**: pick ONE pattern per project — nullable returns OR throw on not found. Don't mix.

---

## Phase 7: Routing & Annotations Hygiene

### 7.1 `@permission(public)` (identifier — invalid)
\`\`\`bash
grep -rnE '@permission\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)' src/ --include="*.gcl"
\`\`\`
Argument must be a String: `@permission("public")`.
**Severity**: CRITICAL — compile error.

### 7.2 `@expose("alias")` ≠ function name
\`\`\`bash
grep -rnE '@expose\("([^"]+)"\)' src/ --include="*_api.gcl" -A 2
\`\`\`
Aliases must match the function name (the TS SDK lookup in `project.d.ts` breaks silently otherwise).
**Severity**: HIGH.

### 7.3 Bare-route reliance
GreyCat exposes `@expose fn foo()` in `m.gcl` at `/m::foo` ALWAYS, with a bare `/foo` alias ONLY when unique project-wide. SDK consumers should always call `/<module>::<fn>` explicitly. `@expose static fn` inside `abstract type T` is reachable ONLY at `/<module>::T::fn` (no bare alias, no `/runtime::` synonym).
**Severity**: MEDIUM.

### 7.4 `@role` declared in `project.gcl`?
\`\`\`bash
grep -n '@role' project.gcl
\`\`\`
Roles `@role("name", "perm1", ...)` belong in `project.gcl`.

### 7.5 Long endpoints vs 20s TTL
\`\`\`bash
grep -rnE '@expose' src/ --include="*_api.gcl" -A 30 | grep -E "for\s*\(.*in\s+|await\(|System::exec"
\`\`\`
`GREYCAT_REQUEST_TTL` defaults to 20s. Past it, the runtime kills the handler — **`try/catch` does NOT catch it**. Raise the env var, or move to task (`task:''` header + poll `/runtime::Task::info`).
**Severity**: HIGH if endpoint can exceed 20s.

---

## Output

Group findings by severity. For each: `📍 file:line` + severity + problem + recommendation.

**Executive summary**:
```
Endpoints analyzed: N
CRITICAL: X (auth bypass, secrets, compile errors)
HIGH:     Y (pagination, N+1, untyped throws, TTL)
MEDIUM:   Z (@volatile, input validation, complexity)
LOW:      W (naming, docs)
```
