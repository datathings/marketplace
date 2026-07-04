---
name: coverage
description: Generate test coverage report and suggest new tests to implement
allowed-tools: Bash, Read, Grep, Glob, Write
---

# Test Coverage Analysis (manual)

**Purpose**: Identify untested code and suggest tests prioritized by risk. GreyCat does not currently ship a `--coverage` flag — analysis is by inventory + cross-reference, not instrumented runs.

**Run After**: Each sprint, before releases, when adding features.

---

## Phase 1: Run Tests

⚠ Wipe `gcdata/` before clean runs — stale persistence causes startup failures and false results. Tests in the same module share state across `@test` fns (mutations visible to next within module).

\`\`\`bash
rm -rf gcdata   # ⚠ destroys local graph data — dev only; never on a store you need
greycat test
\`\`\`

**Exit codes** (documented):
| Code | Meaning |
|------|---------|
| 0 | success — tests ran (or no tests found) |
| 1 | generic CLI error (missing file, bad option) |
| 2 | compile/load error — program not buildable (every test affected) |

If exit code is `2` (compile/load error), every test is affected and coverage numbers are unreliable. A process-level crash or timeout (segfault, killed) obviously invalidates the run too, even though GreyCat exposes no dedicated exit code for it.

**Single test**: `greycat test <test_fn_name>` — the arg is a `@test` function name (e.g. `greycat test test_echo`), not a `module::fn` path or file; omit to run all. **Cross-module helpers**: plain `fn` (not `private`) in `test/test_helpers.gcl`.

---

## Phase 2: Identify Gaps

### Find code to cover
\`\`\`bash
grep -r "^abstract type.*Service" src/ --include="*.gcl"       # services (abstract type convention)
grep -rnE "^(static )?fn [a-z_]" src/ --include="*.gcl"        # free top-level fns (service logic may be plain fns, not only abstract type *Service)
grep -r "@expose" src/ --include="*.gcl" -A 2                  # endpoints (@expose may live in src/api.gcl, not only *_api.gcl)
grep -r "^type [A-Z]" src/ --include="*.gcl"                   # models
\`\`\`

### Cross-reference with tests
\`\`\`bash
find test -name "*_test.gcl"
\`\`\`
For each service function / `@expose` / type method, check whether `test_<snake_case>` exists.

### Risk score (0-100)
```
Risk = Complexity·20 + Usage·30 + Criticality·50

Criticality:
  HIGH (1.0)   — auth, payments, data mutations, @expose
  MEDIUM (0.5) — search, filter, retrieval, validation
  LOW (0.0)    — utils, formatters, helpers, UI-only
```

---

## Phase 3: Suggest Tests

For each gap: test file, function names, ready-to-use template, priority, rationale.

### Templates

**Service function**:
\`\`\`gcl
@test fn test_create_user_valid_input() {
  var user = UserService::createUser("a@b.com", "pw", "user");
  Assert::isTrue(user != null);
  Assert::equals(user.email, "a@b.com");
}
@test fn test_create_user_duplicate_email() {
  UserService::createUser("a@b.com", "p1", "user");
  try { UserService::createUser("a@b.com", "p2", "user"); Assert::isFalse(true); }
  catch (ex) { Assert::isTrue(true); }
}
@test fn test_create_user_invalid_email() {
  try { UserService::createUser("invalid", "pw", "user"); Assert::isFalse(true); }
  catch (ex) { Assert::isTrue(true); }
}
\`\`\`

**API endpoint**:
\`\`\`gcl
@test fn test_search_valid_query() {
  var r = search("privacy", null, 0, 10);
  Assert::isTrue(r != null && r.results != null && r.total >= 0);
}
@test fn test_search_pagination() {
  var p1 = search("test", null, 0, 5);
  var p2 = search("test", null, 5, 5);
  Assert::isTrue(p1.results.size() <= 5 && p2.results.size() <= 5);
  if (p1.results.size() > 0 && p2.results.size() > 0) {
    Assert::isTrue(p1.results[0].id != p2.results[0].id);
  }
}
\`\`\`

**Edge cases**: boundary min/max, null handling, empty collections.

---

## Phase 3b: Frontend coverage (if `frontend/` exists)

The **Lit + Shoelace + Lucide** frontend is tested with **Vitest** (jsdom). Treat two things as coverage gates:
- **Unit/component tests** — data-transform helpers and Lit component rendering against fixture payloads:
  \`\`\`bash
  find frontend -name "*.test.ts" -o -name "*.spec.ts"   # inventory
  pnpm test                                               # vitest run
  \`\`\`
  Gaps: components with no render test, untested formatters/derive helpers, untested error/loading states.
- **Lighthouse as a non-functional gate** — serve, then `pnpm lighthouse:ci` (performance, accessibility, best-practices, seo). Treat any category < 90 as an open coverage item, same as a missing unit test.

---

## Output

```
COVERAGE REPORT
  Services:        tested X / N  (Y%)
  API endpoints:   tested X / N  (Y%)
  Test files:      M  (P functions)

HIGH gaps:    N items
MEDIUM gaps:  N items
LOW gaps:     N items
```

Per high gap: `📍 file:line` + risk score + criticality + complexity + usage + suggested test names + estimated effort.

---

## Workflow

1. Run analysis → report
2. Ask:
   - A) Generate templates for HIGH only (recommended)
   - B) Specific items
   - C) Report only
   - D) Generate all (HIGH + MEDIUM + LOW)
3. Use Write to create `test/xxx_test.gcl` templates with `TODO:` comments
4. `greycat-lang lint` then `greycat test` to verify templates compile

---

## Notes

- Some tests need DB data (setup/teardown / fixtures)
- Auth tests need auth context setup
- Generated templates include TODOs — user completes business logic
- Re-run after sprints to track improvement
