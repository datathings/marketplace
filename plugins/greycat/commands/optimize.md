---
name: optimize
description: Detect and auto-fix performance anti-patterns in GreyCat code
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion
---

# GreyCat Performance Optimizer

**Purpose**: Detect + auto-fix performance anti-patterns — unnecessary persistence, reimplemented natives, useless wrappers, complexity, duplication, memory/storage patterns, concurrency.

**Run When**: Quick performance checks, before releases, when performance degrades.

---

## Step 1: Scan
\`\`\`bash
find src -name "*.gcl" -type f | sort
\`\`\`

---

## Phase 1: Unnecessary Persistence

**Rule**: `node<T>`, `nodeList`, `nodeIndex`, `nodeTime`, `nodeGeo` only for module-level vars or type fields needing persistence. **Local vars, function params/returns** → `T`, `Array<T>`, `Map<K,V>`.

\`\`\`bash
# Local vars with node types
grep -rn "var [a-z_][a-zA-Z0-9_]* = node\(List\|Index\|Time\|Geo\)?<" src --include="*.gcl"
# Function params with node types
grep -rn "fn [a-z_][a-zA-Z0-9_]*(.*node\(List\|Index\|Time\|Geo\)<" src --include="*.gcl"
# Function returns nodeList/nodeIndex
grep -rn "fn [a-z_][a-zA-Z0-9_]*(.*).*: node\(List\|Index\)" src --include="*.gcl"
\`\`\`

**Auto-fix**: `nodeList<node<T>>` → `Array<T>`; `nodeIndex<K, node<V>>` → `Map<K, V>`; `node<T>{obj}` → `obj` (local scope). Note: `greycat-analyzer` does NOT currently flag persistent collections used as locals — review by hand.
**Severity**: MEDIUM (style; not lint-enforced).

---

## Phase 2: Reimplemented Natives

\`\`\`bash
# Custom sort (nested loops)
grep -rn "fn [a-z_]*sort" src --include="*.gcl" -A 20 | grep -B 1 -A 15 "for.*for"
# Custom min/max
grep -rn "fn find_\(max\|min\|maximum\|minimum\)" src --include="*.gcl"
# Custom join
grep -rn "fn [a-z_]*join" src --include="*.gcl" -A 10
\`\`\`

**Replacements**:
- Custom sort → `items.sort_by(Item::field, SortOrder::asc)`
- Custom max → module-level `max(a, b)` (no `Math::max` in std)
- Custom join → manual loop with `Buffer.add` (no native `Array<String>.join` in std 8.0.370-dev)

**Severity**: MEDIUM.

---

## Phase 3: Useless Function Wrappers

\`\`\`bash
for file in $(find src -name "*.gcl"); do
  awk '/^fn [a-z_]/{l=NR;f=$0;getline;if($0~/return.*::/){getline;if($0~/^}/)print FILENAME":"l": "f}}' "$file"
done
\`\`\`

`fn get_user(id) { return UserService::find_by_id(id); }` — single-line forwarder. Remove and call directly.
**Severity**: MEDIUM (LOW priority).

---

## Phase 4: Algorithmic Complexity

### O(n²) nested loops
\`\`\`bash
grep -rn "for.*in.*{" src --include="*.gcl" -A 5 | grep "for.*in.*{" | head -50
\`\`\`
For each: check if inner loop has `if x->id == y->id` → suggest `nodeIndex` for O(1) lookup.

### Linear search where index exists
\`\`\`bash
grep -rn "for.*in.*{" src --include="*.gcl" -A 3 | grep -B 1 "if.*->id =="
\`\`\`

**Fix** — add index in model:
\`\`\`gcl
var orders_by_user_id: nodeIndex<int, nodeList<node<Order>>>;
// In service create():
var user_orders = orders_by_user_id.get(user_id);
if (user_orders == null) { user_orders = nodeList<node<Order>>{}; orders_by_user_id.set(user_id, user_orders); }
user_orders.add(order);
// Query:
for (user in all_users) {
  var orders = orders_by_user_id.get(user->id);
  if (orders != null) for (i, o in orders) { /* ... */ }
}
\`\`\`
**Severity**: CRITICAL on large datasets.

---

## Phase 5: Code Duplication

\`\`\`bash
# Duplicate function signatures
grep -rn "^fn [a-z_][a-zA-Z0-9_]*(" src --include="*.gcl" | awk -F: '{print $3}' | sort | uniq -d

# Repeated error message strings (3+)
grep -rn "throw \"" src --include="*.gcl" | awk -F'"' '{print $2}' | sort | uniq -c | sort -rn | awk '$1>2'
\`\`\`
Extract to shared error constants / helper functions.

---

## Phase 6: Memory & Storage Patterns

### 6.1 String dedup via `node<String>`
\`\`\`bash
grep -rnE '^\s*(tag|tags|category|source|kind|type_name|status|label):\s*String;' src --include="*.gcl"
\`\`\`
Same value across many objects (tags, categories, source identifiers) — use `node<String>` instead of `String`. Graph storage deduplicates automatically.
**Severity**: MEDIUM.

### 6.2 Multi-Index sharing
\`\`\`gcl
// ❌ Two distinct nodes for same entity
by_id.set(item.id, node<Item>{ item });
by_name.set(item.name, node<Item>{ item });

// ✅ One node, shared
var n = node<Item>{ item };
by_id.set(item.id, n);
by_name.set(item.name, n);
\`\`\`
**Severity**: HIGH (storage doubles per extra index).

### 6.3 Re-import without upsert ("orphan factory")
\`\`\`bash
grep -rnE 'fn (import|reload|ingest|reimport)' src --include="*.gcl" -A 30 | grep -E 'node<[A-Z]'
\`\`\`
Importers that wipe `nodeIndex`/`nodeTime`/`nodeGeo`/`nodeList` MUST reuse prior `node<T>` per key. See `/migrate` for upsert pattern.
**Severity**: CRITICAL on regularly-running importers (unbounded gcdata growth).

### 6.4 `greycat defrag` after major reshuffles
Even with upsert, large reshuffles benefit from `./bin/greycat defrag` to reclaim storage.

---

## Phase 7: Concurrency Anti-Patterns

### 7.1 `await(jobs)` in plain @expose
\`\`\`bash
grep -rnE 'await\s*\(' src --include="*_api.gcl"
\`\`\`
Parallel `await` only fires inside task context. Plain `curl POST` runs serially. Either:
- Spawn endpoint as task: `task:''` HTTP header
- Move to CLI fn: `./bin/greycat run compute`
**Severity**: HIGH.

### 7.2 `Array<Job<T>>` typed jobs
\`\`\`bash
grep -rnE 'Array<Job<' src --include="*.gcl"
\`\`\`
Crashes at runtime. Use `Array<Job>` + cast `jobs[i].result() as T`.
**Severity**: CRITICAL.

### 7.3 Batch sizes equal to worker count
Batch `await` jobs in **~120**, never full worker count. Leave ≥8 threads for OS. No nested `await`.
**Severity**: MEDIUM.

### 7.4 `node<T>{...}` constructors inside parallel jobs
\`\`\`bash
grep -rnE 'node<[A-Z][a-zA-Z]+>\s*\{' src --include="*.gcl"
\`\`\`
Inside parallel jobs: only WRITE to pre-existing nodes. No `node<T>{...}` constructors, no edges to shared parents, no index instantiation. Global indices read-only during parallel phases. Pre-allocate sequentially.
**Severity**: HIGH.

### 7.5 `System::exec` as parallelism workaround
\`\`\`bash
grep -rnE 'System::exec' src --include="*.gcl"
\`\`\`
Second `System::exec` in non-task HTTP request throws uncatchable `"terminated PID X"`. Use `task:''` header pattern instead.
**Severity**: HIGH.

---

## Step 8: Report + Apply Fixes

Summary:
```
Analyzed: N files
CRITICAL: X (persistence, complexity, orphan-factory imports, Array<Job<T>>)
HIGH:     Y (multi-index, parallel constructors, await in @expose)
MEDIUM:   Z (natives, string-dedup, batch sizes)
LOW:      W (wrappers, duplication)

Auto-fixable: N/M
```

Ask via AskUserQuestion:
- A) Fix critical only (Recommended)
- B) Fix all auto-fixable
- C) Show detailed report first
- D) Cancel

Apply Edit operations → `greycat-lang lint --fix` → report success / manual-review items.

---

## Notes

- Focus: quick automated wins
- Only applies changes that don't alter logic
- Complementary to `/greycat:backend` (full review)
