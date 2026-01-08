---
name: typecheck
description: Advanced type safety checks beyond greycat-lang lint
allowed-tools: Bash, Read, Grep, Glob
---

# Type Safety Checker

**Purpose**: Catch type safety issues and anti-patterns that greycat-lang lint doesn't detect

**Run After**: Each sprint, before releases, when modifying type definitions

---

## Overview

This command performs advanced type safety analysis in 5 categories:

1. **Volatile Checking** - @volatile on API response types
2. **Collection Type Safety** - Persistent vs non-persistent collections
3. **Null Safety** - Potential null pointer issues
4. **Type Consistency** - Proper type usage patterns
5. **Generic Types** - Static functions with generics (not allowed)

---

## Phase 1: Volatile Decorator Checking

### Objective
Ensure all API response types are marked @volatile.

### Step 1.1: Find @expose Functions

```bash
# Extract all @expose functions and their return types
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 5 | grep "fn.*:" | sed 's/.*: //' | sed 's/ {.*//'
```

### Step 1.2: Check Each Return Type

For each return type found:
1. Check if it's a basic type (int, String, bool, float) → OK
2. Check if it's an Array/Map → Check element type
3. Check if it's a custom type → Verify has @volatile

### Example Output

```
📍 backend/src/api/search_api.gcl:63

⚠️ MISSING @volatile: API response type not marked as volatile

Function:
  @expose
  fn semanticSearch(...): PaginatedResult<SearchResultView> {

Type Definition (backend/src/api/api_types.gcl:45):
  type PaginatedResult<T> {  ← Missing @volatile
      items: Array<T>;
      total: int;
  }

Issue:
  - API response type should be @volatile (not persisted)
  - Missing decorator causes unnecessary database storage

Fix:
  @volatile
  type PaginatedResult<T> {
      items: Array<T>;
      total: int;
  }

Priority: MEDIUM
Impact: Database bloat, performance
```

---

## Phase 2: Collection Type Safety

### Objective
Detect misuse of persistent collections (nodeList/nodeIndex/nodeTime/nodeGeo) in temporary contexts.

### Step 2.1: Find All Collection Usages

```bash
# Find all persistent collection declarations
grep -rn "nodeList\|nodeIndex\|nodeTime\|nodeGeo" backend/src/ --include="*.gcl" -B 3 -A 3
```

### Step 2.2: Classify Usage Context

For each occurrence, determine context:

**✅ LEGITIMATE (Global variable)**:
```gcl
// In backend/src/model/data.gcl (module-level)
var documents: nodeList<node<Document>>;
```

**✅ LEGITIMATE (Type field)**:
```gcl
type Document {
    chunks: nodeIndex<node<Chunk>, bool>;
}
```

**⚠️ SUSPICIOUS (Local variable)**:
```gcl
fn buildResults() {
    var results = nodeList<T> {};  ← Should be Array
}
```

**⚠️ SUSPICIOUS (Function parameter)**:
```gcl
fn process(items: nodeList<T>) {  ← Should be Array
    // ...
}
```

### Example Output

```
📍 backend/src/service/builder.gcl:34

⚠️ TYPE SAFETY: Local variable using persistent collection

Code:
  32: fn buildResults(items: Array<Item>): Array<Result> {
  33:     var results = nodeList<node<Result>> {};
  34:     for (i, item in items) {
  35:         results.add(node<Result>{ ... });
  36:     }
  37:     return results;  ← Type error!
  38: }

Problem:
  - Using nodeList for temporary local variable
  - Function returns Array but creates nodeList
  - Unnecessary persistence overhead

Fix:
  var results = Array<node<Result>> {};

Priority: HIGH
Impact: Type mismatch, performance overhead
```

---

## Phase 3: Null Safety Analysis

### Objective
Find potential null pointer dereferences and missing null checks.

### Step 3.1: Find Chained Dereferences

```bash
# Find .resolve() chains without null checks
grep -rn "\.resolve()\..*\." backend/src/ --include="*.gcl"

# Find arrow operator chains
grep -rn "->.*->" backend/src/ --include="*.gcl"
```

### Step 3.2: Analyze Safety

Check if null checking exists before dereference:
- Look for `if (x == null)` before usage
- Look for optional chaining `?.`
- Look for null assertion `!!`

### Example Output

```
📍 backend/src/service/document_service.gcl:67

⚠️ NULL SAFETY: Potential null pointer dereference

Code:
  65: fn getDocumentAuthor(docId: String): String {
  66:     var doc_node = documents_by_id.get(docId);
  67:     return doc_node.resolve().author.resolve().name;
  68: }

Problem:
  - No null check on doc_node (docId might not exist)
  - No null check on author (document might not have author)
  - Will throw runtime error if any step fails

Fix Option 1 (Null checks):
  fn getDocumentAuthor(docId: String): String {
      var doc_node = documents_by_id.get(docId);
      if (doc_node == null) {
          throw "Document not found: ${docId}";
      }
      var doc = doc_node.resolve();
      if (doc.author == null) {
          throw "Document has no author";
      }
      return doc.author.resolve().name;
  }

Fix Option 2 (Optional chaining):
  fn getDocumentAuthor(docId: String): String? {
      return documents_by_id.get(docId)?.resolve().author?.resolve().name;
  }

Priority: MEDIUM
Impact: Runtime errors
```

---

## Phase 4: Type Consistency

### Objective
Ensure proper type usage patterns across codebase.

### Step 4.1: Check Collection Initialization

Find non-nullable collections that might not be initialized:

```bash
# Find type fields with collections
grep -rn "^\s*[a-z_]*:\s*\(Array\|Map\|nodeList\|nodeIndex\)" backend/src/model/ --include="*.gcl"
```

Check if they're nullable or always initialized.

### Example Output

```
📍 backend/src/model/document.gcl:12

⚠️ TYPE SAFETY: Non-nullable collection not always initialized

Type:
  type Document {
      id: String;
      chunks: nodeList<node<Chunk>>;  ← Non-nullable
  }

Usage (backend/src/service/import.gcl:45):
  var doc = Document {
      id: "123"
      // chunks not initialized ← RUNTIME ERROR
  };

Problem:
  - Non-nullable collection must be initialized on creation
  - Missing initialization causes runtime error

Fix Option 1 (Make nullable):
  chunks: nodeList<node<Chunk>>?;

Fix Option 2 (Always initialize):
  var doc = Document {
      id: "123",
      chunks: nodeList<node<Chunk>> {}
  };

Priority: HIGH
Impact: Runtime errors
```

### Step 4.2: Check Node Storage

Find collections storing objects instead of node references:

```bash
# Find nodeList/nodeIndex not storing nodes
grep -rn "nodeList<[^n]" backend/src/ --include="*.gcl"
grep -rn "nodeIndex<.*,\s*[^n]" backend/src/ --include="*.gcl"
```

### Example Output

```
📍 backend/src/model/data.gcl:8

⚠️ TYPE SAFETY: Storing objects directly in persistent collection

Code:
  var items: nodeList<Item>;  ← Should be nodeList<node<Item>>

Problem:
  - nodeList should store node references, not objects
  - Direct object storage breaks persistence model

Fix:
  var items: nodeList<node<Item>>;

  // When adding:
  items.add(node<Item>{ Item { ... } });

Priority: HIGH
Impact: Persistence errors
```

---

## Phase 5: Generic Type Safety

### Objective
Detect generic type parameters on static functions (not allowed in GreyCat).

### Step 5.1: Find Static Functions with Generics

```bash
# Find static fn with <T> syntax
grep -rn "static fn.*<.*>.*(" backend/src/ --include="*.gcl"
```

### Example Output

```
📍 backend/src/service/utils.gcl:12

❌ TYPE ERROR: Generic type parameter on static function

Code:
  abstract type Utils {
      static fn process<T>(value: T): T {  ← NOT ALLOWED
          return value;
      }
  }

Problem:
  - Generic type parameters (<T>) not allowed on static functions in GreyCat
  - Will cause compilation error

Fix Option 1 (Remove static):
  type Utils<T> {
      fn process(value: T): T {
          return value;
      }
  }
  // Usage: Utils<int>{}.process(42)

Fix Option 2 (Make concrete):
  abstract type Utils {
      static fn processInt(value: int): int { ... }
      static fn processString(value: String): String { ... }
  }

Priority: CRITICAL
Impact: Compilation error
```

---

## Output Format

### Executive Summary

```
===============================================================================
GREYCAT TYPE SAFETY ANALYSIS
===============================================================================

Files Analyzed: 45 .gcl files
Analysis Date: 2026-01-08

ISSUES FOUND:

CRITICAL (Compilation Errors):
  [ ] 2 static functions with generic type parameters

HIGH (Runtime Errors):
  [ ] 5 uninitialized non-nullable collections
  [ ] 4 incorrect node storage in persistent collections
  [ ] 3 local variables using persistent types

MEDIUM (Safety Issues):
  [ ] 8 missing @volatile on API response types
  [ ] 6 potential null pointer dereferences
  [ ] 4 unsafe type casts

LOW (Best Practices):
  [ ] 12 missing type annotations (could be inferred)
  [ ] 7 overly broad types (Object vs specific)

TOTAL ISSUES: 51

ESTIMATED FIX TIME:
  Critical: 30 minutes  (fix immediately)
  High:     2 hours     (fix this sprint)
  Medium:   3 hours     (fix next sprint)
  Low:      1 hour      (nice to have)

===============================================================================
```

### Detailed Report

```
===============================================================================
CRITICAL ISSUES (Fix Immediately)
===============================================================================

1. COMPILATION ERROR: Generic static function
   📍 backend/src/service/utils.gcl:12
   [Details above]

2. COMPILATION ERROR: Generic static function
   📍 backend/src/service/helper.gcl:34
   [Details above]

===============================================================================
HIGH PRIORITY ISSUES
===============================================================================

1. RUNTIME ERROR: Uninitialized collection
   📍 backend/src/model/document.gcl:12
   [Details above]

...

===============================================================================
```

---

## Verification Commands

After fixes, verify:

```bash
# 1. Run greycat-lang lint (should pass)
greycat-lang lint

# 2. Run this type check again
/typecheck

# 3. Run tests
greycat test

# 4. Build project
greycat build
```

---

## Success Criteria

✓ **All files scanned** (.gcl files in backend/src/)
✓ **Volatile decorators checked** on API response types
✓ **Collection usage validated** (persistent vs non-persistent)
✓ **Null safety analyzed** (potential NPEs)
✓ **Type consistency verified** (initialization, node storage)
✓ **Generic types checked** (no generics on static functions)
✓ **Report generated** with prioritized issues
✓ **Zero CRITICAL issues** before release

---

## Notes

- **Complements greycat-lang lint**: Catches issues the linter misses
- **Run regularly**: After type changes, before releases
- **Fix CRITICAL first**: These will cause compilation errors
- **Test after fixes**: Always run tests after type changes
- **Non-blocking**: MEDIUM/LOW can be addressed incrementally

---

## Example Workflow

```bash
# 1. Run type safety check
/typecheck

# 2. Review report (51 issues found)
# - 2 CRITICAL
# - 12 HIGH
# - 18 MEDIUM
# - 19 LOW

# 3. Fix CRITICAL immediately
# - Remove generic params from static functions

# 4. Verify fix
greycat-lang lint  # ✓ Passes
greycat build      # ✓ Builds successfully

# 5. Fix HIGH priority issues
# - Initialize collections
# - Fix node storage
# - Replace nodeList with Array in local vars

# 6. Re-run check
/typecheck
# → CRITICAL: 0
# → HIGH: 0
# → MEDIUM: 18 (to address next sprint)

# 7. Commit
git add backend/
git commit -m "fix: resolve type safety issues (CRITICAL+HIGH)"
```

---

## Integration with Other Commands

**Use Before**:
- `/apicheck` - Fix type issues before API review
- `/backend` - Type check before general cleanup
- `/docs` - Ensure types are correct before documentation

**Use After**:
- Major refactoring
- Type definition changes
- Library updates (might introduce type issues)

---

## Common Patterns

### Pattern 1: API Response Type Missing @volatile

**Detection**: Function returns custom type, type lacks @volatile
**Fix**: Add @volatile decorator to type
**Frequency**: Common in new API endpoints

### Pattern 2: Local Variable Using nodeList

**Detection**: Variable declared inside function with nodeList type
**Fix**: Change to Array
**Frequency**: Very common antipattern

### Pattern 3: Uninitialized Collection

**Detection**: Non-nullable collection field, object created without initializing it
**Fix**: Either make nullable or always initialize
**Frequency**: Common in model types

### Pattern 4: Null Dereference Chain

**Detection**: Multiple .resolve() or -> without null checks
**Fix**: Add null checks or use optional chaining
**Frequency**: Common in service layer

### Pattern 5: Generic Static Function

**Detection**: static fn with <T> syntax
**Fix**: Remove static or remove generics
**Frequency**: Rare, but critical error
