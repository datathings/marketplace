---
name: backend
description: Comprehensive backend review and cleanup for GreyCat projects
allowed-tools: Bash, Read, Grep, Glob, Edit, Write
---

# Backend Review & Cleanup

**Purpose**: Comprehensive analysis of GreyCat backend code to identify dead code, duplications, anti-patterns, and optimization opportunities

**Run After**: Each sprint, before releases, during refactoring sessions

---

## Overview

This command performs a multi-phase analysis:

1. **Phase 1**: Dead Code Detection
2. **Phase 2**: Code Duplication Analysis
3. **Phase 3**: Anti-Pattern Detection
4. **Phase 4**: Optimization Opportunities
5. **Phase 5**: GreyCat Best Practices Check
6. **Phase 6**: Cleanup (optional, with confirmation)

---

## Phase 1: Dead Code Detection

### Objective
Identify unused code that can be safely removed.

### Step 1.1: Find Unused Types

**A. Find All Type Definitions**:
```bash
# Find all types (excluding @volatile API response types initially)
grep -rn "^type [A-Z]" backend/src/ --include="*.gcl" | grep -v "@volatile"
```

**B. Check Usage**:
For each type found, search for references:
```bash
# Search for type name usage across codebase
grep -r "TypeName" backend/ --include="*.gcl" | grep -v "^type TypeName"
```

**C. Categorize**:
- **SAFE TO DELETE**: Type defined but never used (0 references outside definition)
- **KEEP**: Type used in other files
- **REVIEW**: Type only used in same file (might be helper)

**Output Format**:
```
📍 backend/src/model/old_model.gcl:42

❌ UNUSED TYPE: OldDataStructure
   References: 0
   Defined: Line 42
   Safe to delete: YES

   Impact: None (no usages found)
```

### Step 1.2: Find Unused Functions

**A. Find All Functions**:
```bash
# Find static functions (services)
grep -rn "static fn [a-z_]" backend/src/service/ --include="*.gcl"

# Find type methods
grep -rn "^\s*fn [a-z_]" backend/src/model/ --include="*.gcl"

# Find API endpoints
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 1
```

**B. Check Usage**:
For each function:
```bash
# Search for function calls
grep -r "FunctionName(" backend/ --include="*.gcl"
```

**C. Special Cases**:
- **@expose functions**: Keep even if no internal calls (called externally)
- **main()**: Keep (entry point)
- **@test functions**: Keep (test suite)

**Output Format**:
```
📍 backend/src/service/data/helper_service.gcl:156

❌ UNUSED FUNCTION: processOldFormat
   References: 0 call sites
   Type: Static function
   Safe to delete: YES

   Code:
   156: static fn processOldFormat(data: String): Result {
   157:     // ... 20 lines of unused logic
   177: }
```

### Step 1.3: Find Unused Global Variables

**A. Find Global Variables**:
```bash
# Find module-level variables
grep -rn "^var [a-z_]" backend/src/model/ --include="*.gcl"
```

**B. Check Usage**:
```bash
# Search for variable references
grep -r "variableName" backend/ --include="*.gcl"
```

**C. Important**:
- **Global indices** (nodeIndex, nodeList) might have low usage but serve as primary storage
- Check if variable is populated during import/init

**Output Format**:
```
📍 backend/src/model/cache.gcl:12

⚠️ UNUSED VARIABLE: temp_cache
   Type: nodeIndex<String, String>
   References: 1 (only definition)
   Safe to delete: REVIEW (might be populated dynamically)
```

### Step 1.4: Find Unused Imports

**Note**: GreyCat uses `@include` for files, not per-symbol imports. Check if entire files are unused.

```bash
# Files with no references from other files
for file in backend/src/**/*.gcl; do
    filename=$(basename "$file" .gcl)
    refs=$(grep -r "$filename" backend/ --include="*.gcl" | grep -v "$file" | wc -l)
    if [ $refs -eq 0 ]; then
        echo "Unused file: $file"
    fi
done
```

### Step 1.5: Find Commented Code

```bash
# Find large commented blocks (potential dead code)
grep -rn "^//.*fn \|^//.*type \|^//.*var " backend/ --include="*.gcl"
```

**Output Format**:
```
📍 backend/src/service/old_service.gcl:45-78

⚠️ COMMENTED CODE: 33 lines
   Content: Old implementation of processData function
   Recommendation: Delete if no longer needed, or document why kept
```

---

## Phase 2: Code Duplication Analysis

### Objective
Identify repeated code patterns that could be refactored.

### Step 2.1: Find Duplicated Logic Patterns

**Common Patterns to Detect**:

**A. Repeated Validation Logic**:
```bash
# Search for similar validation patterns
grep -rn "if.*== null.*return null" backend/ --include="*.gcl"
grep -rn "if.*\.size\(\) == 0.*return" backend/ --include="*.gcl"
```

**Example Output**:
```
📍 DUPLICATION: Null checks (8 occurrences)

Locations:
  - backend/src/service/user_service.gcl:23
  - backend/src/service/document_service.gcl:45
  - backend/src/service/search_service.gcl:67
  ... (5 more)

Pattern:
  if (input == null) {
      return null;
  }

Suggestion:
  Extract to validation utility:

  abstract type ValidationUtils {
      static fn requireNonNull<T>(value: T?, errorMsg: String): T {
          if (value == null) {
              throw errorMsg;
          }
          return value;
      }
  }

  Usage:
  var validated = ValidationUtils::requireNonNull(input, "Input required");
```

**B. Repeated Query Patterns**:
```bash
# Find similar loops over global indices
grep -rn "for.*in.*_by_id" backend/ --include="*.gcl" -A 3
```

**C. Repeated Type Conversions**:
```bash
# Find repeated mapping logic
grep -rn "\.map\|for.*in.*{" backend/ --include="*.gcl" -B 2 -A 5
```

### Step 2.2: Find Similar Type Definitions

Compare types with similar field structures:

```bash
# Extract all type definitions
grep -rn "^type [A-Z]" backend/src/model/ --include="*.gcl" -A 10
```

Analyze for:
- Types with >70% field overlap
- Potential inheritance opportunities
- Consolidation candidates

**Example Output**:
```
📍 SIMILAR TYPES: DocumentView and DocumentDetailView

DocumentView (backend/src/api/api_types.gcl:45):
  - id: String
  - title: String
  - date: time?
  - type: String

DocumentDetailView (backend/src/api/api_types.gcl:89):
  - id: String
  - title: String
  - date: time?
  - type: String
  - content: String      ← Only difference
  - sections: Array<SectionView>

Similarity: 80% (4/5 fields identical)

Suggestion:
  Consider making DocumentDetailView extend DocumentView,
  or create shared base type for common fields.
```

### Step 2.3: Find Copy-Pasted Code Blocks

Use heuristics to detect copy-pasted code:

1. Identical line sequences (>5 lines)
2. Similar variable names with different prefixes
3. Repeated error handling patterns

**Example Output**:
```
📍 DUPLICATED BLOCK: Error handling (identical in 3 locations)

Locations:
  - backend/src/api/user_api.gcl:67-74
  - backend/src/api/document_api.gcl:123-130
  - backend/src/api/search_api.gcl:45-52

Code (8 lines):
  try {
      // ... operation
  } catch (ex) {
      error("Operation failed: ${ex}");
      return ErrorResponse { message: "Internal error" };
  }

Suggestion:
  Extract to error handling utility:

  abstract type ErrorHandler {
      static fn handle<T>(operation: fn(): T, errorMsg: String): T {
          try {
              return operation();
          } catch (ex) {
              error("${errorMsg}: ${ex}");
              throw "Internal error";
          }
      }
  }
```

---

## Phase 3: Anti-Pattern Detection

### Objective
Identify GreyCat-specific anti-patterns and bad practices.

### Step 3.1: Incorrect Persistent Collection Usage

**Pattern**: Using nodeList/nodeIndex for temporary local variables

```bash
# Find local variable declarations with persistent types
grep -rn "var.*= nodeList\|var.*= nodeIndex" backend/ --include="*.gcl" -B 2 -A 2
```

**Analysis**:
- If variable is inside function body → **LIKELY WRONG** (use Array/Map)
- If variable is type field or global → **CORRECT** (persistent)

**Example Output**:
```
📍 backend/src/service/builder_service.gcl:34

⚠️ ANTI-PATTERN: Local variable using nodeList

Code:
  32: fn buildResults(items: Array<Item>): Array<Result> {
  33:     var results = nodeList<node<Result>> {};  ← WRONG
  34:     for (i, item in items) {
  35:         results.add(node<Result>{ ... });
  36:     }
  37:     return results;  ← Type mismatch (nodeList vs Array)
  38: }

Problem:
  - Using persistent nodeList for temporary local variable
  - Should use Array for non-persisted collections

Fix:
  var results = Array<node<Result>> {};

Impact: Performance overhead, unnecessary persistence
Priority: HIGH
```

### Step 3.2: Missing @volatile on API Response Types

**Pattern**: Types returned by @expose without @volatile

```bash
# Find @expose functions
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 5

# For each, check return type has @volatile
```

**Example Output**:
```
📍 backend/src/api/search_api.gcl:23

⚠️ ANTI-PATTERN: Missing @volatile on API response type

Function:
  @expose
  fn search(...): SearchResults {  ← Return type SearchResults
      ...
  }

Type Definition (backend/src/api/api_types.gcl:45):
  type SearchResults {  ← Missing @volatile decorator
      items: Array<ResultView>;
      total: int;
  }

Problem:
  - API response type should be @volatile (not persisted)
  - Missing @volatile causes unnecessary database storage

Fix:
  @volatile
  type SearchResults {
      items: Array<ResultView>;
      total: int;
  }

Priority: MEDIUM
```

### Step 3.3: Direct Object Storage in Collections

**Pattern**: Storing objects directly instead of node references

```bash
# Find nodeList/nodeIndex storing objects not nodes
grep -rn "nodeList<[^n].*>" backend/ --include="*.gcl"
grep -rn "nodeIndex<.*,\s*[^n].*>" backend/ --include="*.gcl"
```

**Example Output**:
```
📍 backend/src/model/data.gcl:12

⚠️ ANTI-PATTERN: Storing objects directly in nodeList

Code:
  var items: nodeList<Item>;  ← WRONG (should be nodeList<node<Item>>)

Problem:
  - nodeList should store node references, not objects directly
  - Objects should be wrapped in nodes for persistence

Fix:
  var items: nodeList<node<Item>>;

  // When adding:
  items.add(node<Item>{ Item { ... } });

Priority: HIGH (breaks persistence model)
```

### Step 3.4: Missing Collection Initialization

**Pattern**: Non-nullable collections not initialized in constructor

```bash
# Find type fields with collections
grep -rn "^\s*[a-z_]*:\s*\(Array\|Map\|nodeList\|nodeIndex\)" backend/src/model/ --include="*.gcl"
```

Check if initialization happens in object creation.

**Example Output**:
```
📍 backend/src/model/document.gcl:8

⚠️ ANTI-PATTERN: Non-nullable collection not initialized

Type:
  type Document {
      id: String;
      chunks: nodeList<node<Chunk>>;  ← Non-nullable, must initialize
  }

Usage (backend/src/service/import_service.gcl:45):
  var doc = node<Document>{ Document {
      id: "123",
      // chunks not initialized ← WILL FAIL at runtime
  }};

Problem:
  - Non-nullable collections must be initialized on object creation
  - Missing initialization causes runtime errors

Fix Option 1 (Make nullable):
  chunks: nodeList<node<Chunk>>?;

Fix Option 2 (Initialize):
  var doc = node<Document>{ Document {
      id: "123",
      chunks: nodeList<node<Chunk>> {}
  }};

Priority: HIGH (runtime error)
```

### Step 3.5: Incorrect Null Handling

**Pattern**: Missing null checks, incorrect optional chaining

```bash
# Find potential null dereference
grep -rn "\.resolve()\..*\..*\|->.*->.*" backend/ --include="*.gcl"
```

**Example Output**:
```
📍 backend/src/service/data_service.gcl:67

⚠️ ANTI-PATTERN: Potential null pointer dereference

Code:
  var name = user.resolve().profile.name;  ← Can fail if profile is null

Problem:
  - No null check on profile field
  - Will throw runtime error if profile is null

Fix:
  var name = user.resolve().profile?.name ?? "Unknown";

Priority: MEDIUM
```

---

## Phase 4: Optimization Opportunities

### Objective
Identify performance bottlenecks and inefficiencies.

### Step 4.1: Expensive Operations in Loops

```bash
# Find loops with potential expensive operations
grep -rn "for.*in.*{" backend/ --include="*.gcl" -A 10 | grep -E "\.resolve\(\)|\.get\(|->.*->"
```

**Example Output**:
```
📍 backend/src/service/query_service.gcl:89

⚠️ OPTIMIZATION: Repeated node resolution in loop

Code:
  for (i, doc_node in documents) {
      var doc = doc_node.resolve();      ← Resolving in every iteration
      var chunks = doc.chunks.resolve(); ← Nested resolution
      for (j, chunk_node in chunks) {
          var chunk = chunk_node.resolve();
          // process chunk
      }
  }

Problem:
  - Resolving nodes inside nested loops
  - Potential for hundreds/thousands of resolution calls

Optimization:
  // Batch resolve if possible
  // Or restructure query to minimize resolution depth

Estimated Impact: HIGH (if documents is large collection)
Priority: MEDIUM
```

### Step 4.2: Missing Indices on Frequent Queries

Analyze query patterns to suggest missing indices:

```bash
# Find linear searches that could use indices
grep -rn "for.*in.*if.*==.*return" backend/ --include="*.gcl"
```

**Example Output**:
```
📍 backend/src/service/user_service.gcl:34

⚠️ OPTIMIZATION: Linear search could use index

Code:
  fn getUserByEmail(email: String): User? {
      for (i, user_node in users) {
          var user = user_node.resolve();
          if (user.email == email) {
              return user;
          }
      }
      return null;
  }

Problem:
  - O(n) linear search through all users
  - Called frequently (23 call sites found)

Suggestion:
  Create email index in model:

  var users_by_email: nodeIndex<String, node<User>>;

  fn getUserByEmail(email: String): User? {
      var user_node = users_by_email.get(email);
      return user_node?.resolve();  // O(1) lookup
  }

Estimated Impact: HIGH (if users collection is large)
Priority: HIGH
```

### Step 4.3: Inefficient Loops

```bash
# Find loops that could use for-in syntax or better iteration
grep -rn "for (.*=.*;.*<.*\.size().*;.*++)" backend/ --include="*.gcl"
```

**Example Output**:
```
📍 backend/src/service/processor.gcl:56

⚠️ OPTIMIZATION: C-style loop could use for-in

Code:
  for (var i = 0; i < items.size(); i++) {
      var item = items[i];
      process(item);
  }

Optimization:
  for (i, item in items) {
      process(item);
  }

Benefits:
  - More readable
  - Potentially faster (no repeated .size() calls)
  - Idiomatic GreyCat style

Priority: LOW (readability improvement)
```

### Step 4.4: Unnecessary Data in API Responses

```bash
# Find @expose functions returning full objects
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 10
```

Check if response includes unnecessary fields.

**Example Output**:
```
📍 backend/src/api/document_api.gcl:45

⚠️ OPTIMIZATION: API returning too much data

Function:
  @expose
  fn searchDocuments(...): Array<Document> {
      // Returns full Document objects
  }

Problem:
  - Document has many fields (20+)
  - Search results only need: id, title, excerpt
  - Sending unnecessary data over network

Suggestion:
  Create lightweight view type:

  @volatile
  type DocumentSearchResult {
      id: String;
      title: String;
      excerpt: String;
      score: float;
  }

  fn searchDocuments(...): Array<DocumentSearchResult> {
      // Map to view
  }

Estimated Impact: MEDIUM (network payload reduction)
Priority: MEDIUM
```

---

## Phase 5: GreyCat Best Practices Check

### Step 5.1: Service Layer Organization

Check for proper service abstraction:

```bash
# Find abstract types (services)
grep -rn "^abstract type.*Service" backend/src/service/ --include="*.gcl"
```

Verify:
- Services use `abstract type` pattern
- Static functions for operations
- No business logic in API layer

**Example Output**:
```
✓ GOOD: Proper service organization

Services found:
  - UserService (backend/src/service/auth/user_service.gcl)
  - DocumentService (backend/src/service/data/document_service.gcl)
  - SearchService (backend/src/service/search/search_service.gcl)

Pattern compliance:
  ✓ All use abstract type
  ✓ All static functions
  ✓ Clear separation from API layer
```

### Step 5.2: Global Index Definition Order

Global indices must be defined before types that use them:

```bash
# Check definition order in model files
grep -rn "^var\|^type" backend/src/model/ --include="*.gcl"
```

**Example Output**:
```
⚠️ WARNING: Index defined after type that uses it

File: backend/src/model/data.gcl

Line 10: type Document {
Line 15:     // references documents_by_id
Line 20: }
Line 25: var documents_by_id: nodeIndex<String, node<Document>>;  ← Defined after type

Problem:
  - Global variables should be defined before types
  - May cause initialization issues

Fix:
  Move variable definition to top of file (before type definitions)

Priority: MEDIUM
```

### Step 5.3: API Permission Checks

Verify all @expose functions have @permission decorator:

```bash
# Find @expose without @permission
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 1 | grep -v "@permission"
```

**Example Output**:
```
⚠️ WARNING: @expose function missing @permission

Function: backend/src/api/admin_api.gcl:34
  @expose
  fn deleteUser(userId: String): bool {  ← Missing @permission
      ...
  }

Problem:
  - No permission check on sensitive operation
  - Function is publicly accessible

Fix:
  @expose
  @permission("app.admin")  ← Add appropriate permission
  fn deleteUser(userId: String): bool {
      ...
  }

Priority: HIGH (security issue)
```

### Step 5.4: Error Handling at API Boundaries

Check for proper error handling in API functions:

```bash
# Find @expose functions without try-catch
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 20 | grep -v "try\|catch"
```

---

## Phase 6: Cleanup (Interactive)

### Objective
Optionally clean up detected issues with user confirmation.

### Step 6.1: Present Cleanup Options

After analysis, present summary:

```
===============================================================================
CLEANUP OPTIONS
===============================================================================

Found Issues:
  [ ] 12 unused functions (safe to delete)
  [ ] 5 unused types (safe to delete)
  [ ] 3 unused global variables (needs review)
  [ ] 8 duplicated code blocks (extract to utilities)
  [ ] 15 anti-patterns (requires code changes)
  [ ] 23 optimization opportunities

What would you like to do?

A) Delete unused code (functions + types)
B) Fix anti-patterns (persistence issues, missing @volatile)
C) Show detailed report only (no changes)
D) Custom selection (I'll ask for each category)
```

### Step 6.2: Delete Unused Code (If Confirmed)

If user selects option A:

1. **Backup First**:
```bash
git status
# Ensure working directory is clean or create backup branch
```

2. **Delete Unused Functions**:
```bash
# For each unused function, use Edit tool to remove
# Example:
# Edit file: backend/src/service/helper.gcl
# Remove lines 45-67 (unused function processOldFormat)
```

3. **Delete Unused Types**:
```bash
# For each unused type, use Edit tool to remove
```

4. **Verify After Deletion**:
```bash
greycat-lang lint
# Ensure no errors introduced
```

5. **Report**:
```
✓ Cleanup Complete

Deleted:
  - 12 unused functions (saved ~450 lines)
  - 5 unused types (saved ~120 lines)

Total reduction: 570 lines

Next steps:
  1. Run: greycat-lang lint (should pass)
  2. Run: greycat test (verify no broken tests)
  3. Commit changes: git commit -m "Clean up dead code"
```

### Step 6.3: Fix Anti-Patterns (If Confirmed)

If user selects option B:

For each anti-pattern, apply automated fix:

**Example: Fix persistent collection usage**:
```bash
# Find: var results = nodeList<T> {};
# Replace: var results = Array<T> {};
```

**Example: Add @volatile to API response types**:
```bash
# Find type definition for API response
# Add @volatile decorator before type
```

Report each fix applied and verify with lint.

---

## Output Format

### Executive Summary

```
===============================================================================
BACKEND REVIEW & CLEANUP REPORT
===============================================================================
Project: backend/src/
Files Analyzed: 45 .gcl files
Analysis Date: 2024-01-15

SUMMARY:
  Dead Code:           17 items (12 functions, 5 types)
  Duplications:        8 code blocks
  Anti-Patterns:       15 issues
  Optimizations:       23 opportunities
  Best Practice Gaps:  6 issues

PRIORITY BREAKDOWN:
  HIGH Priority:    18 issues (security, runtime errors, major performance)
  MEDIUM Priority:  24 issues (performance, maintainability)
  LOW Priority:     19 issues (readability, minor optimizations)

TOTAL ISSUES:       61

ESTIMATED CLEANUP EFFORT:
  Quick Wins:   2 hours  (delete dead code, fix anti-patterns)
  Refactoring:  1 day    (extract duplications)
  Optimization: 2 days   (add indices, restructure queries)

===============================================================================
```

### Detailed Report (Sample)

```
===============================================================================
HIGH PRIORITY ISSUES
===============================================================================

1. SECURITY: Missing @permission on admin function
   📍 backend/src/api/admin_api.gcl:34
   Priority: HIGH
   Effort: 5 minutes
   [Details above]

2. ANTI-PATTERN: Local variable using nodeList
   📍 backend/src/service/builder.gcl:67
   Priority: HIGH
   Effort: 2 minutes
   [Details above]

3. PERFORMANCE: Missing index on frequent query
   📍 backend/src/service/user_service.gcl:34
   Priority: HIGH
   Effort: 15 minutes
   [Details above]

...

===============================================================================
MEDIUM PRIORITY ISSUES
===============================================================================
...

===============================================================================
```

---

## Success Criteria

✓ **All backend files scanned** (.gcl files in backend/src/)
✓ **Dead code identified** with safe-to-delete classification
✓ **Duplications detected** with refactoring suggestions
✓ **Anti-patterns found** with fix recommendations
✓ **Optimizations suggested** with impact estimates
✓ **Best practices checked** against GreyCat guidelines
✓ **Report generated** with prioritized action items
✓ **Lint passes** after cleanup (if cleanup performed)

---

## Notes

- **Run in Clean State**: Ensure `greycat-lang lint` passes before running
- **Backup Recommended**: Use git to create safety checkpoint
- **Incremental Cleanup**: Fix HIGH priority issues first
- **Test After Changes**: Run `greycat test` after cleanup
- **Review Automated Fixes**: Some patterns need manual review

---

## Example Workflow

```bash
# 1. Run backend review
/backend

# 2. Review report (61 issues found)

# 3. Choose cleanup option
# → A) Delete unused code

# 4. Cleanup executes
# - 12 functions deleted
# - 5 types removed
# - 570 lines saved

# 5. Verify
greycat-lang lint  # ✓ Passes
greycat test       # ✓ All tests pass

# 6. Commit
git add backend/
git commit -m "Backend cleanup: remove dead code, fix anti-patterns"

# 7. Address remaining issues
# - Fix HIGH priority anti-patterns (15 min)
# - Extract duplicated logic (1 hour)
# - Add missing indices (30 min)

# 8. Re-run review to verify improvements
/backend
# → Issues reduced from 61 to 23
```
