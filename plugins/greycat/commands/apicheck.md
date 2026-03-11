---
name: apicheck
description: Review all @expose endpoints for security, performance, and best practices
allowed-tools: Bash, Read, Grep, Glob
---

# API Endpoint Review

**Purpose**: Comprehensive review of all @expose API endpoints for security, performance, best practices, and consistency

**Run After**: Each sprint, before releases, when adding new endpoints

---

## Overview

This command performs a thorough review of all @expose functions across 6 categories:

1. **Security** - Permission checks, input validation, sensitive data
2. **Performance** - Pagination, data volume, query efficiency
3. **Error Handling** - Try-catch blocks, null checks, user-friendly errors
4. **Type Safety** - @volatile decorators, return types, parameter types
5. **Best Practices** - Naming, documentation, consistency
6. **API Design** - REST principles, versioning, backwards compatibility

---

## Phase 1: Security Review

### Objective
Ensure all endpoints are secure and properly protected.

### Step 1.1: Permission Decorator Check

**Find @expose without @permission**:
```bash
# Find all @expose functions
grep -rn "@expose" src/ --include="*_api.gcl" -B 1 -A 5 | grep -A 5 -B 1 "@expose"
```

For each @expose function, verify:
- Has @permission decorator
- Permission level is appropriate for operation
- Permission is not too broad (avoid "public" for mutations)

**Output Format**:
```
📍 src/admin/admin_api.gcl:45

⚠️ SECURITY: Missing @permission decorator

Function:
  @expose
  fn deleteUser(userId: String): bool {  ← No @permission
      UserService::deleteUser(userId);
  }

Problem:
  - Sensitive operation (deletion) has no permission check
  - Function is accessible to all users (default public)
  - Should require admin permission

Recommendation:
  @expose
  @permission("app.admin")  ← Add this
  fn deleteUser(userId: String): bool {
      UserService::deleteUser(userId);
  }

Severity: CRITICAL
Risk: Unauthorized data deletion
```

### Step 1.2: Input Validation

Check for dangerous input patterns:

```bash
# Find endpoints that take String parameters (potential injection)
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep "fn.*String"
```

Verify each endpoint validates input:
- String length limits
- Format validation (email, URL, etc.)
- SQL/command injection prevention
- XSS prevention

**Example Output**:
```
📍 src/search/search_api.gcl:23

⚠️ SECURITY: Insufficient input validation

Function:
  @expose
  @permission("app.user")
  fn search(query: String, ...): SearchResults {
      // No validation on query length
      return SearchService::execute(query);  ← Potentially dangerous
  }

Problem:
  - No check on query string length
  - Could accept extremely long queries (DoS risk)
  - No sanitization before processing

Recommendation:
  @expose
  @permission("app.user")
  fn search(query: String, ...): SearchResults {
      // Validate input
      if (query.size() > 1000) {
          throw "Query too long (max 1000 characters)";
      }
      if (query.size() == 0) {
          throw "Query cannot be empty";
      }

      return SearchService::execute(query);
  }

Severity: HIGH
Risk: Denial of service, resource exhaustion
```

### Step 1.3: Sensitive Data Exposure

Check for endpoints returning sensitive information:

```bash
# Find endpoints returning User, Password, Token types
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep -E "User|Password|Token|Secret|Key"
```

Verify:
- Passwords never returned
- Sensitive fields filtered from responses
- Using view types instead of full objects

**Example Output**:
```
📍 src/user/user_api.gcl:67

⚠️ SECURITY: Potential sensitive data exposure

Function:
  @expose
  @permission("app.user")
  fn getCurrentUser(): User {  ← Returns full User object
      return SecurityService::requireLoggedUser();
  }

Type Definition:
  type User {
      username: String;
      email: String;
      passwordHash: String;  ← Sensitive!
      role: String;
  }

Problem:
  - Returns passwordHash field to client
  - Leaks sensitive authentication data

Recommendation:
  Create view type without sensitive fields:

  @volatile
  type UserView {
      username: String;
      email: String;
      role: String;
      // passwordHash excluded
  }

  @expose
  @permission("app.user")
  fn getCurrentUser(): UserView {
      var user = SecurityService::requireLoggedUser();
      return UserView {
          username: user.username,
          email: user.email,
          role: user.role
      };
  }

Severity: CRITICAL
Risk: Password hash exposure, security breach
```

### Step 1.4: Authentication Bypass

Find endpoints that should check authentication:

```bash
# Find mutation operations with public permission
grep -rn '@permission("public")' src/ --include="*_api.gcl" -A 10 | grep -E "fn (create|update|delete|set|modify|remove)"
```

**Example Output**:
```
📍 src/document/document_api.gcl:89

⚠️ SECURITY: Mutation allowed without authentication

Function:
  @expose
  @permission("public")  ← Should require auth!
  fn updateDocument(id: String, title: String): bool {
      var doc = documents_by_id.get(id);
      if (doc != null) {
          doc->title = title;
          return true;
      }
      return false;
  }

Problem:
  - Data mutation with public permission
  - No authentication required for modifying data
  - Allows anonymous users to edit documents

Recommendation:
  @expose
  @permission("app.user")  ← Require authentication
  fn updateDocument(id: String, title: String): bool {
      var user = SecurityService::requireLoggedUser();
      // ... update logic
  }

Severity: CRITICAL
Risk: Unauthorized data modification
```

---

## Phase 2: Performance Review

### Objective
Ensure endpoints perform efficiently and scale properly.

### Step 2.1: Missing Pagination

Find list endpoints without pagination:

```bash
# Find endpoints returning Array without pagination parameters
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep "Array<"
```

Check if endpoint has offset/limit parameters:

**Example Output**:
```
📍 src/document/document_api.gcl:34

⚠️ PERFORMANCE: Missing pagination on list endpoint

Function:
  @expose
  @permission("app.user")
  fn getAllDocuments(): Array<DocumentView> {  ← No pagination!
      var results = Array<DocumentView> {};
      for (i, doc_node in documents_by_id) {
          results.add(buildView(doc_node.resolve()));
      }
      return results;  ← Could return thousands of documents!
  }

Problem:
  - Returns all documents in single request
  - No limit on result size (could be 10K+ items)
  - Large payload impacts performance and memory

Recommendation:
  @expose
  @permission("app.user")
  fn getAllDocuments(offset: int, limit: int): PaginatedDocuments {
      var validated_offset = PaginationService::validateOffset(offset);
      var validated_limit = PaginationService::validateLimit(limit);

      var results = Array<DocumentView> {};
      var count = 0;
      var skipped = 0;

      for (i, doc_node in documents_by_id) {
          if (skipped < validated_offset) {
              skipped++;
              continue;
          }
          if (count >= validated_limit) {
              break;
          }
          results.add(buildView(doc_node.resolve()));
          count++;
      }

      return PaginatedDocuments {
          items: results,
          total: documents_by_id.size(),
          offset: validated_offset,
          limit: validated_limit,
          hasMore: (validated_offset + validated_limit) < documents_by_id.size()
      };
  }

Severity: HIGH
Risk: Performance degradation, memory issues, timeouts
```

### Step 2.2: Expensive Operations

Find endpoints with potentially expensive operations:

```bash
# Find nested loops in @expose functions
grep -rn "@expose" src/ --include="*_api.gcl" -A 50 | grep -E "for.*for"
```

**Example Output**:
```
📍 src/stats/stats_api.gcl:45

⚠️ PERFORMANCE: Expensive nested loops in endpoint

Function:
  @expose
  @permission("app.user")
  fn getDetailedStats(): StatsView {
      var stats = StatsView {};

      // Nested loop through ALL documents and chunks
      for (i, doc_node in documents_by_id) {      ← O(N)
          var doc = doc_node.resolve();
          for (j, chunk_node in doc.chunks) {      ← O(M)
              var chunk = chunk_node.resolve();    ← O(N*M)!
              // Process chunk...
          }
      }

      return stats;
  }

Problem:
  - O(N*M) complexity (documents × chunks per document)
  - Resolves potentially thousands of nodes
  - No caching or optimization
  - Will timeout with large datasets

Recommendation:
  Option 1: Cache results
  - Compute stats during import/update
  - Store in global variable
  - Return cached value instantly

  Option 2: Limit scope
  - Add filters to reduce dataset
  - Use indices for targeted queries
  - Implement pagination

  Option 3: Background computation
  - Use PeriodicTask to compute stats
  - Store results in nodes
  - API returns pre-computed data

Severity: HIGH
Risk: Timeouts, slow response times, resource exhaustion
```

### Step 2.3: N+1 Query Problem

Find endpoints that might have N+1 query issues:

```bash
# Look for loops with individual node resolutions
grep -rn "for.*in.*" src/ --include="*_api.gcl" -A 5 | grep "\.resolve()"
```

**Example Output**:
```
📍 src/document/document_api.gcl:78

⚠️ PERFORMANCE: N+1 query pattern

Function:
  @expose
  @permission("app.user")
  fn getDocumentsWithAuthors(): Array<DocumentWithAuthor> {
      var results = Array<DocumentWithAuthor> {};

      for (i, doc_node in documents_by_id) {
          var doc = doc_node.resolve();              ← Query 1
          var author = doc.author.resolve();         ← N more queries!
          results.add(DocumentWithAuthor {
              document: doc,
              authorName: author.name                ← Each iteration resolves author
          });
      }

      return results;
  }

Problem:
  - 1 query for documents + N queries for authors
  - With 1000 documents = 1001 database queries
  - Extremely inefficient

Recommendation:
  Option 1: Batch resolve (if supported)
  Option 2: Include author data in document view
  Option 3: Use index to get authors in one pass

  var authors_map = Map<String, Author> {};
  // First, collect all unique author nodes
  for (i, doc_node in documents_by_id) {
      var doc = doc_node.resolve();
      var author_id = doc.author->id;  // Use arrow without resolve
      if (authors_map.get(author_id) == null) {
          authors_map.set(author_id, doc.author.resolve());
      }
  }
  // Then, build results using cached authors
  for (i, doc_node in documents_by_id) {
      var doc = doc_node.resolve();
      var author = authors_map.get(doc.author->id);
      // ... build result
  }

Severity: HIGH
Risk: Slow queries, database load
```

---

## Phase 3: Error Handling Review

### Objective
Ensure all endpoints handle errors gracefully.

### Step 3.1: Missing Try-Catch

Find endpoints without error handling:

```bash
# Find @expose functions without try-catch
grep -rn "@expose" src/ --include="*_api.gcl" -A 30 | grep -L "try\|catch"
```

**Example Output**:
```
📍 src/data/data_api.gcl:56

⚠️ ERROR HANDLING: No try-catch block

Function:
  @expose
  @permission("app.user")
  fn processData(data: String): Result {
      var parsed = parseJSON(data);        ← Can throw
      var validated = validate(parsed);    ← Can throw
      var result = store(validated);       ← Can throw
      return result;
  }

Problem:
  - No error handling for parsing failures
  - No handling for validation errors
  - Exceptions bubble up as generic errors
  - Poor user experience (unclear error messages)

Recommendation:
  @expose
  @permission("app.user")
  fn processData(data: String): Result {
      try {
          var parsed = parseJSON(data);
          var validated = validate(parsed);
          var result = store(validated);
          return result;
      } catch (ex) {
          error("Data processing failed: ${ex}");
          throw "Invalid data format. Please check your input.";
      }
  }

Severity: MEDIUM
Risk: Poor error messages, difficult debugging
```

### Step 3.2: Null Pointer Risks

Find endpoints with potential null dereference:

```bash
# Find .resolve() calls without null checks
grep -rn "\.resolve()\." src/ --include="*_api.gcl"
```

**Example Output**:
```
📍 src/document/document_api.gcl:90

⚠️ ERROR HANDLING: Potential null pointer dereference

Function:
  @expose
  @permission("app.user")
  fn getDocumentAuthor(docId: String): String {
      var doc_node = documents_by_id.get(docId);
      return doc_node.resolve().author.resolve().name;  ← Multiple failure points!
  }

Problem:
  - No null check on doc_node (docId might not exist)
  - No null check on author (document might not have author)
  - Will throw cryptic error if any step fails

Recommendation:
  @expose
  @permission("app.user")
  fn getDocumentAuthor(docId: String): String {
      var doc_node = documents_by_id.get(docId);
      if (doc_node == null) {
          throw "Document not found: ${docId}";
      }

      var doc = doc_node.resolve();
      if (doc.author == null) {
          throw "Document has no author";
      }

      var author = doc.author.resolve();
      return author.name;
  }

  // Or use optional chaining:
  fn getDocumentAuthor(docId: String): String? {
      return documents_by_id.get(docId)?.resolve().author?.resolve().name;
  }

Severity: MEDIUM
Risk: Runtime errors, poor user experience
```

### Step 3.3: Insufficient Error Messages

Check for helpful error messages:

**Example Output**:
```
📍 src/user/user_api.gcl:45

⚠️ ERROR HANDLING: Unhelpful error message

Function:
  @expose
  @permission("app.admin")
  fn createUser(...): User {
      if (email.size() == 0) {
          throw "Error";  ← Generic, unhelpful
      }
      // ...
  }

Problem:
  - Error message is too generic
  - Doesn't explain what went wrong
  - Doesn't tell user how to fix it

Recommendation:
  if (email.size() == 0) {
      throw "Email address is required";  ← Clear and specific
  }

  if (!email.contains("@")) {
      throw "Invalid email format. Email must contain @";
  }

Severity: LOW
Risk: Poor user experience, support burden
```

---

## Phase 4: Type Safety Review

### Objective
Ensure proper use of types, especially @volatile for API responses.

### Step 4.1: Missing @volatile on Response Types

Find response types missing @volatile:

```bash
# Extract return types from @expose functions
grep -rn "@expose" src/ --include="*_api.gcl" -A 5 | grep "fn.*:" | sed 's/.*: //' | sed 's/ {.*//' | sort -u
```

For each type, check if it has @volatile decorator:

**Example Output**:
```
📍 src/search/search_api.gcl:45

⚠️ TYPE SAFETY: Missing @volatile on API response type

Type:
  type SearchResults {  ← Used in API, should be @volatile
      items: Array<ResultView>;
      total: int;
      offset: int;
  }

Used in:
  - src/search/search_api.gcl:23 (return type)

Problem:
  - API response type not marked as @volatile
  - Will be persisted unnecessarily
  - Wastes database storage

Recommendation:
  @volatile
  type SearchResults {
      items: Array<ResultView>;
      total: int;
      offset: int;
  }

Severity: MEDIUM
Risk: Unnecessary persistence, database bloat
```

### Step 4.2: Loose Parameter Types

Find endpoints with overly broad types:

```bash
# Find endpoints using Object, any, or dynamic types
grep -rn "@expose" src/ --include="*_api.gcl" -A 10 | grep -E ": Object|: any"
```

**Example Output**:
```
📍 src/generic/generic_api.gcl:34

⚠️ TYPE SAFETY: Overly broad parameter type

Function:
  @expose
  @permission("app.user")
  fn processData(data: Object): Result {  ← Too generic!
      // ... process
  }

Problem:
  - Parameter type is too broad (Object)
  - No type checking on input
  - Error-prone and hard to maintain

Recommendation:
  Define specific type for input:

  @volatile
  type DataInput {
      name: String;
      value: int;
      metadata: Map<String, String>?;
  }

  @expose
  @permission("app.user")
  fn processData(data: DataInput): Result {
      // Now type-safe
  }

Severity: MEDIUM
Risk: Runtime errors, maintenance difficulty
```

---

## Phase 5: Best Practices Review

### Objective
Ensure endpoints follow GreyCat and API design best practices.

### Step 5.1: Naming Consistency

Check endpoint naming conventions:

```bash
# Extract all function names from @expose
grep -rn "@expose" src/ --include="*_api.gcl" -A 1 | grep "fn " | sed 's/.*fn //' | sed 's/(.*//'
```

Verify:
- Verb-based names (get, create, update, delete, search, list)
- Consistent naming pattern
- Clear and descriptive

**Example Output**:
```
⚠️ NAMING: Inconsistent endpoint naming

Found endpoints:
  ✓ getDocument          (good - clear verb)
  ✓ createUser           (good - clear verb)
  ✓ searchDocuments      (good - clear verb)
  ⚠️ document            (bad - no verb, unclear)
  ⚠️ data                (bad - too generic)
  ⚠️ process             (bad - unclear what it processes)

Recommendation:
  - Rename "document" → "getDocument" or "listDocuments"
  - Rename "data" → "getData" or be more specific
  - Rename "process" → "processSearchQuery" (be specific)

Severity: LOW
Risk: API confusion, poor developer experience
```

### Step 5.2: Missing Documentation

Find endpoints without comments:

```bash
# Find @expose functions without preceding comment
grep -rn "@expose" src/ --include="*_api.gcl" -B 3 | grep -v "//.*"
```

**Example Output**:
```
📍 src/search/search_api.gcl:67

⚠️ DOCUMENTATION: Missing function documentation

Function:
  @expose
  @permission("app.user")
  fn advancedSearch(query: String, filters: SearchFilters): SearchResults {
      // ... complex logic
  }

Problem:
  - No description of what function does
  - No explanation of parameters
  - No documentation of return value
  - Difficult for other developers to use

Recommendation:
  /**
   * Performs advanced search with filters
   *
   * @param query - Search query string (max 1000 chars)
   * @param filters - Optional filters (formex type, date range, etc.)
   * @returns SearchResults with paginated items and metadata
   */
  @expose
  @permission("app.user")
  fn advancedSearch(query: String, filters: SearchFilters): SearchResults {
      // ...
  }

Severity: LOW
Risk: Poor maintainability, unclear API
```

### Step 5.3: Overly Complex Endpoints

Find endpoints with too much logic:

```bash
# Find @expose functions with many lines
for file in $(find src -name "*_api.gcl"); do
    grep -n "@expose" "$file" -A 100 | awk '/^[0-9]+-fn /,/^[0-9]+-}/' | wc -l
done
```

Functions > 50 lines should be reviewed.

**Example Output**:
```
📍 src/complex/complex_api.gcl:23

⚠️ COMPLEXITY: Endpoint too complex

Function: processComplexWorkflow
Lines: 147 lines

Problem:
  - Business logic in API layer (should be in service)
  - Difficult to test
  - Difficult to maintain
  - Violates separation of concerns

Recommendation:
  Move logic to service:

  // In src/workflow/workflow.gcl
  abstract type WorkflowService {
      static fn processComplexWorkflow(input: Input): Result {
          // ... 147 lines of logic here
      }
  }

  // In API (simple delegation)
  @expose
  @permission("app.user")
  fn processComplexWorkflow(input: Input): Result {
      return WorkflowService::processComplexWorkflow(input);
  }

Severity: MEDIUM
Risk: Hard to test, maintain, debug
```

---

## Phase 6: API Design Review

### Objective
Ensure RESTful principles and good API design.

### Step 6.1: HTTP Semantics

While GreyCat uses POST for all calls, function names should reflect intent:

**Example Output**:
```
⚠️ API DESIGN: Misleading function name

Function:
  @expose
  fn getData(...): bool {  ← Returns bool but name says "get"
      // Actually modifies data
      item.status = "processed";
      return true;
  }

Problem:
  - Name implies read operation ("get")
  - Actually performs write operation
  - Misleading to API consumers

Recommendation:
  Rename to reflect mutation:
  - updateDataStatus
  - processData
  - markAsProcessed

Severity: MEDIUM
Risk: API misuse, unexpected side effects
```

### Step 6.2: Return Value Consistency

Check for consistent return patterns:

**Example Output**:
```
⚠️ API DESIGN: Inconsistent return values

Found patterns:
  - getDocument() returns DocumentView?
  - getUser() returns User (throws if not found)
  - getCase() returns null

Problem:
  - Inconsistent null handling
  - Some throw, some return null
  - Confusing for API consumers

Recommendation:
  Standardize on one pattern:

  Option 1: Nullable returns
  - fn getDocument(id): DocumentView?
  - Returns null if not found
  - Caller handles null

  Option 2: Throw on not found
  - fn getDocument(id): DocumentView
  - Throws "Document not found" if missing
  - Caller uses try-catch

  Pick one and use consistently across all endpoints.

Severity: MEDIUM
Risk: Inconsistent API behavior, confusion
```

---

## Output Format

### Executive Summary

```
===============================================================================
API ENDPOINT REVIEW
===============================================================================

Endpoints Analyzed: 34
Files Scanned: 12

ISSUES FOUND:

CRITICAL (Immediate Fix Required):
  [ ] 3 endpoints missing @permission decorator
  [ ] 2 sensitive data exposure risks
  [ ] 1 authentication bypass

HIGH (Fix This Sprint):
  [ ] 5 endpoints missing pagination
  [ ] 4 expensive operations (nested loops)
  [ ] 3 missing error handling
  [ ] 2 N+1 query patterns

MEDIUM (Fix Next Sprint):
  [ ] 8 missing @volatile decorators
  [ ] 6 insufficient input validation
  [ ] 4 overly complex endpoints
  [ ] 3 null pointer risks

LOW (Nice to Have):
  [ ] 12 missing documentation
  [ ] 7 naming inconsistencies
  [ ] 5 minor optimizations

TOTAL ISSUES: 58

ESTIMATED FIX TIME:
  Critical: 2-3 hours
  High:     1 day
  Medium:   2 days
  Low:      1 day

===============================================================================
```

### Detailed Report

Include all findings from each phase with:
- Location (file:line)
- Severity
- Problem description
- Code example
- Recommendation
- Estimated fix time

---

## Success Criteria

✓ **All @expose functions analyzed** (across all API files)
✓ **Security issues identified** (permissions, validation, data exposure)
✓ **Performance issues flagged** (pagination, expensive ops, N+1 queries)
✓ **Error handling gaps found** (try-catch, null checks, messages)
✓ **Type safety checked** (@volatile, parameter types)
✓ **Best practices validated** (naming, documentation, complexity)
✓ **Report generated** with prioritized action items

---

## Notes

- **Run After Every Sprint**: APIs change frequently
- **Before Releases**: Critical for production readiness
- **Security First**: Fix CRITICAL issues immediately
- **Incremental Fixes**: Don't try to fix everything at once
- **Re-run After Fixes**: Verify improvements

---

## Example Workflow

```bash
# 1. Complete sprint with 3 new API endpoints
# 2. Run API review
/apicheck

# 3. Review findings
# - 1 CRITICAL: Missing permission on deleteDocument
# - 2 HIGH: Missing pagination on listDocuments
# - 3 MEDIUM: Missing @volatile on response types

# 4. Fix CRITICAL immediately
# Add @permission("app.admin") to deleteDocument

# 5. Create issues for HIGH/MEDIUM
# Track in sprint backlog

# 6. Re-run review after fixes
/apicheck
# Verify CRITICAL issues resolved

# 7. Before release, ensure all CRITICAL/HIGH fixed
```
