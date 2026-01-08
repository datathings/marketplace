---
name: coverage
description: Generate test coverage report and suggest new tests to implement
allowed-tools: Bash, Read, Grep, Glob, Write
---

# Test Coverage Analysis & Gap Identification

**Purpose**: Analyze current test coverage, identify critical untested code, and generate specific test suggestions prioritized by risk.

**Run After**: Each development sprint, before releases, when adding new features

---

## Overview

This command performs a comprehensive test coverage analysis in 3 phases:

1. **Phase 1**: Generate current coverage report
2. **Phase 2**: Identify critical gaps (untested code)
3. **Phase 3**: Suggest specific tests to implement (prioritized)

---

## Phase 1: Generate Coverage Report

### Step 1.1: Run All Tests

```bash
echo "================================================================================"
echo "PHASE 1: GENERATING TEST COVERAGE REPORT"
echo "================================================================================"
echo ""

echo "Running all tests..."
greycat test
TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed (exit code: $TEST_EXIT_CODE)"
    echo "  Continuing with coverage analysis..."
fi
echo ""
```

### Step 1.2: Collect Test Files

Use Glob to find all test files:

```bash
# Find all test files
find backend/test -name "*_test.gcl" -o -name "*test.gcl" 2>/dev/null
```

Analyze each test file to extract:
- Test function names (`@test fn test_name()`)
- What services/types they test
- Coverage scope (unit vs integration)

### Step 1.3: Build Coverage Map

For each test file, identify:
- **Service/Type Under Test**: Extract from imports or function calls
- **Functions Tested**: Parse test assertions and function calls
- **Coverage Type**: Unit, integration, or end-to-end

Create a coverage map:
```
{
  "PaginationService": {
    "tested_functions": ["validateOffset", "validateLimit"],
    "untested_functions": ["calculatePage", "calculateTotalPages"],
    "test_files": ["pagination_service_test.gcl"]
  },
  "SearchEngine": {
    "tested_functions": ["bm25Score", "normalizeScore"],
    "untested_functions": [],
    "test_files": ["search_engine_test.gcl"]
  }
}
```

---

## Phase 2: Identify Critical Gaps

### Step 2.1: Scan Backend Code

Find all services, API endpoints, and critical business logic:

**A. Find All Services** (abstract types with static functions):
```bash
# Search for abstract types (services)
grep -r "^abstract type.*Service" backend/src/service/ --include="*.gcl"
```

**B. Find All API Endpoints** (@expose functions):
```bash
# Search for @expose functions
grep -r "@expose" backend/src/api/ --include="*.gcl" -A 2
```

**C. Find All Model Types**:
```bash
# Search for type definitions
grep -r "^type [A-Z]" backend/src/model/ --include="*.gcl"
```

**D. Find Critical Business Logic**:
- Functions with complex logic (> 20 lines)
- Functions with multiple branches
- Functions handling money/security/data integrity

### Step 2.2: Cross-Reference with Tests

For each discovered item, check if tests exist:

**Service Functions**:
```gcl
// Service: backend/src/service/auth/user_service.gcl
abstract type UserService {
    static fn createUser(...): User;      // ⚠️ No test found
    static fn getUserByEmail(...): User?; // ✓ Tested in user_service_test.gcl
    static fn deleteUser(...): bool;      // ⚠️ No test found
}
```

**API Endpoints**:
```gcl
// API: backend/src/api/search_api.gcl
@expose
@permission("app.user")
fn search(query: String, ...): SearchResponse {  // ⚠️ No integration test
    // ...
}
```

**Model Validation**:
```gcl
// Model: backend/src/model/user.gcl
type User {
    email: String;

    fn validate(): bool {  // ⚠️ No test found
        // complex validation logic
    }
}
```

### Step 2.3: Calculate Risk Scores

Prioritize gaps by risk (0-100):

**Risk Formula**:
```
Risk = (Complexity × 20) + (Usage × 30) + (Criticality × 50)

Where:
- Complexity: Lines of code / branches (0-1 normalized)
- Usage: Number of call sites (0-1 normalized)
- Criticality: Domain importance (0-1: 0=low, 0.5=medium, 1=high)
```

**Criticality Heuristics**:
- **HIGH (1.0)**: Auth, payments, data mutations, @expose endpoints
- **MEDIUM (0.5)**: Search, filtering, data retrieval, validation
- **LOW (0.0)**: Utilities, formatters, helpers, UI-only code

---

## Phase 3: Suggest Specific Tests

### Step 3.1: Generate Test Suggestions

For each gap, create actionable test suggestions with:

1. **Test file name**: `backend/test/xxx_test.gcl`
2. **Test function names**: Specific to scenario
3. **Test code template**: Ready-to-use skeleton
4. **Priority**: HIGH/MEDIUM/LOW (based on risk score)
5. **Rationale**: Why this test matters

### Step 3.2: Template Generation

Generate test templates based on code patterns:

**Template 1: Service Function Test**
```gcl
// backend/test/user_service_test.gcl

@test
fn test_create_user_valid_input() {
    var user = UserService::createUser("test@example.com", "password123", "user");

    Assert::isTrue(user != null);
    Assert::equals(user.email, "test@example.com");
    Assert::equals(user.role, "user");
}

@test
fn test_create_user_duplicate_email() {
    // Create first user
    UserService::createUser("test@example.com", "pass1", "user");

    // Attempt duplicate
    try {
        UserService::createUser("test@example.com", "pass2", "user");
        Assert::isFalse(true);  // Should have thrown
    } catch (ex) {
        Assert::isTrue(true);   // Expected error
    }
}

@test
fn test_create_user_invalid_email() {
    try {
        UserService::createUser("invalid-email", "password", "user");
        Assert::isFalse(true);
    } catch (ex) {
        var msg = "${ex}";
        Assert::isTrue(msg.contains("email") || msg.contains("invalid"));
    }
}
```

**Template 2: API Endpoint Test**
```gcl
// backend/test/api/search_api_test.gcl

@test
fn test_search_valid_query() {
    var response = search("privacy", null, 0, 10);

    Assert::isTrue(response != null);
    Assert::isTrue(response.results != null);
    Assert::isTrue(response.total >= 0);
}

@test
fn test_search_empty_query() {
    var response = search("", null, 0, 10);

    // Should return empty results, not error
    Assert::equals(response.total, 0);
    Assert::equals(response.results.size(), 0);
}

@test
fn test_search_pagination() {
    var page1 = search("test", null, 0, 5);
    var page2 = search("test", null, 5, 5);

    Assert::isTrue(page1.results.size() <= 5);
    Assert::isTrue(page2.results.size() <= 5);

    // Verify no overlap
    if (page1.results.size() > 0 && page2.results.size() > 0) {
        var id1 = page1.results[0].id;
        var id2 = page2.results[0].id;
        Assert::isTrue(id1 != id2);
    }
}

@test
fn test_search_requires_permission() {
    // This test verifies @permission("app.user") is enforced
    // Requires authentication context setup
    // Implementation depends on your auth system
}
```

**Template 3: Validation Logic Test**
```gcl
// backend/test/model/user_test.gcl

@test
fn test_user_validate_valid_email() {
    var user = User { email: "valid@example.com", ... };
    Assert::isTrue(user.validate());
}

@test
fn test_user_validate_invalid_email() {
    var user = User { email: "not-an-email", ... };
    Assert::isFalse(user.validate());
}

@test
fn test_user_validate_missing_fields() {
    var user = User { email: "", ... };
    Assert::isFalse(user.validate());
}
```

**Template 4: Edge Case Test**
```gcl
@test
fn test_function_boundary_min() {
    var result = Service::process(0);
    // Verify behavior at minimum value
}

@test
fn test_function_boundary_max() {
    var result = Service::process(2147483647);
    // Verify behavior at maximum int
}

@test
fn test_function_null_handling() {
    var result = Service::process(null);
    Assert::isTrue(result == null || /* expected behavior */);
}

@test
fn test_function_empty_collection() {
    var result = Service::processList(Array<T>{});
    Assert::equals(result.size(), 0);
}
```

---

## Output Format

### Summary Report

```
===============================================================================
TEST COVERAGE ANALYSIS REPORT
===============================================================================

CURRENT COVERAGE:
  Total Services:        15
  Tested Services:       8  (53%)
  Untested Services:     7  (47%)

  Total API Endpoints:   22
  Tested Endpoints:      5  (23%)
  Untested Endpoints:    17 (77%)

  Total Test Files:      6
  Total Test Functions:  45

RISK SUMMARY:
  HIGH Priority Gaps:    12 items
  MEDIUM Priority Gaps:  18 items
  LOW Priority Gaps:     7 items

===============================================================================
```

### Detailed Gap Report

```
===============================================================================
HIGH PRIORITY TEST GAPS (Risk Score > 70)
===============================================================================

📍 backend/src/service/auth/user_service.gcl

⚠️ UNTESTED: UserService::createUser
   Risk Score: 95 (HIGH)
   Criticality: HIGH (auth function)
   Complexity: MEDIUM (30 lines, 5 branches)
   Usage: 8 call sites

   Why Test This:
   - Authentication is critical for security
   - Creates persistent user data
   - Has multiple failure modes (duplicate email, invalid input)

   Suggested Tests:
   1. test_create_user_valid_input
   2. test_create_user_duplicate_email
   3. test_create_user_invalid_email
   4. test_create_user_invalid_role
   5. test_create_user_weak_password

   Test File: backend/test/user_service_test.gcl
   Estimated Effort: 30 minutes

---

📍 backend/src/api/payment_api.gcl

⚠️ UNTESTED: processPayment
   Risk Score: 98 (HIGH)
   Criticality: HIGH (payment processing)
   Complexity: HIGH (80 lines, 12 branches)
   Usage: 3 call sites

   Why Test This:
   - Financial transactions require thorough testing
   - Complex error handling (network, validation, auth)
   - Data integrity critical

   Suggested Tests:
   1. test_process_payment_valid
   2. test_process_payment_insufficient_funds
   3. test_process_payment_invalid_card
   4. test_process_payment_network_error
   5. test_process_payment_duplicate_transaction

   Test File: backend/test/api/payment_api_test.gcl
   Estimated Effort: 60 minutes

===============================================================================
MEDIUM PRIORITY TEST GAPS (Risk Score 40-70)
===============================================================================

📍 backend/src/service/search/search_service.gcl

⚠️ UNTESTED: SearchService::buildQuery
   Risk Score: 62 (MEDIUM)
   Criticality: MEDIUM (search functionality)
   Complexity: MEDIUM (40 lines, 8 branches)
   Usage: 12 call sites

   Why Test This:
   - High usage across application
   - Complex query construction logic
   - Multiple filter combinations

   Suggested Tests:
   1. test_build_query_simple
   2. test_build_query_with_filters
   3. test_build_query_with_date_range
   4. test_build_query_empty_input

   Test File: backend/test/search_service_test.gcl
   Estimated Effort: 20 minutes

===============================================================================
```

### Test Implementation Plan

```
===============================================================================
RECOMMENDED TEST IMPLEMENTATION PLAN
===============================================================================

SPRINT 1 (HIGH Priority - 2-3 days):
  [ ] UserService::createUser (6 tests)
  [ ] UserService::deleteUser (4 tests)
  [ ] processPayment API endpoint (5 tests)
  [ ] updateUserRole API endpoint (3 tests)

SPRINT 2 (MEDIUM Priority - 2 days):
  [ ] SearchService::buildQuery (4 tests)
  [ ] DocumentService::getDocumentCandidates (5 tests)
  [ ] PaginationService (remaining functions - 8 tests)

SPRINT 3 (LOW Priority - 1 day):
  [ ] Utility functions (string formatting, date helpers)
  [ ] View builders (formatters, mappers)

TOTAL ESTIMATED EFFORT: 5-6 days
PRIORITY: Complete Sprint 1 before next release

===============================================================================
```

### Generated Test Files

For each high-priority gap, create ready-to-use test file templates:

**File: backend/test/user_service_test.gcl** (generated)
```gcl
// AUTO-GENERATED TEST TEMPLATE
// TODO: Implement test assertions based on your business logic

@test
fn test_create_user_valid_input() {
    // TODO: Implement test
    // Expected: User created successfully with correct fields
    var user = UserService::createUser("test@example.com", "password123", "user");

    Assert::isTrue(user != null);
    Assert::equals(user.email, "test@example.com");
    // Add more assertions...
}

@test
fn test_create_user_duplicate_email() {
    // TODO: Implement test
    // Expected: Should throw error or return null
    UserService::createUser("dup@example.com", "pass1", "user");

    try {
        UserService::createUser("dup@example.com", "pass2", "user");
        Assert::isFalse(true);  // Should not reach here
    } catch (ex) {
        Assert::isTrue(true);   // Expected
    }
}

// ... more tests
```

---

## Execution Steps

### Step 1: Run Coverage Analysis

Execute the command and let it analyze your codebase:

1. Scans all backend files for services, APIs, models
2. Cross-references with existing test files
3. Calculates risk scores
4. Generates prioritized gap report

### Step 2: Review Suggestions

Present the report to the user:
- Show summary statistics
- Highlight HIGH priority gaps
- Explain risk rationale for each item

### Step 3: Offer to Generate Tests

Ask the user:
```
I found 12 HIGH priority test gaps. Would you like me to:

A) Generate test file templates for HIGH priority items (recommended)
B) Generate tests for specific items (I'll ask which ones)
C) Just show the report (you'll write tests manually)
D) Generate all suggested tests (HIGH + MEDIUM + LOW)
```

### Step 4: Generate Test Templates

Based on user choice, create test files:
- Use Write tool to create `backend/test/xxx_test.gcl` files
- Include TODO comments for user to complete
- Add descriptive test names and structure
- Include edge cases and error scenarios

### Step 5: Verify Generated Tests

```bash
# Run linter on generated tests
greycat-lang lint

# Try running the new (incomplete) tests
greycat test

echo ""
echo "✓ Test templates generated successfully"
echo "  Next steps:"
echo "  1. Complete TODO sections in generated test files"
echo "  2. Run 'greycat test' to verify"
echo "  3. Add more test cases as needed"
```

---

## Detection Heuristics

### Finding Untested Code

**Service Functions**:
```bash
# Find all static functions in services
grep -rn "static fn" backend/src/service/ --include="*.gcl"

# For each function, check if test exists
# Search pattern: test_<snake_case_function_name>
```

**API Endpoints**:
```bash
# Find all @expose functions
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 5

# Extract function signature
# Check if integration test exists
```

**Model Methods**:
```bash
# Find all type methods
grep -rn "^\s*fn [a-z]" backend/src/model/ --include="*.gcl"

# Exclude simple getters/setters
# Check for tests in backend/test/model/
```

### Complexity Calculation

**Lines of Code**:
```bash
# Count lines in function (between fn declaration and closing })
sed -n '/fn functionName/,/^}/p' file.gcl | wc -l
```

**Branch Count**:
```bash
# Count if/else/match/for/while keywords
grep -o '\(if\|else\|match\|for\|while\)' file.gcl | wc -l
```

**Call Sites** (usage):
```bash
# Search for function calls across codebase
grep -r "FunctionName(" backend/ --include="*.gcl" | wc -l
```

---

## Success Criteria

✓ **Coverage report generated** with statistics
✓ **All services scanned** for test coverage
✓ **All API endpoints analyzed** for integration tests
✓ **Risk scores calculated** for each gap
✓ **Prioritized suggestions** (HIGH/MEDIUM/LOW)
✓ **Test templates generated** (if requested)
✓ **Templates lint successfully** (`greycat-lang lint` passes)

---

## Notes

- **Data Dependency**: Some tests require database data (use setup/teardown or fixtures)
- **Authentication Tests**: May require auth context setup
- **Integration Tests**: May need running server (`greycat serve`)
- **Generated Tests**: Templates include TODOs - user must complete business logic
- **Iterative**: Run after each sprint to track coverage improvements

---

## Example Workflow

```bash
# 1. Run test gap analysis
/coverage

# 2. Review HIGH priority gaps (12 items)
# 3. Choose to generate templates for HIGH priority

# 4. Templates created in backend/test/
#    - user_service_test.gcl (6 tests)
#    - payment_api_test.gcl (5 tests)

# 5. Complete TODO sections in generated files

# 6. Run tests
greycat test

# 7. Fix any failures, add more tests as needed

# 8. Re-run /coverage to see improved coverage
```

---

## Future Enhancements

- **Coverage Percentage**: Calculate actual line/branch coverage (requires instrumentation)
- **Mutation Testing**: Detect weak tests that don't catch bugs
- **Performance Tests**: Suggest performance benchmarks for critical paths
- **Regression Tests**: Auto-generate tests from production bugs
- **Test Quality**: Analyze existing tests for anti-patterns
