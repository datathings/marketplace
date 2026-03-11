---
name: optimize
description: Detect and auto-fix performance anti-patterns in GreyCat code
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion
---

# GreyCat Performance Optimizer

**Purpose**: Detect and automatically fix performance anti-patterns - unnecessary node allocation, reimplemented native functions, code bloat, algorithmic complexity issues

**Run When**: Quick performance checks, before releases, when performance degrades

---

## Overview

This command performs fast, focused analysis on performance issues:

1. **Unnecessary Persistence** - Using node<T> when plain objects suffice
2. **Native Function Reimplementation** - Custom code duplicating stdlib
3. **Useless Function Wrappers** - One-line functions that just call another function
4. **Algorithmic Complexity** - O(n²) operations where O(n) or O(1) exists
5. **Code Duplication** - Copy-pasted logic

**Output**: Analysis report + auto-fix options

---

## Step 1: Scan Backend Files

**Find all .gcl files**:

```bash
echo "================================================================================"
echo "SCANNING GREYCAT BACKEND"
echo "================================================================================"
echo ""

# Find all GCL files
GCL_FILES=$(find src -name "*.gcl" -type f | sort)
FILE_COUNT=$(echo "$GCL_FILES" | wc -l)

echo "Found $FILE_COUNT files to analyze"
echo ""
```

---

## Step 2: Phase 1 - Unnecessary Persistence Detection

### Concept

**node<T>**, **nodeList**, **nodeIndex**, **nodeTime**, **nodeGeo** should only be used for:
- Module-level variables (global indices)
- Type fields that need persistence

**Local variables, function parameters, and function returns** should use:
- Plain objects: `T` instead of `node<T>`
- Arrays: `Array<T>` instead of `nodeList<node<T>>`
- Maps: `Map<K,V>` instead of `nodeIndex<K,V>`

### Detection Logic

**Scan for**:

```bash
echo "Phase 1: Detecting unnecessary persistence..."
echo ""

# Find local variables with node types
echo "Checking local variables..."
grep -rn "var [a-z_][a-zA-Z0-9_]* = node\(List\|Index\|Time\|Geo\)?<" src --include="*.gcl" | \
    grep -v "^[[:space:]]*var [a-z_][a-zA-Z0-9_]*:" | \  # Exclude module-level (no indentation)
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line"
        echo "     $content"
    done

# Find function parameters with node types
echo ""
echo "Checking function parameters..."
grep -rn "fn [a-z_][a-zA-Z0-9_]*(.*node\(List\|Index\|Time\|Geo\)<" src --include="*.gcl" | \
    while IFS=: read -r file line content; do
        # Skip if it's a service method expecting persisted nodes
        if ! grep -q "Service" <<< "$file"; then
            echo "  ⚠ $file:$line"
            echo "     $content"
        fi
    done

# Find function returns with nodeList/nodeIndex
echo ""
echo "Checking function return types..."
grep -rn "fn [a-z_][a-zA-Z0-9_]*(.*).*: node\(List\|Index\)" src --include="*.gcl" | \
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line"
        echo "     $content"
    done
```

**Categorize findings**:

```
===============================================================================
PHASE 1: UNNECESSARY PERSISTENCE
===============================================================================

🔴 CRITICAL (3 issues):

  src/device/device_api.gcl:45
    fn get_devices(): nodeList<node<Device>>
    → Should return: Array<DeviceView> (API best practice)

  src/processor/processor.gcl:120
    var results = nodeList<node<Item>> {};
    → Should use: Array<Item> {} (local variable, not persisted)

  src/user/user_api.gcl:78
    fn process_users(users: nodeList<node<User>>)
    → Should accept: Array<User> (function parameter)

===============================================================================
```

### Auto-fix Logic

**For local variables**:

```bash
# Example fix:
# BEFORE: var results = nodeList<node<Item>> {};
# AFTER:  var results = Array<Item> {};

# Use Edit tool to replace
# Pattern: nodeList<node<T>> → Array<T>
#          nodeIndex<K, node<V>> → Map<K, V>
#          node<T>{obj} → obj (if local scope)
```

---

## Step 3: Phase 2 - Native Function Reimplementation

### Common Patterns

**Sorting**:
```gcl
// ❌ Custom bubble sort
fn sort_items(items: Array<Item>): Array<Item> {
    for (i in 0..items.size()) {
        for (j in i+1..items.size()) {
            if (items[i]->priority > items[j]->priority) {
                // swap
            }
        }
    }
    return items;
}

// ✅ Use native
items.sort_by(Item::priority, SortOrder::asc);
```

**Min/Max**:
```gcl
// ❌ Custom max finder
fn find_max(values: Array<float>): float {
    var max = values[0];
    for (v in values) {
        if (v > max) { max = v; }
    }
    return max;
}

// ✅ Use native or Tensor
// Math::max(values) or tensor operations
```

**String operations**:
```gcl
// ❌ Custom string join
fn join_strings(strings: Array<String>, sep: String): String {
    var result = "";
    for (i, s in strings) {
        if (i > 0) { result = result + sep; }
        result = result + s;
    }
    return result;
}

// ✅ Use native
strings.join(sep)
```

### Detection

```bash
echo "Phase 2: Detecting reimplemented native functions..."
echo ""

# Look for sorting implementations
echo "Checking for custom sort implementations..."
grep -rn "fn [a-z_]*sort" src --include="*.gcl" -A 20 | \
    grep -B 1 -A 15 "for.*for" | \  # Nested loops suggest bubble/selection sort
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line - Possible custom sort (use .sort_by())"
    done

# Look for min/max implementations
echo ""
echo "Checking for custom min/max..."
grep -rn "fn find_\(max\|min\|maximum\|minimum\)" src --include="*.gcl" | \
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line - Custom min/max (use Math:: or Tensor)"
    done

# Look for string join implementations
echo ""
echo "Checking for custom string operations..."
grep -rn "fn [a-z_]*join" src --include="*.gcl" -A 10 | \
    grep -B 1 "for.*in.*{" | \
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line - Custom join (use .join())"
    done
```

**Report**:

```
===============================================================================
PHASE 2: REIMPLEMENTED NATIVE FUNCTIONS
===============================================================================

🟡 MEDIUM (3 issues):

  src/array_utils/array_utils.gcl:23
    fn sort_by_priority(items: Array<Item>)
    → Use native: items.sort_by(Item::priority, SortOrder::asc)

  src/math_utils/math_utils.gcl:45
    fn find_maximum(values: Array<float>)
    → Use Math:: module or Tensor operations

  src/string_utils/string_utils.gcl:67
    fn join_with_comma(strings: Array<String>)
    → Use native: strings.join(", ")

===============================================================================
```

---

## Step 4: Phase 3 - Useless Function Wrappers

### Pattern

Functions that contain only a single statement calling another function:

```gcl
// ❌ Useless wrapper
fn get_user(id: int): node<User>? {
    return UserService::find_by_id(id);
}

// ✅ Just call UserService::find_by_id(id) directly
```

### Detection

```bash
echo "Phase 3: Detecting useless function wrappers..."
echo ""

# Find functions with single return statement
for file in $GCL_FILES; do
    # Extract function definitions
    awk '/^fn [a-z_]/ {
        func_line = NR;
        func = $0;
        getline; # Read opening brace
        if ($0 ~ /return.*::/) {
            getline; # Read closing brace
            if ($0 ~ /^}/) {
                print FILENAME ":" func_line ": " func
            }
        }
    }' "$file"
done | while IFS=: read -r file line content; do
    echo "  ⚠ $file:$line"
    echo "     $content"
    echo "     (One-line wrapper - consider removing)"
done
```

**Report**:

```
===============================================================================
PHASE 3: USELESS FUNCTION WRAPPERS
===============================================================================

🟡 MEDIUM (5 issues):

  src/user/user_api.gcl:67
    fn get_user(id: int): node<User>?
    → Single-line wrapper for UserService::find_by_id()
    → Consider calling UserService directly

  src/device/device.gcl:34
    fn find_device(id: int): node<Device>?
    → Wraps DeviceService::find_by_id()

  ... (3 more)

===============================================================================
```

---

## Step 5: Phase 4 - Algorithmic Complexity

### O(n²) Nested Loops

**Problem**:
```gcl
// ❌ O(n²) - iterates all users × all orders
for (user in all_users) {
    for (order in all_orders) {
        if (order->user_id == user->id) {
            // Process order for user
        }
    }
}

// ✅ O(n) - use nodeIndex for O(1) lookup
for (user in all_users) {
    var user_orders = orders_by_user_id.get(user->id) ?? Array<node<Order>>{};
    for (order in user_orders) {
        // Process order for user
    }
}
```

### Detection

```bash
echo "Phase 4: Detecting algorithmic complexity issues..."
echo ""

# Find nested loops
echo "Checking for nested loops..."
grep -rn "for.*in.*{" src --include="*.gcl" -A 5 | \
    grep "for.*in.*{" | \
    while IFS=: read -r file line content; do
        # Check if inner loop has conditional matching
        CONTEXT=$(sed -n "${line},$((line+10))p" "$file")
        if grep -q "if.*==.*{" <<< "$CONTEXT"; then
            echo "  ⚠ $file:$line - Possible O(n²) with conditional match"
            echo "     Consider using nodeIndex for O(1) lookup"
        fi
    done

# Find linear searches where index exists
echo ""
echo "Checking for linear searches..."
grep -rn "for.*in.*{" src --include="*.gcl" -A 3 | \
    grep -B 1 "if.*->id ==" | \
    while IFS=: read -r file line content; do
        echo "  ⚠ $file:$line - Linear search by ID"
        echo "     Consider using nodeIndex for direct lookup"
    done
```

**Report**:

```
===============================================================================
PHASE 4: ALGORITHMIC COMPLEXITY
===============================================================================

🔴 CRITICAL (2 issues):

  src/matcher/matcher.gcl:89
    Nested loop: for (user in users) { for (order in orders) { if (order->user_id == user->id) ... } }
    → O(n²) complexity
    → Solution: Create orders_by_user_id: nodeIndex<int, node<Order>>

  src/lookup/lookup.gcl:134
    Linear search: for (item in items) { if (item->id == target_id) ... }
    → O(n) when O(1) possible
    → Solution: Use items_by_id nodeIndex

===============================================================================
```

### Auto-fix

**Suggest index creation**:

```gcl
// Add to model file:
var orders_by_user_id: nodeIndex<int, nodeList<node<Order>>>;

// Update service to maintain index:
abstract type OrderService {
    static fn create(user_id: int, amount: float): node<Order> {
        var order = node<Order>{ Order { user_id: user_id, amount: amount }};

        // Add to user's orders
        var user_orders = orders_by_user_id.get(user_id);
        if (user_orders == null) {
            user_orders = nodeList<node<Order>>{};
            orders_by_user_id.set(user_id, user_orders);
        }
        user_orders.add(order);

        return order;
    }
}

// Use in code:
for (user in all_users) {
    var user_orders = orders_by_user_id.get(user->id);
    if (user_orders != null) {
        for (i, order in user_orders) {
            // Process
        }
    }
}
```

---

## Step 6: Phase 5 - Code Duplication

**Leverage Grep to find similar code blocks**:

```bash
echo "Phase 5: Detecting code duplication..."
echo ""

# Find duplicate function signatures (same name, different files)
grep -rn "^fn [a-z_][a-zA-Z0-9_]*(" src --include="*.gcl" | \
    awk -F: '{print $3}' | \
    sort | \
    uniq -d | \
    while read func; do
        echo "  ⚠ Duplicate function signature: $func"
        grep -rn "$func" src --include="*.gcl"
    done

# Find repeated patterns (e.g., same error message strings)
echo ""
echo "Checking for repeated error messages..."
grep -rn "throw \"" src --include="*.gcl" | \
    awk -F'"' '{print $2}' | \
    sort | \
    uniq -c | \
    sort -rn | \
    head -10 | \
    while read count msg; do
        if [ $count -gt 2 ]; then
            echo "  ⚠ Error message repeated $count times: \"$msg\""
        fi
    done
```

**Report**:

```
===============================================================================
PHASE 5: CODE DUPLICATION
===============================================================================

🟢 LOW (4 issues):

  Duplicate error messages (3+ occurrences):
    - "User not found" (5 times)
    - "Invalid email format" (4 times)
    → Consider creating error constant or helper function

  Similar validation logic in 3 files:
    - src/user/user.gcl:45
    - src/admin/admin.gcl:78
    - src/auth/auth_api.gcl:23
    → Extract to shared validation function

===============================================================================
```

---

## Step 7: Consolidate Report

**Generate summary**:

```
===============================================================================
PERFORMANCE OPTIMIZATION REPORT
===============================================================================

Analyzed: 47 files (src)

Found 17 issues:

🔴 CRITICAL (5):
  1. src/device/device_api.gcl:45 - API returning nodeList instead of Array<View>
  2. src/processor/processor.gcl:120 - Local var using nodeList instead of Array
  3. src/matcher/matcher.gcl:89 - O(n²) nested loop, needs nodeIndex
  4. src/user/user_api.gcl:78 - Function parameter using nodeList
  5. src/lookup/lookup.gcl:134 - Linear search, needs nodeIndex

🟡 MEDIUM (8):
  6-10. Reimplemented native functions (sort, max, join, etc.)
  11-13. Useless function wrappers

🟢 LOW (4):
  14-17. Code duplication (error messages, validation logic)

Auto-fix available for 12/17 issues

Estimated performance improvement: ~35%
  - Removing unnecessary persistence: ~20%
  - Fixing O(n²) operations: ~15%
  - Native functions: ~5%

===============================================================================
```

---

## Step 8: Apply Fixes

**Ask user**:

```typescript
AskUserQuestion({
  questions: [{
    question: "Apply automatic fixes?",
    header: "Auto-fix",
    multiSelect: false,
    options: [
      {
        label: "Fix critical issues only (Recommended)",
        description: "Auto-fix the 5 critical performance issues"
      },
      {
        label: "Fix all auto-fixable issues",
        description: "Fix 12 issues automatically (critical + medium)"
      },
      {
        label: "Show detailed report first",
        description: "Review each issue before fixing"
      },
      {
        label: "Cancel",
        description: "No changes, just report"
      }
    ]
  }]
})
```

**Apply fixes** using Edit tool:

```bash
echo "================================================================================"
echo "APPLYING FIXES"
echo "================================================================================"
echo ""

# Fix 1: API returning nodeList
echo "Fixing: src/device/device_api.gcl:45"
# Use Edit tool to change return type from nodeList<node<Device>> to Array<DeviceView>

# Fix 2: Local var using nodeList
echo "Fixing: src/processor/processor.gcl:120"
# Use Edit tool to change "var results = nodeList<node<Item>> {};" to "var results = Array<Item> {};"

# ... apply other fixes

echo ""
echo "✓ Applied 12 fixes"
```

**Run lint**:

```bash
echo "================================================================================"
echo "VERIFYING FIXES"
echo "================================================================================"
echo ""

greycat-lang lint --fix

LINT_EXIT=$?

if [ $LINT_EXIT -eq 0 ]; then
    echo ""
    echo "✓ All fixes applied successfully, lint passes"
else
    echo ""
    echo "⚠ Some fixes may need manual adjustment"
fi
```

**Final report**:

```
===============================================================================
OPTIMIZATION COMPLETE
===============================================================================

Fixed 12 issues:
  ✓ 5 critical (unnecessary persistence, O(n²) operations)
  ✓ 7 medium (native functions, wrappers)

Remaining 5 issues require manual review:
  ! src/complex/complex.gcl:234 - Complex duplication, needs refactoring
  ! src/advanced/advanced.gcl:567 - Algorithmic improvement needs design

Lint: ✓ Passes

Estimated performance improvement: ~35%

Next steps:
  1. Review remaining issues manually
  2. Run tests: greycat test
  3. Benchmark before/after if needed

===============================================================================
```

---

## Success Criteria

✓ **Performance issues detected** across all categories
✓ **Auto-fixes applied** safely with lint verification
✓ **Detailed report generated** with severity levels
✓ **Estimated improvements** calculated
✓ **greycat-lang lint --fix passes** after fixes

---

## Notes

- **Focus**: Quick, automated performance wins
- **Safe fixes**: Only applies changes that don't alter logic
- **Comprehensive**: Covers persistence, complexity, duplication
- **Complement to backend**: Use /greycat:backend for full code review
- **Regular use**: Run before releases or when performance degrades
