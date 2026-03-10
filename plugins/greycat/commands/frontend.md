---
name: frontend
description: Review frontend codebase for code quality, performance, and best practices
allowed-tools: Bash, Read, Grep, Glob
---

# Frontend Review & Analysis

**Purpose**: Comprehensive analysis of TypeScript frontend for code quality, performance, and best practices

**Run After**: Each sprint, before releases, when adding features

---

## Overview

This command performs multi-category frontend analysis:

1. **Code Quality** - TypeScript practices, error handling
2. **Performance** - Bundle size, computations
3. **Architecture** - Component structure, routing
4. **Security** - XSS, API patterns, data exposure
5. **Dead Code** - Unused exports, files, dependencies

---

## Phase 0: Coding Standards Reference

Before reviewing, understand the required patterns. These are the standards to check against:

### Service Pattern

```typescript
// Named export object with withRetry wrapper
export const DocumentService = {
  getDocument: (id: string): Promise<gc.api_types.Document> =>
    withRetry(() => gc.document(id)),

  search: (query: string): Promise<gc.api_types.SearchResults> =>
    withRetry(() => gc.search(query)),
}
```

---

## Phase 1: Code Quality & Best Practices

### Objective
Ensure TypeScript best practices are followed.

### Step 1.1: TypeScript Type Safety

**Find `any` usage**:
```bash
# Search for 'any' type usage
grep -rn ": any\|<any>" app/ --include="*.ts" --include="*.tsx"
```

**Example Output**:
```
app/components/DataTable.tsx:45

CODE QUALITY: Using 'any' type

Code:
  45: const handleData = (data: any) => {

Problem:
  - Using 'any' bypasses TypeScript type checking
  - Loses type safety benefits

Fix:
  interface DataType {
      id: string;
      value: number;
  }

  const handleData = (data: DataType) => {
      processData(data);
  }

Priority: MEDIUM
Impact: Type safety, maintainability
```

### Step 1.2: Services Without Retry

```bash
# Find service calls not using withRetry
grep -rn "gc\.\w\+(" app/services/ --include="*.ts" | grep -v "withRetry"
```

**Example Output**:
```
app/services/documentService.ts:12

CODE QUALITY: Service call without retry wrapper

Code:
  12: getDocument: (id: string) => gc.document(id),

Fix:
  getDocument: (id: string): Promise<gc.api_types.Document> =>
    withRetry(() => gc.document(id)),

Priority: MEDIUM
Impact: Reliability, error handling
```

---

## Phase 2: Performance Analysis

### Step 2.1: Bundle Size Analysis

```bash
# Run bundle analyzer (if available)
cd app && npm run analyze 2>/dev/null || echo "No bundle analyzer configured"
```

Look for:
- Large dependencies (>100KB)
- Duplicate dependencies
- Tree-shaking opportunities

### Step 2.2: Lazy Loading Opportunities

```bash
# Find large pages without lazy loading
grep -rn "import.*from.*pages" app/ --include="*.ts" --include="*.tsx" | grep -v "lazy\|dynamic"
```

---

## Phase 3: Architecture & Organization

### Step 3.1: Code Duplication

```bash
# Find similar patterns
find app -name "*.ts" -o -name "*.tsx" | xargs wc -l | sort -rn | head -20
```

Manually review large files for duplication.

---

## Phase 4: Security

### Step 4.1: XSS Vulnerabilities

```bash
# Find dangerouslySetInnerHTML or innerHTML usage
grep -rn "dangerouslySetInnerHTML\|innerHTML" app/ --include="*.ts" --include="*.tsx"
```

### Step 4.2: Sensitive Data in Client

```bash
# Find potential secrets in code
grep -rn "api.*key\|secret\|password" app/ --include="*.ts" --include="*.tsx" -i
```

---

## Phase 5: Dead Code Analysis

### Step 5.1: Run Dead Code Detector

**Detect package manager**:
```bash
if [ -f "app/package-lock.json" ]; then
    PKG_MGR="npm"
elif [ -f "app/pnpm-lock.yaml" ]; then
    PKG_MGR="pnpm"
elif [ -f "app/yarn.lock" ]; then
    PKG_MGR="yarn"
else
    PKG_MGR="npm"
fi
```

**Run analyzer**:
```bash
cd app

# Check if knip is configured
if grep -q "knip" package.json; then
    echo "Running dead code analysis..."
    $PKG_MGR run dead-code 2>&1 || echo "No dead-code script found"
else
    echo "No dead code analyzer (knip) configured"
    echo "To add: npm install -D knip"
fi
```

---

## Output Format

### Executive Summary

```
===============================================================================
FRONTEND REVIEW REPORT
===============================================================================

Files Analyzed: [N] (.ts/.tsx files)

ISSUES FOUND:

CRITICAL (Security/Breaking):
  [ ] XSS vulnerabilities
  [ ] Hardcoded secrets

HIGH (Performance):
  [ ] Routes without lazy loading
  [ ] Large dependencies

MEDIUM (Code Quality/Maintainability):
  [ ] Uses of 'any' type
  [ ] Service calls without retry
  [ ] Duplicated code blocks

LOW (Nice to Have):
  [ ] Unused exports
  [ ] Unused dependencies

===============================================================================
```

---

## Success Criteria

- **All frontend files scanned** (.ts/.tsx in app/)
- **TypeScript quality checked** (any usage, type safety)
- **Performance analyzed** (bundle, lazy loading)
- **Architecture reviewed** (duplication, organization)
- **Security checked** (XSS, secrets)
- **Dead code detected** (if knip configured)
- **Report generated** with prioritized issues

---

## Notes

- **Works with any TypeScript frontend**: Framework-agnostic patterns
- **Package manager agnostic**: Detects npm/pnpm/yarn
- **Dead code requires knip**: Install if not present
- **Fix CRITICAL first**: Security issues are urgent
- **Test after changes**: Run tests after applying fixes
- **GreyCat projects**: Services should use withRetry wrapper for gc.* calls
