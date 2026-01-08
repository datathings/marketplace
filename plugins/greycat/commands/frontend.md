---
name: frontend
description: Review frontend codebase for code quality, performance, and best practices
allowed-tools: Bash, Read, Grep, Glob
---

# Frontend Review & Analysis

**Purpose**: Comprehensive analysis of React/TypeScript frontend for code quality, performance, accessibility, and dead code

**Run After**: Each sprint, before releases, when adding features

---

## Overview

This command performs multi-category frontend analysis:

1. **Code Quality** - TypeScript practices, memoization, error handling
2. **Performance** - Re-renders, bundle size, computations
3. **Architecture** - Component structure, hooks, routing
4. **Accessibility** - ARIA, semantic HTML, focus management
5. **Testing** - Coverage gaps, test suggestions
6. **UI/UX Consistency** - Styling, theming, responsiveness
7. **Security** - XSS, API patterns, data exposure
8. **Dead Code** - Unused exports, files, dependencies

---

## Phase 1: Code Quality & Best Practices

### Objective
Ensure TypeScript and React best practices are followed.

### Step 1.1: TypeScript Type Safety

**Find `any` usage**:
```bash
# Search for 'any' type usage
grep -rn ": any\|<any>" frontend/src/ --include="*.ts" --include="*.tsx"
```

**Example Output**:
```
📍 frontend/src/components/DataTable.tsx:45

⚠️ CODE QUALITY: Using 'any' type

Code:
  45: const handleData = (data: any) => {
  46:     processData(data);
  47: }

Problem:
  - Using 'any' bypasses TypeScript type checking
  - Loses type safety benefits
  - Makes refactoring dangerous

Fix:
  interface DataType {
      id: string;
      value: number;
      // ... other fields
  }

  const handleData = (data: DataType) => {
      processData(data);
  }

Priority: MEDIUM
Impact: Type safety, maintainability
```

### Step 1.2: Missing Memoization

**Find expensive computations**:
```bash
# Find components without React.memo
grep -rn "^export const.*: React.FC\|^function.*Component" frontend/src/components/ --include="*.tsx" | grep -v "React.memo"
```

**Example Output**:
```
📍 frontend/src/components/SearchResults.tsx:12

⚠️ PERFORMANCE: Component without memoization

Code:
  12: export const SearchResults: React.FC<Props> = ({ items, onSelect }) => {
  13:     // Heavy rendering logic
  14:     return <div>{items.map(...)}</div>;
  15: };

Problem:
  - Component re-renders on every parent render
  - No props comparison
  - Expensive for large lists

Fix:
  export const SearchResults = React.memo<Props>(
      ({ items, onSelect }) => {
          // ... component logic
      }
  );

  // Or with custom comparison:
  export const SearchResults = React.memo<Props>(
      ({ items, onSelect }) => { ... },
      (prevProps, nextProps) => prevProps.items === nextProps.items
  );

Priority: HIGH (if component renders frequently)
Impact: Performance, user experience
```

### Step 1.3: Prop Drilling

**Find deep prop passing**:
```bash
# Look for components passing many props
grep -rn "props\." frontend/src/components/ --include="*.tsx" | wc -l
```

Identify chains where props are passed through 3+ levels.

**Example Output**:
```
⚠️ ARCHITECTURE: Prop drilling detected

Chain:
  App.tsx → Dashboard.tsx → UserPanel.tsx → UserCard.tsx → UserAvatar.tsx

Props passed: user, theme, onUpdate (through 4 levels)

Problem:
  - Props passed through intermediate components
  - Intermediate components don't use the props
  - Makes refactoring difficult

Fix Option 1 (Context):
  // Create context
  const UserContext = createContext<UserContextType>(null!);

  // In App.tsx
  <UserContext.Provider value={{ user, onUpdate }}>
      <Dashboard />
  </UserContext.Provider>

  // In UserAvatar.tsx
  const { user, onUpdate } = useContext(UserContext);

Fix Option 2 (State Management):
  // Use Zustand/Redux/Jotai
  const useUserStore = create((set) => ({
      user: null,
      setUser: (user) => set({ user })
  }));

Priority: MEDIUM
Impact: Maintainability, code clarity
```

---

## Phase 2: Performance Analysis

### Step 2.1: Missing React Keys

```bash
# Find .map without keys
grep -rn "\.map(" frontend/src/ --include="*.tsx" -A 2 | grep -v "key="
```

**Example Output**:
```
📍 frontend/src/components/List.tsx:34

⚠️ PERFORMANCE: Missing key prop in list

Code:
  34: items.map((item) => (
  35:     <ListItem item={item} />  ← No key
  36: ))

Problem:
  - React can't efficiently track list items
  - May cause unnecessary re-renders
  - Can lead to state bugs

Fix:
  items.map((item) => (
      <ListItem key={item.id} item={item} />
  ))

Priority: HIGH
Impact: Performance, correctness
```

### Step 2.2: Bundle Size Analysis

```bash
# Run bundle analyzer (if available)
cd frontend && npm run analyze 2>/dev/null || echo "No bundle analyzer configured"
```

Look for:
- Large dependencies (>100KB)
- Duplicate dependencies
- Tree-shaking opportunities

**Example Output**:
```
⚠️ PERFORMANCE: Large dependencies detected

Analysis:
  - moment.js: 288KB (could use date-fns: 13KB)
  - lodash: 71KB (could use lodash-es for tree-shaking)
  - @mui/material: imported entirely (400KB), only using 5 components

Recommendations:
  1. Replace moment with date-fns:
     import { format } from 'date-fns';

  2. Use tree-shakeable lodash:
     import debounce from 'lodash-es/debounce';

  3. Import MUI components individually:
     import Button from '@mui/material/Button';

Estimated savings: ~600KB (minified)

Priority: HIGH
Impact: Load time, user experience
```

---

## Phase 3: Architecture & Organization

### Step 3.1: Code Duplication

```bash
# Find similar component patterns
find frontend/src/components -name "*.tsx" -exec wc -l {} \; | sort -rn | head -20
```

Manually review large files for duplication.

**Example Output**:
```
⚠️ ARCHITECTURE: Duplicated form logic

Files with similar patterns:
  - frontend/src/components/UserForm.tsx (150 lines)
  - frontend/src/components/ProductForm.tsx (145 lines)
  - frontend/src/components/OrderForm.tsx (142 lines)

Duplication:
  - Form validation logic (30 lines each)
  - Submit handling (20 lines each)
  - Error display (15 lines each)

Recommendation:
  Create custom hook:

  // useForm.ts
  export function useForm<T>(initialValues: T, onSubmit: (values: T) => void) {
      const [values, setValues] = useState(initialValues);
      const [errors, setErrors] = useState<Record<string, string>>({});

      const handleSubmit = async (e: FormEvent) => {
          e.preventDefault();
          // Shared validation + submit logic
      };

      return { values, errors, handleSubmit, setValues };
  }

  // Usage in each form:
  const { values, errors, handleSubmit } = useForm(initialValues, handleFormSubmit);

Priority: MEDIUM
Impact: Maintainability, DRY principle
```

### Step 3.2: Lazy Loading Opportunities

```bash
# Find large pages without lazy loading
grep -rn "import.*from.*pages" frontend/src/ --include="*.tsx" | grep -v "React.lazy"
```

**Example Output**:
```
⚠️ PERFORMANCE: Missing lazy loading on routes

Current imports:
  import { HomePage } from './pages/HomePage';
  import { SettingsPage } from './pages/SettingsPage';
  import { AdminPage } from './pages/AdminPage';

Problem:
  - All pages loaded upfront
  - Increases initial bundle size
  - Slower initial load

Fix:
  const HomePage = React.lazy(() => import('./pages/HomePage'));
  const SettingsPage = React.lazy(() => import('./pages/SettingsPage'));
  const AdminPage = React.lazy(() => import('./pages/AdminPage'));

  // Wrap routes with Suspense:
  <Suspense fallback={<LoadingSpinner />}>
      <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/admin" element={<AdminPage />} />
      </Routes>
  </Suspense>

Priority: HIGH
Impact: Initial load time
```

---

## Phase 4: Accessibility (a11y)

### Step 4.1: Missing ARIA Labels

```bash
# Find buttons/links without aria-label or text
grep -rn "<button\|<a href" frontend/src/ --include="*.tsx" | grep -v "aria-label\|children"
```

**Example Output**:
```
📍 frontend/src/components/Header.tsx:23

⚠️ ACCESSIBILITY: Missing accessible label

Code:
  23: <button onClick={handleMenu}>
  24:     <MenuIcon />
  25: </button>

Problem:
  - Screen readers can't identify button purpose
  - No text or aria-label

Fix:
  <button onClick={handleMenu} aria-label="Open menu">
      <MenuIcon />
  </button>

Priority: HIGH
Impact: Accessibility compliance
```

### Step 4.2: Semantic HTML

```bash
# Find div soup (many nested divs)
grep -rn "<div><div><div>" frontend/src/ --include="*.tsx"
```

**Example Output**:
```
⚠️ ACCESSIBILITY: Non-semantic markup

Code:
  <div className="header">
      <div className="nav">
          <div className="link">Home</div>
      </div>
  </div>

Fix:
  <header>
      <nav>
          <a href="/">Home</a>
      </nav>
  </header>

Priority: MEDIUM
Impact: SEO, accessibility
```

---

## Phase 5: Testing

### Step 5.1: Untested Components

```bash
# Find components without test files
for file in frontend/src/components/*.tsx; do
    base=$(basename "$file" .tsx)
    if [ ! -f "frontend/src/components/${base}.test.tsx" ]; then
        echo "Missing test: $file"
    fi
done
```

**Example Output**:
```
⚠️ TESTING: Missing test coverage

Untested components:
  - frontend/src/components/SearchBar.tsx
  - frontend/src/components/ResultsList.tsx
  - frontend/src/components/FilterPanel.tsx

Recommendation:
  Create test files with basic coverage:

  // SearchBar.test.tsx
  import { render, screen, fireEvent } from '@testing-library/react';
  import { SearchBar } from './SearchBar';

  describe('SearchBar', () => {
      it('renders input field', () => {
          render(<SearchBar onSearch={jest.fn()} />);
          expect(screen.getByRole('textbox')).toBeInTheDocument();
      });

      it('calls onSearch when submitted', () => {
          const onSearch = jest.fn();
          render(<SearchBar onSearch={onSearch} />);

          fireEvent.change(screen.getByRole('textbox'), {
              target: { value: 'test query' }
          });
          fireEvent.submit(screen.getByRole('form'));

          expect(onSearch).toHaveBeenCalledWith('test query');
      });
  });

Priority: MEDIUM
Impact: Test coverage, confidence in changes
```

---

## Phase 6: UI/UX Consistency

### Step 6.1: Hardcoded Values

```bash
# Find hardcoded colors
grep -rn "color: '#\|backgroundColor: '#" frontend/src/ --include="*.tsx" --include="*.css"
```

**Example Output**:
```
⚠️ UI CONSISTENCY: Hardcoded colors

Locations:
  - frontend/src/components/Button.tsx:23: color: '#1976d2'
  - frontend/src/components/Header.tsx:45: backgroundColor: '#ffffff'
  - frontend/src/pages/Dashboard.tsx:67: color: '#333333'

Problem:
  - Can't switch themes easily
  - Inconsistent color usage
  - Hard to maintain

Fix:
  // Define theme
  const theme = {
      colors: {
          primary: '#1976d2',
          background: '#ffffff',
          text: '#333333'
      }
  };

  // Use in components
  <Button style={{ color: theme.colors.primary }} />

  // Or use CSS variables
  :root {
      --color-primary: #1976d2;
      --color-background: #ffffff;
  }

  .button {
      color: var(--color-primary);
  }

Priority: MEDIUM
Impact: Theming, maintainability
```

---

## Phase 7: Security

### Step 7.1: XSS Vulnerabilities

```bash
# Find dangerouslySetInnerHTML usage
grep -rn "dangerouslySetInnerHTML" frontend/src/ --include="*.tsx"
```

**Example Output**:
```
📍 frontend/src/components/Content.tsx:34

⚠️ SECURITY: Potential XSS vulnerability

Code:
  34: <div dangerouslySetInnerHTML={{ __html: userContent }} />

Problem:
  - Renders user-provided HTML without sanitization
  - XSS attack vector if userContent is untrusted

Fix:
  import DOMPurify from 'dompurify';

  <div dangerouslySetInnerHTML={{
      __html: DOMPurify.sanitize(userContent)
  }} />

Priority: CRITICAL
Impact: Security vulnerability
```

### Step 7.2: Sensitive Data in Client

```bash
# Find potential secrets in code
grep -rn "api.*key\|secret\|password" frontend/src/ --include="*.ts" --include="*.tsx" -i
```

**Example Output**:
```
⚠️ SECURITY: Potential secret in client code

Code:
  const API_KEY = "sk_live_abc123...";  ← Hardcoded secret

Problem:
  - API key visible in client code
  - Exposed in production bundle
  - Security risk

Fix:
  // Use environment variables
  const API_KEY = import.meta.env.VITE_API_KEY;

  // In .env (NOT committed)
  VITE_API_KEY=sk_live_abc123...

  // For public API keys, this is OK
  // For private keys, use backend proxy

Priority: CRITICAL (if private key)
Impact: Security breach
```

---

## Phase 8: Dead Code Analysis

### Step 8.1: Run Dead Code Detector

**Detect package manager**:
```bash
if [ -f "frontend/package-lock.json" ]; then
    PKG_MGR="npm"
elif [ -f "frontend/pnpm-lock.yaml" ]; then
    PKG_MGR="pnpm"
elif [ -f "frontend/yarn.lock" ]; then
    PKG_MGR="yarn"
else
    PKG_MGR="npm"
fi
```

**Run analyzer**:
```bash
cd frontend

# Check if knip is configured
if grep -q "knip" package.json; then
    echo "Running dead code analysis..."
    $PKG_MGR run dead-code 2>&1 || echo "No dead-code script found"
else
    echo "⚠️  No dead code analyzer (knip) configured"
    echo "To add: npm install -D knip"
fi
```

**Example Output**:
```
===============================================================================
DEAD CODE ANALYSIS
===============================================================================

Unused exports (12):
  - src/utils/formatDate.ts: formatDateRange (exported but never imported)
  - src/hooks/useDebounce.ts: default (exported but never imported)
  - src/components/Button.tsx: ButtonProps (type exported but never used)

Unused files (3):
  - src/utils/oldHelper.ts (not imported anywhere)
  - src/components/DeprecatedModal.tsx (not imported anywhere)

Unused dependencies (5):
  - moment (in package.json, not imported)
  - classnames (in package.json, not imported)

Duplicate exports (2):
  - src/utils/index.ts and src/utils/helpers.ts both export 'formatCurrency'

Recommendations:
  1. Remove unused exports or files
  2. Remove unused dependencies:
     npm remove moment classnames
  3. Consolidate duplicate exports

Estimated cleanup: 450 lines, 3 files, ~500KB dependencies
```

**Auto-fix** (optional):
```bash
# Let knip automatically fix removable issues
$PKG_MGR run dead-code:fix

echo "⚠️  Review changes carefully before committing"
echo "Some 'unused' code may be:"
echo "  - Entry points (main.tsx)"
echo "  - Used by external tools"
echo "  - Dynamic imports"
```

---

## Output Format

### Executive Summary

```
===============================================================================
FRONTEND REVIEW REPORT
===============================================================================

Files Analyzed: 127 (.ts/.tsx files)
Analysis Date: 2026-01-08

ISSUES FOUND:

CRITICAL (Security/Breaking):
  [ ] 2 XSS vulnerabilities (dangerouslySetInnerHTML)
  [ ] 1 hardcoded API key

HIGH (Performance/Accessibility):
  [ ] 8 components without memoization
  [ ] 12 missing React keys
  [ ] 15 missing ARIA labels
  [ ] 5 routes without lazy loading

MEDIUM (Code Quality/Maintainability):
  [ ] 23 uses of 'any' type
  [ ] 7 instances of prop drilling
  [ ] 18 duplicated code blocks
  [ ] 12 untested components
  [ ] 31 hardcoded theme values

LOW (Nice to Have):
  [ ] 12 unused exports
  [ ] 5 unused dependencies
  [ ] 8 non-semantic HTML sections

TOTAL ISSUES: 144

ESTIMATED FIX TIME:
  Critical: 2 hours   (fix immediately)
  High:     1 day     (this sprint)
  Medium:   3 days    (next sprint)
  Low:      4 hours   (cleanup sprint)

BUNDLE SIZE OPPORTUNITIES:
  Potential savings: ~600KB (45% reduction)
  - Replace moment.js with date-fns
  - Tree-shake lodash imports
  - Lazy load admin routes

===============================================================================
```

---

## Success Criteria

✓ **All frontend files scanned** (.ts/.tsx in frontend/src/)
✓ **TypeScript quality checked** (any usage, type safety)
✓ **Performance analyzed** (memoization, keys, bundle)
✓ **Architecture reviewed** (duplication, organization)
✓ **Accessibility verified** (ARIA, semantic HTML)
✓ **Testing gaps identified** (missing test files)
✓ **Security checked** (XSS, secrets)
✓ **Dead code detected** (if knip configured)
✓ **Report generated** with prioritized issues

---

## Notes

- **Works with any React/TypeScript project**: Generic patterns
- **Package manager agnostic**: Detects npm/pnpm/yarn
- **Dead code requires knip**: Install if not present
- **Fix CRITICAL first**: Security issues are urgent
- **Test after changes**: Run tests after applying fixes

---

## Example Workflow

```bash
# 1. Run frontend review
/frontend

# 2. Review report (144 issues found)
# - 3 CRITICAL (security)
# - 40 HIGH
# - 101 MEDIUM/LOW

# 3. Fix CRITICAL immediately
# - Sanitize dangerouslySetInnerHTML
# - Move API key to environment variable

# 4. Fix HIGH this sprint
# - Add React.memo to 8 components
# - Add keys to 12 lists
# - Add ARIA labels to 15 buttons

# 5. Run dead code cleanup
cd frontend
pnpm dead-code:fix

# 6. Verify changes
npm run lint
npm run test
npm run build

# 7. Re-run review
/frontend
# → CRITICAL: 0
# → HIGH: 10 (reduced from 40)

# 8. Commit
git add frontend/
git commit -m "fix: resolve frontend security and performance issues"
```
