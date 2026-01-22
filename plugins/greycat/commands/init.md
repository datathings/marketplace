---
name: init
description: Initialize CLAUDE.md with generic GreyCat development guidelines
allowed-tools: Write, Read, Bash
---

# Initialize GreyCat Project Documentation

**Purpose**: Generate CLAUDE.md file with generic GreyCat development best practices and workflow guidelines

**Run When**: Starting new GreyCat project, setting up Claude Code integration

---

## Overview

This command creates a `CLAUDE.md` file with:

1. **Critical Development Rules** - Always lint, verify dependencies, common gotchas
2. **GreyCat Language Guide** - Syntax patterns, common mistakes, best practices
3. **Development Workflow** - Commands, testing, debugging
4. **Project Structure** - Standard GreyCat project layout
5. **Common Pitfalls** - What to avoid when developing with GreyCat

---

## CLAUDE.md Template

```markdown
# CLAUDE.md

[One-line project description]

## ⚠️ CRITICAL RULES

### 1. ALWAYS LINT AFTER EACH CHANGE
\`\`\`bash
greycat-lang lint  # Run IMMEDIATELY after ANY code change - verify 0 errors
\`\`\`

### 2. VERIFY BEFORE DELETING
Use \`Grep\` to search for usages before removing types/functions/variables.

### 3. GREYCAT GOTCHAS
\`\`\`gcl
// ❌ String.substring() doesn't exist
// ✅ String.slice(from, to)

// ❌ Static methods can't have generics
abstract type MyService { static fn process<T>(v: T): T { } }

// ✅ Non-static methods with generics
type MyHelper<T> { fn process(v: T): T { return v; } }

// ❌ Array<T>::new() | nodeList<T> for local vars
// ✅ Array<T> {} | Array<T> for non-persistent data
\`\`\`

### 4. USE GREYCAT SKILL
**Mandatory** for GCL backend work: use \`/greycat\` skill.

---

## Commands

\`\`\`bash
# Backend
greycat-lang lint          # Lint (after EVERY change!)
greycat-lang fmt -p project.gcl -w  # Format all .gcl files
greycat-lang fmt <file> -w          # Format specific file
greycat build/test/serve   # Build/test/start server (port 8080)
greycat run [function]     # Run function (default: main)
greycat codegen ts         # Generate project.d.ts
greycat install            # Install libraries

# Frontend (if exists) - run from root, package.json in root
pnpm install               # First time setup
pnpm dev                   # Dev server (proxies to backend)
pnpm build                 # Build frontend/ → webroot/
pnpm lint                  # Lint TypeScript/React
pnpm test                  # Run tests
\`\`\`

---

## Stack

**Backend**: GreyCat [version] (GCL)
**Frontend** (if exists): React + TypeScript + Vite, Tailwind CSS, React Router, TanStack Query
**Libraries**: \`@library("std", "[version]")\`, \`@library("explorer", "[version]")\`
**Testing**: Vitest + React Testing Library (backend: @test annotation)

**Frontend Setup**: Config files in root (package.json, vite.config.ts, tsconfig.json), source in frontend/, builds to webroot/
**Frontend Dependencies**: Use exact versions (e.g., \`"5.6.9"\` instead of \`"^5.6.9"\`)

---

## Project Structure

\`\`\`
.
├── project.gcl                 # Entry point, libraries, permissions
├── backend/
│   ├── src/
│   │   ├── model/              # Data types and global indices
│   │   ├── service/            # Business logic services
│   │   ├── api/                # REST API endpoints (@expose)
│   │   └── edi/                # Import/export logic (optional)
│   └── test/                   # Test files (*_test.gcl)
├── frontend/                   # Frontend source (if exists)
│   ├── src/
│   │   ├── pages/              # Page components
│   │   ├── components/         # Reusable components
│   │   ├── services/           # API clients
│   │   ├── hooks/              # Custom hooks
│   │   └── utils/              # Utilities
│   └── index.html              # HTML entry point
├── package.json                # Frontend deps (root level)
├── vite.config.ts              # Vite config (root, builds to webroot/)
├── tsconfig.json               # TypeScript config (root level)
├── .gitignore                  # Git ignore rules (GreyCat essentials)
├── webroot/                    # Built frontend (gitignored, served by GreyCat)
├── lib/                        # Installed GreyCat libraries (gitignored)
├── gcdata/                     # Database storage (gitignored)
└── CLAUDE.md                   # This file
\`\`\`

**Frontend Build**: Vite builds frontend/ → webroot/, GreyCat serves webroot/ on \`GREYCAT_WEBROOT=webroot\`

---

## .gitignore

Essential entries for GreyCat projects:

\`\`\`gitignore
# GreyCat
/gcdata                    # Database storage
/lib                       # Installed GreyCat libraries
/webroot                   # Built frontend
project.d.ts               # Generated TypeScript types
project.gcp                # Project cache
/project_types             # Generated type files

# Frontend dependencies
/node_modules
/.pnp
.pnp.js

# Build outputs
/dist
/build
/coverage
/outputs
/generated

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Python (if using scripts)
__pycache__/
*.pyc
*.pyo

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# Testing
.playwright-mcp/

# Misc
.parcel-cache
\`\`\`

---

## Coding Style

### Backend (GreyCat/GCL)

**Documentation** (REQUIRED): \`///\` for ALL functions/types, @param/@return/@throws/@example, \`// ===\` section headers
\`\`\`gcl
/// Retrieve document by ID.
/// @param id Document identifier
/// @return Document object
/// @throws NotFoundError if not found
/// @example document("62009CJ0204")
@expose @permission("public")
fn document(id: String): Document { }
\`\`\`

**Services**: Abstract types with static methods, @volatile for transient types, null-safe with \`?\`
\`\`\`gcl
abstract type SearchService {
    static fn search(query: String): Array<Result> { }
}
@volatile  // REQUIRED for non-persisted types
type SearchResults { items: Array<ResultView>; total: int; }
\`\`\`

**Error Handling** (MANDATORY): try/catch on ALL @expose functions, typed errors only, \`error()\` logging
\`\`\`gcl
@expose @permission("public")
fn document(id: String): Document {
  try {
    var doc = documents_by_id.get(id);
    if (doc == null) {
      throw NotFoundError { message: "Not found", id: id };
    }
    return doc.resolve();
  } catch (ex) {
    error("document(${id}) failed: ${ex}");
    throw ex;
  }
}
\`\`\`

**Collections**: \`Array<T> {}\`, \`Map<K,V> {}\`, \`nodeIndex<K, node<V>>\`, initialize collection attributes
**Naming**: snake_case fields, camelCase functions

### Frontend (React/TypeScript) - if exists

**Components** (MANDATORY): Named export + memo, props interface ABOVE component, JSDoc
\`\`\`tsx
/**
 * Search results display with pagination
 */
interface SearchResultsProps {
  results: SearchResult[]
  isLoading: boolean
}
export const SearchResults = memo(function SearchResults({ results, isLoading }: SearchResultsProps) { })
\`\`\`
**Exception**: Default export for pages only

**Hooks**: use* prefix, useCallback w/ deps, useMemo for derived, return objects
**React Query**: queryKey arrays w/ all deps, staleTime config, enabled for conditional
**Services**: Named export objects, explicit return types from project.d.ts
**State**: URL (useSearchParams), localStorage, Context (theme/user)
**Styling**: Theme constants, Tailwind utilities, NO inline styles except dynamic values
**Naming**: camelCase (variables, functions), PascalCase (components, types)

### Testing

**Backend**: @test annotation, test_function_scenario naming (snake_case), Assert class, \`// ===\` headers
\`\`\`gcl
@test
fn test_search_validQuery() {
    var results = SearchService::search("test");
    Assert::notNull(results);
}
\`\`\`

**Frontend**: Nested describe blocks, fixtures for mock data, test helpers
\`\`\`tsx
describe('Component', () => {
  describe('Feature', () => {
    it('behavior with outcome', () => {
      render(<Component />)
      expect(screen.getByText('...')).toBeInTheDocument()
    })
  })
})
\`\`\`

---

## GreyCat Language Patterns

### Nullability
\`\`\`gcl
var city: City?;                    // nullable
city?.name?.size();                 // optional chaining
city?.name ?? "Unknown";            // nullish coalescing
(answer as String?) ?? "default"    // ⚠️ parens for cast + coalescing

if (country == null) { return null; }
return country->name;               // ✅ no !! after null check
\`\`\`

**⚠️ NO TERNARY** — use if/else

### Nodes (Persistence)
\`\`\`gcl
type Country { name: String; code: int; }
var n = node<Country>{ Country { name: "LU", code: 352 }};
n->name;         // arrow: deref + field
n.resolve();     // method
\`\`\`

### Indexed Collections
| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| \`node<T>\` | — | \`Array<T>\`, \`Map<K,V>\` |
| \`nodeList<node<T>>\` | int | \`Stack<T>\`, \`Queue<T>\` |
| \`nodeIndex<K, node<V>>\` | hash | \`Set<T>\`, \`Tuple<A,B>\` |
| \`nodeTime<node<T>>\` | time | \`Buffer\`, \`Table\`, \`Tensor\` |
| \`nodeGeo<node<T>>\` | geo | — |

**⚠️ CRITICAL**: Initialize collection attributes on creation

---

## Common Pitfalls

| ❌ Don't | ✅ Do |
|---------|-------|
| \`String.substring()\` | \`String.slice(from, to)\` |
| Delete without Grep | Grep first, verify no usages |
| Skip linting | Lint after EACH change |
| \`static fn process<T>\` | Remove static OR generics |
| \`Array<T>::new()\` | \`Array<T> {}\` |
| \`nodeList<T>\` for local vars | \`Array<T>\` for non-persistent |
| Missing @volatile | Always add @volatile |
| Uninitialized collections | Initialize in constructor |
| \`throw "error"\` | \`throw TypedError { ... }\` |
| @expose without try/catch | Always wrap + error() |
| Functions without /// docs | Document ALL functions |

---

## Database (DEV MODE)

**No migrations**: Delete deprecated fields, reset \`gcdata/\` freely, use nullable for new fields
**Reset**: \`rm -rf gcdata && greycat run import\`

---

## Environment

\`\`\`bash
# Backend (.env)
GREYCAT_PORT=8080
GREYCAT_WEBROOT=webroot      # Serve built frontend from webroot/
GREYCAT_CACHE=30000

# Frontend (.env) - if exists
VITE_GREYCAT_URL=http://localhost:8080
\`\`\`

**Vite Config** (vite.config.ts in root):
\`\`\`ts
export default defineConfig({
  root: 'frontend',           // Source files in frontend/
  build: {
    outDir: '../webroot',     // Build to webroot/ for GreyCat
    emptyOutDir: true
  }
})
\`\`\`

**TypeScript Config** (tsconfig.json in root):
\`\`\`json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["frontend/src/*"]
    }
  },
  "include": ["frontend/src"]
}
\`\`\`

---

## Auth & Permissions

**project.gcl**:
\`\`\`gcl
@permission("app.admin", "admin permission");
@permission("app.user", "user permission");
@role("admin", "app.admin", "app.user", "public", "admin", "api");
@role("user", "app.user", "public", "api");
\`\`\`

**Usage**: \`@permission("app.user")\`, \`SecurityService::getLoggedUser()\`

---

## Development Workflow

1. Use \`/greycat\` skill for backend work
2. \`greycat-lang lint\` after EVERY change (0 errors required)
3. \`greycat-lang fmt -p project.gcl -w\` to format all files
4. \`Grep\` before deleting (verify no usages)
5. \`greycat codegen ts\` after backend type changes
6. Test: \`greycat test\` (backend), \`pnpm test\` (frontend)

---

## Consistency Checklist

**Before commit**:
- [ ] \`greycat-lang lint\` shows 0 errors
- [ ] \`greycat-lang fmt -p project.gcl -w\` applied
- [ ] All @expose functions have try/catch with error() logging
- [ ] All functions/types have /// documentation
- [ ] Transient types marked @volatile
- [ ] Frontend: exact versions in package.json (no ^ or ~)
- [ ] Tests pass

---

## LSP (Language Server)

**Start**: \`greycat-lang server --stdio\`
**Features**: Autocomplete, hover docs, go-to-def, diagnostics, format, rename
**Use**: IDE integration for real-time feedback, BUT always \`greycat-lang lint\` before commit

---

More: https://doc.greycat.io/
```

---

## Execution Steps

### Step 1: Check if CLAUDE.md Exists

```bash
if [ -f "CLAUDE.md" ]; then
    echo "⚠️  CLAUDE.md already exists"
    echo "Options:"
    echo "  A) Backup existing and create new"
    echo "  B) Cancel"
    # Ask user for choice
else
    echo "✓ No CLAUDE.md found, creating new file"
fi
```

### Step 2: Detect Project Features

```bash
# Check for frontend (frontend/ directory or package.json with React)
HAS_FRONTEND=false
if [ -d "frontend" ] || ([ -f "package.json" ] && grep -q "react" package.json); then
    HAS_FRONTEND=true
fi

# Detect GreyCat version from project.gcl
GREYCAT_VERSION=$(grep '@library("std"' project.gcl | sed -E 's/.*"([^"]+)".*/\1/' || echo "[version]")
```

### Step 3: Check/Create .gitignore

```bash
if [ ! -f ".gitignore" ]; then
    echo "✓ No .gitignore found, will create with GreyCat essentials"
    CREATE_GITIGNORE=true
elif ! grep -q "gcdata" .gitignore; then
    echo "⚠️  .gitignore exists but missing GreyCat entries"
    echo "Will append GreyCat-specific entries"
    APPEND_GITIGNORE=true
else
    echo "✓ .gitignore exists with GreyCat entries"
fi
```

### Step 4: Generate CLAUDE.md

Use Write tool to create CLAUDE.md with template above, customizing:

- **Replace placeholders**: `[One-line project description]`, `[version]`
- **Remove frontend sections** if no frontend detected:
  - Frontend commands (pnpm commands)
  - Frontend section in Stack
  - Frontend structure in Project Structure
  - Frontend section in Coding Style
  - Frontend testing patterns
  - Frontend environment variables
- **Keep all backend sections** (always present in GreyCat projects)

If CREATE_GITIGNORE or APPEND_GITIGNORE, add/update .gitignore with essential GreyCat entries from template

### Step 5: Report

```
===============================================================================
GREYCAT PROJECT INITIALIZED
===============================================================================

Created: CLAUDE.md (~300 lines)
[Created/Updated: .gitignore with GreyCat essentials]

CLAUDE.md includes:
  ✓ Critical rules (lint, verify, gotchas)
  ✓ Commands (backend [+ frontend])
  ✓ Coding style (backend [+ frontend])
  ✓ GreyCat patterns (nullability, nodes, collections)
  ✓ .gitignore section (GreyCat essentials)
  ✓ Common pitfalls table
  ✓ Development workflow

.gitignore [created/updated]:
  ✓ /gcdata, /lib, /webroot (GreyCat)
  ✓ project.d.ts, project.gcp (generated)
  ✓ /node_modules, .env (frontend)

Next steps:
  1. Replace "[One-line project description]" in CLAUDE.md
  2. Add project-specific sections (data model, architecture)
  3. Commit:
     git add CLAUDE.md .gitignore
     git commit -m "docs: initialize Claude Code development guide"

===============================================================================
```

---

## Success Criteria

✓ **CLAUDE.md created** (~300 lines, compressed format)
✓ **.gitignore created/updated** with GreyCat essentials (gcdata, lib, webroot, project.d.ts, project.gcp)
✓ **Generic rules included** (linting, gotchas, workflows)
✓ **pnpm commands** (not npm)
✓ **Root-level configs** (package.json, vite.config.ts, tsconfig.json), source in frontend/
✓ **Vite builds to webroot/** (GreyCat-compatible)
✓ **Exact version note** for frontend deps
✓ **Customized for frontend presence**

---

## Notes

- **Compressed format**: ~300 lines vs 600 in old template
- **Generic template**: No project-specific details
- **GreyCat essentials**: Creates/updates .gitignore with gcdata/, lib/, webroot/, project.d.ts, project.gcp
- **Frontend structure**: All config in root (package.json, vite.config.ts, tsconfig.json), source in frontend/, builds to webroot/
- **Frontend preferences**: pnpm, exact versions (no ^ or ~), webroot/ for GreyCat compatibility
- **Can be regenerated**: Safe to run multiple times (with backup option)
