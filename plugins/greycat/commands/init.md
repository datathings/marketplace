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

This file provides guidance to Claude Code when working with this GreyCat project.

## ⚠️ CRITICAL DEVELOPMENT RULES

### 1. ALWAYS LINT AFTER EACH CHANGE
\`\`\`bash
greycat-lang lint  # Run IMMEDIATELY after ANY code change
\`\`\`
- Verify 0 errors before proceeding
- Never make multiple changes without linting between them
- If lint fails, FIX IMMEDIATELY before continuing

### 2. VERIFY DEPENDENCIES BEFORE DELETING
- Use `Grep` to search for usages across codebase before removing types/functions/variables
- Check imports, indexers, and API endpoints
- Run `greycat-lang lint` after deletions to catch missing references

### 3. GREYCAT LANGUAGE GOTCHAS

**String Operations**:
\`\`\`gcl
// ❌ String.substring() - doesn't exist
var preview = text.substring(0, 100);

// ✅ String.slice(from, to) - use this
var preview = text.slice(0, 100);
\`\`\`

**Generic Type Parameters**:
\`\`\`gcl
// ❌ WRONG - Generic params on static functions not allowed
abstract type MyService {
    static fn process<T>(value: T): T { }  // ERROR!
}

// ✅ CORRECT - Non-static methods with generics
type MyHelper<T> {
    fn process(value: T): T { return value; }
}
// Usage: MyHelper<int>{}.process(42)
\`\`\`

**Collection Initialization**:
\`\`\`gcl
// ❌ WRONG - Using ::new()
var list = Array<String>::new();

// ✅ CORRECT - Using {}
var list = Array<String> {};
var map = Map<String, int> {};
\`\`\`

**Persistent Collections (Local Variables)**:
\`\`\`gcl
// ❌ WRONG - nodeList for local variable
fn buildResults() {
    var results = nodeList<T> {};  // Should be Array!
}

// ✅ CORRECT - Array for non-persistent data
fn buildResults() {
    var results = Array<T> {};
}
\`\`\`

### 4. USE GREYCAT SKILL FOR BACKEND WORK
**⚠️ MANDATORY**: For ANY GreyCat backend work (GCL files, graph operations, API endpoints), use `/greycat` skill or invoke via Skill tool.

---

## Common Commands

\`\`\`bash
# Backend (GreyCat)
greycat build              # Build project
greycat-lang lint          # Lint (RUN AFTER EACH CHANGE!)
greycat-lang fmt -p project.gcl -w  # Format all .gcl files
greycat-lang fmt <file> -w          # Format specific file
greycat run [function]     # Run function (default: main)
greycat serve              # Start server (port 8080)
greycat test               # Run tests
greycat codegen ts         # Generate TypeScript types → project.d.ts
greycat install            # Install libraries from project.gcl

# Frontend (if exists)
cd frontend
npm install                # First time setup
npm run dev                # Dev server (proxies to backend)
npm run build              # Production build
npm run lint               # Lint TypeScript/React
npm run test               # Run tests
\`\`\`

---

## Project Structure

Standard GreyCat project layout:

\`\`\`
.
├── project.gcl                 # Entry point, libraries, permissions
├── backend/
│   ├── src/
│   │   ├── model/              # Data types and global indices
│   │   ├── service/            # Business logic services
│   │   ├── api/                # REST API endpoints (@expose)
│   │   └── edi/                # Import/export logic
│   └── test/                   # Test files (*_test.gcl)
├── frontend/                   # React/TypeScript frontend (optional)
├── data/                       # Data files, models (optional)
├── lib/                        # Installed GreyCat libraries
├── gcdata/                     # Database storage (gitignored)
└── CLAUDE.md                   # This file
\`\`\`

---

## Development Workflow

### Making Changes

1. **Make code changes** to `.gcl` files
2. **Run linter immediately**:
   \`\`\`bash
   greycat-lang lint
   \`\`\`
3. **Fix any errors** before proceeding
4. **Test your changes**:
   \`\`\`bash
   greycat test
   \`\`\`
5. **[If has frontend] Update TypeScript types**:
   \`\`\`bash
   greycat codegen ts
   \`\`\`
6. **Verify runtime**:
   \`\`\`bash
   greycat serve  # Test in browser or via API
   \`\`\`

### Before Committing

\`\`\`bash
# 1. Format code
greycat-lang fmt -p project.gcl -w

# 2. Lint passes
greycat-lang lint

# 3. Tests pass
greycat test

# 4. Build succeeds
greycat build

# 5. Review changes
git status
git diff
\`\`\`

---

## GreyCat Language Patterns

### Nullability

All types are non-null by default. Use `?` for nullable:

\`\`\`gcl
var city: City?;                    // nullable
city?.name?.size();                 // optional chaining
city?.name ?? "Unknown";            // nullish coalescing
data.get("key")!!;                  // non-null assertion

if (country == null) { return null; }
return country->name;               // ✅ no !! needed after null check
\`\`\`

**⚠️ Cast + coalescing needs parens**:
\`\`\`gcl
// WRONG: answer as String? ?? "default"
// RIGHT:
(answer as String?) ?? "default"
\`\`\`

**⚠️ NO TERNARY OPERATOR** — use if/else:
\`\`\`gcl
var result: String;
if (valid) { result = "yes"; } else { result = "no"; }
\`\`\`

### Nodes (Persistence)

Nodes are 64-bit refs to persistent containers:

\`\`\`gcl
type Country { name: String; code: int; }
var obj = Country { name: "LU", code: 352 };  // RAM only
var n = node<Country>{obj};                    // persisted

*n;              // dereference
n->name;         // ✅ arrow: deref + field
n.resolve();     // method
n->name = "X";   // modify field
node<int>{0}.set(5);  // primitives use .set()
\`\`\`

### Indexed Collections

| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `node<T>` | — | `Array<T>`, `Map<K,V>` |
| `nodeList<node<T>>` | int | `Stack<T>`, `Queue<T>` |
| `nodeIndex<K, node<V>>` | hash | `Set<T>`, `Tuple<A,B>` |
| `nodeTime<node<T>>` | time | `Buffer`, `Table`, `Tensor` |
| `nodeGeo<node<T>>` | geo | `TimeWindow`, `SlidingWindow` |

**⚠️ CRITICAL: Initialize Collection Attributes**:
\`\`\`gcl
// ✅ Correct — initialize collections on creation
var city = node<City>{ City{
    name: "Paris",
    streets: nodeList<node<Street>>{}   // ⚠️ MUST initialize!
}};
\`\`\`

### API Endpoints

\`\`\`gcl
// In backend/src/api/xxx_api.gcl

@expose
@permission("app.user")
fn search(query: String): SearchResults {
    // Implementation
}

// Response types MUST be @volatile
@volatile
type SearchResults {
    items: Array<ResultView>;
    total: int;
}
\`\`\`

### Services

\`\`\`gcl
// In backend/src/service/xxx_service.gcl

abstract type SearchService {
    static fn executeQuery(query: String): Array<Result> {
        // Implementation
    }
}

// Usage:
var results = SearchService::executeQuery("test");
\`\`\`

### Documentation (REQUIRED)

All functions and types MUST have documentation:

\`\`\`gcl
/// Retrieve document by ID with automatic identifier detection.
/// @param id Document identifier (CELEX, ECLI, or internal ID)
/// @return Document object with full details
/// @throws NotFoundError if document not found
/// @example document("62009CJ0204")
fn document(id: String): Document { }

/// Search results container.
/// @volatile Required for API response types
@volatile
type SearchResults {
    /// List of matching items
    items: Array<ResultView>;
    /// Total count for pagination
    total: int;
}
\`\`\`

**Section Headers** for organizing code:
\`\`\`gcl
// ============================================================================
// SEARCH OPERATIONS
// ============================================================================

fn search(...) { }
fn advancedSearch(...) { }

// ============================================================================
// DOCUMENT RETRIEVAL
// ============================================================================

fn getDocument(...) { }
\`\`\`

### Error Handling (MANDATORY)

**All @expose functions MUST have try/catch with error logging**:

\`\`\`gcl
@expose
@permission("app.user")
fn document(id: String): Document {
    try {
        var doc = documents_by_id.get(id);
        if (doc == null) {
            throw NotFoundError { message: "Document not found", id: id };
        }
        return toDocumentView(doc.resolve());
    } catch (ex) {
        error("document(${id}) failed: ${ex}");  // Function name + args
        throw ex;
    }
}
\`\`\`

**Typed Errors** (create in model/ or api/):
\`\`\`gcl
@volatile
type NotFoundError {
    message: String;
    id: String;
}

@volatile
type ValidationError {
    message: String;
    field: String;
}

@volatile
type ServiceError {
    message: String;
}
\`\`\`

**Rules**:
- Never throw raw strings: \`throw "Not found"\` ❌
- Always use typed errors: \`throw NotFoundError { ... }\` ✅
- Re-throwing is fine: \`throw ex;\` ✅
- Log with function name and args: \`error("funcName(${arg}) failed: ${ex}")\`

---

## Common Pitfalls

| ❌ Don't | ✅ Do |
|---------|-------|
| `String.substring()` | `String.slice(from, to)` |
| Delete types without checking | Grep first, verify no usages |
| Multiple changes without linting | Lint after each change |
| `static fn process<T>` | Remove static OR remove generics |
| `Array<T>::new()` | `Array<T> {}` |
| `nodeList<T>` for local vars | `Array<T>` for non-persistent |
| Missing @volatile on API types | Always add @volatile |
| Uninitialized collections | Initialize in constructor |
| `throw "error message"` | `throw TypedError { message: "..." }` |
| @expose without try/catch | Always wrap in try/catch + error() |
| Functions without /// docs | Document all functions with /// |
| Skip formatting | Run `greycat-lang fmt` before commit |

---

## Database Management

**⚠️ Development Mode**: No migrations, delete deprecated fields immediately.

**Reset Database**:
\`\`\`bash
rm -rf gcdata           # ⚠️ DELETES ALL DATA - ask user first!
greycat run import      # Reimport from data files (if applicable)
\`\`\`

**Backup**:
\`\`\`bash
tar -czf gcdata-backup.tar.gz gcdata/
\`\`\`

---

## Testing

### Writing Tests

**Naming Convention**: \`test_functionName_scenario\` (snake_case)

\`\`\`gcl
// In backend/test/my_test.gcl

// ============================================================================
// MY SERVICE TESTS
// ============================================================================

@test
fn test_process_validInput() {
    var result = MyService::process("input");
    Assert::equals(result, "expected");
}

@test
fn test_process_nullInput() {
    var result = MyService::process(null);
    Assert::isTrue(result == null);
}

@test
fn test_process_emptyString() {
    var result = MyService::process("");
    Assert::equals(result, "");
}

// ============================================================================
// EDGE CASES
// ============================================================================

@test
fn test_process_specialCharacters() {
    var result = MyService::process("test@#$%");
    Assert::notNull(result);
}
\`\`\`

**Test Patterns**:
- Use \`// ===\` section headers to organize tests by feature/function
- Name tests: \`test_functionName_scenario\`
- Use \`Assert\` class: \`equals\`, \`isTrue\`, \`isFalse\`, \`notNull\`, \`isNull\`
- One assertion focus per test when possible

### Running Tests

\`\`\`bash
greycat test                           # Run all tests
greycat test backend/test/my_test.gcl  # Run specific test file
\`\`\`

---

## Debugging

\`\`\`gcl
println("msg")          # Console output
info("msg")             # Info level log
warn("msg")             # Warning
error("msg")            # Error
pprint(object)          # Formatted output
\`\`\`

---

## Authentication & Permissions

**Define in project.gcl**:
\`\`\`gcl
@permission("app.admin", "admin permission");
@permission("app.user", "user permission");

@role("admin", "app.admin", "app.user", "public", "admin", "api");
@role("user", "app.user", "public", "api");
@role("public", "public", "api");
\`\`\`

**Use in code**:
\`\`\`gcl
@expose
@permission("app.user")  // Requires authentication
fn search(query: String): Results {
    var user = SecurityService::getLoggedUser();  // Get current user
    // ...
}

@expose
@permission("public")  // No authentication required
fn getPublicData(): Data {
    // ...
}
\`\`\`

---

## Library Management

**Update libraries**:
\`\`\`bash
# Edit project.gcl with new versions
@library("std", "7.6.10-dev");
@library("explorer", "7.6.10-dev");

# Install updated libraries
greycat install

# Verify
greycat-lang lint
\`\`\`

**Common libraries**:
- `std` - Standard library (required)
- `explorer` - Graph explorer UI (recommended for dev)
- `ai` - LLM/embedding support
- `algebra` - ML pipelines
- `sql` - PostgreSQL integration
- `kafka` - Apache Kafka integration

---

## Troubleshooting

**Lint Errors**:
\`\`\`bash
# Run linter and review errors
greycat-lang lint

# Common issues:
# - Missing imports → Check @include in project.gcl
# - Type mismatch → Check type definitions
# - Unknown function → Check spelling, imports
\`\`\`

**Server Won't Start**:
\`\`\`bash
# Check port availability
lsof -i :8080

# Reset database if corrupted
rm -rf gcdata && greycat run import
\`\`\`

**Tests Failing**:
\`\`\`bash
# Run specific test to isolate issue
greycat test backend/test/failing_test.gcl

# Add debug output
println("Debug: ${variable}");
\`\`\`

---

## Backend Consistency Checklist

**Before every commit**:

\`\`\`
[ ] greycat-lang lint shows 0 errors
[ ] greycat-lang fmt -p project.gcl -w applied (code formatted)
[ ] All @expose functions have try/catch with error() logging
[ ] All functions/types have /// documentation
[ ] Transient/response types marked @volatile
[ ] Abstract service types documented
[ ] Tests follow test_functionName_scenario naming
[ ] greycat test passes
\`\`\`

---

## Additional Resources

- **GreyCat Documentation**: https://doc.greycat.io
- **GreyCat Skill**: Use `/greycat` command in Claude Code for expert help
- **Language Server**: Configure LSP for IDE integration (optional)

---

**Last Updated**: [Auto-generate timestamp]
```

---

## Execution Steps

### Step 1: Check if CLAUDE.md Exists

```bash
if [ -f "CLAUDE.md" ]; then
    echo "⚠️  CLAUDE.md already exists"
    echo "Options:"
    echo "  A) Backup existing and create new"
    echo "  B) Append generic rules to existing"
    echo "  C) Cancel"
    # Ask user for choice
else
    echo "✓ No CLAUDE.md found, creating new file"
fi
```

### Step 2: Detect Project Features

```bash
# Check for frontend
HAS_FRONTEND=false
if [ -d "frontend" ] || [ -d "ui" ]; then
    HAS_FRONTEND=true
fi

# Check for tests
HAS_TESTS=false
if [ -d "backend/test" ]; then
    HAS_TESTS=true
fi

# Check for data import
HAS_DATA_IMPORT=false
if [ -d "data" ] || grep -q "fn import(" project.gcl; then
    HAS_DATA_IMPORT=true
fi
```

### Step 3: Generate CLAUDE.md

Use Write tool to create CLAUDE.md with template above, customizing sections based on detected features:

- If no frontend: Remove frontend commands section
- If no tests: Simplify testing section
- If no data import: Remove data import commands

### Step 4: Report

```
===============================================================================
PROJECT INITIALIZED
===============================================================================

Created: CLAUDE.md (2,340 lines)

Sections included:
  ✓ Critical development rules
  ✓ GreyCat language patterns
  ✓ Common commands
  ✓ Project structure
  ✓ Development workflow
  [✓ Frontend commands] (detected frontend/)
  [✓ Testing guide] (detected backend/test/)
  ✓ Database management
  ✓ Troubleshooting

Next steps:
  1. Review CLAUDE.md and customize for your project
  2. Add project-specific sections (architecture, data model, etc.)
  3. Commit to repository:
     git add CLAUDE.md
     git commit -m "docs: add Claude Code development guide"

===============================================================================
```

---

## Customization

After generation, developers should add project-specific sections:

### Additional Sections to Add

**Project Overview**:
```markdown
## Project Overview

[Brief description of what this project does]

**Technology Stack**:
- Backend: GreyCat + [specific libraries]
- [If frontend] Frontend: React + TypeScript
- [Other technologies]

**Key Features**:
- [Feature 1]
- [Feature 2]
```

**Data Model**:
```markdown
## Data Model

**Core Types**:
- **Document** - [Description]
- **User** - [Description]
[etc.]

**Relationships**:
[Explain key relationships]
```

**Architecture**:
```markdown
## Architecture

[Project-specific architecture details]
```

---

## Success Criteria

✓ **CLAUDE.md created** in project root
✓ **Generic rules included** (linting, gotchas, workflows)
✓ **GreyCat patterns documented** (nullability, nodes, collections)
✓ **Commands listed** (build, test, serve, etc.)
✓ **Common pitfalls documented** (what to avoid)
✓ **Customized for project features** (frontend, tests, data)

---

## Notes

- **Generic template**: Applies to any GreyCat project
- **No project-specific details**: Developers add these after generation
- **Always up-to-date**: Based on latest GreyCat best practices
- **Claude Code optimized**: Designed for Claude Code workflows
- **Can be regenerated**: Safe to run multiple times (with backup option)

---

## Example Workflow

```bash
# 1. Start new GreyCat project
greycat init my-project
cd my-project

# 2. Initialize Claude Code documentation
/init

# 3. CLAUDE.md created
# Review and customize for your project

# 4. Add project-specific sections
# - Architecture overview
# - Data model details
# - Specific API endpoints

# 5. Commit
git add CLAUDE.md
git commit -m "docs: initialize Claude Code development guide"

# 6. Start development with Claude Code
# Claude now has full context of GreyCat best practices
```
