---
name: docs
description: Update README, API documentation, and MCP server documentation
allowed-tools: Bash, Read, Grep, Glob, Write, Edit
---

# Documentation Generator

**Purpose**: Generate and update comprehensive project documentation (README, API docs, MCP server docs)

**Run After**: Each sprint, before releases, when adding features/APIs

---

## Overview

This command generates three types of documentation:

1. **README.md** - Project overview, setup, architecture
2. **API Documentation** - OpenAPI/Markdown docs for @expose endpoints
3. **MCP Documentation** - Model Context Protocol server docs (if using @tag("mcp"))

---

## Phase 1: README.md Generation

### Objective
Create comprehensive README that reflects current project state.

### Step 1.1: Analyze Project Structure

```bash
# Detect project type and structure
echo "Analyzing project structure..."

# Check for frontend
HAS_FRONTEND=false
if [ -d "frontend" ]; then
    HAS_FRONTEND=true
fi

# Check for tests
HAS_TESTS=false
if [ -d "backend/test" ]; then
    HAS_TESTS=true
fi

# Count files and data
BACKEND_FILES=$(find backend/src -name "*.gcl" | wc -l)
TEST_FILES=$(find backend/test -name "*_test.gcl" 2>/dev/null | wc -l)
```

### Step 1.2: Extract Project Information

**A. From project.gcl**:
```bash
# Extract library versions
grep "@library" project.gcl

# Extract permissions and roles
grep "@permission\|@role" project.gcl

# Extract main function to understand entry point
```

**B. From Backend**:
```bash
# Find all types (data model)
grep -rn "^type [A-Z]" backend/src/model/ --include="*.gcl"

# Find all services
grep -rn "^abstract type.*Service" backend/src/service/ --include="*.gcl"

# Find all API endpoints
grep -rn "@expose" backend/src/api/ --include="*.gcl"
```

**C. From Frontend** (if exists):
```bash
# Check package.json for dependencies
cat frontend/package.json | grep -E "react|typescript|vite"

# Find pages
find frontend/src/pages -name "*.tsx" 2>/dev/null
```

**D. From Data**:
```bash
# Check for data files
ls data/ 2>/dev/null

# Check for embedding models
find data/ -name "*.gguf" 2>/dev/null
```

### Step 1.3: Generate README Content

**Template Structure**:

```markdown
# [Project Name]

> [Brief one-line description extracted from project purpose]

## Overview

[Auto-generated project description based on detected features]

**Technology Stack**:
- Backend: GreyCat [version] (GCL language)
- [If has_frontend] Frontend: React + TypeScript + [build tool]
- [If has AI libs] AI/ML: llama.cpp, embeddings
- [List other detected libraries: kafka, sql, etc.]

**Key Features**:
[Auto-detected from code analysis]
- Graph-based data storage with [X] node types
- [X] REST API endpoints
- [If has auth] User authentication and role-based access
- [If has vector index] Semantic search with vector embeddings
- [List other detected features]

## Quick Start

### Prerequisites

- GreyCat CLI (version [X] or later)
- [If has_frontend] Node.js 18+ and pnpm/npm
- [List other requirements detected]

### Installation

\`\`\`bash
# Clone the repository
git clone [repository-url]
cd [project-name]

# Install GreyCat libraries
greycat install

# [If has_frontend] Install frontend dependencies
cd frontend
pnpm install
cd ..
\`\`\`

### Running the Application

**Backend**:
\`\`\`bash
# Start GreyCat server
greycat serve

# Server will start on http://localhost:8080
# [If has explorer] Graph explorer available at http://localhost:8080/explorer
\`\`\`

**[If has_frontend] Frontend**:
\`\`\`bash
cd frontend
pnpm dev

# Frontend will start on http://localhost:3000
\`\`\`

**[If has data import] Data Import**:
\`\`\`bash
# Import initial data
greycat run import

# [If has vector data] Import vector embeddings
greycat run importVector
\`\`\`

## Architecture

### Data Model

**Core Node Types**:
[Auto-extracted from backend/src/model/]

\`\`\`
[Generate simple ASCII art or list of types with relationships]
\`\`\`

### Service Layer

[List all services with brief descriptions]

- **[ServiceName]** - [Purpose inferred from functions]
  - `functionName()` - [Description]

### API Endpoints

See [API Documentation](#api-documentation) for full details.

**Available Endpoints**:
[Auto-generated from @expose functions]

| Endpoint | Permission | Description |
|----------|------------|-------------|
| [function name] | [permission] | [purpose] |

## Development

### Project Structure

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
[If has_frontend]
├── frontend/
│   ├── src/
│   │   ├── pages/              # React pages
│   │   ├── components/         # React components
│   │   └── api/                # API client
│   └── package.json
[If has data]
├── data/                       # Data files, models, embeddings
└── README.md
\`\`\`

### Common Commands

**Backend (GreyCat)**:
\`\`\`bash
greycat build                   # Build project
greycat-lang lint               # Lint code (run after each change!)
greycat serve                   # Start server
greycat test                    # Run tests
greycat run [function]          # Run specific function
greycat codegen ts              # Generate TypeScript types
\`\`\`

**[If has_frontend] Frontend**:
\`\`\`bash
cd frontend
pnpm dev                        # Dev server
pnpm build                      # Production build
pnpm lint                       # Lint TypeScript/React
pnpm test                       # Run tests
\`\`\`

### Development Workflow

**CRITICAL: Always lint after each change**

\`\`\`bash
# 1. Make changes to .gcl files
# 2. Run linter immediately
greycat-lang lint

# 3. Fix any errors before proceeding
# 4. Test your changes
greycat test

# 5. [If has_frontend] Update frontend types if backend changed
greycat codegen ts
\`\`\`

## Testing

[If has_tests]
**Backend Tests**:
\`\`\`bash
greycat test                    # Run all tests
greycat test backend/test/specific_test.gcl  # Run specific test
\`\`\`

Current test coverage: [X test files, Y test functions]

[If has_frontend with tests]
**Frontend Tests**:
\`\`\`bash
cd frontend
pnpm test                       # Run tests
pnpm test:coverage              # Coverage report
\`\`\`

## Configuration

**Environment Variables** (.env):
[Auto-detect from .env or common patterns]

\`\`\`bash
GREYCAT_PORT=8080               # Server port
GREYCAT_CACHE=30000             # Cache size
[List other detected env vars]
\`\`\`

**Libraries** (project.gcl):
[List current library versions from project.gcl]

## Authentication & Permissions

[If has auth detected]

**Roles**:
[Extract from @role declarations in project.gcl]

**Permissions**:
[Extract from @permission declarations]

**Usage**:
\`\`\`gcl
// Get logged in user
var user = SecurityService::getLoggedUser();

// Require authentication
var user = SecurityService::requireLoggedUser();

// Check admin role
if (SecurityService::isAdmin()) { ... }
\`\`\`

## API Documentation

See full API documentation at [API.md](API.md) (auto-generated).

[Brief overview of main API categories]

## [If MCP detected] MCP Server

This project exposes GreyCat functions as Model Context Protocol (MCP) tools.

See [MCP.md](MCP.md) for full documentation.

## Database Management

**⚠️ Development Mode**: No migrations, delete deprecated fields immediately.

**Reset Database**:
\`\`\`bash
rm -rf gcdata                   # ⚠️ DELETES ALL DATA
greycat run import              # Reimport from data files
\`\`\`

**Backup**:
\`\`\`bash
tar -czf gcdata-backup.tar.gz gcdata/
\`\`\`

## Troubleshooting

**Lint Errors**:
\`\`\`bash
# Run linter and review errors
greycat-lang lint

# Common issues:
# - Missing imports → Add @include in project.gcl
# - Type mismatch → Check type definitions
# - Unknown function → Check spelling, imports
\`\`\`

**Server Won't Start**:
\`\`\`bash
# Check port availability
lsof -i :8080

# Check database integrity
rm -rf gcdata && greycat run import
\`\`\`

**[If has_frontend] Frontend API Errors**:
\`\`\`bash
# Ensure backend is running
greycat serve

# Regenerate TypeScript types
greycat codegen ts

# Check network proxy in vite.config.ts
\`\`\`

## Project Statistics

[Auto-generate current stats]

- Backend Files: [X] .gcl files
- Data Model: [Y] types
- API Endpoints: [Z] @expose functions
- Services: [N] service classes
- Tests: [M] test files with [P] test functions
- [If has_frontend] Frontend: [Q] pages, [R] components

---

**Built with GreyCat** - [https://greycat.io](https://greycat.io)
```

### Step 1.4: Content Guidelines

**✅ DO INCLUDE**:
- Technical details and architecture
- Setup and installation instructions
- API documentation references
- Development workflow
- Testing instructions
- Configuration details
- Troubleshooting tips
- Project statistics

**❌ DO NOT INCLUDE**:
- Contributor guidelines
- License information
- Author names or credits
- AI assistant mentions (Claude, etc.)
- Copyright notices
- Acknowledgments sections

### Step 1.5: Write README.md

```bash
# Use Write tool to create/update README.md
# Include all sections generated above
```

---

## Phase 2: API Documentation Generation

### Objective
Generate comprehensive API documentation from @expose endpoints.

### Step 2.1: Extract All @expose Functions

```bash
# Find all API endpoints
grep -rn "@expose" backend/src/api/ --include="*.gcl" -A 20
```

For each endpoint, extract:
- **Function name**
- **Parameters** (with types)
- **Return type**
- **Permission required** (@permission decorator)
- **Description** (from comments if available)

### Step 2.2: Generate API.md

**Template**:

```markdown
# API Documentation

Auto-generated API documentation for all exposed endpoints.

**Base URL**: `http://localhost:8080/api`

**Authentication**: Required for endpoints marked with permission

---

## Table of Contents

[Auto-generate ToC from API categories]

- [Authentication](#authentication)
- [Users](#users)
- [Data Management](#data-management)
- [Search](#search)
- [Statistics](#statistics)

---

## Authentication

[If auth endpoints detected]

### Login

**Endpoint**: `POST /api/project::login`

**Permission**: `public` (no authentication required)

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| username | String | Yes | User's username |
| password | String | Yes | User's password |

**Returns**: `User` - Authenticated user object

**Example Request**:
\`\`\`typescript
const response = await axios.post('/api/project::login', [
  'username',
  'password'
]);
\`\`\`

**Example Response**:
\`\`\`json
{
  "username": "admin",
  "email": "admin@example.com",
  "role": "admin"
}
\`\`\`

**Error Cases**:
- Invalid credentials → Throws error
- Account locked → Throws error

---

[Repeat for each endpoint]

## [Category Name]

### [Function Name]

**Endpoint**: `POST /api/project::[functionName]`

**Permission**: `[permission]`

**Description**: [Auto-generated or from comments]

**Parameters**:
[Table of parameters]

**Returns**: [Return type description]

**Example**:
[Auto-generate example based on types]

---

## Type Definitions

[Include all @volatile types used in API responses]

### User

\`\`\`gcl
@volatile
type User {
    username: String;
    email: String;
    role: String;
}
\`\`\`

[Repeat for all API types]

---

## Error Handling

All endpoints may throw errors for:
- **Authentication failures**: User not logged in or insufficient permissions
- **Validation errors**: Invalid input parameters
- **Not found**: Requested resource doesn't exist
- **Internal errors**: Server-side issues

**Error Response Format**:
\`\`\`json
{
  "error": "Error message here"
}
\`\`\`

---

## Usage Examples

### JavaScript/TypeScript

\`\`\`typescript
import axios from 'axios';

// Call API endpoint
const result = await axios.post('/api/project::functionName', [
  param1,
  param2
]);
\`\`\`

### cURL

\`\`\`bash
curl -X POST http://localhost:8080/api/project::functionName \\
  -H "Content-Type: application/json" \\
  -d '["param1", "param2"]'
\`\`\`

---

**Last Updated**: [Auto-generate timestamp]
```

### Step 2.3: Generate OpenAPI Spec (Optional)

Create `openapi.yaml` for API tools:

```yaml
openapi: 3.0.0
info:
  title: [Project Name] API
  version: 1.0.0
  description: Auto-generated API documentation

servers:
  - url: http://localhost:8080/api
    description: Local development server

paths:
  /project::functionName:
    post:
      summary: [Function description]
      security:
        - cookieAuth: []
      requestBody:
        content:
          application/json:
            schema:
              type: array
              items: [parameter types]
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                [return type schema]

[Auto-generate for all endpoints]
```

---

## Phase 3: MCP Documentation Generation

### Objective
Document Model Context Protocol (MCP) server if project uses @tag("mcp").

### Step 3.1: Detect MCP Tools

```bash
# Find functions tagged with @tag("mcp")
grep -rn '@tag("mcp")' backend/ --include="*.gcl" -A 10
```

### Step 3.2: Generate MCP.md

**Template**:

```markdown
# MCP Server Documentation

This GreyCat project exposes functions as Model Context Protocol (MCP) tools.

## What is MCP?

Model Context Protocol allows AI assistants (like Claude) to call GreyCat functions as tools during conversations.

## Available Tools

[Auto-generate from @tag("mcp") functions - extract from /// comments]

---

### [functionName]

[Auto-extract description from /// comments - 1-2 sentences]

**Parameters**:
- `param1` (Type) - [Brief description from @param]
- `param2` (Type) - [Brief description from @param]

**Returns**: [Brief description from @return]

**Example Usage**:
\`\`\`
[functionName]("example query", 10, null)
→ Returns: PaginatedResult with matching items
\`\`\`

---

[Repeat for each MCP tool - keep it SHORT]

## Enabling MCP Server

### Option 1: Built-in MCP Server

GreyCat can expose MCP tools natively:

\`\`\`bash
# Start server with MCP enabled
greycat serve --enable-mcp

# MCP server will be available at:
# stdio: greycat mcp
# http: http://localhost:8080/mcp
\`\`\`

### Option 2: Claude Desktop Integration

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

\`\`\`json
{
  "mcpServers": {
    "[project-name]": {
      "command": "greycat",
      "args": ["mcp"],
      "cwd": "/path/to/project"
    }
  }
}
\`\`\`

Restart Claude Desktop to load the MCP server.

## Testing MCP Tools

### Using MCP Inspector

\`\`\`bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
mcp-inspector greycat mcp
\`\`\`

### Using cURL

\`\`\`bash
# Call MCP tool via HTTP
curl -X POST http://localhost:8080/mcp/tools/call \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "toolName",
    "arguments": {
      "param1": "value1"
    }
  }'
\`\`\`

## Security Considerations

**Permissions**: MCP tools respect @permission decorators
- Tools with `@permission("public")` are accessible without auth
- Tools with `@permission("app.user")` require authentication
- Tools with `@permission("app.admin")` require admin role

**Best Practices**:
- Only expose safe read operations as MCP tools
- Use @permission for all sensitive operations
- Validate all inputs in tool functions
- Log MCP tool usage for audit trail

## Troubleshooting

**MCP Server Not Starting**:
\`\`\`bash
# Check if GreyCat server is running
greycat serve

# Check MCP configuration
greycat mcp --help
\`\`\`

**Tools Not Appearing in Claude**:
- Verify @tag("mcp") decorator on functions
- Restart Claude Desktop after config changes
- Check Claude Desktop logs for errors

**Permission Errors**:
- Ensure user is authenticated for protected tools
- Check @permission matches user role
- Verify SecurityService is configured correctly

---

**Last Updated**: [Auto-generate timestamp]
```

---

## Output Format

### Summary

```
===============================================================================
DOCUMENTATION GENERATED
===============================================================================

Created/Updated:
  ✓ README.md (2,450 lines)
    - Project overview
    - Quick start guide
    - Architecture documentation
    - [X] API endpoints listed
    - Development workflow
    - Troubleshooting guide

  ✓ API.md (1,850 lines)
    - [X] API endpoints documented
    - Request/response examples
    - Type definitions
    - Error handling guide
    - Usage examples (TypeScript, cURL)

  [If MCP detected]
  ✓ MCP.md (680 lines)
    - [Y] MCP tools documented
    - Setup instructions
    - Security guidelines
    - Testing guide

  [If OpenAPI generated]
  ✓ openapi.yaml
    - OpenAPI 3.0 specification
    - Ready for Postman/Swagger

Next Steps:
  1. Review generated documentation for accuracy
  2. Add project-specific details where marked with [TODO]
  3. Commit documentation:
     git add README.md API.md MCP.md
     git commit -m "Update documentation"

===============================================================================
```

---

## Execution Steps

### Step 1: Analyze Project

```bash
echo "Analyzing project structure..."
# Detect: frontend, tests, data, libraries, auth, MCP
```

### Step 2: Generate Documentation

```bash
echo "Generating README.md..."
# Extract info, generate content, write file

echo "Generating API.md..."
# Extract @expose functions, generate docs

[If MCP detected]
echo "Generating MCP.md..."
# Extract @tag("mcp") functions, generate docs
```

### Step 3: Verify

```bash
# Check files created
ls -lh README.md API.md MCP.md

# Validate markdown
# (Optional: use markdown linter if available)
```

### Step 4: Report

Present summary of generated documentation to user.

---

## Customization

### Adding Project-Specific Sections

After generation, you may want to add:

**To README.md**:
- Specific deployment instructions
- Integration guides
- Performance benchmarks
- Known limitations

**To API.md**:
- Complex workflow examples
- Integration patterns
- Rate limiting details
- Versioning strategy

**To MCP.md**:
- Custom tool workflows
- Domain-specific examples
- Advanced security configurations

### Excluding Sections

Some projects may not need all sections. The generator automatically detects and includes only relevant sections:

- No frontend → Skip frontend setup
- No tests → Skip testing section
- No MCP → Skip MCP documentation
- No auth → Skip authentication section

---

## Best Practices

### When to Regenerate

**✅ Regenerate After**:
- Adding new API endpoints
- Changing data model significantly
- Adding/removing libraries
- Updating authentication system
- Before releases

**❌ Don't Regenerate For**:
- Minor bug fixes
- Internal refactoring (no API changes)
- Documentation-only changes
- Test additions

### Maintaining Documentation

**Keep Custom Content**:
- Add custom sections AFTER auto-generated sections
- Use clear markers for custom content
- Regeneration will preserve or warn about conflicts

**Version Control**:
```bash
# Commit documentation with code changes
git add README.md API.md
git commit -m "feat: add new search endpoint

- Added semantic search API
- Updated API documentation"
```

---

## Success Criteria

✓ **README.md generated** with complete project overview
✓ **API.md generated** documenting all @expose endpoints
✓ **MCP.md generated** (if MCP tools detected)
✓ **OpenAPI spec** created (optional)
✓ **Markdown is valid** (no syntax errors)
✓ **Content is accurate** (reflects current code state)
✓ **No forbidden content** (no license, contributors, Claude mentions)

---

## Notes

- **Auto-Detection**: Generator automatically detects project features
- **Markdown Format**: All output in GitHub-flavored Markdown
- **Regeneration Safe**: Can be run multiple times, updates content
- **Customization Welcome**: Add project-specific sections after generation
- **No External Dependencies**: Uses only GreyCat code analysis

---

## Example Workflow

```bash
# 1. Complete a sprint of development
# - Added 3 new API endpoints
# - Added MCP tool for search
# - Updated data model

# 2. Run documentation generator
/docs

# 3. Review generated docs
cat README.md  # Check accuracy
cat API.md     # Verify new endpoints documented
cat MCP.md     # Check MCP tools

# 4. Customize if needed
# Add deployment section to README.md
# Add complex example to API.md

# 5. Commit
git add README.md API.md MCP.md
git commit -m "docs: update documentation for v2.0 release"

# 6. Use in development
# - New team members use README for onboarding
# - Frontend devs reference API.md
# - AI integration team uses MCP.md
```
