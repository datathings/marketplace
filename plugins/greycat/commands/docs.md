---
name: docs
description: Update README, API documentation, and MCP server documentation
allowed-tools: Bash, Read, Grep, Glob, Write, Edit
---

# Documentation Generator

**Purpose**: Generate/update README.md, API.md, and MCP.md.

**Run After**: Each sprint, before releases, when adding features/APIs.

---

## Phase 1: README.md

### Detect features
\`\`\`bash
# Frontend: app/ directory or @greycat/web in package.json
# Tests:    test/ directory
# Libs:     @library in project.gcl
# Auth:     @permission / @role in project.gcl
# MCP:      @tag("mcp") in src/
# Data:     data/ directory, *.gguf files
\`\`\`

### Extract info
\`\`\`bash
grep "@library" project.gcl                              # versions
grep "@permission\|@role" project.gcl                    # auth
grep -rn "^type [A-Z]" src/ --include="*.gcl"            # models
grep -rn "^abstract type.*Service" src/ --include="*.gcl"  # services
grep -rn "@expose" src/ --include="*_api.gcl"            # endpoints
\`\`\`

### Sections to generate

1. **Title + 1-line description**
2. **Overview** — tech stack (GreyCat version, frontend if any), key features (counts: N types, M endpoints, auth?, search?, MCP?)
3. **Quick Start** — prerequisites, `git clone`, `greycat install`, `pnpm install` (if frontend), `greycat serve`, `pnpm dev`, `greycat run import`
4. **Architecture** — data model (auto-extracted types), service layer (list services), API endpoints table (`| Endpoint | Permission | Description |`)
5. **Development** — project structure (see CLAUDE.md template), common commands (`greycat-lang lint/fmt`, `greycat build/test/serve/run/codegen`, `pnpm dev/build/lint/test`), workflow (lint after each change, regen ts after backend types)
6. **Testing** — `greycat test`, current coverage stats
7. **Configuration** — `.env` vars, library versions
8. **Authentication** (if detected) — roles, permissions, `SecurityService` usage
9. **API Documentation** — link to API.md
10. **MCP Server** (if detected) — link to MCP.md
11. **Database** — dev reset (`rm -rf gcdata && greycat run import`), backup (`tar -czf gcdata-backup.tar.gz gcdata/`)
12. **Troubleshooting** — lint errors (missing imports, type mismatch), server won't start (port, gcdata integrity), frontend API errors (`greycat codegen ts`, vite proxy)
13. **Project Statistics** — auto-generated counts

### Content rules
- ✅ Include: technical details, setup, API refs, workflow, testing, config, troubleshooting, stats
- ❌ Exclude: contributor guidelines, license, author credits, AI/Claude mentions, copyright, acknowledgments

---

## Phase 2: API.md

### Extract endpoints
\`\`\`bash
grep -rn "@expose" src/ --include="*_api.gcl" -A 20
\`\`\`

For each: function name, parameters (with types), return type, `@permission`, description from `///` comments.

### Structure

```markdown
# API Documentation
Base URL: `http://localhost:8080/api`

## Table of Contents
[Auto-generate per category]

## [Category]
### [Function Name]
**Endpoint**: `POST /api/<module>::<fn>`
**Permission**: `<level>`
**Description**: ...
**Parameters**: | Name | Type | Required | Description |
**Returns**: ...
**Example**:
\`\`\`typescript
await axios.post('/api/<module>::<fn>', [param1, param2]);
\`\`\`
**Error cases**: ...
```

### Type definitions section
Include all `@volatile` types used in API responses with field tables.

### Error handling
Typed-error envelope (recommended):
```json
{ "code": "NOT_FOUND", "message": "...", "id": "abc-123" }
```
Clients branch on `code`, not `message`.

### Routing conventions
- Use explicit `/<module>::<fn>` from clients — bare alias is fragile, disappears on name collision.
- `@expose static fn` on abstract types: ONLY `/<module>::T::fn`.
- `/runtime::` is a module name, not a routing prefix.

### Long-running / async endpoints
20s TTL default. For long endpoints, invoke as task:
\`\`\`bash
curl -H "task:''" -X POST -d '[]' http://localhost:8080/<module>::<fn>
curl -X POST -d '[42, "task-uuid"]' http://localhost:8080/runtime::Task::info  # poll
GET /files/<user_id>/tasks/<task_id>/result.gcb?json                            # result
\`\`\`
States: `empty → waiting → running → await → ended | error | cancelled | ended_with_errors`.

---

## Phase 3: MCP.md (if `@tag("mcp")` detected)

\`\`\`bash
grep -rn '@tag("mcp")' src/ --include="*.gcl" -A 10
\`\`\`

### Structure

```markdown
# MCP Server Documentation

## What is MCP?
Model Context Protocol — AI assistants call GreyCat functions as tools.

## Available Tools
[For each: name, 1-2 sentence description from ///, parameters with types, returns, example]

## Enabling
- Built-in: `greycat serve --enable-mcp` (stdio: `greycat mcp`, http: `/mcp`)
- Claude Desktop: add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
  \`\`\`json
  { "mcpServers": { "<project>": { "command": "greycat", "args": ["mcp"], "cwd": "/path" } } }
  \`\`\`

## Testing
- `npm install -g @modelcontextprotocol/inspector` → `mcp-inspector greycat mcp`
- HTTP: `curl -X POST .../mcp/tools/call -d '{"name":"tool","arguments":{...}}'`

## Security
- MCP tools respect `@permission` decorators
- Only expose safe read operations as tools
- Log MCP usage for audit
```

---

## Phase 4: OpenAPI Spec (optional)

Generate `openapi.yaml` (OpenAPI 3.0):
\`\`\`yaml
openapi: 3.0.0
info: { title: <name>, version: <ver> }
servers: [{ url: http://localhost:8080/api }]
paths:
  /<module>::<fn>:
    post:
      security: [{ cookieAuth: [] }]
      requestBody: { content: { application/json: { schema: { type: array, items: ... } } } }
      responses: { '200': { ... } }
\`\`\`

---

## Execution

1. Analyze: detect frontend/tests/data/libs/auth/MCP
2. Generate README.md, API.md, MCP.md (conditional) via Write
3. Validate: `ls -lh README.md API.md MCP.md`
4. Report summary

---

## Notes

- Regenerate after: new endpoints, model changes, lib upgrades, auth changes
- Don't regenerate for: bug fixes, internal refactors, doc-only changes
- Custom content goes AFTER auto-generated sections
- Markdown only (no HTML)
