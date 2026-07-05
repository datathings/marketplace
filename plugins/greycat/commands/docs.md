---
name: docs
description: Update README, API documentation, and MCP server documentation
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Task
---

# Documentation Generator

**Purpose**: Generate/update README.md, API.md, MCP.md (+ optional OpenAPI, frontend SEO artifacts).
**Run after**: each sprint, before releases, when adding features/APIs.
**Regenerate after**: new endpoints, model changes, lib upgrades, auth changes, new routes (refresh `sitemap.xml`/`llms.txt`). **Not** for bug fixes, internal refactors, or doc-only changes.

**ultrathink + ultracode** — read the actual code, never emit boilerplate. README.md (feature scan), API.md (`@expose` scan), MCP.md (`@tag("mcp")` scan), and the Phase-5 SEO/`llms.txt` artifacts are INDEPENDENT deliverables: when parallel subagents / ultracode are available, draft them as PARALLEL subagents (one each), then assemble & write. Otherwise generate sequentially.

**Frontend stack** (referred to below as *the frontend stack*): VitePlus (`vp`) + Lit (light DOM) + TypeScript + Shoelace + `@greycat/web` + lucide-static, pnpm; note i18next / maplibre-gl / Vitest (optional) if present.

**Lighthouse** (referred to below): audit both mobile (default) and desktop (`--preset=desktop`), target ≥ 90 per category — via the project's lighthouse script (`:desktop` / `:ci` variants) if present, else the `lighthouse` CLI against the served app (`greycat serve`/`dev` first). Optional tooling.

---

## Phase 1: README.md

### Detect features
```bash
# Frontend: frontend/ directory, or lit / @shoelace-style/shoelace / @greycat/web in package.json
# Tests:    test/ directory
# Libs:     @library in project.gcl
# Auth:     @permission / @role in project.gcl
# MCP:      @tag("mcp") in src/
# Data:     data/ directory, *.gguf files
# SEO:      webroot/{robots.txt,sitemap.xml,llms.txt,*.webmanifest}
```

### Extract info
```bash
grep "@library" project.gcl                              # versions
grep "@permission\|@role" project.gcl                    # auth
grep -rn "^type [A-Z]" src/ --include="*.gcl"            # models
grep -rnE "^abstract type.*Service|^(static )?fn [a-z_]" src/ --include="*.gcl"  # services (abstract-type OR free fns)
grep -rn "@expose" src/ --include="*.gcl"                # endpoints (@expose may live in src/api.gcl, not only *_api.gcl)
```

### Sections to generate

1. **Title + 1-line description**
2. **Overview** — GreyCat version; the frontend stack if any; key feature counts (N types, M endpoints, auth?, search?, MCP?)
3. **Quick Start** — prerequisites, `git clone`, `greycat install`, `pnpm install` (if frontend), `greycat serve` (or `greycat dev` to also run the frontend watcher), `greycat run import`. If frontend: run Lighthouse (above) to audit performance/SEO.
4. **Architecture** — data model (auto-extracted types), service layer (list services), API endpoints table (`| Endpoint | Permission | Description |`)
5. **Development** — project structure (see CLAUDE.md template); common commands: `greycat-lang lint/fmt`, `greycat build/test/serve/run/codegen`, `greycat dev`, `greycat codegen ts`, Lighthouse (above); workflow (lint after each change, `greycat codegen ts` after backend type changes)
6. **Testing** — `greycat test`, current coverage stats
7. **Configuration** — `.env` vars, library versions
8. **Authentication** (if detected) — roles, permissions, `SecurityService` usage
9. **API Documentation** — link to API.md
10. **MCP Server** (if detected) — link to MCP.md
11. **Database** — dev reset (`rm -rf gcdata && greycat run import`); backup/restore via `greycat backup` / `greycat restore <archive>` (prefer over raw `tar` on a live `gcdata/`)
12. **Troubleshooting** — lint errors (missing imports, type mismatch); server won't start (port, gcdata integrity); frontend API errors (regenerate client with `greycat codegen ts`; a stale ABI returns HTTP 422). `greycat dev` serves API + assets on one origin — no proxy to configure.
13. **Project Statistics** — auto-generated counts

### Content rules
- ✅ Include: technical details, setup, API refs, workflow, testing, config, troubleshooting, stats
- ❌ Exclude: contributor guidelines, license, author credits, AI/Claude mentions, copyright, acknowledgments

---

## Phase 2: API.md

### Extract endpoints
```bash
grep -rn "@expose" src/ --include="*.gcl" -A 20        # @expose may live in src/api.gcl, not only *_api.gcl
```
For each: function name, parameters (with types), return type, `@permission`, description from `///` comments.

### Structure

```markdown
# API Documentation
Base URL: `http://localhost:8080`   (no `/api` prefix — GreyCat serves RPC at the root)

## Table of Contents
[Auto-generate per category]

## [Category]
### [Function Name]
**Endpoint**: `POST /<module>::<fn>`   (body: JSON array of positional args)
**Permission**: `<level>`
**Description**: ...
**Parameters**: | Name | Type | Required | Description |
**Returns**: ...
**Example**:
\`\`\`typescript
await axios.post('/<module>::<fn>', [param1, param2]);
\`\`\`
**Error cases**: ...
```

### Type definitions section
List the `@volatile` `…View` types returned by the API with field tables.

### Error handling
Typed-error envelope (recommended); clients branch on `code`, not `message`:
```json
{ "code": "NOT_FOUND", "message": "...", "id": "abc-123" }
```

### Routing conventions
- Path-RPC: `POST /<module>::<fn>` (module = source file basename), body = JSON array of positional args.
- `@expose static fn` on a type: three segments — `POST /<module>::<Type>::<fn>` (e.g. `/runtime::Identity::current_id`).
- A custom `@expose("path")` exposes at that exact arbitrary path — document whatever path the code declares.
- JSON-RPC alternative: `POST /` with `{ "method": "<module>.<fn>", "params": [...] }` (dots, not `::`).
- `/runtime::` is a module name, not a routing prefix.

### Long-running / async endpoints
20s request TTL (the `--request_ttl` flag) by default. To run a call as a background task, set the `task: true` request header — the server returns the `task_id` immediately:
```bash
curl -H "task: true" -X POST -d '[]' http://localhost:8080/<module>::<fn>   # returns task_id
# poll status from GCL/RPC via Task::is_running(task_id) / Task::running() / Task::history(offset, max)
GET /files/<user_id>/tasks/<task_id>/result.gcb?json                        # fetch result once ended
```
States: `empty → waiting → running → await → ended | error | cancelled | ended_with_errors`.

---

## Phase 3: MCP.md (if `@tag("mcp")` detected)

```bash
grep -rn '@tag("mcp")' src/ --include="*.gcl" -A 10
```

### Structure

```markdown
# MCP Server Documentation

## What is MCP?
Model Context Protocol — AI assistants call GreyCat functions as tools.

## Available Tools
[For each: name, 1-2 sentence description from ///, parameters with types, returns, example]
Each tool is a function tagged `@tag("mcp")`. `/// @param <name> <desc>` doc-comment lines surface as the tool's argument schema.

## Enabling
- A served project (`greycat serve` / `greycat dev`) exposes MCP on the same HTTP port — no separate flag or subcommand. `tools/list` returns every `@tag("mcp")` function; `tools/call` invokes one with named arguments matching the function's parameter names.
- To add a tool, tag the function: `@tag("mcp")` (stack with `@tag("openapi")` to also include it in the OpenAPI spec).

## Testing
- Call the served MCP endpoint directly with `tools/list` / `tools/call`, or point an MCP client at the running server's URL.
- Named arguments match the function's parameter names; `@permission` still gates each call.

## Security
- MCP tools respect `@permission` decorators (a call is rejected if the caller lacks the permission)
- Only tag safe read operations with `@tag("mcp")`
- Log MCP usage for audit
```

---

## Phase 4: OpenAPI Spec (optional)

Generate `openapi.yaml` (OpenAPI 3.0):
```yaml
openapi: 3.0.0
info: { title: <name>, version: <ver> }
servers: [{ url: http://localhost:8080 }]
paths:
  /<module>::<fn>:
    post:
      security: [{ cookieAuth: [] }]
      requestBody: { content: { application/json: { schema: { type: array, items: ... } } } }
      responses: { '200': { ... } }
```

---

## Phase 5: Frontend SEO & LLM-friendly discoverability (if frontend detected)

Generate/refresh machine-readable artifacts so search engines **and LLM agents** can navigate the app. Source generated into `frontend/public/` (or written directly to `webroot/`).

### 5.1 `llms.txt` (always)
A concise Markdown map for LLM agents: purpose, key routes, and public `@expose` endpoints (from Phase 2). Optionally also emit `llms-full.txt` with expanded endpoint docs.
```markdown
# <Project Name>
> <one-line description>

## Pages
- [Overview](/): KPIs and summary
- [Detail](/detail): per-entity drill-down

## API (POST JSON body = positional args)
- `/<module>::<fn>` — <description> (permission: <level>)
```

### 5.2 SEO head (`index.html`)
Ensure/insert: `<html lang>`, unique `<title>`, `<meta name="description">`, canonical `<link>`, Open Graph + Twitter Card, `theme-color`, `<meta name="color-scheme">`, and JSON-LD (`schema.org`, e.g. `WebApplication` / `SoftwareApplication`).

### 5.3 `robots.txt` + `sitemap.xml` + web manifest
```
# robots.txt
User-agent: *
Allow: /
Sitemap: https://<host>/sitemap.xml
```
Emit a `sitemap.xml` listing crawlable routes and a `site.webmanifest` (name, icons, theme/background color).

### 5.4 Lighthouse audit doc
Record how to audit (Lighthouse, above). Target ≥ 90 in performance/SEO/accessibility/best-practices; note current scores in the README "Performance" subsection.

---

## Execution

1. Analyze: detect frontend / tests / data / libs / auth / MCP / SEO.
2. Generate README.md, API.md, MCP.md (conditional) via Write — in parallel subagents when available (see top), else sequentially.
3. If frontend: generate/refresh `llms.txt`, SEO head, `robots.txt`, `sitemap.xml`, manifest (Phase 5).
4. Validate: `ls -lh README.md API.md MCP.md` + `ls webroot/{robots.txt,sitemap.xml,llms.txt}`.
5. Report summary.

**Notes**: Custom content goes AFTER auto-generated sections. README/API/MCP are Markdown only (no HTML); the SEO head in `index.html` and the `*.xml`/`*.txt`/manifest artifacts are the Phase 5 exception.
