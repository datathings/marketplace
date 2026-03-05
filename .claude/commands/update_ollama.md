# Update Ollama Skill to Latest Version

This command updates the Ollama library to the latest version and synchronizes all skill files with the updated REST API.

**Working directory:** `plugins/ollama/`

## Repository Structure

```
plugins/ollama/
├── skills/ollama/          # Skill content (SKILL.md and references/)
├── ollama/                 # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json   # Plugin manifest
```

The `ollama/` directory is a **standalone git clone** from git@github.com:ollama/ollama.git - updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Ollama Library to Latest Tag

```bash
cd plugins/ollama/ollama

# Fetch all tags from upstream
git fetch --tags

# Check current version
CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Get latest stable tag (format: v#.#.# — skip RC/alpha/beta)
LATEST_TAG=$(git tag --sort=-version:refname | grep -v 'rc\|alpha\|beta' | head -n 1)
echo "Latest tag: $LATEST_TAG"

# Preview changes between versions
if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50

  # Look for API-related changes
  echo -e "\nAPI-related commits:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" --grep="api\|endpoint\|route\|handler" -i
fi

# Checkout latest stable tag
git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
```

- Latest tags viewable at: https://github.com/ollama/ollama/tags
- Stable release tags use format: `v#.#.#` (e.g., `v0.16.3`, `v0.17.0`)

### 2–6. Analyze & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large source content out of the main context.

Provide the subagent with:
- Path to API docs: `plugins/ollama/ollama/docs/` (all `.md` files)
- Path to Go API types: `plugins/ollama/ollama/api/types.go` (request/response structs)
- Path to skill references: `plugins/ollama/skills/ollama/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan `docs/` and `api/types.go` for:
- New endpoints added since last update
- Deprecated or removed endpoints
- Modified request/response fields (new optional fields, removed fields, type changes)
- New model parameters or Modelfile instructions

**Update Skill Documentation Files** in `skills/ollama/references/`:
- **`api-generation.md`** — `POST /api/generate`, `POST /api/chat` (streaming, tool calling, structured output, image input, thinking models)
- **`api-models.md`** — `GET /api/tags`, `POST /api/show`, `POST /api/pull`, `POST /api/push`, `POST /api/copy`, `DELETE /api/delete`, `GET /api/ps`, `POST /api/create`, blob endpoints, `GET /api/version`
- **`api-embeddings.md`** — `POST /api/embed`, `POST /api/embeddings` (deprecated legacy endpoint)
- **`modelfile.md`** — Modelfile format: FROM, PARAMETER, TEMPLATE, SYSTEM, ADAPTER, LICENSE, MESSAGE instructions and all valid parameters
- **`workflows.md`** — Working examples: quick start, multi-turn chat, tool calling, structured output, RAG/embeddings, custom model creation

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — it is always loaded into context, so every line
  must earn its place; API details belong in `references/` files, not here
- Update the Ollama version/tag in the documentation
- Add notes about newly supported features or model capabilities
- Update the API reference table if files were added or removed

**Remove Deprecated Content:**
- Search documentation for endpoints or fields removed from `docs/` and `api/types.go`
- Remove or replace deprecated references (e.g., the legacy `/api/embeddings` if removed)
- Note breaking changes in SKILL.md

**Validate Changes:**
- Cross-reference all documented endpoints against `docs/api.md` (or equivalent)
- Ensure request/response field names and types match `api/types.go` exactly
- Verify all new public API endpoints are documented

**The subagent should return a summary** of: endpoints added, endpoints removed, fields
changed, and any breaking changes — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
# From repo root
./package.sh ollama
```

This creates `skills/ollama.skill`.

## Key Considerations

- **Source of truth**: Always derive endpoint specs from `docs/api.md` and `api/types.go`, not secondary docs
- **Deprecation handling**: Remove deprecated endpoints/fields completely; don't just mark them
- **New features**: Major new features (new endpoint, structured output, tool calling changes) should get workflow examples
- **Streaming**: Document both streaming and non-streaming variants of generation endpoints
- **Backward compatibility**: Note any breaking changes in SKILL.md

## Success Criteria

- ✅ Ollama updated to latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API endpoints from docs are documented
- ✅ All removed endpoints/fields are deleted from documentation
- ✅ All request/response fields match `api/types.go` exactly
- ✅ Code examples in workflows.md are valid
- ✅ Package builds successfully with `./package.sh ollama`
