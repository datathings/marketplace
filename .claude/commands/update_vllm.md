# Update vLLM Skill to Latest Version

This command updates the vLLM library to the latest version and synchronizes all skill files with the updated API.

**Working directory:** `plugins/vllm/`

## Repository Structure

```
plugins/vllm/
├── skills/vllm/          # Skill content (SKILL.md and references/)
├── vllm/                 # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json   # Plugin manifest
```

The `vllm/` directory is a **standalone git clone** from git@github.com:vllm-project/vllm.git - updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

```bash
cd plugins/vllm/vllm

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Stable releases use v#.#.# format (exclude RC tags)
LATEST_TAG=$(git tag --sort=-version:refname | grep "^v" | grep -v "rc\|alpha\|beta" | head -n 1)
echo "Latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50

  echo -e "\nAPI-related commits:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" --grep="API\|deprecat\|LLM\|SamplingParams\|LoRA\|multimodal" -i
fi

git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
```

- Latest tags: https://github.com/vllm-project/vllm/tags
- Stable release tags use format: `v#.#.#` (e.g., `v0.16.0`, `v0.15.1`)

### 2–6. Analyze API Changes & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for analysis and documentation rewrite. This keeps the large source content out of the main context.

Provide the subagent with:
- Path to source-of-truth files:
  - `plugins/vllm/vllm/vllm/__init__.py` — public exports
  - `plugins/vllm/vllm/vllm/entrypoints/llm.py` — `LLM` class
  - `plugins/vllm/vllm/vllm/sampling_params.py` — `SamplingParams`, `StructuredOutputsParams`
  - `plugins/vllm/vllm/vllm/outputs.py` — output types
  - `plugins/vllm/vllm/vllm/lora/request.py` — `LoRARequest`
  - `plugins/vllm/vllm/vllm/entrypoints/openai/` — server entrypoints and REST API
- Path to skill references: `plugins/vllm/skills/vllm/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan source-of-truth files for:
- New classes, methods, or parameters added since last update
- Deprecated or removed APIs (check `CHANGELOG.md` or release notes)
- Modified method signatures (parameter changes, new defaults, return type changes)
- New types, enums, or configuration options

**Update Skill Documentation Files** in `skills/vllm/references/`:
- **`api-llm.md`** — `LLM` class constructor and all methods (`generate()`, `chat()`, `embed()`, `classify()`, `score()`, `beam_search()`)
- **`api-sampling.md`** — `SamplingParams`, `StructuredOutputsParams`, `BeamSearchParams`, all output types
- **`api-server.md`** — `vllm serve` CLI flags, REST endpoints, OpenAI Python client usage, auth, streaming
- **`api-lora.md`** — `LoRARequest`, offline LoRA, multi-LoRA, server-side LoRA
- **`api-multimodal.md`** — image/audio/video inputs, multi-image, per-model prompt formats
- **`workflows.md`** — complete working examples for all major use cases

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — always loaded into context; API details belong in `references/`
- Update the vLLM version/tag
- Add notes about newly supported features
- Update the API reference table if files were added or removed

**Remove Deprecated Content:**
- Search documentation for APIs removed or deprecated in the new version
- Remove deprecated references completely; note breaking changes in SKILL.md

**Validate Changes:**
- Cross-reference all documented methods against `llm.py` and `sampling_params.py`
- Ensure method signatures match exactly (parameter names, types, defaults)
- Verify all new public API surface is documented

**The subagent should return a summary** of: APIs added, removed, signatures changed, and any breaking changes.

### 7. Package Updated Skill

```bash
./package.sh vllm
```

This creates `skills/vllm.skill`.

## Key Considerations

- **Source of truth**: Always derive signatures from `llm.py` and `sampling_params.py`, not secondary docs or docstrings
- **`generate()` vs `chat()`**: A key gotcha — `generate()` does not apply chat templates; `chat()` does
- **`StructuredOutputsParams`**: Separate import from `SamplingParams`; passed as `SamplingParams(structured_outputs=...)`
- **`generation_config`**: vLLM reads `generation_config.json` from HuggingFace by default; pass `generation_config="vllm"` to disable
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them
- **New features**: Major new features should get workflow examples in `workflows.md`

## Success Criteria

- ✅ Library updated to latest stable tag (non-RC)
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API methods from source-of-truth files are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match source-of-truth exactly
- ✅ Code examples in workflows.md are valid
- ✅ Package builds successfully with `./package.sh vllm`
