# Update pandapower Skill to Latest Version

This command updates the pandapower library to the latest version and synchronizes all skill files with the updated API.

**Working directory:** `plugins/pandapower/`

## Repository Structure

```
plugins/pandapower/
├── skills/pandapower/          # Skill content (SKILL.md and references/)
├── pandapower-repo/            # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json  # Plugin manifest
```

The `pandapower-repo/` directory is a **standalone git clone** from https://github.com/e2nIEE/pandapower - updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

```bash
cd plugins/pandapower/pandapower-repo

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Tags follow v#.#.# format (e.g., v3.4.0, v3.3.0)
LATEST_TAG=$(git tag --sort=-version:refname | grep "^v" | head -n 1)
echo "Latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50

  echo -e "\nAPI-related commits:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" --grep="API\|deprecat\|create_\|run\|calc_" -i
fi

git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
```

- Latest tags: https://github.com/e2nIEE/pandapower/tags
- Stable release tags use format: `v#.#.#` (e.g., `v3.4.0`)

### 2–6. Analyze & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large source code out of the main context.

Provide the subagent with:
- Path to main package: `plugins/pandapower/pandapower-repo/pandapower/__init__.py`
- Key source files:
  - `pandapower/create.py` (element creation functions)
  - `pandapower/run.py` or `pandapower/powerflow.py` (power flow runners)
  - `pandapower/shortcircuit/` (short circuit functions)
  - `pandapower/estimation/` (state estimation)
  - `pandapower/topology/` (graph topology)
  - `pandapower/plotting/` (visualization)
  - `pandapower/toolbox.py` and `pandapower/file_io.py` (utilities and I/O)
  - `pandapower/timeseries/` (time series simulation)
- Path to skill references: `plugins/pandapower/skills/pandapower/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan `__init__.py` exports and key source files for:
- New functions added to the public API
- Deprecated or removed functions
- Modified signatures (parameter changes, new kwargs, return type changes)
- New element types supported in `create_*` functions
- New solver backends or options in `runpp`/`runopp`/`calc_sc`

**Update Skill Documentation Files** in `skills/pandapower/references/`:
- **`api-network.md`** — Network creation (`create_empty_network`, buses, lines, transformers, loads, generators, switches, etc.) and predefined networks
- **`api-powerflow.md`** — `runpp`, `rundcpp`, `runpp_pgm`, `runopp`, `rundcopp`, `runpp_3ph`, `calc_sc`, `estimate`, cost functions, result tables
- **`api-topology.md`** — Graph topology functions (`create_nxgraph`, connected components, feeders, island detection, radiality)
- **`api-plotting.md`** — `simple_plot`, collection creators, `draw_collections`, plotly functions, geodata handling
- **`api-toolbox.md`** — Utility functions, file I/O (JSON/pickle/Excel/SQLite/MATPOWER/CIM), merge/select/drop, `diagnostic`, groups, time series
- **`workflows.md`** — Complete working examples (quick start, distribution analysis, OPF, short circuit, state estimation, time series, topology)

**Update Main SKILL.md:**
- Keep SKILL.md **concise and dense** — always loaded into context; API details belong in `references/`
- Update the version in the Overview section (`**Version:** v#.#.#`)
- Add notes about newly supported features or element types
- Update the API Reference table if files were added/removed

**Remove Deprecated Content:**
- Search all documentation files for functions removed from `__init__.py` exports
- Remove or replace deprecated function references
- Note breaking changes in SKILL.md

**Validate Changes:**
- Cross-reference all documented functions against `pandapower/__init__.py`
- Ensure function signatures match the actual source code
- Verify all new public API functions are documented

**The subagent should return a summary** of: functions added, functions removed, signatures
changed, and any breaking changes — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
# From repo root
./package.sh pandapower
```

This creates `skills/pandapower.skill`.

## Key Considerations

- **Source of truth**: Always derive function signatures from `pandapower/__init__.py` (public exports) and the key source modules
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them
- **New element types**: Check `create.py` for new `create_*` functions (new network elements)
- **Solver changes**: Note any new or removed solver backends in `runpp`/`runopp` kwargs
- **Result tables**: Document new result DataFrame columns added to power flow output

## Success Criteria

- ✅ Library updated to the latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API functions from `__init__.py` are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match the source code exactly
- ✅ Code examples in `workflows.md` are valid
- ✅ Package builds successfully with `./package.sh pandapower`
