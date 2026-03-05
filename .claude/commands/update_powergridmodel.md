# Update power-grid-model Skill to Latest Version

This command updates the power-grid-model library to the latest version and synchronizes
all skill files with the updated API.

**Working directory:** `plugins/powergridmodel/`

## Repository Structure

```
plugins/powergridmodel/
├── skills/powergridmodel/          # Skill content (SKILL.md and references/)
├── power-grid-model/               # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json      # Plugin manifest
```

The `power-grid-model/` directory is a **standalone git clone** from
https://github.com/PowerGridModel/power-grid-model — updated independently via
`git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

```bash
cd plugins/powergridmodel/power-grid-model

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Tags follow v#.#.# format (e.g., v1.13.10)
LATEST_TAG=$(git tag --sort=-version:refname | grep "^v" | head -n 1)
echo "Latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50
fi

git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
```

### 2. Analyze API Changes (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large source content out of the main context.

Provide the subagent with:
- Path to Python package: `plugins/powergridmodel/power-grid-model/src/power_grid_model/`
- Path to skill references: `plugins/powergridmodel/skills/powergridmodel/references/`
- Old tag and new tag (from Step 1 output)

The source of truth for the Python public API is:
- `src/power_grid_model/__init__.py` — top-level exports
- `src/power_grid_model/core/power_grid_model.py` — `PowerGridModel` class
- `src/power_grid_model/validation/` — validation functions
- `src/power_grid_model/enum.py` — all enumerations
- `src/power_grid_model/component_attributes.py` or equivalent — component definitions
- Type stubs (`.pyi`) if present

Look for:
- New component types added
- New calculation methods or options
- Modified function signatures (parameter changes, return types)
- New enumerations or constants
- New optimizer or batch features
- Deprecated or removed functions/attributes

### 3. Update Skill Documentation Files

Update the following files in `skills/powergridmodel/references/`:

- **`api-core.md`** — `PowerGridModel` class, `calculate_power_flow`,
  `calculate_short_circuit`, `calculate_state_estimation`, `update`, `get_indexer`,
  `initialize_array`, serialization utilities
- **`api-components.md`** — All component types with attribute tables, all enumerations
  (`ComponentType`, `LoadGenType`, `WindingType`, `FaultType`, `FaultPhase`,
  `CalculationMethod`, `TapChangingStrategy`, etc.)
- **`api-validation.md`** — `validate_input_data`, `validate_batch_data`,
  `assert_valid_input_data`, `assert_valid_batch_data`, `ValidationException`
- **`api-batch.md`** — Dense/sparse batch formats, independent vs dependent batches,
  Cartesian product, parallel threading, batch error handling
- **`workflows.md`** — Complete working examples (power flow, batch, state estimation,
  short circuit, validation)

### 4. Update Main SKILL.md

- Update the library version/tag in the documentation
- Update component count if new component types were added
- Add notes about newly supported features
- Update the API reference table if files were added/removed
- Keep SKILL.md **concise and dense** — under 500 lines

### 5. Remove Deprecated Content

- Search documentation files for functions removed from the source
- Remove or replace deprecated references
- Note breaking changes prominently

### 6. Validate Changes

- Cross-reference all documented functions/classes against the Python source
- Ensure signatures match exactly
- Verify all new public API members are documented
- Check that numpy dtype definitions for component arrays are accurate

### 7. Package Updated Skill

```bash
./package.sh powergridmodel
```

This creates `skills/powergridmodel.skill`.

## Key Considerations

- **Source of truth**: Always derive signatures from Python source files, not docs
- **Component dtypes**: The numpy structured array field names and types must be exact
- **Enum values**: All enum values must be listed with their correct integer codes
- **Batch formats**: Dense and sparse batch formats have distinct semantics — document both
- **Deprecation handling**: Remove deprecated items completely; don't just mark them
- **New features**: Major new features should get workflow examples
- **Backward compatibility**: Note any breaking changes in SKILL.md

## Success Criteria

- ✅ Library updated to latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API functions/classes from Python source are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match source exactly
- ✅ Code examples in workflows.md are valid
- ✅ Package builds successfully with `./package.sh powergridmodel`
