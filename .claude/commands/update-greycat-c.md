# Update GreyCat C Skill

Update the GreyCat C SDK skill documentation to match newly provided header files.

**Working directory:** `plugins/greycat-c/`

> **Handoff protocol:** The user manually copies new SDK headers and tells Claude where they
> are. **Ask for the header path and new version before doing anything else** if not already
> provided.

## Required Inputs (ask if not given)

1. **Path to new headers** — the directory or file(s) the user copied in (e.g.,
   `plugins/greycat-c/sdk-headers/`)
2. **New SDK version** — the version string to record in SKILL.md (e.g., `7.6.1`)

## Repository Structure

```
plugins/greycat-c/
├── skills/greycat-c/
│   ├── SKILL.md                        # Overview, quick reference
│   └── references/
│       ├── api_reference.md            # C API functions, types, macros
│       └── standard_library.md         # GCL standard library modules
└── .claude-plugin/plugin.json          # Plugin manifest (version field)
```

## Process

### 1. Record Old Version

Read `skills/greycat-c/SKILL.md` and note the currently documented version — this is the
baseline for the diff summary returned to the user.

### 2. Analyze & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the header analysis and
documentation rewrite. This keeps the large header content out of the main context.

Provide the subagent with:
- Path to new header file(s) (from the user)
- Path to existing skill docs: `plugins/greycat-c/skills/greycat-c/`
- Old version and new version strings

The subagent should:

**Analyze the new headers** — extract and categorize all public API elements:
- Function prototypes: name, return type, parameters (types + names)
- `typedef`s and `struct` definitions with all fields
- `enum` types and their values
- Preprocessor macros (`#define`) that are part of the public API
- Any deprecation markers (`__attribute__((deprecated))`, comments, or naming conventions)

**Diff against existing documentation:**
- New symbols not yet documented
- Removed symbols still present in docs
- Signature changes (parameter type/name/order, return type)
- New types, enums, or macros

**Update `references/api_reference.md`:**
- Add entries for new functions/types/macros
- Update changed signatures exactly as they appear in the header
- Remove entries for symbols no longer in the header
- Keep logical grouping (e.g., graph ops, time series, memory, I/O) — add new groups
  if the SDK gained a new domain

**Update `references/standard_library.md`:**
- Sync any changes to GCL standard library modules exposed via the C API
- Add new modules, remove obsolete ones

**Update `SKILL.md`:**
- Keep SKILL.md **concise and dense** — it is always loaded into context, so every line
  must earn its place; full API details belong in `references/` files, not here
- Update the SDK version field
- Update function/type counts in the overview if they changed
- Note any breaking changes or major new capabilities in "Key Considerations"

**The subagent should return a summary** of: symbols added, symbols removed, signatures
changed, and any breaking changes — for the main context to report to the user.

### 3. Update plugin.json Version

Update the version field in `.claude-plugin/plugin.json` to match the new SDK version:

```json
{
  "version": "<new-version>"
}
```

### 4. Package Updated Skill

```bash
# From repo root
./package.sh greycat-c
```

This creates `skills/greycat-c.skill`. Fix any validation errors reported.

## Key Considerations

- **Source of truth is the header** — never paraphrase; copy signatures exactly
- **All public symbols must be documented** — if it's in the header and not `_private`
  or implementation-internal, it belongs in the docs
- **Structs and typedefs matter** — users need field names and types, not just the
  struct name
- **Macros are API** — document `#define` constants and function-like macros that users
  are expected to call
- **No deprecated content** — remove removed symbols completely; don't just mark them

## Success Criteria

- [ ] New SDK version recorded in `SKILL.md` and `plugin.json`
- [ ] All new symbols from the header are documented
- [ ] All removed symbols are deleted from the docs
- [ ] All signatures match the header exactly (types, parameter names, order)
- [ ] `struct` fields and `enum` values are complete
- [ ] `./package.sh greycat-c` runs successfully
