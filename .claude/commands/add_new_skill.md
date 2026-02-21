# Add New Skill to Marketplace

This command adds a new skill plugin to the marketplace from a git repository.

**Start by asking the user for the two required inputs below, then execute all steps.**

## Required Inputs

Ask the user for both before starting:

1. **Git repository URL** - the upstream library/project repo (e.g., `https://github.com/org/my-lib`)
2. **Skill name** - hyphen-case identifier matching the directory name (e.g., `my-lib`)

Optionally ask for a specific tag/version to checkout (default: latest tag, or main/master if no tags exist).

---

## Plugin Structure to Create

```
plugins/<skill-name>/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── skills/<skill-name>/
│   ├── SKILL.md                 # Main skill entry point
│   └── references/              # Detailed docs loaded on-demand
│       ├── api-*.md             # API sections (split by domain)
│       └── workflows.md         # Working examples
├── <repo-dir>/                  # Standalone git clone (gitignored)
│   └── .git/
└── .gitignore                   # Ignores *.skill and <repo-dir>/
```

---

## Process

> **Subagent strategy**: Steps 4 and 5 (repository analysis + skill content generation) are
> context-heavy and **must** be dispatched to subagents using the `Task` tool. Run steps 1–3
> in the main context, then hand off to subagents for the exploration and writing work, then
> return to main context for steps 6–11.

### 1. Create Plugin Directory Structure

```bash
SKILL_NAME="<skill-name>"
PLUGIN_DIR="plugins/$SKILL_NAME"

mkdir -p "$PLUGIN_DIR/.claude-plugin"
mkdir -p "$PLUGIN_DIR/skills/$SKILL_NAME/references"
```

### 2. Clone the Git Repository

```bash
cd "plugins/$SKILL_NAME"

# Clone the upstream repo
git clone <repo-url> <repo-dir>

cd <repo-dir>

# Checkout latest stable tag if available
LATEST_TAG=$(git tag --sort=-version:refname | head -n 1)
if [ -n "$LATEST_TAG" ]; then
  echo "Checking out latest tag: $LATEST_TAG"
  git checkout "$LATEST_TAG"
else
  echo "No tags found, staying on default branch: $(git branch --show-current)"
fi

# Record the version for documentation
CURRENT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || git rev-parse --short HEAD)
echo "Version: $CURRENT_VERSION"

cd ../../..
```

If the user provided a specific tag/version, check that out instead.

### 3. Initialize Skill with init_skill.py

```bash
python3 .claude/skills/skill-creator/scripts/init_skill.py "$SKILL_NAME" \
  --path "plugins/$SKILL_NAME/skills"
```

This generates a template SKILL.md and placeholder directories. The placeholder files
in `scripts/`, `references/`, and `assets/` will be replaced or deleted in Step 5.

### 4 + 5. Analyze Repository & Generate Skill Content (subagent)

**Dispatch a single `general-purpose` subagent** via the `Task` tool for both the exploration
and file-writing phases. This keeps the large volume of source code out of the main context.

Provide the subagent with:
- The absolute path to the cloned repo (`plugins/<skill-name>/<repo-dir>/`)
- The absolute path to the skill output directory (`plugins/<skill-name>/skills/<skill-name>/`)
- The skill name and version/tag checked out in Step 2

> **Important:** The subagent must invoke the **`skill-creator`** skill (via the `Skill` tool)
> before writing any skill content. That skill defines the quality standards, frontmatter rules,
> progressive-disclosure structure, and file-size limits that all skills must follow.

The subagent prompt should instruct it to:

**Phase A — Analyze the Repository:**
- Read `README.md` and any `docs/` directory for overview and usage
- Identify the primary language (C/C++, Python, Rust, TypeScript, etc.)
- **C/C++**: Read all public headers (`include/*.h`, `*.h`) — these are the source of truth
- **Python**: Read `__init__.py` exports, docstrings, type stubs (`.pyi`), or `docs/`
- **TypeScript/JavaScript**: Read `index.d.ts`, `types.d.ts`, or JSDoc comments
- **Rust**: Read `src/lib.rs`, `pub` declarations, and any `docs.rs`-style docs
- **Other**: Read the primary module/package entry points and their public interface
- Look for `examples/` or `samples/` directories for usage patterns
- Identify logical API domains to split into separate reference files (e.g., initialization,
  core operations, utilities, advanced features)
- Note the total number of public API functions/methods for the plugin description

**Phase B — Generate Skill Content:**

#### 5a. Write references/ files

Create one markdown file per logical API domain. Use `api-<domain>.md` naming.
Always create a `workflows.md` with complete working examples.

**Structure for each `references/api-<domain>.md`:**
```markdown
# <Domain Name>

## Table of Contents
1. [Group 1](#group-1)
2. [Group 2](#group-2)

## Group 1

### `function_signature(param: type) -> return_type`
**Description:** What it does.
**Parameters:** param descriptions.
**Returns:** Return value description.
**Example:**
\`\`\`<language>
// working example
\`\`\`
```

**Structure for `references/workflows.md`:**
```markdown
# Workflows

## Table of Contents
1. [Quick Start](#quick-start)
2. [Common Use Case 1](#use-case-1)
...

## Quick Start
\`\`\`<language>
// minimal working example showing the most common scenario
\`\`\`

## Common Use Case 1
\`\`\`<language>
// complete working example
\`\`\`
```

#### 5b. Write SKILL.md

Replace the generated template with a complete SKILL.md. Rules:
- **Concise and dense** — SKILL.md is always loaded into context; every line must earn
  its place; do not pad it with API details that belong in `references/`
- Under 500 lines total; if approaching the limit, move content to `references/`
- `name:` in frontmatter must match the directory name exactly
- Quote descriptions containing colons: `description: "Use when: ..."`

```markdown
---
name: <skill-name>
description: "<1-2 sentences: what the lib does and when to use this skill>"
---

# <Skill Title>

## Overview

<2-3 sentences: what the library is, primary use cases, language/platform>

**Version:** <version from Step 2>
**Language:** <primary language>
**License:** <license from repo>

## Quick Start

\`\`\`<language>
// minimal working example
\`\`\`

## Core Concepts

<3-5 bullet points covering the main concepts users need to know>

## API Reference

| Domain | File | Description |
|--------|------|-------------|
| <Domain 1> | [api-domain1.md](references/api-domain1.md) | <brief desc> |
| <Domain 2> | [api-domain2.md](references/api-domain2.md) | <brief desc> |
| Workflows | [workflows.md](references/workflows.md) | Working examples |

## Common Workflows

See [references/workflows.md](references/workflows.md) for complete examples.

Quick reference:
- **<task 1>**: see workflows.md#...
- **<task 2>**: see workflows.md#...

## Key Considerations

<3-5 bullet points on gotchas, performance notes, or important constraints>
```

#### 5c. Clean up init_skill.py artifacts

The subagent should also delete the unused placeholder files created by `init_skill.py`:
```bash
rm -rf "plugins/$SKILL_NAME/skills/$SKILL_NAME/scripts"
rm -rf "plugins/$SKILL_NAME/skills/$SKILL_NAME/assets"
rm -f  "plugins/$SKILL_NAME/skills/$SKILL_NAME/references/api_reference.md"
```

**The subagent should return a summary** of: the license found, the API domains created,
the reference files written, the version documented, and the plugin description text — so
the main context can use these values in Steps 6–11 without re-reading the files.

### 6. Create plugin.json

Create `plugins/<skill-name>/.claude-plugin/plugin.json`:

```json
{
  "name": "<skill-name>",
  "version": "1.0.0",
  "description": "<one sentence: lib name, version, function count or key capability>",
  "author": {
    "name": "Datathings",
    "email": "contact@datathings.com"
  },
  "license": "<license from upstream repo, e.g. MIT, Apache-2.0>",
  "repository": "https://github.com/datathings/marketplace",
  "keywords": ["<primary-language>", "<lib-name>", "<key-use-case-1>", "<key-use-case-2>"],
  "skills": "./skills/"
}
```

### 7. Create .gitignore

Create `plugins/<skill-name>/.gitignore` to exclude the standalone git clone:

```
*.skill
<repo-dir>/
__pycache__/
*.pyc
```

Where `<repo-dir>` is the directory name used when cloning in Step 2.

### 8. Update Marketplace Registry Files

Two JSON registries need a new entry.

**`.claude-plugin/marketplace.json`** — add to the `plugins` array:

```json
{
  "name": "<skill-name>",
  "source": "./plugins/<skill-name>",
  "description": "<lib name> <language> skill - <key capability summary>",
  "version": "1.0.0"
}
```

**`marketplace.json`** (legacy format) — add to the `skills` array:

```json
{
  "name": "<skill-name>",
  "path": "./plugins/<skill-name>/skills/<skill-name>",
  "version": "1.0.0",
  "description": "<lib name> <language> skill - <key capability summary>"
}
```

Also update the top-level `description` field in `marketplace.json` to mention the new library.

### 9. Update README.md

Add the new skill in four places:

**Plugins table** (in the `## Plugins` section):
```markdown
| **<skill-name>** | Skill | 1.0.0 | <short description of the library and skill coverage> |
```

**Install instructions** (in `### Install Plugins`):
```markdown
<lib name>:
\`\`\`bash
/plugin install <skill-name>@datathings
\`\`\`
```

**Plugin Details section** (add a `### <skill-name>` subsection):
```markdown
### <skill-name>

<2-3 sentences describing what the library does and what the skill provides>
- Key capability 1
- Key capability 2
```

**Standalone Skill Files list** (in `## Standalone Skill Files`):
```markdown
├── <skill-name>.skill     # <short description>
```

### 10. Package the Skill

```bash
# From repo root — package.sh auto-discovers all skills in plugins/*/skills/*/
./package.sh "$SKILL_NAME"
```

This creates `skills/<skill-name>.skill`. Fix any validation errors reported.

> **Note:** `package.sh` and `bump-version.sh` both auto-discover plugins via directory
> globs — no manual edits to those scripts are needed.

### 11. Create Update Command

Create `.claude/commands/update_<skill_name>.md` following the same pattern as
`update-llamacpp.md` and `update-blas-lapack.md`. Tailor it to the specific library
based on what was learned in Step 4 (which headers/files are the source of truth,
tag naming conventions, which reference files were created).

> **Reference example:** Read `.claude/commands/update-llamacpp.md` as the concrete
> model to follow — structure, section names, subagent dispatch pattern, and success
> criteria checklist. Fill in all `<placeholders>` with the actual values for this
> library. The generated command must also instruct Claude to dispatch the heavy
> analysis + documentation work to a subagent, exactly as `update-llamacpp.md` does.

**Template:**

```markdown
# Update <Skill Title> Skill to Latest Version

This command updates the <lib name> library to the latest version and synchronizes
all skill files with the updated API.

**Working directory:** `plugins/<skill-name>/`

## Repository Structure

\`\`\`
plugins/<skill-name>/
├── skills/<skill-name>/          # Skill content (SKILL.md and references/)
├── <repo-dir>/                   # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json   # Plugin manifest
\`\`\`

The `<repo-dir>/` directory is a **standalone git clone** from <repo-url> - updated
independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

\`\`\`bash
cd plugins/<skill-name>/<repo-dir>

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Adapt tag pattern to match this repo's convention (e.g., v#.#.#, b####)
LATEST_TAG=$(git tag --sort=-version:refname | grep "^<tag-prefix>" | head -n 1)
echo "Latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50
fi

git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
\`\`\`

### 2. Analyze API Changes

Scan the source-of-truth files:
<list the specific files that define the public API for this library>

Look for:
- New functions/methods added since last update
- Deprecated or removed functions
- Modified signatures (parameter changes, return types)
- New types, enums, or constants

### 3. Update Skill Documentation Files

Update the following files in `skills/<skill-name>/references/`:
<list each reference file and what it covers>

### 4. Update Main SKILL.md

- Update the library version/tag in the documentation
- Update function/method counts in the overview
- Add notes about newly supported features
- Update the API reference table if files were added/removed

### 5. Remove Deprecated Content

- Search documentation files for functions removed from the source-of-truth files
- Remove or replace deprecated references
- Note breaking changes

### 6. Validate Changes

- Cross-reference all documented functions against the source-of-truth files
- Ensure signatures match exactly
- Verify all new public API functions are documented

### 7. Package Updated Skill

\`\`\`bash
./package.sh <skill-name>
\`\`\`

This creates `skills/<skill-name>.skill`.

## Key Considerations

- **Source of truth**: Always derive signatures from <source files>, not secondary docs
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them
- **New features**: Major new features should get workflow examples
- **Backward compatibility**: Note any breaking changes in SKILL.md

## Success Criteria

- ✅ Library updated to latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API functions from source-of-truth files are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match source-of-truth exactly
- ✅ Code examples in workflows.md are valid
- ✅ Package builds successfully with `./package.sh <skill-name>`
```

Fill in all `<placeholders>` with the actual values for the specific library.

---

## Key Considerations

- **Skill name = directory name**: `skills/<name>/SKILL.md` must have `name: <name>` in frontmatter
- **Quote colons in descriptions**: Unquoted colons break YAML parsing
- **References one level deep**: All reference files link directly from SKILL.md, no subdirectories
- **Gitignore the clone**: The `<repo-dir>/` must be in `.gitignore` (it's a nested git repo)
- **Version in SKILL.md**: Always document which upstream version/tag the skill covers
- **Signatures must be exact**: Copy directly from headers/type defs, do not paraphrase
- **Keep SKILL.md under 500 lines**: Move API details to `references/` files

## Success Criteria

- [ ] `plugins/<skill-name>/` directory exists with correct structure
- [ ] Git repo is cloned and checked out at correct version
- [ ] `skills/<skill-name>/SKILL.md` has valid YAML frontmatter with `name` and `description`
- [ ] `references/` contains at minimum one API file and `workflows.md`
- [ ] `plugin.json` is valid JSON with all required fields
- [ ] `.gitignore` excludes the cloned repo directory and `*.skill`
- [ ] `.claude-plugin/marketplace.json` includes the new plugin entry
- [ ] `marketplace.json` (legacy) includes the new skill entry
- [ ] `README.md` updated in all four locations (table, install, details, skill files list)
- [ ] `./package.sh <skill-name>` runs successfully and produces `skills/<skill-name>.skill`
- [ ] `.claude/commands/update_<skill_name>.md` created with library-specific update instructions
