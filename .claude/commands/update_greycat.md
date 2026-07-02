---
description: Pull latest upstream greycat and upgrade both the greycat and greycat-c skills
allowed-tools: Bash(*), Read, Edit, Write, Grep, Glob, Task
---

# Upgrade GreyCat Skills

Single maintenance command that syncs both GreyCat skills from the upstream `../greycat`
checkout. It:

1. Pulls the latest upstream sources (`git pull` in `../greycat`)
2. Upgrades the **greycat** skill — a straight copy of the upstream skill docs
3. Upgrades the **greycat-c** skill — analyzes the upstream C headers and updates the docs
4. Re-packages both `.skill` files

**All paths below are relative to the marketplace repo root** (where this command is run).
`../greycat` resolves to the sibling GreyCat checkout (a git repo).

## Step 0: Pull Latest Upstream

Pull the newest sources into the sibling checkout:

```bash
git -C ../greycat pull
```

Also read the upstream version — it is the source of truth for the SDK version string used
later:

```bash
cat ../greycat/VERSION   # e.g. 8.0
```

---

## Part A — Upgrade the `greycat` Skill (copy)

The greycat skill is now maintained upstream at `../greycat/skills/` (a `SKILL.md` plus a
`reference/` folder). Upgrading it is a straight copy into the plugin — **no lib install, no
GCL syncing, no signature analysis**.

**Source:** `../greycat/skills/`
**Destination:** `./plugins/greycat/skills/greycat/`

Clean the destination first (removes any renamed/deleted reference files), then copy the
upstream skill in verbatim:

```bash
# Wipe old skill contents (keep the greycat/ dir itself)
rm -rf ./plugins/greycat/skills/greycat/reference
rm -f  ./plugins/greycat/skills/greycat/SKILL.md

# Copy the upstream skill (SKILL.md + reference/*) into place
cp -r ../greycat/skills/. ./plugins/greycat/skills/greycat/
```

Verify the copy — the destination should mirror the source exactly:

```bash
diff -r ../greycat/skills/ ./plugins/greycat/skills/greycat/ && echo "greycat skill in sync"
```

---

## Part B — Upgrade the `greycat-c` Skill (header analysis)

The greycat-c skill documents the native C SDK. Its source of truth is the upstream header
tree, which is **always** at:

**Headers:** `../greycat/core/include/` (`greycat.h` + `gc/*.h`)
**Destination docs:** `./plugins/greycat-c/skills/greycat-c/`

```
plugins/greycat-c/
├── skills/greycat-c/
│   ├── SKILL.md              # Overview, quick reference — tracks "SDK X.X"
│   └── references/
│       ├── api_collections.md
│       ├── api_core.md
│       ├── api_memory_text.md
│       ├── api_runtime_storage.md
│       ├── api_services.md
│       ├── plugin_development.md
│       └── standard_library.md
└── .claude-plugin/plugin.json
```

### B1. Record Old Version

Read `./plugins/greycat-c/skills/greycat-c/SKILL.md` and note the currently documented
`SDK X.X` version — the baseline for the diff summary reported back to the user. The new
version is the value from `../greycat/VERSION` (Step 0).

### B2. Analyze Headers & Update Docs (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the header analysis and doc
rewrite — this keeps the large header content out of the main context.

Provide the subagent with:
- Header path: `../greycat/core/include/` (`greycat.h` and everything under `gc/`)
- Existing skill docs path: `./plugins/greycat-c/skills/greycat-c/`
- Old version (from B1) and new version (from `../greycat/VERSION`)

The subagent should:

**Analyze the headers** — extract and categorize all public API elements:
- Function prototypes: name, return type, parameters (types + names)
- `typedef`s and `struct` definitions with all fields
- `enum` types and their values
- Preprocessor macros (`#define`) that are part of the public API
- Deprecation markers (`__attribute__((deprecated))`, comments, naming conventions)

**Diff against existing documentation:**
- New symbols not yet documented
- Removed symbols still present in the docs
- Signature changes (parameter type/name/order, return type)
- New types, enums, or macros

**Update `references/*.md`** — keep the existing grouping (core, collections, memory/text,
runtime/storage, services, plugin development, standard library); add new groups only if the
SDK gained a new domain:
- Add entries for new functions/types/macros
- Update changed signatures **exactly** as they appear in the header
- Remove entries for symbols no longer in the header

**Update `SKILL.md`:**
- Keep it **concise and dense** — it is always loaded into context; full API details belong
  in `references/`
- Update the `Tracks SDK X.X` marker to the new version
- Update any function/type counts in the overview if they changed
- Note breaking changes or major new capabilities in "Key Considerations"

**The subagent returns a summary** of: symbols added, removed, signatures changed, and any
breaking changes — for the main context to report to the user.

### B3. Bump plugin.json (if needed)

`./plugins/greycat-c/.claude-plugin/plugin.json` carries the **plugin's own** semantic
version (independent of the SDK version). Bump it if the docs changed materially; otherwise
leave it. Do **not** overwrite it with the SDK version — they are different numbers.

---

## Step C: Re-package Both Skills

From the repo root:

```bash
./package.sh greycat
./package.sh greycat-c
```

This produces `skills/greycat.skill` and `skills/greycat-c.skill`. Fix any validation errors
reported.

## Success Criteria

- [ ] `git -C ../greycat pull` completed (upstream up to date)
- [ ] `./plugins/greycat/skills/greycat/` is identical to `../greycat/skills/` (`diff -r` clean)
- [ ] greycat-c docs reflect the current `../greycat/core/include/` headers (all new symbols
      documented, removed symbols deleted, signatures exact)
- [ ] `SKILL.md` `Tracks SDK X.X` matches `../greycat/VERSION`
- [ ] Both `./package.sh greycat` and `./package.sh greycat-c` run without errors
