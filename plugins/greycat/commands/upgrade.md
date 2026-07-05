---
name: upgrade
description: Upgrade GreyCat libraries AND frontend package.json to latest, then lint + test and fix all issues in back and front
allowed-tools: Bash(*), Read, Edit, Grep, Task
---

# Upgrade GreyCat + Frontend

**Purpose**: Upgrade **both ends' dependencies** to latest, then **lint + test both ends and fix every failure the upgrade surfaces until green**:
1. **Backend** — every `@library` in `project.gcl` to latest from get.greycat.io.
2. **Frontend** — every `frontend/package.json` dependency to its latest published version (exact-pinned).
3. **Verify + fix** — backend `greycat-lang lint` + `greycat test`; frontend `pnpm lint` + `vp build` (+ `pnpm test` if the project uses Vitest). Fix all lint/type/test failures the upgrade introduced in **both** back and front, re-running after each fix until everything passes.

**Run When**: Monthly maintenance, before major releases, on announced breaking changes.

## Two tracks — parallelize under ultracode

The **backend track** (bump `@library` → `greycat install` → `greycat-lang fmt --mode=check` + `lint` → `greycat test` → fix loop) and the **frontend track** (bump `package.json` → `greycat codegen ts` → `pnpm lint` → `vp build`/`pnpm test` → fix loop) are largely independent. Under **ultracode/ultrathink**, dispatch them as **two parallel subagents** (`Task`) and reconcile at the end; otherwise run sequentially.

---

## Libraries

**Core**: `std` (required), `explorer` (dev UI).
**Other published**: `ai`, `algebra`, `kafka`, `mqtt`, `opcua`, `ftp`, `ssh`, `osm`, `useragent`, `finance`, `powerflow`, `powergrid` (successor to `powerflow`), `text_search`, `fcs`, and Pro-licensed `ifc`, `sql`, `openid`.

> `s3` is **not** a library — S3 access is part of `std` (IO). Don't add `@library("s3", ...)`.
> **Resolve each library from its OWN endpoint** — don't assume one shared version. The only name quirk: `std`'s URL path is `core`.

---

## Backend Step 1: Resolve latest per lib

URL path-name = `@library` name, except `std` → `core`:
```bash
latest_version() {           # $1 = @library name
  local path="$1"; [ "$1" = "std" ] && path="core"     # std's URL path is 'core'
  curl -s "https://get.greycat.io/files/${path}/dev/latest" | cut -d'/' -f2
}
```

## Backend Step 2: Detect current libraries

```bash
CURRENT_LIBS=$(grep -o '@library("[^"]*"' project.gcl | sed 's/@library("//;s/"//' | sort -u)
[ -z "$CURRENT_LIBS" ] && { echo "ERROR: no @library in project.gcl"; exit 1; }
```

## Backend Step 3: Compare (early-exit if all up-to-date)

For each `lib in $CURRENT_LIBS`:
```bash
CURRENT_VERSION=$(grep "@library(\"$lib\"" project.gcl | sed -n "s/.*@library(\"$lib\", \"\([^\"]*\)\").*/\1/p")
LATEST_VERSION=$(latest_version "$lib")            # per-lib endpoint
[ -z "$LATEST_VERSION" ] && { echo "  WARN: no published version for '$lib' — skipping"; continue; }

[ "$CURRENT_VERSION" = "$LATEST_VERSION" ] && echo "  ✓ $lib $CURRENT_VERSION (up-to-date)" \
    || { echo "  ↑ $lib $CURRENT_VERSION → $LATEST_VERSION"; UPDATES_NEEDED=true; }
```

## Backend Step 4: Confirm (AskUserQuestion)

Show summary table, then ask: **A)** Yes, update all (Recommended) · **B)** Show what changes in `project.gcl` first · **C)** Cancel. For "Show changes" → display before/after `@library` lines, then re-ask.

## Backend Step 5: Update project.gcl

```bash
for lib in $CURRENT_LIBS; do
  LATEST_VERSION=$(latest_version "$lib")           # per-lib endpoint (see Step 1)
  [ -z "$LATEST_VERSION" ] && continue
  sed -i "s/@library(\"$lib\", \"[^\"]*\")/@library(\"$lib\", \"$LATEST_VERSION\")/" project.gcl
done
```

## Backend Step 6: Install + verify + fix

```bash
greycat install
greycat-lang fmt --mode=check
greycat-lang lint
greycat test
```

**Fix loop**: the upgrade CAN break backend code (renamed/removed fns, changed signatures, deprecated decorators). Don't stop at reporting — **fix every lint/test failure**, re-running `greycat-lang lint && greycat test` until both pass. See "Migration of breaking changes" and "Troubleshooting" for common patterns and single-lib rollback.

## ⚠ Backend Step 7: Persisted schema check

Library upgrades can change persisted-type shapes. If `gcdata/` holds data from the previous version, the server may refuse to start on a schema mismatch — **NO automatic migration**.
1. `./bin/greycat serve` — try startup; abort if it errors on schema.
2. If schema rejects:
   - **Dev**: stop server, `rm -rf gcdata`, replay importers.
   - **Prod**: `mv gcdata gcdata_bk_<ts>`, re-import, then `./bin/greycat defrag`.
3. After replays: `./bin/greycat defrag` to reclaim storage.

See `/greycat:migrate` Operation A and D for the full safe-rollback flow.

### Adding new libraries
1. Add `@library("kafka", "<latest>");` to `project.gcl`.
2. Run `/greycat:upgrade` (auto-detects new libs).
3. Verify: `ls lib/ | grep kafka`, `greycat-lang lint`, `greycat run`.

---

## Frontend Step 8: package.json upgrade + fix

Stack: **VitePlus (`vp`) + Lit (light DOM) + TypeScript + Shoelace + @greycat/web + lucide-static, pnpm.**

If `frontend/` exists, upgrade **every** dependency to its latest published version (not just the stack below). Pin **exact latest** (no `^`/`~`). The **core** stack must stay present and current; the tooling rows are upgraded only if the project already uses them:

| Package | Role |
|---------|------|
| `lit` | web components |
| `@shoelace-style/shoelace` | UI kit (must satisfy `@greycat/web`'s peer range) |
| `lucide-static` | icons (native, self-hosted SVG) |
| `@greycat/web` | typed client + `gui-*` widgets (registry tarball, not npm) |
| `vite-plus` | build toolchain (`vp`) |
| `typescript` | language |
| `vitest` | frontend tests — optional, only if the project uses it |
| `lighthouse` | perf/SEO/a11y audits — optional CLI/script |
| `i18next` (+ language detector) | i18n, if multi-locale |
| `maplibre-gl` / `chart.js` / `d3` | maps & data-viz, if used |

```bash
cd frontend
pnpm outdated           # inspect current vs latest for ALL deps
pnpm up --latest        # bumps every dep (deps + devDeps) to latest
# then re-pin to exact versions in package.json (drop ^/~) and reinstall
pnpm install
```
> `@greycat/web` is not on npm — bump it by re-pinning its registry tarball URL to track the project's `std` branch/version (see frontend stack notes), **not** via `pnpm up`.

**pnpm release-age gate**: pnpm 11 may reject just-published packages (and ignores `minimumReleaseAgeExclude` for that audit). To ship trusted just-released tooling, set `minimumReleaseAge: 0` (or add exact exclusions) in `pnpm-workspace.yaml`.

**Verify + fix (frontend)**:
```bash
greycat codegen ts        # regen typed client against upgraded backend
cd frontend
pnpm lint                 # typecheck (always)
pnpm test                 # vitest (only if the project uses it)
vp build                  # production build (always)
```
**Fix loop**: major bumps (Shoelace/Lit/TypeScript/Vite-Plus) can change component APIs, types, and build config — check changelogs. **Fix every `pnpm lint` / `vp build` failure** (plus `pnpm test` if Vitest is configured), re-running until all pass. Then serve (`greycat dev`) and, if the project has a Lighthouse script/CLI, confirm perf/SEO/a11y/best-practices ≥ 90.

---

## Channels

- **Dev**: `https://get.greycat.io/files/<lib>/dev/latest` — latest, may have breaking changes (**default** for this command).
- **Release**: `https://get.greycat.io/files/<lib>/releases/latest` — stable.

⚠ `<lib>` is the **URL path-name**: `@library("std", ...)` → path `core` (e.g. `.../files/core/dev/latest`); all others match (`ai`, `explorer`, `algebra`, `kafka`, …).

---

## Troubleshooting

### Lint fails after update
- Check https://doc.greycat.io/changelog for breaking changes.
- Save output: `greycat-lang lint 2>&1 | tee lint-errors.txt`
- Common patterns: function renamed/removed, signature changed, decorator deprecated.
- **Rollback**: `git checkout project.gcl && greycat install && greycat-lang lint`
- **Incremental**: update ONE lib at a time in `project.gcl`, lint between each.

### Fetch fails
```bash
ping get.greycat.io
curl -v "https://get.greycat.io/files/core/dev/latest"   # std's URL path is 'core'
```

### Install fails
```bash
rm -rf lib/ && greycat install
df -h .                 # disk space
ls -la lib/             # permissions
```

---

## Migration of breaking changes

```bash
# Find deprecated usage
grep -r "DataService::parse" src/ --include="*.gcl"

# Replace
find src/ -name "*.gcl" -exec sed -i 's/DataService::parse/DataService::parseString/g' {} \;

# Verify
greycat-lang lint && greycat test

# Commit
git add project.gcl lib/
git commit -m "Update GreyCat libraries

- std: <old> → <new>
- ...

Breaking changes:
- ..."
```

---

## Notes

- **Bump FIRST, then fix**: bumps are mechanical; this command does **not** stop there — it lints, tests, and **fixes all resulting breakage** in both back and front until green.
- **`greycat --version` = `0.0.0`** on dev builds — not a bug, treat as "latest".
- **gcdata NOT auto-migrated** on schema drift — see "Persisted schema check".
- **Rollback**: `git checkout project.gcl` for code; for prod data, rely on `gcdata_bk_*` rotation.

---

## Verify (both ends green)

**Backend**:
```bash
grep '@library(' project.gcl   # versions bumped
ls -la lib/                    # installed
greycat-lang fmt --mode=check  # formatting
greycat-lang lint              # syntax/types
greycat test                   # tests
greycat serve                  # runtime
```

**Frontend** (if `frontend/` exists):
```bash
cd frontend
grep -E '"(lit|@shoelace-style/shoelace|typescript|vite-plus)"' package.json  # core stack, exact-pinned, no ^/~
pnpm lint                      # typecheck (always)
pnpm test                      # vitest (only if the project uses it)
vp build                       # production build (always)
pnpm lighthouse:ci             # perf/SEO/a11y/best-practices ≥ 90 — optional: the script if present, else the lighthouse CLI
```

**Definition of done**: `greycat-lang lint && greycat test` and `pnpm lint && vp build` (plus `pnpm test` if the project uses Vitest) pass with no upgrade-introduced failures remaining.
