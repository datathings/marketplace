---
name: upgrade
description: Update GreyCat libraries to latest available versions
allowed-tools: Bash(*), Read, Edit, Grep
---

# Update GreyCat Libraries

**Purpose**: Upgrade GreyCat libraries to latest versions from get.greycat.io.

**Run When**: Monthly maintenance, before major releases, on announced breaking changes.

---

## Libraries

**Core**: `std` (required), `explorer` (dev UI)
**Other published libraries**: `ai`, `algebra`, `kafka`, `mqtt`, `opcua`, `ftp`, `ssh`, `osm`, `useragent`, `finance`, `powerflow`, `powergrid` (successor to `powerflow`), `text_search`, `fcs`, and Pro-licensed `ifc`, `sql`, `openid`.

> `s3` is **not** a library — S3 access is part of `std` (IO). Don't add `@library("s3", ...)`.
> Don't assume a single shared version — resolve each library from **its own** endpoint (below). The only name quirk is `std`, whose URL path is `core`.

---

## Step 1: Fetch latest versions

Resolve each library from **its own** endpoint — the URL path-name equals the `@library` name, except `std` → `core`:
\`\`\`bash
latest_version() {           # $1 = @library name
  local path="$1"; [ "$1" = "std" ] && path="core"     # std's URL path is 'core'
  curl -s "https://get.greycat.io/files/${path}/dev/latest" | cut -d'/' -f2
}
\`\`\`

---

## Step 2: Detect current libraries

\`\`\`bash
CURRENT_LIBS=$(grep -o '@library("[^"]*"' project.gcl | sed 's/@library("//;s/"//' | sort -u)
[ -z "$CURRENT_LIBS" ] && { echo "ERROR: no @library in project.gcl"; exit 1; }
\`\`\`

---

## Step 3: Compare versions

For each `lib in $CURRENT_LIBS`:
\`\`\`bash
CURRENT_VERSION=$(grep "@library(\"$lib\"" project.gcl | sed -n "s/.*@library(\"$lib\", \"\([^\"]*\)\").*/\1/p")
LATEST_VERSION=$(latest_version "$lib")            # resolves each lib from its own endpoint
[ -z "$LATEST_VERSION" ] && { echo "  WARN: no published version for '$lib' — skipping"; continue; }

[ "$CURRENT_VERSION" = "$LATEST_VERSION" ] && echo "  ✓ $lib $CURRENT_VERSION (up-to-date)" \
    || { echo "  ↑ $lib $CURRENT_VERSION → $LATEST_VERSION"; UPDATES_NEEDED=true; }
\`\`\`

**Early exit** if all up-to-date.

---

## Step 4: User confirmation (AskUserQuestion)

Show summary table, then ask:
- A) Yes, update all (Recommended)
- B) Show me what will change in project.gcl first
- C) Cancel

For "Show changes" → display before/after `@library` lines, then re-ask.

---

## Step 5: Update project.gcl

\`\`\`bash
for lib in $CURRENT_LIBS; do
  LATEST_VERSION=$(latest_version "$lib")           # per-lib endpoint (see Step 1)
  [ -z "$LATEST_VERSION" ] && continue
  sed -i "s/@library(\"$lib\", \"[^\"]*\")/@library(\"$lib\", \"$LATEST_VERSION\")/" project.gcl
done
\`\`\`

---

## Step 6: Install + verify

\`\`\`bash
greycat install
greycat-lang lint
\`\`\`

---

## Step 7: ⚠ Persisted schema check

Library upgrades can change persisted-type shapes. If `gcdata/` contains data from previous version, the server may refuse to start with a schema mismatch — **NO automatic migration**.

**Next steps**:
1. `./bin/greycat serve` — try startup; abort if it errors on schema
2. If schema rejects:
   - **Dev**: stop server, `rm -rf gcdata`, replay importers
   - **Prod**: `mv gcdata gcdata_bk_<ts>`, re-import, then `./bin/greycat defrag`
3. After replays: `./bin/greycat defrag` to reclaim storage

See `/migrate` Operation A and D for full safe-rollback flow.

---

## Adding new libraries

1. Add `@library("kafka", "<latest>");` to `project.gcl`
2. Run `/upgrade` (auto-detects new libs)
3. Verify: `ls lib/ | grep kafka`, `greycat-lang lint`, `greycat run`

---

## Frontend dependencies (VitePlus + Lit + Shoelace + lucide-static stack)

If `frontend/` exists, keep the **prescribed stack** current too — pin **exact latest** published versions (no `^`/`~`):

| Package | Role |
|---------|------|
| `lit` | web components |
| `@shoelace-style/shoelace` | UI kit (must satisfy `@greycat/web`'s peer range) |
| `lucide-static` | icons (native, self-hosted SVG) |
| `@greycat/web` | typed client + `gui-*` widgets (registry tarball, not npm) |
| `vite-plus` | build toolchain (`vp`) |
| `typescript` | language |
| `vitest` | frontend tests |
| `lighthouse` | perf/SEO/a11y audits |
| `i18next` (+ language detector) | i18n, if multi-locale |
| `maplibre-gl` / `chart.js` / `d3` | maps & data-viz, if used |

\`\`\`bash
# Inspect current vs latest, then pin exact
pnpm outdated
pnpm up --latest lit @shoelace-style/shoelace lucide-static vite-plus typescript vitest lighthouse
# then re-pin to exact versions in package.json (drop ^/~) and reinstall
\`\`\`
> `@greycat/web` is not on npm — bump it by re-pinning its registry tarball URL to track the project's `std` branch/version (see the frontend stack notes), not via `pnpm up`.

**pnpm release-age gate**: pnpm 11 may reject just-published packages (and ignores `minimumReleaseAgeExclude` for that audit). To ship trusted just-released tooling, set `minimumReleaseAge: 0` (or add exact exclusions) in `pnpm-workspace.yaml`.

**After upgrading**: `greycat codegen ts` (regen client) → `pnpm lint` (typecheck) → `vp build` → serve (`greycat dev`) → `pnpm lighthouse:ci` (confirm perf/SEO/a11y/best-practices ≥ 90). Note Shoelace/Lit major bumps can change component APIs — check their changelogs like you would GreyCat's.

---

## Channels

- **Dev**: `https://get.greycat.io/files/<lib>/dev/latest` — latest, may have breaking changes
- **Release**: `https://get.greycat.io/files/<lib>/releases/latest` — stable

⚠ `<lib>` is the **URL path-name**, which differs from the `@library(...)` name for the standard library:
  - `@library("std", ...)` → URL path is `core` (e.g. `https://get.greycat.io/files/core/dev/latest`)
  - All other libs match: `ai`, `explorer`, `algebra`, `kafka`, …

This command uses **dev** by default.

---

## Troubleshooting

### Lint fails after update
- Check https://doc.greycat.io/changelog for breaking changes
- Save lint output: `greycat-lang lint 2>&1 | tee lint-errors.txt`
- Common patterns: function renamed/removed, signature changed, decorator deprecated
- **Rollback**: `git checkout project.gcl && greycat install && greycat-lang lint`
- **Incremental**: edit project.gcl to update ONE lib at a time, lint between each

### Fetch fails
\`\`\`bash
ping get.greycat.io
curl -v "https://get.greycat.io/files/core/dev/latest"   # std's URL path is 'core'
\`\`\`

### Install fails
\`\`\`bash
rm -rf lib/ && greycat install
df -h .                 # disk space
ls -la lib/             # permissions
\`\`\`

---

## Migration of breaking changes

\`\`\`bash
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
\`\`\`

---

## Notes

- **No code changes**: this command only updates library versions (backend `@library` + optional frontend deps)
- **Frontend deps**: keep Lit + Shoelace + lucide-static + VitePlus (`vp`) + TS + Vitest + Lighthouse on exact latest; re-audit with `pnpm lighthouse:ci`
- **gcdata NOT auto-migrated** on schema drift — see "Persisted schema check" above
- **Rollback**: `git checkout project.gcl` for code; for prod data, rely on `gcdata_bk_*` rotation
- **`greycat --version` = `0.0.0`** on dev builds — not a bug, treat as "latest"
- **Resolve each lib from its own endpoint** — don't assume one shared version across libraries

---

## Verify

\`\`\`bash
grep '@library(' project.gcl   # versions
ls -la lib/                    # installed
greycat-lang lint              # syntax
greycat test                   # tests
greycat serve                  # runtime
\`\`\`
