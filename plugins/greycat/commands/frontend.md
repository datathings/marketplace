---
name: frontend
description: Comprehensive GreyCat frontend review — correctness, the prescribed VitePlus + Lit + Web Awesome (or similar) + @greycat/web stack, performance, Lighthouse/SEO, type safety, and testing
argument-hint: "[lens ...] | help"
allowed-tools: Bash, Read, Grep, Glob, Task
---

# Frontend Review

**Purpose**: The single hub for GreyCat web-UI quality — correctness, the one prescribed stack, TypeScript/Lit type safety, Lighthouse performance, LLM-friendly SEO, testing, and common pitfalls. (It absorbs the frontend phases that used to live in the separate `typecheck` / `optimize` / `coverage` commands.)

**Companion**: for GCL backend quality use `/greycat:backend`.

---

## Scope — arguments: `$ARGUMENTS`

Default = **everything**. No upfront questions — resolve the scope from the arguments above and go.

- **Empty** → run every lens (the full checklist below).
- **`help`** → print the lens table below (keyword + what it covers) and **stop** — do not run the review.
- **One or more lens keywords** → run only those checklist sections.
- **Unknown keyword** → print the lens table, point out the bad keyword, and stop.

| Keyword | Lens (checklist §) |
|---------|--------------------|
| `layout` | §1 Layout & config + prescribed-stack presence |
| `ts` | §2–2c TypeScript quality, codegen freshness, SDK-derived strings |
| `components` | §3–3e Lit patterns, Web Awesome, theming, icons, dependencies |
| `init` | §4 Init / login gate |
| `perf` | §5 Performance |
| `lighthouse` | §6 Lighthouse audit (auto-skips if prerequisites are missing — see §6) |
| `seo` | §7 SEO + LLM discoverability |
| `testing` | §8 Type-check / Vitest / coverage gates |

Example: `/greycat:frontend lighthouse seo` audits only §6 + §7.

Any lens not run — out of scope or unmet prerequisites — must appear in the Output as `SKIPPED: <reason>`, never silently omitted (a silent skip reads as "passed").

Read [reference/webapp.md](../skills/greycat/reference/webapp.md) for the full prescribed toolchain, layout, and integration contract — this review checks against exactly that, with one deviation: **the frontend source dir is `frontend/`, not `app/`** (the `~` alias maps to `frontend/`).

---

## How to run this review — ultrathink + ultracode

**Ultrathink (always).** Reason deeply about *why* each rule exists (why shadow DOM, why the init gate, why per-component Web Awesome imports) before flagging — a violation you can't tie to a broken render, a blank page, a 422, or a Lighthouse regression is not a finding.

**Ultracode (when available).** If multi-agent orchestration is on, run this as a **Workflow** — the checklist below splits cleanly into independent lenses:
1. **Fan out** — one agent per **in-scope** lens (see Scope above): *(a)* layout & config + TypeScript, *(b)* Lit/Web Awesome/theming/icons component patterns, *(c)* init/login gate, *(d)* performance + Lighthouse, *(e)* SEO + LLM discoverability, *(f)* testing. Each returns structured findings (`file`, `line`, `severity`, `problem`, `fix`).
2. **Verify** — for CRITICAL/HIGH findings, a second agent confirms it against the served app or the actual config, not just a grep hit.
3. **Synthesize** — one severity-grouped report.

If ultracode is **not** available, walk the checklist yourself as one deep pass.

---

## Prescribed Stack (the baseline this review checks against)

Every GreyCat webapp uses the same toolchain, layout, and design tokens. Deviate only when a project has a concrete reason to.

| Concern | Choice |
|---------|--------|
| Toolchain | **VitePlus** — global `vp` CLI + local `vite-plus` package; explicit `vite.config.ts`, **no plugin** |
| Pages | **MPA** — each route is a real HTML page under `frontend/routes/`; URL == file path, no SPA router |
| Components | **Lit** in **shadow DOM** (the Lit default) — one root element per route, a component only for views reused across routes; each owns its styles in a `static styles` block; `@customElement('app-…')` |
| Language | **TypeScript** — `experimentalDecorators: true`, `useDefineForClassFields: false`, `moduleResolution: "bundler"` |
| UI kit | **Web Awesome** (`wa-*`) by default, or an equivalent web-component design-system library if the project has a concrete reason to pick a different one — button, input, dialog, tabs, tooltip, date-picker, and every other visual element; its own CSS custom-property tokens (`--wa-*` for Web Awesome) are the single source of truth for color, spacing, and type |
| Theme | The chosen kit's own base stylesheet (for Web Awesome: `@awesome.me/webawesome/dist/styles/webawesome.css` — tokens + light/dark) + **`frontend/theme.css`** (the kit's token namespace, brand-overridden), imported **after** the kit's base stylesheet |
| Client | **`@greycat/web/sdk`** for every backend call — headless, ships no components or CSS → `greycat codegen ts` is mandatory (`project.d.ts`) |
| Icons | **lucide-static** — prebuilt SVG strings inlined via Lit `unsafeSVG`; self-hosted, `currentColor` + `aria-hidden`, no CDN fetch |
| Package manager | **pnpm** |

> **`vp` and pnpm are different layers, not alternatives.** pnpm is the *package manager* (fetches deps into `node_modules`); `vp` (VitePlus, rolldown/oxc-based) is the *build toolchain* that bundles `frontend/` → `webroot/`, replacing plain Vite. `vp install` delegates to the package manager, so they work together.

> `@greycat/web` is **not on npm** — it ships as a tarball URL from GreyCat's registry, tracking the same branch (`dev`/`stable`) and version as the project's `std`. Web Awesome and Lit are ordinary semver deps; Web Awesome must satisfy `@greycat/web`'s peer range.

```bash
# Confirm the prescribed stack is present
grep -nE '"(vite-plus|lit|@awesome.me/webawesome|@greycat/web)"' package.json
grep -q 'get.greycat.io/files/sdk/web' package.json || echo "⚠ @greycat/web not pinned to a registry tarball URL"
grep -q '"vite-plus"' package.json || echo "⚠ not on VitePlus — expected the vp toolchain"
```

---

## Checklist

Scan `.ts` under `frontend/` (source; `~` aliases `frontend/`) for:

### 1. Layout & config
- Frontend lives entirely under `frontend/` (`frontend/routes/` pages, `frontend/components/` reused views, `frontend/public/` copied as-is, `frontend/theme.css`). `src/` is the backend and is never renamed; `frontend/` is never in a GreyCat pragma (`@include("frontend")` breaks)
- `~` alias declared in **both** `vite.config.ts` and `tsconfig.json`; imports use `~/components/x`, never `../../`
- `vite.config.ts`: `root: 'frontend/routes'`, `base: './'`, `appType: 'mpa'`, `publicDir: frontend/public`, `outDir: webroot`, `emptyOutDir: true`
- Every route's `index.html` is listed in `build.rollupOptions.input` — Vite does not auto-discover extra HTML pages
```bash
grep -q "root: 'frontend/routes'" vite.config.ts || echo "⚠ root should be frontend/routes (MPA)"
grep -q "appType: 'mpa'" vite.config.ts || echo "⚠ missing appType: 'mpa' — MPA has no SPA fallback"
```

### 2. TypeScript quality
- `experimentalDecorators: true` **and** `useDefineForClassFields: false` in `tsconfig.json` — both load-bearing for Lit; vite-plus (rolldown/oxc) silently drops `@property` / emits unparseable `@customElement` otherwise (`vp build` still reports success, page loads blank)
- `moduleResolution: "bundler"`, `"strict": true`, `~` in `paths`, `project.d.ts` in `include`
- No `any` — use `gc.core.*` / `gc.project.*` types
- `project.d.ts` not manually edited
- Each `@customElement` extends `HTMLElementTagNameMap` (`declare global { interface HTMLElementTagNameMap { 'app-x': AppX } }`) so templates and `document.createElement` are typed
```bash
grep -q '"experimentalDecorators": true' tsconfig.json || echo "⚠ missing experimentalDecorators — Lit decorators break silently"
grep -q '"useDefineForClassFields": false' tsconfig.json || echo "⚠ missing useDefineForClassFields:false — @property shadowed by native field"
```

### 2b. Codegen freshness
Backend type / `@expose` changes require `greycat codegen ts` (regenerates `project.d.ts`). A client built against a stale ABI gets **HTTP 422** at runtime; the frontend type checker lies until regen.

### 2c. Derive backend strings from the SDK, never hard-code
```ts
// ❌ const key = "active";
// ✅ const s = gc.project.Status.active;  s.key === "active"   // enum static from the ABI
```
Enum entries are class statics built during `gc.sdk.init()` — don't touch `gc.<module>.*` at import/module-eval time, only after init resolves.

### 3. Component patterns (Lit, shadow DOM)
- One `LitElement` per file, `@customElement('app-…')` with a consistent kebab prefix
- **Shadow DOM** (the Lit default): each component owns its layout in a `static styles` block (a `css` tagged-template literal), reading the global `--wa-*` tokens — they inherit across the shadow boundary. Don't override `createRenderRoot()` to force light DOM; only `<wa-page>` and other Web-Awesome-owned elements render in light DOM, by their own design
- The route's **root element owns the init/login gate** (see §4) — `gc.sdk.init()` must resolve before any typed call works
- `@property()` for public inputs, `@state()` for internal state; update reactive props, don't recreate DOM
- Charts (chart.js or similar): instantiate after init, destroy in `disconnectedCallback`
```bash
grep -rn 'createRenderRoot' frontend/ --include="*.ts" && echo "⚠ overriding createRenderRoot — app components should stay in shadow DOM unless there's a concrete reason"
```

### 3b. UI-kit usage (Web Awesome `wa-*` by default, or an equivalent kit)
- Import components **individually** for tree-shaking (Web Awesome: `import '@awesome.me/webawesome/dist/components/button/button.js'`), never the whole bundle
- The chosen kit supplies every visual element — atomic controls (button, input, dialog, tabs, tooltip) and full-page layout (Web Awesome's `<wa-page>` for app-shell/sidebar/header, or the equivalent in whatever kit is used). There is no separate rich-widget kit; tables/charts/maps are hand-built (chart.js, a plain table) or a third-party web component, styled with the kit's tokens
- If the project uses a kit other than Web Awesome, confirm it's used consistently — don't mix two component libraries in the same app
```bash
grep -rn "from '@awesome.me/webawesome'" frontend/ --include="*.ts" && echo "⚠ whole-bundle import — switch to per-component imports"
```

### 3c. Theming (kit base stylesheet + `frontend/theme.css`)
- Import the kit's own base stylesheet (Web Awesome: `@awesome.me/webawesome/dist/styles/webawesome.css` — full `--wa-*` token set, light/dark out of the box), then `~/theme.css` **after** it so the brand overrides win the cascade
- `theme.css` holds only the kit's token overrides (brand color, radius, font) plus a small page reset — component-specific rules (`.card { ... }`) belong in that component's own `static styles`, since a bare selector in `theme.css` never crosses the shadow boundary
- Light/dark is class-based on `<html>` (Web Awesome: `wa-light` / `wa-dark`) — a single toggle switches every element and every component together
- No hardcoded colors/sizes in components — every value comes from the kit's design token (`--wa-*` for Web Awesome), re-branded in `theme.css`
```bash
grep -rq "@awesome\.me/webawesome/dist/styles/webawesome\.css" frontend/ || echo "ℹ webawesome.css not imported — confirm the project's chosen UI kit's base stylesheet is imported instead"
grep -rnE "#[0-9a-fA-F]{3,6}\b" frontend/ --include="*.ts" | grep -v theme.css && echo "⚠ raw hex outside theme.css — use the UI kit's design tokens"
```

### 3d. Icons (lucide-static)
- Use **`lucide-static`** — prebuilt SVG strings inlined via Lit `unsafeSVG`. Self-hosted/bundled, **no runtime icon fetch**
- Render with `stroke="currentColor"` + `aria-hidden="true"`; decorative icons must be `aria-hidden`, meaningful ones need an accessible label
- The UI kit's own built-in icons (e.g. chevrons in Web Awesome's `wa-select`) still work out of the box — Web Awesome's default there is Font Awesome, and needs the icon assets copied into `frontend/public/` with `setBasePath` only if you use `<wa-icon>` directly
```bash
grep -rnE "cdn|unpkg|jsdelivr|googleapis\.com/.*icon" frontend/ --include="*.ts" --include="*.html" && echo "⚠ runtime/CDN icon fetch — use lucide-static (inlined SVG) instead"
```

### 3e. Dependencies (on-stack + exact pins)
- The prescribed stack (`lit`, a UI kit — `@awesome.me/webawesome` by default, `lucide-static`, `@greycat/web`, `vite-plus`, `typescript`) must be present and current. Heavy/duplicate libs (`moment`, `lodash`, `jquery`) should be replaced with native / web-component equivalents — they bloat the bundle and hurt LCP/TBT
- Pin **exact** versions (no `^`/`~`) so builds are reproducible; `@greycat/web` is a registry tarball URL, not npm
```bash
grep -nE '"(moment|lodash|jquery)"' package.json && echo "⚠ heavyweight dep — prefer native / web-component equivalent"
grep -nE '"[~^]' package.json && echo "ℹ non-exact pins — pin exact latest for reproducible builds"
```

### 4. Init / login gate (CRITICAL ordering)
`gc.sdk.init()` loads the ABI over an authenticated endpoint, so it needs a session. The route's root element must:
1. `import '@greycat/web/sdk'` (the runtime: `gc` global, `init`, typed bindings, nothing else — it's headless, no components, no CSS) — import once
2. Call `gc.sdk.init()` with no args in `connectedCallback` — succeeds if a prior-login session cookie exists
3. On throw, render a login form → `gc.sdk.init({ auth: { username, password } })` (or `{ token }`)
4. Only after init resolves are `gc.<module>.*` calls usable
- Don't add `@permission("public")` to endpoints just to skip login in dev — that exposes them to anonymous callers
```bash
grep -rq "@greycat/web/sdk" frontend/ || echo "⚠ SDK runtime not imported — expected import '@greycat/web/sdk'"
grep -rn "from '@greycat/web'\b" frontend/ --include="*.ts" | grep -v "@greycat/web/sdk" && echo "⚠ import from something other than '@greycat/web/sdk' — the package is headless, nothing else to import"
```

### 5. Performance (feeds Lighthouse)
- Update reactive properties instead of recreating DOM trees; batch imperative attribute sets
- Virtualize/paginate large lists and tables instead of manual DOM loops over thousands of rows
- Code-split routes are natural in MPA; lazy-load heavy widgets (charts) with dynamic `import()`
- Tree-shake the UI kit (per-component imports); self-host icon assets (no CDN)
- Defer non-critical JS, inline critical CSS, long-cache hashed assets, reserve element sizes to avoid layout shift (CLS)

### 6. Lighthouse audit (performance · SEO · accessibility · best-practices)
**Prerequisites (auto-skip, never fail):** this lens needs the `lighthouse` CLI on PATH **and** a served app. If `which lighthouse` is empty, or nothing answers on the served origin and you cannot start `greycat dev`/`greycat serve`, do **not** fail the review and do **not** silently drop the lens — report it as `SKIPPED: <missing prerequisite>` in the Output and move on.

Target **≥ 90** in every category, on **both form factors**. Lighthouse defaults to **mobile** (emulated device + throttled CPU/network), so a single run only covers mobile — audit **desktop too** (`--preset=desktop`); mobile is the harder gate and easy to miss. Serve first (`greycat dev` or `greycat serve`), then run against the served origin for each:
```bash
# mobile (default) + desktop; ≥ 90 in every category on both
lighthouse http://localhost:8080 --only-categories=performance,accessibility,best-practices,seo
lighthouse http://localhost:8080 --preset=desktop --only-categories=performance,accessibility,best-practices,seo
grep -q '"lighthouse"' package.json || echo "ℹ no lighthouse script — add mobile + :desktop scripts (pnpm), or run the lighthouse CLI as above"
```

### 7. SEO + LLM-friendly SEO (always)
Check the route `index.html` files and the built `webroot/`:
- **Head**: `<html lang>`, unique `<title>`, `<meta name="description">`, canonical `<link rel="canonical">`, Open Graph + Twitter Card tags, `theme-color`, `<meta name="color-scheme">`, `viewport` (`width=device-width, initial-scale=1`)
- **Responsive / mobile usability**: layout adapts from mobile to desktop (fluid grids / `clamp()` / media queries, no fixed-px page widths); tap targets ≥ 24–48px, no horizontal scroll at 360px, text readable without zoom. This is a Lighthouse SEO + best-practices signal and the mobile audit surfaces it
- **Semantics**: landmark elements (`header`/`nav`/`main`/`article`/`footer`), correct heading order, `alt` text, ARIA labels. The prescribed stack renders app content into shadow DOM (Lit's default) — this is fine for JS-executing crawlers (Googlebot renders the DOM before indexing) but a non-JS text scraper (some LLM fetchers) sees only `index.html`'s light-DOM markup. Keep page-critical text (title, headings, meta, key copy) in `index.html` itself or a light-DOM landmark, not solely inside a component's shadow root
- **Structured data**: JSON-LD (`schema.org`) in `<head>`
- **Machine-readable for crawlers AND LLMs**: `robots.txt`, `sitemap.xml`, a web app manifest, and **`llms.txt`** (+ optional `llms-full.txt`) — a concise Markdown index of the app's purpose, key routes, and public endpoints. Ship them via `frontend/public/` (copied to `webroot/`)
- Descriptive link text, stable URLs (MPA gives real per-route URLs), per-route `<title>`/description
```bash
grep -iqrE '<meta name="description"' frontend/routes/ || echo "⚠ no meta description"
grep -iqrE 'og:title|twitter:card'      frontend/routes/ || echo "⚠ no Open Graph / Twitter Card tags"
grep -iqrE 'application/ld\+json'        frontend/routes/ || echo "⚠ no JSON-LD structured data"
for f in robots.txt sitemap.xml llms.txt site.webmanifest manifest.webmanifest; do
  [ -f "webroot/$f" ] || [ -f "frontend/public/$f" ] || echo "⚠ missing $f (SEO/LLM discoverability)"
done
```

### 8. Testing (type-check always · Vitest optional · Lighthouse as a gate)
No frontend test framework is *prescribed* — `greycat codegen ts && pnpm lint` (`tsc --noEmit`) plus the Lighthouse audit are the baseline gates. **Vitest** is the recommended runner **if the project has one** (jsdom for Lit component rendering + data-transform helpers); don't flag its absence, but if present treat untested components / formatters / error+loading states as gaps.
```bash
find frontend -name "*.test.ts" -o -name "*.spec.ts"     # inventory (empty is fine — Vitest is optional)
grep -q '"vitest"' package.json && pnpm test              # run only if the project uses Vitest
```
Treat any **Lighthouse category < 90** (on mobile *or* desktop) as an open coverage item, same weight as a missing unit test.

---

## Output

Start with a one-line scope recap: lenses run, and every lens **not** run as `SKIPPED: <reason>` (argument scope, missing `lighthouse` CLI, app not served, …).

Group by severity:
- **CRITICAL**: missing init/login gate (touching `gc.<module>.*` before `gc.sdk.init()` resolves), importing anything from `@greycat/web` other than `@greycat/web/sdk`, missing `experimentalDecorators`/`useDefineForClassFields`, security (raw `innerHTML`), an off-stack toolchain when VitePlus + Lit + a component UI kit is the standard
- **HIGH**: `theme.css` imported before the UI kit's base stylesheet, whole-bundle UI-kit import, missing route entries in `rollupOptions.input`, `@greycat/web` not pinned to a registry tarball, Lighthouse category < 90 on **mobile or desktop**, non-responsive layout, missing meta description / structured data
- **MEDIUM**: `any` usage, hardcoded colors/sizes instead of the UI kit's design tokens, stale `project.d.ts` (codegen not re-run), relative imports instead of `~`, missing `llms.txt` / sitemap / manifest, runtime/CDN asset fetch, overriding `createRenderRoot()` without a concrete reason
- **LOW**: style inconsistency, `emptyOutDir` mismatch, missing type casts
