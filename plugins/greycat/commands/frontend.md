---
name: frontend
description: Review frontend codebase for code quality, performance, Lighthouse/SEO, and best practices using the Lit + Shoelace + Lucide stack
allowed-tools: Bash, Read, Grep, Glob
---

# Frontend Review

**Purpose**: Review the GreyCat frontend for correctness, the preferred stack, Lighthouse performance, LLM-friendly SEO, and common pitfalls.

Read [reference/webapp.md](../skills/greycat/reference/webapp.md) for `@greycat/web` bundling/integration context.

---

## Preferred Stack (the baseline this review checks against)

Pin **exact latest** published versions (no `^`/`~`); re-check for newer releases on every review/upgrade.

| Concern | Choice |
|---------|--------|
| Components | **Lit** (web components) — one `LitElement` per file, `@customElement('app-…')` |
| Language | **TypeScript** — `experimentalDecorators: true`, `useDefineForClassFields: false`, `moduleResolution: "bundler"` |
| UI kit | **Shoelace** (`@shoelace-style/shoelace`) — layout, cards, tabs, dialogs, date-picker |
| Icons | **Lucide** — `lucide` (tree-shakable icon factory) or `lucide-static` (prebuilt SVG strings, inlined via Lit `unsafeSVG`). Native/self-hosted, `currentColor` + `aria-hidden`. |
| Client | `@greycat/web` (typed SDK + `gui-*` widgets, e.g. virtualized `gui-table`) |
| Build | **Vite** + `@greycat/web/vite-plugin` (`root: 'frontend'`, `outDir: '../webroot'`, `emptyOutDir: false`) |
| i18n (if multi-locale) | **i18next** (+ `i18next-browser-languagedetector`) — drives `hreflang`/`lang` + translated meta |
| Data-viz (optional) | **chart.js** + **d3**; **maplibre-gl** for maps |
| Tests | **Vitest** |
| Audits | **Lighthouse** (devDep) — `pnpm lighthouse` / `:desktop` / `:ci` |

> Larger apps split shared web components into per-module-export packages (e.g. a `core` / `ui` / `map` workspace) so consumers tree-shake to just the icons/components they import.

```bash
# Confirm the preferred stack is present and versions are pinned
grep -nE '"(lit|@shoelace-style/shoelace|lucide|lucide-static|@greycat/web|vite|vitest|typescript|lighthouse)"' package.json
grep -nE '"\^|"~' package.json && echo "⚠ non-exact version range — pin the exact latest"
```

---

## Checklist

Scan `.ts` under `frontend/` (source) for:

### 1. Initialization order
- `import '@greycat/web'` before any `gui-*` or client usage
- `gc.sdk.init()` completes before creating `gui-*` elements or `gc.project.*` calls
- Shoelace theme CSS imported at startup (`@shoelace-style/shoelace/dist/themes/light.css` + `dark.css`); app `styles.css` imported **after** so its bridge tokens win
- Shoelace asset base path set once at startup

### 2. TypeScript quality
- No `any` — use `gc.core.*` / `gc.project.*` types
- `project.d.ts` not manually edited
- Custom components declare `HTMLElementTagNameMap` (and `GreyCat.JSX.IntrinsicElements` if JSX is used)

### 2b. `gc` namespace shadow (CRITICAL)

`@greycat/web` exposes `gc`. **It's shadowed by `@types/node`'s `var gc`** (global GC trigger) at type-resolution time. Direct `gc.X.Y` resolves against the Node global — silent breakage.

**Required**: re-export pattern.
```ts
// frontend/gc-runtime.ts
import * as gcRuntime from '@greycat/web';
export { gcRuntime };

// elsewhere
import { gcRuntime } from '~/gc-runtime';
gcRuntime.sdk.init({ url: '/' });
gcRuntime.project.MyType.create(/* ... */);
```
```bash
grep -rnE '\bgc\.' frontend/ --include="*.ts" | grep -v "frontend/gc-runtime.ts"
```

### 2c. Codegen freshness
Backend type / `@expose` changes require `greycat codegen ts` (script: `pnpm gen`). Frontend type checker lies otherwise.

### 2d. Derive backend strings from `$fields`, never hard-code
```ts
// ❌ const key = "MyEnum.RED";
// ✅ const key = gcRuntime.project.MyEnum.$fields.red;
```

### 3. Component patterns (Lit)
- One `LitElement` per file, `@customElement('app-…')` with a consistent kebab prefix
- `@property()` for public inputs, `@state()` for internal state
- `static styles = css\`…\`` for styles; `html\`…\`` for templates — don't recreate DOM, update reactive props
- Charts: instantiate in `firstUpdated`, **destroy in `disconnectedCallback`** (chart.js leaks otherwise)
- `@greycat/web` `GuiElement` / `GuiValueElement`: `static override styles` + `css()`; call `this._internalUpdate()` in value setters
- Events: `GuiChangeEvent` / `GuiInputEvent` (bubble + composed)
- Cleanup: `addDisposable()` / `abortSignal()` (or Lit `disconnectedCallback`), never manual `removeEventListener`

### 3b. Shoelace usage
- Import components **individually** for tree-shaking (`import '@shoelace-style/shoelace/dist/components/button/button.js'`), never the whole bundle
- Use Shoelace for layout/cards/tabs/dialogs/date-picker; reserve custom Lit components for data-viz widgets
```bash
grep -rn "from '@shoelace-style/shoelace'" frontend/ --include="*.ts" && echo "⚠ whole-bundle import — switch to per-component imports"
```

### 3c. Icons (Lucide — `lucide` or `lucide-static`)
- Use the framework-agnostic **`lucide`** (import only the glyphs you use) or **`lucide-static`** (SVG strings inlined via Lit `unsafeSVG`). Self-hosted/bundled — **no runtime icon fetch**
- Render with `fill="none" stroke="currentColor"` + `aria-hidden="true"`; decorative icons must be `aria-hidden`, meaningful ones need an accessible label
```bash
grep -rnE "cdn|unpkg|googleapis\.com/.*icon" frontend/ --include="*.ts" --include="*.html" && echo "⚠ runtime/CDN icon fetch — use lucide / lucide-static instead"
```

### 4. Common pitfalls
- `className` in JSX (if used): each token a single class name (no spaces in strings)
- No `dangerouslySetInnerHTML` / raw `innerHTML` — use DOM API, Lit templates, or vetted `unsafeSVG` for trusted icon data only
- MPA: no client-side routing — use `<a href>` / `window.location`. SPA: keep crawlable content discoverable (see SEO below)
- Don't `emptyOutDir` the whole `webroot/` — it holds `webroot/explorer/` from `greycat install` (`emptyOutDir: false`)

### 5. Performance (feeds Lighthouse)
- Update reactive properties instead of recreating DOM trees; batch imperative attribute sets
- Use `gui-table` (virtualized) for large datasets, not manual DOM loops
- Code-split routes / lazy-load heavy widgets (charts) with dynamic `import()`
- Tree-shake Shoelace (per-component imports); self-host icons via `lucide-static` (no CDN)
- Defer non-critical JS, inline critical CSS, preconnect to the API origin, long-cache hashed assets, reserve element sizes to avoid layout shift (CLS)

### 6. Lighthouse audit (performance · SEO · accessibility · best-practices)
Target **≥ 90** in every category. Serve first (`greycat serve`), then:
```bash
pnpm lighthouse          # full audit (html + json), opens report
pnpm lighthouse:desktop  # desktop preset
pnpm lighthouse:ci       # json, --only-categories=performance,accessibility,best-practices,seo (CI gate)
```
Flag a missing audit setup:
```bash
grep -q '"lighthouse"' package.json || echo "⚠ no lighthouse script — add pnpm lighthouse / :desktop / :ci"
```

### 7. SEO + LLM-friendly SEO (always)
Check `frontend/index.html` and the built `webroot/`:
- **Head**: `<html lang>`, unique `<title>`, `<meta name="description">`, canonical `<link rel="canonical">`, Open Graph + Twitter Card tags, `theme-color`, `<meta name="color-scheme">`, `viewport`
- **i18n** (if multi-locale via i18next): emit `<link rel="alternate" hreflang="…">` per locale, keep `<html lang>` in sync with the active locale, and translate `<title>`/description per route
- **Semantics**: landmark elements (`header`/`nav`/`main`/`article`/`footer`), correct heading order, `alt` text, ARIA labels. Lit Shadow DOM can hide text from some crawlers — keep primary content in light DOM or **pre-render/SSR the shell**
- **Structured data**: JSON-LD (`schema.org`) in `<head>`
- **Machine-readable for crawlers AND LLMs**: `robots.txt`, `sitemap.xml`, a web app manifest, and **`llms.txt`** (+ optional `llms-full.txt`) at the web root — a concise Markdown index of the app's purpose, key routes, and public endpoints so LLM agents can navigate it
- Descriptive link text, stable URLs, per-route `<title>`/description
```bash
grep -iqE '<meta name="description"' frontend/index.html || echo "⚠ no meta description"
grep -iqE 'og:title|twitter:card'      frontend/index.html || echo "⚠ no Open Graph / Twitter Card tags"
grep -iqE 'application/ld\+json'        frontend/index.html || echo "⚠ no JSON-LD structured data"
for f in robots.txt sitemap.xml llms.txt site.webmanifest manifest.webmanifest; do
  [ -f "webroot/$f" ] || [ -f "frontend/public/$f" ] || echo "⚠ missing $f (SEO/LLM discoverability)"
done
```

---

## Output

Group by severity:
- **CRITICAL**: missing `gc.sdk.init()`, missing Shoelace theme CSS, security (raw `innerHTML`), an off-stack UI framework when Lit + Shoelace is the standard
- **HIGH**: bad component registration, broken event typing, whole-bundle Shoelace import, runtime/CDN icon fetch, Lighthouse category < 90, missing meta description / structured data
- **MEDIUM**: `any` usage, missing cleanup, unbatched updates, missing `llms.txt` / sitemap / manifest
- **LOW**: style inconsistency, missing type casts, non-pinned versions
