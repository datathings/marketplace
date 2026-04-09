---
name: frontend
description: Review frontend codebase for code quality, performance, and best practices using @greycat/web patterns
allowed-tools: Bash, Read, Grep, Glob
---

# Frontend Review

**Purpose**: Review `@greycat/web` frontend code for correctness, best practices, and common pitfalls.

Read [references/frontend.md](../skills/greycat/references/frontend.md) first to understand the @greycat/web patterns.

---

## Checklist

Scan all `.ts` and `.tsx` files under `app/` for these issues:

### 1. Initialization Order
- `import '@greycat/web'` must appear before any `gui-*` or `gc.*` usage
- `gc.sdk.init()` must complete before creating `gui-*` elements or calling `gc.project.*`
- CSS import (`@greycat/web/greycat.css` or `greycat-full.css`) must be present

### 2. TypeScript Quality
- Search for `any` type usage — replace with proper GreyCat types from `gc.core.*` or `gc.project.*`
- Verify `project.d.ts` is not manually edited
- Check that custom components declare both `HTMLElementTagNameMap` and `GreyCat.JSX.IntrinsicElements`

### 3. Component Patterns
- Components extending `GuiElement` should use `static override styles` with `css()` + `?inline` imports
- Components extending `GuiValueElement` should call `this._internalUpdate()` in value setters
- Event dispatching should use `GuiChangeEvent` / `GuiInputEvent` (bubble + composed)
- Cleanup should use `addDisposable()` or `abortSignal()`, not manual `removeEventListener`

### 4. Common Pitfalls
- `className` in JSX: each token must be a single class name (no spaces in strings)
- `GreyCat.JSX.IntrinsicElements` (not global `JSX.IntrinsicElements`)
- No `dangerouslySetInnerHTML` or `innerHTML` — use DOM API or JSX children
- Fragments (`<>...</>`) empty on append — don't store for reuse
- MPA: no client-side routing — verify navigation uses `<a href>` or `window.location`

### 5. Performance
- Look for components re-creating DOM trees on every update instead of updating properties
- Check for missing `setAttrs()` batching when setting multiple properties imperatively
- Verify large data sets use `gui-table` (virtualized) rather than manual DOM loops

---

## Output

Report issues grouped by severity:

- **CRITICAL**: Missing `gc.sdk.init()`, missing CSS imports, security issues
- **HIGH**: Incorrect component registration, broken event typing
- **MEDIUM**: `any` usage, missing cleanup, unbatched updates
- **LOW**: Style inconsistencies, missing type casts
