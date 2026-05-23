---
name: frontend
description: Review frontend codebase for code quality, performance, and best practices using @greycat/web patterns
allowed-tools: Bash, Read, Grep, Glob
---

# Frontend Review

**Purpose**: Review `@greycat/web` frontend code for correctness, best practices, and common pitfalls.

Read [reference/webapp.md](../skills/greycat/reference/webapp.md) for `@greycat/web` bundling/integration context.

---

## Checklist

Scan `.ts` / `.tsx` under `app/` for:

### 1. Initialization order
- `import '@greycat/web'` before any `gui-*` or `gc.*` usage
- `gc.sdk.init()` completes before creating `gui-*` elements or `gc.project.*` calls
- CSS import present (`@greycat/web/greycat.css` or `greycat-full.css`)

### 2. TypeScript quality
- No `any` — use `gc.core.*` / `gc.project.*` types
- `project.d.ts` not manually edited
- Custom components declare both `HTMLElementTagNameMap` and `GreyCat.JSX.IntrinsicElements`

### 2b. `gc` namespace shadow (CRITICAL)

`@greycat/web` exposes `gc`. **It's shadowed by `@types/node`'s `var gc`** (global GC trigger) at type-resolution time. Direct `gc.X.Y` resolves against the Node global — silent breakage.

**Required**: re-export pattern.
\`\`\`ts
// app/gc-runtime.ts
import * as gcRuntime from '@greycat/web';
export { gcRuntime };

// elsewhere
import { gcRuntime } from '~/gc-runtime';
gcRuntime.sdk.init({ url: '/' });
gcRuntime.project.MyType.create(/* ... */);
\`\`\`
\`\`\`bash
grep -rnE '\bgc\.' app/ --include="*.ts" --include="*.tsx" | grep -v "app/gc-runtime.ts"
\`\`\`

### 2c. Codegen freshness
Backend type / `@expose` changes require `./bin/greycat codegen ts`. Frontend type checker lies otherwise.

### 2d. Derive backend strings from `$fields`, never hard-code
\`\`\`ts
// ❌ const key = "MyEnum.RED";
// ✅ const key = gcRuntime.project.MyEnum.$fields.red;
\`\`\`

### 3. Component patterns
- `GuiElement`: `static override styles` + `css()` + `?inline` imports
- `GuiValueElement`: call `this._internalUpdate()` in value setters
- Events: `GuiChangeEvent` / `GuiInputEvent` (bubble + composed)
- Cleanup: `addDisposable()` or `abortSignal()`, never manual `removeEventListener`

### 4. Common pitfalls
- `className` in JSX: each token a single class name (no spaces in strings)
- `GreyCat.JSX.IntrinsicElements`, not global `JSX.IntrinsicElements`
- No `dangerouslySetInnerHTML` / `innerHTML` — use DOM API or JSX children
- Fragments empty on append — don't reuse
- MPA: no client-side routing — use `<a href>` / `window.location`

### 5. Performance
- Don't recreate DOM trees on every update — update properties
- Batch with `setAttrs()` when setting multiple properties imperatively
- Use `gui-table` (virtualized) for large datasets, not manual DOM loops

---

## Output

Group by severity:
- **CRITICAL**: missing `gc.sdk.init()`, missing CSS, security
- **HIGH**: bad component registration, broken event typing
- **MEDIUM**: `any` usage, missing cleanup, unbatched updates
- **LOW**: style inconsistency, missing type casts
