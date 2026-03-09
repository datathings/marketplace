# Frontend Integration

@greycat/web — TypeScript SDK with JSX runtime for real DOM elements. Not on npmjs.org.

## Contents
- Project Structure
- Setup (package.json, vite.config.ts, tsconfig.json)
- Using with Other Frameworks (Vue, React, Svelte, etc.)
- SDK Init
- JSX Rules
- Web Component Pattern
- API Calls
- Enums
- Authentication
- Common Pitfalls

**Flow**: `GCL → greycat codegen ts → project.d.ts → TypeScript → gc namespace`

## Project Structure

Multi-page app (MPA) — each page has its own `index.html` with `gc.sdk.init()`. The vite plugin auto-discovers `.html` files under `app/`.

```
my-project/
├── project.gcl
├── src/<feature>/
├── app/
│   ├── vite-env.d.ts
│   ├── index.html
│   ├── index.tsx
│   ├── <page>/
│   │   ├── index.html
│   │   └── index.tsx
│   └── components/<comp>/
│       ├── <comp>.tsx
│       └── <comp>.css
├── vite.config.ts
├── tsconfig.json
├── package.json
└── project.d.ts                   # auto-generated
```

## Setup

```bash
pnpm add @greycat/web@https://get.greycat.io/files/sdk/web/dev/7.7/7.7.3-dev.tgz
```

**package.json:**
```json
{
  "scripts": {
    "gen": "greycat codegen ts",
    "dev": "pnpm gen && vite",
    "build": "pnpm gen && vite build"
  }
}
```

**vite.config.ts:**
```typescript
import { defineConfig } from 'vite';
import { greycat } from '@greycat/web/vite-plugin';

export default defineConfig({
  plugins: [greycat()],
});
```

**tsconfig.json:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "skipLibCheck": true,
    "jsx": "react-jsx",
    "jsxImportSource": "@greycat/web",
    "sourceMap": true,
    "noEmit": true,
    "paths": {
      "~/*": ["app/*"]
    }
  },
  "include": ["app", "project.d.ts"]
}
```

- `skipLibCheck: true` — required, `@greycat/web` bundles d3/shoelace types with internal conflicts
- `jsx` + `jsxImportSource` — required for `.tsx` files

**app/vite-env.d.ts** — ambient Vite module declarations (no `import` statements, or it becomes a module and loses ambient scope):
```typescript
declare module '*.css?raw' {
  const content: string;
  export default content;
}
```

## Using with Other Frameworks (Vue, React, Svelte, etc.)

The `greycat()` vite plugin works alongside any framework's vite plugin. Two requirements:

1. **`noDefaultConfig: true`** — prevents the greycat plugin from overriding the framework's rollup output config (entry/chunk/asset file names, MPA input discovery)
2. **`root: 'app'`** — always use `app/`, not `frontend/` or `src/`

The greycat plugin still provides:
- Dev server proxy (forwards `{module}::{function}` calls to GreyCat backend)
- `gzip: true` option for production compression

**vite.config.ts** — use this exact template, replacing `frameworkPlugin` with your framework's plugin:
```typescript
import { defineConfig } from 'vite';
import { greycat } from '@greycat/web/vite-plugin';
import frameworkPlugin from '<framework-vite-plugin>';

export default defineConfig({
  root: 'app',
  plugins: [greycat({ noDefaultConfig: true, gzip: true }), frameworkPlugin()],
  build: {
    outDir: '../webroot',
    emptyOutDir: true,
  },
});
```

| Framework | Plugin import | Plugin call |
|-----------|--------------|-------------|
| Vue | `import vue from '@vitejs/plugin-vue'` | `vue()` |
| React | `import react from '@vitejs/plugin-react'` | `react()` |
| Svelte | `import { svelte } from '@sveltejs/vite-plugin-svelte'` | `svelte()` |
| Solid | `import solid from 'vite-plugin-solid'` | `solid()` |

**Vite version**: use Vite 7+ (Rolldown-based) by default. Vite 7 replaces esbuild+Rollup with Rolldown (Rust-based unified bundler) for faster builds. Install with `pnpm add -D vite@latest`.

**tsconfig.json** — adapt `jsx` and `include` for the framework:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "skipLibCheck": true,
    "noEmit": true,
    "paths": {
      "@/*": ["app/*"]
    }
  },
  "include": ["app"]
}
```

## SDK Init

Each page must call `gc.sdk.init()` before rendering:

```typescript
import '@greycat/web';

await gc.sdk.init();
```

## JSX Rules

@greycat/web JSX creates real DOM elements (no virtual DOM).

| Rule | Example |
|------|---------|
| CSS classes | `className="foo"` (not `class`) |
| Conditional classes | `className={{ active: isActive, hidden: !visible }}` |
| Event handlers | `onclick`, `onchange`, `onkeydown` (lowercase) |
| Event target cast | `(e.target as HTMLInputElement).value` |
| Data attributes | `data-id={String(id)}` |
| Fragments | `<>...</>` |
| Conditionals | `{cond ? <A /> : <B />}` |
| Lists | `{items.map(i => <li>{i.name}</li>)}` |
| Ref callback | `$ref={(el) => this._input = el}` |

**`className` uses `classList.add()` internally — each token must be a single class name, no spaces:**
- **string**: `className="foo"` — single class only
- **object**: `className={{ foo: true, bar: cond }}` — truthy keys added, falsy keys removed
- **array**: `className={["foo", "bar"]}` — multiple static classes

## Web Component Pattern

Each component file declares its own ambient types at the bottom:

```tsx
import { registerCustomElement } from '@greycat/web';

export class MyComp extends HTMLElement {} // classic WebComponent

declare global {
  interface HTMLElementTagNameMap {
    'my-comp': MyComp;
  }

  namespace GreyCat {
    namespace JSX {
      interface IntrinsicElements {
        'my-comp': GreyCat.Element<MyComp>;
      }
    }
  }
}

registerCustomElement('my-comp', MyComp);
```

**Registration** in the page entry:
```tsx
import '@greycat/web';

await gc.sdk.init();

await import('./components/my-comp'); // import after init to have access to global `gc` namespace statically
document.body.appendChild(<my-comp />);
```

- JSX namespace is `GreyCat.JSX`, not the global `JSX`
- Register in both `HTMLElementTagNameMap` and `GreyCat.JSX.IntrinsicElements`
- Each component owns its `declare global` block

## API Calls

All calls use POST. Codegen produces typed functions:

```typescript
const todos = await gc.todo_api.getTodos();
await gc.todo_api.updateTodo(id, new gc.todo_api.TodoUpdate("New"));
```

```bash
# curl equivalent
curl -X POST http://localhost:8080/todo_api::getTodos -H "Content-Type: application/json" -d '[]'
```

## Enums

Use `.key!` to get the string value:

```typescript
const role = user.role.key!; // "Admin"
```

## Authentication

```typescript
await gc.sdk.login({ username, password, use_cookie: true });
gc.sdk.token !== null; // check auth
gc.sdk.logout();
```

## Common Pitfalls

| Wrong | Correct |
|-------|---------|
| `class="foo"` in JSX | `className="foo"` |
| Global `JSX.IntrinsicElements` | `GreyCat.JSX.IntrinsicElements` in `declare global` |
| Missing `skipLibCheck` | Always `skipLibCheck: true` |
| CSS `?raw` decl in file with `import` | Separate ambient `vite-env.d.ts` (no imports) |
| Shared `env.d.ts` for all components | Each component declares its own `declare global` |
| `npm install @greycat/web` | Full tgz URL from get.greycat.io |
| Render before `gc.sdk.init()` | Always `await gc.sdk.init()` first |
| Forgot `greycat codegen ts` | Run after every backend change |
| `root: 'frontend'` or `root: 'src'` | Always `root: 'app'` |
| `greycat()` with Vue/React/etc. | `greycat({ noDefaultConfig: true, gzip: true })` |
| Manual vite proxy config | `greycat()` plugin handles proxy automatically |
| `/api/getTodos` fetch path | `POST /{module}::{function}` (e.g., `/todo_api::getTodos`) |
| Vite 6 or older (esbuild+Rollup) | Vite 7+ (Rolldown) — `pnpm add -D vite@7.3.1` |
