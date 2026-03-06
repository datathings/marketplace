# Frontend Integration

@greycat/web — TypeScript SDK with JSX runtime for real DOM elements. Not on npmjs.org.

**Flow**: `GCL → greycat codegen ts → project.d.ts → TypeScript → gc namespace`

## Project Structure

Multi-page app (MPA) — each page has its own `index.html` with `gc.sdk.init()`. The vite plugin auto-discovers `.html` files under `app/`.

```
my-project/
├── project.gcl
├── src/<feature>/
├── app/
│   ├── vite-env.d.ts              # Vite ambient declarations
│   ├── index.html
│   ├── index.tsx
│   ├── <page>/
│   │   ├── index.html
│   │   └── index.tsx
│   └── components/<comp>/
│       ├── <comp>.tsx             # Web component + ambient declarations
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
    "noEmit": true
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
import style from './my-comp.css?raw';

export class MyComp extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this._render();
  }

  private _render() {
    this.shadowRoot!.innerHTML = '';
    this.shadowRoot!.appendChild(
      <>
        <style>{style}</style>
        <div className="container">content</div>
      </>,
    );
  }
}

// Ambient type declarations — colocated with the component
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
```

**Registration** in the page entry:
```tsx
import '@greycat/web';
import { MyComp } from './components/my-comp/my-comp';

await gc.sdk.init();

customElements.define('my-comp', MyComp);
document.body.appendChild(<my-comp />);
```

**Key points:**
- JSX namespace is `GreyCat.JSX`, not the global `JSX`
- Register in both `HTMLElementTagNameMap` and `GreyCat.JSX.IntrinsicElements`
- Each component owns its `declare global` block — no shared declaration file needed

## API Calls

All calls use POST. Codegen produces typed functions:

```typescript
// Simple
const todos = await gc.todo_api.getTodos();

// With complex params — use Type.createFrom()
await gc.todo_api.updateTodo(id, gc.todo_api.TodoUpdate.createFrom({ title: "New" }));
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
| `class="foo"` in JSX | `className="foo"` or `className={{ x: cond }}` |
| Global `JSX.IntrinsicElements` | `GreyCat.JSX.IntrinsicElements` in `declare global` |
| Missing `skipLibCheck` | Always `skipLibCheck: true` |
| CSS `?raw` decl in file with `import` | Separate ambient `vite-env.d.ts` (no imports) |
| Shared `env.d.ts` for all components | Each component declares its own `declare global` |
| `npm install @greycat/web` | Full tgz URL from get.greycat.io |
| Render before `gc.sdk.init()` | Always `await gc.sdk.init()` first |
| Forgot `greycat codegen ts` | Run after every backend change |
