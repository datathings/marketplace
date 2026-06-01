# Adding a webapp to a GreyCat project

The idiomatic way to ship a single-page web frontend bundled with a GreyCat backend: webapp sources under `app/`, Vite or VitePlus config at the project root, build output into `webroot/`, served by `greycat serve` / `greycat dev`. No second process, no separate web server.

This file documents the recommended layout and the wiring. The actual frontend stack (React, Vue, Svelte, vanilla) is up to you — only the bundler integration is opinionated.

## Contents

- The pattern at a glance
- Why this layout works
- Bootstrap
- Running with `greycat dev`
- Calling `@expose`d functions from the browser
- Optional: the typed TS SDK
- Production build
- Common pitfalls

## The pattern at a glance

```
my-project/
├── project.gcl              # GreyCat entrypoint
├── package.json             # Node / Vite / VitePlus deps
├── vite.config.ts           # builds `app/` → `webroot/`
├── tsconfig.json
├── app/                     # webapp source (NOT a GreyCat directory)
│   ├── index.html
│   └── src/
│       └── main.ts
├── src/                     # GreyCat source — @include("src");
│   └── api.gcl
├── webroot/                 # bundle target, served at /
├── files/
├── lib/                     # gitignored, from `greycat install`
├── bin/                     # gitignored, from `greycat install`
└── gcdata/                  # gitignored, runtime storage
```

Two side-by-side worlds at the project root:

- **GreyCat side** — `project.gcl`, `src/`, `lib/`, `gcdata/`, `bin/`, `files/`, `webroot/`.
- **Node side** — `package.json`, `vite.config.ts`, `tsconfig.json`, `app/`.

The Node side builds into `webroot/`. GreyCat serves `webroot/` at `/`.

## Why this layout works

- **`webroot/` is the public static root.** Anything in `webroot/` is reachable at the corresponding URL path on the running server.
- **Unknown paths return 404 — there is no automatic SPA fallback.** If your router needs deep-link support (`/dashboard`, `/users/42`), either use hash routing (`#/dashboard`) or place hashed-bundle assets and a server-side handler for those routes.
- **`greycat dev` auto-detects the bundler.** If a `vite.config.{js,ts}` or `vp.config.{js,ts}` exists in the cwd, `greycat dev` spawns the watcher (preferring `vp build --watch`, then `pnpm vite build --watch`, then `npx vite build --watch`) and runs the server in the same process tree. If the watcher exits non-zero, the server stops.

## Bootstrap

A minimum-viable setup. Pick **one** of the two toolchains below.

### Option A — VitePlus (recommended)

[VitePlus](https://viteplus.dev/) is the unified Vite-team toolchain. `greycat dev` looks for `vp build --watch` first, so this is the lowest-friction option.

```jsonc
// package.json
{
  "name": "my-project-app",
  "private": true,
  "type": "module",
  "scripts": {
    "build": "vp build",
    "dev": "vp build --watch"
  },
  "devDependencies": {
    "viteplus": "<latest>"
  }
}
```

### Option B — Plain Vite

```jsonc
// package.json
{
  "name": "my-project-app",
  "private": true,
  "type": "module",
  "scripts": {
    "build": "vite build",
    "dev": "vite build --watch"
  },
  "devDependencies": {
    "vite": "<latest>"
  }
}
```

### Shared: `vite.config.ts` and `app/`

```ts
// vite.config.ts
import { defineConfig } from "vite";

export default defineConfig({
  root: "app",
  build: {
    outDir: "../webroot",
    emptyOutDir: true,
  },
});
```

The `root: "app"` line tells Vite to treat `app/` as the source root (its `index.html` is the entry). `outDir: "../webroot"` is resolved relative to `root`, so the bundle lands in `<project>/webroot/`. `emptyOutDir: true` clears stale files before each build.

```html
<!-- app/index.html -->
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>My App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

```ts
// app/src/main.ts
async function ping() {
  const res = await fetch("/api::ping", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "[]",
  });
  console.log(await res.json());
}
ping();
```

Install deps, then:

```sh
npm install
greycat install        # one-time: download lib/std and bin/greycat
greycat dev            # build watcher + serve in one process
```

For VitePlus specifics (multiple entries, plugin config, etc.), see https://viteplus.dev/.

## Running with `greycat dev`

```sh
greycat dev
```

On startup, `dev` looks at the cwd for one of `vite.config.{js,ts}` or `vp.config.{js,ts}`. If present, it spawns the watcher and the GreyCat HTTP server side-by-side. The watcher's stdout is prefixed with `dev:` in the server log.

If the watcher exits non-zero, the server stops. To use a different watcher (e.g., a monorepo's custom build script):

```sh
greycat dev --with="pnpm --filter myapp build --watch"
```

`--with` skips auto-detection entirely and spawns exactly what you pass.

`greycat dev` accepts every option `greycat serve` accepts — `--port`, `--log`, `--user`, etc. See [cli.md](cli.md).

## Calling `@expose`d functions from the browser

The primary path is **plain `fetch` against the GreyCat HTTP endpoints**. No SDK required.

```gcl
// src/api.gcl
@expose
fn ping(): String {
    return "pong";
}

@expose
fn add(a: int, b: int): int {
    return a + b;
}
```

Both endpoints require the default `api` permission. The browser obtains a session via `Identity::login` (see [Authentication](#authentication) below) before calling them. **Do not** sprinkle `@permission("public")` on endpoints to skip login during development — that opens them to every anonymous caller on the network. `@permission("public")` is reserved for endpoints that genuinely must serve anonymous traffic (`login` itself, a no-auth health probe), and should be a deliberate user-driven decision, not an agent shortcut.

### Path-RPC (one function per request)

```ts
const res = await fetch("/api::add", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify([10, 32]),       // positional args, in declaration order
});
const result = await res.json();        // 42
```

### JSON-RPC 2.0

```ts
const res = await fetch("/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    jsonrpc: "2.0",
    method: "api::add",                 // module::function (double-colon)
    params: [10, 32],
    id: 1,
  }),
});
const { result } = await res.json();    // 42
```

### Authentication

Authentication is **the default path, not an optional add-on**. A request without a session token can only reach functions explicitly marked `@permission("public")`; every other `@expose` returns 401. Two ways the browser carries the token:

1. **Cookie (typical).** `Identity::login` sets `Set-Cookie: greycat=<TOKEN>` on the response. Subsequent same-origin `fetch` calls send the cookie automatically.
2. **Query parameter.** `?authorization=<TOKEN>` — handy for the boot URL printed by `serve` on first start.

```ts
// app/src/auth.ts
export async function login(username: string, password: string) {
  const res = await fetch("/runtime::Identity::login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([username, password]),
    credentials: "same-origin",          // accept Set-Cookie
  });
  if (!res.ok) throw new Error("login failed");
  // cookie is now set; subsequent fetches authenticate automatically
}
```

Errors surface as standard HTTP statuses: 400 (bad params), 401 (missing auth on non-public endpoint), 404 (no such function), 422 (ABI mismatch — client built against a stale schema), 500 (runtime exception).

## Optional: the typed TS SDK

`greycat codegen ts` reads the compiled program and emits typed bindings for every public type and `@expose`d function. **This is optional.** Many projects ship just plain `fetch` and JSON-RPC.

```sh
greycat codegen ts
```

Codegen auto-detects the TypeScript target when it sees `package.json` / `tsconfig.json` / `jsconfig.json` at the project root. See [cli.md § codegen](cli.md) for the dispatcher. The package name and import paths surface in the codegen output — read what it generates rather than hard-coding an SDK name in your application.

## Production build

`greycat build` produces a `project.gcp` (the compiled GreyCat package). It does **not** build the frontend. Run both:

```sh
npm run build          # populates webroot/
greycat build          # produces project.gcp
```

Deploy artifacts:

- `project.gcp` (or the source tree with `lib/` populated)
- `bin/greycat` (the pinned core binary, from `greycat install`)
- `webroot/` (the built bundle)
- `project.gcl`, `src/` (if shipping source rather than `.gcp`)
- a writable `gcdata/` and `files/` on the host

Suggested `.gitignore` entries for the frontend bits:

```
node_modules/
webroot/
```

`webroot/` is a build artifact — let the deploy pipeline regenerate it. Keep `app/`, `vite.config.ts`, `package.json`, and `package-lock.json` checked in.

## Common pitfalls

1. **Wrong `outDir`.** Vite defaults to `dist/`. You must override it. With `root: "app"`, set `outDir: "../webroot"` (relative to `root`). Without `root`, set `outDir: "webroot"`. Verify by running the build and checking that `webroot/index.html` exists and is the bundled one.
2. **`webroot/` checked into git but also gitignored.** Pick one. If you gitignore it, the deploy must run the frontend build. If you check it in, your repo will churn on every commit.
3. **HTML5 history routing without a fallback handler.** The GreyCat runtime does **not** automatically serve `index.html` for unknown paths — deep links like `/dashboard` return 404. Use hash routing (`#/dashboard`) for the simplest setup, or implement a fallback yourself.
4. **`app/` mistakenly included as a GreyCat directory.** `@include("app")` would try to load `app/*` as `.gcl` modules and fail. `app/` is a sibling of `src/`, not part of the GreyCat project graph. Do not list it in any pragma.
5. **`vp` not found.** `greycat dev` only spawns `vp build --watch` if a `vp` binary is on `PATH` (typically via the VitePlus npm dep + a runner like `pnpm vp` or local `node_modules/.bin/vp`). If you don't have it, `greycat dev` falls back to `pnpm vite build --watch` and then `npx vite build --watch` — whichever resolves first.
6. **Same-origin assumption for cookies.** The session cookie is set with default same-origin scope. Hosting the frontend and the GreyCat API on different origins requires CORS configuration plus `credentials: "include"` and matching `SameSite` / `Secure` attributes — the simpler model is to serve the bundle from the same GreyCat process via `webroot/`.
