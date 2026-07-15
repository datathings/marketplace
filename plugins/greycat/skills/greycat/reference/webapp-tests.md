# UI testing a GreyCat webapp with Playwright

End-to-end tests for the prescribed webapp stack (see [webapp.md](webapp.md)). Playwright drives the built
webapp in a real browser over the same origin `greycat serve` uses - the end-to-end check that the Lit + Web
Awesome + SDK stack renders and the backend answers. Unit-test backend logic with `greycat test`; reach for
Playwright only for browser behavior.

## Contents

- Setup checklist
- `playwright.config.ts`
- Authenticate once, reuse the session
- Selecting through the shadow DOM
- Keep specs independent of persistent state
- Run
- `.gitignore`

## Setup checklist

```
- [ ] Add @playwright/test to package.json; install with `vp install`
- [ ] Install the browser: `pnpm exec playwright install chromium`
- [ ] Create a login user (auth is the default) while no server holds the gcdata lock
- [ ] Add playwright.config.ts (webServer + a `setup` project for auth)
- [ ] Write e2e/auth.setup.ts (log in once, save storageState)
- [ ] Write specs that self-isolate their data
- [ ] Run: `pnpm run test:e2e`
```

**Install through `vp`, not npm.** Add the dependency, then let `vp install` resolve it with the project's
pinned pnpm. `pnpm add`/`pnpm install` run directly relink the store across pnpm majors and error; `npx
playwright` is blocked by the `devEngines` pin. Run every Playwright command through `pnpm exec` / `pnpm run`.

```jsonc
// package.json
{
  "scripts": { "build": "vp build", "test:e2e": "playwright test" },
  "devDependencies": { "@playwright/test": "^1", "vite-plus": "^0.2" }
}
```

```sh
vp install                              # installs @playwright/test with the pinned pnpm
pnpm exec playwright install chromium   # download the browser binary
```

**A real login user is required.** A bare `@expose` needs the `api` permission, so tests authenticate as a
user. Create one when no server holds the `gcdata/` lock (before `greycat dev` starts):

```sh
greycat run runtime::Identity::create e2e user
greycat run runtime::Identity::set_password e2e <password>
```

## `playwright.config.ts`

`greycat dev` builds `app/` into `webroot/` and serves the API on one origin, so the browser reaches pages and
backend calls at the same `baseURL`. A `setup` project authenticates once; the `chromium` project reuses the
saved session.

```ts
import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.E2E_PORT ?? 8080);
const BASE_URL = `http://localhost:${PORT}`;

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false, // tests share one persistent gcdata store
  use: { baseURL: BASE_URL, trace: 'on-first-retry' },
  projects: [
    { name: 'setup', testMatch: /auth\.setup\.ts/ },
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], storageState: 'e2e/.auth/user.json' },
      dependencies: ['setup'],
    },
  ],
  webServer: {
    command: 'greycat dev',
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
```

## Authenticate once, reuse the session

The page's root element gates on `gc.sdk.init()`, so tests must have a session before any typed call works.
Log in through the UI in a `setup` spec and save `storageState`; the SDK persists the session and
`storageState` captures it. Every other spec starts authenticated.

```ts
// e2e/auth.setup.ts
import { test as setup, expect } from '@playwright/test';

const AUTH_FILE = 'e2e/.auth/user.json';
const USER = process.env.E2E_USER ?? 'e2e';
const PASSWORD = process.env.E2E_PASSWORD ?? 'e2e-password';

setup('authenticate', async ({ page }) => {
  await page.goto('/');
  await page.locator('wa-input[name="username"] input').fill(USER);
  await page.locator('wa-input[name="password"] input').fill(PASSWORD);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.locator('wa-input[name="title"]')).toBeVisible({ timeout: 10_000 });
  await page.context().storageState({ path: AUTH_FILE });
});
```

## Selecting through the shadow DOM

Playwright pierces open shadow DOM, so the Lit roots and Web Awesome elements are transparent to its locators:

- `getByRole('button', { name: 'Sign in' })` and `getByLabel('Username')` resolve Web Awesome's rendered
  labels and roles.
- `.fill()` needs the native control inside a `wa-input`; target it as `wa-input[name="title"] input`.
- Scope a repeated item by its text: `page.locator('li', { hasText: title })`, then query the button or
  checkbox inside that row.

## Keep specs independent of persistent state

`gcdata/` persists across runs and `--store` is only a per-zone size cap, not a throwaway-DB path - there is no
per-test database. A spec that asserts on list contents must scope to data it created: use a unique label per
run, then add -> assert -> delete. Keep `fullyParallel: false` unless every assertion is text-scoped.

```ts
// e2e/todos.spec.ts
import { test, expect, type Page } from '@playwright/test';

function uniqueTitle(): string {
  return `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
}
function row(page: Page, title: string) {
  return page.locator('li', { hasText: title });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('wa-input[name="title"]')).toBeVisible({ timeout: 10_000 });
});

test('add, complete, and delete a todo', async ({ page }) => {
  const title = uniqueTitle();
  await page.locator('wa-input[name="title"] input').fill(title);
  await page.getByRole('button', { name: 'Add' }).click();
  await expect(row(page, title)).toBeVisible();

  await row(page, title).locator('wa-checkbox').click();
  await expect(row(page, title)).toHaveClass(/done/);

  await row(page, title).getByRole('button', { name: 'Delete' }).click();
  await expect(row(page, title)).toHaveCount(0);
});
```

## Run

```sh
pnpm run test:e2e            # setup logs in, then the chromium project runs the specs
pnpm exec playwright test todos.spec.ts   # a single file
pnpm exec playwright test --ui            # headed, interactive
```

A failing spec that pins a wrong event or selector is the feedback loop: fix the app or the selector, rerun,
repeat until green. A silently-dropped Web Awesome event handler (see the event-naming pitfall in
[webapp.md](webapp.md)) surfaces here as a click that changes nothing, even though `vp build` and `greycat
build` both pass.

## `.gitignore`

Add the Playwright artifacts to the project `.gitignore`:

```
test-results/
playwright-report/
e2e/.auth/
```
