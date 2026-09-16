# GreyCat CLI

The `greycat` binary is one executable that compiles, runs, serves, and administers a project. Every command takes the project's `project.gcl` (in the current working directory) as its entrypoint.

This reference is what the agent reaches for when the user says "build it", "serve it", "install the deps", "back it up", "generate the client", etc.

Static analysis is part of the same binary: `greycat lint`, `greycat fmt`, `greycat lsp`. See [lang.md](lang.md) for the rule list, formatter modes, and suppression directives.

## Contents

- Synopsis and binary discovery
- Commands (`run`, `serve`, `build`, `test`, …)
- Common options and environment variables
- Registries and tokens
- The `.env` file
- Codegen targets and conventions
- User administration
- Backup / restore / defrag
- Exit codes

## Synopsis

```sh
greycat <command> [options] [param] [short]
```

Run without a command, or with `-h` / `--help`, to print the help screen with the current set of recognized options and environment variables. Help adapts to the command — `greycat serve -h` shows only the options that apply to `serve`.

`-v` / `--version` prints the version. `-vv` / `--version-full` adds the git hash and target triple.

### Binary discovery (`bin/greycat`)

If the project has a `bin/greycat` (created by `greycat install` for a pinned core version), the system `greycat` re-execs into it. Set `GREYCAT_NO_REDIRECT=true` to bypass the redirect.

This is how a project pins a specific runtime version: `@library("std", "1.2.3")` in `project.gcl` resolves to `lib/std/`, and `greycat install` also downloads a matching core binary into `bin/`. Subsequent commands run through the pinned binary automatically.

## Commands

### `greycat new [name]`

Scaffolds a new project. With a `name`, creates a `./<name>/` directory; without one, scaffolds into the current directory. Refuses to run when the target already contains a `project.gcl`.

Writes `project.gcl` pinned to the latest `stable` `std` (resolved from the configured registry, see [Registries](#registries); by default `https://get.greycat.io/files/core/stable/latest`), a `src/api.gcl` with two `@expose`d example functions, a `tests/api_test.gcl` covering them, an `AGENT.md` that auto-loads the installed skill (`lib/std/skills/SKILL.md`), and a `.gitignore`. To add a frontend, follow [webapp.md](webapp.md) after installing.

```sh
greycat new my-app     # into ./my-app/
greycat new            # into the current directory
```

Follow with `greycat install`.

### `greycat run [function]`

Builds the project, then executes `function` (defaults to `main`). Each argument after `function` is JSON-parsed and bound to the matching parameter, coerced best-effort to its declared type, so primitives and complex objects both pass through. Used for one-off scripts, data processing, migrations.

Given `type Person { name: String; }` and `fn foo(a: int, b: String, p: Person) {}`:

```sh
greycat run                                  # runs main()
greycat run foo 42 "hello" '{"name":"John"}' # foo(42, "hello", Person { name: "John" })
```

### `greycat serve`

Builds the project, then serves it as a long-running HTTP/RPC server.

- Binds `GREYCAT_PORT` (default `8080`).
- If a `main` function exists with no parameters, it is enqueued as a startup task.
- `@expose` functions become reachable at `POST /<module>::<fn_name>` and via JSON-RPC at `POST /`.
- The user **root** is auto-created on first run; the URL with its login token is printed to stdout (when on a TTY).
- Serves `/files/...` from `<project>/files/` and `/...` from `<project>/webroot/` (see [project.md](project.md) for layout).

### `greycat dev`

Like `serve`, but auto-detects and spawns a frontend build tool in watch mode alongside the server. With no flag, it looks for `vite.config.{js,ts}` / `vp.config.{js,ts}` and runs the first of `vp build --watch`, `pnpm vite build --watch`, or `npx vite build --watch` that succeeds. Use `--with="<cmd>"` to spawn an arbitrary build command instead.

If the watched build process exits non-zero, `greycat dev` stops the server.

See [webapp.md](webapp.md) for the one prescribed webapp stack (VitePlus + MPA + Lit + Web Awesome `wa-*` components, headless `@greycat/web/sdk`): `app/` sources, `vite.config.ts` at the project root, bundle into `webroot/`.

### `greycat build`

Compiles the project to a `project.gcp` (GreyCat package) artifact alongside `project.gcl`. Stripped of every `*_test.gcl` module. Use to produce a deployable package without starting the runtime.

### `greycat test [function]`

Builds the project (including `*_test.gcl` modules), then runs every function annotated with `@test` (or just `function` if specified). Reports pass/fail counts. `--quiet` hides successful tests. See [annotations.md § @test](annotations.md).

### `greycat lint` / `greycat fmt` / `greycat lsp`

Static analysis over the entrypoint's `@library` / `@include` closure: `lint` reports diagnostics (`--fix` applies auto-fixes), `fmt` rewrites `.gcl` files canonically (`--mode=check` is the CI gate), `lsp` runs the language server over stdio.

```sh
greycat fmt --mode=check   # exit non-zero on formatting drift
greycat lint               # exit non-zero on any diagnostic
```

These three delegate to the `lang` library, loaded from `lib/lang/` or `~/.greycat/lib/lang/`. When it is absent they exit `127`; reinstall greycat, or pin the library with `@library("lang", "<version>");` and run `greycat install`. Full reference in [lang.md](lang.md).

### `greycat install`

Reads every `@library` pragma in the project closure (normally all in `project.gcl`) and downloads each library + the matching core binary into `lib/<name>/` and `bin/`. Skips libraries already at the requested version (tracked in `lib/installed`).

Downloads from `https://get.greycat.io/files/<lib>/<branch>/<major.minor>/<target>/<version>.zip`. `std` resolves under `core/`. That is the legacy registry; see [Registries](#registries) to point the CLI at a JSON-RPC registry and to store its token once.

| Option                    | Meaning                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `[PROJECT]`               | Positional. A `project.gcl` path, or a directory holding one. Defaults to the current directory.                     |
| `--bump[=patch\|latest]`  | Rewrite this project's `@library` pins to the newest release, then install what it wrote. Default bound is `patch`.  |
| `--branch=<name>`         | Branch to bump onto (`stable`, `dev`, ...). **Requires `--bump`.** Defaults to the branch each pin already names.    |
| `--force`                 | Re-download and re-extract every library, ignoring the cache and `lib/installed`.                                    |
| `--dry-run`               | Print what would be installed (or bumped) and exit. Writes nothing.                                                 |
| `--check`                 | Exit non-zero when anything is missing or out of date, without installing. For CI.                                  |
| `--prune`                 | Delete libraries that are installed but no longer declared.                                                         |
| `--offline`               | Install only from the local cache, never the network.                                                               |
| `--jobs=<n>`              | Concurrent downloads (default `4`).                                                                                 |

#### Moving version pins with `--bump`

`--bump` is how a project moves onto a newer release. It rewrites the version literals in `project.gcl` in place and then installs what it wrote, so there is no `curl .../latest` + hand-edit round trip to get wrong. `curl` remains fine for *reading* a version without touching the project (see [libraries.md](libraries.md)).

```sh
greycat install --bump                          # newest patch, same branch as each pin
greycat install --bump=latest                   # newest release on the branch, any major.minor
greycat install --bump --branch=stable          # move onto stable, newest patch there
greycat install --bump=latest --branch=dev      # newest dev, whatever its major.minor
greycat install --bump --dry-run                # preview; project.gcl is not touched
```

Rules it follows:

- **Bound.** `patch` (the default) keeps the declared `major.minor`: `8.1.0-dev` -> `8.1.145-dev`. `latest` takes the newest on the branch: `8.1.0-dev` -> `8.2.160-dev`.
- **Branch.** Read from the pin's own prerelease suffix: `8.2.0-dev` follows `dev`, `7.8.25-stable` follows `stable`. `--branch` overrides it.
- **Scope.** Only this project's declarations are rewritten. Pins inside an installed library under `lib/` belong to that library's author and are never touched.
- **Direction.** Within one branch a pin only moves forward. Switching branch may move it *backwards* (`8.2.0-dev` -> `8.1.150-stable`), because `stable` trails `dev`. That is the point of the switch, not a bug.
- **Unsuffixed pins.** `@library("std", "8.2")` names no branch, so there is nothing to follow: it is reported `no branch in the version; name one with --branch`. Adding `--branch=dev` resolves it to `8.2.160-dev`.

Every library gets one line, present tense, whether it moved or not:

```
    resolving   std
    update      std 8.1.0-dev -> 8.1.145-dev
```

```
    resolving   std
    up to date  std 8.2.160-dev
```

A pin that could not move says why, including what the looser bound would have given: `nothing newer on dev in this minor; --bump=latest would give 8.2.160-dev`.

### `greycat codegen [lang]`

Generates client bindings for the project's library types and `@expose`d functions. Targets: `c`, `ts`, `python`, `rust`, `java`. With no `lang`, auto-detects from project files: `CMakeLists.txt` → `c`, `Cargo.toml` → `rust`, `package.json|tsconfig.json|jsconfig.json` → `ts`, `requirements.txt` → `python`, `pom.xml|gradle.properties` → `java`.

Set `GREYCAT_CORE=1` to generate bindings for the `std` library itself (used by the GreyCat core team to regenerate the SDKs in `core/sdk/`).

### `greycat print <file.gcb>`

Pretty-prints the content of a `.gcb` (GreyCat Binary) file. `--format=json` for JSON output. `--pretty` is a boolean flag (default on); unset `GREYCAT_PRETTY` env var for compact output.

### `greycat bytecode`

Builds the project and dumps its compiled bytecode to stdout. Diagnostic tool — useful for inspecting compiler output.

### `greycat defrag`

Loads the program against the existing `gcdata/`, then compacts the data files. Reduces on-disk size after large deletes / overwrites.

### `greycat backup`

Writes a full or incremental backup to `GREYCAT_BACKUP_PATH` (default `./backup/`). Use `Runtime::backup_full()` / `Runtime::backup_delta()` from GCL for in-process backups instead.

### `greycat restore <archive>`

Restores `gcdata/` from a backup archive. `--verify` validates archive integrity before extraction.

### `greycat token`

Issues a session token for a user (default user id `1` = `root`). `--user=<id>` to choose, `--validity=<duration>` to set TTL (e.g. `1day`, `60min`, `3600s` — short forms like `1h` are not accepted).

### `greycat user <subcmd> ...`

User administration on the embedded LMDB-backed security DB.

```
greycat user list                              list all users
greycat user show <name>                       show user details and grants
greycat user add <name> [role]                 add user (default role: "user")
greycat user remove <name>                     remove user
greycat user role <name> <new_role>            change user role
greycat user grant <name> r|w|rw <target>      grant <name> file access to <target>
greycat user revoke <name> r|w|rw <target>     remove a previously-granted access
```

Built-in users: `id=0` (`public`, anonymous), `id=1` (`root`, role `admin`). Built-in roles live in `lib/std/runtime.gcl`: `public`, `user`, `admin`.

These subcommands operate on the security DB, which is created on the first `serve`/`run`. On a brand-new project they fail with `failed to open users database` until the runtime has booted once. To create the first user in one step, use `greycat run runtime::Identity::create <name> <role>`, which boots the runtime and creates the DB (see [workflow.md](workflow.md) "Creating users").

### `greycat stats`

Prints storage and program stats (zone sizes, fragmentation, type counts). Read-only.

### `greycat build-version` / `greycat build-version-full`

Prints the version recorded in the last build (`build-version-full` includes the git hash). Used by build scripts.

## Common options

Options can be passed on the command line (`--name=value`) or as environment variables (`GREYCAT_NAME=value`). The env-var form is also read from a `.env` file in the current working directory at startup.

| Option / env                            | Default            | Applies to                          | Meaning                                                                                                                                                                                                                            |
| --------------------------------------- | ------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--log` / `GREYCAT_LOG`                 | `info`             | `run`, `serve`, `dev`, …            | Log level: `none`, `error`, `warn`, `info`, `perf`, `trace`.                                                                                                                                                                       |
| `--logfile` / `GREYCAT_LOGFILE`         | `false`            | `run`, `serve`                      | Not consulted by the runtime. Logs always go to `files/root/log.csv` (see [runtime.md](runtime.md)); stdout carries them only on a TTY.                                                                                             |
| `--cache` / `GREYCAT_CACHE`             | 75% of host memory | `run`, `serve`                      | Worker object-cache size. Suffixes: `K`/`M`/`G`/`T` (binary), `KB`/`MB`/`GB`/`TB` (decimal), `KiB`/`MiB`/`GiB`/`TiB`.                                                                                                              |
| `--store` / `GREYCAT_STORE`             | 1 GB               | `run`, `serve`                      | Per-zone storage cap.                                                                                                                                                                                                              |
| `--port` / `GREYCAT_PORT`               | `8080`             | `serve`, `dev`                      | HTTP port. Set to `0` to pick a random free port (printed at startup).                                                                                                                                                             |
| `--webroot` / `GREYCAT_WEBROOT`         | `webroot`          | `serve`, `dev`                      | Project directory served at `/`. Library assets in `lib/<name>/webroot/` are served too and are not affected by this option.                                                                                                        |
| `--workers` / `GREYCAT_WORKERS`         | host CPU count     | `run`, `serve`, `test`              | Number of GreyCat task workers.                                                                                                                                                                                                    |
| `--http_threads`                        | `3`                | `serve`                             | IO threads serving HTTP connections.                                                                                                                                                                                               |
| `--req_workers`                         | `2`                | `serve`                             | Workers dedicated to handling JSON-RPC requests.                                                                                                                                                                                   |
| `--user` / `GREYCAT_USER`               | `public`           | `run`, `serve`, `test`              | **Footgun.** Bypasses auth on every request, running it as `<name>`. Never propose this flag — use `greycat token` + a real `Authorization` header instead. See [runtime.md § the `--user=<name>` impersonation flag](runtime.md). |
| `--validity` / `GREYCAT_VALIDITY`       | `24h`              | `serve`, `token`                    | TTL for session tokens.                                                                                                                                                                                                            |
| `--tz` / `GREYCAT_TZ`                   | host TZ            | `run`, `serve`                      | Default IANA timezone (e.g. `Europe/Luxembourg`).                                                                                                                                                                                  |
| `--mode` / `GREYCAT_MODE`               | none               | (global)                            | If set, runs as if the user had typed that command. Values: `serve`, `run`.                                                                                                                                                        |
| `--key` / `GREYCAT_KEY`                 | none               | `serve`                             | Path to a license / signing key.                                                                                                                                                                                                   |
| `--keysafe` / `GREYCAT_KEYSAFE`         | none               | `serve`                             | Password for the on-disk user secret store.                                                                                                                                                                                        |
| `--unsecure` / `GREYCAT_UNSECURE`       | `false`            | `serve`                             | Allow session tokens behind a non-HTTPS reverse proxy.                                                                                                                                                                             |
| `--backup_path` / `GREYCAT_BACKUP_PATH` | `backup`           | `backup`, `restore`, `run`, `serve` | Where backups go.                                                                                                                                                                                                                  |
| `--max_backup_files`                    | `3`                | `backup`, `run`, `serve`            | Max backup files retained in `backup_path`.                                                                                                                                                                                        |
| `--defrag_ratio`                        | `2.0`              | `run`, `serve`                      | Blocks held per live block tolerated across the whole store before a zone is defragged (`2.0` = half garbage). Must be `> 1.0` — `1.0` is a perfectly compacted store, so startup refuses anything in `(0, 1.0]`. `<= 0` disables auto-defrag.                                                                                                                                                                         |
| `--ca_path` / `GREYCAT_CA_PATH`         | none               | `run`, `serve`, `test`              | Directory of extra CA certs to trust for outbound TLS.                                                                                                                                                                             |
| `--keep_alive`                          | `false`            | `serve`                             | Enable HTTP keep-alive.                                                                                                                                                                                                            |
| `--task_pool_capacity`                  | `10000`            | `serve`                             | Max queued tasks.                                                                                                                                                                                                                  |
| `--request_pool_capacity`               | `512`              | `serve`                             | Max queued HTTP requests.                                                                                                                                                                                                          |
| `--request_ttl`                         | `20s`              | `serve`                             | Force-close requests that exceed this lifetime.                                                                                                                                                                                    |
| `--mcp_content` / `GREYCAT_MCP_CONTENT` | `both`             | `serve`, `dev`                      | How an MCP `tools/call` ships its payload: `both` (spec-recommended duplication), `structured` (`structuredContent` only, empty `content`), `text` (serialized `content` only, and no `outputSchema` is advertised).                |
| `--mcp_instructions`                    | none               | `serve`, `dev`                      | Usage guidance returned to MCP clients as `instructions` in the `initialize` result.                                                                                                                                               |
| `--force`                               | `false`            | `install`                           | Re-download even libraries already at the requested version. `install` has its own option set (`--bump`, `--branch`, `--check`, ...); see [`greycat install`](#greycat-install) above.                                             |
| `--registry` / `GREYCAT_REGISTRY`       | `legacy`           | `new`, `upgrade`, env-only `install` | Registry to resolve libraries and core releases from: `legacy` for `get.greycat.io`, otherwise the base URL of a JSON-RPC registry. See [Registries](#registries). |
| `--registry_token`                      | none               | `new`, `upgrade`, env-only `install` | `GREYCAT_REGISTRY_TOKEN`. Token sent to that registry, for packages that are not anonymously readable. Shown as `<set>` by `-h`. |
| `--with=<cmd>`                          | none               | `dev`                               | Watch-build command to spawn alongside the server.                                                                                                                                                                                 |
| `--worlds` / `GREYCAT_WORLDS`           | `1`                | `run`, `serve`                      | Number of parallel graph "worlds" (branching state) — see [runtime.md § many-worlds](runtime.md). Worker count is multiplied accordingly.                                                                                          |

Run `greycat <command> -h` to see only the options that apply to `<command>` along with their **resolved** values (after `.env` and env-var processing) — handy for debugging configuration.

## Common workflows

A short prescriptive cookbook — adapt paths/users to your project.

```bash
# Local dev — auto-rebuild watcher + verbose logs.
# The boot URL printed on first start includes a root token; copy it,
# or run `greycat token` separately to mint one. DO NOT add --user=<name>
# — that disables auth for every caller on the network. See runtime.md.
greycat dev --log=debug

# CI build (no server):
greycat install && greycat build && greycat test

# Production deploy:
greycat install
greycat build
GREYCAT_LOG=info GREYCAT_BACKUP_PATH=/var/backups/gc greycat serve

# Reset local data — DANGEROUS, wipes the graph:
#   ASK FOR CONFIRMATION before running on anything but a throwaway dev project.
rm -rf gcdata/
greycat run                              # blank-graph boot
```

Useful troubleshooting one-liners:

```bash
du -sh gcdata/                            # how big has the graph grown?
lsof -i :8080                             # who holds port 8080?
ls -lh gcdata/backup/                     # what backups do we have?
greycat stats                             # zone usage and cache hit rates
```

## Registries

`install`, `upgrade` and `new` resolve libraries and core releases from a registry. Unset, or `legacy`, means the historical file tree under `https://get.greycat.io/files/`. Any other value is the base URL of a GreyCat registry server, which answers `registry::resolve`, `registry::latest_version` and `registry::artifact_url` for libraries and their `asset_` counterparts for everything else (the `lang` tooling is published there as an asset).

Two settings, resolved in this order:

| Setting               | Flag (`new`, `upgrade`)     | Environment              | File key   |
| --------------------- | --------------------------- | ------------------------ | ---------- |
| Registry origin       | `--registry=<url>`          | `GREYCAT_REGISTRY`       | `default`  |
| Token for that origin | `--registry_token=<token>`  | `GREYCAT_REGISTRY_TOKEN` | `<origin>` |

The origin is settled first: flag, then environment, then the project `.env`, then the file's `default`. The token is then looked up for exactly that origin, and it is sent to that origin only, on RPC calls and on artifact downloads alike. `install`, `lint`, `fmt` and `lsp` forward their flags to the language tooling, so for them only the environment, `.env` and the file apply.

The file is `<home>/registries`, where `<home>` is `$GREYCAT_HOME` or `~/.greycat`. It is what lets a configured machine work with no variables at all:

```
# ~/.greycat/registries
default = https://registry.example.com
https://registry.example.com = <token>
https://greycat.corp.example = <token>
```

One `key = value` per line, `#` comments, trailing slashes ignored. It holds credentials: keep it `0600`; the CLI warns when other users can read it, and `greycat <command> -h` prints the token as `<set>`, never its value. Whatever the runtime resolves is exported to its own environment before any command runs, so the language tooling loaded in-process sees the same registry and token.

## The `.env` file

On startup, `greycat` looks for a `.env` in the current directory and reads `KEY=VALUE` lines into the process environment **before** applying CLI flags. Standard `.env` semantics:

- `KEY=value` — set.
- `KEY="quoted value with spaces"` — quoted; supports `\n`, `\r`, `\t`, `\\`, `\"` escapes.
- `# comment` — line and inline comments.
- Whitespace around `=` and trailing whitespace are trimmed.

Only `GREYCAT_*` keys are recognized. Other keys are still placed into the environment (visible to `System::getEnv`) but have no built-in meaning.

Precedence (lowest → highest): built-in defaults → `.env` file → process env vars → CLI flags.

## Codegen conventions

Generated SDKs land at standard paths in the project:

| Target   | Output path / convention                                               |
| -------- | ---------------------------------------------------------------------- |
| `c`      | Header(s) under the project's CMake-controlled tree.                   |
| `ts`     | Typed client into `node_modules`-style path or `src/`.                 |
| `python` | Python module — emit looks for `python` library or `requirements.txt`. |
| `rust`   | Crate-shaped output near `Cargo.toml`.                                 |
| `java`   | Maven/Gradle output near `pom.xml` / `gradle.properties`.              |

Codegen reads `@expose`, `@permission`, and `@tag` from the project's compiled program. `@tag("openapi")` and `@tag("mcp")` mark functions for inclusion in the OpenAPI spec exposed at runtime and in the MCP tool list respectively.

## OpenAPI and MCP

A served project exposes:

- **JSON-RPC** at `POST /` — call any `@expose`d function by method `"<module>.<fn_name>"` and a JSON `params` array or object.
- **Path-RPC** at `POST /<module>::<fn_name>` — body is a JSON array of positional args.
- **OpenAPI v3** — call `runtime::OpenApi::v3` (stdlib `Runtime::OpenApi::v3`) to get the spec from the live program. `@tag("openapi")` marks which functions appear in it.
- **MCP** — `tools/list` returns every function tagged `@tag("mcp")`; `tools/call` invokes them with named arguments matching the function's parameter names. By default a result is shipped twice, as `structuredContent` and as serialized `content` text; `--mcp_content` narrows that to one. `--mcp_instructions` sets the `instructions` hint clients may feed to the model.

  Argument binding is deliberately lenient, because the callers are language models: for a **nullable** parameter the strings `"null"` / `"None"` / `"nil"` / `""` bind `null`, and for an object or array parameter a string holding a whole JSON document is re-parsed. Both only apply after strict parsing has already failed, so a `String?` parameter still receives `"None"` verbatim. This applies to every JSON-bodied call — JSON-RPC, path-RPC and task arguments — not only to MCP tools.
- **`/files/...`** — read/write of `<project>/files/` (per-user subdirs, governed by ACL).
- **`/`** static webroot — files from `<project>/webroot/`, then from each `lib/<name>/webroot/`; first match wins.

See [runtime.md](runtime.md) for the auth / request lifecycle details.

## Exit codes

- `0` — success.
- `1` — generic CLI error (missing file, bad option). Also `lint` / `fmt --mode=check` reporting drift.
- `2` — compile/load error (program not buildable, or storage couldn't be upgraded).
- `127` — `lint` / `fmt` / `lsp` invoked without the `lang` library available.

`greycat <cmd> -h` is the source of truth for what flags `<cmd>` accepts on the version of `greycat` you have installed.
