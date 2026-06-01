# GreyCat CLI

The `greycat` binary is one executable that compiles, runs, serves, and administers a project. Every command takes the project's `project.gcl` (in the current working directory) as its entrypoint.

This reference is what the agent reaches for when the user says "build it", "serve it", "install the deps", "back it up", "generate the client", etc.

For static analysis (`lint`, `fmt`, LSP), see the sibling [`greycat-analyzer`](analyzer.md) — a separate binary, not bundled into `bin/`.

## Contents

- Synopsis and binary discovery
- Commands (`run`, `serve`, `build`, `test`, …)
- Common options and environment variables
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

### `greycat run [function]`

Builds the project, then executes `function` (defaults to `main`). Extra arguments after `function` are passed as its `Array<String>` argv. Used for one-off scripts, data processing, migrations.

```sh
greycat run                    # runs main()
greycat run my_fn              # runs my_fn()
greycat run my_fn arg1 arg2    # my_fn receives ["arg1", "arg2"]
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

See [webapp.md](webapp.md) for the recommended project layout (`app/` sources, `vite.config.ts` at the project root, bundle into `webroot/`).

### `greycat build`

Compiles the project to a `project.gcp` (GreyCat package) artifact alongside `project.gcl`. Stripped of every `*_test.gcl` module. Use to produce a deployable package without starting the runtime.

### `greycat test [function]`

Builds the project (including `*_test.gcl` modules), then runs every function annotated with `@test` (or just `function` if specified). Reports pass/fail counts. `--quiet` hides successful tests. See [annotations.md § @test](annotations.md).

### `greycat install`

Reads `@library` pragmas from `project.gcl` and downloads each library + the matching core binary into `lib/<name>/` and `bin/`. Skips libraries already at the requested version (tracked in `lib/installed`). `--force` re-downloads everything.

Downloads from `https://get.greycat.io/files/<lib>/<branch>/<major.minor>/<target>/<version>.zip`. `std` resolves under `core/`.

### `greycat codegen [lang]`

Generates client bindings for the project's library types and `@expose`d functions. Targets: `c`, `ts`, `python`, `rust`, `java`. With no `lang`, auto-detects from project files: `CMakeLists.txt` → `c`, `Cargo.toml` → `rust`, `package.json|tsconfig.json|jsconfig.json` → `ts`, `requirements.txt` → `python`, `pom.xml|gradle.properties` → `java`.

Set `GREYCAT_CORE=1` to generate bindings for the `std` library itself (used by the GreyCat core team to regenerate the SDKs in `core/sdk/`).

### `greycat print <file.gcb>`

Pretty-prints the content of a `.gcb` (GreyCat Binary) file. `--format=json` for JSON output. `--pretty` is a boolean flag (default on); unset `GREYCAT_PRETTY` env var for compact output.

### `greycat bytecode`

Builds the project and dumps its compiled bytecode to stdout. Diagnostic tool — useful for inspecting analyzer / compiler output.

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

### `greycat stats`

Prints storage and program stats (zone sizes, fragmentation, type counts). Read-only.

### `greycat build-version` / `greycat build-version-full`

Prints the version recorded in the last build (`build-version-full` includes the git hash). Used by build scripts.

## Common options

Options can be passed on the command line (`--name=value`) or as environment variables (`GREYCAT_NAME=value`). The env-var form is also read from a `.env` file in the current working directory at startup.

| Option / env                       | Default                | Applies to                | Meaning                                                                          |
| ---------------------------------- | ---------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| `--log` / `GREYCAT_LOG`            | `info`                 | `run`, `serve`, `dev`, …  | Log level: `none`, `error`, `warn`, `info`, `perf`, `trace`.                     |
| `--logfile` / `GREYCAT_LOGFILE`    | `false`                | `run`, `serve`            | Mirror logs to a file (alongside stdout).                                        |
| `--cache` / `GREYCAT_CACHE`        | 75% of host memory     | `run`, `serve`            | Worker object-cache size. Suffixes: `K`/`M`/`G`/`T` (binary), `KB`/`MB`/`GB`/`TB` (decimal), `KiB`/`MiB`/`GiB`/`TiB`. |
| `--store` / `GREYCAT_STORE`        | 1 GB                   | `run`, `serve`            | Per-zone storage cap.                                                            |
| `--port` / `GREYCAT_PORT`          | `8080`                 | `serve`, `dev`            | HTTP port. Set to `0` to pick a random free port (printed at startup).           |
| `--webroot` / `GREYCAT_WEBROOT`    | `webroot`              | `serve`, `dev`            | Directory served at `/`.                                                         |
| `--workers` / `GREYCAT_WORKERS`    | host CPU count         | `run`, `serve`, `test`    | Number of GreyCat task workers.                                                  |
| `--http_threads`                   | `3`                    | `serve`                   | IO threads serving HTTP connections.                                             |
| `--req_workers`                    | `2`                    | `serve`                   | Workers dedicated to handling JSON-RPC requests.                                 |
| `--user` / `GREYCAT_USER`          | `0`                    | `run`, `serve`, `test`    | **Footgun.** Bypasses auth on every request, running it as `<id>`. Never propose this flag — use `greycat token` + a real `Authorization` header instead. See [runtime.md § the `--user=<id>` impersonation flag](runtime.md). |
| `--validity` / `GREYCAT_VALIDITY`  | `24h`                  | `serve`, `token`          | TTL for session tokens.                                                          |
| `--tz` / `GREYCAT_TZ`              | host TZ                | `run`, `serve`            | Default IANA timezone (e.g. `Europe/Luxembourg`).                                |
| `--mode` / `GREYCAT_MODE`          | none                   | (global)                  | If set, runs as if the user had typed that command. Values: `serve`, `run`.      |
| `--key` / `GREYCAT_KEY`            | none                   | `serve`                   | Path to a license / signing key.                                                 |
| `--keysafe` / `GREYCAT_KEYSAFE`    | none                   | `serve`                   | Password for the on-disk user secret store.                                      |
| `--unsecure` / `GREYCAT_UNSECURE`  | `false`                | `serve`                   | Allow session tokens behind a non-HTTPS reverse proxy.                           |
| `--backup_path` / `GREYCAT_BACKUP_PATH` | `backup`         | `backup`, `restore`, `run`, `serve` | Where backups go.                                                  |
| `--max_backup_files`               | `3`                    | `backup`, `run`, `serve`  | Max backup files retained in `backup_path`.                                      |
| `--defrag_ratio`                   | `1.0`                  | `run`, `serve`            | Fragmentation ratio target. Negative disables auto-defrag.                       |
| `--ca_path` / `GREYCAT_CA_PATH`    | none                   | `run`, `serve`, `test`    | Directory of extra CA certs to trust for outbound TLS.                           |
| `--keep_alive`                     | `false`                | `serve`                   | Enable HTTP keep-alive.                                                          |
| `--task_pool_capacity`             | `10000`                | `serve`                   | Max queued tasks.                                                                |
| `--request_pool_capacity`          | `512`                  | `serve`                   | Max queued HTTP requests.                                                        |
| `--request_ttl`                    | `20s`                  | `serve`                   | Force-close requests that exceed this lifetime.                                  |
| `--force`                          | `false`                | `install`                 | Re-download even libraries already at the requested version.                     |
| `--with=<cmd>`                     | none                   | `dev`                     | Watch-build command to spawn alongside the server.                               |
| `--worlds` / `GREYCAT_WORLDS`      | `1`                    | `run`, `serve`            | Number of parallel graph "worlds" (branching state) — see [runtime.md § many-worlds](runtime.md). Worker count is multiplied accordingly. |

Run `greycat <command> -h` to see only the options that apply to `<command>` along with their **resolved** values (after `.env` and env-var processing) — handy for debugging configuration.

## Common workflows

A short prescriptive cookbook — adapt paths/users to your project.

```bash
# Local dev — auto-rebuild watcher + verbose logs.
# The boot URL printed on first start includes a root token; copy it,
# or run `greycat token` separately to mint one. DO NOT add --user=<id>
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

| Target | Output path / convention                                |
| ------ | ------------------------------------------------------- |
| `c`    | Header(s) under the project's CMake-controlled tree.    |
| `ts`   | Typed client into `node_modules`-style path or `src/`. |
| `python` | Python module — emit looks for `python` library or `requirements.txt`. |
| `rust` | Crate-shaped output near `Cargo.toml`.                   |
| `java` | Maven/Gradle output near `pom.xml` / `gradle.properties`. |

Codegen reads `@expose`, `@permission`, and `@tag` from the project's compiled program. `@tag("openapi")` and `@tag("mcp")` mark functions for inclusion in the OpenAPI spec exposed at runtime and in the MCP tool list respectively.

## OpenAPI and MCP

A served project exposes:

- **JSON-RPC** at `POST /` — call any `@expose`d function by method `"<module>.<fn_name>"` and a JSON `params` array or object.
- **Path-RPC** at `POST /<module>::<fn_name>` — body is a JSON array of positional args.
- **OpenAPI v3** — call `runtime::OpenApi::v3` (stdlib `Runtime::OpenApi::v3`) to get the spec from the live program. `@tag("openapi")` marks which functions appear in it.
- **MCP** — `tools/list` returns every function tagged `@tag("mcp")`; `tools/call` invokes them with named arguments matching the function's parameter names.
- **`/files/...`** — read/write of `<project>/files/` (per-user subdirs, governed by ACL).
- **`/`** static webroot — files from `<project>/webroot/`.

See [runtime.md](runtime.md) for the auth / request lifecycle details.

## Exit codes

- `0` — success.
- `1` — generic CLI error (missing file, bad option).
- `2` — compile/load error (program not buildable, or storage couldn't be upgraded).

`greycat <cmd> -h` is the source of truth for what flags `<cmd>` accepts on the version of `greycat` you have installed.
