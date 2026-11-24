# GreyCat CLI

The `greycat` binary is one executable that compiles, runs, serves, and administers a project. Every command takes the project's `project.gcl` (in the current working directory) as its entrypoint.

This reference is what the agent reaches for when the user says "build it", "serve it", "install the deps", "back it up", "generate the client", etc.

Static analysis is part of the same binary: `greycat lint`, `greycat fmt`, `greycat lsp`, `greycat doc`. See [lang.md](lang.md) for the rule list, formatter modes, suppression directives, and the API reference generator.

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
greycat [<command>] [options] [param] [short]
```

The command defaults to `serve`: bare `greycat` — and `greycat --port=8081`, i.e. options with no command in front of them — is `greycat serve`. An unrecognized command is still an error, not an implicit serve.

Run with `-h` / `--help` to print the help screen with the current set of recognized options and environment variables. Help adapts to the command — `greycat serve -h` shows only the options that apply to `serve`.

`-v` / `--version` prints the version. `-vv` / `--version-full` adds the git hash and target triple.

### Binary discovery (`bin/greycat`)

If the project has a `bin/greycat` (created by `greycat install` for a pinned core version), the system `greycat` re-execs into it. Set `GREYCAT_NO_REDIRECT=true` to bypass the redirect.

This is how a project pins a specific runtime version: `@library("std", "1.2.3")` in `project.gcl` resolves to `lib/std/`, and `greycat install` also downloads a matching core binary into `bin/`. Subsequent commands run through the pinned binary automatically.

## Commands

### `greycat new [name]`

Scaffolds a new project. With a `name`, creates a `./<name>/` directory; without one, scaffolds into the current directory. Refuses to run when the target already contains a `project.gcl`.

Writes `project.gcl` pinned to the latest `std` on the branch of the running `greycat` (`dev` for a `9.0.x-dev` build, `stable` for a release), resolved from the registry (see [Registries](#registries)), a `src/api.gcl` with two `@expose`d example functions, a `tests/api_test.gcl` covering them, an `AGENTS.md` that points agents at the installed skill (`lib/std/skills/SKILL.md`), a `CLAUDE.md` that imports it, and a `.gitignore`. To add a frontend, follow [webapp.md](webapp.md) after installing.

```sh
greycat new my-app     # into ./my-app/
greycat new            # into the current directory
```

Follow with `greycat install`.

### `greycat run [function]`

Builds the project, then executes `function` (defaults to `main`). Used for one-off scripts, data processing, migrations.

Arguments are positional, one per parameter, and are checked against the signature before anything starts:

- A `String` or `char` parameter takes its argument verbatim, no quoting needed.
- Any other parameter reads its argument as JSON against the declared type: `42`, `true`, `[1, 2.5]`, `{"name":"John"}`, `null`.
- Where an object is expected without a precise type (`any`, an abstract type), a `"_type"` field names its type: `'{"_type":"api::Circle","radius":2}'`. A `_type` naming a type the program does not define leaves a `Map`; one that does not read as that type is an error.
- Trailing arguments may be omitted when their parameters are nullable; they bind `null`. Too many arguments, a missing non-nullable one, or one that does not read as its type fails with a message naming the parameter.

Given `type Person { name: String; }` and `fn foo(a: int, b: String, p: Person) {}`:

```sh
greycat run                                  # runs main()
greycat run foo 42 hello '{"name":"John"}'   # foo(42, "hello", Person { name: "John" })
```

The result prints in GCL notation, or as JSON with `--format=json`; `--pretty` indents it. When stderr is a TTY, a progress bar shows once the function has run 500ms, with the percentage it reports through `Task::expected_steps` / `Task::add_steps`.

A first Ctrl-C lets the running code and `await` jobs finish, starts none of the jobs still queued, and exits with code 1 and `<function> did not complete: stopped` when the function had not returned. A second Ctrl-C quits at once.

`--world=<name>` (`GREYCAT_WORLD`) runs the function in that world (see [storage.md, Worlds](storage.md#worlds-branches-of-the-store)): it reads and commits to the world's index. Unset, or `main`, runs it in main; a name no world has fails before anything runs.

### `greycat call <function> [args...]`

Calls an `@expose`d function **on a running server** as a task, waits for it, and prints the result. Nothing is built or opened locally: the server owns `gcdata/`, so `call` works while `serve` is up, from any directory. A caller holding the `admin` and `debug` permissions, as the root token does, may call any function, exposed or not (see [http.md, Calling a method on an instance](http.md#calling-a-method-on-an-instance)): `greycat call core::nodeTime::size '"40960nodeTime"'`.

- `call` first downloads the server's ABI (`runtime::Runtime::abi`), then, with a token holding `admin` and `debug`, its program (`runtime::Runtime::program`), and resolves `function` and its signature from the program, or else from the ABI, which lists the exposed functions only. A greycat of another version cannot read the program and says so, reaching the exposed functions only. Arguments follow the same rules as [`greycat run`](#greycat-run-function), checked against that signature before the call is made: `greycat call api::add 2 3`, `greycat call api::greet John`, `greycat call api::area '{"_type":"api::Circle","radius":2}'`. A method that is not `static` takes its receiver as the first argument: `greycat call api::Counter::add '{"_type":"api::Counter","n":3}' 2` (see [http.md, Calling a method on an instance](http.md#calling-a-method-on-an-instance)).
- Arguments and result travel as GCB, the binary encoding of that ABI. The ABI carries the full parameter types, generics included: `'[1, "a"]'` for an `Array<float>` parameter fails before the call is made.
- `--url` (`GREYCAT_URL`) selects the server. Without it, the server is `http://localhost:<port>`: the `--port` given, else the port the server holding this project's store wrote to `gcdata/security/port`, else the default `8080`.
- `--token` (`GREYCAT_TOKEN`) is sent as `Authorization`. Unset, the root token `serve` writes to `gcdata/security/token` is read from the current directory, so running from the project directory needs no flag.
- `--task_class` (`GREYCAT_TASK_CLASS`) picks the worker class the server runs the task in: `small`, `medium` (the default) or `large`. A class the server has no workers for runs in the nearest one that has some.
- `--world` (`GREYCAT_WORLD`) runs the task in that world, sent as the `world` header: `main` is main, and a name the server has no world for fails the call (`400`), as does a world the caller may not use (`403`). Unset, the task runs in main, whatever the caller's default world (see [identity.md, The worlds of an identity](identity.md#the-worlds-of-an-identity)).
- The task is followed through the server's event stream (`runtime::Task::events`): `call` opens it first and names it in the `sse` header of the call. When stderr is a TTY, a progress bar shows the task status, elapsed time, and the percentage the function reports through `Task::expected_steps` / `Task::add_steps`. A server that refuses the stream (Windows, or the `--max_sse` caps reached) fails the call.
- The result prints like `run`'s: GCL notation, or JSON with `--format=json`; `--pretty` indents it. Exit code 0 when the task `ended`; non-zero when it threw, was cancelled, the function is unknown, an argument does not read, or the token is rejected. Ctrl-C cancels the remote task.
- A `@reserved` function (`runtime::Task::history`, `runtime::Task::running`, ...) is answered by the server directly rather than as a task, and `call` reads that answer. `runtime::Task::events` (a stream of events), `runtime::Runtime::abi` (the ABI file) and `runtime::Runtime::program` (the program file) answer with no value to print, so `call` refuses them.

### `greycat repl`

A prompt of calls **on a running server**, whose answers are kept in variables and passed to later calls. It connects as [`greycat call`](#greycat-call-function-args) does, with the same options (`--url`, `--port`, `--token`, `--tls`, `--world`, `--task_class`, `--format`, `--pretty`), and fails without a reachable server. Each call runs as a task: a progress line on a TTY, Ctrl-C cancels it and comes back to the prompt, a second Ctrl-C stops waiting for it (the task may still run). Ctrl-C at the prompt leaves, as Ctrl-D does.

```
greycat> t = mod::series()             # keeps the answer in t
greycat> t.size()                      # a method, t as its receiver; the answer goes to _
3
greycat> mod::scale(t, 2.5, "label")   # a variable, or a JSON literal, per argument
greycat> c = mod::make_counter(3)
greycat> c.n                           # a field of an object, read locally
3
```

- A line is `mod::fn(args)`, `mod::Type::fn(args)`, `v.m(args)`, `v.f` or `:command`, optionally behind `name = `. Calls nest, inner ones first. A literal is JSON and only an argument: a String is quoted (`"bob"`), since a bare word is a variable.
- `v.m(args)` finds `m` through the type of `v`, the generic definition of a specialization (`nodeTime` for a `nodeTime<int>`), then each parent; the override of the receiver's type runs. A `static` `m` is refused: call it on its type.
- The whole line is checked before any of its calls runs: every function resolves, and every literal and variable fits its parameter. The signatures come from the server as for `call`: its program for a caller holding `admin` and `debug`, its ABI (the exposed functions) for any other.
- A node is a reference to the server's graph; an object is a copy, which a method changing it does not update in the variable.
- `:vars` lists the variables with their types, `:help` the syntax, `:quit` (or Ctrl-D) leaves. A failed call leaves the variables as they were.
- On a terminal, the line has history (`gcdata/repl_history`) and tab completion of commands, variables, functions and, after `v.`, the fields and methods of `v`. Piped input is read line by line, with no prompt or banner, so a script gets only the answers on stdout and the errors on stderr.
- When the server restarts with another program (a schema change), the next call reloads its ABI, clears every variable, says so, and runs the line again.

### Store commands: in place or through the server

`backup`, `print`, `stats`, `token` and `user` work on the project's store, whether or not a server is running on it. Each runs one std function (`print` only needs the program):

- **No server holds the store** (`gcdata/lock` is free): the command takes the lock and runs the function in this process, as root, like `greycat run`.
- **A server holds it**, or `--url` is given: the command calls the function on that server, as [`greycat call`](#greycat-call-function-args) does, with the same `--url`, `--port` and `--token` options. Run from the project directory, it finds the server's port and root token under `gcdata/security/`, so it needs no flag.

The output is the same either way. The server-side functions are `runtime::Runtime::backup_full`, `runtime::Runtime::store_stats`, `runtime::Identity::token` and the other `runtime::Identity` functions, all callable directly too.

### `greycat serve`

Builds the project, then serves it as a long-running HTTP/RPC server. This is the default command — running `greycat` with no command does exactly this.

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

Compiles the project to a `project.gcp` (GreyCat package) artifact alongside `project.gcl`. Stripped of every `*_test.gcl` module. Use to produce a deployable package without starting the runtime. A `project.gcp` of another `.gcp` format version, which is every one 8.x built, is refused: rebuild it with `greycat build`. So is one whose bytecode holds a jump, a stack depth or an id the runtime would follow out of range, as a damaged or hand-made file can: the error names the function and the op at fault.

### `greycat test [function]`

Builds the project (including `*_test.gcl` modules), then runs every function annotated with `@test` (or just `function` if specified). Reports pass/fail counts. `--quiet` hides successful tests. A module whose run leaks memory or leaves a graph block marked counts as one more failure (`leak in module ...`, `mark errors in module ...`): GCL code cannot cause either, so it is a bug of the runtime or of a native library to report. Exit code is non-zero when anything failed. See [annotations.md § @test](annotations.md).

### `greycat lint` / `greycat fmt` / `greycat lsp` / `greycat doc`

Static analysis over the entrypoint's `@library` / `@include` closure: `lint` reports diagnostics (`--fix` applies auto-fixes), `fmt` rewrites `.gcl` files canonically (`--mode=check` is the CI gate), `lsp` runs the language server over stdio, `doc` writes the `doc.json` API reference of the package and fails on a broken doc link.

```sh
greycat fmt --mode=check   # exit non-zero on formatting drift
greycat lint               # exit non-zero on any error or warning
```

These delegate to the `lang` library, loaded from `lib/lang/` or `~/.greycat/lib/lang/`. When it is absent they exit `127`; reinstall greycat, or pin the library with `@library("lang", "<version>");` and run `greycat install`. Full reference in [lang.md](lang.md).

### `greycat install`

Reads every `@library` pragma in the project closure (normally all in `project.gcl`) and downloads each library + the matching core binary into `lib/<name>/` and `bin/`. Skips libraries already at the requested version (tracked in `lib/installed`).

Downloads each archive from the registry of its major: `https://registry.greycat.io` for GreyCat 9 and newer (unless [Registries](#registries) names another one), `https://get.greycat.io` for GreyCat 8 and older. The registry answers its URL for this machine's target, or for `noarch`. Libraries of the `toolchain` tier (`std`, `lang`, `rust`, `sdk_web`) download anonymously. Any other library needs a registry token, which [`greycat login`](#greycat-login--greycat-logout) stores once (see [libraries.md](libraries.md#tiers-and-tokens)).

| Option                    | Meaning                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `[LIB]...`                | Positional. Libraries to declare in the entrypoint and install, as `name` or `name@version`. A bare name resolves to the newest release on the branch and major.minor of the `std` pin. Without any, installs what the project declares. |
| `--project=<path>`        | A `project.gcl`, a linked `project.gcp`, or a directory holding one. Defaults to the current directory, where `project.gcl` wins. A `project.gcp` cannot be edited, so `--bump` and `[LIB]...` need the sources. |
| `--bump[=patch\|latest]`  | Rewrite this project's `@library` pins to the newest release, then install what it wrote. Default bound is `patch`.  |
| `--branch=<name>`         | Branch to resolve on (`stable`, `dev`, ...). With `--bump`, defaults to the branch each pin already names; with `[LIB]...`, to the branch of the `std` pin. |
| `--force`                 | Re-download and re-extract every library, ignoring the cache and `lib/installed`.                                    |
| `--dry-run`               | Print what would be installed (or bumped) and exit. Writes nothing.                                                 |
| `--check`                 | Exit non-zero when anything is missing or out of date, without installing. For CI.                                  |
| `--prune`                 | Delete libraries that are installed but no longer declared.                                                         |
| `--offline`               | Install only from the local cache, never the network.                                                               |
| `--jobs=<n>`              | Concurrent downloads (default `4`).                                                                                 |
| `--ca_path=<dir>`         | Extra CA certificates to trust for TLS, on top of the system store and of `GREYCAT_CA_PATH`, `SSL_CERT_FILE` and `SSL_CERT_DIR`. |
| `--color=<when>`          | `auto` (default), `always`, `never`.                                                                                |

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

### `greycat login` / `greycat logout`

`greycat login` stores a token of `https://registry.greycat.io` in `~/.greycat/registry_token` (see [Registries](#registries)), so `install` can fetch libraries outside the `toolchain` tier.

On a desktop it opens the registry in the browser. Sign in there if needed and approve the login. The registry then creates a 365-day token named `greycat login on <host>` and sends the browser back to a listener `login` opened on `127.0.0.1`, which receives a one-time code and exchanges it for the token. The code is useless to anyone else, since the exchange also needs a secret that never leaves the CLI. `login` waits 5 minutes for the browser.

It asks for a token at a prompt instead, which does not echo it, with `--paste` (`GREYCAT_PASTE`), over SSH, on a Linux or BSD machine with no display, or when no browser opens. Create the token on the account page (`https://registry.greycat.io/#/account`). When stdin is not a terminal the token is read from it, as in CI:

```sh
echo "$REGISTRY_TOKEN" | greycat login
```

A pasted token is checked with the registry before it is written, and `login` prints the account it belongs to. It refuses when `GREYCAT_REGISTRY` names another registry, since the stored token is never sent there, and warns when `GREYCAT_REGISTRY_TOKEN` is set, since that variable overrides the file.

`greycat logout` removes the file. The token itself stays valid until it expires or is revoked on the account page.

### `greycat codegen [lang]`

Generates client bindings for the project's library types and `@expose`d functions. Targets: `c`, `ts`, `python`, `rust`, `java`. With no `lang`, auto-detects from project files: `CMakeLists.txt` → `c`, `Cargo.toml` → `rust`, `package.json|tsconfig.json|jsconfig.json` → `ts`, `pom.xml|gradle.properties` → `java`, and `python` when the project declares `@library("python")`. The `python` target needs that library (it writes the `project_types.pyi` stub); without it the command fails and names the library. `python2` is the legacy generator, which writes `project_types.py` without the library.

Set `GREYCAT_CORE=1` to generate bindings for the `std` library itself (used by the GreyCat core team to regenerate the SDKs in `core/sdk/`).

### `greycat print <file.gcb>`

Pretty-prints the content of a `.gcb` (GreyCat Binary) file, decoded with the program of the store, or of the server holding it (see [Store commands](#store-commands-in-place-or-through-the-server)). A project with no store yet is built first. `--format=json` for JSON output. `--pretty` is a boolean flag (default on); unset `GREYCAT_PRETTY` env var for compact output.

### `greycat bytecode`

Builds the project and dumps its compiled bytecode to stdout. Diagnostic tool — useful for inspecting compiler output.

### `greycat defrag`

Loads the program against the existing `gcdata/`, then compacts the data files. Reduces on-disk size after large deletes / overwrites. Needs the store to itself: it fails while a server holds `gcdata/lock`, which compacts online on its own (`--defrag_ratio`).

### `greycat backup`

Writes a full backup with `Runtime::backup_full()`, in this process or on the server holding the store (see [Store commands](#store-commands-in-place-or-through-the-server)). The archive goes to the backup path of whichever runs it: `GREYCAT_BACKUP_PATH` (default `./backup/`). A backup that fails removes its partial archive. Use `Runtime::backup_full()` / `Runtime::backup_delta()` from GCL for in-process backups.

### `greycat restore <archive>`

Restores `gcdata/` from a backup archive. `--verify` stops at the first block that does not decode; `--repair` restores an archive holding such blocks by resetting the nodes they hold. See [storage.md](storage.md#restoring-a-damaged-archive).

### `greycat import <file.gce>`

Reads a store export written by `Runtime::export_store` into the project's store, which must be fresh: a new `gcdata/`, or one holding nothing but the empty nodes of its module variables. It builds the project first, as `run` does, so values are read into the program's current types. It prints `imported <n> nodes, <m> entries`, then a `skipped <type>::<field>: <count> values` line for each field the program no longer has. It exits non-zero and commits nothing when it fails. See [storage.md](storage.md#exporting-the-store).

### `greycat token`

Issues a session token with `Identity::token`, in this process or on the server holding the store (see [Store commands](#store-commands-in-place-or-through-the-server)). `--user=<name|id>` chooses the user, default id `0` (`public`). `--validity=<duration>` sets the TTL (e.g. `1day`, `60min`, `3600s`, `1hour30min`; the units are `us`, `ms`, `s`, `min`, `hour`, `day`, concatenated with no spaces or other words, so `1h`, `1sec` and `10s 5ms` are refused), default the validity the store or server is configured with. `--label=<text>` names the token as `Identity::tokens` and `greycat user tokens` list it (see [identity.md, Tokens](identity.md#tokens)).

### `greycat user <subcmd> ...`

User administration with the `runtime::Identity` functions, in this process or on the server holding the store (see [Store commands](#store-commands-in-place-or-through-the-server)).

```
greycat user list                                     list all users
greycat user show              <name>                 show user details and grants
greycat user add               <name> [role]          add user (default role: "user")
greycat user remove            <name>                 remove user and delete files/<name>/
greycat user role              <name> <new_role>      change user role
greycat user grant             <name> <target> <p1,p2,...>  grant <target> permissions on what <name> owns
greycat user revoke            <name> <target> <p1,p2,...>  take them back from <target>
greycat user grant-permission  <name> <holders> <p1,p2,...> grant them to every holder of the permission <holders>
greycat user revoke-permission <name> <holders> <p1,p2,...> take them back from the holders of <holders>
greycat user world             <name> <world|main>    set its default world, for clients to pick
greycat user worlds            <name> <w1,w2,...|->   set the worlds besides main its calls may run in, - for every world
greycat user tokens            <name>                 list its tokens that have not expired
greycat user revoke-token      <name> <id>            revoke one of its tokens
greycat user revoke-tokens     <name>                 revoke every one of its tokens
```

`world` and `worlds` call `Identity::set_world` and `Identity::set_worlds`, and `show` prints both settings; see [identity.md, The worlds of an identity](identity.md#the-worlds-of-an-identity). These commands, like `backup` and `stats`, always run in main.

`grant` and `revoke` call `Identity::grant` and `Identity::revoke` once for each permission of the list, which edit what `<target>` holds on the folder and tasks of `<name>`. `greycat user grant alice carol files:read,tasks:read` lets carol list and download `files/alice/` and see the tasks of alice, and gives alice nothing from carol. `grant-permission` and `revoke-permission` call `Identity::grant_permission` and `Identity::revoke_permission`, which grant the permissions to every caller whose role holds `<holders>`: `greycat user grant-permission alice api files:read` lets every signed-in user whose role holds `api` read `files/alice/`. A command fails when no permission of the list changed the grant. See [identity.md, Grants and per-user files](identity.md#grants-and-per-user-files), which also covers identities used as groups.

`tokens`, `revoke-token` and `revoke-tokens` call `Identity::tokens`, `Identity::revoke_token` and `Identity::revoke_tokens`. `tokens` prints the id, creation, expiry and label of each, and the ids are what `revoke-token` takes.

`show` prints, besides the role and worlds, the identities that grant the user permissions (`granted by:`) and those it grants permissions to (`grants to:`), each followed by the permissions, as in `carol(files:read,tasks:read)`. A grant to the holders of a permission reads `@grid(files:read)`.

Built-in users: `id=0` (`public`, anonymous), `id=1` (`root`, role `admin`). Built-in roles live in `lib/std/runtime.gcl`: `public`, `user`, `admin`.

On a brand-new project with no store yet, they build it first, so users can be set up before the first `serve`. Setting a password has no subcommand: `greycat run runtime::Identity::set_password <name> <password>` with no server running, `greycat call runtime::Identity::set_password <name> <password>` with one.

### `greycat stats`

Prints storage stats (live data, space on disk, fragmentation, size and count by type, damaged blocks) from `Runtime::store_stats()`, in this process or on the server holding the store (see [Store commands](#store-commands-in-place-or-through-the-server)). Read-only. Exits non-zero when part of the store would not read.

### `greycat build-version` / `greycat build-version-full`

Prints the version recorded in the last build (`build-version-full` includes the git hash). Used by build scripts.

## Common options

Options can be passed on the command line (`--name=value`) or as environment variables (`GREYCAT_NAME=value`). The env-var form is also read from a `.env` file in the current working directory at startup.

| Option / env                            | Default            | Applies to                          | Meaning                                                                                                                                                                                                                            |
| --------------------------------------- | ------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--log` / `GREYCAT_LOG`                 | `info`             | `run`, `serve`, `dev`, …            | Log level: `none`, `error`, `warn`, `info`, `trace`.                                                                                                                                                                                  |
| `--log_flush_size` / `GREYCAT_LOG_FLUSH_SIZE` | `64KB`   | `run`, `serve`, `dev`, `test`       | Log records are buffered and written to `files/root/streams/log.ndjson` once they reach this size. `0` writes each record at once.                                                                                                                                       |
| `--log_flush_step` / `GREYCAT_LOG_FLUSH_STEP` | `1s`     | `run`, `serve`, `dev`, `test`       | Longest time a log record stays buffered before it is written.                                                                                                                                                                                                             |
| `--log_flush_on_error` / `GREYCAT_LOG_FLUSH_ON_ERROR` | `false` | `run`, `serve`, `dev`, `test` | An `error` record writes the buffered log records to the file at once, so the last errors before a crash reach the file.                                                                                                                                                 |
| `--cache` / `GREYCAT_CACHE`             | 75% of host memory | `run`, `serve`                      | Object cache size. On Linux the host memory is the lowest memory limit of the process's cgroups when it is lower than the RAM, as in a container. 3/4 is shared by the task workers (see `--cache_ratio_small` / `--cache_ratio_large`), 1/4 backs the store zones. Suffixes: `K`/`M`/`G`/`T` (binary), `KB`/`MB`/`GB`/`TB` (decimal), `KiB`/`MiB`/`GiB`/`TiB`.                                                                                                              |
| `--store` / `GREYCAT_STORE`             | 1 GB               | `run`, `serve`                      | Map size of the LMDB index `gcdata/index.bin`, which holds the store metadata (commits, zone heads). Graph data lives in `gcdata/data_*.bin` and is not capped by it. A larger value grows an existing index, a smaller one leaves it at its current size. Same suffixes as `--cache`. |
| `--port` / `GREYCAT_PORT`               | `8080`             | `serve`, `dev`                      | HTTP port. Set to `0` to pick a random free port (printed at startup).                                                                                                                                                             |
| `--webroot` / `GREYCAT_WEBROOT`         | `webroot`          | `serve`, `dev`                      | Project directory served at `/`. Library assets in `lib/<name>/webroot/` are served too and are not affected by this option.                                                                                                        |
| `--workers` / `GREYCAT_WORKERS`         | host CPU count     | `run`, `serve`, `test`              | Task workers in total, split into the small, medium and large classes; medium takes what small and large leave and keeps at least one. On Linux the CPU count is lowered to the CPUs the process may run on (`taskset`, `docker --cpuset-cpus`) and to the CPU quota of its cgroups rounded up (`docker --cpus=1.5` gives 2). |
| `--workers_large` / `GREYCAT_WORKERS_LARGE` | 10% of `--workers` | `run`, `serve`, `test`              | How many of `--workers` run large tasks (rounded half up, so none below 5 workers); when idle they also run medium and small ones. |
| `--cache_ratio_small` / `GREYCAT_CACHE_RATIO_SMALL` | `0.4`              | `run`, `serve`, `test`              | Object cache of one small worker, relative to one medium worker. Must be > 0. |
| `--cache_ratio_large` / `GREYCAT_CACHE_RATIO_LARGE` | `3`                | `run`, `serve`, `test`              | Object cache of one large worker, relative to one medium worker. Must be > 0. |
| `--http_threads`                        | `3`                | `serve`                             | IO threads serving HTTP connections.                                                                                                                                                                                               |
| `--workers_small` / `GREYCAT_WORKERS_SMALL` | 20% of `--workers` | `serve`                             | How many of `--workers` run small tasks, which RPC requests run as; rounded, at least 1 from 2 workers. `run` and `test` start none. |
| `--user` / `GREYCAT_USER`               | `public`           | `run`, `serve`, `test`              | **Footgun.** Bypasses auth on every request, running it as `<name>`. Never propose this flag — use `greycat token` + a real `Authorization` header instead. See [identity.md, the `--user=<name>` impersonation flag](identity.md#the---username-impersonation-flag--footgun-do-not-promote). |
| `--validity` / `GREYCAT_VALIDITY`       | `24h`              | `serve`, `token`                    | TTL for session tokens.                                                                                                                                                                                                            |
| `--label` / `GREYCAT_TOKEN_LABEL`       | none               | `token`                             | Label of the token, as `Identity::tokens` lists it.                                                                                                                                                                                |
| `--tz` / `GREYCAT_TZ`                   | host TZ            | `run`, `serve`                      | Default IANA timezone (e.g. `Europe/Luxembourg`).                                                                                                                                                                                  |
| `--mode` / `GREYCAT_MODE`               | none               | (global)                            | If set, runs as if the user had typed that command. Values: `serve`, `run`.                                                                                                                                                        |
| `--key` / `GREYCAT_KEY`                 | none               | `serve`                             | Path to a license / signing key.                                                                                                                                                                                                   |
| `--keysafe` / `GREYCAT_KEYSAFE`         | none               | `serve`                             | Password for the on-disk user secret store.                                                                                                                                                                                        |
| `--unsecure` / `GREYCAT_UNSECURE`       | `false`            | `serve`, `dev`                      | Skip the certificate check of the outbound HTTPS requests of the `http` library. The server ignores it.                                                                                                                            |
| `--backup_path` / `GREYCAT_BACKUP_PATH` | `backup`           | `backup`, `restore`, `run`, `serve` | Where backups go.                                                                                                                                                                                                                  |
| `--max_backup_files`                    | `3`                | `backup`, `run`, `serve`            | Max backup files retained in `backup_path`.                                                                                                                                                                                        |
| `--defrag_ratio`                        | `2.0`              | `run`, `serve`                      | Blocks held per live block tolerated across the whole store before a zone is defragged (`2.0` = half garbage). Must be `> 1.0` — `1.0` is a perfectly compacted store, so startup refuses anything in `(0, 1.0]`. `<= 0` disables auto-defrag.                                                                                                                                                                         |
| `--ca_path` / `GREYCAT_CA_PATH`         | none               | `run`, `serve`, `test`              | Directory of extra CA certs to trust for outbound TLS, on top of the system store. A directory that yields no certificate is skipped with a warning. |
| `SSL_CERT_FILE` (env only)              | none               | every command                       | PEM file of extra CA certificates to trust for outbound TLS, read by the libraries and `greycat install` too. Added on top of the system store, where OpenSSL takes it instead of its default store. A file that yields no certificate is skipped with a warning. |
| `SSL_CERT_DIR` (env only)               | none               | every command                       | Directories of extra CA certificates, separated by `:` (`;` on Windows). Added like `SSL_CERT_FILE`, and a directory that yields no certificate is skipped with a warning. |
| `--tls` / `GREYCAT_TLS`                 | `false`            | `serve`, `dev`, `call`, `print`     | Serve HTTPS on `--port`, with `gcdata/security/tls/cert.pem` and `key.pem`. See [http.md](http.md#https).                                                                                                                                                                                                                    |
| `--acme` / `GREYCAT_ACME`               | none               | `serve`, `dev`                      | Comma-separated domains to obtain and renew the certificate of from Let's Encrypt, which turns `--tls` on. See [http.md](http.md#certificates-from-lets-encrypt).                                                                                                                                                            |
| `--acme_email`                          | none               | `serve`, `dev`                      | Contact email of the ACME account.                                                                                                                                                                                                                                                                                           |
| `--acme_directory`                      | Let's Encrypt      | `serve`, `dev`                      | Directory URL of the ACME authority, `https://acme-v02.api.letsencrypt.org/directory` by default.                                                                                                                                                                                                                            |
| `--keep_alive` / `GREYCAT_KEEP_ALIVE`   | `true`             | `serve`                             | Let a client reuse one connection for several requests. `--keep_alive=false` restores one-connection-per-request.                                                                                                                                                                                                            |
| `--task_pool_capacity`                  | `10000`            | `serve`                             | Max queued tasks.                                                                                                                                                                                                                  |
| `--request_pool_capacity`               | `8192`             | `serve`                             | Max connections served at once. A kept-alive connection holds a slot for its whole idle window, so this bounds concurrent *connections*, not just in-flight requests; past it, new arrivals wait in the listen backlog.                                                                                                                                                                                                          |
| `--request_ttl`                         | `20s`              | `serve`                             | Cancel an RPC request still queued or running after this long; it answers 503 with the reason.                                                                                                                                     |
| `--max_args_memory`                     | `1 MiB`            | `serve`                             | RPC arguments larger than this are streamed to the task's arguments file in `gcdata/tasks/<id>/` instead of held in memory; a function taking a `runtime::HttpRequest` gets such a body as `body_file`.                                                    |
| `--task_retention` / `GREYCAT_TASK_RETENTION` | `24h`        | `serve`, `dev`                      | How long the folder of a task (`gcdata/tasks/<id>/`: its result for `Task::result` and its `File::workingDir()`) is kept once the task ended. See [tasks.md](tasks.md#task-results-and-folders). |
| `--metrics` / `GREYCAT_METRICS`         | unset (none)       | `run`, `serve`                      | Comma-separated metrics to record, each to its own stream: `host` (`host_metrics`), `task` (`task_metrics`), `http` (`http_metrics`, one `HttpMetric` per HTTP request). Example: `--metrics=host,task,http`.                                                                                                                              |
| `--metrics_http_fields` / `GREYCAT_METRICS_HTTP_FIELDS` | unset (none) | `serve`                         | Personal data an `HttpMetric` records, comma-separated: `ip`, `forwarded_for` (`X-Forwarded-For`), `user_agent`. A field left out is null in every record.                                                                                                                |
| `--metrics_http_flush_size` / `GREYCAT_METRICS_HTTP_FLUSH_SIZE` | `64KB` | `serve`, `dev`                  | `HttpMetric` records are buffered and written to `http_metrics` once they reach this size. `0` writes each record at once.                                                                                                                                              |
| `--metrics_http_flush_step` / `GREYCAT_METRICS_HTTP_FLUSH_STEP` | `1s`   | `serve`, `dev`                  | Longest time an `HttpMetric` record stays buffered before it is written.                                                                                                                                                                                         |
| `--metrics_host_step` / `GREYCAT_METRICS_HOST_STEP` | `60s`              | `serve`                             | Period of the `HostMetric` record, when `host` is in `GREYCAT_METRICS`: host CPU, memory, disk, zones and worker classes.                                                                                                                                                 |
| `--metrics_host_flush_size` / `GREYCAT_METRICS_HOST_FLUSH_SIZE` | `64KB` | `run`, `serve`, `dev`            | `HostMetric` records are buffered and written to `host_metrics` once they reach this size. `0` writes each record at once.                                                                                                                                              |
| `--metrics_host_flush_step` / `GREYCAT_METRICS_HOST_FLUSH_STEP` | `1s`   | `run`, `serve`, `dev`            | Longest time a `HostMetric` record stays buffered before it is written.                                                                                                                                                                                                  |
| `--metrics_task_flush_size` / `GREYCAT_METRICS_TASK_FLUSH_SIZE` | `64KB` | `run`, `serve`, `dev`            | `TaskMetric` records are buffered and written to `task_metrics` once they reach this size. `0` writes each record at once.                                                                                                                                              |
| `--metrics_task_flush_step` / `GREYCAT_METRICS_TASK_FLUSH_STEP` | `1s`   | `run`, `serve`, `dev`            | Longest time a `TaskMetric` record stays buffered before it is written.                                                                                                                                                                                                  |
| `--max_sse` / `GREYCAT_MAX_SSE`         | `2048`             | `serve`                             | Cap on open task event streams (`GET /runtime::Task::events`); each holds a request pool slot. Past it a caller gets `429`.                                                                                                       |
| `--max_sse_per_user` / `GREYCAT_MAX_SSE_PER_USER` | `16`     | `serve`                             | Cap on open task event streams per user. Past it a caller gets `429`.                                                                                                                                                             |
| `--mcp_content` / `GREYCAT_MCP_CONTENT` | `both`             | `serve`, `dev`                      | How an MCP `tools/call` ships its payload: `both` (spec-recommended duplication), `structured` (`structuredContent` only, empty `content`), `text` (serialized `content` only, and no `outputSchema` is advertised).                |
| `--mcp_instructions`                    | none               | `serve`, `dev`                      | Usage guidance returned to MCP clients as `instructions` in the `initialize` result.                                                                                                                                               |
| `--openapi` / `GREYCAT_OPENAPI`         | `true`             | `serve`, `dev`                      | Whether every `@expose`d function lands in the OpenAPI v3 document by default. `false` narrows the document to functions carrying `@tag("openapi")`.                                                                                |
| `--force`                               | `false`            | `install`                           | Re-download even libraries already at the requested version. `install` has its own option set (`--bump`, `--branch`, `--check`, ...); see [`greycat install`](#greycat-install) above.                                             |
| `--registry` / `GREYCAT_REGISTRY`       | `https://registry.greycat.io` | `new`, `upgrade`, env-only `install` | Base URL of the registry to resolve libraries and core releases from. An empty value means the default. See [Registries](#registries). |
| `--registry_token`                      | none               | `new`, `upgrade`, env-only `install` | `GREYCAT_REGISTRY_TOKEN`. Token sent to that registry, for packages that are not anonymously readable. Shown as `<set>` by `-h`. |
| `--paste` / `GREYCAT_PASTE`             | `false`            | `login`                             | Ask for a registry token at the prompt instead of approving the login in the browser. See [`greycat login`](#greycat-login--greycat-logout). |
| `--with=<cmd>`                          | none               | `dev`                               | Watch-build command to spawn alongside the server.                                                                                                                                                                                 |

Run `greycat <command> -h` to see only the options that apply to `<command>` along with their **resolved** values (after `.env` and env-var processing) — handy for debugging configuration.

## Common workflows

A short prescriptive cookbook — adapt paths/users to your project.

```bash
# Local dev — auto-rebuild watcher + verbose logs.
# The boot URL printed on first start includes a root token; copy it,
# or run `greycat token` separately to mint one. DO NOT add --user=<name>
# — that disables auth for every caller on the network. See identity.md.
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

`install`, `upgrade` and `new` resolve libraries and core releases of GreyCat 9 and newer from a GreyCat registry, `https://registry.greycat.io` by default. `install` resolves a GreyCat 8 pin from `https://get.greycat.io` instead, whatever `GREYCAT_REGISTRY` says (see [Two majors on one machine](#two-majors-on-one-machine)). It answers `registry::resolve`, `registry::latest_version` and `registry::artifact_url`. The core archive, which holds both `bin/greycat` and `lib/std`, is the `std` library there, and the language tooling is the `lang` library. `GREYCAT_REGISTRY` points a machine at another registry, such as a local or a staging instance.

The token, which libraries outside the `toolchain` tier need, is the first of:

1. `--registry_token=<token>` (`new` and `upgrade` only).
2. `GREYCAT_REGISTRY_TOKEN`, from the environment or the project `.env`.
3. `<home>/registry_token`, where `<home>` is `$GREYCAT_HOME` or `~/.greycat`, holding the bare token. [`greycat login`](#greycat-login--greycat-logout) writes it.

The file belongs to `https://registry.greycat.io`: it is read only when no `GREYCAT_REGISTRY` (or `--registry`) names another registry, so a stored token never reaches another host. A local or staging registry takes its token from `GREYCAT_REGISTRY_TOKEN`. Whichever token applies is sent to the registry only, on RPC calls and on artifact downloads alike, as `Authorization: <token>` with no `Bearer` prefix.

The file holds a credential: keep it `0600`. The CLI warns when other users can read it, and `greycat <command> -h` prints the token as `<set>`, never its value. Whatever the runtime resolves is exported to its own environment before any command runs, so the language tooling loaded in-process sees the same registry and token. `install`, `lint`, `fmt` and `lsp` forward their flags to that tooling, so for them only the environment, `.env` and the file apply.

### Two majors on one machine

A project picks its GreyCat major with its `std` pin, and the global `greycat` (the one on `PATH`, usually `~/.greycat/bin/greycat`) may be of the other major. Two rules decide what happens:

- `install` and `upgrade` always run in the global binary. Every other command, run as `greycat` from `PATH` inside a project, re-execs into the project's `bin/greycat` (see [Binary discovery](#binary-discovery-bingreycat)). Only the first `install` of a project depends on the global major.
- `lint`, `fmt` and `lsp` load the project's `lib/lang/`, else the global `~/.greycat/lib/lang/`. A global `lang` of the other major checks the wrong grammar (a GreyCat 8 `lang` rejects a `nodeIndexSet` module variable). `greycat install lang` pins `lang` on the branch and major of the `std` pin, which ends the mismatch.

**Global GreyCat 9, GreyCat 8 project.** Nothing to set up, since a GreyCat 9 `install` fetches an 8.x pin from `get.greycat.io`:

```sh
greycat install         # std 8.4.x-stable and bin/greycat come from get.greycat.io
greycat install lang    # writes @library("lang", "8.4.x-stable")
```

**Global GreyCat 8, GreyCat 9 project.** An 8.x `install` reads only `get.greycat.io` by default, and fails with `not published for x64-linux; latest: 9.0.267-dev, 8.4.666-stable`. Name the registry in the project `.env`, which the 8.x `install` reads too, and pass the token in the environment for libraries outside the `toolchain` tier (an 8.x CLI does not read `~/.greycat/registry_token`):

```sh
echo 'GREYCAT_REGISTRY=https://registry.greycat.io' >> .env
GREYCAT_REGISTRY_TOKEN="$(cat ~/.greycat/registry_token)" greycat install
greycat install lang
```

Keep the token out of `.env`, which `greycat new` does not gitignore.

**A second toolchain.** Each registry has an install script that puts a toolchain wherever `GREYCAT_HOME` says. The argument is a branch (`stable` when omitted) or, for `registry.greycat.io`, an exact version:

```sh
curl -fsSL https://registry.greycat.io/install.sh | GREYCAT_HOME=~/.greycat9 bash -s dev   # GreyCat 9
curl -fsSL https://get.greycat.io/install.sh | GREYCAT_HOME=~/.greycat8 bash                # GreyCat 8
```

Both scripts empty `$GREYCAT_HOME/lib` first, so give the second toolchain its own home. Both also add its `bin` to the front of `PATH` in `~/.zshrc` or `~/.bashrc`, which makes it the default `greycat` of every new shell. Remove that line to keep the first toolchain as the default, and run the second one by its full path for the first install of a project (`~/.greycat9/bin/greycat install`). From then on the project's `bin/greycat` runs every other command.

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
| `python` | `project_types.pyi`, written by the `python` library, which the project must declare. |
| `rust`   | Crate-shaped output near `Cargo.toml`.                                 |
| `java`   | Maven/Gradle output near `pom.xml` / `gradle.properties`.              |

Codegen reads `@expose`, `@permission`, and `@tag` from the project's compiled program. `@tag("mcp")` marks functions for inclusion in the MCP tool list; `@tag("openapi")` does the same for the OpenAPI spec, though by default every `@expose`d function is in it already (see `--openapi` below).

## OpenAPI and MCP

A served project exposes:

- **JSON-RPC** at `POST /` — call any `@expose`d function by method `"<module>::<fn_name>"` and a JSON `params` array or object.
- **Path-RPC** at `POST /<module>::<fn_name>` — body is a JSON array of positional args.
- **OpenAPI v3** — call `runtime::OpenApi::v3` (stdlib `Runtime::OpenApi::v3`) to get the spec from the live program. `--openapi` / `GREYCAT_OPENAPI` decides what lands in it: `true` (the default) puts every `@expose`d function in the document, `false` narrows it to the ones carrying `@tag("openapi")`.

  Paths are then filtered by the caller's permissions, exactly as MCP `tools/list` is — the document only lists functions that caller could actually invoke, and `components/schemas` only carries the types those functions reference. That is why the endpoint is itself `@permission("public")`: an anonymous caller reaches it but sees only the `@permission("public")` endpoints, while a `role=user` token sees the `api` ones and an admin sees everything. A build step that wants the unfiltered document should generate it from `greycat run`, which runs with every permission bit set.
- **MCP** — `tools/list` returns every function tagged `@tag("mcp")`; `tools/call` invokes them with named arguments matching the function's parameter names. By default a result is shipped twice, as `structuredContent` and as serialized `content` text; `--mcp_content` narrows that to one. `--mcp_instructions` sets the `instructions` hint clients may feed to the model.

  Argument binding is deliberately lenient, because the callers are language models: for a **nullable** parameter the strings `"null"` / `"None"` / `"nil"` / `""` bind `null`, and for an object or array parameter a string holding a whole JSON document is re-parsed. Both only apply after strict parsing has already failed, so a `String?` parameter still receives `"None"` verbatim. This applies to every JSON-bodied call — JSON-RPC, path-RPC and task arguments — not only to MCP tools.
- **`/files/...`** — read/write of `<project>/files/` (per-user subdirs, governed by ACL).
- **`/`** static webroot — files from `<project>/webroot/`, then from each `lib/<name>/webroot/`; first match wins.

See [http.md](http.md) for the auth / request lifecycle details.

## Exit codes

- `0` — success.
- `1` — generic CLI error (missing file, bad option). Also `lint` / `fmt --mode=check` reporting drift.
- `2` — compile/load error (program not buildable, or storage couldn't be upgraded).
- `127` — `lint` / `fmt` / `lsp` invoked without the `lang` library available.

`greycat <cmd> -h` is the source of truth for what flags `<cmd>` accepts on the version of `greycat` you have installed.
