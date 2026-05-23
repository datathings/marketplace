# Working with a GreyCat project

Reach for this file when the task is operational: starting a fresh project, evolving an existing one, adding an endpoint, running tests, shipping a build. The other reference files (`syntax.md`, `types.md`, `stdlib.md`, `cli.md`, `runtime.md`) cover the *what*; this one is the *how*.

## Contents

- Bootstrapping a new project
- The edit / install / run loop
- Definition of done (lint + fmt + build + test)
- Adding an HTTP endpoint
- Adding a graph-persistent type
- Writing and running tests
- Schema / type evolution
- Generating SDKs for external clients
- Deploying
- Common project chores

## Bootstrapping a new project

A minimum viable project is three files: a `project.gcl`, one source file, and a `.env` (optional).

```sh
mkdir my-app && cd my-app
```

```gcl
// project.gcl
@library("std", "8.0.0");
@include("src");

fn main() {
    println("hello, greycat");
}
```

```gcl
// src/api.gcl
@expose
@permission("public")
fn ping(): String {
    return "pong";
}
```

First-time install + serve:

```sh
greycat install                 # downloads lib/std/ and bin/greycat
./bin/greycat serve             # or: greycat serve  (system binary auto-redirects to bin/)
```

The serve command prints the boot URL with the `root` user's token. Open it, or call `ping` with `curl`:

```sh
curl -X POST http://localhost:8080/api::ping
```

For a frontend-bundled project, swap `serve` for `dev` to also spawn a watcher (`vp`, `vite`, or a custom `--with` command). See [webapp.md](webapp.md) for the recommended layout (`app/` sources, Vite/VitePlus config at the project root, bundle into `webroot/`).

## The edit / install / run loop

1. **Edit `.gcl`** under `src/` (or wherever your `@include` covers).
2. **`greycat-analyzer lint`** — fastest signal. Catches `unused-local`, `possibly-null`, `arrow-on-non-deref`, non-exhaustive enum chains, and dozens of other shape issues the runtime accepts silently. See [analyzer.md](analyzer.md).
3. **`greycat-analyzer fmt`** — canonical formatting. The formatter is opinionated and unconfigurable.
4. **Restart the server.** GreyCat does not currently hot-reload schema changes — `serve` rebuilds the project at startup. Save → `Ctrl+C` → `serve` again.
5. **`greycat install`** only when `project.gcl`'s `@library` pragmas change. Library cache lives in `lib/installed`.
6. **`greycat codegen`** if external SDKs need to pick up new endpoint signatures.

For one-off scripts (data import, migration), use `greycat run [function]` — no server, just executes and exits.

## Definition of done

Before declaring a `.gcl` change finished (and before committing), run this gate:

```sh
greycat-analyzer fmt --mode=check       # exit non-zero on formatting drift
greycat-analyzer lint                   # exit non-zero on error-severity diagnostics
greycat build                           # produce project.gcp
greycat test                            # run @test functions
```

`greycat build` is **not** a substitute for `lint` — the runtime accepts code with unused locals, redundant null-checks, and other shape drift. Apply auto-fixes with `greycat-analyzer fmt` (default `--mode=write`) and `greycat-analyzer lint --fix`. See [analyzer.md](analyzer.md) for the rule list and suppression directives.

## Adding an HTTP endpoint

```gcl
// src/api.gcl
type EchoRequest { msg: String; }
type EchoResponse { echoed: String; at: time; }

/// Echoes the incoming message back, with a timestamp.
/// @param req The echo request
@expose
@tag("openapi", "mcp")
fn echo(req: EchoRequest): EchoResponse {
    return EchoResponse {
        echoed: req.msg,
        at: time::now(),
    };
}
```

Notes:

- `@expose` makes it reachable at `POST /<module>::echo` (here `/api::echo`) and via JSON-RPC method `"api.echo"`.
- Without `@permission`, the function requires the `api` permission (default for authenticated callers). Add `@permission("public")` to allow anonymous calls.
- `@tag("openapi")` includes it in the spec returned by `OpenApi::v3`. `@tag("mcp")` exposes it as an MCP tool.
- `/// @param <name> <description>` doc-comment lines surface in the generated OpenAPI / MCP schemas.

## Adding a graph-persistent type

Persistent state is modeled with node-tag attributes. Plain types are also persisted as long as they are not `@volatile`.

```gcl
// src/models/sensor.gcl
type Sensor {
    name: String;
    measurements: nodeTime<float>;       // persistent time-series
}

// src/services/recording.gcl
fn record(s: Sensor, v: float) {
    s.measurements.setAt(time::now(), v);
}

fn window_avg(s: Sensor, from: time, to: time): float {
    var sum = 0.0;
    var n = 0;
    for (t, v in s.measurements[from..to]) {
        sum = sum + v;
        n = n + 1;
    }
    if (n == 0) { return 0.0; }
    return sum / n;
}
```

To make the sensor reachable from the graph root, declare a **module-level node variable** (these are the only kinds of module-level `var` allowed):

```gcl
// src/models/sensors.gcl
var sensors: nodeIndex<String, Sensor>;     // global, persisted, lazily created
```

`sensors` exists across restarts. First access creates it; subsequent calls read it back from `gcdata/`.

See [types.md](types.md) for nullability rules and [stdlib.md](stdlib.md) for the full node-tag API.

## Writing and running tests

```gcl
// test/api_test.gcl
@test
fn test_echo() {
    var resp = echo(EchoRequest { msg: "hi" });
    Assert::equals(resp.echoed, "hi");
}
```

Pin tests to a separate directory and include it from `project.gcl`:

```gcl
@library("std", "8.0.0");
@include("src");
@include("test");
```

```sh
greycat test                    # run every @test function
greycat test test_echo          # run a single test by name
```

`greycat build` strips every `*_test.gcl` module from the production `.gcp` automatically.

### Test users (server tests)

Server-mode integration tests typically need extra users. Create them in a setup function and call it before starting the server:

```gcl
@expose
fn setup_users() {
    // creates "alice" with role "user" if not present
    Identity::create("alice", "user");
    Identity::set_password("alice", "...");
}
```

```sh
greycat run setup_users         # one-shot
greycat serve                   # then serve
```

The test harness in `test/server/` of this repo is a working reference.

## Schema / type evolution

The runtime stores an ABI snapshot in `gcdata/abi` next to the program. On rebuild, the compiler diffs the new program against the snapshot and migrates compatible changes:

| Change                                                  | Migrates automatically?                                  |
| ------------------------------------------------------- | -------------------------------------------------------- |
| Add a nullable attribute                                | Yes — existing instances read it as `null`.              |
| Add a non-null attribute with a default                 | Yes — existing instances get the default.                |
| Add a non-null attribute without default                | No — load fails. Provide a default or write a migration. |
| Remove an attribute                                     | Yes — value is dropped on next save.                     |
| Rename an attribute                                     | No — looks like remove + add.                            |
| Change an attribute's type to a wider one (e.g. `int` → `int?`) | Yes.                                              |
| Change an attribute's type to a narrower one            | No — load fails on a non-conforming value.                |
| Add/remove an enum entry                                | Add: yes. Remove: only if no instance references it.     |
| Rename a type                                           | No — looks like a new type.                              |

When the runtime cannot migrate, it refuses to load and prints the offending shape diff. Either undo the source change, write a migration in `greycat run` mode (read the old graph, transform, save), or restore from backup.

## Generating SDKs

Run `greycat codegen` from the project root. The generator auto-detects the target from project sentinel files:

```sh
greycat codegen                # auto-detect (CMakeLists.txt, Cargo.toml, package.json, ...)
greycat codegen ts             # explicit target
```

Each codegen run reads the current `project.gcl` and emits typed clients for every public type and `@expose`d function, plus client-side proxies for the chosen language. Re-run after every API change.

## Deploying

A production GreyCat deployment is **one binary, one `gcdata/`, one `webroot/`, one `files/`.** No external services. Suggested checklist:

- [ ] `greycat-analyzer fmt --mode=check` and `greycat-analyzer lint` both exit `0` on the deploy commit. See [analyzer.md](analyzer.md).
- [ ] `greycat build` produces a `project.gcp` artifact (or ship the source tree with `lib/` populated).
- [ ] Ship `project.gcl`, `src/`, `lib/`, `webroot/`, and `bin/greycat` (or have the deploy host run `greycat install` once).
- [ ] Provision a writable `gcdata/` (this is the durable state).
- [ ] Provision a writable `files/` if the app accepts uploads.
- [ ] Configure with `.env` (or systemd `Environment=`): `GREYCAT_PORT`, `GREYCAT_LOG`, `GREYCAT_CACHE`, `GREYCAT_STORE`, `GREYCAT_BACKUP_PATH`, `GREYCAT_TZ`.
- [ ] Behind an HTTPS reverse proxy, set `GREYCAT_UNSECURE=true` (otherwise the runtime refuses to mint session tokens over plain HTTP).
- [ ] Schedule backups: either external (snapshot the `gcdata/` directory), or in-process via a scheduled `Runtime::backup_delta()`.
- [ ] Set up monitoring on `Runtime::usage()` if the workload is heavy on memory or storage.

## Common project chores

| Task                                  | How                                                                              |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| Pin a new stdlib version              | Bump `@library("std", "X.Y.Z")` in `project.gcl`, then `greycat install`.        |
| Add a third-party library             | Add `@library("name", "version");` then `greycat install`.                       |
| Roll back to a known graph state      | `greycat restore <archive>`. Stop the server first.                              |
| Inspect a `.gcb` file from disk       | `greycat print path/to/file.gcb` (use `--format=json` for JSON).                  |
| Inspect the compiled bytecode         | `greycat bytecode`.                                                              |
| Reset everything                      | `rm -rf gcdata/ bin/ lib/` then `greycat install` + `greycat serve`. **Data loss.** |
| Add a new admin user                  | `greycat user add alice admin` then `greycat token --user=<id>` for their token. |
| See what's running                    | `Task::running()` from GCL, or hit the corresponding admin endpoint over RPC.    |
| Cancel a stuck task                   | `Task::cancel(task_id)`.                                                         |
| Run a one-shot migration script       | Write a function, then `greycat run my_migration_fn`.                            |
