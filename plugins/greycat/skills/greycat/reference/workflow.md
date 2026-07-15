# Working with a GreyCat project

Reach for this file when the task is operational: starting a fresh project, evolving an existing one, adding an endpoint, running tests, shipping a build. The other reference files (`syntax.md`, `types.md`, `stdlib.md`, `cli.md`, `runtime.md`) cover the _what_; this one is the _how_.

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
fn ping(): String {
    return "pong";
}
```

No `@permission` clause — `ping` requires the default `api` permission (any authenticated caller). Don't reach for `@permission("public")` to dodge auth during bootstrap; the boot URL printed by `serve` carries a `root` token, which is enough to exercise the endpoint.

First-time install + serve:

```sh
greycat install                 # downloads lib/std/ and bin/greycat
./bin/greycat serve             # or: greycat serve  (system binary auto-redirects to bin/)
```

The serve command prints the boot URL with the `root` user's token. Open it, then call `ping` with the token attached:

```sh
NAME="user_name" # eg. alice, root, etc.
VALIDITY="duration" # eg. 3day, 2hour, etc.
TOKEN=$(greycat token --user=$NAME --validity=$VALIDITY)
curl -X POST -H "Authorization: $TOKEN" "http://localhost:8080/api::ping"
```

For a frontend-bundled project, swap `serve` for `dev` to also spawn the VitePlus watcher. See [webapp.md](webapp.md) for the one prescribed stack (VitePlus + MPA + Lit + Web Awesome `wa-*` components), with `app/` sources bundled into `webroot/`.

## The edit / install / run loop

1. **Edit `.gcl`** under `src/` (or wherever your `@include` covers).
2. **`greycat-lang lint`** — fastest signal. Catches `unused-local`, `possibly-null`, `arrow-on-non-deref`, non-exhaustive enum chains, and dozens of other shape issues the runtime accepts silently. See [lang.md](lang.md).
3. **`greycat-lang fmt`** — canonical formatting. The formatter is opinionated and unconfigurable.
4. **Restart the server.** GreyCat does not currently hot-reload schema changes — `serve` rebuilds the project at startup. Save → `Ctrl+C` → `serve` again.
5. **`greycat install`** only when `project.gcl`'s `@library` pragmas change. Library cache lives in `lib/installed`.
6. **`greycat codegen`** if external SDKs need to pick up new endpoint signatures.

For one-off scripts (data import, migration), use `greycat run [function]` — no server, just executes and exits.

## Definition of done

Before declaring a `.gcl` change finished (and before committing), run this gate:

```sh
greycat-lang fmt --mode=check  # exit non-zero on formatting drift
greycat-lang lint              # exit non-zero on error-severity diagnostics
greycat build                  # produce project.gcp
greycat test                   # run @test functions
```

`greycat build` is **not** a substitute for `greycat-lang lint` — the runtime accepts code with unused locals, redundant null-checks, and other shape drift. Apply auto-fixes with `greycat-lang fmt` (default `--mode=write`) and `greycat-lang lint --fix`. See [lang.md](lang.md) for the rule list and suppression directives.

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
- Without `@permission`, the function requires the `api` permission (any authenticated caller) — that is the right default. Add `@permission("admin")` (or a custom permission declared in `project.gcl`) to narrow it further. **Do not** add `@permission("public")` unless the user explicitly asked for an anonymous-access endpoint; making a write-capable endpoint public exposes it to every caller on the network.
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

### Creating users

`greycat run` boots the runtime - creating `gcdata/` and the user database on first use - and can execute any
function, including a stdlib static. So the blessed one-shot way to create a login user is a single command,
no server and no setup wrapper needed:

```sh
greycat run runtime::Identity::create alice user         # user "alice", role "user"
greycat run runtime::Identity::set_password alice s3cret
greycat token --user=alice                               # mint their token
```

Arguments after the function are JSON-parsed and bound to the parameters (see [cli.md](cli.md)), so `alice` /
`user` bind to `Identity::create(name: String, role: String)`. The role must already exist - `user` and
`admin` are built in; declare custom ones with `@role` in `project.gcl`.

For several users at once, wrap the calls in a function and run that:

```gcl
@expose
fn setup_users() {
    Identity::create("alice", "user");
    Identity::set_password("alice", "...");
}
```

```sh
greycat run setup_users
```

The test harness in `test/server/` of this repo is a working reference.

## Schema / type evolution

The runtime stores an ABI snapshot in `gcdata/abi` next to the program. On rebuild, the compiler diffs the new program against the snapshot and migrates compatible changes:

| Change                                                          | Migrates automatically?                                  |
| --------------------------------------------------------------- | -------------------------------------------------------- |
| Add a nullable attribute                                        | Yes — existing instances read it as `null`.              |
| Add a non-null attribute with a default                         | Yes — existing instances get the default.                |
| Add a non-null attribute without default                        | No — load fails. Provide a default or write a migration. |
| Remove an attribute                                             | Yes — value is dropped on next save.                     |
| Rename an attribute                                             | No — looks like remove + add.                            |
| Change an attribute's type to a wider one (e.g. `int` → `int?`) | Yes.                                                     |
| Change an attribute's type to a narrower one                    | No — load fails on a non-conforming value.               |
| Add/remove an enum entry                                        | Add: yes. Remove: only if no instance references it.     |
| Rename a type                                                   | No — looks like a new type.                              |

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

- [ ] `greycat-lang fmt --mode=check` and `greycat-lang lint` both exit `0` on the deploy commit. See [lang.md](lang.md).
- [ ] `greycat build` produces a `project.gcp` artifact (or ship the source tree with `lib/` populated).
- [ ] Ship `project.gcl`, `src/`, `lib/`, `webroot/`, and `bin/greycat` (or have the deploy host run `greycat install` once).
- [ ] Provision a writable `gcdata/` (this is the durable state).
- [ ] Provision a writable `files/` if the app accepts uploads.
- [ ] Configure with `.env` (or systemd `Environment=`): `GREYCAT_PORT`, `GREYCAT_LOG`, `GREYCAT_CACHE`, `GREYCAT_STORE`, `GREYCAT_BACKUP_PATH`, `GREYCAT_TZ`.
- [ ] Schedule backups: either external (snapshot the `gcdata/` directory), or in-process via a scheduled `Runtime::backup_delta()`.
- [ ] Set up monitoring on `Runtime::usage()` if the workload is heavy on memory or storage.

## Common project chores

| Task                             | How                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Pin a new stdlib version         | Bump `@library("std", "X.Y.Z")` in `project.gcl`, then `greycat install`.                                     |
| Add a third-party library        | Add `@library("name", "version");` then `greycat install`.                                                    |
| Roll back to a known graph state | `greycat restore <archive>`. Stop the server first.                                                           |
| Inspect a `.gcb` file from disk  | `greycat print path/to/file.gcb` (use `--format=json` for JSON).                                              |
| Inspect the compiled bytecode    | `greycat bytecode`.                                                                                           |
| Reset everything                 | `rm -rf gcdata/ bin/ lib/` then `greycat install` + `greycat serve`. **Data loss.**                           |
| Add a new user                   | `greycat run runtime::Identity::create alice admin` then `greycat token --user=<name>`. See "Creating users". |
| See what's running               | `Task::running()` from GCL, or hit the corresponding admin endpoint over RPC.                                 |
| Cancel a stuck task              | `Task::cancel(task_id)`.                                                                                      |
| Run a one-shot migration script  | Write a function, then `greycat run my_migration_fn`.                                                         |
