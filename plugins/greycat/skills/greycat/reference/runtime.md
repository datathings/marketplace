# GreyCat runtime

What is alive in a process running `greycat serve` (or `greycat run`). Reach here when the task is about deployment, request lifecycles, tasks, scheduling, the graph store, backups, or how `@expose`d functions actually receive traffic.

## Contents

- Architecture at a glance
- The graph store (`gcdata/`)
- Workers, tasks, and jobs
- The HTTP server
- Identity and permissions
- Tasks and the scheduler
- Backups and many-worlds
- Logging
- File uploads (`/files/`) and static assets (`webroot/`)

## Architecture at a glance

A GreyCat process holds, in one binary:

- **A graph store** — append-friendly object database on disk under `gcdata/`. Survives restart.
- **A task scheduler** — runs functions on a configurable pool of workers, either synchronously (request-driven) or on a periodicity.
- **An HTTP server** — multiplexes JSON-RPC, path-RPC, file IO, static assets, MCP, and OpenAPI on a single port.
- **A bytecode VM** — executes compiled `.gcl` modules.
- **An auth layer** — embedded LMDB store of users / roles / grants; HTTP requests carry a token (cookie or header) that resolves to a user identity.

Everything is in-process. There is no separate database, queue, or web server to deploy.

## The graph store

`gcdata/` is the on-disk graph database. Created on first `serve`/`run`. Contents:

| Entry                       | Meaning                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- |
| `data_<N>.bin`              | Zone files (numbered; one per worker / shard).                                  |
| `meta.bin`, `meta.bin-lock` | Index of zones and live transactions.                                           |
| `program`                   | The compiled program. Updated on each compatible rebuild.                       |
| `abi`                       | ABI snapshot. Stored separately to track type-shape evolution across builds.    |
| `history/`                  | Task history (recent task records — for the `Task::history` API).               |
| `security/`                 | LMDB-backed user / role / grant database, plus the server's private key.        |
| `lock`                      | Process lock — prevents two workers from opening the same store simultaneously. |

**`gcdata/` is the durable state of the application.** Back it up; do not check it into git. Deleting it resets the project to a blank graph.

### Graph-persisted vs transient

A type is graph-persistent (its values can be saved into `gcdata/`) unless it is tagged `@volatile`. Stdlib runtime types like `Log`, `RuntimeInfo`, `RuntimeUsage`, `Identity`, `Task` are `@volatile` — they describe live process state and cannot be stored.

User types holding `nodeTime<T>`, `nodeList<T>`, `nodeIndex<K, V>`, `nodeGeo<T>`, or `node<T>` attributes get persisted lazily as the program writes to those node tags. See [stdlib.md § Node tags](stdlib.md).

### ABI evolution

When you change a type's shape (rename, reorder, change attribute types) and rebuild, the compiler stores a new ABI version next to the existing one and migrates symbols where it can. Incompatible drift causes a load error on next startup — at that point, either fix the source or restore from backup.

## Workers, tasks, and jobs

A **task** is a function call run on a worker thread. Two ways to spawn:

- **HTTP-triggered** — an incoming JSON-RPC / path-RPC call resolves to a function and is enqueued as a task; the response is the task's return value. To dispatch a long-running call as a background task instead of blocking the HTTP response, set request header `task: true` — the server returns the `task_id` immediately. Poll status via `Task::is_running(task_id)` or the `Task::running()` / `Task::history()` helpers, then fetch the result from `GET /files/<user_id>/tasks/<task_id>/result.gcb?json` once the task has ended.
- **Programmatic** — `Scheduler::add(fn, periodicity, ...)` schedules a periodic task; the startup `main()` is enqueued as a task on `serve` boot.

A **job** is a parallel sub-computation kicked off from within a task via `await(...)`. Jobs share the parent task's transaction by default and **only run in parallel inside a task context** — calling `await` from a one-shot `greycat run` script runs them serially.

```gcl
var jobs = Array<Job> { };
for (i, id in user_ids) {
    jobs.add(Job { function: project::process_user, arguments: [id] });
}
try {
    await(jobs, MergeStrategy::strict);
} catch (err) {
    // any job that threw is reachable via its `.result()` accessor
    for (i, job in jobs) { /* inspect job.result() */ }
}
```

`MergeStrategy` controls how node-write conflicts between parallel jobs are resolved when their writes merge back into the parent task:

| Strategy     | Behavior                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------- |
| `strict`     | Default. Any concurrent write to the same node throws. Use when correctness > throughput. |
| `first_wins` | Conflicts resolved in favor of the previously committed value.                            |
| `last_wins`  | Conflicts resolved in favor of the current job's value.                                   |

**Two gotchas:**

1. **Parallel writes must target different nodes** under `strict`, or every batch throws. Partition work by node-id, not by index.
2. **Object references resolved BEFORE `await` are stale AFTER it.** A `node.resolve()` value held across an `await` call still points at the pre-merge revision. Set the variable to `null` before `await`, then re-resolve with `node.resolve()` (or use the `->` shorthand) afterwards.

Worker pool sizes:

- `--workers` (`GREYCAT_WORKERS`) — task workers. Default = CPU count.
- `--req_workers` — request workers (JSON-RPC dispatch threads).
- `--http_threads` — IO threads for socket accept / read / write.

### Task lifecycle

`Task::running()` lists currently executing tasks; `Task::history(offset, max)` reads the recent-task log. `Task::cancel(task_id)` requests cancellation. `TaskStatus` is one of:

| State               | Meaning                                 |
| ------------------- | --------------------------------------- |
| `empty`             | Allocated but not yet enqueued.         |
| `waiting`           | Queued, waiting for a worker.           |
| `running`           | Executing on a worker.                  |
| `await`             | Blocked on `await(...)` for child jobs. |
| `cancelled`         | Cancelled before completion.            |
| `error`             | Threw an uncaught exception.            |
| `ended`             | Completed successfully.                 |
| `ended_with_errors` | Completed, but some child jobs failed.  |
| `breakpoint`        | Paused at a debugger breakpoint.        |

### Transactions and rollback

Every task runs inside an implicit transaction. **Node writes are only committed when the task returns successfully** — an uncaught exception rolls back every change the task made. Useful for "all-or-nothing" workflows:

```gcl
fn import_batch(rows: Array<Row>) {
    for (r in rows) {
        var n = registry.get(r.key);
        n.set(r.value);              // staged, not committed
    }
    if (invariant_violated()) {
        throw "abort";               // rolls back ALL n.set() calls above
    }
}                                    // returns → commits
```

Jobs spawned via `await` join the parent task's transaction by default, so a thrown job error also rolls back the parent task's writes (with `MergeStrategy::strict`).

## The HTTP server

Started by `greycat serve` / `greycat dev`. Routes:

| Route                          | Purpose                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /`                       | JSON-RPC 2.0 entrypoint. `method` is the FQN with `.` separators: `"<module>.<fn>"`, or `"<module>.<Type>.<fn>"` for a static method. Body: `{jsonrpc, method, params, id}`. |
| `POST /<module>::<fn>`         | Path-RPC to a free-standing function. Body: JSON array of positional args (or GCB binary if client sets the GCB content-type).                                               |
| `POST /<module>::<Type>::<fn>` | Path-RPC to an `@expose` static method on a type: three segments (the method's full FQN). E.g. `/runtime::Identity::current_id`, `/openid::Openid::providers`.               |
| `GET /files/...`               | Read from `<project>/files/`. Per-user subdirectory + ACL.                                                                                                                   |
| `POST/PUT /files/...`          | Write to `<project>/files/`. Triggers any handler registered via `Runtime::on_files_put`.                                                                                    |
| `GET /...` (anything else)     | Static assets from `<project>/webroot/`. Unknown paths return 404 — no automatic SPA fallback.                                                                               |
| `GET /` with no path           | Serves `webroot/index.html` if present, else a built-in placeholder.                                                                                                         |

### Authentication

Every request resolves to an identity (user). The server reads the token from one of:

- Cookie `greycat=<token>`,
- Header `Authorization: <token>` (no `Bearer` prefix),
- Query parameter `?authorization=<token>` (handy for the boot URL printed on first `serve`).

No token = anonymous (`user_id = 0`, role `public`). Anonymous callers only reach functions gated by `@permission("public")`.

Tokens are HMAC-signed and live in-memory; restart issues fresh ones. Use `Identity::login(name, pass)` to obtain one, or `greycat token --user=<id>` from the CLI.

### Response shapes

JSON-RPC follows the standard `{jsonrpc, id, result | error}` envelope. Path-RPC returns the raw JSON-encoded result, or the GCB-encoded result when the caller asked for it via `Accept: application/octet-stream`.

Errors: 400 (bad request / params), 403 (forbidden — permission missing), 404 (no such function or file), 422 (ABI mismatch — caller sends a payload incompatible with the server's current ABI), 500 (runtime exception in the function).

## Identity and permissions

The security model is **users × roles × permissions × grants**:

- A **user** has a numeric `id`, unique `name`, and exactly one `role`.
- A **role** is a named bundle of permissions (declared with `@role(name, perm1, perm2, ...)`).
- A **permission** is a named gate (declared with `@permission(name, desc)`).
- A **grant** ties one user's identity to another user's `files/` subdirectory (read, write, or both).

Built-in permissions live in `lib/std/runtime.gcl`:

| Permission | What it grants                                     |
| ---------- | -------------------------------------------------- |
| `public`   | Anonymous access. Default for an anonymous caller. |
| `api`      | Call `@expose`d functions and read `webroot`.      |
| `admin`    | Full administrative access.                        |
| `debug`    | Low-level graph manipulation.                      |

Built-in roles:

| Role     | Grants                            |
| -------- | --------------------------------- |
| `public` | `public`                          |
| `user`   | `public`, `api`                   |
| `admin`  | `public`, `admin`, `api`, `debug` |

`@permission("name")` on a function gates it on that permission. With no `@permission`, an `@expose`d function defaults to requiring `api` — that is the recommended default. Reserve `@permission("public")` for endpoints that genuinely must serve anonymous callers (the `login` endpoint itself, an unauthenticated health probe); never add it to a write-capable endpoint just to avoid wiring up login.

```gcl
@expose
fn list_users(): Array<User> { /*...*/ }         // default: requires "api" (authenticated)

@expose
@permission("admin")
fn restart() { /*...*/ }                          // narrowed: admins only

@expose
@permission("public")
fn ping(): String { return "pong"; }              // anonymous — use only when the endpoint must serve anonymous traffic
```

Identity management at runtime: `Identity::login`, `Identity::token`, `Identity::set_password`, `Identity::create`, `Identity::all`. CLI equivalents under `greycat user`.

### The `--user=<id>` impersonation flag — footgun, do not promote

`greycat serve --user=<id>` (or `GREYCAT_USER=<id>`) makes **every incoming request run as that user, without checking auth**. Anyone who can reach the port executes endpoints with that user's full permission set — there is no "only on localhost" guard. It is **not** a dev convenience to reach for by default. Catastrophic in production; risky on a laptop on a shared / open network.

Do not propose this flag, put it in a recipe, or bake it into a `.env`. The correct dev pattern is:

1. `greycat token --user=<id>` to mint a short-lived token for `<id>` (default `1` = `root`), then attach it via `Authorization: <TOKEN>` header or `?authorization=<TOKEN>`.
2. Or call `Identity::login(name, pass)` from the browser / a script to get a cookie-backed session.

The boot URL printed by `greycat serve` on a fresh `gcdata/` already carries a `root` token — that's the intended bootstrap path.

If the user explicitly asks for `--user=<id>` for a one-off local test, fine; never propose it yourself.

## Tasks and the scheduler

The scheduler lives in `Scheduler::*` (stdlib `runtime.gcl`):

```gcl
Scheduler::add(my_fn, FixedPeriodicity { every: 5min }, null);  // null opts => runs now, then every 5min
Scheduler::list();                           // every scheduled task
Scheduler::activate(my_fn);                  // resume
Scheduler::deactivate(my_fn);                // pause without removing
Scheduler::find(my_fn);                      // PeriodicTask?
```

Periodicities:

| Type                 | Triggers                                                           |
| -------------------- | ------------------------------------------------------------------ |
| `FixedPeriodicity`   | Every `every` duration.                                            |
| `DailyPeriodicity`   | At a wall-clock time-of-day; honors `timezone`.                    |
| `WeeklyPeriodicity`  | On selected `days`, optionally combined with a `DailyPeriodicity`. |
| `MonthlyPeriodicity` | On selected days of the month (`-1` = last day).                   |
| `YearlyPeriodicity`  | On `DateTuple`s within a year.                                     |

`PeriodicOptions { start, max_duration, ... }` further constrains when a task may run.

**`immediate` defaults to `true`.** Passing `null` options (or `immediate: true`) runs the task once right away, then on the periodicity. `main()` itself runs as a task on `serve` boot, so the common "do the work on boot, then every N min" shape double-fires: if `main` calls the same graph-mutating function directly AND registers it with an immediate run, the two executions race and their writes hit the same nodes -> `concurrent modifications`. Let `main` own the initial run and disable the immediate fire:

```gcl
fn main() {
    refresh();   // initial run, inside main's task
    Scheduler::add(refresh, FixedPeriodicity { every: 15min }, PeriodicOptions { immediate: false });
}
```

Inside a task: `Task::id()`, `Task::parentId()`, `Task::expected_steps(n)`, `Task::add_steps(k)`, `Task::no_history(true)` to opt out of the history log.

## Backups and many-worlds

In-process API:

```gcl
Runtime::backup_full();       // full snapshot of gcdata/ into backup_path
Runtime::backup_delta();      // incremental
Runtime::defrag();            // compact zone files
```

CLI equivalents: `greycat backup` and `greycat defrag`.

`GREYCAT_BACKUP_PATH` (default `backup/`) sets the backup destination. `GREYCAT_MAX_BACKUP_FILES` (default `3`) caps retention.

Restore with `greycat restore <archive>`; `--verify` validates the archive before extracting.

**Many-worlds** is the runtime's branching-graph feature: `--worlds=<N>` (`GREYCAT_WORLDS`) opens N parallel graph worlds for simulation / what-if analysis. Worker count is multiplied accordingly; see `gcdata/world_*` after enabling.

## Logging

Levels (lowest → highest verbosity): `none`, `error`, `warn`, `info`, `perf`, `trace`. Set with `--log` or `GREYCAT_LOG`.

GCL-side functions (in `lib/std/runtime.gcl`):

```gcl
error("message");
warn("...");
info("...");
perf("...");
trace("...");
```

Output goes to stdout. `--logfile` also mirrors to a file next to `gcdata/`.

`println(value)`, `print(value)`, `pprint(value)` are unconditional: they always write regardless of log level.

## File uploads and static assets

```
<project>/
├── files/
│   ├── <user_name>/     # one subdirectory per user
│   │   └── ...uploads
│   └── root/            # files for user id=1 (root)
└── webroot/
    └── index.html       # static HTTP root
```

- `files/<user_name>/` — per-user upload area. Reachable via `GET/PUT/POST /files/<user_name>`. Access enforced by user grants — `greycat user grant alice rw root` lets `alice` read/write `files/root/`.
- `webroot/` — public static assets. Served at `/` without authentication. Unknown paths return 404 (no automatic SPA fallback to `index.html` — handle deep links explicitly if your router needs it). `webroot/` is also the recommended bundle output target — see [webapp.md](webapp.md).

`Runtime::on_files_put(handler)` registers a GCL callback that fires for every successful upload (receives the file path).

`File::baseDir()` returns the project's `files/` path; `File::userDir()` returns the current caller's subdirectory.

## When something goes wrong

| Symptom                                       | First thing to check                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| Compile error, then "no program found"        | `greycat build` to see the errors.                                             |
| `serve` boots but 403 on every call           | Token missing/expired, or the user needs a stronger permission.                |
| 422 Unprocessable                             | Client SDK is built against an older ABI — regenerate with `greycat codegen`.  |
| Storage grows continuously                    | Check `--defrag_ratio` and run `greycat defrag` manually.                      |
| `serve` won't start: "lock held"              | Another GreyCat process owns `gcdata/lock`. Stop it.                           |
| `install` fails to fetch                      | Check version pin in `project.gcl`; verify network access to `get.greycat.io`. |
| `Identity::current()` throws on a public call | Function is not `@permission("public")`; anonymous callers fail it.            |
