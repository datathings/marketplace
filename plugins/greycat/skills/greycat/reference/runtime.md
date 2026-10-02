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
| `security/`                 | User / role / grant database (LMDB), the server's private key, the root `password` minted on first boot, and a root `token` (valid one year) re-minted on every `serve`. |
| `lock`                      | Process lock — prevents two workers from opening the same store simultaneously. |

**`gcdata/` is the durable state of the application.** Back it up; do not check it into git. Deleting it resets the project to a blank graph.

### Graph-persisted vs transient

A type is graph-persistent (its values can be saved into `gcdata/`) unless it is tagged `@volatile`. Stdlib runtime types like `Log`, `RuntimeInfo`, `HostPerf`, `TaskPerf`, `Identity`, `Task` are `@volatile` — they describe live process state and cannot be stored.

User types holding `nodeTime<T>`, `nodeList<T>`, `nodeIndex<K, V>`, `nodeGeo<T>`, or `node<T>` attributes get persisted lazily as the program writes to those node tags. See [stdlib.md § Node tags](stdlib.md).

### ABI evolution

When you change a type's shape (rename, reorder, change attribute types) and rebuild, the compiler stores a new ABI version next to the existing one and migrates symbols where it can. Incompatible drift causes a load error on next startup — at that point, either fix the source or restore from backup.

## Workers, tasks, and jobs

A **task** is a function call run on a worker thread. Two ways to spawn:

- **HTTP-triggered** — an incoming JSON-RPC / path-RPC call resolves to a function and is enqueued as a task; the response is the task's return value. To dispatch a long-running call as a background task instead of blocking the HTTP response, set request header `task: true` (or `task: small` / `task: regular` / `task: large` to pick the worker class; `true` is `regular`, any other value is answered `400`) — the server returns the `task_id` immediately. Poll status via `Task::is_running(task_id)` or the `Task::running()` / `Task::history()` helpers, then fetch the result from `GET /files/<user_name>/tasks/<task_id>/result.gcb?json` once the task has ended. `greycat call <fn> [args]` does this from the shell, following the task through `Task::events` rather than polling (see [cli.md](cli.md)).
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
- `--workers_small` (`GREYCAT_WORKERS_SMALL`) — task workers for the small class. Every RPC request (path-RPC or JSON-RPC) runs as a small task on them, so long background tasks never starve requests. Added to `--workers`. An RPC task is a task like any other: it has an id, shows in `Task::running()` and `Task::history`, emits SSE task events and can be stopped with `Task::cancel`; only its arguments and result travel over the connection. An `await` inside it suspends it like any task, freeing the worker while its jobs run.
- `--workers_large` — how many of `--workers` serve the large class; the rest serve regular tasks. A worker runs its own class first and, when that queue is empty, any lighter class, so large workers also drain regular and small tasks. Only sync RPC calls run outside `regular` by default (`small`); choose it where the task starts: `Job { function: f, task_class: TaskClass::large }` for `spawn` (default `regular`) and for each `await` job (default: the parent's class), `PeriodicOptions { task_class: TaskClass::large }` for the scheduler (default `regular`), and the `task` header's value for RPC. MCP calls run `regular`. A class with no workers runs in the nearest class that has some; `Task::task_class()` answers the class a task actually runs in.
- `--max_args_memory` (default 1 MiB) — an RPC call whose arguments are larger is streamed to its task's arguments file while it is read, like a `task: true` call, instead of being held in memory for as long as the call runs. Its headers stay readable; `Task::body()` answers `null`. JSON-RPC envelopes stay in memory, since their method and params are only found by parsing them.
- `--request_ttl` — an RPC request still queued or running after this long is cancelled like any task, and answers `503` with an error saying the time-to-live ran out. A `Task::cancel` call says so instead. Background tasks are not subject to it.
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
    for (_, r in rows) {
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
| `GET /runtime::Task::events`   | Server-Sent Events stream of task events for the caller (see [Task events](#task-events-over-sse)). Authenticated callers only.                                             |
| `GET /files/...`               | Read from `<project>/files/`. Per-user subdirectory + ACL.                                                                                                                   |
| `POST/PUT /files/...`          | Write to `<project>/files/`. Triggers any handler registered via `Runtime::on_files_put`.                                                                                    |
| `GET /...` (anything else)     | Static assets, resolved against `<project>/webroot/` then each `lib/<name>/webroot/` (see below). Unknown paths return 404 — no automatic SPA fallback.                       |
| `GET /` with no path           | Serves `webroot/index.html` if present, else a built-in placeholder.                                                                                                         |

### Authentication

Every request resolves to an identity (user). The server reads the token from one of:

- Cookie `greycat=<token>`,
- Header `Authorization: <token>` (no `Bearer` prefix),
- Query parameter `?authorization=<token>` (handy for the boot URL printed on first `serve`, only for GET requests).

No token = anonymous (`user_id = 0`, role `public`). Anonymous callers only reach functions gated by `@permission("public")`.

Tokens are HMAC-signed and live in-memory; restart issues fresh ones. Use `Identity::login(name, pass)` to obtain one, or `greycat token --user=<name>` from the CLI.

### Where arguments come from

A call carries its arguments in the body or in the query string, and which one is used
depends on whether a body is there — not on the method.

| Request | Binds from |
| --- | --- |
| Body present, no query (or a query that names no parameter) | the body |
| Query names a parameter, no body | the query |
| Both | **400** — the call is refused, not resolved by precedence |

A query parameter binds to the parameter of the same name, through the same binder a JSON
object body goes through, so type checking and "a parameter nothing names is null" behave
identically either way. The value's type follows the *parameter's declared type*, not the
shape of the text: `?to=33612345678` is the `String` where `to` is a `String` and the `int`
where it is an `int`. A key that is not a legal identifier binds through `.` and `-` read
as `_`, so `user-id` and `filter.kind` reach parameters named `user_id` and `filter_kind`.
A `+` is a space.

A key naming no parameter is ignored, which is what keeps `?authorization=<token>` and
other non-binding query strings as harmless as they have always been — and is why they do
not trip the conflict above.

Refusing the both-at-once case is deliberate. The two sources can disagree, and any
precedence rule silently discards half of what the caller sent; that failure is invisible
from the outside, where an error is not.

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
| `api`      | Call `@expose`d functions. (`webroot` is public.)  |
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

A handful of stdlib endpoints are `@permission("public")` because they filter their *own* output by the caller's permission mask rather than gating the call: MCP `tools/list` and `runtime::OpenApi::v3` both drop every function the caller could not invoke, so reaching them anonymously discloses only the `public` endpoints. That pattern — public entry point, per-item filtering inside — is the only reason to make a listing endpoint public; a function that returns data rather than a filtered catalogue still needs a real permission.

Identity management at runtime: `Identity::login`, `Identity::token`, `Identity::set_password`, `Identity::create`, `Identity::all`. CLI equivalents under `greycat user`.

### The `--user=<name>` impersonation flag — footgun, do not promote

`greycat serve --user=<name>` (or `GREYCAT_USER=<name>`) makes **every incoming request run as that user, without checking auth**. Anyone who can reach the port executes endpoints with that user's full permission set — there is no "only on localhost" guard. It is **not** a dev convenience to reach for by default. Catastrophic in production; risky on a laptop on a shared / open network.

Do not propose this flag, put it in a recipe, or bake it into a `.env`. The correct dev pattern is:

1. `greycat token --user=<name>` to mint a short-lived token for `<name>` (default `root`), then attach it via `Authorization: <TOKEN>` header or `?authorization=<TOKEN>`.
2. Or call `Identity::login(name, pass)` from the browser / a script to get a cookie-backed session.

The boot URL printed by `greycat serve` on a fresh `gcdata/` already carries a `root` token — that's the intended bootstrap path.

If the user explicitly asks for `--user=<name>` for a one-off local test, fine; never propose it yourself.

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

### Task events over SSE

`GET /runtime::Task::events` keeps the connection open and pushes one frame per task event, instead of the client polling `Task::running` / `Task::history`. The caller must be authenticated (cookie `greycat=<token>` is what a browser `EventSource` sends; `Authorization: <token>` works for `curl` and `fetch`); an anonymous request gets `401`. The response is `Content-Type: text/event-stream` with `Connection: close` and no length: the stream ends with the connection.

Frames, every `data:` line being the `runtime::Task` in the same JSON shape `Task::running` returns:

```
event: task-started
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running"}

event: task-progress
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running","progress":0.5}

event: task-complete
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","completion":"2026-09-28T10:19:11.002Z","status":"ended"}
```

- Send `Accept: application/octet-stream` on the request and every `data:` line is instead the base64 of the binary response form (ABI header, then the GCB `Task`), which is what the web SDK asks for. JSON is the default.
- `task-started` fires once per root task, when its code starts running on a worker; the gap between `creation` and `start` is the time it waited in the queue. A task resuming from an `await` is not reported again, and one cancelled while still queued never starts: it only gets `task-complete`.
- `task-progress` fires when the whole percentage of `Task::add_steps` over `Task::expected_steps` changes, never more often.
- `task-complete` fires once per root task, whatever its final status (`ended`, `error`, `cancelled`). Jobs spawned by `await` are not reported on their own.
- Who receives a frame is decided per task with the rule `Task::running` uses: the task's owner, an admin, or a user the owner granted read access to.
- `: connected` is sent on open and `: ping` every 15 s; both are comments an `EventSource` ignores. A subscriber whose token has expired, or that stops reading, is closed, and so is one that writes anything on the connection after its request: the stream is one-way.
- Every open stream holds a request pool slot for as long as it lives, so `--max_sse` (default 2048) and `--max_sse_per_user` (default 16) cap them; a caller past either gets `429 Too Many Requests` and should poll and retry later.
- There are no `id:` lines, so a reconnecting client cannot resume; fetch `Task::history` to catch up.
- Not available on Windows, where the endpoint answers `503`.

```js
const events = new EventSource("/runtime::Task::events");   // cookie auth
events.addEventListener("task-started", (e) => console.log(JSON.parse(e.data).start));
events.addEventListener("task-complete", (e) => console.log(JSON.parse(e.data)));
events.addEventListener("task-progress", (e) => console.log(JSON.parse(e.data).progress));
```

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

`println(value)`, `print(value)`, `pprint(value)` are unconditional: they always write to stdout regardless of log level.

### Task performance records (`TaskPerf`)

At `perf` and `trace` (`--log=perf`), every task -- RPC calls, `task: true` calls, scheduled tasks, `greycat run`, and each `await` job -- logs one `TaskPerf` record once it ended and its transaction committed. Nothing is built or written at the default `info` level.

**Where to find them.** A record is the `data` of a `Log` of level `perf` in the `log` stream, `files/root/streams/log.ndjson`. The enclosing `Log` gives what a dashboard groups by:

- `task_id` -- the task id (`Task::id`), shared by a task and its `await` jobs;
- `job_id` -- `null` for the task itself, the job's index in its `await` for an `await` job;
- `src` -- the function the task ran, the key for per-endpoint figures;
- `user_id` -- the caller;
- `time` -- when the record was written, right after the task ended.

**Units and scope.** Durations are microseconds, byte counts are bytes, everything else is a count. A task's figures cover all its runs -- a task suspended in an `await` runs once before and once after each suspension, maybe on different workers -- but not its jobs, which log their own record: add a task's jobs (same `task_id`) to get the total work of an `await` tree. Read back from the stream, `data` is a `TaskPerf`: the log tags every object in `data` with its `_type`, and the `log` stream reader rebuilds it (a program that does not define the type gets a `Map` keyed by the field names below).

**Fields.**

- `wait`, `exec`, `run`, `suspended`, `commit` -- time queued before a worker first picked it up; from then to the end; actually on a worker; parked in `await`; and committing.
- `awaits`, `jobs`, `fusion_conflicts` -- `await` suspensions, the jobs they spawned, and jobs whose transaction could not be merged.
- `catches` -- exceptions that reached a `catch`.
- `read_bytes` / `read_hits` / `read_wasted` and `write_bytes` / `write_hits` -- store zone traffic.
- `dirty_blocks`, `dirty_evictions` -- distinct blocks in the committed transaction (the size of the commit), and dirty blocks the object cache had to write out before the commit to stay within budget. A non-zero `dirty_evictions` means memory is missing for the task. (`write_hits` counts batched zone writes, not blocks.)
- `cache_bytes` / `cache_hits` -- blocks served from the zones' binary cache instead of disk.
- `cache_misses`, `cache_evictions` -- blocks the object cache had to load, and blocks it evicted to stay within budget (non-zero: the budget is too small for the task).
- `cache_blocks`, `memory` and `cache_budget` -- the worker's object cache when the task ended, against its budget, to judge whether `--cache` and the worker count fit the workload.
- `task_class`, `borrowed`, `queued` -- its class, runs taken by a heavier class, and times it waited in a queue instead of going straight to an idle worker.
- `args_bytes`, `result_bytes`, `status` -- argument and result sizes, and how it ended.

**Derived signals.**

- Latency seen by a caller: `wait` + `exec`. A high `wait` with `queued` > 0 is a saturated pool; with `borrowed` > 0 the task's class had no idle worker and a heavier one took it.
- Where `exec` goes: `run` on a worker, of which `commit` committing, and `suspended` parked in `await`. `exec` = `run` + `suspended`.
- Memory: `dirty_evictions` > 0, or `cache_evictions` > 0, means the worker's object cache budget (`cache_budget`, that is `--cache` divided by the worker count) is too small for the task; `memory` against `cache_budget` shows how close the task came to it. `dirty_evictions` / `dirty_blocks` is the share of the commit that left memory early.
- Store access: `cache_misses` blocks were loaded, `cache_hits` of them from the zones' binary cache and `read_hits` from disk; `read_wasted` / `read_bytes` is read amplification.
- Exceptions: `catches` counts throws that reached a `catch`; a throw unwinds and builds an `Error`, so a high count on a hot endpoint costs time.
- Outcome: `status` as in `Task::history` -- `ended`, `error` (a `--request_ttl` expiry included) or `cancelled` (a `Task::cancel`).

**Cost.** Every counter is bumped off the hot paths only: on a store read or write, a cache miss or eviction, a caught exception, a suspension, or once per run. Measured against a build without them, RPC throughput and store-heavy tasks are unchanged within noise, and `--log=perf` costs no measurable time. It does cost log volume: a record is about 550 bytes, so at a high RPC rate it adds up to megabytes per second.

**Dashboard example.** Tasks, total time and memory-starved tasks per function, read from the `log` stream:

```gcl
/// Tasks, total time and memory-starved tasks per function, from the `perf` records.
fn perf_by_function(): Map<String, Array<int>> {
    var by_fn = Map<String, Array<int>> {};
    var reader = Stream::get("log", Log).reader(0);
    while (reader.can_read()) {
        var record = reader.read();
        if (record.level == LogLevel::perf && record.data is TaskPerf) {
            var perf = record.data as TaskPerf;
            if (record.job_id == null) {
                var key = "${record.src}";
                var agg = by_fn.get(key) ?? [0, 0, 0, 0];
                agg[0] = agg[0] + 1;
                agg[1] = agg[1] + perf.wait.to(DurationUnit::microseconds);
                agg[2] = agg[2] + perf.exec.to(DurationUnit::microseconds);
                if (perf.dirty_evictions > 0) {
                    agg[3] = agg[3] + 1;
                }
                by_fn.set(key, agg);
            }
        }
    }
    return by_fn;
}
```

### Host performance records (`HostPerf`)

At `perf` and `trace`, the server also logs one `HostPerf` record every `--host_perf_step` (`GREYCAT_HOST_PERF_STEP`, default `60s`), the only setting. It is the `data` of a `perf` `Log` with no task (`task_id` null) and describes the hosting platform, aggregated:

- `period` -- time since the previous record; `cores`, `load` -- the host's cores and one-minute load average;
- `cpu_user`, `cpu_system` -- CPU time the process used during `period` (divide by `period` and `cores` for the share of the machine);
- `os_memory_total` / `os_memory_used`, `process_resident` / `process_virtual` / `process_shared`, `malloc_total`, `memory_drift` -- host and process memory, and what the runtime does not account for;
- `io_read` / `io_write` -- storage bytes the OS accounted to the process during `period`; `store_read` / `store_write` -- bytes the workers moved to and from the store zones;
- `disk_free`, `disk_meta` -- free space where `gcdata/` lives, and the metadata database's size;
- `tasks_live`, `http_connections`, `sse_subscribers` -- task slots in use, open connections, open event streams;
- `http_bytes_in` / `http_bytes_out` -- bytes read from and written to HTTP clients during `period`; `files_served` / `files_pushed` -- files sent in full (`/files` downloads, webroot assets) and received in full (`/files` uploads);
- `http_max_in` / `http_max_out` / `http_max_file` -- the period's largest single request in and out, and largest file served or pushed;
- `top_out`, `top_in`, `top_files` -- the 3 heaviest users of the period by bytes sent to them, by bytes received from them, and by files served and pushed, each a `HostPerfUser` with `user_id`, `bytes_in`, `bytes_out`, `files_served`, `files_pushed`. A request is charged to its user when it ends (an event stream open longer than a period, when it closes); anonymous and refused requests go to user `0`. Each IO thread keeps its 64 heaviest users of the period, so a lighter user beyond that leaves the lists but never the totals;
- `small`, `regular`, `large` -- one `HostPerfClass` per worker class: `workers`, `busy` (share of the class's time spent running tasks, 0 to 1), `queued` and `idle` at record time, `ended` / `errors` / `cancelled` / `timeouts` during `period`, and `memory` against `cache_budget`;
- `zones` -- a `HostPerfZones` summary: zones in use, total `size`, `committed_blocks` / `reserved_blocks` / `written_blocks` (`written` over `committed` is fragmentation), `bin_cache`, the most fragmented zone and its ratio, and whether a defrag is selected.

Counters cover `period`; the rest are values at record time. A record is about 2 KB, about 3 MB a day at the default step. Taking a sample costs well under a millisecond of the serve loop's time, even with 130 workers. It is built from counters and a few syscalls, once per step, by the serve loop: beyond an atomic add per task run and one per ended task, the only per-request work is the HTTP traffic counting: a relaxed add next to each socket read or write into a slot the IO thread owns, so threads never contend, and one lookup in that thread's user table when the request ends. Below `perf` the loop does one comparison per second; measured against a build without any of it, RPC throughput and file serving are unchanged within noise. Read back from the stream, `data` is a `Map<any?, any?>`; `HostPerf` in `lib/std/runtime.gcl` documents every field.

### Where log records go

Every record that passes the level check is appended to `files/root/log.csv`, in every command that opens the store (`run`, `serve`, `dev`, `test`, `backup`, ...). That file is always written and is never rotated or truncated. `--logfile` / `GREYCAT_LOGFILE` is not consulted.

stdout receives a copy only:

- when stdout is a TTY - the coloured, human-readable form;
- before the store is open - the CSV form, which is why the `Compiled in ...` and `Upgraded to version ...` startup lines appear even when redirected.

A server whose stdout is redirected (`> out.txt`, systemd, docker, CI) therefore emits those startup lines and then goes quiet. The records are not lost - read the file:

```sh
tail -f files/root/log.csv
grep '^warn,\|^error,' files/root/log.csv
grep ',app::my_endpoint,' files/root/log.csv
```

A record is `<level>,<timestamp_us>,<caller columns>,<message>`. The caller is the user id, task and function for GCL frames, or `system` for records the runtime emits outside any frame:

```
info,1788271802495956,1,2,0,app::main,FROM_MAIN
warn,1788271802495974,1,2,0,app::main,A_WARNING
info,1788266269187765,,,,system,,Compiled in 3ms390us
```

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
- `webroot/` — public static assets. Served at `/` without authentication. Unknown paths return 404 (no automatic SPA fallback to `index.html` — handle deep links explicitly if your router needs it). `webroot/` is also the recommended bundle output target, in which case it is generated rather than committed — see [webapp.md](webapp.md).

`Runtime::on_files_put(handler)` registers a GCL callback that fires for every successful upload (receives the file path).

### Static asset resolution order

A library can ship assets of its own in `lib/<name>/webroot/`, which are served at `/` exactly like the project's.
A request for a static path is tried against each root in turn and the first one that has the file wins:

1. `<project>/webroot/` - the project always goes first, so it can shadow any path a library ships.
2. `<project>/lib/<name>/webroot/`, one per library that has such a directory, in the order the `@library`
   pragmas are declared in `project.gcl`.

Two consequences worth planning around. A library's assets survive anything done to the project `webroot/`,
so a bundler clearing its own output directory cannot break them. And the set of roots is resolved once when
the server starts: creating a `webroot/` (or reinstalling a library) while the server runs needs a restart to
take effect.

`--webroot` moves only the project root; library roots are always `lib/<name>/webroot/`.

`File::baseDir()` returns the project's `files/` path; `File::userDir()` returns the current caller's subdirectory.

## When something goes wrong

| Symptom                                       | First thing to check                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| Compile error, then "no program found"        | `greycat build` to see the errors.                                             |
| `serve` boots but 403 on every call           | Token missing/expired, or the user needs a stronger permission.                |
| 422 Unprocessable                             | Client SDK is built against an older ABI — regenerate with `greycat codegen`.  |
| Storage grows continuously                    | Check `--defrag_ratio` and run `greycat defrag` manually.                      |
| `serve` won't start: "lock held"              | Another GreyCat process owns `gcdata/lock`. Stop it.                           |
| Nothing logged once `serve` output is redirected | Expected: stdout only carries the TTY form. Read `files/root/log.csv`.       |
| `install` fails to fetch                      | Check version pin in `project.gcl`; verify network access to `get.greycat.io`. |
| `Identity::current()` throws on a public call | Function is not `@permission("public")`; anonymous callers fail it.            |
