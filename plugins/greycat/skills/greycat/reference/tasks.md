# Tasks, jobs and the scheduler

How GreyCat runs code: every call, scheduled run and `await` job is a task on a worker. Reach here when the task is about worker classes and sizing, background calls, `await` and merge conflicts, transactions, the scheduler, or following tasks over SSE.

## Contents

- Workers, tasks, and jobs
- Tasks and the scheduler
- Task events over SSE

## Workers, tasks, and jobs

A **task** is a function call run on a worker thread. Two ways to spawn:

- **HTTP-triggered** — an incoming JSON-RPC / path-RPC call resolves to a function and is enqueued as a task; the response is the task's return value. To dispatch a long-running call as a background task instead of blocking the HTTP response, set request header `task: true` (or `task: small` / `task: medium` / `task: large` to pick the worker class; `true` is `medium`, any other value is answered `400`) — the server returns the `task_id` immediately. A `@reserved` function (`Task::running`, `Task::cancel`, ...) never takes a task slot. The server ignores the header, answers with the value as it does without it, and adds the response header `task: none`, listed in `Access-Control-Expose-Headers` so that a browser app on another origin can read it. Poll status via `Task::is_running(task_id)` or the `Task::running()` / `Task::history()` helpers, then fetch the result from `GET /files/<user_name>/tasks/<task_id>/result.gcb?json` once the task has ended. `greycat call <fn> [args]` does this from the shell, in `medium` unless `--task_class` names another class, following the task through `Task::events` rather than polling (see [cli.md](cli.md)). A call of either kind that carries the request header `sse: <id>` is reported on the event stream with that id (see [Task events](#task-events-over-sse)).
- **Programmatic** — `Scheduler::add(fn, periodicity, ...)` schedules a periodic task; the startup `main()` is enqueued as a task on `serve` boot.

A **job** is a parallel sub-computation kicked off from within a task via `await(...)`. Jobs share the parent task's transaction by default and **only run in parallel inside a task context** — calling `await` from a one-shot `greycat run` script runs them serially.

```gcl
var jobs = Array<Job> { };
for (i, id in user_ids) {
    jobs.add(Job { function: project::process_user, arguments: [id] });
}
try {
    await(jobs);
} catch (err) {
    // any job that threw is reachable via its `.result()` accessor
    for (i, job in jobs) { /* inspect job.result() */ }
}
```

When the jobs' writes merge back into the parent task, a node or an object field changed on one side only takes that change. A `nodeTime`, `nodeIndex`, `nodeList` or `nodeGeo` written by several jobs merges key by key, so jobs filling the same series or index with **different keys** never conflict. The same key, or the same field, written with different values on both sides is a conflict: `await` throws and nothing it merged is kept. The same rule applies when a task commits over a concurrent one. A `Map` merges key by key, and a plain object merges field by field, so writes to different keys or fields both land; a `Map` value that is a plain object merges field by field under its key. An `Array` or `String` changed differently on both sides conflicts. `MergeStrategy::no_fusion_fail_on_conflict` refuses the merge of a structured node (`nodeTime`, `nodeIndex`, `nodeList`, `nodeGeo`) that the task and another task changed at once: the commit, or the `await` that merges the jobs, fails instead. Plain fields and objects still merge. A task can resolve every conflict its commit meets in its own favour by calling `Task::set_merge_strategy(MergeStrategy::fusion_by_key_last_win)`: the value it wrote overrides the one committed before, key by key, instead of failing the commit. The strategy applies to that task alone, and a task starts with `MergeStrategy::fusion_by_key_fail_on_conflict`.

**Two gotchas:**

1. **Parallel writes must target different keys** (or different fields of an object), or the batch throws. `nodeList::add` appends at the current size, so two jobs appending to the same list collide: give each job its own index range with `set`.
2. **Object references resolved BEFORE `await` are stale AFTER it.** A `node.resolve()` value held across an `await` call still points at the pre-merge revision. Set the variable to `null` before `await`, then re-resolve with `node.resolve()` (or use the `->` shorthand) afterwards.

Worker pool sizes:

- `--workers` (`GREYCAT_WORKERS`) — task workers in total. Default = CPU count. They are split into three classes: small (`--workers_small`, default 20%), large (`--workers_large`, default 10%) and medium, which takes the rest and keeps at least one worker.
- `--workers_small` (`GREYCAT_WORKERS_SMALL`) — task workers for the small class. Every RPC request (path-RPC or JSON-RPC) runs as a small task on them, so long background tasks never starve requests. Default 20% of `--workers`, rounded, at least 1 from 2 workers; `run` and `test` start none. An RPC task is a task like any other: it has an id, shows in `Task::running()` and `Task::history`, emits SSE task events when the call names a stream with the `sse` header, and can be stopped with `Task::cancel`; only its arguments and result travel over the connection. An `await` inside it suspends it like any task, freeing the worker while its jobs run.
- `--workers_large` (`GREYCAT_WORKERS_LARGE`) — how many of `--workers` serve the large class, default 10% rounded half up (none below 5 workers). A worker runs its own class first and, when that queue is empty, any lighter class, so large workers also drain medium and small tasks. Only sync RPC calls run outside `medium` by default (`small`); choose it where the task starts: `Job { function: f, task_class: TaskClass::large }` for `spawn` (default `medium`) and for each `await` job (default: the parent's class), `PeriodicOptions { task_class: TaskClass::large }` for the scheduler (default `medium`), and the `task` header's value for RPC (`--task_class` for `greycat call`). MCP calls run `medium`. A class with no workers runs in the nearest class that has some; `Task::task_class()` answers the class a task actually runs in.
- `--cache` (`GREYCAT_CACHE`, default 75% of RAM) — 1/4 backs the store zones, 3/4 is the object cache of the task workers. It is split by class: a medium worker holds one unit, a small one `--cache_ratio_small` units (default `0.4`) and a large one `--cache_ratio_large` units (default `3`), so with the defaults the small class holds about 8% of the workers' cache and the large one about 30%. A ratio must be > 0. `serve` logs the split at startup, e.g. `Workers: 16 (small 3 x 810.2MiB, medium 11 x 1.98GiB, large 2 x 5.93GiB), cache: 48.0GiB (workers 36.0GiB, zones 12.0GiB)`, and `Runtime::info()` answers it in `RuntimeInfo.small` / `medium` / `large` (`workers`, `mem_worker`, `cache_ratio`).
- `--max_args_memory` (default 1 MiB) — an RPC call whose arguments are larger is streamed to its task's arguments file while it is read, like a call with a `task` header, instead of being held in memory for as long as the call runs. Its headers stay readable; `Task::body()` answers `null`. JSON-RPC envelopes stay in memory, since their method and params are only found by parsing them.
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

Jobs spawned via `await` join the parent task's transaction by default, so a thrown job error also rolls back the parent task's writes.

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

The first frame is `connected`, whose `data:` is the id of this stream, an unsigned 64-bit integer in decimal text whatever the `Accept` header. It can exceed what a JavaScript `Number` holds exactly, so a client keeps it as the string it received:

```
event: connected
data: 4294967303
```

A stream only reports the tasks of the calls that name it: an RPC call (path-RPC or JSON-RPC, with or without the `task` header) carrying the request header `sse: <id>` is reported on the stream with that id. A call without the header is reported nowhere, so two clients of the same user (two tabs, or two libraries sharing a JS VM) each see their own calls only. The id binds the task the call creates, not the root tasks that task spawns, nor the scheduled ones.

Then one frame per event of a bound task, every `data:` line being the `runtime::Task` in the same JSON shape `Task::running` returns:

```
event: task-started
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running","task_class":"medium"}

event: task-progress
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running","progress":0.5,"task_class":"medium"}

event: task-complete
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","completion":"2026-09-28T10:19:11.002Z","status":"ended","task_class":"medium"}
```

- Send `Accept: application/octet-stream` on the request and every task `data:` line is instead the base64 of the binary response form (ABI header, then the GCB `Task`), which is what the web SDK asks for. JSON is the default. The `connected` frame stays plain text.
- `task-started` fires once per task, when its code starts running on a worker; the gap between `creation` and `start` is the time it waited in the queue. A task resuming from an `await` is not reported again, and one cancelled while still queued never starts: it only gets `task-complete`.
- `task-progress` fires when the whole percentage of `Task::add_steps` over `Task::expected_steps` changes, never more often.
- `task-complete` fires once per task, whatever its final status (`ended`, `error`, `cancelled`). Jobs spawned by `await` are not reported on their own.
- `task-breakpoint` fires when the task pauses on a `breakpoint` (in its own code or in one of its `await` jobs), once the worker is paused: `Debug::all` already lists it, so a debugger reacting to the frame can open the session with `Debug::get`. `task-resumed` fires when it goes on, with the status back to `running`. A debug call made while it is paused is not reported.
- A call is only reported when the stream it names belongs to the caller. An id that names another user's stream (admin role and read grants included), a closed stream, or nothing at all (malformed) does not refuse the call: it runs, reported nowhere. To watch every task the caller may see (an admin console), read `Task::running`, `Task::history` and `Task::live`, whose access rule includes the admin role and the grants.
- The id names one stream for its lifetime. A reconnect, such as the automatic one of `EventSource`, opens a new stream with a new id, sent in its own `connected` frame: a client keeps the last id it received, and the calls already bound to the old stream are reported nowhere (fetch `Task::history` to catch up).
- `: ping` is sent every 15 s, a comment an `EventSource` ignores. A subscriber whose token has expired, or that stops reading, is closed, and so is one that writes anything on the connection after its request: the stream is one-way.
- Every open stream holds a request pool slot for as long as it lives, so `--max_sse` (default 2048) and `--max_sse_per_user` (default 16) cap them; a caller past either gets `429 Too Many Requests` and should poll and retry later.
- There are no `id:` lines, so a reconnecting client cannot resume.
- Not available on Windows, where the endpoint answers `503`.

```js
const events = new EventSource("/runtime::Task::events");   // cookie auth
let sse = null;
events.addEventListener("connected", (e) => (sse = e.data));  // again after every reconnect
events.addEventListener("task-started", (e) => console.log(JSON.parse(e.data).start));
events.addEventListener("task-complete", (e) => console.log(JSON.parse(e.data)));
events.addEventListener("task-progress", (e) => console.log(JSON.parse(e.data).progress));

// once `connected` has been received
await fetch("/api::import", { method: "POST", headers: { task: "true", sse }, body: "[]" });
```
