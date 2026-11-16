# Tasks, jobs and the scheduler

How GreyCat runs code: every call, scheduled run and `await` job is a task on a worker. Reach here when the task is about worker classes and sizing, background calls and their results, `await` and merge conflicts, transactions, the scheduler, or following tasks over SSE.

## Contents

- Workers, tasks, and jobs
- Merge conflicts on shared nodes
- Tasks and the scheduler
- Task events over SSE

## Workers, tasks, and jobs

A **task** is a function call run on a worker thread. Two ways to spawn:

- **HTTP-triggered** — an incoming JSON-RPC / path-RPC call resolves to a function and is enqueued as a task; the response is the task's return value. To dispatch a long-running call as a background task instead of blocking the HTTP response, set request header `task: true` (or `task: small` / `task: medium` / `task: large` to pick the worker class; `true` is `medium`, any other value is answered `400`) — the server returns the `task_id` immediately. A `@reserved` function (`Task::running`, `Task::cancel`, ...) never takes a task slot. The server ignores the header, answers with the value as it does without it, and adds the response header `task: none`, listed in `Access-Control-Expose-Headers` so that a browser app on another origin can read it. Poll its status with `Task::tasks([task_id])` (see [Task lifecycle](#task-lifecycle)), then fetch its value with `Task::result(task_id)` once the task has ended (see [Task results and folders](#task-results-and-folders)). `greycat call <fn> [args]` does this from the shell, in `medium` unless `--task_class` names another class, following the task through `Task::events` rather than polling (see [cli.md](cli.md)). A call of either kind that carries the request header `sse: <id>` is reported on the event stream with that id (see [Task events](#task-events-over-sse)).
- **Programmatic** — `Scheduler::add(fn, periodicity, ...)` schedules a periodic task; the startup `main()` is enqueued as a task on `serve` boot.

A **job** is a parallel sub-computation kicked off from within a task via `await(...)`. Jobs share the parent task's transaction by default and run in parallel in every task, the root task of `greycat run` included, except under `greycat test`, which runs them one after the other. The task workers (`--workers`, `GREYCAT_WORKERS`) cap how many jobs run at once, so `GREYCAT_WORKERS=1` also runs them one after the other. Every job's `result()` stays readable after later `await`s, until the task ends: the results stay in files of the task (`error.gcb` for a job that threw), and each call reads one, so a large fan-out can be awaited chunk by chunk and read one result at a time. Keep the value instead of calling `result()` twice. A `Job` kept past its task (in a node, say) throws on `result()`. A writer opened before an `await` loses its file across it: written again after, it truncates the file, so write the rest with a new writer created with `append: true`.

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

When the jobs' writes merge back into the parent task, a node or an object field changed on one side only takes that change. An object resolved before the `await` already shows the jobs' merged writes after it, so there is no need to clear it and resolve it again. A `nodeTime`, `nodeIndex` or `nodeGeo` written by several jobs merges key by key, so jobs filling the same series or index with **different keys** never conflict. A `nodeList` keeps what every job appended ([below](#concurrent-appends-to-a-nodelist)). A `nodeSet` never conflicts: the merge applies what each side added and removed ([stdlib.md](stdlib.md#nodesetv)). Nor does a `nodeIndexSet` under the default strategies: its pairs merge one by one, so two values added under one key are both kept, and `removeKey` or `removeAll` removes only the pairs the task saw ([stdlib.md](stdlib.md#nodeindexsetk-v)). The same key, or the same field, written with different values on both sides is a conflict: `await` throws and nothing it merged is kept. A key one side removes and the other sets to a new value is written differently too, also when the removal empties the node: the merge fails with `fusion error, concurrent modification of the same key in <type>, node=<n>, key=<key>, ...`. The same rule applies when a task commits over a concurrent one. A `Map` merges key by key, and a plain object merges field by field, so writes to different keys or fields both land; a `Map` value that is a plain object merges field by field under its key. An `Array` or `String` changed differently on both sides conflicts. `MergeStrategy::no_fusion_fail_on_conflict` refuses the merge of a structured node (`nodeTime`, `nodeIndex`, `nodeSet`, `nodeIndexSet`, `nodeList`, `nodeGeo`) that the task and another task changed at once: the commit, or the `await` that merges the jobs, fails instead. Plain fields and objects still merge. A task can resolve every conflict its commit meets in its own favour by calling `Task::set_merge_strategy(MergeStrategy::fusion_by_key_last_win)`: the value it wrote overrides the one committed before (key by key for a `Map`), instead of failing the commit. It covers every write the task commits, and a task starts with `MergeStrategy::fusion_by_key_fail_on_conflict`. Read [Merge conflicts on shared nodes](#merge-conflicts-on-shared-nodes) before setting it.

**Two gotchas:**

1. **Parallel writes must target different keys** (or different fields of an object), or the batch throws. Jobs appending to one `nodeList` do not collide ([below](#concurrent-appends-to-a-nodelist)).
2. **Lazily created container fields conflict.** Jobs that each create a `null` map field on the same node fail the merge. See [Merge conflicts on shared nodes](#merge-conflicts-on-shared-nodes).

Worker pool sizes:

- `--workers` (`GREYCAT_WORKERS`) — task workers in total. Default = CPU count, see [cli.md](cli.md). They are split into three classes: small (`--workers_small`, default 20%), large (`--workers_large`, default 10%) and medium, which takes the rest and keeps at least one worker.
- `--workers_small` (`GREYCAT_WORKERS_SMALL`) — task workers for the small class. Every RPC request (path-RPC or JSON-RPC) runs as a small task on them, so long background tasks never starve requests. Default 20% of `--workers`, rounded, at least 1 from 2 workers; `run` and `test` start none. An RPC task is a task like any other: it has an id, shows in `Task::running()` and `Task::history`, emits SSE task events when the call names a stream with the `sse` header, and can be stopped with `Task::cancel`; only its arguments and result travel over the connection. An `await` inside it suspends it like any task, freeing the worker while its jobs run.
- `--workers_large` (`GREYCAT_WORKERS_LARGE`) — how many of `--workers` serve the large class, default 10% rounded half up (none below 5 workers). A worker runs its own class first and, when that queue is empty, any lighter class, so large workers also drain medium and small tasks. Only sync RPC calls run outside `medium` by default (`small`); choose it where the task starts: `Job { function: f, task_class: TaskClass::large }` for `spawn` (default `medium`) and for each `await` job (default: the parent's class), `PeriodicOptions { task_class: TaskClass::large }` for the scheduler (default `medium`), and the `task` header's value for RPC (`--task_class` for `greycat call`). MCP calls run `medium`. A class with no workers runs in the nearest class that has some; `Task::task_class()` answers the class a task actually runs in.
- `--cache` (`GREYCAT_CACHE`, default 75% of host memory, see [cli.md](cli.md)) — 1/4 backs the store zones, 3/4 is the object cache of the task workers. It is split by class: a medium worker holds one unit, a small one `--cache_ratio_small` units (default `0.4`) and a large one `--cache_ratio_large` units (default `3`), so with the defaults the small class holds about 8% of the workers' cache and the large one about 30%. A ratio must be > 0. `serve` logs the split at startup, e.g. `Workers: 16 (small 3 x 810.2MiB, medium 11 x 1.98GiB, large 2 x 5.93GiB), cache: 48.0GiB (workers 36.0GiB, zones 12.0GiB)`, and `Runtime::info()` answers it in `RuntimeInfo.small` / `medium` / `large` (`workers`, `mem_worker`, `cache_ratio`).
- `--max_args_memory` (default 1 MiB) — an RPC call whose arguments are larger is streamed to its task's arguments file in `gcdata/tasks/<id>/` while it is read, like a call with a `task` header, instead of being held in memory for as long as the call runs. A function taking a `runtime::HttpRequest` gets such a body as `body_file` ([http.md](http.md#reading-the-request-writing-the-response)). JSON-RPC envelopes stay in memory, since their method and params are only found by parsing them.
- `--request_ttl` — an RPC request still queued or running after this long is cancelled like any task, and answers `503` with an error saying the time-to-live ran out. A `Task::cancel` call says so instead. Background tasks are not subject to it.
- `--http_threads` — IO threads for socket accept / read / write.

### Task lifecycle

`Task::running()` lists every task from the moment it is queued until it takes its final status, in no particular order. `worker_id` is the index of the worker running a task (workers numbered small first, then medium, then large, as many of each as `Runtime::info()` counts), `null` while it waits or is suspended in an `await`. The jobs of an `await` are not listed on their own: a task's `jobs` holds those that have not ended, each with its own `worker_id` and, as its `task_id`, its number among the jobs of the task: the jobs of its first `await` take `0` to `n - 1`, those of the next one go on from `n`. A job paused on a `breakpoint` reads `running`, and the task awaiting it reads `breakpoint`. `Task::tasks(ids)` and the SSE frames carry `worker_id` but never `jobs`, which keeps them light enough to poll: `Task::tasks(ids)` answers the tasks still running and those in the history, and `Task::live(ids)` only whether each one is still in flight. A task the caller may not see reads `null` in the first and `false` in the second. `Task::history(offset, max)` lists the ended tasks newest first, read from `gcdata/history`, without `progress`, `task_class`, `world`, `worker_id` or `jobs`. It is answered over HTTP only, and calling it from GCL throws. Both return only the tasks the caller may see (its own, all of them for a role holding `tasks:read` such as `admin`, and those of identities granting it `tasks:read`, itself or as the holder of a permission, see [identity.md, Grants and per-user files](identity.md#grants-and-per-user-files)). `offset` counts the entries of every user, though, so a caller that does not see every task gets repeats when it pages with `offset + returned` and must deduplicate by `task_id`. `Task::cancel(task_id)` requests cancellation. `TaskStatus` is one of:

| State               | Meaning                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `empty`             | Allocated but not yet enqueued.                                     |
| `waiting`           | Queued, waiting for a worker.                                       |
| `running`           | Executing on a worker.                                              |
| `await`             | Blocked on `await(...)` for child jobs.                             |
| `cancelled`         | Cancelled before completion.                                        |
| `error`             | Threw an uncaught exception.                                        |
| `ended`             | Completed successfully.                                             |
| `ended_with_errors` | Deprecated, removed in v9.x. Never produced.                        |
| `breakpoint`        | Paused at a debugger breakpoint.                                    |
| `committing`        | Returned; its transaction is being committed or rolled back.        |
| `cancelling`        | Cancelled while running or about to start; being stopped.           |
| `await_cancelling`  | Cancelled while blocked on `await(...)`; stopped once its jobs end. |

A task stays `committing` from the moment its code returns until its writes are committed or rolled back, and only then takes its final status. Once a task reads `ended`, its writes are committed: a task started after that reads them.

`Task::cancel` takes effect or changes nothing, decided by the task's status at that instant:

- `waiting`, still queued: it never runs and reads `cancelled` at once.
- `running`, or picked by a worker but not started: it reads `cancelling`, is interrupted (a `try`/`catch` cannot stop that), and ends `cancelled` with its writes rolled back.
- `await`: it reads `await_cancelling` and its jobs are cancelled too. It never resumes its code; once its jobs end it is rolled back and ends `cancelled`.
- `committing` or a final status: too late, its outcome is decided. The task ends as it would have, `ended` or `error`, and `Task::cancel` answers `false`.
- `breakpoint`: ignored while the task is paused.

`Task::cancel` answers `true` whenever it finds the task before `committing`, even when nothing changes (`breakpoint`, or a task already cancelling), so `true` does not mean the task will end `cancelled`. It answers `false` for an unknown id and for a task the caller may not cancel, rather than an error. The caller may cancel its own tasks, all of them for a role holding `tasks:cancel` such as `admin`, and those of identities granting it `tasks:cancel`, itself or as the holder of a permission. An anonymous caller cannot cancel the tasks of anonymous calls, since every anonymous call runs as the same `public` identity and none of them owns those tasks. A cancelled RPC call answers its client `503`.

`cancelling` and `await_cancelling` are still in flight: code that waits for a task to finish waits on them as it does on `running`. `Task::is_running(task_id)` answers `true` for every such in-flight status (`waiting`, `await`, `breakpoint` and `committing` included), and `false` once the task ended. It answers for any id whoever owns it. `Task::live(ids)` asks about several ids at once and answers `false` for the tasks the caller may not see.

### Task results and folders

Each top-level task has a folder `gcdata/tasks/<id>/`, outside `files/`, so nothing in it is served by `/files`.

- `result.gcb` holds the value a task returned, or the error it failed or was cancelled with, written once its transaction committed. `Task::result(task_id)` answers that value as JSON, or as the GCB file itself for a caller that sends `Accept: application/octet-stream`. It throws, as the RPC call of the task would have, the error of a task that failed (`400`) or was cancelled (`503`). It answers `null` for a task that returned nothing, `404` for an id no task the caller may see has (same visibility as `Task::tasks`), `409` while the task has not ended, and `410` once it ended longer ago than `--task_retention`. Called from GCL it answers the same value and throws in the same cases, so a task reads the result of another one, of a `spawn` for example.
- `work/` is `File::workingDir()`, the task's own scratch directory (the awaiting task's for a job), created by the first file written there.
- Everything else (arguments, transaction logs, the snapshot of an `await`, the arguments and results of its jobs as `<job>.<file>`) is read by the server alone, and removed when the task ends or the server starts.

The folder is removed once its task ended longer ago than `--task_retention` (`GREYCAT_TASK_RETENTION`, default `24h`), checked every hour, or every retention period when it is shorter. Copy what should outlive it to `File::userDir()`.

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

## Merge conflicts on shared nodes

Concurrent tasks each run on their own snapshot and commit on their own. A commit that meets a change another task already committed is merged against it, under the rules of [Workers, tasks, and jobs](#workers-tasks-and-jobs). A conflict fails the commit with `incompatible fusion`. The task then ends `error`, its writes roll back, and an HTTP caller gets an error.

The common case is a container field created on first use. Two tasks that run this function for the same card at the same time both see the field missing:

```gcl
type Face {
    lang: String;
    body: String;
}

type Card {
    id: int;
    faces: Map<String, Face>?;
}

var cards: nodeIndex<int, node<Card>>;

fn face(id: int, lang: String): Face {
    var card = cards.get(id)!!.resolve();
    var kept = card.faces?.get(lang);
    if (kept != null) {
        return kept;
    }
    Runtime::sleep(200ms);                // stands for a slow fetch
    var fetched = Face { lang: lang, body: "fetched" };
    card.faces ?= Map<String, Face> {};   // each task sets the field from null
    card.faces.set(lang, fetched);
    return fetched;
}
```

Each task replaces the field `faces` with a map of its own. The field then holds two different values, so the merge fails on the field itself (`incompatible fusion, field=...`), even though the two maps hold different keys. When the map already exists, tasks writing different keys merge key by key, and tasks writing the same key with different values fail with `incompatible fusion, map key=...`.

The fix is to create the container together with its node. No task then changes the field, and each key merges:

```gcl
type Card {
    id: int;
    faces: Map<String, Face>;   // never null: created with the card
}

var cards: nodeIndex<int, node<Card>>;

fn add_card(id: int) {
    cards.set(id, node<Card> { Card { id: id, faces: Map<String, Face> {} } });
}
```

With the field non-null, `face` reads `card.faces.get(lang)` and writes `card.faces.set(lang, fetched)`, with no `?.` and no `?=`.

Turning an existing nullable field non-null leaves the stored `null`s in place, so they need a one-off fill, described in [Schema / type evolution](workflow.md#schema--type-evolution).

For a value that is a cache, where any writer's copy is as good as another's, the writing task can keep its own value with `Task::set_merge_strategy(MergeStrategy::fusion_by_key_last_win)`. It resolves a conflict on a key, such as two tasks writing the same key. It does not resolve a conflict on a field. There the writing task's map replaces the other one as a whole, and the entries the other task wrote are lost without an error. The lazily created field above therefore needs the container created with its node, not this strategy.

**`fusion_by_key_last_win` applies to every write the task commits, not only to the lines near the call.** The strategy belongs to the task, so it covers the writes of every function the task calls, and the position of the call inside the function does not matter. A task that also writes real state (a counter, a balance, an inventory) then overwrites a concurrent change to that state without an error. To use it safely:

- Set it only in a task whose writes are all cache-like, and keep that endpoint narrow.
- Audit every function the task calls, since their writes are included.
- Or give each key a single writer, one task that fills the missing values while every other task only reads them.

### Concurrent appends to a nodeList

Positions of a `nodeList` are not keys two tasks fight over: every concurrent `add` is kept, in merge order. On a commit the task that committed first comes first, in an `await` the job merged first comes first. Under `fusion_by_key_fail_on_conflict` and `fusion_by_key_last_win`, removals and appends never conflict; a value both sides removed is removed once. `MergeStrategy::no_fusion_fail_on_conflict` still refuses any concurrent change of a list. An object value edited in place follows the rules of objects: edits to different fields merge, the same field written differently conflicts.

Keys are reused when a task removes the last values and adds again, or empties the list with `removeAll()` and refills it, and another task removed or edited a value it removed. A scalar that differs from the removed one merges as a removal plus an append. A scalar equal to it reads as untouched, so a concurrent removal removes it silently, under `fusion_by_key` too. An object of the same type reads as an edit of the removed one: when the other task did not add at that key too, the merge fails as an edit against a removal under `fusion_by_key`, and the side merged last wins under `last_win`. When both tasks removed the last object and added one of the same type, the two merge field by field into one value: different fields merge silently (`Item { a: 7, b: 2 }` and `Item { a: 2, b: 9 }` over `Item { a: 2, b: 2 }` give `Item { a: 7, b: 9 }`), the same field fails under `fusion_by_key`, and `last_win` takes it from the side merged last. A merge that succeeds keeps one value where each task added one. Reuse also happens across tasks: a committed tail removal, then another task's `add`, merged against a third task that started before the removal. There the order depends on the kind of value: a scalar is re-keyed after the other side's append, an object of one type is an in-place edit and stays before it.

A position is not an id, so code that stores an index elsewhere uses a `nodeIndex<int, T>` with a counter it owns ([stdlib.md](stdlib.md#nodelistt)).

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

`GET /runtime::Task::events` keeps the connection open and pushes one frame per task event, instead of the client polling `Task::tasks`. The caller must be authenticated (cookie `greycat=<token>` is what a browser `EventSource` sends; `Authorization: <token>` works for `curl` and `fetch`); an anonymous request gets `401`. The response is `Content-Type: text/event-stream` with `Connection: close` and no length: the stream ends with the connection.

The first frame is `connected`, whose `data:` is the id of this stream, an unsigned 64-bit integer in decimal text whatever the `Accept` header. It can exceed what a JavaScript `Number` holds exactly, so a client keeps it as the string it received. Its `run:` line is the number of the server run, which changes at every start and is only needed to [resume the stream](#resuming-a-stream). Then comes `synced` (see there), and the task frames:

```
event: connected
run: 1794053291
data: 4294967303

event: synced
data: 0
```

A stream only reports the tasks of the calls that name it: an RPC call (path-RPC or JSON-RPC, with or without the `task` header) carrying the request header `sse: <id>` is reported on the stream with that id. A call without the header is reported nowhere, so two clients of the same user (two tabs, or two libraries sharing a JS VM) each see their own calls only. The id binds the task the call creates, not the root tasks that task spawns, nor the scheduled ones.

Then one frame per event of a bound task, every `data:` line being the `runtime::Task` in the same JSON shape `Task::running` returns, without `jobs`:

```
event: task-started
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running","task_class":"medium","world":"main","worker_id":3}

event: task-progress
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","status":"running","progress":0.5,"task_class":"medium","world":"main","worker_id":3}

event: task-complete
data: {"user_id":1,"user_name":"root","task_id":12,"mod":"api","fun":"import","creation":"2026-09-28T10:19:09.949Z","start":"2026-09-28T10:19:09.949Z","completion":"2026-09-28T10:19:11.002Z","status":"ended","task_class":"medium","world":"main","worker_id":3}
```

- Send `Accept: application/octet-stream` on the request and every task `data:` line is instead the base64 of the binary response form (ABI header, then the GCB `Task`), which is what the web SDK asks for. JSON is the default. The `connected` and `synced` frames stay plain text.
- `task-started` fires once per task, when its code starts running on a worker; the gap between `creation` and `start` is the time it waited in the queue. A task resuming from an `await` is not reported again, and one cancelled while still queued never starts: it only gets `task-complete`.
- `task-progress` fires when the whole percentage of `Task::add_steps` over `Task::expected_steps` changes, never more often.
- `task-complete` fires once per task, whatever its final status (`ended`, `error`, `cancelled`). Jobs spawned by `await` are not reported on their own.
- `task-breakpoint` fires when the task pauses on a `breakpoint` (in its own code or in one of its `await` jobs), once the worker is paused: `Debug::all` already lists it, so a debugger reacting to the frame can open the session with `Debug::get`. `task-resumed` fires when it goes on, with the status back to what it was: `running`, or `await` when one of its jobs paused. A debug call made while it is paused is not reported.
- A call is only reported when the stream it names belongs to the caller. An id that names another user's stream (`tasks:read` from the role or a grant included), a closed stream, or nothing at all (malformed) does not refuse the call: it runs, reported nowhere. To watch every task the caller may see (an admin console), read `Task::running`, `Task::history` and `Task::live`, whose access rule includes the admin role and the grants.
- The id names one stream for its lifetime. A reconnect opens a new stream with a new id, sent in its own `connected` frame. The calls bound to the old stream are reported nowhere unless the new stream [resumes it](#resuming-a-stream).
- `: ping` is sent every 15 s, a comment an `EventSource` ignores. A subscriber whose token has expired or been revoked (see [identity.md, Tokens](identity.md#tokens)), or that stops reading, is closed, as is one whose peer left without closing the connection (on Linux, once a frame stays unacknowledged for 30 s, so within 45 s), and so is one that writes anything on the connection after its request: the stream is one-way.
- Every open stream holds a request pool slot for as long as it lives, so `--max_sse` (default 2048) and `--max_sse_per_user` (default 16) cap them; a caller past either gets `429 Too Many Requests` and should poll and retry later.
- Not available on Windows, where the endpoint answers `503`.

`EventSource` cannot send request headers, so its automatic reconnect never resumes: a task bound to a stream that dropped is found with `Task::tasks`. The web SDK reads the stream with `fetch` and resumes.

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

#### Resuming a stream

A client that lost its stream (a page reload, a dropped network) opens a new one with the request header `sse-resume: <run>:<id>`, the `run:` and `data:` of the old `connected` frame. The live root tasks (running, waiting, in `await` or on a `breakpoint`) the caller bound to that id move to the new one: each gets one `task-state` frame, its current `Task` in the same form as the other frames (JSON, or base64 GCB), then reports its next events on the new stream. `synced` follows, with the number of tasks moved as its `data:`, before any other task frame:

```
event: connected
run: 1794053291
data: 8589934599

event: task-state
data: {"user_id":2,"user_name":"alice","task_id":12,"mod":"api","fun":"import","status":"running","progress":0.25,...}

event: synced
data: 1
```

- Tasks only move from a closed stream, so two pages never follow the same task. A stream the server still holds open keeps them, and the resume moves nothing.
- `sse-resume: <run>:<id>; claim` closes the old stream first when the server still holds it open, then moves its tasks. It is for the client that held that stream and lost it, such as after a network drop the server has not noticed yet (it may take up to 45 s on Linux, and the kernel's retransmission timeout elsewhere). Never claim an id another page may still use, such as one copied into a new tab.
- Only the caller's own tasks move. An id of another user (admin role and grants included), of another run, `0`, or a malformed value moves nothing, and the stream opens as without the header.
- A task that ended while no stream was open is not reported: look it up with `Task::tasks`. A task already moved to another stream is not moved again.
- Every stream sends `synced`, a fresh one with `data: 0`, so a client knows the server resumes.
