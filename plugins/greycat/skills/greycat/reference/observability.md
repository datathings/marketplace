# Logging and metrics

What a running GreyCat process reports about itself. Reach here when the task is about log levels, where log records go, `GREYCAT_METRICS`, or the fields of `TaskMetric`, `HostMetric` and `HttpMetric`. The streams that carry them are described in [streams.md](streams.md).

## Contents

- Logging
- Metrics stream
- Task performance records (`TaskMetric`)
- Host performance records (`HostMetric`)
- HTTP request records (`HttpMetric`)
- Where log records go

## Logging

Levels (lowest → highest verbosity): `none`, `error`, `warn`, `info`, `trace`. Set with `--log` or `GREYCAT_LOG`. The levels only gate the log: runtime metrics have their own switch, `GREYCAT_METRICS` (see Metrics stream below).

GCL-side functions (in `lib/std/runtime.gcl`):

```gcl
error("message");
warn("...");
info("...");
trace("...");
```

`println(value)`, `print(value)`, `pprint(value)` are unconditional: they always write to stdout regardless of log level.

### Metrics stream (`GREYCAT_METRICS`)

Metrics are not log records: they have no level, they are never printed, and `--log` does not switch them on. `GREYCAT_METRICS` (`--metrics`) takes a comma-separated list of the metrics to record, `host` (`HostMetric`, every `--metrics_host_step`), `task` (`TaskMetric`, once per task) and `http` (`HttpMetric`, once per HTTP request), for example `GREYCAT_METRICS=host,task,http`. Unset, nothing is recorded and nothing is built. An unknown name is refused at startup. The option applies to `run` and `serve`.

Each metric has its own stream of `root`, one record per line, with no wrapper: the record itself. A `TaskMetric` carries `user_id`, `task_id`, `job_id` and `src` as a log record does.

- `host` -> `files/root/streams/host_metrics.csv`, `HostMetric` records;
- `task` -> `files/root/streams/task_metrics.csv`, `TaskMetric` records;
- `http` -> `files/root/streams/http_metrics.csv`, `HttpMetric` records.

All are CSV streams, buffered by `--metrics_<kind>_flush_size` and `--metrics_<kind>_flush_step` (`<kind>` being `host`, `task` or `http`), and read with `Stream<TaskMetric>::get("task_metrics").reader(from)` and the like. A nested type (`HostMetric.small`, `medium`, `large` and `zones`) is spread over one column per field. A metric's stream exists only while the metric is enabled, so with `host` off `host_metrics.csv` is neither opened nor created, and `Stream<HostMetric>::get("host_metrics")` fails. How streams buffer, read and survive a crash, and how to tail their files over HTTP, is in [streams.md](streams.md).

### Task performance records (`TaskMetric`)

With `task` in `GREYCAT_METRICS`, every task -- RPC calls, calls with a `task` header, scheduled tasks, `greycat run`, and each `await` job -- writes one `TaskMetric` record once it ended and its transaction committed.

**Where to find them.** A record is one line of `files/root/streams/task_metrics.csv`. Its ids and `src` give what a dashboard groups by:

- `task_id` -- the task id (`Task::id`), shared by a task and its `await` jobs;
- `job_id` -- `null` for the task itself, the job's index in its `await` for an `await` job;
- `src` -- the function the task ran, the key for per-endpoint figures;
- `user_id` -- the caller;
- `time` -- when the record was written, right after the task ended.

**Units and scope.** Durations are microseconds, byte counts are bytes, everything else is a count. A task's figures cover all its runs -- a task suspended in an `await` runs once before and once after each suspension, maybe on different workers -- but not its jobs, which log their own record: add a task's jobs (same `task_id`) to get the total work of an `await` tree. Read back from the stream, each record is a `TaskMetric`.

**Fields.**

- `time` -- capture timestamp: when the record was built, right after the task ended and committed;
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
- Memory: `dirty_evictions` > 0, or `cache_evictions` > 0, means the worker's object cache budget (`cache_budget`, that is the per-worker budget of its class, see `--cache_ratio_small` / `--cache_ratio_large`) is too small for the task; `memory` against `cache_budget` shows how close the task came to it. `dirty_evictions` / `dirty_blocks` is the share of the commit that left memory early.
- Store access: `cache_misses` blocks were loaded, `cache_hits` of them from the zones' binary cache and `read_hits` from disk; `read_wasted` / `read_bytes` is read amplification.
- Exceptions: `catches` counts throws that reached a `catch`; a throw unwinds and builds an `Error`, so a high count on a hot endpoint costs time.
- Outcome: `status` as in `Task::history` -- `ended`, `error` (a `--request_ttl` expiry included) or `cancelled` (a `Task::cancel`).

**Cost.** Every counter is bumped off the hot paths only: on a store read or write, a cache miss or eviction, a caught exception, a suspension, or once per run. Measured against a build without them, RPC throughput and store-heavy tasks are unchanged within noise, and turning `task` on costs no measurable time. It does cost volume: a record is about 550 bytes, so at a high RPC rate it adds up to megabytes per second.

**Dashboard example.** Tasks, total time and memory-starved tasks per function, read from the `task_metrics` stream:

```gcl
/// Tasks, total time and memory-starved tasks per function, from the task metrics.
fn perf_by_function(): Map<String, Array<int>> {
    var by_fn = Map<String, Array<int>> {};
    var reader = Stream<TaskMetric>::get("task_metrics").reader(0);
    while (reader.can_read()) {
        var perf = reader.read();
        if (perf.job_id == null) {
            var key = "${perf.src}";
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
    return by_fn;
}
```

### Host performance records (`HostMetric`)

With `host` in `GREYCAT_METRICS`, the server also writes one `HostMetric` record every `--metrics_host_step` (`GREYCAT_METRICS_HOST_STEP`, default `60s`), the only setting. Each record is one line of `files/root/streams/host_metrics.csv`, with no task, and describes the hosting platform, aggregated:

- `time` -- capture timestamp: when the host was sampled, the end of `period`;
- `period` -- time since the previous record; `cores`, `load` -- the host's cores and one-minute load average;
- `cpu_user`, `cpu_system` -- CPU time the process used during `period` (divide by `period` and `cores` for the share of the machine);
- `os_memory_total` / `os_memory_used`, `process_resident` / `process_virtual` / `process_shared`, `malloc_total`, `memory_drift` -- host and process memory, and what the runtime does not account for;
- `io_read` / `io_write` -- storage bytes the OS accounted to the process during `period`; `store_read` / `store_write` -- bytes the workers moved to and from the store zones;
- `disk_free`, `disk_meta` -- free space where `gcdata/` lives, and the metadata database's size;
- `tasks_live`, `http_connections`, `sse_subscribers` -- task slots in use, open connections, open event streams;
- `http_bytes_in` / `http_bytes_out` -- bytes read from and written to HTTP clients during `period`; `files_served` / `files_pushed` -- files sent in full (`/files` downloads, webroot assets) and received in full (`/files` uploads);
- `small`, `medium`, `large` -- one `HostMetricClass` per worker class: `workers`, `busy` (share of the class's time spent running tasks, 0 to 1), `queued` and `idle` at record time, `ended` / `errors` / `cancelled` / `timeouts` during `period`, and `memory` against `cache_budget`;
- `zones` -- a `HostMetricZones` summary: zones in use, total `size`, `committed_blocks` / `reserved_blocks` / `written_blocks` (`written` over `committed` is fragmentation), `bin_cache`, the most fragmented zone and its ratio, and whether a defrag is selected.

Counters cover `period`; the rest are values at record time. A record is about 300 bytes, about 450 KB a day at the default step. Taking a sample costs well under a millisecond of the serve loop's time, even with 130 workers. It is built from counters and a few syscalls, once per step, by the serve loop: beyond an atomic add per task run and one per ended task, the only per-request work is the HTTP traffic counting: a relaxed add next to each socket read or write into a slot the IO thread owns, so threads never contend. With `host` off the loop does one comparison per second; measured against a build without any of it, RPC throughput and file serving are unchanged within noise. Read back from the stream, each record is a `HostMetric`, and `lib/std/runtime.gcl` documents every field.

### HTTP request records (`HttpMetric`)

With `http` in `GREYCAT_METRICS`, the server writes one `HttpMetric` record per HTTP request once its response ended: `/files` downloads and uploads, webroot assets, RPC calls, `task` calls, event streams and refused requests alike. It is the access log of the server, as one line of `files/root/streams/http_metrics.csv`:

- `time` -- when the response ended; `user_id` -- the user the request was authenticated as, `0` when anonymous;
- `ip`, `forwarded_for`, `user_agent` -- the client address, the `X-Forwarded-For` header (repeated headers joined with `, `, never checked) and the `User-Agent` header. Personal data: each one is recorded only when named in `GREYCAT_METRICS_HTTP_FIELDS` (`--metrics_http_fields=ip,forwarded_for,user_agent`), and is null otherwise. Behind a reverse proxy, `ip` is the proxy's;
- `method`, `path` -- the request line. `method` is a `runtime::HttpMethod` (`GET`, `POST`, ...) and `path` has no query string (which may carry a token or call arguments). For a request whose line could not be read, `method` is null and `path` is empty. `path` is always recorded, and a path under `/files/<user_name>/` names a user. The enum is `private`, so a project's own `HttpMethod` (the `http` library's, for one) keeps the bare name;
- `kind` -- an `HttpMetricKind`: `file` (`GET`, `HEAD` or `DELETE` under `/files` or the webroot), `upload` (`PUT`), `rpc` (a call answered in the response), `task` (a call with the `task` header), `sse` (a task event stream, one record for its whole lifetime) or `other` (`OPTIONS`, the login redirect, a request refused before it was routed);
- `status` -- the HTTP status code, `0` when the connection ended before a response;
- `ttfb`, `duration` -- from the request's first byte to the response's first byte, and to its last byte. `duration` minus `ttfb` is the transfer time, which the client's bandwidth governs on a large download;
- `request_bytes`, `response_bytes` -- wire bytes, request line, status line and headers included. A pipelined request is counted apart from the one it arrived with;
- `task_id` -- the task that ran an `rpc` or `task` call, to join with its `TaskMetric`;
- `aborted` -- the exchange was cut short: the client went away before the response was sent in full, an upload stopped early, or the request ran out of `--request_ttl`. A request refused with a status code is not aborted.

A connection closed before it sent a byte of a request (an idle kept-alive connection, a health probe that only connects) writes nothing.

The stream batches records by `--metrics_http_flush_size` (`GREYCAT_METRICS_HTTP_FLUSH_SIZE`, default `64KB`) and `--metrics_http_flush_step` (`GREYCAT_METRICS_HTTP_FLUSH_STEP`, default `1s`). The request whose record fills the buffer writes it, while the other requests keep appending to a second buffer. `--metrics_http_flush_size=0` writes each record at once, one write per request. A record is about 100 bytes plus its path, and the personal fields when they are on.

The cost is in building each record's line. Measured on a release build against one without the metric, with the default batching: about 3 to 5% fewer requests per second on the smallest `/files` downloads (a 14-byte file, the worst case), and no difference beyond noise on RPC calls. With `--metrics_http_flush_size=0`, about 10% fewer on the smallest downloads. With `http` off, the server does nothing more than a few counter updates per request, and throughput is unchanged within noise.

Requests per status, method and path, read from the `http_metrics` stream:

```gcl
/// Requests per status, method and path, from the HTTP metrics.
@expose
@permission("admin")
fn http_summary(): Map<String, int> {
    var res = Map<String, int> {};
    var reader = Stream<HttpMetric>::get("http_metrics").reader(0);
    while (reader.can_read()) {
        var r = reader.read();
        var method = "-";
        var m = r.method;
        if (m != null) {
            method = type::enum_name(m);
        }
        var key = "${r.status} ${method} ${r.path}";
        res.set(key, (res.get(key) ?? 0) + 1);
    }
    return res;
}
```

### Where log records go

Every record that passes the level check is appended to the `log` stream of `root`, the file `files/root/streams/log.ndjson`, in every command that opens the store (`run`, `serve`, `dev`, `test`, `backup`, ...). That file is always written and is never rotated or truncated.

Records are buffered and reach the file in batches: once `--log_flush_size` bytes are pending (`GREYCAT_LOG_FLUSH_SIZE`, default `64KB`), and at the latest `--log_flush_step` after the first of them (`GREYCAT_LOG_FLUSH_STEP`, default `1s`). A stopping process writes what is left, but a killed or crashed one loses what was still buffered. `--log_flush_size=0` writes each record at once. A read of the file through `/files` writes the buffer first (see [streams.md](streams.md#reading-a-stream-over-http)). With `--log_flush_on_error` (`GREYCAT_LOG_FLUSH_ON_ERROR=true`), an `error` record writes the buffer at once, so the errors that precede a crash reach the file while the other levels stay batched.

stdout receives a copy only:

- when stdout is a TTY - the coloured, human-readable form;
- before the store is open - the JSON form, which is why the `Compiled in ...` and `Upgraded to version ...` startup lines appear even when redirected.

A server whose stdout is redirected (`> out.txt`, systemd, docker, CI) therefore emits those startup lines and then goes quiet. The records are not lost - read the file:

```sh
tail -f files/root/streams/log.ndjson
jq -c 'select(.level == "warn" or .level == "error")' files/root/streams/log.ndjson
jq -c 'select(.src == "app::my_endpoint")' files/root/streams/log.ndjson
```

A record is one JSON object per line, a `Log` (documented in `lib/std/runtime.gcl`). `user_id`, `task_id` and `job_id` identify the caller of a GCL frame and `src` is its innermost non-native function. All four are `null` for records the runtime emits outside any task:

```
{"level":"info","time":"2026-10-05T14:36:49.156936Z","user_id":null,"task_id":null,"job_id":null,"src":null,"data":"Compiled in 6ms410us"}
{"level":"info","time":"2026-10-05T14:36:49.165668Z","user_id":1,"task_id":2,"job_id":null,"src":"app::main","data":"FROM_MAIN"}
{"level":"warn","time":"2026-10-05T14:36:49.165703Z","user_id":1,"task_id":2,"job_id":null,"src":"app::main","data":"A_WARNING"}
```

From GCL, `root` reads it back with `Stream<Log>::get("log").reader(from)`.
