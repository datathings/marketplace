# Streams

`io::Stream<T>` is a named, append-only file of `T` records that any number of tasks write at once and a callback drains in batches. Reach here to ingest a high-rate flow of small records (events, telemetry, audit entries) without paying one graph transaction per record, to read the runtime log or metrics from GCL, or to follow a stream of a GreyCat server from another one over HTTP.

## Contents

- How a stream works
- Register, write, drain
- Buffering and durability
- CSV streams
- Built-in streams: log and metrics
- Reading a stream over HTTP
- Pitfalls

## How a stream works

- A stream belongs to the user who registers it, under a name unique for that user. `/` groups names, as in `"ingest/events"`. Removing that user stops the stream (see [identity.md](identity.md#grants-and-per-user-files)).
- Its records live in `files/<user>/streams/<name>.ndjson`, one JSON object per line (`.csv` and one CSV line for a CSV stream). The file survives restarts. The registry of streams does not, so a program registers its streams at every start.
- `write(v)` appends a record to an in-memory buffer, which reaches the file in batches. Each record starts at a byte offset of the file that never changes.
- Once records reach the file, the stream runs its callback as a task, at most `max_dephasing` later. A burst of writes therefore costs one callback.
- The callback reads the records from the offset it consumed last. The stream does not track that offset, so the consumer keeps it in the graph next to what the records produced.

## Register, write, drain

```gcl
type Event {
    id: int;
    payload: String;
}

type IngestState {
    pos: int;
}

var ingest: node<IngestState?>;
var events: nodeIndex<int, Event>;

fn main() {
    Stream<Event>::new("ingest/events", on_events, 500ms, null);
}

@expose
fn emit(e: Event) {
    Stream<Event>::get("ingest/events").write(e);
}

fn on_events() {
    var s = Stream<Event>::get("ingest/events");
    var state = ingest.resolve();
    if (state == null) {
        state = IngestState { pos: 0 };
        ingest.set(state);
    }
    var end = s.last_written_pos();
    var r = s.reader(state.pos);
    while (r.can_read()) {
        var e = r.read();
        events.set(e.id, e);
    }
    state.pos = end;
}
```

- Register in `main`, which `serve` runs at startup. Registering a name whose file already holds records reattaches to it and runs the callback once, so the records written just before the last stop are drained even when no new write arrives.
- `Stream<T>::new(name, callback, max_dephasing, options)` and `Stream<T>::get(name)` are called on the record type. `new` throws when the name is already registered in this process, when it is invalid (empty, `.` or `..` segments, a leading or trailing `/`, a `\`), or when a flush setting is negative. `get` throws when the name is not registered, or registered with another `T`.
- `options` is a `StreamOptions { format, flush_size, flush_step }`, and `null` takes every default.
- The callback is a top-level function without parameters (lambdas do not capture their environment). It runs as a task of the user who registered the stream, with that user's permissions. One stream never runs two callbacks at once, and records that arrive while one runs get a fresh callback after it.
- Take `last_written_pos()` before `reader()`. A record written between the two calls is then read now and again on the next drain, never skipped. Since a drain can see a record twice, keep the handling idempotent, like the `events.set(e.id, e)` above.
- `reader(from)` returns a `Reader<T>` bounded by `last_written_pos()` as of that call, so it only yields complete records.
- `write(v)` takes exactly a `T`. A subtype is refused, since the reader parses every record back as `T`.

## Buffering and durability

The buffer reaches the file when the first of these happens:

- `flush_size` bytes are pending (default 64KB, and `0` writes each record at once).
- `flush_step` has passed since the first pending record (default 1s).
- `reader()`, `last_written_pos()` or `flush()` is called.
- A `GET` or `HEAD` of the file arrives through `/files` (see [Reading a stream over HTTP](#reading-a-stream-over-http)).
- The process stops cleanly.

A killed or crashed process loses the records still in the buffer. A crashed machine also loses the records in the file that the OS had not synced to disk yet. `flush()` writes the buffer and syncs the file, and once it returns every record written before it is on disk. When the stream must survive a machine crash, call `flush()` before persisting the drain offset, since a file cut back below a persisted offset would make the next drain skip the records written after the restart.

A write of the buffer that fails drops the records it held. The failure is logged rather than thrown, the file is cut back to its last complete record, and the next `flush()` throws it.

## CSV streams

```gcl
Stream<Event>::new("ingest/events_csv", on_events_csv, 1s, StreamOptions { format: StreamFormat::csv });
```

A CSV stream writes one line per record with the default `CsvFormat` (`,` separator, no header). A field of a nested type is spread over one column per field, in declaration order. `reader()` returns a `CsvReader<T>` on a CSV stream and a `JsonReader<T>` on an ndjson one, and `format()` tells which.

## Built-in streams: log and metrics

The runtime writes its own records to streams of `root`. A GCL function running as `root` reads them like any other stream, and an HTTP client reads their files (see the next section).

| Stream         | File                                  | Record       | Written when                                  |
| -------------- | ------------------------------------- | ------------ | --------------------------------------------- |
| `log`          | `files/root/streams/log.ndjson`       | `Log`        | always, by every command that opens the store |
| `host_metrics` | `files/root/streams/host_metrics.csv` | `HostMetric` | `GREYCAT_METRICS` names `host`                |
| `task_metrics` | `files/root/streams/task_metrics.csv` | `TaskMetric` | `GREYCAT_METRICS` names `task`                |
| `http_metrics` | `files/root/streams/http_metrics.csv` | `HttpMetric` | `GREYCAT_METRICS` names `http`                |

```gcl
var reader = Stream<TaskMetric>::get("task_metrics").reader(0);
```

A metric stream exists only while its metric is enabled, and `get` throws otherwise. These streams have no callback. Their buffering is set by `--log_flush_size` / `--log_flush_step` and `--metrics_<host|task|http>_flush_size` / `_flush_step`. The record fields, what each metric costs, and dashboard examples are in [observability.md](observability.md).

## Reading a stream over HTTP

The file of a stream is a file of the `/files` API like any other. Its owner, an admin, or an identity granted read on the owner's directory (`greycat user grant <owner> r <reader>`) can read it. A `GET` or `HEAD` of the file of a stream registered in the serving process writes the stream's buffer before it answers, so the answer holds every record written so far.

A client outside the server, another GreyCat server included, follows the stream by polling the file with `Range: bytes=<offset>-`. `offset` starts at 0, or at the `Content-Length` of a `HEAD` to skip the records already there.

- `206 Partial Content` carries the bytes written since `offset`. Consume the complete lines and add their bytes to `offset`. A last line without its `\n` is a record still arriving, which the next poll fetches again.
- `416 Range Not Satisfiable` means nothing is new. Its `Content-Range: bytes */<size>` gives the file size.
- A size below `offset` means the file lost bytes the client already read. Serving the file writes the buffer without syncing it, so a machine crash (not a process crash) can do that. Resume from the new size.
- `bytes=<offset>-<last>` caps the bytes of one poll, as long as the cap stays above the largest record.
- Do not send `If-Range`. Its validator is the file's modification time, which every append changes, so the server would answer the whole file.
- `401` (no valid token), `403` (no read access) and `404` (no such file) do not change when polled again.

```sh
curl -s -H "Authorization: $TOKEN" -H "Range: bytes=$OFFSET-" http://localhost:8080/files/root/streams/log.ndjson
```

From another GreyCat server, the `http` library (`greycat install http`) sends the request. This function copies the new records of the `ingest/events` stream of server A into the graph of server B, and runs again from where it stopped:

```gcl
type Event {
    id: int;
    payload: String;
}

/// Where the tail of the remote stream resumes, kept in the graph next to what it produced.
type RemoteTail {
    offset: int;
}

var events_tail: node<RemoteTail?>;
var events: nodeIndex<int, Event>;

fn pull_events() {
    var state = events_tail.resolve();
    if (state == null) {
        state = RemoteTail { offset: 0 };
        events_tail.set(state);
    }
    var res = Http<String>::send(HttpRequest {
        method: HttpMethod::GET,
        url: "https://server-a.example.com/files/alice/streams/ingest/events.ndjson",
        headers: Map<String, String> {
            "Authorization": System::getEnv("SERVER_A_TOKEN")!!,
            "Range": "bytes=${state.offset}-",
        },
    });
    if (res.status_code == 416) {
        var range = res.headers.get("content-range")!!;
        var size = int::parse(range.slice(range.indexOf('/') + 1, range.size()));
        if (size < state.offset) {
            state.offset = size;
        }
        return;
    }
    if (res.status_code != 206) {
        throw "GET events.ndjson: ${res.status_code} ${res.error_msg}";
    }
    var body = res.content!!;
    var end = body.lastIndexOf('\n');
    if (end < 0) {
        return;
    }
    for (_, line in body.slice(0, end).split('\n')) {
        var e = Json<Event>::parse(line);
        events.set(e.id, e);
    }
    state.offset = state.offset + end + 1;
}
```

`String` offsets are byte offsets, the same unit as `Range`. The `http` library lowercases response header names. Run the function on a period with `Scheduler::add(pull_events, FixedPeriodicity { every: 10s }, null)`. `SERVER_A_TOKEN` holds the token of a user of server A that may read the file.

## Pitfalls

- **`node.set` of an object it already holds throws** (`object is already attached to another node`). Resolve the state once, `set` it only when it is new, and mutate it in place, as the examples above do.
- **A forgotten registration.** After a restart, `get` throws until `main` registers the stream again, and its records stay on disk undrained.
