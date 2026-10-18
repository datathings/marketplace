# GreyCat runtime

What is alive in a process running `greycat serve` (or `greycat run`), and where each part is documented. Reach here for the big picture or when something goes wrong.

## Contents

- Architecture at a glance
- Where each part is documented
- When something goes wrong

## Architecture at a glance

A GreyCat process holds, in one binary:

- **A graph store** — append-friendly object database on disk under `gcdata/`. Survives restart.
- **A task scheduler** — runs functions on a configurable pool of workers, either synchronously (request-driven) or on a periodicity.
- **An HTTP server** — multiplexes JSON-RPC, path-RPC, file IO, static assets, MCP, and OpenAPI on a single port.
- **A bytecode VM** — executes compiled `.gcl` modules.
- **An auth layer** — embedded LMDB store of users / roles / grants; HTTP requests carry a token (cookie or header) that resolves to a user identity.

Everything is in-process. There is no separate database, queue, or web server to deploy.

## Where each part is documented

| Part | Reference |
| --- | --- |
| The graph store (`gcdata/`), ABI evolution, backups, worlds | [storage.md](storage.md) |
| Workers, tasks, `await` jobs, transactions, the scheduler, task events over SSE | [tasks.md](tasks.md) |
| HTTP routes, authentication, argument binding, response shapes | [http.md](http.md) |
| Users, roles, permissions, grants, the `--user` footgun | [identity.md](identity.md) |
| `/files`, per-user folders, uploads, `File`, `webroot/` | [files.md](files.md) |
| Log levels, log records, metrics | [observability.md](observability.md) |
| `Stream<T>` and the files of the log and metric streams | [streams.md](streams.md) |

## When something goes wrong

| Symptom                                       | First thing to check                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| Compile error, then "no program found"        | `greycat build` to see the errors.                                             |
| `serve` boots but 403 on every call           | Token missing/expired, or the user needs a stronger permission.                |
| 422 Unprocessable                             | Client SDK is built against an older ABI — regenerate with `greycat codegen`.  |
| Storage grows continuously                    | Check `--defrag_ratio` and run `greycat defrag` manually.                      |
| `serve` won't start: "lock held"              | Another GreyCat process owns `gcdata/lock`. Stop it.                           |
| Nothing logged once `serve` output is redirected | Expected: stdout only carries the TTY form. Read `files/root/streams/log.ndjson`. |
| `install` fails to fetch                      | Check version pin in `project.gcl`; verify network access to `get.greycat.io`. |
| `Identity::current()` throws on a public call | Function is not `@permission("public")`; anonymous callers fail it.            |
