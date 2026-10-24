# The HTTP server

How `greycat serve` answers HTTP. Reach here when the task is about calling `@expose`d functions from a client: routes, JSON-RPC and path-RPC, authentication tokens, how arguments bind, response shapes and status codes. Files and static assets are in [files.md](files.md).

## Contents

- Routes
- Authentication
- Where arguments come from
- Response shapes

## Routes

Started by `greycat serve` / `greycat dev`. Routes:

| Route                          | Purpose                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /`                       | JSON-RPC 2.0 entrypoint. `method` is the FQN with `.` separators: `"<module>.<fn>"`, or `"<module>.<Type>.<fn>"` for a static method. Body: `{jsonrpc, method, params, id}`. |
| `POST /<module>::<fn>`         | Path-RPC to a free-standing function. Body: JSON array of positional args (or GCB binary if client sets the GCB content-type).                                               |
| `POST /<module>::<Type>::<fn>` | Path-RPC to an `@expose` static method on a type: three segments (the method's full FQN). E.g. `/runtime::Identity::current_id`, `/openid::Openid::providers`.               |
| `GET /runtime::Task::events`   | Server-Sent Events stream of the tasks of the calls that name it (see [tasks.md, Task events](tasks.md#task-events-over-sse)). Authenticated callers only.                                     |
| `GET`/`HEAD /files/...`        | Download a file or list a folder of `<project>/files/` (see [files.md](files.md)). |
| `PUT /files/...`               | Upload a file. `POST` is always RPC, never an upload. |
| `DELETE /files/...`            | Delete a file or a folder. |
| `GET /...` (anything else)     | Static assets, resolved against `<project>/webroot/` then each `lib/<name>/webroot/` (see [files.md](files.md)). Unknown paths return 404 — no automatic SPA fallback.                       |
| `GET /` with no path           | Serves `webroot/index.html` if present, else a built-in placeholder.                                                                                                         |

## Authentication

Every request resolves to an identity (user). The server reads the token from one of:

- Cookie `greycat=<token>`,
- Header `Authorization: <token>` (no `Bearer` prefix),
- Query parameter `?authorization=<token>` (handy for the boot URL printed on first `serve`, only for GET requests).

No token = anonymous (`user_id = 0`, role `public`). Anonymous callers only reach functions gated by `@permission("public")`.

A browser app on another origin sends its token in the `Authorization` header. Every response allows any origin (`Access-Control-Allow-Origin: *`), and the `OPTIONS` preflight allows `GET`, `POST`, `PUT` and `DELETE` with any header, `Authorization` included. Cookies are not sent cross-origin, since a browser refuses credentials to a server that answers `*`.

A token carries the user id and an expiry, signed with the server's key (`gcdata/security/private_key`). The server keeps no session list, so a token stays valid until it expires, across restarts and password changes. The permissions are not in the token. They are read from the user's current role on every request, so a role change applies to open sessions at once. Use `Identity::login(name, pass)` to obtain one, or `greycat token --user=<name>` from the CLI.

## Where arguments come from

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

## Response shapes

JSON-RPC follows the standard `{jsonrpc, id, result | error}` envelope. Path-RPC returns the raw JSON-encoded result, or the GCB-encoded result when the caller asked for it via `Accept: application/octet-stream`.

Errors: 400 (bad request or params), 401 (no valid credentials, the caller may log in and retry), 403 (authenticated, but the permission is missing, with an empty body, or the call lands in a world the caller may not use, with a `core::Error` saying so, see [storage.md](storage.md#worlds-branches-of-the-store)), 404 (no such function or file), 422 (ABI mismatch, the caller sends a payload incompatible with the server's current ABI), 500 (runtime exception in the function).
