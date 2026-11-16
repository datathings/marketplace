# The HTTP server

How `greycat serve` answers HTTP. Reach here when the task is about serving HTTPS, or calling `@expose`d functions from a client: routes, JSON-RPC and path-RPC, authentication tokens, how arguments bind, response shapes and status codes, or a function that reads the raw request (a signed webhook) or writes its own response (a download, a redirect, a text answer). Files and static assets are in [files.md](files.md).

## Contents

- Routes
- HTTPS
- Authentication
- Where arguments come from
- Calling a method on an instance
- Response shapes
- Reading the request, writing the response

## Routes

Started by `greycat serve` / `greycat dev`. Routes:

| Route                          | Purpose                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /`                       | JSON-RPC 2.0 entrypoint. `method` is the FQN, as in the path: `"<module>::<fn>"`, or `"<module>::<Type>::<fn>"` for a method. Body: `{jsonrpc, method, params, id}`. |
| `POST /<module>::<fn>`         | Path-RPC to a free-standing function. Body: JSON array of positional args (or GCB binary if client sets the GCB content-type).                                               |
| `POST /<module>::<Type>::<fn>` | Path-RPC to an `@expose` method on a type: three segments (the method's full FQN). E.g. `/runtime::Identity::current_id`, `/openid::Openid::providers`. A method that is not `static` takes its receiver first (see [Calling a method on an instance](#calling-a-method-on-an-instance)). |
| `GET`/`HEAD`/`POST`/`PUT`/`DELETE`/`PATCH /<module>::<fn>` | A function taking a `runtime::HttpRequest`, which answers every one of these methods on its own path (see [Reading the request](#reading-the-request-writing-the-response)). |
| `GET /runtime::Task::events`   | Server-Sent Events stream of the tasks of the calls that name it (see [tasks.md, Task events](tasks.md#task-events-over-sse)). Authenticated callers only.                                     |
| `GET`/`HEAD /files/...`        | Download a file or list a folder of `<project>/files/` (see [files.md](files.md)). |
| `PUT /files/...`               | Upload a file. `POST` is always RPC, never an upload. |
| `DELETE /files/...`            | Delete a file or a folder. |
| `GET /...` (anything else)     | Static assets, resolved against `<project>/webroot/` then each `lib/<name>/webroot/` (see [files.md](files.md)). Unknown paths return 404 — no automatic SPA fallback.                       |
| `GET /` with no path           | Serves `webroot/index.html` if present, else a built-in placeholder.                                                                                                         |

## HTTPS

`greycat serve --tls` (`GREYCAT_TLS=true`, also from `.env`) serves HTTPS on `--port` instead of HTTP: one port, one
protocol. It reads two PEM files under the project directory:

- `gcdata/security/tls/cert.pem`: the certificate chain, the server certificate first, then its intermediates (what a CA
  hands out as `fullchain.pem`).
- `gcdata/security/tls/key.pem`: the private key of that certificate, not encrypted.

A missing file, or a key that does not match the certificate, stops `serve` with the reason. The files are read at
start: a certificate renewed by hand takes a restart, one obtained with `--acme` (below) does not. The server speaks TLS 1.2 and 1.3 and HTTP/1.1 only, and marks the session
cookie `Secure`. A plain HTTP request on that port gets its connection closed, not a redirect.

For a local try, a self-signed certificate:

```sh
mkdir -p gcdata/security/tls && cd gcdata/security/tls
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout key.pem -out cert.pem -days 30 \
  -subj /CN=localhost -addext subjectAltName=DNS:localhost,IP:127.0.0.1
```

### Certificates from Let's Encrypt

`--acme=<domains>` (`GREYCAT_ACME`, comma-separated) has `serve` obtain the certificate of those domains from an ACME
authority, Let's Encrypt by default, and renew it, which turns `--tls` on. Running it agrees to the subscriber
agreement of that authority.

```sh
greycat serve --port=443 --acme=example.com,www.example.com --acme_email=ops@example.com
```

- Each domain must resolve to this server and reach `--port` on port 443 from the internet, directly or forwarded: the
  authority validates it with the tls-alpn-01 challenge, a TLS handshake there. There is no HTTP-01 or DNS-01, so no
  wildcard domain.
- Until the first certificate is issued, a few seconds after start, the server presents a self-signed one.
- The account key is created in `gcdata/security/acme/account.pem` on the first run, and the certificate and its key
  are written to `gcdata/security/tls/cert.pem` and `key.pem`, which the next start reads.
- The certificate is renewed once two thirds of its validity have passed, or when `--acme` names a domain it lacks, and
  the next connections get the new one without a restart. A failed attempt is logged with the reason given by the
  authority (in the `log` stream, see [observability.md](observability.md#where-log-records-go)) and retried after 5 minutes, then twice as
  long each time up to 6 hours.
- `--acme_email` (`GREYCAT_ACME_EMAIL`) is the contact of the account. `--acme_directory` (`GREYCAT_ACME_DIRECTORY`)
  points at another authority: `https://acme-staging-v02.api.letsencrypt.org/directory` for the staging one of Let's
  Encrypt, whose limits are looser while you try the setup.

### Port 443

Browsers, and the tls-alpn-01 challenge of `--acme`, reach port 443, which only a privileged process binds on Linux.
Under systemd, grant it in the unit with `AmbientCapabilities=CAP_NET_BIND_SERVICE` and run as a plain user that owns
the project directory:

```ini
[Service]
User=greycat
WorkingDirectory=/srv/app
ExecStart=/srv/app/bin/greycat serve --port=443 --acme=example.com --acme_email=ops@example.com
AmbientCapabilities=CAP_NET_BIND_SERVICE
Restart=on-failure
```

Do not `setcap cap_net_bind_service=+ep bin/greycat`. Linux then runs the binary in secure-execution mode, where the
loader ignores the `$ORIGIN/../lib/ssl` path that finds the OpenSSL greycat ships in `lib/ssl/`: it fails with an
error such as ``version `OPENSSL_3.2.0' not found``, or runs on the system's OpenSSL when that one is recent enough. Forwarding port 443
to `--port`, with a firewall rule or a load balancer that passes TCP through, works too.

`greycat call` and `greycat print` with `--tls` call `https://localhost:<port>` without checking its certificate, which
names the public host, not localhost. With `--url` the certificate is checked as for any outbound TLS: trust a
self-signed one with `--ca_path=gcdata/security/tls`.

## Authentication

Every request resolves to an identity (user). The server reads the token from one of:

- Cookie `greycat=<token>`,
- Header `Authorization: <token>` (no `Bearer` prefix),
- Query parameter `?authorization=<token>` (handy for the boot URL printed on first `serve`, only for GET requests). The server answers with a redirect that sets the token as the cookie, with the expiry of the token, and drops it from the URL.

No token = anonymous (`user_id = 0`, role `public`). Anonymous callers only reach functions gated by `@permission("public")`.

A browser app on another origin sends its token in the `Authorization` header. Every response allows any origin (`Access-Control-Allow-Origin: *`), and the `OPTIONS` preflight allows `GET`, `POST`, `PUT`, `DELETE` and `PATCH` with any header, `Authorization` included. Cookies are not sent cross-origin, since a browser refuses credentials to a server that answers `*`.

A token carries the user id, an expiry and the id of the token, signed with the server's key (`gcdata/security/private_key`). The server keeps a row for every token it issues, so a token stays valid across restarts and password changes until it expires or is revoked ([identity.md, Tokens](identity.md#tokens)). A revoked token is refused on the next request. The permissions are not in the token. They are read from the user's current role on every request, so a role change applies to open sessions at once. Use `Identity::login(name, pass)` to obtain one, or `greycat token --user=<name>` from the CLI.

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

A function taking a `runtime::HttpRequest` binds neither: it reads both from the request, see
[Reading the request](#reading-the-request-writing-the-response).

Refusing the both-at-once case is deliberate. The two sources can disagree, and any
precedence rule silently discards half of what the caller sent; that failure is invisible
from the outside, where an error is not.

## Calling a method on an instance

An `@expose`d method that is not `static` is called on a receiver, which the call passes as
`this` before the parameters:

| Arguments | Receiver |
| --- | --- |
| JSON array, GCB | the first value: `[{"_type":"api::Counter","n":3}, 2]` |
| JSON object, query string | the key `this`: `{"this": {...}, "k": 2}`, `?this=40960nodeTime&k=2` |
| a single JSON value | the receiver, for a method without parameters |

The receiver is never null and must be an instance of the method's type, or of a subtype: the
call answers **400** otherwise. As in GCL, the override of the receiver's type runs, so
`/api::Shape::area` with a `Square` runs `Square::area`, and an `abstract` method can be called
this way. A node travels as its reference, the string a call returns it as. The permission
checked is the one of the method named in the path.

The web SDK installs such a method on the prototype of its type, so `counter.add(2)` and
`counter.add.spawn(2)` call it on `counter`. `greycat call` takes the receiver as its first
argument (`greycat call api::Counter::add '{"_type":"api::Counter","n":3}' 2`), and the
OpenAPI and MCP schemas list it as a required `this` property.

A caller whose role holds both the `admin` and the `debug` permissions may call **any**
function this way, exposed or not, `native` included, whatever its own `@permission`:
`POST /core::nodeTime::size` with `["40960nodeTime"]` answers the size of that node. Every
other caller is answered **404** for a function that is not exposed.

`runtime::Runtime::abi` lists the exposed functions, whoever calls. Such a caller may also
download the program the server runs from `runtime::Runtime::program`: `gcdata/program` as it
is (the `.gcp` format), every type and function with its signature and code, under an ETag of
its modification time. Any other caller is answered **403**. `greycat call` and
[`greycat repl`](cli.md#greycat-repl) read it to find a function that is not exposed.

## Response shapes

JSON-RPC follows the standard `{jsonrpc, id, result | error}` envelope. Path-RPC returns the raw JSON-encoded result, or the GCB-encoded result when the caller asked for it via `Accept: application/octet-stream`. A function returning a `runtime::HttpResponse` writes the response itself, see [Writing the response](#writing-the-response).

Errors: 400 (bad request or params), 401 (no valid credentials, the caller may log in and retry), 403 (authenticated, but the permission is missing, with an empty body, or the call lands in a world the caller may not use, with a `core::Error` saying so, see [storage.md](storage.md#worlds-branches-of-the-store)), 404 (no such function or file), 422 (ABI mismatch, the caller sends a payload incompatible with the server's current ABI), 500 (runtime exception in the function).

## Reading the request, writing the response

Two `runtime` types hand an exposed function the HTTP exchange itself. Both are `private`, so they are always named
`runtime::HttpRequest` and `runtime::HttpResponse` (the `http` library has client types of the same bare names).

| `runtime::HttpRequest` field | Holds |
| --- | --- |
| `method: HttpMethod` | `runtime::HttpMethod::GET`, `POST`, `PUT`, ... |
| `uri: String` | the path and the query string as sent, percent-encoding intact |
| `query: Map<String, String>` | the query parameters, decoded, under their own keys (`hub.mode` stays `hub.mode`). A repeated key keeps its last value; `uri` has them all |
| `headers: Map<String, String>` | the headers, names lower-cased, a repeated header joined with `, `. `authorization` and `cookie` are left out: they carry the caller's GreyCat token, which the call already runs under |
| `body: String?` | the body, byte for byte as it arrived (not re-encoded, not checked for UTF-8), when it fits in `--max_args_memory` (1 MiB by default) |
| `body_file: File?` | a larger body, in the task's folder `gcdata/tasks/<id>/`, with `body` null. It is deleted when the task ends, unless the function moved it (`File::rename`) |

### Which functions get the request

An `@expose`d function whose **only** parameter is a non-null `runtime::HttpRequest` gets it filled by the server
instead of binding arguments. Next to another parameter, or nullable, it fails the build with `takes
runtime::HttpRequest as its only parameter, not null`. A function that is not exposed takes one like any value, which
is how a helper checks a request it is handed.

- It answers `GET`, `HEAD`, `POST`, `PUT`, `DELETE` and `PATCH` on its own `/<module>::<fn>` path, and reads which one
  in `method`. Every other path keeps its own handling: `PUT` and `DELETE` elsewhere are `/files`, and an ordinary
  function keeps `POST` and `GET`. `OPTIONS` stays the server's CORS preflight answer.
- `@permission` applies as to any exposed function (none means `api`), checked before the body is read. A provider
  calling a webhook holds no GreyCat token: the function is `@permission("public")` and checks the request itself.
- JSON-RPC refuses it with error `-32600` (`<module>::<fn> takes a runtime::HttpRequest: call it on its own path, not
  through JSON-RPC`), since the envelope carries no raw request. MCP lists it as no tool, and `greycat codegen`
  generates no SDK method for it: call it with `fetch` or any HTTP client.
- A background call (the `task` header) saves the request as the task's argument, so the task reads it whenever it
  runs.

A signed webhook checks the HMAC over `body`, the bytes the provider signed. A payload parsed and serialized again is
other bytes, and a re-spaced body fails the check:

```gcl
type Event {
    id: String;
}

var events: nodeList<Event>;

@expose
@permission("public")
fn on_event(req: runtime::HttpRequest): runtime::HttpResponse {
    var body = req.body;
    var sent = req.headers.get("x-hub-signature-256");
    if (body == null || sent == null) {
        return runtime::HttpResponse { status: 401 };
    }
    var expected = "sha256=${Crypto::sha256_hmac_hex(body, System::getEnv("APP_SECRET") ?? "")}";
    if (!Crypto::equals_constant_time(sent, expected)) {
        return runtime::HttpResponse { status: 401 };
    }
    events.add(Json<Event>::parse(body));
    return runtime::HttpResponse { status: 204 };
}
```

The check needs the body in `body`: for a provider whose payloads can pass 1 MiB, raise `--max_args_memory`.

### Writing the response

Any exposed function that declares `runtime::HttpResponse` (or `runtime::HttpResponse?`) as its return type writes the
response it returns, whatever its parameters: `fn report(id: int): runtime::HttpResponse` binds `id` as usual.

| `runtime::HttpResponse` field | Sends |
| --- | --- |
| `status: int?` | the status code, `200` when null |
| `content_type: String?` | the `Content-Type`. When null: `text/plain; charset=utf-8` for a `body`, the type `/files` gives the extension of a `body_file` (`text/csv; charset=utf-8` for `.csv`), none without a body |
| `headers: Map<String, String>?` | headers sent as given, one value per name. One set here replaces the server's (`Content-Type`, `Access-Control-Allow-Origin`, `Set-Cookie`); `Content-Length`, `Transfer-Encoding` and `Connection` are always the server's |
| `body: String?` | the body, its bytes |
| `body_file: File?` | a file sent from disk as `/files` sends one, never loaded into the worker's memory. It may be the request's own `body_file` |

A `HEAD` gets the headers and the length the body would have. A function that returns null answers `204`. A JSON-RPC
call gets the `HttpResponse` as a JSON value in `result`, and a background call's `Task::result` is the object. MCP and
`greycat codegen` leave such a function out, as they do one taking a request.

```gcl
/// The bare challenge as text, read from the query by its own keys.
@expose
@permission("public")
fn on_verify(req: runtime::HttpRequest): runtime::HttpResponse {
    var challenge = "";
    if (req.query.get("hub.mode") == "subscribe"
        && req.query.get("hub.verify_token") == System::getEnv("VERIFY_TOKEN")
    ) {
        challenge = req.query.get("hub.challenge") ?? "";
    }
    return runtime::HttpResponse { content_type: "text/plain; charset=utf-8", body: challenge };
}

/// A download, sent from disk.
@expose
fn report(id: int): runtime::HttpResponse {
    var file = File::open("${File::userDir()}reports/${id}.csv");
    if (file == null) {
        return runtime::HttpResponse { status: 404 };
    }
    var headers = Map<String, String> {};
    headers.set("content-disposition", "attachment; filename=\"report-${id}.csv\"");
    return runtime::HttpResponse { headers: headers, body_file: file };
}

/// A redirect.
@expose
@permission("public")
fn home(): runtime::HttpResponse {
    var headers = Map<String, String> {};
    headers.set("location", "/app/");
    return runtime::HttpResponse { status: 302, headers: headers };
}

/// One path, several methods: a PUT stores the caller's note, of any size, and a GET answers it.
@expose
fn note(req: runtime::HttpRequest): runtime::HttpResponse {
    var path = "${File::userDir()}note.txt";
    if (req.method == runtime::HttpMethod::PUT) {
        if (req.body_file != null) {
            File::rename(req.body_file.path, path);
        } else {
            var w = TextWriter<String> { path: path };
            w.write(req.body ?? "");
            w.flush();
        }
        return runtime::HttpResponse { status: 201 };
    }
    var file = File::open(path);
    if (file == null) {
        return runtime::HttpResponse { status: 404 };
    }
    return runtime::HttpResponse { body_file: file };
}
```

**A `body_file` may name any file the server process can read.** It is not confined to the caller's folder the way
`/files` is, since the function could open the file anyway. Never build its path from the request (a query parameter,
a header, a JSON field) without checking it: `../` in a name sends `gcdata/` or `/etc/` to the caller. An `int` key, as
in `report`, or a path the function fixes, as in `note`, cannot leave the folder.

### Errors and commits

- A function that throws rolls back its writes and answers `400` with the error as JSON, as any exposed function does.
- A returned `HttpResponse` is a return: its writes commit and the task ends `ended`, **whatever `status` says**, `4xx`
  and `5xx` included. A webhook that wants the provider to retry without keeping anything throws, or answers before
  it writes.
- An invalid response answers `500`, `text/plain; charset=utf-8`, saying what was wrong: both `body` and `body_file`
  set, a `status` outside 100 to 599, a body on a `1xx`, `204` or `304`, a `body_file` that cannot be opened or is a
  folder, or a header name or value holding a CR, LF or NUL. The function's writes still commit.
