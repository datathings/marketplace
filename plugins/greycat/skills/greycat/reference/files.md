# Files and static assets

The files a GreyCat server holds and serves: per-user folders under `files/`, reachable over HTTP at `/files/` and from GCL with `File`, and the public static assets of `webroot/`. Reach here to upload or download files, list a folder, react to uploads, control who reads a folder, find a task's result file, or serve a frontend.

## Contents

- Layout
- The `/files` API
- Access
- Downloads: content type, caching, ranges
- Reacting to uploads and downloads
- Task folders
- `File` and `FileWalker` in GCL
- `webroot/` and static assets

## Layout

```
<project>/
├── files/
│   ├── <user_name>/        # one folder per identity, named after it (never its id)
│   │   ├── streams/        # the files of the user's streams, see streams.md
│   │   └── tasks/<id>/     # the result of a task, see Task folders
│   └── root/               # the folder of root, which holds the log and metric streams
└── webroot/                # public static assets, served at /
```

`files/` and `webroot/` are resolved against the process working directory, so run `greycat` from the project root.

## The `/files` API

Every request authenticates as for RPC (`Authorization: <token>` without `Bearer`, or the `greycat` cookie, see [http.md](http.md)).

| Request | Effect | Answer |
| --- | --- | --- |
| `GET /files/<user>/<path>` | Downloads a file | `200`, or `206` for a range (see Downloads) |
| `HEAD /files/<user>/<path>` | Same headers as `GET`, no body | as `GET` |
| `GET /files/<user>/<dir>/` | Lists a folder (the trailing `/` is required) | `200`, a listing |
| `PUT /files/<user>/<path>` | Uploads the body, creating parent folders and overwriting an existing file | `200`, body `true`, `500` when the file cannot be created (a folder holds its path) |
| `DELETE /files/<user>/<path>` | Deletes a file, or a folder with everything in it | `200`, body `true`, `404` when missing |

- `POST` is always an RPC call, so `POST /files/...` answers `404`. Upload with `PUT`.
- A folder path without its trailing `/` answers `301` to the same path with the `/`.
- The query string is not part of the path: `PUT /files/alice/x.csv?v=2` writes `x.csv`. A `?` in a name is sent encoded, as `%3F`.
- A `..` segment answers `404`, and so does a `.` segment followed by more path (`/files/./alice/data.csv`) and a path that starts with `/` after `/files/` (`/files//etc/hostname`, `/files/%2Fetc/hostname`).
- The `files/` root itself lists the folders the caller may open, and can be neither written nor deleted (`403`).

```sh
curl -X PUT -H "Authorization: $TOKEN" --data-binary @readings.csv http://localhost:8080/files/alice/uploads/readings.csv
curl -H "Authorization: $TOKEN" http://localhost:8080/files/alice/uploads/
curl -X DELETE -H "Authorization: $TOKEN" http://localhost:8080/files/alice/uploads/readings.csv
```

A listing is chosen by the `Accept` header. JSON is the default, an array of `File` objects whose `path` is relative to `files/` (a folder's ends with `/` and has no `size`):

```json
[{"path":"alice/uploads/readings.csv","size":8,"last_modification":"2026-10-16T15:28:37.182739Z"}]
```

`Accept: application/octet-stream` returns the same `Array<File>` as GCB, and `Accept: text/html` an HTML page (with upload and delete controls for the owner and admins).

## Access

The first segment after `/files/` names the folder's owner, an identity. A segment that names no identity makes the path admin only.

| Caller | Access to `files/<name>/` |
| --- | --- |
| The identity itself | read and write |
| Any identity with the `admin` role | read and write, on every folder (`files/root/` included), without a grant |
| An identity listed in the `grants` of `<name>` | what the grant says: `read` lists and downloads, `write` uploads and deletes, `read_write` both |
| Anonymous, when the `grants` of `<name>` list `public` | what the grant says. A grant to `public` does not open the folder to signed-in users. |
| Anyone else | `401` when the request carries no valid credentials, `403` when it does |

```sh
greycat user grant bob r alice     # alice may list and download files/bob/
```

How to grant, revoke and use an identity as a group folder is in [identity.md, Grants and per-user files](identity.md#grants-and-per-user-files). No permission is checked on `/files` beyond this table.

The GCL `File` API checks nothing: any function reads and writes any path. An `@expose`d function that takes a path from its caller must check it itself.

## Downloads: content type, caching, ranges

- The `Content-Type` comes from the whole last extension, in any case: `csv` is `text/csv`, `ndjson` and `jsonl` are `application/x-ndjson`, `json` is `application/json`, `txt` is `text/plain`, plus the usual web, image, font, `pdf`, `zip` and video types. An unknown extension, `.gcb` included, is `application/octet-stream`.
- A `.gz` or `.gzip` file whose inner extension has a type is served with that type and `Content-Encoding: gzip`, so the client decodes `data.csv.gz` as CSV. Any other one (`backup.tar.gz`) is `application/gzip`, stored as it is.
- `ETag` is the modification time. `If-None-Match` with it answers `304`. There is no `Last-Modified`.
- `Range: bytes=a-b`, `bytes=a-` or `bytes=-n` answers `206` with `Content-Range: bytes a-b/total`. A single range only: several ranges, or a malformed one, get the whole file. A start past the end answers `416` with `Content-Range: bytes */total`. `If-Range` only matches the `ETag`.
- `?json` at the end of a `.gcb` URL decodes it and answers JSON, one value per line: `GET /files/<user>/tasks/<id>/result.gcb?json`.
- A `GET` or `HEAD` of a stream's file writes the stream's buffer first, which is how a client tails a stream (see [streams.md, Reading a stream over HTTP](streams.md#reading-a-stream-over-http)).

## Reacting to uploads and downloads

`Runtime::on_files_put(handler)` calls `handler(path: String)` after each `PUT` is written to disk, and `Runtime::on_files_get(handler)` after each download of a file under `/files` (not a `HEAD`, a listing, a refusal or a webroot asset). `null` removes the handler. Register them in `main`, since they live in the running server only.

- Each call runs as a background task (class `medium`) of the user who made the request, with that user's permissions. Anonymous requests run as `public`.
- The handler cannot refuse or change the request, which is already answered or about to be.
- Both handlers get the path of the file relative to the project, `files/alice/uploads/readings.csv`, with `/` separators on every platform and without the query string. It opens as it is from GCL.

```gcl
type Reading {
    sensor: String;
    value: float;
}

var readings: nodeList<Reading>;

fn main() {
    Runtime::on_files_put(on_upload);
}

fn on_upload(path: String) {
    if (!path.endsWith(".csv")) {
        return;
    }
    var reader = CsvReader<Reading> { path: path };
    while (reader.can_read()) {
        readings.add(reader.read());
    }
}
```

## Task folders

`files/<user>/tasks/<id>/` holds `result.gcb`, the result of a task started with the `task` header, by `spawn` or by the scheduler. It appears whole, once the task's transaction committed: the task writes `result.gcb.tmp` next to it, then renames it, so a client polling for the file never reads a partial one, nor a result the commit then turned into an error. What the server keeps for a task while it runs (its arguments, the files of its `await` jobs) lives in `gcdata/jobs/` instead, out of reach of `/files` (see [storage.md](storage.md#the-graph-store)). `File::workingDir()` names it from inside the task. Task folders are deleted 24 hours after the task ends, so copy anything worth keeping elsewhere in the user's folder.

## `File` and `FileWalker` in GCL

Paths are relative to the process working directory, like `files/alice/x.csv`.

```gcl
File::baseDir();                 // "files/"
File::userDir();                 // "files/<user_name>/" of the caller
File::workingDir();              // "files/<user_name>/tasks/<id>/" of the current task, not created
File::mkdir("files/alice/archive/2026");                          // creates every missing folder
File::copy("files/alice/data.csv", "files/alice/archive/2026/data.csv");  // the target folder must exist
File::rename("files/alice/data.csv", "files/alice/old.csv");      // replaces an existing target
File::ls("files/alice/", ".csv", true);    // Array<File>: recursive lists files only, non-recursive also folders
File::delete("files/alice/archive/");      // a file, or a folder with everything in it

var f = File::open("files/alice/old.csv")!!;   // File::open answers null when missing, and a folder's path ends with "/"
f.path;   f.size;   f.last_modification;   // size is null for a folder
f.isDir();   f.name();   f.extension();   f.sha256();
```

`FileWalker` lists one folder, not recursively:

```gcl
var walker = FileWalker { path: "files/alice/" };
while (!walker.isEmpty()) {
    var entry = walker.next();   // File?, null once the listing is over
    if (entry != null && !entry.isDir()) {
        println(entry.path);
    }
}
```

The readers and writers that take a `path` (`CsvReader`, `JsonReader`, `TextWriter`, ...) are in [stdlib.md, IO](stdlib.md#io).

## `webroot/` and static assets

- Any `GET` that is neither an RPC call nor under `/files` is a static asset. `webroot/` is public: no authentication, no permission.
- `GET /` serves `webroot/index.html`, or a built-in sign-in page when there is none. A folder path with a trailing `/` serves its `index.html`.
- An unknown path answers `404`. There is no fallback to `index.html`, so a single-page app with client-side routes needs hash routing or one page per route (the stack of [webapp.md](webapp.md) builds one page per route).
- `ETag`, `If-None-Match` and ranges work as for `/files`. A request for a `.js`, `.css`, `.svg`, `.mjs`, `.cjs`, `.html`, `.json` or `.wasm` file is answered from a `.gz` next to it when the client accepts gzip, so a bundler can ship precompressed assets.

A library can ship assets of its own in `lib/<name>/webroot/`, which are served at `/` exactly like the project's. A request for a static path is tried against each root in turn and the first one that has the file wins:

1. `<project>/webroot/`. The project always goes first, so it can shadow any path a library ships.
2. `<project>/lib/<name>/webroot/`, one per library that has such a directory, in the order the `@library` pragmas are declared in `project.gcl`.

A library's assets survive anything done to the project `webroot/`, so a bundler clearing its own output directory cannot break them. The set of roots is resolved once when the server starts, so creating a `webroot/` (or reinstalling a library) while the server runs needs a restart. `--webroot` moves only the project root, and library roots are always `lib/<name>/webroot/`.
