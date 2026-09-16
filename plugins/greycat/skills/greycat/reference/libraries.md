# Available libraries

Libraries published to `get.greycat.io` and pulled into a project with an `@library("<name>", "<version>")` pragma in `project.gcl`, then fetched by `greycat install`. `std` is required; the rest are optional and domain-specific. Reach for one of these before hand-rolling the same capability.

For how `@library` resolves to `lib/<name>/`, the `std` home fallback, and the whole-graph closure, see [project.md](project.md).

## Catalog

| Library       | Pulls in                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `std`         | Required. Core types, collections, time/duration, IO, runtime, util (`Crypto`, `Random`, `Uuid`, sliding windows, ...). |
| `lang`        | Language tooling behind `greycat lint` / `greycat fmt` / `greycat lsp`. Ships with a global GreyCat install; pin it here to hold a project to one toolchain version. See [lang.md](lang.md). |
| `explorer`    | Graph UI + admin tool served at `/explorer` — dev convenience.                                            |
| `http`        | HTTP/HTTPS client over libcurl: `Http<T>` (`get` / `post` / `put` / `getFile` / `send` / `chunked`), `HttpRequest`, `HttpResponse<T>`, `HttpMethod`, and `HttpReader<T>` for SSE / line-delimited streams. Not part of `std`. |
| `ai`          | LLM inference (llama.cpp): `Model`, `LLM`, `ChatMessage`, embeddings, LoRA.                               |
| `algebra`     | `PCA`, `FFT`, neural nets, k-means, time-series decomposition, climate (UTCI).                            |
| `kafka`       | Typed Kafka producer/consumer: `KafkaReader<T>`, `KafkaWriter<T>`, `KafkaConf`.                           |
| `mqtt`        | MQTT pub/sub client: `Mqtt`, `MqttQoS`.                                                                   |
| `opcua`       | OPC UA client: `OpcuaClient` (browse / read / write / subscribe), `OpcuaEvent`, `OpcuaCertificate`.       |
| `ftp`         | FTP/FTPS client: `Ftp`, `FtpEntry`.                                                                       |
| `ssh`         | SSH and SFTP: `Sftp`, `SftpFile`, `SshPasswordAuth`, `SshKeyAuth`.                                        |
| `osm`         | OpenStreetMap toolkit: Overpass API client + Overpass-QL builder, ring math on `geo` / `GeoBox` / `GeoPoly`, opt-in persistent graph (`OsmNode` / `OsmWay` / `OsmRelation`), GPS edge snapping, elevation enrichment. |
| `useragent`   | User-agent string parsing: `UserAgent::parse`.                                                            |
| `finance`     | IBAN parsing / validation (ISO 13616): `Iban::parse`.                                                     |
| `powerflow`   | Power flow analysis (Newton–Raphson): `PowerNetwork`, `PowerBusResult`, `PowerLineResult`.                |
| `powergrid`   | Successor to `powerflow` — adds load configuration and short-circuit analysis. Same `PowerNetwork` API surface, different namespace. |
| `text_search` | Full-text search: `TextIndex<T>`, 15 search modes including BM25/BM25F, 33-language tokenization, C-accelerated. |
| `fcs`         | Reader for Flow Cytometry Standard (FCS) files: `FcsReader`, `FcsMeta`, `FcsChannel`.                     |
| `ifc`         | IFC (Industry Foundation Classes) BIM reader: `IfcReader`, `IfcEntity`. *(Pro license required.)*         |
| `sql`         | PostgreSQL client: `Postgres`, transactions, COPY. *(Pro license required.)*                              |
| `openid`      | OIDC single sign-on: `OidcProvider`, `Openid`, redirect + PKCE flow. *(Pro license required.)*            |

Run `greycat install` after editing `project.gcl` to fetch/refresh the resolved versions into `<project>/lib/<name>/`.

## After install, load the library's own skill

`greycat install` unpacks a library into `<project>/lib/<name>/`. Most libraries
ship their own agent skill there, and it is the authoritative reference for that
library:

- `lib/<name>/skills/SKILL.md` - the library skill. Read it into context before
  writing any code against the library. It is written against the exact version
  installed, so it supersedes the one-line summary in the catalog above and
  anything inferred from the type names.
- `lib/<name>/README.md` - prose overview. Read it when there is no skill.
- `lib/<name>/<name>.gcl` - the declarations themselves; every type and method
  carries a `///` doc comment. The last resort, and always the ground truth.

A library may also carry `lib/<name>/webroot/`, static assets served at `/` alongside the
project's own `webroot/`. See [runtime.md](runtime.md) for the resolution order.

`std` follows the same layout: the installed copy carries `lib/std/skills/SKILL.md`
plus a `skills/reference/` directory beside it - this very file is one of its pages.

Not every entry in the catalog is a GCL library: some ship only static assets and
`lang` installs a shared object, so there is no skill to read for those.

## Branch and version

Every library is published on two branches: **`stable`** (the default - use it unless you have a reason not to) and **`dev`**. Keep every `@library` on the same branch as `std`: if `std` is pinned to a `dev` version, pin the other libraries to `dev` too. Mixing branches across libraries is advanced usage; 99% of projects keep them uniform, following `std`.

### Let `greycat install` resolve the version

Do not resolve a version by hand and paste it into `project.gcl`. `greycat install --bump` rewrites every pin in the file to the newest release and installs it in one step, and it moves every library at once, which is what keeps them on one branch:

```bash
greycat install --bump                     # newest patch of each pin, same branch it already names
greycat install --bump=latest              # newest release on the branch, any major.minor
greycat install --bump --branch=stable     # move the whole project onto stable
greycat install --bump --dry-run           # preview; project.gcl is not touched
```

`--branch` requires `--bump`. Adding a library still means writing the pragma first. A placeholder version needs the `latest` bound to move, because the default `patch` bound refuses to leave the `major.minor` the placeholder names:

```gcl
@library("kafka", "0.0.0-dev");   // then: greycat install --bump=latest
```

Give the placeholder the branch suffix the rest of the project uses (`-dev` above), or pass `--branch` - without one there is no branch to search.

Full semantics (bounds, branch switching, why a pin was skipped) in [cli.md, `greycat install`](cli.md#greycat-install).

### Reading a version without touching the project

`curl` on the `latest` marker is still the way to answer "what is the newest X" - for a report, for a non-GCL artifact like the web SDK tarball ([webapp.md](webapp.md)), or when the project is not on this machine:

```bash
curl https://get.greycat.io/files/<lib>/stable/latest   # -> 7.8/7.8.25-stable   (default branch)
curl https://get.greycat.io/files/<lib>/dev/latest      # -> 8.0/8.0.39-dev
```

The portion after the `/` (e.g. `7.8.25-stable`) is the version string used in `@library("<lib>", "<version>")`. **Exception:** `std` is published under URL path `core` - fetch with `https://get.greycat.io/files/core/stable/latest`.

A machine can point the CLI at another registry, and store that registry's token once, through `GREYCAT_REGISTRY` or the `~/.greycat/registries` file; see [cli.md](cli.md#registries).
