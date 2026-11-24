# Libraries

Reach here when a task needs a capability beyond `std` (a protocol client, a file format, a database, numerical code, LLM inference): find the library that already does it, read its skill, then pull it in with `greycat install <name>`.

Libraries are published to `registry.greycat.io` and pulled into a project with an `@library("<name>", "<version>")` pragma in `project.gcl`, then fetched by `greycat install`. `std` is required, the rest are optional. For how `@library` resolves to `lib/<name>/`, the `std` home fallback, and the whole-graph closure, see [project.md](project.md).

## Contents

- Libraries worth knowing
- Finding a library in the registry
- Reading a library's skill
- Branch and version
- Tiers and tokens

## Libraries worth knowing

The ones most projects reach for, and the ones most often re-invented by hand. The registry holds many more (domain models, device protocols, file formats), so search it before writing an integration from scratch.

| Library       | What it is                                                                                     |
| ------------- | ---------------------------------------------------------------------------------------------- |
| `http`        | HTTP/HTTPS client over libcurl, with SSE and line-delimited streams. Not part of `std`.        |
| `explorer`    | Web UI served at `/explorer` to look inside a running server: graph, tasks, files, users.      |
| `lang`        | The tooling behind `greycat lint`, `fmt` and `lsp`. Pin it to hold a project to one toolchain. See [lang.md](lang.md). |
| `sdk_web`     | The `@greycat/web` SDK a webapp calls the backend with. See [webapp.md](webapp.md).             |
| `xml`, `s3`   | XML tree and streaming reader. S3 client (AWS and compatible) on `http` + `xml`.                |
| `parquet`, `msgpack`, `archive` | Parquet import, MessagePack files, zip and tar archives.                     |
| `sql`         | PostgreSQL client: queries, transactions, COPY.                                                |
| `kafka`, `mqtt`, `opcua` | Kafka producer and consumer, MQTT pub/sub, OPC UA client.                           |
| `text_search` | Full-text index (BM25, 33 languages).                                                          |
| `algebra`     | PCA, FFT, neural nets, clustering, time-series decomposition.                                  |
| `ai`, `ai_client` | Local LLM inference (llama.cpp). LLM providers over HTTP.                                  |
| `openid`      | OpenID Connect single sign-on.                                                                 |
| `python`      | An embedded CPython interpreter that calls GCL and is called from it.                          |

## Finding a library in the registry

The registry is the catalog. `app::kind_packages` searches the names and summaries of every library, most downloaded first. It needs no token:

```bash
curl -sX POST https://registry.greycat.io/app::kind_packages \
  -H 'Content-Type: application/json' -d '["lib","postgres",0,10]' \
  | jq -r '.packages[] | "\(.name)\t\(.tier.name)\t\(.latest.version)\t\(.summary)"'
# sql   community   9.0.19-dev   This library provides an implementation of a Postgres client...
```

The arguments are the kind (`"lib"`), a search word, an offset and a page size (1-100). A `null` search word lists every library, and `total` in the answer counts all the matches. Search by what the library does (`postgres`, `parquet`, `weather`), since a library's name is not always the protocol's (`sql` above).

Each package carries `name`, `summary`, `tier` (see [Tiers and tokens](#tiers-and-tokens)), `default_branch`, `latest.version` and `branches[]`. `registry::get_library` returns the same shape for one name, and `null` when no library has it:

```bash
curl -sX POST https://registry.greycat.io/registry::get_library \
  -H 'Content-Type: application/json' -d '["http"]'
```

## Reading a library's skill

Most libraries ship their own agent skill, written against the exact version that ships it. It is the reference for that library, and supersedes the one-line summaries above and anything inferred from type names.

Before installing, `registry::skill_url` gives the skill of a branch's newest release, readable by anyone whatever the library's tier:

```bash
R=https://registry.greycat.io
U=$(curl -sX POST $R/registry::skill_url -H 'Content-Type: application/json' -d '["http","dev"]' | tr -d '"')
curl -s "$R$U"    # /files/public/docs/lib/http/dev/9.0.21/skills/SKILL.md
```

`null` means that release ships no skill. Its README is still public, at `/files/public/docs/lib/<name>/<branch>/<M.m.p>/README.md`, and a `GET` on that directory lists every `.md` it holds.

After `greycat install`, the same files are on disk:

- `lib/<name>/skills/SKILL.md` - the library skill. Read it into context before writing any code against the library.
- `lib/<name>/README.md` - prose overview. Read it when there is no skill.
- `lib/<name>/<name>.gcl` - the declarations, each with a `///` doc comment. The last resort, and always the ground truth.

A library may also carry `lib/<name>/webroot/`, static assets served at `/` alongside the project's own `webroot/`. See [files.md](files.md) for the resolution order. `std` follows the same layout: the installed copy carries `lib/std/skills/SKILL.md` plus a `skills/reference/` directory beside it, and this very file is one of its pages. `lang` installs a shared object and has no skill.

## Branch and version

A version is `MAJOR.MINOR.PATCH-BRANCH` (`9.0.21-dev`). Keep every `@library` on the same branch as `std`: if `std` is pinned to a `dev` version, pin the other libraries to `dev` too. Mixing branches across libraries is advanced usage, and almost every project keeps them uniform, following `std`.

`registry.greycat.io` serves GreyCat 9 and newer. GreyCat 8 libraries live on `get.greycat.io`, and `greycat install` picks the registry from the major of each pin. See [cli.md](cli.md#two-majors-on-one-machine) for a machine whose global `greycat` is not the major a project pins.

### Let `greycat install` resolve the version

Do not resolve a version by hand and paste it into `project.gcl`. `greycat install --bump` rewrites every pin in the file to the newest release and installs it in one step, and it moves every library at once, which is what keeps them on one branch:

```bash
greycat install --bump                     # newest patch of each pin, same branch it already names
greycat install --bump=latest              # newest release on the branch, any major.minor
greycat install --bump --branch=stable     # move the whole project onto stable
greycat install --bump --dry-run           # preview; project.gcl is not touched
greycat install kafka                      # declare kafka (newest on the std pin's branch) and install
```

Add a library with `greycat install <name>` (or `<name>@<version>`): it writes the `@library` pragma into the entrypoint, then installs. A hand-written placeholder version needs the `latest` bound to move, because the default `patch` bound refuses to leave the `major.minor` the placeholder names:

```gcl
@library("kafka", "0.0.0-dev");   // then: greycat install --bump=latest
```

Give the placeholder the branch suffix the rest of the project uses (`-dev` above), or pass `--branch`, since without one there is no branch to search.

Full semantics (bounds, branch switching, why a pin was skipped) in [cli.md, `greycat install`](cli.md#greycat-install).

### Reading a version without touching the project

`registry::latest_version` answers "what is the newest X", for a report or when the project is not on this machine:

```bash
curl -sX POST https://registry.greycat.io/registry::latest_version \
  -H 'Content-Type: application/json' -d '["http","dev"]'      # -> "9.0.21-dev"
```

The quoted string is the version used in `@library("<lib>", "<version>")`, `std` included. `null` means the branch has no release of that library.

## Tiers and tokens

Every package sits in a tier, which decides who may download its archives. The catalog, the skills and the READMEs are public whatever the tier.

| `tier.name` | `tier.audience` | Who downloads it                                          |
| ----------- | --------------- | --------------------------------------------------------- |
| `toolchain` | `everyone`      | Anyone, no token: `std`, `lang`, `rust`, `sdk_web`.       |
| `community` | `signed_in`     | Any registry account. The default tier.                   |
| `pro`       | `granted`       | Only the accounts the tier is granted to.                 |

`tier.can_read` in a package says whether the caller may download it, so a search sent with the token shows what `greycat install` will be allowed to fetch. A `401` from `install` on a library outside `toolchain` means it needs a token. `greycat login` stores one in `~/.greycat/registry_token`, and `GREYCAT_REGISTRY` points the CLI at another registry (see [cli.md](cli.md#registries)).
