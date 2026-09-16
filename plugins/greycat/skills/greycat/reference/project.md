# Project model and module system

GreyCat has no `import` / `use` / `include` statements at the source level. Visibility comes from the **project graph**: a single entrypoint, transitive `@library` / `@include` pragmas, and an FQN-based cross-module lookup.

## Contents

- The entrypoint
- `@library` resolution
- `@include` resolution
- Cross-module visibility
- FQN syntax and the `Module::Type::member` lookup
- Multi-project workspaces
- Practical layout

## The entrypoint

Every GreyCat project has a single entrypoint named `project.gcl` at the project root. `@include` pragmas **must** appear in this file — no other module may carry one. `@library` (like `@permission` and `@role`) is accepted in any module of the closure, but the convention is to keep every dependency in the entrypoint, which pins the stdlib version and lists what's part of the project:

```gcl
// project.gcl
@library("std", "1.2.3");
@include("src");
@include("models");
```

The runtime / lang starts loading from the entrypoint. It parses the file, walks every `@library` / `@include` pragma, and loads each module in the closure, following the `@library` pragmas it finds in those modules in turn. A module reached twice is an error (`already declared module`), so `@include` roots must not overlap.

**Never flat-walk a directory for `.gcl` files.** If a file is not reachable from the entrypoint's pragma closure, it is not part of the project. The CLI and LSP both refuse to analyze unreachable files (the LSP reports them as `orphan-module` advisories).

## `@library("name", "version")`

Declares a dependency on a library named `name` at version `version`. Resolution:

1. Look for `<project>/lib/<name>/` and use it if present.
2. For `name == "std"` only: fall back to `<GREYCAT_HOME>/lib/std/` (the runtime install).

A library directory has no entrypoint: every `.gcl` file under `<project>/lib/<name>/` is loaded recursively. A library must therefore not contain a `project.gcl` of its own — it would collide with the consuming project's entrypoint module (`already declared module`). Library modules may carry their own `@library` pragmas, so transitive dependencies resolve and the closure stays whole-graph.

The version string is recorded but not used for resolution-time conflict detection — the loader trusts whatever lives at the resolved path. Mismatches surface as lang errors at use sites (missing types, signature drift).

`@library` is valid in any module of the closure, and `greycat install` resolves the libraries it finds anywhere in that closure. Declare them in `project.gcl` anyway: a pin buried in `src/` is invisible to a reader scanning the entrypoint for the dependency set.

### Versionless `@library("name")`

Omitting the version declares that `lib/<name>/` is the project's own source rather than an installed copy. `greycat install` never fetches it, and `greycat lint` / `greycat fmt` treat its modules like project modules - they are covered by default, with no `--lint-libs` / `--fmt-libs`. This is the shape for a library developed in-tree rather than consumed from the registry.

```gcl
@library("std", "1.2.3");        // installed: fetched by `greycat install`, skipped by lint / fmt
@library("shared");              // owned: lib/shared/ is yours, linted and formatted like src/
```

Only the entrypoint and `@include`d modules confer ownership - a library shipping its own versionless `@library` does not make that name owned in the consuming project. Adding a version pin hands the directory back to `install` and re-suppresses it from lint and fmt.

For the catalog of publishable libraries (what each one pulls in) and how to discover a library's latest version, see [libraries.md](libraries.md). Run `greycat install` after editing `project.gcl` to fetch/refresh the resolved versions into `<project>/lib/<name>/`.

## `@include("path")`

Declares that every `.gcl` file under `<project>/<path>/` (recursively) is part of the project. Files are loaded eagerly.

```gcl
@include("src");                 // loads everything under <project>/src/
@include("models/user");         // loads everything under <project>/models/user/
```

Multiple `@include` pragmas stack additively; their roots must not overlap, since loading a module twice is an error. The argument is a directory path relative to the entrypoint, and `@include` is rejected outside the entrypoint.

## Cross-module visibility

Once the project graph is built, **every declaration in every loaded module is visible from every other module**, modulo the FQN rule for `private` decls.

### Public declarations

A `public` declaration (the default — no `private` keyword) is reachable from anywhere in the project by **bare name**:

```gcl
// in models/user.gcl
type User { id: int; name: String; }

// in src/api.gcl
fn current(): User {                            // bare-name reference works
    return User { id: 1, name: "alice" };
}
```

When two modules declare the same bare name, the lang reports an ambiguous-reference error and requires you to disambiguate via FQN.

### `private` declarations

A `private` declaration is reachable cross-module **only via FQN**:

```gcl
// in mymod.gcl
private type Internal { x: int; }

// in src/api.gcl
var x: Internal = ...;            // ERROR — bare name does not resolve
var x: mymod::Internal = ...;     // OK — FQN resolves through the `private` gate
```

Same-module references to a `private` decl are **unrestricted** — the gate is purely cross-module bare-name.

This is the "still visible, just not bare-name" rule. It is **not** Java-style "module-private" (which would block all external access).

## FQN syntax

A fully-qualified name (FQN) names a declaration with its containing module:

```gcl
mymod::Foo               // module::Decl — disambiguate or reach a private decl
mymod::Foo::staticMember // module::Type::staticMember — module-qualified static access
```

The module name comes from the file stem. A file at `<project>/src/auth/user.gcl` has module name `user`. **Two modules cannot have the same name** — GreyCat will refuse to run on a project with a collision.

The `::` operator is also used for **static member access** on types:

```gcl
Type::static_member
Type::static_method()
Color::red                       // enum field
```

Both uses share the `::` operator. The lang disambiguates by context (does the LHS resolve as a module name or as a type?).

## Multi-project workspaces

A single workspace folder can host multiple independent projects, each with its own `project.gcl`. The lang treats each as a separate closure:

```
workspace/
├── server/
│   ├── project.gcl
│   ├── src/...
│   └── lib/std/
├── client/
│   ├── project.gcl
│   └── src/...
```

Each `project.gcl` defines its own analyzed module set. **Cross-project navigation is intentionally absent** — projects are isolated closures matching the runtime's deployment model. A file under `server/src/` cannot reference a type declared under `client/src/` directly.

The LSP picks the nearest `project.gcl` walking up from any opened `.gcl` file. The CLI takes the entrypoint as an explicit argument:

```sh
greycat lint server
greycat fmt client
```

## Practical layout

A typical GreyCat project:

```
my-project/
├── project.gcl              # @library + @include pragmas (@include is entrypoint-only)
├── .env                     # optional: GREYCAT_* config picked up at startup
├── bin/                     # populated by `greycat install` — gitignored
│   └── greycat              # the pinned core binary for this project's std version
├── lib/                     # populated by `greycat install` — gitignored
│   ├── installed            # name=version cache; tracks what `install` has already done
│   └── std/
│       ├── core.gcl
│       ├── io.gcl
│       ├── runtime.gcl
│       └── util.gcl
├── src/                     # @include("src");
│   ├── api.gcl
│   ├── models/
│   │   ├── user.gcl
│   │   └── session.gcl
│   └── services/
│       └── auth.gcl
├── test/                    # @include("test"); *_test.gcl files stripped by `greycat build`
│   └── api_test.gcl
├── files/                   # runtime data, served at /files/
│   ├── <user_name>/         # per-user upload area (governed by grants)
│   └── root/                # files for user id=1
├── gcdata/                  # graph storage (gitignored)
└── webroot/                 # public static assets, served at /
    └── index.html
```

Conventions:

- One `project.gcl` per project, at the root.
- `bin/` is populated by `greycat install` — gitignored. When a `bin/greycat` exists, the system `greycat` re-execs into it (set `GREYCAT_NO_REDIRECT=true` to bypass).
- `lib/` is populated by `greycat install` — gitignored. `lib/installed` is the install cache, do not commit.
- `src/` for application code.
- `files/` and `gcdata/` are runtime-managed and gitignored.
- `webroot/` for public assets served by the HTTP server. A library can ship its own assets in
  `lib/<name>/webroot/`, served at `/` under the same rules; the project `webroot/` is searched first and
  shadows them. See [runtime.md](runtime.md) for the resolution order.
  Commit it or not depending on who writes it: hand-authored assets are sources and belong in git, but a
  `webroot/` produced by a frontend bundler is generated output and is gitignored like any build directory
  (see [webapp.md](webapp.md), where `vp build` populates it). Either way the deployed server needs a
  populated `webroot/`, so a gitignored one has to be built on the deploy host or shipped as an artifact.

See [cli.md](cli.md) for `greycat install` mechanics and [runtime.md](runtime.md) for what each directory holds at runtime.

## How files become modules

A `.gcl` file is a module. Its module name is derived from its path stem and the `.gcl` extension dropped:

| Path                  | Module name            |
| --------------------- | ---------------------- |
| `project.gcl`         | `project` (entrypoint) |
| `src/api.gcl`         | `api`                  |
| `src/models/user.gcl` | `user`                 |
| `lib/std/core.gcl`    | `core`                 |

## Tests and `_test.gcl`

`greycat build` produces a production `.gcp` package and **strips every module whose filename ends in `_test.gcl`**. This keeps test code out of production deployments without any extra build configuration.

The typical layout is a separate `test/` directory included from `project.gcl`:

```
my-project/
├── project.gcl              # @include("src"); @include("test");
├── src/
│   └── api.gcl
└── test/
    └── api_test.gcl         # contains @test functions
```

```gcl
// project.gcl
@library("std", "1.2.3");
@include("src");
@include("test");
```

```gcl
// test/api_test.gcl
@test
fn test_ping() {
    Assert::equals(ping(), "pong");
}
```

Local runs (`greycat run`, `greycat test`) load both `src/` and `test/`. Production builds drop the `_test.gcl` files automatically. See [annotations.md § `@test`](annotations.md) for the test annotation itself.
