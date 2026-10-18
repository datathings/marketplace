# Identity and permissions

Who a request runs as and what it may do. Reach here when the task is about users, roles, permissions, `@permission`, tokens, grants on per-user folders, group identities, or why `--user` must not be used.

## Contents

- Identity and permissions
- Grants and per-user files
- Identities as groups
- The `--user=<name>` impersonation flag

## Identity and permissions

The security model is **users × roles × permissions × grants**:

- A **user** has a numeric `id`, unique `name`, and exactly one `role`.
- A **role** is a named bundle of permissions (declared with `@role(name, perm1, perm2, ...)`).
- A **permission** is a named gate (declared with `@permission(name, desc)`).
- A **grant** lets one identity open another identity's `files/<name>/` folder (read, write, or both). It is stored in the `grants` of the folder's owner (see [Grants and per-user files](#grants-and-per-user-files)).

Built-in permissions live in `lib/std/runtime.gcl`:

| Permission | What it grants                                     |
| ---------- | -------------------------------------------------- |
| `public`   | Anonymous access. Default for an anonymous caller. |
| `api`      | Call `@expose`d functions. (`webroot` is public.)  |
| `admin`    | Full administrative access.                        |
| `debug`    | Low-level graph manipulation.                      |

Built-in roles:

| Role     | Grants                            |
| -------- | --------------------------------- |
| `public` | `public`                          |
| `user`   | `public`, `api`                   |
| `admin`  | `public`, `admin`, `api`, `debug` |

`@permission("name")` on a function gates it on that permission. With no `@permission`, an `@expose`d function defaults to requiring `api` — that is the recommended default. Reserve `@permission("public")` for endpoints that genuinely must serve anonymous callers (the `login` endpoint itself, an unauthenticated health probe); never add it to a write-capable endpoint just to avoid wiring up login.

```gcl
@expose
fn list_users(): Array<User> { /*...*/ }         // default: requires "api" (authenticated)

@expose
@permission("admin")
fn restart() { /*...*/ }                          // narrowed: admins only

@expose
@permission("public")
fn ping(): String { return "pong"; }              // anonymous — use only when the endpoint must serve anonymous traffic
```

A handful of stdlib endpoints are `@permission("public")` because they filter their *own* output by the caller's permission mask rather than gating the call: MCP `tools/list` and `runtime::OpenApi::v3` both drop every function the caller could not invoke, so reaching them anonymously discloses only the `public` endpoints. That pattern — public entry point, per-item filtering inside — is the only reason to make a listing endpoint public; a function that returns data rather than a filtered catalogue still needs a real permission.

Identity management at runtime: `Identity::login`, `Identity::token`, `Identity::set_password`, `Identity::create`, `Identity::create_with_password`, `Identity::all`, `Identity::set_role`, `Identity::set_grants`, `Identity::grant`, `Identity::revoke`, `Identity::remove`. CLI equivalents under `greycat user`.

`@permission` is checked when a function is called over HTTP or as an MCP tool. A call from GCL code is not checked. `Identity::get_by_id`, `get_by_name`, `all`, `create`, `create_with_password`, `set_role`, `remove`, `grant`, `revoke` and `Role::all` are `admin` over HTTP, but any caller runs them from GCL. An `@expose`d function that forwards untrusted input to them must check its caller itself (`Identity::has_permission("admin")`, for example). `Identity::token`, `set_password` and `set_grants` do compare the caller to the target identity, from GCL as well.

`Identity::create` on a name that already exists sets its role and keeps its password. `Identity::create_with_password(name, role, password)` only creates a new identity, which signs in at once, and throws on a taken name. That makes it the call behind a public sign-up endpoint (see [workflow.md](workflow.md)).

An empty password never signs in. `Identity::set_password(name, "")` removes the password, so the identity cannot sign in until it gets one again.

### Grants and per-user files

Each identity owns `files/<name>/`, and a grant lets another identity open it. Who may open a folder, and what a refusal answers, is in [files.md, Access](files.md#access).

The `grants` of an identity are the list of who may open its folder, not the folders it may open. The same rule decides who sees an identity's tasks (`read`) and cancels them (`write`). It is checked on every request, so a grant or a revoke applies at once to sessions already open. Grants are not transitive: when alice grants `files/alice/` to the identity `team-grid`, the members of `team-grid` do not get into it.

| Call | Effect |
| --- | --- |
| `Identity::set_grants("alice", [IdentityGrant { name: "carol", grant: IdentityGrantType::read }])` | carol may read `files/alice/`. Replaces the whole list. A non-admin may only call it on themselves. |
| `Identity::grant("alice", "carol", IdentityGrantType::read)` | adds carol to the grants of alice, so carol may read `files/alice/` while alice gets nothing on `files/carol/` |
| `Identity::revoke("alice", "carol", IdentityGrantType::read)` | carol no longer reads `files/alice/` |
| `greycat user grant alice r carol` | same as `Identity::grant("alice", "carol", read)` |

`Identity::remove(name)` keeps `files/<name>/` on disk, and an identity created later under the same name owns it again. The grants other identities extend to the removed one stop showing in their `grants` but stay stored, and a new identity can be given the same id. Revoke them before removing an identity.

#### Identities as groups

An identity does not have to be a person. Create one without a password and it cannot sign in (`login` is rejected with 400 whatever the password). Its folder becomes a shared space and its `grants` are the member list:

```gcl
Identity::create("team-grid", "public");
Identity::set_grants("team-grid", [
    IdentityGrant { name: "alice", grant: IdentityGrantType::read_write },
    IdentityGrant { name: "bob", grant: IdentityGrantType::read }
]);
```

```sh
greycat user add team-grid public
greycat user grant team-grid rw alice
greycat user grant team-grid r bob
```

alice uploads to and deletes from `files/team-grid/`, bob lists and downloads, anyone else gets 403 (401 when anonymous). Only an admin changes the members: `set_grants` on `team-grid` refuses a member ("not authorized"), over HTTP and from GCL alike. Nothing in the API tells a group identity from a person, so keep the naming convention (or a list of group names) in the application.

### The `--user=<name>` impersonation flag — footgun, do not promote

`greycat serve --user=<name>` (or `GREYCAT_USER=<name>`) makes **every incoming request run as that user, without checking auth**. Anyone who can reach the port executes endpoints with that user's full permission set — there is no "only on localhost" guard. It is **not** a dev convenience to reach for by default. Catastrophic in production; risky on a laptop on a shared / open network.

Do not propose this flag, put it in a recipe, or bake it into a `.env`. The correct dev pattern is:

1. `greycat token --user=<name>` to mint a short-lived token for `<name>` (default `root`), then attach it via `Authorization: <TOKEN>` header or `?authorization=<TOKEN>`.
2. Or call `Identity::login(name, pass)` from the browser / a script to get a cookie-backed session.

The boot URL printed by `greycat serve` on a fresh `gcdata/` already carries a `root` token — that's the intended bootstrap path.

If the user explicitly asks for `--user=<name>` for a one-off local test, fine; never propose it yourself.
