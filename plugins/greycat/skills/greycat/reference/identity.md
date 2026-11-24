# Identity and permissions

Who a request runs as and what it may do. Reach here when the task is about users, roles, permissions, `@permission`, tokens, a sign-in without a password (a link sent by email), grants on per-user folders, group identities, the worlds a user's calls run in, or why `--user` must not be used.

## Contents

- Identity and permissions
- Tokens
- A sign-in without a password
- Grants and per-user files
- Identities as groups
- The worlds of an identity
- The `--user=<name>` impersonation flag

## Identity and permissions

The security model is **users × roles × permissions × grants**:

- A **user** has a numeric `id`, unique `name`, and exactly one `role`.
- A **role** is a named bundle of permissions (declared with `@role(name, perm1, perm2, ...)`).
- A **permission** is a named gate (declared with `@permission(name, desc)`).
- A **grant** gives permissions on what one identity owns (its `files/<name>/` folder, its tasks) to another identity, or to every holder of a permission. It is stored with the owner (see [Grants and per-user files](#grants-and-per-user-files)).

Built-in permissions live in `lib/std/runtime.gcl`:

| Permission     | What it grants                                                         |
| -------------- | ---------------------------------------------------------------------- |
| `public`       | Anonymous access. Default for an anonymous caller.                     |
| `api`          | Call `@expose`d functions. (`webroot` is public.)                      |
| `admin`        | Call the administrative functions (`@permission("admin")`).            |
| `debug`        | Low-level graph manipulation. With `admin`, call any function over HTTP, exposed or not ([http.md](http.md#calling-a-method-on-an-instance)). |
| `world`        | Create worlds, and remove the ones one created.                        |
| `files:read`   | List and download the files of every identity.                         |
| `files:write`  | Upload and delete the files of every identity.                         |
| `tasks:read`   | See the tasks of every identity, read their results and their events.  |
| `tasks:cancel` | Cancel the tasks of every identity.                                    |

A role holding one of the last four has it on every identity. A grant gives it on what one identity owns. `admin` alone opens no folder and no task.

Built-in roles:

| Role     | Grants                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------- |
| `public` | `public`                                                                                       |
| `user`   | `public`, `api`                                                                                |
| `admin`  | `public`, `admin`, `api`, `debug`, `world`, `files:read`, `files:write`, `tasks:read`, `tasks:cancel` |

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

Identity management at runtime: `Identity::login`, `Identity::logout`, `Identity::sign_in`, `Identity::token`, `Identity::tokens`, `Identity::revoke_token`, `Identity::revoke_tokens`, `Identity::set_password`, `Identity::create`, `Identity::create_with_password`, `Identity::all`, `Identity::set_role`, `Identity::set_grants`, `Identity::grant`, `Identity::revoke`, `Identity::grant_permission`, `Identity::revoke_permission`, `Identity::remove`, `Identity::set_world`, `Identity::set_worlds`. CLI equivalents under `greycat user`.

`@permission` is checked when a function is called over HTTP or as an MCP tool. A call from GCL code is not checked. `Identity::get_by_id`, `get_by_name`, `all`, `create`, `create_with_password`, `set_role`, `remove`, `grant`, `revoke`, `grant_permission`, `revoke_permission`, `set_world`, `set_worlds` and `Role::all` are `admin` over HTTP, but any caller runs them from GCL. An `@expose`d function that forwards untrusted input to them must check its caller itself (`Identity::has_permission("admin")`, for example). `Identity::token`, `tokens`, `revoke_token`, `revoke_tokens`, `set_password`, `set_grants`, `grants`, `granted_to` and `grant_of` do compare the caller to the identities they name, from GCL as well.

`Identity::create` on a name that already exists sets its role and keeps its password. `Identity::create_with_password(name, role, password)` only creates a new identity, which signs in at once, and throws on a taken name. That makes it the call behind a public sign-up endpoint (see [workflow.md](workflow.md)).

A new name is the identity's folder `files/<name>/`, so both calls throw `invalid identity name: <name>` unless it is 1 to 255 bytes of UTF-8 with no control, format (bidi, zero-width), invisible, private use or unassigned character, none of `/ \ * ? " < > |`, no space first or last, and is not `.`, `..`, a name ending in `.` or a Windows device name (`con`, `nul.txt`). A `:`, as in a name from SSO (`<provider>:<sub>`), is refused on a Windows server only, where it opens an alternate data stream. Spaces inside and any script are fine (`Jean Dupont`, `élise`, `alice+test@example.com`). A new name is stored in NFC. Every `Identity` call that takes a name (`login`, `get_by_name`, `create`, `set_password`, `grant`, `remove`, ...) names the identity named exactly so, or else the one whose name reads as it, the same in NFKC with Unicode case folding: `Alice@Corp.com` names `alice@corp.com`, `Straße` names `strasse`, and an `é` sent as `e` and a combining accent, as macOS input often is, names `é`. Accents count: `elise` and `élise` are two identities. So `create` on `Éva` when `éva` exists sets the role of `éva` and returns it, and `create_with_password` throws `identity already exists`: file systems that fold case and normalization (macOS, Windows) would give both one folder. A name from a source that may change its case, such as an SSO username, keeps naming its identity. Several identities read as one name only in a store an earlier version filled (`Élise` and `élise`): that name names none of them, `create` on it throws `identity name reads as several existing ones`, and each keeps its exact name. A new name whose folder holds files throws `the folder of a new identity is not empty: files/<name> (remove it first)`, as the new identity would own them. Such a folder is typically left after `gcdata/` was wiped and `files/` kept. Move it away, or delete it as an admin with `DELETE /files/<name>/` (see [files.md](files.md#access)), then create the identity. An empty folder is removed and does not block the name.

An empty password never signs in. `Identity::set_password(name, "")` removes the password, so the identity cannot sign in until it gets one again.

### Tokens

A token is what a request carries to run as an identity ([http.md, Authentication](http.md#authentication) says where it goes). The server keeps a row for every token it issues, so the tokens of an identity can be listed and revoked one by one. A revoked or expired token is not an error: the call runs as the anonymous caller, and a function that needs a signed-in caller answers `401`.

| Call | What it does |
| --- | --- |
| `Identity::login(name, password)` | issues a token valid for the server's `--validity`, labeled `login`, returns it and sets it as the `greycat` cookie |
| `Identity::token(id, ttl, label)` | issues a token for the identity `id`, valid for `ttl` (`null` for `--validity`), with a label of up to 255 bytes (`null` for none, a longer one is cut) |
| `Identity::tokens(name, offset, max)` | the tokens of `name` that have not expired, ordered by id, a page at a time, as `IdentityToken` (`id`, `label`, `created`, `expires`, `current`) |
| `Identity::revoke_token(name, id)` | revokes one token, `false` when `name` has no token `id` |
| `Identity::revoke_tokens(name, keep_current)` | revokes every token of `name`, except the token of the call when `keep_current` is `true`, and returns how many it revoked |
| `Identity::logout()` | revokes the token of the call and clears the cookie |
| `Identity::sign_in(id, ttl)` | signs the call in as `id` without a password, as `login` does, for `ttl` (`null` for `--validity`). See [below](#a-sign-in-without-a-password) |

`current` is `true` for the token the call carries, which lets a "signed-in devices" page mark the session in use. A call from GCL code that no request started (`greycat run`, a scheduled task) carries none. A token id is below 2^53, so a JavaScript client reads it exactly.

```gcl
@expose
fn sign_out_elsewhere(): int {
    return Identity::revoke_tokens(Identity::current().name, true);
}
```

A revoke applies to the next request, and it closes the `Task::events` streams opened with the token. A task the token already started keeps running as its identity. `Identity::set_password` leaves the sessions open, so a password change that should sign out the other devices calls `revoke_tokens(name, true)` as well. `Identity::remove` revokes every token of the identity.

`greycat serve` issues two root tokens for itself at each start, and revokes the two of the previous start. The URL it prints on a terminal carries one, valid for `--validity` and labeled `serve`, which a start that prints none revokes too. `gcdata/security/token` holds the other, which the CLI uses to call the running server. That one is valid for a year, is not listed, and `revoke_tokens` leaves it. A backup does not hold tokens, so `greycat restore` ends every session (see [storage.md](storage.md#backups)).

### A sign-in without a password

`Identity::sign_in(id, ttl)` opens the session `login` opens, without checking a password: it issues a token labeled `login`, valid for `ttl` (`null` for `--validity`), returns it, runs the rest of the call as `id`, and the response sets it as the `greycat` cookie with that expiry. The identity needs no password, so an account created with `Identity::create` signs in this way. It is the call behind a sign-in the program checks itself, such as a link sent by email. It is not `@expose`d and checks no permission, from GCL or otherwise, so the function calling it is `@permission("public")` (the visitor is not signed in yet) and must refuse whatever it has not verified. It throws `user not found: <id>` for an unknown id, and `the public identity cannot sign in` for id `0`.

```gcl
/// The identity each pending link signs in, by the SHA-256 of its secret.
var pending_links: nodeIndex<String, int>;

/// Signs the visitor in with the secret of a link the server emailed them, once.
@expose
@permission("public")
fn redeem_link(secret: String): String {
    var hash = Crypto::sha256hex(secret);
    var id = pending_links.get(hash);
    if (id == null) {
        throw "this link is not valid";
    }
    pending_links.remove(hash);
    Identity::sign_in(id, 30day);
    return Identity::current().name;
}
```

Keep the secret long and random (a `Uuid::v4()` or more), store only its hash, and let a link expire. A mail scanner opens every link it finds, so the page behind the link redeems it on a click rather than on load.

### Grants and per-user files

Each identity owns `files/<name>/` and its tasks, and grants let others reach them. Who may open a folder, and what a refusal answers, is in [files.md, Access](files.md#access).

A grant gives permissions the program declares on what its owner owns: `files:read` lists and downloads `files/<name>/`, `files:write` uploads and deletes there, `tasks:read` sees the tasks of the owner and reads their results with `Task::result` and their events, and `tasks:cancel` cancels them. A grant of the files opens no task, and the other way round. An application permission of its own (`@permission("doc:edit", ...)`) is granted the same way and checked with `Identity::has_grant(owner, "doc:edit")`. The grants of an identity say who reaches what it owns, not what it reaches. They are checked on every request, so a grant or a revoke applies at once to sessions already open. There are two kinds of grantee, and a caller gets the permissions of both together:

- An **identity** gets the permissions itself. A grant to the identity `public` reaches anonymous callers only, not signed-in users.
- The **holders of a permission** are every caller whose role holds it. The permission is one the program declares with `@permission`, stored by name and resolved on every check, so the grant follows the program across a redeploy. A grant to the holders of `api` reaches every signed-in user whose role holds `api`, the `user` and `admin` roles of the std library included, and no anonymous caller. A grant to the holders of `public` reaches every caller, anonymous ones included, since anonymous callers and every role of the std library hold `public`.

The permissions of a grant are bits shared with the program. When a new program declares its permissions in another order, the first start moves the bits of every grant, and drops the grants of a permission it no longer declares.

Grants are not transitive: when alice grants `files/alice/` to the identity `team-grid`, the members of `team-grid` do not get into it.

| Call | Effect |
| --- | --- |
| `Identity::grant("alice", "carol", "files:read")` | carol may list and download `files/alice/`, while alice gets nothing from carol. Returns whether the grant changed, and throws for a permission the program does not declare. |
| `Identity::revoke("alice", "carol", "files:read")` | takes it back. Returns whether the grant changed. |
| `Identity::grant_permission("alice", "api", "files:read")` | every signed-in user whose role holds `api` may list and download `files/alice/`. Throws for a permission the program does not declare. |
| `Identity::revoke_permission("alice", "api", "files:read")` | takes it back |
| `Identity::set_grants("alice", [IdentityGrant { name: "carol", permissions: ["files:read", "tasks:read"] }])` | replaces every grant of alice, to identities and to holders (`holders: true`, `name` the permission). `holders` false or left out names an identity. A grantee named twice gets the permissions of both entries. |
| `Identity::grants("alice", 0, 100)` | the grants of alice, those to identities by ascending id of the grantee then those to holders by name, 100 at most past the first 0 |
| `Identity::granted_to("carol", 0, 100)` | the identities that grant carol permissions themselves, by ascending id of the owner. Grants to holders are not listed. |
| `Identity::grant_of("alice", "carol")` | the permissions carol holds on what alice owns through both kinds of grants, an empty array for none. The role and ownership are not counted. |
| `Identity::has_grant("alice", "doc:edit")` | whether the caller holds `doc:edit` on what alice owns, through its role, ownership or a grant |
| `greycat user grant alice carol files:read,tasks:read` | `Identity::grant` for each permission of the list |
| `greycat user grant-permission alice api files:read` | same as `Identity::grant_permission("alice", "api", "files:read")` |

`grant`, `revoke`, `grant_permission` and `revoke_permission` are `admin` over HTTP. `set_grants`, `grants` and `granted_to` take the caller itself or an `admin`, and `grant_of` also answers a caller asking about its own access, from GCL as well. Reading an `Identity` does not list its grants, which may be many: page through `grants` and `granted_to`.

`Identity::remove(name)` drops every grant naming the removed identity, in both directions, with its grants to holders, and ends its sessions: its tokens are refused, and a call carrying one runs as an anonymous caller. Its id is never given to another identity. It deletes `files/<name>/` with every file in it, so an identity created later under the same name starts empty. A name that is not one path segment (`..`, `a/b`, which only an earlier version could create) keeps its folder. It throws when the folder cannot be deleted, after removing the identity. Before deleting the folder it cancels the identity's queued, running and awaiting tasks (a running task stops at its next instruction, or once a native call returns), stops its [streams](streams.md) (buffered records are dropped and the callback no longer runs), and closes its `Task::events` streams. The task making the call is spared, so an admin may remove itself.

#### Identities as groups

An identity does not have to be a person. Create one without a password and it cannot sign in (`login` is rejected with 400 whatever the password). Its folder becomes a shared space and its grants are the members: identities one by one, or the holders of a permission, which makes a whole role a member.

```gcl
@permission("grid", "opens the folders of the grid team");
@role("grid_member", "public", "api", "grid");

fn setup_team() {
    Identity::create("team-grid", "public");
    Identity::grant("team-grid", "alice", "files:read");
    Identity::grant("team-grid", "alice", "files:write");
    Identity::grant("team-grid", "bob", "files:read");
    Identity::grant_permission("team-grid", "grid", "files:read");
}
```

```sh
greycat user add team-grid public
greycat user grant team-grid alice files:read,files:write
greycat user grant team-grid bob files:read
greycat user grant-permission team-grid grid files:read
```

alice uploads to and deletes from `files/team-grid/`, bob and every identity of the role `grid_member` list and download, anyone else gets 403 (401 when anonymous). Only an admin changes the members: `set_grants` on `team-grid` refuses a member ("not authorized"), over HTTP and from GCL alike, and `grant_permission` is `admin`. Nothing in the API tells a group identity from a person, so keep the naming convention (or a list of group names) in the application.

### The worlds of an identity

Two settings of an identity concern the [worlds](storage.md#worlds-branches-of-the-store) its calls run in. Both are `admin` over HTTP:

| Call | Effect |
| --- | --- |
| `Identity::set_world("alice", "what-if")` | alice's default world is `what-if`, for a client to send in the `world` header. `null` or `"main"`: main, the default. |
| `Identity::set_worlds("alice", ["what-if"])` | alice's calls may run in main and `what-if` only. `null`: every world, the default. `[]`: main only. |
| `greycat user world alice what-if` | same as `Identity::set_world("alice", "what-if")` |
| `greycat user worlds alice what-if` | same as `Identity::set_worlds("alice", ["what-if"])`; `-` for `null` |

Every identity may use main, whatever its list says: a list names the other worlds, and `"main"` in it changes nothing. The default world only describes the identity: the server reads it for no call, and a call without a `world` header runs in main. A client that wants the default world reads `Identity::current().world` and sends it as the header. The two settings are independent, and a call landing in a world the identity may not use fails: `403` over HTTP, an error from `spawn` (`Job.world`) and `Scheduler::add` (`PeriodicOptions.world`). Two callers use a world whatever the list says: one holding `admin`, and the world's creator (`WorldInfo.user_id`). `World::list()` returns the worlds the caller may use.

`Identity.world` and `Identity.worlds` read the settings back, by name. A world removed since it was set no longer shows: a default world removed reads `null`.

The `world` permission lets a user create worlds with `World::create`, and reset or remove the ones they created with `World::reset` and `World::remove`; an `admin` resets or removes any. The built-in `user` role lacks it, so a project grants it through a role of its own, `@role("modeler", "public", "api", "world")`.

### The `--user=<name>` impersonation flag — footgun, do not promote

`greycat serve --user=<name>` (or `GREYCAT_USER=<name>`) makes **every incoming request run as that user, without checking auth**. Anyone who can reach the port executes endpoints with that user's full permission set — there is no "only on localhost" guard. It is **not** a dev convenience to reach for by default. Catastrophic in production; risky on a laptop on a shared / open network.

Do not propose this flag, put it in a recipe, or bake it into a `.env`. The correct dev pattern is:

1. `greycat token --user=<name>` to mint a short-lived token for `<name>` (without `--user` the token is for `public`, id 0, so always pass it), then attach it via `Authorization: <TOKEN>` header or `?authorization=<TOKEN>`.
2. Or call `Identity::login(name, pass)` from the browser / a script to get a cookie-backed session.

The boot URL printed by `greycat serve` on a fresh `gcdata/` already carries a `root` token — that's the intended bootstrap path.

If the user explicitly asks for `--user=<name>` for a one-off local test, fine; never propose it yourself.
