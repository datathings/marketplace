# Authentication & Permissions

## Identity

HTTP requests use cookie/bearer tokens with User ID + permissions.

**Dev modes**: `--unsecure` (HTTP), `--user=1` (bypass auth, NEVER prod). See [cli.md](cli.md).

## Built-in Permissions

| Permission | Description |
|------------|-------------|
| `public` | Anonymous default |
| `admin` | Full administration |
| `api` | Exposed functions |
| `debug` | Graph manipulation |
| `files` | /files/ and /webroot/ |

## Default Roles

| Role | Permissions |
|------|-------------|
| `public` | public |
| `admin` | public, admin, api, debug, files |
| `user` | public, api, files |

## Defining Permissions

```gcl
@permission("new-perm", "Description of new-perm");
@role("new-role", "public", "api", "new-perm"); // new role
@role("api", "new-perm"); // add perm to std-defined role
```

## Function Decorators

```gcl
@expose // means @permission("api") by default, override by setting another explicitly
fn myFunction(): String { return "Hello"; }

@permission("super", "normal")
fn test() {}
```

## Public Access

```gcl
@role("public", "api", "files");  // Allows API+FILES for public role

@expose
@permission("public") // Allows unregistered user to call `hello`
fn hello(name: String): String { return "Hello ${name}"; }
```

**`@role("public", "api")`** adds the `api` permission to the `public` role. Since `@expose` implies `@permission("api")` by default, this lets unauthenticated users call all `@expose` endpoints and read the ABI (required for the web SDK to work without login). Use this intentionally — it makes the ABI and all default-permission endpoints publicly accessible.

## User Management

### Creating Users

```gcl
fn main() {
    if (User::getByName("alice") == null) {
        SecurityEntity::set(User {
            id: -1,              // -1 = auto-assign
            name: "alice",
            activated: true,
            full_name: "Alice",
            email: null,
            role: "user",        // must match a @role name
            groups: null,
            groups_flags: null,
            external: false,
        });
        User::setPassword("alice", Crypto::sha256hex("password"));
    }
}
```

- `User::getByName(name)` — returns the User object or null (check before creating)
- `User::setPassword(name, hash)` — password must be `Crypto::sha256hex(plaintext)`

### Login Flow (Frontend / Backend)

The frontend SDK sends `base64(username:sha256hex(password))`. The backend must store `Crypto::sha256hex(password)` so the stored value matches.

1. Frontend calls `gc.sdk.login({ username, password, use_cookie: true })`
2. Backend exposes a `@permission("public")` endpoint to return profile data after login
3. Backend uses `User::me()` to get the current user in `@permission("api")` endpoints

`User::login()` requires HTTP context — it cannot be called from backend GCL code or tests.

## Runtime Check

```gcl
if (User::hasPermission("admin")) { /* admin logic */ }
```

## Security Files

Generated in `gcdata/security/`: `password`, `private_key`, `user_policy.gcb`. **DON'T share private_key/user_policy.gcb between DEV/PROD.**

## External SSO

OpenID Connect (Azure AD, Keycloak):
```bash
greycat serve --oid_client_id=<string> --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```
Maps SSO roles to internal roles automatically.

## Explorer Admin

Admin UI: `http://127.0.0.1:8080/explorer` - Manage roles/permissions via web interface.
