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
@permission("app.admin", "desc"); @permission("app.user", "desc");
@role("custom", "app.admin", "app.user");
```

## Function Decorators

```gcl
@expose @permission("api") fn myFunction(): String { return "Hello"; }
@permission("super", "normal") fn test() {  // OR logic
    if (User::hasPermission("normal")) { /* normal */ } else { /* super */ }
}
```

## Public Access

```gcl
@role("public", "api", "files");  // Allow anonymous
@expose @permission("public") fn hello(name: String): String { return "Hello ${name}"; }
```

## Runtime Check

```gcl
if (User::hasPermission("admin")) { /* admin logic */ }
```

## Security Files

Generated in `gcdata/security/`: `password`, `private_key`, `user_policy.gcb`. **DON'T share private_key/user_policy.gcb between DEV/PROD.**

## External SSO

OpenID Connect (Azure AD, Keycloak):
```bash
greycat serve --oid_client_id=<ID> --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```
Maps SSO roles to internal roles automatically.

## Explorer Admin

Admin UI: `http://127.0.0.1:8080/explorer` - Manage roles/permissions via web interface.
