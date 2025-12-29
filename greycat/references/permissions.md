# Authentication & Permissions

## Identity

Every HTTP request must be identified. GreyCat uses cookie or bearer tokens containing User ID and permissions.

### Development Modes

```bash
greycat serve --unsecure  # Allow HTTP (not just HTTPS)
greycat serve --user=1    # Bypass auth (NEVER in production!)
```

## Built-in Permissions

| Permission | Description |
|------------|-------------|
| `public` | Default for anonymous users |
| `admin` | Full server administration |
| `api` | Access exposed functions |
| `debug` | Low-level graph manipulation |
| `files` | Access /files/ and /webroot/ |

## Default Roles

| Role | Permissions |
|------|-------------|
| `public` | public |
| `admin` | public, admin, api, debug, files |
| `user` | public, api, files |

## Defining Permissions

```gcl
// Declare custom permission
@permission("app.admin", "Application admin permission");
@permission("app.user", "Application user permission");

// Define role with permissions
@role("custom", "app.admin", "app.user");
```

## Function Permission Decorator

```gcl
@expose
@permission("api")
fn myFunction(): String { return "Hello"; }

// Multiple permissions (OR logic)
@permission("super", "normal")
fn test() {
    if (User::hasPermission("normal")) {
        // normal user path
    } else {
        // super user path
    }
}
```

## Public Access

### Public Role

```gcl
// Allow anonymous users to access API and files
@role("public", "api", "files");
```

### Public Endpoint

```gcl
@expose
@permission("public")
fn hello(name: String): String {
    return "Hello ${name}";
}
```

## Runtime Permission Check

```gcl
if (User::hasPermission("admin")) {
    // Admin-only logic
}
```

## Security Files

Generated on first run in `gcdata/security/`:
- `password` - Root password
- `private_key` - Token encryption key
- `user_policy.gcb` - Roles/permissions config

> **private_key** and **user_policy.gcb** should NOT be shared between DEV and PROD.

## External SSO

GreyCat supports OpenID Connect (Azure AD, Keycloak):

```bash
greycat serve \
  --oid_client_id=<CLIENT_ID> \
  --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```

GreyCat maps SSO provider roles to internal roles automatically.

## Explorer Admin

Access admin UI at: `http://127.0.0.1:8080/explorer`

Manage roles and permissions through the web interface.
