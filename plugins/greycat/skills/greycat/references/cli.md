# GreyCat CLI Reference

Complete CLI documentation. All projects use `project.gcl`. Run from project root.

## Commands

| Command | Purpose | Param | Exit |
|---------|---------|-------|------|
| `build` | Compile | - | 0 |
| `run` | Execute | Name (def: main) | 0 |
| `serve` | Start server | - | ∞ |
| `test` | Run @test | - | 0=pass |
| `install` | Download libs | - | 0 |
| `print` | Show .gcb | Path | 0 |
| `codegen` | Gen headers | - | 0 |
| `bytecode` | Show bytecode | - | 0 |
| `defrag` | Compact | - | 0 |
| `build-version` | Project ver | - | 0 |
| `build-version-full` | Ver+git | - | 0 |
| `version` | GreyCat ver | - | 0 |

## Common Options

All commands accept these options (CLI / Environment Variable):

| Option | Env Var | Default | Description |
|--------|---------|---------|-------------|
| `--log=<level>` | `GREYCAT_LOG` | info | trace/debug/info/warn/error |
| `--logfile` | `GREYCAT_LOGFILE` | false | Log to file |
| `--cache=<MB>` | `GREYCAT_CACHE` | 8192 | Cache size in MB |
| `--store=<MB>` | `GREYCAT_STORE` | 1000 | Store size in MB |
| `--store_paths=<path>` | `GREYCAT_STORE_PATHS` | ./gcdata | Storage directory |
| `--workers=<N>` | `GREYCAT_WORKERS` | 8 | Worker threads |
| `--user=<id>` | `GREYCAT_USER` | 0 | User ID (1 bypasses auth, **DEV ONLY**) |
| `--tz=<str>` | `GREYCAT_TZ` | - | Timezone (e.g., Europe/Luxembourg) |

## greycat serve

Start server with HTTP API + MCP.

```bash
greycat serve                           # Default: port 8080, secure
greycat serve --port=3000 --workers=8   # Custom config
greycat serve --user=1 --unsecure      # Dev mode (UNSAFE!)
```

**Essential Options:**

| Option | Env Var | Default | Description |
|--------|---------|---------|-------------|
| `--port=<N>` | `GREYCAT_PORT` | 8080 | HTTP port |
| `--workers=<N>` | `GREYCAT_WORKERS` | 8 | Worker threads |
| `--user=<id>` | `GREYCAT_USER` | 0 | **DEV ONLY**: bypass auth |
| `--unsecure` | `GREYCAT_UNSECURE` | false | **DEV ONLY**: allow HTTP |
| `--http_threads=<N>` | `GREYCAT_HTTP_THREADS` | 3 | HTTP handler threads |
| `--req_workers=<N>` | `GREYCAT_REQ_WORKERS` | 2 | Request workers |
| `--task_pool_capacity=<N>` | `GREYCAT_TASK_POOL_CAPACITY` | 10000 | Task queue size |
| `--request_pool_capacity=<N>` | `GREYCAT_REQUEST_POOL_CAPACITY` | 512 | Request queue size |
| `--defrag_ratio=<f64>` | `GREYCAT_DEFRAG_RATIO` | 1.00 | Defrag threshold (<0=off) |
| `--webroot=<str>` | `GREYCAT_WEBROOT` | webroot | Static files path |
| `--keep_alive` | `GREYCAT_KEEP_ALIVE` | false | HTTP keep-alive |
| `--validity=<sec>` | `GREYCAT_VALIDITY` | 86400 | Token validity (seconds) |

**Security/SSO Options:**

| Option | Env Var | Description |
|--------|---------|-------------|
| `--key=<str>` | `GREYCAT_KEY` | Private key path |
| `--keysafe=<str>` | `GREYCAT_KEYSAFE` | Key password |
| `--oid_client_id=<str>` | `GREYCAT_OID_CLIENT_ID` | OpenID client ID |
| `--oid_config_url=<str>` | `GREYCAT_OID_CONFIG_URL` | OpenID config URL |
| `--oid_keys_url=<str>` | `GREYCAT_OID_KEYS_URL` | OpenID keys URL |
| `--oid_public_key=<str>` | `GREYCAT_OID_PUBLIC_KEY` | OpenID public key |

**Backup Options:**

| Option | Env Var | Default | Description |
|--------|---------|---------|-------------|
| `--backup_path=<str>` | `GREYCAT_BACKUP_PATH` | backup | Backup directory |
| `--max_backup_files=<N>` | `GREYCAT_MAX_BACKUP_FILES` | 3 | Max backup files |

**Behavior**: Executes main() as task, serves /webroot/, enables /explorer (if @library("explorer")), starts MCP server.

**⚠️ Security**: `--user=<id>` bypasses ALL auth (NEVER prod). `--unsecure` allows HTTP (dev only).

**SSO Example:**
```bash
greycat serve --oid_client_id=abc123 --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```

## Other Commands

**greycat test** - Run all @test functions. Exit 0 if pass, non-zero if fail.
```bash
greycat test
greycat test --log=debug
```
Finds all @test functions in `*_test.gcl`, runs `setup()` before, `teardown()` after. See [testing.md](testing.md).

**greycat run** - Execute main() or specified function.
```bash
greycat run             # main()
greycat run import_data # project::import_data()
greycat run --user=1    # As user ID 1
```

**greycat install** - Download libraries from @library directives in project.gcl.

**greycat codegen** - Generate typed headers: TypeScript, Python, C, Rust. See [frontend.md](frontend.md).

**greycat defrag** - Compact storage (atomic, safe anytime). When: After deleting data, optimize disk usage.
```bash
greycat defrag
greycat defrag --store_paths=./gcdata
```

## greycat-lang Commands

**greycat-lang lint** - Check GCL for errors. **⚠️ CRITICAL**: Always run after generating/modifying .gcl files. Exit 0=no errors, 1=errors found.
```bash
greycat-lang lint
greycat-lang lint --project=./backend/project.gcl
```

**greycat-lang fmt** - Format GCL files in-place. Respects @format_indent, @format_line_width.

**greycat-lang server** - Start LSP server for IDE integration. Features: Completion, go-to-definition, find-references, hover, diagnostics, formatting.
```bash
greycat-lang server --stdio    # VS Code, Neovim
greycat-lang server --tcp=6008 # Network clients
```

## .env File Support

GreyCat loads `.env` next to project.gcl. CLI options override env vars.

**Dev .env:**
```bash
GREYCAT_PORT=3000
GREYCAT_WORKERS=4
GREYCAT_LOG=debug
GREYCAT_USER=1       # UNSAFE - don't commit
GREYCAT_UNSECURE=true # UNSAFE - don't commit
GREYCAT_TZ=Europe/Luxembourg
```

**Prod .env:**
```bash
GREYCAT_PORT=8080
GREYCAT_WORKERS=16
GREYCAT_LOG=warn
GREYCAT_LOGFILE=true
GREYCAT_STORE=10000
GREYCAT_CACHE=16384
GREYCAT_STORE_PATHS=/var/lib/greycat/data
GREYCAT_OID_CLIENT_ID=prod-client-id
GREYCAT_OID_CONFIG_URL=https://login.microsoftonline.com/TENANT/v2.0/.well-known/openid-configuration
GREYCAT_HTTP_THREADS=8
GREYCAT_REQ_WORKERS=4
GREYCAT_BACKUP_PATH=/var/backups/greycat
GREYCAT_MAX_BACKUP_FILES=7
GREYCAT_TZ=UTC
```

**⚠️ Security**: Always add `.env` to `.gitignore`. Never commit secrets. Never commit GREYCAT_USER or GREYCAT_UNSECURE=true for prod.

## Common Workflows

**Dev Server:** `greycat serve` (with .env: GREYCAT_USER=1, GREYCAT_UNSECURE=true, GREYCAT_LOG=debug)

**CI/CD:** `greycat-lang lint && greycat test --log=info && greycat build-version-full > version.txt`

**Prod Deploy:** `greycat build --log=warn && greycat test && greycat serve`

**Data Reset (Dev):** `rm -rf gcdata && greycat run import` (**⚠️ DELETES DATA**)

**Quality Check:** `greycat-lang lint && greycat-lang fmt && greycat test`

## Troubleshooting

**Build Failures**: `greycat build --log=debug && greycat-lang lint` — Check unresolved symbols, missing @include/@library directives.

**Port In Use**: `lsof -i :8080` or use different port: `greycat serve --port=8081`

**Auth Failures**: Dev: `--user=1 --unsecure`. Prod: verify OpenID config with `--log=debug`.

**Performance**: Increase workers/cache: `greycat serve --workers=16 --cache=16384 --task_pool_capacity=20000`. Compact: `greycat defrag`.

**LSP Not Working**: Verify `which greycat-lang && greycat-lang --version`. Test: `greycat-lang server --stdio`. Install plugin: `/install greycat-lsp`.

**Storage Issues**: Check size: `du -sh gcdata/`. Compact: `greycat defrag`. Dev only: `rm -rf gcdata && greycat run import`.

## Help

```bash
greycat --help
greycat serve --help
greycat -v
greycat -vv  # Full version + git hash
```

## See Also

- [testing.md](testing.md) - Comprehensive testing guide
- [frontend.md](frontend.md) - TypeScript integration, codegen
- [permissions.md](permissions.md) - Auth, @permission, OpenID
- [concurrency.md](concurrency.md) - Jobs, async, PeriodicTask
