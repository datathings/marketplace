# GreyCat CLI Reference

Complete CLI documentation. All projects use `project.gcl`. Run from project root.

## Commands

Pattern: `greycat <command> [options] (param)?`

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

**Logging & Storage:**
- `--log=<level>` / `GREYCAT_LOG` (trace/debug/info/warn/error, def: info)
- `--logfile` / `GREYCAT_LOGFILE` (def: false)
- `--cache=<MB>` / `GREYCAT_CACHE` (def: 8192)
- `--store=<MB>` / `GREYCAT_STORE` (def: 1000)
- `--store_paths=<path>` / `GREYCAT_STORE_PATHS` (def: ./gcdata)

**Execution:**
- `--workers=<N>` / `GREYCAT_WORKERS` (def: 8)
- `--user=<id>` / `GREYCAT_USER` (def: 0, use 1 to bypass auth in dev)
- `--tz=<str>` / `GREYCAT_TZ` (e.g., Europe/Luxembourg)

## greycat serve

Start server with HTTP API + MCP.

```bash
greycat serve                           # Default: port 8080, secure
greycat serve --port=3000 --workers=8   # Custom config
greycat serve --user=1 --unsecure      # Dev mode (UNSAFE!)
```

**Essential Options:**
- `--port=<N>` / `GREYCAT_PORT` (def: 8080)
- `--workers=<N>` / `GREYCAT_WORKERS` (def: 8)
- `--user=<id>` / `GREYCAT_USER` (def: 0, **DEV ONLY**: bypass auth)
- `--unsecure` / `GREYCAT_UNSECURE` (def: false, **DEV ONLY**: allow HTTP)

**Performance:**
- `--http_threads=<N>` / `GREYCAT_HTTP_THREADS` (def: 3)
- `--req_workers=<N>` / `GREYCAT_REQ_WORKERS` (def: 2)
- `--task_pool_capacity=<N>` / `GREYCAT_TASK_POOL_CAPACITY` (def: 10000)
- `--request_pool_capacity=<N>` / `GREYCAT_REQUEST_POOL_CAPACITY` (def: 512)
- `--defrag_ratio=<f64>` / `GREYCAT_DEFRAG_RATIO` (def: 1.00, <0 = off)

**Server:**
- `--webroot=<str>` / `GREYCAT_WEBROOT` (def: webroot)
- `--keep_alive` / `GREYCAT_KEEP_ALIVE` (def: false)
- `--validity=<sec>` / `GREYCAT_VALIDITY` (def: 86400)

**Security/SSO:**
- `--key=<str>` / `GREYCAT_KEY` (private key path)
- `--keysafe=<str>` / `GREYCAT_KEYSAFE` (password)
- `--oid_client_id=<str>` / `GREYCAT_OID_CLIENT_ID`
- `--oid_config_url=<str>` / `GREYCAT_OID_CONFIG_URL`
- `--oid_keys_url=<str>` / `GREYCAT_OID_KEYS_URL`
- `--oid_public_key=<str>` / `GREYCAT_OID_PUBLIC_KEY`

**Backup:**
- `--backup_path=<str>` / `GREYCAT_BACKUP_PATH` (def: backup)
- `--max_backup_files=<N>` / `GREYCAT_MAX_BACKUP_FILES` (def: 3)

**Behavior**: Executes main() as task, serves /webroot/, enables /explorer (if @library("explorer")), starts MCP server.

**⚠️ Security**: `--user=<id>` bypasses ALL auth (NEVER prod). `--unsecure` allows HTTP (dev only).

**SSO Example:**
```bash
greycat serve --oid_client_id=abc123 --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```

## greycat test

Run all @test functions. Exit 0 if pass, non-zero if fail.

```bash
greycat test; greycat test --log=debug
```

**Discovery**: Finds all @test functions in `*_test.gcl`, runs `setup()` before, `teardown()` after.

**Output**: `project::test_name ok (5us) ... tests success: 2, failed: 0, skipped: 0`

See [testing.md](testing.md) for comprehensive guide.

## greycat run

Execute main() or specified function.

```bash
greycat run             # main()
greycat run import_data # project::import_data()
greycat run --user=1    # As user ID 1
```

## greycat install

Download libraries from @library directives in project.gcl.

## greycat codegen

Generate typed headers: TypeScript, Python, C, Rust. See [frontend.md](frontend.md).

## greycat defrag

Compact storage (atomic, safe anytime).

```bash
greycat defrag; greycat defrag --store_paths=./gcdata
```

**When**: After deleting data, optimize disk usage.

## greycat-lang Commands

### greycat-lang lint

Check GCL for errors.

```bash
greycat-lang lint; greycat-lang lint --project=./backend/project.gcl
```

**⚠️ CRITICAL**: Always run after generating/modifying .gcl files. Exit 0=no errors, 1=errors found.

### greycat-lang fmt

Format GCL files in-place. Respects @format_indent, @format_line_width.

### greycat-lang server

Start LSP server for IDE integration.

```bash
greycat-lang server --stdio    # VS Code, Neovim
greycat-lang server --tcp=6008 # Network clients
```

**Features**: Completion, go-to-definition, find-references, hover, diagnostics, formatting.

**Integration**: Install `greycat-lsp` Claude Code plugin for automatic setup.

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

**Dev Server:**
```bash
# Create .env: GREYCAT_USER=1, GREYCAT_UNSECURE=true, GREYCAT_LOG=debug
greycat serve
```

**CI/CD:**
```bash
greycat-lang lint && greycat test --log=info && greycat build-version-full > version.txt
```

**Prod Deploy:**
```bash
greycat build --log=warn && greycat test && greycat serve
```

**Data Reset (Dev):**
```bash
# ⚠️ DELETES DATA
rm -rf gcdata && greycat run import
```

**Quality Check:**
```bash
greycat-lang lint && greycat-lang fmt && greycat test
```

## Troubleshooting

**Build Failures**: `greycat build --log=debug && greycat-lang lint` — Check unresolved symbols, missing @include/@library directives.

**Port In Use**: `lsof -i :8080` or use different port: `greycat serve --port=8081`

**Auth Failures**: Dev: `--user=1 --unsecure`. Prod: verify OpenID config with `--log=debug`.

**Performance**: Increase workers/cache: `greycat serve --workers=16 --cache=16384 --task_pool_capacity=20000`. Compact: `greycat defrag`.

**LSP Not Working**: Verify `which greycat-lang && greycat-lang --version`. Test: `greycat-lang server --stdio`. Install plugin: `/install greycat-lsp`.

**Storage Issues**: Check size: `du -sh gcdata/`. Compact: `greycat defrag`. Dev only: `rm -rf gcdata && greycat run import`.

## Help

```bash
greycat --help; greycat serve --help; greycat -v; greycat -vv  # Full version + git hash
```

## See Also

- [testing.md](testing.md) - Comprehensive testing guide
- [frontend.md](frontend.md) - TypeScript integration, codegen
- [permissions.md](permissions.md) - Auth, @permission, OpenID
- [concurrency.md](concurrency.md) - Jobs, async, PeriodicTask
