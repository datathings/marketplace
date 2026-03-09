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

Use `greycat --help` and `greycat <command> --help` for the details

## Options type

| Type | Description |
|--------|---------|
| `<byte>` | `KB`, `MB`, `GB`, `TB` and `<unit>iB` alternatives |
| `<duration>` | GreyCat duration |
| `<string>` | string text |
| `<int>` | signed 64b number |
| `<float>` | 64b floating-point number |

## greycat serve

Command: `greycat serve`

**Behavior**: Executes `main()` as task, serves `webroot`, listens for `HTTP` and `MCP` requests

**⚠️ Security**: `--user=<id>` bypasses ALL auth (NEVER prod). `--unsecure` allows HTTP (dev only).

**SSO Example:**
```bash
greycat serve --oid_client_id=abc123 --oid_config_url=https://login.microsoftonline.com/{TENANT}/v2.0/.well-known/openid-configuration
```

## Other Commands

- `greycat test`: Runs all @test functions. Exit 0 if pass, non-zero if fail. See [testing.md](testing.md)
- `greycat run`: Executes `project::main`
- `greycat run fn_name`: Executes `project::fn_name`
- `greycat run <fqn>`: Executes `module::type::method` or `module::fn`
- `greycat install`: Downloads libraries from @library directives in project.gcl (impacts: `bin/`, `lib/`, `webroot/`)
- `greycat codegen <target>`: Generates bindings for TypeScript, Python, C, Rust. See [frontend.md](frontend.md).
- `greycat defrag`: Compacts storage (atomic, safe anytime)

## greycat-lang executable

- `greycat-lang lint project.gcl` - Checks GCL for errors. **⚠️ CRITICAL**: Always run after generating/modifying .gcl files. Exit 0=no errors, 1=errors found.
- `greycat-lang fmt -p project.gcl` - Formats GCL files in-place. Respects @format_indent, @format_line_width.
- `greycat-lang server --stdio` - Starts LSP server for IDE integration. Features: Completion, go-to-definition, find-references, hover, diagnostics, formatting.

## .env File Support

GreyCat loads `.env` next to project.gcl. CLI options override env vars.

**⚠️ Security**: Always add `.env` to `.gitignore`. Never commit secrets. Never commit GREYCAT_USER or GREYCAT_UNSECURE=true for prod.

## Common Workflows

**Dev Server:** `greycat serve --user=1 --log=debug`
**CI/CD:** `greycat-lang lint && greycat test && greycat build-version-full > version.txt`
**Prod Deploy:** `greycat test && greycat serve`
**Data Reset (Dev):** `rm -rf gcdata && greycat run import` (**⚠️ DELETES DATA -> ASK FOR CONFIRMATION**)
**Quality Check:** `greycat-lang fmt && greycat-lang lint && greycat test`

## Troubleshooting

**Build Failures**: `greycat run && greycat-lang lint` — Check unresolved symbols, missing @include/@library directives.
**Port In Use**: `lsof -i :8080` or use different port: `greycat serve --port=8081`
**Auth Failures**: Dev: `--user=1`. Prod: verify OpenID config with `--log=debug`.
**Performance**: Increase workers/cache: `greycat serve --workers=16 --cache=16GB --task_pool_capacity=20000`. Compact: `greycat defrag`.
**LSP Not Working**: Verify `which greycat-lang && greycat-lang --version`. Test: `greycat-lang server --stdio`.
**Storage Issues**: Check size: `du -sh gcdata/`. Compact: `greycat defrag`. Dev only: `rm -rf gcdata && greycat run import`.

## See Also

- [testing.md](testing.md) - Comprehensive testing guide
- [frontend.md](frontend.md) - TypeScript integration, codegen
- [permissions.md](permissions.md) - Auth, @permission, OpenID
- [concurrency.md](concurrency.md) - Jobs, async, PeriodicTask
