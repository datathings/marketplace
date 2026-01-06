# Datathings Marketplace

Official Claude Code plugin marketplace by Datathings — skills and LSP support for GreyCat development and local LLM inference.

## Quick Start

**Add the marketplace:**
```bash
/plugin marketplace add datathings/marketplace
```

**Install plugins:**
```bash
/plugin install greycat@datathings
```
```bash
/plugin install greycat-lsp@datathings
```

**Install GreyCat:**

Linux, Mac or FreeBSD (x64, arm64):
```bash
curl -fsSL https://get.greycat.io/install.sh | bash -s dev
```

Windows (x64, arm64):
```powershell
iwr https://get.greycat.io/install_dev.ps1 -useb | iex
```

**Try it:**

Run Claude in a demo folder, then paste this prompt:
```
Create a GreyCat backend with Country, City, Street, House, and Person nodes linked as a geographic hierarchy (country contains cities, cities contain streets, etc.). Generate sample `./data/addresses.csv`, an importer that loads it on startup, and expose all important API endpoints.
```

## Plugins

| Plugin | Type | Version | Description |
|--------|------|---------|-------------|
| **greycat** | Skill | 1.2.0 | Full-stack GreyCat development — GCL language, graph persistence, React integration |
| **greycat-c** | Skill | 1.2.0 | GreyCat C API and Standard Library for native development |
| **greycat-lsp** | LSP | 1.2.0 | Language Server Protocol for `.gcl` files (completion, diagnostics, hover) |
| **llamacpp** | Skill | 1.2.0 | llama.cpp C API reference (163 functions) for local LLM inference |

## Installation

### Add the Marketplace

Via GitHub:
```bash
/plugin marketplace add datathings/marketplace
```

Via Git URL:
```bash
/plugin marketplace add https://github.com/datathings/marketplace.git
```

### Install Plugins

GCL development:
```bash
/plugin install greycat@datathings
```

LSP for .gcl files:
```bash
/plugin install greycat-lsp@datathings
```

C SDK reference:
```bash
/plugin install greycat-c@datathings
```

llama.cpp API:
```bash
/plugin install llamacpp@datathings
```

### Verify

```bash
/plugin list
```

## Plugin Details

### greycat

Activates on `.gcl` files and GreyCat topics. Provides:
- GCL syntax, types, decorators (@expose, @permission, @volatile)
- Indexed collections (nodeIndex, nodeList, nodeTime, nodeGeo)
- Concurrency patterns (Jobs, await)
- Standard library (core, io, runtime, util)
- Pro libraries (ai, algebra, finance, kafka, opcua, powerflow, s3, sql, useragent)
- React integration (@greycat/web)

### greycat-lsp

IDE features for `.gcl` files. **Requires** `greycat-lang` in PATH.
- Code completion, diagnostics, hover info
- Go to definition, find references

```bash
which greycat-lang  # Verify installation
```

### greycat-c

Reference for native C development with GreyCat:
- C API functions, tensor operations
- Native function implementation

### llamacpp

Complete llama.cpp C API (163 functions):
- Model loading, inference, tokenization
- Sampling strategies (XTC, DRY, infill)
- GGUF model support

## Standalone Skill Files

The `./skills/` folder contains pre-packaged `.skill` files (zip archives) for use with other AI tools or manual installation:

```
skills/
├── greycat.skill      # GreyCat full-stack development
├── greycat-c.skill    # GreyCat C API reference
└── llamacpp.skill     # llama.cpp C API reference
```

Each `.skill` file contains a `SKILL.md` with instructions and optional `references/` documentation. To regenerate:

```bash
./package.sh           # Interactive skill selection
./package.sh -a        # Package all skills to ./skills/
./package.sh greycat   # Package a specific skill
```

## Configuration

Add to `.claude/settings.json` (project or `~/.claude/settings.json` for global):

```json
{
  "extraKnownMarketplaces": {
    "datathings": {
      "source": { "source": "github", "repo": "datathings/marketplace" }
    }
  },
  "enabledPlugins": {
    "greycat@datathings": true,
    "greycat-lsp@datathings": true
  }
}
```

## Management Commands

List installed:
```bash
/plugin list
```

Update plugin:
```bash
/plugin update greycat@datathings
```

Remove plugin:
```bash
/plugin uninstall greycat@datathings
```

List marketplaces:
```bash
/plugin marketplace list
```

Remove marketplace:
```bash
/plugin marketplace remove datathings
```

## Development

### Local Marketplace

For developing or testing marketplace plugins locally:
```bash
/plugin marketplace add /path/to/marketplace
```

### Bump Versions

Update all plugin versions at once:
```bash
./bump-version.sh           # Show current versions
./bump-version.sh 1.3.0     # Bump all plugins to 1.3.0
```

### Package Skills

Generate standalone `.skill` files for distribution:
```bash
./package.sh                # Interactive skill selection
./package.sh -a             # Package all skills
./package.sh greycat        # Package specific skill
```

### Native GreyCat C Libraries

To develop native GreyCat C libraries (custom functions implemented in C), install the **greycat-c** plugin:
```bash
/plugin install greycat-c@datathings
```

This provides the C API reference, tensor operations, and native function implementation patterns required for extending GreyCat with C code.

## Troubleshooting

**Skills not activating**: Verify with `/plugin list`, ensure enabled in settings.

**LSP not working**: Check `greycat-lang --version` is installed and in PATH.

## Links

- **GreyCat**: https://greycat.io | https://doc.greycat.io | https://get.greycat.io
- **Datathings**: https://datathings.com
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **Support**: contact@datathings.com
- **Issues**: https://github.com/datathings/marketplace/issues

## License

Apache-2.0
