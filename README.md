# Datathings Marketplace

Official Claude Code plugin marketplace by Datathings — skills and LSP support for GreyCat development and local LLM inference.

## Quick Start

```bash
# Add the marketplace
/plugin marketplace add datathings/marketplace

# Install plugins
/plugin install greycat@datathings
/plugin install greycat-lsp@datathings
```

## Plugins

| Plugin | Type | Version | Description |
|--------|------|---------|-------------|
| **greycat** | Skill | 1.0.0 | Full-stack GreyCat development — GCL language, graph persistence, React integration |
| **greycat-c** | Skill | 1.0.0 | GreyCat C API and Standard Library for native development |
| **greycat-lsp** | LSP | 0.1.0 | Language Server Protocol for `.gcl` files (completion, diagnostics, hover) |
| **llamacpp** | Skill | 1.1.0 | llama.cpp C API reference (163 functions) for local LLM inference |

## Installation

### Add the Marketplace

```bash
# Via GitHub
/plugin marketplace add datathings/marketplace

# Via Git URL
/plugin marketplace add https://github.com/datathings/marketplace.git

# Local development
/plugin marketplace add /path/to/marketplace
```

### Install Plugins

```bash
/plugin install greycat@datathings      # GCL development
/plugin install greycat-lsp@datathings  # LSP for .gcl files
/plugin install greycat-c@datathings    # C SDK reference
/plugin install llamacpp@datathings     # llama.cpp API
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
- Standard libraries (io, algebra, finance, kafka, s3, sql, etc.)
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
./package-all.sh       # Package all skills to ./skills/
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

```bash
/plugin list                              # List installed
/plugin update greycat@datathings         # Update plugin
/plugin uninstall greycat@datathings      # Remove plugin
/plugin marketplace list                  # List marketplaces
/plugin marketplace remove datathings     # Remove marketplace
```

## Troubleshooting

**Skills not activating**: Verify with `/plugin list`, ensure enabled in settings.

**LSP not working**: Check `greycat-lang --version` is installed and in PATH.

## Links

- **GreyCat**: https://doc.greycat.io | https://get.greycat.io
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **Support**: contact@datathings.com
- **Issues**: https://github.com/datathings/marketplace/issues

## License

MIT
