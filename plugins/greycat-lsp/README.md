# GreyCat LSP Plugin

Language Server Protocol support for GreyCat `.gcl` files in Claude Code.

## Overview

This plugin enables IDE-like features when working with `.gcl` files (GreyCat source code). It connects Claude Code to the GreyCat Language Server, providing intelligent code assistance for GreyCat development.

**Use this plugin when:**
- Writing or editing `.gcl` files
- Developing GreyCat backend applications
- Working with `project.gcl` and related source files

## Prerequisites

The `greycat-lang` binary must be installed and available in your PATH.

```bash
# Verify installation
greycat-lang --version
```

See https://get.greycat.io for installation instructions.

## Standalone CLI Commands

Beyond the LSP server, `greycat-lang` provides CLI tools for batch operations and CI/CD:

```bash
# Lint a project
greycat-lang lint project.gcl

# Format GCL files (outputs to stdout by default)
greycat-lang fmt file.gcl
```

These are useful for pre-commit hooks, CI pipelines, or quick command-line checks. The LSP server provides the same functionality (linting via diagnostics, formatting via document formatting) through the standard protocol.

## Features

The GreyCat LSP provides comprehensive IDE features:

### Code Intelligence
- **Code Completion** - Intelligent suggestions with trigger characters
- **Signature Help** - Function parameter hints on `(` and `,`
- **Hover Information** - Type information and documentation on hover
- **Inlay Hints** - Inline type annotations

### Navigation
- **Go to Definition** - Jump to symbol definitions
- **Find References** - Find all usages of a symbol
- **Document Symbols** - Outline view of file structure

### Code Quality
- **Diagnostics** - Real-time error and warning detection (on save)
- **Code Actions** - Quick fixes and refactoring suggestions
- **Semantic Tokens** - Enhanced syntax highlighting

### Editing
- **Document Formatting** - Auto-format `.gcl` files
- **Rename Symbol** - Safely rename across the codebase
- **Code Lens** - Inline actionable information

### Workspace
- **Multi-folder Support** - Works with workspace folders
- **File Watching** - Tracks `.gcl` file creation/deletion
- **Library Detection** - Monitors `lib/installed` for dependencies

## Configuration

The LSP is configured in `.lsp.json`:

```json
{
  "greycat": {
    "command": "greycat-lang",
    "args": ["server"],
    "extensionToLanguage": {
      ".gcl": "greycat"
    }
  }
}
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `command` | `greycat-lang` | Path to the LSP binary |
| `args` | `["server"]` | Command-line arguments |
| `transport` | `stdio` | Communication transport |
| `startupTimeout` | `2000` | Max startup time (ms) |
| `restartOnCrash` | `true` | Auto-restart on crash |
| `maxRestarts` | `3` | Maximum restart attempts |

## Troubleshooting

### LSP not starting

1. Verify `greycat-lang` is in your PATH:
   ```bash
   which greycat-lang
   ```

2. Test the LSP server directly:
   ```bash
   greycat-lang server
   ```

### Completion not working

Ensure you're in a valid GreyCat project with a `project.gcl` file.

### Diagnostics not updating

Diagnostics update on file save. Save the file to see updated errors/warnings.

## License

MIT License - See LICENSE file for details.
