# GreyCat Language Server Protocol (LSP)

**Version**: 7.6.0-dev | **Protocol**: JSON-RPC 2.0 over stdio | **Language**: GreyCat (GCL)

## Overview

The GreyCat LSP provides IDE-like features for `.gcl` files through the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/). Works with VS Code, IntelliJ, Neovim, Emacs, and any LSP-compatible editor.

**Architecture**: `Editor (Client) ◄─── JSON-RPC over stdio ───► greycat-lang server ──► Project Files (*.gcl)`

**Benefits**:
| Feature | Without LSP | With LSP |
|---------|-------------|----------|
| Error Detection | After `greycat-lang lint` | As you type |
| Type Information | Manual docs | Hover tooltip |
| Navigation | Text search | Go to definition (Ctrl+Click) |
| Refactoring | Find/replace (risky) | Safe rename across project |
| Code Discovery | Browse files | Autocomplete members |
| Formatting | Manual | Auto-format on save |

## Quick Start

### Starting the Server

**IDE Integration** (most common):
```bash
greycat-lang server --stdio
```
Server communicates via stdin/stdout using JSON-RPC 2.0. Logs go to stderr.

**Background Service** (for programmatic access):
```bash
mkfifo /tmp/greycat-lsp-{in,out}
greycat-lang server --stdio < /tmp/greycat-lsp-in > /tmp/greycat-lsp-out 2> /tmp/greycat-lsp.log &
LSP_PID=$!

# Cleanup: kill $LSP_PID && rm /tmp/greycat-lsp-{in,out,log}
```

**When to use background LSP**:
- ✅ Programmatic queries during development (Claude Code integration)
- ✅ Custom tooling needing real-time code intelligence
- ✅ Batch processing (avoid startup overhead)
- ❌ Regular IDE usage (IDE manages lifecycle)
- ❌ CI/CD (use `greycat-lang lint` instead)

### IDE Configuration

**VS Code**:
```json
{
  "greycat.languageServer.path": "/path/to/greycat-lang",
  "greycat.languageServer.args": ["server", "--stdio"]
}
```

**Neovim** (nvim-lspconfig):
```lua
require'lspconfig'.greycat.setup{
  cmd = {"greycat-lang", "server", "--stdio"},
  filetypes = {"greycat"},
  root_dir = require'lspconfig'.util.root_pattern("project.gcl", ".git")
}
```

**Emacs** (lsp-mode):
```elisp
(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection '("greycat-lang" "server" "--stdio"))
                  :major-modes '(greycat-mode)
                  :server-id 'greycat))
```

### Verify Installation
```bash
greycat-lang --version
echo 'Content-Length: 2\r\n\r\n{}' | greycat-lang server --stdio  # Should return JSON-RPC
```

## LSP Capabilities

**Core capabilities**:
1. **textDocumentSync** - Real-time file tracking (incremental sync)
2. **completionProvider** - Type-aware autocomplete (`.`, `>`, `:`, `@` triggers)
3. **hoverProvider** - Inline documentation
4. **definitionProvider** - Go to definition
5. **referencesProvider** - Find all references
6. **renameProvider** - Safe project-wide renaming
7. **documentSymbolProvider** - File outline
8. **documentFormattingProvider** - Code formatting
9. **signatureHelpProvider** - Parameter hints
10. **codeActionProvider** - Quick fixes
11. **inlayHintProvider** - Inline type annotations
12. **codeLensProvider** - Actionable insights
13. **semanticTokensProvider** - AST-based syntax highlighting
14. **workspace** - Multi-file support, file watching

## Protocol Communication

### Message Format
All messages use Content-Length headers:
```
Content-Length: 123\r\n
\r\n
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
```

### Initialization Sequence

**1. Client → Server: initialize**
```json
{
  "jsonrpc": "2.0", "id": 1, "method": "initialize",
  "params": {
    "processId": 12345,
    "rootUri": "file:///home/user/project",
    "capabilities": {"textDocument": {"completion": {}, "hover": {}, "definition": {}}}
  }
}
```

**2. Server → Client: capabilities**
```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "serverInfo": {"name": "GreyCat LSP", "version": "7.6.0-dev"},
    "capabilities": {
      "completionProvider": {"triggerCharacters": [".", ">", ":", "@"]},
      "hoverProvider": true,
      "definitionProvider": true
    }
  }
}
```

**3. Client → Server: initialized (notification)**
```json
{"jsonrpc": "2.0", "method": "initialized", "params": {}}
```

**4. Server → Client: diagnostics (automatic)**
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "uri": "file:///path/to/file.gcl",
    "diagnostics": [{
      "range": {"start": {"line": 10, "character": 5}, "end": {"line": 10, "character": 15}},
      "severity": 1,  // 1=Error, 2=Warning, 3=Info, 4=Hint
      "message": "Type mismatch: expected String, got int"
    }]
  }
}
```

### Shutdown Sequence
```json
// 1. Client → Server: shutdown
{"jsonrpc": "2.0", "id": 99, "method": "shutdown"}

// 2. Server → Client: acknowledge
{"jsonrpc": "2.0", "id": 99, "result": null}

// 3. Client → Server: exit (no response)
{"jsonrpc": "2.0", "method": "exit"}
```

## Programmatic Integration

### Python Client Example

```python
import subprocess, json, time, select

class GreyCatLSP:
    def __init__(self, root_path):
        self.proc = subprocess.Popen(
            ["greycat-lang", "server", "--stdio"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self.message_id = 0
        self.root_uri = f"file://{root_path}"
        self._initialize()

    def _send(self, message):
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.proc.stdin.write((header + content).encode())
        self.proc.stdin.flush()

    def _read_responses(self, timeout=2.0):
        responses = []
        end_time = time.time() + timeout
        buffer = b""

        while time.time() < end_time:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0.1)
            if ready:
                chunk = self.proc.stdout.read(4096)
                if not chunk: break
                buffer += chunk

                while b"Content-Length:" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1: break

                    header = buffer[:header_end].decode()
                    length = int(header.split(":")[1].strip())
                    content_start = header_end + 4
                    content_end = content_start + length

                    if len(buffer) < content_end: break

                    content = buffer[content_start:content_end].decode()
                    responses.append(json.loads(content))
                    buffer = buffer[content_end:]

            if responses and time.time() > end_time - 0.5: break

        return responses

    def _initialize(self):
        self.message_id += 1
        self._send({
            "jsonrpc": "2.0", "id": self.message_id, "method": "initialize",
            "params": {
                "processId": None, "rootUri": self.root_uri,
                "capabilities": {"textDocument": {"completion": {}, "hover": {}, "definition": {}, "references": {}}}
            }
        })
        self._read_responses(2.0)
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        time.sleep(0.3)

    def open_file(self, file_path, content):
        self._send({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {
                "textDocument": {"uri": f"file://{file_path}", "languageId": "greycat", "version": 1, "text": content}
            }
        })
        time.sleep(0.3)
        return self._read_responses(1.0)

    def get_completion(self, file_path, line, character):
        self.message_id += 1
        msg_id = self.message_id
        self._send({
            "jsonrpc": "2.0", "id": msg_id, "method": "textDocument/completion",
            "params": {"textDocument": {"uri": f"file://{file_path}"}, "position": {"line": line, "character": character}}
        })
        responses = self._read_responses(2.0)
        for resp in responses:
            if resp.get("id") == msg_id and "result" in resp:
                return resp["result"]
        return None

    def get_hover(self, file_path, line, character):
        self.message_id += 1
        msg_id = self.message_id
        self._send({
            "jsonrpc": "2.0", "id": msg_id, "method": "textDocument/hover",
            "params": {"textDocument": {"uri": f"file://{file_path}"}, "position": {"line": line, "character": character}}
        })
        responses = self._read_responses(2.0)
        for resp in responses:
            if resp.get("id") == msg_id and "result" in resp:
                return resp["result"]
        return None

    def shutdown(self):
        self._send({"jsonrpc": "2.0", "id": 99, "method": "shutdown"})
        time.sleep(0.2)
        self.proc.terminate()
```

## Common Use Cases

### Claude Code Integration
```python
class ClaudeLSPHelper:
    """Helper for Claude Code to query LSP during development"""

    def __init__(self, project_root):
        self.lsp = GreyCatLSP(project_root)

    def get_type_at_position(self, file_path, line, char):
        with open(file_path) as f:
            content = f.read()
        self.lsp.open_file(file_path, content)
        hover = self.lsp.get_hover(file_path, line, char)
        return hover["contents"]["value"] if hover and hover.get("contents") else None

    def get_completions_for_member(self, file_path, line, char):
        with open(file_path) as f:
            content = f.read()
        self.lsp.open_file(file_path, content)
        result = self.lsp.get_completion(file_path, line, char)
        items = result if isinstance(result, list) else result.get("items", [])
        return [item.get("label") for item in items]

    def verify_no_errors(self, file_path):
        with open(file_path) as f:
            content = f.read()
        diagnostics = self.lsp.open_file(file_path, content)
        for response in diagnostics:
            if response.get("method") == "textDocument/publishDiagnostics":
                errors = [d for d in response["params"].get("diagnostics", []) if d.get("severity") == 1]
                if errors:
                    return False, errors
        return True, []
```

**Benefits**:
- ✅ Real-time type checking without `greycat-lang lint`
- ✅ Intelligent API/field suggestions
- ✅ Catch errors before committing
- ✅ Faster development iteration

### Pre-Commit Validation
```bash
#!/bin/bash
# pre-commit hook: validate .gcl files with LSP

for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.gcl$'); do
  echo "Validating $file..."
  python3 validate_gcl.py "$file" || exit 1
done
echo "✅ All files validated"
```

## Troubleshooting

### Common Issues

**LSP not starting**:
```bash
which greycat-lang
greycat-lang --version
echo 'Content-Length: 2\r\n\r\n{}' | greycat-lang server --stdio  # Test manually
```

**No completions/hover**:
- Ensure file opened via `textDocument/didOpen`
- Check trigger characters: `.`, `>`, `:`, `@`
- Verify workspace root is correct
- Allow 2-5 seconds for initial indexing

**Diagnostics not updating**:
- Check file sync: `textDocument/didChange` after edits
- Verify `textDocumentSync` capability enabled
- Check stderr logs for errors

**Performance issues**:
- Large projects (>1000 files) take time to index
- Consider splitting into multiple workspaces
- Check memory limits

### Debug Logging
```bash
export GREYCAT_LSP_DEBUG=1
greycat-lang server --stdio 2> lsp-debug.log
```

## LSP vs. CLI Tools

| Feature | LSP (Real-time) | CLI (`greycat-lang`) |
|---------|----------------|---------------------|
| Error Detection | As you type | After command |
| Performance | Continuous daemon | One-shot execution |
| Context | Full workspace | Single file/command |
| Use Case | Interactive development | CI/CD, scripts |
| Integration | Editor plugins | Shell scripts |
| Validation | Advisory warnings | Authoritative errors |

**Best Practice**:
- Use **LSP** during development for instant feedback
- Use **CLI** (`greycat-lang lint`) as final verification before commit/CI
- LSP diagnostics ≠ compile errors (may show stricter warnings)

## Symbol Kinds (LSP Spec)

| Kind | Name | Description |
|------|------|-------------|
| 1 | File | File-level scope |
| 2 | Module | Module/namespace |
| 5 | Class | Type definition |
| 6 | Method | Type method |
| 12 | Function | Top-level function |
| 14 | Variable | Global/local variable |
| 7 | Property | Type field/property |

## Resources

- **LSP Spec**: https://microsoft.github.io/language-server-protocol/
- **GreyCat Docs**: https://doc.greycat.io/
- **VS Code Extension**: GreyCat Language Support

**Last Updated**: January 2026 | **Server Version**: 7.6.0-dev | **Protocol Version**: LSP 3.17
