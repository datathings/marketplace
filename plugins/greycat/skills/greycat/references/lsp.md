# GreyCat Language Server Protocol (LSP)

**Version**: 7.6.0-dev | **Protocol**: JSON-RPC 2.0 over stdio | **Language**: GreyCat (GCL)

## Overview

The GreyCat LSP provides IDE-like features for `.gcl` files through the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/). Works with VS Code, IntelliJ, Neovim, Emacs, and any LSP-compatible editor.

**Architecture**:
```
Editor (Client) ◄─── JSON-RPC over stdio ───► greycat-lang server ──► Project Files (*.gcl)
```

**Benefits**:
| Feature | Without LSP | With LSP |
|---------|-------------|----------|
| Error Detection | After `greycat-lang lint` | As you type |
| Type Information | Manual docs | Hover tooltip |
| Navigation | Text search | Go to definition (Ctrl+Click) |
| Refactoring | Find/replace (risky) | Safe rename across project |
| Code Discovery | Browse files | Autocomplete members |
| Formatting | Manual | Auto-format on save |

---

## Quick Start

### Starting the Server

**IDE Integration** (most common):
```bash
greycat-lang server --stdio
```
The server communicates via stdin/stdout using JSON-RPC 2.0. Logs go to stderr.

**Background Service** (for programmatic access):
```bash
# Named pipes approach
mkfifo /tmp/greycat-lsp-{in,out}
greycat-lang server --stdio < /tmp/greycat-lsp-in > /tmp/greycat-lsp-out 2> /tmp/greycat-lsp.log &
LSP_PID=$!

# Cleanup: kill $LSP_PID && rm /tmp/greycat-lsp-{in,out,log}
```

**When to use background LSP**:
- ✅ Programmatic queries during development (e.g., Claude Code integration)
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
greycat-lang --version  # Should output: greycat-lang v7.6.0-dev
echo 'Content-Length: 2\r\n\r\n{}' | greycat-lang server --stdio  # Should return JSON-RPC
```

---

## LSP Capabilities (v7.6.0-dev)

### 1. `textDocumentSync` - Real-time File Tracking
```json
{"openClose": true, "save": true, "change": 2}  // Incremental sync
```
Triggers: `didOpen`, `didChange`, `didSave`, `didClose`. LSP maintains synchronized AST.

### 2. `completionProvider` - Type-Aware Autocomplete
```json
{"triggerCharacters": [".", ">", ":", "@"]}
```
**Example Request**:
```json
{
  "jsonrpc": "2.0", "id": 1, "method": "textDocument/completion",
  "params": {
    "textDocument": {"uri": "file:///path/to/file.gcl"},
    "position": {"line": 10, "character": 15}
  }
}
```
**Response**:
```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "items": [{
      "label": "chunks",
      "kind": 5,  // Field
      "detail": "nodeList<node<Chunk>>",
      "documentation": "All chunks in this document"
    }]
  }
}
```
Triggers: `.` (member access), `>` (hierarchy), `:` (namespaces), `@` (decorators)

### 3. `hoverProvider` - Inline Documentation
Shows type signatures and docs on hover.
```json
{
  "contents": {
    "language": "greycat",
    "value": "type Document {\n  celex: String;\n  chunks: nodeList<node<Chunk>>;\n}"
  }
}
```

### 4. `definitionProvider` - Go to Definition
Jump to symbol declaration across project.

### 5. `referencesProvider` - Find All References
```json
{
  "method": "textDocument/references",
  "params": {
    "textDocument": {"uri": "file:///path/to/model.gcl"},
    "position": {"line": 50, "character": 10},
    "context": {"includeDeclaration": true}
  }
}
```

### 6. `renameProvider` - Safe Project-Wide Renaming
Returns `WorkspaceEdit` with all changes across files.

### 7. `documentSymbolProvider` - File Outline
```json
[
  {"name": "Document", "kind": 5, "range": {...}},  // Class/Type
  {"name": "getDocument", "kind": 12, "range": {...}}  // Function
]
```

### 8. `documentFormattingProvider` - Code Formatting
```json
{
  "method": "textDocument/formatting",
  "params": {
    "textDocument": {"uri": "file:///path/to/file.gcl"},
    "options": {"tabSize": 4, "insertSpaces": true}
  }
}
```

### 9. `signatureHelpProvider` - Parameter Hints
```json
{"triggerCharacters": ["("], "retriggerCharacters": [","]}
```
Shows function signatures during calls.

### 10-14. Additional Capabilities
- **codeActionProvider**: Quick fixes for diagnostics
- **inlayHintProvider**: Inline type annotations
- **codeLensProvider**: Actionable insights (e.g., "Run", "Test")
- **semanticTokensProvider**: AST-based syntax highlighting
- **workspace**: Multi-file support, file watching

---

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

---

## Programmatic Integration

### Python Client

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

# Usage
lsp = GreyCatLSP("/path/to/greycat/project")

with open("/path/to/project/backend/src/model/cjue.gcl") as f:
    content = f.read()

lsp.open_file("/path/to/project/backend/src/model/cjue.gcl", content)

completions = lsp.get_completion("/path/to/project/backend/src/model/cjue.gcl", 9, 15)
print(f"Completions: {completions}")

hover = lsp.get_hover("/path/to/project/backend/src/model/cjue.gcl", 9, 15)
print(f"Hover: {hover}")

lsp.shutdown()
```

### JavaScript/TypeScript Client

```javascript
import { spawn } from 'child_process';

class GreyCatLSP {
  constructor(rootPath) {
    this.proc = spawn('greycat-lang', ['server', '--stdio']);
    this.messageId = 0;
    this.rootUri = `file://${rootPath}`;
    this.buffer = Buffer.alloc(0);
    this.callbacks = new Map();
    this.proc.stdout.on('data', (chunk) => this.handleData(chunk));
    this.initialize();
  }

  send(message) {
    const content = JSON.stringify(message);
    const header = `Content-Length: ${Buffer.byteLength(content)}\r\n\r\n`;
    this.proc.stdin.write(header + content);
  }

  handleData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (true) {
      const headerEnd = this.buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;

      const header = this.buffer.slice(0, headerEnd).toString();
      const lengthMatch = header.match(/Content-Length: (\d+)/);
      if (!lengthMatch) break;

      const length = parseInt(lengthMatch[1]);
      const contentStart = headerEnd + 4;
      const contentEnd = contentStart + length;
      if (this.buffer.length < contentEnd) break;

      const content = this.buffer.slice(contentStart, contentEnd).toString();
      const message = JSON.parse(content);

      if (message.id && this.callbacks.has(message.id)) {
        this.callbacks.get(message.id)(message.result);
        this.callbacks.delete(message.id);
      }
      this.buffer = this.buffer.slice(contentEnd);
    }
  }

  request(method, params) {
    return new Promise((resolve) => {
      this.messageId++;
      const id = this.messageId;
      this.callbacks.set(id, resolve);
      this.send({ jsonrpc: '2.0', id, method, params });
      setTimeout(() => {
        if (this.callbacks.has(id)) {
          this.callbacks.delete(id);
          resolve(null);
        }
      }, 5000);
    });
  }

  async initialize() {
    await this.request('initialize', {
      processId: process.pid,
      rootUri: this.rootUri,
      capabilities: { textDocument: { completion: {}, hover: {}, definition: {} } }
    });
    this.send({ jsonrpc: '2.0', method: 'initialized', params: {} });
  }

  openFile(filePath, content) {
    this.send({
      jsonrpc: '2.0', method: 'textDocument/didOpen',
      params: {
        textDocument: {uri: `file://${filePath}`, languageId: 'greycat', version: 1, text: content}
      }
    });
  }

  async getCompletion(filePath, line, character) {
    return this.request('textDocument/completion', {
      textDocument: { uri: `file://${filePath}` },
      position: { line, character }
    });
  }

  shutdown() {
    this.send({ jsonrpc: '2.0', id: 99, method: 'shutdown' });
    setTimeout(() => this.proc.kill(), 200);
  }
}

// Usage
const lsp = new GreyCatLSP('/path/to/project');
setTimeout(async () => {
  const content = require('fs').readFileSync('/path/to/file.gcl', 'utf8');
  lsp.openFile('/path/to/file.gcl', content);
  const completions = await lsp.getCompletion('/path/to/file.gcl', 10, 15);
  console.log('Completions:', completions);
  lsp.shutdown();
}, 1000);
```

---

## Common Use Cases

### 1. Pre-Commit Validation
```bash
#!/bin/bash
# pre-commit hook: validate .gcl files with LSP

for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.gcl$'); do
  echo "Validating $file..."
  # Use Python LSP client to check for diagnostics
  python3 validate_gcl.py "$file" || exit 1
done
echo "✅ All files validated"
```

### 2. Type Checking During Development
```python
#!/usr/bin/env python3
# check_types.py - Query LSP for type information

from greycat_lsp_client import GreyCatLSP

def check_file_types(file_path):
    lsp = GreyCatLSP("/path/to/greycat/project")
    with open(file_path) as f:
        content = f.read()

    lsp.open_file(file_path, content)
    symbols = lsp.get_document_symbols(file_path)

    print(f"📋 Type information for {file_path}:")
    for symbol in symbols:
        hover = lsp.get_hover(file_path, symbol["range"]["start"]["line"], 0)
        if hover and hover.get("contents"):
            print(f"  • {symbol['name']}: {hover['contents']}")

    lsp.shutdown()
```

### 3. Claude Code Integration
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

# Usage
helper = ClaudeLSPHelper("/path/to/project")

# Check type before editing
type_info = helper.get_type_at_position("/path/to/model.gcl", 9, 15)
print(f"Type: {type_info}")

# Get member suggestions while editing
completions = helper.get_completions_for_member("/path/to/api.gcl", 50, 12)
print(f"Available: {completions}")

# Verify no errors after editing
ok, errors = helper.verify_no_errors("/path/to/service.gcl")
print("✅ No errors" if ok else f"❌ Errors: {errors}")
```

**Benefits**:
- ✅ Real-time type checking without `greycat-lang lint`
- ✅ Intelligent API/field suggestions
- ✅ Catch errors before committing
- ✅ Faster development iteration

### 4. Documentation Generation
```python
lsp = GreyCatLSP("/path/to/project")

for gcl_file in glob.glob("backend/src/**/*.gcl", recursive=True):
    with open(gcl_file) as f:
        content = f.read()
    lsp.open_file(gcl_file, content)
    symbols = lsp.get_document_symbols(gcl_file)

    # Generate markdown docs from symbols
    for symbol in symbols:
        if symbol['kind'] == 5:  # Type
            print(f"## {symbol['name']}")
```

### 5. Safe Refactoring
```python
def safe_rename(lsp, file_path, line, char, new_name):
    refs = lsp.find_references(file_path, line, char)
    print(f"Found {len(refs)} references:")
    for ref in refs:
        print(f"  - {ref['uri']}:{ref['range']['start']['line']}")

    if input("Proceed? (y/n): ").lower() == 'y':
        edits = lsp.rename(file_path, line, char, new_name)
        # Apply edits...
```

---

## Troubleshooting

### Common Issues

**LSP not starting**:
```bash
which greycat-lang          # Check installation
greycat-lang --version      # Verify version
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

### Protocol Debugging
```bash
greycat-lang server --stdio 2>&1 | tee lsp-protocol.log  # Capture JSON-RPC traffic
```

---

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

---

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

---

## Resources

- **LSP Spec**: https://microsoft.github.io/language-server-protocol/
- **GreyCat Docs**: https://doc.greycat.io/
- **VS Code Extension**: GreyCat Language Support
---

**Last Updated**: January 2026 | **Server Version**: 7.6.0-dev | **Protocol Version**: LSP 3.17
