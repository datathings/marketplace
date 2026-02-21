# Update ggml Skill to Latest Version

This command updates the ggml library to the latest version and synchronizes all skill files with the updated API.

**Working directory:** `plugins/ggml/`

## Repository Structure

```
plugins/ggml/
├── skills/ggml/              # Skill content (SKILL.md and references/)
│   ├── SKILL.md              # Main skill entry point
│   └── references/
│       ├── api-core.md       # Context, tensors, graph management
│       ├── api-arithmetic.md # Arithmetic, matrix ops, quantization
│       ├── api-activations.md # Activations, norms, shapes, custom ops
│       ├── api-attention.md  # RoPE, Flash Attention, conv, padding
│       ├── api-backend.md    # Backends, memory, scheduler, CPU, utilities
│       ├── api-gguf.md       # GGUF v3 file format
│       ├── api-optimization.md # Training, AdamW/SGD, datasets
│       └── workflows.md      # Complete working examples
├── ggml-repo/                # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json # Plugin manifest
```

The `ggml-repo/` directory is a **standalone git clone** from https://github.com/ggml-org/ggml — updated independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

```bash
cd plugins/ggml/ggml-repo

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# ggml uses v#.#.# versioning
LATEST_TAG=$(git tag --sort=-version:refname | grep "^v" | head -n 1)
echo "Latest tag: $LATEST_TAG"

if [ "$CURRENT_TAG" != "unknown" ] && [ "$CURRENT_TAG" != "$LATEST_TAG" ]; then
  echo "Commits between $CURRENT_TAG and $LATEST_TAG:"
  git log --oneline "$CURRENT_TAG..$LATEST_TAG" | head -n 50
fi

git checkout "$LATEST_TAG"
echo "Now on: $(git describe --tags)"

cd ../../..
```

### 2. Analyze API Changes

The source of truth for the ggml public API is the headers in `plugins/ggml/ggml-repo/include/`:

- `ggml.h` — core: all tensor types, ops (75+), context, quantization, graph
- `gguf.h` — GGUF v3 binary format KV and tensor metadata
- `ggml-backend.h` — backend system: buffers, devices, registry, scheduler
- `ggml-alloc.h` — tensor and graph allocators
- `ggml-opt.h` — training: datasets, AdamW/SGD optimizer, epoch loop
- `ggml-cpu.h` — CPU backend: threadpool, feature detection, type conversions

Look for:
- New functions/structs added since `$CURRENT_TAG`
- Deprecated or removed functions
- Modified signatures (parameter additions, return type changes)
- New enum values (ggml_type, ggml_op, ggml_unary_op, etc.)

### 3. Update Skill Documentation Files

Update the following files in `skills/ggml/references/`:

| File | Source of Truth | What to Update |
|------|----------------|----------------|
| `api-core.md` | `ggml.h` (context, tensor, graph sections) | New tensor creation fns, graph ops, constants |
| `api-arithmetic.md` | `ggml.h` (arithmetic, matrix, quantize) | New ops, loss fns, quantize types |
| `api-activations.md` | `ggml.h` (unary, GLU, norm, shape, custom) | New activations, norm variants, shape ops |
| `api-attention.md` | `ggml.h` (rope, softmax, flash_attn, conv) | New RoPE modes, attention variants, conv ops |
| `api-backend.md` | `ggml-backend.h`, `ggml-alloc.h`, `ggml-cpu.h` | New backends, buffer types, CPU features |
| `api-gguf.md` | `gguf.h` | New KV types, tensor helpers |
| `api-optimization.md` | `ggml-opt.h` | New optimizer params, training helpers |

### 4. Update Main SKILL.md

- Update the library version in the **Version** line (e.g., `**Version:** v0.9.7` → new version)
- Update function/op counts in the description if significantly changed
- Add notes about newly supported features or architectures
- Update the API reference table if reference files were added/removed

### 5. Remove Deprecated Content

- Search reference files for functions removed from the headers
- For renamed functions: update all references to the new name
- Note breaking changes prominently in SKILL.md if any signatures changed

### 6. Update workflows.md

- Verify all code examples compile against the new API
- Add examples for major new features (new backends, ops, training helpers)
- Update imports if header file structure changed

### 7. Validate Changes

Cross-reference each reference file against the corresponding header:
- Every documented function signature must exactly match the header
- Every new public function in headers must be documented
- No removed functions should remain in documentation

### 8. Update plugin.json

```bash
# Update version and description in plugins/ggml/.claude-plugin/plugin.json
# e.g.: "description": "ggml <NEW_TAG> C tensor library skill — ..."
```

### 9. Package Updated Skill

```bash
./package.sh ggml
```

This creates `skills/ggml.skill`.

## Key Considerations

- **Source of truth**: Always derive signatures from the `include/*.h` headers, not secondary docs or examples
- **Enum completeness**: The `ggml_type` and `ggml_op` enums grow frequently — check for new entries
- **Inplace variants**: New ops often get `_inplace` variants; document both
- **Backend-specific headers**: `ggml-cuda.h`, `ggml-metal.h` etc. are separate — focus on the cross-platform API in `ggml-backend.h`
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them

## Success Criteria

- ✅ Library updated to latest stable tag
- ✅ SKILL.md Version line updated to new tag
- ✅ All new public API functions from headers are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match headers exactly
- ✅ Code examples in workflows.md are valid for new API
- ✅ plugin.json version and description updated
- ✅ Package builds successfully with `./package.sh ggml`
