# llama.cpp Skill

A comprehensive skill for working with the llama.cpp C API. Provides complete API reference, working examples, and best practices for building LLM applications with llama.cpp.

## Table of Contents

- [What This Skill Provides](#what-this-skill-provides)
- [When to Use](#when-to-use)
- [Skill Triggers](#skill-triggers)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Packaging the Skill](#packaging-the-skill)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Links](#links)
- [Version History](#version-history)

## What This Skill Provides

- **Complete API Reference**: All non-deprecated llama.cpp C functions organized by category
- **Working Examples**: Production-ready code for common tasks (text generation, chat, embeddings, batch processing, streaming)
- **Workflow Guides**: Step-by-step examples for typical use cases
- **Best Practices**: Patterns for efficient and correct API usage

## When to Use

Use this skill to:
- Look up llama.cpp C API functions and their signatures
- Generate code that uses the llama.cpp library
- Implement text generation, embeddings extraction, or chat applications
- Work with batches, sequences, KV cache management, LoRA adapters, or state persistence
- Get answers about llama.cpp API usage

## Skill Triggers

This skill automatically activates when you:

### Direct llama.cpp References
- Mention "llama.cpp" or "llama-cpp" in your question
- Reference the ggerganov/llama.cpp repository
- Ask about GGUF model files or formats
- Use llama.cpp function names (e.g., `llama_model_load_from_file`)

### LLM Development Tasks
- Ask "how do I use llama.cpp to..."
- Request code for local LLM inference
- Need C/C++ code for AI model integration
- Want to implement text generation, chat, or embeddings locally

### Specific Features
- Questions about model loading, tokenization, or inference
- Batching, KV cache, or sequence management questions
- LoRA adapter integration
- State persistence or session management
- Sampling strategies and configuration

### Common Query Patterns
- "How do I load a GGUF model?"
- "Write C code to generate text with llama.cpp"
- "What function extracts embeddings?"
- "How to implement chat with llama.cpp?"
- "Troubleshooting llama.cpp inference issues"

**Note:** The skill uses keyword matching on queries and code context to determine when to activate.

## Repository Structure

```
plugins/llamacpp/                  # Plugin directory
├── README.md                      # This file - public documentation
├── package.sh                     # Script to package skill into .skill file
│
├── llama.cpp/                     # Upstream llama.cpp (separate git repo)
│   ├── .git/                      # llama.cpp's own git history
│   ├── include/llama.h            # Main C API header (source of truth)
│   ├── include/llama-cpp.h        # C++ wrapper header
│   └── ...                        # Full llama.cpp source code
│
└── skills/llamacpp/               # Skill content directory
    ├── SKILL.md                   # Main skill entry point
    └── references/                # Detailed API documentation (split by category)
        ├── api-core.md            # Initialization, parameters, model loading
        ├── api-model-info.md      # Model properties and architecture
        ├── api-context.md         # Context, memory (KV cache), state
        ├── api-inference.md       # Batch, inference, tokenization, chat
        ├── api-sampling.md        # Sampling strategies (XTC, DRY, etc.)
        ├── api-advanced.md        # LoRA, performance, training
        ├── api.md                 # Legacy complete API reference
        └── workflows.md           # Working examples and patterns
```

**Important:** The `llama.cpp/` directory is a **separate git repository** cloned from https://github.com/ggml-org/llama.cpp. It is NOT a git submodule. This allows us to easily update to new llama.cpp versions and reference the source API headers.

## Installation

### Option 1: Install from Marketplace

The skill can be installed directly from [skillsmp.com](https://skillsmp.com).

### Option 2: Manual Installation

1. Clone the marketplace repository:
   ```bash
   git clone https://github.com/datathings/marketplace.git
   cd marketplace/plugins/llamacpp
   ```

2. Package the skill:
   ```bash
   ./package.sh
   ```

3. Install the generated `llamacpp.skill` file in your skills directory.

## Features

### API Reference (references/api.md)

Complete documentation of all non-deprecated llama.cpp functions organized into 15 categories:
- Initialization & Backend
- Parameter Helpers
- Model Loading & Management
- Model Properties & Metadata
- Context Management
- Memory (KV Cache) Management
- State & Session Management
- Batch Operations
- Inference & Decoding
- Vocabulary & Tokenization
- Chat Templates
- Sampling
- LoRA Adapters
- Performance & Utilities
- Training

### Workflow Examples (references/workflows.md)

Working examples demonstrating:
- Basic text generation
- Chat with system prompts
- Embeddings extraction
- Batch processing
- Multiple sequence management
- LoRA adapter usage
- State save/load
- Custom sampling strategies
- Encoder-decoder models
- Memory management patterns

### Production Examples (references/examples.md)

Complete, production-ready applications:
- Simple text generation (minimal example)
- Interactive chat application (conversation history)
- Embeddings extraction (similarity computation)
- Batch text processing (parallel sequences)
- Streaming generation (real-time delivery)

## Compatibility

This skill documents the llama.cpp C API as of **January 2026** (version b7709).

### Version Support

- **Recommended:** llama.cpp b3000 or newer
- **API Coverage:** 173 non-deprecated functions across 16 categories
- **Deprecated Functions:** 30 (excluded from documentation)
- **New in b7709:** Gemma3n multimodal support with MobileNetV5 vision encoder, pooling type improvements
- **New in b7681:** Direct I/O support, per-device memory margins, improved parameter fitting status reporting

### Version Notes

If using older llama.cpp builds (<b3000), some function signatures may differ. Refer to the llama.cpp repository for version-specific documentation.

### Keeping Updated

To update this skill for newer llama.cpp versions:

1. **Update llama.cpp to latest tag:**
   ```bash
   cd llama.cpp
   git fetch --tags
   LATEST_TAG=$(git tag --sort=-version:refname | grep "^b" | head -n 1)
   git checkout "$LATEST_TAG"
   cd ..
   ```

2. **Analyze API changes** by reading `llama.cpp/include/llama.h`
3. **Update skill documentation** in `llamacpp/references/` to match new API
4. **Update version** in `llamacpp/SKILL.md` and `marketplace.json`
5. **Re-package the skill** with `./package.sh`

For a systematic update process, use the `/update` command in Claude Code.

## Usage

Once installed, this skill will automatically activate when you ask questions about llama.cpp or request code that uses the llama.cpp C API.

Example queries:
- "How do I load a model in llama.cpp?"
- "Write a C program to generate text using llama.cpp"
- "What function do I use to extract embeddings?"
- "Show me how to implement chat with llama.cpp"
- "How do I manage the KV cache for multiple sequences?"

## Packaging the Skill

To create a distributable `.skill` file:

```bash
./package.sh
```

This will create `llamacpp.skill` in the current directory. You can optionally specify an output directory:

```bash
./package.sh -o /path/to/output
```

## Development

### Updating the Skill

1. Modify files in the `llamacpp/` directory
2. Re-package the skill: `./package.sh`
3. Test with your AI assistant

### File Organization

- **SKILL.md**: Main skill file with overview and navigation. Keep this concise (< 300 lines)
- **references/api.md**: Complete API function reference
- **references/workflows.md**: Workflow examples showing how to accomplish tasks
- **references/examples.md**: Production-ready complete applications

## Contributing

When contributing to this skill:

1. Keep SKILL.md lean - detailed content goes in reference files
2. Ensure all code examples compile and work correctly
3. Update the API reference when llama.cpp adds new functions
4. Include error handling in examples
5. Follow the existing organization pattern

## License

This skill documentation is provided as-is. Please refer to the llama.cpp project for the library's license.

## Links

- llama.cpp: https://github.com/ggerganov/llama.cpp
- Datathings: https://datathings.com/
- Skills Marketplace: https://skillsmp.com/
- Repository: https://github.com/datathings/marketplace

## Version History

- **1.1.0** (2026-01-30): Updated to llama.cpp b7885 (no API changes, implementation improvements)
- **1.0.0**: Initial release with complete API reference and examples
