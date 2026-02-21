# Datathings Marketplace

We are [Datathings](https://datathings.com), specializing in high-performance software for large-scale data infrastructure. Our foundation is [GreyCat](https://greycat.io) — a temporal graph database and programming language built for efficiency at scale, with native agentic AI capabilities. On that foundation, we built [Kopr](https://kopr-twin.com): a digital twin managing Luxembourg's entire electricity distribution grid — 1 million grid assets, 330,000 delivery points, and 45 billion meter readings per year, with machine learning running continuously over live sensor data.

The plugins here bring that stack to your AI agent: GreyCat's runtime and language tools, the numerical and GPU computing libraries behind high-performance inference and optimization, and widely-used power systems analysis frameworks for anyone building in that domain.

## Quick Start

**Install GreyCat:**

Linux, Mac or FreeBSD (x64, arm64):
```bash
curl -fsSL https://get.greycat.io/install.sh | bash -s dev
```

Windows (x64, arm64):
```powershell
iwr https://get.greycat.io/install_dev.ps1 -useb | iex
```

**⚠️ Important: Restart your terminal after installing GreyCat**

**Install Claude Code:**

Follow the installation instructions at [https://code.claude.com/docs/en/setup](https://code.claude.com/docs/en/setup)

**Add the marketplace:**

```
/plugin marketplace add datathings/marketplace
```

**Install plugins:**

```
/plugin install greycat@datathings
```

**Try it:**

Run Claude in a demo folder, then paste this prompt:
```
Use your greycat skill to create a GreyCat backend with Country, City, Street, House, and Person nodes linked as a geographic hierarchy with back references for bidirectional navigation (country contains cities, cities contain streets, etc., and children reference their parents). Add geo coordinates (latitude, longitude) to appropriate nodes. Houses should have temperature sensors storing time series data. Generate two sample CSV files: `./data/addresses.csv` (with house IDs) and `./data/temperatures.csv` (with house_id, date, value columns), and create an importer that loads both on startup (import the CSVs on main if the country index size is 0). Expose all important API endpoints. Create comprehensive API documentation and expose meaningful functions as MCP.
```

## Plugins

| Plugin | Category | Type | Version | Description |
|--------|----------|------|---------|-------------|
| **greycat** | GreyCat Technology | Skill | 2.3.0 | Full-stack GreyCat development — GCL language, graph persistence, React integration |
| **greycat-c** | GreyCat Technology | Skill | 2.3.0 | GreyCat C API and Standard Library for native development |
| **greycat-lsp** | GreyCat Technology | LSP | 2.3.0 | Language Server Protocol for `.gcl` files (completion, diagnostics, hover) |
| **llamacpp** | Agentic AI | Skill | 2.3.0 | llama.cpp C API reference (163 functions) for local LLM inference |
| **blas_lapack** | High Performance Math & GPU | Skill | 2.3.0 | CBLAS & LAPACKE C API reference (1284 functions) for numerical linear algebra |
| **ggml** | Agentic AI | Skill | 2.3.0 | ggml C tensor library (560+ functions) for graph computation, GGUF I/O, multi-backend inference, and ML training |
| **cuda** | High Performance Math & GPU | Skill | 2.3.0 | NVIDIA CUDA C/C++ — Runtime API, cuBLAS, cuFFT, cuSPARSE, cuRAND, cuSolver, Thrust, Cooperative Groups |
| **opencl** | High Performance Math & GPU | Skill | 2.3.0 | OpenCL SDK (Khronos) — cross-platform GPU/CPU parallel computing, C API (~60 functions), C++ wrapper (opencl.hpp), SDK utilities |
| **rocm** | High Performance Math & GPU | Skill | 2.3.0 | AMD ROCm 7.2.0 — HIP kernel development, rocBLAS/rocFFT/rocRAND/rocSOLVER libraries, profiling, and CUDA-to-HIP porting |
| **pandapower** | Power Grid Management | Skill | 2.3.0 | pandapower v3.4.0 — Python power systems analysis with AC/DC power flow, OPF, short circuit (IEC 60909), state estimation, and visualization |
| **powergridmodel** | Power Grid Management | Skill | 2.3.0 | power-grid-model v1.13.10 — high-performance Python library for steady-state distribution power system analysis: power flow, state estimation, and IEC 60909 short-circuit calculations |
| **vllm** | Agentic AI | Skill | 2.3.0 | vLLM v0.16.0 — high-throughput Python LLM inference with offline batch, OpenAI-compatible server, LoRA adapters, multimodal inputs, and structured outputs |
| **ollama** | Agentic AI | Skill | 2.3.0 | Ollama v0.16.3 — run and interact with local LLMs via REST API (chat, generate, embed, model management) |

---

## GreyCat Technology

GreyCat is both a database and a programming language — stateful, graph-native, and designed to expose functions directly as HTTP APIs or MCP endpoints. Install these plugins when building with GreyCat or extending it at the native level.

```
/plugin install greycat@datathings
/plugin install greycat-c@datathings
/plugin install greycat-lsp@datathings
```

## Agentic AI

The inference stack for running AI locally: ggml provides the tensor computation engine and GGUF model format, while llama.cpp builds a complete LLM inference API on top of it. Both plug directly into GreyCat-backed applications or any native pipeline.

```
/plugin install llamacpp@datathings
/plugin install ggml@datathings
/plugin install vllm@datathings
/plugin install ollama@datathings
```

## High Performance Math & GPU Computing

The compute stack for high-performance numerical work: foundational linear algebra (BLAS/LAPACK) and full GPU acceleration across NVIDIA CUDA, OpenCL, and AMD ROCm.

```
/plugin install blas_lapack@datathings
/plugin install cuda@datathings
/plugin install opencl@datathings
/plugin install rocm@datathings
```

## Power Grid Management

[Kopr](https://kopr-twin.com) — our electricity distribution digital twin built on GreyCat — manages 1 million grid assets and 45 billion annual meter readings, with optimal power flow and grid analysis capabilities built in. We include these libraries because they were instrumental in our testing and validation work.

```
/plugin install pandapower@datathings
/plugin install powergridmodel@datathings
```

---

## Plugin Details

### GreyCat Technology

#### greycat

Activates on `.gcl` files and GreyCat topics. Provides:
- GCL syntax, types, decorators (@expose, @permission, @volatile)
- Indexed collections (nodeIndex, nodeList, nodeTime, nodeGeo)
- Concurrency patterns (Jobs, await)
- Standard library (core, io, runtime, util)
- Pro libraries (ai, algebra, finance, kafka, opcua, powerflow, s3, sql, useragent)
- React integration (@greycat/web)

#### greycat-lsp

IDE features for `.gcl` files. **Requires** `greycat-lang` in PATH.
- Code completion, diagnostics, hover info
- Go to definition, find references

```bash
which greycat-lang  # Verify installation
```

#### greycat-c

Reference for native C development with GreyCat:
- C API functions, tensor operations
- Native function implementation

### Agentic AI

#### llamacpp

Complete llama.cpp C API (163 functions):
- Model loading, inference, tokenization
- Sampling strategies (XTC, DRY, infill)
- GGUF model support

#### ggml

C tensor computation library powering llama.cpp and many ML inference engines (v0.9.7, 560+ functions):
- Lazy computation graph with CPU/GPU/Metal/Vulkan backends and automatic multi-backend scheduling
- 40+ quantization formats (Q4_0 → Q5_K), GGUF v3 I/O, Flash Attention, RoPE, AdamW/SGD training

#### vllm

vLLM (v0.16.0) — high-throughput Python inference engine for large language models:
- Offline batch inference (`LLM` class) and OpenAI-compatible server (`vllm serve`) with streaming
- LoRA adapters, multimodal inputs, structured outputs (JSON/regex/grammar), and paged attention

#### ollama

Ollama (v0.16.3) — local LLM runtime with a simple REST API on localhost:
- Text generation, chat, and embeddings via REST API with streaming support
- Model management (pull/push/delete) and custom model creation via Modelfile

### High Performance Math & GPU Computing

#### blas_lapack

Complete CBLAS & LAPACKE C API (1284 functions, LAPACK v3.12.1):
- BLAS Level 1/2/3 vector and matrix operations; linear solvers (LU, Cholesky, LDL)
- Eigenvalue/SVD/least squares decompositions; QR/LQ factorizations

#### cuda

NVIDIA CUDA parallel computing platform (cuda-samples v13.1, CUDALibrarySamples main). Complete reference for GPU-accelerated C/C++ development:
- Runtime API (device, memory, streams, kernel launch); math libraries: cuBLAS, cuFFT, cuSPARSE, cuRAND, cuSolver
- Thrust (STL-like GPU algorithms) and Cooperative Groups (thread synchronization)

#### opencl

Khronos Group OpenCL SDK (v2025.07.23) for cross-platform GPU/CPU parallel computing in C and C++:
- C API (~60 functions) and C++ wrapper (opencl.hpp with RAII types) for platform/device management, memory, and kernels
- Full NDRange execution (1D/2D/3D), events, profiling, and out-of-order queues

#### rocm

AMD ROCm GPU computing stack (rocm-7.2.0) for HIP-based GPU development:
- HIP C++ kernels with full compute libraries (rocBLAS, rocFFT, rocRAND, rocSOLVER, rocSPARSE, rocWMMA)
- Profiling (rocProfiler, rocm-smi) and CUDA portability via hipify-perl

### Power Grid Management

#### pandapower

pandapower (v3.4.0) — Python library for modeling and analyzing electric power networks:
- AC/DC power flow, optimal power flow, short-circuit (IEC 60909), and state estimation
- 15+ benchmark networks (IEEE, CIGRE, Kerber) and visualization with matplotlib/plotly

#### powergridmodel

power-grid-model (v1.13.10) — high-performance Python/C++ library for steady-state distribution power system analysis:
- Symmetric and asymmetric three-phase power flow, state estimation, and IEC 60909 short-circuit analysis
- Batch/N-1 contingency analysis with multi-threaded parallel execution; 22 component types

---

## Standalone Skill Files

The `./skills/` folder contains pre-packaged `.skill` files (zip archives) for use with other AI tools or manual installation:

```
skills/
├── greycat.skill       # GreyCat full-stack development
├── greycat-c.skill     # GreyCat C API reference
├── llamacpp.skill      # llama.cpp C API reference
├── blas_lapack.skill   # CBLAS & LAPACKE C API reference
├── ggml.skill          # ggml C tensor library
├── cuda.skill          # NVIDIA CUDA C/C++ GPU programming
├── opencl.skill        # OpenCL cross-platform GPU/CPU parallel computing
├── rocm.skill          # AMD ROCm GPU computing (HIP + libraries)
├── pandapower.skill    # pandapower Python power systems analysis
├── powergridmodel.skill  # power-grid-model Python distribution power system analysis
├── vllm.skill          # vLLM high-throughput Python LLM inference
└── ollama.skill        # Ollama local LLM runtime REST API
```

Each `.skill` file contains a `SKILL.md` with instructions and optional `references/` documentation. To regenerate:

```bash
./package.sh           # Interactive skill selection
./package.sh -a        # Package all skills
./package.sh greycat   # Package specific skill
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
/plugin list                              # List installed plugins
/plugin update greycat@datathings         # Update a plugin
/plugin uninstall greycat@datathings      # Remove a plugin
/plugin marketplace list                  # List marketplaces
/plugin marketplace remove datathings     # Remove marketplace
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
- **Kopr**: https://kopr-twin.com
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **ggml**: https://github.com/ggml-org/ggml
- **LAPACK**: https://github.com/Reference-LAPACK/lapack
- **CUDA samples**: https://github.com/NVIDIA/cuda-samples
- **CUDA Library Samples**: https://github.com/NVIDIA/CUDALibrarySamples
- **OpenCL SDK**: https://github.com/KhronosGroup/OpenCL-SDK
- **ROCm**: https://github.com/ROCm/ROCm
- **ROCm Examples**: https://github.com/ROCm/rocm-examples
- **vLLM**: https://github.com/vllm-project/vllm
- **Ollama**: https://github.com/ollama/ollama
- **pandapower**: https://github.com/pandapower/pandapower
- **power-grid-model**: https://github.com/PowerGridModel/power-grid-model
- **Support**: contact@datathings.com
- **Issues**: https://github.com/datathings/marketplace/issues

## License

Apache-2.0

## Contact Us

We're [Datathings](https://datathings.com) — the team behind [GreyCat](https://greycat.io) and [Kopr](https://kopr-twin.com), Luxembourg's electricity distribution digital twin.

If you're exploring GreyCat for your infrastructure, building agentic AI into production systems, or working on large-scale grid operations and want to talk to people who've done it — reach out at [contact@datathings.com](mailto:contact@datathings.com).
