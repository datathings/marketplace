# Update OpenCL Skill to Latest Version

This command updates the OpenCL SDK library to the latest version and synchronizes
all skill files with the updated API.

**Working directory:** `plugins/opencl/`

## Repository Structure

```
plugins/opencl/
├── skills/opencl/          # Skill content (SKILL.md and references/)
├── OpenCL-SDK/             # Upstream repo clone (NOT a submodule)
└── .claude-plugin/plugin.json   # Plugin manifest
```

The `OpenCL-SDK/` directory is a **standalone git clone** from https://github.com/KhronosGroup/OpenCL-SDK - updated
independently via `git fetch` and `git checkout <tag>`.

## Process

### 1. Update Library to Latest Tag

```bash
cd plugins/opencl/OpenCL-SDK

git fetch --tags

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
echo "Current tag: $CURRENT_TAG"

# Tags use format: v####.##.## (e.g., v2025.07.23)
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

### 2–6. Analyze API Changes & Update Documentation (subagent)

**Dispatch a `general-purpose` subagent** via the `Task` tool for the analysis and
documentation rewrite. This keeps the large header content out of the main context.

Provide the subagent with:
- Path to C headers (source of truth):
  - `plugins/opencl/OpenCL-SDK/external/OpenCL-Headers/CL/cl.h` — core C API
  - `plugins/opencl/OpenCL-SDK/external/OpenCL-Headers/CL/cl_ext.h` — extensions
  - `plugins/opencl/OpenCL-SDK/external/OpenCL-Headers/CL/cl_platform.h` — platform types
  - `plugins/opencl/OpenCL-SDK/external/OpenCL-CLHPP/include/CL/opencl.hpp` — C++ wrapper
  - `plugins/opencl/OpenCL-SDK/lib/include/` — SDK utility headers (if present)
- Path to skill references: `plugins/opencl/skills/opencl/references/`
- Old tag and new tag (from Step 1 output)

The subagent should:

**Analyze API Changes** — scan the source-of-truth headers for:
- New functions/methods added since last update
- Deprecated or removed functions
- Modified signatures (parameter changes, return types)
- New types, enums, structs, or constants

**Update Skill Documentation Files** in `skills/opencl/references/`:
- **`api-platform-device.md`** — clGetPlatformIDs, clGetDeviceIDs, clGetPlatformInfo, clGetDeviceInfo, SDK helpers
- **`api-context-queue.md`** — clCreateContext, clCreateCommandQueueWithProperties, sync functions
- **`api-memory.md`** — clCreateBuffer, clCreateImage, enqueue read/write/copy/fill, SVM, pipes
- **`api-program-kernel.md`** — clCreateProgramWithSource, clBuildProgram, clCreateKernel, clSetKernelArg
- **`api-execution.md`** — clEnqueueNDRangeKernel, events, barriers, profiling, clFinish/clFlush
- **`api-cpp-wrapper.md`** — cl::Platform, cl::Device, cl::Context, cl::CommandQueue, cl::Buffer, cl::Kernel, cl::Program (from opencl.hpp)
- **`workflows.md`** — Working examples and usage patterns

**Update Main SKILL.md:**
- Update the version/tag in the documentation
- Update function/method counts in the overview
- Add notes about newly supported features
- Update the API reference table if files were added/removed
- Keep SKILL.md concise and dense (under 500 lines) — API details belong in `references/`

**Remove Deprecated Content:**
- Search documentation files for functions removed from the source-of-truth headers
- Remove or replace deprecated references
- Note breaking changes

**Validate Changes:**
- Cross-reference all documented functions against source-of-truth headers
- Ensure signatures match exactly
- Verify all new public API functions are documented

**The subagent should return a summary** of: functions added, functions removed, signatures
changed, and any breaking changes — so the main context can report the outcome.

### 7. Package Updated Skill

```bash
./package.sh opencl
```

This creates `skills/opencl.skill`.

## Key Considerations

- **Source of truth**: Always derive C signatures from `cl.h` and `cl_ext.h`, C++ from `opencl.hpp` — not secondary docs
- **Deprecation handling**: Remove deprecated functions completely; don't just mark them
- **New features**: Major new features should get workflow examples in `workflows.md`
- **Backward compatibility**: Note any breaking changes in SKILL.md
- **Tag format**: OpenCL-SDK uses `v####.##.##` date-based tags (e.g., `v2025.07.23`)

## Success Criteria

- ✅ Library updated to latest stable tag
- ✅ SKILL.md documents which version it corresponds to
- ✅ All new API functions from source-of-truth files are documented
- ✅ All removed functions are deleted from documentation
- ✅ All signatures match source-of-truth exactly
- ✅ Code examples in workflows.md are valid
- ✅ Package builds successfully with `./package.sh opencl`
