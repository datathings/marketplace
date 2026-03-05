---
name: opencl
description: "OpenCL SDK (Khronos Group) for cross-platform GPU/CPU parallel computing in C and C++. Use when writing OpenCL kernels, managing devices/contexts/queues, allocating and transferring buffers or images, building and executing programs, or using the C++ wrapper (opencl.hpp / cl::CommandQueue, cl::Buffer, cl::KernelFunctor). Covers OpenCL C API, C++ bindings, and SDK utility libraries (OpenCLUtils, OpenCLSDK)."
---

# OpenCL SDK

**Version:** v2025.07.23 (Khronos Group OpenCL-SDK)
**Language:** C (OpenCL 1.0–3.0) / C++ (opencl.hpp wrapper)
**License:** Apache-2.0
**Repo:** https://github.com/KhronosGroup/OpenCL-SDK

## Overview

OpenCL (Open Computing Language) is a framework for parallel programming across heterogeneous platforms — GPUs, CPUs, FPGAs, and DSPs — from a single API. The SDK bundles:

- **OpenCL-Headers** — C headers (`<CL/cl.h>`, `<CL/cl_ext.h>`)
- **OpenCL-CLHPP** — C++ wrapper (`<CL/opencl.hpp>`)
- **OpenCL-ICD-Loader** — runtime dispatch to installed platform drivers
- **OpenCLUtils / OpenCLSDK** — utility libraries (`<CL/Utils/>`, `<CL/SDK/>`)

## Quick Start (C)

Kernel file `saxpy.cl`:
```c
__kernel void saxpy(float a, __global float *x, __global float *y) {
    int i = get_global_id(0);
    y[i] = fma(a, x[i], y[i]);
}
```

Host:
```c
#include <CL/cl.h>
cl_platform_id plat; cl_device_id dev;
clGetPlatformIDs(1, &plat, NULL);
clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL);
cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
// ... load source, clCreateProgramWithSource, clBuildProgram,
//     clCreateKernel, clSetKernelArg, clEnqueueNDRangeKernel,
//     clEnqueueReadBuffer, clReleaseXxx ...
```

## Quick Start (C++)

```cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

cl::Context ctx{CL_DEVICE_TYPE_DEFAULT};
cl::Device  dev = ctx.getInfo<CL_CONTEXT_DEVICES>()[0];
cl::CommandQueue queue{ctx, dev};
cl::Program prog{ctx, source_string};
prog.build(dev);
auto saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(prog, "saxpy");
saxpy(cl::EnqueueArgs{queue, cl::NDRange{N}}, a, buf_x, buf_y);
```

## Core Concepts

- **Work-item** — one parallel execution unit; maps to one GPU thread
- **Work-group** — block of work-items sharing local memory and barriers
- **NDRange** — N-dimensional index space (up to 3D); defines total parallelism
- **Context** — owns devices, memory objects, programs, and queues
- **Command Queue** — ordered or OOO stream of commands to one device
- **Memory object** — buffer (linear) or image (typed, sampled); device-side
- **Kernel** — a `__kernel` function compiled from OpenCL C source or SPIR-V
- **Event** — synchronization token returned by every enqueue command
- **Address spaces** — `__global` (buffers), `__local` (shared), `__constant` (read-only), `__private` (per-item)

## API Reference

| Domain | Reference File | Key Functions / Types |
|---|---|---|
| Platform & Device | `references/api-platform-device.md` | `clGetPlatformIDs`, `clGetDeviceIDs`, `clGetDeviceInfo`, `cl_util_get_device` |
| Context & Queue | `references/api-context-queue.md` | `clCreateContext`, `clCreateCommandQueueWithProperties`, `clFlush`, `clFinish` |
| Memory Objects | `references/api-memory.md` | `clCreateBuffer`, `clCreateImage`, `clEnqueueRead/WriteBuffer`, `clEnqueueMapBuffer`, SVM |
| Programs & Kernels | `references/api-program-kernel.md` | `clCreateProgramWithSource`, `clBuildProgram`, `clCreateKernel`, `clSetKernelArg` |
| Execution & Events | `references/api-execution.md` | `clEnqueueNDRangeKernel`, `clWaitForEvents`, `clSetEventCallback`, profiling |
| C++ Wrapper | `references/api-cpp-wrapper.md` | `cl::Context`, `cl::Buffer`, `cl::KernelFunctor`, `cl::EnqueueArgs`, exceptions |
| Workflows | `references/workflows.md` | Quick-start, vector add, image blur, async events, binary caching, error handling |

## Common Workflows

See `references/workflows.md` for complete, runnable examples:

- **Vector add (C)** — minimal host+kernel from scratch
- **SAXPY (C++)** — `KernelFunctor` pattern with RAII
- **Device enumeration** — iterate all platforms and devices
- **Image blur** — 2D image creation, `read_imageui` / `write_imageui`
- **Async events** — non-blocking enqueue chains
- **Binary caching** — save/restore compiled programs
- **Error handling** — C goto pattern vs. C++ exceptions

## SDK Utility Libraries

Include `<CL/Utils/Utils.h>` (C) or `<CL/Utils/Utils.hpp>` (C++) and link `OpenCLUtils` / `OpenCLUtilsCpp`.

| Header | API |
|---|---|
| `<CL/Utils/Context.h>` | `cl_util_get_device`, `cl_util_get_context`, `cl_util_print_device_info` |
| `<CL/Utils/File.h>` | `cl_util_read_text_file`, `cl_util_read_exe_relative_text_file`, `cl_util_write_binaries` |
| `<CL/Utils/Error.h>` | `OCLERROR_RET`, `OCLERROR_PAR`, `MEM_CHECK` macros, `cl_util_print_error` |
| `<CL/Utils/Event.h>` | `cl_util_get_event_duration` |
| `<CL/Utils/Device.hpp>` | `cl::util::supports_extension`, `cl::util::supports_feature` |

SDK Library (samples only, not installed): `<CL/SDK/CLI.h>`, `<CL/SDK/Random.h>`, `<CL/SDK/Image.h>`.

## Key Considerations

**Release everything:** Every `clCreate*` call must be paired with the corresponding `clRelease*`. Leak buffers or kernels and you exhaust device memory silently.

**Blocking vs. non-blocking transfers:** `clEnqueueReadBuffer(..., CL_TRUE, ...)` blocks the CPU. Use `CL_FALSE` + events for overlap. Always `clFlush` before blocking on an event from another thread.

**Local work-group size:** Must evenly divide global work size in each dimension. Query `CL_KERNEL_WORK_GROUP_SIZE` for the max; `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` for optimal alignment. Passing `NULL` lets the runtime choose (portable, not always optimal).

**Build log on failure:** `clBuildProgram` returns `CL_BUILD_PROGRAM_FAILURE` — always query `CL_PROGRAM_BUILD_LOG` to get the compiler error message. The SDK's `cl_util_build_program` does this automatically.

**Image format validation:** Not all `cl_image_format` combinations are supported on every device. Call `clGetSupportedImageFormats` before creating images.

**Event callbacks must not block:** Callbacks registered via `clSetEventCallback` are invoked from a runtime thread. Never call `clFinish` or `clWaitForEvents` inside a callback.

**C++ exceptions:** Enable with `#define CL_HPP_ENABLE_EXCEPTIONS` before including `<CL/opencl.hpp>`. Without it, check `cl_int` error parameters manually.

**OpenCL version targeting:** Set `CL_HPP_TARGET_OPENCL_VERSION` (e.g., `300`, `200`, `120`) to control which API surface is available in the C++ wrapper. OpenCL 1.x deprecated `clCreateCommandQueue`; use `clCreateCommandQueueWithProperties` for 2.0+.

**SVM requires OpenCL 2.0+:** Shared Virtual Memory (`clSVMAlloc`) requires device support for `CL_DEVICE_SVM_CAPABILITIES`. Check before use.
