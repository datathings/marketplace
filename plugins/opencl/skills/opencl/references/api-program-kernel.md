# Programs & Kernels API

## Table of Contents
1. [Program Creation](#program-creation)
2. [Building Programs](#building-programs)
3. [Program Binaries & Cache](#program-binaries--cache)
4. [Kernel Creation & Arguments](#kernel-creation--arguments)
5. [Kernel Info Queries](#kernel-info-queries)
6. [SDK Build Helpers](#sdk-build-helpers)
7. [OpenCL C Kernel Language Notes](#opencl-c-kernel-language-notes)

---

## Program Creation

### `clCreateProgramWithSource(context, count, strings, lengths, errcode_ret) -> cl_program`
**Description:** Create a program from OpenCL C source strings.
**Parameters:**
- `count` — number of source strings in `strings`
- `strings` — array of null-terminated (or length-delimited) source strings
- `lengths` — array of lengths; NULL or 0 entry means null-terminated

**Example:**
```c
const char *src = "__kernel void add(__global float *a) { a[get_global_id(0)] += 1.f; }";
cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
```

**From file (using SDK utility):**
```c
size_t len;
char *src = cl_util_read_exe_relative_text_file("kernel.cl", &len, &err);
cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &len, &err);
free(src);
```

### `clCreateProgramWithIL(context, il, length, errcode_ret) -> cl_program`
Create from SPIR-V intermediate language (OpenCL 2.1+). `il` is a raw byte pointer to SPIR-V binary.

### `clCreateProgramWithBuiltInKernels(context, num_devices, device_list, kernel_names, errcode_ret) -> cl_program`
Create a program referencing device built-in kernels by name.

---

## Building Programs

### `clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data) -> cl_int`
**Description:** Compile and link a program for the specified devices.
**Parameters:**
- `num_devices` / `device_list` — target devices; pass 0/NULL to build for all associated devices
- `options` — compiler flags string (NULL for defaults)
- `pfn_notify` — callback `void(cl_program, void*)` called on completion; NULL for synchronous

**Common compiler options:**

| Option | Effect |
|---|---|
| `-cl-fast-relaxed-math` | Aggressive math optimizations |
| `-cl-mad-enable` | Allow fused multiply-add |
| `-cl-opt-disable` | Disable optimizations (debug) |
| `-cl-std=CL2.0` | Require OpenCL C 2.0 |
| `-D DEFINE=value` | Preprocessor define |
| `-I /path/to/headers` | Include path |

**Returns:** `CL_SUCCESS` or `CL_BUILD_PROGRAM_FAILURE` (check build log).

**Build log retrieval:**
```c
if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = malloc(log_size);
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "Build log:\n%s\n", log);
    free(log);
}
```

### `clCompileProgram(program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data) -> cl_int`
Compile a program's source (OpenCL 1.2+). Does not link.
- `num_input_headers` / `input_headers` — programs to use as embedded headers
- `header_include_names` — names by which headers are included in source

### `clLinkProgram(context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret) -> cl_program`
Link compiled programs and/or libraries into an executable (OpenCL 1.2+). Returns a new `cl_program`.

**Link options:**
| Option | Effect |
|---|---|
| `-cl-denorms-are-zero` | Denormalized numbers may be flushed to zero |
| `-cl-no-signed-zeros` | Allow optimizations ignoring sign of zero |
| `-create-library` | Create a library (not an executable) |
| `-enable-link-options` | Allow link options for next link step |

### `clUnloadPlatformCompiler(platform) -> cl_int`
Free compiler resources for a platform (OpenCL 1.2+). The compiler is reloaded on next build/compile.

### `clSetProgramSpecializationConstant(program, spec_id, spec_size, spec_value) -> cl_int`
Set a SPIR-V specialization constant value before building (OpenCL 2.2+).

### `clGetProgramBuildInfo(program, device, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_PROGRAM_BUILD_STATUS` | `cl_build_status` | `CL_BUILD_SUCCESS`, `CL_BUILD_ERROR`, etc. |
| `CL_PROGRAM_BUILD_LOG` | `char[]` | Compiler output / error messages |
| `CL_PROGRAM_BUILD_OPTIONS` | `char[]` | Options used in last build |
| `CL_PROGRAM_BINARY_TYPE` | `cl_program_binary_type` | `CL_PROGRAM_BINARY_TYPE_NONE`, `COMPILED_OBJECT`, `LIBRARY`, `EXECUTABLE` (1.2+) |
| `CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE` | `size_t` | Total global variable size (2.0+) |

---

## Program Binaries & Cache

### `clGetProgramInfo(program, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_PROGRAM_NUM_DEVICES` | `cl_uint` | Number of associated devices |
| `CL_PROGRAM_DEVICES` | `cl_device_id[]` | Associated devices |
| `CL_PROGRAM_BINARY_SIZES` | `size_t[]` | Per-device binary sizes |
| `CL_PROGRAM_BINARIES` | `unsigned char*[]` | Per-device compiled binaries |
| `CL_PROGRAM_NUM_KERNELS` | `size_t` | Number of kernels in program |
| `CL_PROGRAM_KERNEL_NAMES` | `char[]` | Semicolon-separated kernel names |
| `CL_PROGRAM_SOURCE` | `char[]` | Original source string |
| `CL_PROGRAM_IL` | `unsigned char[]` | IL (SPIR-V) content (2.1+) |

### `clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret) -> cl_program`
Restore a previously compiled program from cached binaries. Avoids recompilation overhead.

**SDK file helpers (`<CL/Utils/File.h>`):**
```c
// Save binaries: writes "<name>_<device>.bin" for each device
cl_int cl_util_write_binaries(cl_program prog, const char *name);

// Restore binaries
cl_program cl_util_read_binaries(cl_context ctx, const cl_device_id *devs,
                                  cl_uint num, const char *name, cl_int *err);
```

### `clRetainProgram(program) -> cl_int` / `clReleaseProgram(program) -> cl_int`
Reference count. Pair every create with a release.

---

## Kernel Creation & Arguments

### `clCreateKernel(program, kernel_name, errcode_ret) -> cl_kernel`
**Description:** Extract a single kernel function from a built program.
```c
cl_kernel k = clCreateKernel(prog, "saxpy", &err);
```

### `clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret) -> cl_int`
Extract all kernels at once.

### `clCloneKernel(source_kernel, errcode_ret) -> cl_kernel`
Clone a kernel object, copying all argument values and state (OpenCL 2.1+). Useful for safe concurrent argument setting from multiple threads.

### `clSetKernelArg(kernel, arg_index, arg_size, arg_value) -> cl_int`
**Description:** Set a kernel argument by index. Must be called for every argument before enqueuing.

| Argument type | `arg_size` | `arg_value` |
|---|---|---|
| Scalar (`cl_float`, `cl_int`, etc.) | `sizeof(type)` | pointer to value |
| Buffer / image (`cl_mem`) | `sizeof(cl_mem)` | pointer to `cl_mem` handle |
| Local memory | desired bytes | NULL |
| Sampler | `sizeof(cl_sampler)` | pointer to sampler |

**Example:**
```c
cl_float a = 2.0f;
clSetKernelArg(kernel, 0, sizeof(cl_float), &a);
clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_x);
clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_y);
clSetKernelArg(kernel, 3, local_size * sizeof(float), NULL); // local mem
```

### `clSetKernelArgSVMPointer(kernel, arg_index, arg_value) -> cl_int`
Set an SVM pointer argument (OpenCL 2.0+).

### `clSetKernelExecInfo(kernel, param_name, param_value_size, param_value) -> cl_int`
Provide hints: `CL_KERNEL_EXEC_INFO_SVM_PTRS`, `CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM`.

---

## Kernel Info Queries

### `clGetKernelInfo(kernel, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_KERNEL_FUNCTION_NAME` | `char[]` | Kernel function name |
| `CL_KERNEL_NUM_ARGS` | `cl_uint` | Number of arguments |
| `CL_KERNEL_REFERENCE_COUNT` | `cl_uint` | Reference count |
| `CL_KERNEL_CONTEXT` | `cl_context` | Associated context |
| `CL_KERNEL_PROGRAM` | `cl_program` | Associated program |
| `CL_KERNEL_ATTRIBUTES` | `char[]` | Kernel attributes string (1.2+) |

### `clGetKernelWorkGroupInfo(kernel, device, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_KERNEL_WORK_GROUP_SIZE` | `size_t` | Recommended max work-group size |
| `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` | `size_t` | Optimal multiple for work-group size |
| `CL_KERNEL_LOCAL_MEM_SIZE` | `cl_ulong` | Local memory used by kernel |
| `CL_KERNEL_PRIVATE_MEM_SIZE` | `cl_ulong` | Private memory used per work-item |
| `CL_KERNEL_COMPILE_WORK_GROUP_SIZE` | `size_t[3]` | From `__attribute__((reqd_work_group_size(...)))` |
| `CL_KERNEL_GLOBAL_WORK_SIZE` | `size_t[3]` | Required global work-size for built-in kernels (1.2+) |

### `clGetKernelSubGroupInfo(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret) -> cl_int`
Query sub-group information for a kernel (OpenCL 2.1+).

| Constant | Input | Output Type | Description |
|---|---|---|---|
| `CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE` | `size_t[]` (local work size) | `size_t` | Max sub-group size for given local size |
| `CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE` | `size_t[]` (local work size) | `size_t` | Number of sub-groups for given local size |
| `CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT` | `size_t` (sub-group count) | `size_t[]` | Suggested local size for requested count |
| `CL_KERNEL_MAX_NUM_SUB_GROUPS` | None | `size_t` | Max sub-groups across all local sizes |
| `CL_KERNEL_COMPILE_NUM_SUB_GROUPS` | None | `size_t` | Compile-time sub-group count (0 if unspecified) |

### `clGetKernelArgInfo(kernel, arg_index, param_name, ...) -> cl_int`
Query argument names, types, access qualifiers (OpenCL 1.2+, requires `-cl-kernel-arg-info` build option).

### `clRetainKernel(kernel) -> cl_int` / `clReleaseKernel(kernel) -> cl_int`
Pair every `clCreateKernel` with `clReleaseKernel`.

---

## SDK Build Helpers

### `cl_util_build_program(program, device, options) -> cl_int` (`<CL/Utils/Context.h>`)
Build program and automatically print the build log to stderr on failure.
```c
OCLERROR_RET(cl_util_build_program(prog, device, "-cl-fast-relaxed-math"), err, cleanup);
```

---

## OpenCL C Kernel Language Notes

**Address spaces:**
- `__global` / `global` — device global memory (buffers, images)
- `__local` / `local` — work-group shared memory (fast, limited, user-managed)
- `__constant` / `constant` — read-only constant cache
- `__private` / `private` — per-work-item registers (default)

**Built-in work-item functions:**
```c
get_global_id(dim)      // global work-item index in dimension dim
get_local_id(dim)       // index within work-group
get_group_id(dim)       // work-group index
get_global_size(dim)    // total global work-items in dimension dim
get_local_size(dim)     // work-group size in dimension dim
get_num_groups(dim)     // number of work-groups
```

**Synchronization:**
```c
barrier(CLK_LOCAL_MEM_FENCE);   // sync within work-group (local mem)
barrier(CLK_GLOBAL_MEM_FENCE);  // sync within work-group (global mem)
```

**Vector types:** `float2`, `float4`, `int4`, `uchar16`, etc. Operations are element-wise.
