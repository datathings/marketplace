# Platform & Device API

## Table of Contents
1. [Platform Discovery](#platform-discovery)
2. [Device Discovery](#device-discovery)
3. [Platform Info Queries](#platform-info-queries)
4. [Device Info Queries](#device-info-queries)
5. [SDK Utility Helpers](#sdk-utility-helpers)

---

## Platform Discovery

### `clGetPlatformIDs(num_entries, platforms, num_platforms) -> cl_int`
**Description:** Enumerate all available OpenCL platforms on the host system.
**Parameters:**
- `num_entries` — capacity of `platforms` array (0 to query count only)
- `platforms` — output array of `cl_platform_id`; pass NULL to query count
- `num_platforms` — output: number of platforms found; pass NULL if unwanted

**Returns:** `CL_SUCCESS` or `CL_INVALID_VALUE`.

**Example:**
```c
cl_uint n = 0;
clGetPlatformIDs(0, NULL, &n);                          // query count
cl_platform_id *platforms = malloc(n * sizeof(*platforms));
clGetPlatformIDs(n, platforms, NULL);                   // fill array
```

---

## Device Discovery

### `clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices) -> cl_int`
**Description:** List devices of a given type on a platform.
**Parameters:**
- `platform` — the platform to query
- `device_type` — `CL_DEVICE_TYPE_GPU`, `CL_DEVICE_TYPE_CPU`, `CL_DEVICE_TYPE_ALL`, etc.
- `num_entries` — capacity of `devices` array
- `devices` — output array; pass NULL to query count
- `num_devices` — output count; pass NULL if unwanted

**Returns:** `CL_SUCCESS`, `CL_DEVICE_NOT_FOUND`, or `CL_INVALID_PLATFORM`.

**Example:**
```c
cl_uint nd = 0;
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
cl_device_id *devs = malloc(nd * sizeof(*devs));
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, nd, devs, NULL);
```

### `clCreateSubDevices(device, properties, num_devices, out_devices, num_devices_ret) -> cl_int`
**Description:** Partition a device into sub-devices (OpenCL 1.2+).
**Parameters:**
- `device` — the root device to partition
- `properties` — null-terminated property array (`CL_DEVICE_PARTITION_EQUALLY`, etc.)
- `num_devices` / `out_devices` — capacity / output array
- `num_devices_ret` — count returned

**Returns:** `CL_SUCCESS` or partition-specific error codes.

### `clRetainDevice(device) -> cl_int` / `clReleaseDevice(device) -> cl_int`
Increment / decrement reference count. Sub-devices must be released; root devices need not be.

---

## Platform Info Queries

### `clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
**Description:** Query a string or scalar property of a platform.
**Common `param_name` values:**

| Constant | Type | Description |
|---|---|---|
| `CL_PLATFORM_NAME` | `char[]` | Human-readable name |
| `CL_PLATFORM_VENDOR` | `char[]` | Vendor name |
| `CL_PLATFORM_VERSION` | `char[]` | OpenCL version string |
| `CL_PLATFORM_PROFILE` | `char[]` | `FULL_PROFILE` or `EMBEDDED_PROFILE` |
| `CL_PLATFORM_EXTENSIONS` | `char[]` | Space-separated extension list |

**Pattern (two-call idiom):**
```c
size_t sz;
clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz);
char *name = malloc(sz);
clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, name, NULL);
printf("Platform: %s\n", name);
free(name);
```

---

## Device Info Queries

### `clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
**Description:** Query any property of a device.
**Key `param_name` values:**

| Constant | Type | Description |
|---|---|---|
| `CL_DEVICE_TYPE` | `cl_device_type` | GPU/CPU/accelerator bitmask |
| `CL_DEVICE_NAME` | `char[]` | Device name |
| `CL_DEVICE_VENDOR` | `char[]` | Vendor string |
| `CL_DEVICE_VERSION` | `char[]` | Supported OpenCL version |
| `CL_DRIVER_VERSION` | `char[]` | Driver version |
| `CL_DEVICE_MAX_COMPUTE_UNITS` | `cl_uint` | Number of compute units |
| `CL_DEVICE_MAX_WORK_GROUP_SIZE` | `size_t` | Max work-items per work-group |
| `CL_DEVICE_MAX_WORK_ITEM_SIZES` | `size_t[3]` | Max per-dimension sizes |
| `CL_DEVICE_GLOBAL_MEM_SIZE` | `cl_ulong` | Global memory in bytes |
| `CL_DEVICE_LOCAL_MEM_SIZE` | `cl_ulong` | Local (shared) memory per CU |
| `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | `cl_ulong` | Max single buffer allocation |
| `CL_DEVICE_EXTENSIONS` | `char[]` | Supported extensions |
| `CL_DEVICE_PLATFORM` | `cl_platform_id` | Parent platform |
| `CL_DEVICE_IMAGE_SUPPORT` | `cl_bool` | Whether images are supported |
| `CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT` | `cl_uint` | Preferred float vector width |

**Example:**
```c
cl_ulong mem;
clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
printf("Global memory: %llu MB\n", (unsigned long long)mem / (1024*1024));
```

---

## SDK Utility Helpers

### C API (`<CL/Utils/Context.h>`)

```c
// Create context on platform[plat_id] with device[dev_id] of given type
cl_device_id cl_util_get_device(cl_uint plat_id, cl_uint dev_id,
                                cl_device_type type, cl_int *error);
cl_context   cl_util_get_context(cl_uint plat_id, cl_uint dev_id,
                                 cl_device_type type, cl_int *error);

// Print a summary of device capabilities to stdout
cl_int cl_util_print_device_info(cl_device_id device);

// Allocate and return info strings (caller must free)
char *cl_util_get_device_info(cl_device_id device, cl_device_info info, cl_int *error);
char *cl_util_get_platform_info(cl_platform_id platform, cl_platform_info info, cl_int *error);
```

**Error codes specific to SDK utils:**
- `CL_UTIL_INDEX_OUT_OF_RANGE` (-2000) — platform or device index out of range

### C++ API (`<CL/Utils/Context.hpp>`, `<CL/Utils/Device.hpp>`)

```cpp
// Get a context (throws cl::util::Error if CL_HPP_ENABLE_EXCEPTIONS)
cl::Context cl::util::get_context(cl_uint plat_id, cl_uint dev_id,
                                   cl_device_type type, cl_int *error = nullptr);

// Device capability checks
bool cl::util::supports_extension(const cl::Device& dev, const cl::string& ext);
bool cl::util::opencl_c_version_contains(const cl::Device& dev, const cl::string& ver);
bool cl::util::supports_feature(const cl::Device& dev, const cl::string& feat); // OpenCL 3.0+
```

### C++ Platform utilities (`<CL/Utils/Platform.hpp>`)
```cpp
bool cl::util::supports_extension(const cl::Platform& plat, const cl::string& ext);
bool cl::util::platform_version_contains(const cl::Platform& plat, const cl::string& ver);
```
