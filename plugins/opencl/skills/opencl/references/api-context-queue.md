# Context & Command Queue API

## Table of Contents
1. [Context Creation](#context-creation)
2. [Context Info & Lifecycle](#context-info--lifecycle)
3. [Command Queue Creation](#command-queue-creation)
4. [Command Queue Lifecycle & Sync](#command-queue-lifecycle--sync)
5. [C++ Wrapper](#c-wrapper)

---

## Context Creation

### `clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret) -> cl_context`
**Description:** Create a context for one or more devices. A context owns all memory objects, programs, and queues.
**Parameters:**
- `properties` — NULL-terminated `cl_context_properties` array or NULL (use platform default). Key property: `CL_CONTEXT_PLATFORM`.
- `num_devices` — number of devices in `devices`
- `devices` — array of `cl_device_id`
- `pfn_notify` — optional error callback `void(const char*, const void*, size_t, void*)`
- `user_data` — passed to `pfn_notify`
- `errcode_ret` — output error code (pass NULL to ignore)

**Returns:** A valid `cl_context` or NULL on failure.

**Example:**
```c
cl_int err;
cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
```

### `clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret) -> cl_context`
**Description:** Create a context using all devices of a given type on the platform specified in `properties`.
```c
cl_context_properties props[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
};
cl_context ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU,
                                         NULL, NULL, &err);
```

---

## Context Info & Lifecycle

### `clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
**Key `param_name` values:**

| Constant | Type | Description |
|---|---|---|
| `CL_CONTEXT_NUM_DEVICES` | `cl_uint` | Number of associated devices |
| `CL_CONTEXT_DEVICES` | `cl_device_id[]` | Array of devices |
| `CL_CONTEXT_PROPERTIES` | `cl_context_properties[]` | Properties passed at creation |
| `CL_CONTEXT_REFERENCE_COUNT` | `cl_uint` | Reference count (debug only) |

### `clRetainContext(context) -> cl_int` / `clReleaseContext(context) -> cl_int`
Reference-count the context. Release once for every `clCreateContext*` call.

---

## Command Queue Creation

### `clCreateCommandQueueWithProperties(context, device, properties, errcode_ret) -> cl_command_queue`
**Description:** Create a command queue (OpenCL 2.0+). Preferred over the deprecated `clCreateCommandQueue`.
**Parameters:**
- `context` — valid context containing `device`
- `device` — target device
- `properties` — NULL-terminated `cl_queue_properties` array or NULL

**Key properties:**

| Property | Value | Effect |
|---|---|---|
| `CL_QUEUE_PROPERTIES` | `CL_QUEUE_PROFILING_ENABLE` | Enable event timing |
| `CL_QUEUE_PROPERTIES` | `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` | OOO execution |
| `CL_QUEUE_SIZE` | `cl_uint` | On-device queue size (device queues) |

**Example:**
```c
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
    0
};
cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
```

### `clCreateCommandQueue(context, device, properties, errcode_ret) -> cl_command_queue`
Deprecated in OpenCL 2.0, still widely used for OpenCL 1.x compatibility:
```c
cl_command_queue queue = clCreateCommandQueue(ctx, dev,
    CL_QUEUE_PROFILING_ENABLE, &err);
```

---

## Command Queue Lifecycle & Sync

### `clFlush(command_queue) -> cl_int`
Issue all queued commands to the device immediately. Does not wait for completion. Use before waiting on events from another thread.

### `clFinish(command_queue) -> cl_int`
Block until all commands in the queue have completed. Equivalent to `clFlush` + wait on all events.

```c
// Pattern: enqueue work, then wait
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
clFinish(queue);  // CPU blocks here until GPU done
```

### `clRetainCommandQueue(queue) -> cl_int` / `clReleaseCommandQueue(queue) -> cl_int`
Reference count. Every `clCreateCommandQueue*` must be paired with one `clReleaseCommandQueue`.

### `clGetCommandQueueInfo(queue, param_name, ...) -> cl_int`
Query queue properties, context, device, or reference count.

---

## C++ Wrapper

```cpp
#include <CL/opencl.hpp>

// Create queue with default properties
cl::CommandQueue queue{context, device};

// With profiling enabled
cl::CommandQueue queue{context, device,
    cl::QueueProperties::Profiling};

// With multiple properties
cl::CommandQueue queue{context, device,
    cl::QueueProperties::Profiling |
    cl::QueueProperties::OutOfOrder};

// Flush and finish
queue.flush();
queue.finish();

// Query info
cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
cl::Device  dev = queue.getInfo<CL_QUEUE_DEVICE>();
```

**Context creation in C++:**
```cpp
// Single device
cl::Context ctx{device};

// Multiple devices
std::vector<cl::Device> devs = {dev1, dev2};
cl::Context ctx{devs};

// From type (all GPUs on platform)
cl::Context ctx = cl::Context(CL_DEVICE_TYPE_GPU);

// Get devices from context
auto devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
```
