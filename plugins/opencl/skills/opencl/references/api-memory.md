# Memory Objects API

## Table of Contents
1. [Buffers](#buffers)
2. [Images](#images)
3. [Buffer Transfer Commands](#buffer-transfer-commands)
4. [Image Transfer Commands](#image-transfer-commands)
5. [Mapped Memory](#mapped-memory)
6. [SVM (Shared Virtual Memory)](#svm-shared-virtual-memory)
7. [Samplers](#samplers)
8. [Memory Object Lifecycle](#memory-object-lifecycle)

---

## Buffers

### `clCreateBuffer(context, flags, size, host_ptr, errcode_ret) -> cl_mem`
**Description:** Allocate a buffer object (linear array of bytes) accessible by the device.
**Parameters:**
- `context` — owning context
- `flags` — bitmask controlling access and initialization (see below)
- `size` — size in bytes
- `host_ptr` — pointer for `CL_MEM_USE_HOST_PTR` or `CL_MEM_COPY_HOST_PTR`; NULL otherwise
- `errcode_ret` — output error code

**Key `flags` values:**

| Flag | Description |
|---|---|
| `CL_MEM_READ_WRITE` | Read and write from kernel (default) |
| `CL_MEM_READ_ONLY` | Kernel reads only |
| `CL_MEM_WRITE_ONLY` | Kernel writes only |
| `CL_MEM_COPY_HOST_PTR` | Copy `host_ptr` data to device at creation |
| `CL_MEM_USE_HOST_PTR` | Device uses host memory directly (pinned) |
| `CL_MEM_ALLOC_HOST_PTR` | Allocate pinned host-accessible memory |
| `CL_MEM_HOST_WRITE_ONLY` | Host writes only (1.2+) |
| `CL_MEM_HOST_READ_ONLY` | Host reads only (1.2+) |
| `CL_MEM_HOST_NO_ACCESS` | Host cannot access (1.2+) |
| `CL_MEM_KERNEL_READ_AND_WRITE` | Kernel can read and write (2.0+) |

**Example:**
```c
float data[1024] = { /* ... */ };
cl_mem buf = clCreateBuffer(ctx,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof(data), data, &err);
```

### `clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret) -> cl_mem`
Create a sub-region view of an existing buffer (OpenCL 1.1+). `buffer_create_type` is always `CL_BUFFER_CREATE_TYPE_REGION`; `buffer_create_info` points to a `cl_buffer_region { size_t origin; size_t size; }`.

### `clCreateBufferWithProperties(context, properties, flags, size, host_ptr, errcode_ret) -> cl_mem`
Create a buffer with explicit memory properties (OpenCL 3.0+).
- `properties` — NULL-terminated `cl_mem_properties` array or NULL

### `clCreateImageWithProperties(context, properties, flags, image_format, image_desc, host_ptr, errcode_ret) -> cl_mem`
Create an image with explicit memory properties (OpenCL 3.0+).
- `properties` — NULL-terminated `cl_mem_properties` array or NULL

---

## Images

### `clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret) -> cl_mem`
**Description:** Create a 1D, 2D, or 3D image object. Requires `CL_DEVICE_IMAGE_SUPPORT == CL_TRUE`.
**Parameters:**
- `image_format` — `cl_image_format { cl_channel_order; cl_channel_type; }`
- `image_desc` — `cl_image_desc` specifying type, dimensions, row pitch, etc.

**Common channel orders:** `CL_R`, `CL_RGBA`, `CL_BGRA`, `CL_LUMINANCE`
**Common channel types:** `CL_UNORM_INT8`, `CL_FLOAT`, `CL_HALF_FLOAT`, `CL_UNSIGNED_INT32`

**`cl_image_desc` fields:**
```c
cl_image_desc desc = {
    .image_type   = CL_MEM_OBJECT_IMAGE2D,
    .image_width  = 1920,
    .image_height = 1080,
    .image_depth  = 1,       // for 3D
    .image_row_pitch   = 0,  // 0 = auto
    .image_slice_pitch = 0,
    .num_mip_levels = 0,
    .num_samples    = 0,
    .mem_object     = NULL,  // for 1D image from buffer
};
```

**Example:**
```c
cl_image_format fmt = { CL_RGBA, CL_UNORM_INT8 };
cl_image_desc   desc = { .image_type = CL_MEM_OBJECT_IMAGE2D,
                         .image_width = 512, .image_height = 512 };
cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
```

### `clGetSupportedImageFormats(context, flags, image_type, num_entries, image_formats, num_image_formats) -> cl_int`
Query which `cl_image_format` combinations are supported. Always call this before creating images to validate format support.

### `clGetImageInfo(image, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
Query image properties.

| Constant | Type | Description |
|---|---|---|
| `CL_IMAGE_FORMAT` | `cl_image_format` | Channel order and type |
| `CL_IMAGE_ELEMENT_SIZE` | `size_t` | Element size in bytes |
| `CL_IMAGE_ROW_PITCH` | `size_t` | Row pitch in bytes |
| `CL_IMAGE_SLICE_PITCH` | `size_t` | Slice pitch in bytes |
| `CL_IMAGE_WIDTH` | `size_t` | Image width |
| `CL_IMAGE_HEIGHT` | `size_t` | Image height |
| `CL_IMAGE_DEPTH` | `size_t` | Image depth (3D only) |
| `CL_IMAGE_ARRAY_SIZE` | `size_t` | Number of images in array (1.2+) |
| `CL_IMAGE_NUM_MIP_LEVELS` | `cl_uint` | Mip levels (1.2+) |
| `CL_IMAGE_NUM_SAMPLES` | `cl_uint` | Samples (1.2+) |

### Pipes (OpenCL 2.0+)

### `clCreatePipe(context, flags, pipe_packet_size, pipe_max_packets, properties, errcode_ret) -> cl_mem`
Create a pipe object.
- `flags` — `CL_MEM_READ_WRITE` (required) | `CL_MEM_HOST_NO_ACCESS`
- `pipe_packet_size` — size of each packet in bytes
- `pipe_max_packets` — max number of packets the pipe can hold
- `properties` — NULL-terminated `cl_pipe_properties` array or NULL

### `clGetPipeInfo(pipe, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
Query pipe properties: `CL_PIPE_PACKET_SIZE` (`cl_uint`), `CL_PIPE_MAX_PACKETS` (`cl_uint`), `CL_PIPE_PROPERTIES` (3.0+).

---

## Buffer Transfer Commands

All enqueue functions take `(queue, ..., num_events_in_wait_list, event_wait_list, event)` at the end.

### `clEnqueueReadBuffer(queue, buffer, blocking_read, offset, size, ptr, ...) -> cl_int`
Copy from device buffer to host memory.
- `blocking_read` — `CL_TRUE` blocks until complete; `CL_FALSE` returns immediately (use events to synchronize)

```c
clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
```

### `clEnqueueWriteBuffer(queue, buffer, blocking_write, offset, size, ptr, ...) -> cl_int`
Copy from host memory to device buffer.

```c
clEnqueueWriteBuffer(queue, buf, CL_FALSE, 0, sizeof(data), data, 0, NULL, &write_event);
```

### `clEnqueueCopyBuffer(queue, src_buffer, dst_buffer, src_offset, dst_offset, size, ...) -> cl_int`
Device-to-device buffer copy without host involvement.

### `clEnqueueFillBuffer(queue, buffer, pattern, pattern_size, offset, size, ...) -> cl_int`
Fill a buffer region with a repeated pattern (OpenCL 1.2+).

```c
cl_float zero = 0.f;
clEnqueueFillBuffer(queue, buf, &zero, sizeof(zero), 0, total_bytes, 0, NULL, NULL);
```

### `clEnqueueReadBufferRect(queue, buffer, blocking, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, ...) -> cl_int`
Rectangular (2D/3D) sub-region read from buffer to host (OpenCL 1.1+). Origin and region are 3-element `size_t` arrays.

### `clEnqueueWriteBufferRect(queue, buffer, blocking, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, ...) -> cl_int`
Rectangular sub-region write from host to buffer (OpenCL 1.1+).

### `clEnqueueCopyBufferRect(queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, ...) -> cl_int`
Rectangular device-to-device buffer copy (OpenCL 1.1+).

### `clEnqueueMigrateMemObjects(queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Migrate memory objects to the device associated with the queue (OpenCL 1.2+).
- `flags` — `CL_MIGRATE_MEM_OBJECT_HOST` (migrate to host), `CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED` (skip data transfer)

---

## Image Transfer Commands

### `clEnqueueReadImage(queue, image, blocking, origin[3], region[3], row_pitch, slice_pitch, ptr, ...) -> cl_int`
### `clEnqueueWriteImage(queue, image, blocking, origin[3], region[3], row_pitch, slice_pitch, ptr, ...) -> cl_int`
Transfer image data to/from host. `origin` and `region` are 3-element arrays of `size_t`; for 2D use `{0,0,0}` and `{w,h,1}`.

### `clEnqueueCopyImage(queue, src, dst, src_origin[3], dst_origin[3], region[3], ...) -> cl_int`
Device-to-device image copy.

### `clEnqueueCopyImageToBuffer(queue, src_image, dst_buf, src_origin[3], region[3], dst_offset, ...) -> cl_int`
### `clEnqueueCopyBufferToImage(queue, src_buf, dst_image, src_offset, dst_origin[3], region[3], ...) -> cl_int`
Cross-type copies between images and buffers.

### `clEnqueueFillImage(queue, image, fill_color, origin[3], region[3], ...) -> cl_int`
Fill an image region with a constant color (`cl_float4`, `cl_int4`, or `cl_uint4` depending on channel type).

---

## Mapped Memory

### `clEnqueueMapBuffer(queue, buffer, blocking_map, map_flags, offset, size, ..., errcode_ret) -> void*`
Map buffer region into host address space. Faster than explicit read/write for frequent host access.
- `map_flags` — `CL_MAP_READ`, `CL_MAP_WRITE`, `CL_MAP_WRITE_INVALIDATE_REGION`

```c
float *ptr = (float*)clEnqueueMapBuffer(queue, buf, CL_TRUE,
    CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, NULL, NULL, &err);
// ... modify ptr ...
clEnqueueUnmapMemObject(queue, buf, ptr, 0, NULL, NULL);
```

### `clEnqueueMapImage(queue, image, blocking, map_flags, origin[3], region[3], row_pitch*, slice_pitch*, ..., errcode_ret) -> void*`
Map an image region. Returns row and slice pitch for correct indexing.

### `clEnqueueUnmapMemObject(queue, memobj, mapped_ptr, ...) -> cl_int`
Release a mapped region. Must be called before the memory is used again by device commands.

---

## SVM (Shared Virtual Memory)

OpenCL 2.0+. Allows sharing pointers between host and device.

### `clSVMAlloc(context, flags, size, alignment) -> void*`
Allocate SVM memory. `flags`: `CL_MEM_READ_WRITE`, `CL_MEM_SVM_FINE_GRAIN_BUFFER`, `CL_MEM_SVM_ATOMICS`.

### `clSVMFree(context, svm_pointer)`
Free SVM memory. Not enqueued; synchronize before calling.

### `clEnqueueSVMFree(queue, num_svm_pointers, svm_pointers[], pfn_free_func, user_data, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Enqueue freeing of SVM allocations. `pfn_free_func` is an optional callback `void(cl_command_queue, cl_uint, void*[], void*)` invoked to perform the actual free; pass NULL to use default `clSVMFree`.

### `clEnqueueSVMMemcpy(queue, blocking_copy, dst_ptr, src_ptr, size, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Enqueue a memcpy between SVM pointers (or SVM and host).

### `clEnqueueSVMMemFill(queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Fill an SVM region with a repeated pattern.

### `clEnqueueSVMMap(queue, blocking_map, flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Map an SVM allocation for host access. `flags` is `cl_map_flags` (`CL_MAP_READ`, `CL_MAP_WRITE`, etc.).

### `clEnqueueSVMUnmap(queue, svm_ptr, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Unmap a previously mapped SVM pointer.

### `clEnqueueSVMMigrateMem(queue, num_svm_pointers, svm_pointers, sizes, flags, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Migrate SVM allocations to a device or host (OpenCL 2.1+).
- `svm_pointers` — `const void**` array of SVM pointers
- `sizes` — `const size_t*` array of sizes (0 means entire allocation)
- `flags` — `cl_mem_migration_flags`

### `clSetKernelArgSVMPointer(kernel, arg_index, arg_value) -> cl_int`
Pass SVM pointer as kernel argument.

---

## Samplers

### `clCreateSamplerWithProperties(context, sampler_properties, errcode_ret) -> cl_sampler`
Create a sampler with explicit properties (OpenCL 2.0+). `sampler_properties` is a NULL-terminated `cl_sampler_properties` array.

| Property | Type | Description |
|---|---|---|
| `CL_SAMPLER_NORMALIZED_COORDS` | `cl_bool` | Use [0,1] coords if true, pixel coords if false |
| `CL_SAMPLER_ADDRESSING_MODE` | `cl_addressing_mode` | `CL_ADDRESS_CLAMP`, `CL_ADDRESS_REPEAT`, etc. |
| `CL_SAMPLER_FILTER_MODE` | `cl_filter_mode` | `CL_FILTER_NEAREST` or `CL_FILTER_LINEAR` |

```c
cl_sampler_properties props[] = {
    CL_SAMPLER_NORMALIZED_COORDS, CL_TRUE,
    CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP_TO_EDGE,
    CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
    0
};
cl_sampler sampler = clCreateSamplerWithProperties(ctx, props, &err);
```

### `clGetSamplerInfo(sampler, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
Query sampler properties: `CL_SAMPLER_REFERENCE_COUNT`, `CL_SAMPLER_CONTEXT`, `CL_SAMPLER_NORMALIZED_COORDS`, `CL_SAMPLER_ADDRESSING_MODE`, `CL_SAMPLER_FILTER_MODE`, `CL_SAMPLER_PROPERTIES` (3.0+).

### `clRetainSampler(sampler) -> cl_int` / `clReleaseSampler(sampler) -> cl_int`
Reference count. Pair every `clCreateSampler*` with `clReleaseSampler`.

---

## Memory Object Lifecycle

### `clRetainMemObject(memobj) -> cl_int` / `clReleaseMemObject(memobj) -> cl_int`
Every `clCreateBuffer` / `clCreateImage` must be balanced with `clReleaseMemObject`.

### `clGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret) -> cl_int`
Key params: `CL_MEM_TYPE`, `CL_MEM_FLAGS`, `CL_MEM_SIZE`, `CL_MEM_HOST_PTR`, `CL_MEM_MAP_COUNT`, `CL_MEM_REFERENCE_COUNT`, `CL_MEM_CONTEXT`, `CL_MEM_ASSOCIATED_MEMOBJECT` (1.1+), `CL_MEM_OFFSET` (1.1+), `CL_MEM_USES_SVM_POINTER` (2.0+), `CL_MEM_PROPERTIES` (3.0+).

### `clSetMemObjectDestructorCallback(memobj, pfn_notify, user_data) -> cl_int`
Register a callback invoked when the object's reference count reaches zero.
