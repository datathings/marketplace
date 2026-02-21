# Memory Objects API

## Table of Contents
1. [Buffers](#buffers)
2. [Images](#images)
3. [Buffer Transfer Commands](#buffer-transfer-commands)
4. [Image Transfer Commands](#image-transfer-commands)
5. [Mapped Memory](#mapped-memory)
6. [SVM (Shared Virtual Memory)](#svm-shared-virtual-memory)
7. [Memory Object Lifecycle](#memory-object-lifecycle)

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

**Example:**
```c
float data[1024] = { /* ... */ };
cl_mem buf = clCreateBuffer(ctx,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof(data), data, &err);
```

### `clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret) -> cl_mem`
Create a sub-region view of an existing buffer. `buffer_create_type` is always `CL_BUFFER_CREATE_TYPE_REGION`; `buffer_create_info` points to a `cl_buffer_region { size_t origin; size_t size; }`.

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

### `clEnqueueReadBufferRect` / `clEnqueueWriteBufferRect` / `clEnqueueCopyBufferRect`
Rectangular (2D/3D) sub-region transfers. Take `buffer_origin`, `host_origin`, `region`, `buffer_row_pitch`, `buffer_slice_pitch`, `host_row_pitch`, `host_slice_pitch` parameters.

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

### `clEnqueueSVMMemcpy` / `clEnqueueSVMMemFill` / `clEnqueueSVMMap` / `clEnqueueSVMUnmap`
Enqueue SVM operations similar to buffer operations.

### `clSetKernelArgSVMPointer(kernel, arg_index, arg_value) -> cl_int`
Pass SVM pointer as kernel argument.

---

## Memory Object Lifecycle

### `clRetainMemObject(memobj) -> cl_int` / `clReleaseMemObject(memobj) -> cl_int`
Every `clCreateBuffer` / `clCreateImage` must be balanced with `clReleaseMemObject`.

### `clGetMemObjectInfo(memobj, param_name, ...) -> cl_int`
Key params: `CL_MEM_TYPE`, `CL_MEM_FLAGS`, `CL_MEM_SIZE`, `CL_MEM_HOST_PTR`, `CL_MEM_REFERENCE_COUNT`.

### `clSetMemObjectDestructorCallback(memobj, pfn_notify, user_data) -> cl_int`
Register a callback invoked when the object's reference count reaches zero.
