# Execution & Events API

## Table of Contents
1. [Kernel Execution](#kernel-execution)
2. [Events](#events)
3. [Barriers & Markers](#barriers--markers)
4. [Profiling](#profiling)
5. [Native & Task Execution](#native--task-execution)
6. [SDK Event Helpers](#sdk-event-helpers)

---

## Kernel Execution

### `clEnqueueNDRangeKernel(queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event) -> cl_int`
**Description:** Enqueue a kernel for parallel execution. The most important OpenCL command.
**Parameters:**
- `work_dim` — 1, 2, or 3 (number of dimensions)
- `global_work_offset` — starting indices per dimension; NULL means all zeros
- `global_work_size` — total work-items per dimension (array of `work_dim` elements)
- `local_work_size` — work-items per work-group per dimension; NULL lets runtime choose
- `num_events_in_wait_list` / `event_wait_list` — dependency events
- `event` — output event handle (pass NULL if not needed)

**Example — 1D (SAXPY):**
```c
size_t global = 1024*1024;
size_t local  = 256;
cl_event done;
clEnqueueNDRangeKernel(queue, kernel, 1,
    NULL, &global, &local,
    0, NULL, &done);
```

**Example — 2D (image processing):**
```c
size_t global[2] = { width, height };
size_t local[2]  = { 16, 16 };
clEnqueueNDRangeKernel(queue, kernel, 2,
    NULL, global, local, 0, NULL, NULL);
```

**Choosing local work-group size:**
- Must evenly divide `global_work_size` in each dimension (or use `CL_KERNEL_WORK_GROUP_SIZE` query)
- Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` for optimal granularity
- NULL lets the runtime choose (portable but may not be optimal)

---

## Events

Events are the primary synchronization mechanism between commands.

### `clWaitForEvents(num_events, event_list) -> cl_int`
Block the calling CPU thread until all listed events reach `CL_COMPLETE`.
```c
cl_event events[2] = {write_event, compute_event};
clWaitForEvents(2, events);
```

### `clGetEventInfo(event, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_EVENT_COMMAND_TYPE` | `cl_command_type` | Type of command that created the event |
| `CL_EVENT_COMMAND_EXECUTION_STATUS` | `cl_int` | `CL_QUEUED`, `CL_SUBMITTED`, `CL_RUNNING`, `CL_COMPLETE` |
| `CL_EVENT_COMMAND_QUEUE` | `cl_command_queue` | Queue the command was enqueued to |
| `CL_EVENT_CONTEXT` | `cl_context` | Associated context |

### `clSetEventCallback(event, command_exec_callback_type, pfn_event_notify, user_data) -> cl_int`
Register a callback invoked when an event reaches a specific execution status.
- `command_exec_callback_type` — typically `CL_COMPLETE`
- Callback signature: `void(cl_event event, cl_int event_command_status, void* user_data)`
- **Important:** Callbacks must not call blocking OpenCL functions (no `clFinish`, no `clWaitForEvents`).

```c
void my_callback(cl_event event, cl_int status, void *data) {
    printf("Command complete!\n");
}
clSetEventCallback(my_event, CL_COMPLETE, my_callback, NULL);
```

### `clCreateUserEvent(context, errcode_ret) -> cl_event`
Create an event whose status is controlled manually by the host. Useful for injecting custom synchronization into command streams.

### `clSetUserEventStatus(event, execution_status) -> cl_int`
Set the status of a user event. `execution_status` is typically `CL_COMPLETE` (0) or a negative error code.

### `clRetainEvent(event) -> cl_int` / `clReleaseEvent(event) -> cl_int`
Every event object obtained (from enqueue or `clCreateUserEvent`) must be released when no longer needed.

---

## Barriers & Markers

### `clEnqueueBarrierWithWaitList(queue, num_events_in_wait_list, event_wait_list, event) -> cl_int`
Insert a barrier into the queue. All commands enqueued after the barrier wait for all commands enqueued before it (or for the listed wait events) to complete.

```c
// Ensure writes complete before reads
clEnqueueBarrierWithWaitList(queue, 0, NULL, NULL);
```

### `clEnqueueMarkerWithWaitList(queue, num_events, event_wait_list, event) -> cl_int`
Like a barrier but does not block subsequent commands. Returns an event that becomes complete when the preceding commands finish.

---

## Profiling

Enable profiling by creating the queue with `CL_QUEUE_PROFILING_ENABLE`. Then query event timestamps:

### `clGetEventProfilingInfo(event, param_name, ...) -> cl_int`

| Constant | Type | Description |
|---|---|---|
| `CL_PROFILING_COMMAND_QUEUED` | `cl_ulong` | Time command was enqueued (nanoseconds) |
| `CL_PROFILING_COMMAND_SUBMIT` | `cl_ulong` | Time submitted to device |
| `CL_PROFILING_COMMAND_START` | `cl_ulong` | Time execution started |
| `CL_PROFILING_COMMAND_END` | `cl_ulong` | Time execution ended |

**Example:**
```c
cl_event evt;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, &evt);
clWaitForEvents(1, &evt);

cl_ulong start, end;
clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(end),   &end,   NULL);
printf("Kernel time: %.3f ms\n", (end - start) / 1e6);
clReleaseEvent(evt);
```

**SDK helper (`<CL/Utils/Event.h>`):**
```c
cl_ulong ns = cl_util_get_event_duration(evt,
    CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, &err);
```

**C++ SDK helper (`<CL/Utils/Event.hpp>`):**
```cpp
// Returns std::chrono::nanoseconds by default
auto dur = cl::util::get_duration<CL_PROFILING_COMMAND_START,
                                  CL_PROFILING_COMMAND_END>(event);
```

---

## Native & Task Execution

### `clEnqueueNativeKernel(queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, ...) -> cl_int`
Execute a native C function on the host as a command in the queue. Rarely used; requires `CL_EXEC_NATIVE_KERNEL` device capability.

### `clEnqueueTask(queue, kernel, ...) -> cl_int` (deprecated OpenCL 2.0)
Equivalent to `clEnqueueNDRangeKernel` with `work_dim=1`, `global_work_size={1}`, `local_work_size={1}`.

---

## SDK Event Helpers

### `cl_util_get_event_duration(event, start_prof_info, end_prof_info, error) -> cl_ulong`
Returns nanoseconds between two profiling timestamps.

### SDK timing macros (`<CL/Utils/Context.h>`)
```c
GET_CURRENT_TIMER(name)        // struct timespec name; timespec_get(&name, TIME_UTC)
START_TIMER                    // shorthand: GET_CURRENT_TIMER(start_timer1)
STOP_TIMER(dt)                 // GET_CURRENT_TIMER(t2); dt = t2 - start_timer1 (nanoseconds)
```

### Error handling macros (`<CL/Utils/Error.h>`)
```c
// func returns cl_int; jump to label on error
OCLERROR_RET(func, err_var, label);

// func sets err_var as a parameter; jump to label on error
OCLERROR_PAR(func, err_var, label);

// check malloc/alloc result; jump on NULL
MEM_CHECK(func, err_var, label);
```

**Example pattern (C):**
```c
cl_int error = CL_SUCCESS, end_error = CL_SUCCESS;
cl_mem buf;
OCLERROR_PAR(buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &error),
             error, cleanup);
/* ... */
cleanup:
    OCLERROR_RET(clReleaseMemObject(buf), end_error, done);
done:
    if (error) cl_util_print_error(error);
```
