# C++ Wrapper API (opencl.hpp)

The C++ wrapper (`#include <CL/opencl.hpp>`) provides RAII types, range-based iteration, and exception support over the C API.

## Table of Contents
1. [Setup & Compilation](#setup--compilation)
2. [Platform & Device](#platform--device)
3. [Context](#context)
4. [CommandQueue](#commandqueue)
5. [Buffer & Memory](#buffer--memory)
6. [Program & Kernel](#program--kernel)
7. [Kernel Execution](#kernel-execution)
8. [Events & Synchronization](#events--synchronization)
9. [Error Handling](#error-handling)
10. [SDK C++ Utilities](#sdk-c-utilities)

---

## Setup & Compilation

```cpp
// Enable exceptions (recommended)
#define CL_HPP_ENABLE_EXCEPTIONS

// Target OpenCL version (affects available APIs)
#define CL_HPP_TARGET_OPENCL_VERSION 300  // 300, 200, 120, 110

#include <CL/opencl.hpp>
```

All wrapper objects manage their underlying C handle via reference counting. Copies are shallow (shared ownership); use `cl::detail::Wrapper` copy semantics.

---

## Platform & Device

```cpp
// Get all platforms
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);

// Get default platform
cl::Platform platform = cl::Platform::getDefault();

// Query platform info (returns std::string)
std::string name = platform.getInfo<CL_PLATFORM_NAME>();
std::string vendor = platform.getInfo<CL_PLATFORM_VENDOR>();

// Get devices on platform
std::vector<cl::Device> devices;
platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

// Query device info
cl::Device device = devices[0];
std::string devName = device.getInfo<CL_DEVICE_NAME>();
cl_ulong mem  = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
size_t   wgs  = device.getInfo<CL_MAX_WORK_GROUP_SIZE>();
cl_uint  cus  = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

// Check extension support
bool hasFP64 = device.getInfo<CL_DEVICE_EXTENSIONS>()
                    .find("cl_khr_fp64") != std::string::npos;
```

---

## Context

```cpp
// Single device
cl::Context ctx{device};

// Multiple devices
cl::Context ctx{std::vector<cl::Device>{dev1, dev2}};

// All GPUs (uses default platform)
cl::Context ctx{CL_DEVICE_TYPE_GPU};

// Query associated devices
auto devs = ctx.getInfo<CL_CONTEXT_DEVICES>();
```

---

## CommandQueue

```cpp
// Default queue
cl::CommandQueue queue{ctx, device};

// With properties (C++17 structured constructor)
cl::CommandQueue queue{ctx, device,
    cl::QueueProperties::Profiling};

// Multiple properties
cl::CommandQueue queue{ctx, device,
    cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder};

// Synchronization
queue.flush();
queue.finish();
```

---

## Buffer & Memory

```cpp
// Allocate read/write buffer
cl::Buffer buf{ctx, CL_MEM_READ_WRITE, bytes};

// Allocate and initialize from host data
std::vector<float> host_data(N, 1.0f);
cl::Buffer buf{ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
               N * sizeof(float), host_data.data()};

// From iterator range (copies data, buffer is read-write unless readonly=true)
cl::Buffer buf{ctx, host_data.begin(), host_data.end(), /*readonly=*/true};

// Write to device
queue.enqueueWriteBuffer(buf, CL_TRUE, offset, bytes, host_ptr);

// Read from device
queue.enqueueReadBuffer(buf, CL_TRUE, offset, bytes, host_ptr);

// Device-to-device copy
queue.enqueueCopyBuffer(src, dst, src_off, dst_off, bytes);

// Map / unmap (zero-copy with CL_MEM_ALLOC_HOST_PTR)
float *ptr = (float*)queue.enqueueMapBuffer(buf, CL_TRUE,
    CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes);
// ... modify ptr ...
queue.enqueueUnmapMemObject(buf, ptr);

// Convenience copy (blocks until done)
cl::copy(queue, host_data.begin(), host_data.end(), buf);
cl::copy(queue, buf, host_data.begin(), host_data.end());
```

**Images:**
```cpp
cl::ImageFormat fmt{CL_RGBA, CL_UNORM_INT8};
cl::Image2D img{ctx, CL_MEM_READ_WRITE, fmt, width, height};

// Enqueue image read
std::vector<size_t> origin{0,0,0}, region{width,height,1};
queue.enqueueReadImage(img, CL_TRUE, origin, region, 0, 0, host_pixels.data());
```

---

## Program & Kernel

```cpp
// Build from source string
cl::Program program{ctx, source_string};
try {
    program.build(device);
} catch (cl::BuildError& e) {
    for (auto& [dev, log] : e.getBuildLog())
        std::cerr << log << "\n";
}

// With compiler options
program.build("-cl-fast-relaxed-math -DCL_VERSION=2");

// Build for all context devices
program.build();

// Create kernel by name
cl::Kernel kernel{program, "saxpy"};

// Set arguments manually
kernel.setArg(0, a_float);
kernel.setArg(1, buf_x);
kernel.setArg(2, buf_y);

// KernelFunctor: type-safe call site (sets args + enqueues atomically)
auto saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(program, "saxpy");
saxpy(cl::EnqueueArgs{queue, cl::NDRange{N}}, a, buf_x, buf_y);
```

**KernelFunctor** is the recommended C++ pattern. Template parameters match kernel argument types in order.

---

## Kernel Execution

```cpp
// NDRange dispatch
queue.enqueueNDRangeKernel(kernel,
    cl::NullRange,            // global offset (none)
    cl::NDRange{1024, 1024},  // global work size
    cl::NDRange{16, 16});     // local work size (optional, NDRange() = runtime choice)

// KernelFunctor (preferred: combines arg setting + enqueue)
auto add = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "add");
cl::Event done = add(cl::EnqueueArgs{queue, cl::NDRange{N}}, buf_a, buf_b);
done.wait();
```

**EnqueueArgs** constructor variants:
```cpp
cl::EnqueueArgs{queue, cl::NDRange{N}}
cl::EnqueueArgs{queue, cl::NDRange{N}, cl::NDRange{local}}
cl::EnqueueArgs{queue, events, cl::NDRange{N}}  // with wait events
```

---

## Events & Synchronization

```cpp
cl::Event event;
queue.enqueueNDRangeKernel(kernel, cl::NullRange,
    cl::NDRange{N}, cl::NullRange, nullptr, &event);

// Wait on CPU
event.wait();

// Get status
cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();

// Callback
event.setCallback(CL_COMPLETE, [](cl_event, cl_int, void*) {
    printf("done\n");
}, nullptr);

// User event
cl::UserEvent ue{ctx};
ue.setStatus(CL_COMPLETE);

// Wait list (pass to enqueue)
std::vector<cl::Event> waitList{event1, event2};
queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{N},
    cl::NullRange, &waitList, &next_event);
```

---

## Error Handling

```cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

try {
    cl::Program prog{ctx, source};
    prog.build(device);
    auto kernel = cl::KernelFunctor<cl::Buffer>(prog, "myfunc");
    kernel(cl::EnqueueArgs{queue, cl::NDRange{N}}, buf);
    queue.finish();

} catch (cl::BuildError& e) {
    // Compilation failed
    for (auto& [dev, log] : e.getBuildLog())
        std::cerr << dev.getInfo<CL_DEVICE_NAME>() << ":\n" << log << "\n";
    return e.err();

} catch (cl::Error& e) {
    // Runtime OpenCL error
    std::cerr << "CL error: " << e.what() << " (" << e.err() << ")\n";
    return e.err();

} catch (cl::util::Error& e) {
    // SDK utility error
    std::cerr << "SDK error: " << e.what() << " (" << e.err() << ")\n";
    return e.err();
}
```

Without exceptions, check return codes from methods that accept an `cl_int* err` parameter.

---

## SDK C++ Utilities

### `<CL/Utils/Context.hpp>`
```cpp
// Select device by triplet index (platform_id, device_id, device_type)
cl::Context cl::util::get_context(cl_uint plat, cl_uint dev,
                                   cl_device_type type, cl_int *err = nullptr);
void cl::util::print_device_info(const cl::Device& device);
```

### `<CL/Utils/File.hpp>`
```cpp
std::string cl::util::read_text_file(const char* filename, cl_int *err = nullptr);
std::string cl::util::read_exe_relative_text_file(const char* rel, cl_int *err = nullptr);
std::vector<unsigned char> cl::util::read_binary_file(const char* filename, cl_int *err = nullptr);
cl_int cl::util::write_binaries(const cl::Program::Binaries& bins,
                                const std::vector<cl::Device>& devs,
                                const char* name);
```

### `<CL/Utils/Event.hpp>`
```cpp
// Returns duration in nanoseconds (or other Dur via template param)
template<cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
auto cl::util::get_duration(cl::Event& ev);
```

### `<CL/SDK/Context.hpp>`
```cpp
// Higher-level context helper that wires up SDK CLI options
cl::Context cl::sdk::get_context(const cl::sdk::options::DeviceTriplet& triplet,
                                  cl_int *err = nullptr);
```
