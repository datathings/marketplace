# Workflows

## Table of Contents
1. [Quick Start — Minimal Vector Add (C)](#quick-start--minimal-vector-add-c)
2. [Quick Start — C++ Wrapper (SAXPY)](#quick-start--c-wrapper-saxpy)
3. [Device Selection & Enumeration](#device-selection--enumeration)
4. [Image Processing (Blur)](#image-processing-blur)
5. [Asynchronous Execution with Events](#asynchronous-execution-with-events)
6. [Program Binary Caching](#program-binary-caching)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Build System Integration](#build-system-integration)

---

## Quick Start — Minimal Vector Add (C)

Kernel file `add.cl`:
```c
__kernel void add(__global const float *a,
                  __global const float *b,
                  __global float *c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
```

Host code:
```c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main(void) {
    cl_int err;

    // 1. Platform & device
    cl_platform_id platform;
    cl_device_id   device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    // 2. Context & queue
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);

    // 3. Source & build
    const char *src = /* read add.cl contents */;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    clBuildProgram(prog, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(prog, "add", &err);

    // 4. Buffers
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i * 2; }
    cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_a, &err);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_b, &err);
    cl_mem d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*sizeof(float), NULL, &err);

    // 5. Kernel args & execute
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    size_t global = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    // 6. Read back
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, N*sizeof(float), h_c, 0, NULL, NULL);

    printf("c[0]=%f  c[N-1]=%f\n", h_c[0], h_c[N-1]);

    // 7. Cleanup
    clReleaseMemObject(d_c); clReleaseMemObject(d_b); clReleaseMemObject(d_a);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
```

---

## Quick Start — C++ Wrapper (SAXPY)

Kernel file `saxpy.cl`:
```c
__kernel void saxpy(float a, __global float *x, __global float *y) {
    int i = get_global_id(0);
    y[i] = fma(a, x[i], y[i]);
}
```

Host code:
```cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <CL/Utils/File.hpp>   // cl::util::read_exe_relative_text_file
#include <valarray>
#include <iostream>

int main() {
    try {
        // Platform, device, context (all defaults)
        cl::Context ctx{CL_DEVICE_TYPE_DEFAULT};
        cl::Device  dev = ctx.getInfo<CL_CONTEXT_DEVICES>()[0];
        cl::CommandQueue queue{ctx, dev};

        // Build program
        std::string src = cl::util::read_exe_relative_text_file("saxpy.cl");
        cl::Program prog{ctx, src};
        prog.build(dev);

        // KernelFunctor: type-safe dispatch
        auto saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(prog, "saxpy");

        // Data
        const size_t N = 1 << 20;
        std::valarray<float> x(1.0f, N), y(2.0f, N);
        float a = 3.0f;

        cl::Buffer d_x{ctx, x.begin(), x.end(), /*readonly=*/true};
        cl::Buffer d_y{ctx, y.begin(), y.end(), /*readonly=*/false};

        saxpy(cl::EnqueueArgs{queue, cl::NDRange{N}}, a, d_x, d_y);

        // Read result
        cl::copy(queue, d_y, std::begin(y), std::end(y));
        std::cout << "y[0]=" << y[0] << "\n"; // expected: 3*1+2 = 5

    } catch (cl::Error& e) {
        std::cerr << e.what() << " (" << e.err() << ")\n";
        return 1;
    }
}
```

---

## Device Selection & Enumeration

```c
// Enumerate all platforms and their devices
cl_uint np;
clGetPlatformIDs(0, NULL, &np);
cl_platform_id *plats = malloc(np * sizeof(*plats));
clGetPlatformIDs(np, plats, NULL);

for (cl_uint p = 0; p < np; p++) {
    size_t sz;
    clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, 0, NULL, &sz);
    char *pname = malloc(sz);
    clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sz, pname, NULL);
    printf("Platform[%u]: %s\n", p, pname);
    free(pname);

    cl_uint nd;
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
    cl_device_id *devs = malloc(nd * sizeof(*devs));
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, nd, devs, NULL);

    for (cl_uint d = 0; d < nd; d++) {
        clGetDeviceInfo(devs[d], CL_DEVICE_NAME, 0, NULL, &sz);
        char *dname = malloc(sz);
        clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sz, dname, NULL);
        cl_ulong mem;
        clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        printf("  Device[%u]: %s (%llu MB)\n", d, dname, (unsigned long long)mem>>20);
        free(dname);
    }
    free(devs);
}
free(plats);
```

---

## Image Processing (Blur)

2D image kernel (`blur.cl`):
```c
kernel void blur_box(read_only image2d_t src, write_only image2d_t dst, int radius) {
    int2 coord = { get_global_id(0), get_global_id(1) };
    int w = get_image_width(src), h = get_image_height(src);
    uint4 sum = 0; uint n = 0;
    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 c = coord + (int2)(dx, dy);
            if (c.x >= 0 && c.x < w && c.y >= 0 && c.y < h) {
                sum += read_imageui(src, c); n++;
            }
        }
    write_imageui(dst, coord, (sum + n/2) / n);
}
```

Host (C++):
```cpp
const int W = 1920, H = 1080, radius = 3;

// Validate format
cl_uint nfmt;
clGetSupportedImageFormats(ctx, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &nfmt);
// (check RGBA/UNORM_INT8 is present)

cl::ImageFormat fmt{CL_RGBA, CL_UNORM_INT8};
cl::Image2D src{ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, W, H, 0, pixels.data()};
cl::Image2D dst{ctx, CL_MEM_WRITE_ONLY, fmt, W, H};

auto blur = cl::KernelFunctor<cl::Image2D, cl::Image2D, cl_int>(prog, "blur_box");
blur(cl::EnqueueArgs{queue, cl::NDRange{W, H}}, src, dst, radius);

// Read result
std::vector<uint8_t> out(W * H * 4);
std::vector<size_t> origin{0,0,0}, region{(size_t)W,(size_t)H,1};
queue.enqueueReadImage(dst, CL_TRUE, origin, region, 0, 0, out.data());
```

---

## Asynchronous Execution with Events

```c
cl_event write_ev, kernel_ev, read_ev;

// Async write (non-blocking)
clEnqueueWriteBuffer(queue, d_buf, CL_FALSE, 0, size, h_buf, 0, NULL, &write_ev);

// Kernel waits for write to complete
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL,
                        1, &write_ev, &kernel_ev);

// Read waits for kernel
clEnqueueReadBuffer(queue, d_out, CL_FALSE, 0, size, h_out,
                    1, &kernel_ev, &read_ev);

// Flush all commands to device
clFlush(queue);

// Do CPU work concurrently here...

// Wait for final result
clWaitForEvents(1, &read_ev);

clReleaseEvent(read_ev);
clReleaseEvent(kernel_ev);
clReleaseEvent(write_ev);
```

**Event callback for async notification:**
```c
void on_done(cl_event ev, cl_int status, void *data) {
    // Called from a runtime thread — no blocking CL calls here
    printf("GPU done, status=%d\n", status);
}
clSetEventCallback(kernel_ev, CL_COMPLETE, on_done, NULL);
```

---

## Program Binary Caching

Avoid recompilation by caching device binaries:

```c
// First run: build from source, then save
cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
cl_util_write_binaries(prog, "mykernel");  // writes "mykernel_<device>.bin"

// Subsequent runs: restore from binary
cl_program prog = cl_util_read_binaries(ctx, &device, 1, "mykernel", &err);
if (err == CL_SUCCESS) {
    clBuildProgram(prog, 1, &device, NULL, NULL, NULL);  // link only
} else {
    // Fall back to source compilation
}
```

---

## Error Handling Patterns

### C — goto-based cleanup (SDK pattern)
```c
cl_int error = CL_SUCCESS, end_error = CL_SUCCESS;
cl_mem buf = NULL;
cl_kernel kernel = NULL;

OCLERROR_PAR(buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &error),
             error, cleanup_kernel);
OCLERROR_PAR(kernel = clCreateKernel(prog, "myfunc", &error), error, cleanup_buf);

// ... use buf and kernel ...

cleanup_kernel:
    OCLERROR_RET(clReleaseKernel(kernel), end_error, cleanup_buf);
cleanup_buf:
    OCLERROR_RET(clReleaseMemObject(buf), end_error, done);
done:
    if (error) cl_util_print_error(error);
    return error;
```

### C++ — RAII + exceptions
```cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

try {
    cl::Context ctx{CL_DEVICE_TYPE_GPU};
    cl::CommandQueue queue{ctx, ctx.getInfo<CL_CONTEXT_DEVICES>()[0]};
    cl::Program prog{ctx, source};
    prog.build();
    // All objects auto-released via RAII at scope exit
} catch (cl::BuildError& e) {
    for (auto& [dev, log] : e.getBuildLog())
        std::cerr << log;
} catch (cl::Error& e) {
    std::cerr << e.what() << " code=" << e.err();
}
```

### Check build log on failure (C)
```c
cl_int build_err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
if (build_err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = malloc(log_size);
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "Build log:\n%s\n", log);
    free(log);
}
```

---

## Build System Integration

### CMakeLists.txt (minimal)
```cmake
cmake_minimum_required(VERSION 3.16)
project(MyOpenCLApp)

find_package(OpenCL REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OpenCL::OpenCL)

# For C++ wrapper: add OpenCL-CLHPP to include path
target_include_directories(my_app PRIVATE
    path/to/OpenCL-CLHPP/include)
target_compile_definitions(my_app PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_TARGET_OPENCL_VERSION=200)
```

### Using OpenCL SDK (vcpkg)
```cmake
find_package(OpenCLSDK CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE OpenCL::SDK OpenCL::Utils OpenCL::UtilsCpp)
```

### Compile commands
```bash
# GCC/Clang (Linux)
g++ -std=c++17 main.cpp -lOpenCL -o my_app

# With explicit headers
g++ -std=c++17 -I/usr/include/CL -DCL_HPP_ENABLE_EXCEPTIONS \
    -DCL_HPP_TARGET_OPENCL_VERSION=200 main.cpp -lOpenCL -o my_app
```
