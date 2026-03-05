# cuRAND API

## Table of Contents
1. [Host API — Generator Lifecycle](#host-api--generator-lifecycle)
2. [Host API — Generator Configuration](#host-api--generator-configuration)
3. [Host API — Generation Functions](#host-api--generation-functions)
4. [Device API — In-Kernel Generation](#device-api--in-kernel-generation)
5. [Generator Types Reference](#generator-types-reference)
6. [Error Handling](#error-handling)

---

## Host API — Generator Lifecycle

The host API generates numbers on the GPU and returns results in device memory (or host memory for host generators).

### `curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) -> curandStatus_t`
**Description:** Creates a GPU-side generator. Numbers are generated on the device.
**Example:**
```cpp
curandGenerator_t gen;
CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
```

### `curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type) -> curandStatus_t`
**Description:** Creates a CPU-side generator. Numbers are generated on the host and written to host memory.
**Example:**
```cpp
curandGenerator_t gen;
CURAND_CHECK(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_XORWOW));
```

### `curandDestroyGenerator(curandGenerator_t generator) -> curandStatus_t`
**Description:** Destroys the generator and releases resources.

---

## Host API — Generator Configuration

### `curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) -> curandStatus_t`
**Description:** Sets the seed for reproducibility. For quasi-random generators, use `curandSetQuasiRandomGeneratorDimensions`.
**Example:**
```cpp
CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
```

### `curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) -> curandStatus_t`
**Description:** Advances the sequence by `offset` samples (useful for parallel independent streams).

### `curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) -> curandStatus_t`
**Description:** Controls ordering. `CURAND_ORDERING_PSEUDO_BEST` maximizes throughput; `CURAND_ORDERING_PSEUDO_SEEDED` gives reproducible independent-stream behavior.

### `curandSetStream(curandGenerator_t generator, cudaStream_t stream) -> curandStatus_t`
**Description:** Associates a CUDA stream with the generator.

**Full setup pattern:**
```cpp
curandGenerator_t gen;
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
CURAND_CHECK(curandSetStream(gen, stream));
CURAND_CHECK(curandSetGeneratorOffset(gen, 0ULL));
CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));
```

---

## Host API — Generation Functions

All generation functions write `num` samples to `outputPtr` (device pointer for `curandCreate`, host pointer for `curandCreateHost`).

### `curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) -> curandStatus_t`
**Description:** Generates `num` uniform floats in `(0, 1]`.
**Example:**
```cpp
float *d_data;
CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
CURAND_CHECK(curandGenerateUniform(gen, d_data, N));
CUDA_CHECK(cudaStreamSynchronize(stream));
```

### `curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num) -> curandStatus_t`
**Description:** Double-precision uniform in `(0, 1]`.

### `curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t num, float mean, float stddev) -> curandStatus_t`
**Description:** Gaussian (normal) distribution. `num` must be even.
**Example:**
```cpp
CURAND_CHECK(curandGenerateNormal(gen, d_data, N, 0.0f, 1.0f));  // Standard normal
```

### `curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t num, double mean, double stddev) -> curandStatus_t`
**Description:** Double-precision normal distribution.

### `curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t num, float mean, float stddev) -> curandStatus_t`
**Description:** Log-normal distribution. Values are `exp(N(mean, stddev))`.

### `curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t num, double lambda) -> curandStatus_t`
**Description:** Poisson distribution with rate parameter `lambda`.
**Example:**
```cpp
unsigned int *d_counts;
CUDA_CHECK(cudaMalloc(&d_counts, N * sizeof(unsigned int)));
CURAND_CHECK(curandGeneratePoisson(gen, d_counts, N, 10.0));
```

### `curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num) -> curandStatus_t`
**Description:** Raw 32-bit unsigned integers (full period, no distribution applied).

### `curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num) -> curandStatus_t`
**Description:** Raw 64-bit unsigned integers (XORWOW and MRG32k3a only).

---

## Device API — In-Kernel Generation

For maximum throughput when generating numbers inside a kernel. Include `<curand_kernel.h>`.

**Key types:** `curandState_t` (XORWOW), `curandStateMRG32k3a_t`, `curandStateSobol32_t`, `curandStatePhilox4_32_10_t`.

### Setup kernel:
```c
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread gets same seed, different sequence number, no offset
    curand_init(seed, id, 0, &state[id]);
}
```

### Generation kernel:
```c
__global__ void generate_kernel(curandState *state, float *out, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    curandState localState = state[id];        // copy to local for speed
    float val = curand_uniform(&localState);   // (0, 1]
    out[id] = val;
    state[id] = localState;                    // store back
}
```

**Device generation functions:**
- `curand_uniform(&state)` — float in (0, 1]
- `curand_uniform_double(&state)` — double in (0, 1]
- `curand_normal(&state)` — N(0, 1) float
- `curand_normal_double(&state)` — N(0, 1) double
- `curand_log_normal(&state, mean, stddev)` — log-normal float
- `curand_poisson(&state, lambda)` — Poisson unsigned int
- `curand(&state)` — raw 32-bit unsigned int

**Performance tip:** Use `curand_uniform4(&state)` to generate 4 values in one call (Philox state).

---

## Generator Types Reference

| Type constant | Algorithm | Use case |
|---|---|---|
| `CURAND_RNG_PSEUDO_XORWOW` | XORWOW | Default; fast pseudorandom |
| `CURAND_RNG_PSEUDO_MRG32K3A` | MRG32k3a | Long period, multiple streams |
| `CURAND_RNG_PSEUDO_MTGP32` | Mersenne Twister | High quality pseudorandom |
| `CURAND_RNG_PSEUDO_MT19937` | MT19937 | Standard Mersenne Twister |
| `CURAND_RNG_PSEUDO_PHILOX4_32_10` | Philox-4x32-10 | Counter-based, fastest in device |
| `CURAND_RNG_QUASI_SOBOL32` | Sobol' 32-bit | Low-discrepancy quasi-random |
| `CURAND_RNG_QUASI_SCRAMBLED_SOBOL32` | Scrambled Sobol' | Better uniformity |
| `CURAND_RNG_QUASI_SOBOL64` | Sobol' 64-bit | Higher precision quasi-random |

---

## Error Handling

```cpp
#define CURAND_CHECK(err) do {                                          \
    curandStatus_t _s = (err);                                          \
    if (_s != CURAND_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuRAND error %d at %s:%d\n",                  \
                _s, __FILE__, __LINE__);                                \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)
```

**Header:** `#include <curand.h>` (host API) — link: `-lcurand`.
**Device API header:** `#include <curand_kernel.h>` (no separate link needed, compiled by nvcc).
