# HIP Math and Device Functions Reference

## Table of Contents
1. [Headers](#headers)
2. [Single-Precision Math (float)](#single-precision-math-float)
3. [Double-Precision Math (double)](#double-precision-math-double)
4. [Half-Precision (fp16)](#half-precision-fp16)
5. [Integer and Bit Manipulation](#integer-and-bit-manipulation)
6. [Complex Math](#complex-math)
7. [Type Conversions](#type-conversions)
8. [Precision Notes](#precision-notes)

---

## Headers

```cpp
#include <hip/hip_runtime.h>       // standard math intrinsics (device)
#include <hip/hip_math_constants.h> // HIP_PI, HIP_E, HIP_INF_F, etc.
#include <hip/hip_fp16.h>           // __half, half2 types
#include <hip/hip_complex.h>        // hipFloatComplex, hipDoubleComplex
#include <hip/hip_vector_types.h>   // float2, float4, double2, int4, etc.
```

---

## Single-Precision Math (float)

All functions below operate in device code (`__device__`). The `f`-suffix variants are single-precision. They map directly to GPU hardware instructions where available.

### Trigonometric
```cpp
float sinf(float x);
float cosf(float x);
float tanf(float x);
float asinf(float x);
float acosf(float x);
float atanf(float x);
float atan2f(float y, float x);
void  sincosf(float x, float* sinval, float* cosval);  // compute both at once
```

### Hyperbolic
```cpp
float sinhf(float x);
float coshf(float x);
float tanhf(float x);
float asinhf(float x);
float acoshf(float x);
float atanhf(float x);
```

### Exponential and Logarithm
```cpp
float expf(float x);        // e^x
float exp2f(float x);       // 2^x
float exp10f(float x);      // 10^x
float expm1f(float x);      // e^x - 1 (accurate for small x)
float logf(float x);        // ln(x)
float log2f(float x);       // log2(x)
float log10f(float x);      // log10(x)
float log1pf(float x);      // ln(1+x) (accurate for small x)
float logbf(float x);       // exponent of x
```

### Power and Root
```cpp
float powf(float base, float exp);
float sqrtf(float x);
float rsqrtf(float x);       // 1/sqrt(x) — fast hardware intrinsic
float cbrtf(float x);        // cube root
float rcbrtf(float x);       // 1/cbrt(x)
float hypotf(float x, float y);  // sqrt(x^2+y^2)
```

### Rounding
```cpp
float floorf(float x);
float ceilf(float x);
float truncf(float x);
float roundf(float x);        // round to nearest, ties away from zero
float rintf(float x);         // round to nearest, ties to even
float nearbyintf(float x);
float modff(float x, float* iptr);  // separate integer/fractional parts
```

### Misc
```cpp
float fabsf(float x);         // absolute value
float fmaf(float x, float y, float z);  // fused multiply-add: x*y+z (single rounding)
float fminf(float x, float y);
float fmaxf(float x, float y);
float fdimf(float x, float y);   // max(x-y, 0)
float fmodf(float x, float y);   // remainder
float remainderf(float x, float y);
float erff(float x);          // error function
float erfcf(float x);         // complementary error function
float j0f(float x);           // Bessel function, order 0
float j1f(float x);           // Bessel function, order 1
float lgammaf(float x);       // log(|Gamma(x)|)
float tgammaf(float x);       // Gamma(x)
```

### Fast/Approximate Intrinsics (lower precision, faster)
```cpp
// Prefix __: AMD-specific approximate versions
float __sinf(float x);
float __cosf(float x);
float __expf(float x);
float __logf(float x);
float __powf(float base, float exp);
float __fdividef(float x, float y);  // approximate division
float __frcp_rn(float x);            // approximate 1/x
float __fsqrt_rn(float x);           // approximate sqrt
float __frsqrt_rn(float x);          // approximate 1/sqrt
```

---

## Double-Precision Math (double)

Same functions as single-precision but without `f` suffix:

```cpp
double sin(double x);    double asin(double x);
double cos(double x);    double acos(double x);
double tan(double x);    double atan(double x);  double atan2(double y, double x);
void sincos(double x, double* s, double* c);

double exp(double x);    double exp2(double x);   double log(double x);
double log2(double x);   double log10(double x);  double pow(double base, double exp);
double sqrt(double x);   double rsqrt(double x);  double cbrt(double x);

double floor(double x);  double ceil(double x);   double round(double x);
double trunc(double x);  double rint(double x);

double fabs(double x);
double fma(double x, double y, double z);  // fused multiply-add
double fmin(double x, double y);
double fmax(double x, double y);

double erf(double x);    double erfc(double x);
double lgamma(double x); double tgamma(double x);
```

---

## Half-Precision (fp16)

```cpp
#include <hip/hip_fp16.h>

// Basic type
__half    // 16-bit float
half2     // two __half packed (SIMD operations)

// Conversions
__half __float2half(float f);
float  __half2float(__half h);
__half __double2half(double d);

// Arithmetic (device only)
__half __hadd(__half a, __half b);   // a + b
__half __hsub(__half a, __half b);   // a - b
__half __hmul(__half a, __half b);   // a * b
__half __hdiv(__half a, __half b);   // a / b
__half __hfma(__half a, __half b, __half c); // a*b+c

// Comparison
bool __hgt(__half a, __half b);   // a > b
bool __hlt(__half a, __half b);   // a < b
bool __hge(__half a, __half b);   // a >= b
bool __heq(__half a, __half b);   // a == b

// half2 SIMD operations
half2 __hadd2(half2 a, half2 b);
half2 __hmul2(half2 a, half2 b);
half2 __hfma2(half2 a, half2 b, half2 c);

// Math intrinsics for __half
__half hsin(__half h);
__half hcos(__half h);
__half hexp(__half h);
__half hlog(__half h);
__half hsqrt(__half h);
__half hrsqrt(__half h);
__half hrcp(__half h);
```

---

## Integer and Bit Manipulation

```cpp
// Population count (number of set bits)
int __popc(unsigned int x);
int __popcll(unsigned long long x);

// Count leading zeros
int __clz(int x);
int __clzll(long long x);

// Count trailing zeros
int __ctz(unsigned int x);

// Find first set bit (from LSB)
int __ffs(int x);
int __ffsll(long long x);

// Byte-reverse
unsigned int  __brev(unsigned int x);
unsigned long long __brevll(unsigned long long x);

// Byte permute
unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);

// Multiply-add (64-bit result)
long long __mul64hi(long long x, long long y);   // high 64 bits of 64*64
unsigned long long __umul64hi(unsigned long long x, unsigned long long y);
int __mulhi(int x, int y);                       // high 32 bits of 32*32
unsigned int __umulhi(unsigned int x, unsigned int y);
```

---

## Complex Math

```cpp
#include <hip/hip_complex.h>

// Types
hipFloatComplex   // = float2 {x=real, y=imag}
hipDoubleComplex  // = double2

// Construction
hipFloatComplex  make_hipFloatComplex(float r, float i);
hipDoubleComplex make_hipDoubleComplex(double r, double i);

// Accessors
float  hipCrealf(hipFloatComplex c);   // real part
float  hipCimagf(hipFloatComplex c);   // imaginary part
double hipCreal(hipDoubleComplex c);
double hipCimag(hipDoubleComplex c);

// Arithmetic
hipFloatComplex hipCaddf(hipFloatComplex a, hipFloatComplex b); // a+b
hipFloatComplex hipCsubf(hipFloatComplex a, hipFloatComplex b); // a-b
hipFloatComplex hipCmulf(hipFloatComplex a, hipFloatComplex b); // a*b
hipFloatComplex hipCdivf(hipFloatComplex a, hipFloatComplex b); // a/b
hipFloatComplex hipConjf(hipFloatComplex z);  // conjugate
float           hipCabsf(hipFloatComplex z);  // |z|

// Double-precision versions: hipCadd, hipCsub, hipCmul, hipCdiv, hipConj, hipCabs
```

---

## Type Conversions

```cpp
// Float ↔ int (various rounding modes)
int   __float2int_rn(float f);    // round to nearest
int   __float2int_rz(float f);    // round toward zero
int   __float2int_ru(float f);    // round up
int   __float2int_rd(float f);    // round down
float __int2float_rn(int i);

unsigned int __float2uint_rn(float f);
float __uint2float_rn(unsigned int u);

// Float ↔ long long
long long      __float2ll_rn(float f);
float          __ll2float_rn(long long l);

// Bit-cast (reinterpret without conversion)
int   __float_as_int(float f);
float __int_as_float(int i);
unsigned int  __float_as_uint(float f);
float __uint_as_float(unsigned int u);
```

---

## Precision Notes

- GPU math functions follow IEEE 754 with device-specific rounding.
- `fmaf()` / `fma()` fuses multiply-add into a single instruction — use for stable accumulation.
- Fast intrinsics (`__sinf`, `__expf`, etc.) have ~1–2 ULP error vs ~4–5 ULP for standard; use when speed matters more than precision.
- `rsqrtf(x)` (hardware 1/sqrt) is faster than `1.0f/sqrtf(x)`.
- On AMD: `warpSize = 64` (wavefront 64); intrinsics like `__shfl` work on 64-lane wavefronts.
- Half-precision (`__half`) is suited for matrix math (tensor cores / matrix cores); not all standard math functions are available.
- For portability: prefer `HIP_PI` from `<hip/hip_math_constants.h>` over hardcoded π.

### Key Constants (`hip_math_constants.h`)
```cpp
HIP_PI          // π (double)
HIP_PI_F        // π (float)
HIP_E           // e (double)
HIP_E_F         // e (float)
HIP_INF         // +∞ (double)
HIP_INF_F       // +∞ (float)
HIP_NAN         // NaN (double)
HIP_NAN_F       // NaN (float)
HIP_SQRT2       // √2 (double)
HIP_SQRT2_F     // √2 (float)
```
