# GreyCat C SDK — Services: Crypto, Geo, Time, Math & Util

Self-contained helper domains: cryptography (SHA-256/HMAC/Base64), geospatial (geohashing/Haversine), timezone-aware date/time, WASM math shims, and utilities (Morton codes, hex, parsing, deep equality, sorting, licensing).

_Part of the GreyCat C SDK reference (each file is linked from the skill's SKILL.md). Sibling references: api_core.md · api_memory_text.md · api_collections.md · api_runtime_storage.md · api_services.md._

## Contents

- [gc/crypto.h — Cryptography](#gccrypto-h)
- [gc/geo.h — Geospatial Operations](#gcgeo-h)
- [gc/time.h — Date & Time](#gctime-h)
- [gc/math.h — Math Functions (WASM)](#gcmath-h)
- [gc/util.h — Utility Functions](#gcutil-h)

---

<a id="gccrypto-h"></a>
## gc/crypto.h — Cryptography

SHA-256 hashing, HMAC-SHA-256, and Base64/Base64URL encoding/decoding.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `gc_crypto_sha256_len` | 32 | SHA-256 digest length in bytes |
| `GC_CRYPTO_HMAC_SHA256_BLOCKSIZE` | 64 | HMAC-SHA-256 block size in bytes |

### SHA-256

```c
typedef struct gc_crypto__sha256 {
    union {
        u32_t u32[8];
        unsigned char u8[gc_crypto_sha256_len];  // 32-byte hash
    } u;
} gc_crypto_sha256_t;

typedef struct {
    u32_t s[8];
    union {
        u32_t u32[16];
        unsigned char u8[64];
    } buf;
    size_t bytes;
} gc_crypto_sha256_ctx_t;
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__sha256` | `void gc_crypto__sha256(gc_crypto_sha256_t *sha, const void *p, size_t size)` | Compute SHA-256 hash of `p` (one-shot) |

### HMAC-SHA-256

```c
typedef struct gc_crypto__hmac_sha256 {
    gc_crypto_sha256_t sha;
} gc_crypto_hmac_sha256_t;

typedef struct gc_crypto_hmac_sha256_ctx {
    gc_crypto_sha256_ctx_t sha;
    u64_t k_opad[GC_CRYPTO_HMAC_SHA256_BLOCKSIZE / sizeof(u64_t)];  // HMAC outer padding
} gc_crypto_hmac_sha256_ctx_t;
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__hmac_sha256` | `void gc_crypto__hmac_sha256(gc_crypto_hmac_sha256_ctx_t *ctx, gc_crypto_hmac_sha256_t *hmac, const void *k, size_t ksize, const void *d, size_t dsize)` | Compute HMAC-SHA-256 with key `k` and data `d` |

### Base64 / Base64URL

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_crypto__base64_encode_len` | `u32_t gc_crypto__base64_encode_len(u32_t len)` | Compute output length for Base64 encoding |
| `gc_crypto__base64_encode` | `u32_t gc_crypto__base64_encode(const char *in, u32_t inlen, char *out)` | Encode to Base64. Returns output length. |
| `gc_crypto__base64_decode_len` | `u32_t gc_crypto__base64_decode_len(u32_t input_len)` | Compute output length for Base64 decoding |
| `gc_crypto__base64_decode` | `u32_t gc_crypto__base64_decode(const char *input, u32_t input_len, char *output)` | Decode from Base64. Returns output length. |
| `gc_crypto__base64url_encode_len` | `u32_t gc_crypto__base64url_encode_len(u32_t len)` | Compute output length for Base64URL encoding |
| `gc_crypto__base64url_encode` | `u32_t gc_crypto__base64url_encode(const char *str, u32_t str_len, char *output)` | Encode to Base64URL (URL-safe variant). Returns output length. |
| `gc_crypto__base64url_decode_len` | `u32_t gc_crypto__base64url_decode_len(u32_t input_len)` | Compute output length for Base64URL decoding |
| `gc_crypto__base64url_decode` | `u32_t gc_crypto__base64url_decode(const char *input, u32_t input_len, char *output)` | Decode from Base64URL. Returns output length. |

### Usage Examples

#### SHA-256 one-shot hash

`gc_crypto__sha256` computes the 32-byte digest of a buffer in a single call. The 32 raw bytes are exposed via the `u.u8` union member (or `u.u32` for word access). Pattern taken from how the runtime hashes object/file content.

```c
// ctx is your gc_machine_t *; content is some gc_string_t *
gc_crypto_sha256_t sha256;
gc_crypto__sha256(&sha256, content->buffer, content->size);

// sha256.u.u8 now holds the 32 raw digest bytes (gc_crypto_sha256_len == 32).
// Render it for output by base64url-encoding the digest into the scratch buffer:
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__clear(buf);
gc_buffer__prepare(buf, gc_crypto__base64url_encode_len(gc_crypto_sha256_len));
u32_t encoded_len =
    gc_crypto__base64url_encode((char *) sha256.u.u8, gc_crypto_sha256_len, buf->data);
gc_string_t *res = gc_string__create_from(buf->data, encoded_len, ctx);
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) res}, gc_type_object);
gc_object__un_mark((gc_object_t *) res, ctx);
```

#### Base64 encode / decode through the gen buffer

The `*_len` helpers size the destination first; then the encode/decode call writes into a prepared buffer and returns the exact number of bytes written. This is the canonical native-function pattern used by `std::util::Crypto`. Swap the `base64` calls for `base64url` for the URL-safe alphabet — the call shape is identical.

```c
// Encode: input is a gc_string_t *
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__clear(buf);
u32_t max_len = gc_crypto__base64_encode_len(input->size);
gc_buffer__prepare(buf, max_len);
u32_t encoded_len = gc_crypto__base64_encode(input->buffer, input->size, buf->data);
gc_string_t *res = gc_string__create_from(buf->data, encoded_len, ctx);
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) res}, gc_type_object);
gc_object__un_mark((gc_object_t *) res, ctx);

// Decode: reverse direction, same buffer discipline
gc_buffer__clear(buf);
u32_t dec_max = gc_crypto__base64_decode_len(input->size);
gc_buffer__prepare(buf, dec_max);
u32_t decoded_len = gc_crypto__base64_decode(input->buffer, input->size, buf->data);
gc_string_t *decoded = gc_string__create_from(buf->data, decoded_len, ctx);
gc_machine__set_result(ctx, (gc_slot_t) {.object = (gc_object_t *) decoded}, gc_type_object);
gc_object__un_mark((gc_object_t *) decoded, ctx);
```

#### HMAC-SHA-256 and key derivation chaining

`gc_crypto__hmac_sha256` takes a caller-owned context and output struct (both stack-allocated, no allocator needed). The 32-byte MAC lands in `hmac.sha.u.u8`. The signing key is supplied by the plugin — e.g. a `gc_string_t *` passed as a native-function argument — never read from runtime internals:

```c
// key and password are gc_string_t * arguments to your native function
gc_crypto_hmac_sha256_ctx_t hmac_sha256_ctx;
gc_crypto_hmac_sha256_t hmac_sha256;
gc_crypto__hmac_sha256(&hmac_sha256_ctx, &hmac_sha256,
                       key->buffer, key->size,
                       password->buffer, password->size);
// hmac_sha256.sha.u.u8 holds the 32-byte signature; base64url it for transport.
```

When deriving a chained signing key (e.g. the AWS Sig-V4 kSigning derivation), feed each MAC back in as the key for the next round. Re-zero the context and output between rounds:

```c
unsigned char current_key[gc_crypto_sha256_len];
gc_crypto_hmac_sha256_ctx_t hmac_ctx;
gc_crypto_hmac_sha256_t hmac;

// kDate = HMAC("AWS4" + secret_key, date)
gc_crypto__hmac_sha256(&hmac_ctx, &hmac, tmp_buf->data, tmp_buf->size,
                       (const u8_t *) timestamp, 8);
memcpy(current_key, hmac.sha.u.u8, gc_crypto_sha256_len);

// kRegion = HMAC(kDate, region)
memset(&hmac, 0, sizeof(hmac));
memset(&hmac_ctx, 0, sizeof(hmac_ctx));
gc_crypto__hmac_sha256(&hmac_ctx, &hmac, current_key, gc_crypto_sha256_len,
                       (const u8_t *) region_str, region_len);
memcpy(current_key, hmac.sha.u.u8, gc_crypto_sha256_len);
// ... continue with kService, kSigning, then the final string_to_sign MAC.
```


---

<a id="gcgeo-h"></a>
## gc/geo.h — Geospatial Operations

Geospatial encoding/decoding using a 64-bit geohash and Haversine distance calculation.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_CORE_GEO_EPS` | 0.00000001 | Coordinate epsilon for comparison |
| `GC_CORE_GEO_LAT_MIN` | -85.05112878 | Minimum latitude (Web Mercator) |
| `GC_CORE_GEO_LAT_MAX` | 85.05112878 | Maximum latitude (Web Mercator) |
| `GC_CORE_GEO_LNG_MIN` | -180 | Minimum longitude |
| `GC_CORE_GEO_LNG_MAX` | 180 | Maximum longitude |
| `GC_CORE_GEO_STEP_MAX` | 32 | Maximum hash step resolution |
| `EARTH_RADIUS_M` | 6371000.0 | Earth radius in meters |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_geo__encode` | `geo_t gc_geo__encode(f64_t latitude, f64_t longitude)` | Encode lat/lng to a 64-bit geohash |
| `gc_geo__decode` | `void gc_geo__decode(geo_t hash, f64_t *lat, f64_t *lon)` | Decode a geohash back to lat/lng |
| `gc_geo__distance` | `f64_t gc_geo__distance(geo_t h1, geo_t h2)` | Compute the great-circle distance (in meters) between two geohashes |

### Usage Examples

These three functions are the entire public surface of `gc/geo.h`. A `geo_t` is a 64-bit packed geohash carried in the `.u64` field of a `gc_slot_t`; coordinates are plain `f64_t` degrees (WGS84). There is no allocation involved, so no mark/unmark discipline applies — `geo_t` values are scalars.

**Encode lat/lng into a geohash and store it as a result slot**

The compiler, CSV/JSON readers, and the `Random.uniformGeo` builtin all produce a `geo_t` the same way: decode/compute two `f64_t` degrees, encode, then publish the packed value with `gc_type_geo`.

```c
// Build a geo value from two floats and set it as the call result.
// (mirrors csv_reader.c:564 and std/util/random.c:132)
void my_geo_from_coords(gc_machine_t *ctx) {
    f64_t lat = 49.6116; // Luxembourg City
    f64_t lng = 6.1319;

    geo_t hash = gc_geo__encode(lat, lng);

    gc_machine__set_result(ctx, (gc_slot_t){.u64 = hash}, gc_type_geo);
}
```

**Decode a geohash back to latitude/longitude**

`gc_geo__decode` writes through two `f64_t *` out-params. This is the canonical way every accessor (`geo.lat()`, `geo.lng()`, `geo.toString()`) reads a point.

```c
// Extract latitude from a geo `this` value (mirrors std/core/geo.c:32-38).
void my_geo_lat(gc_machine_t *ctx) {
    u64_t this = gc_machine__this(ctx).u64; // the geo_t, packed in .u64
    f64_t lat;
    f64_t lng;
    gc_geo__decode(this, &lat, &lng);
    gc_machine__set_result(ctx, (gc_slot_t){.f64 = lat}, gc_type_float);
}
```

**Round-trip: decode two endpoints, interpolate, re-encode**

When you need to derive a new point from existing geohashes, decode each endpoint into degrees, do the math in `f64_t`, then re-encode. This is the pattern used by `Random.uniformGeo` (decode min/max corners, pick a point in between, encode the result).

```c
// Midpoint of two geohashes (grounded in std/util/random.c:126-132).
geo_t my_geo_midpoint(geo_t a, geo_t b) {
    f64_t a_lat, a_lng;
    f64_t b_lat, b_lng;
    gc_geo__decode(a, &a_lat, &a_lng);
    gc_geo__decode(b, &b_lat, &b_lng);

    f64_t mid_lat = (a_lat + b_lat) * 0.5;
    f64_t mid_lng = (a_lng + b_lng) * 0.5;
    return gc_geo__encode(mid_lat, mid_lng);
}
```

**Great-circle distance between two points**

`gc_geo__distance` returns the Haversine distance in meters (`EARTH_RADIUS_M = 6371000.0`). It decodes both hashes internally, so you pass packed `geo_t` values directly — no manual decode needed.

```c
// geo.dist(other) -> meters (mirrors std/core/geo.c:27-30).
void my_geo_distance(gc_machine_t *ctx) {
    geo_t this = gc_machine__this(ctx).u64;
    geo_t other = my_arg0(ctx).u64; // the other geo argument

    f64_t meters = gc_geo__distance(this, other);

    gc_machine__set_result(ctx, (gc_slot_t){.f64 = meters}, gc_type_float);
}

// Accumulate total path length over a sequence of geo points
// (mirrors std/core/geo.c:445 — length of a polyline).
f64_t my_geo_path_length(const geo_t *points, u32_t n) {
    f64_t total = 0.0;
    for (u32_t i = 1; i < n; i++) {
        total += gc_geo__distance(points[i - 1], points[i]);
    }
    return total;
}
```


---

<a id="gctime-h"></a>
## gc/time.h — Date & Time

Calendar conversion, timezone handling, ISO 8601 parsing, and formatting. GreyCat timestamps are microseconds since Unix epoch.

### Constants

**Time unit conversions (microseconds):**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_CORE_TIME_1_MILLISECOND` | 1,000 | 1ms in microseconds |
| `GC_CORE_TIME_1_SECOND` | 1,000,000 | 1s in microseconds |
| `GC_CORE_TIME_1_MINUTE` | 60,000,000 | 1min in microseconds |
| `GC_CORE_TIME_1_HOUR` | 3,600,000,000 | 1h in microseconds |
| `GC_CORE_TIME_1_DAY` | 86,400,000,000 | 1d in microseconds |

**Sub-second conversions:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_MICROSECONDS_IN_MILLISECOND` | 1,000 | Microseconds per millisecond |
| `GC_MILLISECONDS_IN_SECOND` | 1,000 | Milliseconds per second |
| `GC_MICROSECONDS_IN_SECOND` | 1,000,000 | Microseconds per second |

**Calendar constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_SECONDS_IN_MINUTE` | 60 | |
| `GC_SECONDS_IN_HOUR` | 3600 | |
| `GC_SECONDS_IN_DAY` | 86400 | |
| `GC_MINUTES_IN_HOUR` | 60 | |
| `GC_HOURS_IN_DAY` | 24 | |
| `GC_DAYS_IN_WEEK` | 7 | |
| `GC_MONTHS_IN_YEAR` | 12 | |
| `GC_DAYS_PER_ERA` | 146097 | Days in a 400-year era |
| `GC_DAYS_PER_CENTURY` | 36524 | Days in a 100-year period |
| `GC_DAYS_PER_4_YEARS` | 1461 | Days in a 4-year leap cycle (3×365+366) |
| `GC_DAYS_PER_YEAR` | 365 | Days in a non-leap year |
| `GC_DAYS_IN_JANUARY` | 31 | |
| `GC_DAYS_IN_FEBRUARY` | 28 | Days in non-leap February |
| `GC_YEARS_PER_ERA` | 400 | Years per era |
| `GC_YEAR_BASE` | 1900 | Base year for `gc_tm_t.tm_year` offset |

**Epoch adjustment constants (for internal calendar math):**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_EPOCH_ORIG_YEAR` | 70 | Unix epoch year offset (1970 - 1900) |
| `GC_EPOCH_ORIG_WEEK_DAY` | 4 | 1970-01-01 was a Thursday (0=Sunday) |
| `GC_EPOCH_ADJUSTMENT_DAYS` | 719468 | Days from 0000-03-01 to 1970-01-01 |
| `GC_ADJUSTED_EPOCH_YEAR` | 0 | Adjusted epoch year (Year 0, March 1) |
| `GC_ADJUSTED_EPOCH_WDAY` | 3 | 0000-03-01 was a Wednesday |
| `GC_MIN_YEAR` | `INT32_MIN + 1900` | Minimum representable year |
| `GC_MAX_YEAR` | `(i64_t) INT32_MAX + 1900` | Maximum representable year |
| `GC_MKTIME_MAX_TM_YEAR` | 10000 | Maximum absolute `tm_year` (year − 1900) supported by `gc_mktime_safe` |
| `GC_MKTIME_MIN_YEAR` | `GC_YEAR_BASE - GC_MKTIME_MAX_TM_YEAR` (= -8100) | Inclusive lower year bound for which `gc_mktime_safe` returns a valid epoch |
| `GC_MKTIME_MAX_YEAR` | `GC_YEAR_BASE + GC_MKTIME_MAX_TM_YEAR` (= 11900) | Inclusive upper year bound for which `gc_mktime_safe` returns a valid epoch |

**Timezone conversion status codes:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GC_DTZ_OK` | 0 | Conversion succeeded |
| `GC_DTZ_ILLEGAL_TIMESTAMP` | 1 | Invalid or illegal timestamp |

**Days-in-month lookup:**

```c
static const int DAYS_IN_MONTH[12] = {31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
// Note: index 1 (February) is 0 — use GC_DAYS_IN_MONTH(m, y) macro instead
```

### Calendar Structure

```c
typedef struct {
    i32_t tm_sec;        // Seconds [0-60] (60 for leap second)
    i32_t tm_min;        // Minutes [0-59]
    i32_t tm_hour;       // Hours [0-23]
    i32_t tm_mday;       // Day of month [1-31]
    i32_t tm_mon;        // Month [0-11]
    i32_t tm_year;       // Year - 1900
    i32_t tm_wday;       // Day of week [0-6] (0 = Sunday)
    i32_t tm_yday;       // Day of year [0-365]
    i32_t tm_us_offset;  // Microsecond offset within the second
} gc_tm_t;
```

### Timezone Parse Result

```c
typedef struct {
    i32_t tz_hh_offset;   // Timezone hour offset
    i32_t tz_mm_offset;   // Timezone minute offset
    bool parsed;          // Whether a timezone was found in the string
} gc_core_time_parse_tz_t;
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gc_time__now` | `i64_t gc_time__now()` | Get current time in microseconds since epoch |
| `gc_time__tm_from_time` | `gc_tm_t gc_time__tm_from_time(i64_t time, u32_t timezone)` | Convert a GreyCat timestamp to calendar fields |
| `gc_gmtime_r_safe` | `void gc_gmtime_r_safe(i64_t tim_p, gc_tm_t *__restrict res)` | Convert a timestamp to UTC calendar (safe, no timezone) |
| `gc_mktime_safe` | `i64_t gc_mktime_safe(gc_tm_t *tim_p)` | Convert calendar fields to a timestamp |
| `gc_dtz_utc_to_time_zone` | `i32_t gc_dtz_utc_to_time_zone(u32_t time_zone, i64_t utc_epoch, i64_t *localized_epoch)` | Convert UTC epoch to a localized epoch. Returns `GC_DTZ_OK` on success. |
| `gc_dtz_time_zone_to_utc` | `i32_t gc_dtz_time_zone_to_utc(u32_t time_zone, i64_t localized_epoch, i64_t *utc_epoch, u32_t *next_utc_offset)` | Convert localized epoch to UTC. |
| `gc_dtz_first_day_of_week` | `u8_t gc_dtz_first_day_of_week(u32_t time_zone)` | Get the first day of the week for a timezone (0=Sunday..6=Saturday). |
| `gc_core_time__parse_iso` | `bool gc_core_time__parse_iso(const char *c, size_t len, gc_tm_t *tm, gc_core_time_parse_tz_t *tz_offsets)` | Parse an ISO 8601 date/time string. |
| `gc__print_iso` | `void gc__print_iso(gc_buffer_t *buffer, const gc_tm_t *calendar, i64_t epoch_us, i64_t localized_epoch_s)` | Format a time as ISO 8601 string. |
| `gc_strftime_safe` | `size_t gc_strftime_safe(char *s, size_t maxsize, const char *format, const gc_tm_t *t, i32_t utc_offset)` | Format a time with a `strftime`-style format string. |
| `gc_dtz_time__print` | `u32_t gc_dtz_time__print(i64_t epoch_us, u32_t tz, const char *format_c_str, char *out, u32_t out_cap)` | Format a timestamp with timezone into a char buffer. Returns bytes written. |
| `gc_dtz_time__parse` | `bool gc_dtz_time__parse(const char *str, u32_t len, u32_t tz, i64_t *out_epoch_us)` | Parse a date/time string with timezone context into a UTC epoch (microseconds). Returns `true` on success. |

### Helper Macros

```c
GC_ISLEAP(y)              // Is year y a leap year?
GC_DAYS_IN_MONTH(m, y)    // Days in month m of year y
GC_DAYS_IN_YEAR(x)        // 365 or 366
GC_MODULO(x, y)           // True modulo (not C remainder)
```

### Usage Examples

All twelve functions are listed in the table above with byte-accurate signatures, but the section has no runnable code. The snippets below are grounded in the real runtime call sites. The recurring idiom: epochs are split into a whole-second part (passed through the calendar/timezone helpers) and a microsecond remainder (`tm_us_offset`) recombined as `utc_epoch * 1000000L + tm_us_offset`.

**Render a microsecond timestamp to ISO 8601 (timezone-aware).** This is the `gc_buffer__add_time` pattern: shift the second-granularity epoch into the target timezone, expand it into calendar fields, carry the sub-second remainder, then emit.

```c
// epoch_us: microseconds since Unix epoch; tz: timezone id from options
const i64_t epoch_s = epoch_us / GC_MICROSECONDS_IN_SECOND;
i64_t aligned_epoch_s;
gc_dtz_utc_to_time_zone(tz, epoch_s, &aligned_epoch_s);

gc_tm_t t = {0};
gc_gmtime_r_safe(aligned_epoch_s, &t);
t.tm_us_offset = (i32_t) (epoch_us % GC_MICROSECONDS_IN_SECOND);

gc__print_iso(buffer, &t, epoch_us, aligned_epoch_s); // appends e.g. 2026-06-22T14:30:00.000000+02:00 into the gc_buffer_t
```

**Format a timestamp with a strftime-style pattern.** Grounded in `gc_dtz_time__print`. `gc_time__tm_from_time` already folds the timezone shift into the returned calendar, so `gc_mktime_safe` reproduces the localized second-count and the difference against the UTC second-count yields the offset that `gc_strftime_safe` needs for `%z`-style fields.

```c
char out[128];
gc_tm_t calendar = gc_time__tm_from_time(epoch_us, tz);
i64_t localized_epoch_s = gc_mktime_safe(&calendar);
i32_t utc_offset = (i32_t) (localized_epoch_s - (epoch_us / GC_MICROSECONDS_IN_SECOND));
size_t n = gc_strftime_safe(out, sizeof(out), "%Y-%m-%d %H:%M:%S", &calendar, utc_offset);
// n bytes written into out (output is truncated to fit maxsize)
```

**Parse an ISO 8601 string into a UTC microsecond epoch.** This is the full `gc_dtz_time__parse` body. When the string carries its own offset (`tz_offsets.parsed`) fold it directly into the calendar; otherwise interpret the wall-clock fields in the caller-supplied timezone via `gc_dtz_time_zone_to_utc`, which reports `GC_DTZ_ILLEGAL_TIMESTAMP` for non-existent local times (e.g. inside a DST spring-forward gap).

```c
gc_tm_t tm = {0};
gc_core_time_parse_tz_t tz_offsets = {0};
if (!gc_core_time__parse_iso(str, len, &tm, &tz_offsets)) {
    // not a valid ISO 8601 string
    return false;
}

i64_t utc_epoch_s;
if (tz_offsets.parsed) {
    tm.tm_hour += tz_offsets.tz_hh_offset;
    tm.tm_min += tz_offsets.tz_mm_offset;
    utc_epoch_s = gc_mktime_safe(&tm);
} else {
    i64_t localized_epoch_s = gc_mktime_safe(&tm);
    if (gc_dtz_time_zone_to_utc(tz, localized_epoch_s, &utc_epoch_s, NULL) == GC_DTZ_ILLEGAL_TIMESTAMP) {
        return false; // local time does not exist in this timezone
    }
}
i64_t epoch_us = utc_epoch_s * 1000000L + tm.tm_us_offset; // recombine sub-second part
```

**Current epoch and first day of week.** `gc_time__now` returns microseconds since epoch directly (note: it is a stub returning 0 on `__wasm__` builds). `gc_dtz_first_day_of_week` drives locale-aware week-bucketing in calendar arithmetic.

```c
i64_t now_us = gc_time__now();
gc_tm_t calendar = gc_time__tm_from_time(now_us, tz);
const u8_t first_day_of_week = gc_dtz_first_day_of_week(tz); // 0=Sunday .. 6=Saturday
i32_t days_since_week_start = GC_MODULO(calendar.tm_wday - first_day_of_week, GC_DAYS_IN_WEEK);
```

**Round-trip note.** `gc_dtz_time_zone_to_utc`'s fourth argument `u32_t *next_utc_offset` (passed `NULL` above) receives the offset that applies just after the conversion — used by duration arithmetic that must re-localize after crossing a DST boundary, where the runtime recomputes `gc_mktime_safe` and converts again (see `src/std/core/time.c`).


---

<a id="gcmath-h"></a>
## gc/math.h — Math Functions (WASM)

On WASM targets, provides standalone implementations of standard math functions and constants (since `<math.h>` is unavailable). On native targets, simply includes `<math.h>` and `<stdlib.h>`.

### Constants (WASM only)

| Constant | Value | Description |
|----------|-------|-------------|
| `M_PI` | 3.14159265358979323846 | Pi (f64) |
| `PI` | 3.14159265358979323846 | Pi (f64) |
| `TAU` | 6.28318530717958647692 | 2*Pi (f64) |
| `HALF_PI` | 1.57079632679489661923 | Pi/2 (f64) |
| `E` | 2.71828182845904523536 | Euler's number (f64) |
| `LN2` | 0.69314718055994530942 | Natural log of 2 (f64) |
| `EPSILON` | 1.19209290e-7f | Machine epsilon (f32) |
| `SQRT2` | 1.41421356237f | Square root of 2 (f32) |

### Types (WASM only)

```c
typedef i64_t time_t;

typedef struct {
    int quot;  // Quotient
    int rem;   // Remainder
} div_t;
```

### Functions (WASM only)

Standard math functions: `abs`, `fabs`, `fabsf`, `hypotf`, `hypot`, `sin`, `sqrtf`, `sqrt`, `div`, `llabs`, `labs`, `exp`, `pow`, `powl`, `log`, `log2`, `log2l`, `cos`, `tan`, `atan`, `atan2`, `round`, `floor`, `ceil`, `ceill`, `isnan`, `isfinite`.

### Usage Examples

These functions are the standard C math API. On native targets they come straight from `<math.h>`/`<stdlib.h>`; on WASM the same prototypes are provided by GreyCat's freestanding implementation in `core/src/wasm/math.c`. Either way you call them identically — include `gc/math.h` and use the `f64_t`/`f32_t` types. The snippets below mirror how the runtime itself uses them, so they compile against both backends.

**Guarding `f64_t` values before storing them (`isnan`, `isfinite`)**

GreyCat values can carry NaN/Inf, which are illegal in many sinks (e.g. serialized buffers). Filter them at the boundary exactly as `gc_buffer` and `gc_geo` do:

```c
#include "gc/math.h"

// Reject non-finite inputs (mirrors gc_geo__encode in core/src/geo.c)
static f64_t sanitize(f64_t v, f64_t fallback) {
    if (!isfinite(v)) { // catches both +/-inf and NaN
        return fallback;
    }
    return v;
}

// Skip NaN slots when emitting (mirrors core/src/buffer.c)
void emit_f64(gc_buffer_t *buf, f64_t f) {
    if (isnan(f)) {
        gc_buffer__add_str(buf, "null", 4); // or gc_buffer__add_pstr(buf, "null")
    } else {
        gc_buffer__add_f64(buf, f);
    }
}
```

**Haversine great-circle distance (`sin`, `cos`, `sqrt`, `atan2`, `M_PI`)**

The geo module computes distance between two points using the canonical haversine formula. This is the idiomatic combination of trig + `sqrt` + `atan2` (taken from `gc_geo__distance`, core/src/geo.c:40):

```c
#include "gc/math.h"

// Use a distinct local name: gc/geo.h already defines EARTH_RADIUS_M (6371000.0).
#define MY_EARTH_RADIUS_M 6371008.8

static inline f64_t to_radians(f64_t degrees) {
    return degrees / 180.0 * M_PI;
}

f64_t haversine_m(f64_t lat1, f64_t lng1, f64_t lat2, f64_t lng2) {
    f64_t dLat = to_radians(lat2 - lat1);
    f64_t dLng = to_radians(lng2 - lng1);
    f64_t a = sin(dLat / 2) * sin(dLat / 2)
            + cos(to_radians(lat1)) * cos(to_radians(lat2))
            * sin(dLng / 2) * sin(dLng / 2);
    if (a > 1.0)      a = 1.0; // clamp against fp rounding
    else if (a < 0.0) a = 0.0;
    f64_t c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return MY_EARTH_RADIUS_M * c;
}
```

**Absolute differences and tolerance checks (`fabs`, `llabs`)**

Tensor comparison uses `llabs` for integer element deltas and `fabs` for float deltas; iterative solvers use `fabs` against a small epsilon (mirrors core/src/tensor.c:1331/1357 and src/std/cubic.c:176):

```c
#include "gc/math.h"

// Integer element delta (core/src/tensor.c)
i64_t int_delta = llabs(t1_data[i] - t2_data[i]);

// Float element delta (core/src/tensor.c)
f64_t f_delta = fabs(t1_data[i] - t2_data[i]);

// Convergence test in an iterative solver (src/std/cubic.c)
f64_t error = fabs(x_tmp - y[i]);
if (error < 1e-9) {
    // converged
}
```

**Rounding for unit conversion and progress (`floor`, `ceil`, `round`)**

`floor` truncates a timestamp down to whole seconds; `ceil` rounds a progress ratio up to a whole percent (mirrors core/src/dtz_helper.c:138 and src/cli/restore.c:267):

```c
#include "gc/math.h"

// Truncate microsecond/nanosecond time to whole seconds
i64_t seconds = (i64_t) floor((double) time / GC_CORE_TIME_1_SECOND);

// Round a fractional percentage up so partial progress never reads as 0
i64_t progress_percent = (i64_t) ceil(current_size / progress_max_size * 100);

// round() ties away from zero: round(2.5) == 3, round(-2.5) == -3
f64_t nearest = round(value);
```

**Powers and logarithms (`pow`, `exp`, `log`, `log2`)**

```c
#include "gc/math.h"

f64_t y     = pow(base, exponent); // base^exponent
f64_t growth = exp(rate * t);      // e^(rate*t)
f64_t ln_x   = log(x);             // natural log
f64_t bits   = log2(n);            // log base 2, e.g. tree depth estimate
```


---

<a id="gcutil-h"></a>
## gc/util.h — Utility Functions

Miscellaneous utilities: Morton code (Z-order curve) encoding/decoding for spatial indexing, hex conversion, number parsing, JSON parsing, deep equality, slot hashing, duration parsing, sorting, and license level checking.

### Morton Code (Z-Order Curve)

These functions interleave/deinterleave integer coordinates into a single 64-bit key for spatial indexing.

**2D:**

| Function | Description |
|----------|-------------|
| `interleave64(xlo, ylo)` | Interleave two `u32_t` values into a `u64_t` |
| `deinterleave64(interleaved)` | Deinterleave a `u64_t` back into two `u32_t` |
| `gc_morton__deinterleave64_2di(interleaved, xlo)` | Deinterleave to two `i32_t` |
| `gc_morton__deinterleave64_2df(interleaved, xlo)` | Deinterleave to two `f32_t` |

**3D:**

| Function | Description |
|----------|-------------|
| `interleave64_3d(xlo)` | Interleave three `u32_t` values |
| `deinterleave64_3d(c, xlo)` | Deinterleave to three `u32_t` |
| `gc_morton__deinterleave64_3di(interleaved, xlo)` | Deinterleave to three `i32_t` |
| `gc_morton__deinterleave64_3df(interleaved, xlo)` | Deinterleave to three `f32_t` |
| `deinterleave64_3d_x0/x1/x2(c)` | Extract individual components |

**4D:**

| Function | Description |
|----------|-------------|
| `interleave64_4d(xlo)` | Interleave four `u16_t` values |
| `deinterleave64_4d(x, xlo)` | Deinterleave to four `u16_t` |
| `gc_morton__deinterleave64_4di(interleaved, xlo)` | Deinterleave to four `i16_t` |
| `gc_morton__deinterleave64_4df(interleaved, xlo)` | Deinterleave to four `f32_t` |
| `deinterleave64_4d_x0/x1/x2/x3(x)` | Extract individual components |

### Hex Conversion

| Function | Description |
|----------|-------------|
| `gc_common__hex2bin(dest, src, len)` | Convert hex string to binary bytes |
| `gc_common__hex2bin_len(len)` | Compute output length for hex-to-binary |
| `gc_common__bin2hex(dest, src, len)` | Convert binary bytes to hex string |
| `gc_common__bin2hex_len(len)` | Compute output length for binary-to-hex |

### Parsing

| Function | Description |
|----------|-------------|
| `gc_common__parse_number(str, str_len)` | Parse an unsigned integer from a string. `str_len` is `u64_t *` (in: available, out: bytes consumed). |
| `gc_common__parse_sign_number(str, str_len)` | Parse a signed integer from a string. `str_len` is `u32_t *` (in: available, out: bytes consumed). |
| `gc_common__parse_date_iso8601(data, len, epoch_utc)` | Parse an ISO 8601 date string to a UTC epoch (microseconds) |
| `gc_json__parse(ctx, str, len, result, type, type_d)` | Parse a JSON string into a GreyCat slot. Returns `true` on success. |
| `gc_duration__parse(str, len, duration)` | Parse a duration string (e.g., "1h30m") into microseconds |

### Deep Equality & Hashing

| Function | Description |
|----------|-------------|
| `gc__deep_equals(left, left_type, right, right_type, prog)` | Deep structural equality comparison of two slots |
| `gc_slot__hash(slot, type, prog)` | Compute a hash value for a slot |

### Sorting

```c
typedef struct gc_sort_slot {
    u64_t value;   // Sort key
    u64_t index;   // Original index
} gc_sort_slot_t;

void gc_sort__piposort(gc_sort_slot_t *array, u64_t nmemb, gc_type_t t, bool asc, gc_allocator_t *allocator);
```

A stable sort implementation for arrays of sort slots. The trailing `gc_allocator_t *allocator` parameter is used for temporary workspace.

### License

```c
typedef enum gc_license_level {
    gc_license_level_community = 0,
    gc_license_level_pro       = 1,
    gc_license_level_server    = 2,
    gc_license_level_platform  = 3
} gc_license_level_t;

gc_license_level_t gc_license__level();  // Get the current license level
```

### Usage Examples

All declarations in this section are currently documented as signature tables only. The snippets below are grounded in real runtime call sites.

**Hex encode/decode using the `_len` helpers to size the buffer.** The `*_len` functions compute the exact output size; always `gc_buffer__prepare` before writing. This mirrors the `Crypto::hexEncode`/`Crypto::hexDecode` native implementations.

```c
// Encode binary bytes -> hex string (Crypto::hexEncode pattern)
void my_hex_encode(gc_machine_t *ctx) {
    const gc_string_t *input = (gc_string_t *) gc_my_Crypto_hex_encode_v(ctx).object;
    gc_buffer_t *buf = gc_machine__get_buffer(ctx);
    gc_buffer__clear(buf);
    u32_t len = gc_common__bin2hex_len(input->size);   // == input->size * 2
    gc_buffer__prepare(buf, len);
    gc_common__bin2hex(buf->data, input->buffer, input->size);

    gc_string_t *res = gc_string__create_from(buf->data, len, ctx);
    gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *) res}, gc_type_object);
    gc_object__un_mark((gc_object_t *) res, ctx);
}

// Decode hex string -> binary bytes (Crypto::hexDecode pattern)
void my_hex_decode(gc_machine_t *ctx) {
    const gc_string_t *input = (gc_string_t *) gc_my_Crypto_hex_decode_v(ctx).object;
    gc_buffer_t *buf = gc_machine__get_buffer(ctx);
    gc_buffer__clear(buf);
    u32_t len = gc_common__hex2bin_len(input->size);   // == input->size / 2
    gc_buffer__prepare(buf, len);
    if (!gc_common__hex2bin(buf->data, input->buffer, input->size)) {
        gc_machine__set_runtime_error(ctx, "invalid hex input");
        return;
    }
    gc_string_t *res = gc_string__create_from(buf->data, len, ctx);
    gc_machine__set_result(ctx, (gc_slot_t){.object = (gc_object_t *) res}, gc_type_object);
    gc_object__un_mark((gc_object_t *) res, ctx);
}
```

**Parsing numbers, durations, and ISO-8601 dates from configuration strings.** `gc_common__parse_number` / `gc_common__parse_sign_number` take `*str_len` as in/out: pass the available length, read back how many bytes were consumed so you can detect trailing garbage or parse a suffix. Note the two functions disagree on the pointer's width — `gc_common__parse_number` takes `u64_t *str_len` while `gc_common__parse_sign_number` takes `u32_t *str_len`; match the local's type to the callee or you'll get a pointer-type mismatch. This is exactly how env-var option parsing works.

```c
// Parse "<number><suffix>" — consume digits, then look at what's left
bool parse_bytes(const char *str, u32_t str_len, u64_t *bytes) {
    u64_t num_len = str_len;                            // in: available, out: consumed (u64_t* required)
    u64_t value = gc_common__parse_number(str, &num_len);
    const char *suffix = str + num_len;                 // remaining chars after the digits
    size_t remaining = str_len - num_len;
    // ... interpret suffix (KiB/MB/...) and scale `value` ...
    *bytes = value;
    return remaining == 0 || /* recognized suffix */ true;
}

// Signed integer with strict full-consumption check
i64_t parse_int_strict(const char *val, u32_t val_len, bool *ok) {
    u32_t len = val_len;
    i64_t n = gc_common__parse_sign_number(val, &len);
    u32_t expected = len + ((val_len > 0 && val[0] == '-') ? 1 : 0);
    *ok = (val_len == expected);                        // reject trailing garbage
    return n;
}

// Duration string ("1h30m") -> microseconds; ISO-8601 date -> UTC epoch (microseconds)
i64_t duration_us, epoch_us;
if (!gc_duration__parse(val, val_len, &duration_us)) { /* invalid duration */ }
if (!gc_common__parse_date_iso8601(token_data, token_len, &epoch_us)) { /* invalid date */ }
```

**Parsing a JSON string into a typed slot.** `gc_json__parse` writes into a `(gc_slot_t, gc_type_t)` pair; `type_d` is the expected type descriptor (0 = untyped). The buffer must be NUL-terminated. On success, if the result is an object you own a mark and must `un_mark` after publishing it. This follows the `Http` JSON response handling.

```c
gc_buffer_t *buf = gc_machine__get_buffer(ctx);
gc_buffer__add_char(buf, '\0');                         // parser expects NUL-terminated input

gc_type_t res_type = gc_type_null;
gc_slot_t res = {.object = NULL};
if (gc_json__parse(ctx, buf->data, buf->size, &res, &res_type, type_desc)) {
    gc_machine__set_result(ctx, res, res_type);
    if (res_type == gc_type_object) {
        gc_object__un_mark(res.object, ctx);            // release the parse-time mark
    }
} else {
    gc_machine__set_runtime_error(ctx, "invalid JSON");
}
```

**Deep equality and slot hashing.** Both take the slot together with its `gc_type_t` and the active `gc_program_t *` (from `gc_machine__program(ctx)`). `gc__deep_equals` does structural comparison; `gc_slot__hash` is used to key node indexes. This is the `Assert::equals` pattern.

```c
const gc_program_t *prog = gc_machine__program(ctx);
gc_slot_t a = gc_my_fn_a(ctx);  gc_type_t a_t = gc_my_fn_a_type(ctx);
gc_slot_t b = gc_my_fn_b(ctx);  gc_type_t b_t = gc_my_fn_b_type(ctx);
if (!gc__deep_equals(a, a_t, b, b_t, prog)) {
    gc_machine__set_runtime_error(ctx, "values are not equal");
    return;
}

// Hash a key for a node-index lookup
u64_t encoded_key = gc_slot__hash(key, key_type, prog);
```

**Sorting with `gc_sort__piposort`.** Build a `gc_sort_slot_t[]` where `.value` is the raw `u64` sort key (read from `slot.u64`) and `.index` is the original position. `t` is the unified element type (`gc_type_undefined` if rows mix types). The trailing allocator supplies temporary workspace; allocate the array from the same allocator. This is the `Table::sort` / array-sort pattern.

```c
const u64_t n = self->rows;
gc_sort_slot_t *to_sort = gc_alloc__calloc(allocator, sizeof(gc_sort_slot_t) * n);
if (to_sort == NULL) { gc_machine__set_runtime_error(ctx, "out of memory"); return; }

u8_t type_and = UINT8_MAX, type_or = 0;
for (u32_t i = 0; i < n; i++) {
    gc_slot_t slot; gc_type_t slot_type;
    gc_table__get_cell(self, i, col, &slot, &slot_type);
    type_and &= slot_type;
    type_or  |= slot_type;
    to_sort[i].value = slot.u64;   // raw bits as the comparison key
    to_sort[i].index = i;          // remember where it came from
}
// uniform type => sort by that type, otherwise fall back to undefined
gc_type_t t = (type_and == type_or) ? (gc_type_t) type_and : gc_type_undefined;
gc_sort__piposort(to_sort, n, t, asc, allocator);
// to_sort[k].index now gives the original row to place at position k
```

**Morton (Z-order) interleaving for spatial keys.** `interleave64` packs two `u32_t` lattice coordinates into one `u64_t` key (used by geo-hash encoding); the matching `deinterleave64_*` / `gc_morton__*` helpers and per-axis `deinterleave64_Nd_xK` accessors unpack a single component without rebuilding the whole array.

```c
// Encode quantized lat/lon lattice coords into one geohash key
u32_t ilat = quantize_lat(latitude);
u32_t ilon = quantize_lon(longitude);
geo_t hash = (geo_t) interleave64(ilat, ilon);

// Decode: full pair or a single axis
u64_t xyhilo = deinterleave64(hash);      // x and y packed into one u64
u32_t ux = (u32_t) xyhilo;                // x component (low half)
u32_t uy = (u32_t) (xyhilo >> 32u);       // y component (high half)

// 3D / 4D: extract one axis directly (t3f / t4f decoding)
u32_t z = deinterleave64_3d_x2(key3d);
u16_t w = deinterleave64_4d_x3(key4d);
```

**License level gate.** `gc_license__level()` returns the current `gc_license_level_t`; compare against the enum to gate features.

```c
if (gc_license__level() < gc_license_level_pro) {
    gc_machine__set_runtime_error(ctx, "this feature requires a Pro license");
    return;
}
```


---
