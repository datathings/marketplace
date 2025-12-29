# Time Handling

## time

64-bit signed integer representing microseconds since 01/01/1970 UTC.

| Property | Value |
|----------|-------|
| Symbol | `time` |
| Min | 21/12/-290308 |
| Max | 10/01/294247 |
| Precision | 1 microsecond |

### Initialize Time

```gcl
var t1 = time::now();                                    // Current system time
var t2 = 5_time;                                         // 5 μs after epoch
var t3 = -1000000_time;                                  // 1s before epoch
var t4 = time::new(23, DurationUnit::hours);            // 23 hours after epoch
var t5 = time::new(1684163705, DurationUnit::seconds);  // POSIX epoch
var t6 = time::parse("07/06/2023 10:57:32", "%d/%m/%Y %H:%M:%S");
```

### Format Time

```gcl
println(t.format("%d/%m/%Y %H:%M:%S", TimeZone::"Europe/Luxembourg"));
```

### Time Operations

```gcl
if (t5 > t4) { }           // Compare
var d = t1 - t2;           // Subtract → duration
var newT = t1 + 10_s;      // Add duration

// Loop with time
for (var t = 0_time; t < 1000_time; t = t + 1_us) { }
```

## Date

Human-friendly representation for UI display. Resource-intensive to create.

### Create Date

```gcl
var date = t.toDate(null);                              // UTC
var dateLux = t.toDate(TimeZone::"Europe/Luxembourg");  // Specific TZ

var d1 = Date::from_time(time::now(), TimeZone::"Europe/Luxembourg");
var d2 = Date::parse("07/06/2023 10:57:32", "%d/%m/%Y %H:%M:%S");
```

### Convert Back to Time

```gcl
var t = date.to_time(null);  // Convert to UTC time

// Handle invalid dates (DST gaps)
var t = date.to_nearest_time(tz);  // Finds nearest valid time
```

## Duration

Span between two time points. Internal unit: microseconds.

```gcl
var d = t2 - t1;                                    // From time subtraction
var d1 = duration::new(5, DurationUnit::microseconds);
var d2 = duration::new(2, DurationUnit::days);
var d3 = duration::newf(1.5, DurationUnit::years);  // Float value

// Shorthand literals
var d4 = 5.6_s;    // 5.6 seconds
var d5 = 7_hour;   // 7 hours
```

### Duration Operations

```gcl
if (d4 > d5) { }    // Compare
var sum = 10_s + 10_s;   // Add
var diff = 10_s - 5_s;   // Subtract

println(d.to(DurationUnit::seconds));   // Integer seconds
println(d.tof(DurationUnit::hours));    // Fractional hours
```

## DurationUnit

Fixed microsecond values:

| Unit | Microseconds | Shorthand |
|------|--------------|-----------|
| microseconds | 1 | N_us |
| milliseconds | 1e3 | N_ms |
| seconds | 1e6 | N_s |
| minutes | 60e6 | N_m |
| hours | 3600e6 | N_h |
| days | 86400e6 | N_d |

## CalendarUnit

Respects Gregorian calendar and timezones:

- microseconds, seconds, minutes, hours
- days, months, years

### Calendar Functions

```gcl
// Shift time by calendar units
t.calendar_add(1, CalendarUnit::months, tz);  // Add 1 month

// Floor to unit boundary
t.calendar_floor(CalendarUnit::years, tz);    // Jan 1st 00:00:00

// Ceil to unit boundary
t.calendar_ceiling(CalendarUnit::years, tz);  // Dec 31st 23:59:59
```

## Date Format Specifiers

| Specifier | Meaning | Example |
|-----------|---------|---------|
| %d | Day zero-padded (01-31) | 22 |
| %D | Short MM/DD/YY | 07/30/09 |
| %e | Day space-padded | 22 |
| %H | Hour 24h (00-23) | 16 |
| %I | Hour 12h (01-12) | 08 |
| %m | Month (01-12) | 08 |
| %M | Minute (00-59) | 52 |
| %S | Second (00-61) | 06 |
| %s | Unix epoch seconds | 1455803239 |
| %y | Year 2-digit | 11 |
| %Y | Year 4-digit | 2016 |

```gcl
var d = Date::parse("31.01.2022-01:30:30", "%d.%m.%Y-%H:%M:%S");
```
