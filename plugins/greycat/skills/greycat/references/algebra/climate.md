# Climate Modeling

Climate data modeling utilities, currently providing UTCI (Universal Thermal Climate Index) calculation for thermal comfort assessment.

## Overview

The Climate module provides specialized algorithms for climate and environmental data analysis. Currently includes:

- **UTCI**: Universal Thermal Climate Index for human thermal comfort assessment

## UTCI - Universal Thermal Climate Index

### Function

```typescript
native fn utci(
  temp_outside: float,  // Outside temperature (°C)
  wind_avg: float,      // Average wind speed (m/s)
  temp_rad: float,      // Mean radiant temperature (°C)
  hygro_per: float      // Relative humidity (%)
): float                // UTCI value (°C)
```

### Description

The Universal Thermal Climate Index (UTCI) is a comprehensive thermal comfort index that combines:
- Air temperature
- Wind speed
- Radiant temperature
- Humidity

It provides an equivalent temperature that represents how the human body perceives the thermal environment.

### Parameters

**temp_outside** (°C)
- Air temperature measured in shade
- Typical range: -50°C to 50°C

**wind_avg** (m/s)
- Wind speed at 10m height
- Adjusted for body height in calculation
- Typical range: 0 to 30 m/s

**temp_rad** (°C)
- Mean radiant temperature
- Average temperature of surrounding surfaces
- Accounts for solar radiation and thermal radiation from surroundings
- Often approximates air temperature in shade

**hygro_per** (%)
- Relative humidity
- Range: 0 to 100

### Returns

UTCI equivalent temperature (°C) representing perceived thermal comfort:

| UTCI Range (°C) | Thermal Stress Category |
|-----------------|------------------------|
| > +46 | Extreme heat stress |
| +38 to +46 | Very strong heat stress |
| +32 to +38 | Strong heat stress |
| +26 to +32 | Moderate heat stress |
| +9 to +26 | No thermal stress |
| 0 to +9 | Slight cold stress |
| -13 to 0 | Moderate cold stress |
| -27 to -13 | Strong cold stress |
| -40 to -27 | Very strong cold stress |
| < -40 | Extreme cold stress |

### Examples

#### Basic Usage

```typescript
// Summer day in Paris
var utci = utci(
  temp_outside: 30.0,   // 30°C air temperature
  wind_avg: 3.0,        // 3 m/s wind
  temp_rad: 35.0,       // 35°C mean radiant (sunny)
  hygro_per: 60.0       // 60% humidity
);

println("UTCI: ${utci}°C"); // Output: ~34-36°C (Strong heat stress)
```

#### Winter Conditions

```typescript
// Cold winter day
var utci = utci(
  temp_outside: -5.0,   // -5°C air temperature
  wind_avg: 10.0,       // 10 m/s wind (strong wind)
  temp_rad: -5.0,       // -5°C radiant (cloudy)
  hygro_per: 80.0       // 80% humidity
);

println("UTCI: ${utci}°C"); // Output: ~-15°C (Strong cold stress)
// Wind chill makes it feel much colder
```

#### Indoor vs Outdoor

```typescript
// Shade (indoor-like)
var utci_shade = utci(25.0, 1.0, 25.0, 50.0);

// Direct sun
var utci_sun = utci(25.0, 1.0, 40.0, 50.0);

println("Shade feels like: ${utci_shade}°C");
println("Sun feels like: ${utci_sun}°C");
// Radiant temperature significantly affects perceived comfort
```

### Time Series Analysis

#### Daily Comfort Profile

```typescript
// Calculate UTCI for each hour of the day
var hourly_utci = nodeTime<float> {};

for (hour in 0..24) {
  var t = time::new(2024, 7, 15, hour, 0, 0);

  // Get weather data for this hour
  var temp = temperature_data.getAt(t);
  var wind = wind_data.getAt(t);
  var humidity = humidity_data.getAt(t);

  // Estimate radiant temperature
  // (Simple model: +10°C in sun hours, equal to air temp otherwise)
  var is_sunny = hour >= 6 && hour <= 18;
  var rad_temp = is_sunny ? temp + 10.0 : temp;

  var utci_val = utci(temp, wind, rad_temp, humidity);
  hourly_utci.setAt(t, utci_val);
}

// Find peak discomfort
var max_utci = -Infinity;
var max_time: time? = null;
for (t, val in hourly_utci) {
  if (val > max_utci) {
    max_utci = val;
    max_time = t;
  }
}

println("Peak thermal stress at ${max_time}: ${max_utci}°C UTCI");
```

#### Seasonal Comparison

```typescript
// Compare thermal comfort across seasons
var seasons = Map<String, Array<float>> {};
seasons.set("Spring", Array<float> {});
seasons.set("Summer", Array<float> {});
seasons.set("Fall", Array<float> {});
seasons.set("Winter", Array<float> {});

for (t, temp in yearly_temperature) {
  var date = Date { time: t, tz: TimeZone::UTC };
  var month = date.month();

  var wind = wind_data.getAt(t) ?? 2.0;
  var humidity = humidity_data.getAt(t) ?? 50.0;
  var rad_temp = temp; // Simplified

  var utci_val = utci(temp, wind, rad_temp, humidity);

  if (month >= 3 && month <= 5) {
    seasons.get("Spring")!!.add(utci_val);
  } else if (month >= 6 && month <= 8) {
    seasons.get("Summer")!!.add(utci_val);
  } else if (month >= 9 && month <= 11) {
    seasons.get("Fall")!!.add(utci_val);
  } else {
    seasons.get("Winter")!!.add(utci_val);
  }
}

// Average comfort by season
for (season, values in seasons) {
  var avg = sum(values) / values.size();
  println("${season} avg UTCI: ${avg}°C");
}
```

### Climate Analysis Applications

#### 1. Heat Wave Detection

```typescript
// Identify dangerous heat conditions
var heat_wave_threshold = 38.0; // Strong heat stress
var consecutive_hours = 0;
var heat_wave_start: time? = null;

for (t, temp in temperature_data) {
  var utci_val = utci(
    temp,
    wind_data.getAt(t) ?? 2.0,
    temp + 5.0, // Estimate radiant
    humidity_data.getAt(t) ?? 50.0
  );

  if (utci_val > heat_wave_threshold) {
    if (heat_wave_start == null) {
      heat_wave_start = t;
    }
    consecutive_hours++;
  } else {
    if (consecutive_hours >= 6) {
      println("Heat wave from ${heat_wave_start} (${consecutive_hours}h)");
    }
    consecutive_hours = 0;
    heat_wave_start = null;
  }
}
```

#### 2. Outdoor Activity Scheduling

```typescript
// Find optimal times for outdoor activities
fn isSafeForActivity(utci_val: float, activity: String): bool {
  if (activity == "running") {
    return utci_val >= 0.0 && utci_val <= 26.0; // No thermal stress
  } else if (activity == "construction") {
    return utci_val >= 5.0 && utci_val <= 32.0; // Slight cold to moderate heat
  } else if (activity == "tourism") {
    return utci_val >= 9.0 && utci_val <= 26.0; // Comfort range
  }
  return false;
}

var safe_hours = Array<time> {};
for (t, temp in forecast_data) {
  var utci_val = utci(temp, wind_forecast.getAt(t), temp, humidity_forecast.getAt(t));

  if (isSafeForActivity(utci_val, "running")) {
    safe_hours.add(t);
  }
}

println("Safe running hours: ${safe_hours.size()}");
```

#### 3. Building Climate Control

```typescript
// Optimize HVAC based on UTCI
for (t, indoor_temp in indoor_temperature) {
  var outdoor_temp = outdoor_temperature.getAt(t);
  var outdoor_wind = wind_data.getAt(t);
  var outdoor_humidity = humidity_data.getAt(t);

  // Calculate outdoor UTCI
  var outdoor_utci = utci(outdoor_temp, outdoor_wind, outdoor_temp, outdoor_humidity);

  // Calculate indoor UTCI (low wind indoors)
  var indoor_utci = utci(indoor_temp, 0.1, indoor_temp, indoor_humidity.getAt(t));

  // Adjust HVAC to maintain comfort (UTCI 18-26°C)
  if (indoor_utci > 26.0) {
    // Cool down
    hvac_setpoint.setAt(t, indoor_temp - 1.0);
  } else if (indoor_utci < 18.0) {
    // Heat up
    hvac_setpoint.setAt(t, indoor_temp + 1.0);
  }
}
```

#### 4. Climate Zone Classification

```typescript
// Classify locations by thermal comfort
fn classifyClimate(yearly_utci_data: nodeTime<float>): String {
  var hot_days = 0;
  var cold_days = 0;
  var comfortable_days = 0;

  for (_, utci_val in yearly_utci_data) {
    if (utci_val > 32.0) {
      hot_days++;
    } else if (utci_val < 0.0) {
      cold_days++;
    } else if (utci_val >= 9.0 && utci_val <= 26.0) {
      comfortable_days++;
    }
  }

  var total_days = yearly_utci_data.size();

  if (comfortable_days > total_days * 0.7) {
    return "Temperate";
  } else if (hot_days > total_days * 0.5) {
    return "Hot";
  } else if (cold_days > total_days * 0.5) {
    return "Cold";
  } else {
    return "Variable";
  }
}

var classification = classifyClimate(utci_time_series);
println("Climate classification: ${classification}");
```

### Validation and Constraints

```typescript
// UTCI is valid for specific ranges
fn validateUTCIInputs(temp: float, wind: float, rad_temp: float, humidity: float): bool {
  if (temp < -50.0 || temp > 50.0) {
    warn("Temperature out of range: ${temp}°C");
    return false;
  }

  if (wind < 0.0 || wind > 30.0) {
    warn("Wind speed out of range: ${wind} m/s");
    return false;
  }

  if (humidity < 0.0 || humidity > 100.0) {
    warn("Humidity out of range: ${humidity}%");
    return false;
  }

  return true;
}

// Safe UTCI calculation with validation
fn safeUTCI(temp: float, wind: float, rad_temp: float, humidity: float): float? {
  if (validateUTCIInputs(temp, wind, rad_temp, humidity)) {
    return utci(temp, wind, rad_temp, humidity);
  }
  return null;
}
```

### Estimating Mean Radiant Temperature

When direct measurements are unavailable:

```typescript
// Simple model for outdoor conditions
fn estimateRadiantTemp(air_temp: float, solar_radiation: float?, cloud_cover: float?): float {
  if (solar_radiation == null || cloud_cover == null) {
    // No data: assume equal to air temp
    return air_temp;
  }

  // Estimate based on solar radiation
  // Typical: +5 to +15°C above air temp in direct sun
  var radiation_factor = (1.0 - cloud_cover) * (solar_radiation / 1000.0);
  var rad_temp = air_temp + radiation_factor * 15.0;

  return rad_temp;
}

// Usage
var rad_temp = estimateRadiantTemp(
  air_temp: 25.0,
  solar_radiation: 800.0,  // W/m²
  cloud_cover: 0.3         // 30% cloud cover
);
// Result: ~28°C
```

### Integration with Weather APIs

```typescript
// Example: Process weather API data
type WeatherData {
  time: time;
  temperature: float;      // °C
  wind_speed: float;       // m/s
  humidity: float;         // %
  solar_radiation: float?; // W/m²
  cloud_cover: float?;     // 0-1
}

fn calculateUTCIFromWeather(weather: WeatherData): float {
  // Estimate radiant temperature
  var rad_temp = estimateRadiantTemp(
    weather.temperature,
    weather.solar_radiation,
    weather.cloud_cover
  );

  return utci(
    weather.temperature,
    weather.wind_speed,
    rad_temp,
    weather.humidity
  );
}

// Process forecast
var comfort_forecast = nodeTime<float> {};
for (weather in forecast) {
  var utci_val = calculateUTCIFromWeather(weather);
  comfort_forecast.setAt(weather.time, utci_val);
}
```

---

## Best Practices

### Data Quality
- Ensure wind speed is at standard 10m height
- Use radiant temperature measurements when available
- For estimates, account for solar angle and surface reflectance

### Interpretation
- UTCI represents thermal perception, not actual temperature
- Consider individual factors (age, clothing, activity) not captured by UTCI
- Use category thresholds, not exact values, for decision-making

### Application Context
- Outdoor thermal comfort assessment
- Heat/cold stress risk evaluation
- Urban planning and building design
- Health impact studies
- Climate change impact analysis

### Limitations
- Assumes standard clothing and metabolic rates
- Most accurate for outdoor environments
- May need calibration for extreme climates
- Does not account for acclimatization

---

## References

- UTCI official documentation: http://www.utci.org
- Blazejczyk et al. (2013): "An introduction to the Universal Thermal Climate Index (UTCI)"
- ISO 7933:2004: Ergonomics of the thermal environment

---

## Future Additions

The Climate module may expand to include:
- Heat Index and Humidex calculations
- Wet Bulb Globe Temperature (WBGT)
- Predicted Mean Vote (PMV) / Predicted Percentage Dissatisfied (PPD)
- Climate zone classification algorithms
- Degree-day calculations (heating/cooling)
- Thermal comfort time series analysis utilities

## See Also

- [Machine Learning (ml.md)](ml.md) - Statistical analysis of climate data
- [Transforms (transforms.md)](transforms.md) - Frequency analysis of climate patterns
- [README.md](README.md) - Library overview
