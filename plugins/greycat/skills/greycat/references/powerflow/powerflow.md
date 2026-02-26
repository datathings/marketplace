# Power Flow Analysis

Electrical power system analysis and simulation for GreyCat.

## Overview

The PowerFlow library provides computational tools for analyzing electrical power networks using power flow calculations (also known as load flow analysis). It enables simulation and analysis of steady-state electrical power systems, calculating voltages, currents, and power flows throughout a network.

Key features include:
- **Power network modeling** with buses, lines, loads, and external grids
- **Load flow computation** using Newton-Raphson or similar algorithms
- **Bus result analysis** including voltage magnitudes, angles, and currents
- **Line result analysis** including power flows, losses, and loading percentages
- **Configurable solver parameters** for convergence tolerance and iteration limits

This library is ideal for power system planning, grid optimization, renewable energy integration studies, distribution network analysis, and electrical engineering simulations.

## Installation

Add the PowerFlow library to your GreyCat project:

```gcl
@library("powerflow", "7.7.138-dev")
```

## Quick Start

### Simple 3-Bus System

```gcl
var network = PowerNetwork {
  tolerance: 0.0001,
  max_iteration: 100
};

// Configure network size
network.configure(
  nb_bus: 3,
  nb_lines: 2,
  nb_ext_grids: 1
);

// Create buses
network.createBus(0, 110.0); // Bus 0: 110 kV
network.createBus(1, 110.0); // Bus 1: 110 kV
network.createBus(2, 110.0); // Bus 2: 110 kV

// Create external grid (slack bus)
network.createExtGrid(0, 1.0); // Bus 0 at 1.0 pu voltage

// Create load at bus 2
network.createLoad(2, 50.0, 20.0); // 50 MW, 20 MVar

// Create transmission lines
network.createLine(
  0,        // line_id
  0, 1,     // from bus 0 to bus 1
  50.0,     // 50 km length
  0.12,     // 0.12 ohm/km resistance
  0.4,      // 0.4 ohm/km reactance
  10.0,     // 10 nF/km capacitance
  0.5       // 0.5 kA max current
);

network.createLine(1, 1, 2, 30.0, 0.12, 0.4, 10.0, 0.5);

// Run power flow calculation
network.compute();

// Get results
var busResult = network.getBusResult(2);
print("Bus 2 voltage: ${busResult.voltage} pu");
print("Bus 2 angle: ${busResult.angle_radians} rad");

var lineResult = network.getLineResult(0);
print("Line 0 loading: ${lineResult.loading_percent}%");
```

## Types

### PowerNetwork

Main type for power system modeling and analysis.

**Fields:**
- `tolerance: float?` (private) - Convergence tolerance for iterative solver (default: 1e-6)
- `max_iteration: int?` (private) - Maximum solver iterations (default: 100)

**Methods:**
- `configure(nb_bus: int, nb_lines: int, nb_ext_grids: int)` - Initialize network size
- `createBus(bus_id: int, vn_kv: float)` - Create a bus
- `createLoad(bus_id: int, p_mw: float, q_mvar: float)` - Create a load at a bus
- `createLine(line_id: int, from_bus_id: int, to_bus_id: int, lenght_km: float, r_ohm_per_km: float, x_ohm_per_km: float, c_n_f_per_km: float, max_i_ka: float)` - Create a transmission line (note: `lenght_km` is a known typo in the API)
- `createExtGrid(bus_id: int, vm_p_u: float)` - Create external grid (slack bus)
- `compute()` - Run power flow calculation
- `getBusResult(bus_id: int): PowerBusResult` - Get bus calculation results
- `getLineResult(line_id: int): PowerLineResult` - Get line calculation results
- `getCheckSum(): Array<float>` - Get checksum for validation

**Example:**

```gcl
var network = PowerNetwork {
  tolerance: 0.0001,      // Tighter convergence
  max_iteration: 50       // Limit iterations
};

network.configure(nb_bus: 10, nb_lines: 15, nb_ext_grids: 2);
```

### PowerBusResult

Results of power flow calculation for a bus.

**Fields:**
- `abs: float` - Voltage magnitude (absolute value)
- `angle_radians: float` - Voltage angle in radians
- `voltage: float` - Voltage in per-unit (pu)
- `voltage_img: float` - Imaginary part of voltage
- `current: float` - Current magnitude
- `current_img: float` - Imaginary part of current

**Example:**

```gcl
network.compute();

var result = network.getBusResult(5);
print("Voltage magnitude: ${result.voltage} pu");
print("Voltage angle: ${result.angle_radians * 180 / 3.14159} degrees");
print("Current: ${result.current} kA");
```

### PowerLineResult

Results of power flow calculation for a transmission line.

**Fields:**
- `p_from_mw: float` - Active power flow into line at "from" bus [MW]
- `q_from_mvar: float` - Reactive power flow into line at "from" bus [MVar]
- `p_to_mw: float` - Active power flow into line at "to" bus [MW]
- `q_to_mvar: float` - Reactive power flow into line at "to" bus [MVar]
- `pl_mw: float` - Active power losses [MW]
- `ql_mvar: float` - Reactive power consumption [MVar]
- `i_from_ka: float` - Current at "from" bus [kA]
- `i_to_ka: float` - Current at "to" bus [kA]
- `i_ka: float` - Maximum of i_from_ka and i_to_ka [kA]
- `vm_from_pu: float` - Voltage magnitude at "from" bus [pu]
- `vm_to_pu: float` - Voltage magnitude at "to" bus [pu]
- `va_from_radians: float` - Voltage angle at "from" bus [radians]
- `va_to_radians: float` - Voltage angle at "to" bus [radians]
- `loading_percent: float` - Line loading [%]

**Example:**

```gcl
var lineResult = network.getLineResult(3);

print("Power flow: ${lineResult.p_from_mw} MW");
print("Power losses: ${lineResult.pl_mw} MW");
print("Line loading: ${lineResult.loading_percent}%");

if (lineResult.loading_percent > 90.0) {
  print("WARNING: Line 3 is heavily loaded!");
}
```

## Methods

### configure()

Initializes the network with specified component counts.

**Signature:** `fn configure(nb_bus: int, nb_lines: int, nb_ext_grids: int)`

**Parameters:**
- `nb_bus: int` - Number of buses in the network
- `nb_lines: int` - Number of transmission lines
- `nb_ext_grids: int` - Number of external grids (slack buses)

**Note:** Must be called before creating any components.

**Example:**

```gcl
var network = PowerNetwork {};

// Network with 20 buses, 25 lines, 1 slack bus
network.configure(nb_bus: 20, nb_lines: 25, nb_ext_grids: 1);
```

### createBus()

Creates a bus (node) in the power network.

**Signature:** `fn createBus(bus_id: int, vn_kv: float)`

**Parameters:**
- `bus_id: int` - Unique bus identifier (0-based)
- `vn_kv: float` - Nominal voltage in kilovolts (e.g., 110.0, 220.0, 400.0)

**Example:**

```gcl
// Create buses at different voltage levels
network.createBus(0, 400.0);  // 400 kV transmission
network.createBus(1, 220.0);  // 220 kV sub-transmission
network.createBus(2, 110.0);  // 110 kV distribution
network.createBus(3, 33.0);   // 33 kV distribution
```

### createLoad()

Creates a load at a specified bus.

**Signature:** `fn createLoad(bus_id: int, p_mw: float, q_mvar: float)`

**Parameters:**
- `bus_id: int` - Bus where load is connected
- `p_mw: float` - Active power in megawatts
- `q_mvar: float` - Reactive power in megavolt-amperes reactive

**Example:**

```gcl
// Residential load
network.createLoad(5, 10.0, 3.0); // 10 MW, 3 MVar

// Industrial load
network.createLoad(8, 50.0, 20.0); // 50 MW, 20 MVar (0.93 power factor)

// Capacitive load (negative reactive power)
network.createLoad(10, 5.0, -2.0); // 5 MW, -2 MVar
```

### createLine()

Creates a transmission line between two buses.

**Signature:** `fn createLine(line_id: int, from_bus_id: int, to_bus_id: int, lenght_km: float, r_ohm_per_km: float, x_ohm_per_km: float, c_n_f_per_km: float, max_i_ka: float)`

**Parameters:**
- `line_id: int` - Unique line identifier
- `from_bus_id: int` - Starting bus
- `to_bus_id: int` - Ending bus
- `lenght_km: float` - Line length in kilometers (note: parameter name is a known typo in the API)
- `r_ohm_per_km: float` - Resistance in ohms per kilometer
- `x_ohm_per_km: float` - Reactance in ohms per kilometer
- `c_n_f_per_km: float` - Capacitance in nanofarads per kilometer
- `max_i_ka: float` - Maximum current rating in kiloamperes

**Example:**

```gcl
// 400 kV overhead line
network.createLine(
  0,          // line_id
  0, 1,       // from bus 0 to bus 1
  100.0,      // 100 km
  0.03,       // 0.03 ohm/km
  0.3,        // 0.3 ohm/km
  12.0,       // 12 nF/km
  2.0         // 2 kA max current
);

// 110 kV underground cable
network.createLine(
  1,          // line_id
  2, 3,       // from bus 2 to bus 3
  20.0,       // 20 km
  0.2,        // 0.2 ohm/km (higher resistance)
  0.12,       // 0.12 ohm/km (lower reactance)
  250.0,      // 250 nF/km (much higher capacitance)
  0.8         // 0.8 kA max current
);
```

### createExtGrid()

Creates an external grid connection (slack bus) that maintains voltage.

**Signature:** `fn createExtGrid(bus_id: int, vm_p_u: float)`

**Parameters:**
- `bus_id: int` - Bus where external grid connects
- `vm_p_u: float` - Voltage magnitude in per-unit (typically 1.0)

**Note:** At least one external grid (slack bus) is required for power flow calculation.

**Example:**

```gcl
// Standard slack bus at nominal voltage
network.createExtGrid(0, 1.0); // 1.0 pu = 100% of nominal

// Slack bus with voltage support
network.createExtGrid(0, 1.05); // 1.05 pu = 105% of nominal
```

### compute()

Executes the power flow calculation.

**Signature:** `fn compute()`

**Behavior:**
- Solves power flow equations iteratively
- Uses Newton-Raphson or similar method
- Respects `tolerance` and `max_iteration` settings
- Throws error if solution doesn't converge

**Example:**

```gcl
// Build network
network.configure(nb_bus: 5, nb_lines: 4, nb_ext_grids: 1);
// ... create buses, lines, loads, ext_grid ...

// Compute power flow
try {
  network.compute();
  print("Power flow converged");
} catch (e) {
  print("Power flow did not converge: ${e}");
}
```

### getBusResult()

Retrieves calculation results for a specific bus.

**Signature:** `fn getBusResult(bus_id: int): PowerBusResult`

**Note:** Must call `compute()` first.

**Example:**

```gcl
network.compute();

for (i in 0..5) {
  var result = network.getBusResult(i);
  print("Bus ${i}: ${result.voltage} pu, ${result.angle_radians} rad");
}
```

### getLineResult()

Retrieves calculation results for a specific line.

**Signature:** `fn getLineResult(line_id: int): PowerLineResult`

**Note:** Must call `compute()` first.

**Example:**

```gcl
network.compute();

var totalLosses = 0.0;

for (i in 0..4) {
  var result = network.getLineResult(i);
  totalLosses = totalLosses + result.pl_mw;
  print("Line ${i} loading: ${result.loading_percent}%");
}

print("Total system losses: ${totalLosses} MW");
```

### getCheckSum()

Returns validation checksums for the network.

**Signature:** `fn getCheckSum(): Array<float>`

**Returns:** Array of checksum values for validation purposes

**Example:**

```gcl
network.compute();
var checksum = network.getCheckSum();
print("Network checksum: ${checksum}");
```

## Common Use Cases

### Distribution Network Analysis

```gcl
var network = PowerNetwork {
  tolerance: 0.0001,
  max_iteration: 100
};

// 11 kV distribution network with 10 buses
network.configure(nb_bus: 10, nb_lines: 9, nb_ext_grids: 1);

// Substation (slack bus)
network.createBus(0, 11.0);
network.createExtGrid(0, 1.0);

// Distribution buses
for (i in 1..10) {
  network.createBus(i, 11.0);
}

// Radial feeder lines
for (i in 0..9) {
  network.createLine(
    i,
    i, i + 1,
    2.0,      // 2 km between buses
    0.32,     // Typical 11 kV line resistance
    0.35,     // Typical 11 kV line reactance
    11.0,     // Typical 11 kV line capacitance
    0.4       // 400 A rating
  );
}

// Loads at each bus
network.createLoad(1, 0.5, 0.2);
network.createLoad(2, 0.8, 0.3);
network.createLoad(3, 1.2, 0.5);
// ... etc

network.compute();

// Check voltage profile
for (i in 0..10) {
  var result = network.getBusResult(i);
  if (result.voltage < 0.95) {
    print("WARNING: Bus ${i} voltage too low: ${result.voltage} pu");
  }
}
```

### Transmission Line Loading Analysis

```gcl
var network = PowerNetwork {};
network.configure(nb_bus: 4, nb_lines: 4, nb_ext_grids: 1);

// Create buses
network.createBus(0, 400.0);
network.createBus(1, 400.0);
network.createBus(2, 400.0);
network.createBus(3, 400.0);

// Slack bus
network.createExtGrid(0, 1.0);

// Mesh network
network.createLine(0, 0, 1, 100.0, 0.03, 0.3, 12.0, 2.0);
network.createLine(1, 1, 2, 100.0, 0.03, 0.3, 12.0, 2.0);
network.createLine(2, 2, 3, 100.0, 0.03, 0.3, 12.0, 2.0);
network.createLine(3, 3, 0, 100.0, 0.03, 0.3, 12.0, 2.0);

// Loads
network.createLoad(1, 200.0, 50.0);
network.createLoad(3, 200.0, 50.0);

network.compute();

// Analyze line loading
var overloadedLines = Array<int>{};

for (i in 0..4) {
  var result = network.getLineResult(i);

  if (result.loading_percent > 100.0) {
    overloadedLines.add(i);
    print("ALERT: Line ${i} overloaded at ${result.loading_percent}%");
  } else if (result.loading_percent > 80.0) {
    print("WARNING: Line ${i} heavily loaded at ${result.loading_percent}%");
  }
}

if (overloadedLines.size() > 0) {
  print("Network requires reinforcement!");
}
```

### Renewable Integration Study

```gcl
var network = PowerNetwork {};
network.configure(nb_bus: 6, nb_lines: 5, nb_ext_grids: 1);

// Create network
for (i in 0..6) {
  network.createBus(i, 110.0);
}

network.createExtGrid(0, 1.0);

// Transmission lines
network.createLine(0, 0, 1, 50.0, 0.12, 0.4, 10.0, 0.6);
network.createLine(1, 1, 2, 50.0, 0.12, 0.4, 10.0, 0.6);
network.createLine(2, 2, 3, 50.0, 0.12, 0.4, 10.0, 0.6);
network.createLine(3, 1, 4, 30.0, 0.12, 0.4, 10.0, 0.6);
network.createLine(4, 4, 5, 30.0, 0.12, 0.4, 10.0, 0.6);

// Base load scenario
network.createLoad(3, 40.0, 15.0);
network.createLoad(5, 30.0, 10.0);

// Scenario 1: Without solar
network.compute();
var result1 = network.getLineResult(0);
var basePower = result1.p_from_mw;

// Scenario 2: With solar generation (negative load)
network.createLoad(2, -20.0, -5.0); // 20 MW solar farm
network.compute();
var result2 = network.getLineResult(0);
var withSolarPower = result2.p_from_mw;

print("Power from grid - Base: ${basePower} MW");
print("Power from grid - With solar: ${withSolarPower} MW");
print("Grid power reduction: ${basePower - withSolarPower} MW");
```

### N-1 Contingency Analysis

```gcl
fn analyzeContingency(network: PowerNetwork, lineToRemove: int) {
  // Simulate line outage by setting very high impedance
  // (Proper implementation would require network reconfiguration)

  network.compute();

  var overloaded = false;
  for (i in 0..network.nb_lines) {
    if (i == lineToRemove) continue;

    var result = network.getLineResult(i);
    if (result.loading_percent > 100.0) {
      print("Line ${i} overloaded: ${result.loading_percent}%");
      overloaded = true;
    }
  }

  return overloaded;
}

// Test each line outage
for (lineId in 0..nb_lines) {
  print("Testing outage of line ${lineId}");
  var failed = analyzeContingency(network, lineId);
  if (failed) {
    print("CRITICAL: System unstable with line ${lineId} out");
  }
}
```

## Best Practices

### Network Modeling

- **Use realistic parameters**: Get line impedances from manufacturer data or standards
- **Proper voltage levels**: Ensure bus nominal voltages match connected equipment
- **Include all loads**: Missing loads lead to inaccurate results
- **Model reactive power**: Both P and Q are important for accurate voltage calculations

```gcl
// Good: Complete model
network.createLoad(5, 50.0, 20.0); // P and Q specified

// Bad: Ignoring reactive power
// network.createLoad(5, 50.0, 0.0); // Unrealistic
```

### Convergence Issues

- **Start with slack bus**: At least one external grid is required
- **Check topology**: Ensure network is connected (no isolated buses)
- **Adjust tolerance**: Tighter tolerance may fail to converge
- **Increase iterations**: Complex networks may need more iterations

```gcl
// For difficult cases
var network = PowerNetwork {
  tolerance: 0.001,     // Relax tolerance
  max_iteration: 200    // Allow more iterations
};
```

### Units and Conventions

- **Voltage**: kV (kilovolts) for buses, pu (per-unit) for results
- **Power**: MW (megawatts) and MVar (megavolt-amperes reactive)
- **Current**: kA (kiloamperes)
- **Impedance**: Ohms per km
- **Capacitance**: Nanofarads per km
- **Angles**: Radians (convert to degrees: `angle * 180 / 3.14159`)

### Result Interpretation

- **Voltage limits**: Typically 0.95-1.05 pu is acceptable
- **Line loading**: Above 80% indicates need for reinforcement
- **Power losses**: Sum `pl_mw` across all lines for total losses
- **Voltage angles**: Large angles indicate long transmission distances or high impedance

```gcl
network.compute();

var busResult = network.getBusResult(10);

// Check voltage
if (busResult.voltage < 0.95 || busResult.voltage > 1.05) {
  print("Voltage violation at bus 10");
}

// Convert angle to degrees
var angleDegrees = busResult.angle_radians * 180.0 / 3.14159;
print("Voltage angle: ${angleDegrees} degrees");
```

### Performance

- **Pre-allocate with configure()**: Specify exact component counts
- **Reuse network objects**: Recreate only when topology changes
- **Batch result queries**: Get all results after single compute()

```gcl
// Efficient
network.compute();
var results = Array<PowerBusResult>{};
for (i in 0..nb_buses) {
  results.add(network.getBusResult(i));
}

// Inefficient
for (i in 0..nb_buses) {
  network.compute(); // Don't recompute!
  var result = network.getBusResult(i);
}
```

### Gotchas

- **Bus/line IDs are 0-based**: First bus is 0, not 1
- **Must configure before creating**: Call `configure()` first
- **Compute before results**: `getBusResult()` requires prior `compute()`
- **Line parameters are per km**: Total impedance = parameter * length
- **Capacitance units**: nF/km, not μF/km
- **Convergence failures**: May indicate unrealistic network configuration
- **Sign conventions**: Positive P/Q = consumption, negative = generation

### Validation

Always validate results:

```gcl
network.compute();

// Check power balance
var totalGeneration = 0.0;
var totalLoad = 0.0;
var totalLosses = 0.0;

for (i in 0..nb_lines) {
  var lineResult = network.getLineResult(i);
  totalLosses = totalLosses + lineResult.pl_mw;
}

// Generation = Load + Losses (approximately)
var error = totalGeneration - totalLoad - totalLosses;
if (abs(error) > 1.0) {
  print("WARNING: Power balance error: ${error} MW");
}
```
