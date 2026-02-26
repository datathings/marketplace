# PowerGrid

Power flow analysis and electrical network simulation for GreyCat.

## Overview

The PowerGrid library provides a Newton-Raphson-based power flow solver for analyzing electrical networks. It models buses, transmission lines, loads, and external grids to compute voltage, current, and power flow distributions across a network.

Key features include:
- **Newton-Raphson solver** for steady-state power flow analysis
- **Bus modeling** with voltage magnitude, phase angle, and current results
- **Transmission line modeling** with impedance, capacitance, and thermal rating parameters
- **Load modeling** with active and reactive power consumption
- **External grid connections** acting as slack/reference buses
- **Per-line and per-bus results** including power flows, losses, loading percentages, and convergence metrics

This library is ideal for grid planning, contingency analysis, voltage stability assessment, and building power systems simulation tools.

## Installation

Add the PowerGrid library to your GreyCat project:

```gcl
@library("powergrid", "7.7.138-dev")
```

## Quick Start

### Simple Two-Bus Network

```gcl
// Create a network with 2 buses, 1 line, 1 external grid, and 1 load
var network = PowerNetwork {};
network.configure(2, 1, 1, 1);

// Create buses at 20 kV
network.createBus(0, 20.0);
network.createBus(1, 20.0);

// Connect an external grid (slack bus) at bus 0
network.createExtGrid(0, 1.0);

// Add a load at bus 1: 10 MW active, 5 MVar reactive
network.createLoad(1, 10.0, 5.0);

// Create a transmission line from bus 0 to bus 1
network.createLine(0, 0, 1, 10.0, 0.1, 0.4, 9.7, 0.5);

// Run power flow analysis
network.compute();

// Read results
var busResult = network.getBusResult(0);
print("Bus 0 voltage: ${busResult.abs} pu, angle: ${busResult.angle_radians} rad");

var lineResult = network.getLineResult(0);
print("Line 0 loading: ${lineResult.loading_percent}%");
```

### Three-Bus Network with Multiple Lines

```gcl
var network = PowerNetwork {};
network.configure(3, 2, 1, 1);

// Create three buses at 110 kV
network.createBus(0, 110.0);
network.createBus(1, 110.0);
network.createBus(2, 110.0);

// External grid at bus 0 with 1.0 per-unit voltage
network.createExtGrid(0, 1.0);

// Load at bus 2
network.createLoad(2, 50.0, 20.0);

// Two transmission lines: bus 0 -> bus 1 and bus 1 -> bus 2
network.createLine(0, 0, 1, 25.0, 0.06, 0.3, 11.0, 0.8);
network.createLine(1, 1, 2, 15.0, 0.08, 0.35, 10.5, 0.6);

network.compute();

// Inspect all bus voltages
for (var i = 0; i < 3; i++) {
  var res = network.getBusResult(i);
  print("Bus ${i}: voltage=${res.abs}, angle=${res.angle_radians} rad");
}

// Inspect line results
for (var i = 0; i < 2; i++) {
  var res = network.getLineResult(i);
  print("Line ${i}: P_from=${res.p_from_mw} MW, losses=${res.pl_mw} MW, loading=${res.loading_percent}%");
}
```

## Types

### PowerNetwork

Main electrical network model and power flow solver. Build a network by configuring its size, then creating buses, lines, loads, and external grids. Call `compute()` to run the Newton-Raphson power flow analysis, then retrieve results.

**Fields:**
- `tolerance: float?` - Convergence tolerance for the Newton-Raphson solver (private)
- `max_iteration: int?` - Maximum number of iterations for the Newton-Raphson solver (private)

**Methods:**
- `configure(nb_bus, nb_lines, nb_ext_grids, nb_loads)` - Set network dimensions
- `createBus(bus_id, vn_kv)` - Create a bus node
- `createLoad(bus_id, p_mw, q_mvar)` - Create a load at a bus
- `createLine(line_id, from_bus_id, to_bus_id, lenght_km, r_ohm_per_km, x_ohm_per_km, c_n_f_per_km, max_i_ka)` - Create a transmission line
- `createExtGrid(bus_id, vm_p_u)` - Create an external grid connection
- `compute()` - Run power flow analysis
- `getBusResult(bus_id): PowerBusResult` - Get results for a bus
- `getLineResult(line_id): PowerLineResult` - Get results for a line
- `getCheckSum(): Array<float>` - Get solver convergence metrics

**Example:**

```gcl
var network = PowerNetwork {};
network.configure(2, 1, 1, 1);

network.createBus(0, 20.0);
network.createBus(1, 20.0);
network.createExtGrid(0, 1.0);
network.createLoad(1, 10.0, 5.0);
network.createLine(0, 0, 1, 10.0, 0.1, 0.4, 9.7, 0.5);

network.compute();
```

### PowerBusResult

Power flow computation result for a bus in the electrical network. Contains voltage, current, and phase angle information after power flow analysis.

**Fields:**
- `abs: float` - Voltage magnitude (absolute value)
- `angle_radians: float` - Voltage phase angle in radians
- `voltage: float` - Real component of voltage
- `voltage_img: float` - Imaginary component of voltage
- `current: float` - Real component of current
- `current_img: float` - Imaginary component of current

**Example:**

```gcl
var result = network.getBusResult(0);

print("Voltage magnitude: ${result.abs}");
print("Phase angle: ${result.angle_radians} rad");
print("Voltage (real): ${result.voltage}");
print("Voltage (imag): ${result.voltage_img}");
print("Current (real): ${result.current}");
print("Current (imag): ${result.current_img}");
```

### PowerLineResult

Power flow computation result for a transmission line. Contains power flows, losses, currents, and voltages at both ends of the line.

**Fields:**
- `p_from_mw: float` - Active power flow into the line at "from" bus [MW]
- `q_from_mvar: float` - Reactive power flow into the line at "from" bus [MVar]
- `p_to_mw: float` - Active power flow into the line at "to" bus [MW]
- `q_to_mvar: float` - Reactive power flow into the line at "to" bus [MVar]
- `pl_mw: float` - Active power losses of the line [MW]
- `ql_mvar: float` - Reactive power consumption of the line [MVar]
- `i_from_ka: float` - Current at from bus [kA]
- `i_to_ka: float` - Current at to bus [kA]
- `i_ka: float` - Maximum of `i_from_ka` and `i_to_ka` [kA]
- `vm_from_pu: float` - Voltage magnitude at from bus in per-unit
- `vm_to_pu: float` - Voltage magnitude at to bus in per-unit
- `va_from_radians: float` - Voltage angle at from bus [radians]
- `va_to_radians: float` - Voltage angle at to bus [radians]
- `loading_percent: float` - Line loading [%]

**Example:**

```gcl
var result = network.getLineResult(0);

print("Power from bus: ${result.p_from_mw} MW, ${result.q_from_mvar} MVar");
print("Power to bus: ${result.p_to_mw} MW, ${result.q_to_mvar} MVar");
print("Losses: ${result.pl_mw} MW (active), ${result.ql_mvar} MVar (reactive)");
print("Current: ${result.i_ka} kA (max of from=${result.i_from_ka}, to=${result.i_to_ka})");
print("Loading: ${result.loading_percent}%");
```

## Methods

### configure()

Configure the network dimensions before adding any components. Must be called first.

**Signature:** `fn configure(nb_bus: int, nb_lines: int, nb_ext_grids: int, nb_loads: int)`

**Parameters:**
- `nb_bus: int` - Total number of buses in the network
- `nb_lines: int` - Total number of transmission lines
- `nb_ext_grids: int` - Total number of external grid connections (slack buses)
- `nb_loads: int` - Total number of loads in the network

**Example:**

```gcl
var network = PowerNetwork {};

// Network with 5 buses, 4 lines, 1 external grid, and 3 loads
network.configure(5, 4, 1, 3);
```

### createBus()

Create a bus (node) in the network.

**Signature:** `fn createBus(bus_id: int, vn_kv: float)`

**Parameters:**
- `bus_id: int` - Unique identifier for this bus (0 to `nb_bus - 1`)
- `vn_kv: float` - Nominal voltage level in kilovolts (kV)

**Example:**

```gcl
// Create buses at different voltage levels
network.createBus(0, 110.0);  // 110 kV high-voltage bus
network.createBus(1, 20.0);   // 20 kV medium-voltage bus
network.createBus(2, 20.0);   // 20 kV medium-voltage bus
```

### createLoad()

Create a load connected to a bus.

**Signature:** `fn createLoad(bus_id: int, p_mw: float, q_mvar: float)`

**Parameters:**
- `bus_id: int` - The bus where this load is connected
- `p_mw: float` - Active power consumption in megawatts (MW)
- `q_mvar: float` - Reactive power consumption in megavolt-amperes reactive (MVar)

**Example:**

```gcl
// Industrial load: 50 MW active, 20 MVar reactive
network.createLoad(1, 50.0, 20.0);

// Residential load: 10 MW active, 3 MVar reactive
network.createLoad(2, 10.0, 3.0);
```

### createLine()

Create a transmission line between two buses.

**Signature:** `fn createLine(line_id: int, from_bus_id: int, to_bus_id: int, lenght_km: float, r_ohm_per_km: float, x_ohm_per_km: float, c_n_f_per_km: float, max_i_ka: float)`

**Parameters:**
- `line_id: int` - Unique identifier for this line (0 to `nb_lines - 1`)
- `from_bus_id: int` - Starting bus ID
- `to_bus_id: int` - Ending bus ID
- `lenght_km: float` - Line length in kilometers (note: parameter name uses `lenght_km`, not `length_km`)
- `r_ohm_per_km: float` - Resistance in ohms per kilometer
- `x_ohm_per_km: float` - Reactance in ohms per kilometer
- `c_n_f_per_km: float` - Capacitance in nanofarads per kilometer
- `max_i_ka: float` - Maximum current rating in kiloamperes (kA)

**Example:**

```gcl
// Overhead line: 25 km, R=0.06 ohm/km, X=0.3 ohm/km, C=11 nF/km, max 0.8 kA
network.createLine(0, 0, 1, 25.0, 0.06, 0.3, 11.0, 0.8);

// Cable line: 5 km, R=0.12 ohm/km, X=0.08 ohm/km, C=240 nF/km, max 0.4 kA
network.createLine(1, 1, 2, 5.0, 0.12, 0.08, 240.0, 0.4);
```

### createExtGrid()

Create an external grid connection (slack bus). The external grid acts as a reference bus with fixed voltage magnitude.

**Signature:** `fn createExtGrid(bus_id: int, vm_p_u: float)`

**Parameters:**
- `bus_id: int` - The bus where the external grid is connected
- `vm_p_u: float` - Voltage magnitude in per-unit (typically 1.0)

**Example:**

```gcl
// Standard external grid at nominal voltage
network.createExtGrid(0, 1.0);

// External grid with slightly elevated voltage (1.02 per-unit)
network.createExtGrid(0, 1.02);
```

### compute()

Run the Newton-Raphson power flow analysis. Solves the power flow equations to compute voltages, currents, and power flows throughout the network. Results can be retrieved via `getBusResult()` and `getLineResult()`.

**Signature:** `fn compute()`

**Behavior:**
- Iterates until convergence within `tolerance` or `max_iteration` is reached
- Throws an error if the solver fails to converge

**Example:**

```gcl
network.compute();

// With error handling
try {
  network.compute();
  print("Power flow converged");
} catch (e) {
  print("Power flow failed to converge: ${e}");
}
```

### getBusResult()

Get power flow results for a specific bus after calling `compute()`.

**Signature:** `fn getBusResult(bus_id: int): PowerBusResult`

**Parameters:**
- `bus_id: int` - The bus ID to query (0 to `nb_bus - 1`)

**Returns:** `PowerBusResult` containing voltage, current, and phase angle data

**Example:**

```gcl
var result = network.getBusResult(0);
print("Bus 0: |V|=${result.abs}, angle=${result.angle_radians} rad");
print("  V = ${result.voltage} + j${result.voltage_img}");
print("  I = ${result.current} + j${result.current_img}");
```

### getLineResult()

Get power flow results for a specific transmission line after calling `compute()`.

**Signature:** `fn getLineResult(line_id: int): PowerLineResult`

**Parameters:**
- `line_id: int` - The line ID to query (0 to `nb_lines - 1`)

**Returns:** `PowerLineResult` containing power flows, losses, currents, and loading

**Example:**

```gcl
var result = network.getLineResult(0);
print("Line 0:");
print("  From: P=${result.p_from_mw} MW, Q=${result.q_from_mvar} MVar, I=${result.i_from_ka} kA");
print("  To:   P=${result.p_to_mw} MW, Q=${result.q_to_mvar} MVar, I=${result.i_to_ka} kA");
print("  Losses: ${result.pl_mw} MW, ${result.ql_mvar} MVar");
print("  Loading: ${result.loading_percent}%");
```

### getCheckSum()

Get numerical checksum for validation purposes.

**Signature:** `fn getCheckSum(): Array<float>`

**Returns:** Array of floats representing solver convergence metrics

**Example:**

```gcl
var checksum = network.getCheckSum();
for (var i = 0; i < checksum.size(); i++) {
  print("Checksum[${i}]: ${checksum[i]}");
}
```

## Common Use Cases

### Basic Load Flow Study

```gcl
var network = PowerNetwork {};
network.configure(3, 2, 1, 2);

// Set up a radial network: ExtGrid -> Bus0 -> Bus1 -> Bus2
network.createBus(0, 110.0);
network.createBus(1, 110.0);
network.createBus(2, 110.0);

network.createExtGrid(0, 1.0);

network.createLoad(1, 30.0, 10.0);
network.createLoad(2, 20.0, 8.0);

network.createLine(0, 0, 1, 20.0, 0.06, 0.3, 11.0, 0.8);
network.createLine(1, 1, 2, 15.0, 0.06, 0.3, 11.0, 0.8);

network.compute();

// Report voltage profile
for (var i = 0; i < 3; i++) {
  var bus = network.getBusResult(i);
  print("Bus ${i}: |V| = ${bus.abs} pu, angle = ${bus.angle_radians} rad");
}

// Report line loading
for (var i = 0; i < 2; i++) {
  var line = network.getLineResult(i);
  print("Line ${i}: loading = ${line.loading_percent}%, losses = ${line.pl_mw} MW");
}
```

### Voltage Drop Analysis

```gcl
var network = PowerNetwork {};
network.configure(4, 3, 1, 3);

network.createBus(0, 20.0);
network.createBus(1, 20.0);
network.createBus(2, 20.0);
network.createBus(3, 20.0);

network.createExtGrid(0, 1.0);

// Distributed loads along a feeder
network.createLoad(1, 5.0, 2.0);
network.createLoad(2, 8.0, 3.0);
network.createLoad(3, 12.0, 5.0);

// Feeder segments
network.createLine(0, 0, 1, 3.0, 0.2, 0.4, 9.7, 0.4);
network.createLine(1, 1, 2, 4.0, 0.2, 0.4, 9.7, 0.4);
network.createLine(2, 2, 3, 5.0, 0.2, 0.4, 9.7, 0.4);

network.compute();

// Check voltage drop along the feeder
var refVoltage = network.getBusResult(0).abs;
for (var i = 0; i < 4; i++) {
  var bus = network.getBusResult(i);
  var dropPercent = (refVoltage - bus.abs) / refVoltage * 100.0;
  print("Bus ${i}: |V| = ${bus.abs} pu, drop = ${dropPercent}%");
}
```

### Line Overload Detection

```gcl
var network = PowerNetwork {};
network.configure(2, 1, 1, 1);

network.createBus(0, 110.0);
network.createBus(1, 110.0);

network.createExtGrid(0, 1.0);
network.createLoad(1, 100.0, 40.0);

network.createLine(0, 0, 1, 50.0, 0.06, 0.3, 11.0, 0.5);

network.compute();

var lineResult = network.getLineResult(0);

if (lineResult.loading_percent > 100.0) {
  print("WARNING: Line 0 is overloaded at ${lineResult.loading_percent}%!");
  print("  Current: ${lineResult.i_ka} kA");
} else if (lineResult.loading_percent > 80.0) {
  print("CAUTION: Line 0 is highly loaded at ${lineResult.loading_percent}%");
} else {
  print("Line 0 loading is normal at ${lineResult.loading_percent}%");
}
```

### Network Loss Summary

```gcl
var network = PowerNetwork {};
var nbBuses = 4;
var nbLines = 3;
network.configure(nbBuses, nbLines, 1, 2);

network.createBus(0, 110.0);
network.createBus(1, 110.0);
network.createBus(2, 110.0);
network.createBus(3, 110.0);

network.createExtGrid(0, 1.0);
network.createLoad(2, 40.0, 15.0);
network.createLoad(3, 25.0, 10.0);

network.createLine(0, 0, 1, 30.0, 0.06, 0.3, 11.0, 0.8);
network.createLine(1, 1, 2, 20.0, 0.06, 0.3, 11.0, 0.6);
network.createLine(2, 1, 3, 25.0, 0.08, 0.35, 10.5, 0.6);

network.compute();

// Sum up total losses
var totalActiveLoss = 0.0;
var totalReactiveLoss = 0.0;

for (var i = 0; i < nbLines; i++) {
  var line = network.getLineResult(i);
  totalActiveLoss = totalActiveLoss + line.pl_mw;
  totalReactiveLoss = totalReactiveLoss + line.ql_mvar;
  print("Line ${i}: P_loss=${line.pl_mw} MW, Q_loss=${line.ql_mvar} MVar");
}

print("Total active losses: ${totalActiveLoss} MW");
print("Total reactive losses: ${totalReactiveLoss} MVar");
```

## Best Practices

### Network Configuration

- **Call `configure()` first**: It must be called before any `createBus()`, `createLine()`, `createLoad()`, or `createExtGrid()` call
- **Match dimensions exactly**: The counts passed to `configure()` must match the number of components you create
- **Bus IDs start at 0**: Use sequential IDs from 0 to `nb_bus - 1`
- **Line IDs start at 0**: Use sequential IDs from 0 to `nb_lines - 1`

```gcl
// Good: dimensions match component count
var network = PowerNetwork {};
network.configure(3, 2, 1, 2);  // 3 buses, 2 lines, 1 ext grid, 2 loads

network.createBus(0, 110.0);
network.createBus(1, 110.0);
network.createBus(2, 110.0);    // 3 buses total

network.createExtGrid(0, 1.0);  // 1 ext grid

network.createLoad(1, 30.0, 10.0);
network.createLoad(2, 20.0, 8.0);  // 2 loads total

network.createLine(0, 0, 1, 20.0, 0.06, 0.3, 11.0, 0.8);
network.createLine(1, 1, 2, 15.0, 0.06, 0.3, 11.0, 0.8);  // 2 lines total
```

### Solver Workflow

- **Follow the four-step workflow**: configure -> create components -> compute -> read results
- **Handle convergence failures**: Wrap `compute()` in try-catch to handle non-convergent cases
- **Call `compute()` before reading results**: `getBusResult()` and `getLineResult()` return results from the last `compute()` call

```gcl
try {
  network.compute();
} catch (e) {
  print("Solver did not converge: ${e}");
  // Consider adjusting network parameters or checking topology
}
```

### Error Handling

- **Wrap `compute()` in try-catch**: The solver throws if it fails to converge
- **Validate topology**: Ensure the network is connected and has at least one external grid (slack bus)
- **Check line loading**: Values above 100% indicate overloaded lines

```gcl
try {
  network.compute();

  // Check for voltage issues
  for (var i = 0; i < nbBuses; i++) {
    var bus = network.getBusResult(i);
    if (bus.abs < 0.9 || bus.abs > 1.1) {
      print("WARNING: Bus ${i} voltage ${bus.abs} pu is outside 0.9-1.1 range");
    }
  }
} catch (e) {
  print("Power flow analysis failed: ${e}");
}
```

### Gotchas

- **`lenght_km` typo**: The `createLine()` parameter is spelled `lenght_km` (not `length_km`). This is a known typo in the API and must be used as-is
- **`configure()` takes 4 parameters**: Unlike the GCL doc comment that shows 3, `configure()` requires `nb_bus`, `nb_lines`, `nb_ext_grids`, **and** `nb_loads`
- **External grid is required**: Every network needs at least one external grid connection to serve as a slack/reference bus
- **Component count must match**: The number of created buses, lines, ext grids, and loads must exactly match the values passed to `configure()`
- **Per-unit voltage**: `createExtGrid()` takes voltage in per-unit (typically 1.0), not in kV
- **`compute()` must be called**: Results are only available after a successful `compute()` call
- **Convergence is not guaranteed**: Extreme loading conditions, disconnected networks, or unrealistic parameters may cause the solver to fail
- **Line direction matters for sign convention**: `p_from_mw` and `p_to_mw` represent power flowing into the line at each end; positive values mean power entering the line
