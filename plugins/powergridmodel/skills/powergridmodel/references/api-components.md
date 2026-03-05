# Component Types Reference

All components are represented as numpy structured arrays. Use `initialize_array('input', '<type>', n)` to create them.

## Table of Contents
1. [Component hierarchy](#component-hierarchy)
2. [Node](#node)
3. [Branch components](#branch-components)
4. [Branch3 components](#branch3-components)
5. [Appliance components](#appliance-components)
6. [Sensor components](#sensor-components)
7. [Fault](#fault)
8. [Regulator components](#regulator-components)
9. [ComponentType enum](#componenttype-enum)
10. [DatasetType enum](#datasettype-enum)
11. [Enumerations](#enumerations)

---

## Component hierarchy

```
base (id)
├── node
├── branch (from_node, to_node, from_status, to_status)
│   ├── line
│   ├── link
│   ├── transformer
│   ├── generic_branch
│   └── asym_line
├── branch3 (node_1..3, status_1..3)
│   └── three_winding_transformer
├── appliance (node, status)
│   ├── source
│   ├── sym_load / sym_gen
│   ├── asym_load / asym_gen
│   └── shunt
├── sensor (measured_object)
│   ├── sym_voltage_sensor / asym_voltage_sensor
│   ├── sym_power_sensor / asym_power_sensor
│   └── sym_current_sensor / asym_current_sensor
├── fault
└── regulator (regulated_object)
    ├── transformer_tap_regulator
    └── voltage_regulator
```

`RealValueOutput`: `float64` for symmetric, `(3,)float64` for asymmetric (per-phase).

---

## Node

**type name:** `node`

### Input
| Attribute | Type | Unit | Required | Description |
|-----------|------|------|----------|-------------|
| `id` | `int32` | - | yes | Unique component ID |
| `u_rated` | `float64` | V | yes | Rated line-to-line voltage (`> 0`) |

### Steady-state output (sym: scalar, asym: 3-element array)
| Attribute | Unit | Description |
|-----------|------|-------------|
| `u_pu` | - | Per-unit voltage magnitude |
| `u_angle` | rad | Voltage angle |
| `u` | V | Voltage magnitude (LL for sym, LN for asym) |
| `p` | W | Active power injection (generator convention) |
| `q` | var | Reactive power injection |

### Short-circuit output
`u_pu`, `u_angle`, `u` (LN).

---

## Branch components

All branches have common input fields: `from_node`, `to_node`, `from_status` (0/1), `to_status` (0/1).

Common steady-state output: `p_from`, `q_from`, `i_from`, `s_from`, `p_to`, `q_to`, `i_to`, `s_to`, `loading`.

Common short-circuit output: `i_from`, `i_from_angle`, `i_to`, `i_to_angle`.

### Line (`line`)
Symmetric or asymmetric cable/overhead line. Both nodes must have equal `u_rated`.

**Input (symmetric params):**
| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `r1` | Ω | yes | Positive-sequence series resistance |
| `x1` | Ω | yes | Positive-sequence series reactance (`r1` and `x1` not both 0) |
| `c1` | F | yes | Positive-sequence shunt capacitance |
| `tan1` | - | yes | Positive-sequence shunt loss factor |
| `r0` | Ω | asym only | Zero-sequence series resistance |
| `x0` | Ω | asym only | Zero-sequence series reactance |
| `c0` | F | asym only | Zero-sequence shunt capacitance |
| `tan0` | - | asym only | Zero-sequence shunt loss factor |
| `i_n` | A | no | Rated current (omit → `loading` = NaN) |

### Link (`link`)
High-admittance connection between busbars inside a substation. No additional attributes. No sensors can be attached.

### Transformer (`transformer`)
Two-winding transformer, possibly with tap changer and different voltage levels.

**Input (key attributes):**
| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `u1` / `u2` | V | yes | Rated voltage at from/to side |
| `sn` | VA | yes | Rated power |
| `uk` | - | yes | Relative short-circuit voltage (e.g. 0.1 = 10%) |
| `pk` | W | yes | Copper loss |
| `i0` | - | yes | No-load current (relative) |
| `p0` | W | yes | Iron loss |
| `winding_from` / `winding_to` | `WindingType` | yes | Winding configuration |
| `clock` | - | yes | Phase shift clock number (−12 to 12) |
| `tap_side` | `BranchSide` | yes | Side where tap changer is located |
| `tap_pos` | - | no | Current tap position |
| `tap_min` / `tap_max` | - | yes | Tap range |
| `tap_size` | V | yes | Voltage step per tap |
| `r_grounding_from/to` | Ω | no | Grounding resistance/reactance |
| `i0_zero_sequence` | - | no | Zero-seq no-load current (for 3-leg core-type) |

### Generic Branch (`generic_branch`)
PI-model branch with direct circuit parameters. Symmetric only. Supports lines and transformers via off-nominal ratio.

| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `r1` / `x1` | Ω | yes | Series resistance/reactance (referenced to "to" side) |
| `g1` / `b1` | S | yes | Shunt conductance/susceptance |
| `k` | - | no (1.0) | Off-nominal tap ratio (`> 0`) |
| `theta` | rad | no (0.0) | Angle shift |
| `sn` | VA | no (0.0) | Rated power |

**Note:** `k` is the **off-nominal ratio**, not the voltage ratio. Must be set explicitly for tap changers.

### Asym Line (`asym_line`)
3- or 4-phase line with per-phase impedance matrix. Symmetric nodes required.

**Input:** Resistance matrix entries `r_aa`, `r_ba`, `r_bb`, `r_ca`, `r_cb`, `r_cc` (and optionally neutral entries `r_na`...`r_nn`). Same pattern for `x_*`. Capacitance either via full matrix `c_aa`...`c_cc` or via `c0` + `c1`.

---

## Branch3 components

Three-port components connecting three different nodes. Sides labeled 1, 2, 3.

**Input:** `node_1`, `node_2`, `node_3`, `status_1`, `status_2`, `status_3`.

**Output:** `p_1..3`, `q_1..3`, `i_1..3`, `s_1..3`, `loading`.

### Three-Winding Transformer (`three_winding_transformer`)
| Attribute | Description |
|-----------|-------------|
| `u1/u2/u3` | Rated voltages at each side |
| `sn_1/sn_2/sn_3` | Rated powers |
| `uk_12/uk_13/uk_23` | Short-circuit voltages between pairs |
| `pk_12/pk_13/pk_23` | Copper losses between pairs |
| `i0` / `p0` | Magnetizing current and iron loss (ref side 1) |
| `winding_1/2/3` | `WindingType` |
| `clock_12` / `clock_13` | Phase shift clocks |
| `tap_side` | `Branch3Side` |
| `tap_pos/min/max/nom/size` | Tap changer parameters |

---

## Appliance components

All appliances: input fields `node` (node ID), `status` (0/1).

Common steady-state output: `p` (W), `q` (var), `i` (A), `s` (VA), `pf` (power factor).

### Source (`source`)
Slack bus / voltage-controlled generator (external grid connection).

| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `u_ref` | p.u. | yes | Reference voltage magnitude |
| `u_ref_angle` | rad | no (0.0) | Reference voltage angle |
| `sk` | VA | no | Short-circuit power (default: infinite) |
| `rx_ratio` | - | no | R/X ratio of source impedance |
| `z01_ratio` | - | no | Z0/Z1 ratio |

### Symmetric Load / Generator (`sym_load` / `sym_gen`)

| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `type` | `LoadGenType` | yes | `const_power`, `const_impedance`, `const_current` |
| `p_specified` | W | yes | Active power (load: consumed, gen: produced) |
| `q_specified` | var | yes | Reactive power |

Updatable in batch: `status`, `p_specified`, `q_specified`.

### Asymmetric Load / Generator (`asym_load` / `asym_gen`)
Same as sym but `p_specified` and `q_specified` are `(3,)float64` arrays (per-phase values).

### Shunt (`shunt`)
Fixed admittance connected to a node.

| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `g1` | S | yes | Positive-sequence conductance |
| `b1` | S | yes | Positive-sequence susceptance |
| `g0` / `b0` | S | asym only | Zero-sequence conductance/susceptance |

---

## Sensor components

Sensors attach to physical components for state estimation. All sensors: `measured_object` (ID of measured component), `u_sigma` or `power_sigma` for measurement uncertainty.

### Generic Voltage Sensor — Concrete Types

| Type | Symmetric | Description |
|------|-----------|-------------|
| `sym_voltage_sensor` | yes | Voltage magnitude (+ optional angle) at a node |
| `asym_voltage_sensor` | no | Per-phase voltage at a node |

**Input:**
| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `measured_object` | - | yes | Node ID |
| `u_sigma` | V | yes | Measurement uncertainty (std dev) |
| `u_measured` | V | yes | Measured voltage magnitude |
| `u_angle_measured` | rad | no | Measured voltage angle (if available) |

### Generic Power Sensor — Concrete Types

| Type | Symmetric | Terminal types allowed |
|------|-----------|----------------------|
| `sym_power_sensor` | yes | branch_from/to, source, shunt, load, generator, branch3_1/2/3, node |
| `asym_power_sensor` | no | same |

**Input:**
| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `measured_object` | - | yes | ID of measured component |
| `measured_terminal_type` | `MeasuredTerminalType` | yes | Which terminal to measure |
| `power_sigma` | VA | yes | Measurement uncertainty |
| `p_measured` | W | yes | Measured active power |
| `q_measured` | var | yes | Measured reactive power |

### Generic Current Sensor — Concrete Types

| Type | Symmetric | Angle type |
|------|-----------|-----------|
| `sym_current_sensor` | yes | local or global angle |
| `asym_current_sensor` | no | local or global angle |

**Input:**
| Attribute | Unit | Required | Description |
|-----------|------|----------|-------------|
| `measured_object` | - | yes | Branch/Branch3 ID |
| `measured_terminal_type` | `MeasuredTerminalType` | yes | `branch_from`, `branch_to`, `branch3_1/2/3` |
| `angle_measurement_type` | `AngleMeasurementType` | yes | `local_angle` or `global_angle` |
| `i_sigma` | A | yes | Current magnitude uncertainty |
| `i_angle_sigma` | rad | yes | Current angle uncertainty |
| `i_measured` | A | yes | Measured current magnitude |
| `i_angle_measured` | rad | yes | Measured current angle |

**Note:** Cannot mix power sensors and current sensors on the same terminal. Global-angle current sensors require at least one voltage phasor measurement.

---

## Fault

**type name:** `fault`

Used in short-circuit calculations.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | `int8` | yes | 1 = fault active |
| `fault_object` | `int32` | yes | Node ID where fault occurs |
| `fault_type` | `FaultType` | yes | `three_phase`, `single_phase_to_ground`, `two_phase`, `two_phase_to_ground` |
| `fault_phase` | `FaultPhase` | yes | `abc`, `a`, `b`, `c`, `ab`, `ac`, `bc` |
| `r_f` | `float64` | no (0) | Fault resistance (Ω) |
| `x_f` | `float64` | no (0) | Fault reactance (Ω) |

**Output:** `id`, `energized`, `i_f` (A), `i_f_angle` (rad) — per-phase arrays.

---

## Regulator components

### Transformer Tap Regulator (`transformer_tap_regulator`)
Controls tap position of a `transformer` or `three_winding_transformer` to regulate voltage.

| Attribute | Required | Description |
|-----------|----------|-------------|
| `regulated_object` | yes | Transformer ID |
| `control_side` | yes | `BranchSide` or `Branch3Side` — side where control voltage is measured |
| `u_set` | yes | Target voltage in p.u. |
| `u_band` | yes | Acceptable voltage band width in p.u. |
| `line_drop_compensation_r/x` | no | Line drop compensation impedance |

**Tap-changing strategies** (pass to `calculate_power_flow`):
| Strategy | Description |
|----------|-------------|
| `disabled` | No automatic tap changing (default) |
| `any_valid_tap` | Any tap within `u_band` (linear search) |
| `min_voltage_tap` | Lowest voltage within `u_band` (binary search) |
| `max_voltage_tap` | Highest voltage within `u_band` (binary search) |
| `fast_any_tap` | Any tap within `u_band` (binary search) |

### Voltage Regulator (`voltage_regulator`)
Regulates the voltage at a node by adjusting a source's reference voltage.

---

## ComponentType enum

```python
from power_grid_model import ComponentType

ComponentType.node                   # "node"
ComponentType.line                   # "line"
ComponentType.asym_line              # "asym_line"
ComponentType.link                   # "link"
ComponentType.generic_branch         # "generic_branch"
ComponentType.transformer            # "transformer"
ComponentType.three_winding_transformer  # "three_winding_transformer"
ComponentType.transformer_tap_regulator # "transformer_tap_regulator"
ComponentType.sym_load               # "sym_load"
ComponentType.sym_gen                # "sym_gen"
ComponentType.asym_load              # "asym_load"
ComponentType.asym_gen               # "asym_gen"
ComponentType.shunt                  # "shunt"
ComponentType.source                 # "source"
ComponentType.sym_voltage_sensor     # "sym_voltage_sensor"
ComponentType.asym_voltage_sensor    # "asym_voltage_sensor"
ComponentType.sym_power_sensor       # "sym_power_sensor"
ComponentType.asym_power_sensor      # "asym_power_sensor"
ComponentType.sym_current_sensor     # "sym_current_sensor"
ComponentType.asym_current_sensor    # "asym_current_sensor"
ComponentType.fault                  # "fault"
ComponentType.voltage_regulator      # "voltage_regulator"
```

## DatasetType enum

```python
from power_grid_model import DatasetType

DatasetType.input       # "input"        — component input arrays
DatasetType.update      # "update"       — batch update arrays
DatasetType.sym_output  # "sym_output"   — symmetric result arrays
DatasetType.asym_output # "asym_output"  — asymmetric result arrays
DatasetType.sc_output   # "sc_output"    — short-circuit result arrays
```

---

## Enumerations

```python
from power_grid_model import (
    LoadGenType, WindingType, BranchSide, Branch3Side,
    MeasuredTerminalType, FaultType, FaultPhase,
    CalculationMethod, TapChangingStrategy,
    ShortCircuitVoltageScaling, AngleMeasurementType,
)

# LoadGenType
LoadGenType.const_power      # ZIP model: constant power (default)
LoadGenType.const_impedance  # ZIP model: constant impedance
LoadGenType.const_current    # ZIP model: constant current

# WindingType
WindingType.wye        # Y
WindingType.wye_n      # YN (with neutral)
WindingType.delta      # D
WindingType.zigzag     # Z
WindingType.zigzag_n   # ZN

# BranchSide
BranchSide.from_side
BranchSide.to_side

# Branch3Side
Branch3Side.side_1 / side_2 / side_3

# MeasuredTerminalType
MeasuredTerminalType.branch_from / branch_to
MeasuredTerminalType.source / shunt / load / generator
MeasuredTerminalType.branch3_1 / branch3_2 / branch3_3
MeasuredTerminalType.node   # total injection at node

# FaultType
FaultType.three_phase
FaultType.single_phase_to_ground
FaultType.two_phase
FaultType.two_phase_to_ground

# FaultPhase
FaultPhase.abc / a / b / c / ab / ac / bc / default_value

# CalculationMethod
CalculationMethod.newton_raphson    # power flow (default) + state est
CalculationMethod.iterative_current # power flow
CalculationMethod.linear            # power flow (approx)
CalculationMethod.linear_current    # power flow (approx)
CalculationMethod.iterative_linear  # state estimation (default)
CalculationMethod.iec60909          # short circuit (default)

# TapChangingStrategy
TapChangingStrategy.disabled / any_valid_tap / min_voltage_tap / max_voltage_tap / fast_any_tap

# ShortCircuitVoltageScaling
ShortCircuitVoltageScaling.minimum / maximum

# AngleMeasurementType
AngleMeasurementType.local_angle / global_angle
```
