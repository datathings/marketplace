# Network Creation API

## Table of Contents
1. [Create Network](#create-network)
2. [Buses](#buses)
3. [Lines](#lines)
4. [Transformers](#transformers)
5. [Loads](#loads)
6. [Generators](#generators)
7. [Static Generators](#static-generators)
8. [External Grids](#external-grids)
9. [Switches](#switches)
10. [Shunts and Other Elements](#shunts-and-other-elements)
11. [Standard Types](#standard-types)
12. [Predefined Networks](#predefined-networks)

---

## Create Network

### `create_empty_network(name="", f_hz=50.0, sn_mva=1, add_stdtypes=True) -> pandapowerNet`
**Description:** Initializes an empty pandapower network data structure (an attribute dict backed by pandas DataFrames).
**Parameters:**
- `name`: Name of the network
- `f_hz`: System frequency in Hz (50 for Europe, 60 for North America)
- `sn_mva`: Reference apparent power for per-unit system
- `add_stdtypes`: Whether to populate the standard type library
**Returns:** Empty `pandapowerNet` object.
**Example:**
```python
import pandapower as pp
net = pp.create_empty_network(name="My Grid", f_hz=50.0, sn_mva=1)
```

---

## Buses

### `create_bus(net, vn_kv, name=None, index=None, geodata=None, type="b", zone=None, in_service=True, max_vm_pu=nan, min_vm_pu=nan) -> int`
**Description:** Adds one bus (node) to `net.bus`.
**Parameters:**
- `vn_kv`: Nominal voltage level in kV
- `name`: Optional bus name
- `geodata`: `(x, y)` tuple for plotting
- `type`: `"b"` (busbar), `"n"` (node), or `"m"` (muff)
- `max_vm_pu`, `min_vm_pu`: Voltage bounds for OPF
**Returns:** Integer index of created bus.
**Example:**
```python
hv_bus = pp.create_bus(net, vn_kv=110., name="HV Bus")
lv_bus = pp.create_bus(net, vn_kv=20., name="LV Bus")
```

### `create_buses(net, nr_buses, vn_kv, index=None, name=None, type="b", geodata=None) -> array`
**Description:** Creates multiple buses at once (vectorized).
**Example:**
```python
buses = pp.create_buses(net, nr_buses=5, vn_kv=0.4, name=["Bus %d" % i for i in range(5)])
```

---

## Lines

### `create_line(net, from_bus, to_bus, length_km, std_type, name=None, index=None, geodata=None, df=1.0, parallel=1, in_service=True, max_loading_percent=nan) -> int`
**Description:** Creates a line element from standard type library.
**Parameters:**
- `from_bus`, `to_bus`: Bus indices at each end
- `length_km`: Line length in km
- `std_type`: Name from standard type library (e.g., `"NAYY 4x50 SE"`, `"149-AL1/24-ST1A 110.0"`)
- `df`: Derating factor (0 to 1)
- `parallel`: Number of parallel circuits
- `max_loading_percent`: Loading limit for OPF
**Returns:** Integer index.
**Example:**
```python
line = pp.create_line(net, from_bus=hv_bus, to_bus=lv_bus, length_km=10., std_type="149-AL1/24-ST1A 110.0")
```

### `create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka, ...) -> int`
**Description:** Creates a line with explicit electrical parameters instead of a standard type.
**Parameters:**
- `r_ohm_per_km`: Resistance per km
- `x_ohm_per_km`: Reactance per km
- `c_nf_per_km`: Capacitance per km in nF
- `max_i_ka`: Maximum current rating in kA
**Example:**
```python
line = pp.create_line_from_parameters(
    net, from_bus=0, to_bus=1, length_km=5.,
    r_ohm_per_km=0.1, x_ohm_per_km=0.4,
    c_nf_per_km=10., max_i_ka=0.5
)
```

---

## Transformers

### `create_transformer(net, hv_bus, lv_bus, std_type, name=None, tap_pos=nan, in_service=True, max_loading_percent=nan, parallel=1) -> int`
**Description:** Creates a two-winding transformer from standard type library.
**Parameters:**
- `hv_bus`: High-voltage side bus index
- `lv_bus`: Low-voltage side bus index
- `std_type`: Standard type name (e.g., `"25 MVA 110/20 kV"`)
- `tap_pos`: Tap changer position (defaults to neutral)
**Returns:** Integer index.
**Example:**
```python
trafo = pp.create_transformer(net, hv_bus=hv_bus, lv_bus=lv_bus, std_type="25 MVA 110/20 kV")
```

### `create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva, vn_hv_kv, vn_lv_kv, vkr_percent, vk_percent, pfe_kw, i0_percent, ...) -> int`
**Description:** Creates a two-winding transformer with explicit parameters.
**Key Parameters:**
- `sn_mva`: Rated apparent power in MVA
- `vn_hv_kv`, `vn_lv_kv`: Rated voltages HV/LV side
- `vk_percent`: Short-circuit voltage in %
- `vkr_percent`: Real part of short-circuit voltage in %
- `pfe_kw`: Iron core losses in kW
- `i0_percent`: No-load current in %
**Example:**
```python
trafo = pp.create_transformer_from_parameters(
    net, hv_bus=hv_bus, lv_bus=lv_bus,
    sn_mva=25., vn_hv_kv=110., vn_lv_kv=20.,
    vkr_percent=0.16, vk_percent=12.5,
    pfe_kw=27., i0_percent=0.06
)
```

### `create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, ...) -> int`
**Description:** Creates a three-winding transformer.

---

## Loads

### `create_load(net, bus, p_mw, q_mvar=0, name=None, in_service=True, scaling=1.0, controllable=nan, max_p_mw=nan, min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan) -> int`
**Description:** Adds a load (consumer) to a bus. Uses consumer sign convention: positive p_mw = consumption.
**Parameters:**
- `p_mw`: Active power (positive = load, negative = generation)
- `q_mvar`: Reactive power
- `scaling`: Multiplier applied to p_mw and q_mvar
- `controllable`: If True, load is a flexibility for OPF
**Example:**
```python
load = pp.create_load(net, bus=lv_bus, p_mw=5.0, q_mvar=1.5, name="Industrial Load")
```

### `create_loads(net, buses, p_mw, q_mvar=0, ...) -> array`
**Description:** Creates multiple loads at once.
**Example:**
```python
pp.create_loads(net, buses=[0, 1, 2], p_mw=[1.0, 2.0, 3.0])
```

---

## Generators

### `create_gen(net, bus, p_mw, vm_pu=1.0, sn_mva=nan, name=None, slack=False, controllable=nan, min_p_mw=nan, max_p_mw=nan, min_q_mvar=nan, max_q_mvar=nan) -> int`
**Description:** Adds a voltage-controlled generator (PV node). Active power and voltage setpoint are inputs.
**Parameters:**
- `p_mw`: Active power in MW (positive = generation)
- `vm_pu`: Voltage setpoint in per unit
- `slack`: If True, this generator is the slack bus (reference)
- `controllable`: OPF controllability flag
**Example:**
```python
gen = pp.create_gen(net, bus=hv_bus, p_mw=50., vm_pu=1.02, name="Gas Turbine", slack=True)
```

### `create_sgen(net, bus, p_mw, q_mvar=0, name=None, in_service=True, controllable=nan, ...) -> int`
**Description:** Adds a static generator (PQ node) - models renewables, inverter-based resources.
**Parameters:**
- `p_mw`: Active power (positive = injection)
- `q_mvar`: Reactive power
**Example:**
```python
sgen = pp.create_sgen(net, bus=lv_bus, p_mw=2.0, q_mvar=0., name="PV Plant")
```

---

## External Grids

### `create_ext_grid(net, bus, vm_pu=1.0, va_degree=0.0, name=None, in_service=True, max_p_mw=nan, min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan) -> int`
**Description:** Creates an external grid connection (slack bus in power flow). Represents the infinite bus / grid equivalent.
**Parameters:**
- `vm_pu`: Voltage setpoint in per unit
- `va_degree`: Voltage angle setpoint in degrees (usually 0 for reference)
**Example:**
```python
ext_grid = pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.02, name="Grid Connection")
```

---

## Switches

### `create_switch(net, bus, element, et, closed=True, type="CB", name=None) -> int`
**Description:** Creates a switch to connect/disconnect buses or branches.
**Parameters:**
- `bus`: Bus to which switch is connected
- `element`: Index of the connected element (bus or branch)
- `et`: Element type: `"b"` (bus-bus), `"l"` (line), `"t"` (trafo), `"t3"` (trafo3w)
- `closed`: Switch state
- `type`: Switch type: `"CB"` (circuit breaker), `"LBS"` (load break), `"DS"` (disconnector)
**Example:**
```python
sw = pp.create_switch(net, bus=hv_bus, element=line, et="l", closed=True, type="CB")
```

---

## Shunts and Other Elements

### `create_shunt(net, bus, q_mvar, p_mw=0., vn_kv=nan, name=None, in_service=True) -> int`
**Description:** Creates a shunt element (capacitor bank / reactor).
**Parameters:**
- `q_mvar`: Reactive power at rated voltage (negative = capacitor, positive = reactor)
- `vn_kv`: Rated voltage

### `create_impedance(net, from_bus, to_bus, rft_pu, xft_pu, sn_mva, rtf_pu=nan, xtf_pu=nan, ...) -> int`
**Description:** Creates a series impedance element.

### `create_storage(net, bus, p_mw, max_e_mwh, q_mvar=0, ...) -> int`
**Description:** Creates a storage element with energy capacity and power limits.

### `create_motor(net, bus, pn_mech_mw, lrc_pu, vn_kv, efficiency_percent, cos_phi, ...) -> int`
**Description:** Creates an induction motor element.

### `create_dcline(net, from_bus, to_bus, p_mw, loss_percent, loss_mw, vm_from_pu, vm_to_pu, ...) -> int`
**Description:** Creates an HVDC line between two AC buses.

### `create_ward(net, bus, ps_mw, qs_mvar, pz_mw, qz_mvar, ...) -> int`
**Description:** Creates a Ward equivalent (external network reduction).

### `create_measurement(net, meas_type, element_type, value, std_dev, element, side=None) -> int`
**Description:** Creates a measurement for state estimation.
**Parameters:**
- `meas_type`: `"v"` (voltage), `"p"` (active power), `"q"` (reactive power), `"i"` (current)
- `element_type`: `"bus"`, `"line"`, `"trafo"`, etc.
- `value`: Measured value
- `std_dev`: Standard deviation (measurement accuracy)

---

## Standard Types

### `pp.available_std_types(net, element="line") -> DataFrame`
**Description:** Lists all available standard types.
```python
# View available line types
print(pp.available_std_types(net, "line"))
# View available transformer types
print(pp.available_std_types(net, "trafo"))
```

### `pp.create_std_type(net, data, name, element="line") -> None`
**Description:** Adds a custom standard type.
```python
my_line_type = {"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.35,
                "c_nf_per_km": 8.0, "max_i_ka": 0.55}
pp.create_std_type(net, my_line_type, name="My Cable", element="line")
```

---

## Predefined Networks

```python
import pandapower.networks as nw

# Test networks
net = nw.case4gs()              # 4-bus MATPOWER case
net = nw.case14()               # IEEE 14-bus
net = nw.case30()               # IEEE 30-bus
net = nw.case57()               # IEEE 57-bus
net = nw.case118()              # IEEE 118-bus
net = nw.case300()              # IEEE 300-bus

# Realistic distribution networks
net = nw.mv_oberrhein()         # Medium voltage radial network
net = nw.create_cigre_network_mv()  # CIGRE MV benchmark
net = nw.create_cigre_network_lv()  # CIGRE LV benchmark
net = nw.example_simple()       # Minimal example network
net = nw.example_multivoltage() # Multi-voltage level example
```
