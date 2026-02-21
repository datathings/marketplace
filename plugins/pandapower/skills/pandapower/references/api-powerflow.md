# Power Flow API

## Table of Contents
1. [AC Power Flow (runpp)](#ac-power-flow)
2. [DC Power Flow (rundcpp)](#dc-power-flow)
3. [Optimal Power Flow (runopp)](#optimal-power-flow)
4. [DC Optimal Power Flow (rundcopp)](#dc-optimal-power-flow)
5. [Three-Phase Power Flow](#three-phase-power-flow)
6. [Short Circuit Calculation](#short-circuit-calculation)
7. [State Estimation](#state-estimation)
8. [Results Access](#results-access)
9. [Cost Functions for OPF](#cost-functions-for-opf)

---

## AC Power Flow

### `runpp(net, algorithm="nr", calculate_voltage_angles=True, init="auto", max_iteration="auto", tolerance_mva=1e-8, trafo_model="t", trafo_loading="current", enforce_p_lims=False, enforce_q_lims=False, check_connectivity=True, voltage_depend_loads=True, consider_line_temperature=False, run_control=False, distributed_slack=False, tdpf=False, **kwargs) -> None`
**Description:** Runs a balanced AC power flow (load flow). Results are written to `net.res_*` tables.
**Parameters:**
- `algorithm`: Solver algorithm:
  - `"nr"` - Newton-Raphson (default, fastest for most cases)
  - `"iwamoto_nr"` - Newton-Raphson with Iwamoto multiplier (more robust)
  - `"bfsw"` - Backward/forward sweep (radial networks)
  - `"gs"` - Gauss-Seidel
  - `"fdbx"`, `"fdxb"` - Fast-decoupled
- `calculate_voltage_angles`: Consider phase angles (needed for meshed/HV networks)
- `init`: Starting point: `"auto"`, `"flat"`, `"dc"`, `"results"`
- `tolerance_mva`: Convergence tolerance in MVA
- `enforce_q_lims`: Respect reactive power limits (converts PV to PQ if violated)
- `voltage_depend_loads`: Enable ZIP load model
- `numba` (kwarg): Use numba JIT for speedup (default True)
- `lightsim2grid` (kwarg): Use lightsim2grid C++ backend (`"auto"`)
- `tdpf`: Temperature-dependent power flow
**Returns:** None (in-place update of `net.res_*` tables).
**Example:**
```python
import pandapower as pp

net = pp.networks.case14()
pp.runpp(net)

# Access results
print(net.res_bus)       # Bus voltages (vm_pu, va_degree)
print(net.res_line)      # Line flows and loading
print(net.res_trafo)     # Transformer flows and loading
print(net.res_load)      # Load results (p_mw, q_mvar consumed)
print(net.res_gen)       # Generator output
```

### `rundcpp(net, trafo_model="t", trafo_loading="current", check_connectivity=True, ...) -> None`
**Description:** Runs a DC (linearized) power flow - fast approximation, no voltage magnitudes.
**Example:**
```python
pp.rundcpp(net)
print(net.res_bus)   # Only va_degree, no vm_pu
print(net.res_line)  # Only p_from_mw, no reactive
```

### `runpp_pgm(net, algorithm="nr", symmetric=True, ...) -> None`
**Description:** Runs power flow using the PowerGridModel C++ backend (faster for large networks).
**Parameters:**
- `symmetric`: True for balanced 3-phase, False for unbalanced

### `set_user_pf_options(net, overwrite=False, **kwargs) -> None`
**Description:** Sets persistent power flow options on the network object that override defaults for all subsequent `runpp()` calls.
**Example:**
```python
pp.set_user_pf_options(net, tolerance_mva=1e-6, algorithm="bfsw")
pp.runpp(net)  # uses bfsw algorithm
```

---

## Optimal Power Flow

### `runopp(net, verbose=False, calculate_voltage_angles=True, check_connectivity=True, suppress_warnings=True, init="flat", delta=1e-10, consider_line_temperature=False, **kwargs) -> None`
**Description:** Runs AC Optimal Power Flow. Minimizes cost functions subject to network and element constraints.

**Prerequisites:**
- Cost functions defined with `create_poly_cost()` or `create_pwl_cost()`
- Controllable elements with min/max limits

**Flexibilities (controllable elements):**
- `net.gen`: `min_p_mw`, `max_p_mw`, `min_q_mvar`, `max_q_mvar`, set `controllable=True`
- `net.sgen`: same + `min_q_mvar`, `max_q_mvar`
- `net.load`: `min_p_mw`, `max_p_mw` (demand response)
- `net.storage`: `min_p_mw`, `max_p_mw`
- `net.ext_grid`: `min_p_mw`, `max_p_mw`, `min_q_mvar`, `max_q_mvar`

**Network constraints:**
- `net.bus`: `min_vm_pu`, `max_vm_pu`
- `net.line`: `max_loading_percent`
- `net.trafo`: `max_loading_percent`

**Parameters:**
- `init`: `"flat"`, `"pf"` (start from power flow), `"results"`

**Example:**
```python
import pandapower as pp

net = pp.create_empty_network()
# ... create network ...

# Add cost function (minimize generation cost)
pp.create_poly_cost(net, element=0, et="gen", cp1_eur_per_mw=50.)

# Set OPF constraints
net.gen.at[0, "min_p_mw"] = 0.
net.gen.at[0, "max_p_mw"] = 100.
net.bus.at[0, "min_vm_pu"] = 0.95
net.bus.at[0, "max_vm_pu"] = 1.05

pp.runopp(net)
print(net.res_gen)   # Optimal dispatch
print(net.res_cost)  # Total cost
```

### `rundcopp(net, verbose=False, check_connectivity=True, suppress_warnings=True, ...) -> None`
**Description:** Runs DC Optimal Power Flow (linearized, no voltage magnitudes). Faster than AC OPF.

---

## Three-Phase Power Flow

### `runpp_3ph(net, calculate_voltage_angles=True, init="auto", max_iteration="auto", tolerance_mva=1e-8, ...) -> None`
**Description:** Runs unbalanced three-phase AC power flow. Requires asymmetric load/sgen data.
**Example:**
```python
from pandapower.pf.runpp_3ph import runpp_3ph
runpp_3ph(net)
print(net.res_bus_3ph)   # Per-phase voltages (vm_a_pu, vm_b_pu, vm_c_pu)
print(net.res_line_3ph)  # Per-phase line currents
```

---

## Short Circuit Calculation

### `pp.shortcircuit.calc_sc(net, bus=None, fault="3ph", case="max", lv_tol_percent=10, topology="auto", ip=False, ith=False, tk_s=1., branch_results=False, ...) -> None`
**Description:** Calculates short-circuit currents according to IEC 60909. Results in `net.res_bus_sc`.
**Parameters:**
- `fault`: `"3ph"` (three-phase), `"2ph"` (phase-to-phase), `"1ph"` (single-phase ground)
- `case`: `"max"` or `"min"` short-circuit current
- `lv_tol_percent`: LV voltage tolerance (6 or 10%)
- `ip`: Calculate aperiodic (peak) short-circuit current
- `ith`: Calculate thermal equivalent current
- `topology`: `"auto"`, `"meshed"`, or `"radial"`
- `branch_results`: Also calculate branch short-circuit currents
**Example:**
```python
import pandapower.shortcircuit as sc

net = pp.networks.case14()
sc.calc_sc(net, fault="3ph", case="max")
print(net.res_bus_sc)   # ikss_ka: initial short-circuit current per bus
```

---

## State Estimation

### `pp.estimation.estimate(net, algorithm="wls", init="flat", tolerance=1e-6, maximum_iterations=50, zero_injection="aux_bus", ...) -> bool`
**Description:** Runs Weighted Least Squares (WLS) state estimation from measurements.

**Requires measurements** created with `create_measurement()`.

**Parameters:**
- `algorithm`: `"wls"`, `"wls_with_zero_constraint"`, `"irwls"`, `"lp"`, `"opt"`, `"af-wls"`
- `init`: `"flat"` or `"results"`
- `zero_injection`: How to handle zero-injection buses

**Returns:** `True` if converged, `False` otherwise.

**Example:**
```python
import pandapower as pp
import pandapower.estimation as est

net = pp.networks.case14()

# Add measurements (e.g., from SCADA)
pp.create_measurement(net, "v", "bus", 1.01, 0.01, element=0)    # Bus 0 voltage
pp.create_measurement(net, "p", "bus", 50., 1., element=0)         # Bus 0 active power
pp.create_measurement(net, "q", "bus", 20., 1., element=0)         # Bus 0 reactive power
pp.create_measurement(net, "p", "line", 30., 1., element=0, side="from")

success = est.estimate(net, algorithm="wls")
if success:
    print(net.res_bus_est)   # Estimated bus voltages
```

---

## Results Access

After running power flow, results are stored in `net.res_*` DataFrames:

| Table | Contents |
|-------|----------|
| `net.res_bus` | `vm_pu`, `va_degree`, `p_mw`, `q_mvar` |
| `net.res_line` | `p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`, `pl_mw`, `ql_mvar`, `i_from_ka`, `i_to_ka`, `loading_percent` |
| `net.res_trafo` | `p_hv_mw`, `q_hv_mvar`, `p_lv_mw`, `q_lv_mvar`, `pl_mw`, `loading_percent` |
| `net.res_gen` | `p_mw`, `q_mvar`, `vm_pu` |
| `net.res_sgen` | `p_mw`, `q_mvar` |
| `net.res_load` | `p_mw`, `q_mvar` |
| `net.res_ext_grid` | `p_mw`, `q_mvar` |
| `net.res_shunt` | `p_mw`, `q_mvar`, `vm_pu` |
| `net.res_bus_sc` | `ikss_ka`, `ip_ka`, `ith_ka` (short circuit) |
| `net.res_bus_est` | State estimation results |

```python
# Check convergence
print(net.converged)  # True/False

# Max line loading
print(net.res_line.loading_percent.max())

# Buses with voltage violations
violating = net.res_bus[(net.res_bus.vm_pu > 1.05) | (net.res_bus.vm_pu < 0.95)]
```

---

## Cost Functions for OPF

### `create_poly_cost(net, element, et, cp0_eur=0, cp1_eur_per_mw=0, cp2_eur_per_mw2=0, cq0_eur=0, cq1_eur_per_mvar=0, cq2_eur_per_mvar2=0) -> int`
**Description:** Creates a polynomial cost function: `cost = cp0 + cp1*P + cp2*P^2`.
**Parameters:**
- `element`: Element index
- `et`: Element type: `"gen"`, `"sgen"`, `"load"`, `"ext_grid"`, `"storage"`, `"dcline"`
**Example:**
```python
pp.create_poly_cost(net, element=0, et="gen", cp1_eur_per_mw=50., cp2_eur_per_mw2=0.1)
```

### `create_pwl_cost(net, element, et, points, power_type="p") -> int`
**Description:** Creates a piecewise linear cost function.
**Parameters:**
- `points`: List of `(p_mw, cost_eur)` breakpoints
**Example:**
```python
pp.create_pwl_cost(net, element=0, et="gen", points=[(0, 0), (50, 2500), (100, 7500)])
```
