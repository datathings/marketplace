---
name: pandapower
description: "Power systems analysis and optimization library (pandapower v3.4.0). Use when working with electric power networks: building network models (buses, lines, transformers, loads, generators), running AC/DC power flow, optimal power flow (OPF), short circuit calculations (IEC 60909), state estimation, time series simulations, network topology analysis, or visualizing power grids in Python."
---

# pandapower

## Overview

pandapower is an open-source Python library for automated analysis and optimization of power systems. It stores network data as pandas DataFrames, provides Newton-Raphson and other power flow solvers (including C++ backends via lightsim2grid and PowerGridModel), and supports advanced studies including OPF, short circuit (IEC 60909), three-phase unbalanced flow, and state estimation.

**Version:** v3.4.0
**Language:** Python
**License:** BSD 3-Clause
**Authors:** University of Kassel (e2n) and Fraunhofer IEE

## Quick Start

```python
import pandapower as pp

# Create network
net = pp.create_empty_network(f_hz=50.)

# Add buses
b_hv = pp.create_bus(net, vn_kv=110., name="HV Bus")
b_mv = pp.create_bus(net, vn_kv=20., name="MV Bus")

# Add external grid (slack/reference)
pp.create_ext_grid(net, bus=b_hv, vm_pu=1.02)

# Add transformer (uses built-in standard type library)
pp.create_transformer(net, hv_bus=b_hv, lv_bus=b_mv, std_type="25 MVA 110/20 kV")

# Add load
pp.create_load(net, bus=b_mv, p_mw=10.0, q_mvar=2.0)

# Run AC power flow
pp.runpp(net)

# Inspect results (stored in net.res_* DataFrames)
print(net.res_bus[["vm_pu", "va_degree"]])
print(net.res_trafo[["loading_percent"]])
print(f"Converged: {net.converged}")
```

## Core Concepts

- **Network as DataFrames:** All elements stored as pandas DataFrames (`net.bus`, `net.line`, `net.load`, etc.); results in `net.res_*` tables after power flow.
- **Consumer sign convention:** Positive `p_mw` means consumption for loads; positive `p_mw` means generation for generators/sgens.
- **Standard types:** Built-in library of line and transformer types; custom types supported via `create_std_type()`.
- **Per-unit system:** Voltages in per unit (p.u.) with `sn_mva` as base; power in MW/Mvar; impedances in ohm/km.
- **In-place results:** `runpp()` and other solvers write results to `net.res_*` tables; check `net.converged` after each run.
- **Modular subpackages:** `pandapower.topology`, `pandapower.plotting`, `pandapower.shortcircuit`, `pandapower.estimation`, `pandapower.timeseries`, `pandapower.control` are separate namespaces.

## API Reference

| Domain | File | Description |
|--------|------|-------------|
| Network Creation | [api-network.md](references/api-network.md) | Buses, lines, transformers, loads, generators, switches, standard types, predefined networks |
| Power Flow | [api-powerflow.md](references/api-powerflow.md) | AC/DC power flow, OPF, 3-phase flow, short circuit, state estimation, result tables |
| Topology | [api-topology.md](references/api-topology.md) | Graph creation, connectivity, distance, island detection |
| Plotting | [api-plotting.md](references/api-plotting.md) | Matplotlib simple plot, custom collections, Plotly interactive, geodata |
| Toolbox | [api-toolbox.md](references/api-toolbox.md) | Element selection, network modification, file I/O, comparison, time series |
| Workflows | [workflows.md](references/workflows.md) | Complete working examples for common studies |

## Common Workflows

- **Build network from scratch:** See [api-network.md](references/api-network.md), then [workflows.md](references/workflows.md#quick-start)
- **AC power flow:** `pp.runpp(net)` — See [api-powerflow.md](references/api-powerflow.md#ac-power-flow)
- **Optimal power flow:** add cost functions + limits, then `pp.runopp(net)` — See [workflows.md](references/workflows.md#optimal-power-flow-workflow)
- **Short circuit:** `pp.shortcircuit.calc_sc(net, fault="3ph")` — See [api-powerflow.md](references/api-powerflow.md#short-circuit-calculation)
- **Plot network:** `pp.plotting.simple_plot(net)` — See [api-plotting.md](references/api-plotting.md)
- **Use benchmark network:** `pp.networks.case14()`, `pp.networks.mv_oberrhein()` — See [api-network.md](references/api-network.md#predefined-networks)
- **Time series simulation:** See [api-toolbox.md](references/api-toolbox.md#time-series) and [workflows.md](references/workflows.md#time-series-analysis)
- **N-1 contingency analysis:** See [workflows.md](references/workflows.md#contingency-analysis)

## Key Considerations

- **Convergence:** Always check `net.converged` after running power flow. Non-convergence often indicates voltage angle issues — try `init="dc"` or `algorithm="iwamoto_nr"`.
- **Geodata format:** Networks created before v2.7 use legacy `bus_geodata`/`line_geodata` DataFrames; run `pp.plotting.geo.convert_geodata_to_geojson(net)` to upgrade.
- **OPF requires cost functions:** `runopp()` will fail without at least one cost function (`create_poly_cost` or `create_pwl_cost`); all controlled elements need `min_p_mw`/`max_p_mw` limits.
- **Solver backends:** Install `lightsim2grid` or `power-grid-model` for 10-100x speedup on large networks; pandapower uses them automatically when available (`pip install pandapower[pgm]`).
- **Three-phase flow:** `runpp_3ph()` is imported from `pandapower.pf.runpp_3ph`, not the top-level namespace.
- **Short circuit:** `calc_sc()` lives in `pandapower.shortcircuit`; single-phase faults require transformer zero-sequence parameters (`vk0_percent`, `vkr0_percent`).
