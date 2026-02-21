# Workflows

## Table of Contents
1. [Quick Start - Minimal Network](#quick-start)
2. [Distribution Network Analysis](#distribution-network-analysis)
3. [Optimal Power Flow](#optimal-power-flow-workflow)
4. [Short Circuit Study](#short-circuit-study)
5. [State Estimation](#state-estimation-workflow)
6. [Time Series Analysis](#time-series-analysis)
7. [Contingency Analysis](#contingency-analysis)
8. [Network from MATPOWER Case](#network-from-matpower)
9. [Topology Analysis](#topology-analysis-workflow)

---

## Quick Start

Minimal working example - create, solve, inspect:

```python
import pandapower as pp

# 1. Create network
net = pp.create_empty_network(f_hz=50.)

# 2. Add buses
b1 = pp.create_bus(net, vn_kv=110., name="HV Bus")
b2 = pp.create_bus(net, vn_kv=20., name="MV Bus")
b3 = pp.create_bus(net, vn_kv=0.4, name="LV Bus")

# 3. Add external grid (slack bus)
pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid")

# 4. Add transformer HV->MV
pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="25 MVA 110/20 kV")

# 5. Add transformer MV->LV
pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.4 MVA 20/0.4 kV")

# 6. Add load
pp.create_load(net, bus=b3, p_mw=0.25, q_mvar=0.05, name="Load")

# 7. Run power flow
pp.runpp(net)

# 8. Check results
print(net.converged)                             # True
print(net.res_bus[["vm_pu", "va_degree"]])       # Bus voltages
print(net.res_trafo[["loading_percent"]])         # Transformer loading
```

---

## Distribution Network Analysis

Typical MV/LV feeder analysis with standard types:

```python
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt

# Use a realistic benchmark network
net = pp.networks.create_cigre_network_mv()

# Run AC power flow
pp.runpp(net, algorithm="bfsw")  # backward/forward sweep for radial networks

if not net.converged:
    raise RuntimeError("Power flow did not converge!")

# Identify violations
v_min, v_max = 0.95, 1.05
voltage_violations = net.res_bus[
    (net.res_bus.vm_pu < v_min) | (net.res_bus.vm_pu > v_max)
]
line_overloads = net.res_line[net.res_line.loading_percent > 100.]
trafo_overloads = net.res_trafo[net.res_trafo.loading_percent > 100.]

print(f"Voltage violations: {len(voltage_violations)} buses")
print(f"Line overloads: {len(line_overloads)} lines")
print(f"Trafo overloads: {len(trafo_overloads)} trafos")
print(f"Max line loading: {net.res_line.loading_percent.max():.1f}%")
print(f"Min voltage: {net.res_bus.vm_pu.min():.4f} pu")

# Total losses
line_losses = net.res_line.pl_mw.sum()
trafo_losses = net.res_trafo.pl_mw.sum()
print(f"Total losses: {line_losses + trafo_losses:.3f} MW")

# Visualize
ax = plot.simple_plot(net, show_plot=False)
ax.set_title(f"CIGRE MV Network - Max loading: {net.res_line.loading_percent.max():.1f}%")
plt.show()
```

---

## Optimal Power Flow Workflow

Dispatch optimization with cost functions and constraints:

```python
import pandapower as pp

# Load an IEEE test case
net = pp.networks.case14()

# Run normal power flow first to get baseline
pp.runpp(net)
print(f"Baseline ext_grid injection: {net.res_ext_grid.p_mw.sum():.2f} MW")

# Set up OPF: make generators controllable with cost functions
for gen_idx in net.gen.index:
    net.gen.at[gen_idx, "controllable"] = True
    net.gen.at[gen_idx, "min_p_mw"] = 0.
    net.gen.at[gen_idx, "max_p_mw"] = net.gen.at[gen_idx, "p_mw"] * 2.

    # Linear generation cost
    pp.create_poly_cost(net, element=gen_idx, et="gen", cp1_eur_per_mw=50.)

# Make ext_grid controllable
for eg_idx in net.ext_grid.index:
    net.ext_grid.at[eg_idx, "controllable"] = True
    net.ext_grid.at[eg_idx, "min_p_mw"] = -1000.
    net.ext_grid.at[eg_idx, "max_p_mw"] = 1000.
    net.ext_grid.at[eg_idx, "min_q_mvar"] = -100.
    net.ext_grid.at[eg_idx, "max_q_mvar"] = 100.
    pp.create_poly_cost(net, element=eg_idx, et="ext_grid", cp1_eur_per_mw=60.)

# Add voltage constraints
net.bus["min_vm_pu"] = 0.95
net.bus["max_vm_pu"] = 1.05

# Add line loading constraints
net.line["max_loading_percent"] = 80.

# Run OPF
pp.runopp(net, verbose=True)

print(f"\nOPF result:")
print(f"  Converged: {net.converged}")
print(f"  Total cost: {net.res_cost:.2f} EUR/h")
print(net.res_gen[["p_mw", "q_mvar"]])
```

---

## Short Circuit Study

IEC 60909 short circuit calculation:

```python
import pandapower as pp
import pandapower.shortcircuit as sc

# Load network
net = pp.networks.case14()

# Three-phase maximum short circuit (for equipment ratings)
sc.calc_sc(net, fault="3ph", case="max", branch_results=True)
print("3-phase max SC currents (kA):")
print(net.res_bus_sc[["ikss_ka"]].round(3))
print(f"Max SC current: {net.res_bus_sc.ikss_ka.max():.3f} kA at bus {net.res_bus_sc.ikss_ka.idxmax()}")

# Single-phase ground fault (for protection coordination)
sc.calc_sc(net, fault="1ph", case="max")
print("\n1-phase ground fault SC currents (kA):")
print(net.res_bus_sc[["ikss_ka"]].round(3))

# Maximum with peak and thermal equivalent currents
sc.calc_sc(net, fault="3ph", case="max", ip=True, ith=True, tk_s=0.5)
print("\nWith peak current (ip) and thermal current (ith):")
print(net.res_bus_sc[["ikss_ka", "ip_ka", "ith_ka"]].round(3))
```

---

## State Estimation Workflow

Estimate grid state from imperfect SCADA measurements:

```python
import pandapower as pp
import pandapower.estimation as est

# Start with a known state
net = pp.networks.case14()
pp.runpp(net)  # "true" state

# Simulate noisy measurements
import numpy as np
np.random.seed(42)

# Add voltage measurements at a few buses
for bus_idx in [0, 3, 5, 8, 12]:
    true_v = net.res_bus.vm_pu.at[bus_idx]
    noisy_v = true_v + np.random.normal(0, 0.005)  # 0.5% std dev
    pp.create_measurement(net, "v", "bus", noisy_v, 0.01, element=bus_idx)

# Add power injection measurements at several buses
for bus_idx in [0, 2, 5, 9]:
    true_p = net.res_bus.p_mw.at[bus_idx]
    true_q = net.res_bus.q_mvar.at[bus_idx]
    pp.create_measurement(net, "p", "bus", true_p + np.random.normal(0, 0.5), 1.0, element=bus_idx)
    pp.create_measurement(net, "q", "bus", true_q + np.random.normal(0, 0.2), 0.5, element=bus_idx)

# Add line flow measurements
for line_idx in [0, 2, 4]:
    true_p = net.res_line.p_from_mw.at[line_idx]
    pp.create_measurement(net, "p", "line", true_p + np.random.normal(0, 0.5), 1.0,
                          element=line_idx, side="from")

# Run state estimation
success = est.estimate(net, algorithm="wls", init="flat")

if success:
    print("State estimation converged!")
    print(net.res_bus_est[["vm_pu", "va_degree"]].round(4))
    # Compare with true state
    err = (net.res_bus_est.vm_pu - net.res_bus.vm_pu).abs()
    print(f"Max voltage estimation error: {err.max():.4f} pu")
else:
    print("State estimation failed to converge")
```

---

## Time Series Analysis

Simulate a 24-hour period with varying loads:

```python
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries
import pandas as pd
import numpy as np

net = pp.networks.create_cigre_network_mv()
pp.runpp(net)  # Verify network is valid

# Create 24-hour load profiles
n_hours = 24
time_steps = range(n_hours)

# Typical daily load shape (normalized)
daily_shape = np.array([0.5, 0.45, 0.43, 0.42, 0.43, 0.47,
                        0.55, 0.70, 0.85, 0.92, 0.95, 0.96,
                        0.94, 0.93, 0.90, 0.88, 0.87, 0.90,
                        0.95, 0.97, 0.96, 0.90, 0.80, 0.65])

# Apply shape to each load
profiles = {}
for load_idx in net.load.index:
    base_p = net.load.at[load_idx, "p_mw"]
    profiles[f"load_{load_idx}"] = base_p * daily_shape * np.random.uniform(0.9, 1.1)

profiles_df = pd.DataFrame(profiles)
ds = DFData(profiles_df)

# Attach profiles via controllers
ConstControl(
    net, element="load", variable="p_mw",
    element_index=net.load.index.tolist(),
    profile_name=[f"load_{i}" for i in net.load.index],
    data_source=ds
)

# Configure output logging
ow = OutputWriter(net, output_path="./ts_results/", output_file_type=".json")
ow.log_variable("res_bus", "vm_pu")
ow.log_variable("res_line", "loading_percent")
ow.log_variable("res_trafo", "loading_percent")

# Run
run_timeseries(net, time_steps=time_steps)

# Load and analyze results
import os
bus_results = pd.read_json("./ts_results/res_bus/vm_pu.json")
print(f"Min voltage over 24h: {bus_results.min().min():.4f} pu")
print(f"Max voltage over 24h: {bus_results.max().max():.4f} pu")
```

---

## Contingency Analysis

N-1 security assessment:

```python
import pandapower as pp
import copy

net = pp.networks.case14()
pp.runpp(net)

# Define limits
V_MIN, V_MAX, I_MAX = 0.95, 1.05, 100.

def check_violations(net):
    """Returns dict of violations after power flow."""
    v = net.res_bus.vm_pu
    violations = {
        "v_under": net.bus.index[v < V_MIN].tolist(),
        "v_over": net.bus.index[v > V_MAX].tolist(),
        "line_overload": net.res_line.index[net.res_line.loading_percent > I_MAX].tolist(),
        "trafo_overload": net.res_trafo.index[net.res_trafo.loading_percent > I_MAX].tolist(),
    }
    return violations

# N-1 for lines
critical_contingencies = []

for line_idx in net.line.index:
    net_n1 = copy.deepcopy(net)
    net_n1.line.at[line_idx, "in_service"] = False

    try:
        pp.runpp(net_n1, init="dc")
        if net_n1.converged:
            v = check_violations(net_n1)
            if any(len(vlist) > 0 for vlist in v.values()):
                critical_contingencies.append({
                    "type": "line", "index": line_idx,
                    "name": net.line.at[line_idx, "name"],
                    "violations": v
                })
    except pp.powerflow.LoadflowNotConverged:
        critical_contingencies.append({
            "type": "line", "index": line_idx, "violations": "NON-CONVERGENT"
        })

print(f"Total contingencies tested: {len(net.line)}")
print(f"Critical (N-1 violations): {len(critical_contingencies)}")
for c in critical_contingencies:
    print(f"  Line {c['index']} ({c.get('name', '')}): {c['violations']}")
```

---

## Network from MATPOWER

```python
import pandapower as pp
from pandapower.converter import from_mpc

# From MATPOWER .mat file
net = from_mpc("path/to/case14.mat", f_hz=60.)

# Or directly from PYPOWER dict
from pypower.api import case14
ppc = case14()
net = pp.converter.from_ppc(ppc, f_hz=60.)

pp.runpp(net)
print(net.res_bus)
```

---

## Topology Analysis Workflow

Graph-based network analysis:

```python
import pandapower as pp
import pandapower.topology as top
import networkx as nx

net = pp.networks.mv_oberrhein()

# Build graph
mg = top.create_nxgraph(net, respect_switches=True, calc_branch_impedances=True)

# Basic graph properties
print(f"Buses (nodes): {mg.number_of_nodes()}")
print(f"Branches (edges): {mg.number_of_edges()}")
print(f"Is connected: {nx.is_connected(nx.Graph(mg))}")

# Find all end buses (leaf nodes)
end_buses = top.get_end_buses(net)
print(f"End buses: {end_buses}")

# Calculate electrical distance from HV substation
hv_bus = net.ext_grid.bus.iloc[0]
distances = top.calc_distance_to_bus(net, bus=hv_bus)
print(f"\nElectrical distances from bus {hv_bus} (km):")
print(distances.sort_values().head(10))

# Identify feeders
feeders = list(top.get_feeders(net, respect_switches=True))
print(f"\nNumber of feeders: {len(feeders)}")
for i, feeder in enumerate(feeders):
    print(f"  Feeder {i}: {len(feeder)} buses")

# Open loop detection
G = nx.Graph(mg)
cycles = nx.cycle_basis(G)
print(f"\nOpen loops (ring sections): {len(cycles)}")
```
