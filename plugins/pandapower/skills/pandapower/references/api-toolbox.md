# Toolbox API

## Table of Contents
1. [Element Selection](#element-selection)
2. [Network Modification](#network-modification)
3. [File I/O](#file-io)
4. [Network Comparison](#network-comparison)
5. [Power Factor Utilities](#power-factor-utilities)
6. [Result Inspection](#result-inspection)
7. [Diagnostic](#diagnostic)
8. [Groups](#groups)
9. [Time Series](#time-series)

---

## Element Selection

### `pp.get_element_index(net, element_type, name, exact_match=True) -> int`
**Description:** Returns index of an element by name.
**Example:**
```python
bus_idx = pp.get_element_index(net, "bus", "HV Substation")
line_idx = pp.get_element_index(net, "line", "Feeder 1", exact_match=True)
```

### `pp.get_element_indices(net, element_type, name, exact_match=True) -> list`
**Description:** Returns list of indices (multiple element types supported).
**Example:**
```python
# Get multiple specific buses
indices = pp.get_element_indices(net, "bus", ["Bus HV1", "Bus HV2", "Bus HV3"])

# Get all elements matching pattern
hv_lines = pp.get_element_indices(net, "line", "HV", exact_match=False)
```

### `pp.get_connected_elements(net, element_type, buses, respect_switches=True, respect_in_service=False) -> set`
**Description:** Returns indices of all elements connected to specified buses.
**Example:**
```python
# All lines connected to buses 0 and 1
lines = pp.get_connected_elements(net, "line", [0, 1])
# All loads connected to a feeder
loads = pp.get_connected_elements(net, "load", feeder_buses)
```

### `pp.pp_elements(bus=True, bus_elements=True, branch_elements=True, other_elements=True, cost_tables=False, res_elements=False) -> set`
**Description:** Returns set of all pandapower element table names.
**Parameters:**
- `cost_tables`: Include `"poly_cost"`, `"pwl_cost"` tables
- `res_elements`: Include result tables (`"res_bus"`, `"res_line"`, etc.)
**Example:**
```python
all_elements = pp.pp_elements()
# {'bus', 'load', 'sgen', 'gen', 'ext_grid', 'line', 'trafo', ...}
```

### `pp.next_bus(net, bus, element, et="line") -> int`
**Description:** Returns the bus at the other end of a branch element.

---

## Network Modification

### `pp.select_subnet(net, buses, include_switch_buses=False, include_results=False, keep_everything_else=False) -> pandapowerNet`
**Description:** Creates a subnet containing only selected buses and their connected elements.
**Example:**
```python
# Extract a feeder subnetwork
feeder_buses = {0, 1, 2, 3, 5}
subnet = pp.select_subnet(net, feeder_buses, include_results=True)
```

### `pp.merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9, **kwargs) -> pandapowerNet`
**Description:** Merges two pandapower networks into one. Elements keep their indices unless both nets have the same indices (net2 gets reindexed).
**Example:**
```python
combined = pp.merge_nets(net1, net2)
```

### `pp.drop_elements(net, element_type, element_index, **kwargs) -> None`
**Description:** Removes elements by index (also drops results, group entries, and associated elements like switches).
**Example:**
```python
pp.drop_elements(net, "line", [0, 3, 7])
pp.drop_elements(net, "load", [1])
```

### `pp.drop_buses(net, buses, drop_elements=True) -> None`
**Description:** Removes buses and optionally all connected elements.

### `pp.drop_inactive_elements(net, respect_switches=True) -> None`
**Description:** Removes all elements not in service AND elements connected to inactive buses.

### `pp.reindex_buses(net, bus_lookup) -> None`
**Description:** Renames bus indices according to a lookup dict.
**Example:**
```python
pp.reindex_buses(net, {0: 100, 1: 101, 2: 102})
```

### `pp.reindex_elements(net, element_type, new_indices=None, old_indices=None, lookup=None) -> None`
**Description:** Renames element indices. Provide either `new_indices`/`old_indices` or a `lookup` dict.

### `pp.create_continuous_bus_index(net, start=0, store_old_index=False) -> dict`
**Description:** Renumbers all buses to a continuous range starting from `start`. Returns old-to-new lookup dict.

### `pp.replace_ext_grid_by_gen(net, ext_grids=None, gen_indices=None, slack=False, cols_to_keep=None, add_cols_to_keep=None) -> None`
**Description:** Replaces external grid(s) with generator(s) (useful for islanding studies).

### `pp.replace_gen_by_ext_grid(net, gens=None, ext_grid_indices=None, cols_to_keep=None, add_cols_to_keep=None) -> None`
**Description:** Converts generators to external grids.

### `pp.close_switch_at_line_with_two_open_switches(net) -> set`
**Description:** Finds lines with open switches at both ends and closes one. Returns set of closed switch indices.

---

## File I/O

### Save / Load (JSON)
```python
# Save as JSON (recommended default format)
pp.to_json(net, "network.json")
net2 = pp.from_json("network.json")
```

### Save / Load (pickle)
```python
pp.to_pickle(net, "network.p")
net2 = pp.from_pickle("network.p")
```

### Save / Load (Excel)
```python
pp.to_excel(net, "network.xlsx", include_empty_tables=False, include_results=True)
net2 = pp.from_excel("network.xlsx")
```

### Save / Load (SQLite / PostgreSQL)
```python
from pandapower import to_sqlite, from_sqlite
to_sqlite(net, "networks.db", table_name="my_network")
net2 = from_sqlite("networks.db", table_name="my_network")

from pandapower import to_postgresql, from_postgresql
to_postgresql(net, service="host=localhost dbname=grids", table_name="net1")
```

### MATPOWER / PYPOWER
```python
# Import from MATPOWER case format
from pandapower.converter import from_mpc
net = from_mpc("case14.mat")          # .mat file
net = pp.converter.from_ppc(ppc_dict)  # PYPOWER dict

# Export to MATPOWER
from pandapower.converter import to_mpc
mpc = to_mpc(net)
```

### CIM
```python
from pandapower.converter import from_cim
net = from_cim("grid.xml")
```

---

## Network Comparison

### `pp.nets_equal(net1, net2, check_only_results=False, check_without_results=False, exclude_elms=None, name_selection=None, **kwargs) -> bool`
**Description:** Checks whether two networks are equal (compares all DataFrames). Keys starting with `"_"` and `"et"` (elapsed time) are ignored.
**Example:**
```python
net_orig = pp.networks.case14()
net_copy = net_orig.deepcopy()
print(pp.nets_equal(net_orig, net_copy))  # True
```

### `pp.dataframes_equal(df1, df2, ignore_index_order=True, **kwargs) -> bool`
**Description:** Compares two DataFrames with floating point tolerance. Use `atol` kwarg for tolerance.

---

## Power Factor Utilities

### `pp.lf_info(net, numv=1, numi=2) -> None`
**Description:** Prints summary of max/min voltages and max line/trafo loading to logger.

### `pp.toolbox.power_factor`
```python
from pandapower.toolbox.power_factor import (
    signing_system_value,
    cosphi_to_mvar,
    mvar_to_cosphi,
    dataframe_to_cosphi
)

# Convert power factor to reactive power
q_mvar = cosphi_to_mvar(p_mw=10.0, cosphi=0.95, qmode="underexcited")

# Convert reactive power to power factor
cosphi, qmode = mvar_to_cosphi(p_mw=10.0, q_mvar=3.29)
```

---

## Result Inspection

### `pp.opf_task(net, delta_pq=1e-3, keep=False, log=True) -> dict`
**Description:** Prints and returns a summary of OPF flexibilities and constraints.

### Accessing result tables
```python
pp.runpp(net)

# Maximum loading
max_line_loading = net.res_line.loading_percent.max()
max_trafo_loading = net.res_trafo.loading_percent.max()

# Voltage statistics
print(f"Min voltage: {net.res_bus.vm_pu.min():.4f} pu")
print(f"Max voltage: {net.res_bus.vm_pu.max():.4f} pu")

# Total generation and load
total_gen = net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum()
total_load = net.res_load.p_mw.sum()
losses = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
```

---

## Diagnostic

### `pp.diagnostic(net, report_style="detailed", warnings_only=False, return_result_dict=True, overload_scaling_factor=0.001, lines_min_length_km=0., lines_min_z_ohm=0., nom_voltage_tolerance=0.3, **kwargs) -> dict`
**Description:** Runs a comprehensive sanity check on the network and reports potential issues.

**Checks include:**
- Disconnected buses
- Overloaded elements (before power flow)
- Invalid parameters (negative R, wrong voltage levels)
- Missing standard types
- Impedance mismatches
- Open loop switching

**Example:**
```python
result = pp.diagnostic(net, report_style="detailed")
# Returns dict with categories: "errors", "warnings", "OK"
```

---

## Groups

### `pp.create_group(net, element_types, element_indices, name="", reference_columns=None, index=None, **kwargs) -> int`
**Description:** Creates a named group of elements (e.g., a feeder or substation).
**Parameters:**
- `element_types`: String or list of element type names (e.g., `["bus", "line", "load"]`)
- `element_indices`: List of lists of indices for each element type
- `reference_columns`: Optional column name(s) to reference instead of DataFrame index (e.g., `"name"`)
**Example:**
```python
group_idx = pp.create_group(
    net,
    element_types=["bus", "line", "load"],
    element_indices=[[0, 1, 2], [0, 1], [0]],
    name="Feeder A"
)
```

### `pp.isin_group(net, element_type, element_index, index=None, drop_empty_lines=True) -> bool`
### `pp.element_associated_groups(net, element_type, element_index, return_empties=True) -> set`

---

## Time Series

### Basic time series simulation
```python
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

import pandas as pd
import numpy as np

# Create load profiles (e.g., 24 hours, 1h resolution)
hours = 24
profiles = pd.DataFrame({
    "load_0": np.random.uniform(0.5, 1.5, hours),  # p_mw
    "load_1": np.random.uniform(0.2, 0.8, hours),
})

ds = DFData(profiles)

# Attach profiles to loads using ConstControl
ConstControl(net, element="load", variable="p_mw",
             element_index=[0, 1], profile_name=["load_0", "load_1"],
             data_source=ds)

# Define what to record
ow = OutputWriter(net, output_path="./results/", output_file_type=".json")
ow.log_variable("res_bus", "vm_pu")
ow.log_variable("res_line", "loading_percent")

# Run time series
run_timeseries(net, time_steps=range(hours))
```
