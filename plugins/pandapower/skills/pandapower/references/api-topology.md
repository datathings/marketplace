# Topology API

## Table of Contents
1. [Graph Creation](#graph-creation)
2. [Connectivity Analysis](#connectivity-analysis)
3. [Distance and Path Analysis](#distance-and-path-analysis)
4. [Network Splitting and Islands](#network-splitting-and-islands)

---

## Graph Creation

### `pp.topology.create_nxgraph(net, respect_switches=True, include_lines=True, include_impedances=True, include_dclines=True, include_trafos=True, include_trafo3ws=True, include_tcsc=True, include_vsc=True, include_line_dc=True, nogobuses=None, notravbuses=None, multi=True, calc_branch_impedances=False, branch_impedance_unit="ohm", library="networkx", include_out_of_service=False, include_switches=True, trafo_length_km=None, switch_length_km=None) -> nx.MultiGraph`
**Description:** Converts a pandapower network into a NetworkX graph. Buses become nodes, branches (lines, trafos) become edges.
**Parameters:**
- `respect_switches`: If True, open switches create disconnected edges
- `include_lines`, `include_trafos`, `include_impedances`, `include_dclines`, `include_trafo3ws`: Which element types to include
- `include_tcsc`, `include_vsc`, `include_line_dc`: Include FACTS/HVDC/DC line elements
- `include_switches`: Include bus-bus switches as edges
- `multi`: If True, returns `MultiGraph` (allows parallel edges). If False, `Graph`
- `calc_branch_impedances`: Add impedance as edge weight
- `branch_impedance_unit`: `"ohm"` or `"pu"`
- `nogobuses`: Buses to exclude entirely
- `notravbuses`: Buses whose connected lines are ignored (terminal nodes)
- `library`: `"networkx"` or `"graph_tool"`
- `trafo_length_km`, `switch_length_km`: Optional edge weights for trafos/switches
**Returns:** `networkx.MultiGraph`
**Example:**
```python
import pandapower as pp
import pandapower.topology as top

net = pp.networks.mv_oberrhein()
mg = top.create_nxgraph(net, respect_switches=True)

# Use any NetworkX algorithm
import networkx as nx
print(nx.is_connected(mg))
print(nx.number_connected_components(mg))
```

---

## Connectivity Analysis

### `pp.topology.connected_component(mg, bus, notravbuses=[]) -> generator`
**Description:** Yields all buses reachable from a given bus via depth-first search.
**Parameters:**
- `mg`: NetworkX graph from `create_nxgraph()`
- `bus`: Starting bus index
- `notravbuses`: Buses that stop traversal (useful for isolating regions)
**Returns:** Generator of connected bus indices.
**Example:**
```python
mg = top.create_nxgraph(net)
# Find all buses connected to bus 0
connected = set(top.connected_component(mg, bus=0))
print(f"Buses connected to bus 0: {connected}")
```

### `pp.topology.connected_components(mg, notravbuses=set()) -> generator`
**Description:** Yields all isolated groups (connected components) in the graph.
**Parameters:**
- `mg`: NetworkX graph
- `notravbuses`: Set of bus indices to treat as boundaries
**Returns:** Generator of sets of bus indices.
**Example:**
```python
mg = top.create_nxgraph(net)
components = list(top.connected_components(mg))
print(f"Number of isolated islands: {len(components)}")
for i, comp in enumerate(components):
    print(f"  Island {i}: buses {comp}")
```

### Detect unsupplied buses
```python
# Using the built-in function (recommended)
unsupplied = top.unsupplied_buses(net, respect_switches=True)
print(f"Unsupplied buses: {unsupplied}")

# Or manually via connected_component
mg = top.create_nxgraph(net)
ext_grid_buses = set(net.ext_grid.bus.values)
all_buses = set(net.bus.index)

supplied = set()
for eg_bus in ext_grid_buses:
    supplied |= set(top.connected_component(mg, eg_bus))

unsupplied_manual = all_buses - supplied
print(f"Unsupplied buses: {unsupplied_manual}")
```

---

## Distance and Path Analysis

### `pp.topology.calc_distance_to_bus(net, bus, respect_switches=True, nogobuses=None, notravbuses=None, weight="weight", g=None) -> pd.Series`
**Description:** Calculates shortest distances from a source bus to all connected buses.
**Parameters:**
- `bus`: Source bus index
- `respect_switches`: Open switches create disconnections
- `weight`: Edge attribute to use as distance (`"weight"` = km, `None` = topological hops)
- `g`: Pre-computed graph (reuse for performance)
**Returns:** `pd.Series` indexed by bus index with distance values.
**Example:**
```python
# Calculate electrical distance from substation (bus 0)
distances = top.calc_distance_to_bus(net, bus=0, weight="weight")
print(distances.sort_values())
```

### `pp.topology.unsupplied_buses(net, mg=None, slacks=None, respect_switches=True) -> set`
**Description:** Finds buses not connected to any slack bus (ext_grid, slack gen, or slack VSC).
**Parameters:**
- `mg`: Pre-computed graph (optional)
- `slacks`: Set of slack bus indices (default: auto-detect from ext_grid, gen.slack, vsc)
**Returns:** Set of unsupplied bus indices.
**Example:**
```python
unsupplied = top.unsupplied_buses(net)
print(f"Unsupplied buses: {unsupplied}")
```

### `pp.topology.determine_stubs(net, roots=None, mg=None, respect_switches=False) -> set`
**Description:** Finds stub buses/lines. Writes `net.bus["on_stub"]` and `net.line["is_stub"]` columns.
**Parameters:**
- `roots`: Bus indices to exclude as roots (default: ext_grid buses)
**Returns:** Set of stub bus indices.

### `pp.topology.lines_on_path(mg, path) -> list`
**Description:** Finds all line indices connecting a given path of buses.

### `pp.topology.elements_on_path(mg, path, element="line") -> list`
**Description:** Finds all element indices of given type connecting a path of buses.

---

## Network Splitting and Islands

### Detect islands after contingency
```python
import pandapower as pp
import pandapower.topology as top

net = pp.networks.mv_oberrhein()

# Simulate line outage
net.line.at[5, "in_service"] = False

# Check for new islands
mg = top.create_nxgraph(net, respect_switches=True)
components = list(top.connected_components(mg))

if len(components) > 1:
    print(f"Network split into {len(components)} islands after outage")
    for island in components:
        has_slack = bool(set(net.ext_grid.bus).intersection(island))
        print(f"  Island buses {island} - {'has slack' if has_slack else 'NO SLACK (unsupplied)'}")
```

### Check radiality
```python
# Is the network radial (tree topology)?
mg = top.create_nxgraph(net, respect_switches=True)
G = nx.Graph(mg)  # Convert MultiGraph to Graph
is_radial = nx.is_tree(G)
print(f"Network is radial: {is_radial}")
```

### Identify parallel paths (meshed sections)
```python
# Find all loops using cycle basis
import networkx as nx
mg = top.create_nxgraph(net, respect_switches=False)
cycles = nx.cycle_basis(nx.Graph(mg))
print(f"Number of loops: {len(cycles)}")
for cycle in cycles:
    print(f"  Loop: {cycle}")
```
