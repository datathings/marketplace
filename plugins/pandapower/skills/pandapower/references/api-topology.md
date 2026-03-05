# Topology API

## Table of Contents
1. [Graph Creation](#graph-creation)
2. [Connectivity Analysis](#connectivity-analysis)
3. [Distance and Path Analysis](#distance-and-path-analysis)
4. [Network Splitting and Islands](#network-splitting-and-islands)

---

## Graph Creation

### `pp.topology.create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True, include_impedances=True, include_dclines=True, include_trafo3ws=True, multi=True, calc_branch_impedances=False, branch_impedance_unit="ohm", library="networkx", include_out_of_service=False) -> nx.MultiGraph`
**Description:** Converts a pandapower network into a NetworkX graph. Buses become nodes, branches (lines, trafos) become edges.
**Parameters:**
- `respect_switches`: If True, open switches create disconnected edges
- `include_lines`, `include_trafos`, etc.: Which element types to include as edges
- `multi`: If True, returns `MultiGraph` (allows parallel edges). If False, `Graph`
- `calc_branch_impedances`: Add impedance as edge weight
- `branch_impedance_unit`: `"ohm"` or `"pu"`
- `nogobuses`: Buses to exclude entirely
- `notravbuses`: Buses whose connected lines are ignored (terminal nodes)
- `library`: `"networkx"` or `"graph_tool"`
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
# Find buses not connected to any external grid
mg = top.create_nxgraph(net)
ext_grid_buses = set(net.ext_grid.bus.values)
all_buses = set(net.bus.index)

supplied = set()
for eg_bus in ext_grid_buses:
    supplied |= set(top.connected_component(mg, eg_bus))

unsupplied = all_buses - supplied
print(f"Unsupplied buses: {unsupplied}")
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

### `pp.topology.find_basic_open_loop_path(net, from_bus, to_bus, respect_switches=True) -> list`
**Description:** Finds the path between two buses in a radial network (assumes tree topology).
**Returns:** List of bus indices along the path.

### `pp.topology.get_end_buses(net, respect_switches=True) -> set`
**Description:** Returns all end buses (leaves) in the network - buses with only one connected branch.

### `pp.topology.get_feeders(net, respect_switches=True, notravbuses=None) -> generator`
**Description:** Yields feeders (subtrees fed from external grids) as sets of bus indices.
**Example:**
```python
for feeder_buses in top.get_feeders(net):
    print(f"Feeder with {len(feeder_buses)} buses")
```

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
