# Plotting API

## Table of Contents
1. [Simple Plot (Matplotlib)](#simple-plot)
2. [Custom Matplotlib Collections](#custom-matplotlib-collections)
3. [Plotly Interactive Plots](#plotly-interactive-plots)
4. [Geodata Handling](#geodata-handling)
5. [Power Flow Result Visualization](#power-flow-result-visualization)

---

## Simple Plot

### `pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_gens=False, plot_sgens=False, load_size=1.0, gen_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color="b", line_color="grey", dcline_color="c", trafo_color="k", ext_grid_color="y", switch_color="k", library="igraph", show_plot=True, ax=None) -> matplotlib.axes.Axes`
**Description:** Plots the network topology with minimal configuration. Generates artificial coordinates if none are stored.
**Parameters:**
- `respect_switches`: Show open switches as gaps
- `bus_size`, `line_width`, `trafo_size`, `ext_grid_size`: Relative element sizes
- `plot_loads`, `plot_gens`, `plot_sgens`: Whether to draw load/gen symbols
- `load_size`, `gen_size`, `sgen_size`: Relative sizes for load/gen/sgen symbols
- `plot_line_switches`: Whether to draw line switch symbols
- `bus_color`, `line_color`, `trafo_color`, `dcline_color`: Colors (matplotlib color strings)
- `library`: Layout algorithm for auto-coordinates: `"igraph"` or `"networkx"`
- `show_plot`: Call `plt.show()` at end
- `ax`: Existing matplotlib axes to draw on
**Returns:** `matplotlib.axes.Axes`
**Example:**
```python
import pandapower as pp
import pandapower.plotting as plot

net = pp.networks.mv_oberrhein()
ax = plot.simple_plot(net, show_plot=True)
```

### Plotting with custom axes (subplots)
```python
import matplotlib.pyplot as plt
import pandapower.plotting as plot

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot.simple_plot(net, ax=axes[0], show_plot=False)
axes[0].set_title("Network Topology")
# ... other plot on axes[1] ...
plt.tight_layout()
plt.show()
```

---

## Custom Matplotlib Collections

Build complex plots by composing `PatchCollection` objects:

### `create_bus_collection(net, buses=None, size=5., patch_type="circle", color=None, z=None, cmap=None, norm=None, infofunc=None, picker=False, **kwargs) -> PatchCollection`
**Description:** Creates matplotlib patch collection for buses.

### `create_line_collection(net, lines=None, line_width=1.0, infofunc=None, cmap=None, norm=None, picker=False, **kwargs) -> LineCollection`
**Description:** Creates matplotlib line collection for lines.

### `create_trafo_collection(net, trafos=None, size=None, color="k", picker=False, infofunc=None, cmap=None, **kwargs) -> PatchCollection`
**Description:** Creates patches for transformers (displayed as circles at midpoint).

### `create_ext_grid_collection(net, ext_grids=None, size=1., color="y", infofunc=None, picker=False, **kwargs) -> PatchCollection`
**Description:** Creates patches for external grid connections.

### `create_load_collection(net, loads=None, size=1., color="b", infofunc=None, picker=False, **kwargs) -> PatchCollection`

### `create_sgen_collection(net, sgens=None, size=1., color="g", infofunc=None, picker=False, **kwargs) -> PatchCollection`

### `draw_collections(collections, figsize=(10, 8), ax=None, plot_colorbars=True, set_aspect=True, show_plot=True) -> matplotlib.axes.Axes`
**Description:** Renders a list of collections onto a matplotlib axes.

### Complete custom plot example
```python
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

net = pp.networks.mv_oberrhein()
pp.runpp(net)

# Color lines by loading
cmap = plt.cm.RdYlGn_r
norm = Normalize(vmin=0, vmax=100)

bc = plot.create_bus_collection(net, size=80, color="blue", zorder=2)
lc = plot.create_line_collection(
    net, line_width=2.0, zorder=1,
    cmap=cmap, norm=norm,
    array=net.res_line.loading_percent
)
ec = plot.create_ext_grid_collection(net, size=200, zorder=3)
tc = plot.create_trafo_collection(net, size=120, zorder=3)

ax = plot.draw_collections([lc, bc, ec, tc], figsize=(12, 8))
ax.set_title("MV Network - Line Loading (%)")
plt.colorbar(lc, ax=ax, label="Loading [%]")
plt.show()
```

---

## Plotly Interactive Plots

### `pp.plotting.plotly.simple_plotly(net, respect_switches=True, use_line_geo=None, on_map=False, map_style="basic", figsize=1.0, aspectratio="auto", line_width=1.0, bus_size=10.0, ext_grid_size=20.0, bus_color="blue", line_color="grey", trafo_color="green", ext_grid_color="yellow", filename="temp-plot.html", auto_open=True, showlegend=True) -> plotly.Figure`
**Description:** Creates an interactive Plotly figure of the network. Opens in browser.
**Example:**
```python
import pandapower.plotting.plotly as pplotly

net = pp.networks.mv_oberrhein()
fig = pplotly.simple_plotly(net)
```

### `pp.plotting.plotly.vlevel_plotly(net, respect_switches=False, auto_open=True, ...) -> plotly.Figure`
**Description:** Interactive plot with buses colored by voltage level.

### `pp.plotting.plotly.pf_res_plotly(net, cmap="Jet", use_line_geo=None, on_map=False, projection=None, map_style="basic", figsize=1, aspectratio="auto", line_width=2, bus_size=10, climits_volt=(0.9, 1.1), climits_load=(0, 100), filename="temp-plot.html", auto_open=True) -> plotly.Figure`
**Description:** Interactive plot showing power flow results (loading and voltages as color maps).
**Parameters:**
- `cmap`: Colormap name (single colormap used for both lines and buses)
- `climits_volt`: Voltage color limits as `(min, max)` tuple in p.u.
- `climits_load`: Loading color limits as `(min, max)` tuple in percent
**Example:**
```python
import pandapower.plotting.plotly as pplotly

net = pp.networks.mv_oberrhein()
pp.runpp(net)
fig = pplotly.pf_res_plotly(net, cmap="Jet")
```

---

## Geodata Handling

### `pp.plotting.create_generic_coordinates(net, mg=None, library="igraph", respect_switches=False, geodata_table="bus", buses=None, overwrite=False, layout_engine="neato") -> None`
**Description:** Automatically generates bus geodata using a graph layout algorithm when no real coordinates exist. Modifies `net.bus.geo` in place.
**Parameters:**
- `library`: `"igraph"` (better layout) or `"networkx"`
- `respect_switches`: Respect switches when generating layout (default False)
- `buses`: Subset of buses to generate coordinates for (default: all)
- `overwrite`: Overwrite existing geodata (default False)
- `layout_engine`: GraphViz layout engine (default `"neato"`)
**Example:**
```python
net = pp.create_empty_network()
# ... add buses and branches ...
# No geodata yet - generate it
pp.plotting.create_generic_coordinates(net, library="igraph")
pp.plotting.simple_plot(net)
```

### Setting real geographic coordinates (GeoJSON)
```python
import json

# Set bus coordinates from (lat, lon) data
for bus_idx, (x, y) in bus_coordinates.items():
    geo = json.dumps({"coordinates": [x, y], "type": "Point"})
    net.bus.at[bus_idx, "geo"] = geo

# Set line coordinates (polyline)
for line_idx, coords in line_coords.items():
    geo = json.dumps({"coordinates": coords, "type": "LineString"})
    net.line.at[line_idx, "geo"] = geo
```

### `pp.plotting.geo.convert_geodata_to_geojson(net) -> None`
**Description:** Converts legacy `bus_geodata`/`line_geodata` DataFrames to GeoJSON format (for networks saved before pandapower v2.7).

### `pp.plotting.to_html(net, filename, ...) -> None`
**Description:** Exports an interactive HTML plot.

---

## Power Flow Result Visualization

### Voltage colormap plot
```python
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

net = pp.networks.mv_oberrhein()
pp.runpp(net)

# Buses colored by voltage magnitude
norm = Normalize(vmin=0.95, vmax=1.05)
cmap = plt.cm.RdYlGn

bc = plot.create_bus_collection(
    net, size=100, zorder=2,
    cmap=cmap, norm=norm,
    array=net.res_bus.vm_pu
)
lc = plot.create_line_collection(net, line_width=1.5, color="grey", zorder=1)

ax = plot.draw_collections([lc, bc], figsize=(10, 8))
plt.colorbar(bc, ax=ax, label="Voltage [p.u.]")
ax.set_title("Bus Voltages after Power Flow")
plt.show()
```

### `pp.plotting.plotting_toolbox.set_line_geodata_from_bus_geodata(net) -> None`
**Description:** Generates straight line geodata from bus positions when only bus coordinates are available.
