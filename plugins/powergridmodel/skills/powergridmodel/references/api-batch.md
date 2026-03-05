# Batch Calculations Reference

## Table of Contents
1. [Overview](#overview)
2. [Dense batch update](#dense-batch-update)
3. [Sparse batch update](#sparse-batch-update)
4. [Independent vs dependent batches](#independent-vs-dependent-batches)
5. [Cartesian product of datasets](#cartesian-product-of-datasets)
6. [Parallel computing](#parallel-computing)
7. [Batch error handling](#batch-error-handling)
8. [Output structure](#output-structure)
9. [Performance tips](#performance-tips)

---

## Overview

Batch calculations run many scenarios in a single call to `calculate_power_flow`, `calculate_state_estimation`, or `calculate_short_circuit` by passing `update_data`. The model's base state is always the state from `PowerGridModel(input_data)` — updates are applied on top but do not persist.

```python
result = model.calculate_power_flow(update_data=update_data)
# result['node'] shape: (n_scenarios, n_nodes)
```

---

## Dense batch update

All scenarios update the same set of components. Use a 2-D numpy array: `(n_scenarios, n_components_per_scenario)`.

```python
from power_grid_model import initialize_array

# 5 scenarios, each updating 2 loads
load_update = initialize_array('update', 'sym_load', (5, 2))
load_update['id'] = [[4, 7]]  # broadcast: same IDs in every scenario
load_update['p_specified'] = [
    [1e6, 2e6],
    [1.5e6, 2.5e6],
    [2e6, 3e6],
    [0.5e6, 1e6],
    [3e6, 0.5e6],
]
load_update['q_specified'] = 0.3e6  # broadcast constant

update_data = {'sym_load': load_update}
result = model.calculate_power_flow(update_data=update_data, threading=0)
# result['node'].shape == (5, n_nodes)
```

**Note:** IDs can be omitted (or set to NaN) for a dense update where every component of the type is updated. This is called a **uniform** update.

---

## Sparse batch update

Different scenarios update different components or different numbers of components. Use a dict with `"indptr"` and `"data"` keys (CSR-like sparse format).

```python
import numpy as np

# 3 scenarios: each disables exactly one of 3 lines
line_update_data = initialize_array('update', 'line', 3)
line_update_data['id']          = [3, 5, 8]
line_update_data['from_status'] = [0, 0, 0]
line_update_data['to_status']   = [0, 0, 0]

# indptr: scenario k uses data[indptr[k]:indptr[k+1]]
# scenario 0 → data[0:1], scenario 1 → data[1:2], scenario 2 → data[2:3]
indptr = np.array([0, 1, 2, 3], dtype=np.int64)

update_data = {
    'line': {
        'indptr': indptr,
        'data': line_update_data,
    }
}
result = model.calculate_power_flow(update_data=update_data)
```

---

## Independent vs dependent batches

**Dependent batch** (sparse): Only specifies components that change; unchanged components keep their base values. Good for N-1 contingency analysis.

**Independent batch** (dense): All components are specified for every scenario. Good for time-series load profiles where every load changes every timestep.

```python
# Independent (dense) — all 3 lines specified in each scenario
line_update = initialize_array('update', 'line', (3, 3))
line_update['id'] = [[3, 5, 8]]       # same IDs every scenario
line_update['from_status'] = 1         # default: all open
line_update['to_status']   = 1

# scenario i disables line i
for i in range(3):
    line_update[i, i]['from_status'] = 0
    line_update[i, i]['to_status']   = 0
```

---

## Cartesian product of datasets

Pass a `list[BatchDataset]` to `update_data` to compute all combinations of two independent batch dimensions. Output is flattened.

```python
# Time-series × contingency: 5 time steps × 3 N-1 scenarios = 15 total scenarios

# Dimension 1: 5 time steps of load profiles
load_update = initialize_array('update', 'sym_load', (5, 1))
load_update['id'] = [[4]]
load_update['p_specified'] = [[1e6], [2e6], [3e6], [2.5e6], [1.5e6]]

# Dimension 2: 3 N-1 line outages
line_update = initialize_array('update', 'line', (3, 3))
# ... fill as shown in independent batch section above

product_update = [
    {'sym_load': load_update},
    {'line': line_update},
]

result = model.calculate_power_flow(update_data=product_update)
# result['node'].shape == (15, n_nodes)   (5 * 3 = 15)
```

---

## Parallel computing

Control threading with the `threading` argument:

| Value | Behavior |
|-------|----------|
| `-1` | Sequential (default) |
| `0` | Use all hardware threads (recommended for large batches) |
| `> 0` | Use exactly N threads |

```python
result = model.calculate_power_flow(
    update_data=update_data,
    threading=0,   # recommended: hardware parallelism
)
```

**Note:** Parallel mode uses shared-memory multi-threading. Memory overhead is minimal since internal states are shared across threads.

---

## Batch error handling

By default, any scenario failure raises an exception. Use `continue_on_batch_error=True` to continue and collect failed scenarios.

```python
result = model.calculate_power_flow(
    update_data=update_data,
    continue_on_batch_error=True,
)

if model.batch_error is not None:
    print(f"Failed scenarios: {model.batch_error.failed_scenarios}")
    print(f"Error messages: {model.batch_error.errors}")
    # result arrays for failed scenarios contain NaN
```

**`PowerGridBatchError` attributes:**
- `.failed_scenarios: list[int]` — indices of failed scenarios
- `.errors: dict[int, Exception]` — per-scenario exceptions
- Access via `model.batch_error` after the calculation.

---

## Output structure

For batch calculations, output arrays are 2-D: `(n_scenarios, n_components)`.

```python
result = model.calculate_power_flow(update_data=update_data)

# Access per-scenario node voltages
node_voltages = result['node']['u_pu']  # shape: (n_scenarios, n_nodes)

# Extract one scenario
from power_grid_model.utils import get_dataset_scenario
scenario_3 = get_dataset_scenario(result, scenario=3)
# scenario_3['node'].shape == (n_nodes,)
```

---

## Performance tips

1. **Use independent batches for dense sampling** (time series). The Y-bus matrix stays constant across scenarios, so it only needs to be factorized once.

2. **Use sparse batches for sparse sampling** (N-1 analysis). Minimizes memory for components that rarely change.

3. **Enable threading** for large batches: `threading=0`.

4. **Pre-build update arrays** once and reuse. Avoid constructing `initialize_array` inside loops.

5. **Filter output**: Use `output_component_types` to only retrieve components you need:
   ```python
   result = model.calculate_power_flow(
       update_data=update_data,
       output_component_types={'node', 'line'},  # skip sensors, etc.
   )
   ```

6. **Columnar output** for large batches reduces memory copies:
   ```python
   from power_grid_model import ComponentAttributeFilterOptions
   result = model.calculate_power_flow(
       output_component_types=ComponentAttributeFilterOptions.relevant
   )
   ```
