# Core API Reference

## Table of Contents
1. [PowerGridModel class](#powergridmodel-class)
2. [calculate_power_flow](#calculate_power_flow)
3. [calculate_state_estimation](#calculate_state_estimation)
4. [calculate_short_circuit](#calculate_short_circuit)
5. [update](#update)
6. [get_indexer](#get_indexer)
7. [initialize_array](#initialize_array)
8. [power_grid_meta_data](#power_grid_meta_data)
9. [attribute_dtype / attribute_empty_value](#attribute_dtype--attribute_empty_value)
10. [Serialization utilities](#serialization-utilities)
11. [self_test](#self_test)

---

## PowerGridModel class

```python
from power_grid_model import PowerGridModel
```

### `PowerGridModel(input_data: SingleDataset, system_frequency: float = 50.0)`
**Description:** Constructs the model from a dictionary of numpy structured arrays.
**Parameters:**
- `input_data`: `dict[str, np.ndarray]` — keys are component type names (e.g. `"node"`, `"line"`), values are 1-D structured arrays produced by `initialize_array`.
- `system_frequency`: Power system frequency in Hz (default `50.0`).

**Properties:**
- `all_component_count: dict[ComponentType, int]` — count of each component type in the model.
- `batch_error: PowerGridBatchError | None` — batch error object after a batch run with `continue_on_batch_error=True`.

**Example:**
```python
model = PowerGridModel(input_data, system_frequency=50.0)
print(model)  # shows component counts
```

---

## calculate_power_flow

### `model.calculate_power_flow(...) -> Dataset`
**Description:** Runs steady-state power flow. Single scenario when `update_data` is omitted; batch when `update_data` is provided.

**Key parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symmetric` | `bool` | `True` | Symmetric (1-phase equiv.) or full 3-phase asymmetric |
| `calculation_method` | `CalculationMethod\|str` | `newton_raphson` | Algorithm to use |
| `error_tolerance` | `float` | `1e-8` | Convergence tolerance in p.u. (iterative methods) |
| `max_iterations` | `int` | `20` | Max iterations (iterative methods) |
| `update_data` | `BatchDataset\|list[BatchDataset]\|None` | `None` | Batch scenarios; `list` triggers Cartesian product |
| `threading` | `int` | `-1` | `-1`=sequential, `0`=all HW threads, `>0`=N threads |
| `output_component_types` | `ComponentAttributeMapping` | `None` | Filter output; `None`=all, set of types, or dict for columnar |
| `continue_on_batch_error` | `bool` | `False` | Continue when some batch scenarios fail |
| `tap_changing_strategy` | `TapChangingStrategy\|str` | `disabled` | Automatic tap optimization strategy |

**Returns:** `dict[str, np.ndarray]` — output arrays keyed by component type.

**Power flow algorithms:**
| Method | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| `newton_raphson` | moderate | high | Default, quadratic convergence |
| `iterative_current` | fast | high | Jacobi-like, linear convergence |
| `linear` | fastest | approx | Single-step, best for const-impedance loads |
| `linear_current` | fast | approx | Single iteration of iterative_current |

**Example:**
```python
from power_grid_model import PowerGridModel, CalculationMethod

# Single scenario
result = model.calculate_power_flow()

# Asymmetric
result = model.calculate_power_flow(symmetric=False)

# Batch with parallel threading
result = model.calculate_power_flow(
    update_data=update_data,
    threading=0,  # all hardware threads
    calculation_method=CalculationMethod.newton_raphson,
)

# Access node results
import pandas as pd
print(pd.DataFrame(result['node']))
# columns: id, energized, u_pu, u_angle, u, p, q
```

---

## calculate_state_estimation

### `model.calculate_state_estimation(...) -> Dataset`
**Description:** Weighted-least-squares state estimation from sensor measurements.
**Requires:** voltage sensors plus enough power/current sensors for observability (`n_measurements >= 2*n_nodes - 1`).

**Key parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symmetric` | `bool` | `True` | Symmetric or asymmetric |
| `calculation_method` | `CalculationMethod\|str` | `iterative_linear` | `iterative_linear` (fast) or `newton_raphson` (accurate) |
| `error_tolerance` | `float` | `1e-8` | Convergence tolerance |
| `max_iterations` | `int` | `20` | Max iterations |
| `update_data` | `BatchDataset\|None` | `None` | Batch scenarios |
| `threading` | `int` | `-1` | Thread count |

**State estimation algorithms:**
| Method | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| `iterative_linear` | fast | moderate | Default; pre-factorizes matrix |
| `newton_raphson` | slower | high | Exact; no linearization approximation |

**Raises:** `NotObservableError` or `SparseMatrixError` if system is not observable.

**Example:**
```python
from power_grid_model import CalculationMethod

result = model.calculate_state_estimation(
    calculation_method=CalculationMethod.newton_raphson
)
```

---

## calculate_short_circuit

### `model.calculate_short_circuit(...) -> Dataset`
**Description:** IEC 60909 short-circuit analysis. Always asymmetric output regardless of fault type.

**Key parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calculation_method` | `CalculationMethod\|str` | `iec60909` | Currently only `iec60909` |
| `short_circuit_voltage_scaling` | `ShortCircuitVoltageScaling\|str` | `maximum` | `minimum` or `maximum` short circuit currents |
| `update_data` | `BatchDataset\|None` | `None` | Batch fault scenarios |
| `threading` | `int` | `-1` | Thread count |

**Voltage scaling factor `c` (IEC 60909):**
| `U_nom` | `c_max` | `c_min` |
|---------|---------|---------|
| <= 1 kV | 1.10 | 0.95 |
| > 1 kV | 1.10 | 1.00 |

**Example:**
```python
from power_grid_model import ShortCircuitVoltageScaling

result = model.calculate_short_circuit(
    short_circuit_voltage_scaling=ShortCircuitVoltageScaling.maximum
)
# result['node'] has: id, energized, u_pu, u_angle, u
# result['line'] has: id, energized, i_from, i_from_angle, i_to, i_to_angle
# result['fault'] has: id, energized, i_f, i_f_angle
```

---

## update

### `model.update(*, update_data: Dataset) -> None`
**Description:** Permanently updates the model's component parameters in place. Useful when the base state changes between simulation runs.

**Warning:** If the update fails, the model is left in an invalid state and should be discarded.

**Example:**
```python
update = initialize_array('update', 'sym_load', 2)
update['id'] = [4, 7]
update['p_specified'] = [3e6, 1e6]
model.update(update_data={'sym_load': update})
```

---

## get_indexer

### `model.get_indexer(component_type: ComponentTypeLike, ids: np.ndarray) -> np.ndarray`
**Description:** Maps component IDs to array indices. Useful for filtering output arrays.

**Example:**
```python
ids = np.array([1, 2, 3])
idx = model.get_indexer('node', ids)
selected_nodes = result['node'][idx]
```

---

## initialize_array

```python
from power_grid_model import initialize_array
```

### `initialize_array(data_type, component_type, shape, empty=False) -> np.ndarray`
**Description:** Creates a numpy structured array pre-filled with NaN/null sentinel values for the given dataset type and component type.

**Parameters:**
- `data_type`: `"input"`, `"update"`, `"sym_output"`, `"asym_output"`, or `"sc_output"`
- `component_type`: Component type string (e.g. `"node"`, `"line"`, `"sym_load"`)
- `shape`: `int` for 1-D array; `(n_scenarios, n_components)` tuple for batch 2-D array
- `empty`: If `True`, skip NaN fill (slightly faster, but values will be undefined)

**Example:**
```python
node = initialize_array('input', 'node', 3)         # 3 nodes
load_batch = initialize_array('update', 'sym_load', (5, 2))  # 5 scenarios, 2 loads
```

---

## power_grid_meta_data

```python
from power_grid_model import power_grid_meta_data
```

**Description:** Dictionary `{DatasetType -> {ComponentType -> ComponentMetaData}}` containing numpy dtype, NaN values, and offsets for every component in every dataset type. Auto-generated from the C++ core at import time.

**Example:**
```python
dtype = power_grid_meta_data['input']['node'].dtype
print(dtype.names)  # ('id', 'u_rated')
```

---

## attribute_dtype / attribute_empty_value

```python
from power_grid_model import attribute_dtype, attribute_empty_value
```

### `attribute_dtype(data_type, component_type, attribute) -> np.dtype`
Returns the numpy dtype of a single attribute (useful for columnar data format).

### `attribute_empty_value(data_type, component_type, attribute) -> np.ndarray`
Returns the NaN sentinel value for a specific attribute.

**Example:**
```python
dtype = attribute_dtype('input', 'node', 'u_rated')   # dtype('<f8')
nan   = attribute_empty_value('input', 'node', 'id')  # -2147483648
```

---

## Serialization utilities

```python
from power_grid_model.utils import (
    json_deserialize_from_file,
    json_serialize_to_file,
    msgpack_deserialize_from_file,
    msgpack_serialize_to_file,
    get_dataset_scenario,
    get_dataset_batch_size,
)
```

| Function | Description |
|----------|-------------|
| `json_deserialize_from_file(path, data_filter)` | Load JSON → `Dataset` |
| `json_serialize_to_file(path, data, dataset_type, indent)` | Save `Dataset` → JSON |
| `msgpack_deserialize_from_file(path, data_filter)` | Load msgpack → `Dataset` |
| `msgpack_serialize_to_file(path, data, dataset_type)` | Save `Dataset` → msgpack |
| `get_dataset_scenario(dataset, scenario)` | Extract single-scenario slice from batch |
| `get_dataset_batch_size(dataset)` | Return number of scenarios in a batch |

---

## self_test

```python
from power_grid_model.utils import self_test
self_test()  # runs a minimal sanity check; raises PowerGridError on failure
```
