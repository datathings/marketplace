# Workflows

## Table of Contents
1. [Quick Start — Basic Power Flow](#quick-start--basic-power-flow)
2. [Asymmetric Power Flow](#asymmetric-power-flow)
3. [State Estimation with Sensors](#state-estimation-with-sensors)
4. [Short Circuit Analysis](#short-circuit-analysis)
5. [Batch Time-Series Power Flow](#batch-time-series-power-flow)
6. [N-1 Contingency Analysis](#n-1-contingency-analysis)
7. [Automatic Tap Changing](#automatic-tap-changing)
8. [Validation Before Calculation](#validation-before-calculation)
9. [Serialization — Load and Save Datasets](#serialization--load-and-save-datasets)
10. [Columnar Output Format](#columnar-output-format)

---

## Quick Start — Basic Power Flow

Minimal example: 10 kV network with one line, one source, one symmetric load.

```
node_1 ---line_3--- node_2
 |                    |
source_5            sym_load_4
```

```python
from power_grid_model import PowerGridModel, LoadGenType, initialize_array

# --- Build input data ---
node = initialize_array('input', 'node', 2)
node['id'] = [1, 2]
node['u_rated'] = [10.5e3, 10.5e3]

line = initialize_array('input', 'line', 1)
line['id'] = [3]
line['from_node'] = [1]
line['to_node']   = [2]
line['from_status'] = [1]
line['to_status']   = [1]
line['r1'] = [0.25]
line['x1'] = [0.2]
line['c1'] = [10e-6]
line['tan1'] = [0.0]
line['i_n'] = [1000]

sym_load = initialize_array('input', 'sym_load', 1)
sym_load['id'] = [4]
sym_load['node'] = [2]
sym_load['status'] = [1]
sym_load['type'] = [LoadGenType.const_power]
sym_load['p_specified'] = [2e6]
sym_load['q_specified'] = [0.5e6]

source = initialize_array('input', 'source', 1)
source['id'] = [5]
source['node'] = [1]
source['status'] = [1]
source['u_ref'] = [1.0]

input_data = {
    'node': node,
    'line': line,
    'sym_load': sym_load,
    'source': source,
}

# --- Create model and run ---
model = PowerGridModel(input_data, system_frequency=50.0)
result = model.calculate_power_flow()

# --- View results ---
import pandas as pd
print(pd.DataFrame(result['node']))
# id  energized    u_pu            u      u_angle
# 1   1         0.999964  10499.6  -0.000198
# 2   1         0.994801  10445.4  -0.003096

print(pd.DataFrame(result['line']))
# id  energized  p_from  q_from  i_from  loading  ...
```

---

## Asymmetric Power Flow

Full three-phase unbalanced analysis. Requires zero-sequence parameters for lines/transformers.

```python
from power_grid_model import PowerGridModel, LoadGenType, initialize_array
import numpy as np

node = initialize_array('input', 'node', 2)
node['id'] = [1, 2]
node['u_rated'] = [10.5e3, 10.5e3]

line = initialize_array('input', 'line', 1)
line['id'] = [3]
line['from_node'] = [1]; line['to_node'] = [2]
line['from_status'] = [1]; line['to_status'] = [1]
line['r1'] = [0.25]; line['x1'] = [0.2]
line['c1'] = [10e-6]; line['tan1'] = [0.0]
# zero-sequence params required for asymmetric
line['r0'] = [0.5]; line['x0'] = [0.4]
line['c0'] = [5e-6]; line['tan0'] = [0.0]

# Asymmetric load: per-phase power (3-element array)
asym_load = initialize_array('input', 'asym_load', 1)
asym_load['id'] = [4]
asym_load['node'] = [2]
asym_load['status'] = [1]
asym_load['type'] = [LoadGenType.const_power]
asym_load['p_specified'] = [[1e6, 0.8e6, 1.2e6]]  # phases a, b, c
asym_load['q_specified'] = [[0.3e6, 0.2e6, 0.4e6]]

source = initialize_array('input', 'source', 1)
source['id'] = [5]; source['node'] = [1]
source['status'] = [1]; source['u_ref'] = [1.0]

input_data = {'node': node, 'line': line, 'asym_load': asym_load, 'source': source}
model = PowerGridModel(input_data)

# Asymmetric calculation
result = model.calculate_power_flow(symmetric=False)
# result['node']['u'] shape: (2, 3) — per-phase LN voltages
print(result['node']['u'])   # [[u_a1, u_b1, u_c1], [u_a2, u_b2, u_c2]]
```

---

## State Estimation with Sensors

```python
from power_grid_model import (
    PowerGridModel, LoadGenType, initialize_array,
    MeasuredTerminalType, CalculationMethod
)

# (Assume node, line, sym_load, source from Quick Start above)
# Add sensors to the input data

# Voltage sensor on node 1
sym_vsensor = initialize_array('input', 'sym_voltage_sensor', 1)
sym_vsensor['id'] = [10]
sym_vsensor['measured_object'] = [1]   # node 1
sym_vsensor['u_sigma'] = [10.0]        # measurement uncertainty: 10 V
sym_vsensor['u_measured'] = [10490.0]  # measured voltage

# Power sensor on the load
sym_psensor = initialize_array('input', 'sym_power_sensor', 1)
sym_psensor['id'] = [11]
sym_psensor['measured_object'] = [4]   # sym_load id=4
sym_psensor['measured_terminal_type'] = [MeasuredTerminalType.load]
sym_psensor['power_sigma'] = [1e4]     # 10 kVA uncertainty
sym_psensor['p_measured'] = [1.98e6]
sym_psensor['q_measured'] = [0.49e6]

input_data['sym_voltage_sensor'] = sym_vsensor
input_data['sym_power_sensor']   = sym_psensor

model = PowerGridModel(input_data)
result = model.calculate_state_estimation(
    calculation_method=CalculationMethod.iterative_linear
)
print(result['node'])  # estimated voltages
```

**Observability requirement:** At least one voltage sensor; total `n_measurements >= 2*n_nodes - 1`.

---

## Short Circuit Analysis

```python
from power_grid_model import (
    PowerGridModel, initialize_array,
    FaultType, FaultPhase, ShortCircuitVoltageScaling
)

# (Assume node, line, source from Quick Start — no loads needed for short circuit)
fault = initialize_array('input', 'fault', 1)
fault['id'] = [20]
fault['status'] = [1]
fault['fault_object'] = [2]          # fault at node 2
fault['fault_type'] = [FaultType.single_phase_to_ground]
fault['fault_phase'] = [FaultPhase.a]
fault['r_f'] = [0.0]   # bolted fault

input_data_sc = {'node': node, 'line': line, 'source': source, 'fault': fault}

model = PowerGridModel(input_data_sc)
result = model.calculate_short_circuit(
    short_circuit_voltage_scaling=ShortCircuitVoltageScaling.maximum
)
print(result['node'])   # u_pu, u_angle, u (LN, always 3-phase output)
print(result['fault'])  # i_f (A), i_f_angle (rad) per phase
print(result['line'])   # i_from, i_from_angle, i_to, i_to_angle
```

**Three-phase fault** (`FaultType.three_phase`) uses symmetric calculation internally; all other faults use asymmetric. Output always has 3-phase arrays.

**Batch short circuit** (multiple fault locations):
```python
fault_update = initialize_array('update', 'fault', (3, 1))
for i, node_id in enumerate([1, 2, 3]):
    fault_update[i, 0]['fault_object'] = node_id

result = model.calculate_short_circuit(update_data={'fault': fault_update})
# result['fault']['i_f'].shape == (3, 1, 3)  — 3 scenarios, 1 fault, 3 phases
```

---

## Batch Time-Series Power Flow

Efficient time-series simulation where load profiles change each timestep.

```python
from power_grid_model import PowerGridModel, LoadGenType, initialize_array
import numpy as np

# (Assume model from Quick Start)
n_timesteps = 100

# Dense batch: all loads updated each timestep
load_update = initialize_array('update', 'sym_load', (n_timesteps, 1))
load_update['id'] = [[4]]  # broadcast same ID

# Simulate varying load profile
t = np.linspace(0, 2 * np.pi, n_timesteps)
load_update['p_specified'] = (2e6 + 0.5e6 * np.sin(t))[:, np.newaxis]
load_update['q_specified'] = 0.5e6

update_data = {'sym_load': load_update}

# Run all 100 scenarios in parallel
result = model.calculate_power_flow(
    update_data=update_data,
    threading=0,   # use all hardware threads
)

# result['node']['u_pu'].shape == (100, 2) — 100 timesteps, 2 nodes
import matplotlib.pyplot as plt
plt.plot(result['node']['u_pu'][:, 1])  # node 2 voltage over time
plt.xlabel('Timestep'); plt.ylabel('Voltage (p.u.)')
plt.show()
```

---

## N-1 Contingency Analysis

Test grid stability when each line is taken out of service.

```python
from power_grid_model import PowerGridModel, initialize_array
import numpy as np

# (Assume model with lines having ids [3, 6, 9])
line_ids = np.array([3, 6, 9])
n_contingencies = len(line_ids)

# Sparse batch: each scenario disables one line
line_update_flat = initialize_array('update', 'line', n_contingencies)
line_update_flat['id']          = line_ids
line_update_flat['from_status'] = 0
line_update_flat['to_status']   = 0

# CSR indptr: scenario k disables data[k]
indptr = np.arange(n_contingencies + 1, dtype=np.int64)

update_data = {
    'line': {
        'indptr': indptr,
        'data': line_update_flat,
    }
}

result = model.calculate_power_flow(
    update_data=update_data,
    continue_on_batch_error=True,  # don't stop if a scenario is infeasible
)

# Check loading of all lines in all scenarios
loading = result['line']['loading']  # shape: (n_contingencies, n_lines)
overloaded = loading > 1.0
print(f"Overloaded (line, scenario): {np.argwhere(overloaded)}")
```

---

## Automatic Tap Changing

Simulate transformer tap regulators to maintain voltage within a band.

```python
from power_grid_model import (
    PowerGridModel, initialize_array, TapChangingStrategy,
    WindingType, BranchSide
)

# Build grid with transformer and tap regulator
transformer = initialize_array('input', 'transformer', 1)
transformer['id'] = [10]
transformer['from_node'] = [1]; transformer['to_node'] = [2]
transformer['from_status'] = [1]; transformer['to_status'] = [1]
transformer['u1'] = [110e3]; transformer['u2'] = [10.5e3]
transformer['sn'] = [40e6]; transformer['uk'] = [0.1]
transformer['pk'] = [100e3]; transformer['i0'] = [0.01]; transformer['p0'] = [50e3]
transformer['winding_from'] = [WindingType.wye_n]
transformer['winding_to']   = [WindingType.wye_n]
transformer['clock'] = [0]; transformer['tap_side'] = [BranchSide.from_side]
transformer['tap_pos'] = [0]; transformer['tap_min'] = [-10]; transformer['tap_max'] = [10]
transformer['tap_size'] = [1000]  # 1 kV per tap step

tap_reg = initialize_array('input', 'transformer_tap_regulator', 1)
tap_reg['id'] = [20]
tap_reg['regulated_object'] = [10]   # transformer id
tap_reg['status'] = [1]
tap_reg['control_side'] = [BranchSide.to_side]
tap_reg['u_set'] = [1.02]    # target: 1.02 p.u.
tap_reg['u_band'] = [0.02]   # ±0.01 p.u. acceptable

input_data['transformer'] = transformer
input_data['transformer_tap_regulator'] = tap_reg
model = PowerGridModel(input_data)

result = model.calculate_power_flow(
    tap_changing_strategy=TapChangingStrategy.any_valid_tap
)
print(result['transformer_tap_regulator'])  # tap_pos in output
```

---

## Validation Before Calculation

Best practice: always validate before constructing the model in production code.

```python
from power_grid_model import PowerGridModel, CalculationType
from power_grid_model.validation import (
    assert_valid_input_data,
    assert_valid_batch_data,
    ValidationException,
    errors_to_string,
)

# Option A: assert (raises on error)
try:
    assert_valid_input_data(
        input_data,
        calculation_type=CalculationType.power_flow,
        symmetric=True,
    )
except ValidationException as exc:
    print(exc)
    raise

# Option B: collect errors manually
from power_grid_model.validation import validate_input_data
errors = validate_input_data(input_data, symmetric=True)
if errors:
    print(errors_to_string(errors, name="my_grid", details=True))
else:
    model = PowerGridModel(input_data)

# Batch validation
from power_grid_model.validation import validate_batch_data
batch_errors = validate_batch_data(input_data, update_data, symmetric=True)
for scenario, errs in batch_errors.items():
    print(f"Scenario {scenario}: {errs}")
```

---

## Serialization — Load and Save Datasets

```python
from pathlib import Path
from power_grid_model.utils import (
    json_deserialize_from_file,
    json_serialize_to_file,
    msgpack_deserialize_from_file,
    msgpack_serialize_to_file,
)
from power_grid_model import DatasetType

# Save input data to JSON
json_serialize_to_file(
    Path("grid_input.json"),
    input_data,
    dataset_type=DatasetType.input,
    indent=2,
)

# Load input data from JSON
loaded_data = json_deserialize_from_file(Path("grid_input.json"))
model = PowerGridModel(loaded_data)

# Save result
result = model.calculate_power_flow()
json_serialize_to_file(Path("grid_result.json"), result)

# Binary (faster): msgpack
msgpack_serialize_to_file(Path("grid_input.msgpack"), input_data)
loaded = msgpack_deserialize_from_file(Path("grid_input.msgpack"))

# Filter on load (only read specific components/attributes)
filtered = json_deserialize_from_file(
    Path("grid_input.json"),
    data_filter={'node': None, 'line': ['r1', 'x1']}
)
```

---

## Columnar Output Format

For memory-efficient access to specific attributes in large batches.

```python
from power_grid_model import ComponentAttributeFilterOptions

# All components, only non-NaN attributes (columnar format)
result = model.calculate_power_flow(
    update_data=update_data,
    output_component_types=ComponentAttributeFilterOptions.relevant,
)
# result['node'] is now a dict: {'u_pu': array, 'u': array, ...}

# Specific components and attributes (dict form)
result = model.calculate_power_flow(
    update_data=update_data,
    output_component_types={
        'node': {'u_pu', 'u'},                       # columnar, 2 attributes
        'line': None,                                 # row-based (all attributes)
        'sym_load': ComponentAttributeFilterOptions.relevant,  # columnar, non-NaN
    }
)
```
