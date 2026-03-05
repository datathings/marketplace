---
name: powergridmodel
description: "Python library for steady-state distribution power system analysis (power flow, state estimation, short-circuit calculations). Use when working with the power-grid-model library to: (1) perform load flow or Newton-Raphson/iterative calculations on electrical grids, (2) run state estimation with sensor data, (3) compute IEC 60909 short-circuit currents, (4) execute batch/time-series or N-1 contingency simulations, or (5) work with grid component types (node, line, transformer, source, sym_load, etc.) and numpy structured arrays."
---

# Power Grid Model

## Overview

`power-grid-model` is a high-performance C++ library with a Python interface for steady-state distribution power system analysis. It operates on numpy structured arrays via a dictionary-based data model and supports symmetric (single-phase equivalent) and asymmetric (full three-phase) calculations.

**Version:** v1.13.10
**Language:** Python (C++ core)
**License:** Mozilla Public License 2.0 (MPL-2.0)
**Repo:** https://github.com/PowerGridModel/power-grid-model

## Quick Start

```python
from power_grid_model import PowerGridModel, LoadGenType, initialize_array

# Build component arrays
node = initialize_array('input', 'node', 2)
node['id'] = [1, 2]; node['u_rated'] = [10.5e3, 10.5e3]

line = initialize_array('input', 'line', 1)
line['id'] = [3]; line['from_node'] = [1]; line['to_node'] = [2]
line['from_status'] = [1]; line['to_status'] = [1]
line['r1'] = [0.25]; line['x1'] = [0.2]; line['c1'] = [10e-6]; line['tan1'] = [0.0]

sym_load = initialize_array('input', 'sym_load', 1)
sym_load['id'] = [4]; sym_load['node'] = [2]; sym_load['status'] = [1]
sym_load['type'] = [LoadGenType.const_power]
sym_load['p_specified'] = [2e6]; sym_load['q_specified'] = [0.5e6]

source = initialize_array('input', 'source', 1)
source['id'] = [5]; source['node'] = [1]; source['status'] = [1]; source['u_ref'] = [1.0]

input_data = {'node': node, 'line': line, 'sym_load': sym_load, 'source': source}

model = PowerGridModel(input_data, system_frequency=50.0)
result = model.calculate_power_flow()
# result['node'] contains: id, energized, u_pu, u_angle, u, p, q
```

## Core Concepts

- **Input format:** `dict[str, np.ndarray]` — structured arrays created via `initialize_array(data_type, component_type, shape)`. Dataset types: `input`, `update`, `sym_output`, `asym_output`, `sc_output`.
- **Symmetric vs asymmetric:** `symmetric=True` (default) solves a balanced single-phase equivalent; `symmetric=False` solves the full three-phase abc system and requires zero-sequence parameters.
- **Batch calculations:** Pass `update_data` (2-D dense array or sparse CSR dict) to run many scenarios in one call. Pass `list[BatchDataset]` for Cartesian product dimensions (time-series x contingency).
- **All IDs must be globally unique** across all component types in the same scenario.
- **NaN as sentinel:** Unset optional fields carry NaN/int-min sentinel values from `initialize_array`; do not leave required fields as NaN.

## API Reference

| Domain | File | Description |
|--------|------|-------------|
| Core API | [api-core.md](references/api-core.md) | `PowerGridModel`, `calculate_power_flow`, `calculate_state_estimation`, `calculate_short_circuit`, `initialize_array`, serialization |
| Components | [api-components.md](references/api-components.md) | All 22 component types, attributes, numpy dtypes, enumerations |
| Validation | [api-validation.md](references/api-validation.md) | `validate_input_data`, `validate_batch_data`, `ValidationError`, `ValidationException` |
| Batch | [api-batch.md](references/api-batch.md) | Dense/sparse batch format, Cartesian product, threading, error handling, output structure |
| Workflows | [workflows.md](references/workflows.md) | Complete working examples for all calculation types |

## Common Workflows

- **Basic power flow:** see [workflows.md](references/workflows.md#quick-start--basic-power-flow)
- **Asymmetric (3-phase) power flow:** see [workflows.md](references/workflows.md#asymmetric-power-flow)
- **State estimation with sensors:** see [workflows.md](references/workflows.md#state-estimation-with-sensors)
- **Short circuit (IEC 60909):** see [workflows.md](references/workflows.md#short-circuit-analysis)
- **Time-series batch simulation:** see [workflows.md](references/workflows.md#batch-time-series-power-flow)
- **N-1 contingency analysis:** see [workflows.md](references/workflows.md#n-1-contingency-analysis)
- **Automatic tap changing:** see [workflows.md](references/workflows.md#automatic-tap-changing)
- **Validation best practices:** see [workflows.md](references/workflows.md#validation-before-calculation)

## Key Considerations

- **Validate before constructing the model.** Many data errors (wrong impedances, isolated nodes) produce silently wrong results rather than exceptions. Use `assert_valid_input_data` or `validate_input_data` from `power_grid_model.validation`.
- **Short-circuit always returns asymmetric output** regardless of fault type. Three-phase faults still produce per-phase output arrays.
- **State estimation requires observability:** at minimum one voltage sensor and `n_measurements >= 2*n_nodes - 1`. Missing sensors cause `NotObservableError` at runtime.
- **Tap changer note:** `generic_branch.k` is the off-nominal ratio, not the voltage ratio — must be set explicitly for tap changers; the library does not compute it from node voltages.
- **Performance:** For batch time-series, use dense (independent) batches; for N-1 analysis, use sparse (dependent) batches. Enable `threading=0` for large batches to use all hardware cores.
- **Asymmetric calculations** require zero-sequence parameters (`r0`, `x0`, `c0`, etc.) for lines and appropriate winding configurations for transformers. `generic_branch` does not support asymmetric mode.
