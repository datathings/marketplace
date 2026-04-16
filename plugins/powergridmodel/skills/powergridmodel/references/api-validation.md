# Validation API Reference

## Table of Contents
1. [Overview](#overview)
2. [validate_input_data](#validate_input_data)
3. [validate_batch_data](#validate_batch_data)
4. [assert_valid_input_data](#assert_valid_input_data)
5. [assert_valid_batch_data](#assert_valid_batch_data)
6. [assert_valid_data_structure](#assert_valid_data_structure)
7. [ValidationError](#validationerror)
8. [ValidationException](#validationexception)
9. [errors_to_string](#errors_to_string)

---

## Overview

```python
from power_grid_model.validation import (
    validate_input_data,
    validate_batch_data,
    assert_valid_input_data,
    assert_valid_batch_data,
    assert_valid_data_structure,
    ValidationError,
    ValidationException,
    errors_to_string,
)
```

Validation is **not** performed automatically during model construction or calculation for performance reasons.
Always validate before constructing `PowerGridModel` in production code — many data errors produce invalid results silently rather than raising exceptions.

---

## validate_input_data

### `validate_input_data(input_data, calculation_type=None, symmetric=True) -> list[ValidationError] | None`
**Description:** Validates a single dataset. Returns `None` if valid, or a list of `ValidationError` objects.

**Parameters:**
- `input_data: SingleDataset` — dict of numpy structured arrays
- `calculation_type: CalculationType | None` — if provided, allows fields unused by this calculation type to be missing
- `symmetric: bool` — `True` for symmetric calcs (relaxes zero-sequence parameter requirements)

**Example:**
```python
from power_grid_model import CalculationType
from power_grid_model.validation import validate_input_data

errors = validate_input_data(input_data, calculation_type=CalculationType.power_flow, symmetric=True)
if errors is not None:
    for e in errors:
        print(e)  # human-readable description
else:
    model = PowerGridModel(input_data)
```

---

## validate_batch_data

### `validate_batch_data(input_data, update_data, calculation_type=None, symmetric=True) -> dict[int, list[ValidationError]] | None`
**Description:** Validates input_data combined with each scenario in update_data. Returns `None` if valid, or `{scenario_index: [errors]}`.

**Note:** `input_data` alone may be invalid if missing values are always overridden in all batch scenarios.

**Parameters:**
- `input_data: SingleDataset`
- `update_data: BatchDataset` — 2-D or sparse batch arrays
- `calculation_type: CalculationType | None`
- `symmetric: bool`

**Example:**
```python
from power_grid_model.validation import validate_batch_data

batch_errors = validate_batch_data(input_data, update_data, symmetric=True)
if batch_errors is not None:
    for scenario_idx, errors in batch_errors.items():
        print(f"Scenario {scenario_idx}: {errors}")
```

---

## assert_valid_input_data

### `assert_valid_input_data(input_data, calculation_type=None, symmetric=True) -> None`
**Description:** Same as `validate_input_data` but raises `ValidationException` on failure instead of returning errors.

**Raises:** `ValidationException` with all validation errors accessible as `.errors`.

**Example:**
```python
from power_grid_model.validation import assert_valid_input_data, ValidationException

try:
    assert_valid_input_data(input_data, symmetric=True)
except ValidationException as exc:
    print(exc)  # prints summary of all errors
```

---

## assert_valid_batch_data

### `assert_valid_batch_data(input_data, update_data, calculation_type=None, symmetric=True) -> None`
**Description:** Same as `validate_batch_data` but raises `ValidationException` on failure.

**Example:**
```python
from power_grid_model.validation import assert_valid_batch_data

assert_valid_batch_data(input_data, update_data, symmetric=True)
# raises ValidationException if any scenario has errors
```

---

## assert_valid_data_structure

### `assert_valid_data_structure(data, dataset_type) -> None`
**Description:** Checks only the structural validity (array shapes and dtypes) without checking value ranges.

**Example:**
```python
from power_grid_model import DatasetType
from power_grid_model.validation import assert_valid_data_structure

assert_valid_data_structure(input_data, DatasetType.input)
```

---

## ValidationError

```python
class ValidationError:
    component: ComponentType | list[ComponentType] | None
    field: AttributeType | list[AttributeType] | list[tuple[ComponentType, AttributeType]] | None
    ids: list[int] | list[tuple[ComponentType, int]] | None
```

Subclass hierarchy:
- `SingleFieldValidationError` -- one component, one field (e.g. `MissingValueError`, `NotGreaterThanError`)
- `MultiFieldValidationError` -- one component, multiple fields (e.g. `TransformerClockError`, `FaultPhaseError`)
- `MultiComponentValidationError` -- multiple components (e.g. `MultiComponentNotUniqueError`)

`str(error)` produces a concise human-readable message. `error.get_context(id_lookup)` returns a dict with details.

**Example:**
```python
errors = validate_input_data(input_data)
for err in errors:
    print(f"Component: {err.component}, Field: {err.field}, IDs: {err.ids}")
    print(str(err))
```

---

## ValidationException

```python
class ValidationException(ValueError):
    errors: list[ValidationError] | dict[int, list[ValidationError]]
    name: str  # "input_data" or "update_data"
```

Raised by `assert_valid_*` functions. `str(exception)` gives a multi-line summary.

---

## errors_to_string

### `errors_to_string(errors, name="the data", details=False, id_lookup=None) -> str`
**Description:** Converts a list or dict of errors to a formatted multi-line string.

**Parameters:**
- `errors`: `list[ValidationError]` or `dict[int, list[ValidationError]]` or `None`
- `name`: Label used in the output string (default `"the data"`)
- `details`: If `True`, include per-error detail lines with object IDs
- `id_lookup`: `list[str]` or `dict[int, str]` to map integer IDs to human-readable names

**Example:**
```python
from power_grid_model.validation import validate_input_data, errors_to_string

errors = validate_input_data(input_data)
print(errors_to_string(errors, name="my grid", details=True))
```

---

## Common validation errors

| Error type | Typical cause |
|-----------|---------------|
| `NotUniqueError` | Two components of the same type share the same `id` |
| `MultiComponentNotUniqueError` | Two components of different types share the same `id` |
| `InvalidEnumValueError` | Enum attribute has an out-of-range integer value |
| `InvalidAssociatedEnumValueError` | Enum value invalid for the given associated component (e.g. Branch3Side for a transformer) |
| `MissingValueError` | Required attribute is NaN/null |
| `NotGreaterThanError` | Field requires `> ref` but is not |
| `NotGreaterOrEqualError` | Field requires `>= ref` but is not |
| `NotLessThanError` | Field requires `< ref` but is not |
| `NotBetweenError` | Field value not in valid range (exclusive) |
| `NotBetweenOrAtError` | Field value not in valid range (inclusive) |
| `NotBooleanError` | `status`, `from_status`, `to_status` not 0 or 1 |
| `InvalidIdError` | `from_node`, `to_node`, etc. refer to non-existent or wrong-type component |
| `IdNotInDatasetError` | Update data references an ID not present in input data |
| `SameValueError` | Two fields have equal values where they should differ (e.g. `from_node == to_node`) |
| `TwoValuesZeroError` | Two fields are both zero where at least one must be nonzero (e.g. `r1` and `x1`) |
| `InfinityError` | A field contains infinite values |
| `TransformerClockError` | Invalid clock number for given winding types |
| `FaultPhaseError` | Fault phase does not match the fault type |
| `PQSigmaPairError` | `p_sigma` and `q_sigma` must both be present or both absent |
| `InvalidVoltageRegulationError` | Voltage regulators on the same node have different `u_ref` values |
| `MixedCurrentAngleMeasurementTypeError` | Different angle measurement types on the same terminal |
| `MixedPowerCurrentSensorError` | Power and current sensors on the same terminal |
| `MissingVoltageAngleMeasurementError` | Global-angle current sensor without voltage angle measurement |
| `UnsupportedMeasuredTerminalType` | `measured_terminal_type` not in the allowed set for current sensors |

## Common runtime errors

| Error type | Typical cause |
|-----------|---------------|
| `NotObservableError` | State estimation: insufficient sensors for observability |
| `SparseMatrixError` | Singular matrix — grid topology or parameter issue |
| `IterationDiverge` / `MaxIterationReached` | Iterative solver did not converge |
| `AutomaticTapCalculationError` | Invalid tap regulator configuration for automatic tap changing |
| `UnsupportedRegulatorCombinationError` | Voltage regulators and transformer tap regulators in the same model |
| `UnsupportedVoltageRegulatorSourceCombinationError` | Source and voltage-regulated appliance on the same node |
| `InvalidCalculationMethod` | Unsupported calculation method for the given calculation type |
| `PowerGridBatchError` | One or more batch scenarios failed (access `.failed_scenarios`, `.errors`) |
