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

### `validate_input_data(input_data, calculation_type=None, symmetric=True) -> list[ValidationError]`
**Description:** Validates a single dataset. Returns a list of `ValidationError` objects; empty list means valid.

**Parameters:**
- `input_data: SingleDataset` — dict of numpy structured arrays
- `calculation_type: CalculationType | None` — if provided, allows fields unused by this calculation type to be missing
- `symmetric: bool` — `True` for symmetric calcs (relaxes zero-sequence parameter requirements)

**Example:**
```python
from power_grid_model import CalculationType
from power_grid_model.validation import validate_input_data

errors = validate_input_data(input_data, calculation_type=CalculationType.power_flow, symmetric=True)
if errors:
    for e in errors:
        print(e)  # human-readable description
else:
    model = PowerGridModel(input_data)
```

---

## validate_batch_data

### `validate_batch_data(input_data, update_data, calculation_type=None, symmetric=True) -> dict[int, list[ValidationError]]`
**Description:** Validates input_data combined with each scenario in update_data. Returns `{scenario_index: [errors]}`.

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
    component: str | list[str]              # e.g. "node" or ["node", "line"]
    field: str | list[str] | list[tuple]   # e.g. "id" or ["r1", "x1"]
    ids: list[int] | list[tuple[str, int]] # IDs of affected components
```

`str(error)` produces a concise human-readable message.

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
class ValidationException(Exception):
    errors: list[ValidationError] | dict[int, list[ValidationError]]
    name: str  # "input data" or "batch data"
```

Raised by `assert_valid_*` functions. `str(exception)` gives a multi-line summary.

---

## errors_to_string

### `errors_to_string(errors, name="data", details=True) -> str`
**Description:** Converts a list or dict of errors to a formatted multi-line string.

**Parameters:**
- `errors`: `list[ValidationError]` or `dict[int, list[ValidationError]]`
- `name`: Label used in the output string
- `details`: If `True`, include per-error detail lines

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
| `IdNotUniqueError` | Two components share the same `id` |
| `InvalidEnumValueError` | Enum attribute has an out-of-range integer value |
| `MissingValueError` | Required attribute is NaN/null |
| `NotGreaterThanZeroError` | Field requires `> 0` but is zero or negative |
| `NotBooleanError` | `status`, `from_status`, `to_status` not 0 or 1 |
| `InvalidIdError` | `from_node`, `to_node`, etc. refer to non-existent node |
| `MultiFieldValidationError` | Multiple inter-dependent fields have invalid combination |
| `NotObservableError` | (runtime) State estimation: insufficient sensors |
| `SparseMatrixError` | (runtime) Singular matrix — grid topology issue |
