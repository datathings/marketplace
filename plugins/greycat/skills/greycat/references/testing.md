# Testing

Run tests: `greycat test`

## Test Functions

Mark with `@test` decorator:

```gcl
@test
fn my_test_function() {
    Assert::isNull(null);
}
```

Output:
```
project::my_test_function ok (5us)
tests success: 1, failed: 0, skipped: 0
```

## Multiple Tests

Tests execute in definition order within a module:

```gcl
@test
fn test1() { }

@test
fn test2() { }
```

> Tests in same module share context. Changes to module variables persist between tests but are NOT saved to disk.

## Setup & Teardown

Run once per module, before/after all tests:

```gcl
var n: node<int?>;

fn setup() {
    n.set(1);  // Runs before any test
}

fn teardown() {
    // Cleanup after tests
}

@test
fn some_test() {
    Assert::equals(*n, 1);
    n.set(42);
}

@test
fn following_test() {
    Assert::equals(*n, 42);  // Sees change from previous test
}
```

## Test File Convention

Files ending with `_test.gcl` are excluded from `greycat build`:

```
├── project.gcl
├── src/
│   ├── api.gcl
│   ├── model.gcl
│   └── model_test.gcl      # Unit tests with source
└── test/
    └── api_test.gcl        # Integration tests
```

## Assert Methods

```gcl
Assert::equals(a, b);              // a == b (any type)
Assert::equalsd(a, b, epsilon);    // Float comparison
Assert::equalst(a, b, epsilon);    // Tensor comparison
Assert::isTrue(v);                 // v == true
Assert::isFalse(v);                // v == false
Assert::isNull(v);                 // v == null
Assert::isNotNull(v);              // v != null
```

## Return Codes

CI-friendly exit codes:

```bash
greycat test
echo $?
# 0 = success
# non-zero = failure
```

## Test Example

```gcl
@test
fn test_country_service() {
    var country = CountryService::create("Luxembourg", "LU");
    Assert::isNotNull(country);
    Assert::equals(country->name, "Luxembourg");

    var found = CountryService::find("Luxembourg");
    Assert::isNotNull(found);
    Assert::equals(found->code, "LU");
}

@test
fn test_fail() {
    throw "Expected failure";  // Will report as failed
}
```
