# Testing

Run: `greycat test` ([cli.md](cli.md) for options)

## Test Functions

```gcl
@test fn my_test_function() { Assert::isNull(null); }
// Output: project::my_test_function ok (5us)
//         tests success: 1, failed: 0, skipped: 0
```

## Multiple Tests

Execute in definition order. Share module context (changes persist between tests, not saved to disk).

## Setup & Teardown

Run once per module:
```gcl
var n: node<int?>;
fn setup() { n.set(1); }  // Before tests
fn teardown() { }         // After tests
@test fn some_test() { Assert::equals(*n, 1); n.set(42); }
@test fn following_test() { Assert::equals(*n, 42); }  // Sees prior change
```

## Running a Specific Test

Target a single test function using `module::function` syntax:
```bash
greycat test dfr_engine_test::test_dfr_variant
```
The module name is the test file name without the `.gcl` extension.

## Test Files

`*_test.gcl` excluded from `greycat build`:
```
src/model.gcl + src/model_test.gcl  # Unit tests with source
test/api_test.gcl                    # Integration tests
```

## Assert Methods

`equals(a, b)`, `equalsd(a, b, epsilon)`, `equalst(a, b, epsilon)`, `isTrue(v)`, `isFalse(v)`, `isNull(v)`, `isNotNull(v)`

## Exit Codes

`greycat test; echo $?` → 0=success, non-zero=failure

## Example

```gcl
@test fn test_country_service() {
    var country = CountryService::create("Luxembourg", "LU");
    Assert::isNotNull(country); Assert::equals(country->name, "Luxembourg");
    var found = CountryService::find("Luxembourg");
    Assert::isNotNull(found); Assert::equals(found->code, "LU");
}
@test fn test_fail() { throw "Expected failure"; }  // Reports as failed
```
