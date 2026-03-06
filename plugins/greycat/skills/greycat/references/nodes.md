# Nodes Deep Dive

## Transactions

Changes only committed after function completes successfully. Errors cause full rollback:

```gcl
var init: node<bool?>;

@expose
fn update_with_fail() {
    init.set(true);              // Attempt modification
    throw "An Error Occurred";   // Causes rollback - init stays null
}
```

## Object (heavy) vs Node (light)

```gcl
// Heavy - embeds full Country object in each City
type City { name: String; country: Country; }

// Light - only 64-bit reference, shared across cities
type City { name: String; country: node<Country>; }
```

## Modifying Node Content

Objects are passed by reference, primitives by value:

```gcl
// Objects - modifications persist
nCountry->name = "Foo";  // Works

// Primitives - must use .set()
var val_ref = node<int>{ 0 };
var resolved_val = val_ref.resolve();
resolved_val = 5;              // NO effect
val_ref.set(5);                // Works
```

## Sampling Large Collections

All node structures support static sampling:

```gcl
var result = nodeTime::sample(
    [timeSeries],      // array of node structures
    start, end,        // range
    1000,              // max points
    SamplingMode::adaptative,
    null, null         // maxDephasing, timezone
);
```

**SamplingMode**: `fixed` (fixed delta), `fixed_reg` (fixed + linear interpolation), `adaptative` (skip to limit results), `dense` (all elements, no sampling)
