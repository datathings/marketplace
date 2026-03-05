# Nodes Deep Dive

## Definition

Nodes are persistent pointers (64-bit IDs) to locations in the Graph. Enable disc persistence for objects that would otherwise exist only in RAM.

```gcl
type Country { name: String; phoneCode: int; location: geo; }

fn main() {
    var luxembourg = Country { name: "Luxembourg", phoneCode: 352, location: geo{49.8153, 6.1296} };
    // RAM only - not persisted

    var n_lux = node<Country>{luxembourg};  // Now persisted to graph
    println(n_lux);  // {"_type":"core.node","ref":"0100000000000000"}
}
```

## Node Operations

```gcl
*n;              // dereference (resolve)
n->name;         // arrow: deref + field access (NOT (*n)->name)
n.resolve();     // explicit method
n->name = "X";   // modify object field
node<int>{0}.set(5);  // primitives use .set()
```

**Operators**:
| Operator | Description |
|----------|-------------|
| `.` | Access fields/methods on instance |
| `::` | Static access operator |
| `->` | Arrow: traverses graph to access attribute/function |

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

## Multi-References with Nodes

Objects can only belong to ONE container. For multiple indices, store node refs:

```gcl
var t_by_id = nodeList<node<T>>{};
var t_by_name = nodeIndex<String, node<T>>{};

var johnNode = node<T>{ T { id: 25473, name: "John" } };
t_by_id.set(johnNode->id, johnNode);
t_by_name.set(johnNode->name, johnNode);  // Both point to same node
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

## Indexed Collections

| Persisted | Key | In-Memory |
|-----------|-----|-----------|
| `node<T>` | — | `Array<T>`, `Map<K,V>` |
| `nodeList<node<T>>` | int | `Stack<T>`, `Queue<T>` |
| `nodeIndex<K, node<V>>` | hash | `Set<T>`, `Tuple<A,B>` |
| `nodeTime<node<T>>` | time | `Buffer`, `Table`, `Tensor` |
| `nodeGeo<node<T>>` | geo | `TimeWindow`, `SlidingWindow` |

**nodeTime** - Indexes by time with interpolation:
```gcl
var temps = nodeTime<float>{};
temps.setAt(t1, 20.5);

for (t: time, temp: float in temps[fromTime..toTime]) { println("Temperature was ${temp} at ${t}"); }

temps.remove(t1);  // Remove by time
```

**nodeList** - Indexes by integer (64-bit):
```gcl
var myStock = nodeList<Palette>{};
for (position: int, content: Palette in myStock[54..78]) { }

myStock.remove(55);  // Remove by index
```

**nodeGeo** - Indexes by geographical position:
```gcl
var buildings = nodeGeo<node<Building>>{};
for (position: geo, building: Building in buildings.filter(GeoBox{...})) { }

buildings.remove(position);  // Remove by geo position
```

**nodeIndex** - Indexes by any hashable key (usually String):
```gcl
var collaborators = nodeIndex<String, node<Person>>{};
collaborators.set("john", johnNode);
for (name: String, collab: node<Person> in collaborators) { }

collaborators.remove("john");  // Remove by key
```

> Use `remove` to delete entries. No `unset` method.

## Sampling Large Collections

All node structures support static sampling for performance:

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
