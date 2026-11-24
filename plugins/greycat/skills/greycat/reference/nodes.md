# Node collections

Reach here when code stores values in the graph through a node type: `node`, `nodeTime`, `nodeList`, `nodeIndex`,
`nodeSet`, `nodeIndexSet`, `nodeTimeSet`, `nodeGeo` or `nodeGeoSet`, reads them back by key, time, position or point,
loops over them, samples them for a chart, or needs to know how concurrent tasks writing to one merge.

## Contents

- Live nodes and writes inside a loop
- node<T>
- nodeTime<T>
- nodeList<T>
- nodeIndex<K, V>
- nodeSet<V>
- nodeIndexSet<K, V>
- nodeTimeSet<T>
- nodeGeo<T>
- nodeGeoSet<T>
- Sampling node tags

## Live nodes and writes inside a loop

These types persist across worker restarts. They are the bridge between in-memory logic and the GreyCat graph database.

A value fetched from the graph (`idx.get(k)`, `n.resolve()`, a `for ... in nodeIndex` loop) is a **live node**: mutate its fields in place and the change persists on commit; do NOT `set` it back, which re-attaches an already-attached node and throws `object is already attached to another node`. `set` is for inserting a brand-new node. See [idioms.md](idioms.md), "Working with the graph".

While a `for` loop walks a `nodeTime`, `nodeList`, `nodeIndex`, `nodeSet`, `nodeIndexSet`, `nodeTimeSet`, `nodeGeo` or `nodeGeoSet`, it can set a new value at a key the node already holds. Adding or removing a key of that node (`setAt`, `set` or `add` with a new key or value, `setAll`, `remove`, `removeAt`, `removeAllAt`, `removeAll`) throws `concurrent modification`, and so it does while a `nodeTimeCursor` over the node stands on one of its values. A node that holds a single value is the exception: the loop reads that value before its body runs, so the body can change the node. Collect the keys to add or remove, and change the node after the loop; other nodes can change inside it.

## node<T>

Singleton, lazily resolved.

```gcl
var n: node<User> = ...;
n.resolve();                       // T — the inner value
n.set(updated_user);               // mutate the pointed-to value
n->name;                           // == n.resolve().name (deref + access)
*n;                                // == n.resolve() — unary `*` is deref-shorthand
```

Tagged `@deref("resolve")` — the `->` operator calls `resolve()` then accesses the result.

## nodeTime<T>

Time-series of `T` values keyed by `time`.

```gcl
var s: nodeTime<float> = /* ... */;
s.setAt(time::now(), 1.0);
s.setAll(time::now(), [1.0, 2.0, 3.0], 1s);
s.getAt(t);                        // exact-match read; null if no entry
s.resolveAt(t);                    // closest-prior read
s.resolveTimeAt(t);                // the actual timestamp resolved to
s.firstTime();   s.first();
s.lastTime();    s.last();
s.prev(t);   s.next(t);
s.size();
s.rangeSize(from, to);
s.removeAt(t);   s.removeAll();

// Iteration with time-window query
for (t, v in s[from..to]) {} // from/to included
for (t, v in s[from..to[) {} // to excluded
for (t, v in s]from..to]) {} // from excluded
for (t, v in s]from..to[) {} // from/to execluded
```

Tagged `@iterable @deref("resolve")`.

## nodeList<T>

A list of values read and removed by position: every index is a position `0..size()-1`, and the values after a removed one shift down. There is no `set`: `greycat lint` reports a call to it as `unknown-member`, and `greycat build` still accepts it, so it fails only when it runs, with `type nodeList has no function named set`. Change an object value in place, or `remove` and `add`.

```gcl
var l = nodeList<float> {};
l.add(1.0);                        // append at position size()
l.add(2.0);
l.get(1);                          // 2.0; null outside 0..size()-1
l.resolve(5);                      // the last value past the end, null below 0
l.firstIndex();  l.first();        // 0, null when empty
l.lastIndex();   l.last();         // size() - 1, null when empty
l.rangeSize(0, 1);                 // positions in the range; reversed bounds are swapped, the range is clamped to the list
l.remove(0);                       // true; later values shift down, false outside the list
for (i, v in l[0..1]) { /* i counts positions; ]a..b], [a..b[ and descending l]4..2] too */ }
```

A list written by an earlier version with sparse keys (from `set`) reads by position: `get(0)` is the first stored value.

A position is not an id: it changes when an earlier value is removed. Code that keeps an index elsewhere uses a `nodeIndex<int, T>` with a counter it owns.

Two tasks appending to one list both keep their values, in merge order, and removals and appends do not conflict: the conflicts left are in [tasks.md](tasks.md#concurrent-appends-to-a-nodelist).

## nodeIndex<K, V>

Sorted index with O(log n) keyed access.

```gcl
var idx: nodeIndex<String, User> = /* ... */;
idx.set("alice", User { /* ... */ });
idx.get("alice");                  // User?
idx.search("alice", 10);           // closest-neighbor search
idx.remove("alice");
idx.size();
for (k, v in idx) { /* ... */ }
```

## nodeSet<V>

Set of unique values, generic over any `V`.

```gcl
var s = nodeSet<String> {};
s.add("red"); // true: "red" was not in the set yet
s.add("red"); // false: the set already holds it
s.contains("red"); // true
s.remove("red");
s.size();
s.removeAll();
// both loop variables bind the value
for (_, v in s) {
    println(v);
}
```

Two values are the same when they are deeply equal, the rule of a `nodeIndex` key: an `int`, `String`, enum entry or
other scalar by value, an object by its content, and a `node<T>` by reference. `null` is never an element: `add(null)`
throws, `contains(null)` is false and `remove(null)` does nothing. A loop walks the values in the order of their
encoding for a scalar `V` and in hash order otherwise. `->` on a `nodeSet` throws.

Tasks that commit to the same set at once never conflict. The merge applies what each side added and removed: a value
added on either side is present, a value removed on either side is absent. `removeAll()` removes the values the task
saw, so a value another task added at the same time stays. `fusion_by_key_last_win` changes nothing for a set, and
`no_fusion_fail_on_conflict` refuses it like any other structured node ([tasks.md](tasks.md#workers-tasks-and-jobs)).

## nodeIndexSet<K, V>

A multimap: each key holds a set of distinct values, as a `nodeIndex<K, nodeSet<V>>` would, without a node per key.

```gcl
var idx = nodeIndexSet<String, String> {};
idx.add("fruit", "apple"); // true: a new pair
idx.add("fruit", "pear"); // true: a second value under the key
idx.add("fruit", "apple"); // false: the key holds it already
idx.get("fruit"); // Array<String> {"apple", "pear"}
idx.get("nut"); // an empty Array
idx.contains("fruit", "pear"); // true
idx.size(); // 2: pairs, not keys
// one turn per pair, the values of a key one after the other
for (k, v in idx) {
    println("${k} ${v}");
}
idx.remove("fruit", "pear"); // one pair
idx.removeKey("fruit"); // the key and every value it holds
idx.removeAll();
```

A pair is a key and one of its values. Keys compare as `nodeIndex` keys do, values by deep equality as the elements of a
`nodeSet` do, and a pair is new when no pair with an equal key and an equal value is there. `null` is never a value:
`add(key, null)` throws `null can not be added to a nodeIndexSet`. A loop walks the keys in the order of a `nodeIndex`,
by encoding for a scalar key and by hash otherwise, the values of one key together and in no set order: a merge or a
removal can change it, and `get` returns them in that order. `->` on a `nodeIndexSet` throws. An object value edited in
place after `add` does not change the pair; change it with `remove(key, old)` then `add(key, new)`. `greycat lint` does
not know `nodeIndexSet` yet: it reports a loop over one as `not-iterable` and a module variable of the type as
`modvar-must-be-node-tag`, and `greycat build` and `greycat run` accept both.

A key with one value costs what it costs in a `nodeIndex`. Up to 32 values, a key keeps them in a chain that every `add`
to the key walks. Its 33rd value moves them into a `nodeSet` of their own, where `add`, `contains` and `remove` take
O(log n). The key keeps that set for the life of the index: `remove`, `removeKey` and `removeAll` empty it and leave it
in the store, and a later `add` fills it again. `NodeReader` lists the set as a `nodeSet`
([storage.md](storage.md#listing-every-node)).

Tasks that commit to the same index at once never conflict under the default strategies: the pairs merge one by one, as
the elements of a `nodeSet`. A pair added on either side is present and a pair removed on either side is absent, so two
values added under one key are both kept. `removeKey` and `removeAll` remove only the pairs the task saw: a pair another
task added at the same time stays. `fusion_by_key_last_win` changes nothing for it, and `no_fusion_fail_on_conflict`
refuses it like any other structured node ([tasks.md](tasks.md#workers-tasks-and-jobs)).

## nodeTimeSet<T>

A time series where each time holds a set of distinct values, such as events that happen at the same instant: the set
counterpart of `nodeTime`, as `nodeIndexSet` is of `nodeIndex`.

```gcl
var events = nodeTimeSet<String> {};
var t0 = '2026-01-01T00:00:00Z';
events.add(t0, "door"); // true: a new pair
events.add(t0, "alarm"); // true: a second value at the time
events.add(t0, "door"); // false: the time holds it already
events.add(t0 + 1min, "light");
events.getAt(t0); // Array<String> {"door", "alarm"}
events.getAt(t0 + 10s); // an empty Array: no value at exactly that time
events.resolveAt(t0 + 10s); // {"door", "alarm"}: the values of the latest time at or before it
events.resolveAt(t0 - 10s); // null: no time at or before it
events.resolveTimeValueAt(t0 + 2min); // (t0 + 1min, {"light"})
events.contains(t0, "alarm"); // true
events.size(); // 3: pairs, not times
events.rangeSize(t0, t0 + 1min); // 3
events.next(t0); // t0 + 1min: prev and next walk times, not values
// one turn per pair, in time order, the values of a time one after the other
for (t, v in events[t0..t0 + 1min]) {
    println("${t} ${v}");
}
events.remove(t0, "alarm"); // one pair
events.removeAllAt(t0); // the time and every value it holds
events.removeAll();
```

A pair is a time and one of its values. Values compare by deep equality, as the elements of a `nodeSet` do, and a pair
is new when its time holds no equal value. `null` is never a value: `add(t, null)` throws
`null can not be added to a nodeTimeSet`. There is no `setAt` or `setAll`: `add` keeps the values the time already
holds, and replacing them is `removeAllAt(t)` then `add(t, v)`.

The reads of a `nodeTime` that resolve a time return every value of the time they land on, as an `Array<T>`:
`resolve()` (at the time of the enclosing `at` scope), `resolveAt`, `resolveAtWithin`, `first` and `last`, and the
`Tuple<time, Array<T>>` of `resolveTimeValueAt`. They return `null` when they land on no time, never an empty Array.
`getAt(t)` returns the values at exactly `t`, an empty Array when there are none. `resolveTimeAt`, `firstTime`,
`lastTime`, `prev` and `next` walk the times. `size()` and `rangeSize(from, to)` count pairs, as many as a loop over
the same range yields, read from the tree's counts in O(log n).

A loop, and every range form of a `nodeTime` (`[a..b]`, `]a..b]`, `[a..b[`, `]a..b[`, descending `]b..a]`), runs once
per pair, in the order of the times (down for a descending range), the values of one time together and in no set order: a merge or a removal can change it, and
`getAt` returns them in that order. `->` and `*n` throw `arrow operator cannot be applied on nodeTimeSet`. A
`nodeTimeSet` is not a `nodeTime`: passed where one is expected, the call fails with
`argument of type 'nodeTimeSet' is not assignable to parameter ...`. An object value edited in place after `add` does
not change the pair; change it with `remove(t, old)` then `add(t, new)`. `greycat lint` does not know `nodeTimeSet`
yet: it reports a loop over one as `not-iterable` and a module variable of the type as `modvar-must-be-node-tag`, and
`greycat build` and `greycat run` accept both.

To walk the values of one time without building an Array, loop over the range from that time to itself. It reads
the values in place, a spilled time's too, and is faster than `getAt` (about 12% on 3 values per time); a time with no
value gives an empty loop. Keep `getAt` when the values must leave as an Array. As in any loop over the node, the body
cannot add or remove a value of it.

```gcl
for (_, v in events[t0..t0]) {
    println(v); // "door", then "alarm"
}
```

A time with one value takes the blocks it takes in a `nodeTime`, and a few bytes more. Past 32 values, a time keeps
its values in a `nodeSet` of its own, as a key of a `nodeIndexSet` does ([above](#nodeindexsetk-v)), which
`NodeReader` lists as a `nodeSet` ([storage.md](storage.md#listing-every-node)).

Tasks that commit to the same `nodeTimeSet` at once merge its pairs as those of a `nodeIndexSet`
([above](#nodeindexsetk-v)): adds and removes never conflict under the default strategies, and `removeAllAt` and
`removeAll` remove only the pairs the task saw. `no_fusion_fail_on_conflict` refuses it like any other structured
node.

## nodeGeo<T>

Spatial index keyed by `geo`.

```gcl
var g: nodeGeo<Station> = /* ... */;
g.set(geo { 49.6, 6.1 }, station);
g.get(geo { 49.6, 6.1 });
g.resolve(geo { 49.7, 6.0 });      // closest-prior read
g.rangeSize(geo { 49, 6 }, geo { 50, 7 });
for (pos, station in g[geo { 49, 6 }..geo { 50, 7 }]) { /* every key of the box, south-west first */ }
```

A range `g[a..b]` is the box spanned by `a` and `b` (any two opposite corners): it yields exactly the keys inside, in curve order, and `rangeSize` counts the same keys. Walking from the higher code, `g[b..a]`, runs the box backwards. No `GeoBox.contains` filter is needed on the keys.

## nodeGeoSet<T>

A spatial index where each point holds a set of distinct values, such as the shops of one address: the set
counterpart of `nodeGeo`. It follows the rules of a `nodeTimeSet` ([above](#nodetimesett)) on points: pairs, deep
equality, `add` and no `set`, `null` refused, `->` throwing, no subtype of `nodeGeo`, the spill past 32 values and the
merge.

```gcl
var shops = nodeGeoSet<String> {};
var square = geo { 49.6, 6.1 };
shops.add(square, "bakery"); // true: a new pair
shops.add(square, "pharmacy"); // true: a second value at the point
shops.add(geo { 49.61, 6.12 }, "school");
shops.get(square); // Array<String> {"bakery", "pharmacy"}
shops.get(geo { 1.0, 1.0 }); // an empty Array
shops.resolve(geo { 49.605, 6.11 }); // {"bakery", "pharmacy"}: the point nodeGeo::resolve lands on
shops.size(); // 3: pairs, not points
shops.rangeSize(geo { 49.0, 6.0 }, geo { 50.0, 7.0 }); // 3
// one turn per pair, in curve order, the values of a point one after the other
for (pos, shop in shops[geo { 49.0, 6.0 }..geo { 50.0, 7.0 }]) {
    println("${pos} ${shop}");
}
shops.search(square, 2); // bakery and pharmacy: one result per value
shops.removeAllAt(square); // the point and every value it holds
```

`get(index)` returns the values at exactly `index`, an empty Array when there are none. `resolve(index)` lands on the
point `nodeGeo::resolve` lands on and returns its values, `first()` and `last()` those of `firstIndex()` and
`lastIndex()`, each `null` when there is no point. There is no `resolveEntry`. `size()` reads the pair count in
O(log n). `rangeSize(from, to)` counts the pairs of the box, walking it as `nodeGeo::rangeSize` does. A loop and the
box range `g[a..b]` run once per pair, in the order of a `nodeGeo`, the values of one point together and in no set
order. `search(center, max)` returns one `SearchResult` per value, nearest first, each counting toward `max`, so `max`
can stop among the values of one point, and the values of two points at the same distance may interleave. The values
of one point are walked as those of one time ([above](#nodetimesett)), over the box from the point to itself:
`shops[square..square]` yields exactly the values `get(square)` returns, about 27% faster than `get` on 3 values
per point. Pass the same `geo` value as both corners.

## Sampling node tags

Every node tag exposes a static `sample(...)` that returns a `Table` of values from a range, plus `info(...)` returning a per-node `NodeInfo<K>` for metadata. Signatures differ by tag:

```gcl
nodeTime::sample(refs: Array<nodeTime>, from: time?, to: time?, maxRows: int,
                 mode: SamplingMode, maxDephasing: duration?, tz: TimeZone?): Table
nodeIndex::sample(refs: Array<nodeIndex>, from: any?, maxRows: int, mode: SamplingMode): Table
nodeSet::sample(refs: Array<nodeSet>, from: any?, maxRows: int, mode: SamplingMode): Table   // one column per set
nodeIndexSet::sample(refs: Array<nodeIndexSet>, from: any?, maxRows: int, mode: SamplingMode): Table   // a row per pair
nodeList::sample(refs: Array<nodeList>, from: int?, to: int?, maxRows: int,
                 mode: SamplingMode, maxDephasing: int?): Table
nodeGeo::sample(refs: Array<nodeGeo>, from: geo?, to: geo?, maxRows: int, mode: SamplingMode): Table
nodeTimeSet::sample(refs: Array<nodeTimeSet>, from: time?, to: time?, maxRows: int,
                    mode: SamplingMode, maxDephasing: duration?, tz: TimeZone?): Table   // a row per pair
nodeGeoSet::sample(refs: Array<nodeGeoSet>, from: geo?, to: geo?, maxRows: int, mode: SamplingMode): Table   // a row per pair
```

`SamplingMode` controls how points within the range are picked:

| Mode          | Behavior                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `fixed`       | Pick samples at a fixed delta between index values. Can return a large set — bound `from`/`to` carefully.  |
| `fixed_reg`   | Like `fixed`, but linearly regresses numerical values between the nearest stored points to smooth output. |
| `adaptative`  | Variable delta — bound the number of skipped elements between samples. Useful for non-monotonic series.   |
| `dense`       | All elements in range; no down-sampling.                                                                  |

`maxRows` is the upper bound on rows returned regardless of mode. On a `nodeGeo` the range `from..to` is the box of the two corners (a `null` corner is the world's) and no row falls outside it: `dense` and `adaptative` pick among the keys of the box, `fixed` and `fixed_reg` lay `maxRows` grid points over the box, evenly spaced along the curve from the south-west corner, each carrying the value of the stored key before it on the curve when one follows it too. On a `nodeTime` or `nodeList`, `fixed` and `fixed_reg` lay rows at even steps from `from` to `to`, every ref of `refs` sampled at the same keys. Going forward (`from < to`) a row holds the value at or before its key, going backward the value at or after it. A row past the last key going forward, or before the first going backward, is `null`: on a `nodeTime`, it holds that edge value only while it is less than `maxDephasing` away from it (`null` is 0). A `nodeList` sample ignores `maxDephasing`. Its `from` and `to` are positions: without `to` it stops at the last position of the shortest list in `refs`, an empty list there gives an empty table, `fixed_reg` gives the same rows as `fixed` (a list has no gaps), and with no `to` and a `from` past the shortest list's end it walks backward. `nodeList::info` reports the range `0..size()-1`, with a `null` `from` for an empty list.

A `nodeTimeSet` or `nodeGeoSet` sample takes its range and grid as a `nodeTime` or `nodeGeo` sample does, and gives
every value a row of its own, the key repeated for each of its values:

- `dense` and `adaptative`: a key column and a value column per ref, each ref filling its own rows from the first. `dense`
  gives a row per pair of the range, in walk order, at most `maxRows` per ref. `adaptative` numbers the pairs of the
  range from 0 and keeps those at positions 0, k, 2k..., k being `ceil(pairs / maxRows)`, so its rows spread over the
  whole range. A ref with fewer rows leaves `null` cells.
- `fixed` and `fixed_reg`: one key column for the grid, then a value column per ref. A grid point carries the values
  of the key whose value a `nodeTime` or `nodeGeo` sample carries there, and gives as many rows as the most values a
  ref carries; a ref carrying fewer gives `null` in its column for the extra rows. The table holds at most `maxRows`
  rows: it stops before a grid point whose rows would pass `maxRows`, so a grid point is never cut, except a first
  grid point whose rows alone pass it, which gives its first `maxRows` rows. A grid point carrying several values
  takes several rows, so the rows can end before `to`. `fixed_reg` does not interpolate between two sets of values: it
  reads as `fixed`.

`info(...)` counts pairs in `size`, with the first and last key in `from` and `to`.
