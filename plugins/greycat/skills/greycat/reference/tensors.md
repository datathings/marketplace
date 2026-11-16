# Tensors and vector search

Reach here when the task stores or computes on numeric arrays (`Tensor`), measures distances between vectors, or
searches the nearest vectors with `VectorIndex`. The full API with every throw is the doc of `Tensor`,
`TensorType`, `TensorDistance` and `VectorIndex` in `lib/std/core.gcl`.

## Contents

- [Create and shape](#create-and-shape)
- [Elements](#elements)
- [Whole-tensor operations](#whole-tensor-operations)
- [Complex tensors](#complex-tensors)
- [Distances](#distances)
- [VectorIndex](#vectorindex)
- [Out of a tensor](#out-of-a-tensor)
- [Pitfalls](#pitfalls)

## Create and shape

A `Tensor` holds numbers of one `TensorType` (`i32`, `i64`, `f32`, `f64`, `c64`, `c128`) in up to 8 dimensions,
row-major. It has no literal: build it empty, then shape it with `init`, which zero-fills, or wrap an array.

```gcl
var m = Tensor {};
m.init(TensorType::f64, [2, 3]); // 2 rows of 3, all 0.0
Assert::equals(m.shape(), [2, 3]);
Assert::equals(m.dim(), 2);
Assert::equals(m.size(), 6);
Assert::equals(m.type(), TensorType::f64);
var v = Tensor::wrap_1d(TensorType::f32, [1.0, 2.0, 3.0]); // shape [3]
var q = VectorIndex::wrap([0.5, 0.5], TensorType::f64); // the same, from an Array<float>
Assert::equals(v.size() + q.size(), 5);
```

- `init(etype, Array<int> {})` gives one element and no dimension. `get` and `set` refuse it: write it with `fill`.
- A dimension of 0 gives a tensor without elements, to grow with `append`:

```gcl
var rows = Tensor {};
rows.init(TensorType::f64, [0, 2]);
rows.setCapacity(1000); // optional: reserves, so append does not reallocate each time
rows.append([1.0, 2.0]); // one row of the right length
rows.append([3.0, 4.0]);
Assert::equals(rows.shape(), [2, 2]);
```

`append` also takes a value (1-d or `[n, 1]` tensors) and a tensor of the same type, as one entry or several. A
refused value leaves the tensor unchanged.

## Elements

A position is one `int` per dimension. `get` returns an `int` for `i32`/`i64`, a `float` otherwise.

```gcl
var t = Tensor {};
t.init(TensorType::i64, [2, 2]);
t.set([0, 1], 7);
t.add([0, 1], 2);
Assert::equals(t.get([0, 1]), 9);
var pos = t.initPos(); // [0, 0]
do {
    println("${pos} = ${t.get(pos)}");
} while (t.incPos(pos)); // row-major; false after the last, pos back to [0, 0]
```

- A `float` stored in `i32`/`i64` is truncated toward zero and held at the integer limits (NaN gives 0); an `int` too
  large for `i32` wraps around.
- An `f32` tensor rounds each value to 32 bits, `1e40` to `Infinity`.
- A position out of the shape, or of the wrong length, throws and names the shape.

## Whole-tensor operations

```gcl
var t = Tensor {};
t.init(TensorType::f64, [3, 3]);
t.fill([1.0, 2.0, 3.0]); // repeated: each row is 1, 2, 3
Assert::equals(t.sum(), 18.0); // exact for integers, 0.0 for an empty tensor
Assert::equals(t.mean(), 2.0); // mean, min and max throw on an empty tensor
Assert::equals(t.max(), 3.0);
var block = t.slice([1, 0], [2, 2]); // a copy of rows 1 and 2, columns 0 and 1
Assert::equals(block.shape(), [2, 2]);
var twice = t.scale(2, false); // a copy; true scales t itself and returns it
Assert::equals(twice.sum(), 36.0);
t.reshape([9]); // the same elements in a new shape
t.slide(2); // drops the first 2 entries of the first dimension
Assert::equals(t.shape(), [7]);
```

- `fill` takes a value, an `Array` (repeated, extra elements left out) or a tensor of the same shape. A float tensor
  copied into an integer one is rounded to the nearest (`2.5` to 2, `3.5` to 4), where a single `float` is truncated.
- `copyElementsFrom(src, startPos, nbElem, destinationPos)` copies between tensors of one type by flat offsets.
- `sum`, `mean`, `min` and `max` refuse complex tensors. `min`/`max` return `int` for integer tensors, and NaN when an
  element is NaN.

## Complex tensors

`get`, `set`, `add` and `fill` with a value work on the real part; `getImag` and `setImag` on the imaginary part.

```gcl
var signal = Tensor::wrap_1d(TensorType::f64, [3.0, 0.0]);
var c = signal.to_complex_tensor(); // c128, imaginary parts 0; an f32 tensor gives c64
c.setImag([0], 4.0); // 3+4i
Assert::equals(c.get_absolute_tensor().get([0]), 5.0); // the modulus, as f64
Assert::equals(c.get_real_tensor().get([0]), 3.0);
Assert::equals(c.get_imaginary_tensor().get([0]), 4.0);
var degrees = c.get_phase_tensor(true); // false for radians
Assert::equalsd(degrees.get([0]) as float, 53.130102354, 1e-9);
```

`get_absolute_tensor` also works on real tensors, keeping their type.

## Distances

`a.distance(b, d)` takes two `f32` or `f64` tensors of one type and one shape, of any rank, and sums in `f64`.

| `TensorDistance` | value |
| ---------------- | ----- |
| `l2sq`           | `Σ (a[i] - b[i])²`, orders as `euclidean`, no square root |
| `euclidean`      | `sqrt(l2sq)` |
| `cosine`         | `1 - a·b / (\|a\| \|b\|)`, from 0 to 2; 1 when `a·b` is 0, 0 for two zero vectors |
| `lorentz`        | hyperboloid model, index 0 the time component |
| `poincare`       | Poincaré ball, norms below 1 |

```gcl
var a = Tensor::wrap_1d(TensorType::f64, [1.0, 0.0]);
var b = Tensor::wrap_1d(TensorType::f64, [0.0, 1.0]);
Assert::equals(a.distance(b, TensorDistance::l2sq), 2.0);
Assert::equals(a.distance(b, TensorDistance::cosine), 1.0);
```

## VectorIndex

An approximate nearest-neighbour index (HNSW) stored in the graph. Put it in a node to persist it; `add` takes the
vector as a `node<Tensor>`.

```gcl
var docs: node<VectorIndex<String>?>;

fn index_doc(embedding: Array<float>, id: String) {
    var index = docs.resolve();
    if (index == null) {
        index = VectorIndex<String> { distance: TensorDistance::cosine, rng: Random { seed: 1 } };
        docs.set(index);
    }
    index.add(node<Tensor> { VectorIndex::wrap(embedding, TensorType::f32) }, id);
}

fn nearest(embedding: Array<float>, k: int): Array<String> {
    var found = Array<String> {};
    for (_, r in docs.resolve()!!.search(VectorIndex::wrap(embedding, TensorType::f32), k)) {
        found.add(r.value); // r.key: the stored vector, r.distance: as Tensor::distance gives it
    }
    return found;
}
```

- `distance` (default `l2sq`) and `rng` are set at instantiation only. Without `rng` the graph depends on the clock;
  set a seeded `Random` for reproducible builds and tests.
- Every vector has the type and size of the first one indexed. NaN or infinite elements are refused.
- `search(q, k)` returns at most `k` results, nearest first: 1 when `k` is `null`, all when `k` is above `size()`.
  Being approximate, it can miss a few true neighbours; compare with `Tensor::distance` over all vectors when exact
  results matter.

## Out of a tensor

```gcl
var t = Tensor::wrap_1d(TensorType::i64, [1, 2, 3]);
var table = t.toTable(); // 1-d: one column; 2-d: rows x columns
Assert::equals(table.rows(), 3);
println(t.toString()); // {"dim":1,"shape":[3],"type":TensorType::i64,"data":[1,2,3]}
```

`BinReader::read_tensor(etype, shape)` reads raw little-endian elements; in tests,
`Assert::equalst(a, b, epsilon)` compares type, shape and every element within `epsilon`.

## Pitfalls

1. **A position or shape built from variables needs its type**: `t.get([i, j])` runs but fails `greycat lint`
   (`Array<any?>`), and an empty `[]` throws at runtime. Write `t.get(Array<int> { i, j })`, `Array<int> {}`.
2. **No `Tensor` literal and no `Tensor::new`**: `Tensor {}` then `init`, `Tensor::wrap_1d` or `VectorIndex::wrap`.
3. **`init` again discards the elements**: it zero-fills every time, and keeps only the memory.
4. **Integer tensors convert silently**: `set(pos, 2.9)` stores 2. Use `f32`/`f64` for measurements.
5. **Distances take floats only**: convert integer data with `wrap_1d(TensorType::f64, ...)` before `distance`.
