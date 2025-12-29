# Data Structures

## Tuple

Simple association for couple of values:

```gcl
var tupleA = Tuple{x:0.5, y:"a"};
var tupleB = (0.5, "b");  // Shorthand
```

## Array & Map

In-memory structures for small data:

```gcl
var arr = Array<float>{1.2, 3.4, 5.0};
var arrB = [1.2, 3.4];  // Shorthand (typing unknown)

for (k, v in arr) { println("Index: ${k}, value ${v}"); }

var map = Map<String, int>{"Hello": 5, "Test": 2};
map.set("Key", 42);
println(map.get("Test"));
```

### Removing Elements

Use the `remove` method to delete elements from arrays and maps:

```gcl
// Array - remove by index
var arr = Array<int>{1, 2, 3, 4, 5};
arr.remove(2);  // Removes element at index 2

// Map - remove by key
var map = Map<String, int>{"a": 1, "b": 2, "c": 3};
map.remove("b");  // Removes key "b" and its value
```

> Note: There is no `unset` method. Use `remove` to delete elements.

> Use `pprint` for readable console output with formatting.

## Windows (FIFO with Statistics)

### TimeWindow
Collect values within time period, auto-discard old:

```gcl
var tw = TimeWindow<float>{ span: 5s };
tw.add(time::new(t, DurationUnit::seconds), value as float);

println("Average: ${tw.avg()}");
println("Min: ${tw.min()}, Max: ${tw.max()}");
println("Size: ${tw.size()}");
```

### SlidingWindow
Fixed number of elements:

```gcl
var sw = SlidingWindow<float>{ span: 5 };  // 5 elements max
sw.add(value as float);
println("Average over ${sw.size()}: ${sw.avg()}");
```

## Table

2D container for result sets, sampling, web components:

```gcl
var t = Table{};
t.init(2, 4);  // 2 rows, 4 columns

t.set_cell(0, 1, "value");
t.set_cell(0, 2, time::now());
t.set_row(1, ["a", 0.0, time::now()]);

t.sort(1, SortOrder::asc);  // Sort by column 1
t.remove_row(0);

info(t.rows());
info(t.get_cell(0, 0));
```

### Table Column Mappings

Transform tables by extracting nested fields:

```gcl
var mappings = Array<TableColumnMapping>{
    TableColumnMapping { column: 0, extractors: Array<any>{"*", "a"} },  // resolve node, get attr
    TableColumnMapping { column: 1, extractors: Array<any>{"a"} },       // resolve field
    TableColumnMapping { column: 2, extractors: Array<any>{0} }          // array index
};
var newTable = Table::applyMappings(t, mappings);
```

## Tensor

Multi-dimensional numerical array for ML batch processing:

```gcl
var t = Tensor{};
t.init(TensorType::f64, Array<int>{4, 3});  // 4 rows, 3 columns

Assert::equals(t.dim(), 2);   // dimensions
Assert::equals(t.size(), 12); // total elements
```

### Tensor Types
`i32`, `i64`, `f32`, `f64`, `c64`, `c128` (complex)

### Set/Get Values

```gcl
t.set(Array<int>{0, 0, 0}, 42.3);
var val = t.get(Array<int>{0, 0, 0});
t.fill(50.3);  // Fill all with value
```

### Iterate Multidimensional

```gcl
var index = t.initPos();
do {
    t.set(index, random.uniformf(-5.0, 5.0));
} while (t.incPos(index));
```

### Append Data

```gcl
// 1D tensor
t.append(3.0);           // single value
t.append([4.0, 4.0]);    // array
t.append(otherTensor);   // another 1D tensor

// ND tensor - append N-1 dimensional tensor
var t4d = Tensor{};
t4d.init(TensorType::f64, Array<int>{0, 2, 3, 4});
t4d.append(tensor3d);  // Must be 2x3x4
```

### Performance

```gcl
t.setCapacity(1000);  // Pre-allocate
t.reset();            // Reuse memory with new shape
```

## Buffer

Efficient string builder:

```gcl
var b = Buffer{};
b.add(1);
b.add(" one ");
b.add([1, 2]);
println(b.toString());  // "1 one Array{1,2}"
```

## Stack (LIFO)

```gcl
var stack = Stack<int>{};
stack.push(i);
println(stack.first());  // bottom
println(stack.last());   // top
var val = stack.pop();   // returns and removes top
```

## Queue (FIFO)

```gcl
var queue = Queue<int>{};
queue.push(i);
println(queue.front());  // first in
println(queue.back());   // last in
var val = queue.pop();   // returns and removes front
```
