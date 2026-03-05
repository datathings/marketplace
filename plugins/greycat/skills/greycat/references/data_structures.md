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

**Removing elements**:
```gcl
// Array - remove by index
var arr = Array<int>{1, 2, 3, 4, 5};
arr.remove(2);  // Removes element at index 2

// Map - remove by key
var map = Map<String, int>{"a": 1, "b": 2, "c": 3};
map.remove("b");  // Removes key "b"
```

> Use `remove` to delete elements. No `unset` method. Use `pprint` for readable console output.

## Windows (FIFO with Statistics)

**TimeWindow** - Collect values within time period, auto-discard old:
```gcl
var tw = TimeWindow<float>{ span: 5s };
tw.add(time::new(t, DurationUnit::seconds), value as float);

println("Average: ${tw.avg()}, Min: ${tw.min()}, Max: ${tw.max()}, Size: ${tw.size()}");
```

**SlidingWindow** - Fixed number of elements:
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

**Table Column Mappings** - Transform tables by extracting nested fields:
```gcl
var mappings = Array<TableColumnMapping>{
    TableColumnMapping { column: 0, extractors: Array<any>{"*", "a"} },  // resolve node, get attr
    TableColumnMapping { column: 1, extractors: Array<any>{"a"} },       // resolve field
    TableColumnMapping { column: 2, extractors: Array<any>{0} }          // array index
};
var newTable = Table::applyMappings(t, mappings);
```

## Tensor

Multi-dimensional numerical arrays:

```gcl
var t = Tensor::new([2, 3, 2]); // Shape: 2x3x2 tensor (12 elements)
t.set(0, 1.0);
var value = t.get(0);
```

**Operations**: `flatten()`, `reshape()`, `min()`, `max()`, `avg()`, `sum()`, `matmul()`

**Shapes**: `[2, 3]` = 2x3 matrix, `[2, 3, 4]` = 2x3x4 3D array

## Buffer

Typed byte arrays (C-like):

```gcl
var b = Buffer::new(1024);  // 1KB
b.write_i32(0, 42);
var value = b.read_i32(0);
```

**Types**: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `bool` (read/write)

## Stack & Queue

```gcl
var stack = Stack<int>{}; stack.push(1); stack.push(2); var top = stack.pop();
var queue = Queue<int>{}; queue.enqueue(1); queue.enqueue(2); var front = queue.dequeue();
```

## Set

Unordered unique elements:

```gcl
var set = Set<String>{}; set.add("a"); set.add("b"); set.add("a");  // Only one "a"
var hasA = set.contains("a");  // true
set.remove("a");
```
