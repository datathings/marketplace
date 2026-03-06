# GreyCat Design Patterns

## Abstract Types & Inheritance

Abstract types enable polymorphism with both concrete and abstract methods.

```gcl
abstract type Building {
    address: String;
    year_built: int;

    fn calculate_tax(): float;  // Abstract - must be implemented
    fn get_age(): int { return 2024 - year_built; }  // Concrete - shared
}

type House extends Building {
    bedrooms: int;
    fn calculate_tax(): float { return bedrooms * 100.0; }
}

type Commercial extends Building {
    square_meters: float;
    fn calculate_tax(): float { return square_meters * 5.0; }
}
```

### Polymorphic Storage

```gcl
var buildings_by_address: nodeIndex<String, node<Building>>;

// Store different subtypes in same index
var house = node<House>{ House { address: "123 Main St", year_built: 2000, bedrooms: 3 }};
buildings_by_address.set(house->address, house);

var shop = node<Commercial>{ Commercial { address: "456 Market St", year_built: 2010, square_meters: 150.0 }};
buildings_by_address.set(shop->address, shop);

// Polymorphic iteration
for (address, building in buildings_by_address) {
    var tax = building->calculate_tax();  // Calls correct implementation
}

// Type-narrowing
var building = buildings_by_address.get("123 Main St").resolve();
if (building is House) {}
```

**Key rules**: abstract methods must be implemented; concrete methods cannot be overridden; `is` for type check, `as` for cast; `node<BaseType>` stores heterogeneous subtypes.

## Relationship Patterns

### One-to-Many

```gcl
type City {
    name: String;
    country: node<Country>;         // Reference (64b)
    streets: nodeList<node<Street>>; // Collection
}

type Street { name: String; city: node<City>; }  // Back-reference
```

### Many-to-Many

```gcl
type Student { name: String; courses: nodeList<node<Course>>; }
type Course { name: String; students: nodeList<node<Student>>; }

abstract type EnrollmentService {
    static fn enroll(student: node<Student>, course: node<Course>) {
        student->courses.add(course);
        course->students.add(student);
    }

    static fn unenroll(student: node<Student>, course: node<Course>) {
        // Rebuild lists without the removed item
        var new_courses = nodeList<node<Course>> {};
        for (i, c in student->courses) { if (c != course) { new_courses.add(c); } }
        student->courses = new_courses;

        var new_students = nodeList<node<Student>> {};
        for (i, s in course->students) { if (s != student) { new_students.add(s); } }
        course->students = new_students;
    }
}
```

## Time-Series Pattern

```gcl
type Sensor {
    id: String;
    location: geo;
    readings: nodeTime<float>;
}

var sensors_by_id: nodeIndex<String, node<Sensor>>;

abstract type SensorService {
    static fn record(sensor: node<Sensor>, value: float, t: time) {
        sensor->readings.setAt(t, value);
    }

    static fn average(sensor: node<Sensor>, start: time, end: time): float {
        var sum = 0.0;
        var count = 0;
        for (t: time, value: float in sensor->readings[start..end]) {
            sum = sum + value;
            count = count + 1;
        }
        return if (count > 0) { sum / count } else { 0.0 };
    }
}
```

## Key Takeaways

1. **Services**: `abstract type` with static functions for business logic
2. **API Layer**: thin wrapper returning `@volatile` types, never nodeList
3. **Persistence**: `node<T>` for persistent, `Array<T>` for temporary
4. **Relationships**: store `node<T>` refs, not embedded objects
5. **Indices**: maintain consistency across multiple indices (set/remove in all)
