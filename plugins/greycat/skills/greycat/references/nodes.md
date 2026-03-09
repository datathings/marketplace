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


## Relationship Patterns

### One-to-Many

```gcl
type City {
    name: String;
    country: node<Country>; // single
    streets: nodeList<node<Street>>; // one-to-many
}

type Street { name: String; city: node<City>; }  // back-reference
```

### Many-to-Many

```gcl
type Student { name: String; courses: nodeList<node<Course>>; }
type Course { name: String; students: nodeList<node<Student>>; }

abstract type Enrollment {
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

    fn average(start: time, end: time): float {
        var sum = 0.0;
        var count = 0;
        for (t, value in this.readings[start..end]) {
            sum = sum + value;
            count = count + 1;
        }
        if (count > 0) {
            return sum / count;
        }
        return 0.0;
    }
}
```

## Key Takeaways

- **Persistence**: `node<T>` for persistent, `T` for temporary (applies also for collections)
- **Relationships**: use `node<T>` refs, not embedded `T` objects
- **Coherence**: maintain consistency across collections (set/remove)
