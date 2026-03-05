# GreyCat Design Patterns

Common design patterns and best practices for GreyCat development.

## Service Pattern

Services encapsulate business logic using `abstract type` with static functions. Separates domain operations from API layer and provides clean CRUD interfaces.

### Basic Service

```gcl
// Model
type Country { name: String; code: String; population: int; }

// Global indices
var countries_by_name: nodeIndex<String, node<Country>>;
var countries_by_code: nodeIndex<String, node<Country>>;

// Service
abstract type CountryService {
    static fn create(name: String, code: String): node<Country> {
        var country = node<Country>{ Country { name: name, code: code, population: 0 }};
        countries_by_name.set(name, country);
        countries_by_code.set(code, country);
        return country;
    }

    static fn find_by_name(name: String): node<Country>? { return countries_by_name.get(name); }
    static fn find_by_code(code: String): node<Country>? { return countries_by_code.get(code); }

    static fn update(country: node<Country>, name: String?, code: String?) {
        if (name != null) { country->name = name; }
        if (code != null) { country->code = code; }
    }

    static fn delete(country: node<Country>) {
        countries_by_name.remove(country->name);
        countries_by_code.remove(country->code);
    }

    static fn list_all(): Array<node<Country>> {
        var results = Array<node<Country>> {};
        for (name, country in countries_by_name) { results.add(country); }
        return results;
    }
}
```

### API Layer Usage

```gcl
// API types with @volatile
@volatile type CountryView { name: String; code: String; population: int; }
@volatile type CountryCreate { name: String; code: String; }

// API endpoints use service
@expose @permission("public")
fn get_countries(): Array<CountryView> {
    var views = Array<CountryView> {};
    for (country in CountryService::list_all()) {
        views.add(CountryView { name: country->name, code: country->code, population: country->population });
    }
    return views;
}

@expose @permission("admin")
fn create_country(data: CountryCreate): CountryView {
    var country = CountryService::create(data.name, data.code);
    return CountryView { name: country->name, code: country->code, population: country->population };
}
```

**Key principles**:
- Service functions return `node<T>` (internal)
- API functions return `Array<XxxView>` with `@volatile` types (external)
- Never return `nodeList` or `nodeIndex` from API endpoints
- Keep business logic in services, thin API layer

## Abstract Types & Inheritance

Abstract types enable polymorphism with both concrete and abstract methods.

```gcl
// Base abstract type
abstract type Building {
    address: String;
    year_built: int;

    fn calculate_tax(): float;  // Abstract - must be implemented
    fn get_age(): int { return 2024 - year_built; }  // Concrete - shared
}

// Concrete implementations
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

// Add different types
var house = node<House>{ House { address: "123 Main St", year_built: 2000, bedrooms: 3 }};
buildings_by_address.set(house->address, house);

var shop = node<Commercial>{ Commercial { address: "456 Market St", year_built: 2010, square_meters: 150.0 }};
buildings_by_address.set(shop->address, shop);

// Iterate polymorphically
for (address, building in buildings_by_address) {
    var tax = building->calculate_tax();  // Calls correct implementation
    var age = building->get_age();        // Shared method
}

// Type checking
fn process_building(building: node<Building>) {
    if (building is House) {
        var house = building as node<House>;
        info("House with ${house->bedrooms} bedrooms");
    } else if (building is Commercial) {
        var shop = building as node<Commercial>;
        info("Commercial with ${shop->square_meters} sqm");
    }
    var tax = building->calculate_tax();  // Polymorphic call
}
```

**Key principles**:
- Abstract types can have both abstract and concrete methods
- Concrete methods cannot be overridden
- Use `is` for type checking, `as` for casting
- Store heterogeneous types with `node<BaseType>`

## CRUD Service Pattern

Complete CRUD with validation and error handling:

```gcl
type User { id: int; email: String; name: String; created_at: time; }

var users_by_id: nodeIndex<int, node<User>>;
var users_by_email: nodeIndex<String, node<User>>;
var user_id_counter: node<int?>;

abstract type UserService {
    // Create with validation
    static fn create(email: String, name: String): node<User> {
        if (users_by_email.get(email) != null) { throw "User with email ${email} already exists"; }

        var id = (user_id_counter.resolve() ?? 0) + 1;
        user_id_counter.set(id);

        var user = node<User>{ User { id: id, email: email, name: name, created_at: Time::now() }};
        users_by_id.set(id, user);
        users_by_email.set(email, user);
        return user;
    }

    // Read
    static fn find_by_id(id: int): node<User>? { return users_by_id.get(id); }
    static fn find_by_email(email: String): node<User>? { return users_by_email.get(email); }
    static fn list_all(): Array<node<User>> {
        var results = Array<node<User>> {};
        for (id, user in users_by_id) { results.add(user); }
        return results;
    }

    // Update
    static fn update_email(user: node<User>, new_email: String) {
        users_by_email.remove(user->email);
        user->email = new_email;
        users_by_email.set(new_email, user);
    }

    static fn update_name(user: node<User>, new_name: String) { user->name = new_name; }

    // Delete
    static fn delete(user: node<User>) {
        users_by_id.remove(user->id);
        users_by_email.remove(user->email);
    }
}
```

## Relationship Patterns

### One-to-Many

```gcl
type City {
    name: String;
    country: node<Country>;  // Reference
    streets: nodeList<node<Street>>;  // Collection
}

type Street { name: String; city: node<City>; }  // Back-reference

var cities_by_name: nodeIndex<String, node<City>>;
var streets_by_name: nodeIndex<String, node<Street>>;

abstract type CityService {
    static fn create(name: String, country: node<Country>): node<City> {
        var city = node<City>{ City { name: name, country: country, streets: nodeList<node<Street>>{} }};
        cities_by_name.set(name, city);
        return city;
    }

    static fn add_street(city: node<City>, street_name: String): node<Street> {
        var street = node<Street>{ Street { name: street_name, city: city }};
        city->streets.add(street);
        streets_by_name.set(street_name, street);
        return street;
    }
}
```

### Many-to-Many

```gcl
type Student { name: String; courses: nodeList<node<Course>>; }
type Course { name: String; students: nodeList<node<Student>>; }

var students_by_name: nodeIndex<String, node<Student>>;
var courses_by_name: nodeIndex<String, node<Course>>;

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
    static fn create(id: String, lat: float, lng: float): node<Sensor> {
        var sensor = node<Sensor>{ Sensor { id: id, location: geo { lat: lat, lng: lng }, readings: nodeTime<float>{} }};
        sensors_by_id.set(id, sensor);
        return sensor;
    }

    static fn record_reading(sensor: node<Sensor>, value: float, timestamp: time) {
        sensor->readings.setAt(timestamp, value);
    }

    static fn get_readings(sensor: node<Sensor>, start: time, end: time): Array<Tuple<time, float>> {
        var results = Array<Tuple<time, float>> {};
        for (t: time, value: float in sensor->readings[start..end]) {
            results.add(Tuple { first: t, second: value });
        }
        return results;
    }

    static fn get_average(sensor: node<Sensor>, start: time, end: time): float {
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

1. **Services**: Use `abstract type` with static functions for business logic
2. **API Layer**: Thin layer that calls services, returns `@volatile` types
3. **Persistence**: Use `node<T>` for persistent objects, `Array<T>` for temporary
4. **Relationships**: Store `node<T>` refs, not embedded objects
5. **Inheritance**: Abstract types for polymorphism, concrete methods shared
6. **Collections**: Always initialize `nodeList`, `nodeIndex`, `nodeTime` in constructors
7. **Validation**: Perform in service layer before persistence
8. **Indices**: Maintain consistency across multiple indices
