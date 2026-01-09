---
name: tutorial
description: Interactive learning modules for GreyCat concepts - from basics to advanced patterns
allowed-tools: AskUserQuestion, Read, Write, Bash, Grep, Glob
---

# GreyCat Interactive Tutorial

**Purpose**: Progressive, hands-on learning of GreyCat concepts with real code examples and validation

**Run When**: Onboarding new developers, learning specific features, refreshing knowledge

---

## Overview

10 sequential modules covering GreyCat from basics to advanced:

1. **Basics** (20 min) - Types, nullability, functions
2. **Persistence** (25 min) - Nodes, when to persist
3. **Collections** (30 min) - Indexed collections, when to use each
4. **Modeling** (25 min) - Data models with relationships
5. **Services** (20 min) - Business logic patterns
6. **APIs** (25 min) - @expose endpoints, @volatile types
7. **Testing** (20 min) - Writing comprehensive tests
8. **Parallelization** (25 min) - Jobs and async patterns
9. **Time & Geo** (30 min) - Time-series and spatial data
10. **Advanced** (30 min) - Inheritance, complex patterns

**Total**: ~4 hours of interactive learning

---

## Progress Tracking

Tutorial creates `.greycat-tutorial-progress` file to track completion:

```json
{
  "started": "2026-01-09T10:30:00Z",
  "current_module": 5,
  "completed_modules": [1, 2, 3, 4],
  "last_session": "2026-01-09T12:00:00Z"
}
```

---

## Step 1: Check Progress & Choose Module

**Read progress file**:

```bash
PROGRESS_FILE=".greycat-tutorial-progress"

if [ -f "$PROGRESS_FILE" ]; then
    echo "Welcome back to GreyCat Tutorial!"
    echo ""

    # Parse progress (simple grep/sed approach)
    CURRENT=$(grep "current_module" "$PROGRESS_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')
    COMPLETED=$(grep "completed_modules" "$PROGRESS_FILE" | sed 's/.*\[\(.*\)\].*/\1/' | tr ',' ' ' | wc -w)

    echo "Progress: $COMPLETED/10 modules complete"
    echo "Current: Module $CURRENT"
    echo ""
else
    echo "Welcome to GreyCat Tutorial!"
    echo ""
    echo "This interactive tutorial will teach you GreyCat through hands-on exercises."
    echo "Estimated time: 4 hours (can pause/resume anytime)"
    echo ""

    # Create progress file
    cat > "$PROGRESS_FILE" <<EOF
{
  "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "current_module": 1,
  "completed_modules": [],
  "last_session": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    CURRENT=1
fi
```

**Ask user**:

```typescript
AskUserQuestion({
  questions: [{
    question: "How would you like to proceed?",
    header: "Tutorial",
    multiSelect: false,
    options: [
      {
        label: `Continue from Module ${CURRENT}`,
        description: "Resume where you left off"
      },
      {
        label: "Start from beginning",
        description: "Reset progress and start Module 1"
      },
      {
        label: "Jump to specific module",
        description: "Choose any module to practice"
      },
      {
        label: "Exit tutorial",
        description: "Come back later"
      }
    ]
  }]
})
```

---

## Module 1: Basics (20 min)

### Concept: Types, Nullability, Functions

**Explain**:

```
GreyCat is a statically-typed language with non-null defaults.

Key concepts:
- Primitives: int, float, bool, String, char, time
- Non-null by default, use ? for nullable
- Functions with return types (no void)
- For loops (no C-style for)
```

**Example**:

```gcl
// Types and nullability
fn greet(name: String?): String {
    if (name == null) {
        return "Hello, stranger!";
    }
    return "Hello, ${name}!";
}

// Using the function
fn main() {
    var msg1 = greet("Alice");      // "Hello, Alice!"
    var msg2 = greet(null);          // "Hello, stranger!"

    info(msg1);
    info(msg2);
}
```

**Hands-on Exercise**:

"Now you try! Write a function that calculates a person's age from birth year."

**Generate template** in `tutorial/module1_basics.gcl`:

```gcl
// Module 1: Basics Exercise
// TODO: Implement this function
fn calculate_age(birth_year: int, current_year: int): int {
    // Your code here
    return 0;  // Replace this
}

// Tests (don't modify)
@test
fn test_calculate_age() {
    Assert::equals(calculate_age(1990, 2024), 34);
    Assert::equals(calculate_age(2000, 2024), 24);
}

fn main() {
    info("Age for 1990: ${calculate_age(1990, 2024)}");
}
```

**Run & Validate**:

```bash
echo "Run your code with: greycat run main"
echo "Test with: greycat test tutorial/module1_basics.gcl"
echo ""
read -p "Press Enter when you've completed the exercise..."

# Run tests
greycat test tutorial/module1_basics.gcl

if [ $? -eq 0 ]; then
    echo "✓ Exercise complete! Moving to next module."
else
    echo "⚠ Tests failed. Review and try again."
    echo "Hint: Age = current_year - birth_year"
fi
```

**Checkpoint**:

```
✓ Module 1 Complete: Basics

You learned:
  - Type system (int, String, bool, time)
  - Nullability (? for nullable, ?? for coalescing)
  - Functions and return types
  - String interpolation

Next: Module 2 - Persistence Fundamentals (25 min)
```

---

## Module 2: Persistence Fundamentals (25 min)

### Concept: node<T> vs Plain Objects

**Explain**:

```
GreyCat separates transient (RAM) from persistent (storage) data.

Plain object:
  type User { name: String; }
  var u = User { name: "Alice" };  // Lives in RAM only

Persistent node:
  var n = node<User>{ User { name: "Alice" } };  // Saved to gcdata/

When to use node<T>:
  ✓ Module-level variables (global data)
  ✓ Type fields (relationships)
  ✗ Local variables (temporary data)
  ✗ Function parameters/returns (unless passing persisted refs)
```

**Example**:

```gcl
type Country { name: String; code: String; }

// Global index (persisted)
var countries_by_code: nodeIndex<String, node<Country>>;

// Create and persist
fn create_country(name: String, code: String): node<Country> {
    var country = node<Country>{ Country {
        name: name,
        code: code
    }};

    countries_by_code.set(code, country);
    return country;
}

// Read from persistence
fn find_country(code: String): node<Country>? {
    return countries_by_code.get(code);
}

fn main() {
    var lux = create_country("Luxembourg", "LU");
    info("Created: ${lux->name}");

    var found = find_country("LU");
    if (found != null) {
        info("Found: ${found->name}");
    }
}
```

**Hands-on Exercise**:

"Create a simple product catalog with persistence."

**Generate** `tutorial/module2_persistence.gcl`:

```gcl
// Module 2: Persistence Exercise

type Product {
    id: int;
    name: String;
    price: float;
}

// TODO: Create a global index for products
// var products_by_id: ???

// TODO: Implement this function
fn create_product(id: int, name: String, price: float): node<Product> {
    // 1. Create a node<Product>
    // 2. Store it in the index
    // 3. Return the node
    return null!!;  // Replace this
}

// TODO: Implement this function
fn find_product(id: int): node<Product>? {
    // Look up product by ID in the index
    return null;  // Replace this
}

// Tests
@test
fn test_product_persistence() {
    var p = create_product(1, "Laptop", 999.99);
    Assert::isNotNull(p);
    Assert::equals(p->name, "Laptop");

    var found = find_product(1);
    Assert::isNotNull(found);
    Assert::equals(found->price, 999.99);
}

fn main() {
    var laptop = create_product(1, "Laptop", 999.99);
    var phone = create_product(2, "Phone", 599.99);

    info("Created products:");
    var found = find_product(1);
    if (found != null) {
        info("  ${found->name}: $${found->price}");
    }
}
```

**Validation** includes checking for nodeIndex declaration.

---

## Module 3: Indexed Collections (30 min)

### Concept: nodeList, nodeIndex, nodeTime, nodeGeo

**Explain**:

```
GreyCat provides specialized persistent collections:

nodeList<node<T>>    - Ordered by integer index (0, 1, 2...)
nodeIndex<K, V>      - Hash-based key-value lookup
nodeTime<T>          - Time-series data (keyed by time)
nodeGeo<node<T>>     - Geo-spatial queries (keyed by geo)

Local alternatives:
Array<T>, Map<K,V>   - For temporary data
```

**Example**:

```gcl
type City {
    name: String;
    population: int;
    streets: nodeList<node<Street>>;  // One-to-many
}

type Street {
    name: String;
}

var cities_by_name: nodeIndex<String, node<City>>;

fn create_city(name: String, pop: int): node<City> {
    var city = node<City>{ City {
        name: name,
        population: pop,
        streets: nodeList<node<Street>>{}  // ⚠️ MUST initialize!
    }};
    cities_by_name.set(name, city);
    return city;
}

fn add_street(city: node<City>, street_name: String) {
    var street = node<Street>{ Street { name: street_name }};
    city->streets.add(street);
}

fn main() {
    var paris = create_city("Paris", 2_200_000);
    add_street(paris, "Champs-Élysées");
    add_street(paris, "Rue de Rivoli");

    info("${paris->name} has ${paris->streets.size()} streets");
}
```

**Hands-on Exercise**:

"Build a school system with students and courses (many-to-many)."

**Generate** `tutorial/module3_collections.gcl` with TODOs for implementing:
- Student and Course types
- nodeIndex for both
- Enrollment (adding to both student->courses and course->students)

---

## Module 4: Data Modeling (25 min)

### Concept: Relationships and Indices

**Explain**:

```
Best practices for data modeling:

1. Store node<T> refs for relationships (not embedded objects)
2. Create global indices for all primary lookups
3. Initialize collection attributes in constructors
4. Keep model files separate from services

File structure:
  backend/src/model/user.gcl       - type User + indices
  backend/src/service/user_service.gcl - business logic
```

**Example** - City/Country hierarchy with complete model.

**Hands-on Exercise**:

"Model a blog system: User writes Posts, Posts have Comments."

---

## Module 5: Services & Business Logic (20 min)

### Concept: Abstract Type Services Pattern

**Explain**:

```
Services encapsulate business logic using abstract types with static functions.

Pattern:
  abstract type XxxService {
      static fn create(...): node<Xxx> { }
      static fn find(...): node<Xxx>? { }
      static fn update(...) { }
      static fn delete(...) { }
  }

Benefits:
  - Centralized business logic
  - Validation in one place
  - Reusable from APIs
```

**Example** with full CRUD service.

**Hands-on Exercise**:

"Implement UserService with validation (email uniqueness, etc)."

---

## Module 6: API Development (25 min)

### Concept: @expose, @permission, @volatile

**Explain**:

```
API Layer best practices:

1. Use @volatile for request/response types
2. Never return nodeList/nodeIndex from APIs
3. Always return Array<XxxView>
4. Use @expose to make function available via HTTP
5. Use @permission for access control

Pattern:
  @volatile type UserView { ... }

  @expose
  @permission("public")
  fn get_users(): Array<UserView> { ... }
```

**Example** with complete CRUD API.

**Hands-on Exercise**:

"Build REST API for blog system from Module 4."

---

## Module 7: Testing (20 min)

### Concept: @test Functions

**Explain**:

```
GreyCat testing:

@test fn test_name() {
    // Arrange
    var user = UserService::create("test@example.com");

    // Act
    var found = UserService::find("test@example.com");

    // Assert
    Assert::isNotNull(found);
    Assert::equals(found->email, "test@example.com");
}

Assertions: equals, isTrue, isFalse, isNull, isNotNull
Setup/teardown: fn setup() { } fn teardown() { }
```

**Hands-on Exercise**:

"Write comprehensive tests for UserService."

---

## Module 8: Parallelization (25 min)

### Concept: Job Pattern

**Explain**:

```
Parallel processing with Jobs:

var jobs = Array<Job<Result>> {};
for (item in items) {
    jobs.add(Job<Result> {
        function: process_fn,
        arguments: [item]
    });
}
await(jobs, MergeStrategy::last_wins);

for (job in jobs) {
    var result = job.result();
}
```

**Hands-on Exercise**:

"Parallelize data processing for 1000 items."

---

## Module 9: Time-Series & Geo (30 min)

### Concept: nodeTime and nodeGeo

**Explain**:

```
Time-series with nodeTime:
  var temps: nodeTime<float>;
  temps.setAt(timestamp, value);
  for (t: time, v: float in temps[start..end]) { }

Geo-spatial with nodeGeo:
  var devices: nodeGeo<node<Device>>;
  devices.set(geo{lat, lng}, device);
  for (pos: geo, d in devices.filter(GeoBox{...})) { }
```

**Hands-on Exercise**:

"Build temperature monitoring with sensor readings over time."

---

## Module 10: Advanced Patterns (30 min)

### Concept: Inheritance & Polymorphism

**Explain**:

```
Abstract types with inheritance:

abstract type Animal {
    name: String;
    fn makeSound(): String;  // Abstract
}

type Dog extends Animal {
    fn makeSound(): String { return "Woof!"; }
}

type Cat extends Animal {
    fn makeSound(): String { return "Meow!"; }
}

// Store polymorphically
var animals: nodeIndex<String, node<Animal>>;
```

**Hands-on Exercise**:

"Implement payment system with multiple payment types (Card, Cash, Crypto)."

---

## Module Navigation

**Between modules**:

```typescript
AskUserQuestion({
  questions: [{
    question: "Module ${N} complete! What next?",
    header: "Progress",
    multiSelect: false,
    options: [
      {
        label: "Continue to Module ${N+1}",
        description: "Next topic: ${next_topic}"
      },
      {
        label: "Review this module",
        description: "Re-read explanation and try exercise again"
      },
      {
        label: "Take a break",
        description: "Save progress and exit"
      }
    ]
  }]
})
```

**Update progress file** after each module completion.

---

## Completion Certificate

**After Module 10**:

```
===============================================================================
🎓 CONGRATULATIONS! 🎓
===============================================================================

You've completed the GreyCat Tutorial!

Modules completed (10/10):
  ✓ Module 1: Basics
  ✓ Module 2: Persistence Fundamentals
  ✓ Module 3: Indexed Collections
  ✓ Module 4: Data Modeling
  ✓ Module 5: Services & Business Logic
  ✓ Module 6: API Development
  ✓ Module 7: Testing
  ✓ Module 8: Parallelization
  ✓ Module 9: Time-Series & Geo
  ✓ Module 10: Advanced Patterns

Time invested: ${hours} hours
Tutorial files: ./tutorial/ directory

Next steps:
  1. Build a real project using what you learned
  2. Explore advanced topics: /greycat:scaffold, /greycat:migrate
  3. Read references: frontend.md, concurrency.md, LIBRARIES.md
  4. Join the community: https://greycat.io/community

Keep learning! 🚀

===============================================================================
```

---

## Tutorial Directory Structure

```
tutorial/
├── module1_basics.gcl
├── module2_persistence.gcl
├── module3_collections.gcl
├── module4_modeling.gcl
├── module5_services.gcl
├── module6_apis.gcl
├── module7_testing.gcl
├── module8_parallelization.gcl
├── module9_timeseries_geo.gcl
└── module10_advanced.gcl
```

---

## Success Criteria

✓ **All modules accessible** with clear explanations
✓ **Hands-on exercises** with validation
✓ **Progress tracked** and resumable
✓ **Self-paced** with pause/resume
✓ **Real code** that runs with greycat run/test
✓ **Completion certificate** at end

---

## Notes

- **Interactive learning**: Each module combines theory + practice
- **Build skills progressively**: Each module builds on previous
- **Real GreyCat code**: All examples are valid, runnable code
- **Validation**: Tests ensure correct understanding
- **Flexible pace**: Can pause/resume anytime
- **Comprehensive**: Covers all major GreyCat concepts
