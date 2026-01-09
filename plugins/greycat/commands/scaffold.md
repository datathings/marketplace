---
name: scaffold
description: Generate models, services, APIs, and tests with proper GreyCat structure
allowed-tools: AskUserQuestion, Read, Write, Bash, Grep, Glob
---

# GreyCat Scaffold Generator

**Purpose**: Generate models, services, API endpoints, and tests with proper GreyCat structure and conventions

**Run When**: Starting new features, adding CRUD operations, creating new domain entities

---

## Overview

This command scaffolds complete GreyCat features with:

1. **Model** - Type definition with proper fields and initialization
2. **Global Indices** - nodeIndex/nodeList/nodeTime/nodeGeo for lookups
3. **Service** - CRUD operations with validation
4. **API Layer** - @expose endpoints with @volatile response types
5. **Tests** - Comprehensive test suite

**Templates**:
- **CRUD Service** - Full create/read/update/delete
- **Time-series Collector** - Model with nodeTime + data collection
- **Graph Traversal** - Model with relationships + traversal queries
- **Custom** - Interactive builder

---

## Step 1: Template Selection

**Ask user** via AskUserQuestion:

```typescript
AskUserQuestion({
  questions: [{
    question: "Which scaffold template would you like to use?",
    header: "Template",
    multiSelect: false,
    options: [
      {
        label: "CRUD Service (Recommended)",
        description: "Complete create/read/update/delete for an entity with global indices"
      },
      {
        label: "Time-series Collector",
        description: "Model with nodeTime index for temporal data collection and querying"
      },
      {
        label: "Graph Traversal API",
        description: "Model with relationships and traversal query endpoints"
      },
      {
        label: "Custom (Guided)",
        description: "Step-by-step builder for specific needs"
      }
    ]
  }]
})
```

---

## Step 2: Detect Project Structure

**Check existing project structure**:

```bash
# Verify we're in a GreyCat project
if [ ! -f "project.gcl" ]; then
    echo "ERROR: Not in a GreyCat project root (project.gcl not found)"
    exit 1
fi

# Check standard directories exist
if [ ! -d "backend/src/model" ]; then
    mkdir -p backend/src/model
fi

if [ ! -d "backend/src/service" ]; then
    mkdir -p backend/src/service
fi

if [ ! -d "backend/src/api" ]; then
    mkdir -p backend/src/api
fi

if [ ! -d "backend/test" ]; then
    mkdir -p backend/test
fi
```

---

## Step 3: Gather Entity Details

**Ask for entity information** via AskUserQuestion:

```typescript
AskUserQuestion({
  questions: [
    {
      question: "What is the entity name? (PascalCase, e.g., Device, User, Order)",
      header: "Entity Name",
      multiSelect: false,
      options: [
        { label: "I'll provide it", description: "Enter custom entity name" }
      ]
    }
  ]
})
```

**Get field definitions** (ask user to provide):

Example format:
```
name: String
email: String
age: int?
created_at: time
```

**Parse fields**:
- Split by newline
- Extract field name, type, nullable (?)
- Validate types against GreyCat types

**Ask for indices** via AskUserQuestion:

```typescript
AskUserQuestion({
  questions: [{
    question: "Which indices do you need for lookups?",
    header: "Indices",
    multiSelect: true,
    options: [
      {
        label: "By ID (nodeIndex<int, node<T>>)",
        description: "Standard ID-based lookup"
      },
      {
        label: "By unique field (nodeIndex<String, node<T>>)",
        description: "Lookup by email, username, or other unique field"
      },
      {
        label: "List (nodeList<node<T>>)",
        description: "Ordered collection for iteration"
      },
      {
        label: "Time-series (nodeTime<node<T>> or nodeTime<primitive>)",
        description: "Temporal data with time-based queries"
      },
      {
        label: "Geo-spatial (nodeGeo<node<T>>)",
        description: "Geographic queries with bounding box/circle/polygon"
      }
    ]
  }]
})
```

---

## Step 4: Analyze Existing Code Style

**Read existing files to detect naming conventions**:

```bash
# Check for existing services
EXISTING_SERVICES=$(find backend/src/service -name "*_service.gcl" 2>/dev/null | head -3)

# Check for existing models
EXISTING_MODELS=$(find backend/src/model -name "*.gcl" 2>/dev/null | head -3)

# Check for existing APIs
EXISTING_APIS=$(find backend/src/api -name "*_api.gcl" 2>/dev/null | head -3)
```

**Detect patterns** using Read tool:
- Indentation (tabs vs spaces, 2/4 spaces)
- Line width (from project.gcl @format_line_width)
- Naming conventions (snake_case files confirmed)
- Error handling style (try/catch vs early return)

---

## Step 5: Generate Files

### A. Generate Model File

**File**: `backend/src/model/{entity_name_snake}.gcl`

**Template for CRUD**:

```gcl
// {EntityName} model and global indices
type {EntityName} {
    {field1}: {Type1};
    {field2}: {Type2};
    // ... all fields
}

// Global indices
var {entity_plural}_by_id: nodeIndex<int, node<{EntityName}>>;
{optional_additional_indices}

// ID counter for auto-increment
var {entity}_id_counter: node<int?>;
```

**Example output** for `Device`:

```gcl
// Device model and global indices
type Device {
    id: int;
    name: String;
    location: geo;
    status: String?;
    created_at: time;
}

// Global indices
var devices_by_id: nodeIndex<int, node<Device>>;
var devices_by_name: nodeIndex<String, node<Device>>;

// ID counter
var device_id_counter: node<int?>;
```

**Use Write tool** to create the file.

### B. Generate Service File

**File**: `backend/src/service/{entity_name_snake}_service.gcl`

**Template**:

```gcl
// {EntityName} service - business logic and CRUD operations
abstract type {EntityName}Service {

    static fn create({params}): node<{EntityName}> {
        // Validation
        {validation_logic}

        // Generate ID
        var id = ({entity}_id_counter.resolve() ?? 0) + 1;
        {entity}_id_counter.set(id);

        // Create entity
        var {entity} = node<{EntityName}>{ {EntityName} {
            id: id,
            {field_assignments}
            created_at: Time::now()
        }};

        // Store in indices
        {entity_plural}_by_id.set(id, {entity});
        {additional_index_inserts}

        return {entity};
    }

    static fn find_by_id(id: int): node<{EntityName}>? {
        return {entity_plural}_by_id.get(id);
    }

    {additional_find_methods}

    static fn list_all(): Array<node<{EntityName}>> {
        var results = Array<node<{EntityName}>> {};
        for (id, {entity} in {entity_plural}_by_id) {
            results.add({entity});
        }
        return results;
    }

    static fn update_{field}({entity}: node<{EntityName}>, new_{field}: {Type}) {
        // Update index if needed
        {index_update_logic}

        // Update field
        {entity}->{field} = new_{field};
    }

    static fn delete({entity}: node<{EntityName}>) {
        {entity_plural}_by_id.remove({entity}->id);
        {additional_index_removals}
    }
}
```

**Example for Device**:

```gcl
// Device service - business logic and CRUD operations
abstract type DeviceService {

    static fn create(name: String, lat: float, lng: float, status: String?): node<Device> {
        // Validation
        if (devices_by_name.get(name) != null) {
            throw "Device with name '${name}' already exists";
        }

        // Generate ID
        var id = (device_id_counter.resolve() ?? 0) + 1;
        device_id_counter.set(id);

        // Create device
        var device = node<Device>{ Device {
            id: id,
            name: name,
            location: geo { lat: lat, lng: lng },
            status: status,
            created_at: Time::now()
        }};

        // Store in indices
        devices_by_id.set(id, device);
        devices_by_name.set(name, device);

        return device;
    }

    static fn find_by_id(id: int): node<Device>? {
        return devices_by_id.get(id);
    }

    static fn find_by_name(name: String): node<Device>? {
        return devices_by_name.get(name);
    }

    static fn list_all(): Array<node<Device>> {
        var results = Array<node<Device>> {};
        for (id, device in devices_by_id) {
            results.add(device);
        }
        return results;
    }

    static fn update_name(device: node<Device>, new_name: String) {
        // Remove from name index
        devices_by_name.remove(device->name);

        // Update field
        device->name = new_name;

        // Re-add to name index
        devices_by_name.set(new_name, device);
    }

    static fn update_status(device: node<Device>, new_status: String?) {
        device->status = new_status;
    }

    static fn delete(device: node<Device>) {
        devices_by_id.remove(device->id);
        devices_by_name.remove(device->name);
    }
}
```

**Use Write tool** to create the file.

### C. Generate API File

**File**: `backend/src/api/{entity_name_snake}_api.gcl`

**Template**:

```gcl
// {EntityName} REST API endpoints

// Request/Response types
@volatile type {EntityName}View {
    {field1}: {Type1};
    {field2}: {Type2};
    // ... all fields
}

@volatile type {EntityName}Create {
    {field1}: {Type1};
    {field2}: {Type2};
    // ... fields except id, created_at
}

@volatile type {EntityName}Update {
    {field1}?: {Type1};
    {field2}?: {Type2};
    // ... updatable fields as nullable
}

// Endpoints
@expose
@permission("public")
fn get_{entity_plural}(): Array<{EntityName}View> {
    var views = Array<{EntityName}View> {};
    var {entity_plural} = {EntityName}Service::list_all();

    for ({entity} in {entity_plural}) {
        views.add({EntityName}View {
            {field_mappings}
        });
    }

    return views;
}

@expose
@permission("public")
fn get_{entity}_by_id(id: int): {EntityName}View {
    var {entity} = {EntityName}Service::find_by_id(id);
    if ({entity} == null) {
        throw "{EntityName} not found";
    }

    return {EntityName}View {
        {field_mappings}
    };
}

@expose
@permission("admin")
fn create_{entity}(data: {EntityName}Create): {EntityName}View {
    var {entity} = {EntityName}Service::create({param_list});

    return {EntityName}View {
        {field_mappings}
    };
}

@expose
@permission("admin")
fn update_{entity}(id: int, data: {EntityName}Update): {EntityName}View {
    var {entity} = {EntityName}Service::find_by_id(id);
    if ({entity} == null) {
        throw "{EntityName} not found";
    }

    {update_calls}

    return {EntityName}View {
        {field_mappings}
    };
}

@expose
@permission("admin")
fn delete_{entity}(id: int) {
    var {entity} = {EntityName}Service::find_by_id(id);
    if ({entity} == null) {
        throw "{EntityName} not found";
    }

    {EntityName}Service::delete({entity});
}
```

**Example for Device**:

```gcl
// Device REST API endpoints

// Request/Response types
@volatile type DeviceView {
    id: int;
    name: String;
    location: geo;
    status: String?;
    created_at: time;
}

@volatile type DeviceCreate {
    name: String;
    lat: float;
    lng: float;
    status: String?;
}

@volatile type DeviceUpdate {
    name: String?;
    status: String?;
}

// Endpoints
@expose
@permission("public")
fn get_devices(): Array<DeviceView> {
    var views = Array<DeviceView> {};
    var devices = DeviceService::list_all();

    for (device in devices) {
        views.add(DeviceView {
            id: device->id,
            name: device->name,
            location: device->location,
            status: device->status,
            created_at: device->created_at
        });
    }

    return views;
}

@expose
@permission("public")
fn get_device_by_id(id: int): DeviceView {
    var device = DeviceService::find_by_id(id);
    if (device == null) {
        throw "Device not found";
    }

    return DeviceView {
        id: device->id,
        name: device->name,
        location: device->location,
        status: device->status,
        created_at: device->created_at
    };
}

@expose
@permission("admin")
fn create_device(data: DeviceCreate): DeviceView {
    var device = DeviceService::create(data.name, data.lat, data.lng, data.status);

    return DeviceView {
        id: device->id,
        name: device->name,
        location: device->location,
        status: device->status,
        created_at: device->created_at
    };
}

@expose
@permission("admin")
fn update_device(id: int, data: DeviceUpdate): DeviceView {
    var device = DeviceService::find_by_id(id);
    if (device == null) {
        throw "Device not found";
    }

    if (data.name != null) {
        DeviceService::update_name(device, data.name!!);
    }

    if (data.status != null) {
        DeviceService::update_status(device, data.status);
    }

    return DeviceView {
        id: device->id,
        name: device->name,
        location: device->location,
        status: device->status,
        created_at: device->created_at
    };
}

@expose
@permission("admin")
fn delete_device(id: int) {
    var device = DeviceService::find_by_id(id);
    if (device == null) {
        throw "Device not found";
    }

    DeviceService::delete(device);
}
```

**Use Write tool** to create the file.

### D. Generate Test File

**File**: `backend/test/{entity_name_snake}_test.gcl`

**Template**:

```gcl
// {EntityName} tests

@test
fn test_{entity}_create() {
    var {entity} = {EntityName}Service::create({test_params});

    Assert::isNotNull({entity});
    Assert::equals({entity}->field1, expected_value1);
    Assert::equals({entity}->field2, expected_value2);
}

@test
fn test_{entity}_find_by_id() {
    var {entity} = {EntityName}Service::create({test_params});
    var found = {EntityName}Service::find_by_id({entity}->id);

    Assert::isNotNull(found);
    Assert::equals(found->id, {entity}->id);
}

@test
fn test_{entity}_find_{unique_field}() {
    var {entity} = {EntityName}Service::create({test_params});
    var found = {EntityName}Service::find_by_{field}({entity}->{field});

    Assert::isNotNull(found);
    Assert::equals(found->{field}, {entity}->{field});
}

@test
fn test_{entity}_list_all() {
    var {entity}1 = {EntityName}Service::create({test_params1});
    var {entity}2 = {EntityName}Service::create({test_params2});

    var all = {EntityName}Service::list_all();

    Assert::isTrue(all.size() >= 2);
}

@test
fn test_{entity}_update() {
    var {entity} = {EntityName}Service::create({test_params});

    {EntityName}Service::update_field({entity}, new_value);

    Assert::equals({entity}->field, new_value);
}

@test
fn test_{entity}_delete() {
    var {entity} = {EntityName}Service::create({test_params});
    var id = {entity}->id;

    {EntityName}Service::delete({entity});

    var found = {EntityName}Service::find_by_id(id);
    Assert::isNull(found);
}

@test
fn test_{entity}_duplicate_validation() {
    var {entity}1 = {EntityName}Service::create({test_params});

    var failed = false;
    try {
        var {entity}2 = {EntityName}Service::create({test_params});  // Same unique field
    } catch (ex) {
        failed = true;
    }

    Assert::isTrue(failed);
}
```

**Example for Device**:

```gcl
// Device tests

@test
fn test_device_create() {
    var device = DeviceService::create("Test Device", 48.8566, 2.3522, "active");

    Assert::isNotNull(device);
    Assert::equals(device->name, "Test Device");
    Assert::equals(device->status, "active");
}

@test
fn test_device_find_by_id() {
    var device = DeviceService::create("Test Device", 48.8566, 2.3522, "active");
    var found = DeviceService::find_by_id(device->id);

    Assert::isNotNull(found);
    Assert::equals(found->id, device->id);
}

@test
fn test_device_find_by_name() {
    var device = DeviceService::create("Unique Device", 48.8566, 2.3522, "active");
    var found = DeviceService::find_by_name("Unique Device");

    Assert::isNotNull(found);
    Assert::equals(found->name, "Unique Device");
}

@test
fn test_device_list_all() {
    var device1 = DeviceService::create("Device 1", 48.8566, 2.3522, "active");
    var device2 = DeviceService::create("Device 2", 51.5074, -0.1278, "inactive");

    var all = DeviceService::list_all();

    Assert::isTrue(all.size() >= 2);
}

@test
fn test_device_update_name() {
    var device = DeviceService::create("Old Name", 48.8566, 2.3522, "active");

    DeviceService::update_name(device, "New Name");

    Assert::equals(device->name, "New Name");

    var found = DeviceService::find_by_name("New Name");
    Assert::isNotNull(found);
}

@test
fn test_device_update_status() {
    var device = DeviceService::create("Test Device", 48.8566, 2.3522, "active");

    DeviceService::update_status(device, "inactive");

    Assert::equals(device->status, "inactive");
}

@test
fn test_device_delete() {
    var device = DeviceService::create("To Delete", 48.8566, 2.3522, "active");
    var id = device->id;

    DeviceService::delete(device);

    var found = DeviceService::find_by_id(id);
    Assert::isNull(found);
}

@test
fn test_device_duplicate_name() {
    var device1 = DeviceService::create("Duplicate", 48.8566, 2.3522, "active");

    var failed = false;
    try {
        var device2 = DeviceService::create("Duplicate", 51.5074, -0.1278, "active");
    } catch (ex) {
        failed = true;
    }

    Assert::isTrue(failed);
}
```

**Use Write tool** to create the file.

---

## Step 6: Run greycat-lang lint --fix

**Run linter immediately** to catch any errors:

```bash
echo "================================================================================"
echo "RUNNING LINTER"
echo "================================================================================"
echo ""

greycat-lang lint --fix

LINT_EXIT=$?

if [ $LINT_EXIT -eq 0 ]; then
    echo ""
    echo "✓ All files passed lint"
else
    echo ""
    echo "⚠ Lint found errors - please review and fix"
fi
```

---

## Step 7: Generate Report

**Summarize generated files**:

```
===============================================================================
SCAFFOLD COMPLETE
===============================================================================

Generated files for entity: Device

✓ backend/src/model/device.gcl (32 lines)
  - Device type with 5 fields
  - 2 global indices (by_id, by_name)
  - ID counter

✓ backend/src/service/device_service.gcl (87 lines)
  - create, find_by_id, find_by_name, list_all
  - update_name, update_status
  - delete with validation

✓ backend/src/api/device_api.gcl (98 lines)
  - 3 volatile types (DeviceView, DeviceCreate, DeviceUpdate)
  - 5 API endpoints (@expose):
    - GET get_devices() [@permission("public")]
    - GET get_device_by_id(id) [@permission("public")]
    - POST create_device(data) [@permission("admin")]
    - PUT update_device(id, data) [@permission("admin")]
    - DELETE delete_device(id) [@permission("admin")]

✓ backend/test/device_test.gcl (112 lines)
  - 8 test cases covering CRUD and validation

===============================================================================

Lint: ✓ All files passed

Next steps:
  1. Review generated code and customize as needed
  2. Run tests: greycat test backend/test/device_test.gcl
  3. Start server: greycat serve
  4. Test endpoints:
     curl http://localhost:8080/create_device -d '{"name":"Test","lat":48.8,"lng":2.3,"status":"active"}'
     curl http://localhost:8080/get_devices

===============================================================================
```

---

## Template Variations

### Time-series Collector Template

**Model differences**:
```gcl
type Sensor {
    id: String;
    location: geo;
    readings: nodeTime<float>;  // Time-series data
}

var sensors_by_id: nodeIndex<String, node<Sensor>>;
```

**Service additions**:
```gcl
static fn record_reading(sensor: node<Sensor>, value: float, timestamp: time) {
    sensor->readings.setAt(timestamp, value);
}

static fn get_readings(sensor: node<Sensor>, start: time, end: time): Array<Tuple<time, float>> {
    var results = Array<Tuple<time, float>> {};
    for (t: time, val: float in sensor->readings[start..end]) {
        results.add(Tuple { first: t, second: val });
    }
    return results;
}

static fn get_average(sensor: node<Sensor>, start: time, end: time): float {
    var sum = 0.0;
    var count = 0;
    for (t: time, val: float in sensor->readings[start..end]) {
        sum = sum + val;
        count = count + 1;
    }
    return if (count > 0) { sum / count } else { 0.0 };
}
```

### Graph Traversal Template

**Model with relationships**:
```gcl
type City {
    id: int;
    name: String;
    country: node<Country>;
    streets: nodeList<node<Street>>;
}

type Street {
    id: int;
    name: String;
    city: node<City>;
    buildings: nodeList<node<Building>>;
}
```

**Traversal queries**:
```gcl
static fn get_city_with_streets(city: node<City>): CityWithStreetsView {
    var street_views = Array<StreetView> {};
    for (i, street in city->streets) {
        street_views.add(StreetView {
            id: street->id,
            name: street->name
        });
    }

    return CityWithStreetsView {
        id: city->id,
        name: city->name,
        streets: street_views
    };
}
```

---

## Success Criteria

✓ **All files generated** with proper structure
✓ **Follows GreyCat conventions** (naming, initialization, persistence)
✓ **greycat-lang lint --fix passes** with 0 errors
✓ **Tests comprehensive** covering CRUD and edge cases
✓ **API layer proper** (@volatile types, never return nodeList)
✓ **Service validation** included
✓ **Indices consistent** maintained across operations

---

## Notes

- **Customization encouraged**: Generated code is a starting point
- **Existing code style**: Command detects and matches project conventions
- **Multiple indices**: Maintains consistency across all indices
- **Validation**: Includes duplicate checks for unique fields
- **Permissions**: Uses @permission("public") for reads, "admin" for writes
- **Error handling**: Throws exceptions for not found / validation failures
