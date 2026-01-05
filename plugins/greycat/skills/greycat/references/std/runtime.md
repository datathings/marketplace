# [std](index.html) > runtime

[Source](runtime.source.html)

Runtime module providing task execution, security, scheduling, system operations, and server management for GreyCat applications.

## Task Execution & Parallelism

### Job<T>
Represents a unit of computation executed in parallel, awaited by a parent Task.

```gcl
// Create jobs for parallel execution
var job1 = Job<int> { function: compute_sum, arguments: [array1] };
var job2 = Job<int> { function: compute_sum, arguments: [array2] };

// Execute jobs in parallel
await([job1, job2], MergeStrategy::strict);

// Retrieve results
var result1 = job1.result();
var result2 = job2.result();
```

### MergeStrategy
Controls how node updates from jobs merge with the main graph.

- **strict**: All or nothing. Any conflict throws an exception. Provides strongest consistency guarantee (recommended default).
- **first_wins**: Partial strategy. Conflicts resolved using previously inserted values. Non-blocking.
- **last_wins**: Partial strategy. Conflicts resolved by overriding with current values. Non-blocking.

```gcl
// Safe merge: fail on conflicts
await(jobs, MergeStrategy::strict);

// Optimistic merge: keep first value on conflict
await(jobs, MergeStrategy::first_wins);

// Force merge: always use latest value
await(jobs, MergeStrategy::last_wins);
```

### Task
Represents an executing or completed task with metadata and status tracking.

```gcl
// Get current task information
var task_id = Task::id();
var parent_id = Task::parentId();

// Report progress (0.0 to 1.0)
Task::progress(0.5); // 50% complete

// Query running tasks
var active_tasks = Task::running();
for (_, task in active_tasks) {
    println("Task ${task.task_id}: ${task.status}");
    println("  User: ${task.user_id}, Duration: ${task.duration}");
}

// Get task history
var past_tasks = Task::history(0, 100); // offset, max

// Check if specific task is running
var is_active = Task::is_running(task_id);

// Cancel a running task
var cancelled = Task::cancel(task_id);
```

### TaskStatus
Enum representing task lifecycle states: empty, waiting, running, await, cancelled, error, ended, ended_with_errors, breakpoint.

## Scheduler

The scheduler manages recurring tasks that execute automatically without user intervention. Essential for maintenance operations, data processing, and system health checks.

### Basic Scheduling

```gcl
fn backup_database() {
    // Perform database backup
    Runtime::backup_delta();
}

fn schedule_backups() {
    // Schedule backup every day at 2 AM
    Scheduler::add(
        backup_database,
        DailyPeriodicity { hour: 2 },
        null
    );
}
```

### Advanced Scheduling

```gcl
fn health_check() {
    // Check system health and log results
    var info = Runtime::info();
    info("Memory usage: ${info.mem_worker} / ${info.mem_total}");
}

fn schedule_health_checks() {
    // Schedule health checks every 5 minutes, starting in 1 hour
    Scheduler::add(
        health_check,
        FixedPeriodicity { every: 5min },
        PeriodicOptions {
            start: time::now() + 1hour,
            max_duration: 30s,
            activated: true
        }
    );
}
```

### Managing Scheduled Tasks

```gcl
fn manage_scheduled_tasks() {
    // Find a specific task
    var ptask = Scheduler::find(health_check);
    if (ptask != null) {
        println("Health check runs every ${ptask.periodicity.every}");
        println("Next execution: ${ptask.next_execution}");
        println("Executed ${ptask.execution_count} times");
    }

    // Temporarily disable a task
    Scheduler::deactivate(backup_database);

    // Re-enable it later
    Scheduler::activate(backup_database);

    // List all scheduled tasks
    var all_tasks = Scheduler::list();
    for (_, ptask in all_tasks) {
        println("${ptask.function}: active=${ptask.is_active}");
    }
}
```

### Periodicity Types

#### FixedPeriodicity
Tasks that repeat at fixed intervals.

```gcl
// Every 30 minutes
FixedPeriodicity { every: 30min }

// Every 2 hours
FixedPeriodicity { every: 2hour }

// Every day (24 hours)
FixedPeriodicity { every: 24hour }
```

#### DailyPeriodicity
Tasks executed daily at specific times. All fields default to 0 (midnight).

```gcl
// Run at 2:30 PM every day
DailyPeriodicity { hour: 14, minute: 30 }

// Run at midnight (all defaults)
DailyPeriodicity {}

// Run at 9 AM Europe/Luxembourg time
DailyPeriodicity {
    hour: 9,
    timezone: TimeZone::"Europe/Luxembourg"
}
```

#### WeeklyPeriodicity
Tasks that run on specific days of the week.

```gcl
// Every Monday and Friday at 9 AM
WeeklyPeriodicity {
    days: [DayOfWeek::Mon, DayOfWeek::Fri],
    daily: DailyPeriodicity { hour: 9 }
}

// Every weekday at midnight
WeeklyPeriodicity {
    days: [
        DayOfWeek::Mon,
        DayOfWeek::Tue,
        DayOfWeek::Wed,
        DayOfWeek::Thu,
        DayOfWeek::Fri
    ]
}
```

#### MonthlyPeriodicity
Tasks that run monthly on specific days.

```gcl
// 15th of every month at 2 PM
MonthlyPeriodicity {
    days: [15],
    daily: DailyPeriodicity { hour: 14 }
}

// First and last day of month at midnight
MonthlyPeriodicity {
    days: [1, -1] // -1 means last day
}

// Three days a month at 9:30 PM
MonthlyPeriodicity {
    days: [1, 15, -1],
    daily: DailyPeriodicity { hour: 9, minute: 30 }
}
```

#### YearlyPeriodicity
Tasks that run on specific calendar dates each year.

```gcl
// New Year's Day and Christmas
YearlyPeriodicity {
    dates: [
        DateTuple { day: 1, month: Month::Jan },
        DateTuple { day: 25, month: Month::Dec }
    ]
}

// Quarterly reports (1st of each quarter)
YearlyPeriodicity {
    dates: [
        DateTuple { day: 1, month: Month::Jan },
        DateTuple { day: 1, month: Month::Apr },
        DateTuple { day: 1, month: Month::Jul },
        DateTuple { day: 1, month: Month::Oct }
    ]
}
```

## Runtime & System Information

### Runtime
System runtime operations and configuration.

```gcl
// Get runtime information
var info = Runtime::info();
println("GreyCat version: ${info.version}");
println("Program version: ${info.program_version}");
println("Architecture: ${info.arch}");
println("Timezone: ${info.timezone}");
println("License: ${info.license.type} (${info.license.company})");
println("IO threads: ${info.io_threads}");
println("Background threads: ${info.bg_threads}");
println("Foreground threads: ${info.fg_threads}");
println("Total memory: ${info.mem_total} MB");
println("Worker memory: ${info.mem_worker} MB");
println("Disk data: ${info.disk_data_bytes} bytes");

// Access graph root
var root = Runtime::root();

// Get ABI (Application Binary Interface)
var abi_info = Runtime::abi();

// Sleep current thread
Runtime::sleep(1s);

// Backup operations
Runtime::backup_full(); // Full backup
Runtime::backup_delta(); // Incremental backup

// Defragmentation
Runtime::defrag();
```

### System
System-level operations including process execution and environment access.

```gcl
// Execute command and wait for result
var output = System::exec("/usr/bin/date", ["+%Y-%m-%d"]);
println("Current date: ${output}");

// Spawn background process
var child = System::spawn("/path/to/script.sh", ["--option", "value"]);

// Wait for completion
var result = child.wait();
println("Exit code: ${result.code}");
println("Output: ${result.stdout}");
println("Errors: ${result.stderr}");

// Or kill the process
child.kill();

// Get timezone
var tz = System::tz();

// Access environment variables
var home = System::getEnv("HOME");
var path = System::getEnv("PATH");
```

### License
License information structure.

```gcl
var license = Runtime::info().license;
println("Type: ${license.type}"); // community, enterprise, testing
println("Name: ${license.name}");
println("Company: ${license.company}");
println("Valid from: ${license.start}");
println("Valid until: ${license.end}");
println("Max memory: ${license.max_memory} MB");
```

## Security & Authentication

### User
User management and authentication.

```gcl
// Login with credentials (returns session token)
var token = User::login("username:password", true); // use_cookie = true

// JWT token login
var session_token = User::tokenLogin(jwt_token, true);

// Logout
User::logout();

// Renew session
var new_token = User::renew(true);

// Get current user info
var current_user_id = User::current();
var me = User::me();
println("Logged in as: ${me.full_name} (${me.email})");
println("Role: ${me.role}");

// Check permissions
var user_permissions = User::permissions();
var has_admin = User::hasPermission("admin");
var has_debug = User::hasPermission("debug");

// Get user by name or ID
var user = User::getByName("alice");
var user_by_id = User::get(1);

// Password management
User::setPassword("alice", "new_secure_password");
var password_valid = User::checkPassword("alice", "test_password");
```

### UserGroup
Groups for organizing users with shared permissions.

```gcl
// All security entities (users and groups)
var entities = SecurityEntity::all();
for (_, entity in entities) {
    println("${entity.name}: active=${entity.activated}");
}

// Create or update entity
var group = UserGroup {
    name: "developers",
    activated: true
};
var group_id = SecurityEntity::set(group);
```

### UserGroupPolicy
Policies defining group access levels.

```gcl
var policy = UserGroupPolicy {
    group_id: 1,
    type: UserGroupPolicyType::write
};

// Policy types: read, write, execute
```

### OpenIDConnect
OpenID Connect authentication configuration.

```gcl
// Get current OIDC configuration
var oidc_config = OpenIDConnect::config();
if (oidc_config != null) {
    println("OIDC URL: ${oidc_config.url}");
    println("Client ID: ${oidc_config.clientId}");
}
```

### Permission & Role
Access control definitions (defined at module level with @permission and @role annotations).

```gcl
// Predefined permissions
// @permission("public", "default, associated with anonymous users")
// @permission("admin", "allows to administrate anything on the server")
// @permission("api", "allows access to exposed functions and webroot files")
// @permission("debug", "allows access to low-level graph manipulation functions")
// @permission("files", "allows access to files under /files/* or webroot according to ACL")

// Predefined roles
// @role("public", "public")
// @role("admin", "public", "admin", "api", "debug", "files")
// @role("user", "public", "api", "files")

// Query all permissions and roles
var all_permissions = Permission::all();
var all_roles = Role::all();
```

## Logging

### Log Types

```gcl
// Log levels
enum LogLevel {
    error;
    warn;
    info;
    perf;
    trace;
}

// Log structure
type Log {
    level: LogLevel;
    time: time;
    user_id: int?;
    id: int?;
    id2: int?;
    src: function?;
    data: any?;
}

// Usage data logging
type LogDataUsage {
    read_bytes: int;
    read_hits: int;
    read_wasted: int;
    write_bytes: int;
    write_hits: int;
    cache_bytes: int;
    cache_hits: int;
}
```

## Debugging

### Debug
Checkpoint snapshots for debugging suspended tasks.

```gcl
// Get all checkpoint IDs
var checkpoint_ids = Debug::all();

// Get specific checkpoint
var debug_info = Debug::get(checkpoint_id);
println("Frames: ${debug_info.frames.size()}");

for (_, frame in debug_info.frames) {
    println("${frame.module}::${frame.type}::${frame.function}");
    println("  ${frame.src}:${frame.line}:${frame.column}");

    for (_, var in frame.scope) {
        println("  ${var.name} = ${var.value}");
    }
}

// Resume execution from checkpoint
Debug::resume(checkpoint_id);
```

## API Documentation

### OpenApi
Generate OpenAPI v3 specifications from exposed functions.

```gcl
// Get OpenAPI specification
var spec = OpenApi::v3();
println("OpenAPI version: ${spec.openapi}");
println("API title: ${spec.info.title}");
println("API version: ${spec.info.version}");

// Iterate through paths
if (spec.paths != null) {
    for (path, item in spec.paths) {
        println("Path: ${path}");
        if (item.post != null) {
            println("  Description: ${item.post.description}");
        }
    }
}
```

## Model Context Protocol (MCP)

GreyCat supports the Model Context Protocol for LLM integration. MCP enables AI assistants to interact with GreyCat tools and resources.

### MCP Initialization

```gcl
// Initialize MCP session
var init_params = McpInitializeParams {
    protocolVersion: "2025-06-18",
    capabilities: McpClientCapabilities {
        roots: McpClientRoots { listChanged: true }
    },
    clientInfo: McpImplementation {
        name: "my-client",
        version: "1.0.0"
    }
};

var result = mcp_initialize(init_params);
println("Server protocol version: ${result.protocolVersion}");
println("Server: ${result.serverInfo.name} ${result.serverInfo.version}");
```

### MCP Tools

```gcl
// List available tools
var tools_list = mcp_tools_list(null);
for (_, tool in tools_list.tools) {
    println("Tool: ${tool.name}");
    println("  Title: ${tool.title}");
    println("  Description: ${tool.description}");
}

// Call a tool
var call_params = McpToolsCallParams {
    name: "my_tool",
    arguments: { param1: "value1", param2: 42 }
};

var call_result = mcp_tools_call(call_params);
if (call_result.isError) {
    error("Tool call failed");
} else {
    for (_, content in call_result.content) {
        if (content.type == McpContentType::text) {
            var text_content = content as McpTextContent;
            println(text_content.text);
        }
    }
}
```

### MCP Content Types

MCP supports various content types for tool results:

```gcl
// Text content
var text = McpTextContent {
    type: McpContentType::text,
    text: "Hello from GreyCat!"
};

// Image content (base64 encoded)
var image = McpImageContent {
    type: McpContentType::image,
    data: base64_image_data,
    mimeType: "image/png"
};

// Audio content (base64 encoded)
var audio = McpAudioContent {
    type: McpContentType::audio,
    data: base64_audio_data,
    mimeType: "audio/mp3"
};

// Resource content
var resource = McpResourceContent {
    type: McpContentType::resource,
    uri: "file:///path/to/resource",
    description: "Important data file",
    mimeType: "application/json",
    size: 1024
};
```

### MCP Annotations

```gcl
// Add metadata to content
var annotations = McpAnnotations {
    audience: [McpRole::user, McpRole::assistant],
    priority: McpPriority::MostImportant,
    lastModified: "2025-01-03T10:30:00Z"
};

var annotated_text = McpTextContent {
    type: McpContentType::text,
    text: "Critical information",
    annotations: annotations
};
```

## Supporting Types

### PeriodicTask
Represents a scheduled periodic task.

```gcl
type PeriodicTask {
    function: function;          // Function to execute
    periodicity: Periodicity;    // Schedule configuration
    options: PeriodicOptions;    // Execution options
    is_active: bool;            // Currently active?
    next_execution: time;       // Next scheduled time
    execution_count: int;       // Total executions
}
```

### PeriodicOptions
Configuration for periodic task behavior.

```gcl
type PeriodicOptions {
    activated: bool?;        // Initially active (default: true)
    start: time?;           // Lifecycle start time (default: now)
    max_duration: duration?; // Max execution time (default: null = unlimited)
}
```

### SecurityPolicy
Complete security policy structure.

```gcl
type SecurityPolicy {
    entities: Array<SecurityEntity>?;
    credentials: Map<String, UserCredential>?;
    fields: SecurityFields?;
    keys: Map<String, String>?;
    keys_last_refresh: time?;
}
```

### SecurityFields
Customizable security-related user fields.

```gcl
// Get current security fields configuration
var fields = SecurityFields::get();

// Set security fields configuration
SecurityFields::set(SecurityFields {
    email: "example@company.com",
    name: "Company LDAP",
    first_name: "givenName",
    last_name: "sn",
    roles: roles_map,
    groups: groups_map
});
```
