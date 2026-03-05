# Concurrency & Tasks

## Jobs (Parallel Execution)

Jobs enable parallel computation with fork-join pattern:

```gcl
fn long_computation(max: int): int {
    var count = 0;
    for (var i = 0; i < max; i++) { count++; }
    return count;
}

fn main() {
    var jobs = Array<Job>{
        Job { function: project::long_computation, arguments: [100_000] },
        Job { function: project::long_computation, arguments: [100_000] }
    };

    await(jobs);  // Blocks until all complete

    for (_, job in jobs) { var result = job.result(); }
}
```

> Jobs only run in parallel when executed within a task context.

### Error Handling

`await` throws if any job fails. Handle individually:

```gcl
fn main() {
    var jobs = Array<Job>{
        Job { function: foo, arguments: [10_s, false] },
        Job { function: foo, arguments: [1_s, true] }  // will fail
    };

    try {
        await(jobs);
    } catch (err) {
        for (i, job in jobs) {
            var res = job.result();
            if (res is Error) { println("Job ${i} failed"); }
            else { println("Job ${i} finished"); }
        }
    }
}
```

### Parallel Writes

Can write to different nodes in parallel, but NOT to the same node:

```gcl
var sensor_list: nodeList<node<Sensor>>;

fn main() {
    var jobs = Array<Job>{ Job { function: project::import }, Job { function: project::import } };
    await(jobs);

    // Aggregate results after await
    for (_, job in jobs) {
        var sensors = job.result();
        for (_, sensor: node<Sensor> in sensors) { sensor_list.add(sensor); }
    }
}

fn import(): Array<node<Sensor>> {
    var sensors = Array<node<Sensor>>{};
    for (var i = 0; i < 10; i++) {
        sensors.add(node<Sensor>{ Sensor { history: nodeTime<int>{} }});
    }
    return sensors;
}
```

### Await Limitations

Objects resolved before `await` become invalid after:

```gcl
// ❌ Wrong - resolved_foo becomes stale after await
fn task(foo: node<Foo>) {
    var resolved_foo = foo.resolve();
    await(jobs);
    resolved_foo.status = "Done";  // ERROR
}

// ✅ Correct - use arrow operator or set to null before await
fn task(foo: node<Foo>) {
    var resolved_foo = foo.resolve();
    resolved_foo = null;  // Clear before await
    await(jobs);
    foo->status = "Done";  // Use arrow operator
}
```

## Tasks (Async HTTP)

Execute functions asynchronously via HTTP header:

```bash
curl -H "task:''" -X POST -d '[]' http://localhost:8080/project::long_computation
# Returns Task object immediately
```

### Task Object

| Field | Type | Description |
|-------|------|-------------|
| user_id | int | User who spawned task |
| task_id | int | Unique task ID |
| mod | String? | Task module |
| fun | String? | Function name |
| creation | time | When spawned |
| start | time? | When started |
| duration | duration? | How long |
| status | TaskStatus | Current status |

**Task Status**: `empty → waiting → running → await → ended/error/cancelled/ended_with_errors`

**Check Status**:
```bash
curl -X POST -d '[1,1]' http://localhost:8080/runtime::Task::info
```

**Retrieve Result**:
```bash
curl -X GET 'http://localhost:8080/files/0/tasks/1/result.gcb?json'
```

## Periodic Tasks

Schedule recurring tasks using the `Scheduler` API:

```gcl
fn my_task() { println("Current time: ${time::now()}"); }

fn main() {
    // Daily at midnight
    Scheduler::add(project::my_task, DailyPeriodicity {}, null);

    // Fixed interval
    Scheduler::add(project::my_task, FixedPeriodicity { every: 1_day }, PeriodicOptions { start: time::now() });
}
```

### Periodicity Types

| Type | Description | Example |
|------|-------------|---------|
| `FixedPeriodicity` | Fixed intervals | `{ every: 30min }`, `{ every: 2hour }` |
| `DailyPeriodicity` | Daily at specific time | `{ hour: 14, minute: 30 }`, `{ hour: 9, timezone: TimeZone::"Europe/Luxembourg" }` |
| `WeeklyPeriodicity` | Specific weekdays | `{ days: [DayOfWeek::Mon, DayOfWeek::Fri], daily: DailyPeriodicity { hour: 9 } }` |
| `MonthlyPeriodicity` | Specific days of month | `{ days: [15], daily: DailyPeriodicity { hour: 14 } }` |
| `YearlyPeriodicity` | Specific calendar dates | `{ dates: [DateTuple { day: 1, month: Month::Jan }], daily: DailyPeriodicity { hour: 0 } }` |

### Periodic Options

```gcl
PeriodicOptions {
    activated: true,              // Task is active (default: true)
    start: time::now() + 1hour,   // Delay first execution
    max_duration: 30s             // Timeout per execution
}
```

### Manage Scheduled Tasks

```gcl
var tasks = Scheduler::list();  // List all tasks
var task = Scheduler::find(project::my_task);  // Find specific task
Scheduler::deactivate(project::my_task);  // Deactivate
Scheduler::activate(project::my_task);    // Activate
```

> Adding a task with the same function replaces the existing configuration.

### PeriodicTask Fields

| Field | Type | Description |
|-------|------|-------------|
| function | function | The scheduled function |
| periodicity | Periodicity | Periodicity configuration |
| options | PeriodicOptions | Applied options |
| is_active | bool | Whether task is active |
| next_execution | time | Next scheduled time |
| execution_count | int | Total executions |
