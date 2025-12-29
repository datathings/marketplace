# [std](index.html) > runtime

[Source](runtime.source.html)

## Scheduler
The scheduler manages recurring tasks that need to happen automatically without user intervention. This is essential for maintenance operations, data processing, and system health checks.

### Basic Scheduling
The scheduler works by associating functions with periodicity rules. Each function can only have one scheduled task at a time, which keeps the system simple and predictable:

```gcl
fn backup_database() {
    // perform database backup...
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

### Advanced Scheduling Options
For more complex scenarios, you can control exactly when a task starts, how long they're allowed to run, and whether they're initially active:

```gcl
fn health_check() {
    // check system health and log results...
}

fn schedule_health_checks() {
    // Schedule health checks every 5 minutes
    // Start in 1 hour, limit each check to 30 seconds
    Scheduler::add(
        health_check,
        FixedPeriodicity { every: 5min },
        PeriodicOptions { 
            start: time::now() + 1hour,
            max_duration: 30s,
        }
    );
}
```

### Managing Scheduled Tasks
You can dynamically control scheduled periodic tasks during runtime without needing to restart your application:

```gcl
fn manage_scheduled_tasks() {
    // Find a specific task
    var ptask = Scheduler::find(health_check);
    if (ptask != null) {
        print("Health check runs every ${ptask.periodicity.every}");
    }
    
    // Temporarily disable a task
    Scheduler::deactivate(backup_database);
    
    // Re-enable it later
    Scheduler::activate(backup_database);
    
    // List all scheduled tasks
    var all_tasks = Scheduler::list();
    for (_, ptask in all_tasks) {
        print("${ptask.function}: active=${ptask.is_active}");
    }
}
```
