# PostgreSQL Integration

Native PostgreSQL database integration for GreyCat with transaction support and CSV import/export.

## Overview

The PostgreSQL library provides direct database connectivity for GreyCat applications, enabling seamless integration with PostgreSQL databases. It supports standard SQL operations, transaction management, and high-performance bulk data operations through CSV import/export.

Key features include:
- **Transaction support** with begin, commit, and rollback operations
- **Query execution** with automatic result mapping to GreyCat Tables
- **CSV import** using PostgreSQL's native COPY command for high-speed data loading
- **CSV export** for efficient data extraction and reporting
- **Type safety** with native GreyCat type system integration

This library is ideal for applications that need persistent storage, complex queries, data warehousing, ETL pipelines, or integration with existing PostgreSQL databases.

## Installation

Add the PostgreSQL library to your GreyCat project:

```gcl
@library("sql", "7.6.23-dev")
```

**Note:** Ensure PostgreSQL client libraries are available in your runtime environment.

## Quick Start

### Basic Connection and Query

```gcl
var db = Postgres {
  url: "localhost",
  port: "5432",
  db_name: "myapp",
  login: "postgres",
  password: "secret"
};

// Execute a simple query
var result = db.execute("SELECT * FROM users WHERE active = true");
print(result);
```

### Transaction Example

```gcl
var db = Postgres {
  url: "localhost",
  port: "5432",
  db_name: "banking",
  login: "bank_user",
  password: "secure_password"
};

// Transfer money between accounts
db.begin();
try {
  db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1");
  db.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2");
  db.commit();
  print("Transfer completed");
} catch (e) {
  db.rollback();
  print("Transfer failed: ${e}");
}
```

## Types

### Postgres

Main database connection type with transaction and query capabilities.

**Fields:**
- `url: String` (private) - PostgreSQL server hostname or IP address
- `port: String` (private) - Server port number (typically `"5432"`)
- `db_name: String` (private) - Name of the database to connect to
- `login: String?` (private) - Username for authentication (optional)
- `password: String?` (private) - Password for authentication (optional)

**Methods:**
- `begin()` - Start a new transaction
- `commit()` - Commit the current transaction
- `rollback()` - Rollback the current transaction
- `execute(query: String): any?` - Execute SQL query, returns Table for SELECT or null
- `csv_to_table(path: String, table_name: String, cols_name: Array<String>?, sep: char)` - Import CSV to table
- `query_to_csv(path: String, query: String, sep: char)` - Export query results to CSV

**Example:**

```gcl
var db = Postgres {
  url: "db.example.com",
  port: "5432",
  db_name: "production_db",
  login: "app_user",
  password: "app_password"
};

// Check connection
var version = db.execute("SELECT version()");
print(version);
```

## Methods

### begin()

Starts a new database transaction.

**Signature:** `fn begin()`

**Behavior:**
- Initiates a new transaction context
- All subsequent `execute()` calls are part of this transaction
- Must be followed by `commit()` or `rollback()`
- Cannot use CSV operations while in a transaction

**Example:**

```gcl
db.begin();
db.execute("INSERT INTO orders (id, amount) VALUES (1, 100)");
db.execute("INSERT INTO order_items (order_id, product_id) VALUES (1, 42)");
db.commit();
```

### commit()

Commits the current transaction, making all changes permanent.

**Signature:** `fn commit()`

**Behavior:**
- Persists all changes made since `begin()`
- Releases locks and resources
- Throws an error if no transaction is active
- Ends the transaction context

**Example:**

```gcl
db.begin();
try {
  db.execute("UPDATE inventory SET quantity = quantity - 10 WHERE product_id = 5");
  db.execute("INSERT INTO sales (product_id, quantity) VALUES (5, 10)");
  db.commit();
} catch (e) {
  db.rollback();
  throw e;
}
```

### rollback()

Rolls back the current transaction, discarding all changes.

**Signature:** `fn rollback()`

**Behavior:**
- Undoes all changes made since `begin()`
- Releases locks and resources
- Safe to call even if transaction is already rolled back
- Ends the transaction context

**Example:**

```gcl
db.begin();
db.execute("DELETE FROM users WHERE last_login < '2020-01-01'");

var count = db.execute("SELECT COUNT(*) FROM users") as Table;
if (count.rows[0][0] as int < 100) {
  // Too many users deleted, rollback
  db.rollback();
  print("Rollback: would have deleted too many users");
} else {
  db.commit();
}
```

### execute()

Executes a SQL query and returns results for SELECT statements.

**Signature:** `fn execute(query: String): any?`

**Parameters:**
- `query: String` - SQL statement to execute

**Returns:**
- `Table` for SELECT queries containing result rows
- `null` for INSERT, UPDATE, DELETE, CREATE, etc.

**Example:**

```gcl
// SELECT query returns Table
var users = db.execute("SELECT id, name, email FROM users ORDER BY name") as Table;
for (row in users.rows) {
  var id = row[0] as int;
  var name = row[1] as String;
  var email = row[2] as String;
  print("User: ${id} - ${name} (${email})");
}

// INSERT returns null
db.execute("INSERT INTO logs (message, level) VALUES ('Started', 'INFO')");

// UPDATE returns null
db.execute("UPDATE users SET last_login = NOW() WHERE id = 42");

// DDL returns null
db.execute("CREATE TABLE IF NOT EXISTS sessions (id SERIAL PRIMARY KEY, token TEXT)");
```

### csv_to_table()

Imports data from a CSV file into a PostgreSQL table using the COPY command.

**Signature:** `fn csv_to_table(path: String, table_name: String, cols_name: Array<String>?, sep: char)`

**Parameters:**
- `path: String` - Absolute path to the CSV file
- `table_name: String` - Target table name
- `cols_name: Array<String>?` - Column names to import (null = all columns)
- `sep: char` - Column separator character (e.g., `','`, `'\t'`, `';'`)

**Requirements:**
- User must have COPY privileges in PostgreSQL
- Cannot be used inside a transaction (call `commit()` or `rollback()` first)
- CSV file must be readable by PostgreSQL server process
- Table must already exist with matching schema

**Example:**

```gcl
var db = Postgres {
  url: "localhost",
  port: "5432",
  db_name: "analytics",
  login: "etl_user",
  password: "etl_password"
};

// Ensure no active transaction
db.execute("CREATE TABLE IF NOT EXISTS sales (
  date DATE,
  product_id INT,
  quantity INT,
  revenue DECIMAL(10,2)
)");

// Import all columns from comma-separated file
db.csv_to_table("/data/sales_2024.csv", "sales", null, ',');

// Import only specific columns from tab-separated file
db.csv_to_table(
  "/data/products.tsv",
  "products",
  ["sku", "name", "price"],
  '\t'
);

// Import semicolon-separated European format
db.csv_to_table("/data/customers.csv", "customers", null, ';');
```

### query_to_csv()

Exports the results of a SQL query to a CSV file.

**Signature:** `fn query_to_csv(path: String, query: String, sep: char)`

**Parameters:**
- `path: String` - Output CSV file path
- `query: String` - SQL SELECT query to execute
- `sep: char` - Column separator character

**Behavior:**
- Executes the query and writes results to file
- Overwrites existing files
- Creates parent directories if they don't exist
- Efficient for large result sets

**Example:**

```gcl
// Export monthly sales report
db.query_to_csv(
  "/reports/sales_january.csv",
  "SELECT date, product, sum(revenue) FROM sales WHERE month = 1 GROUP BY date, product",
  ','
);

// Export user data with semicolon separator
db.query_to_csv(
  "/exports/active_users.csv",
  "SELECT id, email, created_at FROM users WHERE active = true",
  ';'
);

// Export complex join query
db.query_to_csv(
  "/tmp/order_details.csv",
  """
  SELECT o.id, o.date, u.email, p.name, oi.quantity
  FROM orders o
  JOIN users u ON o.user_id = u.id
  JOIN order_items oi ON o.id = oi.order_id
  JOIN products p ON oi.product_id = p.id
  WHERE o.date >= '2024-01-01'
  """,
  ','
);
```

## Common Use Cases

### ETL Pipeline

```gcl
var db = Postgres {
  url: "warehouse.example.com",
  port: "5432",
  db_name: "data_warehouse",
  login: "etl_service",
  password: "etl_secret"
};

// Extract: Export data from source
db.query_to_csv(
  "/tmp/raw_events.csv",
  "SELECT * FROM events WHERE processed = false",
  ','
);

// Transform: Process CSV (external tool or GreyCat logic)
// ...

// Load: Import transformed data
db.execute("CREATE TEMP TABLE staging_events (LIKE events)");
db.csv_to_table("/tmp/transformed_events.csv", "staging_events", null, ',');

// Merge into main table
db.begin();
db.execute("INSERT INTO events SELECT * FROM staging_events");
db.execute("UPDATE events SET processed = true WHERE id IN (SELECT id FROM staging_events)");
db.commit();
```

### Database Migration

```gcl
var sourceDb = Postgres {
  url: "old-server.example.com",
  port: "5432",
  db_name: "legacy_db",
  login: "admin",
  password: "admin_pwd"
};

var targetDb = Postgres {
  url: "new-server.example.com",
  port: "5432",
  db_name: "new_db",
  login: "admin",
  password: "admin_pwd"
};

// Export from source
sourceDb.query_to_csv("/tmp/users.csv", "SELECT * FROM users", ',');
sourceDb.query_to_csv("/tmp/orders.csv", "SELECT * FROM orders", ',');

// Create schema on target
targetDb.execute("CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT, email TEXT)");
targetDb.execute("CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INT, amount DECIMAL)");

// Import to target
targetDb.csv_to_table("/tmp/users.csv", "users", null, ',');
targetDb.csv_to_table("/tmp/orders.csv", "orders", null, ',');

print("Migration complete");
```

### Transactional Business Logic

```gcl
var db = Postgres {
  url: "localhost",
  port: "5432",
  db_name: "ecommerce",
  login: "app",
  password: "app_secret"
};

fn processOrder(userId: int, items: Array<OrderItem>) {
  db.begin();

  try {
    // Create order
    var orderResult = db.execute(
      "INSERT INTO orders (user_id, status, created_at) VALUES (${userId}, 'pending', NOW()) RETURNING id"
    ) as Table;
    var orderId = orderResult.rows[0][0] as int;

    // Add items and update inventory
    for (item in items) {
      // Add order item
      db.execute(
        "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (${orderId}, ${item.productId}, ${item.quantity}, ${item.price})"
      );

      // Decrease inventory
      db.execute(
        "UPDATE products SET stock = stock - ${item.quantity} WHERE id = ${item.productId}"
      );

      // Check if stock went negative
      var stock = db.execute("SELECT stock FROM products WHERE id = ${item.productId}") as Table;
      if (stock.rows[0][0] as int < 0) {
        throw "Insufficient stock for product ${item.productId}";
      }
    }

    // Update order status
    db.execute("UPDATE orders SET status = 'confirmed' WHERE id = ${orderId}");

    db.commit();
    return orderId;

  } catch (e) {
    db.rollback();
    print("Order failed: ${e}");
    throw e;
  }
}
```

### Reporting and Analytics

```gcl
var db = Postgres {
  url: "analytics.example.com",
  port: "5432",
  db_name: "reporting",
  login: "analyst",
  password: "analyst_pwd"
};

// Generate daily sales report
var sales = db.execute("""
  SELECT
    date_trunc('day', created_at) as day,
    count(*) as num_orders,
    sum(amount) as total_revenue,
    avg(amount) as avg_order_value
  FROM orders
  WHERE created_at >= NOW() - INTERVAL '30 days'
  GROUP BY day
  ORDER BY day
""") as Table;

// Export to CSV for spreadsheet analysis
db.query_to_csv(
  "/reports/daily_sales_last_30_days.csv",
  """
  SELECT
    date_trunc('day', created_at) as day,
    count(*) as num_orders,
    sum(amount) as total_revenue,
    avg(amount) as avg_order_value
  FROM orders
  WHERE created_at >= NOW() - INTERVAL '30 days'
  GROUP BY day
  ORDER BY day
  """,
  ','
);

print("Report generated");
```

## Best Practices

### Connection Management

- **Reuse connections**: Create one `Postgres` instance per database, reuse across queries
- **Close transactions**: Always pair `begin()` with `commit()` or `rollback()`
- **Use connection pooling**: In production, consider external connection poolers like PgBouncer

```gcl
// Good: Single connection instance
var db = createDbConnection();
for (i in 0..100) {
  db.execute("INSERT INTO events (data) VALUES ('event-${i}')");
}

// Bad: Creating connection per query
for (i in 0..100) {
  var db = createDbConnection(); // Wasteful!
  db.execute("INSERT INTO events (data) VALUES ('event-${i}')");
}
```

### Transaction Safety

- **Always use try-catch** with transactions
- **Keep transactions short**: Long-running transactions block other operations
- **Avoid CSV operations in transactions**: They are mutually exclusive

```gcl
db.begin();
try {
  // Do work
  db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1");
  db.commit();
} catch (e) {
  db.rollback();
  throw e;
}
```

### SQL Injection Prevention

- **Use parameterized queries** when possible (with external libraries)
- **Validate inputs** before embedding in SQL strings
- **Escape special characters** in user-provided data

```gcl
// Dangerous: SQL injection risk
var userInput = getUserInput();
db.execute("SELECT * FROM users WHERE name = '${userInput}'"); // BAD!

// Better: Validate input first
var userInput = getUserInput();
if (isValidUsername(userInput)) {
  db.execute("SELECT * FROM users WHERE name = '${userInput}'");
} else {
  throw "Invalid username";
}
```

### CSV Performance

- **Use CSV for bulk operations**: Much faster than individual INSERTs
- **Ensure correct permissions**: COPY requires superuser or specific grants
- **Validate CSV format**: Mismatched columns cause import failures

```gcl
// Slow: 10k individual inserts
for (i in 0..10000) {
  db.execute("INSERT INTO data (value) VALUES (${i})");
}

// Fast: Single CSV import
// (assuming data is already in CSV file)
db.csv_to_table("/data/bulk_data.csv", "data", null, ',');
```

### Error Handling

- **Check for null results**: Not all queries return data
- **Handle connection failures**: Network issues, credential problems
- **Log transaction failures**: For debugging and auditing

```gcl
try {
  var result = db.execute("SELECT * FROM users WHERE id = 999");

  if (result == null) {
    print("Query executed but returned no data");
  } else {
    var table = result as Table;
    if (table.rows.size() == 0) {
      print("No users found");
    } else {
      print("Found ${table.rows.size()} users");
    }
  }

} catch (e) {
  print("Database error: ${e}");
  // Log, alert, or retry
}
```

### Gotchas

- **Transactions and COPY are mutually exclusive**: Call `commit()` or `rollback()` before CSV operations
- **File paths for CSV**: Must be accessible by PostgreSQL server (not just GreyCat process)
- **Column count mismatch**: CSV columns must match table schema or specified `cols_name`
- **Type conversions**: Query results need explicit casting from `any` to specific types
- **Transaction isolation**: Default isolation level may cause unexpected behavior with concurrent transactions
- **Result table format**: SELECT results are returned as generic `Table` type, requiring knowledge of column order

### Security

- **Never commit credentials**: Load from environment variables or secure configuration
- **Use least privilege**: Grant only necessary database permissions
- **Enable SSL**: Use encrypted connections in production

```gcl
// Load credentials from environment
var db = Postgres {
  url: env("DB_HOST"),
  port: env("DB_PORT"),
  db_name: env("DB_NAME"),
  login: env("DB_USER"),
  password: env("DB_PASSWORD")
};
```

### Schema Management

- **Version your schema**: Use migration tools
- **Test DDL changes**: Create/alter tables outside transactions
- **Document table structures**: Keep schema documentation updated

```gcl
// Initialize schema if needed
db.execute("""
  CREATE TABLE IF NOT EXISTS schema_version (
    version INT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW()
  )
""");

var version = db.execute("SELECT MAX(version) FROM schema_version") as Table;
if (version.rows.size() == 0) {
  // Apply initial schema
  db.execute("CREATE TABLE users (...)");
  db.execute("CREATE TABLE orders (...)");
  db.execute("INSERT INTO schema_version (version) VALUES (1)");
}
```
