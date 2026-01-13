# Kafka Integration

Apache Kafka producer/consumer integration for event streaming in GreyCat.

## Overview

The Kafka library provides type-safe, high-performance messaging capabilities for GreyCat applications. Built on Apache Kafka's distributed streaming platform, it enables real-time data pipelines and event-driven architectures with automatic serialization and deserialization of GreyCat types.

Key features include:
- **Generic reader/writer types** supporting any GreyCat type `T`
- **Full Kafka configuration** with 172+ parameters for fine-grained control
- **Automatic serialization** of complex GreyCat types to Kafka messages
- **Topic/partition/offset tracking** for precise message positioning
- **Configurable timeouts** to handle network latency and blocking scenarios

This library is ideal for building microservices that communicate via event streams, implementing CQRS patterns, processing real-time data feeds, or integrating GreyCat with existing Kafka-based infrastructures.

## Installation

Add the Kafka library to your GreyCat project:

```gcl
@library("kafka", "7.6.14-dev")
```

## Quick Start

### Simple Consumer

```gcl
// Configure Kafka connection
var conf = KafkaConf {
  "bootstrap.servers": "localhost:9092",
  "group.id": "my-consumer-group",
  "auto.offset.reset": "earliest"
};

// Create a reader for String messages
var reader = KafkaReader<String> {
  conf: conf,
  topics: ["my-topic"],
  timeout: 10s
};

// Read messages
var msg = reader.read();
print("Received: ${msg.value} from ${msg.topic}:${msg.partition}@${msg.offset}");
```

### Simple Producer

```gcl
// Configure Kafka connection
var conf = KafkaConf {
  "bootstrap.servers": "localhost:9092"
};

// Create a writer for String messages
var writer = KafkaWriter<String> {
  conf: conf,
  topic: "my-topic"
};

// Write messages
writer.write("Hello, Kafka!");
writer.flush(); // Ensure delivery
```

## Types

### KafkaReader&lt;T&gt;

Generic consumer for reading messages from Kafka topics.

**Fields:**
- `conf: KafkaConf` (private) - Kafka client configuration
- `topics: Array<String>` (private) - List of topics to subscribe to
- `timeout: duration?` - Maximum time `read()` will block (default: `10s`)

**Methods:**
- `read(): KafkaMsg<T>` - Blocks until a message is available or timeout occurs

**Example:**

```gcl
type SensorReading {
  sensor_id: String;
  temperature: float;
  timestamp: time;
}

var reader = KafkaReader<SensorReading> {
  conf: KafkaConf {
    "bootstrap.servers": "kafka-cluster.example.com:9092",
    "group.id": "sensor-processors",
    "auto.offset.reset": "latest"
  },
  topics: ["sensor-readings", "sensor-alerts"],
  timeout: 30s
};

// Continuous consumption
while (true) {
  var msg = reader.read();
  print("Sensor ${msg.value.sensor_id}: ${msg.value.temperature}°C");
}
```

### KafkaMsg&lt;T&gt;

Container for messages read from Kafka with metadata.

**Fields:**
- `topic: String` - Source topic name
- `partition: int` - Partition number within the topic
- `offset: int` - Unique position within the partition
- `value: T` - The actual message payload

**Example:**

```gcl
var msg = reader.read();

// Access message metadata
print("Topic: ${msg.topic}");
print("Partition: ${msg.partition}");
print("Offset: ${msg.offset}");

// Process the payload
process_data(msg.value);

// Use metadata for logging/tracking
log_message(msg.topic, msg.partition, msg.offset, msg.value);
```

### KafkaWriter&lt;T&gt;

Generic producer for writing messages to a Kafka topic.

**Fields:**
- `conf: KafkaConf` (private) - Kafka client configuration
- `topic: String` (private) - Target topic name

**Methods:**
- `write(value: T)` - Asynchronously writes a message to the topic
- `flush()` - Blocks until all pending messages are delivered

**Example:**

```gcl
type OrderEvent {
  order_id: String;
  user_id: String;
  amount: float;
  status: String;
}

var writer = KafkaWriter<OrderEvent> {
  conf: KafkaConf {
    "bootstrap.servers": "kafka1:9092,kafka2:9092,kafka3:9092",
    "acks": "all",
    "compression.type": "snappy",
    "retries": "10"
  },
  topic: "order-events"
};

// Write an order event
writer.write(OrderEvent {
  order_id: "ORD-12345",
  user_id: "USR-789",
  amount: 99.99,
  status: "created"
});

// Ensure message delivery before shutdown
writer.flush();
```

### KafkaConf

Comprehensive Kafka client configuration with 172+ parameters.

**Required Fields:**
- `"bootstrap.servers": String` - Initial broker list (e.g., `"host1:9092,host2:9092"`)

**Common Optional Fields:**

| Field | Description | Example |
|-------|-------------|---------|
| `"group.id"` | Consumer group identifier | `"my-app-consumers"` |
| `"auto.offset.reset"` | Where to start consuming | `"earliest"` or `"latest"` |
| `"enable.auto.commit"` | Auto-commit offsets | `"true"` |
| `"acks"` | Acknowledgment level | `"all"`, `"1"`, `"0"` |
| `"compression.type"` | Message compression | `"snappy"`, `"gzip"`, `"lz4"` |
| `"max.poll.interval.ms"` | Consumer heartbeat timeout | `"300000"` (5 min) |
| `"batch.size"` | Producer batch size in bytes | `"16384"` |
| `"linger.ms"` | Producer batching delay | `"10"` |
| `"retries"` | Number of retries on failure | `"10"` |
| `"request.timeout.ms"` | Request timeout | `"30000"` (30s) |

**Security Configuration:**

```gcl
var secureConf = KafkaConf {
  "bootstrap.servers": "secure-kafka.example.com:9093",
  "security.protocol": "SASL_SSL",
  "sasl.mechanism": "PLAIN",
  "sasl.username": "kafka-user",
  "sasl.password": "secret-password",
  "ssl.ca.location": "/etc/ssl/certs/ca-cert.pem"
};
```

**Performance Tuning:**

```gcl
var perfConf = KafkaConf {
  "bootstrap.servers": "localhost:9092",
  // Producer performance
  "batch.size": "32768",
  "linger.ms": "20",
  "compression.type": "snappy",
  "buffer.memory": "67108864",
  // Consumer performance
  "fetch.min.bytes": "1024",
  "fetch.max.wait.ms": "500",
  "max.partition.fetch.bytes": "1048576"
};
```

**All Available Fields:**

The `KafkaConf` type supports all standard Kafka configuration parameters including:

**Connection:** `bootstrap.servers`, `client.id`, `broker.address.family`, `socket.timeout.ms`, `socket.connection.setup.timeout.ms`

**Consumer:** `group.id`, `auto.offset.reset`, `enable.auto.commit`, `auto.commit.interval.ms`, `max.poll.interval.ms`, `session.timeout.ms`, `heartbeat.interval.ms`, `partition.assignment.strategy`, `isolation.level`

**Producer:** `acks`, `retries`, `batch.size`, `linger.ms`, `buffer.memory`, `compression.type`, `max.in.flight.requests.per.connection`, `enable.idempotence`, `transactional.id`

**Security:** `security.protocol`, `sasl.mechanism`, `sasl.username`, `sasl.password`, `ssl.key.location`, `ssl.certificate.location`, `ssl.ca.location`

**Advanced:** `fetch.min.bytes`, `fetch.max.bytes`, `max.partition.fetch.bytes`, `request.timeout.ms`, `metadata.max.age.ms`, `reconnect.backoff.ms`

See the official [Apache Kafka documentation](https://kafka.apache.org/documentation/#configuration) for complete parameter details.

## Methods

### KafkaReader.read()

Reads a single message from subscribed topics.

**Signature:** `fn read(): KafkaMsg<T>`

**Behavior:**
- Blocks until a message is available or `timeout` is reached
- Throws an error if timeout expires
- Automatically deserializes to type `T`
- Commits offsets based on consumer configuration

**Example:**

```gcl
var reader = KafkaReader<String> {
  conf: KafkaConf {
    "bootstrap.servers": "localhost:9092",
    "group.id": "my-group"
  },
  topics: ["logs"],
  timeout: 5s
};

try {
  var msg = reader.read();
  print("Message: ${msg.value}");
} catch (e) {
  print("Timeout or error: ${e}");
}
```

### KafkaWriter.write()

Writes a message to the configured topic.

**Signature:** `fn write(value: T)`

**Behavior:**
- Asynchronously sends the message (non-blocking)
- Automatically serializes the value
- Messages may be batched for performance
- No guarantee of delivery until `flush()` is called

**Example:**

```gcl
var writer = KafkaWriter<int> {
  conf: KafkaConf { "bootstrap.servers": "localhost:9092" },
  topic: "metrics"
};

for (i in 0..100) {
  writer.write(i);
}
```

### KafkaWriter.flush()

Blocks until all buffered messages are delivered.

**Signature:** `fn flush()`

**Behavior:**
- Waits for acknowledgment of all pending messages
- Throws an error if any message fails to deliver
- Should be called before application shutdown

**Example:**

```gcl
var writer = KafkaWriter<String> {
  conf: KafkaConf {
    "bootstrap.servers": "localhost:9092",
    "linger.ms": "100" // Buffer for up to 100ms
  },
  topic: "events"
};

writer.write("event1");
writer.write("event2");
writer.write("event3");

// Ensure all 3 messages are sent
writer.flush();
print("All messages delivered");
```

## Common Use Cases

### Event Sourcing

```gcl
type UserEvent {
  user_id: String;
  event_type: String;
  payload: Map<String, any>;
  timestamp: time;
}

// Producer side
var eventWriter = KafkaWriter<UserEvent> {
  conf: KafkaConf {
    "bootstrap.servers": "kafka:9092",
    "acks": "all",
    "enable.idempotence": "true"
  },
  topic: "user-events"
};

fn recordUserAction(userId: String, action: String, data: Map<String, any>) {
  eventWriter.write(UserEvent {
    user_id: userId,
    event_type: action,
    payload: data,
    timestamp: time::now()
  });
}

// Consumer side - rebuild state from events
var eventReader = KafkaReader<UserEvent> {
  conf: KafkaConf {
    "bootstrap.servers": "kafka:9092",
    "group.id": "state-rebuilder",
    "auto.offset.reset": "earliest"
  },
  topics: ["user-events"],
  timeout: 60s
};

var userStates = Map<String, any>{};
while (true) {
  var msg = eventReader.read();
  applyEvent(userStates, msg.value);
}
```

### Multi-Topic Consumer

```gcl
type LogEntry {
  level: String;
  message: String;
  source: String;
  time: time;
}

var multiReader = KafkaReader<LogEntry> {
  conf: KafkaConf {
    "bootstrap.servers": "localhost:9092",
    "group.id": "log-aggregator"
  },
  topics: ["app-logs", "system-logs", "audit-logs"],
  timeout: 15s
};

while (true) {
  var msg = multiReader.read();

  // Process based on source topic
  if (msg.topic == "audit-logs") {
    storeToDatabase(msg.value);
  } else {
    writeToFile(msg.topic, msg.value);
  }
}
```

### High-Throughput Producer

```gcl
type Metric {
  name: String;
  value: float;
  tags: Map<String, String>;
  timestamp: time;
}

var metricsWriter = KafkaWriter<Metric> {
  conf: KafkaConf {
    "bootstrap.servers": "kafka1:9092,kafka2:9092",
    "compression.type": "lz4",
    "batch.size": "65536",
    "linger.ms": "50",
    "acks": "1",
    "max.in.flight.requests.per.connection": "5"
  },
  topic: "metrics"
};

// Write 10k metrics
for (i in 0..10000) {
  metricsWriter.write(Metric {
    name: "cpu.usage",
    value: random(0.0, 100.0),
    tags: Map<String, String>{ "host": "server-${i % 10}" },
    timestamp: time::now()
  });
}

// Ensure all sent before shutdown
metricsWriter.flush();
```

### Request-Response Pattern

```gcl
type Request {
  request_id: String;
  operation: String;
  params: Map<String, any>;
}

type Response {
  request_id: String;
  result: any;
  error: String?;
}

// Send request
var requestWriter = KafkaWriter<Request> {
  conf: KafkaConf { "bootstrap.servers": "localhost:9092" },
  topic: "requests"
};

requestWriter.write(Request {
  request_id: "req-123",
  operation: "calculate",
  params: Map<String, any>{ "x": 10, "y": 20 }
});

// Wait for response
var responseReader = KafkaReader<Response> {
  conf: KafkaConf {
    "bootstrap.servers": "localhost:9092",
    "group.id": "client-123"
  },
  topics: ["responses"],
  timeout: 30s
};

var resp = responseReader.read();
if (resp.value.request_id == "req-123") {
  print("Result: ${resp.value.result}");
}
```

## Best Practices

### Connection Management

- **Reuse readers/writers**: Create once per application, not per message
- **Set appropriate timeouts**: Match `timeout` to your latency requirements
- **Use connection pooling**: Share `KafkaConf` instances when possible

```gcl
// Good: Single reader instance
var reader = createReader();
for (i in 0..1000) {
  var msg = reader.read();
  process(msg);
}

// Bad: Creating reader per read
for (i in 0..1000) {
  var reader = createReader(); // Wasteful!
  var msg = reader.read();
  process(msg);
}
```

### Error Handling

- **Catch timeout exceptions** when reading
- **Flush before shutdown** to avoid data loss
- **Monitor consumer lag** in production

```gcl
var running = true;

while (running) {
  try {
    var msg = reader.read();
    process(msg);
  } catch (TimeoutException e) {
    // Expected during low traffic
    continue;
  } catch (KafkaException e) {
    print("Fatal error: ${e}");
    running = false;
  }
}
```

### Performance Optimization

- **Batch writes**: Use `linger.ms` to batch small messages
- **Compress data**: Enable `compression.type` for large payloads
- **Tune fetch sizes**: Adjust `fetch.min.bytes` and `fetch.max.wait.ms` for consumers

```gcl
// Optimized for throughput
var highThroughputConf = KafkaConf {
  "bootstrap.servers": "kafka:9092",
  "batch.size": "32768",
  "linger.ms": "20",
  "compression.type": "snappy",
  "acks": "1"
};

// Optimized for latency
var lowLatencyConf = KafkaConf {
  "bootstrap.servers": "kafka:9092",
  "linger.ms": "0",
  "acks": "1",
  "compression.type": "none"
};
```

### Consumer Groups

- **Use meaningful group IDs**: Makes monitoring easier
- **Scale consumers horizontally**: Add more consumers to the same group
- **Partition count**: Ensure partitions >= consumer count for parallelism

```gcl
// Multiple consumers in same group share load
var conf = KafkaConf {
  "bootstrap.servers": "kafka:9092",
  "group.id": "order-processors" // Same group = load balanced
};

// Consumer 1
var reader1 = KafkaReader<Order> { conf: conf, topics: ["orders"] };

// Consumer 2 (can run on different machine)
var reader2 = KafkaReader<Order> { conf: conf, topics: ["orders"] };
```

### Security

- **Use SSL/TLS**: Encrypt data in transit
- **Enable SASL**: Authenticate clients
- **Avoid hardcoding credentials**: Load from environment or config files

```gcl
// Load from environment
var conf = KafkaConf {
  "bootstrap.servers": env("KAFKA_BROKERS"),
  "security.protocol": "SASL_SSL",
  "sasl.mechanism": "SCRAM-SHA-512",
  "sasl.username": env("KAFKA_USER"),
  "sasl.password": env("KAFKA_PASSWORD")
};
```

### Gotchas

- **Type safety**: The generic `T` must be the same type used by producers
- **Serialization**: Complex nested types are supported but may impact performance
- **Offset management**: Auto-commit may lead to message loss on crashes; consider manual commits for critical data
- **Timeout errors**: Don't confuse timeout (no messages) with connection errors
- **Flush is blocking**: May take significant time with large batches
