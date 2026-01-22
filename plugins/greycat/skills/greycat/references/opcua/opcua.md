# OPC UA Integration

Industrial automation integration with OPC UA servers for GreyCat.

## Overview

The OPC UA (Open Platform Communications Unified Architecture) library enables GreyCat applications to communicate with industrial control systems, SCADA systems, PLCs, and IoT devices. OPC UA is the industry-standard protocol for industrial automation and data exchange.

Key features include:
- **Read/write operations** for single and multiple nodes
- **Historical data access** with time-range queries
- **Real-time subscriptions** for live data monitoring
- **Event monitoring** for alarms and notifications
- **Metadata browsing** to discover server capabilities
- **Security support** with authentication, encryption, and certificates
- **Server exploration** to traverse node hierarchies

This library is ideal for Industry 4.0 applications, SCADA integration, IoT data collection, predictive maintenance, and building automation systems.

## Installation

Add the OPC UA library to your GreyCat project:

```gcl
@library("opcua", "7.6.37-dev")
```

## Quick Start

### Basic Read Operation

```gcl
var client = OpcuaClient {
  url: "opc.tcp://localhost:4840",
  security_mode: OpcuaSecurityMode::None,
  timeout: 30s
};

// Read a single value
var temperature = client.read("ns=2;s=Temperature");
print("Temperature: ${temperature}");

// Read multiple values at once
var values = client.read_all([
  "ns=2;s=Temperature",
  "ns=2;s=Pressure",
  "ns=2;s=FlowRate"
]);
```

### Subscribe to Live Data

```gcl
var client = OpcuaClient {
  url: "opc.tcp://192.168.1.100:4840",
  security_mode: OpcuaSecurityMode::None
};

// Subscribe to node changes
var subscriptionId = client.subscribe(
  ["ns=2;s=ProductionCounter"],
  fn(data) {
    print("New value: ${data}");
  },
  fn(error) {
    print("Error: ${error}");
  }
);

// Later: cancel subscription
client.cancel_subscription(subscriptionId);
```

## Types

### OpcuaClient

Main client for connecting to OPC UA servers.

**Fields:**
- `url: String` (private) - Server endpoint URL (e.g., `"opc.tcp://localhost:4840"`)
- `security_mode: OpcuaSecurityMode` (private) - Security level
- `security_policy: OpcuaSecurityPolicy?` (private) - Encryption policy
- `credentials: OpcuaCredentials?` (private) - Username/password authentication
- `certificate: OpcuaCertificate?` (private) - Certificate-based authentication
- `timeout: duration?` (private) - Operation timeout (default varies by operation)

**Read Methods:**
- `read(nodeId: String): any?` - Read single node value
- `read_all(nodeIds: Array<String>): Array<any?>` - Read multiple nodes
- `read_with_time(nodeId: String): OpcuaValueDetails?` - Read with timestamps
- `read_all_with_time(nodeIds: Array<String>): Array<OpcuaValueDetails?>` - Read multiple with timestamps
- `read_history(nodeId: String, from: time?, to: time?): Array<OpcuaValueDetails>?` - Read historical data

**Write Methods:**
- `write(nodeId: String, value: any)` - Write value to node

**Browse Methods:**
- `read_metas(nodesIds: Array<String>): Array<OpcuaMeta?>` - Read node metadata
- `get_children(nodeId: String): Array<OpcuaMeta>` - Get child nodes
- `explore(rootId: String): Array<Array<OpcuaMeta>>` - Explore node tree

**Subscription Methods:**
- `subscribe(nodeIds: Array<String>, callback_data: function, callback_error: function): int` - Subscribe to data changes
- `read_events(nodeIds: Array<String>, callback_data: function, callback_error: function): int` - Subscribe to events
- `cancel_subscription(subscription_id: int): bool` - Cancel subscription

**Utility Methods:**
- `call(nodeId: String, parameters: Array<any?>): any?` - Call server method
- `is_connected(): bool` - Check connection status
- `get_address_spaces(): Array` - Get server address spaces
- `get_server_time(): time` - Get server timestamp
- `find_endpoints(url: String, security_mode?, security_policy?): Array<OpcuaEndpointDescription>` (static) - Discover server endpoints

**Example:**

```gcl
// Basic connection
var client = OpcuaClient {
  url: "opc.tcp://plc.factory.com:4840",
  security_mode: OpcuaSecurityMode::None
};

// Secure connection with credentials
var secureClient = OpcuaClient {
  url: "opc.tcp://secure-server.com:4840",
  security_mode: OpcuaSecurityMode::SignAndEncrypt,
  security_policy: OpcuaSecurityPolicy::Basic256Sha256,
  credentials: OpcuaCredentials {
    login: "operator",
    password: "secret123"
  },
  timeout: 60s
};
```

### OpcuaValueDetails

Detailed value information including timestamps and status.

**Fields:**
- `value: any` - The actual value
- `source_time: time?` - When the value was generated at the source
- `server_time: time?` - When the server processed the value
- `status_code: int?` - OPC UA status code (0 = good)

**Example:**

```gcl
var details = client.read_with_time("ns=2;s=Sensor1");
print("Value: ${details.value}");
print("Source time: ${details.source_time}");
print("Server time: ${details.server_time}");
print("Status: ${details.status_code}");
```

### OpcuaMeta

Node metadata including properties and attributes.

**Fields:**
- `node_id: String` - Node identifier
- `node_class: int?` - Node class (1=Object, 2=Variable, 4=Method, etc.)
- `browse_name: Tuple?` - Node's browse name
- `display_name: Tuple?` - Human-readable name
- `description: Tuple?` - Node description
- `data_type: String?` - Data type for variable nodes
- `value: any?` - Current value
- `access_level: int?` - Read/write permissions
- `historizing: bool?` - Whether history is recorded

**Example:**

```gcl
var metas = client.read_metas(["ns=2;s=Temperature"]);
var meta = metas[0];

print("Node ID: ${meta.node_id}");
print("Display Name: ${meta.display_name}");
print("Data Type: ${meta.data_type}");
print("Value: ${meta.value}");
```

### OpcuaEvent

Event notification structure for alarms and conditions.

**Fields:**
- `"EventId": String` - Unique event identifier
- `"EventType": String?` - Type of event
- `"Message": Tuple?` - Event message
- `"SourceNode": String` - Node that generated the event
- `"SourceName": String` - Source name
- `"Time": time` - Event timestamp
- `"ReceiveTime": time?` - When event was received
- `"Severity": int` - Severity level (1-1000)
- `"ConditionName": String?` - Condition name for alarms
- `"ActiveState": Tuple?` - Whether condition is active
- `"Retain": bool?` - Whether event should be retained

**Example:**

```gcl
client.read_events(
  ["ns=2;i=5001"], // Server object for all events
  fn(event: OpcuaEvent) {
    print("Event: ${event.'Message'}");
    print("Severity: ${event.'Severity'}");
    print("Source: ${event.'SourceName'}");
  },
  fn(error) {
    print("Event error: ${error}");
  }
);
```

### OpcuaCredentials

Username/password authentication.

**Fields:**
- `login: String` - Username
- `password: String` - Password

**Example:**

```gcl
var creds = OpcuaCredentials {
  login: "admin",
  password: env("OPCUA_PASSWORD")
};

var client = OpcuaClient {
  url: "opc.tcp://server:4840",
  security_mode: OpcuaSecurityMode::Sign,
  credentials: creds
};
```

### OpcuaCertificate

Certificate-based authentication configuration.

**Fields:**
- `path: String?` - Path to client certificate file
- `private_key_path: String?` - Path to private key file
- `application_uri: String?` - Application URI
- `allow_self_signed: bool?` - Accept self-signed certificates

**Example:**

```gcl
var cert = OpcuaCertificate {
  path: "/etc/opcua/client-cert.pem",
  private_key_path: "/etc/opcua/client-key.pem",
  application_uri: "urn:myapp:client",
  allow_self_signed: false
};

var client = OpcuaClient {
  url: "opc.tcp://server:4840",
  security_mode: OpcuaSecurityMode::SignAndEncrypt,
  security_policy: OpcuaSecurityPolicy::Aes256_Sha256_RsaPss,
  certificate: cert
};
```

### OpcuaSecurityMode (Enum)

Security mode levels.

**Values:**
- `None` - No security (plain text)
- `Sign` - Message signing (integrity)
- `SignAndEncrypt` - Signing + encryption (confidentiality + integrity)
- `Invalid` - Invalid mode

**Example:**

```gcl
// Development: No security
var devClient = OpcuaClient {
  url: "opc.tcp://localhost:4840",
  security_mode: OpcuaSecurityMode::None
};

// Production: Full security
var prodClient = OpcuaClient {
  url: "opc.tcp://production-server:4840",
  security_mode: OpcuaSecurityMode::SignAndEncrypt
};
```

### OpcuaSecurityPolicy (Enum)

Encryption and signing algorithms.

**Values:**
- `None` - No encryption
- `Basic128Rsa15` - 128-bit encryption (deprecated)
- `Basic256` - 256-bit encryption (deprecated)
- `Basic256Sha256` - 256-bit with SHA-256
- `Aes128_Sha256_RsaOaep` - AES-128 with SHA-256
- `Aes256_Sha256_RsaPss` - AES-256 with SHA-256 (strongest)

**Example:**

```gcl
var client = OpcuaClient {
  url: "opc.tcp://server:4840",
  security_mode: OpcuaSecurityMode::SignAndEncrypt,
  security_policy: OpcuaSecurityPolicy::Aes256_Sha256_RsaPss // Strongest
};
```

### OpcuaEndpointDescription

Server endpoint information from discovery.

**Fields:**
- `url: String` - Endpoint URL
- `discovery_urls: Array` - Discovery URLs
- `certificate: String?` - Server certificate
- `security_mode: OpcuaSecurityMode` - Security mode
- `security_policy: OpcuaSecurityPolicy` - Security policy
- `user_identity_token_ids: Array` - Supported authentication methods
- `transport_profile_uri: String` - Transport protocol
- `security_level: int` - Security level

**Example:**

```gcl
var endpoints = OpcuaClient::find_endpoints("opc.tcp://server:4840", null, null);

for (endpoint in endpoints) {
  print("URL: ${endpoint.url}");
  print("Security Mode: ${endpoint.security_mode}");
  print("Security Policy: ${endpoint.security_policy}");
  print("Security Level: ${endpoint.security_level}");
}
```

## Methods

### read()

Reads the current value of a single node.

**Signature:** `fn read(nodeId: String): any?`

**Returns:** The node's value (type depends on node), or `null` if not readable

**Example:**

```gcl
var temp = client.read("ns=2;s=Temperature") as float;
var running = client.read("ns=2;s=MotorRunning") as bool;
var name = client.read("ns=2;s=DeviceName") as String;
```

### read_all()

Reads multiple nodes in a single request.

**Signature:** `fn read_all(nodeIds: Array<String>): Array<any?>`

**Returns:** Array of values in same order as input node IDs

**Example:**

```gcl
var nodeIds = [
  "ns=2;s=Temperature",
  "ns=2;s=Pressure",
  "ns=2;s=FlowRate"
];

var values = client.read_all(nodeIds);
var temp = values[0] as float;
var pressure = values[1] as float;
var flow = values[2] as float;

print("Temp: ${temp}, Pressure: ${pressure}, Flow: ${flow}");
```

### write()

Writes a value to a node.

**Signature:** `fn write(nodeId: String, value: any)`

**Example:**

```gcl
// Write different types
client.write("ns=2;s=Setpoint", 75.5);
client.write("ns=2;s=Enable", true);
client.write("ns=2;s=Mode", "AUTO");

// Control a valve
client.write("ns=2;s=ValvePosition", 50); // 50% open
```

### read_with_time()

Reads a node value with timestamp information.

**Signature:** `fn read_with_time(nodeId: String): OpcuaValueDetails?`

**Example:**

```gcl
var details = client.read_with_time("ns=2;s=Sensor");

print("Value: ${details.value}");
print("Source timestamp: ${details.source_time}");
print("Server timestamp: ${details.server_time}");
print("Quality: ${details.status_code}"); // 0 = good
```

### read_history()

Reads historical data for a node within a time range.

**Signature:** `fn read_history(nodeId: String, from: time?, to: time?): Array<OpcuaValueDetails>?`

**Parameters:**
- `nodeId: String` - Node to query
- `from: time?` - Start time (null = beginning)
- `to: time?` - End time (null = now)

**Example:**

```gcl
// Last hour of data
var history = client.read_history(
  "ns=2;s=ProductionRate",
  time::now() - 1h,
  time::now()
);

for (entry in history) {
  print("${entry.source_time}: ${entry.value}");
}

// All historical data
var allHistory = client.read_history("ns=2;s=TotalCount", null, null);
```

### subscribe()

Subscribes to node value changes for real-time monitoring.

**Signature:** `fn subscribe(nodeIds: Array<String>, callback_data: function, callback_error: function): int`

**Returns:** Subscription ID for later cancellation

**Example:**

```gcl
var subId = client.subscribe(
  ["ns=2;s=Temperature", "ns=2;s=Pressure"],
  fn(data) {
    print("Data update: ${data}");
  },
  fn(error) {
    print("Subscription error: ${error}");
  }
);

// Keep running to receive updates
while (client.is_connected()) {
  // Your application logic
  sleep(1s);
}

// Cleanup
client.cancel_subscription(subId);
```

### read_events()

Subscribes to OPC UA events (alarms, notifications).

**Signature:** `fn read_events(nodeIds: Array<String>, callback_data: function, callback_error: function): int`

**Example:**

```gcl
var eventSubId = client.read_events(
  ["ns=2;i=5001"], // Server object
  fn(event: OpcuaEvent) {
    if (event.'Severity' > 500) {
      print("HIGH SEVERITY EVENT!");
      print("Message: ${event.'Message'}");
      print("Source: ${event.'SourceName'}");
    }
  },
  fn(error) {
    print("Event error: ${error}");
  }
);
```

### get_children()

Gets all child nodes of a given node.

**Signature:** `fn get_children(nodeId: String): Array<OpcuaMeta>`

**Example:**

```gcl
// Discover structure
var children = client.get_children("ns=2;s=Building");

for (child in children) {
  print("Child: ${child.display_name} (${child.node_id})");
}
```

### explore()

Recursively explores the node tree from a root node.

**Signature:** `fn explore(rootId: String): Array<Array<OpcuaMeta>>`

**Returns:** Array of paths, where each path is an array of metadata from root to leaf

**Example:**

```gcl
// Explore entire device tree
var paths = client.explore("ns=2;s=Factory");

for (path in paths) {
  var pathStr = "";
  for (node in path) {
    pathStr = pathStr + " / " + node.display_name;
  }
  print(pathStr);
}
```

### call()

Calls a method on the OPC UA server.

**Signature:** `fn call(nodeId: String, parameters: Array<any?>): any?`

**Example:**

```gcl
// Call a method with parameters
var result = client.call(
  "ns=2;s=Calculator.Add",
  [10, 20]
);
print("10 + 20 = ${result}");

// Start a process
client.call("ns=2;s=Process.Start", []);
```

## Common Use Cases

### Real-Time Monitoring Dashboard

```gcl
var client = OpcuaClient {
  url: "opc.tcp://factory-plc:4840",
  security_mode: OpcuaSecurityMode::None
};

// Subscribe to critical parameters
var sensors = [
  "ns=2;s=Line1.Temperature",
  "ns=2;s=Line1.Pressure",
  "ns=2;s=Line1.Speed",
  "ns=2;s=Line1.Status"
];

var subId = client.subscribe(
  sensors,
  fn(data) {
    // Update dashboard
    updateDashboard(data);

    // Check thresholds
    if (data.temperature > 100.0) {
      sendAlert("Temperature too high: ${data.temperature}");
    }
  },
  fn(error) {
    print("Monitoring error: ${error}");
    reconnect();
  }
);
```

### Historical Data Analysis

```gcl
var client = OpcuaClient {
  url: "opc.tcp://scada-server:4840",
  security_mode: OpcuaSecurityMode::None
};

// Analyze production efficiency over last 24 hours
var history = client.read_history(
  "ns=2;s=ProductionLine.Output",
  time::now() - 24h,
  time::now()
);

var total = 0.0;
var count = 0;

for (entry in history) {
  total = total + (entry.value as float);
  count = count + 1;
}

var average = total / count;
print("24h Average Output: ${average}");
```

### Multi-Server Data Collection

```gcl
type PlcConnection {
  name: String;
  client: OpcuaClient;
  nodes: Array<String>;
}

var plcs = [
  PlcConnection {
    name: "Line 1",
    client: OpcuaClient {
      url: "opc.tcp://plc1:4840",
      security_mode: OpcuaSecurityMode::None
    },
    nodes: ["ns=2;s=Speed", "ns=2;s=Count"]
  },
  PlcConnection {
    name: "Line 2",
    client: OpcuaClient {
      url: "opc.tcp://plc2:4840",
      security_mode: OpcuaSecurityMode::None
    },
    nodes: ["ns=2;s=Speed", "ns=2;s=Count"]
  }
];

// Poll all PLCs
for (plc in plcs) {
  var values = plc.client.read_all(plc.nodes);
  print("${plc.name}: Speed=${values[0]}, Count=${values[1]}");
}
```

### Automated Control Logic

```gcl
var client = OpcuaClient {
  url: "opc.tcp://control-system:4840",
  security_mode: OpcuaSecurityMode::None
};

// Automated temperature control
while (true) {
  var temp = client.read("ns=2;s=Reactor.Temperature") as float;
  var setpoint = client.read("ns=2;s=Reactor.Setpoint") as float;

  if (temp > setpoint + 5.0) {
    // Too hot, increase cooling
    client.write("ns=2;s=CoolingValve", 100);
  } else if (temp < setpoint - 5.0) {
    // Too cold, reduce cooling
    client.write("ns=2;s=CoolingValve", 0);
  } else {
    // Within range, proportional control
    var valve = ((setpoint - temp) / 10.0) * 100;
    client.write("ns=2;s=CoolingValve", valve);
  }

  sleep(5s);
}
```

## Best Practices

### Connection Management

- **Check connection status**: Use `is_connected()` before operations
- **Handle disconnections**: Implement reconnection logic
- **Reuse clients**: Create one client per server, not per operation

```gcl
var client = createOpcuaClient();

fn readSafely(nodeId: String): any? {
  if (!client.is_connected()) {
    print("Not connected, attempting reconnect...");
    client = createOpcuaClient();
  }

  try {
    return client.read(nodeId);
  } catch (e) {
    print("Read failed: ${e}");
    return null;
  }
}
```

### Security

- **Use encryption in production**: Never use `None` security mode over public networks
- **Validate certificates**: Set `allow_self_signed: false` in production
- **Store credentials securely**: Use environment variables or secret management

```gcl
// Production-grade security
var prodClient = OpcuaClient {
  url: "opc.tcp://production.server:4840",
  security_mode: OpcuaSecurityMode::SignAndEncrypt,
  security_policy: OpcuaSecurityPolicy::Aes256_Sha256_RsaPss,
  credentials: OpcuaCredentials {
    login: env("OPCUA_USER"),
    password: env("OPCUA_PASSWORD")
  },
  certificate: OpcuaCertificate {
    path: "/etc/opcua/client-cert.pem",
    private_key_path: "/etc/opcua/client-key.pem",
    allow_self_signed: false
  }
};
```

### Node ID Format

- **Namespace index**: `ns=<number>` (e.g., `ns=2`)
- **String identifier**: `s=<string>` (e.g., `ns=2;s=Temperature`)
- **Numeric identifier**: `i=<number>` (e.g., `ns=0;i=2258`)
- **GUID identifier**: `g=<guid>` (e.g., `ns=2;g=12345678-...`)

```gcl
// Different node ID formats
var temp1 = client.read("ns=2;s=Sensor.Temperature"); // String
var temp2 = client.read("ns=2;i=1001"); // Numeric
var serverTime = client.read("ns=0;i=2258"); // Standard node
```

### Performance

- **Batch reads**: Use `read_all()` instead of multiple `read()` calls
- **Subscription over polling**: More efficient for real-time data
- **Limit history queries**: Use time ranges to avoid large result sets

```gcl
// Good: Single batch read
var values = client.read_all([
  "ns=2;s=Sensor1",
  "ns=2;s=Sensor2",
  "ns=2;s=Sensor3"
]);

// Bad: Multiple individual reads
var val1 = client.read("ns=2;s=Sensor1");
var val2 = client.read("ns=2;s=Sensor2");
var val3 = client.read("ns=2;s=Sensor3");
```

### Error Handling

- **Type casting**: Read values are `any`, cast to expected type
- **Null checks**: `read()` can return `null`
- **Subscription errors**: Always provide error callback

```gcl
try {
  var temp = client.read("ns=2;s=Temperature");

  if (temp == null) {
    print("Temperature not available");
  } else {
    var tempValue = temp as float;
    print("Temperature: ${tempValue}°C");
  }
} catch (e) {
  print("Read error: ${e}");
}
```

### Gotchas

- **Subscription callbacks run asynchronously**: Don't block in callbacks
- **Node IDs are server-specific**: Vary between vendors and configurations
- **Historical data requires server support**: Not all servers provide history
- **Write permissions**: Check `access_level` before attempting writes
- **Security policies must match**: Client and server must agree on policy
- **Timeout configuration**: Set appropriate timeout for slow/remote servers
- **Cleanup subscriptions**: Always cancel when done to free server resources

### Discovery

Use `find_endpoints()` to discover server capabilities:

```gcl
var endpoints = OpcuaClient::find_endpoints(
  "opc.tcp://unknown-server:4840",
  null,
  null
);

print("Server supports ${endpoints.size()} endpoints:");
for (endpoint in endpoints) {
  print("  ${endpoint.security_mode} / ${endpoint.security_policy}");
}

// Connect with best security available
var bestEndpoint = endpoints[0]; // Often sorted by security level
var client = OpcuaClient {
  url: bestEndpoint.url,
  security_mode: bestEndpoint.security_mode,
  security_policy: bestEndpoint.security_policy
};
```
