# FCS (Flow Cytometry Standard)

Flow Cytometry Standard file reader for GreyCat.

## Overview

The FCS library provides native support for reading FCS (Flow Cytometry Standard) files, the universal data format used by flow cytometers and related instruments. It enables GreyCat applications to parse instrument metadata, channel configurations, and event data from `.fcs` files.

Key features include:
- **Streaming event reader** that reads one event at a time for memory-efficient processing
- **Full metadata extraction** including file version, timestamps, and channel definitions
- **Channel configuration access** with short/long names, gain, range, and scale parameters
- **FCS version support** for standard FCS file formats

This library is ideal for bioinformatics pipelines, clinical data analysis, quality control of flow cytometry experiments, and any application that needs to ingest or process `.fcs` instrument files.

## Installation

Add the FCS library to your GreyCat project:

```gcl
@library("fcs", "7.7.151-dev")
```

## Quick Start

### Read Metadata from an FCS File

```gcl
var reader = FcsReader { path: "/data/experiment/sample_001.fcs" };

var meta = reader.meta();
print("File: ${meta.file_name}");
print("Version: ${meta.version}");
print("Total events: ${meta.total_events}");
print("Total channels: ${meta.total_channels}");
print("Begin time: ${meta.begin_time}");
print("End time: ${meta.end_time}");

for (ch in meta.channels) {
  print("Channel: ${ch.short_name} (${ch.long_name}), range: ${ch.range}");
}
```

### Stream All Events

```gcl
var reader = FcsReader { path: "/data/experiment/sample_001.fcs" };

while (reader.can_read()) {
  var event = reader.event();
  // event is an Array<float?> with one value per channel
  print(event);
}
```

## Types

### FcsReader

Streaming reader for FCS files. Reads events one at a time to keep memory usage low regardless of file size.

**Fields:**
- `path: String` (private) - Absolute path to the `.fcs` file on disk

**Methods:**
- `meta(): FcsMeta` - Parse and return file metadata
- `event(): Array<float?>` - Read and return the next event
- `can_read(): bool` - Check whether more events are available

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };

// Inspect metadata first
var meta = reader.meta();
print("Reading ${meta.total_events} events across ${meta.total_channels} channels");

// Then stream events
while (reader.can_read()) {
  var event = reader.event();
  // Process each event...
}
```

### FcsMeta

Metadata extracted from the TEXT segment of an FCS file.

**Fields:**
- `file_name: String?` - Original file name recorded in the FCS header
- `version: String?` - FCS format version (e.g., `"FCS3.0"`, `"FCS3.1"`)
- `channels: Array<FcsChannel>?` - Channel definitions for each parameter
- `total_events: int?` - Total number of events (cells/particles) in the file
- `total_channels: int?` - Total number of measured parameters per event
- `begin_time: String?` - Acquisition start time as recorded by the instrument
- `end_time: String?` - Acquisition end time as recorded by the instrument

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

print("FCS version: ${meta.version}");
print("Events: ${meta.total_events}");
print("Channels: ${meta.total_channels}");
print("Acquisition window: ${meta.begin_time} to ${meta.end_time}");

if (meta.channels != null) {
  for (ch in meta.channels) {
    print("  ${ch.short_name}: ${ch.long_name}");
  }
}
```

### FcsChannel

Describes a single measurement parameter (channel) in the FCS file.

**Fields:**
- `short_name: String` - Abbreviated channel name (e.g., `"FSC-A"`, `"SSC-H"`, `"FITC-A"`)
- `long_name: String` - Full descriptive name (e.g., `"Forward Scatter-Area"`, `"CD3"`)
- `range: int?` - Maximum value the channel can report
- `gain: float?` - Amplifier gain applied during acquisition
- `scale: Tuple<float, float>?` - Log/linear scale parameters (decades, offset) if applicable

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

for (ch in meta.channels) {
  print("Channel: ${ch.short_name}");
  print("  Full name: ${ch.long_name}");
  print("  Range: ${ch.range}");
  print("  Gain: ${ch.gain}");
  print("  Scale: ${ch.scale}");
}
```

## Methods

### meta()

Parses the FCS file header and TEXT segment to extract metadata.

**Signature:** `fn meta(): FcsMeta`

**Parameters:** None

**Returns:** An `FcsMeta` instance containing file-level metadata and channel definitions.

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

// Use metadata to understand file structure before reading events
print("File: ${meta.file_name}");
print("Version: ${meta.version}");
print("Events: ${meta.total_events}, Channels: ${meta.total_channels}");
print("Acquired from ${meta.begin_time} to ${meta.end_time}");
```

### event()

Reads the next event from the DATA segment.

**Signature:** `fn event(): Array<float?>`

**Parameters:** None

**Returns:** An `Array<float?>` with one entry per channel, in the same order as `FcsMeta.channels`. Values may be `null` if the channel data is missing or unreadable.

**Behavior:**
- Each call advances the reader to the next event
- The array length equals `total_channels`
- Call `can_read()` before calling `event()` to avoid reading past the end of the file

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

// Read the first 10 events
var count = 0;
while (reader.can_read() && count < 10) {
  var event = reader.event();
  print("Event ${count}: ${event}");
  count++;
}
```

### can_read()

Checks whether there are more events available to read.

**Signature:** `fn can_read(): bool`

**Parameters:** None

**Returns:** `true` if the reader has more events to return, `false` when all events have been consumed.

**Behavior:**
- Returns `false` once all `total_events` have been read
- Safe to call multiple times without side effects

**Example:**

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };

var eventCount = 0;
while (reader.can_read()) {
  var event = reader.event();
  eventCount++;
}
print("Read ${eventCount} events");
```

## Common Use Cases

### Extract Channel Statistics

```gcl
var reader = FcsReader { path: "/data/experiment/sample.fcs" };
var meta = reader.meta();

// Initialize accumulators for each channel
var sums = Array<float>::new(meta.total_channels, 0.0);
var mins = Array<float>::new(meta.total_channels, float::max);
var maxs = Array<float>::new(meta.total_channels, float::min);
var count = 0;

while (reader.can_read()) {
  var event = reader.event();
  for (var i = 0; i < meta.total_channels; i++) {
    if (event[i] != null) {
      var val = event[i]!!;
      sums[i] = sums[i] + val;
      if (val < mins[i]) { mins[i] = val; }
      if (val > maxs[i]) { maxs[i] = val; }
    }
  }
  count++;
}

// Print statistics per channel
for (var i = 0; i < meta.total_channels; i++) {
  var ch = meta.channels!![i];
  var mean = sums[i] / count as float;
  print("${ch.short_name}: mean=${mean}, min=${mins[i]}, max=${maxs[i]}");
}
```

### Filter Events by Channel Threshold

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

// Find the index of the channel of interest
var targetIndex = -1;
for (var i = 0; i < meta.channels!!.size(); i++) {
  if (meta.channels!![i].short_name == "FITC-A") {
    targetIndex = i;
    break;
  }
}

if (targetIndex == -1) {
  print("Channel FITC-A not found");
} else {
  var threshold = 500.0;
  var positiveCount = 0;
  var totalCount = 0;

  while (reader.can_read()) {
    var event = reader.event();
    totalCount++;
    if (event[targetIndex] != null && event[targetIndex]!! > threshold) {
      positiveCount++;
    }
  }

  var pct = (positiveCount as float / totalCount as float) * 100.0;
  print("FITC-A positive (>${threshold}): ${positiveCount}/${totalCount} (${pct}%)");
}
```

### Batch Process Multiple FCS Files

```gcl
var files = [
  "/data/experiment/tube_1.fcs",
  "/data/experiment/tube_2.fcs",
  "/data/experiment/tube_3.fcs"
];

for (file in files) {
  var reader = FcsReader { path: file };
  var meta = reader.meta();

  print("--- ${meta.file_name} ---");
  print("  Version: ${meta.version}");
  print("  Events: ${meta.total_events}");
  print("  Channels: ${meta.total_channels}");
  print("  Acquired: ${meta.begin_time} to ${meta.end_time}");

  // Print channel list
  for (ch in meta.channels) {
    print("  ${ch.short_name} (${ch.long_name}), range: ${ch.range}, gain: ${ch.gain}");
  }
}
```

### Export Events to a Table

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

// Build header row from channel names
var header = "";
for (var i = 0; i < meta.channels!!.size(); i++) {
  if (i > 0) { header = header + "\t"; }
  header = header + meta.channels!![i].short_name;
}
print(header);

// Print each event as a tab-separated row
while (reader.can_read()) {
  var event = reader.event();
  var row = "";
  for (var i = 0; i < event.size(); i++) {
    if (i > 0) { row = row + "\t"; }
    if (event[i] != null) {
      row = row + "${event[i]!!}";
    } else {
      row = row + "NA";
    }
  }
  print(row);
}
```

## Best Practices

### Memory-Efficient Processing

- **Stream events one at a time**: The reader is designed for sequential access. Avoid loading all events into memory unless the file is small.
- **Check `total_events` first**: Use metadata to decide whether you can afford to buffer events in an array or need to process them in a streaming fashion.

```gcl
var reader = FcsReader { path: "/data/large_sample.fcs" };
var meta = reader.meta();

// For small files, buffering is fine
if (meta.total_events != null && meta.total_events!! < 100000) {
  var events = Array<Array<float?>>{};
  while (reader.can_read()) {
    events.add(reader.event());
  }
  // Random access to events...
}

// For large files, stream and accumulate only what you need
// (see the channel statistics example above)
```

### Channel Lookup

- **Resolve channel indices once**: Find the index of channels you care about before looping over events. Avoid searching by name inside the event loop.

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

// Build a lookup map once
var channelIndex = Map<String, int>{};
for (var i = 0; i < meta.channels!!.size(); i++) {
  channelIndex.set(meta.channels!![i].short_name, i);
}

// Use index inside the hot loop
var fscIdx = channelIndex.get("FSC-A")!!;
var sscIdx = channelIndex.get("SSC-A")!!;

while (reader.can_read()) {
  var event = reader.event();
  var fsc = event[fscIdx];
  var ssc = event[sscIdx];
  // Process forward/side scatter values...
}
```

### Metadata Validation

- **Validate before processing**: FCS metadata fields are nullable. Check for `null` before relying on values like `total_events` or `channels`.

```gcl
var reader = FcsReader { path: "/data/sample.fcs" };
var meta = reader.meta();

if (meta.channels == null || meta.total_events == null) {
  print("Invalid or unsupported FCS file: missing metadata");
} else {
  print("Ready to process ${meta.total_events} events across ${meta.total_channels} channels");
  // Proceed with processing...
}
```

### Error Handling

- **Wrap reader operations in try-catch**: File access, corrupt data, or unsupported FCS versions can cause errors.
- **Handle null event values**: Individual channel values within an event may be `null`.

```gcl
try {
  var reader = FcsReader { path: "/data/sample.fcs" };
  var meta = reader.meta();
  print("Loaded: ${meta.file_name}");

  while (reader.can_read()) {
    var event = reader.event();
    // Process event...
  }
} catch (e) {
  print("Failed to read FCS file: ${e}");
}
```

### Gotchas

- **`path` is private**: You set `path` at construction time but cannot read it back from the `FcsReader` instance.
- **Nullable metadata fields**: All fields on `FcsMeta` except `channels` entries' `short_name` and `long_name` are nullable. Always check for `null` before using values like `total_events`, `version`, or `file_name`.
- **Event values can be null**: Each element in the `Array<float?>` returned by `event()` may be `null`. Guard against null when performing arithmetic.
- **Sequential reading only**: The reader is forward-only. There is no way to seek back to a previous event. If you need random access, buffer events into an array.
- **Call `can_read()` before `event()`**: Calling `event()` when no more events are available will result in an error.
- **One reader per file**: Create a new `FcsReader` instance for each file you want to read. Readers are not reusable or resettable.
- **Channel order matters**: The values in the array returned by `event()` correspond to channels in the same order as `FcsMeta.channels`. Use index-based access to map values to their channel definitions.
- **Scale tuple interpretation**: The `scale` field on `FcsChannel` is a `Tuple<float, float>` representing log scale parameters (decades, offset). A `null` scale typically indicates linear scaling.
