# GreyCat C - API & Standard Library Skill

A comprehensive AI skill providing complete reference for GreyCat C API and GCL Standard Library.

## Overview

This skill enables AI assistants to help with GreyCat development by providing detailed knowledge of:

- **C API** - Native function implementation, tensor operations, object manipulation, memory management
- **Standard Library (std)** - Runtime features, I/O operations, collections, statistics, and utilities

## What's Included

### 1. C API Reference
Complete C API documentation covering:
- Machine context (`gc_machine_t`) and function parameters
- Universal value containers (`gc_slot_t`) and type system
- Object field access and manipulation
- Tensor operations (multi-dimensional arrays)
- Buffer building and string operations
- Memory management (per-worker and global allocators)
- HTTP client, cryptography, and I/O operations

### 2. Standard Library Documentation
Comprehensive coverage of the GCL Standard Library organized into four modules:

- **std::core** - Fundamental types (Date, Tuple), geospatial types (GeoCircle, GeoPoly, GeoBox), enumerations
- **std::runtime** - Scheduler (task automation), logging, security, system operations, OpenID Connect, MCP
- **std::io** - File I/O, CSV/JSON/XML parsing, HTTP client, email, binary serialization
- **std::util** - Collections (Queue, Stack, SlidingWindow), statistics (Gaussian, Histogram), quantizers, crypto utilities

## When to Use This Skill

This skill activates when working on:

1. **Native C Functions** - Implementing custom native functions, working with tensors, managing memory
2. **Scheduled Tasks** - Setting up recurring tasks with the Scheduler
3. **File Operations** - Reading/writing CSV, JSON, XML, or binary files
4. **HTTP Integration** - Making REST API calls or handling HTTP requests
5. **Data Processing** - Statistical analysis, sliding windows, data quantization
6. **Security** - User management, access control, authentication
7. **System Operations** - Running processes, logging, system queries

## Skill Activation Triggers

This skill automatically activates based on context. The AI assistant will use this skill when it detects:

### Code & API Keywords
- **C API**: `gc_machine_t`, `gc_slot_t`, `gc_object_t`, `gc_type_t`, tensor operations, memory allocators
- **Core Types**: `Date`, `Time`, `Duration`, `Tuple`, `Error`, `GeoCircle`, `GeoPoly`, `GeoBox`
- **Runtime**: `Scheduler`, `Task`, `Job`, `Logger`, `User`, `SecurityPolicy`, `OpenIDConnect`
- **I/O**: `TextReader`, `TextWriter`, `CsvReader`, `CsvWriter`, `HttpClient`, `JsonReader`, `XmlParser`
- **Utilities**: `Queue`, `Stack`, `SlidingWindow`, `Gaussian`, `Histogram`, `Quantizer`, `Assert`

### Task Descriptions
- "Implement a native function" / "create a native C function"
- "Schedule a task" / "set up recurring jobs" / "daily/weekly automation"
- "Parse CSV" / "read JSON" / "make HTTP request" / "send email"
- "Calculate statistics" / "sliding window" / "histogram analysis"
- "Work with tensors" / "multi-dimensional arrays"
- "User authentication" / "access control" / "OpenID Connect"
- "Geospatial operations" / "GeoCircle" / "geographic boundaries"

### File Extensions & Contexts
- Working with `.gcl` files (GreyCat Language)
- Discussions of GreyCat runtime, SDK, or native development
- Questions about GCL standard library modules (`std::core`, `std::runtime`, `std::io`, `std::util`)

## Repository Structure

```
plugins/greycat-c/
├── README.md               # This file
├── package.sh              # Script to package the skill
└── skills/greycat-c/       # Skill files
    ├── SKILL.md            # Main skill entry point (concise overview)
    └── references/
        ├── api_reference.md        # Comprehensive C API reference
        └── standard_library.md     # Complete Standard Library documentation
```

## Installation

### Option 1: Clone and Package
```bash
git clone https://github.com/datathings/marketplace.git
cd marketplace/plugins/greycat-c
./package.sh
# Install the greycat-c.skill file to your skills directory
```

### Option 2: Install from Marketplace
Install directly via Claude Code:
```bash
/plugin install greycat-c@datathings
```

## Usage Examples

Once installed, the AI assistant will automatically use this skill when you ask questions or request help with GreyCat development:

```
User: "How do I create a 2D tensor in GreyCat C API?"
Assistant: *Uses the skill to provide accurate C API example*

User: "Help me set up a daily scheduled task in GCL"
Assistant: *Uses the skill to show Scheduler usage with DailyPeriodicity*

User: "How do I read a CSV file in GCL?"
Assistant: *Uses the skill to demonstrate CsvReader usage*
```

## Development

### Skill Structure

This skill follows AI skill best practices:

- **SKILL.md** - Lean entry point (~230 lines) with quick reference and links to detailed docs
- **Progressive Disclosure** - Detailed content in separate reference files, loaded only when needed
- **No Redundancy** - Information lives in one place (either SKILL.md or references)
- **Comprehensive Triggering** - Description includes all use cases to ensure proper triggering

### Making Changes

1. Edit files in `greycat-c/` directory
2. Update version in `marketplace.json` if publishing
3. Run `./package.sh` to create the distributable `.skill` file
4. Test the packaged skill before publishing

## License

This skill documentation is maintained as part of the GreyCat SDK project.

## Contributing

For issues or contributions, please use the repository issue tracker.

## Links

- **GreyCat Website**: https://greycat.io/
- **GreyCat Documentation**: https://doc.greycat.io/
- **GreyCat Installation**: https://get.greycat.io/
- **Datathings**: https://datathings.com/
- **Repository**: https://github.com/datathings/marketplace
- **Skill Marketplace**: https://skillsmp.com/
