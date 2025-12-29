# GreyCat Language Skill

A comprehensive skill for building GreyCat backend applications with the graph-based language and built-in persistence.

## Overview

This skill provides expert guidance for developing with GreyCat - a unique language that combines graph-based programming with built-in persistence. It covers everything from basic syntax to advanced patterns for building production-ready applications.

## What is GreyCat?

GreyCat is a graph-based programming language with built-in persistence. Unlike traditional databases, GreyCat applications are **evolving coded structures** where your data model, business logic, and persistence are unified in a single codebase.

**Key Features:**
- Built-in graph persistence with node references
- Time-series and geo-spatial indexing
- Parallel execution with Job API
- Type-safe with minimal boilerplate
- React frontend integration via TypeScript SDK

## When to Use This Skill

Use this skill when you need help with:
- Building GreyCat applications (`.gcl` files)
- Working with persisted nodes and indexed collections (nodeList, nodeIndex, nodeTime, nodeGeo)
- Creating data models and services
- Writing API endpoints with `@permission` decorators
- Implementing parallel processing with Jobs
- Integrating React frontends with `@greycat/web` SDK
- Running GreyCat commands (serve, test, run import)
- Debugging GreyCat projects

## What's Included

### Core Documentation (SKILL.md)

Concise reference covering:
- Project setup and architecture
- Type system and nullability
- Nodes and persistence
- Indexed collections
- Services and API patterns
- Abstract types and inheritance
- Parallelization basics
- Testing and common pitfalls

### Detailed References

**Backend Development:**
- **nodes.md** - Deep dive into persistence, transactions, indexed collections
- **data_structures.md** - Tensor, Table, Buffer, Windows, Stack, Queue
- **concurrency.md** - Jobs, await, parallel writes, Tasks, PeriodicTask
- **io.md** - CSV/JSON reading/writing, File operations, HTTP client, SMTP
- **time.md** - Time handling, Date, duration, format specifiers
- **permissions.md** - RBAC, @permission, @role, SSO integration
- **testing.md** - @test, Assert, setup/teardown conventions

**Frontend Development:**
- **frontend.md** - Complete React integration guide:
  - @greycat/web SDK setup
  - TypeScript type generation
  - Authentication & authorization
  - React Query integration
  - Error handling and best practices

**Library References:**
- Complete GCL type definitions for all GreyCat libraries:
  - **std/** - Core types, I/O, runtime, utilities
  - **ai/** - LLM integration (llama.cpp)
  - **algebra/** - ML, neural networks, numerical computing
  - **kafka/** - Apache Kafka integration
  - **sql/** - PostgreSQL integration
  - **s3/** - Amazon S3 integration
  - **opcua/** - Industrial protocol integration
  - **finance/** - Financial modeling
  - **powerflow/** - Electrical power flow analysis

## Repository Structure

```
greycat-language.skill/
├── README.md              # This file
├── marketplace.json       # Skill marketplace metadata
├── package.sh            # Packaging script
└── greycat-language/     # Skill content
    ├── SKILL.md          # Main skill documentation
    └── references/       # Detailed reference files
        ├── frontend.md
        ├── nodes.md
        ├── time.md
        ├── concurrency.md
        ├── data_structures.md
        ├── io.md
        ├── permissions.md
        ├── testing.md
        └── [library GCL definitions]
```

## Building the Skill

To package the skill for distribution:

```bash
./package.sh              # Creates greycat.skill in current directory
./package.sh /path/to/dir # Creates greycat.skill in specified directory
```

The script:
- Validates SKILL.md frontmatter
- Packages all files from `greycat-language/` directory
- Creates a `.skill` file (zip archive with .skill extension)
- Excludes development files (node_modules, cache, etc.)

## Installation

### From Marketplace

```bash
# Coming soon - install via skillsmp.com
claude-code skills install greycat-language
```

### Manual Installation

```bash
git clone git@hub.datathings.com:greycat/skill/greycat-language.skill.git
cd greycat-language.skill
./package.sh
# Install greycat.skill file to your Claude Code skills directory
```

## Usage Examples

Once installed, Claude Code will automatically use this skill when you're working on GreyCat projects.

**Example interactions:**
- "Create a GreyCat data model for a city with buildings and residents"
- "Help me implement parallel processing for analyzing cities"
- "Set up React frontend integration with authentication"
- "How do I use nodeTime for time-series data?"
- "Debug this GreyCat function that's failing"

## Quick Start Example

```gcl
// project.gcl
@library("std", "7.5.125-dev");
@include("backend");

@permission("app.user", "app user permission");
@role("user", "app.user", "public", "api", "files");

fn main() { }
```

```gcl
// backend/src/model/city.gcl
var cities_by_name: nodeIndex<String, node<City>>;

type City {
    name: String;
    country: node<Country>;
    streets: nodeList<node<Street>>;
}
```

```gcl
// backend/src/api/city_api.gcl
@volatile
type CityView {
    name: String;
    streetCount: int;
}

@expose
@permission("app.user")
fn getCities(): Array<CityView> {
    var results = Array<CityView>{};
    for (name, city in cities_by_name) {
        results.add(CityView {
            name: city->name,
            streetCount: city->streets.size()
        });
    }
    return results;
}
```

## Contributing

Contributions are welcome! This skill is maintained by the GreyCat team at DataThings.

## Resources

- [Official GreyCat Documentation](https://doc.greycat.io/)
- [GreyCat Installation](https://get.greycat.io/)
- [Skills Marketplace](https://skillsmp.com/)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Open an issue in this repository
- Visit [GreyCat Documentation](https://doc.greycat.io/)
- Contact DataThings support

---

**Built for Claude Code** - Enhancing AI-assisted GreyCat development
