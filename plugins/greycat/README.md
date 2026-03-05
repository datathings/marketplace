# GreyCat Skill

A comprehensive skill for building GreyCat backend applications with the graph-based language and built-in persistence.

## Overview

This skill provides expert guidance for developing with GreyCat - a unique language that combines graph-based programming with built-in persistence. It covers everything from basic syntax to advanced patterns for building production-ready applications.

## What is GreyCat?

GreyCat is a graph-based programming language with built-in persistence. Unlike traditional databases, GreyCat applications are **evolving coded structures** where your data model, business logic, and persistence are unified in a single codebase.

**Key Features:**
- **Unified persistence** - Data, schema, and logic in one evolving codebase (not a separate database)
- **Graph-native references** - Lightweight 64-bit node references instead of heavy object graphs
- **Built-in indexing** - nodeIndex (hash), nodeTime (time-series), nodeGeo (spatial), nodeList (ordered)
- **Parallel execution** - Job API with fork-join pattern and automatic transaction merging
- **Type-safe APIs** - @expose + @permission decorators with full TypeScript SDK generation
- **Zero-config persistence** - No ORM, no migrations, no serialization - just code
- **React integration** - Official @greycat/web SDK with automatic type generation

## Skill Activation Triggers

This skill automatically activates when you're working with:

### File Types & Extensions
- `.gcl` files (GreyCat source code)
- GreyCat project structures with `project.gcl`
- Files containing GreyCat syntax

### Language Features
- **Decorators**: `@expose`, `@permission`, `@volatile`, `@role`, `@library`, `@include`, `@test`, `@format_indent`
- **Indexed Collections**: `nodeList`, `nodeIndex`, `nodeTime`, `nodeGeo`
- **Node Operations**: Persisted nodes, transactions, node references (`node<Type>`)
- **Abstract Types**: Service patterns, polymorphism, inheritance
- **Parallel Processing**: Jobs, `await()`, `PeriodicTask`, fork-join patterns

### Framework Components
- **Backend**: Data models, services, API endpoints, persistence patterns
- **Frontend**: `@greycat/web` SDK, React integration, TypeScript type generation
- **Standard library**: core, io, runtime, util
- **Pro libraries**: ai, algebra, finance, kafka, opcua, powerflow, s3, sql, useragent

### CLI Commands
- `greycat serve` - Start development server
- `greycat test` - Run tests
- `greycat run import` - Execute import functions
- `greycat install` - Download library dependencies
- `greycat-lang lint` - Check for errors

### Use Cases
- Building graph-based data models
- Creating time-series or geo-spatial applications
- Implementing RBAC with permissions and roles
- Developing full-stack applications with React frontends
- Working with persistent node structures
- Debugging GreyCat applications

### When This Skill Does NOT Activate
- General graph databases (Neo4j, ArangoDB, JanusGraph)
- Generic React applications (without @greycat/web)
- SQL databases (PostgreSQL, MySQL, SQLite)
- Traditional ORM frameworks (Prisma, TypeORM, Sequelize)

## Why This Skill?

This skill transforms your AI assistant into a GreyCat development expert by providing:

- **Complete language reference** - All GCL syntax, types, and patterns in one place
- **Best practices** - Avoid common pitfalls with proven patterns (Services, Views, transactions)
- **Full-stack guidance** - Both backend (.gcl) and frontend (React/TypeScript) development
- **Library coverage** - Complete API references for all 10+ GreyCat libraries
- **Progressive disclosure** - Core patterns immediately available, detailed references loaded only when needed

**Perfect for:**
- GreyCat beginners learning the fundamentals
- Experienced developers building production applications
- Full-stack teams integrating React frontends
- Data engineers working with time-series and geo-spatial data

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

## GreyCat vs. Traditional Stacks

| Traditional Approach | GreyCat Approach |
|---------------------|------------------|
| Separate database (Postgres, MongoDB) | Built-in persistence in GCL |
| ORM layer (Prisma, TypeORM) | Direct node references |
| Manual indexing setup | nodeIndex, nodeTime, nodeGeo built-in |
| Separate job queue (Bull, BeeQueue) | Job API with await() |
| API framework (Express, Fastify) | @expose decorators |
| Manual TypeScript types | Auto-generated from GCL |

**When to use GreyCat:**
- Building graph-based applications with complex relationships
- Time-series or geo-spatial data with built-in indexing
- Need persistence without ORM complexity
- Want unified backend + frontend type safety

**When NOT to use GreyCat:**
- Standard CRUD apps (traditional frameworks may be simpler)
- Existing database migration requirements
- Team has no capacity to learn a new language/paradigm

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
plugins/greycat/
├── README.md              # This file
├── package.sh             # Packaging script
└── skills/greycat/        # Skill content
    ├── SKILL.md           # Main skill documentation
    └── references/        # Detailed reference files
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
- Packages all files from `greycat/` directory
- Creates a `.skill` file (zip archive with .skill extension)
- Excludes development files (node_modules, cache, etc.)

## Installation

### From Marketplace

```bash
# Install via skills marketplace
skills install greycat
```

### Manual Installation

```bash
git clone https://github.com/datathings/marketplace.git
cd marketplace/plugins/greycat
./package.sh
# Install the greycat.skill file to your AI assistant's skills directory
```

## Usage Examples

Once installed, this skill will automatically activate when you're working on GreyCat projects.

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

- [GreyCat Website](https://greycat.io/)
- [Official GreyCat Documentation](https://doc.greycat.io/)
- [GreyCat Installation](https://get.greycat.io/)
- [Datathings](https://datathings.com/)
- [Skills Marketplace](https://skillsmp.com/)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Open an issue in this repository
- Visit [GreyCat Documentation](https://doc.greycat.io/)
- Contact DataThings support

---

**Built for AI-assisted development** - Enhancing GreyCat productivity
