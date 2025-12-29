# GreyCat Skill References

Quick navigation to detailed documentation loaded on-demand during development.

## PRIMARY RESOURCES

### Frontend Development (React + TypeScript)
**[frontend.md](frontend.md)** - **COMPREHENSIVE 1,013-line guide**
- @greycat/web SDK setup and authentication
- TypeScript type generation and integration
- React Query patterns and hooks
- Error handling and best practices
- **Start here for full-stack development**

### Backend Core Concepts
- **[nodes.md](nodes.md)** - Persistence, transactions, indexed collections
- **[concurrency.md](concurrency.md)** - Jobs, await, parallel writes, Tasks, PeriodicTask

## SPECIALIZED TOPICS

### Data & I/O
- **[data_structures.md](data_structures.md)** - Tensor, Table, Buffer, Windows, Stack, Queue
- **[io.md](io.md)** - CSV/JSON/File/HTTP/SMTP operations
- **[time.md](time.md)** - Time handling, Date, duration, formatting

### Security & Testing
- **[permissions.md](permissions.md)** - RBAC, @permission, @role, SSO integration
- **[testing.md](testing.md)** - @test, Assert, setup/teardown conventions

## LIBRARY API REFERENCES

Complete GCL type definitions for all GreyCat libraries:
**See [LIBRARIES.md](LIBRARIES.md) for full catalog**

Quick access by category:
- **Core:** std/ (required for all projects)
- **AI/ML:** ai/ (LLM via llama.cpp), algebra/ (neural networks, ML)
- **Integrations:** kafka/, sql/, s3/, opcua/, useragent/
- **Domain:** finance/, powerflow/

## Usage Pattern

1. **Start with SKILL.md** - Core language syntax and patterns
2. **Dive into frontend.md** - For React integration (most comprehensive guide)
3. **Reference specific topics** - As needed during development
4. **Check library APIs** - When using specific @library imports

---

**Progressive Disclosure:** These files are loaded by Claude only when needed for specific development tasks.
