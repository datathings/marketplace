# Graph store, backups and worlds

What lives in `gcdata/` and how to protect it. Reach here when the task is about what is persisted, type-shape evolution, backups and restores, or branching the store into worlds.

## Contents

- The graph store
- Backups
- Worlds (branches of the store)

## The graph store

`gcdata/` is the on-disk graph database. Created on first `serve`/`run`. Contents:

| Entry                       | Meaning                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- |
| `data_<N>.bin`              | Zone files (numbered; one per worker / shard).                                  |
| `index.bin`, `index.bin-lock` | Index of zones and live transactions.                                           |
| `program`                   | The compiled program. Updated on each compatible rebuild.                       |
| `abi`                       | ABI snapshot. Stored separately to track type-shape evolution across builds.    |
| `history/`                  | Task history (recent task records — for the `Task::history` API).               |
| `security/`                 | User / role / grant database (LMDB), the server's private key, the root `password` minted on first boot, and a root `token` (valid one year) re-minted on every `serve`. |
| `lock`                      | Process lock — prevents two workers from opening the same store simultaneously. |

**`gcdata/` is the durable state of the application.** Back it up; do not check it into git. Deleting it resets the project to a blank graph.

### Graph-persisted vs transient

A type is graph-persistent (its values can be saved into `gcdata/`) unless it is tagged `@volatile`. Stdlib runtime types like `Log`, `RuntimeInfo`, `HostMetric`, `TaskMetric`, `HttpMetric`, `Identity`, `Task` are `@volatile` — they describe live process state and cannot be stored.

User types holding `nodeTime<T>`, `nodeList<T>`, `nodeIndex<K, V>`, `nodeGeo<T>`, or `node<T>` attributes get persisted lazily as the program writes to those node tags. See [stdlib.md § Node tags](stdlib.md).

### ABI evolution

When you change a type's shape (rename, reorder, change attribute types) and rebuild, the compiler stores a new ABI version next to the existing one and migrates symbols where it can. Incompatible drift causes a load error on next startup — at that point, either fix the source or restore from backup.

## Backups

In-process API:

```gcl
Runtime::backup_full();       // full snapshot of gcdata/ into backup_path, returns the archive path
Runtime::backup_delta();      // incremental
Runtime::defrag();            // compact zone files
Runtime::store_stats();       // live blocks and bytes, by type and by world
```

CLI equivalents: `greycat backup` and `greycat defrag`.

`GREYCAT_BACKUP_PATH` (default `backup/`) sets the backup destination. `GREYCAT_MAX_BACKUP_FILES` (default `3`) caps retention.

Restore with `greycat restore <archive>`; `--verify` validates the archive before extracting.

A backup holds main and every world (see [Worlds](#worlds-branches-of-the-store)); a block a world shares with main is stored once. It must run in main: called from a task running in a world it raises. `greycat restore` replaces the worlds of the store with those of the archive, and an archive made before worlds existed restores none.

## Worlds (branches of the store)

A world is a named branch of the store: tasks running in it read and commit to the world's own block index instead of main's, so what they write stays out of main. Worlds and main share the zone files and every block they have in common, so a world costs only what it writes, plus its index. Merging a world back into main is not available.

```gcl
World::create("what-if", WorldMode::light);   // returns the world id; admin only
World::list();                                // Array<WorldInfo> {id, name, mode, created, user_id}; exposed, admin only
World::id("what-if");                         // int?, null when there is none
World::current();                             // String, the world the calling task or job runs in, "main" for main
World::remove("what-if");                     // admin only; fails while a task runs in it
```

Names take 1 to 128 of `[A-Za-z0-9_.-]`; `main` is reserved for the main store. Ids are never reused.

Two modes:

| Mode    | Starts with                  | Sees main's later commits                                                                                 |
| ------- | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `full`  | a copy of main's block index | Never: the world is the store as it was at creation.                                                      |
| `light` | an empty index, reading main | For every node the world has not modified. Once it modifies a node, that node stays as the world left it. |

A full world's creation copies main's index into its table, in time proportional to the store's block count (about 50 ms per million blocks), during which no commit runs. A light world is created instantly. In a light world, the unit of isolation is the node: a `node<T>` slot, an object, or a whole `nodeTime` / `nodeIndex` / `nodeList` / `nodeGeo`. A light world modifying a large `nodeIndex` main holds copies the index entries of its whole tree on that first commit (entries only, not the blocks); two tasks of the world making that first write to the same node at once conflict, and one fails with a fusion error. Two tasks of a world writing the same node later merge as they do in main.

Picking the world a task runs in:

| Where         | How                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------- |
| An HTTP call  | Request header `world: <name>`; an unknown name is answered `400`. `main` or no header: main. |
| `spawn`       | `Job.world`: a name, `"main"`, or `null` for the spawning task's world.                       |
| `await`       | Always the awaiting task's world; a job naming another one is an error.                       |
| The scheduler | `PeriodicOptions.world`; `null` is main. A run is skipped while its world does not exist.     |

**What a world changed.** `WorldChangesReader` lists the nodes the current world modified, read as the file readers
read:

```gcl
var r = WorldChangesReader {};
while (r.can_read()) {
    var n = r.read();   // node<T>, nodeTime, nodeIndex, nodeList or nodeGeo
}
```

Each node whose world version differs from main's current version, among the blocks the world wrote, comes once, in
node reference order. It lists what was committed when the task started, not the task's own pending writes. It skips
a `node<T>` the world set back to main's value, but lists a `nodeTime`, `nodeIndex`, `nodeList` or `nodeGeo` whose tree
the world rewrote, even when it holds main's values again. In a full world, a node main changed after the world was created also
shows when it shares a head block with a node the world modified. `available()` counts the nodes left; `pos` counts
those read, and a reader resumed after its task was suspended skips them. It fails in main. Building the list costs
nothing at commit: the reader reads the world's commits table and compares block addresses with main's index.

`Task.world`, in what `Task::running()` and a `task: true` call answer, names the world a task runs in (`"main"` for main). It is `null` in `Task::history()`, which does not record it.

`Runtime::store_stats().worlds` reports each world's index entries, the blocks it holds that main does not and their bytes, and its index size. A world's index is a table of `gcdata/index.bin`, next to main's; `gcdata/worlds/registry` lists the worlds. Never edit or copy them by hand. Defrag moves the blocks of every world along with main's.
