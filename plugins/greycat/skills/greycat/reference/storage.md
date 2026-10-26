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
| `jobs/<task id>/`           | What a task keeps for the server to read back while it runs: its arguments, the transaction logs and snapshot of an `await`, and the arguments, results and transaction logs of its jobs under `<job index>/`. Never served by `/files`, never synced to disk, and emptied when the process starts, since nothing reads it after a restart. |

**`gcdata/` is the durable state of the application.** Back it up; do not check it into git. Deleting it resets the project to a blank graph.

### Graph-persisted vs transient

A type is graph-persistent (its values can be saved into `gcdata/`) unless it is tagged `@volatile`. Stdlib runtime types like `Log`, `RuntimeInfo`, `HostMetric`, `TaskMetric`, `HttpMetric`, `Identity`, `Task` are `@volatile` — they describe live process state and cannot be stored.

User types holding `nodeTime<T>`, `nodeList<T>`, `nodeIndex<K, V>`, `nodeGeo<T>`, or `node<T>` attributes get persisted lazily as the program writes to those node tags. See [stdlib.md § Node tags](stdlib.md).

### Listing every node

`NodeReader` lists every node the store holds, read as the file readers read. With `type` set it keeps the nodes of
one node type, or of every specialization of a node kind (`nodeTime` keeps every `nodeTime<T>`):

```gcl
var r = NodeReader { type: nodeTime<float> };
while (r.can_read()) {
    var series = r.read() as nodeTime<float>;
    println("${series.size()} points");
}
```

It walks the block index in node reference order and lists every node, reachable from a module variable or not, as a
`node<T>`, `nodeTime`, `nodeIndex`, `nodeList` or `nodeGeo` handle. It lists what was committed when the task started,
not the task's own pending writes. Inside a world it lists the world's view (see
[What a world changed](#worlds-branches-of-the-store) for the world's own changes). `last` holds the last node read,
and a reader resumed after its task was suspended, or built with `last` set, continues after it. A `type` that is not
a node type fails. A full scan reads the header of every block of the store, so it lasts about as long as
`Runtime::store_stats()`.

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

### Exporting the store

`Runtime::export_store()` (admin) writes every node the current task sees, reachable from a module variable or not, to `<backup_path>/export-<µs>.gce` and returns that path. Run it from the CLI with `greycat run runtime::Runtime::export_store`.

Unlike a backup, the export does not copy blocks: it writes values by name in the gcx format (see `GcxWriter` in [stdlib.md](stdlib.md#io)), so it can be read after the program's types changed, or by a tool outside GreyCat.

The file holds, in this order:

1. an info `Map` with the keys `format`, `next_key`, `time` and `world`
2. the root object, whose fields are the module variables
3. for each node, in reference order, a node begin record (tag `25`, then the node handle and the node's type), followed by its entries: the value of a `node<T>`, or key and value pairs for the other node kinds, until the next begin record

The byte layout is in `lib/std/io.md` (GcxWriter / GcxReader).

`greycat import backup/export-<µs>.gce`, run in the project against a new `gcdata/`, reads such a file into the store, giving every node its original reference, so the node handles that values hold stay valid. It prints how many nodes and entries it imported.

The program may have changed since the export:

- A field, or a module variable, that the program no longer has is skipped, and reported as `skipped <type>::<field>: <count> values`. Module variables count as fields of `project::Root`.
- A module variable that the program added gets a new empty node.
- A node or a value whose type the program no longer has fails the import.

The import refuses a store that holds anything but the empty nodes of a fresh store. It is all or nothing: when it fails, it exits non-zero and commits nothing.

## Worlds (branches of the store)

A world is a named branch of the store: tasks running in it read and commit to the world's own block index instead of main's, so what they write stays out of main. Worlds and main share the zone files and every block they have in common, so a world costs only what it writes, plus its index. Merging a world back into main is not available.

```gcl
World::create("what-if", WorldMode::snapshot, null);       // returns the world id; `world` permission; base null: main
World::create("variant", WorldMode::overlay, "what-if"); // reads "what-if" rather than main
World::list();                                // Array<WorldInfo> {id, name, mode, created, user_id, base}; the worlds the caller may use
World::id("what-if");                         // int?, null when there is none
World::current();                             // String, the world the calling task or job runs in, "main" for main
World::reset("variant");                      // empties it of what its tasks wrote; same permission and failures as remove
World::remove("variant");                     // `world` permission: its creator or an admin; fails while a task runs in it or a world is based on it
```

Names take 1 to 128 of `[A-Za-z0-9_.-]`; `main` is reserved for the main store. Ids are never reused.

`World::create`, `World::reset` and `World::remove` each write an `info` record to the `log` stream whose `data` is the world's `WorldInfo`, with the calling task's `user_id`, `task_id` and `src` (see [observability.md](observability.md#where-log-records-go)). `WorldInfo.reset` and `WorldInfo.removed` tell them apart: the time of the reset in `reset`'s record and of the removal in `remove`'s, null otherwise (and always null in `World::list()`).

`World::reset` empties a world of what its tasks wrote, as it was when created, from its base as the base is now: an overlay world reads its base again for every node, a snapshot world copies its base again (which takes as long as creating it), and a standalone world holds nothing, its module variables empty nodes again. `WorldChangesReader` then lists nothing. The world keeps its id, name, creator and the identities allowed to use it, which `remove` then `create` would not: a new world gets a new id, and an identity's world settings keep a world by its id. Neither `reset` nor `remove` runs while a task runs in the world, so neither works from a task in that world, nor while another world is based on it. A call that comes into the world while either works fails as for a world that does not exist (`400`, ``no world has the name of the `world` header``).

A snapshot or overlay world starts from its **base**, the third argument of `World::create`: `null` or `"main"` for main, or the name of another world. A standalone world takes none, and the caller must be allowed to use the base (see [identity.md](identity.md#the-worlds-of-an-identity)). A world that is the base of another cannot be removed. `WorldInfo.base` names it, `"main"` for main, null for a standalone world.

An overlay world based on an overlay world reads, for what it does not hold, what its base reads: the base's own blocks, then the base's base, down to the first world that is not an overlay. Each level keeps the nodes it modified, as an overlay world over main does, and follows the levels below for the others. A chain holds at most 8 overlay worlds, and a read the world does not hold looks up one index per overlay world below it.

A snapshot world based on an overlay world starts from what that world reads when it is created. It shares the blocks of the chain as they are, and writes the nodes the chain modified again, as the overlay world reads them, so creating it also takes time in proportion to what the chain modified. It then reads none of the chain: later commits to any of its levels do not reach it, and it starts a chain of its own for the overlay worlds based on it.

Three modes:

| Mode         | Starts with                          | Sees its base's later commits                                                                             |
| ------------ | ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `snapshot`   | a copy of its base's block index     | Never: the world is the base as it was at creation.                                                       |
| `overlay`    | an empty index, reading its base     | For every node the world has not modified. Once it modifies a node, that node stays as the world left it. |
| `standalone` | an empty index, no base              | Never, nor anything another world held: like a new store, its module variables start as empty nodes of its own. |

A snapshot world's creation copies its base's index into its table, in time proportional to the base's block count (about 50 ms per million blocks), during which no commit runs. Overlay and standalone worlds are created instantly. A standalone world's tasks never look up another index, and the first task to run in it commits the world's module variables before it starts. An overlay world, or a chain of them, may be based on a standalone world that holds nothing yet: its first task commits the base's module variables, in the base, then reads them from there, so the overlay world follows everything the base adds later. In an overlay world, the unit of isolation is the node: a `node<T>` slot, an object, or a whole `nodeTime` / `nodeIndex` / `nodeList` / `nodeGeo`. An overlay world modifying a large `nodeIndex` its base holds copies the index entries of its whole tree on that first commit (entries only, not the blocks); two tasks of the world making that first write to the same node at once conflict, and one fails with a fusion error. Two tasks of a world writing the same node later merge as they do in main.

Picking the world a task runs in:

| Where         | How                                                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| An HTTP call  | Request header `world: <name>`, `main` for main; no header: main, whatever the caller's default world (`Identity.world`), which a client sends itself. An unknown or empty name is answered `400`, a world the caller may not use `403`, with a `core::Error` whose `message` says which (a JSON-RPC call: the error's `message`). |
| `spawn`       | `Job.world`: a name, `"main"`, or `null` for the spawning task's world. A named world the task's user may not use is an error.                                                    |
| `await`       | Always the awaiting task's world; a job naming another one is an error.                                                                                                           |
| The scheduler | `PeriodicOptions.world`; `null` is main. A world the adding user may not use is an error. A run is skipped while its world does not exist.                                        |

Which worlds a user may use, and the world their calls default to, are settings of the identity: see [identity.md, The worlds of an identity](identity.md#the-worlds-of-an-identity).

**What a world changed.** `WorldChangesReader` lists the nodes the current world modified, read as the file readers
read:

```gcl
var r = WorldChangesReader {};
while (r.can_read()) {
    var n = r.read();   // node<T>, nodeTime, nodeIndex, nodeList or nodeGeo
}
```

Each node whose world version differs from its base's current version, among the blocks the world wrote, comes once,
in node reference order. It lists what was committed when the task started, not the task's own pending writes. It
skips a `node<T>` the world set back to the base's value, but lists a `nodeTime`, `nodeIndex`, `nodeList` or `nodeGeo`
whose tree the world rewrote, even when it holds the base's values again. In a snapshot world, a node the base changed after
the world was created also shows when it shares a head block with a node the world modified. A standalone world has no
base: every node it holds is listed, its module variables included. `available()` counts the nodes left; `pos` counts
those read, and a reader resumed after its task was suspended skips them. It fails in main. Building the list costs
nothing at commit: the reader reads the world's commits table and compares block addresses with its base's index.

`WorldDiffReader` shows what changed inside one node: each key whose value differs between the current world and its
base, as a `WorldDiff` with `kind` (`WorldDiffKind::added`, `removed` or `changed`), `key`, and the `base` and `world`
values (null on the side that lacks the key):

```gcl
var prices: nodeIndex<String, float>;

fn show_diff() {
    var nodes = WorldChangesReader {};
    while (nodes.can_read()) {
        var diff = WorldDiffReader { node: nodes.read() };
        while (diff.can_read()) {
            var d = diff.read(); // WorldDiff {kind, key, base, world}
            println("${d.kind} ${d.key}: ${d.base} -> ${d.world}");
        }
    }
}
```

Main holds apple 1.0, pear 2.0 and plum 3.0 in `prices`. An overlay world on main sets apple to 0.8, removes pear and
adds kiwi at 1.5. Run in that world, `show_diff` prints, in hash order:

```
WorldDiffKind::removed pear: 2.0 -> null
WorldDiffKind::added kiwi: null -> 1.5
WorldDiffKind::changed apple: 1.0 -> 0.8
```

plum holds the same value on both sides and is not listed: a key whose two values are deep-equal never is, even when
the world rewrote it. `key` is the time of a `nodeTime`, the position of a `nodeList`, the geo of a `nodeGeo`, the key
of a `nodeIndex`, and null for a `node<T>`. Keys come in the node's order: by time, by position, by encoded key for a
`nodeIndex` with a scalar key and a `nodeGeo`, and by hash for a `nodeIndex` with any other key (`String`, objects).

The base is read as it is now, down its overlay chain, not as it was when the world was created. In a snapshot world,
the base's later changes show up as differences. An overlay world reads the base for a node it never modified, so
that node gives nothing; a node it modified stays as the world left it, so the base's later changes to that node show
up. A standalone world has no base: every key it holds is `added`. A node a task created in the world lists every key
`added`, and a `node<T>` it created and left null lists nothing. A `node<T>` module variable already exists in main,
holding null, from the server's start, so a world's first set of it reads `changed`, with `null` as the base value.

Like `WorldChangesReader`, it compares what was committed when the task started, not the task's own pending writes,
and fails in main; it also fails when `node` is not a node. `last` holds the last entry read, and a new reader given
it (`WorldDiffReader { node: n, last: d }`) resumes after that entry, which pages a large diff across tasks. The resume
is exact when neither the world nor its base committed in between; otherwise, in a `nodeIndex` with a non-scalar key,
the rest of the hash bucket of `last.key` may be missed if that key is gone. A reader whose native state was lost
(task suspended, restored) resumes after `last` the same way.

A diff loads every internal block of the node, once, and the leaves that differ, and steps over the leaves both sides
share at the same address: a parent names its children by key, so a parent both sides share still has its children
compared, at one index lookup per child and side. A few edits to a million-key index cost a few dozen blocks. In an
overlay world, a node whose tree the world never wrote costs its head blocks only, whatever its size, even when the
base changed it since. In a snapshot world, such a node costs its internal blocks (about 30 for a million `int` keys)
and, when its root is a single leaf, that one leaf.

`Task.world`, in what `Task::running()` and a `task: true` call answer, names the world a task runs in (`"main"` for main). It is `null` in `Task::history()`, which does not record it.

`Runtime::store_stats().worlds` reports each world's index entries, the blocks it holds that main does not and their bytes, and its index size. A world's index is a table of `gcdata/index.bin`, next to main's; `gcdata/worlds/registry` lists the worlds. Never edit or copy them by hand. Defrag moves the blocks of every world along with main's.
