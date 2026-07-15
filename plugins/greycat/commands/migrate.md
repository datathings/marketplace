---
name: migrate
description: Schema evolution, data migrations, import/export, and storage health management
allowed-tools: AskUserQuestion, Read, Write, Edit, Bash, Grep, Glob
---

# GreyCat Migration & Database Management

**Purpose**: Schema changes, data transformations, bulk import/export, storage maintenance.

⚠ These ops are **sequential, interactive, and destructive** — do them one at a time, never in parallel, and back up before anything risky.

---

## Step 1: Choose Operation (AskUserQuestion)

- **A. Schema Evolution** — add/remove fields, detect breaking changes
- **B. Data Migration** — transform existing data
- **C. Import/Export** — bulk CSV/JSON ops
- **D. Storage Health** — defrag, backup, diagnostics

---

## Operation A: Schema Evolution

### A.1 Scan models
```bash
find src -name "*.gcl" -not -name "*_api.gcl" -not -name "*_reader.gcl" -not -name "*_writer.gcl"
```
Per file: extract `type Name { ... }` definitions + field lists.

### A.2 Ask change type (AskUserQuestion)
Add field · Remove field · Change field type · Rename field (requires migration).

### A.3 Add field — safety analysis

The runtime diffs the new program against the stored ABI on rebuild and auto-migrates compatible changes:
```
  • Add a NULLABLE attribute            → auto-migrates (existing instances read it as null).
  • Add a NON-NULL attribute WITH a default → auto-migrates (existing instances get the default).
  • Add a NON-NULL attribute WITHOUT a default → load FAILS — provide a default or write a migration.
  • Remove an attribute → auto (dropped on next save). Rename → NOT auto (looks like remove + add).
```

Options:
- **A) Nullable (Recommended)**: `field: T?;` — no data touched, safe
- **B) Non-null with default**: `field: T = <default>;` — auto-migrates existing instances to the default
- **C) Nullable → backfill → tighten**: for a computed (non-constant) backfill, safe for production
- **D) Dev reset**: `rm -rf gcdata` (destructive, dev only)
- **E) Cancel**

Option C (computed backfill) — generate migration:
```gcl
// src/migration/migrate_YYYYMMDD_<desc>.gcl
fn migrate_add_device_priority() {
  info("Start: add Device.priority");
  var count = 0; var errors = 0;
  for (id: int, device in devices_by_id) {
    try {
      device->priority = 1;  // user-provided default
      count = count + 1;
      if (count % 100 == 0) info("Migrated ${count}...");
    } catch (ex) { error("Failed ${id}: ${ex}"); errors = errors + 1; }
  }
  info("Done: ${count} updated, ${errors} errors");
}
```
Then: `greycat run migrate_add_device_priority` → set type non-nullable → `greycat-lang lint`.

### A.4 Remove field
1. Grep usage: `grep -r "entity->field" src/ --include="*.gcl"` — abort if found
2. Edit type to remove the field line
3. `greycat-lang lint`
4. Existing nodes still hold old data — harmless but wastes space; consider `greycat defrag` or full reset

### A.5 Change field type — widen vs narrow
```
  • Widen (e.g. int → int?, int → float) → auto-migrates.
  • Narrow (e.g. int? → int, float → int) → load FAILS on any non-conforming existing value.
```
- **Widening** (nullable, or a type that can represent every prior value) is safe — edit the field's type and rebuild.
- **Narrowing** requires a migration: write a `greycat run` script that reads every instance under the old (wider) shape, validates/coerces each value, then rebuild with the narrower type. If any value can't be coerced, the load fails — fix the data first or keep the field nullable.
- Options mirror A.3: **A) widen only if safe (recommended)**, **B) narrow via a validated migration script**, **C) dev reset**, **D) cancel**.

### A.6 Rename field — NOT automatic
The runtime cannot distinguish a rename from a remove + add — it sees the old attribute disappear and a new one appear, and treats them independently (old data is dropped, new field starts empty/default).
1. Add the new field (nullable or with a default, per A.3)
2. Write a migration (`greycat run`) that copies `old->old_field` into `old->new_field` for every existing instance
3. Remove the old field (A.4) only after the migration has run and been verified
4. `greycat-lang lint`

---

## Operation B: Data Migration

Use when schema changes need data transformation (split name → first/last, compute derived fields, fix data quality).

### Template
```gcl
// src/migration/migrate_YYYYMMDD_<desc>.gcl
fn migrate_transform_user_names() {
  info("Start: transform user names");
  var count = 0; var errors = 0; var skipped = 0;

  for (id: int, user in users_by_id) {
    try {
      if (user->first_name != null) { skipped = skipped + 1; continue; }  // idempotent

      var parts = user->name.split(' ');
      if (parts.size() >= 2) {
        user->first_name = parts[0];
        user->last_name  = parts[parts.size() - 1];
      } else {
        user->first_name = user->name;
        user->last_name  = "";
      }

      count = count + 1;
      if (count % 100 == 0) info("Migrated ${count}...");
    } catch (ex) { error("Failed ${id}: ${ex}"); errors = errors + 1; }
  }
  info("Done: ${count} updated, ${skipped} skipped, ${errors} errors");
}
```

Execute (back up first — `greycat run` mutations are irreversible):
```bash
greycat backup                 # snapshot before mutating (restore with `greycat restore`)
greycat run migrate_transform_user_names
```

---

## Operation C: Import/Export

### C.1 Import CSV

Analyze CSV: detect delimiter (`;`/`\t`/`,`), row count, header. Ask target type via AskUserQuestion (list detected models).

Generate importer with the typed `CsvReader<T>` (there is NO `CSV::parse` / `csv.rows`). Define a `@volatile` row type whose attribute order matches the CSV columns, then stream rows:
```gcl
// src/import/import_devices_csv.gcl

// Row shape — attribute ORDER matches CSV column order. `geo` consumes two columns (lat, lng).
@volatile type DeviceRow { name: String; location: geo; status: String?; }

fn import_devices_from_csv() {
  info("Start CSV import");
  var fmt = CsvFormat { header_lines: 1, separator: ',' };   // separator is a CHAR: ',' not ","
  var reader = CsvReader<DeviceRow> { path: "/path/to/devices.csv", format: fmt };
  var count = 0; var errors = 0;
  while (reader.can_read()) {
    try {
      var row = reader.read();
      DeviceService::create(row.name, row.location, row.status);
      count = count + 1;
      if (count % 100 == 0) info("Imported ${count}...");
    } catch (ex) { error("Row failed: ${ex}"); errors = errors + 1; }
  }
  info("Done: ${count} imported, ${errors} errors");
}
```
(Schema inference on an unfamiliar file: `Csv::analyze([path], null)` → `Csv::generate(stats)` to infer types, or `Csv::sample(reader, 1000)` — given an already-constructed `CsvReader` — to peek the first rows into a `Table`.)

### C.1b ⚠ Re-Import discipline — UPSERT, never duplicate

Re-runnable importers MUST reuse the prior `node<T>` per key. Constructing fresh `node<T>{...}` every run orphans previous nodes and bloats `gcdata/` unboundedly.

```gcl
// ❌ Orphan factory
fn import_devices(rows: Array<DeviceRow>) {
  devices_by_id = nodeIndex<int, node<Device>>{};
  for (i, row in rows) devices_by_id.set(row.id, node<Device>{ ... });  // new node every run
}

// ✅ Upsert — reuse the prior node<T> per key
fn import_devices(rows: Array<DeviceRow>) {
  var prev = devices_by_id;
  devices_by_id = nodeIndex<int, node<Device>>{};
  for (i, row in rows) {
    var existing = prev.get(row.id);
    if (existing != null) {
      existing->name = row.name; existing->location = row.location;   // mutate the resolved payload in place
      devices_by_id.set(row.id, existing);                  // re-point the fresh index at the same node
    } else {
      devices_by_id.set(row.id, node<Device>{ ... });
    }
  }
}
```

When `DeviceService::create` is the only construction path, refactor it to accept `existing: node<Device>?` and branch internally. Run `greycat defrag` after large reshuffles.

### C.2 Export CSV

```gcl
// Row shape written out — attribute order = column order.
@volatile type DeviceOut { id: int; name: String; location: geo; status: String?; created_at: time; }

fn export_devices_to_csv() {
  info("Start CSV export");
  var fmt = CsvFormat { header_lines: 1, separator: ',' };
  var writer = CsvWriter<DeviceOut> { path: "exports/devices_export.csv", format: fmt };
  var count = 0;
  for (id: int, device in devices_by_id) {
    writer.write(DeviceOut {
      id: device->id, name: device->name, location: device->location,
      status: device->status, created_at: device->created_at
    });
    count = count + 1;
  }
  writer.flush();
  info("Exported ${count}");
}
```
```bash
mkdir -p exports && greycat run export_devices_to_csv
```

---

## Operation D: Storage Health

### D.1 Defrag
```bash
du -sh gcdata/
greycat defrag
du -sh gcdata/
```

### D.2 Backup
Prefer built-in backup/restore (consistent snapshot of a live store) over raw `tar`:
```bash
greycat backup                 # snapshot into GREYCAT_BACKUP_PATH (default backup/)
greycat restore <archive>      # stop the server first, then restore
```
In-process equivalents: `Runtime::backup_full()` / `Runtime::backup_delta()`. `GREYCAT_MAX_BACKUP_FILES` (default 3) caps retention. A raw `tar -czf snap.tgz gcdata/` is only consistent when the server is stopped.

### D.3 Diagnostics
```bash
du -sh gcdata/
find gcdata/ -type f | wc -l
find gcdata/ -type f -exec du -h {} \; | sort -rh | head -10
# Avg file size < 1KB suggests fragmentation — consider defrag
```

### D.4 Dev reset (destructive)
```bash
# Prompt user first.
rm -rf gcdata && greycat run import
```

### D.5 Production deploy with safe rollback

ROTATE `gcdata/` rather than deleting it — that's your one-command rollback.

```bash
# Pre-deploy
systemctl stop greycat
mv gcdata gcdata_bk_$(date +%Y%m%d-%H%M%S)
./bin/greycat install
./bin/greycat run import
./bin/greycat defrag           # after major reshuffles
./bin/greycat run smoke        # smoke test
systemctl start greycat

# Rollback if smoke fails
systemctl stop greycat
rm -rf gcdata
mv gcdata_bk_<timestamp> gcdata
systemctl start greycat
```

**Reminders**:
- Use project-local `./bin/greycat` in systemd `ExecStart` — host-wide `~/.greycat/bin/greycat` has been observed stale at `0.0.0`.
- Keep 1-2 rotated `gcdata_bk_*` for at least 1 week post-validation.

---

## Success Criteria

✓ Schema changes safe (no data loss) ✓ Migrations succeed with progress logs ✓ Import/export handles errors ✓ Backups created before risky ops ✓ `greycat-lang lint --fix` passes

---

## Notes

- Backup before risky ops; batch large migrations and log progress every N rows.
- Dev: may delete `gcdata`; Prod: rotate `gcdata`, never delete (see D.5).
- **Frontend ripple**: schema/`…View` changes invalidate the generated client — after a migration run `greycat codegen ts` (regenerate `project.d.ts`) and rebuild the frontend (VitePlus (vp) + Lit (shadow DOM) + TypeScript + Web Awesome (or equivalent) + @greycat/web/sdk + lucide-static, pnpm) so types stay in sync; if payload shapes changed, re-audit with `pnpm lighthouse:ci` (that script if present, else the `lighthouse` CLI).
