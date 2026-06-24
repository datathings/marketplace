---
name: migrate
description: Schema evolution, data migrations, import/export, and storage health management
allowed-tools: AskUserQuestion, Read, Write, Edit, Bash, Grep, Glob
---

# GreyCat Migration & Database Management

**Purpose**: Schema changes, data transformations, bulk import/export, storage maintenance.

---

## Step 1: Choose Operation (AskUserQuestion)

Options:
- **A. Schema Evolution** — add/remove fields, detect breaking changes
- **B. Data Migration** — transform existing data
- **C. Import/Export** — bulk CSV/JSON ops
- **D. Storage Health** — defrag, backup, diagnostics

---

## Operation A: Schema Evolution

### A.1 Scan models
\`\`\`bash
find src -name "*.gcl" -not -name "*_api.gcl" -not -name "*_reader.gcl" -not -name "*_writer.gcl"
\`\`\`
For each file: extract `type Name { ... }` definitions and field lists.

### A.2 Ask change type (AskUserQuestion)
- Add new field
- Remove field
- Change field type
- Rename field (requires migration)

### A.3 Add field — safety analysis

```
⚠ Adding non-nullable field to persisted type with existing data FAILS OUTRIGHT.
No automatic migration — gcdata/ must be reset (dev) OR field added nullable first.
```

Options:
- **A) Nullable (Recommended)**: add `field: T?;` — no data touched, safe
- **B) Nullable → backfill → tighten**: safe for production
- **C) Dev reset**: `rm -rf gcdata` (destructive, dev only)
- **D) Cancel**

For Option B, generate migration:
\`\`\`gcl
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
\`\`\`
Then: `greycat run migrate_add_device_priority` → update type to non-nullable → `greycat-lang lint`.

### A.4 Remove field
1. Grep for usage: `grep -r "entity->field" src/ --include="*.gcl"` — abort if found
2. Edit type definition to remove field line
3. `greycat-lang lint`
4. Note: existing nodes still contain old data — harmless but wastes space; consider `greycat defrag` or full reset

---

## Operation B: Data Migration

Use when schema changes need data transformation (split name → first/last, compute derived fields, fix data quality).

### Template
\`\`\`gcl
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
\`\`\`

Execute:
\`\`\`bash
tar -czf gcdata-backup-$(date +%Y%m%d-%H%M%S).tar.gz gcdata/
greycat run migrate_transform_user_names
\`\`\`

---

## Operation C: Import/Export

### C.1 Import CSV

Analyze CSV: detect delimiter (`;`/`\t`/`,`), row count, header.

Ask target type via AskUserQuestion (list detected models).

Generate importer:
\`\`\`gcl
// src/import/import_devices_csv.gcl
fn import_devices_from_csv() {
  info("Start CSV import");
  var csv = CSV::parse("/path/to/devices.csv", true);
  var count = 0; var errors = 0;
  for (row in csv.rows) {
    try {
      var name   = row.get("name")   as String;
      var lat    = row.get("latitude") as float;
      var lng    = row.get("longitude") as float;
      var status = row.get("status") as String?;
      DeviceService::create(name, lat, lng, status);
      count = count + 1;
      if (count % 100 == 0) info("Imported ${count}...");
    } catch (ex) { error("Row failed: ${ex}"); errors = errors + 1; }
  }
  info("Done: ${count} imported, ${errors} errors");
}
\`\`\`

### C.1b ⚠ Re-Import discipline — UPSERT, never duplicate

Re-runnable importers MUST reuse prior `node<T>` per key. Constructing fresh `node<T>{...}` every run orphans previous nodes and bloats `gcdata/` unboundedly.

\`\`\`gcl
// ❌ Orphan factory
fn import_devices() {
  devices_by_id = nodeIndex<int, node<Device>>{};
  for (row in csv.rows) devices_by_id.set(row.id, node<Device>{ ... });  // new node every run
}

// ✅ Upsert
fn import_devices() {
  var prev = devices_by_id;
  devices_by_id = nodeIndex<int, node<Device>>{};
  for (row in csv.rows) {
    var existing = prev.get(row.id);
    if (existing != null) {
      existing->name = row.name; existing->lat = row.lat;
      devices_by_id.set(row.id, existing);
    } else {
      devices_by_id.set(row.id, node<Item>{ ... });
    }
  }
}
\`\`\`

When `DeviceService::create` is the only construction path, refactor it to accept `existing: node<Device>?` and branch internally. Run `greycat defrag` after large reshuffles.

### C.2 Export CSV

\`\`\`gcl
fn export_devices_to_csv() {
  info("Start CSV export");
  var rows = Array<Array<String>> {};
  rows.add(Array<String> { "id", "name", "lat", "lng", "status", "created_at" });
  var count = 0;
  for (id: int, device in devices_by_id) {
    rows.add(Array<String> {
      device->id.toString(), device->name,
      device->location.lat.toString(), device->location.lng.toString(),
      device->status ?? "", device->created_at.toString()
    });
    count = count + 1;
  }
  var csv = CSV::write(rows);
  File::create("exports/devices_export_${time::now()}.csv").write(csv);
  info("Exported ${count}");
}
\`\`\`
\`\`\`bash
mkdir -p exports && greycat run export_devices_to_csv
\`\`\`

---

## Operation D: Storage Health

### D.1 Defrag
\`\`\`bash
du -sh gcdata/
greycat defrag
du -sh gcdata/
\`\`\`

### D.2 Backup
\`\`\`bash
BACKUP=gcdata-backup-$(date +%Y%m%d-%H%M%S).tar.gz
tar -czf "$BACKUP" gcdata/
# restore: rm -rf gcdata && tar -xzf $BACKUP
\`\`\`

### D.3 Diagnostics
\`\`\`bash
du -sh gcdata/
find gcdata/ -type f | wc -l
find gcdata/ -type f -exec du -h {} \; | sort -rh | head -10
# Avg file size < 1KB suggests fragmentation — consider defrag
\`\`\`

### D.4 Dev reset (destructive)
\`\`\`bash
# Prompt user first.
rm -rf gcdata && greycat run import
\`\`\`

### D.5 Production deploy with safe rollback

ROTATE `gcdata/` rather than deleting it — that's your one-command rollback.

\`\`\`bash
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
\`\`\`

**Reminders**:
- Use project-local `./bin/greycat` in systemd `ExecStart` — host-wide `~/.greycat/bin/greycat` has been observed stale at `0.0.0`.
- Keep 1-2 rotated `gcdata_bk_*` for at least 1 week post-validation.

---

## Success Criteria

✓ Schema changes safe (no data loss) ✓ Migrations succeed with progress logs ✓ Import/export handles errors ✓ Backups created before risky ops ✓ `greycat-lang lint --fix` passes

---

## Notes

- Always backup before risky ops
- Batch large migrations + log progress every N rows
- Dev: can delete gcdata; Prod: must migrate (rotate, not delete)
- **Frontend ripple**: schema/`…View` changes invalidate the generated client — after a migration, run `pnpm gen` (regenerate `project.d.ts`) and rebuild the **Lit + Shoelace + Lucide** frontend so types stay in sync; re-audit with `pnpm lighthouse:ci` if payload shapes changed.
