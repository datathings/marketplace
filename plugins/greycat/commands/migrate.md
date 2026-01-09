---
name: migrate
description: Schema evolution, data migrations, import/export, and storage health management
allowed-tools: AskUserQuestion, Read, Write, Edit, Bash, Grep, Glob
---

# GreyCat Migration & Database Management

**Purpose**: Comprehensive database lifecycle management - schema changes, data transformations, bulk operations, storage maintenance

**Run When**: Schema changes, data migrations, bulk import/export, storage maintenance

---

## Overview

This command handles four main database operations:

1. **Schema Evolution** - Add/remove fields safely, detect breaking changes
2. **Data Migration** - Transform existing data after schema changes
3. **Import/Export** - Bulk data operations (CSV, JSON, JSONL)
4. **Storage Health** - Defrag, backup, diagnostics

---

## Step 1: Choose Operation

**Ask user** via AskUserQuestion:

```typescript
AskUserQuestion({
  questions: [{
    question: "What database operation do you need?",
    header: "Operation",
    multiSelect: false,
    options: [
      {
        label: "Schema Evolution",
        description: "Add/remove fields safely, handle schema changes"
      },
      {
        label: "Data Migration",
        description: "Transform existing data after schema changes"
      },
      {
        label: "Import/Export",
        description: "Bulk data import from CSV/JSON or export to files"
      },
      {
        label: "Storage Health",
        description: "Defrag, backup, diagnostics, storage optimization"
      }
    ]
  }]
})
```

---

## Operation A: Schema Evolution

### Phase 1: Detect Changes

**Scan current model files**:

```bash
echo "================================================================================"
echo "SCANNING MODEL DEFINITIONS"
echo "================================================================================"
echo ""

# Find all model files
MODEL_FILES=$(find backend/src/model -name "*.gcl" -type f)

echo "Model files found:"
echo "$MODEL_FILES" | sed 's/^/  /'
echo ""
```

**Parse type definitions** using Grep:

```bash
# Extract all type definitions
for file in $MODEL_FILES; do
    echo "Analyzing: $file"

    # Find type names
    TYPES=$(grep -o "^type [A-Z][a-zA-Z0-9]*" "$file" | awk '{print $2}')

    for type in $TYPES; do
        echo "  Type: $type"

        # Extract fields
        FIELDS=$(grep -A 50 "^type $type" "$file" | \
                 sed -n '/^type/,/^}/p' | \
                 grep -o "^\s*[a-z_][a-zA-Z0-9_]*:" | \
                 sed 's/://;s/^\s*//')

        echo "    Fields: $(echo $FIELDS | tr '\n' ', ')"
    done
    echo ""
done
```

### Phase 2: Safety Analysis

**Ask user what they want to change**:

```typescript
AskUserQuestion({
  questions: [{
    question: "What schema change do you want to make?",
    header: "Change Type",
    multiSelect: false,
    options: [
      {
        label: "Add new field",
        description: "Add a new field to an existing type"
      },
      {
        label: "Remove field",
        description: "Remove an existing field from a type"
      },
      {
        label: "Change field type",
        description: "Modify the type of an existing field"
      },
      {
        label: "Rename field",
        description: "Rename an existing field (requires migration)"
      }
    ]
  }]
})
```

**For "Add new field"**:

```typescript
// Ask for details
AskUserQuestion({
  questions: [
    {
      question: "Which type do you want to modify?",
      header: "Type",
      multiSelect: false,
      options: [/* detected types */]
    },
    {
      question: "Field name?",
      header: "Field Name",
      multiSelect: false,
      options: [{ label: "I'll provide it", description: "" }]
    },
    {
      question: "Field type?",
      header: "Type",
      multiSelect: false,
      options: [
        { label: "String", description: "Text value" },
        { label: "int", description: "64-bit integer" },
        { label: "float", description: "Floating point" },
        { label: "bool", description: "Boolean" },
        { label: "time", description: "Timestamp" },
        { label: "geo", description: "Geographic coordinates" },
        { label: "node<T>", description: "Reference to another type" },
        { label: "Array<T>", description: "List of values" },
        { label: "Custom", description: "I'll specify the type" }
      ]
    },
    {
      question: "Is the field nullable?",
      header: "Nullable",
      multiSelect: false,
      options: [
        { label: "Yes (nullable with ?)", description: "Field can be null (Recommended for new fields)" },
        { label: "No (required)", description: "Field must have a value (requires default or migration)" }
      ]
    }
  ]
})
```

**Safety check**:

```
===============================================================================
SCHEMA CHANGE ANALYSIS
===============================================================================

Type: Device
Change: Add field 'priority: int'

Safety: ⚠ BREAKING CHANGE
Reason: Adding non-nullable field to existing type with persisted data

Impact:
  - Existing Device nodes in gcdata/ don't have 'priority' field
  - Reading existing nodes will fail without migration

Options:
  A) Make field nullable (priority: int?) - SAFE
  B) Provide default value and migrate data - REQUIRES MIGRATION
  C) Cancel and add manually

===============================================================================
```

**Handle based on user choice**:

### Option A: Make Nullable (Safe)

```bash
# Add nullable field to type definition using Edit tool
# Example: add "priority: int?;" to Device type

echo "Adding nullable field to type definition..."

# Use Edit tool to add field
# Original type:
# type Device {
#     id: int;
#     name: String;
# }
#
# Modified:
# type Device {
#     id: int;
#     name: String;
#     priority: int?;
# }

greycat-lang lint --fix

echo "✓ Field added safely (nullable)"
```

### Option B: Provide Default Value (Requires Migration)

**Ask for default value**:

```typescript
AskUserQuestion({
  questions: [{
    question: "What default value should existing nodes get?",
    header: "Default",
    multiSelect: false,
    options: [
      { label: "0", description: "Zero/empty value" },
      { label: "Custom", description: "I'll provide a value" }
    ]
  }]
})
```

**Generate migration function**:

```gcl
// Generated: backend/src/migration/migrate_20260109_add_device_priority.gcl

fn migrate_add_device_priority() {
    info("Starting migration: add Device.priority field");

    var count = 0;
    var errors = 0;

    // Iterate all devices
    for (id: int, device in devices_by_id) {
        try {
            // Set default value
            device->priority = 1;  // User-provided default
            count = count + 1;

            if (count % 100 == 0) {
                info("Migrated ${count} devices...");
            }
        } catch (ex) {
            error("Failed to migrate device ${id}: ${ex}");
            errors = errors + 1;
        }
    }

    info("Migration complete: ${count} devices updated, ${errors} errors");
}
```

**Execute migration**:

```bash
echo "================================================================================"
echo "EXECUTING MIGRATION"
echo "================================================================================"
echo ""

# Run migration function
greycat run migrate_add_device_priority

echo ""
echo "✓ Migration complete"
```

**Update type definition**:

```bash
# Add non-nullable field after migration
# Use Edit tool to add "priority: int;" to Device type

greycat-lang lint --fix
```

### Phase 3: Remove Field (Simpler)

**For "Remove field"**:

1. **Grep for field usage** across codebase:

```bash
echo "Checking usage of field 'old_field' in Device type..."

# Search for usage
USAGE=$(grep -r "device->old_field" backend/ --include="*.gcl")

if [ -z "$USAGE" ]; then
    echo "✓ No usages found - safe to remove"
else
    echo "⚠ Field is used in:"
    echo "$USAGE"
    echo ""
    echo "Please remove these usages first"
    exit 1
fi
```

2. **Remove from type definition**:

```bash
# Use Edit tool to remove field line
echo "Removing field from type definition..."

greycat-lang lint --fix

echo "✓ Field removed"
```

3. **Note about storage**:

```
NOTE: Existing nodes in gcdata/ still contain the old field data.
This is harmless but wastes space. Consider running:
  - greycat defrag (to compact storage)
  - OR: rm -rf gcdata && greycat run import (to rebuild from scratch)
```

---

## Operation B: Data Migration

### Purpose

Transform existing data when schema changes require more than adding/removing fields.

**Examples**:
- Split `name` field into `first_name` and `last_name`
- Compute derived fields from existing data
- Normalize data structures
- Fix data quality issues

### Step 1: Describe Migration

**Ask user**:

```typescript
AskUserQuestion({
  questions: [{
    question: "Describe the data transformation needed:",
    header: "Migration",
    multiSelect: false,
    options: [
      { label: "I'll describe it", description: "Custom transformation logic" }
    ]
  }]
})
```

### Step 2: Generate Migration Template

**Create migration file**:

```gcl
// backend/src/migration/migrate_YYYYMMDD_description.gcl

fn migrate_transform_user_names() {
    info("Starting migration: transform user names");

    var count = 0;
    var errors = 0;
    var skipped = 0;

    // Batch processing (commit every 1000 records)
    var batch_size = 1000;
    var batch_count = 0;

    for (id: int, user in users_by_id) {
        try {
            // Skip if already migrated
            if (user->first_name != null) {
                skipped = skipped + 1;
                continue;
            }

            // Parse full name
            var parts = user->name.split(" ");
            if (parts.size() >= 2) {
                user->first_name = parts[0];
                user->last_name = parts[parts.size() - 1];
            } else {
                user->first_name = user->name;
                user->last_name = "";
            }

            count = count + 1;
            batch_count = batch_count + 1;

            // Progress reporting
            if (count % 100 == 0) {
                info("Migrated ${count} users...");
            }

            // Batch commit (GreyCat handles transactions per function)
            // For very large datasets, consider splitting into multiple function calls

        } catch (ex) {
            error("Failed to migrate user ${id}: ${ex}");
            errors = errors + 1;
        }
    }

    info("Migration complete: ${count} updated, ${skipped} skipped, ${errors} errors");
}
```

### Step 3: Review & Execute

**Show generated migration** to user for review.

**Execute**:

```bash
echo "================================================================================"
echo "EXECUTING MIGRATION"
echo "================================================================================"
echo ""

# Backup first
echo "Creating backup..."
tar -czf gcdata-backup-$(date +%Y%m%d-%H%M%S).tar.gz gcdata/
echo "✓ Backup created"
echo ""

# Run migration
echo "Running migration..."
greycat run migrate_transform_user_names

MIGRATION_EXIT=$?

if [ $MIGRATION_EXIT -eq 0 ]; then
    echo ""
    echo "✓ Migration completed successfully"
else
    echo ""
    echo "⚠ Migration failed - restore from backup if needed"
    echo "To restore: tar -xzf gcdata-backup-*.tar.gz"
fi
```

---

## Operation C: Import/Export

### C1: Import CSV

**Step 1: Analyze CSV**

**Ask for CSV file path**:

```bash
# User provides path
CSV_FILE="path/to/data.csv"

# Read header
HEADER=$(head -1 "$CSV_FILE")
echo "CSV Columns: $HEADER"

# Detect delimiter (comma, semicolon, tab)
if grep -q ";" <<< "$HEADER"; then
    DELIMITER=";"
elif grep -q $'\t' <<< "$HEADER"; then
    DELIMITER=$'\t'
else
    DELIMITER=","
fi

# Count rows
ROW_COUNT=$(wc -l < "$CSV_FILE")
echo "Rows: $ROW_COUNT"
```

**Ask mapping**:

```typescript
AskUserQuestion({
  questions: [{
    question: "Map CSV to which GreyCat type?",
    header: "Target Type",
    multiSelect: false,
    options: [/* detected types from model files */]
  }]
})
```

**Step 2: Generate Import Function**

```gcl
// backend/src/import/import_devices_csv.gcl

fn import_devices_from_csv() {
    info("Starting CSV import...");

    // Read CSV
    var csv = CSV::parse("/path/to/devices.csv", true);  // true = has header

    var count = 0;
    var errors = 0;

    for (row in csv.rows) {
        try {
            // Map CSV columns to type fields
            var name = row.get("name") as String;
            var lat = row.get("latitude") as float;
            var lng = row.get("longitude") as float;
            var status = row.get("status") as String?;

            // Create entity via service
            var device = DeviceService::create(name, lat, lng, status);
            count = count + 1;

            if (count % 100 == 0) {
                info("Imported ${count} devices...");
            }
        } catch (ex) {
            error("Failed to import row: ${ex}");
            errors = errors + 1;
        }
    }

    info("Import complete: ${count} imported, ${errors} errors");
}
```

**Step 3: Execute Import**

```bash
echo "================================================================================"
echo "IMPORTING CSV"
echo "================================================================================"
echo ""

greycat run import_devices_from_csv

echo ""
echo "✓ Import complete"
```

### C2: Export to CSV

**Step 1: Select Data**

**Ask what to export**:

```typescript
AskUserQuestion({
  questions: [{
    question: "Which data to export?",
    header: "Export",
    multiSelect: false,
    options: [
      { label: "All Devices", description: "Export devices_by_id index" },
      { label: "All Users", description: "Export users_by_id index" },
      { label: "Custom query", description: "Filter data before export" }
    ]
  }]
})
```

**Step 2: Generate Export Function**

```gcl
// backend/src/export/export_devices_csv.gcl

fn export_devices_to_csv() {
    info("Starting CSV export...");

    var rows = Array<Array<String>> {};

    // Header
    rows.add(Array<String> { "id", "name", "latitude", "longitude", "status", "created_at" });

    var count = 0;

    // Data rows
    for (id: int, device in devices_by_id) {
        var row = Array<String> {
            device->id.toString(),
            device->name,
            device->location.lat.toString(),
            device->location.lng.toString(),
            device->status ?? "",
            device->created_at.toString()
        };
        rows.add(row);
        count = count + 1;
    }

    // Write CSV
    var csv = CSV::write(rows);
    var file = File::create("exports/devices_export_$(Time::now()).csv");
    file.write(csv);

    info("Export complete: ${count} devices exported");
}
```

**Step 3: Execute Export**

```bash
echo "================================================================================"
echo "EXPORTING TO CSV"
echo "================================================================================"
echo ""

mkdir -p exports

greycat run export_devices_to_csv

echo ""
echo "✓ Export complete"
ls -lh exports/*.csv | tail -1
```

---

## Operation D: Storage Health

### D1: Defrag

**Run defragmentation**:

```bash
echo "================================================================================"
echo "STORAGE DEFRAGMENTATION"
echo "================================================================================"
echo ""

# Check current size
BEFORE_SIZE=$(du -sh gcdata/ | awk '{print $1}')
echo "Current size: $BEFORE_SIZE"
echo ""

# Run defrag
echo "Running defrag..."
greycat defrag

# Check new size
AFTER_SIZE=$(du -sh gcdata/ | awk '{print $1}')
echo ""
echo "After defrag: $AFTER_SIZE"
echo "Saved: $(expr $(du -s gcdata/ | awk '{print $1}') - $(du -s gcdata/ | awk '{print $1}'))B"
echo ""
echo "✓ Defrag complete"
```

### D2: Backup

**Create backup**:

```bash
echo "================================================================================"
echo "CREATING BACKUP"
echo "================================================================================"
echo ""

BACKUP_NAME="gcdata-backup-$(date +%Y%m%d-%H%M%S).tar.gz"

echo "Creating backup: $BACKUP_NAME"
tar -czf "$BACKUP_NAME" gcdata/

BACKUP_SIZE=$(du -sh "$BACKUP_NAME" | awk '{print $1}')

echo ""
echo "✓ Backup created: $BACKUP_NAME ($BACKUP_SIZE)"
echo ""
echo "To restore:"
echo "  rm -rf gcdata"
echo "  tar -xzf $BACKUP_NAME"
```

### D3: Diagnostics

**Analyze storage**:

```bash
echo "================================================================================"
echo "STORAGE DIAGNOSTICS"
echo "================================================================================"
echo ""

# Size
echo "Storage size:"
du -sh gcdata/
echo ""

# File count
echo "Files in storage:"
find gcdata/ -type f | wc -l
echo ""

# Largest files
echo "Largest files:"
find gcdata/ -type f -exec du -h {} \; | sort -rh | head -10
echo ""

# Age
echo "Last modified:"
stat -c "%y" gcdata/ 2>/dev/null || stat -f "%Sm" gcdata/
echo ""

# Fragmentation estimate (simple heuristic)
TOTAL_SIZE=$(du -sb gcdata/ | awk '{print $1}')
FILE_COUNT=$(find gcdata/ -type f | wc -l)
AVG_FILE_SIZE=$(expr $TOTAL_SIZE / $FILE_COUNT)

echo "Average file size: $(numfmt --to=iec $AVG_FILE_SIZE 2>/dev/null || echo ${AVG_FILE_SIZE}B)"

if [ $AVG_FILE_SIZE -lt 1000 ]; then
    echo "⚠ High fragmentation suspected (many small files)"
    echo "  Consider running: greycat defrag"
else
    echo "✓ Fragmentation looks healthy"
fi
```

**Generate report**:

```
===============================================================================
STORAGE HEALTH REPORT
===============================================================================

Size: 2.4 GB
Files: 15,432
Last modified: 2026-01-09 10:30:00

Largest files:
  1.2 GB - gcdata/nodeIndex_devices_by_id
  450 MB - gcdata/nodeTime_sensor_readings
  320 MB - gcdata/nodeList_cities_streets

Fragmentation: ✓ Healthy
Average file size: 158 KB

Recommendations:
  [✓] Storage size normal
  [ ] Consider defrag if performance degrades
  [ ] Setup automated backups
  [✓] No immediate issues detected

===============================================================================
```

---

## Success Criteria

✓ **Schema changes applied** safely without data loss
✓ **Migrations executed** successfully with progress reporting
✓ **Import/export complete** with error handling
✓ **Storage optimized** via defrag when needed
✓ **Backups created** before risky operations
✓ **greycat-lang lint --fix passes** after schema changes

---

## Notes

- **Backup before migrations**: Always backup gcdata/ before risky operations
- **Test migrations**: Run on copy of data first
- **Batch processing**: Large migrations should batch commits
- **Progress reporting**: Log progress every N records
- **Rollback plan**: Keep backups and migration reversal functions
- **Dev vs Prod**: In dev, can delete gcdata/; in prod, must migrate
