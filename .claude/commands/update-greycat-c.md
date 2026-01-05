# Update GreyCat C Skill

Update the GreyCat C SDK skill documentation to match the latest SDK version.

**Working directory:** `plugins/greycat-c/`

## Source of Truth

The GreyCat C SDK headers are the authoritative source. These are typically found in:
- GreyCat installation: headers in the GreyCat SDK distribution
- Or from the GreyCat source repository

## Update Process

1. **Review Current State**
   - Read `skills/greycat-c/SKILL.md` to understand current coverage
   - Check `skills/greycat-c/references/api_reference.md` for C API functions
   - Check `skills/greycat-c/references/standard_library.md` for std library

2. **Identify Updates Needed**
   - Compare with latest GreyCat C SDK headers
   - Note any new functions, changed signatures, or deprecated APIs
   - Check for new standard library modules

3. **Update Documentation**
   - Update `api_reference.md` with new/changed C API functions
   - Update `standard_library.md` with new/changed std modules
   - Update `SKILL.md` overview if scope changed

4. **Validate Changes**
   - Ensure all function signatures are accurate
   - Verify examples compile and work
   - Check for consistency across files

5. **Package the Skill**
   ```bash
   # From repo root
   ./package.sh greycat-c
   ```
   This creates `skills/greycat-c.skill`.

## Files to Update

| File | Content |
|------|---------|
| `skills/greycat-c/SKILL.md` | Overview, quick reference |
| `skills/greycat-c/references/api_reference.md` | C API functions, types, macros |
| `skills/greycat-c/references/standard_library.md` | Standard library modules |

## Checklist

- [ ] C API functions are up to date
- [ ] Standard library modules are current
- [ ] All signatures match SDK headers
- [ ] Examples are working
- [ ] SKILL.md overview reflects current state
- [ ] Version numbers updated if applicable
