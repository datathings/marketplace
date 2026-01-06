# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Datathings Marketplace** - a Claude Code plugin marketplace providing skills and LSP plugins for GreyCat development and local LLM inference.

## Architecture

The marketplace follows a two-level structure:

1. **Marketplace level** (`.claude-plugin/marketplace.json`) - registry of all plugins
2. **Plugin level** (`plugins/<name>/`) - individual skill or LSP plugins

### Plugin Types

**Skill Plugins** (greycat, greycat-c, llamacpp):
- `skills/<name>/SKILL.md` - entry point with YAML frontmatter (`name`, `description`)
- `skills/<name>/references/` - detailed docs loaded on-demand
- `package.sh` - creates `.skill` distribution (zip with .skill extension)

**LSP Plugins** (greycat-lsp):
- `lspServers` in `plugin.json` - LSP server configuration (command, args, transport)
- Requires `greycat-lang` binary in PATH

## Commands

### Package skills
```bash
./package.sh                     # Interactive skill selection
./package.sh -a                  # Package all skills to ./skills/
./package.sh greycat             # Package a specific skill by name
./package.sh -s llamacpp         # Same as above (explicit flag)
./package.sh -c -a               # Clean existing before packaging all
./package.sh -o /output/dir -a   # Custom output directory
```

### Update skills (sync with upstream libs)

Maintenance commands live at the **marketplace root** in `.claude/commands/`:

```bash
/update-greycat          # Sync greycat skill with latest library versions
/update-greycat-c        # Update C SDK documentation
/update-llamacpp         # Sync with upstream llama.cpp release
```

These are **repo maintenance workflows**, not distributed with the skills.

### Create new skill
```bash
.claude/skills/skill-creator/scripts/init_skill.py <skill-name> --path <output-dir>
```

### Bump versions
```bash
./bump-version.sh                # Show current versions
./bump-version.sh 1.3.0          # Bump all plugins to 1.3.0
```

### Test LSP
```bash
which greycat-lang               # Verify binary exists
greycat-lang server              # Start LSP server
```

## Adding a New Plugin

1. Create `plugins/<name>/` directory
2. Add entry to `.claude-plugin/marketplace.json`:
   ```json
   {
     "name": "plugin-name",
     "source": "./plugins/plugin-name",
     "description": "...",
     "version": "1.0.0"
   }
   ```
3. For skills: create `skills/<skill>/SKILL.md` with frontmatter + `package.sh`
4. For LSP: add `lspServers` config in `plugin.json`

## Skill Development Rules

- **SKILL.md under 500 lines** - move details to `references/`
- **Frontmatter is the trigger** - `name` and `description` determine when skill activates
- **References one level deep** - all reference files link directly from SKILL.md
- **No auxiliary files in skills** - no README, CHANGELOG, etc. inside skill directories
- **Progressive disclosure**: metadata always loaded → body on trigger → references as needed

## YAML Frontmatter Requirements

**Critical:** SKILL.md frontmatter must be valid YAML. Common issues:

1. **Quote descriptions with special characters** - colons (`:`) break YAML parsing:
   ```yaml
   # WRONG - colons cause parsing failure
   description: Use when: (1) working with...

   # CORRECT - quoted description
   description: "Use when: (1) working with..."
   ```

2. **Skill name must match directory name**:
   ```
   skills/greycat/SKILL.md  →  name: greycat    ✓
   skills/greycat/SKILL.md  →  name: greycat-lang  ✗
   ```

3. **Only `name` and `description` are required** - other fields optional
