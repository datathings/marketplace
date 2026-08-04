# Language tooling: `greycat lint`, `greycat fmt`, `greycat lsp`

Three `greycat` commands cover static analysis: `lint` (checks), `fmt` (formatter), `lsp` (language
server). Where `greycat build` answers "does this program load?", they answer "is this program _clean_?" -
catching unused locals, redundant null-checks, non-exhaustive enum chains, and dozens of other shape issues.

They are served by the `lang` library, which `greycat` loads from `<cwd>/lib/lang/` first, then
`~/.greycat/lib/lang/`. A global GreyCat install ships it; a project pins its own with
`@library("lang", "<version>");` in `project.gcl` followed by `greycat install`. When neither location has
it, the three commands exit `127` with `greycat: language tools not found`.

## When to reach for it

Run **`greycat fmt --mode=check`** + **`greycat lint`** as the _definition of done_ before declaring any
`.gcl` change finished. Both are fast (well under a second on most projects), exit non-zero on issues, and
integrate cleanly into pre-commit hooks and CI. `greycat build` is not a substitute - it produces a
`project.gcp` even when the source has `unused-local` warnings or formatting drift.

## Contents

- Synopsis
- `lint` - static checks
- `fmt` - formatter
- `lsp` - language server
- Suppression directives
- Recommended pre-commit workflow

## Synopsis

```sh
greycat lint [options] [project]
greycat fmt [options] [project]
greycat lsp
```

Where `[project]` is a path to either:

- A `project.gcl` entrypoint (or a directory containing one), or
- A single `.gcl` file - the tooling walks up to the enclosing project root, analyzes the whole closure for
  cross-module bindings, then scopes its output to just the input file.

When `[project]` is omitted, it looks for `project.gcl` in the current working directory. The closure is
computed from the entrypoint's `@library` / `@include` pragmas - only reachable modules are analyzed.

`greycat lint -h` / `greycat fmt -h` is the source of truth for the flags your installed version accepts.

## `lint` - static checks

```sh
greycat lint                       # lint the project in the cwd
greycat lint path/to/project.gcl   # explicit entrypoint
greycat lint src/api.gcl           # single-file scope (still uses the project closure)
```

Exits `0` only if there are no diagnostics at all. Any warning OR error produces exit `1`.

### Options

| Flag                | Meaning                                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--fix[=<RULE>]`    | Apply auto-fixable suggestions in place (max 5 passes). Bare `--fix` fixes everything; `--fix=<rule>[,<rule>]` fixes only the named rules.                 |
| `--format <FORMAT>` | Diagnostic rendering: `compact` (one line per diagnostic), `pretty` (snippet + caret, default on a TTY), `csv` (per-file timings), `quiet` (summary only). |
| `--level <LEVEL>`   | Minimum severity to print: `error`, `warning`, `hint` (default). Display-only - the summary totals and the exit code are unaffected.                       |
| `--lint-libs`       | Also lint `lib/<name>/` modules. Off by default - project-only.                                                                                            |
| `--list-rules`      | Print every registered rule with a one-line summary, then exit. Use to discover newly added rules.                                                         |
| `--no-suppressions` | Re-emit diagnostics silenced by `// gcl-lint-off ...` directives. Useful for auditing suppression debt.                                                    |
| `--off <RULE>`      | Silence rule(s) globally. Repeatable or comma-list.                                                                                                       |
| `--on <RULE>`       | Enable advisory rule(s) that ship off by default (e.g. `no-breakpoint`).                                                                                  |
| `--color <COLOR>`   | `auto` (default), `always`, `never`. Respects `NO_COLOR`.                                                                                                 |

### Notable rules

Run `greycat lint --list-rules` for the live set. Rules to know:

- **`unused-local`** - `var name = ...;` bound but never read. Auto-fix replaces with `_` for loop iterators.
- **`unused-param`** / **`unused-decl`** / **`unused-generic-param`** - fire on dead surface area. Rename to
  `_` to silence intentionally.
- **`duplicate-decl`** - two top-level decls share a name. Error.
- **`modvar-must-be-node-tag`**, **`modvar-node-cannot-be-nullable`**, **`modvar-node-inner-must-be-nullable`**:
  module-level `var` declarations must be node tags with the right nullability shape. See
  [project.md](project.md).
- **`arrow-on-non-deref`** - `->` used on a non-node receiver. The single most common drift from "I wrote
  TypeScript". See [idioms.md](idioms.md) section on member access.
- **`possibly-null`** - `.` / `->` / `[...]` on a possibly-null receiver. Narrow with `if (x != null) ...`,
  force with `!!`, or coalesce with `??`.
- **`nullable-operand`** - a possibly-null operand on an arithmetic, relational, or logical operator. The
  runtime throws on a `null` operand. Equality (`==` / `!=`) is exempt.
- **`redundant-nullable-access`** / **`redundant-non-null-assertion`** / **`redundant-coalesce`** - `?.`, `!!`,
  or `??` on a value already known to be non-null. Cleanup hints.
- **`non-exhaustive`** - chained `if (x == E::A) ... else if (x == E::B) ...` over an enum that misses a
  variant and has no `else`. The GCL replacement for `switch` exhaustiveness checks.
- **`decidable-condition`** / **`exhaustive-is-check`** - `while (true) {}` and other statically decidable
  conditions; an `is` check every value matches, leaving a branch unreachable. Suppress when intentional.
- **`unused-catch-param`** / **`catch-empty-parens`** - `catch (e)` that never reads `e` (auto-fix drops the
  binding), and `catch ()`, which is an error - the no-binding form is `catch { ... }`.
- **`redundant-semicolon`** - stray `;` after `fn f() {};` / `type T {};`. The runtime rejects it; the
  auto-fix removes it.
- **`no-breakpoint`** - advisory rule, off by default. Enable with `--on=no-breakpoint` to catch `breakpoint;`
  left in committed code.
- **`literal-overflow`** - numeric literal exceeds its type's range, or loses float precision.

In `--list-rules` output, each rule carries its severity plus two markers: `*` means advisory (off by
default; enable with `--on`), `f` means the rule has an auto-fix (apply with `--fix=<rule>`).

### Single-file vs project mode

Pass a single `.gcl` file when you want output scoped to one module. The whole project closure is still
analyzed (so cross-module bindings resolve), then diagnostics are filtered to the file you named. Use this
for IDE-style "what's wrong with _this_ file?" queries; use project mode for CI gates.

## `fmt` - formatter

```sh
greycat fmt                        # default mode: write
greycat fmt --mode=check           # exit non-zero on drift (CI gate)
greycat fmt --mode=diff            # unified diff per file
greycat fmt --mode=stdout          # format the entrypoint only, print to stdout
```

### Modes

| Mode     | Behavior                                                                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write`  | (Default.) Rewrites every resolved `.gcl` file in place. Touch-free for files that are already canonical.                                           |
| `check`  | Exits non-zero on drift, listing every file that would change. **The CI / pre-commit mode.**                                                        |
| `diff`   | Prints a unified diff per file (colored on a TTY). Use to preview what `write` would do without mutating anything.                                  |
| `stdout` | Formats only the entrypoint and prints to stdout. The `@library` / `@include` closure is **ignored** - single-file mode for piping or quick checks. |

### Options

| Flag         | Meaning                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `--fmt-libs` | Also format files under `lib/<name>/`. Off by default - projects shouldn't reformat their deps. |
| `--color`    | `auto` / `always` / `never`. Applies to `diff` mode output.                                    |

The formatter is opinionated and unconfigurable - there's no `.gclfmt` style file. Disagreements with its
output are bugs to file, not knobs to tune.

## `lsp` - language server

```sh
greycat lsp
```

Speaks the Language Server Protocol over stdio (the only supported transport; a `--stdio` flag is accepted
and ignored). Editors with a configured `.gcl` language client get diagnostics, hover, go-to-definition, and
find-references in real time.

A typical editor config runs `greycat lsp` and associates `*.gcl`. The server picks the nearest `project.gcl`
walking up from each opened file - see [project.md](project.md) on multi-project workspaces.

## Suppression directives

Suppress lint diagnostics with line- or range-scoped comments:

```gcl
// gcl-lint-off unused-local
var x = compute();          // `unused-local` is ignored here
// gcl-lint-on unused-local
```

```gcl
// gcl-lint-off unused-local, possibly-null
// ... block that knowingly triggers both ...
// gcl-lint-on unused-local, possibly-null
```

Multi-rule suppressions take a comma-separated rule list. `// gcl-lint-on` (no rule names) re-enables every
rule suppressed in scope.

Project-wide policy goes in `project.gcl` only:

```gcl
@lint_off("no-breakpoint");        // disable globally
@lint_on("possibly-null");         // force-enable an advisory rule globally
```

The `lint-pragma-outside-entrypoint` rule flags `@lint_off` / `@lint_on` in any other module - project policy
belongs in the entrypoint.

There are guard rules for the suppression mechanism itself:

- **`unused-suppression`** - a `// gcl-lint-off ...` that didn't actually silence anything (the underlying
  rule never fired in the scope).
- **`unknown-suppression-rule`** - a `// gcl-lint-off ...` that names a rule the linter doesn't know.
- **`empty-suppression`** - a `// gcl-lint-off` with an empty rule list.
- **`unbalanced-lint-off`** / **`unbalanced-fmt-off`** - a `// gcl-lint-off ...` / `// gcl-fmt-off` with no
  matching `...-on`.
- **`conflicting-lint-pragma`** - a module declaring both `@lint_on("R")` and `@lint_off("R")` for the same
  rule. `@lint_off` wins; the other is dead.

Audit accumulated suppression debt with `greycat lint --no-suppressions`.

## Recommended pre-commit workflow

The "definition of done" for any `.gcl` change:

```sh
greycat fmt --mode=check       # exit non-zero on formatting drift
greycat lint                   # exit non-zero on any diagnostic
greycat build                  # produce project.gcp
greycat test                   # run @test functions
```

In a pre-commit hook:

```sh
#!/usr/bin/env bash
set -euo pipefail
greycat fmt --mode=check
greycat lint
```

Apply formatter output directly with `greycat fmt` (default `--mode=write`). Apply auto-fixable lint
suggestions with `greycat lint --fix`, or narrow it to one rule with `greycat lint --fix=unused-local`. Both
are idempotent - re-running on already-clean code is a no-op.

If the three commands exit `127`, the `lang` library is missing: reinstall GreyCat, or pin it in the project
with `@library("lang", "<version>");` and run `greycat install`.
