# greycat-lang

`greycat-lang` is the static analysis tool for GreyCat projects: linter, formatter, LSP server. Where `greycat build` answers "does this program load?", the lang answers "is this program _clean_?" — catching unused locals, redundant null-checks, non-exhaustive enum chains, and dozens of other shape issues.

## When to reach for it

Run **`greycat-lang`** + **`greycat-lang --mode=check`** as the _definition of done_ before declaring any `.gcl` change finished. Both are fast (sub-second on small projects, well under a second on most), exit non-zero on issues, and integrate cleanly into pre-commit hooks and CI. `greycat build` is not a substitute — it produces a `project.gcp` even when the source has `unused-local` warnings or formatting drift.

## Contents

- Synopsis
- `lint` — static checks
- `fmt` — formatter
- `server` (or `lsp` for the dylib) — LSP
- Suppression directives
- Recommended pre-commit workflow

## Synopsis

```sh
greycat-lang <command> [options] [project]
```

Where `[project]` is a path to either:

- A `project.gcl` entrypoint (or a directory containing one), or
- A single `.gcl` file — the lang walks up to the enclosing project root, analyzes the whole closure for cross-module bindings, then scopes its output to just the input file.

When `[project]` is omitted, the lang looks for `project.gcl` in the current working directory. The closure is computed from the entrypoint's `@library` / `@include` pragmas — only reachable modules are analyzed.

`greycat-lang -V` prints the version. `greycat-lang <cmd> -h` is the source of truth for the flags your installed version accepts.

**IMPORTANT**: It is distributed as a single standalone binary named `greycat-lang` AND as a dynamic library `libgreycat_lang.so` (.dylib, .dll) that greycat uses when found either locally in `./lib/lang/` or globally in `~/.greycat/lib/lang/`. When greycat finds the lang library it will also provide the same commands directly from `greycat`'s CLI:
- `greycat-lang lint` -> `greycat lint`
- `greycat-lang fmt` -> `greycat fmt`
- `greycat-lang server` -> `greycat lsp`
The dynamic library and the static binary are installed automatically when greycat is installed globally, but it can also be installed locally with `@library("lang", "<version>");` as a library in a `project.gcl`.

## `lint` — static checks

```sh
greycat-lang lint                       # lint the project in the cwd
greycat-lang lint path/to/project.gcl   # explicit entrypoint
greycat-lang lint src/api.gcl           # single-file scope (still uses the project closure)
```

Exits `0` only if there are no diagnostics at all. Any warning OR error produces exit `1`.

### Options

| Flag                | Meaning                                                                                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--fix`             | Apply auto-fixable suggestions in place (max 5 passes).                                                                                                    |
| `--format <FORMAT>` | Diagnostic rendering: `compact` (one line per diagnostic), `pretty` (snippet + caret, default on a TTY), `csv` (per-file timings), `quiet` (summary only). |
| `--lint-libs`       | Also lint `lib/<name>/` modules. Off by default — project-only.                                                                                            |
| `--list-rules`      | Print every registered rule with a one-line summary, then exit. Use to discover newly added rules.                                                         |
| `--no-suppressions` | Re-emit diagnostics silenced by `// gcl-lint-off …` directives. Useful for auditing suppression debt.                                                      |
| `--off <RULE>`      | Silence rule(s) globally. Repeatable or comma-list.                                                                                                        |
| `--on <RULE>`       | Enable advisory rule(s) that ship off by default (e.g. `no-breakpoint`).                                                                                   |
| `--color <COLOR>`   | `auto` (default), `always`, `never`. Respects `NO_COLOR`.                                                                                                  |

### Notable rules

Run `greycat-lang lint --list-rules` for the live set. Rules to know:

- **`unused-local`** — `var name = …;` bound but never read. Auto-fix replaces with `_` for loop iterators.
- **`unused-param`** / **`unused-decl`** / **`unused-generic-param`** — fire on dead surface area. Rename to `_` to silence intentionally.
- **`duplicate-decl`** — two top-level decls share a name. Error.
- **`modvar-must-be-node-tag`**, **`modvar-node-cannot-be-nullable`**, **`modvar-node-inner-must-be-nullable`** — module-level `var` declarations must be node tags with the right nullability shape. See [project.md](project.md).
- **`arrow-on-non-deref`** — `->` used on a non-node receiver. The single most common drift from "I wrote TypeScript". See [idioms.md](idioms.md) § member access.
- **`possibly-null`** — `.` / `->` / `[…]` on a possibly-null receiver. Narrow with `if (x != null) …`, force with `!!`, or coalesce with `??`.
- **`redundant-nullable-access`** / **`redundant-non-null-assertion`** / **`redundant-coalesce`** — `?.`, `!!`, or `??` on a value the lang already knows is non-null. Cleanup hints.
- **`non-exhaustive`** — chained `if (x == E::A) … else if (x == E::B) …` over an enum that misses a variant and has no `else`. The GCL replacement for `switch` exhaustiveness checks.
- **`decidable-condition`** — `if (x is int && x is float)`, `while (true) {}`, etc. — static contradictions / tautologies. Suppress when intentional.
- **`redundant-semicolon`** — stray `;` after `fn f() {};` / `type T {};`. The runtime rejects it; the lang auto-fixes.
- **`no-breakpoint`** — advisory rule, off by default. Enable with `--on=no-breakpoint` to catch `breakpoint;` left in committed code.
- **`literal-overflow`** — numeric literal exceeds its type's range, or loses float precision.

`*` next to a rule in `--list-rules` means advisory (off by default; enable with `--on`).

### Single-file vs project mode

Pass a single `.gcl` file when you want output scoped to one module. The lang still walks up to the project root and analyzes the full closure (so cross-module bindings resolve), then filters diagnostics to the file you named. Use this for IDE-style "what's wrong with _this_ file?" queries; use project mode for CI gates.

## `fmt` — formatter

```sh
greycat-lang fmt                        # default mode: write
greycat-lang fmt --mode=check           # exit non-zero on drift (CI gate)
greycat-lang fmt --mode=diff            # unified diff per file
greycat-lang fmt --mode=stdout          # format the entrypoint only, print to stdout
```

### Modes

| Mode     | Behavior                                                                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write`  | (Default.) Rewrites every resolved `.gcl` file in place. Touch-free for files that are already canonical.                                           |
| `check`  | Exits non-zero on drift, listing every file that would change. **The CI / pre-commit mode.**                                                        |
| `diff`   | Prints a unified diff per file (colored on a TTY). Use to preview what `write` would do without mutating anything.                                  |
| `stdout` | Formats only the entrypoint and prints to stdout. The `@library` / `@include` closure is **ignored** — single-file mode for piping or quick checks. |

### Options

| Flag         | Meaning                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `--fmt-libs` | Also format files under `lib/<name>/`. Off by default — projects shouldn't reformat their deps. |
| `--color`    | `auto` / `always` / `never`. Applies to `diff` mode output.                                     |

The formatter is opinionated and unconfigurable — there's no `.gclfmt` style file. Disagreements with its output are bugs to file against the lang, not knobs to tune.

## `server` — LSP

```sh
greycat-lang server [--stdio]
```

Speaks the Language Server Protocol over stdio (the only supported transport — the `--stdio` flag is a no-op accepted for compatibility). Editors with a configured `.gcl` language client get diagnostics, hover, go-to-definition, and find-references in real time.

A typical editor config points at the lang binary on PATH and associates `*.gcl`. The LSP picks the nearest `project.gcl` walking up from each opened file — see [project.md](project.md) § multi-project workspaces.

## Suppression directives

Suppress lint diagnostics with line- or range-scoped comments:

```gcl
// gcl-lint-off unused-local
var x = compute();          // lang ignores `unused-local` here
// gcl-lint-on unused-local
```

```gcl
// gcl-lint-off unused-local, possibly-null
// … block that knowingly triggers both …
// gcl-lint-on unused-local, possibly-null
```

Multi-rule suppressions take a comma-separated rule list. `// gcl-lint-on` (no rule names) re-enables every rule suppressed in scope.

Project-wide policy goes in `project.gcl` only:

```gcl
@lint_off("no-breakpoint");        // disable globally
@lint_on("possibly-null");         // force-enable an advisory rule globally
```

The `lint-pragma-outside-entrypoint` rule flags `@lint_off` / `@lint_on` in any other module — project policy belongs in the entrypoint.

There are guard rules for the suppression mechanism itself:

- **`unused-suppression`** — a `// gcl-lint-off …` that didn't actually silence anything (the underlying rule never fired in the scope).
- **`unknown-suppression-rule`** — a `// gcl-lint-off …` that names a rule the lang doesn't know.
- **`empty-suppression`** — a `// gcl-lint-off` with an empty rule list.
- **`unbalanced-lint-off`** / **`unbalanced-fmt-off`** — a `// gcl-lint-off …` / `// gcl-fmt-off` with no matching `…-on`.
- **`conflicting-lint-pragma`** — a module declaring both `@lint_on("R")` and `@lint_off("R")` for the same rule. `@lint_off` wins; the other is dead.

Audit accumulated suppression debt with `greycat-lang lint --no-suppressions`.

## Recommended pre-commit workflow

The "definition of done" for any `.gcl` change:

```sh
greycat-lang fmt --mode=check       # exit non-zero on formatting drift
greycat-lang lint                   # exit non-zero on error-severity diagnostics
greycat build                           # produce project.gcp
greycat test                            # run @test functions
```

In a pre-commit hook:

```sh
#!/usr/bin/env bash
set -euo pipefail
greycat-lang fmt --mode=check
greycat-lang lint
```

Apply formatter output directly with `greycat-lang fmt` (default `--mode=write`). Apply auto-fixable lint suggestions with `greycat-lang lint --fix`. Both are idempotent — re-running on already-clean code is a no-op.

If the lang isn't installed on a contributor's machine, install it with `cargo install greycat-lang`. There is no project-local pin equivalent to `bin/greycat`; the lang ships out-of-band with the rest of the toolchain.
