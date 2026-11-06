# Linux-port scaffolding

Read-mostly tools for the mechanical sweeps in `docs/linux-port-plan.md` (Phase 2 and 3 are
mostly "delete the branch this build never compiled" and "rename an inherited identifier"). They are
not part of the engine, the launcher or any build; nothing imports them.

Run them from the repo root. `--apply` writes; without it they only report.

## collapse_conditionals.py

Collapses a preprocessor conditional to the branch the Linux build already takes. It builds the
directive tree, evaluates each condition in three-valued logic (`True` / `False` / `Unknown`) with
the platform macros pinned to what Linux defines, and then:

- definitely false -> delete the branch, keeping an `#else` body if there is one
- definitely true  -> delete the guard lines, keeping the body
- unknown terms    -> leave the guard alone and report it, unless an operand can be dropped without
  changing the outcome (a provably-true term in an `&&`, or a provably-false term in an `||`)

Because the rule is "keep what the compiler already kept", this is behaviour-preserving by
construction. That is the whole point: it is a deletion tool, not a port tool, and it never enables
code.

    python3 scripts/port/collapse_conditionals.py --tokens _WIN32,windows.h,Win32,DStorage,IoRing \
        --apply engines/ds4/ds4.c engines/ds4/ds4_cuda.cu
    python3 scripts/port/collapse_conditionals.py \
        --tokens __APPLE__,TARGET_OS,__MACH__,GGML_COMMON_DECL_METAL,GGML_COMMON_IMPL_METAL \
        --apply engines/ds4/ds4.c
    python3 scripts/port/collapse_conditionals.py --tokens DS4_HAS_QWEN4_METAL \
        --never-defined DS4_HAS_QWEN4_METAL --apply engines/ds4/ds4.c

`--never-defined` names macros this build does not define, so a bare `#ifdef MACRO` with an `#else`
can be collapsed. Without it such a macro is unknown, and unknown means leave it alone: an earlier
version treated an unknown in-scope macro as LIVE and deleted the *live* `#else` branch, which is
how chunk 2.2c's first run turned a Linux `mm_min = 8` into the Metal `64`. Check the diff after
any collapse anyway - `git diff --stat` should show deletions only.

Anything it prints as `BLOCKED` needs a human: those were cases like `#if defined(_WIN32) /
#elif defined(POSIX_MADV_DONTNEED)`, where the live branch is the `#elif` and the guard has to be
rewritten rather than removed.

## preprocess_equiv.py

Proves a sweep changed nothing the compiler sees. `-fdirectives-only -P` expands conditionals but
leaves macros and comments, so dead branches vanish from both sides and the diff shows only real
changes. Pass `--rename-file` when the sweep was a rename, so the "before" text gets the same rename
and cancels out; anything left in the diff is then a genuine change.

    git show HEAD:engines/ds4/ds4.c > build/before.c
    python3 scripts/port/preprocess_equiv.py --cc gcc --include engines/ds4 build/before.c engines/ds4/ds4.c
    python3 scripts/port/preprocess_equiv.py --cc gcc --include engines/ds4 \
        --rename-file build/rename.json build/before.c engines/ds4/ds4.c

Pass `--include` or the before file cannot resolve its own `#include` lines and the comparison
means nothing (it reads as a few hundred live lines). Write the before file somewhere the includes
resolve from - `build/` is gitignored; `/tmp` on this box is not dependable.

A fold-to-a-constant edit is outside what this tool can show: deleting lines shifts every position,
so the diff is noise. Review the insertions instead - `git diff -U0 <file> | grep '^+[^+]'` - and
check separately that each `== DEAD` term became false and each `!= DEAD` term became true.

For `.cu` files use `--cc /opt/cuda/bin/nvcc`; nvcc fully expands macros, so `__LINE__` values
inside trace messages show up as differences and are expected to move when lines are deleted.

## code_line_diff.py

`preprocess_equiv`'s diff counts comments, so a chunk that also rewords a comment near the top of
a file shifts every later line and the diff reads as noise. This tool preprocesses both sides the
same way, drops comment-only and blank lines, and diffs with no context: a deletion sweep then
reports 0 changed code lines apart from the edits it intended, and a port reports only the regions
it rewrote. It takes the include directories and `-D` flags from a configured build's
`compile_commands.json` (default `build/llama/compile_commands.json`), keyed on the TU's basename:

    git show HEAD:engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp > build/before.cpp
    python3 scripts/port/code_line_diff.py --cuda-include /opt/cuda/include \
        --include engines/streaming-llama/prefill_llama.cpp/src build/before.cpp \
        engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp llama-weight-fetcher.cpp

Pass `--cuda-include` when the TU has `__has_include(<cuda_runtime.h>)` blocks, and `--include` for
a tool directory the file includes from. Without the build's `-D` flags the CUDA regions vanish from
both sides and the comparison proves less than it looks like it does.

## warning_profile.py

The other regression check for a mechanical sweep, scripted: compile the same TU list in two trees
with the same commands and print the warning texts with file, line and column stripped. A deletion
should differ in nothing; a rename should differ only in the renamed symbol's own warnings.

    git worktree add build/wt-base HEAD
    python3 scripts/port/warning_profile.py build/wt-base .

`--list` names the TUs; the default is the producer's six. The throwaway objects go to `build/`,
which is scratch and gitignored.

## rename_identifiers.py

Whole-token rename from a mapping file, with a prefix rule for names like `metal_graph_`:

    python3 scripts/port/rename_identifiers.py --map /tmp/rename.json --apply engines/ds4

`prefix` entries rewrite the head of the identifier and keep its tail. The word boundary matters:
`_` is a word character, so renaming `METAL_DECODE_LAYER_` cannot touch `DS4_METAL_FOO`, which is
how chunk 2.2b renamed identifiers while deliberately keeping the environment names.

## metal_word_census.py

The gate for a rename sweep of this kind, and the count of what is still left. It classifies every
case-insensitive `metal` mention by its position in the file - code, comment, string - and reports
each bucket, so a `DS4_METAL_PROFILE_X` macro cannot hide behind the same word boundary that hides
`DS4_METAL_FOO`, the environment name that must stay:

    python3 scripts/port/metal_word_census.py --strict          # exit 1 if an identifier names Metal
    python3 scripts/port/metal_word_census.py --strict --list   # names the tokens and the files

Vendored files (`external/`, `linenoise.c`) are skipped, as everywhere in this port. `ENV_NAME`
mentions are the names `envvars.json` documents; `ENV_NAME_CITED` are `DS4_*METAL*` names the table
does not document, which is what a wrapper read such as `glm_graph_env_present("A", "B")` produces -
live knobs, invisible to `envdocs`' inventory.
