# Linux-port scaffolding

Three read-mostly tools for the mechanical sweeps in `docs/linux-port-plan.md` (Phase 2 and 3 are
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

Chunk 2.1 used the first form, 2.2a the second. Anything it prints as `BLOCKED` needs a human: those
were cases like `#if defined(_WIN32) / #elif defined(POSIX_MADV_DONTNEED)`, where the live branch is
the `#elif` and the guard has to be rewritten rather than removed.

## preprocess_equiv.py

Proves a sweep changed nothing the compiler sees. `-fdirectives-only -P` expands conditionals but
leaves macros and comments, so dead branches vanish from both sides and the diff shows only real
changes. Pass `--rename-file` when the sweep was a rename, so the "before" text gets the same rename
and cancels out; anything left in the diff is then a genuine change.

    git show HEAD:engines/ds4/ds4.c > /tmp/before.c
    python3 scripts/port/preprocess_equiv.py --cc gcc /tmp/before.c engines/ds4/ds4.c
    python3 scripts/port/preprocess_equiv.py --cc gcc --rename-file /tmp/rename.json \
        /tmp/before.c engines/ds4/ds4.c

For `.cu` files use `--cc /opt/cuda/bin/nvcc`; nvcc fully expands macros, so `__LINE__` values
inside trace messages show up as differences and are expected to move when lines are deleted.

The other regression check for a mechanical sweep is a warning profile: `git worktree add /tmp/base
HEAD`, build the same object list in both trees, and compare the warning texts with the line numbers
stripped. A rename should differ only in the renamed symbol's own warnings.

## rename_identifiers.py

Whole-token rename from a mapping file, with a prefix rule for names like `metal_graph_`:

    python3 scripts/port/rename_identifiers.py --map /tmp/rename.json --apply engines/ds4

`prefix` entries rewrite the head of the identifier and keep its tail. The word boundary matters:
`_` is a word character, so renaming `METAL_DECODE_LAYER_` cannot touch `DS4_METAL_FOO`, which is
how chunk 2.2b renamed identifiers while deliberately keeping the environment names.
