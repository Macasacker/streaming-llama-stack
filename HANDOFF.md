# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan with the chunk backlog: `docs/linux-port-plan.md`. Commands the user can run:
`docs/linux-port-commands.md`. Repo review: `linux-port.md`. Port scaffolding: `scripts/port/`
(its README explains each tool and when to use it).

What a chunk landed *is* in its commit message and in the plan; this file only carries what neither
records - the open decisions, the box's constraints, the verification method, and the traps.

## What the next session is for

Phase 2 is nearly done: `engines/ds4` has no Win32 host code, no Apple conditional and no
Metal-named identifier. Two things are open, and the first needs an owner decision before work:

1. **Chunk 2.2c - the Metal backend itself.** `--metal`, `--backend metal`, `--metal-graph-test`,
   `--metal-graph-full-test`, `--metal-graph-prompt-test` and `--rocm` are still accepted; 46 sites
   reference `DS4_BACKEND_METAL`. `--metal` parses, gets selected, then fails at engine open.
   Nothing in `slstack`, `scripts` or the presets passes it, so deleting the backend is safe here -
   but it changes observable behaviour (unknown flag instead of accepted-then-failed), which is why
   it was left. Alternative: refuse it in the parser with a message and leave the dead branches for
   chunk 6.1. Ask, then do it.
2. **Chunk 3.1 - the Linux producer** (`engines/streaming-llama`), then 3.2. Inventory below.

Everything verified so far is a compile or host-code gate. Nothing is a performance result, because
no supported model weights exist on this box.

## State

`git log --oneline` from `58ff877` forward is the port, chunk by chunk; `docs/linux-port-plan.md`
maps chunk numbers to commits. As of the last session: `171319a` (2.1), `7cc2ca3` (2.2a),
`3a49385` (2.1 follow-up), `535beef` (2.2b), plus plan/handoff commits. Two of them were unpushed;
check `git status -sb` for `[ahead N]` and say so rather than assuming.

`python3 -B scripts/release_check.py` reports 0 findings, so commits need no `--no-verify`
(`core.hooksPath` runs it as a pre-commit hook). Python suite: **584 tests, 5-6 failures, 5 errors**
(fork baseline 585 / 6 / 6). Two process tests flip between runs (`test_hold_yield.py`'s
`test_real_owned_child_stops_on_kit11_hold` and `test_combo_single_stop.py`'s
`test_disconnect_inside_reasoning_then_next_turn_appends_only`), so compare the failing *names*
against the plan's list, not the counts.

Working tree should be clean at session start. Build outputs (`*.o`, `*.so`, engine binaries) are
gitignored.

## How a mechanical sweep is proven here

Deletion chunks are behaviour-preserving by construction only if the rule is "keep the branch the
compiler already kept". Use the scaffolding in `scripts/port/`:

- `collapse_conditionals.py --tokens ...` does that collapse in three-valued logic and reports
  anything it cannot decide instead of guessing. Its `BLOCKED` output is the manual work: usually a
  `#if DEAD / #elif LIVE` chain where the guard must be rewritten, not removed.
- `preprocess_equiv.py` compares `git show HEAD:<path>` against the working file with conditionals
  resolved and macros untouched, so dead branches vanish on both sides and the diff shows only real
  changes. Pass `--rename-file` for a rename sweep so the rename cancels itself out; a 0-exit means
  the compiled code is identical.
- Warning-profile comparison: `git worktree add /tmp/base HEAD`, build the same object list in both
  trees, diff the warning texts with line numbers stripped. Cheapest honest regression check for a
  rename; chunk 2.2b differed only in the renamed symbol's own warning.

When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
text that was not in the working tree (it read the diff base rather than the tree), yet its second
pass caught three real problems in what the commit *claimed*: combo handoff described as still
honouring `/decode/shutdown` when nothing sets `ds4_combo_active`, expert-cache restore/snapshot
described as refusing when nothing calls them, and a fence banner implying live wave ordering that
`cuda_stream_selected_ds_upload()` stubs out. Reviewers check claims, not just code - re-verify the
wording in docs and messages, not only the C.

## Constraints on this box

1. A Strata serve process (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~29.9 GiB of 32 GiB
   VRAM, leaving ~1.3 GiB. **Its pid changes between sessions** - read it from
   `nvidia-smi --query-compute-apps=pid,used_memory`. Stopping Strata also stops this assistant, so
   GPU work is the user's to run by hand. Never kill it to test. Host-only gates work because the
   transport now creates its CUDA upload stream lazily.
2. No supported model file exists. The GGUFs on disk are
   `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ1_0/TQ2_0/TQ3_0.gguf`
   (54.5 / 92 / 108 GiB); their metadata matches `DS4_SHAPE_QWEN4_EXP` but they are TQ-quantized
   (`type 145/146`), outside ds4's tensor type table (`ds4.c` around line 2481), so the loader
   rejects them. Phase 3's prefill gates and Phase 8 are blocked on a qualified GGUF.

## Producer inventory (Phase 3), re-measured

- `engines/streaming-llama` has **680** Windows-flavoured sites and **2,372** Apple/Metal sites in
  its sources (`--include=*.cpp --include=*.c --include=*.h --include=*.hpp --include=*.inl`).
- There is **no `build/` directory**: the producer has never been built on Linux. Getting a cmake
  build to configure and link is chunk 3.1's first task, before any porting.
- Chunk 3.2 is smaller than the plan's wording suggests. The producer-side decode-library loader is
  already written, in `prefill_llama.cpp/tools/dsv41-prefill/live-session.inc` (880 lines, 35 Win32
  sites): it `LoadLibraryExW`s `ds4-decode.dll` beside the executable and `bind`s `ds4x_open`,
  `ds4x_model_map`, `ds4x_register_weight`, `ds4x_bind_state`, `ds4x_set_expert_hook`,
  `ds4x_decode`, `ds4x_close`, `ds4x_set_affinity`, `ds4x_host_cache_v1/v2`, `ds4x_host_cpu_v1`,
  `ds4x_host_timing_v1` and (when enabled) `ds4x_prefill_workspace` / `ds4x_prefill`. Chunk 1.6's
  `ds4-decode.so` exports exactly those 19 names. So 3.2 is `dlopen`/`dlsym`, a `.so` path, and
  UTF-8 paths in that one file - not new wiring.
- Without `ds4-decode.so` loaded the producer decodes through its own ggml CUDA path
  (`ggml-ds4-native-half`), which is why the engine library has never been exercised end to end.

## Behaviour changes worth re-checking

- `slstack/engines.py:combo_enabled()` no longer requires `os.name == 'nt'`, so `serving=native`
  reaches the combo lifecycle on Linux, where the engine refuses it with
  "combo handoff is not implemented in this build". Its process admission is still
  `slstack/guard.py`'s Windows Job Object branch (chunk 5.1 replaces it with a cgroup v2 scope). If
  native serving misbehaves on Linux, that is the likely cause, not the refusal.
- `slstack/lease.py:active_holds()` has no `mac=` namespace any more, so MAC KIT11 lines are not
  scanned as RTX blockers. `tests/test_hold_yield.py` was rewritten for RTX-only semantics.
- `slstack/router.py` no longer annotates routes: rows say `single-engine` or `ds4-native`, and
  `ultra_handoff` is always false.
- Still Windows-shaped and untouched: `slstack/single_runtime.py` stages `ds4-server.exe` and
  `slstack-weight-fetcher.dll`, and `scripts/build_combo_runtime.py` builds a preset named
  "RTX: ULTRA prefill -> ds4-win decode". Those are 5.1/7.1.
- Engine capabilities the deleted Windows branches used to implement: combo handoff, background
  affinity promotion and output LoRA now refuse with a message naming the build; expert-cache
  restore and snapshot are **inert** (no Linux caller), so `DS4_COMBO_EXPERT_RESTORE_FILE` and
  `DS4_COMBO_EXPERT_SNAPSHOT_FILE`, which `slstack/combo_native.py` and `slstack/engines.py` set
  for a combo run, do nothing. `--ds-read-parts` is parsed into
  `ds4_engine_options.ds_read_parts` and then refused; `DS4_DS_READ_PARTS` is the live path. Each of
  these is listed in the plan's known-gaps section.

## Gotchas

- Regenerate the env table in the same commit as any engine source edit:
  `python3 -B -m slstack.envdocs --refresh-defaults` then `python3 -B -m slstack.envdocs`. Any line
  shift makes `test_environment_contract` error with `Environment table out of date`. Deletion
  chunks churn thousands of rows - check that the variable *name* set is unchanged
  (`git show HEAD:envvars.json` vs the new file) rather than reading the diff.
  `envvars.py:inventory()` scans only `engines/` and `vendor/`, and treats `setenv("NAME")` in a
  source file as a read.
- Gate greps must be scoped: `grep -rn "_WIN32\|windows.h\|..." engines/ds4` also matches build
  outputs, and vendored files (`engines/ds4/external/nlohmann/json.hpp`, `linenoise.c`) are left
  untouched on purpose - 3 hits there are expected, not failures.
- `_` is a word character, so `\bMETAL_` cannot match inside `DS4_METAL_FOO`. That is what lets
  chunk 2.2b rename identifiers while keeping the 215 `DS4_METAL_*` environment names, which the
  owner chose to keep because they are documented in `docs/VARIABLES.md` and named in
  `docs/macdecode-r6/r6-controls.json` and `docs/MACDECODE_ROUND6.md`.
- nvcc rejects a bare `-fPIC`; use `-Xcompiler -fPIC`. A shared library needs every linked object
  `-fPIC` once it contains TLS (`__thread`).
- `make -B <target>` rebuilds its prerequisites too, including the slow mmq objects. Use it only
  when the prerequisite list changed, and give it a long timeout. `make cpu` overwrites the CUDA
  binaries with CPU variants of the same names - rebuild CUDA afterwards.
- `make test` still cannot run: the engine Makefile asks for 10 test sources, only 8 are published.
- `scripts/release_check.py` flags any absolute path under a user's home directory as a "machine
  path" in docs; docs must use repo-relative paths.
- CUDA facts: `nvcc` at `/opt/cuda/bin/nvcc`, libs in `/opt/cuda/lib` and `/opt/cuda/lib64`,
  `-luring` required, `sm_120` maps to `-gencode arch=compute_120a,code=sm_120a
  -DDS4_CUDA_HAVE_MXF4=1`.
- `mlock` of the pinned host tier needs an `RLIMIT_MEMLOCK` or cgroup allowance - still undecided
  (plan's "locked memory" risk), and it gates the decode library's host-cache path.

## Suggested skills

- `adversarial-code-review` after a deletion or rename chunk, with the `scripts/port/` equivalence
  commands handed to the reviewer as evidence (see the method section for what it caught last time).
- `add-unit-test` for extending `engines/ds4/tests/test_batch_io.cpp` or `test_decode_library.cpp`.
- `code-design` / `code-reviewer` when porting the producer's io_uring worker in 3.1.
- `web-research` for liburing, `sm_120a` or llama.cpp cmake details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when this session runs out of context - update this file, don't replace it with
  chat recap.
