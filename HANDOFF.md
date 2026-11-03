# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan with the chunk backlog: `docs/linux-port-plan.md`. Commands the user can run:
`docs/linux-port-commands.md`. Repo review: `linux-port.md`. Port scaffolding: `scripts/port/`
(its README explains each tool and when to use it).

What a chunk landed *is* in its commit message and in the plan; this file only carries what neither
records - the open decisions, the box's constraints, the verification method, and the traps.

## What the next session is for

Phase 2 is closed for code and CLI surface; what is left of it is prose. Phase 3 is open and it is
a port, not a sweep. Three things, in the order worth doing:

1. **Chunk 3.1c - `tools/dsv41-prefill/main.cpp`.** Its whole 1,163-line body sits behind
   `#if defined(_WIN32) && defined(GGML_USE_CUDA)` with a two-line refusal in the `#else`, so
   `llama-dsv41-prefill` links and prints "Offline V4.1 producer requires the opt-in Windows CUDA
   build" - `--check-native-profiles` included. Inside it: `windows.h`, `bcrypt.h` (SHA-256, used
   to bind the producer's own executable), `dxgi1_4.h` (VRAM reporting), `CreateFileW`, and the
   sweep orchestration. The guard is platform code around platform-neutral work, so the rule is
   still "keep the branch the compiler already kept" - but nothing has compiled the live branch on
   Linux before, so expect real porting inside it, not a collapse. `src/boundary-capture.cpp` needs
   the same BCrypt replacement; its Windows branch already refuses on Linux, so read that first.
2. **Chunk 3.2 - the decode library loader.** `live-session.inc` in
   `tools/dsv41-prefill/` `LoadLibraryExW`s `ds4-decode.dll` and binds the 19 `ds4x_*` entry
   points that `make cuda-decode` exports from `ds4-decode.so`. That is `dlopen`/`dlsym`, a `.so`
   path and UTF-8 paths in one file - not new wiring.
3. **Chunk 2.2e - Metal words in comments.** 310 case-insensitive `metal` mentions remain in
   `engines/ds4` outside `DS4_METAL_*`, ~190 inside strings, mostly the inert expert-locality
   profiler. No behaviour attached; regenerate the env table because lines move.

Also open and cheap: `wf_io_uring_worker` (Phase 3 item 1). `llama-weight-fetcher.cpp` already
declares a `struct io_uring ring` under `WF_HAS_IO_URING` and already defines it whenever
`<liburing.h>` exists, but the live Linux reader is still the `pread` worker pool at line ~1205.
It is measurable without model weights: the transport self-test in `engines/ds4/ds4_batch_io.cpp`
byte-compares batched io_uring reads against `pread` on a throwaway file, and the same shape works
for the producer.

## State

`git log --oneline` from `58ff877` forward is the port, chunk by chunk; `docs/linux-port-plan.md`
maps chunk numbers to commits. As of this session: `3bc184a` + `0265129` (3.1a), `dcb5666` (2.2c),
`7973c20` (3.1b), `4940201` (2.2d), plus this plan/handoff commit. `git status -sb` showed no
`[ahead N]` at session start; check it again rather than assuming.

Working tree is clean. `python3 -B scripts/release_check.py` reports 0 findings, so commits need no
`--no-verify` (`core.hooksPath` runs it as a pre-commit hook). Python suite: **584 tests, 6
failures, 5 errors** (fork baseline 585 / 6 / 6). Two process tests flip between runs
(`test_hold_yield.py`'s `test_real_owned_child_stops_on_kit11_hold` and `test_combo_single_stop.py`'s
`test_disconnect_inside_reasoning_then_next_turn_appends_only`), so compare the failing *names*
against the plan's list, not the counts. The stable survivors are Linux gaps: Windows-only guard
and producer paths, the missing `cosmos_windows` vendor module, and GPU/video gates needing a model
on disk. Chunk 5.1 and 8.1 own them.

Nothing verified so far is a performance result. No supported model weights exist on this box.

## Constraints on this box

1. A Strata serve process (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~30.0 GiB of
   32 GiB VRAM. **Its pid changes between sessions and even within one** - read it from
   `nvidia-smi --query-compute-apps=pid,used_memory`. Stopping Strata also stops this assistant, so
   GPU work is the user's to run by hand. Never kill it to test. Host-only gates work;
   `llama-server --version` initialised CUDA and printed the device list without disturbing it.
2. No supported model file exists. The GGUFs on disk are
   `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ1_0/TQ2_0/TQ3_0.gguf`
   (54.5 / 92 / 108 GiB); their metadata matches `DS4_SHAPE_QWEN4_EXP` but they are TQ-quantized
   (`type 145/146`), outside ds4's tensor type table (`ds4.c` around line 2481), so the loader
   rejects them. Phase 3's prefill gates and Phase 8 are blocked on a qualified GGUF.
3. The box changed since the plan's earlier notes: kernel 7.2.14, **93 GB RAM** (not 30), 16 cores,
   359 GB free on the NVMe. CUDA, cmake, ninja, nvcc and Python are unchanged apart from
   Python 3.14.9. Re-read `docs/linux-port-plan.md`'s machine block before trusting a limit.
4. **`/tmp` gets cleared mid-session on this box.** One run lost every scratch file, a configure
   directory and a build log. Use `build/` (gitignored) for logs, before-files and comparison
   worktrees: `git worktree add build/wt-base HEAD` worked.

## How a mechanical sweep is proven here

Deletion chunks are behaviour-preserving by construction only if the rule is "keep the branch the
compiler already kept". Use the scaffolding in `scripts/port/`:

- `collapse_conditionals.py --tokens ... [--never-defined ...]` does that collapse in three-valued
  logic. **It used to treat an in-scope macro whose value it does not know as LIVE**, which deletes
  the live `#else` branch and keeps the dead one - its first run on `ds4.c` turned the Linux
  `mm_min = 8` into the Metal `64`. Fixed in 2.2c, and `--never-defined` is how you tell it a macro
  this build does not define. Its `BLOCKED` output is the manual work.
- `preprocess_equiv.py` compares `git show HEAD:<path>` against the working file with conditionals
  resolved and macros untouched. **It used to write the before file's *path* into its temp file
  unless `--rename-file` was passed**, so every no-rename check compared garbage; fixed in 2.2c.
  Pass `--include engines/ds4`, or the before file's `#include` lines fail and you get a bogus
  457-line count. A 0-exit means the compiled code is identical.
- Warning-profile comparison: `git worktree add build/wt-base HEAD`, build the same object list in
  both trees, diff the warning texts with line numbers stripped. 2.2c: 26 vs 26 occurrences, 25
  distinct texts, none new. Renames should differ only in the renamed symbol's own warnings.
- A chunk that folds a constant (`backend == METAL` is false here) is *not* covered by
  `preprocess_equiv`: the line delta shifts everything and the positional diff is noise. Review
  `git diff -U0 <file> | grep '^+[^+]'` - the insertions - line by line instead, and check
  separately that every `== DEAD` term became false and every `!= DEAD` term became true.

When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
text that was not in the working tree (it read the diff base rather than the tree), yet its second
pass caught three real problems in what the commit *claimed*. Reviewers check claims, not just code:
re-verify the wording in docs and messages, not only the C.

## Behaviour changes landed this session

- `--metal` and `--backend metal` are gone from `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` and
  `ds4-agent`: they now read `unknown option` / `invalid backend` instead of parsing, being
  selected, and failing at engine open. Nothing in `slstack`, `scripts`, `presets`, `examples` or
  `tests` passed either spelling (checked). `--rocm` is unchanged - it is an alias for CUDA.
- `--metal-graph-test|full-test|prompt-test` became `--gpu-graph-*` and stopped forcing the backend.
  The three CUDA graph self-tests are reachable for the first time; none has been run, they need a
  model file. `--metal-graph-generate` (a tombstone that exited 2) is gone.
- `--imatrix-dataset` / `--imatrix-out` now refuse while parsing, before the model is opened. They
  used to force `DS4_BACKEND_METAL` and fail later.
- Refusal text that named Metal says "not implemented in this build": GLM-5.3 vision, V4.1 vision,
  expert profile/hotlist, imatrix collection, network tensor parallelism (`ds4_tp.c`), and the
  Qwen3.8/V4.1 support checks.
- Four subsystems are inert rather than deleted, marked `DS4_MAYBE_UNUSED` because nothing on Linux
  calls them: sequential imatrix collection, `imatrix_read_text_file`, the mixed prefill/decode
  command epoch (`ds4_sessions_eval_batch_with_prefill_gpu`), and `ds4_expert_profile_init` (the
  only thing that activated the locality profiler). `e->share_session_prefill_workspace` and the
  engine's `glm_session_count`/`glm_session_graph_bytes` bookkeeping went quiet with no message.
  All are recorded in the plan's known-gaps section.
- `build.sh`'s llama target now also builds `llama-dsv41-prefill`, and its note says the CLI is a
  Windows-guarded stub rather than claiming the target is unbuilt.

Still Windows-shaped and untouched: `slstack/single_runtime.py` stages `ds4-server.exe` and
`slstack-weight-fetcher.dll`, and `scripts/build_combo_runtime.py` builds a preset named
"RTX: ULTRA prefill -> ds4-win decode". Those are 5.1/7.1. Note for 5.1:
`engines/streaming-llama/prefill_llama.cpp/tools/slstack-weight-fetcher.cpp` is a two-line anchor
file, so the Linux `.so` exports one data symbol and carries no fetcher API - there is no DLL twin
to stage.

## Gotchas

- Regenerate the env table in the same commit as any engine source edit:
  `python3 -B -m slstack.envdocs --refresh-defaults` then `python3 -B -m slstack.envdocs`. Any line
  shift makes `test_environment_contract` error with `Environment table out of date`. Check that the
  variable *name* set is unchanged (`git show HEAD:envvars.json` vs the new file) rather than
  reading the diff; 2.2c deleted Metal-only `getenv` sites and no name dropped, because every
  `DS4_METAL_*` read removed there has another reader left. `envvars.py:inventory()` scans only
  `engines/` and `vendor/`, and treats `setenv("NAME")` as a read.
- Gate greps must be scoped: `grep -rn "..." engines/ds4` also matches build outputs, and
  `build/wt-base` now holds a second copy of every engine file. Use
  `--exclude-dir=build --include=*.c --include=*.h --include=*.cu --include=*.cuh --include=*.inc`.
  Vendored files (`engines/ds4/external/nlohmann/json.hpp`, `linenoise.c`) are left untouched on
  purpose - 3 hits there are expected, not failures.
- `_` is a word character. That protects `DS4_METAL_*` from `\bMETAL_` - which is why 2.2b kept all
  215 names - but it also *hid* the identifiers with a leading underscore (`glm_metal_*`,
  `g_metal_graph_*`, `ds4_engine_metal_graph_*`) from 2.2b's gate. 2.2d caught them with prefix
  entries long enough to start at a real boundary.
- nvcc rejects a bare `-fPIC`; use `-Xcompiler -fPIC`. A shared library needs every linked object
  `-fPIC` once it contains TLS (`__thread`).
- CUDA 13.4 / CCCL 3 facts: `nvcc` at `/opt/cuda/bin/nvcc`, libs in `/opt/cuda/lib` and
  `/opt/cuda/lib64`, `-luring` required, `sm_120` maps to `-gencode
  arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1`. `FindCUDAToolkit` creates **no
  `CUDA::cufile` target**; resolve it with
  `find_library(... cufile PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)`. CUB no longer pulls
  in libcudacxx's `<cuda/iterator>`, so `cuda::discard_iterator` / `make_strided_iterator` need
  that include explicitly. GCC 16 defaults to `-std=gnu++2x`, which changes nlohmann 3.12's
  comparison operators: `std::string == json` no longer compiles, use `.get<std::string>()`.
- `make -B <target>` rebuilds its prerequisites too, including the slow mmq objects. Use it only
  when the prerequisite list changed, and give it a long timeout. `make cpu` overwrites the CUDA
  binaries with CPU variants of the same names - rebuild CUDA afterwards.
- `make test` still cannot run: the engine Makefile asks for 10 test sources, only 8 are published.
- `scripts/release_check.py` flags any absolute path under a user's home directory as a "machine
  path" in docs; docs must use repo-relative paths.
- `mlock` of the pinned host tier needs an `RLIMIT_MEMLOCK` or cgroup allowance - still undecided
  (plan's "locked memory" risk), and it gates the decode library's host-cache path. Decide it
  before touching `llama-dsv41-locked-ram.h` in 3.1.

## Suggested skills

- `adversarial-code-review` after a deletion or rename chunk, with the `scripts/port/` equivalence
  commands handed to the reviewer as evidence (see the method section for what it caught last time).
- `add-unit-test` for extending `engines/ds4/tests/test_batch_io.cpp` or `test_decode_library.cpp`,
  and for any io_uring worker added in Phase 3.
- `code-design` / `code-reviewer` when porting the producer's `main.cpp` and its io_uring worker.
- `web-research` for liburing, CCCL 3 / libcudacxx header changes, or llama.cpp cmake details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when this session runs out of context - update this file, don't replace it with
  chat recap.
