# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan with the chunk backlog: `docs/linux-port-plan.md`. Commands the user can run:
`docs/linux-port-commands.md`. Repo review: `linux-port.md`. Port scaffolding: `scripts/port/`
(its README explains each tool and when to use it).

What a chunk landed *is* in its commit message and in the plan; this file only carries what neither
records - the open decisions, the box's constraints, the verification method, and the traps.

## What the next session is for

Phase 3's producer CLI and its weight fetcher are now Linux: the CLI compiles, links, runs, and
the fetcher reads through io_uring rings with no Win32 code in its file. What remains of Phase 3 is
deeper in the tree, and it is a port, not a sweep. In the order worth doing:

1. **Chunk 3.1f - the weight provider, and then the prefill works.** `src/llama-dsv41-weights.cpp`
   refuses on Linux at eight sites (`:632`, `:924`, `:1001`, `:1164`, `:1283`, `:1323`, `:1399`,
   `:1517`): the physical file lease (`implementation::identity()`), `inspect`, `create_pipeline`,
   the three early/whole-prompt Engram IoRing loaders, the expert cache and the RAM host cache.
   These are what still stop a prefill after the CLI itself runs. `create_pipeline` asks the
   fetcher for `wf_create_required_dstorage`, which refuses here, and the provider's extent paths
   call `wf_fetch_extents_direct` / `wf_ready_direct_bank`, whose Linux arms `return false` - so
   3.1f decides whether those become io_uring reads against the fetcher's ring or go away. Note
   that `params.ds_submit_threads` has no reader in the fetcher any more, so `--ds-submit-threads`
   is admission-only. `llama-dsv41-locked-ram.h` and the carry lease in `llama-dsv41-carry.cpp` are
   the same family, and they need the locked-memory decision below before they can be written.
2. **Chunk 3.1g - `llama-dsv41-engram-prefetch.cpp` (Phase 3 item 3).** io_uring with 1 MiB extents
   at QD64, measured against the `pread` baseline; that change is what took 76.2 s to 43.2 s on the
   Windows machine. `LLAMA_WF_IO_URING=0` now selects the fetcher's pread baseline, which is the
   comparison Phase 8 item 3 asks for.
3. **Chunk 2.2e - Metal words in comments.** 310 case-insensitive `metal` mentions remain in
   `engines/ds4` outside `DS4_METAL_*`, ~190 inside strings, mostly the inert expert-locality
   profiler. No behaviour attached; regenerate the env table because lines move.

Also open: 5.1 (Python orchestrator: `guard.py` Job Object -> a cgroup scope,
`single_runtime.py` PRODUCER_FILES), 7.1 (docs/presets/tests), 4.1 (ggml backend prune), and
8.1 (qualification), all blocked on a qualified GGUF for the last one.

Chunk 3.2 (the decode library loader) landed inside 3.1c, because `live-session.inc` is part of
`main.cpp`'s body: `dlopen`/`dlsym` of `ds4-decode.so` beside the executable. Its binding is
verified by the selftest; a live session against weights is not, and cannot be here.

## State

`git log --oneline` from `58ff877` forward is the port, chunk by chunk; `docs/linux-port-plan.md`
maps chunk numbers to commits. As of this session: `3bc184a` + `0265129` (3.1a), `dcb5666` (2.2c),
`7973c20` (3.1b), `4940201` (2.2d), `3fc4f64` (3.1c), `6fc8cac` (3.1d), `0ffe3ce` + `ff6bb44`
(3.1e), plus the plan/handoff commits. `git status -sb` shows no `[ahead N]` after pushing; check
it rather than assuming.

Working tree is clean apart from scratch under `build/`. `python3 -B scripts/release_check.py`
reports 0 findings, so commits need no `--no-verify` (`core.hooksPath` runs it as a pre-commit
hook). Python suite: **584 tests, 6 failures, 5 errors** (fork baseline 585 / 6 / 6), and the
failing *names* are the plan's list - `test_combo_process` (3), `test_journal_recovery_flags`,
`test_single_liveness`, `test_shared_guard_adapter` (Windows-only guard and producer paths),
`test_streamed_recipe_argv`, `test_dsv41_recipe`, `test_yoco` (missing `cosmos_windows` module),
`test_h3_video_delivery` (video gate). Process tests flip between runs - the parent commit's full
run showed 7 failures / 6 errors, and the two extra names
(`test_actual_shared_guard_executes_the_nested_command`,
`test_native_startup_accepts_lora_before_help_without_loading_weights`) pass when their file is run
alone. Compare the set of names, never the counts. Chunk 5.1 and 8.1 own the survivors.

Nothing verified so far is a performance result for a model. The only measurement is the fetcher's
host read comparison (`docs/evidence/wf-io-selftest.cpp`: io_uring 24 ms vs pread 32 ms for 64 MiB
across 218 ops, four threads at QD8, one NVMe, cold cache) - it isolates the reader, not a prefill.
No supported model weights exist on this box.

## Constraints on this box

1. A Strata serve process (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~30.0 GiB of
   32 GiB VRAM. **Its pid changes between sessions and even within one** - read it from
   `nvidia-smi --query-compute-apps=pid,used_memory`. Stopping Strata also stops this assistant, so
   GPU work is the user's to run by hand. Never kill it to test. Host-only gates work;
   `llama-server --version` and the producer's CUDA-enumeration path both initialised a context
   (~160 MB) without disturbing it, and NVML queries (`nvmlDeviceGetMemoryInfo`) are read-only.
   Anything that allocates device memory is still the user's call.
2. No supported model file exists. The GGUFs on disk are
   `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ1_0/TQ2_0/TQ3_0.gguf`
   (54.5 / 92 / 108 GiB); their metadata matches `DS4_SHAPE_QWEN4_EXP` but they are TQ-quantized
   (`type 145/146`), outside ds4's tensor type table (`ds4.c` around line 2481), so the loader
   rejects them. Phase 3's prefill gates and Phase 8 are blocked on a qualified GGUF.
3. The box changed since the plan's earlier notes: kernel 7.2.14, **93 GB RAM** (not 30), 16 cores,
   359 GB free on the NVMe. CUDA, cmake, ninja, nvcc and Python are unchanged apart from
   Python 3.14.9. Re-read `docs/linux-port-plan.md`'s machine block before trusting a limit.
4. **`/tmp` gets cleared mid-session on this box.** One run lost every scratch file, a configure
   directory and a build log. Use `build/` (gitignored) for logs, before-files, scratch programs and
   comparison worktrees: `git worktree add build/wt-base HEAD` worked. `TMPDIR=$PWD/build/...` also
   fixes the scaffolding tools, which write temp files into `/tmp`.

## How a mechanical sweep is proven here

Deletion chunks are behaviour-preserving by construction only if the rule is "keep the branch the
compiler already kept". Use the scaffolding in `scripts/port/`:

- `collapse_conditionals.py --tokens ... [--never-defined ...]` does that collapse in three-valued
  logic. **It used to treat an in-scope macro whose value it does not know as LIVE**, which deletes
  the live `#else` branch and keeps the dead one - its first run on `ds4.c` turned the Linux
  `mm_min = 8` into the Metal `64`. Fixed in 2.2c, and `--never-defined` is how you tell it a macro
  this build does not define. Its `BLOCKED` output is the manual work. Chunk 3.1d ran it with
  `--tokens _WIN32,windows.h,Win32,DStorage,IoRing,DXGI,BCrypt,NOMINMAX` over the producer CLI files
  and reported 0 blocked, 368 deletions, 0 insertions.
  **A macro listed in `--never-defined` must also be listed in `--tokens`.** A node is only
  considered when its first condition names a token in scope, so 3.1e's first run pinned
  `WF_HAS_DSTORAGE` false but left 2,146 dead lines untouched because that name was never in scope.
- `preprocess_equiv.py` compares `git show HEAD:<path>` against the working file with conditionals
  resolved and macros untouched. **It used to write the before file's *path* into its temp file
  unless `--rename-file` was passed**, so every no-rename check compared garbage; fixed in 2.2c.
  Pass `--include`, or the before file's `#include` lines fail and you get a bogus count. Two more
  limits found in 3.1d: `--include` takes **one** directory (use `--extra "-I<dir> -I<dir>"` for
  several), and `lang` is inferred from the suffix, so `.h` files are preprocessed as C - pass
  `--cc g++ --extra "... -x c++"` or every `<filesystem>`/`<atomic>` include fails.
- A chunk that also edits comments (3.1d reworded provenance comments) makes the positional diff
  noise: one changed line near the top shifts everything. `scripts/port/code_line_diff.py` compares
  the **code** lines only - it preprocesses both sides with the build's own `-I` and `-D` flags,
  drops comment-only and blank lines, and diffs with no context. 3.1e used it to show 47 / 11 / 4 / 7
  changed lines in the four files it touched, every one an intended fold or a dead field, and 0 for
  a pure deletion. For a file whose own include block moved (producer-platform.inc), compare the
  *multiset* of code lines instead of positions.
- Warning-profile comparison: `git worktree add build/wt-base HEAD` then
  `scripts/port/warning_profile.py build/wt-base .`, which compiles the same TU list in both trees
  with the same commands and prints the texts with line numbers stripped. 2.2c: 26 vs 26 occurrences
  in the engine; 3.1e: 27 vs 27, 18 distinct, none added or removed. Renames should differ only in
  the renamed symbol's own warnings.
- A chunk that folds a constant (`backend == METAL` is false here) is *not* covered by
  `preprocess_equiv`: the line delta shifts everything and the positional diff is noise. Review
  `git diff -U0 <file> | grep '^+[^+]'` - the insertions - line by line instead, and check
  separately that every `== DEAD` term became false and every `!= DEAD` term became true.

- Sweep inputs a later session may need to re-run are committed under `docs/evidence/` -
  `rename-2.2d.json` is chunk 2.2d's rename map, `platform-selftest.cpp` is chunk 3.1c's
  platform-layer program, and `wf-io-selftest.cpp` is chunk 3.1e's reader program (io_uring against
  the pread baseline, byte-compared through the real `wf_*` pipeline). Copy them to `build/` and
  compile them; `docs/linux-port-commands.md` has the exact `g++` lines and the no-model producer
  smoke recipe. Anything written under `build/` is scratch and is not in git; chunk 2.2b's map was
  never saved, so its rename is only reconstructable from its commit message.
- When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
  but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
  text that was not in the working tree (it read the diff base rather than the tree), yet its second
  pass caught three real problems in what the commit *claimed*. Reviewers check claims, not just
  code: re-verify the wording in docs and messages, not only the C. Note that a subagent can be
  stopped mid-run and return no output; if a survey agent dies, do the greps yourself.

## Behaviour changes landed this session

Chunk 3.1c/3.1d's behaviour changes (the CLI running on Linux, the POSIX receipt identity fields,
`--carry-storage hybrid` without a drive letter, SIGINT/SIGTERM cancellation, the live carry-resume
RAM floor, no process-memory limit in `host_memory::begin()`, the two `LLAMADECODE_*` diagnostic
lines) are recorded in the plan's own entries for those chunks. This session's:

- The weight fetcher reads with **io_uring**: one ring per reader thread, `io_queue_depth` reads in
  flight each, registered fds where the kernel accepts them. The `pread` workers remain as the
  baseline and run whenever `io_uring_queue_init` fails. `wf_get_io_engine()` (new export) names
  which one a fetcher started, and `LLAMA_WF_IO_URING=0` forces the baseline - the one new
  documented env row, added so Phase 8 item 3's isolated comparison is possible.
- `WF_MODE_AUTO` no longer picks GDS because `cufile.h` exists. It calls `cuFileDriverOpen()` while
  resolving the mode and prefers GDS only if the driver reports support; on this box (no
  `nvidia-fs`) AUTO now resolves to `DIRECT_IO` at once instead of choosing GDS and falling back at
  pipeline start.
- `io_direct` no longer opens `O_DIRECT`, and says so once per file. Linux requires offset, length
  and buffer to be block-aligned; GGUF tensor offsets are multiples of `general.alignment` (32 by
  default), so the chunked reads would have failed `EINVAL` and completed with **zero bytes** while
  still counting as done - silently wrong weights. Reads are buffered. The aligned variant (a second
  fd, 4 KiB-aligned offsets, a bounce for the unaligned head) is open follow-up work, the same
  deferral Phase 1 recorded for the engine transport.
- `--ds-gpu-ring-slots` is gone: it is now an unknown producer option (parse refusal), the receipt
  no longer reports `ds_gpu_ring_slots`, and `wf_params` lost that field. Anything that mirrors
  `wf_params` by field order is stale - notably
  `vendor/engines/cosmos_windows/weight_streaming/ds_backend.py`, which Phase 6 deletes anyway.
- The fetcher's strict extent-fetch entry points (`wf_fetch_extents_direct`, `wf_ready_direct_bank`,
  `wf_fetch_ready_extents`, `wf_wait_ready_extents`, `wf_create_required_dstorage`) keep their Linux
  refusals; `wf_persist_shutdown()` still returns 1. Chunk 3.1f decides what they become.
- `liburing` is now linked (`find_library(LLAMA_URING_LIBRARY uring)` into `llama`, the producer and
  the fetcher `.so`), because `WF_HAS_IO_URING` is decided by `__has_include(<liburing.h>)` and
  nothing used to link the library.
- Still Windows-shaped and untouched: `slstack/single_runtime.py` stages `ds4-server.exe` and
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
  reading the diff; 2.2c deleted Metal-only `getenv` sites and no name dropped. Moving a `getenv`
  between files (3.1c moved several into `producer-platform.inc`) moves rows but not names.
  `envvars.py:inventory()` scans only `engines/` and `vendor/`, and treats `setenv("NAME")` as a
  read - a `setenv` whose name is a variable is invisible to it.
- Gate greps must be scoped: `grep -rn "..." engines/ds4` also matches build outputs, engine
  binaries, `*.o`, `ds4-decode.so` and `__pycache__`, and `build/wt-base` holds a second copy of
  every engine file. Use `--exclude-dir=build --exclude-dir=.git` plus
  `--include=*.c --include=*.h --include=*.cpp --include=*.inc`. Vendored files
  (`engines/ds4/external/nlohmann/json.hpp`, `linenoise.c`) are left untouched on purpose - 3 hits
  there are expected, not failures.
- `_` is a word character. That protects `DS4_METAL_*` from `\bMETAL_` - which is why 2.2b kept all
  215 names - but it also *hid* the identifiers with a leading underscore (`glm_metal_*`,
  `g_metal_graph_*`, `ds4_engine_metal_graph_*`) from 2.2b's gate. 2.2d caught them with prefix
  entries long enough to start at a real boundary.
- **A system header must not be first included inside a namespace.** `struct stat`, `struct
  pollfd` and friends then become members of that namespace and every later `struct stat` at file
  scope is a different, incomplete type. `producer-wire.h` and `producer-platform.inc` include
  `<sys/stat.h>`, `<poll.h>` etc. at global scope for exactly this reason.
- In a class whose methods are named `open`/`read`/`write`/`close`/`flush` (here: `native_file`,
  `exclusive_writer`), call the C functions as `::open`, `::read`, `::write`, `::close` - a bare
  `open(...)` inside the class resolves to the member and recurses.
- `producer-platform.inc` is not a header: it is included once, by `main.cpp`, *after* `require()`,
  `stop()`, `add()` and the bounded constants exist and *before* `live-session.inc`. `stop()` is
  forward-declared (`namespace { void stop(); }`) because `native_file` calls it; defining it before
  the include instead would leave the file's classes referring to a name declared later.
- `renameat2(RENAME_NOREPLACE)` works on ext4 here and fails `EEXIST` on an occupied target, which is
  what `ReplaceIfExists=FALSE` meant. Constants come from `<linux/fs.h>`; filesystem magics from
  `<linux/magic.h>` (this kernel's header has no ZFS/F2FS/9P/CIFS entry - those literals are inline
  in the refusal list).
- `libnvidia-ml.so.1` is present (`/usr/lib`) and is `dlopen`ed, not linked. NVML only answers when
  `CUDA_VISIBLE_DEVICES` holds a `GPU-...` UUID; with it unset the free/total reads return false and
  the L2 offer is 0, exactly as on Windows.
- The fetcher's reader has two rules that are easy to break again. A chunk taken off the queue and
  never submitted never completes, and the pipeline waits on that layer's completion counter - so a
  full SQ ring puts the remainder back on the queue, and `wf_stop` drains the queue and the
  in-flight reads before joining. And `IOSQE_FIXED_FILE` means the SQE's `fd` field is the
  *registered index*, not a descriptor; registration is optional, so both paths must stay wired.
- `WF_HAS_IO_URING` comes from `__has_include(<liburing.h>)`, so liburing must be on the link line
  (`LLAMA_URING_LIBRARY`, resolved in the top-level `CMakeLists.txt` and linked into `llama`, the
  producer and the fetcher `.so`). A header-only install would fail at link, not at run time.
- A **new** env read is not enough for the docs: `envdocs --refresh-defaults` only fills evidence for
  rows that exist. Add the row to `envvars.json` by hand first (keys `var`, `flag`, `default`,
  `code_default`, `classification`, `scopes`, `doc`, `reads`), then refresh, then
  `python3 -B -m slstack.envdocs --check`. Without the row it raises
  `Environment reference has missing or invalid rows`. `LLAMA_WF_IO_URING` is the model.
- `git commit --amend` only reaches HEAD. A wrong number in an earlier chunk's message is worth
  fixing - stash, then `GIT_SEQUENCE_EDITOR="sed -i '1s/^pick /reword /'" GIT_EDITOR=<script that
  copies the corrected message to $1> git rebase -i HEAD~2`. The trees must match afterwards
  (`git diff --stat <old-head> <new-head>` should be empty).
- nvcc rejects a bare `-fPIC`; use `-Xcompiler -fPIC`. A shared library needs every linked object
  `-fPIC` once it contains TLS (`__thread`).
- CUDA 13.4 / CCCL 3 facts: `nvcc` at `/opt/cuda/bin/nvcc`, libs in `/opt/cuda/lib` and
  `/opt/cuda/lib64`, `-luring` required, `sm_120` maps to `-gencode
  arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1`. `FindCUDAToolkit` creates **no
  `CUDA::cufile` target**; resolve it with
  `find_library(... cufile PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)`. CUB no longer pulls
  in libcudacxx's `<cuda/iterator>`, so `cuda::discard_iterator` / `make_strided_iterator` need that
  include explicitly. GCC 16 defaults to `-std=gnu++2x`, which changes nlohmann 3.12's comparison
  operators: `std::string == json` no longer compiles, use `.get<std::string>()`. GCC 16 also
  deprecates `std::filesystem::u8path` and returns `std::u8string` (`basic_string<char8_t>`) from
  `path::u8string()`, which `nlohmann::json::at()` will not accept - use `producer_text::native_path`
  and `path_text`.
- `make -B <target>` rebuilds its prerequisites too, including the slow mmq objects. Use it only
  when the prerequisite list changed, and give it a long timeout. `make cpu` overwrites the CUDA
  binaries with CPU variants of the same names - rebuild CUDA afterwards.
- `make test` still cannot run: the engine Makefile asks for 10 test sources, only 8 are published.
  The producer CLI has no test target at all; that is why `docs/evidence/platform-selftest.cpp` is
  compiled by hand.
- `scripts/release_check.py` flags any absolute path under a user's home directory as a "machine
  path", and it scans every tracked blob - including a `.cpp` committed under `docs/evidence/` - so
  docs and evidence files must use repo-relative paths. `git status -sb` is the push check; the
  pre-commit hook also scans the commit message, so a message must not quote a machine path either.
- `mlock` of the pinned host tier needs an `RLIMIT_MEMLOCK` or cgroup allowance - still undecided
  (plan's "locked memory" risk). It gates `llama-dsv41-locked-ram.h` (chunk 3.1f) and the decode
  library's host-cache path, and `host_memory.begin()` deliberately does not solve it.

## Suggested skills

- `adversarial-code-review` after a deletion or rename chunk, with the `scripts/port/` equivalence
  commands handed to the reviewer as evidence (see the method section for what it caught last time).
- `add-unit-test` for extending `engines/ds4/tests/test_batch_io.cpp` or `test_decode_library.cpp`.
  The producer's reader is covered by `docs/evidence/wf-io-selftest.cpp`, compiled by hand, because
  the producer has no test target.
- `code-design` / `code-reviewer` when porting the weight provider lease and the carry lease
  (chunk 3.1f), and `adversarial-code-review` after it: that chunk changes admission, not just
  text.
- `web-research` for liburing, CCCL 3 / libcudacxx header changes, or llama.cpp cmake details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when this session runs out of context - update this file, don't replace it with
  chat recap.
