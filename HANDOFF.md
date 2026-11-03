# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan with the chunk backlog: `docs/linux-port-plan.md`. Commands the user can run:
`docs/linux-port-commands.md`. Repo review: `linux-port.md`. Port scaffolding: `scripts/port/`
(its README explains each tool and when to use it).

What a chunk landed *is* in its commit message and in the plan; this file only carries what neither
records - the open decisions, the box's constraints, the verification method, and the traps.

## What the next session is for

Phase 3's producer CLI, its weight fetcher and its native weight provider are now Linux: the
provider holds a POSIX lease on the verified model file, builds the bounded relay, acquires stages
in alternating banks, writes named file extents into a compute bank, and admits its RAM expert tier
from `/proc`. A streamed prefill still needs the Engram row loader, and the gate itself needs a
qualified GGUF. In the order worth doing:

1. **Chunk 3.1g - `llama-dsv41-engram-prefetch.cpp` (Phase 3 item 3), and with it the provider's
   three Engram arms.** All 630 lines of the implementation, including the platform-neutral row
   planner (hash to row ids, extents, the admission arena, the prior-cache row copy), sit inside
   `#ifdef _WIN32`: IoRing rings at QD64 over 1 MiB extents, registered files and buffers, a slot
   table, table-0-before-table-1 publication, and cancellation. io_uring measured against the
   `pread` baseline is Phase 8 item 3's comparison, and the 76.2 s to 43.2 s figure came from this
   change on the Windows machine. `load_early()` and `load()` take a `const wchar_t*` and a Windows
   `HANDLE`; the provider's lease holds an `int fd` and a `std::filesystem::path`, so change the
   signature in this file rather than adapting at the call site. Nothing else in the Engram family
   needs porting - `llama-dsv41-engram.{h,cpp}`, `llama-dsv41-engram-row-cache.h` and
   `llama-dsv41-engram-early-decision.h` are already Win32-free - so opening `launch_early`,
   `start_engram_prompt_early` and the synchronous load in `prepare_engram_prompt` is the last
   thing between this tree and a prefill attempt.
2. **Chunk 3.1h - the locked-memory decision, then `llama-dsv41-locked-ram.h` and the carry lease in
   `llama-dsv41-carry.cpp`** (Phase 3 item 5). They are the same family, and neither can be written
   before the decision below: `mlock` of a pinned tier needs an `RLIMIT_MEMLOCK` or a cgroup
   allowance, and `host_memory::begin()` deliberately does not solve it.
4. **Chunk 2.2e - Metal words in comments.** 310 case-insensitive `metal` mentions remain in
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
`7973c20` (3.1b), `4940201` (2.2d), `3fc4f64` (3.1c), `6fc8cac` (3.1d), `0ffe3ce` + `ff6bb44` +
`1325655` (3.1e), `3afcd91` + `1e4cf01` (3.1f), plus the plan/handoff commits. `git status -sb`
shows no `[ahead N]` after pushing; check it rather than assuming.

Working tree is clean apart from scratch under `build/`. `python3 -B scripts/release_check.py`
reports 0 findings, so commits need no `--no-verify` (`core.hooksPath` runs it as a pre-commit
hook). Python suite: **584 tests, 5-6 failures, 5 errors** (fork baseline 585 / 6 / 6), and the
failing *names* are the plan's list - `test_combo_process` (3), `test_journal_recovery_flags`,
`test_single_liveness`, `test_shared_guard_adapter` (Windows-only guard and producer paths),
`test_streamed_recipe_argv`, `test_dsv41_recipe`, `test_yoco` (missing `cosmos_windows` module),
`test_h3_video_delivery` (video gate). Process tests flip between runs - `test_combo_single_stop`
and `test_real_owned_child_stops_on_kit11_hold` pass or fail depending on the run. Compare the set
of names, never the counts. Chunk 5.1 and 8.1 own the survivors.

Five hand-compiled programs cover what has no test target, all passing on this tree:
`docs/evidence/platform-selftest.cpp` (44 checks, the producer's file/hash/VRAM layer),
`docs/evidence/wf-io-selftest.cpp` (139 checks, the fetcher's relay through the real `wf_*`
pipeline), `docs/evidence/wf-extent-selftest.cpp` (33 checks, the strict extent reader),
`docs/evidence/provider-lease-selftest.cpp` (14 checks, the provider's lease through `inspect()`
with no backend), and `docs/evidence/uring-flush-probe.cpp` (3 checks, the io_uring flush rule).
`docs/linux-port-commands.md` has the exact compile lines.

Nothing verified so far is a performance result for a model. The measurements are the fetcher's
reader comparison (`docs/evidence/wf-io-selftest.cpp`: io_uring x1.04 at four threads, x1.44-x1.67
at one, 64 MiB across 218 ops, one NVMe, cold cache) - it isolates the reader, not a prefill. No
supported model weights exist on this box.

## Constraints on this box

1. A Strata serve process (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~29.9 GiB of
   32 GiB VRAM, and the user's browser GPU process holds another 170-320 MB. **Both pids change
   between sessions and even within one** - read them from
   `nvidia-smi --query-compute-apps=pid,used_memory`. Stopping Strata also stops this assistant, so
   GPU work is the user's to run by hand. Never kill it to test. Host-only gates work;
   `llama-server --version`, the producer's CUDA-enumeration path and the fetcher selftests all
   initialised a context without disturbing it, and NVML queries (`nvmlDeviceGetMemoryInfo`) are
   read-only. Anything that allocates device memory is still the user's call, and **the headroom
   moves**: it went from ~900 MB to ~0 mid-session on 2026-11-03, which is why a CUDA-needing gate
   can pass at one hour and fail at the next. Re-run before concluding the code broke; a program
   that only does `cudaMalloc(8 MiB)` says whether the box or the change is at fault.
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
  platform-layer program, `wf-io-selftest.cpp` is chunk 3.1e's reader program (io_uring against the
  pread baseline, byte-compared through the real `wf_*` pipeline), `wf-extent-selftest.cpp` is chunk
  3.1f's extent reader (33 checks, including a seven-part batch against four slots),
  `provider-lease-selftest.cpp` is 3.1f's lease program (14 checks through the public `inspect()`
  with no backend), and `uring-flush-probe.cpp` is the 3.1f probe for the io_uring submit/wait
  rule. Copy them to `build/` and compile them; `docs/linux-port-commands.md` has the exact `g++` lines and the no-model producer
  smoke recipe. Anything written under `build/` is scratch and is not in git; chunk 2.2b's map was
  never saved, so its rename is only reconstructable from its commit message.
- When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
  but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
  text that was not in the working tree (it read the diff base rather than the tree), yet its second
  pass caught three real problems in what the commit *claimed*. Reviewers check claims, not just
  code: re-verify the wording in docs and messages, not only the C. Note that a subagent can be
  stopped mid-run and return no output; if a survey agent dies, do the greps yourself.

## Behaviour changes landed this session

Chunks 3.1c/3.1d/3.1e's behaviour changes (the CLI running on Linux, the POSIX receipt identity
fields, `--carry-storage hybrid` without a drive letter, SIGINT/SIGTERM cancellation, the live
carry-resume RAM floor, no process-memory limit in `host_memory::begin()`, the io_uring reader and
`LLAMA_WF_IO_URING=0`, `WF_MODE_AUTO` probing cuFile, buffered `io_direct`, the deleted
`--ds-gpu-ring-slots`, the sticky failed-read accounting, liburing on the link line) are recorded in
the plan's own entries for those chunks. This session's (chunk 3.1f):

- **The provider's lease is POSIX.** `open(O_RDONLY|O_NOFOLLOW)` relative to the pinned parent
  directory (`openat`), ancestor descriptors held for the provider's lifetime and rebound to their
  names by `identity()`, which runs around every file operation. `identity()` checks `st_dev`, the
  8-byte little-endian `st_ino` inside the 16-byte receipt field, `st_size`, `st_mtime` in Unix ns,
  `S_ISREG` and `st_nlink==1` - the four keys `slstack/native_broker.py` writes into a copy receipt.
  `DRIVE_FIXED` + `QueryDosDeviceW` became a `statfs` refusal of network and RAM-backed types; the
  Engram row reader keeps the 4 KiB-aligned extent but reads it with `pread`, so the
  `GetDiskFreeSpaceW` sector probe is gone.
- **The strict profile is the relay, not DirectStorage.** `wf_create_required_dstorage` (it refused
  unconditionally here) is `wf_create_required_streaming`, and the internal flag is
  `strict_streaming`. It admits one exact typed geometry - auto request size, bounded staging, async
  copy ordering, an explicit pinned staging tier - with no ambient env tuning, no expert cache, no
  relay prefetch buffer, and sticky failures. The provider asks for `WF_MODE_DIRECT_IO` explicitly
  (not `AUTO`, which could resolve to GDS and register the fetcher's allocations before its bank
  pointers are replaced) and `io_direct=false`, so the fetcher's banks are exactly the admitted
  `compute_buffer_bytes`.
- **Admission moved, so a launch recipe needs a larger `--provider-host-bytes` than the Windows
  one.** The relay's pinned staging tier (`(buffers+1)` bank-sized buffers) and the extent reader's
  32 MiB window are charged to the provider's host plan; the GPU-side staging term left the device
  plan, because nothing here stages on the device. `--fetch-profile baseline` is refused while
  inspecting, with a message naming the typed profiles, rather than at pipeline creation.
- **Named extents are io_uring reads into the compute bank.** `wf_fetch_extents_direct`,
  `..._overlap`, `wf_fetch_ready_extents`, `wf_wait_ready_extents` and `wf_ready_direct_bank` went
  from `return false` to real: extents are split into parts, read through io_uring (or the `pread`
  baseline) into a pinned window of four 8 MiB slots, and uploaded on the caller's stream - or the
  fetcher's copy stream for the tail-hint worker, which publishes one event per batch and hands back
  a fence. The relay now publishes the per-stage generation where a stage reaches `GPU_READY` (both
  sites; the deleted queue used to be the only publisher), so `wf_ready_direct_bank` and the hint
  protocol work. `acquire_layer_experts` and `prefill_experts` are live. Bounds are checked against
  the file and the bank before any byte is written; a negative or short read poisons the fetcher; a
  batch naming a stage that left its bank refuses without poisoning (nothing was written).
- `acquire()` gained a guard that a stage landed in the bank its regions were planned in
  (`buffers[id%2]`), because the extents name regions of that bank: a stage recycled into the other
  one would bind the wrong file bytes into the wrong tensors.
- `configure_host_cache` measures the RAM expert tier from `/proc` (`MemAvailable`, `VmRSS`) through
  `src/llama-dsv41-host-memory.h`, keeping the Windows reserve and margin arithmetic and the
  `LLAMADECODE_HOST_CACHE` log line's field names.
- Deleted rather than faked, as `--ds-gpu-ring-slots` was in 3.1e: `--ds-submit-threads` (no reader
  since the IOCP workers went) and `--ds-zero-padding-only` are unknown producer options, with their
  `wf_params` and provider `config` fields, their receipt keys, and
  `tools/slstack-weight-fetcher.def` - a Windows export list no build rule used, whose names were
  already stale. Anything mirroring `wf_params` by field order is now doubly stale - notably
  `vendor/engines/cosmos_windows/weight_streaming/ds_backend.py`, which Phase 6 deletes anyway.
- **Both io_uring readers wait through `io_uring_submit_and_wait()`.** `io_uring_wait_cqe()` does
  not flush entries the kernel has not been told about on this liburing, and `io_uring_submit()` can
  return 0 without being an error, so the old "submit, then wait" pair could leave a layer's reads
  unseen forever. `docs/evidence/uring-flush-probe.cpp` demonstrates both halves.
- Still Windows-shaped and untouched: `slstack/single_runtime.py` stages `ds4-server.exe` and
  `slstack-weight-fetcher.dll`, and `scripts/build_combo_runtime.py` builds a preset named
  "RTX: ULTRA prefill -> ds4-win decode". Those are 5.1/7.1. Note for 5.1:
  `engines/streaming-llama/prefill_llama.cpp/tools/slstack-weight-fetcher.cpp` is a two-line anchor
  file, so the Linux `.so` exports one data symbol and carries no fetcher API - there is no DLL twin
  to stage.

## Gotchas

- The producer builds the fetcher **twice**: `tools/dsv41-prefill` compiles its own copy of
  `src/llama-weight-fetcher.cpp` into the CLI, and `src/` compiles it into `libllama.a`. A manually
  linked selftest uses `build/llama/src/CMakeFiles/llama.dir/llama-weight-fetcher.cpp.o`, which
  `ninja -C build/llama llama-dsv41-prefill` does **not** rebuild - so it silently links a stale
  object and proves nothing about your edit. Run `ninja -C build/llama src/libllama.a` first.
- A chunk the reader never accounts does not hang the pipeline: the scheduler polls
  `wf_layer_io_done` with a timeout, so it spins reporting `progress=NO` with the RAM buffers stuck
  `in_use`, and `wf_stop` still joins. That makes a dropped chunk look like a stall, not a deadlock
  - account every chunk on every exit path.
- `io_uring_wait_cqe` does not return `-EINTR` here: a probe blocking it while a `sigaction`
  handler with `flags=0` (what the producer installs) fires showed no escape, so liburing retries.
  The error paths are defensive, not a live bug.
- `adversarial-code-review` earned its keep again on 3.1f, but only for **one** of its five
  findings: it found the io_uring submit/wait flush hazard (real), while its Critical claim that a
  zero `io_uring_submit()` return is treated as fatal was a misreading of `< 0` as `<= 0`, and its
  other three - generation 0 through the tail hint, a stale YOCO skip mask after a layout switch,
  the extent window outside the job host budget - were refuted by lines already in the tree (the GDS
  arm publishes the generation, a layout switch rebuilds the fetcher,
  `llama-dsv41-sweep.cpp:456` sums `provider.host_bytes`). Verify every finding against the working
  tree, and prove a behavioural claim with a program rather than with recall: the flush claim was
  settled by `docs/evidence/uring-flush-probe.cpp` in thirty seconds. A reviewer that quotes text
  not in the tree is reading the diff base; say so and make it re-read.

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
- The fetcher's reader has three rules that are easy to break again. A chunk taken off the queue and
  never submitted never completes, and the pipeline waits on that layer's completion counter - so a
  full SQ ring puts the remainder back on the queue, and `wf_stop` drains the queue and the
  in-flight reads before joining. `IOSQE_FIXED_FILE` means the SQE's `fd` field is the *registered
  index*, not a descriptor; registration is optional, so both paths must stay wired. And **the wait
  must also be a submit**: `io_uring_wait_cqe()` does not flush the submission queue here, so every
  read queue needs `io_uring_submit_and_wait()`, not `io_uring_submit()` plus a wait. Treat only a
  negative submit return as an error - 0 is legitimate.
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
- The strict extent reader is **CUDA-optional on purpose**: without `WF_HAS_CUDA` the fetcher's
  banks are host buffers and the upload is a `memcpy`, so
  `docs/evidence/wf-extent-selftest.cpp` compiles `src/llama-weight-fetcher.cpp` with no
  `-DGGML_USE_CUDA` and needs no VRAM. Do not "simplify" that by folding the `#ifdef` arms - the
  non-CUDA build is also what a CPU-only llama build links. Testing the CUDA arms needs a
  `ggml_backend_cuda_init` plus a compute stream, and a ggml CUDA backend wants roughly 700 MB of
  device memory past its context; on this box that is more than Strata and the browser leave, so
  those arms have never executed.
- `std::strtoull` returns `unsigned long long`, which on LP64 is not `uint64_t` (`unsigned long`), so
  `std::min(capped, count * width)` fails to compile here and compiled on MSVC. Convert explicitly.
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
  (plan's "locked memory" risk). It gates `llama-dsv41-locked-ram.h` and the carry lease (chunk
  3.1h) and the decode library's host-cache path, and `host_memory.begin()` deliberately does not
  solve it. Note that `cudaHostAlloc`/`cudaMallocHost`, which the fetcher's staging tier and the
  extent window use, do not need that allowance - only the engine's `mlock` does.

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
