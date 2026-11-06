# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan with the chunk backlog: `docs/linux-port-plan.md`. Commands the user can run:
`docs/linux-port-commands.md`. Repo review: `linux-port.md`. Port scaffolding: `scripts/port/`
(its README explains each tool and when to use it).

**After** the legacy removal, the next piece of engineering is planned separately in
`docs/blackwell-enablement-plan.md` - the Blackwell-only scope and support for the
`deepseek-v4.1-flash-IST-DAS-2.00bpw` shards on disk, reusing Basalt and Strata kernels rather than
writing them. It records the decisions taken on 2026-11-06 (reuse into this repo, GGUF stays, both
prefill and decode are targets, locked memory is best-effort) and the measurements those decisions
rest on. Read it before starting anything outside the port backlog.

What a chunk landed *is* in its commit message and in the plan; this file only carries what neither
records - the open decisions, the box's constraints, the verification method, and the traps.

## What the next session is for

Phase 3's producer CLI, its weight fetcher, its native weight provider, its Engram row loader, its
carry store and its locked RAM tier are now Linux: the provider holds a POSIX lease on the verified
model file, builds the bounded relay, acquires stages in alternating banks, writes named file
extents into a compute bank, admits its RAM expert tier from `/proc`, loads a prompt's Engram rows
through io_uring queues, and the carry tier locks as much of its RAM as the kernel allows. Nothing
in the streamed prefill path refuses on Linux any more. What it still needs is a qualified GGUF. In
the order worth doing:

1. **Chunk 2.2e, prose half - Metal words in comments and in the inert profiler's messages.** The
   identifier half landed (`b6c6d9c`): the nine `DS4_GLM_METAL_*` constants and the `bool metal`
   parameter of `qwen4_first_token_test()`, both invisible to chunk 2.2b's gate for the same
   word-boundary reason 2.2d fixed. No code identifier and no CLI surface names Metal now.
   285 mentions remain in `engines/ds4`: 112 comment lines and 173 lines inside quoted strings - all
   printed diagnostics and the expert-locality profiler's JSON text, 249 of them in `ds4.c`. This is
   a per-line decision, not a substitution: some mentions describe what the *retired* backend did and
   mislead in a CUDA-only engine (the `ds4.c` header block's "whole-model Metal ...", the `Metal
   command batch` resume failures, the `Metal SSD streaming` refusal), while others are inherited
   provenance that chunks 2.2d, 3.1c and 3.1g kept on purpose (`ds4_metal.m` in a `ds4_cuda.cu`
   comment, the IoRing the loader was ported from). Rewording printed messages changes engine output:
   two doc lines quote them, and `examples/engine_parity/RESULTS.json` plus
   `examples/engine_environment/mac-*.json` name Metal as historical Mac measurements that rewording
   does not invalidate. Regenerate the env table in the same commit anyway - `envvars.json` quotes
   engine comment text as `consumer` evidence, so comment edits move rows (2.2e 1/3 moved 4).

Also open: 5.1 (Python orchestrator: `guard.py` Job Object -> a cgroup scope,
`single_runtime.py` PRODUCER_FILES), 7.1 (docs/presets/tests), 4.1 (ggml backend prune), and
8.1 (qualification), all blocked on a qualified GGUF for the last one. Phase 3's remaining Win32 is
`mtmd`, `server` and `decode-receiver`, none of which the streamed prefill needs.

What a prefill attempt needs beyond a qualified GGUF: nothing else. The three provider arms
(`launch_early`, `start_engram_prompt_early`, the synchronous load in `prepare_engram_prompt`) are
live, the loader runs on io_uring, and the carry tier no longer refuses an allocation larger than
`RLIMIT_MEMLOCK`, so the first real `--engram-prefetch-bytes` run is the test of the geometry
defaults, which were tuned on a Windows RAID-0 and are still the values in `choose_scan()`. Expect
`CARRY_RAM_LOCK_PARTIAL` in that run's log on this box: against the 8 MiB allowance a multi-gigabyte
carry tier is mostly pageable, and the paging guard, not the page lock, is what keeps it resident.

Chunk 3.2 (the decode library loader) landed inside 3.1c, because `live-session.inc` is part of
`main.cpp`'s body: `dlopen`/`dlsym` of `ds4-decode.so` beside the executable. Its binding is
verified by the selftest; a live session against weights is not, and cannot be here.

## State

`git log --oneline` from `58ff877` forward is the port, chunk by chunk; `docs/linux-port-plan.md`
maps chunk numbers to commits. As of this session: `3bc184a` + `0265129` (3.1a), `dcb5666` (2.2c),
`7973c20` (3.1b), `4940201` (2.2d), `3fc4f64` (3.1c), `6fc8cac` (3.1d), `0ffe3ce` + `ff6bb44` +
`1325655` (3.1e), `3afcd91` + `1e4cf01` (3.1f), `2f79178` (3.1g), `187f2fa` + `09b6158` (the
model-file claim and the Blackwell plan), `d7eae91` (3.1h), `b6c6d9c` (2.2e 1/3), plus the
plan/handoff commits.
`git status -sb` shows no `[ahead N]` after pushing; check it rather than assuming.

Working tree is clean apart from scratch under `build/`. `python3 -B scripts/release_check.py`
reports 0 findings, and **no hook runs it here**: `git config --get core.hooksPath` is empty and
`.git/hooks` holds only the samples, so the check has to be run by hand before committing. Do not
wire `.githooks` up on this box without changing the git identity - `release_check.py --identities`
requires `Front Eagle <fronteagle739@users.noreply.github.com>`, which the fork's history carries,
while this checkout's config is `Mac` and every port commit is `Mac`-authored. Python suite:
**584 tests, 5-6 failures, 5 errors** (fork baseline 585 / 6 / 6), and the
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
Three more are real test sources rather than evidence files, and all three still have no build
rule: `tests/dsv41_engram_early_io.cpp` (15 checks, chunk 3.1g's row loader over a 208 MB synthetic
table file, no GPU and no model), `tests/dsv41_carry_release.cpp` (the carry store's release,
recommit and reseed cycle over 40 byte-preserving layers) and `tests/pinned_carry_memory.cpp`
(chunk 3.1h's RAM tier, now a Linux proof rather than the `SKIP` it printed). The last two also
need no backend: they link only the carry object and two ggml libs. `docs/linux-port-commands.md`
has the exact compile lines for all eight - the lease recipe was missing from it until the last
session, and the two carry recipes until this one.

Nothing verified so far is a performance result for a model. The measurements are the fetcher's
reader comparison (`docs/evidence/wf-io-selftest.cpp`: io_uring x1.04 at four threads, x1.44-x1.67
at one, 64 MiB across 218 ops, one NVMe, cold cache) and the Engram loader's
(`tests/dsv41_engram_early_io.cpp`: ring and `pread` the same within a few ms, because its 208 MB
fixture fits in the page cache - it compares mechanics, not a device). Neither isolates a prefill,
and no supported model weights exist on this box.

## Constraints on this box

1. A Strata serve process (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~29.9-30.1 GiB of
   32 GiB VRAM, and the user's browser GPU process holds another 170-320 MB. **Both pids change
   between sessions and even within one** - read them from
   `nvidia-smi --query-compute-apps=pid,used_memory`. Stopping Strata also stops this assistant, so
   GPU work is the user's to run by hand. Never kill it to test. Host-only gates work;
   `llama-server --version` and the fetcher selftests all initialised a context without disturbing
   it, and NVML queries (`nvmlDeviceGetMemoryInfo`) are read-only. Anything that allocates device
   memory is still the user's call, and **the headroom moves**: it went from ~900 MB to ~0 mid-session
   on 2026-11-03, which is why a CUDA-needing gate can pass at one hour and fail at the next. Re-run
   before concluding the code broke; a program that only does `cudaMalloc(8 MiB)` says whether the
   box or the change is at fault. The margin is thin and it moves: on 2026-11-06 morning
   `cudaFree(0)` and `cudaMemGetInfo` themselves returned `out of memory`, and later the same day
   `cudaMemGetInfo` reported 116 MiB free, `cudaMalloc(8 MiB)` succeeded, and the two gates that
   had been unrun - `docs/evidence/wf-io-selftest.cpp` and the producer's no-model receipt smoke -
   both ran to their documented outcomes. A CUDA gate is therefore *unverified* rather than failing
   whenever the probe says no; probe first, in the same session, before blaming a change.
2. **No qualified model file, and the weights that are on this box cannot be made into one.**
   Re-read with `slstack/gguf_header.py` on 2026-11-06 (`read_header` + `validate_dsv41_types`),
   under `qwen38-gyro/Strata/models/`:
   - `Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ1_0/TQ2_0/TQ3_0.gguf` (54.5 / 92 / 108 GiB): architecture
     `qwen4exp`, 48 blocks, and tensor type ids **144-149** (the TQ quants) - outside the engine's
     type table, so the loader rejects them. Their `Q6_K`/`Q5_K`/`Q8_0`/`BF16`/`F32` tensors are not
     the problem; the TQ ones are.
   - `deepseek-v4.1-flash-IST-DAS-2.00bpw/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-*`: all four
     shards present (138 GB), architecture `deepseek41`, 40 blocks, `split.count=4`. Refused twice
     over: it must be assembled first (`merge_deepseek41_shards.py`, not in this checkout), and its
     per-block expert triplets mix `IQ1_M`/`IQ2_XS`/`IQ2_XXS` while dense tensors are `Q5_K` and
     `BF16`. `validate_dsv41_types` accepts only `(Q4_K,Q4_K,Q4_K)`, `(IQ2_XXS,IQ2_XXS,Q2_K)` or
     `(MXFP4,MXFP4,MXFP4)` per block, dense in `Q8_0`/`Q4_K`/`Q4_0`, `engram_embd` in `I8`,
     `token_embd` in `F16`, everything else `F32`. **Merging would not qualify it** - the quant set
     itself is what is refused, so no requantization means no run.
   - `deepseek-v4.1-flash-shared/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf`: 1 of 9
     shards, 5.7 MB, zero tensors - the download is incomplete, and `Q8_K` is not in the expert set
     either.
   - `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110 GB, 2 tensors) is named by
     `deepseek41.engram.file`. That is the llama.cpp Engram representation, not the one ds4's row
     loader reads: the loader takes 264-byte E4M3+E8M0 rows at the `engram_offsets` **inside the
     model file**. A qualified file needs its Engram tables inline.
   - `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` is the draft; the validator calls it not a native
     `deepseek41` GGUF.
   So Phase 3's prefill gate and Phase 8 need a download or a quantize run in the accepted type set,
   not a merge of what is here. 336 GB free on the NVMe.
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
- A **rewrite** chunk (3.1g replaced one file's IO halves and kept its platform-neutral halves) is
  outside all three tools: there is nothing to collapse, and the positional diff is mostly the new
  code. What proved the unchanged half was a scratch function-level comparison - extract each named
  member from `git show HEAD:<path>` and from the working file by brace matching, drop comments and
  the guard lines, and require equality (`build/engram-neutral-diff.py`, not committed; rewrite it
  if you need it). Eleven members reported identical. Pair it with the warning-profile comparison,
  which for a rewrite should only *lose* warnings (the deleted stubs' own).

- Sweep inputs a later session may need to re-run are committed under `docs/evidence/` -
  `rename-2.2d.json` is chunk 2.2d's rename map, `platform-selftest.cpp` is chunk 3.1c's
  platform-layer program, `wf-io-selftest.cpp` is chunk 3.1e's reader program (io_uring against the
  pread baseline, byte-compared through the real `wf_*` pipeline), `wf-extent-selftest.cpp` is chunk
  3.1f's extent reader (33 checks, including a seven-part batch against four slots),
  `provider-lease-selftest.cpp` is 3.1f's lease program (14 checks through the public `inspect()`
  with no backend), and `uring-flush-probe.cpp` is the 3.1f probe for the io_uring submit/wait
  rule. Chunk 3.1g's program is not under `docs/evidence/` but in the test directory -
  `tests/dsv41_engram_early_io.cpp`, ported with the loader, 15 checks over a 208 MB synthetic table
  file, no GPU and no model - and chunk 3.1h's two are there too (`tests/dsv41_carry_release.cpp`,
  which was red before the change, and `tests/pinned_carry_memory.cpp`), because the tier is engine
  code with a test source rather than a throwaway program. Copy them to `build/` and compile them; `docs/linux-port-commands.md` has the exact `g++` lines and the no-model producer
  smoke recipe. Anything written under `build/` is scratch and is not in git; chunk 2.2b's map was
  never saved, so its rename is only reconstructable from its commit message.
- When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
  but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
  text that was not in the working tree (it read the diff base rather than the tree), yet its second
  pass caught three real problems in what the commit *claimed*. Reviewers check claims, not just
  code: re-verify the wording in docs and messages, not only the C. Note that a subagent can be
  stopped mid-run and return no output; if a survey agent dies, do the greps yourself.

## Behaviour changes landed this session

Chunks 3.1c through 3.1g's behaviour changes (the CLI running on Linux, the POSIX receipt identity
fields, `--carry-storage hybrid` without a drive letter, SIGINT/SIGTERM cancellation, the live
carry-resume RAM floor, no process-memory limit in `host_memory::begin()`, the fetcher's io_uring
reader and `LLAMA_WF_IO_URING=0`, `WF_MODE_AUTO` probing cuFile, buffered `io_direct`, the deleted
`--ds-gpu-ring-slots`, `--ds-submit-threads` and `--ds-zero-padding-only`, the sticky failed-read
accounting, the POSIX provider lease, the bounded relay as the strict profile, the named-extent
reads into a compute bank, the RAM expert tier measured from `/proc`, the move of the relay's
staging tier and the extent window into the provider's host plan, the Engram loader's io_uring
queues with its serial `pread` baseline and `LLAMADECODE_ENGRAM_IO_URING=0`, the deleted
`LLAMADECODE_ENGRAM_IO_PRIORITY`, the loader's own descriptor checked against the provider's lease,
`load()`/`load_early()` taking a path and an `int` fd, and the provider's three Engram arms being
open) are recorded in the plan's entries for those chunks and in their commit messages. This
session's (chunk 3.1h):

- **The carry RAM tier locks as far as the kernel allows instead of refusing the allocation.**
  `mlock` is all-or-nothing per call and `RLIMIT_MEMLOCK` here is 8 MiB soft *and* hard, so the
  previous Linux arm threw `Carry RAM cannot be locked` for any tier past it - which is every real
  carry tier. It now locks in 64 MiB chunks and halves the chunk at each refusal, retrying the same
  offset, so the allowance is used rather than skipped; the tail stays allocated and merely
  pageable. That is option 3 of `docs/blackwell-enablement-plan.md` and Basalt's rule.
  `state().locked_bytes` sits beside `state().bytes`, per mapping in a `locked` map, and
  `state().bytes` is still the admission figure - **no admission change**, so the carry budgets in
  existing recipes still apply. `CARRY_RAM_LOCKED` gained `locked_bytes=`, and a tier that cannot be
  fully locked also prints `CARRY_RAM_LOCK_PARTIAL bytes= locked_bytes= pageable_tail_bytes=
  rlimit_memlock_bytes= errno=`. Historical receipt lines in
  `examples/engine_parity/RESULTS.json` keep the old wording; nothing parses them.
- **Chunks stay whole pages.** `mlock` accounts rounded-up pages, so a partial-page chunk records
  fewer bytes than the kernel holds; the first version of this reported 2047 pages for a tier smaps
  said was 2048. `locked_bytes` is therefore exact against `/proc/self/smaps`.
- **Suspend is `munlock` of the locked bytes then `MADV_DONTNEED`** (`VirtualUnlock` +
  `MEM_DECOMMIT`), and resume re-locks best effort at the same address; the suspend line gained
  `unlocked_bytes=` and the resume line `locked_bytes=`.
- **The resume refusal on physical memory is kept, not dropped.** The Windows arm refused a resume
  larger than available physical memory before committing pages; `resume_pages` now reads MemAvailable
  through `llama_dsv41_host_memory::proc_field`, which is why `llama-dsv41-locked-ram.h` includes
  `llama-dsv41-host-memory.h`. Pages refault on first touch, so nothing else refuses an oversized
  resume.
- **Fixed a leak the collapse exposed.** `allocate` accounted the mapping and then built the
  `shared_ptr`, so if the control block - or the accounting node inserted before it - threw, no
  deleter existed: the mapping stayed mapped and `state().bytes` kept the charge. The accounting and
  the `shared_ptr` are now in one `try`, and the `catch` releases the mapping, the per-mapping entry
  and both counters before rethrowing.
- **Allocation still has no fallback.** An `mmap` failure and an admission refusal throw exactly as
  before; only the *lock* became best effort. `llama-dsv41-carry.cpp` is a pure deletion -
  `code_line_diff.py` reports 0 changed code lines against its parent - so its lease, identity
  checks, positioned I/O, flush and ancestor pin are the Linux arms the compiler already took.
- No environment variable was added or removed: `envvars.json` is still 2,234 names, and the table
  regenerated with no row moved (the tier reads none).
- The geometry defaults in `choose_scan()` are still the Windows RAID-0 tuning, untouched.

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
  The same trap is on both carry programs, which link
  `build/llama/src/CMakeFiles/llama.dir/llama-dsv41-carry.cpp.o`.
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
  the env names - but it also *hid* the identifiers with a leading underscore (`glm_metal_*`,
  `g_metal_graph_*`, `ds4_engine_metal_graph_*`) from 2.2b's gate. 2.2d caught them with prefix
  entries long enough to start at a real boundary, and 2.2e caught the same class again on the other
  side: `DS4_GLM_METAL_*` constants are invisible to `\bMETAL_` because `DS4_GLM_` precedes it, so a
  Metal-name gate needs entries for *every* prefix, not one. `grep -rhoE
  "\b[A-Za-z_0-9]*[Mm][Ee][Tt][Aa][Ll][A-Za-z_0-9]*\b"` over the engine source is the honest census;
  it separates the kept env names from the identifiers by matching `DS4_[A-Z_]*METAL[A-Z_]*` first.
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
  must never be the only submit**: `io_uring_wait_cqe()` does not flush the submission queue here, so
  a queue either waits through `io_uring_submit_and_wait()` (the fetcher's reader) or flushes with
  `io_uring_submit()` and then blocks on something else (the Engram loader's `eventfd` poll, chunk
  3.1g). Treat only a negative submit return as an error - 0 is legitimate.
- **One submitted entry yields exactly one CQE - including a cancel request.** `IORING_OP_ASYNC_CANCEL`
  produces its own completion, so a drain loop that counts only "reads" never reaches zero and the
  cleanup refusal fires on a healthy queue. The Engram loader marks the cancel with a sentinel
  `user_data` and retires one `in_kernel` entry per CQE.
- **Consuming an `eventfd` counter before polling loses a wake.** `io_uring_register_eventfd` posts on
  every completion; if you `read()` the counter and then `poll()`, a completion whose CQE is already
  queued but not yet peeked produces no further event and the poll runs its whole timeout - the first
  version of the Engram loader paid 100 ms on every load (115 ms against 14-20 ms). Check
  `io_uring_cq_ready()` on every queue sharing that eventfd before sleeping.
- `io_uring_peek_cqe` does **not** block on this liburing when the CQ ring is empty and the ring has
  no `IORING_SETUP_IOPOLL`, CQ overflow or pending task work: it returns `-EAGAIN`. A drain-until-empty
  loop is therefore safe; `io_uring_wait_cqe` is not (it needs the submit, and does not return
  `-EINTR`).
- The Engram loader's geometry and reader env reads (`LLAMADECODE_ENGRAM_IO_URING`, `_SCAN`,
  `_READ_ALIGN`, `LLAMADECODE_ENGRAM_EARLY`, `LLAMADECODE_ENGRAM_IOPS`) are function-local statics:
  once per process. A ring-vs-`pread` or scan-vs-random comparison is two runs, not one.
- `tests/dsv41_engram_early_io.cpp` has no build rule and links the `src/` objects, so it needs
  `ninja -C build/llama src/libllama.a` first for the same stale-object reason as the evidence
  files - `tools/dsv41-prefill` compiles its own copy of `llama-dsv41-engram-prefetch.cpp` as well
  as of `llama-weight-fetcher.cpp`, and building the CLI does not rebuild the `src/` one.
- In C++, `::stat(path, &info)` finds the **type** `struct stat`, not the function, and fails to
  compile with "too few arguments". Use `std::filesystem::file_size` or `fstatat` in test code; the
  port of this test hit it.
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
- `scripts/port/warning_profile.py` reads `build/llama/compile_commands.json`, so it cannot cover the
  **engine**: `ds4.c` is not in that file. Compare an engine TU by hand instead -
  `make -C build/wt-base/engines/ds4 ds4.o CUDA_ARCH=sm_120` and the same in the working tree, then
  `grep "warning:" | sed` away the path, line and column, `sed 's/[0-9]+/N/g'`, sort and diff - and
  compile `ds4_ext.o` too, because `-DDS4_EXTERNAL_DECODE=1` is a second compilation of the same
  file with different live arms. Each takes about a minute.
- `scripts/release_check.py` flags any absolute path under a user's home directory as a "machine
  path", and it scans every tracked blob - including a `.cpp` committed under `docs/evidence/` - so
  docs and evidence files must use repo-relative paths. `git status -sb` is the push check; the
  pre-commit hook also scans the commit message, so a message must not quote a machine path either.
- **Locked memory is landed** (chunk 3.1h, option 3 of `docs/blackwell-enablement-plan.md`): chunked
  best-effort `mlock`, the tail allocated and pageable, the shortfall logged as
  `CARRY_RAM_LOCK_PARTIAL`, and the host-memory paging guard as the enforcement. Details in the
  plan's 3.1h entry. `cudaHostAlloc`/`cudaMallocHost`, which the fetcher's staging tier and the
  extent window use, do **not** need an `RLIMIT_MEMLOCK` allowance - only the carry tier's `mlock`
  does, and here the allowance is 8 MiB soft *and* hard, `setrlimit(∞)` returns `EINVAL`, and
  `systemd-run --user` cannot raise `LimitMEMLOCK`, so an unprivileged process can never lock a
  carry tier in full.
- **`mlock` splits the VMA at the lock boundary** and accounts whole pages: a partly locked tier is
  several `/proc/self/smaps` VMAs, each with its own `Locked: kB`, and their sum is what the tier
  holds. A partial-page chunk therefore under-reports what the kernel locked (measured: 2047 pages
  recorded against 2048 reported). Keep lock chunks page-multiples.
- **Parsing `/proc/self/smaps` has two traps.** A VMA header line for an anonymous mapping ends after
  the inode, and glibc's `%255[^\n]` *assigns an empty string and still counts the conversion*, so
  counting conversions cannot tell an anonymous mapping from a named one - skip the five known
  fields and test what is left. And a field line must not be mistaken for a header: parse
  `%llx-%llx %15s %llx %x:%x %llu` and require all seven.
- **The tier's `locked` map is keyed by the mapping address.** That is safe only because a suspended
  mapping stays reserved (its VA cannot be handed to another `mmap` while it is mapped) and because
  the erase happens on the way out. A change that unmaps without erasing would make the next
  allocation at the same address inherit the old entry.
- **`munlock` on pages that are not locked is a no-op success**, so the deleter may unlock the amount
  it captured at allocation even after a suspend/resume cycle re-locked a different amount. The
  accounting comes from the map, not from that unlock.

## Suggested skills

- `adversarial-code-review` after a deletion or rename chunk, with the `scripts/port/` equivalence
  commands handed to the reviewer as evidence (see the method section for what it caught last time).
- The user asked, on 2026-11-06, that this work not be parallelised: subagents are expensive on
  context prefill, and one of them died mid-run with no output anyway. Verify inline - the compile
  checks, the hand-compiled programs and the function-level diff - and ask before fanning out.
- `add-unit-test` for extending `engines/ds4/tests/test_batch_io.cpp` or `test_decode_library.cpp`.
  The producer's reader is covered by `docs/evidence/wf-io-selftest.cpp`, compiled by hand, because
  the producer has no test target; the Engram loader by `tests/dsv41_engram_early_io.cpp`, the carry
  store and its RAM tier by `tests/dsv41_carry_release.cpp` and `tests/pinned_carry_memory.cpp`, all
  three also compiled by hand (`docs/linux-port-commands.md`).
- `code-design` / `code-reviewer` when a chunk changes admission rather than text - chunk 3.1h was
  one, and it went through inline verification only. Chunk 5.1 (`guard.py`'s Job Object -> a cgroup
  scope) is the next one that changes a resource decision.
- `adversarial-code-review` after a chunk that rewrote behaviour, if the user allows a subagent; the
  `scripts/port/` equivalence commands are the evidence to hand it. For 3.1h the proof was inline:
  the collapse report, `preprocess_equiv`, `code_line_diff`, the warning profile, and both carry
  programs, one of which was red before the change.
- `web-research` for liburing, CCCL 3 / libcudacxx header changes, or llama.cpp cmake details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when this session runs out of context - update this file, don't replace it with
  chat recap.
