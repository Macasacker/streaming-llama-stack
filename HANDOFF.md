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

**Phase 4, Phase 5 and chunks 4.3a, 4.3b and 4.3c are done, and Phase 7 is closed.** Phase 3's producer CLI, its weight
fetcher, its native weight provider, its Engram row loader, its carry store and its locked RAM
tier are Linux; the orchestrator (`slstack/`) has no `msvcrt` and no `CREATE_SUSPENDED`, the
guardian caps its tree in the kernel, ULTRA and mode1 producers launch here, and resident
admission plus the locked checkpoint reach `after_resume`. Chunk 4.1 landed, so ggml builds CUDA
and CPU only; chunk 4.2 deleted the nine backend trees that list never named and folded the
HIP and MUSA branches those trees had defined, so `metal_word_census --strict` passes over the
llama tree and nothing in the streamed prefill path refuses on Linux any more. What it still needs
is a qualified GGUF, which this box does not have. Chunk 7.1a deleted the Mac and paired docs,
examples and Mac test sources and rewrote the front page for one Linux machine, 7.1b fixed the
four suite names Phase 7 owned plus two scripts that still named `engines/ds4-win`, and 7.1c fixed
the census gate the remaining text sweep needs and made the rewording chunk 4.1 had claimed but
never did. Chunk 4.3a folded the same HIP and MUSA branches out of the engine's vendored mmq tier,
so `GGML_USE_HIP`, `GGML_USE_MUSA` and the macros their toolchains defined are gone from
`engines/ds4` entirely. Chunk 4.3b folded the host layer's `DS4_ROCM_BUILD` guards, and 4.3c folded
the AMD and Moore Threads compute-capability vocabulary in both trees, so **Phase 4 is closed**:
none of the 22 names in `docs/evidence/collapse-4.3c-tokens.txt` - `AMD_MFMA_AVAILABLE`,
`AMD_WMMA_AVAILABLE`, `CDNA*`, `RDNA*`, `GCN*`, `__GFX*__`, `V_DOT2_F32_F16_AVAILABLE` - appears in
a preprocessor directive anywhere in this checkout, and what survives of that vocabulary is the
host-side runtime `cc` selection code the port keeps on purpose (the plan's 4.3c entry lists it and
says why deleting it is not a text sweep). Chunk 7.1d taught `inventory()` to follow a read through
a wrapper helper and to refuse a name the wrapper discards, and regenerated `docs/VARIABLES.md`
from that: 2,143 documented variables to 2,186, with 114 names added and 71 dropped, and the
`engines/mlx-serve/` exemption gone from `validate_table()`, `envdocs.py` and
`reanchor_env_reads.py`. Chunk 7.1e took the Mac peer's text out of the llama tree and
`decode-receiver`: the census reads IDENTIFIER 0, ENV_NAME_CITED 0 and STRING 0 over both trees,
the llama tree's COMMENT bucket is 37 provenance lines (from 51), and
`gpu_graph_ported_m5_decode_feature_enabled()` lost the parameters nothing ever read.

1. **Chunk 6.1's remainder** - `vendor/engines/cosmos_windows` (the media workers the media
   tests import) and `slstack/media_tools.py`'s pinned Windows FFmpeg archive: port media to
   Linux or delete it. `GGML_ACCELERATE` and the Apple cmake logic in the llama tree
   (`ggml-cpu/CMakeLists.txt`'s APPLE branch, the `apple_m1`/`apple_m2_m3`/`apple_m4` CPU variants,
   `HAVE_BUGGY_APPLE_LINKER`, `_DARWIN_C_SOURCE`, the visionOS definition) is the same kind of
   leftover: it belongs with Phase 7's text and cmake cleanup, not with a backend prune.
2. **Chunk 8.1 - qualification**, blocked on a qualified GGUF: nothing in Phase 8 can start
   without a download or a quantize run in the accepted type set (see "Constraints on this
   box" 2). Its name set includes `test_yoco`, the suite's one stable failure.

What is left of Phase 3's Win32 is `mtmd`, `server` and `decode-receiver`, none of which the
streamed prefill needs. Chunk 4.1 changed nothing about `tools/decode-receiver`'s code - it is the
Mac peer's receiver, its target is `OFF` by default, and it never called the split-graph API; its
13 comment lines and the printed `LLAMA_PD_CACHE_CAPTURE` warning are reworded (chunk 7.1e), and
what remains in it is the `#if defined(_WIN32)` arms, which the port has not been asked to compile
here. Three findings from 2.2e that belong to those chunks and
to no chunk yet: 35 declarations in the engine's GPU headers are dead - declared by the deleted
Mac backend, defined nowhere, called nowhere (the scan is in this file's method section, and the
three that named Metal were deleted in 2.2e's comment half). The other two findings in that group
are settled by chunk 7.1d, one each way: the wrapper reads are now in the table (114 rows added),
and the names handed to a folded-to-false stub, such as
`DS4_METAL_DISABLE_M5_QKV_PAIR_QUAD_FUSE`, turned out to be the *reason* the table had to shrink -
`glm_graph_env_present("DS4_ROCM_X", "DS4_METAL_X")` discards its first parameter, so the 71
`DS4_ROCM_*` rows the old scan carried documented aliases nothing reads and went. Nothing was
renamed: 2.2b kept the names for recipe compatibility, and 7.1d fixed the inventory. Four findings from 5.1 in the same category: `lease.py`'s
CPU-reservation recognition matches hold lines by their *wording*, including the word Windows, so
a Linux CPU hold worded differently preempts the engine - settle the wording with the owner of the
other agent rather than widening the regex; `scripts/perfexp_pair.py` hashes
`<producer>/ds4-decode.dll` and writes `decode_dll_sha256`, which 8.1 regenerates; `media_tools.py`
downloads a pinned Windows FFmpeg zip (chunk 6.1's), while its Linux arm reports the system
`ffmpeg`/`ffprobe` and 7.1b's test asserts that arm; and
`engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp` still
declares `bool gpu_bufs_via_dstorage = false` and branches on it - a folded-to-false constant with
no chunk of its own. Three from 5.2 in the same category: neither the engine's `require_local_volume`
nor `slstack/producer_runner.py`'s copy of that refusal list contains NFS4's magic (`0x6e626470`),
so an NFS4 mount is accepted on both sides - fix both lists together or neither; `combo_native`'s
"timed native model weights cannot use the archival drive" refusal keys on a Windows drive letter
and therefore no-ops on Linux; and `gpu_bufs_via_dstorage`'s neighbour in the docs, the runner's
`commit()`, has no caller in this fork at all.

What a prefill attempt needs beyond a qualified GGUF depends on how it is driven. Through the
producer CLI under the guardian (`build/llama/bin/llama-dsv41-prefill` launched as a guarded
command - the path Phase 3 ported, and the one `docs/linux-port-commands.md` documents): nothing
else. The three provider arms (`launch_early`, `start_engram_prompt_early`, the synchronous load in
`prepare_engram_prompt`) are live, the loader runs on io_uring, and the carry tier no longer refuses
an allocation larger than `RLIMIT_MEMLOCK`, so the first real `--engram-prefetch-bytes` run is the
test of the geometry defaults, which were tuned on a Windows RAID-0 and are still the values in
`choose_scan()`. Expect `CARRY_RAM_LOCK_PARTIAL` in that run's log on this box: against the 8 MiB
allowance a multi-gigabyte carry tier is mostly pageable, and the paging guard, not the page lock,
is what keeps it resident. Expect also a scope ceiling: the guardian caps the tree at
`job_commit_gib`, which on Linux is `host_budget_gib`, so a plan whose host tier exceeds the config's
host budget is now refused by the kernel rather than reported by the poll.
Through `slstack`'s combo / ULTRA session path, nothing of the admission chain is missing any
more: chunk 5.3 landed the resident admission that `combo_producer`'s, `combo_native`'s and
`native_broker`'s `after_resume` call, so a producer is launched, admitted, and reaches the
engine's own readiness gate. Expect `DS4COMBO_RESIDENT_FLOOR_INERT` in that path's log on this box:
`memory.min` is written and read back, but the kernel zeroes it because no ancestor of the scope
carries a floor - what keeps the pages resident is the scope's `memory.swap.max=0` and the
guardian's paging guard. Expect `DS4COMBO_LOCKED_RAM_PARTIAL` too on any checkpoint tier above
8 MiB, for the same reason the carry tier reports `CARRY_RAM_LOCK_PARTIAL`.

Chunk 3.2 (the decode library loader) landed inside 3.1c, because `live-session.inc` is part of
`main.cpp`'s body: `dlopen`/`dlsym` of `ds4-decode.so` beside the executable. Its binding is
verified by the selftest; a live session against weights is not, and cannot be here.

## State

`git log --oneline` from `58ff877` forward is the port, chunk by chunk; `docs/linux-port-plan.md`
maps chunk numbers to commits. As of this session: `3bc184a` + `0265129` (3.1a), `dcb5666` (2.2c),
`7973c20` (3.1b), `4940201` (2.2d), `3fc4f64` (3.1c), `6fc8cac` (3.1d), `0ffe3ce` + `ff6bb44` +
`1325655` (3.1e), `3afcd91` + `1e4cf01` (3.1f), `2f79178` (3.1g), `187f2fa` + `09b6158` (the
model-file claim and the Blackwell plan), `d7eae91` (3.1h), `b6c6d9c` (2.2e 1/4), `f551a8d`
(2.2e 2/4), `a12383b` (2.2e 3/4), `76053fc` (2.2e 4/4), `270d7a9` (5.1a), `65e54bd` (5.1b),
`c17d6a6` (5.1c), `63c8dc1` (5.1d), `5fb07f7` (5.2), `1279063` (5.3), `0c9d01d` (4.1),
`5b941ff` (7.1a), `9a822c8` (7.1b), `050e0b1` (7.1c), `9125df4` (4.2a), `eaccc5e` (4.2b),
`26dacf6` + `cb3b58f` (4.2 recorded, baseline position), `4c368e7` (4.3 scoped), `e47c111` (4.3a),
`463cd31` (4.3b) and `bc63367` (4.3b recorded), `37b2bf1` (4.3c scoped, with the correction to what
4.2 and 4.3 had recorded), `41bc18a` (4.3c, the engine's `cuda/mmq/`) and `a50a7bb` (4.3c, the llama
tree's `ggml/src/ggml-cuda`, `caa8a9e` (4.3c recorded), `f197552` (7.1d) and `fe001b2` (7.1d
recorded), `1d47e54` (7.1e) and this file's and the
plan's commits after it. The branch was in sync with
`origin/linux-only` when this session opened and was pushed again after each chunk; check
`git status -sb` rather than assuming it still is.

The Python suite needs `STUDIO_RUNTIME_ROOT` set or 21 process tests raise `KeyError` for it (the
guardian and combo-process files, plus three others); the repo's gitignored `runtime/` works
(`mkdir -p runtime` then
`env PYTHONPATH=.:tests STUDIO_RUNTIME_ROOT=$PWD/runtime python3 -B -m pytest tests/ -q`). With it,
this tree reads **685 passed, 2 failed, 38 skipped** on the 7.1e run, and read **686 passed,
1 failed, 38 skipped** before it - 12 of the passing checks are chunk 7.1d's
`tests/test_env_inventory.py`, which calls `inventory()` and so adds about 25 s to a run - runs of
it have read 1, 2, 3 and 7 failures, since the two process tests named below move. Without it, the same tree reads a much worse line -
chunk 5.3's page-cache-fixture check also needs the root, and skips when it has none.
Compare failing *names* across runs, and compare them in a baseline worktree rather than reading a
count - the process tests flip and the skip set changes when a platform gate starts running.
Chunk 5.1 un-skipped `tests/test_combo_guard.py`, which was `skipUnless(os.name=='nt')`: 21 guardian
tests that had never run on Linux now do, which is why the skip count fell from 62 to 41. Chunk 5.2
added 22 runner checks and one shared-lock check and un-skipped nothing, so the skip line was still
41. Chunk 5.3 added 20 checks and un-skipped three names - among them the post-resume ownership
contract, which only runs now that the runner is Linux - so the line was 37. Chunk 7.1b fixed the
four names Phase 7 owned and added the POSIX FFmpeg-arm check, which skips the Windows marker case
here, so the line is 38 and one number moved up. The one stable failure is `test_yoco`, whose
preset reads `producer_prefill_sweep_tokens` as `auto` rather than the 49,152 it asserts - chunk
8.1's. Three names appear and disappear between runs,
`test_combo_guard::test_persistent_window_denial_still_stops_owned_job`,
`test_hold_yield::test_real_owned_child_stops_on_kit11_hold` and, seen this session,
`test_combo_single_stop::test_disconnect_inside_reasoning_then_next_turn_appends_only` - that one
asserts the producer evaluated *past* what a disconnecting client kept, so it fails when the box
makes the decode fast enough to catch up; alone-runs of it pass and fail on this tree and on a
worktree of the untouched HEAD alike (this session read pass/fail/fail here and pass/pass/fail on
the baseline). The first two also fail in a worktree of the commit before chunk 5.2, so they
are the box, not the tree.

Working tree is clean apart from scratch under `build/` and `runtime/`; `build/wt-base` is a
worktree now checked out at `fe001b2` - chunk 7.1d's recorded commit, the same tree HEAD has apart
from the working edits - so move it wherever a comparison needs a baseline - `git -C build/wt-base checkout <sha>` - or remove it with
`git worktree remove build/wt-base`. Note that a comparison after a chunk that moved lines in
`ds4.c` needs the *parent* commit, not HEAD, or the re-anchored `envvars.json` and the fold itself
show up as changes. `build/llama` is configured with `build.sh`'s CUTLASS and
split-F32 flags and `build/42/plain` holds the plain flag set from chunk 4.2's gate; both are
scratch, and chunks 4.3a and 4.3b touched no llama source so neither was rebuilt. Chunk 4.3c's
llama half did: `ninja -C build/llama -j8 ggml-cuda` rebuilt all 168 CUDA objects, the rest of the
llama tree relinked, and the five hand-compiled programs plus the three test sources without build
rules were recompiled and re-run against those objects. `engines/ds4` holds all five executables and
`ds4-decode.so` rebuilt from the folded tree (CUDA_ARCH=sm_120), so `./ds4 --help`, `./ds4 --metal`,
`make test-batch-io`, `make test-decode-library` and the platform selftest's `dlopen` arm all run
against the current engine without a rebuild; chunk 4.3c rebuilt the tier's nine CUDA objects and
relinked the five binaries and the `.so`, so that sentence is still true. Chunks 5.1, 5.2 and 5.3 touched no engine source. Chunk 7.1c touched one: `llama-context.cpp`'s
`llama_split_graph_create` refusal line, which sits inside `#ifndef GGML_USE_CUDA` and so cannot
reach a CUDA build's compiled text (`preprocess_equiv` shows 0 differing positions with
`-DGGML_USE_CUDA`). It also regenerated `tools/dsv41-prefill/studio-sampling.c`, whose only change
is the recorded source SHA line, and `build/llama/bin/llama-dsv41-prefill` was rebuilt after it and
passes `--help` and `--check-native-profiles`. Chunk 7.1e touched engine source again
(`ds4.c`'s stub) and llama-tree comment and string text only: the three compilations of `ds4.c`
were run against a worktree of HEAD for the warning comparison, the five binaries and the `.so`
relinked with the folded object and their four host gates re-passed, the whole llama tree rebuilt
(432 edges, no warnings added), `studio-sampling.c` regenerated again (only the SHA line moved)
and the producer CLI rebuilt and re-passed both of its gates; `tools/decode-receiver/main.cpp` has
no build rule and was compiled by hand with the CLI's flags plus `-I include`, `-I common` and
`-I tools/mtmd` - see the commands file. Chunk 4.2 touched no engine (`engines/ds4`) source
and no llama source outside `ggml/`; it rebuilt every CUDA object in both flag sets and the eight
hand-compiled programs were relinked against the rebuilt objects. Rebuild before trusting a gate
that runs an engine binary.
`python3 -B scripts/release_check.py`
reports 0 findings, and **no hook runs it here**: `git config --get core.hooksPath` is empty and
`.git/hooks` holds only the samples, so the check has to be run by hand before committing. Do not
wire `.githooks` up on this box without changing the git identity - `release_check.py --identities`
requires `Front Eagle <fronteagle739@users.noreply.github.com>`, which the fork's history carries,
while this checkout's config is `Mac` and every port commit is `Mac`-authored. Python suite, with
`STUDIO_RUNTIME_ROOT` set (see the paragraph above): **685 passed, 2 failed, 38 skipped** on the
7.1e run. The one
stable failure is `test_yoco` (its preset reads `producer_prefill_sweep_tokens` as `auto`, not the
49152 it asserts), which chunk 8.1 owns, and this run's second failure was the moving process test
`test_disconnect_inside_reasoning_then_next_turn_appends_only` - which this session moved even in
alone-runs, on the working tree and on a worktree of the untouched HEAD alike (three alone-runs on
the baseline read pass, pass, fail), so the box decides it, not the tree. Chunk 7.1d read both
lines on the same tree: 686/1/38 and 685/2/38, the difference being
the same name. Chunk 7.1b
removed the four names this phase owned, including the two that had been attributed to a missing
`cosmos_windows` module - their real cause was that the suite's own command does not put
`vendor/studio` and `vendor/engines` on `sys.path`, which is what `slstack.native_broker.vendor()`
and the launcher's `library_pythonpath` do. Chunk 5.1 fixed four names that used to be in this list
and un-skipped 21 more; chunk 5.3 un-skipped three more and fixed none. Check a suspected
regression by running the same file in a worktree at the baseline
(`git -C build/wt-base checkout <sha>`) rather than by reading a count. Process tests flip
between runs - compare the set of names, never the counts. `envdocs --check` reports **2,186**
documented variables: chunk 4.3b dropped seven whose last reader sat in a folded guard, chunk 4.3c
dropped none, and chunk 7.1d dropped 71 and added 114 - the drops are the `DS4_ROCM_*` aliases whose
only reference is an argument the wrapper discards, and the additions are the wrapper reads the old
scan could not see. `metal_word_census` over `engines/ds4` now reads IDENTIFIER 0, ENV_NAME 306,
ENV_NAME_CITED 0, COMMENT 25 lines, STRING 0, and over the llama tree IDENTIFIER 0, STRING 0 and
COMMENT 37 lines (from 51) - chunk 7.1e took the seven cited names with the stub's parameters and
reworded the Mac peer's text; what survives of the COMMENT bucket is provenance on both trees.

Five hand-compiled programs cover what has no test target, all passing on this tree - and all
relinked and re-run after chunk 4.2b rebuilt every CUDA object, since four of them link
`libggml.a`/`libggml-cuda.a`:
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

Chunk 5.1's Linux gates are ordinary pytest files rather than hand-compiled programs:
`tests/test_cgroup_scope.py` (13 checks after 5.3 - the kernel refusing an oversized tree, a
grandchild's refusal taking the group, the survivor and stale-launcher rules, and the floor half:
whole-page rounding, the readback, the refusal where there is no memory controller, and
`memory.reclaim` proving that the reported honour is the kernel's actual answer) and
`tests/test_advisory_lock.py` (7 checks after 5.2 added the shared arm and the interrupted-wait
rule, one lock across two
processes, contended, blocking and shared) both run in the suite and need no GPU, no model and no
`STUDIO_RUNTIME_ROOT` - except the honour check, which needs a disk-backed fixture directory.
Chunk 5.2's are `tests/test_producer_runner.py` (22 checks: nothing runs
before admission, the never-resumed and closed-child retirements, SIGINT retirement, the force
fallback with a grandchild, `wait_for_exit` over the job, `/proc` module list, the volume and alias
refusals, lease sharing, `commit()`/`memory()` from `/proc`, NVML identity and capability) and
`docs/evidence/producer-launch-gate.py`, which drives the real CLI through the vendored `Producer`
and is the only gate that exercises the launch chain end to end; since 5.3 it also runs the
resident admission against that real producer - the job into a scope the gate created, the floor
written, the receipt written - and still ends at the reserve refusal with the scope removed. It
needs the CUDA headroom probe to say yes and `build/smoke/out` cleared. Chunk 5.3's are `tests/test_working_set.py` (8 checks
against real owned CPU children and real scopes: the floor lands where the job inherited it, a
foreign or missing scope is reported and left alone, the ceiling keeps its reclaim gap, an admission
below one page refuses, and a failed `after_resume` certifies the owned cleanup on Linux) and
`tests/test_combo_ram.py` (7 checks on the `/dev/shm` tier: bounded transfers, a second process
attaching by name and writing through it, exclusive creation, name validation, the stale-name reap,
the teardown that removes the name, and the lock receipt). Those two files' Windows classes stay
skipped here, and `docs/linux-port-commands.md` has the commands that read the receipt and the
partial-lock line directly.

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
   box or the change is at fault. The margin is thin and it moves within a session: on 2026-11-06
   `cudaFree(0)` and `cudaMemGetInfo` returned `out of memory` in the morning, reported 116 MiB free
   mid-session - which is when `docs/evidence/wf-io-selftest.cpp` and the no-model producer receipt
   smoke both ran to their documented outcomes - and returned `out of memory` again an hour later
   with nothing in the tree changed. The probe is committed at
   `docs/evidence/cuda-headroom-probe.cpp` (`docs/linux-port-commands.md` has the line); run it in
   the same session as the gate, and read a CUDA gate as *unverified* rather than failing when it
   says no. It said no again at 20:30 on 2026-11-06, so chunk 2.2e's host gates were the compile,
   the link, `./ds4 --help` and `./ds4 --metal`; `./ds4 --gpu-graph-test` was never run. It said
   **yes** on 2026-11-07 (215-227 MiB free, `cudaMalloc(8 MiB)` ok, with Strata at 29.7 GiB and the
   browser beside it), which is when chunk 5.2's producer launch gate ran to its documented
   reserve refusal - the first CUDA-touching producer gate to complete here. It said **yes** again
   later the same day at 298 MiB free, which is when chunk 4.2b's `wf-io-selftest` (a real CUDA
   context), `llama-server --list-devices` and both `--check-native-profiles` runs completed.
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
   not a merge of what is here. 333 GiB free on the NVMe as of 2026-11-07.
3. The box changed since the plan's earlier notes: kernel 7.2.14-arch1-1, **93 GB RAM** (not 30),
   16 cores, 333 GiB free on a 1.8 TB NVMe. CUDA 13.4.92, cmake 4.4.5, ninja 1.13.2 and
   Python 3.14.9. `docs/linux-port-plan.md`'s machine block carries the same figures now; re-read it
   rather than trusting a limit quoted anywhere else.
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
- A chunk that deletes a **directory** breaks `preprocess_equiv` on the *before* file: the
  HEAD version of `ggml-cpu.cpp` still has `#include "amx/amx.h"`, so extract the deleted headers
  into the before tree (`git show HEAD:<path> > build/41/amx/amx.h`) and pass `--include build/41`
  or the comparison dies on a missing include instead of proving anything. For a file whose *code*
  lines were removed, `preprocess_equiv`'s positional diff reports a shift, not a change - use
  `code_line_diff.py`, which reported `ggml-backend-reg.cpp` and `llama.cpp` at 0 changed lines and
  named exactly the deletions elsewhere.
- **A failed compile leaves the previous executable in place, and `./build/<tool>` then runs it.**
  Chunk 4.1's `provider-lease-selftest` link died on a missing include while an older binary of the
  same name reported `SELFTEST PASSED`. `rm -f` the output before compiling, or compile to a fresh
  name, and check that the binary is the one you just built - a green gate that predates your edit
  is worse than a red one.
- Hand-compiled selftests need the include directory the *header* lives in, not the one the test
  names: `provider-lease-selftest.cpp` includes `llama-dsv41-weights.h`, which is under `src/`, and
  `wf-extent-selftest.cpp` must compile `src/llama-weight-fetcher.cpp` itself **without**
  `-DGGML_USE_CUDA` - linking the CUDA-built object instead fails with "target CUDA device cannot
  be bound" when Strata holds the VRAM, which reads like a broken reader and is only contention.
- `collapse_conditionals.py` resolves a macro it can see; `#ifndef GGML_USE_CUDA` is *kept* on this
  build because the macro is not defined at preprocess time, so CUDA-only code stays guarded and a
  non-CUDA arm survives inside those guards. That is the conservative answer, not a miss - but it
  means a prune is not finished by the collapse alone, and `#if defined(GGML_USE_METAL)` (also
  unknown) is likewise kept. Say which arms the collapse could not decide.
- Chunk 4.2b found four **soundness** bugs in that tool, all fixed, and `docs/evidence/
  collapse-probe.h` is now the gate you run before trusting it on a tree: apply the tool to a copy
  of the probe and diff the preprocessed text of the copy against the preprocessed text of the
  probe - identical means the pass removed only what the preprocessor had already removed. What it
  used to get wrong: an unparsable condition was written back re-spaced instead of reported (the
  tokenizer had no comparison operators, so every `CUDART_VERSION >= X` guard was "unparsable" and
  the first pass on the CUDA tree wrote out 28 rewrite-guards, several of them the same condition
  re-spaced); a directive spanning continuation lines was treated as
  one line, so a rewrite flattened `#if A || \` into one line with the backslash inside the
  expression and a drop left the continuation behind; the dead-`#if`-with-live-`#elif` path dropped
  the `#endif` that closes the promoted guard - `vecdotq.cuh` ended up with four `#if` and three
  `#endif`, which only a per-file balance check catches; and the operand-drop patterns for a macro
  the build *defines* were inverted, so `false && X` lost its false operand and `X || true` its
  true one - both change a condition rather than shortening it. Nesting inside the surviving branch
  of a dead outer node was never visited.
- Run the collapse to a fixed point, and review its insertions line by line: promoting a `#elif`
  can leave a guard that is always true, which only collapses on the next pass, and a surviving
  `#endif // <old condition>` comment describes a condition the pass just shortened. 4.2b rewrote
  26 such comments by hand and resolved 8 `BLOCKED` nodes the same way.
- `scripts/port/cuda_live_equiv.py` is the equivalence check for a sweep that touches a **header**:
  `preprocess_equiv.py` preprocesses both sides with one include set, so the before file reads the
  after tree's `common.cuh` and the comparison proves nothing about the header you changed. This
  one gives each side its own tree's directories. On 4.2b's 20 changed files it reads 0 differing
  live lines for the two that do not include `common.cuh` and exactly one for each of the 18 that do
  - the deleted `#define MUSART_HMASK 40300`, the intended change and the only one.
  `--include` (repeatable, repo-relative) and `--defs` replace its llama-CUDA-backend defaults, so
  it also covers the engine's `cuda/mmq/` tier - whose build defines none of the `GGML_USE_*` names
  and needs `-Icuda/mmq -I.` with `-DDS4_CUDA_HAVE_MXF4=1`. Note what `-fdirectives-only` writes
  out: it resolves `#if`/`#elif`/`#else`/`#endif` and emits none of them, but it does pass `#define`
  and `#undef` through. So a fold's live diff is exactly the non-conditional directives it deleted -
  three lines on 4.3a (`#define MUSART_HMASK` and two `#undef GGML_*_GRAPHS`), and a reworded
  `#endif // <condition>` comment cannot show up there at all.
- For a CUDA sweep the machine code is the stronger gate and needs no GPU: `cuobjdump --dump-sass`
  the objects rebuilt in both trees. Two things always differ and neither is behaviour - nvcc's hash
  of the file contents inside every internal-linkage kernel name (`_GLOBAL__N__<hash>_…`, normalise
  it with `sed -E 's/_GLOBAL__N__[0-9a-f]+_/_H_/'`), and the `__LINE__` that `no_device_code()`
  prints, which appears as one 16-bit immediate per kernel instantiation and must move by exactly
  the number of lines the fold deleted above it. `--dump-ptx` is no help with
  `-gencode …code=sm_120a`: no PTX is embedded, and it prints eight lines of nothing.
  `nm --defined-only`, normalised the same way, is the host-side half of the check.
- That marker immediate is a **half-precision** value, so it only decodes to a line number exactly
  when the line fits fp16's mantissa (75 -> 60 and 326 -> 299 in `mmf.cuh`, 1970 -> 1885 in
  `fattn-mma-f16.cuh` did; the mmq tier's did not). For a sweep that touches a header included by a
  hundred template instances, the honest gate is: mask the marker instruction and require every
  other byte to match, and separately show that *only* that instruction differs. On 4.3c's llama
  tree 77 of 168 objects differed, all of them in the single
  `/*0050*/ HFMA2 R2, -RZ, RZ, 0, <imm>` prologue marker, and masking it reads 0 differing objects.
  Do that mask in Python, not `sed`: `s_…_…_` terminates at the first `_` in the replacement, the
  substitution then fails on every file, and the empty-vs-empty comparison reports 0 differing
  objects for a mask that never ran - the exact failure mode this file's own rule warns about.
- Warning-profile comparison: `git worktree add build/wt-base HEAD` then
  `scripts/port/warning_profile.py build/wt-base .`, which compiles the same TU list in both trees
  with the same commands and prints the texts with line numbers stripped. 2.2c: 26 vs 26 occurrences
  in the engine; 3.1e: 27 vs 27, 18 distinct, none added or removed. Renames should differ only in
  the renamed symbol's own warnings. A deletion chunk should lose occurrences and gain no *kind*:
  4.1 went 36 -> 34 with 19 distinct both sides, the two losses being the deleted Metal branches'
  own `-Wcast-qual` sites.
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

- Sweep inputs a later session may need to re-run are committed under `docs/evidence/`. Two of
  them are scripts rather than data: `collapse-promote.py` (promote a surviving `#elif` above a
  provably-dead prefix, using `collapse_conditionals`'s own evaluator and its in-scope gate, so a
  node whose first condition names no swept macro is left for another chunk) and
  `collapse-guard-comments.py` (comment-only, and it also strips a trailing comment from a dead
  `#elif` so the branch becomes decidable). Run both over the tree after the collapse, then re-run
  the collapse - all three must report nothing before the equivalence gates mean anything.
  `rename-2.2d.json` is chunk 2.2d's rename map, `rename-2.2e.json` is 2.2e's (both halves: the
  `DS4_GLM_METAL_` prefix entry and the seven exact macro entries, so re-running it is idempotent),
  `messages-2.2e.json` is 2.2e's message map, `platform-selftest.cpp` is chunk 3.1c's
  platform-layer program, `wf-io-selftest.cpp` is chunk 3.1e's reader program (io_uring against the
  pread baseline, byte-compared through the real `wf_*` pipeline), `wf-extent-selftest.cpp` is chunk
  3.1f's extent reader (33 checks, including a seven-part batch against four slots),
  `provider-lease-selftest.cpp` is 3.1f's lease program (14 checks through the public `inspect()`
  with no backend), `uring-flush-probe.cpp` is the 3.1f probe for the io_uring submit/wait rule,
  and `cuda-headroom-probe.cpp` is the two-call answer to "can this box make a CUDA context right
  now", which every CUDA gate in `docs/linux-port-commands.md` should be read against. Chunk 4.2b's
  sweep inputs are there too: `collapse-probe.h` (the collapse tool's regression gate) and
  `collapse-4.2-tokens.txt` (the 20 macro names its pass used, so the CUDA collapse can be re-run
  or extended). Chunk 5.2's
  `producer-launch-gate.py` is there too, and is run in place rather than copied to `build/`:
  `rm -rf build/smoke/out && mkdir -p build/smoke/out && python3 -B docs/evidence/producer-launch-gate.py`.
  Chunk 3.1g's program is not under `docs/evidence/` but in the test directory -
  `tests/dsv41_engram_early_io.cpp`, ported with the loader, 15 checks over a 208 MB synthetic table
  file, no GPU and no model - and chunk 3.1h's two are there too (`tests/dsv41_carry_release.cpp`,
  which was red before the change, and `tests/pinned_carry_memory.cpp`), because the tier is engine
  code with a test source rather than a throwaway program. Copy them to `build/` and compile them; `docs/linux-port-commands.md` has the exact `g++` lines and the no-model producer
  smoke recipe. Anything written under `build/` is scratch and is not in git; chunk 2.2b's map was
  never saved, so its rename is only reconstructable from its commit message.
- A **text** chunk (comments, printed messages) is outside `preprocess_equiv`, which keeps comments
  and therefore reports every reworded line plus the shift of everything below it. Use
  `scripts/port/comment_code_diff.py`: strip comments from both sides and compare code lines, so a
  comment-only chunk reports 0 changed lines and a chunk that also deleted declarations reports
  exactly those. Check `__LINE__` first - `ds4.c` has none, `ds4_cuda.cu` has one at line 4421, so
  edits above it would move a reported number and edits below it must stay line-count neutral.
  `scripts/port/string_text_replace.py` is the message sweep: pairs apply inside string literals
  only, longest first, and a literal naming `DS4_*METAL*` is skipped. `scripts/port/metal_word_census.py`
  is the gate and the count; it classifies by position, and `--strict` fails on any code-position
  mention. All three are committed, all three take repo-relative paths.
- The engine has no test target and is not in `compile_commands.json`, so an engine TU is compared
  by hand: `make -C build/wt-base/engines/ds4 ds4.o ds4_ext.o ds4_cuda.o ds4_cuda_ext.o CUDA_ARCH=sm_120`
  and the same in the working tree, then group the warnings by TU (split the log on `-o <name>.o`),
  strip file/line/column, `sed 's/[0-9]+/N/g'`, sort and diff. `make -C engines/ds4 ds4` builds
  several TUs at once and links; `ds4-decode.so` is the `-DDS4_EXTERNAL_DECODE=1` compilation. A TU
  whose source did not change is not rebuilt, so it warns in the baseline log and not in the new one
  - that is not a lost warning. A worktree for the baseline has to be *moved* to the current HEAD
  (`git -C build/wt-base checkout <sha>`) before a comparison, or it compares against the previous
  commit and the earlier chunk's edits show up as changes.
- When subagents are available, `adversarial-code-review` is worth running after a deletion chunk -
  but give it the equivalence commands as evidence. The 2.1 review produced three findings quoting
  text that was not in the working tree (it read the diff base rather than the tree), yet its second
  pass caught three real problems in what the commit *claimed*. Reviewers check claims, not just
  code: re-verify the wording in docs and messages, not only the C. Note that a subagent can be
  stopped mid-run and return no output; if a survey agent dies, do the greps yourself.

## Behaviour changes landed this session

Chunks 3.1c through 3.1h's behaviour changes (the CLI running on Linux, the POSIX receipt identity
fields, `--carry-storage hybrid` without a drive letter, SIGINT/SIGTERM cancellation, the live
carry-resume RAM floor, no process-memory limit in `host_memory::begin()`, the fetcher's io_uring
reader and `LLAMA_WF_IO_URING=0`, `WF_MODE_AUTO` probing cuFile, buffered `io_direct`, the deleted
`--ds-gpu-ring-slots`, `--ds-submit-threads` and `--ds-zero-padding-only`, the sticky failed-read
accounting, the POSIX provider lease, the bounded relay as the strict profile, the named-extent
reads into a compute bank, the RAM expert tier measured from `/proc`, the move of the relay's
staging tier and the extent window into the provider's host plan, the Engram loader's io_uring
queues with its serial `pread` baseline and `LLAMADECODE_ENGRAM_IO_URING=0`, the deleted
`LLAMADECODE_ENGRAM_IO_PRIORITY`, the loader's own descriptor checked against the provider's lease,
`load()`/`load_early()` taking a path and an `int` fd, the provider's three Engram arms being open,
and the carry tier's best-effort `mlock` with `CARRY_RAM_LOCK_PARTIAL`) are recorded in the plan's
entries for those chunks and in their commit messages. Chunk 2.2e's (173 printed lines reworded so
nothing names the retired backend, seven `DS4_METAL_PROFILE_*` macros renamed to `DS4_GPU_*`, three
dead declarations deleted, no environment name added or dropped) are in its four commit messages and
in the plan; its gate is `python3 scripts/port/metal_word_census.py --strict`, not a grep, because
`\bMETAL_` cannot see an identifier with a prefix before it.

This session's (chunk 7.1e, one code commit - engine `ds4.c` and llama-tree comment/string text):

- **The Mac peer's text is out of the llama tree and `decode-receiver`, and what remains says so
  in the past tense.** 51 comment lines and 3 printed strings became 37 provenance comment lines
  and 0 strings: the present-tense Metal claims (`routes compute to Metal`, `Only enable on Metal
  (Mac)`, `including Metal`, the mega-launch header naming the deleted `metal_hidden_buf` field,
  the llama.h split-graph notes) are CUDA/CPU text or `retired`/`deleted` provenance; the
  measurement-history comments (the E3-filter narrative, the rope blob's `ds4_metal.m` provenance,
  `metal/dense.metal`'s arithmetic citations, `common.cuh`'s Moore Threads heading and the 22
  CDNA/RDMA/HIP lines) were read and kept - they explain the host-side `cc` selection this port
  keeps. `mma.cuh`'s `#endif // defined(GGML_USE_HIP)` now quotes the guard it closes
  (`__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA`; upstream's comment never matched it),
  `fattn-new-mma.cu`'s `GGML_USE_HIPBLAS` comparison line is deleted, and the ten 4.2-era
  BLAS/AMX/Vulkan/OpenCL/SYCL/MUSA mentions lost their parentheticals or went past-tense. The
  three strings - two `ds4split` half-proof diagnostics and the `LLAMA_PD_CACHE_CAPTURE` warning -
  are reworded from the committed map `docs/evidence/messages-7.1e.json`; the warning's quoted
  `consumer` text moved its env row's evidence, so the table regenerated in the same commit.
  `decode-receiver`'s fifteen present-tense "Mac" comment lines describing the daemon's own role now say
  "receiver" - this binary IS the receiver now - while dated measurements and quoted historical
  text keep their era's names. Nothing renamed an environment name.
- **The stub's dead parameters are gone.** `gpu_graph_ported_m5_decode_feature_enabled()` in
  `ds4.c` returned false and `(void)`-discarded both parameters; it takes none now and its three
  call sites dropped the six `DS4_METAL_*_FUSE` literals, so `metal_word_census` reads
  `ENV_NAME_CITED 0` over `engines/ds4` (from 7) with no row added or dropped - documenting them
  would have been wrong because nothing reads them. Two comment lies went with them: the
  `DS4_METAL_DISABLE_*_FUSION` wildcard sentence (now names `gpu_graph_env_flag()`) and the graph
  dump hook's "or `DS4_ROCM_GRAPH_DUMP_PREFIX`" clause - that name is a discarded wrapper argument
  whose row 7.1d deleted.
- **`scripts/release_check.py`'s working-tree scan no longer skips `engines/mlx-serve/`** (1.10a
  deleted the gitlink, so the skip hid any file that path ever reappears in); the pinned-OID
  branches in `scan_index()` and `scan_history()` stay, because historical trees really contain
  it. `release_check` still reports 0 findings.
- Gates: census `--strict` IDENTIFIER 0 / ENV_NAME_CITED 0 / STRING 0 over both trees; every
  comment-only file reports 0 changed code lines in `comment_code_diff.py` against a worktree of
  HEAD, and the changed lines are exactly the 3 reworded strings plus the stub's signature and
  call sites (14 removed, 4 added); no edited file contains `__LINE__`. `ds4.c`'s three
  compilations warning-match the baseline (23, 11, 5 occurrences; 20, 8, 3 kinds). The engine's
  five binaries and `ds4-decode.so` relinked; `./ds4 --help`, `./ds4 --metal`, `make
  test-batch-io` and `make test-decode-library` pass. The llama tree rebuilt whole (432 edges);
  `studio-sampling.c` moved only its SHA line; the producer CLI passes `--help` and
  `--check-native-profiles` (headroom probe said yes at 392 MiB). `envdocs --check`: 2,186 names,
  set unchanged, 606 rows re-anchored. Suite 685/2/38 - `test_yoco` plus the moving
  `test_disconnect_inside_reasoning_then_next_turn_appends_only`, which moved in alone-runs on
  the untouched baseline worktree too.

Before that (chunk 7.1d, one commit - `f197552`, no engine or llama source touched):

- **The environment table is now what the engine actually reads, and it is both bigger and
  smaller than it was.** `inventory()` follows a read through a wrapper helper and refuses one
  the wrapper discards, so 114 names gained rows and 71 lost them: 2,143 documented variables to
  **2,186**. Nothing a recipe sets was renamed. What changed for a user: the 114 newly documented
  names are launcher-owned, so an inherited `DS4_METAL_DISABLE_STREAMING_READAHEAD=1` in the shell
  is now removed with the `Environment override:` warning and only `env_overrides`/`ds4_env`
  reaches the engine; the 71 dropped `DS4_ROCM_*` aliases are the reverse - no row, so an
  inherited value passes through untouched and nothing reads it, which is what it always was.
  `engines/mlx-serve/` references and their exemption are gone (`PATH`, `HOME`, `HF_TOKEN` stay,
  documented from trees this checkout has). `docs/VARIABLES.md` is regenerated from the new table.
- **The claim that was wrong**: the census's `ENV_NAME_CITED 118` was read as "118 live knobs the
  table omits". Half of that pair-class was the opposite - `glm_graph_env_present(
  "DS4_ROCM_X", "DS4_METAL_X")` is `(void)rocm_name; return getenv(cuda_name) != NULL;`, so the
  old line-bound scan documented the ignored spelling and missed the live one. `inventory()`'s own
  comparison (both scans over the same tree) shows 0 names lost and no recorded reference removed
  except the four this chunk decided on. `ENV_NAME_CITED` is 0 now - chunk 7.1e deleted the
  stub's parameters and reworded the wildcard comment that made the seven.
- `inventory()` is ~11 s rather than ~3 s, and `tests/test_env_inventory.py` (12 checks) is the
  gate for the passes; `discarded_alias_reads()` and `unresolved_wrapper_reads()` are the two
  things it cannot decide, printed rather than guessed at.

Before that (chunk 4.3c, two commits - `41bc18a` the engine's `cuda/mmq/`, `a50a7bb` the llama
tree's `ggml/src/ggml-cuda`):

- **The AMD and Moore Threads compute-capability vocabulary is gone from every preprocessor
  directive in both trees, and no kernel changed.** The 22 names in
  `docs/evidence/collapse-4.3c-tokens.txt` are undefined everywhere since 4.2b and 4.3a removed their
  last definitions, so pinning them folded 1,020 lines out of the engine tier and 1,799 out of the
  llama tree, and `docs/evidence/collapse-promote.py` resolved the 15 dead-prefix-with-live-`#elif`
  nodes the collapse declines (6 in the engine, 9 in the llama tree) while
  `docs/evidence/collapse-guard-comments.py` rewrote 81 + 145 trailing `#endif`/`#else` comments that
  quoted conditions the pass had shortened. Four `fattn-mma-f16.cuh` guards needed the dead
  parenthesised group deleted by hand, and three `#else` lines were comment fixes wearing a
  decision's clothes. Nothing a user can observe changed: the engine's nine objects are SASS-identical at
  `sm_120` except nvcc's content hash and the one `__LINE__` marker the fold moved by exactly the 309
  lines `mmq.cuh` lost, and the llama tree's 168 objects differ only in the unused-instantiation
  marker instruction (masking it makes all 168 byte-identical), with `nm` matching on all 168.
  No environment name moved; twelve rows had their `path:line` re-anchored.
- **The host-side half stays, deliberately.** `amd_mfma_available(cc)`, `amd_wmma_available(cc)`,
  `GGML_CUDA_CC_IS_CDNA(cc)` and siblings, `ds4_mmq_gfx1151_flag("DS4_ROCM_MMQ_TIGHT_NCOLS", cc)`,
  `mmvq.cu`'s `MMVQ_PARAMETERS_*` chooser, `ds4_q4k_half_common.cuh`'s `mmf_get_max_block_size()` and
  `common.cuh`'s `GGML_CUDA_CC_CDNA*`/`RDNA*` constants are plain C on a runtime `cc`, and guards
  keyed on `__CUDA_ARCH__` alone are undecided by design. Removing those is removing kernel selection
  and would take `DS4_ROCM_MMQ_TIGHT_NCOLS`'s last reader with it; that is a decision with runtime
  gates, not a sweep, and this port does not need it. 30 comment lines in `engines/ds4` and 22 in the
  llama tree's `ggml/src` still named those architectures - chunk 7.1e read them and kept them: they
  explain the retained `GGML_CUDA_CC_*` constants and runtime selection code, which is correct text
  about platforms nothing builds for.

Before that (chunks 4.3a and 4.3b, one commit each):

- **The engine's vendored mmq tier loses the HIP and MUSA branches the deleted backends defined.**
  18 tokens, 13 files, 426 deletions and 42 insertions to a fixed point, plus `vendors/hip.h`
  (308 lines). Six `BLOCKED` nodes resolved by hand - the same six shapes chunk 4.2b met - and 19
  trailing `#endif`/`#else` comments rewritten to the surviving condition. `MUSART_HMASK` goes with
  its last guard; `ds4_ggml_stubs.h` loses its two `#undef`s of macros nothing defines; the engine's
  HIP allocation-pool arm goes, and with it the only reference to `../../ds4_rocm_memory.h`, a
  header this checkout does not contain (neither does `vendors/musa.h`, which a dead branch named).
  No environment name moved: the tier's 23 `DS4_*` reads are all outside the folded branches, so
  six rows were re-anchored and 2,150 rows stand. Nothing a user can observe changed - the nine
  CUDA objects rebuilt at `sm_120` give byte-identical SASS except nvcc's content hash in the
  internal-linkage kernel names and the `__LINE__` `no_device_code()` prints. `cuda/mmq/VENDOR.md`
  is new; the Makefile and `ds4_cuda.cu` pointed at it and it had never existed.
- The CUDA headroom probe said **yes** this session (385 MiB free with Strata at 29.8 GiB and the
  browser beside it), but nothing in either chunk needed a context: both were proved by compiling,
  not by running on the device.
- **The engine's host layer stops compiling for a ROCm build that never existed.** 152 directives on
  `DS4_ROCM_BUILD` over ten files, 1,442 deletions and 39 insertions, one node hand-resolved (a
  `#ifdef DS4_ROCM_BUILD / #elif !defined(DS4_NO_GPU) / #else` whose `#elif` is the CUDA arm and
  whose `#else` is the CPU arm - promoted, not deleted). Seven env rows went with it, 2,150 down to
  2,143: five `DS4_ROCM_*` and the two `DS4_GLM_*` aliases of a folded pair. The other four names the
  fold removed were wrapper-only and the table never carried them, and 93 of the 103 `DS4_ROCM_*`
  names the engine cites are read outside those guards - live reads this chunk left alone, which is
  why the count of dead knobs is nine and not seventy-eight.

Before that (chunk 4.2, two commits `9125df4` and `eaccc5e`):

- **Nine ggml backend trees go, and nothing that was built changes.** `ggml/src/ggml-blas`,
  `ggml-cann`, `ggml-hip`, `ggml-musa`, `ggml-opencl`, `ggml-webgpu`, `ggml-zdnn`, `ggml-zendnn`
  and `ggml-virtgpu` (68 files) with their seven public headers, 32,327 lines over 75 files. None
  of them was compiled here: every `ggml_add_backend()` call behind them tested an option that
  defaults OFF and that `build.sh` never passes, and `GGML_CANN` was never an option at all - only
  the deprecated `LLAMA_CANN` mapping could have set it, so that mapping goes with the tree.
  `ggml/CMakeLists.txt` loses the options and sub-options, five `GGML_PUBLIC_HEADERS` entries and
  the APPLE `GGML_BLAS` default split; `GGML_ACCELERATE` stays because its only reader is the APPLE
  branch of `ggml-cpu/CMakeLists.txt`. `ggml-backend-reg.cpp` loses seven include guards, seven
  registration blocks and seven `ggml_backend_load_best` names - `cuda` and `cpu` are the only
  names left, so a build with `GGML_BACKEND_DL=ON` no longer looks for the others. 12 env rows
  lost their last reader and went: 8 `GGML_CANN_*`, 3 `GGML_OPENCL_*` and
  `GGML_REMOTING_USE_APIR_CAPSET`, 2,162 documented variables down to 2,150. Nothing went for its
  name - `GGML_OP_OFFLOAD_MIN_BATCH` is read in `ggml-cuda/` and `llama-context.cpp` too, so it
  stays and only its `ggml-cann` read reference went.
- **The CUDA backend's HIP and MUSA branches fold with the trees that defined them.**
  `ggml-hip` and `ggml-musa` had no kernels: each globbed `../ggml-cuda/*.cu` and compiled it with
  hipcc or the MUSA clang, defining `GGML_USE_HIP`/`GGML_USE_MUSA` and their sub-options. 22 files,
  1,419 deletions and 58 insertions, to a fixed point. `vendors/hip.h` and `vendors/musa.h` (425
  lines) are deleted - the branches that included them were their only references - and
  `MUSART_HMASK` with them. Eight `BLOCKED` nodes were rewritten by hand, always to the branch the
  compiler had already kept, and 26 `#endif // <condition>` comments were rewritten to the
  surviving condition. No environment name moved with it; 29 rows had their `path:line` re-anchored
  inside `ggml-cuda`. `metal_word_census --strict` passes over the llama tree for the first time
  since chunk 7.1c made it trustworthy.

Before that (chunks 7.1a, 7.1b and 7.1c, one commit each):

- **The Mac and paired docs surface is deleted, and the front page describes one Linux
  machine.** 86 tracked files, 67,638 lines: the six `MACDECODE` rounds, the four
  `macdecode-r*` receipt directories, `macdecode-measurements.json`, `TWO_MACHINE.md`,
  `YOCO_PAIRED_QUALIFICATION.md`, `YOCO_PAIRED_CPU.json`, the October 27 Mac build receipt,
  `examples/engine_environment/mac-*.json`, `scripts/setup-mlx.py`,
  `patches/mlx-core-ring-listeners.*` and ten test sources whose `#include`s named
  `engines/ds4-mac`. `README.md` no longer offers Mac decode, a paired route, `slstack mac`,
  the Mac FP4 file, the "Optional Mac" requirement or the four MLX-decoded model families;
  its inherited numbers now say plainly that the Linux fork has re-measured none of them. Two
  consequences worth knowing: the DSpark drafter row survives because `ds4` on Linux has
  DSpark, and the converter's `--expert-layout ggml-mxfp4` option is documented as something
  nothing here loads. `BACKEND_SETUP.md`, `BACKEND_SOURCE_AUDIT.md`, `SUPPORTED_MODELS.md`,
  `CONE_STATE.md`, `WALKTHROUGH.md` and `examples/single-machine/README.md` were rewritten
  against what the code actually does now: two backends (`ds4`, `llama-server`), `build.sh`,
  the `.sh` launchers, the refusal `check.py` and `router.py` raise for a `general` config,
  and the fact that `start-dsv41-single-*-ultra-yoco.sh` is approximate keep 140 while the
  plain launchers are exact cone - the old text claimed the opposite. `docs/CONE_STATE.md`
  keeps the state-version contract, which `slstack/yoco.py` and `slstack/native_broker.py`
  still enforce, and says outright that the paired proof receipts are gone, so those numbers
  are recorded nowhere. 26 measurement docs gained an inherited-history label under their
  heading instead of being deleted; `PERFORMANCE.md` lost its paired section, its MiMo Mac
  dial table, its Mac decode column and its two-machine memory row.
- **The four suite names Phase 7 owned pass, and two scripts stopped naming `engines/ds4-win`.**
  No production behaviour changed. `test_streamed_recipe_argv` calls `vendor()` before its
  `studio` import and hands the trainer the PYTHONPATH the launcher hands it; `test_dsv41_recipe`
  passes the same to its child; `test_h3_video_delivery`'s FFmpeg marker case is the Windows arm
  and is skipped here, with a new POSIX case asserting that readiness is what PATH reports and
  that a Windows `installed.json` marker neither creates nor breaks it; `test_shared_prefill_profile`
  lost its `slstack.mac` half and calls the export path by its current name.
  `scripts/export_ds4_sampling.py` read `engines/ds4-win/ds4.c` and died on `FileNotFoundError`;
  repointed at `engines/ds4`, regenerating `tools/dsv41-prefill/studio-sampling.c` moves only the
  recorded source SHA, with all nine exported sampler functions byte-identical, which is the proof
  the engine rename never moved the sampler. `scripts/build_combo_runtime.py`'s staged preset lost
  `ds4-win` in its name and its empty `mac={}` block.
- **The Metal census could not see a code identifier after a line comment, and the llama tree's
  answer moved.** `scripts/port/metal_word_census.py` left line-comment state open past the
  newline, so identifiers on later lines were filed as COMMENT and `IDENTIFIER 0` proved nothing.
  The engine's numbers are unchanged (so chunk 2.2e's gate stands); over the llama tree it now
  reports IDENTIFIER 1, STRING 3, COMMENT 51 lines and `--strict` fails - that commit's message and
  this file both said STRING 4, which is a mention/line mix-up re-measured in chunk 4.2. The
  identifier is `USE_METAL_GUEST_SUPPORTS_OP` in `ggml/src/ggml-virtgpu/ggml-remoting.h`, and that
  tree - plus blas, cann, hip, musa, opencl, webgpu, zdnn and zendnn, 68 files and about 1.5 MB in
  total - was chunk 4.2's: none was built, and their env rows survived only because `inventory()`
  scans unbuilt code. `llama_split_graph_create`'s refusal line still printed "need CUDA or Metal" even though
  chunk 4.1's message and this file both claimed it had been reworded; it is reworded now, with
  `preprocess_equiv` proving the CUDA build's live text is unchanged and a CPU-only preprocessing
  showing exactly the one intended message line.

Before that (chunk 4.1, one commit):

- **`ggml` builds CUDA and CPU only.** Six backend trees and their public headers are deleted
  (`ggml-vulkan`, `ggml-sycl`, `ggml-hexagon`, `ggml-rpc`, `ggml-cpu/amx`,
  `ggml-cpu/kleidiai`), plus `ggml/src/ggml-metal`'s sources and the `kleidiai` and `amx`
  directories under `ggml-cpu/`. `ggml/CMakeLists.txt` no longer offers `GGML_METAL`,
  `GGML_VULKAN`, `GGML_SYCL`, `GGML_RPC`, `GGML_HEXAGON`, `GGML_CPU_KLEIDIAI` or the four AMX
  feature options, and neither does `ggml-cpu/CMakeLists.txt`, which also lost the
  `sapphirerapids` variant and the KleidiAI `FetchContent`. `ggml-rmw-vec.cpp` is now built
  unconditionally - it was compiled on every non-Vulkan build already.
- **71 environment rows are gone**, and the table is at 2,162 documented variables from 2,234.
  Every one lost its last reader in the deleted trees; nothing was dropped merely because its
  *name* names a deleted backend. **`SPLIT_DECODE_EVENT_SYNC` stays and is now inert** - its
  branch and the default branch both call `ggml_backend_synchronize`, so the knob changes nothing
  while the read keeps the row honest.
- **A CPU-only llama no longer offers `--rpc`** (`common/arg.cpp` lost `add_rpc_devices` and the
  option), and `llama_split_graph_create`'s no-backend warning was to read "this fork builds
  CUDA" - it did not; chunk 7.1c reworded it.
  Both were already unreachable here: `llama_supports_rpc()` asks the registry for "RPC", which
  no longer exists, and the warning sits in `#ifndef GGML_USE_CUDA`.
- **The Mac peer's split-graph API is now eleven stubs with no caller in this fork**
  (`submit_layer_gated` and `launch_mega` return -99, the four host event helpers and
  `synchronize` do nothing, `attn_out_buf`/`hidden_buf` return NULL,
  `submit_split_layer_async`/`wait_split_layer_async` start nothing and report -99).
  `llama_split_layer_params` lost its five `metal_*` fields and
  `llama_context::get_metal_backend()` is gone. **That is an ABI change to a public struct**,
  and nothing in this checkout constructs those fields; deleting the stub surface itself belongs
  to the chunk that prunes the two-machine API.
- `ggml_cpu_has_amx_tile/_bf16/_fp16` stay (they are `--version`'s CPU feature flags, not a
  backend); `ggml_cpu_has_amx_int8` is gone because its only reader was the deleted AMX buft.

Before that (chunk 5.3, one commit):

- **Resident admission runs on Linux.** `working_set.protect_working_set` is one function now -
  `combo_ram`'s verbatim copy is gone, and `combo_native`/`combo_producer` import the surviving
  one - and its Linux arm writes `memory.min` on the scope the job inherited (`job.scope`), in
  whole pages, kept below the scope's ceiling by the 64 MiB metadata headroom. The budget
  arithmetic, the initialization wait and the receipt keys are unchanged; the row gains
  `mechanism` and a `resident` object. **Anything reading `resident-admission.json` or
  `working-set.json` on Linux now sees `resident.honored` and `resident.blocked_by`** and, where
  the ancestors carry no floor, a `DS4COMBO_RESIDENT_FLOOR_INERT` line. The floor bounds the
  guarded tree, controller included, and a scope this stack did not create (`slstack-` prefix) is
  reported and left untouched - `resident.applied` is then false and nothing is written.
- **The checkpoint tier is `/dev/shm` now.** `LockedRam` creates the section
  `O_CREAT|O_EXCL` at 0600, truncates to the admitted capacity, maps it `MAP_SHARED`, and locks
  it in whole-page chunks that halve at each refusal - the engine's `lock_pages` rule - reporting
  the shortfall as `DS4COMBO_LOCKED_RAM_PARTIAL` rather than refusing, because an unprivileged
  Linux process cannot lock a real tier at all here. A full `VirtualLock` failure is still a
  refusal on Windows. **A lock past `RLIMIT_MEMLOCK` answers ENOMEM (12), not EAGAIN**, on this
  kernel. `attach(name,capacity)` replaces the child-side `OpenFileMappingW` + `MapViewOfFile`
  and returns the same bounded-transfer surface (`SharedSection`, which `LockedRam` now shares).
  The name carries the creating pid and `reap_stale_sections()` removes owned names whose creator
  is gone, because a tmpfs name outlives its process where a section handle did not; `close()`
  unlinks it. Transfers stay <=64 KiB, `sha256` and `lease()` are unchanged, and no environment
  variable was added or dropped.
- **The volume refusal was left alone.** It refuses RAM-backed *producer* paths - weights,
  receipts, outputs - and the checkpoint never goes through `_Native._open` or `volume()`, so the
  exception is written in `combo_ram.py`'s docstring and in `producer_runner.py`'s refusal list
  comment instead of narrowing the rule.

Before that (chunk 5.2, one commit):

- **A ULTRA or mode1 producer can be launched on Linux.** `slstack/producer_runner.py` is the
  counterpart of the vendored Windows runner with the same `_Native`/`_Job` surface, and
  `default_runner()` chooses between them; the four combo call sites, the three combo gate
  scripts and the vendored `Producer`'s default runner now go through that seam. The vendored
  runner file is untouched - only `dsv41_session_process.py`'s two-line default moved, recorded
  in `PROVENANCE.json`'s `local_patches`, and no hash pin moved with it.
- **The child is contained before it runs, and `resume()` is what releases it.**
  `slstack/producer_launch.py` is the child that waits: `setsid`, report its pid, then exec the
  engine when the controller writes one byte. Its `PR_SET_PDEATHSIG` retires a child that was
  never admitted, and `execv` clears that setting - so `release_kill_on_close()` records a
  release rather than performing one, and after resume the tree survives its controller, which
  is what `combo_producer.admitted()` needs. Ctrl-Break is SIGINT to the group and
  `TerminateJobObject(137)` is SIGKILL to it, **so a force-killed producer's receipt now reports
  `exit_code: -9` where Windows recorded the 137 it passed in**; `active()` is the group's live
  members, so `owned_job_empty` means the same thing.
- **The producer no longer breaks away from the guardian's ceiling.** The retired job was created
  with `CREATE_BREAKAWAY_FROM_JOB` and carried no memory limit; here the child stays in the cgroup
  the controller inherited, so `host_budget_gib` applies to the producer tree from its first
  instruction. `job.scope` records it, and `job.pi.process` no longer exists - which is why 5.3
  is the chunk that can finish the resident admission.
- **Leases are advisory, and the volume rule is the provider's.** `journal.advisory_lock` gained
  `shared=True`, the counterpart of `FILE_SHARE_READ`: many read leases coexist, an exclusive
  holder refuses them. `_Native._open` refuses the same statfs magics `require_local_volume`
  refuses, plus traversal, `':'` in a component, symlinks (`O_NOFOLLOW`'s `ELOOP` is reported as
  the reparse refusal) and a type mismatch, and holds the ancestor descriptors for the block.
  What no Linux mechanism reproduces is the *mandatory* share-mode refusal: a process that never
  calls flock is not refused the file, so the identity rechecks remain the real rule.
- **The producer's library directories now reach the loader.** `launch_environment()` puts them
  in `LD_LIBRARY_PATH` beside PATH, so `producer_library_directories` means something here.
- No environment variable was added or dropped (`envdocs --check` still reports 2,234 documented
  variables), no engine source changed, and the geometry defaults in `choose_scan()` are still
  the Windows RAID-0 tuning.

Chunk 5.1 (four commits), whose receipt changes are still worth knowing:

- **The guardian's memory cap is enforced by the kernel, not polled.** `guard.py` created the
  child suspended, assigned it to a Job whose `JobMemoryLimit` was `job_commit_gib`, and resumed
  it; on Linux nothing replaced that, so the only rule was a 50 ms RSS poll.
  `slstack/cgroup_scope.py` creates a cgroup v2 subgroup capped at `job_commit_gib` with
  `memory.swap.max=0` and `memory.oom.group=1`, and the child is moved in between fork and exec,
  so the ceiling holds from its first instruction and its descendants inherit it. On Linux
  `job_commit_gib` *is* `host_budget_gib` (`commit_budget_gib` lost its WDDM arm), so an
  over-budget allocation is refused rather than noticed afterwards: the receipt gains
  `memory_scope` (path, ceiling, peak, OOM counters, membership) and `scope_cleanup`, and a
  kernel termination now reads `HOST MEMORY LIMIT: the kernel terminated the guarded tree at the
  scope ceiling of N bytes (peak M)` instead of the `exit_code: -9, reason: null` it used to
  leave. A scope that still has members at teardown is left in place, still capped, and
  `scope_cleanup.removed` says so. `slstack/guard_child.py` - the shared-guard path's launcher,
  which had no cap at all here because `GuardedJob` builds a `WindowsJob` only on nt - opens the
  same scope and writes what it held beside the spec, which `shared_guard` merges into its
  receipt.
- **Admission is serialised on Linux.** `lease.gpu_lease` took its ledger lock only on nt, so two
  instances of this stack could each read an empty hold file and both write an ACTIVE hold for one
  GPU. One `journal.advisory_lock` (flock) now serves the journal, the live sampler control, the
  requant lock and the hold ledger; a contended non-blocking attempt raises `LockUnavailable`,
  which `gpu_lease` maps to `AdmissionRefused` - the guardian's `gpu_wait_seconds` loop already
  retries that. It is no longer a `PermissionError`.
- **Receipt shape.** `memory_scope`/`scope_cleanup` are new; `peak_private_commit_bytes` is gone
  from the per-process rows (the constant zero that psapi's private counter became on Linux);
  `num_page_faults` stays null because psutil exposes faults per process only on Windows;
  `paging_interpretation` says swap, not pagefile; `paging_before`/`paging_after` are
  `/proc/meminfo` totals under the same key names; and `diagnostic_snapshot()`'s four paging names
  are cumulative-since-boot counters with their units, not PDH per-second rates. **Anything
  grepping guard receipts for the PDH rates or for `peak_private_commit_bytes` reads nothing.**
- **Staged runtimes carry Linux names.** `PRODUCER_FILES` is `('ds4-decode.so',)` (the two
  DirectStorage DLLs exist nowhere here), the launcher is `ds4-server`, the producer builds are
  `combo-prefill` and `combo-prefill-<profile>`, native extensions must be `.so` (their
  environment names still end `_DLL` - engine reads, kept), `require_runtime_python` compares
  interpreter paths exactly rather than case-insensitively, and only TMPDIR and CUDA_CACHE_PATH
  are remapped now. `stack_config`'s refusal says `build.sh`. The fetcher's real artifact is
  `build/llama/bin/libslstack-weight-fetcher.so`, which neither the `{weight_fetcher}` default nor
  `install.py`'s binary check used to name.
- **The vision bridge is Linux.** It refused non-nt before looking at anything, and
  `ds4x_vision_file` is exported by the Linux `ds4-decode.so` (verified: `nm -D` lists it and
  `ctypes.CDLL` binds it without touching CUDA). Library directories now go to `LD_LIBRARY_PATH`,
  the library is looked for beside the producer or in `producer_library_directories`, and the
  request no longer carries a field the worker could not use. Encoding a real image is still
  unrun - there is no qualified model.
- **Two vendored Studio lines changed and one pin moved with them.**
  `dsv41_session_process._note_tree_progress` reads `other_bytes` as the Windows-only field it is;
  `job_paging.JobPaging.sample` reads zero swap for a member whose `/proc/<pid>/status` no longer
  has a VmSwap line, which fixes an intermittent `StopIteration` the monitor thread reported as
  the job's stop reason. Both are recorded in `vendor/studio/PROVENANCE.json`'s `local_patches`,
  and `shared_guard.py`'s pinned hash for `job_paging.py` is now
  `0387ed7fbade07d088c6149332c239f7ddfd3b2878f48d3f154c5d5742800c91`.
- No environment variable was added or removed (`envdocs --check` still reports 2,234 documented
  variables), no engine source changed, and the geometry defaults in `choose_scan()` are still the
  Windows RAID-0 tuning, untouched.

- Still Windows-shaped and untouched: `media_tools.py`'s pinned Windows FFmpeg download with
  `test_h3_video_delivery`'s `.exe` fixture (Phase 6/7.1), `scripts/perfexp_pair.py`'s
  `ds4-decode.dll` hash and its `decode_dll_sha256` receipt key (8.1), and
  `scripts/build_combo_runtime.py`'s preset name `... -> ds4-win decode` with its `mac={}` field
  (7.1). The producer launcher - `studio.dsv41_windows_runner._Native`, which refuses non-nt in
  its constructor - is ported (5.2): it stays as the Windows arm, and nothing selects it by
  import any more. `combo_ram.LockedRam` and `working_set.protect_working_set` are ported too
  (5.3); their `ctypes.WinDLL('kernel32')` calls are the Windows arms of modules that run on both
  platforms, which is what the `docs/linux-port-commands.md` gate of 8 lines means now.

## Gotchas

- **A per-file dedupe set is a name-loss machine.** Chunk 7.1d's first `inventory()` kept
  `references = set()` per file and skipped any reference already in it, so on a line with two
  reads (`getenv(A) == NULL || getenv(B)`) the second name vanished: 17 documented names "lost",
  which read like a regression in the new passes. Dedupe per name. `tests/test_env_inventory.py`
  asserts the tree's read set is a superset of the old scan's, which is what caught it.
- **`text.count('\n', 0, offset)` per hit is O(offset), and `ds4.c` has thousands of hits.**
  The wrapper pass took the inventory from 3 s to 32 s until line numbers came from one
  newline-offset list plus a bisect (~11 s). Compare against the *old* scan's runtime, not against
  nothing, before concluding a pass is too slow.
- **A call statement ends at a `;`, exactly like a prototype.** Skipping `name(...);` as a
  declaration deleted every real wrapper call chunk 7.1d wanted to find (`set_env_value("X", "1");`
  included). The test is the argument list's shape (`type name` pairs), not the terminator.
- **A wrapper's argument position is not a read.** `glm_graph_env_present("DS4_ROCM_X",
  "DS4_METAL_X")` is `(void)rocm_name; return getenv(cuda_name) != NULL;`, so a regex keyed on
  function names containing `env` documented the ignored spelling of each pair and missed the live
  one - which is why fixing this class both added and removed rows. `discarded_alias_reads()` is
  the report, and the same rule means a name at a *mapped* position is a read even when the value
  is thrown away by the caller.
- **A comparison of two scans has to run over the same tree.** `build/wt-base` sits at `41bc18a`,
  so a llama-tree read's line "moved" 91 lines and looked like a bug in the new pass; and
  `sys.path.insert(0, 'build/wt-base')` followed by `from slstack.envvars import inventory`
  imports the *baseline* module, which is how one run reported the new scan losing 17 names it did
  not lose. Load the baseline by file path and check which module you got.

- **A gate that reports zero is only evidence if you have seen it report non-zero.**
  `scripts/port/metal_word_census.py` classified a mention by character state, but its scanner
  never left line-comment state at a newline, so every identifier on a line after a `//` comment
  was filed as COMMENT and `--strict` passed on a tree that still had `USE_METAL_GUEST_SUPPORTS_OP`
  compiled. Chunk 7.1c fixed it. Four lines of probe text are the cheapest way to check a
  classifier: note that the census's `source_files()` skips any path with a `build` component, so
  the probe has to live outside `build/`, and `--list` is what prints the per-file breakdown.
- **`envvars.py:inventory()` scans `engines/` whether or not cmake builds it.** That is why nine
  unbuilt ggml backend trees still hold environment rows (8 `GGML_CANN_*`, 3 `GGML_OPENCL_*`, one
  `GGML_REMOTING_*`), and why deleting a backend, not disabling it, is what drops its rows.
- **A suite command that does not put the vendor trees on `sys.path` is a fact about the tests.**
  The documented run is `PYTHONPATH=.:tests`, so `import studio` and `import cosmos_windows` fail
  unless a test calls `slstack.native_broker.vendor()` first, and a *subprocess* running a vendored
  module by path needs the PYTHONPATH the launcher composes (`envvars`' `library_pythonpath`). No
  `tests/conftest.py` was added on purpose: the repo's established pattern is `vendor()`, and
  hiding the failure would also hide the production requirement.
- **`preprocess_equiv` on a llama TU needs the build's own flags**, taken from
  `build/llama/compile_commands.json`: `-x c++`, the five `-I` directories, and `/opt/cuda/include`
  while `-DGGML_USE_CUDA` is defined. Without them the before side dies on `ggml-cpp.h` or
  `cuda_runtime.h` and reports nothing. Dropping `-DGGML_USE_CUDA` is how you prove a change inside
  `#ifndef GGML_USE_CUDA` reaches only a CPU-only build.
- **Commit messages are claims, and this repo's own docs repeated a false one for a session.**
  Chunk 4.1's message said `llama_split_graph_create`'s refusal line had been reworded; it had not,
  and both the plan and this file copied that. Chunk 7.1c's message and both docs said the llama
  tree reads `STRING 4`; re-extracting that commit's tree into a directory the census does not skip
  (`runtime/`, not `build/` - the census drops any path with a `build` component) and running the
  fixed census reads STRING 3. Chunk 4.2's finding said `engines/ds4/cuda/mmq/VENDOR.md` "says so"
  about the mmq tier being a vendored copy - the file did not exist, and the Makefile and
  `ds4_cuda.cu` pointed at it too, so a claim in a doc can be a claim about a file nobody has ever
  opened. Check quoted text, quoted *numbers* and quoted *paths* against the working tree before
  recording them, the same way you check code.
- **A collapse that promotes a `#elif` leaves the file unbalanced unless its `#endif` stays, and
  nothing but a per-file `#if`/`#endif` count shows it.** `collapse_conditionals.py` used to drop
  the `#endif` on that path; `vecdotq.cuh` came out with four `#if` and three `#endif` and every
  equivalence tool still reported 0 differences, because a stray `#endif` is invisible to
  `-fdirectives-only` at the file's own level. Count the directives per changed file after any
  collapse (`docs/linux-port-commands.md` has the loop).
- **A `BLOCKED` line is a report, not a verdict - read which directive it names.** On the 4.3c
  tree three nodes are `#else // defined(AMD_MFMA_AVAILABLE) || ...`: an `#else` has no condition, and
  the tool only flags them because the trailing comment names a pinned macro. They are comment
  rewrites, not decisions. Three more came back `unparsable` for the same reason -
  `#elif defined(AMD_WMMA_AVAILABLE) //wmma instructions can handle 16x4 tiles...` - and the tool
  correctly refuses to *re-emit* such a condition. Deciding it is nonetheless safe, because the
  preprocessor ignores a trailing comment too: `docs/evidence/collapse-promote.py` strips one before
  evaluating a branch and puts it back on the guard it promotes. The tool itself still reports the
  node, which is the right division of labour - it never writes text it could not parse.
- **A claim about what a tool *cannot* decide is still a claim, and this file carried a wrong one.**
  4.2b and this file both said the AMD and Moore Threads vocabulary cannot be collapsed because its
  guards key on `defined(RDNA3)` or `__CUDA_ARCH__`. `__CUDA_ARCH__` is right and permanent; the
  other half was only true until 4.2b and 4.3a deleted the last `#define AMD_MFMA_AVAILABLE` and
  `vendors/hip.h` - after those, the names are undefined everywhere, `--never-defined` is sound, and
  the fold is mechanical (measured: 662 + 1,040 lines, SASS identical). Re-measure a limitation after
  every chunk that could have removed its premise.
- A fold of the engine's **host** layer has to be compared in both arms the engine compiles: most of
  those TUs build twice, once for CUDA and once with `-DDS4_NO_GPU` for the CPU binaries, and a
  guard dead in one is live in the other. `gcc -E -fdirectives-only` each side with **its own
  tree's** `-I` (`ds4.c` includes `ds4_gpu.h`, which the fold edits), and include `ds4_cuda.cu` even
  though its own source did not change - it includes two headers that did. Build the CPU *objects*
  for the warning comparison, never `make cpu`: that target links, and overwrites the CUDA binaries
  with CPU variants of the same names.
- **A collapse leaves its trailing comments describing conditions that no longer exist.**
  `#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE` survives the fold of
  the guard above it. Chunk 4.2b rewrote 27 of them; a later collapse has to do the same, or the
  next reader is told the build still has a HIP arm.
- **`preprocess_equiv.py` cannot prove a header sweep.** It preprocesses both sides with one include
  set, so the before file reads the *after* tree's header and the difference you meant to prove
  disappears. `scripts/port/cuda_live_equiv.py` gives each side its own tree's directories.
- **A doc-insertion script must preserve a UTF-8 BOM.** `docs/EXPERIMENT_E7.md`, `_E10.md`,
  `_E17.md` and `_E18.md` begin with one, so a check like `line.startswith('# ')` fails on them and
  a naive rewrite silently drops the BOM.
- **`media_tools.install()` has no marker on non-nt.** It returns readiness from `shutil.which`
  and ignores `installed.json` entirely, so a test that wants to prove "a stale marker cannot hide
  changed tools" is a Windows-arm test; patch `shutil.which` both ways to gate the Linux arm.
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
  Chunk 4.1 is the first chunk that dropped names, and it found the two halves of that rule:
  **`reads` are anchored to `path:line`**, so rewording a comment above a read breaks the gate on
  rows whose content did not change - `scripts/port/reanchor_env_reads.py` remaps them by order
  within each file and refuses where a file's read *count* moved, because that is a row decision.
  Run it before `--refresh-defaults`, since the evidence entries carry the same `path:line`. Its
  `--apply` also rewrites every row's `reads` in sorted order, so `--refresh-defaults` has to run in
  the same sequence or `git diff --stat` reports tens of thousands of lines that are only
  reordering - compare the parsed rows (`zip(old, new)` on `var`) rather than reading that number.
  And
  **a dropped row needs a whole-tree reader scan**: `GGML_CUDA_SYCL_GRAPHS`,
  `GGML_CUDA_SYCL_DISABLE_GRAPHS` and `GGML_NO_SYCL_COMPILLER` name SYCL but are read in
  `ggml-cuda/`, so they stay; `SPLIT_DECODE_MEGA_DISPATCH` and four siblings lost their last
  reader when the Metal-only branches went. `scripts/port/drop_rows.py` reports against
  `inventory()` and `--keep` refuses the batch if any name is still read.
  `envvars.py:inventory()` scans only `engines/` and `vendor/`, and treats `setenv("NAME")` as a
  read - a `setenv` whose name is a variable is invisible to it. It also misses a read through a
  wrapper: `glm_graph_env_present("DS4_ROCM_X", "DS4_METAL_X")` is live but undocumented, and 2.2e
  counted 123 such names - `metal_word_census` reads 118 since chunk 4.3b folded four away with
  their guards. Do not rename them to make the table honest; fix the inventory.
- `files=$(find ...) $extra_paths` is not an assignment: bash reads the words after a `VAR=$(…)`
  assignment as a command to run, and dies with "Permission denied" naming the first path. Put the
  list in one quoted assignment.
- Gate greps must be scoped: `grep -rn "..." engines/ds4` also matches build outputs, engine
  binaries, `*.o`, `ds4-decode.so` and `__pycache__`, and `build/wt-base` holds a second copy of
  every engine file. Use `--exclude-dir=build --exclude-dir=.git` plus
  `--include=*.c --include=*.h --include=*.cpp --include=*.inc`. Vendored files
  (`engines/ds4/external/nlohmann/json.hpp`, `linenoise.c`) are left untouched on purpose - 3 hits
  there are expected, not failures.
- `_` is a word character. That protects `DS4_METAL_*` from `\bMETAL_` - which is why 2.2b kept all
  the env names - but it also *hid* the identifiers with a leading underscore (`glm_metal_*`,
  `g_metal_graph_*`, `ds4_engine_metal_graph_*`) from 2.2b's gate. 2.2d caught them with prefix
  entries long enough to start at a real boundary, and 2.2e caught the same class twice more on the
  other side: `DS4_GLM_METAL_*` constants are invisible to `\bMETAL_` because `DS4_GLM_` precedes it,
  and so are the seven `DS4_METAL_PROFILE_*` macros, because `DS4_` does. A gate needs an entry for
  *every* prefix, so it cannot be a grep. `scripts/port/metal_word_census.py` is the gate now: it
  classifies each mention by its position in the file, which is the only test that separates a kept
  environment name from an identifier. The old regex census (`DS4_[A-Z_]*METAL[A-Z_]*` first, then
  whatever is left) cannot make that separation - it filed `DS4_METAL_PROFILE_DECODE_STAGE` with the
  environment names, which is exactly how 2.2e's first run concluded that no identifier was left.
  It also matches uppercase only, so `ds4_metal.m` counts as the comment it is.
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
- **The llama tree's "plain flag set" needs three flags the commands file did not pass until chunk
  4.2.** This fork has no `tests/`, no `examples/CMakeLists.txt` and no `pocs/`, so cmake dies on
  `add_subdirectory` unless you pass `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`, and the
  producer CLI refuses to configure without `-DLLAMA_DSV41_JSON_INCLUDE_DIR="$llama/vendor"`. Both
  flag sets are 489 edges from scratch, not the 210 the commands file used to say; a CUDA-only
  re-run after a `ggml-cuda` header edit is ~200 edges and takes the better part of ten minutes at
  `-j8`. Read `-DGGML_USE_*` out of `compile_commands.json` before a collapse - that list is the
  premise, and a name outside it is a macro no compiler here resolves.
- `scripts/port/warning_profile.py` reads `build/llama/compile_commands.json`, so it cannot cover the
  **engine**: `ds4.c` is not in that file. Compare an engine TU by hand instead -
  `make -C build/wt-base/engines/ds4 ds4.o CUDA_ARCH=sm_120` and the same in the working tree, then
  `grep "warning:" | sed` away the path, line and column, `sed 's/[0-9]+/N/g'`, sort and diff - and
  compile `ds4_ext.o` too, because `-DDS4_EXTERNAL_DECODE=1` is a second compilation of the same
  file with different live arms. Each takes about a minute. For a **CUDA** engine TU the same
  by-hand recipe is in `docs/linux-port-commands.md`, and it has to add `-Xcompiler -Wall
  -Xcompiler -Wextra`: `NVCCFLAGS` passes neither, so the build is silent on both sides and a
  same-flags comparison proves nothing (nvcc 13 has no `--warnings` option). Give each TU its own
  log, and do not compile the two trees concurrently into one scratch object - that race lost two
  warnings on the first run of this and reads as a warning profile change that was not there.
- `python3 -B scripts/release_check.py` flags any absolute path under a user's home directory as a
  "machine path", and it scans every tracked blob - including a `.cpp` committed under
  `docs/evidence/` - so docs and evidence files must use repo-relative paths. It also scans the
  working tree: `python3 -B -m pytest tests/ -q` leaves scratch behind in `jobs/` and
  `single-parent-*/`, whose job-spec JSON contains this box's home path, and those directories are
  untracked but *not* gitignored. Delete them before running the check, or the gate reports
  "machine path" findings the tree did not cause. Pointing `STUDIO_RUNTIME_ROOT` at the repo's
  gitignored `runtime/` keeps that scratch out of the check's way, which is why the suite is run
  that way here.
- `git status -sb` is the push check; the pre-commit hook also scans the commit message, so a
  message must not quote a machine path either.
- **Never move your own shell into a capped cgroup.** `echo $$ > cgroup.procs` under a 60 MiB scope
  OOM-killed the whole command line (exit 137, output lost) rather than the allocating program.
  Join a scope from the child (`preexec_fn`) or from a subprocess you can afford to lose, and probe
  ceilings with a throwaway `python3 -c` inside `systemd-run --user --scope`.
- **A cgroup is not a process attribute.** Membership is inherited across fork and exec, so the
  launcher must move the child in *before* exec (`preexec_fn` writes its own pid to
  `cgroup.procs`) or move itself in and let the command inherit - but if the launcher joins the
  scope it can never `rmdir` it, because it is still a member. That is why `guard_child.py` uses
  `preexec_fn` on the command rather than joining itself: the first version left a scope that could
  not be removed.
- `preexec_fn` runs between fork and exec, so it must not allocate, and it is unsafe in a
  multithreaded process. `open_containment` refuses to build a scope when
  `threading.active_count() > 1` rather than forking anyway; `shared_guard`'s Adapter runs its
  monitor thread, which is why the shared path gets its scope from `guard_child`, a fresh process.
- Read `memory.peak` and `memory.events` **before** `close()`: `rmdir` takes the counters with the
  directory. `memory.oom.group=1` means one over-budget descendant takes the whole tree down - the
  behaviour a job's memory limit had - so a test that expects the parent to survive a grandchild's
  allocation is testing the wrong thing.
- **There is no `TerminateJobObject` on the shared path, and that is deliberate.**
  `shared_guard`'s Adapter overrides `GuardedJob._stop`, so the vendored `os.killpg` on the group
  anchor never runs and the Windows `kernel_job.stop()` after a forced kill has no Linux counterpart.
  `cgroup.kill` exists on this kernel and would be the direct equivalent, but it is an immediate
  SIGKILL of every member: using it would bypass the cooperative request that
  `ProgressOwnedProcesses.stop()` is built to make first. The containment that the retired job gave
  a straggler is kept instead - a scope with members is left in place, still capped - and an
  unconfirmed cleanup holds the lease rather than releasing it. If a later session decides to
  force-kill through the scope, it belongs after grace, not instead of it.
- A scope's name carries the pid that created it (`slstack-<hint>-<pid>-<nonce>`), because
  `open_scope` reaps an *empty* scope whose creating pid is gone before creating the next one - a
  guardian killed by SIGKILL before its own teardown otherwise accumulates capped directories in
  the user's slice (twelve had piled up on this box). Two consequences: a live launcher's fresh,
  still-empty scope is never taken from it, and scopes created before this naming existed match no
  rule and are never reaped (this box's were removed by hand).
- **`memory.min` is honoured only while every ancestor between the subgroup and the root carries
  a floor of its own**, and above `app.slice` on this box the ancestors are root-owned
  (`user@1000.service/memory.min` is uid 0, mode 0644), so an unprivileged process can never
  complete that chain. Measured, not recalled: 64 MiB of page cache in a subgroup floored at
  64 MiB came fully back through `memory.reclaim`, both with `app.slice` at 0 and with `app.slice`
  raised to 64 MiB too. Writing the floor is therefore free and correct, and counting it as
  residency is not: `cgroup_scope.memory_floor` returns `honored` and `blocked_by`, and the guard
  prints `DS4COMBO_RESIDENT_FLOOR_INERT`. Do not "fix" this by raising `app.slice` - that is the
  user's whole session, including Strata and this assistant, and it does not help anyway.
- **A scope that has member processes cannot delegate `memory`:** writing `+memory` to its
  `cgroup.subtree_control` answers EBUSY (16). That is why the producer's floor lands on the scope
  it inherited instead of a scope of its own - a nested scope is impossible and a sibling created
  in `app.slice` would leave the guardian's ceiling. An *empty* scope of ours does delegate.
- **A floor equal to a scope's ceiling leaves nothing to evict**, so pressure inside that scope
  becomes a termination instead of an eviction of page cache. `protect_working_set` keeps the
  64 MiB metadata headroom below `memory.max` for the same reason the Windows arm kept a gap
  between its minimum and its advisory maximum.
- **`memory.reclaim` is how protection is tested without host-wide pressure.** It answers EAGAIN
  when there is nothing to take, and `memory.current` also counts the holder's interpreter RSS, so
  assert on the delta rather than on the absolute.
- **`mlock` past `RLIMIT_MEMLOCK` answers ENOMEM (12) on this kernel**, not the EAGAIN the man page
  leads with; `munlock` of pages the kernel never locked is a no-op success, so a deleter may
  unlock what it recorded at allocation.
- **A `/dev/shm` name outlives the process that made it**, which a kernel section's handle did not:
  a controller killed before its own teardown would pin its tier until reboot. The creating pid is
  therefore in the name and `combo_ram.reap_stale_sections()` removes owned names whose creator is
  gone. A test that leaves one behind leaks RAM, not just a file.
- **`ctypes.c_char.from_buffer(mmap)` exports the buffer**, and a live export makes `close()` raise
  "cannot close exported pointers exist". Read the address and `del` the ctypes object in the same
  step (`combo_ram._address`).
- `patch('mod.func', wraps=mod.func)` makes attribute lookups on the mock delegate to the wrapped
  function, so `mock.return_value` and `mock.spy_return` are unreachable (the test that asserted the
  scope ceiling through it got `'_SentinelObject' object is not subscriptable`, then `AttributeError:
  'function' object has no attribute 'spy_return'`). Record the return value in a `side_effect`
  closure instead.
- This box's Python 3.14.9 has **no `sys.environ`** (`dir(sys)` lacks it); `os.environ` works. Test
  helpers that build a child environment must use `os.environ`.
- `/proc/vmstat`'s `pgpgin`/`pgpgout` are 512-byte blocks despite the names, `pswpin`/`pswpout` are
  pages, and `pgmajfault` is a count, not bytes. `/proc/meminfo` is in kB; `/proc/vmstat` lines have
  no colon, so one parser has to accept both `name: value` and `name value`.
- **The pinned shared guard is a hash, not a filename.** `slstack/shared_guard.py` refuses to run if
  `vendor/studio/studio/job_guard.py` or `job_paging.py` no longer hashes to its pin (computed over
  the bytes after the two-line provenance header). Editing either therefore needs three changes at
  once: the patch, the pin, and a `local_patches` entry in `vendor/studio/PROVENANCE.json` saying
  what changed and why. Chunk 5.1 did that for two one-line Linux fixes. `dsv41_session_process.py`
  is not pinned, but its provenance header claims "Source body unchanged", so reword the header too
  when you patch it.
- **Un-skipping a platform's tests changes the counts without changing the failures.** 5.1 removed
  `skipUnless(os.name=='nt')` from `tests/test_combo_guard.py` and 21 tests started running; the
  skip total moved 62 -> 41 and one of those newly-running tests exposed a bug that had been hidden
  by the skip. Read the failing *names* against a baseline worktree, and expect the skip line to move
  whenever a platform gate is ported.
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
- **Python 3.14 has no `Path.lexically_normal()`.** `dir(Path)` shows only `resolve`. The runner's
  "absolute and normalized" rule therefore compares `str(path)` against
  `os.path.normpath(str(path))` and separately refuses `'..' in path.parts`; the engine's C++ rule
  is `p == p.lexically_normal()`.
- **`O_NOFOLLOW` answers a symlink with `ELOOP`, not with a refusal you can name.** The runner turns
  `errno.ELOOP` into the reparse-point refusal so an alias reads like the Windows `GetFinalPathName
  ByHandleW` refusal; anything else propagates. Opening a directory without `O_DIRECTORY` is
  deliberate - `O_DIRECTORY` would raise before the fstat type rule could report what it saw.
- **The producer refuses a RAM-backed volume, and `/tmp` here is one.** Anything that takes a lease
  (`read_lease`, `exclusive`, `directory`) or launches a job runs `require_local_volume`'s magic list
  against the parent, so a `tempfile.mkdtemp()` fixture under `/tmp` or `/dev/shm` is refused. The
  runner's tests write under the repo's gitignored `runtime/`; `tests/test_advisory_lock.py` keeps
  using `/tmp` because flock has no volume rule.
- **A launched-but-unadmitted child is invisible until it reports itself.** That is why
  `producer_launch.py` publishes its pid on a ready pipe and `_Job.__init__` blocks on it: without
  that handshake the controller would know only a pid that has not yet called `setsid`, so
  `os.getpgid(pid)` still names the *controller's* group and a `killpg` of it would signal the
  controller's own tree. `pass_fds` keeps the two pipe ends the launcher needs; the controller
  closes its copies in `finally` and keeps only the write end as `job.admission`.
- **`PR_SET_PDEATHSIG` does not survive `execv`.** It is the unstarted child's guard only, which is
  exactly the window the job's kill-on-close covered before `release_kill_on_close()`; a test that
  wants "the controller died and the producer survived" must resume first.
- **The engine's own output is exclusive-new.** The no-model producer smoke refuses with "Cannot open
  exclusive-new output or read-only physical input" if `build/smoke/out/` still holds a previous
  run's `payload.bin` or pending receipt, so clear it before reading that gate's outcome.
- **The producer's `--gpu-reserve-bytes` floor is 1 GiB.** With less free than that it refuses at
  the reserve, and that is the deepest no-model outcome reachable here; lowering the flag below
  1 GiB reaches a different refusal ("requires ... explicit GPU reserve>=1GiB") instead of going
  deeper.
- **`LockUnavailable` means "refused", not "failed".** `journal.advisory_lock` maps only
  `EWOULDBLOCK` to it, because `lease.gpu_lease` converts that class into `AdmissionRefused`, which
  the guardian's `gpu_wait_seconds` loop retries; any other `OSError` propagates instead of
  reporting an admission already in progress. Chunk 5.2 found this while chasing a load-dependent
  failure of `test_a_blocking_wait_ends_when_the_holder_leaves`, whose real cause was the test's own
  holder: the polling probe could win the lock first, and the holder died on the refusal, so the
  wait never saw the contention it was waiting for. The holder now retries.
- **A freshly resumed child's module list races its exec.** `/proc/<pid>/maps` gains the
  interpreter's mappings a moment after the admission byte lands, so an observation of
  `job.modules()` polls rather than asserting once. The retired Windows runner had the same race
  with `K32EnumProcessModulesEx`.

## Suggested skills

- `adversarial-code-review` after a deletion or rename chunk, with the `scripts/port/` equivalence
  commands handed to the reviewer as evidence (see the method section for what it caught last time).
  For a text chunk the evidence is `comment_code_diff.py`, `metal_word_census.py` and the engine's
  two compilations; 2.2e was verified inline, not by a subagent. For a collapse the evidence is
  `docs/evidence/collapse-probe.h` run first (the tool itself needs proving before its output is
  trusted), the per-file `#if`/`#endif` balance check, and `cuda_live_equiv.py`.
- The user asked, on 2026-11-06, that this work not be parallelised: subagents are expensive on
  context prefill, and one of them died mid-run with no output anyway. Verify inline - the compile
  checks, the hand-compiled programs and the function-level diff - and ask before fanning out.
- `add-unit-test` for extending `engines/ds4/tests/test_batch_io.cpp` or `test_decode_library.cpp`.
  The producer's reader is covered by `docs/evidence/wf-io-selftest.cpp`, compiled by hand, because
  the producer has no test target; the Engram loader by `tests/dsv41_engram_early_io.cpp`, the carry
  store and its RAM tier by `tests/dsv41_carry_release.cpp` and `tests/pinned_carry_memory.cpp`, all
  three also compiled by hand (`docs/linux-port-commands.md`).
- `code-design` / `code-reviewer` when a chunk changes admission rather than text - chunk 3.1h,
  chunk 5.1's cgroup scope, chunk 5.2's runner (suspended launch, leases, ownership) and chunk 5.3
  (`memory.min`, `LockedRam`) are the four, and each was read through `code-design` before writing.
  What mattered in 5.3: "remove duplicate code, one source of truth" is why `protect_working_set`
  exists once now - `combo_ram`'s verbatim copy is deleted and its two callers import
  `slstack.working_set` - and "a boolean flag argument means the function does two things" is why
  the shared half of the tier is a `SharedSection` with `attach()` beside it rather than a
  `LockedRam(owner=False)` option. In 5.2 the same rule is why `suspended()` keeps the retired
  five-positional-argument shape, the admission byte lives in the launcher rather than in a
  `suspend=` option, and `journal.advisory_lock` gained a `shared` arm instead of a second lock
  helper.
- `adversarial-code-review` after a chunk that rewrote behaviour, if the user allows a subagent; the
  `scripts/port/` equivalence commands are the evidence to hand it. For 3.1h the proof was inline:
  the collapse report, `preprocess_equiv`, `code_line_diff`, the warning profile, and both carry
  programs, one of which was red before the change. For 5.1, 5.2 and 5.3 there is no C equivalence
  to prove - the evidence was the suite plus `tests/test_cgroup_scope.py` (the kernel refusing an
  oversized tree, a grandchild's refusal taking the whole group, and `memory.reclaim` showing that
  an un-honoured floor really is evicted) and
  `tests/test_producer_runner.py` (nothing runs before admission, the force fallback only after two
  requests and both graces, and a killed group with a grandchild in it ending empty), plus
  `tests/test_working_set.py` (a foreign scope left untouched, a real scope floored) and
  `tests/test_combo_ram.py` (a second process attaching the same name and writing through it).
- `web-research` for liburing, CCCL 3 / libcudacxx header changes, llama.cpp cmake details, and the
  cgroup v2 interface (`Documentation/admin-guide/cgroup-v2.rst`: what a subgroup sees, the
  no-internal-processes rule, `memory.oom.group`, `cgroup.kill`, and the ancestor rule for
  `memory.min` - but confirm any protection claim against `memory.reclaim` on the box rather than
  against that document alone).
- `add-unit-test` for the Python side too: `tests/test_cgroup_scope.py`,
  `tests/test_advisory_lock.py`, `tests/test_producer_runner.py`, `tests/test_working_set.py` and
  `tests/test_combo_ram.py` are the pattern for a gate that
  needs no model and no GPU - drive the real kernel mechanism and assert on the counters it reports.
  The runner's file is also the pattern for *not* trusting your own fixture: three of its first
  failures were the test calling a context manager without entering it, and one was an assertion on
  `sys.executable` against `/proc/<pid>/maps`, which reports the resolved interpreter.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when this session runs out of context - update this file, don't replace it with
  chat recap.
