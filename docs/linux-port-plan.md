# Linux port plan: remove Windows, run on Linux

Durable plan for the fork. Keep this file updated; tick tasks as they land and add a
one-line note under each phase when something surprises you.

Repo: `Macasacker/streaming-llama-stack`, branch `linux-only`.
Upstream fork base: `21a9670` (fronteagle739/streaming-llama-stack).
Background reading: [linux-port.md](../linux-port.md) has the original repo review and the blocker survey.

## Machine facts

Arch Linux, RTX 5090 32 GB, compute capability 12.0, driver 615.78, CUDA 13.4
(`nvcc --list-gpu-code` shows `sm_120` and `sm_121`; `compute_120a` compiles),
cmake 4.4.5, ninja 1.13.2, `/usr/include/liburing.h` present, Python 3.14.8
(the stack wants 3.12), ext4 on a single NVMe, 30 GB RAM, ~793 GB free.
`nvidia-fs` is not installed, so GPUDirect Storage is unavailable today. io_uring is the route.

## End state

- No `_WIN32`, `windows.h`, DirectStorage, IoRing, `msvcrt`, `WinDLL`, `.exe` or `.dll` anywhere.
- One ds4 CUDA engine built by `make cuda CUDA_ARCH=sm_120`, including the packed FP4 decode kernels.
- The streaming producer reads weights through io_uring (optionally cuFile when `nvidia-fs` is installed), Engram extents through io_uring, carry banks through `mmap` plus `mlock`.
- Process guarding through cgroups v2 instead of Windows Job Objects.
- `./bin/slstack`, `build.sh`, `scripts/*.sh`, `examples/run_preset.py` are the only entry points.
- README and docs describe Linux. Performance figures are re-measured and re-qualified, or explicitly marked as inherited from the Windows build.

## Decision taken

Three options were on the table for `engines/ds4-win`.

1. Delete it and use `engines/ds4-mac` CUDA on Linux. Cheapest, but you lose the packed FP4
   decode kernels, `ds4_mmq_fp4_packed.cuh` and `ds4_q4k_half.cu`, which only ds4-win has.
   Rejected: the packed layout is what the RTX decode speed comes from.
2. Strip its Win32 files and rename it `ds4-linux`. Keeps packed FP4 decode, costs 333
   references, and leaves two near-duplicate engines whose `ds4.c` diverges by 17k lines.
   Rejected: three copies of the same engine is the maintenance problem this fork already
   has, and a third copy does not fix it.
3. Move its extra kernels into ds4-mac's CUDA build, then delete ds4-win. Chosen. One engine
   with packed FP4, and it also clears the config scope strings that currently say `ds4-win`.

Option 3 costs more up front than option 2 because ds4-mac's and ds4-win's CUDA sources
already diverge. That is the Phase 1 risk noted at the end of this file.

Merge `engines/ds4-win`'s extra CUDA kernels into the ds4-mac CUDA build, then delete
`engines/ds4-win`. Files that must survive:

```
engines/ds4-win/cuda/mmq/ds4_mmq_fp4_packed.cuh   packed FP4 layout flashinfer-sm120-mxfp8-mxfp4-v1
engines/ds4-win/cuda/mmq/ds4_q4k_half.cu
engines/ds4-win/cuda/mmq/ds4_q4k_half_common.cuh
engines/ds4-win/ds4_cuda.cu                        V4.1 decode kernels, 46k lines
engines/ds4-win/ds4_deepseek41_cuda.cuh
engines/ds4-win/ds4_expert_restore_async.cuh
engines/ds4-win/ds4_cache_affinity_cuda.cuh
engines/ds4-win/ds4_cache_affinity_background.cuh
```

`engines/ds4-mac/cuda/mmq` has no packed FP4 or Q4K half path today, so the merge is
real work, not a copy. Check `engines/ds4-mac/Makefile` `MMQ_OBJS` and
`DDS4_CUDA_HAVE_MXF4` handling.

The config scope key `ds4-win` appears in `slstack/envvars.py`, `slstack/process.py`,
`slstack/combo_native.py`, `slstack/general_engine.py`, `slstack/engines.py`,
`slstack/check.py` and the preset files under `examples/`. Rename it to `ds4-cuda` in the
same pass as the engine deletion, and update `docs/VARIABLES.md` afterwards with
`python -B -m slstack.envdocs`.

## Phase 0: done

Commit `8db0ba9a` "Drop Windows support: Linux-only launchers, build and metadata".

- Deleted 38 `.ps1` launchers, vendored DirectStorage SDK trees, `cmake/arm64-windows-llvm.cmake`, `cmake/x64-windows-llvm.cmake`.
- Added `build.sh`, `scripts/start-dsv41-single.sh`, four preset and cone wrappers, four thin wrappers over `bin/slstack`.
- CMake: removed the DirectStorage SDK block, the DLL export target and its `dstorage.dll` copy step, the producer `NOT WIN32` gate and the `dstorage.lib ole32 d3d12 dxgi bcrypt` link line.
- `slstack/install.py` no longer downloads the DirectStorage nupkg, no longer refuses non-NT hosts, runs `build.sh`, checks Linux binaries. `install --sdk` removed.
- `config.example.json` points at Linux binaries. `check.py` no longer requires `dstorage.dll`.
- `.gitignore` gained `*.o`, `*.so` and engine binary names; `.gitattributes`, `NOTICE`, `release_check.py` suffix rules, README updated.
- Tests that drove PowerShell now drive `sh` launchers and `examples/run_preset.py`.
- Fixed `engines/ds4-mac/ds4.c`: it skipped `ds4_gpu.h` under `-DDS4_NO_GPU`, but `ds4_upstream_controls.inc` needs the `DS4_GPU_V41_*` masks. `make cpu` now builds `ds4`.

Gate at that commit: `python3 -B scripts/release_check.py` reports only the pre-existing
`docs/VARIABLES.md:0: regenerate` finding, and
`PYTHONPATH=.:tests python3 -m unittest test_release_check test_dsv41_single_launch test_example_launchers` passes 34 tests.

Known pre-existing failure, not caused by this work: `test_yoco.py`
`test_both_launch_modes_keep_the_flag_and_budgets` asserts
`producer_prefill_sweep_tokens == 49152` and gets `'auto'`. It fails on the upstream
clone too.

## Phase 1: engine consolidation

1. Copy the packed FP4 and Q4K half kernels from `engines/ds4-win/cuda/mmq` into
   `engines/ds4-mac/cuda/mmq`, and add them to `MMQ_OBJS`.
2. Port the V4.1 decode kernel bodies from `engines/ds4-win/ds4_cuda.cu` into
   `engines/ds4-mac/ds4_cuda.cu`. Diff them first: the two files are already 46k and 33k
   lines and diverge.
3. Teach `engines/ds4-mac/ds4.c` to accept `deepseek41.expert_layout = flashinfer-sm120-mxfp8-mxfp4-v1`.
   Currently only `engines/ds4-win/ds4.c:3066` accepts it.
4. Add an external decode library target so the producer can load decode kernels beside it,
   replacing `ds4-decode.dll`. On Linux that is a shared object built from the same units.
5. Delete `engines/ds4-win`. Keep its `PROVENANCE.md` facts in `engines/ds4-mac/PROVENANCE.md`
   and in `NOTICE`, since the kernels came from NovaStar and IngeniousIdiocy.

Gate: `make -C engines/ds4-mac cuda CUDA_ARCH=sm_120` builds `ds4`, `ds4-server`,
`ds4-bench`, `ds4-eval`, `ds4-agent`. Then `make -C engines/ds4-mac test-mxfp4-cuda`.
`ds4-server` also needs `$(PAIRED_OBJS)` and `$(PAIRED_GGML)` on the Linux `cpu` and
`cuda` link lines; the Linux rules at `engines/ds4-mac/Makefile:753-758` omit them, so
`ds4-server` fails to link with `ds4_chain_create` and friends undefined.

## Phase 2: engine host layer, Linux substitutions

Files with Windows host code left to remove:

```
engines/ds4-peer   109 refs: ds4_win32.{c,h}, ds4_distributed_win32.c, ds4_tp_win32.c,
                   linenoise_win32.c, plus guards in ds4.c, ds4_server.c, ds4_agent.c,
                   ds4_bench.c, ds4_cli.c, ds4_help.c, ds4_kvstore.c, ds4_peer.c, ds4_ssd.c, ds4_web.c
engines/ds4-win    333 refs, deleted in Phase 1
```

Substitutions:

| Windows | Linux |
|---|---|
| `CreateFileW`, `FILE_FLAG_NO_BUFFERING`, `FILE_FLAG_RANDOM_ACCESS` | `openat`, `O_DIRECT`, `O_RDONLY` |
| `GetDriveTypeW`, `QueryDosDeviceW`, `BY_HANDLE_FILE_INFORMATION`, volume serial and file index | `fstat`, `st_dev`, `st_ino`, `realpath`, refuse `S_ISLNK`, pin ancestor fds |
| `GetDiskFreeSpaceW` sector alignment | `statfs` or `ioctl BLKSSZGET` |
| `_aligned_malloc` / `_aligned_free` | `posix_memalign` / `free` |
| `VirtualAlloc(MEM_RESERVE\|MEM_COMMIT)` | `mmap` plus `mlock`, or `memfd_create` |
| IoRing (`CreateIoRing`, `BuildIoRingReadFile`, `SubmitIoRing`) | io_uring: `IORING_OP_READ`, registered files, fixed buffers |
| `GetProcAddress`/`GetModuleHandleW` | direct linkage |
| `TerminateProcess` in cleanup paths | `_exit`, keep the same refusal semantics |
| `Sleep`, `GetTickCount64` | `nanosleep`, `clock_gettime` |
| `CRITICAL_SECTION`, `SRWLOCK`, `Interlocked*` | pthreads and atomics |
| `linenoise_win32.c` | the POSIX linenoise path already in `linenoise.c` |

Gate: `grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing" engines/` returns nothing.

## Phase 3: weight fetcher and producer

Locations left, in `engines/streaming-llama/prefill_llama.cpp`:

```
src      324 refs. llama-weight-fetcher.cpp (IOCP workers, DStorage interop, gpu-ring),
         llama-wf-gpu-ring.inl, llama-wf-zero-gaps.h, llama-wf-bounded-policy.h,
         llama-dsv41-engram-prefetch.cpp (108 refs, throws on non-Windows),
         llama-dsv41-weights.cpp (lease and identity, throws at :923 and :1000),
         llama-dsv41-carry.cpp, llama-dsv41-locked-ram.h, llama-dsv41-host-cpu.h,
         llama-dsv41-host-memory.h, llama-dsv41-sweep.cpp, llama-dsv41-cache-import.cpp,
         llama-token-stream.cpp/.h (already has !_WIN32 branches)
ggml     108 refs, only these matter: ggml-cuda.cu:8117, dsv41-flashinfer.cuh,
         dsv41-native-ffn.cuh, dsv41-dense-fp8.cuh, ds4split-expert-projection.cuh,
         ds4split-mmid-observer.cuh. The rest are upstream ggml guards in CPU, Vulkan,
         SYCL and hexagon code, harmless on Linux. Leave them.
tools     54 refs: dsv41-prefill/main.cpp (windows.h, bcrypt.h, dxgi1_4.h),
         producer-abort.h, live-session.inc, carry-copy-bench.cpp, decode-receiver/main.cpp,
         server/*, mtmd/*
```

Work:

1. Implement `wf_io_uring_worker` in `llama-weight-fetcher.cpp` to replace both the IOCP
   worker and the `pread` fallback pool at `:1205`. Keep `WF_MODE_AUTO`, and make it prefer
   GDS only when `cuFileIsGDSInternal` succeeds, otherwise io_uring.
2. Delete `llama-wf-gpu-ring.inl` and the DStorage interop paths. A GPU-side ring needs a
   different mechanism on Linux, so drop the `--ds-gpu-ring-slots` option rather than faking it.
3. Port `llama-dsv41-engram-prefetch.cpp` to io_uring with 1 MiB extents at QD64. This is
   where the 76.2 s to 43.2 s improvement comes from, so measure it separately.
4. Replace the provider lease in `llama-dsv41-weights.cpp` with the Linux identity checks.
   The `V4.1 physical weight provider requires Windows CUDA/DirectStorage` throws at `:923`
   and `:1000` must go.
5. Replace `llama-dsv41-locked-ram.h` and the carry lease in `llama-dsv41-carry.cpp`.
6. Replace BCrypt SHA-256 in `tools/dsv41-prefill/main.cpp` and `src/boundary-capture.cpp`
   with OpenSSL EVP or a vendored SHA-256. The producer binds its own executable SHA, so
   the digest must be byte-exact.
7. Remove the DXGI and CUDA interop paths in the producer, and the `windows.h` includes in
   `mtmd` and `server`.
8. Build the producer: `cmake ... -DLLAMA_DSV41_OFFLINE_PRODUCER=ON` and add
   `llama-dsv41-prefill` to the target list in `build.sh`, then run
   `build/llama/bin/llama-dsv41-prefill --check-native-profiles`.

Gate: the producer links and passes the profile preflight. Then a 262K prefill run.

## Phase 4: Python orchestrator

Windows-specific code to remove, largest first:

```
guard.py         24 refs: windows_job with kernel32 CreateJobObjectW, SetInformationJobObject,
                 AssignProcessToJobObject. Replace with a cgroup v2 scope: systemd-run --scope
                 --memory=..., or write memory.max and kill the cgroup on teardown. lease.py
                 relies on the guardian handle closing the Job to kill GPU descendants.
single_runtime.py 13 refs: PRODUCER_FILES = ds4-decode.dll, dstorage.dll, dstoragecore.dll,
                 and .dll copies
memory_counters.py 9 refs: ctypes.WinDLL('psapi.dll'). Linux path is psutil plus /sys/meminfo.
lease.py          6 refs: msvcrt byte-range lock has an fcntl branch already
journal.py        6 refs: Windows rename race handling
process.py        6 refs: scope names including ds4-win
working_set.py    5 refs: Windows working-set accounting
shared_guard.py   5 refs: Job object sharing
combo_native.py   5 refs: ds4-win env scope
combo_ram.py      4 refs: refuses non-NT at :86, SetProcessWorkingSetSizeEx. Linux equivalent is
                 cgroup memory.min plus mlock.
lora.py           4 refs: Windows drive-letter handling in scale parsing
generation.py     4 refs: msvcrt locking has an fcntl branch; rename-race retry comment
```

Also: `install.py` build path, `envvars.py` `ds4-win` scope and `weight_fetcher` default
`build/llama/bin/slstack-weight-fetcher.dll`, `studio_worker.py` and `vision.py` `.dll`
lookups, `general_engine.py` `ds4-server.exe`, `engines.py:215` message about the Windows
target-only adapter, `discovery.py` skip list with `$recycle.bin` and `windows`,
`combo_producer.py` and `combo_identity.py` Windows commit/reserve text,
`backend_setup.py` and `backends.py` `Scripts/python.exe` branches.

Gate: `./bin/slstack --config <file> check` runs on Linux, and `python3 -m unittest` over
the Python suite in `tests/` passes.

## Phase 5: vendor and media

`vendor/engines/cosmos_windows` is 129 refs of Windows-specific media code, and
`vendor/studio/studio/dsv41_windows_runner.py` is the producer launcher used by
`slstack/combo_fixture.py`, `combo_native.py`, `combo_producer.py`, `combo_session.py`.
Both need Linux twins: `vendor/engines/cosmos_linux` and a `dsv41_linux_runner.py` that
spawns the producer through the cgroup guardian. Media generation is marked experimental
in the README, so this can lag the inference path.

## Phase 6: docs, tests, presets

- `docs/`: 70 Windows references, including `docs/PERFORMANCE.md`, `docs/BACKEND_SETUP.md`,
  `docs/BACKEND_SOURCE_AUDIT.md`, `docs/ROADMAP.md`, `docs/DSV41_GGUF.md`. Rewrite the
  hardware and build sections, keep measurement history labelled as Windows.
- `examples/`: 76 references. Preset JSON and READMEs name `rtx-fast-gpu.ps1`; replace with
  `python -B examples/run_preset.py --preset <name> --machine rtx|mac --config <file>`.
- `tests/`: 132 references. Beyond the four already ported, `engines/ds4-win/tests/*`
  disappear with the engine, and `tests/*.py` gates that assume `.exe`, PowerShell, or Job
  objects need Linux equivalents.
- Regenerate `docs/VARIABLES.md` after the env scope rename:
  `python -B -m slstack.envdocs`.

## Phase 7: qualification

Nothing measured transfers. Re-run against the hash-pinned fixtures and write new receipts:

1. 262K exact-cone single engine, three runs, record prompt clock and decode rates.
2. 200K real-text fixture, password recall, logits SHA-256.
3. Engram extent reads measured separately against the `pread` baseline, so the io_uring gain is isolated.
4. Paired RTX-to-Mac run, KL within mean 0.1 / max 0.5 nats.
5. Update `docs/PERFORMANCE.md` and the README tables only from these receipts.

## Sequencing and risk

Do Phases 1 to 4 in that order. Phase 3 depends on Phase 1 because the producer loads decode
kernels beside it. Phase 5 and 6 can run after the inference path works. Phase 7 last.

Three things that could cost more than expected:

- The packed FP4 decode merge. ds4-mac and ds4-win CUDA sources diverge by tens of
  thousands of lines, so this is a port, not a copy.
- io_uring throughput parity. Windows hits about 25 GB/s on a RAID-0. A single NVMe plus
  io_uring may not match it, and prefill numbers will move regardless of correctness.
- Locked memory limits. `mlock` needs `RLIMIT_MEMLOCK` or a cgroup allowance; the Windows
  path relied on working-set locking plus a Job memory limit. Decide the mechanism before
  rewriting `combo_ram.py` and `locked-ram.h`.

## How to resume this plan

```sh
cd streaming-llama-stack
git log --oneline -1            # last landed commit
git status --short              # working state
grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing\|msvcrt\|WinDLL" \
  engines slstack vendor scripts examples tests | wc -l   # remaining Windows refs
python3 -B scripts/release_check.py
```

At Phase 0 the remaining count is about 940.
