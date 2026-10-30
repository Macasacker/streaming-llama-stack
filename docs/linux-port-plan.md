# Linux-only plan: delete Windows, macOS, and every other platform

Durable plan for this fork. Tick tasks as they land and note surprises under each phase.

Repo: `Macasacker/streaming-llama-stack`, branch `linux-only`.
Fork base: `21a9670` (fronteagle739/streaming-llama-stack).
Context: [linux-port.md](../linux-port.md) has the original repo review and blocker survey.

## Scope

The fork supports one platform: Linux on NVIDIA RTX 50-series, CUDA compute capability 12.0.

Delete outright:

- Windows: DirectStorage, D3D12 interop, Win32 host code, IoRing, Job Objects, PowerShell.
- macOS: Metal sources, Objective-C, Accelerate, the Darwin Makefile branch, the Mac receiver,
  MLX, mlx-serve, omlx, mference (Swift), vendor/mac-mlx.
- Other ggml backends that are not CUDA: Vulkan, SYCL, Hexagon, KleidiAI, RPC, AMX, CPU-only
  MoE paths that exist to serve Mac and Windows hosts.
- ROCm and DGX Spark targets. Linux, but AMD and sm_121 paths nobody here will run. Say the
  word if ROCm should stay.

What survives: CUDA + CPU ggml, the streaming llama producer, one ds4 CUDA engine, the Python
orchestrator, the web UI, MCP, LoRA, and images.

## Machine facts

Arch Linux, RTX 5090 32 GB, compute capability 12.0, driver 615.78, CUDA 13.4
(`nvcc --list-gpu-code` shows `sm_120` and `sm_121`; `compute_120a` compiles), cmake 4.4.5,
ninja 1.13.2, `/usr/include/liburing.h` present, Python 3.14.8 (the stack wants 3.12), ext4 on
a single NVMe, 30 GB RAM, about 793 GB free.
`nvidia-fs` is not installed, so GPUDirect Storage is unavailable today. io_uring is the route.

## Open decisions

1. Two-machine mode. The Mac receiver is the current decode half. With macOS gone, either
   (a) keep disaggregation and build the mode3/YOCO receiver for a Linux decode host, which is
   work the upstream never finished and documented in
   `examples/dual-machine/dsv41-large-ram-untested/README.md`, or (b) drop two-machine mode and
   keep single-machine only. Option (b) removes `slstack/mac.py`, `pairing.py`, `native_broker.py`,
   `slstack/combo_*` handoff code and every `examples/dual-machine` preset. Recommended if the
   goal is less bloat.
2. Supported models. The Mac was the decode path for Kimi K2.5/2.6/2.7, GLM 5.x, MiniMax M2.5/M3,
   Step 3.5/3.7 and MiMo V2.6 Pro. Without MLX those models keep streaming prefill on the RTX but
   have no decode path. Either restrict the supported list to DeepSeek V4.x on ds4 CUDA, or keep
   llama.cpp CPU/CUDA decode for the others. Restricting is the smaller fork.
3. ROCm. 14 files in `engines/ds4-mac/rocm` plus Makefile targets. Delete unless AMD hardware is
   planned.

## Decision taken on engines

Three options were on the table for `engines/ds4-win`.

1. Delete it and use `engines/ds4-mac` CUDA on Linux. Cheapest, but you lose the packed FP4
   decode kernels, `ds4_mmq_fp4_packed.cuh` and `ds4_q4k_half.cu`, which only ds4-win has.
   Rejected: the packed layout is what the RTX decode speed comes from.
2. Strip its Win32 files and rename it `ds4-linux`. Keeps packed FP4 decode, costs 333
   references, and leaves two near-duplicate engines whose `ds4.c` diverges by 17k lines.
   Rejected: a third copy of the same engine is the maintenance problem this fork already has.
3. Move its extra kernels into one CUDA engine and delete both platform engines. Chosen.

End state: `engines/ds4`, one engine, CUDA only, built by `make cuda CUDA_ARCH=sm_120`.
Kernels that must survive from `engines/ds4-win`:

```
cuda/mmq/ds4_mmq_fp4_packed.cuh   packed FP4 layout flashinfer-sm120-mxfp8-mxfp4-v1
cuda/mmq/ds4_q4k_half.cu
cuda/mmq/ds4_q4k_half_common.cuh
ds4_cuda.cu                       V4.1 decode kernels, 46k lines
ds4_deepseek41_cuda.cuh
ds4_expert_restore_async.cuh
ds4_cache_affinity_cuda.cuh
ds4_cache_affinity_background.cuh
```

`engines/ds4-mac/cuda/mmq` has no packed FP4 or Q4K half path today, so the merge is real work.
Check `engines/ds4-mac/Makefile` `MMQ_OBJS` and `DDS4_CUDA_HAVE_MXF4`.

The config scope key `ds4-win` appears in `slstack/envvars.py`, `slstack/process.py`,
`slstack/combo_native.py`, `slstack/general_engine.py`, `slstack/engines.py`, `slstack/check.py`
and the preset files under `examples/`. Rename it to `ds4` in the same pass as the engine
deletion, then regenerate `docs/VARIABLES.md` with `python -B -m slstack.envdocs`.

## Phase 0: done

Commit `8db0ba9a` "Drop Windows support: Linux-only launchers, build and metadata".

- Deleted 38 `.ps1` launchers, both vendored DirectStorage SDK trees,
  `cmake/arm64-windows-llvm.cmake`, `cmake/x64-windows-llvm.cmake`.
- Added `build.sh`, `scripts/start-dsv41-single.sh`, four preset and cone wrappers, four thin
  wrappers over `bin/slstack`.
- CMake: removed the DirectStorage SDK block, the DLL export target and its `dstorage.dll` copy
  step, the producer `NOT WIN32` gate and the `dstorage.lib ole32 d3d12 dxgi bcrypt` link line.
- `slstack/install.py` no longer downloads the DirectStorage nupkg, no longer refuses non-NT
  hosts, runs `build.sh`, checks Linux binaries. `install --sdk` removed.
- `config.example.json` points at Linux binaries, `check.py` no longer requires `dstorage.dll`.
- `.gitignore` gained `*.o`, `*.so` and engine binary names. `.gitattributes`, `NOTICE`,
  `release_check.py` suffix rules and README updated for Linux.
- Tests that drove PowerShell now drive `sh` launchers and `examples/run_preset.py`.
- Fixed `engines/ds4-mac/ds4.c`: it skipped `ds4_gpu.h` under `-DDS4_NO_GPU`, but
  `ds4_upstream_controls.inc` needs the `DS4_GPU_V41_*` masks. `make cpu` now builds `ds4`.

Gate at that commit: `python3 -B scripts/release_check.py` reports only the pre-existing
`docs/VARIABLES.md:0: regenerate` finding, and
`PYTHONPATH=.:tests python3 -m unittest test_release_check test_dsv41_single_launch test_example_launchers`
passes 34 tests.

Pre-existing failure, not ours: `test_yoco.py` `test_both_launch_modes_keep_the_flag_and_budgets`
asserts `producer_prefill_sweep_tokens == 49152` and gets `'auto'`. It fails on the upstream clone.

## Phase 1: one CUDA engine

1. Create `engines/ds4` from `engines/ds4-mac`, keeping the non-Darwin Makefile branch.
2. Delete the Metal and Objective-C sources: 28 files in `engines/ds4-mac/metal`, `ds4_metal.m`,
   `ds4_metal*.inc`, the Accelerate and ImageIO link lines, and the Darwin branch of the Makefile.
   `engines/ds4-mac` has 6,091 Apple/Metal references.
3. Delete `rocm/`, `cuda-spark`, `cuda-generic`, `strix-halo`, `rocm` Makefile targets and the
   gfx1151 flags unless ROCm stays.
4. Merge the ds4-win kernel list above into `engines/ds4/cuda/mmq` and add them to `MMQ_OBJS`.
5. Port the V4.1 decode kernel bodies from `engines/ds4-win/ds4_cuda.cu` into
   `engines/ds4/ds4_cuda.cu`. Diff first; the files are 46k and 33k lines and diverge.
6. Teach `engines/ds4/ds4.c` to accept `deepseek41.expert_layout = flashinfer-sm120-mxfp8-mxfp4-v1`.
   Only `engines/ds4-win/ds4.c:3066` accepts it today.
7. Add an external decode library target so the producer loads decode kernels beside it,
   replacing `ds4-decode.dll`.
8. Delete `engines/ds4-win` and `engines/ds4-peer`. Carry their provenance notes from
   `engines/ds4-win/PROVENANCE.md` and `engines/ds4-peer` into `engines/ds4/PROVENANCE.md` and
   `NOTICE`, since the kernels came from NovaStar and IngeniousIdiocy.

Gate: `make -C engines/ds4 cuda CUDA_ARCH=sm_120` builds `ds4`, `ds4-server`, `ds4-bench`,
`ds4-eval`, `ds4-agent`, and `make -C engines/ds4 test-mxfp4-cuda` passes.
Note `ds4-server` needs `$(PAIRED_OBJS)` and `$(PAIRED_GGML)` on its link line; the Linux rules
at `engines/ds4-mac/Makefile:753-758` omit them and linking fails with undefined `ds4_chain_*`.

## Phase 2: engine host layer

Windows host code left after Phase 1 lives in `engines/ds4` and the streaming engine.

Substitutions:

| Windows | Linux |
|---|---|
| `CreateFileW`, `FILE_FLAG_NO_BUFFERING`, `FILE_FLAG_RANDOM_ACCESS` | `openat`, `O_DIRECT` |
| `GetDriveTypeW`, `QueryDosDeviceW`, `BY_HANDLE_FILE_INFORMATION`, volume serial and index | `fstat`, `st_dev`, `st_ino`, `realpath`, refuse `S_ISLNK`, pin ancestor fds |
| `GetDiskFreeSpaceW` sector alignment | `statfs` or `ioctl BLKSSZGET` |
| `_aligned_malloc` / `_aligned_free` | `posix_memalign` / `free` |
| `VirtualAlloc(MEM_RESERVE\|MEM_COMMIT)` | `mmap` plus `mlock`, or `memfd_create` |
| IoRing | io_uring: `IORING_OP_READ`, registered files, fixed buffers |
| `TerminateProcess` in cleanup paths | `_exit`, same refusal semantics |
| `Sleep`, `GetTickCount64` | `nanosleep`, `clock_gettime` |
| `CRITICAL_SECTION`, `SRWLOCK`, `Interlocked*` | pthreads and atomics |
| `linenoise_win32.c` | POSIX `linenoise.c` |
| `ds4_tp_win32.c`, `ds4_distributed_win32.c` | pthreads and sockets |

Gate: `grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing\|__APPLE__\|TARGET_OS\|Metal" engines/`
returns nothing.

## Phase 3: weight fetcher and producer

Remaining in `engines/streaming-llama/prefill_llama.cpp`:

```
src    324 refs. llama-weight-fetcher.cpp, llama-wf-gpu-ring.inl, llama-wf-zero-gaps.h,
       llama-wf-bounded-policy.h, llama-dsv41-engram-prefetch.cpp (108 refs),
       llama-dsv41-weights.cpp (throws at :923 and :1000), llama-dsv41-carry.cpp,
       llama-dsv41-locked-ram.h, llama-dsv41-host-cpu.h, llama-dsv41-host-memory.h,
       llama-dsv41-sweep.cpp, llama-dsv41-cache-import.cpp, llama-token-stream.cpp/.h
ggml   108 refs, six files matter: ggml-cuda.cu:8117, dsv41-flashinfer.cuh, dsv41-native-ffn.cuh,
       dsv41-dense-fp8.cuh, ds4split-expert-projection.cuh, ds4split-mmid-observer.cuh
tools   54 refs: dsv41-prefill/main.cpp (windows.h, bcrypt.h, dxgi1_4.h), producer-abort.h,
       live-session.inc, carry-copy-bench.cpp, decode-receiver/main.cpp, server/*, mtmd/*
```

1. Implement `wf_io_uring_worker` to replace the IOCP workers and the `pread` fallback pool at
   `llama-weight-fetcher.cpp:1205`. `WF_MODE_AUTO` prefers GDS only when cuFile reports support.
2. Delete `llama-wf-gpu-ring.inl` and the DStorage interop paths. Drop the
   `--ds-gpu-ring-slots` producer option rather than faking it.
3. Port `llama-dsv41-engram-prefetch.cpp` to io_uring with 1 MiB extents at QD64, and measure it
   against the `pread` baseline separately. That change is what took 76.2 s to 43.2 s.
4. Replace the provider lease and the two Windows throws in `llama-dsv41-weights.cpp`.
5. Replace `llama-dsv41-locked-ram.h` and the carry lease in `llama-dsv41-carry.cpp`.
6. Replace BCrypt SHA-256 in `tools/dsv41-prefill/main.cpp` and `src/boundary-capture.cpp` with
   OpenSSL EVP or a vendored SHA-256. The producer binds its own executable SHA.
7. Remove DXGI and Windows includes in the producer, `mtmd` and `server`.
8. Also delete the Mac path `tools/dsv41-prefill/mac/`.
9. Add `llama-dsv41-prefill` to the targets in `build.sh` and run
   `build/llama/bin/llama-dsv41-prefill --check-native-profiles`.

Gate: the producer links, passes the profile preflight, and completes a 262K prefill.

## Phase 4: prune ggml backends

Keep CUDA and CPU. Delete `ggml-vulkan`, `ggml-sycl`, `ggml-hexagon`, `ggml-cpu/kleidiai`,
`ggml-rpc`, `ggml-cpu/amx`, the Metal backend in `ggml-bench` and the `GGML_METAL`, `GGML_VULKAN`,
`GGML_SYCL`, `GGML_RPC`, `GGML_HEXAGON` options in `ggml/CMakeLists.txt`. Set the build to
`-DGGML_CUDA=ON` only. `build.sh` already does that.

Gate: cmake configures and builds with no backend options other than CUDA.

## Phase 5: Python orchestrator

Windows-specific code left, largest first:

```
guard.py          24 refs: kernel32 CreateJobObjectW, SetInformationJobObject, AssignProcessToJobObject.
                  Replace with a cgroup v2 scope, systemd-run --scope --memory=..., or memory.max plus
                  killing the cgroup on teardown. lease.py depends on the guardian killing GPU descendants.
single_runtime.py 13 refs: PRODUCER_FILES = ds4-decode.dll, dstorage.dll, dstoragecore.dll
memory_counters.py 9 refs: ctypes.WinDLL('psapi.dll'). Linux path is psutil plus /sys/meminfo.
lease.py          6 refs: msvcrt branch, fcntl branch already exists
journal.py        6 refs: Windows rename-race handling
process.py        6 refs: scope names including ds4-win
working_set.py    5 refs, shared_guard.py 5 refs: Windows working-set and Job accounting
combo_native.py   5 refs: ds4-win env scope
combo_ram.py      4 refs: refuses non-NT at :86, SetProcessWorkingSetSizeEx. Linux equivalent is cgroup
                  memory.min plus mlock.
lora.py           4 refs: Windows drive-letter handling
generation.py     4 refs: msvcrt branch, rename-race comment
```

macOS-specific code to delete:

```
slstack/mac.py            556 lines, the Mac receiver, relay, pairing printout
slstack/pairing.py        paired RTX side, unless two-machine mode stays
slstack/native_broker.py  the producer broker, unless two-machine mode stays
slstack/backend_setup.py  darwin gate at :225, Metal cmake flags at :102
slstack/mlx_setup.py, omlx_setup.py, decode.py darwin branch at :49
slstack/backends.py       mlx profiles
slstack/memory_counters.py darwin reclaimable counters
slstack/engines.py, combo_*.py, media_*  MLX and Mac paths
slstack/mcp.py tools      Mac media tools, if present
```

Also `scripts/bootstrap.py`, `scripts/setup-mlx.py`, `scripts/bootstrap.py setup-backend` backends
`mlx-receiver`, `mlx-serve`, `omlx`, `mference`.

Gate: `./bin/slstack --config <file> check` runs on Linux, and the Python suite in `tests/` passes.

## Phase 6: vendor and engines bloat

Delete:

- `vendor/mac-mlx`, 11 MB, plus `patches/mlx-core-ring-listeners.*`
- `engines/mlx-receiver`, 396 KB
- `engines/mlx-serve` submodule and the `.gitmodules` entry
- `engines/mference`, 2.6 MB Swift package
- `locks/mac-mlx-py312.txt`, `locks/mac-omlx-py312.txt`
- `vendor/engines/cosmos_windows` Windows-only media paths, or port them to Linux. Media
  generation is experimental in the README, so this can lag inference.

Gate: `git ls-files | wc -l` drops by roughly the count of deleted MLX, Metal and Swift files,
and the repo size drops from about 150 MB.

## Phase 7: docs, examples, presets, tests

- `docs/`: 5,925 Apple/Metal references. Delete `MACDECODE*.md`, `docs/macdecode-r*`,
  `macdecode-measurements.json`, `docs/MEDIA_SETUP_VERIFICATION.md` Mac sections,
  `docs/SUPPORTED_MODELS.md` Mac decode rows, `docs/TWO_MACHINE.md` unless pairing stays.
  Keep measurement history labelled as inherited from the Windows and macOS builds.
- `examples/`: delete `examples/dual-machine/*/mac-big-memory.sh`, the Mac halves of the paired
  presets, `examples/engine_environment/mac-*.json`, `examples/macdecode-r3`,
  `examples/dual-machine/dsv41-ultra-cone/proof/mac-*.json`.
- `presets/dsv41-mac.json` delete.
- `tests/`: 75 Apple/Metal references, plus `engines/ds4-mac/tests/*` and `engines/ds4-win/tests/*`
  disappearing with the engines. Port what remains to Linux gates.
- Regenerate `docs/VARIABLES.md` after the scope rename.

## Phase 8: qualification

Nothing measured transfers. Re-run against the hash-pinned fixtures and write new receipts:

1. 262K exact-cone single engine, three runs, prompt clock and decode rates.
2. 200K real-text fixture, password recall, logits SHA-256.
3. Engram extent reads against the `pread` baseline, so the io_uring gain is isolated.
4. Paired run only if two-machine mode stays, KL within mean 0.1 and max 0.5 nats.
5. Update `docs/PERFORMANCE.md` and the README tables only from these receipts.

## Sequencing and risk

Phases 1, 2, 3 make the inference path work. Phases 4, 5, 6, 7 are the bloat reduction and can
run in any order after that. Phase 8 last.

Costs that could exceed expectations:

- The CUDA merge. ds4-mac and ds4-win sources diverge by tens of thousands of lines, so it is a
  port, not a copy.
- io_uring parity. The measured Windows numbers came from a RAID-0 at about 25 GB/s. A single
  NVMe with io_uring may not match, so prefill will move regardless of correctness.
- Locked memory. `mlock` needs `RLIMIT_MEMLOCK` or a cgroup allowance. Decide the mechanism before
  rewriting `combo_ram.py` and `locked-ram.h`.
- Losing decode for Kimi, GLM, MiniMax and Step. That is a feature removal, not a port, so settle
  decision 2 first.

## How to resume this plan

```sh
cd streaming-llama-stack
git log --oneline -1
grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing\|msvcrt\|WinDLL" engines slstack vendor scripts examples tests | wc -l
grep -rn "__APPLE__\|TARGET_OS\|Metal\|mlx\|MLX\|darwin\|Darwin" engines slstack vendor scripts examples docs tests | wc -l
python3 -B scripts/release_check.py
```

At Phase 0 the Windows count is about 940 and the Apple/MLX count is about 12,300.
