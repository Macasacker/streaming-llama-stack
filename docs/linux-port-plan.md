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

### Decision taken on which engine is the base (2026-10-30, evidence)

Both engines fail to build on Linux today, so neither is a drop-in.

- `make -C engines/ds4-mac cuda CUDA_ARCH=sm_120` fails in `ds4.c`. `ds4.c` calls six `ds4_gpu_*`
  functions whose declarations sit inside `#ifdef __APPLE__` in `ds4_gpu.h` (the block at 158-277):
  `ds4_gpu_dsv41_matmul_q8_0_bf16_rows` (`ds4.c:41397`), `ds4_gpu_dsv41_indexer_topk_tensor`
  (`:42079`), `ds4_gpu_tp_flag_fold_request` (`:42272`), `ds4_gpu_add_tensor_tp_flag` (`:42505`),
  `ds4_gpu_pq_probs` (`:89469`), `ds4_gpu_pq_cand` (`:89471`). GCC 14 makes the implicit
  declaration an error, so the Mac engine's Linux CUDA branch has never been compiled.
- `make -C engines/ds4-win cuda CUDA_ARCH=sm_120` compiled every C/C++ TU and failed only in
  `ds4_cuda.cu`: 12 errors, all identifiers defined inside the `#if defined(_WIN32)` block that
  ran 5933-6711 but used outside it (`g_ds_decode_bandwidth`, `g_prefill_pipeline`,
  `ds4_ds_batch_inflight`, `g_ds_prefetch_*`, `cuda_ds_prefetch_reset`) plus one "transfer of
  control bypasses initialization". Chunk 1.7 resolved this: those blocks are platform-neutral
  code that only calls the `ds4_ds_*` transport API, so the guard came off rather than the code.
- `cuda/mmq/` is near-identical between the two engines: only `mmq.cuh` differs, by 18 lines.
  The Windows copy adds a K2304 row-stride guard so a partial K row never loads the next
  expert's codes or scales (`mmq.cuh:973-986`). So the packed FP4 and Q4K-half kernels move as
  files; there is no 46k-line merge.

Base is therefore `engines/ds4-win`, renamed `engines/ds4`, and `engines/ds4-mac` is deleted in
the same pass. That keeps option 3's end state - exactly one engine - without the kernel merge,
keeps `flashinfer-sm120-mxfp8-mxfp4-v1` acceptance (`ds4.c:3066`) and the packed decode kernels,
and drops 4,114 Apple/Metal references instead of 6,091. Steps 4, 5 and 6 below are superseded.

### Chunks

1.1 Skeleton and build rules. Done: `engines/ds4/` copied from `engines/ds4-win` (154 files,
    no `.o`), Metal/ROCm/Win32 files deleted (`metal/`, `rocm/`, `ds4_metal.m`, `ds4_rocm*`,
    `linenoise_win32.c`, `ds4_distributed_win32.c`, `ds4_tp_win32.c`, `ds4_win32.h/.c`,
    `ds4_dstorage.h/.cpp`, `ds4_engram_ioring.cpp`, `ds4_combo_launcher.c`), Makefile rewritten
    Linux-only (`cuda`, `cuda-generic`, `cpu`, `cuda-decode`, `test`, `clean`; no Darwin branch,
    no spark/strix/rocm/metal targets). CUDA is at `/opt/cuda` on this box, so `CUDA_LDLIBS`
    gained `-L$(CUDA_HOME)/lib` as well as `lib64`.

1.2 Host-memory admission. Done: `ds4_host_memory.h/.c`, the Linux twin of the Windows guard.
    MemAvailable for the physical reserve, VmRSS/VmPeak for process private, cgroup v2
    `memory.max`/`memory.current` for system commit, `DS4_HOST_MEMORY_BUDGET_MB` override, a
    250 ms watchdog that requests the graceful abort, and the same owner rule: never force-kill
    while VRAM may be held. Compiles clean.

1.3 io_uring transport. Done: `ds4_batch_io.h/.cpp` implements the whole `ds4_ds_*` API the
    CUDA path calls outside any platform guard - init from fd with `fstat` identity and re-init on
    an fd swap, 8-slot demand FIFO, span splitting into parts, device arena with a host mirror
    plus a `cudaEvent` recorded after the upload (the Linux stand-in for the D3D12 fence),
    prefill waves, one background batch, byte counters, slow-batch lines, layer trace, and the
    QD1-versus-batched self-test against `pread`. Compiles clean; not yet run against a model.
    Reads are buffered - the engine opens the model `O_RDONLY`. An `O_DIRECT` variant with
    4 KiB-aligned offsets and buffers is separate follow-up work.

1.4 Call-site substitution. Done for the host-memory and transport surface: `ds4.c`,
    `ds4_cuda.cu`, `ds4_cli.c`, `ds4_server.c`, `ds4_bench.c`, `ds4_agent.c`, `ds4_help.c`,
    `ds4_peer.c`, `ds4_web.c`, `ds4_engram.c`, `cuda/mmq/ds4_ggml_stubs.cu`,
    `cuda/mmq/ds4_repack.cu`. `ds4_win32_host_memory_*` became `ds4_host_memory_*`, admission
    guards that were Windows-only now run on Linux, `ExitProcess(72)` became `_exit(72)`,
    `ds4_dstorage.h` became `ds4_batch_io.h`, the Windows-only `DS41_ALLOC` host-slab branch was
    dropped (Linux keeps the managed GPU tensor), and the MMQ pool fixture is enabled on Linux.

1.5 Engram io_uring reader. Written but not wired: `ds4_engram_iouring.c` exports
    `ds4_engram_ioring_read_rows(fd, base, rows, n, out, why, why_n)` at QD64 over 264-byte rows.
    Needs a Makefile rule, and `ds4_engram.c` still gates its ioring call sites on
    `#if defined(_WIN32)` (around lines 257 and 346) - change to `__linux__`.

1.6 Decode library. `make cuda-decode CUDA_ARCH=sm_120` builds `ds4-decode.so`, the dlopen twin
    of `ds4-decode.dll`. Not yet compiled; needs `DS4_EXTERNAL_DECODE` and the `ds4x_*` entry
    points in `ds4_decode_adapter.inc`, and the producer's loader path in
    `engines/streaming-llama` must use `dlopen`/`dlsym` instead of `LoadLibrary`/`GetProcAddress`.

1.7 First CUDA build. Done. `make -C engines/ds4 cuda CUDA_ARCH=sm_120` builds `ds4`,
    `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent` (56-58 MB each) and `make cpu` builds the
    CPU variants. `./ds4 --help` runs.

    What the build needed:
    - `ds4_cuda.cu` had three platform guards that gated platform-neutral code, not Windows code.
      Removed the guard at 5921 (its `#endif /* _WIN32 */` at 6699 plus an orphan `#else` branch
      holding `cuda_ds_prefetch_copy_expert_to_device` stubs), the nested guard at 6032..6312, and
      the guard at 12510..13542 that held `ds4_gpu_prefill_pipeline_begin/layer_done/wait_layer/
      finish/release`. Their internals call only the `ds4_ds_*` transport API, so the io_uring
      transport is what they now run on. Scan of those blocks found no Win32 or D3D calls.
    - One C++ error: `goto done` jumped over `const int logical_tier` initialization at
      `ds4_cuda.cu:27829`. Declared `int logical_tier = 0` above the checks and assigned after.
      Be careful editing near it: the same variable name appears in ~40 other functions, and a
      plain string replace hits the wrong one (26804 in `ds4_gpu_attention_noncausal_raw_batch_heads_tensor`).
    - `cuda/mmq/ds4_ggml_stubs.cu` needed `<unistd.h>` and the host-memory header unguarded.
    - Link additions: `-luring`, `ds4_image.o`, `ds4_prompt_prefix.o`, `ds4_engram.o`,
      `ds4_engram_iouring.o`, and `ds4_eval_cases.c/.h` copied from `engines/ds4-mac` (the
      Windows engine never had them, but `ds4_eval.c` needs `eval_hard_cases`).

1.6 Decode library. Open: `make cuda-decode CUDA_ARCH=sm_120` builds `ds4-decode.so`. Not yet
    compiled. Needs `DS4_EXTERNAL_DECODE` and the `ds4x_*` entry points from
    `ds4_decode_adapter.inc`, and the producer's loader in `engines/streaming-llama` must use
    `dlopen`/`dlsym` instead of `LoadLibrary`/`GetProcAddress`.

1.8 Delete the old engines. Open: delete `engines/ds4-mac` and `engines/ds4-peer`, carry
    provenance from `engines/ds4-win/PROVENANCE.md` and `THIRD_PARTY_NOTICES.md` into
    `engines/ds4/PROVENANCE.md` and `NOTICE`, rename the `ds4-win` scope key to `ds4` in
    `slstack/envvars.py`, `process.py`, `combo_native.py`, `general_engine.py`, `engines.py`,
    `check.py` and the presets, then regenerate `docs/VARIABLES.md`.

Gate for chunk 1.7, met: `make -C engines/ds4 cuda CUDA_ARCH=sm_120` builds `ds4`, `ds4-server`,
`ds4-bench`, `ds4-eval`, `ds4-agent`. `make -C engines/ds4 cpu` builds the CPU variants.
`make test-mxfp4-cuda` cannot run: `tests/test_mxfp4_cuda.cu` is not in the published tree (the
engine `tests/` directory holds only 8 files), so that target says skipped instead of failing on a
missing source.

Known gaps carried forward, not hidden:

- The io_uring transport is correct-by-construction but unmeasured. No model file exists on this
  box, so `ds4_ds_selftest` and the decode pipeline have never run against real weights.
- Reads are buffered. `O_DIRECT` with 4 KiB-aligned offsets and buffers is a separate chunk.
- `ds4_cuda.cu` still has 34 `#if defined(_WIN32)` guards and `ds4.c` has Windows host code left;
  they compile away on Linux. Phase 2 deletes them, and the Phase 2 grep gate cannot pass until
  then.
- `./ds4 --help` still advertises `--metal`. Help text and CLI backend names are Phase 2 work.
- `engines/ds4-win`, `engines/ds4-mac` and `engines/ds4-peer` still exist until chunk 1.8 lands,
  so the fork temporarily has four engines.

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

### Chunk backlog, in the order worth doing

Each chunk is sized to fit one working session and ends with a gate. Tick them in this file.

1.10a Delete the engine trees. Done (`3d36bd6`): `engines/ds4-win`, `engines/ds4-mac`,
    `engines/ds4-peer`, `engines/mlx-receiver`, `engines/mlx-serve`, `engines/mference` and
    `engines/upstream-decode-receiver` are gone, `.gitmodules` deleted with the mlx-serve
    gitlink, and `engines/ds4/PROVENANCE.md` rewritten for the Linux engine. 1.04M lines removed.

1.10b One scope, Linux paths. Done (`6dc410f`): the scope key `ds4-win` is now `ds4` in
    `envvars.py`, `process.py`, `engines.py`, `combo_native.py`, `check.py` and the tests;
    `SCOPE_ALIASES` lost `ds4-mac -> mac-ds4`; `build.sh` builds `engines/ds4`;
    `config.example.json` points at `engines/ds4/ds4-server`. `envvars.json` went 3335 -> 2233
    rows (Metal, ROCm, peer, MLX and Mac-toolchain variables no longer read anywhere in
    `engines/` or `vendor/`), plus one new row for `DS4_DISABLE_HOST_MEMORY_WATCHDOG`.
    `docs/VARIABLES.md` regenerated, `scripts/audit_backend_sources.py` rewritten for the two
    Linux engines.

    Surprise worth remembering: dropping a row is safe only when nothing reads the variable. The
    first pass keyed on the `engines/`+`vendor/` scan alone and deleted rows the launcher still
    owns through `slstack/` (the scan does not cover `slstack/`). Those rows came back.

1.10c Delete the Mac Python surface. Done (`4abb73f`). Two-machine mode cannot survive in a
    Linux-only fork, so decision 1 resolves to single machine only. Deleted `slstack/{mac,
    pairing,paired_generation,paired_vision,model_compatibility,native_relay,mlx_setup,
    mlx_serve_setup,omlx_setup,general_engine,general_templates}.py`,
    `vendor/studio/studio/{peer_launch,mlx_model_presets}.py`, `vendor/mac-mlx`,
    `locks/mac-*.txt`, `presets/dsv41-mac.json`, `examples/dual-machine`,
    `examples/macdecode-r3`, and 14 Mac/paired test files. `backends.py` is now `ds4` and
    `llama-server`; `studio/mac_argv.py` emits `--cuda`; `backend_setup.py` builds
    `engines/ds4` with `make cuda` (arch probed from `nvidia-smi --query-gpu=compute_cap`,
    `sm_120a` fallback) and `llama-server` with `GGML_CUDA=ON`. `examples/run_preset.py` takes
    `--single` and `--setup` only.

    Two deliberate behaviour changes, both raise instead of degrading silently: a `mac` config
    setting is refused (`slstack/config.py`), and image spans through the native broker raise
    (`native_broker.stage_vision`) because SLVP projection was the Mac receiver's job -
    single-engine serving stages images through `slstack/vision.py`.

1.10d Dead Mac and Windows branches in the launcher. Open: unreachable `cfg.get('mac')` branches
    at `slstack/engines.py:81,199,340` and `slstack/router.py:64-65,137-138,168,313,324`, the
    `os.name == 'nt'` branch at `slstack/engines.py:85`, Mac-local token counts in
    `combo_crossover.py`, and the `mac=` hold-ledger semantics in `lease.py`. Gate: the Python
    suite stays at or below 6 failures / 5 errors.

1.9 Settled by chunk 1.10c. Decision 1 is (b), single machine only. Decision 2: the supported
    list is DeepSeek V4.x on ds4 CUDA; Kimi, GLM, MiniMax and Step lose their MLX decode, and
    `general` mode now raises. Decision 3: ROCm stays deleted, there is no AMD hardware here.

2.1 Remaining Win32 host code in the engine: `ds4.c` (~50 refs), `ds4_cuda.cu` (34 guards),
    `ds4_cli.c`, `ds4_server.c`, `ds4_agent.c` (drive-letter path handling), `ds4_web.c`,
    `ds4_ssd.c`, `ds4_kvstore.c`, `ds4.h`, `ds4_gpu.h`, `ds4_peer.c` (deleted with 1.9),
    `ds4_combo_api.inc` (Windows-only; delete). Gate: `grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing" engines/ds4` returns nothing.

2.2 Apple/Metal dead code in the engine (4,109 refs, most inside `#ifdef __APPLE__` blocks in
    `ds4.c`, `ds4_gpu.h`, `ds4_tp.c`, `ds4_image.c`, `ds4_web.c`, `ds4_cli.c`, `ds4_help.c`).
    Gate: `grep -rn "__APPLE__\|TARGET_OS\|Metal\|metal" engines/ds4` returns nothing, and
    `make cuda CUDA_ARCH=sm_120` still builds.

3.1 Streaming producer on Linux. `engines/streaming-llama/prefill_llama.cpp`: implement
    `wf_io_uring_worker` beside the `pread` pool at `llama-weight-fetcher.cpp:1205`, delete
    `llama-wf-gpu-ring.inl` and the DStorage interop and the `--ds-gpu-ring-slots` option, port
    `llama-dsv41-engram-prefetch.cpp` (108 Win32 refs), replace the provider lease and the two
    throws in `llama-dsv41-weights.cpp` (`:923`, `:1000`), replace `llama-dsv41-locked-ram.h` and
    the carry lease, replace BCrypt SHA-256 with OpenSSL EVP in `tools/dsv41-prefill/main.cpp` and
    `src/boundary-capture.cpp`, drop DXGI/Windows includes in the producer, `mtmd` and `server`.
    Gate: the producer links and passes `--check-native-profiles`. A 262K prefill needs weights.

3.2 Producer decode library wiring: `dlopen`/`dlsym` for `ds4-decode.so` in place of
    `LoadLibraryW`/`GetProcAddress`, and `make cuda-decode` producing it.

4.1 ggml backend prune: keep CUDA + CPU; delete Vulkan, SYCL, Hexagon, KleidiAI, RPC, AMX, Metal
    in `ggml-bench`, and the corresponding options in `ggml/CMakeLists.txt`.

5.1 Python orchestrator: `guard.py` Job Object -> cgroup v2 scope via `systemd-run --scope`,
    `single_runtime.py` PRODUCER_FILES, `memory_counters.py` psutil path, `lease.py` msvcrt branch,
    `journal.py` rename-race handling, `working_set.py`/`shared_guard.py`, `combo_ram.py:86`,
    `lora.py` drive letters, `generation.py` msvcrt branch.
    Gate: `./bin/slstack --config <file> check` runs on Linux and the Python suite passes.

6.1 Vendor and engine bloat: `vendor/mac-mlx`, `engines/mlx-receiver`, `engines/mlx-serve`,
    `engines/mference`, the MLX lock files, `patches/mlx-core-ring-listeners.*`.

7.1 Docs, examples, presets, tests: delete the Mac/paired halves, regenerate `docs/VARIABLES.md`,
    relabel inherited Windows/macOS measurements.

8.1 Qualification. Needs a qualified GGUF on disk: 262K exact-cone runs, 200K real-text fixture
    with password recall and logits SHA-256, Engram extent reads against the pread baseline, then
    rewrite `docs/PERFORMANCE.md` and the README tables from those receipts only.

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

## Machine facts checked on this box (2026-10-30)

Arch Linux, kernel 7.2.11-arch1-1, RTX 5090, driver 615.78, CUDA 13.4 at `/opt/cuda` (not
`/usr/local/cuda`), nvcc 13.4 with `sm_120`/`sm_121`, cmake 4.4.5, ninja 1.13.2,
`/usr/include/liburing.h` present, Python 3.14.8, ext4 on one NVMe with 793 GB free, 30 GB RAM.
`nvidia-fs` is not installed, so GDS is unavailable; io_uring is the route.

Two things limit what a gate can prove here:

- The GPU is occupied by another process (`strata`, 29.3 GiB of 32 GiB VRAM) and RAM is at
  22 GiB used, ~7.8 GiB available. Compile gates run; anything that allocates VRAM or mlocks
  large buffers will contend with it.
- No DeepSeek V4 weights exist locally. Phase 3 and Phase 8 gates cannot run until a qualified
  GGUF is present.

## How to resume this plan

```sh
cd streaming-llama-stack
git log --oneline -1
git status --short                 # Phase 1 chunks 1.1-1.5 land here
grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing\|msvcrt\|WinDLL" engines slstack vendor scripts examples tests | wc -l
grep -rn "__APPLE__\|TARGET_OS\|Metal\|mlx\|MLX\|darwin\|Darwin" engines slstack vendor scripts examples docs tests | wc -l
python3 -B scripts/release_check.py
```

Counts: Phase 0 about 940 Windows references and about 12,300 Apple/MLX references. After chunk
1.7 the repo is 1,131 Windows and 13,878 Apple/MLX, with `engines/ds4` at 162 Windows and 4,109
Apple/Metal references. After chunks 1.10a, 1.10b and 1.10c: 495 Apple/MLX references outside
`engines/ds4` (mostly `engines/streaming-llama` and vendored Studio), 281 Windows references
outside `engines/ds4`, `engines/` holds only `ds4` and `streaming-llama`, and the working tree is
about 518 MB with a 55 MB `.git`. Re-count before trusting any of these numbers.

`python3 -B scripts/release_check.py` reports 0 findings since `docs/VARIABLES.md` was regenerated
in chunk 1.10b.

Python suite: `timeout 600 env PYTHONPATH=.:tests python3 -m unittest discover -s tests -p
'test_*.py'` runs 677 tests with 5 failures and 6 errors. The fork baseline was 8 failures and 6
errors, so nothing regressed. The survivors are Linux gaps, not Mac leftovers: Windows-only guard
and producer paths (`test_combo_process`, `test_shared_guard_adapter`, `test_single_liveness`,
`test_journal_recovery_flags`), the missing `cosmos_windows` vendor module
(`test_streamed_recipe_argv`, `test_dsv41_recipe`, `test_yoco`), and GPU/video gates that need a
model on disk (`test_h3_video_delivery`, `test_yoco_transfer`). Chunk 5.1 and 8.1 own them.

Build a CUDA engine with `make -C engines/ds4 cuda CUDA_ARCH=sm_120 -j4` (~4 min). `make cpu`
builds the same binary names as CPU variants, so rebuild CUDA afterwards if you need the CUDA
binaries.

`.githooks` is wired through `core.hooksPath`, so `pre-commit` runs `release_check.py` and will
refuse a commit while the VARIABLES finding stands. Either regenerate the table first or commit
the work-in-progress chunks with `--no-verify` and say so in the commit message.
