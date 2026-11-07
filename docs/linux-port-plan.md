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
- Other ggml backends that are not CUDA: Vulkan, SYCL, Hexagon, KleidiAI, RPC, AMX, BLAS,
  CANN, HIP, MUSA, OpenCL, WebGPU, zDNN, ZenDNN, VirtGPU, CPU-only
  MoE paths that exist to serve Mac and Windows hosts. Chunks 4.1 and 4.2 deleted the trees;
  what is left of that scope is the CUDA backend's AMD and Moore Threads compute-capability
  vocabulary, recorded under chunk 4.2.
- ROCm and DGX Spark targets. Linux, but AMD and sm_121 paths nobody here will run. Say the
  word if ROCm should stay.

What survives: CUDA + CPU ggml, the streaming llama producer, one ds4 CUDA engine, the Python
orchestrator, the web UI, MCP, LoRA, and images.

## Machine facts

Arch Linux, kernel 7.2.14-arch1-1, RTX 5090 32 GB (32,607 MiB), compute capability 12.0,
driver 615.78.08, CUDA 13.4.92 (`nvcc --list-gpu-code` shows `sm_120` and `sm_121`;
`compute_120a` compiles), cmake 4.4.5, ninja 1.13.2, `/usr/include/liburing.h` present,
Python 3.14.9 (the stack wants 3.12), ext4 on a single 1.8 TB NVMe with 333 GiB free, 93 GB
RAM, 16 cores.
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
    of `ds4-decode.dll`, exporting the `ds4x_*` entry points in `ds4_decode_adapter.inc`. The
    producer's loader path in `engines/streaming-llama` must use `dlopen`/`dlsym` instead of
    `LoadLibrary`/`GetProcAddress`; that part is chunk 3.2.

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

1.6 Decode library. Done. `make -C engines/ds4 cuda-decode CUDA_ARCH=sm_120` builds
    `ds4-decode.so` (59 MB, 19 `ds4x_*` entry points). It is a second compilation of the engine:
    `ds4_ext.o`, `ds4_cuda_ext.o`, `ds4_host_memory_ext.o` with `-DDS4_EXTERNAL_DECODE=1`, plus
    the shared objects the entry points reach (`ds4_batch_io.o`, `ds4_engram*.o`,
    `ds4_prompt_prefix.o`, `ds4_cpu_fp4.o`). Every object is now `-fPIC`, which a shared library
    needs once it contains TLS. What landed with it: the upload stream is created lazily so a
    host-destination batch never makes a CUDA context, `batch_add`/`batch_add_gpu` refuse at the
    logical capacity `ds4_ds_batch_capacity()` reports, `ds4x_open` refuses a non-GGUF path
    instead of letting the loader `exit(1)` the producer, and the Win32 intrinsics in the external
    host cache, host compute and CPU FP4 lanes are POSIX (`mlock`/`munlock`, `setpriority`,
    `__cpuid_count`). Nothing in `engines/streaming-llama` loads the library yet, so
    `tests/test_decode_library.cpp` (dlopen + the `ds4x_*` ABI) is the gate until chunk 3.2; the
    producer still decodes through its own ggml CUDA path.

    Gate met: `make cuda CUDA_ARCH=sm_120 -j4`, `make ds4-decode.so CUDA_ARCH=sm_120`,
    `make test-batch-io`, `make test-decode-library`, `make test-cpu-fp4` all pass, with Strata
    holding the GPU. The `mlock` of the pinned host tier needs an `RLIMIT_MEMLOCK` or cgroup
    allowance - still open, listed under the locked-memory risk.

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
- What Linux does not implement, because the only implementation was Windows-only and chunk 2.1
  deleted it: the two-process combo handoff, expert-cache restore and snapshot at relaunch,
  background affinity promotion, and output LoRA. Combo, background promotion and output LoRA
  refuse, with a message naming the build (`DS4_COMBO_ENABLED=1` now fails startup with a reason
  instead of an empty one). Expert-cache restore and snapshot are inert rather than refusing:
  `ds4_expert_cache_restore_prepare` and `ds4_expert_cache_snapshot_write` have no caller on
  Linux, so `DS4_COMBO_EXPERT_RESTORE_FILE` and `DS4_COMBO_EXPERT_SNAPSHOT_FILE`, which
  `slstack/combo_native.py` and `slstack/engines.py` set for a combo run, do nothing until the
  combo path exists. Porting any of them is new work, not a port - decide which are wanted before
  Phase 5 rewrites `combo_ram.py`.
- `--ds-read-parts` is parsed but refused on Linux; `DS4_DS_READ_PARTS` is the only way to set the
  transport's span count. Making the flag call `ds4_ds_configure_read_parts` before the transport is
  initialised is a small, separate change.
- `./ds4 --help` no longer names Metal (chunk 2.2c). What it still advertises without hardware or
  an implementation here: `--rocm` (an alias for CUDA, no AMD device on this box),
  `--ssd-streaming-full-layers` (parses, then is ignored because
  `ds4_backend_supports_glm_streaming_full_layers()` is false), `--expert-profile` (parses, then
  refused with "not implemented in this build"), and `--imatrix-*` (refused while parsing, before
  the model is opened). `--gpu-graph-test`, `--gpu-graph-full-test` and `--gpu-graph-prompt-test`
  are now genuinely reachable; none has been run, because they need a model file.
- Four engine subsystems are inert after chunk 2.2c rather than deleted, because deleting them
  would have cascaded through live code: the routed-expert locality profiler
  (`ds4_expert_profile_init` has no caller), sequential imatrix collection
  (`ds4_engine_collect_sequential_imatrix`, `imatrix_read_text_file`), the mixed prefill/decode
  command epoch (`ds4_sessions_eval_batch_with_prefill_gpu`), and the Pro-Q4 expert-table preload
  (its CUDA helpers keep other callers). All four are marked `DS4_MAYBE_UNUSED` and are the shape a
  port would reuse. Two more things went quiet with no message: `e->share_session_prefill_workspace`
  (the Qwen3.8 shared transient arena was the deleted backend's policy) and the engine's
  `glm_session_count` / `glm_session_graph_bytes` bookkeeping, whose only reader was
  `ds4_engine_glm_graph_budget`.
- `engines/ds4-win`, `engines/ds4-mac` and `engines/ds4-peer` are deleted (chunk 1.8 settled by
  1.10a); `engines/` holds `ds4` and `streaming-llama` only.

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

Gate: `grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing" engines/` returns nothing except
the three vendored hits. Chunk 2.2e is finished: no CLI surface, no code identifier and no printed
message names Metal, and `scripts/port/metal_word_census.py --strict` is the gate for that rather
than a grep. The gate as first written also grepped for `Metal`, which cannot pass: the owner kept
the 215 `DS4_METAL_*` environment names in chunk 2.2b because they are documented in
`docs/VARIABLES.md` and named in `docs/macdecode-r6/r6-controls.json`. What still names Metal is
those environment names themselves and 26 provenance comments.

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

1. Done (3.1e). `wf_io_uring_worker` replaced the IOCP workers and the `pread` pool at
   `llama-weight-fetcher.cpp:1205`: one ring per reader thread, `io_queue_depth` reads in
   flight on each, registered fds where the kernel accepts them, and the `pread` workers kept
   as the baseline when it does not (`LLAMA_WF_IO_URING=0`). `WF_MODE_AUTO` now calls
   `cuFileDriverOpen()` while resolving the mode and prefers GDS only when cuFile reports
   support, instead of choosing GDS because `cufile.h` exists.
2. Done (3.1e). `llama-wf-gpu-ring.inl`, the DStorage interop and the `ds_gpu_ring_slots`
   parameter deleted; `--ds-gpu-ring-slots` is gone from the producer rather than faked.
3. Done (3.1g). `llama-dsv41-engram-prefetch.cpp` reads its planner extents through io_uring
   queues, one per ring, with the serial `pread` kept as the baseline
   (`LLAMADECODE_ENGRAM_IO_URING=0`). Its platform-neutral row half - the hash-to-extent work,
   the admission arena and the prior-cache row copy - is byte-identical after the port. The
   cold-cache comparison against `pread` on a qualified GGUF is still Phase 8 item 3's: on this
   box the fixture is page-cache resident, so the two readers agree. `load_early`/`load` now take
   a `std::filesystem::path` and an `int` descriptor, which is what the provider's lease holds.
4. Done (3.1f, 3.1g). The provider's physical lease, `inspect`, `create_pipeline`,
   `acquire_layer_experts` and `configure_host_cache` are Linux, and the three Engram arms
   (`launch_early`, `start_engram_prompt_early`, the synchronous load in `prepare_engram_prompt`)
   are open. `src/llama-dsv41-weights.cpp` holds no Win32 conditional.
5. Done (3.1h). `llama-dsv41-locked-ram.h` and the carry lease in `llama-dsv41-carry.cpp` are
   Linux, and the tier locks in chunks instead of all-or-nothing (the locked-memory decision, and
   chunk B.4 of `docs/blackwell-enablement-plan.md`). Both carry gates pass.
6. Done (3.1c). BCrypt SHA-256 in `tools/dsv41-prefill/main.cpp` became
   `dsv41_boundary::sha256_native`, whose portable state branch `src/boundary-capture.cpp` already
   had and the producer already links. No OpenSSL, no vendored hasher. The Windows-only replay
   diagnostics in `boundary-capture.cpp` stay inert.
7. Half done. `tools/dsv41-prefill` is free of DXGI and Windows includes (3.1c, 3.1d).
   `mtmd`, `server` and `decode-receiver` still carry theirs.
8. Done (3.1c). `tools/dsv41-prefill/mac/` deleted; nothing referenced it.
9. Done (3.1b, 3.1c). `build.sh` names `llama-dsv41-prefill` and
   `build/llama/bin/llama-dsv41-prefill --check-native-profiles` passes on Linux.

Gate: the producer links, passes the profile preflight, and completes a 262K prefill. The first two
are met; the third needs a qualified GGUF on this box.

## Phase 4: prune ggml backends

Keep CUDA and CPU. Delete every other backend tree, its public header and the cmake option that
offered it: `ggml-vulkan`, `ggml-sycl`, `ggml-hexagon`, `ggml-cpu/kleidiai`, `ggml-rpc`,
`ggml-cpu/amx` and the Metal sources in chunk 4.1; `ggml-blas`, `ggml-cann`, `ggml-hip`,
`ggml-musa`, `ggml-opencl`, `ggml-webgpu`, `ggml-zdnn`, `ggml-zendnn` and `ggml-virtgpu` in
chunk 4.2. There is no `ggml-bench` in this fork, so the plan's original "the Metal backend in
`ggml-bench`" item had nothing to delete. Set the build to `-DGGML_CUDA=ON` only. `build.sh`
already does that.

Gate: cmake configures and builds with no backend options other than CUDA.

4.1 The ggml backend prune. Done. Eight directories and their headers are gone:
    `ggml/src/ggml-vulkan` (27986 lines, mostly `vk.cpp` and the generated
    `spv_*_comp.h` shader blobs), `ggml-sycl`, `ggml-hexagon`, `ggml-rpc`, `ggml-cpu/amx`,
    `ggml-cpu/kleidiai`, `ggml/src/ggml-metal` (the `.m` and `.metal` sources the build had
    already stopped compiling), and the five public headers `ggml-{vulkan,sycl,hexagon,rpc,metal}.h`.
    There is no `ggml-bench` in this fork, so the plan's Metal-in-bench item had nothing to delete.
    - `ggml/CMakeLists.txt` lost the options, the `GGML_METAL_DEFAULT`/`GGML_BLAS_DEFAULT`
      APPLE split for Metal, the AMX feature options, the Vulkan shader-gen toolchain, the
      Hexagon and SYCL settings, the six Metal/SYCL/RPC/Vulkan/Hexagon entries in
      `GGML_PUBLIC_HEADERS`, and the commented-out Metal resource rule.
      `ggml/src/CMakeLists.txt` lost the five backend build calls, the six
      `GGML_DLL_INSTALL_BACKENDS` lines, and the Vulkan/HIP/`__VEC__` guards around
      `ggml-rmw-vec.cpp` - that source is now built unconditionally, which is what the
      non-Vulkan branch already did. `ggml/src/ggml-cpu/CMakeLists.txt` lost the AMX sources
      from the variant list, the `__AMX_*__`/`GGML_AMX_*` definitions in both the MSVC and the
      GCC/Clang feature blocks, the `sapphirerapids` variant, and the whole KleidiAI block
      including its `FetchContent` of the ARM kernels. The top-level `CMakeLists.txt` lost the
      deprecated `LLAMA_METAL`, `LLAMA_METAL_EMBED_LIBRARY`, `LLAMA_RPC`, `LLAMA_SYCL` and
      `LLAMA_SYCL_F16` option mappings, and `ggml/cmake/ggml-config.cmake.in` lost the Metal,
      Vulkan and SYCL dependency branches. `ggml-backend-reg.cpp` lost the five registration
      blocks and the five `ggml_backend_load_best` names, and `common/arg.cpp` lost
      `add_rpc_devices` and the `--rpc` option - `llama_supports_rpc()` was already false here,
      so that option was never reachable.
    - The conditionals the compiler had already resolved were collapsed in the surviving sources
      by `scripts/port/collapse_conditionals.py`, 44 blocks across six files, and the llama-side
      Metal shared-buffer I/O went with them: `llama_split_layer_params` lost its five
      `metal_*` fields, `llama_context::get_metal_backend()` and `split_sync_event` are gone,
      and `ggml_cpu_has_amx_int8()` is gone from `ggml-cpu.h`.
    - `ggml-cpu.c` keeps `ggml_cpu_has_amx_tile()`, `_bf16()` and `_fp16()` returning 0.
      They are not Metal and not a backend - they are the CPU-feature queries behind
      `llama-server --version`'s `CPU ... AMX` flags, and `preprocess_equiv` shows the file
      unchanged for every other line.
    - 71 environment rows lost their last reader and were dropped from `envvars.json`
      (`scripts/port/drop_rows.py`, reading the name list from `build/dropped-names-41.json`):
      36 `GGML_VK_*`, 15 `GGML_CUDA_SYCL_*`, 12 `GGML_SYCL_*`, 3 each for `GGML_METAL_*`,
      `GGML_HAP_*` and `GGML_VIRTGPU_RPC_*`, plus `GGML_CPU_KLEIDIAI`, `GGML_DISABLE_VULKAN`,
      `GGML_RPC` and the llama-side `LLAMA_ARG_RPC`. 2162 documented variables, down from 2234.
      Dropping a row needs a whole-tree read scan, not the backend scan - `GGML_CUDA_GRAPHS`,
      `GGML_CUDA_DISABLE_GRAPHS` and `GGML_NO_COMPILLER` name a deleted backend but are read in
      `ggml-cuda/`, and `envvars.py:inventory()` does not scan `slstack/`, so a launcher-owned
      name has to be checked there separately.
    Behaviour changes, none of them reachable on this build: a CPU-only llama no longer offers
    `--rpc`, and `llama_split_graph_create`'s refusal line reads "no supported backend (this fork
    builds CUDA)" where it named Metal; both are in the CUDA-gated producer path only under
    `#ifndef GGML_USE_CUDA`. The second of those did not actually happen in this chunk - the line
    still printed "need CUDA or Metal" until chunk 7.1c reworded it.
    Gate: cmake configures with no backend option other than CUDA and CPU in both flag sets -
    `build.sh`'s (CUTLASS, `-DGGML_CUDA_DSV41_SPLIT_F32=ON`, sm_120a) and the plain one - and
    both build 489 and 210 targets with exit 0 and no new warning kind
    (`scripts/port/warning_profile.py` against a baseline worktree: 36 -> 34 occurrences and 19
    distinct both sides over `llama-context.cpp`, `llama.cpp`, `ggml-cpu.cpp`, `ggml.c` and
    `arg.cpp` - the two lost occurrences are the deleted Metal branches' own `-Wcast-qual`).
    `--check-native-profiles` passes from both trees, `llama-server --list-devices` still
    enumerates the 5090, `preprocess_equiv` reports 0 differing positions for
    `ggml-backend-reg.cpp`, `ggml-common.h` and `cpu-feats.cpp`, `code_line_diff` reports only
    the intended deletions for `llama-context.cpp`, `ggml-cpu.cpp` and `ggml-cpu.c`, and 0 changed lines for
    `ggml-backend-reg.cpp` and `llama.cpp` (neither file's code changed - the backends they
    dispatch to were already conditionals the compiler dropped), `metal_word_census.py --strict` is clean over
    the whole llama tree and the engine, the four hand-compiled programs
    (`platform-selftest`, `provider-lease-selftest`, `wf-io-selftest`, `wf-extent-selftest`,
    plus the Engram loader's 15) pass after being relinked against the rebuilt objects, the
    Python suite's failing names are a subset of the baseline worktree's, and
    `scripts/release_check.py --summary` reports 0 findings.
    `scripts/port/reanchor_env_reads.py` is new: the table anchors reads to `path:line`, so a
    source edit above a read breaks `envdocs --check` even when the read itself is untouched.
    It maps each recorded reference onto the live read in the same file by order and refuses to
    guess where a file's read count changed - that is a table decision, not a line fix.
    Findings this chunk produced, for the chunk that owns them:
    - The Mac peer's split-graph API is now eleven permanent stubs declared in
      `include/llama.h` with no caller anywhere in this fork:
      `llama_split_graph_submit_layer_gated` and `llama_split_graph_launch_mega` return -99,
      `llama_split_graph_{signal_input_ready,wait_output_ready}_host` and their `_value` forms
      and `llama_split_graph_synchronize` do nothing, `llama_split_graph_attn_out_buf` and
      `llama_split_graph_hidden_buf` return NULL, and
      `llama_submit_split_layer_async`/`llama_wait_split_layer_async` start nothing and report
      -99 because only the deleted backend's event record made the deferred read possible.
      They returned -99 on Linux before this chunk too, so keeping them is
      behaviour-preserving; deleting that public surface belongs to the chunk that prunes the
      two-machine API, not here. The CUDA half of the same API
      (`llama_split_graph_create`, `launch`, `post_input`, `get_output`, `capture_layer`,
      `wait_token_complete`) is live and unchanged.
    - `SPLIT_DECODE_EVENT_SYNC` is now an inert knob: its branch and the default branch both
      call `ggml_backend_synchronize`, and the read is kept so the row stays documented.
    - 61 `metal` mentions remained in the llama tree across 56 lines, all of them reported as
      comment position by `scripts/port/metal_word_census.py --strict`. That report was not sound:
      the census kept line-comment state open past its newline, so an identifier on a later line
      was filed as a comment and `IDENTIFIER 0` proved nothing. Chunk 7.1c fixed the scanner, and
      the same command now reads IDENTIFIER 1, STRING 3, COMMENT 51 lines, with `--strict` failing
      (chunk 7.1c's message and this plan both recorded STRING 4; re-extracting the tree at
      `050e0b1` and running the fixed census reads 3 - a mention count and a line count were mixed
      up). The identifier is `USE_METAL_GUEST_SUPPORTS_OP` in
      `ggml/src/ggml-virtgpu/ggml-remoting.h`,
      one of nine backend directories this chunk's list never named and never deleted -
      `ggml-blas`, `ggml-cann`, `ggml-hip`, `ggml-musa`, `ggml-opencl`, `ggml-webgpu`, `ggml-zdnn`
      and `ggml-zendnn` beside it, 68 tracked files and about 1.5 MB, none of them compiled on this
      build. Their deletion landed as chunk 4.2; the comment and message sweep is chunk 7.1e's, as
      it was for the engine.
    - `common/arg.cpp`'s `--device` parsing and `llama_supports_rpc()` still name "RPC" at
      runtime; both are inert because no RPC registry exists, and neither is in the producer path.

## Phase 5: Python orchestrator

Chunk 5.1 landed the platform half of this phase in four commits (`270d7a9`, `65e54bd`,
`c17d6a6`, `63c8dc1`): the paging counters read `/proc`, one advisory lock replaces the
four `msvcrt`/`fcntl` pairs, the guardian contains its tree in a cgroup v2 scope, and the
staged runtime carries Linux binaries. Chunk 5.2 (`5fb07f7`) landed the producer runner, so
nothing that launches a producer refuses non-nt any more, and chunk 5.3 landed resident
admission and the checkpoint tier, so what refuses non-nt in this phase is one media
installer:

```
combo_ram.py      both arms now: the Windows named section and `VirtualLock` stay, and the
                  Linux arm is an exclusive 0600 `/dev/shm` section, `MAP_SHARED`, chunked
                  best-effort `mlock`, and `attach()` for the child. Resident admission moved
                  out of it - it was a copy of working_set.py's.
working_set.py    resident admission, one implementation. Windows keeps the hard minimum
                  working set; Linux writes `memory.min` on the scope the job inherited and
                  reports whether the kernel will honour it.
media_tools.py    downloads and unpacks a pinned Windows FFmpeg zip; the non-nt arm only
                  reports the system ffmpeg and ffprobe on PATH. Media setup, Phase 6.
```

macOS-specific code to delete:

```
slstack/backend_setup.py  Metal cmake flags
slstack/backends.py       mlx profiles
slstack/engines.py, combo_*.py, media_*  MLX and Mac paths
slstack/mcp.py tools      Mac media tools, if present
```

`slstack/mac.py`, `pairing.py`, the darwin gates in `backend_setup.py` and `decode.py`,
`memory_counters.py`'s `vm_stat` derivation and the MLX setup scripts are already gone.

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

- `docs/`: done in 7.1a. `MACDECODE*.md`, `docs/macdecode-r*`, `macdecode-measurements.json`,
  `docs/TWO_MACHINE.md`, `docs/YOCO_PAIRED_*` and the Mac build receipt are deleted;
  `docs/MEDIA_SETUP_VERIFICATION.md` and `docs/SUPPORTED_MODELS.md` lost their Mac halves;
  `docs/BACKEND_SETUP.md` and `docs/BACKEND_SOURCE_AUDIT.md` were rewritten for the two Linux
  engines. Measurement history that stays is labelled as inherited from the Windows and macOS
  builds.
- `examples/`: done. `examples/dual-machine` and the paired presets went in 1.10c, which also
  took `mac-big-memory.sh`, `examples/macdecode-r3` and the `dsv41-ultra-cone/proof` receipts;
  `examples/engine_environment/mac-*.json` went in 7.1a, and `examples/WALKTHROUGH.md` and
  `examples/single-machine/README.md` now use the Linux launchers.
- `presets/dsv41-mac.json`: deleted in 1.10c.
- `tests/`: the Mac test sources are gone (7.1a) and the suite names this phase owned pass
  (7.1b). What is still Windows-shaped is two test sources with no build rule,
  `tests/dsv41_producer_abort.cpp` and `tests/studio_generation.cpp`, which need POSIX ports
  rather than deletion, and `vendor/engines/cosmos_windows` behind the media tests (chunk 6.1).
- `docs/VARIABLES.md`: still owed. The scope rename is done, but the inventory cannot see a
  wrapper read (chunk 7.1d), so regenerating it now would only re-render the same understated
  table.

## Phase 8: qualification

Nothing measured transfers. Re-run against the hash-pinned fixtures and write new receipts:

1. 262K exact-cone single engine, three runs, prompt clock and decode rates.
2. 200K real-text fixture, password recall, logits SHA-256.
3. Engram extent reads against the `pread` baseline, so the io_uring gain is isolated.
4. Paired run: dropped. Two-machine mode was deleted with macOS (decision 1, chunk 1.10c), so
   there is no second machine to compare KL against; the cross-machine limits in the inherited
   receipts are history.
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

1.10d Dead Mac and Windows branches in the launcher. Done (`ebc5ba4`): the unreachable
    `cfg.get('mac')` branches in `slstack/engines.py` and `slstack/router.py`, the
    `os.name == 'nt'` branch, Mac-local token counts in `combo_crossover.py`, and the `mac=`
    hold-ledger semantics in `lease.py`. Gate met: the Python suite is at 584 tests,
    6 failures / 5 errors.

1.9 Settled by chunk 1.10c. Decision 1 is (b), single machine only. Decision 2: the supported
    list is DeepSeek V4.x on ds4 CUDA; Kimi, GLM, MiniMax and Step lose their MLX decode, and
    `general` mode now raises. Decision 3: ROCm stays deleted, there is no AMD hardware here.

2.1 Remaining Win32 host code in the engine. Done. Every conditional whose expression named
    Win32 now compiles to what the Linux build already compiled: the Windows branch is gone and
    the branch Linux already took is unconditional. 4,088 lines of engine source deleted; no Linux
    code path changed.

    The gate is
    `grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing" engines/ds4` restricted to source
    extensions (`--include=*.c --include=*.cpp --include=*.h --include=*.hpp --include=*.cu
    --include=*.cuh --include=*.inc`), because the gate as first written also matched build
    outputs. It returns three hits, all vendored and deliberately untouched:
    `engines/ds4/external/nlohmann/json.hpp` (two, the header `ds4_lora.cpp` uses) and the
    upstream `linenoise.c` feature comment.

    What went with the Windows branches: the two-process combo handoff (`ds4_combo_api.inc` keeps
    only its refusal stubs; `ds4_native_checkpoint.h`, the BCrypt hasher, is deleted with the last
    thing that included it, as is `tests/combo_native_parity.c`, the Windows-only fixture that
    included it too), asynchronous expert-cache restore and snapshot (`ds4_expert_restore_async.cuh`,
    now stubs), and background affinity promotion (`ds4_cache_affinity_background.cuh`, stubs).
    Combo and background promotion refuse with a message naming the build; restore and snapshot are
    inert because nothing on Linux calls them. `ds4_web.c` is the opposite case - the whole
    file was a Windows guard whose `#else` branch is a real POSIX implementation, so that 1,456-line
    browser-tool implementation is now unconditional and live.

    Refusal text that named Windows as the requirement now says what is true here. One refusal kept
    its behaviour and lost its stale reason: `--ds-read-parts` is refused at engine open, as it was
    on Linux before this chunk, because the code that applied the option value to the transport
    (`ds4_ds_configure_read_parts`) sat in the deleted Windows branch. The CLI parses the flag into
    `ds4_engine_options.ds_read_parts` (`ds4_cli.c:2215`, `ds4_server.c:17017`) and nothing reads it
    on Linux; `DS4_DS_READ_PARTS` is the live path. Wiring the flag to the transport is a one-line
    follow-up, listed under the known gaps.

    Gate met: `make cuda CUDA_ARCH=sm_120 -j4` and `make cuda-decode CUDA_ARCH=sm_120` build with
    the warning profile the pre-change sources had; the CPU objects compile with the same warnings
    as the same objects built from the pre-change sources; `test-batch-io`, `test-decode-library`,
    `test-cpu-fp4` and `./ds4 --help` pass. Equivalence was checked, not assumed: `gcc -E
    -fdirectives-only -P` per changed `.c` file, before and after, yields the same live line count
    and differs only in four refusal strings; `nvcc -E` on `ds4_cuda.cu` differs only in three
    refusal strings and one `__LINE__` inside a HOST_MEMORY trace line. `envvars.json` and
    `docs/VARIABLES.md` were regenerated: no variable name was added or removed, only source line
    numbers and the snippets of read sites that were inside the deleted branches.

2.2a Apple conditional branches in the engine. Done (`7cc2ca3`). Every conditional mentioning
    `__APPLE__`, `TARGET_OS` or `__MACH__` now compiles to what Linux already compiled: 3,475 lines
    gone, including the Mac graph layer that ds4.c carried whole behind `#if defined(__APPLE__)`.
    Three guards needed hand work because their live branch was an `#elif` (the n-gram read
    advisory, `ds4_gpu_device_is_spark()`, two `_WIN32`/`__linux__` pairs in `ds4_agent.c`), and 77
    conditions kept their guard after losing only the Apple operand, since they still test
    `DS4_NO_GPU` or `DS4_ROCM_BUILD`. The only `__APPLE__` left in `engines/ds4` is one inside the
    vendored `external/nlohmann/json.hpp`.

2.2b Metal naming in live CUDA code. Done. Option (b): the environment names stay, the code
    identifiers went - 3,323 lines, 3,311 of them in `ds4.c`. `metal_graph_*` (the CUDA graph
    layer's per-stage tensor accessors) became `gpu_graph_*`, the decode phase enum became
    `gpu_decode_layer_phase` / `GPU_DECODE_LAYER_*`, `metal_ready` became `gpu_ready`, and three
    names that were wrong rather than legacy were corrected: the GLM env helpers' `metal_name` /
    `metal_env` are the names a non-ROCm build reads, so `cuda_name` / `cuda_env`, and
    `model_open`'s `metal_mapping` is `shared_mapping` (MAP_SHARED vs MAP_PRIVATE, always false
    here). All 215 `DS4_METAL_*` names are untouched, and they survive the gate by construction:
    `_` is a word character, so `\bMETAL_` cannot match inside `DS4_METAL_FOO`. Also deleted: the
    dead `GGML_COMMON_DECL_METAL` / `GGML_COMMON_IMPL_METAL` branches of
    `cuda/mmq/ggml-common.h` (the only remaining `#include <metal_stdlib>`), a comment pointing at
    `misc/METAL_TENSOR_PARALLELISM.md` which is not in this checkout, and two comments still
    describing `ds4_metal.m`.
    Gate met: `grep -rn "\bmetal_\|\bMETAL_" engines/ds4` over the source extensions returns
    nothing, `make cuda` and `make cuda-decode` build with the pre-rename warning profile (26 vs
    26, only the renamed unused function differs), the host gates pass, and `gcc -E
    -fdirectives-only -P` of each changed file before and after - with the same rename applied to
    the "before" text - is byte-identical.

2.2c The Metal backend itself. Done (`dcb5666`). Option (a): the backend is deleted, not refused.
    1,573 lines removed and 106 added across ds4.c, ds4_cli.c, ds4_help.c, ds4_server.c,
    ds4_agent.c, ds4_bench.c, ds4_eval.c, ds4_tp.c and the `ds4_backend` enum. `--metal` and
    `--backend metal` are gone from all five binaries, so they now read `unknown option` instead of
    parsing, being selected and failing at engine open. Two behaviour changes are deliberate:
    `--metal-graph-test`, `--metal-graph-full-test` and `--metal-graph-prompt-test` became
    `--gpu-graph-*` and stopped forcing the backend, which makes the three CUDA graph self-tests
    reachable for the first time (their bodies call `gpu_graph_*_test`, but the flag set
    `DS4_BACKEND_METAL` and the engine refused at open); and `--imatrix-dataset` / `--imatrix-out`
    now refuse while parsing rather than after loading the model. Deleted with the enum value: the
    46 backend tests, 10 `#ifdef DS4_HAS_QWEN4_METAL` blocks (never defined, 1,030 lines),
    `ds4_sessions_eval_batch_metal_supported` and `..._with_prefill_metal_supported` with the two
    call branches they gated, `ds4_engine_glm_graph_budget`, `model_warm_weights_sharded`, the
    `iogpu.wired_limit_mb` residency block, the Pro-Q4 expert-table preload body, the server's
    `qwen4_batch_mtp` field, and the Metal imatrix collection body. Kept inert and marked
    `DS4_MAYBE_UNUSED`: `ds4_engine_collect_sequential_imatrix`, `imatrix_read_text_file`,
    `ds4_sessions_eval_batch_with_prefill_gpu`, `ds4_expert_profile_init`.
    Two scaffolding bugs surfaced here and are fixed in the same commit:
    `collapse_conditionals.py` collapsed an in-scope macro of *unknown* value to LIVE, which
    deletes the live `#else` and keeps the dead branch (its first run on ds4.c turned the Linux
    `mm_min = 8` into the Metal `64`; reverted), and `preprocess_equiv.py` wrote the before file's
    *path* into its temp file unless `--rename-file` was given, so every no-rename equivalence
    check compared garbage. The tool now takes `--never-defined` for macros this build does not
    define.
    Gate: `grep -rn "DS4_BACKEND_METAL" engines/ds4` returns 0, the CUDA warning profile is
    identical to a worktree at the parent commit (26 occurrences), the host gates pass, help prints
    no Metal, and `envvars.json` lost no variable name (every `DS4_METAL_*` read deleted there has
    another reader left). `--rocm` is untouched: it is an alias for CUDA, and the ROCm branches
    belong to 6.1.

2.2d Metal-era identifiers with a leading underscore. Done (`4940201`). Chunk 2.2b's rename used
    word boundaries, and `_` is a word character, so `glm_metal_*`, `g_metal_graph_*`,
    `generate_metal_graph_raw_swa`, `qwen4_metal_argmax` and the exported
    `ds4_engine_metal_graph_*` survived its gate. 92 occurrences renamed to the `gpu_*` forms,
    through the map kept at `docs/evidence/rename-2.2d.json` - rename inputs belong in the repo,
    because `build/` is scratch and `/tmp` is not dependable here.
    Proven inert: `preprocess_equiv.py --rename-file` with the same map is byte-identical for
    ds4.c, ds4.h and ds4_cli.c. One `ds4_metal.m` mention stays in a `ds4_cuda.cu` comment as
    inherited provenance.

2.2e Metal words in comments and in the inert subsystem's messages. Done in four parts:
    `b6c6d9c` (1/4), `f551a8d` (2/4), `a12383b` (3/4), `76053fc` (4/4). Counted by
    `scripts/port/metal_word_census.py`, which classifies every mention by its position in the file
    rather than by a word-boundary regex: 111 comment lines and 173 lines inside quoted strings
    remain outside `DS4_*METAL*` environment names, of which 26 are kept as provenance and the rest
    are printed diagnostics or the expert-locality profiler's JSON text (`ds4.c` holds all of the
    printed ones). The same census reports 195
    mentions of environment names the table documents and 124 of names it does not: their only
    reader is a wrapper such as `glm_graph_env_present("DS4_ROCM_X", "DS4_METAL_X")`, which
    `envdocs`' inventory cannot see, so those knobs are live but undocumented. `DS4_BACKEND_METAL`
    returns 0 and `./ds4 --metal` is an unknown option.

    (1/4, done.) Two identifier classes survived chunk 2.2b and 2.2d for the same reason those runs
    needed prefix entries - `_` is a word character, so `\bMETAL_` cannot see a name whose `METAL_`
    is preceded by `DS4_GLM_`: the nine `DS4_GLM_METAL_*` compile-time constants (20 lines) and the
    `bool metal` parameter of `qwen4_first_token_test()`, which the call site already feeds
    `ds4_backend_uses_graph()`. The constants went through `rename_identifiers.py` with one prefix
    entry (`DS4_GLM_METAL_` -> `DS4_GLM_`), the map committed at `docs/evidence/rename-2.2e.json` for
    the same reason 2.2d's is kept: `build/` is scratch and `/tmp` is not dependable here. The
    parameter became `graph_backend` in its six read/void sites. Proven inert: `preprocess_equiv --rename-file` with the same map reports the
    same live text on both sides (127,287 live lines, 0 length delta, and the 4 differing positions
    are only the parameter), and both compilations of `ds4.c` - plain and `-DDS4_EXTERNAL_DECODE=1` -
    were built in a worktree at the parent commit and here with 23 warnings against 23 and 11
    against 11, identical texts and identical `-W` sets. The env table was regenerated in the same
    commit; 4 rows moved, all only in their quoted consumer evidence, and the 2,234-name set is
    unchanged.

    (2/4, done.) A third identifier class, missed by 2.2b, 2.2d and 2.2e's first run for exactly the
    reason those runs needed prefix entries: seven local `#define`s in the graph path -
    `DS4_METAL_PROFILE_DECODE_STAGE`, `..._OUTPUT_STAGE`, `..._ATTN_STAGE`, `..._Q_STAGE`,
    `..._FFN_STAGE`, `DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT` and `DS4_METAL_TRY_SHARED_DOWN_F16` -
    86 lines, all in `ds4.c`. `\bMETAL_` returns 0 for them because `DS4_` precedes `METAL_`, so the
    sentence "no code identifier names Metal" was false until this commit; the grep was the gate and
    the grep could not see the class. Renamed to the `DS4_GPU_*` forms through `rename-2.2e.json`,
    extended with seven exact entries - exact, not prefix, because a `DS4_METAL_` prefix entry would
    have rewritten the 164 environment names 2.2b kept. None of the seven is an environment name, so
    no knob moved. Proven inert: `preprocess_equiv --rename-file` with the same map reports 127,287
    vs 127,287 live lines, 0 differing positions, 0 length delta, and both compilations of `ds4.c`
    give warning profiles identical to a worktree at the parent commit (23 and 11 occurrences, same
    texts). The env table moved 1 row, `DS4_ROCM_Q_STAGE_PROFILE`, and only in its quoted consumer
    text; the 2,234-name set is unchanged. `scripts/port/metal_word_census.py` is committed with
    this part, because the position of a mention is the only test that separates a kept environment
    name from an identifier; `--strict` exits nonzero when a code identifier names Metal, and
    `docs/linux-port-commands.md` now gates on it instead of on the greps.

    (3/4, done.) The comment half: 111 lines decided, 85 rewritten, 26 left as provenance. Each was
    a decision, not a substitution. A mention that described what the *retired* backend did now says
    so - `"the whole-model Metal graph driver"` in the `ds4.c` header block is the CUDA graph
    driver, `Metal needs a shared mapping for no-copy MTLBuffers` is the mapping the graph backend
    uploads spans from, `Metal currently falls through to the CPU path` is a CPU path with no Metal
    fall-through, `Allocation is lazy and restricted to local Metal` belongs to a backend whose arm
    the collapse already folded to `false` - and a mention that is inherited provenance stays: the
    26 kept lines all read as `the retired Metal backend`, `the deleted Metal backend`, `the Mac
    engine's`, or a retired source file (`metal/moe.metal`, `metal/norm.metal`, `metal/dense.metal`,
    `dsv4_hc.metal`, `metal/qwen4.metal`, `ds4_metal.m`).
    `preprocess_equiv` cannot show a comment edit: it keeps comments, and rewording one shifts every
    line below it. The proof is `scripts/port/comment_code_diff.py` - strip comments from both sides,
    compare code lines. Every file this part touched reports 0 changed code lines; the two headers
    report exactly the 16 lines of the three declarations deleted with it, `ds4_gpu_warm_command_queue`,
    `ds4_gpu_hc_rms_scale_project_f16_tensor` and `ds4_gpu_tp_gate_prefetch_plan`, each a leftover of
    the deleted backend with no definition and no caller anywhere in the repo. The same scan finds 35
    more dead declarations in the engine's GPU headers that name no backend; those are a separate
    chunk, not this one. Both binaries link (`ds4`, `ds4-decode.so`), every TU that warns has a
    warning profile identical to a worktree at the parent commit (`ds4.o` 23, `ds4_ext.o` 11,
    `ds4_cli.o` 1, `ds4_server.o` 2, same texts), and the env table regenerated with the 2,234-name
    set unchanged: 485 rows moved, 483 of them only their quoted `source`/`reads` line numbers, two
    because their consumer snippet quotes a comment this part reworded (`DS4_DSPARK_NO_GPU_MARKOV`,
    `DS4_METAL_NO_PREFILL_KERNEL_WARMUP`). `ds4.c` has no `__LINE__`, so moving its lines changes no
    message; `ds4_cuda.cu` has one, far below the four comments edited there, and those four edits
    are line-count neutral.

    (4/4, done.) The message half: 173 lines, all in `ds4.c`, reworded through
    `scripts/port/string_text_replace.py` with the map committed at `docs/evidence/messages-2.2e.json`.
    That tool applies each pair inside string literals only, longest pair first, and skips any literal
    naming a `DS4_*METAL*` environment name - which is why 195 documented knob reads survive a sweep
    that removes every Metal word from engine output. The scheme: a graph-path failure says `GPU graph`,
    a streaming or SSD path drops the backend word (`ds4: SSD streaming could not build layer %u
    spans`), the inert expert-profile lines drop it too (`ds4: expert locality profiler active`), the
    GLM lines say `GLM graph`, and the three refusals that named a backend the CLI no longer accepts
    name only the ones it does - `--ssd-streaming is currently supported only with --cuda/--rocm`,
    `V4.1 requires single-GPU CUDA per rank`, `Qwen3.8 sessions require a graph backend`.
    **Behaviour change:** every one of those lines is engine stderr or stdout, so a receipt that greps
    for the old wording will not match. Nothing in this repo parses them. The Mac receipts under
    `docs/macdecode-r2/` embed historical `ds4: Metal device ...` stderr verbatim, and
    `examples/engine_parity/RESULTS.json` plus the two `examples/engine_environment/mac-*.json` name
    Metal as Mac measurements; rewording the engine does not invalidate those and they are not a
    regression signal.
    Proven inert apart from the text: `comment_code_diff.py` reports exactly 173 changed code lines,
    every one a string literal, and the warning profiles of every compiled TU are unchanged (`ds4.o`
    23, `ds4_ext.o` 11, `ds4_cli.o` 1, `ds4_server.o` 2, same texts) with both `ds4` and
    `ds4-decode.so` linking. The census reports 0 identifiers and 0 string mentions; `docs/VARIABLES.md`
    no longer carries the `ds4: Metal mixed batch prefill_rows=%u decode_rows=%d` line, and the table
    regenerated with the 2,234-name set unchanged and exactly 2 rows moved (`DS4_METAL_GRAPH_DUMP_LOGITS`,
    `DS4_METAL_SESSION_BATCH_LOG`), each only in its quoted consumer text. `./ds4 --help` prints no
    Metal and `./ds4 --metal` is still an unknown option.

3.1a Producer translation units compile with GCC. Done (`3bc184a`, `0265129`). cmake already
    configured; compiling did not. Two includes still named the engine tree deleted in 1.10a
    (`../../../ds4-win/...` in `src/llama-dsv41-weights.h`, `../../../../ds4-win/...` in
    `tools/dsv41-prefill/studio-generation.h`); 38 declarations in `ggml/include` were
    `extern "C" GGML_API ...` where the macro already expands to `extern`, which MSVC accepts and
    GCC rejects (in `ggml-cuda.h` the redundant prefix is gone, in the twelve `ggml-dsv41-*.h`
    headers the group is wrapped in `extern "C" { }` so the C linkage and the visibility attribute
    both survive); a `std::string == json` comparison that C++20 stops compiling because nlohmann
    3.12 swaps its hidden friends for member operators; `std::min(uint64_t, unsigned long long)`
    on LP64; and one missing `#include <cmath>` in `src/llama-hparams.h` that accounted for eleven
    of the failures. Gate: `-fsyntax-only` over all 295 C++ units of the project, zero errors.

3.1b The producer links. Done (`7973c20`). `cmake --build build/llama` with the flags `build.sh` passes produces `llama-server`,
    `llama-export-lora`, `libslstack-weight-fetcher.so` and `llama-dsv41-prefill`. What the CUDA
    side needed: `<cstring>` in `ggml/src/ggml-cuda/common.cuh` (it calls `std::strcmp`),
    `<cuda/iterator>` in `argsort.cu` and `top-k.cu` under the same CCCL version guards that use
    `cuda::make_strided_iterator` and `cuda::discard_iterator`, and the two `VKQ_C_f32` reads in
    `fattn-mma-f16.cuh` moved inside the `LLAMA_FA_VKQ_F32_ACC` guard that declares the array -
    nvcc resolves a non-dependent name even in a discarded `if constexpr`, and that macro is
    commented off, so 22 template instances plus `fattn.cu` and `flash-attn-sparse.cu` failed.
    `src/llama-weight-fetcher.cpp` compiles the cuFile route whenever `cufile.h` exists and nothing
    linked it: `FindCUDAToolkit` has no `CUDA::cufile` target on CUDA 13.4, so the library is
    resolved by name into `LLAMA_CUFILE_LIBRARY` and linked into `llama`, the producer target and
    the fetcher `.so`. Linking it is not gated on the `nvidia-fs` driver - `cuFileDriverOpen`
    reports the missing driver at runtime and `WF_MODE_AUTO` keeps the io_uring/pread route.
    Gate met: `llama-server --version` prints `found 1 CUDA devices: NVIDIA GeForce RTX 5090,
    compute capability 12.0, VMM: yes` while Strata holds the VRAM.

3.1c The offline producer compiles and runs on Linux. Done (`3fc4f64`).
    `tools/dsv41-prefill/main.cpp`'s 1,163-line body, all of it behind
    `#if defined(_WIN32) && defined(GGML_USE_CUDA)` with a two-line refusal in the `#else`, is now
    one body guarded only by `GGML_USE_CUDA`: the guard was platform code around platform-neutral
    work (sweep orchestration, admission, receipt publication, the session protocol), so it is a
    port rather than a collapse - nothing inside had ever been compiled here. Its Win32 primitives
    moved to `tools/dsv41-prefill/producer-platform.inc`, included after the producer's own policy
    helpers (`require`, `stop`, the bounded constants) and before `live-session.inc` so both see it
    in the same namespace. Two small headers come with it: `producer-wire.h` (the private session
    pipe handle plus write-every-byte-or-fail) and `producer-platform.h` (`argument`, `path_text`,
    `native_path`: the single place paths and argv become UTF-8 text, replacing the now-deprecated
    `std::filesystem::u8path` here).
    Files that had to come along, because `main.cpp` includes them inside that body:
    `live-session.inc` (880 lines: session pipes, NVML/DXGI, the decode library, two diagnostic
    writers), `producer-abort.h`, `combo-base.inc`, `studio-generation.h`, `studio-vision.h`, and
    `src/llama-dsv41-host-memory.h`, whose entire namespace sat inside `#if defined(_WIN32)` with
    no Linux branch at all - `main.cpp` and `llama-dsv41-sweep.cpp:502` were the only readers.
    Substitutions, per the Phase 2 table: `open`/`read`/`write`/`fsync` with `O_EXCL` for the
    exclusive-new admission and `O_NOFOLLOW` where a reparse point was refused; identity from
    `fstat` (`st_dev`, `st_ino`, `st_size`, `st_mtime`, `st_nlink==1`, `S_ISREG`); the no-replace
    publication is `renameat2(RENAME_NOREPLACE)`, which fails `EEXIST` exactly as
    `ReplaceIfExists=FALSE` did; `fstatfs` refuses network and RAM-backed filesystems in place of
    `GetDriveTypeW`+`QueryDosDeviceW`; `fstatvfs` for free bytes; SHA-256 is
    `dsv41_boundary::sha256_native`, whose portable branch the boundary-capture TU already carries
    and the producer already links (no OpenSSL, no vendored hasher); host memory is
    MemAvailable/VmRSS+VmPeak/cgroup v2 `memory.max`+`memory.current`, the same sources
    `engines/ds4/ds4_host_memory.c` uses, with the paging guard sampling swap; NVML through
    `dlopen("libnvidia-ml.so.1")`; `sigaction` for SIGINT/SIGTERM in place of
    `SetConsoleCtrlHandler`, and SIGSEGV/SIGBUS/SIGFPE/SIGILL with `SA_RESETHAND` in place of the
    SEH filter (kind `signal`).
    Behaviour changes, all deliberate: `--carry-storage hybrid` no longer demands a `D:` overflow
    path (no drive letters), keeping native experts plus the explicit 1GiB disk reserve on a fixed
    local volume; Linux has neither share modes nor a fixed/removable indicator, so identity
    rechecks around every file operation stand in for the first and remote/RAM-backed volumes are
    refused by type; a host with no configured swap reports 0 paging bytes, so that guard stands
    down instead of failing the run; the carry-resume RAM floor in `llama-dsv41-sweep.cpp` was
    `#ifdef _WIN32` and is live now, so Linux refuses a resume that would cross a floor it never
    checked; `begin()` sets no process-memory limit (`RLIMIT_AS` would break CUDA's address-space
    reservations), which keeps the locked-memory risk open and now names its call site; and the
    receipt's identity fields keep their names and units, so nothing downstream changes shape -
    `volume_serial_u64` is `st_dev`, `file_id_le_hex` is the 8-byte `st_ino` little-endian with the
    high eight zero, `mtime_filetime_ticks` is `st_mtime` in FILETIME ticks.
    Also here: `tools/dsv41-prefill/mac/ds4_native_session_transport.inc` deleted (item 8, nothing
    referenced it), and refusal/usage text that named Windows, Mac or IoRing says what is true here.
    Gate met: `cmake --build build/llama` builds every target with no new warnings,
    `llama-dsv41-prefill --check-native-profiles` prints
    `PASS: delivered native-half and rounded expert profiles,4096-row resource policies; no
    GPU/model initialized` and exits 0, `--help` prints the usage, and no argument reaches the stub
    refusal. The env table regenerated with no variable name added or dropped (only sites moved),
    release_check is 0 findings, and the Python suite is unchanged at 584/6/5 with the same names.
    Proof beyond compiling, since this tree has no test target for the producer: a scratch program
    against the producer's own `boundary-capture.cpp.o` runs 44 checks over the ported layer
    (`3fc4f64`'s message says 45; the program prints 44 `ok` lines and that is the real count)
    (volume admission, exclusive-new refusal, bounded-write refusal, FILETIME/inode identity, the
    read-only rehash, `read_small`, no-replace publication including the occupied-target refusal and
    the retained pending file, symlink refusal, ancestor pinning, the free-byte budget,
    `exclusive_writer`, environment/pid/executable, `ram_admission`, SIGINT latching, the abort
    file, and `dlopen` of `ds4-decode.so` binding all 8 required and 10 optional `ds4x_*` entry
    points). And the real binary was driven end to end with a throwaway 1 MiB model, a copy receipt
    and four LE32 tokens: it bound its own executable SHA, admitted the wire/carry/cache plans,
    created the payload and pending receipt, admitted host memory from `/proc`, enumerated the
    RTX 5090, and refused on the device ordinal and then on the VRAM reserve - leaving
    `status: failed` in the retained pending receipt, whose payload identity matches `stat`. Strata
    held the GPU throughout.

3.1d The producer's Windows arms deleted. Done (`6fc8cac`). 3.1c kept the Win32 code as the
    reference the port was written against; the fork's scope says delete it, chunks 2.1/2.2a
    deleted rather than split in `engines/ds4`, and the reference is now in git and in `3fc4f64`.
    `collapse_conditionals.py --tokens _WIN32,windows.h,Win32,DStorage,IoRing,DXGI,BCrypt,NOMINMAX`
    collapsed every remaining guard to the branch this build compiles: 368 lines gone across eight
    files, zero insertions, nothing BLOCKED. Proven on the preprocessed text with comments stripped,
    so only code is compared: `main.cpp`, `producer-platform.h`, `producer-wire.h`,
    `producer-abort.h`, `studio-generation.h`, `combo-base.inc` and `llama-dsv41-sweep.cpp` have
    **0 changed code lines**; `live-session.inc` and `producer-platform.inc` lose only
    `shared_video_memory_usage()`, which reported the WDDM non-local segment that Linux has no twin
    for (its `vram_shared_usage` field left the `LLAMADECODE_VRAM_HEADROOM` line, which now prints
    `device_total`/`device_used`); `llama-dsv41-host-memory.h` changes only its paging stop reason,
    which now says "swap growth limit 256 MB" while the receipt's `pagefile_growth_peak_bytes` field
    name stays. Comment text naming `CreateFileW`, `BCrypt`, `GetDriveTypeW`, `DXGI` or
    `LoadLibraryExW` is deliberate provenance, including the three comments that record the Windows
    guarantees Linux does not provide (share modes, fixed-vs-removable media, the Job Object
    process-memory limit). `tools/dsv41-prefill/carry-copy-bench.cpp`, an 89-line Windows-only
    DirectStorage copy bench no build rule compiled, is deleted with them.
    Gate met: the whole producer tree builds with the two pre-existing `-Wmisleading-indentation`
    warnings only, `--check-native-profiles` passes, the 44-check selftest passes, the receipt smoke
    still binds its own SHA and still refuses on the VRAM reserve, the env table loses no name,
    release_check is 0 findings, and the Python suite is unchanged.
    `grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing" engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill`
    returns nothing. The Phase 2 gate does not hold for the rest of the producer tree yet: 25 files
    under `src/`, 6 under `common/`, 4 under `tools/mtmd`, 3 under `tools/server`,
    `tools/decode-receiver`, and the non-CUDA ggml backends still carry Win32 conditionals.

3.1e The weight fetcher's Windows arms deleted, and io_uring wired. Done (`0ffe3ce`, `ff6bb44`,
    plus `1325655`, the hardening follow-up). Phase 3 items 1 and 2, in `src/llama-weight-fetcher.cpp` - the
    largest producer TU still carrying Win32 host code (53 hits, 3,702 lines of them the
    DirectStorage/D3D12 interop: the wave planner, the fence waits, the adapter-LUID match, the
    placed-quarter probe, the ring scheduler). `llama-dsv41-engram-prefetch.cpp` (95 hits),
    `llama-mmap.cpp` (18), `llama-dsv41-weights.cpp` (17), `llama-token-stream.cpp` (15),
    `llama-tb-transport.cpp` (13) and `llama-dsv41-carry.cpp` (13) are still Windows-shaped and
    belong to 3.1f/3.1g.
    (1/2) `collapse_conditionals.py --tokens
    _WIN32,windows.h,Win32,DStorage,IoRing,IOCP,LLAMA_DIRECTSTORAGE,WF_HAS_DSTORAGE
    --never-defined WF_HAS_DSTORAGE,LLAMA_DIRECTSTORAGE` - nothing BLOCKED, file down from
    6,790 lines to 3,149 with that sweep, and to 3,313 after the reader below. A macro in
    `--never-defined` must also be in `--tokens`: a node is only considered when its condition
    names a token in scope, and the first run left 2,146
    dead lines because `WF_HAS_DSTORAGE` was pinned false but not in scope. What survives is
    the refusal Linux already made: `wf_create_required_dstorage()` says "Required
    DirectStorage is not compiled in", the extent-fetch entry points keep `return false`,
    `wf_persist_shutdown()` keeps returning 1. `llama-wf-gpu-ring.inl` deleted with the
    `ds_gpu_ring_slots` parameter and the `--ds-gpu-ring-slots` producer option, so the
    receipt no longer reports that field, and `llama_wf_bounded::exact_two_buffer_profile`
    went with its only caller. Proven on the preprocessed code lines: 47 / 11 / 4 / 7 changed
    in the fetcher, the provider, the sweep and the CLI, every one an intended fold or a dead
    field; the 3,702 deleted lines never appear in the "before" text, which is the point.
    `WF_MODE_AUTO` behaviour change: it called `cuFileDriverOpen()` only after choosing GDS,
    so this box picked GDS on the strength of `cufile.h` and fell back at pipeline start. It
    now probes while resolving the mode and prefers GDS only if the driver reports support.
    (2/2) the reader. One `io_uring` ring per reader thread, `io_queue_depth` reads in flight
    on each, the chunk pointer as `user_data`, registered files where the kernel accepts them
    (`IOSQE_FIXED_FILE`), and completions accounted by one `wf_complete_chunk` shared with the
    `pread` baseline. Chunks the SQ ring cannot take are put back on the queue - an
    unsubmitted chunk never completes, and the pipeline waits on that layer's counter. Both
    readers drain the queue and their in-flight reads at `wf_stop` for the same reason.
    `WF_HAS_IO_URING` is decided by `__has_include(<liburing.h>)`, so the library is now on
    the link line: `find_library(LLAMA_URING_LIBRARY uring)` linked into `llama`, the producer
    and the fetcher `.so`, exactly as `LLAMA_CUFILE_LIBRARY` is.
    Deliberate behaviour changes: `io_direct` no longer opens `O_DIRECT` - Linux requires the
    offset, length and buffer all to be block-aligned, GGUF tensor data is aligned to
    `general.alignment` (32 bytes by default), and the chunked reads would fail `EINVAL` and
    complete with zero bytes; reads are buffered, and the aligned variant is follow-up work.
    `LLAMA_WF_IO_URING=0` forces the pread baseline (new documented row) so the gain can be
    isolated, and `wf_get_io_engine()` reports which reader a fetcher started.
    Gate met: the producer targets build with the pre-change warning profile (27 occurrences,
    18 distinct texts, none added or removed, compiled in a worktree at the parent commit with
    the same commands); `docs/evidence/wf-io-selftest.cpp` drives the real `wf_*` pipeline over
    a 64 MiB throwaway file and byte-compares what each reader stages against the file, then
    reports both. 139 checks pass, and over three cold rounds the whole pass takes, in one run,

        4 threads (ring QD8 vs one pread per thread): io_uring 19.6 ms, pread 19.7 ms -> x1.00
        1 thread  (ring QD8 vs one pread per thread): io_uring 21.4 ms, pread 33.6 ms -> x1.57

    Four threads is the pipeline's shape and there the readers are a wash (x1.00-x1.01) - four
    `pread` threads already keep the NVMe busy. What the rings add is queue depth per thread, which
    is the one-thread row: x1.57 to x1.82 across runs. Do not quote `wf_get_stats`'s
    `total_nvme_read_ms` for this: it sums each layer's own submit-to-last-completion window, and
    layers overlap, so it double-counts and reads x1.37 where the wall clock reads x1.00. It is a
    per-layer latency metric, not a bandwidth number.
    `--check-native-profiles` still PASSes,
    the 44-check platform selftest passes, the receipt smoke still refuses at the VRAM
    reserve, and `grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing\|IOCP"` over the fetcher
    and the producer CLI returns nothing once comment lines are dropped
    (`| grep -vE ":[0-9]+:\s*//"`). The two hits that survive are `tools/dsv41-prefill/
    producer-platform.inc` lines 5 and 22, provenance comments naming the Win32 layer that file was
    ported from - chunk 3.1d's "returns nothing" was written without that filter and was wrong as
    printed.
    `envvars.json` is 2,234 rows: the same
    names plus `LLAMA_WF_IO_URING`, `release_check` 0 findings. Nothing here is a bandwidth
    result for a real model; the fetcher has still never read weights.

    The hardening follow-up came from reviewing the chunk. A read that delivered fewer bytes than it
    asked for counted as completed - the same hazard the O_DIRECT refusal exists to avoid, still live
    for a short read or a truncated file - and `wf_register_layer` checked span bounds only on the
    strict DirectStorage path. Now a negative completion, and a short read of bytes the file does
    hold, set the sticky transfer failure; `wf_register_layer` refuses a span running past its file's
    size in every mode; `wf_wait_layer_gpu` returns nullptr once the fetcher has failed, so a refused
    registration cannot hand compute a buffer holding whatever staging had before (it did: a probe
    with an 8 KiB span at `size-1024` returned the buffer pointer); and the ring worker keeps the
    chunks it submitted in an outstanding list, so a failed `io_uring_submit` or `io_uring_wait_cqe`
    drains what the kernel already completed and accounts the rest as failed reads instead of leaving
    a layer's counter below its total - the pipeline polls with timeouts, so that was a `progress=NO`
    spin rather than a hang. A probe with `sigaction` flags 0 (what the producer installs) showed
    `io_uring_wait_cqe` does not return `-EINTR` on this kernel: it blocked through a 20 s timeout,
    so that path is defensive rather than a bug fixed.

3.1f The weight provider's lease, pipeline and named extents. Done (`3afcd91`, `1e4cf01`,
    `38765aa`). Phase 3 items 4 and the extent half of 5. The decision the chunk was asked to
    make - do `wf_fetch_extents_direct` and `wf_ready_direct_bank` become io_uring reads or go away -
    came out as reads: an extent batch is split into parts, read through io_uring (or the `pread`
    baseline) into a pinned window of four 8 MiB slots, and uploaded into the named bank. With a
    caller stream the uploads go on that stream, so they are ordered after the compute queued on it;
    the tail-hint worker uploads on the fetcher's copy stream and publishes one `cudaEvent` per
    batch, waited on by fence. `wf_ready_direct_bank` had nothing to report because the generation
    was only ever published by the deleted queue, so the relay publishes one where a stage reaches
    `GPU_READY` (both sites) and never reuses it. Extents are bounds-checked against their file and
    their bank before any byte is written, and a negative or short completion poisons the fetcher.
    The reader is CUDA-optional: without a CUDA build the banks are host buffers and the upload is a
    `memcpy`, which is how `docs/evidence/wf-extent-selftest.cpp` drives it with no VRAM free.

    The lease is `open(O_RDONLY|O_NOFOLLOW)` on the file the receipt names, opened relative to the
    pinned parent (`openat`), with the ancestor descriptors rebound to their names by `identity()`,
    which runs around every file operation. `identity()` compares `st_dev`, the 8-byte little-endian
    `st_ino` in the 16-byte receipt field, `st_size`, `st_mtime` in Unix ns, `S_ISREG` and
    `st_nlink==1` - exactly the four keys `slstack/native_broker.py` writes. `DRIVE_FIXED` and
    `QueryDosDeviceW` became a `statfs` refusal of network and RAM-backed types; a reparse point is
    refused at open; the sector-alignment probe went with `FILE_FLAG_NO_BUFFERING`, and the Engram
    row extent keeps its 4 KiB alignment so a row sees identical bytes.

    Behaviour changes, all deliberate: the strict create (`wf_create_required_dstorage`, always a
    refusal here) is `wf_create_required_streaming` and means the exact typed relay - bounded
    staging, async copy ordering, an explicit pinned tier, no ambient tuning, no expert cache or
    relay prefetch buffer - so `required_dstorage` became `strict_streaming` with its eleven
    behaviour branches unchanged; the provider passes `WF_MODE_DIRECT_IO` explicitly (not `AUTO`,
    which could resolve to GDS and register the fetcher's allocations before its bank pointers are
    replaced) and `io_direct=false` (buffered reads never round a chunk up, so the fetcher's banks
    are exactly the admitted `compute_buffer_bytes`); admission moved - the pinned host staging tier
    (`(buffers+1)` banks) and the 32 MiB extent window are charged to the host plan and the GPU-side
    staging term left the device plan, so a launch recipe needs a larger
    `--provider-host-bytes` than the Windows one; `--fetch-profile baseline` is refused while
    inspecting instead of at pipeline creation; `acquire()` gained a guard that a stage landed in
    the bank its regions were planned in, because the extents name regions of that bank; and
    `--ds-submit-threads` and `--ds-zero-padding-only` are unknown producer options now, with their
    `wf_params` fields, config fields, receipt keys and the stale `tools/slstack-weight-fetcher.def`
    export list. `wf_extent_fetch_stats`'s submit/fence/verify fields stay 0 and say so, so the
    `LLAMADECODE_DS4X_TOKEN` line keeps its shape.

    The third commit is the review's one real catch plus the lease's missing ancestor pin: both
    io_uring readers now wait through `io_uring_submit_and_wait()` (see the flush probe below),
    ancestors are held as descriptors and rebound to their names by `identity()`, the model and row
    descriptors open with `openat()` on the pinned parent, `inspect()` takes the lease before it
    asks about the device (which is what makes `docs/evidence/provider-lease-selftest.cpp` - 14
    checks, no backend - possible), and `params.prefetch_depth=3` is named rather than inherited
    because a stage landing in bank `id%2` depends on the relay being ahead in bank assignment.
    `build/bank-parity.cpp` (scratch) checks that order and reports no mismatch at the named depth
    and at depth 1; the decode acquire order is not covered there and is the first thing to probe
    when a GPU has headroom.

    What the review caught, and what it did not: `io_uring_wait_cqe()` does not flush
    the submission queue on this kernel and liburing 2.15, and `io_uring_submit()` can return 0
    without being an error, so both io_uring readers now wait through `io_uring_submit_and_wait()`
    - `docs/evidence/uring-flush-probe.cpp` demonstrates the two halves of that claim. The review's
    other four findings were checked against the tree and were not real: `direct_generation` is
    published in the GDS arm too, so a tail hint cannot see generation 0; a layout switch rebuilds
    the fetcher, so a YOCO skip mask cannot carry over into the decode layout; the provider's host
    plan, which now carries the staging tier and the extent window, is what
    `llama-dsv41-sweep.cpp`'s combined host budget sums; and a stale or zero ready-batch generation
    refuses without poisoning.

3.1g The Engram row loader, and the provider's three Engram arms. Done (`2f79178`, plus this
    record commit). Phase 3 items 3 and 4.
    Every implementation line of `src/llama-dsv41-engram-prefetch.cpp` was inside `#ifdef _WIN32`:
    IoRing queues at QD64 over 1 MiB extents, file handles and slot arenas registered as ring
    operations, a slot table, table-0-before-table-1 publication, and `CancelIoEx` cleanup.
    The row half of that file is platform-neutral and stays byte-identical - a scratch comparison
    against the parent commit (`build/engram-neutral-diff.py`) checks eleven members and reports no
    change: `prepare()`, `budget()`, `planned_requests()`, `env_u32()`, `prepare_early()`,
    `plan_geometry()`, `choose_scan()`, `scan_geometry()`, `predicted_seconds()`, `read()`,
    `wait_table()`. The IO half is rewritten:

    - One read queue per ring: io_uring when `io_uring_queue_init` succeeds, the serial `pread`
      baseline when it does not or with `LLAMADECODE_ENGRAM_IO_URING=0` (new documented row, the
      same convention as `LLAMA_WF_IO_URING`). Both readers run the identical planner extents, slot
      arena, hit bitmaps, publication order and counters, which is what makes the comparison mean
      anything. Fixed-file registration is optional, as in the fetcher's reader; the slot arena is
      named per request instead of registered, so the ring-op registration and its completion
      accounting loop are gone (`io_uring_register_files` is a synchronous syscall).
    - Completions are drained with `io_uring_peek_cqe` and waited on through one shared `eventfd`
      plus `poll(..., 100 ms)`, which replaces `SetIoRingCompletionEvent` and
      `WaitForMultipleObjects`. The submission queue is flushed before every wait, so no wait
      relies on a wait also being a submit (3.1f's rule). `io_uring_submit()` returning 0 is a
      legitimate submit of nothing new, and entries it did not flush stay in the ring, so the
      Windows "the whole batch or nothing" refusal is gone - `in_sq` counts the unflushed entries
      and the batching thresholds read it. Completion ownership is unchanged: the same
      `sequence<<9|slot` and `sequence<<16|slot` nonces.
    - The first version consumed the eventfd counter before polling, which swallowed the wake of a
      completion whose CQE was already queued but not yet peeked; every early load then paid one
      full 100 ms poll. The wait checks each queue's own CQ ring (`io_uring_cq_ready`) first, and
      the 200-extent early load went from ~115 ms to 14-20 ms.
    - Cleanup: `CancelIoEx(handle, NULL)` became `IORING_OP_ASYNC_CANCEL` keyed on the descriptor
      (`IORING_ASYNC_CANCEL_ALL`, plus `_FD_FIXED` when the file is registered), then the same 5 s
      drain and the same refusal to free an arena an unconfirmed read can still write into. That
      refusal now prints `DSV41_ENGRAM_RING_CLEANUP_REFUSED` and exits with `_exit(79)` instead of
      `TerminateProcess(..., 79)`. Unsubmitted SQ entries are flushed and cancelled rather than
      abandoned, so nothing can write after the arena is freed.
    - The loader opens its own descriptor on the model's name and refuses it unless `st_dev`,
      `st_ino`, `st_size`, `S_ISREG` and `st_nlink == 1` still match the provider's lease - the
      fact `FILE_ID_INFO` proved - with `O_NOFOLLOW` where Windows used
      `FILE_FLAG_OPEN_REPARSE_POINT`. `FILE_FLAG_NO_BUFFERING` has no twin: the reads are buffered,
      the 4 KiB-aligned extent geometry is kept, and the `FileStorageInfo` sector probe that
      `LLAMADECODE_ENGRAM_READ_ALIGN=512` required is gone; the `PERFEXP_E13` line now reports the
      geometry alone.
    - Deleted rather than faked, as `--ds-submit-threads` was in 3.1f: `LLAMADECODE_ENGRAM_IO_PRIORITY`
      set a per-handle Windows IO priority hint, which Linux has no per-request equivalent for
      (process scheduling and `ionice` govern), and its measured `low` value starved the reads.
      Both reads, both log lines and the `envvars.json` row are gone; the row set otherwise is
      unchanged, so `envvars.json` is still 2,234 names, with `LLAMADECODE_ENGRAM_IO_URING`
      replacing it.
    - `load()` and `load_early()` take `const std::filesystem::path&` and an `int` descriptor
      instead of a `const wchar_t*` and a `HANDLE`; the change is in this file, not adapted at the
      call site. Two log fields naming the old transport changed: `own_iorings=1` and
      `own_ioring=1` became `own_rings=1` and `own_ring=1`, and both load lines gained
      `reader=io_uring|pread`.
    - The provider's three Engram arms are open, which removes the last three `#ifdef _WIN32` in
      `src/llama-dsv41-weights.cpp`: `launch_early` captures the path and the descriptor and its
      thread loads on its own queues, and `prepare_engram_prompt` runs its synchronous load on the
      early loader as it did on Windows.

    Gate met, host-only. `tests/dsv41_engram_early_io.cpp` is ported with the loader - it was the
    Windows-only synthetic-table test for exactly this file, and 208 MB of generated table file is
    not a model. 15 checks pass on the ring reader and on the `pread` baseline, with the scan
    geometry chosen, forced and off, at 1/3/4/8 rings and depth 32..4096 with batch 1..4096, the
    512 B geometry, the single-queue reference loader, the tight-admission arena that shrinks the
    IO arena inside admission, prior-request reuse, the row cache, the descriptor refusal, the
    batched-submit variant and cancellation; two checks were added for the descriptor refusal and
    the row cache, which the Windows test did not cover. `docs/evidence/uring-flush-probe.cpp`,
    `wf-extent-selftest.cpp` (33 checks), `provider-lease-selftest.cpp` (14 checks) and
    `platform-selftest.cpp` (44 checks) still pass. The warning profile for the two touched TUs
    loses five warnings and adds none (the removed ones are the Windows stubs' own
    `-Wunused-parameter` and `-Wsuggest-attribute=noreturn`). `release_check` is 0 findings, the
    env table loses no other name, and `--check-native-profiles` still PASSes.
    The readers agree on this fixture because it is page-cache resident: the single-queue load of
    131,781 row reads takes 51-54 ms on the rings and 50-53 ms with `pread`, and the early loader's
    200 extents 14-20 ms either way. That compares the mechanics, not the device; the 76.2 s to
    43.2 s figure this change is meant to reproduce is a RAID-0 measurement, and Phase 8 item 3
    owns the cold-cache number here against a qualified GGUF.
    Not run: the no-model producer receipt smoke, which reaches `ggml_backend_cuda_get_device_memory`.
    The box cannot create a CUDA context while Strata holds the VRAM - a probe calling only
    `cudaFree(0)` and `cudaMemGetInfo` returns `out of memory` - so that gate is the box, not this
    change. `llama-server --version`, which enumerates without a context, still prints the device.

3.1h The carry lease and the locked RAM tier. Done. Phase 3 item 5, and chunk B.4 of
    `docs/blackwell-enablement-plan.md`, which specified it.
    `src/llama-dsv41-carry.cpp` and `src/llama-dsv41-locked-ram.h` were the last two Windows-shaped
    files the streamed prefill needs. The mechanical half is `collapse_conditionals.py --tokens
    _WIN32,windows.h,Win32,VirtualLock,VirtualAlloc,VirtualFree,BY_HANDLE_FILE_INFORMATION
    --never-defined _WIN32`: 174 deletions, 0 insertions, 0 blocked, and both files then hold no
    `_WIN32`. Proven as a sweep - `preprocess_equiv.py` reports 0 differing live lines for each
    file before and after the collapse, `code_line_diff.py` reports 0 changed code lines in
    `llama-dsv41-carry.cpp` against its parent (a pure deletion), and the warning profile over
    `llama-dsv41-carry.cpp`, `llama-dsv41-weights.cpp` and `llama-dsv41-sweep.cpp` is 14
    occurrences against 14, 6 distinct, none added or removed.

    The tier itself then changed, because the Linux arm it collapsed to was all-or-nothing and the
    allowance on Linux is not a formality: `RLIMIT_MEMLOCK` is 8 MiB soft *and* hard for an
    unprivileged process here, `mlock` refuses the whole call past it, and `setrlimit` cannot raise
    the hard limit without privilege. The decision (`docs/blackwell-enablement-plan.md`) is Basalt's
    rule - lock what the allowance permits, keep the tail allocated and merely pageable, report the
    shortfall, and let the host-memory paging guard enforce residency:

    - `lock_pages` locks in 64 MiB chunks and halves the chunk at each refusal, retrying the same
      offset, so a tier far above the allowance still locks what it can. Chunks stay whole pages:
      `mlock` accounts rounded-up pages, and a partial-page chunk records fewer bytes than the
      kernel holds (measured: a 12 MiB tier against the 8 MiB allowance reported 2047 pages while
      `/proc/self/smaps` said 2048).
    - `locked_bytes` sits beside `bytes` in `lock_state`, per mapping in a `locked` map, so
      `state().bytes` stays the admission figure the carry tests read. `CARRY_RAM_LOCKED` gained
      `locked_bytes=`, and a tier that cannot be fully locked also prints
      `CARRY_RAM_LOCK_PARTIAL bytes= locked_bytes= pageable_tail_bytes= rlimit_memlock_bytes= errno=`.
      The historical receipt lines in `examples/engine_parity/RESULTS.json` keep the old wording;
      nothing parses them.
    - `suspend_pages` unlocks the bytes it holds locked and then `MADV_DONTNEED`s the range, which
      is what `VirtualUnlock` plus `MEM_DECOMMIT` did; `resume_pages` re-locks best effort and its
      log line gained `locked_bytes=`, the suspend line `unlocked_bytes=`.
    - The Windows arm also refused a resume larger than available physical memory before committing
      pages. That refusal is kept rather than dropped: `resume_pages` reads MemAvailable through
      `llama_dsv41_host_memory::proc_field`, which is why the header now includes
      `llama-dsv41-host-memory.h`. Pages refault on first touch, so nothing else refuses an
      oversized resume.
    - Fixed a real leak the collapse exposed: `allocate` built the `shared_ptr` after accounting the
      mapping, so if the control block (or the accounting node) threw, no deleter existed to release
      the mapping and `state().bytes` kept the charge. The accounting and the `shared_ptr` are now
      inside one `try`, and the `catch` releases the mapping, the per-mapping entry and both
      counters and rethrows.

    Gate, host-only, both previously unusable on this box: `tests/dsv41_carry_release.cpp` was red
    before this chunk - `Carry RAM cannot be locked` past ~6 MiB against the 8 MiB allowance, on its
    third 2.95 MB allocation - and now prints its PASS line with the third allocation reporting
    `locked_bytes=2478080 pageable=1`. `tests/pinned_carry_memory.cpp` printed
    `SKIP: Windows working-set lock proof`; it is ported to `/proc/self/smaps` (per-VMA `Locked:`
    kB, the VMA has no pathname so it is anonymous) plus `mincore` residency, and keeps the
    portable assertions the Windows version made - zeroed and writable, private, release unmaps, the
    `allocate(bytes, bytes-1)` refusal, an accounting-allocation failure leaving
    `state().bytes == 0` *and* no mapping of that size behind, and the accounting after two
    allocations and one release. It adds the partial case the Windows tier could not have: a tier
    beyond the allowance is allocated, `mlock` splits its VMA at the lock boundary, the locked VMAs'
    `Locked:` sum equals `state().locked_bytes`, and the tail page is neither locked nor faulted in.
    It passes at 1 MiB, 2 MiB, 4 MiB, 8 MiB (the allowance exactly), 12 MiB, 64 MiB and 100 MiB;
    a 100 MiB tier locks exactly 8 MiB here. Neither test has a build rule -
    `docs/linux-port-commands.md` has both compile lines.
    Everything else re-run after the tier change: `platform-selftest.cpp` (44 checks),
    `wf-extent-selftest.cpp` (33), `provider-lease-selftest.cpp` (14), `uring-flush-probe.cpp` (3),
    `tests/dsv41_engram_early_io.cpp` (15, ring and `pread`), and `--check-native-profiles`.
    The box had just enough VRAM headroom that the two CUDA gates that were unrun also ran:
    `wf-io-selftest.cpp` (139 checks, one-thread wall x1.69 against the `pread` baseline) and the
    no-model producer receipt smoke, which reached its documented refusal - `Selected GPU free bytes
    do not exceed the explicit reserve`, `out/payload.bin` still 0 bytes, pending receipt retained -
    instead of aborting in `ggml_backend_cuda_get_device_memory`. That margin was 116 MiB at the
    time and gone again an hour later; `docs/evidence/cuda-headroom-probe.cpp` is the two-call probe
    that says whether a CUDA gate can run at all, and it belongs in the same session as the gate.
    The Python suite is unchanged by the chunk: 584 tests, the plan's own list of failing names,
    `release_check` 0 findings, and the env table regenerated with all 2,234 names and no row moved
    (the tier reads no environment variable).

3.1 Streaming producer on Linux. The producer's file, fetcher, provider, Engram loader and carry
    tier are Linux. What is left of Phase 3 is the Win32 host code in `mtmd`, `server` and
    `decode-receiver`, none of which the streamed prefill needs, and the ggml-side refs Phase 4
    deletes.
    Gate: a 262K prefill, which needs a qualified GGUF on this box (see the machine facts).
    Nothing in the prefill path refuses on Linux now: the provider holds the lease, the fetcher and
    the Engram loader read on io_uring, and the carry tier locks what the allowance permits, so the
    first real `--engram-prefetch-bytes` run is the test of the geometry defaults in
    `choose_scan()`, which were tuned on a Windows RAID-0.
    Findings for whoever continues: `tools/slstack-weight-fetcher.cpp` is a two-line anchor
    file, so the `.so` it builds exports one data symbol and carries no fetcher API - chunk
    5.1's `single_runtime.py` PRODUCER_FILES cannot stage it as the DLL's twin; and
    `vendor/engines/cosmos_windows/weight_streaming/ds_backend.py` mirrors `wf_params` by ctypes
    field order, which the 3.1e and 3.1f deletions shifted - that module is Phase 6's to delete.
    The CUDA arms of the extent reader (`docs/linux-port-commands.md` has the link line) have still
    never executed: a `ggml` CUDA backend wants about 700 MB past what Strata and the browser leave.

3.2 Producer decode library wiring. Done as part of 3.1c, because `live-session.inc` is part of
    `main.cpp`'s body: `dlopen`/`dlsym` of `ds4-decode.so` beside the executable in place of
    `LoadLibraryExW`/`GetProcAddress`, with `make cuda-decode` producing it. Verified without a
    model by the selftest's `open_decode_library()` check: all 8 required and all 10 optional
    `ds4x_*` exports bind. A live session against weights is still unrun.

4.1 ggml backend prune: done (`0c9d01d`), see the Phase 4 entry above for what it deleted and
    what it proved.

4.2 The nine ggml backends Phase 4's list never named. Done in two commits (`9125df4`, `eaccc5e`).
    4.2a deleted `ggml/src/ggml-blas`, `ggml-cann`, `ggml-hip`, `ggml-musa`, `ggml-opencl`,
    `ggml-webgpu`, `ggml-zdnn`, `ggml-zendnn` and `ggml-virtgpu` - 68 tracked files - with the
    seven public headers they exported (hip and musa exported none), 32,327 lines over 75 files,
    the nine `ggml_add_backend()` calls, the options and sub-options behind them
    (`GGML_BLAS`/`GGML_BLAS_VENDOR` and the APPLE default split, `GGML_MUSA` and its two,
    `GGML_HIP` and its five, `GGML_WEBGPU` and its four, `GGML_ZDNN`, `GGML_VIRTGPU` and
    `GGML_VIRTGPU_BACKEND`, `GGML_OPENCL` and its four, `GGML_ZENDNN`, `ZENDNN_ROOT`), the
    deprecated `LLAMA_CANN` mapping - which was the only thing that could ever have set
    `GGML_CANN`, because no option defined it - the BLAS, OpenCL and HIP dependency branches of
    `ggml-config.cmake.in`, seven include guards and seven registration blocks and seven
    `ggml_backend_load_best` names in `ggml-backend-reg.cpp`, and five `GGML_PUBLIC_HEADERS`
    entries. `GGML_ACCELERATE` stays: its only reader is the APPLE branch of
    `ggml-cpu/CMakeLists.txt`, so it is an Apple cmake setting rather than a backend. 12 env rows
    lost their last reader (8 `GGML_CANN_*`, 3 `GGML_OPENCL_*`, `GGML_REMOTING_USE_APIR_CAPSET`),
    2,162 documented variables down to 2,150; `GGML_OP_OFFLOAD_MIN_BATCH` stays because it is also
    read in `ggml-cuda/` and `llama-context.cpp`, so only its stale `ggml-cann` reference went.
    `code_line_diff` reports 7 changed code lines in `ggml-backend-reg.cpp`, exactly the seven
    `load_best` calls - everything else deleted there was inside a guard this build never defined,
    which is why `preprocess_equiv` cannot see it at all.
    4.2b collapsed what those trees made dead. `ggml-hip` and `ggml-musa` had no kernels: each
    globbed `../ggml-cuda/*.cu` and compiled it with hipcc or the MUSA clang, defining
    `GGML_USE_HIP`/`GGML_USE_MUSA` and their sub-options, so the CUDA backend's AMD and Moore
    Threads branches are now text no compiler of this fork ever resolves. 22 files, 1,419
    deletions and 58 insertions over `ggml/src/ggml-cuda`, `ggml-common.h` and `ggml-cuda.h`, to a
    fixed point (a second pass is required: promoting a `#elif` can leave a guard that is then
    always true). `vendors/hip.h` and `vendors/musa.h` (425 lines) go - the branches that included
    them were their only references - and `MUSART_HMASK` with them, its last reader being the
    collapsed `__hgt2_mask` guard. Eight `BLOCKED` nodes were resolved by hand, and 26 `#endif //
    <condition>` comments were rewritten to the surviving condition: a collapse that shortens a
    guard leaves its trailing comment describing a condition that no longer exists.
    - Four soundness bugs in `scripts/port/collapse_conditionals.py` were fixed on the way, and
      `docs/evidence/collapse-probe.h` is now the gate for the tool: apply it to a copy and diff
      the preprocessed text of the copy against the preprocessed text of the probe - identical,
      with every decision path covered including the ones that must stay untouched. An
      unparsable condition was written back re-spaced instead of reported (the tokenizer had no
      comparison operators, so every `CUDART_VERSION >= X` guard counted as unparsable and the first
      pass on the CUDA tree wrote out 28 rewrite-guards, several of them the same condition
      re-spaced); a directive spanning continuation lines was treated
      as one line, so a rewrite flattened `#if A || \` into a single line with the backslash inside
      the expression and a drop left the continuation behind; the dead-`#if`-with-live-`#elif`
      path dropped the `#endif` that closes the promoted guard, which is how `vecdotq.cuh` ended up
      with four `#if` and three `#endif`; and the operand-drop patterns for a macro the build does
      define were inverted, so `false && X` lost its false operand and `X || true` its true one -
      each of those changes a condition rather than shortening it. Nesting inside the surviving
      branch of a dead outer node was never visited.
    - `scripts/port/cuda_live_equiv.py` is new: `preprocess_equiv.py` preprocesses both sides with
      one include set, which proves nothing when the sweep touches a header, because the before
      file then includes the after tree's `common.cuh`. Each side gets its own tree's directories;
      over the 20 changed files it reports 0 differing live lines for `ggml-cuda.h` and
      `ggml-common.h` and exactly one for each of the 18 that include `common.cuh` - the deleted
      `#define MUSART_HMASK 40300`, which is the intended change and the only one. Conditionals are
      resolved and macros left alone, so `__LINE__` inside `CUDA_CHECK` does not move the diff. A
      file present on one side only is reported, not failed: a deletion is not a sweep.
    Gate: the CUDA headroom probe said yes first (298 MiB free), both flag sets build with exit 0,
    `--check-native-profiles` passes from both, `llama-server --list-devices` still enumerates the
    5090, `warning_profile` over eight CUDA TUs reads 13 occurrences and 6 distinct on both sides,
    the eight hand-compiled programs pass relinked against the rebuilt objects,
    `metal_word_census --strict` passes over the llama tree and the engine (IDENTIFIER 1 -> 0:
    `USE_METAL_GUEST_SUPPORTS_OP` went with `ggml-virtgpu`, which is what reopened this phase),
    `envdocs --check` reports 2,150 documented variables with 29 rows re-anchored and no name
    added or dropped, the suite's failing names are a subset of the baseline worktree's, and
    `release_check` reports 0 findings.
    Findings this chunk produced, for the chunk that owns them:
    - `engines/ds4/cuda/mmq/` is a vendored copy of the same ggml CUDA tier
      (`cuda/mmq/VENDOR.md` says so), and it carries the identical HIP and MUSA branches: 96
      directive lines naming `GGML_USE_HIP`/`GGML_USE_MUSA`/`MUSART_*` across 11 files -
      `common.cuh`, `ds4_ggml_stubs.cu`, `ds4_mmq_d2r.cu`, `ds4_mmq_d2r.cuh`, `ds4_mmq.h`,
      `ds4_repack.cu`, `mma.cuh`, `mmq.cuh`, `vecdotq.cuh` and the two `test/` sources - plus its
      own `vendors/hip.h` (308 lines). The engine's Makefile never defines any of those names, so
      it is the same fold as this chunk's, with the engine's own gates (the four hand-compiled
      TUs and the warning comparison in `docs/linux-port-commands.md`) rather than the llama
      build's. `common.cuh` there even includes a `vendors/musa.h` that does not exist in the
      engine, which is only visible because the branch naming it is dead. Not done here: this
      chunk's scope was the llama tree, and the engine's mmq tier is what Phase 2 owns.
    - The CUDA backend's AMD and Moore Threads compute-capability vocabulary is still
      dead-but-present: `GGML_CUDA_CC_IS_CDNA/_RDNA/_MTHREADS`, `AMD_MFMA_AVAILABLE`,
      `AMD_WMMA_AVAILABLE` and the `GGML_CUDA_CC_CDNA*/RDNA*/QY*/PH1` constants. Their guards'
      first term is `defined(RDNA3)` or `__CUDA_ARCH__`, and the collapse cannot pin
      `__CUDA_ARCH__` - it is genuinely undefined for the host pass and defined for the device
      pass - so no branch keyed on it is decidable by that rule. Pruning it is a chunk of its own,
      not a collapse. `mma.cuh`'s `#endif // defined(GGML_USE_HIP)` comment never matched its
      guard, upstream.
    - Ten retired-backend mentions remain in the llama tree, all in comment position
      (`src/llama-context.cpp` x2, `src/llama-prefill-stream.cpp`, `common/common.h` x2,
      `ggml/src/ggml-cpu/simd-mappings.h`, `ggml/src/ggml-backend.cpp`,
      `ggml/include/ggml-backend.h`, `ggml/src/ggml-cuda/mma.cuh`, `common.cuh`'s Moore Threads
      heading): chunk 7.1e's text sweep.
    - The "plain flag set" in `docs/linux-port-commands.md` cannot configure as written: this fork
      has no `tests/`, no `examples/CMakeLists.txt` and no `pocs/`, so it needs
      `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF` and
      `-DLLAMA_DSV41_JSON_INCLUDE_DIR` (it failed the same way at the parent commit - the command
      predated chunk 7.1a). Corrected in the commands file.
    - Apple cmake logic survives in the llama tree: `GGML_ACCELERATE` and the APPLE branch of
      `ggml-cpu/CMakeLists.txt`, the `apple_m1`/`apple_m2_m3`/`apple_m4` CPU variants in
      `ggml_add_cpu_backend_variant`, the `HAVE_BUGGY_APPLE_LINKER` probe, `_DARWIN_C_SOURCE` for
      Darwin/iOS/tvOS, and the visionOS definition in `ggml/src/CMakeLists.txt`. None of it is a
      backend, none of it is built here, and no chunk has claimed it.

4.3a The engine's vendored mmq tier folds the same way. Done, one commit. `engines/ds4/cuda/mmq/`
    is a copy of the same ggml CUDA tier (chunk 4.2's finding said so, and it was right about the
    copy and wrong about the file it cited: `cuda/mmq/VENDOR.md` did not exist, although the
    Makefile and `ds4_cuda.cu` both pointed at it - it is written now, and says what was pruned).
    The engine's Makefile passes `-gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1`
    and nothing else that reaches a preprocessor, so none of the names 4.2b pinned is defined here
    either, and the same collapse applies: 18 tokens, 13 files, 426 deletions and 42 insertions to
    a fixed point, plus `vendors/hip.h` (308 lines) deleted.
    - Six `BLOCKED` nodes, resolved by hand to the branch nvcc had already kept, and they are the
      same six shapes 4.2b met: the VMM guard, the flash-attention guard whose MUSA term sits
      inside a negated group, the `ggml_cuda_hmax2` `#if` whose live branch is the `#elif`, the
      `__hgt2_mask` guard spanning two physical lines, the `#elif defined(GGML_COMMON_IMPL_CUDA) ||
      …_HIP || …_MUSA` in `ggml-common.h`, and mmq.cuh's `(GGML_USE_HIP && !CDNA) || __CUDA_ARCH__
      < VOLTA` stream-k guard. 19 `#endif`/`#else` comments quoted conditions the pass had just
      shortened and were rewritten, one of them (`mma.cuh`'s `#endif // defined(GGML_USE_HIP)`) a
      comment that never matched its guard even upstream - it now names the three MMA macros that
      guard does test.
    - `MUSART_HMASK` goes, its last reader was the guard above it, exactly as in the llama tree.
      Two `#undef`s go too: `ds4_ggml_stubs.h` disabled `GGML_HIP_GRAPHS` and `GGML_MUSA_GRAPHS`
      beside `GGML_CUDA_USE_GRAPHS`, and `#undef` of a macro nothing defines is a no-op.
    - Dead branches named two headers that are not in this checkout at all: `vendors/musa.h` (the
      llama tree has one, the engine never did) and `../../ds4_rocm_memory.h`, included by the HIP
      arm of the engine's own allocation pool. Both were invisible because the branch naming them
      was dead; the pool's HIP arm, its mutex and its `ds4_rocm_allocation_fits()` call went with
      it. No environment name moved: the tier's 23 `DS4_*` reads are all outside the folded
      branches, so six rows had their `path:line` re-anchored and 2,150 rows stand.
    Gate: `docs/evidence/collapse-probe.h` first (the tool's own equivalence, identical), then the
    per-file `#if`/`#endif` balance over every `.cu`/`.cuh`/`.h` in the tier (no UNBALANCED line).
    `scripts/port/cuda_live_equiv.py` grew `--include` and `--defs` for this tree - its include set
    was the llama CUDA backend's, and the engine's build defines none of the `GGML_USE_*` names -
    and over the 13 comparable paths it reports 0 differing live lines for the six that do not
    resolve through `common.cuh`, exactly three for the six that do - the two `#undef`s and the
    `#define MUSART_HMASK`, which are the intended deletions and all of them - and two for
    `ds4_ggml_stubs.h`, which holds the `#undef`s but never sees the `#define`. `vendors/hip.h` is
    reported as a deletion, not a sweep. The stronger check for a CUDA sweep is the machine code:
    `cuobjdump --dump-sass` of the nine CUDA objects rebuilt at `sm_120` is byte-identical between
    a baseline worktree and this tree except nvcc's content hash inside the internal-linkage kernel
    names and one 16-bit immediate in 50 instantiations of `mul_mat_q` - the `__LINE__`
    `no_device_code()` prints, which moved 3861 -> 3758, exactly the 103 lines the fold deleted
    above it in `mmq.cuh`. `nm` symbol sets match with that hash normalised, and `-Wall -Wextra`
    warnings are 40 occurrences and 16 kinds on both sides.
    `make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=sm_120` and
    `make -B ds4-decode.so CUDA_ARCH=sm_120` exit 0; `./ds4 --help`, `./ds4 --metal` (unknown
    option), `make test-batch-io` (io_uring matched pread, 64 MiB) and `make test-decode-library`
    (19 entry points) pass.
    `metal_word_census --strict` is unchanged over both trees, `envdocs --check` reads 2,150 with
    `test_environment_contract` 11 tests OK, the Python suite reads 673 passed / 38 skipped with
    `test_yoco` and the one moving process test failing - the same names the baseline has - and
    `release_check` reports 0 findings.
    Left, and not this chunk's: the AMD and Moore Threads compute-capability vocabulary, in this
    directory and in the llama tree - `GGML_CUDA_CC_IS_CDNA`/`_RDNA`/`_MTHREADS`, the
    `CDNA*`/`RDNA*`/`QY*`/`PH1` constants, and 210 mentions of `AMD_MFMA_AVAILABLE`/
    `AMD_WMMA_AVAILABLE` here beside 381 in `ggml-cuda/`. Their guards key on `__CUDA_ARCH__`, which
    is genuinely undefined for the host pass and defined for the device pass, so no collapse can
    decide them; deleting them is deleting AMD kernel paths by hand. The engine's host layer has a
    third layer of the same kind: 152 `#if`s on `DS4_ROCM_BUILD`, which no build of this fork
    defines either, spread over `ds4.c` and nine other files - mechanical, and bigger than this
    chunk, and the 78 `DS4_ROCM_*` rows would need a whole-tree reader scan before any of them go.

5.1 Python orchestrator. Done in four commits: `270d7a9` (memory_counters reads
    /proc/meminfo, /proc/vmstat and /proc/swaps), `65e54bd` (journal.advisory_lock, the
    lease admission lock, and the nt-only branches in shutdown/config/preferences/process/
    media_profiles/lora/vision), `c17d6a6` (slstack/cgroup_scope.py: the guardian's Job
    Object becomes a cgroup v2 subgroup, and guard_child gets the ceiling the shared-guard
    path never had), `63c8dc1` (single_runtime and build_combo_runtime stage Linux
    binaries; the fetcher's real artifact name).
    Gate: `./bin/slstack --config config.example.json check` runs here and reports
    `paging_counter_support` ok; the suite is 625 passed, 5 failed, 41 skipped with
    STUDIO_RUNTIME_ROOT set, and the five are the survivors 7.1 and 8.1 own.
    Findings this chunk produced, none of them its own to fix:
    - `tests/test_combo_guard.py` was `skipUnless(os.name=='nt')`, so 21 guardian tests
      had never run on Linux. Un-skipping them found a real intermittent failure in the
      vendored paging observer (a member exiting mid-read made `next()` raise
      StopIteration into the monitor thread, which the receipt printed as the job's stop
      reason). job_paging.py now reads zero for a status with no VmSwap line; the shared
      guard's pinned hash for that file moved with it and PROVENANCE records the patch.
    - `lease.py`'s admission lock was taken only on nt, so two instances of this stack
      could each read an empty hold ledger and both write an ACTIVE hold for one GPU.
    - `gpu_lease`'s GPU-idle check and `config.load`'s largest-GPU default now both run
      on Linux; before this chunk a config without `gpu_uuid` failed the check gate on
      `--id=`.
    - `lease.reservation_admission` recognises CPU-only reservations by their *wording*
      (`ACTIVE: KIT11FIX Windows native plugin build;...`, `Windows-only CPU build/test,
      host budget N GiB, reserve M GiB;...`). Any reservation worded differently preempts
      the engine. That is a coordination contract with the other agent on this box, not a
      platform bug - settle the wording with the owner before changing either side.
    - The guardian's scope ceiling is `job_commit_gib`, which on Linux is
      `host_budget_gib` (`process.commit_budget_gib` lost its WDDM arm). The kernel
      therefore refuses an over-budget allocation before the 50 ms RSS poll can report
      one, and the receipt's reason is derived from the scope's OOM counters.

5.2 Producer runner on Linux. Done in `5fb07f7`. `slstack/producer_runner.py` is the
    counterpart of `vendor/studio/studio/dsv41_windows_runner.py` with the same
    `_Native`/`_Job` surface, and `default_runner()` chooses between the two, so the four
    combo call sites and the vendored `Producer` no longer name a platform.
    Suspension is `slstack/producer_launch.py`: a child that takes its own process group,
    reports its pid, and execs the engine only when the controller writes one byte, so no
    engine instruction runs before containment is confirmed - what
    `CreateProcessW(CREATE_SUSPENDED)` + `AssignProcessToJobObject` + `ResumeThread` gave.
    Its `PR_SET_PDEATHSIG` retires a child that was never admitted, and `execv` clears that
    setting, which is why `release_kill_on_close()` records the release rather than making
    one. Ctrl-Break is SIGINT to the group, `TerminateJobObject(137)` is SIGKILL to it, and
    `ActiveProcesses` is the group's live members, so a force-killed producer's receipt
    reports exit `-9` where Windows recorded the 137 it passed in. Containment is the cgroup
    the controller already sits in, not a new sibling scope: the retired job was created with
    `CREATE_BREAKAWAY_FROM_JOB` and carried no memory limit, and moving the producer out
    would have taken it out of the guardian's ceiling, so `job.scope` only records what the
    tree inherited. Leases reuse chunk 5.1's lock - `journal.advisory_lock(shared=True)` is
    `FILE_SHARE_READ`'s counterpart, and `advisory_lock` now reports `LockUnavailable` only for an
    attempt the kernel refused (`EWOULDBLOCK`), because `lease.gpu_lease` turns that class into
    `AdmissionRefused` and an unrelated `OSError` - an interrupted wait, a closed descriptor - must
    not surface as an admission already in progress - and `_Native._open` applies the provider's own
    statfs magic refusal list, with `O_NOFOLLOW`, `realpath` equality and `st_dev`/`st_ino` in place
    of `QueryDosDeviceW`, `GetFinalPathNameByHandleW` and `FileIdInfo`. The lock is advisory:
    what a share mode refused against a non-cooperating opener is not reproduced here, the
    identity rechecks are. `launch_environment()` puts the producer's library directories in
    `LD_LIBRARY_PATH` as well as PATH, because the Linux loader is what reads them.
    Gate: `tests/test_producer_runner.py` (22 checks, no model and no GPU allocation) and
    `docs/evidence/producer-launch-gate.py`, which runs the real CLI through the vendored
    `Producer` with the no-model argv and reaches "Selected GPU free bytes do not exceed the
    explicit reserve" with `owned_job_empty`, `owned_process_signaled` and `owned_pid_gone`
    true and no launcher left blocked. Readiness needs a qualified GGUF, and the engine's
    own `--gpu-reserve-bytes` floor is 1 GiB, so on a busier box this gate aborts inside
    ggml CUDA init instead - read it against the headroom probe. The cooperative SIGINT path
    is proven against CPU fixtures, not against the engine, which exits before a retirement
    window can open.
    Findings this chunk produced, none of them its own to fix: neither the engine's nor this
    runner's volume list refuses NFS4's magic (`0x6e626470`), only `NFS_SUPER_MAGIC`, so an
    NFS4 mount is accepted on both sides; `combo_native`'s model-drive refusal keys on a
    Windows drive letter and so no-ops here; and `working_set`/`combo_ram` reach for
    `job.pi.process`, which this runner does not expose - chunk 5.3 reads `job.scope` and
    `job.pid` instead.

5.3 Resident admission and the locked checkpoint. Done in `1279063`. The two verbatim copies of
    `protect_working_set` are one function now: resident admission (`protect_working_set`,
    `wait_owned_initialization`) lives only in `slstack/working_set.py`, which `combo_native` and
    `combo_producer` import directly, and `slstack/combo_ram.py` keeps the section. Linux has no
    per-process page floor, so the arm is `memory.min` on the scope the job inherited
    (`job.scope`), written in whole pages and kept below the scope's ceiling by the same 64 MiB
    metadata headroom the Windows arm kept between its minimum and its advisory maximum - a
    subgroup whose floor equals its ceiling has nothing left to evict, so pressure there would
    become a termination instead of an eviction of page cache.
    Two decisions, both settled by measuring the kernel here rather than by reasoning:
    - **No scope of its own for the producer.** A scope that has member processes cannot
      delegate the memory controller - writing `+memory` to `cgroup.subtree_control` answers
      EBUSY (16) - so a producer scope nested under the guarded one is impossible, and a
      sibling created in `app.slice` would take the producer out of the ceiling chunk 5.2
      deliberately kept. The floor therefore bounds the guarded tree, controller included,
      which is the same tree the ceiling already bounds and which host admission proves fits.
      A scope this stack did not create (no `slstack-` prefix: a terminal's or a session
      manager's) is reported and left alone - raising a floor on somebody else's tree is not
      this stack's to do.
    - **The floor is written and reported, not counted as residency.** The kernel zeroes a
      subgroup's protection as soon as any ancestor between it and the root carries no floor,
      and the ancestors above `app.slice` here are root-owned (`user@1000.service/memory.min`
      is uid 0, mode 0644), so an unprivileged process cannot complete that chain. Measured
      with `memory.reclaim` against a child holding 64 MiB of page cache in a subgroup floored
      at 64 MiB: the kernel took all of it back, both with `app.slice` at 0 and with `app.slice`
      raised to 64 MiB too. So the row says `honored: false` with the blocking ancestor and the
      run prints `DS4COMBO_RESIDENT_FLOOR_INERT`; what actually holds the pages is the scope's
      `memory.swap.max=0` and the guardian's paging guard, as chunk 3.1h concluded for the
      carry tier. A deployment that floors its own slice chain gets residency from this write
      with no change to the code.
    `LockedRam` is the same tier on `/dev/shm`: `O_CREAT|O_EXCL` 0600, `ftruncate`, `MAP_SHARED`
    mmap, chunked best-effort `mlock` halving at each refusal (the engine's `lock_pages` rule,
    whole pages, `DS4COMBO_LOCKED_RAM_PARTIAL` for the shortfall) and `munlock` of what it
    captured. The checkpoint's RAM-backed directory is not the producer's volume rule: that rule
    refuses RAM-backed *producer* paths (weights, receipts, outputs) and the section never goes
    through `_Native._open` or `volume()`, so the refusal list was left alone and the exception is
    stated where the rule lives. A tmpfs name outlives the process that made it, which a kernel
    section's handle did not, so the name carries the creating pid and `reap_stale_sections()`
    removes owned names whose creator is gone - the rule `cgroup_scope.reap_stale` uses for
    capped directories. `attach(name,capacity)` is the child-side `OpenFileMappingW` +
    `MapViewOfFile`, and it hands back the same bounded-transfer surface (`SharedSection`), which
    is now the half `LockedRam` shares with it.
    Gate: `tests/test_working_set.py` (8 checks on Linux, one Windows arm still skipped) against
    real owned CPU children and real scopes - the floor lands on the inherited scope, a foreign
    scope and a missing scope are reported without being touched, the ceiling gap is kept, an
    admission that rounds below one page refuses, and a failed `after_resume` certifies the owned
    cleanup on Linux now that the runner is Linux. `tests/test_cgroup_scope.py` (5 checks added)
    covers the whole-page rounding, the readback and the ceiling it reports, a limit written as
    `max`, the refusal where there is no memory controller, and the claim that the reported honour
    is the kernel's actual answer, applied through `memory.reclaim`. `tests/test_combo_ram.py`
    (7 checks un-skipped from the Windows class) covers bounded transfers, the cross-process
    attach and the child's write, exclusive creation, name validation, the stale-name reap, the
    teardown that removes the name, and the lock receipt. The five chunk-5.x Python files read 68
    passed, 4 skipped against the same command at `c4ddbf8` reading 48 passed, 8 skipped - 20 more
    running, three names un-skipped, and one duplicated Windows assertion deleted with the copy of
    `protect_working_set` it belonged to. The full suite reads 669 passed, 6 failed, 37 skipped
    against this chunk's own 648/7/41 baseline, and every failing name in both runs is one of the
    five this plan's Phase 7 and Phase 8 own or one of the three process tests that flip between
    runs.
    `docs/evidence/producer-launch-gate.py` now rides chunk 5.2's launch with chunk 5.3's
    admission: the job is moved into a scope the gate created, `protect_working_set` floors it,
    and the run still ends at the reserve refusal with the ownership certificate clean and the
    scope removed. `combo_ultra_parity.py`'s shared-RAM handoff still needs a qualified GGUF, so
    it is unrun.
    Findings this chunk produced: `mlock` past `RLIMIT_MEMLOCK` answers ENOMEM (12) on this
    kernel, not the EAGAIN the man page leads with; and a whole-user floor chain is not reachable
    without privileges, so a resident admission on Linux is a claim about the tree this stack
    created, never about the host.

6.1 Vendor and engine bloat. Partly landed. `vendor/mac-mlx`, `engines/mlx-receiver`,
    `engines/mlx-serve`, `engines/mference` and the MLX lock files went in 1.10a and 1.10c;
    `patches/mlx-core-ring-listeners.*` and `scripts/setup-mlx.py` (whose `slstack.mlx_setup`
    import was already dead) went in 7.1a. What is left is
    `vendor/engines/cosmos_windows`: Windows-shaped media workers that the media tests import,
    and `slstack/media_tools.py`, which downloads a pinned Windows FFmpeg archive and on
    non-nt only reports the system `ffmpeg`/`ffprobe` on PATH. Port media to Linux or delete it;
    the README already calls it experimental.

7.1 Docs, examples, presets, tests. Three commits landed; two pieces remain.

7.1a Done (`5b941ff`): the Mac and paired doc surface is deleted - six `MACDECODE` rounds, four
    `macdecode-r*` receipt directories, `macdecode-measurements.json`, `TWO_MACHINE.md`,
    `YOCO_PAIRED_*`, the October 27 Mac build receipt, `examples/engine_environment/mac-*.json`,
    `scripts/setup-mlx.py`, `patches/mlx-core-ring-listeners.*` and ten test sources whose
    includes named `engines/ds4-mac`; 86 tracked files, 67,638 lines. `README.md` describes one
    Linux machine, and `BACKEND_SETUP.md`, `BACKEND_SOURCE_AUDIT.md`, `SUPPORTED_MODELS.md`,
    `CONE_STATE.md`, `WALKTHROUGH.md` and `examples/single-machine/README.md` were rewritten to
    what this box actually runs (two backends, `build.sh`, the `.sh` launchers, and the fact that
    the `-ultra-yoco` filenames are approximate keep 140 rather than exact cone). 26 measurement
    docs gain an inherited-history label under their heading, which is what the phase asked for
    instead of deleting the Windows and macOS numbers; `PERFORMANCE.md` loses the Mac decode
    columns and its paired section and keeps the RTX prefill history.
    Gate: no tracked markdown links a file that does not exist, no tracked file names a deleted
    path, `release_check` 0 findings.

7.1b Done (`9a822c8`): the four suite names this phase owned pass. `test_streamed_recipe_argv`
    calls `vendor()` before importing `studio` and hands the trainer the same PYTHONPATH the
    launcher does; `test_dsv41_recipe` passes that PYTHONPATH to its child; `test_h3_video_delivery`
    keeps the Windows marker case skipped here and adds the POSIX arm, where readiness is what
    PATH reports and a Windows marker neither creates nor breaks it; `test_shared_prefill_profile`
    drops its `slstack.mac` half and says native export where it said paired. `scripts/export_ds4_sampling.py`
    read `engines/ds4-win/ds4.c`, which is gone: repointed at `engines/ds4`,
    regenerating `studio-sampling.c` moves only the recorded source SHA and leaves all nine
    exported sampler functions byte-identical. `scripts/build_combo_runtime.py`'s preset name
    loses `ds4-win` and its empty `mac={}` block.
    Gate: the full suite reads 672 passed, 3 failed, 38 skipped, the three being the two process
    tests that flip between runs plus `test_yoco` (chunk 8.1's), against the baseline's 668/7/37;
    `release_check` 0 findings; `envdocs --check` still 2,162 names.

7.1c Done (`050e0b1`): `scripts/port/metal_word_census.py` kept reading a `//` comment as open
    after its newline, so every identifier on a later line was filed as COMMENT and an
    `IDENTIFIER 0` answer proved nothing. The engine's numbers are unchanged by the fix, so
    chunk 2.2e's gate stands; the llama tree moves to IDENTIFIER 1, STRING 3, COMMENT 51 lines,
    and `--strict` now fails there - the 4 recorded for STRING in that commit's message and in this
    plan for a session was a mention/line mix-up, re-measured against `050e0b1` in chunk 4.2. `llama_split_graph_create`'s refusal line still printed
    "need CUDA or Metal" although chunk 4.1's message, this plan and the handoff all claimed it
    had been reworded: it is reworded now, with `preprocess_equiv` showing 0 differing positions
    on the CUDA build (the line is inside `#ifndef GGML_USE_CUDA`) and exactly one on a CPU-only
    preprocessing. `docs/linux-port-commands.md` carries the corrected recipe and numbers.

7.1d The env inventory cannot see a wrapper read, so `docs/VARIABLES.md` understates the engine.
    Pending, and it is the largest remaining piece of this phase. `envvars.py:inventory()` matches
    literal `getenv("NAME")`; a read through a wrapper such as `glm_graph_env_present(
    "DS4_ROCM_X", "DS4_METAL_X")`, `gpu_graph_tp_env_flag("NAME", dflt)`,
    `cuda_stream_tier_bound_env("NAME")` or `ds41_env_flag(name)` is invisible, and the census
    counts 123 `DS4_*METAL*`/`ENV_NAME_CITED` names in that class today. Fix the inventory (a
    second pass: functions whose body calls `getenv` on a parameter, then the string literals at
    their call sites, joining continuation lines), then add a row per name by hand before
    `--refresh-defaults`, then regenerate. Renaming the names is not the fix - 2.2b kept them for
    recipe compatibility. The same chunk must decide what `engines/mlx-serve/` read references
    mean now that the gitlink is gone: `envvars.py:validate_table()` and `envdocs.py` still
    exempt references under that path, and `envvars.json` holds 28 of them, so those rows are
    documented reads in a tree this checkout does not contain.
    Gate: `envdocs --check` passes with the new rows, the name set only grows, and
    `metal_word_census` reports `ENV_NAME_CITED 0`.

7.1e The Mac peer's text in the llama tree. Pending: 51 comment lines across 20 files and 3
    printed strings (`ds4split-expert-projection.cuh`, `ds4split-projection-fixture.cuh`,
    `tools/decode-receiver/main.cpp`), plus `decode-receiver`'s 13 comment lines and its printed
    `LLAMA_PD_CACHE_CAPTURE` warning that describes the deleted Mac backend. Use
    `scripts/port/comment_code_diff.py` and `string_text_replace.py`, check `__LINE__` first, and
    keep provenance comments that say what was deleted rather than pretending it exists.
    Gate: `metal_word_census --strict` reports 0 identifiers and the comment count drops to the
    provenance lines, `comment_code_diff.py` reports 0 changed code lines for a comment-only
    sweep, and the warning profile loses no kind.

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

## Machine facts checked on this box (re-checked 2026-11-06)

Arch Linux, kernel 7.2.14-arch1-1, RTX 5090 32 GiB (driver 615.78.08), CUDA 13.4 at `/opt/cuda`
(not `/usr/local/cuda`), nvcc 13.4 with `sm_120`/`sm_121`, cmake 4.4.5, ninja 1.13.2,
`/usr/include/liburing.h` present, `/opt/cuda/lib64/libcufile.so` present, Python 3.14.9, ext4 on
one NVMe with 359 GB free, 93 GB RAM (the 30 GB figure in earlier notes is stale; the box has more),
16 cores.

`nvidia-fs` is not installed, so GDS is unavailable at runtime even though the toolkit library
links; io_uring is the route.

cgroup v2 is unified (`/sys/fs/cgroup` is cgroup2fs) and the user's `app.slice` delegates
`memory pids`, so an unprivileged process can create its own subgroup there and cap it - that is
what chunk 5.1's guardian does. `systemd-run --user --scope -p MemoryMax=...` works too, but it
puts the command one generation below the caller, which breaks the guardian's identity checks
(`owner==child.pid` or `candidate.ppid()==child.pid`), so the port uses the subgroup directly.
The process's own scope does *not* delegate `memory` to its children (its
`cgroup.subtree_control` is empty), which is why `controller_parent()` walks up out of it.
Measured in chunk 5.3: a scope this stack created delegates `memory` while it is empty and
answers EBUSY (16) as soon as it has a member process - the no-internal-processes rule, and
the reason the producer cannot get a scope *inside* the guarded one.
Exceeding `memory.max` with `memory.swap.max=0` SIGKILLs the tree, and
`memory.oom.group=1` takes every member, not only the offender. `memory.min` is the half that
does not work here: the kernel zeroes a subgroup's protection when any ancestor between it and
the root carries no floor, and above `app.slice` the ancestors are root-owned
(`user@1000.service/memory.min` is uid 0, mode 0644), so an unprivileged process cannot complete
the chain. Measured with `memory.reclaim` against a child holding 64 MiB of page cache in a
subgroup floored at 64 MiB: all of it came back, with `app.slice` at 0 and also with `app.slice`
raised to 64 MiB. `RLIMIT_MEMLOCK` stays 8 MiB soft *and* hard, so `mlock` remains best effort
(chunk 3.1h), and on this kernel a lock past it answers ENOMEM (12) rather than the EAGAIN the
man page leads with. The scope ceiling is a different mechanism and needs no allowance.

Two things limit what a gate can prove here:

- The GPU is occupied by another process (`strata`, ~30 GiB of 32 GiB VRAM, and its pid changes
  between sessions). Compile and link gates run - `llama-server --version` enumerates a device
  without a context even with Strata up - but anything that allocates VRAM contends with it, and on
  2026-11-06 with 630 MiB nominally free, **creating a context itself failed**: a program doing only
  `cudaFree(0)` and `cudaMemGetInfo` returned `out of memory`, so the producer's no-model receipt
  smoke aborts in `ggml_backend_cuda_get_device_memory` rather than reaching its reserve refusal.
  Treat every CUDA gate as unrun until the VRAM is actually free, and check with that two-call
  probe before blaming a change.
- No DeepSeek V4 weights exist locally **in a form ds4 can load**. Re-read 2026-11-06 with
  `slstack/gguf_header.py`: `qwen38-gyro/Strata/models/deepseek-v4.1-flash-IST-DAS-2.00bpw/` holds
  all four shards (138 GB) of a `deepseek41`, 40-block model with `split.count=4`, but its expert
  triplets mix `IQ1_M`/`IQ2_XS`/`IQ2_XXS` and its dense tensors are `Q5_K`/`BF16`, which
  `validate_dsv41_types` refuses - the accepted triplets are `(Q4_K,Q4_K,Q4_K)`,
  `(IQ2_XXS,IQ2_XXS,Q2_K)` and `(MXFP4,MXFP4,MXFP4)`, dense `Q8_0`/`Q4_K`/`Q4_0`, `engram_embd`
  `I8`, `token_embd` `F16`, the rest `F32`. Assembling the shards would not qualify it; the quant
  set is the blocker. The `UD-Q8_K_XL` set is 1 of 9 shards, the Engram table is a separate
  `IQ4_NL` file (`deepseek41.engram.file`) rather than rows inside the model, and the three Gyro
  `qwen4exp` files carry TQ tensor types (ids 144-149) outside the engine's table. Phase 3's
  prefill gate and Phase 8 need a model in the accepted type set - a download or a quantize run -
  and 336 GB free on the NVMe is enough for either. The HANDOFF records the detail.

## How to resume this plan

```sh
cd streaming-llama-stack
git log --oneline -1
git status --short                 # Phase 1 chunks 1.1-1.5 land here
grep -rn "_WIN32\|windows.h\|Win32\|DStorage\|IoRing\|msvcrt\|WinDLL" engines slstack vendor scripts examples tests | wc -l
grep -rn "__APPLE__\|TARGET_OS\|Metal\|mlx\|MLX\|darwin\|Darwin" engines slstack vendor scripts examples docs tests | wc -l
python3 -B scripts/release_check.py
```

Counts, re-counted after chunk 7.1c with the commands above plus `-I` (skip binary files - the
built engine binaries and `*.o` under `engines/ds4` match these patterns and are not source) and
`--exclude-dir=build --exclude-dir=.git --exclude-dir=__pycache__`: 443 Windows references and
691 Apple/MLX ones repo-wide, down from the 488 and 920 this file quoted after chunk 5.1 because
7.1a deleted the Mac docs, receipts and test sources. Earlier figures in this file (504, 532 and
491) were counted differently - some without `-I` - so they are not comparable; re-run the command
rather than trusting any number here. Inside `slstack/` and `scripts/`, with `--exclude-dir=port` so
`scripts/port`'s own sweep `--tokens` list does not count, the gate in
`docs/linux-port-commands.md` reports 5 lines after chunk 7.1b: the two `ctypes.WinDLL('kernel32')`
calls in the Windows arms of `combo_ram.py` and `working_set.py`, which run on both platforms, and
three provenance lines (`slstack/producer_runner.py`'s mapping table and `working_set.py`'s
docstring) naming the retired API those arms replace.
`media_tools.py`'s Windows FFmpeg download is not in that
number - it keys on `os.name` and `.exe` names rather than on those tokens (Phase 6). Inside
`engines/ds4` `metal_word_census.py` reports IDENTIFIER 0, ENV_NAME 195 lines, ENV_NAME_CITED 123
lines and COMMENT 26 lines: the documented `DS4_METAL_*` names are kept on purpose, the 123 cited
ones are the wrapper reads chunk 7.1d owes the inventory, and the 26 comment lines are the
provenance chunk 2.2e kept on purpose. Over the llama tree the same gate reports IDENTIFIER 0,
STRING 3 and COMMENT 51 lines and `--strict` passes, since chunk 4.2 deleted the tree holding the
one code-position mention (`USE_METAL_GUEST_SUPPORTS_OP` in `ggml/src/ggml-virtgpu/`). The 51
comment lines and 3 printed strings are the Mac peer's measurement history and the ten
retired-backend mentions 4.2 left behind - chunk 7.1e's. Re-count before trusting any of these
numbers.
Windows in `engines/ds4`: 3 hits, all vendored (`external/nlohmann/json.hpp` x2, the `linenoise.c`
comment). Windows in `engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill`, in
`src/llama-weight-fetcher.{cpp,h}`, in `src/llama-dsv41-weights.cpp` and in
`src/llama-dsv41-engram-prefetch.{cpp,h}` and in `src/llama-dsv41-carry.cpp` and
`src/llama-dsv41-locked-ram.h`: 0 conditional or include hits (chunks 3.1d, 3.1e, 3.1f, 3.1g,
3.1h); the one survivor is a provenance comment in the loader that names the IoRing it was ported
from, so gate greps still need `| grep -vE ":[0-9]+:\s*//"`. `tests/dsv41_engram_early_io.cpp` and
`tests/pinned_carry_memory.cpp` are Linux with the code they test; two test sources remain
Windows-shaped (`dsv41_producer_abort.cpp`, `studio_generation.cpp`) and none of them has a build
rule. The repo-wide pattern above now reads 443 Windows references and 691 Apple/MLX ones, unchanged
by chunk 4.2 - the nine deleted backend trees carried no `_WIN32`, `windows.h` or `IoRing` text at
all, and their platform abstraction was their own. The rest
of the producer tree is still Windows-shaped: 15 files under `src/` by that pattern (one of them a
comment), 6 under
`common/`, 4 under `tools/mtmd`, 3 under `tools/server`, 1 under `tools/decode-receiver`. Re-count
before trusting any of these numbers.

`python3 -B scripts/release_check.py` reports 0 findings since `docs/VARIABLES.md` was regenerated
in chunk 1.10b.

Python suite: `timeout 900 env PYTHONPATH=.:tests STUDIO_RUNTIME_ROOT=$PWD/runtime python3 -B
-m pytest tests/ -q` reports **672 passed, 3 failed, 38 skipped** after chunk 7.1b. Set
`STUDIO_RUNTIME_ROOT` (the repo's gitignored `runtime/` works: `mkdir -p runtime`) or 21
process tests raise `KeyError: STUDIO_RUNTIME_ROOT`, and chunk 5.3's floor-honour check needs a
disk-backed fixture directory too; without it the same tree reads a much worse line, which is why
counts alone mislead. Two process tests vary run to run
(`test_real_owned_child_stops_on_kit11_hold`,
`test_persistent_window_denial_still_stops_owned_job`), so compare the set of names, not the
number. The one stable failure is `test_yoco`, whose preset reads `producer_prefill_sweep_tokens`
as `auto` rather than the 49,152 it asserts - chunk 8.1's. Chunk 7.1b removed the four names Phase
7 owned: `test_dsv41_recipe` and `test_streamed_recipe_argv` (they needed the vendored paths on
`sys.path` the launcher hands a child), `test_h3_video_delivery`'s Windows marker case (now
skipped here, with a POSIX arm added, which is why the skip line moved 37 to 38) and
`test_shared_prefill_profile`'s `slstack.mac` import.

`python3 -B -m slstack.envdocs --check` reports 2,150 documented variables. Chunk 5.1 moved no
row and dropped no name; chunk 4.1 is the first chunk that did, because deleting a backend deletes
its `getenv` sites - 71 rows went, and 7 more were re-anchored to new line numbers after the
comment rewording. Chunk 4.2 dropped 12 more (the `GGML_CANN_*`, `GGML_OPENCL_*` and
`GGML_REMOTING_*` rows whose last reader was a tree nobody built) and re-anchored 29 rows whose
reads moved inside `ggml-cuda` when its HIP and MUSA branches folded. No name was added.

Build a CUDA engine with `make -C engines/ds4 cuda CUDA_ARCH=sm_120 -j4` (~4 min). `make cpu`
builds the same binary names as CPU variants, so rebuild CUDA afterwards if you need the CUDA
binaries.

Mechanical sweeps (deleting a platform's dead branches, renaming an inherited identifier) use the
scaffolding in `scripts/port/` - `collapse_conditionals.py`, `preprocess_equiv.py`,
`rename_identifiers.py` - and its README states the proof each one gives. A deletion chunk is only
behaviour-preserving if the rule is "keep the branch the compiler already kept", and a rename is
only safe if the preprocessed live text is unchanged; do not hand-edit either sweep.

`.githooks` holds `pre-commit` and `commit-msg`, but **this checkout does not run them**:
`git config --get core.hooksPath` is empty and `.git/hooks` holds only the samples, so
`release_check.py` has to be run by hand before committing. Wiring it up would also refuse every
commit here: `--identities` requires the `Front Eagle <fronteagle739@users.noreply.github.com>`
identity that the fork's history carries, while this box's git config is `Mac`, and the port's own
commits are `Mac`-authored. Regenerate the env table before committing work-in-progress chunks
rather than reaching for `--no-verify` over a hook that is not installed.
