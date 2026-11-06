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
the three vendored hits, and no code identifier or CLI surface names Metal. The gate as first
written also grepped for `Metal`, which cannot pass: the owner kept the 215 `DS4_METAL_*`
environment names in chunk 2.2b because they are documented in `docs/VARIABLES.md` and named in
`docs/macdecode-r6/r6-controls.json`. What is left for those names to sit in is comment text and
the inert expert-profile subsystem's own messages - chunk 2.2e below.

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
5. Open. Replace `llama-dsv41-locked-ram.h` and the carry lease in `llama-dsv41-carry.cpp`. This is
   the locked-memory decision (`mlock` needs an `RLIMIT_MEMLOCK` or cgroup allowance) and it also
   gates the decode library's host-cache path.
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

2.2e Metal words in comments and in the inert subsystem's messages. Open. 310 case-insensitive
    `metal` mentions remain in `engines/ds4` outside `DS4_METAL_*`, about 190 of them inside quoted
    strings, mostly in the expert-locality profiler that chunk 2.2c left inert. Rewording them is
    text work with no behaviour attached; it does not need the equivalence tool, but it does need
    the env-table regeneration because lines move.

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

3.1 Streaming producer on Linux. What is left of Phase 3 is still a port, not a sweep.
    Decide the locked-memory mechanism and replace `llama-dsv41-locked-ram.h` and the carry
    lease; then the Win32 host code in `mtmd`, `server` and `decode-receiver`.
    Gate: a 262K prefill, which needs a qualified GGUF on this box (see the machine facts).
    Findings for whoever starts it: `tools/slstack-weight-fetcher.cpp` is a two-line anchor
    file, so the `.so` it builds exports one data symbol and carries no fetcher API - chunk
    5.1's `single_runtime.py` PRODUCER_FILES cannot stage it as the DLL's twin; and
    `vendor/engines/cosmos_windows/weight_streaming/ds_backend.py` mirrors `wf_params` by ctypes
    field order, which the 3.1e and 3.1f deletions shifted - that module is Phase 6's to delete.
    Two things need re-running when the GPU has headroom: `docs/evidence/wf-io-selftest.cpp`
    already passes, and the CUDA arms of the extent reader (`docs/linux-port-commands.md` has the
    link line) have never executed - a `ggml` CUDA backend wants about 700 MB past what Strata and
    the browser leave.

3.2 Producer decode library wiring. Done as part of 3.1c, because `live-session.inc` is part of
    `main.cpp`'s body: `dlopen`/`dlsym` of `ds4-decode.so` beside the executable in place of
    `LoadLibraryExW`/`GetProcAddress`, with `make cuda-decode` producing it. Verified without a
    model by the selftest's `open_decode_library()` check: all 8 required and all 10 optional
    `ds4x_*` exports bind. A live session against weights is still unrun.

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

## Machine facts checked on this box (re-checked 2026-11-06)

Arch Linux, kernel 7.2.14-arch1-1, RTX 5090 32 GiB (driver 615.78.08), CUDA 13.4 at `/opt/cuda`
(not `/usr/local/cuda`), nvcc 13.4 with `sm_120`/`sm_121`, cmake 4.4.5, ninja 1.13.2,
`/usr/include/liburing.h` present, `/opt/cuda/lib64/libcufile.so` present, Python 3.14.9, ext4 on
one NVMe with 359 GB free, 93 GB RAM (the 30 GB figure in earlier notes is stale; the box has more),
16 cores.

`nvidia-fs` is not installed, so GDS is unavailable at runtime even though the toolkit library
links; io_uring is the route.

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

Counts, re-counted after chunks 2.2c/2.2d, 3.1b, 3.1c and 3.1d with the commands above, adding
`--exclude-dir=build --exclude-dir=.git` (the port's own scratch trees and object files now live
under `build/`, which is gitignored but matched the greps as first written; the engine's own
binaries, `*.o`, `ds4-decode.so` and the `__pycache__` directories match too, so these totals read
higher than the source alone): 1,107 Apple/MLX references repo-wide, unchanged by the producer work,
and 700 Windows ones - down from 774, the drop being `tools/dsv41-prefill` (chunk 3.1d) and the
comments chunk 3.1c reworded. Both figures also include this plan's own quoted grep patterns and the
inherited measurement history in `docs/`. Inside `engines/ds4` the code-identifier gate
(`\bmetal_\|\bMETAL_`) returns 0, `DS4_BACKEND_METAL` returns 0, and the remaining 269 Apple/Metal
matches are the 215 `DS4_METAL_*` environment names kept on purpose plus comment text (chunk 2.2e).
Windows in `engines/ds4`: 3 hits, all vendored (`external/nlohmann/json.hpp` x2, the `linenoise.c`
comment). Windows in `engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill`, in
`src/llama-weight-fetcher.{cpp,h}`, in `src/llama-dsv41-weights.cpp` and in
`src/llama-dsv41-engram-prefetch.{cpp,h}`: 0 conditional or include hits (chunks 3.1d, 3.1e, 3.1f,
3.1g); the one survivor is a provenance comment in the loader that names the IoRing it was ported
from, so gate greps still need `| grep -vE ":[0-9]+:\s*//"`. `tests/dsv41_engram_early_io.cpp` is
Linux with the loader; four test sources remain Windows-shaped (`cuda_other22_graph.cpp`,
`dsv41_producer_abort.cpp`, `pinned_carry_memory.cpp`, `studio_generation.cpp`) and none of them
has a build rule. The repo-wide pattern above now counts 532, down from 633 before 3.1g. The rest
of the producer tree is still Windows-shaped: 18 files under `src/` by that pattern (one of them a
comment), 6 under
`common/`, 4 under `tools/mtmd`, 3 under `tools/server`, 1 under `tools/decode-receiver`, plus the
ggml backends Phase 4 deletes. Re-count before trusting any of these numbers.

`python3 -B scripts/release_check.py` reports 0 findings since `docs/VARIABLES.md` was regenerated
in chunk 1.10b.

Python suite: `timeout 600 env PYTHONPATH=.:tests python3 -m unittest discover -s tests -p
'test_*.py'` runs 584 tests with 5-6 failures and 5 errors (fork baseline 6 failures and 6 errors
over 585). Two of the process tests vary run to run (`test_real_owned_child_stops_on_kit11_hold`,
`test_disconnect_inside_reasoning_then_next_turn_appends_only`), so compare the set of names, not
the count. The stable survivors are Linux gaps, not Mac leftovers: Windows-only guard and producer
paths (`test_combo_process`, `test_shared_guard_adapter`, `test_single_liveness`,
`test_journal_recovery_flags`), the missing `cosmos_windows` vendor module
(`test_streamed_recipe_argv`, `test_dsv41_recipe`, `test_yoco`), and GPU/video gates that need a
model on disk (`test_h3_video_delivery`, `test_yoco_transfer`). Chunk 5.1 and 8.1 own them.

Build a CUDA engine with `make -C engines/ds4 cuda CUDA_ARCH=sm_120 -j4` (~4 min). `make cpu`
builds the same binary names as CPU variants, so rebuild CUDA afterwards if you need the CUDA
binaries.

Mechanical sweeps (deleting a platform's dead branches, renaming an inherited identifier) use the
scaffolding in `scripts/port/` - `collapse_conditionals.py`, `preprocess_equiv.py`,
`rename_identifiers.py` - and its README states the proof each one gives. A deletion chunk is only
behaviour-preserving if the rule is "keep the branch the compiler already kept", and a rename is
only safe if the preprocessed live text is unchanged; do not hand-edit either sweep.

`.githooks` is wired through `core.hooksPath`, so `pre-commit` runs `release_check.py` and will
refuse a commit while the VARIABLES finding stands. Either regenerate the table first or commit
the work-in-progress chunks with `--no-verify` and say so in the commit message.
