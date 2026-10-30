# streaming-llama-stack: repo review and Linux port notes

Repo: https://github.com/fronteagle739/streaming-llama-stack
Reviewed: 2026-10-30, via `gh` plus a local clone at `/tmp/sl-stack`
MIT, single author "Front Eagle", created 2026-10-25, last pushed 2026-10-30.
466 commits between Oct 22 and Oct 30. 5 stars, 0 issues, 0 PRs, 0 releases, no GitHub Actions beyond `.github/FUNDING.yml`.

## What it is

Runs a 483 GiB DeepSeek V4.1 Flash mixture-of-experts model on one RTX 5090 with 32 GB VRAM and 32 GB system RAM, by streaming weights from NVMe into the GPU through DirectStorage. Optionally splits the work across two machines: the RTX prefills, a 512 GB Mac Studio M3 Ultra decodes.

It ships a llama.cpp fork with streaming prefill, ds4 decode kernels for V4.1 Flash, an OpenAI-compatible API, MCP tools, and the llama.cpp web UI.

### Measured numbers, all on real text

199,952 tokens of *I Promessi Sposi* with a password at the start and a recall question at the end. Every run answered the password.

| setup | prefill | decode |
|---|---|---|
| RTX packed FP4, 200K | 43.2 s = 4,632 tok/s | 7.6-7.9 max quality, 22.7 max speed |
| RTX Q4_K/Q8_0, 200K | 59.6 s = 3,356 tok/s | 7.5, 17.8 |
| RTX packed FP4, 100K | 25.5 s = 3,917 tok/s | not reported |
| Mac Studio alone, 200K | 384 tok/s over 521 s | 36.6 |
| RTX prefill to Mac decode, 200K | 55.4 s effective, link-bound at 5 Gb/s | 37.0 |

The README retires its own earlier figures. The old 5,240 tok/s came from 100,000 identical tokens, which needs about 1,200 Engram rows per table against 1.66M on real text, so it hid roughly 37 s of Engram and spill cost. That honesty is unusual and it makes the remaining numbers worth trusting.

## Repo shape

```
engines/     78 MB, about 1.08M lines of C/C++/CUDA/Metal/Python/Swift
  streaming-llama/    llama.cpp fork at b831118, 1034 files, 601k lines, plus vendored FlashInfer sm120
  ds4-win/            147 files, 277k lines, DirectStorage CUDA engine
  ds4-mac/            123 files, 252k lines, fork of antirez/ds4 at bd66c40 plus IngeniousIdiocy V4.1 ports
  ds4-peer/           39 files, 140k lines, cross-machine peer variant
  mlx-receiver/       MLX adapters, GLM and Kimi MTP drafters
  mference/           Swift package
vendor/      15 MB, cosmos_windows media engines, gguf-tools, mac-mlx
slstack/     85 Python files, 14k lines, the CLI and orchestrator
docs/        117 files, PERFORMANCE.md, about 20 EXPERIMENT_E*.md, qualification JSONs
examples/    17 MB, presets plus recorded proof runs
tests/       202 files, C++ and CUDA gates plus Python gates
```

`docs/VARIABLES.md` is 103k lines generated from `envvars.json`, which is 6.4 MB. It documents every knob in the engines.

## How it works

Streaming prefill reads each layer's weights once per pass and pushes the whole prompt chunk through that layer before moving on, so the weight-read cost is shared across the chunk. A complete V4.1 Q4 pass reads 315.8 GB. DirectStorage prefetches the next layer while the GPU computes the current one. Passes are 49,152 tokens.

Engram rows are the big win. They used to be about 3M random 4-8 KiB reads per sweep. Reading them as 1 MiB extents through one IoRing at QD64 took the 200K FP4 prompt from 76.2 s to 43.2 s.

Decode keeps dense weights in VRAM, caches routed experts in VRAM, and reads misses from NVMe. After about 100 tokens the cache holds the experts the conversation uses.

The quality/speed dial trades quality for speed by preferring cache-resident experts over disk fetches. Max quality never does that, so it is the full model. It is settable per phase from the web UI, the API, or MCP.

Exact cone means the ULTRA defaults compute every encoder row plus the decoder's final 2,541-row dependency cone, so replay is bit-aligned to full prefill.

YOCO is optional approximate prefill from DeepSeek's DeepInfer, keeping the last 140 positions. The streaming variant skips decoder layers 21-39 on every pass except the last, and is 1.26-1.45x faster with slightly different output.

Two-machine mode prefills on the RTX and hands one finished state to the Mac over a direct TCP stream. The wire is BF16, 0.66 GB at 200K. Follow-ups under 11,776 new tokens prefill locally on the Mac.

## Verification posture

This is the most distinctive thing about the repo. Evidence is hash-bound rather than anecdotal. Qualification JSONs pin `model_sha256`, `fixture_sha256`, per-stage `producer_sha256` and `decode_sha256`, and `logits_sha256`. Cross-machine KL is reported as mean 0.028 and max 0.054 nats against a 0.1 mean and 0.5 max limit.

`scripts/release_check.py` is a privacy gate wired into `.githooks/pre-commit` and `commit-msg`. It scans staged bytes and commit messages for private IP ranges, credential patterns, machine paths, and the owner's real name. `release-artifacts.json` lists SHA-256 of every proof screenshot with a manual review note saying the PNGs contain no text or EXIF metadata.

## Findings worth flagging

Nothing runs here. The RTX path is Windows-only, targets compute capability 12.0, and needs Visual Studio Build Tools, CUDA 13.2, CMake, Ninja, Node 24 and Python 3.12. With no CI and no releases, the 202 test files are executed only by the author.

The official safetensors conversion path is not qualified. The README says so directly: "this native safetensors path has not yet been run end to end and qualified on this stack." The measured FP4 file was requantised from Q4, and ROADMAP lists native-precision conversion as open work.

Three forked copies of ds4 exist. `ds4_distributed.c` is 8,437 lines and byte-identical across win, mac and peer. `ds4.c` diverges by 17k lines between win and mac and 27k between win and peer. There is no shared core, so a fix has to be applied three times.

Provenance links dangle. `engines/ds4-mac/THIRD_PARTY-V41.md` points at `docs/DSPARK-V41.md`, `docs/REPRODUCE-V41.md` and `bench/SPEC-BENCH-V41.md`, none of which exist in the published tree. The Mac engine's reproduction instructions are missing. The vendored llama.cpp READMEs also link to pruned upstream docs, which is harmless.

Generated docs are stale. `python3 -B scripts/release_check.py` exits 1 with `docs/VARIABLES.md:0: regenerate with python -B -m slstack.envdocs`. The 103k-line variable table does not match the current engines even after the commit titled "Refresh the environment table source references".

LAN exposure is the default. The Mac receiver binds `0.0.0.0` through `mac_bind`, and the RTX broker binds `0.0.0.0` unless pairing narrows it to the local IP. The gate is a `secrets.token_urlsafe(32)` bearer token checked with `hmac.compare_digest` and stored at mode 0600, and it is printed to stdout when the receiver starts.

Caveats the README states itself: TileLang attention is numerically different from native attention and later continuations can diverge; Max speed introduced syntax errors in C++ tests; the host-RAM expert tier is a net loss on the 32 GB machine; the 1M result and warm-decode controls come from older builds; graph speedup is not established.

## Linux: where it stands today

Two halves, and they are in very different states.

The decoder is already Linux-capable. `engines/ds4-mac/Makefile` has a non-Darwin branch that builds CUDA, mapping `CUDA_ARCH=sm_120` to `-gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1`. It also has `rocm/*.cuh` targets and DGX Spark targets. `ds4_engram.c` uses plain `pread`, and `ds4_ssd.c` has no Win32 references.

The streaming prefill producer is hard-gated to Windows. It will not configure, let alone run.

### Blockers, with evidence

| blocker | location |
|---|---|
| `if(NOT WIN32 OR NOT GGML_CUDA) message(FATAL_ERROR "The offline V4.1 producer requires Windows and GGML_CUDA")` | `engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill/CMakeLists.txt:3` |
| `throw std::runtime_error("V4.1 physical weight provider requires Windows CUDA/DirectStorage")` | `src/llama-dsv41-weights.cpp:923` for inspect, `:1000` for create_pipeline |
| Throws `"Early Engram IoRing requires Windows"` and `"Whole-prompt Engram IoRing requires Windows"` | `src/llama-dsv41-engram-prefetch.cpp:1164, 1283, 1323`, 108 Win32 references in that file |
| Win32 file lease and volume identity: `CreateFileW`, `GetDriveTypeW`, `QueryDosDeviceW`, `BY_HANDLE_FILE_INFORMATION`, `GetDiskFreeSpaceW` sector check, `_aligned_malloc` | `llama-dsv41-weights.cpp` lines 30 through 960, `llama-dsv41-carry.cpp:15-219` |
| Locked carry RAM through `VirtualAlloc(MEM_RESERVE\|MEM_COMMIT)` | `src/llama-dsv41-locked-ram.h` |
| DirectStorage plus D3D12 and CUDA interop | 3 files, about 32 call sites: `llama-weight-fetcher.cpp`, `llama-wf-gpu-ring.inl`, `tools/dsv41-prefill/carry-copy-bench.cpp` |
| Windows Job Objects for the child memory guard | `slstack/guard.py:80`, function `windows_job` |
| Locked shared-RAM checkpoint refuses non-NT | `slstack/combo_ram.py:86`, `raise RuntimeError('The one-RTX shared checkpoint requires Windows')` |
| Installer and backend gating assume Windows or Apple Silicon | `slstack/install.py:106`, `slstack/backend_setup.py:225` |
| Every launcher is PowerShell | `build.ps1`, `slstack.ps1`, `scripts/start-dsv41-single-*.ps1`, `engines/ds4-win/windows_build.ps1`, `flashinfer-sm120/build.ps1` |
| RTX decode kernels ship as `ds4-decode.dll` | `engines/ds4-win/` has 747 Win32 references |

### Already Linux-ready

Do not underestimate this part. The port is mostly substitution, not a rewrite.

- `bin/slstack` is a POSIX sh launcher. `slstack/generation.py` and `slstack/lease.py` already carry `fcntl` fallbacks beside the `msvcrt` paths. `memory_counters.py` falls back to `psutil` on non-Darwin. `nvidia-smi` works.
- `llama-weight-fetcher.h` already declares `WF_MODE_GDS`, described as requiring the GDS driver on Linux, and `WF_MODE_AUTO`, described as trying DSTORAGE then GDS then DIRECT_IO then MMAP. Mode selection is at `llama-weight-fetcher.cpp:4529-4546`.
- A real GDS read path exists at `llama-weight-fetcher.cpp:1617-1627` using `cuFileRead`, guarded by `__has_include(<cufile.h>)`.
- A Linux I/O path exists: threaded `pread` workers commented "Fallback: threaded pread workers for Linux (until io_uring is wired up)" at `:1205`. `struct io_uring ring` is declared at `:441` and `<liburing.h>` is included conditionally at `:38`, but never wired.
- `LLAMA_WF_MODE=auto|direct_io|mmap` is already read from the environment in `llama-weight-fetcher-adapter.cpp:36-52`.
- The roughly 70 `src/llama-dsv41-*.cpp` and `.h` prefill units are mostly platform-neutral, at 0 to 2 Win32 references each. The CUDA kernels compile under Linux nvcc.
- CUTLASS 3.9.2 is fetched by `git clone`, which works on Linux unchanged.

## Port work, by workstream

### Storage to GPU

DirectStorage does not exist on Linux, so replacement is mandatory. Three options:

1. GDS, through `cufile.h` and `nvidia-fs.ko`. The pinned driver documents XFS and EXT4 in ordered mode on NVMe, and its requirements list names Tesla or Quadro-class GPUs. Verify GeForce plus your kernel and driver before committing, because cuFile can report no GDS support and fall back to bounce buffers silently, which defeats the point.
2. io_uring with `IORING_OP_READ`, registered files and fixed buffers. This is the direct analogue of Windows IoRing and the safer bet on consumer hardware. Writing `wf_io_uring_worker` in place of `wf_fallback_worker` is the highest-value single change.
3. Keep the `pread` pool as the correctness baseline.

### Engram extents

Port `llama-dsv41-engram-prefetch.cpp` and `engines/ds4-win/ds4_engram_ioring.cpp` to io_uring. Direct I/O needs 4 KiB-aligned offsets and buffers, so `FILE_FLAG_NO_BUFFERING` becomes `O_DIRECT` plus `posix_memalign`, and the `GetDiskFreeSpaceW` sector check becomes `statfs` or `ioctl BLKSSZGET`.

### File identity and lease

The provider deliberately refuses symlinked or redirected volumes, requiring `QueryDosDeviceW` to start with `\Device\HarddiskVolume`. Linux equivalent: `openat`, `fstat`, `realpath`, verify `st_dev` matches the mount, refuse `S_ISLNK`, and pin ancestor directory fds. Same guarantee, different syscalls.

### Locked RAM

`VirtualAlloc` plus working-set locking becomes `mmap` plus `mlock`, which needs `RLIMIT_MEMLOCK` or a cgroup allowance, or `memfd_create`. `SetProcessWorkingSetSizeEx` in `combo_ram.py` has no Linux equivalent, so model it with cgroup v2 `memory.min` and `memory.max` through `systemd-run --scope`.

### Process guard

`windows_job` uses a Job Object with `JobMemoryLimit` and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The Linux equivalent is a cgroup v2 scope whose removal kills the descendants. `lease.py` byte-range locks already work through `flock`.

### Build

Write a shell twin of `build.ps1`. Drop `LLAMA_DSV41_DSTORAGE_INCLUDE_DIR` and `LLAMA_DSV41_DSTORAGE_LIBRARY`, and drop the `ole32 d3d12 dxgi bcrypt` link libraries. Relax the `NOT WIN32` fatal error. Keep `-DCMAKE_CUDA_ARCHITECTURES=120a`, `GGML_BACKEND_DL=OFF` and `BUILD_SHARED_LIBS=OFF`, because the producer binds its own executable SHA. The FlashInfer sm120 AOT exporter, `export_kernel.py` and `export_dense_fp8.py`, needs the CUTLASS DSL runtime installed on Linux.

### Decode engine

Two choices.

Use ds4-mac's CUDA build on Linux with the canonical `ggml-mxfp4` file that has no layout flag. That works today, but only ds4-win accepts `flashinfer-sm120-mxfp8-mxfp4-v1`, at `engines/ds4-win/ds4.c:3066` and `cuda/mmq/ds4_mmq_fp4_packed.cuh`. So you lose the packed tensor-core layout and its decode speed.

Port ds4-win into a Linux ds4 engine to keep the packed decode kernels and turn `ds4-decode.dll` into a `.so` loaded beside the producer. That is 747 Win32 references.

### Two-machine mode

A Linux decode host needs the mode3/YOCO state receiver, which the repo says is missing. `examples/dual-machine/dsv41-large-ram-untested/README.md` states plainly that the required CPU receiver is missing and that the Windows/Linux RAM-resident decode backend does not implement the mode3/YOCO receiver or broker integration. The RTX broker in `slstack/native_broker.py` binds `0.0.0.0` and authenticates with the pairing token, so the transport is fine. The receiver on the far side is not.

## Three ways to scope it

| option | what you get | work |
|---|---|---|
| Linux as decode host only, keep Windows RTX prefill | RTX streams prefill, Linux box decodes from RAM | Implement mode3/YOCO exact-cone v2 receiver in ds4. Medium, and it is the documented gap either way. |
| Linux plus RTX 5090, decode only | `make cuda CUDA_ARCH=sm_120` in `engines/ds4-mac`, canonical FP4 file | Small. Fastest thing that could plausibly work today. |
| Full Linux prefill port | streaming prefill on Linux | Large. io_uring, Engram extents, provider lease, locked RAM, build scripts, re-qualification. |

## Cheapest experiment first

Run these on a Linux box with an RTX 50-series before committing to the port.

1. `cd engines/ds4-mac && make cuda CUDA_ARCH=sm_120`. If it compiles, the decode half is real.
2. Build llama-server with `GGML_CUDA=ON` and `LLAMA_WF_MODE=direct_io`, then measure the `pread` worker pool against the Windows numbers. That tells you how much of the 4,632 tok/s figure comes from DirectStorage rather than generic streaming, and whether io_uring alone closes the gap.

Every README number is Windows-only, so any Linux result needs fresh qualification against the hash-pinned fixtures in `docs/*.json`. The existing numbers will not transfer.
