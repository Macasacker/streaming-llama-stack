# Plan B: Blackwell-only engine, and the DeepSeek V4.1 Flash file that is on disk

Not started. **Gated on the Linux port's legacy removal** (`docs/linux-port-plan.md`): the
remaining Win32 host code, the Metal comment sweep, the Python orchestrator, the vendor bloat and
the docs/preset prune. Nothing here needs a Windows or Mac path, so every chunk below is cheaper
after that work is gone, and several of them touch files the port still has to sweep.

Decisions taken on 2026-11-06, recorded so nobody re-litigates them:

- **Reuse, not reimplement (option A).** Take kernels and tools from Basalt and Strata - both MIT,
  both already handling this model - and keep this repo's architecture: GGUF tensor extents, the
  SHA receipt, the physical lease, the bounded compute banks, the carry store. **Not** `.basalt`:
  Basalt's pack repacks experts per layer into `[gate|up|down]` blobs
  (`tools/convert/model_sections.py`), and that container is incompatible with extent-offset
  identity, which is what the producer verifies.
- **Target the file on disk**, not a download: `qwen38-gyro/Strata/models/deepseek-v4.1-flash-IST-DAS-2.00bpw/`,
  four shards, 138 GB, `deepseek41`, 40 blocks, 384 experts, 6 active, mixed `IQ1_M`/`IQ2_XS`/
  `IQ2_XXS` experts and `Q5_K`/`BF16` dense tensors, Engram in a separate 110.6 GB file
  (`deepseek41.engram.file`). Basalt's credits name the IST DAS GSQ-RCO quants, so the family is
  supported upstream. NVFP4 is out of scope: it comes from an NVIDIA Model Optimizer safetensors
  checkpoint (`convert.py --checkpoint`), not from these GGUFs.
- **Both workloads are targets**: long-context prefill throughput and interactive decode tok/s. They
  stress different code - see the ceilings below - and every benchmark chunk records both.
- **Locked memory is option 3**: lock what the allowance permits, keep the tail allocated and
  pageable, report the shortfall, and let the host-memory paging guard enforce residency. This is
  what Basalt does (`build/basalt/src/platform/memory.cpp:100-113`: 256 MiB chunks,
  `r.ok = off > 0`, note `mlock failed (raise ulimit -l)`), so it is a precedent and not a guess.
- **Blackwell only**: `sm_120a`, one CUDA architecture, no other vendor backend. Basalt already
  enforces that shape (`CMakeLists.txt:4` sets arch 120, line 44 `FATAL_ERROR` on anything else).

## Measured on this box (2026-11-06; re-measure before trusting)

| Fact | Value | How |
|---|---|---|
| GPU link | PCIe 16.0 GT/s x16 (Gen5), max 32.0 GT/s | `/sys/bus/pci/devices/0000:01:00.0` |
| NVMe | Samsung 990 PRO 2TB, Gen4 x4, **2.6 GB/s** sequential direct | `dd iflag=direct bs=1M` over 3.2 GB |
| VRAM | 32 GiB; Strata holds ~30 GiB, browser ~230 MB | `nvidia-smi --query-compute-apps` |
| CUDA context | cannot be created at that occupancy: `cudaFree(0)` and `cudaMemGetInfo` return `out of memory` | two-call probe |
| RAM / swap | 93 GiB total, 38 GiB used, 9 GiB swap unused | `free -h` |
| `RLIMIT_MEMLOCK` | 8 MiB soft **and** hard; `setrlimit(∞)` → `EINVAL`; `systemd-run --user` cannot raise `LimitMEMLOCK` (property accepted, `ulimit -l` still 8192); no passwordless sudo | probes in `build/` |
| `mlock` behaviour | all-or-nothing per call: 12 MiB fails with `ENOMEM` and locks 0 bytes; `/proc/self/smaps` reports `Locked: kB` per VMA | `build/mlock-probe`, smaps probe |

The arithmetic that follows from it. Per expert: 3 matrices x 5120 x 2304 = 35.4M values. Per token:
6 experts = 212M values, which at this quant's ~2 bpw is **~53 MB of expert bytes per token**. The
model is 138 GB, RAM is 93 GB, VRAM is 32 GiB - so a subset of experts is resident somewhere and the
rest streams.

```
53 MB/token from NVMe at 2.6 GB/s      -> ~49 tok/s   (device-bound)
53 MB/token from host RAM over Gen5    -> ~940 tok/s  (link-bound, before compute)
```

So decode tok/s on this box is decided by the **resident policy and the overlap**, not by dequant
instruction counts. Long-context prefill is a different shape: at 262K tokens the expert traffic is
the same per token, but the Engram rows (48 rows/token x 144 B ≈ 6.9 KB/token) and one-time prompt
reads dominate differently, and the sweep/chunk schedule decides whether the link is ever idle.
That asymmetry is the whole reason to build A rather than run Basalt: Basalt's default context is
8,192 and it is a serve engine.

## Prerequisite: the legacy work, in the order that unblocks this

| Port chunk | What it removes | Needed before |
|---|---|---|
| 3.1h | Win32 arms of `llama-dsv41-carry.cpp` and `llama-dsv41-locked-ram.h`, and the all-or-nothing `mlock` | B.5, B.6 (both allocate through that tier) |
| 4.1 | ggml's `blas`, `cann`, `hexagon`, `hip`, `metal`, `musa`, `opencl`, `rpc`, `sycl`, `vulkan`, `zendnn` | B.7 |
| 5.1 | `guard.py` Job Object → cgroup scope, `single_runtime.py` PRODUCER_FILES | the launch recipes B.8 needs |
| 2.2e, 6.1, 7.1 | Metal words in comments, Mac/MLX vendor and engines, docs/presets/tests | B.8's receipts and the README tables |
| `mtmd`, `server`, `decode-receiver` Win32 | their Windows host code | only if those binaries stay in scope |

## The chunks

**B.1 - assemble the shards, losslessly.** No new engine code. `gguf-split --merge` source exists in
Strata's vendored llama.cpp (`third_party/llama.cpp/tools/gguf-split/gguf-split.cpp`, MIT); our
vendored tree has no `gguf-split` tool directory. Decide: vendor that one tool into
`engines/streaming-llama/prefill_llama.cpp/tools/`, or write a Python merger on
`slstack/gguf_header.py` (Basalt ships `tools/gguf_reader.py`/`gguf_writer.py` as the pattern).
Verify with `read_header`: `split.count == 1`, tensor count 1008, byte-for-byte identity of every
tensor payload against the shards, and the SHA-256 of the merged file recorded as a receipt. Disk:
138 GB output, 336 GB free. **Host-only.**

**B.2 - settle the Engram row geometry before anything reads it.** The file's own arithmetic says
144 bytes per 256-value row (110,595,290,816 total over 768,022,850 rows, and the second table's
offset is exactly 144 x the first table's rows), and Basalt asserts
`row_values=256, row_bytes=144` (`tools/convert/test/test_convert_deepseek41.py:96`). But the header
type id is 20 (`IQ4_NL`), whose stride for 256 values is 68 bytes. 144 is eight 32-value Q4_0 blocks
(4.5 bpw). Read Basalt's `src/ngram/ple_reader.cpp` (`ROW_BYTES`) and Strata's
`src/dsv41/ds4_engram_io.hpp` (`n` row bytes plus `n2` scale bytes, "210 a 256 values, 18 a 32")
and pin one geometry with a fixture that decodes real rows from the file and compares against a
reference dequant. Getting this wrong reads garbage Engram contributions silently. **Host-only.**

**B.3 - relax the accepted-type rule.** `slstack/gguf_header.py:validate_dsv41_types` currently
requires `(Q4_K,Q4_K,Q4_K)`, `(IQ2_XXS,IQ2_XXS,Q2_K)` or `(MXFP4,MXFP4,MXFP4)` per block, dense
`Q8_0`/`Q4_K`/`Q4_0`, `engram_embd` `I8`, `token_embd` `F16`, rest `F32`. Add `IQ1_M` (29) and
`IQ2_XS` (17) to the expert set, `Q5_K`/`BF16` to the dense set where the engine has kernels, and
the Engram row type from B.2 - each addition gated on B.5/B.6 existing, in the same commit, so the
launcher never admits a file the engine cannot run. Extend the Python tests in `tests/` with the
real header. **Host-only.**

**B.4 - the locked-memory chunk (this is port 3.1h).** Delete the Windows arms with
`scripts/port/collapse_conditionals.py --tokens _WIN32,windows.h,Win32,VirtualLock,VirtualAlloc,VirtualFree,BY_HANDLE_FILE_INFORMATION --never-defined _WIN32`
(verified: 174 deletions, 0 insertions, 0 blocked, both files then free of `_WIN32`). Replace the
single `mlock` with chunked best-effort locking and track `locked_bytes` beside `bytes`; keep
`state().bytes` as the admission accounting, because the carry tests read it; make suspend/resume
use `munlock` + `madvise(MADV_DONTNEED)` and re-lock on resume; keep `mmap` failure and admission
refusal throwing. Basalt's precedent for partial success is quoted above.
Gates, both host-only and both already runnable: `tests/dsv41_carry_release.cpp` **fails today**
with `Carry RAM cannot be locked` after ~6 MiB (measured: allocations of 2.95 MB x 3 against the
8 MiB allowance), and goes green; `tests/pinned_carry_memory.cpp` currently prints
`SKIP: Windows working-set lock proof` - port it to `/proc/self/smaps` `Locked:` per VMA plus
`mincore` residency, keep its portable assertions (zeroed, writable, refusal on
`allocate(bytes, bytes-1)`, control-allocation failure leaving `state().bytes == 0`, accounting
after two allocations and one release). Note the control-allocation path is a real bug on the
current code: if the `shared_ptr` control block throws, nothing releases the mapping or the
accounting, so `allocate` must clean up and rethrow. A partial 3.1h diff implementing exactly this
is saved as `build/31h-partial.patch` (scratch, regenerable); it made both tests pass and left the
build clean.

**B.5 - `IQ1_M` and `IQ2_XS` into the engine.** Add `DS4_TENSOR_IQ2_XS = 17` and
`DS4_TENSOR_IQ1_M = 29` beside the enum at `ds4.c:2228-2243` (ids already match Strata's
`src/dsv41/ds4_iq.cuh`), their block bytes in `routed_expert_block_bytes` (`ds4.c:4975-4987`, which
today `ds4_die`s on anything else), the row-block accounting, the rejection message path at
`ds4.c:5890`, the expert-cache-by-type recognition at `ds4.c:8820`, and the gate/up/down dispatch at
`ds4.c:12083-12227`. Port the dots from Basalt `src/kernels/cuda/iq_kernels.cu` - `Split<16>`,
`Split<17>`, `Split<29>`, all against a q8_1 activation, plus the down-format paths - and their CPU
twins (`src/kernels/cpu/iq_avx2_rows.inc`, `iq_avx512.cpp`) if `ggml-cpu` survives B.7. Our existing
shape to match is `dev_dot_iq2_xxs_q8_K_block*` at `ds4_cuda.cu:28762+`. Gate: a host-only parity
program comparing the new dequant/dot against llama.cpp's CPU reference for both types over a
deterministic fixture (Basalt's `tools/iq_fixture.py` is the pattern), then a CUDA parity on the
decode kernel when the GPU is free. Keep their file headers so provenance survives.

**B.6 - the Engram row loader with the real geometry.** `llama-dsv41-engram-prefetch.cpp` pins
`row_bytes=264` (256 E4M3 + 8 E8M0) in the implementation constants and in `read()`'s byte check;
the planner (`llama-dsv41-engram-plan.h`) already takes `row_bytes`, `cap` and `read_alignment` as
arguments, and `choose_scan`'s arena arithmetic scales with them. Make the layout carry the row
geometry (264 for the native form, B.2's value for the on-disk one), add the engine-side decode for
that representation next to `engram_decode_row`, and extend
`tests/dsv41_engram_early_io.cpp` - which builds a synthetic table file today - to run the second
layout too, both readers and both geometries. **Host-only.**

**B.7 - the Blackwell-only scope.** Delete ggml's eleven vendor backend directories (port 4.1), the
engine's `make cpu` and `make cuda-generic` targets, and the `__CUDA_ARCH__ >= 800` branches in
`ds4_qwen4_cuda.cuh`; adopt Basalt's contract - refuse any `CUDA_ARCH` that is not `sm_120a` rather
than building something else. Two consequences to decide first: the CUDA-optional evidence programs
(`docs/evidence/wf-extent-selftest.cpp` compiles the fetcher with no `-DGGML_USE_CUDA` on purpose,
because this box often has no VRAM) need `ggml-cpu` or they must be rewritten, and dropping the
non-`deepseek41` architectures (`deepseek4`, `glm-dsa`, `glm5-next`, `qwen4exp`, `clip`) is a feature
removal that also deletes engine shapes, presets and tests. **Host-only until it breaks a CUDA gate.**

**B.8 - measure both workloads, with Basalt as the bar.** Decode: tokens/s at 8K and 64K context
with the expert tier resident in host RAM and with it streaming from NVMe, reported against the
~49 and ~940 tok/s ceilings so a number is readable as "near the link" or "far from it". Prefill:
tokens/s and wall clock at 16K / 64K / 262K with Engram prefetch on and off. Basalt gives the
comparison: build it (`cmake -S build/basalt -B build/basalt-build -DCMAKE_CUDA_ARCHITECTURES=120`;
risk - their requirement is CUDA 13 + GCC 13+, this box has CUDA 13.4 with GCC 16, so pin
`CMAKE_CUDA_HOST_COMPILER` if nvcc rejects it), convert the shards to a pack, and run
`bin/basalt-bench` / `bin/basalt-probe`. Every CUDA run here needs Strata stopped, which also stops
this assistant: **the user runs B.8.** Record receipts the way Phase 8 does (logits SHA-256,
password recall) before rewriting `docs/PERFORMANCE.md`.

**B.9 - attribution.** `NOTICE` lists the antirez/ds4, NovaStar ds4-win and IngeniousIdiocy ds4-v41
ancestry today. Add Strata (`Niko1221/Strata`, MIT), Basalt (`jesdga95/basalt`, MIT) and Project
Maya (`mw00/project-maya` v1.1.17, MIT) with the specific files taken, and keep each ported file's
header comment. Basalt's `THIRD_PARTY_LICENSES` is the model for the wording.

## Open questions (decide before B.5)

1. Does "Blackwell only" also mean **`deepseek41` only**? If yes, B.7 deletes four other
   architectures' shapes, kernels, presets and tests - a large reduction, and it removes the
   `qwen4exp` path that the three Gyro files on disk need.
2. Is **`ggml-cpu` kept**? It is what the CUDA-optional gates compile against and what backs ops CUDA
   does not implement. Keeping it costs a little surface; deleting it costs the host-only evidence.
3. **Do we measure Basalt before we build?** B.8 before B.5 gives a bar to aim at; B.5 first gets a
   working engine sooner with no number to judge it against.
4. NVFP4 later, from an OM checkpoint, or never?

## Out of scope

`.basalt` as a container; multi-GPU and PCIe split; any Windows or macOS path; NVFP4 and
MTP/DSpark-drafting parity with Basalt beyond what this repo already carries.

## Scratch that exists now

`build/basalt` (shallow clone, 20 MB, MIT) and `build/31h-partial.patch` (the 3.1h collapse plus the
chunked-lock rewrite and the ported `pinned_carry_memory`, which made both carry tests pass). Both
are under `build/`, gitignored and not part of the tree; re-create them with the commands above
rather than trusting they survived - `/tmp` is cleared on this box and `build/` is scratch.
