# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan: `docs/linux-port-plan.md` (chunk backlog near the end). Repo review: `linux-port.md`.
Runnable commands for the user: `docs/linux-port-commands.md`.

## What the next session is for

Chunks 1.6, 2.1 and the Apple half of 2.2 are committed. Chunk 2.2 is **not** finished, and the
rest of it needs a decision from the owner rather than a mechanical pass: `engines/ds4` still
holds about 3,700 Metal references, and they are live CUDA code, not dead Mac branches - the graph
helpers are named `metal_graph_*`, the decode layer phases `METAL_DECODE_LAYER_*`, and roughly 200
environment variables are `DS4_METAL_*` while doing CUDA work. Renaming the identifiers is safe and
mechanical; renaming the env vars is a user-facing contract change (they are documented in
`docs/VARIABLES.md` and named in `docs/macdecode-r6/r6-controls.json` and `docs/MACDECODE_ROUND6.md`).
Pick one: rename to `DS4_*` and update the consumers, keep the names and narrow the chunk 2.2 gate
to code identifiers, or accept both spellings for a while. Everything verified so far is a compile
or host-code gate; nothing is a performance result, because no model weights are on disk.

## State

Commits, oldest first: `58ff877` (chunk 1: `engines/ds4` from the Windows engine), `be340bb`
(1.7: CUDA build passes on Linux), `3d36bd6` (1.10a: delete Windows/Mac/peer engine trees),
`6dc410f` (1.10b: one environment scope), `4abb73f` (1.10c: delete the Mac Python surface),
`bc439fe` (plan update), `5c32214` (handoff + command notes + nvcc shared-link fix),
`ebc5ba4` (1.10d: dead Mac/Windows branches in the launcher), `90b8605` (1.6: the decode library
builds and is gated without a GPU or a model file), `8c2ac02` (plan and handoff for 1.6),
`171319a` (2.1: Win32 host code deleted from `engines/ds4`), `b7c5906` (plan and handoff for 2.1),
`7cc2ca3` (2.2a: Apple conditional branches deleted), `3a49385` (2.1 follow-up: the claims about
what the deleted Windows paths leave behind).

`python3 -B scripts/release_check.py` reports 0 findings, so commits need no `--no-verify`.
Python suite: **584 tests, 5-6 failures, 5 errors** (fork baseline 585 tests, 6 failures, 6
errors). Two process tests vary run to run, so compare the failing names rather than the count.

## What chunk 2.2a landed (`7cc2ca3`)

3,475 more lines of engine source deleted with the same mechanical rule, applied to
`__APPLE__`/`TARGET_OS`/`__MACH__`. ds4.c carried the Mac graph layer behind `#if defined(__APPLE__)`,
so the Metal implementations went and the CUDA ones stayed. Three guards needed hand work because
a later branch was the live one: the n-gram read advisory (`#ifdef __APPLE__ / #elif
defined(POSIX_FADV_RANDOM)`), `ds4_gpu.h`'s `ds4_gpu_device_is_spark()` behind
`#elif !defined(__APPLE__)`, and two `#if defined(_WIN32) / #elif defined(__linux__)` pairs in
`ds4_agent.c`. 77 conditions in ds4.c kept their guard and lost only the Apple operand, because
they still test `DS4_NO_GPU` or `DS4_ROCM_BUILD`.

## What chunk 2.1 landed

4,088 lines of engine source deleted, no Linux code path changed. The rule was mechanical: for each
conditional mentioning `_WIN32`/`windows.h`/`Win32`/`DStorage`/`IoRing`, keep what Linux already
compiled and delete the rest. Largest sites: `ds4_cuda.cu` 2,242, `ds4.c` 748,
`ds4_combo_api.inc` 285, `ds4_expert_restore_async.cuh` 249, `ds4_agent.c` 224,
`ds4_cache_affinity_background.cuh` 179, `ds4_server.c` 71, `ds4_web.c` 48.

Deleted subsystems, each now a refusal that names the build rather than a silent success:
combo handoff (`ds4_combo_api.inc` keeps its stubs; `ds4_native_checkpoint.h` deleted with its
BCrypt hasher), expert-cache restore and snapshot, background affinity promotion, output LoRA.
`ds4_web.c`'s guard hid the real POSIX browser-tool implementation in its `#else` branch; that is
now unconditional and live.

Gate results: `make cuda CUDA_ARCH=sm_120 -j4`, `make cuda-decode`, the CPU objects, and
`test-batch-io` / `test-decode-library` / `test-cpu-fp4` / `./ds4 --help` all pass with the
pre-change warning profile. `gcc -E -fdirectives-only -P` before vs after is identical per changed
`.c` file except four refusal strings; `nvcc -E` on `ds4_cuda.cu` differs only in three strings and
one `__LINE__` in a HOST_MEMORY trace. `envvars.json` and `docs/VARIABLES.md` regenerated; the
variable name set is unchanged.

## What chunk 1.6 landed (`90b8605`)

`make -C engines/ds4 cuda-decode CUDA_ARCH=sm_120` now builds `ds4-decode.so` (59 MB, 19
`ds4x_*` exports). Changes:

- `Makefile`: `-fPIC` in `CFLAGS` and `-Xcompiler -fPIC` in `NVCCFLAGS` (a non-PIC object cannot
  go into a shared library: `ds4.o` failed with `relocation R_X86_64_TPOFF32 against
  g_parallel_depth`). New objects compiled a second time with `-DDS4_EXTERNAL_DECODE=1`:
  `ds4_ext.o`, `ds4_cuda_ext.o`, `ds4_host_memory_ext.o` (the last one is what suppresses the
  host-memory watchdog in the decode library). `ds4-decode.so` also needs
  `ds4_engram.o ds4_engram_iouring.o ds4_prompt_prefix.o ds4_cpu_fp4.o` - without them `dlopen`
  fails with `undefined symbol: ds4_engram_hash`, then `ds4_cpu_fp4_init`.
  New targets: `test-batch-io`, `test-decode-library`, `ds4_cpu_fp4.o`, and `test-cpu-fp4` now
  builds with `-march=native`.
- `ds4_batch_io.cpp`: the CUDA upload stream is created lazily (`upload_stream_ensure`) instead
  of at `ds4_ds_init_from_fd`, so a host-destination transport never creates a CUDA context.
  `batch_add`/`batch_add_gpu` now refuse at `logical_capacity()` (ring capacity / read_parts),
  which is what `ds4_ds_batch_capacity()` reports - they previously accepted up to the raw ring
  capacity. `ds4_batch_io.h` comment corrected: `batch_count` counts pending logical spans.
- `ds4.c:4366`: `_InterlockedIncrement` → `__atomic_add_fetch` (MSVC intrinsic in the
  `DS4_EXTERNAL_DECODE` verify lane).
- `ds4_export.h` (new): `DS4X_EXPORT` as `__attribute__((visibility("default")))`. Needed because
  `ds4_cuda.cu` does not include `ds4_gpu.h`. `ds4_gpu.h` includes it, `ds4_cuda.cu` includes it
  directly, `ds4_decode_adapter.inc` and the two external `.inc` files now use `DS4X_EXPORT`.
- `ds4_decode_adapter.inc`: `ds4x_open` now preflights the GGUF header (`ds4x_gguf_preflight`) and
  refuses with an error message instead of loading the file. Without it the shared loader's
  `ds4_die` called `exit(1)` on a wrong model path, which would kill the producer rather than
  report a refusal. A truncated or corrupt GGUF still dies in the loader, as it does standalone.
- `ds4_external_host_cache.inc`: `VirtualLock`/`VirtualUnlock` → `mlock`/`munlock`,
  `GetLastError` → `errno`. `mlock` needs `RLIMIT_MEMLOCK` or a cgroup allowance sized to the
  pinned tier - that decision is still open (plan's "locked memory" risk).
- `ds4_cuda.cu`: `#include <sys/mman.h>` and `ds4_export.h`.
- `ds4_cpu_fp4.cpp`: `<intrin.h>`/`<windows.h>` → `<immintrin.h>`/`<cpuid.h>`/`<sys/resource.h>`,
  `SetThreadPriority(GetCurrentThread(), ...)` → `setpriority(PRIO_PROCESS, 0, -1)`,
  `__cpuidex(r,7,0)` → `__cpuid_count(7,0,a,b,c,d)`, `__cpuid(r,1)` → `__cpuid(1,a,b,c,d)`
  (GCC's macro takes the level first).
- New harnesses: `tests/test_batch_io.cpp` and `tests/test_decode_library.cpp`.

## Gate results (all re-run, all green)

- `make -C engines/ds4 cuda CUDA_ARCH=sm_120 -j4` builds the executables.
- `make -C engines/ds4 ds4-decode.so CUDA_ARCH=sm_120` links; `nm -D ds4-decode.so | grep -c ds4x_`
  reports 19 entry points.
- `make -C engines/ds4 test-batch-io` PASS: 18 `ds4_ds_selftest` combos (chunk 1/4/8 MiB x batch
  1/8/16, `read_parts` 1 and 8) matched `pread`, plus refusal checks, the split submit/wait
  pipeline, and the background ring. GPU memory used did not change, which is the point of the lazy
  upload stream: it runs while Strata holds the GPU. The target creates `/tmp/ds4-batch-io.bin` if
  missing; running `./tests/test_batch_io` by hand without that file exits 2.
- `make -C engines/ds4 test-decode-library` PASS: `dlopen` binds all 19 `ds4x_*` entry points,
  `ds4x_host_cpu_v1` accepts the caller's dot kernel, `ds4x_open` refuses a non-GGUF file with
  "ds4x_open model file is not a GGUF file", and `ds4x_model_map` / `ds4x_close` are safe with
  nothing open. No GPU, no model.
- `make -C engines/ds4 test-cpu-fp4` PASS (AVX-512 CPU FP4 lane on host).
- `python3 -B scripts/release_check.py` reports 0 findings.

## Two hard constraints

1. Strata (`qwen38-gyro/Strata/build/strata --serve --native
   models/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf ...`, pid 1020467) holds ~29.2 GiB of 32 GiB VRAM,
   leaving ~1.29 GiB free. Stopping Strata also stops this assistant, so GPU work must be run by
   the user by hand. Never kill Strata to test. Host-only gates are now available thanks to the
   lazy upload stream.
2. No supported model file exists on disk. The large GGUFs are
   `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ1_0/TQ2_0/TQ3_0.gguf` (54.5 / 92 /
   108 GiB). The metadata matches `DS4_SHAPE_QWEN4_EXP`, but they are TQ-quantized
   (`type 145/146`), outside ds4's tensor type table (`ds4.c:2481-2496`), so the loader rejects
   them.

## Next chunks, in order

From `docs/linux-port-plan.md`: 2.2 (Apple/Metal inside `engines/ds4`, 4,100 refs left, 3,959 of
them in `ds4.c`; gate `grep -rn "__APPLE__\|TARGET_OS\|Metal\|metal" engines/ds4 --include=<source
extensions>` returns nothing and `make cuda` still builds), then 3.1 (Linux producer), 3.2 (producer
`dlopen` of `ds4-decode.so`), 4.1, 5.1, 6.1, 7.1, 8.1.

Chunk 2.1 landed: every conditional whose expression named Win32 now compiles to what Linux already
compiled, 4,088 lines of engine source deleted, and `engines/ds4` is clean of the gate tokens
except the two vendored files (`external/nlohmann/json.hpp`, `linenoise.c`), which were left alone
on purpose. Four capabilities the Windows branches used to implement are gone: combo handoff,
background affinity promotion and output LoRA now refuse with a message naming the build, while
expert-cache restore and snapshot are inert - nothing on Linux calls
`ds4_expert_cache_restore_prepare`/`ds4_expert_cache_snapshot_write`, so the two
`DS4_COMBO_EXPERT_*_FILE` variables `slstack` sets for a combo run do nothing. `ds4_web.c` gained
the opposite: its `#else` branch is a real POSIX browser-tool implementation, now unconditional.

Chunk 1.6's producer side is NOT wired here: nothing in `engines/streaming-llama` loads
`ds4-decode.so` today - `grep -rn ds4x_ engines/streaming-llama` finds only
`llama-dsv41-ds4x-keep.h`. The producer decodes through its own ggml CUDA path
(`ggml-ds4-native-half`). The `LoadLibraryW`/`GetProcAddress` sites left in the producer are
`llama-dsv41-engram-prefetch.cpp:95-104` (IoRing), `llama-mmap.cpp:532`, and
`ggml/src/ggml-backend-dl.cpp` - all chunk 3.1/4.1 work, not 1.6.

## Behaviour changes worth re-checking

- `slstack/engines.py:combo_enabled()` no longer requires `os.name == 'nt'`, so `serving=native`
  now reaches the combo lifecycle on Linux. Its process admission is `slstack/guard.py`'s Windows
  Job Object branch, which chunk 5.1 replaces with a cgroup v2 scope. If native serving misbehaves
  on Linux, that is the likely cause, not the removal itself.
- `slstack/lease.py:active_holds()` dropped the `mac=` namespace; MAC hold lines are no longer
  scanned. `tests/test_hold_yield.py` was rewritten for RTX-only semantics.
- `slstack/router.py` no longer annotates routes: rows say `single-engine` or `ds4-native`, and
  `ultra_handoff` is always false.
- Still Windows-shaped and untouched: `slstack/single_runtime.py` stages `ds4-server.exe` and
  `slstack-weight-fetcher.dll`, and `scripts/build_combo_runtime.py` builds a preset named
  "RTX: ULTRA prefill → ds4-win decode" with `mac={}`. Those belong to 5.1/7.1.

## Gotchas learned this session

- Proof of a behavior-preserving dead-branch deletion: run `gcc -E -fdirectives-only -P` (or
  `nvcc -E`, then strip the `# N "file"` markers) on the pre-change file from `git show HEAD:<path>`
  and on the changed file, and diff. Comments and dead branches vanish, so the diff shows only what
  actually changed in the compiled code. For chunk 2.1 that diff was four refusal strings and one
  `__LINE__`.
- The strip was done by a script that builds the directive tree, evaluates each Windows-named
  condition in three-valued logic (`_WIN32` false, other macros unknown), and deletes only what is
  definitely dead or definitely live; anything unknown it reported and left alone. Two sites needed
  hand work: a `#if _WIN32 / #elif defined(POSIX_MADV_DONTNEED)` chain (keep the `#elif` as the new
  `#if`) and `#ifdef __APPLE__ / #elif !defined(_WIN32)`. Reuse that shape for 2.2 with
  `__APPLE__` false.
- `envvars.json` churns heavily on a deletion chunk: every row records a source line number, so a
  4,000-line deletion rewrites thousands of rows. Check the variable *names* are unchanged
  (`git show HEAD:envvars.json` vs the new file) rather than reading the diff.
- nvcc rejects a bare `-fPIC`; use `-Xcompiler -fPIC`. A shared library also needs every linked
  object compiled with `-fPIC` when it contains TLS (`__thread`).
- `make -B` on a target rebuilds its prerequisites too, including the slow mmq objects. Use
  `make -B <target>` only when the prerequisite list changed, and give it a long timeout.
- `envvars.py:inventory()` treats `setenv("NAME")` in a source file as a read, so a test that sets
  an engine variable creates an undocumented table row. The batch-io harness no longer sets
  `DS4_DS_QUEUE_CAPACITY`; it uses the built-in default.
- Regenerating the table: `python3 -B -m slstack.envdocs --refresh-defaults` then
  `python3 -B -m slstack.envdocs`. Any line-number shift in an engine source makes
  `test_environment_contract` error with `Environment table out of date: stale=[...]` until both
  commands have run, so regenerate in the same commit as the source edit. It writes
  `envvars.json` with indent 2 - a hand-rolled `json.dump(indent=1)` rewrite produces a 54k-line
  diff. `validate_table()` deliberately keeps `engines/mlx-serve/` references that point at the
  deleted tree.
- `scripts/release_check.py` flags any absolute path under a user's home directory as a
  "machine path" in docs; docs must use repo-relative paths.
- `make cpu` overwrites the CUDA binaries with CPU variants of the same names. Rebuild CUDA after.
- `make test` still cannot run: the Makefile asks for 10 test sources, only 8 are published.
- CUDA facts: `nvcc` at `/opt/cuda/bin/nvcc`, libs in `/opt/cuda/lib` and `/opt/cuda/lib64`,
  `-luring` required, `sm_120` maps to `-gencode arch=compute_120a,code=sm_120a
  -DDS4_CUDA_HAVE_MXF4=1`.

## Suggested skills

- `add-unit-test` for extending the batch-io or decode-library harnesses.
- `adversarial-code-review` before declaring 2.2 done, but hand the reviewer the equivalence
  command as evidence: `gcc -E -fdirectives-only -P` on `git show HEAD:<path>` and on the working
  file, then diff. The review of 2.1 produced three findings, all quoting text that is not in the
  working tree, so a deletion chunk needs that proof alongside the reviewer.
- `web-research` for liburing or `sm_120a` details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when the next session runs out of context.
