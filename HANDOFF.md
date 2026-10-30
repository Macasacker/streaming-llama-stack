# Handoff: Linux-only port of streaming-llama-stack

Working dir: this checkout (`streaming-llama-stack`), branch `linux-only`.
Durable plan: `docs/linux-port-plan.md` (chunk backlog near the end). Repo review: `linux-port.md`.
Runnable commands for the user: `docs/linux-port-commands.md`.

## What the next session is for

Continue Phase 1 and start Phase 2 of the Linux-only port. The engine compiles; it has never run
against model weights, so nothing here is a performance result.

## State

Commits, oldest first: `58ff877` (chunk 1: `engines/ds4` from the Windows engine), `be340bb`
(chunk 1.7: CUDA build passes on Linux), `3d36bd6` (1.10a: delete Windows/Mac/peer engine trees),
`6dc410f` (1.10b: one environment scope), `4abb73f` (1.10c: delete the Mac Python surface),
`bc439fe` (plan update). `python3 -B scripts/release_check.py` reports 0 findings, so commits no
longer need `--no-verify` (`.githooks/pre-commit` runs `release_check.py --staged`).

Python suite: 585 tests, 6 failures, 5 errors. Fork baseline was 8 failures / 6 errors over 683
tests. Nothing regressed; the survivors are Windows-only guard paths, the missing `cosmos_windows`
vendor module, and GPU/video gates that need a model on disk.

## Uncommitted right now

- `engines/ds4/Makefile`: line 95 changed to `-Xcompiler -fPIC` for the `ds4-decode.so` nvcc link
  (nvcc rejects a bare `-fPIC`). That fixed the compile step.
- `docs/linux-port-commands.md`: new, untracked.
- `engines/ds4/tests/test_kv_payload_roundtrip`: built binary, untracked. Add `engines/ds4/tests/*`
  binaries to `.gitignore` if you keep it.

**Open failure:** `make -C engines/ds4 cuda-decode CUDA_ARCH=sm_120` now fails at the link step
(`collect2: error: ld returned 1 exit status`). My two attempts to capture the diagnostics were
interrupted, so the undefined symbols are still unknown. Reproduce with:

```bash
cd engines/ds4
timeout 900 make -B ds4-decode.so CUDA_ARCH=sm_120 > /tmp/decode-build.log 2>&1
grep "undefined reference" /tmp/decode-build.log | sed 's/.*undefined reference to //' | sort | uniq -c | sort -rn
```

Chunk 1.6 also needs the producer side: `dlopen`/`dlsym` instead of `LoadLibraryW`/`GetProcAddress`
in `engines/streaming-llama`, and `DS4_EXTERNAL_DECODE` (`ds4.c:3053`, `:4263`, `:4361`, `:43393`,
`:43772`) plus the `ds4x_*` entry points in `ds4_decode_adapter.inc`.

## Two hard constraints

1. Strata (`qwen38-gyro/Strata/build/strata --serve ...`, pid 600295) holds ~30.9 GiB of 32 GiB
   VRAM, leaving ~1.3 GiB free. Stopping Strata also stops this assistant, so GPU work must be run
   by the user by hand. Never kill Strata to test.
2. No supported model file exists on disk. The only large GGUF is
   `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (54.5 GiB, `qwen4exp`
   metadata matching `DS4_SHAPE_QWEN4_EXP`), but it is TQ1_0-quantized (`type 145/146`), which is
   outside ds4's tensor type table (`ds4.c:2481-2496`), so the loader rejects it.

## Next chunks, in order

Taken from `docs/linux-port-plan.md`: 1.6 (`ds4-decode.so` + producer loader), 1.10d (dead
`cfg['mac']` branches at `slstack/engines.py:81,199,340` and `slstack/router.py:64-65,137-138,168,313,324`,
plus `os.name == 'nt'` at `engines.py:85`), 2.1 (Windows code inside `engines/ds4`, ~162 refs),
2.2 (Apple/Metal dead code inside `engines/ds4`, ~4,100 refs), 3.1 (Linux producer), 3.2, 4.1, 5.1,
6.1, 7.1, 8.1. Decisions 1-3 are settled: single machine only, DeepSeek V4.x on ds4 CUDA, ROCm
stays deleted.

## Highest-value test available without a GPU

`ds4_ds_selftest()` (`engines/ds4/ds4_batch_io.cpp:950`) byte-compares batched io_uring reads
against `pread`. Pure host code: no GPU, no model. Its only call site sits inside
`#if defined(_WIN32)` at `ds4_cuda.cu:9508`, so the Linux transport has never been exercised. Build
`tests/test_batch_io.c` against `ds4_batch_io.o ds4_host_memory.o` and run it on a throwaway 64 MiB
file, then re-wire the selftest for Linux as part of chunk 2.1.

## Gotchas learned this session

- `logical_tier` is declared in ~40 functions in `ds4_cuda.cu`. A string replace hits the wrong
  one (26804 vs 27829). Match by line number.
- `goto done` at `ds4_cuda.cu:27826` jumped over `const int logical_tier`. Fixed by declaring
  `int logical_tier = 0` above the checks.
- nvcc rejects `-fPIC`; use `-Xcompiler -fPIC`.
- Dropping an `envvars.json` row is safe only when nothing reads the variable, and
  `slstack/envvars.py:inventory()` scans `engines/` and `vendor/` only — rows owned by `slstack/`
  must survive. Regenerate with `python -B -m slstack.envdocs --refresh-defaults` then
  `python -B -m slstack.envdocs`.
- `make cpu` overwrites the CUDA binaries with CPU variants of the same names. Rebuild CUDA after.
- Three platform guards in `ds4_cuda.cu` wrapped platform-neutral code, not Windows code
  (5921-6699, nested 6032-6312, 12510-13542). Removing the guards was the fix; their internals
  call only the `ds4_ds_*` transport API.
- `make test` cannot run: the Makefile asks for 10 test sources that are not in the published tree.
- `ds4_cpu_fp4.cpp:31` includes MSVC `<intrin.h>`.
- CUDA facts: `nvcc` at `/opt/cuda/bin/nvcc`, `CUDA_HOME` derived from it, libs in both
  `/opt/cuda/lib` and `/opt/cuda/lib64`, `-luring` required. `sm_120` maps to
  `-gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1`.

## Suggested skills

- `add-unit-test` for the batch-io harness and any fixture generator.
- `adversarial-code-review` before declaring a chunk done, especially for 2.1 and 2.2 where large
  blocks are deleted.
- `web-research` if you need liburing or CUDA 13.4 `sm_120a` details.
- `unslop` / `writing-style` on docs and commit messages.
- `handoff` again when the next session runs out of context.
