# Linux port commands

Commands for this machine: Arch Linux, RTX 5090 (32 GiB), CUDA 13.4 at `/opt/cuda`,
`nvcc 13.4`, `liburing` headers present, Python 3.14.8. Repo: this checkout, branch `linux-only`.
Every command below is run from the repository root unless it says otherwise.

Two realities that shape every run:

- Strata (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~30.9 GiB of VRAM, leaving about
  1.3 GiB free. CUDA gates need it stopped, and stopping it also stops this assistant. So the CUDA
  commands below are for you to run by hand, not for the agent.
- There is no supported model file on this machine. The engine accepts only `deepseek41`,
  `deepseek4`, `glm-dsa`, `glm5-next`, `qwen4exp` (plus `clip` projectors). The local
  `qwen38-gyro/Strata/models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` is `qwen4exp` metadata but is
  TQ1_0-quantized, which ds4's tensor type table rejects.

## Builds

```bash
# CUDA engine, ~4 minutes. Builds ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent.
make -C engines/ds4 cuda CUDA_ARCH=sm_120 -j4

# CPU-only variants of the same binary names (rebuild CUDA afterwards if you need the CUDA ones).
make -C engines/ds4 cpu -j4
```

`make -C engines/ds4 cuda-decode CUDA_ARCH=sm_120` currently fails at the link step. An uncommitted
Makefile fix replaces `-fPIC` with `-Xcompiler -fPIC` for the nvcc shared link; the remaining link
error is not yet diagnosed. See `HANDOFF.md`.

## Gates that pass today

```bash
python3 -B scripts/release_check.py          # Release check: 0 findings
timeout 300 env PYTHONPATH=.:tests python3 -m unittest \
  test_release_check test_dsv41_single_launch test_example_launchers   # 32 tests, OK
timeout 600 env PYTHONPATH=.:tests python3 -m unittest discover -s tests -p 'test_*.py'
# 585 tests, 6 failures, 5 errors. Fork baseline was 8 failures / 6 errors over 683 tests.
# The survivors are Windows-only guard paths, the missing cosmos_windows vendor module,
# and GPU/video gates with no model on disk.

timeout 300 env PYTHONPATH=.:tests python3 -m slstack --help | head -20
cd engines/ds4 && ./ds4 --help | head -12     # CUDA binary, no model needed
```

## Gates that fail today, and why

```bash
make -C engines/ds4 test
# No rule to make target 'tests/test_layer_pack.c' - the published tree ships only 8 test
# sources, but the Makefile asks for 10. Prune the test target or restore the sources.

make -C engines/ds4 test-cpu-fp4
# ds4_cpu_fp4.cpp:31 includes <intrin.h> (MSVC). Needs <immintrin.h> on Linux.

make -C engines/ds4 test-kv-payload-roundtrip CUDA_ARCH=sm_120
# Builds and runs through engine open, prints HOST_MEMORY_ADMITTED / HOST_MEMORY_SNAPSHOT
# from the Linux /proc + cgroup code, then fails: cannot open model 'ds4flash.gguf'.
```

## Model-free transport test worth building first

`ds4_ds_selftest()` in `engines/ds4/ds4_batch_io.cpp:950` byte-compares batched io_uring reads
against `pread`. It is pure host code, so it needs no GPU and no model. Its only call site is
inside a `#if defined(_WIN32)` block at `ds4_cuda.cu:9508`, so nothing on Linux exercises the
transport yet. A `tests/test_batch_io.c` linking `ds4_batch_io.o ds4_host_memory.o` against a
throwaway 64 MiB file is the highest-value gate available while Strata is running:

```bash
dd if=/dev/urandom of=/tmp/io-bench.bin bs=1M count=64
# then, once the harness exists:
./engines/ds4/tests/test_batch_io /tmp/io-bench.bin 4,8,16 8,32,64
```

## CUDA gates (only after you stop Strata)

```bash
# Free VRAM first.
pkill -f 'Strata/build/strata' ; nvidia-smi --query-gpu=memory.free --format=csv,noheader

cd engines/ds4

# Metadata and tensor validation, no inference, no VRAM beyond context init.
./ds4 --inspect --model /path/to/supported.gguf --cuda

# Prove the io_uring batch path equals pread on a real file. Needs the Linux wiring
# (currently Windows-gated) plus a supported GGUF.
DS4_DS_SELFTEST=64 DS4_DS_SELFTEST_CHUNKS_MIB=1,4,8,16 DS4_DS_SELFTEST_BATCHES=8,32,64 ./ds4 --cuda --model /path/to/supported.gguf
```

## Engine scope, for reference

Accepted architectures (`engines/ds4/ds4.c:588-602`, validated at `ds4.c:7873-7890`):
`deepseek41`, `deepseek4`, `glm-dsa`, `glm5-next`, `qwen4exp`, `clip`. Shapes are hardcoded
(`DS4_SHAPE_*`, `ds4.c:660-950`), including a 9-layer `qwen4exp` mini shape used for smoke tests.
Tensor types accepted (`ds4.c:2481-2496`): F32, F16, Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, Q8_K, IQ2_XXS, I8, I32, BF16, MXFP4. Anything else dies during validation with the tensor
name and type. Decode kernels that actually work on CUDA are `deepseek41` Flash only — see
`docs/SUPPORTED_MODELS.md`.
