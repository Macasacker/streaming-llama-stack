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

`make -C engines/ds4 cuda-decode CUDA_ARCH=sm_120` builds `ds4-decode.so` (the producer's decode
library, compiled a second time with `-DDS4_EXTERNAL_DECODE=1`). It links `ds4_ext.o`,
`ds4_cuda_ext.o` and `ds4_host_memory_ext.o` rather than the engine's own objects.

### The streaming producer (chunk 3.1)

`build.sh llama` configures and builds it. The CUTLASS checkout it clones is pinned to
`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`, and `libcufile.so` from `/opt/cuda` is linked when
present (the GDS driver is not installed here, so `cuFileDriverOpen` fails at runtime and the
fetcher stays on its `pread` workers).

```bash
llama=engines/streaming-llama/prefill_llama.cpp
mkdir -p build/cutlass-v3.9.2 build/llama/bin
test -f build/cutlass-v3.9.2/include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h ||
  git clone --quiet --depth 1 --branch v3.9.2 https://github.com/NVIDIA/cutlass build/cutlass-v3.9.2
cmake -G Ninja -S "$llama" -B build/llama \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_DSV41_OFFLINE_PRODUCER=ON -DLLAMA_DSV41_NATIVE_HALF_PROFILES=ON \
  -DLLAMA_DSV41_JSON_INCLUDE_DIR="$llama/vendor" \
  -DGGML_CUDA_DSV41_SPLIT_F32=ON -DGGML_CUDA_CUTLASS_PATH="$PWD/build/cutlass-v3.9.2"
cmake --build build/llama -j4 -- -k 0     # -k 0: collect every failure in one pass
```

Use `build/` for scratch logs, not `/tmp`: this box cleared `/tmp` mid-session once and took a
build log and a configure directory with it.

## Gates that pass today

```bash
python3 -B scripts/release_check.py          # Release check: 0 findings
timeout 300 env PYTHONPATH=.:tests python3 -m unittest \
  test_release_check test_dsv41_single_launch test_example_launchers   # 32 tests, OK
timeout 600 env PYTHONPATH=.:tests python3 -m unittest discover -s tests -p 'test_*.py'
# 584 tests, 6 failures, 5 errors. Fork baseline was 6 failures / 6 errors over 585 tests.
# The survivors are Windows-only guard paths, the missing cosmos_windows vendor module,
# and GPU/video gates with no model on disk.

# io_uring expert-read transport against pread: no GPU, no model, runs while Strata is up.
make -C engines/ds4 test-batch-io            # needs /tmp/ds4-batch-io.bin (64 MiB); the target creates it

# dlopen ds4-decode.so and bind its ds4x_* entry points: no GPU, no model.
make -C engines/ds4 test-decode-library

# The engine has no Win32 host code left. Vendored files are excluded on purpose.
grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing" engines/ds4 \
  --include=*.c --include=*.cpp --include=*.h --include=*.hpp \
  --include=*.cu --include=*.cuh --include=*.inc
# 3 hits, all vendored: engines/ds4/external/nlohmann/json.hpp (2) and the linenoise.c comment

# No Metal-named identifiers or backend left in the engine. The DS4_METAL_* environment names are
# kept on purpose and cannot match: `_` is a word character, so \bMETAL_ does not match DS4_METAL_FOO.
grep -rn "\bmetal_\|\bMETAL_" engines/ds4 \
  --include=*.c --include=*.cpp --include=*.h --include=*.hpp \
  --include=*.cu --include=*.cuh --include=*.inc
# 0 hits (chunk 2.2d caught the names with a leading underscore, e.g. glm_metal_)
grep -rn "DS4_BACKEND_METAL" engines/ds4 \
  --include=*.c --include=*.h --include=*.cu --include=*.cuh --include=*.inc
# 0 hits (chunk 2.2c)

cd engines/ds4 && ./ds4 --help | head -12     # CUDA binary, no model needed
./ds4 --metal -m /nonexistent                 # ds4: unknown option: --metal
./ds4 --gpu-graph-test -m /nonexistent        # reaches engine open; the CUDA graph self-test

# The producer links and runs. llama-server initialises CUDA even with Strata holding the VRAM.
./build/llama/bin/llama-server --version      # prints "found 1 CUDA devices: ... compute capability 12.0"
./build/llama/bin/llama-dsv41-prefill --help  # prints the usage: chunk 3.1c ported main.cpp
./build/llama/bin/llama-dsv41-prefill --check-native-profiles
# PASS: delivered native-half and rounded expert profiles,4096-row resource policies; no GPU/model
./build/llama/bin/llama-dsv41-prefill         # exit 1; the abort line names the parse refusal
# No Win32 code left in the producer CLI directory (chunk 3.1d).
grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing" \
  engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill   # 0 hits
```

### The producer's platform layer, with no test target to run

The producer CLI has no cmake test target, so chunk 3.1c proved its ported
file/hash/identity/VRAM layer with a program compiled by hand. Its source is committed
at `docs/evidence/platform-selftest.cpp`; copy it into `build/` to run it. It needs only
the boundary-capture object the producer already builds and two small ggml libs - no
CUDA, no model, no GPU:

```bash
llama=engines/streaming-llama/prefill_llama.cpp
cp docs/evidence/platform-selftest.cpp build/
cp engines/ds4/ds4-decode.so build/          # the loader check resolves it beside the selftest
g++ -std=gnu++17 -O1 -I$llama/tools/dsv41-prefill -I$llama/src -I$llama/ggml/src \
    -I$llama/ggml/include -I$llama/vendor -o build/platform-selftest \
    build/platform-selftest.cpp \
    build/llama/tools/dsv41-prefill/CMakeFiles/llama-dsv41-prefill.dir/__/__/src/boundary-capture.cpp.o \
    build/llama/ggml/src/libggml-base.a build/llama/ggml/src/libggml-cpu.a -lpthread -ldl
./build/platform-selftest                    # 44 checks, "SELFTEST PASSED: 0 failure(s)"
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) \
  ./build/platform-selftest | grep nvml      # NVML total/used, matching nvidia-smi
```

It covers volume admission (ext4 passes, `/tmp` on tmpfs is refused), the exclusive-new
output, bounded writes, the read-only rehash, `read_small`, no-replace publication,
symlink refusal, ancestor pinning, the free-byte budget, `exclusive_writer`,
`/proc/self/exe`, `getpid`, `set_environment`, `ram_admission`, SIGINT latching, the
abort file, and `dlopen` of `ds4-decode.so` binding every `ds4x_*` entry point.

### Driving the real producer with no model

A throwaway 1 MiB file, a copy receipt naming it and four LE32 tokens take the CLI
through its whole front half - its own executable SHA, the receipt, the sweep/wire/cache/carry
plans, the output volume budget, host-memory admission, CUDA enumeration - and it then
refuses at the VRAM reserve. Reaching that check costs a CUDA context (~160 MB), which
does not disturb Strata's 30 GiB.

```bash
python3 - <<'SNIPPET'
import os,struct,hashlib,json
root=os.path.abspath("build/smoke"); os.makedirs(root+"/out",exist_ok=True)
model=root+"/model.gguf"
open(model,"wb").write(bytes(1<<20)); st=os.stat(model)
open(root+"/copy-receipt.json","w").write(json.dumps({"receipts":{"model.gguf":{
  "sha256":hashlib.sha256(open(model,"rb").read()).hexdigest(),
  "final_identity":{"dev":st.st_dev,"ino":st.st_ino,"size":st.st_size,"mtime_ns":st.st_mtime_ns}}}}))
with open(root+"/tokens.le32","wb") as f:
    for t in (1,2,3,4): f.write(struct.pack("<I",t))
SNIPPET
sha=$(sha256sum build/llama/bin/llama-dsv41-prefill | cut -d" " -f1)
./build/llama/bin/llama-dsv41-prefill \
  --model "$PWD/build/smoke/model.gguf" --copy-receipt "$PWD/build/smoke/copy-receipt.json" \
  --tokens-le32 "$PWD/build/smoke/tokens.le32" --cuda-device 0 --context 8192 --build-sha "$sha" \
  --provider-device-bytes 1073741824 --provider-host-bytes 1073741824 \
  --cache-bytes 4294967296 --graph-bytes 268435456 \
  --job-device-bytes 4294967296 --job-host-bytes 17179869184 --gpu-reserve-bytes 1073741824 \
  --carry "$PWD/build/smoke/out/carry.bin" --carry-file-bytes 4294967296 --carry-host-bytes 4294967296 \
  --carry-outstanding-bytes 0 --output-outstanding-bytes 0 \
  --payload "$PWD/build/smoke/out/payload.bin" --receipt "$PWD/build/smoke/out/receipt.json"
# exit 1: "Selected GPU free bytes do not exceed the explicit reserve".
# out/payload.bin stays 0 bytes and out/receipt.json.pending-<pid>-<ticks> holds the failed
# receipt, whose payload_actual_identity is that file's st_dev, st_ino and st_mtime.
```

## Gates that fail today, and why

```bash
make -C engines/ds4 test
# No rule to make target 'tests/test_layer_pack.c' - the published tree ships only 8 test
# sources, but the Makefile asks for 10. Prune the test target or restore the sources.

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
