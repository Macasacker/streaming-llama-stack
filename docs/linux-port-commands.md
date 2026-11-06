# Linux port commands

Commands for this machine: Arch Linux, RTX 5090 (32 GiB), CUDA 13.4 at `/opt/cuda`,
`nvcc 13.4`, `liburing` headers present, Python 3.14.8. Repo: this checkout, branch `linux-only`.
Every command below is run from the repository root unless it says otherwise.

Two realities that shape every run:

- Strata (`qwen38-gyro/Strata/build/strata --serve ...`) holds ~30.9 GiB of VRAM, leaving about
  1.3 GiB free. CUDA gates need it stopped, and stopping it also stops this assistant. So the CUDA
  commands below are for you to run by hand, not for the agent.
- There is no model file ds4 can load on this machine. The engine accepts only `deepseek41`,
  `deepseek4`, `glm-dsa`, `glm5-next`, `qwen4exp` (plus `clip` projectors), and
  `slstack/gguf_header.py:validate_dsv41_types` is stricter still: per block the expert triplet must
  be `(Q4_K,Q4_K,Q4_K)`, `(IQ2_XXS,IQ2_XXS,Q2_K)` or `(MXFP4,MXFP4,MXFP4)`, dense projections must
  be `Q8_0`/`Q4_K`/`Q4_0`, `engram_embd` must be `I8`, `token_embd` `F16`, everything else `F32`,
  and `split.count` must be 1. What is on disk fails that: the three
  `Qwen3.8-Flash-Next-Gyro-*.gguf` are `qwen4exp` with TQ tensor types (ids 144-149); the
  `deepseek-v4.1-flash-IST-DAS-2.00bpw/` shards are `deepseek41` but split four ways with
  `IQ1_M`/`IQ2_XS` experts and `Q5_K`/`BF16` dense tensors (assembling them would not qualify the
  quant set); `UD-Q8_K_XL` is 1 of 9 shards; and the Engram table is a separate `IQ4_NL` file
  (`deepseek41.engram.file`) instead of rows inside the model. Run `read_header` on any candidate
  before planning a gate around it.

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

# No Metal-named identifier or backend left in the engine. The word-boundary greps below are cheap
# but NOT sufficient: `_` is a word character, so \bMETAL_ cannot see DS4_METAL_FOO (an environment
# name, kept on purpose) and equally cannot see DS4_METAL_PROFILE_DECODE_STAGE (a macro, which chunk
# 2.2e's first run missed). The census classifies each mention by its position in the file, which is
# the only honest test, and --strict fails on anything the compiler sees.
python3 -B scripts/port/metal_word_census.py --strict
# IDENTIFIER 0, ENV_NAME 195, ENV_NAME_CITED 124, COMMENT 111 lines, STRING 173 lines
python3 -B scripts/port/metal_word_census.py --strict --list   # names the tokens and the files
# --strict exit 1 means a code identifier names Metal again.
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
# The Engram row loader (its io_uring queues and the pread baseline): host only, no model.
./build/engram-early-io build/engram/table.bin    # 15 checks, "ALL PASS" - recipe below
# The provider's physical lease: no CUDA context, no model.
./build/provider-lease-selftest                    # 14 checks - recipe below
# The carry store's release/recommit cycle and the locked RAM tier: host only, no model.
./build/carry-release                              # recipe below
./build/pinned                                     # recipe below
# No Win32 code left in the producer CLI directory (chunk 3.1d), in the weight fetcher
# (chunk 3.1e), in the weight provider (chunk 3.1f), in the Engram row loader (chunk 3.1g) or in
# the carry store and its RAM tier (chunk 3.1h).
# Drop comment lines: the survivors are provenance comments naming the Win32 layer a file was
# ported from.
grep -rn "_WIN32\|windows\.h\|Win32\|DStorage\|IoRing\|IOCP" \
  engines/streaming-llama/prefill_llama.cpp/tools/dsv41-prefill \
  engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.cpp \
  engines/streaming-llama/prefill_llama.cpp/src/llama-weight-fetcher.h \
  engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-weights.cpp \
  engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-engram-prefetch.cpp \
  engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-engram-prefetch.h \
  engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-carry.cpp \
  engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-locked-ram.h | grep -vE ":[0-9]+:\s*//"
# 0 hits
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

### The weight fetcher's io_uring reader, against the pread baseline

Chunk 3.1e replaced the fetcher's `pread` worker pool with one io_uring ring per reader
thread. Its program is committed at `docs/evidence/wf-io-selftest.cpp`; it drives the real
`wf_*` pipeline over a 64 MiB throwaway file, byte-compares what each reader stages into the
GPU buffers against the file, checks that a span running past the file's end fails the fetcher
instead of reading zeros, and prints both readers at two queue geometries. It needs a CUDA
context (~160 MB) and a few MB of device buffers, so it runs while Strata holds the rest of the
VRAM. No model.

```bash
llama=engines/streaming-llama/prefill_llama.cpp
ninja -C build/llama src/libllama.a      # the object below is only rebuilt by this target
mkdir -p build && cp docs/evidence/wf-io-selftest.cpp build/
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -I/opt/cuda/include -o build/wf-io-selftest \
    build/wf-io-selftest.cpp \
    build/llama/src/CMakeFiles/llama.dir/llama-weight-fetcher.cpp.o \
    build/llama/ggml/src/libggml.a build/llama/ggml/src/libggml-cpu.a build/llama/ggml/src/libggml-base.a \
    build/llama/ggml/src/ggml-cuda/libggml-cuda.a \
    -L/opt/cuda/lib64 -lcufile -lcudart -lcublas -lcublasLt -lcuda -luring -lpthread -ldl
./build/wf-io-selftest            # 139 checks, "SELFTEST PASSED: 0 failure(s)"
```

Build the object first: `ninja -C build/llama llama-dsv41-prefill` compiles its own copy of the
fetcher under `tools/`, and the link above reads the one under `src/`, so skipping that step links
a stale object and the run says nothing about your edit.

What it prints here, whole-pass wall clock, best of three cold rounds:

```text
info 4 threads io_uring wall 19.6 ms / window sum 23.9 ms | pread wall 19.7 ms / window sum 32.6 ms | wall x1.00, window x1.37
info 1 thread  io_uring wall 21.4 ms / window sum 36.3 ms | pread wall 33.6 ms / window sum 58.8 ms | wall x1.57, window x1.62
```

Quote the wall clock. `total_nvme_read_ms` is the sum of each layer's own read window and layers
overlap, so it double-counts; four `pread` threads already saturate the device, and the ring's
actual advantage is queue depth per thread, which is the one-thread row.

`LLAMA_WF_IO_URING=0` is the only way to select the baseline; the fetcher otherwise uses io_uring
whenever `io_uring_queue_init` succeeds. Reads are buffered: `O_DIRECT` is not implemented, because
GGUF tensor offsets are multiples of `general.alignment` (32 bytes by default) and Linux would
reject the unaligned chunked reads with `EINVAL`.

### The strict extent reader, with no model and no VRAM

Chunk 3.1f gave the fetcher a reader for the provider's named extents (the
expert-cache pool and the next-tail hint). Its program is committed at
`docs/evidence/wf-extent-selftest.cpp`; it drives the real `wf_*` API over a
throwaway 32 MiB file, byte-compares what each extent landed at inside a bank,
and checks the refusals. It compiles the fetcher **without** `-DGGML_USE_CUDA`, so
the banks are host buffers and the upload is a `memcpy`: the same reader, bounds
checks, splitting, generation and accounting as the CUDA build, with no device
allocation and no model.

```bash
llama=engines/streaming-llama/prefill_llama.cpp
cp docs/evidence/wf-extent-selftest.cpp build/
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -I$llama/ggml/src \
    -c $llama/src/llama-weight-fetcher.cpp -o build/wf-extent-fetcher.o
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -o build/wf-extent-selftest \
    build/wf-extent-selftest.cpp build/wf-extent-fetcher.o \
    build/llama/ggml/src/libggml-base.a build/llama/ggml/src/libggml-cpu.a -lpthread -luring
./build/wf-extent-selftest    # 32 checks, "SELFTEST PASSED: 0 failure(s)"
```

The CUDA arms of that reader (pinned slots, the upload on the caller's stream, one
`cudaEvent` per ready batch) need VRAM headroom past the ~700 MB a `ggml` CUDA
backend costs, so they run when the GPU has it: link the same program against
`build/llama/src/CMakeFiles/llama.dir/llama-weight-fetcher.cpp.o` (rebuild it with
`ninja -C build/llama src/libllama.a` first) and the `libggml-cuda.a` set, as in
the io_uring section above.

### Does the io_uring wait flush the submission queue?

Both of the producer's io_uring readers queue reads and then wait for their
completions, and `io_uring_submit()` can return 0 without being an error. On this
kernel and liburing 2.15, `io_uring_wait_cqe()` does **not** flush entries the
kernel has not been told about, so a zero return would leave a layer's reads
unseen forever - the pipeline polls with a timeout, so it would read as a stall.
Both readers therefore wait through `io_uring_submit_and_wait()`. The probe that
established this is committed at `docs/evidence/uring-flush-probe.cpp`:

```bash
cp docs/evidence/uring-flush-probe.cpp build/
g++ -std=gnu++17 -O1 -o build/uring-flush-probe build/uring-flush-probe.cpp -luring
./build/uring-flush-probe    # 3 checks: a queued read with no submit has no
                             # completion; submit-and-wait delivers it
```

### The provider's physical lease, with no model and no CUDA context

Chunk 3.1f gave the provider a POSIX lease on the verified model file. Its program is committed at
`docs/evidence/provider-lease-selftest.cpp`; `inspect()` takes the lease before it asks anything
about the device, so the receipt binding, the single-link ordinary-file rule, the symlink refusals
and the pinned-ancestor checks all run with no backend and no VRAM. It links the CUDA objects but
never creates a context, so it runs while another process holds the GPU:

```bash
llama=engines/streaming-llama/prefill_llama.cpp
ninja -C build/llama src/libllama.a      # the four objects below are only rebuilt by this target
cp docs/evidence/provider-lease-selftest.cpp build/
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -I$llama/ggml/src -I$llama/vendor \
    -o build/provider-lease-selftest build/provider-lease-selftest.cpp \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-weights.cpp.o \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-engram.cpp.o \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-engram-prefetch.cpp.o \
    build/llama/src/CMakeFiles/llama.dir/llama-weight-fetcher.cpp.o \
    build/llama/ggml/src/libggml.a build/llama/ggml/src/libggml-base.a build/llama/ggml/src/libggml-cpu.a \
    build/llama/ggml/src/ggml-cuda/libggml-cuda.a \
    -L/opt/cuda/lib64 -lcufile -lcudart -lcublas -lcublasLt -lcuda -luring -lpthread -ldl
./build/provider-lease-selftest    # 14 checks, "SELFTEST PASSED: 0 failure(s)"
```

### The Engram row loader, with no model and no GPU

Chunk 3.1g replaced the loader's IoRing queues with io_uring and kept the serial `pread` as the
baseline. `tests/dsv41_engram_early_io.cpp` is its test - the Windows-only synthetic-table test for
exactly this file, ported with it. It generates a 208 MB deterministic table file under `build/`,
hashes prompts to row ids, and byte-compares every loaded row against the file and against the
single-queue reference loader. No CUDA, no model, no VRAM:

```bash
llama=engines/streaming-llama/prefill_llama.cpp
ninja -C build/llama src/libllama.a      # the two objects below are only rebuilt by this target
mkdir -p build/engram
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -I$llama/vendor \
    -o build/engram-early-io tests/dsv41_engram_early_io.cpp \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-engram-prefetch.cpp.o \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-engram.cpp.o \
    build/llama/ggml/src/libggml-base.a build/llama/ggml/src/libggml-cpu.a -lpthread -luring
./build/engram-early-io build/engram/table.bin                       # 15 checks, "ALL PASS"
LLAMADECODE_ENGRAM_IO_URING=0 ./build/engram-early-io build/engram/table.bin   # pread baseline
LLAMADECODE_ENGRAM_SCAN=0 ./build/engram-early-io build/engram/table.bin       # one read per row
LLAMADECODE_ENGRAM_SCAN=1 ./build/engram-early-io build/engram/table.bin       # forced scan geometry
LLAMADECODE_ENGRAM_READ_ALIGN=512 ./build/engram-early-io build/engram/table.bin
LLAMA_DSV41_ENGRAM_SUBMIT_BATCH=1 ./build/engram-early-io build/engram/table.bin
```

The env reads that choose a reader or a geometry are once per process, so each line above is a
separate run; the fixture is reused after the first. What the runs cover: per-table publication
order, the image-mask history resets across planner blocks, restored history tails, the tight
admission that shrinks the IO arena inside admission, previous-request row reuse, the row cache,
the refusal of a descriptor that is not the one the receipt hashed, and cancellation draining every
submitted read before it throws.

Both readers agree on this fixture because 208 MB sits in the page cache here - the single-queue
load of 131,781 row reads is 51-54 ms on the rings and 50-53 ms with `pread`, and the early
loader's 200 extents are 14-20 ms either way. That compares mechanics, not a device: the cold-cache
number against a real model is Phase 8 item 3.

### The carry RAM tier, and the carry store's release/recommit cycle

Chunk 3.1h made the carry tier a Linux `mlock` tier that locks in chunks instead of refusing the
whole allocation when `RLIMIT_MEMLOCK` is smaller than it (8 MiB for an unprivileged process here).
Neither program has a build rule; both need only the carry object, so build `src/libllama.a` first -
`ninja -C build/llama llama-dsv41-prefill` compiles its own copy of `llama-dsv41-carry.cpp` and does
**not** rebuild the `src/` one. No CUDA, no model, no VRAM.

```bash
llama=engines/streaming-llama/prefill_llama.cpp
ninja -C build/llama src/libllama.a
g++ -std=gnu++17 -O1 -I$llama/src -I$llama/ggml/include -I$llama/ggml/src -I$llama/vendor \
    -o build/carry-release tests/dsv41_carry_release.cpp \
    build/llama/src/CMakeFiles/llama.dir/llama-dsv41-carry.cpp.o \
    build/llama/ggml/src/libggml-base.a build/llama/ggml/src/libggml-cpu.a -lpthread
./build/carry-release     # sealed certificate, 40 byte-preserving layers, release/recommit/reseed,
                          # suspended destruction, incomplete refusal
```

The third of its three ~2.95 MB allocations is the one the all-or-nothing lock used to refuse; it now
reports `locked_bytes=2478080` and the run passes.

`tests/pinned_carry_memory.cpp` is the tier's own proof. It reads `/proc/self/smaps` for the VMA that
holds the mapping (no pathname means anonymous, `Locked: kB` is what the kernel holds) and `mincore`
for residency, so it needs no Windows and no GPU. Its second arm is the case the old code refused
outright: a tier larger than the allowance, whose VMA `mlock` splits at the lock boundary.

```bash
g++ -std=gnu++17 -O1 -o build/pinned tests/pinned_carry_memory.cpp
./build/pinned                                   # 2 MiB tier, plus a tier beyond the allowance
./build/pinned $((8*1024*1024))                  # exactly the allowance
./build/pinned 104857600                         # 100 MiB: locks 8 MiB, keeps 96 MiB pageable
```

What it asserts: zeroed and writable, private and anonymous, fully locked and resident inside the
allowance, partly locked with an untouched tail beyond it, release unmaps the region, the
`allocate(bytes, bytes-1)` refusal leaves no accounting, and an allocation whose accounting node or
`shared_ptr` control block throws leaves no mapping behind either - the smaps census of anonymous
mappings of that size is the same after the failure as before.

### Before any CUDA gate: is there headroom

Strata's serve process holds about 30 GiB of the 32 GiB and the browser GPU process a few hundred MB
more, so what is left for a context is roughly 120 MB - when it is left at all. It moves inside a
session: on 2026-11-06 `cudaMemGetInfo` reported 116 MiB free in one hour and `out of memory` an hour
later, with nothing in the tree changed. Run the probe committed at
`docs/evidence/cuda-headroom-probe.cpp` first; it allocates at most 8 MiB and never touches the
serve process.

```bash
g++ -std=gnu++17 -O1 -I/opt/cuda/include -o build/cuda-headroom-probe \
    docs/evidence/cuda-headroom-probe.cpp -L/opt/cuda/lib64 -lcudart
./build/cuda-headroom-probe; echo "exit=$?"
# exit 0: "cudaFree(0)=no error cudaMemGetInfo=no error free=116 MiB total=32193 MiB"
#         and cudaMalloc(8 MiB)=no error -> the CUDA gates here can run.
# exit 1: "out of memory" on all three -> every CUDA gate is unrun, not failing.
```

### Driving the real producer with no model

A throwaway 1 MiB file, a copy receipt naming it and four LE32 tokens take the CLI
through its whole front half - its own executable SHA, the receipt, the sweep/wire/cache/carry
plans, the output volume budget, host-memory admission, CUDA enumeration - and it then
refuses at the VRAM reserve. Reaching that check costs a CUDA context (~120 MB), and **it is a
producer gate this box can only run while another process leaves that much**: with Strata at
30 GiB and the browser beside it, a probe that only calls `cudaFree(0)` and `cudaMemGetInfo` can
return `out of memory`, and then this command aborts inside `ggml_backend_cuda_get_device_memory`
instead of printing the reserve refusal. Check that two-call probe first and free VRAM if it fails;
the abort is the box, not the tree.

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
