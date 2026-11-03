#!/bin/sh
# Linux build for the stack: web UI, streaming llama producer, and the ds4 CUDA engine.
#   ./build.sh [config] [target]      target: all | llama | ds4 | webui
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config=${1:-config.example.json}
target=${2:-all}
cuda_arch=${SLSTACK_CUDA_ARCHITECTURES:-120a}

python=${SLSTACK_PYTHON:-}
if [ -z "$python" ]; then
    for candidate in python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then python=$candidate; break; fi
    done
fi
[ -n "$python" ] || { echo 'Install Python 3.12 or newer, or set SLSTACK_PYTHON.' >&2; exit 1; }

runtime=$(cd "$root" && PYTHONDONTWRITEBYTECODE=1 "$python" -B -c \
    'import sys; from slstack.config import load; print(load(sys.argv[1])["runtime_root"])' "$config")
case "$runtime" in
    /*) ;;
    *) runtime="$root/$runtime" ;;
esac
build_temp="$runtime/build-temp"
mkdir -p "$build_temp" "$runtime/npm-cache"
export TMPDIR="$build_temp" TEMP="$build_temp" TMP="$build_temp" npm_config_cache="$runtime/npm-cache"
export PYTHONDONTWRITEBYTECODE=1

if [ "$target" = webui ] || [ "$target" = all ]; then
    ui="$root/engines/streaming-llama/prefill_llama.cpp/tools/server/webui"
    npm ci --ignore-scripts --prefix "$ui"
    npx --prefix "$ui" svelte-kit sync
    npx --prefix "$ui" vite build
fi

if [ "$target" = ds4 ] || [ "$target" = all ]; then
    # CUDA decode engine: Linux, io_uring batch reads, packed FP4 and Q4_K kernels.
    make -C "$root/engines/ds4" cuda CUDA_ARCH="sm_${cuda_arch%a}"
fi

if [ "$target" = llama ] || [ "$target" = all ]; then
    # Pinned CUTLASS 3.9.2 for the split-F32 attention used by the Fast tier.
    cutlass="$root/build/cutlass-v3.9.2"
    cutlass_commit=ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
    if [ ! -f "$cutlass/include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h" ]; then
        git clone --quiet --depth 1 --branch v3.9.2 https://github.com/NVIDIA/cutlass "$cutlass"
    fi
    [ "$(git -C "$cutlass" rev-parse HEAD)" = "$cutlass_commit" ] || {
        echo "CUTLASS checkout does not match the pinned revision." >&2; exit 1; }

    llama="$root/engines/streaming-llama/prefill_llama.cpp"
    mkdir -p "$root/build/llama/bin"
    cmake -G Ninja -S "$llama" -B "$root/build/llama" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
        -DGGML_BACKEND_DL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_DSV41_OFFLINE_PRODUCER=ON \
        -DLLAMA_DSV41_NATIVE_HALF_PROFILES=ON \
        -DLLAMA_DSV41_JSON_INCLUDE_DIR="$llama/vendor" \
        -DGGML_CUDA_DSV41_SPLIT_F32=ON \
        -DGGML_CUDA_CUTLASS_PATH="$cutlass"
    cmake --build "$root/build/llama" --parallel 2 --target llama-server llama-export-lora slstack-weight-fetcher llama-dsv41-prefill
    echo 'llama-dsv41-prefill runs on Linux: --check-native-profiles passes; a prefill still needs qualified weights and the Linux weight-provider lease (docs/linux-port-plan.md chunk 3.1d).' >&2
fi

echo "Built target '$target' against runtime root $runtime"
