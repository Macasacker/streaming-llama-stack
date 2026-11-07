# `cuda/mmq/` - the vendored ggml CUDA prefill tier

`mma.cuh`, `mmq.cuh`, `vecdotq.cuh`, `unary.cuh`, `common.cuh`, `ggml-common.h` and the
`ggml.h` / `ggml-impl.h` / `ggml-cuda.h` redirects are copies of the mul_mat_q tier of
llama.cpp's ggml CUDA backend, MIT, `Copyright (c) 2023-2026 The ggml authors`; the licence and
the directory listing are in `../../THIRD_PARTY_NOTICES.md`. `ds4_ggml_stubs.h` is the shim the
copies include instead of real ggml headers: it declares the ggml surface the kernels reference and
deliberately leaves the names `common.cuh` already owns to `common.cuh`. The `ds4_*` files are this
fork's entry points on top of those copies - the MoE gate/up and down kernels, the D2R route, the
repack kernels and the Q4_K half panel - and `test/` holds prototypes for them with no build rule.

Nothing here is kept in sync with the llama tree's own `ggml/src/ggml-cuda/`. The two copies came
from different upstream commits (this one defines `GGML_CUDA_CC_CDNA4` and the NVFP4 block, the
llama copy does not) and each has diverged since. A sweep over one is a sweep over both, with that
tree's build as the gate: `make -C engines/ds4 cuda CUDA_ARCH=sm_120` here, `build.sh llama` there.

## What the Linux port pruned

The engine builds for one platform: `Makefile` passes `-gencode arch=compute_120a,code=sm_120a
-DDS4_CUDA_HAVE_MXF4=1` and nothing else that reaches a preprocessor, so no `GGML_USE_*` is ever
defined by a build of this fork.

Chunk 4.3a (2026-11-07) folded the branches the deleted `ggml-hip` and `ggml-musa` backends used to
define, the same mechanical collapse chunk 4.2b ran over the llama tree: 426 lines out of 13 files,
42 lines written back, every guard reduced to the branch nvcc had already kept; `vendors/hip.h`
adds another 308 deletions. Gone with them:

- `vendors/hip.h` (308 lines) - its only referrers were `#if defined(GGML_USE_HIP)` includes.
  `vendors/musa.h` is not in this checkout at all, which is only visible because the branch naming
  it was dead. `vendors/cuda.h` stays; it is the header the live include resolves to, and it is
  what defines `FP8_AVAILABLE`.
- `GGML_COMMON_DECL_HIP`/`_MUSA`/`_SYCL` and their `GGML_COMMON_IMPL_*` siblings, so the HIP, MUSA
  and SYCL arms of `ggml-common.h` - `<hip/hip_fp16.h>`, `<musa_fp16.h>`, `<sycl/half_type.hpp>` -
  are gone and only the CUDA arm remains.
- The AMD and Moore Threads toolchain macros: `HIP_VERSION`, `MUSART_VERSION`, `MUSART_HMASK`,
  `__MUSA_ARCH__`, `GGML_HIP_*`, `GGML_MUSA_*`, `DS4_HIP_MMQ_Y`, and the engine's
  `ds4_rocm_allocation_fits()` pool arm, which included `../../ds4_rocm_memory.h` - a header that
  does not exist in this checkout either.

Still present and still dead, deliberately left to the chunk that deletes the AMD and Moore Threads
kernel paths by hand: `GGML_CUDA_CC_IS_CDNA`/`_RDNA`/`_MTHREADS`, the `GGML_CUDA_CC_CDNA*`/`RDNA*`/
`QY*`/`PH1` constants, and the `AMD_MFMA_AVAILABLE`/`AMD_WMMA_AVAILABLE` readers - 210 mentions of
those two macros in this directory. Nothing defines them any more (their `#define`s were inside the
HIP branches this chunk folded), but their guards key on `__CUDA_ARCH__` or on those names, and
`__CUDA_ARCH__` is genuinely undefined for the host pass and defined for the device pass, so a
collapse cannot decide them. See the chunk's entry in `docs/linux-port-plan.md`.
