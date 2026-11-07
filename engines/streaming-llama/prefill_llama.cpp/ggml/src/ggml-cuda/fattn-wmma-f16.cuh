#pragma once

#include "common.cuh"



// WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
static bool ggml_cuda_should_use_wmma_fattn(const int cc) {
    if ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
        return true;
    } else if (GGML_CUDA_CC_IS_CDNA(cc)){
        return false;
    } else if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    } else {
        return false;
    }
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
