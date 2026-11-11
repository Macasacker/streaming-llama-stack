#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_radix_queued(void);
// Dispatches a given top-k tensor will add to that counter. The executor's coverage check sums
// this instead of re-deriving the backend's branch (ggml-dsv41-topk-plan.h).
GGML_BACKEND_API uint64_t ggml_backend_cuda_dsv41_topk_radix_dispatches(const ggml_tensor * src0,
                                                                        const ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
