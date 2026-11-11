#pragma once
#include <cstdint>
// The exact radix top-k dispatch rule, as one function both sides of the coverage check read.
//
// ggml-cuda/top-k.cu decides which kernels a ggml top-k tensor runs; the V4.1 executor counts the
// dispatches it expects and refuses the layer when the counter disagrees
// ("Actual exact radix top-k first/merge dispatch coverage differs"). With the rule written twice,
// the two copies drifted: single-row decode was moved onto the deterministic tiles network
// (tests/dsv41_topk_single_row.cpp), while the executor's scan still dropped every rows<=1 node,
// so any decode step over more than 8192 index columns dispatched two radix kernels the check
// counted as zero and aborted the producer. See docs/SESSION_FAILURE_PLAN.md.
//
// Pure arithmetic on purpose: no device, no ggml, no CUDA, so it is testable without a card.
inline constexpr int DSV41_TOPK_TILE_SIZE = 8192;

// Dispatches added to ggml_backend_cuda_dsv41_topk_radix_queued() by one top-k tensor.
//   columns           - src0->ne[0], the number of scores per row
//   k                 - dst->ne[0], how many are selected
//   tagged            - the bounded streaming selector tag (ggml_dsv41_stream_topk_requested)
//   deterministic_rows - the row rule that keeps decode on the same network as prefill
//   bitonic_fits      - the padded row fits the single-kernel argsort's shared memory
//   radix_enabled     - the tile size and k select the radix kernel and its shared memory fits
//   cub_available     - the CUB path was compiled in; without it nothing takes the radix kernel
inline uint64_t dsv41_topk_radix_dispatches(int64_t columns, int k, bool tagged, bool deterministic_rows,
                                            bool bitonic_fits, bool radix_enabled, bool cub_available) {
    if (tagged)return radix_enabled ? 1 : 0;                 // one bounded tile pass
    if (!cub_available)return 0;                             // argsort fallback: no radix kernel
    if (deterministic_rows) {
        if (bitonic_fits)return 0;                           // one argsort kernel over the whole row
        if (k <= DSV41_TOPK_TILE_SIZE) {
            if (!radix_enabled)return 0;                     // tile sort, not the radix select
            uint64_t dispatches = 2;                         // the first pass, then the final merge
            int64_t width = ((columns + DSV41_TOPK_TILE_SIZE - 1) / DSV41_TOPK_TILE_SIZE) * int64_t(k);
            while (width > DSV41_TOPK_TILE_SIZE) {
                width = ((width + DSV41_TOPK_TILE_SIZE - 1) / DSV41_TOPK_TILE_SIZE) * int64_t(k);
                ++dispatches;
            }
            return dispatches;
        }
    }
    return 0;                                                // DeviceTopK: no radix kernel
}
