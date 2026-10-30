/* io_uring-backed parallel expert reads for the CUDA SSD-streaming path (Linux only).
 *
 * WHY THIS EXISTS
 * ---------------
 * The streaming path reads every missed expert one pread at a time, three reads per expert
 * (gate/up/down). One outstanding request on one fd cannot raise queue depth, so a thread pool
 * over that fd buys nothing.
 *
 * At each MoE layer the router picks top-k experts simultaneously, so every miss for that layer is
 * known before any read is issued. They should go out as ONE batch. io_uring with IORING_OP_READ
 * and a registered fd is the Linux analogue of the Windows IoRing/DirectStorage queue: submissions
 * are queued in userspace, the kernel issues them concurrently, and completions are drained from a
 * CQ ring without a syscall per read.
 *
 * DESIGN
 * ------
 * Demand batches go into host destinations. The optional GPU proof gate keeps a host staging mirror
 * of the device arena: reads land in the mirror, one cudaMemcpyAsync per request moves it into the
 * device arena on a dedicated upload stream, and a cudaEvent recorded after the copy is what the
 * compute stream waits on. That is the same ordering the D3D12 fence gave: the host never blocks
 * compute queueing on the upload.
 *
 * NO SILENT FALLBACK: every entry point reports failure explicitly. Callers must decide loudly.
 *
 * Buffered reads only, through the fd the engine already opened. An O_DIRECT variant is later work:
 * it needs 4 KiB-aligned offsets and buffers and must not pad a production read into its neighbour.
 */
#ifndef DS4_BATCH_IO_H
#define DS4_BATCH_IO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the ring for the file behind `fd`. Returns 1 on success, 0 on failure
 * (with a detailed reason already printed to stderr). Safe to call again; re-initialises
 * if `fd` refers to a different file, which happens when the MTP model swaps in. */
int ds4_ds_init_from_fd(int fd);
/* Configure the normal engine option before transport initialization. 0/1 disables
 * splitting; 2..64 selects parts; default 8. DS4_DS_READ_PARTS is an explicit override.
 * Refuses changes to a live transport. */
int ds4_ds_configure_read_parts(uint32_t parts);

/* 1 once the ring is live. */
int ds4_ds_available(void);

/* Begin a new batch. */
void ds4_ds_batch_reset(void);

/* Add one read to the pending batch. `dst_host` must stay valid until submit returns.
 * Returns 1 on success, 0 if the batch is full or the arguments are invalid. */
int ds4_ds_batch_add(uint64_t file_offset, uint32_t bytes, void *dst_host);

/* Proof gate for device destinations. The arena is a device buffer with a host mirror;
 * `out_cuda_ptr` is its CUDA device address. GPU requests in one batch must not be mixed
 * with host requests. */
int ds4_ds_gpu_arena_create(uint64_t bytes, void **out_cuda_ptr);
/* Non-zero while the device arena exists. Any CUDA-side pointer obtained from
 * ds4_ds_gpu_arena_create() is DANGLING once this returns 0. */
int ds4_ds_gpu_arena_alive(void);
void ds4_ds_gpu_arena_release(void);
int ds4_ds_batch_add_gpu(uint64_t file_offset, uint32_t bytes, uint64_t arena_offset);
/* Which of ds4_ds_batch_add_gpu()'s refusal branches fired last; 0 = the last call succeeded.
 * 1 !init 2 !gpu_arena 3 bytes==0 4 offset/size out of arena 5 batch full
 * 6 device add into an already host batch 7 source outside the file. */
int ds4_ds_gpu_add_fail_reason(void);

/* Prefill pipeline. Requests are built once per model/layer, then replayed into the shared
 * device arena. Completion is exported as a CUDA event so the compute stream can wait on the
 * upload without the host blocking. */
typedef struct ds4_ds_wave_span {
    uint64_t file_offset;
    uint32_t bytes;
    uint32_t arena_slot;
    uint64_t arena_offset;
} ds4_ds_wave_span;
int ds4_ds_prefill_arenas_create(const uint64_t *bytes, uint32_t count, void **out_cuda_ptrs);
void ds4_ds_prefill_arenas_release(void);
int ds4_ds_prefill_waves_create(uint32_t count);
int ds4_ds_prefill_wave_build(uint32_t index, const ds4_ds_wave_span *spans, uint32_t count);
int ds4_ds_prefill_wave_submit(uint32_t index, uint64_t *out_fence_value);
int ds4_ds_prefill_wave_cuda_wait(uint64_t fence_value, void *cuda_stream);
/* DS4COMBO diagnostic: CUDA waits (on cuda_stream) for the device writes of the last
 * GPU-destination ds4_ds_batch_submit_wait(). */
int ds4_ds_gpu_fence_wait_last(void *cuda_stream);
int ds4_ds_prefill_waves_check_errors(void);
void ds4_ds_prefill_waves_release(void);

/* How many requests are pending. */
uint32_t ds4_ds_batch_count(void);
/* Pending bytes and conservative logical batch_add span capacity. With the default
 * DS4_DS_READ_PARTS=8, a span can expand to several physical requests; capacity already
 * accounts for that expansion. batch_count counts physical requests. */
uint64_t ds4_ds_batch_bytes(void);
uint32_t ds4_ds_batch_capacity(void);

/* Cheap phase/token context for transport bandwidth and slow-batch reporting.
 * set_batch_layer snapshots it onto the next split or synchronous batch. */
void ds4_ds_stats_set_phase(int decode);
void ds4_ds_stats_set_token(uint64_t token_index);
void ds4_ds_stats_set_batch_layer(uint32_t first_layer, uint32_t last_layer);
int ds4_ds_layer_trace_enabled(void);

/* Submit every pending request, wait for completion, and verify no request failed.
 * Returns 1 only if ALL requests completed successfully. A short read or a kernel error
 * is a failure: consumers must not touch the destination until this returns 1. */
int ds4_ds_batch_submit_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms);

/* Split submit/wait API. The transport owns an eight-slot FIFO ring; batch_wait drains
 * the oldest submitted slot. */
typedef struct ds4_ds_background_span { uint64_t offset; uint32_t bytes; void *destination; } ds4_ds_background_span;
/* One optional low-priority background batch; independent from demand FIFO/counters.
 * Destinations must remain pinned/alive until poll=1 or finish succeeds. */
int ds4_ds_background_submit(const ds4_ds_background_span *spans, uint32_t count);
int ds4_ds_background_poll(uint64_t *bytes, double *ms);
int ds4_ds_background_finish(void);
void ds4_ds_get_byte_counters(uint64_t *demand, double *demand_ms, uint64_t *background);
int ds4_ds_batch_submit(void);
uint32_t ds4_ds_batch_inflight(void);
uint32_t ds4_ds_pipeline_capacity(void);
/* How many slots the decode pipeline may keep in flight. */
uint32_t ds4_ds_pipeline_depth(void);
/* 1 = oldest submitted batch is complete, 0 = it is still in flight, -1 = none.
 * Non-consuming. */
int ds4_ds_batch_oldest_ready(void);
int ds4_ds_batch_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms);

/* Self-test: read `span_bytes` of the model twice -- once as sequential single-request submits
 * (queue depth 1) and once in batches of `batch` -- comparing both against pread for byte
 * equality and reporting both throughputs. Returns 1 if both transports matched pread. */
int ds4_ds_selftest(int fd, uint64_t start_offset, uint64_t span_bytes, uint32_t chunk_bytes,
                    uint32_t batch);

void ds4_ds_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_BATCH_IO_H */
