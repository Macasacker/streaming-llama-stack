/* io_uring transport for the CUDA SSD-streaming expert path. See ds4_batch_io.h for rationale.
 *
 * Compiled by the host C++ compiler, not nvcc: liburing has no reason to be parsed by nvcc, and
 * this file only needs cuda_runtime_api.h for the device arena and the upload copies.
 *
 * The Windows original used DirectStorage queues plus a D3D12 fence imported into CUDA. The Linux
 * analogue is io_uring for the reads and a cudaEvent recorded after the upload copy for ordering:
 * same contract, same loud failures, no silent fallback.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <liburing.h>

#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "ds4_batch_io.h"

namespace {

constexpr uint32_t DS4_DS_SLOTS = 8;
constexpr uint32_t DS4_DS_PREFILL_ARENAS = 3;
constexpr uint64_t DEFAULT_QUEUE_CAPACITY = 256;

/* One read request, after span splitting. */
struct ds4_req {
    uint64_t offset;
    uint32_t bytes;
    void *host;           /* host destination; mirror offset for device-destination batches */
    uint64_t device_offset; /* 0 unless the batch is a device-destination batch */
};

/* Carried in a CQE's user_data so completion can be attributed to an owner and the
 * expected byte count checked. Lives in the owning batch's tag vector, which is
 * reserved before submission so the address stays stable while the request is in flight. */
struct ds4_tag {
    uint32_t owner;
    uint32_t bytes;
};

/* Latency controls, parsed once before any ring is created. Malformed opt-ins fail
 * initialization rather than silently falling back. */
uint64_t g_queue_capacity = DEFAULT_QUEUE_CAPACITY;
uint32_t g_read_parts = 8;
uint32_t g_requested_read_parts = 8;
uint32_t g_completion_spin_us = 0;
double   g_slow_batch_ms = 0.0;

bool controls_from_env(void) {
    struct Control { const char *name; unsigned long lo, hi, value; } controls[] = {
        {"DS4_DS_QUEUE_CAPACITY", 64, 1024, (unsigned long)DEFAULT_QUEUE_CAPACITY},
        {"DS4_DS_COMPLETION_SPIN_US", 0, 10000, 0},
        {"DS4_DS_READ_PARTS", 0, 64, g_requested_read_parts},
        {"DS4_DS_SLOW_BATCH_MS", 0, 100000, 0}
    };
    for (auto &c : controls) {
        const char *s = getenv(c.name);
        if (!s || !*s) continue;
        char *end = NULL;
        const unsigned long v = strtoul(s, &end, 10);
        if (*s < '0' || *s > '9' || end == s || *end || v < c.lo || v > c.hi) {
            fprintf(stderr, "ds4: invalid %s=%s (expected %lu..%lu)\n", c.name, s, c.lo, c.hi);
            return false;
        }
        c.value = v;
    }
    if ((controls[0].value & (controls[0].value - 1)) != 0) {
        fprintf(stderr, "ds4: DS4_DS_QUEUE_CAPACITY must be a power of two\n");
        return false;
    }
    g_queue_capacity = controls[0].value;
    g_completion_spin_us = (uint32_t)controls[1].value;
    g_read_parts = controls[2].value <= 1 ? 1 : (uint32_t)controls[2].value;
    g_slow_batch_ms = (double)controls[3].value;
    return true;
}

/* Preserve small/unaligned requests exactly. Never pad a production read into its
 * neighbour: only split complete 4KiB units of the original span. */
uint32_t request_parts(uint64_t offset, uint32_t bytes) {
    return g_read_parts > 1 && (offset & 4095u) == 0 && (bytes & 4095u) == 0 &&
           bytes / 4096u >= g_read_parts ? g_read_parts : 1;
}

/* Split a span into parts so several physical reads go out at once. Remainder blocks are
 * distributed rather than lumped into one disproportionately slow final request. */
void append_request_parts(const ds4_req &original, uint32_t parts, std::vector<ds4_req> &out) {
    const uint32_t total = original.bytes;
    const uint32_t blocks = total / 4096u;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        ds4_req req = original;
        req.bytes = parts == 1 ? total
            : (blocks / parts + (i < blocks % parts ? 1u : 0u)) * 4096u;
        req.offset = original.offset + offset;
        req.host = (char *)original.host + offset;
        req.device_offset = original.device_offset + offset;
        out.push_back(req);
        offset += req.bytes;
    }
}

double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void report(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

struct ring_owner {
    io_uring ring = {};
    bool live = false;
    uint64_t capacity = 0;
};

bool ring_create(ring_owner &owner, uint64_t capacity, const char *what) {
    if (owner.live) return true;
    const int rc = io_uring_queue_init((unsigned)capacity, &owner.ring, 0);
    if (rc < 0) {
        report("ds4: %s io_uring_queue_init(%llu) failed: %s\n", what,
               (unsigned long long)capacity, strerror(-rc));
        return false;
    }
    owner.live = true;
    owner.capacity = capacity;
    return true;
}

bool file_identity(int fd, uint64_t &dev, uint64_t &ino, uint64_t &size) {
    struct stat st;
    if (fstat(fd, &st) != 0) return false;
    dev = (uint64_t)st.st_dev;
    ino = (uint64_t)st.st_ino;
    size = (uint64_t)st.st_size;
    return true;
}

ring_owner g_demand;      /* decode/prefill demand batches */
ring_owner g_background;  /* one low-priority batch, independent counters */
ring_owner g_wave_ring;   /* prefill pipeline waves */

int      g_fd = -1;
uint64_t g_file_size = 0;
uint64_t g_file_dev = 0;
uint64_t g_file_ino = 0;
int      g_init_attempted = 0;
int      g_init_ok = 0;

/* Device destination arena: a CUDA buffer plus the host mirror the reads land in. */
void     *g_gpu_cuda_ptr = nullptr;
char     *g_gpu_mirror = nullptr;
uint64_t  g_gpu_arena_size = 0;
cudaStream_t g_upload_stream = nullptr;
cudaEvent_t  g_last_upload_event = nullptr; /* upload of the last device batch_submit_wait */
int      g_gpu_add_fail_reason = 0;

/* Prefill pipeline arenas and waves. */
void     *g_prefill_cuda_ptr[DS4_DS_PREFILL_ARENAS] = {};
char     *g_prefill_mirror[DS4_DS_PREFILL_ARENAS] = {};
uint64_t  g_prefill_arena_size[DS4_DS_PREFILL_ARENAS] = {};
uint32_t  g_prefill_arena_count = 0;

struct ds4_wave {
    std::vector<ds4_req> spans;
    std::vector<ds4_tag> tags;
    uint32_t arena_slot;
    uint64_t fence;
    uint32_t pending;
    int error;
    bool submitted;
    cudaEvent_t event;
};
std::vector<ds4_wave> g_prefill_waves;
uint64_t g_prefill_fence_next = 1;

/* Demand FIFO: slots hold submitted batches; drain waits for the oldest. */
struct ds4_slot {
    std::vector<ds4_req> reqs;
    std::vector<ds4_tag> tags;
    uint64_t bytes = 0;
    uint32_t pending = 0;
    int error = 0;
    bool busy = false;
    bool gpu = false;
    double t0 = 0.0;
    uint64_t token = UINT64_MAX;
    uint32_t first_layer = 0, last_layer = 0;
    int phase_decode = 0;
    int token_valid = 0;
    int tags_valid = 0;
};
ds4_slot g_slots[DS4_DS_SLOTS];
uint32_t g_slot_next = 0;   /* next slot to fill */
uint32_t g_slot_drain = 0;  /* oldest slot to wait */
uint32_t g_slot_busy = 0;   /* how many outstanding */

std::vector<ds4_req> g_batch;
uint64_t g_batch_bytes = 0;
bool g_batch_gpu = false;

int      g_phase_decode = 0;
uint64_t g_profile_token = UINT64_MAX;
int      g_profile_token_valid = 0;
struct ds4_sync_tags { uint32_t first_layer, last_layer; uint64_t token; int phase_decode, token_valid, valid; };
ds4_sync_tags g_sync_tags = {};

uint64_t g_total_bytes = 0;
double   g_total_ms = 0.0;
uint64_t g_batches = 0;
uint64_t g_requests = 0;
double   g_max_ms = 0.0;
uint64_t g_prefill_bytes = 0;
uint64_t g_prefill_waves_done = 0;
uint64_t g_prefill_requests = 0;

bool     g_background_busy = false;
uint64_t g_background_bytes = 0;
uint32_t g_background_pending = 0;
int      g_background_error = 0;
double   g_background_started = 0.0;
std::vector<ds4_req> g_background_reqs;
std::vector<ds4_tag> g_background_tags;

struct ds4_layer_sample {
    uint64_t token, bytes;
    uint32_t first_layer, last_layer, requests;
    double batch_ms, wait_ms;
};
std::vector<ds4_layer_sample> g_layer_samples;

bool layer_trace_enabled(void) {
    static const bool enabled = []() {
        const char *s = getenv("DS4_DS_LAYER_TRACE");
        return s && !strcmp(s, "1");
    }();
    return enabled;
}

/* A completion can arrive even when the read failed or came up short, so every CQE result is
 * checked before any consumer is allowed to touch the destination memory. */
bool cqe_result(const io_uring_cqe *cqe, const ds4_tag &tag, const char *what, int *error) {
    if (cqe->res < 0) {
        report("ds4: %s request FAILED errno=%s bytes=%u\n", what, strerror(-cqe->res), tag.bytes);
        *error = cqe->res;
        return false;
    }
    if ((uint32_t)cqe->res != tag.bytes) {
        report("ds4: %s request SHORT res=%d expected=%u\n", what, cqe->res, tag.bytes);
        *error = -EIO;
        return false;
    }
    return true;
}

/* Move a completed device batch into the CUDA arena on the upload stream. */
bool upload_reqs(const std::vector<ds4_req> &reqs, void *device, const char *what) {
    for (const auto &req : reqs) {
        const cudaError_t err = cudaMemcpyAsync(
            (char *)device + req.device_offset, req.host, req.bytes,
            cudaMemcpyHostToDevice, g_upload_stream);
        if (err != cudaSuccess) {
            report("ds4: %s upload FAILED: %s\n", what, cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

void arena_release(void) {
    if (g_last_upload_event) { cudaEventDestroy(g_last_upload_event); g_last_upload_event = nullptr; }
    if (g_gpu_cuda_ptr) { cudaFree(g_gpu_cuda_ptr); g_gpu_cuda_ptr = nullptr; }
    if (g_gpu_mirror) { free(g_gpu_mirror); g_gpu_mirror = nullptr; }
    g_gpu_arena_size = 0;
    g_gpu_add_fail_reason = 0;
}

void prefill_arenas_release(void) {
    for (uint32_t i = 0; i < DS4_DS_PREFILL_ARENAS; i++) {
        if (g_prefill_cuda_ptr[i]) { cudaFree(g_prefill_cuda_ptr[i]); g_prefill_cuda_ptr[i] = nullptr; }
        if (g_prefill_mirror[i]) { free(g_prefill_mirror[i]); g_prefill_mirror[i] = nullptr; }
        g_prefill_arena_size[i] = 0;
    }
    g_prefill_arena_count = 0;
}

void waves_release(void) {
    for (auto &w : g_prefill_waves) if (w.event) cudaEventDestroy(w.event);
    g_prefill_waves.clear();
    g_prefill_fence_next = 1;
}

void release_all(void) {
    if (!ds4_ds_background_finish()) abort();
    if (g_background.live) { io_uring_queue_exit(&g_background.ring); g_background.live = false; }
    if (g_wave_ring.live) { io_uring_queue_exit(&g_wave_ring.ring); g_wave_ring.live = false; }
    if (g_demand.live) { io_uring_queue_exit(&g_demand.ring); g_demand.live = false; }
    if (g_upload_stream) { cudaStreamDestroy(g_upload_stream); g_upload_stream = nullptr; }
    waves_release();
    prefill_arenas_release();
    arena_release();
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
    g_init_ok = 0;
    g_fd = -1;
}

/* Queue every request of a slot and submit them together. */
bool slot_submit(ds4_slot &s, uint32_t slot_index, ring_owner &ring, const char *what) {
    s.tags.clear();
    s.tags.reserve(s.reqs.size());
    for (const auto &req : s.reqs) {
        s.tags.push_back({slot_index, req.bytes});
        io_uring_sqe *sqe = io_uring_get_sqe(&ring.ring);
        if (!sqe) {
            report("ds4: %s SQE ring exhausted at %zu requests\n", what, s.reqs.size());
            s.error = -EAGAIN;
            s.pending = 0;
            return false;
        }
        io_uring_prep_read(sqe, g_fd, req.host, req.bytes, req.offset);
        io_uring_sqe_set_data(sqe, &s.tags.back());
        s.pending++;
    }
    const int submitted = io_uring_submit(&ring.ring);
    if (submitted < (int)s.pending) {
        report("ds4: %s submit returned %d for %u requests\n", what, submitted, s.pending);
        s.error = submitted < 0 ? submitted : -EIO;
        s.pending = 0;
        return false;
    }
    return true;
}

/* Drain until one slot is complete. Returns false if any request failed. */
bool slot_drain(uint32_t slot, double *out_wait_ms) {
    ds4_slot &s = g_slots[slot];
    const double start = wall_ms();
    const double spin = (double)g_completion_spin_us / 1000.0;
    while (s.pending > 0) {
        io_uring_cqe *cqe = nullptr;
        int got = io_uring_peek_cqe(&g_demand.ring, &cqe);
        if (got == -EAGAIN && spin > 0.0 && wall_ms() - start < spin) {
            sched_yield();
            continue;
        }
        if (got == -EAGAIN) {
            got = io_uring_wait_cqe(&g_demand.ring, &cqe);
            if (got < 0) {
                report("ds4: demand completion wait failed: %s\n", strerror(-got));
                s.error = -EINVAL;
                s.pending = 0;
                return false;
            }
            got = 0;
        }
        if (got < 0) {
            report("ds4: demand completion peek failed: %s\n", strerror(-got));
            s.error = -EINVAL;
            s.pending = 0;
            return false;
        }
        const ds4_tag *tag = (const ds4_tag *)cqe->user_data;
        if (tag && tag->owner < DS4_DS_SLOTS) {
            ds4_slot &target = g_slots[tag->owner];
            cqe_result(cqe, *tag, "demand", &target.error);
            target.pending--;
        }
        io_uring_cqe_seen(&g_demand.ring, cqe);
    }
    if (out_wait_ms) *out_wait_ms = wall_ms() - start;
    return s.error == 0;
}

/* Submit the pending batch into the next FIFO slot. */
int submit_slot(void) {
    if (g_batch.empty()) return 1;
    if (!g_init_ok) { report("ds4: batch submit before transport init\n"); return 0; }
    if (g_slot_busy >= DS4_DS_SLOTS) {
        report("ds4: demand FIFO full (%u slots outstanding)\n", g_slot_busy);
        return 0;
    }
    ds4_slot &s = g_slots[g_slot_next];
    s.reqs.clear();
    for (const auto &req : g_batch) {
        append_request_parts(req, request_parts(req.offset, req.bytes), s.reqs);
    }
    s.bytes = g_batch_bytes;
    s.gpu = g_batch_gpu;
    s.pending = 0;
    s.error = 0;
    s.t0 = wall_ms();
    s.token = g_profile_token;
    s.token_valid = g_profile_token_valid;
    s.first_layer = g_sync_tags.first_layer;
    s.last_layer = g_sync_tags.last_layer;
    s.phase_decode = g_phase_decode;
    s.tags_valid = g_sync_tags.valid;
    const uint32_t index = g_slot_next;
    if (!slot_submit(s, index, g_demand, "demand")) return 0;
    s.busy = true;
    g_slot_busy++;
    g_slot_next = (g_slot_next + 1) % DS4_DS_SLOTS;
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
    return 1;
}

void record_batch_stats(double batch_ms, double wait_ms, size_t requests, uint64_t bytes,
                        uint32_t first_layer, uint32_t last_layer, uint64_t token,
                        int phase, bool gpu) {
    g_total_bytes += bytes;
    g_total_ms += batch_ms;
    g_batches++;
    g_requests += requests;
    if (batch_ms > g_max_ms) g_max_ms = batch_ms;
    if (g_slow_batch_ms > 0.0 && batch_ms >= g_slow_batch_ms) {
        report("ds4: DS SLOW ordinal=%llu lat=%.3fms wait_ms=%.3f requests=%zu "
               "bytes=%.2fMiB rate=%.3fGiB/s phase=%s token=%lld layer=%u-%u dest=%s inflight=%u\n",
               (unsigned long long)g_batches, batch_ms, wait_ms, requests,
               (double)bytes / (1024.0 * 1024.0),
               bytes / (1024.0 * 1024.0 * 1024.0) / (batch_ms / 1000.0),
               phase ? "decode" : "prefill",
               token == UINT64_MAX ? -1 : (long long)token, first_layer, last_layer,
               gpu ? "device" : "host", g_slot_busy + 1u);
    }
    if (layer_trace_enabled()) {
        g_layer_samples.push_back({token, bytes, first_layer, last_layer,
                                   (uint32_t)requests, batch_ms, wait_ms});
    }
}

void dump_stats(void) {
    if (g_batches) {
        report("ds4: DS TRANSPORT batches=%llu requests=%llu bytes=%.2fMiB total_ms=%.3f "
               "mean=%.3fms max=%.3fms rate=%.3fGiB/s\n",
               (unsigned long long)g_batches, (unsigned long long)g_requests,
               (double)g_total_bytes / (1024.0 * 1024.0), g_total_ms,
               g_total_ms / (double)g_batches, g_max_ms,
               g_total_bytes / (1024.0 * 1024.0 * 1024.0) / (g_total_ms / 1000.0));
    }
    if (g_prefill_waves_done) {
        report("ds4: DS PREFILL waves=%llu requests=%llu bytes=%.2fMiB\n",
               (unsigned long long)g_prefill_waves_done,
               (unsigned long long)g_prefill_requests,
               (double)g_prefill_bytes / (1024.0 * 1024.0));
    }
    if (g_background_bytes) {
        report("ds4: DS BACKGROUND bytes=%.2fMiB\n",
               (double)g_background_bytes / (1024.0 * 1024.0));
    }
    for (const auto &s : g_layer_samples) {
        report("RTXDISKLAT_LAYER token=%llu layer=%u-%u bytes=%llu requests=%u batch_ms=%.9g wait_ms=%.9g\n",
               (unsigned long long)s.token, s.first_layer, s.last_layer,
               (unsigned long long)s.bytes, s.requests, s.batch_ms, s.wait_ms);
    }
    g_layer_samples.clear();
}

} // namespace

extern "C" int ds4_ds_configure_read_parts(uint32_t parts) {
    if (g_init_ok) {
        report("ds4: DS4_DS_READ_PARTS cannot change a live transport\n");
        return 0;
    }
    if (parts > 64) {
        report("ds4: invalid read parts=%u (expected 0..64)\n", parts);
        return 0;
    }
    g_requested_read_parts = parts <= 1 ? 1 : parts;
    g_read_parts = g_requested_read_parts;
    return 1;
}

extern "C" int ds4_ds_init_from_fd(int fd) {
    if (fd < 0) { report("ds4: transport init given fd=%d\n", fd); return 0; }
    uint64_t dev = 0, ino = 0, size = 0;
    if (!file_identity(fd, dev, ino, size)) {
        report("ds4: transport init: fstat(fd=%d) failed: %s\n", fd, strerror(errno));
        return 0;
    }
    if (g_init_ok && g_fd == fd && g_file_dev == dev && g_file_ino == ino && g_file_size == size) {
        return 1;
    }
    if (g_init_ok) release_all(); /* the MTP model swaps the fd; re-point, do not read the wrong file */
    g_init_attempted = 1;
    if (!controls_from_env()) return 0;
    if (!ring_create(g_demand, g_queue_capacity, "demand")) return 0;
    if (!ring_create(g_background, 32, "background")) return 0;
    if (cudaStreamCreateWithFlags(&g_upload_stream, cudaStreamNonBlocking) != cudaSuccess) {
        report("ds4: transport init: upload stream creation failed\n");
        io_uring_queue_exit(&g_demand.ring);
        g_demand.live = false;
        io_uring_queue_exit(&g_background.ring);
        g_background.live = false;
        return 0;
    }
    g_fd = fd;
    g_file_dev = dev;
    g_file_ino = ino;
    g_file_size = size;
    g_init_ok = 1;
    report("ds4: DS TRANSPORT io_uring capacity=%llu read_parts=%u size=%.2fGiB\n",
           (unsigned long long)g_queue_capacity, g_read_parts,
           (double)size / (1024.0 * 1024.0 * 1024.0));
    return 1;
}

extern "C" int ds4_ds_available(void) { return g_init_ok ? 1 : 0; }

extern "C" void ds4_ds_batch_reset(void) {
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
    g_gpu_add_fail_reason = 0;
}

extern "C" int ds4_ds_batch_add(uint64_t file_offset, uint32_t bytes, void *dst_host) {
    if (!g_init_ok || !dst_host || !bytes) return 0;
    if (g_batch_gpu) { g_gpu_add_fail_reason = 6; return 0; }
    if (file_offset + bytes > g_file_size) return 0;
    if (g_batch.size() >= g_queue_capacity) return 0;
    g_batch.push_back({file_offset, bytes, dst_host, 0});
    g_batch_bytes += bytes;
    return 1;
}

extern "C" int ds4_ds_gpu_arena_create(uint64_t bytes, void **out_cuda_ptr) {
    if (!g_init_ok || !bytes || !out_cuda_ptr) return 0;
    void *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess || !device) {
        report("ds4: GPU arena cudaMalloc %.2fMiB failed\n", (double)bytes / (1024.0 * 1024.0));
        return 0;
    }
    char *mirror = nullptr;
    if (posix_memalign((void **)&mirror, 4096, bytes) != 0 || !mirror) {
        cudaFree(device);
        report("ds4: GPU arena host mirror alloc %.2fMiB failed\n", (double)bytes / (1024.0 * 1024.0));
        return 0;
    }
    arena_release();
    g_gpu_cuda_ptr = device;
    g_gpu_mirror = mirror;
    g_gpu_arena_size = bytes;
    *out_cuda_ptr = device;
    return 1;
}

extern "C" int ds4_ds_gpu_arena_alive(void) { return g_gpu_cuda_ptr ? 1 : 0; }

extern "C" void ds4_ds_gpu_arena_release(void) { arena_release(); }

extern "C" int ds4_ds_batch_add_gpu(uint64_t file_offset, uint32_t bytes, uint64_t arena_offset) {
    if (!g_init_ok) { g_gpu_add_fail_reason = 1; return 0; }
    if (!g_gpu_cuda_ptr) { g_gpu_add_fail_reason = 2; return 0; }
    if (!bytes) { g_gpu_add_fail_reason = 3; return 0; }
    if (arena_offset + bytes > g_gpu_arena_size) { g_gpu_add_fail_reason = 4; return 0; }
    if (g_batch.size() >= g_queue_capacity) { g_gpu_add_fail_reason = 5; return 0; }
    if (!g_batch_gpu && !g_batch.empty()) { g_gpu_add_fail_reason = 6; return 0; }
    if (file_offset + bytes > g_file_size) { g_gpu_add_fail_reason = 7; return 0; }
    g_batch_gpu = true;
    g_batch.push_back({file_offset, bytes, g_gpu_mirror + arena_offset, arena_offset});
    g_batch_bytes += bytes;
    g_gpu_add_fail_reason = 0;
    return 1;
}

extern "C" int ds4_ds_gpu_add_fail_reason(void) { return g_gpu_add_fail_reason; }

extern "C" int ds4_ds_prefill_arenas_create(const uint64_t *bytes, uint32_t count,
                                            void **out_cuda_ptrs) {
    if (!g_init_ok || !bytes || !out_cuda_ptrs || !count) return 0;
    if (count > DS4_DS_PREFILL_ARENAS) {
        report("ds4: prefill arenas %u requested, %u supported\n", count, DS4_DS_PREFILL_ARENAS);
        return 0;
    }
    arena_release();
    prefill_arenas_release();
    for (uint32_t i = 0; i < count; ++i) {
        void *device = nullptr;
        if (cudaMalloc(&device, bytes[i]) != cudaSuccess || !device) {
            report("ds4: prefill arena %u cudaMalloc %.2fMiB failed\n", i,
                   (double)bytes[i] / (1024.0 * 1024.0));
            prefill_arenas_release();
            return 0;
        }
        char *mirror = nullptr;
        if (posix_memalign((void **)&mirror, 4096, bytes[i]) != 0 || !mirror) {
            cudaFree(device);
            prefill_arenas_release();
            return 0;
        }
        g_prefill_cuda_ptr[i] = device;
        g_prefill_mirror[i] = mirror;
        g_prefill_arena_size[i] = bytes[i];
        out_cuda_ptrs[i] = device;
    }
    g_prefill_arena_count = count;
    return 1;
}

extern "C" void ds4_ds_prefill_arenas_release(void) { prefill_arenas_release(); }

extern "C" int ds4_ds_prefill_waves_create(uint32_t count) {
    if (!g_init_ok || !g_prefill_arena_count) return 0;
    waves_release();
    if (!ring_create(g_wave_ring, g_queue_capacity, "prefill wave")) return 0;
    g_prefill_waves.clear();
    g_prefill_waves.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        ds4_wave w;
        w.arena_slot = 0;
        w.pending = 0;
        w.error = 0;
        w.submitted = false;
        w.fence = 0;
        if (cudaEventCreateWithFlags(&w.event, cudaEventDisableTiming) != cudaSuccess) {
            w.event = nullptr;
        }
        g_prefill_waves.push_back(w);
    }
    return 1;
}

extern "C" int ds4_ds_prefill_wave_build(uint32_t index, const ds4_ds_wave_span *spans,
                                         uint32_t count) {
    if (!g_init_ok || index >= g_prefill_waves.size() || !spans) return 0;
    ds4_wave &w = g_prefill_waves[index];
    w.spans.clear();
    w.pending = 0;
    w.error = 0;
    w.submitted = false;
    for (uint32_t i = 0; i < count; ++i) {
        const ds4_ds_wave_span &s = spans[i];
        if (!s.bytes || s.arena_slot >= g_prefill_arena_count) return 0;
        if (s.arena_offset + s.bytes > g_prefill_arena_size[s.arena_slot]) return 0;
        if (s.file_offset + s.bytes > g_file_size) return 0;
        w.arena_slot = s.arena_slot;
        ds4_req req{s.file_offset, s.bytes,
                    g_prefill_mirror[s.arena_slot] + s.arena_offset, s.arena_offset};
        append_request_parts(req, request_parts(req.offset, req.bytes), w.spans);
    }
    return 1;
}

extern "C" int ds4_ds_prefill_wave_submit(uint32_t index, uint64_t *out_fence_value) {
    if (!g_init_ok || index >= g_prefill_waves.size() || !g_wave_ring.live) return 0;
    ds4_wave &w = g_prefill_waves[index];
    if (w.spans.empty()) return 0;
    w.pending = 0;
    w.error = 0;
    w.tags.clear();
    w.tags.reserve(w.spans.size());
    for (const auto &req : w.spans) {
        w.tags.push_back({index, req.bytes});
        io_uring_sqe *sqe = io_uring_get_sqe(&g_wave_ring.ring);
        if (!sqe) {
            report("ds4: prefill wave SQE ring exhausted at %zu requests\n", w.spans.size());
            w.error = -EAGAIN;
            w.pending = 0;
            return 0;
        }
        io_uring_prep_read(sqe, g_fd, req.host, req.bytes, req.offset);
        io_uring_sqe_set_data(sqe, &w.tags.back());
        w.pending++;
    }
    const int submitted = io_uring_submit(&g_wave_ring.ring);
    if (submitted < (int)w.pending) {
        report("ds4: prefill wave submit returned %d for %u requests\n", submitted, w.pending);
        w.error = submitted < 0 ? submitted : -EIO;
        w.pending = 0;
        return 0;
    }
    w.submitted = true;
    w.fence = g_prefill_fence_next++;
    if (out_fence_value) *out_fence_value = w.fence;
    return 1;
}

/* Wait for a wave's reads on the host, move them into its device arena, then let the compute
 * stream wait on the recorded event. The host never blocks compute queueing on the upload. */
extern "C" int ds4_ds_prefill_wave_cuda_wait(uint64_t fence_value, void *cuda_stream) {
    if (!g_init_ok) return 0;
    for (auto &w : g_prefill_waves) {
        if (w.fence != fence_value || !w.submitted) continue;
        while (w.pending > 0) {
            io_uring_cqe *cqe = nullptr;
            const int got = io_uring_wait_cqe(&g_wave_ring.ring, &cqe);
            if (got < 0) {
                report("ds4: prefill wave completion wait failed: %s\n", strerror(-got));
                w.error = -EINVAL;
                w.pending = 0;
                return 0;
            }
            const ds4_tag *tag = (const ds4_tag *)cqe->user_data;
            if (tag && tag->owner < g_prefill_waves.size()) {
                ds4_wave &target = g_prefill_waves[tag->owner];
                cqe_result(cqe, *tag, "prefill wave", &target.error);
                target.pending--;
            }
            io_uring_cqe_seen(&g_wave_ring.ring, cqe);
        }
        if (w.error != 0) return 0;
        if (!upload_reqs(w.spans, g_prefill_cuda_ptr[w.arena_slot], "prefill wave")) {
            w.error = -EIO;
            return 0;
        }
        uint64_t bytes = 0;
        for (const auto &req : w.spans) bytes += req.bytes;
        if (w.event) cudaEventRecord(w.event, g_upload_stream);
        if (cuda_stream) {
            if (cudaStreamWaitEvent((cudaStream_t)cuda_stream, w.event, 0) != cudaSuccess) {
                report("ds4: prefill wave stream wait FAILED\n");
                return 0;
            }
        } else if (w.event) {
            cudaEventSynchronize(w.event);
        }
        w.submitted = false;
        g_prefill_bytes += bytes;
        g_prefill_waves_done++;
        g_prefill_requests += w.spans.size();
        return 1;
    }
    report("ds4: prefill wave fence %llu not outstanding\n", (unsigned long long)fence_value);
    return 0;
}

extern "C" int ds4_ds_gpu_fence_wait_last(void *cuda_stream) {
    if (!g_last_upload_event) return 1;
    if (cuda_stream) {
        return cudaStreamWaitEvent((cudaStream_t)cuda_stream, g_last_upload_event, 0) == cudaSuccess
            ? 1 : 0;
    }
    return cudaEventSynchronize(g_last_upload_event) == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_ds_prefill_waves_check_errors(void) {
    for (const auto &w : g_prefill_waves) if (w.error != 0) return 0;
    return 1;
}

extern "C" void ds4_ds_prefill_waves_release(void) { waves_release(); }

extern "C" uint32_t ds4_ds_batch_count(void) { return (uint32_t)g_batch.size(); }
extern "C" uint64_t ds4_ds_batch_bytes(void) { return g_batch_bytes; }

/* Logical capacity: a span that splits into parts consumes several physical SQE slots. */
extern "C" uint32_t ds4_ds_batch_capacity(void) {
    const uint64_t cap = g_init_ok ? g_queue_capacity : 0;
    return g_read_parts > 1 ? (uint32_t)(cap / g_read_parts) : (uint32_t)cap;
}

extern "C" void ds4_ds_stats_set_phase(int decode) { g_phase_decode = decode ? 1 : 0; }

extern "C" void ds4_ds_stats_set_token(uint64_t token_index) {
    g_profile_token = token_index;
    g_profile_token_valid = 1;
}

extern "C" void ds4_ds_stats_set_batch_layer(uint32_t first_layer, uint32_t last_layer) {
    g_sync_tags.first_layer = first_layer;
    g_sync_tags.last_layer = last_layer;
    g_sync_tags.phase_decode = g_phase_decode;
    g_sync_tags.token = g_profile_token;
    g_sync_tags.token_valid = g_profile_token_valid;
    g_sync_tags.valid = 1;
}

extern "C" int ds4_ds_layer_trace_enabled(void) { return layer_trace_enabled() ? 1 : 0; }

extern "C" int ds4_ds_batch_submit(void) { return submit_slot(); }

extern "C" uint32_t ds4_ds_batch_inflight(void) { return g_slot_busy; }

extern "C" int ds4_ds_batch_oldest_ready(void) {
    if (g_slot_busy == 0) return -1;
    return g_slots[g_slot_drain].busy && g_slots[g_slot_drain].pending == 0 ? 1 : 0;
}

extern "C" uint32_t ds4_ds_pipeline_capacity(void) { return DS4_DS_SLOTS; }

extern "C" uint32_t ds4_ds_pipeline_depth(void) {
    static uint32_t cached = 0;
    if (cached == 0) {
        cached = 2;
        const char *env = getenv("DS4_DS_PIPELINE_DEPTH");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == 0 && v >= 1) {
                cached = v > DS4_DS_SLOTS ? DS4_DS_SLOTS : (uint32_t)v;
            }
        }
        if (cached != 2) report("ds4: io_uring pipeline depth = %u slots (max %u)\n", cached, DS4_DS_SLOTS);
    }
    return cached;
}

extern "C" int ds4_ds_batch_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms) {
    if (g_slot_busy == 0) {
        if (out_bytes) *out_bytes = 0;
        if (out_ms) *out_ms = 0.0;
        if (out_wait_ms) *out_wait_ms = 0.0;
        return 1;
    }
    ds4_slot &s = g_slots[g_slot_drain];
    double wait_ms = 0.0;
    if (!slot_drain(g_slot_drain, &wait_ms)) return 0;
    if (s.gpu) {
        if (!upload_reqs(s.reqs, g_gpu_cuda_ptr, "demand")) { s.error = -EIO; return 0; }
        if (!g_last_upload_event) {
            cudaEventCreateWithFlags(&g_last_upload_event, cudaEventDisableTiming);
        }
        cudaEventRecord(g_last_upload_event, g_upload_stream);
    }
    if (s.error != 0) return 0;
    const double batch_ms = wall_ms() - s.t0;
    if (out_bytes) *out_bytes = s.bytes;
    if (out_ms) *out_ms = batch_ms;
    if (out_wait_ms) *out_wait_ms = wait_ms;
    record_batch_stats(batch_ms, wait_ms, s.reqs.size(), s.bytes,
                       s.first_layer, s.last_layer, s.token, s.phase_decode, s.gpu);
    s.busy = false;
    s.reqs.clear();
    s.tags.clear();
    g_slot_busy--;
    g_slot_drain = (g_slot_drain + 1) % DS4_DS_SLOTS;
    return 1;
}

extern "C" int ds4_ds_batch_submit_wait(uint64_t *out_bytes, double *out_ms, double *out_wait_ms) {
    if (!submit_slot()) return 0;
    return ds4_ds_batch_wait(out_bytes, out_ms, out_wait_ms);
}

extern "C" int ds4_ds_background_submit(const ds4_ds_background_span *spans, uint32_t count) {
    if (!g_init_ok || !spans || !count) return 0;
    if (g_background_busy) { report("ds4: background batch already in flight\n"); return 0; }
    if (!g_background.live) return 0;
    g_background_reqs.clear();
    g_background_tags.clear();
    g_background_pending = 0;
    g_background_error = 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const ds4_ds_background_span &s = spans[i];
        if (!s.bytes || !s.destination || s.offset + s.bytes > g_file_size) return 0;
        ds4_req req{s.offset, s.bytes, s.destination, 0};
        append_request_parts(req, request_parts(req.offset, req.bytes), g_background_reqs);
        bytes += s.bytes;
    }
    g_background_tags.reserve(g_background_reqs.size());
    for (size_t i = 0; i < g_background_reqs.size(); ++i) {
        g_background_tags.push_back({0, g_background_reqs[i].bytes});
        io_uring_sqe *sqe = io_uring_get_sqe(&g_background.ring);
        if (!sqe) {
            report("ds4: background SQE ring exhausted at %zu requests\n", g_background_reqs.size());
            g_background_error = -EAGAIN;
            return 0;
        }
        io_uring_prep_read(sqe, g_fd, g_background_reqs[i].host, g_background_reqs[i].bytes,
                           g_background_reqs[i].offset);
        io_uring_sqe_set_data(sqe, &g_background_tags[i]);
        g_background_pending++;
    }
    const int submitted = io_uring_submit(&g_background.ring);
    if (submitted < (int)g_background_pending) {
        report("ds4: background submit returned %d for %u requests\n",
               submitted, g_background_pending);
        g_background_error = submitted < 0 ? submitted : -EIO;
        g_background_pending = 0;
        return 0;
    }
    g_background_busy = true;
    g_background_started = wall_ms();
    g_background_bytes += bytes;
    return 1;
}

extern "C" int ds4_ds_background_poll(uint64_t *bytes, double *ms) {
    if (!g_background_busy) return 1;
    while (g_background_pending > 0) {
        io_uring_cqe *cqe = nullptr;
        const int got = io_uring_peek_cqe(&g_background.ring, &cqe);
        if (got == -EAGAIN) return 0;
        if (got < 0) {
            report("ds4: background completion peek failed: %s\n", strerror(-got));
            g_background_error = -EINVAL;
            g_background_pending = 0;
            g_background_busy = false;
            return 0;
        }
        const ds4_tag *tag = (const ds4_tag *)cqe->user_data;
        if (tag) cqe_result(cqe, *tag, "background", &g_background_error);
        io_uring_cqe_seen(&g_background.ring, cqe);
        g_background_pending--;
    }
    if (bytes) *bytes = g_background_bytes;
    if (ms) *ms = wall_ms() - g_background_started;
    g_background_busy = false;
    return g_background_error == 0 ? 1 : 0;
}

extern "C" int ds4_ds_background_finish(void) {
    if (!g_background_busy) return g_background_error == 0;
    while (g_background_pending > 0) {
        io_uring_cqe *cqe = nullptr;
        const int got = io_uring_wait_cqe(&g_background.ring, &cqe);
        if (got < 0) {
            report("ds4: background completion wait failed: %s\n", strerror(-got));
            g_background_error = -EINVAL;
            g_background_pending = 0;
            g_background_busy = false;
            break;
        }
        const ds4_tag *tag = (const ds4_tag *)cqe->user_data;
        if (tag) cqe_result(cqe, *tag, "background", &g_background_error);
        io_uring_cqe_seen(&g_background.ring, cqe);
        g_background_pending--;
    }
    g_background_busy = false;
    return g_background_error == 0;
}

extern "C" void ds4_ds_get_byte_counters(uint64_t *demand, double *demand_ms, uint64_t *background) {
    if (demand) *demand = g_total_bytes;
    if (demand_ms) *demand_ms = g_total_ms;
    if (background) *background = g_background_bytes;
}

extern "C" int ds4_ds_selftest(int fd, uint64_t start_offset, uint64_t span_bytes,
                               uint32_t chunk_bytes, uint32_t batch) {
    if (fd < 0 || !span_bytes || !chunk_bytes) return 0;
    uint64_t dev = 0, ino = 0, size = 0;
    if (!file_identity(fd, dev, ino, size) || start_offset + span_bytes > size) {
        report("ds4: selftest range %.2fMiB at %.2fMiB outside the file\n",
               (double)span_bytes / (1024.0 * 1024.0),
               (double)start_offset / (1024.0 * 1024.0));
        return 0;
    }
    char *reference = nullptr;
    if (posix_memalign((void **)&reference, 4096, span_bytes) != 0 || !reference) {
        report("ds4: selftest reference alloc %.2fMiB failed\n",
               (double)span_bytes / (1024.0 * 1024.0));
        return 0;
    }
    uint64_t done = 0;
    while (done < span_bytes) {
        const ssize_t got = pread(fd, reference + done, span_bytes - done, start_offset + done);
        if (got <= 0) {
            report("ds4: selftest pread FAILED at %.2fMiB: %s\n",
                   (double)(start_offset + done) / (1024.0 * 1024.0), strerror(errno));
            free(reference);
            return 0;
        }
        done += (uint64_t)got;
    }

    char *buffer = nullptr;
    if (posix_memalign((void **)&buffer, 4096, span_bytes) != 0 || !buffer) {
        report("ds4: selftest buffer alloc %.2fMiB failed\n",
               (double)span_bytes / (1024.0 * 1024.0));
        free(reference);
        return 0;
    }

    if (!ds4_ds_init_from_fd(fd)) {
        free(reference);
        free(buffer);
        return 0;
    }

    /* Leg 1: one request per submit, queue depth 1. Leg 2: `batch` requests per submit. */
    for (int leg = 0; leg < 2; ++leg) {
        const uint32_t per_submit = leg == 0 ? 1 : batch;
        const double t0 = wall_ms();
        uint64_t walked = 0;
        bool ok = true;
        while (walked < span_bytes && ok) {
            if (leg == 0) {
                const uint32_t chunk = chunk_bytes < span_bytes - walked
                    ? chunk_bytes : (uint32_t)(span_bytes - walked);
                ds4_ds_batch_reset();
                if (!ds4_ds_batch_add(start_offset + walked, chunk, buffer + walked)) { ok = false; break; }
                ok = ds4_ds_batch_submit_wait(nullptr, nullptr, nullptr) == 1;
                walked += chunk;
            } else {
                ds4_ds_batch_reset();
                uint32_t issued = 0;
                while (issued < per_submit && walked < span_bytes) {
                    const uint32_t chunk = chunk_bytes < span_bytes - walked
                        ? chunk_bytes : (uint32_t)(span_bytes - walked);
                    if (!ds4_ds_batch_add(start_offset + walked, chunk, buffer + walked)) { ok = false; break; }
                    walked += chunk;
                    issued++;
                }
                ok = ds4_ds_batch_submit_wait(nullptr, nullptr, nullptr) == 1;
            }
        }
        const double leg_ms = wall_ms() - t0;
        if (!ok) {
            report("ds4: selftest leg %d (%s) FAILED\n", leg + 1, leg == 0 ? "QD1" : "batched");
            free(reference);
            free(buffer);
            return 0;
        }
        if (memcmp(reference, buffer, span_bytes) != 0) {
            report("ds4: selftest leg %d (%s) MISMATCH against pread\n", leg + 1,
                   leg == 0 ? "QD1" : "batched");
            free(reference);
            free(buffer);
            return 0;
        }
        report("ds4: DS SELFTEST leg=%s requests_per_submit=%u bytes=%.2fMiB ms=%.3f "
               "rate=%.3fGiB/s matches_pread=1\n",
               leg == 0 ? "qd1" : "batched", per_submit,
               (double)span_bytes / (1024.0 * 1024.0), leg_ms,
               (double)span_bytes / (1024.0 * 1024.0 * 1024.0) / (leg_ms / 1000.0));
    }
    free(reference);
    free(buffer);
    dump_stats();
    return 1;
}

extern "C" void ds4_ds_shutdown(void) {
    dump_stats();
    release_all();
}
