#include "llama-dsv41-layer-log.h"
#include "llama-weight-fetcher.h"
#include "llama-wf-bounded-policy.h"
#include "llama-wf-zero-gaps.h"

#include <algorithm>
#include <array>
#include <memory>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Platform-specific async I/O
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  if __has_include(<liburing.h>)
#    define WF_HAS_IO_URING 1
#    include <liburing.h>
#  endif

// CUDA (optional — only if headers are actually available)
// GGML_USE_CUDA may be defined by the build system, but the CUDA toolkit
// include path is only present for ggml-cuda targets, not the llama library.
// Use __has_include to detect actual availability.
#if defined(GGML_USE_CUDA) && __has_include(<cuda_runtime.h>)
#  define WF_HAS_CUDA 1
#  include <cuda_runtime.h>
#  include "ggml-cuda.h"  // for ggml_backend_cuda_get_stream
// GDS / cuFile (optional)
#  if __has_include(<cufile.h>)
#    define WF_HAS_GDS 1
#    include <cufile.h>
#  endif
#endif

// DirectStorage + D3D12-CUDA interop (Windows only, requires CUDA + DirectStorage SDK)

static bool wf_verbose = false;  // set to true when trace is enabled
#define WF_ERR(fmt, ...)   fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__)
#define WF_LOG(fmt, ...)   do { if (wf_verbose) fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__); } while(0)
#define WF_TRACE(wf, fmt, ...) do { if ((wf)->params.trace) fprintf(stderr, "[weight-fetcher] " fmt "\n", ##__VA_ARGS__); } while(0)

#ifdef WF_HAS_CUDA
static int wf_resolve_cuda_device_index(void * gpu_backend) {
    if (!gpu_backend) {
        return -1;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device((ggml_backend_t) gpu_backend);
    if (!dev) {
        return -1;
    }

    const char * dev_name = ggml_backend_dev_name(dev);
    if (dev_name) {
        const size_t prefix_len = strlen(GGML_CUDA_NAME);
        if (strncmp(dev_name, GGML_CUDA_NAME, prefix_len) == 0) {
            char * end = nullptr;
            long idx = strtol(dev_name + prefix_len, &end, 10);
            if (end && *end == '\0' && idx >= 0 && idx < ggml_backend_cuda_get_device_count()) {
                return (int) idx;
            }
        }
    }

    const char * dev_desc = ggml_backend_dev_description(dev);
    if (dev_desc) {
        char cuda_desc[256] = {};
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            ggml_backend_cuda_get_device_description(i, cuda_desc, sizeof(cuda_desc));
            if (strcmp(dev_desc, cuda_desc) == 0) {
                return i;
            }
        }
    }

    return -1;
}

struct wf_cuda_device_scope {
    int  prev_device = -1;
    bool restore     = false;
    bool ok          = false;

    explicit wf_cuda_device_scope(int target_device) {
        if (target_device < 0) {
            return;
        }

        cudaError_t err_prev = cudaGetDevice(&prev_device);
        if (err_prev != cudaSuccess) {
            prev_device = -1;
        }

        if (prev_device == target_device) {
            ok = true;
            return;
        }

        cudaError_t err = cudaSetDevice(target_device);
        if (err == cudaSuccess) {
            ok = true;
            restore = prev_device >= 0;
        } else {
            WF_ERR("cudaSetDevice(%d) failed: %s", target_device, cudaGetErrorString(err));
        }
    }

    ~wf_cuda_device_scope() {
        if (restore) {
            cudaSetDevice(prev_device);
        }
    }
};
#endif

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

// One tensor's worth of fetch metadata
struct wf_span_internal {
    char         tensor_name[128];
    uint64_t     file_offset;
    size_t       nbytes;
    void *       mmap_addr;          // fallback pointer (may be NULL)
    int          file_idx;
    bool         is_expert;
    int          expert_idx;
    size_t       gpu_offset;         // packed offset within layer's GPU buffer
};

struct wf_cache_key {
    int      layer_idx;
    int      expert_idx;
    uint64_t tensor_hash;

    bool operator==(const wf_cache_key & o) const {
        return layer_idx == o.layer_idx && expert_idx == o.expert_idx && tensor_hash == o.tensor_hash;
    }
};

struct wf_cache_key_hash {
    size_t operator()(const wf_cache_key & k) const {
        uint64_t h = k.tensor_hash;
        h ^= (uint64_t)(uint32_t)k.layer_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.expert_idx + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct wf_cache_entry {
    size_t slot = 0;
    int file_idx = -1;
    uint64_t file_offset = 0;
    size_t nbytes = 0;
    size_t pins = 0;
    std::list<wf_cache_key>::iterator lru_it;
};

struct wf_cache_tier {
    void * arena = nullptr;
    size_t arena_bytes = 0;
    size_t slot_size = 0;
    size_t slot_count = 0;
    std::vector<bool> used;
    std::list<wf_cache_key> lru; // front = MRU, back = LRU
    std::unordered_map<wf_cache_key, wf_cache_entry, wf_cache_key_hash> entries;
    std::mutex mutex;
};

// Per-layer state
struct wf_layer {
    std::vector<wf_span_internal> spans;
    size_t                        total_bytes = 0;

    // Pipeline tracking
    std::atomic<wf_layer_stage>   stage{WF_STAGE_IDLE};
    std::atomic<uint64_t>         direct_generation{0};
    int                           gpu_buf_idx = -1;    // which ping-pong buffer (-1 = none)
    int                           ram_buf_idx = -1;    // which RAM staging buffer (-1 = none)

    // MoE: active expert mask (empty = fetch all)
    std::vector<int>              active_experts;
    bool                          expert_filter_set = false;
    bool                          skipped = false; // immutable while this pass runs
    bool                          persistent = false;          // LLAMADECODE B1b
    std::atomic<bool>             persistent_loaded{false};    // bytes resident in its bank region
    bool                          cache_inserted = false;
    std::vector<bool>             cache_hits;
    std::vector<int>              expert_prefetch_ids;
    std::vector<bool>             expert_prefetch_loaded_spans;
    bool                          expert_prefetch_planned = false;
    bool                          expert_prefetch_loaded = false;

    // Timing
    float                         nvme_read_ms  = 0.0f;
    float                         staging_ms    = 0.0f;
    float                         h2d_ms        = 0.0f;

    wf_layer() = default;
    wf_layer(wf_layer && o) noexcept
        : spans(std::move(o.spans)), total_bytes(o.total_bytes),
          stage(o.stage.load()), direct_generation(o.direct_generation.load()), gpu_buf_idx(o.gpu_buf_idx), ram_buf_idx(o.ram_buf_idx),
          active_experts(std::move(o.active_experts)), expert_filter_set(o.expert_filter_set),
          skipped(o.skipped), persistent(o.persistent), persistent_loaded(o.persistent_loaded.load()),
          cache_inserted(o.cache_inserted), cache_hits(std::move(o.cache_hits)),
          expert_prefetch_ids(std::move(o.expert_prefetch_ids)),
          expert_prefetch_loaded_spans(std::move(o.expert_prefetch_loaded_spans)),
          expert_prefetch_planned(o.expert_prefetch_planned),
          expert_prefetch_loaded(o.expert_prefetch_loaded),
          nvme_read_ms(o.nvme_read_ms), staging_ms(o.staging_ms), h2d_ms(o.h2d_ms) {}
    wf_layer & operator=(wf_layer &&) = delete;
    wf_layer(const wf_layer &) = delete;
    wf_layer & operator=(const wf_layer &) = delete;
};

// Ring buffer slot for GPU or RAM
struct wf_buffer_slot {
    void *  ptr       = nullptr;       // GPU device ptr or pinned host ptr
    size_t  capacity  = 0;
    int     layer_idx = -1;            // which layer occupies this slot (-1 = free)
    bool    owns_ptr  = false;         // true when WF allocated ptr itself
    std::atomic<bool> in_use{false};
#ifdef WF_HAS_CUDA
    cudaEvent_t upload_done  = nullptr; // signalled when H2D is complete
    cudaEvent_t compute_done = nullptr; // signalled when GPU compute finishes
#endif

    wf_buffer_slot() = default;
    wf_buffer_slot(wf_buffer_slot && o) noexcept
        : ptr(o.ptr), capacity(o.capacity), layer_idx(o.layer_idx), owns_ptr(o.owns_ptr), in_use(o.in_use.load())
#ifdef WF_HAS_CUDA
        , upload_done(o.upload_done), compute_done(o.compute_done)
#endif
    {}
    wf_buffer_slot(const wf_buffer_slot &) = delete;
    wf_buffer_slot & operator=(const wf_buffer_slot &) = delete;
};

// File handle (platform-abstracted)
struct wf_file_handle {
    int      fd = -1;
    size_t   file_size = 0;
};

// One per in-flight read: where in the file to read, how many bytes, where they land.
struct wf_io_chunk {
    int        layer_idx;
    int        span_idx;
    int        file_idx;               // which file handle to read from
    uint64_t   file_offset;            // absolute file offset for this chunk
    size_t     nbytes;                 // bytes requested for this chunk
    void *     dest;                   // destination in RAM staging buffer
    bool       pooled;                 // true if from chunk_pool (don't delete)
};

// Per-layer I/O completion tracking
struct wf_layer_io_tracker {
    std::atomic<int>   outstanding{0};    // chunks still in flight
    std::atomic<int>   total_chunks{0};   // total chunks submitted
    std::atomic<int>   completed{0};      // chunks finished
    std::atomic<size_t> bytes_done{0};    // bytes completed so far
    std::chrono::high_resolution_clock::time_point t_start;

    wf_layer_io_tracker() = default;
    wf_layer_io_tracker(wf_layer_io_tracker && o) noexcept
        : outstanding(o.outstanding.load()), total_chunks(o.total_chunks.load()),
          completed(o.completed.load()), bytes_done(o.bytes_done.load()),
          t_start(o.t_start) {}
    wf_layer_io_tracker & operator=(wf_layer_io_tracker &&) = delete;
    wf_layer_io_tracker(const wf_layer_io_tracker &) = delete;
    wf_layer_io_tracker & operator=(const wf_layer_io_tracker &) = delete;
};

// ---------------------------------------------------------------------------
// Chunk pool — avoids per-chunk heap allocation on the hot path
// ---------------------------------------------------------------------------

struct wf_chunk_pool {
    std::vector<wf_io_chunk>      storage;   // pre-allocated chunk objects
    std::vector<wf_io_chunk *>    free_list;
    std::mutex                    mu;

    void init(int capacity) {
        storage.resize(capacity);
        free_list.reserve(capacity);
        for (int i = capacity - 1; i >= 0; i--) {
            free_list.push_back(&storage[i]);
        }
    }

    wf_io_chunk * acquire() {
        std::lock_guard<std::mutex> lock(mu);
        if (free_list.empty()) return nullptr;
        auto * c = free_list.back();
        free_list.pop_back();
        return c;
    }

    void release(wf_io_chunk * c) {
        std::lock_guard<std::mutex> lock(mu);
        free_list.push_back(c);
    }
};

// ---------------------------------------------------------------------------
// Main context
// ---------------------------------------------------------------------------

struct wf_context {
    std::atomic<uint64_t> direct_generation{0}; // never reset between passes
    wf_params                     params;
    wf_transfer_mode              effective_mode = WF_MODE_MMAP;
    // Common lifecycle/query code reads this gate on every platform. The
    // DirectStorage resources themselves remain Windows-only, but an inert
    // false field must exist for Mac receiver builds as well.
    bool                          ds_direct_mode = false;
    bool                          required_dstorage = false;
#ifdef WF_HAS_CUDA
    int                           cuda_device_index = -1;
#endif

    // Per-layer metadata
    std::vector<wf_layer>         layers;

    // Per-layer I/O tracking
    std::vector<wf_layer_io_tracker> io_trackers;

    // Buffer pools
    std::vector<wf_buffer_slot>   gpu_bufs;        // ping-pong GPU buffers
    std::vector<wf_buffer_slot>   ram_bufs;        // pinned RAM staging buffers

    // Optional expert cache. Both tiers are default-off and independently sized.
    wf_cache_tier                 expert_vram_cache;
    wf_cache_tier                 expert_ram_cache;
    size_t                        expert_cache_max_span = 0;
    size_t                        expert_cache_vram_request = 0;
    size_t                        expert_cache_ram_request = 0;
    bool                          expert_cache_initialized = false;
    uint64_t                      expert_cache_passes = 0;

    // Optional per-layer EMA expert predictor (default off).
    bool                          expert_prefetch_initialized = false;
    bool                          expert_prefetch_enabled = false;
    int                           expert_prefetch_n = 0;
    int                           expert_prefetch_dist = 2;
    uint64_t                      expert_prefetch_passes = 0;
    std::vector<std::vector<float>> expert_prefetch_ema;
    std::vector<int>              expert_prefetch_last_active;
    uint64_t                      expert_prefetch_log_predicted = 0;
    uint64_t                      expert_prefetch_log_hits = 0;
    uint64_t                      expert_prefetch_log_misses = 0;
    size_t                        expert_prefetch_log_waste = 0;
    std::mutex                    expert_prefetch_mutex;

    // Half-buffer prefetch: DMA first half of next layer while current computes.
    // Avoids CNR complexity — compute stays on full-layer fast path.
    void *                        prefetch_buf = nullptr;   // GPU buffer, half of max_layer
    size_t                        prefetch_buf_size = 0;
    int                           prefetch_layer_idx = -1;  // layer loaded in prefetch_buf (-1 = none)
    std::atomic<bool>             prefetch_ready{false};    // true when first-half DMA is complete
    std::future<void>             prefetch_future;          // async prefetch handle

    // File handles (one per split file)
    std::vector<wf_file_handle>   files;

    // Reader threads: one io_uring ring per thread when the kernel offers it,
    // pread workers otherwise.  Queue depth is per thread in both cases.
    std::vector<std::thread>      io_threads;
    std::atomic<bool>             io_shutdown{false};

    // Which reader the worker threads run: io_uring rings when the kernel
    // offers one, synchronous pread otherwise.
    bool                          use_io_uring = false;

    // Reader queue: chunks waiting to be submitted.
    std::deque<wf_io_chunk *>     io_chunk_queue;
    std::mutex                    io_queue_mutex;
    std::condition_variable       io_queue_cv;

    // Pipeline scheduler thread
    std::thread                   pipeline_thread;
    std::mutex                    pipeline_mutex;
    std::condition_variable       pipeline_cv;
    std::atomic<bool>             pipeline_running{false};
    std::atomic<int>              pipeline_next_layer{0};
    // Pass direction (reverse-ds backward streaming). Default +1/0 == forward == current inference behavior;
    // wf_start_reverse() sets dir=-1/start_layer=n_layers-1 for the training backward. See TRAINING_ROADMAP.md.
    int                           dir{+1};
    int                           start_layer{0};

    // DMA control thread — polls compute_done events and fires Submit() instantly.
    // Per-buffer pending slots allow both buffers to be serviced without blocking.
    std::thread                   dma_thread;
    struct dma_buf_slot {
        std::atomic<int>  pending_layer{-1};  // layer to DMA into this buffer (-1 = none)
        // Lock-free compute-done flag (LLAMA_WF_LOCKFREE_READY=1). Set true by a
        // cudaLaunchHostFunc callback enqueued on the compute stream AFTER the
        // layer's compute; the DMA thread polls this atomic (no CUDA driver lock)
        // instead of cudaEventQuery, so the host's kernel-launch burst no longer
        // starves the DMA's readiness check. Starts true (buffer free).
        std::atomic<bool> compute_ready{true};
        dma_buf_slot() = default;
        dma_buf_slot(dma_buf_slot && o) noexcept
            : pending_layer(o.pending_layer.load()), compute_ready(o.compute_ready.load()) {}
        dma_buf_slot(const dma_buf_slot &) = delete;
    };
    std::vector<dma_buf_slot>     dma_buf_pending;    // one per GPU buffer
    std::atomic<int>              dma_next_layer{0};  // next layer to schedule via DMA thread
    std::atomic<int>              dma_in_flight_buf{-1}; // buffer idx with DMA currently in-flight (-1 = idle)
    // LLAMADECODE_ENGRAM_EARLY=pause: transfers of stages >= hold_from wait (INT_MAX: none).
    std::atomic<int>              hold_from{INT_MAX};
    // Pre-enqueued buffer: the layer staged for the next DMA, waiting for the
    // copy stream to be free.  The pipeline thread fires it instantly on fence
    // completion, then stages the NEXT layer for zero inter-transfer gap.
    std::atomic<int>              enqueued_buf{-1};   // buf idx with enqueued requests (-1 = none)
    std::atomic<int>              enqueued_layer{-1}; // layer idx of enqueued requests

    // Chunk pool (avoids heap alloc per I/O chunk)
    wf_chunk_pool                 chunk_pool;

    // Stats — hot-path counters are atomic, cold stats under mutex
    std::atomic<size_t>           stat_bytes_read_nvme{0};
    std::atomic<int>              stat_io_ops{0};
    wf_stats                      stats;
    std::mutex                    stats_mutex;

    // Wall-clock timing for accurate bandwidth
    std::chrono::high_resolution_clock::time_point pass_start;
    std::chrono::high_resolution_clock::time_point pass_end;

    // Per-transfer DMA timing log (submit → fence complete)
    struct dma_timing_entry {
        int      layer_idx;
        int      buf_idx;
        size_t   nbytes;
        std::chrono::high_resolution_clock::time_point t_submit;
        std::chrono::high_resolution_clock::time_point t_fence;
    };
    std::vector<dma_timing_entry>  dma_timing_log;
    std::mutex                     dma_timing_mutex;

#ifdef WF_HAS_CUDA
    cudaStream_t                  copy_stream = nullptr;
    // Staging double-buffer (LLAMA_WF_STAGING_DBLBUF=1): per-half copy_stream events
    // so the wave writing half (w%2) waits for wave w's async D2D (which read that
    // half) to finish — fixes the single-staging-reuse race that corrupts the naive
    // async-D2D path.
    cudaEvent_t                   ds_stage_evt[2] = { nullptr, nullptr };
    // CUDA compute stream from ggml backend (needed because ggml-cuda uses
    // cudaStreamNonBlocking, so the null stream won't synchronize with it).
    cudaStream_t                  cuda_compute_stream = nullptr;
    // GDS
    bool                          gds_available = false;
#ifdef WF_HAS_GDS
    CUfileHandle_t                cufile_handle = nullptr;
#endif
#endif

    // Sticky fatal state for a DirectStorage queue/device. Kept outside the
    // feature guard so common wait code can reject a failed transfer in every
    // build configuration.
    std::atomic<bool>            ds_failed{false};


    // Layer-major mode: when true, skip ubatch boundary reset.
    // The caller manages layer lifecycle directly via wf_wait_layer_gpu /
    // wf_release_layer, processing all ubatches per layer before moving on.
    bool                         layer_major_mode = false;
};

// ---------------------------------------------------------------------------
// Three-tier expert cache helpers
// ---------------------------------------------------------------------------

static uint64_t wf_cache_name_hash(const char * s) {
    uint64_t h = 1469598103934665603ULL;
    for (; s && *s; ++s) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static wf_cache_key wf_cache_make_key(int layer_idx, const wf_span_internal & span) {
    return { layer_idx, span.expert_idx, wf_cache_name_hash(span.tensor_name) };
}

static bool wf_cache_content_matches(const wf_cache_entry & e, const wf_span_internal & span) {
    return e.file_idx == span.file_idx && e.file_offset == span.file_offset && e.nbytes == span.nbytes;
}

static bool wf_span_is_active_expert(const wf_context * wf, const wf_layer & layer,
                                     const wf_span_internal & span) {
    if (!span.is_expert) return false;
    if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE || !layer.expert_filter_set) return true;
    return std::find(layer.active_experts.begin(), layer.active_experts.end(), span.expert_idx)
           != layer.active_experts.end();
}

static size_t wf_cache_env_bytes(const char * name) {
    const char * value = std::getenv(name);
    if (!value || !value[0] || value[0] == '-') return 0;
    char * end = nullptr;
    unsigned long long mb = std::strtoull(value, &end, 10);
    if (end == value || (end && *end != '\0') || mb == 0) return 0;
    const unsigned long long max_mb = (unsigned long long)SIZE_MAX / (1024ULL * 1024ULL);
    return (size_t)std::min(mb, max_mb) * 1024ULL * 1024ULL;
}

static uint64_t wf_physical_ram_bytes() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    return (pages > 0 && page_size > 0) ? (uint64_t)pages * (uint64_t)page_size : 0;
}

static void wf_cache_touch_locked(wf_cache_tier & tier,
                                  std::unordered_map<wf_cache_key, wf_cache_entry, wf_cache_key_hash>::iterator it) {
    tier.lru.erase(it->second.lru_it);
    tier.lru.push_front(it->first);
    it->second.lru_it = tier.lru.begin();
}

static size_t wf_cache_take_slot_locked(wf_cache_tier & tier, wf_cache_key * evicted_key,
                                        wf_cache_entry * evicted_entry,
                                        const wf_cache_key * exclude = nullptr) {
    for (size_t i = 0; i < tier.used.size(); ++i) {
        if (!tier.used[i]) {
            tier.used[i] = true;
            return i;
        }
    }
    for (auto rit = tier.lru.rbegin(); rit != tier.lru.rend(); ++rit) {
        if (exclude && *rit == *exclude) continue;
        auto found = tier.entries.find(*rit);
        if (found == tier.entries.end() || found->second.pins != 0) continue;
        size_t slot = found->second.slot;
        if (evicted_key) *evicted_key = found->first;
        if (evicted_entry) *evicted_entry = found->second;
        tier.lru.erase(found->second.lru_it);
        tier.entries.erase(found);
        return slot;
    }
    return SIZE_MAX;
}

static void wf_cache_publish_locked(wf_cache_tier & tier, const wf_cache_key & key,
                                    const wf_span_internal & span, size_t slot) {
    auto old = tier.entries.find(key);
    if (old != tier.entries.end()) {
        tier.lru.erase(old->second.lru_it);
        tier.used[old->second.slot] = false;
        tier.entries.erase(old);
    }
    tier.used[slot] = true;
    tier.lru.push_front(key);
    tier.entries.emplace(key, wf_cache_entry{
        slot, span.file_idx, span.file_offset, span.nbytes, 0, tier.lru.begin()
    });
}

static void wf_cache_init(wf_context * wf) {
    if (!wf || wf->expert_cache_initialized) return;
    wf->expert_cache_initialized = true;
    if (wf->required_dstorage) return;
    wf->expert_cache_vram_request = wf_cache_env_bytes("LLAMA_WF_EXPERT_CACHE_MB");
    wf->expert_cache_ram_request = wf_cache_env_bytes("LLAMA_WF_EXPERT_CACHE_RAM_MB");
    if (!wf->expert_cache_vram_request && !wf->expert_cache_ram_request) return;

    for (const auto & layer : wf->layers) {
        for (const auto & span : layer.spans) {
            if (span.is_expert) wf->expert_cache_max_span = std::max(wf->expert_cache_max_span, span.nbytes);
        }
    }
    if (!wf->expert_cache_max_span) {
        WF_ERR("expert cache requested but no expert spans were registered; disabling");
        wf->expert_cache_vram_request = wf->expert_cache_ram_request = 0;
        return;
    }

    const uint64_t physical = wf_physical_ram_bytes();
    const size_t ram_limit = physical ? (size_t)std::min<uint64_t>(physical / 4, SIZE_MAX) : 0;
    if (ram_limit && wf->expert_cache_ram_request > ram_limit) {
        WF_ERR("warning: LLAMA_WF_EXPERT_CACHE_RAM_MB requests %.1f MB; clamping to 25%% of physical RAM (%.1f MB)",
               wf->expert_cache_ram_request / (1024.0 * 1024.0), ram_limit / (1024.0 * 1024.0));
        wf->expert_cache_ram_request = ram_limit;
    }

#ifdef WF_HAS_CUDA
    auto init_tier = [&](wf_cache_tier & tier, size_t requested, bool device) {
        if (!requested) return;
        tier.slot_size = wf->expert_cache_max_span;
        tier.slot_count = requested / tier.slot_size;
        if (!tier.slot_count) {
            WF_ERR("warning: expert-cache %s tier is smaller than one %.2f MB slot; disabling",
                   device ? "VRAM" : "RAM", tier.slot_size / (1024.0 * 1024.0));
            return;
        }
        tier.arena_bytes = tier.slot_count * tier.slot_size;
        cudaError_t err = device ? cudaMalloc(&tier.arena, tier.arena_bytes)
                                 : cudaMallocHost(&tier.arena, tier.arena_bytes);
        if (err != cudaSuccess) {
            WF_ERR("warning: expert-cache %s allocation of %.1f MB failed (%s); tier disabled",
                   device ? "VRAM" : "pinned-RAM", tier.arena_bytes / (1024.0 * 1024.0),
                   cudaGetErrorString(err));
            tier.arena = nullptr;
            tier.arena_bytes = tier.slot_count = 0;
            return;
        }
        tier.used.assign(tier.slot_count, false);
    };
    init_tier(wf->expert_vram_cache, wf->expert_cache_vram_request, true);
    init_tier(wf->expert_ram_cache, wf->expert_cache_ram_request, false);
#else
    WF_ERR("warning: expert cache requested in a build without CUDA; disabling");
#endif

    if (wf->expert_vram_cache.arena || wf->expert_ram_cache.arena) {
        WF_ERR("expert cache enabled: slot=%.2f MB, VRAM=%zu slots/%.1f MB, pinned-RAM=%zu slots/%.1f MB",
               wf->expert_cache_max_span / (1024.0 * 1024.0),
               wf->expert_vram_cache.slot_count, wf->expert_vram_cache.arena_bytes / (1024.0 * 1024.0),
               wf->expert_ram_cache.slot_count, wf->expert_ram_cache.arena_bytes / (1024.0 * 1024.0));
        if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE) {
            WF_ERR("warning: expert cache is enabled without WF_MOE_FETCH_ACTIVE; FETCH_ALL eviction pressure usually removes its benefit");
        }
    }
}

static bool wf_cache_enabled(const wf_context * wf) {
    return wf && (wf->expert_vram_cache.arena || wf->expert_ram_cache.arena);
}

// Read-only tier probe used by the predictor. An expert is considered cached
// only when every one of its registered weight spans is present in either tier.
static bool wf_cache_contains(wf_context * wf, int layer_idx, const wf_span_internal & span) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return false;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && wf_cache_content_matches(vi->second, span)) return true;
    auto ri = ram.entries.find(key);
    return ri != ram.entries.end() && wf_cache_content_matches(ri->second, span);
}

#ifdef WF_HAS_CUDA
static void * wf_cache_slot_ptr(const wf_cache_tier & tier, size_t slot) {
    return (char *)tier.arena + slot * tier.slot_size;
}

// Lookup in VRAM -> RAM order and copy directly to the requested destination.
// host_destination is used by the legacy RAM-staged path; CUDA stream ordering
// guarantees its later H2D cannot overtake a VRAM-hit D2H.
static bool wf_cache_lookup(wf_context * wf, int layer_idx, const wf_span_internal & span,
                            void * destination, bool host_destination, bool pin = false,
                            bool defer_copy = false) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return false;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;

    if (vram.arena) {
        std::lock_guard<std::mutex> lock(vram.mutex);
        auto it = vram.entries.find(key);
        if (it != vram.entries.end() && wf_cache_content_matches(it->second, span)) {
            void * src = wf_cache_slot_ptr(vram, it->second.slot);
            if (!defer_copy) {
                cudaMemcpyAsync(destination, src, span.nbytes,
                                host_destination ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
                                wf->copy_stream);
            }
            if (pin) ++it->second.pins;
            wf_cache_touch_locked(vram, it);
            std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
            wf->stats.cache_hits++;
            wf->stats.cache_vram_hits++;
            wf->stats.cache_bytes_saved += span.nbytes;
            return true;
        }
        if (it != vram.entries.end()) {
            vram.used[it->second.slot] = false;
            vram.lru.erase(it->second.lru_it);
            vram.entries.erase(it);
        }
    }

    if (ram.arena) {
        std::scoped_lock lock(vram.mutex, ram.mutex);
        auto it = ram.entries.find(key);
        if (it != ram.entries.end() && wf_cache_content_matches(it->second, span)) {
            void * src = wf_cache_slot_ptr(ram, it->second.slot);
            if (!defer_copy) {
                cudaMemcpyAsync(destination, src, span.nbytes,
                                host_destination ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice,
                                wf->copy_stream);
            }
            if (pin) ++it->second.pins;
            wf_cache_touch_locked(ram, it);

            // Promote RAM hits into VRAM. If VRAM is full, demote its LRU entry
            // into a different RAM slot before reusing the VRAM slot.
            if (vram.arena) {
                wf_cache_key evicted_key{};
                wf_cache_entry evicted{};
                size_t vslot = wf_cache_take_slot_locked(vram, &evicted_key, &evicted);
                if (vslot != SIZE_MAX) {
                    if (evicted.nbytes && ram.slot_count > 1) {
                        wf_cache_key dropped_key{};
                        wf_cache_entry dropped{};
                        size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped, &key);
                        if (rslot != SIZE_MAX) {
                            cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), wf_cache_slot_ptr(vram, vslot),
                                            evicted.nbytes, cudaMemcpyDeviceToHost, wf->copy_stream);
                            wf_span_internal evicted_span{};
                            evicted_span.file_idx = evicted.file_idx;
                            evicted_span.file_offset = evicted.file_offset;
                            evicted_span.nbytes = evicted.nbytes;
                            wf_cache_publish_locked(ram, evicted_key, evicted_span, rslot);
                        }
                    }
                    cudaMemcpyAsync(wf_cache_slot_ptr(vram, vslot), src, span.nbytes,
                                    cudaMemcpyHostToDevice, wf->copy_stream);
                    wf_cache_publish_locked(vram, key, span, vslot);
                }
            }
            std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
            wf->stats.cache_hits++;
            wf->stats.cache_ram_hits++;
            wf->stats.cache_bytes_saved += span.nbytes;
            return true;
        }
        if (it != ram.entries.end()) {
            ram.used[it->second.slot] = false;
            ram.lru.erase(it->second.lru_it);
            ram.entries.erase(it);
        }
    }

    std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
    wf->stats.cache_misses++;
    return false;
}

static bool wf_cache_copy_existing(wf_context * wf, int layer_idx, const wf_span_internal & span,
                                   void * destination) {
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto ri = ram.entries.find(key);
    if (ri != ram.entries.end() && ri->second.pins && wf_cache_content_matches(ri->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(ram, ri->second.slot), span.nbytes,
                        cudaMemcpyHostToDevice, wf->copy_stream);
        return true;
    }
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && wf_cache_content_matches(vi->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(vram, vi->second.slot), span.nbytes,
                        cudaMemcpyDeviceToDevice, wf->copy_stream);
        return true;
    }
    if (ri != ram.entries.end() && wf_cache_content_matches(ri->second, span)) {
        cudaMemcpyAsync(destination, wf_cache_slot_ptr(ram, ri->second.slot), span.nbytes,
                        cudaMemcpyHostToDevice, wf->copy_stream);
        return true;
    }
    return false;
}

static void wf_cache_unpin(wf_context * wf, int layer_idx, const wf_span_internal & span) {
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);
    auto vi = vram.entries.find(key);
    if (vi != vram.entries.end() && vi->second.pins) --vi->second.pins;
    auto ri = ram.entries.find(key);
    if (ri != ram.entries.end() && ri->second.pins) --ri->second.pins;
}

static void wf_cache_insert_span(wf_context * wf, int layer_idx, const wf_span_internal & span,
                                 const void * gpu_source, bool allow_ram_insert) {
    if (!wf_cache_enabled(wf) || !span.is_expert) return;
    const wf_cache_key key = wf_cache_make_key(layer_idx, span);
    auto & vram = wf->expert_vram_cache;
    auto & ram = wf->expert_ram_cache;
    std::scoped_lock lock(vram.mutex, ram.mutex);

    if (vram.arena) {
        auto existing = vram.entries.find(key);
        if (existing != vram.entries.end() && wf_cache_content_matches(existing->second, span)) {
            wf_cache_touch_locked(vram, existing);
        } else {
            wf_cache_key evicted_key{};
            wf_cache_entry evicted{};
            size_t vslot = wf_cache_take_slot_locked(vram, &evicted_key, &evicted);
            if (vslot != SIZE_MAX) {
                if (evicted.nbytes && ram.arena) {
                    wf_cache_key dropped_key{};
                    wf_cache_entry dropped{};
                    size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped);
                    if (rslot != SIZE_MAX) {
                        cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), wf_cache_slot_ptr(vram, vslot),
                                        evicted.nbytes, cudaMemcpyDeviceToHost, wf->copy_stream);
                        wf_span_internal evicted_span{};
                        evicted_span.file_idx = evicted.file_idx;
                        evicted_span.file_offset = evicted.file_offset;
                        evicted_span.nbytes = evicted.nbytes;
                        wf_cache_publish_locked(ram, evicted_key, evicted_span, rslot);
                    }
                }
                cudaMemcpyAsync(wf_cache_slot_ptr(vram, vslot), gpu_source, span.nbytes,
                                cudaMemcpyDeviceToDevice, wf->copy_stream);
                wf_cache_publish_locked(vram, key, span, vslot);
            }
        }
    }

    if (allow_ram_insert && ram.arena) {
        auto existing = ram.entries.find(key);
        if (existing == ram.entries.end() || !wf_cache_content_matches(existing->second, span)) {
            wf_cache_key dropped_key{};
            wf_cache_entry dropped{};
            size_t rslot = wf_cache_take_slot_locked(ram, &dropped_key, &dropped);
            if (rslot != SIZE_MAX) {
                cudaMemcpyAsync(wf_cache_slot_ptr(ram, rslot), gpu_source, span.nbytes,
                                cudaMemcpyDeviceToHost, wf->copy_stream);
                wf_cache_publish_locked(ram, key, span, rslot);
            }
        } else {
            wf_cache_touch_locked(ram, existing);
        }
    }
}

static void wf_cache_insert_layer(wf_context * wf, int layer_idx) {
    if (!wf_cache_enabled(wf) || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    auto & layer = wf->layers[layer_idx];
    if (layer.cache_inserted || layer.gpu_buf_idx < 0) return;
    layer.cache_inserted = true;
    void * base = wf->gpu_bufs[layer.gpu_buf_idx].ptr;
    const bool ram_idle = !wf->copy_stream || cudaStreamQuery(wf->copy_stream) == cudaSuccess;
    for (const auto & span : layer.spans) {
        if (!wf_span_is_active_expert(wf, layer, span)) continue;
        wf_cache_insert_span(wf, layer_idx, span, (char *)base + span.gpu_offset, ram_idle);
    }
    if (wf->gpu_bufs[layer.gpu_buf_idx].upload_done) {
        cudaEventRecord(wf->gpu_bufs[layer.gpu_buf_idx].upload_done, wf->copy_stream);
    }
}
#endif

// ---------------------------------------------------------------------------
// Platform I/O helpers
// ---------------------------------------------------------------------------

static bool wf_open_file(wf_file_handle * fh, const char * path, bool direct_io) {
    // Unbuffered reads on Linux require the offset, the length and the buffer to
    // all be multiples of the device's block size.  GGUF tensor data is aligned
    // to general.alignment (32 bytes by default), so the chunks this pipeline
    // builds would fail with EINVAL and complete with zero bytes.  Reads are
    // therefore buffered; the aligned variant is Phase 3 follow-up work.
    if (direct_io) {
        WF_LOG("unbuffered reads are not implemented on this platform; reading %s buffered", path);
    }
    fh->fd = open(path, O_RDONLY);
    if (fh->fd < 0) {
        WF_ERR("failed to open %s (errno %d)", path, errno);
        return false;
    }
    struct stat st;
    fstat(fh->fd, &st);
    fh->file_size = (size_t)st.st_size;
    return true;
}

static void wf_close_file(wf_file_handle * fh) {
    if (fh->fd >= 0) {
        close(fh->fd);
        fh->fd = -1;
    }
}

// ---------------------------------------------------------------------------
// Async reader
//
// Chunks reach the reader threads through one queue.  Each thread owns an
// io_uring ring and keeps io_queue_depth reads in flight on it, with the chunk
// pointer as the completion's user_data; completions update the per-layer
// counters the pipeline polls.  Where io_uring is not available - no kernel
// support, or a policy that blocks it - the same queue is served by synchronous
// pread workers, so the pipeline is unchanged either way.
// ---------------------------------------------------------------------------

// One chunk's read finished: account it, wake the pipeline if its layer is now
// complete, and hand the chunk back to the pool.
static void wf_complete_chunk(wf_context * wf, wf_io_chunk * chunk, ssize_t res, int thread_id) {
    const size_t bytes_read = res > 0 ? (size_t) res : 0;
    if (res < 0) {
        WF_ERR("read failed: layer %d span %d offset %llu: errno %d",
               chunk->layer_idx, chunk->span_idx,
               (unsigned long long) chunk->file_offset, (int) -res);
    }

    int il = chunk->layer_idx;
    if (il >= 0 && il < (int) wf->io_trackers.size()) {
        auto & tracker = wf->io_trackers[il];
        tracker.bytes_done.fetch_add(bytes_read, std::memory_order_relaxed);
        const int done = tracker.completed.fetch_add(1, std::memory_order_acq_rel) + 1;
        tracker.outstanding.fetch_sub(1, std::memory_order_relaxed);

        if (done == tracker.total_chunks.load(std::memory_order_relaxed)) {
            auto t_done = std::chrono::high_resolution_clock::now();
            float total_ms = std::chrono::duration<float, std::milli>(t_done - tracker.t_start).count();
            const size_t total_bytes = tracker.bytes_done.load();

            WF_TRACE(wf, "reader[%d]: layer %d complete: %zu bytes in %.2fms (%.2f GB/s)",
                     thread_id, il, total_bytes, total_ms,
                     total_bytes / (total_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));

            wf->layers[il].nvme_read_ms = total_ms;
            wf->pipeline_cv.notify_one();
        }
    }

    // Atomic stats - no mutex on the hot path
    wf->stat_bytes_read_nvme.fetch_add(bytes_read, std::memory_order_relaxed);
    wf->stat_io_ops.fetch_add(1, std::memory_order_relaxed);

    WF_TRACE(wf, "reader[%d]: layer %d span %d: %zu bytes at offset %llu",
             thread_id, il, chunk->span_idx, bytes_read,
             (unsigned long long) chunk->file_offset);

    if (chunk->pooled) {
        wf->chunk_pool.release(chunk);
    } else {
        delete chunk;
    }
}

// Take up to `max` queued chunks without waiting.
static size_t wf_take_chunks(wf_context * wf, std::vector<wf_io_chunk *> & out, size_t max) {
    std::lock_guard<std::mutex> lock(wf->io_queue_mutex);
    while (out.size() < max && !wf->io_chunk_queue.empty()) {
        out.push_back(wf->io_chunk_queue.front());
        wf->io_chunk_queue.pop_front();
    }
    return out.size();
}

// Baseline reader: one pread per chunk, queue depth 1 per thread.  Used when
// io_uring is unavailable, and as the comparison the io_uring gain is measured
// against.
static void wf_pread_worker(wf_context * wf, int thread_id) {
    std::vector<wf_io_chunk *> batch;
    while (!wf->io_shutdown.load(std::memory_order_relaxed)) {
        batch.clear();
        if (wf_take_chunks(wf, batch, 1) == 0) {
            std::unique_lock<std::mutex> lock(wf->io_queue_mutex);
            wf->io_queue_cv.wait(lock, [&] {
                return !wf->io_chunk_queue.empty() || wf->io_shutdown.load();
            });
            if (wf->io_shutdown.load() && wf->io_chunk_queue.empty()) return;
            continue;
        }
        wf_file_handle * fh = &wf->files[batch[0]->file_idx];
        ssize_t n = pread(fh->fd, batch[0]->dest, batch[0]->nbytes, batch[0]->file_offset);
        wf_complete_chunk(wf, batch[0], n >= 0 ? n : (ssize_t) -errno, thread_id);
    }

    // Shutdown drains what is still queued, so every layer's completion counter
    // reaches its total and nobody waits on a layer that will never finish.
    for (;;) {
        batch.clear();
        if (wf_take_chunks(wf, batch, 32) == 0) return;
        for (auto * chunk : batch) {
            wf_file_handle * fh = &wf->files[chunk->file_idx];
            ssize_t n = pread(fh->fd, chunk->dest, chunk->nbytes, chunk->file_offset);
            wf_complete_chunk(wf, chunk, n >= 0 ? n : (ssize_t) -errno, thread_id);
        }
    }
}

#ifdef WF_HAS_IO_URING
// Probe once per fetcher, so the threads are started on a reader the kernel
// actually accepts rather than discovering it inside a worker.
static bool wf_io_uring_available(void) {
    struct io_uring probe;
    if (io_uring_queue_init(64, &probe, 0) != 0) {
        return false;
    }
    io_uring_queue_exit(&probe);
    return true;
}

static void wf_io_uring_worker(wf_context * wf, int thread_id) {
    const int qd = std::max(1, wf->params.io_queue_depth);
    struct io_uring ring;
    if (io_uring_queue_init((unsigned) std::max(64, qd * 4), &ring, 0) != 0) {
        WF_ERR("io_uring_queue_init failed on reader %d (errno %d); this thread reads with pread",
               thread_id, errno);
        wf_pread_worker(wf, thread_id);
        return;
    }

    // Registered files: the SQE carries the index instead of taking a reference
    // on the fd per operation.  Optional - a kernel that refuses it still reads.
    std::vector<int> fds;
    for (auto & fh : wf->files) fds.push_back(fh.fd);
    const bool fixed_files = !fds.empty() && io_uring_register_files(&ring, fds.data(), (int) fds.size()) == 0;

    int inflight = 0;
    std::vector<wf_io_chunk *> batch;
    for (;;) {
        const bool exiting = wf->io_shutdown.load(std::memory_order_relaxed);

        // Top the ring up to this thread's queue depth.
        batch.clear();
        wf_take_chunks(wf, batch, (size_t) (qd - inflight));
        size_t taken = 0;
        for (auto * chunk : batch) {
            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, fixed_files ? chunk->file_idx : wf->files[chunk->file_idx].fd,
                               chunk->dest, (unsigned) chunk->nbytes, chunk->file_offset);
            io_uring_sqe_set_data(sqe, chunk);
            if (fixed_files) sqe->flags |= IOSQE_FIXED_FILE;
            inflight++;
            taken++;
        }
        if (taken < batch.size()) {
            // The SQ ring had no room for every chunk.  Put the rest back rather
            // than dropping them: an unsubmitted chunk never completes, and the
            // pipeline waits on that layer's completion count.
            std::lock_guard<std::mutex> lock(wf->io_queue_mutex);
            wf->io_chunk_queue.insert(wf->io_chunk_queue.begin(), batch.begin() + taken, batch.end());
        }
        if (inflight) io_uring_submit(&ring);

        if (inflight > 0) {
            struct io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(&ring, &cqe) != 0) {
                WF_ERR("io_uring_wait_cqe failed on reader %d", thread_id);
                break;
            }
            unsigned seen = 0;
            do {
                auto * chunk = static_cast<wf_io_chunk *>(io_uring_cqe_get_data(cqe));
                if (chunk) wf_complete_chunk(wf, chunk, cqe->res, thread_id);
                io_uring_cqe_seen(&ring, cqe);
                seen++;
            } while (seen < (unsigned) inflight && io_uring_peek_cqe(&ring, &cqe) == 0);
            inflight -= (int) seen;
            continue;
        }

        // Nothing queued and nothing in flight: wait for work, or leave.
        if (exiting) break;
        std::unique_lock<std::mutex> lock(wf->io_queue_mutex);
        wf->io_queue_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
            return !wf->io_chunk_queue.empty() || wf->io_shutdown.load();
        });
    }

    // Every submitted read completes; the pipeline waits on those counters.
    while (inflight > 0) {
        struct io_uring_cqe * cqe = nullptr;
        if (io_uring_wait_cqe(&ring, &cqe) != 0) break;
        auto * chunk = static_cast<wf_io_chunk *>(io_uring_cqe_get_data(cqe));
        if (chunk) wf_complete_chunk(wf, chunk, cqe->res, thread_id);
        io_uring_cqe_seen(&ring, cqe);
        inflight--;
    }
    io_uring_queue_exit(&ring);
}
#endif  // WF_HAS_IO_URING


#ifdef WF_HAS_IO_URING
// Pick the reader for this fetcher.  LLAMA_WF_IO_URING=0 forces the pread
// baseline, so the two readers can be measured against each other on the same
// file and the same geometry (Phase 8 item 3).
static bool wf_io_uring_wanted(void) {
    const char * want = std::getenv("LLAMA_WF_IO_URING");
    return !(want && want[0] == '0');
}
#endif

// One reader thread: io_uring when the fetcher selected it, pread otherwise.
// The io_uring worker falls back to pread itself if its own ring cannot start.
static void wf_io_worker(wf_context * wf, int thread_id) {
#ifdef WF_HAS_IO_URING
    if (wf->use_io_uring) {
        wf_io_uring_worker(wf, thread_id);
        return;
    }
#endif
    wf_pread_worker(wf, thread_id);
}

// ---------------------------------------------------------------------------
// Pipeline scheduler
//
// Manages the 4-stage pipeline:
//   NVMe → RAM → Pinned → GPU (or GDS: NVMe → GPU)
//
// Runs on its own thread, advancing layers through the pipeline stages.
// ---------------------------------------------------------------------------

static int wf_find_free_ram_buf(wf_context * wf) {
    for (int i = 0; i < (int)wf->ram_bufs.size(); i++) {
        if (!wf->ram_bufs[i].in_use) return i;
    }
    return -1;
}

static int wf_find_free_gpu_buf(wf_context * wf) {
    for (int i = 0; i < (int)wf->gpu_bufs.size(); i++) {
        if (!wf->gpu_bufs[i].in_use) {
#ifdef WF_HAS_CUDA
            // When skip_eval_sync is active, compute may still be running on this
            // buffer even though in_use was cleared by wf_release_layer().
            // Check the compute_done event to avoid DMA/compute race.
            if (wf->gpu_bufs[i].compute_done) {
                cudaError_t err = cudaEventQuery(wf->gpu_bufs[i].compute_done);
                if (err == cudaErrorNotReady) {
                    continue;  // GPU still using this buffer, try next
                }
                // cudaSuccess or other error — buffer is safe to reuse
            }
#endif
            return i;
        }
    }
    return -1;
}

// Submit NVMe read requests for all spans in a layer.
// Single-pass: handles mmap/skip inline, counts and submits chunks together.
// Returns total number of async chunks submitted (0 = all data was handled via mmap/memcpy).
static int wf_submit_layer_reads(wf_context * wf, int layer_idx, void * dest_buf) {
    auto & layer = wf->layers[layer_idx];
    auto & tracker = wf->io_trackers[layer_idx];
    const size_t block_size = wf->params.io_block_size;

    // Reset tracker
    tracker.outstanding.store(0);
    tracker.total_chunks.store(0);
    tracker.completed.store(0);
    tracker.bytes_done.store(0);
    tracker.t_start = std::chrono::high_resolution_clock::now();

    int total_chunks = 0;
    size_t offset = 0;
#ifdef WF_HAS_CUDA
    if (wf_cache_enabled(wf)) layer.cache_hits.assign(layer.spans.size(), false);
    else layer.cache_hits.clear();
#endif

    // Batch fallback submissions: collect chunks first, notify workers once
    std::vector<wf_io_chunk *> batch_chunks;

    for (int si = 0; si < (int)layer.spans.size(); si++) {
        auto & span = layer.spans[si];

        // MoE filtering: skip non-active experts
        if (wf->params.moe_mode == WF_MOE_FETCH_ACTIVE && span.is_expert && layer.expert_filter_set) {
            bool active = false;
            for (int eid : layer.active_experts) {
                if (eid == span.expert_idx) { active = true; break; }
            }
            if (!active) {
                memset((char *)dest_buf + offset, 0, span.nbytes);
                {
                    std::lock_guard<std::mutex> lock(wf->stats_mutex);
                    wf->stats.experts_skipped++;
                    wf->stats.expert_bytes_saved += span.nbytes;
                }
                offset += span.nbytes;
                continue;
            }
        }

#ifdef WF_HAS_CUDA
        if (span.is_expert && wf_cache_lookup(wf, layer_idx, span,
                (char *)dest_buf + offset, true, true, true)) {
            layer.cache_hits[si] = true;
            offset += span.nbytes;
            continue;
        }
#endif

        // mmap fallback: synchronous memcpy (no async I/O needed)
        if (wf->effective_mode == WF_MODE_MMAP && span.mmap_addr) {
            memcpy((char *)dest_buf + offset, span.mmap_addr, span.nbytes);
            offset += span.nbytes;
            continue;
        }

        // Submit chunks for this span
        size_t span_offset = 0;
        while (span_offset < span.nbytes) {
            size_t remaining = span.nbytes - span_offset;
            size_t chunk_size = std::min(block_size, remaining);
            // For direct I/O, round up to sector size (4096) so the OS accepts the read.
            // The buffer has padding (10% extra) so the overread is safe.
            if (wf->params.io_direct && (chunk_size & 4095)) {
                chunk_size = (chunk_size + 4095) & ~(size_t)4095;
            }

            // Try pool first, fall back to heap
            wf_io_chunk * chunk = wf->chunk_pool.acquire();
            bool pooled = (chunk != nullptr);
            if (!chunk) chunk = new wf_io_chunk();

            chunk->layer_idx     = layer_idx;
            chunk->span_idx      = si;
            chunk->file_idx      = span.file_idx;
            chunk->file_offset   = span.file_offset + span_offset;
            chunk->nbytes        = chunk_size;
            chunk->dest          = (char *)dest_buf + offset + span_offset;
            chunk->pooled        = pooled;

            batch_chunks.push_back(chunk);

            total_chunks++;
            span_offset += chunk_size;
        }

        offset += span.nbytes;
    }

    if (total_chunks == 0) return 0;

    tracker.total_chunks.store(total_chunks);
    tracker.outstanding.store(total_chunks);

    // Hand the layer's chunks to the reader threads in one lock acquisition
    if (!batch_chunks.empty()) {
        std::lock_guard<std::mutex> lock(wf->io_queue_mutex);
        for (auto * c : batch_chunks) {
            wf->io_chunk_queue.push_back(c);
        }
        wf->io_queue_cv.notify_all();  // wake all readers at once
    }

    WF_TRACE(wf, "submit_layer_reads: layer %d: %d chunks submitted (%zu bytes, block=%zuKiB)",
             layer_idx, total_chunks, layer.total_bytes, block_size / 1024);
    return total_chunks;
}

// Check if all I/O for a layer has completed
static bool wf_layer_io_done(wf_context * wf, int layer_idx) {
    auto & tracker = wf->io_trackers[layer_idx];
    return tracker.completed.load(std::memory_order_acquire) >=
           tracker.total_chunks.load(std::memory_order_relaxed);
}

// H2D transfer: RAM staging → GPU buffer
static void wf_upload_to_gpu(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int ram_idx = layer.ram_buf_idx;
    int gpu_idx = layer.gpu_buf_idx;

    if (ram_idx < 0 || gpu_idx < 0) {
        WF_ERR("upload_to_gpu: layer %d has no buffer assignment (ram=%d gpu=%d)",
               layer_idx, ram_idx, gpu_idx);
        return;
    }

    void * src = wf->ram_bufs[ram_idx].ptr;
    void * dst = wf->gpu_bufs[gpu_idx].ptr;
    size_t n   = layer.total_bytes;

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef WF_HAS_CUDA
    if (wf->copy_stream) {
        const bool has_cache_hits = layer.cache_hits.size() == layer.spans.size() &&
                std::find(layer.cache_hits.begin(), layer.cache_hits.end(), true) != layer.cache_hits.end();
        if (!has_cache_hits) {
            cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, wf->copy_stream);
        } else {
            size_t span_offset = 0;
            size_t run_start = 0;
            for (int si = 0; si < (int)layer.spans.size(); ++si) {
                const auto & span = layer.spans[si];
                if (layer.cache_hits[si]) {
                    if (span_offset > run_start) {
                        cudaMemcpyAsync((char *)dst + run_start, (char *)src + run_start,
                                        span_offset - run_start, cudaMemcpyHostToDevice, wf->copy_stream);
                    }
                    if (!wf_cache_copy_existing(wf, layer_idx, span,
                            (char *)dst + span.gpu_offset)) {
                        // Defensive fallback: the pinned entry should still exist,
                        // but staging remains a correct source if it does not.
                        cudaMemcpyAsync((char *)dst + span.gpu_offset, (char *)src + span_offset,
                                        span.nbytes, cudaMemcpyHostToDevice, wf->copy_stream);
                    }
                    wf_cache_unpin(wf, layer_idx, span);
                    run_start = span_offset + span.nbytes;
                }
                span_offset += span.nbytes;
            }
            if (n > run_start) {
                cudaMemcpyAsync((char *)dst + run_start, (char *)src + run_start,
                                n - run_start, cudaMemcpyHostToDevice, wf->copy_stream);
            }
        }
        // Record event so compute stream can wait
        if (wf->gpu_bufs[gpu_idx].upload_done) {
            cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, wf->copy_stream);
        }
    } else
#endif
    {
        // Fallback: synchronous copy (should not normally happen)
        memcpy(dst, src, n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    layer.h2d_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // NOTE: RAM buffer is NOT freed here — the async CUDA DMA may still be
    // reading from it.  The pipeline's H2D_FLIGHT → GPU_READY check will
    // free it once cudaEventQuery confirms the copy finished.

    {
        std::lock_guard<std::mutex> lock(wf->stats_mutex);
        wf->stats.bytes_uploaded_gpu += n;
        wf->stats.total_h2d_ms += layer.h2d_ms;
    }

    WF_TRACE(wf, "upload_to_gpu: layer %d: %zu bytes in %.2fms", layer_idx, n, layer.h2d_ms);
}

// Check if H2D async copy has completed (so we can free the RAM buffer safely).
// For synchronous memcpy fallback this always returns true.
static bool wf_h2d_complete(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return true;
#ifdef WF_HAS_CUDA
    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.upload_done) {
        cudaError_t err = cudaEventQuery(slot.upload_done);
        return (err == cudaSuccess);
    }
#endif
    return true;  // no CUDA or no event → memcpy was synchronous
}

#ifdef WF_HAS_GDS
// GDS path: NVMe → GPU directly
static void wf_gds_read_layer(wf_context * wf, int layer_idx) {
    auto & layer = wf->layers[layer_idx];
    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return;

    void * gpu_ptr = wf->gpu_bufs[gpu_idx].ptr;
    size_t offset = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto & span : layer.spans) {
        // MoE filtering
        if (wf->params.moe_mode == WF_MOE_FETCH_ACTIVE && span.is_expert && layer.expert_filter_set) {
            bool active = false;
            for (int eid : layer.active_experts) {
                if (eid == span.expert_idx) { active = true; break; }
            }
            if (!active) {
                offset += span.nbytes;
                continue;
            }
        }

        // cuFileRead: NVMe → GPU VRAM via DMA
        if (span.is_expert && wf_cache_lookup(wf, layer_idx, span,
                (char *)gpu_ptr + offset, false)) {
            offset += span.nbytes;
            continue;
        }

        ssize_t n = cuFileRead(wf->cufile_handle, (char *)gpu_ptr + offset,
                               span.nbytes, span.file_offset, 0);
        if (n < 0 || (size_t)n != span.nbytes) {
            WF_ERR("GDS read failed: layer %d tensor %s: expected %zu got %zd",
                   layer_idx, span.tensor_name, span.nbytes, n);
        }
        offset += span.nbytes;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    layer.nvme_read_ms = ms;

    // Record upload event (GDS writes directly to GPU)
    if (wf->gpu_bufs[gpu_idx].upload_done) {
        cudaEventRecord(wf->gpu_bufs[gpu_idx].upload_done, wf->copy_stream);
    }

    WF_TRACE(wf, "gds_read: layer %d: %zu bytes in %.2fms (%.2f GB/s)",
             layer_idx, offset, ms, offset / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));
}
#endif

// ---------------------------------------------------------------------------
// DirectStorage backend helpers
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Predictive expert prefetch
// ---------------------------------------------------------------------------

static int wf_env_positive_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return end != value && end && *end == '\0' && parsed > 0 && parsed <= INT_MAX
            ? (int) parsed : fallback;
}

static bool wf_expert_fully_cached(wf_context * wf, int layer_idx, int expert_idx) {
    bool found = false;
    for (const auto & span : wf->layers[layer_idx].spans) {
        if (!span.is_expert || span.expert_idx != expert_idx) continue;
        found = true;
        if (!wf_cache_contains(wf, layer_idx, span)) return false;
    }
    return found;
}

static void wf_expert_prefetch_init(wf_context * wf) {
    if (!wf || wf->expert_prefetch_initialized) return;
    wf->expert_prefetch_initialized = true;
    if (wf->required_dstorage) return;
    const char * gate = std::getenv("LLAMA_WF_EXPERT_PREFETCH");
    if (!(gate && gate[0] && gate[0] != '0')) return;
    if (wf->params.moe_mode != WF_MOE_FETCH_ACTIVE) {
        WF_ERR("warning: LLAMA_WF_EXPERT_PREFETCH requires WF_MOE_FETCH_ACTIVE; disabling");
        return;
    }
    WF_ERR("warning: LLAMA_WF_EXPERT_PREFETCH requires DirectStorage + CUDA; disabling");
    return;
}

static void wf_expert_prefetch_schedule(wf_context * wf, int current_layer_idx) {
    if (!wf || !wf->expert_prefetch_enabled || wf->dir != 1) return;
    const int target = current_layer_idx + wf->expert_prefetch_dist;
    if (target < 0 || target >= (int) wf->layers.size()) return;

    std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
    auto & layer = wf->layers[target];
    auto & scores = wf->expert_prefetch_ema[target];
    if (scores.empty() || layer.stage.load() != WF_STAGE_IDLE || layer.expert_filter_set) return;
    int p = wf->expert_prefetch_n;
    if (p <= 0) p = (wf->expert_prefetch_last_active[target] * 3 + 1) / 2;
    if (p <= 0) return; // no routing history for this layer yet

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(scores.size());
    for (int eid = 0; eid < (int) scores.size(); ++eid) {
        if (scores[eid] > 0.0f && !wf_expert_fully_cached(wf, target, eid)) {
            ranked.emplace_back(scores[eid], eid);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });
    if ((int) ranked.size() > p) ranked.resize(p);
    if (ranked.empty()) return;

    layer.expert_prefetch_ids.clear();
    for (const auto & item : ranked) layer.expert_prefetch_ids.push_back(item.second);
    layer.active_experts = layer.expert_prefetch_ids;
    layer.expert_filter_set = true;
    layer.expert_prefetch_planned = true;
    layer.expert_prefetch_loaded = false;
    layer.expert_prefetch_loaded_spans.clear();
    WF_TRACE(wf, "expert prefetch: layer %d planned %zu experts at distance %d",
             target, layer.expert_prefetch_ids.size(), wf->expert_prefetch_dist);
}

static void wf_expert_prefetch_update_ema(wf_context * wf, int layer_idx,
                                          const int * expert_ids, int n_active) {
    auto & scores = wf->expert_prefetch_ema[layer_idx];
    if (scores.empty()) return;
    for (float & score : scores) score *= 0.75f;
    std::vector<bool> seen(scores.size(), false);
    for (int i = 0; i < n_active; ++i) {
        int eid = expert_ids[i];
        if (eid >= 0 && eid < (int) scores.size() && !seen[eid]) {
            scores[eid] += 0.25f;
            seen[eid] = true;
        }
    }
    wf->expert_prefetch_last_active[layer_idx] = n_active;
}


// ---------------------------------------------------------------------------
// DMA control thread
//
// Polls compute_done events on ALL GPU buffers and fires DS Submit() the
// instant each buffer's compute finishes.  Per-buffer pending slots mean
// buffer A's DMA starts immediately even while buffer B's compute is still
// running — no single-thread blocking bottleneck.
//
// Flow:
//   consumer: record compute_done → set dma_buf_pending[gpu_idx] = next_layer
//   DMA thread: poll cudaEventQuery on each pending buf → enqueue + Submit()
//   consumer: continues immediately — zero blocking
// ---------------------------------------------------------------------------

#ifdef WF_HAS_CUDA
static bool wf_lockfree_ready_enabled();  // fwd decl (defined near wf_cuda_record_compute_done)
#endif


// ---------------------------------------------------------------------------
// Pipeline scheduler
//
// Runs on its own thread, managing the 4-stage pipeline.  The key insight is
// that stages are non-blocking:
//
//   IDLE → NVME_READ:  submit async I/O (returns immediately)
//   NVME_READ → RAM_READY:  poll completion counter (set by the reader threads)
//   RAM_READY → H2D_FLIGHT: cudaMemcpyAsync (returns immediately)
//   H2D_FLIGHT → GPU_READY: CUDA event (checked by compute stream, not us)
//
// The scheduler loop checks all in-flight layers each iteration and advances
// them when their current stage is done.  This keeps all pipeline stages busy
// concurrently across different layers.
// ---------------------------------------------------------------------------

static void wf_pipeline_thread_fn(wf_context * wf) {
    const int n_layers = wf->params.n_layers;
    const int depth    = wf->params.prefetch_depth;
    int loop_count = 0;
    auto heartbeat_time = std::chrono::high_resolution_clock::now();

    WF_LOG("pipeline thread entered: %d layers, depth=%d, %d RAM bufs, %d GPU bufs",
           n_layers, depth, (int)wf->ram_bufs.size(), (int)wf->gpu_bufs.size());

  try {
    while (wf->pipeline_running.load()) {
        int next = wf->pipeline_next_layer.load();
        bool any_progress = false;

        // --- Phase A: check DMA completion ---
        // Non-DirectStorage path: scan all layers for I/O completion
        {
        for (int il = 0; il < n_layers; il++) {
            auto & layer = wf->layers[il];
            wf_layer_stage stage = layer.stage.load();

            if (stage == WF_STAGE_NVME_READ) {
                if (wf_layer_io_done(wf, il)) {
                    layer.stage.store(WF_STAGE_RAM_READY);
                    any_progress = true;
                    WF_TRACE(wf, "pipeline: layer %d NVMe read complete (%.2fms)",
                             il, layer.nvme_read_ms);
                }
            }

            if (layer.stage.load() == WF_STAGE_H2D_FLIGHT) {
                if (wf_h2d_complete(wf, il)) {
                    int ram_idx = layer.ram_buf_idx;
                    if (ram_idx >= 0) {
                        wf->ram_bufs[ram_idx].in_use = false;
                        wf->ram_bufs[ram_idx].layer_idx = -1;
                        layer.ram_buf_idx = -1;
                    }
                    layer.stage.store(WF_STAGE_GPU_READY);
                    wf->stats.layers_fetched++;
                    any_progress = true;

                    WF_TRACE(wf, "pipeline: layer %d H2D complete → GPU_READY", il);
                }
            }
        }
        } // end non-DS path

        // Step 2: Upload RAM_READY layers to GPU — IN CONSUMER ORDER.
        //         The consumer processes layers sequentially (0, 1, 2, ...).
        //         With only N GPU buffers, uploading out of order can deadlock:
        //         layers K+1, K+2 fill both GPU buffers while K is still waiting,
        //         and the consumer can't reach K+1 until K is done.
        //         Fix: find the first non-consumed layer and only upload from there.
        {
            // Find the first layer that the consumer hasn't finished yet
            int first_needed = n_layers;
            for (int il = 0; il < n_layers; il++) {
                wf_layer_stage s = wf->layers[il].stage.load();
                if (s != WF_STAGE_DONE && s != WF_STAGE_COMPUTING &&
                    s != WF_STAGE_GPU_READY) {
                    first_needed = il;
                    break;
                }
            }

            // Upload at most n_gpu_bufs layers starting from first_needed
            int n_gpu = (int)wf->gpu_bufs.size();
            for (int il = first_needed; il < n_layers && il < first_needed + n_gpu; il++) {
                auto & layer = wf->layers[il];
                if (layer.stage.load() != WF_STAGE_RAM_READY) continue;

                int gpu_idx = wf_find_free_gpu_buf(wf);
                if (gpu_idx < 0) break;

                wf->gpu_bufs[gpu_idx].in_use = true;
                wf->gpu_bufs[gpu_idx].layer_idx = il;
                layer.gpu_buf_idx = gpu_idx;
                layer.stage.store(WF_STAGE_H2D_FLIGHT);

                wf_upload_to_gpu(wf, il);
                any_progress = true;

                WF_TRACE(wf, "pipeline: layer %d H2D enqueued (ram_buf=%d)", il, layer.ram_buf_idx);
            }
        }

        // --- Phase B: submit new layers into the pipeline ---
        //
        // With round-robin chunk distribution, each layer's I/O is interlocked
        // sequential.  Use fewer, larger layers to give the RAID controller
        // enough sequential runway to reach full speed.
        {
            int scan_end = std::min(n_layers, next + depth);


            for (int il = 0; il < scan_end; il++) {
                auto & layer = wf->layers[il];
                if (layer.stage.load() != WF_STAGE_IDLE) continue;

#ifdef WF_HAS_GDS
                if (wf->effective_mode == WF_MODE_GDS && wf->gds_available) {
                    int gpu_idx = wf_find_free_gpu_buf(wf);
                    if (gpu_idx < 0) break;

                    wf->gpu_bufs[gpu_idx].in_use = true;
                    wf->gpu_bufs[gpu_idx].layer_idx = il;
                    layer.gpu_buf_idx = gpu_idx;
                    layer.stage.store(WF_STAGE_NVME_READ);

                    wf_gds_read_layer(wf, il);

                    layer.stage.store(WF_STAGE_GPU_READY);
                    wf->stats.layers_fetched++;
                    any_progress = true;
                    continue;
                }
#endif

                // Standard path: submit async NVMe reads → RAM staging buffer
                int ram_idx = wf_find_free_ram_buf(wf);
                if (ram_idx < 0) break;

                wf->ram_bufs[ram_idx].in_use = true;
                wf->ram_bufs[ram_idx].layer_idx = il;
                layer.ram_buf_idx = ram_idx;
                layer.stage.store(WF_STAGE_NVME_READ);

                WF_LOG("pipeline: submitting layer %d reads (ram_buf=%d, buf=%p)",
                       il, ram_idx, wf->ram_bufs[ram_idx].ptr);

                int n_chunks = wf_submit_layer_reads(wf, il, wf->ram_bufs[ram_idx].ptr);

                WF_LOG("pipeline: layer %d submitted %d chunks", il, n_chunks);

                if (n_chunks == 0) {
                    layer.stage.store(WF_STAGE_RAM_READY);
                }

                any_progress = true;
            }
        }


        // --- Heartbeat: log pipeline state every ~1 second (detect stalls AND hangs) ---
        auto now = std::chrono::high_resolution_clock::now();
        float hb_elapsed = std::chrono::duration<float>(now - heartbeat_time).count();
        if (hb_elapsed >= 1.0f) {
            heartbeat_time = now;
            int n_idle = 0, n_nvme = 0, n_ram = 0, n_h2d = 0, n_gpu = 0, n_comp = 0, n_done = 0;
            for (int il = 0; il < n_layers; il++) {
                switch (wf->layers[il].stage.load()) {
                    case WF_STAGE_IDLE:      n_idle++; break;
                    case WF_STAGE_NVME_READ: n_nvme++; break;
                    case WF_STAGE_RAM_READY: n_ram++;  break;
                    case WF_STAGE_H2D_FLIGHT:n_h2d++;  break;
                    case WF_STAGE_GPU_READY: n_gpu++;  break;
                    case WF_STAGE_COMPUTING: n_comp++; break;
                    case WF_STAGE_DONE:      n_done++; break;
                }
            }
            // Count free buffers
            int free_ram = 0, free_gpu = 0;
            for (auto & s : wf->ram_bufs) if (!s.in_use) free_ram++;
            for (auto & s : wf->gpu_bufs) if (!s.in_use) free_gpu++;

            WF_LOG("HEARTBEAT: next=%d idle=%d nvme=%d ram=%d h2d=%d gpu=%d comp=%d done=%d | free_ram=%d free_gpu=%d progress=%s",
                   next, n_idle, n_nvme, n_ram, n_h2d, n_gpu, n_comp, n_done,
                   free_ram, free_gpu, any_progress ? "yes" : "NO");

            // Dump first few non-DONE layers for debugging
            int dumped = 0;
            for (int il = 0; il < n_layers && dumped < 4; il++) {
                wf_layer_stage s = wf->layers[il].stage.load();
                if (s != WF_STAGE_DONE && s != WF_STAGE_IDLE) {
                    auto & trk = wf->io_trackers[il];
                    WF_LOG("  layer[%d]: stage=%d ram=%d gpu=%d io=%d/%d",
                           il, (int)s, wf->layers[il].ram_buf_idx,
                           wf->layers[il].gpu_buf_idx,
                           trk.completed.load(), trk.total_chunks.load());
                    dumped++;
                }
            }
        }

        if (!any_progress) {
            loop_count++;
            // Poll aggressively when DMA is in-flight (fence check is the hot path).
            // Sleep longer when idle (no DMA, no pending work).
            int sleep_us = (wf->dma_in_flight_buf.load() >= 0) ? 5 : 100;
            std::unique_lock<std::mutex> lock(wf->pipeline_mutex);
            wf->pipeline_cv.wait_for(lock, std::chrono::microseconds(sleep_us));
        } else {
            loop_count = 0;
        }
    }
  } catch (const std::exception & e) {
    WF_ERR("PIPELINE THREAD CRASHED: %s", e.what());
  } catch (...) {
    WF_ERR("PIPELINE THREAD CRASHED: unknown exception");
  }
  WF_LOG("pipeline thread exiting");
}

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

static wf_context * wf_create_impl(const wf_params * params, bool required_dstorage) {
    if (!params || !params->model_path || params->n_layers <= 0) {
        WF_ERR("wf_create: invalid params");
        return nullptr;
    }

    if(params->ds_zero_padding_only&&(!required_dstorage||!params->ds_compute_direct||
       !params->ds_async_drain||params->n_gpu_buffers!=2)){
        WF_ERR("Padding-only zero requires the explicit strict two-bank direct profile");
        return nullptr;
    }

    if (required_dstorage) {
        WF_ERR("Required DirectStorage is not compiled in; no fallback allocations attempted");
        return nullptr;
    }

    if (std::getenv("LLAMA_WF_DS_REQUEST_MIB")) WF_ERR("[dsv41-dstorage-config] request_bytes=%zu typed_setting=1", params->io_block_size);
    // Typed profile admission happens before context/factory/heap allocation.
    // Ambient tuning still refuses in strict mode; the caller uses explicit
    // source-pinned settings rather than inheriting unbounded environment knobs.
    if ((params->ds_staging_limit_bytes || params->ds_async_drain) &&
            (!required_dstorage || !llama_wf_bounded::exact_relay_profile(
                params->max_layer_bytes,uint32_t(params->n_gpu_buffers),
                params->ds_staging_limit_bytes,params->io_block_size))) {
        WF_ERR("Required DirectStorage: unsupported typed relay profile");
        return nullptr;
    }

    if(params->ds_compute_direct&&(!required_dstorage||params->n_gpu_buffers!=2||params->n_ram_buffers||!params->ds_async_drain||params->max_layer_bytes>SIZE_MAX/8)){
        WF_ERR("Required direct compute needs exactly two explicit VRAM banks and async copy ordering");return nullptr;
    }
    wf_verbose = params->trace;

    auto * wf = new wf_context();
    wf->params = *params;
    wf->required_dstorage = required_dstorage;
    // R3 DS-DIRECT gate (default-off == today's exact behavior). Set BEFORE any mode-specific init.
    {
        const char * e = std::getenv("LLAMA_WF_DS_DIRECT");
        wf->ds_direct_mode = params->ds_compute_direct||(e && e[0] && e[0] != '0');
        if (wf->ds_direct_mode) WF_LOG("DS-DIRECT mode ENABLED (LLAMA_WF_DS_DIRECT): permanent ping-pong heaps, no staging tier");
    }
    wf->layers.resize(params->n_layers);
    wf->io_trackers.resize(params->n_layers);
    memset(&wf->stats, 0, sizeof(wf->stats));
#ifdef WF_HAS_CUDA
    wf->cuda_device_index = wf_resolve_cuda_device_index(params->gpu_backend);
    if (params->gpu_backend) {
        WF_LOG("resolved CUDA device index: %d", wf->cuda_device_index);
    }
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
    if (required_dstorage && (wf->cuda_device_index < 0 || !cuda_scope.ok)) {
        WF_ERR("Required DirectStorage: target CUDA device cannot be bound");
        delete wf;
        return nullptr;
    }
    if (required_dstorage && !ggml_backend_cuda_get_stream((ggml_backend_t)params->gpu_backend)) {
        WF_ERR("Required DirectStorage: target CUDA compute stream is unavailable");
        delete wf;
        return nullptr;
    }
#endif

    // -- Resolve transfer mode --
    wf->effective_mode = params->mode;
    if (params->mode == WF_MODE_AUTO) {
#ifdef WF_HAS_GDS
        // GPUDirect Storage is preferred only when the cuFile driver reports it can
        // serve reads on this host. The nvidia-fs module is absent on most Linux
        // installs, so AUTO must land on the io_uring reader here instead of
        // failing later inside the GDS setup.
        const CUfileError_t gds_probe = cuFileDriverOpen();
        if (gds_probe.err == CU_FILE_SUCCESS) {
            wf->effective_mode = WF_MODE_GDS;
            wf->gds_available = true;
        } else {
            WF_LOG("WF_MODE_AUTO: cuFile reports no GDS support (err %d), using the async reader",
                   (int) gds_probe.err);
            wf->effective_mode = WF_MODE_DIRECT_IO;
        }
#else
        wf->effective_mode = WF_MODE_DIRECT_IO;
#endif
    }

    // DirectStorage requires Windows + LLAMA_DIRECTSTORAGE + CUDA at compile time
    if (wf->effective_mode == WF_MODE_DSTORAGE) {
        WF_LOG("WF_MODE_DSTORAGE not compiled in — falling back to DIRECT_IO");
        wf->effective_mode = WF_MODE_DIRECT_IO;
    }

    wf->stats.effective_mode = wf->effective_mode;

    WF_LOG("transfer mode: %s",
           wf->effective_mode == WF_MODE_DSTORAGE  ? "DirectStorage (NVMe → GPU via D3D12)" :
           wf->effective_mode == WF_MODE_GDS        ? "GPUDirect Storage" :
           wf->effective_mode == WF_MODE_DIRECT_IO  ? "Direct I/O" :
           wf->effective_mode == WF_MODE_MMAP       ? "mmap (legacy)" : "unknown");

    // -- Open file handles --
    {
        wf_file_handle fh;
        bool use_direct = (wf->effective_mode == WF_MODE_DIRECT_IO) && params->io_direct;
        if (!wf_open_file(&fh, params->model_path, use_direct)) {
            delete wf;
            return nullptr;
        }
        wf->files.push_back(fh);
        WF_LOG("opened %s (%.2f GB)", params->model_path,
               fh.file_size / (1024.0 * 1024.0 * 1024.0));
    }

    // Open split files
    for (int i = 0; i < params->n_split_files && params->split_paths; i++) {
        wf_file_handle fh;
        bool use_direct = (wf->effective_mode == WF_MODE_DIRECT_IO) && params->io_direct;
        if (!wf_open_file(&fh, params->split_paths[i], use_direct)) {
            WF_ERR("warning: failed to open split file %s", params->split_paths[i]);
            if (required_dstorage) { wf_destroy(wf); return nullptr; }
        }
        wf->files.push_back(fh);
    }

    // -- Allocate GPU ping-pong buffers --
    size_t buf_size = params->max_layer_bytes;
    if (buf_size == 0) {
        WF_ERR("warning: max_layer_bytes not set, using 512 MB default");
        buf_size = 512 * 1024 * 1024;
    }
    // Padding: 10% only when raw direct I/O can overread rounded chunks.
    // DirectStorage and buffered RAM-staged reads only need allocator alignment
    // slack, which matters for large Kimi runs that fit VRAM by a narrow margin.
    if (wf->effective_mode == WF_MODE_DSTORAGE ||
        (wf->effective_mode == WF_MODE_DIRECT_IO && !params->io_direct)) {
        buf_size += std::max(buf_size / 100, (size_t)(64 * 1024));
    } else {
        buf_size += buf_size / 10;
    }
    // Round up to 64KB for D3D12 / direct I/O alignment
    buf_size = (buf_size + 65535) & ~(size_t)65535;

    WF_LOG("GPU buffer allocation: %d bufs × %.2f MB = %.2f MB total (max_layer=%.2f MB, mode=%d)",
           params->n_gpu_buffers, buf_size / (1024.0 * 1024.0),
           (buf_size * params->n_gpu_buffers) / (1024.0 * 1024.0),
           params->max_layer_bytes / (1024.0 * 1024.0), (int)wf->effective_mode);

    // The legacy env-mode prefetch relay requires one compute slot. Strict
    // direct mode retains the two requested imported compute banks.
    wf->gpu_bufs.resize(wf->ds_direct_mode&&!params->ds_compute_direct ? 1 : params->n_gpu_buffers);

    bool gpu_bufs_via_dstorage = false;

    if (!gpu_bufs_via_dstorage) {
        for (int i = 0; i < params->n_gpu_buffers; i++) {
            auto & slot = wf->gpu_bufs[i];
            slot.capacity = buf_size;

#ifdef WF_HAS_CUDA
            cudaError_t err = cudaMalloc(&slot.ptr, buf_size);
            if (err != cudaSuccess) {
                WF_ERR("cudaMalloc failed for GPU buffer %d (%.2f MB): %s",
                       i, buf_size / (1024.0 * 1024.0), cudaGetErrorString(err));
                for (int j = 0; j < i; j++) cudaFree(wf->gpu_bufs[j].ptr);
                for (auto & fh : wf->files) wf_close_file(&fh);
                delete wf;
                return nullptr;
            }
            slot.owns_ptr = true;
            if (!slot.upload_done)  cudaEventCreate(&slot.upload_done);
            if (!slot.compute_done) cudaEventCreate(&slot.compute_done);
#else
            slot.ptr = aligned_alloc(4096, buf_size);
            slot.owns_ptr = true;
#endif
            WF_LOG("GPU buffer %d: %.2f MB at %p", i, buf_size / (1024.0 * 1024.0), slot.ptr);
        }
    }

    // Prefetch buffer: allocated lazily after graph_alloc, using remaining VRAM.
    // Set desired size here; actual allocation happens in wf_allocate_prefetch().
    {
        const char * pf_env = std::getenv("LLAMA_WF_PREFETCH_PCT");
        int pf_pct = (pf_env && pf_env[0]) ? std::atoi(pf_env) : 50;
        if (pf_pct < 0) pf_pct = 0;
        if (pf_pct > 100) pf_pct = 100;
        if (pf_pct > 0 && !wf->ds_direct_mode) {  // relay path for all configs (direct path has MoE corruption)
            wf->prefetch_buf_size = ((buf_size * pf_pct / 100) + 65535) & ~(size_t)65535;
            WF_LOG("prefetch: desired %.2f MB (%d%%), will allocate after graph_alloc",
                   wf->prefetch_buf_size / (1024.0 * 1024.0), pf_pct);
        }
        // R3 DS-direct: prefetch_buf + prefetch_buf_size were already set to the FULL permanent heap
        // (HEAP bytes) by wf_allocate_dual_use_heaps inside wf_dstorage_init; do NOT shrink them here.
        if (wf->ds_direct_mode) {
            WF_LOG("prefetch: DS-direct keeps full permanent heap (%.2f MB), skipping pct-based resize",
                   wf->prefetch_buf_size / (1024.0 * 1024.0));
        }
    }

    // The retired DirectStorage path would have streamed NVMe → GPU direct with
    // no RAM buffers, chunk pool or reader threads. It is refused at create,
    // so every surviving mode allocates the staging tier below.
    bool is_dstorage = (wf->effective_mode == WF_MODE_DSTORAGE);

    // -- Allocate pinned RAM staging buffers (not needed for DirectStorage) --
    if (!is_dstorage) {
        wf->ram_bufs.resize(params->n_ram_buffers);
        for (int i = 0; i < params->n_ram_buffers; i++) {
            auto & slot = wf->ram_bufs[i];
            slot.capacity = buf_size;

#ifdef WF_HAS_CUDA
            cudaError_t err = cudaMallocHost(&slot.ptr, buf_size);
            if (err != cudaSuccess) {
                WF_ERR("cudaMallocHost failed for RAM buffer %d (%.2f MB): %s",
                       i, buf_size / (1024.0 * 1024.0), cudaGetErrorString(err));
                slot.ptr = aligned_alloc(4096, buf_size);
                if (!slot.ptr) {
                    WF_ERR("malloc also failed for RAM buffer %d", i);
                }
            } else {
                // Phase 316d (2026-05-04): track WF per-cycle pinned-host allocs.
                // This is the actual allocator that fires in the WF=on path
                // (prefill_pinned_alloc only fires when WF is off).
                fprintf(stderr, "*** WF_PIN_ALLOC: slot=%d ptr=%p +%.1fMB\n",
                        i, slot.ptr, buf_size / (1024.0 * 1024.0));
                fflush(stderr);
            }
#else
            slot.ptr = aligned_alloc(4096, buf_size);
#endif
            if (slot.ptr) {
                WF_LOG("RAM buffer %d: %.2f MB (pinned)", i, buf_size / (1024.0 * 1024.0));
            }
        }
    }

    // -- Create CUDA copy stream --
#ifdef WF_HAS_CUDA
    const cudaError_t stream_created = cudaStreamCreateWithFlags(&wf->copy_stream,
        required_dstorage&&params->ds_async_drain?cudaStreamNonBlocking:cudaStreamDefault);
    const cudaError_t event0_created = cudaEventCreate(&wf->ds_stage_evt[0]);  // staging-drain event (LLAMA_WF_ASYNC_DRAIN)
    const cudaError_t event1_created = cudaEventCreate(&wf->ds_stage_evt[1]);
    if (required_dstorage && (stream_created != cudaSuccess || event0_created != cudaSuccess || event1_created != cudaSuccess)) {
        WF_ERR("Required DirectStorage: copy stream/event initialization failed");
        wf_destroy(wf);
        return nullptr;
    }
#endif

    // -- Initialize chunk pool (not needed for DirectStorage) --
    if (!is_dstorage) {
        size_t chunks_per_layer = (buf_size + params->io_block_size - 1) / params->io_block_size;
        int pool_size = (int)(chunks_per_layer * params->prefetch_depth * 2);
        pool_size = std::max(pool_size, 256);
        wf->chunk_pool.init(pool_size);
        WF_LOG("chunk pool: %d pre-allocated (%zu per layer × %d depth)",
               pool_size, chunks_per_layer, params->prefetch_depth);
    }

    // -- Create the reader queue and the reader threads --
    if (!is_dstorage) {
        wf->io_shutdown.store(false);
        int n_io = std::max(1, params->io_threads);
#ifdef WF_HAS_IO_URING
        wf->use_io_uring = wf_io_uring_wanted() && wf_io_uring_available();
        if (wf_io_uring_wanted() && !wf->use_io_uring) {
            WF_LOG("io_uring is not available here; reading with pread");
        }
#endif
        for (int i = 0; i < n_io; i++) {
            wf->io_threads.emplace_back(wf_io_worker, wf, i);
        }
        WF_LOG("started %d %s readers (QD=%d each, block=%zuKiB)",
               n_io, wf->use_io_uring ? "io_uring" : "pread",
               std::max(1, params->io_queue_depth), params->io_block_size / 1024);
    } // !is_dstorage

    // -- GDS setup --
#ifdef WF_HAS_GDS
    if (wf->effective_mode == WF_MODE_GDS) {
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            WF_ERR("GDS driver open failed, falling back to direct I/O");
            wf->effective_mode = WF_MODE_DIRECT_IO;
            wf->gds_available = false;
        } else {
            // Register GPU buffers with GDS
            for (auto & slot : wf->gpu_bufs) {
                CUfileError_t reg_status = cuFileBufRegister(slot.ptr, slot.capacity, 0);
                if (reg_status.err != CU_FILE_SUCCESS) {
                    WF_ERR("GDS buffer registration failed, falling back");
                    wf->effective_mode = WF_MODE_DIRECT_IO;
                    wf->gds_available = false;
                    break;
                }
            }
            if (wf->gds_available) {
                WF_LOG("GDS: GPU buffers registered, DMA path active");
            }
        }
    }
#endif

    // Extract the CUDA compute stream from the GPU backend.
    // ggml-cuda uses cudaStreamNonBlocking, so we MUST wait on the actual
    // compute stream (not the null stream) for D3D12 fence synchronization.
#ifdef WF_HAS_CUDA
    if (params->gpu_backend) {
        wf->cuda_compute_stream = (cudaStream_t)ggml_backend_cuda_get_stream(
            (ggml_backend_t)params->gpu_backend);
        if (wf->cuda_compute_stream) {
            WF_LOG("CUDA compute stream: %p (non-blocking, will use for DS fence sync)",
                   (void *)wf->cuda_compute_stream);
        }
    }
#endif

    // Cold-start: warm the D3D12<->CUDA DirectStorage interop now (at load) so the
    // first real streaming-prefill fetch cannot race the driver's lazy VA/semaphore
    // wiring. Placed here so cuda_compute_stream (captured just above) and copy_stream
    // are both available; covers BOTH the cold and PERSIST-INFRA-reuse init branches
    // (both funnel through wf_create's tail) plus the per-request fence/semaphore that
    // wf_dstorage_create_stateful re-creates. Gated LLAMA_WF_WARM_INTEROP (default on).
    // The DEFINITION of wf_warm_ds_interop lives inside `#ifdef WF_HAS_DSTORAGE` (opened at
    // line 1616), but this CALL was unguarded. On Windows WF_HAS_DSTORAGE is defined so both
    // exist and MSVC is happy; on macOS the definition vanishes and the call does not, so the
    // Mac build fails with 'use of undeclared identifier'. Guard the call to match the
    // definition -- byte-identical on Windows, compilable everywhere else.

    return wf;
}

wf_context * wf_create(const wf_params * params) {
    return wf_create_impl(params, false);
}

wf_context * wf_create_required_dstorage(const wf_params * params) {
    return wf_create_impl(params, true);
}

bool wf_has_failed(const wf_context * wf) {
    return !wf || wf->ds_failed.load(std::memory_order_acquire);
}

void wf_destroy(wf_context * wf) {
    if (!wf) return;

    wf_stop(wf);

#ifdef WF_HAS_CUDA
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
#endif


    // Close files
    for (auto & fh : wf->files) wf_close_file(&fh);

    // Free GPU buffers
    {
        for (auto & slot : wf->gpu_bufs) {
#ifdef WF_HAS_CUDA
            if (slot.ptr && slot.owns_ptr) cudaFree(slot.ptr);
            slot.ptr = nullptr;
            slot.owns_ptr = false;
            if (slot.upload_done)  cudaEventDestroy(slot.upload_done);
            if (slot.compute_done) cudaEventDestroy(slot.compute_done);
            slot.upload_done  = nullptr;
            slot.compute_done = nullptr;
            slot.capacity     = 0;
            slot.layer_idx    = -1;
            slot.in_use.store(false);
#else
            if (slot.ptr && slot.owns_ptr) free(slot.ptr);
            slot.ptr = nullptr;
            slot.owns_ptr = false;
#endif
        }
    }

    // Free RAM buffers
    {
        int slot_idx = 0;
        for (auto & slot : wf->ram_bufs) {
#ifdef WF_HAS_CUDA
            if (slot.ptr) {
                // Phase 316d (2026-05-04): track WF per-cycle pinned-host frees.
                fprintf(stderr, "*** WF_PIN_FREE: slot=%d ptr=%p -%.1fMB\n",
                        slot_idx, slot.ptr, slot.capacity / (1024.0 * 1024.0));
                fflush(stderr);
                cudaFreeHost(slot.ptr);
            }
#else
            free(slot.ptr);
#endif
            slot_idx++;
        }
    }

#ifdef WF_HAS_CUDA
    if (wf->copy_stream && wf_cache_enabled(wf)) {
        cudaStreamSynchronize(wf->copy_stream);
    }
    if (wf->expert_vram_cache.arena) {
        cudaFree(wf->expert_vram_cache.arena);
        wf->expert_vram_cache.arena = nullptr;
    }
    if (wf->expert_ram_cache.arena) {
        cudaFreeHost(wf->expert_ram_cache.arena);
        wf->expert_ram_cache.arena = nullptr;
    }
    if (wf->copy_stream) cudaStreamDestroy(wf->copy_stream);
    if (wf->ds_stage_evt[0]) cudaEventDestroy(wf->ds_stage_evt[0]);
    if (wf->ds_stage_evt[1]) cudaEventDestroy(wf->ds_stage_evt[1]);
#endif

#ifdef WF_HAS_GDS
    if (wf->gds_available) {
        for (auto & slot : wf->gpu_bufs) {
            if (slot.ptr) cuFileBufDeregister(slot.ptr);
        }
        cuFileDriverClose();
    }
#endif

    delete wf;
}

void wf_register_layer(wf_context * wf, int layer_idx,
                        const wf_weight_span * spans, int n_spans,
                        size_t total_bytes) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    if (wf->required_dstorage) {
        const int span_limit = wf->params.moe_mode == WF_MOE_FETCH_ACTIVE ? 2048 : 128;
        bool valid = !wf->pipeline_running.load() && n_spans > 0 && n_spans <= span_limit && spans &&
            total_bytes > 0 && total_bytes <= wf->params.max_layer_bytes;
        size_t previous = 0;
        for (int i = 0; valid && i < n_spans; ++i) {
            const auto & s = spans[i];
            const size_t offset = s.alloc_offset ? s.alloc_offset : previous;
            const size_t allocation = s.alloc_size ? s.alloc_size : s.nbytes;
            valid = s.tensor_name && s.tensor_name[0] && !s.mmap_addr && s.nbytes > 0 &&
                s.file_idx >= 0 && size_t(s.file_idx) < wf->files.size() &&
                allocation >= s.nbytes && offset >= previous && offset <= total_bytes && allocation <= total_bytes - offset;
            if (valid) {
                const uint64_t file_bytes = wf->files[s.file_idx].file_size;
                valid = s.file_offset <= file_bytes && s.nbytes <= file_bytes - s.file_offset;
                previous = offset + allocation;
            }
        }
        if (!valid) {
            WF_ERR("Required DirectStorage: invalid physical layer registration (no mmap/host fallback permitted)");
            wf->ds_failed.store(true, std::memory_order_release);
            return;
        }
    }

    auto & layer = wf->layers[layer_idx];
    layer.total_bytes = total_bytes;
    layer.spans.resize(n_spans);

    size_t offset = 0;
    for (int i = 0; i < n_spans; i++) {
        auto & dst = layer.spans[i];
        auto & src = spans[i];
        if (src.tensor_name) {
            strncpy(dst.tensor_name, src.tensor_name, sizeof(dst.tensor_name) - 1);
            dst.tensor_name[sizeof(dst.tensor_name) - 1] = '\0';
        } else {
            dst.tensor_name[0] = '\0';
        }
        dst.file_offset = src.file_offset;
        dst.nbytes      = src.nbytes;
        dst.mmap_addr   = src.mmap_addr;
        dst.file_idx    = src.file_idx;
        dst.is_expert   = src.is_expert;
        dst.expert_idx  = src.expert_idx;
        // Use alloc_offset from ggml graph allocator when available.
        // This ensures the DMA destination matches where the adapter remap
        // will point tensors during compute (prevents data/pointer mismatch
        // for small tensors at high offsets in large MoE layers).
        dst.gpu_offset = (src.alloc_offset > 0) ? src.alloc_offset : offset;
        offset = dst.gpu_offset + ((src.alloc_size > 0) ? src.alloc_size : src.nbytes);
    }
    // TASK-A probe (LLAMA_WF_SPAN_DUMP=<il>): dump this layer's span table + the padded extent
    // vs registered total_bytes. The DS wave capacity/copy math uses total_bytes while request
    // placement uses padded gpu_offsets — any overhang is silently dropped/under-copied.
    static const int span_dump_il = []() {
        const char * e = std::getenv("LLAMA_WF_SPAN_DUMP");
        return (e && e[0] != '\0') ? atoi(e) : -1;
    }();
    if (layer_idx == span_dump_il) {
        fprintf(stderr, "[wf] SPAN_DUMP layer %d: n_spans=%d total_bytes=%zu padded_extent=%zu overhang=%lld\n",
                layer_idx, n_spans, total_bytes, offset, (long long) offset - (long long) total_bytes);
        for (int i = 0; i < n_spans; i++) {
            const auto & s = layer.spans[i];
            fprintf(stderr, "[wf]   span[%03d] gpu_off=%12zu nbytes=%11zu file=%d file_off=%12llu mmap=%p '%s'\n",
                    i, s.gpu_offset, s.nbytes, s.file_idx, (unsigned long long) s.file_offset,
                    s.mmap_addr, s.tensor_name);
        }
        fflush(stderr);
    }
}

void wf_hold_from(wf_context * wf, int first_held_stage) {
    if (!wf) return;
    wf->hold_from.store(first_held_stage, std::memory_order_release);
    wf->pipeline_cv.notify_all();
}

void wf_start(wf_context * wf) {
    if (!wf || wf->pipeline_running.load()) return;
    if (wf->required_dstorage && wf->ds_failed.load(std::memory_order_acquire)) return;

    wf_cache_init(wf);
    wf_expert_prefetch_init(wf);
    if (wf_cache_enabled(wf) && wf->expert_cache_passes == 0) wf->expert_cache_passes = 1;

    wf->pipeline_running.store(true);
    // FORWARD pass: always reset direction/start to ascending. Critical for training, where each step does
    // forward (wf_start) then backward (wf_start_reverse, dir=-1): without this reset the NEXT step's forward
    // would inherit dir=-1 and the multi-buffer recycle would advance the wrong way (forward stuck@layer 3,
    // dma_next=-1). (Single-buffer masked this — it bypasses the DMA-thread recycle via sync DMA.)
    wf->dir = +1;
    wf->start_layer = 0;
    wf->pipeline_next_layer.store(0);

    // Reset layer stages
    for (auto & layer : wf->layers) {
        layer.stage.store(WF_STAGE_IDLE);
        layer.gpu_buf_idx = -1;
        layer.ram_buf_idx = -1;
        layer.cache_inserted = false;
        layer.expert_filter_set = false;
        layer.expert_prefetch_ids.clear();
        layer.expert_prefetch_loaded_spans.clear();
        layer.expert_prefetch_planned = false;
        layer.expert_prefetch_loaded = false;
    }

    // Reset buffer pool
    for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }

    // Reset atomic stats (before DS startup so startup bytes are counted)
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);


    // Record wall-clock start
    wf->pass_start = std::chrono::high_resolution_clock::now();


    wf->pipeline_thread = std::thread(wf_pipeline_thread_fn, wf);
    WF_LOG("pipeline started (prefetch depth=%d, %d GPU bufs, %d RAM bufs, pool=%d chunks)",
           wf->params.prefetch_depth, wf->params.n_gpu_buffers, wf->params.n_ram_buffers,
           (int)wf->chunk_pool.storage.size());
}

// reverse-ds: start a BACKWARD pass that streams layers N-1..0 (training backward). Mirrors wf_start with reverse
// init (start_layer=N-1, dir=-1, ping-pong N-1/N-2, dma_next=N-3). NEW symbol — wf_start (forward/inference) is
// untouched. NOTE: requires the dir-aware DMA + pipeline threads (separate edit) to actually advance descending;
// until those land this sets up correctly but the threads still step +1. See TRAINING_ROADMAP.md.
void wf_start_reverse(wf_context * wf) {
    if (!wf || wf->pipeline_running.load()) return;
    wf_cache_init(wf);
    const int N = (int)wf->layers.size();

    wf->dir = -1;
    wf->start_layer = N - 1;
    wf->pipeline_running.store(true);
    wf->pipeline_next_layer.store(N - 1);

    for (auto & layer : wf->layers) {
        layer.stage.store(WF_STAGE_IDLE);
        layer.gpu_buf_idx = -1;
        layer.ram_buf_idx = -1;
        layer.cache_inserted = false;
    }
    for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);


    wf->pass_start = std::chrono::high_resolution_clock::now();
    wf->pipeline_thread = std::thread(wf_pipeline_thread_fn, wf);
    WF_LOG("pipeline started REVERSE (start_layer=%d, dir=-1)", N - 1);
}

void wf_request_layer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    // Advance the pipeline to cover this layer (direction-aware: forward advances up, reverse advances down)
    int current_next = wf->pipeline_next_layer.load();
    bool ahead = (wf->dir > 0) ? (layer_idx >= current_next) : (layer_idx <= current_next);
    if (ahead) {
        wf->pipeline_next_layer.store(layer_idx + wf->dir);
        wf->pipeline_cv.notify_one();
    }
}

void * wf_wait_layer_gpu(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return nullptr;


    auto & layer = wf->layers[layer_idx];

    if(layer.skipped) return nullptr; // omitted weights must never be acquired

    // Skip empty sub-layers (e.g., dense layer's MoE slot)
    if (layer.total_bytes == 0 && layer.spans.empty()) {
        layer.stage.store(WF_STAGE_DONE);
        // Return first buffer pointer — nothing will be read from it
        return wf->gpu_bufs.empty() ? nullptr : wf->gpu_bufs[0].ptr;
    }

    // ---- Ubatch boundary detection ----
    // When llama_decode() splits a batch into ubatches, each ubatch runs the
    // full forward pass through all layers.  After ubatch N finishes, all layers
    // are at WF_STAGE_DONE.  When ubatch N+1 starts, pre_layer_cb(0) calls us
    // with layer_idx==0 still at DONE.  We must reset the pipeline inline so
    // the DMA/pipeline threads can service the new pass.
    //
    // Critical ordering:
    //   1. Drain D3D12 fence (all prior DMA truly complete)
    //   2. Reset DMA thread pending slots (prevent stale submissions)
    //   3. Reset all layer/buffer state
    //   4. Set ds_layer_fence_val BEFORE setting stage to NVME_READ
    //      (pipeline thread checks fence when it sees NVME_READ)
    //   5. Submit startup DS layers
    //   6. Set dma_next_layer for the DMA thread to pick up remaining layers
    if (layer_idx == 0 && layer.stage.load() == WF_STAGE_DONE && !wf->layer_major_mode) {
        WF_ERR("ubatch boundary: resetting pipeline for new pass");

        if (wf_cache_enabled(wf)) {
            ++wf->expert_cache_passes;
            if (wf->expert_cache_passes % 100 == 0) {
                std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
                WF_ERR("expert cache @ %llu tokens: VRAM hits=%llu, RAM hits=%llu, misses=%llu, saved=%.2f GB",
                       (unsigned long long)wf->expert_cache_passes,
                       (unsigned long long)wf->stats.cache_vram_hits,
                       (unsigned long long)wf->stats.cache_ram_hits,
                       (unsigned long long)wf->stats.cache_misses,
                       wf->stats.cache_bytes_saved / (1024.0 * 1024.0 * 1024.0));
            }
        }
        if (wf->expert_prefetch_enabled) {
            ++wf->expert_prefetch_passes;
            if (wf->expert_prefetch_passes % 100 == 0) {
                std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
                const uint64_t predicted = wf->stats.expert_prefetch_predicted - wf->expert_prefetch_log_predicted;
                const uint64_t hits = wf->stats.expert_prefetch_hits - wf->expert_prefetch_log_hits;
                const uint64_t misses = wf->stats.expert_prefetch_misses - wf->expert_prefetch_log_misses;
                const size_t waste = wf->stats.expert_prefetch_wasted_bytes - wf->expert_prefetch_log_waste;
                WF_ERR("expert prefetch @ %llu tokens (last 100): predicted=%llu, hits=%llu, misses=%llu, waste=%.2f MB",
                       (unsigned long long) wf->expert_prefetch_passes,
                       (unsigned long long) predicted, (unsigned long long) hits,
                       (unsigned long long) misses, waste / (1024.0 * 1024.0));
                wf->expert_prefetch_log_predicted = wf->stats.expert_prefetch_predicted;
                wf->expert_prefetch_log_hits = wf->stats.expert_prefetch_hits;
                wf->expert_prefetch_log_misses = wf->stats.expert_prefetch_misses;
                wf->expert_prefetch_log_waste = wf->stats.expert_prefetch_wasted_bytes;
            }
        }


        // Step 2.5: Drain the CUDA compute stream — ubatch N's kernels may still
        // be in-flight reading from GPU buffers we're about to recycle.  Without
        // this, the new ubatch's DMA overwrites buffers while old kernels read them.
#ifdef WF_HAS_CUDA
        if (wf->cuda_compute_stream) {
            const cudaError_t synced = cudaStreamSynchronize(wf->cuda_compute_stream);
            if (wf->required_dstorage && synced != cudaSuccess) {
                wf->ds_failed.store(true,std::memory_order_release);return nullptr;
            }
        }
#endif

        // Step 3: Reset all layer and buffer state
        for (auto & l : wf->layers) {
            l.stage.store(WF_STAGE_IDLE);
            l.gpu_buf_idx = -1;
            l.ram_buf_idx = -1;
            l.expert_filter_set = false;
            l.cache_inserted = false;
            l.expert_prefetch_ids.clear();
            l.expert_prefetch_loaded_spans.clear();
            l.expert_prefetch_planned = false;
            l.expert_prefetch_loaded = false;
        }
        for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
        for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }

        // Reset atomic stats for new pass
        wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
        wf->stat_io_ops.store(0, std::memory_order_relaxed);
        wf->pipeline_next_layer.store(wf->start_layer);

        WF_ERR("ubatch boundary reset complete");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // If layer isn't loaded yet and we have DS waves, do synchronous DMA
    // CRITICAL: wait for ALL GPU work first. The previous layer's CUDA
    // kernels are launched asynchronously — they may still be executing
    // when we overwrite the buffer with the new layer's weights.

    // Spin-wait for GPU_READY (should be very short if pipeline is ahead)
    int wait_iters = 0;
    while (layer.stage.load() != WF_STAGE_GPU_READY &&
           layer.stage.load() != WF_STAGE_COMPUTING &&
           !wf->ds_failed.load(std::memory_order_acquire)) {
        // Request the layer if not yet in pipeline
        wf_request_layer(wf, layer_idx);
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        // Diagnostic: print status every ~500ms if stuck
        if (++wait_iters % 5000 == 0) {
            wf_layer_stage s = layer.stage.load();
            int ram = layer.ram_buf_idx;
            int gpu = layer.gpu_buf_idx;
            auto & trk = wf->io_trackers[layer_idx];
            WF_ERR("wait_layer_gpu: layer %d stuck (stage=%d, ram=%d, gpu=%d, "
                   "io=%d/%d done, %zu bytes, next=%d)",
                   layer_idx, (int)s, ram, gpu,
                   trk.completed.load(), trk.total_chunks.load(),
                   trk.bytes_done.load(),
                   wf->pipeline_next_layer.load());
            // Also dump buffer status
            for (int bi = 0; bi < (int)wf->ram_bufs.size(); bi++) {
                WF_ERR("  ram_buf[%d]: in_use=%d layer=%d",
                       bi, (int)wf->ram_bufs[bi].in_use.load(), wf->ram_bufs[bi].layer_idx);
            }
            for (int bi = 0; bi < (int)wf->gpu_bufs.size(); bi++) {
                WF_ERR("  gpu_buf[%d]: in_use=%d layer=%d",
                       bi, (int)wf->gpu_bufs[bi].in_use.load(), wf->gpu_bufs[bi].layer_idx);
            }
        }
    }

    if (wf->ds_failed.load(std::memory_order_acquire)) return nullptr;

    auto t1 = std::chrono::high_resolution_clock::now();
    float stall_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (stall_ms > 0.1f) {
        {
            std::lock_guard<std::mutex> lock(wf->stats_mutex);
            wf->stats.total_stall_ms += stall_ms;
            wf->stats.stall_count++;
        }
        WF_TRACE(wf, "wait_layer_gpu: layer %d stalled %.2fms", layer_idx, stall_ms);
    }

    layer.stage.store(WF_STAGE_COMPUTING);

    // Compute for layer L-DIST is the trigger for L's provisional expert set.
    // The ordinary DMA pipeline then resolves the existing tier lookup and
    // writes only remaining predicted spans into L's eventual ring slot.
    wf_expert_prefetch_schedule(wf, layer_idx);

    int gpu_idx = layer.gpu_buf_idx;
    if (gpu_idx < 0) return nullptr;

    // Wave-based transfer: data is already in the compute buffer
    // (wf_dstorage_transfer_layer does DS → staging → D2D → compute
    // before setting GPU_READY). No reassembly needed here.

    // Prefetch scheduling moved to wf_prefetch_next_layer (called from post_layer_cb)

    return wf->gpu_bufs[gpu_idx].ptr;
}

// Prefetch first half of the next layer into prefetch_buf.
// Called from post_layer_cb after compute completes — runs synchronously
// but the first-half DMA is "paid for" here so wf_wait_layer_gpu only
// needs the second half, saving ~50% DMA time per layer.
void wf_prefetch_next_layer(wf_context * wf, int current_layer_idx) {
    (void)wf; (void)current_layer_idx;
}

// Allocate prefetch buffer using remaining VRAM after graph_alloc.
// Call once after the streaming graph is allocated.
void wf_allocate_prefetch(wf_context * wf) {
    if (wf && wf->required_dstorage) return;
#ifdef WF_HAS_CUDA
    if (!wf || wf->prefetch_buf || wf->prefetch_buf_size == 0) return;
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);

    // Query available VRAM
    size_t cuda_free = 0, cuda_total = 0;
    cudaMemGetInfo(&cuda_free, &cuda_total);

    // Use up to 80% of remaining free VRAM, capped at desired size
    size_t max_alloc = (cuda_free * 80) / 100;
    size_t alloc_size = std::min(wf->prefetch_buf_size, max_alloc);
    alloc_size = (alloc_size + 65535) & ~(size_t)65535;

    if (alloc_size < 64 * 1024 * 1024) {
        // Less than 64 MB free — not worth it
        wf->prefetch_buf_size = 0;
        return;
    }

    cudaError_t err = cudaMalloc(&wf->prefetch_buf, alloc_size);
    if (err == cudaSuccess) {
        wf->prefetch_buf_size = alloc_size;
        fprintf(stderr, "*** PREFETCH: allocated %.2f MB (%.0f MB was free)\n",
                alloc_size / (1024.0 * 1024.0), cuda_free / (1024.0 * 1024.0));
    } else {
        wf->prefetch_buf = nullptr;
        wf->prefetch_buf_size = 0;
        fprintf(stderr, "*** PREFETCH: allocation failed (wanted %.2f MB, free=%.0f MB)\n",
                alloc_size / (1024.0 * 1024.0), cuda_free / (1024.0 * 1024.0));
    }
#else
    (void)wf;
#endif
}

void wf_free_prefetch(wf_context * wf) {
#ifdef WF_HAS_CUDA
    if (!wf || !wf->prefetch_buf) return;
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
    cudaFree(wf->prefetch_buf);
    fprintf(stderr, "*** PREFETCH: freed %.2f MB\n", wf->prefetch_buf_size / (1024.0 * 1024.0));
    wf->prefetch_buf = nullptr;
    wf->prefetch_buf_size = 0;
    wf->prefetch_layer_idx = -1;
#else
    (void)wf;
#endif
}

void wf_release_layer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;

    auto & layer = wf->layers[layer_idx];

    // Skip empty sub-layers — nothing to release
    if (layer.skipped || (layer.total_bytes == 0 && layer.spans.empty())) {
        layer.stage.store(WF_STAGE_DONE);
        // Don't advance dma_next_layer — the DMA thread handles empty layers
        // via its own skip logic when it encounters them in pending_layer.
        return;
    }

    int gpu_idx = layer.gpu_buf_idx;

    layer.stage.store(WF_STAGE_DONE);
    layer.gpu_buf_idx = -1;
    layer.ram_buf_idx = -1;


    if (gpu_idx >= 0) {
        wf->gpu_bufs[gpu_idx].in_use = false;
        wf->gpu_bufs[gpu_idx].layer_idx = -1;
    }

    // Wake pipeline scheduler — a buffer just freed up
    wf->pipeline_cv.notify_one();
}

void wf_swap_sublayer(wf_context * wf, int old_layer_idx, int new_layer_idx) {
    if (!wf || old_layer_idx < 0 || old_layer_idx >= (int)wf->layers.size()) return;
    if (new_layer_idx < 0 || new_layer_idx >= (int)wf->layers.size()) return;

    auto & old_layer = wf->layers[old_layer_idx];
    int gpu_idx = old_layer.gpu_buf_idx;

    // Mark old layer done
    old_layer.stage.store(WF_STAGE_DONE);
    old_layer.gpu_buf_idx = -1;
    old_layer.ram_buf_idx = -1;

    // Skip if new layer is empty
    if (wf->layers[new_layer_idx].total_bytes == 0) {
        wf->layers[new_layer_idx].stage.store(WF_STAGE_DONE);
        if (gpu_idx >= 0) {
            wf->gpu_bufs[gpu_idx].in_use = false;
            wf->gpu_bufs[gpu_idx].layer_idx = -1;
        }
        return;
    }


    if (gpu_idx >= 0) {
        wf->gpu_bufs[gpu_idx].in_use = false;
        wf->gpu_bufs[gpu_idx].layer_idx = -1;
    }
    wf->pipeline_cv.notify_one();
}

void wf_end_pass(wf_context * wf) {
    if (!wf) return;

    if (wf->params.post_mode == WF_POST_HANDOFF_MMAP) {
        wf_stop(wf);
        WF_LOG("pass ended — handing off to mmap for decode");
    } else {
        // KEEP_FETCHER mode: stop threads cleanly so wf_start() can relaunch.
        // Without this, pipeline_running stays true and wf_start() bails.
        wf->pipeline_running.store(false);
        wf->pipeline_cv.notify_all();

        if (wf->pipeline_thread.joinable()) {
            wf->pipeline_thread.join();
        }

        // Reset layer/buffer state for next pass
        wf->pipeline_next_layer.store(0);
        for (auto & layer : wf->layers) {
            layer.stage.store(WF_STAGE_IDLE);
            layer.gpu_buf_idx = -1;
            layer.ram_buf_idx = -1;
            layer.expert_filter_set = false;
            layer.cache_inserted = false;
        }
        for (auto & slot : wf->gpu_bufs) { slot.in_use = false; slot.layer_idx = -1; }
        for (auto & slot : wf->ram_bufs) { slot.in_use = false; slot.layer_idx = -1; }


        WF_LOG("pass ended — pipeline stopped and reset for next pass");
    }
}

void wf_stop(wf_context * wf) {
    if (!wf) return;

    // Stop pipeline thread
    wf->pipeline_running.store(false);
    wf->pipeline_cv.notify_all();


    if (wf->pipeline_thread.joinable()) {
        wf->pipeline_thread.join();
    }


    // Stop reader threads
    wf->io_shutdown.store(true);

    wf->io_queue_cv.notify_all();

    for (auto & t : wf->io_threads) {
        if (t.joinable()) t.join();
    }
    wf->io_threads.clear();
    wf->io_shutdown.store(false);

    WF_LOG("pipeline stopped");
}

void wf_quiesce(wf_context * wf) {
    if (!wf) return;

#ifdef WF_HAS_CUDA
    wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
    if (wf->copy_stream) cudaStreamSynchronize(wf->copy_stream);
    if (wf->cuda_compute_stream) cudaStreamSynchronize(wf->cuda_compute_stream);
    cudaDeviceSynchronize();
#endif
}

bool wf_set_layer_skipped(wf_context * wf, int layer_idx, bool skipped) {
    if(!wf || !wf->required_dstorage ||
       wf->pipeline_running.load() || layer_idx<0 || layer_idx>=int(wf->layers.size()) ||
       wf->layers[layer_idx].persistent) return false;
    wf->layers[layer_idx].skipped=skipped;
    return true;
}

bool wf_set_layer_persistent(wf_context * wf, int layer_idx, bool persistent) {
    if (!wf || !wf->required_dstorage || layer_idx < 0 || layer_idx >= (int)wf->layers.size() || wf->pipeline_running.load()) return false;
    auto & layer = wf->layers[layer_idx];
    if (layer.spans.empty()) return false;
    for (const auto & span : layer.spans) if (span.is_expert) return false;  // routed stages are rewritten every token
    layer.persistent = persistent;
    layer.persistent_loaded.store(false, std::memory_order_release);
    return true;
}

void wf_set_active_experts(wf_context * wf, int layer_idx,
                           const int * expert_ids, int n_active) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    if (n_active < 0 || (n_active > 0 && !expert_ids)) return;

    auto & layer = wf->layers[layer_idx];
    if (!wf->expert_prefetch_enabled) {
        // The required DMA thread reads this filter under the same lock.
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        if (n_active > 0) layer.active_experts.assign(expert_ids, expert_ids + n_active);
        else layer.active_experts.clear();
        layer.expert_filter_set = true;
        return;
    }
    auto actual_contains = [&](int eid) {
        for (int i = 0; i < n_active; ++i) if (expert_ids[i] == eid) return true;
        return false;
    };

    bool planned = false;
    {
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        planned = layer.expert_prefetch_planned;
    }
    // If DS has started consuming the provisional filter, let that bounded
    // transfer finish before replacing the vector it reads.
    while (planned) {
        wf_layer_stage stage = layer.stage.load();
        if (stage == WF_STAGE_IDLE || stage == WF_STAGE_GPU_READY ||
                stage == WF_STAGE_COMPUTING || stage == WF_STAGE_DONE) break;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    std::vector<int> misses;
    bool loaded = false;
    size_t wasted_bytes = 0;
    uint64_t hits = 0;
    uint64_t predicted = 0;
    {
        std::lock_guard<std::mutex> lock(wf->expert_prefetch_mutex);
        loaded = layer.expert_prefetch_loaded;
        if (planned && loaded) {
            predicted = layer.expert_prefetch_ids.size();
            for (int eid : layer.expert_prefetch_ids) {
                if (actual_contains(eid)) ++hits;
            }
            for (int i = 0; i < n_active; ++i) {
                if (std::find(layer.expert_prefetch_ids.begin(), layer.expert_prefetch_ids.end(), expert_ids[i]) ==
                        layer.expert_prefetch_ids.end() &&
                        std::find(misses.begin(), misses.end(), expert_ids[i]) == misses.end()) {
                    misses.push_back(expert_ids[i]);
                }
            }
            for (int si = 0; si < (int) layer.spans.size() &&
                    si < (int) layer.expert_prefetch_loaded_spans.size(); ++si) {
                const auto & span = layer.spans[si];
                if (layer.expert_prefetch_loaded_spans[si] &&
                        !actual_contains(span.expert_idx)) {
                    wasted_bytes += span.nbytes;
                }
            }
        }
        if (n_active > 0) layer.active_experts.assign(expert_ids, expert_ids + n_active);
        else layer.active_experts.clear();
        layer.expert_filter_set = true;
        layer.expert_prefetch_planned = false;
        layer.expert_prefetch_loaded = false;
        layer.expert_prefetch_ids.clear();
        layer.expert_prefetch_loaded_spans.clear();
        wf_expert_prefetch_update_ema(wf, layer_idx, expert_ids, n_active);
    }

    if (planned && loaded) {
        std::lock_guard<std::mutex> stats_lock(wf->stats_mutex);
        wf->stats.expert_prefetch_predicted += predicted;
        wf->stats.expert_prefetch_hits += hits;
        wf->stats.expert_prefetch_misses += misses.size();
        wf->stats.expert_prefetch_wasted_bytes += wasted_bytes;
    }
}

wf_moe_mode wf_get_moe_mode(wf_context * wf) {
    if (!wf) return WF_MOE_FETCH_ALL;
    return wf->params.moe_mode;
}

void * wf_get_gpu_buffer(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return nullptr;
    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return nullptr;
    return wf->gpu_bufs[gpu_idx].ptr;
}

size_t wf_get_span_gpu_offset(wf_context * wf, int layer_idx, int span_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return 0;
    auto & layer = wf->layers[layer_idx];
    if (span_idx < 0 || span_idx >= (int)layer.spans.size()) return 0;
    return layer.spans[span_idx].gpu_offset;
}

void wf_cuda_wait_upload(wf_context * wf, int layer_idx) {
#ifdef WF_HAS_CUDA
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    if (wf->ds_failed.load(std::memory_order_acquire)) return;


    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return;

    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.upload_done) {
        // Make the compute stream wait for the upload event.
        // Uses the actual ggml-cuda stream (not null — ggml uses NonBlocking).
        const cudaError_t waited = cudaStreamWaitEvent(wf->cuda_compute_stream, slot.upload_done, 0);
        if (wf->required_dstorage && waited != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }

    // Insert after upload completion is ordered. All cache transfers use the
    // WF copy stream, and the helper re-records upload_done after its copies.
    wf_cache_insert_layer(wf, layer_idx);
    if (slot.upload_done) {
        const cudaError_t waited = cudaStreamWaitEvent(wf->cuda_compute_stream, slot.upload_done, 0);
        if (wf->required_dstorage && waited != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }

    // ROUND-10 PROBE (LLAMA_DSV4_WF_CKSUM=<tsv>): post-upload GPU-vs-file checksum of the
    // indexer-related weight spans, per layer, first visit only. The round-9 audit eliminated
    // every other layer-selective mechanism; the surviving gap is upload fidelity of specific
    // layers' spans. Same-run comparison: hash the uploaded GPU bytes AND the mmap source bytes.
    static const char * wf_cksum_path = std::getenv("LLAMA_DSV4_WF_CKSUM");
    if (!wf->required_dstorage && wf_cksum_path && wf_cksum_path[0]) {
        static std::mutex cksum_mutex;
        static std::set<std::pair<int, std::string>> done_set;
        std::lock_guard<std::mutex> lk(cksum_mutex);
        auto & layer = wf->layers[layer_idx];
        if (slot.upload_done) cudaEventSynchronize(slot.upload_done);
        for (const auto & span : layer.spans) {
            if (!strstr(span.tensor_name, "indexer")) continue;
            auto key = std::make_pair(layer_idx, std::string(span.tensor_name));
            if (done_set.count(key)) continue;
            done_set.insert(key);
            std::vector<uint8_t> gpu_bytes(span.nbytes);
            const cudaError_t ce = cudaMemcpy(gpu_bytes.data(),
                    (const uint8_t *) slot.ptr + span.gpu_offset, span.nbytes, cudaMemcpyDeviceToHost);
            auto fnv = [](const uint8_t * p, size_t n) {
                uint64_t h = 1469598103934665603ull;
                for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
                return h;
            };
            const uint64_t gpu_hash  = (ce == cudaSuccess) ? fnv(gpu_bytes.data(), span.nbytes) : 0;
            const uint64_t file_hash = span.mmap_addr ? fnv((const uint8_t *) span.mmap_addr, span.nbytes) : 0;
            if (FILE * fp = fopen(wf_cksum_path, "ab+")) {
                fprintf(fp, "%d\t%s\t%zu\t%016llx\t%016llx\t%s\n",
                        layer_idx, span.tensor_name, span.nbytes,
                        (unsigned long long) gpu_hash, (unsigned long long) file_hash,
                        ce != cudaSuccess ? "COPYERR" : (span.mmap_addr == nullptr ? "NOFILE" :
                        (gpu_hash == file_hash ? "MATCH" : "MISMATCH")));
                fclose(fp);
            }
        }
    }
#else
    (void)wf; (void)layer_idx;
#endif
}

#ifdef WF_HAS_CUDA
// Lock-free readiness (LLAMA_WF_LOCKFREE_READY=1): the DMA thread polls a host
// atomic instead of cudaEventQuery, removing ~40k/s CUDA-driver-lock grabs that
// contend with the host's kernel-launch burst (the reason the per-op OP_PROFILE
// sync crutch is needed to feed the DMA). With this, compute can run ahead without
// host-pacing -> compute ~29.3s + low wf -> ~685 tps @20k target. Default off.
static bool wf_lockfree_ready_enabled() {
    static const bool e = []() {
        const char * v = std::getenv("LLAMA_WF_LOCKFREE_READY");
        return v && v[0] && v[0] != '0';
    }();
    return e;
}
static void CUDART_CB wf_compute_ready_cb(void * userData) {
    auto * flag = static_cast<std::atomic<bool> *>(userData);
    if (flag) flag->store(true, std::memory_order_release);
}
#endif

void wf_cuda_record_compute_done(wf_context * wf, int layer_idx) {
#ifdef WF_HAS_CUDA
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return;
    int gpu_idx = wf->layers[layer_idx].gpu_buf_idx;
    if (gpu_idx < 0) return;

    // Record event on the compute stream. The event fires when all prior
    // compute work completes. The DMA thread checks this event via
    // cudaEventQuery before reusing the buffer — NO CPU-side sync needed.
    //
    // Previously used cudaDeviceSynchronize() here, which blocked the CPU
    // (and DMA thread) until ALL GPU work finished. This created dead time
    // between NVMe bursts (visible as spiky transfer graphs). With events,
    // the DMA thread can start the next layer's NVMe read immediately,
    // and the GPU handles ordering internally via event queries.
    //
    // For GPU_BUFS=1 (sync path): wf_wait_layer_gpu already does
    // cudaStreamSynchronize(compute_stream) before the next D2D copy.
    // For GPU_BUFS=2 (DMA thread): the DMA thread polls cudaEventQuery.

    auto & slot = wf->gpu_bufs[gpu_idx];
    if (slot.compute_done) {
        const cudaError_t recorded = cudaEventRecord(slot.compute_done, wf->cuda_compute_stream);
        if (wf->required_dstorage && recorded != cudaSuccess) { wf->ds_failed.store(true,std::memory_order_release);return; }
    }
    if (!wf->required_dstorage && wf_lockfree_ready_enabled() && gpu_idx < (int)wf->dma_buf_pending.size()) {
        // Fires on the compute stream AFTER this layer's compute completes,
        // marking buffer gpu_idx reusable lock-free (DMA polls compute_ready).
        // cudaLaunchHostFunc is illegal during stream capture (n_ubatches==1 fast
        // path) — guard it; in capture mode the event-query path is used instead.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(wf->cuda_compute_stream, &cap);
        if (cap == cudaStreamCaptureStatusNone) {
            cudaLaunchHostFunc(wf->cuda_compute_stream, wf_compute_ready_cb,
                               &wf->dma_buf_pending[gpu_idx].compute_ready);
        } else {
            // Capturing: can't enqueue a host func. Mark ready directly so the DMA
            // (which also won't be mid-pipeline during single-ubatch capture) isn't stuck.
            wf->dma_buf_pending[gpu_idx].compute_ready.store(true, std::memory_order_release);
        }
    }
#else
    (void)wf; (void)layer_idx;
#endif
}

bool wf_cuda_get_vram_info(size_t * free_bytes, size_t * total_bytes) {
#ifdef WF_HAS_CUDA
    if (!free_bytes || !total_bytes) return false;
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return err == cudaSuccess;
#else
    (void)free_bytes; (void)total_bytes;
    return false;
#endif
}

int wf_get_n_gpu_bufs(wf_context * wf) {
    return wf ? (int)wf->gpu_bufs.size() : 0;
}

size_t wf_get_gpu_buf_capacity(wf_context * wf) {
    if (!wf || wf->gpu_bufs.empty()) return 0;
    return wf->gpu_bufs[0].capacity;
}

void * wf_get_gpu_buf_ptr(wf_context * wf, int buf_idx) {
    if (!wf || buf_idx < 0 || buf_idx >= (int)wf->gpu_bufs.size()) return nullptr;
    return wf->gpu_bufs[buf_idx].ptr;
}

int wf_get_cuda_device_index(wf_context * wf) {
#ifdef WF_HAS_CUDA
    return wf ? wf->cuda_device_index : 0;
#else
    (void)wf; return 0;
#endif
}

bool wf_is_ds_direct(wf_context * wf) {
    return wf && wf->ds_direct_mode;
}

void wf_set_gpu_buf_ptr(wf_context * wf, int buf_idx, void * ptr) {
    if (!wf || buf_idx < 0 || buf_idx >= (int)wf->gpu_bufs.size()) return;
    // Free the old cudaMalloc'd buffer if present
    if (wf->gpu_bufs[buf_idx].ptr && wf->gpu_bufs[buf_idx].ptr != ptr && wf->gpu_bufs[buf_idx].owns_ptr) {
#ifdef WF_HAS_CUDA
        wf_cuda_device_scope cuda_scope(wf->cuda_device_index);
        cudaFree(wf->gpu_bufs[buf_idx].ptr);
#else
        free(wf->gpu_bufs[buf_idx].ptr);
#endif
    }
    wf->gpu_bufs[buf_idx].ptr = ptr;
    wf->gpu_bufs[buf_idx].owns_ptr = false;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

bool wf_get_required_layer_fetch(wf_context * wf, int layer, uint64_t * bytes, double * milliseconds) {
    if (!wf || !bytes || !milliseconds) return false;
    *bytes = 0; *milliseconds = 0;
    return false;
}

void wf_get_stats(wf_context * wf, wf_stats * stats) {
    if (!wf || !stats) return;

    // Merge atomic counters into stats struct
    {
        std::lock_guard<std::mutex> lock(wf->stats_mutex);
        *stats = wf->stats;
    }

    // Overwrite with atomic hot-path counters
    stats->bytes_read_nvme = wf->stat_bytes_read_nvme.load(std::memory_order_relaxed);
    stats->io_ops          = wf->stat_io_ops.load(std::memory_order_relaxed);

    // Sum per-layer NVMe read times for total_nvme_read_ms
    // (each layer's nvme_read_ms is wall-clock from submit to last-chunk-done)
    float sum_nvme_ms = 0.0f;
    for (auto & layer : wf->layers) {
        sum_nvme_ms += layer.nvme_read_ms;
    }
    stats->total_nvme_read_ms = sum_nvme_ms;

    // Sum per-layer H2D times
    float sum_h2d_ms = 0.0f;
    for (auto & layer : wf->layers) {
        sum_h2d_ms += layer.h2d_ms;
    }
    stats->total_h2d_ms = sum_h2d_ms;

    // Wall-clock bandwidth: total bytes / wall time since wf_start()
    auto now = std::chrono::high_resolution_clock::now();
    float wall_ms = std::chrono::duration<float, std::milli>(now - wf->pass_start).count();

    if (wall_ms > 0.0f) {
        stats->nvme_read_bw_gbps = (float)stats->bytes_read_nvme /
            (wall_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
        stats->end_to_end_bw_gbps = stats->nvme_read_bw_gbps;
    }
    if (sum_h2d_ms > 0) {
        stats->ram_to_gpu_bw_gbps = (float)stats->bytes_uploaded_gpu /
            (sum_h2d_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
    }

#ifdef WF_HAS_GDS
    stats->gds_active = wf->gds_available;
#else
    stats->gds_active = false;
#endif
}

void wf_reset_stats(wf_context * wf) {
    if (!wf) return;
    std::lock_guard<std::mutex> lock(wf->stats_mutex);
    memset(&wf->stats, 0, sizeof(wf->stats));
    wf->stats.effective_mode = wf->effective_mode;
    wf->stat_bytes_read_nvme.store(0, std::memory_order_relaxed);
    wf->stat_io_ops.store(0, std::memory_order_relaxed);
    wf->expert_prefetch_log_predicted = 0;
    wf->expert_prefetch_log_hits = 0;
    wf->expert_prefetch_log_misses = 0;
    wf->expert_prefetch_log_waste = 0;
    wf->pass_start = std::chrono::high_resolution_clock::now();
}

void wf_print_stats(wf_context * wf) {
    wf_stats s;
    wf_get_stats(wf, &s);

    // wf_print_stats is explicitly called — always print
    WF_ERR("=== Weight Fetcher Stats ===");
    WF_ERR("  mode:           %s%s",
           s.effective_mode == WF_MODE_DSTORAGE  ? "DirectStorage" :
           s.effective_mode == WF_MODE_GDS       ? "GPUDirect Storage" :
           s.effective_mode == WF_MODE_DIRECT_IO ? "Direct I/O" :
           s.effective_mode == WF_MODE_MMAP      ? "mmap" : "unknown",
           s.gds_active ? " (active)" : "");
    WF_ERR("  NVMe read:      %.2f GB in %.2fms (%.2f GB/s)",
           s.bytes_read_nvme / (1024.0 * 1024.0 * 1024.0),
           s.total_nvme_read_ms, s.nvme_read_bw_gbps);
    WF_ERR("  H2D upload:     %.2f GB in %.2fms (%.2f GB/s)",
           s.bytes_uploaded_gpu / (1024.0 * 1024.0 * 1024.0),
           s.total_h2d_ms, s.ram_to_gpu_bw_gbps);
    WF_ERR("  end-to-end:     %.2f GB/s", s.end_to_end_bw_gbps);
    WF_ERR("  GPU stalls:     %d (%.2fms total)", s.stall_count, s.total_stall_ms);
    WF_ERR("  I/O ops:        %d", s.io_ops);
    WF_ERR("  layers fetched: %d", s.layers_fetched);
    if (s.experts_skipped > 0) {
        WF_ERR("  MoE experts skipped: %d (%.2f MB saved)",
               s.experts_skipped, s.expert_bytes_saved / (1024.0 * 1024.0));
    }
    if (wf_cache_enabled(wf)) {
        WF_ERR("  expert cache: VRAM hits=%llu, RAM hits=%llu, misses=%llu (%.2f GB saved)",
               (unsigned long long)s.cache_vram_hits,
               (unsigned long long)s.cache_ram_hits,
               (unsigned long long)s.cache_misses,
               s.cache_bytes_saved / (1024.0 * 1024.0 * 1024.0));
    }
    if (wf->expert_prefetch_enabled) {
        WF_ERR("  expert prefetch: predicted=%llu, hits=%llu, misses=%llu, waste=%.2f MB",
               (unsigned long long) s.expert_prefetch_predicted,
               (unsigned long long) s.expert_prefetch_hits,
               (unsigned long long) s.expert_prefetch_misses,
               s.expert_prefetch_wasted_bytes / (1024.0 * 1024.0));
    }

    // Dump per-transfer DMA timing log
    {
        std::lock_guard<std::mutex> lk(wf->dma_timing_mutex);
        if (!wf->dma_timing_log.empty()) {
            auto t0 = wf->dma_timing_log[0].t_submit;
            double total_gap_ms = 0;
            double total_xfer_ms = 0;
            int n_entries = (int)wf->dma_timing_log.size();
            int n_slow = 0;  // transfers > 1.5× expected time

            for (int i = 0; i < n_entries; i++) {
                auto & e = wf->dma_timing_log[i];
                double submit_ms = std::chrono::duration<double, std::milli>(e.t_submit - t0).count();
                double fence_ms  = e.t_fence.time_since_epoch().count() > 0
                                 ? std::chrono::duration<double, std::milli>(e.t_fence - t0).count()
                                 : -1.0;
                double xfer_ms   = fence_ms >= 0 ? (fence_ms - submit_ms) : -1.0;
                double expected_ms = e.nbytes / (20.0e9 / 1000.0);  // at 20 GB/s
                double gap_ms    = (i > 0 && wf->dma_timing_log[i-1].t_fence.time_since_epoch().count() > 0)
                                 ? std::chrono::duration<double, std::milli>(e.t_submit - wf->dma_timing_log[i-1].t_fence).count()
                                 : 0.0;

                if (xfer_ms > 0) total_xfer_ms += xfer_ms;
                if (i > 0) total_gap_ms += gap_ms;
                if (xfer_ms > expected_ms * 1.5) n_slow++;

                // Print first 10, last 5, and any slow transfers
                if (i < 10 || i >= n_entries - 5 || xfer_ms > expected_ms * 1.5) {
                    WF_ERR("  DMA[%3d] layer=%3d buf=%d %6.1f MB | submit=%8.1fms fence=%8.1fms xfer=%6.1fms (%.1f GB/s) gap=%5.1fms%s",
                           i, e.layer_idx, e.buf_idx, e.nbytes / (1024.0 * 1024.0),
                           submit_ms, fence_ms, xfer_ms,
                           xfer_ms > 0 ? (e.nbytes / (xfer_ms / 1000.0)) / 1e9 : 0.0,
                           gap_ms,
                           xfer_ms > expected_ms * 1.5 ? " *** SLOW" : "");
                } else if (i == 10) {
                    WF_ERR("  ... (%d transfers omitted, showing slow ones) ...", n_entries - 15);
                }
            }

            double avg_xfer = n_entries > 0 ? total_xfer_ms / n_entries : 0;
            double avg_gap  = n_entries > 1 ? total_gap_ms / (n_entries - 1) : 0;
            WF_ERR("  DMA summary: %d transfers, avg xfer=%.1fms, avg gap=%.1fms, %d slow (>1.5x expected)",
                   n_entries, avg_xfer, avg_gap, n_slow);
        }
    }
}

bool wf_gds_available(void) {
#ifdef WF_HAS_GDS
    CUfileError_t status = cuFileDriverOpen();
    if (status.err == CU_FILE_SUCCESS) {
        cuFileDriverClose();
        return true;
    }
#endif
    return false;
}

wf_layer_stage wf_get_layer_stage(wf_context * wf, int layer_idx) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return WF_STAGE_IDLE;
    return wf->layers[layer_idx].stage.load();
}

wf_transfer_mode wf_get_effective_mode(wf_context * wf) {
    if (!wf) return WF_MODE_MMAP;
    return wf->effective_mode;
}

const char * wf_get_io_engine(const wf_context * wf) {
    if (!wf) return "none";
    return wf->use_io_uring ? "io_uring" : "pread";
}

bool wf_get_layer_file_range(wf_context * wf, int layer_idx,
                             uint64_t * out_first_offset,
                             size_t * out_total_bytes) {
    if (!wf || layer_idx < 0 || layer_idx >= (int)wf->layers.size()) return false;
    auto & layer = wf->layers[layer_idx];
    if (out_total_bytes) *out_total_bytes = layer.total_bytes;
    // Find first non-zero file offset among spans
    uint64_t first_off = 0;
    for (auto & span : layer.spans) {
        if (span.file_offset > 0) {
            first_off = span.file_offset;
            break;
        }
    }
    if (out_first_offset) *out_first_offset = first_off;
    return true;
}

const char * wf_get_model_path(wf_context * wf) {
    if (!wf) return nullptr;
    return wf->params.model_path;
}

void wf_set_layer_major_mode(wf_context * wf, bool enabled) {
    if (!wf) return;
    wf->layer_major_mode = enabled;
    WF_ERR("layer-major mode %s", enabled ? "ENABLED" : "DISABLED");
}

bool wf_get_layer_major_mode(wf_context * wf) {
    if (!wf) return false;
    return wf->layer_major_mode;
}


int wf_persist_shutdown(const char * why) {
    (void) why;
    return 1;
}

// Demand and one optional tail-hint worker share this accounting and DS slot.
static std::mutex g_wf_extent_stats_mutex;
static struct { double submit_ms, fence_ms, verify_ms, cuda_ms; uint64_t requests, batches; } g_wf_extent_stats = {};
void wf_extent_fetch_stats(double * out4, uint64_t * requests, uint64_t * batches) {
    std::lock_guard<std::mutex> lock(g_wf_extent_stats_mutex);
    if (out4) { out4[0] = g_wf_extent_stats.submit_ms; out4[1] = g_wf_extent_stats.fence_ms; out4[2] = g_wf_extent_stats.verify_ms; out4[3] = g_wf_extent_stats.cuda_ms; }
    if (requests) *requests = g_wf_extent_stats.requests;
    if (batches) *batches = g_wf_extent_stats.batches;
}
bool wf_ready_direct_bank(wf_context * wf, int stage, int bank, uint64_t * generation) {
    if (generation) *generation=0;
    if (!wf || !generation || !wf->required_dstorage || !wf->params.ds_compute_direct ||
        wf->params.n_gpu_buffers!=2 ||
        stage<0 || stage>=int(wf->layers.size()) || bank<0 || bank>=2) return false;
    const auto&layer=wf->layers[stage];
    // The caller pins this ready stage until its worker has joined. Its DMA
    // publisher has released all ownership/generation fields via stage.store.
    if(layer.stage.load(std::memory_order_acquire)!=WF_STAGE_GPU_READY || layer.gpu_buf_idx!=bank) return false;
    *generation=layer.direct_generation.load(std::memory_order_acquire);
    return *generation!=0;
}
static bool wf_ready_generation(wf_context*wf,int stage,int bank,uint64_t generation){
    uint64_t live=0;return generation&&wf_ready_direct_bank(wf,stage,bank,&live)&&live==generation;
}
static bool wf_fetch_extents_impl(wf_context * wf, int bank, const wf_extent * extents, int n,
                             void * cuda_stream, uint64_t * bytes, double * milliseconds,
                             int ready_stage, uint64_t generation, uint64_t * deferred_fence,
                             void (*overlap)(void *) = nullptr, void * overlap_user = nullptr) {
    std::lock_guard<std::mutex> stats_lock(g_wf_extent_stats_mutex);
    if(deferred_fence)*deferred_fence=0;
    if (bytes) *bytes = 0;
    if (milliseconds) *milliseconds = 0;
    (void) wf; (void) bank; (void) extents; (void) n; (void) cuda_stream;
    (void) ready_stage; (void) generation; (void) overlap; (void) overlap_user;
    return false;
}

bool wf_fetch_extents_direct(wf_context * wf,int bank,const wf_extent * extents,int n,
        void * stream,uint64_t * bytes,double * milliseconds){
    return wf_fetch_extents_impl(wf,bank,extents,n,stream,bytes,milliseconds,-1,0,nullptr);
}
bool wf_fetch_extents_direct_overlap(wf_context * wf,int bank,const wf_extent * extents,int n,
        void * stream,uint64_t * bytes,double * milliseconds,void (*overlap)(void *),void * overlap_user){
    return wf_fetch_extents_impl(wf,bank,extents,n,stream,bytes,milliseconds,-1,0,nullptr,overlap,overlap_user);
}
bool wf_fetch_ready_extents(wf_context * wf,int stage,int bank,uint64_t generation,
        const wf_extent * extents,int n,uint64_t * fence,uint64_t * bytes,double * milliseconds){
    if(!fence)return false;
    return wf_fetch_extents_impl(wf,bank,extents,n,nullptr,bytes,milliseconds,stage,generation,fence);
}
bool wf_wait_ready_extents(wf_context * wf,int stage,int bank,uint64_t generation,uint64_t fence,void * stream){
    (void)wf;(void)stage;(void)bank;(void)generation;(void)fence;(void)stream;return false;
}
