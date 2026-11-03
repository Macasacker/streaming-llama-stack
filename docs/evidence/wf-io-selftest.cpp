// wf-io-selftest.cpp - the weight fetcher's reader, proven against pread.
//
// Chunk 3.1e replaced the fetcher's pread worker pool with io_uring rings. This
// program drives the real `wf_*` pipeline over a throwaway file and byte-compares
// what each reader stages into the GPU buffers against the file itself, then
// reports both readers at two geometries, and checks that a read which does not
// deliver its bytes stops the fetcher instead of staging uninitialised memory.
// No model weights, and no VRAM beyond the fetcher's own staging buffers.
//
// Compile and run from the repository root against the objects the producer build
// already made; docs/linux-port-commands.md carries the exact line.
//
// Two timings are reported because they mean different things. `total_nvme_read_ms`
// sums each layer's own submit-to-last-completion window, and layers overlap, so
// that sum is larger than the pass and its ratio flatters whichever reader has the
// shorter windows. The wall clock of the whole pass is the number to quote.
//
// Exit code 0 only if every check passed.

#include "llama-weight-fetcher.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t MIB          = 1024ull * 1024ull;
constexpr size_t FILE_BYTES   = 64ull * MIB;
constexpr int    N_LAYERS     = 8;
constexpr size_t LAYER_BYTES  = FILE_BYTES / N_LAYERS;

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s   %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// Deterministic pseudo-random contents, so a re-run compares against the same bytes.
std::vector<uint8_t> make_payload() {
    std::vector<uint8_t> data(FILE_BYTES);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < data.size(); i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s ^ (s >> 17) + i;
        data[i] = (uint8_t) (s >> 32);
    }
    return data;
}

// One layer's spans: contiguous in the file and in the GPU buffer, with sizes that
// are multiples of 32 but deliberately not of 4096 - what GGUF tensor data looks
// like, and what makes the reader split its chunks.
std::vector<wf_weight_span> layer_spans(int layer, size_t & total_bytes, int file_idx = 0) {
    std::vector<wf_weight_span> spans;
    size_t offset = 0;
    int index = 0;
    while (offset < LAYER_BYTES) {
        const size_t room = LAYER_BYTES - offset;
        const size_t want = 512 * 1024 - 128 + (size_t) (layer * 8 + index * 32);
        const size_t nbytes = std::min(room, want);
        wf_weight_span s = {};
        s.tensor_name = "selftest.span";
        s.file_offset = (uint64_t) layer * LAYER_BYTES + offset;
        s.nbytes      = nbytes;
        s.file_idx    = file_idx;
        spans.push_back(s);
        offset += nbytes;
        index++;
    }
    total_bytes = offset;
    return spans;
}

struct result {
    const char * engine      = "none";
    double       window_ms   = 1e18;   // sum of the per-layer read windows
    double       wall_ms     = 1e18;   // whole pass, start to last release
    uint64_t     bytes       = 0;
    int          ops         = 0;
    bool         matched     = false;
    bool         failed      = false;  // wf_has_failed() after the pass
};

wf_context * make_fetcher(const char * path, int threads, int queue_depth) {
    wf_params p = wf_params_default();
    p.model_path      = path;
    p.n_layers        = N_LAYERS;
    p.max_layer_bytes = LAYER_BYTES;
    p.n_gpu_buffers   = 2;
    p.n_ram_buffers   = 2;
    p.io_block_size   = 512 * 1024;
    p.io_queue_depth  = queue_depth;
    p.io_threads      = threads;
    p.prefetch_depth  = 2;
    p.mode            = WF_MODE_DIRECT_IO;
    p.moe_mode        = WF_MOE_FETCH_ALL;
    p.post_mode       = WF_POST_KEEP_FETCHER;
    p.gpu_backend     = nullptr;   // the fetcher uses the current CUDA device
    p.trace           = false;
    return wf_create(&p);
}

// Drop the file from the page cache so each round starts from cold storage.
void drop_cache(const char * path) {
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

result run(const char * path, const std::vector<uint8_t> & payload, int threads, int queue_depth) {
    result out;
    wf_context * wf = make_fetcher(path, threads, queue_depth);
    if (!wf) {
        check(false, "wf_create");
        return out;
    }
    out.engine = wf_get_io_engine(wf);

    std::vector<size_t> totals(N_LAYERS, 0);
    for (int layer = 0; layer < N_LAYERS; layer++) {
        std::vector<wf_weight_span> spans = layer_spans(layer, totals[layer]);
        check(totals[layer] == LAYER_BYTES && !wf_has_failed(wf),
              "layer spans cover the layer and stay inside the file");
        wf_register_layer(wf, layer, spans.data(), (int) spans.size(), totals[layer]);
    }

    const auto wall0 = std::chrono::steady_clock::now();
    wf_start(wf);

    std::vector<uint8_t> readback(LAYER_BYTES);
    out.matched = true;
    for (int layer = 0; layer < N_LAYERS; layer++) {
        void * gpu = wf_wait_layer_gpu(wf, layer);
        if (!gpu) {
            std::printf("FAIL   layer %d never reached the GPU\n", layer);
            out.matched = false;
            break;
        }
        const cudaError_t err = cudaMemcpy(readback.data(), gpu, LAYER_BYTES, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::printf("FAIL   layer %d readback: %s\n", layer, cudaGetErrorString(err));
            out.matched = false;
            break;
        }
        const size_t base = (size_t) layer * LAYER_BYTES;
        if (memcmp(readback.data(), payload.data() + base, LAYER_BYTES) != 0) {
            size_t first = 0;
            while (first < LAYER_BYTES && readback[first] == payload[base + first]) first++;
            std::printf("FAIL   layer %d differs from the file at +0x%zx\n", layer, first);
            out.matched = false;
        }
        wf_release_layer(wf, layer);
    }
    out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();

    wf_stop(wf);
    wf_stats stats = {};
    wf_get_stats(wf, &stats);
    out.window_ms = stats.total_nvme_read_ms;
    out.bytes     = stats.bytes_read_nvme;
    out.ops       = stats.io_ops;
    out.failed    = wf_has_failed(wf);
    wf_destroy(wf);
    return out;
}

// A span that runs past the end of the file must stop the fetcher, not read zero
// bytes and call that a completed transfer.
void check_short_read_refused(const char * path) {
    wf_context * wf = make_fetcher(path, 2, 4);
    if (!wf) {
        check(false, "truncation probe: wf_create");
        return;
    }
    std::vector<wf_weight_span> spans;
    wf_weight_span s = {};
    s.tensor_name = "selftest.overrun";
    s.file_offset = FILE_BYTES - 1024;
    s.nbytes      = 8 * 1024;          // 7 KiB past the end
    s.file_idx    = 0;
    spans.push_back(s);
    wf_register_layer(wf, 0, spans.data(), 1, s.nbytes);
    check(wf_has_failed(wf), "a span running past EOF fails the fetcher at registration");
    check(wf_wait_layer_gpu(wf, 0) == nullptr, "the overrunning layer is never handed to compute");
    wf_stop(wf);
    wf_destroy(wf);
}

void report(const char * label, const result & uring, const result & baseline) {
    std::printf("info %-9s io_uring wall %.1f ms / window sum %.1f ms | pread wall %.1f ms / window sum %.1f ms | wall x%.2f, window x%.2f\n",
                label, uring.wall_ms, uring.window_ms, baseline.wall_ms, baseline.window_ms,
                baseline.wall_ms / uring.wall_ms, baseline.window_ms / uring.window_ms);
}

}  // namespace

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "build/wf-io.bin";

    const std::vector<uint8_t> payload = make_payload();
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        std::printf("FAIL  cannot write %s\n", path);
        return 1;
    }
    std::fwrite(payload.data(), 1, payload.size(), f);
    std::fclose(f);
    check(payload.size() == FILE_BYTES, "throwaway file written");

    check_short_read_refused(path);

    struct geometry { const char * label; int threads; int qd; };
    const geometry geometries[] = {
        {"4 threads", 4, 8},   // overlapping layers: the pipeline's shape
        {"1 thread ", 1, 8},   // one ring, queue depth 8 vs pread's queue depth 1
    };

    for (const auto & g : geometries) {
        setenv("LLAMA_WF_IO_URING", "1", 1);
        result uring = { "io_uring", 1e18, 1e18, 0, 0, true, false };
        for (int round = 0; round < 3; round++) {
            drop_cache(path);
            result r = run(path, payload, g.threads, g.qd);
            check(r.matched, "io_uring staged every byte the file holds");
            check(!r.failed, "io_uring reported no failed read");
            check(std::strcmp(r.engine, "io_uring") == 0, "the default reader is io_uring");
            uring.ops = r.ops;
            uring.bytes = r.bytes;
            uring.matched = uring.matched && r.matched;
            uring.window_ms = std::min(uring.window_ms, r.window_ms);
            uring.wall_ms = std::min(uring.wall_ms, r.wall_ms);
        }

        setenv("LLAMA_WF_IO_URING", "0", 1);
        result baseline = { "pread", 1e18, 1e18, 0, 0, true, false };
        for (int round = 0; round < 3; round++) {
            drop_cache(path);
            result r = run(path, payload, g.threads, g.qd);
            check(r.matched, "pread baseline staged every byte the file holds");
            check(!r.failed, "pread reported no failed read");
            check(std::strcmp(r.engine, "pread") == 0, "LLAMA_WF_IO_URING=0 selects the pread baseline");
            baseline.ops = r.ops;
            baseline.bytes = r.bytes;
            baseline.matched = baseline.matched && r.matched;
            baseline.window_ms = std::min(baseline.window_ms, r.window_ms);
            baseline.wall_ms = std::min(baseline.wall_ms, r.wall_ms);
        }

        check(uring.ops == baseline.ops && uring.ops > 0, "both readers issued the same number of reads");
        // io_direct rounds a chunk up to 4 KiB, so a read can transfer more than the
        // span asked for; what must match is that both readers did the same thing.
        check(uring.bytes == baseline.bytes && uring.bytes >= FILE_BYTES,
              "both readers transferred the same bytes, covering the whole file");
        report(g.label, uring, baseline);
    }

    std::printf("SELFTEST %s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
