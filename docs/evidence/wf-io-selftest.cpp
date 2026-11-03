// wf-io-selftest.cpp - the weight fetcher's reader, proven against pread.
//
// Chunk 3.1e replaced the fetcher's pread worker pool with io_uring rings. This
// program drives the real `wf_*` pipeline over a throwaway file and byte-compares
// what each reader stages into the GPU buffers against the file itself, then
// reports the two readers' read times side by side. No model weights, and no
// VRAM beyond the fetcher's own staging buffers (a few MB).
//
// Compile and run it from the repository root, against the objects the producer
// build already produced; docs/linux-port-commands.md carries the exact line.
//
// Exit code 0 only if every layer matched for both readers.

#include "llama-weight-fetcher.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t      MIB        = 1024ull * 1024ull;
constexpr size_t      FILE_BYTES = 64ull * MIB;
constexpr int         N_LAYERS   = 8;
constexpr size_t      LAYER_BYTES = FILE_BYTES / N_LAYERS;
constexpr size_t      ALIGN      = 32;      // GGUF general.alignment default

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
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        data[i] = (uint8_t) (s >> 32);
    }
    return data;
}

// One layer's spans: contiguous in the file and in the GPU buffer, with sizes
// that are multiples of 32 but deliberately not of 4096, which is what GGUF
// tensor data looks like and what makes the reader split and round its chunks.
std::vector<wf_weight_span> layer_spans(int layer, size_t & total_bytes) {
    std::vector<wf_weight_span> spans;
    size_t offset = 0;
    int index = 0;
    while (offset < LAYER_BYTES) {
        size_t room = LAYER_BYTES - offset;
        size_t want = 512 * 1024 - 128 + (size_t) (layer * 8 + index * 32);  // multiple of 32
        size_t nbytes = std::min(room, std::min(want, room));
        wf_weight_span s = {};
        s.tensor_name = "selftest.span";
        s.file_offset = (uint64_t) layer * LAYER_BYTES + offset;
        s.nbytes      = nbytes;
        s.alloc_size  = 0;      // pack sequentially
        s.alloc_offset = 0;
        s.mmap_addr   = nullptr;
        s.file_idx    = 0;
        s.is_expert   = false;
        s.expert_idx  = -1;
        spans.push_back(s);
        offset += nbytes;
        index++;
    }
    total_bytes = offset;
    return spans;
}

struct result {
    const char * engine  = "none";
    double       read_ms = 0.0;
    uint64_t     bytes   = 0;
    int          ops     = 0;
    bool         matched = false;
};

// Drop the file from the page cache so each reader starts from cold storage.
void drop_cache(const char * path) {
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

result run(const char * path, const std::vector<uint8_t> & payload) {
    wf_params p = wf_params_default();
    p.model_path       = path;
    p.n_layers         = N_LAYERS;
    p.max_layer_bytes  = LAYER_BYTES;
    p.n_gpu_buffers    = 2;
    p.n_ram_buffers    = 2;
    p.io_block_size    = 512 * 1024;
    p.io_queue_depth   = 8;
    p.io_threads       = 4;
    p.prefetch_depth   = 2;
    p.mode             = WF_MODE_DIRECT_IO;
    p.moe_mode         = WF_MOE_FETCH_ALL;
    p.post_mode        = WF_POST_KEEP_FETCHER;
    p.gpu_backend      = nullptr;   // the fetcher uses the current CUDA device
    p.trace            = false;

    result out;
    wf_context * wf = wf_create(&p);
    if (!wf) {
        check(false, "wf_create");
        return out;
    }
    out.engine = wf_get_io_engine(wf);

    std::vector<size_t> totals(N_LAYERS, 0);
    for (int layer = 0; layer < N_LAYERS; layer++) {
        std::vector<wf_weight_span> spans = layer_spans(layer, totals[layer]);
        check(totals[layer] == LAYER_BYTES, "layer spans cover the whole layer");
        wf_register_layer(wf, layer, spans.data(), (int) spans.size(), totals[layer]);
    }

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

    wf_stop(wf);

    wf_stats stats = {};
    wf_get_stats(wf, &stats);
    out.read_ms = stats.total_nvme_read_ms;
    out.bytes   = stats.bytes_read_nvme;
    out.ops     = stats.io_ops;

    wf_destroy(wf);
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "build/wf-io.bin";

    std::vector<uint8_t> payload = make_payload();
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        std::printf("FAIL  cannot write %s\n", path);
        return 1;
    }
    std::fwrite(payload.data(), 1, payload.size(), f);
    std::fclose(f);
    check(payload.size() == FILE_BYTES, "throwaway file written");

    // Three cold rounds each, page cache dropped before every one, so the
    // comparison is the best of three rather than one noisy sample.
    setenv("LLAMA_WF_IO_URING", "1", 1);
    result uring = { "io_uring", 1e18, 0, 0, true };
    for (int round = 0; round < 3; round++) {
        drop_cache(path);
        result r = run(path, payload);
        check(r.matched, "io_uring reader staged every byte the file holds");
        check(std::strcmp(r.engine, "io_uring") == 0, "the default reader is io_uring");
        uring.ops    = r.ops;
        uring.bytes  = r.bytes;
        uring.matched = uring.matched && r.matched;
        uring.read_ms = std::min(uring.read_ms, r.read_ms);
    }

    setenv("LLAMA_WF_IO_URING", "0", 1);
    result pread_baseline = { "pread", 1e18, 0, 0, true };
    for (int round = 0; round < 3; round++) {
        drop_cache(path);
        result r = run(path, payload);
        check(r.matched, "pread baseline staged every byte the file holds");
        check(std::strcmp(r.engine, "pread") == 0, "LLAMA_WF_IO_URING=0 selects the pread baseline");
        pread_baseline.ops    = r.ops;
        pread_baseline.bytes  = r.bytes;
        pread_baseline.matched = pread_baseline.matched && r.matched;
        pread_baseline.read_ms = std::min(pread_baseline.read_ms, r.read_ms);
    }

    check(uring.ops == pread_baseline.ops && uring.ops > 0, "both readers issued the same number of reads");
    // io_direct rounds a chunk up to 4 KiB, so a read can transfer more than the
    // span asks for; what must match is that both readers did the same thing.
    check(uring.bytes == pread_baseline.bytes && uring.bytes >= FILE_BYTES,
          "both readers transferred the same bytes, covering the whole file");

    std::printf("info io_uring : best of 3 cold rounds, %.1f ms for %llu MiB in %d ops (%llu MiB transferred)\n",
                uring.read_ms, (unsigned long long) (FILE_BYTES / MIB), uring.ops,
                (unsigned long long) (uring.bytes / MIB));
    std::printf("info pread    : best of 3 cold rounds, %.1f ms for %llu MiB in %d ops (%llu MiB transferred)\n",
                pread_baseline.read_ms, (unsigned long long) (FILE_BYTES / MIB),
                pread_baseline.ops, (unsigned long long) (pread_baseline.bytes / MIB));
    if (pread_baseline.read_ms > 0) {
        std::printf("info io_uring vs pread: x%.2f\n", pread_baseline.read_ms / uring.read_ms);
    }

    std::printf("SELFTEST %s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
