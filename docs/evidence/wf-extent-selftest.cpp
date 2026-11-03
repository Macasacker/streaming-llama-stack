// Chunk 3.1f: the strict extent reader, driven through the real wf_* API with no
// model file. It creates a required-streaming fetcher over a throwaway file,
// transfers named extents into a compute bank, and byte-compares what landed
// there against the file. It also checks the refusals: an untyped profile, a
// strict profile missing its staging tier or its async ordering, a non-strict
// caller, an extent past the file's end, an extent past the bank, a generation
// that no longer names a resident stage, and the ready-batch fence from a worker
// thread.
//
// It is deliberately compiled without CUDA, so the fetcher's host banks and
// memcpy upload stand in for VRAM and the comparison reads the bank directly.
// The same code path runs with CUDA, with pinned slots and a stream-ordered
// upload; docs/evidence/wf-io-selftest.cpp covers the relay that feeds it. No
// GPU, no model, no weights.
//
// Compile and run: see docs/linux-port-commands.md ("The strict extent reader").
#include "llama-weight-fetcher.h"
#include "llama-wf-bounded-policy.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
static void check(const char * what, bool ok) {
    std::printf("%-6s %s\n", ok ? "ok" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}

// The pipeline publishes a prefetched stage asynchronously; wait for it instead
// of assuming the consumer's first wait already advanced it.
static bool wait_resident(wf_context * wf, int stage, int bank, uint64_t * generation) {
    for (int round = 0; round < 400; ++round) {
        if (wf_ready_direct_bank(wf, stage, bank, generation)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

static uint64_t next_seed = 0x5857ull;
static uint8_t lcg(void) {
    next_seed = next_seed * 6364136223846793005ull + 1442695040888963407ull;
    return uint8_t(next_seed >> 33);
}

static bool bank_matches(wf_context * wf, int bank, uint64_t bank_offset, uint64_t file_offset,
                         size_t nbytes, const std::vector<uint8_t> & file) {
    const char * base = static_cast<const char *>(wf_get_gpu_buf_ptr(wf, bank));
    return base && std::memcmp(base + bank_offset, file.data() + file_offset, nbytes) == 0;
}

int main(void) {
    const std::string path = "build/wf-extent-selftest.bin";
    std::vector<uint8_t> file(size_t(32u << 20));
    for (auto & b : file) b = lcg();
    {
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) { std::printf("FAIL cannot write the fixture\n"); return 1; }
        const size_t wrote = std::fwrite(file.data(), 1, file.size(), f);
        std::fclose(f);
        if (wrote != file.size()) { std::printf("FAIL the fixture is short\n"); return 1; }
    }

    auto plan = [&]() {
        wf_params p = wf_params_default();
        p.model_path = path.c_str();
        p.n_layers = 2;
        p.n_gpu_buffers = 2;
        p.n_ram_buffers = 2;
        p.max_layer_bytes = size_t(24u << 20);
        p.io_block_size = llama_wf_bounded::auto_io_block(p.max_layer_bytes);
        p.ds_staging_limit_bytes = llama_wf_bounded::small_staging_bytes;
        p.ds_async_drain = true;
        p.ds_compute_direct = true;
        p.mode = WF_MODE_DIRECT_IO;
        p.io_threads = 2;
        p.io_queue_depth = 8;
        p.io_direct = false;
        return p;
    };
    auto span_at = [](const char * name, uint64_t file_offset, size_t nbytes) {
        wf_weight_span s{};
        s.tensor_name = name;
        s.file_offset = file_offset;
        s.nbytes = nbytes;
        s.alloc_size = nbytes;
        s.alloc_offset = 0;
        s.file_idx = 0;
        return s;
    };

    // The strict profile admits one exact typed geometry, and says so before it
    // allocates anything.
    {
        wf_params p = plan();
        p.ds_staging_limit_bytes = 0;
        p.io_block_size = 512 * 1024;
        check("an untyped strict profile is refused", wf_create_required_streaming(&p) == nullptr);
        p = plan();
        p.n_ram_buffers = 0;
        check("a strict profile without a pinned staging tier is refused", wf_create_required_streaming(&p) == nullptr);
        p = plan();
        p.ds_async_drain = false;
        check("a strict profile without async copy ordering is refused", wf_create_required_streaming(&p) == nullptr);
        p = plan();
        p.ds_compute_direct = true;
        p.n_gpu_buffers = 1;
        check("direct compute needs exactly two banks", wf_create_required_streaming(&p) == nullptr);
        wf_params relay = wf_params_default();
        relay.model_path = path.c_str();
        relay.n_layers = 1;
        relay.ds_staging_limit_bytes = llama_wf_bounded::small_staging_bytes;
        check("a non-strict caller cannot ask for a typed profile", wf_create(&relay) == nullptr);
    }

    wf_params strict = plan();
    wf_context * wf = wf_create_required_streaming(&strict);
    if (!wf) { std::printf("FAIL the strict relay profile was refused\n"); return 1; }
    check("the strict relay profile is accepted", !wf_has_failed(wf));
    check("direct compute banks are reported", wf_is_ds_direct(wf) && wf_get_n_gpu_bufs(wf) == 2);
    std::printf("info reader %s, bank capacity %zu bytes\n", wf_get_io_engine(wf),
                wf_get_gpu_buf_capacity(wf));

    const std::vector<wf_weight_span> layer0 {span_at("extent_selftest.a", 0, 1u << 20)};
    const std::vector<wf_weight_span> layer1 {span_at("extent_selftest.b", 1u << 20, 1u << 20)};
    wf_register_layer(wf, 0, layer0.data(), int(layer0.size()), 1u << 20);
    wf_register_layer(wf, 1, layer1.data(), int(layer1.size()), 1u << 20);
    check("strict registration reported no failure", !wf_has_failed(wf));

    wf_start(wf);
    check("layer 0 reached the first bank", wf_wait_layer_gpu(wf, 0) == wf_get_gpu_buf_ptr(wf, 0));
    uint64_t fetched = 0; double fetch_ms = 0;
    check("the completed layer reports its bytes", wf_get_required_layer_fetch(wf, 0, &fetched, &fetch_ms) &&
          fetched == (1u << 20));
    check("layer 0's bytes are the file's", bank_matches(wf, 0, 0, 0, 1u << 20, file));

    uint64_t generation = 0;
    check("a stage the consumer already took is not a ready target", !wf_ready_direct_bank(wf, 0, 0, &generation));
    check("the prefetched stage names its own bank", wait_resident(wf, 1, 1, &generation) && generation != 0);
    uint64_t same = 0;
    check("the generation is stable while the stage is resident",
          wf_ready_direct_bank(wf, 1, 1, &same) && same == generation);
    check("the stage does not name the other bank", !wf_ready_direct_bank(wf, 1, 0, &same));

    // Extents into bank 0, above the region layer 0 occupies. The first is larger
    // than one 8 MiB staging slot, so it is split across two parts.
    const std::vector<wf_extent> extents {
        wf_extent {uint64_t(2u << 20), uint64_t(4u << 20), uint64_t(10u << 20), 0},
        wf_extent {uint64_t(14u << 20), uint64_t(15u << 20), uint64_t(1u << 20), 0},
        wf_extent {uint64_t(20u << 20), uint64_t(16u << 20), uint64_t(8u << 20), 0},
    };
    uint64_t bytes = 0; double ms = 0;
    const bool transferred = wf_fetch_extents_direct(wf, 0, extents.data(), int(extents.size()), nullptr, &bytes, &ms);
    check("the extent batch transferred", transferred && !wf_has_failed(wf));
    check("the extent batch counted every byte",
          transferred && bytes == extents[0].nbytes + extents[1].nbytes + extents[2].nbytes);
    check("extent bytes landed where they were named",
          bank_matches(wf, 0, extents[0].bank_offset, extents[0].file_offset, extents[0].nbytes, file) &&
          bank_matches(wf, 0, extents[1].bank_offset, extents[1].file_offset, extents[1].nbytes, file) &&
          bank_matches(wf, 0, extents[2].bank_offset, extents[2].file_offset, extents[2].nbytes, file));
    check("the relay's own stage bytes survived the extent write", bank_matches(wf, 0, 0, 0, 1u << 20, file));

    // The overlap hook runs while the first reads are in flight.
    {
        // Reuse the region the second extent wrote: the hook is the point of the
        // call, and the bytes are re-verified afterwards.
        std::vector<wf_extent> one {wf_extent {uint64_t(24u << 20), uint64_t(15u << 20), uint64_t(1u << 20), 0}};
        bool hooked = false;
        uint64_t n = 0; double m = 0;
        const bool ok = wf_fetch_extents_direct_overlap(wf, 0, one.data(), int(one.size()), nullptr, &n, &m,
            [](void * self) { *static_cast<bool *>(self) = true; }, &hooked);
        check("the overlap hook ran and the extent landed", ok && hooked && n == one[0].nbytes &&
              bank_matches(wf, 0, one[0].bank_offset, one[0].file_offset, one[0].nbytes, file));
    }

    // A ready batch, submitted from its own worker thread and waited on by fence.
    if (generation != 0) {
        std::vector<wf_extent> ready {wf_extent {uint64_t(12u << 20), uint64_t(16u << 20), uint64_t(3u << 20), 0}};
        uint64_t fence = 0; uint64_t ready_bytes = 0; double ready_ms = 0;
        std::atomic<bool> started {false};
        std::thread worker([&] {
            started = wf_fetch_ready_extents(wf, 1, 1, generation, ready.data(), int(ready.size()),
                                             &fence, &ready_bytes, &ready_ms);
        });
        worker.join();
        check("the ready batch transferred on the worker",
              started.load() && fence != 0 && ready_bytes == ready[0].nbytes && !wf_has_failed(wf));
        check("the consumer orders itself after the ready batch", wf_wait_ready_extents(wf, 1, 1, generation, fence, nullptr));
        check("the ready batch landed at the named bank region",
              bank_matches(wf, 1, ready[0].bank_offset, ready[0].file_offset, ready[0].nbytes, file));
        uint64_t probe = 7;
        check("a batch naming a stale generation is refused without poisoning the fetcher",
              !wf_fetch_ready_extents(wf, 1, 1, generation + 1, ready.data(), int(ready.size()),
                                      &probe, &ready_bytes, &ready_ms) && probe == 0 && !wf_has_failed(wf));
        check("an unknown fence cannot be waited on", !wf_wait_ready_extents(wf, 1, 1, generation, fence + 1000, nullptr));
        wf_release_layer(wf, 1);
        check("a fence cannot be waited on after its stage left its bank",
              !wf_wait_ready_extents(wf, 1, 1, generation, fence, nullptr));
    } else check("the second stage was resident for the ready batch", false);

    wf_stop(wf);
    wf_quiesce(wf);
    wf_destroy(wf);

    // Failures are sticky, so the refusals run against a fresh fetcher.
    wf_params second_plan = plan();
    wf_context * second = wf_create_required_streaming(&second_plan);
    if (!second) { std::printf("FAIL the second strict fetcher was refused\n"); return 1; }
    const std::vector<wf_weight_span> only {span_at("extent_selftest.d", 0, 1u << 20)};
    wf_register_layer(second, 0, only.data(), int(only.size()), 1u << 20);
    wf_start(second);
    check("a layer still transfers before a refusal", wf_wait_layer_gpu(second, 0) != nullptr);
    const size_t capacity = wf_get_gpu_buf_capacity(second);
    uint64_t n = 0; double m = 0;
    const std::vector<wf_extent> past_end {wf_extent {uint64_t(32u << 20) - 1024, 0, 4096, 0}};
    check("an extent running past its file's end is refused",
          !wf_fetch_extents_direct(second, 0, past_end.data(), 1, nullptr, &n, &m));
    check("a refused extent fails the fetcher for good", wf_has_failed(second));
    check("nothing transfers after the fetcher failed",
          !wf_fetch_extents_direct(second, 0, past_end.data(), 1, nullptr, &n, &m) && n == 0);
    wf_stop(second);
    wf_quiesce(second);
    wf_destroy(second);

    wf_params third_plan = plan();
    wf_context * third = wf_create_required_streaming(&third_plan);
    if (!third) { std::printf("FAIL the third strict fetcher was refused\n"); return 1; }
    wf_register_layer(third, 0, only.data(), int(only.size()), 1u << 20);
    wf_start(third);
    check("a layer transfers before the bank refusal", wf_wait_layer_gpu(third, 0) != nullptr);
    const std::vector<wf_extent> past_bank {wf_extent {0, uint64_t(capacity) - 1024, 4096, 0}};
    check("an extent running past its compute bank is refused",
          !wf_fetch_extents_direct(third, 0, past_bank.data(), 1, nullptr, &n, &m) && wf_has_failed(third));
    wf_stop(third);
    wf_quiesce(third);
    wf_destroy(third);

    std::remove(path.c_str());
    std::printf("SELFTEST %s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
