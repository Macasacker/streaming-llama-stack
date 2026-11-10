// wf_read_bench.cpp - measure the fetcher's read shape without a GPU.
//
// The engine's strict relay reads a whole stage's spans into host memory through
// io_uring: io_threads rings, io_queue_depth reads in flight per ring, topped up
// after every completion. This program replays that loop against a real GGUF's
// span list, so the read path can be measured while the card serves something
// else.
//
// Build:
//   g++ -O2 -std=c++17 scripts/wf_read_bench.cpp -luring -o /tmp/wf_read_bench
// Run (usually through scripts/wf_read_matrix.py, which drops the page cache):
//   /tmp/wf_read_bench --spans /tmp/blk22.spans --file <gguf> --threads 8 --qd 4 --block 4194304
//
// --spans is one "<file_offset> <nbytes>" per line, the way a stage's spans are
// laid out; scripts/wf_read_spans.py generates it from the tensor directory.
// --direct mirrors params.io_direct: length rounded up to a sector, O_DIRECT fd.
//
// Two deliberate deviations from the provider, both to keep this runnable next
// to a serving engine: destinations come from a pool of threads x qd buffers
// rather than one 2.6 GiB staging slot per prefetch depth, and nothing uploads
// to the card. The upload, the expert-cache lookup and the staging slot gate are
// the pipeline half of fetch_ms; this measures only NVMe plus the reader pool.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <liburing.h>

namespace {

struct Span { uint64_t off; uint64_t len; };

struct Chunk {
    uint64_t off;
    uint32_t len;
    char *dest;
};

struct Opts {
    std::string spans_path;
    std::string file;
    int threads = 8;
    int qd = 4;
    size_t block = 4 * 1024 * 1024;   // auto_io_block() for a 3.7 GiB stage
    int passes = 2;
    bool direct = false;
};

std::deque<Chunk *> g_queue;          // pointers into the current pass's plan
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;

std::atomic<uint64_t> g_done_bytes{0};
std::atomic<uint64_t> g_short_reads{0};
std::atomic<int> g_done_chunks{0};
std::atomic<bool> g_failed{false};
std::atomic<bool> g_stop{false};      // shutdown signal, distinct from a transfer error

// Destination buffers, handed out at submission and returned at completion.
std::mutex g_pool_mutex;
std::vector<char *> g_free;

char * take_buffer() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_free.empty()) return nullptr;
    char * b = g_free.back();
    g_free.pop_back();
    return b;
}

void give_buffer(char * b) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_free.push_back(b);
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Same shape as wf_io_uring_worker(): top the ring up to qd - inflight, submit,
// submit_and_wait(1), then drain whatever else is already posted.
void reader(int fd, int qd) {
    struct io_uring ring;
    if (io_uring_queue_init((unsigned) std::max(64, qd * 4), &ring, 0) != 0) {
        std::fprintf(stderr, "io_uring_queue_init failed: %s\n", std::strerror(errno));
        g_failed.store(true);
        return;
    }

    int inflight = 0;
    std::vector<Chunk *> outstanding;

    auto forget = [&](Chunk * chunk) {
        for (size_t i = 0; i < outstanding.size(); i++)
            if (outstanding[i] == chunk) {
                outstanding[i] = outstanding.back();
                outstanding.pop_back();
                return;
            }
    };

    for (;;) {
        std::vector<Chunk *> batch;
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            while ((int) batch.size() < qd - inflight && !g_queue.empty()) {
                batch.push_back(g_queue.front());
                g_queue.pop_front();
            }
        }
        size_t taken = 0;
        for (auto * c : batch) {
            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
            char * dest = take_buffer();
            if (!sqe || !dest) break;
            c->dest = dest;
            io_uring_prep_read(sqe, fd, c->dest, c->len, c->off);
            io_uring_sqe_set_data(sqe, c);
            outstanding.push_back(c);
            inflight++;
            taken++;
        }
        if (taken < batch.size()) {
            // No SQE or no buffer for the rest. Put them back rather than
            // dropping them: an unpublished chunk never completes.
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            g_queue.insert(g_queue.begin(), batch.begin() + taken, batch.end());
        }
        if (inflight) io_uring_submit(&ring);

        if (inflight > 0) {
            struct io_uring_cqe * cqe = nullptr;
            if (io_uring_submit_and_wait(&ring, 1) < 0) { g_failed.store(true); break; }
            if (io_uring_wait_cqe(&ring, &cqe) != 0) { g_failed.store(true); break; }
            unsigned seen = 0;
            do {
                auto * c = static_cast<Chunk *>(io_uring_cqe_get_data(cqe));
                if (c) {
                    if (cqe->res < 0) g_failed.store(true);
                    else if ((uint32_t) cqe->res < c->len) g_short_reads.fetch_add(1);
                    g_done_bytes.fetch_add(cqe->res > 0 ? (uint64_t) cqe->res : 0,
                                           std::memory_order_relaxed);
                    g_done_chunks.fetch_add(1, std::memory_order_relaxed);
                    give_buffer(c->dest);
                    forget(c);
                }
                io_uring_cqe_seen(&ring, cqe);
                seen++;
            } while (seen < (unsigned) inflight && io_uring_peek_cqe(&ring, &cqe) == 0);
            inflight -= (int) seen;
            continue;
        }

        // Nothing queued and nothing in flight: leave on shutdown (the same
        // check wf_io_uring_worker() makes), otherwise wait for a pass.
        if (g_stop.load()) break;
        std::unique_lock<std::mutex> lock(g_queue_mutex);
        g_queue_cv.wait_for(lock, std::chrono::milliseconds(20),
                            [&] { return !g_queue.empty() || g_stop.load(); });
    }

    while (inflight > 0) {
        struct io_uring_cqe * cqe = nullptr;
        if (io_uring_wait_cqe(&ring, &cqe) != 0) break;
        auto * c = static_cast<Chunk *>(io_uring_cqe_get_data(cqe));
        if (c) {
            g_done_bytes.fetch_add(cqe->res > 0 ? (uint64_t) cqe->res : 0,
                                   std::memory_order_relaxed);
            g_done_chunks.fetch_add(1, std::memory_order_relaxed);
            give_buffer(c->dest);
            forget(c);
        }
        io_uring_cqe_seen(&ring, cqe);
        inflight--;
    }
    io_uring_queue_exit(&ring);
}

// Split a stage's spans into requests of at most `block` bytes, as
// wf_submit_layer_reads() does. io_direct rounds a chunk up to a sector.
std::vector<Chunk> make_plan(const std::vector<Span> & spans, const Opts & o, size_t * bytes) {
    std::vector<Chunk> plan;
    size_t total = 0;
    for (const auto & s : spans) {
        size_t within = 0;
        while (within < s.len) {
            size_t chunk = std::min(o.block, s.len - within);
            if (o.direct) chunk = (chunk + 4095) & ~(size_t) 4096;
            plan.push_back({s.off + within, (uint32_t) chunk, nullptr});
            within += chunk;
        }
        total += s.len;
    }
    *bytes = total;
    return plan;
}

}  // namespace

int main(int argc, char ** argv) {
    Opts o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto flag = [&](const char * name) { return a.rfind(name, 0) == 0; };
        auto value = [&]() -> std::string {
            size_t eq = a.find('=');
            if (eq != std::string::npos) return a.substr(eq + 1);
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (flag("--spans")) o.spans_path = value();
        else if (flag("--file")) o.file = value();
        else if (flag("--threads")) o.threads = std::atoi(value().c_str());
        else if (flag("--qd")) o.qd = std::atoi(value().c_str());
        else if (flag("--block")) o.block = (size_t) std::atoll(value().c_str());
        else if (flag("--passes")) o.passes = std::atoi(value().c_str());
        else if (flag("--direct")) o.direct = std::atoi(value().c_str()) != 0;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.spans_path.empty() || o.file.empty() || o.threads < 1 || o.qd < 1 || o.passes < 1) {
        std::fprintf(stderr,
                     "usage: --spans PATH --file PATH [--threads N] [--qd Q] [--block BYTES] "
                     "[--passes N] [--direct 0|1]\n");
        return 2;
    }

    std::vector<Span> spans;
    {
        std::ifstream in(o.spans_path);
        Span s{};
        while (in >> s.off >> s.len) spans.push_back(s);
    }
    if (spans.empty()) { std::fprintf(stderr, "no spans in %s\n", o.spans_path.c_str()); return 2; }

    // Pre-allocate every chunk slot so the pointers published to the queue stay
    // valid for the whole pass.
    size_t stage_bytes = 0;
    for (const auto & s : spans) stage_bytes += s.len;
    std::vector<Chunk> plan;
    plan.reserve(spans.size() * ((stage_bytes / o.block) + 2));
    size_t planned_bytes = 0;
    plan = make_plan(spans, o, &planned_bytes);

    const int slots = o.threads * o.qd;
    // A ceiling, not a convenience: this box is serving an engine while this
    // runs, and an unbounded threads x qd x block pool was OOM-killed at 63 GiB
    // during the QD sweep. 1 GiB covers every case in the matrix; fewer buffers
    // in flight than requested is a weaker measurement, being killed is no
    // measurement at all.
    const size_t pool_budget = 1024ull * 1024 * 1024;
    long avail_kib = 0;
    {
        std::ifstream in("/proc/meminfo");
        std::string line;
        while (std::getline(in, line))
            if (line.rfind("MemAvailable:", 0) == 0) avail_kib = std::atol(line.c_str() + 13);
    }
    if (avail_kib * 1024 < (long) pool_budget * 8) {
        std::fprintf(stderr, "only %ld MiB available; refusing to allocate a %zu MiB pool\n",
                     avail_kib / 1024, pool_budget / (1024 * 1024));
        return 3;
    }
    int usable_slots = slots;
    while (usable_slots > 1 && (size_t) usable_slots * o.block > pool_budget) usable_slots--;
    if (usable_slots < slots)
        std::fprintf(stderr, "pool capped at %d of %d slots (%zu MiB x %d)\n",
                     usable_slots, slots, o.block / (1024 * 1024), slots);
    for (int i = 0; i < usable_slots; i++) {
        char * b = (char *) aligned_alloc(4096, o.block);
        if (!b) { std::fprintf(stderr, "aligned_alloc(%zu) failed\n", o.block); return 1; }
        std::memset(b, 0, o.block);       // pre-touch: faults are not read time
        g_free.push_back(b);
    }

    const int fd = open(o.file.c_str(), O_RDONLY | (o.direct ? O_DIRECT : 0));
    if (fd < 0) { std::fprintf(stderr, "open %s: %s\n", o.file.c_str(), std::strerror(errno)); return 1; }

    std::vector<std::thread> pool;
    for (int i = 0; i < o.threads; i++) pool.emplace_back(reader, fd, o.qd);

    double best = 0.0, worst = 0.0;
    size_t chunk_count = 0;
    for (int pass = 0; pass < o.passes; pass++) {
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            for (auto & c : plan) g_queue.push_back(&c);
            g_queue_cv.notify_all();
        }
        chunk_count += plan.size();
        const int64_t start = now_ms();
        while (g_done_chunks.load() < (int) chunk_count && !g_failed.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const double ms = (double) (now_ms() - start);
        const double rate = (double) planned_bytes / (1u << 30) / (ms / 1000.0);
        if (pass == 0) best = worst = rate;
        else { if (rate < worst) worst = rate; if (rate > best) best = rate; }
        std::fprintf(stderr, "pass %d: %zu chunks, %.2f GiB in %.0f ms -> %.2f GiB/s%s\n",
                     pass, plan.size(), (double) planned_bytes / (1u << 30), ms, rate,
                     pass == 0 ? " (cold)" : "");
    }

    g_stop.store(true);
    { std::lock_guard<std::mutex> lock(g_queue_mutex); g_queue_cv.notify_all(); }
    for (auto & t : pool) t.join();

    std::printf("threads=%d qd=%d block=%.2fMiB direct=%d short_reads=%llu\n",
                o.threads, o.qd, o.block / (1024.0 * 1024.0), (int) o.direct,
                (unsigned long long) g_short_reads.load());
    std::printf("stage %.2f GiB, %zu spans: cold %.2f GiB/s, warm best %.2f GiB/s\n",
                (double) planned_bytes / (1u << 30), spans.size(), worst, best);
    close(fd);
    for (auto * b : g_free) free(b);
    return g_failed.load() ? 1 : 0;
}
