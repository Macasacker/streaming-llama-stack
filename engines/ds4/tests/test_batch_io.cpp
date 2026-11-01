/* Model-free gate for the io_uring transport in ds4_batch_io.cpp.
 *
 * The transport is what the SSD-streaming expert path reads through. Its call sites were guarded
 * as platform-specific in ds4_cuda.cu, so nothing on Linux ever exercised it. This harness drives it against a throwaway file: no GPU, no GGUF, no VRAM. It is the only
 * correctness gate for the read path that can run while another process holds the GPU.
 *
 * The core check is the engine's own ds4_ds_selftest(): read a span twice, once at queue depth 1
 * and once in batches, and byte-compare both against pread. This runs that matrix over several
 * chunk and batch sizes, for read_parts 1 and 8, so both the plain requests and the 4KiB span
 * splitting are covered.
 *
 * Everything else pins the refusals. The transport has no silent fallback, so each rejection is
 * asserted rather than tolerated: an invalid add, a read past EOF, a full batch, a GPU add with no
 * arena, a parts change on a live transport, and the split submit/wait pipeline.
 *
 * Usage: test_batch_io <file> [chunk_MiB,...] [batch,...]
 * The file must be at least 32 MiB. Exit 0 only if every check passed.
 */
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4_batch_io.h"

namespace {

int g_failures = 0;

void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    g_failures++;
}

void check(bool condition, const char *fmt, ...) {
    if (condition) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    g_failures++;
}

char *alloc_aligned(uint64_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 4096, bytes) != 0 || !p) return nullptr;
    return (char *)p;
}

/* One logical span read through the batch API, then compared against pread. */
void check_batch_read(int fd, uint64_t offset, uint32_t bytes, char *dst, const char *what) {
    char *reference = alloc_aligned(bytes);
    if (!reference) { fail("%s: reference alloc failed", what); return; }
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, reference + done, bytes - done, offset + done);
        if (got <= 0) { fail("%s: pread at %llu: %s", what, (unsigned long long)(offset + done), strerror(errno)); free(reference); return; }
        done += (uint64_t)got;
    }
    ds4_ds_batch_reset();
    check(ds4_ds_batch_add(offset, bytes, dst) == 1, "%s: batch_add refused", what);
    check(ds4_ds_batch_bytes() == bytes, "%s: batch_bytes %llu != %u", what,
          (unsigned long long)ds4_ds_batch_bytes(), bytes);
    /* One logical span is one pending entry; the 4KiB split happens at submit. */
    check(ds4_ds_batch_count() == 1, "%s: %u pending entries, expected 1", what, ds4_ds_batch_count());
    memset(dst, 0xA5, bytes);
    uint64_t got_bytes = 0;
    check(ds4_ds_batch_submit_wait(&got_bytes, nullptr, nullptr) == 1, "%s: submit_wait failed", what);
    check(got_bytes == bytes, "%s: submit_wait reported %llu bytes, expected %u", what,
          (unsigned long long)got_bytes, bytes);
    check(memcmp(reference, dst, bytes) == 0, "%s: bytes differ from pread", what);
    free(reference);
}

/* One run of the whole suite at a given read_parts. The transport is re-initialised per config,
 * because read_parts is refused once a transport is live. */
void run_suite(int fd, uint64_t file_size, uint32_t parts, uint64_t span,
               const uint32_t *chunks, size_t chunk_count,
               const uint32_t *batches, size_t batch_count) {
    check(ds4_ds_configure_read_parts(parts) == 1, "parts=%u: configure refused", parts);

    if (!ds4_ds_init_from_fd(fd)) { fail("parts=%u: transport init failed", parts); return; }
    check(ds4_ds_available() == 1, "parts=%u: available() says no", parts);
    check(ds4_ds_configure_read_parts(2) == 0, "parts=%u: a live transport accepted a parts change", parts);

    const uint32_t capacity = ds4_ds_batch_capacity();
    check(capacity > 0, "parts=%u: capacity 0", parts);

    char *buffer = alloc_aligned(span);
    if (!buffer) { fail("parts=%u: buffer alloc", parts); ds4_ds_shutdown(); return; }

    /* Refusals. */
    ds4_ds_batch_reset();
    check(ds4_ds_batch_add(0, 0, buffer) == 0, "parts=%u: zero-byte add accepted", parts);
    check(ds4_ds_batch_add(0, 4096, nullptr) == 0, "parts=%u: null destination accepted", parts);
    check(ds4_ds_batch_add(file_size - 1, 2, buffer) == 0, "parts=%u: read past EOF accepted", parts);
    check(ds4_ds_batch_count() == 0, "parts=%u: refused adds still queued", parts);
    /* A GPU destination without an arena must refuse and say why (reason 2), without touching CUDA. */
    check(ds4_ds_batch_add_gpu(0, 4096, 0) == 0, "parts=%u: GPU add accepted with no arena", parts);
    check(ds4_ds_gpu_add_fail_reason() == 2, "parts=%u: refusal reason %d, expected 2 (no arena)",
          parts, ds4_ds_gpu_add_fail_reason());

    /* A full batch refuses the next add instead of dropping it. */
    ds4_ds_batch_reset();
    uint32_t added = 0;
    while (added < capacity + 1 && ds4_ds_batch_add(0, 4096, buffer + (size_t)added * 4096)) added++;
    check(added == capacity, "parts=%u: %u adds accepted, capacity said %u", parts, added, capacity);
    ds4_ds_batch_reset();

    /* Plain reads, both aligned and unaligned, plus a small tail read that must not be padded. */
    check_batch_read(fd, 0, 4u << 20, buffer, "aligned 4MiB");
    check_batch_read(fd, 4096 + 17, 1000, buffer, "unaligned 1000B");
    check_batch_read(fd, span - 3000, 3000, buffer, "tail 3000B");

    /* Many spans in one batch. */
    {
        char *reference = alloc_aligned(span);
        if (!reference) { fail("parts=%u: reference alloc", parts); free(buffer); ds4_ds_shutdown(); return; }
        uint64_t done = 0;
        while (done < span) {
            const ssize_t got = pread(fd, reference + done, span - done, done);
            if (got <= 0) { fail("parts=%u: reference pread: %s", parts, strerror(errno)); break; }
            done += (uint64_t)got;
        }
        ds4_ds_batch_reset();
        uint32_t issued = 0;
        while (issued < 32) {
            const uint32_t bytes = (uint32_t)(span / 32);
            if (!ds4_ds_batch_add((uint64_t)issued * bytes, bytes, buffer + (size_t)issued * bytes)) break;
            issued++;
        }
        check(issued == 32, "parts=%u: only %u/32 spans accepted", parts, issued);
        memset(buffer, 0x5A, span);
        uint64_t got_bytes = 0;
        check(ds4_ds_batch_submit_wait(&got_bytes, nullptr, nullptr) == 1, "parts=%u: 32-span batch failed", parts);
        check(got_bytes == span, "parts=%u: batch reported %llu bytes, expected %llu", parts,
              (unsigned long long)got_bytes, (unsigned long long)span);
        check(memcmp(reference, buffer, span) == 0, "parts=%u: 32-span batch differs from pread", parts);
        free(reference);
    }

    /* Split submit/wait: the submit returns before completion, and wait drains it. */
    {
        ds4_ds_batch_reset();
        for (uint32_t i = 0; i < 4; i++) {
            if (!ds4_ds_batch_add((uint64_t)i * (span / 4), span / 4, buffer + (size_t)i * (span / 4))) {
                fail("parts=%u: pipeline add refused", parts);
                break;
            }
        }
        check(ds4_ds_batch_submit() == 1, "parts=%u: submit failed", parts);
        check(ds4_ds_batch_inflight() >= 1, "parts=%u: nothing inflight after submit", parts);
        check(ds4_ds_batch_oldest_ready() >= 0, "parts=%u: oldest_ready returned -1 unexpectedly", parts);
        uint64_t got_bytes = 0;
        check(ds4_ds_batch_wait(&got_bytes, nullptr, nullptr) == 1, "parts=%u: batch_wait failed", parts);
        check(got_bytes == span, "parts=%u: pipeline reported %llu bytes, expected %llu", parts,
              (unsigned long long)got_bytes, (unsigned long long)span);
        char *reference = alloc_aligned(span);
        if (reference) {
            uint64_t done = 0;
            while (done < span) {
                const ssize_t got = pread(fd, reference + done, span - done, done);
                if (got <= 0) { fail("parts=%u: pipeline reference pread: %s", parts, strerror(errno)); break; }
                done += (uint64_t)got;
            }
            check(memcmp(reference, buffer, span) == 0, "parts=%u: pipeline bytes differ from pread", parts);
            free(reference);
        }
    }

    /* The low-priority background batch is a separate ring; it must also match pread. */
    {
        ds4_ds_background_span spans[3];
        char *reference = alloc_aligned(span / 2);
        if (!reference) { fail("parts=%u: background reference alloc", parts); free(buffer); ds4_ds_shutdown(); return; }
        const uint32_t part = span / 6;
        for (uint32_t i = 0; i < 3; i++) {
            spans[i].offset = span + (uint64_t)i * part;
            spans[i].bytes = part;
            spans[i].destination = buffer + (size_t)i * part;
        }
        uint64_t done = 0;
        while (done < part * 3) {
            const ssize_t got = pread(fd, reference + done, part * 3 - done, span);
            if (got <= 0) { fail("parts=%u: background reference pread: %s", parts, strerror(errno)); break; }
            done += (uint64_t)got;
        }
        uint64_t background_before = 0;
        ds4_ds_get_byte_counters(nullptr, nullptr, &background_before);
        memset(buffer, 0x3C, part * 3);
        check(ds4_ds_background_submit(spans, 3) == 1, "parts=%u: background submit failed", parts);
        check(ds4_ds_background_submit(spans, 3) == 0, "parts=%u: second background submit accepted", parts);
        uint64_t bytes = 0;
        double ms = 0.0;
        while (ds4_ds_background_poll(&bytes, &ms) == 0) sched_yield();
        check(ds4_ds_background_finish() == 1, "parts=%u: background finish reported an error", parts);
        uint64_t background_after = 0;
        ds4_ds_get_byte_counters(nullptr, nullptr, &background_after);
        check(background_after - background_before == part * 3,
              "parts=%u: background counter moved %llu, expected %u", parts,
              (unsigned long long)(background_after - background_before), part * 3);
        check(bytes >= part * 3, "parts=%u: background reported %llu bytes, expected at least %u", parts,
              (unsigned long long)bytes, part * 3);
        check(memcmp(reference, buffer, part * 3) == 0, "parts=%u: background bytes differ from pread", parts);
        free(reference);
    }

    /* The engine's own selftest, over the chunk/batch matrix. */
    for (size_t c = 0; c < chunk_count; c++) {
        for (size_t b = 0; b < batch_count; b++) {
            const uint32_t chunk = chunks[c] * (1u << 20);
            const uint32_t batch = batches[b];
            if (batch > capacity) continue;   // batch_add would refuse the span count
            if (chunk > span) continue;
            if (ds4_ds_selftest(fd, 0, span, chunk, batch) != 1) {
                fail("parts=%u chunk=%uMiB batch=%u: selftest failed", parts, chunks[c], batch);
            } else {
                printf("ok  selftest parts=%u chunk=%uMiB batch=%u span=%lluMiB\n",
                       parts, chunks[c], batch, (unsigned long long)(span >> 20));
            }
        }
    }

    uint64_t demand = 0, background = 0;
    double demand_ms = 0.0;
    ds4_ds_get_byte_counters(&demand, &demand_ms, &background);
    check(demand > 0, "parts=%u: demand counter stayed 0", parts);
    check(background > 0, "parts=%u: background counter stayed 0", parts);

    ds4_ds_shutdown();
    check(ds4_ds_available() == 0, "parts=%u: available after shutdown", parts);
    free(buffer);
}

int parse_list(const char *text, uint32_t *out, size_t cap, size_t *count) {
    *count = 0;
    const char *p = text;
    while (*p && *count < cap) {
        char *end = nullptr;
        const unsigned long v = strtoul(p, &end, 10);
        if (end == p || (*end != ',' && *end != '\0')) return 0;
        out[(*count)++] = (uint32_t)v;
        if (*end == ',') p = end + 1;
        else break;
    }
    return *count > 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [chunk_MiB,...] [batch,...]\n", argv[0]);
        return 2;
    }
    uint32_t chunks[8] = {1, 4, 8}, batches[8] = {1, 8, 16};
    size_t chunk_count = 3, batch_count = 3;
    if (argc > 2 && !parse_list(argv[2], chunks, 8, &chunk_count)) {
        fprintf(stderr, "invalid chunk list: %s\n", argv[2]);
        return 2;
    }
    if (argc > 3 && !parse_list(argv[3], batches, 8, &batch_count)) {
        fprintf(stderr, "invalid batch list: %s\n", argv[3]);
        return 2;
    }

    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s: %s\n", argv[1], strerror(errno));
        return 2;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "fstat %s: %s\n", argv[1], strerror(errno));
        return 2;
    }
    const uint64_t span = 32u << 20;
    if ((uint64_t)st.st_size < span * 2) {
        fprintf(stderr, "%s is %.2fMiB; needs at least %.2fMiB\n", argv[1],
                (double)st.st_size / (1024.0 * 1024.0), (double)(span * 2) / (1024.0 * 1024.0));
        return 2;
    }

    /* parts 1 exercises plain requests; parts 8 exercises 4KiB span splitting. */
    run_suite(fd, (uint64_t)st.st_size, 1, span, chunks, chunk_count, batches, batch_count);
    ds4_ds_shutdown();
    run_suite(fd, (uint64_t)st.st_size, 8, span, chunks, chunk_count, batches, batch_count);
    ds4_ds_shutdown();
    close(fd);

    if (g_failures) {
        fprintf(stderr, "BATCH_IO: %d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("BATCH_IO: PASS io_uring transport matched pread (%s, %.2fMiB)\n", argv[1],
           (double)st.st_size / (1024.0 * 1024.0));
    return 0;
}
