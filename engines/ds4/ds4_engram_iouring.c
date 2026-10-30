/* Engram row reader on io_uring (Linux).
 *
 * Why: the pread path issues one 264-byte read per row, one at a time, across ~3M rows per sweep.
 * Random 264-byte reads at queue depth 1 are latency-bound, not bandwidth-bound. io_uring keeps a
 * batch of rows in flight, which is the Linux analogue of the Windows IoRing reader, and the same
 * raw bytes land in the same output buffer so decode_row() output is identical either way.
 *
 * Rows are 264 bytes, so this is buffered I/O: O_DIRECT needs 4 KiB alignment and cannot express a
 * row read. The 1 MiB extent reads are a prefill-side concern, not this path.
 */
#include "ds4_engram.h"

#include <errno.h>
#include <liburing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { ENGRAM_IORING_QUEUE_DEPTH = 64 };

/* Carried in a CQE's user_data so completion can be attributed to a row and the
 * expected byte count checked. Lives in the caller-owned tag array. */
struct engram_row_tag {
    uint32_t index;
    uint32_t bytes;
};

bool ds4_engram_ioring_read_rows(int fd, uint64_t base, const uint32_t *rows, size_t n,
                                 uint8_t *out, char *why, size_t why_n) {
    if (!out || !rows || !why || why_n == 0) {
        snprintf(why, why_n, "bad arguments");
        errno = EINVAL;
        return false;
    }
    if (n == 0) return true;

    struct io_uring ring = {};
    const int init = io_uring_queue_init(ENGRAM_IORING_QUEUE_DEPTH, &ring, 0);
    if (init < 0) {
        snprintf(why, why_n, "io_uring_queue_init: %s", strerror(-init));
        errno = -init;
        return false;
    }

    struct engram_row_tag *tags = calloc(n, sizeof(*tags));
    if (!tags) {
        snprintf(why, why_n, "tag allocation failed");
        io_uring_queue_exit(&ring);
        errno = ENOMEM;
        return false;
    }

    bool ok = true;
    for (size_t start = 0; ok && start < n; start += ENGRAM_IORING_QUEUE_DEPTH) {
        const size_t count = n - start < ENGRAM_IORING_QUEUE_DEPTH
            ? n - start : (size_t)ENGRAM_IORING_QUEUE_DEPTH;
        uint32_t pending = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t row = start + i;
            tags[row] = (struct engram_row_tag){(uint32_t)row, DS4_ENGRAM_ROW_BYTES};
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                snprintf(why, why_n, "SQE ring exhausted at %zu of %zu rows", i, count);
                errno = EAGAIN;
                ok = false;
                break;
            }
            io_uring_prep_read(sqe, fd, out + row * DS4_ENGRAM_ROW_BYTES,
                               DS4_ENGRAM_ROW_BYTES,
                               base + (uint64_t)rows[row] * DS4_ENGRAM_ROW_BYTES);
            io_uring_sqe_set_data(sqe, &tags[row]);
            pending++;
        }
        if (!ok) break;

        const int submitted = io_uring_submit(&ring);
        if (submitted < (int)pending) {
            snprintf(why, why_n, "io_uring_submit returned %d for %u requests", submitted, pending);
            errno = submitted < 0 ? -submitted : EIO;
            ok = false;
            break;
        }

        while (pending > 0) {
            struct io_uring_cqe *cqe = NULL;
            const int got = io_uring_wait_cqe(&ring, &cqe);
            if (got < 0) {
                snprintf(why, why_n, "io_uring_wait_cqe: %s", strerror(-got));
                errno = -got;
                ok = false;
                break;
            }
            const struct engram_row_tag *tag = (const struct engram_row_tag *)cqe->user_data;
            if (!tag) {
                snprintf(why, why_n, "completion carried no row tag");
                errno = EIO;
                ok = false;
                io_uring_cqe_seen(&ring, cqe);
                break;
            }
            if (cqe->res < 0) {
                snprintf(why, why_n, "row %u read FAILED: %s", tag->index, strerror(-cqe->res));
                errno = -cqe->res;
                ok = false;
            } else if ((uint32_t)cqe->res != tag->bytes) {
                snprintf(why, why_n, "row %u read SHORT res=%d expected=%u",
                         tag->index, cqe->res, tag->bytes);
                errno = EIO;
                ok = false;
            }
            io_uring_cqe_seen(&ring, cqe);
            pending--;
        }
    }

    free(tags);
    io_uring_queue_exit(&ring);
    if (!ok) {
        /* Preserve the errno the caller's pread path would have produced. */
        if (errno == 0) errno = EIO;
    }
    return ok;
}
