// Does io_uring_wait_cqe() flush submission-queue entries that were never
// submitted? Both of the producer's io_uring readers queue reads and then wait,
// and io_uring_submit() can return 0 without being an error, so a wait that does
// not flush would leave a layer's reads unseen forever.
//
// On this kernel and liburing the wait does NOT flush: a queued read with no
// submit has no completion to peek, and it completes only once the SQ is
// submitted - which is why both readers wait through
// io_uring_submit_and_wait(). This program asserts exactly that, and exits 0.
//
// Build and run: see docs/linux-port-commands.md ("Does the io_uring wait flush
// the submission queue?").
#include <liburing.h>

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static int failures = 0;
static void check(const char * what, bool ok) {
    std::printf("%-6s %s\n", ok ? "ok" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}

int main(void) {
    const char * path = "build/uring-flush-probe.bin";
    const int fd = ::open(path, O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) { std::perror("open"); return 1; }
    char payload[4096];
    std::memset(payload, 0xa5, sizeof(payload));
    if (::write(fd, payload, sizeof(payload)) != sizeof(payload)) { std::perror("write"); return 1; }
    struct io_uring ring{};
    if (io_uring_queue_init(8, &ring, 0) != 0) { std::perror("io_uring_queue_init"); return 1; }

    char buf[4096]{};
    struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
    if (!sqe) { std::printf("FAIL the ring gave no SQE\n"); return 1; }
    io_uring_prep_read(sqe, fd, buf, sizeof(buf), 0);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(1));

    struct io_uring_cqe * cqe = nullptr;
    const int first = io_uring_peek_cqe(&ring, &cqe);
    std::printf("info first peek returned %d with cqe=%p\n", first, (void*)cqe);
    check("a queued read with no submit has no completion to wait for", first != 0 || cqe == nullptr);
    const int entered = io_uring_submit_and_wait(&ring, 1);
    const int second = io_uring_peek_cqe(&ring, &cqe);
    std::printf("info submit_and_wait=%d second peek=%d cqe=%p\n", entered, second, (void*)cqe);
    check("submit-and-wait flushes it and one completion arrives", entered >= 0 && second == 0 && cqe != nullptr);
    if (cqe) {
        std::printf("info res=%ld user_data=%p buf[0]=0x%02x\n", (long)cqe->res, io_uring_cqe_get_data(cqe), (unsigned)(unsigned char)buf[0]);
        check("the read delivered the whole buffer", cqe->res == static_cast<int>(sizeof(buf)) && buf[0] == (char)0xa5);
        io_uring_cqe_seen(&ring, cqe);
    } else check("the read delivered the whole buffer", false);

    io_uring_queue_exit(&ring);
    ::close(fd);
    std::remove(path);
    std::printf("SELFTEST %s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
