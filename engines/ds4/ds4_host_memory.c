/* Host-memory admission for the Linux CUDA engine.
 *
 * Same rules as the engine this fork came from, read from Linux sources:
 *   physical reserve   -> MemAvailable from /proc/meminfo
 *   process private    -> VmRSS / VmPeak from /proc/self/status
 *   system commit      -> cgroup v2 memory.current and memory.max when the process
 *                         is inside a scoped cgroup, else MemTotal + SwapTotal
 *   process cap        -> the same derived ceiling, overridable with DS4_HOST_MEMORY_BUDGET_MB
 *
 * Owner rule: never force-kill this process while it may hold
 * VRAM. A breach after the first GPU allocation requests a graceful abort instead, so
 * the current request fails and new ones are refused while the process stays up.
 */
#include "ds4_host_memory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define DS4_HOST_RESERVE_BYTES (UINT64_C(512) << 20)

static pthread_mutex_t g_host_memory_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_host_memory_limit;
static int g_host_memory_ready;
static volatile int g_graceful_abort;
static volatile int g_gpu_resources_live;
static uint64_t g_rss_peak_bytes;

static uint64_t read_u64_field(const char *path, const char *field, uint64_t scale) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    uint64_t value = 0;
    size_t field_len = strlen(field);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, field_len) != 0) continue;
        unsigned long long v = 0;
        if (sscanf(line + field_len, "%llu", &v) == 1) value = (uint64_t)v * scale;
        break;
    }
    fclose(f);
    return value;
}

/* The cgroup this process lives in, when systemd-run --scope gave it one.
 * Reads memory.max ("max" for unlimited reads as 0) and memory.current. */
static int cgroup_self(char *self, size_t cap) {
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (!f) return 0;
    char line[512];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "0::/", 4) != 0) continue;
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (strlen(line) < cap) {
            snprintf(self, cap, "%s", line);
            found = 1;
        }
    }
    fclose(f);
    return found;
}

static uint64_t cgroup_limit_bytes(void) {
    char self[512] = "0::/";
    if (!cgroup_self(self, sizeof(self))) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.max", self);
    FILE *m = fopen(path, "r");
    if (!m) return 0;
    char value[64] = {0};
    uint64_t limit = 0;
    if (fgets(value, sizeof(value), m) && strcmp(value, "max\n") != 0) {
        unsigned long long v = 0;
        if (sscanf(value, "%llu", &v) == 1) limit = (uint64_t)v;
    }
    fclose(m);
    return limit;
}

static uint64_t cgroup_current_bytes(void) {
    char self[512] = "0::/";
    if (!cgroup_self(self, sizeof(self))) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.current", self);
    return read_u64_field(path, "memory.current", 1);
}

static uint64_t process_rss_bytes(void) {
    uint64_t rss = read_u64_field("/proc/self/status", "VmRSS:", 1024);
    if (rss > g_rss_peak_bytes) g_rss_peak_bytes = rss;
    return rss;
}

uint64_t ds4_host_memory_peak(void) { return g_rss_peak_bytes; }

bool ds4_host_memory_read(ds4_host_memory_status *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    out->physical_total_bytes = read_u64_field("/proc/meminfo", "MemTotal:", 1024);
    out->physical_available_bytes = read_u64_field("/proc/meminfo", "MemAvailable:", 1024);
    if (!out->physical_total_bytes || !out->physical_available_bytes) return false;
    const uint64_t swap = read_u64_field("/proc/meminfo", "SwapTotal:", 1024);
    const uint64_t cgroup = cgroup_limit_bytes();
    out->system_commit_limit_bytes = cgroup ? cgroup : out->physical_total_bytes + swap;
    const uint64_t current = cgroup_current_bytes();
    out->system_commit_bytes = current ? current : process_rss_bytes();
    out->process_private_bytes = process_rss_bytes();
    out->process_private_peak_bytes = g_rss_peak_bytes;
    out->process_limit_bytes = g_host_memory_limit;
    out->reserve_bytes = DS4_HOST_RESERVE_BYTES;
    return true;
}

static uint64_t host_commit_ceiling(const ds4_host_memory_status *s) {
    return s->system_commit_limit_bytes > DS4_HOST_RESERVE_BYTES ?
        s->system_commit_limit_bytes - DS4_HOST_RESERVE_BYTES : 0;
}

static void host_memory_report(const char *event, const char *reason,
                               uint64_t requested, const ds4_host_memory_status *s) {
    char line[1024];
    const int n = snprintf(line, sizeof(line),
        "ds4: HOST_MEMORY_%s reason=%s requested_bytes=%llu "
        "private_bytes=%llu private_peak_bytes=%llu private_limit_bytes=%llu "
        "available_physical_bytes=%llu total_physical_bytes=%llu "
        "system_commit_bytes=%llu system_commit_limit_bytes=%llu system_commit_budget_bytes=%llu reserve_bytes=%llu\n",
        event, reason ? reason : "unspecified",
        (unsigned long long)requested,
        (unsigned long long)s->process_private_bytes,
        (unsigned long long)s->process_private_peak_bytes,
        (unsigned long long)s->process_limit_bytes,
        (unsigned long long)s->physical_available_bytes,
        (unsigned long long)s->physical_total_bytes,
        (unsigned long long)s->system_commit_bytes,
        (unsigned long long)s->system_commit_limit_bytes,
        (unsigned long long)host_commit_ceiling(s),
        (unsigned long long)s->reserve_bytes);
    if (n > 0) {
        fwrite(line, 1, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1u, stderr);
        fflush(stderr);
    }
}

static const char *host_memory_failure(uint64_t requested,
        const ds4_host_memory_status *s, bool host_allocation) {
    if (!s) return "REFUSED_QUERY";
    if (ds4_host_memory_graceful_abort_requested()) return "REFUSED_GRACEFUL_ABORT";
    const uint64_t ceiling = host_commit_ceiling(s);
    const char *failure = NULL;
    if (s->process_private_bytes > s->process_limit_bytes ||
        requested > s->process_limit_bytes - s->process_private_bytes)
        failure = "REFUSED_PRIVATE_LIMIT";
    else if (s->physical_available_bytes < DS4_HOST_RESERVE_BYTES ||
        (host_allocation && requested > s->physical_available_bytes - DS4_HOST_RESERVE_BYTES))
        failure = "REFUSED_PHYSICAL_RESERVE";
    else if (s->system_commit_bytes > ceiling ||
        requested > ceiling - s->system_commit_bytes)
        failure = "REFUSED_SYSTEM_COMMIT_BUDGET";
    return failure;
}

bool ds4_host_memory_fits_snapshot(const ds4_host_memory_status *state,
        uint64_t requested_bytes, bool host_allocation) {
    return host_memory_failure(requested_bytes, state, host_allocation) == NULL;
}

static bool host_memory_fits(uint64_t requested, const char *reason,
        const ds4_host_memory_status *s, bool host_allocation) {
    const char *failure = host_memory_failure(requested, s, host_allocation);
    if (!failure) return true;
    host_memory_report(failure, reason, requested, s);
    errno = ENOMEM;
    return false;
}

static uint64_t parse_budget_override(uint64_t cap, bool *ok) {
    const char *override = getenv("DS4_HOST_MEMORY_BUDGET_MB");
    if (!override) return cap;
    uint64_t mb = 0;
    const char *p = override;
    bool valid = *p != '\0';
    while (*p) {
        if (*p < '0' || *p > '9' || mb > (UINT64_MAX - (uint64_t)(*p - '0')) / 10u) {
            valid = false;
            break;
        }
        mb = mb * 10u + (uint64_t)(*p++ - '0');
    }
    if (!valid || !mb || mb > UINT64_MAX / UINT64_C(1048576) || mb * UINT64_C(1048576) > cap) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_BUDGET: DS4_HOST_MEMORY_BUDGET_MB must be a positive integer no larger than safe cap %llu bytes\n",
                (unsigned long long)cap);
        *ok = false;
        return 0;
    }
    *ok = true;
    return mb * UINT64_C(1048576);
}

static void *host_memory_watchdog(void *unused) {
    (void)unused;
    for (;;) {
        struct timespec req = {0, 250 * 1000 * 1000};
        nanosleep(&req, NULL);
        ds4_host_memory_status state;
        if (!ds4_host_memory_read(&state)) {
            memset(&state, 0, sizeof(state));
            host_memory_report("WATCHDOG_QUERY_FAILED", "host watchdog", 0, &state);
            ds4_host_memory_request_graceful_abort("host watchdog query failed");
            continue;
        }
        if (!ds4_host_memory_graceful_abort_requested() &&
            !host_memory_fits(0, "host watchdog", &state, false)) {
            host_memory_report("WATCHDOG_ABORT", "host reserve breached", 0, &state);
            ds4_host_memory_request_graceful_abort("host reserve breached");
        }
    }
    return NULL;
}
bool ds4_host_memory_begin(void) {
    pthread_mutex_lock(&g_host_memory_lock);
    if (g_host_memory_ready) {
        pthread_mutex_unlock(&g_host_memory_lock);
        return true;
    }
    bool ok = false;
    ds4_host_memory_status state;
    if (!ds4_host_memory_read(&state)) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_QUERY: cannot measure memory at startup\n");
        goto done;
    }
    if (state.physical_total_bytes <= DS4_HOST_RESERVE_BYTES ||
        state.physical_available_bytes < DS4_HOST_RESERVE_BYTES) {
        host_memory_report("REFUSED_INITIAL_PHYSICAL", "startup", 0, &state);
        goto done;
    }
    const uint64_t ceiling = host_commit_ceiling(&state);
    uint64_t cap = state.system_commit_bytes > ceiling ? UINT64_MAX : ceiling - state.system_commit_bytes;
    cap = state.process_private_bytes > UINT64_MAX - cap ? UINT64_MAX : state.process_private_bytes + cap;
    bool budget_ok = true;
    cap = parse_budget_override(cap, &budget_ok);
    if (!budget_ok) goto done;
    if (cap <= state.process_private_bytes) {
        host_memory_report("REFUSED_INITIAL_PRIVATE_LIMIT", "startup", 0, &state);
        goto done;
    }
    g_host_memory_limit = cap;
    g_host_memory_ready = 1;
#ifdef DS4_EXTERNAL_DECODE
    /* ds4-decode.so: the producer owns admission and lifetime. No cgroup cap and
     * no watchdog thread here, which could abort the host process. */
    host_memory_report("ADMITTED_EXTERNAL_NO_SCOPE_NO_WATCHDOG", "startup", 0, &state);
    ok = true;
    goto done;
#endif
    if (!getenv("DS4_DISABLE_HOST_MEMORY_WATCHDOG")) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, host_memory_watchdog, NULL) != 0) {
            fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_WATCHDOG: cannot start host watchdog (%d)\n", errno);
            g_host_memory_limit = 0;
            g_host_memory_ready = 0;
            goto done;
        }
        pthread_detach(thread);
    }
    host_memory_report("ADMITTED", "startup", 0, &state);
    ok = true;
done:
    pthread_mutex_unlock(&g_host_memory_lock);
    return ok;
}

static bool host_memory_check_kind(uint64_t requested_bytes, const char *reason,
        bool host_allocation) {
    if (!ds4_host_memory_begin()) return false;
    ds4_host_memory_status state;
    if (!ds4_host_memory_read(&state)) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_QUERY: cannot measure memory for %s\n",
                reason ? reason : "allocation");
        return false;
    }
    return host_memory_fits(requested_bytes, reason, &state, host_allocation);
}

bool ds4_host_memory_check(uint64_t requested_bytes, const char *reason) {
    return host_memory_check_kind(requested_bytes, reason, true);
}

bool ds4_host_memory_check_commit(uint64_t requested_bytes, const char *reason) {
    return host_memory_check_kind(requested_bytes, reason, false);
}

void ds4_host_memory_snapshot(const char *reason) {
    ds4_host_memory_status state;
    if (ds4_host_memory_read(&state))
        host_memory_report("SNAPSHOT", reason ? reason : "unspecified", 0, &state);
    else
        fprintf(stderr, "ds4: HOST_MEMORY_SNAPSHOT_UNAVAILABLE reason=%s\n",
                reason ? reason : "unspecified");
}

void ds4_host_memory_request_graceful_abort(const char *why) {
    int expected = 0;
    if (__sync_val_compare_and_swap(&g_graceful_abort, expected, 1) == 0) {
        fprintf(stderr, "DS4_GRACEFUL_ABORT_REQUESTED reason=%s: refusing new allocations and failing the "
                        "current request; the process stays up for a graceful shutdown (never force-killed "
                        "while it may hold VRAM)\n", why ? why : "unspecified");
        fflush(stderr);
    }
}

bool ds4_host_memory_graceful_abort_requested(void) {
    return g_graceful_abort != 0;
}

void ds4_host_memory_mark_gpu_resources_live(void) {
    g_gpu_resources_live = 1;
}

bool ds4_host_memory_gpu_resources_live(void) {
    return g_gpu_resources_live != 0;
}

int ds4_host_memory_trim_mapped_pages(const void *addr, size_t length) {
    if (!addr || !length) return 0;
    return madvise((void *)addr, length, MADV_DONTNEED) == 0;
}

void ds4_host_memory_enable_utf8_console(void) { /* stdout is UTF-8 on Linux already */ }
