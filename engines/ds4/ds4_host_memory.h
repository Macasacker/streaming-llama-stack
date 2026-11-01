#ifndef DS4_HOST_MEMORY_H
#define DS4_HOST_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Host-memory admission for allocations the engine makes outside VRAM.
 * Refusal sets errno to ENOMEM and prints a HOST_MEMORY_* line to stderr. */
typedef struct ds4_host_memory_status {
    uint64_t physical_total_bytes;
    uint64_t physical_available_bytes;
    uint64_t system_commit_bytes;       /* cgroup memory.current, else this process RSS */
    uint64_t system_commit_limit_bytes; /* cgroup memory.max, else MemTotal + SwapTotal */
    uint64_t process_private_bytes;
    uint64_t process_private_peak_bytes;
    uint64_t process_limit_bytes;
    uint64_t reserve_bytes;
} ds4_host_memory_status;

#ifdef __cplusplus
extern "C" {
#endif

/* Derive the process cap once, from measured memory. Also starts the watchdog
 * unless DS4_EXTERNAL_DECODE or DS4_DISABLE_HOST_MEMORY_WATCHDOG. */
bool ds4_host_memory_begin(void);
/* host_allocation=true checks the physical reserve as well. */
bool ds4_host_memory_check(uint64_t requested_bytes, const char *reason);
bool ds4_host_memory_check_commit(uint64_t requested_bytes, const char *reason);
/* Pure validation of a measured snapshot, without allocating. */
bool ds4_host_memory_fits_snapshot(const ds4_host_memory_status *state,
                                   uint64_t requested_bytes, bool host_allocation);
bool ds4_host_memory_read(ds4_host_memory_status *out);
void ds4_host_memory_snapshot(const char *reason);
uint64_t ds4_host_memory_peak(void);
/* Owner rule: never force-kill while VRAM may be held. After the first GPU
 * allocation a breach requests a graceful abort instead of an exit. */
void ds4_host_memory_request_graceful_abort(const char *why);
bool ds4_host_memory_graceful_abort_requested(void);
/* Flags that GPU resources are live, so a budget breach aborts the request instead
 * of force-killing the process. No caller in this build yet: the engine reaches the
 * same state through its own first-allocation path, and the Linux producer (chunk
 * 3.1) is the intended caller. */
void ds4_host_memory_mark_gpu_resources_live(void);
bool ds4_host_memory_gpu_resources_live(void);
/* Optional reclaim hint for a mapped file view; the mapping survives. */
int ds4_host_memory_trim_mapped_pages(const void *addr, size_t length);
void ds4_host_memory_enable_utf8_console(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_HOST_MEMORY_H */
