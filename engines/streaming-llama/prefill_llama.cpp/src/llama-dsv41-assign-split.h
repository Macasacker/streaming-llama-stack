#pragma once
// The split of one decode step's expert-hook wall time (log only; tested in
// tests/dsv41_assign_split.cpp). Emitted as LLAMADECODE_ASSIGN_SPLIT with
// LLAMADECODE_BYTE_LEDGER=1, one line per decoded token, beside LLAMADECODE_STEP_BYTES.
//
// `assign_ms` is what the producer's hook answers in - the provider's `assign_expert_slots`
// from docs/DECODE_GAP_PLAN.md - and it is a third of a step (median 34.5 ms of 58.7 ms,
// docs/PREFILL_DECODE_PLAN_20261110.md). On its own it says only "the fetch and the tiers took
// this long", which cannot tell a storage read from a PCIe copy, and Step 3a of that plan is a
// fetch-rate target. This line is that split, from counters that already exist plus the extent
// batch's own read/stage/upload breakdown.
//
//   assign_ms = fetch_ms + l2_copy_ms + ram_exclusive_ms + overlap_after_ms + bookkeep_ms
//
// `fetch_ms` already contains the RAM-tier work that ran beside the reads (that is what its
// overlap hook is for), so `ram_overlap_in_fetch_ms` is reported but never added again.
// `upload_device_ms` is the device timeline of the copies and overlaps the host-side fields;
// it is not part of the sum.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace llama_dsv41_assign_split {

struct parts {
    uint32_t position = 0;
    double assign_ms = 0;   // the whole hook, as the provider timed it
    double hook_ms = 0;     // the provider's inner hook timer (a subset of assign_ms)
    double fetch_ms = 0;    // named-extent batch wall time inside the hook
    double read_ms = 0;     // blocked on storage: io_uring submit/wait, or pread
    double stage_ms = 0;    // blocked on a pinned slot an earlier upload still holds
    double upload_issue_ms = 0;   // cudaMemcpyAsync + event record, host side
    double upload_device_ms = 0;  // the copies' own device timeline, overlapping the three above
    uint64_t fetch_bytes = 0;
    uint64_t batches = 0;
    uint64_t upload_windows_unread = 0;   // device windows not yet complete at the next batch
    double l2_copy_ms = 0;  // VRAM L2 slot copies and the stream sync that waits for them
    double ram_ms = 0;      // RAM-tier upload, store and their waits
    double ram_overlap_ms = 0;        // RAM-tier work run beside an extent fetch
    double ram_overlap_in_fetch_ms = 0;   // of that, the part inside fetch_ms
    uint32_t misses = 0;
};

// The parts of `assign_ms`, each counted once.
inline double ram_exclusive_ms(const parts & p) {
    return std::max(0.0, p.ram_ms - p.ram_overlap_ms);
}
inline double overlap_after_fetch_ms(const parts & p) {
    return std::max(0.0, p.ram_overlap_ms - p.ram_overlap_in_fetch_ms);
}
inline double accounted_ms(const parts & p) {
    return p.fetch_ms + p.l2_copy_ms + ram_exclusive_ms(p) + overlap_after_fetch_ms(p);
}
// What the hook spent on slot bookkeeping, tier lookups and everything the counters above
// do not name. Zero when the counters already exceed the hook: that cannot be bookkeeping,
// and `over_attributed_ms` says by how much the counters overshot.
inline double bookkeep_ms(const parts & p) {
    const double spent = accounted_ms(p);
    return p.assign_ms > spent ? p.assign_ms - spent : 0.0;
}
inline double over_attributed_ms(const parts & p) {
    const double spent = accounted_ms(p);
    return spent > p.assign_ms ? spent - p.assign_ms : 0.0;
}
inline double host_wall_ms(const parts & p) { return p.read_ms + p.stage_ms + p.upload_issue_ms; }
// The batch's wall time not attributed to a wait (bounds checks, the extent mutex).
inline double fetch_remainder_ms(const parts & p) {
    const double wall = host_wall_ms(p);
    return p.fetch_ms > wall ? p.fetch_ms - wall : 0.0;
}

inline std::string fixed(double value) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.3f", value);
    return std::string(buffer);
}
// Decimal GB/s, the same convention `[dsv41-fetch]` uses. A zero denominator reads as 0
// rather than inf: a step with no misses has no read to rate.
inline std::string rate(uint64_t bytes, double ms) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.3f", ms > 0.0 ? double(bytes) / ms / 1e6 : 0.0);
    return std::string(buffer);
}

inline std::string line(const parts & p) {
    return "LLAMADECODE_ASSIGN_SPLIT position=" + std::to_string(p.position) +
        " assign_ms=" + fixed(p.assign_ms) + " fetch_ms=" + fixed(p.fetch_ms) +
        " read_ms=" + fixed(p.read_ms) + " stage_ms=" + fixed(p.stage_ms) +
        " upload_issue_ms=" + fixed(p.upload_issue_ms) + " upload_device_ms=" + fixed(p.upload_device_ms) +
        " fetch_remainder_ms=" + fixed(fetch_remainder_ms(p)) +
        " fetch_bytes=" + std::to_string(p.fetch_bytes) + " batches=" + std::to_string(p.batches) +
        " read_GB_s=" + rate(p.fetch_bytes, p.read_ms) +
        " upload_GB_s=" + rate(p.fetch_bytes, p.upload_device_ms) +
        " l2_copy_ms=" + fixed(p.l2_copy_ms) + " ram_ms=" + fixed(p.ram_ms) +
        " ram_exclusive_ms=" + fixed(ram_exclusive_ms(p)) +
        " ram_overlap_ms=" + fixed(p.ram_overlap_ms) +
        " ram_overlap_in_fetch_ms=" + fixed(p.ram_overlap_in_fetch_ms) +
        " bookkeep_ms=" + fixed(bookkeep_ms(p)) + " over_attributed_ms=" + fixed(over_attributed_ms(p)) +
        " hook_ms=" + fixed(p.hook_ms) + " misses=" + std::to_string(p.misses) +
        " upload_windows_unread=" + std::to_string(p.upload_windows_unread);
}

} // namespace llama_dsv41_assign_split
