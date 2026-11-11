// LLAMADECODE_ASSIGN_SPLIT: the arithmetic and the line format of one decode step's
// expert-hook split (docs/PREFILL_DECODE_PLAN_20261110.md Step 0).
// Hand-compiled, like tests/dsv41_step_ledger.cpp:
//   c++ -std=c++17 -I engines/streaming-llama/prefill_llama.cpp/src tests/dsv41_assign_split.cpp
// The numbers are the shape of the measured median step: assign_ms 34.5 of a 58.7 ms step,
// 0.102 GiB fetched, and a RAM tier that sometimes runs beside the extent fetch.
#include "llama-dsv41-assign-split.h"
#include <cstdio>
#include <string>

namespace h = llama_dsv41_assign_split;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool has(const std::string & text, const char * field) {
    return text.find(field) != std::string::npos;
}

int main() {
    const uint64_t bytes = 109547110;                    // 0.102 GiB, the plan's median step
    h::parts p;
    p.position = 41;
    p.assign_ms = 34.5;                                  // the hook the step pays
    p.hook_ms = 33.9;
    p.fetch_ms = 31.0;                                   // extent batches inside the hook
    p.read_ms = 20.0; p.stage_ms = 8.0; p.upload_issue_ms = 2.0;
    p.upload_device_ms = 18.0;                           // overlaps the three host waits
    p.fetch_bytes = bytes; p.batches = 3;
    p.l2_copy_ms = 1.0;
    p.ram_ms = 2.0; p.ram_overlap_ms = 0.5; p.ram_overlap_in_fetch_ms = 0.5;
    p.misses = 11;
    const auto text = h::line(p);
    std::printf("%s\n", text.c_str());

    expect(has(text, "LLAMADECODE_ASSIGN_SPLIT position=41 assign_ms=34.500 "),
           "tag, position and the hook's own wall time lead the line");
    expect(h::host_wall_ms(p) == 30.0, "read, stage and upload issue are the batch's host wall");
    expect(has(text, "fetch_remainder_ms=1.000 "), "the 1 ms of batch the waits do not name is reported, not hidden");
    expect(h::accounted_ms(p) == 33.5, "fetch plus L2 copies plus the RAM tier accounts for 33.5 of 34.5 ms");
    expect(has(text, "bookkeep_ms=1.000 ") && has(text, "over_attributed_ms=0.000 "),
           "the rest of the hook is named as bookkeeping");
    expect(has(text, "ram_exclusive_ms=1.500 "), "RAM work that ran inside the fetch is not counted twice");
    expect(has(text, "read_GB_s=5.477 ") && has(text, "upload_GB_s=6.086 "),
           "both halves of the relay carry their own decimal rate");

    // The same counters with the overlap on the other side of the fetch charge the RAM tier
    // differently, and that is the whole reason ram_overlap_in_fetch_ms exists: fetch_ms already
    // contains the work the fetch's overlap hook did, so counting that part again in the tier's
    // total would read as a hook shorter than its own counters.
    h::parts after = p; after.ram_overlap_ms = 0.8; after.ram_overlap_in_fetch_ms = 0.0;
    expect(h::accounted_ms(after) == 34.0,
           "a RAM tier that ran beside nothing of the fetch is charged in full");
    expect(h::accounted_ms(after) - h::accounted_ms(p) == 0.5,
           "and only the part that ran inside fetch_ms is discounted, not the whole overlap");
    expect(has(h::line(after), "ram_exclusive_ms=1.200 ") && has(h::line(after), "bookkeep_ms=0.500 "),
           "the 0.8 ms of overlap that ran after the fetch is charged as the tier's own time");

    // Counters that overshoot the hook are reported as overshoot, never as negative bookkeeping.
    h::parts over = p; over.assign_ms = 20.0;
    const auto ot = h::line(over);
    expect(has(ot, "bookkeep_ms=0.000 ") && has(ot, "over_attributed_ms=13.500 "),
           "a hook shorter than its own counters says so instead of reading as negative work");

    // A step with no misses has no fetch to rate, and must not divide by zero.
    h::parts idle;
    idle.position = 2; idle.assign_ms = 3.2;
    const auto it = h::line(idle);
    expect(has(it, "read_GB_s=0.000 ") && has(it, "upload_GB_s=0.000 "),
           "a step that fetched nothing reports no rate rather than a division by zero");
    expect(h::fetch_remainder_ms(idle) == 0.0 && h::bookkeep_ms(idle) == 3.2,
           "and all of its hook is bookkeeping");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
