// LLAMADECODE decode-step byte ledger: tier arithmetic and line format.
// Numbers are the shape of one real batch-1 decode step: 6 experts x 40 layers,
// a ~10 MB expert slice, so a 240-triple step needs ~2.4 GB.
#include "llama-dsv41-step-ledger.h"
#include <cstdio>

namespace h = llama_dsv41_step_ledger;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t triple = 9950000;                       // one (gate,up,down) expert slice
    const uint64_t slots = 240;                            // 6 experts x 40 layers
    h::step s;
    s.position = 7; s.step_ms = 964.3; s.experts = slots; s.slot_bytes = triple;
    s.from_file = 40 * triple;      // 40 misses, one per layer
    s.from_ram = 60 * triple;       // 60 RAM-tier uploads
    s.from_l2 = 40 * triple;        // 40 second-tier hits
    s.from_l1 = 100 * triple;       // 100 already resident in VRAM
    s.copied = 80 * triple;         // victim out + back copies
    s.stored = 40 * triple;
    s.l1_hits = 100; s.l2_hits = 40; s.ram_hits = 60; s.misses = 40;
    const auto text = h::line(s);
    std::printf("%s\n", text.c_str());
    expect(h::footprint(s) == slots * triple, "footprint is experts times the triple's bytes");
    expect(h::served(s) == s.from_file + s.from_ram + s.from_l2 + s.from_l1, "every served byte is attributed to a tier");
    expect(text.find("unattributed=0 ") != std::string::npos && text.find("over_served=0 ") != std::string::npos,
           "a fully attributed step reports no unattributed and no over-served bytes");
    expect(text.find("disk_percent=16.7 ") != std::string::npos, "40 of 240 triples came from the file");
    expect(text.find("footprint=2388000000 ") != std::string::npos, "the step's footprint is the ~2.4 GB the plan counts");
    expect(text.find("LLAMADECODE_STEP_BYTES position=7 step_ms=964.300 ") == 0, "tag, position and step time lead the line");

    // A step the ledger cannot fully attribute must say so rather than look clean.
    h::step leak = s; leak.from_l1 = 0; leak.l1_hits = 0;
    const auto lt = h::line(leak);
    expect(lt.find("unattributed=995000000 ") != std::string::npos && lt.find("over_served=0 ") != std::string::npos,
           "100 triples no tier claims are reported as unattributed");
    h::step extra = s; extra.experts = 200;
    expect(h::line(extra).find("over_served=398000000 ") != std::string::npos && h::line(extra).find("unattributed=0 ") != std::string::npos,
           "and serving more triples than the step asked for is reported as over_served");
    h::step short_step = s; short_step.from_l1 -= 3 * triple;
    expect(h::line(short_step).find("unattributed=29850000 ") != std::string::npos, "bytes no tier claims are reported");

    // An empty step (no decode layout yet) must not divide by zero.
    h::step none; none.step_ms = 12.5;
    const auto nt = h::line(none);
    expect(nt.find("footprint=0 served=0 ") != std::string::npos && nt.find("disk_percent=0.0 ") != std::string::npos,
           "a step with no footprint reports zero percent from disk");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
