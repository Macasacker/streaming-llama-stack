#pragma once
// LLAMADECODE decode-step byte ledger (log only; tested in tests/dsv41_step_ledger.cpp).
// One decoded token routes six experts in every one of the 40 layers, so a step's
// footprint is the number of expert triples it asked the cache to supply times the
// bytes of one triple. This line says which tier each of those bytes came from: the
// model file (NVMe), the RAM expert tier, the VRAM L2 tier, or VRAM L1 residency
// (which moves nothing). Copied and stored bytes are listed separately because they
// are traffic a step pays without adding footprint. It changes no cache decision.
#include <cstdint>
#include <cstdio>
#include <string>

namespace llama_dsv41_step_ledger {

struct step {
    uint32_t position = 0;
    double step_ms = 0;
    uint64_t experts = 0;     // expert triples this step asked the cache to supply
    uint64_t slot_bytes = 0;  // bytes of one triple in the cache's current geometry
    uint64_t from_file = 0;   // read from the model file into VRAM L1
    uint64_t from_ram = 0;    // uploaded from the RAM expert tier
    uint64_t from_l2 = 0;     // served from the VRAM L2 tier (copied into L1, or read in place)
    uint64_t from_l1 = 0;     // already resident in VRAM L1: footprint, nothing moved
    uint64_t copied = 0;      // device-to-device bytes (L1<->L2 slot copies)
    uint64_t stored = 0;      // bytes copied into the RAM tier (L1/L2 victims)
    uint64_t l1_hits = 0, l2_hits = 0, ram_hits = 0, misses = 0;
};

// What the step needed, and what the tiers actually gave it. A step whose tiers are
// fully accounted has both differences below at zero.
inline uint64_t footprint(const step & s) { return s.experts * s.slot_bytes; }
inline uint64_t served(const step & s) { return s.from_file + s.from_ram + s.from_l2 + s.from_l1; }

// LLAMADECODE_STEP_BYTES position=P step_ms=M experts=N slot_bytes=S footprint=F served=X
//   from_file=A from_ram=B from_l2=C from_l1=D disk_percent=P unattributed=U over_served=V
//   copied=W stored=T l1_hits=.. l2_hits=.. ram_hits=.. misses=..
inline std::string line(const step & s) {
    const uint64_t need = footprint(s), got = served(s);
    char disk[16] = "0.0";
    if (need) std::snprintf(disk, sizeof(disk), "%.1f", 100.0 * double(s.from_file) / double(need));
    std::string out = "LLAMADECODE_STEP_BYTES position=" + std::to_string(s.position) +
        " step_ms=" + [&]{char b[16]; std::snprintf(b, sizeof(b), "%.3f", s.step_ms); return std::string(b); }() +
        " experts=" + std::to_string(s.experts) + " slot_bytes=" + std::to_string(s.slot_bytes) +
        " footprint=" + std::to_string(need) + " served=" + std::to_string(got) +
        " from_file=" + std::to_string(s.from_file) + " from_ram=" + std::to_string(s.from_ram) +
        " from_l2=" + std::to_string(s.from_l2) + " from_l1=" + std::to_string(s.from_l1) +
        " disk_percent=" + disk +
        " unattributed=" + std::to_string(need > got ? need - got : 0) +
        " over_served=" + std::to_string(got > need ? got - need : 0) +
        " copied=" + std::to_string(s.copied) + " stored=" + std::to_string(s.stored) +
        " l1_hits=" + std::to_string(s.l1_hits) + " l2_hits=" + std::to_string(s.l2_hits) +
        " ram_hits=" + std::to_string(s.ram_hits) + " misses=" + std::to_string(s.misses);
    return out;
}

} // namespace llama_dsv41_step_ledger
