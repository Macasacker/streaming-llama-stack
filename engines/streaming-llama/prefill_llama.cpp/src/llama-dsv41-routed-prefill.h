#pragma once
// Which layers demand-read their routed experts, and for how long that answer holds
// (log-and-policy only; tested in tests/dsv41_routed_prefill.cpp).
//
// `LLAMADECODE_PREFILL_ROUTED` is read once, when the weight provider builds its pipeline, so
// it is the producer's setting for its whole life. A resident single-engine producer is
// launched with turn 1's prompt and appends every later turn to itself, which is how an agent
// session that opened at 16 666 tokens kept reading whole stages for layers 0-20 on each
// 45-token append: 21 layers x ~630 ms of prefetch for 5 new experts, 13.2 s of an 18.6-22.5 s
// turn (docs/PREFILL_DECODE_PLAN_20261110.md).
//
// `auto` - the default - resolves per sweep from that sweep's own new-row count, the rule
// `combo_producer.routed_prefill_choice` already applies: demand-read every layer at <= 2 048
// rows, and above it keep the overlapped whole-stage prefetch for the early layers, where a
// full prompt measured 843 ms routed against 600 ms prefetched. An explicit 0/1/tail still
// pins the mode for the producer's life, which is how a matched A/B stays matched.
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace llama_dsv41_routed_prefill {

enum class mode { off, all, tail, sweep };

// The first layer that demand-reads, for one sweep. `off` answers past the last layer: no
// layer routes and every stage is prefetched whole.
constexpr uint32_t layer_count = 40;
constexpr uint32_t tail_from_layer = 21;
constexpr uint64_t sweep_rows_limit = 2048;

inline mode parse(const char * value) {
    if (!value || !*value) return mode::sweep;
    if (!std::strcmp(value, "0")) return mode::off;
    if (!std::strcmp(value, "1")) return mode::all;
    if (!std::strcmp(value, "tail")) return mode::tail;
    if (!std::strcmp(value, "auto")) return mode::sweep;
    throw std::invalid_argument("LLAMADECODE_PREFILL_ROUTED must be 0, 1, tail or auto");
}

// True when the environment named a mode, so no sweep may change it.
inline bool pinned(mode m) { return m != mode::sweep; }

inline uint32_t from_layer(mode m, uint64_t rows) {
    if (m == mode::off) return layer_count;
    if (m == mode::all) return 0;
    if (m == mode::tail) return tail_from_layer;
    return rows <= sweep_rows_limit ? 0u : tail_from_layer;
}

// Whether the provider routes at all. Everything except `off` routes from `from_layer` up.
inline bool routes(mode m) { return m != mode::off; }

// What DS4PERF_ROUTED_PREFILL names for a sweep: the resolved shape, not the switch that asked
// for it. `source` says which of the two decided it.
inline const char * resolved_name(uint32_t from) {
    return from >= layer_count ? "off" : from == 0u ? "all" : "tail";
}

// What DS4PERF_ROUTED_PREFILL names: the resolved shape, not the switch that asked for it.
inline const char * name(mode m) {
    switch (m) {
        case mode::off: return "off";
        case mode::all: return "all";
        case mode::tail: return "tail";
        case mode::sweep: return "sweep";
    }
    return "off";
}

} // namespace llama_dsv41_routed_prefill
