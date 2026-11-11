// The routed-prefill choice: what the environment names, what a sweep resolves, and what the
// registered spans have to agree with (docs/PREFILL_DECODE_PLAN_20261110.md Step 1).
// Hand-compiled, like tests/dsv41_step_ledger.cpp:
//   c++ -std=c++17 -I engines/streaming-llama/prefill_llama.cpp/src tests/dsv41_routed_prefill.cpp
#include "llama-dsv41-routed-prefill.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace h = llama_dsv41_routed_prefill;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    expect(h::parse(nullptr) == h::mode::sweep, "an unset variable is `auto`: the sweep decides");
    expect(h::parse("") == h::mode::sweep, "and an empty one is too");
    expect(h::parse("auto") == h::mode::sweep && !h::pinned(h::mode::sweep),
           "auto names the same mode and stays unpinned, so a sweep may still change it");
    expect(h::parse("0") == h::mode::off && h::pinned(h::mode::off), "0 is an override and pins off");
    expect(h::parse("1") == h::mode::all && h::pinned(h::mode::all), "1 is an override and pins all layers");
    expect(h::parse("tail") == h::mode::tail && h::pinned(h::mode::tail), "tail is an override and pins 21");
    bool threw = false;
    try { h::parse("maybe"); } catch (const std::invalid_argument &) { threw = true; }
    expect(threw, "anything else refuses rather than silently becoming a mode");

    // The rule combo_producer.routed_prefill_choice applies to a paired request, now per sweep.
    expect(h::from_layer(h::mode::sweep, 1) == 0 && h::from_layer(h::mode::sweep, 2048) == 0,
           "a sweep of up to 2 048 new rows demand-reads every layer");
    expect(h::from_layer(h::mode::sweep, 2049) == h::tail_from_layer &&
           h::from_layer(h::mode::sweep, 16666) == h::tail_from_layer,
           "above it the early layers keep the overlapped whole-stage prefetch, where routed measured 843 ms against 600 ms");
    expect(h::from_layer(h::mode::all, 16666) == 0 && h::from_layer(h::mode::tail, 45) == h::tail_from_layer,
           "a pinned mode ignores the sweep's rows entirely");
    expect(h::from_layer(h::mode::off, 45) == h::layer_count && !h::routes(h::mode::off),
           "off routes nothing, and answers past the last layer so no layer demands a read");
    expect(h::routes(h::mode::sweep) && h::routes(h::mode::tail), "auto and tail both route something");

    // What DS4PERF_ROUTED_PREFILL prints. A log has to tell a chosen shape from a pinned one.
    expect(std::string(h::resolved_name(0)) == "all" && std::string(h::resolved_name(21)) == "tail" &&
           std::string(h::resolved_name(h::layer_count)) == "off",
           "the resolved shape is named by its first routed layer");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
