// The exact radix top-k dispatch count, as one rule for both sides of the coverage check.
//
// The V4.1 executor refuses a layer when the radix dispatch counter disagrees with what it
// expected ("Actual exact radix top-k first/merge dispatch coverage differs"). With the branch
// written twice, single-row decode was moved onto the deterministic tiles network while the
// executor's scan still dropped rows<=1, so every decode step over 8192 index columns dispatched
// two kernels the check counted as zero and aborted the producer (pi's session of 2026-11-11,
// runtime/single-engine/ultra-ecac43f59d753fb5: rows=1, keys=48645).
//
// No CUDA and no ggml: the flags the backend would read are passed in, so this runs on any box.
#include "ggml-dsv41-topk-plan.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

static int failures = 0;
static void expect(bool ok, const char *what, const char *detail = "") {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail[0] ? " " : "", detail);
    if (!ok) ++failures;
}

// What an RTX 5090 (sm_120) reports. The limits decide the branch, so the table is read against
// a real device's numbers rather than against an abstraction.
static const size_t SMPB  = 99u * 1024;
static const size_t SMPBO = 99u * 1024;

static int64_t next_power_of_2(int64_t v) { int64_t n = 1; while (n < v) n *= 2; return n; }

static bool bitonic_fits(int64_t columns) {
    return next_power_of_2(columns) * sizeof(int) <= std::min<size_t>(SMPB, 48 * 1024);
}
static bool radix_on(int k) {
    return k == 512 && 8192u * sizeof(int) + sizeof(unsigned) <= SMPBO;
}
// The row rule the backend uses: with LLAMA_DSV4_TOPK_CUB_1ROW unset, a single row is
// deterministic too, which is the change that made the executor's rows<=1 skip wrong.
static bool cub_single_row_control() {
    const char * env = std::getenv("LLAMA_DSV4_TOPK_CUB_1ROW");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}
static bool deterministic(int64_t rows) { return rows > 1 || !cub_single_row_control(); }

struct shape { int64_t columns; int k; };

int main() {
    const char * cub_control = std::getenv("LLAMA_DSV4_TOPK_CUB_1ROW");
    std::printf("mode=%s smpb=%zu smpbo=%zu\n",
                cub_control && cub_control[0] == '1' ? "CUB_1ROW control" : "deterministic (default)",
                SMPB, SMPBO);

    // The grid tests/dsv41_topk_single_row.cpp runs, over the prefill batch (3 rows).
    struct want { shape s; uint64_t rows3; uint64_t rows1; };
    const want cases[] = {
        {{272,   272},  0, 0},   // whole row fits the single argsort kernel
        {{384,     6},  0, 0},   // the MoE gate: 384 experts, 6 selected
        {{8192,  512},  0, 0},   // exactly one tile still fits the single kernel
        {{16384, 512},  2, 2},   // tiles: first pass plus one merge, whatever the row count
        {{48645, 512},  2, 2},   // the shape that aborted the session: 6 tiles, 3072 candidates
        {{65536, 512},  2, 2},   // 8 tiles
        {{131072,2048}, 0, 0},   // k above the radix kernel's 512: the tile sort is not radix
        {{1000000,512}, 3, 3},   // two merge rounds before the final one
    };
    for (const auto & c : cases) {
        char what[160];
        const uint64_t three = dsv41_topk_radix_dispatches(c.s.columns, c.s.k, false, deterministic(3),
                                                           bitonic_fits(c.s.columns), radix_on(c.s.k), true);
        const uint64_t one = dsv41_topk_radix_dispatches(c.s.columns, c.s.k, false, deterministic(1),
                                                         bitonic_fits(c.s.columns), radix_on(c.s.k), true);
        // Under the CUB_1ROW control a single row genuinely takes DeviceTopK, so 0 is right there.
        const uint64_t want_one = cub_single_row_control() ? 0 : c.rows1;
        std::snprintf(what, sizeof what, "columns=%lld k=%d prefill", (long long)c.s.columns, c.s.k);
        expect(three == c.rows3, what, "");
        std::snprintf(what, sizeof what, "columns=%lld k=%d single-row decode", (long long)c.s.columns, c.s.k);
        expect(one == want_one, what, "");
    }

    // A single row over 8192 columns is the regression: the deleted scan answered 0 here.
    expect(dsv41_topk_radix_dispatches(48645, 512, false, deterministic(1), bitonic_fits(48645), radix_on(512), true)
           == (cub_single_row_control() ? 0 : 2),
           "the decode step that aborted answers what the backend actually dispatches");

    // The bounded streaming selector is one pass and no merge, whatever the width.
    expect(dsv41_topk_radix_dispatches(5000, 512, true, deterministic(1), false, radix_on(512), true) == 1,
           "a tagged selector answers 1");
    expect(dsv41_topk_radix_dispatches(5000, 272, true, deterministic(1), false, radix_on(272), true) == 0,
           "a tagged selector at k=272 answers 0");

    // The branch the count must fall to, not guess.
    expect(dsv41_topk_radix_dispatches(65536, 512, false, false, false, radix_on(512), true) == 0,
           "DeviceTopK (the 1-row CUB control) dispatches no radix kernel");
    expect(dsv41_topk_radix_dispatches(65536, 512, false, deterministic(3), false, false, true) == 0,
           "LLAMA_DSV41_TOPK_RADIX unset dispatches no radix kernel");
    expect(dsv41_topk_radix_dispatches(65536, 512, false, deterministic(3), false, radix_on(512), false) == 0,
           "without CUB compiled in nothing is radix");

    // Merge rounds grow with the width, exactly as the backend's loop.
    uint64_t steps = 2; int64_t width = ((1000000 + 8191) / 8192) * 512;
    while (width > 8192) { width = ((width + 8191) / 8192) * 512; ++steps; }
    expect(dsv41_topk_radix_dispatches(1000000, 512, false, deterministic(3), bitonic_fits(1000000), radix_on(512), true) == steps,
           "the merge rounds follow the dispatch loop");

    if (failures) { std::printf("FAIL %d check(s)\n", failures); return 1; }
    std::printf("PASS all top-k radix coverage checks\n");
    return 0;
}
