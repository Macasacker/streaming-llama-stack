# Cutting turn cost: the layout switch, the fetch window, and the tiers that don't fill

Goal: make an agentic turn on this box cheap enough to work in. The bar is wall clock per
turn, not tokens/s. Decode already reaches Strata's median here; what a user waits for is
setup and fetch, and both are measured below.

Measured on this box (RTX 5090 32 GiB, 93 GiB RAM, NVMe), 2026-11-09 to 2026-11-10.
Sources: `runtime/single-engine/ultra-73c876d05eb4e674` (pi's first real session, 6 turns,
one producer), `runtime/single-engine/ultra-c5205b797e321bb6` (bias 0.5, `--routed`,
255 steps), `runtime/single-engine/ultra-b5de641b30a32858` (bias 0, `--routed`, 255 steps),
`runtime/debug/run-20261109-*` (runs E, G, H, I).

## Where the time actually goes

Summed over the real session, from its own log lines:

| bucket | seconds | share of the 303 s a turn costs | of which fetch |
|---|---|---|---|
| prompt heads (6 turns) | 163.2 | 54 % | 86.2 s from the routed layers |
| **decode-layout rebuilds (5 turns)** | **56.5** | **19 %** | none — it is allocation |
| decode steps (952, excluding the rebuild) | 82.3 | 27 % | 59.5 s (72 % of that timer) |

Per turn, prompt head plus decode: 21.6-66.6 s, **median 53.1 s**, 303 s summed. The layout
rebuild sits inside the decode timer's first step, which is why 56.5 s of the 139.8 s reported
as `decode_ms` is not decoding.

Two corrections to what this stack's earlier plans assumed:

- **Decode is not the gap.** Run K's steady step is median 44.0 ms = 22.7 tok/s against
  Strata's 23.9 median. The `tok_s=10.56` the producer reports is arithmetic over
  `DECODE_COMPLETE`, which includes step 0 — the layout switch lands inside the first
  decoded step and adds 10.6 s to it. Steady mean is ~50 ms = 20 tok/s, p90 86 ms. The same
  session's per-turn numbers carry the same trap: 56.5 s of its `decode_ms` is the rebuild.
- **A decode step is its miss fetch.** Regressing `assign_ms` on `from_file` over run K's
  255 steps gives `assign_ms = -1.6 + 221.4 ms/GiB` with a 0.5 ms residual on a 25 ms
  median, i.e. the miss path runs at 4.52 GiB/s (bias 0: 3.34 GiB/s). `copy_sync_ms` is
  0.77 ms, so the device-to-device traffic is not what a step waits on.

## Step 0 — rebuild, because two measurements cannot be read without it

`build/llama/bin/llama-dsv41-prefill` and `libslstack-weight-fetcher.so` are from
2026-11-09 18:29. `DS4X_EXTENT_WINDOW` (added 2026-11-10) is not in them, so the question
"is the extent relay's host window pinned at all" — `ulimit -l` is 8192 KB against a
4 × 8 MiB window, and `wf_extent_window_init` falls back to `aligned_alloc` silently — is
still unanswered. `./build.sh` is CPU-only work.

Accept: a producer log containing `DS4X_EXTENT_WINDOW slots=4 … transport=pinned|mixed|pageable`.

**Answered.** The rebuilt binary logs `DS4X_EXTENT_WINDOW slots=4 slot_bytes=8388608 pinned=4
bytes=33554432 transport=pinned` on all three producers of run `20261110-184342`. `ulimit -l` is
still 8192 KB, so the window pins anyway: `cudaHostAlloc` is not charged against `RLIMIT_MEMLOCK`
on this kernel (chunk 3.1h, `HANDOFF.md`), only the carry tier's `mlock` is, which is what
`CARRY_RAM_LOCK_PARTIAL … errno=12` is refusing. Step 4 is therefore a carry-tier lever and not a
prerequisite for Step 2's window.

## Step 1 — stop paying the decode-layout rebuild every turn (−56 s per session)

`live-session.inc:320` rebuilds whenever `!provider->decode_layout()`, which is true after
every prefill sweep, and `layout_ms` is `provider->prepare_external_decode()`. Inside it,
`restart_prompt` (`llama-dsv41-weights.cpp:1215-1227`) does, on a layout flip:

    release_host(); release_l2(); wf_destroy(fetcher); buffers.clear(); create_pipeline();

so the two compute banks are freed and re-admitted, the fetcher and its pinned staging tier
are destroyed and rebuilt, and 924 tensors are re-registered. The resident pass it then runs
logs 1.5-1.6 s (`LLAMADECODE_EXTERNAL_RESIDENT ms=1543-1624`), so ~10 s is the teardown and
reallocation. Turn 2 shows the switch is avoidable: `layout_rebuilt=0 layout_ms=0.000`.

Work in this order:

1. **Split the line before changing anything.** `LLAMADECODE_LAYOUT_SWITCH decode=n
   switch_ms=…` already exists at `llama-dsv41-weights.cpp:1227` but covers the whole flip.
   Report `tiers_ms` (release_host/release_l2), `fetcher_ms` (destroy/create),
   `banks_ms` (free/alloc), `register_ms`. Nothing else in this plan is worth doing blind.
2. **Do not flip the layout when the turn only appends rows.** Turn 2 appended 22 rows and
   kept the decode layout; turns that appended 1 405+ flipped. Find the branch that decides
   it and widen it, or make the append sweep run in the decode layout.
3. **If a flip is required, keep what can be kept.** The banks and the fetcher are the same
   admitted geometry in both layouts — the comment at `llama-dsv41-weights.cpp:1216` says so
   and the code then destroys them anyway. Re-registering is already the cheap path
   (`reregister()` with `moved=924`, `register_ms=0.2-1.2`).

Accept: `layout_ms` ≤ 1.5 s on every turn after the first, measured on a multi-turn session;
session wall time down ≥ 40 s; no `DS4PERF_GPU_JOB_ADMISSION` refusal and no
`LLAMADECODE_VRAM_HEADROOM` dip below the reserve while both layouts exist.

**Item 1 answered** - the split is in `LLAMADECODE_LAYOUT_SWITCH` now, and it puts the whole flip
in one place. Run `20261110-194525` (report `runtime/debug/handoff-20261110-194525.md`), both of
its producers:

    LLAMADECODE_LAYOUT_SWITCH decode=1 switch_ms=9580.781 same_admission=1 tiers_ms=0.004
      destroy_ms=2665.004 detach_ms=0.016 build_ms=6915.458 quiesce_ms=0.298
      build_fetcher_ms=6915.083 build_banks_ms=0.005 build_register_ms=0.197 build_other_ms=0.174

So 9.6 s is **2.7 s of `wf_destroy` plus 6.9 s of `wf_create`**. The tier release, the staged-tensor
detach, the bank wrapping and the 82-layer registration add up to 0.2 ms each: item 2's "avoid the
flip" and item 3's "keep what can be kept" are the same lever, and the thing to keep is the
fetcher context.

`wf_create_impl` and `wf_destroy` are now split the same way (`WF_CREATE_PHASE` /
`WF_DESTROY_PHASE`), and that says which allocation the flip pays twice - run `20261110-200526`,
all three producers:

    WF_CREATE_PHASE  total_ms=6936.7 context_ms=0.0 files_ms=0.0 banks_ms=0.7
                     staging_ms=6935.5 stream_ms=0.0 pool_ms=0.0 readers_ms=0.5
                     gpu_bufs=2 ram_bufs=3 buf_bytes=7305363456 strict=1
    WF_DESTROY_PHASE total_ms=2699.2 stop_ms=0.0 extent_ms=11.6 files_ms=0.0
                     banks_ms=9.5 staging_ms=2678.1 rest_ms=0.0

**It is the pinned RAM staging, not the VRAM banks.** `cudaMallocHost` of three 7 305 363 456-byte
staging buffers (21.9 GiB pinned host) is 6.9 s, and `cudaFreeHost` of them is 2.7 s; the GPU
buffers cost 0.7 ms to make and 9.5 ms to free, because in direct mode the caller owns those heaps.
And `buf_bytes` is identical in the cold create and the flip's create: the staging tier the flip
rebuilds is the same geometry it just destroyed, which is what the comment above the rebuild
already claims and what the code then throws away anyway.

So Step 1 is: keep the staging tier across the flip. Keeping the whole fetcher is the clean way -
create it once at the decode layout's 82 layers and let the flip re-register layers (0.15 ms) and
set the per-layer persistent/skipped masks, which `wf_set_layer_persistent` and
`wf_set_layer_skipped` already do for the YOCO pass. Expected: `switch_ms` from 9.6 s to well
under the 1.5 s bar. Both creates are `strict=1` at the same `buf_bytes`, so nothing about the
admission changes; what has to be checked is that a layer the prefill layout does not register
never gets fetched, and that no admission refusal or VRAM dip appears while both layouts exist.

**Item 3 landed, and it is narrower than keeping the fetcher.** What the split pointed at was the
pinned tier, so that is the only thing carried: `wf_staging_take` moves a stopped context's
staging buffers out of it, `wf_params_adopt_staging` hands them to the next create (which reuses
them when the slot count and slot size match and logs `fresh_mismatch` when they do not), and
`invalidate()` frees a tier a failed create stranded. The fetcher context is still destroyed and
rebuilt, because its layer table genuinely differs between the two layouts - `n_layers` 42 vs 82 -
and that part costs 25 ms, not 7 s.

Measured against the pre-change control (`20261110-200526` vs `20261110-201948`, report
`runtime/debug/handoff-flip-200526-vs-201948.md`), three producers each:

    control   switch 29 205 = tiers 0 + destroy 8 385 + detach 0 + build 20 820 + quiesce 1
    candidate switch    125 = tiers 0 + destroy    63 + detach 0 + build     61 + quiesce 1

`WF_STAGING reused slots=3 bytes=7305363456 tier_slots=3 tier_bytes=7305363456` on all three, and
`staging_ms` is 0.0 in both the create and the destroy. Per flip: **9.7 s → 42 ms**. The
handoff's own `layout_ms` went 11.06-11.35 s → 1.34-1.42 s.

The clauses around it: no `DS4PERF_GPU_JOB_ADMISSION` refusal and no `PRODUCER_ABORT`,
`min_free_since_l2` 914 423 808 against the 128 MiB reserve, system pagefile +53 KB, and the short
request answers `Lisbon` in both runs. The routed column is unchanged - 284.44 GiB in 98.3 s →
97.8 s, x1.005 - which is the point: the flip was never a fetch.

One number that is **not** attributable: the short producer's step median reads 82.5 → 180.9 ms.
That producer's miss is 0.570 GiB/step against the control's 0.232 and its per-byte fit degenerates
(21 steps, one distinct x), and the same shape appeared as the first producer of the 19:45 run at
188 ms. It is the first producer of a window paying a cold file, not the flip.

What this changes downstream: the plan's `layout_ms ≤ 1.5 s` clause is met per flip at 1.34-1.42 s,
but that bar asks for a **multi-turn session** and the probe runs three single-turn producers -
that measurement is still outstanding. Item 2 (don't flip when a turn only appends rows) is now
worth ~40 ms per turn instead of ~10 s, so it drops behind Step 5 in the queue. Step 1's 56 s per
session is gone.

Two notes on reading the window. It ran only two producers: the long request failed
`HTTP 409: Another request owns this decoder` after the 256-token producer kept the decoder - the
probe fix listed below, now done, makes the probe wait for the release. And the control is not
stable across windows: the same default-geometry control measured 2.77 GiB/s at 19:30 and
3.21 GiB/s at 19:45, so absolute rates are only comparable inside one window.

Risk: VRAM. Holding both layouts means the banks of one are live while the other is built.
The plan already runs with `min_free_since_l2` ~1.2-1.7 GiB above a 128 MiB reserve, so this
has to be measured against the device bill (27 574 MiB) and not assumed. If both layouts
cannot coexist, the answer is item 2, not item 3.

## Step 2 — widen the extent relay's window (−24 s per session)

The routed fetch and the decode miss fetch both go through `wf_fetch_extents_direct`: one
ring sized `2 * extent_slots`, `extent_slots = 4` host slots of `extent_slot_bytes = 8 MiB`
— 32 MiB in flight — and a slot is reusable only after `slot_ready()` synchronises its
upload event. Measured on disk alone, same span geometry as a live stage:

| shape | in flight | cold |
|---|---|---|
| as built (1 ring, 4 × 8 MiB) | 32 MiB | 3.59-3.65 GiB/s |
| 1 ring, 8 × 8 MiB | 64 MiB | 4.82-5.19 |
| whole-stage relay shape (8 rings × QD4 × 4 MiB) | 128 MiB | 5.67 |

`extent_slots` and `extent_slot_bytes` are `constexpr` (`llama-weight-fetcher.cpp:494-495`).
Make them named settings the provider plans, account their host bytes in
`LLAMADECODE_HOST_PLAN`, and take them to 8 slots first.

**Landed and measured.** `LLAMADECODE_EXTENT_SLOTS` (1..64, unset 4) and
`LLAMADECODE_EXTENT_SLOT_MIB` (whole MiB, unset 8) are read once by the provider,
`LLAMADECODE_EXTENT_WINDOW_PLAN slots=8 slot_bytes=8388608 window_bytes=67108864 source=env`
prints what it resolved, the window's bytes replace the hardcoded `32*mib` in `planned.host_bytes`
(`llama-dsv41-weights.cpp:966`) so `LLAMADECODE_HOST_PLAN` charges exactly what the fetcher pins,
and `create_pipeline` passes the same geometry through `wf_params.extent_window_slots` /
`extent_window_slot_bytes`. Out-of-range values refuse at create rather than clamp — a clamped
window would charge one geometry and pin another. `docs/evidence/wf-extent-selftest.cpp` covers
twelve 1 MiB parts against an eight-slot window and both refusals (38 checks, no GPU).
`~/bin/ds4-first-run` forwards the two names into `env_overrides`, because the launcher drops
inherited environment.

Measured against the control (`20261110-184342`, default 4 × 8 MiB) with the candidate
(`20261110-185850`, 8 × 8 MiB, `DS4X_EXTENT_WINDOW slots=8 … pinned=8 transport=pinned` on every
producer). The three producers are the same three requests, and **the routed column read identical
bytes in both runs** — 91.84 / 92.78 / 99.82 GiB, same 40 / 40 / 355 layer-fetches, same 83.2 %
reuse on the long prompt — so this is a rate change, not a work change:

| producer | routed column | prompt-head fetch | client wall | step median/p90 (excl step 0) | decode relay, per byte |
|---|---|---|---|---|---|
| short 588 | 3.13 → **4.14 GiB/s** (1.32×) | 29.4 → 22.2 s | 55 → 46 s | 81.5 → 84.2 / 121.7 → 118.7 ms | 4.25 → 4.48 GiB/s |
| decode 256 @590 | 3.01 → **4.19 GiB/s** (1.39×) | 30.8 → 22.2 s | — | 47.9 → 53.6 / 97.6 → 100.2 ms | 3.83 → 3.75 GiB/s |
| long 64 854 | 3.01 → **4.24 GiB/s** (1.41×) | 33.2 → 23.5 s | 78.8 → 67.9 s | 76.8 → 77.4 / 136.9 → 121.7 ms | 3.86 → 4.48 GiB/s |
| **summed** | **3.05 → 4.19 GiB/s (1.37×)** | **93.4 → 67.9 s = −25.5 s** | | | |

The routed column clears its bar (≥ 1.3×; the plan predicted −24 s and the session's three prompt
heads gave up 25.5 s of fetch). The step column does not, and the reason is visible only after the
bytes are divided out: regressing `assign_ms` on `ds_bytes` over the 254 steps the two
256-token producers decoded gives **251 → 242 ms/GiB** (`ds_ms`: 3.98 → 4.13 GiB/s per byte), and
over the long-prompt producer **238 → 179 ms/GiB** (4.20 → 5.58 GiB/s). So the relay got faster
per byte — but that candidate decoded with more to fetch (0.102 → 0.121 GiB/step, 14.2 → 15.5
misses/step, tier hit rate 93.7 % → 93.2 %), and step median rose 5.8 ms because of the volume.
Generation content differs between runs, so bias-0.5 placement is never matched and step medians
cannot be compared directly at this resolution; the per-byte fit is the attributable number.

What that leaves: the median step is 53.6 ms of which `ds_ms` is 32.2 ms, i.e. the miss stream is
60 % of it and the window is no longer its limit. At 4.13 GiB/s the 0.121 GiB median miss costs
29 ms, so 24 tok/s (41.7 ms) needs ~35 MiB fewer bytes per step, not more slots. That is Step 3.

Accept, and the state of each clause: routed ≥ 1.3× its bytes/s — **met** (1.37× summed; the
plan's 15.0 s turn is the 22.2 s short-head equivalent). Host plan still `sum_matches=1` — **met**,
with the provider component moving by exactly +33 554 432 bytes in all three producers
(23 205 996 032 → 23 239 550 464, and 23 407 986 560 → 23 441 540 992) against 24.08 GiB of row-cache
headroom. No `DS4PERF_GPU_JOB_ADMISSION` refusal, `min_free_since_l2` 1.598 GiB against a 128 MiB
reserve, swap +7.7 MiB for the whole run. Decode median down ≥ 1.5 ms and p90 down ≥ 6 ms —
**not met as written**; see the paragraph above. Re-read it as per-byte, or re-measure it against
the bias-0, no-draft reference where placement is reproducible.

Risk: the slots are pinned host memory, so a bigger window is more RAM charged against the
job's host admission (8 slots is +32 MiB against 24 GiB of plan headroom, which is why the charge
is planned rather than discovered). It is **not** charged against `RLIMIT_MEMLOCK` — see Step 0's
answer. The remaining share of the gap is the upload gate, not the slot count: `slot_ready()`
still waits the previous slot's upload event before its next read, and the engine now reads
4.19 GiB/s where the standalone 32 MiB shape read 3.59-3.65 and the 64 MiB shape measured 4.82-5.19
— i.e. the widened window took the engine to roughly where its own shape stands, and the next
instrument on it is `fetch_ms` split into `read_ms` / `stage_ms` / `upload_ms`.

**8 slots is the ceiling, measured.** The window doubled once more (`20261110-191513` control at
the default 4 × 8 MiB, `20261110-191856` at `LLAMADECODE_EXTENT_SLOTS=16` — 128 MiB in flight,
`DS4X_EXTENT_WINDOW slots=16 … pinned=16 transport=pinned` on all three producers; report
`runtime/debug/handoff-20261110-191513.md`). Same three requests, and the routed column read the
same bytes in both runs — 91.84 / 92.78 / 99.82 GiB over 40 / 40 / 355 fetches — so again a rate
change, not a work change:

| producer | routed column | prompt-head fetch | client wall | step median/p90 (excl step 0) | decode relay, per byte |
|---|---|---|---|---|---|
| short 588 | 2.95 → **4.47 GiB/s** (1.52×) | 31.1 → 20.5 s | 57 → 45 s | 76.5 → 83.8 / 154.2 → 148.4 ms | 3.65 → 4.54 GiB/s |
| decode 256 @590 | 2.79 → **3.81 GiB/s** (1.37×) | 33.3 → 24.4 s | — | 53.3 → 56.0 / 117.7 → 118.9 ms | 3.32 → 3.52 GiB/s |
| long 64 854 | 2.79 → **3.40 GiB/s** (1.22×) | 35.8 → 29.4 s | 82.5 → 76.2 s | 83.0 → 80.0 / 154.1 → 131.0 ms | 3.67 → 4.41 GiB/s |
| **summed** | **2.84 → 3.83 GiB/s (1.35×)** | **100.2 → 74.3 s = −25.9 s** | | | |

The lift is what 8 slots already got (1.35× here, 1.38× for the 8-slot pair) and the fetch saving
is the same to the tenth of a second (−25.9 s against −25.5 s), so 64 MiB in flight is where the
window stops being the constraint: the doubling past it buys nothing while pinning 128 MiB of host
per producer instead of 64. It is not flat across the three — the long head gains 1.22× where the
short head gains 1.52× — which is the `slot_ready()` upload gate above rather than a slot shortage.
The host clauses hold again: `sum_matches=1` with the provider component at exactly +100 663 296
bytes on each producer, no `DS4PERF_GPU_JOB_ADMISSION` refusal, `min_free_since_l2` 1.643 GiB
against the 128 MiB reserve. Step median is again unreadable at bias 0.5 (+9.6 / +2.7 / −3.0 ms).
Set `LLAMADECODE_EXTENT_SLOTS=8` and leave it; the decode median is Step 3's number.

## Step 3 — make the RAM tier hold something (−42 s per session, conditional)

The tier is implemented and the decode step is fetch-bound, so DRAM is where the miss
stream should go. It does not currently work that way. Run E (tier funded at 12.9 GiB,
1 288 slots):

    LLAMADECODE_HOST_WORKING_SET observed_passes=32 selected=900 bounded_admission=1
    host_hits=883 of misses=14055   (6.3 %)   stored 0.023 GiB/step   from_ram 0.032 GiB/step
    device copies +1.88 GiB/step against a run with no tier

The allowlist admitted 900 keys, ~634 entries were actually stored into 1 288 slots, and
6.3 % of misses were absorbed — which is why Step 2 of the decode plan recorded the tier as
neutral. That verdict is about this admission, not about DRAM. `store_host` returns early for
any key not in the allowlist built from 32 observed passes
(`llama-dsv41-weights.cpp:241-244`), and returns early if the key is already stored.

Work: make the RAM tier an LRU the eviction stream can actually fill, sized against
`host_budget_gib` and the paging guard, and measure `host_hits / misses` as the acceptance
number rather than `from_ram` share.

Accept: `host_hits` ≥ 50 % of `misses` on a ≥ 200-step decode at bias 0.5, decode median
≤ 30 ms, and no `DSV41_HOST_MEMORY_PAGING_STOP`. If the guard fires, the fix is the host
budget or `vm.swappiness` — never `paging_limit`.

**Measured, and the bar is above the tier's ceiling** (`20261110-193000` control with the tier
off, `20261110-193454` funded at `LLAMADECODE_HOST_CACHE_MB=12288` with continuous admission,
report `runtime/debug/handoff-20261110-193000.md`). `LLAMADECODE_HOST_CACHE slots=1288
bytes=12879544320 … available_after=58054209536` — the tier funds, and it is 1 288 slots against
an expert space of `40 × 384 = 15 360` keys, so a *perfect* admission absorbs 8.4 % of a miss
stream that draws from the whole space. The bar asks for 50 %.

What the new admission (`LLAMADECODE_HOST_CACHE_ADMIT_TOUCH=2`, which stores a victim on its
second eviction instead of after the 32-pass snapshot) measured:

| producer | step median/p90 (excl step 0) | decode relay, per byte | host_hits of misses |
|---|---|---|---|
| short 588 | 113.7 → 113.2 ms | 3.38 → 3.41 GiB/s | 0 of 892 |
| decode 256 @590 | 57.7 → **69.5 ms** | 3.13 → **2.54 GiB/s** | 115 of 4 131 (0.028) |
| long 64 854 | 76.8 → **86.8 ms** | 3.88 → 3.04 GiB/s | 0 of 1 699 |
| **summed** | | | **115 of 6 722 (0.017)** |

Two things that table says. The writeback is on the step: the two producers that decoded enough
to measure both lost ~10-12 ms of median step, and the decode producer's per-byte relay rate fell
from 3.13 to 2.54 GiB/s, so the tier is a net cost at this size — 290 entries stored, 115 of them
asked for. And the second-touch filter is not what limits it: the misses are first touches of
experts that are never evicted twice, so no admission policy reaches 50 % at 1 288 slots.

The paging guard fired on the first funded producer and stopped it after 0 decoded steps:
`DSV41_HOST_MEMORY_PAGING_STOP pagefile_in_use_bytes=1315008512 baseline_bytes=1016692736
growth_bytes=298315776 limit_bytes=256000000`. MemAvailable was still 58 GiB, so this is Step 4,
not Step 3: the box pages anonymous memory before it reclaims the engine's page cache, and 12.9
GiB of tier is enough to trip a 256 MB machine-wide growth limit. Step 3 is blocked on
`vm.swappiness` (root) and is not worth more windows until that is done.

Left in place: `LLAMADECODE_HOST_CACHE_ADMIT_TOUCH` (unset = the shipped snapshot, so the control
is unchanged) and the probe's forwarding of the three RAM-tier names. `scripts/ds4_run_metrics.py`
reports `ram hits/miss` per producer and sums the tier per run, which is the number this clause
turns on. The next instrument, if Step 3 is reopened, is a reuse counter on the miss path —
distinct keys missed against total misses, and their distance against the slot count — because
that is the only measurement that can say whether *any* tier size gets to 50 %.

Risk: pinned/anonymous host memory is what pushed this desktop into swap during Step 2 of
the decode plan. Fund it from `producer_job_host_gib`, watch `available_after`, and keep the
`runtime_min_free_gib` and paging guards honest.

## Step 4 — owner actions outside the repo

- `LimitMEMLOCK` raised for the launcher's scope (root). Today `rlimit_memlock_bytes=8388608`
  and both runs log `CARRY_RAM_LOCK_PARTIAL … errno=12` with 2.5 GiB of carry unpinable.
- `vm.swappiness` near 0 (root). The engine leans on reclaimable page cache from a 340 GiB
  file; the box swaps anonymous memory first.
- Note for whoever runs them: the paging guard reads machine-wide `SwapTotal - SwapFree`,
  not the producer's own swap.

## Step 5 — quant, the ceiling (−90 s per session, the expensive one)

A 2.00 bpw pack is ~0.38× our expert bytes, and the fetch is 72 % of decode and 53 % of the
prompt head, so it shrinks both. It is also the only lever that makes the whole routed set
(`40 × 384` expert triples = 142 GiB today) small enough to pin. It needs a quant the
provider's validator accepts, and it unlocks the in-place two-tier read
(`external_expert_slots_l2` refuses unless the experts are `GGML_TYPE_MXFP4`,
`llama-dsv41-weights.cpp:1574`), which is what makes pinned VRAM readable instead of
copy-fed.

This is a conversion plus numerical qualification against the no-bias, no-draft reference.
Do it after steps 1-3, which are cheap and measured, not instead of them.

## Step 6 — what an append costs when it fetches nothing

Turn 2 appended 22 rows over 40 layers, logged no `[dsv41-prefill-experts]` line at all
(no line with `rows<=32` exists in the session), and still took 6.0 s — ~150 ms per layer
with zero expert bytes read. `--routed` let the reuse be measured at all: on the long prompt
it reports 64 131 active and 10 768 newly read over 355 layer-tiles (**83.2 % reused**), and
the rate stayed 836.9 tok/s. Under `tail` the early layers log nothing, so no earlier run
could say whether they reused — this is the first measurement of it, not proof that routing
changed it. Reuse is therefore not the bottleneck, and the path needs per-layer append time
split from fetch time before any byte-side work on the early layers is proposed.

## The measurement window, so the card handoff is one command

`scripts/gpu_handoff.py` (also `~/bin/gpu-handoff`) takes the card from Strata and gives it back:
it reads the running `serve.server`'s argv, environment and working directory from `/proc`, sends
it SIGTERM (the clean path its own `lifeline.install_stop_handlers` takes), waits for the device to
fall under the probe's idle gate, runs one `ds4-first-run --bias --routed` per `--probe NAME=VALUE`
pair, writes the comparison with `scripts/ds4_run_metrics.py`, and restores Strata with the command
it was started with - not the command a launcher script would have chosen. It restores even when a
probe fails, refuses before stopping anything if a process that is not Strata holds the card, and
holds a lock so two handoffs cannot overlap.

    ~/bin/gpu-handoff --probe '' --probe 'LLAMADECODE_EXTENT_SLOTS=8'   # matched pair, ~8 min

`scripts/ds4_run_metrics.py` is the part that can be read on its own, and it encodes the two traps
this plan has already been caught by: step 0 is excluded from every step figure, and the decode
step is compared per byte (`assign_ms` regressed on `ds_bytes`) as well as in absolute milliseconds,
because a bias-0.5 run never places the same experts twice.

## Probe fixes, so the next measurement is one command

- Run J's third request failed `HTTP 409: Another request owns this decoder`: the
  256-token request's producer still held the decoder. The probe must wait for the release,
  not assume the cooperative stop did it. **Done** - `~/bin/ds4-first-run` now posts through a
  `post()` helper that re-sends while the body says `owns this decoder` (six waits of 15 s, then
  it reports that the decoder never released), and the long prefill's python path retries the
  same way instead of printing the 503 and moving on.
- A producer that outlives its run also poisons the *next* run's evidence. Measured 2026-11-10
  19:33: the producer that aborted on the paging gate served the following probe's short request,
  so its directory appeared in two stamps and the report showed a snapshot-policy producer under a
  `ADMIT_TOUCH=2` run. `scripts/gpu_handoff.py` now waits for the probe stack (`llama-dsv41-prefill
  | slstack | ds4-server`) to be gone before starting a probe after the first, and skips the probe
  if it isn't; `scripts/ds4_run_metrics.py` assigns each producer directory to the stamp that
  created it, by the creation time of its oldest file, rather than to every stamp that mentions it.
- `~/bin/ds4-first-run` should print `layout_ms`/`layout_rebuilt` per request and report
  decode rates excluding step 0, or every future run will repeat the 10.56-vs-20 tok/s
  confusion.
- Cold read measurements here use blocks nothing has read. Do **not** `POSIX_FADV_DONTNEED`
  while an engine is serving: one such run left Strata at 7.6 GiB swapped with `SwapFree` at
  19 MiB.

## Acceptance for the plan

On the real session's shape (6 turns, growing context, one producer):

- median turn wall ≤ 30 s (measured today: 21.6-66.6 s, median 53.1 s),
- `layout_ms` ≤ 1.5 s on every turn after the first,
- decode steady median ≥ 24 tok/s — **not** what steps 1-2 give. After Step 2 landed, the
  254-step producer's steady median is 53.6 ms = 18.7 tok/s (control 47.9 ms = 20.9), and its miss
  stream is 60 % of the step at 4.13 GiB/s per byte. 24 tok/s needs fewer bytes per step, which is
  Step 3 — and p90 ≤ 55 ms, which needs the same, because the tail is the miss stream and not the
  setup,
- no admission refusal, no VRAM reserve breach, no paging growth,
- output compared by hand against the bias-0, no-draft reference.

Steps 1 and 2 together are ~80 s off the 303 s this session costs, and touch no numerics.
Step 2's share is measured: 25.5 s of prompt-head fetch off the probe's three heads (93.4 → 67.9 s
for the same 284.44 GiB), with no numerics touched, and 8 slots is its ceiling. **Step 1 has
landed**: the flip is 9.7 s → 42 ms per rebuild by carrying the pinned staging tier across it, so
the session's 56 s of layout rebuilds are gone and what remains of Step 1 is item 2, worth ~40 ms
per turn. Step 3 is **not** that 42 s: funded at 12.9 GiB it absorbed 1.7 % of the probe's misses
and cost 10-12 ms of step median to do it, and 1 288 slots cannot hold 50 % of a stream that draws
from 15 360 keys. It is blocked on Step 4's `vm.swappiness` before it is worth reopening. Step 5
is the biggest number and the
longest road; it is not the first thing to do.

Client-visible wall time is not in these logs: the router does work outside the producer's
timers. If the 30 s bar is met inside the producer and not on the client, the next instrument
is the router's own per-request accounting.
