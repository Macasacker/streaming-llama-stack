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

Accept: the routed column of a prompt head measured at ≥ 1.3× its bytes/s (turn 5's
15.0 s → ≤ 11.5 s), decode median step down ≥ 1.5 ms and p90 down ≥ 6 ms, and the host plan
still `sum_matches=1`. If Step 0 says `transport=pageable`, run the same measurement with
`LimitMEMLOCK` raised (Step 4) before concluding anything about slot count.

Risk: the slots are `cudaHostAlloc`'d, so a bigger window is more pinned memory, charged
against the same `RLIMIT_MEMLOCK` that already refuses the carry bank.

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

## Probe fixes, so the next measurement is one command

- Run J's third request failed `HTTP 409: Another request owns this decoder`: the
  256-token request's producer still held the decoder. The probe must wait for the release,
  not assume the cooperative stop did it.
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
- decode steady median ≥ 24 tok/s — steps 1-2 are enough for that — and p90 ≤ 55 ms,
  which needs step 3, because the tail is the miss stream and not the setup,
- no admission refusal, no VRAM reserve breach, no paging growth,
- output compared by hand against the bias-0, no-draft reference.

Steps 1 and 2 together are ~80 s off the 303 s this session costs, and touch no numerics.
Step 3 is the next 42 s and depends on the tier filling. Step 5 is the biggest number and the
longest road; it is not the first thing to do.

Client-visible wall time is not in these logs: the router does work outside the producer's
timers. If the 30 s bar is met inside the producer and not on the client, the next instrument
is the router's own per-request accounting.
