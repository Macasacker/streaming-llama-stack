# Prefill and decode: the next measured levers

Measured 2026-11-10 on the live session `runtime/single-engine/ultra-134bbe7538e2a44c`
(this session: 262k preset, one producer, context 16k-89k, 22 prompt heads, 27 decode
turns). Sources: the producer log's `[dsv41-perf]`, `DS4PERF_LAYER_WALL_LEDGER`,
`DS4PERF_PROMPT_HEAD_COMPLETE`, `LLAMADECODE_DS4X_TOKEN`, `LLAMADECODE_DECODE_COMPLETE`.

The plan is ordered by measured gain per effort. Steps 1 and 2 touch no numerics. Step 3
changes bytes per step. Steps 4 and 5 are the structural ones.

## What the numbers say

### Prefill

| turn shape | prompt head | rate |
|---|---|---|
| 16,666 new tokens | 29.2 s | 570 tok/s |
| 16,187 new tokens | 31.0 s | 522 tok/s |
| 5,652 new tokens | 29.9 s | 189 tok/s |
| 45 new tokens (append) | 19.1 s | 424 ms/token |
| 85 new tokens (append) | 19.9 s | 234 ms/token |
| 80 new tokens (append) | 19.2 s | 240 ms/token |
| 2,799 new tokens (append) | 27.2 s | 103 tok/s |

Per layer, from `[dsv41-perf]` (27 passes x 40 layers):

| layer group | acquire | upload | compute | prepare |
|---|---|---|---|---|
| 0-20 (whole-stage prefetch) | 600-654 ms | 0-3 ms | 5-200 ms | 0-3 ms |
| 21-39 (routed demand read) | 0 ms | 6 ms | 244-855 ms | 0-1 ms |

An append turn therefore pays **21 layers x ~630 ms = 13.2 s of whole-stage prefetch** and
19 layers x ~244 ms = 4.6 s of routed compute. The prefetch reads each layer's full expert
set (~2.6 GiB) for 5 new experts; the routed layers fetch 11 ms for the same 5 experts.

The reason is the producer, not the model: `routed_prefill: "tail"` in `producer.json` was
chosen when the producer was created for the first turn (16,666 tokens), and the producer
survives every later turn. `routed_prefill_choice` (`slstack/combo_producer.py:407`) runs
once per producer, not once per sweep, so every append turn keeps `tail` and layers 0-20
stay on whole-stage prefetch.

### Decode

11,225 steps, median 58.7 ms, p10 32.0, p90 130.6, mean 77.0. Per turn 6.4-15.4 tok/s.

| component | median | p10 | p90 |
|---|---|---|---|
| step (`LLAMADECODE_STEP`) | 58.7 ms | 32.0 | 130.6 |
| `decode_ms` (`DS4X_TOKEN`) | 59.5 ms | 31.3 | 136.9 |
| `ds_ms` / `assign_ms` (fetch + upload) | 34.5 ms | 7.4 | 111.8 |
| compute (`decode_ms - ds_ms`) | **24.7 ms** | 23.0 | 26.5 |
| `ds_bytes` | 0.102 GiB | 0.019 | 0.334 |
| misses | 11 | 2 | 36 |

Regressing `assign_ms` on `ds_bytes` over all steps gives **304.8 ms/GiB = 3.28 GiB/s** with
an 8.4 ms intercept. The compute floor is constant across the 16k-89k context range, so it is
not attention scaling; it is per-step work.

24 tok/s is 41.7 ms. Today's 58.7 ms is 24.7 compute + 8.4 overhead + 31 ms fetch. The three
levers are therefore: fewer bytes per step, a faster miss fetch, or fewer steps per token.

## Step 0 - instrument before changing anything (no card)

- Decode: split `assign_ms` into `read_ms` / `stage_ms` / `upload_ms`, and split the 24.7 ms
  compute into attention / FFN / HC. Gate on `LLAMADECODE_BYTE_LEDGER=1` so the default costs
  nothing.
- Prefill: `[dsv41-perf]` already carries upload/compute; add the routed fetch's read/upload
  split inside `prefill_experts`, and log the resolved routed choice per sweep
  (`routed_prefill=... from_layer=... rows=...`).
- The existing `DS4X_EXTENT_WINDOW` line already says the window geometry and transport.

Accept: one line per step and per layer with the split, and no behaviour change. This is the
only step that can run while the card is busy.

## Step 1 - Prefill append: choose routed per sweep (the cheap win)

Root cause is above: `LLAMADECODE_PREFILL_ROUTED` is read once in `create_pipeline`
(`engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-weights.cpp:1191`) and sets
`lazy_from_layer`; the producer is reused across turns.

Fix: make the choice per sweep. Add a provider setter (or a per-sweep flag) that sets
`lazy_from_layer` from the sweep's new-row count, using the same rule as
`routed_prefill_choice`: all routed at <= 2,048 rows, `tail` above it. Keep the env as an
override.

Expected: early layers 600 ms -> ~244 ms, so an append turn 18.6-22.5 s -> ~11-15 s. That is
~8 s/turn, ~140 s over a 20-turn session.

Accept: append prompt head <= 12 s at 69k context; full prompt unchanged; no numerics change
(the routed path is already qualified); no admission refusal.

Risk: routed early layers on a full prompt cost more (843 ms against 600 ms), so the >2,048
branch must stay whole-stage. Measure both branches in one window.

## Step 2 - Prefill full prompt: overlap the routed fetch and raise its rate

The late layers' 750-855 ms includes the routed fetch's 643 ms, serialised (fetch_ms is
91-101 % of its own span). The early layers' 600 ms acquire is whole-stage prefetch at
~4.3 GiB/s.

- Split read/stage/upload first (Step 0), then overlap the routed fetch with compute on the
  late layers instead of serialising it inside the executor.
- The extent relay already runs 8 x 8 MiB pinned; the standalone shape reads 4.82-5.19 GiB/s
  cold against the engine's 4.19, so the upload gate is the remaining share.

Expected: 16k prompt 29.2 s -> ~24 s; routed fetch >= 5 GiB/s.

Accept: 16k prompt <= 24 s, routed fetch >= 5 GiB/s, no numerics change, no VRAM reserve
breach while both banks exist.

## Step 3 - Decode: fetch rate, bytes per step, compute floor

3a. Split `assign_ms` and target 5 GiB/s. At today's 0.102 GiB/step that is 24.7 + 8.4 + 18 =
51 ms, 19.7 tok/s.

3b. Fewer bytes per step. Route-tail skip (5 of 6 experts, -17 %) is a routing-contract change
and a quality trade. The quant (2.00 bpw, ~0.38x expert bytes) is the big one: at 0.039 GiB/step
and 5 GiB/s the step is 24.7 + 8.4 + 8 = 41 ms, 24.6 tok/s. It needs a conversion and a
numerical qualification against the no-bias, no-draft reference.

3c. The 24.7 ms compute floor. Instrument attention / FFN / HC before proposing anything; if
attention is the floor at 69k context, that is the next lever, not the fetch.

Accept: median step <= 45 ms (22 tok/s) after 3a+3b; <= 41.7 ms (24 tok/s) with quant or a
verified draft.

## Step 4 - Drafter: tokens per step

The DSpark sidecar is repacked and accepted by `ds4 --inspect`
(`ds4-models/dspark-DeepSeek-V4.1-Flash-1017-mtp-q8.gguf`); the native path has never been
measured. Run `scripts/ds4_dspark_probe.sh` with `--dspark --mtp-model` and
`DS4_ALLOW_MTP_STREAMING=1`, and read `DSpark V4.1 cycle drafted= accepted=`.

If acceptance >= 0.5, extend `ds4x` with a verify-shaped entry. The delta and owner-digest
contracts gate it, not a throughput number.

Accept: measured acceptance rate and tok/s with and without the draft, plus a quality
comparison at the same sampling settings.

Risk: state correctness. A draft that advances state the checkpoint cannot describe is worse
than slow.

## Step 5 - Retention across turns

This session's routed prefill reuse is 47.2 % (140,794 active, 74,344 newly read). An append
turn still re-reads stages the previous turn fetched. Reads are buffered and the page cache is
where to test retention; do not drop it while an engine is serving.

Measure distinct-key misses against total misses and their distance against the slot count
before sizing anything.

Accept: an append turn re-reads <= 1 stage; no paging growth.

## Measurement protocol

- One probe per window; pair stamps afterwards with `scripts/ds4_run_metrics.py --stamp A
  --stamp B`.
- Step 1: `~/bin/gpu-handoff --probe '' --probe 'LLAMADECODE_PREFILL_ROUTED=1'`, then read
  `[dsv41-perf]` early/late and `DS4PERF_PROMPT_HEAD_COMPLETE`.
- Decode: `scripts/analyze_dsv41_decode.py`, and read `assign_ms` per byte and
  `min_free_since_l2` before believing any rate.
- Do not run while this session is serving: the stack holds the card.

## Acceptance for the plan

- append turn prompt head <= 12 s at 69k context (measured 18.6-22.5 s),
- full prompt 16k <= 24 s (measured 29.2 s),
- decode median step <= 45 ms (22 tok/s) after 3a+3b; <= 41.7 ms (24 tok/s) with quant or a
  verified draft,
- no admission refusal, no VRAM reserve breach, no paging growth,
- output compared by hand against the no-bias, no-draft reference.
