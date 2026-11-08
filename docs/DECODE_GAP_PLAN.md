# Closing the decode gap: DS4.1-Flash single engine vs Strata + DeepSeek

Measured on this box (RTX 5090, 32 GiB VRAM, 93 GiB RAM, NVMe), 2026-11-08.

| | Strata + DeepSeek V4.1-Flash (IST-DAS 2.00 bpw, 129 GB) | This stack (IQ2_XXS-Q2_K imatrix, 365 GB) |
|---|---|---|
| decode | median **23.9 tok/s**, mean 24.2, p10 21.4, p90 27.2 (n=67 requests ≥200 tokens) | **1.04 tok/s** (16 tokens in 15419 ms = 964 ms/token) |
| prefill | ~1450 tok/s | ~2000 tok/s (64854 tokens in 33.2 s) |
| prompt prefix reuse | mean 81.6 % | unmeasured (`cache_n=0` on the first turn) |
| expert cache | `--expert-cache auto`, `--resident-experts`, learned `expert-profile`, hit rate 98.9-99.6 % on gyro traffic | `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=2` (engine default), **hit rate 0** |
| speculative decode | `--spec 4 --spec-min-p 0.5 --mtp mtp/rt` | none |

Baseline source: `~/workspace/local-llm-playground/qwen38-gyro/Strata/strata-deepseek-v4.1-flash-ist-das-2.00bpw.log`,
read with `~/workspace/local-llm-playground/qwen38-gyro/decode-rates.py`.

## Why decode costs what it costs

Routing is six expert ids per row per layer (`llama-dsv41-prefill-routes.h` rejects a row with any
other count). One expert slice for one layer measured 9.95 MB in the prefill fetch lines
(`[dsv41-prefill-experts] … active=229 bytes=2279301120`). A batch-1 decode step therefore moves

    6 experts x 40 layers x ~10 MB ≈ 2.4 GB

and the ids change every token. Prefill amortises the same per-layer fetch over thousands of rows
(2.28 GB covered 3414 rows ≈ 0.67 MB per row per layer), which is why prefill is fast and decode is
not. Measured fetch bandwidth on this disk is 2.6-3.2 GB/s (`bytes / fetch_ms`).

Target arithmetic: 24 tok/s is 42 ms per step.

| source of the 2.4 GB | rate | step | tok/s |
|---|---|---|---|
| NVMe, as today | ~2.8 GB/s | ~860 ms | ~1.2 |
| host RAM, no hit | ~25-40 GB/s | ~60-100 ms | 10-17 |
| VRAM | ~88 GB/s | ~27 ms | ~37 |

So the gap is a cache-placement problem, and the ceiling without speculative decoding is roughly
10-17 tok/s even with a perfect host cache. Strata clears 20 because its quant is ~0.85 GB per step
and several accepted draft tokens amortise each step.

## Step 1 - Measure the step budget before changing anything

Nothing below is worth doing until decode reports what a step actually costs. Tonight the producer
log has no per-token byte accounting: the ds4x decode path emits no `LLAMADECODE_STEP` line, and
prefill's `[dsv41-prefill-experts]` shows `new == active`, i.e. zero reuse.

- Add a per-step accounting line to the decode path: bytes fetched from file, bytes served from the
  host cache, bytes served from the VRAM expert cache, and step wall time. One line per token,
  gated on an existing trace variable so it costs nothing by default.
- Record the same for a 200-token decode run, so a median can be compared against Strata's 23.9.
- Capture `LLAMADECODE_HOST_PLAN`, `DS4PERF_GPU_JOB_ADMISSION` and `DS4X_OPEN` in the same run.

Accept: a run whose log answers "how many of these 2.4 GB came from disk" without inference.
Risk: none; instrumentation only, off unless the trace variable is set.

## Step 2 - Size the two cache tiers and buy their room

The expert cache is 2 GiB by engine default, which is smaller than one step's footprint.

- Raise `--ds4-cuda-streaming-expert-cache-reserve-gb` from 2 to 4-6 (VRAM tier) and give the host
  tier 16-24 GiB. Both need VRAM and RSS headroom the current plan does not have:
  device bill 27574 MiB against ~30.5 GiB free, `host_budget_gib` 31.5 against a measured
  `peak_rss` of 26.2 GiB.
- Fund them by trimming, in this order: `producer_graph_cap_gib` 5.5 → 3.5, context 262k → 128k
  for the decode-focused config, and re-check `--provider-host-bytes` / `--job-host-bytes` against
  the host plan. RAM available is ~75 GiB, so a host budget near 50 GiB is affordable; keep
  `runtime_min_free_gib` and the paging guard intact.
- Re-run the gate output in `~/bin/ds4-first-run` to confirm `admitted_device ≥ device bill + reserve`
  and `admitted_host ≥ combined host plan` before any request.

Accept: expert-cache hit rate ≥ 90 % on a repeated decode, and decode ≥ 4 tok/s with no admission
refusal and no `budget_breaches`. Record the hit rate and tok/s per setting so the trade is visible.
Risk: less context or less graph scratch changes what the engine can serve; keep the 262k config
separate from the decode-tuned one rather than degrading the delivered 262k plan.

## Step 3 - Stop the pinned path from going pageable

`CARRY_RAM_LOCK_PARTIAL bytes=… locked_bytes=8388608 rlimit_memlock_bytes=8388608 errno=12`: the
carry bank can only lock 8 MiB, so most of the resident carry traffic is pageable and the host cache
reads pay for it.

- Run the launcher under `LimitMEMLOCK=infinity` (systemd unit or the wrapper's service scope, not a
  shell `ulimit -l`, which is capped by `ulimit -Hl` = 8192 KB here).
- Confirm `CARRY_RAM_LOCK_PARTIAL` disappears from the producer log and re-measure the decode step.

Accept: no `CARRY_RAM_LOCK_PARTIAL` line, and a measured step-time delta attributed to it.
Risk: a memlock this large needs the host budget to be honest about what is wired; if the RAM guard
(`runtime_min_free_gib`, paging monitor) starts refusing, that is real information, not a regression
to silence.

## Step 4 - Cut bytes per step

Cache placement caps out near 10-17 tok/s. The remaining levers change the footprint itself.

- **Quant.** Strata's 2.00 bpw DeepSeek file is 129 GB against our 365 GB, i.e. ~0.85 GB per step
  against 2.4 GB. A smaller expert quant is the only lever that both raises the achievable hit rate
  and lowers the ceiling distance. The provider pins the accepted expert type set
  (`docs/FIRST_RUN.md` records which files on this box are refused and why), so this needs a
  quant the validator accepts, not an arbitrary one.
- **Route-tail skip.** Strata's `STRATA_ROUTE_TAIL_SKIP=7` buys +10-20 % decode on miss-heavy cards
  by skipping a missed expert ranked ≤7 in a verify window, at the cost of slightly different
  answers. Equivalent here would be a bounded skip on the decode routing path; treat it as a
  quality trade to measure, not a default.
- **Bias calibration.** The reasoning phase asks for a 0.35 expert-affinity bias and
  `ds4x_set_affinity` refuses any mean > 0 without a spread file bound to this exact
  model bytes/experts/layers. `presets/dsv41-affinity-spreads.txt` is calibrated for a
  315838021632-byte file, so this 365713686528-byte one is refused as wrong-model. Produce one with
  `DS4_CACHE_AFFINITY_CALIBRATE`, then set `cache_affinity_spread_file` (which
  `slstack/combo_producer.py` turns into `LLAMADECODE_DS4_SPREAD`). Until then the probe runs with
  `quality_speed: 0`.
- **Speculative decoding.** Not present in this path at all. It is the structural difference between
  24 tok/s and the ~10-17 tok/s host-cache ceiling, and it is the largest piece of work here; scope
  it as its own project rather than a tuning step.

Accept for step 4: a measured decode rate for each change in isolation, and the quality difference
where the change trades accuracy (tail skip, quant).

## Stop condition

If steps 1-3 land and decode still sits below ~4 tok/s, the conclusion is that this file cannot
serve interactive decode on a 32 GiB card, and the honest configuration is: this stack for long
262k prefill (where it already beats Strata at ~2000 vs ~1450 tok/s), gyro/Strata for chat decode.

## What tonight fixed, so the plan is measured on a working pipeline

- `wf_spawn_readers()`: reader threads are recreated on `wf_start`. `wf_stop` clears them and the
  prefill→decode layout switch is a stop/start, so decode ran with `readers=0` and 184 outstanding
  chunks that nothing could settle.
- `wf_pick_gpu_buf()` keeps stage `id` in bank `id % 2` (the invariant the provider's extents, expert
  cache and tail hints depend on), and the new `compute_recorded` flag stops a freshly created,
  never-recorded fence from looking permanently busy.
- `session_pipe::fill()` no longer reads a 10 ms poll timeout as a closed parent pipe;
  `tests/test_producer_session_wire.py` compiles the production class over real pipes and proves the
  silent-parent case and the real-closure case.
- Residual probes now take the element count from the tensor's own type and bytes
  (`pinned_storage::residual_probe()`); the previous F32-only probe read twice a BF16 extent and
  poisoned the CUDA context, which is what `DSV41_PINNED_COPY_CLEANUP_FAILED` was reporting.
- Weight upload copies per span offset instead of one whole-relay copy, so provider padding no longer
  shifts every tensor after the first non-128-multiple one (`DS4PERF_STAGE_DIFF_SUM … mismatching=0`
  across stages).
- `slstack release-holds` plus the first-run trap: an interrupted run left an unreleased cooperative
  GPU hold that no admission could ignore, and a producer wedged in `wait_layer_gpu` survives the
  cooperative stop holding ~30 GiB of VRAM.
