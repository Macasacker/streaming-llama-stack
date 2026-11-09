# Closing the decode gap: DS4.1-Flash single engine vs Strata + DeepSeek

This engine exists to serve DeepSeek V4.1-Flash, on this hardware, for agentic software
engineering. That workload is decode-dominated: long generations, tool-call turns, and many turns
against one growing context. Decode throughput and turn reuse are therefore the product, and
prefill speed only decides how quickly a turn starts. 1.04 tok/s is not a usable answer for that
job, so the target of this plan is the Strata + DeepSeek number, not a re-scoped role for the
stack.

Measured on this box (RTX 5090, 32 GiB VRAM, 93 GiB RAM, NVMe), 2026-11-08.

| | Strata + DeepSeek V4.1-Flash (IST-DAS 2.00 bpw, 129 GB) | This stack (IQ2_XXS-Q2_K imatrix, 365 GB) |
|---|---|---|
| decode | median **23.9 tok/s**, mean 24.2, p10 21.4, p90 27.2 (n=67 requests ≥200 tokens) | **1.04 tok/s** mean (16 tokens in 15419 ms = 964 ms/token); median step 292 ms = 3.4 tok/s, first step 11.4 s (Step 1) |
| prefill | ~1450 tok/s | ~2000 tok/s (64854 tokens in 33.2 s) |
| prompt prefix reuse | mean 81.6 % | unmeasured (`cache_n=0` on the first turn) |
| context served | `--max-context 131072` (prompts in the log run 51-69 k tokens) | 262k |
| experts per word | `--experts 5` - five of the model's six, Strata's "Faster" setting; `CPU lane 5.7-9.8 % of the experts` on the same lines | all six, no CPU lane |
| expert cache | `--expert-cache auto`, `--resident-experts`, learned `expert-profile`. **This log records no hit-rate line** - the 98.9-99.6 % figures are gyro traffic, not this run | VRAM tiers 449 L1 + 1636 L2 slots; `host_slots=0` in every step, so no RAM tier |
| drafting | `--dspark on --dspark-conf 0.65`: DSpark, a block-5 drafter with its own 128-expert MoE (3 per token, target layers 37-39, per the pack's `ds4.txt`), loaded from a separate `dspark-…-Q8_0.gguf`. **No accepted-draft count is recorded in this log** | none |

The `--spec 4 --spec-min-p 0.5 --mtp mtp/rt` line is the gyro-m (Qwen3.8) serve, not this baseline;
the DeepSeek run carries no `--spec` and no `--mtp` flag. Its drafter is DSpark.

Baseline source: `~/workspace/local-llm-playground/qwen38-gyro/Strata/strata-deepseek-v4.1-flash-ist-das-2.00bpw.log`,
read with `~/workspace/local-llm-playground/qwen38-gyro/decode-rates.py`. The engine line it records
is `strata-dsv41 --serve --pack …/deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1
--vision on --dspark-conf 0.65 --experts 5 --dspark on`.

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

So the gap is a cache-placement problem, and the ceiling without tokens-per-step is roughly
10-17 tok/s even with a perfect host cache. Strata clears 20 with a smaller step and a drafter: its
quant is ~0.85 GB per step against our 2.22 GB measured, it routes five experts per word rather
than six (≈0.71 GB), and 5.7-9.8 % of the experts are computed on the CPU lane from RAM instead of
being uploaded. At 24 tok/s a 0.85 GB step is ~20 GB/s, which is RAM/VRAM bandwidth, not NVMe. How
much of that 24 comes from DSpark is not readable from this log - it records no accepted-draft
count - so the drafter's share is unknown, not assumed.

## Step 1 - Measure the step budget before changing anything

Nothing below is worth doing until decode reports what a step actually costs. Before this change the
producer logged a step's wall time and its file bytes (`LLAMADECODE_STEP`, `LLAMADECODE_DS4X_TOKEN`),
but not which tier served the rest: hit counts without bytes, so the RAM tier could not be told apart
from a VRAM L2 hit. Prefill's `[dsv41-prefill-experts]` shows `new == active`, i.e. zero reuse.

- Add a per-step accounting line to the decode path: bytes fetched from file, bytes served from the
  host cache, bytes served from the VRAM expert cache, and step wall time. One line per token,
  gated on an existing trace variable so it costs nothing by default.
- Record the same for a 200-token decode run, so a median can be compared against Strata's 23.9.
- Capture `LLAMADECODE_HOST_PLAN`, `DS4PERF_GPU_JOB_ADMISSION` and `DS4X_OPEN` in the same run.

Accept: a run whose log answers "how many of these 2.4 GB came from disk" without inference.
Risk: none; instrumentation only, off unless the trace variable is set.

### Done (2026-11-09)

- `LLAMADECODE_STEP_BYTES` is one line per decoded token: `experts`, `slot_bytes`, `footprint`, and
  the bytes each tier served (`from_file`, `from_ram`, `from_l2`, `from_l1`), plus device copies,
  RAM-tier writes, per-tier hit counts and `step_ms`. It is built by
  `src/llama-dsv41-step-ledger.h` and emitted from `ds4_external::step()` in
  `tools/dsv41-prefill/live-session.inc`, gated on the existing `LLAMADECODE_BYTE_LEDGER=1`. The
  counters (`assign_expert_slots` in `src/llama-dsv41-weights.cpp`, exposed through `cache_stats()`)
  are increments, so the switch costs a line and nothing else. A step whose tiers do not add up to
  its footprint says so in `unattributed=` / `over_served=`.
- `tests/dsv41_step_ledger.cpp` proves the arithmetic and the line format:
  `c++ -std=c++17 -I engines/streaming-llama/prefill_llama.cpp/src tests/dsv41_step_ledger.cpp`.
- `scripts/analyze_dsv41_decode.py <producer log>` reads a run end to end: median/mean/p10/p90 of
  `step_ms` as tok/s, the tier split and implied read rate, prefill expert reuse, and the admission,
  host-plan and ds4x-open lines. A log taken before the ledger existed is still readable - it falls
  back to `LLAMADECODE_DS4X_TOKEN` and labels the tier split as inferred, because that line carries
  file bytes only and cannot separate the RAM tier from VRAM L2.
- `~/bin/ds4-first-run` prints that summary after every run. The switch has to be in the producer's
  own environment (the launcher drops inherited variables - `slstack/envvars.py` prints
  "inherited X ignored"), so the probe starts from a derived config,
  `runtime/ds4-first-run.local.json`, which adds `LLAMADECODE_BYTE_LEDGER=1` to `rtx.local.json`.

Reading the 16-token run in the table above with the new script changes what the gap is:

| | |
|---|---|
| first decode step | 11 389 ms - the prefill→decode layout switch and a cold cache (2.389 GiB fetched, `misses=240`) |
| the other 14 steps | median 292 ms, p10 137, p90 504 → 3.4 tok/s steady |
| footprint | ~2.22 GiB/step (240 expert triples × ~9.5 MiB), not 2.4 GiB |
| from NVMe | 0.98 GiB/step, 44 % of the footprint, implied read rate 3.40 GiB/s |
| from VRAM L1+L2 | 1.25 GiB/step, 56 %; `l1_slots=449 l2_slots=1636` |
| prefill reuse | 0 % - 9 908 experts active, 9 908 newly read over 40 layers |

So 1.04 tok/s is the mean of a run dominated by its own first step; steady state on the current
placement is ~3.4 tok/s with 44 % of the step still read from disk. The distance to 23.9 is real,
but it now has two separate targets rather than one: the cold first step, and the 44 % of a warm
step that comes off NVMe.

Not yet satisfied: the accept needs a run of ≥200 tokens (the probe's non-`--quick` path decodes
256). The three runs after the measured one aborted before reaching a token on `Live bias refused`;
the message now names the refused bias and the spread file, which is Step 4's calibration item and
the current blocker for any measurement run.

## Step 2 - Size the two cache tiers and buy their room

The measured run already corrects this step's premise. The ds4x decode path does not use ds4's own
streaming expert cache, so `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=2` is not the knob that sizes
what decode hits; the provider's VRAM tiers were 449 L1 slots (~4.2 GiB) and 1636 L2 slots
(16.28 GiB, `LLAMADECODE_L2 slots=1636 bytes=16283566080`), and there was **no RAM tier at all** -
`host_slots=0 host_bytes=0` in every one of the 15 steps. The tier to buy is the third one, and the
2 GiB figure was reading the other engine's cache.

- Give the host tier 16-24 GiB (`LLAMADECODE_HOST_CACHE_MB` / `DS4_CUDA_STREAMING_HOST_CACHE_N` for
  the provider's RAM tier) and check whether the VRAM L1 pool should grow with it: 449 L1 slots is
  less than two steps' worth of distinct experts at 240 triples per step. The VRAM reserve
  (`--ds4-cuda-streaming-expert-cache-reserve-gb`) only matters on the native ds4 decode path.
  Both tiers need VRAM and RSS headroom the current plan does not have: device bill 27574 MiB
  against ~30.5 GiB free, `host_budget_gib` 31.5 against a measured `peak_rss` of 26.2 GiB.
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
  shell `ulimit -l`, which is capped by `ulimit -Hl` = 8192 KB here). Measured on this box: a user
  scope cannot carry it either - `systemd-run --user --scope -p LimitMEMLOCK=…` answers "Unknown
  assignment" - and `sudo` asks for a password, so this needs a system-level unit or a
  `/etc/security/limits.conf` entry. That is root, and it is the owner's action, not the agent's.
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
- **Route-tail skip.** The baseline already runs this lever: `--experts 5` drops the sixth expert per
  word (Strata's "Faster", automatic on cards ≤16 GiB), which is ~17 % fewer expert bytes per step,
  and its release notes record +7-8 % decode with a 19/19 versus 18/19 task pass rate against the
  reference. Strata's `STRATA_ROUTE_TAIL_SKIP=7` is the miss-aware version: skip a missed expert
  ranked ≤7 in a verify window, +10-20 % on miss-heavy cards. Equivalent here would be a bounded
  skip on the decode routing path - `llama-dsv41-prefill-routes.h` currently rejects any row that is
  not exactly six ids, so it is a routing-contract change, not a flag. Treat it as a quality trade
  to measure, not a default.
- **Bias calibration.** The reasoning phase asks for a 0.35 expert-affinity bias and
  `ds4x_set_affinity` refuses any mean > 0 without a spread file bound to this exact
  model bytes/experts/layers. `presets/dsv41-affinity-spreads.txt` is calibrated for a
  315838021632-byte file, so this 365713686528-byte one is refused as wrong-model. Produce one with
  `DS4_CACHE_AFFINITY_CALIBRATE`, then set `cache_affinity_spread_file` (which
  `slstack/combo_producer.py` turns into `LLAMADECODE_DS4_SPREAD`). Until then the probe runs with
  `quality_speed: 0`. The affinity bias is also what keeps the hot expert set small, so it is part
  of step 2's hit rate rather than only a quality knob.

  Decided and landed 2026-11-09: an unbound box decodes unbiased, and it fails at launch instead
  of mid-decode.
  - `slstack/affinity_spread.py` reads the spread's own header line and checks it against the model
    file's bytes (and the 384/40 geometry). The hash stays the engine's - the launcher does not
    reimplement `cuda_affinity_header_hash`.
  - `slstack/single_runtime.py:affinity_spread` replaces the unconditional preset fallback in
    `stack_config`: an explicit `cache_affinity_spread_file` that names a different model is a
    launch error, and with nothing configured the packaged preset is offered only when it matches
    this file. `stage` refuses to package a runtime with no bound calibration.
  - `slstack/generation.py`: the published control profile is unbiased when nothing is bound, and
    `Store.bias` refuses a nonzero value with the same message. A control file that was nonzero
    before this rule existed is zeroed at the start of a request - `combo_single.run` calls
    `unusable_bias()`/`zero_bias()` and prints what it changed to the engine log - because leaving
    it is a session that dies after its prefill.
  - `tests/test_affinity_spread.py` and two cases in `tests/test_generation_controls.py` cover it;
    `tests/test_combo_single.py`'s harness now declares a calibration, since it tests the
    bias-application path. `LLAMADECODE_LIVE_BIAS` in the producer log shows which bias a run
    actually applied.

Accept for step 4: a measured decode rate for each change in isolation, and the quality difference
where the change trades accuracy (tail skip, quant).

## Step 5 - Decode several tokens per step

The per-step bytes are paid once per step, so tokens per step is the only lever that multiplies
throughput without shrinking the model or the cache. It is the one difference from Strata's 24 tok/s
that the baseline's own log cannot be read for - it runs a drafter but records no acceptance rate -
and it is the work that makes the engine usable for agentic decoding rather than merely faster.

- Establish what V4.1-Flash offers here. Measured on the file this box serves: **no draft head of
  any kind** - 1046 tensors, none named `mtp`/`draft`/`dspark`, and no matching metadata key. The
  baseline's drafter is a separate artifact: the Strata pack's `ds4.txt` declares
  `dspark_block_size 5`, `dspark_n_routed_experts 128`, `dspark_num_experts_per_tok 3`,
  `dspark_target_layer_ids 37,38,39`, `dspark_markov_rank 256`, `dspark_noise_token_id 128799`, and
  setup packs it from `dspark-…-Q8_0.gguf`. `engines/ds4` carries the DSpark kernels and a
  `dspark_capture_enabled` graph path, so the question is two-part: is there a DSpark artifact for
  the 1017 weights, and does `ds41_graph_step` (the ds4x external decode entry) reach the drafter at
  all, verifying k candidate ids in one step against the existing carry and HC state without
  re-running the routed FFN per candidate.

### The lookup (done 2026-11-09)

**The artifact exists, and it is for these weights.**
`~/workspace/qwen38-gyro/Strata-data/models/deepseek-v4.1-flash-shared/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`,
7 976 914 240 bytes, sha256 `6ba6ac1b…3927a` as recorded by its `.done` file. Its header says
`general.name "DeepSeek V4.1 Flash 1017"`, `general.version "1017"`, arch `dflash`, 78 tensors,
`dflash.block_count 3` - the three stages over target layers 37, 38, 39 - `dflash.expert_count 128`
with `expert_used_count 3`, `tokenizer.ggml.mask_token_id 128799`, markov rank 256, vocab 129 280,
hidden 5120. Every one of those matches the pack's `ds4.txt`, so it is the baseline's drafter, on
this box.

The `Q8_0` in its name is the dense part. By tensor bytes it is 6.72 GiB MXFP4 (9 routed-expert
tensors, 17 bytes per 32 elements = 4.25 bpw), 0.62 GiB Q8_0 (attention, shared experts, main_proj),
0.02 GiB F32, and 2 × 66 MB BF16 Markov tables. The experts are therefore ~5.98 MB per tensor
slice, and one proposed block routes 3 experts per stage across 3 stages and 3 tensors:

    3 experts x 3 tensors x 3 stages x ~5.98 MB ≈ 161 MB per draft block

against the target's 2.22 GiB per step. A block of 5 drafts plus the seed verifies at most 6 tokens
for 2.22 + 0.16 GiB ≈ 0.40 GiB per token, where a serial step is 2.22 GiB per token. That is the
whole argument for step 5, and the acceptance rate is what turns it into a measured number rather
than arithmetic.

**It is not consumable as released.** Four things stand between the file and `engines/ds4`, each
read from the source rather than inferred:

| gate | what the released file carries | what ds4 requires |
|---|---|---|
| `support_model_detect` (`ds4.c:3161`) | `blk.N.<role>.weight` and flat `fc`, `enc.output_norm`, `output_norm`, `conf_proj`, `markov_w1/w2` | names starting `mtp.<stage>.`, plus `.main_proj.`, `.markov_head.`, `.confidence_head.`; without them the kind is `DS4_SUPPORT_NONE` |
| `dspark_weights_validate_metadata` (`ds4.c:6517`) | none of it - the hyper-parameters are in `ds4.txt` | `deepseek41.dspark.block_size`, `.markov_rank`, `.noise_token_id`, `.target_layer_ids`. `.n_routed_experts` defaults to 0, which makes the block validator expect 384-expert tensors and reject the 128-expert ones |
| `deepseek41_sidecar_checkpoint_matches` (`ds4.c:3192`) | no `general.source.revision` at all | both files must carry it, equal, and one of the two pinned revisions. The language file has `1bae0e1088993b9b82de08f59de5be5a34e42170`, which is accepted |
| `dspark_weights_validate_layout` (`ds4.c:6638`) | Markov tables BF16 | DENSE = F16/F32/Q8_0, and the Markov argmax path (`ds4.c:36611`) wants Q8_0 |

Everything else already agrees: shape for shape the file matches
`scripts/dsv41/reference-layout.json`'s `components.dspark` map, MXFP4 experts are accepted
(`tensor_is_routed_expert_type` includes `DS4_TENSOR_MXFP4`), F32 `hc_attn_fn`/`hc_ffn_fn` are
accepted as PLAIN, and the final stage's `hc_head_*` trio is bound with `required = !v41`, so its
absence is correct for V4.1 rather than a gap.

**Landed: `scripts/repack_dsv41_dspark_sidecar.py`.** It renames to the `mtp.<stage>.` contract,
takes the six draft hyper-parameters from the pack's `ds4.txt` (refusing to guess them), adds
`general.source.revision`, re-encodes only the two Markov tables BF16 → Q8_0, and copies every
other payload verbatim. It runs the engine's own gates before writing and re-reads the result to
check both the gates and byte identity of each copied tensor. Written
`ds4-models/dspark-DeepSeek-V4.1-Flash-1017-mtp-q8.gguf`, 7 914 860 960 bytes
(Markov 66 MB → 35 MB each), sha256 `84ce012e5274f92d…01cf24`, verified clean. The engine agrees,
with no GPU involved:

    ./engines/ds4/ds4 --inspect --cpu -m <language gguf> --mtp-model <repacked gguf>
      support model (DSpark, stages=3)
      mtp/dspark: stages=3 block=5 markov_rank=256 noise_token=128799 target_layers=37,38,39
      mtp/dspark tensors: main_proj=yes main_norm=yes markov=yes confidence=yes final_head=yes
      support binding: tensors=78 missing=0 invalid=0 metadata_errors=0

One claim in that file is not self-evidenced: the released sidecar states no source revision, so
`general.source.revision` is copied from the language checkpoint and the reasoning - same release
named in `general.version 1017`, every draft tensor shape matching the reference layout - is written
into `deepseek41.dspark.repack_provenance` rather than left implicit. It should be checked against
the upstream repo before the drafter is trusted for quality, not only for shape.
`tests/test_repack_dsv41_dspark_sidecar.py` (6 cases, no GPU) pins the two header-encoding bugs
found while writing it: a BOOL metadata value decoded as an int re-encodes as a uint32 three bytes
longer and desynchronises the rest of the header, and an array whose items each repeat the element
type pushes the tensor directory past the window. Both produced a file that parsed as garbage
instead of failing, which is why the repack's own reader is what the test drives.

**The second half of the lookup: the producer's decode entry cannot reach the drafter.** `ds4x`
exports 22 symbols - `ds4x_open`, `ds4x_decode`, `ds4x_prefill`, `ds4x_bind_state`,
`ds4x_set_expert_hook{,_l2}`, `ds4x_set_affinity`, `ds4x_host_cache_v{1,2}`, `ds4x_host_timing_v1`,
the weight registry, `ds4x_vision_file`, `ds4x_close` - and none of them drafts. `ds41_graph_alloc`
under `g_ds4x_external_alloc` deliberately sizes the graph for one-token decode (`prefill_cap` for
16 rows, `carry_cap = 0`, no growth), and nothing on that path sets `g->draft`. `ds4_native_session`
refuses a state read or delta while `s->ds41_graph.draft` is set (`ds4_native_session.inc:128`), so
the checkpoint contracts do not describe drafted state today. The drafter is live only on the native
`ds4-server` path (`--dspark --mtp-model <gguf>`), which this stack already launches through
`slstack/backends.py` (`dspark`, `mtp_model`, `mtp_draft`), and which refuses to combine
`--ssd-streaming` with `--mtp-model` unless `DS4_ALLOW_MTP_STREAMING=1` or
`DS4_CUDA_DSPARK_RESIDENT=1` is set (`ds4.c:73341`). A 340.6 GiB file cannot be served there without
streaming, so that combination is the whole question, not a footnote.

That leaves the choice Step 5 was written to make, and it is now a choice between two known things
rather than a guess:

1. **Measure the drafter on the native path first.** Card free, one short request,
   `--dspark --mtp-model <repacked gguf>` with `DS4_ALLOW_MTP_STREAMING=1`, and read the acceptance
   rate the engine reports. It buys the number that steps 5 and the acceptance bar need, and it is
   weak evidence about the serving shape, not a change to it.
2. **Then decide whether to extend `ds4x`.** A `ds4x_verify`-shaped entry that drafts k ids and
   commits them through `llama_dsv41_decode_delta` and the owner-digest checkpoints is the
   structural change; the session path refusing drafted state today is exactly the risk Step 5's
   risk line names, so it gates on the delta and digest contracts, not on a throughput number.
- Verify against the existing state contracts rather than around them: `llama_dsv41_decode_delta`
  and the owner-digest checkpoints already certify device state, so a draft must extend them and
  keep the accepted-token count honest in `usage`.
- Measure acceptance rate and quality against the no-draft path at the same sampling settings. The
  baseline's log records no accepted-draft count, so it cannot be the reference for this; the
  `236 of 287` figure is a gyro-m line. Ours has to be measured, not inherited.

Accept: sustained decode ≥ 12 tok/s from cache placement alone (steps 2-4) and ≥ 24 tok/s with a
verified draft, on a real multi-turn session, with acceptance rate and a quality comparison
recorded.
Risk: state correctness. A draft that advances device state the checkpoint cannot describe is worse
than slow, so the delta and digest contracts gate it, not a throughput number.

## Step 6 - Reuse across agent turns

Strata reuses 81.6 % of prompt tokens on live traffic, and an agent session's second turn is
mostly the first turn's context. Reuse is worth more than raw tok/s in this workload because it
removes prefill from every turn after the first.

- Measure it here: the engine already reports `cache_n`/`prompt_n` and tonight's first turn showed
  `cache_n=0`, which is expected for a cold session. Run a multi-turn session and read `cache_n` on
  the second turn.
- The carry store and `DS4_COMBO_RESTORE_MAPPING` / expert-restore file are the mechanism; confirm a
  second turn takes the append path (`restart_prompt`/append prefill) instead of retiring the
  producer, and that the resident expert set survives the turn boundary.

Accept: ≥ 70 % of the second turn's prompt tokens reported as reused, and the retired-producer
path taken only on a real prefix divergence.

## Acceptance

The bar is the measured Strata + DeepSeek number on this box, on agentic traffic, at 262k context:

- sustained decode median ≥ 24 tok/s on generations of ≥ 200 tokens,
- ≥ 70 % prompt reuse on the second turn of a session,
- no admission refusal, no budget breach, no paging growth,
- answers that match the no-bias, no-draft reference within the tolerance the affinity bias is
  allowed to introduce.

Compare like for like. The 23.9 tok/s baseline is `--max-context 131072`, five experts per word
instead of six, DSpark on, and a CPU lane for 5.7-9.8 % of the experts, on a 129 GB quant. Hitting
24 tok/s at 262k with six experts and no drafter is a harder number than the table suggests; the
honest reading of this bar is that one of those trades has to be made and recorded, not that all of
them can stay.

Steps 1-4 are tuning and are worth doing in order. Step 5 is the structural one: without tokens per
step, cache placement alone cannot reach the bar with a 365 GB file. If steps 1-4 land and steps 5-6
have not, the honest report is "understood, not accepted": the gap is real, and the fix is draft
decoding plus reuse, not a narrower role for the engine.

## Where this stands (2026-11-09, later in the day)

Step 1 is landed and measured, the baseline has been re-read, the bias blocker is closed as a
launch-time decision, and Step 5's lookup is answered: the DSpark artifact for the 1017 weights is
on this box, it is now repacked into the contract `engines/ds4` binds and the engine accepts it
(`support binding: tensors=78 missing=0 invalid=0`), and the producer's `ds4x` decode ABI has no
path to it. Everything still open needs the card.

1. **Free the card.** `strata --serve --pack packs/gyro-m` holds ~29 GiB here and is still serving -
   measured 30 849 MiB used, 1 346 MiB free, with a `strata-vision` generation in flight - so the
   stack's lease gate refuses admission and no measurement below can run. Stopping it is the
   owner's call, not the agent's.
2. **Run the probe with calibration:** `~/bin/ds4-first-run --calibrate runtime/qualification/
   dsv41-1017-iq2xxs-spreads.txt` (not `--quick`). The flag puts `DS4_CACHE_AFFINITY_CALIBRATE` in
   the derived config's `env_overrides`, so one sitting gives Step 1's accept (256 decoded tokens
   with `LLAMADECODE_STEP_BYTES`), whether the RAM tier is admitted at all, and a spread bound to
   these 365713686528 bytes for Step 4's bias item. Read the result with
   `scripts/analyze_dsv41_decode.py`, which prints the calibration line the producer wrote (or
   `incomplete affinity calibration` if some layer never reached 16 samples). `runtime/qualification/`
   is still empty.
3. **Then Step 6 and Step 2**, against the numbers from (2) rather than the arithmetic here:
   `scripts/dsv41_reuse_probe.py` is the multi-turn reuse probe (it reads `timings.cache_n` on the
   second turn and marks PASS/FAIL against the 70 % bar), and the host-tier sizing whose 3.5 GiB
   offer is derived, not measured (`--job-host-bytes` 30 GiB minus a 26.2 GiB peak RSS minus the
   256 MiB margin in `configure_host_cache`).
4. **Step 5, once the card is free:** `scripts/ds4_dspark_probe.sh` measures the drafter's
   acceptance rate on the native `--dspark --mtp-model` path with `DS4_ALLOW_MTP_STREAMING=1` -
   `DS4_DSPARK_SPEC_LOG=1` prints one `DSpark V4.1 cycle drafted= accepted=` line per verify cycle,
   and the script turns those into an acceptance rate next to the tok/s. Run it twice, with and
   without `--no-draft`, at the same `max_tokens`. That is what makes the 5.5× arithmetic above a
   measurement. Only then decide whether `ds4x` grows a verify-shaped entry.

Two files this work touched live outside the repo: `~/bin/ds4-first-run` (the probe; it now runs the
analysis and takes `--quick` / `--calibrate PATH`) and `runtime/ds4-first-run.local.json` (the
derived config that puts `LLAMADECODE_BYTE_LEDGER=1`, and any calibration path, in the producer's
environment; the probe regenerates it from `rtx.local.json` on each run). Step 5's repacked
sidecar is a third: `ds4-models/dspark-DeepSeek-V4.1-Flash-1017-mtp-q8.gguf`, next to the language
file, regenerable from `scripts/repack_dsv41_dspark_sidecar.py`.

Verification baseline for this box, so a later session does not chase it: `pytest tests/` was
31 failed / ~680 passed before any of this work - the cgroup and RAM-scope tests in
`tests/test_working_set.py`, the guard and process tests, `tests/test_yoco.py` - and it is still
those same 31 (now 31 failed / 691 passed, six of the passes from the sidecar test).
`tests/test_combo_single_stop.py::test_disconnect_inside_reasoning_then_next_turn_appends_only`
fails 3 to 4 runs in 10 under load on a clean checkout as well. The C++ selftests
(`tests/dsv41_host_plan_log.cpp`, `tests/dsv41_step_ledger.cpp`) are hand-compiled; `build.sh` has
no C++ harness.

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
