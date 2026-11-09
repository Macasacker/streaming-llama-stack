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

### Step 1 accept, measured (2026-11-09 19:28)

A 256-token decode on the current placement, read with the script over the run's three producer
logs (`runtime/debug/run-20261109-192819.log`):

| session | steps | median step | mean step | tok/s median | from_file | hit rate |
|---|---|---|---|---|---|---|
| short (23 tokens) | 23 | 312.9 ms | 801.3 ms | 3.20 | 36.8 % | 63.2 % |
| **decode (256 tokens)** | **255** | **167.5 ms** | 226.8 ms | **5.97** (mean 4.41) | **22.8 %** | **77.2 %** |
| long prefill (64 tokens, after 64 854) | 63 | 229.6 ms | 440.7 ms | 4.35 | 30.2 % | 69.8 % |

The 255-step row is the plan's accept: a run of ≥ 200 tokens whose log says where each step's
2.22 GiB came from, with `from_l1 42.6 %`, `from_l2 34.6 %`, `from_file 22.8 %`, `from_ram 0 %`,
`copied 2.007 GiB/step` device-to-device, and zero steps with unattributed bytes. Steady decode
is 5.97 tok/s median, not 3.4, and the disk share of a warm step is 22.8 %, not 44 %. The p10 of
88.6 ms says the cache can serve a step at 11 tok/s when the hot set holds; p90 283.7 ms is a step
that pays NVMe again. Against the bar: 5.97 of 23.9 tok/s.

Prefill on this box measured 859 tok/s (64 854 tokens in 75.5 s) in the same run, well below the
2 000 tok/s in the table above, and prefill expert reuse is again 0 % (10 009 active, 10 009 newly
read over 40 layers). That is a second, unexplained number: prefill rate is not this plan's bar,
but a 2.3× spread between two measurements of the same file on the same box means the table's
prefill row is not a stable reference.

Three things this run also proved, each of which changes a later step:

1. **`host_slots=0` is a default, not a refusal.** `configure_host_cache`
   (`llama-dsv41-weights.cpp:1613`) returns immediately when `LLAMADECODE_HOST_CACHE_MB` is unset,
   empty or `0`; the doc line that says "auto uses admitted physical/process headroom" reads as
   though unset were auto, and it is not. The room was there - `row_cache_headroom=5.80 GiB`,
   `available 61 GiB`, `admitted_host=32.08 GiB` - and nothing claimed it. Step 2's first move is
   therefore to set the variable, and its second is to raise what the offer is bounded by:
   `offer = min(requested, physical, --job-host-bytes - rss - 256 MiB)`, which on today's numbers
   (30 GiB job host, ~26 GiB RSS) is ~3.5 GiB even with `auto`. That is the plan's derived 3.5 GiB,
   now confirmed as the formula's output rather than a guess.
2. **The calibration was collected and lost.** Two sessions show
   `ds4: cannot write affinity calibration` followed by `LLAMADECODE_PRODUCER_ABORT`: the publish
   gate passed (`ds4_gpu_cache_affinity_finish` only writes when every layer reached 16 samples,
   and 255 decoded tokens cleared it), then `fopen` failed because `--calibrate` was given a path
   *relative to the repo* while the producer resolves it against its own working directory, and
   ds4 `abort()`s on that write failure (`ds4_cuda.cu:1005`). `runtime/qualification/` is empty.
   `~/bin/ds4-first-run` now resolves the path to absolute and proves the target writable before
   starting anything.
3. **That abort is why turn reuse cannot be measured yet.** Each of the three requests got its own
   producer directory (`ultra-e989bff…`, `ultra-f80a97f…`, `ultra-a955e1d9…`) and reported
   `cached_tokens=0`, because the producer killed itself at session close instead of surviving to
   the next turn. Step 6 needs the fixed path first; with it, `cache_n=0` on turn 2 would be the
   append path refusing, which is the thing Step 6 is actually about.

The bias rule worked as designed on a live run for the first time: `LLAMADECODE_LIVE_BIAS value=0`,
`DS4X_AFFINITY mean=0 … enabled=1`, all three requests served, no `Live bias refused`. Step 3's
pageable carry is unchanged and still needs root: `CARRY_RAM_LOCK_PARTIAL bytes=2522877952
locked_bytes=8388608 pageable_tail_bytes=2514489344 errno=12` in the 255-step session, and
3.32 GiB in the prefill-heavy one.

Step 1's accept is met by the 255-step row. What is still open is the calibration itself: the file
was collected and lost to the relative path, so Step 4's bias item and Step 6's reuse measurement
both wait on a rerun with an absolute `--calibrate` path.

## Step 2 - Size the two cache tiers and buy their room

The measured run already corrects this step's premise. The ds4x decode path does not use ds4's own
streaming expert cache, so `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=2` is not the knob that sizes
what decode hits; the provider's VRAM tiers were 449 L1 slots (~4.2 GiB) and 1636 L2 slots
(16.28 GiB, `LLAMADECODE_L2 slots=1636 bytes=16283566080`), and there was **no RAM tier at all** -
`host_slots=0 host_bytes=0` in every one of the 15 steps. The tier to buy is the third one, and the
2 GiB figure was reading the other engine's cache.

- Give the host tier 16-24 GiB. The knob is `LLAMADECODE_HOST_CACHE_MB`, and it is **off unless
  set**: `configure_host_cache` returns when it is unset, empty or `0`, so every run so far has had
  no RAM tier because nobody asked for one, not because admission refused it. `auto` takes
  `min(physical headroom, --job-host-bytes - rss - 256 MiB)`, which at today's numbers is ~3.5 GiB;
  reaching 16-24 GiB needs `producer_job_host_gib` raised as well, not just the variable set.
  `DS4_CUDA_STREAMING_HOST_CACHE_N` caps the same tier by expert count. Check whether the VRAM L1
  pool should grow with it: 449 L1 slots is less than two steps' worth of distinct experts at 240
  triples per step, and L1 already serves 42.6 % of a warm step's bytes. The VRAM reserve
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

Measured verdict, 2026-11-09 (runs E, I, G, H in the next subsections): **the bar is met, but by the
affinity bias and not by the RAM tier.** At bias 0 the hit rate is 75.7-77.0 % and the median step
is ~150 ms (6.6-6.7 tok/s) whether the 12.9 GiB tier is funded or not; at bias 0.5 with the tier
off it is 93.0-93.4 % and ~47-50 ms (19.8-21.3 tok/s). The tier cost 1.88 GiB/step of extra
device-to-device traffic and served 1.4 % of the footprint, so `LLAMADECODE_HOST_CACHE_MB` stays
off on this box. What the bar does not capture is the distribution: our p10 (25.2 ms) already
matches Strata's (21.4 ms) while our p90 (94.8 ms) is 3.5× its 27.2 ms, so the mean a reader
experiences is still 10.2 against Strata's 24.2. That tail is Step 5's problem, not a cache-sizing
problem.

### First funded attempt (2026-11-09 19:44): the tier exists, the paging guard stopped it

`host_budget_gib` 50, `producer_job_host_gib` 48, `producer_graph_cap_gib` 3.5,
`LLAMADECODE_HOST_CACHE_MB=auto`. The device side improved as designed and admitted cleanly -
graph 5 120 MiB (was 5 632), bill 27 062 MiB, gate needs 28 086 MiB free, `measured_admission` 30.5
GiB - and the tier was sized for the first time:

    LLAMADECODE_HOST_CACHE slots=2833 bytes=28257361920 offer=28204511232
                          host_limit=51539607552 rss_before=23006941184 available_after=41876819968

26.3 GiB of RAM tier, 2 833 slots, and 41.9 GiB still available afterwards. Then the producer
aborted after one decoded token:

    HTTP 503  "Persistent producer stopped: paging detected (pagefile usage rose 293703680 bytes)"
    LLAMADECODE_PRODUCER_ABORT kind=host_memory reason=… (swap growth limit 256 MB)

That is the risk this step names, firing immediately rather than silently. It is not an availability
failure - `runtime_min_free_gib` never tripped, 41.9 GiB stayed available - it is the *swap* guard:
`paging_limit = 256 000 000` bytes of growth in system-wide swap-in-use
(`llama-dsv41-host-memory.h:159`, a constant carried over from Studio's Windows pagefile check, and
the box has 10 GiB of swap with 1.4 GiB already in use). `auto` took the entire 26.3 GiB offer as
anonymous memory, and with the default `vm.swappiness` the kernel pages anonymous memory out before
it reclaims the model's file cache, so a 26 GiB anonymous jump crosses 256 MB of swap even while
41 GiB is available. The calibration was lost to the same death, but honestly this time -
`ds4: incomplete affinity calibration; no file published`, because the producer reached one token,
not the 16 samples per layer.

Three ways to take Step 2, in the order they should be tried:

1. **Cap the tier explicitly.** `LLAMADECODE_HOST_CACHE_MB: "12288"` - a 12 GiB tier, half the
   anonymous jump, still twice what `auto` offered at the old 30 GiB job host. Set on this box
   after this run.
2. **Lower the offer instead**, by taking `producer_job_host_gib` back toward 40, which bounds
   `auto` without a second knob.
3. **`vm.swappiness` near 0** is the setting that fits this box's design - the engine leans on
   reclaimable page cache from a 340 GiB file, and the guard's 256 MB limit assumes a Windows
   pagefile. That is a sysctl, so it is root and the owner's action, in the same category as
   Step 3's `LimitMEMLOCK`.

What must not happen is raising `paging_limit` to make the run pass. The guard is the only thing
that says "this machine is now paging"; the plan's line stands - a refusing guard is information,
not a regression to silence. What is still owed is the number the tier was bought for: hit rate
against 90 % and `from_ram` greater than zero on a ≥ 200-step decode.

### Second funded attempt (2026-11-09 19:51): the tier served bytes and decode got worse

`LLAMADECODE_HOST_CACHE_MB=12288`. No paging stop this time - the guard saw 36 864 bytes of system
swap growth, unattributed - the tier was admitted at 12.88 GiB / 1 288 slots, and it did serve
bytes. It also made decode slower than the run with no tier at all:

| config | steps | median step | tok/s median | from_file | from_ram | hit rate | device copies |
|---|---|---|---|---|---|---|---|
| no tier, graph cap 5.5, job host 30 | 255 | 167.5 ms | **5.97** | 22.8 % | 0 % | **77.2 %** | 2.007 GiB |
| tier 12.9 GiB, graph cap 3.5, job host 48 | 160 | 219.5 ms | **4.56** | 26.5 % | 1.5 % | **73.5 %** | 3.938 GiB |

The 160-step session also ended on `LLAMADECODE_PRODUCER_ABORT kind=exception reason=Physical free
VRAM fell below the decode VRAM reserve` partway through a 256-token request, and the
post-prefill session came in at 3.80 tok/s with a 68.6 % hit rate.

Two things are being blamed for that and only one of them can be. The tier explains `from_ram` and
the RAM-tier stores (0.028 GiB/step), but nothing explains copies nearly doubling - 1.93 GiB more
device-to-device traffic per step than a run whose footprint is the same 2.22 GiB. The graph cap
went 5.5 → 3.5 in the same edit, and prefill fell from 2 000 tok/s in the table to 859 and then
848 tok/s in these two runs, which is the same direction and the same suspicious magnitude. So
`producer_graph_cap_gib` is back to 5.5 and the next run changes only the tier:

| run | changes from the 19:28 baseline | what it decides |
|---|---|---|
| A | tier 12.9 GiB, cap 5.5 | whether the tier itself costs the 1.3 tok/s and the copies |
| B | A, plus `cache_affinity_spread_file` bound | what the bias buys, measured against A |

A is the run Step 2's accept is judged on. If A still shows ~4.5 tok/s and doubled copies, the RAM
tier is not the lever on this box at this footprint, and Step 4's quant lever moves up the list.

### A and B (20:09, 20:13): neither reached 200 steps, and the reason is 30 MiB

Run A changed the tier and nothing else. Run B additionally bound the calibration. Both ended the
256-token decode on the same guard, at 16 and 32 steps:

| session | steps | median | hit rate | from_ram | device copies | how it ended |
|---|---|---|---|---|---|---|
| A short (24 tok) | 23 | 295.4 ms | 63.2 % | 0 % | 3.328 GiB | completed |
| A 256-token | 16 | 390.9 ms | 50.8 % | 0 % | 2.956 GiB | **VRAM reserve breach** |
| A long prefill | 0 | - | - | - | - | HTTP 503, producer never opened |
| B short (28 tok) | 27 | 176.8 ms | 66.5 % | 0 % | 3.420 GiB | completed |
| B 256-token | 32 | 169.0 ms | 64.5 % | 0 % | 3.665 GiB | **VRAM reserve breach** |
| B long prefill | 63 | 199.1 ms | 69.0 % | 0.2 % | 3.794 GiB | completed, 64 tok |

None of these can decide Step 2 - the plan's bar is a run of at least 200 tokens, and a 16-step or
27-step session has a cold cache by construction. Prefill reproduced a third time: 922 tok/s, in
the same 848-922 band as the previous two runs.

The breach is now measured rather than assumed, from the headroom lines the guard prints:

```
steady, just after L2 fill   nvml_free_now=152 961 024  reserve=134 217 728   device_used=34 037 956 608
dip that killed run A        nvml_free_now= 80 412 672  min_free_since_l2=80 412 672  samples=1959
the 255-step run that lived  min_free_since_l2=169 869 312  reserve=134 217 728
```

So the plan leaves about **150 MiB above a 128 MiB reserve**, and the decode's transient
allocation dips ~70-80 MiB below its own steady state. The run that survived 255 steps sat 42 MiB
higher and never dipped into the reserve. That is the whole margin, which is why the funded-tier
runs die at 16-32 steps and the no-tier run does not: the tier's pinned rings are host memory, but
whatever the decode adds on the device is enough to cross 128 MiB when there are only 150 MiB of
slack.

`LLAMADECODE_VRAM_MARGIN_MIB` is the knob, and its own comment states the owner rule: the margin
"may be cut but not zeroed", default 64 MiB, and 128+0 "breached the reserve in every run". The
stack does not set it, so 64 applies. Raised to **1024** in `rtx.local.json`: the L2 stops growing
~100 slots earlier (out of ~1640, about 6 %) and the free VRAM sits ~1.15 GiB above the reserve
instead of 0.15 GiB. That is moving the plan back from the edge, not widening the guard - the
reserve stays 128 MiB and the guard still fires if the plan is too big.

Run B also settled something about Step 4 that the probe had been getting wrong. The spread reached
the engine (`DS4X_AFFINITY spread=runtime/qualification/dsv41-1017-iq2xxs-spreads.txt`) - the bound
size and hash check above is what let it through - but the bias stayed at zero:

```
LLAMADECODE_LIVE_BIAS value=0 position=588
LLAMADECODE_DS4X_AFFINITY mean=0 warmup=0 ramp=0 spread=… enabled=0
```

Because the bias is not a config value: `combo_single` calls `generation.bias(quality_speed)` at
`slstack/combo_single.py:266`, and the probe's requests send no `quality_speed`, so every request
resolved to 0.0. Binding a calibration makes a nonzero bias *possible* - `Store.bias` raises when
nothing is bound (`slstack/generation.py:128`) - it does not apply one. `~/bin/ds4-first-run`
now takes `--bias[=VALUE]`, puts `quality_speed` on all three requests, and refuses to start if
`cache_affinity_spread_file` is unset, since that raises on the first request instead of on the
card. `scripts/analyze_dsv41_decode.py` prints the headroom and breach lines, so the reserve's
real margin is in every summary from now on.

The next two runs, in this order, with Strata off:

```bash
~/bin/ds4-first-run              # C: tier 12 GiB, margin 1024, bias off - Step 2's number
~/bin/ds4-first-run --bias       # D: same, with quality_speed 0.5 on every request - Step 4's
```

C is the comparison against the 19:28 baseline's 5.97 tok/s and 77.2 % hit rate; read
`min_free_since_l2` in the headroom line before believing anything else, and note the L2 slot count
(~1540 rather than ~1639) so the smaller cache is visible in the hit rate. D is the same run with
the bias live; Step 4's accept also needs the output compared by hand, since the bias re-ranks
experts and can trade speed for quality.

### C and D (20:40, 20:44): the margin change was inert, and the guard is a coin flip

C ran with the same L2 as B: 1633 slots, `offered=16 263 151 616` from `nvml_free=16 474 963 968`.
That subtracts 202 MiB, which is `128 + 64 + 10` - the engine's default margin, not the 1024 the
config asked for. `scripts/dsv41_single.py` is why:

```python
env = dict(result.get('env_overrides') or {})
env.update(COMMON_ENV)          # COMMON_ENV overwrites whatever the local config set
```

and `COMMON_ENV` pins `LLAMADECODE_VRAM_MARGIN_MIB: '64'` and `LLAMADECODE_VRAM_RESERVE_MIB: '128'`.
`DEFAULT_ENV` just below it uses `setdefault`, so ordinary keys do survive a local config -
`LLAMADECODE_HOST_CACHE_MB` reached the producer, which is why the tier was funded at all - but
those two are gated and a local value is silently dropped. Running `dsv41_single.overlay('262k',
rtx.local.json)` shows it directly: the merged env carries margin 64 while `LLAMADECODE_L2_MAX_SLOTS`
carries whatever the config asked. So the margin override in `rtx.local.json` was removed rather
than left in looking like it was doing something.

The gate is deliberate - the comment on it says 128 + 0 breached the reserve in every measured run -
so the lever is the ungated one: **`LLAMADECODE_L2_MAX_SLOTS: "1500"`**, which `grow_decode_cache`
applies to any caller and which the variable's own doc describes as being for reproducible tier
comparisons. 1633 → 1500 slots is 133 × 9.95 MiB ≈ **1.3 GiB** returned to the device, moving the
steady free from ~154 MiB to ~1.5 GiB above a 128 MiB reserve.

And C proved the abort is not bad luck. The session that *survived* its decode - the 64-token
long-prefill run, 63 steps, 4.20 tok/s - recorded:

```
LLAMADECODE_VRAM_HEADROOM nvml_free_now=199098368 min_free_since_l2=35520512 samples=3249 reserve=134217728
```

The sampler saw free VRAM at 34 MiB, well under the 128 MiB the guard refuses below, and the run
lived anyway because `check_decode_vram_reserve` reads NVML at its own call sites rather than at
the dip. The 256-token session died on the same dip at 30 MiB. So the plan currently over-subscribes
the card by about a gigabyte and whether a run finishes depends on when the guard happens to look.

C's measurement content, for the record: no session reached 200 steps; `from_ram` was 0.0 % and
0.1 % with `stored` at 0.015 GiB/step; prefill was 864.9 tok/s, a fourth reproduction of the
848-922 band.

D failed before any producer started: `[Errno 98] Address already in use`, returned as
`engine_unavailable` in 0 s. C's forced producer kill left the *worker* alive, and the worker binds
its own port (`runtime/single-engine.json` says 8089, the router takes 8088), so D's engine could
not listen. `~/bin/ds4-first-run` now reads both ports from the configs, force-kills stack processes
still holding either after the cooperative stop, and waits for them to clear before starting -
matching on the command line (`slstack`, `llama-dsv41-prefill`, `ds4-server`) rather than on the
port alone, so a different service on a neighbouring port is never killed: Strata serves its own
UI on 8084 and stays out of it.

The next run is therefore E, with the L2 capped and the bias still off:

```bash
~/bin/ds4-first-run          # E: tier 12 GiB + L2 1500 slots, bias off - Step 2's number
~/bin/ds4-first-run --bias   # F: the same configuration with quality_speed 0.5 - Step 4's
```

Read `LLAMADECODE_L2 slots=` first to confirm the cap took effect (it should read 1500, not 1633),
then `LLAMADECODE_VRAM_HEADROOM`'s `min_free_since_l2` before trusting any rate.

### E (20:55): Step 2 has a number, and it clears everything except the 90 % bar

The cap took effect (`LLAMADECODE_L2 slots=1500`, `total_vram_slots=1949`), and for the first time
since the RAM tier was funded the decode finished its 256 tokens:

```
LLAMADECODE_VRAM_HEADROOM nvml_free_now=1611923456 min_free_since_l2=1448738816 samples=7679 reserve=134217728
LLAMADECODE_DECODE_COMPLETE generated=256 evaluated=255 stopped=0 decode_ms=52736.117 tok_s=4.835396
```

| run | L2 | tier | steps | median | tok/s median | hit rate | from_ram | device copies |
|---|---|---|---|---|---|---|---|---|
| 19:28 baseline | 1633 | off | 255 | 167.5 ms | 5.97 | 77.2 % | 0 % | 2.007 GiB |
| E | 1500 | 12.9 GiB | 255 | **151.7 ms** | **6.59** | 77.0 % | **1.4 %** | 3.953 GiB |

So the tier costs device-to-device traffic (1.95 GiB/step more than the no-tier run) and buys a
10 % better median step and the first nonzero `from_ram` on a long decode: 0.032 GiB/step, which is
3-4 of its 1 288 slots per token. The hit rate is the same 77 % as without it, which is the point:
the tier is fed by L1/L2 victims and a RAM hit consumes the slot
(`engines/ds4/ds4_cuda.cu:6845-6853` invalidates it after the upload), so it cannot absorb the 23 % miss
stream that reaches NVMe every step. 90 % needs the miss stream to land in RAM, and this placement
policy does not do that.

Against Step 2's accept bar:

- decode ≥ 4 tok/s: **pass**, 6.59 tok/s median, 4.84 tok/s mean over 255 steps.
- `host_slots > 0` and `from_ram > 0`: **pass**, 1 288 slots and 1.4 % of the footprint.
- no admission refusal, no paging stop: **pass** - the guard saw 2.3 MiB of system swap growth,
  unattributed, and let the run finish.
- hit rate ≥ 90 %: **fail at 77.0 %**, for the mechanism reason above rather than a sizing one.
- no `budget_breaches`: pass.

Prefill reproduced again at 904.8 tok/s, fifth run in the 848-922 band. The port cleanup landed in
the same run: `stack worker still holding a port (8088 8089): 372891 - forcing kill`, after
`Single-engine graceful shutdown failed; stopping the owned job` in the router's log - which is
exactly the case F would otherwise have died on.

### F (21:00): the bias applies, and the paging guard measures the machine, not the producer

`--bias` did what Step 4's first item asks:

```
LLAMADECODE_DS4X_AFFINITY mean=0.5 warmup=0 ramp=0 spread=runtime/qualification/…  enabled=1
LLAMADECODE_LIVE_BIAS value=0.5 position=588
```

No `Live bias refused`, the spread bound, bias nonzero at the reasoning phase. It then died at the
first decoded step:

```
DSV41_HOST_MEMORY_PAGING_BASELINE pagefile_in_use_bytes=1394110464 growth_bytes=0 limit_bytes=256000000
DSV41_HOST_MEMORY_PAGING_STOP     pagefile_in_use_bytes=1719590912 growth_bytes=325480448 low_memory_events=0
```

That is not the producer swapping. `slstack/cgroup_scope.py:107` sets `memory.swap.max=0` on the
scope the producer runs in, so it cannot swap at all, and the guard's `pagefile_in_use` is
`SwapTotal - SwapFree` from `/proc/meminfo` - machine-wide. The same guard log prints
"pagefile-specific hard-fault events are not available on this platform", which is why it can only
report growth and never a cause. With 56 GiB of RAM available and `low_memory_events=0`, the 325 MiB
went to the desktop: reading `/proc/*/status` afterwards, 1.82 GiB of swap sits spread across
uvicorn (0.16), Hyprland (0.12), four `pi` processes (0.34), Brave (0.16), ghostty, the Hermes
gateway and the VPN UI - nothing from the stack.

So the honest reading is that the producer's 12.9 GiB pinned tier plus its 23 GiB RSS pushes
*other* processes' anonymous pages out, and the guard attributes that to the run. E did the same
thing and grew only 2.3 MiB; F's first step read 2.225 GiB entirely from file (bias re-ranks
experts, so nothing in the cache is the expert the router now wants) and pushed the reclaim further.

Step 4's speed question therefore needs a run that does not add 12.9 GiB of pinned memory to a box
that is already swapping 1.33 GiB. Next is G, with the tier off, which also pairs with the 19:28
baseline as the clean single-variable comparison - bias on versus off, same device plan:

```bash
# rtx.local.json: LLAMADECODE_HOST_CACHE_MB "0", LLAMADECODE_L2_MAX_SLOTS "1500" (keep)
~/bin/ds4-first-run --bias     # G: bias on, tier off
```

If G trips the same guard, the bias path itself is what pushes this desktop into swap and the fix is
machine-level (`vm.swappiness`, root) rather than anything in the stack. Raising `paging_limit` is
still not the answer - the guard is telling the truth about a box that is swapping; it is just not
telling the truth about *who* swapped.

### G and H (21:06, 21:10): ~20 tok/s and a 93 % hit rate - and H was not the control

G ran with the tier off and the bias live, and finished its 256 tokens:

```
LLAMADECODE_LIVE_BIAS value=0.5 position=590
LLAMADECODE_DS4X_AFFINITY mean=0.5 … enabled=1
LLAMADECODE_VRAM_HEADROOM … min_free_since_l2=1543700480 reserve=134217728
LLAMADECODE_DECODE_COMPLETE generated=256 evaluated=255 decode_ms=26064.967 tok_s=9.783
```

| run | bias | tier | steps | median | tok/s median | mean | hit rate | from_file | copies |
|---|---|---|---|---|---|---|---|---|---|
| E | 0 | 12.9 GiB | 255 | 151.7 ms | 6.59 | 4.84 | 77.0 % | 23.0 % | 3.953 GiB |
| G | 0.5 | off | 255 | 50.4 ms | **19.83** | 9.78 | **93.4 %** | **6.6 %** | **1.693 GiB** |
| H | 0.5 | off | 255 | 46.9 ms | **21.31** | 10.22 | **93.0 %** | 7.0 % | 1.727 GiB |

H was meant to be the unbiased control and it was not. `runtime/generation/live.json` holds
`"value": 0.5, "revision": 14`, and `combo_single.py:753` calls `set_quality` **only when the
request body contains `quality_speed`** - so a request that omits the key leaves the previous run's
published bias in force. H's log reads `mean=0.5 … enabled=1` for exactly that reason. The accidental
second bias run is still worth having: it reproduces G within 7 % on the median (19.83 / 21.31) and
within 0.4 points on the hit rate, which is the first pair of runs on this box that agrees at the
255-step scale.

What E and G differ by is two things at once - the bias and the tier - so neither of those numbers
is attributable yet. The 93 % hit rate and 6.6 % from-file could be the bias shrinking the hot set
(it re-ranks experts, so the working set stops reaching past VRAM), or the tier going away
(copies fell from 3.953 to 1.693 GiB/step, which is the tier's traffic, not the bias's), or both.
Run I is the control: bias explicitly 0, tier off, same L2 cap.

`~/bin/ds4-first-run` now sends `quality_speed` on **every** request - `0` when `--bias` is not
given - and prints the persisted control value before starting, because "no `--bias`" no longer
means "unbiased" on a box that has run `--bias` before. It also prints the 503 body of the long
prefill (three runs' refusals were unreadable because the probe printed only the exception) and the
reasoning/answer split of the decode.

That split matters for Step 4's quality item. All three decode responses were read off disk: each
spent the entire 256-token budget in `reasoning_content` with an empty message and
`finish_reason=length`, so none of them reached an answer. The reasoning text is coherent in all
three and shows no repetition degeneration (166-186 words, unique-word ratio 0.69-0.75, worst
repeated 5-gram 1-2 occurrences - H repeats its own opening draft line once). So the bias has not
produced garbage, but "compare the output" needs a request whose budget outlasts the reasoning
phase; the 48-token short request is what actually answers ("Lisbon" in all three).
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

  Corrected the same day, by the first calibration this box produced. The rule above checked the
  model file's *length*, and that is not what the engine binds: `model_unmap_engram`
  (`engines/ds4/ds4.c`) sets `ds4_model.size` to the start of the disk-only Engram region - those
  rows are deliberately outside the mapping - and `cuda_affinity_model` compares that size plus the
  FNV-1a hash of the first 64 KiB of the mapping. The producer calibrated this 365 713 686 528-byte
  file and wrote `DS4AFFINITY1 162955640832 384 40 436cb28b0673b077` - the offset of
  `blk.1.engram_embd.weight` - and the launcher refused its own calibration as wrong-model.
  `slstack/affinity_spread.py` now derives the bound bytes from the GGUF tensor directory the way
  ds4 does, and mirrors the hash too: reading 64 KiB costs nothing at launch and turns a
  same-length-different-file refusal, which would otherwise happen at the first decoded token, into
  a launch error. Verified against the real file: derived bound 162 955 640 832, derived hash
  `436cb28b0673b077`, both equal to the header the engine wrote; the packaged preset still does not
  match and bias stays off.

  The calibration exists now - `runtime/qualification/dsv41-1017-iq2xxs-spreads.txt`, 40 layers,
  61 samples each, written by the 160-step session at 19:55. Binding it is a separate measurement
  from the cache work, because the bias changes routing and therefore changes the hit rate: run the
  cache comparison unbiased first, then set `cache_affinity_spread_file` and compare.

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

### Answered by production traffic (2026-11-09 21:38-21:45): reuse works

The first real pi session against the stack ran six turns through **one** producer
(`runtime/single-engine/ultra-73c876d05eb4e674`), and every turn after the first took the append
path. From `runtime/single-engine/single-request-*.json` and the session's `DS4PERF_PROMPT_HEAD_COMPLETE`
lines:

| turn | prompt tokens | reused | new | append ms | decode ms | generated | tok/s | finish |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 666 | 0 | 16 666 | 33 300 | 21 164 | 81 | 3.78 | stop |
| 2 | 16 768 | 16 746 (99.9 %) | 22 | 6 006 | 15 572 | 190 | 12.14 | tool_calls |
| 3 | 18 362 | 16 957 (92.3 %) | 1 405 | 28 888 | 34 812 | 270 | 7.73 | tool_calls |
| 4 | 25 021 | 18 631 (74.5 %) | 6 390 | 33 849 | 32 752 | 281 | 8.55 | tool_calls |
| 5 | 26 899 | 25 301 (94.1 %) | 1 598 | 28 542 | 23 217 | 133 | 5.69 | tool_calls |
| 6 | 42 381 | 27 031 (63.8 %) | 15 350 | 32 642 | 12 271 | 3 | 0.16 | cancelled |

Turn 6 was cancelled by the client three tokens in (`cancelled=1`), which is why its tok/s is
noise. Reuse averaged ~86 % across turns 2-5 and the mechanism is exactly what the step predicted:
`prefix_source: "memory-token"`, reused equal to the previous turn's context length.

This also corrects what the probe runs seemed to show. `cached_tokens=0` in every earlier run was
not the append path refusing - `combo_single.py:539-540` does publish `reused_prefix_tokens` as
both `usage.prompt_tokens_details.cached_tokens` and `timings.cache_n`. It was that each probe
request is a different prompt, so each one retired the producer and opened a new session, and a
fresh session has nothing to reuse. A client that grows its context monotonically - which is what
an agent does - keeps one producer alive and reuses.

What the numbers do not support is the claim that reuse removes prefill from later turns. The
append path has a large fixed cost: 22 new tokens took 6.0 s, 1 405 took 28.9 s, 1 598 took 28.5 s,
6 390 took 33.8 s and 15 350 took 32.6 s - so turn 5 paid ~28 s to prefill 1 598 tokens (56 tok/s)
while a cold 16 666-token prompt cost 33.3 s (500 tok/s). Reuse of 25 301 tokens bought almost
nothing on wall clock. Turn costs were 20-76 s, dominated by this, not by decode. The session's
`DS4PERF_UNALIGNED_APPEND_COMPLETE` lines show the appended rows handled as 2 048-row tiles per
layer, so the sweep granularity, not the token count, looks like the floor - that is the next thing
to measure before touching Step 4's quant lever.

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

1. **Run I is done; the setting to keep is bias on, tier off.** `rtx.local.json` already carries
   `cache_affinity_spread_file`, `LLAMADECODE_HOST_CACHE_MB: "0"` and `LLAMADECODE_L2_MAX_SLOTS:
   "1500"`, and Step 4's first item (the bias applies at a nonzero value with no refusal) is closed
   by F, G and H. Any further decode measurement should therefore be `~/bin/ds4-first-run --bias`;
   a `--bias`-less run is the control and should be used only when a comparison needs it.
2. **Step 5, next on the critical path.** With p10 matching Strata and p90 3.5× worse, the
   remaining 2.4× on the mean is a tail problem, and the drafter is how the reference implementation
   gets under it. `scripts/ds4_dspark_probe.sh` measures the native `--dspark --mtp-model` path
   against the repacked sidecar, twice (`--no-draft` for the comparison), and prints acceptance rate
   beside tok/s. Only after that number exists does the question of extending `ds4x`'s producer ABI
   with a verify-shaped entry make sense - the producer currently has no drafter entry at all.
3. **Step 6 is measured** (see the Step 6 section): pi's first real session reused 74-99 % of each
   turn's prompt through one producer. What is left there is not reuse but cost - the append path
   spends 6-34 s per turn whatever the new-token count, so turn 5 paid ~28 s to prefill 1 598
   tokens. Measure the append's fixed cost (tile sweep vs row count) before Step 4's quant lever;
   that is the difference between a 20-76 s agent turn and a usable one.

### I (21:16): the control, and the attribution is clean

`bias: this run sends quality_speed=0 on every request; persisted control is value=0.5 revision=14`
then `LLAMADECODE_LIVE_BIAS value=0`, `DS4X_AFFINITY mean=0 … enabled=0` - the control is a control.
Same device plan as G and H (tier off, L2 1500, graph cap 5.5), 255 steps completed:

| run | bias | tier | steps | median | tok/s median | mean | hit | from_file | copies |
|---|---|---|---|---|---|---|---|---|---|
| E | 0 | 12.9 GiB | 255 | 151.7 ms | 6.59 | 4.84 | 77.0 % | 23.0 % | 3.953 GiB |
| **I** | **0** | **off** | 255 | **149.2 ms** | **6.70** | 4.83 | **75.7 %** | 24.3 % | **2.073 GiB** |
| G | 0.5 | off | 255 | 50.4 ms | 19.83 | 9.78 | 93.4 % | 6.6 % | 1.693 GiB |
| H | 0.5 | off | 255 | 46.9 ms | 21.31 | 10.22 | 93.0 % | 7.0 % | 1.727 GiB |

So, one variable at a time:

- **The RAM tier is neutral-to-negative.** I against E: +0.11 tok/s median, hit rate down 1.3
  points, and 1.88 GiB/step of extra device-to-device traffic for the tier's presence. It served
  1.4 % of E's bytes and bought nothing measurable. Keep `LLAMADECODE_HOST_CACHE_MB` off.
- **The affinity bias is the lever.** 149.2 ms → 46.9-50.4 ms median, 3.0-3.2×, and the hit rate
  goes 75.7 % → 93.0-93.4 % - Step 2's 90 % bar is met, but by bias, not by a host cache. The bias
  re-ranks experts so the hot set stops reaching past VRAM: from-file falls 24.3 % → 6.6 %.

One caveat on what "bias 0.5" means here. `quality_speed` publishes a **scalar** -
`runtime/generation/live.json` went to `per_phase: false, value: 0.5` - so every phase got 0.5,
not the five-point profile that `generation.defaults()` publishes (`reasoning 1.36125`,
`answer 0.35`). The 3× is a flat-bias measurement; a per-phase run is a separate setting and will
need its own number.

Step 2's accept bar as written (hit ≥ 90 % from the tier) is the wrong instrument on this box; the
finding that replaces it is this table. Prefill reproduced a sixth time at 847.2 tok/s, and this
run's long-prefill request **succeeded** where G's and H's got 503s, so that refusal is not
systematic - the probe prints the body now if it returns.

### What the remaining gap actually is: the tail, not the median

Strata's baseline is `median 23.9 tok/s, mean 24.2, p10 21.4, p90 27.2` over 67 runs ≥ 200 tokens -
median and mean within 1.3 % of each other because its distribution is tight. Ours at bias 0.5:

| | p10 step | median step | p90 step | median tok/s | mean tok/s |
|---|---|---|---|---|---|
| Strata | 21.4 ms | 41.8 ms | 27.2 ms | 23.9 | 24.2 |
| G (bias 0.5) | 25.2 ms | 50.4 ms | 94.8 ms | 19.83 | 9.78 |
| H (bias 0.5) | 25.2 ms | 46.9 ms | 92.8 ms | 21.31 | 10.22 |

The p10 already matches Strata's - 25.2 ms against 21.4 ms, so a step whose experts are resident
runs at the same speed. The gap is entirely the tail: our p90 is 94.8 ms against 27.2 ms, 3.5×, and
that is what drags the mean to 10.2 against 24.2. A user reading a 500-token answer experiences the
mean, so the honest distance is still **2.4×**, not 12 %.

The tail is steps that pay NVMe again - 6.6 % of bytes on average, concentrated rather than spread.
Two things attack it, and neither is cache sizing: Step 5's drafter, which removes full steps from
the critical path (Strata's own note in this plan says it clears 20 tok/s "with a smaller step and a
drafter"), and whatever makes a miss cost less than the 2.22 GiB footprint it currently costs. That
is the ordering from here.
3. **Then Step 6**, against the numbers above rather than the arithmetic here:
   `scripts/dsv41_reuse_probe.py` is the multi-turn reuse probe (it reads `timings.cache_n` on the
   second turn and marks PASS/FAIL against the 70 % bar). It cannot be read yet - A and B each put
   every request in its own producer, and all three reported `cached_tokens=0`. With a run whose
   producers survive, `cache_n=0` on turn 2 means the append path refused, which is the measurement
   Step 6 wants. The host-tier sizing is item 1's `offer` line, no longer a derived guess.
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
