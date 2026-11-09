# How far is a first engine run on this box?

Measured on 2026-11-07 against the `linux-only` branch at `bc45882`. This is a status
snapshot, not a durable contract: the VRAM and disk figures are this box's right now, so
re-measure before acting on them. The durable port state lives in `HANDOFF.md` and
`docs/linux-port-plan.md`; the commands live in `docs/linux-port-commands.md`.

## What is already true

| Check | Result |
|---|---|
| `./bin/slstack --config config.example.json check` | 13 rows. One **required** failure: `model` - "required model path missing". Everything else passes, including `ds4_executable`, `llama_producer`, `host_memory`, `vram`, `journal` |
| `core_kv_residency` | fails, but `required: false` - the residency/paging reconciliation row Phase 8 owns |
| Built binaries | `engines/ds4/{ds4,ds4-server,ds4-bench,ds4-eval,ds4-agent,ds4-decode.so}` and `build/llama/bin/{llama-dsv41-prefill,llama-server,libslstack-weight-fetcher.so,ds4-decode.so}`; `ds4 --help` prints, `scripts/start-dsv41-single-262k.sh --dry-run` resolves |
| Decode library placement | Single-engine serving requires `ds4-decode.so` **beside the producer**, not only under `engines/ds4/` - `slstack/single_runtime.py` checks `producer.parent`, and the producer `dlopen`s it from there. `./build.sh <config> llama` now stages the copy; a stack built before that change needs `cp engines/ds4/ds4-decode.so build/llama/bin/` |
| Host | 93 GB RAM, ~57 GB available; ext4 NVMe, 323.4 GiB free (`statvfs` `f_bavail`) |

Nothing in the streamed-prefill path refuses on Linux any more. That is what chunks
3.1a through 3.1h and 5.1 through 5.3 landed, each with its own gate.

## The model file: one public file qualifies, and it is verified

`antirez/deepseek-v4.1-flash-gguf` carries
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, 365,713,686,528 bytes =
**340.6 GiB**. Reading only its header (a 16 MiB ranged fetch) and running this repo's own
`slstack/gguf_header.py` over the complete header - 5.8 MB, 1,046 tensors - gives:

```
general.architecture = deepseek41
split.count          = absent (validator treats it as 1)
deepseek41.num_hidden_layers = 40
ffn_gate_exps / ffn_up_exps = IQ2_XXS, ffn_down_exps = Q2_K   -> accepted triplet
dense, output, attn_q_a, attn_kv = Q8_0
token_embd = F16, engram_embd = I8 [264, 384006168]
deepseek41.engram.encoding = e4m3_e8m0_32_row264
validate_dsv41_types: accepted
```

That is the file type `docs/linux-port-plan.md` says this box has never had. The Engram
tables are inline with the 264-byte row stride that
`engines/streaming-llama/prefill_llama.cpp/src/llama-model.cpp:1953` and
`engines/ds4/ds4.c:2624` require, so `tests/dsv41_engram_early_io.cpp`'s row loader has
something real to read.

By contrast, every file currently on this box refuses. Re-checked now with the same
validator:

| On disk | Refused because |
|---|---|
| `Qwen3.8-Flash-Next-Gyro-{S,M,L}-TQ*.gguf` | `general.architecture = qwen4exp`, expert tensors at type ids 144+ - outside the engine's type table |
| `deepseek-v4.1-flash-IST-DAS-2.00bpw/…-0000N-of-00004.gguf` | `split.count = 4`, and the per-block triplet is `IQ1_M/IQ1_M/IQ1_M`; merging would not qualify the quant set |
| `deepseek-v4.1-flash-shared/…UD-Q8_K_XL-00001-of-00009.gguf` | 1 of 9 shards, 5.7 MB, **zero tensors**; `Q8_K` is not in the accepted expert set either |
| `deepseek-v4.1-flash-shared/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | `general.architecture = deepseek41-engram`, 2 tensors - the llama.cpp Engram representation, not ds4's row format |

## Two things stand between here and a run

1. **Disk.** 340.6 GiB needed, 323.4 GiB available. Removing the unusable
   `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (103 GiB) or the IST-DAS shard set (126 GiB)
   clears it. Both are files the engine refuses; that is the user's call, not this
   document's recommendation.
2. **VRAM.** Strata's serve process holds ~29.8 GiB of 32 GiB and the browser GPU process
   ~98 MiB; `docs/evidence/cuda-headroom-probe.cpp` reported 360 MiB free at the time of
   writing. The presets assume the card is nearly whole (`LLAMADECODE_VRAM_RESERVE_MIB=128`,
   `LLAMADECODE_VRAM_MARGIN_MIB=64`, and `producer_graph_floor_gib: 3.0` on the 1M preset).
   Those two VRAM numbers are pinned by `COMMON_ENV` in `scripts/dsv41_single.py`, applied with
   `env.update(COMMON_ENV)` **after** the config's `env_overrides`, so a local config cannot raise
   them and a value set there is dropped without a word. `DEFAULT_ENV` below it uses `setdefault`,
   so every other key does take. To give the decode device slack, use the ungated
   `LLAMADECODE_L2_MAX_SLOTS` instead. Measured 2026-11-09: a run asking for
   `LLAMADECODE_VRAM_MARGIN_MIB: "1024"` sized its L2 with the default 64, and
   `dsv41_single.overlay('262k', rtx.local.json)` prints the merged 64.
   Stopping Strata also stops the assistant, so GPU work is the user's to run by hand.
   Admission itself gates on the device, not on which process holds it: `slstack/lease.py:210`
   refuses when `memory.used > gpu_idle_max_mib` or `utilization.gpu > 5`. With the serve
   process stopped this desktop still reports ~1,864 MiB used - the compositor and browser
   GPU buffers, most of it not attributed to any compute client - so the example config's
   `gpu_idle_max_mib: 1024` cannot admit here. `rtx.local.json` sets 2560, which still
   refuses a real model serve (29 GiB) and still fails while one is decoding (utilisation
   was measured at 94%).
   A second refusal is not about the card at all: a producer that is `kill -9`'d leaves an
   `ACTIVE` line in `runtime/state/gpu.holds` that never expires by inference, and the next
   admission fails with `RTX admission refused: ACTIVE holds exist: …` even on a quiet GPU
   (measured 2026-11-09 21:28, after a run whose producer was force-killed). Release it with
   `python3 -m slstack --config rtx.local.json release-holds`, which takes only holds whose
   guardian process is gone and refuses while the card is busy - so free the card first.
   `scripts/start-dsv41-single.sh` now runs that before starting; `~/bin/ds4-first-run` already
   did.

Download rate measured against that repository with a 64 MiB ranged request: 63.6 MB/s,
so ~96 minutes for the file.

## When both are cleared

```sh
# resolve the start config and print it, start nothing
./scripts/start-dsv41-single-262k.sh --model <path>/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --dry-run

# header-only qualification before launching CUDA
python3 -B -c "import sys; sys.path.insert(0,'.'); from slstack.gguf_header import read_header, validate_dsv41_types as v; v(read_header('<path>')); print('accepted')"

./bin/slstack --config config.example.json check          # the `model` row must flip to ok
./scripts/start-dsv41-single-262k.sh --model <same-path>
```

The launcher stays **attached** after the router answers, so `Ctrl+C` is the shutdown: it asks the
stack to retire gracefully, waits for the start process to exit, force-kills a producer that cannot
honour the stop (`wait_layer_gpu` blocks cooperative retirement), clears any process still holding
the router or engine port, and releases the GPU hold. It does the same clearing *before* starting:
`--stop` returns `{"stopping": true}` without guaranteeing the engine worker dies, and a worker
still holding 8089 makes the next start answer `[Errno 98] Address already in use` on every request
(measured 2026-11-09 21:34). The server itself runs as a detached child, so closing the terminal
without signalling leaves it serving; interrupting the launcher does not.
`--detach` returns immediately and keeps the old behaviour, where stopping means
`scripts/start-dsv41-single.sh --preset 262k --stop`. The interrupt trap is installed before the
readiness wait, because a stack that never finished starting used to leave its worker and producer
holding the card with nothing attached to stop them.

The launcher is the only route that serves this file on one 32 GB card. The native
`ds4-server` backend refuses a V4.1 GGUF larger than VRAM both ways: without
`--ssd-streaming` the admission check demands the mapped region resident
("V4.1 needs 156.54 GiB before the expert cache", `ds4.c:72498`), and with it the layer
packer places experts-bearing layers on CPU, so `multi_tier` is set and the flag is
refused outright (`ds4.c:73269`). `--ssd-streaming` is the GLM/Metal branch on this fork
(`ds4.c:17845`), and the single-engine decode path sets it to false anyway
(`ds4_decode_adapter.inc:169`); it is therefore no longer in `config.example.json`.

The first request starts the producer on the GPU lease, so nothing else may hold the
card: `gpu_idle_max_mib` is 1024 and a serve process using 29 GiB fails it.

### The plan gate

Before the producer is launched at all, `slstack/combo_producer.py` (`device_gate_bytes`, and the
same check in `native_broker.py`) requires free VRAM covering the plan's **device bill** plus
`--gpu-reserve-bytes`: provider + cache + graph + hidden + pinned. The delivered ULTRA bill is
29,090 MiB + 1,024 MiB = **30,114 MiB**, and the producer itself refuses a reserve under 1 GiB
(`tools/dsv41-prefill/main.cpp:459`), so the plan expects a nearly whole card -
which is what the reference machine was, with Studio paused and no browser GPU process.
On this desktop the Wayland compositor, Xwayland, `vicinae-server` and the terminal hold
about 1.4 GiB even with every browser closed, and `nvidia-smi` reports roughly 400 MiB
beyond `used + free` as never available, so the best gap measured 30,834 MiB free.

`--job-device-bytes` is only a ceiling: the producer admits `min(job, free - reserve)` against
the same bill (`src/llama-dsv41-device-admission.h:29`). Charging the ceiling instead of the bill
asked for 30,948 MiB - 834 MiB more than the plan needs - and a graph clamp that lowered the
ceiling bought gate slack without changing what the producer admits.

`producer_graph_cap_gib` clamps only `--graph-bytes`. It never goes below
`producer_graph_floor_gib` (5 GiB default), and the measured planned graph peak is 3.71 GiB at
100K, so 5.5 GiB keeps about 1.9 GiB of margin over the peak; the producer still refuses loudly
in its own admission if that stops being true. At that cap the gate needs 28,598 MiB at a short
prompt and 29,858 MiB at 100K, against 30,716 MiB free; `rtx.local.json` sets it. The cache and
device bills are unchanged, so the context capacity is the delivered one.

The host side needs the two admissions the fold planner does not model, and they are stated in
`rtx.local.json`. The weight provider charges its own host plan - the relay's pinned staging tier,
one whole compute bank per slot, plus fetch request metadata (`src/llama-dsv41-weights.cpp:921-926`)
- which measured 22,132 MiB on this model against a delivered `--provider-host-bytes` of 2,654 MiB;
`producer_provider_host_gib` (24) admits it. `producer_job_host_gib` (30) raises `--job-host-bytes`,
which is what the sweep's combined host check measures against (`src/llama-dsv41-sweep.cpp:456,497`;
it measured 25,062 MiB against a delivered 4,352 MiB). Neither key sizes anything: carry capacity,
sweeps, the context cache and the Engram row cache (explicitly `LLAMADECODE_ENGRAM_CACHE_MB=0`) are
unchanged, and the provider refuses loudly if its plan exceeds the admission. `host_budget_gib` (8 ->
31.5) has to cover the job host bill plus the controller margin, because the guardian's cgroup scope
is the aggregate resident admission.

The 262k preset is the exact-cone one (`yoco_keep 2541`, `ctx_cache_precision f32`);
`start-dsv41-single-1m.sh` is the 1M context route and `--keep 140` is the approximate
YOCO cone - see `examples/single-machine/README.md` for what each name actually means.

## What the first run will report, and what it will still leave unverified

- `CARRY_RAM_LOCK_PARTIAL` in the carry tier, and `DS4COMBO_LOCKED_RAM_PARTIAL` for any
  checkpoint tier above 8 MiB: `RLIMIT_MEMLOCK` is 8 MiB here, so the paging guard, not the
  page lock, is what keeps pages resident.
- `DS4COMBO_RESIDENT_FLOOR_INERT` on the combo path: `memory.min` is written and read back,
  but the kernel zeroes it because no ancestor of the scope carries a floor.
- `producer_prefill_sweep_tokens: auto` resolves against geometry defaults in `choose_scan()`
  that were tuned on a Windows RAID-0 at ~25 GB/s. This NVMe is a single device. The first
  real run is the measurement of those defaults; nothing here has re-measured them.
- A plan whose host tier exceeds `host_budget_gib` is now refused by the kernel rather than
  reported by the RSS poll, because the guardian caps the tree in a cgroup v2 scope.
- Rates in `docs/PERFORMANCE.md` and `README.md` stay inherited from the Windows build.
  Chunk 8.1's qualification (three runs at the exact cone, the 200K fixture, the Engram
  extent comparison against `pread`, and `test_yoco`) is what replaces them, and it starts
  only once a qualified file exists.
