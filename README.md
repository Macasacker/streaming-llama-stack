# Streaming Llama Stack

Run frontier-size mixture-of-experts models on one desktop GPU. The stack streams
the weights from NVMe into an RTX 5090 over io_uring, so a 483 GiB model runs on a
machine with 32 GB of VRAM and 32 GB of system RAM.

It ships a llama.cpp fork with streaming prefill, the ds4 decode kernels for
DeepSeek V4.1 Flash, an OpenAI-compatible API, MCP tools and the llama.cpp web UI.

## Results

DeepSeek V4.1 Flash on one RTX 5090 desktop with 32 GB of RAM. Every figure below
is **real text**: *I Promessi
Sposi* with a password at the start and a recall question at the end, 199,952 tokens
(the 100K rows use its first 100,000 tokens). Every run below answered the password
correctly. Logits are bit-identical to the
previous build. Earlier README figures (for example 5,240 tok/s) were measured on
100,000 *identical* tokens. That file needs about 1,200 Engram rows per table, against
1.66M on real text, so it hid about 37 s of Engram and spill cost at 200K. Those
figures are retired.

These figures are inherited: they were measured on the upstream Windows build,
which read weights through DirectStorage. The Linux fork in this repository reads
them through io_uring and has re-measured none of them. No model file in the
accepted quant set has been opened here yet, so every number needs re-qualification
against the same hash-pinned fixtures before it can be restated. See
[linux-port.md](linux-port.md) and chunk 8.1 of
[docs/linux-port-plan.md](docs/linux-port-plan.md).

### Single machine: RTX 5090

| Model file | Prompt | Prefill (prompt clock) | Launch to first token | Decode, Max quality | Decode, Max speed |
|---|---:|---:|---:|---:|---:|
| Packed FP4 (tensor-core-native layout) | 200K | **43.2 s = 4,632 tok/s** | 47.2 s | **7.6-7.9 tok/s** | **22.7 tok/s** |
| Packed FP4 | 100K | **25.5 s = 3,917 tok/s** | 28.7 s | - | - |
| Q4_K/Q8_0 | 200K | 59.6 s = 3,356 tok/s | 63.9 s | 7.5 tok/s | 17.8 tok/s |

Before this release the same FP4 200K prompt took 76.2 s (2,623 tok/s), and 100K took
40.3 s. Three changes made the difference:
- Engram rows are read as 1 MiB extents through one queue at QD64, instead of about 3M
  random 4-8 KiB reads per sweep. This volume caps 4 KiB random reads at ~280K IOPS.
- One early load at open holds every sweep's rows.
- The 1M-context hidden bank holds the whole sweep, so no rows spill (graph floor 3.0 GiB).

The remaining gap to the synthetic figure is real-text expert diversity in the
executor (31.1 s against 29.3 s), the tail weight stream (6.8 s) and an Engram wait
(about 2.5 s). Q4 Max speed was measured with the owner's desktop apps open. The
planner then refused the whole-prompt Engram admission (9.37 GB needed, 9.30 GB room),
so each sweep loaded its own rows: still exact, but slower.

### Presets and scripts (fastest defaults)

- `python -B examples/run_preset.py --single ultra-cone --context 1m --config my.local.json`
  (packed FP4 file in your config; Max quality or Max speed is the live quality-speed
  control).
- `scripts/start-dsv41-single-262k.sh --webui`, or one of the other labelled launchers
  under `scripts/`.

The host-RAM expert tier and the CPU expert lane stay opt-in toggles.

[Full numbers, with the conditions for each](docs/PERFORMANCE.md). Kimi, GLM, MiniMax
and Step streamed their prefill on the RTX and decoded on a Mac. That route went with
the Mac half; their rates stay in `docs/PERFORMANCE.md` as history, not as something
this fork can run.

### Host-RAM expert tier

The host-RAM tier stays **off by default** (`LLAMADECODE_HOST_CACHE_MB=0`). Round 4
(2026-10-30) moved its data movement off the critical path: the VRAM-to-RAM copy-in
runs on its own stream with one event, the storage batch is submitted first and
RAM hits upload on a separate H2D stream while it reads, and Max-speed routing prices
a RAM expert at half a miss (`LLAMADECODE_RAM_ROUTE_COST=0.5`). On this 32 GB box the
tier is a small gain only at FP4 Max quality and still a loss elsewhere: the measured
build could not CUDA-register the locked tier in-process, so each hit is copied through
pinned staging (about 1 ms of CPU per expert), and a 6.6 GB tier holds 351 experts,
removing only 6% of misses.

Packed FP4 unless marked Q4, 899 warm steps, VRAM headroom 128 + 64 MiB, 6.6 GB tier:

| Context, endpoint | Tier off | Tier on |
|---|---:|---:|
| 50K Max quality | 7.30 tok/s | 7.27 / 7.39 tok/s |
| 100K Max quality | 7.36 tok/s | 7.48 tok/s |
| 50K Max speed | 23.88 tok/s | 22.94 tok/s |
| 100K Max speed | 23.27 tok/s | 22.71 tok/s |
| Q4 50K Max quality | 7.26 tok/s | 6.94 tok/s |
| Q4 100K Max quality | 7.34 tok/s | 6.97 tok/s |

The earlier synchronous tier (`LLAMADECODE_HOST_ASYNC=0`) measured 7.06 tok/s on the
same 50K Max quality run. Max-speed routing that counts RAM as VRAM
(`LLAMADECODE_RAM_ROUTE_COST=0`) measured 21.58 tok/s against 22.85 at 0.5.
Free-RAM sizing has a practical limit: an 8.7 GB tier left 2.2 GB available and the
pagefile guard stopped the run.

**Decode VRAM headroom** is now 128 MiB reserve plus a 64 MiB L2 growth margin
(was 512 + 768 MiB), about 60 more cached experts. 128 MiB with no margin breached
the reserve in every run. If the planned L2 allocation is refused, it steps down
256 MiB at a time (`LLAMADECODE_L2_ALLOC_REFUSED`, at most 2 GiB) and the reserve
guard still refuses on a real shortfall. Packed FP4 Max quality with the new headroom:
50K 7.05-7.30, 100K 7.20-7.36, 200K 8.29-8.83 tok/s.

It is expected to matter more on machines with much more RAM, or where the tier can be
CUDA-registered; that is untested and listed in the
[roadmap](docs/ROADMAP.md).

## How it works

**Streaming prefill.** Prefill reads each layer's weights once per pass and runs
the whole prompt chunk through that layer before moving on. The cost of reading
the weights is shared by every token in the pass. A complete DeepSeek V4.1 Q4
pass reads 315.8 GB; Engram rows are gathered separately. The storage layer reads the
next layer while the GPU computes the current one. DeepSeek V4.1 uses
49,152-token passes, with optional RAM-admitted chunk grouping.
[Routed reads, the exact replay cone and byte accounting](docs/MULTICHUNK_PREFILL.md).

**Expert-cache decode.** Dense weights stay in VRAM. Routed experts live in a VRAM
cache, and misses are read from NVMe. After the first hundred or so tokens the
cache holds the experts the conversation uses, and decode speeds up.

**Quality/speed dial.** The faster settings prefer experts that are already in the
cache instead of fetching from disk. Max quality never does this: it is the full
model. You set the dial from the web UI, the API or MCP, separately for reasoning,
tool calls, answers and code. The model can set it itself.

**Exact cone.** The ULTRA defaults compute every encoder row and the decoder's final
2,541-row dependency cone. Aligned replay preserves full-prefill arithmetic; the
matched 8K and 49,555-token controls are exact. The state protocol that the native
broker's export and import use keeps its version negotiation. [Design](docs/MULTICHUNK_PREFILL.md) ·
[Cone state protocol](docs/CONE_STATE.md).

**YOCO.** Optional approximate prefill from DeepSeek's
[DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
release, keeping the last 140 positions. Our streaming variant skips the reads and
compute of the decoder layers (21–39) on every pass except the last. It speeds up
long prefills by 1.26–1.45× and changes outputs slightly.
[Speed and quality](docs/ULTRAYOCO_QUALIFICATION.md).

The two-machine route is gone. The RTX did the prefill and a Mac Studio decoded, and
the receiver for that handoff was macOS-only; this fork serves one machine.

Streaming prefill is general across the architectures the llama fork builds, but the
serving half of this stack is the ds4-kernel decode of DeepSeek V4.x, and the launcher
refuses a `general` or `mac` config.

## Who can run this

| Mode | Hardware needed |
|---|---|
| Single RTX | Linux RTX 50-series desktop, 32 GB VRAM, 32 GB system RAM and fast NVMe; the inherited measurements were taken on an RTX 5090. |

## Requirements

- **GPU:** RTX 50-series (compute capability 12.0). Measured on an RTX 5090, 32 GB.
- **Storage:** fast local NVMe. Measured on a NVMe RAID-0 at about 25 GB/s. Storage
  speed sets prefill and cold decode speed. GPUDirect Storage needs the `nvidia-fs`
  driver and XFS or EXT4 in ordered mode; io_uring works without it.
- **System RAM:** 32 GB. The engine needs its 9.25 GiB budget plus a 512 MiB
  desktop reserve free, and stops cleanly if RAM drops below 0.5 GiB free.
- **Build tools:** Linux, Python 3.12, a C++ toolchain, CMake, Ninja, CUDA 13.2 or
  newer with `sm_120a`, liburing, and Node.js 24.

## Any-GGUF single engine

A native DeepSeek V4.1 Flash GGUF using supported tensor types:
Q4_K, IQ2_XXS/Q2_K or MXFP4 routed experts, with Q8_0, Q4_K or Q4_0 dense
matrices. The measured file uses Q4_K experts and Q8_0 dense matrices (483 GiB).
Our Q4_K/Q8_0 build is GPU-tested; IQ2_XXS/Q2_K remains CPU-tested.
MXFP4 has measured GPU prefill results with a separate tensor-core-native
packed layout. An experimental dense FP8 path is also available.
[FP4/FP8 layouts and measurements](docs/PERFORMANCE.md#fp4-and-experimental-fp8). Public MXFP4 shards must first be merged with antirez's
`merge_deepseek41_shards.py` tool. “Any-GGUF” covers these supported V4.1 Flash
layouts and geometry, rather than arbitrary GGUF architectures.
[Layouts and assembly instructions](docs/DSV41_GGUF.md) and the
[conversion script](scripts/convert_dsv41_gguf.py) cover existing safetensors,
with a header-only dry run. The installer never downloads weights.

## Making the model files

DeepSeek V4.1 Flash uses GGUF files produced from the original
[DeepSeek-V4.1-Flash-1017 safetensors checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
(revision `1bae0e10…`, about 475 GiB) with `scripts/convert_dsv41_gguf.py`. The FP4
file keeps DeepSeek's released E2M1/E8M0 expert values in the tensor-core-native byte
layout, `deepseek41.expert_layout = flashinfer-sm120-mxfp8-mxfp4-v1`. The converter
still writes the canonical `ggml-mxfp4` layout when asked, but nothing in this fork
loads it: the engine that did was the Mac one.

Status: this native safetensors path has not yet been run end to end and qualified on this stack.

### Which file is which

| File | Purpose | Runs on | Preset or script | Command (after the plan step below) |
|---|---|---|---|---|
| **Packed FP4** | Streaming prefill and ds4 decode | Linux RTX 5090 | `python -B examples/run_preset.py --single ultra-cone`; `--model` of `scripts/start-dsv41-single-262k.sh` | `--expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8` |
| **DSpark drafter** | Speculative drafts for ds4 decode | Linux RTX 5090 | `dspark_model` in local config, or `--dspark` on the native server | `--component dspark` (Q4_K experts, Q8_0 attention/shared/projections; no `--expert-format`/`--expert-layout`) |
| **Q4** | The Q4_K/Q8_0 alternative | Linux RTX 5090 | `scripts/start-dsv41-single-262k.sh` / `-1m.sh` with the Q4 path in your config | `--expert-format q4` (the default) |

Rule of thumb: **MXFP4 experts with the layout flag are the packed FP4 file; Q4_K
experts are Q4.** You can check a file without reading its payload:

```sh
python -B -c "import sys; sys.path.insert(0,'.'); from slstack.gguf_header import read_header, TYPE_NAMES; h=read_header(sys.argv[1]); m=h['metadata']; t=[x for x in h['tensors'] if x['name'].endswith('_exps.weight')][0]; print('experts', TYPE_NAMES.get(t['type']), '| layout', m.get('deepseek41.expert_layout', 'none = ggml-mxfp4 if MXFP4'))" <file.gguf>
```

### Commands

Inputs: the downloaded checkpoint directory (`--source`; nothing downloads weights)
and a directory for the pinned antirez/ds4 `gguf-tools` sources (`--ds4-tools`).
`--fetch-tools` fills that directory with the hash-pinned source files only, never
weights. `--execute` also needs `--out`, `--host-budget-gib` and, for main and
DSpark, `--quants-library`: a built `libds4quants` (`make -C <ds4-tools>
quants-shared`, C compiler and pthreads required).

**1. Plan** (opens only `config.json` and the index, never the shards; `--plan` must be a new file):

```sh
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> --fetch-tools \
  --expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8 \
  --dry-run --plan <new-plan.json>
```

**2. Convert**, with the same file flags as the plan:

```sh
# Packed FP4
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --expert-format native-fp4 --expert-layout flashinfer-sm120-mxfp8-mxfp4-v1 --dense-format q8 \
  --execute --out <fp4.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8

# DSpark drafter
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --component dspark --execute --out <dspark.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8

# Q4
python -B scripts/convert_dsv41_gguf.py --source <HF checkpoint dir> --ds4-tools <ds4-tools> \
  --expert-format q4 --execute --out <q4.gguf> --quants-library <libds4quants> --threads 8 --host-budget-gib 8
```

- `--threads` is 1..32. `--host-budget-gib` is required with `--execute` and must
  fit in physical RAM above `--runtime-floor-gib` (default and minimum 4).
- `--resume` continues an interrupted run from its journal. Existing outputs are
  never overwritten. Use a distinct `--out` for each file; logs and the memory
  receipt go beside it (or under `--runtime-root`).
- The header is checked against the plan before success is reported.

### Disk and memory

| File | Output size | Free disk needed | Host RAM |
|---|---:|---|---|
| Packed FP4 (`q8` dense) | ~502 GB | output + 32 GiB reserve, checked by the writer | `--host-budget-gib` |
| DSpark drafter | ~7.8 GiB | output + 32 GiB reserve | `--host-budget-gib` |
| Q4 | ~483 GiB | output + 32 GiB reserve | `--host-budget-gib` |

The ~475 GiB checkpoint stays read-only and is needed in addition to the outputs.
Conversion is CPU-only.

## Quickstart

```sh
git clone https://github.com/fronteagle739/streaming-llama-stack.git
cd streaming-llama-stack
./bin/slstack install --all
./bin/slstack install --all --check
```

Create `my.local.json` from [config.example.json](config.example.json) with your
model path and GPU UUID. `*.local.*` files are ignored by Git.

```json
{
  "model": "<model_dir>/<deepseek-v4.1-flash>.gguf",
  "gpu_uuid": "<GPU-UUID>",
  "host_budget_gib": 9.25,
  "min_available_gib": 0.5,
  "runtime_min_free_gib": 0.5
}
```

```sh
./bin/slstack --config my.local.json check
scripts/start-dsv41-single-262k.sh --webui
```

Open the printed URL (default `http://127.0.0.1:8088`). The same port serves
`/v1/chat/completions`, `/v1/models`, `/health` and `/mcp`. The first check hashes
the model once. The first chat loads the engine.

Other presets (run one at a time; stop with the same script and `--stop`):

```sh
scripts/start-dsv41-single-1m.sh --webui               # 1M context, exact cone
scripts/start-dsv41-single-262k-ultra-yoco.sh --webui  # 262K, approximate YOCO keep 140
scripts/start-dsv41-single-1m-ultra-yoco.sh --webui    # 1M, approximate YOCO keep 140
```

Each script accepts `--config`, `--model`, `--webui`, `--vision`, `--dry-run` and
`--stop`. The `-ultra-yoco` filenames are approximate replay (keep 140); the plain ones
are the exact cone (keep 2541), which is also the native broker's negotiated state v2
default. [Defaults and optional experiments](docs/MERGEPERF.md).

## Supported models

| Model | RTX streaming prefill | Decode |
|---|---|---|
| DeepSeek V4.1 Flash | Yes | ds4 kernels on the RTX |

Nothing else is qualified. The llama fork builds the other MoE architectures for
prefill and `ds4` reads `glm-dsa`, `glm5-next` and `qwen4exp` metadata, but no run
here has loaded a model outside DeepSeek V4.1 Flash, and Kimi K2.5/K2.6/K2.7,
GLM-5.x, MiniMax M2.5/M3, Step-3.5/3.7 and MiMo V2.6 Pro decoded on macOS through
receivers this fork deleted. The launcher refuses a `general` config. `llama-server`
still serves any architecture llama.cpp understands, with its own prefill and no
weight streaming. [Model notes and measured speeds](docs/SUPPORTED_MODELS.md).

## Features

- **Web UI** with live prefill progress, decode speed, the quality/speed slider and
  per-phase settings. [Live controls](docs/LIVE_GENERATION.md)
- **Persistent sessions.** The conversation stays loaded between turns, and
  follow-ups only process new tokens. [Prefix reuse](docs/PREFIX_REUSE.md)
- **Images** in prompts.
- **MCP** endpoint with the quality/speed controls and media tools.
- **Media generation (experimental):** Anima, Qwen-Image, Krea and H3 image/video
  from the chat. No media run has been made on this box: the setup check verifies the
  system FFmpeg and the pipeline imports, and the streamed weight transport refuses on
  Linux. [Media setup](docs/MEDIA_SETUP_VERIFICATION.md)
- **LoRA (experimental)** for the native ds4 path, with frozen-prefix training.
  [Example](examples/train_jeff_lora/README.md)
- **Configuration reference:** [Variables](docs/VARIABLES.md),
  [walkthrough](examples/WALKTHROUGH.md), [roadmap](docs/ROADMAP.md).

## Limitations

- CUDA kernels target compute capability 12.0 (RTX 50-series).
- Nothing in this fork has run against model weights. The streaming producer is
  ported (provider lease, io_uring fetcher, Engram extent reader, carry tier), but
  no GGUF in the [accepted layouts](docs/DSV41_GGUF.md) is on this box, so the
  inherited figures above stay inherited and the port's gates are host-side ones.
  See [docs/linux-port-plan.md](docs/linux-port-plan.md).
- One machine only. Two-machine pairing was deleted with macOS, so there is no
  decode host for a model too large for the RTX.
- The single-engine DeepSeek V4.1 path accepts the [supported GGUF layouts](docs/DSV41_GGUF.md); split files require assembly first.
- The faster dial settings cost quality. Max speed introduced syntax errors in C++
  tests; use Max quality or Balanced for code.
- Approximate YOCO140 is opt-in; the default dependency cone keeps 2,541 rows.
- The matched eager control uses the Q4 shipping-default Q8 dense path with
  stored FP4 experts; the earlier KERNELS2 row measures the full Q4_K/Q8_0 file.
- The latest 100K FP4 run uses experimental prefill graphs and does not establish
  a repeatable graph speedup. No 50K or 200K run on that same final build is recorded.
- The 1M result and the 49,555-token warm-decode controls use earlier YOCO140
  prefill defaults. Warm decode has not been remeasured on the final OTHER22 build.
- FP4 prefill timing does not qualify decode or broad recall quality. Dense FP8
  remains experimental and has no demonstrated prompt gain.
- TileLang attention is the selected default but is numerically different from
  native attention. The tested retrieval answer matches; later forced continuations
  can diverge. Set `LLAMA_DSV41_TILELANG_ATTENTION=0` for native attention.
  Mode 2 now runs a hand-written SM120 kernel with the same arithmetic in a
  different summation order (`LLAMA_DSV41_SPARSE_HAND=0` restores the TileLang image).
  [Quality measurements](examples/kernels2/PERFORMANCE.md).
- The first prompt includes engine startup, and decode is slower until the expert
  cache warms up (about 100 tokens).
- The separate host-RAM tier has no consistent gain across dial settings.
  CPU/mix continuations can differ and fail strict sequence-equivalence checks.
- Media generation and LoRA are experimental.

## Credits and licence

MIT licence for the stack code ([LICENSE](LICENSE)). Embedded projects keep their
own licences; see [NOTICE](NOTICE).

- [DeepSeek](https://github.com/deepseek-ai) for the model, and
  [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
  for YOCO. TensorSharp was a secondary reference.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for the inference base and web UI.
- [antirez/ds4](https://github.com/antirez/ds4) and
  [ds4-win-NovaStar](https://github.com/mishal-harbi/ds4-win-NovaStar)
  for the native kernels.
- [NVIDIA GPUDirect Storage](https://github.com/NVIDIA/gds-nvidia-fs) for
  storage-to-GPU transfers, and Linux io_uring for the read path.

<sub>Useful to you? [Ko-fi](https://ko-fi.com/FrontEagle).</sub>
