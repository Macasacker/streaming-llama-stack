# Single RTX machine: DSV4.1

One machine, one engine: streaming-llama prefill followed by ds4 decode in the same
process. `examples/run_preset.py` selects the arithmetic profile; the labelled
`scripts/start-dsv41-single*.sh` launchers wrap the same presets for detached use.

Short follow-ups reuse resident experts through the sweep graph by default. The
measured 32-token append took 2.7-2.9 s, with small numerical differences
(KL 0.0000235-0.00250 at 2K-32K; recall and top-1 passed). Set
`"LLAMADECODE_APPEND_RESIDENT_SWEEP": "0"` in `env_overrides` to use full sweeps.
See the [measured tradeoff](../../docs/EXPERIMENT_E2C.md). Those numbers came from the
Windows build; nothing has been re-measured here.

| Profile (`--single`) | Prefill setting |
| --- | --- |
| `accurate` | Accurate arithmetic; YOCO off |
| `fast` | Fast arithmetic; YOCO off |
| `ultra` | ULTRA arithmetic; YOCO off |
| `ultra-cone` | Recommended: exact ULTRA dependency cone, keep 2541 |
| `ultra-yoco` | Approximate keep-140 replay; the exact cone is `ultra-cone` |

`rtx.local.json` is tracked: it is the configuration the author's RTX 5090 box launches
with, and its budgets are part of what a measurement reported here means. On another
machine, edit it rather than copying `config.example.json` over it: point `model` at the
original supported DSV4.1 1017 Q4 GGUF, and set the built-producer and runtime-root paths
if they differ from the stack defaults. Install or build the stack first; these examples do
not download model weights or select arbitrary replacement quantizations.

From the stack root:

```sh
python -B examples/run_preset.py --single ultra-cone --context 262k --config ./rtx.local.json --dry-run
python -B examples/run_preset.py --single ultra-cone --context 262k --config ./rtx.local.json --webui
python -B examples/run_preset.py --single accurate --context 262k --config ./rtx.local.json
```

or, detached:

```sh
scripts/start-dsv41-single-262k.sh --config ./rtx.local.json --webui
scripts/start-dsv41-single-1m.sh --config ./rtx.local.json
scripts/start-dsv41-single-262k.sh --config ./rtx.local.json --stop
```

`--context 262k` is the default. `--context 1m` selects 999,424 configured tokens with
the growable bf16 context cache; that is capacity, not a measured one-million-token
prefill result. `--dry-run` starts nothing and writes nothing. Stop a running stack with
the same script and `--stop`, or Ctrl-C in a foreground run; its guardian owns the
engine tree. A foreground run also follows the engine server's log, so a running turn
reports itself (`--no-progress` turns those lines off; see
[docs/FIRST_RUN.md](../../docs/FIRST_RUN.md)).

The exact cone is the default for `scripts/start-dsv41-single-262k.sh` and
`scripts/start-dsv41-single-1m.sh`. `scripts/start-dsv41-single-262k-ultra-yoco.sh` and
its 1M twin are the approximate route, as is `--single ultra-yoco`; `--keep 140` on
`scripts/start-dsv41-single.sh` says it directly. The cone is bit-exact against the same
full-prefill control at 8K and 49,555 text tokens, on the build that measured it.
Real-image numerical qualification remains separate.

YOCO is approximate. Use a vision-enabled local config and producer for image prompts;
do not advertise the text-only YOCO speed as image throughput. Prefill quality is
separate from the live decode bias slider and phase sampler controls.

Exact resident sweep follow-ups are enabled with `LLAMADECODE_APPEND_RESIDENT_SWEEP=1`.
The per-token append path changes logits and stays opt-in
(`LLAMADECODE_APPEND_DECODE_MAX=96`). LFU admission defaults to 1 with a half life of
16; an explicit 0 keeps LRU. Certified FP4 index scoring is enabled by default on
supported SM120 bf16-index paths. The 100K comparison was bit-identical and faster;
unsupported inputs use the bf16 scorer. Set `LLAMA_DSV41_INDEX_FP4` to `"0"` in local
`env_overrides` to disable it. See [E6 measurements](../../docs/EXPERIMENT_E6.md).
