# Supported models

The streaming engine is a llama.cpp fork. It reads weights from NVMe over io_uring
and runs prefill one layer at a time across the whole prompt chunk. The ds4 decode
kernels are specific to DeepSeek V4.1 Flash, and that is the only route this fork
qualifies.

## What runs here

| Model | Architecture | Streaming prefill | Decode |
|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek41` | Yes | ds4 kernels on the RTX 5090 |

The macOS decoders were deleted with the platform, so Kimi K2.5/K2.6/K2.7, GLM-5
through GLM-5.3, MiniMax M2.5/M3, Step-3.5/3.7 Flash and MiMo V2.6 Pro have no
serving route in this fork. The launcher refuses a `general` config rather than
starting a half-built engine:

```
The Mac-paired general engine was deleted with macOS; this fork serves DeepSeek
V4.x on the Linux ds4 engine
```

The llama fork still builds those architectures, and `ds4` still reads `glm-dsa`,
`glm5-next` and `qwen4exp` metadata, so the code paths exist without a qualification
behind them. `llama-server` serves any architecture llama.cpp understands over HTTP,
with its own prefill and no weight streaming.

## Inherited measurements

The rates below came from the upstream Windows build (RTX 5090, 32 GB of system RAM,
a 25 GB/s NVMe RAID-0 reading weights through DirectStorage) and from the deleted
macOS decode half on an M3 Ultra with 512 GB. Nothing here has been re-measured on
Linux, and the Mac column cannot be reproduced in this fork at all.
[All conditions](PERFORMANCE.md).

| Model | Architecture | Best RTX prefill | Decode, as measured then |
|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek41` | 1,635 tok/s at 103K (YOCO); 1,163 tok/s at 48.5K | RTX: 19-21 tok/s warm at Max speed, 6.0 at Max quality. Mac: 31 tok/s |
| Kimi K2.5 / K2.6 / K2.7 | `deepseek2` (MLA) | 652 tok/s at 16K (K2.5); 576 tok/s at 23K (K2.6) | Mac MLX: 24.3 tok/s (K2.6) |
| GLM-5 / 5.1 / 5.2 / 5.3 | `glm-dsa` | 764 tok/s at 16K (GLM-5) | Mac MLX: 14.1 tok/s (GLM-5.3, short prompt) |
| MiniMax M2.5 / M3 | `minimax-m2`, `minimax-m3` | 1,576 tok/s at 9K; 439 tok/s at 196K (M2.5 Q3) | Mac: 20-35 tok/s (M2.5) |
| Step-3.5 / 3.7 Flash | `step35` | 2,294 tok/s at about 25K (3.5 Q8_0) | Mac: 25-33 tok/s (3.5) |
| MiMo V2.6 Pro | — | — | Mac ds4 with expert streaming: 6.0 tok/s at Max quality, 9.0 at bias 1.36 |

Kimi K2.7 uses the same architecture as K2.5 and K2.6. GLM-5.3 Flash (`glm5-next`)
and Qwen3.8 Flash Next (`qwen4exp`) are different architectures and were never in the
streaming engine; Qwen3.8 Flash Next ran on the Mac through mlx-serve, which is also
deleted.

## DeepSeek V4.1 Flash files

The combined single-engine producer accepts native `deepseek41` Flash GGUFs with
Q4_K, IQ2_XXS gate/up plus Q2_K down, or MXFP4 experts. Dense matrices accept Q8_0,
Q4_K and Q4_0; auxiliaries retain the architecture's F16/F32/I8 representations.
Unsupported tensors fail before hashing or engine startup with their name and type.
Receipts bind the selected file's digest and physical identity; filenames and a
particular measured model SHA are not admission rules.

The measured 483 GiB file uses Q4_K experts and Q8_0 dense matrices. Quality/speed
and YOCO measurements refer to that file. Other quantizations have CPU and header
coverage, not GPU quality or throughput measurements.

[Conversion instructions](DSV41_GGUF.md) cover existing raw safetensors and the
measured main/DSpark/Vision layouts. Public MXFP4 shards require lossless assembly
into one GGUF before this single-file loader can open them. No model is downloaded by
the wrapper, and no shard set on this box is in the accepted type set yet.

## Choosing a route

One route: `scripts/start-dsv41-single-262k.sh`, `scripts/start-dsv41-single-1m.sh`,
or `python -B examples/run_preset.py --single ultra-cone --config <local.json>`.
The native server (`start --serving native`) and the two-process coordinator are
described in [COMBO](COMBO.md). [Backend setup](BACKEND_SETUP.md) covers building the
two engines that exist here.
