# Experimental media setup

`./bin/slstack install --all` installs the pinned media environments, checks pipeline
imports, supplies hash-verified configuration and tokenizer assets, and builds the
native engines and the WebUI. `install --all --check` checks readiness. Model weights
and any required text encoders or VAEs are supplied locally.

Video tools on Linux are the system `ffmpeg` and `ffprobe` on `PATH`:
`slstack/media_tools.py` reports what it finds there. The pinned archive download and
its `installed.json` receipt are the Windows arm, and the media pipeline as a whole has
not been verified on this box, so treat media generation as unported until a run says
otherwise.

## Model families

The general media worker and Qwen-Image 2.1 worker use separate environments and dependency locks. Krea's bundled pipeline uses supplied tokenizer/config assets and discovered local weights. Complete pipeline folders are model choices; individual shards are not.

The retained adapters cover Anima, Krea turbo, Qwen-Image 2.1, Z-Image, Flux2 Klein and
Kroma with explicit compatible adapters. A matching family name does not guarantee
matching tensor shapes or components. [Model discovery and adapters](STACK_MEDIA_AND_STATE.md)

Krea resident tensors and VAE read through the weight fetcher, which is io_uring on
Linux, with tensor-coverage checks and owned copies before stream-buffer release.
Select Streamed or Resident per model in Settings -> Stack. An adapter is applied only
when explicitly selected for that generation.

## H3 video

H3 defaults to PyTorch SDPA and a labelled audiovisual caption. If omitted, frame count is 121 at 24 fps. The tool uses the chat model's generation prompt without starting a separate prompt-enhancement model. Optional attention packages can be selected through `h3_attention_backend` when installed.

Known fused Comfy exports receive their corresponding QKV layout; model configuration can declare `qkv_layout` as `sectioned` or `interleaved`. The loader indexes immediate transformer shards and excludes nested composition caches. Complete single-file transformers need no synthetic manifest.

Recorded 121-frame examples produced 5.175-second H.264/AAC MP4 clips, including
explicit three-step adapters and an eight-step request. Those runs came from the
Windows build. This is model-specific behavior, rather than coverage of every
checkpoint or platform.

## Chat interaction

Tool cards show arguments, progress, errors and results. Completed images enter vision
context, and video plays inline. There is no fixed tool-round limit; malformed tool
syntax is reported instead of executed as a guessed call. Chat preserves the
conversation across media GPU handoffs using its configured state/checkpoint behavior.

See the [walkthrough](../examples/WALKTHROUGH.md) and
[media/state reference](STACK_MEDIA_AND_STATE.md).
