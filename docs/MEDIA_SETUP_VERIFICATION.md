# Experimental media setup

`./bin/slstack install --all` installs the pinned media environments, checks pipeline
imports, supplies hash-verified configuration and tokenizer assets, and builds the
native engines and the WebUI. `install --all --check` checks readiness. Model weights
and any required text encoders or VAEs are supplied locally.

Video tools are the system `ffmpeg` and `ffprobe`. `slstack/media_tools.py` locates them
and runs each with `-version`, so a name on `PATH` that cannot start is reported as not
ready and `install --all` refuses it. The retired setup downloaded and hashed a pinned
Windows FFmpeg archive into the runtime instead; there is no Linux equivalent in this
fork. This box answers ready with `/usr/bin/ffmpeg` and `/usr/bin/ffprobe`, both n9.0.2.

What has not been verified is the pipeline itself. No media runtime is installed here (the
locks ask for a CUDA torch environment of about 24 GiB), no media weights are on this box,
and the streamed weight transport refuses on Linux because the fetcher library exports no
`wf_*` API - `tests/test_media_fetcher_abi.py` reads the export table and prints the count.
Treat media generation as unported until a run says otherwise; the portable tiers in
`vendor/engines/cosmos_windows/weight_streaming/` are the Linux weight paths, and the
fetcher route in `vendor/studio/studio/` is gated to Windows.

## Model families

The general media worker and Qwen-Image 2.1 worker use separate environments and dependency locks. Krea's bundled pipeline uses supplied tokenizer/config assets and discovered local weights. Complete pipeline folders are model choices; individual shards are not.

The retained adapters cover Anima, Krea turbo, Qwen-Image 2.1, Z-Image, Flux2 Klein and
Kroma with explicit compatible adapters. A matching family name does not guarantee
matching tensor shapes or components. [Model discovery and adapters](STACK_MEDIA_AND_STATE.md)

Krea's loader is chosen per platform: the fetcher-backed resident and streamed reads are the
Windows arm, and on Linux tensors, text encoder and VAE are read through `safetensors`
(`vendor/studio/studio/krea2_diffusion.py`), in both cases with tensor-coverage checks and
owned copies before stream-buffer release. Select Streamed or Resident per model in
Settings -> Stack. An adapter is applied only when explicitly selected for that generation.

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
