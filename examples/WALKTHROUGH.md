# Setup and chat walkthrough

Use the exact Flash 1017 Q4 model described in the [README](../README.md#requirements).
Media and LoRA workflows are experimental, and media has not been verified on Linux.

## Install

```sh
git clone https://github.com/fronteagle739/streaming-llama-stack.git
cd streaming-llama-stack
./bin/slstack install --all
./bin/slstack install --all --check
```

Create `my.local.json` with `model`, `gpu_uuid` and explicit host budgets. Start from
the README example; other keys come from
[config.example.json](../config.example.json). The install step builds the producer,
`ds4-decode.so`, the native server and the chat UI through `build.sh`. It does not
fetch model weights.

## Start a conversation

```sh
./bin/slstack --config my.local.json check
scripts/start-dsv41-single-262k.sh --dry-run
scripts/start-dsv41-single-262k.sh --webui
```

Open the printed router URL. The first prompt loads the producer. Full-file hashing
runs once when the model identity is first recorded and reports progress. Ask a short
question, append a document, then ask a follow-up: the producer retains the conversation
and processes new tokens. Appended prefill shows processing progress; completed replies
show decode tok/s.

The preset uses the exact ULTRA dependency cone (keep 2541), growable f32 context at
262,144 tokens, automatic sweeps with a 61,440-token fold cap and reads over io_uring.
The 1M script admits 999,424 tokens with bf16 context. `--dry-run` prints the resolved
start config and runs nothing.

Ask the model to switch itself to Max speed using its quality/speed tool. The tool
reports requested and applied settings. For precise code, select Max quality or
Balanced. [Live controls](../docs/LIVE_GENERATION.md)

## Media

Select existing model or component folders in Settings -> Stack. Anima requires its
pipeline directory and a matching text encoder; other families have their own
components. Request a small image, such as a red lighthouse, with explicit dimensions,
steps and a seed. The tool card shows progress and returns the image. Uploads and
generated images can condition later tools.
[Media setup](../docs/MEDIA_SETUP_VERIFICATION.md)

The earlier native-server example used `start --serving native`:
[chat](walkthrough/chat-desktop.png), [phone chat](walkthrough/chat-phone.png),
[quality tool](walkthrough/quality-desktop.png) and [image](walkthrough/image-desktop.png).
These screenshots illustrate that native path; they were taken on the Windows build.
Current checkpoint behavior is described in
[media and state](../docs/STACK_MEDIA_AND_STATE.md).

## Stop or change preset

```sh
scripts/start-dsv41-single-262k.sh --stop
scripts/start-dsv41-single-1m.sh --webui
# Stop this preset before starting another:
scripts/start-dsv41-single-1m.sh --stop
```

Each launcher accepts `--config`, `--model`, `--webui`, `--vision`, `--dry-run` and
`--stop`. Use the same config to stop. The launcher runs detached and prints its log
path; the guardian owns the engine tree, and `--stop` asks for a graceful shutdown.
Native shutdown drains storage work and releases host and GPU memory.

For approximate YOCO use `scripts/start-dsv41-single-262k-ultra-yoco.sh`, or
`--keep 140` on `scripts/start-dsv41-single.sh`; without that the launcher uses exact
cone. [Quality and timing](../docs/ULTRAYOCO_QUALIFICATION.md). Short resident
follow-ups are also an explicit local toggle:
[defaults and experiments](../docs/MERGEPERF.md).
