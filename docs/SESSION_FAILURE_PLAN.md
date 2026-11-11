# The five ways pi's session on this box failed, and what each one costs

Goal: make a long pi session on the single engine finish without a tool call turning into
`Malformed tool call`, without a transient reader on the card refusing admission, and without
one abort at context > 8192 columns killing every later request in the session.

Observed 2026-11-11, from a real pi session against `serving=single_engine`
(`rtx.local.json`: `gpu_idle_max_mib` 2560, `gpu_min_free_mib` 0, `gpu_wait_seconds` 0,
model `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`). Five distinct failures, reported
by the user in the order they appeared:

| # | symptom | class | depth |
|---|---|---|---|
| 1 | `503 … RTX is not idle enough for exclusive admission; no other process was ejected: 1415, 30780` | admission gate | shallow |
| 2 | `Persistent producer aborted … Actual exact radix top-k first/merge dispatch coverage differs` | engine self-check | **deep** |
| 3 | `Retry failed after 3 attempts: <same abort>` | retry policy, amplifier of 2 | medium |
| 4 | `409 … Another request owns this decoder` | single-owner serialization | shallow |
| 5 | `503 … [Errno 98] Address already in use` | stale socket on start | medium |
| 6 | failed tool call `Malformed tool call … raw: <｜DSML｜ …>` | DSML parser tolerance | shallow, verified |

Numbered below as they are fixed. One fix per commit, sequentially, each with a test that fails
before the change. Item 2 is the only one that needs a rebuild of the CUDA backend.

## Item 6 first: the DSML tags the model emits are not the tags the parser accepts

Fixed first because it is the one already reproduced offline, and because it silently breaks
agentic use: a dropped tool call is not an error the model can see and retry, it is a turn lost.

The emitted text (verbatim from the failed call's `raw`):

```
<｜DSML｜
 calls>
<｜DSML｜ invoke name="bash">
<｜DSML｜ parameter name="command" string="true">grep -n … | head -30; echo
---; sed -n '1,120p' scripts/dsv41_single.py</｜DSML｜>
</｜DSML｜ invoke>
</｜DSML｜ calls>
```

Against `slstack/ds4_tool_parse.py`, two deviations from the style the stack renders
(`ds4_render.py:36-41`):

- the opening tag carries whitespace inside it: `<｜DSML｜` + newline + ` calls>`, where
  `DS41_TOOL_CALLS_START` is `<｜DSML｜ calls>`. `parse_deepseek_generated` matches the tag with
  plain `_find`, so no style matches, `start is None`, and the whole block is returned as
  *content*;
- the parameter closes with the bare `</｜DSML｜>`, not `</｜DSML｜ parameter>`. Even with the
  first deviation corrected, `_find(text, param_end, value_start)` returns `None` and the parse
  returns `ok=False`.

Both reproduce: `finish_message(<as emitted>, has_tools=True, think=False, finish='stop')` gives
`calls=0, tool_error=True`; `<with the space restored>` gives `ok=False, tool_error=True`. The
canonical block the stack renders itself parses into one call, so the grammar is not wrong — the
tolerance is.

`finish_message` then routes to the "invalid arguments are never guessed or executed" branch
(`ds4_tool_parse.py:304-306`), which sets `tool_error`; `router.py:120-122` / `router.py:188-190`
turn that into the `invalid-<uuid>` / `Malformed tool call` activity pi showed. That part is
correct behaviour — the stack must not execute a call it cannot parse.

Why the model drifts, as far as the evidence goes: `｜DSML｜` (id 128825) is GGUF token type
`NORMAL`, not `CONTROL` — it is not a special token to llama.cpp, and the vocab has no
`<｜DSML｜ calls>` entry. The stack's own render of the tag is
`Ġ<`, `｜DSML｜`, `Ġcalls`, `>Ċ` (the trailing newline is folded into the `>` piece by the
DeepSeek pre-tokenizer), so the tag is four ids with no boundary protecting it. An IQ2_XXS
model inserting a newline id inside that run, and shortening a close tag to the bare marker, is
generation drift, not a tokenizer bug. Verification of that claim, and the limit of it, is the
round-trip test added with the fix: the stack's `rendered()` → `detokenize()` of the tools
prompt is byte-exact, so nothing on our side introduces the newline.

Fix, in `slstack/ds4_tool_parse.py`:

1. Match a DSML tag as *prefix, optional whitespace, name* rather than one literal, for the
   styles whose prefix ends at `｜DSML｜` / `DSML｜`. Style 1 (`<tool_calls>`) keeps exact
   matching. This covers both `<｜DSML｜ calls>` and `<｜DSML｜\n calls>` and any tab/CRLF between.
2. Accept the bare marker close (`</｜DSML｜>`, `</DSML｜>`) as the close of whichever element is
   open, longest-match first so a real `</｜DSML｜ parameter>` still wins.
3. Widen escaping in `ds4_render.py` so `tool_text_needs_escape` treats the bare close as a
   terminator too. Without this, a command string containing `</｜DSML｜>` would be written
   unescaped and would now truncate the value — the round-trip test pins it.
4. Keep the refusal to guess: a block that still cannot be parsed produces `tool_error`, and the
   diagnostic gains the reason (`which tag did not match`) so the agent sees something
   actionable instead of a bare "was not executed".

Tests: extend `tests/test_ds4_tool_parse.py` with the drifted block from this session, the
whitespace variants, the bare close, a nested-object parameter closed bare, a value containing
the bare close (must round-trip, not truncate), and genuinely broken markup (must still yield
`tool_error` and no calls). Plus `tests/test_ds4_render.py`: render → parse identity for a call
whose arguments contain `</｜DSML｜>`.

## Item 1: the admission gate reports the wrong reason and never waits

`lease.py:277` raises one message for three conditions:

```python
used, free, busy = [int(x.strip()) for x in result.stdout.strip().split(',')]
if used > cfg['gpu_idle_max_mib'] or free < int(cfg.get('gpu_min_free_mib', 0)) or busy > 5:
    raise AdmissionRefused('RTX is not idle enough for exclusive admission; no other process was ejected: ' + result.stdout)
```

The reported numbers were `1415, 30780, 7` — used 1415 MiB is under `gpu_idle_max_mib` 2560 and
free is not checked (`gpu_min_free_mib` 0), so what refused the session was **`busy > 5` at 7 %
GPU utilization**, i.e. a reader on the card (`brave --type=gpu-process`, 140 MiB, was live on
this box during the session). The message says "not idle enough" and prints memory only, so it
reads as a memory problem.

Two more things make it sting:

- `busy > 5` is a single instantaneous `nvidia-smi` sample. Utilization is the noisiest of the
  three quantities and one 7 % sample is not contention.
- `gpu_wait_seconds` is 0 in `rtx.local.json`, and the retry loop that honours it lives only in
  `guard.supervise` (`guard.py:415-445`, `except (AdmissionRefused, PermissionError)` → sleep →
  retry). The single engine takes its lease **per request** at producer launch
  (`combo_single.py:219-220`, `376-377` via `lease_factory`), which is inside the handler, so a
  mid-session refusal surfaces as a 503 with no wait at all.

Fix:

1. Name the tripped condition in the message — `memory used …`, `memory free …`, `utilization
   …` — with its own threshold, and print the process list from `nvidia-smi
   --query-compute-apps` so the holder is visible rather than asserted absent.
2. Make the utilization threshold a config key (`gpu_idle_max_util`, default 5) validated in
   `config.py` alongside `gpu_min_free_mib`.
3. Sample utilization twice over a short interval and require both samples to exceed it, so a
   one-shot reader does not refuse the window.
4. Honour `gpu_wait_seconds` at the engine's own admission point: wrap the lease acquisition in
   the same wait/retry the guardian uses, so a busy card costs a wait rather than a failed turn.

Tests: `tests/test_gpu_admission.py` (new) with `subprocess.run` patched — asserts the message
names the condition and the threshold, that two low samples admit and two high samples refuse,
that `gpu_idle_max_util` is read, that a bad value is rejected by `config.validate`, and that
the engine retries then admits when the first refusal clears.

## Item 4: a second concurrent request is refused, not queued

`combo_single.py:712` acquires the decoder lock with `blocking=False` for
`/v1/chat/completions` and answers 409 `Another request owns this decoder`. The control routes
(`:658`) and `shutdown` (`:680`) do wait — with `timeout=request_timeout_s`. So the one route an
agent actually hammers is the only one that refuses immediately, and pi (or a client issuing
parallel tool calls, or a retry while the first request is still draining) gets a hard 409.

The decoder genuinely cannot run two requests: one resident producer, one conversation. The
correct behaviour is to *queue* and run the second when the first finishes, and to 409 only when
the queue itself is full or its wait expires.

Fix:

1. Wait on the lock for the admission window (`cfg['request_timeout_s']`, already the bound the
   control routes use) instead of refusing, and expose `single_engine_queue_depth` (default 4)
   so a queue overflow is refused with the depth in the message.
2. Say which it is in the 409 (`queue full`, not "another request owns the decoder"), so a
   client can distinguish "wait and you'll be served" from "you are not going to be served".

Tests: `tests/test_single_engine_queue.py` (new) against a fake engine whose `run` blocks on an
event: a second request waits and then completes; more than `queue_depth` queued requests get one
409 naming the queue; a control-route wait still times out at its bound.

## Item 5: a stale socket makes the next start answer 503 `[Errno 98]`

`combo_single.serve` binds with `ThreadingHTTPServer((cfg['host'], cfg['port']), …)`
(`:818`) and nothing checks the port first. A force-killed producer that carried the pinned
staging tiers unwinds in the kernel for seconds and keeps the socket occupied without ever
reaching `LISTEN` — already documented here in `scripts/gpu_handoff.py:103-140`, which works
around it with `stack_ports()` and `ports_held()` (`ss -tanH`, any state). The engine has no
equivalent: the bind raises `OSError`, the handler's catch-all turns it into
`503 … [Errno 98] Address already in use` (`combo_single.py:797`), and the caller gets a raw
errno with no holder and no retry.

Fix:

1. Pre-flight the ports the stack binds before `serve()`, using the same any-state check
   `gpu_handoff.ports_held` uses; poll until free within `gpu_wait_seconds`, then fail with a
   message naming the port and the pid holding it (from `ss -tanp` / `/proc`).
2. Move that check to the launcher (`scripts/dsv41_single.py`) as well, so a start fails loudly
   before the lease is taken rather than mid-request.

Tests: `tests/test_engine_port_wait.py` (new) — a socket bound on the configured port makes the
engine wait then refuse with the port and pid named; releasing it lets `serve()` bind; nothing is
attempted before the port is free.

## Item 2: the top-k coverage self-check does not know about single-row decode

This is the deep one, and the one that ended the session: it repeats identically on every
request, so the whole session is lost rather than one turn.

The check (`llama-dsv41-executor.cpp:2031-2041`):

```cpp
if(ggml_dsv41_stream_topk_requested(node)){… expected+=node->ne[0]==512;continue;}
if(node->op!=GGML_OP_TOP_K||node->ne[0]!=512||!node->src[0]||node->src[0]->ne[0]<=8192
   ||ggml_nrows(node->src[0])<=1)continue;
uint64_t width=(uint64_t(node->src[0]->ne[0])+8191)/8192*512;expected+=2;
while(width>8192){width=(width+8191)/8192*512;++expected;}
…
const auto actual=ggml_backend_cuda_dsv41_topk_radix_queued()-topk_radix_before;
require(actual==expected,"Actual exact radix top-k first/merge dispatch coverage differs");
```

The CUDA side (`ggml-cuda/top-k.cu:60-146`) decides the dispatch count from
`deterministic_rows = nrows > 1 || force_bitonic_single_row || !cub_single_row`, and
`cub_single_row` is only true when `LLAMA_DSV4_TOPK_CUB_1ROW=1`. So **by default a single-row
TOP_K wider than the shared-memory bitonic limit takes the same tiles/merge path as prefill and
increments `dsv41_topk_radix_queued`** (`argsort.cu:841, 845, 882, 886`), while the executor's
scan drops every `nrows <= 1` node. For a decode step whose index width exceeds 8192 columns —
which is any decode step once the conversation is past ~8k tokens — `actual` exceeds `expected`
by 2 plus one per merge pass, and the require fires.

That is a one-sided update: `tests/dsv41_topk_single_row.cpp` documents the deliberate change
("a single-row ggml top-k (decode) must select the same ids in the same order as a multi-row
top-k … including widths above the 12288-column shared-memory bitonic limit") and the coverage
scan still assumes the pre-change world where one row went to `DeviceTopK` and dispatched no
radix kernel. The check only runs when `LLAMA_DSV41_TOPK_RADIX=1`, which the box's `rtx-llama`
launch config sets (`docs/VARIABLES.md:105`); it is not in `rtx.local.json`.

The producer log of the session that failed settles it
(`runtime/single-engine/ultra-ecac43f59d753fb5/stderr.log`, 25654 and the lines above it):

```
DS4PERF_UNALIGNED_APPEND_COMPLETE layer=1 start=0 absolute_start=97290 count=1 …
PERFEXP_INDEX_FP4_MMA tokens=1 keys=48645 spans=7,7 packed_bytes=3310064 certificate=exact_integer_sum
[dsv41-executor] layer=2 rows=1 … graph_cache_hit=0
LLAMADECODE_PRODUCER_ABORT kind=exception … sequence=3 reason=Actual exact radix top-k … differs
```

The turn's first decode step at layer 2 is `rows=1` over `keys=48645` — one row, far above 8192
columns — and no `DS4PERF_TOPK_RADIX_COMPLETE` line was written for that layer, because the
require fired before it. Layer 1 (`count=1`, gate width 384) printed `actual=0` and passed, so
the disagreement is exactly the wide single-row node: the backend adds 2 dispatches (first pass
plus one merge at `first_width = 6*512 = 3072 ≤ 8192`), the scan adds 0. The same session was at
`absolute_start=97290`, so every later request re-prefills past 8192 keys and aborts again —
which is why the error repeated instead of costing one turn.

The shape arithmetic is read from the source and the log carries the row count, key count and the
missing completion line for the failing layer. The gate in step 1 below is what pins it.

Fix, in order:

0. **Make the abort readable before changing anything.** The require carries no numbers. Log
   `expected`, `actual`, and one line per TOP_K node (`ne[0]`, `src[0]->ne[0]`, `nrows`, tagged
   or not) into the producer's diagnostic stream, so the abort names the shape that broke the
   count. This alone turns the next occurrence into a diagnosis.
1. **One predicate, not two.** Add a pure function — `dsv41_topk_radix_dispatches(ncols, k,
   nrows, deterministic_rows, radix_enabled)` — in a header both `top-k.cu` and the executor
   include, and have both sides derive their count from it. The executor passes the same
   `deterministic_rows` expression the backend uses; the counter stays the ground truth.
2. Have the executor's scan stop special-casing `nrows<=1` and use the shared predicate instead,
   so single-row decode is counted exactly when the backend dispatches it.

Tests: `tests/dsv41_topk_coverage.cpp` (new, table-driven, no GPU) over the shape grid already in
`dsv41_topk_single_row.cpp` — `{272,272}, {384,6}, {8192,512}, {16384,512}, {65536,512},
{131072,2048}` at `nrows` 1 and 3, with `LLAMA_DSV4_TOPK_CUB_1ROW` both unset and `1` — asserting
the predicate's count equals the value the CUDA path would add. `tests/dsv41_topk_single_row.cpp`
gains an assertion that the counter delta equals the predicate for each shape (runs only when a
CUDA backend exists; skipped otherwise, as the gate already does). Then re-run the gate on the
card when it is free, and record the numbers in this document.

## Item 3: a deterministic abort is retried three times against a cold engine

`combo_single.py:786-791` retires the producer on *any* request exception, and pi retries the
stream error three times (`auto_retry_end` → "Retry failed after 3 attempts"). So item 2 cost
this session three cold launches: each retry re-prefills the whole conversation and aborts at
the same node. `docs/TURN_COST_PLAN.md` measures a prompt head at 21-66 s per turn, so the
amplifier is roughly a minute of wall clock per failure, on top of the abort itself.

Fix:

1. Distinguish an engine *self-check* abort from a transient failure. The producer reports
   `LLAMADECODE_PRODUCER_ABORT kind=… reason=…`; a `kind=exception` abort whose reason is a
   coverage require is deterministic, and retrying it unchanged cannot help. Keep the retirement
   (the producer is gone) but mark the error non-retryable so the client sees one failure with
   the reason rather than three.
2. Say what was lost in the message: whether the conversation prefix survived. `retire` keeps
   the snapshot (`save_snapshot('conversation')`) and `restore` reloads it, so the prefix is not
   necessarily gone — the error should not imply it is.

Tests: `tests/test_producer_abort.py` gains the classification case — a coverage-abort line maps
to a non-retryable error whose message carries the reason, while a `bad_alloc` line stays
retryable — and that the prefix-retention state is reported.

## Order, and what is not a stack bug

1. Item 6 (DSML tolerance) — offline, reproducible, no rebuild.
2. Item 1 (admission reason, threshold, wait) — Python only.
3. Item 4 (queue instead of 409) — Python only.
4. Item 5 (port pre-flight) — Python only.
5. Item 2 (shared top-k predicate + readable abort) — needs `build.sh` and a free card.
6. Item 3 (retry classification) — Python, after 2 so the reason string it keys on exists.

Baseline before any change: `env PYTHONPATH=.:tests STUDIO_RUNTIME_ROOT=$PWD/runtime
python3 -B -m pytest tests/ -q`. The recorded baseline on this box is 31 failed / 696 passed
(`HANDOFF.md`), the same 31 each run; each commit re-runs the suite and the count must not move.

Two of the reported symptoms are not defects here:

- `no other process was ejected: 1415, 30780` and the 29.7 GiB / 97 % reader seen during this
  investigation (`qwen38-gyro/Strata/build/strata --serve`) are real contention for one card.
  The stack's exclusive-admission contract is correct; only its reporting and its refusal to
  wait are wrong. Run `scripts/gpu_handoff.py` to hand the card over rather than relaxing
  `gpu_idle_max_mib`.
- `Malformed tool call was not executed` is the right *outcome* for markup the parser cannot
  read: the stack must not execute a guessed call. What is wrong is that it cannot read markup
  the model legitimately produces.
