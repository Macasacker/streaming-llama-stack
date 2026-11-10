#!/usr/bin/env bash
# One-shot first-run probe for the DSV4.1 single-engine stack.
# Run this while nothing else holds the GPU; it writes evidence under
# runtime/debug/ so the run can be read afterwards.
#   --quick   one 16-token request with a progress heartbeat, then stop: it proves the
#             producer opens, prefills and decodes, and closes the session cleanly so the
#             producer log survives, without spending minutes on the 256-token and
#             ~75K-token probes.
#   --calibrate PATH     run the producer with DS4_CACHE_AFFINITY_CALIBRATE=PATH, which writes a
#             routing calibration bound to this model's bytes after at least 16 decode samples per
#             layer. Use it on a full run, not --quick: 16 tokens is exactly the minimum, and the
#             producer publishes the file only when every layer reached it. The path is made
#             absolute and checked for writability first, because the producer resolves it against
#             its own working directory and aborts the run if the open fails. The result goes to
#             cache_affinity_spread_file once the engine accepts it at a nonzero bias.
#   --bias[=VALUE]  put quality_speed on every request so the affinity bias is actually applied.
#             Without it the run sends quality_speed 0 explicitly, because the published value
#             lives in runtime/generation/live.json across runs: omitting it is not an unbiased
#             run. The value drives the bias (`combo_single` calls `generation.bias(quality_speed)`
#             at slstack/combo_single.py:266); binding a calibration only makes a nonzero value
#             possible - `Store.bias` raises when nothing is bound (slstack/generation.py:128).
#   --routed  demand-read experts on every layer instead of only the tail: sets
#             LLAMADECODE_PREFILL_ROUTED=1 in the derived config, which
#             slstack/combo_producer.py:413 passes through untouched. The default for a prompt
#             above 2 048 rows is `tail`, which reads layers 0-20 whole - all 384 experts per
#             block, ~75 GiB a turn - and that is the larger half of a turn's prompt head
#             (docs/DECODE_GAP_PLAN.md, Step 6). This is a measurement flag, not a recommendation:
#             the routing change trades the overlapped full-stage prefetch that the matched ULTRA
#             control was qualified with, so read it against a run without it.
set -uo pipefail
quick=0
calibrate=
bias=
routed=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) quick=1; shift ;;
    # Write a routing calibration for this model's bytes: the producer needs a decode of at least
    # 16 tokens to collect 16 samples per layer, so use this on a non-quick run and at bias 0
    # (quality_speed 0 in the request; slstack/generation.py publishes no bias when none is bound).
    --calibrate) calibrate="$2"; shift 2 ;;
    # A bound calibration changes nothing until a request asks for a profile that carries bias.
    --bias) case "$1" in *=*) bias="${1#*=}"; shift ;; *) bias="0.5"; shift ;; esac ;;
    # Without --bias every request sends quality_speed 0 rather than nothing: the published bias
    # survives in runtime/generation/live.json, so an omitted value is not an unbiased run.
    --routed) routed=1; shift ;;
    *) echo "unknown option: $1 (use --quick, --calibrate PATH, --bias[=VALUE] or --routed)" >&2; exit 2 ;;
  esac
done
# A producer crash on Linux hands systemd-coredump the whole address space (~22 GiB of
# host staging here), which lands as a multi-gigabyte file under /var/lib/systemd/coredump.
# The stderr logs carry the diagnosis; the core does not.
ulimit -c 0 2>/dev/null || true

REPO="$HOME/workspace/local-llm-playground/streaming-llama-stack"
MODEL="$HOME/workspace/local-llm-playground/ds4-models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
OUT="$REPO/runtime/debug"
cd "$REPO" || exit 1   # step 5 reads runtime/bench-50k.txt relative to the repo
STAMP="$(date +%Y%m%d-%H%M%S)"
STAMP_EPOCH="$(date +%s)"
mkdir -p "$OUT"
RUNLOG="$OUT/run-$STAMP.log"
exec > >(tee -a "$RUNLOG") 2>&1

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
read_used_free_busy() { nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr ',' ' '; }
idle_gate() { python3 -c "import json;print(json.load(open('$REPO/rtx.local.json')).get('gpu_idle_max_mib',1024))"; }

# The lease (slstack/lease.py:210) refuses admission when device memory used exceeds
# gpu_idle_max_mib OR utilization exceeds 5%. On a desktop that baseline is the
# compositor plus browser GPU buffers (~1.8 GiB here), not a model serve process.

echo "=== $STAMP ==="
echo "model: $MODEL"
[ -f "$MODEL" ] || { echo "FATAL: model file missing"; exit 1; }

# --- 0. the card must be free ---------------------------------------------------------------
read -r U F B < <(read_used_free_busy)
IDLE=$(idle_gate)
echo "gpu used=${U} MiB free=${F} MiB util=${B}%  | lease gate: used <= ${IDLE} MiB, util <= 5%"
if [ "${F:-0}" -lt 24000 ]; then
  echo "ABORT: only ${F} MiB free - stop the gyro-m serve (and any deepseek-strata) first."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5
  exit 2
fi
if [ "${U:-0}" -gt "$IDLE" ]; then
  echo "ABORT: ${U} MiB used exceeds gpu_idle_max_mib=${IDLE}, so the lease will refuse admission."
  echo "Anything holding >1 GiB here is a model serve, not the desktop baseline: raise the value or stop it."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5
  exit 3
fi
if [ "${B:-0}" -gt 5 ]; then
  echo "ABORT: utilization ${B}% > 5%; the lease also gates on that. Quiet the desktop or retry."
  exit 3
fi

# --- 1. make sure no stale stack is holding a port ------------------------------------------
# Retirement is cooperative, and a producer wedged inside wait_layer_gpu cannot honour it: it
# keeps its VRAM (it held ~15 GiB after a stop request returned), so the stop is followed by a
# forced kill of any producer that outlives it. The worker needs the same treatment: it binds its
# own port (the router's is one thing, the single engine's config port is another), so a worker
# that outlives a cooperative stop makes the *next* run answer 503 "[Errno 98] Address already in
# use" on its first request, without ever starting a producer - measured 2026-11-09 20:44, where
# run C's forced producer kill left its worker holding the engine port and run D spent 0s.
stack_ports() {
  python3 - <<'PY'
import json
ports = set()
for f in ('rtx.local.json', 'runtime/single-engine.json'):
    try:
        port = int(json.load(open(f)).get('port') or 0)
    except (OSError, ValueError, TypeError):
        continue
    if port:
        ports.add(port)
print(' '.join(str(port) for port in sorted(ports)))
PY
}

port_holders() {          # pids listening on any port in the space-separated list
  local pattern
  pattern="$(printf '%s|' $1 | sed 's/|$//')"
  ss -ltnp 2>/dev/null | awk -v pat=":(${pattern})$" '$4 ~ pat && match($0, /pid=[0-9]+/) {
      print substr($0, RSTART + 4, RLENGTH - 4) }' | sort -u
}

# Match on the command line, not only the port, so a different service on a neighbouring port
# (Strata serves its own UI on 8084) is never killed by this.
stack_holders() {
  local pid
  for pid in $(port_holders "$1"); do
    [ "$pid" = "$$" ] && continue
    if tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -qE 'slstack|llama-dsv41-prefill|ds4-server'; then
      echo "$pid"
    fi
  done
}

PORTS="$(stack_ports)"
stop_stack() {
  "$REPO/scripts/start-dsv41-single.sh" --preset 262k --stop >/dev/null 2>&1 || true
  local pids
  pids="$(pgrep -f 'llama-dsv41-prefill' || true)"
  if [ -n "$pids" ]; then
    echo "producer still alive after cooperative stop: $pids - forcing kill"
    kill -9 $pids 2>/dev/null || true
  fi
  pids="$(stack_holders "$PORTS")"
  if [ -n "$pids" ]; then
    echo "stack worker still holding a port (${PORTS}): $pids - forcing kill"
    kill -9 $pids 2>/dev/null || true
  fi
}
stop_stack
sleep 3
for _ in $(seq 1 10); do
  [ -z "$(stack_holders "$PORTS")" ] && break
  sleep 2
done
if [ -n "$(stack_holders "$PORTS")" ]; then
  echo "ABORT: ports ${PORTS} still bound after --stop; inspect with: ss -ltnp | grep -E ':(${PORTS// /|})'"
  exit 3
fi

# --- 1a. release a hold left by an interrupted run -------------------------------------------
# Ctrl-C kills the engine worker without retiring its lease. An unreleased ACTIVE hold never
# expires by inference, so every later admission is refused until it is released explicitly.
# The release refuses while the card is busy above the engine's own idle gate.
echo "--- stale GPU hold ---"
(cd "$REPO" && timeout 120 python3 -m slstack --config "$REPO/rtx.local.json" release-holds) \
  || echo 'release-holds refused; inspect runtime/state/gpu.holds and runtime/state/leases'

# The request body's quality setting is what decides the bias, so --bias has to be on the wire
# rather than in the config. It also has to be on the wire when the bias is NOT wanted: the value
# lives in runtime/generation/live.json across runs, so a request that omits quality_speed leaves
# whatever the last run published in force. Measured 2026-11-09 21:10, where the control run for a
# bias experiment inherited value=0.5 from the run before it and decoded biased.
with_bias() {
  python3 - "$1" "${bias:-0}" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
body['quality_speed'] = float(sys.argv[2])
print(json.dumps(body))
PY
}
# The bias this run starts from, before any request changes it.
PERSISTED="$(python3 -B -c "
import json
try:
    row = json.load(open('runtime/generation/live.json'))
except (OSError, ValueError):
    print('none'); raise SystemExit(0)
print('value=%s per_phase=%s revision=%s' % (row.get('value'), row.get('per_phase'), row.get('revision')))" 2>/dev/null)"
echo "bias: this run sends quality_speed=${bias:-0} on every request; persisted control is ${PERSISTED:-none}"
if [ -n "$bias" ]; then
  # Store.bias refuses a value above 0 while nothing is bound (slstack/generation.py:128), which
  # would surface as a failed request minutes into the run. Say so before the card is taken.
  if ! python3 -B -c "
import json, sys
sys.exit(0 if json.load(open('rtx.local.json')).get('cache_affinity_spread_file') else 1)"; then
    echo "ABORT: --bias ${bias} needs cache_affinity_spread_file set in rtx.local.json, otherwise"
    echo "  generation.Store.bias raises on the first request. Point it at a calibration written by"
    echo "  --calibrate for this model, e.g. runtime/qualification/dsv41-1017-iq2xxs-spreads.txt."
    exit 4
  fi
fi

# --- 1b. what the ULTRA plan actually needs -------------------------------------------------

# --- 1b. what the ULTRA plan actually needs -------------------------------------------------
# The decode step ledger has to be in the producer's own environment: the launcher drops
# inherited variables (slstack/envvars.py prints "inherited X ignored"), so this probe runs
# on a derived config that adds LLAMADECODE_BYTE_LEDGER=1 to the local one. Everything else
# - paths, ports, budgets - is rtx.local.json unchanged.
TRACE_CONFIG="$REPO/runtime/ds4-first-run.local.json"
python3 - "$REPO/rtx.local.json" "$TRACE_CONFIG" "${calibrate:-}" "${routed}" <<'PY' || exit 1
import json, os, sys
src, dst, calibrate, routed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == '1'
try:
    cfg = json.load(open(src))
except OSError as exc:
    print('cannot read %s: %s' % (src, exc)); raise SystemExit(1)
env = dict(cfg.get('env_overrides') or {}); env['LLAMADECODE_BYTE_LEDGER'] = '1'
if routed:
    # '1' routes from layer 0; the unset default resolves through routed_prefill_choice, which
    # answers 'tail' above 2 048 rows and leaves layers 0-20 reading every expert in the block.
    env['LLAMADECODE_PREFILL_ROUTED'] = '1'
    print('routed prefill on every layer: LLAMADECODE_PREFILL_ROUTED=1')
if calibrate:
    if any(c in calibrate for c in '*?['):
        print('calibrate path must be a plain file name, not a glob: %s' % calibrate); raise SystemExit(1)
    # The producer resolves this against its own working directory, not the repo's, and
    # ds4 aborts() when the open fails (ds4_cuda.cu:1005). Measured 2026-11-09: a relative
    # path cost the whole run - 255 decoded steps whose samples had already passed the
    # 16-per-layer publish gate, then "ds4: cannot write affinity calibration" and abort().
    # So make it absolute and prove it is writable before starting anything.
    target = os.path.abspath(calibrate)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    try:
        handle = open(target, 'xb'); handle.close(); os.unlink(target)
    except OSError as exc:
        print('calibration target %s is not writable: %s' % (target, exc)); raise SystemExit(1)
    env['DS4_CACHE_AFFINITY_CALIBRATE'] = target
    print('calibration target: %s' % target)
cfg['env_overrides'] = env
json.dump(cfg, open(dst, 'w'), indent=2)
print('trace config: %s (LLAMADECODE_BYTE_LEDGER=1%s)' % (dst, ', DS4_CACHE_AFFINITY_CALIBRATE=' + calibrate if calibrate else ''))
PY

# The launcher's --dry-run writes the start config without starting anything, so the
# resolved device bill can be read before a request burns the lease.
"$REPO/scripts/start-dsv41-single-262k.sh" --config "$TRACE_CONFIG" --model "$MODEL" --dry-run >/dev/null 2>&1
python3 -B - <<'PY'
import psutil, sys
sys.path.insert(0, '.')
from slstack.config import load
from slstack.combo_producer import prefill_arguments, pending_job_bytes
cfg = load('runtime/dsv41-single-262k-ultra-cone.local.json')
args, plan = prefill_arguments(cfg, 588, psutil.virtual_memory().available)
graph = int(args['--graph-bytes']) / 2**20
# The producer admits min(job, free - reserve) against its own device plan, so the gate has to
# charge the component bill (provider + cache + graph + hidden + pinned), not the job ceiling.
# The ceiling is only an upper bound: charging it asks for VRAM the plan never needs.
bill = pending_job_bytes(cfg, args) / 2**20
job = int(args['--job-device-bytes']) / 2**20
res = int(args['--gpu-reserve-bytes']) / 2**20
print('plan: graph %.0f MiB, device bill %.0f MiB (job ceiling %.0f MiB), reserve %.0f MiB'
      ' -> combo gate needs %.0f MiB free' % (graph, bill, job, res, bill + res))
print('host: job %.0f MiB, provider %.0f MiB | graph cap key: %s'
      % (int(args['--job-host-bytes']) / 2**20, int(args['--provider-host-bytes']) / 2**20,
         cfg.get('producer_graph_cap_gib')))
PY

# --- 1c. start ------------------------------------------------------------------------------
# The launcher stays attached by default so Ctrl+C is the shutdown; this probe drives the stack
# from the same shell, so it asks for --detach and stops the stack itself at the end.
"$REPO/scripts/start-dsv41-single-262k.sh" --config "$TRACE_CONFIG" --model "$MODEL" --detach || exit 4
# Interrupting this probe must still stop the engine, or its worker and producer keep the card.
trap 'echo "interrupted: stopping the engine"; stop_stack; exit 130' INT TERM
for i in $(seq 1 60); do
  curl -sf -m 3 http://127.0.0.1:8088/health >/dev/null && break
  sleep 2
done
echo "health: $(curl -s -m 5 http://127.0.0.1:8088/health | head -c 400)"

# --- 3. short request (this is what starts the producer; expect minutes) --------------------
# --quick sends one 16-token request and prints a heartbeat while it runs. The producer's
# stderr.log is only written when the session closes, so a silent cold decode reads as a stall;
# the heartbeat and the captured response body make the difference visible. Non-streaming on
# purpose: that is the path this engine has actually served on Linux.
if [ "$quick" -eq 1 ]; then
  echo "--- quick request (16 tokens, heartbeat every 10s) ---"
  # quality_speed 0 pins the expert-affinity bias at zero unless --bias asked for one. An
  # unbound box decodes unbiased and fails at launch instead of mid-decode
  # (see slstack/affinity_spread.py).
  python3 - "$OUT/short-$STAMP.json" "${bias:-0}" <<'PY'
import json, subprocess, sys, threading, time, urllib.error, urllib.request
out = sys.argv[1]
body = json.dumps({"messages": [{"role": "user", "content": "Capital of Portugal, one line."}],
                   "max_tokens": 16, "quality_speed": float(sys.argv[2]) if sys.argv[2] else 0}).encode()
req = urllib.request.Request('http://127.0.0.1:8088/v1/chat/completions',
                             data=body, headers={'content-type': 'application/json'})
start = time.time()
stop = threading.Event()

def beat():
    while not stop.wait(10):
        try:
            reading = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
            used, util = [part.strip() for part in reading.split(',')]
        except Exception:
            used, util = '?', '?'
        print('  … %ds  gpu %s MiB used, %s%% util' % (int(time.time() - start), used, util), flush=True)

threading.Thread(target=beat, daemon=True).start()
try:
    raw = urllib.request.urlopen(req, timeout=1800).read()
except urllib.error.HTTPError as exc:
    print('quick request failed: HTTP %d %s' % (exc.code, exc.read(2000).decode('utf-8', 'replace')))
    raise SystemExit(5)
except Exception as exc:
    print('quick request failed:', repr(exc)); raise SystemExit(5)
finally:
    stop.set()
dt = time.time() - start
d = json.loads(raw)
open(out, 'w').write(json.dumps(d))
usage = d.get('usage') or {}
timings = d.get('timings') or {}
print('quick wall %.1fs  usage %s' % (dt, usage))
if timings:
    print('timings', json.dumps(timings)[:300])
prompt_tokens = usage.get('prompt_tokens') or 0
completion_tokens = usage.get('completion_tokens') or 0
if timings.get('predicted_ms') and completion_tokens:
    per_token = timings['predicted_ms'] / completion_tokens
    print('decode rate %.2f tok/s  (%.0f ms/token)' % (1000.0 / per_token, per_token))
if timings.get('prompt_ms') and prompt_tokens:
    print('prefill rate %.1f tok/s  (cold: includes weight streaming)' %
          (prompt_tokens * 1000.0 / timings['prompt_ms']))
choice = (d.get('choices') or [{}])[0]
print('content', ((choice.get('message') or {}).get('content') or d.get('error') or '')[:200])
PY
else
  echo "--- short request ---"
  T0=$(date +%s)
  curl -s -m 1800 -o "$OUT/short-$STAMP.json" -w 'short wall %.2fs http %{http_code}\n' \
    http://127.0.0.1:8088/v1/chat/completions -H 'content-type: application/json' \
    -d "$(with_bias '{"messages":[{"role":"user","content":"Capital of Portugal, one line."}],"max_tokens":48}')"
  echo "short wall elapsed: $(( $(date +%s) - T0 ))s"
  python3 -c "
import json,sys
d=json.load(open('$OUT/short-$STAMP.json'))
print('usage', d.get('usage'))
c=d.get('choices',[{}])[0]
print('content', (c.get('message') or {}).get('content','')[:300] if 'error' not in d else d['error'])
" 2>/dev/null || head -c 400 "$OUT/short-$STAMP.json"
fi

if [ "$quick" -eq 0 ]; then
# --- 4. timed decode ------------------------------------------------------------------------
echo "--- decode 256 tokens ---"
curl -s -m 1800 -o "$OUT/decode-$STAMP.json" -w 'decode wall %.2fs\n' \
  http://127.0.0.1:8088/v1/chat/completions -H 'content-type: application/json' \
  -d "$(with_bias '{"messages":[{"role":"user","content":"Write 250 words about harbor ledgers."}],"max_tokens":256}')"
python3 - "$OUT/decode-$STAMP.json" <<'PY' || head -c 400 "$OUT/decode-$STAMP.json"
import json, sys
# The 256-token budget has been going entirely into reasoning_content with an empty message and
# finish_reason=length, so "the decode produced 256 tokens" says nothing about whether it reached
# an answer. Print the split, because Step 4's quality check needs to know what was measured.
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print('no usable response body'); raise SystemExit(1)
print('usage', d.get('usage') or {})
choice = (d.get('choices') or [{}])[0]
message = choice.get('message') or {}
def words(text):
    return len((text or '').split())
print('finish', choice.get('finish_reason'), ' reasoning words', words(message.get('reasoning_content')),
      ' answer words', words(message.get('content')))
PY

# --- 5. long prefill (~75K tokens) ----------------------------------------------------------
echo "--- long prefill ---"
python3 - "$OUT/long-$STAMP.json" "${bias:-0}" <<'PY'
import json, sys, time, urllib.error, urllib.request
out = sys.argv[1]
text = open('runtime/bench-50k.txt').read() * 3
print('prompt chars', len(text))
request = {"messages": [{"role": "user", "content":
    text + "\nWhat did the harbor master record first? Answer in one line."}],
    "max_tokens": 64, "quality_speed": float(sys.argv[2])}
body = json.dumps(request).encode()
req = urllib.request.Request('http://127.0.0.1:8088/v1/chat/completions',
                             data=body, headers={'content-type': 'application/json'})
t = time.time()
try:
    raw = urllib.request.urlopen(req, timeout=1800).read()
except urllib.error.HTTPError as exc:
    # The refusal reason is in the body, and losing it is what left three runs' long-prefill
    # 503s undiagnosed: "HTTP Error 503" on its own says nothing about which guard refused.
    print('long request failed: HTTP %d %s' % (exc.code, exc.read(3000).decode('utf-8', 'replace')))
    raise SystemExit(5)
except Exception as exc:
    print('long request failed:', repr(exc)); raise SystemExit(5)
dt = time.time() - t
d = json.loads(raw)
open(out, 'w').write(json.dumps(d))
u = d.get('usage') or {}
pt = u.get('prompt_tokens') or 0
print('long wall %.1fs  usage %s' % (dt, u))
if pt: print('prefill rate %.1f tok/s' % (pt / dt))
c = d.get('choices', [{}])[0]
print('content', ((c.get('message') or {}).get('content') or '')[:200])
PY
fi

# --- 6. capture the engine/producer evidence -----------------------------------------------
echo "--- gpu/ram now ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5
free -g | head -2

ENGINE_LOG="$(ls -t "$REPO"/runtime/logs/dsv41-single-262k-*.log 2>/dev/null | head -1)"
echo "--- engine log: $ENGINE_LOG ---"
grep -nE "DS4X_OPEN|CARRY_RAM_LOCK|DS4COMBO|Engram|expert cache|VRAM|HOST_MEMORY|idle enough|admission|refused|503|error|Error|died|Traceback" \
  "$ENGINE_LOG" | tail -40 | cut -c1-180
cp "$ENGINE_LOG" "$OUT/engine-$STAMP.log"

for j in $(ls -t "$REPO"/runtime/jobs/*.log 2>/dev/null | head -6); do
  echo "--- job log: $j ---"
  grep -nE "DS4X_OPEN|CARRY_RAM_LOCK|DS4COMBO|idle enough|admission|refused|Engram|VRAM|error|died|Traceback" "$j" | tail -25 | cut -c1-180
  tail -8 "$j" | cut -c1-180
done

# The producer's own session log holds the numbers this run is for: admission, the open
# line, expert-cache sizing, prefill rate and the per-token decode steps. It is written when
# the session closes, so this runs after the stop. The steps of a long session can sit in the
# rotated file, so both are read.
# --- 7. release the card -------------------------------------------------------------------
stop_stack
echo "gpu free after stop: $(free_mib) MiB"

PRODUCER_DIR="$(ls -td "$REPO"/runtime/single-engine/ultra-* 2>/dev/null | head -1)"
PRODUCER_LOG="$(ls -t "$REPO"/runtime/single-engine/ultra-*/stderr.log 2>/dev/null | head -1)"
echo "--- producer log: ${PRODUCER_LOG:-none} ---"
[ -n "$PRODUCER_LOG" ] && grep -aE "DS4PERF_GPU_JOB_ADMISSION|DS4X_OPEN|CARRY_RAM_LOCK|EXPERT|dsv41-prefill-experts|STATEFUL_CHUNK_COMPLETE|PROMPT_HEAD_COMPLETE|ABORT" \
  "$PRODUCER_LOG" | tail -10 | cut -c1-200
# Step 1 of docs/DECODE_GAP_PLAN.md: the step budget. Median step time is the number the
# plan's bar is set against (Strata 23.9 tok/s), and the byte lines say where each step's
# footprint came from. Every request gets its own producer directory, so the run's logs are
# every log file written since this stamp. Filter on the files, not the directories: retiring
# an older session touches its directory's mtime without writing to it, so a directory filter
# pulls in previous runs - measured 2026-11-09, where two directories from the 19:28 run were
# listed as part of the 19:44 run because retirement bumped them at 19:45:32.
LOGS=(); while IFS= read -r f; do LOGS+=("$f"); done < <(
  find "$REPO/runtime/single-engine" -maxdepth 2 -type f \
       \( -name 'stderr.log' -o -name 'stderr-rotated.log' \) \
       -newermt "@${STAMP_EPOCH}" 2>/dev/null | sort)
if [ ${#LOGS[@]} -eq 0 ] && [ -n "$PRODUCER_DIR" ]; then
  for f in "$PRODUCER_DIR/stderr.log" "$PRODUCER_DIR/stderr-rotated.log"; do
    [ -f "$f" ] && LOGS+=("$f"); done
fi
[ ${#LOGS[@]} -gt 0 ] && "$REPO/scripts/analyze_dsv41_decode.py" --min-steps 200 "${LOGS[@]}"

echo "evidence in $OUT (stamp $STAMP)"
