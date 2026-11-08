#!/usr/bin/env bash
# One-shot first-run probe for the DSV4.1 single-engine stack.
# Run this while nothing else holds the GPU; it writes evidence under
# runtime/debug/ so the run can be read afterwards.
#   --quick   one 16-token request with a progress heartbeat, then stop: it proves the
#             producer opens, prefills and decodes, and closes the session cleanly so the
#             producer log survives, without spending minutes on the 256-token and
#             ~75K-token probes.
set -uo pipefail
quick=0
if [ "${1:-}" = "--quick" ]; then quick=1; shift; fi
# A producer crash on Linux hands systemd-coredump the whole address space (~22 GiB of
# host staging here), which lands as a multi-gigabyte file under /var/lib/systemd/coredump.
# The stderr logs carry the diagnosis; the core does not.
ulimit -c 0 2>/dev/null || true

REPO="$HOME/workspace/local-llm-playground/streaming-llama-stack"
MODEL="$HOME/workspace/local-llm-playground/ds4-models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
OUT="$REPO/runtime/debug"
cd "$REPO" || exit 1   # step 5 reads runtime/bench-50k.txt relative to the repo
STAMP="$(date +%Y%m%d-%H%M%S)"
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

# --- 1. make sure no stale stack is holding 8088 --------------------------------------------
# Retirement is cooperative, and a producer wedged inside wait_layer_gpu cannot honour it: it
# keeps its VRAM (it held ~15 GiB after a stop request returned), so the stop is followed by a
# forced kill of any producer that outlives it.
stop_stack() {
  "$REPO/scripts/start-dsv41-single.sh" --preset 262k --stop >/dev/null 2>&1 || true
  local pids
  pids="$(pgrep -f 'llama-dsv41-prefill' || true)"
  if [ -n "$pids" ]; then
    echo "producer still alive after cooperative stop: $pids - forcing kill"
    kill -9 $pids 2>/dev/null || true
  fi
}
stop_stack
sleep 3
if ss -ltn 2>/dev/null | grep -q ':8088 '; then
  echo "ABORT: 8088 still bound after --stop; find it with: ss -ltnp | grep 8088"
  exit 3
fi

# --- 1a. release a hold left by an interrupted run -------------------------------------------
# Ctrl-C kills the engine worker without retiring its lease. An unreleased ACTIVE hold never
# expires by inference, so every later admission is refused until it is released explicitly.
# The release refuses while the card is busy above the engine's own idle gate.
echo "--- stale GPU hold ---"
(cd "$REPO" && timeout 120 python3 -m slstack --config "$REPO/rtx.local.json" release-holds) \
  || echo 'release-holds refused; inspect runtime/state/gpu.holds and runtime/state/leases'

# --- 1b. what the ULTRA plan actually needs -------------------------------------------------
# The launcher's --dry-run writes the start config without starting anything, so the
# resolved device bill can be read before a request burns the lease.
"$REPO/scripts/start-dsv41-single-262k.sh" --config "$REPO/rtx.local.json" --model "$MODEL" --dry-run >/dev/null 2>&1
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
"$REPO/scripts/start-dsv41-single-262k.sh" --config "$REPO/rtx.local.json" --model "$MODEL" || exit 4
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
  # quality_speed 0 pins the expert-affinity bias at zero. The reasoning phase's profile bias is
  # 0.35, and ds4x_set_affinity refuses any mean>0 without a spread-calibration file bound to this
  # exact model bytes/experts/layers: presets/dsv41-affinity-spreads.txt is calibrated for a
  # 315838021632-byte V4.1 file, so this 365713686528-byte one is refused as wrong-model.
  python3 - "$OUT/short-$STAMP.json" <<'PY'
import json, subprocess, sys, threading, time, urllib.error, urllib.request
out = sys.argv[1]
body = json.dumps({"messages": [{"role": "user", "content": "Capital of Portugal, one line."}],
                   "max_tokens": 16, "quality_speed": 0}).encode()
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
    -d '{"messages":[{"role":"user","content":"Capital of Portugal, one line."}],"max_tokens":48}'
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
  -d '{"messages":[{"role":"user","content":"Write 250 words about harbor ledgers."}],"max_tokens":256}'
python3 -c "
import json
d=json.load(open('$OUT/decode-$STAMP.json'))
u=d.get('usage') or {}
print('usage', u)
" 2>/dev/null || head -c 400 "$OUT/decode-$STAMP.json"

# --- 5. long prefill (~75K tokens) ----------------------------------------------------------
echo "--- long prefill ---"
python3 - "$OUT/long-$STAMP.json" <<'PY'
import json, sys, time, urllib.request
out = sys.argv[1]
text = open('runtime/bench-50k.txt').read() * 3
print('prompt chars', len(text))
body = json.dumps({"messages": [{"role": "user", "content":
    text + "\nWhat did the harbor master record first? Answer in one line."}],
    "max_tokens": 64}).encode()
req = urllib.request.Request('http://127.0.0.1:8088/v1/chat/completions',
                             data=body, headers={'content-type': 'application/json'})
t = time.time()
try:
    raw = urllib.request.urlopen(req, timeout=1800).read()
except Exception as exc:
    print('long request failed:', exc); raise SystemExit(5)
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
# the session closes, so this runs after the stop.
# --- 7. release the card -------------------------------------------------------------------
stop_stack
echo "gpu free after stop: $(free_mib) MiB"

PRODUCER_LOG="$(ls -t "$REPO"/runtime/single-engine/ultra-*/stderr.log 2>/dev/null | head -1)"
echo "--- producer log: ${PRODUCER_LOG:-none} ---"
[ -n "$PRODUCER_LOG" ] && grep -aE "DS4PERF_GPU_JOB_ADMISSION|DS4X_OPEN|CARRY_RAM_LOCK|EXPERT|dsv41-prefill-experts|STATEFUL_CHUNK_COMPLETE|PROMPT_HEAD_COMPLETE|ABORT" \
  "$PRODUCER_LOG" | tail -10 | cut -c1-200
[ -n "$PRODUCER_LOG" ] && python3 - "$PRODUCER_LOG" <<'PY'
import re, sys
steps = [float(m.group(1)) for m in re.finditer(r'LLAMADECODE_STEP\b[^\n]*step_ms=([0-9.]+)',
                                                open(sys.argv[1], errors='replace').read())]
if steps:
    mean = sum(steps) / len(steps)
    print('decode steps=%d first_ms=%.1f mean_ms=%.1f rate=%.2f tok/s' %
          (len(steps), steps[0], mean, 1000.0 / mean))
else:
    print('no LLAMADECODE_STEP lines: decode did not reach a token, or this producer does not log steps')
PY

echo "evidence in $OUT (stamp $STAMP)"
