#!/usr/bin/env bash
# Step 5 measurement: what the DSpark drafter actually buys, on the native ds4-server path.
#
# The producer's ds4x decode ABI has no drafter entry (docs/DECODE_GAP_PLAN.md Step 5), so the
# only way to read an acceptance rate today is engines/ds4/ds4-server with --dspark, which is a
# different serving shape: its own SSD expert streaming, no LLAMADECODE_STEP_BYTES ledger. This
# is a measurement of the drafter, not a change to what the stack serves.
#
#   scripts/ds4_dspark_probe.sh              # drafter on
#   scripts/ds4_dspark_probe.sh --no-draft   # same run without the drafter, for the baseline
#
# --ssd-streaming and --mtp-model are refused by ds4.c:73341 unless DS4_ALLOW_MTP_STREAMING=1,
# which this sets: the fd-swap machinery for a second model file exists and the comment there
# records why the gate is now opt-in-overridable. DS4_DSPARK_SPEC_LOG=1 makes every verify cycle
# print drafted/accepted to stderr, which is the number this probe exists to collect.
set -uo pipefail

REPO="$HOME/workspace/local-llm-playground/streaming-llama-stack"
MODEL="$HOME/workspace/local-llm-playground/ds4-models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
SIDECAR="$HOME/workspace/local-llm-playground/ds4-models/dspark-DeepSeek-V4.1-Flash-1017-mtp-q8.gguf"
PORT=8090
CTX=${CTX:-131072}
MAX_TOKENS=${MAX_TOKENS:-256}
EXPERT_CACHE=${EXPERT_CACHE:-24GB}
DRAFT=1
OUT="$REPO/runtime/debug"

for arg in "$@"; do
  case "$arg" in
    --no-draft) DRAFT=0 ;;
    *) echo "unknown option: $arg (use --no-draft)" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 1
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/dspark-$STAMP-server.log"
RESP="$OUT/dspark-$STAMP-response.json"
[ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; exit 1; }
if [ "$DRAFT" -eq 1 ] && [ ! -f "$SIDECAR" ]; then
  echo "FATAL: repacked sidecar missing. Build it with:"
  echo "  python3 scripts/repack_dsv41_dspark_sidecar.py \\"
  echo "    ~/workspace/qwen38-gyro/Strata-data/models/deepseek-v4.1-flash-shared/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \\"
  echo "    $MODEL --ds4-txt ~/workspace/qwen38-gyro/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/ds4.txt \\"
  echo "    --out $SIDECAR"
  exit 1
fi

# One model at a time on this card: 30 GiB is the whole device.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "gpu used=${USED} MiB free=${FREE} MiB"
if [ "${FREE:-0}" -lt 24000 ]; then
  echo "ABORT: ${FREE} MiB free is not enough for this serve. Stop the gyro-m serve first:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5
  exit 2
fi
pgrep -f 'llama-dsv41-prefill|ds4-server' >/dev/null && {
  echo "ABORT: a stack or ds4-server is already running:"
  pgrep -af 'llama-dsv41-prefill|ds4-server' | head -5
  echo "Stop the stack first: scripts/start-dsv41-single.sh --preset 262k --stop"
  exit 2
}

SERVER=(engines/ds4/ds4-server --cuda -m "$MODEL" --ctx "$CTX" --host 127.0.0.1 --port "$PORT"
        --ssd-streaming --ssd-streaming-cache-experts "$EXPERT_CACHE")
if [ "$DRAFT" -eq 1 ]; then
  SERVER+=(--dspark --mtp-model "$SIDECAR" --dspark-confidence 0.65)
fi
export DS4_ALLOW_MTP_STREAMING=1 DS4_DSPARK_SPEC_LOG=1
ulimit -c 0 2>/dev/null || true

echo "--- starting (${SERVER[*]}) ; log $LOG ---"
(cd engines/ds4 && exec "$REPO/${SERVER[0]}" "${SERVER[@]:1}") >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping ds4-server $SERVER_PID"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT INT TERM

# No /health on ds4-server; /v1/models answers only once the model is open.
for i in $(seq 1 120); do
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early:"; tail -30 "$LOG"; exit 3; }
  sleep 5
done
curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "ABORT: /v1/models never answered; last log lines:"; tail -30 "$LOG"; exit 3; }
grep -aE "mtp/dspark|support|DSpark|ssd|streaming|expert" "$LOG" | head -12

echo "--- request: $MAX_TOKENS tokens ---"
T0=$(date +%s)
curl -s -m 3600 -o "$RESP" -w 'wall %.2fs http %{http_code}\n' \
  "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write 250 words about harbor ledgers.\"}],\"max_tokens\":$MAX_TOKENS,\"stream\":false}"
WALL=$(( $(date +%s) - T0 ))
python3 - "$RESP" "$WALL" "$LOG" "$DRAFT" <<'PY'
import json, re, sys
resp, wall, log, draft = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == '1'
try:
    data = json.load(open(resp))
except Exception as exc:
    print('no JSON response (%r); server log tail:' % exc)
    print(open(log, errors='replace').read()[-2000:])
    raise SystemExit(4)
usage = data.get('usage') or {}
completion = usage.get('completion_tokens') or 0
print('usage %s  wall %ds  %.2f tok/s' % (usage, wall,
                                          completion / wall if wall and completion else 0))
cycles = []
for line in open(log, errors='replace'):
    match = re.search(r'DSpark V4\.1 cycle pos=\d+ drafted=(\d+) accepted=(\d+) replacement=(-?\d+)', line)
    if match:
        cycles.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
if not cycles:
    print('no DSpark cycle lines' + ('' if draft else ' (drafter was off, as asked)'))
    raise SystemExit(0)
drafted = sum(d for _, d, _ in cycles)
accepted = sum(a for _, _, a in cycles)
tokens = sum(1 + a for _, _, a in cycles)
full = sum(1 for _, d, a in cycles if a == d)
print('cycles %d  drafted %d  accepted %d  acceptance %.1f %%  full accepts %d (%.1f %%)'
      % (len(cycles), drafted, accepted, 100.0 * accepted / drafted if drafted else 0,
         full, 100.0 * full / len(cycles)))
print('tokens from draft cycles %d; per cycle %.2f tokens for %.2f drafted'
      % (tokens, tokens / len(cycles), drafted / len(cycles)))
print('Compare the two runs at the same max_tokens: drafter tok/s / no-draft tok/s is what the')
print('drafter buys, and acceptance is what decides whether ds4x should grow a verify entry.')
PY
