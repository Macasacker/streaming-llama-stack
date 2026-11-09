#!/bin/sh
# Start the DSV4.1 single engine (ULTRA exact-cone prefill + in-process decode) through
# `slstack start`, detached. Settings come from scripts/dsv41_single.py.
#
#   --config <file>      local paths/ports/budgets (default: my.local.json in the clone)
#   --model <gguf>       model path, overrides "model" in the config
#   --webui              also serve the llama.cpp web UI
#   --vision             native image prefill
#   --dry-run            print the resolved start config, start nothing
#   --stop               ask the running stack to shut down gracefully
#   --preset 262k|1m     required
#   --keep 2541|140      dependency cone: 2541 exact cone (default), 140 approximate YOCO
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

preset=
config=
model=
keep=2541
webui=0
vision=0
dry_run=0
stop=0
while [ $# -gt 0 ]; do
    case "$1" in
        --preset) preset=$2; shift 2 ;;
        --config) config=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --keep) keep=$2; shift 2 ;;
        --webui) webui=1; shift ;;
        --vision) vision=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --stop) stop=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$preset" in 262k|1m) ;; *) echo 'usage: --preset 262k|1m' >&2; exit 2 ;; esac
case "$keep" in 2541|140|0) ;; *) echo '--keep must be 2541, 140 or 0' >&2; exit 2 ;; esac

python=${SLSTACK_PYTHON:-python3}
if [ -z "$config" ] && [ -f "$root/my.local.json" ]; then config=$root/my.local.json; fi

case "$keep" in
    2541) suffix=-ultra-cone; profile=ultra-cone ;;
    140) suffix=-ultra-yoco; profile=ultra-yoco ;;
    *) suffix=; profile=ultra ;;
esac
start_config="$root/runtime/dsv41-single-$preset$suffix.local.json"

gen_args="-B $root/scripts/dsv41_single.py --preset $preset --write $start_config --profile $profile"
[ "$vision" = 1 ] && gen_args="$gen_args --vision"
[ -n "$config" ] && gen_args="$gen_args --config $config"
[ -n "$model" ] && gen_args="$gen_args --model $model"
[ "$dry_run" = 1 ] && gen_args="$gen_args --dry-run"

PYTHONDONTWRITEBYTECODE=1 "$python" $gen_args

read_config() {
    PYTHONDONTWRITEBYTECODE=1 "$python" -B -c \
        'import json,sys; from slstack.config import load; print(load(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}
cd "$root"
bind_host=$(read_config config.example.json host)
port=$(read_config config.example.json port)
runtime=$(read_config config.example.json runtime_root)
if [ -f "$start_config" ]; then
    v=$(read_config "$start_config" host); [ -n "$v" ] && bind_host=$v
    v=$(read_config "$start_config" port); [ -n "$v" ] && port=$v
    v=$(read_config "$start_config" runtime_root); [ -n "$v" ] && runtime=$v
fi
case "$runtime" in
    /*) ;;
    *) runtime="$root/$runtime" ;;
esac
url="http://$bind_host:$port"

if [ "$stop" = 1 ]; then
    curl -sS -X POST "$url/v1/stack/shutdown" -H 'content-type: application/json' -d '{}'
    echo "Shutdown requested at $url; the start process exits once the engine has stopped."
    exit 0
fi

start_args=start
[ "$webui" = 1 ] && start_args="$start_args --webui"
command="$python -B $root/scripts/bootstrap.py --config $start_config $start_args"

if [ "$dry_run" = 1 ]; then
    echo "Start config: $start_config"
    echo "Would run (detached, from $root): $command"
    echo "Router / web UI: $url"
    exit 0
fi

# A producer that was force-killed leaves an ACTIVE hold that never expires by inference, and the
# next admission refuses while any ACTIVE hold exists. Measured 2026-11-09 21:28: the first request
# to a freshly started stack answered "RTX admission refused: ACTIVE holds exist: ACTIVE: DS4STACK
# fa4d5b6a…" because the previous run's producer had been killed rather than retired. release-holds
# only takes holds whose guardian process is gone and refuses while the card is busy above the idle
# gate, so a live stack keeps its own hold; a refusal is printed, not fatal, because the admission
# below reports the real reason.
hold_config=${config:-$root/config.example.json}
if holds=$(PYTHONDONTWRITEBYTECODE=1 "$python" -B -m slstack --config "$hold_config" release-holds 2>&1); then
    echo "GPU holds: $(printf '%s\n' "$holds" | tail -1 | cut -c1-200)"
else
    echo "release-holds refused: $(printf '%s\n' "$holds" | tail -2 | tr '\n' ' ')"
    echo "  If a dead holder's ACTIVE line is in $runtime/state/gpu.holds, free the card and rerun:"
    echo "  $python -B -m slstack --config $hold_config release-holds"
fi

logs="$runtime/logs"
mkdir -p "$logs"
log="$logs/dsv41-single-$preset-$(date +%Y%m%d-%H%M%S).log"
nohup env PYTHONDONTWRITEBYTECODE=1 sh -c "$command" >"$log" 2>&1 &
pid=$!
echo "Started DSV4.1 single engine ($preset) pid $pid; log: $log"

# Wait while the launcher is alive and the router is not yet answering (no fixed deadline).
while kill -0 "$pid" 2>/dev/null; do
    if curl -sS -m 2 "$url/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
if ! curl -sS -m 2 "$url/health" >/dev/null 2>&1; then
    tail -n 40 "$log" 2>/dev/null || true
    echo "The stack exited before its router answered; see $log" >&2
    exit 1
fi
echo "Router ready: $url"
echo "Health:   curl $url/health"
echo "Shutdown: $0 --preset $preset --stop   (or POST $url/v1/stack/shutdown with body {})"
