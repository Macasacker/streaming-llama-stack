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
#   --detach             start and return; the stack keeps running until --stop
#                        (default: stay attached, Ctrl+C stops the stack and releases the GPU)
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
detach=0
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
        --detach) detach=1; shift ;;
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

# The stack's two listeners: the router's port and the single engine worker's own port. Killing by
# port plus command line (rather than a broad pattern) is what keeps another service - Strata's own
# server on 8084, say - out of the cleanup.
stack_listeners() {
    PYTHONDONTWRITEBYTECODE=1 "$python" -B - "$1" "$runtime/single-engine.json" <<'PY'
import json, os, re, subprocess, sys

action = sys.argv[1]
ports = set()
for path in sys.argv[2:]:
    try:
        port = int(json.load(open(path)).get('port') or 0)
    except (OSError, ValueError, TypeError):
        continue
    if port:
        ports.add(port)
listed = []
for line in subprocess.run(['ss', '-ltnp'], capture_output=True, text=True).stdout.splitlines():
    fields = line.split()
    if len(fields) < 5 or not any(fields[3].endswith(':' + str(port)) for port in ports):
        continue
    match = re.search(r'pid=(\d+)', line)
    if not match:
        continue
    pid = int(match.group(1))
    try:
        cmd = open('/proc/%d/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        continue
    if not any(own in cmd for own in ('slstack', 'llama-dsv41-prefill', 'ds4-server')):
        continue
    listed.append(pid)
    if action == 'kill':
        try:
            os.kill(pid, 9)
        except OSError:
            pass
if listed:
    print('%s stack listeners on %s: %s' % (action.capitalize(), ' '.join(str(p) for p in sorted(ports)),
                                            ' '.join(map(str, sorted(set(listed))))))
PY
}

stop_owned() {
    # Cooperative first: the start process retires the engine worker and its producer, and that is
    # what releases the GPU hold. A producer wedged inside wait_layer_gpu cannot honour it, which is
    # why the forced cleanup follows instead of trusting the shutdown call.
    curl -sS -m 15 -X POST "$url/v1/stack/shutdown" -H 'content-type: application/json' -d '{}' >/dev/null 2>&1 || true
    waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 45 ]; do sleep 1; waited=$((waited + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        echo "start process still alive after 45s - stopping it"
        kill "$pid" 2>/dev/null || true
    fi
    pids="$(pgrep -f 'llama-dsv41-prefill' || true)"
    if [ -n "$pids" ]; then
        echo "producer still alive: $pids - forcing kill"
        kill -9 $pids 2>/dev/null || true
    fi
    stack_listeners kill
    PYTHONDONTWRITEBYTECODE=1 "$python" -B -m slstack --config "${config:-$root/config.example.json}" release-holds 2>&1 | tail -2 \
        || echo 'release-holds refused; inspect runtime/state/gpu.holds'
    echo "GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) MiB"
}

# Installed before the readiness wait: interrupting a stack that never finished starting is the
# case that used to leave its worker and producer holding the card with nobody attached to stop
# them (measured 2026-11-09, where a hold from a killed producer refused the next run's requests).
trap 'echo; echo "interrupted: stopping the stack"; stop_owned; exit 130' INT TERM

# Wait while the launcher is alive and the router is not yet answering (no fixed deadline).
while kill -0 "$pid" 2>/dev/null; do
    if curl -sS -m 2 "$url/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
if ! curl -sS -m 2 "$url/health" >/dev/null 2>&1; then
    tail -n 40 "$log" 2>/dev/null || true
    echo "The stack exited before its router answered; see $log" >&2
    stop_owned
    exit 1
fi
echo "Router ready: $url"
echo "Health:   curl $url/health"
echo "Shutdown: $0 --preset $preset --stop   (or POST $url/v1/stack/shutdown with body {})"

if [ "$detach" = 1 ]; then
    echo "Detached as requested: this shell returns and the stack keeps running until --stop."
    exit 0
fi

# Stay attached so Ctrl+C is the shutdown. The server itself is still a detached child (nohup), so
# closing the terminal without signalling does not take it down; interrupting this launcher does.
echo "Attached: Ctrl+C stops the stack and releases the GPU. Ctrl+Z then 'bg' detaches instead."
while kill -0 "$pid" 2>/dev/null; do sleep 1; done
echo "The start process exited on its own; the stack is stopped."
