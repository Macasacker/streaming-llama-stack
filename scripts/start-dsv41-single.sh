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
#   --progress           one turn progress line per second on the engine's output, followed in
#                        this window while attached (slstack/turn_progress.py; default on)
#   --no-progress        turn those lines off
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
progress=
follow=
while [ $# -gt 0 ]; do
    case "$1" in
        --preset) preset=$2; shift 2 ;;
        --config) config=$2; shift 2 ;;
        --model) model=$2; shift 2 ;;
        --keep) keep=$2; shift 2 ;;
        --webui) webui=1; shift ;;
        --vision) vision=1; shift ;;
        --progress) progress=true; shift ;;
        --no-progress) progress=false; shift ;;
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
if [ -n "$progress" ]; then
    if [ "$progress" = true ]; then gen_args="$gen_args --progress"; else gen_args="$gen_args --no-progress"; fi
fi
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

# The stack's two listeners: the router's port and the single engine worker's own port. Matching the
# command line as well as the port is what keeps another service - Strata's own server on 8084, say -
# out of the cleanup.
stack_listeners() {
    PYTHONDONTWRITEBYTECODE=1 "$python" -B - "$1" "$start_config" "$runtime/single-engine.json" <<'PY'
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

# A cooperative stop is not always honoured. A worker whose producer is wedged inside wait_layer_gpu
# keeps its port: measured 2026-11-09 21:34, where --stop answered {"stopping": true}, the router
# exited, the old engine worker went on holding 8089, and the first request to the next stack came
# back "[Errno 98] Address already in use" four times. Clear what is left before starting, so a
# start either gets a clean stack or reports what it had to take.
clear_leftovers() {
    pids=$(pgrep -f 'llama-dsv41-prefill' || true)
    if [ -n "$pids" ]; then
        echo "Leftover producer processes: $pids - forcing kill"
        kill -9 $pids 2>/dev/null || true
    fi
    stack_listeners kill
    holders="$(stack_listeners list)"
    waited=0
    while [ -n "$holders" ] && [ "$waited" -lt 20 ]; do
        sleep 1; waited=$((waited + 1)); holders="$(stack_listeners list)"
    done
    if [ -n "$holders" ]; then
        echo "Still holding the stack ports after 20s: $holders"
        echo "  Inspect: ss -ltnp | grep -E ':(${port}|8089)'   and   curl -s -X POST $url/v1/stack/shutdown -d '{}'"
    fi
}

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

clear_leftovers

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

# The engine's turn progress lines (slstack/turn_progress.py) go to the engine server's own log,
# not to this launcher's: the router guards the engine worker, and that worker guards the server.
# Following the newest one puts the turn lines in this window beside the launcher's output.
follow_engine_log() {
    newest=$(ls -t "$runtime/single-engine"/server-*.log 2>/dev/null | head -1)
    if [ -z "$newest" ]; then
        echo "No engine log under $runtime/single-engine yet; the turn lines will appear once the engine server writes one."
        return 0
    fi
    echo "Turn progress: following $newest from now on (what it already holds stays in the file)."
    tail -f -n 0 "$newest" &
    follow=$!
}
stop_following() {
    if [ -n "$follow" ] && kill -0 "$follow" 2>/dev/null; then kill "$follow" 2>/dev/null || true; fi
    follow=
}
nohup env PYTHONDONTWRITEBYTECODE=1 sh -c "$command" >"$log" 2>&1 &
pid=$!
echo "Started DSV4.1 single engine ($preset) pid $pid; log: $log"

# stop_owned waits on the start process's pid, so it is defined after that process exists.
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
    clear_leftovers
    stop_following
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
    echo "Turn progress (if on): $(ls -t "$runtime/single-engine"/server-*.log 2>/dev/null | head -1 || echo "no engine log yet")"
    exit 0
fi

follow_engine_log

# Stay attached so Ctrl+C is the shutdown. The server itself is still a detached child (nohup), so
# closing the terminal without signalling does not take it down; interrupting this launcher does.
echo "Attached: Ctrl+C stops the stack and releases the GPU. Ctrl+Z then 'bg' detaches instead."
while kill -0 "$pid" 2>/dev/null; do sleep 1; done
stop_following
echo "The start process exited on its own; the stack is stopped."
