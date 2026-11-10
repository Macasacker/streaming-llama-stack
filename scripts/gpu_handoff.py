#!/usr/bin/env python3
"""Hand the card from Strata to the DSV4.1 probe and back, so a measurement costs one command.

The turn-cost plan's measurements need the GPU quiet, and this box spends the card on Strata
between them. Doing it by hand - stop the server, run the probe, paste the log, start the server -
is what made each Step a several-minute interruption. This does the whole window and leaves a
report file:

    scripts/gpu_handoff.py                                    one probe, then Strata back
    scripts/gpu_handoff.py --probe '' --probe 'LLAMADECODE_EXTENT_SLOTS=8'    a matched pair
    scripts/gpu_handoff.py --dry-run                          print the plan, touch nothing

Strata is stopped the way its own server asks to be stopped (SIGTERM reaches `request_stop`, the
engine ends with its parent), and started again with the argv, environment and working directory it
was running with - not with whatever the launcher script would have chosen. It is restored even
when the probes fail, unless --keep-down says otherwise. The card is refused before anything is
stopped if a process other than Strata already holds more than the probe's own idle gate.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

# realpath, not abspath: ~/bin/gpu-handoff is a symlink into the repo, and resolving the symlink
# is what keeps ROOT the repo rather than the directory the symlink happens to sit in.
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
FIRST_RUN = os.path.expanduser('~/bin/ds4-first-run')
LOCK = '/tmp/gpu-handoff.lock'
STRATA_LOG_DIR = os.path.join(ROOT, 'runtime', 'logs')


def gpu_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    # The first device, which is what ds4-first-run's own free_mib reads and what its lease gate
    # and Strata's resident load both target.
    for line in out.splitlines():
        if line.strip():
            return int(line.split(',')[0])
    return 0


def gpu_util_pct():
    """Device utilisation, the second half of the probe's own lease gate.

    `ds4-first-run` refuses a run above 5 % because the lease does, and nvidia-smi samples the
    compositor there too: a probe started at 6 % dies in 0 s and the handoff has stopped Strata
    for nothing (measured 2026-11-10 20:02, where the run aborted on `utilization 6% > 5%`).
    """
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.strip():
            return int(line.split(',')[0])
    return 0


IDLE_UTIL_MAX_PCT = 5


def card_quiet(gate):
    return gpu_used_mib() <= gate and gpu_util_pct() <= IDLE_UTIL_MAX_PCT


def gpu_holders():
    """(pid, MiB, command line) of every compute process on the card."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    holders = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        try:
            cmdline = open('/proc/%d/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode(errors='replace').strip()
        except OSError:
            cmdline = ''
        holders.append((pid, int(parts[1]), cmdline))
    return holders


def stack_processes():
    """Pids of the probe's own stack still running: worker, producer, single engine.

    A producer the probe force-killed can outlive the run holding the decoder, and then serves
    the *next* probe's first request from its own directory - measured 2026-11-10 19:33, where a
    producer aborted on the paging gate and the following run's short request was logged inside
    it, under a different env. Waiting for the card is not enough; the stack has to be gone.
    """
    out = subprocess.run(['pgrep', '-f', r'llama-dsv41-prefill|slstack|ds4-server'],
                         capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def idle_gate_mib():
    try:
        with open(os.path.join(ROOT, 'rtx.local.json')) as handle:
            return float(json.load(handle).get('gpu_idle_max_mib', 2560))
    except (OSError, ValueError):
        return 2560.0


def find_strata(port):
    """The `serve.server` process answering on that port, with what it was started with.

    The argv, environment and working directory are read from /proc rather than re-derived from a
    launcher: the server the owner started by hand has flags no script would add, and restoring a
    different command line would silently change the model, the context or the pack it serves.
    """
    for pid in pgrep_serve(port):
        try:
            return {'pid': pid, 'argv': split_cmdline('/proc/%d/cmdline' % pid),
                    'cwd': os.readlink('/proc/%d/cwd' % pid),
                    'env': read_env('/proc/%d/environ' % pid), 'port': port}
        except OSError:
            continue
    return None


def pgrep_serve(port):
    pids = []
    try:
        out = subprocess.run(['pgrep', '-f', r'serve\.server.*--port\s+%d\b' % port],
                             capture_output=True, text=True).stdout
    except OSError:
        return []
    for line in out.split():
        if line.isdigit() and int(line) != os.getpid():
            pids.append(int(line))
    return pids


def descendants(root_pid):
    """Every process below the Strata server: the engine and its vision helper hold the VRAM,
    and both end with their parent (`PR_SET_PDEATHSIG`). Without them the card looks occupied by
    a stranger, and the handoff refuses a window it was asked to open."""
    children = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open('/proc/%s/status' % entry) as handle:
                for line in handle:
                    if line.startswith('PPid:'):
                        children.setdefault(int(line.split()[1]), []).append(int(entry))
                        break
        except OSError:
            continue
    seen, stack = {root_pid}, [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def read_cmdline(path):
    return open(path, 'rb').read().replace(b'\0', b' ').decode(errors='replace').strip()


def split_cmdline(path):
    return [part.decode(errors='replace') for part in open(path, 'rb').read().split(b'\0') if part]


def read_env(path):
    env = {}
    for item in open(path, 'rb').read().split(b'\0'):
        if not item or b'=' not in item:
            continue
        name, _, value = item.decode(errors='replace').partition('=')
        env[name] = value
    return env


def process_gone(pid):
    """A process the launcher has not reaped yet is a zombie, and a zombie holds no VRAM: the
    producer the probe force-killed stays in /proc until its parent waits for it."""
    try:
        with open('/proc/%d/status' % pid) as handle:
            for line in handle:
                if line.startswith('State:'):
                    return line.split()[1] == 'Z'
    except OSError:
        return True
    return False


def wait_until(what, probe, timeout_s, poll_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if probe():
            return True
        time.sleep(poll_s)
    print('timed out after %.0f s waiting for %s' % (timeout_s, what), file=sys.stderr)
    return False


def stop_strata(spec, timeout_s):
    os.kill(spec['pid'], signal.SIGTERM)
    print('SIGTERM to strata server pid %d (its engine ends with it)' % spec['pid'])
    if not wait_until('strata server %d to exit' % spec['pid'],
                      lambda: process_gone(spec['pid']), timeout_s):
        return False
    return wait_until('the card to fall under the probe idle gate',
                      lambda: card_quiet(idle_gate_mib()), timeout_s)


def start_strata(spec, timeout_s):
    os.makedirs(STRATA_LOG_DIR, exist_ok=True)
    log = os.path.join(STRATA_LOG_DIR, 'strata-restore-%d-%s.log' % (spec['port'], time.strftime('%Y%m%d-%H%M%S')))
    handle = open(log, 'ab')
    env = dict(spec['env'])
    child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdout=handle, stderr=handle,
                             start_new_session=True)
    print('restarted strata: pid %d, same argv from %s, output in %s' % (child.pid, spec['cwd'], log))

    def healthy():
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/health' % spec['port'], timeout=5) as reply:
                body = json.loads(reply.read().decode())
                return reply.status == 200 and body.get('status') == 'ok'
        except (OSError, ValueError):
            return False

    if not wait_until('strata /health on port %d' % spec['port'], healthy, timeout_s):
        return False
    try:
        with urllib.request.urlopen('http://127.0.0.1:%d/health' % spec['port'], timeout=5) as reply:
            print('strata healthy: %s' % reply.read().decode().strip())
    except OSError:
        pass
    return True


def run_probe(extra_env, flags, timeout_s):
    env = dict(os.environ)
    for pair in extra_env:
        name, _, value = pair.partition('=')
        if name:
            env[name] = value
            print('probe env: %s=%s' % (name, value))
    print('probe: %s %s' % (FIRST_RUN, ' '.join(flags)))
    started = time.time()
    result = subprocess.run(['bash', FIRST_RUN] + flags, cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout_s)
    stamp = re.search(r'=== (\d{8}-\d{6}) ===', result.stdout)
    print('probe finished in %.0f s, exit %d, stamp %s' % (time.time() - started, result.returncode,
                                                           stamp.group(1) if stamp else 'unknown'))
    if result.returncode:
        print(result.stdout[-2000:] + '\n' + result.stderr[-2000:], file=sys.stderr)
    return stamp.group(1) if stamp else None, result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe', action='append', default=[],
                        help="extra 'NAME=VALUE' env for one probe run; repeat it for a matched pair "
                             "(use --probe '' for the control)")
    parser.add_argument('--flag', action='append', default=[],
                        help='flag for ds4-first-run, repeatable (default --bias --routed)')
    parser.add_argument('--strata-port', type=int, default=8084)
    parser.add_argument('--no-strata', action='store_true', help="don't stop or restart Strata")
    parser.add_argument('--keep-down', action='store_true', help='leave Strata stopped when done')
    parser.add_argument('--timeout', type=float, default=900, help='per-probe timeout seconds')
    parser.add_argument('--stop-timeout', type=float, default=180,
                        help='seconds to wait for Strata to end and release the card')
    parser.add_argument('--restore-timeout', type=float, default=420)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    flags = args.flag or ['--bias', '--routed']
    probes = args.probe or ['']

    holders = gpu_holders()
    gate = idle_gate_mib()
    strata = None if args.no_strata else find_strata(args.strata_port)
    print('card: %d MiB used, idle gate %.0f MiB, holders %s' %
          (gpu_used_mib(), gate, ', '.join('%d=%d MiB' % (p, m) for p, m, _ in holders)))
    owned = descendants(strata['pid']) if strata else set()
    foreign = [(pid, mib, cmd[:90]) for pid, mib, cmd in holders if mib > gate and pid not in owned]
    if foreign:
        for line in foreign:
            print('refusing: a process that is not Strata holds more than the gate: pid %d, %d MiB, %s' % line,
                  file=sys.stderr)
        return 2
    if strata is None and not args.no_strata:
        print('no strata server on port %d; running the probes without a handoff' % args.strata_port)
    if args.dry_run:
        print('would stop %s' % (strata['argv'] if strata else 'nothing'))
        for extra in probes:
            print('would run: %s %s%s' % (FIRST_RUN, ' '.join(flags), (' with ' + extra) if extra else ''))
        print('would restore with the same argv, then poll /health on port %d' % args.strata_port)
        return 0

    lock = open(LOCK, 'w')
    try:
        fcntl_flock(lock)
    except OSError:
        print('another gpu-handoff is running', file=sys.stderr)
        return 2

    stamps, failures = [], []
    restored = True
    try:
        if strata is not None and not stop_strata(strata, args.stop_timeout):
            print('strata did not release the card; not running probes', file=sys.stderr)
            return 3
        for index, extra in enumerate(probes):
            if not wait_until('the card to be free between probes', lambda: card_quiet(gate), 240):
                failures.append('card still busy before a probe')
                break
            # Only between probes: the first one clears any stale stack itself, and ds4-first-run
            # retires and force-kills holders before it sends anything.
            if index and not wait_until('the previous probe stack to be gone',
                                        lambda: not stack_processes(), 120):
                # Its env is the previous probe's, so anything it serves would be attributed here.
                print('probe stack still running: %s; its log would mix two probes, so the next '
                      'probe is skipped' % ' '.join(str(pid) for pid in stack_processes()),
                      file=sys.stderr)
                failures.append('a producer outlived the previous probe')
                break
            stamp, code = run_probe([pair for pair in extra.split() if pair], flags, args.timeout)
            if stamp:
                stamps.append(stamp)
            if code:
                failures.append('probe exit %d%s' % (code, (' (%s)' % extra) if extra else ''))
        report = ''
        if stamps:
            report = os.path.join(ROOT, 'runtime', 'debug', 'handoff-%s.md' % stamps[0])
            argv = [sys.executable, os.path.join(ROOT, 'scripts', 'ds4_run_metrics.py')]
            for stamp in stamps:
                argv += ['--stamp', stamp]
            subprocess.run(argv + ['--output', report], check=False)
            print('report: %s' % report)
    finally:
        if strata is not None and not args.keep_down:
            # The probe force-kills its producer, and the driver takes a moment to hand the memory
            # back; starting Strata into a card it cannot fill leaves it failing on its own load.
            # Memory only here: Strata needs the card's VRAM back, and a compositor sitting at
            # 6 % must not hold its restore for a minute.
            wait_until('the card to be free before restoring Strata', lambda: gpu_used_mib() <= gate, 120)
            restored = start_strata(strata, args.restore_timeout)
    print('stamps: %s' % ' '.join(stamps))
    print('strata restored: %s' % restored)
    if failures:
        print('problems: %s' % '; '.join(failures), file=sys.stderr)
    return 0 if stamps and restored and not failures else 1


def fcntl_flock(handle):
    import fcntl
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    sys.exit(main())
