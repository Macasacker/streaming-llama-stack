"""Shared short GPU leases. No production service is stopped or adopted."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import psutil
from .journal import LockUnavailable, advisory_lock, atomic_json


class AdmissionRefused(RuntimeError):
    pass


GPU_IDLE_MAX_UTIL_DEFAULT = 5
GPU_UTIL_SAMPLE_SECONDS = .25        # the gap between the two utilization readings


def gpu_sample(cfg):
    """One reading of the selected card as (used MiB, free MiB, utilization %).

    An unreadable reading is a RuntimeError, not a refusal: a broken query must never read as
    somebody else owning the card.
    """
    result = subprocess.run(['nvidia-smi', '--id=' + cfg['gpu_uuid'],
                             '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Cannot certify GPU ownership: ' + result.stderr)
    fields = result.stdout.strip().split(',')
    if len(fields) != 3:
        raise RuntimeError('Cannot certify GPU ownership: unreadable reading ' + result.stdout.strip())
    return [int(x.strip()) for x in fields]


def gpu_holder_line():
    """Who holds device memory, as one line for a refusal message.

    Admission never ejects a process, so a refusal has to name the holder rather than assert
    there is none. A failed query must not turn a refusal into a crash, so the evidence
    degrades to `unknown` here; the release path keeps the strict reading.
    """
    try:
        rows = gpu_holders()
    except (RuntimeError, OSError, ValueError):return 'unknown'
    return ' '.join(f'{row["pid"]} ({row["used_memory_mib"]} MiB)' for row in rows) or 'none'


def gpu_not_idle(cfg):
    """Why the card is not idle enough for exclusive admission, or None when it is.

    Each condition names the threshold it tripped: one message for three different causes is how
    a 7 % reader on the card reads as a memory problem (docs/SESSION_FAILURE_PLAN.md). Memory is
    read once — a holder of gigabytes is not transient. Utilization is the noisiest quantity on
    the card, so it refuses only when a second sample a moment later still exceeds it.
    """
    used, free, busy = gpu_sample(cfg)
    limit = cfg['gpu_idle_max_mib']
    if used > limit:
        return f'{used} MiB used exceeds gpu_idle_max_mib {limit}'
    floor = int(cfg.get('gpu_min_free_mib', 0))
    if free < floor:
        return f'{free} MiB free is below gpu_min_free_mib {floor}'
    ceiling = int(cfg.get('gpu_idle_max_util', GPU_IDLE_MAX_UTIL_DEFAULT))
    if busy > ceiling:
        time.sleep(GPU_UTIL_SAMPLE_SECONDS)
        second = gpu_sample(cfg)[2]
        if second > ceiling:
            return f'{second}% utilization (first sample {busy}%) exceeds gpu_idle_max_util {ceiling}'
    return None


def read_hold_text(path):
    """Wait briefly for another admission's write to land; never use stale holds."""
    for attempt in range(51):
        try:
            return Path(path).read_text(encoding='utf-8-sig')
        except PermissionError:
            if attempt == 50:
                raise
            time.sleep(.02)


def recover_leases(runtime_root, task='DS4STACK'):
    """Recover legacy kill-on-close guardians; retain cooperative owners for inspection.

    A cooperative hold is deliberately left unreleased: an interrupted retirement may still
    leave the producer holding the GPU, and the hold is the evidence an operator reads. The
    explicit release for a dead cooperative guardian is release_dead_cooperative().
    """
    root = Path(runtime_root) / 'state' / 'leases'
    if not root.exists():
        return []
    released = []
    for file in root.glob('*.json'):
        row = json.loads(file.read_text())
        if row.get('task') != task or row.get('released') or row.get('cleanup_uncertain') or row.get('cooperative_retirement'):
            continue
        try:
            proc = psutil.Process(row['guardian_pid'])
            if abs(proc.create_time() - row['guardian_created']) < .01 and proc.is_running():
                continue
        except psutil.NoSuchProcess:
            pass
        # On Windows the guardian handle closes its Job and terminates all GPU descendants.
        # Recovering an active hold never kills an unrelated process or guesses expiry.
        with Path(row['hold_file']).open('a', encoding='utf-8') as stream:
            stream.write(f'\nRELEASED: {task} {row["id"]}; recovered dead guardian {row["guardian_pid"]}; {datetime.now(timezone.utc).isoformat()}\n')
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(file, {**row, 'released': True, 'recovered': True})
        released.append(row['id'])
    return released


def recover_runtime_leases(runtime_root, task='DS4STACK'):
    """Recover this configured owner's dead guardians in its two known runtimes."""
    root = Path(runtime_root)
    return [identity for location in (root, root / 'single-engine')
            for identity in recover_leases(location, task)]


def release_dead_cooperative(runtime_root, task='DS4STACK', *, holders=None):
    """Operator-initiated release of a cooperative hold whose guardian process is gone.

    recover_leases keeps cooperative holds because an interrupted retirement may still leave
    the producer holding the GPU. This path is explicit and refuses while the caller's holder
    probe still reports the card busy, so releasing the ledger can only follow a card that is
    actually free.
    """
    root = Path(runtime_root) / 'state' / 'leases'
    if not root.exists():
        return []
    released = []
    for file in root.glob('*.json'):
        row = json.loads(file.read_text())
        if row.get('task') != task or row.get('released') or row.get('cleanup_uncertain'):
            continue
        if not row.get('cooperative_retirement'):
            continue          # recover_leases owns the kill-on-close guardians
        try:
            proc = psutil.Process(row['guardian_pid'])
            if abs(proc.create_time() - row['guardian_created']) < .01 and proc.is_running():
                continue      # a live cooperative owner is inspected, not released
        except psutil.NoSuchProcess:
            pass
        if holders is not None and holders():
            raise AdmissionRefused('Refusing to release a cooperative hold while the GPU is still held')
        with Path(row['hold_file']).open('a', encoding='utf-8') as stream:
            stream.write(f'\nRELEASED: {task} {row["id"]}; operator released dead cooperative guardian '
                         f'{row["guardian_pid"]}; {datetime.now(timezone.utc).isoformat()}\n')
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(file, {**row, 'released': True, 'operator_released': True})
        released.append(row['id'])
    return released


def release_runtime_cooperative(runtime_root, task='DS4STACK', *, holders=None):
    """Release dead cooperative holds in this configured owner's two known runtimes."""
    root = Path(runtime_root)
    return [identity for location in (root, root / 'single-engine')
            for identity in release_dead_cooperative(location, task, holders=holders)]


def gpu_holders():
    """Compute processes the driver reports, as evidence that the card is still held.

    A failed query is refused rather than read as an empty card: releasing a hold on the
    strength of a missing measurement is exactly the guess this ledger exists to avoid.
    """
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError('Cannot measure GPU holders: ' + (result.stderr or result.stdout).strip()[:200])
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append(dict(pid=int(parts[0]), used_memory_mib=parts[1]))
    return rows


def active_holds(text):
    """Unreleased ledger holds, parsed by the vendored Studio priority_holds.

    The RTX resource is the only one this fork coordinates on: the Mac receiver was
    deleted with macOS, so no MAC hold line can appear in the ledger.
    """
    from .native_broker import vendor
    vendor()
    from studio.job_guard import priority_holds
    return priority_holds(text, ('*',), resource='RTX')


def reservation_admission(lines, budget_gib, floor_gib, *, available=None, owned_rss=None):
    """Owner-approved KIT11FIX CPU exception; other/new/larger holds preempt."""
    blocking=[];reserved=0
    for line in lines:
        original_cpu=(line.startswith('ACTIVE: KIT11FIX Windows native plugin build;') and
             'host budget 3 GiB; physical RAM reserve 6 GiB;' in line and
             'no GPU or Studio restart requested.' in line and
             not re.search(r'exclusive|GPU hold',line,re.I))
        # The current qualification names the same CPU-only reservation with
        # a smaller2GiB job cap. Keep the owner's full3+6GiB reservation rather
        # than treating that CPU build as a GPU hold or reducing its allowance.
        # An unrecognised, GPU, or larger reservation still preempts immediately.
        current_cpu=re.fullmatch(
            r'ACTIVE: KIT11FIX [a-z][a-z0-9-]*; expires [0-9T:.+Z-]+; '
            r'Windows-only CPU build/test, host budget ([0-9]+(?:\.[0-9]+)?) GiB, reserve ([0-9]+(?:\.[0-9]+)?) GiB; '
            r'yield RAM for bounded [A-Z][A-Z0-9]* window; no Studio restart',line)
        current_cpu=(current_cpu is not None and
                     0<float(current_cpu[1])<=3 and 0<float(current_cpu[2])<=6)
        if original_cpu or current_cpu:reserved+=9*2**30
        else:blocking.append(line)
    available=psutil.virtual_memory().available if available is None else available
    floor=max(floor_gib,2)*2**30 if reserved else floor_gib*2**30
    # At admission reserve the full budget. During execution the extra
    # outstanding-allocation rule applies to the owner's CPU reservation.
    # With no reservation, the shared guard enforces actual RSS/commit/floor;
    # each subsequent engine/locked-state allocation also has its own admission.
    outstanding=max(0,budget_gib*2**30-(owned_rss or 0))
    required=reserved+floor+(outstanding if reserved or owned_rss is None else 0)
    if available<required:
        blocking.append(f'RAM reservation: available={available}, required={required}, owned_rss={owned_rss}')
    return blocking,dict(available=available,required=required,foreign_reserved_bytes=reserved,
                         own_budget_bytes=budget_gib*2**30,owned_rss=owned_rss,floor_bytes=floor)


def hold_admission(lines, *, budget_gib=0, floor_gib=2, available=None):
    """Explicit CPU-only reservations share the GPU only with reserved RAM intact."""
    blocking, reserved = [], 0.0
    for line in lines:
        budget = re.search(r'host budget ([\d.]+) GiB', line)
        reserve = re.search(r'physical RAM reserve ([\d.]+) GiB', line)
        if (budget and reserve and 'no GPU or Studio restart requested' in line
                and not re.search(r'exclusive|GPU hold', line, re.I)):
            reserved += float(budget[1]) + float(reserve[1])
        else:
            blocking.append(line)
    floor = max(floor_gib, reserved + 2) if reserved else floor_gib
    available = psutil.virtual_memory().available if available is None else available
    if reserved and available < (budget_gib + floor) * 2**30:
        blocking.append(f'CPU reservations require {budget_gib + floor:g} GiB available; '
                        f'available={available / 2**30:.3f} GiB')
    return blocking


class HoldWatcher:
    """Fail closed on a new foreign hold; unchanged ledgers cost one stat call."""
    def __init__(self, path, identity, *, task='DS4STACK'):
        self.path, self.identity = Path(path), identity
        self.task = task
        self.stamp = None
        self.blocking = None
        self.foreign = []

    def reason(self):
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self.stamp:
            if stat.st_size > 16 * 1024**2:
                raise RuntimeError('Shared hold ledger exceeds its observation budget')
            blocking = []
            for line in active_holds(read_hold_text(self.path)):
                match = re.match(r'^ACTIVE:\s+(\S+)\s+(\S+)', line.strip())
                if not match or (match[1], match[2].rstrip(';')) != (self.task, self.identity):
                    blocking.append(line)
            self.foreign = blocking
            self.stamp = stamp
        self.blocking = '\n'.join(hold_admission(self.foreign)) or None
        return self.blocking


@contextmanager
def gpu_lease(cfg):
    task = cfg.get('task', 'DS4STACK')
    if not isinstance(task, str) or re.fullmatch('[A-Z][A-Z0-9]{0,31}', task) is None:
        raise ValueError('GPU lease task must be a bounded coordination identifier')
    hold = Path(cfg['hold_file']) if cfg.get('hold_file') else Path(cfg['runtime_root']) / 'state' / 'gpu.holds'
    parent = cfg.get('_gpu_parent_lease')
    if parent:
        # Only a live process ancestor may delegate its existing reservation.
        # Children never release it or claim another GPU window inside it.
        # The timestamp is a reservation estimate, not a wall-clock kill switch.
        # Live ancestor identity, the unreleased hold and the independent
        # progress/RAM/paging guards remain authoritative during long work.
        record = Path(parent)
        row = json.loads(record.read_text(encoding='utf-8'))
        owner = next((p for p in [psutil.Process(), *psutil.Process().parents()]
                      if p.pid == row.get('guardian_pid')), None)
        if (row.get('task') != task or row.get('released') is not False
                or row.get('cleanup_uncertain') or owner is None
                or abs(owner.create_time() - row.get('guardian_created', 0)) >= .01
                or Path(row.get('hold_file', '')).resolve() != hold.resolve()
                or record.name != str(row.get('id')) + '.json'
                or cfg['host_budget_gib'] > row.get('host_budget_gib', 0)
                or cfg['min_available_gib'] < row.get('min_available_gib', float('inf'))):
            raise AdmissionRefused('Inherited GPU reservation is not a live compatible ancestor lease')
        lines = active_holds(read_hold_text(hold))
        own = [line for line in lines if re.match(
            r'^ACTIVE:\s+' + re.escape(task) + r'\s+' + re.escape(row['id']) + r'(?:;|\s|$)', line)]
        foreign = [line for line in lines if line not in own]
        if len(own) != 1 or hold_admission(foreign, budget_gib=cfg['host_budget_gib'],
                                         floor_gib=cfg['min_available_gib']):
            raise AdmissionRefused('Inherited GPU reservation is absent or a foreign hold blocks it')
        yield row['id']
        return
    if not cfg.get('hold_file'):
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.touch(exist_ok=True)
    # The advisory lock serialises admission among instances of this stack.
    # Other agents' leases are respected via the shared append-only hold file.
    with hold.open('r+', encoding='utf-8-sig') as stream:
        try:
            with advisory_lock(stream):
                lines = stream.read()
                active = hold_admission(active_holds(lines), budget_gib=cfg['host_budget_gib'],
                                        floor_gib=cfg['min_available_gib'])
                if active:
                    raise AdmissionRefused('RTX admission refused: ACTIVE holds exist: ' + '\n'.join(active))
                # Other processes hold device memory (a serve process, a browser GPU process),
                # so the selected GPU's own baseline decides, not the absence of clients.
                reason = gpu_not_idle(cfg)
                if reason:
                    raise AdmissionRefused('RTX is not idle enough for exclusive admission; no other process was ejected: '
                                           + reason + '; holding the card: ' + gpu_holder_line())
                identity = uuid.uuid4().hex
                expiry = datetime.now(timezone.utc) + timedelta(minutes=cfg['lease_minutes'])
                lease_path = Path(cfg['runtime_root']) / 'state' / 'leases' / (identity + '.json')
                lease_row = {'task': task, 'id': identity, 'hold_file': str(hold),
                             'guardian_pid': os.getpid(), 'guardian_created': psutil.Process().create_time(),
                             'host_budget_gib': cfg['host_budget_gib'],
                             'min_available_gib': cfg['min_available_gib'], 'expires': expiry.isoformat(),
                             'released': False,
                             # Cooperative trees survive a lost guardian. Its death
                             # alone is never evidence that the GPU owner retired.
                             'cooperative_retirement': True}
                atomic_json(lease_path, lease_row)
                line = f"\nACTIVE: {task} {identity}; exclusive configured GPU; expires {expiry.isoformat()}; host RSS {cfg['host_budget_gib']} GiB; floor {cfg['min_available_gib']} GiB\n"
                stream.seek(0, 2)
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except LockUnavailable as error:
            raise AdmissionRefused('RTX admission is already in progress: ' + str(error)) from error
    uncertain=False
    try:
        yield identity
    except BaseException as error:
        uncertain=bool(getattr(error,'owned_cleanup_uncertain',False))
        if uncertain:atomic_json(lease_path,{**lease_row,'cleanup_uncertain':True})
        raise
    finally:
        if not uncertain:
            with hold.open('a', encoding='utf-8') as stream:
                stream.write(f'\nRELEASED: {task} {identity}; {datetime.now(timezone.utc).isoformat()}\n')
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(lease_path, {**lease_row, 'released': True})
