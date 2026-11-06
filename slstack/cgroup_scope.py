"""Kernel-enforced containment for one guarded process tree.

The retired Windows Job Object gave a guardian two things Linux has no per-process
equivalent of: a memory ceiling that applies from the child's first instruction, and
containment of descendants born between the guardian's ownership polls. A cgroup v2
subgroup provides both, because membership is inherited across fork and exec.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import secrets
import threading

CGROUP_ROOT = Path('/sys/fs/cgroup')
SCOPE_PREFIX = 'slstack-'
GiB = 1024 ** 3


class ScopeUnavailable(RuntimeError):
    """No writable cgroup subgroup here can be given the memory controller."""


def controller_parent(root=CGROUP_ROOT):
    """Nearest ancestor of this process's own cgroup whose children get `memory`.

    A subgroup only sees controllers its parent has delegated, and a directory with
    member processes delegates nothing, so the walk has to leave our own scope.
    """
    own = None
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, separator, location = line.partition('::')
        if separator and hierarchy == '0':
            own = root / location.lstrip('/')
            break
    if own is None:
        raise ScopeUnavailable('no cgroup v2 entry for this process in /proc/self/cgroup')
    candidate = own
    while True:
        delegated = (candidate / 'cgroup.subtree_control').read_text().split() if (candidate / 'cgroup.subtree_control').is_file() else []
        if 'memory' in delegated and os.access(candidate, os.W_OK):
            return candidate
        if candidate == root:
            raise ScopeUnavailable(f'memory controller is not delegated below {root}: '
                                   f'{own} and its ancestors only offer {" ".join(delegated) or "nothing"}')
        candidate = candidate.parent


def scope_name(hint, pid):
    """slstack-<hint>-<creating pid>-<nonce>: the pid says who may still own it."""
    return SCOPE_PREFIX + (hint + '-' if hint else '') + str(pid) + '-' + secrets.token_hex(6)


def reap_stale(parent):
    """Remove empty scopes whose creating process is gone.

    A guardian that is killed before its own teardown leaves a capped directory behind.
    Only an empty one whose creator no longer exists is reaped, so a scope a live launcher
    just created - even before it moved its child in - is never taken from it.
    """
    removed = []
    for candidate in parent.glob(SCOPE_PREFIX + '*'):
        fields = candidate.name.removeprefix(SCOPE_PREFIX).rsplit('-', 2)
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        if Path('/proc', fields[1]).is_dir():
            continue
        try:
            if (candidate / 'cgroup.procs').read_text().strip():
                continue
            candidate.rmdir()
            removed.append(candidate.name)
        except OSError:
            continue
    return removed


def open_scope(memory_bytes, *, hint='job', root=CGROUP_ROOT):
    """A named subgroup capped at `memory_bytes` with paging refused inside it.

    `memory.oom.group` makes the kernel terminate the whole tree rather than one
    member, as exceeding a job's memory limit did. Swap is pinned to zero because the
    guardian's own paging allowance is far smaller than any engine budget: letting the
    subgroup page would replace a refusal with host-wide eviction.
    """
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError('Scope ceiling must be a positive byte count')
    parent = controller_parent(root)
    reap_stale(parent)
    scope = parent / scope_name(hint, os.getpid())
    try:
        scope.mkdir()
    except OSError as error:
        raise ScopeUnavailable(f'cannot create a containment subgroup in {parent}: {error}') from error
    try:
        (scope / 'memory.max').write_text(str(memory_bytes))
        (scope / 'memory.swap.max').write_text('0')
        (scope / 'memory.oom.group').write_text('1')
    except OSError as error:
        scope.rmdir()
        raise ScopeUnavailable(f'cannot set the ceiling on {scope.name}: {error}') from error
    return MemoryScope(scope, memory_bytes)


class MemoryScope:
    """One created subgroup. `pre_exec` moves a launched child in before its first instruction."""

    def __init__(self, path, memory_bytes):
        self.path = Path(path)
        self.memory_bytes = int(memory_bytes)
        self.closed = False

    def pre_exec(self):
        """Join between fork and exec, so the ceiling applies from the first instruction.

        Nothing here may allocate or block: it runs in the child after fork.
        """
        (self.path / 'cgroup.procs').write_text(str(os.getpid()))

    def members(self):
        text = (self.path / 'cgroup.procs').read_text()
        return [int(line) for line in text.split() if line]

    def resident_bytes(self):
        return int((self.path / 'memory.current').read_text())

    def peak_bytes(self):
        peak = self.path / 'memory.peak'
        return int(peak.read_text()) if peak.is_file() else None

    def killed_members(self):
        """OOM terminations the kernel accounted to this subgroup."""
        events = {}
        for line in (self.path / 'memory.events').read_text().splitlines():
            name, _, value = line.partition(' ')
            if name in ('oom', 'oom_kill', 'oom_group_kill', 'max'):
                events[name] = int(value.split()[0]) if value.split() else 0
        return events

    def receipt(self):
        return {'path': str(self.path), 'memory_max_bytes': self.memory_bytes,
                'swap_max_bytes': 0, 'oom_group': True,
                'peak_bytes': self.peak_bytes(), 'oom_events': self.killed_members(),
                'remaining_members': self.members() if not self.closed else []}

    def close(self):
        """Remove the subgroup. It is left in place, still capped, if members remain.

        The guardian retires its tree cooperatively; a scope holding survivors is a
        containment we keep rather than a cleanup we claim succeeded.
        """
        if self.closed:
            return {'path': str(self.path), 'remaining_members': [], 'removed': True}
        remaining = self.members()
        if remaining:
            return {'path': str(self.path), 'remaining_members': remaining, 'removed': False}
        try:
            self.path.rmdir()
        except OSError as error:
            return {'path': str(self.path), 'remaining_members': self.members(),
                    'removed': False, 'error': repr(error)}
        self.closed = True
        return {'path': str(self.path), 'remaining_members': [], 'removed': True}


def scope_hint(spec):
    """A subgroup name a person can read in /sys/fs/cgroup, taken from the receipt stem."""
    return re.sub('[^a-z0-9]+', '-', Path(spec['receipt']).stem.lower()).strip('-')[:40]


def open_containment(spec):
    """The kernel ceiling for a guarded tree, or why there is none. Exactly one is None.

    Without a scope the guardian's RSS poll is the only memory rule, so the absence
    belongs in the receipt rather than being assumed by whoever reads a success.
    """
    threads = threading.active_count()
    if threads > 1:
        # pre_exec runs between fork and exec; other threads may hold allocator locks
        # at that moment, so the move is refused rather than attempted.
        return None, f'no scope: the launcher has {threads} threads, so the child cannot be moved before exec'
    budget = spec.get('job_commit_gib')
    if not budget:
        return None, 'no scope: the spec declares no commit budget'
    try:
        return open_scope(int(float(budget) * GiB), hint=scope_hint(spec)), None
    except (ScopeUnavailable, ValueError) as error:
        return None, 'no scope: ' + str(error)
