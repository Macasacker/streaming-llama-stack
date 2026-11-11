"""The ports this stack binds, and the wait for a stale holder to let go.

A producer force-killed while it carried 21 GiB of pinned host staging and an mlock'd tier takes
seconds to unwind in the kernel. Its socket stays occupied without ever reaching LISTEN, so a
`ss -ltn` pre-flight finds nothing and the next start answers 503 `[Errno 98] Address already in
use` on its first request — measured twice on this box, and worked around for the probe in
`scripts/gpu_handoff.py:103-140`. The engine needs the same check, and it needs it before it
binds rather than inside a request. See docs/SESSION_FAILURE_PLAN.md.
"""
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

PORT_FILES = ('/proc/net/tcp', '/proc/net/tcp6')
PORT_POLL_SECONDS = .25


class PortHeld(RuntimeError):
    """A socket on a port this stack must bind, after waiting for it to clear."""


def stack_ports(cfg):
    """Every port this stack binds: the serving port and the port named in the decode URL."""
    ports = []
    if isinstance(cfg.get('port'), int):ports.append(cfg['port'])
    url = cfg.get('ds4_url')
    if isinstance(url, str) and urlsplit(url).port:ports.append(urlsplit(url).port)
    return sorted(set(ports))


def socket_inodes(port, path):
    """Inodes of the sockets on `port` in one /proc/net table, in any state."""
    inodes = set()
    try:text = Path(path).read_text()
    except OSError:return inodes
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10:continue
        local = fields[1]
        if ':' not in local:continue
        try:
            if int(local.split(':')[1], 16) == port:inodes.add(fields[9])
        except ValueError:continue
    return inodes


def port_holder(port, exclude_pid=None):
    """None when the port is free, else the pid holding it or 'unattributed'.

    'unattributed' is a real result and not a pass: a socket exists that this process cannot
    map to a readable `/proc/<pid>/fd`, which is exactly the case where killing by name would be
    a guess.
    """
    inodes = set().union(*(socket_inodes(port, path) for path in PORT_FILES))
    if not inodes:return None
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == exclude_pid:continue
        try:descriptors = list((entry / 'fd').iterdir())
        except OSError:continue
        for descriptor in descriptors:
            try:target = os.readlink(descriptor)
            except OSError:continue
            if target.startswith('socket:[') and target[8:-1] in inodes:return int(entry.name)
    return 'unattributed'


def held_ports(cfg, exclude_pid=None):
    """(port, holder) for every port the stack binds that somebody else still holds."""
    out = []
    for port in stack_ports(cfg):
        holder = port_holder(port, exclude_pid=exclude_pid)
        if holder is not None:out.append((port, holder))
    return out


def wait_for_ports(cfg, seconds, *, stopping=None, exclude_pid=None, sleep=time.sleep):
    """Return once every port the stack binds is free; refuse with who had held them.

    Nothing is killed here. The holder may be another agent's live engine, and this stack's
    contract is that admission never ejects a process.
    """
    deadline = time.monotonic() + seconds
    while True:
        held = held_ports(cfg, exclude_pid=exclude_pid)
        if not held:return
        if time.monotonic() >= deadline or (stopping is not None and stopping.is_set()):
            who = ', '.join(f'port {port} held by ' + (str(holder) if holder != 'unattributed' else
                            'a process this user cannot read') for port, holder in held)
            raise PortHeld(f'{who}; waited {seconds:g} s and no process was killed — hand the card '
                           'and its ports over with scripts/gpu_handoff.py')
        sleep(PORT_POLL_SECONDS)
