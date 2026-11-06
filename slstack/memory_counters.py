"""Read-only OS paging counters; global activity is not process attribution."""
import mmap
import sys
import time
from pathlib import Path

import psutil

KB = 1024
# /proc/vmstat's pgpgin/pgpgout are counted in 512-byte blocks, not pages, despite the name.
VMSTAT_BLOCK = 512
# Receipt name -> (vmstat field, bytes per counted unit); None means one page. These are
# Linux cumulative counters since boot, never the per-second rates the PDH counters reported.
VMSTAT_COUNTERS = {'pages_input': ('pgpgin', VMSTAT_BLOCK), 'pages_output': ('pgpgout', VMSTAT_BLOCK),
                   'page_reads': ('pswpin', None), 'page_writes': ('pswpout', None)}
# Disk-backed fault events are a count, not a byte total.
VMSTAT_EVENTS = {'major_faults': 'pgmajfault'}


def _procfs_values(text, *, scale_kb=False):
    """`name: value [kB]` and `name value` lines as integers; a malformed line is unreadable."""
    values = {}
    for line in text.splitlines():
        name, separator, remainder = line.partition(':')
        if not separator:
            name, _, remainder = line.partition(' ')
        fields = remainder.split()
        if not fields:
            continue
        amount = int(fields[0])
        values[name.strip()] = amount * KB if scale_kb else amount
    return values


def meminfo_bytes(text=None):
    """/proc/meminfo in bytes; pass text to read a captured sample instead of the live file."""
    return _procfs_values(Path('/proc/meminfo').read_text() if text is None else text, scale_kb=True)


def vmstat_counters(text=None):
    """Cumulative kernel paging counters, each with the unit its raw field is in."""
    raw = _procfs_values(Path('/proc/vmstat').read_text() if text is None else text)
    page_size = mmap.PAGESIZE
    counters = {}
    for name, (field, block) in VMSTAT_COUNTERS.items():
        if field not in raw:
            counters[name] = {'error': 'vmstat field missing: ' + field}
            continue
        counters[name] = {'bytes': raw[field] * (block or page_size), 'raw': raw[field],
                          'unit_bytes': block or page_size}
    for name, field in VMSTAT_EVENTS.items():
        counters[name] = ({'count': raw[field], 'raw': raw[field]} if field in raw
                          else {'error': 'vmstat field missing: ' + field})
    return counters


def swap_bytes(info=None):
    """(used, total) swap slots from /proc/meminfo; a box with no swap is (0, 0)."""
    info = meminfo_bytes() if info is None else info
    return info['SwapTotal'] - info['SwapFree'], info['SwapTotal']


def swap_device_count():
    return max(0, len(Path('/proc/swaps').read_text().splitlines()) - 1)


def available_physical_bytes():
    """MemAvailable: free plus the reclaimable page cache, which is what admission can take."""
    return psutil.virtual_memory().available


def pagefile_used_bytes():
    """Actual swap slots in use, rather than the system commitment.

    https://docs.kernel.org/admin-guide/ram-overview.html and /proc/meminfo's SwapTotal/SwapFree
    This counter does not attribute I/O to a process.
    """
    return swap_bytes()[0]


class PagefilePressure:
    """Shared policy for system noise; private commitment is a separate limit."""
    threshold = 256 * 1024**2

    def __init__(self, sample=pagefile_used_bytes):
        self.sample = sample
        self.initial = sample()
        self.maximum_growth = 0
        self.warning = None

    def check(self):
        growth = max(0, self.sample() - self.initial)
        self.maximum_growth = max(self.maximum_growth, growth)
        if growth > self.threshold:
            return f'SYSTEM PAGEFILE PRESSURE: growth={growth} exceeds {self.threshold}'
        if growth:
            if self.warning is None:
                print(f'WARNING: system pagefile grew {growth} bytes; not attributed to this job',
                      file=sys.stderr, flush=True)
            self.warning = f'System pagefile grew {self.maximum_growth} bytes; not attributed to this job'
        return None

    def receipt(self):
        return {'initial_used_bytes': self.initial,
                'maximum_growth_bytes': self.maximum_growth,
                'kill_above_bytes': self.threshold, 'warning': self.warning,
                'scope': 'system-wide; not per-process attribution'}


def diagnostic_snapshot():
    result = {'time_ns': time.time_ns(), 'available_physical_bytes': psutil.virtual_memory().available,
              'scope': 'system-wide; other processes may contribute; kernel counters are cumulative since boot',
              'page_size': mmap.PAGESIZE}
    used, total = swap_bytes()
    result['paging_file_usage'] = {'first': used, 'second': total}
    result.update(vmstat_counters())
    return result


def snapshot():
    """Global swap totals, read from the same two /proc/meminfo fields every guard poll.

    Swap use is allocated pages, not process commitment or disk-I/O proof.
    The kernel's cumulative counters are explicitly available outside the guard.
    """
    started = time.monotonic()
    used, total = swap_bytes()
    return dict(time_ns=time.time_ns(), available_physical_bytes=psutil.virtual_memory().available,
                scope='system-wide; other processes may contribute', api='/proc/meminfo',
                page_size=mmap.PAGESIZE, pagefile_count=swap_device_count(),
                paging_file_usage=dict(first=used, second=total),
                query_seconds=time.monotonic() - started)
