"""Admit the owned native producer's host pages as resident before accepting readiness.

Windows did this with a hard minimum working set on the owned process
(`SetProcessWorkingSetSizeEx`, `HARDWS_MIN_ENABLE`), which the retired fork reused
unchanged from a qualified private fork at 543a76c5126f6bcc2400307dd13c82b4407857dd
(slstack/combo_ram.py). Linux has no per-process page floor: the counterpart is
`memory.min` on the cgroup the job inherited, which bounds a tree rather than a process,
and which the kernel only honours while every ancestor between that cgroup and the root
carries a floor of its own. Where it does not, residency is enforced by the scope's
`memory.swap.max=0` - the tree's pages are resident or the tree is terminated - and by
the guardian's paging guard, exactly as chunk 3.1h concluded for the carry tier. The
receipt therefore says which of those is holding the admission, in both arms.
"""
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import time
import psutil
from .cgroup_scope import is_own_scope, memory_floor, page_floor, read_limit
from .journal import atomic_json
from .memory_counters import available_physical_bytes

OWNED_INIT_RSS = 8 << 20


def kernel_api():
    """The Win32 working-set bindings, refused on any other platform.

    The Linux arm needs no handle at all: the job's scope is a path, and the kernel
    answers the readback through the same file the write went to.
    """
    if os.name != 'nt': raise RuntimeError('Native producer working-set protection requires Windows')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetProcessWorkingSetSizeEx.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(w.DWORD)]
    k.GetProcessWorkingSetSizeEx.restype=w.BOOL
    k.SetProcessWorkingSetSizeEx.argtypes=[w.HANDLE,ctypes.c_size_t,ctypes.c_size_t,w.DWORD]
    k.SetProcessWorkingSetSizeEx.restype=w.BOOL
    k.GetProcessId.argtypes=[w.HANDLE];k.GetProcessId.restype=w.DWORD
    return k


def wait_owned_initialization(job,rss,*,threshold=OWNED_INIT_RSS,stall_s=30.0,clock=time.monotonic,sleep=time.sleep):
    """Wait for a new owned process's working set to reach `threshold` before protecting it.
    Owner rule (no fixed timeouts): it lives while the process is alive and its RSS grows; it
    fails only when the process exits or its RSS makes no progress for stall_s, with the reason.
    (A fixed 2 s deadline failed right after a large producer freed ~11 GB: page zeroing is slow.)"""
    started=progress=clock();best=-1;polls=0;announced=False
    while True:
        current=rss();polls+=1;now=clock()
        if current>=threshold:return dict(seconds=now-started,polls=polls,rss_bytes=current)
        code=job.poll()
        if code is not None:
            raise RuntimeError(f'Owned process exited (code {code}) before resident admission: rss={current} of {threshold} after {now-started:.2f}s')
        if current>best:best=current;progress=now
        elif now-progress>=stall_s:
            raise TimeoutError(f'Owned working-set initialization made no progress for {now-progress:.1f}s: '
                f'rss={current} of {threshold}, process alive, {now-started:.1f}s since launch')
        if not announced and now-started>=2:
            print('DS4COMBO_OWNED_WS_INIT_WAIT '+json.dumps(dict(pid=job.pid,rss_bytes=current,threshold_bytes=threshold,seconds=now-started)),flush=True)
            announced=True
        sleep(.001 if now-started<2 else .05)


def protect_working_set(job,host_bytes,floor_bytes,aggregate_bytes,*,receipt=None):
    """Protect admitted host pages before bootstrap/readiness of the owned job.

    A hard maximum could force paging; it stays disabled. RSS/commit/floor and
    actual paging remain enforced by the shared guardian.
    https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex
    """
    if any(type(n) is not int or n<=0 for n in (host_bytes,floor_bytes,aggregate_bytes)):
        raise ValueError('Explicit integer working-set/aggregate/floor budgets required')
    controller_rss=psutil.Process().memory_info().rss
    metadata_headroom=64<<20
    if host_bytes+controller_rss+metadata_headroom>aggregate_bytes:
        raise MemoryError('Engine plus controller/state exceeds aggregate resident admission')
    available=available_physical_bytes()
    if host_bytes+floor_bytes>available:
        raise MemoryError('Resident engine working-set admission cannot fit physical RAM')
    # Wait for the new process working set; its scope/RSS/floor budgets already
    # apply before resume. This admission precedes bootstrap and readiness.
    started=time.monotonic();process=psutil.Process(job.pid)
    wait_owned_initialization(job,lambda:process.memory_info().rss)   # progress-based (no fixed deadline)
    row=dict(pid=job.pid,host_budget_bytes=host_bytes,aggregate_budget_bytes=aggregate_bytes,
        controller_rss_bytes=controller_rss,metadata_headroom_bytes=metadata_headroom,
        physical_floor_bytes=floor_bytes,available_before=available)
    row.update(_windows_minimum(job,host_bytes,metadata_headroom) if os.name=='nt'
             else _scope_minimum(job,host_bytes,metadata_headroom))
    row['initialization_wait_seconds']=time.monotonic()-started
    if receipt is not None:atomic_json(receipt,row)
    return row


def _windows_minimum(job,host_bytes,metadata_headroom):
    """The hard minimum working set on the owned process, as the retired arm set it."""
    k=kernel_api();handle=job.pi.process
    if k.GetProcessId(handle)!=job.pid:raise ValueError('Suspended process handle/owner PID differs')
    lo,hi,flags=ctypes.c_size_t(),ctypes.c_size_t(),w.DWORD()
    if not k.GetProcessWorkingSetSizeEx(handle,ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    original=dict(minimum=lo.value,maximum=hi.value,flags=flags.value)
    # HARDWS_MIN_ENABLE=1; HARDWS_MAX_DISABLE=8. The existing locked allocator
    # preserves this flag and retains a larger existing working-set allowance.
    # Keep space between the minimum and the advisory maximum; equal limits
    # are rejected on the target Windows host. The RSS budget stays host_bytes.
    if not k.SetProcessWorkingSetSizeEx(handle,host_bytes,host_bytes+metadata_headroom,1|8):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.GetProcessWorkingSetSizeEx(handle,ctypes.byref(lo),ctypes.byref(hi),ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    if lo.value!=host_bytes or hi.value<host_bytes or not flags.value&1 or flags.value&4:
        raise MemoryError('Windows did not retain the admitted hard-minimum working set')
    return dict(mechanism='process working set',original=original,
        minimum=lo.value,maximum=hi.value,flags=flags.value)


def _scope_minimum(job,host_bytes,metadata_headroom):
    """`memory.min` on the scope the job inherited, or the reason it was not written.

    The floor bounds a tree, so it covers the guarded controller as well as the producer;
    that is the same tree the scope's ceiling already bounds, and host admission refuses an
    engine that does not fit it. The floor is kept below that ceiling by the same metadata
    headroom the Windows arm kept between its minimum and its advisory maximum: a subgroup
    whose floor equals its ceiling has nothing left to reclaim, so pressure there turns
    into a termination instead of an eviction of page cache.
    """
    scope=getattr(job,'scope',None)
    if scope is None:
        return dict(mechanism='cgroup memory.min',resident=dict(
            applied=False,reason='the job reported no kernel scope, so no floor could be set'))
    if not is_own_scope(scope):
        return dict(mechanism='cgroup memory.min',resident=dict(
            applied=False,scope=str(scope),
            reason='the job inherited a scope this stack did not create, so its floor is not ours to raise'))
    limit=read_limit(Path(scope)/'memory.max')
    requested=page_floor(host_bytes)
    floor=page_floor(requested if limit is None else min(requested,max(0,limit-metadata_headroom)))
    if floor<=0:
        raise MemoryError(f'Resident admission of {host_bytes} bytes rounds below one page '
                          f'inside a scope ceiling of {limit if limit is not None else "unknown"} bytes')
    resident=memory_floor(scope,floor)
    resident['requested_bytes']=requested
    if limit is not None:resident['headroom_kept_bytes']=limit-floor
    if not resident['honored']:
        # Loud, as the carry tier's partial lock is: the number is written, and it is not
        # what keeps the pages resident on this host.
        print('DS4COMBO_RESIDENT_FLOOR_INERT '+json.dumps(dict(pid=job.pid,**resident)),flush=True)
    return dict(mechanism='cgroup memory.min',resident=resident)
