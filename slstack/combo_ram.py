"""Parent-owned shared RAM held for the whole handoff, shared by name across processes.

On Windows this is an anonymous kernel section (`CreateFileMappingW` over
`INVALID_HANDLE_VALUE`), never a filesystem checkpoint. Linux's counterpart is a named
section under `/dev/shm`, which is RAM by construction: the producer's volume rule refuses
RAM-backed *producer* paths - weights, receipts, outputs - and does not reach this tier,
because staying in RAM is the whole point of it. Nothing here goes through the runner's
lease machinery, and the section is created 0600 so another user cannot open the name.

All transport reads/writes stay <=64KiB. On Windows a failure to lock is a visible refusal.
Linux cannot lock a tier larger than `RLIMIT_MEMLOCK` (8 MiB soft and hard for an
unprivileged process here - chunk 3.1h), so the lock is chunked and best effort, the
shortfall is a visible line, and what actually holds the pages resident is the guarded
scope's `memory.swap.max=0` plus the guardian's paging guard.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import math
import mmap
import os
import resource
import threading
import time
import uuid
import psutil
from pathlib import Path
from .guard import admit
from .memory_counters import available_physical_bytes

GiB=1024**3
PAGE=65536            # the transport's owner-page bound, not the kernel's page size
LOCK_CHUNK=64<<20     # first chunk to try, halved at every refusal, as the engine's RAM tier does
SHM_DIRECTORY=Path('/dev/shm')
KERNEL_PAGE=os.sysconf('SC_PAGE_SIZE') if os.name!='nt' else PAGE
SECTION_PREFIX='DS4COMBO-'


def runtime_min_free_gib(cfg):
    value=cfg.get('runtime_min_free_gib',0.5)
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<2**34:
        raise ValueError('runtime_min_free_gib must be finite and nonnegative (<2**34); 0 disables it')
    return float(value)


def wait_for_host_capacity(host_bytes,floor_bytes,cancelled,*,timeout_s=30,available=None,clock=time.monotonic,sleep=time.sleep):
    """Allow OS RAM reclamation after certified engine retirement. Owner rule (no fixed
    timeouts): the wait lives while available RAM keeps rising (reclamation is progressing);
    timeout_s is the NO-PROGRESS window, renewed by every new maximum, not a total cap."""
    if any(type(n) is not int or n<=0 for n in (host_bytes,floor_bytes)):
        raise ValueError('Explicit integer host/floor budgets required')
    if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or not 0<timeout_s<=600:
        raise ValueError('Host admission no-progress window must be 0-600 seconds')
    available=available or available_physical_bytes
    started=progress=clock();minimum=maximum=None;polls=0;announced=False
    while True:
        free=available();polls+=1;now=clock()
        if type(free) is not int or free<0:raise ValueError('Host admission physical telemetry is invalid')
        minimum=free if minimum is None else min(minimum,free)
        if maximum is None or free>maximum:maximum=free;progress=now
        row=dict(host_budget_bytes=host_bytes,physical_floor_bytes=floor_bytes,required_bytes=host_bytes+floor_bytes,
            available_bytes=free,minimum_available_bytes=minimum,maximum_available_bytes=maximum,seconds=now-started,
            seconds_without_progress=now-progress,polls=polls,timeout_seconds=timeout_s)
        if cancelled():
            error=RuntimeError('Host admission cancelled before producer launch');error.admission_receipt=row;raise error
        if free>=host_bytes+floor_bytes:return row
        if now-progress>=timeout_s:
            error=MemoryError(f"Producer host admission refused: available RAM stopped rising for {now-progress:.1f}s "
                f"(available={free}, required={host_bytes+floor_bytes}, waited {now-started:.1f}s)")
            error.admission_receipt=row;raise error
        if not announced:
            print('DS4COMBO_HOST_ADMISSION_WAIT '+json.dumps(row),flush=True);announced=True
        sleep(min(.1,timeout_s-(now-progress)))


def kernel_api():
    if os.name!='nt':raise RuntimeError('The Windows shared checkpoint section requires Windows')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileMappingW.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD,w.DWORD,w.DWORD,w.LPCWSTR]
    k.CreateFileMappingW.restype=w.HANDLE
    k.OpenFileMappingW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];k.OpenFileMappingW.restype=w.HANDLE
    k.MapViewOfFile.argtypes=[w.HANDLE,w.DWORD,w.DWORD,w.DWORD,ctypes.c_size_t];k.MapViewOfFile.restype=ctypes.c_void_p
    k.UnmapViewOfFile.argtypes=[ctypes.c_void_p];k.UnmapViewOfFile.restype=w.BOOL
    k.VirtualLock.argtypes=[ctypes.c_void_p,ctypes.c_size_t];k.VirtualLock.restype=w.BOOL
    k.VirtualUnlock.argtypes=[ctypes.c_void_p,ctypes.c_size_t];k.VirtualUnlock.restype=w.BOOL
    k.CloseHandle.argtypes=[w.HANDLE];k.CloseHandle.restype=w.BOOL
    k.GetCurrentProcess.restype=w.HANDLE
    k.GetProcessWorkingSetSize.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessWorkingSetSize.restype=w.BOOL
    k.SetProcessWorkingSetSize.argtypes=[w.HANDLE,ctypes.c_size_t,ctypes.c_size_t];k.SetProcessWorkingSetSize.restype=w.BOOL
    k.SetProcessWorkingSetSizeEx.argtypes=[w.HANDLE,ctypes.c_size_t,ctypes.c_size_t,w.DWORD]
    k.SetProcessWorkingSetSizeEx.restype=w.BOOL
    return k


def new_section_name(session):
    """DS4COMBO-<session nonce>-<creating pid>-<nonce>: the pid says who may still own it.

    A tmpfs name outlives the process that made it, which a kernel section's handle did
    not, so the creating pid is part of the name and `reap_stale_sections` uses it. The
    shape is otherwise opaque: nothing but `attach` reads a name, on either platform.
    """
    return SECTION_PREFIX+session+'-'+str(os.getpid())+'-'+uuid.uuid4().hex


def reap_stale_sections():
    """Remove this user's checkpoint sections whose creating process is gone.

    A controller killed before its own teardown leaves the tier allocated in RAM until
    reboot, so an owned name with a dead creator is reclaimed by whoever creates next -
    the same rule `cgroup_scope.reap_stale` applies to capped directories.
    """
    removed = []
    if os.name=='nt':return removed
    try:entries=os.listdir(str(SHM_DIRECTORY))
    except OSError:return removed
    for entry in entries:
        if not entry.startswith(SECTION_PREFIX):continue
        fields=entry.removeprefix(SECTION_PREFIX).rsplit('-',2)
        if len(fields)!=3 or not fields[1].isdigit():continue
        if Path('/proc',fields[1]).is_dir():continue
        location=SHM_DIRECTORY/entry
        try:
            if location.stat().st_uid!=os.getuid():continue   # another user's tier is not ours to remove
            location.unlink();removed.append(entry)
        except OSError:
            continue
    return removed


def section_name(name):
    """The name as a `/dev/shm` entry, refusing anything that could escape that directory."""
    text=str(name)
    if not text.startswith(SECTION_PREFIX) or any(sep in text for sep in ('/','\\','..')):
        raise ValueError('Checkpoint section name must be a bare DS4COMBO section name')
    return SHM_DIRECTORY/text


def memlock_allowance_bytes():
    """What this process may lock at all; RLIM_INFINITY reports as no ceiling."""
    soft=resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return (1<<64)-1 if soft==resource.RLIM_INFINITY else int(soft)


class SharedSection:
    """One mapped extent with bounded transfers, readable by name from another process.

    The owner-page bound is the transport's rule, not the kernel's, so it holds on both
    platforms; only how a byte range is reached differs.
    """

    def __init__(self,name,capacity,address=None,mapping=None,handle=None):
        self.name=name;self.capacity=capacity
        self.address=address;self.mapping=mapping;self.handle=handle
        self.lock=threading.RLock();self.leases=0

    def _range(self,offset,count):
        if not self.address:raise RuntimeError('Checkpoint RAM is closed or not physically locked')
        if type(offset) is not int or type(count) is not int or not 0<=count<=PAGE or not 0<=offset<=self.capacity or count>self.capacity-offset:
            raise ValueError('Checkpoint transfer must be bounded to64KiB and its admitted extent')

    def read(self,offset,count):
        with self.lock:
            self._range(offset,count)
            return (ctypes.string_at(self.address+offset,count) if os.name=='nt'
                    else self.mapping[offset:offset+count])

    def write(self,offset,data):
        with self.lock:
            if type(data) is not bytes:raise TypeError('Checkpoint page must be immutable bytes')
            self._range(offset,len(data))
            if os.name=='nt':ctypes.memmove(self.address+offset,data,len(data))
            else:self.mapping[offset:offset+len(data)]=data

    def sha256(self,offset,count):
        if type(count) is not int or count<0 or offset<0 or count>self.capacity-offset:raise ValueError('Hash extent exceeds checkpoint RAM')
        digest=hashlib.sha256()
        with self.lease():
            for start in range(0,count,PAGE):digest.update(self.read(offset+start,min(PAGE,count-start)))
        return digest.hexdigest()

    @contextmanager
    def lease(self):
        with self.lock:
            self._range(0,0);self.leases+=1
        try:yield self
        finally:
            with self.lock:self.leases-=1

    def close(self):
        with self.lock:
            if self.leases:raise RuntimeError('Cannot release checkpoint RAM while transport owns it')
            self._release()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def _release(self):
        """Platform teardown of the mapping and whatever the OS handed with it."""
        if os.name=='nt':
            k=kernel_api()
            if self.address and not k.UnmapViewOfFile(self.address):
                raise ctypes.WinError(ctypes.get_last_error())
            if self.handle and not k.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            if self.mapping is not None:self.mapping.close()
            if self.handle is not None:os.close(self.handle)
        self.address=None;self.mapping=None;self.handle=None


def attach(name,capacity):
    """Open a checkpoint section another process created, without re-admitting anything.

    The child-side counterpart of `OpenFileMappingW` + `MapViewOfFile` (read/write, as
    that pair's protection was): the same name, the same bounded transfer surface, no
    lock request and no host admission - the owner holds both. The name is what travels
    between the processes, exactly as it did on Windows.
    """
    if type(capacity) is not int or capacity<=0:raise ValueError('Attached extent must be an explicit byte count')
    if os.name=='nt':
        k=kernel_api();handle=k.OpenFileMappingW(6,False,name)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        address=k.MapViewOfFile(handle,6,0,0,capacity)
        if not address:
            k.CloseHandle(handle);raise ctypes.WinError(ctypes.get_last_error())
        return SharedSection(name,capacity,address,None,handle)
    location=section_name(name)
    descriptor=os.open(str(location),os.O_RDWR)
    try:
        view=mmap.mmap(descriptor,capacity,access=mmap.ACCESS_WRITE)
        return SharedSection(name,capacity,_address(view),view,None)
    except BaseException:
        os.close(descriptor);raise


def _address(view):
    """The mapping's base address, with the buffer export released straight away."""
    probe=ctypes.c_char.from_buffer(view)
    base=ctypes.addressof(probe)
    del probe                       # a live export would make the later close() refuse
    return base


class LockedRam(SharedSection):
    def __init__(self,capacity,*,session,host_budget_gib,min_available_gib):
        if type(capacity) is not int or not 0<capacity<=2*GiB:raise ValueError('Checkpoint RAM capacity must be explicit and <=2GiB')
        if len(session)!=32 or any(c not in '0123456789abcdef' for c in session):raise ValueError('Exact session nonce required')
        self.host_budget_gib=host_budget_gib;self.floor_gib=min_available_gib
        if psutil.Process().memory_info().rss+capacity>host_budget_gib*GiB:
            raise MemoryError('Controller plus locked checkpoint exceeds the explicit host budget')
        admit(capacity/GiB,min_available_gib)
        self.name=('Local\\' if os.name=='nt' else '')+new_section_name(session)
        reap_stale_sections()
        SharedSection.__init__(self,self.name,capacity)
        self.locked=False;self.locked_bytes=0;self.lock_reason=None
        try:
            self._create()
            self._lock()
            self.locked=True
            if available_physical_bytes()<min_available_gib*GiB:
                raise MemoryError('Available physical RAM floor failed while locking checkpoint')
        except BaseException:
            self.close();raise

    def _create(self):
        if os.name=='nt':
            k=kernel_api()
            ctypes.set_last_error(0)
            handle=k.CreateFileMappingW(w.HANDLE(-1),None,4,self.capacity>>32,self.capacity&0xffffffff,self.name)
            if not handle or ctypes.get_last_error()==183:raise ctypes.WinError(ctypes.get_last_error())
            self.handle=handle
            address=k.MapViewOfFile(handle,6,0,0,self.capacity)
            if not address:
                k.CloseHandle(handle);self.handle=None;raise ctypes.WinError(ctypes.get_last_error())
            self.address=address
            # The locked tier has to stay inside the controller's own allowance too, and the
            # legacy allocator reads a current minimum back, so both bounds keep their gap.
            process=k.GetCurrentProcess();lo,hi=ctypes.c_size_t(),ctypes.c_size_t()
            if not k.GetProcessWorkingSetSize(process,ctypes.byref(lo),ctypes.byref(hi)):
                raise ctypes.WinError(ctypes.get_last_error())
            needed=psutil.Process().memory_info().rss+self.capacity+(32<<20)
            if not k.SetProcessWorkingSetSizeEx(process,max(lo.value,needed),max(hi.value,needed+(64<<20)),1|8):
                raise ctypes.WinError(ctypes.get_last_error())
            if not k.VirtualLock(self.address,self.capacity):raise ctypes.WinError(ctypes.get_last_error())
            self.locked_bytes=self.capacity
            return
        location=section_name(self.name)
        try:
            descriptor=os.open(str(location),os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        except FileExistsError as error:
            raise OSError(error.errno,'A checkpoint section with this name already exists',str(location)) from error
        self.handle=descriptor
        try:
            os.ftruncate(descriptor,self.capacity)
            self.mapping=mmap.mmap(descriptor,self.capacity,flags=mmap.MAP_SHARED)
            self.address=_address(self.mapping)
        except BaseException:
            os.close(descriptor);self.handle=None
            try:os.unlink(str(location))
            except FileNotFoundError:pass
            raise

    def _lock(self):
        """`mlock` in whole-page chunks, halving at each refusal, as the engine's tier does.

        None of this is a refusal: the allowance is smaller than any real tier on an
        unprivileged Linux host, and the mapping stays resident because the guarded scope
        refuses swap. The shortfall is reported, never assumed away.
        """
        if os.name=='nt':return
        libc=ctypes.CDLL(None,use_errno=True)
        locked=0;chunk=min(LOCK_CHUNK,self.capacity);reason=0
        while locked<self.capacity:
            count=min(chunk,self.capacity-locked)
            if libc.mlock(ctypes.c_void_p(self.address+locked),count)==0:
                locked+=count;continue
            if not reason:reason=ctypes.get_errno()
            if count==KERNEL_PAGE:break
            chunk=max(KERNEL_PAGE,(count//KERNEL_PAGE//2)*KERNEL_PAGE)
        self.locked_bytes=locked;self.lock_reason=reason
        if locked<self.capacity:
            print('DS4COMBO_LOCKED_RAM_PARTIAL '+json.dumps(dict(name=self.name,capacity_bytes=self.capacity,
                locked_bytes=locked,shortfall_bytes=self.capacity-locked,errno=reason,
                memlock_allowance_bytes=memlock_allowance_bytes(),
                enforcement='the guarded scope memory.swap.max=0 and the guardian paging guard'),
                separators=(',',':')),flush=True)

    def _release(self):
        if os.name=='nt':
            if self.locked_bytes and not kernel_api().VirtualUnlock(self.address,self.capacity):
                raise ctypes.WinError(ctypes.get_last_error())
            self.locked_bytes=0;self.locked=False
            SharedSection._release(self)
            return
        location=section_name(self.name)
        locked=self.locked_bytes
        self.locked_bytes=0
        if locked and self.address:
            libc=ctypes.CDLL(None,use_errno=True)
            done=0
            while done<locked:                    # unlocking pages the kernel never locked is a no-op
                count=min(LOCK_CHUNK,locked-done)
                if libc.munlock(ctypes.c_void_p(self.address+done),count)!=0:break
                done+=count
        SharedSection._release(self)
        try:os.unlink(str(location))              # the name outlives the owner only as long as the tier does
        except FileNotFoundError:pass
        self.locked=False
