"""Launch the owned native producer on Linux, as the vendored runner does on Windows.

`vendor/studio/studio/dsv41_windows_runner.py` refuses non-nt in its constructor and is
imported by every combo/ULTRA path, so this module is its counterpart with the same
`_Native`/`_Job` surface. It reuses what chunk 5.1 landed rather than re-implementing it:
`journal.advisory_lock` is the file lease, SIGINT is the cooperative request, and `/proc`
answers what psapi and the Job Object reported.

| retired Windows mechanism | this one |
|---|---|
| `CreateProcessW(CREATE_SUSPENDED)` + `AssignProcessToJobObject` + `ResumeThread` | `slstack/producer_launch.py` waits between fork and exec in its own session; resume is one byte |
| Job containment, from a job created with `CREATE_BREAKAWAY_FROM_JOB` and no memory limit | the cgroup the controller already sits in, inherited across fork and exec, so the guardian's ceiling keeps holding |
| `KILL_ON_JOB_CLOSE`, released by `release_kill_on_close` | the launcher's `PR_SET_PDEATHSIG`, cleared by the exec that resume performs |
| `TerminateJobObject(125)` on a never-resumed root | `SIGKILL` of the unstarted launcher, which ran no engine code |
| `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT)` in the owned console | `SIGINT` to the owned process group |
| `TerminateJobObject(137)` | `SIGKILL` to the owned process group, so the receipt's exit code is `-9` |
| `QueryInformationJobObject(ActiveProcesses)` | live members of the process group |
| `CreateFileW` share modes, `QueryDosDeviceW`, `GetFinalPathNameByHandleW`, `FileIdInfo` | `flock`, the provider's own statfs magic refusal list, `O_NOFOLLOW` plus `realpath` equality, `st_dev`/`st_ino` |
| `K32EnumProcessModulesEx` | `/proc/<pid>/maps` |

The lease is advisory here: a process that never locks the file is not refused the way a
Windows share mode refused it, which is why the provider's identity rechecks stay in front.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
import errno
import logging
import os
from pathlib import Path
import select
import signal
import stat as stat_module
import subprocess
import sys
import time
import uuid

from .cgroup_scope import CGROUP_ROOT
from .journal import advisory_lock
from .memory_counters import available_physical_bytes, meminfo_bytes

from .native_broker import vendor

vendor()
from studio.dsv41_session_contract import require
from studio.dsv41_windows_runner import StuckProcess, stuck_message

log = logging.getLogger(__name__)

LAUNCHER = Path(__file__).with_name('producer_launch.py')
# The launcher confirms its own containment before this constructor returns. Interpreter
# startup is all that happens inside the bound, so it is generous.
ADMISSION_SECONDS = 30.0
CONFIRM_SECONDS = 2.0
FORCE_CONFIRM_SECONDS = 30.0
# struct statfs's first word is read through libc because os.statvfs does not report a type.
STATFS_BUFFER_BYTES = 128
# The same refusal list the weight provider applies (require_local_volume in
# src/llama-dsv41-weights.cpp): Linux has no fixed-versus-removable indicator and no
# substituted-drive namespace, so network and RAM-backed filesystems are refused by their
# statfs magic numbers and an unknown local type is accepted, as the producer's rule does.
REFUSED_VOLUME_MAGICS = frozenset({
    0x6969,        # NFS_SUPER_MAGIC
    0x65735546,    # FUSE_SUPER_MAGIC
    0x5346544D,    # SMB
    0xfe534d42,    # SMB2
    0xff53ff53,    # CIFS
    0x53464146,    # AFS
    0x010209,      # 9P
    0x7461636f,    # OCFS2
    0x01161970,    # GFS2
    0x564c,        # NCP
    0x01021994,    # TMPFS_MAGIC
    0x858458f6,    # RAMFS_MAGIC
    0x516194e9,    # devtmpfs
})


class LaunchUnconfirmed(RuntimeError):
    """The child existed but never reported the containment the launch claims."""


def statfs_magic(libc, location):
    """The filesystem type of `location`, or the OSError that says it cannot be read."""
    buffer = ctypes.create_string_buffer(STATFS_BUFFER_BYTES)
    if libc.statfs(os.fsencode(str(location)), buffer) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot inspect volume of ' + str(location))
    return ctypes.cast(buffer, ctypes.POINTER(ctypes.c_long))[0]


def group_members(pgid):
    """Live process ids sharing a process group, read from /proc.

    A group is what containment means here: membership survives fork and exec, so helpers
    the engine spawns are retired with it, as a Job Object retired its tree. A stopped
    process counts - the unstarted child is owned, not gone - and a zombie is retired.
    """
    members = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open('/proc/' + entry + '/stat', 'rb') as stream:
                fields = stream.read().rpartition(b')')[2].split()
        except OSError:
            continue
        # After "pid (comm)": state, ppid, pgrp.
        if len(fields) > 2 and fields[0] != b'Z' and int(fields[2]) == pgid:
            members.append(int(entry))
    return members


def process_cgroup(pid):
    """The kernel containment this process inherited, as a path (`None` if unreadable)."""
    try:
        text = Path('/proc/' + str(pid) + '/cgroup').read_text()
    except OSError:
        return None
    for line in text.splitlines():
        hierarchy, separator, location = line.partition('::')
        if separator and hierarchy == '0':
            return str(CGROUP_ROOT / location.lstrip('/'))
    return None


def read_admission_confirmation(stream, *, pid):
    """The launcher's own pid once it has taken the group; anything else refuses the job."""
    ready, _, _ = select.select([stream], [], [], ADMISSION_SECONDS)
    if not ready:
        raise LaunchUnconfirmed('Owned launcher did not confirm containment within %.0fs' % ADMISSION_SECONDS)
    report = os.read(stream, 32).decode('ascii', errors='replace')
    if not report.isdigit() or int(report) != pid:
        raise LaunchUnconfirmed('Owned launcher reported %r instead of its pid %d' % (report, pid))
    return int(report)


def launch_environment(env, binary_directory, library_directories):
    """Where the launched producer finds its own program and its shared objects.

    Windows put both on the child's PATH, because that is where its loader looked for the
    DLLs beside the executable. Linux asks the dynamic linker, so the library directories
    go to `LD_LIBRARY_PATH`; the executable's directory stays on PATH for the tools the
    producer runs.
    """
    from .config import path
    libraries = [str(path(location)) for location in library_directories]
    updated = dict(env)
    updated['PATH'] = os.pathsep.join([str(binary_directory), *libraries, env.get('PATH', '')])
    updated['LD_LIBRARY_PATH'] = os.pathsep.join([*libraries, env.get('LD_LIBRARY_PATH', '')])
    return updated


class _Native:

    def __init__(self):
        require(os.name != 'nt', 'POSIX producer runner required')
        self.libc = ctypes.CDLL(None, use_errno=True)

    def memory(self):
        """Available physical RAM, which is what host admission is measured against."""
        return available_physical_bytes()

    def commit(self):
        """(committed, limit) as the private-allocation analogue of a commit charge.

        Linux keeps no commit-charge counter: anonymous plus shared memory plus the swap
        slots actually reserved is the closest total, and nothing in this fork reads it.
        """
        info = meminfo_bytes()
        committed = info['AnonPages'] + info['Shmem'] + info['SwapTotal'] - info['SwapFree']
        return int(committed), int(info['MemTotal'] + info['SwapTotal'])

    def volume(self, location):
        """The device id of a local volume, refusing a network or RAM-backed one."""
        magic = statfs_magic(self.libc, location)
        require(magic not in REFUSED_VOLUME_MAGICS,
                'Producer requires a fixed local volume, not a network or RAM-backed one')
        return os.stat(str(location)).st_dev

    def _open(self, location, *, directory=False, exclusive=False):
        """A descriptor on the exact file or directory, refusing aliases and wrong types."""
        path = Path(location)
        # Python 3.14 dropped Path.lexically_normal, so the normalized-form check is
        # os.path.normpath: it collapses '.' and '..' lexically without touching symlinks,
        # which is the rule the provider's `p == p.lexically_normal()` applies in C++.
        require(path.is_absolute() and str(path) == os.path.normpath(str(path))
                and ('..' not in path.parts) and (directory or path.name),
                'A physical local absolute path is required')
        require(all(':' not in part for part in path.parts[1:]), 'Alternate data streams refused')
        device = self.volume(path if directory else path.parent)
        # O_DIRECTORY would raise before the type rule below could report it, and Windows
        # opened the handle first and refused on the attributes it read back.
        flags = os.O_NOFOLLOW | os.O_CLOEXEC
        flags |= os.O_RDWR | os.O_CREAT | os.O_EXCL if exclusive else os.O_RDONLY
        try:
            descriptor = os.open(str(path), flags)
        except OSError as error:
            # O_NOFOLLOW answers a symlink with ELOOP; the Windows build read the handle
            # and refused it as a reparse point, so the alias is reported as that refusal.
            require(error.errno != errno.ELOOP, 'Reparse or unexpected file type refused')
            raise
        try:
            info = os.fstat(descriptor)
            is_expected_type = (stat_module.S_ISDIR(info.st_mode) if directory
                                else stat_module.S_ISREG(info.st_mode))
            require(is_expected_type, 'Reparse or unexpected file type refused')
            require(info.st_dev == device, 'File is not on the inspected volume')
            require(os.path.realpath(str(path)) == str(path), 'Physical path changed or aliases an input')
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager
    def directory(self, location):
        """Hold the ancestors and the target open for the duration of the block.

        Windows handles refused deletion of the directory; a descriptor pins the inode, so
        a path replaced underneath the lease is a different identity rather than a surprise.
        """
        held = []
        try:
            for parent in [*reversed(Path(location).parents), Path(location)]:
                held.append(self._open(parent, directory=True))
            yield
        finally:
            for descriptor in reversed(held):
                os.close(descriptor)

    @contextmanager
    def read_lease(self, location):
        """A shared lock and an open read stream on a file the producer will use."""
        with self.directory(Path(location).parent):
            descriptor = self._open(location)
            try:
                stream = os.fdopen(descriptor, 'rb')
            except BaseException:
                os.close(descriptor)
                raise
            with stream, advisory_lock(stream, shared=True):
                yield stream

    @contextmanager
    def exclusive(self, location):
        """Create the file only if absent and hold it for the duration of the block.

        A Windows share mode of zero refused every other opener; `flock` refuses only other
        lockers, so the refusal that stays real here is the create-if-absent one.
        """
        with self.directory(Path(location).parent):
            descriptor = self._open(location, exclusive=True)
            try:
                stream = os.fdopen(descriptor, 'rb+')
            except BaseException:
                os.close(descriptor)
                raise
            with stream, advisory_lock(stream):
                yield

    def gpu(self, ordinal, expected_uuid):
        """Free bytes of the configured GPU, after its identity and capability are proven."""
        driver = ctypes.CDLL('libcuda.so.1')

        def cuda(name, args, *values):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = (args, ctypes.c_int)
            require(fn(*values) == 0, 'CUDA identity query failed: ' + name)
        cuda('cuInit', [ctypes.c_uint], 0)
        device = ctypes.c_int()
        cuda('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), ordinal)
        identifier = (ctypes.c_ubyte * 16)()
        cuda('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int], identifier, device)
        require('GPU-' + str(uuid.UUID(bytes=bytes(identifier))) == expected_uuid,
                'CUDA ordinal is not the configured GPU UUID')
        nvml = ctypes.CDLL('libnvidia-ml.so.1')

        def call(name, args, *values):
            fn = getattr(nvml, name)
            fn.argtypes, fn.restype = (args, ctypes.c_int)
            require(fn(*values) == 0, 'NVML admission unavailable: ' + name)
        call('nvmlInit_v2', [])
        try:
            handle = ctypes.c_void_p()
            call('nvmlDeviceGetHandleByUUID', [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
                 expected_uuid.encode('ascii'), ctypes.byref(handle))
            name = ctypes.create_string_buffer(128)
            call('nvmlDeviceGetName', [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint], handle, name, len(name))
            major, minor = ctypes.c_int(), ctypes.c_int()
            call('nvmlDeviceGetCudaComputeCapability', [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                                       ctypes.POINTER(ctypes.c_int)],
                 handle, ctypes.byref(major), ctypes.byref(minor))
            require(major.value * 10 + minor.value >= 120,
                    'Selected CUDA device lacks the compute capability required by this backend')
            memory = (ctypes.c_ulonglong * 3)()
            call('nvmlDeviceGetMemoryInfo', [ctypes.c_void_p, ctypes.c_void_p], handle, memory)
            return int(memory[1])
        finally:
            call('nvmlShutdown', [])

    def suspended(self, argv, env, directory, stdin, stdout, stderr):
        return _Job(self, argv, env, directory, stdin, stdout, stderr)

    def modules(self, process):
        """Every path the owned process has mapped, which is the loaded-module list."""
        pid = getattr(process, 'pid', process)
        try:
            text = Path('/proc/' + str(pid) + '/maps').read_text()
        except OSError as error:
            raise OSError('Mapped-memory observation failed for pid %s: %s' % (pid, error)) from error
        paths = []
        for line in text.splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6 or not fields[5].startswith('/'):
                continue
            location = Path(fields[5])
            if location not in paths:
                paths.append(location)
        return paths


class _Job:
    """One owned producer tree: contained before it runs, retired cooperatively."""

    def __init__(self, native, argv, env, directory, stdin, stdout, stderr):
        require(argv and all(isinstance(item, str) for item in argv),
                'Exact string launch arguments required')
        require(Path(argv[0]).is_absolute() and os.access(argv[0], os.X_OK),
                'Absolute executable producer required')
        require(Path(directory).is_absolute() and Path(directory).is_dir(),
                'Absolute owned working directory required')
        require(all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()),
                'String launch environment required')
        require(all(type(fd) is int and fd >= 0 for fd in (stdin, stdout, stderr)),
                'Owned descriptor numbers are required for the producer transport')

        self.n, self.child, self.admission = native, None, None
        self.args = list(argv)      # what wait_for_exit names when a grace window expires
        self.assigned = self.resumed = False
        self.kill_on_close = True
        self.forced_kill = False
        self.force_reason = None
        self.scope = None
        wait_read, wait_write = os.pipe()
        ready_read, ready_write = os.pipe()
        try:
            self.child = subprocess.Popen(
                [sys.executable, '-I', str(LAUNCHER), str(wait_read), str(ready_write), *argv],
                cwd=str(directory), env=env, stdin=stdin, stdout=stdout, stderr=stderr,
                pass_fds=(wait_read, ready_write))
            self.pgid = self.child.pid
            self.scope = process_cgroup(self.child.pid)
            read_admission_confirmation(ready_read, pid=self.child.pid)
            self.admission = wait_write
            self.assigned = True
        except BaseException:
            # Windows terminated an unassigned suspended child and required the exit
            # before it raised; the unstarted launcher is the same kind of child.
            if self.child is not None:
                self.signal_unstarted(signal.SIGKILL)
                require(self.confirm_exit(CONFIRM_SECONDS), 'Unassigned suspended child exit unconfirmed')
            raise
        finally:
            for descriptor in (wait_read, ready_read, ready_write):
                os.close(descriptor)

    @property
    def pid(self):
        return self.child.pid if self.child is not None else 0

    def members(self):
        return group_members(self.pgid)

    def active(self):
        return len(self.members())

    def signal_unstarted(self, how):
        """Signal only the launcher, which is the whole tree while nothing has exec'd."""
        try:
            os.kill(self.child.pid, how)
        except ProcessLookupError:
            pass

    def confirm_exit(self, seconds):
        try:
            return self.child.wait(timeout=seconds) is not None
        except subprocess.TimeoutExpired:
            return False

    def resume(self):
        # A lost controller is not permission to bypass cooperative retirement: the exec
        # this byte triggers clears the launcher's parent-death signal, which is the release.
        self.release_kill_on_close()
        try:
            os.write(self.admission, b'\1')
        except OSError as error:
            raise RuntimeError('Owned worker resume failed: ' + repr(error)) from error
        self.resumed = True

    def modules(self):
        return self.n.modules(self.child.pid)

    def poll(self):
        return None if self.child is None else self.child.poll()

    def wait(self, timeout=None):
        """The counterpart of waiting on the process handle, for the retirement waits."""
        return self.child.wait(timeout=timeout)

    def interrupt(self):
        """SIGINT to every member of the owned group: Ctrl-Break's counterpart."""
        if not self.members():
            raise OSError('Owned producer group is gone')
        os.killpg(self.pgid, signal.SIGINT)
        return True

    def _wait_empty(self, seconds):
        from slstack.shutdown import RetirementProgress, process_members
        progress = RetirementProgress(max(seconds, .001),
            diagnostic=getattr(self, 'retirement_diagnostic', None), gpu_free=getattr(self, 'gpu_free', None))
        while self.active():
            progress.observe(process_members(self))
            if progress.window.expired():
                return False
            time.sleep(0.05)
        return True

    def release_kill_on_close(self):
        """Stop treating the controller's exit as a kill of the tree.

        Windows cleared `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` on the live job. Here the rule
        is the launcher's parent-death signal and the exec has already cleared it, so this
        only records the release for `close()` of a child that never started.
        """
        self.kill_on_close = False

    def stop(self, graceful_seconds=60, stuck_seconds=None):
        """Two cooperative requests with renewable grace, then a loud force fallback.

        A never-resumed root ran no code and holds no engine resources.
        """
        if not self.active():
            return 'exited'
        if not self.resumed:
            self.signal_unstarted(signal.SIGKILL)
            deadline = time.monotonic() + FORCE_CONFIRM_SECONDS
            while self.active():
                require(time.monotonic() < deadline, 'Unstarted worker exit cannot be confirmed')
                time.sleep(0.05)
            self.confirm_exit(CONFIRM_SECONDS)
            return 'terminate-unstarted'
        stuck_seconds = max(graceful_seconds, stuck_seconds if stuck_seconds is not None else graceful_seconds)
        failures = []
        for attempt, window in ((1, graceful_seconds), (2, stuck_seconds - graceful_seconds)):
            try:
                self.interrupt()
            except Exception as exc:
                failures.append('SIGINT %d: %s' % (attempt, repr(exc)[:200]))
            if self._wait_empty(window):
                self.confirm_exit(CONFIRM_SECONDS)
                code = self.poll()
                how = 'graceful' if attempt == 1 else 'graceful-late'
                return how if code in (0, 130) else '%s-crashed exit=%s' % (how, code)
        reason = 'no shutdown progress for %ds after cooperative requests%s' % (stuck_seconds, ('; ' + '; '.join(failures)) if failures else '')
        self.forced_kill = True; self.force_reason = reason
        log.critical('FORCE KILL LAST FALLBACK: owned PID=%s; %s', self.pid, reason)
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self._wait_empty(30):
            self.confirm_exit(CONFIRM_SECONDS)
            return 'forced-kill: ' + reason
        self.release_kill_on_close()
        message = stuck_message(self.pid, reason)
        log.critical(message)
        print(message, file=sys.stderr, flush=True)
        raise StuckProcess(message)

    def close(self):
        """Hand back the controller's side of the launch.

        Closing the job handle retired a never-started tree while kill-on-close was set;
        the admission descriptor going away says the same thing, and the launcher leaves
        with 125 without running engine code. A resumed tree survives, as it did once the
        job's flag was cleared.
        """
        if self.admission is not None:
            os.close(self.admission)
            self.admission = None
        if self.child is not None and not self.resumed and self.kill_on_close:
            self.signal_unstarted(signal.SIGKILL)
            if not self.confirm_exit(CONFIRM_SECONDS):
                message = stuck_message(self.pid, 'unstarted child unconfirmed at close')
                log.critical(message)
                print(message, file=sys.stderr, flush=True)


def default_runner(runner=None):
    """The caller's runner, or the one this platform needs."""
    if runner is not None:
        return runner
    if os.name == 'nt':
        from studio.dsv41_windows_runner import _Native as WindowsRunner
        return WindowsRunner()
    return _Native()
