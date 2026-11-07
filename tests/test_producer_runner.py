"""CPU fixtures for the Linux producer runner: containment, admission, retirement, leases.

No model, and no GPU beyond read-only NVML queries. The fixture directory is repo-local on
purpose: the runner refuses a RAM-backed volume, which is what /tmp is on this box.
"""
import ctypes
from pathlib import Path
import os
import sys
import time
import unittest
from unittest.mock import patch
import uuid

from slstack.journal import LockUnavailable, advisory_lock
from slstack.memory_counters import meminfo_bytes
from slstack.native_broker import vendor
from slstack.producer_runner import (REFUSED_VOLUME_MAGICS, LaunchUnconfirmed, _Job, _Native,
                                     default_runner, group_members, read_admission_confirmation,
                                     statfs_magic)

vendor()
from studio.dsv41_session_contract import Refused

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'import time\nprint("RAN",flush=True)\nwhile True:time.sleep(.05)\n'
HANDLER = ('import signal,sys,time\n'
           'def leave(kind,frame):print("SIGINT_SEEN",flush=True);sys.exit(130)\n'
           'signal.signal(signal.SIGINT,leave)\n'
           'print("RAN",flush=True)\n'
           'while True:time.sleep(.05)\n')
IGNORES = ('import signal,subprocess,sys,time\n'
           'signal.signal(signal.SIGINT,signal.SIG_IGN)\n'
           'subprocess.Popen([sys.executable,"-c","import time\\nwhile True:time.sleep(.05)"])\n'
           'print("RAN",flush=True)\n'
           'while True:time.sleep(.05)\n')


def fixture(name):
    location = ROOT / 'runtime' / 'runner-tests' / name
    location.mkdir(parents=True, exist_ok=True)
    return location


def launchers_running():
    """Children this run launched but never admitted, found by their launcher script."""
    found = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            command = Path('/proc/' + entry + '/cmdline').read_bytes()
        except OSError:
            continue
        if b'producer_launch.py' in command:
            found.append(int(entry))
    return found


class OwnedChildTests(unittest.TestCase):

    def setUp(self):
        self.native = _Native()
        self.directory = fixture('owned')
        self.job = None
        self.held = []
        self.input_write = None
        self.output_read = None

    def tearDown(self):
        if self.job is not None:
            try:
                self.job.stop(2, 3)
            except Exception:
                pass
            self.job.close()
        for descriptor in self.held:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def launch(self, source, *, resume=True):
        child_in, controller_in = os.pipe()
        controller_out, child_out = os.pipe()
        child_err, controller_err = os.pipe()
        self.held.extend([child_in, controller_in, controller_out, child_out, child_err, controller_err])
        self.job = self.native.suspended([sys.executable, '-u', '-c', source], dict(os.environ),
                                         self.directory, child_in, child_out, child_err)
        for descriptor in (child_in, child_out, child_err):
            os.close(descriptor)
        self.input_write, self.output_read = controller_in, controller_out
        if resume:
            self.job.resume()
        return self.job

    def available(self, seconds=.3):
        import select
        ready, _, _ = select.select([self.output_read], [], [], seconds)
        return os.read(self.output_read, 4096) if ready else b''

    def test_nothing_runs_before_admission(self):
        job = self.launch(SCRIPT, resume=False)
        self.assertIsNone(job.poll())
        self.assertEqual(job.active(), 1)
        self.assertFalse(job.resumed)
        self.assertEqual(self.available(), b'')
        job.resume()
        self.assertEqual(self.available(2), b'RAN\n')

    def test_the_child_is_contained_in_the_cgroup_it_inherited(self):
        job = self.launch(SCRIPT)
        own = '/sys/fs/cgroup' + Path('/proc/self/cgroup').read_text().split('::')[1].strip()
        self.assertEqual(job.scope, own)
        self.assertEqual(Path(job.scope).is_dir(), True)
        self.assertEqual(os.getpgid(job.pid), job.pid)

    def test_never_resumed_child_is_retired_without_running_engine_code(self):
        job = self.launch(SCRIPT, resume=False)
        self.assertEqual(job.stop(2, 3), 'terminate-unstarted')
        self.assertEqual(self.available(), b'')
        self.assertEqual(job.active(), 0)
        self.assertEqual(job.poll(), -9)
        self.job = None

    def test_closing_an_unstarted_child_retires_it(self):
        job = self.launch(SCRIPT, resume=False)
        job.close()
        self.assertEqual(job.poll(), -9)
        self.assertEqual(group_members(job.pgid), [])
        self.job = None

    def test_a_resumed_tree_survives_its_close(self):
        job = self.launch(SCRIPT)
        job.close()
        self.assertEqual(job.active(), 1)
        self.assertIsNone(job.poll())

    def test_the_cooperative_request_retires_the_group(self):
        job = self.launch(HANDLER)
        self.assertEqual(self.available(2), b'RAN\n')
        started = time.monotonic()
        self.assertEqual(job.stop(10, 20), 'graceful')
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.available(2), b'SIGINT_SEEN\n')
        self.assertEqual(job.poll(), 130)
        self.assertFalse(job.forced_kill)
        self.job = None

    def test_force_fallback_follows_both_requests_and_both_graces(self):
        job = self.launch(IGNORES)
        self.assertEqual(self.available(2), b'RAN\n')
        self.assertGreaterEqual(job.active(), 2)
        events = []
        with patch.object(_Job, 'interrupt', autospec=True, side_effect=lambda owner: events.append('request') or True):
            with self.assertLogs('slstack.producer_runner', level='CRITICAL'):
                result = job.stop(1, 3)
        self.assertTrue(result.startswith('forced-kill:'))
        self.assertEqual(events, ['request', 'request'])
        self.assertTrue(job.forced_kill)
        self.assertEqual(job.poll(), -9)
        self.assertEqual(group_members(job.pgid), [])
        self.job = None

    def test_resume_after_the_child_is_gone_refuses(self):
        job = self.launch(SCRIPT, resume=False)
        os.kill(job.pid, 9)
        job.confirm_exit(2)
        with self.assertRaises(RuntimeError):
            job.resume()
        self.job = None

    def test_the_retirement_wait_uses_the_job_handle(self):
        """wait_for_exit drives poll()/wait() on the owned job, as combo_producer's close does."""
        import threading
        from slstack.shutdown import wait_for_exit
        job = self.launch(HANDLER)
        self.assertEqual(self.available(2), b'RAN\n')
        threading.Timer(.3, lambda: job.interrupt()).start()
        self.assertEqual(wait_for_exit(job, 10, diagnostic=lambda: 0), 130)
        self.job = None

    def test_modules_are_the_mapped_paths(self):
        job = self.launch(SCRIPT)
        paths = job.modules()
        self.assertIn(Path(sys.executable).resolve(), paths)
        self.assertTrue(any(path.name.startswith('lib') for path in paths))

    def test_launch_validation_refuses_before_any_child_exists(self):
        self.assertEqual(launchers_running(), [])
        missing = self.directory / 'not-created'
        cases = ((['relative-binary'], dict(os.environ), str(self.directory)),
                 ([sys.executable], {'key': 1}, str(self.directory)),
                 ([sys.executable], dict(os.environ), str(missing)),
                 ([sys.executable], dict(os.environ), 'relative'),
                 ([], dict(os.environ), str(self.directory)))
        for argv, env, directory in cases:
            with self.subTest(argv=argv, directory=directory):
                with self.assertRaises(Refused):
                    self.native.suspended(argv, env, Path(directory), 0, 1, 2)
        self.assertEqual(launchers_running(), [])

    def test_unconfirmed_admission_leaves_no_launcher_behind(self):
        with patch('slstack.producer_runner.read_admission_confirmation',
                   side_effect=LaunchUnconfirmed('no confirmation')):
            with self.assertRaises(LaunchUnconfirmed):
                self.native.suspended([sys.executable, '-c', 'pass'], dict(os.environ), self.directory, 0, 1, 2)
        self.assertEqual(launchers_running(), [])

    def test_runner_selection(self):
        self.assertIsInstance(default_runner(), _Native)
        marker = object()
        self.assertIs(default_runner(marker), marker)

    def test_admission_confirmation_reads_the_reported_pid(self):
        read, write = os.pipe()
        try:
            os.write(write, str(os.getpid()).encode())
            self.assertEqual(read_admission_confirmation(read, pid=os.getpid()), os.getpid())
            os.write(write, b'99999999')
            with self.assertRaises(LaunchUnconfirmed):
                read_admission_confirmation(read, pid=os.getpid())
            os.close(write)
            with self.assertRaises(LaunchUnconfirmed):
                read_admission_confirmation(read, pid=os.getpid())
        finally:
            os.close(read)


class VolumeAndLeaseTests(unittest.TestCase):

    def setUp(self):
        self.native = _Native()
        self.directory = fixture('lease')
        self.file = self.directory / 'payload.bin'
        self.file.write_bytes(b'content')

    def test_ram_backed_volume_is_refused(self):
        self.assertIn(statfs_magic(self.native.libc, '/dev/shm'), REFUSED_VOLUME_MAGICS)
        with self.assertRaises(Refused), self.native.read_lease('/dev/shm/producer-runner-probe.bin'):
            pass

    def test_relative_traversing_and_aliased_paths_are_refused(self):
        with self.assertRaises(Refused), self.native.read_lease('runtime/runner-tests/lease/payload.bin'):
            pass
        traversing = str(self.directory / '..' / 'lease' / 'payload.bin')
        with self.assertRaises(Refused), self.native.read_lease(traversing):
            pass
        link = self.directory / 'alias.bin'
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(self.file.name)
        try:
            with self.assertRaises(Refused), self.native.read_lease(link):
                pass
        finally:
            link.unlink()

    def test_directory_lease_refuses_a_file_and_read_lease_refuses_a_directory(self):
        with self.assertRaises(Refused), self.native.directory(self.file):
            pass
        with self.assertRaises(Refused), self.native.read_lease(self.directory):
            pass

    def test_read_leases_coexist_and_refuse_an_exclusive_holder(self):
        with self.native.read_lease(self.file), self.native.read_lease(self.file):
            with self.assertRaises(LockUnavailable), self.file.open('rb') as stream:
                with advisory_lock(stream):
                    pass

    def test_exclusive_creation_refuses_an_existing_file(self):
        held = self.directory / 'held.lock'
        if held.exists():
            held.unlink()
        with self.native.exclusive(held):
            with self.assertRaises(FileExistsError):
                with self.native.exclusive(held):
                    pass

    def test_memory_and_commit_are_read_from_proc(self):
        import psutil
        self.assertEqual(self.native.memory(), psutil.virtual_memory().available)
        committed, limit = self.native.commit()
        info = meminfo_bytes()
        self.assertEqual(limit, info['MemTotal'] + info['SwapTotal'])
        self.assertLess(committed, limit)


class DeviceIdentityTests(unittest.TestCase):

    def setUp(self):
        self.native = _Native()

    def configured_identity(self):
        identifier = (ctypes.c_ubyte * 16)()
        device = ctypes.c_int()
        driver = ctypes.CDLL('libcuda.so.1')
        if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0) or \
                driver.cuDeviceGetUuid_v2(identifier, device):
            self.skipTest('no CUDA driver here')
        return 'GPU-' + str(uuid.UUID(bytes=bytes(identifier)))

    def test_the_configured_device_answers_without_a_context(self):
        try:
            ctypes.CDLL('libnvidia-ml.so.1')
        except OSError as error:
            self.skipTest('no NVML on this box: ' + repr(error))
        identity = self.configured_identity()
        free = self.native.gpu(0, identity)
        reported = subprocess_free(identity)
        self.assertGreater(free, 0)
        if reported is not None:
            self.assertLess(abs(free - reported), 1 << 30)
        with self.assertRaisesRegex(Refused, 'configured GPU UUID'):
            self.native.gpu(0, 'GPU-' + str(uuid.UUID(int=2)))

    def test_capability_is_required_and_free_bytes_are_reported(self):
        class Function:
            def __init__(self, name):
                self.name = name

            def __call__(self, *args):
                if self.name == 'cuDeviceGetUuid_v2':
                    for index, value in enumerate(uuid.UUID(int=1).bytes):
                        args[0][index] = value
                if self.name == 'nvmlDeviceGetCudaComputeCapability':
                    ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = capability[0]
                    ctypes.cast(args[2], ctypes.POINTER(ctypes.c_int))[0] = capability[1]
                if self.name == 'nvmlDeviceGetMemoryInfo':
                    args[1][1] = 123
                return 0

        class Library:
            def __getattr__(self, name):
                return Function(name)

        capability = [12, 0]
        expected = 'GPU-' + str(uuid.UUID(int=1))
        with patch.object(ctypes, 'CDLL', return_value=Library()):
            native = _Native.__new__(_Native)
            self.assertEqual(native.gpu(0, expected), 123)
            with self.assertRaisesRegex(Refused, 'configured GPU UUID'):
                native.gpu(0, 'GPU-' + str(uuid.UUID(int=2)))
            capability[:] = [8, 6]
            with self.assertRaisesRegex(Refused, 'compute capability'):
                native.gpu(0, expected)


def subprocess_free(identity):
    """nvidia-smi's free MiB for the same device, as the cross-check of NVML's answer."""
    import subprocess
    try:
        reply = subprocess.run(['nvidia-smi', '--id=' + identity, '--query-gpu=memory.free',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if reply.returncode:
        return None
    try:
        return int(float(reply.stdout.strip()) * (1 << 20))
    except ValueError:
        return None


if __name__ == '__main__':
    unittest.main()
