"""Resident admission of the owned producer's host pages, through the mechanism this OS has.

Windows set a hard minimum working set on the owned process; Linux sets `memory.min` on the
cgroup the job inherited, which bounds the guarded tree rather than one process and which the
kernel only honours while every ancestor between the scope and the root carries a floor. Both
arms are gated against a real child, and the Linux arm against a real scope.
"""
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import unittest
import uuid

from slstack.cgroup_scope import ScopeUnavailable, ancestor_floors, open_scope
from slstack.native_broker import vendor
from slstack.producer_runner import default_runner
from slstack.working_set import protect_working_set

vendor()
from studio.dsv41_session_process import Producer

MiB = 1 << 20
HOST = 128 * MiB
FLOOR = 4 << 30
AGGREGATE = 512 * MiB
CPU_CHILD = 'import sys;sys.stdin.buffer.read(1);print("CPU_RESIDENT_PASS")'
SUPPORTED = True
try:
    open_scope(64 * MiB, hint='probe').close()
except ScopeUnavailable as reason:
    SUPPORTED = reason


def runtime_root():
    return Path(os.environ['STUDIO_RUNTIME_ROOT'])


def launch_cpu_child(scope=None):
    """One owned CPU child, optionally moved into a scope of ours before it runs.

    The guarded path produces this shape itself: the guardian creates the scope and the
    guarded command's children inherit membership across fork and exec. Writing the pid
    here is that same move, done before resume so the floor is admitted against a scope
    the job really belongs to.
    """
    native = default_runner()
    source, send = os.pipe()
    output, sink = os.pipe()
    errors, diagnostic = os.pipe()
    job = native.suspended([sys.executable, '-B', '-c', CPU_CHILD], dict(os.environ),
                           str(runtime_root()), source, sink, diagnostic)
    for descriptor in (source, sink, diagnostic):
        os.close(descriptor)
    if scope is not None:
        (scope.path / 'cgroup.procs').write_text(str(job.pid))
        job.scope = str(scope.path)
    job.resume()
    return job, send, output, errors


def retire(job, descriptors):
    job.stop()
    job.close()
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


@unittest.skipUnless(os.name == 'nt', 'Windows working set contract')
class WorkingSetTests(unittest.TestCase):
    # Reused CPU admission/pipe fixture from the protection provider.
    def test_owned_child_admission_and_legacy_allocator_preserve_hard_minimum(self):
        from studio.dsv41_windows_runner import _Native
        native = _Native(); fds = []; job = None
        try:
            source, send = os.pipe(); output, sink = os.pipe(); errors, diagnostic = os.pipe()
            fds = [source, send, output, sink, errors, diagnostic]
            job = native.suspended([sys.executable, '-B', '-c', CPU_CHILD],
                dict(os.environ), os.environ['STUDIO_RUNTIME_ROOT'], source, sink, diagnostic)
            for fd in (source, sink, diagnostic):os.close(fd);fds.remove(fd)
            job.resume()
            receipt = runtime_root() / ('resident-cpu-' + uuid.uuid4().hex + '.json')
            admission = protect_working_set(job, HOST, FLOOR, AGGREGATE, receipt=receipt)
            self.assertEqual(json.loads(receipt.read_text()), admission)
            self.assertEqual(admission['pid'], job.pid)
            self.assertEqual(admission['minimum'], HOST)
            self.assertEqual(admission['flags'] & 5, 1)
            # The native locked allocator uses this older API with the larger
            # current minimum. Verify the OS preserves the hard-minimum flag.
            from slstack.working_set import kernel_api
            k = kernel_api()
            k.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            k.SetProcessWorkingSetSize.restype = ctypes.c_int
            self.assertTrue(k.SetProcessWorkingSetSize(job.pi.process, admission['minimum'], admission['maximum']))
            low, high, flags = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_ulong()
            self.assertTrue(k.GetProcessWorkingSetSizeEx(job.pi.process, ctypes.byref(low), ctypes.byref(high), ctypes.byref(flags)))
            self.assertEqual(low.value, HOST)
            self.assertEqual(flags.value & 5, 1)
            os.write(send, b'G'); deadline = time.monotonic() + 10
            while job.poll() is None and time.monotonic() < deadline:time.sleep(.01)
            self.assertEqual(job.poll(), 0)
            self.assertIn(b'CPU_RESIDENT_PASS', os.read(output, 4096))
        finally:
            if job is not None:job.stop();job.close()
            for fd in fds:os.close(fd)


@unittest.skipUnless(SUPPORTED is True, 'no delegated cgroup memory controller: ' + str(SUPPORTED))
class ScopeResidentAdmissionTests(unittest.TestCase):
    def test_the_floor_lands_on_the_scope_the_job_inherited(self):
        scope = open_scope(256 * MiB, hint='resident')
        job = None
        try:
            job, send, output, errors = launch_cpu_child(scope)
            receipt = runtime_root() / ('resident-cpu-' + uuid.uuid4().hex + '.json')
            admission = protect_working_set(job, HOST, FLOOR, AGGREGATE, receipt=receipt)
            self.assertEqual(json.loads(receipt.read_text()), admission)
            self.assertEqual(admission['pid'], job.pid)
            self.assertEqual(admission['mechanism'], 'cgroup memory.min')
            self.assertEqual(admission['resident']['floor_bytes'], HOST)
            self.assertEqual(admission['resident']['ceiling_bytes'], 256 * MiB)
            self.assertEqual(admission['resident']['original_floor_bytes'], 0)
            self.assertEqual(int((scope.path / 'memory.min').read_text()), HOST)
            # Whether the kernel honours it is the ancestors' business, not ours: report the
            # same answer the chain gives, rather than a guarantee the kernel will not keep.
            honored = all(value is not None and value >= HOST
                          for _, value in ancestor_floors(scope.path))
            self.assertEqual(admission['resident']['honored'], honored)
            self.assertEqual(admission['resident']['blocked_by'],
                             None if honored else str(next(node for node, value in
                                                           ancestor_floors(scope.path)
                                                           if value is None or value < HOST)))
            os.write(send, b'G')
            deadline = time.monotonic() + 10
            while job.poll() is None and time.monotonic() < deadline:time.sleep(.01)
            self.assertEqual(job.poll(), 0)
            self.assertIn(b'CPU_RESIDENT_PASS', os.read(output, 4096))
        finally:
            if job is not None:retire(job, [send, output, errors])
            scope.close()

    def test_a_scope_the_stack_did_not_create_is_reported_and_left_alone(self):
        job = None
        try:
            job, send, output, errors = launch_cpu_child()
            inherited = job.scope
            before = (Path(inherited) / 'memory.min').read_text()
            admission = protect_working_set(job, HOST, FLOOR, AGGREGATE)
            self.assertIs(admission['resident']['applied'], False)
            self.assertEqual(admission['resident']['scope'], inherited)
            self.assertIn('did not create', admission['resident']['reason'])
            self.assertEqual((Path(inherited) / 'memory.min').read_text(), before)
            os.write(send, b'G')
            self.assertEqual(job.wait(timeout=10), 0)
        finally:
            if job is not None:retire(job, [send, output, errors])

    def test_a_job_with_no_scope_admits_without_a_floor(self):
        job = None
        try:
            job, send, output, errors = launch_cpu_child()
            job.scope = None
            admission = protect_working_set(job, HOST, FLOOR, AGGREGATE)
            self.assertIs(admission['resident']['applied'], False)
            self.assertIn('no kernel scope', admission['resident']['reason'])
            os.write(send, b'G')
            self.assertEqual(job.wait(timeout=10), 0)
        finally:
            if job is not None:retire(job, [send, output, errors])

    def test_refuses_unbudgeted_controller_before_any_protection(self):
        class NoScope:
            pid = os.getpid()

            @property
            def scope(self):raise AssertionError('A refused aggregate must not touch a scope')
        with self.assertRaisesRegex(MemoryError, 'aggregate resident'):
            protect_working_set(NoScope(), HOST, FLOOR, HOST + (1 << 20))

    def test_the_floor_keeps_reclaim_room_below_the_scope_ceiling(self):
        """A floor equal to the ceiling leaves nothing to evict, so pressure would kill the tree."""
        scope = open_scope(160 * MiB, hint='tight-ceiling')
        job = None
        try:
            job, send, output, errors = launch_cpu_child(scope)
            admission = protect_working_set(job, HOST, FLOOR, AGGREGATE)
            self.assertEqual(admission['resident']['requested_bytes'], HOST)
            self.assertEqual(admission['resident']['floor_bytes'], 96 * MiB)
            self.assertEqual(admission['resident']['headroom_kept_bytes'], 64 * MiB)
            self.assertEqual(int((scope.path / 'memory.min').read_text()), 96 * MiB)
        finally:
            if job is not None:retire(job, [send, output, errors])
            scope.close()

    def test_an_admission_that_rounds_below_one_page_refuses(self):
        scope = open_scope(64 * MiB, hint='sub-page')
        job = None
        try:
            job, send, output, errors = launch_cpu_child(scope)
            with self.assertRaisesRegex(MemoryError, 'rounds below one page'):
                protect_working_set(job, HOST, 512 << 20, 64 << 30)   # ceiling minus headroom is 0
        finally:
            if job is not None:retire(job, [send, output, errors])
            scope.close()

    def test_an_admission_below_one_page_refuses_rather_than_protecting_nothing(self):
        scope = open_scope(256 * MiB, hint='one-byte-admission')
        job = None
        try:
            job, send, output, errors = launch_cpu_child(scope)
            with self.assertRaisesRegex(MemoryError, 'rounds below one page'):
                protect_working_set(job, 1, 512 << 20, 64 << 30)
        finally:
            if job is not None:retire(job, [send, output, errors])
            scope.close()

    def test_post_resume_failure_certifies_native_child_cleanup(self):
        def refusal(job):raise MemoryError('TEST resident admission refused')
        with self.assertRaisesRegex(MemoryError, 'resident admission refused') as caught:
            Producer(uuid.uuid4().hex, [sys.executable, '-B', '-c', 'import time;time.sleep(30)'],
                dict(os.environ), runtime_root(), host_bytes=256 << 20,
                physical_floor=4 << 30, operation_seconds=10, idle_seconds=10,
                native=default_runner(), after_resume=refusal)
        proof = caught.exception.owned_cleanup
        self.assertTrue(all(proof.get(k) for k in ('owned_job_empty', 'owned_process_signaled',
                                                  'owned_pid_gone')), proof)


if __name__ == '__main__':
    unittest.main()
