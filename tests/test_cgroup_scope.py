"""cgroup v2 containment: the ceiling applies from the first instruction and to descendants,
and the floor half reports what the kernel will actually honour."""
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

from slstack.cgroup_scope import (ScopeUnavailable, ancestor_floors, controller_parent,
                                  is_own_scope, memory_floor, open_containment, open_scope,
                                  page_floor, read_limit, reap_stale, scope_hint, scope_name,
                                  PAGE_BYTES)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MiB = 1024 ** 2
ALLOCATE = ('import sys\n'
            'want = int(sys.argv[1]) * 1024 * 1024\n'
            'data = bytearray()\n'
            'while len(data) < want:\n'
            '    data.extend(b"x" * (2 * 1024 * 1024))\n'
            'print("ALLOCATED", len(data), flush=True)\n')
NESTED = ('import subprocess,sys\n'
          'child = subprocess.run([sys.executable, "-B", "-c", %r, sys.argv[1]], '
          'capture_output=True, text=True)\n'
          'print("NESTED_EXIT", child.returncode, flush=True)\n' % ALLOCATE)
HOLDER = 'import time; time.sleep(30)'
SUPPORTED = True
try:
    controller_parent()
except ScopeUnavailable as reason:
    SUPPORTED = reason


def launch(code, *args, scope=None):
    return subprocess.run([sys.executable, '-B', '-c', code, *args], cwd=ROOT,
                          capture_output=True, text=True,
                          preexec_fn=scope.pre_exec if scope else None)


@unittest.skipUnless(SUPPORTED is True, 'no delegated cgroup memory controller: ' + str(SUPPORTED))
class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.folder = pathlib.Path(tempfile.mkdtemp(prefix='cgroup-scope-'))

    def spec(self, job_commit_gib=1.0):
        return dict(job_commit_gib=job_commit_gib, receipt=str(self.folder / 'guard.receipt.json'))

    def test_the_ceiling_refuses_an_oversized_child(self):
        scope = open_scope(64 * MiB, hint='oversized')
        try:
            result = launch(ALLOCATE, '200', scope=scope)
            self.assertLess(result.returncode, 0, 'the kernel must refuse, not merely report')
            self.assertNotIn('ALLOCATED', result.stdout)
            events = scope.receipt()['oom_events']
            self.assertGreaterEqual(events.get('oom_kill', 0), 1)
        finally:
            self.assertTrue(scope.close()['removed'])

    def test_a_child_inside_the_ceiling_runs_and_the_scope_is_removed_afterwards(self):
        scope = open_scope(256 * MiB, hint='fitting')
        result = launch(ALLOCATE, '8', scope=scope)
        self.assertEqual(result.returncode, 0)
        self.assertIn('ALLOCATED', result.stdout)
        proof = scope.close()
        self.assertTrue(proof['removed'], proof)
        self.assertEqual(proof['remaining_members'], [])
        self.assertFalse(scope.path.is_dir())

    def test_membership_is_inherited_across_exec(self):
        scope = open_scope(64 * MiB, hint='nested')
        try:
            result = launch(NESTED, '200', scope=scope)
            # The grandchild was born inside the scope, so the kernel refused it without
            # the guardian ever having to catch it between ownership polls, and
            # memory.oom.group terminated the whole tree as a job's memory limit did.
            self.assertLess(result.returncode, 0)
            self.assertNotIn('ALLOCATED', result.stdout)
            self.assertEqual(scope.members(), [])
            self.assertGreaterEqual(scope.receipt()['oom_events'].get('oom_group_kill', 0), 1)
        finally:
            self.assertTrue(scope.close()['removed'])

    def test_a_scope_with_survivors_is_left_in_place_still_capped(self):
        scope = open_scope(256 * MiB, hint='survivor')
        holder = subprocess.Popen([sys.executable, '-B', '-c', HOLDER], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, preexec_fn=scope.pre_exec)
        proof = scope.close()
        self.assertFalse(proof['removed'])
        self.assertEqual(proof['remaining_members'], [holder.pid])
        self.assertEqual(int((scope.path / 'memory.max').read_text()), 256 * MiB)
        holder.kill()
        holder.wait(timeout=10)
        self.assertTrue(scope.close()['removed'])

    def test_containment_reports_the_budget_and_why_there_is_no_scope(self):
        scope, absent = open_containment(self.spec())
        self.assertIsNone(absent)
        self.assertEqual(scope.memory_bytes, 1024 ** 3)
        self.assertEqual(scope_hint(self.spec()), 'guard-receipt')
        scope.close()
        empty, absent = open_containment(self.spec(job_commit_gib=None))
        self.assertIsNone(empty)
        self.assertIn('no commit budget', absent)

    def test_a_scope_of_a_dead_launcher_is_reaped_when_the_next_one_is_created(self):
        parent = controller_parent()
        dead = next(pid for pid in range(os.getpid() + 1000, os.getpid() + 2000)
                    if not pathlib.Path('/proc', str(pid)).is_dir())
        stale = parent / scope_name('stale-fixture', dead)
        stale.mkdir()
        (stale / 'memory.max').write_text(str(64 * MiB))
        self.assertIn(stale.name, reap_stale(parent))
        self.assertFalse(stale.is_dir())
        scope = open_scope(64 * MiB, hint='after-reap')
        self.assertTrue(scope.path.is_dir())
        scope.close()

    def test_a_scope_of_a_live_launcher_is_left_alone(self):
        parent = controller_parent()
        live = parent / scope_name('live-fixture', os.getpid())
        live.mkdir()
        try:
            self.assertNotIn(live.name, reap_stale(parent))
            self.assertTrue(live.is_dir())
        finally:
            live.rmdir()

    def test_a_zero_ceiling_is_refused_before_any_directory_is_created(self):
        with self.assertRaises(ValueError):
            open_scope(0, hint='zero')
        with self.assertRaises(ValueError):
            open_scope(None, hint='none')


@unittest.skipUnless(SUPPORTED is True, 'no delegated cgroup memory controller: ' + str(SUPPORTED))
class FloorTests(unittest.TestCase):
    """`memory.min`, the floor half. It is written in whole pages and its honour depends on
    the ancestors, so the gate asserts the kernel's answer rather than the write's success."""
    MiB = 1024 ** 2
    TOUCH = ('import os,sys,mmap,time\n'
             'path, scope, mb = sys.argv[1], sys.argv[2], int(sys.argv[3])\n'
             'with open(scope + "/cgroup.procs", "w") as stream:\n'
             '    stream.write(str(os.getpid()))\n'
             'fd = os.open(path, os.O_RDWR | os.O_CREAT)\n'
             'os.ftruncate(fd, mb * 1024 * 1024)\n'
             'view = mmap.mmap(fd, mb * 1024 * 1024)\n'
             'for offset in range(0, mb * 1024 * 1024, 4096):\n'
             '    view[offset] = 7\n'
             'view.flush()\n'
             'print("TOUCHED", flush=True)\n'
             'time.sleep(60)\n')

    def test_only_a_scope_this_stack_created_is_ours_to_floor(self):
        self.assertTrue(is_own_scope(controller_parent() / scope_name('owned', os.getpid())))
        self.assertFalse(is_own_scope(controller_parent()))
        self.assertFalse(is_own_scope('/sys/fs/cgroup/user.slice'))

    def test_the_floor_is_a_whole_number_of_pages_and_is_read_back(self):
        scope = open_scope(512 * self.MiB, hint='floor')
        try:
            requested = 130 * self.MiB + 17
            self.assertEqual(page_floor(requested), requested - requested % PAGE_BYTES)
            row = memory_floor(scope.path, page_floor(requested))
            self.assertEqual(row['floor_bytes'], page_floor(requested))
            self.assertEqual(int((scope.path / 'memory.min').read_text()), row['floor_bytes'])
            self.assertEqual(row['original_floor_bytes'], 0)
            self.assertEqual(row['ceiling_bytes'], 512 * self.MiB)
            self.assertTrue(row['retained'])
            self.assertEqual((scope.path / 'memory.min').read_text().strip(), str(row['floor_bytes']))
            self.assertEqual(row['honored'], all(value is not None and value >= row['floor_bytes']
                                                 for _, value in ancestor_floors(scope.path)))
            if not row['honored']:
                self.assertIn(row['blocked_by'], [str(node) for node, _ in ancestor_floors(scope.path)])
                self.assertIn('zeroes this protection', row['reason'])
        finally:
            self.assertTrue(scope.close()['removed'])

    def test_a_limit_written_as_max_reads_as_no_limit(self):
        folder = pathlib.Path(tempfile.mkdtemp(prefix='cgroup-limits-'))
        try:
            unlimited = folder / 'memory.max'
            unlimited.write_text('max\n')
            self.assertIsNone(read_limit(unlimited))
            number = folder / 'memory.min'
            number.write_text('4096\n')
            self.assertEqual(read_limit(number), 4096)
            self.assertIsNone(read_limit(folder / 'absent'))
        finally:
            for created in folder.iterdir():
                created.unlink()
            folder.rmdir()

    def test_a_floor_is_refused_where_there_is_no_memory_controller_or_no_floor_to_set(self):
        empty = pathlib.Path(tempfile.mkdtemp(prefix='no-memory-controller-'))
        try:
            with self.assertRaises(ScopeUnavailable):
                memory_floor(empty, 4096)
        finally:
            empty.rmdir()
        scope = open_scope(64 * self.MiB, hint='empty-floor')
        try:
            with self.assertRaises(ValueError):
                memory_floor(scope.path, 0)
        finally:
            scope.close()

    @unittest.skipUnless((controller_parent() / 'memory.reclaim').is_file(),
                         'no memory.reclaim interface to apply pressure with')
    @unittest.skipUnless(os.environ.get('STUDIO_RUNTIME_ROOT'),
                         'needs a disk-backed fixture directory (STUDIO_RUNTIME_ROOT)')
    def test_the_reported_honour_is_the_kernel_s_actual_answer(self):
        """A subgroup's pages stay only while every ancestor carries the floor too.

        The claim is checked against the kernel, not against the write: the child holds
        64 MiB of page cache inside the scope and the parent then asks for exactly that
        back. An inert floor loses the pages; an honoured one does not.
        """
        scope = open_scope(512 * self.MiB, hint='honour')
        fixture = pathlib.Path(os.environ['STUDIO_RUNTIME_ROOT']) / ('floor-fixture-' + os.urandom(8).hex() + '.bin')
        holder = None
        try:
            row = memory_floor(scope.path, 64 * self.MiB)
            holder = subprocess.Popen([sys.executable, '-B', '-c', self.TOUCH, str(fixture), str(scope.path), '64'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert holder.stdout.readline().startswith('TOUCHED')
            before = scope.resident_bytes()
            try:
                (scope.path / 'memory.reclaim').write_text(str(64 * self.MiB))
            except OSError:
                pass       # nothing to take back is an answer of its own, read below
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and scope.resident_bytes() == before:
                time.sleep(.05)
            reclaimed = before - scope.resident_bytes()
            if row['honored']:
                self.assertLess(reclaimed, 8 * self.MiB, f'an honoured floor lost {reclaimed} bytes')
            else:
                self.assertGreater(reclaimed, 32 * self.MiB,
                                   f"{row['reason']}, yet the kernel kept {reclaimed} bytes off")
        finally:
            if holder is not None:
                holder.kill(); holder.wait(timeout=10)
            fixture.unlink(missing_ok=True)
            scope.close()


if __name__ == '__main__':
    unittest.main()
