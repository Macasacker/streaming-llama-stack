"""cgroup v2 containment: the ceiling applies from the first instruction and to descendants."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from slstack.cgroup_scope import (ScopeUnavailable, controller_parent, open_containment,
                                  open_scope, reap_stale, scope_hint, scope_name)

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


if __name__ == '__main__':
    unittest.main()
