import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from slstack.config import load
from slstack.process import OwnedProcess
from slstack.lease import gpu_lease, active_holds, recover_leases
from slstack.shared_guard import CleanupUnconfirmed


class SharedGuardAdapterTests(unittest.TestCase):
    def test_canonical_owned_paging_warning_and_failure_boundaries(self):
        from studio.job_paging import PagingWatch
        observed = {'bytes': 12*1024}
        warnings = []
        watch = PagingWatch(lambda: 0, warning=warnings.append,
                            observer=SimpleNamespace(sample=lambda proc: observed['bytes']))
        limits = dict(rss_limit=1024**2, commit_limit=1024**2, floor=1)
        self.assertIsNone(watch.check(None, 0, 0, **limits))
        self.assertTrue(any('owned pagefile/swap' in row for row in warnings))
        observed['bytes'] = 64*1024**2
        self.assertIsNone(watch.check(None, 0, 0, **limits))
        observed['bytes'] += 4096
        self.assertIn('exceeds 64 MiB', watch.check(None, 0, 0, **limits))

    def test_an_owned_member_that_exits_mid_observation_is_not_a_failure(self):
        # /proc/<pid>/status loses its VmSwap line the moment the member dies. The observer
        # must report zero paged bytes for it, not raise StopIteration into the monitor
        # thread, whose failure the receipt then prints as the job's stop reason.
        from studio.job_paging import JobPaging
        class Exited:
            pid = 4242
            def create_time(self): return 1.0
            def children(self, recursive=True): return []
        class ZombieStatus:
            def read_text(self): return 'Name:\tpython\nState:\tZ (zombie)\n'
        with patch('studio.job_paging.Path', lambda path: ZombieStatus()):
            self.assertEqual(JobPaging().sample(Exited()), 0)

    def test_uncertain_cleanup_keeps_lease_and_is_never_auto_recovered(self):
        cfg = load(overrides={'gpu_uuid':'GPU-test','host_budget_gib':.25,'min_available_gib':2})
        runtime = Path(os.environ.get('STUDIO_RUNTIME_ROOT',cfg['runtime_root'])) / ('uncertain-lease-test-' + uuid.uuid4().hex)
        runtime.mkdir()
        hold = runtime/'holds.txt'
        hold.write_text('',encoding='utf-8')
        cfg.update(runtime_root=str(runtime),hold_file=str(hold))
        with patch('slstack.lease.subprocess.run',return_value=Mock(returncode=0,stdout='0, 32000, 0')):
            with self.assertRaises(CleanupUnconfirmed):
                with gpu_lease(cfg):
                    raise CleanupUnconfirmed('TEST only: synthetic uncertainty')
        self.assertEqual(len(active_holds(hold.read_text(encoding='utf-8-sig'))),1)
        journal = next((runtime/'state/leases').glob('*.json'))
        self.assertTrue(json.loads(journal.read_text())['cleanup_uncertain'])
        with patch('slstack.lease.psutil.Process',side_effect=AssertionError('uncertain lease inspected')):
            self.assertEqual(recover_leases(runtime),[])
        print('Synthetic uncertainty receipt:',journal)

    def test_owned_stdin_command_uses_pinned_paging_guard_and_certified_cleanup(self):
        cfg = load(overrides={'gpu_uuid':'GPU-test','host_budget_gib':.5,
                              'min_available_gib':2,'job_commit_gib':.5,'paging_telemetry':True})
        if os.environ.get('STUDIO_RUNTIME_ROOT'):
            cfg['runtime_root'] = os.environ['STUDIO_RUNTIME_ROOT']
        job = OwnedProcess(cfg,[sys.executable,'-B','-c',
            'import sys,time; print("INPUT="+sys.stdin.read().strip(),flush=True); time.sleep(.4)'],
            'shared-guard-adapter-stdin',stdin='TEST stdin\n',timeout=10)
        try:
            receipt = job.wait()
            self.assertIn('INPUT=TEST stdin',job.log.read_text())
            self.assertTrue(receipt['cleanup_confirmed'])
            self.assertEqual(receipt['paging_policy']['owned_paging_limit_bytes'],64*1024**2)
            self.assertEqual(receipt['shared_guard_commit'],'2774636281ab32bf871497f0a8caae23c6b6cdc7')
            certificate = json.loads(Path(receipt['shared_guard_certificate']).read_text())
            self.assertTrue(certificate['release_allowed'])
            self.assertTrue(certificate['cleanup']['confirmed'])
            self.assertEqual(certificate['command_exit_code'],0)
            # The pinned observer attributes swap bytes per owned process from /proc here;
            # the page-level scan is the Windows working-set query, so a zero page count
            # is not a claim that nothing was observed.
            self.assertEqual(receipt['owned_paging']['attribution'],'bounded process pagefile observation')
            self.assertIsInstance(receipt['owned_paging']['observed_paged_bytes'],int)
            self.assertGreaterEqual(receipt['owned_paging']['observed_paged_bytes'],0)
            # The launcher contained the whole tree in a cgroup scope of its own.
            self.assertEqual(receipt['memory_scope']['memory_max_bytes'],int(.5*2**30))
            self.assertEqual(receipt['memory_scope']['oom_events']['oom_kill'],0)
            self.assertTrue(receipt['scope_cleanup']['removed'])
        finally:
            if job.proc.poll() is None:
                job.stop()


if __name__ == '__main__':
    unittest.main()
