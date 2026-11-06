"""Admission serialisation: one advisory lock per hold ledger, contended or waited."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest

from slstack.journal import LockUnavailable, advisory_lock
from slstack.lease import AdmissionRefused, gpu_lease

ROOT = Path(__file__).resolve().parents[1]
PROBE = ('import sys\n'
         'from slstack.journal import advisory_lock, LockUnavailable\n'
         'stream = open(sys.argv[1], "r+")\n'
         'try:\n'
         '    with advisory_lock(stream):\n'
         '        print("ACQUIRED")\n'
         'except LockUnavailable:\n'
         '    print("CONTENDED")')
HOLDER = ('import sys, time\n'
          'from slstack.journal import advisory_lock\n'
          'stream = open(sys.argv[1], "r+")\n'
          'with advisory_lock(stream):\n'
          '    time.sleep(float(sys.argv[2]))')


def run(code, *args):
    return subprocess.run([sys.executable, '-B', '-c', code, *args], capture_output=True,
                          text=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))


class AdvisoryLockTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='advisory-lock-'))
        self.ledger = self.folder / 'gpu.holds'
        self.ledger.write_text('')

    def test_a_second_holder_sees_the_lock_as_contended(self):
        with self.ledger.open('r+') as stream, advisory_lock(stream):
            self.assertEqual(run(PROBE, str(self.ledger)).stdout.strip(), 'CONTENDED')

    def test_the_lock_is_released_when_the_block_ends(self):
        with self.ledger.open('r+') as stream, advisory_lock(stream):
            pass
        self.assertEqual(run(PROBE, str(self.ledger)).stdout.strip(), 'ACQUIRED')

    def test_a_blocking_wait_ends_when_the_holder_leaves(self):
        holder = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(self.ledger), '2'],
                                  cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        try:
            deadline = time.monotonic() + 10
            while run(PROBE, str(self.ledger)).stdout.strip() != 'CONTENDED':
                if time.monotonic() >= deadline:
                    self.fail('the holder never took the lock')
                time.sleep(.05)
            started = time.monotonic()
            with self.ledger.open('r+') as waiting, advisory_lock(waiting, blocking=True):
                waited = time.monotonic() - started
            self.assertGreaterEqual(waited, .2)
            self.assertLess(waited, 10)
        finally:
            holder.wait(timeout=10)

    def test_a_contended_admission_is_refused_not_crashed(self):
        cfg = dict(hold_file=str(self.ledger), runtime_root=str(self.folder), task='DS4STACK',
                   host_budget_gib=1, min_available_gib=1, lease_minutes=1, gpu_uuid='GPU-test')
        with self.ledger.open('r+') as stream, advisory_lock(stream):
            with self.assertRaises(AdmissionRefused) as caught:
                with gpu_lease(cfg):
                    self.fail('admission ran while another process held the ledger')
        self.assertIn('already in progress', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
