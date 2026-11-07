"""Admission serialisation: one advisory lock per hold ledger, contended or waited."""
import errno
import fcntl
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

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
          'from slstack.journal import advisory_lock, LockUnavailable\n'
          'stream = open(sys.argv[1], "r+")\n'
          'deadline = time.monotonic() + 10\n'
          'while True:\n'
          '    try:\n'
          '        with advisory_lock(stream):\n'
          '            print("HELD", flush=True)\n'
          '            time.sleep(float(sys.argv[2]))\n'
          '        break\n'
          '    except LockUnavailable:\n'
          '        # The polling probe in the same test can win the lock first; a holder that\n'
          '        # dies on that race leaves the test waiting for a contention it never sees.\n'
          '        if time.monotonic() >= deadline: raise\n'
          '        time.sleep(.05)')
SHARED_PROBE = ('import sys\n'
                'from slstack.journal import advisory_lock, LockUnavailable\n'
                'stream = open(sys.argv[1], "r+")\n'
                'try:\n'
                '    with advisory_lock(stream, shared=True):\n'
                '        print("SHARED")\n'
                'except LockUnavailable:\n'
                '    print("CONTENDED")')


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

    def test_shared_holders_coexist_and_refuse_an_exclusive_holder(self):
        """The reader's half of the rule chunk 5.2's read lease needs: many readers, no writer."""
        with self.ledger.open('r+') as stream, advisory_lock(stream, shared=True):
            self.assertEqual(run(SHARED_PROBE, str(self.ledger)).stdout.strip(), 'SHARED')
            with self.assertRaises(LockUnavailable):
                with self.ledger.open('r+') as writer, advisory_lock(writer):
                    pass
        with self.ledger.open('r+') as stream, advisory_lock(stream):
            self.assertEqual(run(SHARED_PROBE, str(self.ledger)).stdout.strip(), 'CONTENDED')

    def test_a_signal_during_a_blocking_wait_is_not_contention(self):
        """EINTR from a delivered signal must not read as "held by another process"."""
        import signal
        holder = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(self.ledger), '1.5'],
                                  cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        fired = []
        try:
            deadline = time.monotonic() + 10
            while run(PROBE, str(self.ledger)).stdout.strip() != 'CONTENDED':
                if time.monotonic() >= deadline:
                    self.fail('the holder never took the lock')
                time.sleep(.05)
            def handler(signum, frame):
                fired.append(signum)
            previous = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, .2)
            try:
                started = time.monotonic()
                with self.ledger.open('r+') as stream, advisory_lock(stream, blocking=True):
                    waited = time.monotonic() - started
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        finally:
            holder.wait(timeout=10)
        self.assertTrue(fired, 'the signal never arrived during the wait')
        self.assertGreaterEqual(waited, .2)
        self.assertLess(waited, 10)

    def test_only_a_refused_attempt_is_reported_as_contention(self):
        """lease.gpu_lease turns LockUnavailable into AdmissionRefused, so nothing else may."""
        with self.ledger.open('r+') as stream:
            with patch.object(fcntl, 'flock', side_effect=OSError(errno.EBADF, 'descriptor closed')):
                with self.assertRaises(OSError) as caught:
                    with advisory_lock(stream):
                        pass
            self.assertNotIsInstance(caught.exception, LockUnavailable)
            with patch.object(fcntl, 'flock', side_effect=OSError(errno.EWOULDBLOCK, 'busy')):
                with self.assertRaises(LockUnavailable):
                    with advisory_lock(stream):
                        pass

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
