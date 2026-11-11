"""The single engine waits out a busy card (gpu_wait_seconds) rather than answering 503: the
guardian's admission loop already waits, but this engine takes its hold per request."""
import contextlib
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / 'tests'))
from slstack import combo_single as s
from slstack.lease import AdmissionRefused
from test_combo_single import FakeProducer, char_tokenizer


class RefusingLease:
    """A lease_factory that refuses `refusals` times before it hands out the hold."""
    def __init__(self, refusals):
        self.refusals = refusals;self.calls = 0
    @contextlib.contextmanager
    def __call__(self):
        self.calls += 1
        if self.calls <= self.refusals:
            raise AdmissionRefused('RTX is not idle enough for exclusive admission; no other process was ejected')
        yield 'lease-1'


def engine(lease, wait, directory):
    cfg = dict(context=4096, lease_minutes=19, model='unused', runtime_root=directory,
               gpu_wait_seconds=wait, cache_affinity_spread_file='spreads.txt')
    return s.SingleEngine(cfg, tokenizer=char_tokenizer(),
                          producer_factory=lambda tokens: FakeProducer(tokens, [], directory),
                          lease_factory=lease)


class EngineAdmissionWaitTests(unittest.TestCase):
    def test_a_busy_card_is_waited_out_and_the_request_is_served(self):
        directory = tempfile.mkdtemp();lease = RefusingLease(2)
        started = time.monotonic()
        e = engine(lease, 1.0, directory)
        e.launch([1, 2, 3])
        self.assertEqual(lease.calls, 3)
        self.assertEqual(e.lease_id, 'lease-1')
        self.assertIsNotNone(e.producer)
        self.assertGreaterEqual(time.monotonic() - started, .5)   # it waited; it did not refuse
        e.close()

    def test_without_a_wait_budget_the_first_refusal_is_the_answer(self):
        directory = tempfile.mkdtemp();lease = RefusingLease(99)
        e = engine(lease, 0, directory)
        with self.assertRaises(AdmissionRefused):
            e.launch([1, 2, 3])
        self.assertEqual(lease.calls, 1)
        self.assertIsNone(e.lease_id);self.assertIsNone(e.producer)

    def test_a_wait_ends_at_its_deadline(self):
        directory = tempfile.mkdtemp();lease = RefusingLease(99)
        e = engine(lease, .4, directory)
        with self.assertRaises(AdmissionRefused):
            e.launch([1, 2, 3])
        self.assertEqual(lease.calls, 2)   # one retry inside 0.4 s, then the refusal stands
        self.assertIsNone(e.leases)

    def test_a_shutdown_stops_the_wait(self):
        directory = tempfile.mkdtemp();lease = RefusingLease(99)
        e = engine(lease, 60, directory);e.stopping.set()
        started = time.monotonic()
        with self.assertRaises(AdmissionRefused):
            e.launch([1, 2, 3])
        self.assertLess(time.monotonic() - started, 1)


if __name__ == '__main__':unittest.main()
