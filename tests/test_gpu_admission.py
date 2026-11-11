"""RTX admission (slstack.lease): a refusal names the condition it tripped, and a single noisy
utilization sample does not refuse the window. No GPU is read — nvidia-smi is scripted."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import config as config_module
from slstack.lease import AdmissionRefused, gpu_lease, gpu_not_idle


class Nvidia:
    """Scripted nvidia-smi answers: one tuple per --query-gpu reading, in order."""
    def __init__(self, samples, holders='28914, 140\n65643, 29740'):
        self.samples = list(samples);self.holders = holders;self.readings = 0

    def __call__(self, command, **kwargs):
        if any(arg.startswith('--query-compute-apps') for arg in command):
            return subprocess.CompletedProcess(command, 0, self.holders + '\n', '')
        sample = self.samples[min(self.readings, len(self.samples) - 1)]
        self.readings += 1
        return subprocess.CompletedProcess(command, 0, '%d, %d, %d\n' % sample, '')


def cfg(root, **over):
    base = dict(task='DS4STACK', runtime_root=root, hold_file=str(Path(root) / 'holds'), gpu_uuid='GPU-abc',
                gpu_idle_max_mib=2560, gpu_min_free_mib=0, gpu_idle_max_util=5, gpu_wait_seconds=0,
                host_budget_gib=50, min_available_gib=.5, lease_minutes=19)
    base.update(over);return base


class NotIdleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def reason(self, samples, **over):
        with patch('slstack.lease.subprocess.run', Nvidia(samples)):
            return gpu_not_idle(cfg(self.dir, **over))

    def test_an_idle_card_is_not_refused(self):
        self.assertIsNone(self.reason([(1415, 30780, 0)]))

    def test_a_memory_holder_names_the_memory_threshold(self):
        reason = self.reason([(4000, 20000, 0)])
        self.assertIn('4000 MiB used', reason)
        self.assertIn('gpu_idle_max_mib 2560', reason)
        self.assertNotIn('gpu_idle_max_util', reason)

    def test_a_missing_free_floor_names_the_free_threshold(self):
        reason = self.reason([(1000, 5000, 0)], gpu_min_free_mib=20000)
        self.assertIn('5000 MiB free', reason)
        self.assertIn('gpu_min_free_mib 20000', reason)

    def test_utilization_is_refused_only_when_a_second_sample_agrees(self):
        # The refusal this box actually saw: 1415 MiB used (under the limit), 7% busy.
        self.assertIn('9% utilization', self.reason([(1415, 30780, 7), (1415, 30780, 9)]))
        self.assertIn('first sample 7%', self.reason([(1415, 30780, 7), (1415, 30780, 9)]))
        self.assertIn('gpu_idle_max_util 5', self.reason([(1415, 30780, 7), (1415, 30780, 9)]))
        self.assertIsNone(self.reason([(1415, 30780, 7), (1415, 30780, 1)]))

    def test_the_ceiling_is_configurable_and_only_one_sample_is_taken_under_it(self):
        with patch('slstack.lease.subprocess.run', Nvidia([(1415, 30780, 7)])) as nvidia:
            self.assertIsNone(gpu_not_idle(cfg(self.dir, gpu_idle_max_util=20)))
        self.assertEqual(nvidia.readings, 1)

    def test_an_unreadable_reading_is_an_error_not_a_refusal(self):
        with patch('slstack.lease.subprocess.run', lambda command, **kwargs:
                   subprocess.CompletedProcess(command, 1, '', 'bad uuid')):
            with self.assertRaisesRegex(RuntimeError, 'Cannot certify GPU ownership'):
                gpu_not_idle(cfg(self.dir))


class LeaseRefusalTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        Path(self.dir, 'holds').touch()

    def test_the_refusal_keeps_its_contract_and_names_the_holder(self):
        with patch('slstack.lease.subprocess.run', Nvidia([(1415, 30780, 7), (1415, 30780, 8)])), \
             patch('slstack.lease.psutil.virtual_memory') as memory:
            memory.return_value.available = 90 * 2**30
            with self.assertRaises(AdmissionRefused) as caught:
                with gpu_lease(cfg(self.dir)):
                    self.fail('a busy card must not be admitted')
        text = str(caught.exception)
        self.assertIn('no other process was ejected', text)   # the stack never kills another owner
        self.assertIn('utilization', text)
        self.assertIn('28914 (140 MiB)', text)                # and it says who holds the card
        self.assertIn('65643 (29740 MiB)', text)

    def test_a_holder_query_that_fails_does_not_replace_the_refusal(self):
        class NoHolders(Nvidia):
            def __call__(self, command, **kwargs):
                if any(arg.startswith('--query-compute-apps') for arg in command):
                    return subprocess.CompletedProcess(command, 1, '', 'nvidia-smi failed')
                return super().__call__(command, **kwargs)
        with patch('slstack.lease.subprocess.run', NoHolders([(1415, 30780, 9), (1415, 30780, 9)])), \
             patch('slstack.lease.psutil.virtual_memory') as memory:
            memory.return_value.available = 90 * 2**30
            with self.assertRaises(AdmissionRefused) as caught:
                with gpu_lease(cfg(self.dir)):
                    self.fail('a busy card must not be admitted')
        self.assertIn('utilization', str(caught.exception))
        self.assertIn('unknown', str(caught.exception))


class ConfigTests(unittest.TestCase):
    def test_the_utilization_ceiling_is_a_bounded_integer(self):
        for bad in (-1, 101, 5.5, 'five'):
            with self.assertRaises(ValueError):
                config_module.check_gpu_utilization_ceiling(bad)
        for good in (0, 5, 100):
            self.assertIsNone(config_module.check_gpu_utilization_ceiling(good))


if __name__ == '__main__':unittest.main()
