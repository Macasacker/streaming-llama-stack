"""scripts/ds4_run_metrics.py names the routed shape each producer ran.

docs/PREFILL_DECODE_PLAN_20261110.md Step 1 measures two branches in one window, and the
comparison is meaningless unless the report says which shape each side resolved. These feed a
synthetic producer directory (the module reads a directory, rotated log first) and check both
the metric and the printed row.
"""
import os
import tempfile
import unittest

import importlib.util

spec = importlib.util.spec_from_file_location(
    'ds4_run_metrics', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ds4_run_metrics.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LOG = '''DS4PERF_ROUTED_PREFILL mode=tail from_layer=21 rows=16666 source=sweep
[dsv41-prefill-experts] stage=22 rows=6 active=30 new=6 resident=30 bytes=20971520 fetch_ms=4.0
DS4PERF_ROUTED_PREFILL mode=all from_layer=0 rows=45 source=sweep
[dsv41-prefill-experts] stage=1 rows=6 active=6 new=6 resident=6 bytes=2097152 fetch_ms=1.0
LLAMADECODE_DS4X_TOKEN position=1 decode_ms=110.0 ds_bytes=109547110 ds_ms=31.0 l1_hits=0 l2_hits=0 misses=240 l1_slots=449 l2_slots=0 ds_submit_ms=0.0 ds_fence_ms=0.0 ds_verify_ms=0.0 ds_cuda_ms=31.0 ds_requests=720 ds_batches=40 assign_ms=34.5 copy_sync_ms=0.0 l2_stored=0 host_hits=0 host_stored=0 host_slots=0 host_bytes=0
LLAMADECODE_DS4X_TOKEN position=2 decode_ms=58.7 ds_bytes=109547110 ds_ms=31.0 l1_hits=5 l2_hits=1 misses=11 l1_slots=449 l2_slots=1500 ds_submit_ms=0.0 ds_fence_ms=0.0 ds_verify_ms=0.0 ds_cuda_ms=31.0 ds_requests=12 ds_batches=3 assign_ms=34.5 copy_sync_ms=0.0 l2_stored=0 host_hits=0 host_stored=0 host_slots=0 host_bytes=0
LLAMADECODE_DS4X_TOKEN position=3 decode_ms=60.2 ds_bytes=109547110 ds_ms=30.0 l1_hits=6 l2_hits=0 misses=10 l1_slots=449 l2_slots=1500 ds_submit_ms=0.0 ds_fence_ms=0.0 ds_verify_ms=0.0 ds_cuda_ms=30.0 ds_requests=12 ds_batches=3 assign_ms=33.0 copy_sync_ms=0.0 l2_stored=0 host_hits=0 host_stored=0 host_slots=0 host_bytes=0
'''


class RunMetrics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, 'stderr.log'), 'w') as handle:
            handle.write(LOG)

    def tearDown(self):
        os.remove(os.path.join(self.dir, 'stderr.log'))
        os.rmdir(self.dir)

    def test_the_routed_shape_and_its_source_are_recorded(self):
        out = module.metrics(self.dir)
        choice = out['routed_choice']
        self.assertEqual(choice['sweeps'], 2)
        self.assertEqual(choice['modes'], ['all', 'tail'])
        self.assertEqual(choice['sources'], ['sweep'])
        self.assertEqual(choice['rows'], [16666, 45])

    def test_a_log_from_before_the_line_exists_reads_as_no_shape_rather_than_a_guess(self):
        with open(os.path.join(self.dir, 'stderr.log'), 'w') as handle:
            handle.write('LLAMADECODE_DS4X_TOKEN position=2 decode_ms=58.7 ds_bytes=1 ds_ms=1 '
                         'assign_ms=1 misses=0 l1_hits=0 l2_hits=0 l1_slots=0 l2_slots=0 '
                         'host_hits=0 host_stored=0 host_slots=0 host_bytes=0\n')
        out = module.metrics(self.dir)
        self.assertEqual(out['routed_choice'], {'sweeps': 0, 'modes': [], 'sources': [], 'rows': []})
        row = module.table([dict(out, label='run-a')])
        self.assertIn('| - |', row)          # the mode column reads as absent, not as off
        self.assertIn('routed mode', row)

    def test_the_mode_column_prints_what_both_branches_ran(self):
        row = module.table([dict(module.metrics(self.dir), label='run-a')])
        self.assertIn('all/tail', row)


if __name__ == '__main__':
    unittest.main()
