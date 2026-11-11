"""scripts/analyze_dsv41_decode.py reads the Step 0 instrumentation.

docs/PREFILL_DECODE_PLAN_20261110.md Step 0 accepts "one line per step and per layer with the
split", and the split is only worth emitting if the run can be read with the script the plan
names. These feed a synthetic producer log with LLAMADECODE_ASSIGN_SPLIT lines and check what
the report answers about the expert hook.
"""
import contextlib
import importlib.util
from io import StringIO
import os
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'decode_analysis', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'analyze_dsv41_decode.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(log):
    handle = tempfile.NamedTemporaryFile('w', suffix='.log', delete=False)
    handle.write(log)
    handle.close()
    out = StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = module.report(handle.name, 2)
    finally:
        os.unlink(handle.name)
    return out.getvalue(), code


LOG = '''LLAMADECODE_STEP index=1 token=1 position=40 step_ms=58.7 engine=ds4
LLAMADECODE_STEP index=2 token=2 position=41 step_ms=60.1 engine=ds4
LLAMADECODE_DS4X_TOKEN position=41 decode_ms=59.5 ds_bytes=109547110 ds_ms=31.0 l1_hits=5 l2_hits=1 misses=11 l1_slots=449 l2_slots=1636 ds_submit_ms=0.0 ds_fence_ms=0.0 ds_verify_ms=0.0 ds_cuda_ms=31.0 ds_requests=12 ds_batches=3 assign_ms=34.5 copy_sync_ms=1.0 l2_stored=1 host_hits=0 host_stored=0 host_slots=0 host_bytes=0
LLAMADECODE_STEP_BYTES position=41 step_ms=59.500 experts=240 slot_bytes=456446 footprint=109547040 served=109547040 from_file=109547110 from_ram=0 from_l2=0 from_l1=0 disk_percent=100.0 unattributed=0 over_served=0 copied=0 stored=0 l1_hits=5 l2_hits=1 ram_hits=0 misses=11
LLAMADECODE_ASSIGN_SPLIT position=41 assign_ms=34.500 fetch_ms=31.000 read_ms=20.000 stage_ms=8.000 upload_issue_ms=2.000 upload_device_ms=18.000 fetch_remainder_ms=1.000 fetch_bytes=109547110 batches=3 read_GB_s=5.477 upload_GB_s=6.086 l2_copy_ms=1.000 ram_ms=2.000 ram_exclusive_ms=1.500 ram_overlap_ms=0.500 ram_overlap_in_fetch_ms=0.500 bookkeep_ms=1.000 over_attributed_ms=0.000 hook_ms=33.900 misses=11 upload_windows_unread=0
DS4PERF_ROUTED_PREFILL mode=tail from_layer=21 rows=16666 source=env
DS4X_STAGE position=41 total_ms=58.400 hc_pre=2.100 attention=9.400 hc_mix=1.800 router_shared=3.200 expert_hook=33.100 routed_experts=11.200 after_moe=0.600 between_layers=0.400 head=0.200 host_ids_read=0.300 host_cache_hook=32.600 host_slot_write=0.100 host_fp4_launch=0.000 pool_calls=0 pool_ms=0.000 attn_project=1.100 attn_rope_store=0.400 attn_select=2.200 attn_core=5.700
[dsv41-prefill-experts] stage=22 rows=6 active=30 new=6 resident=30 bytes=20 fetch_ms=2.0 read_ms=1.500 stage_ms=0.250 upload_issue_ms=0.200 upload_device_ms=1.100 begin_epoch_ms=1.0 end_epoch_ms=3.0
'''


class DecodeStepAnalysis(unittest.TestCase):
    def test_the_report_answers_where_the_expert_hook_spent_its_time(self):
        text, code = run(LOG)
        self.assertEqual(code, 0)
        self.assertIn('expert-hook split from LLAMADECODE_ASSIGN_SPLIT: 1 steps with a hook', text)
        self.assertIn('assign_ms', text)
        self.assertIn('median   34.50 ms', text)
        self.assertIn('read_ms', text)
        self.assertIn('median   20.00 ms', text)
        self.assertIn('upload_device_ms', text)
        self.assertIn('median   18.00 ms', text)
        # The two rates the plan's Step 3a target is judged against.
        self.assertIn('implied read rate 5.48 GB/s', text)
        self.assertIn('implied upload rate 6.09 GB/s', text)
        self.assertNotIn('NOTE', text)

    def test_a_log_without_the_split_says_what_to_run(self):
        text, _ = run('LLAMADECODE_STEP index=1 token=1 position=5 step_ms=100.0 engine=ds4\n')
        self.assertIn('expert-hook split: no LLAMADECODE_ASSIGN_SPLIT line', text)
        self.assertIn('run with LLAMADECODE_BYTE_LEDGER=1', text)

    def test_unread_upload_windows_and_overshoot_are_reported_as_notes(self):
        log = LOG.replace('upload_windows_unread=0', 'upload_windows_unread=4') \
                 .replace('over_attributed_ms=0.000', 'over_attributed_ms=3.000')
        text, _ = run(log)
        self.assertIn('NOTE 4 upload windows had not completed', text)
        self.assertIn('NOTE 1 steps whose counters overshoot the hook', text)

    def test_prefill_split_and_routed_choice_are_read(self):
        text, _ = run(LOG)
        self.assertIn('routed prefill fetch: read', text)
        self.assertIn('routed prefill upload:', text)
        self.assertIn('prefill expert reuse: 1 layers, 30 active, 6 newly read (80.0 % reused)', text)
        self.assertIn('routed prefill per sweep: 1 sweeps, last mode=tail from_layer=21 rows=16666 source=env', text)

    def test_the_compute_floor_is_split_on_the_device_timeline(self):
        # The plan's other half of Step 0: whether the 24.7 ms floor is attention or FFN.
        text, _ = run(LOG)
        self.assertIn('device stage attribution from DS4X_STAGE: 1 tokens', text)
        self.assertIn('attention', text)
        self.assertIn('median    9.40 ms', text)
        self.assertIn('median   11.20 ms', text)   # routed_experts, the FFN share

    def test_a_log_without_the_stage_profile_names_the_switch(self):
        log = LOG.replace('DS4X_STAGE position=41', 'SOMETHING_ELSE position=41')
        text, _ = run(log)
        self.assertIn('device stage attribution: no DS4X_STAGE line', text)
        self.assertIn('DS4X_STAGE_PROFILE=1', text)


if __name__ == '__main__':
    unittest.main()
