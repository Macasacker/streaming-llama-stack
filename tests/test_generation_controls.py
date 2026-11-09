"""Live control publication and OpenAI override regression tests; no model needed."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from slstack.generation import Store, defaults, validate
from slstack.ds4_render import parse_request


class LiveControls(unittest.TestCase):
    def test_phase_defaults_and_studio_named_bias_contract(self):
        p = defaults()['phases']
        self.assertGreater(p['reasoning']['bias'], p['answer']['bias'])
        self.assertGreater(p['reasoning']['temperature'], p['answer']['temperature'])
        self.assertLess(p['tool_call']['temperature'], p['answer']['temperature'])
        with tempfile.TemporaryDirectory() as root:
            # A calibration bound to the model is what makes a nonzero bias applicable at all;
            # see test_uncalibrated_box_publishes_an_unbiased_profile_and_refuses_a_bias.
            store=Store(dict(runtime_root=root, cache_affinity_spread_file='spreads.txt'))
            self.assertEqual(store.bias('max_speed')['value'],1.36125)
            self.assertFalse(store.get()['per_phase'])
            store.bias(dict(reasoning='fast',answer='balanced',tool_call='max_quality'))
            self.assertEqual(store.get()['phases']['tool_call']['bias'],0)
            self.assertTrue(store.get()['per_phase'])

    def test_partial_phase_updates_preserve_other_samplers_and_revision(self):
        with tempfile.TemporaryDirectory() as root:
            a=Store(dict(runtime_root=root));b=Store(dict(runtime_root=root))
            original=a.get()
            a.update(dict(phases=dict(reasoning=dict(temperature=1.2))))
            b.update(dict(phases=dict(tool_call=dict(top_k=8))))
            row=a.get()
            self.assertEqual(row['revision'],original['revision']+2)
            self.assertEqual(row['phases']['reasoning']['temperature'],1.2)
            self.assertEqual(row['phases']['tool_call']['top_k'],8)
            self.assertEqual(row['phases']['answer'],original['phases']['answer'])

    def test_uncalibrated_box_publishes_an_unbiased_profile_and_refuses_a_bias(self):
        # ds4x_set_affinity refuses any mean > 0 without a spread file bound to this exact model,
        # and it refuses at the first decoded token. Nothing here may offer one.
        with tempfile.TemporaryDirectory() as root:
            store = Store(dict(runtime_root=root, cache_affinity_spread_file=''))
            row = store.get()
            self.assertEqual(row['value'], 0)
            self.assertEqual({phase: knobs['bias'] for phase, knobs in row['phases'].items()},
                             dict(reasoning=0, answer=0, tool_call=0, code=0))
            self.assertEqual(row['phases']['answer']['temperature'], .6)   # samplers unchanged
            with self.assertRaises(ValueError) as caught:
                store.bias('fast')
            self.assertIn('DS4_CACHE_AFFINITY_CALIBRATE', str(caught.exception))
            with self.assertRaises(ValueError):
                store.bias(dict(reasoning='balanced', answer='balanced', tool_call='max_quality', code='max_quality'))
            self.assertEqual(store.bias(0)['value'], 0)

    def test_a_stale_nonzero_profile_is_reported_and_zeroed_without_losing_samplers(self):
        with tempfile.TemporaryDirectory() as root:
            bound = Store(dict(runtime_root=root, cache_affinity_spread_file='spreads.txt'))
            bound.bias(dict(reasoning='max_speed', answer='balanced', tool_call='max_quality', code='max_quality'))
            bound.update(dict(phases=dict(reasoning=dict(temperature=1.2))))
            before = bound.get()
            unbound = Store(dict(runtime_root=root, cache_affinity_spread_file=''))
            self.assertEqual(unbound.unusable_bias(), dict(reasoning=1.36125, answer=.35))
            after = unbound.zero_bias()
            self.assertEqual({phase: knobs['bias'] for phase, knobs in after['phases'].items()},
                             dict(reasoning=0, answer=0, tool_call=0, code=0))
            self.assertEqual(after['revision'], before['revision'] + 1)
            self.assertEqual(after['phases']['reasoning']['temperature'], 1.2)
            self.assertIsNone(unbound.unusable_bias())
            self.assertIsNone(bound.unusable_bias())   # a calibrated box keeps its profile

    def test_bad_settings_never_replace_the_last_confirmed_file(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(dict(runtime_root=root));before=store.path.read_bytes()
            for patch in [dict(value=float('nan')),dict(sampler=dict(top_k=1.5)),
                          dict(phases=dict(tool_call=dict(temperature=-1))),dict(per_phase=1),dict(unknown=1)]:
                with self.assertRaises(ValueError):store.update(patch)
                self.assertEqual(store.path.read_bytes(),before)

    def test_readers_observe_complete_revisions_during_slider_updates(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(dict(runtime_root=root));errors=[];stop=threading.Event()
            def read():
                while not stop.is_set():
                    try:validate(store.get())
                    except Exception as exc:errors.append(exc)
            thread=threading.Thread(target=read);thread.start()
            try:
                for i in range(20):store.update(dict(value=i/20,per_phase=False))
            finally:stop.set();thread.join()
            self.assertEqual(errors,[])
            self.assertEqual(store.get()['revision'],21)

    def test_explicit_sampler_values_survive_request_parsing(self):
        row=parse_request(dict(messages=[dict(role='user',content='q')],temperature=0,top_p=.8,top_k=7,min_p=.1,seed=42))
        self.assertEqual(row['sampling'],dict(temperature=0,top_p=.8,top_k=7,min_p=.1,seed=42))
