"""Turn progress lines (slstack/turn_progress.py) and their wiring into the single-engine turn."""
import contextlib
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from slstack import combo_single as s
from slstack.turn_progress import ExpertCache, TurnProgress, phase_of
from test_combo_single import Harness, FakeProducer, ids

DS4X = (b'LLAMADECODE_DS4X_TOKEN position=93200 decode_ms=312.169 ds_bytes=806215680 ds_ms=269.251 '
        b'l1_hits=159 l2_hits=0 misses=81 l1_slots=449 l2_slots=1500 ds_submit_ms=0.000 ds_fence_ms=0.000 '
        b'ds_verify_ms=0.000 ds_cuda_ms=269.251 ds_requests=243 ds_batches=40 assign_ms=269.459 '
        b'copy_sync_ms=0.000 l2_stored=0 host_hits=0 host_stored=0 host_slots=0 host_bytes=0\n')


class Clock:
    """A clock the test advances; the reporter never sleeps, so the lines come out in order."""

    def __init__(self, started=0.0):
        self.t = started

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def reporter(clock, *, enabled=True, every=1.0):
    lines = []
    return TurnProgress({'turn_progress': enabled}, say=lines.append, clock=clock, every=every), lines


class LineTests(unittest.TestCase):
    def test_the_prompt_is_named_as_it_is_read(self):
        clock = Clock();tp, lines = reporter(clock)
        clock.advance(1.0);tp.reading(None, 399161)
        self.assertEqual(lines, ['[ds4single] reading the prompt: 399,161 tokens, 1 s so far'])
        # The producer gave a sweep position: the line says how much of the prompt is read.
        clock.advance(1.0);tp.reading(199580, 399161)
        self.assertEqual(lines[1], '[ds4single] reading the prompt: 199,580 of 399,161 tokens, 2 s so far')
        # One line per second: a second snapshot inside the same window is not printed.
        clock.advance(.4);tp.reading(240000, 399161)
        self.assertEqual(len(lines), 2)

    def test_the_decode_lines_are_the_phase_and_the_cap(self):
        clock = Clock();tp, lines = reporter(clock)
        tp.token(1, lambda: 'thinking', 32768)      # the first token has no span to divide by
        self.assertEqual(lines, [])
        for n, phase in ((47, 'thinking'), (149, 'writing a tool call: edit'), (274, 'answering')):
            clock.advance(1.0);tp.token(n, lambda p=phase: p, 32768)
        self.assertEqual([line.split(': ')[0] for line in lines],
                         ['[ds4single] thinking', '[ds4single] writing a tool call', '[ds4single] answering'])
        self.assertEqual(lines[0], '[ds4single] thinking: 47 of max 32768 tokens, 47.0 tok/s, 1 s')
        self.assertEqual(lines[2], '[ds4single] answering: 274 of max 32768 tokens, 91.3 tok/s, 3 s')

    def test_one_token_says_nothing_about_speed(self):
        clock = Clock();tp, lines = reporter(clock)
        clock.advance(.2);tp.token(1, lambda: 'thinking', 32768)
        self.assertEqual(lines, [])

    def test_the_done_line_carries_the_turn_and_the_cache(self):
        clock = Clock();tp, lines = reporter(clock)
        cache = ExpertCache();cache.vram, cache.pcie = 962, 26
        clock.advance(10.0)
        tp.done(1071, 'stop', False, cache, decode_ms=9304.0)
        self.assertEqual(lines, ['[ds4single] done: 1071 tokens in 10 s (115.1 tok/s) (stop, cancel=False), '
                                 'expert cache 97.4% hit (+2.6% of the routed experts over PCIe)'])

    def test_a_turn_with_no_cache_counters_does_not_claim_one(self):
        clock = Clock();tp, lines = reporter(clock)
        clock.advance(2.0)
        tp.done(235, 'tool_calls', True, ExpertCache(), decode_ms=0.0)
        self.assertEqual(lines, ['[ds4single] done: 235 tokens in 2 s (no speed) (tool_calls, cancel=True)'])

    def test_an_error_ends_the_turn_with_a_line_of_its_own(self):
        clock = Clock();tp, lines = reporter(clock)
        clock.advance(4.0)
        tp.token(10, lambda: 'thinking', 32768)
        tp.failed('RuntimeError: producer frontier differs')
        self.assertEqual(lines[-1], '[ds4single] failed: RuntimeError: producer frontier differs after 4 s')

    def test_turn_progress_false_prints_nothing(self):
        clock = Clock()
        tp, lines = reporter(clock, enabled=False, every=0.0)
        clock.advance(1.0)
        tp.reading(10, 100);tp.token(5, lambda: 'thinking', 32768)
        tp.done(5, 'stop', False, ExpertCache(), decode_ms=1000.0)
        self.assertEqual(lines, [])

    def test_turn_progress_must_be_a_switch(self):
        with self.assertRaisesRegex(ValueError, 'turn_progress must be true or false'):
            TurnProgress({'turn_progress': 'sometimes'})

    def test_the_prompt_line_stops_once_the_first_token_arrived(self):
        clock = Clock();tp, lines = reporter(clock, every=0.0)
        tp.token(1, lambda: 'thinking', 32768)
        clock.advance(.6);tp.token(2, lambda: 'thinking', 32768)
        tp.reading(399161, 399161)
        self.assertEqual(lines, ['[ds4single] thinking: 2 of max 32768 tokens, 3.3 tok/s, 1 s'])


class PhaseTests(unittest.TestCase):
    def test_think_then_answer_then_a_named_call(self):
        self.assertEqual(phase_of('<think>plan', True), 'thinking')
        self.assertEqual(phase_of('<think>plan</think>done', True), 'answering')
        self.assertEqual(phase_of('plain text', False), 'answering')
        self.assertEqual(phase_of('<think>x</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="edit">', True),
                         'writing a tool call: edit')
        # The invoke tag is written token by token; the phase is the tool call before its name lands.
        self.assertEqual(phase_of('<think>x</think><｜DSML｜tool_calls>', True), 'writing a tool call')
        # A model that opens a tool block without closing the thinking is a call, not thinking.
        self.assertEqual(phase_of('<think>x<｜DSML｜invoke name="bash"', True), 'writing a tool call: bash')


class ExpertCacheTests(unittest.TestCase):
    def test_the_resident_and_l2_slots_are_the_hit_and_the_rest_crosses_the_bus(self):
        cache = ExpertCache();cache.feed(DS4X)
        self.assertEqual(cache.counters(), dict(routed=240, vram_hits=159, over_pcie=81, hit_percent=66.2))
        self.assertEqual(cache.summary(), ', expert cache 66.2% hit (+33.8% of the routed experts over PCIe)')

    def test_a_partial_line_is_not_counted(self):
        cache = ExpertCache()
        cache.feed(DS4X[:-30])          # the tail of the line has not arrived yet
        self.assertEqual(cache.routed, 0)
        cache.feed(DS4X[:-30] + DS4X[-30:])
        self.assertEqual(cache.routed, 240)

    def test_other_producer_lines_are_ignored(self):
        cache = ExpertCache()
        cache.feed(b'LLAMADECODE_STEP index=0 token=671 position=93200 step_ms=2588.732 engine=ds4\n')
        self.assertEqual(cache.summary(), '')


class CacheProducer(FakeProducer):
    """A producer that writes the ds4 decode's per-token cache line beside its STEP lines."""

    def decode(self, max_tokens, stop_tokens=()):
        reply = super().decode(max_tokens, stop_tokens)
        with self.process.state_lock:self.process.diagnostic.extend(DS4X * 2)
        return reply


class EngineTurnTests(unittest.TestCase):
    def test_one_turn_prints_reading_and_done_and_records_the_cache(self):
        harness = Harness([])
        harness.scripts.append(ids(harness.tok, 'plan</think>All done') + [1])
        harness.engine.producer_factory = lambda tokens: CacheProducer(tokens, harness.scripts, harness.dir)
        clock = Clock();turn, lines = reporter(clock)
        final, metrics = {}, {}
        clock.advance(1.0);turn.reading(None, 12)
        clock.advance(1.0)
        harness.engine.run(dict(messages=[dict(role='user', content='q')], max_tokens=50),
                           lambda kind, *values: final.update(values[0]) if kind == 'final' else None,
                           metrics, progress=turn)
        self.assertEqual(lines[0], '[ds4single] reading the prompt: 12 tokens, 1 s so far')
        self.assertRegex(lines[-1], r'^\[ds4single\] done: \d+ tokens in \d+ s \(\d+\.\d tok/s\) \(stop, cancel=False\), ')
        self.assertIn('expert cache 66.2% hit (+33.8% of the routed experts over PCIe)', lines[-1])
        self.assertEqual(final['parsed']['content'], 'All done')
        self.assertEqual(metrics['expert_cache'], dict(routed=480, vram_hits=318, over_pcie=162, hit_percent=66.2))
        self.assertEqual(metrics['finish_reason'], 'stop')

    def test_a_turn_without_the_reporter_prints_nothing(self):
        harness = Harness([])
        harness.scripts.append(ids(harness.tok, '<think>plan</think>Hi') + [1])
        metrics = {}
        harness.engine.run(dict(messages=[dict(role='user', content='q')], max_tokens=50),
                           lambda kind, *values: None, metrics)
        self.assertIn('finish_reason', metrics)


class HandlerTests(unittest.TestCase):
    """The lines a served request writes to the engine's stdout, through the real handler."""

    def serve(self, **cfg_extra):
        os.environ['DS4_SERVER_AUTH_TOKEN'] = 't' * 40
        harness = Harness([])
        harness.scripts.append(ids(harness.tok, 'plan</think>Hey you') + [1])
        harness.engine.producer_factory = lambda tokens: CacheProducer(tokens, harness.scripts, harness.dir)
        cfg = dict(context=4096, lease_minutes=19, runtime_root=harness.dir, model_name='DeepSeek-V4.1-Flash', **cfg_extra)
        proxy = s.SingleProxy(cfg, 'run', None, engine=harness.engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), s.handler(proxy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown);self.addCleanup(server.server_close)
        self.url = 'http://127.0.0.1:%d' % server.server_address[1]

    def call(self, body):
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + 't' * 40, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:return response.read().decode()

    def test_a_served_turn_ends_with_its_done_line(self):
        self.serve()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            text = self.call(dict(messages=[dict(role='user', content='q')], stream=True))
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))
        self.assertIn('[ds4single] done: ', out.getvalue())
        self.assertIn('(stop, cancel=False)', out.getvalue())
        self.assertIn('expert cache 66.2% hit (+33.8% of the routed experts over PCIe)', out.getvalue())
        self.assertNotIn('stream/final', out.getvalue())

    def test_a_refused_request_says_the_turn_failed(self):
        self.serve()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(urllib.error.HTTPError):
                self.call(dict(model='m'))
        self.assertIn('[ds4single] failed: ', out.getvalue())
        self.assertNotIn('[ds4single] done:', out.getvalue())

    def test_turn_progress_false_writes_nothing(self):
        self.serve(turn_progress=False)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.call(dict(messages=[dict(role='user', content='q')]))
        self.assertEqual(out.getvalue(), '')


class ConfigurationTests(unittest.TestCase):
    def test_the_switch_defaults_to_on_and_reaches_the_engine_config(self):
        from slstack.config import load
        from slstack.single_runtime import serving_config
        self.assertIs(load(overrides={'gpu_uuid': 'GPU-test'})['turn_progress'], True)
        row = serving_config({'host_budget_gib': 8, 'context': 262144, 'turn_progress': False},
                             runtime_root='r', studio_root='s', llama_producer='p', library_directories=[], revision='x')
        self.assertIs(row['turn_progress'], False)

    def test_a_non_switch_is_refused_at_load(self):
        from slstack.config import load
        with self.assertRaisesRegex(ValueError, 'turn_progress must be true or false'):
            load(overrides={'gpu_uuid': 'GPU-test', 'turn_progress': 1})

    def test_the_launch_generator_carries_the_flag(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import dsv41_single
        self.assertNotIn('turn_progress', dsv41_single.overlay('262k'))
        self.assertIs(dsv41_single.overlay('262k', turn_progress=False)['turn_progress'], False)
        self.assertIs(dsv41_single.overlay('262k', dict(turn_progress=True), turn_progress=False)['turn_progress'], False)

    def test_the_shell_launcher_writes_the_switch_into_its_start_config(self):
        import json
        import subprocess
        written = ROOT / 'runtime' / 'dsv41-single-262k-ultra-cone.local.json'
        before = written.read_text(encoding='utf-8') if written.exists() else None
        try:
            for flag, want in (('--no-progress', False), ('--progress', True)):
                result = subprocess.run(['sh', str(ROOT / 'scripts' / 'start-dsv41-single.sh'),
                                         '--preset', '262k', '--dry-run', flag],
                                        capture_output=True, text=True, timeout=120, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIs(json.loads(written.read_text(encoding='utf-8'))['turn_progress'], want)
        finally:
            if before is not None:written.write_text(before, encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
